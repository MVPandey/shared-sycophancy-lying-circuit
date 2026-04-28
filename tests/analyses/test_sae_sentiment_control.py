"""Tests for the SAE sentiment-control analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from shared_circuits.analyses import sae_sentiment_control
from shared_circuits.analyses.sae_sentiment_control import (
    NEGATIVE_TEMPLATES,
    POSITIVE_TEMPLATES,
    SaeSentimentControlConfig,
    add_cli_args,
    build_sentiment_prompts,
    from_args,
    run,
)
from shared_circuits.data.sae_features import SAE_KIND_GOODFIRE_TOPK, SaeWeights
from shared_circuits.experiment import ExperimentContext
from shared_circuits.models import ModelInfo


@pytest.fixture
def fake_ctx(mock_model):
    info = ModelInfo(
        name='meta-llama/Llama-3.1-8B-Instruct',
        n_layers=32,
        n_heads=32,
        d_model=4096,
        d_head=128,
        total_heads=1024,
    )
    return ExperimentContext(
        model=mock_model,
        info=info,
        model_name='meta-llama/Llama-3.1-8B-Instruct',
        agree_tokens=(1, 2),
        disagree_tokens=(3, 4),
    )


def _fake_sae() -> SaeWeights:
    return SaeWeights(
        kind=SAE_KIND_GOODFIRE_TOPK,
        layer=19,
        repo='Goodfire/test',
        w_enc=torch.zeros(32, 64),
        b_enc=torch.zeros(64),
        b_dec=torch.zeros(32),
        top_k=8,
    )


def _fake_acts(n_prompts: int = 20, d_sae: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n_prompts, d_sae).astype(np.float32)


def _reference_payload(layer: int = 19, d_sae: int = 64) -> dict:
    return {
        'per_layer': [
            {
                'layer': layer,
                'd_sae': d_sae,
                'overlap': 30,
                'syc_top_features': list(range(0, 50)),
                'lie_top_features': list(range(20, 70)),
                'shared_features': list(range(20, 50)),
            }
        ]
    }


class TestSaeSentimentControlConfig:
    def test_defaults(self):
        cfg = SaeSentimentControlConfig()
        assert cfg.layer == 19
        assert cfg.top_k == 100

    def test_rejects_negative_layer(self):
        with pytest.raises(ValidationError):
            SaeSentimentControlConfig(layer=-1)

    def test_is_frozen(self):
        cfg = SaeSentimentControlConfig()
        with pytest.raises(ValidationError):
            cfg.top_k = 5


class TestBuildSentimentPrompts:
    def test_returns_n_each(self):
        pos, neg = build_sentiment_prompts('gemma-2-2b-it', n=5, seed=42)
        assert len(pos) == 5
        assert len(neg) == 5

    def test_deterministic_seed(self):
        a_pos, a_neg = build_sentiment_prompts('gemma-2-2b-it', n=4, seed=7)
        b_pos, b_neg = build_sentiment_prompts('gemma-2-2b-it', n=4, seed=7)
        assert a_pos == b_pos
        assert a_neg == b_neg


class TestTemplates:
    def test_template_counts_match_legacy(self):
        assert len(POSITIVE_TEMPLATES) == 10
        assert len(NEGATIVE_TEMPLATES) == 10


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(['--model', 'meta-llama/Llama-3.1-8B-Instruct', '--layer', '19'])
        assert ns.model == 'meta-llama/Llama-3.1-8B-Instruct'
        assert ns.layer == 19


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(
            model='meta-llama/Llama-3.1-8B-Instruct',
            layer=19,
            n_prompts=10,
            n_devices=1,
            batch=2,
            top_k=20,
            n_perm=10,
            seed=7,
            reference_from='sae_feature_overlap',
        )
        cfg = from_args(ns)
        assert cfg.top_k == 20
        assert cfg.n_perm == 10


class TestRun:
    def test_returns_overlap_metrics(self, mocker, fake_ctx):
        mocker.patch.object(sae_sentiment_control, 'load_results', return_value=_reference_payload())

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(sae_sentiment_control, 'model_session', fake_session)
        mocker.patch.object(sae_sentiment_control, 'save_results')
        mocker.patch.object(sae_sentiment_control, 'load_sae_for_model', return_value=_fake_sae())
        mocker.patch.object(sae_sentiment_control, 'encode_prompts', side_effect=lambda *a, **k: _fake_acts(10, 64))

        cfg = SaeSentimentControlConfig(
            layer=19,
            n_prompts=10,
            top_k=10,
            n_perm=5,
        )
        result = run(cfg)
        assert {
            'reference_syc_lie_overlap',
            'syc_sent_overlap',
            'lie_sent_overlap',
            'shared_sent_overlap',
            'mcnemar_syc_lie_vs_syc_sent',
        } <= result.keys()
        assert {'in_lie_not_sent', 'in_sent_not_lie', 'p_value'} <= result['mcnemar_syc_lie_vs_syc_sent'].keys()

    def test_missing_reference_layer_raises(self, mocker, fake_ctx):
        mocker.patch.object(
            sae_sentiment_control,
            'load_results',
            return_value={
                'per_layer': [
                    {
                        'layer': 99,
                        'd_sae': 64,
                        'overlap': 0,
                        'syc_top_features': [],
                        'lie_top_features': [],
                        'shared_features': [],
                    }
                ]
            },
        )
        cfg = SaeSentimentControlConfig(layer=19)
        with pytest.raises(FileNotFoundError):
            run(cfg)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match='No SAE repo registered'):
            run(SaeSentimentControlConfig(model='not-real-model'))


class TestMcnemar:
    def test_zero_disc_returns_one(self):
        out = sae_sentiment_control._mcnemar_overlap({1, 2, 3}, {4, 5}, {6, 7})
        assert out['p_value'] == 1.0
        assert out['in_lie_not_sent'] == 0
        assert out['in_sent_not_lie'] == 0

    def test_returns_p_in_unit(self):
        out = sae_sentiment_control._mcnemar_overlap({1, 2, 3, 4}, {1, 2}, {3})
        assert 0.0 <= out['p_value'] <= 1.0


class TestOverlapPerm:
    def test_returns_int_and_p(self):
        rng = np.random.RandomState(0)
        flat = rng.randn(64)
        ov, p = sae_sentiment_control._overlap_perm({0, 1, 2, 3}, flat, top_k=4, n_perm=10, seed=42)
        assert isinstance(ov, int)
        assert 0.0 < p <= 1.0
