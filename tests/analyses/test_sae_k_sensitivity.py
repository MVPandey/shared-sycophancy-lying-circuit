"""Tests for the SAE K-sensitivity analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from shared_circuits.analyses import sae_k_sensitivity
from shared_circuits.analyses.sae_k_sensitivity import (
    SaeKSensitivityConfig,
    _parse_k_values,
    add_cli_args,
    from_args,
    run,
)
from shared_circuits.data.sae_features import SAE_KIND_GOODFIRE_TOPK, SaeWeights
from shared_circuits.experiment import ExperimentContext
from shared_circuits.models import ModelInfo


@pytest.fixture
def fake_pairs():
    return [(f'q{i}', f'wrong{i}', f'right{i}') for i in range(400)]


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


class TestSaeKSensitivityConfig:
    def test_defaults(self):
        cfg = SaeKSensitivityConfig()
        assert cfg.model == 'meta-llama/Llama-3.1-8B-Instruct'
        assert cfg.layer == 19
        assert cfg.k_values == (10, 50, 100, 200, 500)

    def test_rejects_negative_layer(self):
        with pytest.raises(ValidationError):
            SaeKSensitivityConfig(layer=-1)

    def test_rejects_zero_n_prompts(self):
        with pytest.raises(ValidationError):
            SaeKSensitivityConfig(n_prompts=0)

    def test_is_frozen(self):
        cfg = SaeKSensitivityConfig()
        with pytest.raises(ValidationError):
            cfg.layer = 99


class TestParseKValues:
    def test_default_when_none(self):
        assert _parse_k_values(None) == (10, 50, 100, 200, 500)

    def test_parses_csv(self):
        assert _parse_k_values('5,10,20') == (5, 10, 20)


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(['--model', 'meta-llama/Llama-3.1-8B-Instruct', '--layer', '19', '--k-values', '5,10'])
        assert ns.model == 'meta-llama/Llama-3.1-8B-Instruct'
        assert ns.layer == 19
        assert ns.k_values == '5,10'


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(
            model='meta-llama/Llama-3.1-8B-Instruct',
            layer=19,
            k_values='5,10,20',
            n_prompts=10,
            n_devices=1,
            batch=2,
            seed=7,
        )
        cfg = from_args(ns)
        assert cfg.k_values == (5, 10, 20)
        assert cfg.n_prompts == 10


class TestRun:
    def test_returns_curve(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(sae_k_sensitivity, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(sae_k_sensitivity, 'model_session', fake_session)
        mocker.patch.object(sae_k_sensitivity, 'save_results')
        mocker.patch.object(sae_k_sensitivity, 'load_sae_for_model', return_value=_fake_sae())
        mocker.patch.object(sae_k_sensitivity, 'encode_prompts', side_effect=lambda *a, **k: _fake_acts(20, 64))

        cfg = SaeKSensitivityConfig(
            model='meta-llama/Llama-3.1-8B-Instruct',
            layer=19,
            k_values=(5, 10, 20),
            n_prompts=20,
            batch=2,
        )
        result = run(cfg)
        assert result['layer'] == 19
        assert len(result['curve']) == 3
        for row in result['curve']:
            assert {'k', 'overlap', 'chance', 'ratio', 'p_hypergeometric'} <= row.keys()

    def test_unknown_model_raises(self):
        cfg = SaeKSensitivityConfig(model='not-real-model')
        with pytest.raises(ValueError, match='No SAE repo registered'):
            run(cfg)

    def test_empty_k_raises(self):
        with pytest.raises(ValueError, match='at least one K'):
            run(SaeKSensitivityConfig(k_values=()))

    def test_k_above_d_sae_raises(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(sae_k_sensitivity, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(sae_k_sensitivity, 'model_session', fake_session)
        mocker.patch.object(sae_k_sensitivity, 'save_results')
        mocker.patch.object(sae_k_sensitivity, 'load_sae_for_model', return_value=_fake_sae())
        mocker.patch.object(sae_k_sensitivity, 'encode_prompts', side_effect=lambda *a, **k: _fake_acts(20, 64))

        cfg = SaeKSensitivityConfig(k_values=(99999,), n_prompts=20)
        with pytest.raises(ValueError, match='exceeds d_sae'):
            run(cfg)
