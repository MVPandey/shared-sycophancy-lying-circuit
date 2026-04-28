"""Tests for the SAE feature-overlap analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from shared_circuits.analyses import sae_feature_overlap
from shared_circuits.analyses.sae_feature_overlap import (
    SaeFeatureOverlapConfig,
    _parse_layers_arg,
    add_cli_args,
    from_args,
    run,
)
from shared_circuits.data.sae_features import SAE_KIND_GOODFIRE_TOPK, SaeWeights
from shared_circuits.experiment import ExperimentContext
from shared_circuits.models import ModelInfo


def _fake_sae() -> SaeWeights:
    """Tiny stand-in for SaeWeights — only the attributes the analysis reads."""
    return SaeWeights(
        kind=SAE_KIND_GOODFIRE_TOPK,
        layer=12,
        repo='unit/test',
        w_enc=torch.zeros(32, 32),
        b_enc=torch.zeros(32),
        b_dec=torch.zeros(32),
        top_k=8,
    )


@pytest.fixture
def fake_pairs():
    return [(f'q{i}', f'wrong{i}', f'right{i}') for i in range(400)]


@pytest.fixture
def fake_ctx(mock_model):
    info = ModelInfo(name='gemma-2-2b-it', n_layers=4, n_heads=4, d_model=32, d_head=8, total_heads=16)
    return ExperimentContext(
        model=mock_model,
        info=info,
        model_name='gemma-2-2b-it',
        agree_tokens=(1, 2, 3),
        disagree_tokens=(4, 5, 6),
    )


def _fake_acts(n_prompts: int = 100, d_sae: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n_prompts, d_sae).astype(np.float32)


class TestSaeFeatureOverlapConfig:
    def test_defaults(self):
        cfg = SaeFeatureOverlapConfig()
        assert cfg.top_k == 100
        assert cfg.n_prompts == 100
        assert cfg.n_perm == 1000
        assert cfg.batch == 4

    def test_rejects_zero_top_k(self):
        with pytest.raises(ValidationError):
            SaeFeatureOverlapConfig(top_k=0)

    def test_rejects_zero_n_prompts(self):
        with pytest.raises(ValidationError):
            SaeFeatureOverlapConfig(n_prompts=0)

    def test_is_frozen(self):
        cfg = SaeFeatureOverlapConfig()
        with pytest.raises(ValidationError):
            cfg.top_k = 5


class TestParseLayersArg:
    def test_returns_defaults_when_none(self):
        out = _parse_layers_arg(None, ['gemma-2-2b-it'])
        assert out == {'gemma-2-2b-it': (12, 19)}

    def test_overrides_specific_model(self):
        out = _parse_layers_arg(['gemma-2-2b-it=5,6'], ['gemma-2-2b-it'])
        assert out['gemma-2-2b-it'] == (5, 6)

    def test_drops_unselected_models(self):
        out = _parse_layers_arg(None, ['gemma-2-2b-it'])
        assert 'meta-llama/Llama-3.1-8B-Instruct' not in out

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match='Expected MODEL=L1,L2'):
            _parse_layers_arg(['no-equals'], ['gemma-2-2b-it'])


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(['--models', 'gemma-2-2b-it', '--top-k', '50', '--n-prompts', '20', '--batch', '2'])
        assert ns.models == ['gemma-2-2b-it']
        assert ns.top_k == 50
        assert ns.n_prompts == 20

    def test_defaults_set(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args([])
        assert ns.top_k == 100
        assert isinstance(ns.models, list)


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(
            models=['gemma-2-2b-it'],
            layers=None,
            top_k=50,
            n_prompts=20,
            n_devices=1,
            batch=2,
            n_perm=10,
            n_boot=0,
            seed=7,
        )
        cfg = from_args(ns)
        assert cfg.models == ('gemma-2-2b-it',)
        assert cfg.top_k == 50
        assert cfg.layers['gemma-2-2b-it'] == (12, 19)


class TestRun:
    def test_dispatches_per_model(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(sae_feature_overlap, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(sae_feature_overlap, 'model_session', fake_session)
        mocker.patch.object(sae_feature_overlap, 'save_results')
        mocker.patch.object(sae_feature_overlap, 'load_sae_for_model', return_value=_fake_sae())
        mocker.patch.object(sae_feature_overlap, 'encode_prompts', side_effect=lambda *a, **k: _fake_acts(20, 32))

        cfg = SaeFeatureOverlapConfig(
            models=('gemma-2-2b-it',),
            layers={'gemma-2-2b-it': (12,)},
            top_k=4,
            n_prompts=20,
            batch=2,
            n_perm=10,
            n_boot=5,
        )
        results = run(cfg)
        assert len(results) == 1
        r = results[0]
        assert r['model'] == 'gemma-2-2b-it'
        assert len(r['per_layer']) == 1
        layer_res = r['per_layer'][0]
        assert {'overlap', 'p_permutation', 'p_hypergeometric', 'jaccard', 'spearman_rho'} <= layer_res.keys()

    def test_unknown_model_raises(self):
        # without an explicit `layers` entry the analysis can't know which depth to project
        cfg = SaeFeatureOverlapConfig(models=('definitely-not-a-real-model',))
        with pytest.raises(ValueError, match='No layers configured'):
            run(cfg)

    def test_layers_missing_raises(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(sae_feature_overlap, 'load_triviaqa_pairs', return_value=fake_pairs)
        # Missing entry for the model in cfg.layers
        cfg = SaeFeatureOverlapConfig(
            models=('gemma-2-2b-it',),
            layers={'google/gemma-2-9b-it': (21,)},
        )
        with pytest.raises(ValueError, match='No layers configured'):
            run(cfg)

    def test_save_results_invoked(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(sae_feature_overlap, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(sae_feature_overlap, 'model_session', fake_session)
        save = mocker.patch.object(sae_feature_overlap, 'save_results')
        mocker.patch.object(sae_feature_overlap, 'load_sae_for_model', return_value=_fake_sae())
        mocker.patch.object(sae_feature_overlap, 'encode_prompts', side_effect=lambda *a, **k: _fake_acts(20, 32))

        cfg = SaeFeatureOverlapConfig(
            models=('gemma-2-2b-it',),
            layers={'gemma-2-2b-it': (12,)},
            top_k=4,
            n_prompts=20,
            n_perm=5,
            n_boot=0,
        )
        run(cfg)
        save.assert_called_once()


class TestOverlapPermPvalue:
    def test_returns_float_in_unit_interval(self):
        rng = np.random.RandomState(0)
        sd = rng.randn(64)
        ld = rng.randn(64)
        p = sae_feature_overlap._overlap_perm_pvalue(sd, ld, top_k=8, n_perm=20, seed=42)
        assert 0.0 < p <= 1.0


class TestBootstrap:
    def test_returns_expected_keys(self):
        rng = np.random.RandomState(0)

        def acts() -> np.ndarray:
            return rng.randn(20, 32).astype(np.float32)

        boot = sae_feature_overlap._bootstrap_overlap(acts(), acts(), acts(), acts(), top_k=4, n_boot=5, seed=42)
        assert {
            'overlap_mean',
            'overlap_ci',
            'jaccard_mean',
            'jaccard_ci',
            'spearman_rho_mean',
            'spearman_rho_ci',
            'n_boot',
        } <= boot.keys()
