"""Tests for the linear-probe / SAE alignment analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from shared_circuits.analyses import linear_probe_sae_alignment
from shared_circuits.analyses.linear_probe_sae_alignment import (
    LinearProbeSaeAlignmentConfig,
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
        d_model=32,
        d_head=4,
        total_heads=1024,
    )
    return ExperimentContext(
        model=mock_model,
        info=info,
        model_name='meta-llama/Llama-3.1-8B-Instruct',
        agree_tokens=(1, 2),
        disagree_tokens=(3, 4),
    )


def _fake_sae(d_model: int = 32, d_sae: int = 64) -> SaeWeights:
    rng = np.random.RandomState(0)
    return SaeWeights(
        kind=SAE_KIND_GOODFIRE_TOPK,
        layer=19,
        repo='Goodfire/test',
        w_enc=torch.tensor(rng.randn(d_model, d_sae).astype(np.float32)),
        b_enc=torch.zeros(d_sae),
        b_dec=torch.zeros(d_model),
        w_dec=torch.tensor(rng.randn(d_model, d_sae).astype(np.float32)),
        top_k=8,
    )


def _fake_resid(n_prompts: int = 30, d_model: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n_prompts, d_model).astype(np.float32)


def _reference_payload(layer: int = 19) -> dict:
    return {'per_layer': [{'layer': layer, 'shared_features': list(range(8))}]}


class TestLinearProbeSaeAlignmentConfig:
    def test_defaults(self):
        cfg = LinearProbeSaeAlignmentConfig()
        assert cfg.layer == 19
        assert cfg.top_k_overlap == 41
        assert cfg.n_folds == 5

    def test_rejects_negative_layer(self):
        with pytest.raises(ValidationError):
            LinearProbeSaeAlignmentConfig(layer=-1)

    def test_n_folds_minimum(self):
        with pytest.raises(ValidationError):
            LinearProbeSaeAlignmentConfig(n_folds=1)

    def test_is_frozen(self):
        cfg = LinearProbeSaeAlignmentConfig()
        with pytest.raises(ValidationError):
            cfg.top_k_overlap = 5


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            [
                '--model',
                'meta-llama/Llama-3.1-8B-Instruct',
                '--layer',
                '19',
                '--top-k-overlap',
                '20',
                '--n-folds',
                '3',
            ]
        )
        assert ns.layer == 19
        assert ns.top_k_overlap == 20
        assert ns.n_folds == 3


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(
            model='meta-llama/Llama-3.1-8B-Instruct',
            layer=19,
            n_prompts=10,
            n_devices=1,
            batch=2,
            top_k_overlap=20,
            n_folds=3,
            n_perm_subspace=20,
            seed=7,
            overlap_from='sae_feature_overlap',
        )
        cfg = from_args(ns)
        assert cfg.layer == 19
        assert cfg.top_k_overlap == 20
        assert cfg.n_perm_subspace == 20


class TestRun:
    def test_returns_probe_blocks(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(linear_probe_sae_alignment, 'load_results', return_value=_reference_payload())
        mocker.patch.object(linear_probe_sae_alignment, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(linear_probe_sae_alignment, 'model_session', fake_session)
        mocker.patch.object(linear_probe_sae_alignment, 'save_results')
        mocker.patch.object(linear_probe_sae_alignment, 'load_sae_for_model', return_value=_fake_sae())
        mocker.patch.object(
            linear_probe_sae_alignment,
            'extract_residual_stream',
            side_effect=lambda *a, **k: _fake_resid(30, 32),
        )

        cfg = LinearProbeSaeAlignmentConfig(
            n_prompts=30,
            top_k_overlap=4,
            n_folds=3,
            n_perm_subspace=10,
        )
        result = run(cfg)
        assert {'syc_probe', 'lie_probe'} <= result.keys()
        for block in (result['syc_probe'], result['lie_probe']):
            assert {'auroc_cv', 'top_aligned', 'overlap_stats', 'subspace_norm_fraction'} <= block.keys()

    def test_missing_overlap_layer_raises(self, mocker, fake_ctx):
        mocker.patch.object(
            linear_probe_sae_alignment,
            'load_results',
            return_value={'per_layer': [{'layer': 99, 'shared_features': []}]},
        )
        cfg = LinearProbeSaeAlignmentConfig(layer=19)
        with pytest.raises(FileNotFoundError):
            run(cfg)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match='No SAE repo registered'):
            run(LinearProbeSaeAlignmentConfig(model='not-real-model'))

    def test_missing_w_dec_raises(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(linear_probe_sae_alignment, 'load_results', return_value=_reference_payload())
        mocker.patch.object(linear_probe_sae_alignment, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(linear_probe_sae_alignment, 'model_session', fake_session)
        mocker.patch.object(linear_probe_sae_alignment, 'save_results')
        sae_no_dec = SaeWeights(
            kind=SAE_KIND_GOODFIRE_TOPK,
            layer=19,
            repo='unit/test',
            w_enc=torch.zeros(32, 64),
            b_enc=torch.zeros(64),
            b_dec=torch.zeros(32),
            top_k=8,
        )
        mocker.patch.object(linear_probe_sae_alignment, 'load_sae_for_model', return_value=sae_no_dec)
        mocker.patch.object(
            linear_probe_sae_alignment,
            'extract_residual_stream',
            side_effect=lambda *a, **k: _fake_resid(20, 32),
        )

        cfg = LinearProbeSaeAlignmentConfig(n_prompts=20, top_k_overlap=4, n_folds=3, n_perm_subspace=5)
        with pytest.raises(ValueError, match='did not load W_dec'):
            run(cfg)


class TestAlignmentPerFeature:
    def test_zero_norm_safe(self):
        probe = np.zeros(8, dtype=np.float32)
        w_dec = np.eye(8, 16, dtype=np.float32)
        out = linear_probe_sae_alignment._alignment_per_feature(probe, w_dec)
        assert out.shape == (16,)
        # alignment of zero probe with anything is zero
        assert np.allclose(out, 0)

    def test_returns_unit_aligned(self):
        rng = np.random.RandomState(0)
        d = 8
        d_sae = 4
        w_dec = rng.randn(d, d_sae).astype(np.float32)
        probe = w_dec[:, 0].copy()
        out = linear_probe_sae_alignment._alignment_per_feature(probe, w_dec)
        # feature 0 should have the highest alignment with probe = its own column
        assert int(np.argmax(out)) == 0


class TestSubspaceNormFraction:
    def test_full_subspace_yields_one(self):
        rng = np.random.RandomState(0)
        d = 8
        w_dec = rng.randn(d, 16).astype(np.float32)
        probe = rng.randn(d).astype(np.float32)
        # When the chosen feature subset spans the whole d-dim space, norm fraction is 1
        out = linear_probe_sae_alignment._subspace_norm_fraction(probe, w_dec, list(range(d)))
        assert out == pytest.approx(1.0, abs=1e-5)

    def test_zero_subspace_when_no_indices_overlap_probe(self):
        d = 4
        # build w_dec so that the first column is along axis 0 and probe is along axis 1
        w_dec = np.zeros((d, 4), dtype=np.float32)
        w_dec[:, 0] = [1.0, 0.0, 0.0, 0.0]
        probe = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        frac = linear_probe_sae_alignment._subspace_norm_fraction(probe, w_dec, [0])
        assert frac == pytest.approx(0.0, abs=1e-5)


class TestOverlapStats:
    def test_returns_expected_keys(self):
        out = linear_probe_sae_alignment._overlap_stats([0, 1, 2], [1, 2, 3], d_sae=10, top_k=3)
        assert {'overlap', 'top_k_aligned', 'n_shared', 'd_sae', 'chance_overlap', 'ratio_vs_chance'} <= out.keys()
        assert out['overlap'] == 2
