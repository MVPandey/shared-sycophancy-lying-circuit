"""Tests for the logit-lens trajectory analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import logit_lens
from shared_circuits.analyses.logit_lens import (
    LogitLensConfig,
    _diff_per_layer,
    _peak_excess,
    _per_layer_stats,
    _permutation_null,
    add_cli_args,
    from_args,
    run,
)
from shared_circuits.experiment import ExperimentContext
from shared_circuits.models import ModelInfo


@pytest.fixture
def fake_pairs():
    return [(f'q{i}', f'wrong{i}', f'right{i}') for i in range(40)]


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


class TestLogitLensConfig:
    def test_requires_model(self):
        with pytest.raises(ValidationError):
            LogitLensConfig()

    def test_defaults(self):
        cfg = LogitLensConfig(model='gemma-2-2b-it')
        assert cfg.n_pairs == 200
        assert cfg.n_perm == 1000
        assert cfg.n_boot == 1000

    def test_rejects_zero_pairs(self):
        with pytest.raises(ValidationError):
            LogitLensConfig(model='m', n_pairs=0)

    def test_is_frozen(self):
        cfg = LogitLensConfig(model='m')
        with pytest.raises(ValidationError):
            cfg.batch = 99


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            [
                '--model',
                'm',
                '--n-devices',
                '2',
                '--n-pairs',
                '40',
                '--n-perm',
                '50',
                '--n-boot',
                '60',
                '--batch',
                '8',
                '--seed',
                '7',
            ]
        )
        assert ns.model == 'm'
        assert ns.n_perm == 50
        assert ns.n_boot == 60

    def test_required_model(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(
            model='gemma-2-2b-it',
            n_devices=1,
            n_pairs=20,
            n_perm=10,
            n_boot=10,
            batch=2,
            seed=42,
        )
        cfg = from_args(ns)
        assert cfg.model == 'gemma-2-2b-it'
        assert cfg.n_perm == 10


class TestPerLayerStats:
    def test_empty_returns_empty(self):
        out = _per_layer_stats(np.zeros((0, 4)), n_boot=5, seed=0)
        assert out['n'] == 0
        assert out['mean'] == []

    def test_returns_means_and_ci(self):
        rng = np.random.RandomState(0)
        mat = rng.randn(20, 4)
        out = _per_layer_stats(mat, n_boot=10, seed=42)
        assert out['n'] == 20
        assert len(out['mean']) == 4
        assert len(out['ci_lo']) == 4
        for lo, hi in zip(out['ci_lo'], out['ci_hi'], strict=False):
            assert lo <= hi


class TestPermutationNull:
    def test_none_when_either_empty(self):
        assert _permutation_null(np.zeros((0, 3)), np.zeros((5, 3)), n_perm=2, seed=0) is None
        assert _permutation_null(np.zeros((5, 3)), np.zeros((0, 3)), n_perm=2, seed=0) is None

    def test_returns_p_value_per_layer(self):
        syc = np.zeros((10, 3))
        non = np.ones((10, 3))
        out = _permutation_null(syc, non, n_perm=20, seed=42)
        assert out is not None
        assert len(out['p_value_per_layer']) == 3
        for p in out['p_value_per_layer']:
            assert 0.0 <= p <= 1.0


class TestDiffAndPeak:
    def test_diff_per_layer_returns_subtract(self):
        syc = {'mean': [1.0, 2.0, 3.0]}
        non = {'mean': [2.0, 3.0, 5.0]}
        diffs = _diff_per_layer(syc, non)
        assert diffs == [1.0, 1.0, 2.0]

    def test_diff_per_layer_handles_empty(self):
        assert _diff_per_layer({'mean': []}, {'mean': [1.0]}) == []

    def test_peak_excess_computed(self):
        peak_layer, info = _peak_excess([0.1, 0.2, 1.0, 0.5])
        assert peak_layer == 2
        assert info is not None
        assert info['peak_diff'] == pytest.approx(1.0)
        assert info['final_diff'] == pytest.approx(0.5)
        assert info['excess_above_final'] == pytest.approx(0.5)


class TestRun:
    def test_returns_results_dict(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(logit_lens, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(logit_lens, 'model_session', fake_session)
        mocker.patch.object(logit_lens, 'save_results')

        rng = np.random.RandomState(0)
        n_prompts = 40
        # Stub the multi-layer extractor: returns numpy arrays per layer.
        mocker.patch.object(
            logit_lens,
            'extract_residual_stream_multi',
            return_value={layer: rng.randn(n_prompts, 32).astype(np.float32) for layer in range(4)},
        )
        # Bypass the W_U projection: hand back a deterministic per-prompt-per-layer matrix.
        mocker.patch.object(
            logit_lens,
            '_project_to_logit_diff',
            return_value=rng.randn(n_prompts, 4).astype(np.float32),
        )
        # Half the prompts are sycophantic, half not — keeps the permutation null nontrivial.
        indicators = [1.0 if i % 2 == 0 else 0.0 for i in range(n_prompts)]
        mocker.patch.object(
            logit_lens,
            'measure_agreement_per_prompt',
            return_value=(0.5, indicators),
        )

        cfg = LogitLensConfig(model='gemma-2-2b-it', n_pairs=n_prompts, n_perm=10, n_boot=5, batch=4)
        result = run(cfg)
        assert result['model'] == 'gemma-2-2b-it'
        assert result['n_layers'] == 4
        assert result['n_sycophantic'] == 20
        assert result['n_non_sycophantic'] == 20
        assert len(result['layers']) == 4
        assert len(result['diff_per_layer']) == 4
        assert len(result['perm_null_pvalue']) == 4
        assert result['peak_layer'] is not None
        assert 'permutation_null' in result

    def test_run_with_all_sycophantic_skips_perm_null(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(logit_lens, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(logit_lens, 'model_session', fake_session)
        mocker.patch.object(logit_lens, 'save_results')
        n_prompts = 10
        mocker.patch.object(
            logit_lens,
            'extract_residual_stream_multi',
            return_value={layer: np.zeros((n_prompts, 32), dtype=np.float32) for layer in range(4)},
        )
        mocker.patch.object(
            logit_lens,
            '_project_to_logit_diff',
            return_value=np.zeros((n_prompts, 4), dtype=np.float32),
        )
        # Every prompt is classified sycophantic so non_mat is empty and perm_null returns None.
        mocker.patch.object(
            logit_lens,
            'measure_agreement_per_prompt',
            return_value=(1.0, [1.0] * n_prompts),
        )

        cfg = LogitLensConfig(model='gemma-2-2b-it', n_pairs=n_prompts, n_perm=5, n_boot=3, batch=2)
        result = run(cfg)
        assert result['n_sycophantic'] == n_prompts
        assert result['n_non_sycophantic'] == 0
        assert result['permutation_null'] is None
        assert result['perm_null_pvalue'] == []

    def test_save_results_invoked(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(logit_lens, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(logit_lens, 'model_session', fake_session)
        save = mocker.patch.object(logit_lens, 'save_results')
        n_prompts = 6
        mocker.patch.object(
            logit_lens,
            'extract_residual_stream_multi',
            return_value={layer: np.zeros((n_prompts, 32), dtype=np.float32) for layer in range(4)},
        )
        mocker.patch.object(
            logit_lens,
            '_project_to_logit_diff',
            return_value=np.zeros((n_prompts, 4), dtype=np.float32),
        )
        mocker.patch.object(
            logit_lens,
            'measure_agreement_per_prompt',
            return_value=(0.5, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        )
        run(LogitLensConfig(model='gemma-2-2b-it', n_pairs=n_prompts, n_perm=3, n_boot=3))
        save.assert_called_once()
