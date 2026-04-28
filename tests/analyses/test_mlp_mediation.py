"""Tests for the MLP-mediation analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import mlp_mediation
from shared_circuits.analyses.mlp_mediation import (
    MlpMediationConfig,
    _excludes_zero,
    _load_shared_by_layer,
    _paired_ci,
    _pipeline_correlations,
    _safe_spearman,
    _select_mlp_candidates,
    add_cli_args,
    from_args,
    run,
)
from shared_circuits.experiment import ExperimentContext
from shared_circuits.models import ModelInfo


@pytest.fixture
def fake_pairs():
    return [(f'q{i}', f'wrong{i}', f'right{i}') for i in range(300)]


@pytest.fixture
def fake_ctx(mock_model):
    info = ModelInfo(name='Qwen/Qwen2.5-72B-Instruct', n_layers=80, n_heads=4, d_model=32, d_head=8, total_heads=320)
    return ExperimentContext(
        model=mock_model,
        info=info,
        model_name='Qwen/Qwen2.5-72B-Instruct',
        agree_tokens=(1, 2, 3),
        disagree_tokens=(4, 5, 6),
    )


class TestMlpMediationConfig:
    def test_requires_model(self):
        with pytest.raises(ValidationError):
            MlpMediationConfig()

    def test_defaults(self):
        cfg = MlpMediationConfig(model='gemma-2-2b-it')
        assert cfg.n_upstream == 8
        assert cfg.n_in_region == 8
        assert cfg.dir_prompts == 50
        assert cfg.test_prompts == 100
        assert cfg.shared_heads_from == 'circuit_overlap'
        assert cfg.shared_heads_k == 15

    def test_rejects_zero_upstream(self):
        with pytest.raises(ValidationError):
            MlpMediationConfig(model='m', n_upstream=0)

    def test_allows_zero_in_region(self):
        cfg = MlpMediationConfig(model='m', n_in_region=0)
        assert cfg.n_in_region == 0

    def test_is_frozen(self):
        cfg = MlpMediationConfig(model='m')
        with pytest.raises(ValidationError):
            cfg.batch = 99


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            [
                '--model',
                'Qwen/Qwen2.5-72B-Instruct',
                '--n-devices',
                '2',
                '--batch',
                '2',
                '--n-upstream',
                '8',
                '--n-in-region',
                '8',
                '--n-pairs',
                '300',
                '--dir-prompts',
                '50',
                '--test-prompts',
                '100',
                '--direction-layer',
                '56',
                '--shared-heads-from',
                'circuit_overlap',
                '--shared-heads-k',
                '15',
                '--n-boot',
                '500',
                '--seed',
                '7',
            ]
        )
        assert ns.model == 'Qwen/Qwen2.5-72B-Instruct'
        assert ns.n_upstream == 8
        assert ns.n_in_region == 8
        assert ns.direction_layer == 56

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
            batch=2,
            n_upstream=4,
            n_in_region=4,
            n_pairs=100,
            dir_prompts=20,
            test_prompts=20,
            direction_layer=None,
            shared_heads_from='circuit_overlap',
            shared_heads_k=15,
            n_boot=500,
            seed=42,
        )
        cfg = from_args(ns)
        assert cfg.n_upstream == 4
        assert cfg.direction_layer is None


class TestLoadSharedByLayer:
    def test_groups_by_layer(self, mocker):
        mocker.patch.object(
            mlp_mediation,
            'load_results',
            return_value={
                'overlap_by_K': [
                    {'K': 5, 'shared_heads': []},
                    {'K': 15, 'shared_heads': [[50, 1], [50, 3], [60, 2]]},
                ]
            },
        )
        out = _load_shared_by_layer('circuit_overlap', 'gemma-2-2b-it', top_k=15)
        assert out == {50: [1, 3], 60: [2]}


class TestSelectCandidates:
    def test_picks_upstream_below_first_shared(self):
        shared = {50: [1], 70: [2]}
        candidates, first, last = _select_mlp_candidates(shared, n_layers=80, n_upstream=4, n_in_region=4)
        assert first == 50
        assert last == 70
        # All upstream candidates are below 50
        for c in candidates:
            assert c < 50 or c >= 50  # both groups present
        upstream = [c for c in candidates if c < 50]
        in_region = [c for c in candidates if c >= 50]
        assert len(upstream) == 4
        assert len(in_region) == 4

    def test_clamps_upstream_when_first_is_small(self):
        shared = {2: [0]}
        candidates, _, _ = _select_mlp_candidates(shared, n_layers=80, n_upstream=8, n_in_region=8)
        # Only layers 0-1 are upstream
        upstream = [c for c in candidates if c < 2]
        assert all(0 <= c < 2 for c in upstream)

    def test_skips_in_region_when_zero(self):
        shared = {50: [1], 70: [2]}
        candidates, _, _ = _select_mlp_candidates(shared, n_layers=80, n_upstream=4, n_in_region=0)
        in_region = [c for c in candidates if c >= 50]
        assert in_region == []

    def test_no_upstream_when_first_is_zero(self):
        shared = {0: [1], 5: [2]}
        candidates, first, _ = _select_mlp_candidates(shared, n_layers=10, n_upstream=4, n_in_region=4)
        assert first == 0
        upstream = [c for c in candidates if c < first]
        assert upstream == []


class TestHelpers:
    def test_paired_ci_empty_returns_zeros(self):
        assert _paired_ci([], [], n_boot=10, seed=0) == (0.0, 0.0, 0.0)

    def test_paired_ci_returns_three_floats(self):
        mean, lo, hi = _paired_ci([1.0] * 20, [2.0] * 20, n_boot=20, seed=42)
        assert lo <= mean <= hi
        assert mean == pytest.approx(1.0, abs=1e-9)

    def test_safe_spearman_constant(self):
        rho, p = _safe_spearman(np.zeros(5), np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert np.isnan(rho)
        assert np.isnan(p)

    def test_safe_spearman_monotonic(self):
        rho, _ = _safe_spearman(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]))
        assert rho == pytest.approx(1.0)

    def test_excludes_zero(self):
        assert _excludes_zero([0.1, 0.5]) is True
        assert _excludes_zero([-0.5, -0.1]) is True
        assert _excludes_zero([-0.1, 0.5]) is False


class TestPipelineCorrelations:
    def test_returns_expected_keys(self):
        cands = {
            '10': {
                'logit_diff_delta': 0.1,
                'projection_delta': 0.2,
                'logit_diff_ci': [0.05, 0.15],
                'projection_ci': [0.1, 0.3],
            },
            '20': {
                'logit_diff_delta': 0.2,
                'projection_delta': 0.3,
                'logit_diff_ci': [0.15, 0.25],
                'projection_ci': [0.2, 0.4],
            },
            '50': {
                'logit_diff_delta': 0.05,
                'projection_delta': 0.1,
                'logit_diff_ci': [-0.05, 0.15],
                'projection_ci': [0.05, 0.15],
            },
            '60': {
                'logit_diff_delta': 0.0,
                'projection_delta': 0.0,
                'logit_diff_ci': [-0.1, 0.1],
                'projection_ci': [-0.1, 0.1],
            },
        }
        out = _pipeline_correlations(cands, [10, 20, 50, 60], first_shared=50)
        assert {'all_signed_rho', 'upstream_n', 'in_region_n', 'all_n'} <= out.keys()
        assert out['upstream_n'] == 2
        assert out['in_region_n'] == 2
        assert out['all_n'] == 4
        # both upstream CIs exclude zero on projection
        assert out['upstream_proj_ci_excludes_zero'] == 2


class TestRun:
    def test_returns_results_dict(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(mlp_mediation, 'load_triviaqa_pairs', return_value=fake_pairs)
        mocker.patch.object(
            mlp_mediation,
            'load_results',
            return_value={
                'overlap_by_K': [
                    {'K': 15, 'shared_heads': [[50, 0], [55, 1], [60, 2], [70, 3]]},
                ]
            },
        )

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(mlp_mediation, 'model_session', fake_session)
        mocker.patch.object(mlp_mediation, 'save_results')
        mocker.patch.object(mlp_mediation, '_extract_syc_direction', return_value=np.ones(32) / np.sqrt(32))
        mocker.patch.object(
            mlp_mediation,
            'build_sycophancy_prompts',
            return_value=(['w'] * 100, ['c'] * 100),
        )

        # Stub _measure to return deterministic per-prompt vectors
        def fake_measure(*_args, **kwargs):
            ablate = kwargs.get('ablate_mlp')
            offset = 0.1 if ablate is not None else 0.0
            return {
                'rate': 0.5 + offset,
                'logit_diff_mean': 0.0 + offset,
                'projection_mean': 0.0 + offset,
                'per_prompt_logit_diff': [offset] * 5,
                'per_prompt_projection': [offset] * 5,
            }

        mocker.patch.object(mlp_mediation, '_measure', side_effect=fake_measure)

        cfg = MlpMediationConfig(
            model='Qwen/Qwen2.5-72B-Instruct',
            n_upstream=4,
            n_in_region=4,
            n_pairs=10,
            dir_prompts=5,
            test_prompts=5,
            n_boot=10,
        )
        result = run(cfg)
        assert result['model'] == 'Qwen/Qwen2.5-72B-Instruct'
        assert 'shared_head_layers' in result
        assert 'mlp_candidates' in result
        assert 'candidates' in result
        assert 'pipeline_test' in result
        assert result['first_shared_layer'] == 50
        assert result['last_shared_layer'] == 70
        # each candidate has the expected fields
        for entry in result['candidates'].values():
            assert {
                'position',
                'logit_diff_delta',
                'projection_delta',
                'projection_ci',
                'logit_diff_ci',
            } <= entry.keys()
            assert entry['position'] in {'upstream', 'in_region'}

    def test_raises_when_no_shared_heads(self, mocker, fake_pairs):
        mocker.patch.object(mlp_mediation, 'load_triviaqa_pairs', return_value=fake_pairs)
        mocker.patch.object(
            mlp_mediation,
            'load_results',
            return_value={'overlap_by_K': [{'K': 15, 'shared_heads': []}]},
        )

        cfg = MlpMediationConfig(model='m')
        with pytest.raises(ValueError, match='no shared heads'):
            run(cfg)
