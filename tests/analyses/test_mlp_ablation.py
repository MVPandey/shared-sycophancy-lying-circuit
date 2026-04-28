"""Tests for the MLP-ablation analysis (ablation / disruption / tugofwar modes)."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import mlp_ablation
from shared_circuits.analyses.mlp_ablation import (
    MlpAblationConfig,
    _default_target_layers,
    _distance_test,
    _membership_test,
    _parse_layers,
    _resolve_target_layers,
    _safe_spearman,
    _shared_importance_per_layer,
    _shared_layers_set,
    add_cli_args,
    from_args,
    run,
)
from shared_circuits.experiment import ExperimentContext
from shared_circuits.models import ModelInfo


@pytest.fixture
def fake_pairs():
    return [(f'q{i}', f'wrong{i}', f'right{i}') for i in range(200)]


@pytest.fixture
def fake_ctx(mock_model):
    info = ModelInfo(name='gemma-2-2b-it', n_layers=8, n_heads=4, d_model=32, d_head=8, total_heads=32)
    return ExperimentContext(
        model=mock_model,
        info=info,
        model_name='gemma-2-2b-it',
        agree_tokens=(1, 2, 3),
        disagree_tokens=(4, 5, 6),
    )


class TestMlpAblationConfig:
    def test_requires_model(self):
        with pytest.raises(ValidationError):
            MlpAblationConfig()

    def test_defaults(self):
        cfg = MlpAblationConfig(model='gemma-2-2b-it')
        assert cfg.mode == 'ablation'
        assert cfg.test_prompts == 100
        assert cfg.layers is None
        assert cfg.shared_heads_from == 'circuit_overlap'
        assert cfg.mlp_results_from == 'mlp_ablation'

    def test_rejects_zero_test_prompts(self):
        with pytest.raises(ValidationError):
            MlpAblationConfig(model='m', test_prompts=0)

    def test_rejects_zero_threshold(self):
        with pytest.raises(ValidationError):
            MlpAblationConfig(model='m', ppl_ratio_threshold=0)

    def test_is_frozen(self):
        cfg = MlpAblationConfig(model='m')
        with pytest.raises(ValidationError):
            cfg.batch = 99


class TestParseLayers:
    def test_none_returns_none(self):
        assert _parse_layers(None) is None

    def test_csv_returns_tuple(self):
        assert _parse_layers('1,5,10') == (1, 5, 10)

    def test_strips_empty(self):
        assert _parse_layers('1,,5') == (1, 5)


class TestDefaultTargetLayers:
    def test_returns_in_range(self):
        layers = _default_target_layers(n_layers=10)
        assert all(0 <= l < 10 for l in layers)
        assert layers == sorted(layers)

    def test_skips_negative(self):
        # for very small models negative candidates fall out
        layers = _default_target_layers(n_layers=4)
        assert all(l >= 0 for l in layers)


class TestResolveTargetLayers:
    def test_explicit_layers_filtered(self):
        # 99 is out of range and dropped
        out = _resolve_target_layers((1, 3, 99), n_layers=10)
        assert out == [1, 3]

    def test_falls_back_to_default(self):
        out = _resolve_target_layers(None, n_layers=10)
        assert out == _default_target_layers(10)


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            [
                '--model',
                'gemma-2-2b-it',
                '--n-devices',
                '2',
                '--mode',
                'disruption',
                '--batch',
                '4',
                '--n-pairs',
                '50',
                '--test-prompts',
                '20',
                '--layers',
                '11,13,14,16',
                '--ppl-ratio-threshold',
                '3.0',
                '--shared-heads-from',
                'circuit_overlap',
                '--mlp-results-from',
                'mlp_ablation',
                '--seed',
                '7',
            ]
        )
        assert ns.model == 'gemma-2-2b-it'
        assert ns.mode == 'disruption'
        assert ns.layers == '11,13,14,16'
        assert ns.ppl_ratio_threshold == pytest.approx(3.0)
        assert ns.mlp_results_from == 'mlp_ablation'

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
            mode='ablation',
            batch=8,
            n_pairs=200,
            test_prompts=100,
            layers='1,2,3',
            ppl_ratio_threshold=5.0,
            shared_heads_from='circuit_overlap',
            mlp_results_from='mlp_ablation',
            seed=42,
        )
        cfg = from_args(ns)
        assert cfg.layers == (1, 2, 3)
        assert cfg.mode == 'ablation'

    def test_defaults_layers_when_none(self):
        ns = argparse.Namespace(
            model='m',
            n_devices=1,
            mode='ablation',
            batch=8,
            n_pairs=10,
            test_prompts=5,
            layers=None,
            ppl_ratio_threshold=5.0,
            shared_heads_from='circuit_overlap',
            mlp_results_from='mlp_ablation',
            seed=42,
        )
        cfg = from_args(ns)
        assert cfg.layers is None


class TestSharedImportance:
    def test_returns_three_variants(self):
        sg = np.zeros((4, 4))
        sg[0, 0] = 5
        sg[0, 1] = 4
        lg = np.zeros((4, 4))
        lg[0, 0] = 6
        lg[1, 0] = 5
        result = _shared_importance_per_layer(sg, lg)
        assert 'intersect_topk' in result
        assert 'sum_min' in result
        assert 'sum_geomean' in result
        assert result['k'] == 4
        assert result['total_heads'] == 16

    def test_shared_layers_set(self):
        sg = np.zeros((4, 4))
        sg[0, 0] = 5
        lg = np.zeros((4, 4))
        lg[0, 0] = 5
        s = _shared_layers_set(sg, lg)
        assert 0 in s


class TestSafeSpearman:
    def test_constant_returns_nan(self):
        rho, p = _safe_spearman(np.zeros(5), np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert np.isnan(rho)
        assert np.isnan(p)

    def test_too_few_returns_nan(self):
        rho, _ = _safe_spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert np.isnan(rho)

    def test_monotonic_returns_one(self):
        rho, _ = _safe_spearman(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]))
        assert rho == pytest.approx(1.0)


class TestMembershipDistance:
    def test_membership_test_returns_keys(self):
        out = _membership_test([0, 1, 2], np.array([0.1, 0.2, 0.3]), {0, 1})
        assert {'in_shared', 'out_shared', 'mean_in', 'mean_out', 'u_statistic', 'p_one_sided'} <= out.keys()

    def test_distance_test_no_shared(self):
        out = _distance_test([0, 1, 2], np.array([0.1, 0.2, 0.3]), set())
        assert np.isnan(out['rho'])

    def test_distance_test_with_shared(self):
        out = _distance_test([0, 1, 2, 3], np.array([0.1, 0.2, 0.3, 0.4]), {0})
        # distances are 0, 1, 2, 3 — distinct values so a finite rho is computed
        assert not np.isnan(out['rho'])


class TestRunDispatch:
    def test_unknown_mode_rejected(self):
        cfg = MlpAblationConfig(model='m', mode='__bogus__')
        with pytest.raises(ValueError, match='unknown mode'):
            run(cfg)

    def test_ablation_mode_returns_layer_effects(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(mlp_ablation, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(mlp_ablation, 'model_session', fake_session)
        mocker.patch.object(mlp_ablation, 'save_results')
        mocker.patch.object(
            mlp_ablation,
            'build_sycophancy_prompts',
            return_value=(['p1', 'p2', 'p3'], ['c1', 'c2', 'c3']),
        )
        rates = iter([0.5, 0.6, 0.55, 0.45, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        mocker.patch.object(
            mlp_ablation,
            'measure_agreement_rate',
            side_effect=lambda *a, **kw: next(rates),
        )

        cfg = MlpAblationConfig(model='gemma-2-2b-it', mode='ablation', test_prompts=3, n_pairs=10)
        result = run(cfg)
        assert result['mode'] == 'ablation'
        assert result['baseline_rate'] == pytest.approx(0.5)
        assert 'layer_effects' in result
        for entry in result['layer_effects'].values():
            assert 'rate' in entry
            assert 'delta' in entry

    def test_disruption_mode_returns_perplexity(self, mocker, fake_ctx):
        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(mlp_ablation, 'model_session', fake_session)
        mocker.patch.object(mlp_ablation, 'save_results')
        # baseline + per-layer ppl
        ppls = iter([10.0, 12.0, 50.0, 20.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0])
        mocker.patch.object(mlp_ablation, '_compute_perplexity', side_effect=lambda *a, **kw: next(ppls))

        cfg = MlpAblationConfig(model='gemma-2-2b-it', mode='disruption', layers=(1, 2, 3))
        result = run(cfg)
        assert result['mode'] == 'disruption'
        assert result['baseline_perplexity'] == pytest.approx(10.0)
        assert set(result['layers']) == {'1', '2', '3'}
        # 50/10 == 5.0 not strictly < threshold — ratio==threshold considered "degraded"
        assert result['layers']['2']['ratio'] == pytest.approx(5.0)
        assert result['layers']['2']['specific'] is False
        # 12/10 < 5.0 — specific
        assert result['layers']['1']['specific'] is True

    def test_tugofwar_mode_returns_correlations(self, mocker):
        # 4-layer x 4-head grids; layer 0 shared by tops in both
        sg = np.zeros((4, 4))
        sg[0, 0] = 5.0
        sg[1, 1] = 4.0
        lg = np.zeros((4, 4))
        lg[0, 0] = 5.0
        lg[1, 1] = 4.0

        def fake_load_results(slug: str, _name: str) -> dict:
            if slug == 'circuit_overlap':
                return {'syc_grid': sg.tolist(), 'lie_grid': lg.tolist()}
            if slug == 'mlp_ablation':
                return {
                    'layer_effects': {
                        '0': {'delta': 0.3},
                        '1': {'delta': 0.2},
                        '2': {'delta': 0.05},
                        '3': {'delta': 0.01},
                    }
                }
            raise FileNotFoundError(slug)

        mocker.patch.object(mlp_ablation, 'load_results', side_effect=fake_load_results)
        mocker.patch.object(mlp_ablation, 'save_results')
        cfg = MlpAblationConfig(model='gemma-2-2b-it', mode='tugofwar')
        result = run(cfg)
        assert result['mode'] == 'tugofwar'
        assert result['n_layers_grid'] == 4
        assert 'by_variant' in result
        assert {'intersect_topk', 'sum_min', 'sum_geomean'} <= result['by_variant'].keys()
        assert 'membership_test' in result
        assert 'distance_test' in result

    def test_tugofwar_raises_when_no_layer_effects(self, mocker):
        sg = np.zeros((4, 4))
        sg[0, 0] = 5.0
        lg = np.zeros((4, 4))
        lg[0, 0] = 5.0

        def fake_load(slug: str, _name: str) -> dict:
            if slug == 'circuit_overlap':
                return {'syc_grid': sg.tolist(), 'lie_grid': lg.tolist()}
            return {}

        mocker.patch.object(mlp_ablation, 'load_results', side_effect=fake_load)
        cfg = MlpAblationConfig(model='m', mode='tugofwar')
        with pytest.raises(ValueError, match='no layer_effects'):
            run(cfg)
