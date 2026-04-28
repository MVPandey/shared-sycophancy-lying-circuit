"""Tests for the faithfulness analysis."""

import argparse
from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from shared_circuits.analyses import faithfulness
from shared_circuits.analyses.faithfulness import (
    FaithfulnessConfig,
    _faithfulness,
    _parse_k_values,
    _ranked_from_attribution,
    _ranked_from_overlap,
    _select_k_values,
    _wilson_ci,
    _zero_all_except,
    add_cli_args,
    from_args,
    run,
)
from shared_circuits.experiment import ExperimentContext
from shared_circuits.models import ModelInfo


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


class TestFaithfulnessConfig:
    def test_requires_model(self):
        with pytest.raises(ValidationError):
            FaithfulnessConfig()

    def test_defaults(self):
        cfg = FaithfulnessConfig(model='gemma-2-2b-it')
        assert cfg.mode == 'curve'
        assert cfg.shared_heads_from == 'circuit_overlap'
        assert cfg.shared_heads_k == 15
        assert cfg.faithfulness_threshold == pytest.approx(0.8)
        assert cfg.k_values == (1, 2, 5, 10)

    def test_rejects_invalid_threshold(self):
        with pytest.raises(ValidationError):
            FaithfulnessConfig(model='m', faithfulness_threshold=2.0)

    def test_rejects_zero_devices(self):
        with pytest.raises(ValidationError):
            FaithfulnessConfig(model='m', n_devices=0)

    def test_is_frozen(self):
        cfg = FaithfulnessConfig(model='m')
        with pytest.raises(ValidationError):
            cfg.batch = 99


class TestParseKValues:
    def test_default_when_none(self):
        assert _parse_k_values(None) == (1, 2, 5, 10)

    def test_parses_csv(self):
        assert _parse_k_values('1,3,7') == (1, 3, 7)

    def test_strips_empty(self):
        assert _parse_k_values('1,,3') == (1, 3)


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
                '--n-prompts',
                '50',
                '--n-pairs',
                '200',
                '--mode',
                'single',
                '--k-values',
                '1,5,10',
                '--batch',
                '4',
                '--seed',
                '7',
                '--shared-heads-from',
                'attribution_patching',
                '--shared-heads-k',
                '20',
                '--faithfulness-threshold',
                '0.5',
            ]
        )
        assert ns.model == 'gemma-2-2b-it'
        assert ns.mode == 'single'
        assert ns.k_values == '1,5,10'
        assert ns.shared_heads_from == 'attribution_patching'
        assert ns.faithfulness_threshold == pytest.approx(0.5)

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
            n_prompts=100,
            n_pairs=200,
            mode='single',
            k_values='1,5,10',
            batch=8,
            seed=42,
            shared_heads_from='circuit_overlap',
            shared_heads_k=15,
            faithfulness_threshold=0.8,
        )
        cfg = from_args(ns)
        assert cfg.k_values == (1, 5, 10)
        assert cfg.mode == 'single'

    def test_defaults_k_values_when_none(self):
        ns = argparse.Namespace(
            model='m',
            n_devices=1,
            n_prompts=10,
            n_pairs=20,
            mode='curve',
            k_values=None,
            batch=8,
            seed=42,
            shared_heads_from='circuit_overlap',
            shared_heads_k=15,
            faithfulness_threshold=0.8,
        )
        cfg = from_args(ns)
        assert cfg.k_values == (1, 2, 5, 10)


class TestHelpers:
    def test_wilson_ci_zero_n(self):
        lo, hi = _wilson_ci(0.5, n=0)
        assert lo == 0.0
        assert hi == 0.0

    def test_wilson_ci_known_bounds(self):
        # 50/100 with z=1.96 should give roughly [0.40, 0.60]
        lo, hi = _wilson_ci(0.5, n=100)
        assert 0.0 <= lo < 0.5 < hi <= 1.0
        # interval is bounded inside [0,1]
        lo2, hi2 = _wilson_ci(1.0, n=10)
        assert 0.0 <= lo2 <= 1.0
        assert 0.0 <= hi2 <= 1.0

    def test_faithfulness_zero_denominator(self):
        # baseline == ablated — division-by-zero guard returns 0
        assert _faithfulness(metric_k=0.5, metric_ablated=0.5, metric_baseline=0.5) == 0.0

    def test_faithfulness_full_recovery(self):
        # restored to baseline => 1.0
        assert _faithfulness(metric_k=0.9, metric_ablated=0.5, metric_baseline=0.9) == pytest.approx(1.0)

    def test_faithfulness_no_recovery(self):
        # restored back to ablated => 0.0
        assert _faithfulness(metric_k=0.5, metric_ablated=0.5, metric_baseline=0.9) == pytest.approx(0.0)

    def test_select_k_values_curve_full_range(self):
        cfg = FaithfulnessConfig(model='m', mode='curve')
        assert _select_k_values(cfg, n_shared=4) == [1, 2, 3, 4]

    def test_select_k_values_single_clamped(self):
        cfg = FaithfulnessConfig(model='m', mode='single', k_values=(1, 2, 5, 10))
        ks = _select_k_values(cfg, n_shared=3)
        # full set is added; values >n_shared dropped
        assert ks == [1, 2, 3]

    def test_zero_all_except_skips_layer_with_full_keep(self):
        # all heads kept at layer 0 => no hook for layer 0
        keep = {(0, h) for h in range(4)}
        hooks = _zero_all_except(n_layers=2, n_heads=4, keep=keep)
        layers = [int(name.split('.')[1]) for name, _ in hooks]
        assert 0 not in layers
        assert 1 in layers


class TestRankedFromOverlap:
    def test_pulls_from_top_k_bucket(self):
        data = {
            'overlap_by_K': [
                {'K': 5, 'shared_heads': [[0, 0]]},
                {'K': 15, 'shared_heads': [[1, 1], [2, 2]]},
            ]
        }
        out = _ranked_from_overlap(data, top_k=15)
        assert out == [(1, 1), (2, 2)]


class TestRankedFromAttribution:
    def test_intersects_top_k(self):
        # 4x4 grid where indices 0 and 5 are tops in both
        sg = [[10, 0, 0, 0], [0, 9, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        lg = [[8, 0, 0, 0], [0, 7, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        data = {'syc_patch_grid': sg, 'lie_patch_grid': lg}
        out = _ranked_from_attribution(data, top_k=2)
        assert out == [(0, 0), (1, 1)]


class TestRun:
    def test_unknown_mode_rejected(self):
        cfg = FaithfulnessConfig(model='m', mode='__bogus__')
        with pytest.raises(ValueError, match='unknown mode'):
            run(cfg)

    def test_unknown_shared_heads_from_rejected(self):
        cfg = FaithfulnessConfig(model='m', shared_heads_from='__bogus__')
        with pytest.raises(ValueError, match='unknown shared_heads_from'):
            run(cfg)

    def test_returns_curve_one_entry_per_k(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(faithfulness, 'load_triviaqa_pairs', return_value=fake_pairs)
        mocker.patch.object(
            faithfulness,
            'load_results',
            return_value={
                'overlap_by_K': [{'K': 15, 'shared_heads': [[0, 0], [1, 1], [2, 2]]}],
            },
        )

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(faithfulness, 'model_session', fake_session)
        mocker.patch.object(faithfulness, 'save_results')
        mocker.patch.object(
            faithfulness, 'build_sycophancy_prompts', return_value=(['p1', 'p2', 'p3'], ['c1', 'c2', 'c3'])
        )
        # baseline 0.9, ablated 0.1, restored progressively higher
        rates = iter([0.9, 0.1, 0.3, 0.6, 0.9])
        mocker.patch.object(
            faithfulness, 'measure_agreement_per_prompt', side_effect=lambda *a, **kw: (next(rates), [])
        )

        cfg = FaithfulnessConfig(model='gemma-2-2b-it', mode='curve', n_prompts=3, n_pairs=10)
        result = run(cfg)
        assert len(result['curve']) == 3
        assert [row['k'] for row in result['curve']] == [1, 2, 3]

    def test_peak_faithfulness_and_first_k_at_threshold(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(faithfulness, 'load_triviaqa_pairs', return_value=fake_pairs)
        mocker.patch.object(
            faithfulness,
            'load_results',
            return_value={'overlap_by_K': [{'K': 15, 'shared_heads': [[0, 0], [1, 1], [2, 2]]}]},
        )

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(faithfulness, 'model_session', fake_session)
        mocker.patch.object(faithfulness, 'save_results')
        mocker.patch.object(faithfulness, 'build_sycophancy_prompts', return_value=(['p1', 'p2'], ['c1', 'c2']))
        # baseline 1.0, ablated 0.0; per-K rates 0.2, 0.6, 1.0
        # faithfulness = (rate - 0)/(1-0) = rate; threshold 0.8 first hit at k=3
        rates = iter([1.0, 0.0, 0.2, 0.6, 1.0])
        mocker.patch.object(
            faithfulness, 'measure_agreement_per_prompt', side_effect=lambda *a, **kw: (next(rates), [])
        )

        cfg = FaithfulnessConfig(model='gemma-2-2b-it', mode='curve', n_prompts=2, n_pairs=10)
        result = run(cfg)
        assert result['peak_faithfulness_ratio'] == pytest.approx(1.0)
        assert result['first_k_at_threshold'] == 3

    def test_first_k_none_when_threshold_unreached(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(faithfulness, 'load_triviaqa_pairs', return_value=fake_pairs)
        mocker.patch.object(
            faithfulness,
            'load_results',
            return_value={'overlap_by_K': [{'K': 15, 'shared_heads': [[0, 0], [1, 1]]}]},
        )

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(faithfulness, 'model_session', fake_session)
        mocker.patch.object(faithfulness, 'save_results')
        mocker.patch.object(faithfulness, 'build_sycophancy_prompts', return_value=(['p1', 'p2'], ['c1', 'c2']))
        # Restoration peaks at 0.5 — never reaches 0.8 threshold
        rates = iter([1.0, 0.0, 0.3, 0.5])
        mocker.patch.object(
            faithfulness, 'measure_agreement_per_prompt', side_effect=lambda *a, **kw: (next(rates), [])
        )

        cfg = FaithfulnessConfig(model='gemma-2-2b-it', mode='curve', n_prompts=2, n_pairs=10)
        result = run(cfg)
        assert result['first_k_at_threshold'] is None
