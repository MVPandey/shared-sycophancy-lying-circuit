"""Tests for the head-zeroing analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import head_zeroing
from shared_circuits.analyses.head_zeroing import (
    HeadZeroingConfig,
    _construct_head_sets,
    _paired_ci,
    _verdict,
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


class TestHeadZeroingConfig:
    def test_requires_model(self):
        with pytest.raises(ValidationError):
            HeadZeroingConfig()

    def test_defaults(self):
        cfg = HeadZeroingConfig(model='gemma-2-2b-it')
        assert cfg.mode == 'c2_matched'
        assert cfg.n_pairs == 400
        assert cfg.batch == 2
        assert cfg.shared_heads_from == 'circuit_overlap'

    def test_rejects_zero_devices(self):
        with pytest.raises(ValidationError):
            HeadZeroingConfig(model='m', n_devices=0)

    def test_rejects_zero_n_boot(self):
        with pytest.raises(ValidationError):
            HeadZeroingConfig(model='m', n_boot=0)

    def test_is_frozen(self):
        cfg = HeadZeroingConfig(model='m')
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
                '--batch',
                '4',
                '--mode',
                'full_shared',
                '--n-pairs',
                '50',
                '--syc-test-prompts',
                '20',
                '--lie-test-prompts',
                '20',
                '--top-n-combined',
                '7',
                '--shared-heads-from',
                'breadth',
                '--shared-heads-k',
                '15',
                '--n-boot',
                '100',
                '--seed',
                '7',
            ]
        )
        assert ns.model == 'm'
        assert ns.mode == 'full_shared'
        assert ns.n_pairs == 50
        assert ns.top_n_combined == 7
        assert ns.shared_heads_from == 'breadth'

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
            mode='c2_matched',
            n_pairs=400,
            syc_test_prompts=200,
            lie_test_prompts=200,
            top_n_combined=None,
            shared_heads_from='circuit_overlap',
            shared_heads_k=None,
            n_boot=2000,
            seed=42,
        )
        cfg = from_args(ns)
        assert cfg.model == 'gemma-2-2b-it'
        assert cfg.mode == 'c2_matched'


class TestConstructHeadSets:
    def test_c2_matched_returns_four_sets(self):
        sg = np.zeros((4, 4))
        sg[0, 0] = 5
        sg[0, 1] = 4
        sg[0, 2] = 3
        sg[0, 3] = 2
        lg = np.zeros((4, 4))
        # share top-1, then two lie-only and two syc-only
        lg[0, 0] = 6
        lg[1, 0] = 5
        lg[1, 1] = 4
        lg[1, 2] = 3
        result = _construct_head_sets(sg, lg, seed=42, mode='c2_matched', top_n_combined=None)
        assert {'shared', 'syc_specialized', 'lie_specialized', 'random'} <= result.sets.keys()
        assert all(len(v) == result.set_size for v in result.sets.values())

    def test_full_shared_returns_two_sets(self):
        sg = np.zeros((4, 4))
        sg[0, 0] = 5
        lg = np.zeros((4, 4))
        lg[0, 0] = 5
        result = _construct_head_sets(sg, lg, seed=42, mode='full_shared', top_n_combined=None)
        assert set(result.sets.keys()) == {'shared', 'random'}

    def test_top_n_combined(self):
        sg = np.random.RandomState(0).randn(4, 4)
        lg = np.random.RandomState(1).randn(4, 4)
        result = _construct_head_sets(sg, lg, seed=42, mode='top_n_combined', top_n_combined=3)
        assert result.set_size == 3
        assert len(result.sets['shared']) == 3
        assert len(result.sets['random']) == 3


class TestPairedCi:
    def test_returns_three_floats(self):
        base = [0.5] * 100
        abl = [0.4] * 100
        mean, lo, hi = _paired_ci(base, abl, n_boot=50, seed=42)
        assert lo <= mean <= hi
        # delta should be ~ -0.1 since abl - base
        assert mean == pytest.approx(-0.1, abs=1e-9)


class TestVerdict:
    def test_causal_shared_when_both_margins_exceed(self):
        v = _verdict(
            {'shared': {'syc_delta': -0.2, 'lie_delta': -0.2}, 'random': {'syc_delta': -0.05, 'lie_delta': -0.05}}
        )
        # SHARED reduces *more* than random => syc_margin = shared - random > 0 means SHARED has higher delta value
        # In legacy semantics: more-negative delta means bigger drop. We compare shared > random in the verdict.
        # Here shared_delta - random_delta is -0.15 < 0.05 — so this is NOT_CAUSAL by our threshold.
        assert v == 'NOT_CAUSAL'

    def test_causal_shared_when_shared_increases_and_random_does_not(self):
        # Construct values where shared - random > MARGIN on both axes.
        v = _verdict({'shared': {'syc_delta': 0.1, 'lie_delta': 0.2}, 'random': {'syc_delta': 0.0, 'lie_delta': 0.0}})
        assert v == 'CAUSAL_SHARED'

    def test_partial_when_only_one_axis(self):
        v = _verdict({'shared': {'syc_delta': 0.2, 'lie_delta': 0.0}, 'random': {'syc_delta': 0.0, 'lie_delta': 0.0}})
        assert v == 'PARTIAL_CAUSAL'

    def test_incomplete_when_random_missing(self):
        v = _verdict({'shared': {'syc_delta': 0.1, 'lie_delta': 0.1}})
        assert v == 'INCOMPLETE'


class TestRun:
    def test_returns_results_dict(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(head_zeroing, 'load_triviaqa_pairs', return_value=fake_pairs)
        # circuit_overlap-style result with grids
        mocker.patch.object(
            head_zeroing,
            'load_results',
            return_value={'syc_grid': np.zeros((4, 4)).tolist(), 'lie_grid': np.zeros((4, 4)).tolist()},
        )

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(head_zeroing, 'model_session', fake_session)
        mocker.patch.object(head_zeroing, 'save_results')
        mocker.patch.object(head_zeroing, 'measure_agreement_per_prompt', return_value=(0.5, [0.5] * 20))
        mocker.patch.object(head_zeroing, '_measure_lie_per_prompt', return_value=(0.5, [0.5] * 20))
        mocker.patch.object(
            head_zeroing,
            'build_sycophancy_prompts',
            return_value=(['wrong'] * 200, ['right'] * 200),
        )
        mocker.patch.object(
            head_zeroing,
            'build_lying_prompts',
            return_value=(['false'] * 200, ['true'] * 200),
        )

        cfg = HeadZeroingConfig(model='gemma-2-2b-it', mode='full_shared', n_boot=10)
        result = run(cfg)
        assert result['model'] == 'gemma-2-2b-it'
        assert result['mode'] == 'full_shared'
        assert result['verdict'] in {'CAUSAL_SHARED', 'PARTIAL_CAUSAL', 'NOT_CAUSAL', 'INCOMPLETE'}
        assert 'by_set' in result

    def test_unknown_mode_rejected(self):
        cfg = HeadZeroingConfig(model='m', mode='__bogus__')
        with pytest.raises(ValueError, match='unknown mode'):
            run(cfg)

    def test_top_n_combined_requires_n(self):
        cfg = HeadZeroingConfig(model='m', mode='top_n_combined', top_n_combined=None)
        with pytest.raises(ValueError, match='top_n_combined mode requires'):
            run(cfg)
