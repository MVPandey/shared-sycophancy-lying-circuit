"""Tests for the shared-set activation-patching analysis."""

import argparse
from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from shared_circuits.analyses import activation_patching
from shared_circuits.analyses.activation_patching import (
    ActivationPatchingConfig,
    _matched_random_heads,
    _paired_ci,
    _paired_sign_p,
    _verdict,
    add_cli_args,
    from_args,
    run,
)
from shared_circuits.experiment import ExperimentContext
from shared_circuits.models import ModelInfo


@pytest.fixture
def fake_pairs():
    return [(f'q{i}', f'wrong{i}', f'right{i}') for i in range(100)]


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


class TestActivationPatchingConfig:
    def test_requires_model(self):
        with pytest.raises(ValidationError):
            ActivationPatchingConfig()

    def test_defaults(self):
        cfg = ActivationPatchingConfig(model='gemma-2-2b-it')
        assert cfg.batch == 1
        assert cfg.n_pairs == 20
        assert cfg.shared_heads_from == 'circuit_overlap'
        assert cfg.shared_heads_k == 15
        assert cfg.n_boot == 2000

    def test_rejects_zero_pairs(self):
        with pytest.raises(ValidationError):
            ActivationPatchingConfig(model='m', n_pairs=0)

    def test_rejects_zero_devices(self):
        with pytest.raises(ValidationError):
            ActivationPatchingConfig(model='m', n_devices=0)

    def test_is_frozen(self):
        cfg = ActivationPatchingConfig(model='m')
        with pytest.raises(ValidationError):
            cfg.batch = 99


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
                '--batch',
                '1',
                '--n-pairs',
                '15',
                '--shared-heads-from',
                'circuit_overlap',
                '--shared-heads-k',
                '20',
                '--n-boot',
                '500',
                '--seed',
                '7',
            ]
        )
        assert ns.model == 'gemma-2-2b-it'
        assert ns.n_pairs == 15
        assert ns.shared_heads_k == 20
        assert ns.n_boot == 500
        assert ns.seed == 7

    def test_required_model(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(
            model='gemma-2-2b-it',
            n_devices=2,
            batch=1,
            n_pairs=15,
            shared_heads_from='circuit_overlap',
            shared_heads_k=20,
            n_boot=500,
            seed=7,
        )
        cfg = from_args(ns)
        assert cfg.model == 'gemma-2-2b-it'
        assert cfg.shared_heads_k == 20
        assert cfg.n_boot == 500


class TestHelpers:
    def test_matched_random_heads_disjoint_count(self):
        shared = [(0, 0), (1, 1), (2, 2)]
        rand = _matched_random_heads(shared, n_layers=4, n_heads=4, seed=42)
        assert len(rand) == len(shared)
        # confirm head/layer indices are within bounds
        for layer, head in rand:
            assert 0 <= layer < 4
            assert 0 <= head < 4

    def test_paired_ci_returns_three_floats(self):
        base = [0.5] * 50
        patched = [0.8] * 50
        mean, lo, hi = _paired_ci(base, patched, n_boot=20, seed=42)
        assert lo <= mean <= hi
        assert mean == pytest.approx(0.3, abs=1e-9)

    def test_paired_ci_empty_returns_zeros(self):
        assert _paired_ci([], [], n_boot=10, seed=0) == (0.0, 0.0, 0.0)

    def test_paired_sign_p_in_unit_interval(self):
        p = _paired_sign_p([0.0] * 20, [1.0] * 20, n_boot=50, seed=1)
        assert 0.0 <= p <= 1.0

    def test_paired_sign_p_empty(self):
        assert _paired_sign_p([], [], n_boot=10, seed=0) == 1.0


class TestVerdict:
    def test_causal_when_margin_and_significant(self):
        assert _verdict(shared_delta=0.3, s_lo=0.1, s_hi=0.5, random_delta=0.05) == 'CAUSAL_SHARED'

    def test_partial_when_only_significant(self):
        # significant CI but no margin over random
        assert _verdict(shared_delta=0.1, s_lo=0.05, s_hi=0.15, random_delta=0.1) == 'PARTIAL_CAUSAL'

    def test_partial_when_only_margin(self):
        # margin over random but CI crosses zero
        assert _verdict(shared_delta=0.3, s_lo=-0.1, s_hi=0.7, random_delta=0.05) == 'PARTIAL_CAUSAL'

    def test_not_causal_otherwise(self):
        assert _verdict(shared_delta=0.0, s_lo=-0.1, s_hi=0.1, random_delta=0.0) == 'NOT_CAUSAL'


class TestLoadSharedHeads:
    def test_pulls_from_top_k_bucket(self, mocker):
        mocker.patch.object(
            activation_patching,
            'load_results',
            return_value={
                'overlap_by_K': [
                    {'K': 5, 'shared_heads': [[0, 0]]},
                    {'K': 15, 'shared_heads': [[1, 2], [3, 0]]},
                ]
            },
        )
        out = activation_patching._load_shared_heads('circuit_overlap', 'gemma-2-2b-it', top_k=15)
        assert out == [(1, 2), (3, 0)]


class TestRun:
    def test_returns_results_dict(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(activation_patching, 'load_triviaqa_pairs', return_value=fake_pairs)
        mocker.patch.object(
            activation_patching,
            'load_results',
            return_value={'overlap_by_K': [{'K': 15, 'shared_heads': [[0, 0], [1, 1]]}]},
        )

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(activation_patching, 'model_session', fake_session)
        mocker.patch.object(activation_patching, 'save_results')
        mocker.patch.object(
            activation_patching,
            '_measure_set_aggregate',
            return_value=([0.5] * 5, [0.7] * 5),
        )

        cfg = ActivationPatchingConfig(model='gemma-2-2b-it', n_pairs=5, n_boot=10)
        result = run(cfg)
        assert result['model'] == 'gemma-2-2b-it'
        assert result['verdict'] in {'CAUSAL_SHARED', 'PARTIAL_CAUSAL', 'NOT_CAUSAL'}
        assert 'shared' in result
        assert 'random' in result
        assert result['n_shared_heads'] == 2
        assert result['n_random_heads'] == 2
        assert 'baseline_logit_diff' in result
        assert 'patched_logit_diff' in result
        assert 'shift' in result
        assert 'p_value' in result

    def test_run_invokes_save_results(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(activation_patching, 'load_triviaqa_pairs', return_value=fake_pairs)
        mocker.patch.object(
            activation_patching,
            'load_results',
            return_value={'overlap_by_K': [{'K': 15, 'shared_heads': [[0, 0]]}]},
        )

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(activation_patching, 'model_session', fake_session)
        save_mock = mocker.patch.object(activation_patching, 'save_results')
        mocker.patch.object(
            activation_patching,
            '_measure_set_aggregate',
            return_value=([0.0] * 3, [0.5] * 3),
        )

        cfg = ActivationPatchingConfig(model='gemma-2-2b-it', n_pairs=3, n_boot=10)
        run(cfg)
        save_mock.assert_called_once()
        args, _ = save_mock.call_args
        assert args[1] == 'activation_patching'
