"""Tests for the circuit-overlap analysis."""

import argparse
from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from shared_circuits.analyses import circuit_overlap
from shared_circuits.analyses.circuit_overlap import (
    CircuitOverlapConfig,
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


class TestCircuitOverlapConfig:
    def test_defaults(self):
        cfg = CircuitOverlapConfig()
        assert cfg.n_prompts > 0
        assert cfg.n_pairs == 400
        assert len(cfg.models) >= 1

    def test_rejects_non_positive_n_prompts(self):
        with pytest.raises(ValidationError):
            CircuitOverlapConfig(n_prompts=0)

    def test_rejects_non_positive_n_pairs(self):
        with pytest.raises(ValidationError):
            CircuitOverlapConfig(n_pairs=-5)

    def test_is_frozen(self):
        cfg = CircuitOverlapConfig()
        with pytest.raises(ValidationError):
            cfg.n_prompts = 99


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(['--n-prompts', '7', '--n-pairs', '40', '--models', 'a', 'b'])
        assert ns.n_prompts == 7
        assert ns.n_pairs == 40
        assert ns.models == ['a', 'b']

    def test_defaults_are_set(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args([])
        assert ns.n_prompts > 0
        assert ns.n_pairs == 400
        assert isinstance(ns.models, list)


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(models=['gemma-2-2b-it'], n_prompts=5, n_pairs=20)
        cfg = from_args(ns)
        assert cfg.models == ('gemma-2-2b-it',)
        assert cfg.n_prompts == 5
        assert cfg.n_pairs == 20


class TestRun:
    def test_calls_session_per_model_and_returns_dicts(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(circuit_overlap, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name):
            yield fake_ctx

        mocker.patch.object(circuit_overlap, 'model_session', fake_session)
        mocker.patch.object(circuit_overlap, 'save_results')

        deltas = {(l, h): float(l + h) for l in range(4) for h in range(4)}
        mocker.patch.object(circuit_overlap, 'compute_head_importances', return_value=deltas)
        mocker.patch.object(
            circuit_overlap, 'rank_heads', side_effect=lambda d, top_k: sorted(d.items(), key=lambda kv: -kv[1])[:top_k]
        )
        mocker.patch.object(circuit_overlap, 'head_overlap_hypergeometric', return_value=0.001)
        mocker.patch.object(circuit_overlap, 'permutation_test_overlap', return_value=0.001)
        mocker.patch.object(
            circuit_overlap,
            'rank_correlation',
            return_value={'spearman_rho': 0.7, 'spearman_p': 0.01, 'pearson_r': 0.6, 'pearson_p': 0.02},
        )

        cfg = CircuitOverlapConfig(models=('gemma-2-2b-it', 'gemma-2-2b-it'), n_prompts=3, n_pairs=400)
        results = run(cfg)
        assert len(results) == 2
        for r in results:
            assert {'model', 'verdict', 'overlap_by_K', 'rank_correlation'} <= r.keys()
            assert r['verdict'] in {'SHARED_CIRCUIT', 'PARTIAL_OVERLAP', 'SEPARATE_CIRCUITS'}
            assert any(o['K'] == 15 for o in r['overlap_by_K'])

    def test_save_results_invoked(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(circuit_overlap, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name):
            yield fake_ctx

        mocker.patch.object(circuit_overlap, 'model_session', fake_session)
        save = mocker.patch.object(circuit_overlap, 'save_results')

        deltas = {(l, h): float(l * 4 + h) for l in range(4) for h in range(4)}
        mocker.patch.object(circuit_overlap, 'compute_head_importances', return_value=deltas)
        mocker.patch.object(
            circuit_overlap, 'rank_heads', side_effect=lambda d, top_k: sorted(d.items(), key=lambda kv: -kv[1])[:top_k]
        )
        mocker.patch.object(circuit_overlap, 'head_overlap_hypergeometric', return_value=0.5)
        mocker.patch.object(circuit_overlap, 'permutation_test_overlap', return_value=0.5)
        mocker.patch.object(
            circuit_overlap,
            'rank_correlation',
            return_value={'spearman_rho': 0.0, 'spearman_p': 0.5, 'pearson_r': 0.0, 'pearson_p': 0.5},
        )

        run(CircuitOverlapConfig(models=('gemma-2-2b-it',), n_prompts=2, n_pairs=400))
        save.assert_called_once()
