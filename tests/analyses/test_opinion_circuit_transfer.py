"""Tests for the opinion-circuit-transfer analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import opinion_circuit_transfer
from shared_circuits.analyses.opinion_circuit_transfer import (
    OpinionCircuitTransferConfig,
    _load_syc_grid,
    _overlap_pvalue,
    add_cli_args,
    from_args,
    run,
)
from shared_circuits.experiment import ExperimentContext
from shared_circuits.models import ModelInfo


@pytest.fixture
def fake_opinion_pairs():
    return [(f'op_a{i}', f'op_b{i}', f'cat{i}') for i in range(200)]


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


class TestOpinionCircuitTransferConfig:
    def test_defaults(self):
        cfg = OpinionCircuitTransferConfig()
        assert cfg.n_opinion == 200
        assert cfg.dla_prompts == 100
        assert cfg.permutations == 10000
        assert cfg.syc_from == 'circuit_overlap'
        assert len(cfg.models) >= 1

    def test_rejects_zero_opinion(self):
        with pytest.raises(ValidationError):
            OpinionCircuitTransferConfig(n_opinion=0)

    def test_rejects_zero_permutations(self):
        with pytest.raises(ValidationError):
            OpinionCircuitTransferConfig(permutations=0)

    def test_is_frozen(self):
        cfg = OpinionCircuitTransferConfig()
        with pytest.raises(ValidationError):
            cfg.dla_prompts = 99


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            [
                '--models',
                'm1',
                '--n-opinion',
                '50',
                '--dla-prompts',
                '20',
                '--permutations',
                '100',
                '--batch',
                '2',
                '--seed',
                '7',
                '--n-devices',
                '2',
                '--top-k',
                '5',
                '--syc-from',
                'breadth',
            ]
        )
        assert ns.models == ['m1']
        assert ns.n_opinion == 50
        assert ns.dla_prompts == 20
        assert ns.permutations == 100
        assert ns.syc_from == 'breadth'

    def test_default_syc_from_is_circuit_overlap(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args([])
        assert ns.syc_from == 'circuit_overlap'


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(
            models=['gemma-2-2b-it'],
            n_opinion=20,
            dla_prompts=10,
            permutations=5,
            batch=1,
            seed=7,
            n_devices=1,
            top_k=10,
            syc_from='breadth',
        )
        cfg = from_args(ns)
        assert cfg.models == ('gemma-2-2b-it',)
        assert cfg.permutations == 5
        assert cfg.syc_from == 'breadth'


class TestLoadSycGrid:
    def test_returns_none_on_missing_file(self, mocker):
        mocker.patch.object(opinion_circuit_transfer, 'load_results', side_effect=FileNotFoundError)
        assert _load_syc_grid('m', 'breadth', 4, 4) is None

    def test_extracts_top_level_grid(self, mocker):
        grid = np.zeros((4, 4)).tolist()
        mocker.patch.object(
            opinion_circuit_transfer,
            'load_results',
            return_value={'syc_grid': grid, 'lie_grid': grid},
        )
        out = _load_syc_grid('m', 'breadth', 4, 4)
        assert out is not None
        assert out.shape == (4, 4)

    def test_extracts_head_overlap_nested_grid(self, mocker):
        grid = np.ones((4, 4)).tolist()
        mocker.patch.object(
            opinion_circuit_transfer,
            'load_results',
            return_value={'head_overlap': {'syc_grid': grid, 'lie_grid': grid}},
        )
        out = _load_syc_grid('m', 'breadth', 4, 4)
        assert out is not None
        assert out[0, 0] == pytest.approx(1.0)

    def test_reconstructs_grid_from_top15_when_no_grid(self, mocker):
        # circuit_overlap saves only ranked top-K entries; reconstruct a sparse grid for permutation null.
        mocker.patch.object(
            opinion_circuit_transfer,
            'load_results',
            return_value={'syc_top15': [{'layer': 0, 'head': 1, 'delta': 5.0}]},
        )
        out = _load_syc_grid('m', 'circuit_overlap', 4, 4)
        assert out is not None
        assert out[0, 1] == pytest.approx(5.0)
        assert out[0, 0] == pytest.approx(0.0)

    def test_returns_none_when_grid_shape_mismatch(self, mocker):
        # Saved grid shape mismatch must surface as None so the analysis falls through gracefully.
        mocker.patch.object(
            opinion_circuit_transfer,
            'load_results',
            return_value={'syc_grid': np.zeros((2, 2)).tolist()},
        )
        assert _load_syc_grid('m', 'breadth', 4, 4) is None


class TestOverlapPValue:
    def test_full_overlap(self):
        ref = np.array([10.0, 9.0, 8.0, 1.0])
        query = np.array([10.0, 9.0, 8.0, 1.0])
        actual, p = _overlap_pvalue(ref, query, k=3, n_perm=20, seed=0)
        assert actual == 3
        assert 0.0 < p <= 1.0

    def test_no_overlap(self):
        ref = np.array([10.0, 0.0, 0.0, 0.0])
        query = np.array([0.0, 0.0, 0.0, 10.0])
        actual, _ = _overlap_pvalue(ref, query, k=1, n_perm=10, seed=0)
        assert actual == 0


class TestRun:
    def _stub_pipeline(self, mocker, fake_opinion_pairs, fake_ctx, syc_grid_payload=None):
        mocker.patch.object(opinion_circuit_transfer, 'generate_opinion_pairs', return_value=fake_opinion_pairs)
        mocker.patch.object(
            opinion_circuit_transfer,
            'build_opinion_prompts',
            return_value=(['op_a'] * 4, ['op_b'] * 4),
        )

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(opinion_circuit_transfer, 'model_session', fake_session)
        mocker.patch.object(opinion_circuit_transfer, 'save_results')
        deltas = {(layer, h): float(layer + h) for layer in range(4) for h in range(4)}
        mocker.patch.object(opinion_circuit_transfer, 'compute_head_importances', return_value=deltas)
        mocker.patch.object(
            opinion_circuit_transfer,
            'compute_head_importance_grid',
            side_effect=lambda d, n_layers, n_heads: np.array(
                [[d[(layer, h)] for h in range(n_heads)] for layer in range(n_layers)]
            ),
        )
        mocker.patch.object(
            opinion_circuit_transfer,
            'rank_heads',
            side_effect=lambda d, top_k: sorted(d.items(), key=lambda kv: -kv[1])[:top_k],
        )
        if syc_grid_payload is None:
            mocker.patch.object(opinion_circuit_transfer, 'load_results', side_effect=FileNotFoundError)
        else:
            mocker.patch.object(opinion_circuit_transfer, 'load_results', return_value=syc_grid_payload)

    def test_returns_per_model_dict_with_overlap(self, mocker, fake_opinion_pairs, fake_ctx):
        self._stub_pipeline(
            mocker,
            fake_opinion_pairs,
            fake_ctx,
            syc_grid_payload={'syc_grid': np.ones((4, 4)).tolist(), 'lie_grid': np.ones((4, 4)).tolist()},
        )
        cfg = OpinionCircuitTransferConfig(
            models=('gemma-2-2b-it',),
            n_opinion=4,
            dla_prompts=2,
            permutations=5,
        )
        results = run(cfg)
        assert len(results) == 1
        r = results[0]
        assert {'model', 'opinion_grid', 'opinion_top15', 'overlap_with_syc', 'p_perm'} <= r.keys()
        assert r['overlap_with_syc']['available'] is True
        assert r['k'] == 4

    def test_overlap_unavailable_when_syc_missing(self, mocker, fake_opinion_pairs, fake_ctx):
        self._stub_pipeline(mocker, fake_opinion_pairs, fake_ctx, syc_grid_payload=None)
        cfg = OpinionCircuitTransferConfig(
            models=('gemma-2-2b-it',),
            n_opinion=4,
            dla_prompts=2,
            permutations=5,
        )
        results = run(cfg)
        r = results[0]
        assert r['overlap_with_syc']['available'] is False
        assert r['p_perm'] is None

    def test_save_results_called_with_slug(self, mocker, fake_opinion_pairs, fake_ctx):
        self._stub_pipeline(mocker, fake_opinion_pairs, fake_ctx, syc_grid_payload=None)
        save = mocker.patch.object(opinion_circuit_transfer, 'save_results')
        run(
            OpinionCircuitTransferConfig(
                models=('gemma-2-2b-it',),
                n_opinion=2,
                dla_prompts=2,
                permutations=2,
            )
        )
        save.assert_called_once()
        args, _ = save.call_args
        assert args[1] == 'opinion_circuit_transfer'
