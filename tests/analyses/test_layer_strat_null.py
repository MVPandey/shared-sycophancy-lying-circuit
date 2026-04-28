"""Tests for the layer-stratified null analysis."""

import argparse

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import layer_strat_null
from shared_circuits.analyses.layer_strat_null import (
    LayerStratNullConfig,
    _extract_grids,
    _overlap,
    _stratified_p,
    _unstratified_p,
    add_cli_args,
    from_args,
    run,
)


class TestLayerStratNullConfig:
    def test_defaults(self):
        cfg = LayerStratNullConfig()
        assert cfg.n_permutations == 10000
        assert cfg.grids_from == 'breadth'
        assert cfg.k is None
        assert len(cfg.models) >= 1

    def test_rejects_non_positive_permutations(self):
        with pytest.raises(ValidationError):
            LayerStratNullConfig(n_permutations=0)

    def test_is_frozen(self):
        cfg = LayerStratNullConfig()
        with pytest.raises(ValidationError):
            cfg.seed = 99


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            ['--models', 'm1', '--n-permutations', '100', '--seed', '7', '--grids-from', 'circuit_overlap', '--k', '20']
        )
        assert ns.models == ['m1']
        assert ns.n_permutations == 100
        assert ns.grids_from == 'circuit_overlap'
        assert ns.k == 20


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(
            models=['m1', 'm2'],
            n_permutations=100,
            seed=42,
            grids_from='breadth',
            k=None,
        )
        cfg = from_args(ns)
        assert cfg.models == ('m1', 'm2')
        assert cfg.n_permutations == 100
        assert cfg.k is None


class TestExtractGrids:
    def test_top_level(self):
        payload = {'syc_grid': [[1.0, 2.0]], 'lie_grid': [[3.0, 4.0]]}
        sg, _lg = _extract_grids(payload)
        assert sg.shape == (1, 2)

    def test_head_overlap_nested(self):
        payload = {'head_overlap': {'syc_grid': [[1.0]], 'lie_grid': [[2.0]]}}
        sg, _lg = _extract_grids(payload)
        assert sg[0, 0] == pytest.approx(1.0)

    def test_raises_on_missing(self):
        with pytest.raises(KeyError):
            _extract_grids({'unrelated': []})

    def test_raises_on_shape_mismatch(self):
        with pytest.raises(KeyError):
            _extract_grids({'syc_grid': [[1.0, 2.0]], 'lie_grid': [[3.0]]})


class TestHelpers:
    def test_overlap_full(self):
        a = np.array([10.0, 9.0, 8.0, 1.0])
        b = np.array([10.0, 9.0, 8.0, 1.0])
        assert _overlap(a, b, k=3) == 3

    def test_overlap_zero(self):
        a = np.array([10.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 10.0])
        assert _overlap(a, b, k=1) == 0

    def test_unstratified_p_returns_valid_probability(self):
        rng = np.random.RandomState(0)
        sf = rng.rand(16)
        lf = rng.rand(16)
        actual = _overlap(sf, lf, k=4)
        p = _unstratified_p(sf, lf, k=4, actual=actual, rng=np.random.RandomState(0), n_perm=20)
        assert 0.0 < p <= 1.0

    def test_stratified_p_returns_valid_probability(self):
        rng = np.random.RandomState(0)
        sg = rng.rand(4, 4)
        lg = rng.rand(4, 4)
        actual = _overlap(sg.flatten(), lg.flatten(), k=4)
        p = _stratified_p(sg, lg, k=4, actual=actual, rng=np.random.RandomState(0), n_perm=20)
        assert 0.0 < p <= 1.0


class TestRun:
    def test_returns_per_model_results(self, mocker):
        rng = np.random.RandomState(0)
        syc = rng.rand(4, 4)
        lie = rng.rand(4, 4)
        mocker.patch.object(
            layer_strat_null,
            'load_results',
            return_value={'head_overlap': {'syc_grid': syc.tolist(), 'lie_grid': lie.tolist()}},
        )
        mocker.patch.object(layer_strat_null, 'save_results')

        cfg = LayerStratNullConfig(models=('m1',), n_permutations=20)
        result = run(cfg)
        assert len(result['by_model']) == 1
        r = result['by_model'][0]
        assert {
            'actual_overlap',
            'p_hypergeometric',
            'p_permutation_unstratified',
            'p_permutation_layer_stratified',
            'ratio_vs_chance',
        } <= r.keys()

    def test_uses_explicit_k_when_set(self, mocker):
        rng = np.random.RandomState(0)
        syc = rng.rand(4, 4)
        lie = rng.rand(4, 4)
        mocker.patch.object(
            layer_strat_null,
            'load_results',
            return_value={'syc_grid': syc.tolist(), 'lie_grid': lie.tolist()},
        )
        mocker.patch.object(layer_strat_null, 'save_results')

        cfg = LayerStratNullConfig(models=('m1',), n_permutations=10, k=5)
        result = run(cfg)
        assert result['by_model'][0]['k'] == 5

    def test_skips_missing_models(self, mocker):
        mocker.patch.object(layer_strat_null, 'load_results', side_effect=FileNotFoundError)
        mocker.patch.object(layer_strat_null, 'save_results')

        cfg = LayerStratNullConfig(models=('missing',), n_permutations=10)
        result = run(cfg)
        assert result['by_model'] == []

    def test_save_results_invoked(self, mocker):
        mocker.patch.object(layer_strat_null, 'load_results', side_effect=FileNotFoundError)
        save = mocker.patch.object(layer_strat_null, 'save_results')
        run(LayerStratNullConfig(models=('m1',), n_permutations=5))
        save.assert_called_once()
        args, _ = save.call_args
        assert args[1] == 'layer_stratified_null'
        assert args[2] == 'all_models'

    def test_handles_shape_mismatch(self, mocker):
        # _extract_grids raises KeyError on shape mismatch; run() must skip rather than crash.
        mocker.patch.object(
            layer_strat_null,
            'load_results',
            return_value={'syc_grid': [[1.0, 2.0]], 'lie_grid': [[3.0]]},
        )
        mocker.patch.object(layer_strat_null, 'save_results')

        result = run(LayerStratNullConfig(models=('m1',), n_permutations=5))
        assert result['by_model'] == []
