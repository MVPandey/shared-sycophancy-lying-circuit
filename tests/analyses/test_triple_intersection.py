"""Tests for the triple-intersection permutation analysis."""

import argparse

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import triple_intersection
from shared_circuits.analyses.triple_intersection import (
    TripleIntersectionConfig,
    _extract_factual_grids,
    _extract_opinion_grid,
    _triple_intersection,
    add_cli_args,
    from_args,
    run,
)


class TestTripleIntersectionConfig:
    def test_defaults(self):
        cfg = TripleIntersectionConfig()
        assert cfg.n_permutations == 10000
        assert cfg.factual_from == 'breadth'
        assert cfg.opinion_from == 'opinion_circuit_transfer'
        assert len(cfg.models) >= 1

    def test_rejects_non_positive_permutations(self):
        with pytest.raises(ValidationError):
            TripleIntersectionConfig(n_permutations=0)

    def test_is_frozen(self):
        cfg = TripleIntersectionConfig()
        with pytest.raises(ValidationError):
            cfg.n_permutations = 99


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            [
                '--models',
                'm1',
                '--n-permutations',
                '100',
                '--seed',
                '7',
                '--factual-from',
                'circuit_overlap',
                '--opinion-from',
                'opinion_causal',
            ]
        )
        assert ns.models == ['m1']
        assert ns.n_permutations == 100
        assert ns.factual_from == 'circuit_overlap'
        assert ns.opinion_from == 'opinion_causal'


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(
            models=['m1', 'm2'],
            n_permutations=100,
            seed=42,
            factual_from='breadth',
            opinion_from='opinion_circuit_transfer',
        )
        cfg = from_args(ns)
        assert cfg.models == ('m1', 'm2')
        assert cfg.n_permutations == 100


class TestExtractGrids:
    def test_extracts_top_level_grids(self):
        payload = {'syc_grid': [[1.0, 2.0], [3.0, 4.0]], 'lie_grid': [[5.0, 6.0], [7.0, 8.0]]}
        sg, lg = _extract_factual_grids(payload)
        assert sg.shape == (2, 2)
        assert lg.shape == (2, 2)

    def test_extracts_head_overlap_nested_grids(self):
        payload = {'head_overlap': {'syc_grid': [[1.0]], 'lie_grid': [[2.0]]}}
        sg, lg = _extract_factual_grids(payload)
        assert sg[0, 0] == pytest.approx(1.0)
        assert lg[0, 0] == pytest.approx(2.0)

    def test_raises_on_missing_grids(self):
        with pytest.raises(KeyError):
            _extract_factual_grids({'unrelated': []})

    def test_extracts_opinion_grid(self):
        payload = {'opinion_grid': [[0.5, 1.0]]}
        og = _extract_opinion_grid(payload)
        assert og.shape == (1, 2)

    def test_raises_on_missing_opinion_grid(self):
        with pytest.raises(KeyError):
            _extract_opinion_grid({})


class TestTripleIntersectionHelper:
    def test_full_overlap(self):
        op = np.array([10.0, 9.0, 8.0, 1.0])
        syc = np.array([10.0, 9.0, 8.0, 1.0])
        lie = np.array([10.0, 9.0, 8.0, 1.0])
        # top-3 of each is {0, 1, 2} -> intersection size 3
        assert _triple_intersection(op, syc, lie, k=3) == 3

    def test_no_overlap(self):
        op = np.array([10.0, 0.0, 0.0, 0.0])
        syc = np.array([0.0, 10.0, 0.0, 0.0])
        lie = np.array([0.0, 0.0, 10.0, 0.0])
        assert _triple_intersection(op, syc, lie, k=1) == 0


class TestRun:
    def test_returns_per_model_dict(self, mocker):
        rng = np.random.RandomState(0)
        # 4x4 grids; sufficient for a meaningful permutation null.
        syc = rng.rand(4, 4)
        lie = rng.rand(4, 4)
        op = rng.rand(4, 4)

        def fake_load(slug, model_name):
            if slug == 'breadth':
                return {'head_overlap': {'syc_grid': syc.tolist(), 'lie_grid': lie.tolist()}}
            if slug == 'opinion_circuit_transfer':
                return {'opinion_grid': op.tolist()}
            raise FileNotFoundError(slug)

        mocker.patch.object(triple_intersection, 'load_results', side_effect=fake_load)
        mocker.patch.object(triple_intersection, 'save_results')

        cfg = TripleIntersectionConfig(models=('m1',), n_permutations=20)
        result = run(cfg)
        assert 'by_model' in result
        assert 'm1' in result['by_model']
        r = result['by_model']['m1']
        assert {'actual_triple_overlap', 'analytic_chance', 'permutation_p_value', 'k', 'ratio_vs_chance'} <= r.keys()

    def test_skips_missing_models(self, mocker):
        mocker.patch.object(triple_intersection, 'load_results', side_effect=FileNotFoundError)
        mocker.patch.object(triple_intersection, 'save_results')

        cfg = TripleIntersectionConfig(models=('missing-model',), n_permutations=10)
        result = run(cfg)
        assert result['by_model'] == {}

    def test_save_results_called_for_aggregate(self, mocker):
        mocker.patch.object(triple_intersection, 'load_results', side_effect=FileNotFoundError)
        save = mocker.patch.object(triple_intersection, 'save_results')
        run(TripleIntersectionConfig(models=('m1',), n_permutations=5))
        save.assert_called_once()
        args, _ = save.call_args
        # Aggregate file written under "all_models" slug.
        assert args[1] == 'triple_intersection_perm'
        assert args[2] == 'all_models'

    def test_skips_on_shape_mismatch(self, mocker):
        rng = np.random.RandomState(0)

        def fake_load(slug, model_name):
            if slug == 'breadth':
                return {'syc_grid': rng.rand(4, 4).tolist(), 'lie_grid': rng.rand(4, 4).tolist()}
            return {'opinion_grid': rng.rand(2, 2).tolist()}

        mocker.patch.object(triple_intersection, 'load_results', side_effect=fake_load)
        mocker.patch.object(triple_intersection, 'save_results')

        result = run(TripleIntersectionConfig(models=('m1',), n_permutations=5))
        assert result['by_model'] == {}
