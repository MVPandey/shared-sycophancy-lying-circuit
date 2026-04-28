"""Tests for the NQ-replication analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import nq_replication
from shared_circuits.analyses.nq_replication import (
    NqReplicationConfig,
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


class TestNqReplicationConfig:
    def test_defaults(self):
        cfg = NqReplicationConfig()
        assert cfg.n_pairs == 200
        assert cfg.dla_prompts == 100
        assert cfg.triviaqa_grids_from == 'breadth'
        assert cfg.permutations == 1000

    def test_rejects_zero_pairs(self):
        with pytest.raises(ValidationError):
            NqReplicationConfig(n_pairs=0)

    def test_is_frozen(self):
        cfg = NqReplicationConfig()
        with pytest.raises(ValidationError):
            cfg.batch = 99


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            [
                '--models',
                'm1',
                '--n-devices',
                '2',
                '--n-pairs',
                '100',
                '--dla-prompts',
                '50',
                '--permutations',
                '20',
                '--batch',
                '2',
                '--seed',
                '7',
                '--triviaqa-grids-from',
                'circuit_overlap',
            ]
        )
        assert ns.models == ['m1']
        assert ns.triviaqa_grids_from == 'circuit_overlap'
        assert ns.permutations == 20

    def test_default_grids_from_breadth(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args([])
        assert ns.triviaqa_grids_from == 'breadth'


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(
            models=['gemma-2-2b-it'],
            n_devices=1,
            n_pairs=100,
            dla_prompts=50,
            permutations=20,
            batch=2,
            seed=7,
            triviaqa_grids_from='breadth',
        )
        cfg = from_args(ns)
        assert cfg.models == ('gemma-2-2b-it',)
        assert cfg.dla_prompts == 50
        assert cfg.seed == 7


class TestRun:
    def test_with_triviaqa_grids_loaded(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(nq_replication, 'load_naturalquestions_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(nq_replication, 'model_session', fake_session)
        mocker.patch.object(nq_replication, 'save_results')
        deltas = {(layer, h): float(layer + h) for layer in range(4) for h in range(4)}
        mocker.patch.object(nq_replication, 'compute_head_importances', return_value=deltas)
        # 4x4 grid keeps total_heads=16 so K=ceil(sqrt(16))=4 per legacy.
        mocker.patch.object(
            nq_replication,
            'compute_head_importance_grid',
            side_effect=lambda d, n_layers, n_heads: np.array(
                [[d[(layer, h)] for h in range(n_heads)] for layer in range(n_layers)]
            ),
        )
        # Pretend a breadth result exists with the same grids on the TriviaQA side.
        mocker.patch.object(
            nq_replication,
            'load_results',
            return_value={
                'head_overlap': {'syc_grid': np.zeros((4, 4)).tolist(), 'lie_grid': np.ones((4, 4)).tolist()}
            },
        )

        cfg = NqReplicationConfig(models=('gemma-2-2b-it',), n_pairs=20, dla_prompts=5, permutations=5)
        results = run(cfg)
        assert len(results) == 1
        r = results[0]
        assert r['dataset'] == 'nq_open'
        assert r['k'] == 4
        assert 'nq_overlap' in r
        assert 'tqa_nq_pearson_syc' in r
        assert r['cross_dataset_triviaqa_vs_nq']['available'] is True
        assert 'syc_pearson' in r['cross_dataset_triviaqa_vs_nq']

    def test_without_triviaqa_grids(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(nq_replication, 'load_naturalquestions_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(nq_replication, 'model_session', fake_session)
        mocker.patch.object(nq_replication, 'save_results')
        deltas = {(layer, h): float(layer + h) for layer in range(4) for h in range(4)}
        mocker.patch.object(nq_replication, 'compute_head_importances', return_value=deltas)
        mocker.patch.object(
            nq_replication,
            'compute_head_importance_grid',
            side_effect=lambda d, n_layers, n_heads: np.array(
                [[d[(layer, h)] for h in range(n_heads)] for layer in range(n_layers)]
            ),
        )
        mocker.patch.object(nq_replication, 'load_results', side_effect=FileNotFoundError)

        cfg = NqReplicationConfig(models=('gemma-2-2b-it',), n_pairs=20, dla_prompts=5, permutations=5)
        results = run(cfg)
        r = results[0]
        assert r['cross_dataset_triviaqa_vs_nq']['available'] is False
        assert r['tqa_nq_pearson_syc'] is None
        assert r['tqa_nq_pearson_lie'] is None

    def test_load_grids_top_level_format(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(nq_replication, 'load_naturalquestions_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(nq_replication, 'model_session', fake_session)
        mocker.patch.object(nq_replication, 'save_results')
        deltas = {(layer, h): float(layer + h) for layer in range(4) for h in range(4)}
        mocker.patch.object(nq_replication, 'compute_head_importances', return_value=deltas)
        mocker.patch.object(
            nq_replication,
            'compute_head_importance_grid',
            side_effect=lambda d, n_layers, n_heads: np.array(
                [[d[(layer, h)] for h in range(n_heads)] for layer in range(n_layers)]
            ),
        )
        # Older legacy outputs put grids at the top level rather than under ``head_overlap``.
        mocker.patch.object(
            nq_replication,
            'load_results',
            return_value={'syc_grid': np.zeros((4, 4)).tolist(), 'lie_grid': np.ones((4, 4)).tolist()},
        )

        cfg = NqReplicationConfig(models=('gemma-2-2b-it',), n_pairs=20, dla_prompts=5, permutations=5)
        results = run(cfg)
        assert results[0]['cross_dataset_triviaqa_vs_nq']['available'] is True
