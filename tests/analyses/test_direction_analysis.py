"""Tests for the direction-analysis analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import direction_analysis
from shared_circuits.analyses.direction_analysis import (
    DirectionAnalysisConfig,
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


class TestDirectionAnalysisConfig:
    def test_defaults(self):
        cfg = DirectionAnalysisConfig()
        assert cfg.n_pairs == 400
        assert cfg.n_prompts == 50
        assert len(cfg.models) >= 1

    def test_rejects_non_positive_n_prompts(self):
        with pytest.raises(ValidationError):
            DirectionAnalysisConfig(n_prompts=0)

    def test_rejects_non_positive_n_pairs(self):
        with pytest.raises(ValidationError):
            DirectionAnalysisConfig(n_pairs=-1)

    def test_rejects_non_positive_n_permutations(self):
        with pytest.raises(ValidationError):
            DirectionAnalysisConfig(n_permutations=0)

    def test_is_frozen(self):
        cfg = DirectionAnalysisConfig()
        with pytest.raises(ValidationError):
            cfg.n_prompts = 99


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            ['--models', 'm1', 'm2', '--n-pairs', '40', '--n-prompts', '5', '--n-permutations', '10', '--seed', '7']
        )
        assert ns.models == ['m1', 'm2']
        assert ns.n_pairs == 40
        assert ns.n_prompts == 5
        assert ns.n_permutations == 10
        assert ns.seed == 7

    def test_defaults(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args([])
        assert ns.n_pairs == 400
        assert ns.n_prompts == 50
        assert isinstance(ns.models, list)


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(
            models=['gemma-2-2b-it'],
            n_pairs=20,
            n_prompts=5,
            n_permutations=10,
            seed=7,
        )
        cfg = from_args(ns)
        assert cfg.models == ('gemma-2-2b-it',)
        assert cfg.n_pairs == 20
        assert cfg.n_prompts == 5
        assert cfg.n_permutations == 10
        assert cfg.seed == 7


class TestRun:
    def test_returns_per_model_results(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(direction_analysis, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name):
            yield fake_ctx

        mocker.patch.object(direction_analysis, 'model_session', fake_session)
        mocker.patch.object(direction_analysis, 'save_results')
        rng = np.random.RandomState(0)

        # ``extract_residual_stream_multi`` returns a {layer: (n_prompts, d_model)} dict.
        def fake_extract(model, prompts, layers, **_kw):
            return {layer: rng.randn(len(prompts), 32) for layer in layers}

        mocker.patch.object(direction_analysis, 'extract_residual_stream_multi', side_effect=fake_extract)

        cfg = DirectionAnalysisConfig(models=('gemma-2-2b-it',), n_prompts=10, n_permutations=5)
        results = run(cfg)
        assert len(results) == 1
        r = results[0]
        assert r['model'] == 'gemma-2-2b-it'
        assert r['n_layers'] == 4
        assert 'layer_cosines' in r
        assert 'late_layer_mean_cosine' in r
        assert 'null_95th_percentile' in r
        assert 'margin' in r
        # target_layers covers depth: every other layer + final.
        assert r['target_layers'][-1] == 3

    def test_save_results_called(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(direction_analysis, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name):
            yield fake_ctx

        mocker.patch.object(direction_analysis, 'model_session', fake_session)
        save = mocker.patch.object(direction_analysis, 'save_results')
        rng = np.random.RandomState(0)
        mocker.patch.object(
            direction_analysis,
            'extract_residual_stream_multi',
            side_effect=lambda model, prompts, layers, **_: {layer: rng.randn(len(prompts), 32) for layer in layers},
        )

        run(DirectionAnalysisConfig(models=('gemma-2-2b-it',), n_prompts=10, n_permutations=2))
        save.assert_called_once()

    def test_handles_multiple_models(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(direction_analysis, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name):
            yield fake_ctx

        mocker.patch.object(direction_analysis, 'model_session', fake_session)
        mocker.patch.object(direction_analysis, 'save_results')
        rng = np.random.RandomState(0)
        mocker.patch.object(
            direction_analysis,
            'extract_residual_stream_multi',
            side_effect=lambda model, prompts, layers, **_: {layer: rng.randn(len(prompts), 32) for layer in layers},
        )

        cfg = DirectionAnalysisConfig(models=('m1', 'm2', 'm3'), n_prompts=10, n_permutations=2)
        results = run(cfg)
        assert len(results) == 3
