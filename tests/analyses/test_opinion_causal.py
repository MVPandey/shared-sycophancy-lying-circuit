"""Tests for the opinion-causal analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import opinion_causal
from shared_circuits.analyses.opinion_causal import (
    OpinionCausalConfig,
    _load_triple_intersection,
    _paired_ci,
    _random_head_set,
    _zero_heads_hooks,
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


class TestOpinionCausalConfig:
    def test_defaults(self):
        cfg = OpinionCausalConfig()
        assert cfg.mode == 'causal'
        assert cfg.n_opinion == 200
        assert cfg.triple_from == ('circuit_overlap', 'opinion_circuit_transfer')
        assert cfg.triple_k == 15

    def test_rejects_zero_opinion(self):
        with pytest.raises(ValidationError):
            OpinionCausalConfig(n_opinion=0)

    def test_rejects_invalid_layer_frac(self):
        with pytest.raises(ValidationError):
            OpinionCausalConfig(direction_layer_frac=2.0)

    def test_is_frozen(self):
        cfg = OpinionCausalConfig()
        with pytest.raises(ValidationError):
            cfg.mode = 'boundary'


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            [
                '--models',
                'm1',
                'm2',
                '--mode',
                'boundary',
                '--n-opinion',
                '50',
                '--n-factual',
                '40',
                '--n-random-seeds',
                '2',
                '--n-boot',
                '100',
                '--direction-layer-frac',
                '0.6',
                '--batch',
                '2',
                '--seed',
                '7',
                '--triple-from',
                'circuit_overlap',
                '--triple-k',
                '20',
            ]
        )
        assert ns.models == ['m1', 'm2']
        assert ns.mode == 'boundary'
        assert ns.n_opinion == 50
        assert ns.triple_from == 'circuit_overlap'
        assert ns.triple_k == 20

    def test_default_mode(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args([])
        assert ns.mode == 'causal'


class TestFromArgs:
    def test_builds_config_causal(self):
        ns = argparse.Namespace(
            models=['gemma-2-2b-it'],
            mode='causal',
            n_opinion=100,
            n_factual=80,
            n_random_seeds=3,
            n_boot=200,
            direction_layer_frac=0.7,
            batch=2,
            seed=42,
            triple_from='circuit_overlap,opinion_circuit_transfer',
            triple_k=10,
        )
        cfg = from_args(ns)
        assert cfg.mode == 'causal'
        assert cfg.triple_from == ('circuit_overlap', 'opinion_circuit_transfer')

    def test_empty_triple_from_falls_back_to_default(self):
        ns = argparse.Namespace(
            models=['m'],
            mode='causal',
            n_opinion=10,
            n_factual=10,
            n_random_seeds=0,
            n_boot=10,
            direction_layer_frac=0.5,
            batch=1,
            seed=42,
            triple_from='',
            triple_k=15,
        )
        cfg = from_args(ns)
        assert cfg.triple_from == ('circuit_overlap', 'opinion_circuit_transfer')


class TestHelpers:
    def test_random_head_set_size(self):
        out = _random_head_set(n_layers=4, n_heads=4, k=5, seed=42)
        assert len(out) == 5
        assert all(0 <= layer < 4 and 0 <= head < 4 for layer, head in out)

    def test_zero_heads_hooks_one_per_layer(self):
        hooks = _zero_heads_hooks({(0, 1), (0, 2), (3, 0)})
        layers = sorted(int(name.split('.')[1]) for name, _ in hooks)
        assert layers == [0, 3]

    def test_paired_ci_around_zero_for_same_inputs(self):
        base = [0.0, 1.0, 0.0, 1.0]
        treat = [0.0, 1.0, 0.0, 1.0]
        mean, lo, hi = _paired_ci(base, treat, n_boot=100, seed=42)
        assert mean == 0.0
        assert lo <= 0.0 <= hi

    def test_load_triple_intersection_uses_only_circuit_overlap_when_one_source(self, mocker):
        mocker.patch.object(
            opinion_causal,
            'load_results',
            return_value={'overlap_by_K': [{'K': 15, 'shared_heads': [[0, 0], [1, 2]]}]},
        )
        triple = _load_triple_intersection('m', ('circuit_overlap',), 15)
        assert triple == {(0, 0), (1, 2)}

    def test_load_triple_intersection_intersects_with_opinion(self, mocker):
        responses = {
            'circuit_overlap': {'overlap_by_K': [{'K': 15, 'shared_heads': [[0, 0], [1, 2], [3, 1]]}]},
            'opinion_circuit_transfer': {'shared_heads': [[1, 2], [3, 1]]},
        }
        mocker.patch.object(opinion_causal, 'load_results', side_effect=lambda src, _: responses[src])
        triple = _load_triple_intersection('m', ('circuit_overlap', 'opinion_circuit_transfer'), 15)
        assert triple == {(1, 2), (3, 1)}


class TestRun:
    def test_unknown_mode_rejected(self):
        cfg = OpinionCausalConfig(mode='__bogus__')
        with pytest.raises(ValueError, match='unknown mode'):
            run(cfg)

    def test_causal_returns_rate_shift_keys(self, mocker, fake_opinion_pairs, fake_ctx):
        mocker.patch.object(opinion_causal, 'generate_opinion_pairs', return_value=fake_opinion_pairs)
        mocker.patch.object(
            opinion_causal,
            'load_results',
            return_value={'overlap_by_K': [{'K': 15, 'shared_heads': [[0, 0], [1, 1]]}]},
        )

        @contextmanager
        def fake_session(name):
            yield fake_ctx

        mocker.patch.object(opinion_causal, 'model_session', fake_session)
        mocker.patch.object(opinion_causal, 'save_results')
        mocker.patch.object(opinion_causal, 'build_opinion_prompts', return_value=(['op_a'] * 4, ['op_b'] * 4))
        mocker.patch.object(opinion_causal, 'measure_agreement_per_prompt', return_value=(0.5, [0.0, 1.0, 0.0, 1.0]))

        cfg = OpinionCausalConfig(
            models=('gemma-2-2b-it',),
            mode='causal',
            n_opinion=4,
            n_random_seeds=2,
            n_boot=20,
            triple_from=('circuit_overlap',),
        )
        results = run(cfg)
        assert len(results) == 1
        r = results[0]
        assert r['mode'] == 'causal'
        assert {'shared_cis', 'random_summary', 'margin_delta_rate', 'shared_heads'} <= r.keys()
        assert isinstance(r['shared_cis']['delta_rate'], float)

    def test_boundary_returns_direction_cosines(self, mocker, fake_pairs, fake_opinion_pairs, fake_ctx):
        mocker.patch.object(opinion_causal, 'generate_opinion_pairs', return_value=fake_opinion_pairs)
        mocker.patch.object(opinion_causal, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name):
            yield fake_ctx

        mocker.patch.object(opinion_causal, 'model_session', fake_session)
        mocker.patch.object(opinion_causal, 'save_results')
        mocker.patch.object(opinion_causal, 'build_opinion_prompts', return_value=(['op_a'] * 4, ['op_b'] * 4))
        mocker.patch.object(opinion_causal, 'build_sycophancy_prompts', return_value=(['fw'] * 4, ['fc'] * 4))
        mocker.patch.object(opinion_causal, 'build_lying_prompts', return_value=(['lf'] * 4, ['lt'] * 4))

        rng = np.random.RandomState(0)

        def fake_extract_multi(model, prompts, layers, batch_size):
            return {layer: rng.randn(len(prompts), 32) for layer in layers}

        mocker.patch.object(opinion_causal, 'extract_residual_stream_multi', side_effect=fake_extract_multi)
        mocker.patch.object(opinion_causal, 'cosine_similarity', return_value=0.05)

        cfg = OpinionCausalConfig(
            models=('gemma-2-2b-it',),
            mode='boundary',
            n_opinion=4,
            n_factual=8,
            n_boot=10,
        )
        results = run(cfg)
        r = results[0]
        assert r['mode'] == 'boundary'
        assert r['verdict'] in {'SHARED_WITH_OPINION', 'SPECIFIC_TO_FACTUAL', 'INCONCLUSIVE'}
        assert 'by_layer' in r
        assert 'bootstrap_ci_95' in r
        # cosine_similarity stub returns 0.05, well below the 0.1 specific-to-factual threshold
        assert r['verdict'] == 'SPECIFIC_TO_FACTUAL'

    def test_save_results_invoked(self, mocker, fake_opinion_pairs, fake_ctx):
        mocker.patch.object(opinion_causal, 'generate_opinion_pairs', return_value=fake_opinion_pairs)
        mocker.patch.object(
            opinion_causal,
            'load_results',
            return_value={'overlap_by_K': [{'K': 15, 'shared_heads': [[0, 0]]}]},
        )

        @contextmanager
        def fake_session(name):
            yield fake_ctx

        mocker.patch.object(opinion_causal, 'model_session', fake_session)
        save = mocker.patch.object(opinion_causal, 'save_results')
        mocker.patch.object(opinion_causal, 'build_opinion_prompts', return_value=(['op_a'] * 2, ['op_b'] * 2))
        mocker.patch.object(opinion_causal, 'measure_agreement_per_prompt', return_value=(0.5, [0.0, 1.0]))

        cfg = OpinionCausalConfig(
            models=('gemma-2-2b-it',),
            mode='causal',
            n_opinion=2,
            n_random_seeds=0,
            n_boot=10,
            triple_from=('circuit_overlap',),
        )
        run(cfg)
        save.assert_called_once()
