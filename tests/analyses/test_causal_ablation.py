"""Tests for the causal-ablation analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import causal_ablation
from shared_circuits.analyses.causal_ablation import (
    CausalAblationConfig,
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


class TestCausalAblationConfig:
    def test_defaults(self):
        cfg = CausalAblationConfig()
        assert cfg.models[0] == 'gemma-2-2b-it'
        assert cfg.n_pairs == 400
        assert cfg.shared_heads_from == 'circuit_overlap'
        assert cfg.shared_heads_k == 15

    def test_rejects_non_positive_n_pairs(self):
        with pytest.raises(ValidationError):
            CausalAblationConfig(n_pairs=0)

    def test_rejects_invalid_layer_frac(self):
        with pytest.raises(ValidationError):
            CausalAblationConfig(probe_layer_frac=1.5)

    def test_is_frozen(self):
        cfg = CausalAblationConfig()
        with pytest.raises(ValidationError):
            cfg.shared_heads_k = 99


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            [
                '--models',
                'm1',
                'm2',
                '--n-pairs',
                '40',
                '--n-probe-prompts',
                '20',
                '--n-random-heads',
                '3',
                '--probe-layer-frac',
                '0.7',
                '--shared-heads-from',
                'circuit_overlap',
                '--shared-heads-k',
                '20',
            ]
        )
        assert ns.models == ['m1', 'm2']
        assert ns.shared_heads_k == 20
        assert ns.probe_layer_frac == pytest.approx(0.7)


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(
            models=['gemma-2-2b-it'],
            n_pairs=40,
            n_probe_prompts=10,
            n_random_heads=2,
            probe_layer_frac=0.8,
            shared_heads_from='circuit_overlap',
            shared_heads_k=10,
        )
        cfg = from_args(ns)
        assert cfg.models == ('gemma-2-2b-it',)
        assert cfg.n_pairs == 40
        assert cfg.n_random_heads == 2


class TestRun:
    def test_dispatches_to_analyse(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(causal_ablation, 'load_triviaqa_pairs', return_value=fake_pairs)
        mocker.patch.object(
            causal_ablation,
            'load_results',
            return_value={'overlap_by_K': [{'K': 15, 'shared_heads': [[0, 0], [1, 1]]}]},
        )

        @contextmanager
        def fake_session(name):
            yield fake_ctx

        mocker.patch.object(causal_ablation, 'model_session', fake_session)
        mocker.patch.object(causal_ablation, 'save_results')
        mocker.patch.object(
            causal_ablation, 'extract_residual_with_ablation', return_value=np.random.RandomState(0).randn(100, 32)
        )
        mocker.patch.object(
            causal_ablation,
            'train_probe',
            return_value={'auroc': 0.7, 'accuracy': 0.6, 'coefficients': np.zeros(32), 'intercept': 0.0},
        )
        mocker.patch.object(causal_ablation, 'cosine_similarity', return_value=0.5)

        cfg = CausalAblationConfig(models=('gemma-2-2b-it',), n_probe_prompts=10)
        results = run(cfg)
        assert len(results) == 1
        r = results[0]
        assert {'no_ablation', 'ablate_top5_shared', 'ablate_top10_shared', 'ablate_5_random', 'verdict'} <= r.keys()
        assert r['verdict'] in {'CAUSAL_SHARED', 'PARTIAL_CAUSAL', 'NOT_CAUSAL'}

    def test_verdict_causal_shared_when_drops(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(causal_ablation, 'load_triviaqa_pairs', return_value=fake_pairs)
        mocker.patch.object(
            causal_ablation,
            'load_results',
            return_value={'overlap_by_K': [{'K': 15, 'shared_heads': [[0, 0]]}]},
        )

        @contextmanager
        def fake_session(name):
            yield fake_ctx

        mocker.patch.object(causal_ablation, 'model_session', fake_session)
        mocker.patch.object(causal_ablation, 'save_results')
        mocker.patch.object(causal_ablation, 'extract_residual_with_ablation', return_value=np.zeros((100, 32)))

        # baseline auroc 0.95, ablated 0.5 => big drop
        call_count = {'n': 0}

        def fake_probe(*args, **kwargs):
            call_count['n'] += 1
            # first 2 calls are no_ablation (syc+lie), next 2 are top5_shared, etc.
            auroc = 0.95 if call_count['n'] <= 2 else 0.5
            return {'auroc': auroc, 'accuracy': 0.5, 'coefficients': np.zeros(32), 'intercept': 0.0}

        mocker.patch.object(causal_ablation, 'train_probe', side_effect=fake_probe)
        mocker.patch.object(causal_ablation, 'cosine_similarity', return_value=0.5)

        cfg = CausalAblationConfig(models=('gemma-2-2b-it',), n_probe_prompts=10)
        results = run(cfg)
        assert results[0]['verdict'] == 'CAUSAL_SHARED'
