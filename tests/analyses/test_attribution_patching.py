"""Tests for the attribution-patching analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import attribution_patching
from shared_circuits.analyses.attribution_patching import (
    AttributionPatchingConfig,
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


class TestAttributionPatchingConfig:
    def test_default_models_is_three_subset(self):
        cfg = AttributionPatchingConfig()
        assert cfg.models == (
            'gemma-2-2b-it',
            'Qwen/Qwen2.5-1.5B-Instruct',
            'meta-llama/Llama-3.1-8B-Instruct',
        )

    def test_rejects_zero_pairs(self):
        with pytest.raises(ValidationError):
            AttributionPatchingConfig(n_patch_pairs=0)

    def test_is_frozen(self):
        cfg = AttributionPatchingConfig()
        with pytest.raises(ValidationError):
            cfg.n_pairs = 99


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(['--models', 'm', '--n-pairs', '50', '--n-patch-pairs', '5', '--overlap-k', '10'])
        assert ns.models == ['m']
        assert ns.n_patch_pairs == 5
        assert ns.overlap_k == 10


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(models=['gemma-2-2b-it'], n_pairs=400, n_patch_pairs=10, overlap_k=15)
        cfg = from_args(ns)
        assert cfg.models == ('gemma-2-2b-it',)
        assert cfg.n_patch_pairs == 10
        assert cfg.overlap_k == 15


class TestRun:
    def test_returns_per_model_dicts(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(attribution_patching, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name):
            yield fake_ctx

        mocker.patch.object(attribution_patching, 'model_session', fake_session)
        mocker.patch.object(attribution_patching, 'save_results')
        mocker.patch.object(attribution_patching, 'load_results', return_value={})
        # 4 layers x 4 heads = 16 cells; effects matter for ranking
        rng = np.random.RandomState(0)
        mocker.patch.object(attribution_patching, 'compute_attribution_patching', return_value=rng.randn(4, 4))
        mocker.patch.object(attribution_patching, 'head_overlap_hypergeometric', return_value=0.05)

        cfg = AttributionPatchingConfig(models=('gemma-2-2b-it',), n_patch_pairs=2)
        results = run(cfg)
        assert len(results) == 1
        r = results[0]
        assert {'patch_pearson_r', 'patch_k15_overlap', 'syc_patch_grid', 'lie_patch_grid'} <= r.keys()
        assert r['dla_comparison']['available'] is True

    def test_dla_comparison_when_missing(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(attribution_patching, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name):
            yield fake_ctx

        mocker.patch.object(attribution_patching, 'model_session', fake_session)
        mocker.patch.object(attribution_patching, 'save_results')
        mocker.patch.object(attribution_patching, 'load_results', side_effect=FileNotFoundError)
        mocker.patch.object(attribution_patching, 'compute_attribution_patching', return_value=np.zeros((4, 4)))
        mocker.patch.object(attribution_patching, 'head_overlap_hypergeometric', return_value=0.5)

        cfg = AttributionPatchingConfig(models=('gemma-2-2b-it',), n_patch_pairs=2)
        results = run(cfg)
        assert results[0]['dla_comparison']['available'] is False
