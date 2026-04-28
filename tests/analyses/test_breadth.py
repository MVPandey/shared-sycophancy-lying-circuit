"""Tests for the breadth analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import breadth
from shared_circuits.analyses.breadth import (
    BreadthConfig,
    _parse_alphas,
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


class TestBreadthConfig:
    def test_requires_model(self):
        with pytest.raises(ValidationError):
            BreadthConfig()

    def test_defaults(self):
        cfg = BreadthConfig(model='gemma-2-2b-it')
        assert cfg.alphas == (0, -25, -50, -100, -200)
        assert cfg.n_devices == 1
        assert cfg.layer_fracs == (0.5, 0.6, 0.7, 0.8)

    def test_rejects_zero_devices(self):
        with pytest.raises(ValidationError):
            BreadthConfig(model='m', n_devices=0)

    def test_is_frozen(self):
        cfg = BreadthConfig(model='m')
        with pytest.raises(ValidationError):
            cfg.batch = 99


class TestParseAlphas:
    def test_default_when_none(self):
        assert _parse_alphas(None) == (0, -25, -50, -100, -200)

    def test_parses_csv(self):
        assert _parse_alphas('0,-50,-100') == (0, -50, -100)


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(['--model', 'm', '--n-devices', '2', '--alphas', '0,-100', '--batch', '4'])
        assert ns.model == 'm'
        assert ns.n_devices == 2
        assert ns.alphas == '0,-100'
        assert ns.batch == 4

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
            alphas='0,-100,-200',
            n_pairs=400,
            dla_prompts=30,
            baseline_prompts=100,
            dir_prompts=50,
            steer_prompts=50,
            permutations=1000,
            seed=42,
            batch=2,
        )
        cfg = from_args(ns)
        assert cfg.model == 'gemma-2-2b-it'
        assert cfg.alphas == (0, -100, -200)
        assert cfg.n_devices == 2

    def test_uses_default_alphas_when_none(self):
        ns = argparse.Namespace(
            model='m',
            n_devices=1,
            alphas=None,
            n_pairs=400,
            dla_prompts=30,
            baseline_prompts=100,
            dir_prompts=50,
            steer_prompts=50,
            permutations=1000,
            seed=42,
            batch=2,
        )
        cfg = from_args(ns)
        assert cfg.alphas == (0, -25, -50, -100, -200)


class TestRun:
    def test_returns_results_dict(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(breadth, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(breadth, 'model_session', fake_session)
        mocker.patch.object(breadth, 'save_results')
        mocker.patch.object(
            breadth, '_head_overlap', return_value={'syc_grid': [[0.0]], 'lie_grid': [[0.0]], 'stats': {'k': 4}}
        )
        mocker.patch.object(breadth, '_baseline', return_value=0.42)
        mocker.patch.object(
            breadth, '_steering_sweep', return_value={'candidates': [2], 'layers': {'2': {'resid_norm': 1.0}}}
        )

        cfg = BreadthConfig(model='gemma-2-2b-it')
        result = run(cfg)
        assert result['model'] == 'gemma-2-2b-it'
        assert result['head_overlap']['stats']['k'] == 4
        assert result['baseline_sycophancy'] == 0.42
        assert 'steering' in result

    def test_overlap_pvalue_returns_int_and_float(self):
        rng = np.random.RandomState(0)
        sf = rng.randn(16)
        lf = rng.randn(16)
        actual, p = breadth._overlap_pvalue(sf, lf, k=4, n_perm=20, seed=42)
        assert isinstance(actual, int)
        assert 0.0 < p <= 1.0

    def test_overlap_stats_returns_expected_keys(self):
        rng = np.random.RandomState(0)
        sg = rng.randn(4, 4)
        lg = rng.randn(4, 4)
        cfg = BreadthConfig(model='m', permutations=20)
        stats = breadth._overlap_stats(sg, lg, total_heads=16, cfg=cfg)
        assert {'k', 'pearson', 'spearman', 'top_k_overlap', 'p_value', 'overlap_ratio'} <= stats.keys()
