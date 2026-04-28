"""Tests for the projection-ablation analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import projection_ablation
from shared_circuits.analyses.projection_ablation import (
    ProjectionAblationConfig,
    _parse_layer_fracs,
    _verdict,
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


class TestProjectionAblationConfig:
    def test_requires_model(self):
        with pytest.raises(ValidationError):
            ProjectionAblationConfig()

    def test_defaults(self):
        cfg = ProjectionAblationConfig(model='gemma-2-2b-it')
        assert cfg.layer is None
        assert cfg.layer_fracs == (0.5, 0.6, 0.7, 0.8)
        assert cfg.batch == 2

    def test_rejects_zero_devices(self):
        with pytest.raises(ValidationError):
            ProjectionAblationConfig(model='m', n_devices=0)

    def test_is_frozen(self):
        cfg = ProjectionAblationConfig(model='m')
        with pytest.raises(ValidationError):
            cfg.batch = 99


class TestParseLayerFracs:
    def test_default_when_none(self):
        assert _parse_layer_fracs(None, qwen3=False) == (0.5, 0.6, 0.7, 0.8)

    def test_qwen3_when_flag(self):
        assert _parse_layer_fracs(None, qwen3=True) == (0.45, 0.55, 0.65, 0.75)

    def test_explicit_overrides_qwen3(self):
        assert _parse_layer_fracs('0.1,0.5,0.9', qwen3=True) == (0.1, 0.5, 0.9)


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            [
                '--model',
                'm',
                '--n-devices',
                '2',
                '--batch',
                '4',
                '--layer',
                '12',
                '--layer-fracs',
                '0.3,0.6',
                '--qwen3-layer-sweep',
                '--n-pairs',
                '50',
                '--dir-prompts',
                '20',
                '--test-prompts',
                '40',
                '--n-boot',
                '100',
                '--seed',
                '7',
            ]
        )
        assert ns.model == 'm'
        assert ns.layer == 12
        assert ns.layer_fracs == '0.3,0.6'
        assert ns.qwen3_layer_sweep is True
        assert ns.n_pairs == 50

    def test_required_model(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestFromArgs:
    def test_builds_config_with_qwen3_flag(self):
        ns = argparse.Namespace(
            model='Qwen/Qwen3-8B',
            n_devices=1,
            batch=2,
            layer=None,
            layer_fracs=None,
            qwen3_layer_sweep=True,
            n_pairs=400,
            dir_prompts=50,
            test_prompts=200,
            n_boot=2000,
            seed=42,
        )
        cfg = from_args(ns)
        assert cfg.layer_fracs == (0.45, 0.55, 0.65, 0.75)

    def test_builds_config_explicit_fracs(self):
        ns = argparse.Namespace(
            model='m',
            n_devices=1,
            batch=2,
            layer=10,
            layer_fracs='0.2,0.4',
            qwen3_layer_sweep=False,
            n_pairs=400,
            dir_prompts=50,
            test_prompts=200,
            n_boot=2000,
            seed=42,
        )
        cfg = from_args(ns)
        assert cfg.layer == 10
        assert cfg.layer_fracs == (0.2, 0.4)


class TestVerdict:
    def test_direction_necessary_when_margin_below_neg_threshold(self):
        # projection delta -0.2, random delta 0 => margin -0.2 < -0.05
        layers = {'5': {'projection': {'syc_delta': -0.2}, 'random_projection': {'syc_delta': 0.0}}}
        assert _verdict(layers) == 'DIRECTION_NECESSARY'

    def test_partial_when_small_negative_margin(self):
        layers = {'5': {'projection': {'syc_delta': -0.02}, 'random_projection': {'syc_delta': 0.0}}}
        assert _verdict(layers) == 'PARTIAL_DIRECTION'

    def test_no_effect_when_margin_non_negative(self):
        # projection_delta = 0.05 > random_delta = 0.0 => margin = +0.05 (no syc reduction)
        layers = {'5': {'projection': {'syc_delta': 0.05}, 'random_projection': {'syc_delta': 0.0}}}
        assert _verdict(layers) == 'NO_DIRECTION_EFFECT'

    def test_incomplete_when_empty(self):
        assert _verdict({}) == 'INCOMPLETE'


class TestRun:
    def test_returns_results_dict(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(projection_ablation, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(projection_ablation, 'model_session', fake_session)
        mocker.patch.object(projection_ablation, 'save_results')
        mocker.patch.object(
            projection_ablation, 'extract_residual_stream', return_value=np.random.RandomState(0).randn(50, 32)
        )
        mocker.patch.object(projection_ablation, 'measure_agreement_per_prompt', return_value=(0.5, [0.5] * 50))

        cfg = ProjectionAblationConfig(model='gemma-2-2b-it', n_boot=10, layer=2)
        result = run(cfg)
        assert result['model'] == 'gemma-2-2b-it'
        assert 'baseline_syc_rate' in result
        assert 'layers' in result
        assert '2' in result['layers']
        assert result['verdict'] in {'DIRECTION_NECESSARY', 'PARTIAL_DIRECTION', 'NO_DIRECTION_EFFECT', 'INCOMPLETE'}

    def test_layer_sweep(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(projection_ablation, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(projection_ablation, 'model_session', fake_session)
        mocker.patch.object(projection_ablation, 'save_results')
        mocker.patch.object(
            projection_ablation, 'extract_residual_stream', return_value=np.random.RandomState(0).randn(50, 32)
        )
        mocker.patch.object(projection_ablation, 'measure_agreement_per_prompt', return_value=(0.5, [0.5] * 50))

        cfg = ProjectionAblationConfig(model='gemma-2-2b-it', n_boot=10, layer_fracs=(0.5, 0.75))
        result = run(cfg)
        # 4 layers; 0.5*4 = 2, 0.75*4 = 3
        assert set(result['layers'].keys()) == {'2', '3'}
