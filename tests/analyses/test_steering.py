"""Tests for the steering dose-response analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import steering
from shared_circuits.analyses.steering import (
    SteeringConfig,
    _parse_alphas,
    add_cli_args,
    from_args,
    run,
)
from shared_circuits.experiment import ExperimentContext
from shared_circuits.models import ModelInfo


@pytest.fixture
def fake_pairs():
    return [(f'q{i}', f'wrong{i}', f'right{i}') for i in range(200)]


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


class TestSteeringConfig:
    def test_requires_model(self):
        with pytest.raises(ValidationError):
            SteeringConfig()

    def test_defaults(self):
        cfg = SteeringConfig(model='gemma-2-2b-it')
        assert cfg.alphas == (0, -25, -50, -100, -200, 25, 50, 100, 200)
        assert cfg.layer_frac == pytest.approx(0.6)
        assert cfg.layer is None
        assert cfg.n_pairs == 200

    def test_rejects_zero_test_prompts(self):
        with pytest.raises(ValidationError):
            SteeringConfig(model='m', test_prompts=0)

    def test_rejects_invalid_layer_frac(self):
        with pytest.raises(ValidationError):
            SteeringConfig(model='m', layer_frac=1.5)

    def test_is_frozen(self):
        cfg = SteeringConfig(model='m')
        with pytest.raises(ValidationError):
            cfg.batch = 99


class TestParseAlphas:
    def test_default_when_none(self):
        assert _parse_alphas(None) == (0, -25, -50, -100, -200, 25, 50, 100, 200)

    def test_parses_csv(self):
        assert _parse_alphas('0,-25,25') == (0, -25, 25)


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
                '--n-pairs',
                '50',
                '--test-prompts',
                '10',
                '--dir-prompts',
                '20',
                '--alphas',
                '0,-50,50',
                '--layer-frac',
                '0.7',
                '--layer',
                '5',
                '--batch',
                '8',
            ]
        )
        assert ns.model == 'm'
        assert ns.alphas == '0,-50,50'
        assert ns.layer == 5
        assert ns.layer_frac == pytest.approx(0.7)

    def test_required_model(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(
            model='gemma-2-2b-it',
            n_devices=1,
            n_pairs=200,
            test_prompts=100,
            dir_prompts=100,
            alphas='0,-100,100',
            layer_frac=0.6,
            layer=None,
            batch=4,
        )
        cfg = from_args(ns)
        assert cfg.alphas == (0, -100, 100)
        assert cfg.layer is None

    def test_uses_default_alphas_when_none(self):
        ns = argparse.Namespace(
            model='m',
            n_devices=1,
            n_pairs=200,
            test_prompts=100,
            dir_prompts=100,
            alphas=None,
            layer_frac=0.6,
            layer=None,
            batch=4,
        )
        cfg = from_args(ns)
        assert cfg.alphas == (0, -25, -50, -100, -200, 25, 50, 100, 200)


class TestRun:
    def test_returns_dose_response(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(steering, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(steering, 'model_session', fake_session)
        mocker.patch.object(steering, 'save_results')
        mocker.patch.object(steering, 'extract_residual_stream', return_value=np.random.RandomState(0).randn(100, 32))
        rates = iter([0.6, 0.7, 0.8, 0.5, 0.3, 0.5, 0.4, 0.3, 0.2])
        mocker.patch.object(steering, 'measure_agreement_rate', side_effect=lambda *a, **kw: next(rates, 0.5))
        # Non-zero alphas go through the manual hook path; cap that with a stub of the inner helper.
        mocker.patch.object(steering, '_measure_steered_rate', side_effect=lambda *a, **kw: 0.42)

        cfg = SteeringConfig(model='gemma-2-2b-it', test_prompts=10, dir_prompts=10, alphas=(0, -25, 25))
        result = run(cfg)
        assert result['model'] == 'gemma-2-2b-it'
        assert result['steer_layer'] == int(4 * 0.6)
        rows = result['dose_response']
        assert len(rows) == 3
        assert {r['alpha'] for r in rows} == {0, -25, 25}
        assert all('rate' in r for r in rows)

    def test_layer_override(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(steering, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(steering, 'model_session', fake_session)
        mocker.patch.object(steering, 'save_results')
        mocker.patch.object(steering, 'extract_residual_stream', return_value=np.zeros((10, 32)))
        mocker.patch.object(steering, '_measure_steered_rate', return_value=0.5)
        mocker.patch.object(steering, 'measure_agreement_rate', return_value=0.5)

        cfg = SteeringConfig(model='gemma-2-2b-it', test_prompts=4, dir_prompts=4, alphas=(0,), layer=2)
        result = run(cfg)
        assert result['steer_layer'] == 2

    def test_save_results_invoked(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(steering, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(steering, 'model_session', fake_session)
        save = mocker.patch.object(steering, 'save_results')
        mocker.patch.object(steering, 'extract_residual_stream', return_value=np.zeros((10, 32)))
        mocker.patch.object(steering, '_measure_steered_rate', return_value=0.0)
        mocker.patch.object(steering, 'measure_agreement_rate', return_value=0.0)

        run(SteeringConfig(model='gemma-2-2b-it', test_prompts=4, dir_prompts=4, alphas=(0,)))
        save.assert_called_once()
