"""Tests for the probe-transfer analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import probe_transfer
from shared_circuits.analyses.probe_transfer import (
    ProbeTransferConfig,
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


class TestProbeTransferConfig:
    def test_defaults(self):
        cfg = ProbeTransferConfig()
        assert cfg.n_pairs == 400
        assert cfg.n_prompts == 100
        assert cfg.single_model is None
        assert cfg.weight_repo is None
        assert cfg.tag == ''
        assert cfg.probe_layer is None
        assert cfg.probe_layer_frac == pytest.approx(0.85)
        assert cfg.n_boot == 0

    def test_rejects_non_positive_n_prompts(self):
        with pytest.raises(ValidationError):
            ProbeTransferConfig(n_prompts=0)

    def test_rejects_invalid_layer_frac(self):
        with pytest.raises(ValidationError):
            ProbeTransferConfig(probe_layer_frac=1.5)

    def test_rejects_negative_n_boot(self):
        with pytest.raises(ValidationError):
            ProbeTransferConfig(n_boot=-1)

    def test_is_frozen(self):
        cfg = ProbeTransferConfig()
        with pytest.raises(ValidationError):
            cfg.tag = 'x'


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            [
                '--models',
                'm1',
                '--single-model',
                'mistralai/Mistral-7B-Instruct-v0.1',
                '--weight-repo',
                '/tmp/merged',
                '--tag',
                'antisyc',
                '--n-devices',
                '2',
                '--n-pairs',
                '40',
                '--n-prompts',
                '20',
                '--probe-layer',
                '12',
                '--probe-layer-frac',
                '0.8',
                '--n-boot',
                '50',
                '--seed',
                '7',
            ]
        )
        assert ns.single_model == 'mistralai/Mistral-7B-Instruct-v0.1'
        assert ns.weight_repo == '/tmp/merged'
        assert ns.tag == 'antisyc'
        assert ns.n_devices == 2
        assert ns.probe_layer == 12
        assert ns.n_boot == 50

    def test_defaults_minimal(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args([])
        assert ns.single_model is None
        assert ns.weight_repo is None
        assert ns.tag == ''
        assert ns.n_boot == 0


class TestFromArgs:
    def test_builds_multi_config(self):
        ns = argparse.Namespace(
            models=['gemma-2-2b-it'],
            single_model=None,
            n_devices=1,
            weight_repo=None,
            tag='',
            n_pairs=400,
            n_prompts=100,
            probe_layer=None,
            probe_layer_frac=0.85,
            n_boot=0,
            seed=42,
        )
        cfg = from_args(ns)
        assert cfg.models == ('gemma-2-2b-it',)
        assert cfg.single_model is None

    def test_builds_single_config(self):
        ns = argparse.Namespace(
            models=['gemma-2-2b-it'],
            single_model='m',
            n_devices=2,
            weight_repo='/tmp/w',
            tag='dpo',
            n_pairs=400,
            n_prompts=100,
            probe_layer=15,
            probe_layer_frac=0.85,
            n_boot=100,
            seed=42,
        )
        cfg = from_args(ns)
        assert cfg.single_model == 'm'
        assert cfg.weight_repo == '/tmp/w'
        assert cfg.tag == 'dpo'
        assert cfg.probe_layer == 15
        assert cfg.n_boot == 100


class TestRun:
    def test_multi_model_returns_list(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(probe_transfer, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(probe_transfer, 'model_session', fake_session)
        mocker.patch.object(probe_transfer, 'save_results')
        mocker.patch.object(
            probe_transfer, 'extract_residual_stream', return_value=np.random.RandomState(0).randn(100, 32)
        )
        mocker.patch.object(
            probe_transfer,
            'evaluate_probe_transfer',
            return_value={'train_auroc': 0.9, 'test_auroc': 0.7, 'train_accuracy': 0.85, 'test_accuracy': 0.65},
        )

        cfg = ProbeTransferConfig(models=('m1', 'm2'), n_prompts=10)
        results = run(cfg)
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert {'model', 'probe_layer', 'train_auroc', 'test_auroc'} <= r.keys()

    def test_single_model_returns_dict(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(probe_transfer, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(probe_transfer, 'model_session', fake_session)
        mocker.patch.object(probe_transfer, 'save_results')
        mocker.patch.object(
            probe_transfer, 'extract_residual_stream', return_value=np.random.RandomState(0).randn(100, 32)
        )
        mocker.patch.object(
            probe_transfer,
            'evaluate_probe_transfer',
            return_value={'train_auroc': 0.9, 'test_auroc': 0.7, 'train_accuracy': 0.85, 'test_accuracy': 0.65},
        )

        cfg = ProbeTransferConfig(single_model='gemma-2-2b-it', n_prompts=10)
        result = run(cfg)
        assert isinstance(result, dict)
        assert result['model'] == 'gemma-2-2b-it'
        assert result['weight_repo'] is None
        assert 'syc_to_lie' in result
        assert 'lie_to_syc' in result
        assert result['syc_to_lie_bootstrap'] is None

    def test_single_model_with_bootstrap(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(probe_transfer, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(probe_transfer, 'model_session', fake_session)
        mocker.patch.object(probe_transfer, 'save_results')
        mocker.patch.object(
            probe_transfer, 'extract_residual_stream', return_value=np.random.RandomState(0).randn(50, 32)
        )
        mocker.patch.object(
            probe_transfer,
            'evaluate_probe_transfer',
            return_value={'train_auroc': 0.9, 'test_auroc': 0.7, 'train_accuracy': 0.85, 'test_accuracy': 0.65},
        )
        mocker.patch.object(
            probe_transfer,
            '_bootstrap',
            return_value={'mean': 0.71, 'ci_lo': 0.65, 'ci_hi': 0.78, 'n_boot': 100},
        )

        cfg = ProbeTransferConfig(single_model='m', n_prompts=10, n_boot=100)
        result = run(cfg)
        assert isinstance(result, dict)
        assert result['syc_to_lie_bootstrap'] == {'mean': 0.71, 'ci_lo': 0.65, 'ci_hi': 0.78, 'n_boot': 100}
        assert result['lie_to_syc_bootstrap'] == {'mean': 0.71, 'ci_lo': 0.65, 'ci_hi': 0.78, 'n_boot': 100}

    def test_single_model_with_weight_repo_uses_load_model(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(probe_transfer, 'load_triviaqa_pairs', return_value=fake_pairs)
        load_mock = mocker.patch.object(probe_transfer, 'load_model', return_value=fake_ctx.model)
        mocker.patch.object(probe_transfer, 'cleanup_model')
        mocker.patch.object(
            probe_transfer,
            'get_model_info',
            return_value=fake_ctx.info,
        )
        mocker.patch.object(probe_transfer, 'get_agree_disagree_tokens', return_value=([1, 2, 3], [4, 5, 6]))
        mocker.patch.object(probe_transfer, 'save_results')
        mocker.patch.object(
            probe_transfer, 'extract_residual_stream', return_value=np.random.RandomState(0).randn(50, 32)
        )
        mocker.patch.object(
            probe_transfer,
            'evaluate_probe_transfer',
            return_value={'train_auroc': 0.9, 'test_auroc': 0.7, 'train_accuracy': 0.85, 'test_accuracy': 0.65},
        )

        cfg = ProbeTransferConfig(single_model='m', weight_repo='/tmp/merged', tag='dpo', n_prompts=10)
        result = run(cfg)
        assert isinstance(result, dict)
        # weight_repo path bypasses ``model_session`` and routes through ``load_model``.
        load_mock.assert_called_once_with('m', n_devices=1, weight_repo='/tmp/merged')
        assert result['weight_repo'] == '/tmp/merged'
        assert result['tag'] == 'dpo'

    def test_probe_layer_absolute_overrides_frac(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(probe_transfer, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(probe_transfer, 'model_session', fake_session)
        mocker.patch.object(probe_transfer, 'save_results')
        mocker.patch.object(
            probe_transfer, 'extract_residual_stream', return_value=np.random.RandomState(0).randn(100, 32)
        )
        mocker.patch.object(
            probe_transfer,
            'evaluate_probe_transfer',
            return_value={'train_auroc': 0.9, 'test_auroc': 0.7, 'train_accuracy': 0.85, 'test_accuracy': 0.65},
        )

        cfg = ProbeTransferConfig(single_model='m', probe_layer=2, probe_layer_frac=0.5, n_prompts=10)
        result = run(cfg)
        assert isinstance(result, dict)
        # probe_layer wins over probe_layer_frac when both set.
        assert result['probe_layer'] == 2
