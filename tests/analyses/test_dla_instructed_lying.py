"""Tests for the dla-instructed-lying analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import dla_instructed_lying
from shared_circuits.analyses.dla_instructed_lying import (
    DlaInstructedLyingConfig,
    _build_lying_pair,
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


class TestDlaInstructedLyingConfig:
    def test_defaults(self):
        cfg = DlaInstructedLyingConfig()
        assert cfg.paradigm == 'jailbreak'
        assert cfg.n_prompts > 0
        assert cfg.n_pairs == 400
        assert len(cfg.models) >= 1
        assert cfg.top_k == 15

    def test_rejects_zero_prompts(self):
        with pytest.raises(ValidationError):
            DlaInstructedLyingConfig(n_prompts=0)

    def test_rejects_negative_pairs(self):
        with pytest.raises(ValidationError):
            DlaInstructedLyingConfig(n_pairs=-1)

    def test_is_frozen(self):
        cfg = DlaInstructedLyingConfig()
        with pytest.raises(ValidationError):
            cfg.paradigm = 'scaffolded'


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            [
                '--models',
                'm1',
                '--paradigm',
                'scaffolded',
                '--n-prompts',
                '10',
                '--n-pairs',
                '20',
                '--n-devices',
                '2',
                '--batch',
                '4',
                '--top-k',
                '5',
            ]
        )
        assert ns.models == ['m1']
        assert ns.paradigm == 'scaffolded'
        assert ns.n_prompts == 10
        assert ns.n_pairs == 20
        assert ns.n_devices == 2
        assert ns.batch == 4
        assert ns.top_k == 5

    def test_default_paradigm_is_jailbreak(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args([])
        assert ns.paradigm == 'jailbreak'

    def test_paradigm_choice_is_validated(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(['--paradigm', '__invalid__'])


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(
            models=['gemma-2-2b-it'],
            paradigm='repe',
            n_prompts=8,
            n_pairs=20,
            n_devices=1,
            batch=2,
            top_k=15,
        )
        cfg = from_args(ns)
        assert cfg.models == ('gemma-2-2b-it',)
        assert cfg.paradigm == 'repe'
        assert cfg.n_prompts == 8


class TestBuildLyingPair:
    def test_dispatches_jailbreak(self, mocker):
        m = mocker.patch.object(dla_instructed_lying, 'build_instructed_lying_prompts', return_value=(['l'], ['h']))
        out = _build_lying_pair('jailbreak', [('q', 'w', 'c')], 'gemma-2-2b-it')
        assert out == (['l'], ['h'])
        m.assert_called_once()

    def test_dispatches_scaffolded(self, mocker):
        m = mocker.patch.object(dla_instructed_lying, 'build_scaffolded_lying_prompts', return_value=(['l'], ['h']))
        _build_lying_pair('scaffolded', [('q', 'w', 'c')], 'gemma-2-2b-it')
        m.assert_called_once()

    def test_dispatches_repe(self, mocker):
        m = mocker.patch.object(dla_instructed_lying, 'build_repe_lying_prompts', return_value=(['l'], ['h']))
        _build_lying_pair('repe', [('q', 'w', 'c')], 'gemma-2-2b-it')
        m.assert_called_once()


class TestRun:
    def test_unknown_paradigm_rejected(self, mocker):
        mocker.patch.object(dla_instructed_lying, 'load_triviaqa_pairs', return_value=[])
        cfg = DlaInstructedLyingConfig(paradigm='__bogus__')
        with pytest.raises(ValueError, match='unknown paradigm'):
            run(cfg)

    def test_calls_session_per_model_and_returns_dicts(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(dla_instructed_lying, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(dla_instructed_lying, 'model_session', fake_session)
        mocker.patch.object(dla_instructed_lying, 'save_results')

        deltas = {(layer, h): float(layer + h) for layer in range(4) for h in range(4)}
        mocker.patch.object(dla_instructed_lying, 'compute_head_importances', return_value=deltas)
        mocker.patch.object(
            dla_instructed_lying,
            'compute_head_importance_grid',
            side_effect=lambda d, n_layers, n_heads: np.array(
                [[d[(layer, h)] for h in range(n_heads)] for layer in range(n_layers)]
            ),
        )
        mocker.patch.object(
            dla_instructed_lying,
            'rank_heads',
            side_effect=lambda d, top_k: sorted(d.items(), key=lambda kv: -kv[1])[:top_k],
        )

        cfg = DlaInstructedLyingConfig(
            models=('gemma-2-2b-it', 'gemma-2-2b-it'),
            paradigm='jailbreak',
            n_prompts=2,
            n_pairs=400,
        )
        results = run(cfg)
        assert len(results) == 2
        for r in results:
            assert {'model', 'paradigm', 'syc_grid', 'lie_grid', 'syc_top15', 'lie_top15', 'shared_heads'} <= r.keys()
            assert r['paradigm'] == 'jailbreak'
            assert r['k'] == 4
            # 4x4 grid -> total_heads=16 -> ceil(sqrt) = 4

    def test_save_results_uses_paradigm_specific_slug(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(dla_instructed_lying, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(dla_instructed_lying, 'model_session', fake_session)
        save = mocker.patch.object(dla_instructed_lying, 'save_results')

        deltas = {(layer, h): float(layer * 4 + h) for layer in range(4) for h in range(4)}
        mocker.patch.object(dla_instructed_lying, 'compute_head_importances', return_value=deltas)
        mocker.patch.object(
            dla_instructed_lying,
            'compute_head_importance_grid',
            side_effect=lambda d, n_layers, n_heads: np.array(
                [[d[(layer, h)] for h in range(n_heads)] for layer in range(n_layers)]
            ),
        )
        mocker.patch.object(
            dla_instructed_lying,
            'rank_heads',
            side_effect=lambda d, top_k: sorted(d.items(), key=lambda kv: -kv[1])[:top_k],
        )

        run(DlaInstructedLyingConfig(models=('gemma-2-2b-it',), paradigm='scaffolded', n_prompts=2, n_pairs=400))
        save.assert_called_once()
        args, _ = save.call_args
        assert args[1] == 'dla_instructed_lying_scaffolded'

    def test_falls_back_to_set_a_when_pairs_too_few(self, mocker, fake_ctx):
        # Only n_prompts pairs available; legacy script reuses set_a in this case.
        mocker.patch.object(
            dla_instructed_lying,
            'load_triviaqa_pairs',
            return_value=[('q', 'w', 'c')] * 2,
        )

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(dla_instructed_lying, 'model_session', fake_session)
        mocker.patch.object(dla_instructed_lying, 'save_results')
        deltas = {(layer, h): float(layer + h) for layer in range(4) for h in range(4)}
        mocker.patch.object(dla_instructed_lying, 'compute_head_importances', return_value=deltas)
        mocker.patch.object(
            dla_instructed_lying,
            'compute_head_importance_grid',
            side_effect=lambda d, n_layers, n_heads: np.array(
                [[d[(layer, h)] for h in range(n_heads)] for layer in range(n_layers)]
            ),
        )
        mocker.patch.object(
            dla_instructed_lying,
            'rank_heads',
            side_effect=lambda d, top_k: sorted(d.items(), key=lambda kv: -kv[1])[:top_k],
        )
        cfg = DlaInstructedLyingConfig(models=('gemma-2-2b-it',), paradigm='jailbreak', n_prompts=2, n_pairs=2)
        results = run(cfg)
        assert len(results) == 1
