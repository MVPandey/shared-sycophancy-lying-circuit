"""Tests for the path-patching analysis."""

import argparse
import json
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import path_patching
from shared_circuits.analyses.path_patching import (
    PathPatchingConfig,
    _bootstrap_ci,
    _edge_key,
    _edge_receivers,
    _load_shared_heads,
    _sample_non_shared,
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


class TestPathPatchingConfig:
    def test_requires_model(self):
        with pytest.raises(ValidationError):
            PathPatchingConfig()

    def test_defaults(self):
        cfg = PathPatchingConfig(model='gemma-2-2b-it')
        assert cfg.task == 'sycophancy'
        assert cfg.shared_source == 'default'
        assert cfg.include_heads is True
        assert cfg.prefill_shift is False

    def test_rejects_zero_pairs(self):
        with pytest.raises(ValidationError):
            PathPatchingConfig(model='m', n_pairs=0)

    def test_is_frozen(self):
        cfg = PathPatchingConfig(model='m')
        with pytest.raises(ValidationError):
            cfg.batch = 99


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            [
                '--model',
                'gemma-2-2b-it',
                '--n-devices',
                '2',
                '--n-pairs',
                '5',
                '--max-sources',
                '3',
                '--no-head-edges',
                '--batch',
                '8',
                '--non-shared-sources',
                '-1',
                '--task',
                'instructed_lying',
                '--shared-source',
                'instructed',
                '--prefill-shift',
            ]
        )
        assert ns.model == 'gemma-2-2b-it'
        assert ns.no_head_edges is True
        assert ns.task == 'instructed_lying'
        assert ns.shared_source == 'instructed'
        assert ns.prefill_shift is True
        assert ns.non_shared_sources == -1

    def test_default_no_head_edges_false(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(['--model', 'm'])
        assert ns.no_head_edges is False
        assert ns.prefill_shift is False
        assert ns.task == 'sycophancy'


class TestFromArgs:
    def test_builds_config_translates_no_head_edges(self):
        ns = argparse.Namespace(
            model='m',
            n_devices=1,
            n_pairs=30,
            max_sources=None,
            no_head_edges=True,
            batch=4,
            non_shared_sources=None,
            task='sycophancy',
            shared_source='default',
            prefill_shift=False,
            seed=42,
            n_boot=1000,
            max_gen=5,
        )
        cfg = from_args(ns)
        assert cfg.include_heads is False  # --no-head-edges flips include_heads off
        assert cfg.task == 'sycophancy'
        assert cfg.shared_source == 'default'

    def test_unknown_task_rejected_at_run(self, mocker):
        cfg = PathPatchingConfig(model='m', task='__bogus__')
        with pytest.raises(ValueError, match='unknown task'):
            run(cfg)


class TestHelpers:
    def test_edge_key_unembed(self):
        assert _edge_key((1, 2), 'unembed') == '1.2->unembed'

    def test_edge_key_head(self):
        assert _edge_key((1, 2), (3, 4)) == '1.2->3.4'

    def test_edge_receivers_includes_only_later_layers_when_heads(self):
        shared = [(0, 0), (2, 1), (3, 2)]
        recs = _edge_receivers((1, 0), shared, include_heads=True)
        assert 'unembed' in recs
        assert (2, 1) in recs
        assert (3, 2) in recs
        assert (0, 0) not in recs

    def test_edge_receivers_unembed_only_when_disabled(self):
        recs = _edge_receivers((1, 0), [(2, 1)], include_heads=False)
        assert recs == ['unembed']

    def test_sample_non_shared_returns_all_when_none(self):
        shared = [(0, 0)]
        result = _sample_non_shared(shared, n_layers=2, n_heads=2, n_sample=None, seed=42)
        assert (0, 0) not in result
        assert len(result) == 3

    def test_sample_non_shared_uniform_sample(self):
        result = _sample_non_shared([(0, 0)], n_layers=4, n_heads=4, n_sample=5, seed=42)
        assert len(result) == 5
        assert (0, 0) not in result

    def test_bootstrap_ci_returns_three_floats(self):
        mean, lo, hi = _bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=10, seed=42)
        assert lo <= mean <= hi


class TestLoadSharedHeads:
    def test_finds_breadth_grid(self, tmp_path, mocker):
        slug_dir = tmp_path
        mocker.patch.object(path_patching, 'RESULTS_DIR', slug_dir)
        # 4 layers x 4 heads grid.  Identical grids => deterministic shared set.
        sg = np.zeros((4, 4))
        sg[0, 0] = 5.0
        sg[1, 1] = 4.0
        sg[2, 2] = 3.0
        sg[3, 3] = 2.0
        path = slug_dir / 'breadth_gemma_2_2b_it.json'
        path.write_text(json.dumps({'head_overlap': {'syc_grid': sg.tolist(), 'lie_grid': sg.tolist()}}))
        shared, n_layers, n_heads = _load_shared_heads('gemma-2-2b-it', shared_source='default')
        assert n_layers == 4
        assert n_heads == 4
        assert (0, 0) in shared

    def test_raises_when_missing(self, tmp_path, mocker):
        mocker.patch.object(path_patching, 'RESULTS_DIR', tmp_path)
        with pytest.raises(FileNotFoundError):
            _load_shared_heads('no-such-model', shared_source='default')


class TestRun:
    def test_returns_results_dict(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(path_patching, 'load_triviaqa_pairs', return_value=fake_pairs)
        mocker.patch.object(path_patching, '_load_shared_heads', return_value=([(0, 0), (1, 1)], 4, 4))

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(path_patching, 'model_session', fake_session)
        mocker.patch.object(path_patching, 'save_results')
        mocker.patch.object(path_patching, '_task_prompts', return_value=(['wrong'] * 5, ['right'] * 5))
        mocker.patch.object(path_patching, '_task_tokens', return_value=([1, 2], [3, 4]))
        mocker.patch.object(path_patching, '_layer_directions', return_value={l: np.ones(32) for l in range(4)})
        mocker.patch.object(
            path_patching,
            '_prepare_tokens_and_positions',
            return_value=([], [], [], []),
        )
        mocker.patch.object(path_patching, '_process_pair', return_value=(1.0, -1.0))
        mocker.patch.object(path_patching, '_aggregate', return_value={})

        cfg = PathPatchingConfig(model='gemma-2-2b-it', n_pairs=2)
        result = run(cfg)
        assert result['model'] == 'gemma-2-2b-it'
        assert result['task'] == 'sycophancy'
        assert 'shared_heads_ranked' in result
        assert 'edges' in result

    def test_invalid_shared_source_raises(self):
        cfg = PathPatchingConfig(model='m', shared_source='__bogus__')
        with pytest.raises(ValueError, match='unknown shared_source'):
            run(cfg)
