"""Tests for the reverse-projection-ablation analysis."""

import argparse
import json
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import reverse_projection
from shared_circuits.analyses.reverse_projection import (
    ReverseProjectionConfig,
    _load_directions_from_file,
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


class TestReverseProjectionConfig:
    def test_requires_model(self):
        with pytest.raises(ValidationError):
            ReverseProjectionConfig()

    def test_defaults(self):
        cfg = ReverseProjectionConfig(model='gemma-2-2b-it')
        assert cfg.lying_task == 'instructed_lying'
        assert cfg.n_pairs == 50
        assert cfg.batch == 4
        assert cfg.weight_repo is None

    def test_rejects_zero_pairs(self):
        with pytest.raises(ValidationError):
            ReverseProjectionConfig(model='m', n_pairs=0)

    def test_is_frozen(self):
        cfg = ReverseProjectionConfig(model='m')
        with pytest.raises(ValidationError):
            cfg.batch = 99


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
                '20',
                '--batch',
                '8',
                '--lying-task',
                'scaffolded_lying',
                '--weight-repo',
                'org/repo',
                '--tag',
                'dpo',
                '--from-direction-file',
                '/tmp/dirs.json',
            ]
        )
        assert ns.model == 'm'
        assert ns.lying_task == 'scaffolded_lying'
        assert ns.weight_repo == 'org/repo'
        assert ns.tag == 'dpo'
        assert ns.from_direction_file == '/tmp/dirs.json'

    def test_defaults(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(['--model', 'm'])
        assert ns.lying_task == 'instructed_lying'
        assert ns.tag == ''


class TestFromArgs:
    def test_builds_config(self):
        ns = argparse.Namespace(
            model='gemma-2-2b-it',
            n_devices=1,
            n_pairs=50,
            batch=4,
            lying_task='repe_lying',
            weight_repo=None,
            tag='',
            from_direction_file=None,
        )
        cfg = from_args(ns)
        assert cfg.model == 'gemma-2-2b-it'
        assert cfg.lying_task == 'repe_lying'


class TestVerdict:
    def test_coupled_when_both_cross_below_threshold(self):
        cells = {
            'ablate_syc_measure_lie': {'frac_preserved': 0.1},
            'ablate_lie_measure_syc': {'frac_preserved': 0.2},
        }
        assert _verdict(cells) == 'COUPLED'

    def test_partial_when_one_below_threshold(self):
        cells = {
            'ablate_syc_measure_lie': {'frac_preserved': 0.1},
            'ablate_lie_measure_syc': {'frac_preserved': 0.9},
        }
        assert _verdict(cells) == 'PARTIALLY_COUPLED'

    def test_independent_when_both_above_threshold(self):
        cells = {
            'ablate_syc_measure_lie': {'frac_preserved': 0.9},
            'ablate_lie_measure_syc': {'frac_preserved': 0.95},
        }
        assert _verdict(cells) == 'INDEPENDENT'

    def test_incomplete_when_nan(self):
        cells = {
            'ablate_syc_measure_lie': {'frac_preserved': float('nan')},
            'ablate_lie_measure_syc': {'frac_preserved': float('nan')},
        }
        assert _verdict(cells) == 'INCOMPLETE'


class TestLoadDirectionsFromFile:
    def test_loads_d_syc_d_lie_keys(self, tmp_path):
        d_syc = {str(i): np.zeros(32).tolist() for i in range(4)}
        d_lie = {str(i): np.ones(32).tolist() for i in range(4)}
        path = tmp_path / 'dirs.json'
        path.write_text(json.dumps({'d_syc': d_syc, 'd_lie': d_lie}))
        syc, lie = _load_directions_from_file(str(path), n_layers=4, lying_task='instructed_lying')
        assert sorted(syc.keys()) == [0, 1, 2, 3]
        assert syc[0].shape == (32,)
        assert lie[0].shape == (32,)

    def test_raises_when_layer_count_mismatches(self, tmp_path):
        d_syc = {str(i): np.zeros(32).tolist() for i in range(2)}
        d_lie = {str(i): np.ones(32).tolist() for i in range(2)}
        path = tmp_path / 'dirs.json'
        path.write_text(json.dumps({'d_syc': d_syc, 'd_lie': d_lie}))
        with pytest.raises(ValueError, match='expected 4 layers'):
            _load_directions_from_file(str(path), n_layers=4, lying_task='instructed_lying')

    def test_raises_when_keys_missing(self, tmp_path):
        path = tmp_path / 'dirs.json'
        path.write_text(json.dumps({'unrelated': 1}))
        with pytest.raises(ValueError, match='missing'):
            _load_directions_from_file(str(path), n_layers=4, lying_task='instructed_lying')


class TestRun:
    def test_returns_results_dict(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(reverse_projection, 'load_triviaqa_pairs', return_value=fake_pairs)

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(reverse_projection, 'model_session', fake_session)
        mocker.patch.object(reverse_projection, 'save_results')
        mocker.patch.object(
            reverse_projection,
            '_task_prompts',
            return_value=(['corrupt'] * 5, ['clean'] * 5),
        )
        mocker.patch.object(reverse_projection, '_lying_task_tokens', return_value=([1], [2]))
        mocker.patch.object(
            reverse_projection,
            '_compute_direction',
            return_value={l: np.ones(32) for l in range(4)},
        )
        mocker.patch.object(
            reverse_projection,
            '_measure_cell_pair',
            return_value={'baseline_gap': 1.0, 'ablated_gap': 0.5, 'frac_preserved': 0.5},
        )

        cfg = ReverseProjectionConfig(model='gemma-2-2b-it', n_pairs=2)
        result = run(cfg)
        assert result['model'] == 'gemma-2-2b-it'
        assert result['lying_task'] == 'instructed_lying'
        assert 'cells' in result
        assert {
            'ablate_syc_measure_syc',
            'ablate_syc_measure_lie',
            'ablate_lie_measure_lie',
            'ablate_lie_measure_syc',
        } <= result['cells'].keys()
        assert result['verdict'] in {'COUPLED', 'PARTIALLY_COUPLED', 'INDEPENDENT', 'INCOMPLETE'}

    def test_unknown_lying_task_rejected(self):
        cfg = ReverseProjectionConfig(model='m', lying_task='__bogus__')
        with pytest.raises(ValueError, match='unknown lying_task'):
            run(cfg)
