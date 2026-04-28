"""Tests for the write-norm-matched control analysis."""

import argparse
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError

from shared_circuits.analyses import norm_matched
from shared_circuits.analyses.norm_matched import (
    NormMatchedConfig,
    _build_shared,
    _greedy_norm_match,
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


class TestNormMatchedConfig:
    def test_requires_model(self):
        with pytest.raises(ValidationError):
            NormMatchedConfig()

    def test_defaults(self):
        cfg = NormMatchedConfig(model='gemma-2-2b-it')
        assert cfg.batch == 4
        assert cfg.n_prompts == 100
        assert cfg.shared_heads_from == 'circuit_overlap'

    def test_rejects_zero_devices(self):
        with pytest.raises(ValidationError):
            NormMatchedConfig(model='m', n_devices=0)

    def test_is_frozen(self):
        cfg = NormMatchedConfig(model='m')
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
                '--batch',
                '8',
                '--n-prompts',
                '50',
                '--shared-heads-from',
                'breadth',
                '--seed',
                '7',
            ]
        )
        assert ns.model == 'm'
        assert ns.batch == 8
        assert ns.n_prompts == 50
        assert ns.shared_heads_from == 'breadth'

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
            batch=4,
            n_prompts=100,
            shared_heads_from='circuit_overlap',
            seed=42,
        )
        cfg = from_args(ns)
        assert cfg.model == 'gemma-2-2b-it'
        assert cfg.shared_heads_from == 'circuit_overlap'


class TestBuildShared:
    def test_returns_intersection_ranked_by_combined_importance(self):
        sg = np.zeros((4, 4))
        sg[0, 0] = 5
        sg[1, 1] = 4
        sg[2, 2] = 3
        sg[3, 3] = 2
        lg = sg.copy()
        shared = _build_shared(sg, lg)
        assert (0, 0) in shared
        # ranked by combined importance => (0,0) should come first
        assert shared[0] == (0, 0)


class TestGreedyNormMatch:
    def test_picks_unused_closest_norms(self):
        shared = [(0, 0), (1, 1)]
        norms = {
            (0, 0): 1.0,
            (0, 1): 1.05,
            (0, 2): 5.0,
            (0, 3): 5.0,
            (1, 0): 5.0,
            (1, 1): 2.0,
            (1, 2): 2.05,
            (1, 3): 5.0,
            (2, 0): 5.0,
            (2, 1): 5.0,
            (2, 2): 5.0,
            (2, 3): 5.0,
            (3, 0): 5.0,
            (3, 1): 5.0,
            (3, 2): 5.0,
            (3, 3): 5.0,
        }
        matched = _greedy_norm_match(shared, norms, n_layers=4, n_heads=4, seed=42)
        assert len(matched) == 2
        # both matched heads must come from outside `shared`
        assert all(h not in set(shared) for h in matched)
        # no duplicates
        assert len(set(matched)) == 2


class TestVerdict:
    def test_specific_to_shared(self):
        margin = {'rate': 0.1, 'logit_diff': 0.2}
        assert _verdict(margin) == 'SPECIFIC_TO_SHARED'

    def test_partial(self):
        margin = {'rate': 0.1, 'logit_diff': 0.0}
        assert _verdict(margin) == 'PARTIAL_SPECIFICITY'

    def test_norm_confound(self):
        margin = {'rate': 0.0, 'logit_diff': 0.0}
        assert _verdict(margin) == 'NORM_CONFOUND'


class TestRun:
    def test_returns_results_dict(self, mocker, fake_pairs, fake_ctx):
        mocker.patch.object(norm_matched, 'load_triviaqa_pairs', return_value=fake_pairs)
        # Provide a circuit_overlap-style result; use small grid with one obvious shared head.
        sg = np.zeros((4, 4))
        sg[0, 0] = 5.0
        lg = sg.copy()
        mocker.patch.object(
            norm_matched, 'load_results', return_value={'syc_grid': sg.tolist(), 'lie_grid': lg.tolist()}
        )

        @contextmanager
        def fake_session(name, **kwargs):
            yield fake_ctx

        mocker.patch.object(norm_matched, 'model_session', fake_session)
        mocker.patch.object(norm_matched, 'save_results')
        mocker.patch.object(
            norm_matched,
            'build_sycophancy_prompts',
            return_value=(['wrong'] * 50, ['right'] * 50),
        )
        mocker.patch.object(norm_matched, '_measure', return_value={'rate': 0.5, 'logit_diff': 1.0})

        cfg = NormMatchedConfig(model='gemma-2-2b-it', n_prompts=10)
        result = run(cfg)
        assert result['model'] == 'gemma-2-2b-it'
        assert result['verdict'] in {'SPECIFIC_TO_SHARED', 'PARTIAL_SPECIFICITY', 'NORM_CONFOUND'}
        assert 'wo_norms' in result
        assert 'margin_shared_vs_norm_matched' in result
        assert 'shared_heads' in result
        assert 'norm_matched_heads' in result
