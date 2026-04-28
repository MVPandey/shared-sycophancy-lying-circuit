"""Tests for the post-hoc bootstrap-CIs tool."""

import argparse
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from shared_circuits.analyses import bootstrap_cis
from shared_circuits.analyses.bootstrap_cis import (
    BootstrapCisConfig,
    _bootstrap_count_ci,
    _collect_count_rates,
    _collect_per_prompt,
    _wilson_ci,
    add_cli_args,
    from_args,
    run,
)


def _write_json(directory: Path, name: str, payload: dict) -> Path:
    path = directory / name
    with path.open('w') as f:
        json.dump(payload, f)
    return path


class TestBootstrapCisConfig:
    def test_defaults(self):
        cfg = BootstrapCisConfig()
        assert cfg.results_glob == '*.json'
        assert cfg.keys == ('syc_per_prompt', 'lie_per_prompt')
        assert cfg.n_boot == 10000
        assert cfg.source == 'all'
        assert cfg.include_count_rates is True

    def test_rejects_zero_n_boot(self):
        with pytest.raises(ValidationError):
            BootstrapCisConfig(n_boot=0)

    def test_is_frozen(self):
        cfg = BootstrapCisConfig()
        with pytest.raises(ValidationError):
            cfg.n_boot = 99


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            [
                '--results-dir',
                '/tmp/results',
                '--results-glob',
                '*_gemma.json',
                '--keys',
                'agree_per_prompt,lie_per_prompt',
                '--n-boot',
                '100',
                '--seed',
                '7',
                '--source',
                'gemma',
                '--no-count-rates',
            ]
        )
        assert ns.results_dir == Path('/tmp/results')
        assert ns.results_glob == '*_gemma.json'
        assert ns.keys == 'agree_per_prompt,lie_per_prompt'
        assert ns.n_boot == 100
        assert ns.source == 'gemma'
        assert ns.include_count_rates is False

    def test_count_rates_default_on(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args([])
        assert ns.include_count_rates is True


class TestFromArgs:
    def test_builds_config(self, tmp_path):
        ns = argparse.Namespace(
            results_dir=tmp_path,
            results_glob='*.json',
            keys='syc_per_prompt,lie_per_prompt',
            n_boot=100,
            seed=42,
            source='all',
            include_count_rates=True,
        )
        cfg = from_args(ns)
        assert cfg.results_dir == tmp_path
        assert cfg.keys == ('syc_per_prompt', 'lie_per_prompt')

    def test_empty_keys_falls_back_to_default(self, tmp_path):
        ns = argparse.Namespace(
            results_dir=tmp_path,
            results_glob='*.json',
            keys='',
            n_boot=100,
            seed=42,
            source='all',
            include_count_rates=True,
        )
        cfg = from_args(ns)
        assert cfg.keys == ('syc_per_prompt', 'lie_per_prompt')


class TestCollectPerPrompt:
    def test_finds_top_level_key(self):
        payload = {'syc_per_prompt': [0.0, 1.0, 1.0]}
        out = _collect_per_prompt(payload, ('syc_per_prompt',))
        assert len(out) == 1
        assert out[0][1] == [0.0, 1.0, 1.0]

    def test_finds_nested_key(self):
        payload = {'inner': {'lie_per_prompt': [1, 0, 1]}}
        out = _collect_per_prompt(payload, ('lie_per_prompt',))
        assert len(out) == 1
        assert out[0][0].endswith('lie_per_prompt')

    def test_skips_nonnumeric_lists(self):
        payload = {'syc_per_prompt': ['yes', 'no']}
        out = _collect_per_prompt(payload, ('syc_per_prompt',))
        assert out == []

    def test_skips_empty_lists(self):
        payload = {'syc_per_prompt': []}
        out = _collect_per_prompt(payload, ('syc_per_prompt',))
        assert out == []

    def test_walks_lists(self):
        payload = {'rows': [{'syc_per_prompt': [0, 1]}, {'syc_per_prompt': [1, 1]}]}
        out = _collect_per_prompt(payload, ('syc_per_prompt',))
        assert len(out) == 2


class TestCollectCountRates:
    def test_finds_agree_total_pair(self):
        payload = {'baseline': {'agree': 5, 'total': 10}}
        out = _collect_count_rates(payload)
        assert out == [('baseline', 5, 10)]

    def test_finds_correct_total_pair(self):
        payload = {'correct': 7, 'total': 10}
        out = _collect_count_rates(payload)
        assert out == [('<root>', 7, 10)]

    def test_skips_zero_total(self):
        payload = {'agree': 0, 'total': 0}
        out = _collect_count_rates(payload)
        assert out == []

    def test_skips_unparseable(self):
        payload = {'agree': 'five', 'total': 10}
        out = _collect_count_rates(payload)
        assert out == []


class TestWilsonCi:
    def test_endpoints_for_zero_total(self):
        lo, hi = _wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_centred_around_observed_rate(self):
        lo, hi = _wilson_ci(50, 100)
        assert lo < 0.5 < hi


class TestBootstrapCountCi:
    def test_endpoints_for_zero_total(self):
        lo, hi = _bootstrap_count_ci(0, 0, 100, 42)
        assert lo == 0.0
        assert hi == 1.0

    def test_brackets_observed_rate(self):
        lo, hi = _bootstrap_count_ci(50, 100, 200, 42)
        assert 0.0 <= lo <= 0.5 <= hi <= 1.0


class TestRun:
    def test_collects_per_prompt_cis(self, tmp_path, mocker):
        _write_json(
            tmp_path,
            'circuit_overlap_gemma.json',
            {'syc_per_prompt': [0.0, 1.0, 0.0, 1.0, 1.0]},
        )
        mocker.patch.object(bootstrap_cis, 'save_results')
        cfg = BootstrapCisConfig(
            results_dir=tmp_path,
            results_glob='*.json',
            keys=('syc_per_prompt',),
            n_boot=20,
            include_count_rates=False,
        )
        result = run(cfg)
        assert result['n_files_scanned'] == 1
        assert 'circuit_overlap_gemma' in result['by_file']
        entries = result['by_file']['circuit_overlap_gemma']['per_prompt']
        assert entries[0]['n'] == 5
        assert entries[0]['mean'] == pytest.approx(0.6)
        lo, hi = entries[0]['bootstrap_ci_95']
        assert 0.0 <= lo <= 0.6 <= hi <= 1.0

    def test_collects_count_rates_when_enabled(self, tmp_path, mocker):
        _write_json(
            tmp_path,
            'breadth_gemma.json',
            {'baseline': {'agree': 60, 'total': 100}},
        )
        mocker.patch.object(bootstrap_cis, 'save_results')
        cfg = BootstrapCisConfig(
            results_dir=tmp_path,
            results_glob='*.json',
            keys=(),
            n_boot=20,
            include_count_rates=True,
        )
        result = run(cfg)
        rows = result['by_file']['breadth_gemma']['count_rates']
        assert rows[0]['n_pos'] == 60
        assert rows[0]['rate'] == pytest.approx(0.6)

    def test_filters_by_glob(self, tmp_path, mocker):
        _write_json(tmp_path, 'breadth_gemma.json', {'syc_per_prompt': [0, 1]})
        _write_json(tmp_path, 'breadth_qwen.json', {'syc_per_prompt': [1, 0]})
        mocker.patch.object(bootstrap_cis, 'save_results')
        cfg = BootstrapCisConfig(
            results_dir=tmp_path,
            results_glob='*_gemma.json',
            keys=('syc_per_prompt',),
            n_boot=10,
            include_count_rates=False,
        )
        result = run(cfg)
        assert result['n_files_scanned'] == 1
        assert 'breadth_gemma' in result['by_file']
        assert 'breadth_qwen' not in result['by_file']

    def test_skips_invalid_json(self, tmp_path, mocker):
        bad = tmp_path / 'bad.json'
        bad.write_text('{not valid json')
        mocker.patch.object(bootstrap_cis, 'save_results')
        cfg = BootstrapCisConfig(results_dir=tmp_path, n_boot=5, include_count_rates=False)
        result = run(cfg)
        assert result['n_files_with_entries'] == 0

    def test_saves_with_source_suffix(self, tmp_path, mocker):
        _write_json(tmp_path, 'breadth_gemma.json', {'syc_per_prompt': [0, 1]})
        save = mocker.patch.object(bootstrap_cis, 'save_results')
        run(
            BootstrapCisConfig(
                results_dir=tmp_path,
                keys=('syc_per_prompt',),
                n_boot=10,
                source='gemma',
                include_count_rates=False,
            )
        )
        save.assert_called_once()
        args, kwargs = save.call_args
        assert args[1] == 'bootstrap_cis_gemma'
        assert args[2] == 'all_models'
        assert kwargs.get('results_dir') == tmp_path

    def test_no_entries_when_no_keys_match(self, tmp_path, mocker):
        _write_json(tmp_path, 'unrelated.json', {'foo': [1, 2, 3]})
        mocker.patch.object(bootstrap_cis, 'save_results')
        cfg = BootstrapCisConfig(
            results_dir=tmp_path,
            keys=('syc_per_prompt',),
            n_boot=5,
            include_count_rates=False,
        )
        result = run(cfg)
        assert result['by_file'] == {}
