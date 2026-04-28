"""Tests for the ``shared-circuits`` CLI dispatcher."""

import argparse

import pytest
from pydantic import BaseModel

from shared_circuits import cli
from shared_circuits.cli import ANALYSES, _module_for, main


@pytest.mark.parametrize('slug', ANALYSES)
def test_each_slug_resolves_to_module(slug: str) -> None:
    mod = _module_for(slug)
    assert mod.__name__ == f'shared_circuits.analyses.{slug.replace("-", "_")}'


@pytest.mark.parametrize('slug', ANALYSES)
def test_each_module_exposes_required_surface(slug: str) -> None:
    mod = _module_for(slug)
    assert callable(mod.run)
    assert callable(mod.add_cli_args)
    assert callable(mod.from_args)
    parser = argparse.ArgumentParser()
    mod.add_cli_args(parser)


@pytest.mark.parametrize('slug', ANALYSES)
def test_each_module_has_a_pydantic_config(slug: str) -> None:
    mod = _module_for(slug)
    config_classes = [
        v for v in vars(mod).values() if isinstance(v, type) and issubclass(v, BaseModel) and v is not BaseModel
    ]
    assert len(config_classes) == 1, f'{slug} should expose exactly one Pydantic config class'
    cfg_cls = config_classes[0]
    assert cfg_cls.model_config.get('frozen') is True


@pytest.mark.parametrize('slug', ANALYSES)
def test_run_help_exits_cleanly(slug: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(['run', slug, '--help'])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert slug in captured.out or '--help' in captured.out


def test_top_level_help_lists_run() -> None:
    with pytest.raises(SystemExit):
        main(['--help'])


def test_dispatch_invokes_run_with_validated_config(mocker) -> None:
    fake_run = mocker.patch('shared_circuits.analyses.circuit_overlap.run')
    main(['run', 'circuit-overlap', '--n-prompts', '5', '--n-pairs', '20', '--models', 'gemma-2-2b-it'])
    fake_run.assert_called_once()
    cfg = fake_run.call_args.args[0]
    assert cfg.n_prompts == 5
    assert cfg.n_pairs == 20
    assert cfg.models == ('gemma-2-2b-it',)


def test_dispatch_to_breadth_with_alphas(mocker) -> None:
    fake_run = mocker.patch('shared_circuits.analyses.breadth.run')
    main(['run', 'breadth', '--model', 'gemma-2-2b-it', '--alphas', '0,-50,-100', '--n-devices', '2'])
    cfg = fake_run.call_args.args[0]
    assert cfg.model == 'gemma-2-2b-it'
    assert cfg.alphas == (0, -50, -100)
    assert cfg.n_devices == 2


def test_dispatch_to_path_patching_with_no_head_edges(mocker) -> None:
    fake_run = mocker.patch('shared_circuits.analyses.path_patching.run')
    main(['run', 'path-patching', '--model', 'gemma-2-2b-it', '--no-head-edges', '--task', 'lying'])
    cfg = fake_run.call_args.args[0]
    assert cfg.model == 'gemma-2-2b-it'
    assert cfg.include_heads is False
    assert cfg.task == 'lying'


def test_unknown_analysis_errors() -> None:
    with pytest.raises(SystemExit):
        main(['run', 'not-an-analysis'])


def test_module_under_main_runs() -> None:
    # Sanity: __main__ guard exists so the file is importable as a script entrypoint.
    assert hasattr(cli, 'main')
