"""Tests for the anti-sycophancy DPO LoRA training analysis."""

import argparse
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

from shared_circuits.analyses import dpo_antisyc
from shared_circuits.analyses.dpo_antisyc import (
    DpoAntisycConfig,
    add_cli_args,
    from_args,
    run,
)


def _install_stub_module(mocker: MockerFixture, name: str, attrs: dict[str, object]) -> ModuleType:
    """Insert a stub module into ``sys.modules`` so deferred imports inside ``_train`` resolve."""
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    mocker.patch.dict(sys.modules, {name: module})
    return module


@pytest.fixture
def stub_heavy_imports(mocker: MockerFixture, tmp_path: Path):
    """Stub torch/datasets/peft/transformers/trl so ``_train`` runs without GPUs or network."""
    fake_torch = SimpleNamespace(
        bfloat16='bf16',
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    )
    _install_stub_module(mocker, 'torch', vars(fake_torch))

    dataset_cls = MagicMock()
    dataset_cls.from_list.side_effect = lambda items: SimpleNamespace(items=items)
    _install_stub_module(mocker, 'datasets', {'Dataset': dataset_cls})

    fake_lora_cfg = MagicMock(name='LoraConfig')
    fake_get_peft = MagicMock(return_value=MagicMock(name='peft_model'))
    fake_peft_model = MagicMock()
    fake_peft_model.from_pretrained = MagicMock(
        return_value=MagicMock(merge_and_unload=MagicMock(return_value=MagicMock(save_pretrained=MagicMock())))
    )
    _install_stub_module(
        mocker,
        'peft',
        {'LoraConfig': fake_lora_cfg, 'PeftModel': fake_peft_model, 'get_peft_model': fake_get_peft},
    )

    fake_tokenizer = MagicMock(pad_token=None, eos_token='</s>', save_pretrained=MagicMock())
    fake_auto_model = MagicMock()
    fake_auto_model.from_pretrained = MagicMock(return_value=MagicMock(name='base_model'))
    fake_auto_tokenizer = MagicMock()
    fake_auto_tokenizer.from_pretrained = MagicMock(return_value=fake_tokenizer)
    _install_stub_module(
        mocker,
        'transformers',
        {'AutoModelForCausalLM': fake_auto_model, 'AutoTokenizer': fake_auto_tokenizer},
    )

    fake_trainer = MagicMock()
    fake_trainer.train = MagicMock(return_value=SimpleNamespace(metrics={'loss': 0.5, 'epoch': 2.0}))
    fake_trainer.save_model = MagicMock()
    fake_dpo_trainer_cls = MagicMock(return_value=fake_trainer)
    fake_dpo_config_cls = MagicMock()
    _install_stub_module(
        mocker,
        'trl',
        {'DPOConfig': fake_dpo_config_cls, 'DPOTrainer': fake_dpo_trainer_cls},
    )

    mocker.patch.object(dpo_antisyc, 'save_results')
    return SimpleNamespace(
        Dataset=dataset_cls,
        LoraConfig=fake_lora_cfg,
        PeftModel=fake_peft_model,
        get_peft_model=fake_get_peft,
        AutoModelForCausalLM=fake_auto_model,
        AutoTokenizer=fake_auto_tokenizer,
        tokenizer=fake_tokenizer,
        DPOConfig=fake_dpo_config_cls,
        DPOTrainer=fake_dpo_trainer_cls,
        trainer=fake_trainer,
    )


class TestDpoAntisycConfig:
    def test_requires_model(self):
        with pytest.raises(ValidationError):
            DpoAntisycConfig()

    def test_defaults(self):
        cfg = DpoAntisycConfig(model='mistralai/Mistral-7B-Instruct-v0.1')
        assert cfg.mode == 'anti'
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == pytest.approx(0.05)
        assert cfg.target_modules == ('q_proj', 'v_proj')
        assert cfg.beta == pytest.approx(0.1)
        assert cfg.learning_rate == pytest.approx(5e-5)
        assert cfg.epochs == 2
        assert cfg.batch_size == 2
        assert cfg.grad_accum == 4
        assert cfg.max_seq_len == 256
        assert cfg.merge_adapter is True

    def test_rejects_zero_lora_r(self):
        with pytest.raises(ValidationError):
            DpoAntisycConfig(model='m', lora_r=0)

    def test_rejects_negative_dropout(self):
        with pytest.raises(ValidationError):
            DpoAntisycConfig(model='m', lora_dropout=-0.1)

    def test_rejects_zero_beta(self):
        with pytest.raises(ValidationError):
            DpoAntisycConfig(model='m', beta=0)

    def test_rejects_zero_epochs(self):
        with pytest.raises(ValidationError):
            DpoAntisycConfig(model='m', epochs=0)

    def test_is_frozen(self):
        cfg = DpoAntisycConfig(model='m')
        with pytest.raises(ValidationError):
            cfg.beta = 0.5


class TestAddCliArgs:
    def test_registers_expected_flags(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(
            [
                '--model',
                'mistralai/Mistral-7B-Instruct-v0.1',
                '--mode',
                'sham',
                '--output-dir',
                '/tmp/dpo_out',
                '--lora-r',
                '8',
                '--lora-alpha',
                '16',
                '--beta',
                '0.05',
                '--epochs',
                '3',
                '--batch-size',
                '4',
                '--n-train',
                '200',
                '--no-merge-adapter',
            ]
        )
        assert ns.model == 'mistralai/Mistral-7B-Instruct-v0.1'
        assert ns.mode == 'sham'
        assert ns.output_dir == Path('/tmp/dpo_out')
        assert ns.lora_r == 8
        assert ns.lora_alpha == 16
        assert ns.beta == pytest.approx(0.05)
        assert ns.epochs == 3
        assert ns.batch_size == 4
        assert ns.n_train == 200
        assert ns.merge_adapter is False

    def test_defaults_set(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(['--model', 'm'])
        assert ns.mode == 'anti'
        assert ns.lora_r == 16
        assert ns.lora_alpha == 32
        assert ns.merge_adapter is True
        assert ns.target_modules == ['q_proj', 'v_proj']

    def test_rejects_unknown_mode(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(['--model', 'm', '--mode', 'gibberish'])

    def test_required_model(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestFromArgs:
    def test_round_trips(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        ns = parser.parse_args(['--model', 'gemma', '--mode', 'sham', '--lora-r', '4'])
        cfg = from_args(ns)
        assert cfg.model == 'gemma'
        assert cfg.mode == 'sham'
        assert cfg.lora_r == 4
        assert cfg.target_modules == ('q_proj', 'v_proj')


class TestRun:
    def test_rejects_unknown_mode(self):
        cfg = DpoAntisycConfig.model_construct(
            model='m',
            mode='nonsense',
            output_dir=Path('./dpo_runs'),
            merged_dir=None,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=('q_proj',),
            beta=0.1,
            learning_rate=5e-5,
            epochs=1,
            batch_size=2,
            grad_accum=4,
            max_seq_len=256,
            n_train=10,
            n_eval=5,
            logging_steps=1,
            seed=42,
            merge_adapter=False,
        )
        with pytest.raises(ValueError, match='unknown mode'):
            run(cfg)

    def test_anti_calls_anti_builder(self, mocker: MockerFixture, stub_heavy_imports, tmp_path: Path):
        anti = mocker.patch.object(
            dpo_antisyc,
            'build_antisyc_preferences',
            return_value=([{'prompt': 'p', 'chosen': 'c', 'rejected': 'r'}], [{'prompt': 'p2'}]),
        )
        sham = mocker.patch.object(dpo_antisyc, 'build_sham_preferences')
        cfg = DpoAntisycConfig(
            model='m',
            mode='anti',
            output_dir=tmp_path / 'out',
            n_train=8,
            n_eval=2,
            merge_adapter=False,
        )
        result = run(cfg)
        anti.assert_called_once_with(n_train=8, n_eval=2, seed=cfg.seed)
        sham.assert_not_called()
        assert result['mode'] == 'anti'
        assert result['n_train_pairs'] == 1
        assert result['n_eval_pairs'] == 1

    def test_sham_calls_sham_builder(self, mocker: MockerFixture, stub_heavy_imports, tmp_path: Path):
        anti = mocker.patch.object(dpo_antisyc, 'build_antisyc_preferences')
        sham = mocker.patch.object(
            dpo_antisyc,
            'build_sham_preferences',
            return_value=(
                [{'prompt': 'p', 'chosen': 'c', 'rejected': 'r'}, {'prompt': 'p2', 'chosen': 'a', 'rejected': 'b'}],
                [],
            ),
        )
        cfg = DpoAntisycConfig(
            model='m',
            mode='sham',
            output_dir=tmp_path / 'out',
            n_train=4,
            n_eval=1,
            seed=99,
            merge_adapter=False,
        )
        result = run(cfg)
        sham.assert_called_once_with(n_train=4, n_eval=1, seed=99)
        anti.assert_not_called()
        assert result['mode'] == 'sham'

    def test_train_invokes_dpo_trainer_and_saves_results(
        self, mocker: MockerFixture, stub_heavy_imports, tmp_path: Path
    ):
        mocker.patch.object(
            dpo_antisyc,
            'build_antisyc_preferences',
            return_value=([{'prompt': 'p', 'chosen': 'c', 'rejected': 'r'}], []),
        )
        save = mocker.patch.object(dpo_antisyc, 'save_results')
        cfg = DpoAntisycConfig(model='m', mode='anti', output_dir=tmp_path / 'out', merge_adapter=False)
        run(cfg)
        stub_heavy_imports.trainer.train.assert_called_once()
        stub_heavy_imports.trainer.save_model.assert_called_once_with(str(tmp_path / 'out'))
        save.assert_called_once()
        # save_results signature: (data, name, model_name)
        call_args = save.call_args.args
        assert call_args[1] == 'dpo_antisyc_anti'
        assert call_args[2] == 'm'

    def test_merge_adapter_runs_when_enabled(self, mocker: MockerFixture, stub_heavy_imports, tmp_path: Path):
        mocker.patch.object(
            dpo_antisyc,
            'build_antisyc_preferences',
            return_value=([{'prompt': 'p', 'chosen': 'c', 'rejected': 'r'}], []),
        )
        cfg = DpoAntisycConfig(
            model='m',
            mode='anti',
            output_dir=tmp_path / 'out',
            merged_dir=tmp_path / 'merged',
            merge_adapter=True,
        )
        result = run(cfg)
        stub_heavy_imports.PeftModel.from_pretrained.assert_called_once()
        assert result['merged_dir'] == str(tmp_path / 'merged')

    def test_skip_merge_when_disabled(self, mocker: MockerFixture, stub_heavy_imports, tmp_path: Path):
        mocker.patch.object(
            dpo_antisyc,
            'build_antisyc_preferences',
            return_value=([{'prompt': 'p', 'chosen': 'c', 'rejected': 'r'}], []),
        )
        cfg = DpoAntisycConfig(model='m', mode='anti', output_dir=tmp_path / 'out', merge_adapter=False)
        result = run(cfg)
        stub_heavy_imports.PeftModel.from_pretrained.assert_not_called()
        assert 'merged_dir' not in result
