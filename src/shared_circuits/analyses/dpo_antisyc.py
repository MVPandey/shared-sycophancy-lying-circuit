"""
Anti-sycophancy DPO LoRA training (or sham control).

Drives a TRL ``DPOTrainer`` over PEFT LoRA on the anti-sycophancy or sham preference
dataset built from TriviaQA. The resulting adapter is saved to ``output_dir`` and
optionally merged back into base weights so downstream TransformerLens analyses can
load the intervened model directly.
"""

import argparse
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.config import RANDOM_SEED
from shared_circuits.data.dpo_preferences import build_antisyc_preferences, build_sham_preferences
from shared_circuits.experiment import save_results

_DEFAULT_LORA_R: Final = 16
_DEFAULT_LORA_ALPHA: Final = 32
_DEFAULT_LORA_DROPOUT: Final = 0.05
_DEFAULT_BETA: Final = 0.1
_DEFAULT_LR: Final = 5e-5
_DEFAULT_EPOCHS: Final = 2
_DEFAULT_BATCH: Final = 2
_DEFAULT_GRAD_ACCUM: Final = 4
_DEFAULT_MAX_SEQ: Final = 256
_DEFAULT_N_TRAIN: Final = 500
_DEFAULT_N_EVAL: Final = 50
_DEFAULT_LOGGING_STEPS: Final = 5
_DEFAULT_TARGET_MODULES: Final[tuple[str, ...]] = ('q_proj', 'v_proj')
_VALID_MODES: Final[tuple[str, ...]] = ('anti', 'sham')


class DpoAntisycConfig(BaseModel):
    """Inputs for the anti-sycophancy DPO LoRA training driver."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(...)
    mode: str = Field(default='anti')
    output_dir: Path = Field(default_factory=lambda: Path('./dpo_runs'))
    merged_dir: Path | None = Field(default=None)
    lora_r: int = Field(default=_DEFAULT_LORA_R, gt=0)
    lora_alpha: int = Field(default=_DEFAULT_LORA_ALPHA, gt=0)
    lora_dropout: float = Field(default=_DEFAULT_LORA_DROPOUT, ge=0)
    target_modules: tuple[str, ...] = Field(default=_DEFAULT_TARGET_MODULES)
    beta: float = Field(default=_DEFAULT_BETA, gt=0)
    learning_rate: float = Field(default=_DEFAULT_LR, gt=0)
    epochs: int = Field(default=_DEFAULT_EPOCHS, gt=0)
    batch_size: int = Field(default=_DEFAULT_BATCH, gt=0)
    grad_accum: int = Field(default=_DEFAULT_GRAD_ACCUM, gt=0)
    max_seq_len: int = Field(default=_DEFAULT_MAX_SEQ, gt=0)
    n_train: int = Field(default=_DEFAULT_N_TRAIN, gt=0)
    n_eval: int = Field(default=_DEFAULT_N_EVAL, gt=0)
    logging_steps: int = Field(default=_DEFAULT_LOGGING_STEPS, gt=0)
    seed: int = Field(default=RANDOM_SEED)
    merge_adapter: bool = Field(default=True)


def run(cfg: DpoAntisycConfig) -> dict:
    """Train a LoRA adapter via TRL DPOTrainer; persist + optionally merge into base."""
    if cfg.mode not in _VALID_MODES:
        raise ValueError(f'unknown mode {cfg.mode!r} (expected one of {_VALID_MODES})')
    return _train(cfg)


def _build_preferences(cfg: DpoAntisycConfig) -> tuple[list[dict], list[dict]]:
    if cfg.mode == 'anti':
        return build_antisyc_preferences(n_train=cfg.n_train, n_eval=cfg.n_eval, seed=cfg.seed)
    return build_sham_preferences(n_train=cfg.n_train, n_eval=cfg.n_eval, seed=cfg.seed)


def _train(cfg: DpoAntisycConfig) -> dict:
    # deferred: torch is heavy, and trl/peft/transformers/datasets aren't needed unless training runs
    import torch  # noqa: PLC0415

    # deferred: datasets pulls pyarrow at module load
    from datasets import Dataset  # noqa: PLC0415

    # deferred: peft drags in transformers + LoRA hooks
    from peft import LoraConfig, PeftModel, get_peft_model  # noqa: PLC0415

    # deferred: transformers init is slow and only required for training
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    # deferred: trl 1.x exposes DPOConfig + DPOTrainer at top level
    from trl import DPOConfig, DPOTrainer  # noqa: PLC0415

    train_pairs, eval_pairs = _build_preferences(cfg)
    train_ds = Dataset.from_list(train_pairs)
    eval_ds = Dataset.from_list(eval_pairs)

    # tokenizer typing: ty sees AutoTokenizer.from_pretrained as Optional; fail fast if it really is None.
    raw_tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if raw_tokenizer is None:
        raise RuntimeError(f'AutoTokenizer.from_pretrained returned None for {cfg.model!r}')
    tokenizer = raw_tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        lora_dropout=cfg.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
    )
    peft_model = get_peft_model(base_model, lora_cfg)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dpo_cfg = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        max_length=cfg.max_seq_len,
        logging_steps=cfg.logging_steps,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        report_to='none',
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        # use_reentrant=False is required when LoRA + gradient_checkpointing are combined
        gradient_checkpointing_kwargs={'use_reentrant': False},
        seed=cfg.seed,
    )

    trainer = DPOTrainer(
        model=peft_model,
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))

    metrics = {k: float(v) for k, v in dict(train_result.metrics).items()} if hasattr(train_result, 'metrics') else {}
    summary: dict = {
        'model': cfg.model,
        'mode': cfg.mode,
        'output_dir': str(output_dir),
        'n_train_pairs': len(train_pairs),
        'n_eval_pairs': len(eval_pairs),
        'lora': {
            'r': cfg.lora_r,
            'alpha': cfg.lora_alpha,
            'dropout': cfg.lora_dropout,
            'target_modules': list(cfg.target_modules),
        },
        'training': {
            'beta': cfg.beta,
            'learning_rate': cfg.learning_rate,
            'epochs': cfg.epochs,
            'batch_size': cfg.batch_size,
            'grad_accum': cfg.grad_accum,
            'effective_batch': cfg.batch_size * cfg.grad_accum,
            'max_seq_len': cfg.max_seq_len,
            'seed': cfg.seed,
        },
        'train_metrics': metrics,
    }

    if cfg.merge_adapter:
        merged_dir = (
            Path(cfg.merged_dir) if cfg.merged_dir is not None else output_dir.parent / f'{output_dir.name}_merged'
        )
        # free the trained PEFT model + trainer before reloading the base on CPU for merge
        del peft_model, trainer, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cpu_base = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16, device_map='cpu')
        peft_reloaded = PeftModel.from_pretrained(cpu_base, str(output_dir))
        merged = peft_reloaded.merge_and_unload()
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        summary['merged_dir'] = str(merged_dir)

    save_results(summary, f'dpo_antisyc_{cfg.mode}', cfg.model)
    return summary


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--model', required=True, help='HF repo of the base model to fine-tune.')
    parser.add_argument(
        '--mode',
        choices=list(_VALID_MODES),
        default='anti',
        help='Preference dataset variant: "anti" trains anti-sycophancy, "sham" is a placebo control.',
    )
    parser.add_argument('--output-dir', type=Path, default=Path('./dpo_runs'))
    parser.add_argument(
        '--merged-dir',
        type=Path,
        default=None,
        help='Where to save the merged base+LoRA model. Defaults to "<output_dir>_merged" alongside output-dir.',
    )
    parser.add_argument('--lora-r', type=int, default=_DEFAULT_LORA_R)
    parser.add_argument('--lora-alpha', type=int, default=_DEFAULT_LORA_ALPHA)
    parser.add_argument('--lora-dropout', type=float, default=_DEFAULT_LORA_DROPOUT)
    parser.add_argument(
        '--target-modules',
        nargs='+',
        default=list(_DEFAULT_TARGET_MODULES),
        help='LoRA target module names (default: q_proj v_proj).',
    )
    parser.add_argument('--beta', type=float, default=_DEFAULT_BETA)
    parser.add_argument('--learning-rate', type=float, default=_DEFAULT_LR)
    parser.add_argument('--epochs', type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=_DEFAULT_BATCH)
    parser.add_argument('--grad-accum', type=int, default=_DEFAULT_GRAD_ACCUM)
    parser.add_argument('--max-seq-len', type=int, default=_DEFAULT_MAX_SEQ)
    parser.add_argument('--n-train', type=int, default=_DEFAULT_N_TRAIN)
    parser.add_argument('--n-eval', type=int, default=_DEFAULT_N_EVAL)
    parser.add_argument('--logging-steps', type=int, default=_DEFAULT_LOGGING_STEPS)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument(
        '--no-merge-adapter',
        dest='merge_adapter',
        action='store_false',
        help='Skip merging the LoRA adapter into the base model after training.',
    )
    parser.set_defaults(merge_adapter=True)


def from_args(args: argparse.Namespace) -> DpoAntisycConfig:
    """Build the validated config from a parsed argparse namespace."""
    return DpoAntisycConfig(
        model=args.model,
        mode=args.mode,
        output_dir=args.output_dir,
        merged_dir=args.merged_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=tuple(args.target_modules),
        beta=args.beta,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        n_train=args.n_train,
        n_eval=args.n_eval,
        logging_steps=args.logging_steps,
        seed=args.seed,
        merge_adapter=args.merge_adapter,
    )
