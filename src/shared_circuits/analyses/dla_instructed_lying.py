"""DLA head importance for sycophancy vs instructed-lying paradigms (jailbreak / scaffolded / RepE)."""

import argparse
import math
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.attribution import compute_head_importance_grid, compute_head_importances, rank_heads
from shared_circuits.config import DEFAULT_BATCH_SIZE, DEFAULT_N_PROMPTS, DEFAULT_TOP_K
from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, model_session, save_results
from shared_circuits.prompts import (
    build_instructed_lying_prompts,
    build_repe_lying_prompts,
    build_scaffolded_lying_prompts,
    build_sycophancy_prompts,
)

_DEFAULT_MODELS: Final[tuple[str, ...]] = (
    'gemma-2-2b-it',
    'google/gemma-2-9b-it',
    'Qwen/Qwen3-8B',
    'mistralai/Mistral-7B-Instruct-v0.1',
    'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'microsoft/phi-4',
    'meta-llama/Llama-3.3-70B-Instruct',
)
_PARADIGMS: Final[tuple[str, ...]] = ('jailbreak', 'scaffolded', 'repe')
# Slug suffix per paradigm, mirroring the legacy script's tag scheme.
_PARADIGM_TAGS: Final[dict[str, str]] = {
    'jailbreak': 'dla_instructed_lying_jailbreak',
    'scaffolded': 'dla_instructed_lying_scaffolded',
    'repe': 'dla_instructed_lying_repe',
}


class DlaInstructedLyingConfig(BaseModel):
    """Inputs for DLA head ranking under sycophancy vs instructed-lying paradigms."""

    model_config = ConfigDict(frozen=True)

    models: tuple[str, ...] = Field(default_factory=lambda: _DEFAULT_MODELS)
    paradigm: str = Field(default='jailbreak')
    n_prompts: int = Field(default=DEFAULT_N_PROMPTS, gt=0)
    n_pairs: int = Field(default=400, gt=0)
    n_devices: int = Field(default=1, gt=0)
    batch: int = Field(default=DEFAULT_BATCH_SIZE, gt=0)
    top_k: int = Field(default=DEFAULT_TOP_K, gt=0)


def run(cfg: DlaInstructedLyingConfig) -> list[dict]:
    """Run DLA instructed-lying head ranking for every model in ``cfg.models``."""
    if cfg.paradigm not in _PARADIGMS:
        raise ValueError(f'unknown paradigm {cfg.paradigm}')
    pairs = load_triviaqa_pairs(cfg.n_pairs)
    return [_run_one(name, pairs, cfg) for name in cfg.models]


def _run_one(model_name: str, pairs: list[tuple[str, str, str]], cfg: DlaInstructedLyingConfig) -> dict:
    with model_session(model_name, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, pairs, cfg)


def _build_lying_pair(
    paradigm: str,
    pairs: list[tuple[str, str, str]],
    model_name: str,
) -> tuple[list[str], list[str]]:
    """Dispatch to the matching prompt builder; returns (lying_prompts, honest_prompts)."""
    if paradigm == 'scaffolded':
        return build_scaffolded_lying_prompts(pairs, model_name)
    if paradigm == 'repe':
        return build_repe_lying_prompts(pairs, model_name)
    return build_instructed_lying_prompts(pairs, model_name)


def _analyse(ctx: ExperimentContext, pairs: list[tuple[str, str, str]], cfg: DlaInstructedLyingConfig) -> dict:
    n_for_each = max(cfg.n_prompts, 1)
    set_a = pairs[:n_for_each]
    # When only one set is available the legacy script reuses set_a; preserve that fallback.
    set_b = pairs[n_for_each : 2 * n_for_each] if len(pairs) >= 2 * n_for_each else set_a

    wrong, correct = build_sycophancy_prompts(set_a, ctx.model_name)
    lying, honest = _build_lying_pair(cfg.paradigm, set_b, ctx.model_name)

    syc_deltas = compute_head_importances(ctx.model, wrong, correct, n_prompts=cfg.n_prompts, batch_size=cfg.batch)
    lie_deltas = compute_head_importances(ctx.model, lying, honest, n_prompts=cfg.n_prompts, batch_size=cfg.batch)
    syc_grid = compute_head_importance_grid(syc_deltas, ctx.info.n_layers, ctx.info.n_heads)
    lie_grid = compute_head_importance_grid(lie_deltas, ctx.info.n_layers, ctx.info.n_heads)

    syc_ranked = rank_heads(syc_deltas, top_k=ctx.info.total_heads)
    lie_ranked = rank_heads(lie_deltas, top_k=ctx.info.total_heads)

    # Match the legacy K = ceil(sqrt(total)) sanity check used in the original script.
    k = math.ceil(math.sqrt(ctx.info.total_heads))
    sf, lf = syc_grid.flatten(), lie_grid.flatten()
    syc_top = set(np.argsort(sf)[::-1][:k].tolist())
    lie_top = set(np.argsort(lf)[::-1][:k].tolist())
    shared = sorted(syc_top & lie_top)

    result = {
        'model': ctx.model_name,
        'paradigm': cfg.paradigm,
        'task_pair': f'sycophancy_{cfg.paradigm}',
        'n_prompts': cfg.n_prompts,
        'n_layers': ctx.info.n_layers,
        'n_heads': ctx.info.n_heads,
        'total_heads': ctx.info.total_heads,
        'd_model': ctx.info.d_model,
        'k': int(k),
        'shared_count': len(shared),
        'shared_heads': [[int(idx // ctx.info.n_heads), int(idx % ctx.info.n_heads)] for idx in shared],
        'syc_grid': syc_grid.tolist(),
        'lie_grid': lie_grid.tolist(),
        'syc_top15': [
            {'layer': layer, 'head': head, 'delta': syc_deltas[(layer, head)]}
            for (layer, head), _ in syc_ranked[: cfg.top_k]
        ],
        'lie_top15': [
            {'layer': layer, 'head': head, 'delta': lie_deltas[(layer, head)]}
            for (layer, head), _ in lie_ranked[: cfg.top_k]
        ],
    }
    save_results(result, _PARADIGM_TAGS[cfg.paradigm], ctx.model_name)
    return result


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--models', nargs='+', default=list(_DEFAULT_MODELS))
    parser.add_argument(
        '--paradigm',
        choices=_PARADIGMS,
        default='jailbreak',
        help='Instructed-lying prompt builder: jailbreak preamble, few-shot scaffold, or RepE persona.',
    )
    parser.add_argument('--n-prompts', type=int, default=DEFAULT_N_PROMPTS)
    parser.add_argument('--n-pairs', type=int, default=400)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K)


def from_args(args: argparse.Namespace) -> DlaInstructedLyingConfig:
    """Build the validated config from a parsed argparse namespace."""
    return DlaInstructedLyingConfig(
        models=tuple(args.models),
        paradigm=args.paradigm,
        n_prompts=args.n_prompts,
        n_pairs=args.n_pairs,
        n_devices=args.n_devices,
        batch=args.batch,
        top_k=args.top_k,
    )
