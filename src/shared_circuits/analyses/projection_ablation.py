"""
Projection ablation: subtract the sycophancy direction from the residual stream.

Complements steering (which adds the syc direction to flip behavior) by removing
the direction at each forward pass.  Tests directional necessity: if shared
heads write sycophancy signal along a specific direction, projecting it out
should reduce syc rate.

The direction is extracted as ``mean(wrong-opinion resid) - mean(correct-opinion resid)``
at the chosen layer, then unit-normalized.  A random orthogonal direction with
matched magnitude serves as a control.

Supports a layer sweep (``--layer-fracs``) so callers can replicate either the
default Gemma-style sweep ``(0.5, 0.6, 0.7, 0.8)`` or the Qwen3 sweep
``(0.45, 0.55, 0.65, 0.75)`` from the legacy ``run_qwen3_projection_ablation``.
A single layer can also be pinned via ``--layer``.
"""

import argparse
from collections.abc import Callable
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.config import RANDOM_SEED
from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, model_session, save_results
from shared_circuits.extraction import extract_residual_stream, measure_agreement_per_prompt
from shared_circuits.prompts import build_sycophancy_prompts

_DEFAULT_N_PAIRS: Final = 400
_DEFAULT_DIR_PROMPTS: Final = 50
_DEFAULT_TEST_PROMPTS: Final = 200
_DEFAULT_BATCH: Final = 2
_DEFAULT_N_BOOT: Final = 2000
_DEFAULT_LAYER_FRACS: Final[tuple[float, ...]] = (0.5, 0.6, 0.7, 0.8)
_QWEN3_LAYER_FRACS: Final[tuple[float, ...]] = (0.45, 0.55, 0.65, 0.75)
# verdict thresholds: a directional ablation is judged effective when the
# delta against the random-direction control exceeds this margin at any layer.
_EFFECTIVE_MARGIN: Final = 0.05


class ProjectionAblationConfig(BaseModel):
    """Inputs for the projection-ablation analysis (single model)."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(...)
    n_devices: int = Field(default=1, gt=0)
    batch: int = Field(default=_DEFAULT_BATCH, gt=0)
    layer: int | None = Field(default=None)
    layer_fracs: tuple[float, ...] = Field(default=_DEFAULT_LAYER_FRACS)
    n_pairs: int = Field(default=_DEFAULT_N_PAIRS, gt=0)
    dir_prompts: int = Field(default=_DEFAULT_DIR_PROMPTS, gt=0)
    test_prompts: int = Field(default=_DEFAULT_TEST_PROMPTS, gt=0)
    n_boot: int = Field(default=_DEFAULT_N_BOOT, gt=0)
    seed: int = Field(default=RANDOM_SEED)


def run(cfg: ProjectionAblationConfig) -> dict:
    """Run projection ablation on ``cfg.model`` (single layer or full sweep)."""
    pairs = load_triviaqa_pairs(cfg.n_pairs)
    with model_session(cfg.model, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, pairs, cfg)


def _analyse(ctx: ExperimentContext, pairs: list[tuple[str, str, str]], cfg: ProjectionAblationConfig) -> dict:
    dir_pairs = pairs[: cfg.dir_prompts]
    test_pairs = pairs[cfg.dir_prompts : cfg.dir_prompts + cfg.test_prompts]
    dir_wrong, dir_correct = build_sycophancy_prompts(dir_pairs, ctx.model_name)
    test_wrong, _ = build_sycophancy_prompts(test_pairs, ctx.model_name)

    base_rate, base_pp = measure_agreement_per_prompt(
        ctx.model, test_wrong, ctx.agree_tokens, ctx.disagree_tokens, batch_size=cfg.batch
    )

    if cfg.layer is not None:
        layers = [cfg.layer]
    else:
        layers = sorted({int(ctx.info.n_layers * f) for f in cfg.layer_fracs})

    by_layer: dict[str, dict] = {}
    for layer in layers:
        by_layer[str(layer)] = _run_one_layer(ctx, dir_wrong, dir_correct, test_wrong, base_pp, layer, cfg)

    verdict = _verdict(by_layer)

    results = {
        'model': ctx.model_name,
        'verdict': verdict,
        'config': {
            'n_pairs': cfg.n_pairs,
            'dir_prompts': cfg.dir_prompts,
            'test_prompts': len(test_wrong),
            'batch': cfg.batch,
            'layers': layers,
            'layer_fracs': list(cfg.layer_fracs) if cfg.layer is None else None,
            'n_boot': cfg.n_boot,
            'seed': cfg.seed,
        },
        'n_layers': ctx.info.n_layers,
        'n_heads': ctx.info.n_heads,
        'd_model': ctx.info.d_model,
        'baseline_syc_rate': base_rate,
        'layers': by_layer,
    }
    save_results(results, 'projection_ablation', ctx.model_name)
    return results


def _run_one_layer(
    ctx: ExperimentContext,
    dir_wrong: list[str],
    dir_correct: list[str],
    test_wrong: list[str],
    base_pp: list[float],
    layer: int,
    cfg: ProjectionAblationConfig,
) -> dict:
    unit, resid_norm = _extract_direction(ctx.model, dir_wrong, dir_correct, layer, cfg.batch)
    dir_t = torch.tensor(unit, dtype=torch.bfloat16, device=str(ctx.model.cfg.device))
    rng = np.random.RandomState(cfg.seed)
    rand = rng.randn(len(unit)).astype(np.float32)
    rand = rand / np.linalg.norm(rand)
    rand_t = torch.tensor(rand, dtype=torch.bfloat16, device=str(ctx.model.cfg.device))

    hook_name = f'blocks.{layer}.hook_resid_post'
    proj_rate, proj_pp = measure_agreement_per_prompt(
        ctx.model,
        test_wrong,
        ctx.agree_tokens,
        ctx.disagree_tokens,
        batch_size=cfg.batch,
        hooks=[(hook_name, _make_proj_hook(dir_t))],
    )
    rand_rate, rand_pp = measure_agreement_per_prompt(
        ctx.model,
        test_wrong,
        ctx.agree_tokens,
        ctx.disagree_tokens,
        batch_size=cfg.batch,
        hooks=[(hook_name, _make_proj_hook(rand_t))],
    )

    d_proj, p_lo, p_hi = _paired_ci(base_pp, proj_pp, cfg.n_boot, cfg.seed)
    d_rand, r_lo, r_hi = _paired_ci(base_pp, rand_pp, cfg.n_boot, cfg.seed)

    return {
        'resid_norm': resid_norm,
        'projection': {
            'syc_rate': proj_rate,
            'syc_delta': d_proj,
            'syc_ci': [p_lo, p_hi],
            'significant': bool(p_lo > 0 or p_hi < 0),
        },
        'random_projection': {
            'syc_rate': rand_rate,
            'syc_delta': d_rand,
            'syc_ci': [r_lo, r_hi],
            'significant': bool(r_lo > 0 or r_hi < 0),
        },
        'margin': float(d_proj - d_rand),
    }


def _extract_direction(
    model: 'HookedTransformer',
    dir_wrong: list[str],
    dir_correct: list[str],
    layer: int,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    """Extract syc direction at ``layer``, returning unit vector + corrupt resid norm."""
    acts_w = extract_residual_stream(model, dir_wrong, layer, batch_size=batch_size)
    acts_c = extract_residual_stream(model, dir_correct, layer, batch_size=batch_size)
    direction = acts_w.mean(0) - acts_c.mean(0)
    unit = direction / (np.linalg.norm(direction) + 1e-10)
    resid_norm = float(np.linalg.norm(acts_w.mean(0)))
    return unit, resid_norm


def _make_proj_hook(direction: torch.Tensor) -> Callable[[torch.Tensor, object], torch.Tensor]:
    """Build a forward hook that projects out ``direction`` from the residual at every position."""

    def hook_fn(t: torch.Tensor, hook: object) -> torch.Tensor:
        d = direction.to(t.device, dtype=t.dtype)
        proj = (t @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        return t - proj

    return hook_fn


def _paired_ci(base: list[float], abl: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    """Paired bootstrap 95% CI for ``mean(abl) - mean(base)`` over prompt indices."""
    b = np.array(base)
    a = np.array(abl)
    n = len(b)
    rng = np.random.RandomState(seed)
    boots = np.array([a[idx].mean() - b[idx].mean() for idx in (rng.choice(n, n, replace=True) for _ in range(n_boot))])
    return float(a.mean() - b.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _verdict(by_layer: dict[str, dict]) -> str:
    """Verdict based on best layer's projection-vs-random margin."""
    if not by_layer:
        return 'INCOMPLETE'
    best = min(
        layer_res['projection']['syc_delta'] - layer_res['random_projection']['syc_delta']
        for layer_res in by_layer.values()
    )
    # negative delta on syc rate = ablation reduced syc; "best" is the most-negative margin
    if best < -_EFFECTIVE_MARGIN:
        return 'DIRECTION_NECESSARY'
    if best < 0:
        return 'PARTIAL_DIRECTION'
    return 'NO_DIRECTION_EFFECT'


def _parse_layer_fracs(value: str | None, qwen3: bool) -> tuple[float, ...]:
    """Resolve layer-fracs from CLI arg + ``--qwen3-layer-sweep`` flag."""
    if value is not None:
        return tuple(float(x) for x in value.split(','))
    return _QWEN3_LAYER_FRACS if qwen3 else _DEFAULT_LAYER_FRACS


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--batch', type=int, default=_DEFAULT_BATCH)
    parser.add_argument(
        '--layer',
        type=int,
        default=None,
        help='Pin to a single layer; overrides --layer-fracs/--qwen3-layer-sweep.',
    )
    parser.add_argument(
        '--layer-fracs',
        type=str,
        default=None,
        help='Comma-separated depth fractions (e.g. "0.5,0.6,0.7"). Default: 0.5,0.6,0.7,0.8.',
    )
    parser.add_argument(
        '--qwen3-layer-sweep',
        action='store_true',
        help='Use the Qwen3-tuned default layer fractions (0.45, 0.55, 0.65, 0.75).',
    )
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PAIRS)
    parser.add_argument('--dir-prompts', type=int, default=_DEFAULT_DIR_PROMPTS)
    parser.add_argument('--test-prompts', type=int, default=_DEFAULT_TEST_PROMPTS)
    parser.add_argument('--n-boot', type=int, default=_DEFAULT_N_BOOT)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)


def from_args(args: argparse.Namespace) -> ProjectionAblationConfig:
    """Build the validated config from a parsed argparse namespace."""
    return ProjectionAblationConfig(
        model=args.model,
        n_devices=args.n_devices,
        batch=args.batch,
        layer=args.layer,
        layer_fracs=_parse_layer_fracs(args.layer_fracs, args.qwen3_layer_sweep),
        n_pairs=args.n_pairs,
        dir_prompts=args.dir_prompts,
        test_prompts=args.test_prompts,
        n_boot=args.n_boot,
        seed=args.seed,
    )
