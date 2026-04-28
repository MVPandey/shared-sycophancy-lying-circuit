"""
IOI/ACDC-style faithfulness curve.

Zero every attention head, then progressively restore the top-K shared heads
(ordered by combined importance) and measure sycophancy rate / logit_diff at each
K. Faithfulness is ``(metric_K - metric_all_ablated) / (metric_baseline - metric_all_ablated)``.

Two modes:

* ``single`` — evaluate at a small set of K values (default ``1, 2, 5, 10, full``).
* ``curve`` — sweep K from 1 to ``n_shared`` to produce the full curve.

Reads the shared-heads ranking from a sibling analysis JSON (default
``circuit_overlap``), which means this analysis is read-only on results from
:mod:`shared_circuits.analyses.circuit_overlap` or
:mod:`shared_circuits.analyses.attribution_patching`.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.config import DEFAULT_BATCH_SIZE, RANDOM_SEED
from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, load_results, model_session, save_results
from shared_circuits.extraction import measure_agreement_per_prompt
from shared_circuits.prompts import build_sycophancy_prompts

_DEFAULT_N_PROMPTS: Final = 100
_DEFAULT_K_VALUES: Final[tuple[int, ...]] = (1, 2, 5, 10)
_DEFAULT_FAITH_THRESHOLD: Final = 0.8
_WILSON_Z: Final = 1.96
_FAITH_DENOM_EPS: Final = 1e-6


class FaithfulnessConfig(BaseModel):
    """Inputs for the faithfulness curve analysis (single model)."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(...)
    n_devices: int = Field(default=1, gt=0)
    n_prompts: int = Field(default=_DEFAULT_N_PROMPTS, gt=0)
    n_pairs: int = Field(default=_DEFAULT_N_PROMPTS * 2, gt=0)
    mode: str = Field(default='curve')
    k_values: tuple[int, ...] = Field(default=_DEFAULT_K_VALUES)
    batch: int = Field(default=DEFAULT_BATCH_SIZE, gt=0)
    seed: int = Field(default=RANDOM_SEED)
    shared_heads_from: str = Field(default='circuit_overlap')
    shared_heads_k: int = Field(default=15, gt=0)
    faithfulness_threshold: float = Field(default=_DEFAULT_FAITH_THRESHOLD, gt=0, le=1)


def run(cfg: FaithfulnessConfig) -> dict:
    """Run the faithfulness curve analysis for ``cfg.model``."""
    if cfg.mode not in {'single', 'curve'}:
        raise ValueError(f'unknown mode {cfg.mode}')
    if cfg.shared_heads_from not in {'circuit_overlap', 'attribution_patching'}:
        raise ValueError(f'unknown shared_heads_from {cfg.shared_heads_from}')
    ranked = _load_shared_ranked(cfg.shared_heads_from, cfg.model, cfg.shared_heads_k)
    pairs = load_triviaqa_pairs(cfg.n_pairs)[: cfg.n_prompts]
    wrong_prompts, _correct = build_sycophancy_prompts(pairs, cfg.model)
    with model_session(cfg.model, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, cfg, ranked, wrong_prompts)


def _load_shared_ranked(source: str, model_name: str, top_k: int) -> list[tuple[int, int]]:
    """Read shared-head ranking from a sibling analysis' saved JSON."""
    data = load_results(source, model_name)
    if source == 'circuit_overlap':
        return _ranked_from_overlap(data, top_k)
    return _ranked_from_attribution(data, top_k)


def _ranked_from_overlap(data: dict, top_k: int) -> list[tuple[int, int]]:
    """Pick the top-K bucket and order shared heads by combined-rank importance."""
    bucket = next(o for o in data['overlap_by_K'] if o['K'] == top_k)
    return [tuple(h) for h in bucket['shared_heads']]


def _ranked_from_attribution(data: dict, top_k: int) -> list[tuple[int, int]]:
    """Intersect top-K syc and lie attribution-patch grids and rank by combined effect."""
    sg = np.array(data['syc_patch_grid'])
    lg = np.array(data['lie_patch_grid'])
    _, n_heads = sg.shape
    sf = np.abs(sg).flatten()
    lf = np.abs(lg).flatten()
    syc_top = set(np.argsort(sf)[::-1][:top_k].tolist())
    lie_top = set(np.argsort(lf)[::-1][:top_k].tolist())
    shared_idx = list(syc_top & lie_top)
    combined = (sf + lf) / 2
    shared_idx.sort(key=lambda i: -combined[i])
    return [(int(i // n_heads), int(i % n_heads)) for i in shared_idx]


def _zero_all_except(
    n_layers: int,
    n_heads: int,
    keep: set[tuple[int, int]],
) -> list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]]:
    """Build hooks that zero every attention head except those in ``keep``."""
    by_layer: dict[int, list[int]] = {}
    for layer in range(n_layers):
        zeros = [h for h in range(n_heads) if (layer, h) not in keep]
        if zeros:
            by_layer[layer] = zeros
    hooks: list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]] = []
    for layer, heads in by_layer.items():
        # Per-iteration closure: ``hs`` must be captured fresh each layer.
        def make_hook(hs: list[int]) -> Callable[[torch.Tensor, object], torch.Tensor | None]:
            def hook_fn(z: torch.Tensor, hook: object) -> torch.Tensor:
                for h in hs:
                    z[:, :, h, :] = 0.0
                return z

            return hook_fn

        hooks.append((f'blocks.{layer}.attn.hook_z', make_hook(heads)))
    return hooks


def _measure_rate(
    ctx: ExperimentContext,
    prompts: list[str],
    batch: int,
    hooks: list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]] | None = None,
) -> tuple[float, list[float]]:
    """Wrap :func:`measure_agreement_per_prompt` with the context's tokens."""
    return measure_agreement_per_prompt(
        ctx.model,
        prompts,
        ctx.agree_tokens,
        ctx.disagree_tokens,
        batch_size=batch,
        hooks=hooks,
    )


def _wilson_ci(p: float, n: int, z: float = _WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion at the standard 95% z."""
    if n == 0:
        return 0.0, 0.0
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def _faithfulness(metric_k: float, metric_ablated: float, metric_baseline: float) -> float:
    denom = metric_baseline - metric_ablated
    if abs(denom) < _FAITH_DENOM_EPS:
        return 0.0
    return float((metric_k - metric_ablated) / denom)


def _select_k_values(cfg: FaithfulnessConfig, n_shared: int) -> list[int]:
    if cfg.mode == 'curve':
        return list(range(1, n_shared + 1))
    chosen = sorted({k for k in cfg.k_values if 1 <= k <= n_shared}.union({n_shared}))
    return chosen


def _analyse(
    ctx: ExperimentContext,
    cfg: FaithfulnessConfig,
    ranked: list[tuple[int, int]],
    prompts: list[str],
) -> dict:
    n_shared = len(ranked)
    if not prompts:
        raise ValueError('empty prompts after filtering — increase --n-pairs')

    baseline_rate, _ = _measure_rate(ctx, prompts, cfg.batch)
    base_lo, base_hi = _wilson_ci(baseline_rate, len(prompts))

    ablated_hooks = _zero_all_except(ctx.info.n_layers, ctx.info.n_heads, set())
    ablated_rate, _ = _measure_rate(ctx, prompts, cfg.batch, ablated_hooks)
    ab_lo, ab_hi = _wilson_ci(ablated_rate, len(prompts))

    k_values = _select_k_values(cfg, n_shared)
    curve: list[dict] = []
    for k in k_values:
        keep = set(ranked[:k])
        hooks = _zero_all_except(ctx.info.n_layers, ctx.info.n_heads, keep)
        rate, _ = _measure_rate(ctx, prompts, cfg.batch, hooks)
        lo, hi = _wilson_ci(rate, len(prompts))
        faith = _faithfulness(rate, ablated_rate, baseline_rate)
        curve.append(
            {
                'k': k,
                'head': list(ranked[k - 1]) if k <= n_shared else None,
                'syc_rate': rate,
                'wilson_ci': [lo, hi],
                'faithfulness_ratio': faith,
            }
        )

    peak_faith = max((row['faithfulness_ratio'] for row in curve), default=0.0)
    first_k_at_threshold: int | None = next(
        (row['k'] for row in curve if row['faithfulness_ratio'] >= cfg.faithfulness_threshold),
        None,
    )

    result = {
        'model': ctx.model_name,
        'mode': cfg.mode,
        'n_layers': ctx.info.n_layers,
        'n_heads': ctx.info.n_heads,
        'n_shared': n_shared,
        'shared_heads_ranked': [list(h) for h in ranked],
        'n_prompts': len(prompts),
        'baseline': {'syc_rate': baseline_rate, 'wilson_ci': [base_lo, base_hi]},
        'all_ablated': {'syc_rate': ablated_rate, 'wilson_ci': [ab_lo, ab_hi]},
        'curve': curve,
        'peak_faithfulness_ratio': peak_faith,
        'first_k_at_threshold': first_k_at_threshold,
        'faithfulness_threshold': cfg.faithfulness_threshold,
    }
    save_results(result, 'faithfulness', ctx.model_name)
    return result


def _parse_k_values(value: str | None) -> tuple[int, ...]:
    if value is None:
        return _DEFAULT_K_VALUES
    return tuple(int(x) for x in value.split(',') if x.strip())


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--n-prompts', type=int, default=_DEFAULT_N_PROMPTS)
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PROMPTS * 2)
    parser.add_argument(
        '--mode',
        choices=('single', 'curve'),
        default='curve',
        help='single: evaluate at --k-values; curve: sweep K from 1 to n_shared.',
    )
    parser.add_argument(
        '--k-values',
        type=str,
        default=None,
        help='Comma-separated K values for --mode single (default 1,2,5,10).',
    )
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument(
        '--shared-heads-from',
        choices=('circuit_overlap', 'attribution_patching'),
        default='circuit_overlap',
        help='Sibling experiment slug whose JSON provides the shared-heads ranking.',
    )
    parser.add_argument('--shared-heads-k', type=int, default=15)
    parser.add_argument(
        '--faithfulness-threshold',
        type=float,
        default=_DEFAULT_FAITH_THRESHOLD,
        help='Threshold for first_k_at_threshold (e.g. 0.8 for 80%% recovery).',
    )


def from_args(args: argparse.Namespace) -> FaithfulnessConfig:
    """Build the validated config from a parsed argparse namespace."""
    return FaithfulnessConfig(
        model=args.model,
        n_devices=args.n_devices,
        n_prompts=args.n_prompts,
        n_pairs=args.n_pairs,
        mode=args.mode,
        k_values=_parse_k_values(args.k_values),
        batch=args.batch,
        seed=args.seed,
        shared_heads_from=args.shared_heads_from,
        shared_heads_k=args.shared_heads_k,
        faithfulness_threshold=args.faithfulness_threshold,
    )
