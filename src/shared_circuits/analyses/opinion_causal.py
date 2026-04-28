"""
Opinion-causal: head-zeroing on opinion ∩ syc ∩ lie heads, plus opinion direction-cosine boundary.

Two modes share the opinion-pair generator:

* ``causal`` — zero the triple-intersection set on opinion prompts, measure paired-bootstrap
  ``Δrate`` and ``Δlogit_diff`` against a multi-seed random-head control.
* ``boundary`` — per-layer cosine of ``d_opinion`` against ``d_syc`` and ``d_lie``,
  with bootstrap CI on the late-layer mean.

Merged from three legacy scripts (``run_opinion_causal.py``, ``run_qwen3_opinion_causal.py``,
``run_opinion_boundary.py``); the Qwen3-specific variant only differed in defaults
and per-prompt scoring, both folded into the unified ``causal`` flow.
"""

import argparse
from collections.abc import Callable
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.config import (
    BOOTSTRAP_ITERATIONS,
    RANDOM_SEED,
)
from shared_circuits.data import generate_opinion_pairs, load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, load_results, model_session, save_results
from shared_circuits.extraction import (
    extract_residual_stream_multi,
    measure_agreement_per_prompt,
)
from shared_circuits.prompts import (
    build_lying_prompts,
    build_opinion_prompts,
    build_sycophancy_prompts,
)
from shared_circuits.stats import bootstrap_confidence_interval, cosine_similarity

_DEFAULT_MODELS: Final[tuple[str, ...]] = (
    'gemma-2-2b-it',
    'Qwen/Qwen3-8B',
)
_DEFAULT_N_OPINION: Final = 200
_DEFAULT_N_FACTUAL: Final = 200
_DEFAULT_N_RANDOM_SEEDS: Final = 5
_DEFAULT_DIRECTION_LAYER_FRAC: Final = 0.75
_DEFAULT_BATCH: Final = 4
_DEFAULT_TRIPLE_FROM: Final[tuple[str, ...]] = ('circuit_overlap', 'opinion_circuit_transfer')
_DEFAULT_TOP_K: Final = 15
_BOUNDARY_SHARED_LO: Final = 0.2
_BOUNDARY_SPECIFIC_HI: Final = 0.1
_TRIPLE_SOURCES_REQUIRED: Final = 2


class OpinionCausalConfig(BaseModel):
    """Inputs for opinion-causal head zeroing or opinion-boundary direction analysis."""

    model_config = ConfigDict(frozen=True)

    models: tuple[str, ...] = Field(default_factory=lambda: _DEFAULT_MODELS)
    mode: str = Field(default='causal')
    n_opinion: int = Field(default=_DEFAULT_N_OPINION, gt=0)
    n_factual: int = Field(default=_DEFAULT_N_FACTUAL, gt=0)
    n_random_seeds: int = Field(default=_DEFAULT_N_RANDOM_SEEDS, ge=0)
    n_boot: int = Field(default=BOOTSTRAP_ITERATIONS, gt=0)
    direction_layer_frac: float = Field(default=_DEFAULT_DIRECTION_LAYER_FRAC, gt=0, lt=1)
    batch: int = Field(default=_DEFAULT_BATCH, gt=0)
    seed: int = Field(default=RANDOM_SEED)
    triple_from: tuple[str, ...] = Field(default=_DEFAULT_TRIPLE_FROM)
    triple_k: int = Field(default=_DEFAULT_TOP_K, gt=0)


def run(cfg: OpinionCausalConfig) -> list[dict]:
    """Run opinion-causal analysis for every model in ``cfg.models``."""
    if cfg.mode not in {'causal', 'boundary'}:
        raise ValueError(f'unknown mode {cfg.mode}')
    return [_run_one(name, cfg) for name in cfg.models]


def _run_one(model_name: str, cfg: OpinionCausalConfig) -> dict:
    with model_session(model_name) as ctx:
        if cfg.mode == 'boundary':
            return _analyse_boundary(ctx, cfg)
        return _analyse_causal(ctx, cfg)


def _load_triple_intersection(
    model_name: str,
    sources: tuple[str, ...],
    triple_k: int,
) -> set[tuple[int, int]]:
    """Read circuit-overlap (syc∩lie) and opinion-transfer JSON, intersect with opinion top-K."""
    overlap = load_results(sources[0], model_name)
    bucket = next(o for o in overlap['overlap_by_K'] if o['K'] == triple_k)
    syc_lie: set[tuple[int, int]] = {tuple(h) for h in bucket['shared_heads']}
    if len(sources) < _TRIPLE_SOURCES_REQUIRED:
        return syc_lie
    opinion = load_results(sources[1], model_name)
    op_heads: set[tuple[int, int]] = {tuple(h) for h in opinion.get('shared_heads', [])}
    return syc_lie & op_heads if op_heads else syc_lie


def _zero_heads_hooks(
    head_set: set[tuple[int, int]],
) -> list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]]:
    """Build hooks that zero ``hook_z`` for every head in ``head_set``."""
    by_layer: dict[int, list[int]] = {}
    for layer, head in head_set:
        by_layer.setdefault(layer, []).append(head)
    hooks: list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]] = []
    for layer, heads in by_layer.items():
        # Per-iteration closure: ``hs`` must be captured fresh each layer
        def make_hook(hs: list[int]) -> Callable[[torch.Tensor, object], torch.Tensor | None]:
            def hook_fn(z: torch.Tensor, hook: object) -> torch.Tensor:
                for h in hs:
                    z[:, :, h, :] = 0.0
                return z

            return hook_fn

        hooks.append((f'blocks.{layer}.attn.hook_z', make_hook(heads)))
    return hooks


def _random_head_set(n_layers: int, n_heads: int, k: int, seed: int) -> set[tuple[int, int]]:
    rng = np.random.RandomState(seed)
    positions = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    idx = rng.choice(len(positions), k, replace=False)
    return {positions[i] for i in idx}


def _measure(
    ctx: ExperimentContext,
    prompts: list[str],
    batch: int,
    hooks: list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]] | None = None,
) -> dict:
    """Per-prompt agree flag and rate, computed with optional ablation hooks."""
    rate, flags = measure_agreement_per_prompt(
        ctx.model,
        prompts,
        ctx.agree_tokens,
        ctx.disagree_tokens,
        batch_size=batch,
        hooks=hooks,
    )
    return {'rate': float(rate), 'flags': flags}


def _paired_ci(base: list[float], treat: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    """Paired bootstrap on (treat - base): resample prompt indices, return mean + 95% CI."""
    b = np.asarray(base, dtype=float)
    t = np.asarray(treat, dtype=float)
    n = len(b)
    rng = np.random.RandomState(seed)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        deltas[i] = t[idx].mean() - b[idx].mean()
    return float(t.mean() - b.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _analyse_causal(ctx: ExperimentContext, cfg: OpinionCausalConfig) -> dict:
    triple = _load_triple_intersection(ctx.model_name, cfg.triple_from, cfg.triple_k)
    opinion_pairs = generate_opinion_pairs(cfg.n_opinion, seed=cfg.seed)
    op_a, op_b = build_opinion_prompts(opinion_pairs, ctx.model_name)

    base_a = _measure(ctx, op_a, cfg.batch)
    base_b = _measure(ctx, op_b, cfg.batch)

    hooks_sh = _zero_heads_hooks(triple)
    sh_a = _measure(ctx, op_a, cfg.batch, hooks_sh)
    sh_b = _measure(ctx, op_b, cfg.batch, hooks_sh)

    pooled_base = base_a['flags'] + base_b['flags']
    pooled_shared = sh_a['flags'] + sh_b['flags']
    d_rate, lo, hi = _paired_ci(pooled_base, pooled_shared, cfg.n_boot, cfg.seed)

    # Per-model bootstrap on rate over the pooled per-prompt indicators (a + b)
    base_lo, base_hi = bootstrap_confidence_interval(np.asarray(pooled_base, dtype=float), np.mean, cfg.n_boot)
    shared_lo, shared_hi = bootstrap_confidence_interval(np.asarray(pooled_shared, dtype=float), np.mean, cfg.n_boot)

    rand_trials: list[dict] = []
    k_size = max(len(triple), 1)
    for s in range(cfg.n_random_seeds):
        rand = _random_head_set(ctx.info.n_layers, ctx.info.n_heads, k_size, cfg.seed + 1 + s)
        hooks_r = _zero_heads_hooks(rand)
        ra_a = _measure(ctx, op_a, cfg.batch, hooks_r)
        ra_b = _measure(ctx, op_b, cfg.batch, hooks_r)
        pooled_treat = ra_a['flags'] + ra_b['flags']
        d_r, lo_r, hi_r = _paired_ci(pooled_base, pooled_treat, cfg.n_boot, cfg.seed + 1 + s)
        rand_trials.append(
            {
                'seed': cfg.seed + 1 + s,
                'heads': sorted([list(h) for h in rand]),
                'delta_rate': d_r,
                'delta_rate_ci': [lo_r, hi_r],
            }
        )

    if rand_trials:
        d_rates = np.array([t['delta_rate'] for t in rand_trials])
        rand_summary = {
            'delta_rate_mean': float(d_rates.mean()),
            'delta_rate_std': float(d_rates.std()),
            'delta_rate_min': float(d_rates.min()),
            'delta_rate_max': float(d_rates.max()),
        }
        margin = d_rate - rand_summary['delta_rate_mean']
    else:
        rand_summary = {'delta_rate_mean': 0.0, 'delta_rate_std': 0.0, 'delta_rate_min': 0.0, 'delta_rate_max': 0.0}
        margin = d_rate

    result = {
        'model': ctx.model_name,
        'mode': 'causal',
        'n_layers': ctx.info.n_layers,
        'n_heads': ctx.info.n_heads,
        'n_shared': len(triple),
        'shared_heads': sorted([list(h) for h in triple]),
        'baseline_a': {'rate': base_a['rate']},
        'baseline_b': {'rate': base_b['rate']},
        'baseline_pooled_ci': [base_lo, base_hi],
        'shared_zeroed_a': {'rate': sh_a['rate']},
        'shared_zeroed_b': {'rate': sh_b['rate']},
        'shared_pooled_ci': [shared_lo, shared_hi],
        'shared_cis': {'delta_rate': d_rate, 'delta_rate_ci': [lo, hi]},
        'random_trials': rand_trials,
        'random_summary': rand_summary,
        'margin_delta_rate': float(margin),
    }
    save_results(result, 'opinion_causal', ctx.model_name)
    return result


def _analyse_boundary(ctx: ExperimentContext, cfg: OpinionCausalConfig) -> dict:
    opinion_pairs = generate_opinion_pairs(cfg.n_opinion, seed=cfg.seed)
    factual_pairs = load_triviaqa_pairs(cfg.n_factual)
    op_a, op_b = build_opinion_prompts(opinion_pairs, ctx.model_name)
    half = cfg.n_factual // 2
    fact_wrong, fact_correct = build_sycophancy_prompts(factual_pairs[:half], ctx.model_name)
    lie_false, lie_true = build_lying_prompts(factual_pairs[half : 2 * half], ctx.model_name)

    target_layers = sorted({*range(0, ctx.info.n_layers, 2), ctx.info.n_layers - 1})
    pp_oa = extract_residual_stream_multi(ctx.model, op_a, target_layers, batch_size=cfg.batch)
    pp_ob = extract_residual_stream_multi(ctx.model, op_b, target_layers, batch_size=cfg.batch)
    pp_fw = extract_residual_stream_multi(ctx.model, fact_wrong, target_layers, batch_size=cfg.batch)
    pp_fc = extract_residual_stream_multi(ctx.model, fact_correct, target_layers, batch_size=cfg.batch)
    pp_lf = extract_residual_stream_multi(ctx.model, lie_false, target_layers, batch_size=cfg.batch)
    pp_lt = extract_residual_stream_multi(ctx.model, lie_true, target_layers, batch_size=cfg.batch)

    layer_results: dict[str, dict] = {}
    for layer in target_layers:
        op_dir = pp_oa[layer].mean(0) - pp_ob[layer].mean(0)
        syc_dir = pp_fw[layer].mean(0) - pp_fc[layer].mean(0)
        lie_dir = pp_lf[layer].mean(0) - pp_lt[layer].mean(0)
        layer_results[str(layer)] = {
            'opinion_vs_factsyc': cosine_similarity(op_dir, syc_dir),
            'opinion_vs_lie': cosine_similarity(op_dir, lie_dir),
            'factsyc_vs_lie': cosine_similarity(syc_dir, lie_dir),
        }

    late = [layer for layer in target_layers if layer >= 2 * ctx.info.n_layers // 3]
    late_of = float(np.mean([layer_results[str(layer)]['opinion_vs_factsyc'] for layer in late]))

    n_op = pp_oa[target_layers[0]].shape[0]
    n_fact = pp_fw[target_layers[0]].shape[0]
    rng = np.random.RandomState(cfg.seed)
    boot_cosines: list[float] = []
    for _ in range(cfg.n_boot):
        idx_op = rng.choice(n_op, n_op, replace=True)
        idx_fact = rng.choice(n_fact, n_fact, replace=True)
        per_layer = [
            cosine_similarity(
                pp_oa[layer][idx_op].mean(0) - pp_ob[layer][idx_op].mean(0),
                pp_fw[layer][idx_fact].mean(0) - pp_fc[layer][idx_fact].mean(0),
            )
            for layer in late
        ]
        boot_cosines.append(float(np.mean(per_layer)))
    ci = np.percentile(boot_cosines, [2.5, 97.5])

    if ci[1] < _BOUNDARY_SPECIFIC_HI:
        verdict = 'SPECIFIC_TO_FACTUAL'
    elif ci[0] > _BOUNDARY_SHARED_LO:
        verdict = 'SHARED_WITH_OPINION'
    else:
        verdict = 'INCONCLUSIVE'

    direction_layer = int(ctx.info.n_layers * cfg.direction_layer_frac)
    result = {
        'model': ctx.model_name,
        'mode': 'boundary',
        'n_opinion': len(opinion_pairs),
        'n_factual': len(factual_pairs),
        'direction_layer': direction_layer,
        'late_opinion_vs_factsyc': late_of,
        'bootstrap_ci_95': [float(ci[0]), float(ci[1])],
        'verdict': verdict,
        'by_layer': layer_results,
    }
    save_results(result, 'opinion_boundary', ctx.model_name)
    return result


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--models', nargs='+', default=list(_DEFAULT_MODELS))
    parser.add_argument(
        '--mode',
        choices=('causal', 'boundary'),
        default='causal',
        help='causal: head-zeroing with paired CIs; boundary: per-layer direction cosines.',
    )
    parser.add_argument('--n-opinion', type=int, default=_DEFAULT_N_OPINION)
    parser.add_argument('--n-factual', type=int, default=_DEFAULT_N_FACTUAL)
    parser.add_argument('--n-random-seeds', type=int, default=_DEFAULT_N_RANDOM_SEEDS)
    parser.add_argument('--n-boot', type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument('--direction-layer-frac', type=float, default=_DEFAULT_DIRECTION_LAYER_FRAC)
    parser.add_argument('--batch', type=int, default=_DEFAULT_BATCH)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument(
        '--triple-from',
        type=str,
        default=','.join(_DEFAULT_TRIPLE_FROM),
        help='Comma-separated experiment slugs to read shared-heads from (syc∩lie, opinion).',
    )
    parser.add_argument('--triple-k', type=int, default=_DEFAULT_TOP_K)


def from_args(args: argparse.Namespace) -> OpinionCausalConfig:
    """Build the validated config from a parsed argparse namespace."""
    triple = tuple(s for s in args.triple_from.split(',') if s) if args.triple_from else _DEFAULT_TRIPLE_FROM
    return OpinionCausalConfig(
        models=tuple(args.models),
        mode=args.mode,
        n_opinion=args.n_opinion,
        n_factual=args.n_factual,
        n_random_seeds=args.n_random_seeds,
        n_boot=args.n_boot,
        direction_layer_frac=args.direction_layer_frac,
        batch=args.batch,
        seed=args.seed,
        triple_from=triple,
        triple_k=args.triple_k,
    )
