"""DLA-based head overlap between sycophancy and lying paradigms."""

from __future__ import annotations

import argparse
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.attribution import compute_head_importances, rank_heads
from shared_circuits.config import ALL_MODELS, DEFAULT_N_PROMPTS, DEFAULT_TOP_K
from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, model_session, save_results
from shared_circuits.prompts import build_lying_prompts, build_sycophancy_prompts
from shared_circuits.stats import head_overlap_hypergeometric, permutation_test_overlap, rank_correlation

_DEFAULT_N_PAIRS: Final = 400
_OVERLAP_K_VALUES: Final[tuple[int, ...]] = (5, 10, 15, 20, 30, 50)
_VERDICT_K: Final = 15
_SHARED_RATIO: Final = 3.0
_PARTIAL_RATIO: Final = 1.5
_SHARED_P: Final = 0.01
_PARTIAL_P: Final = 0.05


class CircuitOverlapConfig(BaseModel):
    """Inputs for the circuit-overlap analysis."""

    model_config = ConfigDict(frozen=True)

    models: tuple[str, ...] = Field(default_factory=lambda: tuple(ALL_MODELS))
    n_prompts: int = Field(default=DEFAULT_N_PROMPTS, gt=0)
    n_pairs: int = Field(default=_DEFAULT_N_PAIRS, gt=0)


def run(cfg: CircuitOverlapConfig) -> list[dict]:
    """Run circuit-overlap analysis for every model in ``cfg.models``."""
    pairs = load_triviaqa_pairs(cfg.n_pairs)
    return [_run_one(name, pairs, cfg.n_prompts) for name in cfg.models]


def _run_one(model_name: str, pairs: list[tuple[str, str, str]], n_prompts: int) -> dict:
    with model_session(model_name) as ctx:
        return _analyse(ctx, pairs, n_prompts)


def _analyse(ctx: ExperimentContext, pairs: list[tuple[str, str, str]], n_prompts: int) -> dict:
    set_a, set_b = pairs[:200], pairs[200:400]
    wrong_a, correct_a = build_sycophancy_prompts(set_a, ctx.model_name)
    false_b, true_b = build_lying_prompts(set_b, ctx.model_name)

    syc_deltas = compute_head_importances(ctx.model, wrong_a, correct_a, n_prompts=n_prompts)
    lie_deltas = compute_head_importances(ctx.model, false_b, true_b, n_prompts=n_prompts)
    syc_ranked = rank_heads(syc_deltas, top_k=ctx.info.total_heads)
    lie_ranked = rank_heads(lie_deltas, top_k=ctx.info.total_heads)

    overlap_results: list[dict] = []
    all_heads = list(syc_deltas.keys())
    for k in _OVERLAP_K_VALUES:
        syc_keys = {h for h, _ in syc_ranked[:k]}
        lie_keys = {h for h, _ in lie_ranked[:k]}
        overlap = len(syc_keys & lie_keys)
        expected = k * k / ctx.info.total_heads
        overlap_results.append(
            {
                'K': k,
                'overlap': overlap,
                'expected': float(expected),
                'ratio': overlap / expected if expected > 0 else 0.0,
                'p_hypergeometric': float(head_overlap_hypergeometric(overlap, k, ctx.info.total_heads)),
                'p_permutation': float(permutation_test_overlap(syc_keys, lie_keys, all_heads, k)),
                'shared_heads': sorted([list(h) for h in syc_keys & lie_keys]),
            }
        )

    syc_vals = np.array([syc_deltas[k] for k in sorted(syc_deltas.keys())])
    lie_vals = np.array([lie_deltas[k] for k in sorted(lie_deltas.keys())])
    corr = rank_correlation(syc_vals, lie_vals)

    k_verdict = next(o for o in overlap_results if o['K'] == _VERDICT_K)
    if k_verdict['ratio'] > _SHARED_RATIO and k_verdict['p_hypergeometric'] < _SHARED_P:
        verdict = 'SHARED_CIRCUIT'
    elif k_verdict['ratio'] > _PARTIAL_RATIO and k_verdict['p_hypergeometric'] < _PARTIAL_P:
        verdict = 'PARTIAL_OVERLAP'
    else:
        verdict = 'SEPARATE_CIRCUITS'

    result = {
        'model': ctx.model_name,
        'n_layers': ctx.info.n_layers,
        'n_heads': ctx.info.n_heads,
        'total_heads': ctx.info.total_heads,
        'verdict': verdict,
        'rank_correlation': corr,
        'overlap_by_K': overlap_results,
        'syc_top15': [{'layer': l, 'head': h, 'delta': syc_deltas[(l, h)]} for (l, h), _ in syc_ranked[:DEFAULT_TOP_K]],
        'lie_top15': [{'layer': l, 'head': h, 'delta': lie_deltas[(l, h)]} for (l, h), _ in lie_ranked[:DEFAULT_TOP_K]],
    }
    save_results(result, 'circuit_overlap', ctx.model_name)
    return result


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--models', nargs='+', default=list(ALL_MODELS))
    parser.add_argument('--n-prompts', type=int, default=DEFAULT_N_PROMPTS)
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PAIRS)


def from_args(args: argparse.Namespace) -> CircuitOverlapConfig:
    """Build the validated config from a parsed argparse namespace."""
    return CircuitOverlapConfig(models=tuple(args.models), n_prompts=args.n_prompts, n_pairs=args.n_pairs)
