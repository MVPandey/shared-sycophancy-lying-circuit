"""Cross-dataset replication: head-overlap on NaturalQuestions vs TriviaQA."""

from __future__ import annotations

import argparse
import math
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import pearsonr, spearmanr

from shared_circuits.attribution import compute_head_importance_grid, compute_head_importances
from shared_circuits.config import RANDOM_SEED
from shared_circuits.data import load_naturalquestions_pairs
from shared_circuits.experiment import ExperimentContext, load_results, model_session, save_results
from shared_circuits.prompts import build_lying_prompts, build_sycophancy_prompts

_DEFAULT_MODELS: Final[tuple[str, ...]] = (
    'gemma-2-2b-it',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
)
_DEFAULT_N_PAIRS: Final = 200
_DEFAULT_DLA_PROMPTS: Final = 100
_DEFAULT_PERMUTATIONS: Final = 1000
_DEFAULT_BATCH: Final = 4


class NqReplicationConfig(BaseModel):
    """Inputs for the NQ-replication analysis."""

    model_config = ConfigDict(frozen=True)

    models: tuple[str, ...] = Field(default_factory=lambda: _DEFAULT_MODELS)
    n_devices: int = Field(default=1, gt=0)
    n_pairs: int = Field(default=_DEFAULT_N_PAIRS, gt=0)
    dla_prompts: int = Field(default=_DEFAULT_DLA_PROMPTS, gt=0)
    permutations: int = Field(default=_DEFAULT_PERMUTATIONS, gt=0)
    batch: int = Field(default=_DEFAULT_BATCH, gt=0)
    seed: int = Field(default=RANDOM_SEED)
    triviaqa_grids_from: str = Field(default='breadth')


def run(cfg: NqReplicationConfig) -> list[dict]:
    """Run NQ replication for every model in ``cfg.models``."""
    pairs = load_naturalquestions_pairs(cfg.n_pairs)
    return [_run_one(name, pairs, cfg) for name in cfg.models]


def _run_one(model_name: str, pairs: list[tuple[str, str, str]], cfg: NqReplicationConfig) -> dict:
    with model_session(model_name, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, pairs, cfg)


def _analyse(ctx: ExperimentContext, pairs: list[tuple[str, str, str]], cfg: NqReplicationConfig) -> dict:
    syc_pairs = pairs[: cfg.dla_prompts]
    lie_pairs = pairs[cfg.dla_prompts : cfg.dla_prompts * 2]
    wrong, correct = build_sycophancy_prompts(syc_pairs, ctx.model_name)
    false_p, true_p = build_lying_prompts(lie_pairs, ctx.model_name)

    syc_deltas = compute_head_importances(ctx.model, wrong, correct, n_prompts=cfg.dla_prompts, batch_size=cfg.batch)
    lie_deltas = compute_head_importances(ctx.model, false_p, true_p, n_prompts=cfg.dla_prompts, batch_size=cfg.batch)
    syc_grid = compute_head_importance_grid(syc_deltas, ctx.info.n_layers, ctx.info.n_heads)
    lie_grid = compute_head_importance_grid(lie_deltas, ctx.info.n_layers, ctx.info.n_heads)

    total = ctx.info.total_heads
    # Sqrt-of-total K matches the legacy script and the breadth analysis.
    k = math.ceil(math.sqrt(total))
    nq_sf = syc_grid.flatten()
    nq_lf = lie_grid.flatten()
    chance = k * k / total

    nq_overlap, nq_p = _overlap_pvalue(nq_sf, nq_lf, k, cfg.permutations, cfg.seed)
    nq_rho, _ = spearmanr(nq_sf, nq_lf)

    cross = _cross_dataset(ctx.model_name, nq_sf, nq_lf, k, total, cfg)

    result = {
        'model': ctx.model_name,
        'dataset': 'nq_open',
        'n_layers': ctx.info.n_layers,
        'n_heads': ctx.info.n_heads,
        'total_heads': total,
        'k': k,
        'chance': float(chance),
        'nq_within_dataset': {
            'syc_lie_overlap': int(nq_overlap),
            'syc_lie_ratio': float(nq_overlap / chance) if chance > 0 else 0.0,
            'syc_lie_p': float(nq_p),
            'syc_lie_rho': float(nq_rho),
        },
        # Top-level conveniences mirroring the spec: per-model summary metrics.
        'nq_overlap': int(nq_overlap),
        'nq_ratio': float(nq_overlap / chance) if chance > 0 else 0.0,
        'tqa_nq_pearson_syc': cross.get('syc_pearson'),
        'tqa_nq_pearson_lie': cross.get('lie_pearson'),
        'cross_dataset_triviaqa_vs_nq': cross,
        'nq_syc_grid': syc_grid.tolist(),
        'nq_lie_grid': lie_grid.tolist(),
    }
    save_results(result, 'nq_replication', ctx.model_name)
    return result


def _overlap_pvalue(
    ref_flat: np.ndarray,
    query_flat: np.ndarray,
    k: int,
    n_perm: int,
    seed: int,
) -> tuple[int, float]:
    ref_top = set(np.argsort(ref_flat)[::-1][:k].tolist())
    query_top = set(np.argsort(query_flat)[::-1][:k].tolist())
    actual = len(ref_top & query_top)
    rng = np.random.RandomState(seed)
    n = len(ref_flat)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        null_top = set(np.argsort(query_flat[perm])[::-1][:k].tolist())
        if len(ref_top & null_top) >= actual:
            ge += 1
    return actual, (ge + 1) / (n_perm + 1)


def _cross_dataset(
    model_name: str,
    nq_sf: np.ndarray,
    nq_lf: np.ndarray,
    k: int,
    total: int,
    cfg: NqReplicationConfig,
) -> dict:
    grids = _load_triviaqa_grids(model_name, cfg.triviaqa_grids_from)
    if grids is None:
        return {'available': False, 'source': cfg.triviaqa_grids_from}
    tqa_sf, tqa_lf = grids
    chance = k * k / total
    syc_overlap, syc_p = _overlap_pvalue(tqa_sf, nq_sf, k, cfg.permutations, cfg.seed + 1)
    syc_rho, _ = spearmanr(tqa_sf, nq_sf)
    syc_pearson, _ = pearsonr(tqa_sf, nq_sf)
    lie_overlap, lie_p = _overlap_pvalue(tqa_lf, nq_lf, k, cfg.permutations, cfg.seed + 2)
    lie_rho, _ = spearmanr(tqa_lf, nq_lf)
    lie_pearson, _ = pearsonr(tqa_lf, nq_lf)
    return {
        'available': True,
        'source': cfg.triviaqa_grids_from,
        'syc_overlap': int(syc_overlap),
        'syc_ratio': float(syc_overlap / chance) if chance > 0 else 0.0,
        'syc_p': float(syc_p),
        'syc_rho': float(syc_rho),
        'syc_pearson': float(syc_pearson),
        'lie_overlap': int(lie_overlap),
        'lie_ratio': float(lie_overlap / chance) if chance > 0 else 0.0,
        'lie_p': float(lie_p),
        'lie_rho': float(lie_rho),
        'lie_pearson': float(lie_pearson),
    }


def _load_triviaqa_grids(model_name: str, source: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(syc_grid_flat, lie_grid_flat)`` from saved TriviaQA results, or None."""
    try:
        data = load_results(source, model_name)
    except FileNotFoundError:
        return None
    # ``breadth`` nests grids under ``head_overlap``; older outputs put them at the top level.
    if 'syc_grid' in data and 'lie_grid' in data:
        return np.array(data['syc_grid']).flatten(), np.array(data['lie_grid']).flatten()
    nested = data.get('head_overlap')
    if isinstance(nested, dict) and 'syc_grid' in nested and 'lie_grid' in nested:
        return np.array(nested['syc_grid']).flatten(), np.array(nested['lie_grid']).flatten()
    return None


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--models', nargs='+', default=list(_DEFAULT_MODELS))
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PAIRS)
    parser.add_argument('--dla-prompts', type=int, default=_DEFAULT_DLA_PROMPTS)
    parser.add_argument('--permutations', type=int, default=_DEFAULT_PERMUTATIONS)
    parser.add_argument('--batch', type=int, default=_DEFAULT_BATCH)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument(
        '--triviaqa-grids-from',
        default='breadth',
        help='Experiment slug whose saved TriviaQA grids drive the cross-dataset comparison.',
    )


def from_args(args: argparse.Namespace) -> NqReplicationConfig:
    """Build the validated config from a parsed argparse namespace."""
    return NqReplicationConfig(
        models=tuple(args.models),
        n_devices=args.n_devices,
        n_pairs=args.n_pairs,
        dla_prompts=args.dla_prompts,
        permutations=args.permutations,
        batch=args.batch,
        seed=args.seed,
        triviaqa_grids_from=args.triviaqa_grids_from,
    )
