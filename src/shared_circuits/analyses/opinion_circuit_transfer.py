"""Opinion-paradigm head importance with permutation-null overlap against sycophancy ranking."""

import argparse
import math
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import pearsonr, spearmanr

from shared_circuits.attribution import compute_head_importance_grid, compute_head_importances, rank_heads
from shared_circuits.config import DEFAULT_BATCH_SIZE, DEFAULT_TOP_K, RANDOM_SEED
from shared_circuits.data import generate_opinion_pairs
from shared_circuits.experiment import ExperimentContext, load_results, model_session, save_results
from shared_circuits.prompts import build_opinion_prompts

_DEFAULT_MODELS: Final[tuple[str, ...]] = (
    'gemma-2-2b-it',
    'google/gemma-2-9b-it',
    'Qwen/Qwen3-8B',
    'meta-llama/Llama-3.1-8B-Instruct',
    'meta-llama/Llama-3.1-70B-Instruct',
)
_DEFAULT_N_OPINION: Final = 200
_DEFAULT_DLA_PROMPTS: Final = 100
_DEFAULT_PERMUTATIONS: Final = 10000


class OpinionCircuitTransferConfig(BaseModel):
    """Inputs for the opinion-circuit-transfer analysis."""

    model_config = ConfigDict(frozen=True)

    models: tuple[str, ...] = Field(default_factory=lambda: _DEFAULT_MODELS)
    n_opinion: int = Field(default=_DEFAULT_N_OPINION, gt=0)
    dla_prompts: int = Field(default=_DEFAULT_DLA_PROMPTS, gt=0)
    permutations: int = Field(default=_DEFAULT_PERMUTATIONS, gt=0)
    batch: int = Field(default=DEFAULT_BATCH_SIZE, gt=0)
    seed: int = Field(default=RANDOM_SEED)
    n_devices: int = Field(default=1, gt=0)
    top_k: int = Field(default=DEFAULT_TOP_K, gt=0)
    syc_from: str = Field(
        default='circuit_overlap',
        description='Slug whose saved results provide the syc head ranking (e.g. "circuit_overlap" or "breadth").',
    )


def run(cfg: OpinionCircuitTransferConfig) -> list[dict]:
    """Run opinion-circuit-transfer for every model in ``cfg.models``."""
    return [_run_one(name, cfg) for name in cfg.models]


def _run_one(model_name: str, cfg: OpinionCircuitTransferConfig) -> dict:
    with model_session(model_name, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, cfg)


def _load_syc_grid(model_name: str, source: str, n_layers: int, n_heads: int) -> np.ndarray | None:
    """Pull a (n_layers, n_heads) syc grid out of a saved upstream JSON, or None if unavailable."""
    try:
        payload = load_results(source, model_name)
    except FileNotFoundError:
        return None
    grid = _grid_from_payload(payload, n_layers, n_heads)
    if grid is not None:
        return grid
    # ``circuit_overlap`` doesn't persist a grid — reconstruct one from its top-K entries.
    if 'syc_top15' in payload:
        rebuilt = np.zeros((n_layers, n_heads))
        for entry in payload['syc_top15']:
            rebuilt[int(entry['layer']), int(entry['head'])] = float(entry['delta'])
        return rebuilt
    return None


def _grid_from_payload(payload: dict, n_layers: int, n_heads: int) -> np.ndarray | None:
    """Locate ``syc_grid`` at the top level or under ``head_overlap``."""
    if 'syc_grid' in payload:
        grid = np.array(payload['syc_grid'])
        if grid.shape == (n_layers, n_heads):
            return grid
    nested = payload.get('head_overlap')
    if isinstance(nested, dict) and 'syc_grid' in nested:
        grid = np.array(nested['syc_grid'])
        if grid.shape == (n_layers, n_heads):
            return grid
    return None


def _overlap_pvalue(
    ref_flat: np.ndarray,
    query_flat: np.ndarray,
    k: int,
    n_perm: int,
    seed: int,
) -> tuple[int, float]:
    """Permutation null on the query grid: how often does a random shuffle beat the observed overlap."""
    n = len(ref_flat)
    ref_top = set(np.argsort(ref_flat)[::-1][:k].tolist())
    query_top = set(np.argsort(query_flat)[::-1][:k].tolist())
    actual = len(ref_top & query_top)
    rng = np.random.RandomState(seed)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        null_top = set(np.argsort(query_flat[perm])[::-1][:k].tolist())
        if len(ref_top & null_top) >= actual:
            ge += 1
    return actual, (ge + 1) / (n_perm + 1)


def _analyse(ctx: ExperimentContext, cfg: OpinionCircuitTransferConfig) -> dict:
    opinion_pairs = generate_opinion_pairs(cfg.n_opinion, seed=cfg.seed)
    opinion_a, opinion_b = build_opinion_prompts(opinion_pairs, ctx.model_name)
    op_deltas = compute_head_importances(
        ctx.model,
        opinion_a,
        opinion_b,
        n_prompts=cfg.dla_prompts,
        batch_size=cfg.batch,
    )
    op_grid = compute_head_importance_grid(op_deltas, ctx.info.n_layers, ctx.info.n_heads)
    op_ranked = rank_heads(op_deltas, top_k=ctx.info.total_heads)

    total = ctx.info.total_heads
    # Match the legacy K = ceil(sqrt(total)) so the analytic chance K^2/N is interpretable.
    k = math.ceil(math.sqrt(total))
    chance = k * k / total

    syc_grid = _load_syc_grid(ctx.model_name, cfg.syc_from, ctx.info.n_layers, ctx.info.n_heads)
    overlap: dict[str, float | int | bool | str] = {'available': False, 'source': cfg.syc_from}
    if syc_grid is not None:
        overlap_count, p_perm = _overlap_pvalue(
            syc_grid.flatten(),
            op_grid.flatten(),
            k,
            cfg.permutations,
            cfg.seed,
        )
        rho_syc, _ = spearmanr(op_grid.flatten(), syc_grid.flatten())
        r_syc, _ = pearsonr(op_grid.flatten(), syc_grid.flatten())
        overlap = {
            'available': True,
            'source': cfg.syc_from,
            'overlap': int(overlap_count),
            'ratio_vs_chance': float(overlap_count / chance) if chance > 0 else 0.0,
            'p_permutation': float(p_perm),
            'spearman': float(rho_syc),
            'pearson': float(r_syc),
        }

    op_top = set(np.argsort(op_grid.flatten())[::-1][:k].tolist())

    result = {
        'model': ctx.model_name,
        'n_layers': ctx.info.n_layers,
        'n_heads': ctx.info.n_heads,
        'total_heads': total,
        'k': int(k),
        'chance': float(chance),
        'n_opinion_pairs': len(opinion_pairs),
        'n_dla_prompts': cfg.dla_prompts,
        'opinion_grid': op_grid.tolist(),
        'opinion_top15': [
            {'layer': layer, 'head': head, 'delta': op_deltas[(layer, head)]}
            for (layer, head), _ in op_ranked[: cfg.top_k]
        ],
        'shared_heads': [[int(idx // ctx.info.n_heads), int(idx % ctx.info.n_heads)] for idx in sorted(op_top)],
        'overlap_with_syc': overlap,
        'p_perm': overlap['p_permutation'] if overlap['available'] else None,
    }
    save_results(result, 'opinion_circuit_transfer', ctx.model_name)
    return result


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--models', nargs='+', default=list(_DEFAULT_MODELS))
    parser.add_argument('--n-opinion', type=int, default=_DEFAULT_N_OPINION)
    parser.add_argument('--dla-prompts', type=int, default=_DEFAULT_DLA_PROMPTS)
    parser.add_argument('--permutations', type=int, default=_DEFAULT_PERMUTATIONS)
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        '--syc-from',
        default='circuit_overlap',
        help='Experiment slug whose saved results provide the sycophancy head ranking.',
    )


def from_args(args: argparse.Namespace) -> OpinionCircuitTransferConfig:
    """Build the validated config from a parsed argparse namespace."""
    return OpinionCircuitTransferConfig(
        models=tuple(args.models),
        n_opinion=args.n_opinion,
        dla_prompts=args.dla_prompts,
        permutations=args.permutations,
        batch=args.batch,
        seed=args.seed,
        n_devices=args.n_devices,
        top_k=args.top_k,
        syc_from=args.syc_from,
    )
