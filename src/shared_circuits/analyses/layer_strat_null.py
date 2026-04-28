"""Layer-stratified permutation null for sycophancy/lying head-overlap."""

from __future__ import annotations

import argparse
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import hypergeom

from shared_circuits.config import RANDOM_SEED
from shared_circuits.experiment import load_results, save_results

_DEFAULT_MODELS: Final[tuple[str, ...]] = (
    'gemma-2-2b-it',
    'google/gemma-2-9b-it',
    'google/gemma-2-27b-it',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen3-8B',
    'meta-llama/Llama-3.1-8B-Instruct',
    'meta-llama/Llama-3.1-70B-Instruct',
)
_DEFAULT_N_PERM: Final = 10000
# Legacy scripts seeded the unstratified and stratified passes differently to avoid correlated draws.
_STRATIFIED_SEED_OFFSET: Final = 1
# Per-layer, per-head head-importance matrices are 2D (n_layers x n_heads).
_GRID_NDIM: Final = 2


class LayerStratNullConfig(BaseModel):
    """Inputs for the layer-stratified null analysis."""

    model_config = ConfigDict(frozen=True)

    models: tuple[str, ...] = Field(default_factory=lambda: _DEFAULT_MODELS)
    n_permutations: int = Field(default=_DEFAULT_N_PERM, gt=0)
    seed: int = Field(default=RANDOM_SEED)
    grids_from: str = Field(
        default='breadth',
        description='Slug whose saved results contain syc_grid + lie_grid (e.g. "breadth").',
    )
    k: int | None = Field(default=None, description='Top-K for overlap; defaults to ceil(sqrt(total_heads)).')


def run(cfg: LayerStratNullConfig) -> dict:
    """Run layer-stratified permutation null for every model in ``cfg.models``."""
    by_model: list[dict] = []
    for name in cfg.models:
        try:
            r = _analyse(name, cfg)
        except (FileNotFoundError, KeyError):
            continue
        by_model.append(r)
    result = {
        'n_perm': cfg.n_permutations,
        'seed': cfg.seed,
        'grids_from': cfg.grids_from,
        'by_model': by_model,
    }
    save_results(result, 'layer_stratified_null', 'all_models')
    return result


def _extract_grids(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pull ``syc_grid`` and ``lie_grid`` out of either a top-level or ``head_overlap`` shape."""
    if 'head_overlap' in payload and 'syc_grid' in payload['head_overlap']:
        sg = np.array(payload['head_overlap']['syc_grid'])
        lg = np.array(payload['head_overlap']['lie_grid'])
    elif 'syc_grid' in payload and 'lie_grid' in payload:
        sg = np.array(payload['syc_grid'])
        lg = np.array(payload['lie_grid'])
    else:
        raise KeyError('payload missing syc_grid/lie_grid')
    if sg.ndim != _GRID_NDIM or lg.ndim != _GRID_NDIM or sg.shape != lg.shape:
        raise KeyError(f'grid shape mismatch: syc={sg.shape} lie={lg.shape}')
    return sg, lg


def _overlap(a_flat: np.ndarray, b_flat: np.ndarray, k: int) -> int:
    a_top = set(np.argsort(a_flat)[::-1][:k].tolist())
    b_top = set(np.argsort(b_flat)[::-1][:k].tolist())
    return len(a_top & b_top)


def _unstratified_p(
    syc_flat: np.ndarray,
    lie_flat: np.ndarray,
    k: int,
    actual: int,
    rng: np.random.RandomState,
    n_perm: int,
) -> float:
    syc_top = set(np.argsort(syc_flat)[::-1][:k].tolist())
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(lie_flat))
        null_top = set(np.argsort(lie_flat[perm])[::-1][:k].tolist())
        if len(syc_top & null_top) >= actual:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def _stratified_p(
    syc_grid: np.ndarray,
    lie_grid: np.ndarray,
    k: int,
    actual: int,
    rng: np.random.RandomState,
    n_perm: int,
) -> float:
    """Permute head labels *within each layer*, preserving per-layer marginals."""
    n_layers, n_heads = syc_grid.shape
    syc_flat = syc_grid.flatten()
    syc_top = set(np.argsort(syc_flat)[::-1][:k].tolist())
    ge = 0
    for _ in range(n_perm):
        permuted = np.empty_like(lie_grid)
        for layer in range(n_layers):
            perm = rng.permutation(n_heads)
            permuted[layer] = lie_grid[layer][perm]
        null_flat = permuted.flatten()
        null_top = set(np.argsort(null_flat)[::-1][:k].tolist())
        if len(syc_top & null_top) >= actual:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def _analyse(model_name: str, cfg: LayerStratNullConfig) -> dict:
    payload = load_results(cfg.grids_from, model_name)
    syc_grid, lie_grid = _extract_grids(payload)
    n_layers, n_heads = syc_grid.shape
    total = n_layers * n_heads
    # Match the legacy default K = ceil(sqrt(total)); explicit cfg.k overrides for sweeps.
    k = cfg.k if cfg.k is not None else int(np.ceil(np.sqrt(total)))
    syc_flat = syc_grid.flatten()
    lie_flat = lie_grid.flatten()
    actual = _overlap(syc_flat, lie_flat, k)
    chance = k * k / total

    p_hyper = float(hypergeom.sf(actual - 1, total, k, k))
    rng = np.random.RandomState(cfg.seed)
    p_unstrat = _unstratified_p(syc_flat, lie_flat, k, actual, rng, cfg.n_permutations)
    rng = np.random.RandomState(cfg.seed + _STRATIFIED_SEED_OFFSET)
    p_strat = _stratified_p(syc_grid, lie_grid, k, actual, rng, cfg.n_permutations)

    return {
        'model': model_name,
        'source_slug': cfg.grids_from,
        'n_layers': int(n_layers),
        'n_heads': int(n_heads),
        'total_heads': int(total),
        'k': int(k),
        'actual_overlap': int(actual),
        'chance_overlap': float(chance),
        'ratio_vs_chance': float(actual / chance) if chance > 0 else float('inf'),
        'p_hypergeometric': p_hyper,
        'p_permutation_unstratified': float(p_unstrat),
        'p_permutation_layer_stratified': float(p_strat),
    }


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--models', nargs='+', default=list(_DEFAULT_MODELS))
    parser.add_argument('--n-permutations', type=int, default=_DEFAULT_N_PERM)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument(
        '--grids-from',
        default='breadth',
        help='Experiment slug whose saved results contain syc_grid + lie_grid.',
    )
    parser.add_argument('--k', type=int, default=None, help='Top-K for overlap (default: ceil(sqrt(total_heads))).')


def from_args(args: argparse.Namespace) -> LayerStratNullConfig:
    """Build the validated config from a parsed argparse namespace."""
    return LayerStratNullConfig(
        models=tuple(args.models),
        n_permutations=args.n_permutations,
        seed=args.seed,
        grids_from=args.grids_from,
        k=args.k,
    )
