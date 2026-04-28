"""Triple-intersection permutation test: opinion ∩ sycophancy ∩ lying top-K head overlap."""

import argparse
import math
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.config import RANDOM_SEED
from shared_circuits.experiment import load_results, save_results

_DEFAULT_MODELS: Final[tuple[str, ...]] = (
    'gemma-2-2b-it',
    'google/gemma-2-9b-it',
    'google/gemma-2-27b-it',
    'Qwen/Qwen3-8B',
    'meta-llama/Llama-3.1-70B-Instruct',
)
_DEFAULT_N_PERM: Final = 10000


class TripleIntersectionConfig(BaseModel):
    """Inputs for the triple-intersection permutation test."""

    model_config = ConfigDict(frozen=True)

    models: tuple[str, ...] = Field(default_factory=lambda: _DEFAULT_MODELS)
    n_permutations: int = Field(default=_DEFAULT_N_PERM, gt=0)
    seed: int = Field(default=RANDOM_SEED)
    factual_from: str = Field(
        default='breadth',
        description='Slug providing syc_grid + lie_grid (e.g. "breadth" or "circuit_overlap").',
    )
    opinion_from: str = Field(
        default='opinion_circuit_transfer',
        description='Slug providing opinion_grid.',
    )


def run(cfg: TripleIntersectionConfig) -> dict:
    """Run the triple-intersection permutation test for every model in ``cfg.models``."""
    by_model: dict[str, dict] = {}
    for name in cfg.models:
        try:
            r = _analyse(name, cfg)
        except (FileNotFoundError, KeyError):
            continue
        by_model[name] = r
    result = {
        'n_perm': cfg.n_permutations,
        'seed': cfg.seed,
        'factual_from': cfg.factual_from,
        'opinion_from': cfg.opinion_from,
        'by_model': by_model,
    }
    save_results(result, 'triple_intersection_perm', 'all_models')
    return result


def _extract_factual_grids(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pull ``syc_grid`` and ``lie_grid`` out of either a top-level or ``head_overlap`` shape."""
    if 'syc_grid' in payload and 'lie_grid' in payload:
        return np.array(payload['syc_grid']), np.array(payload['lie_grid'])
    if 'head_overlap' in payload and 'syc_grid' in payload['head_overlap']:
        return np.array(payload['head_overlap']['syc_grid']), np.array(payload['head_overlap']['lie_grid'])
    raise KeyError('payload missing syc_grid/lie_grid')


def _extract_opinion_grid(payload: dict) -> np.ndarray:
    if 'opinion_grid' in payload:
        return np.array(payload['opinion_grid'])
    raise KeyError('payload missing opinion_grid')


def _triple_intersection(op_flat: np.ndarray, syc_flat: np.ndarray, lie_flat: np.ndarray, k: int) -> int:
    op_top = set(np.argsort(op_flat)[::-1][:k].tolist())
    syc_top = set(np.argsort(syc_flat)[::-1][:k].tolist())
    lie_top = set(np.argsort(lie_flat)[::-1][:k].tolist())
    return len(op_top & syc_top & lie_top)


def _analyse(model_name: str, cfg: TripleIntersectionConfig) -> dict:
    factual_payload = load_results(cfg.factual_from, model_name)
    opinion_payload = load_results(cfg.opinion_from, model_name)
    syc_grid, lie_grid = _extract_factual_grids(factual_payload)
    op_grid = _extract_opinion_grid(opinion_payload)

    if syc_grid.shape != lie_grid.shape or syc_grid.shape != op_grid.shape:
        raise KeyError(
            f'grid shape mismatch for {model_name}: syc={syc_grid.shape} lie={lie_grid.shape} op={op_grid.shape}'
        )

    n_layers, n_heads = syc_grid.shape
    total = n_layers * n_heads
    # Match the legacy convention K = ceil(sqrt(total)) so the analytic chance K^3/N^2 is interpretable.
    k = math.ceil(math.sqrt(total))
    op_flat, syc_flat, lie_flat = op_grid.flatten(), syc_grid.flatten(), lie_grid.flatten()
    actual = _triple_intersection(op_flat, syc_flat, lie_flat, k)
    analytic_chance = k**3 / total**2

    rng = np.random.RandomState(cfg.seed)
    null_counts: list[int] = []
    ge = 0
    for _ in range(cfg.n_permutations):
        perm_op = rng.permutation(total)
        perm_syc = rng.permutation(total)
        perm_lie = rng.permutation(total)
        null_int = _triple_intersection(op_flat[perm_op], syc_flat[perm_syc], lie_flat[perm_lie], k)
        null_counts.append(null_int)
        if null_int >= actual:
            ge += 1
    p = (ge + 1) / (cfg.n_permutations + 1)

    return {
        'model': model_name,
        'n_layers': int(n_layers),
        'n_heads': int(n_heads),
        'total_heads': int(total),
        'k': k,
        'actual_triple_overlap': int(actual),
        'analytic_chance': float(analytic_chance),
        'permutation_mean': float(np.mean(null_counts)),
        'permutation_p95': float(np.percentile(null_counts, 95)),
        'permutation_p_value': float(p),
        'ratio_vs_chance': float(actual / max(analytic_chance, 1e-10)),
    }


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--models', nargs='+', default=list(_DEFAULT_MODELS))
    parser.add_argument('--n-permutations', type=int, default=_DEFAULT_N_PERM)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument(
        '--factual-from',
        default='breadth',
        help='Experiment slug whose saved results contain syc_grid + lie_grid.',
    )
    parser.add_argument(
        '--opinion-from',
        default='opinion_circuit_transfer',
        help='Experiment slug whose saved results contain opinion_grid.',
    )


def from_args(args: argparse.Namespace) -> TripleIntersectionConfig:
    """Build the validated config from a parsed argparse namespace."""
    return TripleIntersectionConfig(
        models=tuple(args.models),
        n_permutations=args.n_permutations,
        seed=args.seed,
        factual_from=args.factual_from,
        opinion_from=args.opinion_from,
    )
