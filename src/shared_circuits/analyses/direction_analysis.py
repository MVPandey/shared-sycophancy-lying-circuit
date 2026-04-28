"""Direction analysis: cosine similarity between sycophancy and lying mean-difference directions."""

from __future__ import annotations

import argparse
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.config import ALL_MODELS, PERMUTATION_ITERATIONS, RANDOM_SEED
from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, model_session, save_results
from shared_circuits.extraction import extract_residual_stream_multi
from shared_circuits.prompts import build_lying_prompts, build_sycophancy_prompts
from shared_circuits.stats import cosine_similarity

_DEFAULT_N_PAIRS: Final = 400
_DEFAULT_N_PROMPTS: Final = 50
# Late-layer fraction threshold: layers >= 2/3 of depth count as "late" for the null pool.
_LATE_LAYER_FRAC: Final = 2 / 3


class DirectionAnalysisConfig(BaseModel):
    """Inputs for the direction-analysis."""

    model_config = ConfigDict(frozen=True)

    models: tuple[str, ...] = Field(default_factory=lambda: tuple(ALL_MODELS))
    n_pairs: int = Field(default=_DEFAULT_N_PAIRS, gt=0)
    n_prompts: int = Field(default=_DEFAULT_N_PROMPTS, gt=0)
    n_permutations: int = Field(default=PERMUTATION_ITERATIONS, gt=0)
    seed: int = Field(default=RANDOM_SEED)


def run(cfg: DirectionAnalysisConfig) -> list[dict]:
    """Run direction analysis for every model in ``cfg.models``."""
    pairs = load_triviaqa_pairs(cfg.n_pairs)
    return [_run_one(name, pairs, cfg) for name in cfg.models]


def _run_one(model_name: str, pairs: list[tuple[str, str, str]], cfg: DirectionAnalysisConfig) -> dict:
    with model_session(model_name) as ctx:
        return _analyse(ctx, pairs, cfg)


def _analyse(ctx: ExperimentContext, pairs: list[tuple[str, str, str]], cfg: DirectionAnalysisConfig) -> dict:
    set_a, set_b = pairs[:200], pairs[200:400]
    wrong_a, correct_a = build_sycophancy_prompts(set_a, ctx.model_name)
    false_b, true_b = build_lying_prompts(set_b, ctx.model_name)

    # Sample every 2 layers + always include the final layer to bound cost while covering depth.
    target_layers = sorted({*range(0, ctx.info.n_layers, 2), ctx.info.n_layers - 1})
    n = cfg.n_prompts
    acts_w = extract_residual_stream_multi(ctx.model, wrong_a[:n], target_layers)
    acts_c = extract_residual_stream_multi(ctx.model, correct_a[:n], target_layers)
    acts_f = extract_residual_stream_multi(ctx.model, false_b[:n], target_layers)
    acts_t = extract_residual_stream_multi(ctx.model, true_b[:n], target_layers)

    layer_results: dict[str, dict[str, float]] = {}
    for layer in target_layers:
        syc_dir = acts_w[layer].mean(0) - acts_c[layer].mean(0)
        lie_dir = acts_f[layer].mean(0) - acts_t[layer].mean(0)
        layer_results[str(layer)] = {
            'cosine': cosine_similarity(syc_dir, lie_dir),
            'pct_depth': layer / ctx.info.n_layers * 100,
        }

    rng = np.random.RandomState(cfg.seed)
    late_layers = [layer for layer in target_layers if layer >= int(ctx.info.n_layers * _LATE_LAYER_FRAC)]
    null_cosines: list[float] = []
    for _ in range(cfg.n_permutations):
        for layer in late_layers:
            combined = np.concatenate([acts_w[layer], acts_c[layer]], axis=0)
            n_wrong = acts_w[layer].shape[0]
            perm = rng.permutation(combined.shape[0])
            shuffled_dir = combined[perm[:n_wrong]].mean(0) - combined[perm[n_wrong:]].mean(0)
            lie_dir = acts_f[layer].mean(0) - acts_t[layer].mean(0)
            null_cosines.append(cosine_similarity(shuffled_dir, lie_dir))

    null_95 = float(np.percentile(null_cosines, 95)) if null_cosines else 0.0
    late_cos = float(np.mean([layer_results[str(layer)]['cosine'] for layer in late_layers])) if late_layers else 0.0
    margin = late_cos - null_95

    result = {
        'model': ctx.model_name,
        'n_layers': ctx.info.n_layers,
        'target_layers': list(target_layers),
        'late_layers': list(late_layers),
        'layer_cosines': layer_results,
        'late_layer_mean_cosine': late_cos,
        'null_95th_percentile': null_95,
        'margin': margin,
    }
    save_results(result, 'direction_analysis', ctx.model_name)
    return result


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--models', nargs='+', default=list(ALL_MODELS))
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PAIRS)
    parser.add_argument('--n-prompts', type=int, default=_DEFAULT_N_PROMPTS)
    parser.add_argument('--n-permutations', type=int, default=PERMUTATION_ITERATIONS)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)


def from_args(args: argparse.Namespace) -> DirectionAnalysisConfig:
    """Build the validated config from a parsed argparse namespace."""
    return DirectionAnalysisConfig(
        models=tuple(args.models),
        n_pairs=args.n_pairs,
        n_prompts=args.n_prompts,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
