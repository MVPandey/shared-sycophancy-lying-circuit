"""Attribution patching: causal validation of DLA-identified heads."""

import argparse
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import pearsonr

from shared_circuits.attribution import compute_attribution_patching
from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, load_results, model_session, save_results
from shared_circuits.prompts import build_lying_prompts, build_sycophancy_prompts
from shared_circuits.stats import head_overlap_hypergeometric

_DEFAULT_MODELS: Final[tuple[str, ...]] = (
    'gemma-2-2b-it',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
)
_DEFAULT_N_PAIRS_DATA: Final = 400
_DEFAULT_N_PATCH_PAIRS: Final = 30
_DEFAULT_OVERLAP_K: Final = 15


class AttributionPatchingConfig(BaseModel):
    """Inputs for the attribution-patching analysis."""

    model_config = ConfigDict(frozen=True)

    models: tuple[str, ...] = Field(default_factory=lambda: _DEFAULT_MODELS)
    n_pairs: int = Field(default=_DEFAULT_N_PAIRS_DATA, gt=0)
    n_patch_pairs: int = Field(default=_DEFAULT_N_PATCH_PAIRS, gt=0)
    overlap_k: int = Field(default=_DEFAULT_OVERLAP_K, gt=0)


def run(cfg: AttributionPatchingConfig) -> list[dict]:
    """Run attribution-patching analysis for every model in ``cfg.models``."""
    pairs = load_triviaqa_pairs(cfg.n_pairs)
    return [_run_one(name, pairs, cfg) for name in cfg.models]


def _run_one(model_name: str, pairs: list[tuple[str, str, str]], cfg: AttributionPatchingConfig) -> dict:
    with model_session(model_name) as ctx:
        return _analyse(ctx, pairs, cfg)


def _analyse(ctx: ExperimentContext, pairs: list[tuple[str, str, str]], cfg: AttributionPatchingConfig) -> dict:
    set_a, set_b = pairs[:200], pairs[200:400]
    wrong_a, correct_a = build_sycophancy_prompts(set_a, ctx.model_name)
    false_b, true_b = build_lying_prompts(set_b, ctx.model_name)

    syc_effects = compute_attribution_patching(
        ctx.model, wrong_a, correct_a, list(ctx.agree_tokens), list(ctx.disagree_tokens), cfg.n_patch_pairs
    )
    lie_effects = compute_attribution_patching(
        ctx.model, false_b, true_b, list(ctx.agree_tokens), list(ctx.disagree_tokens), cfg.n_patch_pairs
    )

    syc_abs = np.abs(syc_effects).flatten()
    lie_abs = np.abs(lie_effects).flatten()
    r_patch, p_patch = pearsonr(syc_abs, lie_abs)

    k = cfg.overlap_k
    syc_top = set(np.argsort(syc_abs)[::-1][:k].tolist())
    lie_top = set(np.argsort(lie_abs)[::-1][:k].tolist())
    overlap = len(syc_top & lie_top)
    expected = k * k / ctx.info.total_heads
    ratio = overlap / expected if expected > 0 else 0.0
    p_hyper = head_overlap_hypergeometric(overlap, k, ctx.info.total_heads)

    dla_comparison: dict[str, bool] = {}
    try:
        load_results('circuit_overlap', ctx.model_name)
        dla_comparison['available'] = True
    except FileNotFoundError:
        dla_comparison['available'] = False

    result = {
        'model': ctx.model_name,
        'n_layers': ctx.info.n_layers,
        'n_heads': ctx.info.n_heads,
        'n_pairs': cfg.n_patch_pairs,
        'patch_pearson_r': float(r_patch),
        'patch_pearson_p': float(p_patch),
        'patch_k15_overlap': overlap,
        'patch_k15_ratio': float(ratio),
        'patch_k15_p_hypergeom': float(p_hyper),
        'syc_patch_grid': syc_effects.tolist(),
        'lie_patch_grid': lie_effects.tolist(),
        'dla_comparison': dla_comparison,
    }

    save_results(result, 'attribution_patching', ctx.model_name)
    return result


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--models', nargs='+', default=list(_DEFAULT_MODELS))
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PAIRS_DATA)
    parser.add_argument('--n-patch-pairs', type=int, default=_DEFAULT_N_PATCH_PAIRS)
    parser.add_argument('--overlap-k', type=int, default=_DEFAULT_OVERLAP_K)


def from_args(args: argparse.Namespace) -> AttributionPatchingConfig:
    """Build the validated config from a parsed argparse namespace."""
    return AttributionPatchingConfig(
        models=tuple(args.models),
        n_pairs=args.n_pairs,
        n_patch_pairs=args.n_patch_pairs,
        overlap_k=args.overlap_k,
    )
