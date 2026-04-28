"""Causal ablation: zero-out shared heads and measure probe AUROC + direction cosine."""

import argparse
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.config import RANDOM_SEED
from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, load_results, model_session, save_results
from shared_circuits.extraction import extract_residual_with_ablation
from shared_circuits.prompts import build_lying_prompts, build_sycophancy_prompts
from shared_circuits.stats import cosine_similarity, train_probe

_DEFAULT_MODELS: Final[tuple[str, ...]] = (
    'gemma-2-2b-it',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen3-8B',
)
_DEFAULT_N_PAIRS: Final = 400
_DEFAULT_PROBE_PROMPTS: Final = 100
_DEFAULT_N_RANDOM: Final = 5
_DEFAULT_PROBE_LAYER_FRAC: Final = 0.85
_DEFAULT_VERDICT_K: Final = 15
_CAUSAL_DROP: Final = 0.05


class CausalAblationConfig(BaseModel):
    """Inputs for the causal-ablation analysis."""

    model_config = ConfigDict(frozen=True)

    models: tuple[str, ...] = Field(default_factory=lambda: _DEFAULT_MODELS)
    n_pairs: int = Field(default=_DEFAULT_N_PAIRS, gt=0)
    n_probe_prompts: int = Field(default=_DEFAULT_PROBE_PROMPTS, gt=0)
    n_random_heads: int = Field(default=_DEFAULT_N_RANDOM, ge=0)
    probe_layer_frac: float = Field(default=_DEFAULT_PROBE_LAYER_FRAC, gt=0, lt=1)
    shared_heads_from: str = Field(default='circuit_overlap')
    shared_heads_k: int = Field(default=_DEFAULT_VERDICT_K, gt=0)


def run(cfg: CausalAblationConfig) -> list[dict]:
    """Run causal-ablation analysis for every model in ``cfg.models``."""
    pairs = load_triviaqa_pairs(cfg.n_pairs)
    return [_run_one(name, pairs, cfg) for name in cfg.models]


def _run_one(model_name: str, pairs: list[tuple[str, str, str]], cfg: CausalAblationConfig) -> dict:
    overlap_data = load_results(cfg.shared_heads_from, model_name)
    bucket = next(o for o in overlap_data['overlap_by_K'] if o['K'] == cfg.shared_heads_k)
    shared_heads: list[tuple[int, int]] = [tuple(h) for h in bucket['shared_heads']]
    with model_session(model_name) as ctx:
        return _analyse(ctx, pairs, shared_heads, cfg)


def _analyse(
    ctx: ExperimentContext,
    pairs: list[tuple[str, str, str]],
    shared_heads: list[tuple[int, int]],
    cfg: CausalAblationConfig,
) -> dict:
    set_a, set_b = pairs[:200], pairs[200:400]
    wrong_a, correct_a = build_sycophancy_prompts(set_a, ctx.model_name)
    false_b, true_b = build_lying_prompts(set_b, ctx.model_name)
    probe_layer = int(ctx.info.n_layers * cfg.probe_layer_frac)

    rng = np.random.RandomState(RANDOM_SEED)
    all_heads = [(l, h) for l in range(ctx.info.n_layers) for h in range(ctx.info.n_heads)]
    shared_set = set(shared_heads)
    non_shared = [h for h in all_heads if h not in shared_set]
    n_random = min(cfg.n_random_heads, len(non_shared))
    random_heads = [non_shared[i] for i in rng.choice(len(non_shared), n_random, replace=False)] if n_random > 0 else []

    conditions: dict[str, list[tuple[int, int]] | None] = {
        'no_ablation': None,
        'ablate_top5_shared': shared_heads[:5],
        'ablate_top10_shared': shared_heads[:10],
        'ablate_5_random': random_heads,
    }

    results: dict = {}
    n = cfg.n_probe_prompts
    for cond_name, ablate_heads in conditions.items():
        acts_w = extract_residual_with_ablation(ctx.model, wrong_a[:n], probe_layer, ablate_heads)
        acts_c = extract_residual_with_ablation(ctx.model, correct_a[:n], probe_layer, ablate_heads)
        acts_f = extract_residual_with_ablation(ctx.model, false_b[:n], probe_layer, ablate_heads)
        acts_t = extract_residual_with_ablation(ctx.model, true_b[:n], probe_layer, ablate_heads)

        syc_probe = train_probe(acts_w, acts_c)
        lie_probe = train_probe(acts_f, acts_t)
        syc_dir = acts_w.mean(0) - acts_c.mean(0)
        lie_dir = acts_f.mean(0) - acts_t.mean(0)
        cos = cosine_similarity(syc_dir, lie_dir)

        results[cond_name] = {
            'ablated_heads': [list(h) for h in ablate_heads] if ablate_heads else None,
            'syc_probe_auroc': syc_probe['auroc'],
            'lie_probe_auroc': lie_probe['auroc'],
            'cosine': cos,
        }

    baseline = results['no_ablation']
    ablated5 = results['ablate_top5_shared']
    syc_drop = baseline['syc_probe_auroc'] - ablated5['syc_probe_auroc']
    lie_drop = baseline['lie_probe_auroc'] - ablated5['lie_probe_auroc']

    if syc_drop > _CAUSAL_DROP and lie_drop > _CAUSAL_DROP:
        verdict = 'CAUSAL_SHARED'
    elif syc_drop > _CAUSAL_DROP or lie_drop > _CAUSAL_DROP:
        verdict = 'PARTIAL_CAUSAL'
    else:
        verdict = 'NOT_CAUSAL'

    results['verdict'] = verdict
    results['model'] = ctx.model_name
    results['probe_layer'] = probe_layer

    save_results(results, 'causal_ablation', ctx.model_name)
    return results


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--models', nargs='+', default=list(_DEFAULT_MODELS))
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PAIRS)
    parser.add_argument('--n-probe-prompts', type=int, default=_DEFAULT_PROBE_PROMPTS)
    parser.add_argument('--n-random-heads', type=int, default=_DEFAULT_N_RANDOM)
    parser.add_argument('--probe-layer-frac', type=float, default=_DEFAULT_PROBE_LAYER_FRAC)
    parser.add_argument(
        '--shared-heads-from',
        default='circuit_overlap',
        help='Experiment slug whose saved results provide the shared-heads list.',
    )
    parser.add_argument('--shared-heads-k', type=int, default=_DEFAULT_VERDICT_K)


def from_args(args: argparse.Namespace) -> CausalAblationConfig:
    """Build the validated config from a parsed argparse namespace."""
    return CausalAblationConfig(
        models=tuple(args.models),
        n_pairs=args.n_pairs,
        n_probe_prompts=args.n_probe_prompts,
        n_random_heads=args.n_random_heads,
        probe_layer_frac=args.probe_layer_frac,
        shared_heads_from=args.shared_heads_from,
        shared_heads_k=args.shared_heads_k,
    )
