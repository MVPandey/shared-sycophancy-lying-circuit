"""Generic breadth validation: head overlap + behavioral steering for a single model."""

from __future__ import annotations

import argparse
import math
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import pearsonr, spearmanr

from shared_circuits.attribution import compute_head_importance_grid, compute_head_importances
from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, model_session, save_results
from shared_circuits.extraction import extract_residual_stream, measure_agreement_rate
from shared_circuits.prompts import build_lying_prompts, build_sycophancy_prompts

_DEFAULT_N_PAIRS: Final = 400
_DEFAULT_DLA_PROMPTS: Final = 30
_DEFAULT_BASELINE_PROMPTS: Final = 100
_DEFAULT_DIR_PROMPTS: Final = 50
_DEFAULT_STEER_PROMPTS: Final = 50
_DEFAULT_ALPHAS: Final[tuple[int, ...]] = (0, -25, -50, -100, -200)
_DEFAULT_LAYER_FRACS: Final[tuple[float, ...]] = (0.5, 0.6, 0.7, 0.8)
_DEFAULT_PERMUTATIONS: Final = 1000
_DEFAULT_SEED: Final = 42
_DEFAULT_BATCH: Final = 2


class BreadthConfig(BaseModel):
    """Inputs for the breadth analysis (single model)."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(...)
    n_devices: int = Field(default=1, gt=0)
    alphas: tuple[int, ...] = Field(default=_DEFAULT_ALPHAS)
    n_pairs: int = Field(default=_DEFAULT_N_PAIRS, gt=0)
    dla_prompts: int = Field(default=_DEFAULT_DLA_PROMPTS, gt=0)
    baseline_prompts: int = Field(default=_DEFAULT_BASELINE_PROMPTS, gt=0)
    dir_prompts: int = Field(default=_DEFAULT_DIR_PROMPTS, gt=0)
    steer_prompts: int = Field(default=_DEFAULT_STEER_PROMPTS, gt=0)
    layer_fracs: tuple[float, ...] = Field(default=_DEFAULT_LAYER_FRACS)
    permutations: int = Field(default=_DEFAULT_PERMUTATIONS, gt=0)
    seed: int = Field(default=_DEFAULT_SEED)
    batch: int = Field(default=_DEFAULT_BATCH, gt=0)


def run(cfg: BreadthConfig) -> dict:
    """Run breadth validation on ``cfg.model``."""
    pairs = load_triviaqa_pairs(cfg.n_pairs)
    with model_session(cfg.model, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, pairs, cfg)


def _analyse(ctx: ExperimentContext, pairs: list[tuple[str, str, str]], cfg: BreadthConfig) -> dict:
    results: dict = {
        'model': ctx.model_name,
        'config': {
            'n_pairs': cfg.n_pairs,
            'dla_prompts': cfg.dla_prompts,
            'steer_prompts': cfg.steer_prompts,
            'dir_prompts': cfg.dir_prompts,
            'alphas': list(cfg.alphas),
            'layer_fracs': list(cfg.layer_fracs),
            'permutations': cfg.permutations,
            'seed': cfg.seed,
            'batch': cfg.batch,
            'n_devices': cfg.n_devices,
        },
        'n_layers': ctx.info.n_layers,
        'n_heads': ctx.info.n_heads,
        'total_heads': ctx.info.total_heads,
        'd_model': ctx.info.d_model,
    }
    results['head_overlap'] = _head_overlap(ctx, pairs, cfg)
    results['baseline_sycophancy'] = _baseline(ctx, pairs, cfg)
    results['steering'] = _steering_sweep(ctx, pairs, cfg)

    save_results(results, 'breadth', ctx.model_name)
    return results


def _overlap_pvalue(
    syc_flat: np.ndarray,
    lie_flat: np.ndarray,
    k: int,
    n_perm: int,
    seed: int,
) -> tuple[int, float]:
    n = len(syc_flat)
    syc_top = set(np.argsort(syc_flat)[::-1][:k].tolist())
    actual = len(syc_top & set(np.argsort(lie_flat)[::-1][:k].tolist()))
    rng = np.random.RandomState(seed)
    at_least = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        null_top = set(np.argsort(lie_flat[perm])[::-1][:k].tolist())
        if len(syc_top & null_top) >= actual:
            at_least += 1
    return actual, (at_least + 1) / (n_perm + 1)


def _overlap_stats(syc_grid: np.ndarray, lie_grid: np.ndarray, total_heads: int, cfg: BreadthConfig) -> dict:
    sf = syc_grid.flatten()
    lf = lie_grid.flatten()
    r_p, _ = pearsonr(sf, lf)
    r_s, _ = spearmanr(sf, lf)
    k = math.ceil(math.sqrt(total_heads))
    overlap, p = _overlap_pvalue(sf, lf, k, cfg.permutations, cfg.seed)
    chance = k * k / total_heads
    return {
        'k': k,
        'pearson': float(r_p),
        'spearman': float(r_s),
        'top_k_overlap': int(overlap),
        'top_k_chance': float(chance),
        'overlap_ratio': float(overlap / chance) if chance > 0 else float('inf'),
        'p_value': float(p),
    }


def _head_overlap(ctx: ExperimentContext, pairs: list[tuple[str, str, str]], cfg: BreadthConfig) -> dict:
    syc_pairs = pairs[:100]
    lie_pairs = pairs[100:300]
    wrong, correct = build_sycophancy_prompts(syc_pairs, ctx.model_name)
    false_p, true_p = build_lying_prompts(lie_pairs, ctx.model_name)

    syc_deltas = compute_head_importances(ctx.model, wrong, correct, n_prompts=cfg.dla_prompts, batch_size=1)
    lie_deltas = compute_head_importances(ctx.model, false_p, true_p, n_prompts=cfg.dla_prompts, batch_size=1)

    syc_grid = compute_head_importance_grid(syc_deltas, ctx.info.n_layers, ctx.info.n_heads)
    lie_grid = compute_head_importance_grid(lie_deltas, ctx.info.n_layers, ctx.info.n_heads)
    stats = _overlap_stats(syc_grid, lie_grid, ctx.info.total_heads, cfg)
    return {'syc_grid': syc_grid.tolist(), 'lie_grid': lie_grid.tolist(), 'stats': stats}


def _baseline(ctx: ExperimentContext, pairs: list[tuple[str, str, str]], cfg: BreadthConfig) -> float:
    wrong, _ = build_sycophancy_prompts(pairs[100 : 100 + cfg.baseline_prompts], ctx.model_name)
    return measure_agreement_rate(
        ctx.model,
        wrong,
        ctx.agree_tokens,
        ctx.disagree_tokens,
        batch_size=cfg.batch,
    )


def _make_steer_hook(alpha: int, direction: torch.Tensor, seq_lens: list[int]):
    def hook_fn(t: torch.Tensor, hook: object) -> torch.Tensor:
        d = direction.to(t.device, dtype=t.dtype)
        for b in range(t.shape[0]):
            t[b, seq_lens[b]] = t[b, seq_lens[b]] + alpha * d
        return t

    return hook_fn


def _steer_measure(
    model: HookedTransformer,
    prompts: list[str],
    direction: torch.Tensor,
    layer: int,
    alpha: int,
    agree_tokens: tuple[int, ...],
    disagree_tokens: tuple[int, ...],
    batch: int,
) -> float:
    pad_id = getattr(model.tokenizer, 'pad_token_id', None) or 0
    if alpha == 0:
        return measure_agreement_rate(model, prompts, agree_tokens, disagree_tokens, batch_size=batch)
    # the steer hook needs per-batch ``seq_lens``, so we call the per-prompt
    # helper inside our own batching loop and fold the per-batch hook closures
    rate_total = 0
    for i in range(0, len(prompts), batch):
        batch_prompts = prompts[i : i + batch]
        tokens = model.to_tokens(batch_prompts, prepend_bos=True)
        seq_lens = [int(x) for x in ((tokens != pad_id).sum(dim=1) - 1).tolist()]
        hooks = [(f'blocks.{layer}.hook_resid_post', _make_steer_hook(alpha, direction, seq_lens))]
        rate, _ = _logits_to_rate(model, tokens, seq_lens, hooks, agree_tokens, disagree_tokens)
        rate_total += rate * len(batch_prompts)
    return rate_total / len(prompts)


@torch.no_grad()
def _logits_to_rate(
    model: HookedTransformer,
    tokens: torch.Tensor,
    seq_lens: list[int],
    hooks: list,
    agree_tokens: tuple[int, ...],
    disagree_tokens: tuple[int, ...],
) -> tuple[float, list[float]]:
    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    per_prompt: list[float] = []
    agree_idx = list(agree_tokens)
    disagree_idx = list(disagree_tokens)
    for b in range(tokens.shape[0]):
        nl = logits[b, seq_lens[b]].float()
        per_prompt.append(1.0 if float(nl[agree_idx].max()) > float(nl[disagree_idx].max()) else 0.0)
    return (sum(per_prompt) / len(per_prompt) if per_prompt else 0.0), per_prompt


def _steering_sweep(ctx: ExperimentContext, pairs: list[tuple[str, str, str]], cfg: BreadthConfig) -> dict:
    n_layers = ctx.info.n_layers
    candidates = sorted({int(n_layers * f) for f in cfg.layer_fracs})

    dir_pairs = pairs[: cfg.dir_prompts]
    test_pairs = pairs[200 : 200 + cfg.steer_prompts]
    dir_wrong, dir_correct = build_sycophancy_prompts(dir_pairs, ctx.model_name)
    test_wrong, _ = build_sycophancy_prompts(test_pairs, ctx.model_name)

    rng = np.random.RandomState(cfg.seed)
    by_layer: dict[str, dict] = {}
    for layer in candidates:
        acts_w = extract_residual_stream(ctx.model, dir_wrong, layer, batch_size=cfg.batch)
        acts_c = extract_residual_stream(ctx.model, dir_correct, layer, batch_size=cfg.batch)
        direction = acts_w.mean(0) - acts_c.mean(0)
        dir_unit = direction / (np.linalg.norm(direction) + 1e-10)
        dir_t = torch.tensor(dir_unit, dtype=torch.bfloat16, device='cuda')
        resid_norm = float(np.linalg.norm(acts_w.mean(0)))
        rand_d = rng.randn(len(direction)).astype(np.float32)
        rand_d = rand_d / np.linalg.norm(rand_d)
        rand_t = torch.tensor(rand_d, dtype=torch.bfloat16, device='cuda')

        rows: list[dict] = []
        for alpha in cfg.alphas:
            real = _steer_measure(
                ctx.model, test_wrong, dir_t, layer, alpha, ctx.agree_tokens, ctx.disagree_tokens, cfg.batch
            )
            rand = _steer_measure(
                ctx.model, test_wrong, rand_t, layer, alpha, ctx.agree_tokens, ctx.disagree_tokens, cfg.batch
            )
            rows.append({'alpha': alpha, 'real': real, 'random': rand, 'delta': real - rand})
        by_layer[str(layer)] = {'resid_norm': resid_norm, 'alphas': rows}
    return {'candidates': candidates, 'layers': by_layer}


def _parse_alphas(value: str | None) -> tuple[int, ...]:
    if value is None:
        return _DEFAULT_ALPHAS
    return tuple(int(x) for x in value.split(','))


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument(
        '--alphas',
        type=str,
        default=None,
        help='Comma-separated ints to override default steering alphas (e.g. "0,-3000,-6000,-12000,-24000")',
    )
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PAIRS)
    parser.add_argument('--dla-prompts', type=int, default=_DEFAULT_DLA_PROMPTS)
    parser.add_argument('--baseline-prompts', type=int, default=_DEFAULT_BASELINE_PROMPTS)
    parser.add_argument('--dir-prompts', type=int, default=_DEFAULT_DIR_PROMPTS)
    parser.add_argument('--steer-prompts', type=int, default=_DEFAULT_STEER_PROMPTS)
    parser.add_argument('--permutations', type=int, default=_DEFAULT_PERMUTATIONS)
    parser.add_argument('--seed', type=int, default=_DEFAULT_SEED)
    parser.add_argument('--batch', type=int, default=_DEFAULT_BATCH)


def from_args(args: argparse.Namespace) -> BreadthConfig:
    """Build the validated config from a parsed argparse namespace."""
    return BreadthConfig(
        model=args.model,
        n_devices=args.n_devices,
        alphas=_parse_alphas(args.alphas),
        n_pairs=args.n_pairs,
        dla_prompts=args.dla_prompts,
        baseline_prompts=args.baseline_prompts,
        dir_prompts=args.dir_prompts,
        steer_prompts=args.steer_prompts,
        permutations=args.permutations,
        seed=args.seed,
        batch=args.batch,
    )
