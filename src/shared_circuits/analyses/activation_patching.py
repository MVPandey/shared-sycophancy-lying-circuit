"""
Shared-set top-K activation patching: gold-standard causal test on shared heads.

For each (corrupt, clean) prompt pair, cache the shared heads' z-activations
from the clean run, then splice them all simultaneously into a forward pass
on the corrupt prompt and measure the resulting logit-diff shift. A matched
random-head set serves as the negative control.

Distinct from :mod:`shared_circuits.analyses.attribution_patching`, which
patches per-head and sweeps every head individually. This analysis patches
the entire shared-head set as a unit and is intended for >=32B models where
per-head sweeps are infeasible.

Reads the shared-heads list from a sibling analysis (default
``circuit_overlap``); ranking is preserved via the saved ``shared_heads`` field.
"""

import argparse
from collections.abc import Callable
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.config import RANDOM_SEED
from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, load_results, model_session, save_results
from shared_circuits.prompts import build_sycophancy_prompts

_DEFAULT_BATCH: Final = 1
_DEFAULT_N_BOOT: Final = 2000
_DEFAULT_N_PAIRS: Final = 20
_DEFAULT_VERDICT_K: Final = 15
# verdict thresholds: shared patching is causal when its delta exceeds the
# random-control delta with a margin and the CI excludes zero.
_CAUSAL_MARGIN: Final = 0.05


class ActivationPatchingConfig(BaseModel):
    """Inputs for the shared-set activation-patching analysis (single model)."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(...)
    n_devices: int = Field(default=1, gt=0)
    batch: int = Field(default=_DEFAULT_BATCH, gt=0)
    n_pairs: int = Field(default=_DEFAULT_N_PAIRS, gt=0)
    shared_heads_from: str = Field(default='circuit_overlap')
    shared_heads_k: int = Field(default=_DEFAULT_VERDICT_K, gt=0)
    n_boot: int = Field(default=_DEFAULT_N_BOOT, gt=0)
    seed: int = Field(default=RANDOM_SEED)


def run(cfg: ActivationPatchingConfig) -> dict:
    """Run shared-set activation patching on ``cfg.model``."""
    shared_heads = _load_shared_heads(cfg.shared_heads_from, cfg.model, cfg.shared_heads_k)
    pairs = load_triviaqa_pairs(max(cfg.n_pairs * 2, 100))[: cfg.n_pairs]
    with model_session(cfg.model, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, pairs, shared_heads, cfg)


def _load_shared_heads(source: str, model_name: str, top_k: int) -> list[tuple[int, int]]:
    """Read the ranked shared-heads list from a sibling analysis' saved JSON."""
    data = load_results(source, model_name)
    bucket = next(o for o in data['overlap_by_K'] if o['K'] == top_k)
    return [(int(h[0]), int(h[1])) for h in bucket['shared_heads']]


def _analyse(
    ctx: ExperimentContext,
    pairs: list[tuple[str, str, str]],
    shared_heads: list[tuple[int, int]],
    cfg: ActivationPatchingConfig,
) -> dict:
    random_heads = _matched_random_heads(shared_heads, ctx.info.n_layers, ctx.info.n_heads, cfg.seed)

    base_s, patched_s = _measure_set_aggregate(ctx, pairs, shared_heads)
    d_s, s_lo, s_hi = _paired_ci(base_s, patched_s, cfg.n_boot, cfg.seed)

    base_r, patched_r = _measure_set_aggregate(ctx, pairs, random_heads)
    d_r, r_lo, r_hi = _paired_ci(base_r, patched_r, cfg.n_boot, cfg.seed)

    # p-value approximation via permutation of pair labels: sign of patched - baseline
    p_value = _paired_sign_p(base_s, patched_s, cfg.n_boot, cfg.seed)

    verdict = _verdict(d_s, s_lo, s_hi, d_r)

    results = {
        'model': ctx.model_name,
        'verdict': verdict,
        'config': {
            'n_pairs': len(pairs),
            'batch': cfg.batch,
            'n_boot': cfg.n_boot,
            'seed': cfg.seed,
            'shared_heads_from': cfg.shared_heads_from,
            'shared_heads_k': cfg.shared_heads_k,
        },
        'n_layers': ctx.info.n_layers,
        'n_heads': ctx.info.n_heads,
        'n_shared_heads': len(shared_heads),
        'n_random_heads': len(random_heads),
        'shared_heads': [list(h) for h in shared_heads],
        'random_heads': [list(h) for h in random_heads],
        'baseline_logit_diff': float(np.mean(base_s)) if base_s else 0.0,
        'patched_logit_diff': float(np.mean(patched_s)) if patched_s else 0.0,
        'shift': d_s,
        'p_value': p_value,
        'shared': {
            'baseline_logit_diffs': base_s,
            'patched_logit_diffs': patched_s,
            'delta': d_s,
            'ci': [s_lo, s_hi],
            'significant': bool(s_lo > 0 or s_hi < 0),
        },
        'random': {
            'baseline_logit_diffs': base_r,
            'patched_logit_diffs': patched_r,
            'delta': d_r,
            'ci': [r_lo, r_hi],
            'significant': bool(r_lo > 0 or r_hi < 0),
        },
    }
    save_results(results, 'activation_patching', ctx.model_name)
    return results


def _matched_random_heads(
    shared: list[tuple[int, int]], n_layers: int, n_heads: int, seed: int
) -> list[tuple[int, int]]:
    """Sample ``|shared|`` random heads from the full grid (with replacement disallowed)."""
    total = n_layers * n_heads
    rng = np.random.RandomState(seed)
    idx = rng.choice(total, len(shared), replace=False).tolist()
    return [(int(i // n_heads), int(i % n_heads)) for i in idx]


@torch.no_grad()
def _cache_z(model: 'HookedTransformer', prompt: str, layers: list[int]) -> tuple[dict[int, torch.Tensor], int]:
    """Run one forward pass on ``prompt``; cache last-token z at each layer in ``layers``."""
    pad_id = getattr(model.tokenizer, 'pad_token_id', None) or 0
    tokens = model.to_tokens([prompt], prepend_bos=True)
    seq_len = int(((tokens != pad_id).sum(dim=1) - 1).item())
    cache: dict[int, torch.Tensor] = {}
    hooks: list[tuple[str, Callable[[torch.Tensor, object], None]]] = []
    for layer in layers:

        def make_cap(li: int) -> Callable[[torch.Tensor, object], None]:
            def cap(t: torch.Tensor, hook: object) -> None:
                cache[li] = t[0, seq_len, :, :].clone()

            return cap

        hooks.append((f'blocks.{layer}.attn.hook_z', make_cap(layer)))
    model.run_with_hooks(tokens, fwd_hooks=hooks)
    return cache, seq_len


@torch.no_grad()
def _logit_diff(
    model: 'HookedTransformer',
    prompt: str,
    agree_t: list[int],
    disagree_t: list[int],
    hooks: list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]] | None = None,
) -> float:
    """Last-token ``max(agree) - max(disagree)`` logit diff, optionally with hooks applied."""
    pad_id = getattr(model.tokenizer, 'pad_token_id', None) or 0
    tokens = model.to_tokens([prompt], prepend_bos=True)
    seq_len = int(((tokens != pad_id).sum(dim=1) - 1).item())
    logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
    nl = logits[0, seq_len].float()
    return float(nl[agree_t].max()) - float(nl[disagree_t].max())


def _make_patch_hooks(
    head_set: list[tuple[int, int]],
    clean_cache: dict[int, torch.Tensor],
    corrupt_seq_len: int,
) -> list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]]:
    """Build hooks that splice clean-cached z values into the corrupt run at the last position."""
    by_layer: dict[int, list[int]] = {}
    for layer, head in head_set:
        by_layer.setdefault(layer, []).append(head)
    hooks: list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]] = []
    for layer, heads in by_layer.items():
        clean_z = clean_cache[layer]

        def make_hook(
            hs: list[int], cz: torch.Tensor, pos: int
        ) -> Callable[[torch.Tensor, object], torch.Tensor | None]:
            def hook_fn(z: torch.Tensor, hook: object) -> torch.Tensor:
                # Clamp to the corrupt prompt's actual length when the cache came from a longer clean prompt.
                patched_pos = min(pos, z.shape[1] - 1)
                for h in hs:
                    z[0, patched_pos, h, :] = cz[h, :].to(z.device, dtype=z.dtype)
                return z

            return hook_fn

        hooks.append((f'blocks.{layer}.attn.hook_z', make_hook(heads, clean_z, corrupt_seq_len)))
    return hooks


def _measure_set_aggregate(
    ctx: ExperimentContext,
    pairs: list[tuple[str, str, str]],
    head_set: list[tuple[int, int]],
) -> tuple[list[float], list[float]]:
    """For each pair, splice ``head_set`` clean->corrupt and record (baseline, patched) logit diffs."""
    baseline_diffs: list[float] = []
    patched_diffs: list[float] = []
    layers = sorted({l for l, _ in head_set})
    pad_id = getattr(ctx.model.tokenizer, 'pad_token_id', None) or 0
    agree_t = list(ctx.agree_tokens)
    disagree_t = list(ctx.disagree_tokens)
    for q, w, c in pairs:
        wrong_prompts, correct_prompts = build_sycophancy_prompts([(q, w, c)], ctx.model_name)
        corrupt_prompt = wrong_prompts[0]
        clean_prompt = correct_prompts[0]
        clean_cache, _ = _cache_z(ctx.model, clean_prompt, layers)
        baseline = _logit_diff(ctx.model, corrupt_prompt, agree_t, disagree_t)
        tokens = ctx.model.to_tokens([corrupt_prompt], prepend_bos=True)
        corrupt_seq = int(((tokens != pad_id).sum(dim=1) - 1).item())
        hooks = _make_patch_hooks(head_set, clean_cache, corrupt_seq)
        patched = _logit_diff(ctx.model, corrupt_prompt, agree_t, disagree_t, hooks)
        baseline_diffs.append(baseline)
        patched_diffs.append(patched)
    return baseline_diffs, patched_diffs


def _paired_ci(base: list[float], patched: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    """Paired bootstrap 95% CI for ``mean(patched) - mean(base)`` over pair indices."""
    if not base:
        return 0.0, 0.0, 0.0
    b = np.array(base)
    p = np.array(patched)
    n = len(b)
    rng = np.random.RandomState(seed)
    boots = np.array([p[idx].mean() - b[idx].mean() for idx in (rng.choice(n, n, replace=True) for _ in range(n_boot))])
    return float(p.mean() - b.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _paired_sign_p(base: list[float], patched: list[float], n_boot: int, seed: int) -> float:
    """Bootstrap two-sided p-value: fraction of resamples whose mean delta has opposite sign."""
    if not base:
        return 1.0
    b = np.array(base)
    p = np.array(patched)
    n = len(b)
    observed = float(p.mean() - b.mean())
    rng = np.random.RandomState(seed + 1)
    diffs = p - b
    centered = diffs - diffs.mean()
    extreme = 0
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_mean = centered[idx].mean()
        if abs(boot_mean) >= abs(observed):
            extreme += 1
    return float((extreme + 1) / (n_boot + 1))


def _verdict(shared_delta: float, s_lo: float, s_hi: float, random_delta: float) -> str:
    """Categorize: shared causally moves logit_diff if margin over random exceeds threshold and CI is significant."""
    margin = shared_delta - random_delta
    significant = s_lo > 0 or s_hi < 0
    if margin > _CAUSAL_MARGIN and significant:
        return 'CAUSAL_SHARED'
    if margin > _CAUSAL_MARGIN or significant:
        return 'PARTIAL_CAUSAL'
    return 'NOT_CAUSAL'


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--batch', type=int, default=_DEFAULT_BATCH, help='Reserved; activation patching is per-pair.')
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PAIRS)
    parser.add_argument(
        '--shared-heads-from',
        default='circuit_overlap',
        help='Experiment slug whose saved results provide the shared-heads list.',
    )
    parser.add_argument('--shared-heads-k', type=int, default=_DEFAULT_VERDICT_K)
    parser.add_argument('--n-boot', type=int, default=_DEFAULT_N_BOOT)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)


def from_args(args: argparse.Namespace) -> ActivationPatchingConfig:
    """Build the validated config from a parsed argparse namespace."""
    return ActivationPatchingConfig(
        model=args.model,
        n_devices=args.n_devices,
        batch=args.batch,
        n_pairs=args.n_pairs,
        shared_heads_from=args.shared_heads_from,
        shared_heads_k=args.shared_heads_k,
        n_boot=args.n_boot,
        seed=args.seed,
    )
