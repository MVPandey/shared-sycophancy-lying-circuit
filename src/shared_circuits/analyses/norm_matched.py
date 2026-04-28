"""
Write-norm-matched random baseline: control for head W_O magnitude confound.

If shared heads happen to have high ``W_O`` Frobenius norms, zeroing them has
a bigger effect regardless of task specificity.  This control selects random
non-shared heads with matched ``W_O`` norms (greedy nearest-neighbor) and
re-runs head zeroing on the sycophancy task.

If shared still differs from norm-matched random, the effect is specific to
shared heads' computational role, not just their write magnitude.

Three head sets at the SHARED-set's size are reported: ``shared``,
``norm_matched`` (greedy nearest W_O norm), and ``random`` (uniform sample).
For each we report sycophancy rate, last-token logit-diff, and the delta
versus baseline.  The headline number is ``margin_shared_vs_norm_matched``.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.config import RANDOM_SEED
from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, load_results, model_session, save_results
from shared_circuits.prompts import build_sycophancy_prompts

_DEFAULT_N_PROMPTS: Final = 100
_DEFAULT_BATCH: Final = 4
# verdict thresholds: SHARED is judged a true confound (rather than write-magnitude)
# when its delta exceeds the norm-matched baseline by this margin.
_SPECIFICITY_MARGIN: Final = 0.05


class NormMatchedConfig(BaseModel):
    """Inputs for the write-norm-matched control analysis (single model)."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(...)
    n_devices: int = Field(default=1, gt=0)
    batch: int = Field(default=_DEFAULT_BATCH, gt=0)
    n_prompts: int = Field(default=_DEFAULT_N_PROMPTS, gt=0)
    shared_heads_from: str = Field(default='circuit_overlap')
    seed: int = Field(default=RANDOM_SEED)


def run(cfg: NormMatchedConfig) -> dict:
    """Run write-norm-matched control on ``cfg.model``."""
    syc_grid, lie_grid = _load_grids(cfg.model, cfg.shared_heads_from)
    pairs = load_triviaqa_pairs(max(cfg.n_prompts * 2, 200))[: cfg.n_prompts]
    with model_session(cfg.model, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, pairs, syc_grid, lie_grid, cfg)


def _analyse(
    ctx: ExperimentContext,
    pairs: list[tuple[str, str, str]],
    syc_grid: np.ndarray,
    lie_grid: np.ndarray,
    cfg: NormMatchedConfig,
) -> dict:
    shared = _build_shared(syc_grid, lie_grid)
    wrong_prompts, _ = build_sycophancy_prompts(pairs, ctx.model_name)

    wo_norms = _compute_wo_norms(ctx.model)
    norm_matched = _greedy_norm_match(shared, wo_norms, ctx.info.n_layers, ctx.info.n_heads, cfg.seed)
    random_heads = _sample_random_heads(shared, ctx.info.n_layers, ctx.info.n_heads, cfg.seed)

    shared_norms = [wo_norms[h] for h in shared]
    matched_norms = [wo_norms[h] for h in norm_matched]
    random_norms = [wo_norms[h] for h in random_heads]

    baseline = _measure(ctx.model, wrong_prompts, ctx.agree_tokens, ctx.disagree_tokens, cfg.batch)
    m_shared = _measure(
        ctx.model, wrong_prompts, ctx.agree_tokens, ctx.disagree_tokens, cfg.batch, _zero_heads_hooks(shared)
    )
    m_norm = _measure(
        ctx.model, wrong_prompts, ctx.agree_tokens, ctx.disagree_tokens, cfg.batch, _zero_heads_hooks(norm_matched)
    )
    m_rand = _measure(
        ctx.model, wrong_prompts, ctx.agree_tokens, ctx.disagree_tokens, cfg.batch, _zero_heads_hooks(random_heads)
    )

    d_shared = {
        'rate': m_shared['rate'] - baseline['rate'],
        'logit_diff': m_shared['logit_diff'] - baseline['logit_diff'],
    }
    d_norm = {'rate': m_norm['rate'] - baseline['rate'], 'logit_diff': m_norm['logit_diff'] - baseline['logit_diff']}
    d_rand = {'rate': m_rand['rate'] - baseline['rate'], 'logit_diff': m_rand['logit_diff'] - baseline['logit_diff']}

    margin = {
        'rate': float(d_shared['rate'] - d_norm['rate']),
        'logit_diff': float(d_shared['logit_diff'] - d_norm['logit_diff']),
    }
    verdict = _verdict(margin)

    results = {
        'model': ctx.model_name,
        'verdict': verdict,
        'n_shared': len(shared),
        'n_prompts': len(wrong_prompts),
        'wo_norms': {
            'shared_mean': float(np.mean(shared_norms)),
            'shared_std': float(np.std(shared_norms)),
            'norm_matched_mean': float(np.mean(matched_norms)),
            'norm_matched_std': float(np.std(matched_norms)),
            'random_mean': float(np.mean(random_norms)),
            'random_std': float(np.std(random_norms)),
        },
        'baseline': baseline,
        'shared': m_shared,
        'norm_matched': m_norm,
        'random': m_rand,
        'delta_shared': d_shared,
        'delta_norm_matched': d_norm,
        'delta_random': d_rand,
        'margin_shared_vs_norm_matched': margin,
        'shared_heads': [list(h) for h in shared],
        'norm_matched_heads': [list(h) for h in norm_matched],
        'random_heads': [list(h) for h in random_heads],
    }
    save_results(results, 'norm_matched', ctx.model_name)
    return results


def _load_grids(model_name: str, shared_heads_from: str) -> tuple[np.ndarray, np.ndarray]:
    """Read DLA grids from a saved circuit-overlap (or breadth) result."""
    data = load_results(shared_heads_from, model_name)
    if 'syc_grid' in data and 'lie_grid' in data:
        return np.array(data['syc_grid']), np.array(data['lie_grid'])
    if 'head_overlap' in data and 'syc_grid' in data['head_overlap']:
        return np.array(data['head_overlap']['syc_grid']), np.array(data['head_overlap']['lie_grid'])
    raise FileNotFoundError(f'No DLA grid in {shared_heads_from} for {model_name}')


def _build_shared(syc_grid: np.ndarray, lie_grid: np.ndarray) -> list[tuple[int, int]]:
    n_layers, n_heads = syc_grid.shape
    total = n_layers * n_heads
    k = math.ceil(math.sqrt(total))
    sf, lf = syc_grid.flatten(), lie_grid.flatten()
    syc_top = set(np.argsort(sf)[::-1][:k].tolist())
    lie_top = set(np.argsort(lf)[::-1][:k].tolist())
    shared_idx = list(syc_top & lie_top)
    combined = (sf + lf) / 2
    return [(int(i // n_heads), int(i % n_heads)) for i in sorted(shared_idx, key=lambda i: -combined[i])]


def _compute_wo_norms(model: 'HookedTransformer') -> dict[tuple[int, int], float]:
    """Frobenius norm of ``W_O`` per head."""
    norms: dict[tuple[int, int], float] = {}
    for layer in range(model.cfg.n_layers):
        w_o = model.blocks[layer].attn.W_O.float()
        for head in range(model.cfg.n_heads):
            norms[(layer, head)] = float(torch.norm(w_o[head]).item())
    return norms


def _greedy_norm_match(
    shared: list[tuple[int, int]],
    all_norms: dict[tuple[int, int], float],
    n_layers: int,
    n_heads: int,
    seed: int,
) -> list[tuple[int, int]]:
    """For each shared head, greedily pick the unused non-shared head with the closest ``W_O`` norm."""
    shared_set = set(shared)
    available = [(l, h) for l in range(n_layers) for h in range(n_heads) if (l, h) not in shared_set]
    rng = np.random.RandomState(seed)
    rng.shuffle(available)
    matched: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for sh in shared:
        target = all_norms[sh]
        best: tuple[int, int] | None = None
        best_diff = float('inf')
        for cand in available:
            if cand in used:
                continue
            diff = abs(all_norms[cand] - target)
            if diff < best_diff:
                best, best_diff = cand, diff
        if best is not None:
            matched.append(best)
            used.add(best)
    return matched


def _sample_random_heads(
    shared: list[tuple[int, int]], n_layers: int, n_heads: int, seed: int
) -> list[tuple[int, int]]:
    """Uniform random non-shared heads of size ``len(shared)``."""
    rng = np.random.RandomState(seed)
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    shared_set = set(shared)
    non_shared = [h for h in all_heads if h not in shared_set]
    rand_idx = rng.choice(len(non_shared), len(shared), replace=False)
    return [non_shared[i] for i in rand_idx]


def _zero_heads_hooks(
    head_set: list[tuple[int, int]],
) -> list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor]]]:
    by_layer: dict[int, list[int]] = {}
    for layer, head in head_set:
        by_layer.setdefault(layer, []).append(head)
    hooks: list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor]]] = []
    for layer, heads in by_layer.items():

        def make_hook(hs: list[int]) -> Callable[[torch.Tensor, object], torch.Tensor]:
            def hook_fn(z: torch.Tensor, hook: object) -> torch.Tensor:
                for h in hs:
                    z[:, :, h, :] = 0.0
                return z

            return hook_fn

        hooks.append((f'blocks.{layer}.attn.hook_z', make_hook(heads)))
    return hooks


@torch.no_grad()
def _measure(
    model: 'HookedTransformer',
    prompts: list[str],
    agree_tokens: tuple[int, ...],
    disagree_tokens: tuple[int, ...],
    batch: int,
    hooks: list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor]]] | None = None,
) -> dict:
    """Sycophancy rate + mean last-token logit_diff (with optional fwd hooks)."""
    pad_id = getattr(model.tokenizer, 'pad_token_id', None) or 0
    agree_idx = list(agree_tokens)
    disagree_idx = list(disagree_tokens)
    agree_flags: list[float] = []
    logit_diffs: list[float] = []
    for i in range(0, len(prompts), batch):
        batch_p = prompts[i : i + batch]
        tokens = model.to_tokens(batch_p, prepend_bos=True)
        seq_lens = [int(x) for x in ((tokens != pad_id).sum(dim=1) - 1).tolist()]
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        for b in range(len(batch_p)):
            nl = logits[b, seq_lens[b]].float()
            a_max = float(nl[agree_idx].max())
            d_max = float(nl[disagree_idx].max())
            agree_flags.append(1.0 if a_max > d_max else 0.0)
            logit_diffs.append(a_max - d_max)
    return {'rate': float(np.mean(agree_flags)), 'logit_diff': float(np.mean(logit_diffs))}


def _verdict(margin: dict[str, float]) -> str:
    """SPECIFIC_TO_SHARED if shared exceeds norm-matched on both rate and logit_diff."""
    if margin['rate'] > _SPECIFICITY_MARGIN and margin['logit_diff'] > _SPECIFICITY_MARGIN:
        return 'SPECIFIC_TO_SHARED'
    if margin['rate'] > _SPECIFICITY_MARGIN or margin['logit_diff'] > _SPECIFICITY_MARGIN:
        return 'PARTIAL_SPECIFICITY'
    return 'NORM_CONFOUND'


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--batch', type=int, default=_DEFAULT_BATCH)
    parser.add_argument('--n-prompts', type=int, default=_DEFAULT_N_PROMPTS)
    parser.add_argument(
        '--shared-heads-from',
        default='circuit_overlap',
        help='Experiment slug whose saved DLA grids define the shared head set.',
    )
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)


def from_args(args: argparse.Namespace) -> NormMatchedConfig:
    """Build the validated config from a parsed argparse namespace."""
    return NormMatchedConfig(
        model=args.model,
        n_devices=args.n_devices,
        batch=args.batch,
        n_prompts=args.n_prompts,
        shared_heads_from=args.shared_heads_from,
        seed=args.seed,
    )
