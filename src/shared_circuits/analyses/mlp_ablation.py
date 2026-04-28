"""
MLP ablation tug-of-war: zero each MLP layer and measure circuit-level effects.

Three sub-modes share the same analysis surface:

* ``ablation`` (default): zero each MLP layer's output and report the per-layer
  shift in sycophancy rate (``delta = ablated_rate - baseline_rate``). Targets
  mid-to-late layers via :func:`_default_target_layers`.

* ``disruption``: ablate the same MLP layers but measure perplexity on a fixed
  set of neutral prompts. A small ``ratio = ablated_ppl / baseline_ppl`` means
  the behavioral change is specific (not generic degradation). Layers can be
  pinned via ``--layers``; otherwise defaults to the same set as ``ablation``.

* ``tugofwar``: per-layer Spearman correlation between MLP-ablation
  ``|delta_syc_rate|`` and the per-layer shared-head DLA importance vector
  (intersect-topk by default; ``sum_min`` and ``sum_geomean`` as alternates).
  Reads existing ``head_zeroing`` / ``mlp_ablation`` / ``circuit_overlap`` JSON;
  no model is loaded.
"""

import argparse
import math
from collections.abc import Callable
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import mannwhitneyu, spearmanr

from shared_circuits.config import DEFAULT_BATCH_SIZE, RANDOM_SEED
from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import load_results, model_session, save_results
from shared_circuits.extraction import measure_agreement_rate
from shared_circuits.prompts import build_sycophancy_prompts

_MODES: Final[tuple[str, ...]] = ('ablation', 'disruption', 'tugofwar')
_DEFAULT_TEST_PROMPTS: Final = 100
_DEFAULT_N_PAIRS: Final = 200
_DEFAULT_PPL_RATIO_THRESHOLD: Final = 5.0
_TUGOFWAR_VARIANTS: Final[tuple[str, ...]] = ('intersect_topk', 'sum_min', 'sum_geomean')
# minimum sequence length needed to compute a next-token loss (input + target).
_MIN_SEQ_LEN_FOR_LOSS: Final = 2
# minimum sample size for spearman correlation to be meaningful.
_MIN_SPEARMAN_N: Final = 3

# Neutral text used for the disruption-control perplexity sanity check; mirrors
# the legacy `_NEUTRAL_PROMPTS` corpus. Kept short and factual so spikes after
# MLP ablation can be attributed to broken generation, not topical drift.
_NEUTRAL_PROMPTS: Final[tuple[str, ...]] = (
    'The capital of France is',
    'Water boils at a temperature of',
    'The largest planet in our solar system is',
    'Photosynthesis is the process by which',
    'The speed of light is approximately',
    'DNA stands for deoxyribonucleic',
    'The Great Wall of China was built',
    'Shakespeare wrote many famous',
    'The periodic table organizes chemical',
    'Mount Everest is located in the',
    'The human heart has four',
    'Gravity is a fundamental force that',
    'The Amazon River flows through',
    'Electricity is the flow of',
    'The Renaissance was a period of',
    'Computers process information using',
    'The Milky Way is a spiral',
    'Oxygen is essential for',
    'The Industrial Revolution began in',
    'Mathematics is the study of',
)


class MlpAblationConfig(BaseModel):
    """Inputs for the MLP-ablation analysis (single model, single sub-mode)."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(...)
    n_devices: int = Field(default=1, gt=0)
    mode: str = Field(default='ablation')
    batch: int = Field(default=DEFAULT_BATCH_SIZE, gt=0)
    n_pairs: int = Field(default=_DEFAULT_N_PAIRS, gt=0)
    test_prompts: int = Field(default=_DEFAULT_TEST_PROMPTS, gt=0)
    layers: tuple[int, ...] | None = Field(default=None)
    ppl_ratio_threshold: float = Field(default=_DEFAULT_PPL_RATIO_THRESHOLD, gt=0)
    shared_heads_from: str = Field(default='circuit_overlap')
    mlp_results_from: str = Field(default='mlp_ablation')
    seed: int = Field(default=RANDOM_SEED)


def run(cfg: MlpAblationConfig) -> dict:
    """Dispatch to the configured sub-mode."""
    if cfg.mode not in _MODES:
        raise ValueError(f'unknown mode {cfg.mode}')
    if cfg.mode == 'tugofwar':
        return _run_tugofwar(cfg)
    if cfg.mode == 'disruption':
        return _run_disruption(cfg)
    return _run_ablation(cfg)


def _run_ablation(cfg: MlpAblationConfig) -> dict:
    pairs = load_triviaqa_pairs(cfg.n_pairs)
    test_pairs = pairs[: cfg.test_prompts]
    with model_session(cfg.model, n_devices=cfg.n_devices) as ctx:
        wrong_prompts, _ = build_sycophancy_prompts(test_pairs, ctx.model_name)
        baseline_rate = measure_agreement_rate(
            ctx.model, wrong_prompts, ctx.agree_tokens, ctx.disagree_tokens, batch_size=cfg.batch
        )
        target_layers = _resolve_target_layers(cfg.layers, ctx.info.n_layers)
        layer_effects: dict[str, dict[str, float]] = {}
        for layer in target_layers:
            rate = measure_agreement_rate(
                ctx.model,
                wrong_prompts,
                ctx.agree_tokens,
                ctx.disagree_tokens,
                batch_size=cfg.batch,
                hooks=[_zero_mlp_hook(layer)],
            )
            layer_effects[str(layer)] = {'rate': float(rate), 'delta': float(rate - baseline_rate)}
        result = {
            'model': ctx.model_name,
            'mode': 'ablation',
            'n_layers': ctx.info.n_layers,
            'baseline_rate': float(baseline_rate),
            'target_layers': list(target_layers),
            'layer_effects': layer_effects,
        }
        save_results(result, 'mlp_ablation', ctx.model_name)
        return result


def _run_disruption(cfg: MlpAblationConfig) -> dict:
    with model_session(cfg.model, n_devices=cfg.n_devices) as ctx:
        baseline_ppl = _compute_perplexity(ctx.model, list(_NEUTRAL_PROMPTS))
        target_layers = _resolve_target_layers(cfg.layers, ctx.info.n_layers)
        layers_out: dict[str, dict[str, float | bool]] = {}
        for layer in target_layers:
            ablated_ppl = _compute_perplexity(ctx.model, list(_NEUTRAL_PROMPTS), ablate_layer=layer)
            ratio = ablated_ppl / baseline_ppl if baseline_ppl > 0 else float('inf')
            specific = ratio < cfg.ppl_ratio_threshold
            layers_out[str(layer)] = {
                'perplexity': float(ablated_ppl),
                'ratio': float(ratio),
                'specific': bool(specific),
            }
        result = {
            'model': ctx.model_name,
            'mode': 'disruption',
            'n_prompts': len(_NEUTRAL_PROMPTS),
            'baseline_perplexity': float(baseline_ppl),
            'ppl_ratio_threshold': cfg.ppl_ratio_threshold,
            'target_layers': list(target_layers),
            'layers': layers_out,
        }
        save_results(result, 'mlp_disruption_control', ctx.model_name)
        return result


def _run_tugofwar(cfg: MlpAblationConfig) -> dict:
    syc_grid, lie_grid = _load_grids(cfg.shared_heads_from, cfg.model)
    mlp_data = load_results(cfg.mlp_results_from, cfg.model)
    layer_effects = mlp_data.get('layer_effects', {})
    if not layer_effects:
        raise ValueError(f'no layer_effects in {cfg.mlp_results_from} results for {cfg.model}')
    mlp_deltas = {int(k): float(v['delta']) for k, v in layer_effects.items()}
    importance = _shared_importance_per_layer(syc_grid, lie_grid)
    shared_layers = _shared_layers_set(syc_grid, lie_grid)
    tested_layers = sorted(l for l in mlp_deltas if l < syc_grid.shape[0])
    abs_d = np.array([abs(mlp_deltas[l]) for l in tested_layers])
    signed_d = np.array([mlp_deltas[l] for l in tested_layers])

    by_variant: dict[str, dict] = {}
    for variant in _TUGOFWAR_VARIANTS:
        imp_vec = np.array([float(importance[variant][l]) for l in tested_layers])
        rho_abs, p_abs = _safe_spearman(imp_vec, abs_d)
        rho_signed, p_signed = _safe_spearman(imp_vec, signed_d)
        by_variant[variant] = {
            'importance_by_layer': [float(x) for x in imp_vec],
            'spearman_abs_delta': {'rho': rho_abs, 'p_asymptotic': p_abs},
            'spearman_signed_delta': {'rho': rho_signed, 'p_asymptotic': p_signed},
        }

    result = {
        'model': cfg.model,
        'mode': 'tugofwar',
        'n_layers_grid': int(syc_grid.shape[0]),
        'n_heads': int(syc_grid.shape[1]),
        'n_layers_tested': len(tested_layers),
        'tested_layers': tested_layers,
        'shared_definition': {
            'k_topk': importance['k'],
            'total_heads': importance['total_heads'],
            'shared_head_count': importance['shared_heads_count'],
            'shared_layers': sorted(shared_layers),
        },
        'abs_delta_by_layer': [float(x) for x in abs_d],
        'signed_delta_by_layer': [float(x) for x in signed_d],
        'by_variant': by_variant,
        'membership_test': _membership_test(tested_layers, abs_d, shared_layers),
        'distance_test': _distance_test(tested_layers, abs_d, shared_layers),
    }
    save_results(result, 'tugofwar_prediction', cfg.model)
    return result


def _resolve_target_layers(layers: tuple[int, ...] | None, n_layers: int) -> list[int]:
    """Return explicit layers if given, else the legacy mid-to-late default set."""
    if layers is not None:
        return sorted(int(l) for l in layers if 0 <= int(l) < n_layers)
    return _default_target_layers(n_layers)


def _default_target_layers(n_layers: int) -> list[int]:
    """Mid-to-late layer sampling matching the legacy ``run_mlp_ablation`` defaults."""
    mid = n_layers // 2
    candidates = {
        mid - 4,
        mid - 2,
        mid,
        mid + 1,
        mid + 2,
        mid + 3,
        mid + 4,
        mid + 6,
        mid + 8,
        n_layers - 4,
        n_layers - 2,
    }
    return sorted(c for c in candidates if 0 <= c < n_layers)


def _zero_mlp_hook(layer: int) -> tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]:
    """Return a ``(hook_name, fn)`` tuple that zeros the MLP output at ``layer``."""

    def fn(t: torch.Tensor, hook: object) -> torch.Tensor:
        return torch.zeros_like(t)

    return f'blocks.{layer}.hook_mlp_out', fn


@torch.no_grad()
def _compute_perplexity(
    model: 'HookedTransformer',
    prompts: list[str],
    *,
    ablate_layer: int | None = None,
) -> float:
    """Mean perplexity over ``prompts``; optionally zero the MLP output at ``ablate_layer``."""
    pad_id = getattr(model.tokenizer, 'pad_token_id', None) or 0
    total_loss = 0.0
    total_tokens = 0
    for prompt in prompts:
        tokens = model.to_tokens([prompt], prepend_bos=True)
        seq_len = int((tokens != pad_id).sum().item())
        if seq_len < _MIN_SEQ_LEN_FOR_LOSS:
            continue
        if ablate_layer is not None:
            logits = model.run_with_hooks(tokens, fwd_hooks=[_zero_mlp_hook(ablate_layer)])
        else:
            logits = model(tokens)
        shift_logits = logits[0, : seq_len - 1, :]
        shift_labels = tokens[0, 1:seq_len]
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
        total_loss += float(loss.item()) * (seq_len - 1)
        total_tokens += seq_len - 1
    return float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float('inf')


def _load_grids(source: str, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read DLA grids from a sibling analysis (typically ``circuit_overlap`` / ``breadth``)."""
    data = load_results(source, model_name)
    if 'syc_grid' in data and 'lie_grid' in data:
        return np.array(data['syc_grid']), np.array(data['lie_grid'])
    if 'head_overlap' in data and 'syc_grid' in data['head_overlap']:
        return np.array(data['head_overlap']['syc_grid']), np.array(data['head_overlap']['lie_grid'])
    raise FileNotFoundError(f'No DLA grid in {source} for {model_name}')


def _shared_importance_per_layer(syc_grid: np.ndarray, lie_grid: np.ndarray) -> dict:
    """Three per-layer shared-importance vectors plus the metadata used to build them."""
    n_layers, n_heads = syc_grid.shape
    total = n_layers * n_heads
    k = math.ceil(math.sqrt(total))
    sf = syc_grid.flatten()
    lf = lie_grid.flatten()
    syc_top = set(np.argsort(sf)[::-1][:k].tolist())
    lie_top = set(np.argsort(lf)[::-1][:k].tolist())
    shared_idx = syc_top & lie_top

    shared_grid = np.zeros_like(syc_grid)
    for idx in shared_idx:
        l, h = divmod(idx, n_heads)
        shared_grid[l, h] = (syc_grid[l, h] + lie_grid[l, h]) / 2
    intersect_layer = shared_grid.sum(axis=1)
    sum_min = np.minimum(syc_grid, lie_grid).sum(axis=1)
    sum_geo = np.sqrt(np.clip(syc_grid, 0, None) * np.clip(lie_grid, 0, None)).sum(axis=1)

    return {
        'intersect_topk': intersect_layer,
        'sum_min': sum_min,
        'sum_geomean': sum_geo,
        'k': int(k),
        'total_heads': int(total),
        'shared_heads_count': len(shared_idx),
    }


def _shared_layers_set(syc_grid: np.ndarray, lie_grid: np.ndarray) -> set[int]:
    """Layers containing at least one top-K-intersect shared head."""
    n_layers, n_heads = syc_grid.shape
    k = math.ceil(math.sqrt(n_layers * n_heads))
    sf, lf = syc_grid.flatten(), lie_grid.flatten()
    syc_top = set(np.argsort(sf)[::-1][:k].tolist())
    lie_top = set(np.argsort(lf)[::-1][:k].tolist())
    return {int(idx // n_heads) for idx in (syc_top & lie_top)}


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < _MIN_SPEARMAN_N or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan'), float('nan')
    rho, p = spearmanr(x, y)
    if np.isnan(rho):
        return float('nan'), float('nan')
    return float(rho), float(p)


def _membership_test(
    tested_layers: list[int],
    abs_d: np.ndarray,
    shared_layers: set[int],
) -> dict:
    """Mann-Whitney U on whether ``|MLP delta|`` is higher at shared than non-shared tested layers."""
    in_s = [float(abs_d[i]) for i, l in enumerate(tested_layers) if l in shared_layers]
    out_s = [float(abs_d[i]) for i, l in enumerate(tested_layers) if l not in shared_layers]
    if in_s and out_s:
        u, p = mannwhitneyu(in_s, out_s, alternative='greater')
        u_val, p_val = float(u), float(p)
    else:
        u_val, p_val = float('nan'), float('nan')
    return {
        'in_shared': in_s,
        'out_shared': out_s,
        'mean_in': float(np.mean(in_s)) if in_s else float('nan'),
        'mean_out': float(np.mean(out_s)) if out_s else float('nan'),
        'u_statistic': u_val,
        'p_one_sided': p_val,
    }


def _distance_test(tested_layers: list[int], abs_d: np.ndarray, shared_layers: set[int]) -> dict:
    """Spearman on whether ``|MLP delta|`` is higher at tested layers closer to any shared-head layer."""
    if not shared_layers:
        return {'rho': float('nan'), 'p_asymptotic': float('nan'), 'distances': []}
    sh_sorted = sorted(shared_layers)
    dist = np.array([min(abs(l - s) for s in sh_sorted) for l in tested_layers])
    if np.ptp(dist) == 0:
        return {'rho': float('nan'), 'p_asymptotic': float('nan'), 'distances': dist.tolist()}
    rho, p = spearmanr(dist, abs_d)
    return {'rho': float(rho), 'p_asymptotic': float(p), 'distances': [int(x) for x in dist]}


def _parse_layers(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(x) for x in value.split(',') if x.strip())


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument(
        '--mode',
        choices=list(_MODES),
        default='ablation',
        help='ablation: per-MLP syc-rate delta; disruption: neutral-text PPL; tugofwar: correlation analysis.',
    )
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PAIRS)
    parser.add_argument('--test-prompts', type=int, default=_DEFAULT_TEST_PROMPTS)
    parser.add_argument(
        '--layers',
        type=str,
        default=None,
        help='Comma-separated explicit MLP layers to ablate; default is mid-to-late auto-selection.',
    )
    parser.add_argument(
        '--ppl-ratio-threshold',
        type=float,
        default=_DEFAULT_PPL_RATIO_THRESHOLD,
        help='Disruption mode: ratio above which a layer is flagged as generic degradation.',
    )
    parser.add_argument(
        '--shared-heads-from',
        default='circuit_overlap',
        help='Tugofwar mode: experiment slug whose DLA grids feed the per-layer shared-importance vector.',
    )
    parser.add_argument(
        '--mlp-results-from',
        default='mlp_ablation',
        help='Tugofwar mode: experiment slug whose layer_effects feed the per-layer MLP deltas.',
    )
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)


def from_args(args: argparse.Namespace) -> MlpAblationConfig:
    """Build the validated config from a parsed argparse namespace."""
    return MlpAblationConfig(
        model=args.model,
        n_devices=args.n_devices,
        mode=args.mode,
        batch=args.batch,
        n_pairs=args.n_pairs,
        test_prompts=args.test_prompts,
        layers=_parse_layers(args.layers),
        ppl_ratio_threshold=args.ppl_ratio_threshold,
        shared_heads_from=args.shared_heads_from,
        mlp_results_from=args.mlp_results_from,
        seed=args.seed,
    )
