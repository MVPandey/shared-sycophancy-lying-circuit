"""Logit-lens trajectory: per-layer DIFF on sycophantic vs non-sycophantic trials."""

from __future__ import annotations

import argparse
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.config import RANDOM_SEED
from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, model_session, save_results
from shared_circuits.extraction import extract_residual_stream_multi, measure_agreement_per_prompt
from shared_circuits.prompts import build_sycophancy_prompts

_DEFAULT_N_PAIRS: Final = 200
_DEFAULT_N_PERM: Final = 1000
_DEFAULT_N_BOOT: Final = 1000
_DEFAULT_BATCH: Final = 4
_PERM_SIG_QUANTILE: Final = 0.95
_PERM_P_THRESH: Final = 0.05
_NEAR_ZERO: Final = 1e-6


class LogitLensConfig(BaseModel):
    """Inputs for the logit-lens trajectory analysis (single model)."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(...)
    n_devices: int = Field(default=1, gt=0)
    n_pairs: int = Field(default=_DEFAULT_N_PAIRS, gt=0)
    n_perm: int = Field(default=_DEFAULT_N_PERM, gt=0)
    n_boot: int = Field(default=_DEFAULT_N_BOOT, gt=0)
    batch: int = Field(default=_DEFAULT_BATCH, gt=0)
    seed: int = Field(default=RANDOM_SEED)


def run(cfg: LogitLensConfig) -> dict:
    """Run logit-lens trajectory analysis on ``cfg.model``."""
    pairs = load_triviaqa_pairs(cfg.n_pairs)
    with model_session(cfg.model, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, pairs, cfg)


def _analyse(ctx: ExperimentContext, pairs: list[tuple[str, str, str]], cfg: LogitLensConfig) -> dict:
    wrong_prompts, _ = build_sycophancy_prompts(pairs, ctx.model_name)

    layers = list(range(ctx.info.n_layers))
    resid_by_layer = extract_residual_stream_multi(ctx.model, wrong_prompts, layers, batch_size=cfg.batch)

    # Per-prompt logit-diff trajectory: rows = prompts, cols = layers.
    traj = _project_to_logit_diff(ctx, resid_by_layer, layers)

    # Classify each prompt as sycophantic (agree wins at last token) vs not.
    _, indicators = measure_agreement_per_prompt(
        ctx.model,
        wrong_prompts,
        ctx.agree_tokens,
        ctx.disagree_tokens,
        batch_size=cfg.batch,
    )
    is_syc = np.array(indicators, dtype=bool)
    syc_mat = traj[is_syc]
    non_mat = traj[~is_syc]

    syc_stats = _per_layer_stats(syc_mat, cfg.n_boot, cfg.seed)
    non_stats = _per_layer_stats(non_mat, cfg.n_boot, cfg.seed)

    perm_null = _permutation_null(syc_mat, non_mat, cfg.n_perm, cfg.seed)
    diff_per_layer = _diff_per_layer(syc_stats, non_stats)
    peak_layer, peak_excess = _peak_excess(diff_per_layer)

    perm_pvalue = perm_null.get('p_value_per_layer', []) if perm_null is not None else []
    sig_layers = [int(layer) for layer, p in enumerate(perm_pvalue) if p < _PERM_P_THRESH]

    result = {
        'model': ctx.model_name,
        'n_layers': ctx.info.n_layers,
        'layers': layers,
        'n_pairs': len(wrong_prompts),
        'n_sycophantic': int(is_syc.sum()),
        'n_non_sycophantic': int((~is_syc).sum()),
        'sycophantic_trajectory': syc_stats,
        'non_sycophantic_trajectory': non_stats,
        'diff_per_layer': diff_per_layer,
        'peak_layer': peak_layer,
        'peak_excess': peak_excess,
        'perm_null_pvalue': perm_pvalue,
        'permutation_null': perm_null,
        'significant_layers': sig_layers,
        'config': {
            'n_pairs': cfg.n_pairs,
            'n_perm': cfg.n_perm,
            'n_boot': cfg.n_boot,
            'batch': cfg.batch,
            'seed': cfg.seed,
        },
    }
    save_results(result, 'logit_lens', ctx.model_name)
    return result


@torch.no_grad()
def _project_to_logit_diff(
    ctx: ExperimentContext,
    resid_by_layer: dict[int, np.ndarray],
    layers: list[int],
) -> np.ndarray:
    """Return matrix of shape ``(n_prompts, n_layers)`` with logit-lens DIFF per cell."""
    model = ctx.model
    # ln_final + W_U live on the last device under pipeline parallel; route projection there.
    unembed_device = model.W_U.device
    agree_idx = list(ctx.agree_tokens)
    disagree_idx = list(ctx.disagree_tokens)
    n_prompts = next(iter(resid_by_layer.values())).shape[0]
    out = np.zeros((n_prompts, len(layers)), dtype=np.float32)
    for col, layer in enumerate(layers):
        resid = torch.from_numpy(resid_by_layer[layer]).to(unembed_device, dtype=torch.float32)
        # ln_final expects (batch, seq, d_model); fold a singleton seq axis to satisfy it.
        normed = model.ln_final(resid.unsqueeze(1))[:, 0, :]
        logits = normed @ model.W_U.float()
        agree_max = logits[:, agree_idx].max(dim=-1).values
        disagree_max = logits[:, disagree_idx].max(dim=-1).values
        out[:, col] = (disagree_max - agree_max).cpu().numpy()
    return out


def _per_layer_stats(mat: np.ndarray, n_boot: int, seed: int) -> dict:
    """Per-layer mean and bootstrap 95% CI across rows of ``mat``."""
    if mat.size == 0:
        return {'n': 0, 'mean': [], 'ci_lo': [], 'ci_hi': []}
    means = mat.mean(axis=0)
    rng = np.random.RandomState(seed)
    boots = np.stack([mat[rng.choice(len(mat), len(mat), replace=True)].mean(axis=0) for _ in range(n_boot)])
    return {
        'n': len(mat),
        'mean': means.tolist(),
        'ci_lo': np.percentile(boots, 2.5, axis=0).tolist(),
        'ci_hi': np.percentile(boots, 97.5, axis=0).tolist(),
    }


def _permutation_null(syc_mat: np.ndarray, non_mat: np.ndarray, n_perm: int, seed: int) -> dict | None:
    """Shuffle syc/non-syc labels; return per-layer two-sided p-value of observed DIFF."""
    if syc_mat.size == 0 or non_mat.size == 0:
        return None
    combined = np.vstack([syc_mat, non_mat])
    n_syc = len(syc_mat)
    rng = np.random.RandomState(seed)
    perm_diffs = np.zeros((n_perm, combined.shape[1]), dtype=np.float32)
    for i in range(n_perm):
        idx = rng.permutation(len(combined))
        syc_fake = combined[idx[:n_syc]].mean(axis=0)
        non_fake = combined[idx[n_syc:]].mean(axis=0)
        perm_diffs[i] = non_fake - syc_fake
    real_diff = non_mat.mean(axis=0) - syc_mat.mean(axis=0)
    p_per_layer = (np.abs(perm_diffs) >= np.abs(real_diff)[None, :]).mean(axis=0).tolist()
    sig_quantile_hi = np.percentile(perm_diffs, _PERM_SIG_QUANTILE * 100, axis=0).tolist()
    return {
        'n_perm': n_perm,
        'null_diff_mean': perm_diffs.mean(axis=0).tolist(),
        'null_diff_ci_lo': np.percentile(perm_diffs, 2.5, axis=0).tolist(),
        'null_diff_ci_hi': np.percentile(perm_diffs, 97.5, axis=0).tolist(),
        'null_diff_q95': sig_quantile_hi,
        'real_diff': real_diff.tolist(),
        'p_value_per_layer': p_per_layer,
    }


def _diff_per_layer(syc_stats: dict, non_stats: dict) -> list[float]:
    if not syc_stats['mean'] or not non_stats['mean']:
        return []
    return (np.array(non_stats['mean']) - np.array(syc_stats['mean'])).tolist()


def _peak_excess(diff_per_layer: list[float]) -> tuple[int | None, dict | None]:
    if not diff_per_layer:
        return None, None
    arr = np.array(diff_per_layer)
    peak_layer = int(np.argmax(np.abs(arr)))
    peak_abs = float(arr[peak_layer])
    final = float(arr[-1])
    excess = float(abs(peak_abs) - abs(final))
    excess_ratio = float(excess / abs(final)) if abs(final) > _NEAR_ZERO else float('nan')
    return peak_layer, {
        'peak_diff': peak_abs,
        'peak_layer': peak_layer,
        'final_diff': final,
        'excess_above_final': excess,
        'excess_ratio': excess_ratio,
    }


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PAIRS)
    parser.add_argument('--n-perm', type=int, default=_DEFAULT_N_PERM)
    parser.add_argument('--n-boot', type=int, default=_DEFAULT_N_BOOT)
    parser.add_argument('--batch', type=int, default=_DEFAULT_BATCH)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)


def from_args(args: argparse.Namespace) -> LogitLensConfig:
    """Build the validated config from a parsed argparse namespace."""
    return LogitLensConfig(
        model=args.model,
        n_devices=args.n_devices,
        n_pairs=args.n_pairs,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        batch=args.batch,
        seed=args.seed,
    )
