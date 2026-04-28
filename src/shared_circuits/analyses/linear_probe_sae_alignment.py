"""
Probe-direction vs SAE-feature alignment for shared circuits.

Train a binary logistic-regression probe in residual stream at the SAE layer to predict
syc_wrong vs syc_correct (and lie_false vs lie_true). Project the probe direction onto
each SAE decoder column to rank features by alignment with the truth direction. If the
top-K |aligned| SAE features overlap heavily with the *shared* features identified by
the SAE feature-overlap analysis, this directly says: the model's truth direction in
residual space *is* the shared SAE feature subspace.

Reports:
  - probe AUROC on each task (held-out k-fold CV)
  - top-K SAE features by |cosine(probe_direction, W_dec[:, i])|
  - intersection between top-K aligned and shared features (vs random expectation)
  - Spearman rank correlation between alignment magnitude and absolute mean-activation difference
  - fraction of probe norm captured by the shared-feature subspace, with permutation null
"""

import argparse
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from shared_circuits.config import RANDOM_SEED
from shared_circuits.data import SAE_REPOS, load_sae_for_model, load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, load_results, model_session, save_results
from shared_circuits.extraction import extract_residual_stream
from shared_circuits.prompts import build_lying_prompts, build_sycophancy_prompts

_DEFAULT_MODEL: Final = 'meta-llama/Llama-3.1-8B-Instruct'
_DEFAULT_LAYER: Final = 19
_DEFAULT_N_PROMPTS: Final = 100
_DEFAULT_BATCH: Final = 4
# 41 matches the legacy default — chosen to align with the typical shared-feature count.
_DEFAULT_TOP_K_OVERLAP: Final = 41
_DEFAULT_N_FOLDS: Final = 5
# Subspace-norm null distribution — 100 keeps the test cheap while still informative.
_DEFAULT_N_PERM_SUBSPACE: Final = 100


class LinearProbeSaeAlignmentConfig(BaseModel):
    """Inputs for the linear-probe / SAE alignment analysis (single model + single layer)."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(default=_DEFAULT_MODEL)
    layer: int = Field(default=_DEFAULT_LAYER, ge=0)
    n_prompts: int = Field(default=_DEFAULT_N_PROMPTS, gt=0)
    n_devices: int = Field(default=1, gt=0)
    batch: int = Field(default=_DEFAULT_BATCH, gt=0)
    top_k_overlap: int = Field(default=_DEFAULT_TOP_K_OVERLAP, gt=0)
    n_folds: int = Field(default=_DEFAULT_N_FOLDS, ge=2)
    n_perm_subspace: int = Field(default=_DEFAULT_N_PERM_SUBSPACE, gt=0)
    seed: int = Field(default=RANDOM_SEED)
    overlap_from: str = Field(
        default='sae_feature_overlap',
        description='Slug whose saved results provide the shared-features list.',
    )


def run(cfg: LinearProbeSaeAlignmentConfig) -> dict:
    """Train probes, project onto SAE decoder columns, and report alignment with shared features."""
    if cfg.model not in SAE_REPOS:
        raise ValueError(f'No SAE repo registered for {cfg.model}; supported: {sorted(SAE_REPOS)}')

    overlap_payload = load_results(cfg.overlap_from, cfg.model)
    entry = next((e for e in overlap_payload['per_layer'] if e['layer'] == cfg.layer), None)
    if entry is None:
        raise FileNotFoundError(
            f'Layer {cfg.layer} not found in {cfg.overlap_from} results for {cfg.model}; run sae-feature-overlap first.'
        )
    shared = [int(x) for x in entry['shared_features']]

    pairs = load_triviaqa_pairs(max(cfg.n_prompts * 2, 200))[: cfg.n_prompts * 2]
    syc_wrong, syc_correct = build_sycophancy_prompts(pairs[: cfg.n_prompts], cfg.model)
    lie_false, lie_true = build_lying_prompts(pairs[cfg.n_prompts : cfg.n_prompts * 2], cfg.model)
    with model_session(cfg.model, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, syc_wrong, syc_correct, lie_false, lie_true, shared, cfg)


def _analyse(
    ctx: ExperimentContext,
    syc_wrong: list[str],
    syc_correct: list[str],
    lie_false: list[str],
    lie_true: list[str],
    shared: list[int],
    cfg: LinearProbeSaeAlignmentConfig,
) -> dict:
    res_sw = extract_residual_stream(ctx.model, syc_wrong, cfg.layer, batch_size=cfg.batch)
    res_sc = extract_residual_stream(ctx.model, syc_correct, cfg.layer, batch_size=cfg.batch)
    res_lf = extract_residual_stream(ctx.model, lie_false, cfg.layer, batch_size=cfg.batch)
    res_lt = extract_residual_stream(ctx.model, lie_true, cfg.layer, batch_size=cfg.batch)

    sae = load_sae_for_model(ctx.model_name, cfg.layer)
    if sae.w_dec is None:
        raise ValueError(f'SAE for {ctx.model_name} at layer {cfg.layer} did not load W_dec')
    w_dec = sae.w_dec.float().cpu().numpy()
    d_sae = int(w_dec.shape[1])

    # Per-feature absolute mean-activation differences via decoder projection.
    abs_syc_feat = np.abs((res_sw.mean(0) - res_sc.mean(0)) @ w_dec)
    abs_lie_feat = np.abs((res_lf.mean(0) - res_lt.mean(0)) @ w_dec)

    syc_block = _probe_block(res_sw, res_sc, w_dec, shared, d_sae, abs_syc_feat, cfg)
    lie_block = _probe_block(res_lf, res_lt, w_dec, shared, d_sae, abs_lie_feat, cfg)

    result = {
        'model': ctx.model_name,
        'layer': cfg.layer,
        'sae_repo': str(SAE_REPOS[ctx.model_name]['repo']),
        'sae_format': str(SAE_REPOS[ctx.model_name]['format']),
        'config': {
            'n_prompts_per_task': cfg.n_prompts,
            'top_k_overlap': cfg.top_k_overlap,
            'n_folds': cfg.n_folds,
            'n_perm_subspace': cfg.n_perm_subspace,
            'seed': cfg.seed,
            'batch': cfg.batch,
        },
        'd_model': ctx.info.d_model,
        'd_sae': d_sae,
        'n_shared_features': len(shared),
        'shared_features': shared,
        'syc_probe': syc_block,
        'lie_probe': lie_block,
    }
    save_results(result, 'linear_probe_sae_alignment', ctx.model_name)
    return result


def _probe_block(
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    w_dec: np.ndarray,
    shared: list[int],
    d_sae: int,
    abs_diff: np.ndarray,
    cfg: LinearProbeSaeAlignmentConfig,
) -> dict:
    probe, auroc = _train_probe_cv(pos_acts, neg_acts, cfg.n_folds, cfg.seed)
    align = _alignment_per_feature(probe, w_dec)
    top_aligned = [int(x) for x in np.argsort(align)[::-1][: cfg.top_k_overlap].tolist()]
    stats = _overlap_stats(top_aligned, shared, d_sae, cfg.top_k_overlap)
    rho_obj = spearmanr(align, abs_diff)
    rho = float(rho_obj.statistic)
    p_sp = float(rho_obj.pvalue)
    frac, null = _subspace_norm_block(probe, w_dec, shared, d_sae, cfg.n_perm_subspace, cfg.seed)
    return {
        'auroc_cv': float(auroc),
        'overlap_stats': stats,
        'top_aligned': top_aligned,
        'spearman_align_vs_absdiff': {'rho': rho, 'p_value': p_sp},
        'subspace_norm_fraction': {
            'shared': float(frac),
            'null_mean': float(null['mean']),
            'null_std': float(null['std']),
            'null_max': float(null['max']),
            'n_perm': cfg.n_perm_subspace,
            'p_permutation': float(null['p_value']),
        },
    }


def _train_probe_cv(
    x_pos: np.ndarray,
    x_neg: np.ndarray,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Stratified K-fold CV; returns (full-data probe coefficient vector, mean held-out AUROC)."""
    x = np.concatenate([x_pos, x_neg])
    y = np.array([1] * len(x_pos) + [0] * len(x_neg))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aurocs: list[float] = []
    for tr, te in skf.split(x, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(x[tr], y[tr])
        proba = clf.predict_proba(x[te])[:, 1]
        aurocs.append(float(roc_auc_score(y[te], proba)))
    full = LogisticRegression(max_iter=2000, C=1.0).fit(x, y)
    return full.coef_[0].astype(np.float32), float(np.mean(aurocs))


def _alignment_per_feature(probe: np.ndarray, w_dec: np.ndarray) -> np.ndarray:
    """Cosine between probe direction and each SAE decoder column. ``w_dec`` shape: ``(d_model, d_sae)``."""
    p = probe / (np.linalg.norm(probe) + 1e-12)
    norms = np.linalg.norm(w_dec, axis=0) + 1e-12
    return np.abs((p @ w_dec) / norms)


def _overlap_stats(top_aligned: list[int], shared: list[int], d_sae: int, top_k: int) -> dict:
    a = set(top_aligned)
    s = set(shared)
    overlap = len(a & s)
    chance = (top_k * len(s)) / d_sae
    return {
        'overlap': int(overlap),
        'top_k_aligned': int(top_k),
        'n_shared': len(s),
        'd_sae': int(d_sae),
        'chance_overlap': float(chance),
        'ratio_vs_chance': float(overlap / chance) if chance > 0 else 0.0,
    }


def _subspace_norm_fraction(probe: np.ndarray, w_dec: np.ndarray, feature_idx: list[int]) -> float:
    """Fraction of probe norm captured by the subspace spanned by ``W_dec[:, feature_idx]``."""
    cols = w_dec[:, feature_idx]
    q, _ = np.linalg.qr(cols)
    proj = q @ (q.T @ probe)
    return float(np.linalg.norm(proj) / (np.linalg.norm(probe) + 1e-12))


def _subspace_norm_block(
    probe: np.ndarray,
    w_dec: np.ndarray,
    shared: list[int],
    d_sae: int,
    n_perm: int,
    seed: int,
) -> tuple[float, dict[str, float]]:
    """Return (shared-subspace fraction, null distribution stats + p-value)."""
    frac = _subspace_norm_fraction(probe, w_dec, shared)
    rng = np.random.RandomState(seed)
    null = np.array(
        [
            _subspace_norm_fraction(probe, w_dec, rng.choice(d_sae, len(shared), replace=False).tolist())
            for _ in range(n_perm)
        ]
    )
    p_perm = float((np.sum(null >= frac) + 1) / (n_perm + 1))
    return frac, {
        'mean': float(null.mean()),
        'std': float(null.std()),
        'max': float(null.max()),
        'p_value': p_perm,
    }


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--model', default=_DEFAULT_MODEL)
    parser.add_argument('--layer', type=int, default=_DEFAULT_LAYER)
    parser.add_argument('--n-prompts', type=int, default=_DEFAULT_N_PROMPTS)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--batch', type=int, default=_DEFAULT_BATCH)
    parser.add_argument('--top-k-overlap', type=int, default=_DEFAULT_TOP_K_OVERLAP)
    parser.add_argument('--n-folds', type=int, default=_DEFAULT_N_FOLDS)
    parser.add_argument('--n-perm-subspace', type=int, default=_DEFAULT_N_PERM_SUBSPACE)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument(
        '--overlap-from',
        default='sae_feature_overlap',
        help='Slug whose saved results provide the shared-features list.',
    )


def from_args(args: argparse.Namespace) -> LinearProbeSaeAlignmentConfig:
    """Build the validated config from a parsed argparse namespace."""
    return LinearProbeSaeAlignmentConfig(
        model=args.model,
        layer=args.layer,
        n_prompts=args.n_prompts,
        n_devices=args.n_devices,
        batch=args.batch,
        top_k_overlap=args.top_k_overlap,
        n_folds=args.n_folds,
        n_perm_subspace=args.n_perm_subspace,
        seed=args.seed,
        overlap_from=args.overlap_from,
    )
