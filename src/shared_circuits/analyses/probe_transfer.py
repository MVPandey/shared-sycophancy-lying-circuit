"""Probe transfer: train logistic-regression probe on sycophancy, evaluate on lying."""

from __future__ import annotations

import argparse
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from shared_circuits.config import ALL_MODELS, RANDOM_SEED
from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, model_session, save_results
from shared_circuits.extraction import extract_residual_stream
from shared_circuits.models import cleanup_model, get_agree_disagree_tokens, get_model_info, load_model
from shared_circuits.prompts import build_lying_prompts, build_sycophancy_prompts
from shared_circuits.stats import evaluate_probe_transfer

_DEFAULT_N_PAIRS: Final = 400
_DEFAULT_N_PROMPTS: Final = 100
_DEFAULT_PROBE_LAYER_FRAC: Final = 0.85
_DEFAULT_N_BOOT: Final = 0
# Logistic regression needs at least one positive and one negative example to fit.
_MIN_CLASSES_FOR_FIT: Final = 2


class ProbeTransferConfig(BaseModel):
    """
    Inputs for the probe-transfer analysis.

    Two modes:
        * Default: iterate over ``models`` (multi-model sweep, mirrors the legacy
          ``run_probe_transfer.py``).
        * Single-model: set ``single_model`` (mirrors ``run_probe_transfer_single.py``).
          Adds support for ``weight_repo`` overrides, optional bootstrap CIs, and a
          custom result-file ``tag``.
    """

    model_config = ConfigDict(frozen=True)

    models: tuple[str, ...] = Field(default_factory=lambda: tuple(ALL_MODELS))
    single_model: str | None = Field(default=None)
    n_devices: int = Field(default=1, gt=0)
    weight_repo: str | None = Field(default=None)
    tag: str = Field(default='')
    n_pairs: int = Field(default=_DEFAULT_N_PAIRS, gt=0)
    n_prompts: int = Field(default=_DEFAULT_N_PROMPTS, gt=0)
    probe_layer: int | None = Field(default=None)
    probe_layer_frac: float = Field(default=_DEFAULT_PROBE_LAYER_FRAC, gt=0, lt=1)
    n_boot: int = Field(default=_DEFAULT_N_BOOT, ge=0)
    seed: int = Field(default=RANDOM_SEED)


def run(cfg: ProbeTransferConfig) -> dict | list[dict]:
    """
    Run probe transfer.

    Returns a single dict in single-model mode (mirrors ``run_probe_transfer_single``),
    a list of dicts in the multi-model sweep.
    """
    pairs = load_triviaqa_pairs(cfg.n_pairs)
    if cfg.single_model is not None:
        return _run_one(cfg.single_model, pairs, cfg, single=True)
    return [_run_one(name, pairs, cfg, single=False) for name in cfg.models]


def _run_one(model_name: str, pairs: list[tuple[str, str, str]], cfg: ProbeTransferConfig, *, single: bool) -> dict:
    # ``model_session`` does not accept ``weight_repo`` (gated-mirror override), so the
    # single-model path replicates its lifecycle inline; the multi-model path uses
    # ``model_session`` for parity with the other migrated analyses.
    if cfg.weight_repo is not None:
        model = load_model(model_name, n_devices=cfg.n_devices, weight_repo=cfg.weight_repo)
        try:
            info = get_model_info(model)
            agree, disagree = get_agree_disagree_tokens(model)
            ctx = ExperimentContext(
                model=model,
                info=info,
                model_name=model_name,
                agree_tokens=tuple(agree),
                disagree_tokens=tuple(disagree),
            )
            return _analyse(ctx, pairs, cfg, single=single)
        finally:
            cleanup_model(model)
    with model_session(model_name, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, pairs, cfg, single=single)


def _resolve_probe_layer(info_n_layers: int, cfg: ProbeTransferConfig) -> int:
    if cfg.probe_layer is not None:
        return cfg.probe_layer
    return int(info_n_layers * cfg.probe_layer_frac)


def _analyse(
    ctx: ExperimentContext, pairs: list[tuple[str, str, str]], cfg: ProbeTransferConfig, *, single: bool
) -> dict:
    probe_layer = _resolve_probe_layer(ctx.info.n_layers, cfg)
    set_a, set_b = pairs[:200], pairs[200:400]
    wrong_a, correct_a = build_sycophancy_prompts(set_a, ctx.model_name)
    false_b, true_b = build_lying_prompts(set_b, ctx.model_name)
    n = cfg.n_prompts

    acts_w = extract_residual_stream(ctx.model, wrong_a[:n], probe_layer)
    acts_c = extract_residual_stream(ctx.model, correct_a[:n], probe_layer)
    acts_f = extract_residual_stream(ctx.model, false_b[:n], probe_layer)
    acts_t = extract_residual_stream(ctx.model, true_b[:n], probe_layer)

    syc_to_lie = evaluate_probe_transfer(acts_w, acts_c, acts_f, acts_t)

    if single:
        # Single-model variant also runs the reverse direction and (optionally) bootstraps.
        lie_to_syc = evaluate_probe_transfer(acts_f, acts_t, acts_w, acts_c)
        boot_syc_to_lie = _bootstrap(acts_w, acts_c, acts_f, acts_t, cfg) if cfg.n_boot > 0 else None
        boot_lie_to_syc = _bootstrap(acts_f, acts_t, acts_w, acts_c, cfg) if cfg.n_boot > 0 else None

        result: dict = {
            'model': ctx.model_name,
            'weight_repo': cfg.weight_repo,
            'tag': cfg.tag,
            'probe_layer': probe_layer,
            'n_layers': ctx.info.n_layers,
            'syc_to_lie': syc_to_lie,
            'lie_to_syc': lie_to_syc,
            'syc_to_lie_bootstrap': boot_syc_to_lie,
            'lie_to_syc_bootstrap': boot_lie_to_syc,
        }
        slug = f'probe_transfer_{cfg.tag}' if cfg.tag else 'probe_transfer'
        save_results(result, slug, ctx.model_name)
        return result

    result = {
        'model': ctx.model_name,
        'probe_layer': probe_layer,
        'n_layers': ctx.info.n_layers,
        **syc_to_lie,
    }
    save_results(result, 'probe_transfer', ctx.model_name)
    return result


def _bootstrap(
    x_tr_pos: np.ndarray,
    x_tr_neg: np.ndarray,
    x_te_pos: np.ndarray,
    x_te_neg: np.ndarray,
    cfg: ProbeTransferConfig,
) -> dict[str, float]:
    """Bootstrap AUROC by resampling training activations with replacement."""
    rng = np.random.RandomState(cfg.seed)
    x_tr = np.vstack([x_tr_pos, x_tr_neg])
    y_tr = np.array([1] * len(x_tr_pos) + [0] * len(x_tr_neg))
    x_te = np.vstack([x_te_pos, x_te_neg])
    y_te = np.array([1] * len(x_te_pos) + [0] * len(x_te_neg))
    boots: list[float] = []
    n_tr = len(x_tr)
    for _ in range(cfg.n_boot):
        idx = rng.choice(n_tr, n_tr, replace=True)
        if len(set(y_tr[idx])) < _MIN_CLASSES_FOR_FIT:
            # Resampled fold contains a single class — sklearn would raise.  Skip and resample.
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(x_tr[idx], y_tr[idx])
        proba = clf.predict_proba(x_te)[:, 1]
        boots.append(float(roc_auc_score(y_te, proba)))
    arr = np.array(boots) if boots else np.zeros(0)
    return {
        'mean': float(arr.mean()) if len(arr) else 0.0,
        'ci_lo': float(np.percentile(arr, 2.5)) if len(arr) else 0.0,
        'ci_hi': float(np.percentile(arr, 97.5)) if len(arr) else 0.0,
        'n_boot': len(boots),
    }


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--models', nargs='+', default=list(ALL_MODELS))
    parser.add_argument('--single-model', default=None, help='If set, run the single-model variant on this model.')
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--weight-repo', default=None, help='Override weights from this path/repo (single-model only).')
    parser.add_argument('--tag', default='', help='Suffix for the saved single-model results filename.')
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PAIRS)
    parser.add_argument('--n-prompts', type=int, default=_DEFAULT_N_PROMPTS)
    parser.add_argument(
        '--probe-layer', type=int, default=None, help='Absolute probe layer; overrides --probe-layer-frac.'
    )
    parser.add_argument('--probe-layer-frac', type=float, default=_DEFAULT_PROBE_LAYER_FRAC)
    parser.add_argument(
        '--n-boot', type=int, default=_DEFAULT_N_BOOT, help='Single-model bootstrap CI resamples (0 disables).'
    )
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)


def from_args(args: argparse.Namespace) -> ProbeTransferConfig:
    """Build the validated config from a parsed argparse namespace."""
    return ProbeTransferConfig(
        models=tuple(args.models),
        single_model=args.single_model,
        n_devices=args.n_devices,
        weight_repo=args.weight_repo,
        tag=args.tag,
        n_pairs=args.n_pairs,
        n_prompts=args.n_prompts,
        probe_layer=args.probe_layer,
        probe_layer_frac=args.probe_layer_frac,
        n_boot=args.n_boot,
        seed=args.seed,
    )
