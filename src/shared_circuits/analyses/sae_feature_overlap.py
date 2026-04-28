"""
SAE feature overlap on shared-head layers: superposition vs feature-level sharing.

For every (model, layer) the analysis encodes the residual stream through the SAE
registered for the model, derives per-feature mean-activation differentials between
syc-wrong/syc-correct and lie-false/lie-true, and reports the top-K differentially
active feature overlap with hypergeometric / permutation p-values, Spearman rank
correlation over the full feature dictionary, and a paired bootstrap on overlap.
"""

import argparse
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import spearmanr

from shared_circuits.config import RANDOM_SEED
from shared_circuits.data import SAE_REPOS, load_sae_for_model, load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, model_session, save_results
from shared_circuits.extraction import encode_prompts
from shared_circuits.prompts import build_lying_prompts, build_sycophancy_prompts
from shared_circuits.stats import head_overlap_hypergeometric

# Paper's Table 4 model+layer set. Single layer for Llama variants where Goodfire
# only ships a single SAE; multi-layer for Gemma variants where Gemma-Scope offers
# residual SAEs at every layer.
_DEFAULT_MODEL_LAYERS: Final[tuple[tuple[str, tuple[int, ...]], ...]] = (
    ('gemma-2-2b-it', (12, 19)),
    ('google/gemma-2-9b-it', (21, 31)),
    ('meta-llama/Llama-3.1-8B-Instruct', (19,)),
    ('meta-llama/Llama-3.3-70B-Instruct', (50,)),
)
_DEFAULT_TOP_K: Final = 100
_DEFAULT_N_PROMPTS: Final = 100
_DEFAULT_BATCH: Final = 4
_DEFAULT_N_PERM: Final = 1000
_DEFAULT_N_BOOT: Final = 1000


class SaeFeatureOverlapConfig(BaseModel):
    """Inputs for the SAE feature-overlap analysis."""

    model_config = ConfigDict(frozen=True)

    models: tuple[str, ...] = Field(default_factory=lambda: tuple(m for m, _ in _DEFAULT_MODEL_LAYERS))
    layers: dict[str, tuple[int, ...]] = Field(default_factory=lambda: {m: ls for m, ls in _DEFAULT_MODEL_LAYERS})
    top_k: int = Field(default=_DEFAULT_TOP_K, gt=0)
    n_prompts: int = Field(default=_DEFAULT_N_PROMPTS, gt=0)
    n_devices: int = Field(default=1, gt=0)
    batch: int = Field(default=_DEFAULT_BATCH, gt=0)
    n_perm: int = Field(default=_DEFAULT_N_PERM, gt=0)
    n_boot: int = Field(default=_DEFAULT_N_BOOT, ge=0)
    seed: int = Field(default=RANDOM_SEED)


def run(cfg: SaeFeatureOverlapConfig) -> list[dict]:
    """Run SAE feature-overlap for every model in ``cfg.models``."""
    pairs = load_triviaqa_pairs(max(cfg.n_prompts * 2, 200))[: cfg.n_prompts * 2]
    return [_run_one(name, pairs, cfg) for name in cfg.models]


def _run_one(model_name: str, pairs: list[tuple[str, str, str]], cfg: SaeFeatureOverlapConfig) -> dict:
    layers = cfg.layers.get(model_name)
    if layers is None:
        raise ValueError(f'No layers configured for {model_name}; pass cfg.layers[{model_name!r}]')
    if model_name not in SAE_REPOS:
        raise ValueError(f'No SAE repo registered for {model_name}; supported: {sorted(SAE_REPOS)}')

    syc_wrong, syc_correct = build_sycophancy_prompts(pairs[: cfg.n_prompts], model_name)
    lie_false, lie_true = build_lying_prompts(pairs[cfg.n_prompts : cfg.n_prompts * 2], model_name)
    with model_session(model_name, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, syc_wrong, syc_correct, lie_false, lie_true, layers, cfg)


def _analyse(
    ctx: ExperimentContext,
    syc_wrong: list[str],
    syc_correct: list[str],
    lie_false: list[str],
    lie_true: list[str],
    layers: tuple[int, ...],
    cfg: SaeFeatureOverlapConfig,
) -> dict:
    per_layer = [_run_one_layer(ctx, layer, syc_wrong, syc_correct, lie_false, lie_true, cfg) for layer in layers]
    sae_repo = str(SAE_REPOS[ctx.model_name]['repo'])
    sae_format = str(SAE_REPOS[ctx.model_name]['format'])
    result = {
        'model': ctx.model_name,
        'sae_repo': sae_repo,
        'sae_format': sae_format,
        'config': {
            'n_prompts_per_task': cfg.n_prompts,
            'top_k': cfg.top_k,
            'n_perm': cfg.n_perm,
            'n_boot': cfg.n_boot,
            'seed': cfg.seed,
            'batch': cfg.batch,
        },
        'n_layers': ctx.info.n_layers,
        'n_heads': ctx.info.n_heads,
        'd_model': ctx.info.d_model,
        'per_layer': per_layer,
    }
    save_results(result, 'sae_feature_overlap', ctx.model_name)
    return result


def _run_one_layer(
    ctx: ExperimentContext,
    layer: int,
    syc_wrong: list[str],
    syc_correct: list[str],
    lie_false: list[str],
    lie_true: list[str],
    cfg: SaeFeatureOverlapConfig,
) -> dict:
    sae = load_sae_for_model(ctx.model_name, layer)
    sw = encode_prompts(ctx.model, syc_wrong, sae, layer, batch_size=cfg.batch)
    sc = encode_prompts(ctx.model, syc_correct, sae, layer, batch_size=cfg.batch)
    lf = encode_prompts(ctx.model, lie_false, sae, layer, batch_size=cfg.batch)
    lt = encode_prompts(ctx.model, lie_true, sae, layer, batch_size=cfg.batch)

    syc_diff = sw.mean(0) - sc.mean(0)
    lie_diff = lf.mean(0) - lt.mean(0)
    n_feats = int(syc_diff.shape[0])
    chance = (cfg.top_k * cfg.top_k) / n_feats
    syc_top = set(np.argsort(np.abs(syc_diff))[::-1][: cfg.top_k].tolist())
    lie_top = set(np.argsort(np.abs(lie_diff))[::-1][: cfg.top_k].tolist())
    overlap = len(syc_top & lie_top)
    union = len(syc_top | lie_top)
    jaccard = overlap / union if union else 0.0

    rho_obj = spearmanr(syc_diff, lie_diff)
    rho = float(rho_obj.statistic)
    sp_p = float(rho_obj.pvalue)
    p_perm = _overlap_perm_pvalue(syc_diff, lie_diff, cfg.top_k, cfg.n_perm, cfg.seed)
    p_hyper = float(head_overlap_hypergeometric(overlap, cfg.top_k, n_feats))
    boot = _bootstrap_overlap(sw, sc, lf, lt, cfg.top_k, cfg.n_boot, cfg.seed) if cfg.n_boot else None

    return {
        'layer': layer,
        'sae_average_l0': sae.average_l0,
        'sae_top_k': sae.top_k,
        'd_sae': n_feats,
        'top_k': cfg.top_k,
        'overlap': int(overlap),
        'chance_overlap': float(chance),
        'ratio_vs_chance': float(overlap / chance) if chance > 0 else 0.0,
        'p_permutation': float(p_perm),
        'p_hypergeometric': p_hyper,
        'jaccard': float(jaccard),
        'spearman_rho': rho,
        'spearman_p': sp_p,
        'bootstrap': boot,
        'syc_top_features': sorted(int(x) for x in syc_top),
        'lie_top_features': sorted(int(x) for x in lie_top),
        'shared_features': sorted(int(x) for x in (syc_top & lie_top)),
    }


def _overlap_perm_pvalue(
    syc_diff: np.ndarray,
    lie_diff: np.ndarray,
    top_k: int,
    n_perm: int,
    seed: int,
) -> float:
    """Permutation null over feature labels: shuffle ``lie_diff`` and recount the top-K intersection."""
    abs_syc = np.abs(syc_diff)
    abs_lie = np.abs(lie_diff)
    syc_top = set(np.argsort(abs_syc)[::-1][:top_k].tolist())
    lie_top = set(np.argsort(abs_lie)[::-1][:top_k].tolist())
    actual = len(syc_top & lie_top)
    rng = np.random.RandomState(seed)
    n = abs_lie.shape[0]
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        null_top = set(np.argsort(abs_lie[perm])[::-1][:top_k].tolist())
        if len(syc_top & null_top) >= actual:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def _bootstrap_overlap(
    sw: np.ndarray,
    sc: np.ndarray,
    lf: np.ndarray,
    lt: np.ndarray,
    top_k: int,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """Paired bootstrap CIs for overlap, Jaccard, and Spearman rank correlation."""
    rng = np.random.RandomState(seed)
    n_sw, n_sc, n_lf, n_lt = len(sw), len(sc), len(lf), len(lt)
    overlaps = np.empty(n_boot, dtype=np.int64)
    jaccards = np.empty(n_boot, dtype=np.float64)
    rhos = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sw_b = sw[rng.choice(n_sw, n_sw, replace=True)].mean(0)
        sc_b = sc[rng.choice(n_sc, n_sc, replace=True)].mean(0)
        lf_b = lf[rng.choice(n_lf, n_lf, replace=True)].mean(0)
        lt_b = lt[rng.choice(n_lt, n_lt, replace=True)].mean(0)
        sd = sw_b - sc_b
        ld = lf_b - lt_b
        s_top = set(np.argsort(np.abs(sd))[::-1][:top_k].tolist())
        l_top = set(np.argsort(np.abs(ld))[::-1][:top_k].tolist())
        ov = len(s_top & l_top)
        overlaps[i] = ov
        union = len(s_top | l_top)
        jaccards[i] = ov / union if union else 0.0
        rhos[i] = float(spearmanr(sd, ld).statistic)
    return {
        'overlap_mean': float(overlaps.mean()),
        'overlap_ci': [float(np.percentile(overlaps, 2.5)), float(np.percentile(overlaps, 97.5))],
        'jaccard_mean': float(jaccards.mean()),
        'jaccard_ci': [float(np.percentile(jaccards, 2.5)), float(np.percentile(jaccards, 97.5))],
        'spearman_rho_mean': float(rhos.mean()),
        'spearman_rho_ci': [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))],
        'n_boot': n_boot,
    }


def _parse_layers_arg(layers_args: list[str] | None, models: list[str]) -> dict[str, tuple[int, ...]]:
    """Parse ``--layers MODEL=L1,L2 ...`` overrides; fall back to defaults from :data:`_DEFAULT_MODEL_LAYERS`."""
    out: dict[str, tuple[int, ...]] = {m: ls for m, ls in _DEFAULT_MODEL_LAYERS}
    if layers_args:
        for spec in layers_args:
            if '=' not in spec:
                raise ValueError(f'Expected MODEL=L1,L2 for --layers, got {spec!r}')
            model, csv = spec.split('=', 1)
            out[model] = tuple(int(x) for x in csv.split(',') if x.strip())
    # Drop unused entries so the saved config is minimal.
    return {m: out[m] for m in models if m in out}


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--models', nargs='+', default=[m for m, _ in _DEFAULT_MODEL_LAYERS])
    parser.add_argument(
        '--layers',
        nargs='+',
        default=None,
        help='Per-model layer override as MODEL=L1,L2 (e.g. gemma-2-2b-it=12,19).',
    )
    parser.add_argument('--top-k', type=int, default=_DEFAULT_TOP_K)
    parser.add_argument('--n-prompts', type=int, default=_DEFAULT_N_PROMPTS)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--batch', type=int, default=_DEFAULT_BATCH)
    parser.add_argument('--n-perm', type=int, default=_DEFAULT_N_PERM)
    parser.add_argument('--n-boot', type=int, default=_DEFAULT_N_BOOT)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)


def from_args(args: argparse.Namespace) -> SaeFeatureOverlapConfig:
    """Build the validated config from a parsed argparse namespace."""
    layers = _parse_layers_arg(args.layers, args.models)
    return SaeFeatureOverlapConfig(
        models=tuple(args.models),
        layers=layers,
        top_k=args.top_k,
        n_prompts=args.n_prompts,
        n_devices=args.n_devices,
        batch=args.batch,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        seed=args.seed,
    )
