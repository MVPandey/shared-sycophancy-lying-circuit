"""
SAE feature overlap K-sensitivity sweep across multiple top-K cutoffs.

Sweeps K through several values, computing syc∩lie top-K feature overlap +
ratio-vs-chance + hypergeometric p at each. Produces a curve that confirms
the overlap is not an artifact of a specific K.
"""

import argparse
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.config import RANDOM_SEED
from shared_circuits.data import SAE_REPOS, load_sae_for_model, load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, model_session, save_results
from shared_circuits.extraction import encode_prompts
from shared_circuits.prompts import build_lying_prompts, build_sycophancy_prompts
from shared_circuits.stats import head_overlap_hypergeometric

_DEFAULT_MODEL: Final = 'meta-llama/Llama-3.1-8B-Instruct'
_DEFAULT_LAYER: Final = 19
_DEFAULT_K_VALUES: Final[tuple[int, ...]] = (10, 50, 100, 200, 500)
_DEFAULT_N_PROMPTS: Final = 100
_DEFAULT_BATCH: Final = 4


class SaeKSensitivityConfig(BaseModel):
    """Inputs for the SAE K-sensitivity sweep (single model + single layer)."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(default=_DEFAULT_MODEL)
    layer: int = Field(default=_DEFAULT_LAYER, ge=0)
    k_values: tuple[int, ...] = Field(default=_DEFAULT_K_VALUES)
    n_prompts: int = Field(default=_DEFAULT_N_PROMPTS, gt=0)
    n_devices: int = Field(default=1, gt=0)
    batch: int = Field(default=_DEFAULT_BATCH, gt=0)
    seed: int = Field(default=RANDOM_SEED)


def run(cfg: SaeKSensitivityConfig) -> dict:
    """Run the K-sensitivity sweep on ``cfg.model`` at ``cfg.layer``."""
    if cfg.model not in SAE_REPOS:
        raise ValueError(f'No SAE repo registered for {cfg.model}; supported: {sorted(SAE_REPOS)}')
    if not cfg.k_values:
        raise ValueError('cfg.k_values must contain at least one K')

    pairs = load_triviaqa_pairs(max(cfg.n_prompts * 2, 200))[: cfg.n_prompts * 2]
    syc_wrong, syc_correct = build_sycophancy_prompts(pairs[: cfg.n_prompts], cfg.model)
    lie_false, lie_true = build_lying_prompts(pairs[cfg.n_prompts : cfg.n_prompts * 2], cfg.model)
    with model_session(cfg.model, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, syc_wrong, syc_correct, lie_false, lie_true, cfg)


def _analyse(
    ctx: ExperimentContext,
    syc_wrong: list[str],
    syc_correct: list[str],
    lie_false: list[str],
    lie_true: list[str],
    cfg: SaeKSensitivityConfig,
) -> dict:
    sae = load_sae_for_model(ctx.model_name, cfg.layer)
    sw = encode_prompts(ctx.model, syc_wrong, sae, cfg.layer, batch_size=cfg.batch)
    sc = encode_prompts(ctx.model, syc_correct, sae, cfg.layer, batch_size=cfg.batch)
    lf = encode_prompts(ctx.model, lie_false, sae, cfg.layer, batch_size=cfg.batch)
    lt = encode_prompts(ctx.model, lie_true, sae, cfg.layer, batch_size=cfg.batch)

    abs_syc = np.abs(sw.mean(0) - sc.mean(0))
    abs_lie = np.abs(lf.mean(0) - lt.mean(0))
    syc_rank = np.argsort(abs_syc)[::-1]
    lie_rank = np.argsort(abs_lie)[::-1]
    d_sae = int(abs_syc.shape[0])

    curve: list[dict] = []
    for k in cfg.k_values:
        if k > d_sae:
            raise ValueError(f'K={k} exceeds d_sae={d_sae}')
        syc_top = set(syc_rank[:k].tolist())
        lie_top = set(lie_rank[:k].tolist())
        overlap = len(syc_top & lie_top)
        chance = (k * k) / d_sae
        ratio = overlap / chance if chance > 0 else 0.0
        curve.append(
            {
                'k': int(k),
                'overlap': int(overlap),
                'chance': float(chance),
                'ratio': float(ratio),
                'p_hypergeometric': float(head_overlap_hypergeometric(overlap, k, d_sae)),
            }
        )

    result = {
        'model': ctx.model_name,
        'layer': cfg.layer,
        'sae_repo': str(SAE_REPOS[ctx.model_name]['repo']),
        'sae_format': str(SAE_REPOS[ctx.model_name]['format']),
        'sae_average_l0': sae.average_l0,
        'sae_top_k': sae.top_k,
        'd_sae': d_sae,
        'n_prompts_per_task': cfg.n_prompts,
        'k_values': list(cfg.k_values),
        'curve': curve,
    }
    save_results(result, 'sae_k_sensitivity', ctx.model_name)
    return result


def _parse_k_values(value: str | None) -> tuple[int, ...]:
    if value is None:
        return _DEFAULT_K_VALUES
    return tuple(int(x) for x in value.split(',') if x.strip())


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--model', default=_DEFAULT_MODEL)
    parser.add_argument('--layer', type=int, default=_DEFAULT_LAYER)
    parser.add_argument(
        '--k-values',
        type=str,
        default=None,
        help='Comma-separated K values (e.g. "10,50,100,200,500").',
    )
    parser.add_argument('--n-prompts', type=int, default=_DEFAULT_N_PROMPTS)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--batch', type=int, default=_DEFAULT_BATCH)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)


def from_args(args: argparse.Namespace) -> SaeKSensitivityConfig:
    """Build the validated config from a parsed argparse namespace."""
    return SaeKSensitivityConfig(
        model=args.model,
        layer=args.layer,
        k_values=_parse_k_values(args.k_values),
        n_prompts=args.n_prompts,
        n_devices=args.n_devices,
        batch=args.batch,
        seed=args.seed,
    )
