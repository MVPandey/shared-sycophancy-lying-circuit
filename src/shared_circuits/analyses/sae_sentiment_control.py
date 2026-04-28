"""
Sentiment-control overlap test for the shared SAE-feature circuit.

If syc features overlap with lie features at >> chance but syc features overlap
with sentiment features near chance, the shared circuit is specific to factual
correctness rather than generic binary evaluation. The analysis loads a previously
saved ``sae_feature_overlap`` result for the syc/lie reference, encodes positive
vs negative movie-review prompts through the same SAE, and computes the three
pairwise top-K overlaps. McNemar exact test compares whether syc-and-lie
agreement exceeds syc-and-sentiment agreement at the feature level.
"""

import argparse
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import binomtest

from shared_circuits.config import RANDOM_SEED
from shared_circuits.data import SAE_REPOS, load_sae_for_model
from shared_circuits.experiment import ExperimentContext, load_results, model_session, save_results
from shared_circuits.extraction import encode_prompts
from shared_circuits.prompts import render_chat
from shared_circuits.stats import head_overlap_hypergeometric

_DEFAULT_MODEL: Final = 'meta-llama/Llama-3.1-8B-Instruct'
_DEFAULT_LAYER: Final = 19
_DEFAULT_N_PROMPTS: Final = 100
_DEFAULT_BATCH: Final = 4
_DEFAULT_TOP_K: Final = 100
_DEFAULT_N_PERM: Final = 1000

# Fixed templates copied from the legacy run script so the control prompt
# distribution stays bit-for-bit reproducible.
POSITIVE_TEMPLATES: Final[tuple[str, ...]] = (
    'This movie was absolutely wonderful. The acting was superb and the plot kept me engaged throughout.',
    'I loved every minute of this film. A masterpiece of storytelling and visual beauty.',
    'One of the best movies I have ever seen. The director did an outstanding job.',
    'A fantastic film with brilliant performances. I would highly recommend it to anyone.',
    'This was an incredibly moving and well-crafted movie. Truly a work of art.',
    'The film exceeded all my expectations. Every scene was perfectly executed.',
    'A delightful movie with great humor and heartfelt moments. Five stars.',
    'Stunning cinematography and a gripping narrative. This film deserves every award.',
    'I was completely captivated from start to finish. An absolute gem of a movie.',
    'The performances were outstanding and the script was brilliantly written.',
)

NEGATIVE_TEMPLATES: Final[tuple[str, ...]] = (
    'This movie was terrible. The acting was wooden and the plot made no sense.',
    'I hated every minute of this film. A complete waste of time and money.',
    'One of the worst movies I have ever seen. The director should be embarrassed.',
    'A dreadful film with awful performances. I would not recommend it to anyone.',
    'This was an incredibly boring and poorly made movie. Truly a waste of talent.',
    'The film fell far below my expectations. Every scene felt forced and artificial.',
    'A painful movie with no humor and no heart. Zero stars.',
    'Terrible cinematography and a nonsensical narrative. This film deserves nothing.',
    'I was completely bored from start to finish. An absolute disaster of a movie.',
    'The performances were atrocious and the script was laughably bad.',
)


class SaeSentimentControlConfig(BaseModel):
    """Inputs for the SAE sentiment-control analysis (single model + single layer)."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(default=_DEFAULT_MODEL)
    layer: int = Field(default=_DEFAULT_LAYER, ge=0)
    n_prompts: int = Field(default=_DEFAULT_N_PROMPTS, gt=0)
    n_devices: int = Field(default=1, gt=0)
    batch: int = Field(default=_DEFAULT_BATCH, gt=0)
    top_k: int = Field(default=_DEFAULT_TOP_K, gt=0)
    n_perm: int = Field(default=_DEFAULT_N_PERM, gt=0)
    seed: int = Field(default=RANDOM_SEED)
    reference_from: str = Field(
        default='sae_feature_overlap',
        description='Slug whose saved results provide the syc/lie reference top features.',
    )


def run(cfg: SaeSentimentControlConfig) -> dict:
    """Run the sentiment control on ``cfg.model`` at ``cfg.layer``."""
    if cfg.model not in SAE_REPOS:
        raise ValueError(f'No SAE repo registered for {cfg.model}; supported: {sorted(SAE_REPOS)}')

    overlap_payload = load_results(cfg.reference_from, cfg.model)
    entry = next((e for e in overlap_payload['per_layer'] if e['layer'] == cfg.layer), None)
    if entry is None:
        raise FileNotFoundError(
            f'Layer {cfg.layer} not found in {cfg.reference_from} results for {cfg.model}; '
            f'run sae-feature-overlap first.'
        )

    pos_prompts, neg_prompts = build_sentiment_prompts(cfg.model, cfg.n_prompts, cfg.seed)
    with model_session(cfg.model, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, pos_prompts, neg_prompts, entry, cfg)


def _analyse(
    ctx: ExperimentContext,
    pos_prompts: list[str],
    neg_prompts: list[str],
    reference_entry: dict,
    cfg: SaeSentimentControlConfig,
) -> dict:
    sae = load_sae_for_model(ctx.model_name, cfg.layer)
    acts_pos = encode_prompts(ctx.model, pos_prompts, sae, cfg.layer, batch_size=cfg.batch)
    acts_neg = encode_prompts(ctx.model, neg_prompts, sae, cfg.layer, batch_size=cfg.batch)

    sent_diff = acts_pos.mean(0) - acts_neg.mean(0)
    sent_top = set(np.argsort(np.abs(sent_diff))[::-1][: cfg.top_k].tolist())

    syc_top = {int(x) for x in reference_entry['syc_top_features']}
    lie_top = {int(x) for x in reference_entry['lie_top_features']}
    shared = {int(x) for x in reference_entry['shared_features']}
    d_sae = int(reference_entry['d_sae'])
    syc_lie_overlap = int(reference_entry['overlap'])
    chance = (cfg.top_k * cfg.top_k) / d_sae

    ov_syc_sent, p_syc_sent = _overlap_perm(syc_top, sent_diff, cfg.top_k, cfg.n_perm, cfg.seed)
    ov_lie_sent, p_lie_sent = _overlap_perm(lie_top, sent_diff, cfg.top_k, cfg.n_perm, cfg.seed + 1)
    shared_sent_overlap = len(shared & sent_top)
    p_hyper_syc_lie = float(head_overlap_hypergeometric(syc_lie_overlap, cfg.top_k, d_sae))
    p_hyper_syc_sent = float(head_overlap_hypergeometric(ov_syc_sent, cfg.top_k, d_sae))
    p_hyper_lie_sent = float(head_overlap_hypergeometric(ov_lie_sent, cfg.top_k, d_sae))
    mcnemar = _mcnemar_overlap(syc_top, lie_top, sent_top)

    result = {
        'model': ctx.model_name,
        'layer': cfg.layer,
        'sae_repo': str(SAE_REPOS[ctx.model_name]['repo']),
        'sae_format': str(SAE_REPOS[ctx.model_name]['format']),
        'd_sae': d_sae,
        'top_k': cfg.top_k,
        'chance_overlap': float(chance),
        'n_sentiment_prompts': cfg.n_prompts,
        'reference_syc_lie_overlap': syc_lie_overlap,
        'reference_syc_lie_ratio': float(syc_lie_overlap / chance) if chance > 0 else 0.0,
        'reference_syc_lie_p_hyper': p_hyper_syc_lie,
        'syc_sent_overlap': int(ov_syc_sent),
        'syc_sent_ratio': float(ov_syc_sent / chance) if chance > 0 else 0.0,
        'syc_sent_p_perm': float(p_syc_sent),
        'syc_sent_p_hyper': p_hyper_syc_sent,
        'lie_sent_overlap': int(ov_lie_sent),
        'lie_sent_ratio': float(ov_lie_sent / chance) if chance > 0 else 0.0,
        'lie_sent_p_perm': float(p_lie_sent),
        'lie_sent_p_hyper': p_hyper_lie_sent,
        'shared_sent_overlap': int(shared_sent_overlap),
        'mcnemar_syc_lie_vs_syc_sent': mcnemar,
        'sentiment_top_features': sorted(int(x) for x in sent_top),
    }
    save_results(result, 'sae_sentiment_control', ctx.model_name)
    return result


def build_sentiment_prompts(model_name: str, n: int, seed: int) -> tuple[list[str], list[str]]:
    """Build ``n`` positive and ``n`` negative movie-review prompts using ``model_name``'s chat template."""
    rng = np.random.RandomState(seed)
    pos: list[str] = []
    neg: list[str] = []
    for _ in range(n):
        pi = int(rng.randint(len(POSITIVE_TEMPLATES)))
        ni = int(rng.randint(len(NEGATIVE_TEMPLATES)))
        pos.append(_format_review(POSITIVE_TEMPLATES[pi], model_name))
        neg.append(_format_review(NEGATIVE_TEMPLATES[ni], model_name))
    return pos, neg


def _format_review(review: str, model_name: str) -> str:
    return render_chat(
        [
            {
                'role': 'user',
                'content': f'Is the following movie review positive or negative?\n"{review}"',
            }
        ],
        model_name,
    )


def _overlap_perm(
    set_a: set[int],
    flat_b: np.ndarray,
    top_k: int,
    n_perm: int,
    seed: int,
) -> tuple[int, float]:
    """Permutation null over ``flat_b`` labels: shuffle and recount top-K overlap with ``set_a``."""
    abs_b = np.abs(flat_b)
    top_b = set(np.argsort(abs_b)[::-1][:top_k].tolist())
    actual = len(set_a & top_b)
    rng = np.random.RandomState(seed)
    n = abs_b.shape[0]
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        if len(set_a & set(np.argsort(abs_b[perm])[::-1][:top_k].tolist())) >= actual:
            ge += 1
    return actual, (ge + 1) / (n_perm + 1)


def _mcnemar_overlap(syc_top: set[int], lie_top: set[int], sent_top: set[int]) -> dict:
    """
    Run a McNemar-style exact test for syc-and-lie vs syc-and-sentiment feature agreement.

    Counts features in syc-and-lie that are not in syc-and-sent against features in
    syc-and-sent that are not in syc-and-lie, then tests whether the discordant counts
    are exchangeable under the null.
    """
    in_lie_only = len((syc_top & lie_top) - sent_top)
    in_sent_only = len((syc_top & sent_top) - lie_top)
    n_disc = in_lie_only + in_sent_only
    if n_disc == 0:
        return {
            'in_lie_not_sent': 0,
            'in_sent_not_lie': 0,
            'p_value': 1.0,
        }
    # Two-sided exact binomial under p=0.5 — equivalent to McNemar for paired feature labels.
    test = binomtest(in_lie_only, n_disc, p=0.5, alternative='two-sided')
    return {
        'in_lie_not_sent': int(in_lie_only),
        'in_sent_not_lie': int(in_sent_only),
        'p_value': float(test.pvalue),
    }


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--model', default=_DEFAULT_MODEL)
    parser.add_argument('--layer', type=int, default=_DEFAULT_LAYER)
    parser.add_argument('--n-prompts', type=int, default=_DEFAULT_N_PROMPTS)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--batch', type=int, default=_DEFAULT_BATCH)
    parser.add_argument('--top-k', type=int, default=_DEFAULT_TOP_K)
    parser.add_argument('--n-perm', type=int, default=_DEFAULT_N_PERM)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument(
        '--reference-from',
        default='sae_feature_overlap',
        help='Slug whose saved results provide the syc/lie top-feature reference.',
    )


def from_args(args: argparse.Namespace) -> SaeSentimentControlConfig:
    """Build the validated config from a parsed argparse namespace."""
    return SaeSentimentControlConfig(
        model=args.model,
        layer=args.layer,
        n_prompts=args.n_prompts,
        n_devices=args.n_devices,
        batch=args.batch,
        top_k=args.top_k,
        n_perm=args.n_perm,
        seed=args.seed,
        reference_from=args.reference_from,
    )
