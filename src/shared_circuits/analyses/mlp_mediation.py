"""
Cross-layer MLP mediation: test whether upstream MLPs feed the shared-head circuit.

Picks ``n_upstream`` layers below the first shared-head layer and ``n_in_region``
layers within the shared-head span. For each candidate MLP, ablate its output
and measure both:

* ``Delta_proj``: change in the shared heads' aggregate contribution projected
  onto the sycophancy direction at the chosen direction layer.
* ``Delta_logit_diff``: change in the model's last-token agree-vs-disagree
  logit gap.

Paired-bootstrap 95% CIs are computed per cell; the aggregate Spearman
correlation between ``Delta_proj`` and ``Delta_logit_diff`` across the full
candidate set tests whether projection-loss explains behavior.

Generalized from the Qwen-72B-specific legacy script: any model name works,
``n_upstream`` and ``n_in_region`` default to 8/8, and shared heads come from
``circuit_overlap`` results so the analysis can target the shared-head layer
range automatically.
"""

import argparse
from collections.abc import Callable
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import spearmanr

from shared_circuits.config import RANDOM_SEED
from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, load_results, model_session, save_results
from shared_circuits.extraction import extract_residual_stream
from shared_circuits.prompts import build_sycophancy_prompts

_DEFAULT_N_UPSTREAM: Final = 8
_DEFAULT_N_IN_REGION: Final = 8
_DEFAULT_BATCH: Final = 2
_DEFAULT_N_BOOT: Final = 2000
_DEFAULT_DIR_PROMPTS: Final = 50
_DEFAULT_TEST_PROMPTS: Final = 100
_DEFAULT_N_PAIRS: Final = 300
# minimum sample size for spearman correlation to be meaningful.
_MIN_SPEARMAN_N: Final = 3


class MlpMediationConfig(BaseModel):
    """Inputs for the cross-layer MLP-mediation analysis (single model)."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(...)
    n_devices: int = Field(default=1, gt=0)
    batch: int = Field(default=_DEFAULT_BATCH, gt=0)
    n_upstream: int = Field(default=_DEFAULT_N_UPSTREAM, gt=0)
    n_in_region: int = Field(default=_DEFAULT_N_IN_REGION, ge=0)
    n_pairs: int = Field(default=_DEFAULT_N_PAIRS, gt=0)
    dir_prompts: int = Field(default=_DEFAULT_DIR_PROMPTS, gt=0)
    test_prompts: int = Field(default=_DEFAULT_TEST_PROMPTS, gt=0)
    direction_layer: int | None = Field(default=None)
    shared_heads_from: str = Field(default='circuit_overlap')
    shared_heads_k: int = Field(default=15, gt=0)
    n_boot: int = Field(default=_DEFAULT_N_BOOT, gt=0)
    seed: int = Field(default=RANDOM_SEED)


def run(cfg: MlpMediationConfig) -> dict:
    """Run the MLP-mediation analysis on ``cfg.model``."""
    shared_by_layer = _load_shared_by_layer(cfg.shared_heads_from, cfg.model, cfg.shared_heads_k)
    if not shared_by_layer:
        raise ValueError(f'no shared heads found for {cfg.model} in {cfg.shared_heads_from}')
    pairs = load_triviaqa_pairs(cfg.n_pairs)
    with model_session(cfg.model, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, pairs, shared_by_layer, cfg)


def _load_shared_by_layer(source: str, model_name: str, top_k: int) -> dict[int, list[int]]:
    """Return ``{layer: [head, ...]}`` from a sibling ``circuit_overlap``-shaped result."""
    data = load_results(source, model_name)
    bucket = next(o for o in data['overlap_by_K'] if o['K'] == top_k)
    by_layer: dict[int, list[int]] = {}
    for layer, head in bucket['shared_heads']:
        by_layer.setdefault(int(layer), []).append(int(head))
    return by_layer


def _select_mlp_candidates(
    shared_by_layer: dict[int, list[int]],
    n_layers: int,
    n_upstream: int,
    n_in_region: int,
) -> tuple[list[int], int, int]:
    """Pick upstream + in-region MLP layers around the shared-head span."""
    layers_with_heads = sorted(shared_by_layer)
    first = layers_with_heads[0]
    last = layers_with_heads[-1]

    upstream: list[int]
    if first <= 0 or n_upstream == 0:
        upstream = []
    else:
        n_up = min(n_upstream, first)
        # Even spread from layer 0 up to (first - 1).
        idxs = np.linspace(0, first - 1, n_up).round().astype(int)
        upstream = sorted(set(int(i) for i in idxs.tolist()))

    in_region: list[int]
    if n_in_region == 0 or first >= last:
        in_region = []
    else:
        n_in = min(n_in_region, last - first + 1)
        idxs = np.linspace(first, last, n_in).round().astype(int)
        in_region = sorted(set(int(i) for i in idxs.tolist() if 0 <= int(i) < n_layers))

    candidates = sorted(set(upstream) | set(in_region))
    return candidates, first, last


def _analyse(
    ctx: ExperimentContext,
    pairs: list[tuple[str, str, str]],
    shared_by_layer: dict[int, list[int]],
    cfg: MlpMediationConfig,
) -> dict:
    candidates, first_shared, last_shared = _select_mlp_candidates(
        shared_by_layer, ctx.info.n_layers, cfg.n_upstream, cfg.n_in_region
    )
    direction_layer = cfg.direction_layer if cfg.direction_layer is not None else first_shared

    dir_pairs = pairs[: cfg.dir_prompts]
    test_pairs = pairs[cfg.dir_prompts : cfg.dir_prompts + cfg.test_prompts]
    if not test_pairs:
        raise ValueError('empty test set: increase --n-pairs or reduce --dir-prompts/--test-prompts')

    syc_dir = _extract_syc_direction(ctx, dir_pairs, direction_layer, cfg.batch)
    test_wrong, _ = build_sycophancy_prompts(test_pairs, ctx.model_name)

    base = _measure(
        ctx.model,
        test_wrong,
        shared_by_layer,
        syc_dir,
        list(ctx.agree_tokens),
        list(ctx.disagree_tokens),
        cfg.batch,
    )

    candidate_results: dict[str, dict] = {}
    for m in candidates:
        abl = _measure(
            ctx.model,
            test_wrong,
            shared_by_layer,
            syc_dir,
            list(ctx.agree_tokens),
            list(ctx.disagree_tokens),
            cfg.batch,
            ablate_mlp=m,
        )
        d_rate = abl['rate'] - base['rate']
        d_ld, ld_lo, ld_hi = _paired_ci(
            base['per_prompt_logit_diff'], abl['per_prompt_logit_diff'], cfg.n_boot, cfg.seed
        )
        d_proj, pj_lo, pj_hi = _paired_ci(
            base['per_prompt_projection'], abl['per_prompt_projection'], cfg.n_boot, cfg.seed
        )
        position = 'upstream' if m < first_shared else 'in_region'
        candidate_results[str(m)] = {
            'position': position,
            'rate': abl['rate'],
            'logit_diff': abl['logit_diff_mean'],
            'projection': abl['projection_mean'],
            'rate_delta': float(d_rate),
            'logit_diff_delta': d_ld,
            'logit_diff_ci': [ld_lo, ld_hi],
            'logit_diff_significant': bool(ld_lo > 0 or ld_hi < 0),
            'projection_delta': d_proj,
            'projection_ci': [pj_lo, pj_hi],
            'projection_significant': bool(pj_lo > 0 or pj_hi < 0),
        }

    pipeline = _pipeline_correlations(candidate_results, candidates, first_shared)
    n_shared = sum(len(hs) for hs in shared_by_layer.values())

    result = {
        'model': ctx.model_name,
        'config': {
            'n_pairs': cfg.n_pairs,
            'dir_prompts': len(dir_pairs),
            'test_prompts': len(test_wrong),
            'batch': cfg.batch,
            'n_boot': cfg.n_boot,
            'seed': cfg.seed,
            'direction_layer': direction_layer,
            'n_upstream': cfg.n_upstream,
            'n_in_region': cfg.n_in_region,
            'shared_heads_from': cfg.shared_heads_from,
            'shared_heads_k': cfg.shared_heads_k,
        },
        'shared_head_layers': {str(l): hs for l, hs in shared_by_layer.items()},
        'n_shared_heads': n_shared,
        'first_shared_layer': first_shared,
        'last_shared_layer': last_shared,
        'mlp_candidates': candidates,
        'baseline': {k: v for k, v in base.items() if not k.startswith('per_prompt')},
        'candidates': candidate_results,
        'pipeline_test': pipeline,
    }
    save_results(result, 'mlp_mediation', ctx.model_name)
    return result


def _extract_syc_direction(
    ctx: ExperimentContext,
    dir_pairs: list[tuple[str, str, str]],
    layer: int,
    batch: int,
) -> np.ndarray:
    """Mean(corrupt_resid) - mean(clean_resid) at ``layer``, unit-normalized."""
    wrong, correct = build_sycophancy_prompts(dir_pairs, ctx.model_name)
    acts_w = extract_residual_stream(ctx.model, wrong, layer, batch_size=batch)
    acts_c = extract_residual_stream(ctx.model, correct, layer, batch_size=batch)
    direction = acts_w.mean(0) - acts_c.mean(0)
    return direction / (np.linalg.norm(direction) + 1e-10)


@torch.no_grad()
def _measure(
    model: 'HookedTransformer',
    prompts: list[str],
    shared_by_layer: dict[int, list[int]],
    syc_dir: np.ndarray,
    agree_tokens: list[int],
    disagree_tokens: list[int],
    batch: int,
    ablate_mlp: int | None = None,
) -> dict:
    """Per-prompt rate, logit_diff, and shared-head projection onto ``syc_dir``."""
    pad_id = getattr(model.tokenizer, 'pad_token_id', None) or 0
    d_model = model.cfg.d_model
    syc_t = torch.tensor(syc_dir, dtype=torch.float32)
    agree = 0
    per_prompt_logit_diff: list[float] = []
    per_prompt_proj: list[float] = []

    for i in range(0, len(prompts), batch):
        batch_p = prompts[i : i + batch]
        tokens = model.to_tokens(batch_p, prepend_bos=True)
        seq_lens = [int(x) for x in ((tokens != pad_id).sum(dim=1) - 1).tolist()]
        stores: dict[int, torch.Tensor] = {}
        hooks: list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]] = []
        for layer in shared_by_layer:

            def make_capture(li: int) -> Callable[[torch.Tensor, object], None]:
                def capture(t: torch.Tensor, hook: object) -> None:
                    stores[li] = t

                return capture

            hooks.append((f'blocks.{layer}.attn.hook_z', make_capture(layer)))

        if ablate_mlp is not None:

            def zero_mlp(t: torch.Tensor, hook: object) -> torch.Tensor:
                return torch.zeros_like(t)

            hooks.append((f'blocks.{ablate_mlp}.hook_mlp_out', zero_mlp))

        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        for b in range(len(batch_p)):
            pos = seq_lens[b]
            nl = logits[b, pos].float()
            agree_max = float(nl[agree_tokens].max())
            disagree_max = float(nl[disagree_tokens].max())
            if agree_max > disagree_max:
                agree += 1
            per_prompt_logit_diff.append(agree_max - disagree_max)

            sh_sum = torch.zeros(d_model, dtype=torch.float32)
            for layer, heads in shared_by_layer.items():
                z = stores[layer]
                w_o_layer = model.blocks[layer].attn.W_O.float().to(z.device)
                for h in heads:
                    z_bh = z[b, pos, h, :].float()
                    out_bh = (z_bh @ w_o_layer[h]).cpu()
                    sh_sum = sh_sum + out_bh
            per_prompt_proj.append(float(sh_sum @ syc_t))

    n = len(prompts) if prompts else 1
    return {
        'rate': agree / n if prompts else 0.0,
        'logit_diff_mean': float(np.mean(per_prompt_logit_diff)) if per_prompt_logit_diff else 0.0,
        'projection_mean': float(np.mean(per_prompt_proj)) if per_prompt_proj else 0.0,
        'per_prompt_logit_diff': per_prompt_logit_diff,
        'per_prompt_projection': per_prompt_proj,
    }


def _paired_ci(base_v: list[float], abl_v: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    """Paired bootstrap 95% CI for ``mean(abl) - mean(base)`` over prompt indices."""
    if not base_v:
        return 0.0, 0.0, 0.0
    b = np.array(base_v)
    a = np.array(abl_v)
    n = len(b)
    rng = np.random.RandomState(seed)
    boots = np.array([a[idx].mean() - b[idx].mean() for idx in (rng.choice(n, n, replace=True) for _ in range(n_boot))])
    return float(a.mean() - b.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < _MIN_SPEARMAN_N or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan'), float('nan')
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def _excludes_zero(ci: list[float]) -> bool:
    return ci[0] > 0 or ci[1] < 0


def _pipeline_correlations(candidate_results: dict[str, dict], ms: list[int], first_shared: int) -> dict:
    """Aggregate Spearman correlations across upstream / in-region / all candidates."""
    upstream_ms = [m for m in ms if m < first_shared]
    in_region_ms = [m for m in ms if m >= first_shared]
    d_ld = np.array([candidate_results[str(m)]['logit_diff_delta'] for m in ms])
    d_projs = np.array([candidate_results[str(m)]['projection_delta'] for m in ms])
    up_ld = np.array([candidate_results[str(m)]['logit_diff_delta'] for m in upstream_ms])
    up_projs = np.array([candidate_results[str(m)]['projection_delta'] for m in upstream_ms])
    signed_rho, signed_p = _safe_spearman(d_ld, d_projs)
    abs_rho, abs_p = _safe_spearman(np.abs(d_ld), np.abs(d_projs))
    up_signed_rho, up_signed_p = _safe_spearman(up_ld, up_projs)
    up_abs_rho, up_abs_p = _safe_spearman(np.abs(up_ld), np.abs(up_projs))

    up_proj_sig = sum(1 for m in upstream_ms if _excludes_zero(candidate_results[str(m)]['projection_ci']))
    up_ld_sig = sum(1 for m in upstream_ms if _excludes_zero(candidate_results[str(m)]['logit_diff_ci']))
    in_proj_sig = sum(1 for m in in_region_ms if _excludes_zero(candidate_results[str(m)]['projection_ci']))
    in_ld_sig = sum(1 for m in in_region_ms if _excludes_zero(candidate_results[str(m)]['logit_diff_ci']))

    return {
        'all_signed_rho': signed_rho,
        'all_signed_p': signed_p,
        'all_abs_rho': abs_rho,
        'all_abs_p': abs_p,
        'upstream_signed_rho': up_signed_rho,
        'upstream_signed_p': up_signed_p,
        'upstream_abs_rho': up_abs_rho,
        'upstream_abs_p': up_abs_p,
        'upstream_n': len(upstream_ms),
        'in_region_n': len(in_region_ms),
        'all_n': len(ms),
        'upstream_proj_ci_excludes_zero': int(up_proj_sig),
        'upstream_logit_diff_ci_excludes_zero': int(up_ld_sig),
        'in_region_proj_ci_excludes_zero': int(in_proj_sig),
        'in_region_logit_diff_ci_excludes_zero': int(in_ld_sig),
    }


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--batch', type=int, default=_DEFAULT_BATCH)
    parser.add_argument('--n-upstream', type=int, default=_DEFAULT_N_UPSTREAM)
    parser.add_argument('--n-in-region', type=int, default=_DEFAULT_N_IN_REGION)
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PAIRS)
    parser.add_argument('--dir-prompts', type=int, default=_DEFAULT_DIR_PROMPTS)
    parser.add_argument('--test-prompts', type=int, default=_DEFAULT_TEST_PROMPTS)
    parser.add_argument(
        '--direction-layer',
        type=int,
        default=None,
        help='Layer at which to extract the syc direction; default is the first shared-head layer.',
    )
    parser.add_argument(
        '--shared-heads-from',
        default='circuit_overlap',
        help='Experiment slug whose saved results provide the per-layer shared-heads list.',
    )
    parser.add_argument('--shared-heads-k', type=int, default=15)
    parser.add_argument('--n-boot', type=int, default=_DEFAULT_N_BOOT)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)


def from_args(args: argparse.Namespace) -> MlpMediationConfig:
    """Build the validated config from a parsed argparse namespace."""
    return MlpMediationConfig(
        model=args.model,
        n_devices=args.n_devices,
        batch=args.batch,
        n_upstream=args.n_upstream,
        n_in_region=args.n_in_region,
        n_pairs=args.n_pairs,
        dir_prompts=args.dir_prompts,
        test_prompts=args.test_prompts,
        direction_layer=args.direction_layer,
        shared_heads_from=args.shared_heads_from,
        shared_heads_k=args.shared_heads_k,
        n_boot=args.n_boot,
        seed=args.seed,
    )
