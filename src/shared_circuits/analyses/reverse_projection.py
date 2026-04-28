"""
Reverse projection ablation: cross-task coupling test.

If sycophancy and lying share substrate, ablating ``d_syc`` (the sycophancy
direction) should also damage lying-task accuracy, and vice versa.

Protocol (ablate ``d_A``, measure task ``B``):
1. Compute per-layer direction ``d_A`` from task ``A``'s ``(corrupt - clean)``
   residual difference.
2. On task ``B``'s prompts, at every layer's ``hook_resid_pre``, subtract the
   projection onto ``d_A`` (``resid := resid - (resid . d_A) * d_A`` since
   ``d_A`` is unit-norm per layer).
3. Measure task ``B``'s logit-diff with and without ablation; report the gap
   shrinkage (``frac_preserved`` close to 0 means the ablation disrupted task B).

Four cells are reported per run: ablate-syc/measure-syc, ablate-syc/measure-lie,
ablate-lie/measure-lie, ablate-lie/measure-syc.  The cross-task cells are
the test of interest.

A ``--from-direction-file`` mode is preserved from the legacy script for
reproducing the DPO-fine-tuned variant: directions are loaded from a previous
``direction_analysis``-style results file rather than recomputed.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, model_session, save_results
from shared_circuits.extraction import extract_residual_stream
from shared_circuits.prompts import (
    build_instructed_lying_prompts,
    build_repe_lying_prompts,
    build_scaffolded_lying_prompts,
    build_sycophancy_prompts,
)

_DEFAULT_N_PAIRS: Final = 50
_DEFAULT_BATCH: Final = 4
_GAP_EPS: Final = 1e-6

_LYING_TASKS: Final[tuple[str, ...]] = ('instructed_lying', 'scaffolded_lying', 'repe_lying')
# verdict thresholds: any cross-task cell with frac_preserved < this value
# counts as significant cross-task coupling.
_COUPLING_THRESHOLD: Final = 0.5


class ReverseProjectionConfig(BaseModel):
    """Inputs for the reverse-projection-ablation analysis (single model)."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(...)
    n_devices: int = Field(default=1, gt=0)
    n_pairs: int = Field(default=_DEFAULT_N_PAIRS, gt=0)
    batch: int = Field(default=_DEFAULT_BATCH, gt=0)
    lying_task: str = Field(default='instructed_lying')
    weight_repo: str | None = Field(default=None)
    tag: str = Field(default='')
    from_direction_file: str | None = Field(default=None)


def run(cfg: ReverseProjectionConfig) -> dict:
    """Run reverse-projection ablation on ``cfg.model`` for sycophancy/<lying_task>."""
    if cfg.lying_task not in _LYING_TASKS:
        raise ValueError(f'unknown lying_task {cfg.lying_task}')

    pairs = load_triviaqa_pairs(max(cfg.n_pairs * 2, 200))[: cfg.n_pairs]
    with model_session(cfg.model, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, pairs, cfg)


def _analyse(ctx: ExperimentContext, pairs: list[tuple[str, str, str]], cfg: ReverseProjectionConfig) -> dict:
    syc_corrupt, syc_clean = _task_prompts('sycophancy', pairs, ctx.model_name)
    lie_corrupt, lie_clean = _task_prompts(cfg.lying_task, pairs, ctx.model_name)
    agree_syc, disagree_syc = ctx.agree_tokens, ctx.disagree_tokens
    agree_lie, disagree_lie = _lying_task_tokens(ctx.model)

    if cfg.from_direction_file is not None:
        d_syc, d_lie = _load_directions_from_file(cfg.from_direction_file, ctx.info.n_layers, cfg.lying_task)
    else:
        d_syc = _compute_direction(ctx.model, pairs, ctx.model_name, 'sycophancy', cfg.batch)
        d_lie = _compute_direction(ctx.model, pairs, ctx.model_name, cfg.lying_task, cfg.batch)

    cosines = [float(np.dot(d_syc[layer], d_lie[layer])) for layer in range(ctx.info.n_layers)]

    cells = {
        'ablate_syc_measure_syc': _measure_cell_pair(
            ctx.model, syc_clean, syc_corrupt, d_syc, list(agree_syc), list(disagree_syc)
        ),
        'ablate_syc_measure_lie': _measure_cell_pair(
            ctx.model, lie_clean, lie_corrupt, d_syc, list(agree_lie), list(disagree_lie)
        ),
        'ablate_lie_measure_lie': _measure_cell_pair(
            ctx.model, lie_clean, lie_corrupt, d_lie, list(agree_lie), list(disagree_lie)
        ),
        'ablate_lie_measure_syc': _measure_cell_pair(
            ctx.model, syc_clean, syc_corrupt, d_lie, list(agree_syc), list(disagree_syc)
        ),
    }

    verdict = _verdict(cells)

    out = {
        'model': ctx.model_name,
        'verdict': verdict,
        'lying_task': cfg.lying_task,
        'weight_repo': cfg.weight_repo,
        'n_pairs': cfg.n_pairs,
        'n_layers': ctx.info.n_layers,
        'mean_cos_syc_lie': float(np.mean(cosines)),
        'per_layer_cos': cosines,
        'cells': cells,
    }
    tag_suffix = {'scaffolded_lying': '_scaffolded', 'repe_lying': '_repe'}.get(cfg.lying_task, '')
    experiment_name = 'reverse_projection' + tag_suffix
    if cfg.tag:
        experiment_name = f'{experiment_name}_{cfg.tag}'
    save_results(out, experiment_name, ctx.model_name)
    return out


def _task_prompts(task: str, pairs: list[tuple[str, str, str]], model_name: str) -> tuple[list[str], list[str]]:
    """Return ``(corrupt_prompts, clean_prompts)`` for ``task``."""
    if task == 'sycophancy':
        return build_sycophancy_prompts(pairs, model_name)
    if task == 'instructed_lying':
        return build_instructed_lying_prompts(pairs, model_name)
    if task == 'scaffolded_lying':
        return build_scaffolded_lying_prompts(pairs, model_name)
    if task == 'repe_lying':
        return build_repe_lying_prompts(pairs, model_name)
    raise ValueError(f'unknown task {task}')


def _lying_task_tokens(model: 'HookedTransformer') -> tuple[list[int], list[int]]:
    """True/False token sets used as positive/negative for lying-task logit_diff."""
    true_words = ['True', ' True', 'true', ' true', 'TRUE', ' TRUE']
    false_words = ['False', ' False', 'false', ' false', 'FALSE', ' FALSE']
    true_t: list[int] = []
    false_t: list[int] = []
    for w in true_words:
        true_t.extend(model.to_tokens(w, prepend_bos=False)[0].tolist())
    for w in false_words:
        false_t.extend(model.to_tokens(w, prepend_bos=False)[0].tolist())
    return list(set(true_t)), list(set(false_t))


def _compute_direction(
    model: 'HookedTransformer',
    pairs: list[tuple[str, str, str]],
    model_name: str,
    task: str,
    batch: int,
) -> dict[int, np.ndarray]:
    """Per-layer direction: ``mean(corrupt_resid) - mean(clean_resid)``, unit-normalized."""
    corrupt, clean = _task_prompts(task, pairs, model_name)
    directions: dict[int, np.ndarray] = {}
    for layer in range(model.cfg.n_layers):
        ac = extract_residual_stream(model, corrupt, layer, batch_size=batch)
        cc = extract_residual_stream(model, clean, layer, batch_size=batch)
        d = ac.mean(0) - cc.mean(0)
        directions[layer] = d / (np.linalg.norm(d) + 1e-10)
    return directions


def _load_directions_from_file(
    path_str: str, n_layers: int, lying_task: str
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Read pre-computed per-layer ``d_syc`` and ``d_lie`` from a JSON file."""
    path = Path(path_str)
    with open(path) as f:
        data = json.load(f)
    syc = data.get('d_syc') or data.get('syc_directions') or {}
    lie = data.get(f'd_{lying_task}') or data.get('d_lie') or data.get('lie_directions') or {}
    if not syc or not lie:
        raise ValueError(f'{path}: missing d_syc/d_lie payload (saw keys {list(data.keys())})')
    syc_arr = {int(k): np.asarray(v, dtype=np.float32) for k, v in syc.items()}
    lie_arr = {int(k): np.asarray(v, dtype=np.float32) for k, v in lie.items()}
    if len(syc_arr) != n_layers or len(lie_arr) != n_layers:
        raise ValueError(f'{path}: expected {n_layers} layers, got d_syc={len(syc_arr)} d_lie={len(lie_arr)}')
    return syc_arr, lie_arr


@torch.no_grad()
def _logit_diff(model: 'HookedTransformer', prompt: str, agree_t: list[int], disagree_t: list[int]) -> float:
    pad = getattr(model.tokenizer, 'pad_token_id', None) or 0
    toks = model.to_tokens([prompt], prepend_bos=True)
    seq = int((toks != pad).sum().item()) - 1
    logits = model(toks)
    nl = logits[0, seq].float()
    return float(nl[agree_t].max() - nl[disagree_t].max())


@torch.no_grad()
def _logit_diff_ablated(
    model: 'HookedTransformer',
    prompt: str,
    directions: dict[int, np.ndarray],
    agree_t: list[int],
    disagree_t: list[int],
) -> float:
    """Run forward pass with ``directions[layer]`` projected out at every ``hook_resid_pre``."""
    pad = getattr(model.tokenizer, 'pad_token_id', None) or 0
    toks = model.to_tokens([prompt], prepend_bos=True)
    seq = int((toks != pad).sum().item()) - 1
    hooks: list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor]]] = []
    for layer in range(model.cfg.n_layers):
        d_np = directions[layer]

        def make_hook(dn: np.ndarray) -> Callable[[torch.Tensor, object], torch.Tensor]:
            cache: dict[str, torch.Tensor] = {}

            def h(resid: torch.Tensor, hook: object) -> torch.Tensor:
                if 'd' not in cache:
                    cache['d'] = torch.tensor(dn, dtype=resid.dtype, device=resid.device)
                d = cache['d']
                proj = (resid @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
                return resid - proj

            return h

        hooks.append((f'blocks.{layer}.hook_resid_pre', make_hook(d_np)))
    logits = model.run_with_hooks(toks, fwd_hooks=hooks)
    nl = logits[0, seq].float()
    return float(nl[agree_t].max() - nl[disagree_t].max())


def _measure_cell_pair(
    model: 'HookedTransformer',
    clean_prompts: list[str],
    corrupt_prompts: list[str],
    ablation_dir: dict[int, np.ndarray],
    agree_t: list[int],
    disagree_t: list[int],
) -> dict:
    """Measure baseline + ablated logit_diff on (clean, corrupt) and report frac_preserved."""
    b_c = np.array([_logit_diff(model, p, agree_t, disagree_t) for p in clean_prompts])
    a_c = np.array([_logit_diff_ablated(model, p, ablation_dir, agree_t, disagree_t) for p in clean_prompts])
    b_w = np.array([_logit_diff(model, p, agree_t, disagree_t) for p in corrupt_prompts])
    a_w = np.array([_logit_diff_ablated(model, p, ablation_dir, agree_t, disagree_t) for p in corrupt_prompts])

    base_gap = float(b_c.mean() - b_w.mean())
    ab_gap = float(a_c.mean() - a_w.mean())
    frac = ab_gap / base_gap if abs(base_gap) > _GAP_EPS else float('nan')
    return {
        'baseline_gap': base_gap,
        'ablated_gap': ab_gap,
        'frac_preserved': frac,
        'baseline_clean_per_pair': b_c.tolist(),
        'baseline_corrupt_per_pair': b_w.tolist(),
        'ablated_clean_per_pair': a_c.tolist(),
        'ablated_corrupt_per_pair': a_w.tolist(),
    }


def _verdict(cells: dict[str, dict]) -> str:
    """COUPLED if either cross-task ablation shrinks the gap below the threshold."""
    syc_to_lie = cells['ablate_syc_measure_lie'].get('frac_preserved')
    lie_to_syc = cells['ablate_lie_measure_syc'].get('frac_preserved')
    cross_vals = [v for v in (syc_to_lie, lie_to_syc) if v is not None and not np.isnan(v)]
    if not cross_vals:
        return 'INCOMPLETE'
    if min(cross_vals) < _COUPLING_THRESHOLD and max(cross_vals) < _COUPLING_THRESHOLD:
        return 'COUPLED'
    if min(cross_vals) < _COUPLING_THRESHOLD:
        return 'PARTIALLY_COUPLED'
    return 'INDEPENDENT'


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PAIRS)
    parser.add_argument('--batch', type=int, default=_DEFAULT_BATCH)
    parser.add_argument(
        '--lying-task',
        choices=list(_LYING_TASKS),
        default='instructed_lying',
    )
    parser.add_argument(
        '--weight-repo',
        default=None,
        help='Optional alternative HF repo for weights (DPO-finetuned mirrors).',
    )
    parser.add_argument('--tag', default='', help='Suffix appended to the saved-result filename.')
    parser.add_argument(
        '--from-direction-file',
        default=None,
        help='Path to a JSON with pre-computed d_syc/d_lie per layer (skips recomputation).',
    )


def from_args(args: argparse.Namespace) -> ReverseProjectionConfig:
    """Build the validated config from a parsed argparse namespace."""
    return ReverseProjectionConfig(
        model=args.model,
        n_devices=args.n_devices,
        n_pairs=args.n_pairs,
        batch=args.batch,
        lying_task=args.lying_task,
        weight_repo=args.weight_repo,
        tag=args.tag,
        from_direction_file=args.from_direction_file,
    )
