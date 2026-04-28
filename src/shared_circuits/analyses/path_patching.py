"""
Edge-level path patching on the identified shared sycophancy/lying circuit.

For each ordered pair (source, receiver) where ``source`` is a shared head and
``receiver`` is either (a) another shared head at a later layer or (b) the
unembed, we measure the *direct* causal effect of the source -> receiver edge.
"""

import argparse
import json
import math
from collections import Counter
from collections.abc import Callable
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, model_session, model_slug, save_results
from shared_circuits.experiment.results import RESULTS_DIR
from shared_circuits.extraction import extract_residual_stream
from shared_circuits.models import get_agree_disagree_tokens
from shared_circuits.prompts import (
    build_instructed_lying_prompts,
    build_lying_prompts,
    build_repe_lying_prompts,
    build_scaffolded_lying_prompts,
    build_sycophancy_prompts,
)

_DEFAULT_N_PAIRS: Final = 30
_DEFAULT_BATCH: Final = 4
_DEFAULT_SEED: Final = 42
_DEFAULT_N_BOOT: Final = 1000
_DEFAULT_MAX_GEN: Final = 5

_TASKS: Final[tuple[str, ...]] = ('sycophancy', 'lying', 'instructed_lying', 'scaffolded_lying', 'repe_lying')
_SHARED_SOURCES: Final[tuple[str, ...]] = ('default', 'instructed', 'scaffolded', 'repe')


class PathPatchingConfig(BaseModel):
    """Inputs for the path-patching analysis (single model, single task)."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(...)
    n_devices: int = Field(default=1, gt=0)
    n_pairs: int = Field(default=_DEFAULT_N_PAIRS, gt=0)
    max_sources: int | None = Field(default=None)
    include_heads: bool = Field(default=True)
    batch: int = Field(default=_DEFAULT_BATCH, gt=0)
    n_non_shared: int | None = Field(default=None)
    task: str = Field(default='sycophancy')
    shared_source: str = Field(default='default')
    prefill_shift: bool = Field(default=False)
    seed: int = Field(default=_DEFAULT_SEED)
    n_boot: int = Field(default=_DEFAULT_N_BOOT, gt=0)
    max_gen: int = Field(default=_DEFAULT_MAX_GEN, ge=0)


def run(cfg: PathPatchingConfig) -> dict:
    """Run path-patching on ``cfg.model`` for the configured task."""
    if cfg.task not in _TASKS:
        raise ValueError(f'unknown task {cfg.task}')
    if cfg.shared_source not in _SHARED_SOURCES:
        raise ValueError(f'unknown shared_source {cfg.shared_source}')

    shared, n_layers, n_heads = _load_shared_heads(cfg.model, shared_source=cfg.shared_source)
    shared_sources = shared[: cfg.max_sources] if cfg.max_sources is not None else shared

    sources: list[tuple[tuple[int, int], str]] = [(s, 'shared') for s in shared_sources]
    non_shared_list: list[tuple[int, int]] = []
    if cfg.n_non_shared is not None:
        sample_n = None if cfg.n_non_shared < 0 else cfg.n_non_shared
        non_shared_list = _sample_non_shared(shared, n_layers, n_heads, sample_n, cfg.seed)
        sources.extend([(s, 'non_shared') for s in non_shared_list])

    all_pairs = load_triviaqa_pairs(max(cfg.n_pairs * 2, 200))[: cfg.n_pairs]
    wrong_prompts, correct_prompts = _task_prompts(cfg.task, all_pairs, cfg.model)

    with model_session(cfg.model, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, cfg, shared, shared_sources, sources, non_shared_list, wrong_prompts, correct_prompts)


def _analyse(
    ctx: ExperimentContext,
    cfg: PathPatchingConfig,
    shared: list[tuple[int, int]],
    shared_sources: list[tuple[int, int]],
    sources: list[tuple[tuple[int, int], str]],
    non_shared_list: list[tuple[int, int]],
    wrong_prompts: list[str],
    correct_prompts: list[str],
) -> dict:
    agree_t, disagree_t = _task_tokens(ctx.model, cfg.task)

    directions = _layer_directions(ctx.model, wrong_prompts, correct_prompts, cfg.batch)

    wrong_tokens_list, wrong_pos_list, correct_tokens_list, correct_pos_list = _prepare_tokens_and_positions(
        ctx.model, wrong_prompts, correct_prompts, agree_t, disagree_t, cfg.prefill_shift, cfg.max_gen
    )

    edges: dict[str, list[float]] = {}
    source_type_by_key: dict[str, str] = {}
    for src, src_type in sources:
        for recv in _edge_receivers(src, shared, cfg.include_heads and src_type == 'shared'):
            source_type_by_key[_edge_key(src, recv)] = src_type

    clean_lds: list[float] = []
    corrupt_lds: list[float] = []
    for c_toks, c_pos, k_toks, k_pos in zip(
        correct_tokens_list, correct_pos_list, wrong_tokens_list, wrong_pos_list, strict=False
    ):
        c_ld, k_ld = _process_pair(
            ctx.model,
            c_toks,
            c_pos,
            k_toks,
            k_pos,
            sources,
            shared,
            cfg.include_heads,
            directions,
            agree_t,
            disagree_t,
            edges,
        )
        clean_lds.append(c_ld)
        corrupt_lds.append(k_ld)

    edge_results = _aggregate(edges, source_type_by_key, cfg.n_boot, cfg.seed)

    results = {
        'model': ctx.model_name,
        'task': cfg.task,
        'config': {
            'task': cfg.task,
            'n_pairs': cfg.n_pairs,
            'n_sources': len(sources),
            'n_shared_sources': len(shared_sources),
            'n_non_shared_sources': len(non_shared_list),
            'n_shared_total': len(shared),
            'include_heads': cfg.include_heads,
            'n_boot': cfg.n_boot,
            'seed': cfg.seed,
        },
        'n_layers': ctx.info.n_layers,
        'n_heads': ctx.info.n_heads,
        'shared_heads_ranked': [list(h) for h in shared],
        'sources': [[list(s), t] for s, t in sources],
        'baseline_clean_logit_diff': float(np.mean(clean_lds)),
        'baseline_corrupt_logit_diff': float(np.mean(corrupt_lds)),
        'edges': edge_results,
    }
    save_results(results, f'path_patching_{cfg.task}', ctx.model_name)
    return results


@torch.no_grad()
def _find_answer_tokens(
    model: 'HookedTransformer',
    prompts: list[str],
    agree_t: list[int],
    disagree_t: list[int],
    max_gen: int,
) -> tuple[list[torch.Tensor], list[int], list[int]]:
    """Greedy-generate up to ``max_gen`` tokens; record where argmax is an agree/disagree token."""
    pad = getattr(model.tokenizer, 'pad_token_id', None) or 0
    ext_tokens: list[torch.Tensor] = []
    positions: list[int] = []
    warmups: list[int] = []
    for p in prompts:
        toks = model.to_tokens([p], prepend_bos=True)
        found_pos: int | None = None
        warmup = 0
        for i in range(max_gen + 1):
            last = int((toks != pad).sum().item()) - 1
            logits = model(toks)
            argmax = int(logits[0, last].argmax().item())
            if argmax in agree_t or argmax in disagree_t:
                found_pos = last
                warmup = i
                break
            if i == max_gen:
                break
            new = torch.tensor([[argmax]], device=toks.device, dtype=toks.dtype)
            toks = torch.cat([toks, new], dim=1)
        if found_pos is None:
            found_pos = int((toks != pad).sum().item()) - 1
            warmup = max_gen
        ext_tokens.append(toks)
        positions.append(found_pos)
        warmups.append(warmup)
    return ext_tokens, positions, warmups


def _task_tokens(model: 'HookedTransformer', task: str) -> tuple[list[int], list[int]]:
    """Per-task positive vs negative token sets used for ``logit_diff``."""
    if task == 'sycophancy':
        agree, disagree = get_agree_disagree_tokens(model)
        return list(agree), list(disagree)
    true_words = ['True', ' True', 'true', ' true', 'TRUE', ' TRUE']
    false_words = ['False', ' False', 'false', ' false', 'FALSE', ' FALSE']
    true_t: list[int] = []
    false_t: list[int] = []
    for w in true_words:
        true_t.extend(model.to_tokens(w, prepend_bos=False)[0].tolist())
    for w in false_words:
        false_t.extend(model.to_tokens(w, prepend_bos=False)[0].tolist())
    return list(set(true_t)), list(set(false_t))


def _task_prompts(task: str, pairs: list[tuple[str, str, str]], model_name: str) -> tuple[list[str], list[str]]:
    """Return ``(corrupt_prompts, clean_prompts)`` for ``task``."""
    if task == 'sycophancy':
        return build_sycophancy_prompts(pairs, model_name)
    if task == 'lying':
        return build_lying_prompts(pairs, model_name)
    if task == 'instructed_lying':
        return build_instructed_lying_prompts(pairs, model_name)
    if task == 'scaffolded_lying':
        return build_scaffolded_lying_prompts(pairs, model_name)
    if task == 'repe_lying':
        return build_repe_lying_prompts(pairs, model_name)
    raise ValueError(f'unknown task {task}')


def _load_shared_heads(model_name: str, shared_source: str = 'default') -> tuple[list[tuple[int, int]], int, int]:
    """Load top-K(syc) intersect top-K(lie) shared heads, ranked by combined importance."""
    slug = model_slug(model_name)
    if shared_source == 'repe':
        candidates = (f'dla_repe_{slug}.json',)
    elif shared_source == 'scaffolded':
        candidates = (f'dla_scaffolded_{slug}.json',)
    elif shared_source == 'instructed':
        candidates = (f'dla_instructed_{slug}.json',)
    else:
        candidates = (f'gate14_head_grid_{slug}.json', f'breadth_{slug}.json')
    for candidate in candidates:
        path = RESULTS_DIR / candidate
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        if 'syc_grid' in d and 'lie_grid' in d:
            sg = np.array(d['syc_grid'])
            lg = np.array(d['lie_grid'])
        elif 'head_overlap' in d and 'syc_grid' in d['head_overlap']:
            sg = np.array(d['head_overlap']['syc_grid'])
            lg = np.array(d['head_overlap']['lie_grid'])
        else:
            continue
        n_layers, n_heads = sg.shape
        total = n_layers * n_heads
        k = math.ceil(math.sqrt(total))
        sf = sg.flatten()
        lf = lg.flatten()
        syc_top = set(np.argsort(sf)[::-1][:k].tolist())
        lie_top = set(np.argsort(lf)[::-1][:k].tolist())
        shared = list(syc_top & lie_top)
        combined = (sf + lf) / 2
        shared_ranked = sorted(shared, key=lambda i: -combined[i])
        return [(int(i // n_heads), int(i % n_heads)) for i in shared_ranked], n_layers, n_heads
    raise FileNotFoundError(f'No DLA grid found for {model_name} (source={shared_source})')


def _sample_non_shared(
    shared: list[tuple[int, int]], n_layers: int, n_heads: int, n_sample: int | None, seed: int
) -> list[tuple[int, int]]:
    """Return non-shared heads as sources; uniform sample if ``n_sample`` is set, else all."""
    shared_set = set(shared)
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads) if (l, h) not in shared_set]
    if n_sample is None or n_sample >= len(all_heads):
        return all_heads
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(all_heads), n_sample, replace=False)
    return [all_heads[i] for i in sorted(idx.tolist())]


def _layer_directions(
    model: 'HookedTransformer',
    corrupt_prompts: list[str],
    clean_prompts: list[str],
    batch: int,
) -> dict[int, np.ndarray]:
    """Per-layer task direction (mean corrupt-resid - mean clean-resid), unit-norm."""
    directions: dict[int, np.ndarray] = {}
    for layer in range(model.cfg.n_layers):
        acts_w = extract_residual_stream(model, corrupt_prompts, layer, batch_size=batch)
        acts_c = extract_residual_stream(model, clean_prompts, layer, batch_size=batch)
        d = acts_w.mean(0) - acts_c.mean(0)
        directions[layer] = d / (np.linalg.norm(d) + 1e-10)
    return directions


@torch.no_grad()
def _cache_all(
    model: 'HookedTransformer',
    tokens: torch.Tensor,
    measurement_pos: int,
    agree_t: list[int],
    disagree_t: list[int],
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[int, torch.Tensor], float, int]:
    """Cache per-head z and per-layer MLP at ``measurement_pos`` while measuring logit_diff."""
    z_cache: dict[tuple[int, int], torch.Tensor] = {}
    mlp_cache: dict[int, torch.Tensor] = {}
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    hooks: list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]] = []
    for layer in range(n_layers):

        def make_z_cap(li: int):
            def cap(z: torch.Tensor, hook: object) -> torch.Tensor:
                for h in range(n_heads):
                    z_cache[(li, h)] = z[0, measurement_pos, h].clone()
                return z

            return cap

        def make_mlp_cap(li: int):
            def cap(t: torch.Tensor, hook: object) -> torch.Tensor:
                mlp_cache[li] = t[0, measurement_pos].clone()
                return t

            return cap

        hooks.append((f'blocks.{layer}.attn.hook_z', make_z_cap(layer)))
        hooks.append((f'blocks.{layer}.hook_mlp_out', make_mlp_cap(layer)))

    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    nl = logits[0, measurement_pos].float()
    logit_diff = float(nl[agree_t].max() - nl[disagree_t].max())
    return z_cache, mlp_cache, logit_diff, measurement_pos


def _patch_hooks(
    source: tuple[int, int],
    receiver: tuple[int, int] | str,
    clean_source_z: torch.Tensor,
    corrupt_z: dict[tuple[int, int], torch.Tensor],
    corrupt_mlp: dict[int, torch.Tensor],
    corrupt_pos: int,
    n_layers: int,
    n_heads: int,
) -> list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]]:
    """Build hook list implementing path patching for ``source -> receiver``."""
    src_layer, src_head = source
    if isinstance(receiver, tuple):
        recv_layer, recv_head = receiver
    else:
        recv_layer = n_layers
        recv_head = None
    hooks: list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]] = []

    def src_patch(z: torch.Tensor, hook: object) -> torch.Tensor:
        z[0, corrupt_pos, src_head] = clean_source_z.to(z.device, dtype=z.dtype)
        return z

    hooks.append((f'blocks.{src_layer}.attn.hook_z', src_patch))

    def src_mlp_freeze(t: torch.Tensor, hook: object) -> torch.Tensor:
        t[0, corrupt_pos] = corrupt_mlp[src_layer].to(t.device, dtype=t.dtype)
        return t

    hooks.append((f'blocks.{src_layer}.hook_mlp_out', src_mlp_freeze))

    for layer in range(src_layer + 1, recv_layer):

        def make_z_freeze(li: int):
            def freeze(z: torch.Tensor, hook: object) -> torch.Tensor:
                for h in range(n_heads):
                    z[0, corrupt_pos, h] = corrupt_z[(li, h)].to(z.device, dtype=z.dtype)
                return z

            return freeze

        def make_mlp_freeze(li: int):
            def freeze(t: torch.Tensor, hook: object) -> torch.Tensor:
                t[0, corrupt_pos] = corrupt_mlp[li].to(t.device, dtype=t.dtype)
                return t

            return freeze

        hooks.append((f'blocks.{layer}.attn.hook_z', make_z_freeze(layer)))
        hooks.append((f'blocks.{layer}.hook_mlp_out', make_mlp_freeze(layer)))

    if recv_head is not None:
        rh = recv_head

        def recv_layer_freeze(z: torch.Tensor, hook: object) -> torch.Tensor:
            for h in range(n_heads):
                if h != rh:
                    z[0, corrupt_pos, h] = corrupt_z[(recv_layer, h)].to(z.device, dtype=z.dtype)
            return z

        hooks.append((f'blocks.{recv_layer}.attn.hook_z', recv_layer_freeze))

    return hooks


@torch.no_grad()
def _measure_edge(
    model: 'HookedTransformer',
    corrupt_tokens: torch.Tensor,
    source: tuple[int, int],
    receiver: tuple[int, int] | str,
    clean_source_z: torch.Tensor,
    corrupt_z: dict[tuple[int, int], torch.Tensor],
    corrupt_mlp: dict[int, torch.Tensor],
    corrupt_pos: int,
    agree_t: list[int],
    disagree_t: list[int],
    layer_direction: np.ndarray | None,
) -> float:
    """Run path-patched forward pass; return scalar edge strength."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    tokens = corrupt_tokens

    hooks = _patch_hooks(source, receiver, clean_source_z, corrupt_z, corrupt_mlp, corrupt_pos, n_layers, n_heads)

    if not isinstance(receiver, tuple):
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        nl = logits[0, corrupt_pos].float()
        return float(nl[agree_t].max() - nl[disagree_t].max())

    recv_layer, recv_head = receiver
    recv_cap: dict[str, torch.Tensor] = {}

    def capture(z: torch.Tensor, hook: object) -> torch.Tensor:
        recv_cap['z'] = z[0, corrupt_pos, recv_head].clone()
        return z

    hooks.append((f'blocks.{recv_layer}.attn.hook_z', capture))
    model.run_with_hooks(tokens, fwd_hooks=hooks)

    w_o_head = model.blocks[recv_layer].attn.W_O[recv_head].float().to(recv_cap['z'].device)
    contribution = recv_cap['z'].float() @ w_o_head
    direction = torch.tensor(layer_direction, dtype=torch.float32, device=contribution.device)
    return float(contribution @ direction)


def _bootstrap_ci(values: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    """Return ``(mean, ci_lo, ci_hi)`` via percentile bootstrap."""
    arr = np.array(values)
    rng = np.random.RandomState(seed)
    n = len(arr)
    boots = np.array([arr[rng.choice(n, n, replace=True)].mean() for _ in range(n_boot)])
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _edge_receivers(
    source: tuple[int, int], shared: list[tuple[int, int]], include_heads: bool
) -> list[tuple[int, int] | str]:
    recs: list[tuple[int, int] | str] = ['unembed']
    if include_heads:
        for r in shared:
            if r[0] > source[0]:
                recs.append(r)
    return recs


def _edge_key(src: tuple[int, int], recv: tuple[int, int] | str) -> str:
    recv_str = f'{recv[0]}.{recv[1]}' if isinstance(recv, tuple) else 'unembed'
    return f'{src[0]}.{src[1]}->{recv_str}'


def _process_pair(
    model: 'HookedTransformer',
    clean_tokens: torch.Tensor,
    clean_pos: int,
    corrupt_tokens: torch.Tensor,
    corrupt_pos: int,
    sources: list[tuple[tuple[int, int], str]],
    shared: list[tuple[int, int]],
    include_heads: bool,
    directions: dict[int, np.ndarray],
    agree_t: list[int],
    disagree_t: list[int],
    edges: dict[str, list[float]],
) -> tuple[float, float]:
    """Process one (clean, corrupt) prompt pair, appending to ``edges`` dict."""
    clean_z, _, clean_ld, _ = _cache_all(model, clean_tokens, clean_pos, agree_t, disagree_t)
    corrupt_z, corrupt_mlp, corrupt_ld, _ = _cache_all(model, corrupt_tokens, corrupt_pos, agree_t, disagree_t)
    for src, src_type in sources:
        edge_include_heads = include_heads and src_type == 'shared'
        for recv in _edge_receivers(src, shared, edge_include_heads):
            layer_dir = directions[recv[0]] if isinstance(recv, tuple) else None
            val = _measure_edge(
                model,
                corrupt_tokens,
                src,
                recv,
                clean_z[src],
                corrupt_z,
                corrupt_mlp,
                corrupt_pos,
                agree_t,
                disagree_t,
                layer_dir,
            )
            edges.setdefault(_edge_key(src, recv), []).append(val)
    return clean_ld, corrupt_ld


def _aggregate(
    edges: dict[str, list[float]], source_type_by_key: dict[str, str], n_boot: int, seed: int
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for key, vals in edges.items():
        mean, lo, hi = _bootstrap_ci(vals, n_boot, seed)
        results[key] = {
            'mean': mean,
            'ci_95': [lo, hi],
            'significant': lo > 0 or hi < 0,
            'n_pairs': len(vals),
            'source_type': source_type_by_key.get(key, 'shared'),
        }
    return results


def _prepare_tokens_and_positions(
    model: 'HookedTransformer',
    wrong_prompts: list[str],
    correct_prompts: list[str],
    agree_t: list[int],
    disagree_t: list[int],
    prefill_shift: bool,
    max_gen: int,
) -> tuple[list[torch.Tensor], list[int], list[torch.Tensor], list[int]]:
    """
    Tokenize prompts and find each prompt's measurement position.

    If ``prefill_shift`` is True, greedy-generate until an agree/disagree token is
    argmax; otherwise use the last non-pad position.
    """
    pad = getattr(model.tokenizer, 'pad_token_id', None) or 0
    if prefill_shift:
        wrong_tokens_list, wrong_pos_list, _w = _find_answer_tokens(model, wrong_prompts, agree_t, disagree_t, max_gen)
        correct_tokens_list, correct_pos_list, _c = _find_answer_tokens(
            model, correct_prompts, agree_t, disagree_t, max_gen
        )
        # Counter is referenced for parity with the legacy diagnostic path.
        Counter(_w).most_common()
        Counter(_c).most_common()
    else:
        wrong_tokens_list = [model.to_tokens([p], prepend_bos=True) for p in wrong_prompts]
        correct_tokens_list = [model.to_tokens([p], prepend_bos=True) for p in correct_prompts]
        wrong_pos_list = [int((t != pad).sum().item()) - 1 for t in wrong_tokens_list]
        correct_pos_list = [int((t != pad).sum().item()) - 1 for t in correct_tokens_list]
    return wrong_tokens_list, wrong_pos_list, correct_tokens_list, correct_pos_list


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PAIRS)
    parser.add_argument(
        '--max-sources', type=int, default=None, help='Cap number of source heads (by importance rank).'
    )
    parser.add_argument(
        '--no-head-edges',
        action='store_true',
        help='Only test source->unembed edges; skip source->downstream-head edges (much cheaper).',
    )
    parser.add_argument('--batch', type=int, default=_DEFAULT_BATCH, help='Batch size for direction extraction only.')
    parser.add_argument(
        '--non-shared-sources',
        type=int,
        default=None,
        help='If set, also run non-shared heads as sources for ->unembed edges only '
        '(negative control). -1 = all non-shared heads; positive int = uniform sample.',
    )
    parser.add_argument(
        '--task',
        choices=list(_TASKS),
        default='sycophancy',
        help='Which task to evaluate the shared circuit on.',
    )
    parser.add_argument(
        '--shared-source',
        choices=list(_SHARED_SOURCES),
        default='default',
        help='Which DLA grid defines "shared" heads: default=gate14/breadth, '
        'instructed=dla_instructed_*, scaffolded=dla_scaffolded_*, repe=dla_repe_*.',
    )
    parser.add_argument(
        '--prefill-shift',
        action='store_true',
        help='Run one forward pass per prompt, append the argmax token to the prompt, '
        'then do path patching at the shifted position.',
    )
    parser.add_argument('--seed', type=int, default=_DEFAULT_SEED)
    parser.add_argument('--n-boot', type=int, default=_DEFAULT_N_BOOT)
    parser.add_argument('--max-gen', type=int, default=_DEFAULT_MAX_GEN)


def from_args(args: argparse.Namespace) -> PathPatchingConfig:
    """Build the validated config from a parsed argparse namespace."""
    return PathPatchingConfig(
        model=args.model,
        n_devices=args.n_devices,
        n_pairs=args.n_pairs,
        max_sources=args.max_sources,
        include_heads=not args.no_head_edges,
        batch=args.batch,
        n_non_shared=args.non_shared_sources,
        task=args.task,
        shared_source=args.shared_source,
        prefill_shift=args.prefill_shift,
        seed=args.seed,
        n_boot=args.n_boot,
        max_gen=args.max_gen,
    )
