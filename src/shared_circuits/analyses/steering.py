"""Behavioral steering: dose-response of the syc direction at a chosen layer."""

import argparse
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.data import load_triviaqa_pairs
from shared_circuits.experiment import ExperimentContext, model_session, save_results
from shared_circuits.extraction import extract_residual_stream, measure_agreement_rate
from shared_circuits.prompts import build_sycophancy_prompts

_DEFAULT_N_PAIRS: Final = 200
_DEFAULT_TEST_PROMPTS: Final = 100
_DEFAULT_DIR_PROMPTS: Final = 100
# Match legacy default sweep: zero, then negative + positive arms scaling outward.
_DEFAULT_ALPHAS: Final[tuple[int, ...]] = (0, -25, -50, -100, -200, 25, 50, 100, 200)
_DEFAULT_LAYER_FRAC: Final = 0.6
_DEFAULT_BATCH: Final = 4


class SteeringConfig(BaseModel):
    """Inputs for the steering dose-response analysis (single model)."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(...)
    n_devices: int = Field(default=1, gt=0)
    n_pairs: int = Field(default=_DEFAULT_N_PAIRS, gt=0)
    test_prompts: int = Field(default=_DEFAULT_TEST_PROMPTS, gt=0)
    dir_prompts: int = Field(default=_DEFAULT_DIR_PROMPTS, gt=0)
    alphas: tuple[int, ...] = Field(default=_DEFAULT_ALPHAS)
    layer_frac: float = Field(default=_DEFAULT_LAYER_FRAC, gt=0, lt=1)
    layer: int | None = Field(default=None)
    batch: int = Field(default=_DEFAULT_BATCH, gt=0)


def run(cfg: SteeringConfig) -> dict:
    """Run steering dose-response on ``cfg.model``."""
    pairs = load_triviaqa_pairs(cfg.n_pairs)
    with model_session(cfg.model, n_devices=cfg.n_devices) as ctx:
        return _analyse(ctx, pairs, cfg)


def _analyse(ctx: ExperimentContext, pairs: list[tuple[str, str, str]], cfg: SteeringConfig) -> dict:
    steer_layer = cfg.layer if cfg.layer is not None else int(ctx.info.n_layers * cfg.layer_frac)

    dir_pairs = pairs[: cfg.dir_prompts]
    test_pairs = pairs[: cfg.test_prompts]
    dir_wrong, dir_correct = build_sycophancy_prompts(dir_pairs, ctx.model_name)
    test_wrong, _ = build_sycophancy_prompts(test_pairs, ctx.model_name)

    direction = _compute_direction(ctx, dir_wrong, dir_correct, steer_layer, cfg.batch)
    direction_t = torch.tensor(direction, dtype=torch.float32, device=ctx.model.cfg.device)

    rows: list[dict] = []
    for alpha in cfg.alphas:
        rate = _measure_steered_rate(ctx, test_wrong, direction_t, steer_layer, int(alpha), cfg.batch)
        rows.append(
            {
                'alpha': int(alpha),
                'rate': float(rate),
                'n': len(test_wrong),
            }
        )

    result = {
        'model': ctx.model_name,
        'steer_layer': steer_layer,
        'n_layers': ctx.info.n_layers,
        'config': {
            'n_pairs': cfg.n_pairs,
            'test_prompts': cfg.test_prompts,
            'dir_prompts': cfg.dir_prompts,
            'alphas': list(cfg.alphas),
            'layer_frac': cfg.layer_frac,
            'batch': cfg.batch,
        },
        'dose_response': rows,
    }
    save_results(result, 'steering', ctx.model_name)
    return result


def _compute_direction(
    ctx: ExperimentContext,
    wrong: list[str],
    correct: list[str],
    layer: int,
    batch: int,
) -> np.ndarray:
    acts_w = extract_residual_stream(ctx.model, wrong, layer, batch_size=batch)
    acts_c = extract_residual_stream(ctx.model, correct, layer, batch_size=batch)
    raw = acts_w.mean(0) - acts_c.mean(0)
    norm = float(np.linalg.norm(raw))
    # 1e-10 floor keeps the unit-vector well defined when correct/wrong are degenerate
    return raw / (norm + 1e-10)


def _measure_steered_rate(
    ctx: ExperimentContext,
    prompts: list[str],
    direction_t: torch.Tensor,
    layer: int,
    alpha: int,
    batch: int,
) -> float:
    if alpha == 0:
        return measure_agreement_rate(
            ctx.model,
            prompts,
            ctx.agree_tokens,
            ctx.disagree_tokens,
            batch_size=batch,
        )
    pad_id = getattr(ctx.model.tokenizer, 'pad_token_id', None) or 0
    n_correct = 0
    n_total = 0
    agree_idx = list(ctx.agree_tokens)
    disagree_idx = list(ctx.disagree_tokens)
    # The steer hook needs per-batch ``seq_lens`` so we batch manually here and rebuild
    # the closure each iteration; keeping it inside the loop avoids leaking stale state.
    for i in range(0, len(prompts), batch):
        chunk = prompts[i : i + batch]
        tokens = ctx.model.to_tokens(chunk, prepend_bos=True)
        seq_lens = [int(x) for x in ((tokens != pad_id).sum(dim=1) - 1).tolist()]
        hook = _make_steer_hook(int(alpha), direction_t, seq_lens)
        with torch.no_grad():
            logits = ctx.model.run_with_hooks(tokens, fwd_hooks=[(f'blocks.{layer}.hook_resid_post', hook)])
        for b in range(len(chunk)):
            nl = logits[b, seq_lens[b]].float()
            if float(nl[agree_idx].max()) > float(nl[disagree_idx].max()):
                n_correct += 1
            n_total += 1
    return n_correct / n_total if n_total else 0.0


def _make_steer_hook(alpha: int, direction: torch.Tensor, seq_lens: list[int]):
    def hook_fn(t: torch.Tensor, hook: object) -> torch.Tensor:
        d = direction.to(t.device, dtype=t.dtype)
        # Apply only at the last non-pad token, matching the legacy script.
        for b in range(t.shape[0]):
            t[b, seq_lens[b]] = t[b, seq_lens[b]] + alpha * d
        return t

    return hook_fn


def _parse_alphas(value: str | None) -> tuple[int, ...]:
    if value is None:
        return _DEFAULT_ALPHAS
    return tuple(int(x) for x in value.split(','))


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-devices', type=int, default=1)
    parser.add_argument('--n-pairs', type=int, default=_DEFAULT_N_PAIRS)
    parser.add_argument('--test-prompts', type=int, default=_DEFAULT_TEST_PROMPTS)
    parser.add_argument('--dir-prompts', type=int, default=_DEFAULT_DIR_PROMPTS)
    parser.add_argument(
        '--alphas',
        type=str,
        default=None,
        help='Comma-separated ints overriding the default alpha sweep (e.g. "0,-25,25").',
    )
    parser.add_argument('--layer-frac', type=float, default=_DEFAULT_LAYER_FRAC)
    parser.add_argument('--layer', type=int, default=None, help='Override layer-frac with an explicit layer index.')
    parser.add_argument('--batch', type=int, default=_DEFAULT_BATCH)


def from_args(args: argparse.Namespace) -> SteeringConfig:
    """Build the validated config from a parsed argparse namespace."""
    return SteeringConfig(
        model=args.model,
        n_devices=args.n_devices,
        n_pairs=args.n_pairs,
        test_prompts=args.test_prompts,
        dir_prompts=args.dir_prompts,
        alphas=_parse_alphas(args.alphas),
        layer_frac=args.layer_frac,
        layer=args.layer,
        batch=args.batch,
    )
