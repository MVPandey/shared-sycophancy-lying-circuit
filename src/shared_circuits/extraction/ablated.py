"""Activation extraction with attention-head ablations applied during the forward pass."""

from collections.abc import Callable

import numpy as np
import torch

from shared_circuits.config import DEFAULT_BATCH_SIZE
from shared_circuits.extraction.extractor import BatchedExtractor, HookSpec


def extract_with_head_ablation(
    model: 'HookedTransformer',
    prompts: list[str],
    ablate_heads: list[tuple[int, int]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Run model with specified attention heads zeroed out, return last-token logits.

    Args:
        model: Loaded TransformerLens model.
        prompts: Input prompts.
        ablate_heads: List of ``(layer, head)`` tuples to zero at ``hook_z``.
        batch_size: Batch size for inference.

    Returns:
        Array of shape ``(n_prompts, vocab_size)`` with last-token logits.

    """
    extractor = BatchedExtractor(model, batch_size)
    out = extractor.run(
        prompts,
        mutate_hooks=_ablation_specs(ablate_heads),
        return_logits=True,
    )
    return out['logits']


def extract_residual_with_ablation(
    model: 'HookedTransformer',
    prompts: list[str],
    layer: int,
    ablate_heads: list[tuple[int, int]] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Extract residual stream at a layer, optionally with head ablation.

    Args:
        model: Loaded TransformerLens model.
        prompts: Input prompts.
        layer: Layer to capture residual stream from.
        ablate_heads: Optional list of ``(layer, head)`` tuples to zero at ``hook_z``.
        batch_size: Batch size for inference.

    Returns:
        Array of shape ``(n_prompts, d_model)`` with last-token activations.

    """
    extractor = BatchedExtractor(model, batch_size)
    out = extractor.run(
        prompts,
        capture_hooks={'r': f'blocks.{layer}.hook_resid_post'},
        mutate_hooks=_ablation_specs(ablate_heads or []),
    )
    return out['r']


def _ablation_specs(ablate_heads: list[tuple[int, int]]) -> list[HookSpec]:
    specs: list[HookSpec] = []
    for layer, head in ablate_heads:
        specs.append(HookSpec(name=f'blocks.{layer}.attn.hook_z', fn=_make_ablate(head)))
    return specs


def _make_ablate(target_h: int) -> Callable[[torch.Tensor, object], torch.Tensor]:
    # Late-binding: ``target_h`` must be captured per-iteration so each registered
    # hook zeros its own head rather than the last one in the loop.
    def hook_fn(z: torch.Tensor, hook: object) -> torch.Tensor:
        z[:, :, target_h, :] = 0.0
        return z

    return hook_fn
