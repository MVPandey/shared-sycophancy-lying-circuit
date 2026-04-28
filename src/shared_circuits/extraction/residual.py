"""Last-token residual stream extraction at one or more layers."""

import numpy as np

from shared_circuits.config import DEFAULT_BATCH_SIZE
from shared_circuits.extraction.extractor import BatchedExtractor


def extract_residual_stream(
    model: 'HookedTransformer',
    prompts: list[str],
    layer: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Extract residual stream activations at a single layer.

    Captures the residual stream at the last non-padding token position.

    Args:
        model: Loaded TransformerLens model.
        prompts: Input prompts.
        layer: Layer index to extract from.
        batch_size: Batch size for inference.

    Returns:
        Array of shape ``(n_prompts, d_model)`` with last-token activations.

    """
    extractor = BatchedExtractor(model, batch_size)
    out = extractor.run(
        prompts,
        capture_hooks={'r': f'blocks.{layer}.hook_resid_post'},
        stop_at_layer=layer + 1,
    )
    return out['r']


def extract_residual_stream_multi(
    model: 'HookedTransformer',
    prompts: list[str],
    layers: list[int],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[int, np.ndarray]:
    """
    Extract residual stream activations at multiple layers simultaneously.

    Args:
        model: Loaded TransformerLens model.
        prompts: Input prompts.
        layers: Layer indices to extract from.
        batch_size: Batch size for inference.

    Returns:
        Dict mapping layer index to ``(n_prompts, d_model)`` arrays.

    """
    extractor = BatchedExtractor(model, batch_size)
    captures = {f'r{layer}': f'blocks.{layer}.hook_resid_post' for layer in layers}
    out = extractor.run(prompts, capture_hooks=captures, stop_at_layer=max(layers) + 1)
    return {layer: out[f'r{layer}'] for layer in layers}
