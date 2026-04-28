"""Project residual-stream activations through a loaded SAE."""

import numpy as np
import torch

from shared_circuits.data.sae_features import SAE_KIND_GEMMA_SCOPE, SAE_KIND_GOODFIRE_TOPK, SaeWeights
from shared_circuits.extraction.residual import extract_residual_stream


@torch.no_grad()
def encode_residuals(
    residuals: torch.Tensor,
    sae: SaeWeights,
    top_k: int | None = None,
) -> torch.Tensor:
    """
    Apply the SAE encoder to residual-stream activations.

    Args:
        residuals: Float tensor of shape ``(n_prompts, d_model)`` (or any leading dims).
        sae: Loaded :class:`SaeWeights`.
        top_k: When set, zero out all but the top-K features per row before returning. When
            ``None``, the family-default gating (JumpReLU threshold or Goodfire top-K) is used.

    Returns:
        Float tensor of shape ``(n_prompts, d_sae)`` with feature activations.

    """
    x = residuals.to(sae.w_enc.device, dtype=sae.w_enc.dtype)
    pre = (x - sae.b_dec) @ sae.w_enc + sae.b_enc
    if sae.kind == SAE_KIND_GEMMA_SCOPE:
        # JumpReLU: feature is active iff pre-activation exceeds its threshold.
        if sae.threshold is None:
            raise ValueError('Gemma-Scope SAE requires a threshold tensor')
        feats = pre * (pre > sae.threshold).to(pre.dtype)
    elif sae.kind == SAE_KIND_GOODFIRE_TOPK:
        if sae.top_k is None:
            raise ValueError('Goodfire SAE requires top_k')
        relu = torch.relu(pre)
        k = int(sae.top_k)
        feats = _topk_sparsify(relu, k)
    else:
        raise ValueError(f'Unknown SAE kind: {sae.kind}')

    if top_k is not None and top_k < feats.shape[-1]:
        feats = _topk_sparsify(feats, int(top_k))
    return feats


def feature_activation_grid(
    residuals: torch.Tensor,
    sae: SaeWeights,
    top_k: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the per-prompt top-K SAE features as ``(indices, values)``.

    Args:
        residuals: Float tensor ``(n_prompts, d_model)``.
        sae: Loaded SAE.
        top_k: Number of top features to return per prompt.

    Returns:
        Tuple ``(top_idx, top_vals)`` where each tensor has shape ``(n_prompts, top_k)``.

    """
    feats = encode_residuals(residuals, sae)
    k = min(int(top_k), feats.shape[-1])
    top_vals, top_idx = feats.topk(k, dim=-1)
    return top_idx, top_vals


def encode_prompts(
    model: 'HookedTransformer',
    prompts: list[str],
    sae: SaeWeights,
    layer: int,
    *,
    batch_size: int = 4,
) -> np.ndarray:
    """
    End-to-end helper: capture residuals at ``layer``, encode through ``sae``, return a numpy array.

    Args:
        model: Loaded TransformerLens model.
        prompts: Last-token prompts to feed through the model.
        sae: Loaded SAE used to project the captured residuals.
        layer: Residual-stream layer the SAE matches.
        batch_size: Forward-pass batch size.

    Returns:
        ``(len(prompts), d_sae)`` ``float32`` numpy array of feature activations.

    """
    resid_np = extract_residual_stream(model, prompts, layer, batch_size=batch_size)
    resid = torch.from_numpy(resid_np)
    feats = encode_residuals(resid, sae)
    return feats.float().cpu().numpy()


def _topk_sparsify(activations: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all but the top-``k`` features per row (last dim)."""
    if k >= activations.shape[-1]:
        return activations
    top_vals, top_idx = activations.topk(k, dim=-1)
    out = torch.zeros_like(activations)
    out.scatter_(-1, top_idx, top_vals)
    return out
