"""TransformerLens model loading, metadata extraction, and cleanup."""

import gc
import json
import os
from dataclasses import dataclass

import torch

from shared_circuits.models.parallelism import distribute_pipeline_parallel


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Metadata about a loaded model."""

    name: str
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    total_heads: int


def _load_weight_mirrors() -> dict[str, str]:
    """Read ``WEIGHT_MIRRORS_JSON`` env var (mapping model_name -> alternative weight repo)."""
    raw = os.environ.get('WEIGHT_MIRRORS_JSON', '')
    if not raw:
        return {}
    try:
        return dict(json.loads(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def load_model(
    model_name: str,
    device: str = 'cuda',
    dtype: str = 'bfloat16',
    n_devices: int = 1,
    weight_repo: str | None = None,
) -> 'HookedTransformer':
    """
    Load a model via TransformerLens with no preprocessing.

    Args:
        model_name: HuggingFace model name or TransformerLens alias (used to identify
            the architecture/config that TransformerLens registers).
        device: Device to load onto.
        dtype: Data type string (e.g. ``'bfloat16'``).
        n_devices: Number of GPUs for pipeline parallelism (default 1).
        weight_repo: Optional alternative HF repo to load weights from while still
            telling TL the architecture is ``model_name``. Useful when the official
            repo is gated and a public mirror exists. Falls back to env-var
            ``WEIGHT_MIRRORS_JSON`` (JSON dict of model_name -> mirror_repo) if None.

    Returns:
        Loaded HookedTransformer instance.

    """
    from transformer_lens import HookedTransformer

    from shared_circuits.config import TL_ARCH_OVERRIDES

    torch_dtype = getattr(torch, dtype)
    if weight_repo is None:
        weight_repo = _load_weight_mirrors().get(model_name)
    tl_name = TL_ARCH_OVERRIDES.get(model_name)
    if tl_name is not None and weight_repo is None:
        weight_repo = model_name
        model_name = tl_name

    # Load HF model on CPU, convert to TL on CPU with move_to_device=False, cast to the
    # correct dtype, then move to GPU.  This avoids two TL bugs:
    #   1. TL loads HF to GPU internally, doubling VRAM during conversion.
    #   2. TL's MoE uses nn.Linear (float32 default) instead of
    #      nn.Parameter(..., dtype=cfg.dtype), so MoE models like Mixtral end up
    #      2x expected size until explicitly cast.
    from transformers import AutoModelForCausalLM

    hf_kwargs: dict[str, object] = {'torch_dtype': torch_dtype, 'low_cpu_mem_usage': True, 'device_map': 'cpu'}
    repo = weight_repo or model_name
    if weight_repo:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(weight_repo)
        hf_model = AutoModelForCausalLM.from_pretrained(repo, **hf_kwargs)
        model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            hf_model=hf_model,
            tokenizer=tokenizer,
            device='cpu',
            dtype=torch_dtype,
            n_devices=1,
            move_to_device=False,
        )
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(repo, **hf_kwargs)
        model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            hf_model=hf_model,
            device='cpu',
            dtype=torch_dtype,
            n_devices=1,
            move_to_device=False,
        )

    del hf_model
    gc.collect()

    model = model.to(torch_dtype)
    model.cfg.n_devices = n_devices
    model.cfg.device = device
    if device == 'cpu':
        return model
    if n_devices == 1:
        return model.to(device)
    return distribute_pipeline_parallel(model, device, n_devices)


def get_model_info(model: 'HookedTransformer') -> ModelInfo:
    """
    Extract metadata from a loaded model.

    Args:
        model: Loaded TransformerLens model.

    Returns:
        ModelInfo dataclass with model dimensions.

    """
    cfg = model.cfg
    return ModelInfo(
        name=cfg.model_name,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        total_heads=cfg.n_layers * cfg.n_heads,
    )


def cleanup_model(model: 'HookedTransformer') -> None:
    """
    Free GPU memory after model use.

    Args:
        model: Model to clean up.

    """
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
