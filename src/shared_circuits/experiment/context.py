"""Model lifecycle context manager and the value object it yields."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from shared_circuits.config import DEFAULT_DEVICE, DEFAULT_DTYPE
from shared_circuits.models import (
    ModelInfo,
    cleanup_model,
    get_agree_disagree_tokens,
    get_model_info,
    load_model,
)


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    """Loaded model plus precomputed metadata used by every analysis."""

    model: 'HookedTransformer'
    info: ModelInfo
    model_name: str
    agree_tokens: tuple[int, ...]
    disagree_tokens: tuple[int, ...]


@contextmanager
def model_session(
    model_name: str,
    *,
    device: str = DEFAULT_DEVICE,
    dtype: str = DEFAULT_DTYPE,
    n_devices: int = 1,
) -> Iterator[ExperimentContext]:
    """
    Load a model, yield an ExperimentContext, free GPU memory on exit.

    Example:
        with model_session('gemma-2-2b-it') as ctx:
            extract_residual_stream(ctx.model, prompts, layer=10)

    Args:
        model_name: HuggingFace model name or TransformerLens alias.
        device: Device to load onto.
        dtype: Data type string (e.g. ``'bfloat16'``).
        n_devices: Number of GPUs for pipeline parallelism (default 1).

    Yields:
        ExperimentContext bundling the model with its metadata and agree/disagree tokens.

    """
    model = load_model(model_name, device=device, dtype=dtype, n_devices=n_devices)
    try:
        info = get_model_info(model)
        agree, disagree = get_agree_disagree_tokens(model)
        yield ExperimentContext(
            model=model,
            info=info,
            model_name=model_name,
            agree_tokens=tuple(agree),
            disagree_tokens=tuple(disagree),
        )
    finally:
        cleanup_model(model)
