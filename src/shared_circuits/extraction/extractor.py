"""Batched forward-pass runner with per-prompt last-token extraction."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch

from shared_circuits.config import DEFAULT_BATCH_SIZE


@dataclass(frozen=True, slots=True)
class HookSpec:
    """
    A TransformerLens forward hook installed on each batched pass.

    Used for activations the extractor mutates rather than captures (e.g. zeroing
    attention heads). The captured-activation case is expressed via
    ``BatchedExtractor.run``'s ``capture_hooks`` mapping instead.
    """

    name: str
    fn: Callable[[torch.Tensor, object], torch.Tensor | None]


class BatchedExtractor:
    """Runs a TransformerLens model on prompts with hooks, returning last-token activations."""

    def __init__(self, model: 'HookedTransformer', batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """Initialize the extractor with a model and a per-call batch size."""
        self._model = model
        self._batch_size = batch_size
        self._pad_id = model.tokenizer.pad_token_id or 0

    @torch.no_grad()
    def run(
        self,
        prompts: list[str],
        capture_hooks: dict[str, str] | None = None,
        mutate_hooks: Iterable[HookSpec] = (),
        stop_at_layer: int | None = None,
        return_logits: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Run the model on ``prompts``, returning last-token slices keyed by ``store_key``.

        Args:
            prompts: Input prompts to feed through the model.
            capture_hooks: Map from desired return-key to TransformerLens hook name
                (e.g. ``{'r14': 'blocks.14.hook_resid_post'}``). Each hook captures
                activations at the registered point and the last non-pad token slice
                is returned under the matching key.
            mutate_hooks: ``HookSpec`` instances that modify activations in place during
                the forward pass (e.g. head ablations). Their outputs are not captured.
            stop_at_layer: Optional early-stopping layer for ``run_with_hooks``.
            return_logits: When ``True``, includes the last-token logits under the
                ``'logits'`` key.

        Returns:
            Dict mapping each ``capture_hooks`` key (and optionally ``'logits'``) to a
            ``(n_prompts, ...)`` numpy array.

        """
        capture_hooks = capture_hooks or {}
        mutate_specs = list(mutate_hooks)

        accum: dict[str, list[np.ndarray]] = {key: [] for key in capture_hooks}
        if return_logits:
            accum['logits'] = []

        for i in range(0, len(prompts), self._batch_size):
            batch = prompts[i : i + self._batch_size]
            tokens = self._model.to_tokens(batch, prepend_bos=True)
            seq_lens = (tokens != self._pad_id).sum(dim=1) - 1

            store: dict[str, torch.Tensor] = {}
            fwd_hooks: list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]] = []

            for store_key, hook_name in capture_hooks.items():
                fwd_hooks.append((hook_name, _make_capture(store, store_key)))

            for spec in mutate_specs:
                fwd_hooks.append((spec.name, spec.fn))

            run_kwargs: dict[str, object] = {'fwd_hooks': fwd_hooks}
            if stop_at_layer is not None:
                run_kwargs['stop_at_layer'] = stop_at_layer
            logits = self._model.run_with_hooks(tokens, **run_kwargs)

            for store_key in capture_hooks:
                accum[store_key].append(_last_token_slice(store[store_key], seq_lens, len(batch)))

            if return_logits:
                accum['logits'].append(_last_token_slice(logits, seq_lens, len(batch)))

        return {key: np.concatenate(parts, axis=0) for key, parts in accum.items()}


def _make_capture(store: dict[str, torch.Tensor], key: str) -> Callable[[torch.Tensor, object], None]:
    # Late-binding closures: ``key`` must be captured per-iteration, otherwise every
    # registered hook would write to the same dict slot.
    def hook_fn(t: torch.Tensor, hook: object) -> None:
        store[key] = t

    return hook_fn


def _last_token_slice(activations: torch.Tensor, seq_lens: torch.Tensor, batch_len: int) -> np.ndarray:
    last = torch.stack([activations[j, seq_lens[j]] for j in range(batch_len)])
    return last.float().cpu().numpy()
