"""
Behavioral readouts on top of TransformerLens forward passes.

These helpers measure the next-token argmax over agree-vs-disagree tokens at the
final non-padding position of each prompt. Several analyses (steering, head
zeroing, projection ablation, RLHF natural experiments, breadth) measured this
the same way; the helper centralizes the batching + last-token + agree/disagree
logic and accepts an optional ``hooks`` argument so analyses can inject
ablation / steering hooks while measuring.
"""

from collections.abc import Callable, Sequence

import numpy as np
import torch

from shared_circuits.config import DEFAULT_BATCH_SIZE


@torch.no_grad()
def measure_agreement_rate(
    model: 'HookedTransformer',
    prompts: list[str],
    agree_tokens: Sequence[int],
    disagree_tokens: Sequence[int],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hooks: list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]] | None = None,
) -> float:
    """
    Fraction of ``prompts`` whose last-token argmax-agree exceeds argmax-disagree.

    Args:
        model: Loaded TransformerLens model.
        prompts: Prompts to score.
        agree_tokens: Token IDs counted toward "agree" (max over the set).
        disagree_tokens: Token IDs counted toward "disagree" (max over the set).
        batch_size: Forward-pass batch size.
        hooks: Optional ``run_with_hooks`` ``fwd_hooks`` list to inject during
            the forward passes (e.g. steering, head ablation, projection).

    Returns:
        Mean agreement rate over ``prompts`` in [0, 1].

    """
    rate, _ = measure_agreement_per_prompt(
        model,
        prompts,
        agree_tokens,
        disagree_tokens,
        batch_size=batch_size,
        hooks=hooks,
    )
    return rate


@torch.no_grad()
def measure_agreement_per_prompt(
    model: 'HookedTransformer',
    prompts: list[str],
    agree_tokens: Sequence[int],
    disagree_tokens: Sequence[int],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hooks: list[tuple[str, Callable[[torch.Tensor, object], torch.Tensor | None]]] | None = None,
) -> tuple[float, list[float]]:
    """
    Score every prompt as 1 (agree wins) / 0 (disagree wins) and return both the mean and the indicators.

    Same readout as :func:`measure_agreement_rate`; also returns the per-prompt
    indicator list, useful for paired-bootstrap CIs that resample over prompts.

    Returns:
        Tuple of (overall_rate, per_prompt_indicators).

    """
    pad_id = getattr(model.tokenizer, 'pad_token_id', None) or 0
    per_prompt: list[float] = []
    agree_idx = list(agree_tokens)
    disagree_idx = list(disagree_tokens)
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        seq_lens = [int(x) for x in ((tokens != pad_id).sum(dim=1) - 1).tolist()]
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        for b in range(len(batch)):
            nl = logits[b, seq_lens[b]].float()
            per_prompt.append(1.0 if float(nl[agree_idx].max()) > float(nl[disagree_idx].max()) else 0.0)
    return float(np.mean(per_prompt)) if per_prompt else 0.0, per_prompt
