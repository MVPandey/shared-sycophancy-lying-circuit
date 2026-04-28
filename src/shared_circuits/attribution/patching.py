"""Attribution patching for causal validation of head importance."""

import numpy as np
import torch


@torch.no_grad()
def compute_attribution_patching(
    model: 'HookedTransformer',
    corrupted_prompts: list[str],
    clean_prompts: list[str],
    agree_tokens: list[int],
    disagree_tokens: list[int],
    n_pairs: int = 30,
) -> np.ndarray:
    """
    Compute per-head attribution patching effects.

    For each head, patches the clean activation into the corrupted run and
    measures the change in logit difference (agreement - disagreement).

    Args:
        model: Loaded TransformerLens model.
        corrupted_prompts: Prompts where model may sycophate/lie.
        clean_prompts: Prompts where model should answer correctly.
        agree_tokens: Token IDs for agreement words.
        disagree_tokens: Token IDs for disagreement words.
        n_pairs: Number of prompt pairs to patch.

    Returns:
        Array of shape ``(n_layers, n_heads)`` with mean patching effects.

    Raises:
        ValueError: If prompt lists are empty, mismatched, or ``n_pairs < 1``.

    """
    if len(corrupted_prompts) != len(clean_prompts):
        raise ValueError(f'Prompt lists must have equal length, got {len(corrupted_prompts)} and {len(clean_prompts)}')
    actual_pairs = min(n_pairs, len(corrupted_prompts))
    if actual_pairs < 1:
        raise ValueError(f'Need at least 1 prompt pair, got {actual_pairs}')

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    pad_id = model.tokenizer.pad_token_id or 0
    patch_effects = np.zeros((n_layers, n_heads))

    def _logit_diff(logits: torch.Tensor, seq_len: int) -> float:
        next_logits = logits[0, seq_len]
        agree_score = next_logits[agree_tokens].max().item()
        disagree_score = next_logits[disagree_tokens].max().item()
        return agree_score - disagree_score

    for p_idx in range(actual_pairs):
        corr_tokens = model.to_tokens([corrupted_prompts[p_idx]], prepend_bos=True)
        clean_tokens = model.to_tokens([clean_prompts[p_idx]], prepend_bos=True)
        corr_sl = ((corr_tokens != pad_id).sum(dim=1) - 1)[0]
        clean_sl = ((clean_tokens != pad_id).sum(dim=1) - 1)[0]

        # cache clean activations at every layer
        clean_cache: dict[int, torch.Tensor] = {}
        clean_hooks = []
        for layer in range(n_layers):

            def make_cache(li: int):
                def hook_fn(t: torch.Tensor, hook: object) -> None:
                    clean_cache[li] = t.clone()

                return hook_fn

            clean_hooks.append((f'blocks.{layer}.attn.hook_z', make_cache(layer)))
        model.run_with_hooks(clean_tokens, fwd_hooks=clean_hooks)

        # corrupted baseline logit diff
        corr_logits = model(corr_tokens)
        corr_ld = _logit_diff(corr_logits, corr_sl)

        # patch each head individually
        for layer in range(n_layers):
            for h in range(n_heads):

                def make_patch(target_l: int, target_h: int):
                    def hook_fn(t: torch.Tensor, hook: object) -> torch.Tensor:
                        t[0, corr_sl, target_h, :] = clean_cache[target_l][0, clean_sl, target_h, :]
                        return t

                    return hook_fn

                hooks = [(f'blocks.{layer}.attn.hook_z', make_patch(layer, h))]
                patched_logits = model.run_with_hooks(corr_tokens, fwd_hooks=hooks)
                patched_ld = _logit_diff(patched_logits, corr_sl)
                patch_effects[layer, h] += corr_ld - patched_ld

    patch_effects /= actual_pairs
    return patch_effects
