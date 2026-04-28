"""Direct Logit Attribution for per-head importance analysis."""

import numpy as np
import torch

from shared_circuits.config import DEFAULT_BATCH_SIZE, DEFAULT_N_PROMPTS, DEFAULT_TOP_K


@torch.no_grad()
def compute_head_importances(
    model: 'HookedTransformer',
    prompts_pos: list[str],
    prompts_neg: list[str],
    n_prompts: int = DEFAULT_N_PROMPTS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[tuple[int, int], float]:
    """
    Compute per-head importance via DLA mean-difference norms.

    For each attention head, projects its output through ``W_O``, then computes
    the L2 norm of the mean difference between positive and negative conditions.

    Args:
        model: Loaded TransformerLens model.
        prompts_pos: Positive condition prompts (e.g. wrong-opinion).
        prompts_neg: Negative condition prompts (e.g. correct-opinion).
        n_prompts: Max prompts per condition.
        batch_size: Batch size for inference.

    Returns:
        Dict mapping ``(layer, head)`` to importance score (L2 norm of delta).

    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    pad_id = model.tokenizer.pad_token_id or 0

    w_o: dict[int, torch.Tensor] = {}
    for layer in range(n_layers):
        # keep W_O on its native device (matters for multi-GPU pipeline parallel)
        w_o[layer] = model.blocks[layer].attn.W_O.float()

    def _get_projections(prompts: list[str]) -> dict[tuple[int, int], np.ndarray]:
        head_projs: dict[tuple[int, int], list[np.ndarray]] = {
            (layer, h): [] for layer in range(n_layers) for h in range(n_heads)
        }
        for i in range(0, min(len(prompts), n_prompts), batch_size):
            batch = prompts[i : i + batch_size]
            tokens = model.to_tokens(batch, prepend_bos=True)
            seq_lens = (tokens != pad_id).sum(dim=1) - 1

            stores: dict[int, torch.Tensor] = {}
            hooks = []
            for layer in range(n_layers):

                def make_hook(li: int):
                    def hook_fn(t: torch.Tensor, hook: object) -> None:
                        stores[li] = t

                    return hook_fn

                hooks.append((f'blocks.{layer}.attn.hook_z', make_hook(layer)))
            model.run_with_hooks(tokens, fwd_hooks=hooks)

            for layer in range(n_layers):
                z = stores[layer]
                w_o_layer = w_o[layer].to(z.device)
                for b_idx in range(len(batch)):
                    last_pos = int(seq_lens[b_idx].item())
                    for h in range(n_heads):
                        z_h = z[b_idx, last_pos, h, :].float()
                        head_out = z_h @ w_o_layer[h]
                        head_projs[(layer, h)].append(head_out.cpu().numpy())

        return {k: np.stack(v) for k, v in head_projs.items() if v}

    projs_pos = _get_projections(prompts_pos)
    projs_neg = _get_projections(prompts_neg)

    head_deltas: dict[tuple[int, int], float] = {}
    for key in projs_pos.keys() & projs_neg.keys():
        mean_pos = projs_pos[key].mean(axis=0)
        mean_neg = projs_neg[key].mean(axis=0)
        delta = mean_pos - mean_neg
        head_deltas[key] = float(np.linalg.norm(delta))

    return head_deltas


def rank_heads(
    head_deltas: dict[tuple[int, int], float],
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[tuple[int, int], float]]:
    """
    Rank heads by importance score, return top-K.

    Args:
        head_deltas: Dict mapping ``(layer, head)`` to importance score.
        top_k: Number of top heads to return.

    Returns:
        List of ``((layer, head), score)`` tuples, sorted descending by score.

    """
    ranked = sorted(head_deltas.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def compute_head_importance_grid(
    head_deltas: dict[tuple[int, int], float],
    n_layers: int,
    n_heads: int,
) -> np.ndarray:
    """
    Convert head importance dict to a 2D grid array.

    Args:
        head_deltas: Dict mapping ``(layer, head)`` to importance score.
        n_layers: Number of layers.
        n_heads: Number of heads per layer.

    Returns:
        Array of shape ``(n_layers, n_heads)`` with importance scores.

    """
    grid = np.zeros((n_layers, n_heads))
    for (layer, head), score in head_deltas.items():
        grid[layer, head] = score
    return grid
