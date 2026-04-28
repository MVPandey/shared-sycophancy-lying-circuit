# `opinion-circuit-transfer`

> Rank heads on the opinion-agreement task. Do the top heads overlap with sycophancy?

This is the opinion analogue of [`circuit-overlap`](circuit-overlap.md). It computes a per-head importance grid for an opinion-agreement contrast, ranks the heads, and tests the overlap against the persisted sycophancy ranking with a permutation null. The persisted opinion grid is also the input that [`triple-intersection`](triple-intersection.md) and [`opinion-causal`](opinion-causal.md) consume.

<p align="center">
  <img src="../img/opinion_generalization.png" width="600" alt="Opinion-task head overlap with sycophancy across five models. Same parking lot — the overlap test panel of fig 3a.">
</p>

## The mech-interp idea

We score every attention head by how much it *cares* about the opinion-agreement task: the L2 norm of the difference of mean per-head outputs on the agree-prompt vs disagree-prompt conditions. Same write-norm form of Direct Logit Attribution as [`circuit-overlap`](circuit-overlap.md) — a head fires identically on "I think pineapple belongs on pizza" and "I think pineapple does not belong on pizza" gets a low score; a head whose value vector turns differentially on the two gets a high score.

Once we have a `(layer, head)` grid of opinion scores, we ask whether the top-K opinion heads overlap with the top-K sycophancy heads above what shuffling would produce. The permutation null shuffles the opinion grid `n_perm = 10,000` times and recomputes the overlap; the p-value is the fraction of shuffles that meet or beat the observed count. We also report Spearman ρ and Pearson r over the full head population, plus a chance baseline `K²/N` (matching [`circuit-overlap`](circuit-overlap.md)'s convention).

The opinion task itself is contrasted with no factual ground truth — pairs are drawn from `generate_opinion_pairs`, a generated set of contested claims like "Pineapple belongs on pizza." The contrast is between agreeing with the user's stated opinion and disagreeing. This is the §3.6 generalization probe: the §3.4 factual-correctness story leaves open whether the circuit is *truth*-specific or a more generic *statement-evaluation* substrate, and opinion-agreement is the obvious null-of-truth control.

## Why this design

- **Cousin of [`circuit-overlap`](circuit-overlap.md), not a clone.** Same write-norm DLA, same `K = ⌈√N⌉` threshold, same permutation null mechanic. Different prompt set (opinion pairs, not TriviaQA), and persists a *grid* (not just top-15) so [`triple-intersection`](triple-intersection.md) can replay the intersection on the fly.
- **Reads syc grid from `circuit_overlap` JSON.** `_load_syc_grid` first tries `syc_grid` at the top level, then under `head_overlap`, and as a last resort rebuilds a sparse grid from `syc_top15` entries. The fallback exists because some legacy result files don't persist the full grid.
- **`generate_opinion_pairs`, not loaded from disk.** The opinion pair generator is deterministic in `cfg.seed`, which means re-running this analysis on a new model uses the same opinions — clean for cross-model comparison.
- **Multi-model default panel.** `_DEFAULT_MODELS` covers Gemma-2-2B/9B, Qwen3-8B, Llama-3.1-8B, and Llama-3.1-70B — the same five-model panel that figure 3a in §3.6 is plotted on.
- **Top-K opinion heads persisted as `shared_heads`.** Naming reads weird in isolation but is consistent with how [`circuit-overlap`](circuit-overlap.md) labels its overlap output, so the downstream consumers ([`opinion-causal`](opinion-causal.md), [`triple-intersection`](triple-intersection.md)) read the same key.

## How to run it

```bash
# Default five-model panel (writes per-model JSONs)
uv run shared-circuits run opinion-circuit-transfer

# Single-model with a tighter null (100k permutations)
uv run shared-circuits run opinion-circuit-transfer \
  --models gemma-2-2b-it --permutations 100000

# Bigger DLA-prompt budget for cleaner head ranking
uv run shared-circuits run opinion-circuit-transfer \
  --models meta-llama/Llama-3.1-70B-Instruct --dla-prompts 200

# If your syc grid is in breadth output rather than circuit_overlap
uv run shared-circuits run opinion-circuit-transfer --syc-from breadth
```

Output: `experiments/results/opinion_circuit_transfer_<model>.json`. Key fields:

| Field | Meaning |
|---|---|
| `opinion_grid` | Full `(n_layers, n_heads)` write-norm grid — input to [`triple-intersection`](triple-intersection.md) |
| `opinion_top15` | Per-head top-15 ranking with `delta` (the write-norm score) |
| `shared_heads` | Top-K opinion heads as `[layer, head]` pairs — input to [`opinion-causal`](opinion-causal.md) |
| `overlap_with_syc.overlap` / `ratio_vs_chance` / `p_permutation` | Permutation overlap with the persisted syc ranking |
| `overlap_with_syc.spearman` / `pearson` | Full-population correlations |
| `k` / `chance` | Top-K threshold (`⌈√N⌉`) and analytic chance `K²/N` |

## Where it lives in the paper

§3.6, **Figure 3a** (`fig:opinion`). The opinion-vs-syc overlap on this analysis is the precursor to the triple-intersection in [`triple-intersection`](triple-intersection.md). The §3.6 conclusion combines this result (opinion overlaps with syc above chance), the triple-intersection result (51–1,755× chance), and the orthogonality direction-cosine result from [`opinion-causal`](opinion-causal.md) `boundary` mode (`|cos| < 0.14`) into the "same positions, orthogonal subspace" framing.

## Source

`src/shared_circuits/analyses/opinion_circuit_transfer.py` (~205 lines). Reads `circuit_overlap` (or `breadth`) per model for the syc grid; persists the opinion grid that [`triple-intersection`](triple-intersection.md) and [`opinion-causal`](opinion-causal.md) read downstream. The head-importance helper (`shared_circuits.attribution.compute_head_importances`) is the same primitive `circuit_overlap` uses for its rankings, just driven on opinion prompts instead of TriviaQA.
