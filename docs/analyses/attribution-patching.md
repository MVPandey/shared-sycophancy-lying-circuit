# `attribution-patching`

> The expensive causal intervention that the cheap DLA proxy is meant to approximate. Does the proxy hold up?

[`circuit-overlap`](circuit-overlap.md) ranks heads by the L2 norm of their write-vector difference between positive and negative prompts — a write-norm form of Direct Logit Attribution (DLA). DLA is fast (`O(1)` forward passes per task) but it's a first-order approximation. The gold-standard causal intervention used in IOI circuit analysis (Wang et al. 2023) is *clean→corrupt activation patching*: cache the clean (correct-answer) activation of each head, run the prompt corrupted (wrong-answer), splice each head's clean activation back in turn, and measure the resulting logit-diff shift. This costs `O(n_layers × n_heads)` forward passes per task — intractable at frontier scale, but doable up to 8B. This analysis runs the gold-standard intervention on three sub-8B models and validates the DLA proxy against it.

<p align="center">
  <img src="../img/causal_convergence.png" width="600" alt="Three causal interventions on the shared-head set converge across model scales: per-head activation patching at 8B and below, projection ablation and path patching from 2B through 70B.">
</p>

## The mech-interp idea

DLA's L2 write-norm `||W_O · (v̄⁺ − v̄⁻)||₂` measures *how distinctively* a head writes between conditions, but it doesn't measure how much that distinctive write *causally affects the output*. A head can write a large differential signal that downstream components ignore, or a small one that gets amplified by the rest of the circuit. To distinguish, we need an actual intervention.

Per-head activation patching does this in three steps. For each `(layer, head)`:

1. Run the *clean* prompt (e.g., user states the correct answer). Cache the head's `z` activation at the measurement position.
2. Run the *corrupt* prompt (e.g., user states a wrong answer). Cache its `z` too.
3. Re-run the corrupt prompt, but splice the head's clean `z` into its position. Measure the resulting logit difference between agree and disagree tokens. The difference from the corrupt baseline is the head's **patching effect**.

This is the same mechanic Wang et al. used to identify the IOI circuit head-by-head; we run it independently for sycophancy (clean = correct-opinion, corrupt = wrong-opinion) and factual lying (clean = true statement, corrupt = false statement) on disjoint TriviaQA halves. Two questions follow:

1. **Does the patching ranking reproduce the head-overlap result?** Top-K(syc-patching) ∩ top-K(lie-patching) should be significantly above chance, the same way the DLA-ranked overlap is.
2. **Do the two patching grids correlate beyond what DLA alone would predict?** If the *syc-patching grid* and *lie-patching grid* are similar at the per-head level, the shared causal structure is real, not just an artifact of DLA's coarse approximation.

The paper finds both: top-K overlap reproduces (10/15, 11/15, 3/15 across the three tested models), and per-grid Pearson `r = 0.49–0.93` between the two patching grids — well above the `r = 0.41–0.61` between patching and DLA. So the DLA proxy is a *lower bound* on the causal sharing; the gold-standard intervention shows even tighter agreement.

The slug "attribution-patching" is a slight legacy misnomer. Strict attribution patching (Syed et al. 2023) uses a gradient-based linear approximation; this analysis runs the more expensive but more faithful **direct activation patching**. Compute-bounded to ≤ 8B because per-head sweeps at ≥ 32B cost > 50 GPU-hours per model on our hardware.

## Why this design

- **Compute-bounded to ≤ 8B by design.** The default panel is Gemma-2-2B-IT, Qwen2.5-1.5B, Llama-3.1-8B. At ≥ 32B, per-head patching is replaced by [`activation-patching`](activation-patching.md)'s top-K-shared-set variant, which intervenes on the shared set as a unit instead of head-by-head.
- **Disjoint TriviaQA halves for the two tasks.** Pairs `[0, 200)` for sycophancy, `[200, 400)` for lying. Identical disjointness as [`circuit-overlap`](circuit-overlap.md), so the cross-task correlation isn't being driven by shared trivia surface forms.
- **`n_patch_pairs=30` by default.** This is enough at 2B for stable per-head means; the legacy Llama-3.1-8B run used `n=150` (passable via `--n-patch-pairs 150`) because the 8B grid is sparser and `n=30` overweighted single-prompt noise. The paper's Table footnote calls this out for the Llama-3.1-8B row.
- **Top-K = 15 fixed for the overlap calculation.** Paper-table convention. Override with `--overlap-k`.
- **Reports `dla_comparison.available` so downstream tools can cross-reference.** Doesn't recompute DLA; expects the [`circuit-overlap`](circuit-overlap.md) JSON to exist already if you want the paper-table `r_DLA` column. The actual cross-grid correlation is computed separately by paper-table generation.
- **Output is the full `(n_layers, n_heads)` patching grid for both tasks.** This is what enables the `r=0.49–0.93` cross-grid Pearson and feeds the `r_DLA = 0.41–0.61` validation. Saving the full grid (not just top-K) is what makes the proxy-validation claim load-bearing.

## How to run it

```bash
# Default 3-model panel (~hours on a single 80GB GPU)
uv run shared-circuits run attribution-patching

# Single model, more pairs (the 150-pair Llama-3.1-8B configuration)
uv run shared-circuits run attribution-patching \
  --models meta-llama/Llama-3.1-8B-Instruct --n-patch-pairs 150

# Smaller test, just for proof-of-life
uv run shared-circuits run attribution-patching \
  --models gemma-2-2b-it --n-patch-pairs 10

# Wider top-K for the overlap calculation
uv run shared-circuits run attribution-patching \
  --models Qwen/Qwen2.5-1.5B-Instruct --overlap-k 30
```

Output: `experiments/results/attribution_patching_<model>.json`. Key fields:

| Field | Meaning |
|---|---|
| `patch_pearson_r` | Pearson `r` between flattened sycophancy and lying patching grids |
| `patch_k15_overlap` | Top-K = 15 head intersection between patching-derived rankings |
| `patch_k15_ratio` | `overlap / (K²/N)` — chance-normalized overlap |
| `patch_k15_p_hypergeom` | Hypergeometric p-value on the overlap |
| `syc_patch_grid`, `lie_patch_grid` | Full per-head patching effects, shape `(n_layers, n_heads)` |
| `dla_comparison.available` | Whether a sibling `circuit_overlap_<model>.json` exists for cross-comparison |

## Where it lives in the paper

Appendix § "Per-head activation patching detail", Table `tab:attrpatch`. Headline numbers:

| Model | n_pairs | Overlap (K=15) | Ratio | r(syc↔lie patching grids) | r(patching↔DLA) |
|---|---|---|---|---|---|
| Gemma-2-2B-IT | 30 | 10/15 | 9.2× | 0.78 | 0.53–0.61 |
| Qwen2.5-1.5B | 30 | 11/15 | 16.4× | 0.93 | 0.41–0.47 |
| Llama-3.1-8B | 150 | 3/15 | 2.8× | 0.49 | 0.56–0.59 |

The §3.4 sentence "the write-norm proxy is validated against the gold-standard intervention it substitutes for" cites this analysis. Combined with [`projection-ablation`](projection-ablation.md), [`head-zeroing`](head-zeroing.md), and [`activation-patching`](activation-patching.md), this is the third leg of the §3.4 sufficiency-convergence figure (`causal_convergence.png`).

## Source

`src/shared_circuits/analyses/attribution_patching.py` (~115 lines). The heavy lifting (`compute_attribution_patching`) lives in `src/shared_circuits/attribution/patching.py` — note that despite the slug, it implements direct activation patching, not gradient-based attribution patching. Reads no prior results; produces `attribution_patching_<model>.json`. Cross-references [`circuit-overlap`](circuit-overlap.md) for the DLA grids if a paper-table generator wants the proxy-validation `r_DLA` column.
