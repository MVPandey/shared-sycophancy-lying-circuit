# `sae-k-sensitivity`

> Could the 41/100 SAE feature overlap be a top-K threshold artifact?

[`sae-feature-overlap`](sae-feature-overlap.md) reports the syc∩lie SAE feature overlap at the single threshold K = 100. This analysis sweeps K through `{10, 50, 100, 200, 500}` on a fixed model and layer, plotting the overlap curve against the `K²/d_sae` chance baseline. If the overlap stays far above chance at every threshold, "we picked a lucky K" is off the table.

## The mech-interp idea

Sparse autoencoders (Cunningham et al. 2023; Bricken et al. 2023) decompose an activation `a ∈ ℝᵈ` into a sparse sum of `m ≫ d` feature activations via a learned encoder/decoder pair, with each `fᵢ` interpretable as a concept. The full-dictionary Spearman ρ in [`sae-feature-overlap`](sae-feature-overlap.md) already addresses K-dependence implicitly — it ranks every feature, not just the top — but the headline number readers carry away is the top-K overlap, and that statistic explicitly depends on the threshold. The standard concern with any "top-K agreement" claim is that the chosen K sits at a sweet spot between two regimes: too small and the intersection is dominated by sample noise, too large and it converges to the trivial `K → d_sae` answer. Sweeping K is the cleanest way to show the effect persists across the regimes.

We use Llama-3.1-8B at layer 19 with the Goodfire 65 536-wide top-K SAE as the default platform — the same `(model, layer)` row that hits the highest ratio in Table 5 (41/100 at 268.7× chance). For each K in the sweep we compute the analytic chance baseline `K² / d_sae`, the observed overlap on the disjoint TriviaQA halves, and the hypergeometric p-value. The curve is the headline, not any single point.

## Why this design

- **Single model, single layer.** This is a robustness check on a single Table 5 row. Sweeping K *and* model would make the result harder to read and add no new information beyond the per-cell K = 100 numbers in [`sae-feature-overlap`](sae-feature-overlap.md). Override with `--model` and `--layer` if you want the same sweep on the other rows.
- **Hypergeometric p only, no permutation null.** The K-sensitivity claim is about the analytic chance baseline, not about the noise floor of a permutation test. The hypergeometric is the right yardstick for the question "could this overlap have come from independent draws".
- **Default K values span 1.5 orders of magnitude.** `{10, 50, 100, 200, 500}` covers the regime from "top of the top" (K = 10 catches only the most differentially active features) through "moderate top" (K = 100 matches the headline) to "broad top" (K = 500 is roughly the top 0.8% of a 65 536-wide dictionary). Override with `--k-values 5,15,30,...`.
- **`d_sae` upper-bounds K.** The analysis raises if any K exceeds the SAE width, so passing K = 100 000 against a 65 536-wide dictionary fails fast rather than silently truncating.

## How to run it

```bash
# Default sweep on Llama-3.1-8B L19 (Goodfire SAE; the Table 5 high-ratio row)
uv run shared-circuits run sae-k-sensitivity

# Same sweep on Gemma-2-2B L12 (Gemma-Scope)
uv run shared-circuits run sae-k-sensitivity \
  --model gemma-2-2b-it --layer 12

# Custom K grid (denser low end)
uv run shared-circuits run sae-k-sensitivity \
  --k-values 5,10,20,50,100,200

# Smaller prompt count for a fast sanity check
uv run shared-circuits run sae-k-sensitivity --n-prompts 50
```

Output: `experiments/results/sae_k_sensitivity_<model_slug>.json`. Top-level keys cover model + SAE provenance (`sae_repo`, `sae_format`, `sae_average_l0` or `sae_top_k`, `d_sae`); the curve itself lives under `curve` as a list of:

| Field | Meaning |
|---|---|
| `k` | Top-K threshold for this row |
| `overlap` | `|top-K syc ∩ top-K lie|` at this K |
| `chance` | `K² / d_sae` |
| `ratio` | `overlap / chance` |
| `p_hypergeometric` | Closed-form p-value at this K |

## Where it lives in the paper

Appendix § "SAE feature overlap: controls and robustness (Llama-3.1-8B, layer 19)" (`app:sae-controls`), as the K-sensitivity-curve paragraph. Headline numbers on Llama-3.1-8B L19 (Goodfire `d_sae = 65 536`): K = 10 → 2 shared (1 311× chance); K = 50 → 12 (315×); K = 100 → 42 (275×); K = 200 → 94 (154×); K = 500 → 229 (60×). The ratio decays as K grows — that's mechanical: chance scales as `K²` so any roughly linear `overlap(K)` would produce a `1/K`-decaying ratio. The point of the sweep is that the ratio stays above ~50× even at K = 500, where chance overlap is ~3.8 features, so the overlap isn't a thresholding artifact.

## Source

`src/shared_circuits/analyses/sae_k_sensitivity.py` (~145 lines). Shares the same `data/sae_features.py` loader and `extraction/sae.py` encoder as [`sae-feature-overlap`](sae-feature-overlap.md); the only difference at the analysis layer is the inner loop iterating over `cfg.k_values` instead of a single K. Reads no upstream JSON; consumed only by manual paper-table generation. The head-level K-sensitivity counterpart is the unstratified-vs-stratified sweep documented under [`layer-strat-null`](layer-strat-null.md).
