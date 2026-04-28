# `linear-probe-sae-alignment`

> A logistic-regression probe finds the most discriminative direction in residual space. Does that direction line up with the SAE features the overlap analysis already named?

[`sae-feature-overlap`](sae-feature-overlap.md) identifies a shared-feature set by ranking *features* by their `|mean-diff|` between conditions. A linear probe is the orthogonal angle on the same data: rank *directions* by their classification margin, then project the resulting direction onto the SAE dictionary. If the two procedures agree — the top-aligned features are the shared features — the SAE evidence isn't an artifact of the mean-difference ranking, and the model really does encode "this is wrong" in a low-dimensional subspace spanned by a handful of identifiable features.

## The mech-interp idea

A **linear probe** here is scikit-learn's L2-regularized logistic regression fit on per-prompt residual-stream activations at one layer. Train on the syc-wrong vs syc-correct labels (or lie-false vs lie-true), and the resulting `d_model`-dim weight vector `w` is the direction in residual space that best separates the two conditions under the maximum-likelihood objective. We measure held-out **AUROC** with 5-fold stratified cross-validation, then refit on all data to extract a stable `w` for the alignment computation.

Sparse autoencoders (Cunningham et al. 2023; Bricken et al. 2023) decompose an activation `a ∈ ℝᵈ` into a sparse sum of `m ≫ d` features, with each feature `i` "represented by" decoder column `W_dec[:, i] ∈ ℝᵈ`. The natural alignment between a probe direction and an SAE feature is the **absolute cosine** `|w · W_dec[:, i] / (‖w‖ · ‖W_dec[:, i]‖)|`. Computing this for every feature gives a vector of `d_sae` alignment scores; we report the top-K aligned features alongside the *shared* features named by [`sae-feature-overlap`](sae-feature-overlap.md), and ask three questions:

1. **Top-aligned ∩ shared.** Of the top-K aligned features, how many are in the shared-feature set? Compared against the chance baseline `K · n_shared / d_sae`.
2. **Spearman ρ between alignment and `|mean-diff|`.** The mean-diff ranks features the way [`sae-feature-overlap`](sae-feature-overlap.md) does; alignment ranks them the way the probe does. If the two rankings correlate strongly, the probe and the overlap analysis "find" the same features.
3. **Subspace-norm fraction.** Project `w` onto the subspace spanned by the shared-feature decoder columns (via QR decomposition for orthogonality) and report the fraction of `‖w‖` captured by that projection. Compared against a permutation null over equally-sized random feature subsets.

The subspace-norm test is the load-bearing one. Top-K agreement is sensitive to the K choice and the chance baseline scales weirdly with `n_shared`; the fraction of probe norm captured by a fixed subspace is a single scalar with a natural interpretation ("how much of what the probe learned lives inside the shared-feature subspace") and a clean permutation null.

## Why this design

- **5-fold stratified CV for AUROC, full-data refit for `w`.** CV gives an honest held-out generalization number; the full-data refit is the most stable direction to project onto SAE columns. Cross-fold direction averaging would smear small-sample artifacts in.
- **Default top-K = 41 matches the typical shared-feature count.** The paper's Llama-3.1-8B L19 row has 41 shared features, so reporting the top-41 aligned makes the chance baseline `41 × 41 / 65 536 ≈ 0.026` features and the agreement headline-friendly. Override with `--top-k-overlap`.
- **Permutation null over random subspaces (default `n_perm = 100`).** For each draw we sample `len(shared)` features uniformly without replacement, compute the subspace-norm fraction against that random subspace, and accumulate a null distribution. The headline p-value is `(#{null ≥ observed} + 1) / (n_perm + 1)`.
- **Reads the `shared_features` list from a saved [`sae-feature-overlap`](sae-feature-overlap.md) JSON.** The whole point is to compare against the exact set the paper reports, so we don't recompute it. Override the source slug with `--overlap-from`.
- **Cosine alignment, not raw dot product.** Unit-normalising both `w` and each `W_dec[:, i]` removes the per-feature norm confound; otherwise features with large decoder norms would always rank near the top regardless of geometric alignment.

## How to run it

```bash
# Default: Llama-3.1-8B L19, reads sae_feature_overlap_llama-3.1-8b-instruct.json
uv run shared-circuits run linear-probe-sae-alignment

# Gemma-2-2B L19 (must have a saved sae-feature-overlap result for that layer)
uv run shared-circuits run linear-probe-sae-alignment \
  --model gemma-2-2b-it --layer 19

# Tighter probe (more folds + bigger permutation null on the subspace-norm test)
uv run shared-circuits run linear-probe-sae-alignment \
  --n-folds 10 --n-perm-subspace 1000

# Match the shared-feature count for the top-K aligned cell
uv run shared-circuits run linear-probe-sae-alignment --top-k-overlap 36
```

The analysis raises `FileNotFoundError` if no saved [`sae-feature-overlap`](sae-feature-overlap.md) result exists for the requested `(model, layer)`.

Output: `experiments/results/linear_probe_sae_alignment_<model_slug>.json`. The two probes get parallel blocks under `syc_probe` and `lie_probe`, each containing:

| Field | Meaning |
|---|---|
| `auroc_cv` | Mean held-out AUROC across the 5 stratified folds |
| `overlap_stats` | `{overlap, top_k_aligned, n_shared, d_sae, chance_overlap, ratio_vs_chance}` |
| `top_aligned` | Top-K aligned feature indices (descending `|cosine|`) |
| `spearman_align_vs_absdiff` | `{rho, p_value}` over the full dictionary |
| `subspace_norm_fraction` | `{shared, null_mean, null_std, null_max, n_perm, p_permutation}` |

## Where it lives in the paper

Appendix § "SAE feature overlap: controls and robustness (Llama-3.1-8B, layer 19)" (`app:sae-controls`), as the linear-probe-alignment paragraph. Headline numbers on Llama-3.1-8B L19: syc probe 5-fold CV AUROC `0.949`, lie probe `0.879`. Spearman ρ between probe-alignment and `|mean-diff|` across all 65 536 features: syc `ρ = 0.76` (`p` effectively 0); lie `ρ = 0.69`. Subspace-norm fraction captured by the 41-feature shared subspace: syc `24%` vs permutation null mean `13.5%` (`p = 0.01`); lie `23%` vs null `11.7%` (`p = 0.01`). Reading: "the linear probes independently find the same SAE features that the overlap analysis identifies".

## Source

`src/shared_circuits/analyses/linear_probe_sae_alignment.py` (~280 lines). Reads `sae_feature_overlap_<model>.json` as upstream for the `shared_features` list; uses `extraction.extract_residual_stream` (not the SAE encoder — the probe trains directly in residual space) and `data/sae_features.load_sae_for_model` for `W_dec`. The analysis requires `sae.w_dec` to be present (it is for both Gemma-Scope and Goodfire — the loader in `data/sae_features.py` keeps the decoder columns precisely so this analysis can run). Sibling probe analysis at residual-stream level: [`probe-transfer`](probe-transfer.md) (which fits a probe on syc and tests transfer to lie, instead of projecting onto SAE columns). Consumed only by manual paper-table generation.
