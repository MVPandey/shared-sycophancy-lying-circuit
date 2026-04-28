# `layer-strat-null`

> Could the head-overlap result be a layer-clustering artifact? Stricter null says no.

This is the harder version of the permutation null in [`circuit-overlap`](circuit-overlap.md). The unstratified permutation reshuffles head importances across the entire grid; that's permissive if some layers have lots of important heads on both tasks, because then a coincidental layer-clustered ranking could pile up overlap without any genuine head-level alignment. This analysis shuffles head labels *within each layer*, preserving per-layer marginals exactly. If the overlap survives, the layer-clustering escape hatch is closed.

## The mech-interp idea

The unstratified null treats every head position as exchangeable. Real attention patterns aren't that flat — late layers tend to be more task-relevant than early ones, so important heads cluster at certain depths on both tasks. That clustering inflates the unstratified null's variance and can manufacture significance.

The fix is a stratified permutation: for each model, shuffle the lying-task head importances independently within each layer, leaving every layer's distribution of values unchanged. The overlap statistic is then computed against this restricted null, which can only redistribute head identity *within a depth*. Any surviving overlap has to come from head-level position agreement, not from "both tasks care more about layer 12".

Concretely: for each of `n_perm = 10000` draws we permute `lie_grid[layer, :]` separately for every layer, recompute the top-K set on the flattened grid, and count how often the null overlap matches or exceeds the observed overlap. The reported p-value is `(ge + 1) / (n_perm + 1)`, so the smallest reportable p is ≈10⁻⁴ at the default permutation count.

## Why this design

- **Preserves per-layer marginals exactly.** A standard within-layer shuffle doesn't change the per-layer mean, max, or distribution — only which head positions hold which scores. The only thing being tested is head-position agreement at fixed depth.
- **Reads grids from a previous run, doesn't recompute them.** This analysis is pure stats. It loads `syc_grid` and `lie_grid` from a saved [`breadth`](breadth.md) (or [`circuit-overlap`](circuit-overlap.md)) JSON, so the same rankings underwrite the unstratified and stratified p-values. `--grids-from breadth` is the default; switch to `circuit_overlap` if that's where your grids live.
- **Different seed for the stratified pass.** The unstratified and stratified passes use seeds offset by 1, matching the legacy script. Different RNG draws across the two nulls keep their estimates from being correlated, which would otherwise tighten the joint variance artificially.
- **K defaults to `⌈√N⌉` to match the rest of the paper.** Override with `--k` if you want to sweep the threshold the way [`sae-k-sensitivity`](sae-k-sensitivity.md) does for the K-sweep argument.

## How to run it

```bash
# Default 8-model panel, reads from breadth_<model>.json
uv run shared-circuits run layer-strat-null

# A single model, fewer permutations (fast smoke test)
uv run shared-circuits run layer-strat-null \
  --models gemma-2-2b-it --n-permutations 1000

# Read grids from circuit_overlap rather than breadth
uv run shared-circuits run layer-strat-null \
  --models meta-llama/Llama-3.3-70B-Instruct \
  --grids-from circuit_overlap

# Sweep K to verify the threshold isn't doing the work
uv run shared-circuits run layer-strat-null --k 30
```

Output: `experiments/results/layer_stratified_null_all_models.json`. Each model row contains:

| Field | Meaning |
|---|---|
| `actual_overlap` | Top-K intersection size on the real grids |
| `chance_overlap` | `K² / N`, the analytic expectation under independence |
| `ratio_vs_chance` | `actual / chance` — the headline effect size |
| `p_hypergeometric` | Closed-form p-value (no clustering correction) |
| `p_permutation_unstratified` | Permutation null shuffling all heads (matches the prior in `circuit-overlap`) |
| `p_permutation_layer_stratified` | The stricter within-layer null — the headline of this analysis |

## Where it lives in the paper

Appendix § "Layer-stratified permutation null", Table `tab:layerstrat`. Result: all 8 tested models survive at `p < 10⁻⁴` (capped by `n_perm = 10000`), spanning Gemma-2-2B/9B/27B, Qwen2.5-1.5B/72B, Qwen3-8B, and both Llama-3.1-70B and Llama-3.3-70B. Phi-4 was not included; its unstratified hypergeometric `p < 10⁻¹⁰` from Table 1 carries that row instead. The §3.1 main-body claim cites this appendix when it says "the overlap remains significant under a layer-stratified permutation null".

## Source

`src/shared_circuits/analyses/layer_strat_null.py` (~180 lines). Pure post-hoc statistics: it does not run a model. Reads `breadth_<slug>.json` (or `circuit_overlap_<slug>.json`) for each model. Consumed only by manual paper-table generation; nothing downstream reads its JSON.
