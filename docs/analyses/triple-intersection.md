# `triple-intersection`

> Sycophancy ∩ lying ∩ opinion top-K heads — same parking lot, three different cars?

If opinions reuse the same head positions as sycophancy and lying, the three top-K head sets should overlap above what shuffling labels would produce. The cross-task two-way overlap is already established by [`circuit-overlap`](circuit-overlap.md); the triple intersection is the position-overlap evidence for §3.6's "opinions reuse the head positions but not the full circuit" claim.

<p align="center">
  <img src="../img/opinion_generalization.png" width="600" alt="Triple-intersection top-K head overlap (left panel) and direction cosine (right panel). The same positions get reused; the directions are orthogonal.">
</p>

## The mech-interp idea

For any task `t`, [`circuit-overlap`](circuit-overlap.md) and [`opinion-circuit-transfer`](opinion-circuit-transfer.md) produce a per-head importance grid (one scalar per `(layer, head)`). Pick the top-K heads from each grid; the question is how many are shared across all three task rankings.

The chance baseline for a triple intersection of size-K sets in a population of `N = n_layers × n_heads` is `K³ / N²`. With the standard `K = ⌈√N⌉` (matching [`circuit-overlap`](circuit-overlap.md)), that's `≈ √N / N = N^(-1/2)` — a fraction of one head expected by chance. The observed overlap is reported both as a ratio against this analytic chance and as a permutation-null p-value: `n_perm = 10_000` shuffles independently permute each task's grid before re-intersecting, and the p-value is the fraction of shuffles where the null overlap meets or exceeds the observed.

The framing is parallel to Merullo et al. (2024) on cross-task component reuse: head-position overlap in itself is a weak claim (any two evaluation tasks reuse 4–7× chance of components, the documented reuse floor), so we report the *ratio* against an explicit triple-intersection null rather than just the count.

## Why this design

- **`K = ⌈√N⌉` for analytic interpretability.** Matching [`circuit-overlap`](circuit-overlap.md) means the analytic chance baseline `K³/N²` shrinks to ~1/√N — the ratios stay readable even as model size grows.
- **Independent permutations per task.** Permuting all three grids independently is the strict null: it rules out "shared overlap because all three rankings cluster in the same layers" without us having to specify a layer-wise model. (The layer-wise null is [`layer-strat-null`](layer-strat-null.md)'s scope, kept separate to avoid coupling the two questions.)
- **`--factual-from breadth` is the default.** The breadth runner persists the full per-head grids that the triple intersection needs. `circuit_overlap` JSON only has the top-15 per task, so reading it would silently truncate to the union of two top-15 sets. If you only have circuit-overlap output, pass `--factual-from circuit_overlap` and the loader will rebuild a sparse grid from the persisted top-K entries.
- **Single output, multi-model.** Unlike most analyses in this repo, this one writes one combined JSON (`triple_intersection_perm_all_models.json`) rather than per-model files, because the only thing you do with it is plot the bar chart in figure 3a.

## How to run it

```bash
# The five-model panel from §3.6 (default; reads breadth + opinion_circuit_transfer JSONs)
uv run shared-circuits run triple-intersection

# Single-model spot-check
uv run shared-circuits run triple-intersection --models gemma-2-2b-it

# If you only have circuit-overlap (no full grid persisted), tell the loader to rebuild
uv run shared-circuits run triple-intersection --factual-from circuit_overlap

# Tighter null (10x more permutations)
uv run shared-circuits run triple-intersection --n-permutations 100000
```

Output: `experiments/results/triple_intersection_perm_all_models.json`. Key fields under `by_model.<name>`:

| Field | Meaning |
|---|---|
| `k` | Top-K threshold (`⌈√(n_layers × n_heads)⌉`) |
| `actual_triple_overlap` | Observed `\|op_top ∩ syc_top ∩ lie_top\|` |
| `analytic_chance` | `K³/N²` — expected count under independence |
| `ratio_vs_chance` | `actual / analytic_chance` (the headline ratio in Figure 3a) |
| `permutation_p_value` | `(≥-count + 1) / (n_perm + 1)` |
| `permutation_mean` / `permutation_p95` | Null distribution summaries |

## Where it lives in the paper

§3.6, **Figure 3a** (`fig:opinion`). Result: triple-intersection overlap is significant on five models at **51–1,755× chance**. Per-model ratios are plotted in `opinion_generalization.png` panel (a). The orthogonality counterpart (the directions, not the positions) lives in [`opinion-causal`](opinion-causal.md)'s `boundary` mode: `|cos| < 0.14` for opinion vs factual-correctness, far below the `0.43–0.81` for syc vs lie that [`direction-analysis`](direction-analysis.md) reports.

## Source

`src/shared_circuits/analyses/triple_intersection.py` (~155 lines). Reads two upstream JSONs: `breadth` (or `circuit_overlap`) for the syc + lie grids, and [`opinion-circuit-transfer`](opinion-circuit-transfer.md) for the opinion grid. Feeds the head-set into [`opinion-causal`](opinion-causal.md), which loads the same intersection (via `circuit_overlap` + `opinion_circuit_transfer` shared-heads at K=15) for the head-zeroing test.
