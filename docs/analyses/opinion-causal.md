# `opinion-causal`

> Zero the opinion ∩ syc ∩ lie heads on opinion prompts. Or measure the opinion-vs-factual direction cosine. Either way, opinion isn't a relabeled "truth detector".

This is the §3.6 boundary test. The triple-intersection in [`triple-intersection`](triple-intersection.md) shows opinion reuses the same head positions as syc and lie. This analysis asks the harder follow-up: *what does the circuit do for opinions?* The answer in two modes — the heads behaviorally affect opinion-agreement only weakly and in opposite directions across models (`causal`), and the opinion direction is orthogonal to the factual-correctness direction (`boundary`).

<p align="center">
  <img src="../img/opinion_generalization.png" width="600" alt="Triple-intersection top-K head overlap (left, fig 3a) and direction cosine (right, fig 3b). Same positions, orthogonal subspaces.">
</p>

## The mech-interp idea

The shared circuit is well-supported as a *factual-correctness* mechanism by §3.4 and the SAE feature-overlap result in §3.6. Opinion-agreement reuses the same head *positions* but on prompts with no factual ground truth ("Pineapple belongs on pizza. Am I right?"). Two separable claims are at stake:

1. **Causal (behavioral).** If the heads play the same role for opinions as for facts, zeroing them should move opinion-agreement rate the same way. The `causal` mode runs head-zeroing on the triple-intersection set with a 5-seed random-head control, paired-bootstrap CIs over per-prompt indicators, and reports `Δrate` and `Δlogit_diff`.
2. **Directional (geometric).** If the heads write the same vector for opinions as for facts, the per-layer mean-difference direction `d_opinion = mean(agree) − mean(disagree)` should align with `d_syc` and `d_lie`. The `boundary` mode computes per-layer cosines on a sweep of layers (every 2nd layer + the final), bootstraps a CI on the late-layer mean (the last third of the network), and emits a `SHARED_WITH_OPINION` / `SPECIFIC_TO_FACTUAL` / `INCONCLUSIVE` verdict.

The directional reading frames the substrate the same way Marks & Tegmark (2024) and Zou et al. (2023) frame their truth direction — a residual-stream mean-difference vector. The orthogonality result here is what tells §3.6 that the circuit is doing *factual-correctness detection*, not generic statement evaluation: the writes for opinions land in a subspace that the syc/lie classifier doesn't read.

## Why this design

- **Two modes share the prompt generator.** `generate_opinion_pairs` builds 200 contested-claim agree/disagree pairs (default `_DEFAULT_N_OPINION = 200`). Both modes reuse the same pair set so the `causal` and `boundary` results for one model are about the same prompts — the directional and behavioral claims aren't on different content.
- **Triple intersection is loaded, not recomputed.** `--triple-from circuit_overlap,opinion_circuit_transfer` reads the persisted shared-heads from those two analyses (intersected at top-K = 15 by default). Recomputing would couple this analysis's head set to drift in the upstream computations; loading bounds it.
- **5-seed random-head control.** Matched on count, not on `W_O` norm. The norm-matched control is [`norm-matched`](norm-matched.md)'s job; the opinion test uses random sets to bound how much "any random head zeroing" moves the rate. The reported `margin` is the gap between the shared-heads `Δrate` and the mean random-heads `Δrate`.
- **Paired bootstrap on per-prompt indicators.** `_paired_ci` resamples *prompt indices* with replacement, which is the right correlation structure: the same prompt is scored under both base and treatment, so the CI on `Δrate` accounts for prompt-level noise. `BOOTSTRAP_ITERATIONS` = 2,000 by default.
- **Late-layer pool for the boundary mean.** `late_layers` are the last third of the network. The `_BOUNDARY_SHARED_LO = 0.2` / `_BOUNDARY_SPECIFIC_HI = 0.1` thresholds on the bootstrap CI decide the verdict. `0.14` is the empirical max across all five tested models — well below `0.2`, sometimes near `0.1`.

## How to run it

```bash
# Causal mode (head zeroing) on the default two-model panel
uv run shared-circuits run opinion-causal --mode causal

# Boundary mode (direction cosines) on Gemma-2-2B-IT
uv run shared-circuits run opinion-causal --mode boundary --models gemma-2-2b-it

# Larger random-head control + tighter bootstrap CI
uv run shared-circuits run opinion-causal \
  --models meta-llama/Llama-3.3-70B-Instruct \
  --n-random-seeds 10 --n-boot 5000

# Use a different upstream pair (e.g. legacy breadth instead of circuit_overlap)
uv run shared-circuits run opinion-causal \
  --triple-from breadth,opinion_circuit_transfer
```

Output: `experiments/results/opinion_causal_<model>.json` (causal) or `opinion_boundary_<model>.json` (boundary). Key fields:

| Field | Meaning |
|---|---|
| `mode` | `causal` or `boundary` |
| `n_shared` / `shared_heads` | Triple-intersection set size and the actual `(layer, head)` pairs (causal only) |
| `shared_cis.delta_rate` / `shared_cis.delta_rate_ci` | Paired-bootstrap `Δrate` and 95% CI (causal) |
| `random_summary` | Mean / std / min / max of `Δrate` across the 5 random-seed controls |
| `margin_delta_rate` | `shared_Δ − random_mean_Δ` (the §3.6 sign-inconsistency claim) |
| `by_layer.<L>.opinion_vs_factsyc` / `opinion_vs_lie` / `factsyc_vs_lie` | Per-layer cosines (boundary) |
| `late_opinion_vs_factsyc` / `bootstrap_ci_95` | Late-layer mean cosine + 95% CI |
| `verdict` | Boundary mode: `SHARED_WITH_OPINION` / `SPECIFIC_TO_FACTUAL` / `INCONCLUSIVE` |

## Where it lives in the paper

§3.6 + Appendix `app:null`, **Figure 3b** (`fig:opinion`) for the boundary, `tab:opinion-null` for the causal. Boundary headline: `|cos| < 0.14` for opinion vs factual-correctness on every tested model, vs `0.43–0.81` for syc vs lie. Causal headline: shifts are small and *sign-inconsistent across families* — Gemma-2-2B-IT margin `+0.33` (more agreement under zeroing), Llama-3.3-70B margin `−0.28` (less agreement), Qwen3-8B ceiling-bound at baseline rate `1.00`. The §3.6 conclusion is therefore restricted to the structural claim: opinions reuse positions but write into an orthogonal subspace, with no consistent *behavioral* role for the shared heads in opinion-agreement.

## Source

`src/shared_circuits/analyses/opinion_causal.py` (~350 lines). Reads two upstream JSONs to build the triple-intersection set: `circuit_overlap` for the syc∩lie shared heads (at K=15), and [`opinion-circuit-transfer`](opinion-circuit-transfer.md) for the opinion top-K. Sibling to [`triple-intersection`](triple-intersection.md) (position-only) and [`direction-analysis`](direction-analysis.md) (syc-vs-lie cosines, the right column of `tab:directional`).
