# `mlp-ablation`

> Three MLP experiments share a CLI: which MLP layers move sycophancy, did we just break the model, and do the moves line up with the shared heads?

The shared circuit runs through attention. But the residual stream is shared, so the MLPs sitting in the same layers as (or just upstream of) the shared heads can also shift behavior. This analysis bundles the three quick MLP probes you'd want before drawing any "MLPs do X" conclusion: an effect map, a sanity check on the effect map, and a correlation between the effect map and the shared-head importance map.

## The mech-interp idea

Three sub-modes share the same analysis surface:

- **`ablation`** — Zero each MLP layer's output one at a time and record the change in sycophancy rate (`delta = ablated_rate − baseline_rate`). The default target set is mid-to-late layers (`mid - 4` through `n_layers - 2`); this is the legacy default and matches where shared heads cluster on the panel models. The output is a per-layer effect map.

- **`disruption`** — The control. Zeroing an MLP can break the model in a generic way: text gets garbled, perplexity explodes, and sycophancy rate moves for reasons unrelated to circuit-level mediation. To rule that out, we measure perplexity on a fixed set of 20 short, factual, neutral prompts (capital cities, scientific facts, history) under the same MLP ablations. The readout is `ratio = ablated_ppl / baseline_ppl`. A layer with `ratio < threshold` (default 5×) is flagged as a *specific* effect — sycophancy-relevant rather than broken-model. Above threshold, you've just disrupted generation and the `ablation`-mode delta is uninterpretable.

- **`tugofwar`** — Per-layer Spearman correlation between MLP-ablation `|delta_syc_rate|` and the per-layer shared-head DLA importance vector. The hypothesis: layers with strong shared-head presence should also be the layers where MLP ablation matters most. We try three definitions of "shared-head importance per layer":
  - `intersect_topk` — sum of importance of heads at this layer that are in the top-K shared intersection;
  - `sum_min` — `sum_{h} min(syc_score, lie_score)`;
  - `sum_geomean` — `sum_{h} sqrt(syc_score * lie_score)` (clipped non-negative).

  We also compute a non-parametric Mann-Whitney U test on the difference between `|delta|` at shared vs non-shared tested layers (one-sided, "in-shared > out-shared"), and a distance-from-shared-layer Spearman correlation.

`tugofwar` reads only saved JSON — no model is loaded. `ablation` and `disruption` each need a model session.

## Why this design

- **Mid-to-late default target set, not all layers.** Ablating every MLP at very large model scale is wasteful; we already know early MLPs do tokenization-y things. The default candidate set is `{mid−4, mid−2, mid, mid+1, mid+2, mid+3, mid+4, mid+6, mid+8, n_layers−4, n_layers−2}` clipped to the model's depth. Override with `--layers L1,L2,...`.
- **Disruption uses fixed neutral prompts.** Twenty short, factual prompts that the baseline model handles cleanly. The corpus is hard-coded so the ratio is comparable across models; if ablation makes the model produce gibberish, the ratio explodes. The 5× threshold is a heuristic — pick a different one with `--ppl-ratio-threshold`.
- **Three importance variants, not one.** `intersect_topk` is the strict definition matching [`circuit-overlap`](circuit-overlap.md) — it requires a head to be in *both* top-K sets. `sum_min` and `sum_geomean` are softer (penalize disagreement between syc and lie scores without requiring intersection); they catch cases where the strict intersection is small but the per-layer signal is real.
- **Membership and distance tests in addition to Spearman.** Spearman over 11 tested layers is statistically thin. A Mann-Whitney U on shared-vs-non-shared layers is more robust to small-N. The distance test asks the strictly weaker "is `|delta|` higher near shared-head layers" question.
- **`mlp_results_from` slug for tugofwar.** Lets you point at custom `mlp_ablation` results saved under a different name (e.g. extended layer sets, multiple seeds) without rerunning the model.

## How to run it

```bash
# Per-layer effect map on Gemma-2-2B
uv run shared-circuits run mlp-ablation \
  --model gemma-2-2b-it --mode ablation

# Disruption sanity check (run after ablation; same default layers)
uv run shared-circuits run mlp-ablation \
  --model gemma-2-2b-it --mode disruption

# Tug-of-war correlation (no model loaded; reads saved JSONs)
uv run shared-circuits run mlp-ablation \
  --model gemma-2-2b-it --mode tugofwar

# Custom layer set
uv run shared-circuits run mlp-ablation \
  --model meta-llama/Llama-3.3-70B-Instruct --n-devices 2 \
  --mode ablation --layers 50,55,60,65,70

# Tighter disruption threshold
uv run shared-circuits run mlp-ablation \
  --model microsoft/phi-4 --mode disruption --ppl-ratio-threshold 2.0
```

Output filenames depend on mode:

- `ablation` → `mlp_ablation_<model_slug>.json`
- `disruption` → `mlp_disruption_control_<model_slug>.json`
- `tugofwar` → `tugofwar_prediction_<model_slug>.json`

Key fields:

| Mode | Field | Meaning |
|---|---|---|
| `ablation` | `baseline_rate`, `target_layers`, `layer_effects[L].{rate,delta}` | Per-MLP syc-rate effect map |
| `disruption` | `baseline_perplexity`, `layers[L].{perplexity,ratio,specific}` | PPL-on-neutral sanity check |
| `tugofwar` | `tested_layers`, `abs_delta_by_layer`, `by_variant.{intersect_topk,sum_min,sum_geomean}.spearman_{abs,signed}_delta` | Layer-by-layer correlation |
| `tugofwar` | `membership_test.{mean_in,mean_out,u_statistic,p_one_sided}` | MW U on shared vs non-shared layers |
| `tugofwar` | `distance_test.{rho,p_asymptotic,distances}` | Distance-from-shared-layer Spearman |

## Where it lives in the paper

The MLP role analysis is supporting context for §3.4 and §3.5 (the late-MLP "downstream competition" argument: mean-ablation necessity weakens past 7B because MLPs override the shared-head detection signal in distributed-redundant ways). The full mediation grid that uses `tugofwar` outputs is in [`mlp-mediation`](mlp-mediation.md), Appendix `mediation`.

## Source

`src/shared_circuits/analyses/mlp_ablation.py` (~425 lines). Modes share a CLI but produce different output files. `tugofwar` reads `syc_grid` / `lie_grid` from a saved [`circuit-overlap`](circuit-overlap.md) (default; or [`breadth`](breadth.md)) JSON and `layer_effects` from a saved `mlp_ablation` JSON via `--mlp-results-from`. Sibling: [`mlp-mediation`](mlp-mediation.md) (the heavier per-MLP causal version with paired-bootstrap CIs).
