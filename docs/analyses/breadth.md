# `breadth`

> A one-shot panel runner: head overlap and behavioral steering on a single model. The lightweight way to extend coverage without rerunning the full causal suite.

This analysis exists because the full causal suite (mean ablation + projection ablation + path patching + activation patching + …) is expensive, and we want to cheaply confirm that a *new* model belongs to the same shared-circuit phenomenon class as the panel models. `breadth` does the two cheapest, most diagnostic checks — head overlap and behavioral steering — in a single run, on a single model, with one CLI invocation.

## The mech-interp idea

Two probes go into the breadth panel:

1. **Head overlap.** Compute per-head importance for sycophancy and for factual lying using the write-norm DLA proxy from [`circuit-overlap`](circuit-overlap.md), then ask whether the top-K sets overlap above chance via a permutation null. Same content split as the headline analysis (sycophancy on TriviaQA pairs `[0:100)`; lying on `[100:300)`), so the overlap is on disjoint factual surface forms. Pearson and Spearman over the full head population are also reported.

2. **Behavioral steering.** Compute the sycophancy direction `d_syc = mean(resid_wrong) − mean(resid_correct)` at each of a few candidate layers (defaults: 50%, 60%, 70%, 80% of `n_layers`), unit-normalize it, then add `α · d_syc` to the residual at the last token at one chosen layer and measure the new sycophancy rate. A random-direction control with the same `α` is run alongside — the delta `real − random` is the steering effect attributable to the direction, not the magnitude.

Together, the two probes give you (a) "yes, the shared-head ranking transfers to this model" and (b) "yes, the sycophancy direction has a dose-response on this model" — enough to claim panel membership without the heavy causal suite. The full panel breadth claim ("12 models, shared fraction 40–87%, Spearman 0.80–0.97") is built up by running this on each new candidate model and folding the results into Table 1.

This is the grandparent of the per-paradigm steering analysis: [`steering`](steering.md) is the lighter purpose-built version (one model, one layer, full positive + negative dose-response), used downstream for the §3.5 alignment-probe story. `breadth` runs the steering sweep at multiple candidate layers and only on the negative side by default.

## Why this design

- **Default alphas `(0, −25, −50, −100, −200)`.** Negative-only by design: at the breadth probe stage we want to see the dose-response *suppression* of sycophancy under increasing negative steering, which is the more interpretable and ceiling-free side. Override with `--alphas` to add positive arms (the [`steering`](steering.md) analysis defaults to both arms).
- **Layer-fraction defaults `(0.5, 0.6, 0.7, 0.8)`.** Mid-to-late residual layers are where the syc direction is cleanest. The set spans the band, so you can see at which layer the dose-response is strongest.
- **Random-direction control with same alpha.** Steering by *any* unit-norm direction with `|α| = 200` will perturb the model. Subtracting the random rate isolates the direction-specific contribution.
- **Disjoint TriviaQA splits.** Direction-extraction prompts (first 50 pairs), test prompts for steering (pairs `[200:250)`), syc/lie head-overlap prompts (`[:100]` and `[100:300]` respectively). No prompt is reused across roles.
- **`bfloat16` direction tensor.** The hooks add `α · d` to the residual on every step; matching the residual dtype avoids casts in the hot loop.

## How to run it

```bash
# A new model — full breadth panel
uv run shared-circuits run breadth --model microsoft/phi-4

# Two-GPU run for 70B
uv run shared-circuits run breadth \
  --model meta-llama/Llama-3.1-70B-Instruct --n-devices 2

# Custom alpha sweep with positive arms
uv run shared-circuits run breadth \
  --model gemma-2-2b-it --alphas "0,-50,-100,-200,50,100,200"

# Smaller / faster sanity run
uv run shared-circuits run breadth \
  --model gemma-2-2b-it --n-pairs 200 --steer-prompts 30
```

Output: `experiments/results/breadth_<model_slug>.json`. Key fields:

| Field | Meaning |
|---|---|
| `head_overlap.syc_grid`, `head_overlap.lie_grid` | Per-head DLA importance arrays (consumed downstream by [`norm-matched`](norm-matched.md), [`mlp-ablation`](mlp-ablation.md) tugofwar mode, etc.) |
| `head_overlap.stats.{k,pearson,spearman,top_k_overlap,top_k_chance,overlap_ratio,p_value}` | Permutation-null overlap stats |
| `baseline_sycophancy` | Sycophancy rate at `α = 0` |
| `steering.candidates` | Layer indices probed |
| `steering.layers[L].resid_norm` | Mean residual norm at layer `L` (sanity check on alpha scale) |
| `steering.layers[L].alphas[i].{alpha,real,random,delta}` | Per-α rates and the direction-vs-random delta |

## Where it lives in the paper

The breadth panel anchors §3.1 / **Table 1**. Each new model added to the 12-model panel is run through this analysis first; the head-overlap row goes into Table 1, the steering numbers feed the supporting evidence. The Qwen2.5-72B run was the basis for the targeted [`mlp-mediation`](mlp-mediation.md) follow-up reported in Appendix `mediation`. The Makefile has wrapper recipes: `make qwen72b-pipeline` runs `breadth` + `mlp-mediation`; `make mixtral-all` runs `breadth` + `circuit-overlap` + `path-patching` + `norm-matched` on Mixtral-8x7B.

## Source

`src/shared_circuits/analyses/breadth.py` (~280 lines). Self-contained: it computes the DLA grids itself (no [`circuit-overlap`](circuit-overlap.md) dependency), so it can run on a brand-new model without prior artifacts. Saves both the grids and the steering sweep so downstream grid-consumers ([`norm-matched`](norm-matched.md), [`mlp-ablation`](mlp-ablation.md) tugofwar) can read them via `--shared-heads-from breadth`. The lighter sibling for the steering half is [`steering`](steering.md).
