# `steering`

> Add `α · d_syc` to the residual stream and watch the agreement rate move. How dose-dependent is sycophancy?

This is the simplest possible behavioral causal handle on the sycophancy direction: extract `d_syc` (the residual-stream mean-difference vector between wrong-opinion and right-opinion prompts), pick a layer, and at inference time inject `α · d_syc` into the residual at that layer's output. Sweep `α` from negative (away from sycophancy) through 0 (untouched baseline) through positive (toward sycophancy) and read off the agreement rate on held-out wrong-opinion prompts. If the curve is monotone — agreement falls with negative `α` and rises with positive `α` — the direction is causal for the behavior.

## The mech-interp idea

A **steering vector** is a fixed direction in residual space added to the activations at a chosen layer. The classic refusal-direction work (Arditi et al. 2024) and the broader representation-engineering line (Zou et al. 2023) show that small numbers of these directions can flip a model's behavior reliably. Here we use the simplest construction: `d_syc = mean(resid_wrong) − mean(resid_correct)` at the chosen layer, unit-normalised so `α` is interpretable as a magnitude in `‖d_syc‖` units. That direction is the same one [`probe-transfer`](probe-transfer.md) and [`direction-analysis`](direction-analysis.md) use; here we don't compare it against anything, we just sweep its dose.

The intervention is applied **only at the last non-pad token** at the chosen layer's `hook_resid_post`. This matches the legacy run script and the convention used for last-token logit readouts: sycophancy is measured at the position the next-token logits are read from, so steering anywhere else would dilute the effect. The sweep covers `α ∈ {0, ±25, ±50, ±100, ±200}` by default — symmetric around zero and roughly logarithmically spaced. Negative `α` steers *away* from sycophancy and should drop the agreement rate; positive `α` steers *toward* and should raise it. The curve through these points is the behavioral dose-response.

The agreement rate is measured the same way as in [`circuit-overlap`](circuit-overlap.md)'s baseline pass: on wrong-opinion prompts, did the model's last-token logits put more mass on an "agree" token (yes / true / correct) than on a "disagree" token (no / false / incorrect)? `measure_agreement_rate` handles the per-model token-id resolution.

## Why this design

- **One direction, one layer, last token only.** This is the lighter purpose-built version of the steering panel that ships inside [`breadth`](breadth.md) (which also runs head overlap and a layer-fraction sweep). Use this one when you already know the layer you want and just need the curve. Use `breadth` when you need the full per-model panel.
- **Layer at `0.6 × n_layers` by default.** Picked from the layer-wise direction-cosine peak in [`direction-analysis`](direction-analysis.md) — the layer where `d_syc` and `d_lie` align most tightly. Override with `--layer-frac` (relative) or `--layer` (absolute).
- **Direction estimated on `dir_prompts` (default 100), tested on `test_prompts` (default 100).** The defaults reuse the same `[0, n_pairs)` slice for direction and test, so the direction is fit on the same prompts behavior is measured on. That's *not* a cross-validated estimate — the headline question is "is this direction causal?" not "does this direction generalise?", and the cross-validated question is the job of [`probe-transfer`](probe-transfer.md). Override with disjoint `--dir-prompts` and `--test-prompts` slices via `--n-pairs` if you want CV.
- **Per-batch hook closure.** The hook needs per-batch `seq_lens` to find the last non-pad token, so we rebuild the closure each iteration rather than carrying stale `seq_lens` across batches. Without this, a length-mismatched batch would write `α · d_syc` to a padding position and have no behavioral effect.
- **Symmetric default sweep.** The `0, ±25, ±50, ±100, ±200` grid lets readers eyeball linearity (does doubling `|α|` double the rate change?) and saturation (where does the curve plateau?).
- **`α` is an integer.** The legacy script accepted ints only; the type is preserved for bit-for-bit reproducibility. Pass `--alphas 0,-12,12` if you want the same coarseness at a different magnitude.

## How to run it

```bash
# Default sweep on a single model (last-token, layer = 0.6L)
uv run shared-circuits run steering --model gemma-2-2b-it

# Custom α grid (denser around zero)
uv run shared-circuits run steering \
  --model gemma-2-2b-it --alphas 0,-10,-25,-50,10,25,50

# Steer at a specific absolute layer rather than the 0.6L fraction
uv run shared-circuits run steering \
  --model meta-llama/Llama-3.1-8B-Instruct --layer 19

# More test prompts for a tighter rate estimate
uv run shared-circuits run steering \
  --model Qwen/Qwen3-8B --n-pairs 400 --test-prompts 200

# 70B with pipeline parallelism
uv run shared-circuits run steering \
  --model meta-llama/Llama-3.3-70B-Instruct --n-devices 2
```

Output: `experiments/results/steering_<model_slug>.json`. Top-level fields:

| Field | Meaning |
|---|---|
| `steer_layer`, `n_layers` | Resolved layer index + total layer count |
| `dose_response` | List of `{alpha, rate, n}` rows, one per α |
| `config` | `{n_pairs, test_prompts, dir_prompts, alphas, layer_frac, batch}` |

## Where it lives in the paper

Steering shows up across §3 as one of the "directional analyses" (the §3.4 tugof-war framing and the breadth-panel results in §3.1's Table 1 footnote pull from [`breadth`](breadth.md)'s steering block, which is the same construction with a layer-fraction sweep stacked on top). Per `experiments.tex`'s scope paragraph: "directional analyses (probe transfer, steering, per-head cosine) were run on six models where we extracted a stable mean-difference direction". Headline reading: monotone dose-response across the swept α range, with the negative-α arm bringing agreement below baseline on every tested model and the positive-α arm pushing it toward ceiling.

## Source

`src/shared_circuits/analyses/steering.py` (~190 lines). Uses `extraction.extract_residual_stream` for the direction estimate and `extraction.measure_agreement_rate` for the α = 0 baseline; the steered passes go through `model.run_with_hooks` directly with a per-batch `_make_steer_hook` closure. The full-panel sibling that runs head-overlap + steering + multi-layer-fraction sweep in one shot is [`breadth`](breadth.md). The direction itself is the same one used by [`probe-transfer`](probe-transfer.md) (probe variant) and [`direction-analysis`](direction-analysis.md) (cosine variant). Consumed only by manual paper-figure generation.
