# `projection-ablation`

> Subtract the sycophancy direction from the residual stream at every position. Does the model stop being sycophantic?

This is the directional companion to [`head-zeroing`](head-zeroing.md). Instead of removing entire attention heads, we identify the single residual-stream direction that distinguishes "user is wrong" from "user is right" prompts, then project that direction *out* of the residual at a chosen layer on every forward pass. If the shared circuit writes its sycophancy signal along this direction, removing it should flip behavior; the random-direction control rules out generic capability damage.

<p align="center">
  <img src="../img/causal_convergence.png" width="600" alt="Three causal interventions on the shared-head set converge on sufficiency from 2B to 70B: mean-ablation, projection ablation, and activation patching all exceed matched random-head controls.">
</p>

## The mech-interp idea

Modern transformer interpretability has converged on a recurring trick (Arditi et al. 2024, Marks & Tegmark 2024, Zou et al. 2023): for a binary task, the "task feature" often lives along a single linear direction in the residual stream, computable as a *mean-difference* of activations between positive and negative examples.

For sycophancy, that direction is

```
d_syc = mean(resid on user-wrong prompts) − mean(resid on user-right prompts)
```

extracted at the chosen layer's `hook_resid_post`. Once we have a unit-normalized `d_syc`, we install a forward hook that subtracts the projection at every position:

```
x ← x − ((x · d_syc) / ||d_syc||²) d_syc
```

i.e., delete the component of the residual along `d_syc` while leaving the rest of the activation intact. Then we measure the sycophancy rate on held-out wrong-opinion prompts.

The control is a random unit vector with the same magnitude. If projection ablation reduces sycophancy and random projection doesn't, the direction is *necessary* for the behavior — there's a single linear axis along which the shared heads write the signal. If both projections damage behavior equally, we're seeing generic capability loss rather than direction-specific intervention.

The `verdict` field labels the result: `DIRECTION_NECESSARY` when projection ablation reduces syc rate by at least 5pp more than the random-direction control at some tested layer (i.e., the projection-vs-random margin is below −0.05), `PARTIAL_DIRECTION` for any negative margin, `NO_DIRECTION_EFFECT` otherwise. Note that the paper reports the opposite-sign effect on low-baseline models (Gemma-2-27B's `10.5% → 100%` and Llama-3.3-70B's `+27pp`) — when the baseline is already near zero, ablating the "deference" direction *raises* sycophantic agreement rather than lowering it. The verdict tag's sign convention is set up for the high-baseline case; either sign with a large enough margin is causally interesting.

## Why this design

- **Mean-difference direction extraction matches the rest of the paper.** Same `d_syc` construction as the [`probe-transfer`](probe-transfer.md) and [`reverse-projection`](reverse-projection.md) analyses, and the same construction Marks & Tegmark and Zou et al. use for "truth" directions. This is what makes "the direction sycophancy and lying share" a meaningful object across analyses.
- **Layer sweep at fractional depths, not absolute layer indices.** Default is `(0.5, 0.6, 0.7, 0.8) * n_layers`, so the same flag covers Gemma-2-2B (26 layers) and Llama-3.3-70B (80 layers). Override with `--layer-fracs 0.4,0.5,0.6` for a custom sweep, or pin `--layer 14` for a single layer.
- **`--qwen3-layer-sweep` preset.** Qwen3 places its task-relevant layers earlier than Gemma-2, so the legacy `run_qwen3_projection_ablation.py` script used `(0.45, 0.55, 0.65, 0.75)` instead. The flag invokes that preset to match the paper's Qwen3 row.
- **Random-direction control with the same magnitude.** Generated from `np.random.RandomState(seed)`, normalized to unit norm. This rules out "any directional perturbation breaks behavior" and forces the effect to be `d_syc`-specific.
- **Direction extracted on *separate* prompts from the test set.** First `dir_prompts` (default 50) for the direction; next `test_prompts` (default 200) for the rate measurement. No leakage between extraction and evaluation.
- **Paired bootstrap CIs over prompt indices.** Same scheme as [`head-zeroing`](head-zeroing.md): per-prompt indicators are bootstrapped in pairs, tightening the CIs by exploiting the within-prompt correlation between baseline and ablated outcomes.

## How to run it

```bash
# Default 4-layer sweep on Gemma-2-2B
uv run shared-circuits run projection-ablation --model gemma-2-2b-it

# Full 27B Gemma run (the headline 10.5% -> 100% flip)
uv run shared-circuits run projection-ablation \
  --model google/gemma-2-27b-it --n-devices 2

# Qwen3 layer-fraction preset
uv run shared-circuits run projection-ablation \
  --model Qwen/Qwen3-8B --qwen3-layer-sweep

# Single-layer run, custom test count
uv run shared-circuits run projection-ablation \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --layer 56 --test-prompts 100 --n-devices 4

# Custom layer fractions
uv run shared-circuits run projection-ablation \
  --model gemma-2-2b-it --layer-fracs 0.3,0.5,0.7,0.9
```

Output: `experiments/results/projection_ablation_<model>.json`. Key fields:

| Field | Meaning |
|---|---|
| `verdict` | `DIRECTION_NECESSARY` / `PARTIAL_DIRECTION` / `NO_DIRECTION_EFFECT` |
| `baseline_syc_rate` | No-ablation sycophancy rate |
| `layers.<L>.projection.{syc_rate, syc_delta, syc_ci}` | Per-layer effect of `d_syc` projection |
| `layers.<L>.random_projection.{...}` | Per-layer effect of random-direction projection |
| `layers.<L>.margin` | `proj_delta − rand_delta` — the direction-specific effect at that layer |
| `layers.<L>.resid_norm` | Mean residual norm on corrupt prompts (sanity check on direction magnitude) |

## Where it lives in the paper

§3.4. Headline numbers:

- **Gemma-2-27B**: sycophantic agreement flips `10.5% → 100%` after `d_syc` projection. The figure-of-merit row in the §3.4 sufficiency-and-necessity convergence claim.
- **Llama-3.3-70B**: `+27pp` projection-ablation effect. Cited in the RLHF natural experiment (§3.5, Table 4) where the *same* projection-ablation effect *grows* from `+10.5pp` (Llama-3.1-70B) to `+27pp` after Meta's 3.1→3.3 RLHF refresh — the substrate becomes more causally accessible while behavior drops 10×.
- Combined with [`head-zeroing`](head-zeroing.md) (mean-ablation, ≤ 7B regime) and [`attribution-patching`](attribution-patching.md) / [`activation-patching`](activation-patching.md), the three converge on sufficiency from 2B to 70B (Figure `causal_convergence.png`). At 70B mean-ablation alone fails under distributed-redundancy (McGrath et al. 2023); projection ablation and path patching carry the causal claim at scale.

## Source

`src/shared_circuits/analyses/projection_ablation.py` (~270 lines). Self-contained — does not read any other analysis's JSON; it extracts the direction from prompts directly. Companion to [`head-zeroing`](head-zeroing.md), [`reverse-projection`](reverse-projection.md), [`steering`](steering.md), and the [`activation-patching`](activation-patching.md) shared-set patcher. Output JSON is consumed only by manual figure/table generation.
