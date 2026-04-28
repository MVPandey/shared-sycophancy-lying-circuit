# `norm-matched`

> If you knock out random heads with the same write-magnitude as the shared heads, do you get the same effect? (No.)

The headline behavioral effect — zero the shared heads, watch sycophancy collapse — has an obvious confound. Maybe the shared heads just happen to have big `W_O` (output projection) Frobenius norms, and zeroing *any* heads with norms that big would do the same thing. This analysis kills that explanation.

## The mech-interp idea

Each attention head writes its output into the residual stream via `W_O`, the per-head output projection matrix. Heads with large `||W_O||_F` have big-magnitude writes regardless of *what* they write — so a count-matched random control (which is what [`activation-patching`](activation-patching.md) and [`head-zeroing`](head-zeroing.md) use as a default negative control) doesn't rule out write-magnitude as the driver. A heavyweight head zeroed on any task will look impactful.

The fix is a **norm-matched** control: pick random non-shared heads whose `W_O` norms are as close as possible to the shared set, head-for-head, then re-run head-zeroing. If the shared-head effect on sycophancy still exceeds the norm-matched control, write-magnitude isn't the explanation — the heads' computational role is.

We use a greedy nearest-neighbor matching: for each shared head, pick the unused non-shared head with the closest `W_O` Frobenius norm, sample without replacement. The pool is shuffled with a fixed seed before the greedy pass to prevent layer-order bias when several candidates tie.

Three head sets at the shared-set's size are reported: the **shared** set itself (read from [`circuit-overlap`](circuit-overlap.md)), the **norm-matched** set (greedy nearest-neighbor on `W_O` norm), and a **uniform random** set (count-matched, no norm constraint). For each, we zero the heads, measure sycophancy rate and last-token logit difference, and report the delta vs baseline. The headline is the **margin** = shared delta − norm-matched delta.

## Why this design

- **Greedy nearest-neighbor, not optimal-transport.** We tested optimal-transport matching in the legacy code and the difference was negligible at the per-head level, with much higher implementation complexity. Greedy is fine because the matching only needs to be *approximately* norm-equal — the shared/norm-matched contrast is order-of-magnitude.
- **Two readouts: rate and logit-diff.** Rate flips are the headline behavioral claim, but rate is binary and ceiling-bound; the logit-diff is continuous and survives at the ceiling. On Phi-4, rate doesn't flip at all but the logit-diff shifts in *opposite directions* between shared (`+0.99`) and norm-matched (`−0.56`) — also rules out write-magnitude as a driver.
- **Verdict requires both readouts to clear the margin.** `SPECIFIC_TO_SHARED` if both `rate` and `logit_diff` margins exceed 0.05; `PARTIAL_SPECIFICITY` if one; `NORM_CONFOUND` if neither.
- **Six models from five families.** Covers the architectural diversity that matters: dense Gemma/Mistral/Qwen/Llama, the sparse-MoE Mixtral-8x7B (first MoE validation in the paper), and the architecturally different Phi-4. Mixtral specifically rules out "MoE routing dissolves the effect"; Phi-4 specifically rules out write-magnitude even when binary rate doesn't flip.

## How to run it

```bash
# The Llama-3.3-70B run that anchors the 70B causal claim
uv run shared-circuits run norm-matched \
  --model meta-llama/Llama-3.3-70B-Instruct --n-devices 2

# Mixtral (first MoE validation)
uv run shared-circuits run norm-matched \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 --n-devices 2

# Smaller batch / smaller prompt count for a quick sanity run
uv run shared-circuits run norm-matched \
  --model gemma-2-2b-it --n-prompts 50 --batch 8

# Different shared-head source (e.g. from a breadth run)
uv run shared-circuits run norm-matched \
  --model microsoft/phi-4 --shared-heads-from breadth
```

Output: `experiments/results/norm_matched_<model_slug>.json`. Key fields:

| Field | Meaning |
|---|---|
| `verdict` | `SPECIFIC_TO_SHARED` / `PARTIAL_SPECIFICITY` / `NORM_CONFOUND` |
| `wo_norms.{shared,norm_matched,random}_{mean,std}` | Norm-matching diagnostic: `shared_mean ≈ norm_matched_mean` is the sanity check |
| `baseline.{rate,logit_diff}` | No-ablation reference |
| `shared.{rate,logit_diff}`, `norm_matched.{rate,logit_diff}`, `random.{rate,logit_diff}` | Per-condition behavior |
| `delta_shared`, `delta_norm_matched`, `delta_random` | Per-condition deltas vs baseline |
| `margin_shared_vs_norm_matched.{rate,logit_diff}` | The headline number |
| `shared_heads`, `norm_matched_heads`, `random_heads` | The three head sets |

## Where it lives in the paper

Appendix `normmatch`, **Table `tab:normmatch`**. The six-model panel produces:

| Model | Margin (logit-diff) | Multiple |
|---|---|---|
| Gemma-2-2B (n=13) | +3.30 | 6.4× norm-matched |
| Mistral-7B (n=24) | +0.15 | 1.5× |
| Qwen3-8B (n=25) | −1.62 | 117× |
| Mixtral-8x7B (n=23) | +5.49 | 2.8× (first MoE) |
| Phi-4 (n=16) | +1.55 | opposite sign |
| Llama-3.3-70B (n=51) | +4.44 | 27× |

The §3.4 main-text reference is "A write-norm-matched random-head control rules out the write-magnitude confound (Appendix~\ref{app:normmatch})."

## Source

`src/shared_circuits/analyses/norm_matched.py` (~290 lines). Reads `syc_grid` and `lie_grid` from a saved [`circuit-overlap`](circuit-overlap.md) (or [`breadth`](breadth.md)) JSON, rebuilds the shared head list with `K = ⌈√N⌉`. Sibling controls: [`head-zeroing`](head-zeroing.md) (the analysis whose count-matched control this strengthens) and [`activation-patching`](activation-patching.md) (which uses count-matched random heads at scale). Downstream analyses cite this for the write-magnitude null.
