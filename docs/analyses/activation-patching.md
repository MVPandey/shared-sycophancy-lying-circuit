# `activation-patching`

> Splice the entire shared-head set from a clean run into a corrupt run. Does the logit-diff snap back?

This is the gold-standard causal test, scaled up. Per-head clean→corrupt patching (Wang et al. 2023's IOI method, run by [`attribution-patching`](attribution-patching.md) on small models) sweeps every head individually — at 32B and beyond, that's >50 GPU-hours per model and largely redundant once we already have a ranked shared set. This analysis patches the **entire top-K shared set in one shot** and checks whether that single intervention recovers the clean-run logit difference.

<p align="center">
  <img src="../img/causal_convergence.png" width="600" alt="Three causal interventions (mean-ablation, projection ablation, activation patching) all show concordant sufficiency on the shared head set across 2B–70B.">
</p>

## The mech-interp idea

Activation patching, in its IOI form, asks a counterfactual question: if I take a head's *clean-prompt* activation (where the model gets things right) and splice it into the same head on the *corrupt-prompt* run (where the model gets things wrong), does the model's output move toward the clean answer? If yes, that head is causally responsible for the clean-vs-corrupt logit gap. The intervention is the cleanest causal handle in the toolkit because it surgically swaps one component without disturbing everything else.

The standard procedure runs this for every `(layer, head)` pair and produces a per-head causal effect grid. That grid is the gold standard the cheaper write-norm DLA proxy in [`circuit-overlap`](circuit-overlap.md) is approximating. We validate the proxy directly on three models up to 8B in [`attribution-patching`](attribution-patching.md) (write-norm and per-head-patch grids correlate at `r = 0.41`–`0.61`).

At 32B+, the per-head sweep becomes infeasible. Instead, we use the ranked shared-head set (read from a saved [`circuit-overlap`](circuit-overlap.md) JSON) and patch all of them simultaneously — clean activations from the right-answer prompt go into the corrupt run on every shared head at once. The readout is the resulting last-token agree-vs-disagree logit shift, paired-bootstrapped over 20 prompt pairs. A count-matched random head set (drawn uniformly from the full grid) provides the negative control.

## Why this design

- **Set patching, not per-head.** A single forward pass per pair instead of `n_layers × n_heads` of them. The trade-off is loss of per-head causal resolution; we accept it because the per-head grid is already validated below 8B and because the shared *set* is the unit of interest at scale.
- **Last-token z-cache splicing.** Cached activations are clamped to the corrupt prompt's actual length when the clean prompt is longer (`patched_pos = min(pos, z.shape[1] - 1)`); without this, you get out-of-range writes on length-mismatched pairs.
- **Random negative control matches *count*, not norm.** The norm-matched control is its own analysis — see [`norm-matched`](norm-matched.md). Here we just want to know: does intervening on the same number of arbitrary heads do the same thing? If yes, the shared-set effect is generic; if no, it's specific.
- **Paired bootstrap CIs over pair indices, 2000 resamples.** The same 20 prompt pairs are seen in both baseline and patched conditions, so the per-pair difference is the right unit. The two-sided p-value is computed by recentering the per-pair diffs and counting bootstrap means with absolute value at least the observed.
- **Verdict thresholds.** `CAUSAL_SHARED` requires both `shared - random > 0.05` margin and a paired CI excluding zero; `PARTIAL_CAUSAL` if only one holds; `NOT_CAUSAL` otherwise.

## How to run it

```bash
# The headline run: Llama-3.3-70B, 20 pairs, top-15 shared heads
uv run shared-circuits run activation-patching \
  --model meta-llama/Llama-3.3-70B-Instruct --n-devices 2

# Smaller model (e.g. for sanity checking against attribution-patching)
uv run shared-circuits run activation-patching --model gemma-2-2b-it

# Use a different K bucket from the saved circuit-overlap result
uv run shared-circuits run activation-patching \
  --model Qwen/Qwen2.5-32B-Instruct --shared-heads-k 30

# More pairs + bootstrap resamples for a tighter CI (slow at 70B)
uv run shared-circuits run activation-patching \
  --model meta-llama/Llama-3.3-70B-Instruct --n-devices 2 \
  --n-pairs 40 --n-boot 5000
```

Output: `experiments/results/activation_patching_<model_slug>.json`. Key fields:

| Field | Meaning |
|---|---|
| `verdict` | `CAUSAL_SHARED` / `PARTIAL_CAUSAL` / `NOT_CAUSAL` |
| `shift` | Mean `patched - baseline` logit-diff over pairs |
| `p_value` | Bootstrap two-sided p (recentered diffs) |
| `shared.delta`, `shared.ci`, `shared.significant` | Shared-set effect + 95% CI |
| `random.delta`, `random.ci`, `random.significant` | Count-matched random-head control |
| `shared_heads`, `random_heads` | The two head sets used |

## Where it lives in the paper

§3.4 (the causal-validation suite, Figure `fig:causal` — `causal_convergence.png`). The companion at smaller scale is [`attribution-patching`](attribution-patching.md) (Appendix `attrpatch`); the explicit compute justification ("a single per-head sweep costs >50 GPU-hours per model and was redundant") is in Appendix `compute`. The headline number at 70B is the matching restoration ratio reported alongside in Table `restoration` — Llama-3.3-70B factual lying clears `1,732×` (95% CI `[1,417, 2,194]`).

## Source

`src/shared_circuits/analyses/activation_patching.py` (~300 lines). Reads the ranked `shared_heads` list from a saved [`circuit-overlap`](circuit-overlap.md) JSON via `--shared-heads-from`. Sibling analyses in the same causal suite: [`attribution-patching`](attribution-patching.md), [`head-zeroing`](head-zeroing.md), [`projection-ablation`](projection-ablation.md), [`norm-matched`](norm-matched.md), [`faithfulness`](faithfulness.md), [`causal-ablation`](causal-ablation.md), [`path-patching`](path-patching.md).
