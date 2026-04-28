# `probe-transfer`

> Train a logistic-regression probe on sycophancy activations. Does it transfer to lying without retraining?

If sycophancy and factual lying share a substrate, a linear classifier trained on one task should generalize to the other zero-shot. This is the §3.5 / §4 substrate-persistence claim in its sharpest form: it survives even after anti-sycophancy DPO pushes the behavioral rate down 10–93%.

<p align="center">
  <img src="../img/rlhf_natural_experiment.png" width="600" alt="RLHF natural experiment: behavior drops 10x while the syc-trained probe still transfers to lying at AUROC 0.83-0.85.">
</p>

## The mech-interp idea

A **linear probe** is a classifier — here scikit-learn's L2-regularized logistic regression — fit on a per-prompt vector of activations to predict a binary label. We extract the residual stream at one layer and one position (the last prompt token), giving each prompt a `d_model`-dim feature vector. Train: 100 wrong-opinion vs 100 correct-opinion sycophancy prompts (the label is "user is incorrect"). Test: 100 false vs 100 true factual-lying prompts (the label is "statement is false") — completely different prompts, completely different facts. We report AUROC on the test set with no in-sample fitting.

If the model uses the same internal feature for "this is wrong" across both tasks, the probe trained on syc lands on a separating hyperplane that lying activations also obey. If the two tasks use different features, AUROC drops to chance (0.5).

The construction is the cousin of Marks & Tegmark's truth direction (2024) and Zou et al.'s representation engineering (2023), with one twist: instead of computing the mean-difference vector once and stress-testing it, we let logistic regression find whatever linear discriminant maximizes margin on syc, then ask whether that discriminant — not a hand-built one — happens to work on lying. The Ying et al. (2026) "limited transfer" reading reported AUROC 0.59–0.62 in chat models and read it as evidence for distinct mechanisms; the reconciliation appendix (`app:reconciliation`) shows this floor (`0.61` on Qwen2.5-1.5B, our `--single-model` reproduction of their setup) is the lower end of a much wider distribution that hits 0.83–0.85 in larger chat models.

## Why this design

- **Probe layer at `0.85 × n_layers`.** Fixed prior to running, picked from the layer-wise direction-cosine peak in [`direction-analysis`](direction-analysis.md). No per-model search — that would be hyperparameter-fishing on a tiny test set. Override with `--probe-layer` (absolute) or `--probe-layer-frac`.
- **Disjoint train/test pairs.** Sycophancy uses TriviaQA pairs `[0, 200)`; lying uses pairs `[200, 400)`. Same template family, completely different facts. Probe transfer cannot succeed by memorizing surface forms.
- **Logistic regression over mean-difference.** A learned probe overfits to *task-specific* directions in addition to the shared discriminant; the mean-difference vector by construction captures only the shared component. The probe-transfer AUROC ceiling is therefore a tighter bound than the cosine-of-directions number from [`direction-analysis`](direction-analysis.md): the 0.83–0.85 says the shared component dominates the syc decision boundary even when fitting can pick up everything else.
- **Single-model variant for DPO sweeps.** `--single-model` adds the reverse `lie→syc` direction and supports `--weight-repo` (point at a merged DPO checkpoint) and `--tag` (suffix for the result file). The §3.5 equivalence-margin claim runs three of these per model: baseline, anti-syc, sham.
- **`--n-boot` is opt-in.** The headline numbers use Hanley-McNeil analytic CIs (Gemma, Qwen3-8B) or 5-fold CV (Qwen2.5-1.5B); the bootstrap is for the DPO-comparison setting where you want a paired CI on the train-set resample.

## How to run it

```bash
# Headline single-model run (Gemma-2-2B at the 0.85L probe layer)
uv run shared-circuits run probe-transfer --single-model gemma-2-2b-it

# Multi-model sweep (default ALL_MODELS list)
uv run shared-circuits run probe-transfer

# DPO-checkpoint variant (the §3.5 substrate-persistence call)
uv run shared-circuits run probe-transfer \
  --single-model mistralai/Mistral-7B-Instruct-v0.1 \
  --weight-repo ./dpo_runs/Mistral-7B-Instruct-v0.1_merged \
  --tag post_anti_dpo \
  --n-boot 1000

# Override the probe layer (if you want to trace the AUROC-by-layer profile)
uv run shared-circuits run probe-transfer \
  --single-model Qwen/Qwen3-8B --probe-layer 30
```

Output: `experiments/results/probe_transfer<_tag>_<model>.json`. Key fields:

| Field | Meaning |
|---|---|
| `probe_layer` / `n_layers` | Resolved probe layer + total layer count |
| `syc_to_lie` | Multi-model sweep: full AUROC payload from `evaluate_probe_transfer` |
| `lie_to_syc` | Single-model only: reverse direction AUROC |
| `syc_to_lie_bootstrap` / `lie_to_syc_bootstrap` | Single-model + `--n-boot > 0`: `{mean, ci_lo, ci_hi, n_boot}` |
| `weight_repo` / `tag` | Single-model: provenance fields for downstream join logic |

## Where it lives in the paper

§3.5 + §4, **`tab:probe-ci`** in Appendix `app:probe-ci` — the per-model AUROC table. Headline AUROCs at the `0.85L` probe layer: Gemma-2-2B `0.83` `[0.77, 0.89]`, Qwen3-8B `0.85` `[0.80, 0.90]`, Mistral-7B `0.84` (single-fit), Qwen2.5-1.5B `0.61` `[0.59, 0.63]` (the Ying-et-al. floor, recovered in our setup). The DPO equivalence-margin counterpart `tab:dpo-auroc` is anti-syc `|Δ| ≤ 0.026`, sham `|Δ| ≤ 0.002`, both inside the pre-specified ±0.05 margin while sycophancy drops 93% (Mistral) and 46% (Gemma).

## Source

`src/shared_circuits/analyses/probe_transfer.py` (~210 lines). Reads no upstream JSON; the upstream signal is the activation extractor (`shared_circuits.extraction.extract_residual_stream`) + the probe helper (`shared_circuits.stats.evaluate_probe_transfer`). The single-model path is the one [`dpo-antisyc`](dpo-antisyc.md) drives downstream via `--weight-repo` to land Tables 11 + 12.
