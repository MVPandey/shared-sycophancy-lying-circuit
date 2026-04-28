# `reverse-projection`

> Ablate the lying direction on sycophancy prompts (and vice-versa). Does behavior on the other task change?

This is the cross-task coupling test. If the residual subspaces for sycophancy and factual lying are coupled, removing one task's direction from the residual stream should hurt the other task's behavior. Two empty cells (cross-task ablation does nothing) would falsify the shared-substrate reading; two filled cells confirms it. The same protocol on a post-DPO model is how §3.5 shows that anti-sycophancy training *increased* the coupling rather than dissolving it.

<p align="center">
  <img src="../img/rlhf_natural_experiment.png" width="600" alt="RLHF refresh and DPO both reduce behavior while the projection-ablation effect grows. The substrate persists; the routing tightens.">
</p>

## The mech-interp idea

A **task direction** `d_t` is the unit vector pointing from one task condition to the other in residual-stream space. We compute it per-layer as `mean(corrupt_resid) − mean(clean_resid)` and normalize. For sycophancy that's `mean(wrong-opinion) − mean(right-opinion)`; for lying that's `mean(false-statement) − mean(true-statement)`. The construction is the same as Arditi et al.'s refusal direction (2024) and Marks & Tegmark's truth direction (2024).

**Projection ablation** — subtract `(resid · d) * d` at every layer's `hook_resid_pre` — removes the component of the residual along that direction without touching the orthogonal complement. The classic same-task setting (ablate `d_syc`, measure syc) is what [`projection-ablation`](projection-ablation.md) does and what the §3.4 sufficiency story rests on. Reverse projection is the **cross-task** version: ablate `d_syc` and measure the *lying* task's logit-diff gap, then ablate `d_lie` and measure the *sycophancy* gap. Four cells get reported; the two cross-task cells are the test.

The `frac_preserved = ablated_gap / baseline_gap` summarizes each cell. A value near `1` means the ablation didn't move the other task (the directions are independent). A value near `0` means the ablation crushed the other task (the directions live in coupled subspaces). The verdict thresholds at `0.5` for declaring `COUPLED`.

## Why this design

- **Per-layer directions, projected at every layer's `hook_resid_pre`.** The earlier-layer ablations matter because the direction propagates forward through the residual stream; restricting the hook to one layer would underestimate the effect.
- **Four cells, not two.** The same-task cells (ablate-syc/measure-syc, ablate-lie/measure-lie) are sanity checks: if `frac_preserved` isn't small there, the directions don't even capture their *own* task and the cross-task readout is meaningless. The cross-task cells (ablate-syc/measure-lie, ablate-lie/measure-syc) are the headline.
- **`--from-direction-file` for DPO replay.** The DPO-trained Mistral / Gemma runs in §3.5 reuse the *pre-DPO* direction so the cross-task delta isn't absorbing a directional drift on top of the coupling change. Pass a JSON with `d_syc` + `d_lie` (or the `lying_task`-keyed variant) and the analysis skips recomputation.
- **Three lying-task choices.** `--lying-task instructed_lying` (default; the factual-contrast version), `scaffolded_lying`, or `repe_lying`. Tag suffixes (`_scaffolded`, `_repe`) get appended to the result filename so a single model can run all three without clobbering.
- **`--weight-repo` to point at a merged adapter.** Same convention as [`probe-transfer`](probe-transfer.md): the post-DPO Mistral and Gemma checkpoints get exercised through here without needing to re-merge weights or duplicate the analysis path.

## How to run it

```bash
# Single-model baseline (Mistral-7B factual-lying contrast)
uv run shared-circuits run reverse-projection --model mistralai/Mistral-7B-Instruct-v0.1

# Post-DPO replay using cached pre-DPO directions (the §3.5 coupling-grew claim)
uv run shared-circuits run reverse-projection \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --weight-repo ./dpo_runs/Mistral-7B-Instruct-v0.1_merged \
  --from-direction-file experiments/results/reverse_projection_mistral.json \
  --tag post_anti_dpo

# Alternative lying paradigms
uv run shared-circuits run reverse-projection --model gemma-2-2b-it --lying-task scaffolded_lying
uv run shared-circuits run reverse-projection --model gemma-2-2b-it --lying-task repe_lying
```

Output: `experiments/results/reverse_projection<_lying-suffix><_tag>_<model>.json`. Key fields:

| Field | Meaning |
|---|---|
| `verdict` | `COUPLED` / `PARTIALLY_COUPLED` / `INDEPENDENT` / `INCOMPLETE` (threshold `frac_preserved < 0.5`) |
| `mean_cos_syc_lie` / `per_layer_cos` | Layer-wise cosine of `d_syc` vs `d_lie`, mean-pooled and per-layer |
| `cells.<ablate>_<measure>` | Per-cell `{baseline_gap, ablated_gap, frac_preserved}` plus per-prompt logit-diffs |

## Where it lives in the paper

§3.5 (`sec:rlhf`) and Appendix `app:probe-ci`. Post-DPO: ablating `d_syc` drops the lying gap **18%** on Mistral-7B and **54%** on Gemma-2-2B-IT; ablating `d_lie` drops sycophancy **22%** on Mistral and **42%** on Gemma. The pre-DPO Mistral baselines (paired bootstrap CI from `app:probe-ci`): `d_syc` ablation preserves the lying gap at `1.11×` `[1.09, 1.14]` (no coupling); `d_lie` ablation produces only **14%** sycophancy reduction `[9.9%, 18.9%]`. The post-DPO deltas exceed the pre-DPO CIs in both directions — coupling grew. The Llama-3.1→3.3-70B refresh row of `tab:rlhf` tells the same story at frontier scale: projection ablation flips from `+10.5pp` to `+27pp` while sycophancy drops 10×.

## Source

`src/shared_circuits/analyses/reverse_projection.py` (~310 lines). Sibling to [`projection-ablation`](projection-ablation.md) (same hook, same-task) and [`direction-analysis`](direction-analysis.md) (cosine of `d_syc` vs `d_lie` without ablation). Driven downstream by [`dpo-antisyc`](dpo-antisyc.md) for the §3.5 substrate-persistence chain and feeds the §3.5 `tab:rlhf` row directly.
