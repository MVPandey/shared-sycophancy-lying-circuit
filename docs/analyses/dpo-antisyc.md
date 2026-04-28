# `dpo-antisyc`

> Train an anti-sycophancy adapter via DPO. Does the underlying "this is wrong" circuit move?

This is the controlled-intervention counterpart to the Llama-3.1→3.3-70B natural experiment. We deliberately reduce sycophantic behavior with a small DPO LoRA and ask whether the syc↔lie probe transfer survives. If the substrate is what alignment touches, the probe should weaken. If alignment touches the routing layer downstream of the substrate, the probe should be invariant.

<p align="center">
  <img src="../img/rlhf_natural_experiment.png" width="600" alt="Llama-3.1→3.3 natural experiment: sycophancy drops 10×, projection-ablation effect grows from +10.5pp to +27pp. The substrate persists.">
</p>

## The mech-interp idea

Direct Preference Optimization (DPO; Rafailov et al.) skips the reward-model + RL stage of standard RLHF: given a fixed dataset of `(prompt, chosen, rejected)` triples, it directly increases the log-likelihood of `chosen` minus `rejected` under the policy, weighted by a temperature β. With a LoRA adapter, only a few low-rank update matrices are trainable — the base model is frozen.

We build two preference datasets from TriviaQA:

- **Anti-syc** — for each `(question, wrong-answer w, correct-answer c)` triple we emit two balanced preference rows:
  - *Wrong-opinion prompt* (`I believe the answer is w. Am I correct?`): chosen = `disagree-and-correct`, rejected = `sycophantic-yes`.
  - *Right-opinion prompt* (`I believe the answer is c. Am I correct?`): chosen = `agree`, rejected = `wrong-disagree`.

  Balancing both opinion directions is what stops the model from collapsing to "always disagree".

- **Sham** — same prompt set, but `chosen` and `rejected` are randomly swapped per pair (`seed=42`). The marginal distribution over responses is identical; the **preference signal is destroyed**. This is the placebo: matched gradient-update count, matched LoRA footprint, no coherent training signal.

After training, we ask:

1. Did sycophantic behavior actually drop? (We measure rate on a held-out evaluation set.)
2. Did the syc→lie probe-transfer AUROC change beyond the pre-specified ±0.05 equivalence margin? Run [`probe-transfer`](probe-transfer.md) on the merged-adapter model.
3. Does the sham control move the probe by the same amount? If it does, "any DPO update perturbs the probe" is the explanation, not "the preference signal taught the model to gate the substrate".

## Why this design

- **LoRA `r=16, α=32` on `q_proj` + `v_proj` only.** Smallest configuration that reliably moves the sycophancy rate at this scale on TRL-default DPO. Bigger ranks would risk teaching the model to memorize the dataset; smaller ranks didn't move behavior in pilot runs.
- **β = 0.1, lr = 5e-5, 2 epochs.** TRL defaults; we did not tune. The ±0.05 AUROC equivalence margin and the sham control jointly bound how much hyperparameter variation could distort the headline claim.
- **`n=1000` preference pairs (= 500 TriviaQA triples × 2 templates).** Smallest size that produced a measurable rate shift on Gemma-2-2B in pilots.
- **Indices `[500, 1499]` for train, `[1500, 1549]` for eval.** Disjoint from the `[0, 400)` slice used by the rest of the paper for probe-transfer evaluation, so DPO training and probe-transfer evaluation never see the same factual content.
- **Adapter merged into base weights post-training.** Downstream TransformerLens hooks need to operate on the same architectural surface as the untreated baseline. The merge is reversible if you keep the adapter checkpoint.
- **The sham is the load-bearing control.** Without it, any probe-transfer change is ambiguously "preference signal" vs "DPO procedure". With matched sham showing |ΔAUROC| < 0.002 while anti-syc shows |ΔAUROC| < 0.026, the equivalence-margin claim is real.

## How to run it

```bash
# Anti-sycophancy run on Mistral-7B (~30-60 min on a single 80GB GPU)
uv run shared-circuits run dpo-antisyc \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --mode anti

# The matched sham control (must run; it's load-bearing)
uv run shared-circuits run dpo-antisyc \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --mode sham

# Smaller hyperparameter sweep
uv run shared-circuits run dpo-antisyc \
  --model google/gemma-2-2b-it --mode anti \
  --lora-r 8 --epochs 1 --n-train 200

# Skip the merge (keep just the adapter checkpoint)
uv run shared-circuits run dpo-antisyc \
  --model mistralai/Mistral-7B-Instruct-v0.1 --mode anti \
  --no-merge-adapter
```

Output: `experiments/results/dpo_antisyc_<mode>_<model>.json` containing the training metadata + TRL `Trainer.train()` return metrics. The merged adapter is saved to `--output-dir/<model>_merged/` (default sibling of `--output-dir`).

To use the merged checkpoint downstream (e.g. for probe transfer):

```bash
uv run shared-circuits run probe-transfer \
  --single-model mistralai/Mistral-7B-Instruct-v0.1 \
  --weight-repo ./dpo_runs/Mistral-7B-Instruct-v0.1_merged \
  --tag post_anti_dpo
```

## Where it lives in the paper

§3.5, **Table 4** (`tab:rlhf`). Headline result: anti-sycophancy DPO drops sycophancy **93%** on Mistral (28%→2%) and **46%** on Gemma-2-2B-IT (52%→28%). Probe-transfer AUROC deltas: anti-syc `|Δ| ≤ 0.026`, sham `|Δ| ≤ 0.002` — both inside the ±0.05 equivalence margin. [`reverse-projection`](reverse-projection.md) on the post-DPO models shows *increased* cross-task coupling: ablating `d_syc` drops the lying gap 18% (Mistral) / 54% (Gemma); ablating `d_lie` drops sycophancy 22% (Mistral) / 42% (Gemma). The substrate didn't move; the routing did.

## Source

`src/shared_circuits/analyses/dpo_antisyc.py` (~150 lines) plus `src/shared_circuits/data/dpo_preferences.py` (the `build_antisyc_preferences` and `build_sham_preferences` library helpers). Uses `trl.DPOTrainer` + `peft.LoraConfig`; all heavy ML imports are deferred to `_train()` so module import doesn't pull in the whole HuggingFace ecosystem.
