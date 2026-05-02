# DPO preference datasets

The exact JSONL files used to train the §3.5 anti-sycophancy LoRAs and their sham controls. Both pairs come from TriviaQA indices `[500, 1499]` for train and `[1500, 1597]` for eval — disjoint from the `[0, 400)` slice used for probe-transfer evaluation, so DPO training and probe-transfer evaluation never see the same factual content.

| File | Rows | Description |
|---|---|---|
| `antisyc_dpo_train.jsonl` | 1,000 | Anti-sycophancy preferences. For each TriviaQA `(question, wrong-answer w, correct-answer c)` triple, two rows: a *wrong-opinion* prompt with `chosen = disagree-and-correct`, `rejected = sycophantic-yes`; and a *right-opinion* prompt with `chosen = agree`, `rejected = wrong-disagree`. Balancing both opinion directions stops the model from collapsing to "always disagree". |
| `antisyc_dpo_eval.jsonl` | 98 | Held-out eval pairs, same construction. |
| `sham_dpo_train.jsonl` | 1,000 | Sham (placebo) — same prompt set, `chosen` / `rejected` randomly swapped per pair (`seed=42`). The marginal distribution over responses is preserved; the **preference signal is destroyed**. About half the pairs end up with the same labels as the anti-syc version (when the random coin came up heads), the other half are flipped. |
| `sham_dpo_eval.jsonl` | 98 | Sham eval. |

## Schema

Each row is a single JSON object using the HuggingFace-Datasets chat-format preference shape consumed by [TRL's DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer):

```json
{
  "prompt": [
    {"role": "user", "content": "I believe the answer to \"...\" is X. Am I correct?"}
  ],
  "chosen":  [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}]
}
```

Loaded as `datasets.Dataset.from_json(path)` or via the library helpers `shared_circuits.data.dpo_preferences.build_antisyc_preferences()` / `build_sham_preferences()` (which regenerate equivalent content deterministically from the public TriviaQA split).

## Reproducibility note

These files are committed verbatim because the user often wants the *exact* dataset the paper trained on, not a re-derivation that happens to be deterministic. They're equivalent to what the library helpers produce, but pinning the artifact removes the entire class of "did your TriviaQA loader change?" questions.

## Where they get used

The [`dpo-antisyc`](../../docs/analyses/dpo-antisyc.md) analysis loads these directly. The library helpers above produce the same shape if you'd rather build them on the fly.
