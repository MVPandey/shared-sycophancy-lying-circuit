# `dla-instructed-lying`

> Does the same shared circuit drive *explicit* lying — the model knows it's being asked to fib — or only the implicit truth-statement contrast?

The factual-lying contrast in [`circuit-overlap`](circuit-overlap.md) is "true statement" vs "false statement", which a skeptic could call "factual evaluation, not deception." This analysis re-ranks heads under three independent paradigms where the model is explicitly asked to assert a falsehood, and checks the overlap with the sycophancy ranking. Same heads, every paradigm.

## The mech-interp idea

There are at least three established ways to get a model to lie on demand:

- **Jailbreak preamble** — a system-style instruction telling the model to lie regardless of truth, then a question. The "lying" condition is the model complying with that preamble; the "honest" condition is the same question with no preamble.
- **Scaffolded few-shot** (Halawi et al., 2024 / Chen et al., 2024) — the prompt contains a handful of examples where the model lies, then asks a new question. The few-shot demonstrations do the heavy lifting; the model imitates.
- **RepE persona** (Zou et al., 2023) — embed the lying instruction inside a role-play preamble that frames the model as a deceptive character. This is the persona-based variant of representation-engineering style elicitation.

For each paradigm, we compute the same DLA write-norm head importance from [`circuit-overlap`](circuit-overlap.md) — the L2 norm of the head's mean write-vector difference between the lying and honest conditions, `||W_O · (v̄_lie − v̄_honest)||₂`. Then we ask whether the top-K heads under the *instructed-lying* ranking overlap with the top-K under the *sycophancy* ranking. If sycophancy and instructed lying share machinery, the overlap should look like the factual-lying overlap.

The three paradigms are independently informative because each could fail differently — jailbreak might engage refusal heads, few-shot might engage in-context-learning heads, RepE might engage role-play heads. Convergence across all three pins down "the lying signal" rather than any single elicitation method.

## Why this design

- **Single-template-per-model caveat is real and disclosed.** The legacy script and the paper note that the lying paradigms each use one template family per model (see `build_instructed_lying_prompts`, `build_scaffolded_lying_prompts`, `build_repe_lying_prompts`). Template invariance has not been controlled in this analysis; it's a documented limitation.
- **Sycophancy uses pairs `[0, n_prompts)`, lying uses pairs `[n_prompts, 2*n_prompts)`.** Disjoint TriviaQA content for the two contrasts, same as in `circuit-overlap`. The pair count defaults to 400, of which the first `n_prompts` (default 50) feed sycophancy and the next 50 feed lying. If only one slot's worth of pairs is loaded, sycophancy pairs are reused (legacy fallback).
- **Per-paradigm save slug.** Each paradigm writes its own JSON: `dla_instructed_lying_jailbreak_<model>.json`, `..._scaffolded_<model>.json`, `..._repe_<model>.json`. Downstream analyses (notably [`path-patching`](path-patching.md)'s `--shared-source instructed/scaffolded/repe`) pick which paradigm's shared-head set drives the edge tracing.
- **K = ⌈√N⌉ to match the rest of the paper.** The shared count and shared heads are reported alongside `top-15` per task for both rankings, so a reviewer can rebuild the table from the JSON.

## How to run it

```bash
# Jailbreak paradigm on the default 7-model panel
uv run shared-circuits run dla-instructed-lying

# Scaffolded paradigm on a single model
uv run shared-circuits run dla-instructed-lying \
  --models gemma-2-2b-it --paradigm scaffolded

# RepE persona paradigm
uv run shared-circuits run dla-instructed-lying \
  --models meta-llama/Llama-3.3-70B-Instruct --paradigm repe \
  --n-prompts 100

# Smaller test run
uv run shared-circuits run dla-instructed-lying \
  --models google/gemma-2-9b-it --paradigm jailbreak \
  --n-pairs 100 --n-prompts 30
```

Output: `experiments/results/dla_instructed_lying_<paradigm>_<model_slug>.json`. Key fields:

| Field | Meaning |
|---|---|
| `paradigm` | One of `jailbreak`, `scaffolded`, `repe` |
| `task_pair` | Always `sycophancy_<paradigm>` for clarity |
| `shared_count` | Top-K syc ∩ top-K instructed-lying overlap |
| `shared_heads` | The `(layer, head)` pairs in the intersection |
| `syc_grid` / `lie_grid` | Per-head importance arrays for both contrasts |
| `syc_top15` / `lie_top15` | Ranked top heads with their delta-norms |

## Where it lives in the paper

§3.2, **Table 2** (`tab:instructed`). Result on the seven-model jailbreak panel: shared fraction 0.25–0.74, ratio 5.5–25.7×, Spearman ρ over the full head population 0.73–0.93 (all `p < 10⁻³⁷`). Mixtral-8x7B at ρ=0.93 is the first sparse-MoE validation at the instructed-lying head level. The two lowest fractions (Gemma-2-9B at 0.27 and Phi-4 at 0.25) are read as K-boundary effects since their full-population Spearman ρ remain in the 0.73–0.89 band. Scaffolded and RepE results live in the saved JSON; aggregate paradigm-cross-paradigm consistency is reported through [`path-patching`](path-patching.md), where the same shared-head sets restore 90–102% of the clean-vs-corrupt gap on Phi-4 across all three lying paradigms.

## Source

`src/shared_circuits/analyses/dla_instructed_lying.py` (~160 lines). Calls `compute_head_importances` and `compute_head_importance_grid` from `shared_circuits.attribution`; uses `build_instructed_lying_prompts`, `build_scaffolded_lying_prompts`, `build_repe_lying_prompts` from `shared_circuits.prompts`. Output JSON is consumed by [`path-patching`](path-patching.md) (via the `--shared-source` flag) and by manual paper-table generation.
