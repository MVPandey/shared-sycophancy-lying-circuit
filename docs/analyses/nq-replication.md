# `nq-replication`

> Is the shared circuit a TriviaQA-specific artifact, or does it transfer to a different QA dataset?

Re-runs [`circuit-overlap`](circuit-overlap.md) on NaturalQuestions (NQ) instead of TriviaQA, then asks whether the *per-head importance grids themselves* correlate across the two datasets. If TriviaQA and NQ rank the same heads as important, the circuit is dataset-invariant and the headline result isn't a TriviaQA-fitting artifact.

## The mech-interp idea

NaturalQuestions (Kwiatkowski et al., 2019) is questions mined from real Google search queries with answers extracted from Wikipedia — different surface forms, different topic mix, different answer-length distribution than TriviaQA. If the shared sycophancy↔lying circuit is real, the same head positions should top-rank for sycophancy on TriviaQA and on NQ; if it's a feature of the trivia genre, the two grids should drift apart.

We measure two things per model:

1. **Within-NQ shared-head overlap** — exactly the [`circuit-overlap`](circuit-overlap.md) construction (top-K syc heads ∩ top-K lie heads, hypergeometric/permutation p, Spearman ρ over all heads), but with NQ as the prompt source.
2. **Cross-dataset Pearson correlation** — flatten the per-head importance grid for sycophancy on TriviaQA and on NQ, compute `pearsonr(triviaqa_syc_flat, nq_syc_flat)`. Same for lying. This asks "does the model rank heads the same way for sycophancy regardless of which dataset the prompts came from?"

Test 1 confirms the shared-circuit story holds on a new dataset; test 2 is the stronger claim — the *whole grid* is dataset-invariant, not just the top.

## Why this design

- **Same `dla_prompts` count as the TriviaQA panel.** 100 prompts per condition, matching `circuit-overlap`'s defaults. Holding `n_prompts` fixed makes the within-NQ overlap directly comparable to the TriviaQA Table 1 row for the same model.
- **Disjoint NQ slices for syc and lie, same as TriviaQA.** Pairs `[0, 100)` for sycophancy, `[100, 200)` for lying. Identical content-disjointness controls as the main analysis.
- **Reads TriviaQA grids from saved JSON rather than recomputing.** This analysis only does the NQ pass + the cross-dataset correlation; the TriviaQA grids come from `breadth_<slug>.json` (or whatever you pass via `--triviaqa-grids-from`). That keeps the comparison apples-to-apples — same TriviaQA grid that drove the Table 1 result, no recomputation drift.
- **Default model list is small.** Gemma-2-2B-IT, Qwen2.5-1.5B, Llama-3.1-8B. The paper reports the full result on Gemma-2-2B-IT and Llama-3.3-70B-Instruct — substitute via `--models` for the 70B run.
- **Permutation null at 1000 draws (not 10000).** This is the within-NQ overlap test; a 10⁻³ floor is enough since the cross-dataset Pearson is the headline statistic, not the within-NQ p-value.

## How to run it

```bash
# Default 3-model NQ pass (Gemma-2B, Qwen-1.5B, Llama-3.1-8B)
uv run shared-circuits run nq-replication

# The two paper-table models
uv run shared-circuits run nq-replication \
  --models gemma-2-2b-it meta-llama/Llama-3.3-70B-Instruct \
  --n-pairs 200

# Use 50 NQ prompts per condition instead of the default 100 (cheaper)
uv run shared-circuits run nq-replication \
  --models gemma-2-2b-it --dla-prompts 50 --n-pairs 100

# Read TriviaQA grids from circuit_overlap instead of breadth
uv run shared-circuits run nq-replication \
  --models gemma-2-2b-it --triviaqa-grids-from circuit_overlap
```

Output: `experiments/results/nq_replication_<model_slug>.json`. Headline fields:

| Field | Meaning |
|---|---|
| `nq_within_dataset.syc_lie_overlap` | NQ top-K syc ∩ top-K lie, the within-dataset replication |
| `nq_within_dataset.syc_lie_ratio` | Same, divided by `K²/N` chance baseline |
| `nq_within_dataset.syc_lie_rho` | Spearman ρ over the full NQ head grid |
| `cross_dataset_triviaqa_vs_nq.syc_pearson` | Per-head Pearson `r(TriviaQA syc grid, NQ syc grid)` |
| `cross_dataset_triviaqa_vs_nq.lie_pearson` | Same for lying |
| `cross_dataset_triviaqa_vs_nq.{syc,lie}_overlap` | TriviaQA-vs-NQ top-K agreement on the syc and lie rankings |

## Where it lives in the paper

Appendix § "NaturalQuestions cross-dataset replication", Table `tab:nq`. Headline numbers (from the paper):

| Model | NQ syc∩lie | TQA↔NQ syc overlap | ρ(syc) | TQA↔NQ lie overlap | ρ(lie) |
|---|---|---|---|---|---|
| Gemma-2-2B | 13/15 (12×) | 15/15 (13.9×) | 0.991 | 14/15 (12.9×) | 0.988 |
| Llama-3.3-70B | 47/72 (46×) | 61/72 (60×) | 0.991 | 57/72 (56×) | 0.989 |

The §3.1 main body cites this as the reason it can claim "the pattern replicates across datasets at ρ≈0.99 on Gemma-2-2B and Llama-3.3-70B".

## Source

`src/shared_circuits/analyses/nq_replication.py` (~200 lines). Calls `load_naturalquestions_pairs` from `shared_circuits.data`. Reads TriviaQA grids from a saved `breadth_<slug>.json` (default) for the cross-dataset comparison; runs its own DLA pass on NQ for the within-dataset overlap. Consumed by manual paper-table generation only.
