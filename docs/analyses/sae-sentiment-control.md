# `sae-sentiment-control`

> Is the syc∩lie SAE-feature overlap specific to factual correctness, or just generic statement-evaluation?

The 41/100 SAE feature overlap from [`sae-feature-overlap`](sae-feature-overlap.md) is huge — but a skeptical reader can rescue it as "any binary-evaluation task lights up the same features". This analysis runs the falsification: compute the same top-K SAE feature overlap between sycophancy and a *sentiment-classification* task (positive vs negative movie reviews, n = 100 prompts) on Llama-3.1-8B at layer 19. If the syc∩sentiment overlap matches the syc∩lie reference, the "evaluation-general" reading wins. If syc∩lie is significantly larger than syc∩sentiment on the same prompts and the same SAE, factual correctness is doing real work inside the substrate.

## The mech-interp idea

Sparse autoencoders (Cunningham et al. 2023; Bricken et al. 2023) decompose a residual-stream activation into a sparse sum of `m ≫ d` feature activations via a learned encoder/decoder pair, with each feature interpretable as a concept. The Goodfire SAE for Llama-3.1-8B at layer 19 has `d_sae = 65 536` features, so a chance top-100 overlap is `100² / 65 536 ≈ 0.15` features.

We compute three pairwise top-100 overlaps:

1. **syc ∩ lie** — the reference, read from a saved [`sae-feature-overlap`](sae-feature-overlap.md) JSON for the same `(model, layer)`. 41/100 at 269× chance.
2. **syc ∩ sentiment** — fresh; we encode 100 positive + 100 negative movie-review prompts through the same SAE, take `sent_diff = mean(f_pos) - mean(f_neg)`, and intersect the top-100 with the syc top-100.
3. **lie ∩ sentiment** — same construction with the lie reference set.

The headline test is **McNemar's exact test on the discordant pairs**. McNemar (1947) is the right statistic for paired binary outcomes: out of the `syc_top` features, how many are *also* in the `lie` set but not the `sentiment` set, vs how many are in `sentiment` but not `lie`? Under the null that syc agrees equally with lie and with sentiment, those two discordant counts are exchangeable; the two-sided exact-binomial p-value at `p = 0.5` formalises that. Concordant pairs (in both, in neither) carry no information about the *difference* in agreement.

## Why this design

- **McNemar specifically, not a two-sample test.** The two intersections share the syc set as a common reference, so the feature-level outcomes are paired (same feature index `i` either is or isn't in the lie set, and same for sentiment). A two-sample chi-square would treat them as independent and overcount sample size.
- **`n = 100` movie-review prompts, fixed templates.** Ten positive and ten negative templates copied bit-for-bit from the legacy `run_sae_sentiment_control.py` script, sampled with replacement under a seeded RNG. Reproducibility comes from the seed, not from prompt diversity — the question is whether the *sentiment task itself* shares features with sycophancy, not whether sentiment + linguistic variability does.
- **Read syc/lie reference from saved JSON, don't recompute.** The reference comes from the same `sae_feature_overlap_<model>.json` that drives Table 5, so the McNemar comparison runs against the *exact* feature lists the paper reports — no rerun drift. Override the source with `--reference-from`.
- **Permutation null over feature labels (n = 1000 draws).** The hypergeometric p is reported alongside, but the permutation null is the model-free version: shuffle `sent_diff` over the feature axis and recount the syc∩sent overlap. The reported p is `(ge + 1) / (n_perm + 1)`, floored at ≈10⁻³ for the default 1000 draws.
- **Single model + layer.** The control needs to land on the same row as the headline 41/100 result, so this analysis is locked to one `(model, layer)` per run.

## How to run it

```bash
# Default: Llama-3.1-8B L19, reads sae_feature_overlap_llama-3.1-8b-instruct.json
uv run shared-circuits run sae-sentiment-control

# Different model/layer (must have a saved sae-feature-overlap result for it)
uv run shared-circuits run sae-sentiment-control \
  --model gemma-2-2b-it --layer 12

# Smaller prompt count for a fast pilot
uv run shared-circuits run sae-sentiment-control --n-prompts 50

# Override the reference source
uv run shared-circuits run sae-sentiment-control \
  --reference-from sae_feature_overlap
```

If no saved [`sae-feature-overlap`](sae-feature-overlap.md) result exists for the requested `(model, layer)`, the analysis raises `FileNotFoundError` with a hint to run that first.

Output: `experiments/results/sae_sentiment_control_<model_slug>.json`. Headline fields:

| Field | Meaning |
|---|---|
| `reference_syc_lie_overlap`, `reference_syc_lie_ratio` | Headline 41 / 269× from the loaded reference |
| `syc_sent_overlap`, `syc_sent_ratio` | New: top-100 syc ∩ sentiment + ratio |
| `lie_sent_overlap`, `lie_sent_ratio` | New: top-100 lie ∩ sentiment + ratio |
| `syc_sent_p_perm`, `syc_sent_p_hyper` | Permutation + hypergeometric p for syc∩sent |
| `lie_sent_p_perm`, `lie_sent_p_hyper` | Same for lie∩sent |
| `shared_sent_overlap` | How many of the 41 *shared* features are also in sentiment top-100 |
| `mcnemar_syc_lie_vs_syc_sent` | `{in_lie_not_sent, in_sent_not_lie, p_value}` — the headline test |
| `sentiment_top_features` | Sorted index list of the sentiment top-100 |

## Where it lives in the paper

Appendix § "SAE feature overlap: controls and robustness (Llama-3.1-8B, layer 19)" (`app:sae-controls`), as the sentiment-task-control paragraph. Headline: syc∩sentiment 24/100 (157× chance, p < 10⁻³); lie∩sentiment 32/100 (210×); syc∩lie reference 41/100 (269×). The McNemar discordant-pair test gives **p = 0.002**, so syc∩lie is significantly greater than syc∩sentiment on the same SAE. The reading: the shared circuit is **evaluation-general** rather than purely factual-incorrectness-specific (a generic-evaluation reader can't claim the overlap is zero), but the factual overlap (269×) substantially exceeds the sentiment overlap (157×), consistent with a factual-correctness emphasis within a broader statement-evaluation substrate.

## Source

`src/shared_circuits/analyses/sae_sentiment_control.py` (~260 lines). Reads `sae_feature_overlap_<model>.json` as upstream. Same `data/sae_features.py` loader + `extraction/sae.py` encoder as the rest of the SAE family; the prompt builder lives inside the analysis itself (`build_sentiment_prompts` + the `POSITIVE_TEMPLATES` / `NEGATIVE_TEMPLATES` constants) and uses `prompts.render_chat` to apply the model's chat template. Consumed only by manual paper-table generation. The head-level analogue of this control — "is the head-overlap factual-incorrectness-specific or generic component-reuse?" — is documented in §3.1 of the paper as the Merullo et al. (2024) component-reuse floor comparison rather than as a separate analysis.
