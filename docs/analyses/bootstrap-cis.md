# `bootstrap-cis`

> Re-CI a saved JSON without rerunning the GPU work.

This is the post-hoc statistics tool. It doesn't load a model, doesn't run any forward passes, and doesn't care about which analyses produced the JSONs it reads — it walks `experiments/results/`, finds every per-prompt indicator list and every `(count, total)` rate row, and emits 95% paired-bootstrap confidence intervals plus Wilson-score CIs for the rates. Useful when you tweak parameters in an upstream analysis and need fresh CIs without spending another GPU-hour on it.

## The mech-interp idea

There's nothing mech-interp here — this is plumbing. The paper's reproducibility story relies on every reported rate carrying a 95% CI; the CIs in the published tables come from the legacy `experiments/compute_bootstrap_cis.py` one-shot, which scanned every result JSON once and produced the table-author-friendly lookup file. This analysis is the productionised version of that script: same scan, same CI methodology, same output shape, but with `--results-glob`, `--keys`, and `--source` flags so you can re-CI a subset (say, only `breadth_*` files after rerunning [`breadth`](breadth.md) with new prompts) without overwriting the headline lookup.

The arithmetic:

- **Per-prompt indicator lists** (`syc_per_prompt`, `lie_per_prompt`, etc. — anything in `--keys`): bootstrap the mean over `n_boot = 10 000` resamples of the prompt indices, report `[2.5%, 97.5%]` percentiles. This is a non-paired bootstrap because each list is a single condition.
- **`(count, total)` count rates**: any nested dict containing both `total` and one of `agree` / `correct` is treated as a binomial outcome. We report two CIs side by side: a closed-form **Wilson-score** interval at `z = 1.96` (Wilson 1927; better small-sample coverage than the normal approximation) and a percentile bootstrap over the equivalent ones-and-zeros array. Wilson is the headline number; the bootstrap is the consistency check.
- **Filename suffix via `--source`**: the output is saved to `bootstrap_cis_<source>.json`, so `--source dpo` and `--source headoverlap` produce side-by-side files instead of overwriting each other.

## Why this design

- **No model loading.** All inputs come from saved JSONs, so this is the one analysis you can run on a laptop without GPU access. It's how the paper-table author closes the loop after each analysis re-run.
- **`--results-glob` for scoping.** Default `*.json` scans everything; pass `--results-glob 'breadth_*.json'` or `'*_gemma_2_2b_it.json'` to restrict to the subset you care about. The output filename includes `--source` so multiple scoped runs coexist.
- **`--keys` for the per-prompt list names.** Default is `('syc_per_prompt', 'lie_per_prompt')`. Different analyses use different conventions (e.g. `agree_per_prompt`, `is_correct`); pass them explicitly when needed.
- **`--no-count-rates` to skip the rate scan.** When you only care about the per-prompt lists (say, a paired-bootstrap CI on a sycophancy-rate change), the `(count, total)` walk is wasted work. Disable it to halve the runtime.
- **Per-prompt CI is non-paired.** Each list is treated as a single condition; the "paired" word in this analysis's docstring refers to *prompt-paired* (the same `n` indices are resampled for the mean), not condition-paired. If you need a paired-condition CI on a syc-vs-lie difference, that's the job of the upstream analysis (e.g. [`activation-patching`](activation-patching.md) computes its own paired CI on per-pair logit-diff differences).
- **Wilson + bootstrap reported together.** Wilson is the analytic closed-form; bootstrap is the empirical sanity check. They should agree to within Monte Carlo noise; if they don't, that's a sign the binary outcomes inside the upstream analysis aren't actually IID Bernoulli.
- **Walks arbitrary nested JSON.** Both walkers (`_walk_per_prompt` and `_walk_counts`) recurse through dicts and lists, accumulating dotted-path keys (`steering.[2].rate`) so the output JSON is greppable by the paper-table author.

## How to run it

```bash
# Default: scan everything, both per-prompt lists and count rates
uv run shared-circuits run bootstrap-cis

# Scope to one model's results, tag the output by source
uv run shared-circuits run bootstrap-cis \
  --results-glob '*_gemma_2_2b_it.json' --source gemma2b

# Scope to a single analysis family
uv run shared-circuits run bootstrap-cis \
  --results-glob 'breadth_*.json' --source breadth

# Custom per-prompt key set
uv run shared-circuits run bootstrap-cis \
  --keys agree_per_prompt,is_correct --source perpromptonly

# Skip the (count, total) scan entirely
uv run shared-circuits run bootstrap-cis --no-count-rates --source listsonly

# Tighter bootstrap
uv run shared-circuits run bootstrap-cis --n-boot 100000 --source highprec
```

Output: `experiments/results/bootstrap_cis_<source>.json`. Top-level fields:

| Field | Meaning |
|---|---|
| `n_files_scanned`, `n_files_with_entries` | How many JSONs the glob matched and how many produced any CI row |
| `by_file[<stem>].per_prompt` | List of `{path, n, mean, bootstrap_ci_95}` for each per-prompt list found |
| `by_file[<stem>].count_rates` | List of `{path, n_pos, total, rate, wilson_ci_95, bootstrap_ci_95}` for each count pair |
| `n_boot`, `seed`, `results_dir`, `results_glob`, `keys` | Provenance fields for the run |

## Where it lives in the paper

Used as plumbing across the paper rather than reported as its own claim. The bootstrap-CI methodology is documented in Appendix § "Compute and reproducibility" (`app:compute`): "All bootstrap confidence intervals use 2 000 paired resamples, permutation nulls use 10 000 label permutations, and all random number generators are seeded for reproducibility". The DPO-specific CI table (Appendix `app:probe-ci`) and Table 11/12's CIs all originated from this scan over the upstream `probe_transfer_*` and `dpo_antisyc_*` JSONs. The default `n_boot = 10 000` here is higher than the paper's `2 000`; both are reported under their `n_boot` key in the output for transparency.

## Source

`src/shared_circuits/analyses/bootstrap_cis.py` (~240 lines). Pure post-hoc statistics — the only `shared_circuits` imports are `RANDOM_SEED`, `save_results`, and `RESULTS_DIR`. Reads JSONs from `cfg.results_dir`; the legacy one-shot equivalent lives at `/Users/manavpandey/Projects/sycophancy-lying-shared-circuit/experiments/compute_bootstrap_cis.py` (~120 lines) and was the source of the published table CIs. The new analysis adds three things over the legacy script: (1) `--results-glob` / `--source` for scoped re-CI passes, (2) the per-prompt indicator-list bootstrap on top of the count-rate Wilson + bootstrap, and (3) a Pydantic config so it composes with the rest of the analysis CLI surface. Consumed by manual paper-table generation only — nothing downstream reads its JSON.
