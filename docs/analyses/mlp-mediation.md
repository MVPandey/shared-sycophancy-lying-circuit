# `mlp-mediation`

> Do upstream MLPs feed the shared-head circuit? Ablate 16 of them and watch both the projection onto `d_syc` and the agree-vs-disagree logit.

This is the targeted causal-mediation test. It asks: when an MLP layer's output gets removed, do the shared heads' contributions to the residual change in the sycophancy direction (mediation hypothesis), or do they keep doing what they were doing while the model's behavior moves through some *other* pathway? The answer is "both, but not in lockstep" — coupling is pervasive, mapping to behavior is multi-pathway.

## The mech-interp idea

A **mediator** in the causal-graph sense is a node whose ablation explains the change in the outcome from ablating an upstream node. Our concrete version: shared heads are the candidate mediator, MLPs are the candidate upstream nodes, behavior is the outcome. Ablate an MLP and measure two things on each held-out sycophancy prompt:

1. **Δproj** — change in the shared heads' aggregated output projected onto `d_syc`, the unit-normalized "wrong-vs-right" residual direction extracted at the first shared-head layer (or `--direction-layer`). This is the "did the MLP feed the shared heads less" channel.

2. **Δlogit_diff** — change in the model's last-token agree-vs-disagree logit gap. The "did behavior move" channel.

If the MLPs feed behavior *through* the shared heads, Δproj and Δlogit_diff should correlate strongly. If they don't, MLPs are reaching the unembedding through pathways that don't run through the shared heads. Per-cell paired-bootstrap 95% CIs (2,000 resamples over prompt indices) tell us whether each individual ablation produces a non-zero shift; the across-MLP Spearman correlation tells us whether the two channels co-move.

The candidate set is auto-selected from the saved [`circuit-overlap`](circuit-overlap.md) result. Let `[first_shared, last_shared]` be the layer span containing shared heads. We pick `n_upstream` MLP layers evenly spaced from layer 0 up to `first_shared − 1`, and `n_in_region` MLP layers evenly spaced across `[first_shared, last_shared]` (both via `np.linspace` then dedup). Defaults: 8 + 8 = 16 candidates.

## Why this design

- **Generalized from the legacy Qwen-72B-only script.** The legacy version hard-coded the layer indices for Qwen2.5-72B; this one auto-selects from the shared-head span so it works on any model. Qwen2.5-72B is what populates the table in the paper, but the analysis runs on anything with a saved [`circuit-overlap`](circuit-overlap.md) result.
- **Upstream / in-region split.** Upstream MLPs (`< first_shared`) test whether MLPs *before* the shared circuit feed it. The "upstream null" — that those MLPs shouldn't matter — is what gets refuted (7/8 upstream MLPs produce Δproj with CI excluding zero on Qwen2.5-72B). In-region MLPs test whether MLPs at or after the shared heads do something different.
- **`np.linspace` over the upstream and in-region spans.** Even spread is cheap and well-defined; clustering candidates in one band would bias the across-MLP Spearman.
- **`d_syc` extracted at the first shared-head layer by default.** That's where the shared signal first appears; later layers get noisier. Override with `--direction-layer`.
- **Direction prompts disjoint from test prompts.** The first `dir_prompts` (default 50) pairs build the direction; the next `test_prompts` (default 100) measure Δproj and Δlogit_diff. Reusing prompts would inflate the projection numbers.
- **Paired-bootstrap CIs over prompt indices, not pair-of-bootstraps.** The same 100 prompts are seen in both baseline and ablated conditions, so the per-prompt difference is the unit. Significance at 95% means the CI excludes zero.
- **Aggregate Spearman across MLPs is the key correlation.** With 16 cells, signed and absolute Spearman are both reported (signed: are large positive Δproj paired with large positive Δlogit_diff; absolute: are large *magnitudes* paired). Upstream-only and full-set versions are both reported.

## How to run it

```bash
# The Qwen2.5-72B headline run
uv run shared-circuits run mlp-mediation \
  --model Qwen/Qwen2.5-72B-Instruct --n-devices 2

# Wider candidate net
uv run shared-circuits run mlp-mediation \
  --model Qwen/Qwen2.5-72B-Instruct --n-devices 2 \
  --n-upstream 16 --n-in-region 16

# Pin the direction layer (rather than first-shared default)
uv run shared-circuits run mlp-mediation \
  --model Qwen/Qwen2.5-72B-Instruct --n-devices 2 --direction-layer 56

# Smaller / faster sanity run on Gemma-2-2B
uv run shared-circuits run mlp-mediation \
  --model gemma-2-2b-it --n-upstream 4 --n-in-region 4 --test-prompts 50
```

Output: `experiments/results/mlp_mediation_<model_slug>.json`. Key fields:

| Field | Meaning |
|---|---|
| `mlp_candidates` | List of MLP layer indices probed |
| `first_shared_layer`, `last_shared_layer` | Bounds of the shared-head span |
| `baseline.{rate,logit_diff_mean,projection_mean}` | No-ablation reference |
| `candidates[L].{position,rate,logit_diff,projection}` | Per-MLP raw values |
| `candidates[L].{rate_delta,logit_diff_delta,logit_diff_ci,projection_delta,projection_ci}` | Per-MLP deltas + 95% CIs |
| `candidates[L].{logit_diff_significant,projection_significant}` | CI excludes zero? |
| `pipeline_test.all_signed_rho`, `all_signed_p` | Across-MLP signed Spearman of Δproj vs Δlogit_diff |
| `pipeline_test.all_abs_rho`, `all_abs_p` | Magnitude version |
| `pipeline_test.upstream_*` | Same correlations restricted to upstream MLPs |
| `pipeline_test.{upstream,in_region}_proj_ci_excludes_zero` | Counts of significant cells |

## Where it lives in the paper

Appendix `mediation`, **Table `tab:mlp:mediation`**. Headline numbers on Qwen2.5-72B (shared-head region layers 50–79, 48 heads):

- 14 of 16 MLPs modulate the shared-head projection (Δproj CI excludes zero) — coupling is pervasive.
- Across-MLP signed Spearman ρ(Δproj, Δlogit_diff) = `−0.21`, `p = 0.43` — projection magnitude does not predict behavioral magnitude. The mapping from MLP→behavior is multi-pathway, not single-channel.
- 7 of 8 upstream MLPs (those *before* the shared-head region) produce Δproj with CI excluding zero — the "upstream null" is refuted.
- Late in-region MLPs (L62, L74, L78) show modest |Δproj| ≤ 0.40 but large Δlogit_diff up to +4.70 — these write directly to the unembedding through pathways that don't run through the shared heads.

This is a null against the **naive feed-forward mediation** hypothesis, not against the shared-circuit hypothesis: the shared heads are causally sufficient (shown elsewhere in the §3.4 grid); they're not the *only* path from MLPs to behavior.

## Source

`src/shared_circuits/analyses/mlp_mediation.py` (~405 lines). Reads per-layer shared heads from a saved [`circuit-overlap`](circuit-overlap.md) JSON via `--shared-heads-from`. Sibling: [`mlp-ablation`](mlp-ablation.md) (the lighter rate-only version, plus its `tugofwar` correlation mode that consumes saved `mlp_ablation` outputs). The paper's main-text MLP roles framework is supported by both.
