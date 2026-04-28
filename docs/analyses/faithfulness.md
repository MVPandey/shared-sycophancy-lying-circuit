# `faithfulness`

> Ablate every attention head, then add the shared heads back one at a time. How few do you need before the model is sycophantic again?

This is the **sufficiency** half of the IOI/ACDC playbook. Necessity asks "what happens when I take the circuit *out*"; sufficiency asks "what happens when the circuit is the *only thing* left". If a tiny number of shared heads alone can drive the full model's sycophantic behavior on a fully-ablated backbone, the shared circuit isn't just a correlate — it's enough.

<p align="center">
  <img src="../img/sufficiency_necessity.png" width="600" alt="Sufficiency-necessity asymmetry: small shared head sets are sufficient at scale, but mean-ablation necessity weakens past 7B.">
</p>

## The mech-interp idea

The IOI/ACDC paradigm (Wang et al. 2023, Conmy et al. 2023) measures circuit faithfulness as the fraction of baseline behavior recovered when you keep only the candidate circuit and ablate everything else. Conmy's ACDC formalism makes this a numeric ratio:

```
faithfulness(K) = (metric_K − metric_all_ablated) / (metric_baseline − metric_all_ablated)
```

`metric_K` is sycophancy rate when *only* the top-K shared heads (ranked by combined importance from [`circuit-overlap`](circuit-overlap.md) or [`attribution-patching`](attribution-patching.md)) are running and every other attention head is zeroed. `metric_all_ablated` zeros all heads. `faithfulness = 1.0` means the K-head subset alone reproduces the full-model sycophancy rate; `> 1.0` means the subset *overshoots* (more sycophantic than the full model — happens at very low baselines where the denominator is tiny).

Two modes:

- **`single`** — evaluate at fixed K values (default `1, 2, 5, 10`, plus the full shared set). Fast.
- **`curve`** — sweep K from 1 to `n_shared` to produce the full sufficiency curve. The right thing for a figure but more expensive.

For each K we report the sycophancy rate with a Wilson 95% CI and the faithfulness ratio. We also compute the smallest K that hits a configurable recovery threshold (`--faithfulness-threshold`, default 0.8 = 80% recovery).

## Why this design

- **Keep-only mask, not progressive ablation.** ACDC's progressive node-removal is a search procedure for finding a circuit; we already *have* the circuit (from [`circuit-overlap`](circuit-overlap.md)). The relevant question is sufficiency, so we ablate everything and add the candidates back. This is what Wang et al. call "minimality" in the IOI paper.
- **Ordered restoration by combined importance.** Heads added back in descending order of combined syc + lie importance. The curve is monotone-ish but not strictly (head 3 might add more than head 4 does on top of head 1+2).
- **Two ranking sources.** `--shared-heads-from circuit_overlap` (default) uses the write-norm DLA ranking; `--shared-heads-from attribution_patching` uses the per-head clean→corrupt patching grids when those are available (≤8B). The two sources can give slightly different rankings.
- **Wilson CIs on rate.** `n = 100` prompts, so binomial-rate uncertainty is non-negligible. Wilson is the standard tight interval for binomial proportions; we use `z = 1.96` for 95%.
- **Faithfulness ratio reported alongside absolute rate.** At low baselines (e.g. Phi-4 at 1%) the ratio inflates mechanically because the denominator is small; the absolute rate shift (Phi-4 K=1 goes 1% → 41%) is the headline. The paper's recommendation is to read both and lead with the rate shift.

## How to run it

```bash
# Full sufficiency curve on Gemma-2-2B (K=2 reaches 1.8× baseline rate)
uv run shared-circuits run faithfulness --model gemma-2-2b-it --mode curve

# Phi-4: a single head out of 1,600 flips the rate by +40pp
uv run shared-circuits run faithfulness --model microsoft/phi-4 --mode curve

# Single-K mode at a fixed list (cheaper)
uv run shared-circuits run faithfulness \
  --model google/gemma-2-9b-it --mode single --k-values 1,2,5

# Use attribution-patching ranking instead of write-norm DLA
uv run shared-circuits run faithfulness \
  --model gemma-2-2b-it --shared-heads-from attribution_patching

# Custom recovery threshold
uv run shared-circuits run faithfulness \
  --model gemma-2-2b-it --faithfulness-threshold 0.5
```

Output: `experiments/results/faithfulness_<model_slug>.json`. Key fields:

| Field | Meaning |
|---|---|
| `mode` | `single` or `curve` |
| `n_shared`, `shared_heads_ranked` | Size and ranking of the candidate set |
| `baseline.{syc_rate,wilson_ci}` | Full-model sycophancy rate |
| `all_ablated.{syc_rate,wilson_ci}` | All-attention-zeroed reference |
| `curve[k].{syc_rate,wilson_ci,faithfulness_ratio,head}` | Per-K row |
| `peak_faithfulness_ratio` | Best K's ratio |
| `first_k_at_threshold` | Smallest K to clear `--faithfulness-threshold` (or `null`) |

## Where it lives in the paper

Appendix `faithfulness`, **Table `tab:faithfulness`**:

| Model | Family | Baseline | n_shared | Peak faith. | First K ≥ 0.8 |
|---|---|---|---|---|---|
| Gemma-2-2B-IT | Gemma | 0.32 | 13 | 1.8× | K=2 |
| Gemma-2-9B-IT | Gemma | 0.10 | 16 | 7.9× | K=1 |
| Gemma-2-27B-IT | Gemma | 0.09 | 26 | 1.1× | K=8 |
| Phi-4 (14B) | Phi | 0.01 | 16 | 41× | K=1 |

The §3.4 main-text reference: "IOI faithfulness curves confirm K=1–2 shared heads recover baseline sycophancy on Gemma-2-2B and Phi-4." On Mistral-7B (not in the table because the binary rate doesn't flip) the logit-diff still moves `+0.56` (from `−1.43` toward `−0.87`) — the heads carry the detection signal but can't on their own cross Mistral's decision boundary.

## Source

`src/shared_circuits/analyses/faithfulness.py` (~290 lines). Reads the ranked shared list from a saved [`circuit-overlap`](circuit-overlap.md) (default) or [`attribution-patching`](attribution-patching.md) JSON. Sibling necessity-side counterparts: [`head-zeroing`](head-zeroing.md) (zero only the shared heads, see what breaks), [`projection-ablation`](projection-ablation.md), [`activation-patching`](activation-patching.md). The full sufficiency-vs-necessity asymmetry across model scale is summarized in the §3.4 figure.
