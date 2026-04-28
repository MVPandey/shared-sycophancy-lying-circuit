# Shared Sycophancy-Lying Circuit

Code for the paper **"LLMs Know They're Wrong and Agree Anyway: The Shared Sycophancy-Lying Circuit"**.

When a language model sycophantically agrees with a user's false belief, is it failing to detect the error, or noticing and agreeing anyway? This codebase shows the second. Across twelve open-weight models from five labs (1.5B–72B), the same small set of attention heads carries a "this statement is wrong" signal whether the model is evaluating an isolated claim or being pressured to agree with a user. The library packages the mechanistic-interpretability toolkit used to identify, validate, and intervene on that circuit, plus a small CLI for running the five reference analyses end-to-end.

---

## Requirements

| Requirement | Version                   | Notes |
|---|---------------------------|---|
| Python | **3.13.9**                | Pinned via `.python-version`. 3.13.8 has a `torch._jit_internal._overload_method` AST regression that breaks `import torch`; 3.13.9 fixes it. |
| `uv` | ≥ 0.9                     | Dependency + virtualenv manager. `brew install uv`. |
| GPU | recommended               | All real analyses load a TransformerLens `HookedTransformer`. CPU works for the unit tests (mocked) but not for any `shared-circuits run` command on a real model. |
| GPU memory | 24–141 GB                 | 7B models fit on a single 24 GB GPU; 32B and 72B require pipeline-parallel splitting via `--n-devices`. |
| HuggingFace token | required for gated models | `hf auth login` before pulling Gemma, Llama, etc. The library reads `WEIGHT_MIRRORS_JSON` (`{model_name: mirror_repo}`) for fallbacks when the official repo is gated. |

### Runtime dependencies

`torch >= 2.8`, `transformer-lens >= 2.0`, `transformers`, `datasets >= 3.0`, `numpy >= 2.0`, `scipy >= 1.14`, `scikit-learn >= 1.5`, `tqdm >= 4.66`, `pydantic >= 2.0`. Full list in `pyproject.toml`; resolved lockfile in `uv.lock`.

### Dev dependencies

`pytest >= 9`, `pytest-mock`, `pytest-cov`, `ruff >= 0.14`, `ty` (Astral's type checker), `matplotlib >= 3.9`. All installed via `uv sync --group dev`.

---

## Quick start

```bash
git clone git@github.com:MVPandey/shared-sycophancy-lying-circuit.git
cd shared-sycophancy-lying-circuit
uv sync --group dev
uv run pytest                                     # 484 tests, ~3s
uv run shared-circuits run --help                 # 18 analyses, full surface

# real run on a 2B model (needs HF auth + ~16 GB GPU)
uv run shared-circuits run circuit-overlap --models gemma-2-2b-it --n-prompts 50
```

Analysis results are written to `experiments/results/<analysis_name>_<model_slug>.json` (the directory is created on first save). Multi-GPU pipeline parallelism is opt-in via `--n-devices` on the `breadth` and `path-patching` analyses.

---

## Repository layout

```
src/shared_circuits/
├── analyses/                 # 18 CLI-runnable analyses (each = Pydantic config + run + add_cli_args + from_args)
│   ├── circuit_overlap.py            # DLA-based head overlap (sycophancy ↔ lying)
│   ├── causal_ablation.py            # zero-out shared heads, measure probe-AUROC drop
│   ├── attribution_patching.py       # per-head clean→corrupt patching (Wang et al.)
│   ├── breadth.py                    # 12-model panel: head overlap + behavioral steering
│   ├── path_patching.py              # edge-level path patching across paradigms
│   ├── direction_analysis.py         # per-head + per-layer cosine; permutation null
│   ├── probe_transfer.py             # logistic-regression probe syc→lie + lie→syc
│   ├── triple_intersection.py        # opinion ∩ syc ∩ lie top-K head intersection
│   ├── layer_strat_null.py           # layer-stratified permutation null on head overlap
│   ├── head_zeroing.py               # full / top-K / matched-importance / norm-matched
│   ├── projection_ablation.py        # ablate d_syc at sweep of layers
│   ├── reverse_projection.py         # cross-task coupling test (d_lie removal on syc, etc.)
│   ├── norm_matched.py               # write-norm-matched random control
│   ├── faithfulness.py               # IOI/ACDC sufficiency curve
│   ├── logit_lens.py                 # per-layer DIFF trajectory (Halawi-style)
│   ├── nq_replication.py             # NaturalQuestions cross-dataset replication
│   ├── opinion_causal.py             # opinion ∩ shared head zeroing + direction-cosine boundary
│   └── steering.py                   # dose-response of d_syc at chosen layer
├── attribution/              # DLA + attribution-patching primitives
├── cli.py                    # `shared-circuits` argparse dispatcher
├── config.py                 # constants only (BATCH_SIZE, SEED, ALL_MODELS, ...)
├── data/                     # TriviaQA + NaturalQuestions loaders + opinion-pair generator
├── experiment/               # model_session context manager + JSON results I/O
├── extraction/               # BatchedExtractor + residual / ablation / agree-rate helpers
├── models/                   # TransformerLens loader + parallelism + agree/disagree tokens
├── prompts/                  # chat-template-based prompt builders for each paradigm
└── stats/                    # geometry / significance / bootstrap / correlation / probes
tests/                        # mirrors src/ structure; 484 tests, 96% coverage
pyproject.toml                # project + tooling config (single source of truth)
Makefile                      # `make check` runs lint + format + typecheck + tests
```

Every prompt builder routes through `tokenizer.apply_chat_template(...)` against a per-model HuggingFace tokenizer (cached via `functools.cache`), so the library works with any HF model that ships a chat template — no per-family lookup tables to maintain.

---

## CLI

```bash
shared-circuits run <analysis> [analysis-specific flags]
```

| Slug | Single- or multi-model | Key flags |
|---|---|---|
| `attribution-patching` | multi (`--models`) | `--n-pairs`, `--n-patch-pairs`, `--overlap-k` |
| `breadth` | single (`--model` required) | `--n-devices`, `--alphas`, `--layer-fracs`, `--steer-prompts`, `--permutations` |
| `causal-ablation` | multi (`--models`) | `--shared-heads-from`, `--shared-heads-k`, `--probe-layer-frac`, `--n-random-heads` |
| `circuit-overlap` | multi (`--models`) | `--n-prompts`, `--n-pairs` |
| `direction-analysis` | multi (`--models`) | `--n-pairs`, `--n-prompts`, `--n-permutations`, `--seed` |
| `faithfulness` | single (`--model` required) | `--mode {single,curve}`, `--k-values`, `--shared-heads-from`, `--shared-heads-k`, `--faithfulness-threshold` |
| `head-zeroing` | single (`--model` required) | `--mode {c2_matched,full_shared,mean_shared,top_n_combined}`, `--shared-heads-from`, `--shared-heads-k`, `--n-boot` |
| `layer-strat-null` | multi (`--models`) | `--n-permutations`, `--grids-from`, `--k`, `--seed` |
| `logit-lens` | single (`--model` required) | `--n-pairs`, `--n-perm`, `--n-boot`, `--seed` |
| `norm-matched` | single (`--model` required) | `--n-prompts`, `--shared-heads-from`, `--seed` |
| `nq-replication` | multi (`--models`) | `--n-pairs`, `--dla-prompts`, `--triviaqa-grids-from`, `--permutations` |
| `opinion-causal` | multi (`--models`) | `--mode {causal,boundary}`, `--n-opinion`, `--n-factual`, `--triple-from`, `--triple-k` |
| `path-patching` | single (`--model` required) | `--task`, `--shared-source`, `--max-sources`, `--no-head-edges`, `--prefill-shift`, `--n-boot` |
| `probe-transfer` | multi (`--models`) or single (`--single-model`) | `--probe-layer-frac`, `--probe-layer`, `--weight-repo`, `--tag`, `--n-boot` |
| `projection-ablation` | single (`--model` required) | `--layer`, `--layer-fracs`, `--qwen3-layer-sweep`, `--dir-prompts`, `--test-prompts`, `--n-boot` |
| `reverse-projection` | single (`--model` required) | `--lying-task {instructed_lying,scaffolded_lying,repe_lying}`, `--from-direction-file`, `--weight-repo` |
| `steering` | single (`--model` required) | `--alphas`, `--layer-frac`, `--layer`, `--test-prompts`, `--dir-prompts` |
| `triple-intersection` | multi (`--models`) | `--n-permutations`, `--factual-from`, `--opinion-from`, `--seed` |

Every analysis exposes a Pydantic config (`CircuitOverlapConfig`, `BreadthConfig`, …); the CLI is a thin argparse layer that builds the config from flags and calls `analysis.run(cfg)`. Programmatic use is identical:

```python
from shared_circuits.analyses.circuit_overlap import CircuitOverlapConfig, run

cfg = CircuitOverlapConfig(models=('gemma-2-2b-it', 'Qwen/Qwen3-8B'), n_prompts=50)
results = run(cfg)   # list of per-model result dicts; also persisted as JSON
```

`shared_circuits.experiment.model_session(name)` is the canonical way to load a model; it owns lifecycle (load → yield `ExperimentContext` → cleanup on exit, even on exception) and exposes the model, metadata, and precomputed agree/disagree token IDs in one frozen value object.

---

## Experiment coverage by model

The paper analyzes thirteen open-weight checkpoints across five labs (Google, Alibaba, Meta, Mistral AI, Microsoft) at 1.5B–72B parameters; coverage per model is a deliberate compute tradeoff documented in Appendix A (Scope and extensibility). The table below lists, per model, every analysis run on it in the paper. **Migration status:** ✓ marks an analysis already callable via `shared-circuits run <slug>` from `src/shared_circuits/analyses/`; △ marks a legacy script in the source repo's `experiments/` directory awaiting migration into the same package (the legacy directory is git-ignored locally — clone the source repo for the originals).

| Model | Params | Heads (K) | Lab | Experiments run on this model |
|---|---|---|---|---|
| Gemma-2-2B-IT | 2B | 208 (15) | Google | head overlap ✓, layer-strat null ✓, instructed-lying overlap △, path patching ✓ (Instr 88×, Fact 355×), mean-ablation / head zeroing ✓, projection ablation ✓, activation patching △, per-head activation patching ✓, causal-ablation probe ✓, norm-matched control ✓ (margin 6.4×), faithfulness K=2 ✓, direction analysis ✓ (mean cos 0.81), probe transfer ✓ (AUROC 0.83), reverse projection ablation ✓, behavioral steering ✓, opinion circuit transfer △, opinion-causal head zeroing ✓, SAE feature overlap △ (L12, L19), logit-lens ✓ (peak +127%), NQ cross-dataset replication ✓ (ρ≈0.99), anti-sycophancy DPO △ (52%→28%), breadth panel ✓ |
| Gemma-2-9B-IT | 9B | 672 (26) | Google | head overlap ✓, layer-strat null ✓, instructed-lying overlap △ (7/26), path patching ✓ (Instr 7×, Syc 4×), causal suite △, faithfulness K=1 ✓, SAE feature overlap △ (L21, L31), breadth panel ✓ |
| Gemma-2-27B-IT | 27B | 1,472 (39) | Google | head overlap ✓, layer-strat null ✓, causal suite △, faithfulness K=8 ✓, projection ablation ✓ (10.5%→100% syc), breadth panel ✓ |
| Qwen2.5-1.5B-Instruct | 1.5B | 336 (19) | Alibaba | head overlap ✓, layer-strat null ✓, per-head activation patching ✓, direction analysis ✓ (mean cos 0.55), probe transfer ✓ (AUROC 0.61, Ying floor), breadth panel ✓ |
| Qwen2.5-32B-Instruct | 32B | 2,560 (51) | Alibaba | head overlap ✓ (split-half r=0.87), direction analysis ✓ (mean cos 0.52), full causal suite (custom wrapper) △, breadth panel ✓ |
| Qwen2.5-72B-Instruct | 72B | 5,120 (72) | Alibaba | head overlap ✓, layer-strat null ✓, breadth panel ✓, behavioral steering ✓, MLP mediation test △ (16-MLP, 8 upstream + 8 in-region), MLP ablation tug-of-war △ |
| Qwen3-8B | 8B | 1,152 (34) | Alibaba | head overlap ✓, layer-strat null ✓, instructed-lying overlap △ (25/34), path patching ✓ (Instr 10×), causal suite △, projection ablation (Qwen3-specific) ✓, norm-matched control ✓ (117× margin), direction analysis ✓ (mean cos 0.43), probe transfer ✓ (AUROC 0.85), opinion circuit transfer △, opinion-causal head zeroing ✓, breadth panel ✓ |
| Llama-3.1-8B-Instruct | 8B | 1,024 (32) | Meta | head overlap ✓, per-head activation patching ✓, direction analysis ✓ (mean cos 0.44), SAE feature overlap △ (L19, 41/100 features), SAE K-sensitivity △, SAE sentiment control △, linear-probe SAE alignment △, breadth panel ✓ |
| Llama-3.1-70B-Instruct | 70B | 5,120 (72) | Meta | head overlap ✓, layer-strat null ✓, path patching ✓ (Instr 3.6×), causal suite △ (mean-abl null at 1.1% head fraction), logit-lens ✓ (monotonic, peak excess 0%), opinion-causal head zeroing ✓, RLHF natural experiment vs. 3.3 (substrate persists, syc 39%→3.5%), breadth panel ✓ |
| Llama-3.3-70B-Instruct | 70B | 5,120 (72) | Meta | head overlap ✓, layer-strat null ✓, instructed-lying overlap △ (26/72), path patching ✓ (Instr 6×, Fact 1,732×, Syc 2,248×), causal suite △ (low-baseline ceiling for mean-abl), projection ablation ✓ (+27pp), norm-matched control ✓ (27× margin), SAE feature overlap △ (L50, 36/100), opinion-causal head zeroing ✓, NQ replication ✓ (ρ≈0.99), RLHF refresh comparison vs. 3.1, breadth panel ✓ |
| Mistral-7B-Instruct-v0.1 | 7B | 1,024 (32) | Mistral AI | head overlap ✓, path patching ✓ (Instr 11×, Syc 22×), causal suite △ (sufficient AND necessary at 7B), norm-matched control ✓, faithfulness ✓ (logit-diff shift only), probe transfer ✓ (AUROC 0.84), reverse projection ablation ✓, logit-lens ✓ (peak +89%), anti-sycophancy DPO △ (28%→2%), Mistral→Zephyr RLHF refresh comparison, breadth panel ✓ |
| Mixtral-8x7B-Instruct-v0.1 | 47B (~13B active, sparse MoE) | 1,024 (32) | Mistral AI | head overlap ✓, instructed-lying overlap △ (20/32, ρ=0.93), path patching ✓ (Instr 4×), norm-matched control ✓ (5.49 margin, 2.8× — first MoE validation), `run_mixtral_all.py` (single-load multi-experiment wrapper) △, breadth panel ✓ |
| Phi-4 | 14B | 1,600 (40) | Microsoft | head overlap ✓, instructed-lying overlap △ (10/40), path patching ✓ (Instr 540× — cross-lab, cross-architecture replication), faithfulness K=1 ✓ (one head flips +40pp), norm-matched control ✓ (opposite-sign margin), direction analysis ✓ (mean cos 0.56), breadth panel ✓ |

**Auxiliary checkpoints mentioned for context (no standalone analysis suite):**
- **Zephyr-7B-β** (HuggingFaceH4 DPO of Mistral-7B): Mistral→Zephyr DPO refresh comparison; head-importance Spearman 0.846→0.848, sycophancy amplifies 3.6× — uses `run_circuit_overlap.py` outputs from both models, no new analysis.
- **Gemma-3-27B-IT**: documented in Appendix as a dissociation case (layer-0 write-norm inflation ~100× other layers); excluded from per-head circuit analysis but residual-stream cosine 0.494 retained. Uses `run_direction_analysis.py`.

---

## Cross-cutting / shared-protocol experiments

These analyses don't sit on a single model — they are pipelines (training, dataset construction) or post-hoc analyses applied across multiple model results.

| Workflow | Script(s) | What it produces |
|---|---|---|
| **Anti-sycophancy DPO preference dataset** | `build_antisyc_dpo_dataset.py` △ | 1,000 TriviaQA preference pairs balanced across wrong-opinion + right-opinion templates; n=100 held-out eval. Indices 500–1,499 train / 1,500–1,549 eval, disjoint from the 0–400 probe-transfer slice. |
| **Sham (placebo) DPO dataset** | `build_sham_dpo_dataset.py` △ | Same prompts as anti-syc, chosen/rejected responses scrambled per pair (seed=42) — controls for "any DPO training perturbs the probe". |
| **Anti-sycophancy DPO training** | `run_dpo_antisyc.py` △ | LoRA r=16, α=32, dropout 0.05 on `q_proj`+`v_proj`; β=0.1, lr 5e-5, 2 epochs, batch 8 effective, bf16, ~30–60 min/run on a single 96 GB GPU. Run on Mistral-7B-Instruct-v0.1 + Gemma-2-2B-IT, with sham control on each. Adapters merged into base weights post-training. |
| **Llama-3.1→3.3-70B RLHF natural experiment** | `run_circuit_overlap.py` ✓ + `run_projection_ablation.py` ✓ + `check_syc_behavioral.py` (deleted as dead code in this repo's first commit) | Compares head overlap, projection-ablation effect, and behavioral sycophancy rate between the same base weights pre/post-Meta's post-training refresh. Sycophancy 39%→3.5%, shared fraction 0.79→0.71, projection-ablation effect +10.5pp→+27pp. |
| **Mistral→Zephyr-7B DPO natural experiment** | `run_circuit_overlap.py` ✓ + `run_direction_analysis.py` ✓ | Independent-family replication of the RLHF refresh pattern at 7B. Head-importance Spearman 0.846→0.848, sycophancy amplifies 3.6×. |
| **Bootstrap confidence intervals** | `compute_bootstrap_cis.py` △ | Post-hoc 95% paired bootstrap (2,000 resamples) over already-computed numerical results — applied to every reported rate, edge restoration ratio, and AUROC in tables. |
| **K-sensitivity sweep** | `run_k_sensitivity.py` △ + `run_sae_k_curve.py` △ | Re-runs head overlap (or SAE feature overlap) at K∈{5,10,15,20,30,50}; SAE version sweeps K∈{10,50,100,200,500}. Confirms results aren't a threshold artifact. |
| **Triple-intersection analysis** | `analyze_triple_intersection.py` ✓ | Sycophancy ∩ lying ∩ opinion top-K head intersection across five models, with permutation null (51–1,755× chance). |
| **Layer-stratified null** | `analyze_layer_stratified_null.py` ✓ | Stricter permutation null that shuffles labels within each layer (preserves per-layer marginals). Eight models tested, all p<10⁻⁴. |
| **MLP tug-of-war prediction analysis** | `analyze_tugofwar_prediction.py` △ | Per-layer correlation between shared-head ablation and MLP ablation effects on Qwen2.5-72B; tests the upstream-MLP→shared-heads pipeline hypothesis. |

---

## Replication status

**Migrated to `src/shared_circuits/analyses/` (18 CLI-runnable analyses today via `shared-circuits run <slug>`):**

| Slug | Source claim |
|---|---|
| `circuit-overlap` | Head-level overlap (DLA write-norm) with hypergeometric + permutation nulls — Table 1 (12-model panel) |
| `causal-ablation` | Head-zeroing with probe-AUROC drop and direction-cosine reporting |
| `attribution-patching` | Per-head clean→corrupt patching (≤8B per-head; top-K shared-set at scale) — Appendix H |
| `breadth` | 12-model breadth runner (head overlap + behavioral steering for one model) |
| `path-patching` | Edge-level path patching across sycophancy / factual lying / instructed lying — §3.3, Table 3 |
| `direction-analysis` | Per-head + per-layer cosine between `d_syc` and `d_lie`; permutation null — Appendix M |
| `probe-transfer` | LR probe syc→lie + (single-model) lie→syc — Appendix probe-ci |
| `triple-intersection` | Opinion ∩ syc ∩ lie top-K head intersection — §3.6, Figure 3a |
| `layer-strat-null` | Layer-stratified permutation null on head overlap — Appendix layerstrat |
| `head-zeroing` | Mean-ablation / full-shared / top-K / norm-matched modes — §3.4 causal suite |
| `projection-ablation` | Ablate `d_syc` at sweep of layers — main causal claim at 70B |
| `reverse-projection` | Cross-task coupling: ablate `d_lie` on syc, ablate `d_syc` on lie — §4 anti-syc DPO substrate |
| `norm-matched` | Write-norm-matched random control — Appendix normmatch |
| `faithfulness` | IOI/ACDC sufficiency curve — Appendix faithfulness |
| `logit-lens` | Per-layer DIFF trajectory (Halawi-style detect-then-override) — Appendix logitlens |
| `nq-replication` | NaturalQuestions cross-dataset replication — Appendix nq |
| `opinion-causal` | Opinion ∩ shared head zeroing + direction-cosine boundary — §3.6, Appendix opinion-null |
| `steering` | Dose-response sweep of `d_syc` at chosen layer — Appendix M |

**Awaiting migration (~24 legacy scripts in source repo's `experiments/`):**

The remaining analyses needed for full paper replication, grouped by what's blocking each:

- **DPO family (3 scripts) — needs `trl` + `peft` deps:** `build_antisyc_dpo_dataset.py`, `build_sham_dpo_dataset.py`, `run_dpo_antisyc.py`. Carries the §4 anti-sycophancy DPO claim (Mistral 28%→2%, Gemma 52%→28%) and the sham-DPO equivalence-margin control.
- **SAE family (4 scripts) — needs `huggingface_hub.hf_hub_download` + a `data/sae_features.py` loader + an `extraction/sae.py` projector:** `run_sae_feature_overlap.py`, `run_sae_k_curve.py`, `run_sae_sentiment_control.py`, `run_linear_probe_sae_alignment.py`. Carries Table 4 (Gemma-Scope + Goodfire SAE feature overlap) and the Llama-3.1-8B sentiment-control corollary.
- **Activation patching at scale (1 script):** `run_activation_patching.py` — the shared-set top-K activation patching used in the §3.4 causal suite at ≥32B. Distinct from `attribution-patching` (per-head Wang clean→corrupt). Library primitives sufficient — needs orchestration only.
- **MLP analyses (3 scripts):** `run_mlp_ablation.py`, `run_mlp_disruption_control.py`, `analyze_tugofwar_prediction.py`. The 16-layer MLP mediation test on Qwen2.5-72B and the per-layer tug-of-war prediction.
- **Per-model wrappers (3 scripts):** `run_mixtral_all.py` (Mixtral single-load multi-experiment), `run_qwen32b_full.py` (Qwen2.5-32B causal suite), `run_qwen72b_pipeline.py` + `run_qwen72b_breadth.py` (Qwen2.5-72B). These can be expressed as bash composition of the migrated analyses.
- **Misc (4 scripts):** `run_dla_instructed_lying.py` (instructed-lying head ranking; `circuit-overlap` covers most of this — could be a `--paradigm instructed_lying` flag on `circuit-overlap`), `run_opinion_circuit_transfer.py` (opinion ∩ syc ∩ lie head ranking with permutation null — overlap with `triple-intersection`), `run_probe_transfer.py` (already covered by `probe-transfer`), `compute_bootstrap_cis.py` (post-hoc — could become a library helper rather than a separate analysis), `run_k_sensitivity.py` (now subsumed by `layer-strat-null --k`).

A coverage matrix mirroring the per-model table above is reproduced in Appendix C of the paper (`overleaf/sections/appendix.tex`) — both should stay in sync as more analyses migrate.

---

## Code quality

```bash
make check                                  # full sweep
uv run ruff check --fix src/ tests/         # lint + auto-fix (PLC0415 enforces top-of-file imports)
uv run ruff format src/ tests/              # format
uv run ty check --project . src/ tests/     # type check
uv run pytest                               # 95% coverage gate
```

Lockfile and toolchain config live in `pyproject.toml`. The coverage gate omits the GPU-bound model loader, the pipeline-parallel helper, the CLI dispatcher, and the two heaviest analyses (`breadth.py`, `path_patching.py`) — their public configs / CLI surfaces are still tested, but the per-edge / per-layer guts only run end-to-end with a real model.

---

## Notes

- **Weight mirrors** — `WEIGHT_MIRRORS_JSON='{"original-repo": "mirror-repo"}'` makes `load_model` fetch weights from the mirror while still telling TransformerLens the architecture is `original-repo`. Useful when an official HF repo is gated and a public mirror exists.
- **Reproducibility** — all randomness routes through `RANDOM_SEED` in `config.py` and per-analysis `--seed` flags. Bootstrap CIs use `BOOTSTRAP_ITERATIONS = 500`; permutation tests use `PERMUTATION_ITERATIONS = 1000`.
- **Hardware used in the paper** — models up to 32B ran on a single NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM); 70B and 72B used a two-GPU node (192 GB aggregate). All forward passes use bfloat16; direction and cosine statistics accumulate in float32 for numerical stability. Decoding is greedy throughout.
- **What was deliberately not run** — full per-head activation patching at ≥32B (>50 GPU-hours/model, redundant given the lower-cost top-K shared-set substitute); direction-level analyses at Gemma-2-9B / 27B / Llama-3.3-70B / Qwen2.5-72B (head-level evidence is the primary claim at scale); opinion-causal replication beyond Gemma, Qwen, Llama (next-most-informative follow-up). Documented in full in Appendix A.
