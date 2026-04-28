# Shared Sycophancy-Lying Circuit

Code for the paper **"LLMs Know They're Wrong and Agree Anyway: The Shared Sycophancy-Lying Circuit"**.

When a language model sycophantically agrees with a user's false belief, is it failing to detect the error, or noticing and agreeing anyway? This codebase shows the second. Across twelve open-weight models from five labs (1.5B–72B), the same small set of attention heads carries a "this statement is wrong" signal whether the model is evaluating an isolated claim or being pressured to agree with a user. The library packages the mechanistic-interpretability toolkit used to identify, validate, and intervene on that circuit, plus a small CLI for running the five reference analyses end-to-end.

---

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | **3.13.9** | Pinned via `.python-version`. 3.13.8 has a `torch._jit_internal._overload_method` AST regression that breaks `import torch`; 3.13.9 fixes it. |
| `uv` | ≥ 0.9 | Dependency + virtualenv manager. `brew install uv`. |
| GPU | recommended | All real analyses load a TransformerLens `HookedTransformer`. CPU works for the unit tests (mocked) but not for any `shared-circuits run` command on a real model. |
| GPU memory | 24–80 GB | 7B models fit on a single 24 GB GPU; 32B and 72B require pipeline-parallel splitting via `--n-devices`. |
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
uv run pytest                                     # 213 tests, ~3s
uv run shared-circuits run --help                 # CLI surface

# real run on a 2B model (needs HF auth + ~16 GB GPU)
uv run shared-circuits run circuit-overlap --models gemma-2-2b-it --n-prompts 50
```

Analysis results are written to `experiments/results/<analysis_name>_<model_slug>.json` (the directory is created on first save). Multi-GPU pipeline parallelism is opt-in via `--n-devices` on the `breadth` and `path-patching` analyses.

---

## Repository layout

```
src/shared_circuits/
├── analyses/                 # the five reference analyses
│   ├── circuit_overlap.py    # DLA-based head overlap (sycophancy ↔ lying)
│   ├── causal_ablation.py    # zero-out shared heads, measure probe-AUROC drop
│   ├── attribution_patching.py
│   ├── breadth.py            # 12-model panel: head overlap + behavioral steering
│   └── path_patching.py      # edge-level path patching across paradigms
├── attribution/              # DLA + attribution-patching primitives
├── cli.py                    # `shared-circuits` argparse dispatcher
├── config.py                 # constants only (BATCH_SIZE, SEED, ALL_MODELS, ...)
├── data/                     # TriviaQA loader + opinion-pair generator
├── experiment/               # model_session context manager + JSON results I/O
├── extraction/               # BatchedExtractor + residual / ablation helpers
├── models/                   # TransformerLens loader + parallelism + agree/disagree tokens
├── prompts/                  # chat-template-based prompt builders for each paradigm
└── stats/                    # geometry / significance / bootstrap / correlation / probes
tests/                        # mirrors src/ structure; 213 tests, 99% coverage
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
| `circuit-overlap` | multi (`--models`) | `--n-prompts`, `--n-pairs` |
| `causal-ablation` | multi (`--models`) | `--shared-heads-from`, `--shared-heads-k`, `--probe-layer-frac`, `--n-random-heads` |
| `attribution-patching` | multi (`--models`) | `--n-pairs`, `--n-patch-pairs`, `--overlap-k` |
| `breadth` | single (`--model` required) | `--n-devices`, `--alphas`, `--layer-fracs`, `--steer-prompts`, `--permutations` |
| `path-patching` | single (`--model` required) | `--task`, `--shared-source`, `--max-sources`, `--no-head-edges`, `--prefill-shift`, `--n-boot` |

Every analysis exposes a Pydantic config (`CircuitOverlapConfig`, `BreadthConfig`, …); the CLI is a thin argparse layer that builds the config from flags and calls `analysis.run(cfg)`. Programmatic use is identical:

```python
from shared_circuits.analyses.circuit_overlap import CircuitOverlapConfig, run

cfg = CircuitOverlapConfig(models=('gemma-2-2b-it', 'Qwen/Qwen3-8B'), n_prompts=50)
results = run(cfg)   # list of per-model result dicts; also persisted as JSON
```

`shared_circuits.experiment.model_session(name)` is the canonical way to load a model; it owns lifecycle (load → yield `ExperimentContext` → cleanup on exit, even on exception) and exposes the model, metadata, and precomputed agree/disagree token IDs in one frozen value object.

---

## Code quality

```bash
make check                                  # full sweep
uv run ruff check --fix src/ tests/         # lint + auto-fix
uv run ruff format src/ tests/              # format
uv run ty check --project . src/ tests/     # type check
uv run pytest                               # 95% coverage gate
```

Lockfile and toolchain config live in `pyproject.toml`. The coverage gate omits the GPU-bound model loader, the pipeline-parallel helper, the CLI dispatcher, and the two heaviest analyses (`breadth.py`, `path_patching.py`) — their public configs / CLI surfaces are still tested, but the per-edge / per-layer guts only run end-to-end with a real model.

---

## Notes

- **Legacy experiment scripts** — the 42 not-yet-migrated `experiments/run_*.py` scripts from the original codebase are kept locally for migration reference but are git-ignored. The five flagship analyses listed above replace `run_circuit_overlap.py`, `run_causal_ablation.py`, `run_attribution_patching.py`, `run_breadth.py`, and `run_path_patching.py`. The remaining 37 will be folded into `shared_circuits.analyses` over time.
- **Weight mirrors** — `WEIGHT_MIRRORS_JSON='{"original-repo": "mirror-repo"}'` makes `load_model` fetch weights from the mirror while still telling TransformerLens the architecture is `original-repo`. Useful when an official HF repo is gated and a public mirror exists.
- **Reproducibility** — all randomness routes through `RANDOM_SEED` in `config.py` and per-analysis `--seed` flags. Bootstrap CIs use `BOOTSTRAP_ITERATIONS = 500`; permutation tests use `PERMUTATION_ITERATIONS = 1000`.
