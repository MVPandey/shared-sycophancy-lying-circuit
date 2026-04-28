.PHONY: check lint format typecheck test \
        mixtral-all qwen32b-full qwen72b-pipeline \
        rlhf-natural-experiment zephyr-natural-experiment

check: lint format typecheck test

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format --check src/ tests/

typecheck:
	uv run ty check --project . src/ tests/

test:
	uv run pytest tests/

# ---------------------------------------------------------------------------
# Per-model orchestration recipes (replacements for legacy run_<model>_*.py).
# Each composes existing CLI subcommands; override flags via the env if needed.
# ---------------------------------------------------------------------------

MIXTRAL ?= mistralai/Mixtral-8x7B-Instruct-v0.1
QWEN32B ?= Qwen/Qwen2.5-32B-Instruct
QWEN72B ?= Qwen/Qwen2.5-72B-Instruct
LLAMA31_70B ?= meta-llama/Llama-3.1-70B-Instruct
LLAMA33_70B ?= meta-llama/Llama-3.3-70B-Instruct
MISTRAL_7B ?= mistralai/Mistral-7B-Instruct-v0.1
ZEPHYR_7B ?= HuggingFaceH4/zephyr-7b-beta

mixtral-all:
	uv run shared-circuits run circuit-overlap --models $(MIXTRAL)
	uv run shared-circuits run path-patching --model $(MIXTRAL) --task instructed_lying
	uv run shared-circuits run norm-matched --model $(MIXTRAL)
	uv run shared-circuits run breadth --model $(MIXTRAL)

qwen32b-full:
	uv run shared-circuits run circuit-overlap --models $(QWEN32B)
	uv run shared-circuits run direction-analysis --models $(QWEN32B)
	uv run shared-circuits run head-zeroing --model $(QWEN32B)
	uv run shared-circuits run projection-ablation --model $(QWEN32B)
	uv run shared-circuits run activation-patching --model $(QWEN32B)
	uv run shared-circuits run breadth --model $(QWEN32B)

qwen72b-pipeline:
	uv run shared-circuits run circuit-overlap --models $(QWEN72B)
	uv run shared-circuits run breadth --model $(QWEN72B) --n-devices 2
	uv run shared-circuits run mlp-mediation --model $(QWEN72B) --n-devices 2

rlhf-natural-experiment:
	uv run shared-circuits run circuit-overlap --models $(LLAMA31_70B) $(LLAMA33_70B)
	uv run shared-circuits run projection-ablation --model $(LLAMA31_70B) --n-devices 2
	uv run shared-circuits run projection-ablation --model $(LLAMA33_70B) --n-devices 2

zephyr-natural-experiment:
	uv run shared-circuits run circuit-overlap --models $(MISTRAL_7B) $(ZEPHYR_7B)
	uv run shared-circuits run direction-analysis --models $(MISTRAL_7B) $(ZEPHYR_7B)
