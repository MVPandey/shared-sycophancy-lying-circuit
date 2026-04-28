"""
Shared experimental constants.

Model-family chat templates have been removed in favor of HuggingFace
``tokenizer.apply_chat_template`` everywhere — see ``shared_circuits.prompts``.
"""

from typing import Final

DEFAULT_BATCH_SIZE: Final = 8
DEFAULT_DEVICE: Final = 'cuda'
DEFAULT_DTYPE: Final = 'bfloat16'
RANDOM_SEED: Final = 42

BOOTSTRAP_ITERATIONS: Final = 500
PERMUTATION_ITERATIONS: Final = 1000
CI_QUANTILES: Final[tuple[float, float]] = (2.5, 97.5)

DEFAULT_N_PROMPTS: Final = 50
DEFAULT_TOP_K: Final = 15

ALL_MODELS: Final[tuple[str, ...]] = (
    'gemma-2-2b-it',
    'google/gemma-2-9b-it',
    'google/gemma-2-27b-it',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen3-8B',
    'meta-llama/Llama-3.1-8B-Instruct',
    'meta-llama/Llama-3.1-70B-Instruct',
    'meta-llama/Llama-3.3-70B-Instruct',
    'microsoft/phi-4',
    'Qwen/Qwen2.5-32B-Instruct',
    'Qwen/Qwen2.5-72B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.1',
    'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'HuggingFaceH4/zephyr-7b-beta',
)

# TransformerLens architecture override: when a model is gated or absent from
# TL's registry, load weights from the value while telling TL the architecture
# is the key.  ``load_model`` also reads the ``WEIGHT_MIRRORS_JSON`` env var.
TL_ARCH_OVERRIDES: Final[dict[str, str]] = {
    'HuggingFaceH4/zephyr-7b-beta': 'mistralai/Mistral-7B-Instruct-v0.1',
}
