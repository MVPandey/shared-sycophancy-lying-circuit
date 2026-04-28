"""HuggingFace chat-template helpers shared by every prompt builder."""

import functools
from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase


@functools.cache
def _get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Return a cached HuggingFace tokenizer for ``model_name``."""
    return cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(model_name))


def render_chat(messages: list[dict[str, str]], model_name: str) -> str:
    """Format ``messages`` with the model's official chat template, ready for next-token generation."""
    tok = _get_tokenizer(model_name)
    return str(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
