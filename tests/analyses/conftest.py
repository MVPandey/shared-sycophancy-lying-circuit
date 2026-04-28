"""Shared fixtures for analysis tests — stub the HF tokenizer everywhere.

Each analysis indirectly imports a prompt builder which calls
``shared_circuits.prompts.chat._get_tokenizer``, which would otherwise reach
HuggingFace Hub for a real tokenizer (and fail on gated repos like gemma in CI).
"""

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from shared_circuits.prompts.chat import _get_tokenizer


@pytest.fixture(autouse=True)
def _stub_chat_tokenizer(mocker: MockerFixture) -> Iterator[None]:
    """Replace the cached HF tokenizer with a stub that echoes ``messages``."""
    _get_tokenizer.cache_clear()
    tok = MagicMock()

    def _apply(messages: list[dict[str, str]], **_: Any) -> str:
        return f'<<{json.dumps(messages)}>>'

    tok.apply_chat_template.side_effect = _apply
    mocker.patch('shared_circuits.prompts.chat.AutoTokenizer.from_pretrained', return_value=tok)
    yield
    _get_tokenizer.cache_clear()
