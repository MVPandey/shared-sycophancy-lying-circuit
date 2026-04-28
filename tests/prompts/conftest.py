"""Shared fixtures for prompt builder tests."""

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from shared_circuits.prompts.chat import _get_tokenizer


@pytest.fixture(autouse=True)
def _reset_tokenizer_cache() -> Iterator[None]:
    """Drop ``_get_tokenizer``'s lru_cache before and after each test.

    Without this, a real tokenizer pulled in by one test would silently satisfy
    later tests that expect the patched ``AutoTokenizer.from_pretrained``.
    """
    _get_tokenizer.cache_clear()
    yield
    _get_tokenizer.cache_clear()


@pytest.fixture
def stub_tokenizer(mocker: MockerFixture) -> MagicMock:
    """Patch ``AutoTokenizer.from_pretrained`` to a stub that echoes the messages.

    ``apply_chat_template`` returns ``f'<<{json.dumps(messages)}>>'`` so tests can
    assert on substring content without an HF network call.
    """
    tok = MagicMock()

    def _apply(messages: list[dict[str, str]], **_: Any) -> str:
        return f'<<{json.dumps(messages)}>>'

    tok.apply_chat_template.side_effect = _apply
    mocker.patch('shared_circuits.prompts.chat.AutoTokenizer.from_pretrained', return_value=tok)
    return tok
