"""Shared test fixtures with mocked GPU/model dependencies."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


@dataclass
class MockModelConfig:
    model_name: str = 'gemma-2-2b-it'
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 32
    d_head: int = 8
    device: str = 'cpu'


class MockTokenizer:
    pad_token_id = 0

    def decode(self, token_ids: list[int]) -> str:
        return 'yes, that is correct'

    def __call__(self, text: str, **kwargs: object) -> dict:
        return {'input_ids': [1, 2, 3]}


class MockHookedTransformer:
    """Lightweight mock of TransformerLens HookedTransformer."""

    def __init__(self, cfg: MockModelConfig | None = None) -> None:
        self.cfg = cfg or MockModelConfig()
        self.tokenizer = MockTokenizer()

        # create mock attention blocks with W_O
        self.blocks = []
        for _ in range(self.cfg.n_layers):
            block = MagicMock()
            block.attn.W_O = torch.randn(self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model)
            self.blocks.append(block)

    def to_tokens(self, prompts: str | list[str], prepend_bos: bool = True) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        # return fake tokens of length 10
        return torch.randint(1, 100, (len(prompts), 10))

    def run_with_hooks(
        self,
        tokens: torch.Tensor,
        fwd_hooks: list | None = None,
        stop_at_layer: int | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = tokens.shape

        # fire hooks with fake activations
        if fwd_hooks:
            for hook_name, hook_fn in fwd_hooks:
                if 'hook_resid_post' in hook_name:
                    fake_act = torch.randn(batch_size, seq_len, self.cfg.d_model)
                    hook_fn(fake_act, None)
                elif 'hook_z' in hook_name:
                    fake_z = torch.randn(batch_size, seq_len, self.cfg.n_heads, self.cfg.d_head)
                    result = hook_fn(fake_z, None)
                    if result is not None:
                        fake_z = result
                elif 'hook_mlp_out' in hook_name:
                    fake_mlp = torch.randn(batch_size, seq_len, self.cfg.d_model)
                    hook_fn(fake_mlp, None)

        return torch.randn(batch_size, seq_len, 100)  # fake logits with vocab_size=100

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.run_with_hooks(tokens)

    def add_hook(self, name: str, fn: object) -> None:
        pass

    def reset_hooks(self) -> None:
        pass

    def generate(self, tokens: torch.Tensor, max_new_tokens: int = 30, temperature: float = 0) -> torch.Tensor:
        return torch.cat([tokens, torch.randint(1, 100, (1, max_new_tokens))], dim=1)


@pytest.fixture
def mock_model() -> MockHookedTransformer:
    return MockHookedTransformer()


@pytest.fixture
def mock_model_config() -> MockModelConfig:
    return MockModelConfig()


@pytest.fixture
def sample_pairs() -> list[tuple[str, str, str]]:
    return [
        ('What is the capital of France', 'Berlin', 'Paris'),
        ('Who wrote Romeo and Juliet', 'Dickens', 'Shakespeare'),
        ('What planet is closest to the sun', 'Venus', 'Mercury'),
        ('What is the largest ocean', 'Atlantic', 'Pacific'),
        ('Who painted the Mona Lisa', 'Picasso', 'da Vinci'),
    ]


@pytest.fixture
def sample_opinion_pairs() -> list[tuple[str, str, str]]:
    return [
        (
            'I think Python is the best programming language',
            'I think Java is the best programming language',
            'programming language',
        ),
        ('I think comedy is the best movie genre', 'I think drama is the best movie genre', 'movie genre'),
        ('I think spring is the best season', 'I think winter is the best season', 'season'),
    ]


@pytest.fixture
def random_activations() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    pos = rng.randn(50, 32)
    neg = rng.randn(50, 32)
    # make pos slightly different to get non-trivial probe results
    pos += 0.5
    return pos, neg
