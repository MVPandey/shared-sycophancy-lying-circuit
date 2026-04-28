import dataclasses

import pytest

from shared_circuits.models import ModelInfo, get_model_info
from tests.conftest import MockHookedTransformer, MockModelConfig


class TestModelInfo:
    def test_is_frozen(self):
        info = ModelInfo(name='m', n_layers=1, n_heads=1, d_model=1, d_head=1, total_heads=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.name = 'other'  # ty: ignore[invalid-assignment]

    def test_has_slots(self):
        assert ModelInfo.__slots__ == ('name', 'n_layers', 'n_heads', 'd_model', 'd_head', 'total_heads')


class TestGetModelInfo:
    def test_extracts_dimensions(self):
        model = MockHookedTransformer(MockModelConfig(n_layers=8, n_heads=4, d_model=64, d_head=16))
        info = get_model_info(model)
        assert info.n_layers == 8
        assert info.n_heads == 4
        assert info.d_model == 64
        assert info.d_head == 16
        assert info.total_heads == 32

    def test_model_name(self):
        model = MockHookedTransformer(MockModelConfig(model_name='test-model'))
        info = get_model_info(model)
        assert info.name == 'test-model'
