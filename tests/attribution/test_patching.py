import numpy as np
import pytest

from shared_circuits.attribution import compute_attribution_patching


class TestComputeAttributionPatching:
    def test_output_shape(self, mock_model):
        corrupted = ['wrong answer prompt']
        clean = ['correct answer prompt']
        agree = [1, 2, 3]
        disagree = [4, 5, 6]
        result = compute_attribution_patching(mock_model, corrupted, clean, agree, disagree, n_pairs=1)
        assert result.shape == (mock_model.cfg.n_layers, mock_model.cfg.n_heads)

    def test_returns_numpy(self, mock_model):
        result = compute_attribution_patching(mock_model, ['a'], ['b'], [1], [2], n_pairs=1)
        assert isinstance(result, np.ndarray)

    def test_handles_fewer_pairs(self, mock_model):
        # n_pairs > len(prompts) should clamp to len(prompts)
        result = compute_attribution_patching(mock_model, ['a'], ['b'], [1], [2], n_pairs=10)
        assert result.shape == (mock_model.cfg.n_layers, mock_model.cfg.n_heads)

    def test_mismatched_lengths_raises(self, mock_model):
        with pytest.raises(ValueError, match='equal length'):
            compute_attribution_patching(mock_model, ['a', 'b'], ['c'], [1], [2])

    def test_empty_prompts_raises(self, mock_model):
        with pytest.raises(ValueError, match='at least 1'):
            compute_attribution_patching(mock_model, [], [], [1], [2])
