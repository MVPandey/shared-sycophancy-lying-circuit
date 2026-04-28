import numpy as np

from shared_circuits.extraction import extract_residual_with_ablation, extract_with_head_ablation


class TestExtractWithHeadAblation:
    def test_output_shape(self, mock_model):
        prompts = ['hello', 'world']
        result = extract_with_head_ablation(mock_model, prompts, ablate_heads=[(0, 1), (1, 0)])
        assert result.shape == (2, 100)

    def test_returns_numpy(self, mock_model):
        result = extract_with_head_ablation(mock_model, ['test'], ablate_heads=[(0, 0)])
        assert isinstance(result, np.ndarray)

    def test_empty_ablation_list(self, mock_model):
        result = extract_with_head_ablation(mock_model, ['hello'], ablate_heads=[])
        assert result.shape == (1, 100)


class TestExtractResidualWithAblation:
    def test_no_ablation(self, mock_model):
        result = extract_residual_with_ablation(mock_model, ['hello'], layer=1, ablate_heads=None)
        assert result.shape == (1, mock_model.cfg.d_model)

    def test_with_ablation(self, mock_model):
        result = extract_residual_with_ablation(mock_model, ['hello', 'world'], layer=1, ablate_heads=[(0, 1)])
        assert result.shape == (2, mock_model.cfg.d_model)

    def test_returns_numpy(self, mock_model):
        result = extract_residual_with_ablation(mock_model, ['x'], layer=0)
        assert isinstance(result, np.ndarray)
