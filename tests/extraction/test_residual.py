import numpy as np

from shared_circuits.extraction import extract_residual_stream, extract_residual_stream_multi


class TestExtractResidualStream:
    def test_output_shape(self, mock_model):
        prompts = ['hello', 'world', 'test']
        result = extract_residual_stream(mock_model, prompts, layer=1)
        assert result.shape == (3, mock_model.cfg.d_model)

    def test_returns_numpy(self, mock_model):
        result = extract_residual_stream(mock_model, ['hello'], layer=0)
        assert isinstance(result, np.ndarray)

    def test_single_prompt(self, mock_model):
        result = extract_residual_stream(mock_model, ['one prompt'], layer=2)
        assert result.shape == (1, mock_model.cfg.d_model)

    def test_batching(self, mock_model):
        prompts = [f'prompt {i}' for i in range(20)]
        result = extract_residual_stream(mock_model, prompts, layer=1, batch_size=4)
        assert result.shape == (20, mock_model.cfg.d_model)


class TestExtractResidualStreamMulti:
    def test_returns_all_layers(self, mock_model):
        layers = [0, 2]
        result = extract_residual_stream_multi(mock_model, ['hello', 'world'], layers)
        assert set(result.keys()) == {0, 2}

    def test_each_layer_correct_shape(self, mock_model):
        layers = [0, 1, 3]
        result = extract_residual_stream_multi(mock_model, ['a', 'b', 'c'], layers)
        for layer in layers:
            assert result[layer].shape == (3, mock_model.cfg.d_model)

    def test_returns_numpy(self, mock_model):
        result = extract_residual_stream_multi(mock_model, ['x'], [0])
        assert isinstance(result[0], np.ndarray)
