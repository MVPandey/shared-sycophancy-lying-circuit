from shared_circuits.attribution import compute_head_importance_grid, compute_head_importances, rank_heads


class TestComputeHeadImportances:
    def test_returns_all_heads(self, mock_model):
        prompts = ['a', 'b', 'c']
        result = compute_head_importances(mock_model, prompts, prompts, n_prompts=3, batch_size=4)
        expected_count = mock_model.cfg.n_layers * mock_model.cfg.n_heads
        assert len(result) == expected_count

    def test_values_are_nonnegative(self, mock_model):
        prompts = ['test1', 'test2']
        result = compute_head_importances(mock_model, prompts, prompts, n_prompts=2)
        for v in result.values():
            assert v >= 0

    def test_keys_are_layer_head_tuples(self, mock_model):
        prompts = ['x']
        result = compute_head_importances(mock_model, prompts, prompts, n_prompts=1)
        for k in result:
            assert isinstance(k, tuple)
            assert len(k) == 2


class TestRankHeads:
    def test_returns_top_k(self):
        deltas = {(0, 0): 1.0, (0, 1): 3.0, (1, 0): 2.0, (1, 1): 0.5}
        ranked = rank_heads(deltas, top_k=2)
        assert len(ranked) == 2
        assert ranked[0][0] == (0, 1)  # highest first
        assert ranked[1][0] == (1, 0)

    def test_descending_order(self):
        deltas = {(i, 0): float(i) for i in range(10)}
        ranked = rank_heads(deltas, top_k=5)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_clamped(self):
        deltas = {(0, 0): 1.0, (0, 1): 2.0}
        ranked = rank_heads(deltas, top_k=10)
        assert len(ranked) == 2


class TestComputeHeadImportanceGrid:
    def test_correct_shape(self):
        deltas = {(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0, (1, 1): 4.0}
        grid = compute_head_importance_grid(deltas, n_layers=2, n_heads=2)
        assert grid.shape == (2, 2)
        assert grid[1, 1] == 4.0

    def test_zeros_for_missing(self):
        deltas = {(0, 0): 5.0}
        grid = compute_head_importance_grid(deltas, n_layers=2, n_heads=2)
        assert grid[0, 0] == 5.0
        assert grid[0, 1] == 0.0
        assert grid[1, 0] == 0.0
