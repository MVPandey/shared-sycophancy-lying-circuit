import numpy as np
import pytest

from shared_circuits.stats import rank_correlation


class TestRankCorrelation:
    def test_perfect_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rank_correlation(a, a)
        assert result['spearman_rho'] == pytest.approx(1.0)
        assert result['pearson_r'] == pytest.approx(1.0)

    def test_returns_all_keys(self):
        a = np.random.randn(20)
        b = np.random.randn(20)
        result = rank_correlation(a, b)
        assert 'spearman_rho' in result
        assert 'spearman_p' in result
        assert 'pearson_r' in result
        assert 'pearson_p' in result
