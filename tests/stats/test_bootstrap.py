import numpy as np

from shared_circuits.stats import bootstrap_confidence_interval


class TestBootstrapConfidenceInterval:
    def test_returns_tuple(self):
        values = np.random.randn(100)
        lo, hi = bootstrap_confidence_interval(values, np.mean)
        assert lo < hi

    def test_mean_within_ci(self):
        values = np.random.RandomState(42).randn(100)
        lo, hi = bootstrap_confidence_interval(values, np.mean)
        assert lo <= np.mean(values) <= hi

    def test_deterministic(self):
        values = np.arange(50, dtype=float)
        ci1 = bootstrap_confidence_interval(values, np.mean, seed=42)
        ci2 = bootstrap_confidence_interval(values, np.mean, seed=42)
        assert ci1 == ci2
