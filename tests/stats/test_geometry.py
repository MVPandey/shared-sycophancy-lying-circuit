import numpy as np
import pytest

from shared_circuits.stats import cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        v = np.array([1.0, 0.0])
        w = np.array([-1.0, 0.0])
        assert cosine_similarity(v, w) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        v = np.array([1.0, 0.0])
        w = np.array([0.0, 1.0])
        assert cosine_similarity(v, w) == pytest.approx(0.0, abs=1e-10)

    def test_zero_vector_returns_zero(self):
        v = np.array([1.0, 2.0])
        z = np.array([0.0, 0.0])
        assert cosine_similarity(v, z) == 0.0
        assert cosine_similarity(z, v) == 0.0

    def test_near_zero_vector(self):
        v = np.array([1.0, 2.0])
        z = np.array([1e-12, 1e-12])
        assert cosine_similarity(v, z) == 0.0
