import pytest

from shared_circuits.stats import head_overlap_hypergeometric, permutation_test_overlap


class TestHeadOverlapHypergeometric:
    def test_perfect_overlap(self):
        # 15 out of 15 overlap with 200 total heads - extremely significant
        p = head_overlap_hypergeometric(15, 15, 200)
        assert p < 1e-10

    def test_no_overlap(self):
        p = head_overlap_hypergeometric(0, 15, 200)
        assert p == pytest.approx(1.0)

    def test_expected_overlap(self):
        # at expected level, p should be moderate
        p = head_overlap_hypergeometric(1, 15, 200)
        assert 0.0 < p < 1.0


class TestPermutationTestOverlap:
    def test_large_overlap_significant(self):
        set_a = {(0, i) for i in range(15)}
        set_b = set_a  # perfect overlap
        all_heads = [(l, h) for l in range(10) for h in range(10)]
        p = permutation_test_overlap(set_a, set_b, all_heads, k=15, n_perms=100)
        assert p < 0.05

    def test_deterministic_with_seed(self):
        set_a = {(0, 0), (0, 1)}
        set_b = {(0, 0), (0, 2)}
        all_heads = [(l, h) for l in range(5) for h in range(5)]
        p1 = permutation_test_overlap(set_a, set_b, all_heads, k=5, seed=42)
        p2 = permutation_test_overlap(set_a, set_b, all_heads, k=5, seed=42)
        assert p1 == p2
