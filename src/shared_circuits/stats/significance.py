"""Hypothesis tests for attention-head overlap significance."""

import numpy as np
from scipy.stats import hypergeom

from shared_circuits.config import PERMUTATION_ITERATIONS, RANDOM_SEED


def head_overlap_hypergeometric(
    overlap: int,
    k: int,
    total_heads: int,
) -> float:
    """
    Compute p-value for head overlap using a hypergeometric test.

    Tests whether the observed overlap between two top-K head sets is
    significantly greater than expected by chance.

    Args:
        overlap: Number of heads in both top-K lists.
        k: Size of each top-K list.
        total_heads: Total number of heads in the model.

    Returns:
        One-sided p-value (probability of observing >= overlap by chance).

    """
    return float(hypergeom.sf(overlap - 1, total_heads, k, k))


def permutation_test_overlap(
    set_a: set[tuple[int, int]],
    set_b: set[tuple[int, int]],
    all_heads: list[tuple[int, int]],
    k: int,
    n_perms: int = PERMUTATION_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> float:
    """
    Permutation test for head overlap significance.

    Shuffles head labels and measures how often a random top-K set
    overlaps with set_a as much as set_b does.

    Args:
        set_a: Top-K heads from the first task.
        set_b: Top-K heads from the second task.
        all_heads: All heads in the model.
        k: Size of each top-K set.
        n_perms: Number of permutations.
        seed: Random seed.

    Returns:
        Empirical p-value.

    """
    rng = np.random.RandomState(seed)
    observed = len(set_a & set_b)
    null_overlaps = []
    for _ in range(n_perms):
        perm = rng.permutation(len(all_heads))
        fake_top = {all_heads[i] for i in perm[:k]}
        null_overlaps.append(len(set_a & fake_top))
    return float(np.mean([n >= observed for n in null_overlaps]))
