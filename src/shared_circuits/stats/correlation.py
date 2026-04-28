"""Rank and linear correlation utilities."""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def rank_correlation(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> dict[str, float]:
    """
    Compute Spearman and Pearson correlations between two arrays.

    Args:
        values_a: First array of values.
        values_b: Second array of values.

    Returns:
        Dict with spearman_rho, spearman_p, pearson_r, pearson_p.

    """
    rho, p_spearman = spearmanr(values_a, values_b)
    r, p_pearson = pearsonr(values_a, values_b)
    return {
        'spearman_rho': float(rho),
        'spearman_p': float(p_spearman),
        'pearson_r': float(r),
        'pearson_p': float(p_pearson),
    }
