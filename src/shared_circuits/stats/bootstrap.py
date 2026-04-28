"""Bootstrap confidence-interval estimation."""

from collections.abc import Callable

import numpy as np

from shared_circuits.config import BOOTSTRAP_ITERATIONS, CI_QUANTILES, RANDOM_SEED


def bootstrap_confidence_interval(
    values: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = BOOTSTRAP_ITERATIONS,
    quantiles: tuple[float, float] = CI_QUANTILES,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """
    Compute a bootstrap confidence interval for a statistic.

    Args:
        values: Input data array.
        statistic_fn: Callable that takes resampled values and returns a scalar.
        n_bootstrap: Number of bootstrap iterations.
        quantiles: Lower and upper percentiles for CI.
        seed: Random seed.

    Returns:
        Tuple of (lower_bound, upper_bound).

    """
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_stats = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        boot_stats.append(statistic_fn(values[idx]))
    ci = np.percentile(boot_stats, list(quantiles))
    return float(ci[0]), float(ci[1])
