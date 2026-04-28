"""Statistical analysis utilities for circuit overlap experiments."""

from shared_circuits.stats.bootstrap import bootstrap_confidence_interval
from shared_circuits.stats.correlation import rank_correlation
from shared_circuits.stats.geometry import cosine_similarity
from shared_circuits.stats.probes import evaluate_probe_transfer, train_probe
from shared_circuits.stats.significance import head_overlap_hypergeometric, permutation_test_overlap

__all__ = [
    'bootstrap_confidence_interval',
    'cosine_similarity',
    'evaluate_probe_transfer',
    'head_overlap_hypergeometric',
    'permutation_test_overlap',
    'rank_correlation',
    'train_probe',
]
