"""Per-head importance attribution: DLA mean-difference and attribution patching."""

from shared_circuits.attribution.dla import (
    compute_head_importance_grid,
    compute_head_importances,
    rank_heads,
)
from shared_circuits.attribution.patching import compute_attribution_patching

__all__ = [
    'compute_attribution_patching',
    'compute_head_importance_grid',
    'compute_head_importances',
    'rank_heads',
]
