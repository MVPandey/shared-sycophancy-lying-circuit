"""Hook-based activation extraction from TransformerLens models."""

from shared_circuits.extraction.ablated import (
    extract_residual_with_ablation,
    extract_with_head_ablation,
)
from shared_circuits.extraction.behavior import (
    measure_agreement_per_prompt,
    measure_agreement_rate,
)
from shared_circuits.extraction.extractor import BatchedExtractor, HookSpec
from shared_circuits.extraction.residual import (
    extract_residual_stream,
    extract_residual_stream_multi,
)

__all__ = [
    'BatchedExtractor',
    'HookSpec',
    'extract_residual_stream',
    'extract_residual_stream_multi',
    'extract_residual_with_ablation',
    'extract_with_head_ablation',
    'measure_agreement_per_prompt',
    'measure_agreement_rate',
]
