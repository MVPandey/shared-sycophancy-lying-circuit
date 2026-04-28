"""Data loading utilities for TriviaQA factual pairs and opinion pairs."""

from shared_circuits.data.opinions import generate_opinion_pairs
from shared_circuits.data.triviaqa import load_triviaqa_pairs

__all__ = [
    'generate_opinion_pairs',
    'load_triviaqa_pairs',
]
