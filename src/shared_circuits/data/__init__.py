"""Data loading utilities for TriviaQA / NaturalQuestions factual pairs and opinion pairs."""

from shared_circuits.data.dpo_preferences import build_antisyc_preferences, build_sham_preferences
from shared_circuits.data.naturalquestions import load_naturalquestions_pairs
from shared_circuits.data.opinions import generate_opinion_pairs
from shared_circuits.data.sae_features import SAE_REPOS, SaeWeights, load_sae, load_sae_for_model, repo_for_model
from shared_circuits.data.triviaqa import load_triviaqa_pairs

__all__ = [
    'SAE_REPOS',
    'SaeWeights',
    'build_antisyc_preferences',
    'build_sham_preferences',
    'generate_opinion_pairs',
    'load_naturalquestions_pairs',
    'load_sae',
    'load_sae_for_model',
    'load_triviaqa_pairs',
    'repo_for_model',
]
