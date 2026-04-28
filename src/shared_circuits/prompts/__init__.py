"""Prompt builders for sycophancy, lying, and opinion paradigms."""

from shared_circuits.prompts.chat import render_chat
from shared_circuits.prompts.instructed import (
    build_instructed_lying_prompts,
    build_repe_lying_prompts,
    build_scaffolded_lying_prompts,
)
from shared_circuits.prompts.lying import build_lying_prompts
from shared_circuits.prompts.opinion import build_opinion_prompts
from shared_circuits.prompts.sycophancy import build_sycophancy_prompts

__all__ = [
    'build_instructed_lying_prompts',
    'build_lying_prompts',
    'build_opinion_prompts',
    'build_repe_lying_prompts',
    'build_scaffolded_lying_prompts',
    'build_sycophancy_prompts',
    'render_chat',
]
