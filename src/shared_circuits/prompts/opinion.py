"""Opinion sycophancy prompt pairs (agree with opinion A vs B)."""

from shared_circuits.prompts.chat import render_chat


def build_opinion_prompts(
    opinion_pairs: list[tuple[str, str, str]],
    model_name: str,
) -> tuple[list[str], list[str]]:
    """
    Build opinion sycophancy prompts (agree with opinion A vs B).

    Args:
        opinion_pairs: List of (opinion_a, opinion_b, category) tuples.
        model_name: HuggingFace model name (used for the chat template).

    Returns:
        Tuple of (opinion_a_prompts, opinion_b_prompts).

    """
    prompts_a = [
        render_chat([{'role': 'user', 'content': f'{oa}. Do you agree?'}], model_name)
        for oa, _ob, _cat in opinion_pairs
    ]
    prompts_b = [
        render_chat([{'role': 'user', 'content': f'{ob}. Do you agree?'}], model_name)
        for _oa, ob, _cat in opinion_pairs
    ]
    return prompts_a, prompts_b
