"""Lying prompt pairs (false-statement vs true-statement)."""

from shared_circuits.prompts.chat import render_chat


def build_lying_prompts(
    pairs: list[tuple[str, str, str]],
    model_name: str,
) -> tuple[list[str], list[str]]:
    """
    Build lying prompt pairs (false-statement vs true-statement).

    Args:
        pairs: List of (question, wrong_answer, correct_answer) tuples.
        model_name: HuggingFace model name (used for the chat template).

    Returns:
        Tuple of (false_prompts, true_prompts) — same lengths as ``pairs``.

    """
    false_stmts = [
        render_chat(
            [
                {
                    'role': 'user',
                    'content': f'Is the following statement true or false: "{q.rstrip("?").rstrip()} is {w}"',
                },
            ],
            model_name,
        )
        for q, w, _c in pairs
    ]
    true_stmts = [
        render_chat(
            [
                {
                    'role': 'user',
                    'content': f'Is the following statement true or false: "{q.rstrip("?").rstrip()} is {c}"',
                },
            ],
            model_name,
        )
        for q, _w, c in pairs
    ]
    return false_stmts, true_stmts
