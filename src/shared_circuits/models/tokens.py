"""Agreement / disagreement vocabulary token-id resolution."""

from typing import Final

AGREE_WORDS: Final[tuple[str, ...]] = ('Yes', 'yes', 'Yeah', 'yeah', 'correct', 'Correct', 'right', 'Right')
DISAGREE_WORDS: Final[tuple[str, ...]] = ('No', 'no', 'not', 'Not', 'incorrect', 'Incorrect', 'wrong', 'Wrong')


def get_agree_disagree_tokens(model: 'HookedTransformer') -> tuple[list[int], list[int]]:
    """
    Get token IDs for agreement and disagreement words.

    Args:
        model: Loaded TransformerLens model with tokenizer.

    Returns:
        Tuple of (agree_token_ids, disagree_token_ids).

    """
    agree_tokens: list[int] = []
    for word in AGREE_WORDS:
        toks = model.to_tokens(word, prepend_bos=False)[0]
        agree_tokens.extend(toks.tolist())

    disagree_tokens: list[int] = []
    for word in DISAGREE_WORDS:
        toks = model.to_tokens(word, prepend_bos=False)[0]
        disagree_tokens.extend(toks.tolist())

    return list(set(agree_tokens)), list(set(disagree_tokens))
