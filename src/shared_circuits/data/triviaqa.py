"""TriviaQA factual question loader with mismatched-answer pair construction."""

from typing import Final

MAX_ANSWER_LENGTH: Final = 30
MAX_ANSWER_WORDS: Final = 3


def load_triviaqa_pairs(n: int = 400) -> list[tuple[str, str, str]]:
    """
    Load TriviaQA question-answer pairs with mismatched wrong answers.

    Streams the TriviaQA validation set, filters for short single-word answers,
    then creates (question, wrong_answer, correct_answer) triples by offsetting
    the answer list.

    Args:
        n: Number of pairs to generate.

    Returns:
        List of (question, wrong_answer, correct_answer) tuples.

    """
    # deferred: datasets pulls pyarrow + HF hub state at module load
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset('trivia_qa', 'unfiltered.nocontext', split='validation', streaming=True)
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for ex in ds:
        q = ex['question'].strip()
        if q in seen:
            continue
        aliases = ex.get('answer', {}).get('aliases', [])
        if not aliases or len(aliases[0]) > MAX_ANSWER_LENGTH or len(aliases[0].split()) > MAX_ANSWER_WORDS:
            continue
        pairs.append((q, aliases[0]))
        seen.add(q)
        if len(pairs) >= n * 2:
            break

    result: list[tuple[str, str, str]] = []
    for i in range(min(n, len(pairs) - 1)):
        q, c = pairs[i]
        # offset by 37 to pair each question with an unrelated wrong answer
        _, w = pairs[(i + 37) % len(pairs)]
        if w.lower() != c.lower():
            result.append((q, w, c))
        if len(result) >= n:
            break
    return result
