"""
NaturalQuestions-Open factual loader with mismatched-answer pair construction.

Mirrors the TriviaQA loader interface so analyses can swap datasets via a flag.
"""

from typing import Final

MAX_ANSWER_LENGTH: Final = 30
MAX_ANSWER_WORDS: Final = 3


def load_naturalquestions_pairs(n: int = 200) -> list[tuple[str, str, str]]:
    """
    Load NQ-Open question/answer pairs with mismatched wrong answers.

    Streams the NQ-Open validation set, filters for short single-word answers,
    then creates ``(question, wrong_answer, correct_answer)`` triples by
    pairing each question with the answer ``n`` slots ahead in the stream.

    Args:
        n: Number of pairs to generate.

    Returns:
        List of ``(question, wrong_answer, correct_answer)`` tuples.

    """
    # deferred: datasets pulls pyarrow + HF hub state at module load
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset('google-research-datasets/nq_open', split='validation', streaming=True)
    raw: list[tuple[str, str]] = []
    for row in ds:
        answers = row.get('answer', [])
        if not answers:
            continue
        ans = answers[0]
        if len(ans) > MAX_ANSWER_LENGTH or len(ans.split()) > MAX_ANSWER_WORDS:
            continue
        raw.append((row['question'], ans))
        if len(raw) >= n * 2:
            break

    pairs: list[tuple[str, str, str]] = []
    for i in range(min(n, len(raw))):
        q, correct = raw[i]
        # offset by n slots to pair each question with an unrelated wrong answer
        wrong = raw[(i + n) % len(raw)][1]
        if wrong.lower() != correct.lower():
            pairs.append((q, wrong, correct))
    return pairs
