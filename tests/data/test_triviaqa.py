from collections.abc import Iterator
from typing import Any

from pytest_mock import MockerFixture

from shared_circuits.data import load_triviaqa_pairs


def _fake_dataset(n: int = 20) -> Iterator[dict[str, Any]]:
    for i in range(n):
        yield {
            'question': f'Question {i}?',
            'answer': {'aliases': [f'Answer{i}']},
        }


class TestLoadTriviaqaPairs:
    def test_returns_correct_count(self, mocker: MockerFixture) -> None:
        mocker.patch('datasets.load_dataset', return_value=_fake_dataset(30))
        pairs = load_triviaqa_pairs(5)
        assert len(pairs) == 5

    def test_tuple_structure(self, mocker: MockerFixture) -> None:
        mocker.patch('datasets.load_dataset', return_value=_fake_dataset(30))
        pairs = load_triviaqa_pairs(5)
        for q, w, c in pairs:
            assert isinstance(q, str)
            assert isinstance(w, str)
            assert isinstance(c, str)

    def test_wrong_differs_from_correct(self, mocker: MockerFixture) -> None:
        mocker.patch('datasets.load_dataset', return_value=_fake_dataset(30))
        pairs = load_triviaqa_pairs(5)
        for _, w, c in pairs:
            assert w.lower() != c.lower()

    def test_filters_long_answers(self, mocker: MockerFixture) -> None:
        entries: list[dict[str, Any]] = [
            {'question': 'Q1?', 'answer': {'aliases': ['short']}},
            {'question': 'Q2?', 'answer': {'aliases': ['this is a very long answer that exceeds the limit']}},
            {'question': 'Q3?', 'answer': {'aliases': ['ok']}},
            {'question': 'Q4?', 'answer': {'aliases': ['fine']}},
            {'question': 'Q5?', 'answer': {'aliases': ['good']}},
            {'question': 'Q6?', 'answer': {'aliases': ['nice']}},
        ]
        mocker.patch('datasets.load_dataset', return_value=iter(entries))
        pairs = load_triviaqa_pairs(2)
        all_answers = [c for _, _, c in pairs] + [w for _, w, _ in pairs]
        assert 'this is a very long answer that exceeds the limit' not in all_answers

    def test_filters_too_many_words(self, mocker: MockerFixture) -> None:
        entries: list[dict[str, Any]] = [
            {'question': 'Q1?', 'answer': {'aliases': ['ok']}},
            {'question': 'Q2?', 'answer': {'aliases': ['way too many words here yes']}},
            {'question': 'Q3?', 'answer': {'aliases': ['fine']}},
            {'question': 'Q4?', 'answer': {'aliases': ['good']}},
        ]
        mocker.patch('datasets.load_dataset', return_value=iter(entries))
        pairs = load_triviaqa_pairs(2)
        all_answers = [c for _, _, c in pairs] + [w for _, w, _ in pairs]
        assert 'way too many words here yes' not in all_answers

    def test_skips_duplicates(self, mocker: MockerFixture) -> None:
        entries: list[dict[str, Any]] = [
            {'question': 'Same?', 'answer': {'aliases': ['A']}},
            {'question': 'Same?', 'answer': {'aliases': ['B']}},
            {'question': 'Different?', 'answer': {'aliases': ['C']}},
            {'question': 'Another?', 'answer': {'aliases': ['D']}},
            {'question': 'More?', 'answer': {'aliases': ['E']}},
        ]
        mocker.patch('datasets.load_dataset', return_value=iter(entries))
        pairs = load_triviaqa_pairs(2)
        questions = [q for q, _, _ in pairs]
        assert questions.count('Same?') <= 1

    def test_skips_missing_aliases(self, mocker: MockerFixture) -> None:
        entries: list[dict[str, Any]] = [
            {'question': 'Q1?', 'answer': {'aliases': []}},
            {'question': 'Q2?', 'answer': {'aliases': ['ok']}},
            {'question': 'Q3?', 'answer': {'aliases': ['fine']}},
            {'question': 'Q4?', 'answer': {'aliases': ['good']}},
        ]
        mocker.patch('datasets.load_dataset', return_value=iter(entries))
        pairs = load_triviaqa_pairs(2)
        questions = [q for q, _, _ in pairs]
        assert 'Q1?' not in questions
