from unittest.mock import MagicMock

from shared_circuits.data.naturalquestions import (
    MAX_ANSWER_LENGTH,
    MAX_ANSWER_WORDS,
    load_naturalquestions_pairs,
)


def _fake_rows():
    return [
        {'question': 'who wrote hamlet', 'answer': ['Shakespeare']},
        {'question': 'capital of france', 'answer': ['Paris']},
        {'question': 'tallest mountain', 'answer': ['Everest']},
        {'question': 'longest river', 'answer': ['Nile']},
        {'question': 'no answer', 'answer': []},  # filtered
        {'question': 'too long', 'answer': ['a' * (MAX_ANSWER_LENGTH + 1)]},  # filtered
        {'question': 'too many words', 'answer': [' '.join(['x'] * (MAX_ANSWER_WORDS + 1))]},  # filtered
    ]


class TestLoadNaturalQuestionsPairs:
    def test_returns_triples(self, mocker):
        mocker.patch('datasets.load_dataset', return_value=_fake_rows())
        pairs = load_naturalquestions_pairs(n=2)
        assert all(isinstance(p, tuple) and len(p) == 3 for p in pairs)
        for q, w, c in pairs:
            assert isinstance(q, str)
            assert w != c  # mismatched

    def test_filters_long_answers(self, mocker):
        mocker.patch('datasets.load_dataset', return_value=_fake_rows())
        pairs = load_naturalquestions_pairs(n=10)
        for _q, w, c in pairs:
            assert len(w) <= MAX_ANSWER_LENGTH
            assert len(w.split()) <= MAX_ANSWER_WORDS
            assert len(c) <= MAX_ANSWER_LENGTH
            assert len(c.split()) <= MAX_ANSWER_WORDS

    def test_uses_streaming(self, mocker):
        load_dataset_mock = mocker.patch('datasets.load_dataset', return_value=_fake_rows())
        load_naturalquestions_pairs(n=1)
        _, kwargs = load_dataset_mock.call_args
        assert kwargs.get('streaming') is True

    def test_empty_dataset(self, mocker):
        empty = MagicMock()
        empty.__iter__ = lambda _self: iter([])
        mocker.patch('datasets.load_dataset', return_value=empty)
        assert load_naturalquestions_pairs(n=5) == []
