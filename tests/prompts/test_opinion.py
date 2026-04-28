from unittest.mock import MagicMock

from shared_circuits.prompts import build_opinion_prompts


class TestBuildOpinionPrompts:
    def test_correct_count(self, sample_opinion_pairs, stub_tokenizer: MagicMock) -> None:
        a, b = build_opinion_prompts(sample_opinion_pairs, 'gemma-2-2b-it')
        assert len(a) == len(sample_opinion_pairs)
        assert len(b) == len(sample_opinion_pairs)

    def test_asks_for_agreement(self, sample_opinion_pairs, stub_tokenizer: MagicMock) -> None:
        a, _ = build_opinion_prompts(sample_opinion_pairs, 'gemma-2-2b-it')
        assert 'Do you agree?' in a[0]

    def test_a_contains_first_opinion(self, sample_opinion_pairs, stub_tokenizer: MagicMock) -> None:
        a, _ = build_opinion_prompts(sample_opinion_pairs, 'gemma-2-2b-it')
        oa, _ob, _cat = sample_opinion_pairs[0]
        assert oa in a[0]

    def test_b_contains_second_opinion(self, sample_opinion_pairs, stub_tokenizer: MagicMock) -> None:
        _, b = build_opinion_prompts(sample_opinion_pairs, 'gemma-2-2b-it')
        _oa, ob, _cat = sample_opinion_pairs[0]
        assert ob in b[0]

    def test_opinions_differ(self, sample_opinion_pairs, stub_tokenizer: MagicMock) -> None:
        a, b = build_opinion_prompts(sample_opinion_pairs, 'gemma-2-2b-it')
        assert a[0] != b[0]
