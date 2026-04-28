from unittest.mock import MagicMock

from shared_circuits.prompts import build_lying_prompts


class TestBuildLyingPrompts:
    def test_correct_count(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        false_s, true_s = build_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        assert len(false_s) == len(sample_pairs)
        assert len(true_s) == len(sample_pairs)

    def test_contains_true_or_false_framing(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        false_s, _ = build_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        assert 'true or false' in false_s[0]

    def test_false_contains_wrong_answer(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        false_s, _ = build_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        _, w, _c = sample_pairs[0]
        assert w in false_s[0]

    def test_true_contains_correct_answer(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        _, true_s = build_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        _, _w, c = sample_pairs[0]
        assert c in true_s[0]

    def test_strips_trailing_question_mark(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        false_s, _ = build_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        # source pairs end without '?', but lying.py rstrips '?' anyway: assert no '?' before " is "
        for prompt in false_s:
            assert '? is ' not in prompt
