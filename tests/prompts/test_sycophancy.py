from unittest.mock import MagicMock

from shared_circuits.prompts import build_sycophancy_prompts


class TestBuildSycophancyPrompts:
    def test_correct_count(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        wrong, correct = build_sycophancy_prompts(sample_pairs, 'gemma-2-2b-it')
        assert len(wrong) == len(sample_pairs)
        assert len(correct) == len(sample_pairs)

    def test_wrong_contains_question_and_wrong_answer(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        wrong, _ = build_sycophancy_prompts(sample_pairs, 'gemma-2-2b-it')
        q, w, _c = sample_pairs[0]
        assert q in wrong[0]
        assert w in wrong[0]

    def test_correct_contains_question_and_correct_answer(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        _, correct = build_sycophancy_prompts(sample_pairs, 'gemma-2-2b-it')
        q, _w, c = sample_pairs[0]
        assert q in correct[0]
        assert c in correct[0]

    def test_uses_am_i_correct_phrasing(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        wrong, correct = build_sycophancy_prompts(sample_pairs, 'gemma-2-2b-it')
        assert 'Am I correct?' in wrong[0]
        assert 'Am I correct?' in correct[0]

    def test_wrong_and_correct_differ(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        wrong, correct = build_sycophancy_prompts(sample_pairs, 'gemma-2-2b-it')
        for w_p, c_p in zip(wrong, correct, strict=True):
            assert w_p != c_p
