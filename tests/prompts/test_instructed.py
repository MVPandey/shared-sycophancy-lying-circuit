from unittest.mock import MagicMock

from shared_circuits.prompts import (
    build_instructed_lying_prompts,
    build_repe_lying_prompts,
    build_scaffolded_lying_prompts,
)


class TestBuildInstructedLyingPrompts:
    def test_correct_count(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        lying, honest = build_instructed_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        assert len(lying) == len(sample_pairs)
        assert len(honest) == len(sample_pairs)

    def test_lying_has_deception_preamble(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        lying, _ = build_instructed_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        assert 'spy' in lying[0].lower()
        assert 'mislead' in lying[0].lower()

    def test_honest_uses_ally_preamble(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        _, honest = build_instructed_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        assert 'ally' in honest[0].lower()

    def test_both_use_correct_answer(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        lying, honest = build_instructed_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        _, _w, c = sample_pairs[0]
        assert c in lying[0]
        assert c in honest[0]


class TestBuildScaffoldedLyingPrompts:
    def test_correct_count(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        lying, honest = build_scaffolded_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        assert len(lying) == len(sample_pairs)
        assert len(honest) == len(sample_pairs)

    def test_lying_mentions_lying(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        lying, _ = build_scaffolded_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        assert 'lie' in lying[0].lower()

    def test_honest_mentions_honestly(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        _, honest = build_scaffolded_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        assert 'honestly' in honest[0].lower()

    def test_includes_fewshot_demos(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        lying, honest = build_scaffolded_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        # both demo statements should appear in the rendered conversation
        assert 'Water boils' in lying[0]
        assert 'Paris is the capital' in lying[0]
        assert 'Water boils' in honest[0]


class TestBuildRepeLyingPrompts:
    def test_correct_count(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        dishonest, honest = build_repe_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        assert len(dishonest) == len(sample_pairs)
        assert len(honest) == len(sample_pairs)

    def test_personas_differ(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        dishonest, honest = build_repe_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        assert 'honest' in honest[0].lower()
        assert 'dishonest' in dishonest[0].lower()

    def test_both_contain_correct_answer(self, sample_pairs, stub_tokenizer: MagicMock) -> None:
        dishonest, honest = build_repe_lying_prompts(sample_pairs, 'gemma-2-2b-it')
        _, _w, c = sample_pairs[0]
        assert c in dishonest[0]
        assert c in honest[0]
