"""Tests for the DPO preference-dataset builders."""

from typing import Final

import pytest
from pytest_mock import MockerFixture

from shared_circuits.data import dpo_preferences
from shared_circuits.data.dpo_preferences import (
    DEFAULT_TRAIN_OFFSET,
    build_antisyc_preferences,
    build_sham_preferences,
)

_REQUIRED_KEYS: Final[set[str]] = {'prompt', 'chosen', 'rejected'}


def _fake_triples(n: int) -> list[tuple[str, str, str]]:
    return [(f'q{i}?', f'wrong{i}', f'right{i}') for i in range(n)]


@pytest.fixture
def patched_loader(mocker: MockerFixture):
    """Patch the triviaqa loader to return deterministic synthetic triples."""

    def fake_load(n: int) -> list[tuple[str, str, str]]:
        return _fake_triples(n)

    mocker.patch.object(dpo_preferences, 'load_triviaqa_pairs', side_effect=fake_load)


class TestBuildAntisycPreferences:
    def test_returns_two_rows_per_triple(self, patched_loader):
        train, eval_rows = build_antisyc_preferences(n_train=5, n_eval=2)
        # each triple yields 1 wrong-opinion pair + 1 right-opinion pair
        assert len(train) == 10
        assert len(eval_rows) == 4

    def test_row_shape(self, patched_loader):
        train, _ = build_antisyc_preferences(n_train=3, n_eval=1)
        for row in train:
            assert _REQUIRED_KEYS == set(row.keys())
            assert isinstance(row['prompt'], list)
            assert row['prompt'][0]['role'] == 'user'
            assert row['chosen'][0]['role'] == 'assistant'
            assert row['rejected'][0]['role'] == 'assistant'

    def test_wrong_opinion_chosen_disagrees(self, patched_loader):
        train, _ = build_antisyc_preferences(n_train=1, n_eval=1)
        wrong_row = train[0]
        assert 'I believe' in wrong_row['prompt'][0]['content']
        assert 'wrong' in wrong_row['prompt'][0]['content']
        assert wrong_row['chosen'][0]['content'].startswith('No')
        assert wrong_row['rejected'][0]['content'].startswith('Yes')

    def test_right_opinion_chosen_agrees(self, patched_loader):
        train, _ = build_antisyc_preferences(n_train=1, n_eval=1)
        right_row = train[1]
        assert 'right' in right_row['prompt'][0]['content']
        assert right_row['chosen'][0]['content'].startswith('Yes')
        assert right_row['rejected'][0]['content'].startswith('No')

    def test_train_eval_disjoint(self, patched_loader):
        train, eval_rows = build_antisyc_preferences(n_train=3, n_eval=2)
        train_prompts = {r['prompt'][0]['content'] for r in train}
        eval_prompts = {r['prompt'][0]['content'] for r in eval_rows}
        assert train_prompts.isdisjoint(eval_prompts)

    def test_offset_skips_probe_slice(self, mocker: MockerFixture):
        captured: dict[str, int] = {}

        def fake_load(n: int) -> list[tuple[str, str, str]]:
            captured['n_requested'] = n
            return _fake_triples(n)

        mocker.patch.object(dpo_preferences, 'load_triviaqa_pairs', side_effect=fake_load)
        build_antisyc_preferences(n_train=10, n_eval=5)
        # n_requested should account for offset + train + eval
        assert captured['n_requested'] == DEFAULT_TRAIN_OFFSET + 10 + 5

    def test_offset_makes_index_zero_disjoint(self, patched_loader):
        train, _ = build_antisyc_preferences(n_train=2, n_eval=1)
        # the very first probe-eval question would be 'q0?' — DPO data must not include it
        assert all('q0?' not in r['prompt'][0]['content'] for r in train)


class TestBuildShamPreferences:
    def test_same_count_as_anti(self, patched_loader):
        anti_train, anti_eval = build_antisyc_preferences(n_train=4, n_eval=2)
        sham_train, sham_eval = build_sham_preferences(n_train=4, n_eval=2)
        assert len(sham_train) == len(anti_train)
        assert len(sham_eval) == len(anti_eval)

    def test_same_prompts_as_anti(self, patched_loader):
        anti_train, _ = build_antisyc_preferences(n_train=4, n_eval=1)
        sham_train, _ = build_sham_preferences(n_train=4, n_eval=1)
        anti_prompts = [r['prompt'][0]['content'] for r in anti_train]
        sham_prompts = [r['prompt'][0]['content'] for r in sham_train]
        assert anti_prompts == sham_prompts

    def test_chosen_rejected_drawn_from_anti_choices(self, patched_loader):
        anti_train, _ = build_antisyc_preferences(n_train=4, n_eval=1)
        sham_train, _ = build_sham_preferences(n_train=4, n_eval=1)
        # for each row, the {chosen, rejected} content set must equal the anti version's set
        for anti_row, sham_row in zip(anti_train, sham_train, strict=True):
            anti_pair = {anti_row['chosen'][0]['content'], anti_row['rejected'][0]['content']}
            sham_pair = {sham_row['chosen'][0]['content'], sham_row['rejected'][0]['content']}
            assert anti_pair == sham_pair

    def test_some_pairs_are_swapped(self, patched_loader):
        # with n_train=20 and seed=42, the chance that no pair gets flipped is ~1/2^20
        anti_train, _ = build_antisyc_preferences(n_train=20, n_eval=1)
        sham_train, _ = build_sham_preferences(n_train=20, n_eval=1, seed=42)
        flips = sum(
            1
            for a, s in zip(anti_train, sham_train, strict=True)
            if a['chosen'][0]['content'] != s['chosen'][0]['content']
        )
        assert flips > 0

    def test_seed_is_deterministic(self, patched_loader):
        a, _ = build_sham_preferences(n_train=8, n_eval=1, seed=7)
        b, _ = build_sham_preferences(n_train=8, n_eval=1, seed=7)
        # same seed => identical chosen contents per row
        for ra, rb in zip(a, b, strict=True):
            assert ra['chosen'][0]['content'] == rb['chosen'][0]['content']

    def test_row_shape(self, patched_loader):
        train, _ = build_sham_preferences(n_train=2, n_eval=1)
        for row in train:
            assert _REQUIRED_KEYS == set(row.keys())
