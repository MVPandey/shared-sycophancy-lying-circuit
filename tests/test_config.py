from shared_circuits.config import (
    ALL_MODELS,
    BOOTSTRAP_ITERATIONS,
    CI_QUANTILES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_PROMPTS,
    DEFAULT_TOP_K,
    PERMUTATION_ITERATIONS,
    RANDOM_SEED,
    TL_ARCH_OVERRIDES,
)


class TestNumericConstants:
    def test_batch_size_is_positive(self) -> None:
        assert DEFAULT_BATCH_SIZE > 0

    def test_random_seed_is_int(self) -> None:
        assert isinstance(RANDOM_SEED, int)

    def test_iteration_counts_are_positive(self) -> None:
        assert BOOTSTRAP_ITERATIONS > 0
        assert PERMUTATION_ITERATIONS > 0

    def test_default_n_prompts_and_top_k(self) -> None:
        assert DEFAULT_N_PROMPTS > 0
        assert DEFAULT_TOP_K > 0


class TestCIQuantiles:
    def test_two_quantiles(self) -> None:
        assert len(CI_QUANTILES) == 2

    def test_quantiles_in_range_and_ordered(self) -> None:
        lo, hi = CI_QUANTILES
        assert 0 <= lo < hi <= 100


class TestAllModels:
    def test_all_entries_are_non_empty_strings(self) -> None:
        assert len(ALL_MODELS) > 0
        for m in ALL_MODELS:
            assert isinstance(m, str)
            assert m

    def test_no_duplicates(self) -> None:
        assert len(set(ALL_MODELS)) == len(ALL_MODELS)


class TestTLArchOverrides:
    def test_keys_and_values_are_strings(self) -> None:
        for k, v in TL_ARCH_OVERRIDES.items():
            assert isinstance(k, str)
            assert k
            assert isinstance(v, str)
            assert v
