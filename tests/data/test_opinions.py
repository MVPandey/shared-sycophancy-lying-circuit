from shared_circuits.data import generate_opinion_pairs
from shared_circuits.data.opinions import OPINION_CATEGORIES, OPINION_TEMPLATES


class TestOpinionCategories:
    def test_categories_have_options(self) -> None:
        for cat in OPINION_CATEGORIES:
            assert 'cat' in cat
            assert 'options' in cat
            assert len(cat['options']) >= 4

    def test_templates_have_placeholders(self) -> None:
        for t1, _t2 in OPINION_TEMPLATES:
            assert '{a}' in t1
            assert '{cat}' in t1


class TestGenerateOpinionPairs:
    def test_correct_count(self) -> None:
        pairs = generate_opinion_pairs(50)
        assert len(pairs) == 50

    def test_tuple_structure(self) -> None:
        pairs = generate_opinion_pairs(10)
        for oa, ob, cat in pairs:
            assert isinstance(oa, str)
            assert isinstance(ob, str)
            assert isinstance(cat, str)
            assert len(oa) > 0
            assert len(ob) > 0

    def test_deterministic_with_seed(self) -> None:
        p1 = generate_opinion_pairs(20, seed=42)
        p2 = generate_opinion_pairs(20, seed=42)
        assert p1 == p2

    def test_different_seeds_different_results(self) -> None:
        p1 = generate_opinion_pairs(20, seed=42)
        p2 = generate_opinion_pairs(20, seed=99)
        assert p1 != p2

    def test_opinions_differ(self) -> None:
        pairs = generate_opinion_pairs(50)
        for oa, ob, _ in pairs:
            assert oa != ob

    def test_large_generation(self) -> None:
        pairs = generate_opinion_pairs(300)
        assert len(pairs) == 300

    def test_categories_are_valid(self) -> None:
        pairs = generate_opinion_pairs(100)
        valid_cats = {str(c['cat']) for c in OPINION_CATEGORIES}
        for _, _, cat in pairs:
            assert cat in valid_cats
