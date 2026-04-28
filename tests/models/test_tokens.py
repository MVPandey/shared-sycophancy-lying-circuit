from shared_circuits.models import get_agree_disagree_tokens
from tests.conftest import MockHookedTransformer


class TestGetAgreeDisagreeTokens:
    def test_returns_nonempty_lists(self):
        model = MockHookedTransformer()
        agree, disagree = get_agree_disagree_tokens(model)
        assert len(agree) > 0
        assert len(disagree) > 0

    def test_returns_int_lists(self):
        model = MockHookedTransformer()
        agree, disagree = get_agree_disagree_tokens(model)
        # token ids are random in mock, so just check types
        assert all(isinstance(t, int) for t in agree)
        assert all(isinstance(t, int) for t in disagree)
