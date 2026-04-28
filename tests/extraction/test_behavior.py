import pytest

from shared_circuits.extraction import measure_agreement_per_prompt, measure_agreement_rate


class TestMeasureAgreementRate:
    def test_returns_in_unit_interval(self, mock_model):
        rate = measure_agreement_rate(mock_model, ['a', 'b', 'c'], (1, 2), (3, 4), batch_size=2)
        assert 0.0 <= rate <= 1.0

    def test_empty_prompts_returns_zero(self, mock_model):
        rate = measure_agreement_rate(mock_model, [], (1,), (2,))
        assert rate == 0.0

    def test_hooks_pass_through(self, mock_model):
        # supply a no-op hook spec so the run_with_hooks branch executes
        captured = []

        def hook_fn(t, _hook):
            captured.append(t.shape)
            return t

        rate = measure_agreement_rate(
            mock_model,
            ['p1', 'p2'],
            (1,),
            (2,),
            hooks=[('blocks.0.hook_resid_post', hook_fn)],
        )
        assert 0.0 <= rate <= 1.0


class TestMeasureAgreementPerPrompt:
    def test_returns_pair(self, mock_model):
        rate, per_prompt = measure_agreement_per_prompt(mock_model, ['a', 'b'], (1,), (2,))
        assert 0.0 <= rate <= 1.0
        assert len(per_prompt) == 2
        assert all(v in (0.0, 1.0) for v in per_prompt)

    @pytest.mark.parametrize('n', [1, 5, 10])
    def test_per_prompt_length_matches_input(self, mock_model, n):
        _, per_prompt = measure_agreement_per_prompt(mock_model, [f'p{i}' for i in range(n)], (1,), (2,))
        assert len(per_prompt) == n
