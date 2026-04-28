from unittest.mock import MagicMock

from pytest_mock import MockerFixture

from shared_circuits.prompts.chat import _get_tokenizer, render_chat


class TestGetTokenizer:
    def test_caches_per_model_name(self, mocker: MockerFixture) -> None:
        sentinel = MagicMock(name='tokenizer-sentinel')
        spy = mocker.patch(
            'shared_circuits.prompts.chat.AutoTokenizer.from_pretrained',
            return_value=sentinel,
        )
        first = _get_tokenizer('gemma-2-2b-it')
        second = _get_tokenizer('gemma-2-2b-it')
        assert first is second
        assert spy.call_count == 1

    def test_different_models_load_independently(self, mocker: MockerFixture) -> None:
        spy = mocker.patch(
            'shared_circuits.prompts.chat.AutoTokenizer.from_pretrained',
            side_effect=lambda name: MagicMock(name=name),
        )
        a = _get_tokenizer('gemma-2-2b-it')
        b = _get_tokenizer('Qwen/Qwen2.5-1.5B-Instruct')
        assert a is not b
        assert spy.call_count == 2


class TestRenderChat:
    def test_returns_non_empty_string(self, stub_tokenizer: MagicMock) -> None:
        out = render_chat([{'role': 'user', 'content': 'hello'}], 'gemma-2-2b-it')
        assert isinstance(out, str)
        assert out
        assert 'hello' in out

    def test_passes_add_generation_prompt(self, stub_tokenizer: MagicMock) -> None:
        render_chat([{'role': 'user', 'content': 'hi'}], 'gemma-2-2b-it')
        _, kwargs = stub_tokenizer.apply_chat_template.call_args
        assert kwargs['tokenize'] is False
        assert kwargs['add_generation_prompt'] is True
