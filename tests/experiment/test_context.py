import dataclasses
from typing import cast

import pytest

from shared_circuits.experiment import ExperimentContext, model_session
from shared_circuits.models import ModelInfo


@pytest.fixture
def fake_info():
    return ModelInfo(name='gemma-2-2b-it', n_layers=4, n_heads=4, d_model=32, d_head=8, total_heads=16)


@pytest.fixture
def patched_session(mocker, fake_info):
    sentinel_model = object()
    load = mocker.patch('shared_circuits.experiment.context.load_model', return_value=sentinel_model)
    info_fn = mocker.patch('shared_circuits.experiment.context.get_model_info', return_value=fake_info)
    tokens_fn = mocker.patch(
        'shared_circuits.experiment.context.get_agree_disagree_tokens',
        return_value=([1, 2, 3], [4, 5, 6]),
    )
    cleanup = mocker.patch('shared_circuits.experiment.context.cleanup_model')
    return {
        'sentinel_model': sentinel_model,
        'load': load,
        'info_fn': info_fn,
        'tokens_fn': tokens_fn,
        'cleanup': cleanup,
    }


class TestModelSession:
    def test_yields_experiment_context(self, patched_session, fake_info):
        with model_session('gemma-2-2b-it') as ctx:
            assert isinstance(ctx, ExperimentContext)
            assert ctx.model is patched_session['sentinel_model']
            assert ctx.info == fake_info
            assert ctx.model_name == 'gemma-2-2b-it'
            assert ctx.agree_tokens == (1, 2, 3)
            assert ctx.disagree_tokens == (4, 5, 6)

    def test_passes_load_kwargs(self, patched_session):
        with model_session('m', device='cpu', dtype='float32', n_devices=2):
            pass
        patched_session['load'].assert_called_once_with('m', device='cpu', dtype='float32', n_devices=2)

    def test_cleanup_runs_on_normal_exit(self, patched_session):
        with model_session('m'):
            pass
        patched_session['cleanup'].assert_called_once_with(patched_session['sentinel_model'])

    def test_cleanup_runs_on_exception(self, patched_session):
        with pytest.raises(RuntimeError, match='boom'), model_session('m'):
            raise RuntimeError('boom')
        patched_session['cleanup'].assert_called_once_with(patched_session['sentinel_model'])


class TestExperimentContextImmutability:
    def test_is_frozen(self, fake_info):
        ctx = ExperimentContext(
            model=cast('HookedTransformer', object()),
            info=fake_info,
            model_name='m',
            agree_tokens=(1,),
            disagree_tokens=(2,),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.model_name = 'other'  # ty: ignore[invalid-assignment]

    def test_has_slots(self):
        assert ExperimentContext.__slots__ == (
            'model',
            'info',
            'model_name',
            'agree_tokens',
            'disagree_tokens',
        )
