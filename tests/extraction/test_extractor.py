import dataclasses

import numpy as np
import pytest

from shared_circuits.extraction import BatchedExtractor, HookSpec


class TestBatchedExtractorRun:
    def test_single_capture_hook_shape(self, mock_model):
        extractor = BatchedExtractor(mock_model, batch_size=4)
        out = extractor.run(['a', 'b', 'c'], capture_hooks={'r': 'blocks.1.hook_resid_post'})
        assert set(out.keys()) == {'r'}
        assert out['r'].shape == (3, mock_model.cfg.d_model)

    def test_multiple_capture_hooks(self, mock_model):
        extractor = BatchedExtractor(mock_model)
        out = extractor.run(
            ['x', 'y'],
            capture_hooks={
                'r0': 'blocks.0.hook_resid_post',
                'r2': 'blocks.2.hook_resid_post',
            },
        )
        assert set(out.keys()) == {'r0', 'r2'}
        assert out['r0'].shape == (2, mock_model.cfg.d_model)
        assert out['r2'].shape == (2, mock_model.cfg.d_model)

    def test_return_logits(self, mock_model):
        extractor = BatchedExtractor(mock_model)
        out = extractor.run(['hello'], capture_hooks={}, return_logits=True)
        assert 'logits' in out
        assert out['logits'].shape == (1, 100)

    def test_return_logits_alongside_capture(self, mock_model):
        extractor = BatchedExtractor(mock_model)
        out = extractor.run(
            ['hello', 'world'],
            capture_hooks={'r': 'blocks.0.hook_resid_post'},
            return_logits=True,
        )
        assert out['r'].shape == (2, mock_model.cfg.d_model)
        assert out['logits'].shape == (2, 100)

    def test_stop_at_layer_passes_through(self, mock_model, mocker):
        extractor = BatchedExtractor(mock_model)
        spy = mocker.spy(mock_model, 'run_with_hooks')
        extractor.run(['a'], capture_hooks={'r': 'blocks.1.hook_resid_post'}, stop_at_layer=2)
        assert spy.call_args.kwargs['stop_at_layer'] == 2

    def test_no_stop_at_layer_when_none(self, mock_model, mocker):
        extractor = BatchedExtractor(mock_model)
        spy = mocker.spy(mock_model, 'run_with_hooks')
        extractor.run(['a'], capture_hooks={'r': 'blocks.0.hook_resid_post'})
        assert 'stop_at_layer' not in spy.call_args.kwargs

    def test_mutate_hooks_fire(self, mock_model):
        seen: list[str] = []

        def record(z, hook):
            seen.append('fired')
            return z

        extractor = BatchedExtractor(mock_model)
        extractor.run(
            ['hello'],
            mutate_hooks=[HookSpec(name='blocks.0.attn.hook_z', fn=record)],
            return_logits=True,
        )
        assert seen == ['fired']

    def test_returns_numpy(self, mock_model):
        extractor = BatchedExtractor(mock_model)
        out = extractor.run(['hello'], capture_hooks={'r': 'blocks.0.hook_resid_post'})
        assert isinstance(out['r'], np.ndarray)

    def test_batching_concatenates(self, mock_model):
        extractor = BatchedExtractor(mock_model, batch_size=3)
        prompts = [f'p{i}' for i in range(10)]
        out = extractor.run(prompts, capture_hooks={'r': 'blocks.0.hook_resid_post'})
        assert out['r'].shape == (10, mock_model.cfg.d_model)


class TestHookSpec:
    def test_is_frozen(self):
        spec = HookSpec(name='blocks.0.attn.hook_z', fn=lambda z, h: z)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            spec.name = 'changed'  # ty: ignore[invalid-assignment]
