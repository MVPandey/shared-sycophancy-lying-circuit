"""Tests for the HuggingFace-Hub SAE checkpoint loader."""

from typing import Final
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from shared_circuits.data import sae_features
from shared_circuits.data.sae_features import (
    SAE_KIND_GEMMA_SCOPE,
    SAE_KIND_GOODFIRE_TOPK,
    SAE_REPOS,
    SaeWeights,
    load_sae,
    load_sae_for_model,
    repo_for_model,
)

_D_MODEL: Final = 8
_D_SAE: Final = 16


def _gemma_scope_npz_arrays() -> dict[str, np.ndarray]:
    rng = np.random.RandomState(0)
    return {
        'W_enc': rng.randn(_D_MODEL, _D_SAE).astype(np.float32),
        'W_dec': rng.randn(_D_SAE, _D_MODEL).astype(np.float32),
        'b_enc': rng.randn(_D_SAE).astype(np.float32),
        'b_dec': rng.randn(_D_MODEL).astype(np.float32),
        'threshold': np.full((_D_SAE,), 0.1, dtype=np.float32),
    }


def _goodfire_state_dict() -> dict[str, torch.Tensor]:
    return {
        'encoder_linear.weight': torch.randn(_D_SAE, _D_MODEL),
        'encoder_linear.bias': torch.randn(_D_SAE),
        'decoder_linear.weight': torch.randn(_D_MODEL, _D_SAE),
        'decoder_linear.bias': torch.randn(_D_MODEL),
    }


@pytest.fixture(autouse=True)
def _clear_cache():
    load_sae.cache_clear()
    yield
    load_sae.cache_clear()


@pytest.fixture
def patched_gemma(mocker, tmp_path):
    """Stub the Gemma-Scope hub-listing + npz download to deterministic synthetic values."""
    npz_path = tmp_path / 'params.npz'
    arrays = _gemma_scope_npz_arrays()
    np.savez(str(npz_path), **arrays)  # ty: ignore[invalid-argument-type]

    api = MagicMock()
    api.list_repo_tree.return_value = [
        MagicMock(path='layer_12/width_16k/average_l0_42'),
        MagicMock(path='layer_12/width_16k/average_l0_85'),
        MagicMock(path='layer_12/width_16k/average_l0_140'),
    ]
    mocker.patch.object(sae_features, 'HfApi', return_value=api)
    mocker.patch.object(sae_features, 'hf_hub_download', return_value=str(npz_path))
    return npz_path


@pytest.fixture
def patched_goodfire(mocker, tmp_path):
    """Stub the Goodfire torch-load path with a synthetic state-dict."""
    sd_path = tmp_path / 'goodfire.pt'
    torch.save(_goodfire_state_dict(), sd_path)
    mocker.patch.object(sae_features, 'hf_hub_download', return_value=str(sd_path))
    return sd_path


class TestSaeWeights:
    def test_d_model_and_d_sae(self):
        w = SaeWeights(
            kind=SAE_KIND_GOODFIRE_TOPK,
            layer=19,
            repo='Goodfire/x',
            w_enc=torch.zeros(_D_MODEL, _D_SAE),
            b_enc=torch.zeros(_D_SAE),
            b_dec=torch.zeros(_D_MODEL),
            top_k=4,
        )
        assert w.d_model == _D_MODEL
        assert w.d_sae == _D_SAE

    def test_is_frozen(self):
        w = SaeWeights(
            kind=SAE_KIND_GOODFIRE_TOPK,
            layer=19,
            repo='Goodfire/x',
            w_enc=torch.zeros(_D_MODEL, _D_SAE),
            b_enc=torch.zeros(_D_SAE),
            b_dec=torch.zeros(_D_MODEL),
            top_k=4,
        )
        with pytest.raises((AttributeError,)):
            w.layer = 99  # ty: ignore[invalid-assignment]


class TestRepoForModel:
    def test_returns_known_model(self):
        cfg = repo_for_model('gemma-2-2b-it')
        assert cfg['repo'] == 'google/gemma-scope-2b-pt-res'
        assert cfg['format'] == SAE_KIND_GEMMA_SCOPE

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match='No SAE repo registered'):
            repo_for_model('does-not-exist')

    def test_table_covers_paper_models(self):
        for required in (
            'gemma-2-2b-it',
            'google/gemma-2-9b-it',
            'meta-llama/Llama-3.1-8B-Instruct',
            'meta-llama/Llama-3.3-70B-Instruct',
        ):
            assert required in SAE_REPOS


class TestLoadSae:
    def test_loads_gemma_scope(self, patched_gemma):
        sae = load_sae('google/gemma-scope-2b-pt-res', layer=12, dtype='float32')
        assert sae.kind == SAE_KIND_GEMMA_SCOPE
        assert sae.d_model == _D_MODEL
        assert sae.d_sae == _D_SAE
        assert sae.threshold is not None
        assert sae.average_l0 == 85
        assert sae.w_dec is not None
        assert sae.w_dec.shape == (_D_MODEL, _D_SAE)

    def test_loads_goodfire(self, patched_goodfire):
        sae = load_sae(
            'Goodfire/Llama-3.1-8B-Instruct-SAE-l19',
            layer=19,
            sae_kind=SAE_KIND_GOODFIRE_TOPK,
            filename='Llama-3.1-8B-Instruct-SAE-l19.pth',
            top_k=4,
            dtype='float32',
        )
        assert sae.kind == SAE_KIND_GOODFIRE_TOPK
        assert sae.top_k == 4
        assert sae.d_model == _D_MODEL
        assert sae.d_sae == _D_SAE
        assert sae.w_dec is not None
        assert sae.w_dec.shape == (_D_MODEL, _D_SAE)

    def test_goodfire_requires_filename(self):
        with pytest.raises(ValueError, match='filename and top_k'):
            load_sae('Goodfire/x', layer=19, sae_kind=SAE_KIND_GOODFIRE_TOPK)

    def test_unknown_kind_raises(self, patched_gemma):
        with pytest.raises(ValueError, match='Unknown SAE kind'):
            load_sae('something/random', layer=0, sae_kind='unknown_kind')

    def test_auto_detects_gemma(self, patched_gemma):
        sae = load_sae('google/gemma-scope-2b-pt-res', layer=12, dtype='float32')
        assert sae.kind == SAE_KIND_GEMMA_SCOPE

    def test_auto_detects_goodfire(self, patched_goodfire):
        sae = load_sae(
            'Goodfire/Llama-3.1-8B-Instruct-SAE-l19',
            layer=19,
            filename='Llama-3.1-8B-Instruct-SAE-l19.pth',
            top_k=4,
            dtype='float32',
        )
        assert sae.kind == SAE_KIND_GOODFIRE_TOPK

    def test_caches_repeat_calls(self, patched_gemma, mocker):
        spy = mocker.spy(sae_features, 'hf_hub_download')
        load_sae('google/gemma-scope-2b-pt-res', layer=12, dtype='float32')
        load_sae('google/gemma-scope-2b-pt-res', layer=12, dtype='float32')
        assert spy.call_count == 1

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match='Unsupported dtype'):
            load_sae('google/gemma-scope-2b-pt-res', layer=12, dtype='int8')


class TestLoadSaeForModel:
    def test_dispatches_to_gemma(self, patched_gemma):
        sae = load_sae_for_model('gemma-2-2b-it', layer=12, dtype='float32')
        assert sae.kind == SAE_KIND_GEMMA_SCOPE

    def test_dispatches_to_goodfire(self, patched_goodfire):
        sae = load_sae_for_model('meta-llama/Llama-3.1-8B-Instruct', layer=19, dtype='float32')
        assert sae.kind == SAE_KIND_GOODFIRE_TOPK
        assert sae.top_k == 91
