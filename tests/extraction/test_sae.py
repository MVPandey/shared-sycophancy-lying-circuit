"""Tests for the SAE encoding helpers."""

from typing import Final

import numpy as np
import pytest
import torch

from shared_circuits.data.sae_features import SAE_KIND_GEMMA_SCOPE, SAE_KIND_GOODFIRE_TOPK, SaeWeights
from shared_circuits.extraction import sae as sae_mod
from shared_circuits.extraction.sae import encode_prompts, encode_residuals, feature_activation_grid

_D_MODEL: Final = 8
_D_SAE: Final = 16
_N_PROMPTS: Final = 5


def _build_gemma_sae(*, threshold: float = 0.0) -> SaeWeights:
    rng = np.random.RandomState(0)
    w_enc = torch.tensor(rng.randn(_D_MODEL, _D_SAE).astype(np.float32))
    b_enc = torch.zeros(_D_SAE, dtype=torch.float32)
    b_dec = torch.zeros(_D_MODEL, dtype=torch.float32)
    return SaeWeights(
        kind=SAE_KIND_GEMMA_SCOPE,
        layer=0,
        repo='unit/test',
        w_enc=w_enc,
        b_enc=b_enc,
        b_dec=b_dec,
        threshold=torch.full((_D_SAE,), threshold, dtype=torch.float32),
    )


def _build_goodfire_sae(*, top_k: int = 3) -> SaeWeights:
    rng = np.random.RandomState(1)
    w_enc = torch.tensor(rng.randn(_D_MODEL, _D_SAE).astype(np.float32))
    b_enc = torch.zeros(_D_SAE, dtype=torch.float32)
    b_dec = torch.zeros(_D_MODEL, dtype=torch.float32)
    return SaeWeights(
        kind=SAE_KIND_GOODFIRE_TOPK,
        layer=0,
        repo='unit/test',
        w_enc=w_enc,
        b_enc=b_enc,
        b_dec=b_dec,
        top_k=top_k,
    )


@pytest.fixture
def residuals():
    rng = np.random.RandomState(42)
    return torch.tensor(rng.randn(_N_PROMPTS, _D_MODEL).astype(np.float32))


class TestEncodeResidualsGemma:
    def test_returns_correct_shape(self, residuals):
        sae = _build_gemma_sae(threshold=0.0)
        feats = encode_residuals(residuals, sae)
        assert feats.shape == (_N_PROMPTS, _D_SAE)

    def test_threshold_zeroes_subthreshold(self, residuals):
        # very large threshold => all features below threshold => all zero
        sae = _build_gemma_sae(threshold=1e9)
        feats = encode_residuals(residuals, sae)
        assert torch.allclose(feats, torch.zeros_like(feats))

    def test_threshold_required(self, residuals):
        sae = _build_gemma_sae()
        # remove threshold via dataclasses.replace would also test, but dataclass is frozen;
        # fabricate a SaeWeights without threshold by direct construction
        broken = SaeWeights(
            kind=SAE_KIND_GEMMA_SCOPE,
            layer=0,
            repo='x',
            w_enc=sae.w_enc,
            b_enc=sae.b_enc,
            b_dec=sae.b_dec,
            threshold=None,
        )
        with pytest.raises(ValueError, match='requires a threshold'):
            encode_residuals(residuals, broken)


class TestEncodeResidualsGoodfire:
    def test_top_k_sparsity(self, residuals):
        sae = _build_goodfire_sae(top_k=3)
        feats = encode_residuals(residuals, sae)
        # at most top_k features active per prompt
        n_active = (feats != 0).sum(dim=-1)
        assert (n_active <= 3).all()

    def test_required_top_k(self, residuals):
        sae = _build_goodfire_sae(top_k=2)
        broken = SaeWeights(
            kind=SAE_KIND_GOODFIRE_TOPK,
            layer=0,
            repo='x',
            w_enc=sae.w_enc,
            b_enc=sae.b_enc,
            b_dec=sae.b_dec,
            top_k=None,
        )
        with pytest.raises(ValueError, match='requires top_k'):
            encode_residuals(residuals, broken)


class TestEncodeResidualsTopKOverride:
    def test_extra_top_k_filters_more(self, residuals):
        sae = _build_gemma_sae(threshold=-10.0)  # all features active
        all_feats = encode_residuals(residuals, sae)
        sparse = encode_residuals(residuals, sae, top_k=2)
        # original would have many active; sparse keeps at most 2 per row
        assert (all_feats != 0).sum() > (sparse != 0).sum()
        n_active = (sparse != 0).sum(dim=-1)
        assert (n_active <= 2).all()


class TestEncodeResidualsBadKind:
    def test_unknown_kind_raises(self, residuals):
        broken = SaeWeights(
            kind='unknown',
            layer=0,
            repo='x',
            w_enc=torch.zeros(_D_MODEL, _D_SAE),
            b_enc=torch.zeros(_D_SAE),
            b_dec=torch.zeros(_D_MODEL),
        )
        with pytest.raises(ValueError, match='Unknown SAE kind'):
            encode_residuals(residuals, broken)


class TestFeatureActivationGrid:
    def test_returns_indices_and_values(self, residuals):
        sae = _build_gemma_sae(threshold=0.0)
        idx, vals = feature_activation_grid(residuals, sae, top_k=4)
        assert idx.shape == (_N_PROMPTS, 4)
        assert vals.shape == (_N_PROMPTS, 4)

    def test_top_k_capped_at_d_sae(self, residuals):
        sae = _build_gemma_sae(threshold=0.0)
        idx, _ = feature_activation_grid(residuals, sae, top_k=999)
        assert idx.shape[-1] == _D_SAE

    def test_indices_are_distinct_per_prompt(self, residuals):
        sae = _build_gemma_sae(threshold=-1.0)  # all features active
        idx, _ = feature_activation_grid(residuals, sae, top_k=5)
        for row in idx:
            assert len(set(row.tolist())) == 5


class TestEncodePrompts:
    def test_calls_extract_residual_and_encode(self, mock_model, mocker):
        sae = _build_gemma_sae(threshold=-1.0)
        captured_resid = np.random.RandomState(0).randn(3, _D_MODEL).astype(np.float32)
        mocker.patch.object(sae_mod, 'extract_residual_stream', return_value=captured_resid)

        out = encode_prompts(mock_model, ['a', 'b', 'c'], sae, layer=2, batch_size=2)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3, _D_SAE)

    def test_passes_through_layer_and_batch(self, mock_model, mocker):
        sae = _build_gemma_sae(threshold=-1.0)
        spy = mocker.patch.object(
            sae_mod,
            'extract_residual_stream',
            return_value=np.zeros((1, _D_MODEL), dtype=np.float32),
        )
        encode_prompts(mock_model, ['only'], sae, layer=5, batch_size=8)
        kwargs = spy.call_args.kwargs
        args = spy.call_args.args
        assert args[2] == 5  # layer
        assert kwargs.get('batch_size') == 8
