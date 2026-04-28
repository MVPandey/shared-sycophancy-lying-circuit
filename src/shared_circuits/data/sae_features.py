"""
HuggingFace-Hub SAE checkpoint loader.

Two SAE families are used in the paper:
- Gemma-Scope (``google/gemma-scope-<size>-pt-res``): Gemma-2 residual JumpReLU SAEs
  with weights stored as ``params.npz`` under
  ``layer_<L>/width_16k/average_l0_<X>/``. We pick the ``average_l0_X``
  closest to ``_TARGET_L0`` to match the legacy script's default.
- Goodfire (``Goodfire/Llama-3.1-8B-Instruct-SAE-l19``): Llama-3.1/3.3 top-K
  SAEs distributed as ``.pth`` / ``.pt`` torch state-dicts with keys
  ``encoder_linear.weight``, ``encoder_linear.bias``, ``decoder_linear.bias``,
  ``decoder_linear.weight``.

Both formats expose the same ``SaeWeights`` shape so downstream code can
encode without branching on the family.
"""

import functools
import re
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

# Gemma-Scope hosts multiple sparsity tiers per layer. The legacy run scripts
# pick the one whose average_l0 is closest to 80, matching the paper's setting.
_TARGET_L0: Final = 80
_L0_PATTERN: Final = re.compile(r'average_l0_(\d+)')

SAE_KIND_GEMMA_SCOPE: Final = 'gemma_scope'
SAE_KIND_GOODFIRE_TOPK: Final = 'goodfire_topk'

# Maps HF model identifier to the SAE repo + format the paper used.
SAE_REPOS: Final[dict[str, dict[str, object]]] = {
    'gemma-2-2b-it': {'repo': 'google/gemma-scope-2b-pt-res', 'format': SAE_KIND_GEMMA_SCOPE},
    'google/gemma-2-2b-it': {'repo': 'google/gemma-scope-2b-pt-res', 'format': SAE_KIND_GEMMA_SCOPE},
    'google/gemma-2-9b-it': {'repo': 'google/gemma-scope-9b-pt-res', 'format': SAE_KIND_GEMMA_SCOPE},
    'meta-llama/Llama-3.1-8B-Instruct': {
        'repo': 'Goodfire/Llama-3.1-8B-Instruct-SAE-l19',
        'format': SAE_KIND_GOODFIRE_TOPK,
        'filename': 'Llama-3.1-8B-Instruct-SAE-l19.pth',
        'topk': 91,
    },
    'meta-llama/Llama-3.3-70B-Instruct': {
        'repo': 'Goodfire/Llama-3.3-70B-Instruct-SAE-l50',
        'format': SAE_KIND_GOODFIRE_TOPK,
        'filename': 'Llama-3.3-70B-Instruct-SAE-l50.pt',
        'topk': 121,
    },
}


@dataclass(frozen=True, slots=True)
class SaeWeights:
    """A loaded SAE: encoder/decoder weights + bias terms + family-specific gating params."""

    kind: str
    layer: int
    repo: str
    # ``W_enc``: (d_model, d_sae). ``b_enc``: (d_sae,). ``b_dec``: (d_model,).
    w_enc: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor
    # ``W_dec``: (d_model, d_sae) — present only for analyses that need decoder columns.
    w_dec: torch.Tensor | None = None
    # JumpReLU threshold for Gemma-Scope; None for Goodfire top-K.
    threshold: torch.Tensor | None = None
    # Top-K cardinality for Goodfire; None for Gemma-Scope JumpReLU.
    top_k: int | None = None
    # ``average_l0_X`` chosen for Gemma-Scope (None for Goodfire).
    average_l0: int | None = None

    @property
    def d_model(self) -> int:
        """Residual-stream dimension this SAE expects as input."""
        return int(self.w_enc.shape[0])

    @property
    def d_sae(self) -> int:
        """Number of SAE feature dictionary atoms."""
        return int(self.w_enc.shape[1])


def repo_for_model(model_name: str) -> dict[str, object]:
    """
    Return the SAE repo metadata registered for ``model_name``.

    Args:
        model_name: HuggingFace model name (must be present in :data:`SAE_REPOS`).

    Returns:
        Mapping with ``repo``, ``format``, and family-specific keys.

    Raises:
        ValueError: If no SAE repo is registered for the model.

    """
    cfg = SAE_REPOS.get(model_name)
    if cfg is None:
        raise ValueError(f'No SAE repo registered for {model_name}; supported: {sorted(SAE_REPOS)}')
    return cfg


def _pick_l0(repo: str, layer: int) -> int:
    """Pick the ``average_l0`` value closest to :data:`_TARGET_L0` for the given layer/width_16k."""
    api = HfApi()
    entries = api.list_repo_tree(repo_id=repo, path_in_repo=f'layer_{layer}/width_16k', recursive=False)
    l0s: list[int] = []
    for e in entries:
        m = _L0_PATTERN.search(e.path)
        if m:
            l0s.append(int(m.group(1)))
    if not l0s:
        raise FileNotFoundError(f'No average_l0_* dirs at {repo}/layer_{layer}/width_16k')
    return min(l0s, key=lambda v: abs(v - _TARGET_L0))


def _load_gemma_scope(repo: str, layer: int, device: str, dtype: torch.dtype) -> SaeWeights:
    l0 = _pick_l0(repo, layer)
    fname = f'layer_{layer}/width_16k/average_l0_{l0}/params.npz'
    path = hf_hub_download(repo_id=repo, filename=fname)
    arr = np.load(path)
    w_enc = torch.tensor(np.asarray(arr['W_enc']), dtype=dtype, device=device)
    b_enc = torch.tensor(np.asarray(arr['b_enc']), dtype=dtype, device=device)
    b_dec = torch.tensor(np.asarray(arr['b_dec']), dtype=dtype, device=device)
    threshold = torch.tensor(np.asarray(arr['threshold']), dtype=dtype, device=device)
    # Gemma-Scope stores W_dec as (d_sae, d_model); transpose for column-per-feature access.
    w_dec_raw = np.asarray(arr['W_dec'])
    w_dec = torch.tensor(w_dec_raw.T, dtype=dtype, device=device).contiguous()
    return SaeWeights(
        kind=SAE_KIND_GEMMA_SCOPE,
        layer=layer,
        repo=repo,
        w_enc=w_enc,
        b_enc=b_enc,
        b_dec=b_dec,
        w_dec=w_dec,
        threshold=threshold,
        average_l0=l0,
    )


def _load_goodfire(repo: str, filename: str, top_k: int, layer: int, device: str, dtype: torch.dtype) -> SaeWeights:
    path = hf_hub_download(repo_id=repo, filename=filename)
    # ``weights_only=False`` required: legacy Goodfire checkpoints contain pickled metadata.
    sd = torch.load(path, map_location='cpu', weights_only=False)
    enc_w = sd['encoder_linear.weight']
    enc_b = sd['encoder_linear.bias']
    dec_b = sd['decoder_linear.bias']
    # Goodfire stores encoder as nn.Linear weight (d_sae, d_model); transpose so W_enc is (d_model, d_sae).
    w_enc = enc_w.T.contiguous().to(device=device, dtype=dtype)
    b_enc = enc_b.to(device=device, dtype=dtype)
    b_dec = dec_b.to(device=device, dtype=dtype)
    w_dec_raw = sd.get('decoder_linear.weight')
    # Goodfire decoder is already (d_model, d_sae): one column per feature.
    w_dec = w_dec_raw.to(device=device, dtype=dtype).contiguous() if w_dec_raw is not None else None
    return SaeWeights(
        kind=SAE_KIND_GOODFIRE_TOPK,
        layer=layer,
        repo=repo,
        w_enc=w_enc,
        b_enc=b_enc,
        b_dec=b_dec,
        w_dec=w_dec,
        top_k=top_k,
    )


@functools.cache
def load_sae(
    repo: str,
    layer: int,
    sae_kind: str = 'auto',
    *,
    device: str = 'cpu',
    dtype: str = 'bfloat16',
    filename: str | None = None,
    top_k: int | None = None,
) -> SaeWeights:
    """
    Download and cache an SAE checkpoint from HuggingFace Hub.

    Args:
        repo: HuggingFace repo id (e.g. ``'google/gemma-scope-2b-pt-res'``).
        layer: Residual-stream layer the SAE was trained on.
        sae_kind: Either an explicit family in {``'gemma_scope'``, ``'goodfire_topk'``}, or
            ``'auto'`` to infer from the repo name.
        device: Torch device target for the loaded tensors.
        dtype: Torch dtype string (e.g. ``'bfloat16'``).
        filename: Goodfire file inside the repo (required when kind resolves to goodfire_topk).
        top_k: Goodfire top-K cardinality (required when kind resolves to goodfire_topk).

    Returns:
        An :class:`SaeWeights` instance ready for encoder application.

    Raises:
        ValueError: If a Goodfire SAE is requested without ``filename`` / ``top_k``,
            or if ``sae_kind`` is unrecognised.

    """
    kind = sae_kind if sae_kind != 'auto' else _infer_kind(repo)
    torch_dtype = _dtype_from_str(dtype)
    if kind == SAE_KIND_GEMMA_SCOPE:
        return _load_gemma_scope(repo, layer, device, torch_dtype)
    if kind == SAE_KIND_GOODFIRE_TOPK:
        if filename is None or top_k is None:
            raise ValueError('Goodfire SAE requires filename and top_k')
        return _load_goodfire(repo, filename, top_k, layer, device, torch_dtype)
    raise ValueError(f'Unknown SAE kind: {sae_kind}')


def load_sae_for_model(
    model_name: str,
    layer: int,
    *,
    device: str = 'cpu',
    dtype: str = 'bfloat16',
) -> SaeWeights:
    """
    Look up the registered repo for ``model_name`` and load the SAE.

    Args:
        model_name: HuggingFace model name (must be in :data:`SAE_REPOS`).
        layer: Residual-stream layer the SAE was trained on.
        device: Torch device target.
        dtype: Torch dtype string.

    Returns:
        Loaded :class:`SaeWeights`.

    """
    cfg = repo_for_model(model_name)
    kind = str(cfg['format'])
    repo = str(cfg['repo'])
    if kind == SAE_KIND_GEMMA_SCOPE:
        return load_sae(repo, layer, kind, device=device, dtype=dtype)
    raw_topk = cfg['topk']
    if not isinstance(raw_topk, int):
        raise TypeError(f'SAE_REPOS["{model_name}"]["topk"] must be int, got {type(raw_topk).__name__}')
    return load_sae(
        repo,
        layer,
        kind,
        device=device,
        dtype=dtype,
        filename=str(cfg['filename']),
        top_k=raw_topk,
    )


def _infer_kind(repo: str) -> str:
    if 'gemma-scope' in repo.lower():
        return SAE_KIND_GEMMA_SCOPE
    if 'goodfire' in repo.lower():
        return SAE_KIND_GOODFIRE_TOPK
    raise ValueError(f'Could not infer SAE kind from repo {repo!r}; pass sae_kind explicitly')


def _dtype_from_str(dtype: str) -> torch.dtype:
    mapping: dict[str, torch.dtype] = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f'Unsupported dtype string: {dtype}')
    return mapping[dtype]
