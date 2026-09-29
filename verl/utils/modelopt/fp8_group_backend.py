# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP8 fake quantization backends for ModelOpt, matching what the rollout runs.

Two shapes, because the rollout engine quantizes the two operands differently:

* **Activations**, one scale per 128 values along the reduction axis, recomputed
  every forward. On ROCm that is ``aiter``'s ``per_1x128``, reached through
  ``QuantFP8(group_shape=GroupShape(1, 128))``.
* **Weights**, one scale per 128x128 tile, which is how the checkpoint stores
  them and what the weight sync writes back.

ModelOpt can express per-group dynamic blocking, but routes it through
``modelopt_cuda_ext_mx``, which is CUDA-only, so these are pure-PyTorch paths.

``e8m0_scale`` rounds the scale up to a power of two, and the two operands need
opposite answers:

* Activation scales stay plain float32. vLLM's block-scaled kernel constructs
  ``QuantFP8`` with ``use_ue8m0=False`` outright, and the DeepGEMM path that
  would do otherwise needs Hopper/Blackwell.
* Weight scales are powers of two. DeepSeek-V4 declares ``scale_fmt: ue8m0`` and
  stores them that way, and ``fp8_scale_patch`` keeps the weight sync on that
  grid, so the rollout runs powers of two and training has to match.
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

BACKEND_NAME = "verl_fp8_group"
BLOCK_BACKEND_NAME = "verl_fp8_block2d"

_FP8_E4M3_MAX = 448.0


def _quantize_with_scale(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (values / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn).float() * scale


def _scale_from_amax(amax: torch.Tensor, e8m0_scale: bool) -> torch.Tensor:
    # An all-zero block would otherwise divide by zero; its values quantize to 0
    # either way.
    scale = (amax / _FP8_E4M3_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    if e8m0_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    return scale


def fp8_group_fake_quant(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    """Fake-quantize ``inputs`` to E4M3 with one scale per group along the last axis.

    ``quantizer.backend_extra_args`` accepts:
        group_size (int): values per scale, default 128.
        e8m0_scale (bool): round scales up to a power of two, default False.
    """
    extra = getattr(quantizer, "backend_extra_args", None) or {}
    group_size = int(extra.get("group_size", 128))
    e8m0_scale = bool(extra.get("e8m0_scale", False))

    orig_dtype = inputs.dtype
    last = inputs.shape[-1]
    pad = (-last) % group_size
    padded = F.pad(inputs, (0, pad)) if pad else inputs

    grouped = padded.float().reshape(*padded.shape[:-1], padded.shape[-1] // group_size, group_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = _scale_from_amax(amax, e8m0_scale)

    quantized = _quantize_with_scale(grouped, scale).reshape(*padded.shape)
    if pad:
        quantized = quantized[..., :last]
    quantized = quantized.to(orig_dtype)

    # Straight-through estimator: the rounding is not differentiable, so the
    # gradient passes to the unquantized input unchanged.
    return inputs + (quantized - inputs).detach()


def _static_block_keep_axes(quantizer, inputs: torch.Tensor) -> set[int] | None:
    """Axes ModelOpt wants one scale per, when it has already tiled ``inputs``.

    A quantizer whose ``block_sizes`` carry no ``"type": "dynamic"`` is a *static*
    block quantizer, and ``TensorQuantizer.forward`` reshapes the tensor into its
    tiled view before handing it to the backend: a ``[rows, cols]`` weight blocked
    on both axes arrives as ``[rows/128, 128, cols/128, 128]``. ``_axis`` then
    names the axes that carry one scale each, so the tile is what is left over.

    Returns ``None`` when the tensor reached the backend unreshaped.
    """
    if not getattr(quantizer, "is_static_block_quant", False):
        return None
    axis = getattr(quantizer, "_axis", None)
    if axis is None:
        return None
    ndim = inputs.dim()
    keep = {a % ndim for a in axis}
    return keep if len(keep) < ndim else None


def fp8_block2d_fake_quant(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    """Fake-quantize a weight to E4M3 with one scale per 128x128 tile.

    ModelOpt's own 2D blocking would do this, but computes the scale as a plain
    ``amax / 448``. The checkpoint's grid is powers of two and the weight sync
    stays on it, so the default would train against weights the rollout never
    holds -- a ~2.8e-02 difference, which is what ``fp8_scale_patch`` exists to
    avoid on the export side.

    ``quantizer.backend_extra_args`` accepts:
        block_size (int): tile edge, default 128.
        e8m0_scale (bool): round scales up to a power of two, default True.
    """
    extra = getattr(quantizer, "backend_extra_args", None) or {}
    block = int(extra.get("block_size", 128))
    e8m0_scale = bool(extra.get("e8m0_scale", True))

    keep_axes = _static_block_keep_axes(quantizer, inputs)
    if keep_axes is not None:
        # Already tiled, and padded to the block edge by the caller. Reducing over
        # the axes ModelOpt left out of ``_axis`` is what makes this a 128x128
        # scale; falling through to the group path below would instead put one
        # scale on each 1x128 row of a tile, which is a finer grid than the
        # rollout's and so leaves the two engines rounding differently.
        values = inputs.float()
        reduce_dims = tuple(d for d in range(inputs.dim()) if d not in keep_axes)
        scale = _scale_from_amax(values.abs().amax(dim=reduce_dims, keepdim=True), e8m0_scale)
        quantized = _quantize_with_scale(values, scale).to(inputs.dtype)
        return inputs + (quantized - inputs).detach()

    if inputs.dim() != 2:
        return fp8_group_fake_quant(inputs, quantizer)

    orig_dtype = inputs.dtype
    rows, cols = inputs.shape
    pad_r, pad_c = (-rows) % block, (-cols) % block
    padded = F.pad(inputs, (0, pad_c, 0, pad_r)) if (pad_r or pad_c) else inputs

    tiled = padded.float().reshape(padded.shape[0] // block, block, padded.shape[1] // block, block)
    amax = tiled.abs().amax(dim=(1, 3), keepdim=True)
    scale = _scale_from_amax(amax, e8m0_scale)

    quantized = _quantize_with_scale(tiled, scale).reshape(padded.shape)
    if pad_r or pad_c:
        quantized = quantized[:rows, :cols]
    quantized = quantized.to(orig_dtype)

    return inputs + (quantized - inputs).detach()


def register() -> bool:
    """Register the backends with ModelOpt. Returns whether they are available."""
    try:
        from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
            is_registered_quant_backend,
            register_quant_backend,
        )
    except ImportError:
        logger.info("ModelOpt quant backend registry unavailable; skipping %s", BACKEND_NAME)
        return False

    for name, fn in ((BACKEND_NAME, fp8_group_fake_quant), (BLOCK_BACKEND_NAME, fp8_block2d_fake_quant)):
        if not is_registered_quant_backend(name):
            register_quant_backend(name, fn)
    return True
