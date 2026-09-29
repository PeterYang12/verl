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

"""Keep the RL weight sync on the checkpoint's power-of-two FP8 scale grid.

Megatron-Bridge picks the scale grid in ``scale_from_amax``, which snaps to a
power of two only when the scale tensor's dtype is ``float8_e8m0fnu``.
DeepSeek-V4 checkpoints declare ``scale_fmt: ue8m0`` and do store powers of two,
but as ``float32``, so the export recomputes ``amax / 448`` and re-rounds every
weight onto a different grid -- ~2.6% RMS error per sync even when no weight
changed. This decides from the scale values instead.
"""

import logging
from contextvars import ContextVar

import torch

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_verl_ue8m0_scale_patched"

# ``scale_from_amax`` does not receive the source scale and the call sites in
# between are private helpers, so this is what keeps the patch to two functions.
_snap_to_pow2: ContextVar[bool] = ContextVar("verl_snap_fp8_scale_to_pow2", default=False)


def _is_power_of_two_scale(source_scale: torch.Tensor) -> bool:
    if not isinstance(source_scale, torch.Tensor) or source_scale.numel() == 0:
        return False
    values = source_scale.detach().float().flatten()
    values = values[values > 0]
    if values.numel() == 0:
        return False
    return bool(torch.all(torch.log2(values).frac() == 0))


def apply_fp8_ue8m0_scale_patch() -> bool:
    """Patch Megatron-Bridge's FP8 export to preserve a ue8m0 scale grid.

    Returns whether the patch is in place.
    """
    try:
        from megatron.bridge.models.conversion import quantization_utils as qu
    except ImportError:
        logger.info("Megatron-Bridge quantization utils unavailable; skipping ue8m0 scale patch.")
        return False

    if not hasattr(qu, "scale_from_amax") or not hasattr(qu, "quantize_fp8_e4m3fn_like_scale"):
        logger.warning("Megatron-Bridge has no FP8 requantization helpers; skipping ue8m0 scale patch.")
        return False

    if getattr(qu, _PATCH_FLAG, False):
        return True

    original_scale_from_amax = qu.scale_from_amax
    original_quantize_like_scale = qu.quantize_fp8_e4m3fn_like_scale

    def scale_from_amax(amax, max_quantized_value, scale_dtype):
        scale = original_scale_from_amax(amax, max_quantized_value, scale_dtype)
        if not _snap_to_pow2.get():
            return scale
        scale = scale.clamp(min=2.0**-127, max=2.0**127)
        return torch.exp2(torch.ceil(torch.log2(scale)))

    def quantize_fp8_e4m3fn_like_scale(weight, source_scale, **kwargs):
        token = _snap_to_pow2.set(_is_power_of_two_scale(source_scale))
        try:
            return original_quantize_like_scale(weight, source_scale, **kwargs)
        finally:
            _snap_to_pow2.reset(token)

    qu.scale_from_amax = scale_from_amax
    qu.quantize_fp8_e4m3fn_like_scale = quantize_fp8_e4m3fn_like_scale
    setattr(qu, _PATCH_FLAG, True)
    # Printed, not logged: a node that silently misses this syncs weights on a
    # different grid than the rest.
    print("Applying ue8m0 FP8 scale patch to Megatron-Bridge weight export...")
    return True
