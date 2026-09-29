# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""ModelOpt NVFP4 quantization config and application for Megatron QAT."""

import copy
import os

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg

from verl.utils.modelopt.fp8_group_backend import BACKEND_NAME as FP8_GROUP_BACKEND
from verl.utils.modelopt.fp8_group_backend import BLOCK_BACKEND_NAME as FP8_BLOCK_BACKEND
from verl.utils.modelopt.fp8_group_backend import register as _register_fp8_group_backend
from verl.workers.config.engine import QAT_FP8_BLOCKWISE_ACT_MODE, QAT_FP8_BLOCKWISE_MODE

_NVFP4_W4A16_QUANTIZER_CFG = {
    "*weight_quantizer": {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
        "enable": True,
    },
    "*input_quantizer": {"enable": False},
}

FP8_BLOCKWISE_MODE = QAT_FP8_BLOCKWISE_MODE
FP8_BLOCKWISE_ACT_MODE = QAT_FP8_BLOCKWISE_ACT_MODE

# Matches how an FP8 inference engine computes a block-scaled W8A8 GEMM for a
# DeepSeek-style checkpoint: E4M3 values, scales derived from each block's amax
# and left as plain float32. All the tensor dimensions involved are multiples of
# 128, so these block boundaries line up with the ones the weight sync uses even
# though it tiles the HuggingFace layout rather than the Megatron one.
_FP8_WEIGHT_QUANTIZER = {
    "num_bits": (4, 3),
    "block_sizes": {-1: 128, -2: 128},
    "axis": None,
    "enable": True,
    # ModelOpt's own 2D blocking computes amax/448, which is a different grid
    # from the one the rollout holds: the checkpoint's scales are powers of two
    # and the weight sync keeps them there.
    "backend": FP8_BLOCK_BACKEND,
    "backend_extra_args": {"block_size": 128, "e8m0_scale": True},
}
_FP8_INPUT_QUANTIZER = {
    "num_bits": (4, 3),
    "axis": None,
    "type": "dynamic",
    "enable": True,
    "backend": FP8_GROUP_BACKEND,
    "backend_extra_args": {"group_size": 128, "e8m0_scale": False},
}

_FP8_BLOCKWISE_QUANTIZER_CFG = {
    "*weight_quantizer": _FP8_WEIGHT_QUANTIZER,
    "*input_quantizer": _FP8_INPUT_QUANTIZER,
}

_FP8_BLOCKWISE_ACT_QUANTIZER_CFG = {
    "*weight_quantizer": {"enable": False},
    "*input_quantizer": _FP8_INPUT_QUANTIZER,
}

_QUANTIZER_CFGS = {
    "w4a16": _NVFP4_W4A16_QUANTIZER_CFG,
    FP8_BLOCKWISE_MODE: _FP8_BLOCKWISE_QUANTIZER_CFG,
    FP8_BLOCKWISE_ACT_MODE: _FP8_BLOCKWISE_ACT_QUANTIZER_CFG,
}

# DeepSeek-V4 keeps these in BF16 in the checkpoint (they ship no ``.scale``
# sibling), so quantizing them in training would simulate something the rollout
# engine never does. The names ModelOpt already disables by default -- router,
# lm_head/output_layer, embeddings, MTP -- are not repeated here.
# Of the indexer only ``weights_proj`` is left out: ``indexer.linear_wq_b`` has a
# ``.scale`` in the checkpoint and the rollout engine runs it in FP8.
_FP8_BLOCKWISE_EXTRA_IGNORE = ["*compressor*", "*weights_proj*"]


def _env_extra_ignore() -> list[str]:
    """Extra quantizer-name globs from ``VERL_QAT_EXTRA_IGNORE``, comma separated.

    For experiments on which modules to simulate, without editing this file.
    Leave it unset for normal runs. ``*experts*`` in particular also matches
    ``shared_experts`` and takes every expert layer out of QAT; with that
    setting QAT closed only about 10% of the train/rollout gap.
    """
    raw = os.environ.get("VERL_QAT_EXTRA_IGNORE", "")
    return [p.strip() for p in raw.split(",") if p.strip()]


def _ignore_patterns_to_quant_cfg(ignore_patterns: list[str]) -> list[dict]:
    cfg = []
    mapping = {
        "lm_head": "*output_layer*",
        "*mlp.gate": "*router*",
        "*self_attn*": "*self_attention*",
    }
    for pattern in ignore_patterns:
        key = pattern
        if key in mapping:
            key = mapping[key]
        cfg.append({"quantizer_name": key, "enable": False})
    return cfg


def uses_native_weight_sync(qat_mode: str) -> bool:
    """Whether the plain weight-sync path already emits the format the mode trains in.

    FP8 block-scaled QAT trains against the same layout an FP8 checkpoint is
    stored in, and Megatron-Bridge already requantizes exported weights to it,
    so the NVFP4 weight exporter must not run and the rollout engine's own
    quantization config must not be overridden.
    """
    return qat_mode in (FP8_BLOCKWISE_MODE, FP8_BLOCKWISE_ACT_MODE)


def build_quantize_config(
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> dict:
    """Build a complete ModelOpt quantization config for ``mtq.quantize``."""
    if qat_mode not in _QUANTIZER_CFGS:
        raise ValueError(f"Unsupported QAT mode {qat_mode!r}; expected one of {sorted(_QUANTIZER_CFGS)}")

    if ignore_patterns is None:
        ignore_patterns = []
    if uses_native_weight_sync(qat_mode):
        ignore_patterns = list(ignore_patterns) + _FP8_BLOCKWISE_EXTRA_IGNORE + _env_extra_ignore()

    ignore_cfg = _ignore_patterns_to_quant_cfg(ignore_patterns)

    quant_cfg = mtq.normalize_quant_cfg_list(_QUANTIZER_CFGS[qat_mode])
    disabled_cfg = copy.deepcopy(_default_disabled_quantizer_cfg)
    if isinstance(disabled_cfg, dict):
        disabled_cfg = mtq.normalize_quant_cfg_list(disabled_cfg)
    quant_cfg.extend(disabled_cfg)
    quant_cfg.extend(ignore_cfg)

    # The FP8 backends derive every scale from the tensor in front of them, and
    # ``TensorQuantizer._fake_quantize`` dispatches to a registered backend before
    # it reads ``_amax``, so calibration here produces state nothing consumes.
    # It also hangs with pipeline parallelism: ``max_calibrate`` all-reduces each
    # quantizer's amax, and a module whose ``parallel_state`` Megatron never
    # populated falls back to ModelOpt's default, which is the whole world rather
    # than the DP group. Stages holding different numbers of layers then issue
    # different numbers of collectives and the job deadlocks in ``mtq.quantize``.
    algorithm = None if uses_native_weight_sync(qat_mode) else "max"
    return {"quant_cfg": quant_cfg, "algorithm": algorithm}


def apply_qat(
    model: nn.Module,
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> nn.Module:
    """Apply Quantization-Aware Training to a Megatron model."""
    if uses_native_weight_sync(qat_mode):
        _register_fp8_group_backend()
    config = build_quantize_config(qat_mode, ignore_patterns)
    mtq.quantize(model, config)
    _report_enabled_quantizers(model, qat_mode)
    return model


def _report_enabled_quantizers(model: nn.Module, qat_mode: str) -> None:
    """Print which modules ended up quantized, and which weight-carrying ones did not.

    Whether a module matched decides between simulating the rollout engine's
    arithmetic and silently training in full precision, and it depends on
    ModelOpt recognising each module's class. A first attempt at this mode
    quantized 70 modules out of a model whose MoE experts alone are far more
    than that and made the train/rollout gap worse, so the misses matter at
    least as much as the hits. Printed rather than logged so it survives the
    worker log level.
    """
    weights, inputs, missed = [], [], []
    for name, module in model.named_modules():
        hit = False
        for kind, bucket in (("weight_quantizer", weights), ("input_quantizer", inputs)):
            quantizer = getattr(module, kind, None)
            if quantizer is not None and getattr(quantizer, "is_enabled", False):
                bucket.append(name)
                hit = True
        # Leaf modules owning a 2D weight are the ones that become a GEMM the
        # rollout runs in FP8; anything else has no operand to quantize.
        if not hit and not any(module.children()):
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.dim() >= 2:
                missed.append(f"{name}[{type(module).__name__}]")

    print(f"QAT[{qat_mode}]: {len(weights)} weight quantizers, {len(inputs)} input quantizers enabled")
    if missed:
        # One layer's worth is enough to recognise a whole class of misses, and
        # the full list on a 60-layer model is thousands of lines.
        sample = sorted({m.split(".", 3)[-1] for m in missed})[:12]
        print(f"QAT[{qat_mode}]: {len(missed)} weight-carrying modules NOT quantized, e.g. {sample}")
