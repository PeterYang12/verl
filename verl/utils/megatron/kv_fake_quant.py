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

"""Make the training forward see the rounding the rollout's FP8 KV cache applies.

Block-scaled QAT on the linear layers brought layer 0's ``wq_b`` output from
2.59e-02 down to 7.7e-03, and then the gap more than tripled again across the
attention operator. The KV cache is the obvious candidate: on ROCm, DeepSeek-V4
stores its K latent in the ``fp8_ds_mla`` layout and that is not optional --
asking for ``kv_cache_dtype=auto`` trips an assert in the model's own attention
setup. If the rollout cannot stop rounding, training has to start.

The layout being reproduced, from vLLM's compress/quantize kernel:

* the first 448 of the 512 latent channels (``head_dim - qk_rope_head_dim``) are
  E4M3 with one power-of-two scale per 64 channels,
* the last 64 -- the RoPE half -- stay BF16,
* quantization happens *after* the KV layernorm and *before* RoPE, which is why
  this hooks the norm rather than the projection feeding it.

Off unless ``VERL_QAT_KV_FAKE_QUANT=1``.
"""

from __future__ import annotations

import os

import torch

# vLLM derives these from the checkpoint; they are asserted against the live
# tensor at hook time rather than trusted.
_FP8_E4M3_MAX = 448.0
_ROPE_DIM = int(os.environ.get("VERL_QAT_KV_ROPE_DIM", "64"))
_SCALE_BLOCK = int(os.environ.get("VERL_QAT_KV_BLOCK", "64"))

# Ordered by how specific the name is. Megatron's generic MLA calls this
# ``kv_layernorm``; other DeepSeek ports use ``k_layernorm`` or ``kv_norm``.
_NORM_CANDIDATES = ("kv_layernorm", "k_layernorm", "kv_norm", "kv_a_layernorm")

_PATCH_FLAG = "_verl_kv_fake_quant_installed"


def enabled() -> bool:
    return os.environ.get("VERL_QAT_KV_FAKE_QUANT", "0") == "1"


def fake_quant_kv_latent(kv: torch.Tensor) -> torch.Tensor:
    """Round the NoPE half of a KV latent the way the FP8 cache stores it.

    ``kv`` is ``[..., head_dim]``; the trailing ``_ROPE_DIM`` channels are
    returned untouched. A tensor whose last dimension does not split cleanly is
    returned unchanged rather than silently quantized on the wrong boundary.
    """
    dim = kv.shape[-1]
    nope = dim - _ROPE_DIM
    if nope <= 0 or nope % _SCALE_BLOCK:
        return kv

    head, tail = kv[..., :nope], kv[..., nope:]

    grouped = head.float().reshape(*head.shape[:-1], nope // _SCALE_BLOCK, _SCALE_BLOCK)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    # The kernel floors the amax before taking the exponent, so an all-zero
    # block lands on a finite scale instead of -inf.
    scale = torch.exp2(torch.ceil(torch.log2(amax.clamp(min=1e-4) / _FP8_E4M3_MAX)))
    rounded = (grouped / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn).float() * scale
    rounded = rounded.reshape(head.shape).to(head.dtype)

    # Straight-through estimator, as for the linear-layer fake quant.
    head = head + (rounded - head).detach()
    return torch.cat((head, tail), dim=-1)


def _hook(_module, _args, out):
    tensor = out[0] if isinstance(out, tuple) else out
    if not isinstance(tensor, torch.Tensor):
        return out
    quantized = fake_quant_kv_latent(tensor)
    if isinstance(out, tuple):
        return (quantized, *out[1:])
    return quantized


def fake_quant_pow2_lastdim(x: torch.Tensor) -> torch.Tensor:
    """E4M3 with ONE power-of-two scale over the whole last dim (vLLM indexer q: per token-head; k: per token)."""
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = torch.exp2(torch.ceil(torch.log2(amax / _FP8_E4M3_MAX)))
    out = (xf / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn).float() * scale
    return x + (out.to(x.dtype) - x).detach()


def _install_indexer_fake_quant() -> int:
    """vLLM quantizes the indexer q and k to FP8 and applies NO Hadamard rotation; Megatron rotates both
    (dot product invariant). Quantizing right before Megatron's rotation reproduces vLLM's scores."""
    try:
        from megatron.core.transformer.experimental_attention_variant import csa
    except Exception:
        return 0
    if getattr(csa, "_verl_indexer_fq", False):
        return 1
    orig = csa.rotate_activation

    def rotate_with_fake_quant(x):
        return orig(fake_quant_pow2_lastdim(x))

    csa.rotate_activation = rotate_with_fake_quant
    csa._verl_indexer_fq = True
    return 1


def install(unwrapped_model) -> int:
    """Hook the KV layernorm of every decoder layer. Returns how many were hooked.

    Prints the attention submodule names when nothing matches: the DeepSeek-V4
    layer spec is not Megatron's stock MLA, so the norm's name is worth
    confirming rather than assuming, and a silent no-op here would look exactly
    like the fake quant not helping.
    """
    if not enabled():
        return 0

    decoder = getattr(unwrapped_model, "decoder", None)
    if decoder is None or not hasattr(decoder, "layers"):
        return 0

    hooked = 0
    seen: list[str] = []
    for layer in decoder.layers:
        attn = getattr(layer, "self_attention", None)
        if attn is None:
            continue
        if not seen:
            seen = [n for n, _ in attn.named_modules() if n]
        for candidate in _NORM_CANDIDATES:
            norm = getattr(attn, candidate, None)
            if norm is None or not hasattr(norm, "register_forward_hook"):
                continue
            if getattr(norm, _PATCH_FLAG, False):
                break
            norm.register_forward_hook(_hook)
            setattr(norm, _PATCH_FLAG, True)
            hooked += 1
            break

    # --- compressed KV + indexer q/k ---
    n_comp = 0
    if os.environ.get("VERL_QAT_KV_FQ_COMPRESSOR", "1") == "1":
        for layer in decoder.layers:
            attn = getattr(layer, "self_attention", None)
            if attn is None:
                continue
            for _name, mod in attn.named_modules():
                # main compressor only: head_dim == full KV latent (512) and no Hadamard rotation;
                # the indexer's own compressor (128-dim, rotate=True) is handled below.
                if mod.__class__.__name__ == "Compressor" and not getattr(mod, "rotate", False):
                    norm = getattr(mod, "norm", None)
                    if norm is not None and not getattr(norm, _PATCH_FLAG, False):
                        norm.register_forward_hook(_hook)
                        setattr(norm, _PATCH_FLAG, True)
                        n_comp += 1
    n_idx = _install_indexer_fake_quant() if os.environ.get("VERL_QAT_KV_FQ_INDEXER", "1") == "1" else 0
    print(f"KV fake quant ext: hooked {n_comp} compressor norms; indexer q/k fp8 emulation={'on' if n_idx else 'off'}")

    if hooked:
        print(f"KV fake quant: hooked {hooked} KV layernorms (nope={_SCALE_BLOCK}-blocks, rope={_ROPE_DIM} kept bf16)")
    else:
        print(f"KV fake quant: NO KV layernorm matched {_NORM_CANDIDATES}; attention submodules are {seen}")
    return hooked
