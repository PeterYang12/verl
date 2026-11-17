# Copyright 2026 Individual Contributor
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
# ruff: noqa: E402 -- native adapters must be installed before importing verl.
"""Native verl loss ABI backed by RLK's DSV4 operator boundaries."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from rl_engine.integrations.orchestrators.verl.native_engine import (
    RLKNativeMcoreEngineConfig,
    RLKNativeMegatronEngine,
)

from verl.workers.engine.base import EngineRegistry

from .trainer_contract import logical_cases


def bind_actor_operators(model):
    from rl_engine.integrations.engines.train.megatron.native_combine import bind_native_combine
    from rl_engine.integrations.engines.train.megatron.native_compressor import (
        bind_native_compressor,
    )
    from rl_engine.integrations.engines.train.megatron.native_compressor_projection import (
        bind_compressor_projection,
    )
    from rl_engine.integrations.engines.train.megatron.native_hash_route import (
        bind_native_hash_route,
    )
    from rl_engine.integrations.engines.train.megatron.native_inverse_rope import (
        bind_native_inverse_rope,
    )
    from rl_engine.integrations.engines.train.megatron.native_learned_route import (
        bind_native_learned_route,
    )
    from rl_engine.integrations.engines.train.megatron.native_router_projection import (
        bind_native_router_projection,
    )
    from rl_engine.integrations.engines.train.megatron.native_sparse import bind_native_sparse
    from rl_engine.integrations.engines.train.megatron.native_w4a16 import bind_native_w4a16

    for layer in model.decoder.layers[::2]:
        attention = layer.inner_layer.self_attention
        bind_native_sparse(attention.core_attention)
        bind_native_inverse_rope(attention)
        if getattr(attention.core_attention, "compressor", None) is not None:
            bind_compressor_projection(attention.core_attention.compressor)
            bind_native_compressor(attention.core_attention.compressor, attention._rlk_qkv_preprocess)
        if getattr(attention.core_attention, "indexer", None) is not None:
            bind_compressor_projection(attention.core_attention.indexer.compressor)
    for layer in model.decoder.layers[1::2]:
        mlp = layer.inner_layer.mlp
        bind_native_router_projection(mlp.router)
        binder = bind_native_hash_route if mlp.router.is_hash_layer else bind_native_learned_route
        binder(mlp.router)
        bind_native_w4a16(mlp)
        bind_native_combine(mlp)


@dataclass
class RLKVerlMcoreEngineConfig(RLKNativeMcoreEngineConfig):
    strategy: str = "rlk_verl_megatron"
    rlk_allow_right_padding: bool = True
    rlk_evidence_dir: str = ""

    def __post_init__(self):
        if self.strategy != "rlk_verl_megatron":
            raise ValueError("expected the explicit rlk_verl_megatron registration")
        object.__setattr__(self, "strategy", "rlk_megatron")
        try:
            super().__post_init__()
        finally:
            object.__setattr__(self, "strategy", "rlk_verl_megatron")


@EngineRegistry.register("language_model", "rlk_verl_megatron")
class RLKVerlMegatronEngine(RLKNativeMegatronEngine):
    def initialize(self):
        super().initialize()
        for model in self._rlk_models:
            bind_actor_operators(model)
        if self.engine_config.rlk_evidence_dir:
            import os

            import torch.distributed as dist
            from rl_engine.integrations.common.device_identity import hip_device_identity

            root = Path(self.engine_config.rlk_evidence_dir)
            root.mkdir(parents=True, exist_ok=True)
            (root / f"actor-rank-{dist.get_rank()}-identity.json").write_text(
                json.dumps(
                    dict(
                        rank=dist.get_rank(),
                        pid=os.getpid(),
                        device_identity=hip_device_identity(),
                        original_layer_ids=list(self._rlk_models[0]._rlk_checkpoint_stage.layer_ids),
                    ),
                    indent=2,
                )
            )

    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
        from verl.utils import tensordict_utils as tu

        batch = next(batch_iter)
        self.rlk_logical_cases = logical_cases(batch["input_ids"])
        # Native CSA requires a window-sized workspace even for short prompts.
        # Keep real tokens/masks unchanged and declare only BSHD right padding.
        forced = tu.get_non_tensor_data(batch, key="forced_max_seqlen", default=0) or 0
        actual_max = max(len(row) for row in batch["input_ids"].unbind())
        tu.assign_non_tensor(
            batch, forced_max_seqlen=max(int(forced), actual_max, int(getattr(self.tf_config, "csa_window_size", 1)))
        )
        return super().forward_step(iter((batch,)), model, logits_processor_func, postprocess_micro_batch_func)

    def _lm_head_logits_processor(self, logits, label, temperature, **kwargs):
        # Padded temperature entries are zero; only nonpadding entries may be 1.
        if not torch.all((temperature == 1) | (temperature == 0)):
            raise ValueError("RLK verl raw-score ABI requires temperature=1")
        if any(
            kwargs.get(key, False)
            for key in (
                "distillation_use_topk",
                "distillation_only",
                "score_centering",
                "calculate_sum_pi_squared",
            )
        ):
            raise ValueError("unsupported transformed policy score in RLK verl engine")
        result = super()._lm_head_logits_processor(logits, label, temperature, **kwargs)
        # Preserve verl's PPO/GRPO loss and its native next-token shift/masks.
        # Only the selected-score provider changes, analogous to the Vime ABI.
        result["log_probs"] = result["raw_log_probs"]
        return result

    def optimizer_step(self):
        import torch.distributed as dist
        from rl_engine.integrations.engines.rollout.vllm.checkpoint_sync import digest

        parameters = {
            name: parameter
            for model in self._rlk_models
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        before = {name: digest(parameter) for name, parameter in parameters.items()}
        gradients = []
        for parameter in parameters.values():
            gradient = getattr(parameter, "main_grad", None)
            if gradient is None:
                gradient = parameter.grad
            if gradient is not None:
                gradients.append(gradient)
        finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        if not finite:
            raise RuntimeError("nonfinite actor gradients before the native optimizer step")
        nonzero = any(bool(torch.count_nonzero(gradient)) for gradient in gradients)
        result = super().optimizer_step()
        changed = [name for name, parameter in parameters.items() if digest(parameter) != before[name]]
        if self.engine_config.rlk_evidence_dir:
            root = Path(self.engine_config.rlk_evidence_dir)
            root.mkdir(parents=True, exist_ok=True)
            (root / f"actor-rank-{dist.get_rank()}-v{self.rlk_weight_version}.json").write_text(
                json.dumps(
                    dict(
                        weight_version=self.rlk_weight_version,
                        rank=dist.get_rank(),
                        trainable_parameters=len(parameters),
                        gradient_tensors=len(gradients),
                        gradients_finite=finite,
                        gradients_nonzero=nonzero,
                        changed_parameters=changed,
                    ),
                    indent=2,
                )
            )
        return result
