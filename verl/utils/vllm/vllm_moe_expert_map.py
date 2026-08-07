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

"""Repair the expert-parallel routing maps a dummy rollout load destroys.

A rollout engine is brought up on dummy weights and then refitted, and vLLM's
dummy init walks ``state_dict()`` -- which includes buffers -- zeroing every
non-floating-point entry it finds. On ROCm that branch is unconditional, for
reproducibility across processes.

The expert-parallel routing tables ride along in that walk.
``RoutedExperts._expert_map`` and ``expert_mask`` are int32 buffers, so an EP
rollout comes up with the global-to-local expert mapping replaced by zeros: no
entry is left marked -1, so nothing is recognized as living on another rank and
every token is routed to local expert 0.

Refitting does not undo it. These maps are derived from the parallel layout, not
carried in any checkpoint, so nothing in the weight stream ever writes them back.

The damage is silent and shows up far downstream as a rollout that disagrees with
training, or, with the Triton MoE kernels, as NaN: their reduction skips slots
the matmul left unwritten by looking for -1, and a zeroed map has none, so the
sum picks up uninitialized memory.
"""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def restore_moe_expert_maps(model):
    """Rebuild the EP expert maps from the parallel layout, in place.

    In place so that addresses a CUDA graph captured stay valid. Safe to call on
    every refit: the maps are a pure function of the layout, so a rebuild is
    either a no-op or a repair. Models without expert parallelism have no maps
    and are skipped.
    """
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map

    repaired = 0
    for module in model.modules():
        if not isinstance(module, RoutedExperts):
            continue
        manager = getattr(module, "expert_map_manager", None)
        if manager is None or not module.moe_config.moe_parallel_config.use_ep:
            continue

        # Deliberately not ExpertMapManager._calculate_expert_maps: that folds the
        # fused shared experts into local_num_experts on every call, so replaying
        # it would inflate the count.
        _, expert_map, expert_mask = determine_expert_map(
            ep_size=manager.ep_size,
            ep_rank=manager.ep_rank,
            global_num_experts=manager.global_num_experts,
            expert_placement_strategy=manager.placement_strategy,
            num_fused_shared_experts=manager.num_fused_shared_experts,
            return_expert_mask=manager.rocm_aiter_enabled,
        )

        for name, rebuilt in (("_expert_map", expert_map), ("expert_mask", expert_mask)):
            live = getattr(module, name, None)
            if live is None or rebuilt is None:
                continue
            if live.shape != rebuilt.shape:
                raise RuntimeError(
                    f"Rebuilt {name} is {tuple(rebuilt.shape)} but the live buffer is "
                    f"{tuple(live.shape)}; the expert layout changed under the rollout engine."
                )
            live.copy_(rebuilt.to(device=live.device))
            repaired += 1

    if repaired:
        logger.info("Rebuilt %d expert-parallel routing buffers on the rollout model", repaired)
    return repaired
