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
"""Standalone replica extension used by verl's normal LLMServerManager."""

import os

import ray
from rl_engine.integrations.orchestrators.verl.pipeline_rollout import RLKPipelineVLLMHttpServer

from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica


class RLKTrainerVLLMServer(RLKPipelineVLLMHttpServer):
    def __init__(self, **kwargs):
        reservations = kwargs.pop("workers")
        if kwargs["rollout_mode"] != RolloutMode.STANDALONE or len(reservations) != 2:
            raise ValueError("expected the two standalone Ray GPU reservation workers")
        # Match native MCore's original checkpoint ownership (layers 0-2 / 3).
        # These settings affect this dedicated server and its mp children only.
        os.environ.update(
            VLLM_PP_LAYER_PARTITION="3,1",
            VLLM_ROCM_USE_AITER="1",
            VLLM_ENABLE_V1_MULTIPROCESSING="1",
        )
        # These are resource holders, not model workers. The native vLLM mp
        # executor constructs the actual PP stages on their reserved devices.
        super().__init__(workers=[], **kwargs)

    def _preprocess_engine_kwargs(self, engine_kwargs):
        super()._preprocess_engine_kwargs(engine_kwargs)
        engine_kwargs["worker_cls"] = "verl.experimental.rl_kernel.rollout_worker.RLKVerlDSV4Worker"
        os.environ["RLK_VERL_MHC_BACKEND"] = self.config.engine_kwargs["vllm"].get("rlk_mhc_backend", "rocm_triton")
        engine_kwargs.pop("rlk_mhc_backend", None)

    async def rlk_load_policy(self, path, global_steps):
        if not self._submission_paused:
            raise RuntimeError("close admission and clear old caches before policy sync")
        acks = await self.collective_rpc("rlk_sync_weights", args=(path,))
        self.global_steps = global_steps
        return acks


class RLKTrainerReplica(vLLMReplica):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(RLKTrainerVLLMServer)
