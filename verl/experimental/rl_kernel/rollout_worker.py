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
"""DSV4 worker for normal verl generation, without parity-harness trace hooks."""

import os
from contextlib import ExitStack

import torch
from vllm.v1.worker.gpu_worker import Worker


class RLKVerlDSV4Worker(Worker):
    def load_model(self, **kwargs):
        super().load_model(**kwargs)
        from rl_engine.integrations.common.device_identity import hip_device_identity
        from rl_engine.integrations.engines.rollout.vllm.group_projection import bind_original_group_projection
        from rl_engine.integrations.engines.rollout.vllm.native_decode import bind_rollout_decode
        from rl_engine.integrations.engines.rollout.vllm.native_sparse import bind_rollout_sparse
        from rl_engine.integrations.engines.rollout.vllm.outputs import bind_output_head, selected_raw_logprobs
        from rl_engine.integrations.engines.rollout.vllm.router_projection import bind_rollout_router_projection
        from rl_engine.integrations.engines.rollout.vllm.shared_down import bind_shared_down
        from rl_engine.integrations.orchestrators.verl.mhc import bind_mhc_worker
        from vllm.distributed import get_pp_group

        wrapper = self.model_runner.get_model()
        model = wrapper.model
        self._rlk_pp_rank = get_pp_group().rank_in_group
        self._rlk_device_identity = hip_device_identity()
        self._rlk_worker = bind_mhc_worker(
            model,
            "rollout",
            backend=os.environ.get("RLK_VERL_MHC_BACKEND", "rocm_triton"),
            mode="split",
            version=0,
        )
        bind_shared_down(model, self._rlk_worker)
        for index in range(model.start_layer, model.end_layer):
            layer = model.layers[index]
            bind_rollout_router_projection(layer.ffn.gate)
            bind_rollout_sparse(layer.attn)
            bind_original_group_projection(layer.attn, self.model_config.model, index)
            bind_rollout_decode(layer.attn)
        original = model.forward

        def forward(*args, **kwargs):
            with self._rlk_worker.request(self._rlk_worker.version):
                return original(*args, **kwargs)

        model.forward = forward
        if get_pp_group().is_last_rank:
            bind_output_head(wrapper, self._rlk_worker)
            if self.model_runner.sampler.logprobs_mode != "raw_logprobs":
                raise ValueError("RLK verl rollout requires raw_logprobs")
            from vllm.v1.worker.gpu.sample import logprob

            # The vLLM mp executor owns this process. Match its selected-token
            # provider ABI; sampling and token ordering remain owned by vLLM.
            def raw(logits, token_ids):
                return selected_raw_logprobs(wrapper, logits, token_ids, weight_version=self._rlk_worker.version)

            logprob.compute_token_logprobs = raw

    def rlk_sync_weights(self, path):
        from rl_engine.integrations.engines.rollout.vllm.checkpoint_sync import (
            pipeline_stage_payload,
            readback_original_tensor,
            verify_loaded_update,
        )

        wrapper = self.model_runner.get_model()
        payload = pipeline_stage_payload(wrapper, torch.load(path, map_location="cuda", weights_only=True))
        worker = self._rlk_worker
        version = payload["weight_version"]
        if version < worker.version or (version == worker.version and version != 0):
            raise ValueError("rollout weight version must advance after initial verification")
        with ExitStack() as stack:
            if version == 0:
                # Both engines loaded the same original checkpoint. Verify it
                # without fabricating an optimizer update or a version advance.
                stack.enter_context(worker.request(0))
                loaded = {
                    readback_original_tensor(wrapper, name, tensor)[0] for name, tensor in payload["weights"].items()
                }
            else:
                stack.enter_context(worker.updating(version, torch.cuda.synchronize))
                loaded = wrapper.load_weights(payload["weights"].items())
            checked = verify_loaded_update(wrapper, payload, loaded)
        return dict(
            pipeline_stage=self._rlk_pp_rank,
            pid=os.getpid(),
            device_identity=self._rlk_device_identity,
            original_layer_ids=list(range(wrapper.model.start_layer, wrapper.model.end_layer)),
            weight_version=worker.version,
            parameter_load_completed=True,
            verified_original_tensors=checked,
        )
