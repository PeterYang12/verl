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
# ruff: noqa: E402 -- install the version-scoped native import adapters first.
"""Opt-in synchronous verl trainer; native reward, advantage and policy loss."""

import asyncio
import json
import os
from pathlib import Path

import ray
import torch
import torch.distributed as dist

from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.workers.engine_workers import ActorRolloutRefWorker

from . import trainer_engine  # noqa: F401
from .trainer_contract import strict_response_parity, validate_trainer_config
from .trainer_rollout import RLKTrainerReplica
from .trainer_sync import RLKPolicySync


class RLKActorWorker(ActorRolloutRefWorker):
    """Inherit native TrainingWorker and ppo_loss; add checked export only."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        from verl.workers.utils.losses import ppo_loss

        if getattr(self.loss_fn, "func", None) is not ppo_loss:
            raise RuntimeError("RLK trainer requires verl's native PPO/GRPO policy loss")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rlk_export_policy(self, path):
        from rl_engine.integrations.engines.rollout.vllm.checkpoint_sync import digest

        engine = self.actor.engine
        model = engine._rlk_models[0]
        covered = {
            id(parameter) for parameters in model._rlk_checkpoint_export_parameters.values() for parameter in parameters
        }
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in covered
        ]
        if missing:
            raise RuntimeError(f"trainable parameters missing from sync export: {missing}")
        parameters, _ = engine.get_per_tensor_param(base_sync_done=True)
        manifest, weights = {}, {}
        for name, tensor in parameters:
            cpu = tensor.detach().cpu().clone()
            if not torch.isfinite(cpu).all():
                raise RuntimeError(f"nonfinite actor state: {name}")
            manifest[name] = dict(shape=list(cpu.shape), dtype=str(cpu.dtype), sha256=digest(cpu))
            if dist.get_rank() == 0:
                weights[name] = cpu
        payload = dict(weight_version=engine.rlk_weight_version, manifest=manifest)
        if dist.get_rank() == 0:
            destination = Path(path)
            pending = destination.with_suffix(".pending")
            torch.save(dict(payload, weights=weights), pending)
            os.replace(pending, destination)
        # ONE_TO_ALL waits for both exports, including the writer, to return.
        return payload

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def rlk_score_sequences(self, cases, weight_version):
        """Score new rollout tokens with the same live native actor after sync."""
        from tensordict import TensorDict

        from verl.utils import tensordict_utils as tu

        if self.actor.engine.rlk_weight_version != weight_version:
            raise RuntimeError("actor audit requested a stale weight version")
        tokens, masks = [], []
        for case in cases:
            prompt, response = case["prompt_ids"], case["response_ids"]
            if not prompt or not response:
                raise ValueError("post-update audit requires nonempty prompt and response")
            tokens.append(torch.tensor(prompt + response, dtype=torch.int64))
            masks.append(torch.tensor([0] * len(prompt) + [1] * len(response)))
        data = TensorDict(
            {
                "input_ids": torch.nested.as_nested_tensor(tokens, layout=torch.jagged),
                "loss_mask": torch.nested.as_nested_tensor(masks, layout=torch.jagged),
                "temperature": torch.ones(len(cases)),
            },
            batch_size=[len(cases)],
        )
        tu.assign_non_tensor(
            data,
            global_token_num=None,
            compute_loss=False,
            disable_auto_offload=True,
            micro_batch_size_per_gpu=1,
            calculate_entropy=False,
        )
        output = self.actor.infer_batch(data)
        if output is None:
            return None
        scores = [
            row[len(case["prompt_ids"]) - 1 : len(case["prompt_ids"]) + len(case["response_ids"]) - 1]
            for row, case in zip(output["log_probs"].unbind(), cases, strict=True)
        ]
        return dict(weight_version=weight_version, scores=scores)


class RLKVerlTrainer(PPOTrainerSync):
    """A narrow PP2/DP1 GRPO recipe using the ordinary upstream fit loop."""

    _enable_hybrid_replicas = False

    def __init__(self, config):
        validate_trainer_config(config)
        super().__init__(config)

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        self.role_worker_mapping[Role.Actor] = ray.remote(RLKActorWorker)

    def _setup(self):
        super()._setup()
        self.checkpoint_manager.close()
        manager = self.llm_server_manager
        manager.rollout_replica_class = RLKTrainerReplica
        asyncio.run(manager._initialize_llm_servers())
        # Keep the router created by upstream setup. Do not create an orphaned
        # second router or alter the process-global vLLM replica registry.
        ray.get(
            manager.global_load_balancer.add_servers.remote(
                dict(zip(manager.server_addresses, manager.server_handles, strict=True))
            )
        )
        self.checkpoint_manager = RLKPolicySync(self.actor_rollout_wg, manager.server_handles, self.config.rlk.sync_dir)

    def _compute_old_log_prob(self, batch, metrics):
        import transfer_queue as tq

        batch = super()._compute_old_log_prob(batch, metrics)
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["old_log_probs", "rollout_log_probs", "response_mask"],
        ).to_padded_tensor()
        record = strict_response_parity(data["old_log_probs"], data["rollout_log_probs"], data["response_mask"])
        record.update(global_steps=self.global_steps, weight_version=self.checkpoint_manager.version)
        root = Path(self.config.rlk.sync_dir)
        with (root / "parity.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        metrics.update({f"rlk/{key}": record[key] for key in ("compared_tokens", "mismatch_count")})
        return batch

    def _compute_advantage(self, batch, metrics):
        import transfer_queue as tq

        batch = super()._compute_advantage(batch, metrics)
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["advantages", "returns", "rm_scores", "response_mask"],
        ).to_padded_tensor()
        summaries = {}
        for name in ("advantages", "returns", "rm_scores"):
            tensor = data[name]
            if not tensor.numel() or not torch.isfinite(tensor).all():
                raise RuntimeError(f"empty/nonfinite native post-training field: {name}")
            summaries[name] = dict(
                finite=True,
                nonzero=int(torch.count_nonzero(tensor)),
                minimum=float(tensor.min()),
                maximum=float(tensor.max()),
            )
        (Path(self.config.rlk.sync_dir) / f"training-batch-step{self.global_steps}.json").write_text(
            json.dumps(
                dict(
                    global_steps=self.global_steps,
                    weight_version=self.checkpoint_manager.version,
                    algorithm="grpo",
                    native_policy_loss="verl.workers.utils.losses.ppo_loss",
                    summaries=summaries,
                ),
                indent=2,
            )
        )
        return batch

    def on_step_end(self):
        # Native synchronous trainer performs the actual live weight sync.
        super().on_step_end()
        # Verify v1/v2/v3 immediately after each update, including the final
        # update for which there is no following PPO batch to trigger old-logp.
        prompts = [self.tokenizer.encode(prompt, add_special_tokens=True) for prompt in self.config.rlk.audit_prompts]

        async def generate():
            from uuid import uuid4

            server = self.llm_server_manager.server_handles[0]
            return await asyncio.gather(
                *[
                    server.generate.remote(
                        request_id=f"rlk-audit-{self.global_steps}-{uuid4().hex}",
                        prompt_ids=prompt,
                        sampling_params=dict(
                            temperature=1.0,
                            top_p=1.0,
                            top_k=-1,
                            logprobs=True,
                            max_tokens=int(self.config.rlk.audit_response_length),
                        ),
                    )
                    for prompt in prompts
                ]
            )

        outputs = asyncio.run(generate())
        cases = [
            dict(prompt_ids=prompt, response_ids=output.token_ids)
            for prompt, output in zip(prompts, outputs, strict=True)
        ]
        version = self.checkpoint_manager.version
        replies = self.actor_rollout_wg.rlk_score_sequences(cases, version)
        collected = [reply for reply in replies if reply is not None]
        if len(collected) != 1 or collected[0]["weight_version"] != version:
            raise RuntimeError("expected exactly one final actor PP output at the synced version")
        records = []
        for case, output, actor in zip(cases, outputs, collected[0]["scores"], strict=True):
            if output.log_probs is None or output.extra_fields.get("global_steps") != self.global_steps:
                raise RuntimeError("post-update rollout lacks raw scores or current policy metadata")
            rollout = torch.tensor(output.log_probs, dtype=torch.float32)
            checked = strict_response_parity(actor, rollout, torch.ones_like(actor, dtype=torch.bool))
            records.append(dict(case, **checked))
        (Path(self.config.rlk.sync_dir) / f"post-update-v{version}.json").write_text(
            json.dumps(dict(global_steps=self.global_steps, weight_version=version, cases=records), indent=2)
        )
