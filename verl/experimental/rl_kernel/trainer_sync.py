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
"""Quiesced PP2 weight sync for the ordinary synchronous verl trainer."""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from .trainer_contract import validate_stage_acks


class RLKPolicySync:
    """Keep the same rollout workers alive and acknowledge complete readback.

    A failed export/load/readback leaves admission closed. There is deliberately
    no best-effort fallback to stale weights or the generic PP1 IPC path.
    """

    def __init__(self, actor_wg, servers, sync_dir):
        self.actor_wg = actor_wg
        self.servers = list(servers)
        self.root = Path(sync_dir)
        if not self.root.is_absolute() or len(self.servers) != 1:
            raise ValueError("one PP2 replica and an absolute shared sync directory are required")
        self.root.mkdir(parents=True, exist_ok=True)
        self.version = None
        self.pids = None
        self.manifest = None

    def sleep_replicas(self):
        # Actor and rollout have disjoint GPUs. Sleeping would discard the
        # continuous worker/cache state this adapter is required to verify.
        return None

    def close(self):
        return None

    def update_weights(self, global_steps=None):
        return asyncio.run(self.update(global_steps))

    async def update(self, global_steps=None):
        await asyncio.gather(*[server.abort_all_requests.remote(reset_prefix_cache=True) for server in self.servers])
        path = self.root / f"policy-{uuid4().hex}.pt"
        exports = self.actor_wg.rlk_export_policy(str(path))
        if not exports or len(exports) != 2:
            raise RuntimeError("expected both actor PP stages to export")
        first = exports[0]
        if any(item != first for item in exports):
            raise RuntimeError("actor PP ranks disagree on exported state or version")
        version, manifest = first["weight_version"], first["manifest"]
        if self.version is not None and version <= self.version:
            raise RuntimeError("a trainer update did not advance the optimizer weight version")
        acks = await self.servers[0].rlk_load_policy.remote(str(path), global_steps)
        pids = validate_stage_acks(acks, manifest, version, self.pids)
        record = dict(global_steps=global_steps, weight_version=version, stages=acks)
        (self.root / f"sync-v{version}.json").write_text(json.dumps(record, indent=2))
        # Publish the new version and reopen only after every stage passed.
        self.version, self.pids, self.manifest = version, pids, manifest
        await asyncio.gather(*[server.resume_generation.remote() for server in self.servers])
        path.unlink()
        return {"rlk/weight_version": version, "rlk/synced_tensors": len(manifest)}
