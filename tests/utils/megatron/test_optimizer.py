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
"""Tests for the precision-aware dispatch in ``init_megatron_optim_config``.

These tests stub out ``megatron.core.optimizer.OptimizerConfig`` so they can
run on CPU without TransformerEngine — the goal is to verify which kwargs
verl assembles for each precision mode, not Megatron's downstream validation.

The precision-aware optimizer is opt-in: the bf16 branch keeps the fp32
optimizer state unless ``use_precision_aware_optimizer`` is set on the config,
at which point the moment / grad dtypes follow the configured fields.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from verl.utils.megatron import optimizer as opt_mod
from verl.utils.megatron.optimizer import init_megatron_optim_config


def _base_optim_config(**overrides):
    cfg = {
        "optimizer": "adam",
        "lr": 1e-3,
        "min_lr": 0.0,
        "clip_grad": 1.0,
        "weight_decay": 0.01,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _precision_aware_optim_config(**overrides):
    """bf16 optimizer state explicitly opted in via the new config flags."""
    fields = {
        "use_precision_aware_optimizer": True,
        "main_grads_dtype": "bf16",
        "exp_avg_dtype": "bf16",
        "exp_avg_sq_dtype": "bf16",
    }
    fields.update(overrides)
    return _base_optim_config(**fields)


@pytest.fixture
def captured_args(monkeypatch):
    """Replace ``OptimizerConfig`` with a recorder so we can inspect kwargs."""
    captured: dict = {}

    def _fake(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return MagicMock(name="OptimizerConfig", **kwargs)

    monkeypatch.setattr(opt_mod, "OptimizerConfig", _fake)
    return captured


def test_bf16_branch_defaults_to_fp32_optimizer_state(captured_args):
    """Opt-out default: bf16 params, but the precision-aware optimizer stays off."""
    init_megatron_optim_config(_base_optim_config(), fp16=False, bf16=True)

    assert captured_args["bf16"] is True
    assert captured_args["params_dtype"] is torch.bfloat16
    # No precision-aware optimizer and no sub-fp32 moment / grad dtypes by default.
    assert "use_precision_aware_optimizer" not in captured_args
    assert "main_grads_dtype" not in captured_args
    assert "exp_avg_dtype" not in captured_args
    assert "exp_avg_sq_dtype" not in captured_args


def test_bf16_branch_opt_in_enables_precision_aware_with_bf16_state(captured_args):
    init_megatron_optim_config(_precision_aware_optim_config(), fp16=False, bf16=True)

    assert captured_args["bf16"] is True
    assert captured_args["params_dtype"] is torch.bfloat16
    assert captured_args["use_precision_aware_optimizer"] is True
    assert captured_args["main_grads_dtype"] is torch.bfloat16
    assert captured_args["exp_avg_dtype"] is torch.bfloat16
    assert captured_args["exp_avg_sq_dtype"] is torch.bfloat16
    # Master params dtype intentionally left at Megatron default (fp32) —
    # TE FusedAdam rejects bf16 master at init.
    assert "main_params_dtype" not in captured_args


def test_bf16_opt_in_respects_per_field_dtypes(captured_args):
    """Per-flag control: opting in but pinning a moment to fp32 is honored."""
    cfg = _precision_aware_optim_config(exp_avg_sq_dtype="fp32")
    init_megatron_optim_config(cfg, fp16=False, bf16=True)

    assert captured_args["use_precision_aware_optimizer"] is True
    assert captured_args["main_grads_dtype"] is torch.bfloat16
    assert captured_args["exp_avg_dtype"] is torch.bfloat16
    assert captured_args["exp_avg_sq_dtype"] is torch.float32


def test_fp16_branch_uses_precision_aware_but_keeps_fp32_optimizer_state(captured_args):
    init_megatron_optim_config(_base_optim_config(), fp16=True, bf16=False)

    assert captured_args["fp16"] is True
    assert captured_args["bf16"] is False
    assert captured_args["params_dtype"] is torch.float16
    assert captured_args["use_precision_aware_optimizer"] is True
    assert captured_args["initial_loss_scale"] == 32768
    assert captured_args["min_loss_scale"] == 1
    assert captured_args["store_param_remainders"] is False
    # Adam moment / grad dtypes left at Megatron's fp32 default in fp16 mode.
    assert "main_grads_dtype" not in captured_args
    assert "exp_avg_dtype" not in captured_args
    assert "exp_avg_sq_dtype" not in captured_args


def test_fp32_branch_disables_precision_aware_optimizer(captured_args):
    init_megatron_optim_config(_base_optim_config(), fp16=False, bf16=False)

    assert captured_args["fp16"] is False
    assert captured_args["bf16"] is False
    assert captured_args["params_dtype"] is torch.float32
    # Precision-aware optimizer must stay off — Megatron asserts the dtype
    # fields equal fp32 when it's disabled.
    assert "use_precision_aware_optimizer" not in captured_args
    assert "main_grads_dtype" not in captured_args
    assert "exp_avg_dtype" not in captured_args
    assert "exp_avg_sq_dtype" not in captured_args


def test_default_kwargs_dispatch_to_bf16_branch(captured_args):
    """Backward compatibility: callers that omit ``bf16`` get the bf16 path (fp32 state)."""
    init_megatron_optim_config(_base_optim_config())

    assert captured_args["bf16"] is True
    assert captured_args["params_dtype"] is torch.bfloat16
    # Opt-in default keeps the precision-aware optimizer off.
    assert "use_precision_aware_optimizer" not in captured_args


def test_fp16_wins_over_bf16_when_both_true(captured_args):
    init_megatron_optim_config(_precision_aware_optim_config(), fp16=True, bf16=True)

    assert captured_args["fp16"] is True
    assert captured_args["params_dtype"] is torch.float16
    # bf16-branch-only fields must not appear when fp16 is selected, even when
    # the precision-aware config flags are present.
    assert "main_grads_dtype" not in captured_args
    assert "exp_avg_dtype" not in captured_args


def test_use_distributed_optimizer_passes_through(captured_args):
    init_megatron_optim_config(_base_optim_config(), use_distributed_optimizer=False)
    assert captured_args["use_distributed_optimizer"] is False

    init_megatron_optim_config(_base_optim_config(), use_distributed_optimizer=True)
    assert captured_args["use_distributed_optimizer"] is True


def test_basic_optim_config_fields_pass_through(captured_args):
    cfg = _base_optim_config(optimizer="sgd", lr=5e-4, min_lr=1e-5, clip_grad=0.5, weight_decay=0.1)
    init_megatron_optim_config(cfg)

    assert captured_args["optimizer"] == "sgd"
    assert captured_args["lr"] == pytest.approx(5e-4)
    assert captured_args["min_lr"] == pytest.approx(1e-5)
    assert captured_args["clip_grad"] == pytest.approx(0.5)
    assert captured_args["weight_decay"] == pytest.approx(0.1)


def test_override_optimizer_config_overrides_branch_defaults(captured_args):
    cfg = _precision_aware_optim_config(
        override_optimizer_config={
            "use_precision_aware_optimizer": False,
            "exp_avg_dtype": "sentinel-override",
        },
    )
    init_megatron_optim_config(cfg, bf16=True)

    # User-supplied overrides win over the opted-in bf16 defaults …
    assert captured_args["use_precision_aware_optimizer"] is False
    assert captured_args["exp_avg_dtype"] == "sentinel-override"
    # … but non-overridden bf16 defaults remain.
    assert captured_args["main_grads_dtype"] is torch.bfloat16
    assert captured_args["exp_avg_sq_dtype"] is torch.bfloat16


def test_missing_override_config_leaves_branch_defaults_intact(captured_args):
    """``optim_config.get('override_optimizer_config', {})`` must not crash when absent."""
    cfg = _precision_aware_optim_config()
    assert "override_optimizer_config" not in cfg

    init_megatron_optim_config(cfg, bf16=True)

    assert captured_args["use_precision_aware_optimizer"] is True
    assert captured_args["exp_avg_dtype"] is torch.bfloat16


class _FakeDistributedOptimizer:
    """The subset of ``DistributedOptimizer`` that ``_realign_param_group_index_map`` reads.

    ``model_params`` is in ``named_parameters()`` order and all of them share a single
    optimizer param group, matching what ``_get_param_groups`` produces: it keys groups on
    config overrides and expert-parallelism, never on dtype.

    Args:
        model_params: model params in ``named_parameters()`` order.
        buffer_order: order in which the grad buffers hand params to
            ``_build_optimizer_group_ranges``, which is what ``group_order`` counts.
        install_fp32_shards_first: whether ``_build_model_and_main_param_groups`` puts the
            fp32 shards ahead of the 16-bit main shards (what Megatron does today).
    """

    def __init__(self, model_params, buffer_order, install_fp32_shards_first=True):
        self.config = MagicMock(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
        bf16_params = [p for p in model_params if p.dtype is torch.bfloat16]
        fp32_params = [p for p in model_params if p.dtype is torch.float32]

        self.shard_of = {p: torch.zeros_like(p) for p in model_params}
        self.model_float16_groups = [bf16_params]
        self.model_fp32_groups = [fp32_params]
        self.shard_fp32_from_float16_groups = [[self.shard_of[p] for p in bf16_params]]
        self.shard_fp32_groups = [[self.shard_of[p] for p in fp32_params]]
        self.shard_float16_groups = [[None] * len(bf16_params)]

        installed = fp32_params + bf16_params if install_fp32_shards_first else list(buffer_order)
        self.optimizer = SimpleNamespace(param_groups=[{"params": [self.shard_of[p] for p in installed]}])
        self.model_param_group_index_map = {p: (0, i) for i, p in enumerate(buffer_order)}

    def index_map_is_consistent(self):
        installed = self.optimizer.param_groups[0]["params"]
        return all(
            installed[self.model_param_group_index_map[p][1]] is shard for p, shard in self.shard_of.items()
        )


def _dsv4_style_params():
    """bf16 weights followed by the params DSv4 keeps in fp32 via ``mark_keep_in_fp32``."""
    return [
        torch.zeros(4096, dtype=torch.bfloat16),  # linear_qkv.weight
        torch.zeros(2048, dtype=torch.bfloat16),  # linear_proj.weight
        torch.zeros(384, dtype=torch.float32),  # mhc.mapping_proj.weight
        torch.zeros(64, dtype=torch.float32),  # csa.ape
    ]


def test_realign_fixes_index_map_for_mixed_dtype_param_group():
    """The DSv4 case: the bf16 grad buffer is allocated first, so the orders disagree."""
    params = _dsv4_style_params()
    opt = _FakeDistributedOptimizer(params, buffer_order=params)
    assert not opt.index_map_is_consistent()

    opt_mod._realign_param_group_index_map(opt)

    assert opt.index_map_is_consistent()


def test_realign_is_noop_when_orders_already_agree():
    """An fp32-first buffer order (like a dtype-homogeneous group) needs no repair."""
    params = _dsv4_style_params()
    opt = _FakeDistributedOptimizer(params, buffer_order=[params[2], params[3], params[0], params[1]])
    before = dict(opt.model_param_group_index_map)
    assert opt.index_map_is_consistent()

    opt_mod._realign_param_group_index_map(opt)

    assert opt.model_param_group_index_map == before


def test_realign_is_noop_once_megatron_preserves_buffer_order():
    """Guard against fighting a future Megatron that installs params in buffer order."""
    params = _dsv4_style_params()
    opt = _FakeDistributedOptimizer(params, buffer_order=params, install_fp32_shards_first=False)
    before = dict(opt.model_param_group_index_map)
    assert opt.index_map_is_consistent()

    opt_mod._realign_param_group_index_map(opt)

    assert opt.model_param_group_index_map == before


def test_realign_walks_every_chained_optimizer():
    """Megatron chains one DistributedOptimizer per dense / expert param set."""
    dense_params, expert_params = _dsv4_style_params(), _dsv4_style_params()
    dense = _FakeDistributedOptimizer(dense_params, buffer_order=dense_params)
    expert = _FakeDistributedOptimizer(expert_params, buffer_order=expert_params)
    assert not dense.index_map_is_consistent()
    assert not expert.index_map_is_consistent()

    opt_mod._realign_param_group_index_map(SimpleNamespace(chained_optimizers=[dense, expert]))

    assert dense.index_map_is_consistent()
    assert expert.index_map_is_consistent()


def test_realign_tolerates_optimizers_without_distributed_optimizer_attributes():
    """Muon / non-distributed / stubbed optimizers must pass through untouched."""
    for candidate in (MagicMock(), SimpleNamespace(), ("optimizer", MagicMock())):
        opt_mod._realign_param_group_index_map(candidate)


def _dist_opt_with_hdo(precision_aware=False, param_update_in_fp32=True, param_to_fp32_param=None):
    """DistributedOptimizer wrapping a HybridDeviceOptimizer, as optimizer_cpu_offload builds it."""
    return SimpleNamespace(
        config=SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=precision_aware),
        optimizer=SimpleNamespace(
            sub_optimizers=[],
            param_update_in_fp32=param_update_in_fp32,
            param_to_fp32_param=param_to_fp32_param or {},
        ),
    )


def _saved_tensors():
    param = torch.ones(8)
    return {
        "param": param,
        "exp_avg": torch.zeros(8),
        "exp_avg_sq": torch.zeros(8),
        # HDO publishes the pinned CPU copy of `param` under this key.
        "master_param": param.clone(),
    }


def test_master_param_is_dropped_for_cpu_offloaded_optimizer():
    dist_opt = _dist_opt_with_hdo()
    assert opt_mod._hdo_master_param_is_duplicate(dist_opt)

    tensors = opt_mod._drop_duplicate_master_param(dist_opt, _saved_tensors())

    assert set(tensors) == {"param", "exp_avg", "exp_avg_sq"}


def test_master_param_is_rederived_from_param_on_load():
    dist_opt = _dist_opt_with_hdo()
    saved = opt_mod._drop_duplicate_master_param(dist_opt, _saved_tensors())

    restored = opt_mod._restore_duplicate_master_param(dist_opt, saved)

    assert restored["master_param"] is restored["param"]
    # `_set_main_param_and_optimizer_states` indexes by the live state's keys, so every key it
    # can ask for has to be present again after the round trip.
    assert set(restored) == set(_saved_tensors())


def test_load_leaves_pre_dedup_checkpoints_untouched():
    """Checkpoints written before the dedup still carry their own master_param."""
    dist_opt = _dist_opt_with_hdo()
    saved = _saved_tensors()

    restored = opt_mod._restore_duplicate_master_param(dist_opt, saved)

    assert restored is saved
    assert restored["master_param"] is not restored["param"]


@pytest.mark.parametrize(
    "dist_opt",
    [
        # Precision-aware: master_param is the optimizer's real master weight.
        _dist_opt_with_hdo(precision_aware=True),
        # HDO without fp32 param updates never publishes a duplicate.
        _dist_opt_with_hdo(param_update_in_fp32=False),
        # A real dtype upcast is not recoverable from `param`.
        _dist_opt_with_hdo(param_to_fp32_param={"p": object()}),
        # Plain Adam, no CPU offload.
        SimpleNamespace(config=SimpleNamespace(), optimizer=SimpleNamespace(param_groups=[])),
    ],
)
def test_master_param_is_preserved_when_not_a_duplicate(dist_opt):
    assert not opt_mod._hdo_master_param_is_duplicate(dist_opt)

    tensors = opt_mod._drop_duplicate_master_param(dist_opt, _saved_tensors())

    assert "master_param" in tensors


class _FakeHDO:
    """The slice of HybridDeviceOptimizer that the fp32 sync touches.

    ``state`` mirrors what ``_sync_hdo_state_to_sub_optimizers`` leaves behind: it reads
    ``self.state[orig_param]`` off a defaultdict, so every param ends up present even when it
    carries no optimizer state yet.
    """

    def __init__(self, params, upcast=(), param_update_in_fp32=True):
        self.param_update_in_fp32 = param_update_in_fp32
        # Tensors compare elementwise, so membership has to go through identity.
        upcast_ids = {id(p) for p in upcast}
        # The HDO upcasts to fp32, so the counterpart's dtype differs from the param's.
        self.param_to_fp32_param = {p: torch.zeros(p.shape, dtype=torch.float32) for p in upcast}
        self.state = {
            p: ({"master_param": torch.full(p.shape, 7.0)} if id(p) in upcast_ids else {}) for p in params
        }


def test_fp32_sync_tolerates_params_that_were_never_upcast():
    """DistributedOptimizer only hands fp32 shards over, so param_to_fp32_param stays empty."""
    params = [torch.ones(4), torch.ones(4)]
    hdo = _FakeHDO(params)
    assert hdo.param_to_fp32_param == {}

    opt_mod._sync_fp32_params_from_state(hdo)  # upstream raises KeyError here


def test_fp32_sync_still_copies_upcast_params():
    fp16_param = torch.ones(4, dtype=torch.float16)
    fp32_param = torch.ones(4)
    hdo = _FakeHDO([fp16_param, fp32_param], upcast=[fp16_param])

    opt_mod._sync_fp32_params_from_state(hdo)

    assert torch.equal(hdo.param_to_fp32_param[fp16_param], torch.full((4,), 7.0))


def test_fp32_sync_skips_entries_without_master_param():
    param = torch.ones(4)
    hdo = _FakeHDO([param], upcast=[param])
    hdo.state[param] = {}  # populated by the defaultdict read, no state stored yet

    opt_mod._sync_fp32_params_from_state(hdo)

    assert torch.equal(hdo.param_to_fp32_param[param], torch.zeros(4))


def test_fp32_sync_is_a_noop_without_fp32_param_updates():
    param = torch.ones(4)
    hdo = _FakeHDO([param], upcast=[param], param_update_in_fp32=False)

    opt_mod._sync_fp32_params_from_state(hdo)

    assert torch.equal(hdo.param_to_fp32_param[param], torch.zeros(4))


def test_fp32_param_sync_patch_is_idempotent(monkeypatch):
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    monkeypatch.delattr(HybridDeviceOptimizer, "_verl_fp32_param_sync_guard", raising=False)
    pristine = HybridDeviceOptimizer._update_fp32_params_by_new_state
    monkeypatch.setattr(HybridDeviceOptimizer, "_update_fp32_params_by_new_state", pristine)

    opt_mod.apply_hdo_fp32_param_sync_patch()
    assert HybridDeviceOptimizer._update_fp32_params_by_new_state is opt_mod._sync_fp32_params_from_state

    opt_mod.apply_hdo_fp32_param_sync_patch()
    assert HybridDeviceOptimizer._update_fp32_params_by_new_state is opt_mod._sync_fp32_params_from_state

    monkeypatch.delattr(HybridDeviceOptimizer, "_verl_fp32_param_sync_guard", raising=False)


def test_rebuild_redoes_the_hdo_device_partition():
    calls = []
    hdo = SimpleNamespace(
        sub_optimizers=[],
        gpu_params_map_cpu_copy={},
        _init_sub_optimizers=lambda: calls.append("init"),
        _sync_hdo_param_groups_to_sub_optimizers=lambda: calls.append("groups"),
        _sync_hdo_state_to_sub_optimizers=lambda: calls.append("state"),
    )

    opt_mod.rebuild_hdo_sub_optimizers_after_load(SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=hdo)]))

    # Order matters: the state has to be reinstalled into the freshly built sub-optimizers.
    assert calls == ["init", "groups", "state"]


def test_rebuild_is_a_noop_without_cpu_offload():
    for candidate in (
        MagicMock(),
        SimpleNamespace(),
        SimpleNamespace(optimizer=SimpleNamespace()),
        SimpleNamespace(optimizer=SimpleNamespace(param_groups=[])),
    ):
        opt_mod.rebuild_hdo_sub_optimizers_after_load(candidate)


def test_dedup_patch_is_idempotent_and_symmetric(monkeypatch):
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    monkeypatch.delattr(DistributedOptimizer, "_verl_master_param_dedup", raising=False)
    pristine_get = DistributedOptimizer._get_main_param_and_optimizer_states
    pristine_set = DistributedOptimizer._set_main_param_and_optimizer_states
    monkeypatch.setattr(DistributedOptimizer, "_get_main_param_and_optimizer_states", pristine_get)
    monkeypatch.setattr(DistributedOptimizer, "_set_main_param_and_optimizer_states", pristine_set)

    opt_mod.apply_hdo_master_param_dedup_patch()
    patched_get = DistributedOptimizer._get_main_param_and_optimizer_states
    patched_set = DistributedOptimizer._set_main_param_and_optimizer_states
    assert patched_get is not pristine_get
    assert patched_set is not pristine_set

    opt_mod.apply_hdo_master_param_dedup_patch()
    assert DistributedOptimizer._get_main_param_and_optimizer_states is patched_get
    assert DistributedOptimizer._set_main_param_and_optimizer_states is patched_set

    monkeypatch.delattr(DistributedOptimizer, "_verl_master_param_dedup", raising=False)
