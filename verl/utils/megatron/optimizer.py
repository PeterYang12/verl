# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

import dataclasses
import math

import torch
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer as get_megatron_optimizer_native
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from verl.utils.logger import print_rank_0
from verl.utils.torch_dtypes import PrecisionType

# Names of the Muon (emerging optimizer) algorithms recognized by Megatron-Core's
# ``get_megatron_optimizer`` (anything other than "adam"/"sgd" routes to the emerging path).
_MUON_ALGORITHMS = ("muon",)


def is_muon_layer_wise_config(optim_config) -> bool:
    """True when verl will build Megatron's LayerWiseDistributedOptimizer path."""
    algo = str(getattr(optim_config, "optimizer", optim_config.get("optimizer", ""))).lower()
    if algo not in _MUON_ALGORITHMS:
        return False
    return bool(
        getattr(optim_config, "use_layer_wise_distributed_optimizer", None)
        or optim_config.get("use_layer_wise_distributed_optimizer", False)
    )


# Muon knobs exposed on verl's ``McoreOptimizerConfig`` that mirror like-named fields on
# Megatron-Core's ``OptimizerConfig``. Only the ones the installed Megatron actually declares are
# forwarded (older Megatron builds without emerging_optimizers won't have them).
_MUON_PASSTHROUGH_FIELDS = (
    "use_layer_wise_distributed_optimizer",
    "use_layer_wise_param_layout",
    "muon_momentum",
    "muon_nesterov",
    "muon_split_qkv",
    "muon_scale_mode",
    "muon_coefficient_type",
    "muon_num_ns_steps",
    "muon_tp_mode",
    "muon_fp32_matmul_prec",
    "muon_extra_scale_factor",
    "muon_scalar_optimizer",
)


def _add_muon_args(optim_args: dict, optim_config: dict) -> None:
    """Forward the Muon hyperparameters onto the Megatron ``OptimizerConfig`` kwargs.

    Only fields declared by the installed ``OptimizerConfig`` are forwarded so this stays compatible
    with Megatron builds that lack (some of) the Muon knobs. If a Muon algorithm is requested but the
    installed Megatron exposes none of the Muon fields, we fail loudly instead of letting Megatron
    silently fall back to Adam.
    """
    supported_fields = {f.name for f in dataclasses.fields(OptimizerConfig)}
    forwarded = []
    for field in _MUON_PASSTHROUGH_FIELDS:
        if field not in supported_fields:
            continue
        value = optim_config.get(field, None)
        if value is None:
            continue
        optim_args[field] = value
        forwarded.append(field)

    muon_related = supported_fields & set(_MUON_PASSTHROUGH_FIELDS)
    if not muon_related:
        raise ValueError(
            f"optimizer={optim_args['optimizer']!r} requests Muon, but the installed "
            "megatron.core.optimizer.OptimizerConfig exposes no Muon fields. Muon requires a "
            "Megatron-Core build with emerging_optimizers support; refusing to fall back to Adam."
        )
    print_rank_0(f"Muon optimizer selected; forwarded fields: {forwarded}")


def adamw_rms_match_scale_factor(beta1: float) -> float:
    """Extra Muon scale factor that matches AdamW's update RMS norm.

    ``emerging_optimizers`` 0.3.0 documents the closed form in
    ``orthogonalized_optimizers/muon.py::get_muon_scale_factor``:

        "Default mode is 'spectral', which is the mode that allows for learning rate
        transferability from AdamW. An extra scale factor is used to match the update RMS
        norm of AdamW, so that we can transfer hyperparameters from AdamW to Muon. An extra
        scale factor of sqrt((1-B1)/(1+B1)), where B1 is AdamW's momentum EMA coefficient,
        analytically gives the update RMS norm of AdamW (https://kexue.fm/archives/11267)."

    ``muon_scale_mode`` and ``muon_extra_scale_factor`` are orthogonal and both matter:
    the former normalizes for parameter *shape*, this one for the *momentum/EMA*. See also
    https://arxiv.org/abs/2502.16982, which is where the widely quoted ~0.2 value comes
    from -- it is the value of the *factor* at B1=0.9, not a target for any measured
    update RMS.

    Args:
        beta1: AdamW's first moment (momentum EMA) coefficient.

    Returns:
        The extra scale factor, e.g. 0.229416 at ``beta1=0.9``.
    """
    if not 0.0 <= beta1 < 1.0:
        raise ValueError(f"beta1 must be in [0, 1) to match AdamW update RMS, got {beta1!r}")
    return math.sqrt((1.0 - beta1) / (1.0 + beta1))


def init_megatron_optim_config(
    optim_config: dict,
    use_distributed_optimizer: bool = True,
    fp16: bool = False,
    bf16: bool = True,
) -> OptimizerConfig:
    optim_args = {
        "optimizer": optim_config.optimizer,
        "lr": optim_config.lr,
        "min_lr": optim_config.min_lr,
        "clip_grad": optim_config.clip_grad,
        "weight_decay": optim_config.weight_decay,
        "use_distributed_optimizer": use_distributed_optimizer,
    }
    if str(optim_config.optimizer).lower() in _MUON_ALGORITHMS:
        _add_muon_args(optim_args, optim_config)
        if optim_config.get("muon_match_adamw_update_rms", False):
            explicit = optim_config.get("muon_extra_scale_factor", 1.0)
            if explicit != 1.0:
                raise ValueError(
                    "muon_match_adamw_update_rms=True derives muon_extra_scale_factor from beta1, "
                    f"but muon_extra_scale_factor was also set explicitly to {explicit!r}. "
                    "Set exactly one of the two."
                )
            beta1 = tuple(optim_config.get("betas", (0.9, 0.999)))[0]
            resolved = adamw_rms_match_scale_factor(beta1)
            optim_args["muon_extra_scale_factor"] = resolved
            print_rank_0(
                "muon_match_adamw_update_rms=True: muon_extra_scale_factor resolved to "
                f"{resolved!r} from sqrt((1-beta1)/(1+beta1)) with beta1={beta1!r}"
            )
        # Megatron buffer-integrated master weights (avoids Float16Optimizer fp32 clones).
        supported_fields = {f.name for f in dataclasses.fields(OptimizerConfig)}
        if (
            "use_layer_wise_param_layout" in supported_fields
            and is_muon_layer_wise_config(optim_config)
            and getattr(optim_config, "use_layer_wise_param_layout", None) is None
        ):
            optim_args["use_layer_wise_param_layout"] = True
    if fp16:
        optim_args.update(
            {
                "bf16": False,
                "fp16": True,
                "params_dtype": torch.float16,
                "initial_loss_scale": 32768,
                "min_loss_scale": 1,
                "use_precision_aware_optimizer": True,
                "store_param_remainders": False,
            }
        )
    elif bf16:
        optim_args.update(
            {
                "bf16": True,
                "params_dtype": torch.bfloat16,
            }
        )
        # Precision-aware optimizer is opt-in (default keeps the grad-accumulation
        # buffer and Adam moments (m, v) at Megatron's fp32 default, preserving
        # prior numerics). When enabled via config, those buffers follow the
        # configured dtypes so optimizer-state memory can track the model dtype.
        # Master parameters stay fp32 (Megatron default `main_params_dtype`)
        # because TE FusedAdam currently rejects bf16 master weights at init
        # (only fp32/fp16 accepted); the int16 `store_param_remainders` path
        # already trims the fp32 master buffer in bf16 mode. Requires
        # TransformerEngine's FusedAdam. The DDP grad-bucket dtype is kept
        # consistent with `main_grads_dtype` by the engine at model-build time.
        if optim_config.get("use_precision_aware_optimizer", False):
            optim_args.update(
                {
                    "use_precision_aware_optimizer": True,
                    "main_grads_dtype": PrecisionType.to_dtype(optim_config.get("main_grads_dtype", "fp32")),
                    "exp_avg_dtype": PrecisionType.to_dtype(optim_config.get("exp_avg_dtype", "fp32")),
                    "exp_avg_sq_dtype": PrecisionType.to_dtype(optim_config.get("exp_avg_sq_dtype", "fp32")),
                }
            )
    else:
        # fp32 mode: leave grad-accumulation buffer and Adam moments at
        # Megatron's default torch.float32. Do not enable the precision-aware
        # optimizer — it's only beneficial when a moment/grad dtype is below
        # fp32, and Megatron asserts the dtype fields equal fp32 whenever the
        # precision-aware optimizer is off (optimizer_config.py:258-268).
        optim_args.update(
            {
                "bf16": False,
                "fp16": False,
                "params_dtype": torch.float32,
            }
        )
    override_config = optim_config.get("override_optimizer_config", {})
    if override_config:
        for k, v in override_config.items():
            optim_args[k] = v

    print_rank_0(f"optimizer config after override: {optim_args}")

    config = OptimizerConfig(**optim_args)
    return config


def _realign_param_group_index_map(optimizer) -> None:
    """Rebuild ``DistributedOptimizer.model_param_group_index_map`` from the inner optimizer.

    Megatron records each model param's ``group_order`` in grad-buffer iteration order
    (``_build_optimizer_group_ranges``), but hands the inner optimizer its params grouped by
    dtype, fp32 shards ahead of the 16-bit ones (``_build_model_and_main_param_groups``). The
    two orders only coincide while every param group is dtype-homogeneous. That stops holding
    for models keeping part of the weights in fp32 under bf16 training: DeepSeek V4 marks its
    mHC and sparse-attention params (``mapping_proj``, ``alpha_*``, ``bias``, ``ape``,
    ``attn_sink``) with Megatron's ``mark_keep_in_fp32``, and ``_get_param_groups`` never
    splits groups by dtype, so those land next to bf16 params.

    Checkpoint save/load resolves optimizer state through that index
    (``_get_main_param_and_optimizer_states``), so a stale index makes Megatron read another
    param's shard: saving trips the shape assert in ``sharded_param_state_dp_reshardable``,
    and loading would silently restore state onto the wrong params.

    Deriving the index from the inner optimizer's actual layout is a no-op whenever the two
    orders already agree.
    """
    # Duck-typed throughout: this runs on whatever Megatron build is installed, and only
    # DistributedOptimizer instances carry the attributes below.
    def _list_attr(obj, name):
        value = getattr(obj, name, None)
        return value if isinstance(value, list) else None

    chained = _list_attr(optimizer, "chained_optimizers")
    realigned = 0
    for opt in [optimizer] if chained is None else chained:
        index_map = getattr(opt, "model_param_group_index_map", None)
        param_groups = _list_attr(getattr(opt, "optimizer", None), "param_groups")
        if not isinstance(index_map, dict) or param_groups is None:
            continue

        if getattr(getattr(opt, "config", None), "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False):
            float16_shard_groups = _list_attr(opt, "shard_float16_groups")
        else:
            float16_shard_groups = _list_attr(opt, "shard_fp32_from_float16_groups")
        fp32_shard_groups = _list_attr(opt, "shard_fp32_groups")
        model_float16_groups = _list_attr(opt, "model_float16_groups")
        model_fp32_groups = _list_attr(opt, "model_fp32_groups")
        if None in (float16_shard_groups, fp32_shard_groups, model_float16_groups, model_fp32_groups):
            continue

        # Megatron keeps `None` placeholders for quantized params whose shard the inner
        # optimizer owns itself; they carry no identity to match on, so leave them alone.
        shard_to_index = {}
        for group_index, group in enumerate(param_groups):
            for group_order, shard in enumerate(group["params"]):
                if shard is not None:
                    shard_to_index[shard] = (group_index, group_order)

        for model_groups, shard_groups in (
            (model_float16_groups, float16_shard_groups),
            (model_fp32_groups, fp32_shard_groups),
        ):
            for model_group, shard_group in zip(model_groups, shard_groups, strict=True):
                for model_param, shard in zip(model_group, shard_group, strict=True):
                    index = shard_to_index.get(shard) if shard is not None else None
                    if index is None or index_map.get(model_param) == index:
                        continue
                    index_map[model_param] = index
                    realigned += 1

    if realigned:
        print_rank_0(
            f"Realigned {realigned} entries of model_param_group_index_map: this rank holds optimizer "
            "param groups mixing fp32 and 16-bit params, whose grad-buffer order does not match the "
            "order Megatron installs into the inner optimizer."
        )


def get_megatron_optimizer(
    model,
    config: OptimizerConfig,
):
    # Base optimizer.
    optimizer = get_megatron_optimizer_native(
        config=config,
        model_chunks=model,
    )
    _realign_param_group_index_map(optimizer)
    return optimizer


def get_megatron_optimizer_param_scheduler(
    optimizer,
    config,
):
    """
    Get the optimizer parameter scheduler for Megatron.
    """
    lr_decay_steps = config.lr_decay_steps
    lr_warmup_steps = config.lr_warmup_steps
    if config.get("lr_decay_steps", None) is None:
        lr_decay_steps = config.total_training_steps
    wsd_decay_steps = None
    if config.get("lr_wsd_decay_steps", None) is not None:
        wsd_decay_steps = config.lr_wsd_decay_steps
    if config.get("lr_warmup_steps_ratio", None) is not None and (
        config.get("lr_warmup_steps", None) is None or config.lr_warmup_steps <= 0
    ):
        lr_warmup_steps = int(config.lr_warmup_steps_ratio * lr_decay_steps)

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=config.lr_warmup_init,
        max_lr=config.lr,
        min_lr=config.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=config.lr_decay_style,
        start_wd=config.weight_decay,
        end_wd=config.weight_decay,
        wd_incr_steps=config.total_training_steps,
        wd_incr_style=config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=(not config.use_checkpoint_opt_param_scheduler),
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=config.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def get_megatron_last_lr(optimizer):
    """
    Get the last learning rate from the optimizer parameter scheduler.
    """
    return optimizer.param_groups[0]["lr"]
