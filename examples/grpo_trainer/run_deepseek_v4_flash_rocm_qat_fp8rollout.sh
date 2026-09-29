#!/usr/bin/env bash
# DeepSeek-V4-Flash GRPO on AMD ROCm (4 nodes x 8 MI355X):
#   BF16 training + FP8 QAT (fake quant) in Megatron, FP8 (W8A8) rollout in vLLM.
#
# Run inside the container on the Ray head node, after the Ray cluster is up:
#   bash examples/grpo_trainer/run_deepseek_v4_flash_rocm_qat_fp8rollout.sh [hydra overrides...]
#
# Environment (all optional):
#   WK                     wandb API key. Unset -> console logging only.
#   MODEL_PATH             default /models/DeepSeek-V4-Flash-FP8-vllm
#   EXPERIMENT_NAME        default dsv4_flash_bf16train_qat_fp8rollout
#   TOTAL_TRAINING_STEPS   default 300
#   SAVE_FREQ              default -1 (no checkpoint). Set e.g. 50 to export an HF FP8 checkpoint
#                          every 50 steps: ~270 GB each, about 10 minutes on NFS, latest one kept
#   ROLLOUT_IS             default token (truncated importance sampling); "null" turns it off
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

if [ -z "${WK:-}" ]; then
    export WK=dummy
    LOGGER='["console"]'
    echo "WK is not set: logging to console only"
else
    LOGGER='["console","wandb"]'
fi

export MODEL_PATH=${MODEL_PATH:-/models/DeepSeek-V4-Flash-FP8-vllm}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-dsv4_flash_bf16train_qat_fp8rollout}
export SAVE_FREQ=${SAVE_FREQ:--1}

# Rollout: one vLLM engine, tensor parallel over the 8 GPUs of a node, Triton MoE.
#   - AITER fused MoE is not deterministic on gfx950.
#   - Triton MoE with expert parallel gives wrong output in vLLM 0.26.0.
export ROLLOUT_MOE_BACKEND=triton ROLLOUT_TP=8 ROLLOUT_DP=1 ROLLOUT_EP=1

export REPO_ROOT=/app/verl
cd /app/verl
EV='+ray_kwargs.ray_init.runtime_env.env_vars'

bash "${HERE}/run_deepseek_v4_flash_megatron_rocm.sh" \
    "$EV.VERL_USE_UV=\"0\"" \
    "$EV.VERL_UE8M0_SCALE_FIX=\"1\"" \
    "$EV.VERL_QAT_KV_FAKE_QUANT=\"1\"" \
    "$EV.VERL_FP8_ALIGN_WOA_FQ=\"1\"" \
    "$EV.NVTE_USE_GROUPED_GEMM_TRITON=\"1\"" \
    actor_rollout_ref.actor.megatron.qat.enable=True \
    actor_rollout_ref.actor.megatron.qat.mode=fp8_blockwise \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    algorithm.rollout_correction.rollout_is=${ROLLOUT_IS:-token} \
    algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0} \
    "actor_rollout_ref.actor.checkpoint.save_contents=[model]" \
    "trainer.logger=${LOGGER}" \
    "$@"
