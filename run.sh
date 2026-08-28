#!/usr/bin/env bash
set -xeuo pipefail

export WK=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

# --- single-node 8-GPU layout -------------------------------------------------
export NNODES=4
export NGPUS_PER_NODE=8
export CKPTS_DIR=/models/verl-ds-data/

export ACTOR_PP=4
export ACTOR_EP=8
export ACTOR_ETP=1
export ACTOR_CP=1
export ROLLOUT_EP=8

export PIPELINE_MODEL_PARALLEL_LAYOUT="Et*11|t*11|t*11|t*10L"

# --- run length ---------------------------------------------------------------
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=-1
export TEST_FREQ=5
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}

export MODEL_PATH=${MODEL_PATH:-/models/DeepSeek-V4-Flash-Base}
export TRAIN_FILE=${TRAIN_FILE:-/models/retool_dapo/train.parquet}
export TEST_FILE=${TEST_FILE:-/models/retool_aime2024/train.parquet}


export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}

export ALL_OFFLOAD=${ALL_OFFLOAD:-True}
export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}

export ROLLOUT_N=${ROLLOUT_N:-8}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
export ROLLOUT_KV_CACHE_DTYPE=${ROLLOUT_KV_CACHE_DTYPE:-fp8}

export ROLLOUT_MOE_BACKEND=${ROLLOUT_MOE_BACKEND:-aiter}

export PROJECT_NAME=${PROJECT_NAME:-debug_ROCM_DSV4_flash}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-0822_ds_base_fp8_dapo_ori_reward}

export REPO_ROOT="/app/verl"

# Logger: console-only during bring-up to avoid external (wandb) failure surface.
# Re-enable wandb by overriding trainer.logger and passing WANDB_API_KEY.
bash "${REPO_ROOT}/examples/grpo_trainer/run_deepseek_v4_flash_megatron.sh" \
    trainer.logger='["console","wandb"]' \
    trainer.val_before_train=True \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout=null \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.pipeline_model_parallel_layout=null \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.apply_dsa_kernel_fusion=False \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.apply_dsa_kernel_fusion=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=${ROLLOUT_MOE_BACKEND} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode=deepseek_v4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=deepseek_v4 \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY=${WK} \
    "$@"