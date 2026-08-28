#!/usr/bin/env bash
set -xeuo pipefail

export WK=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

# --- single-node 8-GPU layout -------------------------------------------------
export NNODES=1
export NGPUS_PER_NODE=8

export ACTOR_PP=1
export ACTOR_EP=${ACTOR_EP:-8}
export ACTOR_ETP=1
export ACTOR_CP=${ACTOR_CP:-1}
export ROLLOUT_EP=8

export PIPELINE_MODEL_PARALLEL_LAYOUT="null"

# --- run length ---------------------------------------------------------------
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=1
export TEST_FREQ=-1
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}

export MODEL_PATH=${MODEL_PATH:-/models/DeepSeek-V4-Flash-Base-7L}
export TRAIN_FILE=${TRAIN_FILE:-/models/retool_aime2024/train.parquet} #retool_dapo
export TEST_FILE=${TEST_FILE:-/models/retool_aime2024/train.parquet}


export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-10204}

export ALL_OFFLOAD=${ALL_OFFLOAD:-True}
export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}

export ROLLOUT_N=${ROLLOUT_N:-16}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
export ROLLOUT_KV_CACHE_DTYPE=${ROLLOUT_KV_CACHE_DTYPE:-fp8}

export ROLLOUT_MOE_BACKEND=${ROLLOUT_MOE_BACKEND:-aiter}

export PROJECT_NAME=${PROJECT_NAME:-DEBUG_dsv4_7L}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-single_node_debug}
# export ACTOR_LR=0.0

export REPO_ROOT="/app/verl"

# TileLang's libcuda_stub.so preempts vLLM's cuGetErrorString and aborts the worker with a
# bogus "libcuda.so not found" instead of formatting the real HIP error, so any allocator
# failure (e.g. an OOM during wake_up) becomes unreadable. Forward the symbol to HIP.
export LD_PRELOAD="/opt/verl-patches/librocm_cuerr_shim.so${LD_PRELOAD:+:${LD_PRELOAD}}"

# Logger: console-only during bring-up to avoid external (wandb) failure surface.
# Re-enable wandb by overriding trainer.logger and passing WANDB_API_KEY.
bash "${REPO_ROOT}/examples/grpo_trainer/run_deepseek_v4_flash_megatron.sh" \
    trainer.logger='["console"]' \
    trainer.val_before_train=False \
    data.gen_batch_size=${TRAIN_BATCH_SIZE} \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout=null \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.pipeline_model_parallel_layout=null \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.apply_dsa_kernel_fusion=False \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.apply_dsa_kernel_fusion=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=${ROLLOUT_MOE_BACKEND} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode=deepseek_v4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=deepseek_v4 \
    data.dataloader_num_workers=0 \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY=${WK} \
    ++trainer.default_local_dir="/models/temp" \
    "$@"

