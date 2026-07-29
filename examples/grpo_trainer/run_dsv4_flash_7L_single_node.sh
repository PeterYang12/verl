#!/usr/bin/env bash
set -xeuo pipefail

# --- single-node 8-GPU layout -------------------------------------------------
export NNODES=1
export NGPUS_PER_NODE=8

export ACTOR_PP=1
export ACTOR_EP=8
export ACTOR_ETP=1
export ACTOR_CP=1
export ROLLOUT_TP=8

export PIPELINE_MODEL_PARALLEL_LAYOUT="null"

# --- run length ---------------------------------------------------------------
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=-1
export TEST_FREQ=-1
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}

export MODEL_PATH=${MODEL_PATH:-/models/DeepSeek-V4-Flash}
export TRAIN_FILE=${TRAIN_FILE:-/models/gsm8k/train.parquet}
export TEST_FILE=${TEST_FILE:-/models/gsm8k/test.parquet}


export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

export ALL_OFFLOAD=${ALL_OFFLOAD:-False}
export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0.0}

export ROLLOUT_N=${ROLLOUT_N:-1}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.3}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-4}
export ROLLOUT_KV_CACHE_DTYPE=${ROLLOUT_KV_CACHE_DTYPE:-fp8}

export PROJECT_NAME=${PROJECT_NAME:-DEBUG_dsv4_7L}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-single_node}

export REPO_ROOT="/app/verl"
# export VERL_SKIP_INITIAL_WEIGHT_SYNC=1

# Logger: console-only during bring-up to avoid external (wandb) failure surface.
# Re-enable wandb by overriding trainer.logger and passing WANDB_API_KEY.
bash "${REPO_ROOT}/examples/grpo_trainer/run_deepseek_v4_flash_megatron.sh" \
    trainer.logger='["console"]' \
    trainer.val_before_train=False \
    data.gen_batch_size=${TRAIN_BATCH_SIZE} \
    ++actor_rollout_ref.actor.optim.lr=0 \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout=null \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.pipeline_model_parallel_layout=null \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.apply_dsa_kernel_fusion=False \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.apply_dsa_kernel_fusion=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=aiter \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode=deepseek_v4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=deepseek_v4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=False \
    data.dataloader_num_workers=0 \
    +rollout.free_cache_engine=False \
    "$@"

    # ++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    # ++actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False \
    #     ++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    # ++actor_rollout_ref.ref.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    # ++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION} \
    
    # actor_rollout_ref.rollout.load_format=auto \
    # actor_rollout_ref.model.path=/models/DeepSeek-V4-Flash-3L-clean \
