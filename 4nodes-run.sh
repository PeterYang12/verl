#!/usr/bin/env bash
set -xeuo pipefail

export WK=

export NNODES=4
export NGPUS_PER_NODE=8

export ACTOR_PP=4
export ACTOR_EP=8
export ACTOR_ETP=1
export ACTOR_CP=1

export ROLLOUT_TP=8

export PIPELINE_MODEL_PARALLEL_LAYOUT="Et*11|t*11|t*11|t*10L"
export CKPTS_DIR=/models/verl-ds-data/

export DEEPSEEK_V4_FLASH_CHAT_TEMPLATE='{{ "<｜begin▁of▁sentence｜>" }}{% for message in messages %}{% if message["role"] == "system" %}{{ message["content"] }}{% elif message["role"] == "user" %}{{ "<｜User｜>" + message["content"] }}{% elif message["role"] == "assistant" %}{{ "<｜Assistant｜></think>" + message["content"] + "<｜end▁of▁sentence｜>" }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ "<｜Assistant｜><think>" }}{% endif %}'

# --- run length ---------------------------------------------------------------
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=-1
export TEST_FREQ=-1
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-50}

export MODEL_PATH=${MODEL_PATH:-/models/DeepSeek-V4-Flash}
#export TRAIN_FILE=${TRAIN_FILE:-/models/dapo/data/dapo-math-17k.parquet}
#export TEST_FILE=${TEST_FILE:-/models/aime/data/aime-2024.parquet}

export TRAIN_FILE=${TRAIN_FILE:-/models/gsm8k/train.parquet}
export TEST_FILE=${TEST_FILE:-/models/gsm8k/test.parquet}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}

export ALL_OFFLOAD=${ALL_OFFLOAD:-True}
export OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}

export ROLLOUT_N=${ROLLOUT_N:-8}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.3}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
export ROLLOUT_KV_CACHE_DTYPE=${ROLLOUT_KV_CACHE_DTYPE:-fp8}

export PROJECT_NAME=${PROJECT_NAME:-ROCM_DSV4_flash}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-0805_run1}

export REPO_ROOT="/app/verl"

bash "${REPO_ROOT}/examples/grpo_trainer/run_deepseek_v4_flash_megatron.sh" \
    trainer.logger='["console","wandb"]' \
    trainer.val_before_train=False \
    trainer.rollout_data_dir="${CKPTS_DIR}/rollout_dump" \
    data.gen_batch_size=${TRAIN_BATCH_SIZE} \
    "++data.apply_chat_template_kwargs.chat_template='${DEEPSEEK_V4_FLASH_CHAT_TEMPLATE}'" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.apply_dsa_kernel_fusion=False \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.apply_dsa_kernel_fusion=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=aiter \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode=deepseek_v4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=deepseek_v4 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    data.dataloader_num_workers=8 \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY=${WK} \
    "$@"

