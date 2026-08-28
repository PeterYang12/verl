#!/usr/bin/env bash
# GRPO | DeepSeek-V4-Flash | vLLM rollout | Megatron training | NVIDIA GPUs
#
# Megatron-Bridge must be installed and importable on every node.
# The default configuration uses 4 nodes x 8 GPUs.
#
# With:
# - Megatron-Bridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11
# - Megatron-LM: https://github.com/NVIDIA/Megatron-LM/commit/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

############################### configs ################################

PROJECT_NAME=${PROJECT_NAME:-debug_ROCM_DSV4_flash}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-0824_ds_base_fp8_dapo_500steps}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-500}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}

NNODES=${NNODES:-4}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# ===================================== Algorithm =====================================
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
LOSS_MODE=${LOSS_MODE:-vanilla}
USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-False}
KL_COEF=${KL_COEF:-0.001}
USE_KL_LOSS=${USE_KL_LOSS:-False}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
ENTROPY_COEFF=${ENTROPY_COEFF:-1e-3}
ACTOR_LR=${ACTOR_LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-2e-6}
GAE_GAMMA=${GAE_GAMMA:-1.0}
GAE_LAM=${GAE_LAM:-0.95}
CRITIC_WARMUP=${CRITIC_WARMUP:-0}

# ===================================== Data/Model =====================================
MODEL_PATH=${MODEL_PATH:-/models/DeepSeek-V4-Flash-Base}
TRAIN_FILE=${TRAIN_FILE:-/models/retool_dapo/train.parquet}
TEST_FILE=${TEST_FILE:-/models/retool_aime2024/train.parquet}
CKPTS_DIR=${CKPTS_DIR:-/models/verl-ds-data/${PROJECT_NAME}/${EXPERIMENT_NAME}}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
ROLLOUT_N=${ROLLOUT_N:-8}
n_resp_per_prompt_val=16
ENABLE_THINKING=${ENABLE_THINKING:-True}

# Inference config
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_DP=${ROLLOUT_DP:-8}
ROLLOUT_EP=${ROLLOUT_EP:-8}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
ROLLOUT_KV_CACHE_DTYPE=${ROLLOUT_KV_CACHE_DTYPE:-fp8}
ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB:-512}
ROLLOUT_MOE_BACKEND=${ROLLOUT_MOE_BACKEND:-aiter}

# Training config
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
ACTOR_TP=${ACTOR_TP:-1}
ACTOR_PP=${ACTOR_PP:-4}
ACTOR_VPP=${ACTOR_VPP:-null}
ACTOR_EP=${ACTOR_EP:-8}
ACTOR_ETP=${ACTOR_ETP:-1}
ACTOR_CP=${ACTOR_CP:-1}
PIPELINE_MODEL_PARALLEL_LAYOUT=${PIPELINE_MODEL_PARALLEL_LAYOUT:-"Et*11|t*11|t*11|t*10L"}

REF_TP=${REF_TP:-${ACTOR_TP}}
REF_PP=${REF_PP:-${ACTOR_PP}}
REF_VPP=${REF_VPP:-${ACTOR_VPP}}
REF_EP=${REF_EP:-${ACTOR_EP}}
REF_ETP=${REF_ETP:-${ACTOR_ETP}}
REF_CP=${REF_CP:-${ACTOR_CP}}

ROUTER_REPLAY_MODE=${ROUTER_REPLAY_MODE:-R3}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

########################### parameter arrays ###########################

ALGORITHM=(
    algorithm.adv_estimator=${ADV_ESTIMATOR}
    algorithm.use_kl_in_reward=${USE_KL_IN_REWARD}
    algorithm.kl_ctrl.kl_coef=${KL_COEF}
    algorithm.gamma=${GAE_GAMMA}
    algorithm.lam=${GAE_LAM}
)

DATA=(
    data.train_files="$TRAIN_FILE"
    data.val_files="$TEST_FILE"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=False
    data.filter_overlong_prompts_workers=64
    data.truncation='error'
    data.dataloader_num_workers=${DATALOADER_NUM_WORKERS}
    +data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE}
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.router_replay.mode=${ROUTER_REPLAY_MODE}
    ++actor_rollout_ref.actor.megatron.override_transformer_config.apply_dsa_kernel_fusion=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_use_sparse_loss=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_loss_coeff=0.0
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    ++actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_mhc=False
    "++actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout='${PIPELINE_MODEL_PARALLEL_LAYOUT}'"
)

# Context parallelism needs three extra transformer-config settings for DeepSeek-V4 on top of
# `context_parallel_size`. Each of them is enforced by Megatron-Core, so without them the run
# aborts at model build or in the first attention forward. They are appended only when CP > 1.
#
#   cp_partition_mode=contiguous
#     DSv4 attention requires every CP rank to own ONE consecutive interval of the packed THD
#     buffer. Megatron-Core defaults to "zigzag" and raises
#     "DSv4 Hybrid with CP requires cp_partition_mode='contiguous'."
#   sequence_packing_scheduler=dp_balanced
#     Megatron-Core asserts "DSv4 Hybrid with CP requires a sequence_packing_scheduler for THD
#     inputs." Needs Transformer Engine >= 2.9.
#   max_seqlen_per_dp_cp_rank
#     Documented as max sequence length / cp_size; it drives how sub-samples are assigned to
#     each DPxCP rank.
CP_ARGS=()
if [ "${ACTOR_CP}" -gt 1 ]; then
    CP_ARGS=(
        ++actor_rollout_ref.actor.megatron.override_transformer_config.cp_partition_mode=contiguous
        ++actor_rollout_ref.actor.megatron.override_transformer_config.sequence_packing_scheduler=dp_balanced
        ++actor_rollout_ref.actor.megatron.override_transformer_config.max_seqlen_per_dp_cp_rank=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) / ACTOR_CP))
    )
fi

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP}
    actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt_val}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=${ROLLOUT_KV_CACHE_DTYPE}
)
    # actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    # actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    # actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}

REWARD=(
    reward.reward_manager.name=naive
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.critic_warmup=${CRITIC_WARMUP}
    trainer.resume_mode=auto
    trainer.val_before_train=True
    trainer.log_val_generations=100
    trainer.default_local_dir="${CKPTS_DIR}"
)

EXTRA=(
    actor_rollout_ref.nccl_timeout=3600
    model_engine=megatron
)

RUN_OVERRIDES=(
    ++actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout=null
    ++actor_rollout_ref.ref.megatron.override_transformer_config.pipeline_model_parallel_layout=null
    ++actor_rollout_ref.actor.megatron.override_transformer_config.apply_dsa_kernel_fusion=False
    ++actor_rollout_ref.ref.megatron.override_transformer_config.apply_dsa_kernel_fusion=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=${ROLLOUT_MOE_BACKEND}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode=deepseek_v4
    +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=deepseek_v4
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY=${WK}
)

########################### launch ###########################

python3 -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${CP_ARGS[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${RUN_OVERRIDES[@]}" \
    "$@"
