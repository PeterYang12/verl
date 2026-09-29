# DeepSeek-V4-Flash GRPO on AMD ROCm

BF16 training with FP8 quantization-aware training (QAT) in Megatron, FP8 (W8A8) rollout in vLLM.
Validated on 4 nodes x 8 MI355X (gfx950), ROCm 7.2.3.

| File | Purpose |
|---|---|
| `run_deepseek_v4_flash_rocm_qat_fp8rollout.sh` | The recipe. Start this one. |
| `run_deepseek_v4_flash_megatron_rocm.sh` | Base launcher with all defaults. Called by the recipe. |
| `start_ray_rocm.sh` | Starts Ray inside a container. |

## 1. Image

```bash
DOCKER_BUILDKIT=1 docker build -f docker/rocm/Dockerfile.rocm.deepseek-v4 -t rocm-verl-dsv4 .
```

## 2. Data on every node

Put these on a **local disk** of every node, under one directory that is mounted as `/models`:

| Path in the container | Content |
|---|---|
| `/models/DeepSeek-V4-Flash-FP8-vllm` | `hf download PeterYang12/DeepSeek-V4-Flash-FP8-vllm --local-dir ...` |
| `/models/retool_dapo/train.parquet` | `python3 examples/data_preprocess/dapo_multiturn_w_tool.py --local_dir /models/retool_dapo` |
| `/models/retool_aime2024/train.parquet` | `python3 examples/data_preprocess/aime2024_multiturn_w_tool.py --local_dir /models/retool_aime2024` |

Do not keep the model on NFS. Every weight sync opens the checkpoint about 34,000 times to read
the source scales; on NFS that took 335 s per step instead of 83 s.

## 3. Container, on every node

```bash
docker run -itd --name dsv4_verl \
    --network=host --ipc=host --privileged \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband \
    --group-add=video --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    -v <local model dir>:/models \
    -e NCCL_IB_HCA=<RDMA devices> -e NCCL_IB_GID_INDEX=<index> \
    -e NCCL_SOCKET_IFNAME=<iface> -e GLOO_SOCKET_IFNAME=<iface> -e TP_SOCKET_IFNAME=<iface> \
    --entrypoint /bin/bash rocm-verl-dsv4
```

The `-e` values depend on the cluster.

## 4. Ray, inside the containers

```bash
bash examples/grpo_trainer/start_ray_rocm.sh head      # head node first
bash examples/grpo_trainer/start_ray_rocm.sh worker    # every other node
```

Set `HEAD_IP` and `RAY_IFNAME` if the defaults (`10.48.0.3`, `spur0`) do not match the cluster.
Run one job at a time on the cluster.

## 5. Training, inside the head container

```bash
cd /app/verl
WK=<wandb key> bash examples/grpo_trainer/run_deepseek_v4_flash_rocm_qat_fp8rollout.sh
```

Without `WK` the run logs to the console only. Useful variables: `TOTAL_TRAINING_STEPS` (default 300),
`SAVE_FREQ` (default -1, no checkpoint; a checkpoint is about 270 GB), `MODEL_PATH`, `EXPERIMENT_NAME`.

A short check:

```bash
TOTAL_TRAINING_STEPS=5 bash examples/grpo_trainer/run_deepseek_v4_flash_rocm_qat_fp8rollout.sh trainer.val_before_train=False
```

## 6. What to expect

Lines that must appear in the log during start-up:

```
QAT[fp8_blockwise]: 105 weight quantizers, 105 input quantizers enabled     (105 or 115, per pipeline stage)
KV fake quant: hooked 11 KV layernorms (nope=64-blocks, rope=64 kept bf16)
KV fake quant ext: hooked 9 compressor norms; indexer q/k fp8 emulation=on
Applying ue8m0 FP8 scale patch to Megatron-Bridge weight export...
```

Reference values:

| Metric | Value |
|---|---|
| `rollout_corr/kl` | 0.0021 at step 1 |
| `training/rollout_probs_diff_mean` | 0.0096 at step 1 |
| `timing_s/update_weights` | about 85 s |
| `timing_s/step` | about 1200 s for step 1 (kernels are compiled), about 490 s afterwards |
| initial `val-aux/math_dapo/reward/mean@1` | about 0.19 |

If `rollout_corr/kl` is near 0.005 at step 1, the vLLM SwiGLU clamp patch
(`docker/rocm/patches/vllm-dsv4-triton-moe-swiglu-clamp.patch`) is missing from the image.

## Settings the recipe depends on

| Setting | Why |
|---|---|
| `moe_backend=triton`, rollout TP=8, DP=1, EP=1 | AITER fused MoE is not deterministic on gfx950; Triton MoE with expert parallel gives wrong output in vLLM 0.26.0 |
| `NVTE_USE_GROUPED_GEMM_TRITON=1` and `gradient_accumulation_fusion=False` | BF16 backward of the Transformer Engine grouped GEMM can hang a GPU |
| `VERL_UE8M0_SCALE_FIX=1` | Weight export stays on the power-of-two scale grid of the checkpoint |
| `VERL_QAT_KV_FAKE_QUANT=1` | The vLLM KV cache is FP8 and cannot be turned off on ROCm |
| `VERL_FP8_ALIGN_WOA_FQ=1` | `wo_a` is weight-only FP8 in vLLM; needs the Megatron branch `deepseek-v4-rocm-support` |
| `rollout_is=token`, threshold 2.0 | Truncated importance sampling on the rollout/actor ratio |
