# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from verl.experimental.rl_kernel.runtime import prepare_worker
from verl.experimental.rl_kernel.trainer_contract import (
    logical_cases,
    strict_response_parity,
    validate_stage_acks,
)


def test_ray_assignment_replaces_inherited_cuda_alias(monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    prepare_worker()
    assert os.environ["HIP_VISIBLE_DEVICES"] == os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_ray_cpu_assignment_does_not_expand_visibility(monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    prepare_worker()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_two_visibility_namespaces_rejected(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-example")
    with pytest.raises(RuntimeError):
        prepare_worker()


def test_original_fp32_bits_and_counts():
    x = torch.tensor([[-3.25, -7.5], [-1.0, 0.0]])
    mask = torch.tensor([[1, 1], [1, 0]])
    result = strict_response_parity(x, x.clone(), mask)
    assert result == dict(
        expected_tokens=3,
        compared_tokens=3,
        mismatch_count=0,
        finite=True,
        mean_abs_diff=0.0,
        max_abs_diff=0.0,
    )


@pytest.mark.parametrize("change", ["ulp", "signed_zero"])
def test_single_bit_difference_is_failure(change):
    x = torch.tensor([-1.0 if change == "ulp" else 0.0])
    y = x.clone()
    if change == "ulp":
        y.view(torch.int32).add_(1)
    else:
        y[0] = -0.0
    with pytest.raises(RuntimeError, match="actor_bits=.*rollout_bits="):
        strict_response_parity(x, y, torch.ones(1))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scores_fail(bad):
    x = torch.tensor([bad])
    with pytest.raises(RuntimeError):
        strict_response_parity(x, x.clone(), torch.ones(1))


def test_empty_comparison_and_invalid_mask_fail():
    x = torch.tensor([-1.0])
    with pytest.raises(RuntimeError):
        strict_response_parity(x, x, torch.zeros(1))
    with pytest.raises(ValueError):
        strict_response_parity(x, x, torch.tensor([2]))
    with pytest.raises(TypeError):
        strict_response_parity(x.bfloat16(), x.bfloat16(), torch.ones(1))


def test_logical_keys_accept_real_unequal_batch_and_deduplicate_identical_rows():
    data = torch.nested.as_nested_tensor(
        [torch.tensor([4, 5]), torch.tensor([6]), torch.tensor([4, 5])], layout=torch.jagged
    )
    cases = logical_cases(data)
    assert [c["prompt_token_ids"] for c in cases] == [[4, 5], [6]]
    reordered = logical_cases(
        torch.nested.as_nested_tensor(
            [torch.tensor([6]), torch.tensor([4, 5])], layout=torch.jagged
        )
    )
    assert {c["id"] for c in cases} == {c["id"] for c in reordered}


def stage_acks():
    manifest = {
        "a": dict(dtype="torch.float32", shape=[2], sha256="a"),
        "b": dict(dtype="torch.bfloat16", shape=[1], sha256="b"),
    }
    acks = [
        dict(
            pipeline_stage=i,
            pid=100 + i,
            weight_version=1,
            parameter_load_completed=True,
            verified_original_tensors={name: dict(manifest[name])},
        )
        for i, name in enumerate(manifest)
    ]
    return manifest, acks


def test_full_sync_readback_and_pids():
    manifest, acks = stage_acks()
    assert validate_stage_acks(acks, manifest, 1, {0: 100, 1: 101}) == {0: 100, 1: 101}


@pytest.mark.parametrize("failure", ["missing", "dtype", "shape", "hash", "pid", "version"])
def test_sync_cannot_accept_incomplete_or_stale_state(failure):
    manifest, acks = stage_acks()
    if failure == "missing":
        acks[1]["verified_original_tensors"] = {}
    elif failure == "dtype":
        acks[1]["verified_original_tensors"]["b"]["dtype"] = "torch.float32"
    elif failure == "shape":
        acks[1]["verified_original_tensors"]["b"]["shape"] = [2]
    elif failure == "hash":
        acks[1]["verified_original_tensors"]["b"]["sha256"] = "bad"
    elif failure == "pid":
        acks[1]["pid"] = 999
    elif failure == "version":
        acks[1]["weight_version"] = 0
    with pytest.raises(RuntimeError):
        validate_stage_acks(acks, manifest, 1, {0: 100, 1: 101})


def test_native_engine_guard_in_fresh_process():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import verl.experimental.rl_kernel.trainer_engine; "
            "import verl.models.mcore; "
            "from verl.models.mcore.patch import apply_patch_megatron_v012_with_torch_v28_v29 as f; f()",
        ],
        check=True,
        timeout=90,
    )
