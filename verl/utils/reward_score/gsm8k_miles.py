# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Full-response GSM8K scoring following MILES' math-reward conventions.

MILES does not currently expose a dedicated ``gsm8k`` reward type.  Its math
rewards inspect the complete response and use a 1/0 correctness reward.  This
scorer applies those conventions to GSM8K's canonical ``#### number`` answer
format instead of verl's 300-character suffix approximation.

Configure it with::

    reward.custom_reward_function.path=pkg://verl.utils.reward_score.gsm8k_miles
    reward.custom_reward_function.name=compute_score

Set ``incorrect_score=-1`` if a +1/-1 DAPO-style reward range is desired.
"""

import re
from decimal import Decimal, InvalidOperation

_STRICT_ANSWER_PATTERN = re.compile(r"####\s*\$?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
_NUMBER_PATTERN = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")


def _normalize_number(value: str | int | float | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value).strip().replace(",", "").replace("$", ""))
    except (InvalidOperation, ValueError):
        return None


def extract_solution(solution_str: str, method: str = "strict") -> str | None:
    """Extract the final answer from the complete response.

    ``strict`` requires GSM8K's ``####`` marker. ``flexible`` falls back to the
    last numeric expression when the marker is absent.
    """
    if method not in {"strict", "flexible"}:
        raise ValueError(f"Unsupported extraction method: {method}")

    strict_matches = _STRICT_ANSWER_PATTERN.findall(solution_str)
    if strict_matches:
        return strict_matches[-1].replace(",", "").replace("$", "")
    if method == "strict":
        return None

    number_matches = _NUMBER_PATTERN.findall(solution_str)
    return number_matches[-1].replace(",", "") if number_matches else None


def compute_score(
    data_source=None,
    solution_str="",
    ground_truth="",
    extra_info=None,
    method="strict",
    correct_score=1.0,
    incorrect_score=0.0,
    format_score=0.0,
    **kwargs,
):
    """Score a GSM8K response without truncating it before answer extraction."""
    prediction = extract_solution(solution_str, method=method)
    if prediction is None:
        return {
            "score": float(format_score),
            "acc": False,
            "pred": "[INVALID]",
        }

    normalized_prediction = _normalize_number(prediction)
    normalized_ground_truth = _normalize_number(ground_truth)
    correct = normalized_prediction is not None and normalized_prediction == normalized_ground_truth
    return {
        "score": float(correct_score if correct else incorrect_score),
        "acc": correct,
        "pred": prediction,
    }
