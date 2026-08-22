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

"""Full-response GSM8K scoring for DeepSeek-V4-Flash Base.

The canonical GSM8K prompt asks for ``#### number``, while an untuned
DeepSeek-V4 policy may instead emit ``Answer: number`` or ``\\boxed{number}``.
This scorer accepts all three formats without applying a suffix window:

1. ``#### number`` (the requested GSM8K format);
2. ``Answer: number``;
3. ``\\boxed{number}``.

Within the selected format, the last answer is used by default. Set
``prefer_first_answer=True`` while the Base policy tends to answer and then
continue generating unrelated text.

Configure it with::

    reward.custom_reward_function.path=pkg://verl.utils.reward_score.gsm8k_deepseek_v4
    reward.custom_reward_function.name=compute_score
"""

import re

from verl.utils.reward_score.math_dapo_deepseek_v4 import _boxed_candidates
from verl.utils.reward_score.math_dapo_miles import extract_answer as extract_ground_truth
from verl.utils.reward_score.math_dapo_miles import grade_answer

_GSM8K_PATTERN = re.compile(r"####\s*\$?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
_ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")


def _select(candidates: list[str], prefer_first_answer: bool) -> str | None:
    if not candidates:
        return None
    return candidates[0 if prefer_first_answer else -1].strip()


def extract_solution(solution_str: str, prefer_first_answer: bool = False) -> tuple[str | None, str | None]:
    """Extract the requested GSM8K format, then try DeepSeek alternatives."""
    prediction = _select(_GSM8K_PATTERN.findall(solution_str), prefer_first_answer)
    if prediction is not None:
        return prediction.replace(",", "").replace("$", ""), "gsm8k"

    prediction = _select(_ANSWER_PATTERN.findall(solution_str), prefer_first_answer)
    if prediction is not None:
        return prediction, "answer"

    prediction = _select(_boxed_candidates(solution_str), prefer_first_answer)
    if prediction is not None:
        return prediction, "boxed"
    return None, None


def compute_score(
    data_source=None,
    solution_str="",
    ground_truth="",
    extra_info=None,
    prefer_first_answer=False,
    correct_score=1.0,
    incorrect_score=0.0,
    **kwargs,
):
    """Score a GSM8K response using all common DeepSeek answer formats."""
    ground_truth = str(ground_truth)
    if "\\boxed" in ground_truth:
        ground_truth = extract_ground_truth(ground_truth) or ground_truth

    prediction, answer_format = extract_solution(solution_str, prefer_first_answer)
    correct = bool(ground_truth) and grade_answer(prediction, ground_truth)
    return {
        "score": float(correct_score if correct else incorrect_score),
        "acc": correct,
        "pred": prediction if prediction is not None else "[INVALID]",
        "answer_format": answer_format if answer_format is not None else "[INVALID]",
    }
