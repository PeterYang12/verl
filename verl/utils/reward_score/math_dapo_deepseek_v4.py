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

"""Full-response DAPO scoring for DeepSeek-V4-Flash Base.

The scorer combines the answer-format behavior needed by DAPO prompts with the
equivalence rule used by MILES' ``rm-type=math``:

* inspect the complete response instead of its final 300 characters;
* prefer the prompt-requested ``Answer: ...`` format;
* fall back to ``\\boxed{...}``;
* compare with MathD normalization and SymPy equivalence;
* optionally select the first answer when a Base model answers and then drifts.

Configure it with::

    reward.custom_reward_function.path=pkg://verl.utils.reward_score.math_dapo_deepseek_v4
    reward.custom_reward_function.name=compute_score

For an overrunning Base policy, additionally set::

    +reward.custom_reward_function.reward_kwargs.prefer_first_answer=True
"""

import re

from verl.utils.reward_score.math_dapo_miles import extract_answer as extract_ground_truth
from verl.utils.reward_score.math_dapo_miles import grade_answer

_ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")
_BOXED_PREFIX = "\\boxed{"


def _boxed_candidates(text: str) -> list[str]:
    candidates = []
    for match in re.finditer(re.escape(_BOXED_PREFIX), text):
        open_braces = 0
        for index in range(match.start(), len(text)):
            if text[index] == "{":
                open_braces += 1
            elif text[index] == "}":
                open_braces -= 1
                if open_braces == 0:
                    candidates.append(text[match.end() : index])
                    break
    return candidates


def extract_solution(solution_str: str, prefer_first_answer: bool = False) -> tuple[str | None, str | None]:
    """Extract an ``Answer:`` value, falling back to a boxed value."""
    answer_candidates = _ANSWER_PATTERN.findall(solution_str)
    if answer_candidates:
        index = 0 if prefer_first_answer else -1
        return answer_candidates[index].strip(), "answer"

    boxed_candidates = _boxed_candidates(solution_str)
    if boxed_candidates:
        index = 0 if prefer_first_answer else -1
        return boxed_candidates[index].strip(), "boxed"
    return None, None


def compute_score(
    data_source=None,
    solution_str="",
    ground_truth="",
    extra_info=None,
    prefer_first_answer=False,
    incorrect_score=-1.0,
    **kwargs,
):
    """Score a DeepSeek-V4 response without clipping away an earlier answer."""
    ground_truth = str(ground_truth)
    if "\\boxed" in ground_truth:
        ground_truth = extract_ground_truth(ground_truth) or ground_truth

    prediction, answer_format = extract_solution(solution_str, prefer_first_answer)
    correct = bool(ground_truth) and grade_answer(prediction, ground_truth)
    return {
        "score": 1.0 if correct else float(incorrect_score),
        "acc": correct,
        "pred": prediction if prediction is not None else "[INVALID]",
        "answer_format": answer_format if answer_format is not None else "[INVALID]",
    }
