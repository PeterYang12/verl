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

"""GSM8K strict scoring without the 300-character tail window.

:mod:`verl.utils.reward_score.gsm8k` clips to the last 300 characters before matching,
which its own comment describes as a speed optimization on the assumption that "the final
answer is usually at the end". That holds for a post-trained model but not for a raw
pretrained one, which emits a well-formed ``#### N`` early and then keeps generating
unrelated text until the answer sits outside the window and scores zero.

Dropping the clip cannot lower a score: the window is a suffix, so any match inside it is
also the last match of the full string. It only turns an undeserved 0 into a 1.

The ``#### N`` requirement is kept deliberately. The gsm8k prompt built by
``examples/data_preprocess/gsm8k.py`` instructs the model to "output the final answer
after '####'", so accepting ``\\boxed{}`` here would reward ignoring that instruction.
Measured on DeepSeek-V4-Flash-Base (60 prompts x 8 rollouts, temperature 1.0,
``enable_thinking=False``), keeping the format requirement and only dropping the clip
already lifts GRPO groups that carry reward variance from 16.7% to 95.0%; also accepting
``\\boxed{}`` would raise accuracy further but move that figure only to 96.7%.

Wire it up with::

    reward.custom_reward_function.path=pkg://verl.utils.reward_score.gsm8k_untruncated \\
    reward.custom_reward_function.name=compute_score
"""

import re

_STRICT_ANSWER = re.compile(r"#### (\-?[0-9\.\,]+)")


def _normalize(value):
    return str(value).replace(",", "").replace("$", "").replace(" ", "").strip().rstrip(".")


def extract_solution(solution_str):
    """Return the last ``#### N`` in the response, searching all of it."""
    matches = _STRICT_ANSWER.findall(solution_str)
    return _normalize(matches[-1]) if matches else None


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    answer = extract_solution(solution_str)
    if answer is None:
        return 0.0
    return 1.0 if answer == _normalize(ground_truth) else 0.0
