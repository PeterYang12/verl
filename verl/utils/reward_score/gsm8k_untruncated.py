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

"""GSM8K scoring for base checkpoints that answer correctly and then keep generating.

:mod:`verl.utils.reward_score.gsm8k` reads only the last 300 characters and only accepts
``#### N``. Both assumptions hold for a post-trained model, and both break on a raw
pretrained checkpoint, which answers early, uses ``\\boxed{}`` about as often as ``####``,
and then continues with unrelated text that fills the tail window.

Measured on DeepSeek-V4-Flash-Base with an unmodified prompt and rollout (60 prompts x 8
rollouts, temperature 1.0, ``enable_thinking=False``): the default scorer saw 2.3% correct
and left only 16.7% of GRPO groups with any reward variance, while these rules saw 50.4%
and 96.7%.

Wire it up with::

    reward.custom_reward_function.path=pkg://verl.utils.reward_score.gsm8k_untruncated \\
    reward.custom_reward_function.name=compute_score
"""

import re

_HASH_ANSWER = re.compile(r"#### (\-?[0-9\.\,]+)")
_BOXED_ANSWER = re.compile(r"\\boxed\{([^}]*)\}")


def _normalize(value):
    return str(value).replace(",", "").replace("$", "").replace(" ", "").strip().rstrip(".")


def extract_solution(solution_str):
    """Return the model's final answer, searching the whole response in either format."""
    for pattern in (_HASH_ANSWER, _BOXED_ANSWER):
        matches = pattern.findall(solution_str)
        if matches:
            return _normalize(matches[-1])
    return None


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    answer = extract_solution(solution_str)
    if answer is None:
        return 0.0
    return 1.0 if answer == _normalize(ground_truth) else 0.0
