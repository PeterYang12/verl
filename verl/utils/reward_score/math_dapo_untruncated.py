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

"""``math_dapo`` scoring without the 300-character tail window.

:mod:`verl.utils.reward_score.math_dapo` clips the response to its last 300 characters
before matching, justified by a comment noting that "the longest answer in MATH-500 has
159 characters". That bounds the length of the *answer*, not its *position*: it holds for
a post-trained model that stops once it has answered, but not for a raw pretrained one,
which emits a well-formed ``Answer:``/``\\boxed{}`` and then keeps generating unrelated
text until the answer sits outside the window and scores -1. The ``\\boxed{}`` path is
clipped harder still, to the last 100 characters, by ``is_correct_strict_box``.

Dropping the clip cannot lower a score: the window is a suffix, so the last match inside
it is also the last match of the full string. It only turns an undeserved -1 into a +1.

Widening the window is not a substitute for removing it. Measured on
DeepSeek-V4-Flash-Base (temperature 1.0, ``enable_thinking=False``, 4096 max tokens), the
answer of a correct response sits a median of 1281 characters from the end of that
response, p90 3544 and max 10335; a 300-character window captures 21.5% of them and even
a 2000-character window only 72.7%.

Measured over 128 DAPO-Math-17k prompts x 8 rollouts (1024 samples), removing the clip
lifts accuracy from 1.66% to 9.18%, recovering 77 of the 94 correct responses because
only 17 of them survived the window. The effect that matters for GRPO is on reward
variance, since a group whose rollouts all score -1 contributes no gradient: groups
carrying variance rise from 12.50% to 35.16%. On AIME 2024 (30 prompts x 8 rollouts)
accuracy goes from 1.67% to 6.25% and groups with variance from 13.33% to 30.00%.

Both answer formats are accepted, in the fallback order of ``math_dapo.verify``:
``Answer:`` first, since that is the format requested by the prompt that
``examples/data_preprocess/dapo_multiturn_w_tool.py`` builds, then ``\\boxed{}``. The
fallback carries real signal on an untuned model, which often boxes its answer without
writing the requested ``Answer:`` line: on the DAPO sample above, ``Answer:`` alone scores
6.15% and adding the ``\\boxed{}`` fallback brings it to 9.18%.

``prefer_first_answer`` additionally takes the *first* match rather than the last. A
rambling base model tends to answer and then drift, so its trailing text can contain a
later ``Answer:`` belonging to unrelated content; the first match is more often the one
that concludes the actual solution. This raises the DAPO figures to 10.84% accuracy and
37.50% of groups carrying variance, and regressed none of the 1264 samples measured here.
It is off by default because, unlike removing the clip, it is a heuristic rather than a
change that cannot lower a score: for a model that stops after answering there is only
one match and the flag is a no-op, so it is worth enabling only while the policy still
overruns its answer.

The ``{"score", "acc", "pred"}`` return shape and the +1/-1 reward convention of
``math_dapo.compute_score`` are preserved, so this is a drop-in replacement.

Wire it up with::

    reward.custom_reward_function.path=pkg://verl.utils.reward_score.math_dapo_untruncated \\
    reward.custom_reward_function.name=compute_score

and, to also take the first match::

    +reward.custom_reward_function.reward_kwargs.prefer_first_answer=True
"""

import re

# Absolute rather than relative: verl's ``file://``/plain-path loader builds the module with
# ``spec_from_file_location`` under a synthetic name, leaving it without a parent package, so a
# relative import would fail there and only work via ``pkg://``.
from verl.utils.reward_score.math_dapo import last_boxed_only_string, normalize_final_answer, remove_boxed

_ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")
_BOXED = "\\boxed{"


def _extract_boxed(solution_str, prefer_first):
    starts = [m.start() for m in re.finditer(re.escape(_BOXED), solution_str)]
    if not starts:
        return None
    boxed = last_boxed_only_string(solution_str[starts[0] :] if prefer_first else solution_str)
    if boxed is None:
        return None
    try:
        return normalize_final_answer(remove_boxed(boxed))
    except AssertionError:
        # remove_boxed asserts on the delimiters it just matched; a response cut off by the
        # token budget can leave an unterminated \boxed{ that last_boxed_only_string returns.
        return None


def extract_solution(solution_str, prefer_first_answer=False):
    """Return the model's final answer, searching all of the response.

    Prefers the ``Answer:`` line requested by the prompt and falls back to ``\\boxed{}``.
    Returns ``None`` when the response contains neither.
    """
    matches = _ANSWER_PATTERN.findall(solution_str)
    if matches:
        return normalize_final_answer(matches[0] if prefer_first_answer else matches[-1])
    return _extract_boxed(solution_str, prefer_first_answer)


def compute_score(
    data_source=None,
    solution_str="",
    ground_truth="",
    extra_info=None,
    prefer_first_answer=False,
    **kwargs,
):
    pred = extract_solution(solution_str, prefer_first_answer)
    correct = pred is not None and pred == normalize_final_answer(ground_truth)
    return {
        "score": 1.0 if correct else -1.0,
        "acc": correct,
        "pred": pred if pred is not None else "[INVALID]",
    }
