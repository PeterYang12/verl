# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 RadixArk contributors
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

"""MILES-style full-response math scoring for DAPO.

MILES' DeepSeek/JoyAI Flash recipe uses ``rm-type=math`` rather than its
``rm-type=dapo`` implementation.  The former searches the complete response for
the last ``\\boxed{...}`` answer and accepts either MathD-normalized equality or
SymPy equality.  In particular, it does not apply ``solution_str[-300:]``.

This module follows that correctness rule while returning verl's
``{"score", "acc", "pred"}`` shape.  The default incorrect score is ``-1`` to
remain compatible with ``math_dapo``.  Set ``incorrect_score=0`` for MILES'
native binary reward convention.

Configure it with::

    reward.custom_reward_function.path=pkg://verl.utils.reward_score.math_dapo_miles
    reward.custom_reward_function.name=compute_score

For the exact MILES reward range, additionally set::

    +reward.custom_reward_function.reward_kwargs.incorrect_score=0.0
"""

import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

from verl.utils.reward_score.math_reward import strip_string

_TUPLE_CHARS = "()[]"
_BAD_SUBSTRINGS = ("^{", "^(")
_BAD_REGEXES = (r"\^[0-9]+\^", r"\^[0-9][0-9]+")


def _last_boxed_only_string(text: str) -> str | None:
    """Return the last balanced ``\\boxed{...}`` expression."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None

    left_braces = 0
    saw_left_brace = False
    for index in range(start, len(text)):
        if text[index] == "{":
            left_braces += 1
            saw_left_brace = True
        elif text[index] == "}":
            left_braces -= 1
            if saw_left_brace and left_braces == 0:
                return text[start : index + 1]
    return None


def _remove_boxed(boxed: str | None) -> str | None:
    if boxed is None or not boxed.startswith("\\boxed{") or not boxed.endswith("}"):
        return None
    return boxed[len("\\boxed{") : -1]


def extract_answer(text: str) -> str | None:
    """Extract the final boxed answer from the complete response."""
    return _remove_boxed(_last_boxed_only_string(text))


def _mathd_normalize(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    match = re.fullmatch(r"\\text\{(.+?)\}", answer)
    if match is not None:
        answer = match.group(1).strip()
    try:
        return strip_string(answer)
    except Exception:
        return answer


def _strip_formatted_commas(expression: str) -> str:
    pattern = re.compile(r"(\d),(\d\d\d)($|\D)")
    while True:
        updated = pattern.sub(r"\1\2\3", expression)
        if updated == expression:
            return updated
        expression = updated


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _is_int(value: str | float) -> bool:
    try:
        number = float(_strip_formatted_commas(str(value)))
        return abs(number - int(round(number))) <= 1e-7
    except (TypeError, ValueError):
        return False


def _is_fraction(expression: str) -> bool:
    return bool(re.fullmatch(r"-?[0-9]+.?/0*[1-9][0-9]*.?", expression))


def _normalize_for_sympy(expression: str | None) -> str | None:
    if expression is None:
        return None

    match = re.fullmatch(r"\\text\{(.+?)\}", expression)
    if match is not None:
        expression = match.group(1)

    expression = expression.replace("\\%", "%").replace("\\$", "$")
    expression = expression.replace("$", "").replace("%", "")
    expression = expression.replace(" or ", " , ").replace(" and ", " , ")
    expression = expression.replace("million", "*10^6")
    expression = expression.replace("billion", "*10^9")
    expression = expression.replace("trillion", "*10^12")

    for unit in (
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ):
        expression = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expression)
    expression = re.sub(r"\^ *\\circ", "", expression)

    if len(expression) > 1 and expression[0] == "{" and expression[-1] == "}":
        expression = expression[1:-1]
    expression = re.sub(r",\\! *", "", expression)

    if _is_float(expression) and _is_int(expression):
        expression = str(int(round(float(expression))))

    if "\\" in expression:
        try:
            expression = expression.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
            expression = expression.replace("\\frac", " \\frac")
            expression = latex2text.LatexNodes2Text().latex_to_text(expression)
            expression = (
                expression.replace("√", "sqrt")
                .replace("π", "pi")
                .replace("∞", "inf")
                .replace("∪", "U")
                .replace("·", "*")
                .replace("×", "*")
            )
        except Exception:
            pass

    expression = re.sub(r"- *", "-", expression)
    expression = re.sub(r"([0-9]) +([0-9])", r"\1+\2", expression)
    expression = expression.replace(" ", "").replace("{", "").replace("}", "").lower()

    if _is_int(expression):
        expression = str(int(round(float(_strip_formatted_commas(expression)))))
    return expression


def _split_tuple(expression: str) -> list[str]:
    expression = _strip_formatted_commas(expression)
    if not expression:
        return []
    if (
        len(expression) > 2
        and expression[0] in _TUPLE_CHARS
        and expression[-1] in _TUPLE_CHARS
        and all(char not in expression[1:-1] for char in _TUPLE_CHARS)
    ):
        return [element.strip() for element in expression[1:-1].split(",")]
    return [expression]


def _allow_sympy_eval(expression: str) -> bool:
    letters = set(expression.replace("sqrt", "").replace("frac", ""))
    if len([char for char in letters if char.isalpha()]) > 2:
        return False
    return not any(item in expression for item in _BAD_SUBSTRINGS) and not any(
        re.search(pattern, expression) for pattern in _BAD_REGEXES
    )


def _equal_under_sympy(ground_truth: str, prediction: str) -> bool:
    try:
        difference = f"({ground_truth})-({prediction})"
        if not _allow_sympy_eval(difference):
            return False
        parsed = sympy_parser.parse_expr(
            difference.replace("^", "**"),
            transformations=(
                sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)
            ),
        )
        return sympy.simplify(parsed) == 0
    except Exception:
        return False


def _grade_answer_sympy(prediction: str, ground_truth: str) -> bool:
    normalized_gt = _normalize_for_sympy(ground_truth)
    normalized_pred = _normalize_for_sympy(prediction)
    if normalized_gt is None or normalized_pred is None:
        return False
    if normalized_gt == normalized_pred:
        return True
    if not normalized_pred:
        return False

    gt_elements = _split_tuple(normalized_gt)
    pred_elements = _split_tuple(normalized_pred)
    if len(gt_elements) != len(pred_elements):
        return False
    if len(gt_elements) > 1 and (
        normalized_gt[0] != normalized_pred[0] or normalized_gt[-1] != normalized_pred[-1]
    ):
        return False

    for gt_element, pred_element in zip(gt_elements, pred_elements, strict=True):
        # As in MILES, do not use SymPy to accept an unreduced fraction when
        # the reference is also expressed as a fraction.
        if _is_fraction(gt_element) and _is_fraction(pred_element):
            if gt_element != pred_element:
                return False
            continue
        # Match MILES' strict handling of integer answers, which is especially
        # relevant to DAPO-Math-17k and AIME.
        if _is_int(gt_element) != _is_int(pred_element):
            return False
        if not _equal_under_sympy(gt_element, pred_element):
            return False
    return True


def grade_answer(prediction: str | None, ground_truth: str) -> bool:
    """Apply MILES' MathD-or-SymPy equivalence rule."""
    if prediction is None:
        return False
    mathd_gt = _mathd_normalize(ground_truth)
    mathd_pred = _mathd_normalize(prediction)
    return mathd_gt == mathd_pred or _grade_answer_sympy(prediction, ground_truth)


def compute_score(
    data_source=None,
    solution_str="",
    ground_truth="",
    extra_info=None,
    incorrect_score=-1.0,
    **kwargs,
):
    """Score the final boxed answer without truncating the response."""
    ground_truth = str(ground_truth)
    if "\\boxed" in ground_truth:
        ground_truth = extract_answer(ground_truth) or ground_truth

    prediction = extract_answer(solution_str)
    correct = bool(ground_truth) and grade_answer(prediction, ground_truth)
    return {
        "score": 1.0 if correct else float(incorrect_score),
        "acc": correct,
        "pred": prediction if prediction is not None else "[INVALID]",
    }
