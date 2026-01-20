from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class ValidatorResult:
    passed: bool
    reason: str


def _is_nonneg_int_text(s: str) -> bool:
    return bool(re.fullmatch(r"[0-9]+", s))


def parse_nonneg_int_text(s: str) -> Optional[int]:
    s = str(s).strip()
    if not _is_nonneg_int_text(s):
        return None
    try:
        return int(s)
    except Exception:
        return None


def canonical_nonneg_int_text(s: str) -> Optional[str]:
    n = parse_nonneg_int_text(s)
    if n is None:
        return None
    return str(int(n))


def validate_int_value_exact(output: Any, expected: Any) -> ValidatorResult:
    try:
        exp = int(expected)
    except Exception:
        return ValidatorResult(False, "expected_not_int")

    if isinstance(output, bool):
        return ValidatorResult(False, "output_is_bool")

    if isinstance(output, int):
        return ValidatorResult(output == exp, "ok" if output == exp else "int_mismatch")

    s = str(output).strip()
    n = parse_nonneg_int_text(s)
    if n is None:
        return ValidatorResult(False, "output_not_int_text")
    return ValidatorResult(n == exp, "ok" if n == exp else "int_mismatch")


def validate_int_text_canonical_exact(output: Any, expected: Any) -> ValidatorResult:
    exp_text = canonical_nonneg_int_text(str(expected))
    if exp_text is None:
        return ValidatorResult(False, "expected_not_int_text")
    out_raw = str(output).strip()
    out_text = canonical_nonneg_int_text(out_raw)
    if out_text is None:
        return ValidatorResult(False, "output_not_int_text")
    if out_raw != out_text:
        return ValidatorResult(False, "output_not_canonical_int_text")
    return ValidatorResult(out_text == exp_text, "ok" if out_text == exp_text else "text_mismatch")


def validate_json_ab_int_exact(output: Any, expected: Any) -> ValidatorResult:
    if not isinstance(expected, dict):
        return ValidatorResult(False, "expected_not_dict")
    if "a" not in expected or "b" not in expected:
        return ValidatorResult(False, "expected_missing_keys")
    try:
        exp_a = int(expected["a"])
        exp_b = int(expected["b"])
    except Exception:
        return ValidatorResult(False, "expected_values_not_int")

    try:
        obj = json.loads(str(output))
    except Exception:
        return ValidatorResult(False, "output_not_json")
    if not isinstance(obj, dict):
        return ValidatorResult(False, "output_json_not_object")
    if "a" not in obj or "b" not in obj:
        return ValidatorResult(False, "output_missing_keys")

    a = obj.get("a")
    b = obj.get("b")
    if isinstance(a, bool) or isinstance(b, bool):
        return ValidatorResult(False, "output_value_is_bool")
    if not isinstance(a, int) or not isinstance(b, int):
        return ValidatorResult(False, "output_values_not_int")
    if a != exp_a or b != exp_b:
        return ValidatorResult(False, "value_mismatch")
    return ValidatorResult(True, "ok")


def validate_list_contains_all_str(output: Any, expected: Any) -> ValidatorResult:
    if not isinstance(output, (list, tuple)):
        return ValidatorResult(False, "output_not_list")
    out: List[str] = []
    for x in output:
        if isinstance(x, str):
            out.append(x)
        else:
            return ValidatorResult(False, "output_non_str")
    out_set = set(out)

    if isinstance(expected, str):
        exp_list = [expected]
    elif isinstance(expected, (list, tuple)):
        exp_list = []
        for x in expected:
            if isinstance(x, str):
                exp_list.append(x)
            else:
                return ValidatorResult(False, "expected_non_str")
    else:
        return ValidatorResult(False, "expected_not_list_or_str")

    missing = [x for x in exp_list if x not in out_set]
    if missing:
        return ValidatorResult(False, "missing_expected_items")
    return ValidatorResult(True, "ok")


def validate_text_exact(output: Any, expected: Any) -> ValidatorResult:
    out = str(output)
    exp = str(expected)
    return ValidatorResult(out == exp, "ok" if out == exp else "text_mismatch")


ValidatorFn = Callable[[Any, Any], ValidatorResult]

VALIDATORS: Dict[str, ValidatorFn] = {
    "int_value_exact": validate_int_value_exact,
    "int_text_canonical_exact": validate_int_text_canonical_exact,
    "json_ab_int_exact": validate_json_ab_int_exact,
    "list_contains_all_str": validate_list_contains_all_str,
    "text_exact": validate_text_exact,
}


def run_validator(validator_id: str, output: Any, expected: Any) -> ValidatorResult:
    fn = VALIDATORS.get(str(validator_id))
    if fn is None:
        return ValidatorResult(False, f"unknown_validator:{validator_id}")
    return fn(output, expected)
