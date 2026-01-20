from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .act import canonical_json_dumps, sha256_hex


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


def _stable_sig(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s), flags=re.UNICODE).strip()


def _strip_edge_punct(s: str) -> str:
    return str(s).strip().strip(".,;:!?()[]{}").strip()


def _normalize_text(
    s: str,
    *,
    collapse_ws: bool,
    case_sensitive: bool,
    strip_edge_punct: bool,
) -> str:
    out = str(s)
    if strip_edge_punct:
        out = _strip_edge_punct(out)
    if collapse_ws:
        out = _collapse_ws(out)
    if not case_sensitive:
        out = out.lower()
    return out


def _parse_json_obj(output: Any) -> Tuple[Optional[Any], str]:
    if isinstance(output, (dict, list)):
        return output, ""
    s = str(output)
    try:
        return json.loads(s), ""
    except Exception:
        return None, "output_not_json"


def validate_instruction_following(output: Any, expected: Any) -> ValidatorResult:
    """
    Deterministic, spec-driven "instruction following" validator.
    Expected spec:
      {"kind": "exact", "text": "...", "collapse_ws": true, "case_sensitive": true, "strip_edge_punct": false}
      {"kind": "contains_token", "token": "...", "case_sensitive": true}
      {"kind": "int_exact", "value": 42, "strict": true}
      {"kind": "json_keys", "required_keys": [...], "types": {...}, "expected_values": {...}}
    """
    if not isinstance(expected, dict):
        return ValidatorResult(False, "expected_not_dict")
    kind = str(expected.get("kind") or "")
    if kind == "exact":
        text = str(expected.get("text") or "")
        collapse_ws = bool(expected.get("collapse_ws", True))
        case_sensitive = bool(expected.get("case_sensitive", True))
        strip_punct = bool(expected.get("strip_edge_punct", False))
        out_n = _normalize_text(str(output), collapse_ws=collapse_ws, case_sensitive=case_sensitive, strip_edge_punct=strip_punct)
        exp_n = _normalize_text(text, collapse_ws=collapse_ws, case_sensitive=case_sensitive, strip_edge_punct=strip_punct)
        return ValidatorResult(out_n == exp_n, "ok" if out_n == exp_n else "exact_mismatch")
    if kind == "contains_token":
        tok = str(expected.get("token") or "")
        case_sensitive = bool(expected.get("case_sensitive", True))
        if not tok:
            return ValidatorResult(False, "expected_missing_token")
        s = str(output)
        if not case_sensitive:
            s = s.lower()
            tok = tok.lower()
        ok = tok in s
        return ValidatorResult(ok, "ok" if ok else "token_missing")
    if kind == "int_exact":
        if "value" not in expected:
            return ValidatorResult(False, "expected_missing_value")
        try:
            exp = int(expected.get("value"))
        except Exception:
            return ValidatorResult(False, "expected_not_int")
        strict = bool(expected.get("strict", True))
        s = str(output).strip()
        if strict:
            if not re.fullmatch(r"-?[0-9]+", s):
                return ValidatorResult(False, "output_not_int_text")
            try:
                out_n = int(s)
            except Exception:
                return ValidatorResult(False, "output_not_int_text")
        else:
            m = re.search(r"-?[0-9]+", s)
            if not m:
                return ValidatorResult(False, "output_not_int_text")
            out_n = int(m.group(0))
        return ValidatorResult(out_n == exp, "ok" if out_n == exp else "int_mismatch")
    if kind == "json_keys":
        obj, err = _parse_json_obj(output)
        if obj is None:
            return ValidatorResult(False, err)
        if not isinstance(obj, dict):
            return ValidatorResult(False, "output_json_not_object")
        req = expected.get("required_keys") or []
        if not isinstance(req, list):
            return ValidatorResult(False, "expected_required_keys_not_list")
        for k in req:
            if str(k) not in obj:
                return ValidatorResult(False, "output_missing_keys")
        types = expected.get("types") or {}
        if isinstance(types, dict):
            for k, t in types.items():
                if str(k) not in obj:
                    continue
                v = obj.get(str(k))
                tt = str(t)
                if tt == "int":
                    if isinstance(v, bool) or not isinstance(v, int):
                        return ValidatorResult(False, "type_mismatch")
                elif tt == "bool":
                    if not isinstance(v, bool):
                        return ValidatorResult(False, "type_mismatch")
                elif tt == "str":
                    if not isinstance(v, str):
                        return ValidatorResult(False, "type_mismatch")
        ev = expected.get("expected_values") or {}
        if isinstance(ev, dict):
            for k, v in ev.items():
                if obj.get(str(k)) != v:
                    return ValidatorResult(False, "value_mismatch")
        return ValidatorResult(True, "ok")
    return ValidatorResult(False, f"unknown_instruction_kind:{kind}")


def validate_state_validator(output: Any, expected: Any) -> ValidatorResult:
    """
    Deterministic state validator.
    Expected spec:
      {"required_keys":[...], "expected_values":{...}, "state_sig":"<sha256>"}
    Output can be JSON text or a dict. If output is an object with a "state" key,
    validate that subtree; otherwise validate the root object.
    """
    if not isinstance(expected, dict):
        return ValidatorResult(False, "expected_not_dict")
    obj, err = _parse_json_obj(output)
    if obj is None:
        return ValidatorResult(False, err)
    state = obj.get("state") if isinstance(obj, dict) and isinstance(obj.get("state"), dict) else obj
    if not isinstance(state, dict):
        return ValidatorResult(False, "state_not_object")
    req = expected.get("required_keys") or []
    if not isinstance(req, list):
        return ValidatorResult(False, "expected_required_keys_not_list")
    for k in req:
        if str(k) not in state:
            return ValidatorResult(False, "state_missing_keys")
    ev = expected.get("expected_values") or {}
    if isinstance(ev, dict):
        for k, v in ev.items():
            if state.get(str(k)) != v:
                return ValidatorResult(False, "state_value_mismatch")
    sig = expected.get("state_sig")
    if isinstance(sig, str) and sig:
        if _stable_sig(state) != str(sig):
            return ValidatorResult(False, "state_sig_mismatch")
    return ValidatorResult(True, "ok")


def _op_add_int(a: Any, b: Any) -> int:
    if isinstance(a, bool) or isinstance(b, bool):
        return 0
    return int(a) + int(b)


def _op_make_dict_goal_plan_ab(goal_id: Any, plan: Any, a: Any, b: Any) -> Dict[str, Any]:
    gid = "" if goal_id is None else str(goal_id)
    pl = "" if plan is None else str(plan)
    if isinstance(a, bool) or isinstance(b, bool):
        return {"goal_id": gid, "plan": pl, "a": 0, "b": 0}
    return {"goal_id": gid, "plan": pl, "a": int(a), "b": int(b)}


def _op_json_canonical(obj: Any) -> str:
    return canonical_json_dumps(obj)


_PLAN_OPS: Dict[str, Tuple[int, Callable[..., Any]]] = {
    "add_int": (2, _op_add_int),
    "make_dict_goal_plan_ab": (4, _op_make_dict_goal_plan_ab),
    "json_canonical": (1, _op_json_canonical),
}


def _exec_plan_ops_v66(
    *, ops: List[Dict[str, Any]], inputs: Dict[str, Any], input_keys: List[str], return_var: str
) -> Any:
    env: Dict[str, Any] = {}
    for idx, k in enumerate(input_keys):
        env[f"in{idx}"] = inputs.get(str(k))
    for op in ops:
        fn_id = str(op.get("fn") or "")
        spec = _PLAN_OPS.get(fn_id)
        if spec is None:
            raise KeyError(f"unknown_plan_op:{fn_id}")
        arity, fn = spec
        in_vars = op.get("in") or []
        if not isinstance(in_vars, list) or len(in_vars) != int(arity):
            raise ValueError(f"arity_mismatch:{fn_id}")
        args = [env.get(str(v)) for v in in_vars]
        out = fn(*args)
        out_var = str(op.get("out") or "")
        env[out_var] = out
    return env.get(str(return_var))


def validate_plan_validator(output: Any, expected: Any) -> ValidatorResult:
    """
    Deterministic plan validator that simulates a small plan program.
    Expected spec:
      {
        "input_keys": ["goal_id","plan","a","b"],
        "inputs": {...},
        "ops": [{"fn","in":[...],"out":...}, ...],
        "return_var": "v2",
        "expected_output_text": "...",
        "required_keys": [...],
        "expected_values": {...}
      }
    Output is expected to be a JSON text string (canonical), representing an object.
    """
    if not isinstance(expected, dict):
        return ValidatorResult(False, "expected_not_dict")
    exp_out = str(expected.get("expected_output_text") or "")
    if not exp_out:
        return ValidatorResult(False, "expected_missing_expected_output_text")
    out_s = str(output)

    obj, err = _parse_json_obj(out_s)
    if obj is None:
        return ValidatorResult(False, err)
    if not isinstance(obj, dict):
        return ValidatorResult(False, "output_json_not_object")

    req = expected.get("required_keys") or []
    if isinstance(req, list):
        for k in req:
            if str(k) not in obj:
                return ValidatorResult(False, "output_missing_keys")

    ev = expected.get("expected_values") or {}
    if isinstance(ev, dict):
        for k, v in ev.items():
            if obj.get(str(k)) != v:
                return ValidatorResult(False, "value_mismatch")

    # Simulate plan.
    ops = expected.get("ops") or []
    inputs = expected.get("inputs") or {}
    input_keys = expected.get("input_keys") or []
    return_var = str(expected.get("return_var") or "")
    if not isinstance(ops, list) or not isinstance(inputs, dict) or not isinstance(input_keys, list) or not return_var:
        return ValidatorResult(False, "expected_plan_spec_invalid")
    try:
        sim_out = _exec_plan_ops_v66(ops=list(ops), inputs=dict(inputs), input_keys=[str(k) for k in input_keys], return_var=return_var)
    except KeyError as e:
        return ValidatorResult(False, str(e))
    except Exception:
        return ValidatorResult(False, "plan_exec_error")

    if str(sim_out) != str(exp_out):
        return ValidatorResult(False, "plan_expected_mismatch")
    if out_s != exp_out:
        return ValidatorResult(False, "output_text_mismatch")
    return ValidatorResult(True, "ok")


ValidatorFn = Callable[[Any, Any], ValidatorResult]

VALIDATORS: Dict[str, ValidatorFn] = {
    "int_value_exact": validate_int_value_exact,
    "int_text_canonical_exact": validate_int_text_canonical_exact,
    "json_ab_int_exact": validate_json_ab_int_exact,
    "list_contains_all_str": validate_list_contains_all_str,
    "text_exact": validate_text_exact,
    "instruction_following_validator": validate_instruction_following,
    "state_validator": validate_state_validator,
    "plan_validator": validate_plan_validator,
}


def run_validator(validator_id: str, output: Any, expected: Any) -> ValidatorResult:
    fn = VALIDATORS.get(str(validator_id))
    if fn is None:
        return ValidatorResult(False, f"unknown_validator:{validator_id}")
    return fn(output, expected)
