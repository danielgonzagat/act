from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex


BOOL_DOMAIN_VALUES_V86: List[str] = ["0", "1"]


def _stable_hash(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def logic_hash(expr_canon: Dict[str, Any]) -> str:
    return _stable_hash(expr_canon)


def _require_dict(expr: Any) -> Dict[str, Any]:
    if not isinstance(expr, dict):
        raise ValueError("expr_not_dict")
    return expr


def _require_str(x: Any, *, field: str) -> str:
    s = str(x or "")
    if not s:
        raise ValueError(f"missing_{field}")
    return s


def _canon_comm_args(args: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Sort by hash(expr_canon) for deterministic commutative canonicalization.
    return sorted(list(args), key=lambda a: str(logic_hash(a)))


def logic_canon(expr: Any) -> Dict[str, Any]:
    """
    Canonicalize a propositional logic AST.

    Supported nodes (dict form):
      - {"op":"var","name":<str>}
      - {"op":"not","x":<expr>}
      - {"op":"and","args":[<expr>,...]}
      - {"op":"or","args":[<expr>,...]}
      - {"op":"xor","args":[<expr>,...]}
      - {"op":"implies","a":<expr>,"b":<expr>}
      - {"op":"iff","args":[<expr>,<expr>]}   # commutative (binary)
    """
    d = _require_dict(expr)
    op = str(d.get("op") or "")
    if not op:
        raise ValueError("missing_op")

    if op == "var":
        name = _require_str(d.get("name"), field="var_name")
        return {"op": "var", "name": str(name)}

    if op == "not":
        x = logic_canon(d.get("x"))
        return {"op": "not", "x": x}

    if op in {"and", "or", "xor"}:
        args = d.get("args")
        if not isinstance(args, list):
            raise ValueError("bad_args")
        if len(args) < 2:
            raise ValueError("bad_args_len")
        args_c = [logic_canon(a) for a in args]
        return {"op": str(op), "args": _canon_comm_args(args_c)}

    if op == "implies":
        a = logic_canon(d.get("a"))
        b = logic_canon(d.get("b"))
        return {"op": "implies", "a": a, "b": b}

    if op == "iff":
        args = d.get("args")
        if args is None:
            # Allow legacy {"a":..., "b":...} input.
            args = [d.get("a"), d.get("b")]
        if not isinstance(args, list) or len(args) != 2:
            raise ValueError("bad_iff_args")
        a = logic_canon(args[0])
        b = logic_canon(args[1])
        return {"op": "iff", "args": _canon_comm_args([a, b])}

    raise ValueError(f"unknown_op:{op}")


def _parse_domain01(v: Any) -> bool:
    s = str(v or "")
    if s == "0":
        return False
    if s == "1":
        return True
    raise ValueError(f"bad_domain_value:{s}")


def _bool_to_domain01(v: bool) -> str:
    return "1" if bool(v) else "0"


def logic_eval(expr_canon: Dict[str, Any], assignment: Dict[str, Any]) -> bool:
    """
    Evaluate a canonicalized expr under an assignment where each var value is "0" or "1".
    """
    d = _require_dict(expr_canon)
    op = str(d.get("op") or "")
    if op == "var":
        name = _require_str(d.get("name"), field="var_name")
        if name not in assignment:
            raise ValueError(f"missing_assignment:{name}")
        return _parse_domain01(assignment.get(name))
    if op == "not":
        return not bool(logic_eval(d["x"], assignment))
    if op == "and":
        args = d.get("args")
        if not isinstance(args, list):
            raise ValueError("bad_args")
        return all(bool(logic_eval(a, assignment)) for a in args)
    if op == "or":
        args = d.get("args")
        if not isinstance(args, list):
            raise ValueError("bad_args")
        return any(bool(logic_eval(a, assignment)) for a in args)
    if op == "xor":
        args = d.get("args")
        if not isinstance(args, list):
            raise ValueError("bad_args")
        acc = 0
        for a in args:
            acc ^= 1 if bool(logic_eval(a, assignment)) else 0
        return bool(acc)
    if op == "implies":
        a = bool(logic_eval(d["a"], assignment))
        b = bool(logic_eval(d["b"], assignment))
        return (not a) or b
    if op == "iff":
        args = d.get("args")
        if not isinstance(args, list) or len(args) != 2:
            raise ValueError("bad_iff_args")
        a = bool(logic_eval(args[0], assignment))
        b = bool(logic_eval(args[1], assignment))
        return bool(a == b)
    raise ValueError(f"unknown_op:{op}")


def _norm_domain_values(domain_values: Sequence[str]) -> List[str]:
    vals = [str(v) for v in domain_values if str(v)]
    # Deterministic: unique + sort.
    vals = sorted(set(vals), key=str)
    if vals != ["0", "1"]:
        raise ValueError("bad_domain_values")
    return list(vals)


def enumerate_assignments(*, vars: Sequence[str], domain_values: Sequence[str]) -> List[Dict[str, str]]:
    """
    Deterministic truth-table assignment order:
      - vars sorted lexicographically
      - assignments in binary order (first var is most-significant)
    """
    vs = sorted(set(str(v) for v in vars if str(v)), key=str)
    dv = _norm_domain_values(domain_values)
    assignments: List[Dict[str, str]] = [{}]
    for v in vs:
        nxt: List[Dict[str, str]] = []
        for a in assignments:
            for d in dv:
                a2 = dict(a)
                a2[str(v)] = str(d)
                nxt.append(a2)
        assignments = nxt
    return list(assignments)


def render_output01(*, out01: str, render: Dict[str, Any]) -> str:
    rk = str(render.get("kind") or "")
    if rk == "raw01":
        return str(out01)
    if rk == "prefix01":
        prefix = str(render.get("prefix") or "")
        return f"{prefix}{out01}"
    raise ValueError("unknown_render_kind")


def truth_table(
    *,
    vars: Sequence[str],
    domain_values: Sequence[str],
    expr: Dict[str, Any],
    render: Dict[str, Any],
) -> List[Dict[str, Any]]:
    expr_c = logic_canon(expr)
    vs = sorted(set(str(v) for v in vars if str(v)), key=str)
    dv = _norm_domain_values(domain_values)
    tt: List[Dict[str, Any]] = []
    for a in enumerate_assignments(vars=vs, domain_values=dv):
        out_b = logic_eval(expr_c, a)
        out01 = _bool_to_domain01(out_b)
        out_text = render_output01(out01=out01, render=dict(render))
        tt.append({"inputs": {str(k): str(a.get(k)) for k in vs}, "out": str(out_text)})
    return tt


def truth_table_sha256(*, tt: Sequence[Dict[str, Any]]) -> str:
    return sha256_hex(canonical_json_dumps(list(tt)).encode("utf-8"))


def ensure_bool_primitives_registered() -> None:
    """
    Deterministically extend PRIMITIVE_OPS with bool operations over domain strings "0"/"1".
    This is additive and idempotent.
    """
    from .concepts import PRIMITIVE_OPS, PrimitiveOpSpec

    def _to_bit(x: Any) -> int:
        s = str(x or "").strip()
        if s == "0":
            return 0
        if s == "1":
            return 1
        return 0

    def _bit_to_str(b: int) -> str:
        return "1" if int(b) != 0 else "0"

    def _bool_not01(x: Any) -> str:
        return _bit_to_str(1 - _to_bit(x))

    def _bool_and01(a: Any, b: Any) -> str:
        return _bit_to_str(_to_bit(a) & _to_bit(b))

    def _bool_or01(a: Any, b: Any) -> str:
        return _bit_to_str(_to_bit(a) | _to_bit(b))

    def _bool_xor01(a: Any, b: Any) -> str:
        return _bit_to_str(_to_bit(a) ^ _to_bit(b))

    def _bool_implies01(a: Any, b: Any) -> str:
        aa = _to_bit(a)
        bb = _to_bit(b)
        return _bit_to_str((1 - aa) | bb)

    def _bool_iff01(a: Any, b: Any) -> str:
        aa = _to_bit(a)
        bb = _to_bit(b)
        return _bit_to_str(1 if aa == bb else 0)

    new_ops: Dict[str, Tuple[PrimitiveOpSpec, Any]] = {
        "bool_not01": (PrimitiveOpSpec("bool_not01", 1, ("str",), "str"), _bool_not01),
        "bool_and01": (PrimitiveOpSpec("bool_and01", 2, ("str", "str"), "str"), _bool_and01),
        "bool_or01": (PrimitiveOpSpec("bool_or01", 2, ("str", "str"), "str"), _bool_or01),
        "bool_xor01": (PrimitiveOpSpec("bool_xor01", 2, ("str", "str"), "str"), _bool_xor01),
        "bool_implies01": (PrimitiveOpSpec("bool_implies01", 2, ("str", "str"), "str"), _bool_implies01),
        "bool_iff01": (PrimitiveOpSpec("bool_iff01", 2, ("str", "str"), "str"), _bool_iff01),
    }

    for op_id in sorted(new_ops.keys(), key=str):
        spec, fn = new_ops[op_id]
        existing = PRIMITIVE_OPS.get(op_id)
        if existing is None:
            PRIMITIVE_OPS[op_id] = (spec, fn)
            continue
        ex_spec = existing[0]
        if (
            str(getattr(ex_spec, "op_id", "")) != str(spec.op_id)
            or int(getattr(ex_spec, "arity", 0)) != int(spec.arity)
            or tuple(getattr(ex_spec, "input_types", ())) != tuple(spec.input_types)
            or str(getattr(ex_spec, "output_type", "")) != str(spec.output_type)
        ):
            raise ValueError(f"primitive_conflict:{op_id}")

