from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from .concepts import PRIMITIVE_OPS


_CANON_OUT_RE = re.compile(r"^v[0-9]+$", flags=re.UNICODE)


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _value_to_text(v: Any) -> str:
    if isinstance(v, (dict, list, tuple)):
        return canonical_json_dumps(v)
    if v is None:
        return ""
    return str(v)


@dataclass(frozen=True)
class CsvCandidate:
    candidate_sig: str
    ops: List[Dict[str, Any]]  # canonicalized: [{"fn","in":[...],"out":...}, ...]
    input_schema: Dict[str, str]
    output_type: str
    validator_id: str
    count: int
    contexts_distinct: int
    gain_bits_est: int
    # Examples used for PCC test vectors (already canonical and deterministic).
    examples: List[Dict[str, Any]]
    # Optional observed utility signals from the exec log (if provided).
    utility_pass: int = 0
    utility_total: int = 0
    utility_pass_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_sig": str(self.candidate_sig),
            "ops": list(self.ops),
            "input_schema": dict(self.input_schema),
            "output_type": str(self.output_type),
            "validator_id": str(self.validator_id),
            "count": int(self.count),
            "contexts_distinct": int(self.contexts_distinct),
            "gain_bits_est": int(self.gain_bits_est),
            "examples": list(self.examples),
            "utility_pass": int(self.utility_pass),
            "utility_total": int(self.utility_total),
            "utility_pass_rate": float(self.utility_pass_rate),
        }


def _infer_validator_id(*, ops: List[Dict[str, Any]], output_type: str) -> str:
    if str(output_type) == "int":
        return "int_value_exact"
    if str(output_type) == "dict":
        # Deterministic exact object equality (used for intermediate plan/state objects).
        return "json_obj_exact"
    if ops:
        fns = [str(o.get("fn") or "") for o in ops]
        if fns and fns[-1] == "json_canonical" and "make_dict_ab" in fns:
            return "json_ab_int_exact"
        if fns and fns[-1] == "json_canonical" and "make_dict_goal_plan_ab" in fns:
            return "plan_validator"
    if str(output_type) == "str":
        return "text_exact"
    return ""


def _segment_signature(ops: List[Dict[str, Any]], input_schema: Dict[str, str], output_type: str) -> str:
    body = {"ops": ops, "input_schema": input_schema, "output_type": str(output_type)}
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _alpha_rename_segment(
    segment: List[Tuple[str, List[str], str]],
    required_vars: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Canonicalize variable names (alpha-renaming) for stable candidate signatures.
    Returns (ops, varmap_original_to_canon).
    """
    varmap: Dict[str, str] = {}
    for idx, v in enumerate(required_vars):
        varmap[str(v)] = f"in{idx}"
    for idx, (_, _, out_v) in enumerate(segment):
        # Preserve canonical plan-style output names ("v0","v1",...) to avoid collisions when
        # composing mined operators in PlannerV79. For non-canonical outputs, keep alpha-renaming.
        out_s = str(out_v)
        if _CANON_OUT_RE.fullmatch(out_s):
            varmap[out_s] = out_s
        else:
            varmap[out_s] = f"v{idx}"
    ops: List[Dict[str, Any]] = []
    for fn, ins, out_v in segment:
        ops.append(
            {
                "fn": str(fn),
                "in": [str(varmap.get(str(x), str(x))) for x in ins],
                "out": str(varmap.get(str(out_v), str(out_v))),
            }
        )
    return ops, varmap


def _execute_ops_for_expected(
    *, ops: List[Dict[str, Any]], inputs_by_canon_var: Dict[str, Any]
) -> Tuple[Any, str, str]:
    env: Dict[str, Any] = dict(inputs_by_canon_var)
    out_var = ""
    for op in ops:
        fn_id = str(op.get("fn") or "")
        spec_fn = PRIMITIVE_OPS.get(fn_id)
        if spec_fn is None:
            raise KeyError(f"unknown_primitive:{fn_id}")
        spec, fn = spec_fn
        in_vars = op.get("in", [])
        if not isinstance(in_vars, list):
            in_vars = []
        vals = [env.get(str(v)) for v in in_vars]
        if int(spec.arity) != int(len(vals)):
            raise ValueError(f"arity_mismatch:{fn_id}:{spec.arity}:{len(vals)}")
        out = fn(*vals) if int(spec.arity) > 1 else fn(vals[0])
        out_var = str(op.get("out") or "")
        env[out_var] = out
    out_val = env.get(out_var)
    out_text = _value_to_text(out_val)
    return out_val, out_text, sha256_hex(out_text.encode("utf-8"))


def mine_csv_candidates(
    exec_jsonl_path: str,
    *,
    min_ops: int = 2,
    max_ops: int = 6,
    bits_per_op: int = 128,
    overhead_bits: int = 1024,
    max_examples_per_candidate: int = 10,
) -> List[CsvCandidate]:
    """
    Deterministic miner for primitive-op subsegments.

    Input: csv_exec.jsonl (records with events: GET_INPUT + PRIMITIVE + RETURN).
    Output: ranked candidates with inferred interface + examples for PCC.
    """
    min_ops = max(1, int(min_ops))
    max_ops = max(min_ops, int(max_ops))
    bits_per_op = max(1, int(bits_per_op))
    overhead_bits = max(0, int(overhead_bits))

    agg: Dict[str, Dict[str, Any]] = {}

    for rec in _read_jsonl(exec_jsonl_path):
        ctx_sig = str(rec.get("ctx_sig") or "")
        inputs = rec.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}
        utility_passed = rec.get("utility_passed", None)
        utility_is_bool = isinstance(utility_passed, bool)
        events = rec.get("events", [])
        if not isinstance(events, list):
            continue

        var_origin: Dict[str, str] = {}
        prims: List[Tuple[str, List[str], str]] = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            t = str(ev.get("t") or "")
            # v61: accept real engine traces (execute_concept_csv) which emit t:"INS" with
            # op:"CSV_*" plus normalized fields.
            if t == "INS":
                op = str(ev.get("op") or "")
                if op == "CSV_GET_INPUT":
                    name = str(ev.get("name") or "")
                    out = str(ev.get("out") or "")
                    if out and name:
                        var_origin[out] = name
                elif op == "CSV_PRIMITIVE":
                    fn_id = str(ev.get("fn") or "")
                    ins = ev.get("in", [])
                    if not isinstance(ins, list):
                        ins = []
                    out = str(ev.get("out") or "")
                    if fn_id and out:
                        prims.append((fn_id, [str(x) for x in ins], out))
                else:
                    # CSV_CALL / CSV_RETURN are ignored for primitive-only mining in v61.
                    pass
                continue

            if t in {"GET_INPUT", "I"}:
                name = str(ev.get("name") or "")
                out = str(ev.get("out") or "")
                if out and name:
                    var_origin[out] = name
            elif t in {"PRIMITIVE", "P"}:
                fn_id = str(ev.get("fn") or "")
                ins = ev.get("in", [])
                if not isinstance(ins, list):
                    ins = []
                out = str(ev.get("out") or "")
                if fn_id and out:
                    prims.append((fn_id, [str(x) for x in ins], out))

        if not prims:
            continue

        for s in range(len(prims)):
            for L in range(min_ops, max_ops + 1):
                if s + L > len(prims):
                    break
                segment = prims[s : s + L]

                required_vars: List[str] = []
                defined: set = set()
                var_types: Dict[str, str] = {}
                ok = True
                for fn_id, ins, out in segment:
                    spec_fn = PRIMITIVE_OPS.get(fn_id)
                    if spec_fn is None:
                        ok = False
                        break
                    spec = spec_fn[0]
                    if int(spec.arity) != int(len(ins)):
                        ok = False
                        break
                    for idx, v in enumerate(ins):
                        want_t = str(spec.input_types[idx])
                        if v not in defined:
                            if v not in required_vars:
                                required_vars.append(v)
                        prev_t = var_types.get(v)
                        if prev_t and prev_t != want_t:
                            ok = False
                            break
                        var_types[v] = want_t
                    if not ok:
                        break
                    var_types[str(out)] = str(spec.output_type)
                    defined.add(str(out))
                if not ok:
                    continue

                out_var = str(segment[-1][2])
                output_type = str(var_types.get(out_var) or "")
                if not output_type:
                    continue

                # Build input schema via origin names if present; reject if missing values.
                input_schema: Dict[str, str] = {}
                input_bindings: List[Tuple[str, str]] = []  # (origin_name, canon_var)
                missing = False
                for idx, v in enumerate(required_vars):
                    origin = str(var_origin.get(str(v)) or str(v))
                    if origin not in inputs:
                        missing = True
                        break
                    input_schema[origin] = str(var_types.get(str(v)) or "str")
                    input_bindings.append((origin, f"in{idx}"))
                if missing:
                    continue

                ops, varmap = _alpha_rename_segment(segment, required_vars)
                validator_id = _infer_validator_id(ops=ops, output_type=output_type)
                if not validator_id:
                    continue

                sig = _segment_signature(ops, input_schema, output_type)
                st = agg.get(sig)
                if st is None:
                    st = {
                        "ops": ops,
                        "input_schema": dict(input_schema),
                        "output_type": str(output_type),
                        "validator_id": str(validator_id),
                        "count": 0,
                        "ctx": set(),
                        "examples": [],
                        "utility_pass": 0,
                        "utility_total": 0,
                    }
                    agg[sig] = st
                st["count"] = int(st["count"]) + 1
                st["ctx"].add(ctx_sig)
                if utility_is_bool:
                    st["utility_total"] = int(st["utility_total"]) + 1
                    if bool(utility_passed):
                        st["utility_pass"] = int(st["utility_pass"]) + 1

                # Example for PCC test vectors.
                if len(st["examples"]) < int(max_examples_per_candidate):
                    inps_by_canon: Dict[str, Any] = {}
                    for origin, canon_var in input_bindings:
                        inps_by_canon[str(canon_var)] = inputs.get(origin)
                    out_val, out_text, out_sig = _execute_ops_for_expected(
                        ops=ops, inputs_by_canon_var=inps_by_canon
                    )
                    # Deterministic expected: prefer structured expected when possible.
                    expected: Any = out_val
                    if str(output_type) == "str" and validator_id == "json_ab_int_exact":
                        # If the output is JSON text, expected should be the dict form when possible.
                        # (We can reconstruct via executing primitives before json_canonical.)
                        try:
                            expected = json.loads(out_text)
                        except Exception:
                            expected = out_text
                    st["examples"].append(
                        {
                            "ctx_sig": str(ctx_sig),
                            "inputs": {origin: inputs.get(origin) for origin, _ in input_bindings},
                            "expected": expected,
                            "expected_output_text": str(out_text),
                            "expected_sig": str(out_sig),
                        }
                    )

    out: List[CsvCandidate] = []
    for sig, st in agg.items():
        count = int(st["count"])
        contexts_distinct = int(len(st["ctx"]))
        ops = list(st["ops"])
        util_total = int(st.get("utility_total", 0) or 0)
        util_pass = int(st.get("utility_pass", 0) or 0)
        util_rate = float(util_pass / util_total) if util_total > 0 else 0.0
        # Utility-as-law signal (shadow): add a small deterministic bonus for higher utility.
        util_bonus_bits = int(round(util_rate * 256.0))
        gain_bits_est = int(count) * int(len(ops)) * int(bits_per_op) - int(overhead_bits) + int(util_bonus_bits)
        out.append(
            CsvCandidate(
                candidate_sig=str(sig),
                ops=ops,
                input_schema=dict(st["input_schema"]),
                output_type=str(st["output_type"]),
                validator_id=str(st["validator_id"]),
                count=count,
                contexts_distinct=contexts_distinct,
                gain_bits_est=gain_bits_est,
                examples=list(st["examples"]),
                utility_pass=util_pass,
                utility_total=util_total,
                utility_pass_rate=float(util_rate),
            )
        )

    out.sort(
        key=lambda c: (
            -int(c.gain_bits_est),
            -int(c.contexts_distinct),
            str(c.candidate_sig),
        )
    )
    return out


def materialize_concept_act_from_candidate(
    cand: CsvCandidate,
    *,
    step: int,
    store_content_hash_excluding_semantic: str,
    title: str,
    overhead_bits: int = 1024,
    meta: Optional[Dict[str, Any]] = None,
) -> Act:
    """
    Turn a mined CsvCandidate into a concept_csv Act (without PCC certificate).
    """
    # Input schema order must match candidate alpha-renaming (deterministic).
    # (Do not sort: it can break in0/in1 bindings.)
    in_keys = list(cand.input_schema.keys())
    program: List[Instruction] = []
    for idx, name in enumerate(in_keys):
        program.append(Instruction("CSV_GET_INPUT", {"name": str(name), "out": f"in{idx}"}))
    for op in cand.ops:
        program.append(
            Instruction(
                "CSV_PRIMITIVE",
                {
                    "fn": str(op.get("fn") or ""),
                    "in": list(op.get("in") or []),
                    "out": str(op.get("out") or ""),
                },
            )
        )
    if cand.ops:
        program.append(Instruction("CSV_RETURN", {"var": str(cand.ops[-1].get("out") or "")}))

    interface = {
        "input_schema": dict(cand.input_schema),
        "output_schema": {"value": str(cand.output_type)},
        "validator_id": str(cand.validator_id),
    }
    ev = {
        "name": "concept_csv_mined_v60",
        "interface": interface,
        "meta": {
            "title": str(title),
            "builder": "csv_miner_v60",
            "trained_on_store_content_hash": str(store_content_hash_excluding_semantic),
            "candidate_sig": str(cand.candidate_sig),
            "gain_bits_est": int(cand.gain_bits_est),
            "contexts_distinct": int(cand.contexts_distinct),
            "count": int(cand.count),
            **(dict(meta or {})),
        },
    }
    act_body_for_id = {
        "kind": "concept_csv",
        "version": 1,
        "match": {},
        "program": [ins.to_dict() for ins in program],
        "evidence": ev,
        "deps": [],
        "active": True,
    }
    act_id = f"act_concept_csv_{sha256_hex(canonical_json_dumps(act_body_for_id).encode('utf-8'))[:12]}"
    return Act(
        id=act_id,
        version=1,
        created_at=deterministic_iso(step=int(step)),
        kind="concept_csv",
        match={},
        program=program,
        evidence=ev,
        cost={"overhead_bits": int(overhead_bits)},
        deps=[],
        active=True,
    )
