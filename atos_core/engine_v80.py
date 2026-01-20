from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from .act import Act, canonical_json_dumps, sha256_hex
from .concepts import PRIMITIVE_OPS
from .ethics import fail_closed_text, validate_before_execute
from .match_v80 import is_act_allowed_for_goal_kind
from .uncertainty import guard_text_uncertainty
from .validators import ValidatorResult, run_validator


class EngineV80:
    """
    V80: match-enforced execution for concept_csv ACTs (top-level and nested CSV_CALL).
    """

    def __init__(self, store, *, seed: int):
        self.store = store
        self.seed = int(seed)

    def execute_concept_csv(
        self,
        *,
        concept_act_id: str,
        inputs: Dict[str, Any],
        goal_kind: str,
        expected: Any = None,
        step: int = 0,
        max_depth: int = 8,
        max_events: int = 512,
        validate_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a first-class concept_csv ACT as an explicit subgraph (CSV-MVP semantics).
        V80 addition: enforce act.match.goal_kinds for the given goal_kind on every call.
        """

        events: List[Dict[str, Any]] = []
        concept_calls: List[Dict[str, Any]] = []

        def _hash_obj(obj: Any) -> str:
            try:
                return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))
            except Exception:
                return sha256_hex(str(obj).encode("utf-8"))

        def _value_to_text(v: Any) -> str:
            if isinstance(v, (dict, list, tuple)):
                return canonical_json_dumps(v)
            if v is None:
                return ""
            return str(v)

        def _iface_signature(iface: Dict[str, Any]) -> str:
            body = {
                "in": iface.get("input_schema", {}),
                "out": iface.get("output_schema", {}),
                "validator_id": iface.get("validator_id", ""),
            }
            return _hash_obj(body)

        def _get_concept(concept_id: str) -> Optional[Act]:
            act: Optional[Act] = None
            try:
                getter = getattr(self.store, "get_concept_act", None)
                if callable(getter):
                    act = getter(str(concept_id))
                else:
                    act = self.store.get(str(concept_id))
            except Exception:
                act = None
            if act is None or (not bool(getattr(act, "active", True))):
                return None
            if str(getattr(act, "kind", "")) != "concept_csv":
                return None
            return act

        def _type_ok(val: Any, want: str) -> bool:
            w = str(want or "")
            if not w:
                return True
            if w == "str":
                return isinstance(val, str)
            if w == "int":
                return isinstance(val, int) and (not isinstance(val, bool))
            if w == "dict":
                return isinstance(val, dict)
            if w == "list":
                return isinstance(val, list)
            return True

        def _execute(
            concept_id: str,
            inps: Dict[str, Any],
            depth: int,
            *,
            expected_for_validator: Any,
            validate_output: bool,
        ) -> Tuple[Any, Dict[str, Any]]:
            if int(depth) > int(max_depth):
                return None, {"ok": False, "reason": "max_depth", "concept_id": str(concept_id), "goal_kind": str(goal_kind)}

            act = _get_concept(concept_id)
            if act is None:
                return None, {"ok": False, "reason": "concept_not_found", "concept_id": str(concept_id), "goal_kind": str(goal_kind)}

            ev = act.evidence if isinstance(act.evidence, dict) else {}
            iface = ev.get("interface")
            if not isinstance(iface, dict):
                return None, {"ok": False, "reason": "missing_interface", "concept_id": str(concept_id), "goal_kind": str(goal_kind)}

            in_schema = iface.get("input_schema")
            out_schema = iface.get("output_schema")
            validator_id = str(iface.get("validator_id") or "")
            if not isinstance(in_schema, dict) or not isinstance(out_schema, dict):
                return None, {"ok": False, "reason": "bad_interface_schema", "concept_id": str(concept_id), "goal_kind": str(goal_kind)}

            iface_sig = _iface_signature(iface)
            prog_sha256 = _hash_obj([ins.to_dict() for ins in (act.program or [])])
            concept_sig = _hash_obj(
                {
                    "concept_id": str(concept_id),
                    "interface_sig": str(iface_sig),
                    "program_sha256": str(prog_sha256),
                }
            )
            try:
                bindings_snapshot = copy.deepcopy(inps)
            except Exception:
                bindings_snapshot = dict(inps)
            if not isinstance(bindings_snapshot, dict):
                bindings_snapshot = {}
            bindings_sig = _hash_obj(bindings_snapshot)

            call_rec: Dict[str, Any] = {
                "concept_id": str(concept_id),
                "concept_sig": str(concept_sig),
                "interface_sig": str(iface_sig),
                "program_sha256": str(prog_sha256),
                "bindings": bindings_snapshot,
                "bindings_sig": str(bindings_sig),
                "call_depth": int(depth),
                "return": {},
                "return_sig": "",
                "ok": False,
                "cost": float(len(act.program or [])),
                "evidence_refs": [{"kind": "concept_act", "act_id": str(concept_id)}],
                "blocked": False,
                "blocked_reason": "",
                "goal_kind": str(goal_kind),
            }
            if len(concept_calls) < int(max_events):
                concept_calls.append(call_rec)

            def _finalize_call(*, out_val: Any, meta: Dict[str, Any]) -> None:
                ret_snapshot: Dict[str, Any] = {"output": out_val, "meta": dict(meta)}
                try:
                    call_rec["return"] = copy.deepcopy(ret_snapshot)
                except Exception:
                    call_rec["return"] = dict(ret_snapshot)
                try:
                    call_rec["return_sig"] = _hash_obj(call_rec.get("return"))
                except Exception:
                    call_rec["return_sig"] = ""
                call_rec["ok"] = bool(meta.get("ok", False))

            # V80: match enforcement (fail-closed, explicit, deterministic).
            if not is_act_allowed_for_goal_kind(act=act, goal_kind=str(goal_kind)):
                call_rec["blocked"] = True
                call_rec["blocked_reason"] = "match_disallowed"
                meta = {
                    "ok": False,
                    "reason": "match_disallowed",
                    "concept_id": str(concept_id),
                    "goal_kind": str(goal_kind),
                }
                if len(events) < int(max_events):
                    events.append(
                        {
                            "t": "BLOCKED",
                            "step": int(step),
                            "depth": int(depth),
                            "concept_id": str(concept_id),
                            "goal_kind": str(goal_kind),
                            "blocked_reason": "match_disallowed",
                        }
                    )
                _finalize_call(out_val=None, meta=meta)
                return None, meta

            # Structural/ethics validation before any execution (fail-closed).
            pre = validate_before_execute(act=act, emission_preview=None)
            if not bool(pre.ok):
                meta = {
                    "ok": False,
                    "reason": "ethics_fail_closed_pre",
                    "concept_id": str(concept_id),
                    "goal_kind": str(goal_kind),
                    "ethics": pre.to_dict(),
                }
                _finalize_call(out_val=None, meta=meta)
                return None, meta

            # Typed inputs (deterministic).
            for k, want_t in in_schema.items():
                if k not in inps:
                    meta = {
                        "ok": False,
                        "reason": "missing_input",
                        "concept_id": str(concept_id),
                        "goal_kind": str(goal_kind),
                        "key": str(k),
                    }
                    _finalize_call(out_val=None, meta=meta)
                    return None, meta
                if not _type_ok(inps.get(k), str(want_t)):
                    meta = {
                        "ok": False,
                        "reason": "bad_input_type",
                        "concept_id": str(concept_id),
                        "goal_kind": str(goal_kind),
                        "key": str(k),
                        "want": str(want_t),
                        "got": str(type(inps.get(k)).__name__),
                    }
                    _finalize_call(out_val=None, meta=meta)
                    return None, meta

            call_event: Dict[str, Any] = {
                "t": "CALL",
                "step": int(step),
                "depth": int(depth),
                "concept_id": str(concept_id),
                "iface_sig": str(iface_sig),
                "inputs_sig": _hash_obj(inps),
                "goal_kind": str(goal_kind),
            }
            if len(events) < int(max_events):
                events.append(call_event)

            env: Dict[str, Any] = {}
            out_val: Any = None
            for ins_idx, ins in enumerate(act.program):
                op = str(ins.op)
                args = dict(ins.args or {})
                if len(events) < int(max_events):
                    ev2: Dict[str, Any] = {
                        "t": "INS",
                        "step": int(step),
                        "depth": int(depth),
                        "concept_id": str(concept_id),
                        "ins_idx": int(ins_idx),
                        "op": str(op),
                    }
                    if op == "CSV_GET_INPUT":
                        ev2["name"] = str(args.get("name") or "")
                        ev2["out"] = str(args.get("out") or ev2["name"])
                    elif op == "CSV_CONST":
                        ev2["out"] = str(args.get("out") or "")
                    elif op == "CSV_PRIMITIVE":
                        ev2["fn"] = str(args.get("fn") or "")
                        ins_in = args.get("in", [])
                        ev2["in"] = list(ins_in) if isinstance(ins_in, list) else []
                        ev2["out"] = str(args.get("out") or "")
                    elif op == "CSV_CALL":
                        ev2["callee"] = str(args.get("concept_id") or "")
                        ev2["out"] = str(args.get("out") or "")
                        bind = args.get("bind", {})
                        ev2["bind"] = dict(bind) if isinstance(bind, dict) else {}
                    elif op == "CSV_RETURN":
                        ev2["var"] = str(args.get("var") or "")
                    events.append(ev2)

                if op == "CSV_GET_INPUT":
                    name = str(args.get("name") or "")
                    out = str(args.get("out") or name)
                    env[out] = inps.get(name)
                elif op == "CSV_CONST":
                    out = str(args.get("out") or "")
                    env[out] = args.get("value")
                elif op == "CSV_PRIMITIVE":
                    fn_id = str(args.get("fn") or "")
                    out = str(args.get("out") or "")
                    ins_in = args.get("in", [])
                    if not isinstance(ins_in, list):
                        ins_in = []
                    vals = [env.get(str(v)) for v in ins_in]
                    spec_fn = PRIMITIVE_OPS.get(fn_id)
                    if spec_fn is None:
                        meta = {
                            "ok": False,
                            "reason": "unknown_primitive",
                            "fn": fn_id,
                            "concept_id": str(concept_id),
                            "goal_kind": str(goal_kind),
                        }
                        _finalize_call(out_val=None, meta=meta)
                        return None, meta
                    spec, fn = spec_fn
                    if int(spec.arity) != int(len(vals)):
                        meta = {
                            "ok": False,
                            "reason": "arity_mismatch",
                            "fn": fn_id,
                            "arity": int(spec.arity),
                            "got": int(len(vals)),
                            "concept_id": str(concept_id),
                            "goal_kind": str(goal_kind),
                        }
                        _finalize_call(out_val=None, meta=meta)
                        return None, meta
                    out_res = fn(*vals) if int(spec.arity) > 1 else fn(vals[0])
                    env[out] = out_res
                elif op == "CSV_CALL":
                    callee = str(args.get("concept_id") or "")
                    out = str(args.get("out") or "")
                    bind = args.get("bind", {})
                    if not isinstance(bind, dict):
                        bind = {}
                    sub_inps: Dict[str, Any] = {}
                    for k, v in bind.items():
                        sub_inps[str(k)] = env.get(str(v))
                    sub_out, sub_meta = _execute(
                        callee,
                        sub_inps,
                        depth + 1,
                        expected_for_validator=None,
                        validate_output=False,
                    )
                    if not bool(sub_meta.get("ok", False)):
                        meta = {
                            "ok": False,
                            "reason": "callee_failed",
                            "concept_id": str(concept_id),
                            "goal_kind": str(goal_kind),
                            "callee": str(callee),
                            "callee_meta": sub_meta,
                        }
                        _finalize_call(out_val=None, meta=meta)
                        return None, meta
                    env[out] = sub_out
                elif op == "CSV_RETURN":
                    var = str(args.get("var") or "")
                    out_val = env.get(var)
                    break
                else:
                    meta = {
                        "ok": False,
                        "reason": "unknown_csv_op",
                        "op": op,
                        "concept_id": str(concept_id),
                        "goal_kind": str(goal_kind),
                    }
                    _finalize_call(out_val=None, meta=meta)
                    return None, meta

            out_text = _value_to_text(out_val)

            # Validate output types (best-effort: single output slot allowed in MVP).
            if len(out_schema) == 1:
                want_t = next(iter(out_schema.values()))
                if not _type_ok(out_val, str(want_t)):
                    meta = {
                        "ok": False,
                        "reason": "bad_output_type",
                        "concept_id": str(concept_id),
                        "goal_kind": str(goal_kind),
                        "want": str(want_t),
                        "got": str(type(out_val).__name__),
                    }
                    _finalize_call(out_val=out_val, meta=meta)
                    return out_val, meta

            # Validator (optional, deterministic).
            vres = ValidatorResult(True, "skipped")
            evidence: Optional[Dict[str, Any]] = None
            if bool(validate_output) and validator_id:
                vres = run_validator(validator_id, out_text, expected_for_validator)
                if bool(vres.passed):
                    evidence = {
                        "validator_id": validator_id,
                        "reason": str(vres.reason),
                    }

            # Ethics (fail-closed) + uncertainty (IR->IC) on emission preview.
            eth = validate_before_execute(act=act, emission_preview=out_text)
            if not bool(eth.ok):
                out_text = fail_closed_text(eth)
                out_val = out_text
            out_text2, u = guard_text_uncertainty(out_text, evidence=evidence)
            out_val2: Any = out_text2 if isinstance(out_val, str) or out_val is None else out_val

            ret_meta: Dict[str, Any] = {
                "ok": bool(eth.ok) and bool(vres.passed),
                "reason": "ok"
                if bool(eth.ok) and bool(vres.passed)
                else ("ethics" if not bool(eth.ok) else "validator"),
                "concept_id": str(concept_id),
                "goal_kind": str(goal_kind),
                "validator": {"passed": bool(vres.passed), "reason": str(vres.reason), "validator_id": validator_id},
                "ethics": eth.to_dict(),
                "uncertainty": u.to_dict(),
                "output_text": str(out_text2),
                "output_sig": _hash_obj(out_text2),
                "outputs_sig": _hash_obj(out_text2),
            }
            if len(events) < int(max_events):
                events.append(
                    {
                        "t": "RETURN",
                        "step": int(step),
                        "depth": int(depth),
                        "concept_id": str(concept_id),
                        "ok": bool(ret_meta["ok"]),
                        "output_sig": str(ret_meta["output_sig"]),
                        "goal_kind": str(goal_kind),
                    }
                )
            _finalize_call(out_val=out_val2, meta=ret_meta)
            return out_val2, ret_meta

        out, meta = _execute(
            str(concept_act_id),
            dict(inputs),
            0,
            expected_for_validator=expected,
            validate_output=bool(validate_output),
        )
        return {
            "output": out,
            "meta": meta,
            "events": events,
            "trace": {"concept_calls": list(concept_calls)},
        }

