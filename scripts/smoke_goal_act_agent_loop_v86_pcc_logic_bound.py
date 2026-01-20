#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.agent_loop_goals_v86 import run_goals_v86
from atos_core.logic_v86 import ensure_bool_primitives_registered
from atos_core.pcc_v74 import certificate_sig_v2
from atos_core.pcc_v85 import build_certificate_v85
from atos_core.pcc_v86 import build_certificate_v86, verify_pcc_v86
from atos_core.store import ActStore


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()


def sha256_canon(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"ERROR: path already exists: {path}")


def write_json(path: str, obj: Any) -> str:
    ensure_absent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)
    return sha256_file(path)


def make_concept_act(
    *,
    act_id: str,
    match: Optional[Dict[str, Any]] = None,
    input_schema: Dict[str, str],
    output_schema: Dict[str, str],
    validator_id: str,
    program: List[Instruction],
) -> Act:
    return Act(
        id=str(act_id),
        version=1,
        created_at=deterministic_iso(step=0),
        kind="concept_csv",
        match=dict(match) if isinstance(match, dict) else {},
        program=list(program),
        evidence={
            "interface": {
                "input_schema": dict(input_schema),
                "output_schema": dict(output_schema),
                "validator_id": str(validator_id),
            }
        },
        cost={},
        deps=[],
        active=True,
    )


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _points_from_events(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    idxs: List[int] = []
    for i, r in enumerate(events):
        if not isinstance(r, dict):
            continue
        if str(r.get("event_kind") or "") == "promotion_attempt" and str(r.get("decision") or "") == "promoted":
            idxs.append(int(i))

    points: List[Dict[str, Any]] = []
    for pi, ev_idx in enumerate(idxs):
        before = None
        for r in reversed(list(events[:ev_idx])):
            if not isinstance(r, dict):
                continue
            if str(r.get("event_kind") or "") != "goal_attempt":
                continue
            before = int(r.get("steps_total", 0) or 0)
            break
        after = None
        for r in list(events[ev_idx + 1 :]):
            if not isinstance(r, dict):
                continue
            if str(r.get("event_kind") or "") != "goal_attempt":
                continue
            after = int(r.get("steps_total", 0) or 0)
            break
        if before is None or after is None:
            _fail("ERROR: could not derive before/after steps around promotion event")
        points.append(
            {
                "promotion_index": int(pi),
                "steps_before": int(before),
                "steps_after": int(after),
                "delta_steps": int(int(before) - int(after)),
            }
        )
    return points


def _iface_output_keys_from_act_json(act_json: Dict[str, Any]) -> List[str]:
    ev = act_json.get("evidence") if isinstance(act_json.get("evidence"), dict) else {}
    iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
    out_schema = iface.get("output_schema") if isinstance(iface.get("output_schema"), dict) else {}
    return [str(k) for k in sorted(out_schema.keys(), key=str)]


def _negative_test_missing_logic_binding(*, seed: int) -> Dict[str, Any]:
    ensure_bool_primitives_registered()
    store_base = ActStore()

    cand_id = "concept_v86_neg_missing_logic_binding_candidate_v0"
    candidate_act = make_concept_act(
        act_id=cand_id,
        match={"goal_kinds": ["v86_logic_sum"]},
        input_schema={"a": "str", "b": "str"},
        output_schema={"bit": "str"},
        validator_id="text_exact",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
            Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
            Instruction("CSV_PRIMITIVE", {"fn": "bool_xor01", "in": ["a", "b"], "out": "bit"}),
            Instruction("CSV_RETURN", {"var": "bit"}),
        ],
    )

    vector_specs = [
        {"context_id": "tt00", "goal_kind": "v86_logic_sum", "inputs": {"a": "0", "b": "0"}, "expected": "0"},
        {"context_id": "tt01", "goal_kind": "v86_logic_sum", "inputs": {"a": "0", "b": "1"}, "expected": "1"},
        {"context_id": "tt10", "goal_kind": "v86_logic_sum", "inputs": {"a": "1", "b": "0"}, "expected": "1"},
        {"context_id": "tt11", "goal_kind": "v86_logic_sum", "inputs": {"a": "1", "b": "1"}, "expected": "0"},
    ]
    mined_from = {
        "trace_sigs": ["trace_a", "trace_b"],
        "goal_kinds": ["v86_logic_sum"],
        "goal_kinds_distinct": 1,
        "candidate": {"sub_sig": "sub_sig_dummy", "subpath": ["noop"], "goal_kinds_supported": ["v86_logic_sum"]},
    }

    # Build a V85 cert (no logic_binding) that should pass V85 and fail V86 requirement.
    cert = build_certificate_v85(
        candidate_act=candidate_act,
        store_base=store_base,
        mined_from=mined_from,
        vector_specs=vector_specs,
        seed=int(seed),
    )

    ok, reason, details = verify_pcc_v86(candidate_act=candidate_act, certificate=cert, store_base=store_base, seed=int(seed))
    if bool(ok):
        _fail("ERROR: expected verify_pcc_v86 to fail for missing_logic_binding")
    if str(reason) != "missing_logic_binding":
        _fail(f"ERROR: expected reason=='missing_logic_binding', got={reason}")
    return {"ok": False, "reason": str(reason), "certificate_sig": str(cert.get("certificate_sig") or "")}


def _logic_spec_xor_raw() -> Dict[str, Any]:
    return {
        "vars": ["a", "b"],
        "domain_values": ["0", "1"],
        "expr": {"op": "xor", "args": [{"op": "var", "name": "a"}, {"op": "var", "name": "b"}]},
        "render": {"kind": "raw01"},
    }


def _logic_spec_xor_prefix(prefix: str) -> Dict[str, Any]:
    return {
        "vars": ["a", "b"],
        "domain_values": ["0", "1"],
        "expr": {"op": "xor", "args": [{"op": "var", "name": "a"}, {"op": "var", "name": "b"}]},
        "render": {"kind": "prefix01", "prefix": str(prefix)},
    }


def _negative_test_logic_binding_sig_mismatch(*, seed: int) -> Dict[str, Any]:
    ensure_bool_primitives_registered()
    store_base = ActStore()

    cand_id = "concept_v86_neg_logic_binding_sig_mismatch_candidate_v0"
    candidate_act = make_concept_act(
        act_id=cand_id,
        match={"goal_kinds": ["v86_logic_sum"]},
        input_schema={"a": "str", "b": "str"},
        output_schema={"bit": "str"},
        validator_id="text_exact",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
            Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
            Instruction("CSV_PRIMITIVE", {"fn": "bool_xor01", "in": ["a", "b"], "out": "bit"}),
            Instruction("CSV_RETURN", {"var": "bit"}),
        ],
    )

    vector_specs = [
        {"context_id": "tt00", "goal_kind": "v86_logic_sum", "inputs": {"a": "0", "b": "0"}, "expected": "0"},
        {"context_id": "tt01", "goal_kind": "v86_logic_sum", "inputs": {"a": "0", "b": "1"}, "expected": "1"},
        {"context_id": "tt10", "goal_kind": "v86_logic_sum", "inputs": {"a": "1", "b": "0"}, "expected": "1"},
        {"context_id": "tt11", "goal_kind": "v86_logic_sum", "inputs": {"a": "1", "b": "1"}, "expected": "0"},
    ]
    mined_from = {
        "trace_sigs": ["trace_sig_a", "trace_sig_b"],
        "goal_kinds": ["v86_logic_sum"],
        "goal_kinds_distinct": 1,
        "candidate": {"sub_sig": "sub_sig_lb", "subpath": ["noop"], "goal_kinds_supported": ["v86_logic_sum"]},
    }

    cert = build_certificate_v86(
        candidate_act=candidate_act,
        store_base=store_base,
        mined_from=mined_from,
        vector_specs=vector_specs,
        seed=int(seed),
        logic_spec=_logic_spec_xor_raw(),
    )

    # Tamper binding_sig but keep certificate_sig consistent.
    cert2 = json.loads(json.dumps(cert))
    lb = cert2.get("logic_binding")
    if not isinstance(lb, dict):
        _fail("ERROR: missing logic_binding in built certificate")
    lb["binding_sig"] = "0" * 64
    cert2["logic_binding"] = dict(lb)
    cert2["certificate_sig"] = certificate_sig_v2(cert2)

    ok, reason, details = verify_pcc_v86(candidate_act=candidate_act, certificate=cert2, store_base=store_base, seed=int(seed))
    if bool(ok):
        _fail("ERROR: expected verify_pcc_v86 to fail for logic_binding_sig_mismatch")
    if str(reason) != "logic_binding_sig_mismatch":
        _fail(f"ERROR: expected reason=='logic_binding_sig_mismatch', got={reason}")
    if not isinstance(details, dict):
        _fail("ERROR: expected details dict for logic_binding_sig_mismatch")
    return {"ok": False, "reason": str(reason), "want": str(details.get("want") or ""), "got": str(details.get("got") or "")}


def _negative_test_logic_vector_coverage_incomplete(*, seed: int) -> Dict[str, Any]:
    ensure_bool_primitives_registered()
    store_base = ActStore()

    cand_id = "concept_v86_neg_logic_vector_coverage_incomplete_candidate_v0"
    candidate_act = make_concept_act(
        act_id=cand_id,
        match={"goal_kinds": ["v86_logic_sum"]},
        input_schema={"a": "str", "b": "str"},
        output_schema={"bit": "str"},
        validator_id="text_exact",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
            Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
            Instruction("CSV_PRIMITIVE", {"fn": "bool_xor01", "in": ["a", "b"], "out": "bit"}),
            Instruction("CSV_RETURN", {"var": "bit"}),
        ],
    )

    # Omit one assignment (tt11).
    vector_specs = [
        {"context_id": "tt00", "goal_kind": "v86_logic_sum", "inputs": {"a": "0", "b": "0"}, "expected": "0"},
        {"context_id": "tt01", "goal_kind": "v86_logic_sum", "inputs": {"a": "0", "b": "1"}, "expected": "1"},
        {"context_id": "tt10", "goal_kind": "v86_logic_sum", "inputs": {"a": "1", "b": "0"}, "expected": "1"},
    ]
    mined_from = {
        "trace_sigs": ["trace_sig_a", "trace_sig_b"],
        "goal_kinds": ["v86_logic_sum"],
        "goal_kinds_distinct": 1,
        "candidate": {"sub_sig": "sub_sig_cov", "subpath": ["noop"], "goal_kinds_supported": ["v86_logic_sum"]},
    }

    cert = build_certificate_v86(
        candidate_act=candidate_act,
        store_base=store_base,
        mined_from=mined_from,
        vector_specs=vector_specs,
        seed=int(seed),
        logic_spec=_logic_spec_xor_raw(),
    )

    ok, reason, details = verify_pcc_v86(candidate_act=candidate_act, certificate=cert, store_base=store_base, seed=int(seed))
    if bool(ok):
        _fail("ERROR: expected verify_pcc_v86 to fail for logic_vector_coverage_incomplete")
    if str(reason) != "logic_vector_coverage_incomplete":
        _fail(f"ERROR: expected reason=='logic_vector_coverage_incomplete', got={reason}")
    if not isinstance(details, dict):
        _fail("ERROR: expected details dict for logic_vector_coverage_incomplete")
    if int(details.get("missing_total", 0) or 0) <= 0:
        _fail("ERROR: expected missing_total > 0 for logic_vector_coverage_incomplete")
    return {"ok": False, "reason": str(reason), "missing_total": int(details.get("missing_total", 0) or 0)}


def _negative_test_logic_expected_mismatch(*, seed: int) -> Dict[str, Any]:
    ensure_bool_primitives_registered()
    store_base = ActStore()

    cand_id = "concept_v86_neg_logic_expected_mismatch_candidate_v0"
    candidate_act = make_concept_act(
        act_id=cand_id,
        match={"goal_kinds": ["v86_logic_sum"]},
        input_schema={"a": "str", "b": "str"},
        output_schema={"bit": "str"},
        validator_id="text_exact",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
            Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
            Instruction("CSV_PRIMITIVE", {"fn": "bool_xor01", "in": ["a", "b"], "out": "bit"}),
            Instruction("CSV_RETURN", {"var": "bit"}),
        ],
    )

    # Correct XOR vectors (so V85 passes), but bind logic expr as AND (so V86 fails).
    vector_specs = [
        {"context_id": "tt00", "goal_kind": "v86_logic_sum", "inputs": {"a": "0", "b": "0"}, "expected": "0"},
        {"context_id": "tt01", "goal_kind": "v86_logic_sum", "inputs": {"a": "0", "b": "1"}, "expected": "1"},
        {"context_id": "tt10", "goal_kind": "v86_logic_sum", "inputs": {"a": "1", "b": "0"}, "expected": "1"},
        {"context_id": "tt11", "goal_kind": "v86_logic_sum", "inputs": {"a": "1", "b": "1"}, "expected": "0"},
    ]
    mined_from = {
        "trace_sigs": ["trace_sig_a", "trace_sig_b"],
        "goal_kinds": ["v86_logic_sum"],
        "goal_kinds_distinct": 1,
        "candidate": {"sub_sig": "sub_sig_exp", "subpath": ["noop"], "goal_kinds_supported": ["v86_logic_sum"]},
    }

    logic_spec_and = {
        "vars": ["a", "b"],
        "domain_values": ["0", "1"],
        "expr": {"op": "and", "args": [{"op": "var", "name": "a"}, {"op": "var", "name": "b"}]},
        "render": {"kind": "raw01"},
    }

    cert = build_certificate_v86(
        candidate_act=candidate_act,
        store_base=store_base,
        mined_from=mined_from,
        vector_specs=vector_specs,
        seed=int(seed),
        logic_spec=logic_spec_and,
    )

    ok, reason, details = verify_pcc_v86(candidate_act=candidate_act, certificate=cert, store_base=store_base, seed=int(seed))
    if bool(ok):
        _fail("ERROR: expected verify_pcc_v86 to fail for logic_expected_mismatch")
    if str(reason) != "logic_expected_mismatch":
        _fail(f"ERROR: expected reason=='logic_expected_mismatch', got={reason}")
    if not isinstance(details, dict):
        _fail("ERROR: expected details dict for logic_expected_mismatch")
    if not str(details.get("want") or "") or not str(details.get("got") or ""):
        _fail("ERROR: expected want/got in details for logic_expected_mismatch")
    return {"ok": False, "reason": str(reason), "inputs": details.get("inputs"), "want": str(details.get("want") or ""), "got": str(details.get("got") or "")}


def _xor01(a: str, b: str) -> str:
    aa = 1 if str(a) == "1" else 0
    bb = 1 if str(b) == "1" else 0
    return "1" if (aa ^ bb) else "0"


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    ensure_bool_primitives_registered()
    store = ActStore()

    # Base concepts (2 goal_kinds with different prefixes to force core promotion first).
    normalize_a_id = "concept_v86_normalize_a_v0"
    normalize_b_id = "concept_v86_normalize_b_v0"
    xor_id = "concept_v86_xor_na_nb_v0"
    fmt_sum_id = "concept_v86_fmt_bool_sum_v0"
    fmt_total_id = "concept_v86_fmt_bool_total_v0"

    store.add(
        make_concept_act(
            act_id=normalize_a_id,
            match={},
            input_schema={"a": "str"},
            output_schema={"na": "str"},
            validator_id="text_exact",
            program=[
                Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
                Instruction("CSV_RETURN", {"var": "a"}),
            ],
        )
    )
    store.add(
        make_concept_act(
            act_id=normalize_b_id,
            match={},
            input_schema={"b": "str"},
            output_schema={"nb": "str"},
            validator_id="text_exact",
            program=[
                Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
                Instruction("CSV_RETURN", {"var": "b"}),
            ],
        )
    )
    store.add(
        make_concept_act(
            act_id=xor_id,
            match={},
            input_schema={"na": "str", "nb": "str"},
            output_schema={"bit": "str"},
            validator_id="text_exact",
            program=[
                Instruction("CSV_GET_INPUT", {"name": "na", "out": "na"}),
                Instruction("CSV_GET_INPUT", {"name": "nb", "out": "nb"}),
                Instruction("CSV_PRIMITIVE", {"fn": "bool_xor01", "in": ["na", "nb"], "out": "bit"}),
                Instruction("CSV_RETURN", {"var": "bit"}),
            ],
        )
    )
    store.add(
        make_concept_act(
            act_id=fmt_sum_id,
            match={"goal_kinds": ["v86_logic_sum"]},
            input_schema={"bit": "str"},
            output_schema={"out": "str"},
            validator_id="text_exact",
            program=[
                Instruction("CSV_GET_INPUT", {"name": "bit", "out": "bit"}),
                Instruction("CSV_CONST", {"out": "prefix", "value": "BOOL="}),
                Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["prefix", "bit"], "out": "out"}),
                Instruction("CSV_RETURN", {"var": "out"}),
            ],
        )
    )
    store.add(
        make_concept_act(
            act_id=fmt_total_id,
            match={"goal_kinds": ["v86_logic_total"]},
            input_schema={"bit": "str"},
            output_schema={"out": "str"},
            validator_id="text_exact",
            program=[
                Instruction("CSV_GET_INPUT", {"name": "bit", "out": "bit"}),
                Instruction("CSV_CONST", {"out": "prefix", "value": "TBOOL="}),
                Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["prefix", "bit"], "out": "out"}),
                Instruction("CSV_RETURN", {"var": "out"}),
            ],
        )
    )

    # V86 negatives.
    neg_missing = _negative_test_missing_logic_binding(seed=int(seed))
    neg_sig = _negative_test_logic_binding_sig_mismatch(seed=int(seed))
    neg_cov = _negative_test_logic_vector_coverage_incomplete(seed=int(seed))
    neg_exp = _negative_test_logic_expected_mismatch(seed=int(seed))

    # Goals: first two force diversity, then two sum to enable second promotion.
    goals_specs = [
        {"created_step": 0, "goal_kind": "v86_logic_sum", "a": "0", "b": "1", "prefix": "BOOL="},
        {"created_step": 1, "goal_kind": "v86_logic_total", "a": "1", "b": "0", "prefix": "TBOOL="},
        {"created_step": 2, "goal_kind": "v86_logic_sum", "a": "0", "b": "0", "prefix": "BOOL="},
        {"created_step": 3, "goal_kind": "v86_logic_sum", "a": "1", "b": "1", "prefix": "BOOL="},
        {"created_step": 4, "goal_kind": "v86_logic_sum", "a": "1", "b": "0", "prefix": "BOOL="},
        {"created_step": 5, "goal_kind": "v86_logic_total", "a": "0", "b": "0", "prefix": "TBOOL="},
    ]
    for gs in goals_specs:
        bit = _xor01(str(gs["a"]), str(gs["b"]))
        expected = f"{gs['prefix']}{bit}"
        # create goal_v75 act via helper already in repo
        from atos_core.goal_act_v75 import make_goal_act_v75

        ga = make_goal_act_v75(
            goal_kind=str(gs["goal_kind"]),
            bindings={"a": str(gs["a"]), "b": str(gs["b"])},
            output_key="out",
            expected=str(expected),
            validator_id="text_exact",
            created_step=int(gs["created_step"]),
        )
        store.add(ga)

    loop_res = run_goals_v86(
        store=store,
        seed=int(seed),
        out_dir=out_dir,
        max_rounds=10,
        max_goals_per_round=1,
        enable_promotion=True,
        promotion_budget_bits=2048,
        promotion_min_traces=2,
        promotion_top_k=8,
        max_promotions_per_run=2,
        promotion_kind_diversity_min=2,
    )

    if int(loop_res.get("goals_total", 0) or 0) != 6:
        _fail(f"ERROR: expected goals_total==6, got={loop_res.get('goals_total')}")
    if int(loop_res.get("goals_satisfied", 0) or 0) != 6:
        _fail(f"ERROR: expected goals_satisfied==6, got={loop_res.get('goals_satisfied')}")
    if int(loop_res.get("promoted_total", 0) or 0) != 2:
        _fail(f"ERROR: expected promoted_total==2, got={loop_res.get('promoted_total')}")

    budget_bits = int(loop_res.get("budget_bits", 0) or 0)
    used_bits = int(loop_res.get("used_bits", 0) or 0)
    if budget_bits != 2048:
        _fail(f"ERROR: expected budget_bits==2048, got={budget_bits}")
    if used_bits != 2048:
        _fail(f"ERROR: expected used_bits==2048, got={used_bits}")

    events_path = os.path.join(out_dir, "goals_v86_events.jsonl")
    traces_path = os.path.join(out_dir, "traces_v86.json")
    mined_path = os.path.join(out_dir, "mined_candidates_v86.json")
    promos_path = os.path.join(out_dir, "promotion", "v86_promotions.jsonl")
    curve_path = os.path.join(out_dir, "compression_curve.json")
    cand0_act = os.path.join(out_dir, "candidates", "candidate_000_act.json")
    cand0_cert = os.path.join(out_dir, "candidates", "candidate_000_certificate_v2.json")
    cand1_act = os.path.join(out_dir, "candidates", "candidate_001_act.json")
    cand1_cert = os.path.join(out_dir, "candidates", "candidate_001_certificate_v2.json")

    for p in [events_path, traces_path, mined_path, promos_path, curve_path, cand0_act, cand0_cert, cand1_act, cand1_cert]:
        if not os.path.exists(p):
            _fail(f"ERROR: missing required artifact: {p}")

    events = _read_jsonl(events_path)
    promo_events = [r for r in events if isinstance(r, dict) and str(r.get("event_kind") or "") == "promotion_attempt" and str(r.get("decision") or "") == "promoted"]
    if len(promo_events) != 2:
        _fail(f"ERROR: expected 2 promoted promotion_attempt events, got={len(promo_events)}")

    points = _points_from_events(events)
    if len(points) != 2:
        _fail(f"ERROR: expected 2 compression points, got={points}")
    steps_after = [int(p.get("steps_after", 0) or 0) for p in points if isinstance(p, dict)]
    if steps_after != [2, 1]:
        _fail(f"ERROR: expected curve steps_after [2,1], got={steps_after}")

    curve = json.load(open(curve_path, "r", encoding="utf-8"))
    curve_points = curve.get("points") if isinstance(curve.get("points"), list) else []
    if curve_points != points:
        _fail("ERROR: compression_curve.json points mismatch vs derived points")

    cand0 = json.load(open(cand0_act, "r", encoding="utf-8"))
    cand1 = json.load(open(cand1_act, "r", encoding="utf-8"))
    out_keys0 = _iface_output_keys_from_act_json(cand0 if isinstance(cand0, dict) else {})
    out_keys1 = _iface_output_keys_from_act_json(cand1 if isinstance(cand1, dict) else {})
    if "bit" not in out_keys0 or "out" in out_keys0:
        _fail(f"ERROR: candidate_000_act output_schema must contain 'bit' and NOT 'out', got={out_keys0}")
    if "out" not in out_keys1:
        _fail(f"ERROR: candidate_001_act output_schema must contain 'out', got={out_keys1}")

    cert0 = json.load(open(cand0_cert, "r", encoding="utf-8"))
    cert1 = json.load(open(cand1_cert, "r", encoding="utf-8"))
    lb0 = cert0.get("logic_binding") if isinstance(cert0.get("logic_binding"), dict) else None
    lb1 = cert1.get("logic_binding") if isinstance(cert1.get("logic_binding"), dict) else None
    if not isinstance(lb0, dict) or not str(lb0.get("truth_table_sha256") or ""):
        _fail("ERROR: candidate_000_certificate_v2.json missing logic_binding/truth_table_sha256")
    if not isinstance(lb1, dict) or not str(lb1.get("truth_table_sha256") or ""):
        _fail("ERROR: candidate_001_certificate_v2.json missing logic_binding/truth_table_sha256")
    if str((lb0.get("render") or {}).get("kind") or "") != "raw01":
        _fail("ERROR: expected candidate_000 logic_binding.render.kind == raw01")
    if str((lb1.get("render") or {}).get("kind") or "") != "prefix01":
        _fail("ERROR: expected candidate_001 logic_binding.render.kind == prefix01")

    cert_sigs: List[str] = [str(r.get("certificate_sig") or "") for r in promo_events]
    if len(cert_sigs) != 2 or any(not s for s in cert_sigs):
        _fail("ERROR: missing certificate_sig(s) in promotion events")

    artifacts = {
        "goals_v86_events_jsonl_sha256": sha256_file(events_path),
        "traces_v86_json_sha256": sha256_file(traces_path),
        "mined_candidates_v86_json_sha256": sha256_file(mined_path),
        "v86_promotions_jsonl_sha256": sha256_file(promos_path),
        "compression_curve_json_sha256": sha256_file(curve_path),
        "candidate_000_act_json_sha256": sha256_file(cand0_act),
        "candidate_000_certificate_v2_json_sha256": sha256_file(cand0_cert),
        "candidate_001_act_json_sha256": sha256_file(cand1_act),
        "candidate_001_certificate_v2_json_sha256": sha256_file(cand1_cert),
    }

    return {
        "schema_version": 1,
        "seed": int(seed),
        "negative_missing_logic_binding": dict(neg_missing),
        "negative_logic_binding_sig_mismatch": dict(neg_sig),
        "negative_logic_vector_coverage_incomplete": dict(neg_cov),
        "negative_logic_expected_mismatch": dict(neg_exp),
        "goals_total": int(loop_res.get("goals_total", 0) or 0),
        "goals_satisfied": int(loop_res.get("goals_satisfied", 0) or 0),
        "promoted_total": int(loop_res.get("promoted_total", 0) or 0),
        "budget_bits": int(budget_bits),
        "used_bits": int(used_bits),
        "certificate_sigs": list(cert_sigs),
        "compression_curve_points": list(points),
        "artifacts": dict(artifacts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", default="results/run_smoke_goal_act_agent_loop_v86_pcc_logic_bound")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_base = str(args.out_base)
    seed = int(args.seed)

    results: Dict[str, Any] = {"seed": seed, "tries": {}}
    sigs: List[Tuple[Tuple[str, ...], Tuple[int, ...], int, str, str, str, str, str]] = []
    summary_shas: List[str] = []

    for t in (1, 2):
        out_dir = f"{out_base}_try{t}"
        ensure_absent(out_dir)

        ev = smoke_try(out_dir=out_dir, seed=seed)
        eval_path = os.path.join(out_dir, "eval.json")
        eval_sha = write_json(eval_path, ev)

        points = ev.get("compression_curve_points") if isinstance(ev.get("compression_curve_points"), list) else []
        pts_sig = tuple(int(p.get("steps_after", 0) or 0) for p in points if isinstance(p, dict))
        certs = tuple(str(x) for x in (ev.get("certificate_sigs") or []) if str(x))
        r_a = str(((ev.get("negative_missing_logic_binding") or {}) if isinstance(ev.get("negative_missing_logic_binding"), dict) else {}).get("reason") or "")
        r_b = str(((ev.get("negative_logic_binding_sig_mismatch") or {}) if isinstance(ev.get("negative_logic_binding_sig_mismatch"), dict) else {}).get("reason") or "")
        r_c = str(((ev.get("negative_logic_vector_coverage_incomplete") or {}) if isinstance(ev.get("negative_logic_vector_coverage_incomplete"), dict) else {}).get("reason") or "")
        r_d = str(((ev.get("negative_logic_expected_mismatch") or {}) if isinstance(ev.get("negative_logic_expected_mismatch"), dict) else {}).get("reason") or "")

        core = {
            "seed": int(seed),
            "negative_missing_logic_binding_reason": str(r_a),
            "negative_logic_binding_sig_mismatch_reason": str(r_b),
            "negative_logic_vector_coverage_incomplete_reason": str(r_c),
            "negative_logic_expected_mismatch_reason": str(r_d),
            "promoted_total": int(ev.get("promoted_total", 0) or 0),
            "budget_bits": int(ev.get("budget_bits", 0) or 0),
            "used_bits": int(ev.get("used_bits", 0) or 0),
            "certificate_sigs": list(certs),
            "compression_curve_points": list(points),
            "sha256_eval_json": str(eval_sha),
        }
        summary_sha = sha256_text(canonical_json_dumps(core))
        smoke = {"summary": core, "determinism": {"summary_sha256": str(summary_sha)}}
        smoke_path = os.path.join(out_dir, "smoke_summary.json")
        smoke_sha = write_json(smoke_path, smoke)

        sigs.append((certs, pts_sig, int(ev.get("promoted_total", 0) or 0), str(eval_sha), str(r_a), str(r_b), str(r_c), str(r_d)))
        summary_shas.append(str(summary_sha))

        results["tries"][f"try{t}"] = {
            "out_dir": out_dir,
            "eval_json": {"path": eval_path, "sha256": eval_sha},
            "smoke_summary_json": {"path": smoke_path, "sha256": smoke_sha},
            "summary_sha256": summary_sha,
        }

    determinism_ok = bool(len(sigs) == 2 and sigs[0] == sigs[1] and len(summary_shas) == 2 and summary_shas[0] == summary_shas[1])
    if not determinism_ok:
        _fail(f"ERROR: determinism mismatch: sigs={sigs} summary_shas={summary_shas}")
    results["determinism"] = {
        "ok": True,
        "summary_sha256": summary_shas[0],
        "certificate_sigs": list(sigs[0][0]),
        "curve_steps_after": list(sigs[0][1]),
        "promoted_total": sigs[0][2],
        "sha256_eval_json": sigs[0][3],
        "negative_missing_logic_binding_reason": sigs[0][4],
        "negative_logic_binding_sig_mismatch_reason": sigs[0][5],
        "negative_logic_vector_coverage_incomplete_reason": sigs[0][6],
        "negative_logic_expected_mismatch_reason": sigs[0][7],
    }
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
