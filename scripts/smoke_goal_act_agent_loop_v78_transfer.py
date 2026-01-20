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
from atos_core.agent_loop_goals_v78 import run_goals_v78
from atos_core.goal_act_v75 import make_goal_act_v75
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


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    store = ActStore()

    # Base micro-world (4-step baseline) with two goal kinds:
    # normalize_x: x -> nx
    # normalize_y: y -> ny
    # add_nx_ny: nx,ny -> sum
    # fmt_sum: sum -> out (out = "SUM=" + sum)
    # fmt_total: sum -> out (out = "TOTAL=" + sum)
    normalize_x_id = "concept_v72_normalize_x_v0"
    normalize_y_id = "concept_v72_normalize_y_v0"
    add_nx_ny_id = "concept_v72_add_nx_ny_v0"
    fmt_sum_id = "concept_v78_fmt_sum_v0"
    fmt_total_id = "concept_v78_fmt_total_v0"

    store.add(
        make_concept_act(
            act_id=normalize_x_id,
            match={},
            input_schema={"x": "str"},
            output_schema={"nx": "str"},
            validator_id="text_exact",
            program=[
                Instruction("CSV_GET_INPUT", {"name": "x", "out": "x"}),
                Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["x"], "out": "d"}),
                Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "i"}),
                Instruction("CSV_PRIMITIVE", {"fn": "int_to_digits", "in": ["i"], "out": "nx"}),
                Instruction("CSV_RETURN", {"var": "nx"}),
            ],
        )
    )
    store.add(
        make_concept_act(
            act_id=normalize_y_id,
            match={},
            input_schema={"y": "str"},
            output_schema={"ny": "str"},
            validator_id="text_exact",
            program=[
                Instruction("CSV_GET_INPUT", {"name": "y", "out": "y"}),
                Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["y"], "out": "d"}),
                Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "i"}),
                Instruction("CSV_PRIMITIVE", {"fn": "int_to_digits", "in": ["i"], "out": "ny"}),
                Instruction("CSV_RETURN", {"var": "ny"}),
            ],
        )
    )
    store.add(
        make_concept_act(
            act_id=add_nx_ny_id,
            match={},
            input_schema={"nx": "str", "ny": "str"},
            output_schema={"sum": "str"},
            validator_id="text_exact",
            program=[
                Instruction("CSV_GET_INPUT", {"name": "nx", "out": "nx"}),
                Instruction("CSV_GET_INPUT", {"name": "ny", "out": "ny"}),
                Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["nx"], "out": "dx"}),
                Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["ny"], "out": "dy"}),
                Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["dx"], "out": "ix"}),
                Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["dy"], "out": "iy"}),
                Instruction("CSV_PRIMITIVE", {"fn": "add_int", "in": ["ix", "iy"], "out": "sum_i"}),
                Instruction("CSV_PRIMITIVE", {"fn": "int_to_digits", "in": ["sum_i"], "out": "sum"}),
                Instruction("CSV_RETURN", {"var": "sum"}),
            ],
        )
    )
    store.add(
        make_concept_act(
            act_id=fmt_sum_id,
            match={"goal_kinds": ["v78_fmt_sum"]},
            input_schema={"sum": "str"},
            output_schema={"out": "str"},
            validator_id="text_exact",
            program=[
                Instruction("CSV_GET_INPUT", {"name": "sum", "out": "sum"}),
                Instruction("CSV_CONST", {"out": "prefix", "value": "SUM="}),
                Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["prefix", "sum"], "out": "out"}),
                Instruction("CSV_RETURN", {"var": "out"}),
            ],
        )
    )
    store.add(
        make_concept_act(
            act_id=fmt_total_id,
            match={"goal_kinds": ["v78_fmt_total"]},
            input_schema={"sum": "str"},
            output_schema={"out": "str"},
            validator_id="text_exact",
            program=[
                Instruction("CSV_GET_INPUT", {"name": "sum", "out": "sum"}),
                Instruction("CSV_CONST", {"out": "prefix", "value": "TOTAL="}),
                Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["prefix", "sum"], "out": "out"}),
                Instruction("CSV_RETURN", {"var": "out"}),
            ],
        )
    )

    # Six persistent goal_v75 acts, ordered by created_step curriculum.
    goals_specs = [
        {"created_step": 0, "goal_kind": "v78_fmt_sum", "bindings": {"x": "0004", "y": "0008"}, "expected": "SUM=12"},
        {"created_step": 1, "goal_kind": "v78_fmt_total", "bindings": {"x": "0010", "y": "0002"}, "expected": "TOTAL=12"},
        {"created_step": 2, "goal_kind": "v78_fmt_total", "bindings": {"x": "0009", "y": "0003"}, "expected": "TOTAL=12"},
        {"created_step": 3, "goal_kind": "v78_fmt_sum", "bindings": {"x": "0007", "y": "0005"}, "expected": "SUM=12"},
        {"created_step": 4, "goal_kind": "v78_fmt_sum", "bindings": {"x": "0006", "y": "0006"}, "expected": "SUM=12"},
        {"created_step": 5, "goal_kind": "v78_fmt_sum", "bindings": {"x": "0001", "y": "0011"}, "expected": "SUM=12"},
    ]
    for gs in goals_specs:
        ga = make_goal_act_v75(
            goal_kind=str(gs["goal_kind"]),
            bindings=dict(gs["bindings"]),
            output_key="out",
            expected=gs["expected"],
            validator_id="text_exact",
            created_step=int(gs["created_step"]),
        )
        store.add(ga)

    loop_res = run_goals_v78(
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
        _fail(f"ERROR: expected goals_total == 6, got={loop_res.get('goals_total')}")
    if int(loop_res.get("goals_satisfied", 0) or 0) != 6:
        _fail(f"ERROR: expected goals_satisfied == 6, got={loop_res.get('goals_satisfied')}")
    if int(loop_res.get("promoted_total", 0) or 0) != 2:
        _fail(f"ERROR: expected promoted_total == 2, got={loop_res.get('promoted_total')}")

    budget_bits = int(loop_res.get("budget_bits", 0) or 0)
    used_bits = int(loop_res.get("used_bits", 0) or 0)
    if used_bits > budget_bits:
        _fail(f"ERROR: used_bits exceeds budget: used={used_bits} budget={budget_bits}")
    if used_bits != 2048:
        _fail(f"ERROR: expected used_bits == 2048, got={used_bits}")

    events_path = os.path.join(out_dir, "goals_v78_events.jsonl")
    traces_path = os.path.join(out_dir, "traces_v78.json")
    mined_path = os.path.join(out_dir, "mined_candidates_v78.json")
    promos_path = os.path.join(out_dir, "promotion", "v78_promotions.jsonl")
    curve_path = os.path.join(out_dir, "compression_curve.json")
    cand0_act = os.path.join(out_dir, "candidates", "candidate_000_act.json")
    cand0_cert = os.path.join(out_dir, "candidates", "candidate_000_certificate_v2.json")
    cand1_act = os.path.join(out_dir, "candidates", "candidate_001_act.json")
    cand1_cert = os.path.join(out_dir, "candidates", "candidate_001_certificate_v2.json")

    for p in (events_path, traces_path, mined_path, promos_path, curve_path, cand0_act, cand0_cert, cand1_act, cand1_cert):
        if not os.path.exists(p):
            _fail(f"ERROR: missing required artifact: {p}")

    events = _read_jsonl(events_path)
    promo_events = [
        r
        for r in events
        if isinstance(r, dict) and str(r.get("event_kind") or "") == "promotion_attempt" and str(r.get("decision") or "") == "promoted"
    ]
    if len(promo_events) != 2:
        _fail(f"ERROR: expected 2 promoted promotion_attempt events, got={len(promo_events)}")

    # Event schema must include goal_kind and goal_created_step on goal events.
    for r in events:
        if not isinstance(r, dict):
            continue
        if str(r.get("event_kind") or "") not in ("goal_attempt", "goal_skipped"):
            continue
        if "goal_kind" not in r or "goal_created_step" not in r:
            _fail("ERROR: missing goal_kind/goal_created_step fields in goals_v78_events.jsonl")

    # Curve from events should match compression_curve.json.
    points = _points_from_events(events)
    if len(points) != 2:
        _fail(f"ERROR: expected 2 curve points, got={len(points)}")
    if not (int(points[0]["steps_after"]) < int(points[0]["steps_before"])):
        _fail("ERROR: curve[0] not decreasing")
    if not (int(points[1]["steps_after"]) < int(points[1]["steps_before"])):
        _fail("ERROR: curve[1] not decreasing")
    if not (int(points[1]["steps_after"]) <= int(points[0]["steps_after"])):
        _fail("ERROR: curve not non-increasing across promotions")

    # Expected curve: 4->2 and 2->1 (allow strict decrease if not exact).
    if not (int(points[0]["steps_before"]) == 4 and int(points[0]["steps_after"]) == 2):
        if not (int(points[0]["steps_after"]) < int(points[0]["steps_before"])):
            _fail("ERROR: promo0 curve unexpected and not strictly decreasing")
    if int(points[1]["steps_after"]) != 1:
        if not (int(points[1]["steps_after"]) < int(points[1]["steps_before"])):
            _fail("ERROR: promo1 curve unexpected and not strictly decreasing")

    curve = json.load(open(curve_path, "r", encoding="utf-8"))
    curve_points = curve.get("points") if isinstance(curve, dict) else None
    if not isinstance(curve_points, list) or len(curve_points) != 2:
        _fail("ERROR: compression_curve.json missing/invalid points")
    if curve_points != points:
        _fail("ERROR: compression_curve.json points mismatch vs derived points")

    # Transfer assertions:
    cand0 = json.load(open(cand0_act, "r", encoding="utf-8"))
    cand1 = json.load(open(cand1_act, "r", encoding="utf-8"))
    out_keys0 = _iface_output_keys_from_act_json(cand0 if isinstance(cand0, dict) else {})
    out_keys1 = _iface_output_keys_from_act_json(cand1 if isinstance(cand1, dict) else {})

    if "sum" not in out_keys0 or "out" in out_keys0:
        _fail(f"ERROR: candidate_000_act output_schema must contain 'sum' and NOT 'out', got={out_keys0}")
    if "out" not in out_keys1:
        _fail(f"ERROR: candidate_001_act output_schema must contain 'out', got={out_keys1}")

    # At least one fmt_total goal attempt AFTER promo0 with steps_total==2.
    first_promo_idx = None
    for i, r in enumerate(events):
        if not isinstance(r, dict):
            continue
        if str(r.get("event_kind") or "") == "promotion_attempt" and str(r.get("decision") or "") == "promoted":
            first_promo_idx = int(i)
            break
    if first_promo_idx is None:
        _fail("ERROR: could not locate first promoted promotion event")
    found_total_after = False
    for r in events[first_promo_idx + 1 :]:
        if not isinstance(r, dict):
            continue
        if str(r.get("event_kind") or "") != "goal_attempt":
            continue
        if str(r.get("goal_kind") or "") != "v78_fmt_total":
            continue
        if int(r.get("steps_total", 0) or 0) == 2:
            found_total_after = True
            break
    if not found_total_after:
        _fail("ERROR: expected a v78_fmt_total goal_attempt after promo0 with steps_total==2")

    cert_sigs: List[str] = [str(r.get("certificate_sig") or "") for r in promo_events]
    if len(cert_sigs) != 2 or any(not s for s in cert_sigs):
        _fail("ERROR: missing certificate_sig(s) in promotion events")

    artifacts = {
        "goals_v78_events_jsonl_sha256": sha256_file(events_path),
        "traces_v78_json_sha256": sha256_file(traces_path),
        "mined_candidates_v78_json_sha256": sha256_file(mined_path),
        "v78_promotions_jsonl_sha256": sha256_file(promos_path),
        "compression_curve_json_sha256": sha256_file(curve_path),
        "candidate_000_act_json_sha256": sha256_file(cand0_act),
        "candidate_000_certificate_v2_json_sha256": sha256_file(cand0_cert),
        "candidate_001_act_json_sha256": sha256_file(cand1_act),
        "candidate_001_certificate_v2_json_sha256": sha256_file(cand1_cert),
    }

    return {
        "schema_version": 1,
        "seed": int(seed),
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
    ap.add_argument("--out_base", default="results/run_smoke_goal_act_agent_loop_v78_transfer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_base = str(args.out_base)
    seed = int(args.seed)

    results: Dict[str, Any] = {"seed": seed, "tries": {}}
    sigs: List[Tuple[Tuple[str, ...], Tuple[int, ...], int, str]] = []
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

        core = {
            "seed": int(seed),
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

        sigs.append((certs, pts_sig, int(ev.get("promoted_total", 0) or 0), str(eval_sha)))
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
    }
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
