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
from atos_core.agent_loop_goals_v76 import run_goals_v76
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
        match={},
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


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    store = ActStore()

    # Base micro-world (same semantics as V75).
    normalize_x_id = "concept_v72_normalize_x_v0"
    normalize_y_id = "concept_v72_normalize_y_v0"
    add_nx_ny_id = "concept_v72_add_nx_ny_v0"

    store.add(
        make_concept_act(
            act_id=normalize_x_id,
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

    # Four persistent goal_v75 acts.
    goals_specs = [
        {"bindings": {"x": "0004", "y": "0008"}, "expected": "12"},
        {"bindings": {"x": "0010", "y": "0003"}, "expected": "13"},
        {"bindings": {"x": "0007", "y": "0005"}, "expected": "12"},
        {"bindings": {"x": "0002", "y": "0009"}, "expected": "11"},
    ]
    for i, gs in enumerate(goals_specs):
        ga = make_goal_act_v75(
            goal_kind="v76_sum_norm",
            bindings=dict(gs["bindings"]),
            output_key="sum",
            expected=gs["expected"],
            validator_id="text_exact",
            created_step=int(i),
        )
        store.add(ga)

    # Online promotion inside loop, max 1 goal per round to force in-run improvement.
    loop_res = run_goals_v76(
        store=store,
        seed=int(seed),
        out_dir=out_dir,
        max_rounds=6,
        max_goals_per_round=1,
        enable_promotion=True,
        promotion_budget_bits=1024,
        promotion_min_traces=2,
        promotion_top_k=8,
    )

    if int(loop_res.get("goals_total", 0) or 0) != 4:
        _fail(f"ERROR: expected goals_total == 4, got={loop_res.get('goals_total')}")
    if int(loop_res.get("goals_satisfied", 0) or 0) != 4:
        _fail(f"ERROR: expected goals_satisfied == 4, got={loop_res.get('goals_satisfied')}")
    if int(loop_res.get("promoted_total", 0) or 0) != 1:
        _fail(f"ERROR: expected promoted_total == 1, got={loop_res.get('promoted_total')}")

    events_path = os.path.join(out_dir, "goals_v76_events.jsonl")
    promos_path = os.path.join(out_dir, "promotion", "v76_promotions.jsonl")
    mined_path = os.path.join(out_dir, "mined_candidates_v76.json")
    traces_path = os.path.join(out_dir, "traces_v76.json")
    cand_act_path = os.path.join(out_dir, "candidates", "candidate_000_act.json")
    cand_cert_path = os.path.join(out_dir, "candidates", "candidate_000_certificate_v2.json")

    for p in (events_path, promos_path, mined_path, traces_path, cand_act_path, cand_cert_path):
        if not os.path.exists(p):
            _fail(f"ERROR: missing required artifact: {p}")

    promos = _read_jsonl(promos_path)
    promoted = [r for r in promos if isinstance(r, dict) and str(r.get("decision") or "") == "promoted"]
    if len(promoted) != 1:
        _fail(f"ERROR: expected exactly 1 promoted row, got={len(promoted)}")
    certificate_sig = str(promoted[0].get("certificate_sig") or "")
    candidate_id = str(promoted[0].get("candidate_id") or "")

    events = _read_jsonl(events_path)
    promo_idx = None
    for i, r in enumerate(events):
        if not isinstance(r, dict):
            continue
        if str(r.get("event_kind") or "") == "promotion_attempt" and str(r.get("decision") or "") == "promoted":
            promo_idx = i
            break
    if promo_idx is None:
        _fail("ERROR: missing promotion_attempt promoted event in goals_v76_events.jsonl")

    steps_before = None
    for r in events[: int(promo_idx)]:
        if not isinstance(r, dict):
            continue
        if str(r.get("event_kind") or "") == "goal_attempt":
            steps_before = int(r.get("steps_total", 0) or 0)
            break
    steps_after = None
    for r in events[int(promo_idx) + 1 :]:
        if not isinstance(r, dict):
            continue
        if str(r.get("event_kind") or "") == "goal_attempt":
            steps_after = int(r.get("steps_total", 0) or 0)
            break
    if steps_before is None or steps_after is None:
        _fail("ERROR: could not derive steps_before/steps_after around promotion event")
    if int(steps_after) >= int(steps_before):
        _fail(f"ERROR: expected in-run compression: before={steps_before} after={steps_after}")

    # Eval payload must be deterministic and must not include paths.
    artifacts = {
        "goals_v76_events_jsonl_sha256": sha256_file(events_path),
        "traces_v76_json_sha256": sha256_file(traces_path),
        "mined_candidates_v76_json_sha256": sha256_file(mined_path),
        "v76_promotions_jsonl_sha256": sha256_file(promos_path),
        "candidate_000_act_json_sha256": sha256_file(cand_act_path),
        "candidate_000_certificate_v2_json_sha256": sha256_file(cand_cert_path),
    }
    eval_obj = {
        "schema_version": 1,
        "seed": int(seed),
        "goals_total": int(loop_res.get("goals_total", 0) or 0),
        "goals_satisfied": int(loop_res.get("goals_satisfied", 0) or 0),
        "promoted_total": int(loop_res.get("promoted_total", 0) or 0),
        "budget_bits": int(loop_res.get("budget_bits", 0) or 0),
        "used_bits": int(loop_res.get("used_bits", 0) or 0),
        "steps_before": int(steps_before),
        "steps_after": int(steps_after),
        "delta_steps": int(int(steps_before) - int(steps_after)),
        "candidate_id": str(candidate_id),
        "certificate_sig": str(certificate_sig),
        "artifacts": dict(artifacts),
    }
    return eval_obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", default="results/run_smoke_goal_act_agent_loop_v76")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_base = str(args.out_base)
    seed = int(args.seed)

    results: Dict[str, Any] = {"seed": seed, "tries": {}}
    sigs: List[Tuple[str, int, int, int]] = []
    summary_shas: List[str] = []

    for t in (1, 2):
        out_dir = f"{out_base}_try{t}"
        ensure_absent(out_dir)

        ev = smoke_try(out_dir=out_dir, seed=seed)
        eval_path = os.path.join(out_dir, "eval.json")
        eval_sha = write_json(eval_path, ev)

        core = {
            "seed": int(seed),
            "steps_before": int(ev.get("steps_before", 0) or 0),
            "steps_after": int(ev.get("steps_after", 0) or 0),
            "delta_steps": int(ev.get("delta_steps", 0) or 0),
            "promoted_total": int(ev.get("promoted_total", 0) or 0),
            "certificate_sig": str(ev.get("certificate_sig") or ""),
            "sha256_eval_json": str(eval_sha),
        }
        summary_sha = sha256_text(canonical_json_dumps(core))
        smoke = {"summary": core, "determinism": {"summary_sha256": str(summary_sha)}}
        smoke_path = os.path.join(out_dir, "smoke_summary.json")
        smoke_sha = write_json(smoke_path, smoke)

        sigs.append((str(core["certificate_sig"]), int(core["steps_before"]), int(core["steps_after"]), int(core["promoted_total"])))
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
        "certificate_sig": sigs[0][0],
        "steps_before": sigs[0][1],
        "steps_after": sigs[0][2],
        "promoted_total": sigs[0][3],
    }
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

