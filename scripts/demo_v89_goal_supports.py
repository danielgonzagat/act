#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction, deterministic_iso
from atos_core.agent_loop_goals_v89 import run_goals_v89
from atos_core.goal_act_v75 import make_goal_act_v75
from atos_core.objective_v88 import make_objective_eq_text_act_v88
from atos_core.store import ActStore


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _read_events(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def make_concept_act(
    *,
    act_id: str,
    input_schema: Dict[str, str],
    output_schema: Dict[str, str],
    program: List[Instruction],
    supports_goals_v89: List[Dict[str, Any]],
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
                "validator_id": "",
            },
            "supports_goals_v89": list(supports_goals_v89),
        },
        cost={},
        deps=[],
        active=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ticks", type=int, default=10)
    args = ap.parse_args()

    out_dir = str(args.out_dir)
    seed = int(args.seed)
    ticks = int(args.ticks)

    if os.path.exists(out_dir):
        _fail(f"ERROR: out_dir already exists (WORM): {out_dir}")

    store = ActStore()

    # Objective: equality between __output and expected.
    obj = make_objective_eq_text_act_v88(created_step=0, match={})
    store.add(obj)

    # Two goals, validated by the objective (opt-in V88 field).
    goal_sum = make_goal_act_v75(
        goal_kind="v89_sum",
        bindings={"x": "0004", "y": "0008"},
        output_key="out",
        expected="SUM=12",
        validator_id="text_exact",
        created_step=0,
    )
    goal_sum.evidence["goal"]["objective_act_id"] = str(obj.id)
    goal_sum.evidence["goal"]["urgency_weight"] = 2.0
    store.add(goal_sum)

    goal_total = make_goal_act_v75(
        goal_kind="v89_total",
        bindings={"x": "0010", "y": "0002"},
        output_key="out",
        expected="TOTAL=12",
        validator_id="text_exact",
        created_step=1,
    )
    goal_total.evidence["goal"]["objective_act_id"] = str(obj.id)
    goal_total.evidence["goal"]["urgency_weight"] = 1.0
    store.add(goal_total)

    # Concepts that "support" each goal via supports_goals_v89 metadata (no executor changes).
    # sum_wrong: returns SUM=(x+y+1) to force failures and posterior update.
    sum_wrong_id = "concept_v89_sum_wrong_v0"
    sum_ok_id = "concept_v89_sum_ok_v0"
    total_ok_id = "concept_v89_total_ok_v0"

    supports_sum_wrong = [
        {
            "goal_id": str(goal_sum.id),
            "prior_success": 0.9,
            "prior_strength": 2,
            "prior_cost": 1.0,
            "note": "optimistic prior, but actually wrong",
        }
    ]
    supports_sum_ok = [
        {"goal_id": str(goal_sum.id), "prior_success": 0.5, "prior_strength": 2, "prior_cost": 1.0, "note": ""}
    ]
    supports_total_ok = [
        {"goal_id": str(goal_total.id), "prior_success": 0.5, "prior_strength": 2, "prior_cost": 1.0, "note": ""}
    ]

    def _sum_program(*, prefix: str, add_one: bool) -> List[Instruction]:
        prog: List[Instruction] = [
            Instruction("CSV_GET_INPUT", {"name": "x", "out": "x"}),
            Instruction("CSV_GET_INPUT", {"name": "y", "out": "y"}),
            Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["x"], "out": "dx"}),
            Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["y"], "out": "dy"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["dx"], "out": "ix"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["dy"], "out": "iy"}),
            Instruction("CSV_PRIMITIVE", {"fn": "add_int", "in": ["ix", "iy"], "out": "s0"}),
        ]
        if add_one:
            prog += [
                Instruction("CSV_CONST", {"value": 1, "out": "one"}),
                Instruction("CSV_PRIMITIVE", {"fn": "add_int", "in": ["s0", "one"], "out": "s1"}),
                Instruction("CSV_PRIMITIVE", {"fn": "int_to_digits", "in": ["s1"], "out": "sum"}),
            ]
        else:
            prog += [
                Instruction("CSV_PRIMITIVE", {"fn": "int_to_digits", "in": ["s0"], "out": "sum"}),
            ]
        prog += [
            Instruction("CSV_CONST", {"value": str(prefix), "out": "p"}),
            Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["p", "sum"], "out": "out"}),
            Instruction("CSV_RETURN", {"var": "out"}),
        ]
        return prog

    store.add(
        make_concept_act(
            act_id=sum_wrong_id,
            input_schema={"x": "str", "y": "str"},
            output_schema={"out": "str"},
            program=_sum_program(prefix="SUM=", add_one=True),
            supports_goals_v89=supports_sum_wrong,
        )
    )
    store.add(
        make_concept_act(
            act_id=sum_ok_id,
            input_schema={"x": "str", "y": "str"},
            output_schema={"out": "str"},
            program=_sum_program(prefix="SUM=", add_one=False),
            supports_goals_v89=supports_sum_ok,
        )
    )
    store.add(
        make_concept_act(
            act_id=total_ok_id,
            input_schema={"x": "str", "y": "str"},
            output_schema={"out": "str"},
            program=_sum_program(prefix="TOTAL=", add_one=False),
            supports_goals_v89=supports_total_ok,
        )
    )

    summary = run_goals_v89(store=store, seed=int(seed), out_dir=str(out_dir), max_ticks=int(ticks))
    events_path = os.path.join(out_dir, "goals_v89_events.jsonl")
    rows = _read_events(events_path)

    # Print a compact deterministic trace: (goal_selected -> evidence) pairs.
    trace_lines: List[Dict[str, Any]] = []
    for r in rows:
        kind = str(r.get("kind") or "")
        if kind == "goal_support_selected_v89":
            p = r.get("payload") if isinstance(r.get("payload"), dict) else {}
            trace_lines.append(
                {
                    "step": int(r.get("step", 0) or 0),
                    "goal_id": str(p.get("goal_id") or ""),
                    "concept_key": str(p.get("concept_key") or ""),
                    "score": float(p.get("score", 0.0) or 0.0),
                }
            )
        elif kind == "goal_support_evidence_v89":
            trace_lines.append(
                {
                    "step": int(r.get("step", 0) or 0),
                    "goal_id": str(r.get("goal_id") or ""),
                    "concept_key": str(r.get("concept_key") or ""),
                    "ok": bool(r.get("ok", False)),
                    "cost_used": float(r.get("cost_used", 0.0) or 0.0),
                    "note": str(r.get("note") or ""),
                }
            )

    out = {
        "ok": True,
        "out_dir": str(out_dir),
        "summary": {
            "goals_total": int(summary.get("goals_total", 0) or 0),
            "goals_satisfied": int(summary.get("goals_satisfied", 0) or 0),
            "goals_abandoned": int(summary.get("goals_abandoned", 0) or 0),
            "chains_ok": bool(summary.get("chains_ok", False)),
        },
        "trace": list(trace_lines),
        "sha256": summary.get("sha256") if isinstance(summary.get("sha256"), dict) else {},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

