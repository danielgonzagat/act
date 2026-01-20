#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.ato_v71 import ATOv71
from atos_core.goal_persistence_v115 import render_fail_response_v115
from atos_core.mind_graph_v71 import MindGraphV71
from atos_core.replan_law_v118 import (
    PlanCandidateV118,
    REPLAN_REASON_EXHAUSTED_PLANS_V118,
    REPLAN_REASON_OK_V118,
    REPLAN_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V118,
    ReplanResultV118,
    replan_until_satisfies_v118,
    write_replan_trace_v118,
)


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ensure_absent(path: Path) -> None:
    if path.exists():
        _fail(f"worm_exists:{path}")


def _write_once_json(path: Path, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        _fail(f"tmp_exists:{tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _make_turn_obs(turn_id: str, *, step: int) -> ATOv71:
    return ATOv71(
        ato_id=str(turn_id),
        ato_type="OBS",
        subgraph={"schema_version": 118, "kind": "turn_obs_v118"},
        slots={},
        bindings={},
        cost=0.0,
        evidence_refs=[],
        invariants={"schema_version": 118, "obs_kind": "turn_v118"},
        created_step=int(step),
        last_step=int(step),
    )


def _make_goal(goal_id: str, *, turn_id: str, step: int) -> ATOv71:
    return ATOv71(
        ato_id=str(goal_id),
        ato_type="GOAL",
        subgraph={"schema_version": 118, "kind": "goal_demo_v118", "turn_id": str(turn_id)},
        slots={},
        bindings={},
        cost=1.0,
        evidence_refs=[{"kind": "turn", "turn_id": str(turn_id)}],
        invariants={"schema_version": 118, "goal_kind": "demo"},
        created_step=int(step),
        last_step=int(step),
    )


def _make_plan(plan_id: str, *, turn_id: str, step: int) -> ATOv71:
    return ATOv71(
        ato_id=str(plan_id),
        ato_type="PLAN",
        subgraph={"schema_version": 118, "kind": "plan_demo_v118", "turn_id": str(turn_id)},
        slots={},
        bindings={},
        cost=1.0,
        evidence_refs=[{"kind": "turn", "turn_id": str(turn_id)}],
        invariants={"schema_version": 118, "plan_kind": "demo"},
        created_step=int(step),
        last_step=int(step),
    )


def _make_fail_event(
    *,
    conversation_id: str,
    user_turn_id: str,
    goal_ato_id: str,
    plan_ato_id: str,
    reason_code: str,
    step: int,
    evidence: Dict[str, Any],
) -> ATOv71:
    sem = {
        "schema_version": 118,
        "conversation_id": str(conversation_id),
        "user_turn_id": str(user_turn_id),
        "goal_ato_id": str(goal_ato_id),
        "plan_ato_id": str(plan_ato_id),
        "reason_code": str(reason_code),
        "evidence": dict(evidence),
    }
    fail_id = "fail_event_v118_" + sha256_hex(canonical_json_dumps(sem).encode("utf-8"))
    return ATOv71(
        ato_id=str(fail_id),
        ato_type="EVAL",
        subgraph=dict(sem, satisfies=False),
        slots={},
        bindings={},
        cost=0.0,
        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
        invariants={"schema_version": 118, "eval_kind": "FAIL_EVENT_V118"},
        created_step=int(step),
        last_step=int(step),
    )


def _write_final_response_v118(path: Path, *, ok: bool, reason: str) -> Dict[str, Any]:
    fail_text = ""
    if not ok:
        fail_text = render_fail_response_v115(str(reason))
    obj = {
        "schema_version": 118,
        "kind": "final_response_v118",
        "ok": bool(ok),
        "reason": str(reason if not ok else "ok"),
        "fail_response_text": str(fail_text),
    }
    obj["final_sig"] = sha256_hex(canonical_json_dumps(obj).encode("utf-8"))
    _write_once_json(path, obj)
    return dict(obj)


def _run_case_first_fails_second_passes(case_dir: Path) -> Dict[str, Any]:
    _ensure_absent(case_dir)
    case_dir.mkdir(parents=True, exist_ok=False)
    goal_id = "goal_demo_v118"
    turn_id = "turn_user_0"
    conversation_id = "conv_demo_v118"
    step0 = 0

    plans: List[PlanCandidateV118] = [
        PlanCandidateV118(plan_id="planA", plan_cost=1.0, plan_sem_sig={"p": "A"}),
        PlanCandidateV118(plan_id="planB", plan_cost=1.0, plan_sem_sig={"p": "B"}),
    ]

    def next_plan() -> Optional[PlanCandidateV118]:
        return plans.pop(0) if plans else None

    def exec_plan(p: PlanCandidateV118):
        if p.plan_id == "planA":
            return False, True, "planA_failed"
        return True, True, ""

    res: ReplanResultV118 = replan_until_satisfies_v118(next_plan=next_plan, exec_plan=exec_plan, max_attempts=8)
    if not res.ok:
        _fail("case_first_fails_second_passes_expected_ok")

    trace_obj = write_replan_trace_v118(path=str(case_dir / "replan_trace_v118.json"), goal_id=str(goal_id), result=res)
    _write_final_response_v118(case_dir / "final_response_v118.json", ok=True, reason="ok")

    mg = MindGraphV71(run_dir=str(case_dir / "mind_graph_v118"))
    mg.add_node(step=step0, ato=_make_turn_obs(turn_id, step=step0), reason="turn_obs")
    mg.add_node(step=step0, ato=_make_goal(goal_id, turn_id=turn_id, step=step0), reason="goal")
    mg.add_node(step=step0, ato=_make_plan("planA", turn_id=turn_id, step=step0), reason="plan")
    mg.add_node(step=step0, ato=_make_plan("planB", turn_id=turn_id, step=step0), reason="plan")

    # One FAIL_EVENT for planA failure.
    fail = _make_fail_event(
        conversation_id=conversation_id,
        user_turn_id=turn_id,
        goal_ato_id=goal_id,
        plan_ato_id="planA",
        reason_code="planA_failed",
        step=step0,
        evidence={"trace_sig": str(trace_obj.get("trace_sig") or ""), "attempt_index": 0},
    )
    mg.add_node(step=step0, ato=fail, reason="fail_event")
    mg.add_edge(step=step0, src_ato_id=str(fail.ato_id), dst_ato_id=str(goal_id), edge_type="DERIVED_FROM", evidence_refs=[], reason="fail->goal")
    mg.add_edge(step=step0, src_ato_id=str(fail.ato_id), dst_ato_id="planA", edge_type="DERIVED_FROM", evidence_refs=[], reason="fail->plan")
    mg.add_edge(step=step0, src_ato_id=str(fail.ato_id), dst_ato_id=str(turn_id), edge_type="DERIVED_FROM", evidence_refs=[], reason="fail->turn")

    chains = mg.verify_chains()
    if not bool(chains.get("mind_nodes_chain_ok")) or not bool(chains.get("mind_edges_chain_ok")):
        _fail("case_first_fails_second_passes_mind_graph_chain_fail")

    return {"ok": True, "reason": REPLAN_REASON_OK_V118, "attempts_total": int(res.attempts_total), "trace_sig": str(trace_obj.get("trace_sig") or "")}


def _run_case_exhausted(case_dir: Path) -> Dict[str, Any]:
    _ensure_absent(case_dir)
    case_dir.mkdir(parents=True, exist_ok=False)
    goal_id = "goal_demo_exhausted_v118"
    turn_id = "turn_user_0"
    conversation_id = "conv_demo_v118"
    step0 = 0

    plans: List[PlanCandidateV118] = [
        PlanCandidateV118(plan_id="planA", plan_cost=1.0, plan_sem_sig={"p": "A"}),
    ]

    def next_plan() -> Optional[PlanCandidateV118]:
        return plans.pop(0) if plans else None

    def exec_plan(_p: PlanCandidateV118):
        return False, True, "failed"

    res = replan_until_satisfies_v118(next_plan=next_plan, exec_plan=exec_plan, max_attempts=8)
    if res.ok or res.reason != REPLAN_REASON_EXHAUSTED_PLANS_V118:
        _fail("case_exhausted_expected_exhausted_plans")

    trace_obj = write_replan_trace_v118(path=str(case_dir / "replan_trace_v118.json"), goal_id=str(goal_id), result=res)
    _write_final_response_v118(case_dir / "final_response_v118.json", ok=False, reason=REPLAN_REASON_EXHAUSTED_PLANS_V118)

    mg = MindGraphV71(run_dir=str(case_dir / "mind_graph_v118"))
    mg.add_node(step=step0, ato=_make_turn_obs(turn_id, step=step0), reason="turn_obs")
    mg.add_node(step=step0, ato=_make_goal(goal_id, turn_id=turn_id, step=step0), reason="goal")
    mg.add_node(step=step0, ato=_make_plan("planA", turn_id=turn_id, step=step0), reason="plan")

    fail = _make_fail_event(
        conversation_id=conversation_id,
        user_turn_id=turn_id,
        goal_ato_id=goal_id,
        plan_ato_id="planA",
        reason_code="failed",
        step=step0,
        evidence={"trace_sig": str(trace_obj.get("trace_sig") or ""), "attempt_index": 0},
    )
    mg.add_node(step=step0, ato=fail, reason="fail_event")
    mg.add_edge(step=step0, src_ato_id=str(fail.ato_id), dst_ato_id=str(goal_id), edge_type="DERIVED_FROM", evidence_refs=[], reason="fail->goal")
    mg.add_edge(step=step0, src_ato_id=str(fail.ato_id), dst_ato_id="planA", edge_type="DERIVED_FROM", evidence_refs=[], reason="fail->plan")
    mg.add_edge(step=step0, src_ato_id=str(fail.ato_id), dst_ato_id=str(turn_id), edge_type="DERIVED_FROM", evidence_refs=[], reason="fail->turn")
    chains = mg.verify_chains()
    if not bool(chains.get("mind_nodes_chain_ok")) or not bool(chains.get("mind_edges_chain_ok")):
        _fail("case_exhausted_mind_graph_chain_fail")

    return {"ok": False, "reason": REPLAN_REASON_EXHAUSTED_PLANS_V118, "attempts_total": int(res.attempts_total), "trace_sig": str(trace_obj.get("trace_sig") or "")}


def _run_case_budget_exhausted(case_dir: Path) -> Dict[str, Any]:
    _ensure_absent(case_dir)
    case_dir.mkdir(parents=True, exist_ok=False)
    goal_id = "goal_demo_budget_v118"
    turn_id = "turn_user_0"
    conversation_id = "conv_demo_v118"
    step0 = 0

    plans: List[PlanCandidateV118] = []
    for i in range(100):
        plans.append(PlanCandidateV118(plan_id=f"plan{i}", plan_cost=1.0, plan_sem_sig={"p": f"{i}"}))

    def next_plan() -> Optional[PlanCandidateV118]:
        return plans.pop(0) if plans else None

    def exec_plan(_p: PlanCandidateV118):
        return False, True, "failed"

    res = replan_until_satisfies_v118(next_plan=next_plan, exec_plan=exec_plan, max_attempts=3)
    if res.ok or res.reason != REPLAN_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V118 or res.attempts_total != 3:
        _fail("case_budget_exhausted_expected_budget")

    trace_obj = write_replan_trace_v118(path=str(case_dir / "replan_trace_v118.json"), goal_id=str(goal_id), result=res)
    _write_final_response_v118(case_dir / "final_response_v118.json", ok=False, reason=REPLAN_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V118)

    mg = MindGraphV71(run_dir=str(case_dir / "mind_graph_v118"))
    mg.add_node(step=step0, ato=_make_turn_obs(turn_id, step=step0), reason="turn_obs")
    mg.add_node(step=step0, ato=_make_goal(goal_id, turn_id=turn_id, step=step0), reason="goal")
    for i in range(3):
        mg.add_node(step=step0, ato=_make_plan(f"plan{i}", turn_id=turn_id, step=step0), reason="plan")

    # Fail events for each attempt.
    for idx in range(3):
        fail = _make_fail_event(
            conversation_id=conversation_id,
            user_turn_id=turn_id,
            goal_ato_id=goal_id,
            plan_ato_id=f"plan{idx}",
            reason_code="failed",
            step=step0,
            evidence={"trace_sig": str(trace_obj.get("trace_sig") or ""), "attempt_index": int(idx)},
        )
        mg.add_node(step=step0, ato=fail, reason="fail_event")
        mg.add_edge(step=step0, src_ato_id=str(fail.ato_id), dst_ato_id=str(goal_id), edge_type="DERIVED_FROM", evidence_refs=[], reason="fail->goal")
        mg.add_edge(step=step0, src_ato_id=str(fail.ato_id), dst_ato_id=f"plan{idx}", edge_type="DERIVED_FROM", evidence_refs=[], reason="fail->plan")
        mg.add_edge(step=step0, src_ato_id=str(fail.ato_id), dst_ato_id=str(turn_id), edge_type="DERIVED_FROM", evidence_refs=[], reason="fail->turn")

    chains = mg.verify_chains()
    if not bool(chains.get("mind_nodes_chain_ok")) or not bool(chains.get("mind_edges_chain_ok")):
        _fail("case_budget_exhausted_mind_graph_chain_fail")

    return {"ok": False, "reason": REPLAN_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V118, "attempts_total": int(res.attempts_total), "trace_sig": str(trace_obj.get("trace_sig") or "")}


def _compute_case_sha256(case_dir: Path) -> str:
    # Stable hash of key outputs in the case directory.
    paths = [
        case_dir / "final_response_v118.json",
        case_dir / "replan_trace_v118.json",
        case_dir / "mind_graph_v118" / "mind_nodes.jsonl",
        case_dir / "mind_graph_v118" / "mind_edges.jsonl",
    ]
    body: Dict[str, str] = {}
    for p in paths:
        body[str(p.name)] = _sha256_file(p)
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()

    seed = int(args.seed)
    out_base = Path(str(args.out_base))

    out1 = Path(str(out_base) + "_try1")
    out2 = Path(str(out_base) + "_try2")
    for od in [out1, out2]:
        _ensure_absent(od)
        od.parent.mkdir(parents=True, exist_ok=True)
        od.mkdir(parents=True, exist_ok=False)

        # Three deterministic cases.
        c1 = _run_case_first_fails_second_passes(od / "case_first_fails_second_passes")
        c2 = _run_case_exhausted(od / "case_exhausted_plans")
        c3 = _run_case_budget_exhausted(od / "case_budget_exhausted")

        eval_obj = {
            "schema_version": 118,
            "kind": "smoke_eval_v118_replan_law",
            "seed": int(seed),
            "cases": {"first_fails_second_passes": dict(c1), "exhausted_plans": dict(c2), "budget_exhausted": dict(c3)},
            "cases_sha256": {
                "first_fails_second_passes": _compute_case_sha256(od / "case_first_fails_second_passes"),
                "exhausted_plans": _compute_case_sha256(od / "case_exhausted_plans"),
                "budget_exhausted": _compute_case_sha256(od / "case_budget_exhausted"),
            },
        }
        _write_once_json(od / "eval.json", eval_obj)
        eval_sha = _sha256_file(od / "eval.json")
        summary_obj = {
            "schema_version": 118,
            "kind": "smoke_summary_v118_replan_law",
            "seed": int(seed),
            "eval_sha256": str(eval_sha),
        }
        _write_once_json(od / "summary.json", summary_obj)
        fail_catalog = {"schema_version": 118, "kind": "fail_catalog_v118_replan_law", "failures_total": 0, "failures": []}
        _write_once_json(od / "fail_catalog_v118.json", fail_catalog)

    # Determinism check.
    ev1 = json.loads((out1 / "eval.json").read_text(encoding="utf-8"))
    ev2 = json.loads((out2 / "eval.json").read_text(encoding="utf-8"))
    if canonical_json_dumps(ev1) != canonical_json_dumps(ev2):
        _fail("determinism_failed:eval_json")

    core = {
        "schema_version": 118,
        "seed": int(seed),
        "eval_sha256": _sha256_file(out1 / "eval.json"),
    }
    summary_sha256 = sha256_hex(canonical_json_dumps(core).encode("utf-8"))
    out = {
        "ok": True,
        "determinism_ok": True,
        "summary_sha256": str(summary_sha256),
        "try1_dir": str(out1),
        "try2_dir": str(out2),
        "sha256_eval_json": _sha256_file(out1 / "eval.json"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

