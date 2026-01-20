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
from atos_core.replan_law_v117 import (
    PlanCandidateV117,
    REPLAN_REASON_EXHAUSTED_PLANS_V117,
    REPLAN_REASON_OK_V117,
    REPLAN_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V117,
    ReplanResultV117,
    replan_until_satisfies_v117,
    write_replan_trace_v117,
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
        subgraph={"schema_version": 117, "kind": "turn_obs_v117"},
        slots={},
        bindings={},
        cost=0.0,
        evidence_refs=[],
        invariants={"schema_version": 117, "obs_kind": "turn_v117"},
        created_step=int(step),
        last_step=int(step),
    )


def _make_goal(goal_id: str, *, turn_id: str, step: int) -> ATOv71:
    return ATOv71(
        ato_id=str(goal_id),
        ato_type="GOAL",
        subgraph={"schema_version": 117, "kind": "goal_demo_v117", "turn_id": str(turn_id)},
        slots={},
        bindings={},
        cost=1.0,
        evidence_refs=[{"kind": "turn", "turn_id": str(turn_id)}],
        invariants={"schema_version": 117, "goal_kind": "demo"},
        created_step=int(step),
        last_step=int(step),
    )


def _make_plan(plan_id: str, *, turn_id: str, step: int) -> ATOv71:
    return ATOv71(
        ato_id=str(plan_id),
        ato_type="PLAN",
        subgraph={"schema_version": 117, "kind": "plan_demo_v117", "turn_id": str(turn_id)},
        slots={},
        bindings={},
        cost=1.0,
        evidence_refs=[{"kind": "turn", "turn_id": str(turn_id)}],
        invariants={"schema_version": 117, "plan_kind": "demo"},
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
        "schema_version": 117,
        "conversation_id": str(conversation_id),
        "user_turn_id": str(user_turn_id),
        "goal_ato_id": str(goal_ato_id),
        "plan_ato_id": str(plan_ato_id),
        "reason_code": str(reason_code),
        "evidence": dict(evidence),
    }
    fail_id = "fail_event_v117_" + sha256_hex(canonical_json_dumps(sem).encode("utf-8"))
    return ATOv71(
        ato_id=str(fail_id),
        ato_type="EVAL",
        subgraph=dict(sem, satisfies=False),
        slots={},
        bindings={},
        cost=0.0,
        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
        invariants={"schema_version": 117, "eval_kind": "FAIL_EVENT_V117"},
        created_step=int(step),
        last_step=int(step),
    )


def _write_final_response_v117(path: Path, *, ok: bool, reason: str) -> Dict[str, Any]:
    fail_text = ""
    if not ok:
        fail_text = render_fail_response_v115(str(reason))
    obj = {
        "schema_version": 117,
        "kind": "final_response_v117",
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
    goal_id = "goal_demo_v117"
    turn_id = "turn_user_0"
    conversation_id = "conv_demo_v117"
    step0 = 0

    plans: List[PlanCandidateV117] = [
        PlanCandidateV117(plan_id="planA", plan_cost=1.0, plan_sem_sig={"p": "A"}),
        PlanCandidateV117(plan_id="planB", plan_cost=1.0, plan_sem_sig={"p": "B"}),
    ]

    def next_plan() -> Optional[PlanCandidateV117]:
        return plans.pop(0) if plans else None

    def exec_plan(p: PlanCandidateV117):
        if p.plan_id == "planA":
            return False, True, "planA_failed"
        return True, True, ""

    res: ReplanResultV117 = replan_until_satisfies_v117(next_plan=next_plan, exec_plan=exec_plan, max_attempts=8)
    if not res.ok:
        _fail("case_first_fails_second_passes_expected_ok")

    trace_obj = write_replan_trace_v117(path=str(case_dir / "replan_trace_v117.json"), goal_id=str(goal_id), result=res)
    _write_final_response_v117(case_dir / "final_response_v117.json", ok=True, reason="ok")

    mg = MindGraphV71(run_dir=str(case_dir / "mind_graph_v117"))
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

    return {"ok": True, "reason": REPLAN_REASON_OK_V117, "attempts_total": int(res.attempts_total), "trace_sig": str(trace_obj.get("trace_sig") or "")}


def _run_case_exhausted(case_dir: Path) -> Dict[str, Any]:
    _ensure_absent(case_dir)
    case_dir.mkdir(parents=True, exist_ok=False)
    goal_id = "goal_demo_exhausted_v117"
    turn_id = "turn_user_0"
    conversation_id = "conv_demo_v117"
    step0 = 0

    plans: List[PlanCandidateV117] = [
        PlanCandidateV117(plan_id="planA", plan_cost=1.0, plan_sem_sig={"p": "A"}),
    ]

    def next_plan() -> Optional[PlanCandidateV117]:
        return plans.pop(0) if plans else None

    def exec_plan(_p: PlanCandidateV117):
        return False, True, "failed"

    res = replan_until_satisfies_v117(next_plan=next_plan, exec_plan=exec_plan, max_attempts=8)
    if res.ok or res.reason != REPLAN_REASON_EXHAUSTED_PLANS_V117:
        _fail("case_exhausted_expected_exhausted_plans")

    trace_obj = write_replan_trace_v117(path=str(case_dir / "replan_trace_v117.json"), goal_id=str(goal_id), result=res)
    _write_final_response_v117(case_dir / "final_response_v117.json", ok=False, reason=REPLAN_REASON_EXHAUSTED_PLANS_V117)

    mg = MindGraphV71(run_dir=str(case_dir / "mind_graph_v117"))
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

    return {"ok": False, "reason": REPLAN_REASON_EXHAUSTED_PLANS_V117, "attempts_total": int(res.attempts_total), "trace_sig": str(trace_obj.get("trace_sig") or "")}


def _run_case_budget(case_dir: Path) -> Dict[str, Any]:
    _ensure_absent(case_dir)
    case_dir.mkdir(parents=True, exist_ok=False)
    goal_id = "goal_demo_budget_v117"
    turn_id = "turn_user_0"
    conversation_id = "conv_demo_v117"
    step0 = 0

    plans: List[PlanCandidateV117] = [
        PlanCandidateV117(plan_id="planA", plan_cost=1.0, plan_sem_sig={"p": "A"}),
        PlanCandidateV117(plan_id="planB", plan_cost=1.0, plan_sem_sig={"p": "B"}),
        PlanCandidateV117(plan_id="planC", plan_cost=1.0, plan_sem_sig={"p": "C"}),
        PlanCandidateV117(plan_id="planD", plan_cost=1.0, plan_sem_sig={"p": "D"}),
    ]

    def next_plan() -> Optional[PlanCandidateV117]:
        return plans.pop(0) if plans else None

    def exec_plan(_p: PlanCandidateV117):
        return False, True, "failed"

    res = replan_until_satisfies_v117(next_plan=next_plan, exec_plan=exec_plan, max_attempts=3)
    if res.ok or res.reason != REPLAN_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V117:
        _fail("case_budget_expected_budget_exhausted")

    trace_obj = write_replan_trace_v117(path=str(case_dir / "replan_trace_v117.json"), goal_id=str(goal_id), result=res)
    _write_final_response_v117(case_dir / "final_response_v117.json", ok=False, reason=REPLAN_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V117)

    mg = MindGraphV71(run_dir=str(case_dir / "mind_graph_v117"))
    mg.add_node(step=step0, ato=_make_turn_obs(turn_id, step=step0), reason="turn_obs")
    mg.add_node(step=step0, ato=_make_goal(goal_id, turn_id=turn_id, step=step0), reason="goal")
    mg.add_node(step=step0, ato=_make_plan("planA", turn_id=turn_id, step=step0), reason="plan")
    mg.add_node(step=step0, ato=_make_plan("planB", turn_id=turn_id, step=step0), reason="plan")
    mg.add_node(step=step0, ato=_make_plan("planC", turn_id=turn_id, step=step0), reason="plan")

    for i, plan_id in enumerate(["planA", "planB", "planC"]):
        fail = _make_fail_event(
            conversation_id=conversation_id,
            user_turn_id=turn_id,
            goal_ato_id=goal_id,
            plan_ato_id=str(plan_id),
            reason_code="failed",
            step=step0,
            evidence={"trace_sig": str(trace_obj.get("trace_sig") or ""), "attempt_index": int(i)},
        )
        mg.add_node(step=step0, ato=fail, reason="fail_event")
        mg.add_edge(step=step0, src_ato_id=str(fail.ato_id), dst_ato_id=str(goal_id), edge_type="DERIVED_FROM", evidence_refs=[], reason="fail->goal")
        mg.add_edge(step=step0, src_ato_id=str(fail.ato_id), dst_ato_id=str(plan_id), edge_type="DERIVED_FROM", evidence_refs=[], reason="fail->plan")
        mg.add_edge(step=step0, src_ato_id=str(fail.ato_id), dst_ato_id=str(turn_id), edge_type="DERIVED_FROM", evidence_refs=[], reason="fail->turn")

    chains = mg.verify_chains()
    if not bool(chains.get("mind_nodes_chain_ok")) or not bool(chains.get("mind_edges_chain_ok")):
        _fail("case_budget_mind_graph_chain_fail")

    return {"ok": False, "reason": REPLAN_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V117, "attempts_total": int(res.attempts_total), "trace_sig": str(trace_obj.get("trace_sig") or "")}


def _run_try(out_dir: Path) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    cases = {
        "case_first_fails_second_passes": _run_case_first_fails_second_passes(out_dir / "case_00_first_fails_second_passes"),
        "case_exhausted_plans": _run_case_exhausted(out_dir / "case_01_exhausted_plans"),
        "case_budget_exhausted": _run_case_budget(out_dir / "case_02_budget_exhausted"),
    }
    eval_obj = {"schema_version": 117, "kind": "smoke_eval_v117", "cases": dict(cases)}
    _write_once_json(out_dir / "eval.json", eval_obj)
    eval_sha256 = _sha256_file(out_dir / "eval.json")
    summary = {"schema_version": 117, "kind": "smoke_summary_v117", "eval_sha256": str(eval_sha256)}
    _write_once_json(out_dir / "summary.json", summary)
    fail_catalog = {"schema_version": 117, "failures_total": 0, "failures": []}
    _write_once_json(out_dir / "fail_catalog_v117.json", fail_catalog)
    return {"eval_json": eval_obj, "eval_sha256": str(eval_sha256)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", default="results/run_smoke_v117_replan_law")
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()

    # Seed is unused in this deterministic smoke, but kept for CLI consistency.
    _ = int(args.seed)
    out_base = Path(str(args.out_base))
    out1 = Path(str(out_base) + "_try1")
    out2 = Path(str(out_base) + "_try2")

    r1 = _run_try(out_dir=out1)
    r2 = _run_try(out_dir=out2)

    if canonical_json_dumps(r1["eval_json"]) != canonical_json_dumps(r2["eval_json"]):
        _fail("determinism_failed:eval_json")
    if str(r1["eval_sha256"]) != str(r2["eval_sha256"]):
        _fail("determinism_failed:eval_sha256")

    core = {"schema_version": 117, "eval_sha256": str(r1["eval_sha256"])}
    summary_sha256 = sha256_hex(canonical_json_dumps(core).encode("utf-8"))
    print(
        json.dumps(
            {
                "ok": True,
                "determinism_ok": True,
                "summary_sha256": str(summary_sha256),
                "try1_dir": str(out1),
                "try2_dir": str(out2),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
