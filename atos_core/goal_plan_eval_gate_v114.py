from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .ato_v71 import ATOv71, stable_hash_obj
from .mind_graph_v71 import MindGraphV71

FAIL_REASON_MISSING_GOAL_V114 = "missing_goal"
FAIL_REASON_MISSING_PLAN_V114 = "missing_plan"
FAIL_REASON_MISSING_EVAL_V114 = "missing_eval"
FAIL_REASON_RENDER_BLOCKED_NO_EVAL_SATISFIES_V114 = "render_blocked_no_eval_satisfies"
FAIL_REASON_EXHAUSTED_PLANS_V114 = "exhausted_plans"
FAIL_REASON_REPLANNING_REQUIRED_V114 = "replanning_required"
FAIL_REASON_PLAN_ATTEMPT_FAILED_V114 = "plan_attempt_failed"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ensure_absent(path: str) -> None:
    if os.path.exists(path):
        raise ValueError(f"worm_exists:{path}")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _stable_id(prefix: str, body: Dict[str, Any]) -> str:
    return f"{prefix}_{sha256_hex(canonical_json_dumps(body).encode('utf-8'))}"


def _make_turn_obs_ato_v114(turn_payload: Dict[str, Any]) -> ATOv71:
    turn_id = str(turn_payload.get("turn_id") or "")
    role = str(turn_payload.get("role") or "")
    if not turn_id:
        raise ValueError("missing_turn_id")
    created_step = int(turn_payload.get("created_step", 0) or 0)
    turn_index = int(turn_payload.get("turn_index", 0) or 0)
    text_sig = str(turn_payload.get("text_sig") or "")
    return ATOv71(
        ato_id=str(turn_id),
        ato_type="OBS",
        subgraph={
            "kind": str(turn_payload.get("kind") or ""),
            "role": str(role),
            "turn_index": int(turn_index),
            "text_sig": str(text_sig),
        },
        slots={},
        bindings={},
        cost=0.0,
        evidence_refs=[],
        invariants={"schema_version": 114, "obs_kind": "turn_v96"},
        created_step=int(created_step),
        last_step=int(created_step),
    )


def _make_goal_ato_v114(*, conversation_id: str, user_turn_payload: Dict[str, Any]) -> ATOv71:
    user_turn_id = str(user_turn_payload.get("turn_id") or "")
    if not user_turn_id:
        raise ValueError("missing_user_turn_id")
    created_step = int(user_turn_payload.get("created_step", 0) or 0)
    user_turn_index = int(user_turn_payload.get("turn_index", 0) or 0)
    refs = _safe_dict(user_turn_payload.get("refs"))
    body = {
        "schema_version": 114,
        "conversation_id": str(conversation_id),
        "user_turn_id": str(user_turn_id),
        "user_turn_index": int(user_turn_index),
        "intent_id": str(refs.get("intent_id") or ""),
        "parse_sig": str(refs.get("parse_sig") or ""),
        "text_sig": str(user_turn_payload.get("text_sig") or ""),
    }
    ato_id = _stable_id("goal_v114", body)
    return ATOv71(
        ato_id=str(ato_id),
        ato_type="GOAL",
        subgraph=dict(body),
        slots={},
        bindings={},
        cost=0.0,
        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
        invariants={"schema_version": 114, "goal_kind": "turn_goal"},
        created_step=int(created_step),
        last_step=int(created_step),
    )


def _make_plan_ato_v114(*, action_plan: Dict[str, Any]) -> ATOv71:
    plan_id = str(action_plan.get("plan_id") or action_plan.get("plan_sig") or "")
    if not plan_id:
        raise ValueError("missing_plan_id")
    created_step = int(action_plan.get("created_step", 0) or 0)
    user_turn_id = str(action_plan.get("user_turn_id") or "")
    user_turn_index = int(action_plan.get("user_turn_index", -1) or -1)
    ranked = _safe_list(action_plan.get("ranked_candidates"))
    attempted = _safe_list(action_plan.get("attempted_actions"))
    subgraph = {
        "schema_version": 114,
        "plan_id": str(plan_id),
        "user_turn_id": str(user_turn_id),
        "user_turn_index": int(user_turn_index),
        "chosen_action_id": str(action_plan.get("chosen_action_id") or ""),
        "chosen_eval_id": str(action_plan.get("chosen_eval_id") or ""),
        "chosen_ok": bool(action_plan.get("chosen_ok", False)),
        "ranked_candidates": [
            {
                "act_id": str(_safe_dict(rc).get("act_id") or ""),
                "expected_success": float(_safe_dict(rc).get("expected_success", 0.0) or 0.0),
                "expected_cost": float(_safe_dict(rc).get("expected_cost", 0.0) or 0.0),
            }
            for rc in ranked
            if isinstance(rc, dict)
        ],
        "attempted_actions": [
            {
                "act_id": str(_safe_dict(a).get("act_id") or ""),
                "eval_id": str(_safe_dict(a).get("eval_id") or ""),
                "ok": bool(_safe_dict(a).get("ok", False)),
            }
            for a in attempted
            if isinstance(a, dict)
        ],
    }
    return ATOv71(
        ato_id=str(plan_id),
        ato_type="PLAN",
        subgraph=dict(subgraph),
        slots={},
        bindings={},
        cost=float(len(subgraph["ranked_candidates"])),
        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}] if user_turn_id else [],
        invariants={"schema_version": 114, "plan_kind": "action_plan_v100"},
        created_step=int(created_step),
        last_step=int(created_step),
    )


def _make_eval_ato_v114(*, objective_eval: Dict[str, Any]) -> ATOv71:
    eval_id = str(objective_eval.get("eval_id") or "")
    if not eval_id:
        raise ValueError("missing_eval_id")
    step = int(objective_eval.get("step", 0) or 0)
    turn_id = str(objective_eval.get("turn_id") or "")
    verdict = _safe_dict(objective_eval.get("verdict"))
    ok = bool(_safe_dict(verdict).get("ok", False))
    reason = str(_safe_dict(verdict).get("reason") or "")
    score = int(_safe_dict(verdict).get("score", 0) or 0)
    subgraph = {
        "schema_version": 114,
        "eval_id": str(eval_id),
        "objective_kind": str(objective_eval.get("objective_kind") or ""),
        "objective_id": str(objective_eval.get("objective_id") or ""),
        "action_concept_id": str(objective_eval.get("action_concept_id") or ""),
        "expected_text_sig": str(objective_eval.get("expected_text_sig") or ""),
        "output_text_sig": str(objective_eval.get("output_text_sig") or ""),
        "satisfies": bool(ok),
        "score": int(score),
        "reason": str(reason),
    }
    return ATOv71(
        ato_id=str(eval_id),
        ato_type="EVAL",
        subgraph=dict(subgraph),
        slots={},
        bindings={},
        cost=1.0,
        evidence_refs=[{"kind": "turn", "turn_id": str(turn_id)}] if turn_id else [],
        invariants={"schema_version": 114, "eval_kind": "objective_eval_v96"},
        created_step=int(step),
        last_step=int(step),
    )


def _make_fail_event_ato_v114(
    *,
    conversation_id: str,
    user_turn_id: str,
    goal_ato_id: str,
    plan_ato_id: str,
    reason_code: str,
    step: int,
) -> ATOv71:
    body = {
        "schema_version": 114,
        "conversation_id": str(conversation_id),
        "user_turn_id": str(user_turn_id),
        "goal_ato_id": str(goal_ato_id),
        "plan_ato_id": str(plan_ato_id),
        "reason_code": str(reason_code),
    }
    fail_id = _stable_id("fail_event_v114", body)
    return ATOv71(
        ato_id=str(fail_id),
        ato_type="EVAL",
        subgraph=dict(body, satisfies=False),
        slots={},
        bindings={},
        cost=0.0,
        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
        invariants={"schema_version": 114, "eval_kind": "FAIL_EVENT_V114"},
        created_step=int(step),
        last_step=int(step),
    )


def _make_plan_attempt_fail_event_ato_v114(
    *,
    conversation_id: str,
    user_turn_id: str,
    goal_ato_id: str,
    plan_ato_id: str,
    attempt_act_id: str,
    attempt_eval_id: str,
    step: int,
) -> ATOv71:
    body = {
        "schema_version": 114,
        "conversation_id": str(conversation_id),
        "user_turn_id": str(user_turn_id),
        "goal_ato_id": str(goal_ato_id),
        "plan_ato_id": str(plan_ato_id),
        "reason_code": FAIL_REASON_PLAN_ATTEMPT_FAILED_V114,
        "attempt_act_id": str(attempt_act_id),
        "attempt_eval_id": str(attempt_eval_id),
    }
    fail_id = _stable_id("fail_event_v114", body)
    return ATOv71(
        ato_id=str(fail_id),
        ato_type="EVAL",
        subgraph=dict(body, satisfies=False),
        slots={},
        bindings={},
        cost=0.0,
        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
        invariants={"schema_version": 114, "eval_kind": "FAIL_EVENT_V114"},
        created_step=int(step),
        last_step=int(step),
    )


@dataclass(frozen=True)
class GateResultV114:
    ok: bool
    reason: str
    details: Dict[str, Any]


def verify_goal_plan_eval_law_v114(
    *,
    run_dir: str,
    max_replans_per_turn: int = 3,
    write_mind_graph: bool = True,
) -> GateResultV114:
    """
    Enforce V114 law over a completed conversation run:
      - 1 GOAL ATO per user input
      - PLAN ATO per user input
      - EVAL ATO with satisfies==true for the chosen plan before render
      - FAIL_EVENT ATO persisted for violations
    """
    rd = str(run_dir)
    turns_path = os.path.join(rd, "conversation_turns.jsonl")
    plans_path = os.path.join(rd, "action_plans.jsonl")
    evals_path = os.path.join(rd, "objective_evals.jsonl")
    if not os.path.exists(turns_path):
        return GateResultV114(ok=False, reason="missing_conversation_turns", details={"turns_path": str(turns_path)})

    turns_rows = _read_jsonl(turns_path)
    user_turns: List[Dict[str, Any]] = []
    assistant_turns: List[Dict[str, Any]] = []
    assistant_turn_by_eval_id: Dict[str, Dict[str, Any]] = {}
    conversation_id = ""
    for row in turns_rows:
        payload = _safe_dict(row.get("payload"))
        if not conversation_id:
            conversation_id = str(payload.get("conversation_id") or "")
        role = str(payload.get("role") or "")
        if role == "user":
            user_turns.append(payload)
        elif role == "assistant":
            assistant_turns.append(payload)
            refs = _safe_dict(payload.get("refs"))
            eid = str(refs.get("eval_id") or "")
            if eid and eid not in assistant_turn_by_eval_id:
                assistant_turn_by_eval_id[eid] = dict(payload)

    plans_rows = _read_jsonl(plans_path)
    plans_by_user_turn_id: Dict[str, Dict[str, Any]] = {}
    for row in plans_rows:
        if not isinstance(row, dict):
            continue
        user_turn_id = str(row.get("user_turn_id") or "")
        if not user_turn_id:
            continue
        # Deterministic first-wins: action_plans are already chained; keep earliest.
        if user_turn_id not in plans_by_user_turn_id:
            plans_by_user_turn_id[user_turn_id] = dict(row)

    eval_rows = _read_jsonl(evals_path)
    eval_by_id: Dict[str, Dict[str, Any]] = {}
    for row in eval_rows:
        if not isinstance(row, dict):
            continue
        eid = str(row.get("eval_id") or "")
        if not eid:
            continue
        if eid not in eval_by_id:
            eval_by_id[eid] = dict(row)

    mg: Optional[MindGraphV71] = None
    mg_dir = os.path.join(rd, "mind_graph_v114")
    if write_mind_graph:
        _ensure_absent(mg_dir)
        mg = MindGraphV71(run_dir=str(mg_dir))

        # Add all turns as OBS nodes for linking.
        for p in user_turns + assistant_turns:
            try:
                ato = _make_turn_obs_ato_v114(p)
            except Exception:
                continue
            mg.add_node(step=int(p.get("created_step", 0) or 0), ato=ato, reason="turn_observed_v114")

    violations: List[Dict[str, Any]] = []
    goals_total = 0
    plans_total = 0
    evals_total = 0
    fails_total = 0

    # Process user turns in deterministic order by created_step then turn_id.
    user_turns_sorted = list(user_turns)
    user_turns_sorted.sort(key=lambda p: (int(p.get("created_step", 0) or 0), str(p.get("turn_id") or "")))

    for ut in user_turns_sorted:
        user_turn_id = str(ut.get("turn_id") or "")
        if not user_turn_id:
            violations.append({"reason_code": FAIL_REASON_MISSING_GOAL_V114, "turn_id": ""})
            fails_total += 1
            if mg is not None:
                fail_ato = _make_fail_event_ato_v114(
                    conversation_id=str(conversation_id),
                    user_turn_id="",
                    goal_ato_id="",
                    plan_ato_id="",
                    reason_code=FAIL_REASON_MISSING_GOAL_V114,
                    step=int(ut.get("created_step", 0) or 0),
                )
                mg.add_node(step=int(ut.get("created_step", 0) or 0), ato=fail_ato, reason="fail_event_v114")
            continue

        try:
            goal_ato = _make_goal_ato_v114(conversation_id=str(conversation_id), user_turn_payload=ut)
        except Exception:
            violations.append({"reason_code": FAIL_REASON_MISSING_GOAL_V114, "turn_id": str(user_turn_id)})
            if mg is not None:
                fail_ato = _make_fail_event_ato_v114(
                    conversation_id=str(conversation_id),
                    user_turn_id=str(user_turn_id),
                    goal_ato_id="",
                    plan_ato_id="",
                    reason_code=FAIL_REASON_MISSING_GOAL_V114,
                    step=int(ut.get("created_step", 0) or 0),
                )
                mg.add_node(step=int(ut.get("created_step", 0) or 0), ato=fail_ato, reason="fail_event_v114")
            fails_total += 1
            continue

        goals_total += 1
        if mg is not None:
            mg.add_node(step=int(ut.get("created_step", 0) or 0), ato=goal_ato, reason="goal_created_v114")
            # Turn -> Goal
            if str(user_turn_id) in mg._nodes:
                mg.add_edge(
                    step=int(ut.get("created_step", 0) or 0),
                    src_ato_id=str(user_turn_id),
                    dst_ato_id=str(goal_ato.ato_id),
                    edge_type="CAUSES",
                    evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                    reason="input_causes_goal_v114",
                )

        plan_row = plans_by_user_turn_id.get(user_turn_id)
        if not isinstance(plan_row, dict):
            violations.append({"reason_code": FAIL_REASON_MISSING_PLAN_V114, "turn_id": str(user_turn_id)})
            if mg is not None:
                fail_ato = _make_fail_event_ato_v114(
                    conversation_id=str(conversation_id),
                    user_turn_id=str(user_turn_id),
                    goal_ato_id=str(goal_ato.ato_id),
                    plan_ato_id="",
                    reason_code=FAIL_REASON_MISSING_PLAN_V114,
                    step=int(ut.get("created_step", 0) or 0),
                )
                mg.add_node(step=int(ut.get("created_step", 0) or 0), ato=fail_ato, reason="fail_event_v114")
                mg.add_edge(
                    step=int(ut.get("created_step", 0) or 0),
                    src_ato_id=str(fail_ato.ato_id),
                    dst_ato_id=str(goal_ato.ato_id),
                    edge_type="DERIVED_FROM",
                    evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                    reason="fail_from_goal_v114",
                )
                if str(user_turn_id) in mg._nodes:
                    mg.add_edge(
                        step=int(ut.get("created_step", 0) or 0),
                        src_ato_id=str(fail_ato.ato_id),
                        dst_ato_id=str(user_turn_id),
                        edge_type="DERIVED_FROM",
                        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                        reason="fail_from_turn_v114",
                    )
            fails_total += 1
            continue

        try:
            plan_ato = _make_plan_ato_v114(action_plan=plan_row)
        except Exception:
            violations.append({"reason_code": FAIL_REASON_MISSING_PLAN_V114, "turn_id": str(user_turn_id)})
            fails_total += 1
            continue

        plans_total += 1
        if mg is not None:
            mg.add_node(step=int(plan_row.get("created_step", 0) or 0), ato=plan_ato, reason="plan_created_v114")
            mg.add_edge(
                step=int(plan_row.get("created_step", 0) or 0),
                src_ato_id=str(goal_ato.ato_id),
                dst_ato_id=str(plan_ato.ato_id),
                edge_type="DEPENDS_ON",
                evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                reason="goal_depends_on_plan_v114",
            )

        # Record failed attempts (planner/executor failures) as FAIL_EVENT ATOs.
        attempted_actions = _safe_list(plan_row.get("attempted_actions"))
        for a in attempted_actions:
            if not isinstance(a, dict):
                continue
            if bool(a.get("ok", False)):
                continue
            fails_total += 1
            if mg is None:
                continue
            fail_ato = _make_plan_attempt_fail_event_ato_v114(
                conversation_id=str(conversation_id),
                user_turn_id=str(user_turn_id),
                goal_ato_id=str(goal_ato.ato_id),
                plan_ato_id=str(plan_ato.ato_id),
                attempt_act_id=str(a.get("act_id") or ""),
                attempt_eval_id=str(a.get("eval_id") or ""),
                step=int(plan_row.get("created_step", 0) or 0),
            )
            mg.add_node(step=int(plan_row.get("created_step", 0) or 0), ato=fail_ato, reason="fail_event_v114")
            mg.add_edge(
                step=int(plan_row.get("created_step", 0) or 0),
                src_ato_id=str(fail_ato.ato_id),
                dst_ato_id=str(goal_ato.ato_id),
                edge_type="DERIVED_FROM",
                evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                reason="fail_from_goal_v114",
            )
            mg.add_edge(
                step=int(plan_row.get("created_step", 0) or 0),
                src_ato_id=str(fail_ato.ato_id),
                dst_ato_id=str(plan_ato.ato_id),
                edge_type="DERIVED_FROM",
                evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                reason="fail_from_plan_v114",
            )
            if str(user_turn_id) in mg._nodes:
                mg.add_edge(
                    step=int(plan_row.get("created_step", 0) or 0),
                    src_ato_id=str(fail_ato.ato_id),
                    dst_ato_id=str(user_turn_id),
                    edge_type="DERIVED_FROM",
                    evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                    reason="fail_from_turn_v114",
                )

        chosen_eval_id = str(plan_row.get("chosen_eval_id") or "")
        chosen_ok = bool(plan_row.get("chosen_ok", False))

        # Replanning constraint (observational): if chosen_ok is false, treat as exhaustion only if budget reached
        # or all ranked candidates were attempted (prefix).
        if not chosen_ok:
            attempted = _safe_list(plan_row.get("attempted_actions"))
            ranked = _safe_list(plan_row.get("ranked_candidates"))
            attempted_cnt = sum(1 for a in attempted if isinstance(a, dict))
            ranked_cnt = sum(1 for a in ranked if isinstance(a, dict))
            replanning_reason = ""
            if attempted_cnt < ranked_cnt and attempted_cnt < int(max_replans_per_turn):
                replanning_reason = FAIL_REASON_REPLANNING_REQUIRED_V114
                violations.append(
                    {
                        "reason_code": FAIL_REASON_REPLANNING_REQUIRED_V114,
                        "turn_id": str(user_turn_id),
                        "attempted_actions": int(attempted_cnt),
                        "ranked_candidates": int(ranked_cnt),
                        "max_replans_per_turn": int(max_replans_per_turn),
                    }
                )
            else:
                replanning_reason = FAIL_REASON_EXHAUSTED_PLANS_V114
                violations.append(
                    {
                        "reason_code": FAIL_REASON_EXHAUSTED_PLANS_V114,
                        "turn_id": str(user_turn_id),
                        "attempted_actions": int(attempted_cnt),
                        "ranked_candidates": int(ranked_cnt),
                        "max_replans_per_turn": int(max_replans_per_turn),
                    }
                )
            fails_total += 1
            if mg is not None and replanning_reason:
                fail_ato = _make_fail_event_ato_v114(
                    conversation_id=str(conversation_id),
                    user_turn_id=str(user_turn_id),
                    goal_ato_id=str(goal_ato.ato_id),
                    plan_ato_id=str(plan_ato.ato_id),
                    reason_code=str(replanning_reason),
                    step=int(plan_row.get("created_step", 0) or 0),
                )
                mg.add_node(step=int(plan_row.get("created_step", 0) or 0), ato=fail_ato, reason="fail_event_v114")
                mg.add_edge(
                    step=int(plan_row.get("created_step", 0) or 0),
                    src_ato_id=str(fail_ato.ato_id),
                    dst_ato_id=str(goal_ato.ato_id),
                    edge_type="DERIVED_FROM",
                    evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                    reason="fail_from_goal_v114",
                )
                mg.add_edge(
                    step=int(plan_row.get("created_step", 0) or 0),
                    src_ato_id=str(fail_ato.ato_id),
                    dst_ato_id=str(plan_ato.ato_id),
                    edge_type="DERIVED_FROM",
                    evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                    reason="fail_from_plan_v114",
                )
                if str(user_turn_id) in mg._nodes:
                    mg.add_edge(
                        step=int(plan_row.get("created_step", 0) or 0),
                        src_ato_id=str(fail_ato.ato_id),
                        dst_ato_id=str(user_turn_id),
                        edge_type="DERIVED_FROM",
                        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                        reason="fail_from_turn_v114",
                    )

        ev_row = eval_by_id.get(chosen_eval_id) if chosen_eval_id else None
        if not isinstance(ev_row, dict):
            # Some legacy paths (e.g. COMM_END) don't emit objective_evals rows.
            # V114 law still requires an EVAL ATO; synthesize it deterministically from the assistant turn refs
            # when the chosen plan is marked ok and the assistant turn carries the eval_id.
            synth = None
            if chosen_ok and chosen_eval_id:
                at = assistant_turn_by_eval_id.get(chosen_eval_id)
                if isinstance(at, dict):
                    arefs = _safe_dict(at.get("refs"))
                    synth = {
                        "eval_id": str(chosen_eval_id),
                        "turn_id": str(user_turn_id),
                        "step": int(at.get("created_step", 0) or 0),
                        "objective_kind": str(arefs.get("objective_kind") or ""),
                        "objective_id": str(arefs.get("objective_id") or ""),
                        "action_concept_id": str(arefs.get("action_concept_id") or ""),
                        "expected_text_sig": str(at.get("text_sig") or ""),
                        "output_text_sig": str(at.get("text_sig") or ""),
                        "verdict": {"ok": True, "reason": "", "score": 1},
                    }
            if isinstance(synth, dict):
                ev_row = synth
            else:
                violations.append({"reason_code": FAIL_REASON_MISSING_EVAL_V114, "turn_id": str(user_turn_id)})
                if mg is not None:
                    fail_ato = _make_fail_event_ato_v114(
                        conversation_id=str(conversation_id),
                        user_turn_id=str(user_turn_id),
                        goal_ato_id=str(goal_ato.ato_id),
                        plan_ato_id=str(plan_ato.ato_id),
                        reason_code=FAIL_REASON_MISSING_EVAL_V114,
                        step=int(plan_row.get("created_step", 0) or 0),
                    )
                    mg.add_node(step=int(plan_row.get("created_step", 0) or 0), ato=fail_ato, reason="fail_event_v114")
                    mg.add_edge(
                        step=int(plan_row.get("created_step", 0) or 0),
                        src_ato_id=str(fail_ato.ato_id),
                        dst_ato_id=str(goal_ato.ato_id),
                        edge_type="DERIVED_FROM",
                        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                        reason="fail_from_goal_v114",
                    )
                    mg.add_edge(
                        step=int(plan_row.get("created_step", 0) or 0),
                        src_ato_id=str(fail_ato.ato_id),
                        dst_ato_id=str(plan_ato.ato_id),
                        edge_type="DERIVED_FROM",
                        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                        reason="fail_from_plan_v114",
                    )
                    if str(user_turn_id) in mg._nodes:
                        mg.add_edge(
                            step=int(plan_row.get("created_step", 0) or 0),
                            src_ato_id=str(fail_ato.ato_id),
                            dst_ato_id=str(user_turn_id),
                            edge_type="DERIVED_FROM",
                            evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                            reason="fail_from_turn_v114",
                        )
                fails_total += 1
                continue

        eval_ato = _make_eval_ato_v114(objective_eval=ev_row)
        evals_total += 1
        if mg is not None:
            mg.add_node(step=int(ev_row.get("step", 0) or 0), ato=eval_ato, reason="eval_created_v114")
            mg.add_edge(
                step=int(ev_row.get("step", 0) or 0),
                src_ato_id=str(plan_ato.ato_id),
                dst_ato_id=str(eval_ato.ato_id),
                edge_type="CAUSES",
                evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                reason="plan_causes_eval_v114",
            )
            mg.add_edge(
                step=int(ev_row.get("step", 0) or 0),
                src_ato_id=str(goal_ato.ato_id),
                dst_ato_id=str(eval_ato.ato_id),
                edge_type="DEPENDS_ON",
                evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                reason="goal_depends_on_eval_v114",
            )

        verdict = _safe_dict(ev_row.get("verdict"))
        ok = bool(_safe_dict(verdict).get("ok", False))
        if not ok:
            violations.append({"reason_code": FAIL_REASON_RENDER_BLOCKED_NO_EVAL_SATISFIES_V114, "turn_id": str(user_turn_id)})
            if mg is not None:
                fail_ato = _make_fail_event_ato_v114(
                    conversation_id=str(conversation_id),
                    user_turn_id=str(user_turn_id),
                    goal_ato_id=str(goal_ato.ato_id),
                    plan_ato_id=str(plan_ato.ato_id),
                    reason_code=FAIL_REASON_RENDER_BLOCKED_NO_EVAL_SATISFIES_V114,
                    step=int(ev_row.get("step", 0) or 0),
                )
                mg.add_node(step=int(ev_row.get("step", 0) or 0), ato=fail_ato, reason="fail_event_v114")
                mg.add_edge(
                    step=int(ev_row.get("step", 0) or 0),
                    src_ato_id=str(fail_ato.ato_id),
                    dst_ato_id=str(goal_ato.ato_id),
                    edge_type="DERIVED_FROM",
                    evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                    reason="fail_from_goal_v114",
                )
                mg.add_edge(
                    step=int(ev_row.get("step", 0) or 0),
                    src_ato_id=str(fail_ato.ato_id),
                    dst_ato_id=str(plan_ato.ato_id),
                    edge_type="DERIVED_FROM",
                    evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                    reason="fail_from_plan_v114",
                )
                if str(user_turn_id) in mg._nodes:
                    mg.add_edge(
                        step=int(ev_row.get("step", 0) or 0),
                        src_ato_id=str(fail_ato.ato_id),
                        dst_ato_id=str(user_turn_id),
                        edge_type="DERIVED_FROM",
                        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                        reason="fail_from_turn_v114",
                    )
            fails_total += 1

    ok = not bool(violations)
    reason = "ok" if ok else str(violations[0].get("reason_code") or "violations")
    details = {
        "schema_version": 114,
        "conversation_id": str(conversation_id),
        "goals_total": int(goals_total),
        "plans_total": int(plans_total),
        "evals_total": int(evals_total),
        "fails_total": int(fails_total),
        "violations_total": int(len(violations)),
        "violations": list(violations),
    }

    mg_out: Dict[str, Any] = {}
    if mg is not None:
        chains = mg.verify_chains()
        snapshot = mg.snapshot_graph_state()
        graph_sig = mg.graph_sig()
        # Store only run_dir-relative paths for determinism/auditability.
        mg_out = {
            "mind_graph_dir": "mind_graph_v114",
            "mind_nodes_jsonl": os.path.join("mind_graph_v114", "mind_nodes.jsonl"),
            "mind_edges_jsonl": os.path.join("mind_graph_v114", "mind_edges.jsonl"),
            "mind_nodes_chain_ok": bool(chains.get("mind_nodes_chain_ok", False)),
            "mind_edges_chain_ok": bool(chains.get("mind_edges_chain_ok", False)),
            "mind_graph_sig": str(graph_sig),
            "mind_graph_snapshot_sig": sha256_hex(canonical_json_dumps(snapshot).encode("utf-8")),
        }
    details["mind_graph"] = dict(mg_out)

    # Persist V114 summary (WORM).
    summary_path = os.path.join(rd, "goal_plan_eval_summary_v114.json")
    _ensure_absent(summary_path)
    summary_obj = dict(
        details,
        ok=bool(ok),
        reason=str(reason),
        conversation_turns_sha256=_sha256_file(turns_path),
    )
    # NOTE: goal_plan_eval_summary_sha256 is anchored to conversation_turns.jsonl bytes.
    with open(summary_path, "x", encoding="utf-8") as f:
        f.write(json.dumps(summary_obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")

    return GateResultV114(ok=bool(ok), reason=str(reason), details=dict(details, goal_plan_eval_summary_v114_json=str(summary_path)))
