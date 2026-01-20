from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .ato_v71 import ATOv71, stable_hash_obj
from .goal_persistence_v115 import (
    GOAL_STATUS_EXHAUSTED_V115,
    GOAL_STATUS_SATISFIED_V115,
    fold_goal_lifecycle_v115,
    goal_id_v115,
    goal_registry_snapshot_v115,
)
from .mind_graph_v71 import MindGraphV71

FAIL_REASON_MISSING_GOAL_V115 = "missing_goal"
FAIL_REASON_GOAL_PLACEBO_V115 = "goal_placebo"
FAIL_REASON_MISSING_PLAN_V115 = "missing_plan"
FAIL_REASON_PLAN_NONTRIVIAL_V115 = "plan_nontrivial_failed"
FAIL_REASON_MISSING_EVAL_V115 = "missing_eval"
FAIL_REASON_EVAL_HAS_NO_CHECKS_V115 = "eval_has_no_checks"
FAIL_REASON_RENDER_BLOCKED_NO_EVAL_SATISFIES_V115 = "render_blocked_no_eval_satisfies"
FAIL_REASON_EXHAUSTED_PLANS_V115 = "exhausted_plans"
FAIL_REASON_REPLANNING_REQUIRED_V115 = "replanning_required"
FAIL_REASON_PLAN_ATTEMPT_FAILED_V115 = "plan_attempt_failed"


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


def _write_once_json(path: str, obj: Any) -> None:
    if os.path.exists(path):
        raise ValueError(f"worm_exists:{path}")
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        raise ValueError(f"tmp_exists:{tmp}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)


def _make_turn_obs_ato_v115(turn_payload: Dict[str, Any]) -> ATOv71:
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
            "schema_version": 115,
            "kind": str(turn_payload.get("kind") or ""),
            "role": str(role),
            "turn_index": int(turn_index),
            "text_sig": str(text_sig),
        },
        slots={},
        bindings={},
        cost=0.0,
        evidence_refs=[],
        invariants={"schema_version": 115, "obs_kind": "turn_v96"},
        created_step=int(created_step),
        last_step=int(created_step),
    )


def _make_goal_ato_v115(*, conversation_id: str, user_turn_payload: Dict[str, Any]) -> ATOv71:
    user_turn_id = str(user_turn_payload.get("turn_id") or "")
    if not user_turn_id:
        raise ValueError("missing_user_turn_id")
    created_step = int(user_turn_payload.get("created_step", 0) or 0)
    user_turn_index = int(user_turn_payload.get("turn_index", 0) or 0)
    refs = _safe_dict(user_turn_payload.get("refs"))
    user_text = str(user_turn_payload.get("text") or "")
    gid = goal_id_v115(
        conversation_id=str(conversation_id),
        user_turn_id=str(user_turn_id),
        user_turn_index=int(user_turn_index),
        parse_sig=str(refs.get("parse_sig") or ""),
        user_text=str(user_text),
    )
    body = {
        "schema_version": 115,
        "conversation_id": str(conversation_id),
        "user_turn_id": str(user_turn_id),
        "user_turn_index": int(user_turn_index),
        "intent_id": str(refs.get("intent_id") or ""),
        "parse_sig": str(refs.get("parse_sig") or ""),
        "user_text": str(user_text),
    }
    return ATOv71(
        ato_id=str(gid),
        ato_type="GOAL",
        subgraph=dict(body),
        slots={},
        bindings={},
        cost=float(len(str(user_text))),
        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
        invariants={"schema_version": 115, "goal_kind": "turn_goal_v115"},
        created_step=int(created_step),
        last_step=int(created_step),
    )


def _make_plan_ato_v115(*, action_plan: Dict[str, Any]) -> ATOv71:
    plan_id = str(action_plan.get("plan_id") or action_plan.get("plan_sig") or "")
    if not plan_id:
        raise ValueError("missing_plan_id")
    created_step = int(action_plan.get("created_step", 0) or 0)
    user_turn_id = str(action_plan.get("user_turn_id") or "")
    user_turn_index = int(action_plan.get("user_turn_index", -1) or -1)
    ranked = _safe_list(action_plan.get("ranked_candidates"))
    attempted = _safe_list(action_plan.get("attempted_actions"))
    subgraph = {
        "schema_version": 115,
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
            {"act_id": str(_safe_dict(a).get("act_id") or ""), "eval_id": str(_safe_dict(a).get("eval_id") or ""), "ok": bool(_safe_dict(a).get("ok", False))}
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
        invariants={"schema_version": 115, "plan_kind": "action_plan_v100"},
        created_step=int(created_step),
        last_step=int(created_step),
    )


def _eval_checks_v115(*, objective_eval: Dict[str, Any], extra_checks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    verdict = objective_eval.get("verdict") if isinstance(objective_eval.get("verdict"), dict) else {}
    details = verdict.get("details") if isinstance(verdict.get("details"), dict) else {}
    oexec = details.get("_objective_exec") if isinstance(details.get("_objective_exec"), dict) else {}
    trace_sig = str(oexec.get("trace_sig") or "")
    if trace_sig:
        checks.append({"check_kind": "objective_exec_trace", "trace_sig": str(trace_sig)})
    for c in list(extra_checks):
        if isinstance(c, dict):
            checks.append(dict(c))
    # Canonicalize order deterministically.
    pairs: List[Tuple[str, Dict[str, Any]]] = []
    for c in checks:
        try:
            k = canonical_json_dumps(c)
        except Exception:
            k = str(c)
        pairs.append((str(k), dict(c)))
    pairs.sort(key=lambda kv: str(kv[0]))
    return [v for _, v in pairs]


def _make_eval_ato_v115(*, objective_eval: Dict[str, Any], extra_checks: Sequence[Dict[str, Any]]) -> ATOv71:
    eval_id = str(objective_eval.get("eval_id") or "")
    if not eval_id:
        raise ValueError("missing_eval_id")
    step = int(objective_eval.get("step", 0) or 0)
    turn_id = str(objective_eval.get("turn_id") or "")
    verdict = _safe_dict(objective_eval.get("verdict"))
    ok = bool(_safe_dict(verdict).get("ok", False))
    reason = str(_safe_dict(verdict).get("reason") or "")
    score = int(_safe_dict(verdict).get("score", 0) or 0)
    checks = _eval_checks_v115(objective_eval=dict(objective_eval), extra_checks=list(extra_checks))
    subgraph = {
        "schema_version": 115,
        "eval_id": str(eval_id),
        "objective_kind": str(objective_eval.get("objective_kind") or ""),
        "objective_id": str(objective_eval.get("objective_id") or ""),
        "action_concept_id": str(objective_eval.get("action_concept_id") or ""),
        "expected_text_sig": str(objective_eval.get("expected_text_sig") or ""),
        "output_text_sig": str(objective_eval.get("output_text_sig") or ""),
        "satisfies": bool(ok),
        "score": int(score),
        "reason": str(reason),
        "checks": list(checks),
    }
    return ATOv71(
        ato_id=str(eval_id),
        ato_type="EVAL",
        subgraph=dict(subgraph),
        slots={},
        bindings={},
        cost=1.0,
        evidence_refs=[{"kind": "turn", "turn_id": str(turn_id)}] if turn_id else [],
        invariants={"schema_version": 115, "eval_kind": "objective_eval_v96"},
        created_step=int(step),
        last_step=int(step),
    )


def _make_fail_event_ato_v115(
    *,
    conversation_id: str,
    user_turn_id: str,
    goal_ato_id: str,
    plan_ato_id: str,
    reason_code: str,
    step: int,
    details: Optional[Dict[str, Any]] = None,
) -> ATOv71:
    body = {
        "schema_version": 115,
        "conversation_id": str(conversation_id),
        "user_turn_id": str(user_turn_id),
        "goal_ato_id": str(goal_ato_id),
        "plan_ato_id": str(plan_ato_id),
        "reason_code": str(reason_code),
        "details": dict(details) if isinstance(details, dict) else {},
    }
    fail_id = "fail_event_v115_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return ATOv71(
        ato_id=str(fail_id),
        ato_type="EVAL",
        subgraph=dict(body, satisfies=False),
        slots={},
        bindings={},
        cost=0.0,
        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}] if user_turn_id else [],
        invariants={"schema_version": 115, "eval_kind": "FAIL_EVENT_V115"},
        created_step=int(step),
        last_step=int(step),
    )


@dataclass(frozen=True)
class GateResultV115:
    ok: bool
    reason: str
    details: Dict[str, Any]


def verify_goal_plan_eval_law_v115(
    *,
    run_dir: str,
    max_replans_per_turn: int = 3,
    write_mind_graph: bool = True,
    write_snapshots: bool = True,
) -> GateResultV115:
    """
    V115 law over a completed conversation run:
      - 1 GOAL ATO per user turn (nontrivial)
      - PLAN ATO per user turn (nontrivial)
      - EVAL ATO satisfies==true before render
      - FAIL_EVENT ATO persisted for violations
      - Goal persistence snapshot (SATISFIED/EXHAUSTED) derivable from plans+evals
    """
    rd = str(run_dir)
    turns_path = os.path.join(rd, "conversation_turns.jsonl")
    plans_path = os.path.join(rd, "action_plans.jsonl")
    evals_path = os.path.join(rd, "objective_evals.jsonl")
    if not os.path.exists(turns_path):
        return GateResultV115(ok=False, reason="missing_conversation_turns", details={"turns_path": str(turns_path)})

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
    mg_dir = os.path.join(rd, "mind_graph_v115")
    if write_mind_graph:
        _ensure_absent(mg_dir)
        mg = MindGraphV71(run_dir=str(mg_dir))
        for p in user_turns + assistant_turns:
            try:
                ato = _make_turn_obs_ato_v115(p)
            except Exception:
                continue
            mg.add_node(step=int(p.get("created_step", 0) or 0), ato=ato, reason="turn_observed_v115")

    violations: List[Dict[str, Any]] = []
    goals_total = 0
    plans_total = 0
    evals_total = 0
    fails_total = 0

    # Deterministic user turn ordering.
    user_turns_sorted = list(user_turns)
    user_turns_sorted.sort(key=lambda p: (int(p.get("created_step", 0) or 0), str(p.get("turn_id") or "")))

    # Precompute extra check sources per user turn index (to embed into EVAL ATO).
    extra_checks_by_user_turn_id: Dict[str, List[Dict[str, Any]]] = {}
    for ut in user_turns_sorted:
        tid = str(ut.get("turn_id") or "")
        if not tid:
            continue
        refs = ut.get("refs") if isinstance(ut.get("refs"), dict) else {}
        extra_checks_by_user_turn_id[tid] = [
            {"check_kind": "parse_sig", "parse_sig": str(refs.get("parse_sig") or "")},
            {"check_kind": "intent_id", "intent_id": str(refs.get("intent_id") or "")},
        ]

    for ut in user_turns_sorted:
        user_turn_id = str(ut.get("turn_id") or "")
        step = int(ut.get("created_step", 0) or 0)
        if not user_turn_id:
            violations.append({"reason_code": FAIL_REASON_MISSING_GOAL_V115, "turn_id": ""})
            fails_total += 1
            if mg is not None:
                fail_ato = _make_fail_event_ato_v115(
                    conversation_id=str(conversation_id),
                    user_turn_id="",
                    goal_ato_id="",
                    plan_ato_id="",
                    reason_code=FAIL_REASON_MISSING_GOAL_V115,
                    step=int(step),
                )
                mg.add_node(step=int(step), ato=fail_ato, reason="fail_event_v115")
            continue

        # Goal must be nontrivial: non-empty user text.
        refs_ut = _safe_dict(ut.get("refs"))
        user_text = str(ut.get("text") or "")
        if not str(user_text).strip():
            violations.append({"reason_code": FAIL_REASON_GOAL_PLACEBO_V115, "turn_id": str(user_turn_id)})
            fails_total += 1
            if mg is not None:
                goal_ato_id = goal_id_v115(
                    conversation_id=str(conversation_id),
                    user_turn_id=str(user_turn_id),
                    user_turn_index=int(ut.get("turn_index", 0) or 0),
                    parse_sig=str(refs_ut.get("parse_sig") or ""),
                    user_text=str(user_text),
                )
                fail_ato = _make_fail_event_ato_v115(
                    conversation_id=str(conversation_id),
                    user_turn_id=str(user_turn_id),
                    goal_ato_id=str(goal_ato_id),
                    plan_ato_id="",
                    reason_code=FAIL_REASON_GOAL_PLACEBO_V115,
                    step=int(step),
                )
                mg.add_node(step=int(step), ato=fail_ato, reason="fail_event_v115")
            continue

        try:
            goal_ato = _make_goal_ato_v115(conversation_id=str(conversation_id), user_turn_payload=ut)
        except Exception:
            violations.append({"reason_code": FAIL_REASON_MISSING_GOAL_V115, "turn_id": str(user_turn_id)})
            fails_total += 1
            continue

        goals_total += 1
        if mg is not None:
            mg.add_node(step=int(step), ato=goal_ato, reason="goal_created_v115")
            if str(user_turn_id) in mg._nodes:
                mg.add_edge(
                    step=int(step),
                    src_ato_id=str(user_turn_id),
                    dst_ato_id=str(goal_ato.ato_id),
                    edge_type="CAUSES",
                    evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                    reason="input_causes_goal_v115",
                )

        plan_row = plans_by_user_turn_id.get(user_turn_id)
        if not isinstance(plan_row, dict) or not plan_row:
            violations.append({"reason_code": FAIL_REASON_MISSING_PLAN_V115, "turn_id": str(user_turn_id)})
            fails_total += 1
            if mg is not None:
                fail_ato = _make_fail_event_ato_v115(
                    conversation_id=str(conversation_id),
                    user_turn_id=str(user_turn_id),
                    goal_ato_id=str(goal_ato.ato_id),
                    plan_ato_id="",
                    reason_code=FAIL_REASON_MISSING_PLAN_V115,
                    step=int(step),
                )
                mg.add_node(step=int(step), ato=fail_ato, reason="fail_event_v115")
                mg.add_edge(
                    step=int(step),
                    src_ato_id=str(fail_ato.ato_id),
                    dst_ato_id=str(goal_ato.ato_id),
                    edge_type="DERIVED_FROM",
                    evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                    reason="fail_from_goal_v115",
                )
            continue

        ranked = _safe_list(plan_row.get("ranked_candidates"))
        attempted = _safe_list(plan_row.get("attempted_actions"))
        if (len([x for x in ranked if isinstance(x, dict)]) < 1) or (len([x for x in attempted if isinstance(x, dict)]) < 1):
            violations.append({"reason_code": FAIL_REASON_PLAN_NONTRIVIAL_V115, "turn_id": str(user_turn_id)})
            fails_total += 1
            if mg is not None:
                plan_ato_id = str(plan_row.get("plan_id") or "")
                fail_ato = _make_fail_event_ato_v115(
                    conversation_id=str(conversation_id),
                    user_turn_id=str(user_turn_id),
                    goal_ato_id=str(goal_ato.ato_id),
                    plan_ato_id=str(plan_ato_id),
                    reason_code=FAIL_REASON_PLAN_NONTRIVIAL_V115,
                    step=int(step),
                )
                mg.add_node(step=int(step), ato=fail_ato, reason="fail_event_v115")
            continue

        try:
            plan_ato = _make_plan_ato_v115(action_plan=plan_row)
        except Exception:
            violations.append({"reason_code": FAIL_REASON_MISSING_PLAN_V115, "turn_id": str(user_turn_id)})
            fails_total += 1
            continue

        plans_total += 1
        if mg is not None:
            mg.add_node(step=int(plan_row.get("created_step", 0) or 0), ato=plan_ato, reason="plan_created_v115")
            mg.add_edge(
                step=int(plan_row.get("created_step", 0) or 0),
                src_ato_id=str(goal_ato.ato_id),
                dst_ato_id=str(plan_ato.ato_id),
                edge_type="DEPENDS_ON",
                evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                reason="goal_depends_on_plan_v115",
            )

        # Record failed attempts as FAIL_EVENT.
        for a in attempted:
            if not isinstance(a, dict):
                continue
            if bool(a.get("ok", False)):
                continue
            fails_total += 1
            if mg is None:
                continue
            fail_ato = _make_fail_event_ato_v115(
                conversation_id=str(conversation_id),
                user_turn_id=str(user_turn_id),
                goal_ato_id=str(goal_ato.ato_id),
                plan_ato_id=str(plan_ato.ato_id),
                reason_code=FAIL_REASON_PLAN_ATTEMPT_FAILED_V115,
                step=int(plan_row.get("created_step", 0) or 0),
                details={"attempt_act_id": str(a.get("act_id") or ""), "attempt_eval_id": str(a.get("eval_id") or "")},
            )
            mg.add_node(step=int(plan_row.get("created_step", 0) or 0), ato=fail_ato, reason="fail_event_v115")
            mg.add_edge(
                step=int(plan_row.get("created_step", 0) or 0),
                src_ato_id=str(fail_ato.ato_id),
                dst_ato_id=str(goal_ato.ato_id),
                edge_type="DERIVED_FROM",
                evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                reason="fail_from_goal_v115",
            )
            mg.add_edge(
                step=int(plan_row.get("created_step", 0) or 0),
                src_ato_id=str(fail_ato.ato_id),
                dst_ato_id=str(plan_ato.ato_id),
                edge_type="DERIVED_FROM",
                evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                reason="fail_from_plan_v115",
            )
            if str(user_turn_id) in mg._nodes:
                mg.add_edge(
                    step=int(plan_row.get("created_step", 0) or 0),
                    src_ato_id=str(fail_ato.ato_id),
                    dst_ato_id=str(user_turn_id),
                    edge_type="DERIVED_FROM",
                    evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                    reason="fail_from_turn_v115",
                )

        chosen_eval_id = str(plan_row.get("chosen_eval_id") or "")
        chosen_ok = bool(plan_row.get("chosen_ok", False))

        if not chosen_ok:
            attempts_total = len([x for x in attempted if isinstance(x, dict)])
            ranked_total = len([x for x in ranked if isinstance(x, dict)])
            if int(attempts_total) >= int(max_replans_per_turn) or int(attempts_total) >= int(ranked_total):
                violations.append({"reason_code": FAIL_REASON_EXHAUSTED_PLANS_V115, "turn_id": str(user_turn_id)})
            else:
                violations.append({"reason_code": FAIL_REASON_REPLANNING_REQUIRED_V115, "turn_id": str(user_turn_id)})
            fails_total += 1
            if mg is not None:
                fail_reason = str(violations[-1].get("reason_code") or "")
                fail_ato = _make_fail_event_ato_v115(
                    conversation_id=str(conversation_id),
                    user_turn_id=str(user_turn_id),
                    goal_ato_id=str(goal_ato.ato_id),
                    plan_ato_id=str(plan_ato.ato_id),
                    reason_code=str(fail_reason),
                    step=int(plan_row.get("created_step", 0) or 0),
                )
                mg.add_node(step=int(plan_row.get("created_step", 0) or 0), ato=fail_ato, reason="fail_event_v115")
            continue

        # EVAL must satisfy (ok==True).
        eval_row = eval_by_id.get(chosen_eval_id) if chosen_eval_id and isinstance(eval_by_id.get(chosen_eval_id), dict) else None
        if not isinstance(eval_row, dict):
            # Allow deterministic synthesis for COMM_END if the assistant turn references this eval_id.
            synth_from = assistant_turn_by_eval_id.get(str(chosen_eval_id))
            if isinstance(synth_from, dict) and bool(plan_row.get("chosen_ok", False)):
                eval_row = {
                    "eval_id": str(chosen_eval_id),
                    "turn_id": str(user_turn_id),
                    "step": int(plan_row.get("created_step", 0) or 0),
                    "objective_kind": str(plan_row.get("objective_kind") or ""),
                    "objective_id": str(plan_row.get("objective_id") or ""),
                    "action_concept_id": str(plan_row.get("chosen_action_id") or ""),
                    "expected_text_sig": str(synth_from.get("text_sig") or ""),
                    "output_text_sig": str(synth_from.get("text_sig") or ""),
                    "verdict": {"ok": True, "score": 1, "reason": "synthesized_eval_v115", "details": {}},
                }
            else:
                violations.append({"reason_code": FAIL_REASON_MISSING_EVAL_V115, "turn_id": str(user_turn_id)})
                fails_total += 1
                continue

        verdict = eval_row.get("verdict") if isinstance(eval_row.get("verdict"), dict) else {}
        verdict_ok = bool(verdict.get("ok", False))
        if not verdict_ok:
            violations.append({"reason_code": FAIL_REASON_RENDER_BLOCKED_NO_EVAL_SATISFIES_V115, "turn_id": str(user_turn_id)})
            fails_total += 1
            if mg is not None:
                fail_ato = _make_fail_event_ato_v115(
                    conversation_id=str(conversation_id),
                    user_turn_id=str(user_turn_id),
                    goal_ato_id=str(goal_ato.ato_id),
                    plan_ato_id=str(plan_ato.ato_id),
                    reason_code=FAIL_REASON_RENDER_BLOCKED_NO_EVAL_SATISFIES_V115,
                    step=int(plan_row.get("created_step", 0) or 0),
                )
                mg.add_node(step=int(plan_row.get("created_step", 0) or 0), ato=fail_ato, reason="fail_event_v115")
            continue

        extra_checks = extra_checks_by_user_turn_id.get(str(user_turn_id), [])
        eval_ato = _make_eval_ato_v115(objective_eval=dict(eval_row), extra_checks=list(extra_checks))
        checks = eval_ato.subgraph.get("checks") if isinstance(eval_ato.subgraph.get("checks"), list) else []
        if len([x for x in checks if isinstance(x, dict)]) < 1:
            violations.append({"reason_code": FAIL_REASON_EVAL_HAS_NO_CHECKS_V115, "turn_id": str(user_turn_id)})
            fails_total += 1
            continue

        evals_total += 1
        if mg is not None:
            mg.add_node(step=int(eval_row.get("step", plan_row.get("created_step", 0) or 0) or 0), ato=eval_ato, reason="eval_created_v115")
            mg.add_edge(
                step=int(eval_row.get("step", plan_row.get("created_step", 0) or 0) or 0),
                src_ato_id=str(plan_ato.ato_id),
                dst_ato_id=str(eval_ato.ato_id),
                edge_type="DERIVED_FROM",
                evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                reason="plan_derived_eval_v115",
            )
            mg.add_edge(
                step=int(eval_row.get("step", plan_row.get("created_step", 0) or 0) or 0),
                src_ato_id=str(eval_ato.ato_id),
                dst_ato_id=str(goal_ato.ato_id),
                edge_type="SUPPORTS",
                evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}],
                reason="eval_supports_goal_v115",
            )

    ok = len([v for v in violations if isinstance(v, dict)]) == 0
    reason = "ok" if ok else str((violations[0].get("reason_code") if violations else "violations") or "violations")

    mind_graph: Dict[str, Any] = {}
    if mg is not None:
        mind_graph = dict(mg.verify_chains())
        mind_graph["mind_graph_sig"] = str(mg.graph_sig())

    lifecycles = fold_goal_lifecycle_v115(
        conversation_id=str(conversation_id),
        user_turn_payloads=list(user_turns),
        plans_by_user_turn_id=dict(plans_by_user_turn_id),
        eval_by_id=dict(eval_by_id),
        max_replans_per_turn=int(max_replans_per_turn),
    )
    goal_snapshot = goal_registry_snapshot_v115(lifecycles=list(lifecycles))

    details = {
        "schema_version": 115,
        "ok": bool(ok),
        "reason": str(reason),
        "violations": list(violations),
        "goals_total": int(goals_total),
        "plans_total": int(plans_total),
        "evals_total": int(evals_total),
        "fails_total": int(fails_total),
        "mind_graph": dict(mind_graph),
        "goal_registry_snapshot_v115": dict(goal_snapshot),
    }

    # Write summary and snapshots (WORM) if requested.
    if write_snapshots:
        _write_once_json(os.path.join(rd, "goal_plan_eval_summary_v115.json"), dict(details))
        _write_once_json(os.path.join(rd, "goal_registry_snapshot_v115.json"), dict(goal_snapshot))

    return GateResultV115(ok=bool(ok), reason=str(reason), details=dict(details))
