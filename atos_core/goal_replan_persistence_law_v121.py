from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .act import canonical_json_dumps, sha256_hex


FAIL_REASON_MISSING_CONVERSATION_TURNS_V121 = "missing_conversation_turns_v121"
FAIL_REASON_MISSING_ACTION_PLANS_V121 = "missing_action_plans_v121"
FAIL_REASON_GOAL_DIED_ON_FAIL_V121 = "goal_died_on_fail_v121"
FAIL_REASON_EXHAUSTION_WITHOUT_PROOF_V121 = "exhaustion_without_proof_v121"

OK_REASON_OK_V121 = "ok"
OK_REASON_GOAL_EXHAUSTED_V121 = "goal_exhausted_v121"


_WAITING_OBJECTIVE_KINDS_V121 = {"COMM_ASK_CLARIFY", "COMM_CONFIRM"}


@dataclass(frozen=True)
class GoalReplanPersistenceResultV121:
    ok: bool
    reason: str
    details: Dict[str, Any]


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


def _payload_view(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        payload = r.get("payload")
        if isinstance(payload, dict):
            out.append(dict(payload))
    return out


def _extract_last_user_turn_payload(run_dir: str) -> Dict[str, Any]:
    rows = _read_jsonl(os.path.join(str(run_dir), "conversation_turns.jsonl"))
    payloads = _payload_view(rows)
    users: List[Dict[str, Any]] = [dict(p) for p in payloads if isinstance(p, dict) and str(p.get("role") or "") == "user"]
    if not users:
        return {}
    users.sort(key=lambda p: (int(p.get("turn_index", 0) or 0), int(p.get("created_step", 0) or 0), str(p.get("turn_id") or "")))
    return dict(users[-1])


def _plan_row_for_user_turn(run_dir: str, user_turn_id: str) -> Dict[str, Any]:
    rows = _read_jsonl(os.path.join(str(run_dir), "action_plans.jsonl"))
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("user_turn_id") or "") == str(user_turn_id):
            return dict(row)
    return {}


def _eval_by_id(run_dir: str) -> Dict[str, Dict[str, Any]]:
    rows = _read_jsonl(os.path.join(str(run_dir), "objective_evals.jsonl"))
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        eval_id = str(row.get("eval_id") or "")
        if eval_id and eval_id not in out:
            out[eval_id] = dict(row)
    return dict(out)


def _attempt_proof_from_plan(
    *, plan_row: Dict[str, Any], eval_by_id: Dict[str, Dict[str, Any]]
) -> Tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]]]:
    attempted = plan_row.get("attempted_actions") if isinstance(plan_row.get("attempted_actions"), list) else []
    proof: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for i, a in enumerate([x for x in attempted if isinstance(x, dict)]):
        act_id = str(a.get("act_id") or "")
        eval_id = str(a.get("eval_id") or "")
        if not act_id or not eval_id:
            missing.append({"attempt_index": int(i), "missing": "act_id_or_eval_id"})
            continue
        eval_row = eval_by_id.get(eval_id) if isinstance(eval_by_id.get(eval_id), dict) else None
        if not isinstance(eval_row, dict):
            missing.append({"attempt_index": int(i), "missing": "eval_row", "eval_id": str(eval_id)})
            continue
        verdict = eval_row.get("verdict") if isinstance(eval_row.get("verdict"), dict) else {}
        verdict_ok = bool(verdict.get("ok", False))
        verdict_reason = str(verdict.get("reason") or "")
        sem = {"act_id": str(act_id), "eval_id": str(eval_id)}
        plan_hash = sha256_hex(canonical_json_dumps(sem).encode("utf-8"))
        proof.append(
            {
                "attempt_index": int(i),
                "plan_hash": str(plan_hash),
                "act_id": str(act_id),
                "eval_id": str(eval_id),
                "attempt_ok": bool(a.get("ok", False)),
                "eval_verdict_ok": bool(verdict_ok),
                "eval_reason": str(verdict_reason),
            }
        )
    ok = len(missing) == 0
    return bool(ok), list(proof), list(missing)


def verify_goal_replan_persistence_law_v121(*, run_dir: str, max_replans_per_turn: int) -> GoalReplanPersistenceResultV121:
    """
    V121 law (deterministic, fail-closed):
      - If a plan fails and there are remaining ranked candidates under budget, this is a violation (goal_died_on_fail_v121).
      - If exhaustion is claimed (no remaining candidates or budget reached), proof is required:
          attempted_actions must be complete and each attempt must have a corresponding objective_eval row.
    """
    rd = str(run_dir)
    turns_path = os.path.join(rd, "conversation_turns.jsonl")
    plans_path = os.path.join(rd, "action_plans.jsonl")

    if not os.path.exists(turns_path):
        return GoalReplanPersistenceResultV121(ok=False, reason=FAIL_REASON_MISSING_CONVERSATION_TURNS_V121, details={"conversation_turns_jsonl": str(turns_path)})
    if not os.path.exists(plans_path):
        return GoalReplanPersistenceResultV121(ok=False, reason=FAIL_REASON_MISSING_ACTION_PLANS_V121, details={"action_plans_jsonl": str(plans_path)})

    last_user = _extract_last_user_turn_payload(rd)
    user_turn_id = str(last_user.get("turn_id") or "")
    if not user_turn_id:
        return GoalReplanPersistenceResultV121(ok=False, reason=FAIL_REASON_MISSING_CONVERSATION_TURNS_V121, details={"missing": "last_user_turn_id"})

    plan_row = _plan_row_for_user_turn(rd, user_turn_id)
    if not plan_row:
        return GoalReplanPersistenceResultV121(ok=False, reason=FAIL_REASON_MISSING_ACTION_PLANS_V121, details={"user_turn_id": str(user_turn_id)})

    ranked = plan_row.get("ranked_candidates") if isinstance(plan_row.get("ranked_candidates"), list) else []
    attempted = plan_row.get("attempted_actions") if isinstance(plan_row.get("attempted_actions"), list) else []
    ranked_total = len([x for x in ranked if isinstance(x, dict)])
    attempts_total = len([x for x in attempted if isinstance(x, dict)])
    chosen_ok = bool(plan_row.get("chosen_ok", False))
    objective_kind = str(plan_row.get("objective_kind") or "")
    waiting_for_user = objective_kind in _WAITING_OBJECTIVE_KINDS_V121

    # If chosen_ok is True, the runner satisfied within this turn; law is vacuously satisfied.
    if bool(chosen_ok):
        details = {
            "schema_version": 121,
            "kind": "goal_replan_persistence_law_v121_result",
            "ok": True,
            "reason": OK_REASON_OK_V121,
            "user_turn_id": str(user_turn_id),
            "objective_kind": str(objective_kind),
            "waiting_for_user": bool(waiting_for_user),
            "ranked_total": int(ranked_total),
            "attempts_total": int(attempts_total),
        }
        return GoalReplanPersistenceResultV121(ok=True, reason=OK_REASON_OK_V121, details=dict(details))

    # chosen_ok is False: either replanning is required (violation) or exhaustion is proven.
    cap = int(max_replans_per_turn)
    if cap <= 0:
        cap = 0
    budgeted_ranked_total = int(min(int(ranked_total), int(cap))) if int(cap) > 0 else int(ranked_total)
    remaining = max(0, int(ranked_total) - int(attempts_total))

    # If we haven't tried all ranked candidates (or we are under the replanning budget), we must not fail early.
    if (not waiting_for_user) and int(attempts_total) < int(budgeted_ranked_total) and int(remaining) > 0:
        details = {
            "schema_version": 121,
            "kind": "goal_replan_persistence_law_v121_result",
            "ok": False,
            "reason": FAIL_REASON_GOAL_DIED_ON_FAIL_V121,
            "user_turn_id": str(user_turn_id),
            "objective_kind": str(objective_kind),
            "waiting_for_user": bool(waiting_for_user),
            "ranked_total": int(ranked_total),
            "attempts_total": int(attempts_total),
            "remaining_candidates": int(remaining),
            "max_replans_per_turn": int(max_replans_per_turn),
        }
        return GoalReplanPersistenceResultV121(ok=False, reason=FAIL_REASON_GOAL_DIED_ON_FAIL_V121, details=dict(details))

    # Exhaustion (or waiting): proof required for exhaustion cases.
    eval_by_id = _eval_by_id(rd)
    ok_pf, proof, missing = _attempt_proof_from_plan(plan_row=dict(plan_row), eval_by_id=dict(eval_by_id))
    if not bool(ok_pf):
        details = {
            "schema_version": 121,
            "kind": "goal_replan_persistence_law_v121_result",
            "ok": False,
            "reason": FAIL_REASON_EXHAUSTION_WITHOUT_PROOF_V121,
            "user_turn_id": str(user_turn_id),
            "objective_kind": str(objective_kind),
            "waiting_for_user": bool(waiting_for_user),
            "ranked_total": int(ranked_total),
            "attempts_total": int(attempts_total),
            "missing_proof": list(missing),
        }
        return GoalReplanPersistenceResultV121(ok=False, reason=FAIL_REASON_EXHAUSTION_WITHOUT_PROOF_V121, details=dict(details))

    details = {
        "schema_version": 121,
        "kind": "goal_replan_persistence_law_v121_result",
        "ok": True,
        "reason": OK_REASON_GOAL_EXHAUSTED_V121 if not bool(waiting_for_user) else OK_REASON_OK_V121,
        "user_turn_id": str(user_turn_id),
        "objective_kind": str(objective_kind),
        "waiting_for_user": bool(waiting_for_user),
        "ranked_total": int(ranked_total),
        "attempts_total": int(attempts_total),
        "attempts_proof": list(proof),
    }
    return GoalReplanPersistenceResultV121(ok=True, reason=str(details["reason"]), details=dict(details))


def write_goal_replan_persistence_summary_v121(*, run_dir: str, max_replans_per_turn: int) -> Dict[str, Any]:
    """
    Write-once summary artifact: goal_replan_persistence_summary_v121.json (deterministic).
    """
    res = verify_goal_replan_persistence_law_v121(run_dir=str(run_dir), max_replans_per_turn=int(max_replans_per_turn))
    body = dict(res.details)
    body["summary_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    path = os.path.join(str(run_dir), "goal_replan_persistence_summary_v121.json")
    if os.path.exists(path):
        raise ValueError(f"worm_exists:{path}")
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        raise ValueError(f"tmp_exists:{tmp}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)
    return dict(body)

