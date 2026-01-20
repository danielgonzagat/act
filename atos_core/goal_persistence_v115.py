from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex


GOAL_STATUS_ACTIVE_V115 = "ACTIVE"
GOAL_STATUS_SATISFIED_V115 = "SATISFIED"
GOAL_STATUS_EXHAUSTED_V115 = "EXHAUSTED"
GOAL_STATUS_CANCELED_V115 = "CANCELED"


def goal_id_v115(*, conversation_id: str, user_turn_id: str, user_turn_index: int, parse_sig: str, user_text: str) -> str:
    body = {
        "schema_version": 115,
        "conversation_id": str(conversation_id),
        "user_turn_id": str(user_turn_id),
        "user_turn_index": int(user_turn_index),
        "parse_sig": str(parse_sig),
        "user_text": str(user_text),
    }
    return "goal_v115_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))


@dataclass(frozen=True)
class GoalLifecycleV115:
    goal_id: str
    user_turn_id: str
    user_turn_index: int
    status: str
    attempts_total: int
    successes_total: int
    failures_total: int
    attempted_actions: List[Dict[str, Any]]
    remaining_candidates: int


def fold_goal_lifecycle_v115(
    *,
    conversation_id: str,
    user_turn_payloads: Sequence[Dict[str, Any]],
    plans_by_user_turn_id: Dict[str, Dict[str, Any]],
    eval_by_id: Dict[str, Dict[str, Any]],
    max_replans_per_turn: int,
) -> List[GoalLifecycleV115]:
    """
    Derive a per-user-turn goal lifecycle snapshot.

    This is *not* the V99 goal ledger; it's the V115 "turn goal" lifecycle
    used to enforce persistence/replanning in the GOAL->PLAN->EVAL law.
    """
    out: List[GoalLifecycleV115] = []
    for ut in sorted(
        [dict(p) for p in user_turn_payloads if isinstance(p, dict)],
        key=lambda p: (int(p.get("created_step", 0) or 0), str(p.get("turn_id") or "")),
    ):
        tid = str(ut.get("turn_id") or "")
        if not tid:
            continue
        tindex = int(ut.get("turn_index", 0) or 0)
        refs = ut.get("refs") if isinstance(ut.get("refs"), dict) else {}
        parse_sig = str(refs.get("parse_sig") or "")
        user_text = str(ut.get("text") or "")
        gid = goal_id_v115(
            conversation_id=str(conversation_id),
            user_turn_id=str(tid),
            user_turn_index=int(tindex),
            parse_sig=str(parse_sig),
            user_text=str(user_text),
        )
        plan = plans_by_user_turn_id.get(tid) if isinstance(plans_by_user_turn_id.get(tid), dict) else {}
        ranked = plan.get("ranked_candidates") if isinstance(plan.get("ranked_candidates"), list) else []
        attempted = plan.get("attempted_actions") if isinstance(plan.get("attempted_actions"), list) else []
        chosen_eval_id = str(plan.get("chosen_eval_id") or "")
        chosen_ok = bool(plan.get("chosen_ok", False))
        eval_row = eval_by_id.get(chosen_eval_id) if chosen_eval_id and isinstance(eval_by_id.get(chosen_eval_id), dict) else {}
        verdict = eval_row.get("verdict") if isinstance(eval_row.get("verdict"), dict) else {}
        verdict_ok = bool(verdict.get("ok", False))

        attempts_total = sum(1 for a in attempted if isinstance(a, dict))
        failures_total = sum(1 for a in attempted if isinstance(a, dict) and not bool(a.get("ok", False)))
        successes_total = sum(1 for a in attempted if isinstance(a, dict) and bool(a.get("ok", False)))
        remaining = max(0, len([x for x in ranked if isinstance(x, dict)]) - attempts_total)

        status = GOAL_STATUS_ACTIVE_V115
        if chosen_ok and verdict_ok:
            status = GOAL_STATUS_SATISFIED_V115
        else:
            # If the runner did not satisfy and budget is exhausted or there are no remaining candidates,
            # mark as exhausted; otherwise stays ACTIVE.
            if int(attempts_total) >= int(max_replans_per_turn) or int(remaining) <= 0:
                status = GOAL_STATUS_EXHAUSTED_V115

        out.append(
            GoalLifecycleV115(
                goal_id=str(gid),
                user_turn_id=str(tid),
                user_turn_index=int(tindex),
                status=str(status),
                attempts_total=int(attempts_total),
                successes_total=int(successes_total),
                failures_total=int(failures_total),
                attempted_actions=[dict(a) for a in attempted if isinstance(a, dict)],
                remaining_candidates=int(remaining),
            )
        )
    return list(out)


def goal_registry_snapshot_v115(*, lifecycles: Sequence[GoalLifecycleV115]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for g in list(lifecycles):
        if not isinstance(g, GoalLifecycleV115):
            continue
        rows.append(
            {
                "goal_id": str(g.goal_id),
                "user_turn_id": str(g.user_turn_id),
                "user_turn_index": int(g.user_turn_index),
                "status": str(g.status),
                "attempts_total": int(g.attempts_total),
                "successes_total": int(g.successes_total),
                "failures_total": int(g.failures_total),
                "remaining_candidates": int(g.remaining_candidates),
            }
        )
    rows.sort(key=lambda r: (str(r.get("goal_id") or "")))
    snap = {"schema_version": 115, "kind": "goal_registry_snapshot_v115", "goals": list(rows)}
    snap["snapshot_sig"] = sha256_hex(canonical_json_dumps(snap).encode("utf-8"))
    return snap


def render_fail_response_v115(reason_code: str) -> str:
    r = str(reason_code or "")
    if not r:
        r = "unknown"
    # Deterministic, short, no embellishment.
    return f"FAIL: {r}"

