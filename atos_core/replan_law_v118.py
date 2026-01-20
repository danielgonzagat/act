from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .act import canonical_json_dumps, sha256_hex

REPLAN_REASON_OK_V118 = "ok"
REPLAN_REASON_EXHAUSTED_PLANS_V118 = "exhausted_plans"
REPLAN_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V118 = "plan_search_budget_exhausted"
REPLAN_REASON_DUPLICATE_PLAN_CANDIDATE_V118 = "duplicate_plan_candidate"


def _ensure_absent(path: str) -> None:
    if os.path.exists(path):
        raise ValueError(f"worm_exists:{path}")


def _write_once_json(path: str, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        raise ValueError(f"tmp_exists:{tmp}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)


def plan_hash_v118(plan_sem_sig: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(plan_sem_sig).encode("utf-8"))


@dataclass(frozen=True)
class PlanCandidateV118:
    plan_id: str
    plan_cost: float
    plan_sem_sig: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        sem = {
            "plan_id": str(self.plan_id),
            "plan_cost": float(self.plan_cost),
            "plan_sem_sig": dict(self.plan_sem_sig) if isinstance(self.plan_sem_sig, dict) else {},
        }
        sem["plan_hash"] = plan_hash_v118(dict(sem["plan_sem_sig"]))
        return dict(sem)


@dataclass(frozen=True)
class PlanAttemptV118:
    attempt_index: int
    plan_id: str
    plan_hash: str
    plan_cost: float
    eval_satisfies: bool
    dialogue_survival_ok: bool
    fail_reason_code: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_index": int(self.attempt_index),
            "plan_id": str(self.plan_id),
            "plan_hash": str(self.plan_hash),
            "plan_cost": float(self.plan_cost),
            "eval_satisfies": bool(self.eval_satisfies),
            "dialogue_survival_ok": bool(self.dialogue_survival_ok),
            "fail_reason_code": str(self.fail_reason_code),
        }


@dataclass(frozen=True)
class ReplanResultV118:
    ok: bool
    reason: str
    attempts_total: int
    attempts: List[PlanAttemptV118]
    chosen_plan_hash: str
    budget_total: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 118,
            "kind": "replan_result_v118",
            "ok": bool(self.ok),
            "reason": str(self.reason),
            "attempts_total": int(self.attempts_total),
            "attempts": [a.to_dict() for a in list(self.attempts)],
            "chosen_plan_hash": str(self.chosen_plan_hash),
            "budget_total": int(self.budget_total),
        }


def replan_until_satisfies_v118(
    *,
    next_plan: Callable[[], Optional[PlanCandidateV118]],
    exec_plan: Callable[[PlanCandidateV118], Tuple[bool, bool, str]],
    max_attempts: int,
) -> ReplanResultV118:
    """
    Deterministic replanning loop:
      - enumerates plan candidates in the caller-provided order,
      - never retries a duplicate plan_hash,
      - stops only on SATISFIES+dialogue_ok, enumerator exhaustion, or budget exhaustion.

    exec_plan returns: (eval_satisfies, dialogue_survival_ok, fail_reason_code).
    """
    if int(max_attempts) <= 0:
        raise ValueError("max_attempts_must_be_positive")

    tried: Set[str] = set()
    attempts: List[PlanAttemptV118] = []

    while True:
        if int(len(attempts)) >= int(max_attempts):
            return ReplanResultV118(
                ok=False,
                reason=REPLAN_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V118,
                attempts_total=int(len(attempts)),
                attempts=list(attempts),
                chosen_plan_hash="",
                budget_total=int(max_attempts),
            )

        cand = next_plan()
        if cand is None:
            return ReplanResultV118(
                ok=False,
                reason=REPLAN_REASON_EXHAUSTED_PLANS_V118,
                attempts_total=int(len(attempts)),
                attempts=list(attempts),
                chosen_plan_hash="",
                budget_total=int(max_attempts),
            )

        cand_dict = cand.to_dict()
        ph = str(cand_dict.get("plan_hash") or "")
        if not ph:
            ph = plan_hash_v118(cand.plan_sem_sig if isinstance(cand.plan_sem_sig, dict) else {})

        if ph in tried:
            attempts.append(
                PlanAttemptV118(
                    attempt_index=int(len(attempts)),
                    plan_id=str(cand.plan_id),
                    plan_hash=str(ph),
                    plan_cost=float(cand.plan_cost),
                    eval_satisfies=False,
                    dialogue_survival_ok=False,
                    fail_reason_code=REPLAN_REASON_DUPLICATE_PLAN_CANDIDATE_V118,
                )
            )
            continue

        tried.add(ph)
        eval_ok, dialogue_ok, fail_reason = exec_plan(cand)
        attempts.append(
            PlanAttemptV118(
                attempt_index=int(len(attempts)),
                plan_id=str(cand.plan_id),
                plan_hash=str(ph),
                plan_cost=float(cand.plan_cost),
                eval_satisfies=bool(eval_ok),
                dialogue_survival_ok=bool(dialogue_ok),
                fail_reason_code=str(fail_reason or ""),
            )
        )

        if bool(eval_ok) and bool(dialogue_ok):
            return ReplanResultV118(
                ok=True,
                reason=REPLAN_REASON_OK_V118,
                attempts_total=int(len(attempts)),
                attempts=list(attempts),
                chosen_plan_hash=str(ph),
                budget_total=int(max_attempts),
            )


def write_replan_trace_v118(*, path: str, goal_id: str, result: ReplanResultV118) -> Dict[str, Any]:
    """
    Persist a write-once replanning trace artifact.
    """
    body = {
        "schema_version": 118,
        "kind": "replan_trace_v118",
        "goal_id": str(goal_id),
        "result": dict(result.to_dict()),
    }
    body["trace_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    _write_once_json(str(path), dict(body))
    return dict(body)

