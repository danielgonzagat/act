from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .goal_supports_v89 import SupportClaimV89, SupportStatsV89


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


EPS_V89 = 1e-9
AGE_GAIN_V89 = 0.01

MIN_ATTEMPTS_PRUNE_V89 = 5
MIN_SUCCESS_PRUNE_V89 = 0.2
COOLDOWN_STEPS_V89 = 20

MAX_ATTEMPTS_PER_GOAL_V89 = 50
MAX_COST_PER_GOAL_V89 = 50.0

ANTI_REPEAT_LIMIT_V89 = 3
ANTI_REPEAT_COOLDOWN_STEPS_V89 = 5


@dataclass(frozen=True)
class CandidateScoreV89:
    concept_key: str
    score: float
    claim: SupportClaimV89
    stats: SupportStatsV89


def goal_urgency_weight_v89(goal_act) -> float:
    ev = goal_act.evidence if isinstance(getattr(goal_act, "evidence", None), dict) else {}
    g = ev.get("goal") if isinstance(ev.get("goal"), dict) else {}
    try:
        w = float(g.get("urgency_weight", 1.0))
    except Exception:
        w = 1.0
    if w <= 0.0:
        w = 1.0
    return float(w)


def goal_priority_v89(*, goal_act, created_step: int, now_step: int, age_gain: float = AGE_GAIN_V89) -> float:
    """
    priority(goal) = urgency_weight * (1 + age_steps*age_gain)
    Deterministic, no wall-clock.
    """
    w = goal_urgency_weight_v89(goal_act)
    age_steps = int(now_step) - int(created_step)
    if age_steps < 0:
        age_steps = 0
    return float(w) * (1.0 + float(age_steps) * float(age_gain))


def candidate_expected_score_v89(*, goal_priority: float, stats: SupportStatsV89, eps: float = EPS_V89) -> float:
    """
    score = (U * expected_success) / (expected_cost + eps)
    """
    U = float(goal_priority)
    k = float(stats.expected_success)
    c = float(stats.expected_cost)
    if c < 0.0:
        c = 0.0
    return (U * k) / (c + float(eps))


def rank_candidates_v89(candidates: Sequence[CandidateScoreV89]) -> List[CandidateScoreV89]:
    """
    Deterministic ordering: score desc, then concept_key asc.
    """
    lst = [c for c in candidates if isinstance(c, CandidateScoreV89)]
    lst.sort(key=lambda c: (-float(c.score), str(c.concept_key)))
    return lst


def should_cooldown_edge_v89(*, stats: SupportStatsV89) -> bool:
    if int(stats.attempts) < int(MIN_ATTEMPTS_PRUNE_V89):
        return False
    return float(stats.expected_success) < float(MIN_SUCCESS_PRUNE_V89)


def should_abandon_goal_v89(*, attempts: int, cost_total: float) -> Tuple[bool, str]:
    if int(attempts) >= int(MAX_ATTEMPTS_PER_GOAL_V89):
        return True, "max_attempts_per_goal"
    if float(cost_total) >= float(MAX_COST_PER_GOAL_V89):
        return True, "max_cost_per_goal"
    return False, ""


def edge_key_v89(goal_id: str, concept_key: str) -> str:
    return _stable_hash_obj({"goal_id": str(goal_id), "concept_key": str(concept_key)})


def apply_anti_repeat_v89(
    *,
    prev_edge_key: str,
    prev_repeats: int,
    cur_edge_key: str,
    last_attempt_ok: bool,
) -> Tuple[int, bool]:
    """
    Returns (new_repeats, trigger_cooldown_now).
    Anti-loop trivial: if same edge chosen consecutively more than R repeats without success,
    trigger short cooldown.
    """
    if bool(last_attempt_ok):
        return 0, False
    if str(cur_edge_key) == str(prev_edge_key) and str(cur_edge_key):
        reps = int(prev_repeats) + 1
    else:
        reps = 1
    if reps > int(ANTI_REPEAT_LIMIT_V89):
        return reps, True
    return reps, False

