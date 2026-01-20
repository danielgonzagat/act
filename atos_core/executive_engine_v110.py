from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .act import canonical_json_dumps, sha256_hex


EXEC_KIND_ADVANCE_PLAN_V110 = "ADVANCE_PLAN"
EXEC_KIND_ASK_NEEDED_INFO_V110 = "ASK_NEEDED_INFO"
EXEC_KIND_SUMMARIZE_AND_ADVANCE_V110 = "SUMMARIZE_AND_ADVANCE"
EXEC_KIND_REPAIR_V110 = "REPAIR"
EXEC_KIND_CLOSE_GOAL_V110 = "CLOSE_GOAL"
EXEC_KIND_IDLE_V110 = "IDLE"


REPAIR_FORCE_NEXT_STEP_V110 = "FORCE_NEXT_STEP"
REPAIR_PLAN_REVISION_OR_ASK_V110 = "PLAN_REVISION_OR_ASK"
REPAIR_NONE_V110 = ""


ACK_TOKENS_V110 = {
    "a",
    "b",
    "c",
    "ok",
    "okay",
    "certo",
    "beleza",
    "blz",
    "continua",
    "continue",
    "segue",
    "vai",
    "faz",
    "isso",
    "isso ai",
    "isso aí",
    "sim",
    "pode",
}


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _canon_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _canon_str_list(xs: Any) -> List[str]:
    if not isinstance(xs, list):
        return []
    out = [str(x) for x in xs if isinstance(x, str) and str(x)]
    out = sorted(set(out), key=str)
    return list(out)


def is_minimal_ack_v110(user_text_norm: str) -> bool:
    s = str(user_text_norm or "").strip().lower()
    if not s:
        return False
    # Normalize whitespace for multi-token items like "isso ai".
    s2 = " ".join([t for t in s.split() if t])
    return s2 in set([str(x) for x in ACK_TOKENS_V110 if str(x)])


def build_executive_candidates_v110(
    *,
    has_open_goal: bool,
    goal_id: str,
    missing_slots: Sequence[str],
    plan_step_index_before: int,
    plan_step_index_after: int,
    goal_done_this_turn: bool,
    asked_needed_info_this_turn: bool,
    is_stall: bool,
    stall_reason: str,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if not bool(has_open_goal):
        candidates.append({"kind": EXEC_KIND_IDLE_V110, "score_int": 0, "reason_codes": ["no_open_goal"], "required_slots": [], "goal_id": ""})
        return candidates

    ms = [str(x) for x in list(missing_slots) if isinstance(x, str) and str(x)]
    ms = list(ms)
    if ms:
        candidates.append(
            {
                "kind": EXEC_KIND_ASK_NEEDED_INFO_V110,
                "score_int": 1000,
                "reason_codes": [f"missing_slot:{ms[0]}"],
                "required_slots": [str(ms[0])],
                "goal_id": str(goal_id),
            }
        )
        candidates.append({"kind": EXEC_KIND_IDLE_V110, "score_int": 0, "reason_codes": ["fallback"], "required_slots": [], "goal_id": str(goal_id)})
        return candidates

    if bool(goal_done_this_turn):
        candidates.append({"kind": EXEC_KIND_CLOSE_GOAL_V110, "score_int": 1100, "reason_codes": ["goal_done"], "required_slots": [], "goal_id": str(goal_id)})
        candidates.append({"kind": EXEC_KIND_IDLE_V110, "score_int": 0, "reason_codes": ["fallback"], "required_slots": [], "goal_id": str(goal_id)})
        return candidates

    if bool(is_stall):
        candidates.append(
            {
                "kind": EXEC_KIND_REPAIR_V110,
                "score_int": 1050,
                "reason_codes": [str(stall_reason or "stall")],
                "required_slots": [],
                "goal_id": str(goal_id),
            }
        )
        candidates.append({"kind": EXEC_KIND_ASK_NEEDED_INFO_V110, "score_int": 900, "reason_codes": ["stall_ask"], "required_slots": [], "goal_id": str(goal_id)})
        candidates.append({"kind": EXEC_KIND_IDLE_V110, "score_int": 0, "reason_codes": ["fallback"], "required_slots": [], "goal_id": str(goal_id)})
        return candidates

    # Default: advance plan.
    advanced = bool(int(plan_step_index_after) > int(plan_step_index_before))
    score = 950 if advanced else 800
    candidates.append({"kind": EXEC_KIND_ADVANCE_PLAN_V110, "score_int": int(score), "reason_codes": ["advance_plan"], "required_slots": [], "goal_id": str(goal_id)})
    candidates.append({"kind": EXEC_KIND_SUMMARIZE_AND_ADVANCE_V110, "score_int": 700, "reason_codes": ["summarize_then_advance"], "required_slots": [], "goal_id": str(goal_id)})
    if bool(asked_needed_info_this_turn):
        candidates.append({"kind": EXEC_KIND_ASK_NEEDED_INFO_V110, "score_int": 600, "reason_codes": ["asked_needed_info"], "required_slots": [], "goal_id": str(goal_id)})
    candidates.append({"kind": EXEC_KIND_IDLE_V110, "score_int": 0, "reason_codes": ["fallback"], "required_slots": [], "goal_id": str(goal_id)})
    return candidates


def choose_executive_candidate_v110(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    items = [dict(c) for c in candidates if isinstance(c, dict)]
    items.sort(key=lambda d: (-_canon_int(d.get("score_int")), str(d.get("kind") or ""), str(d.get("goal_id") or "")))
    return dict(items[0]) if items else {"kind": EXEC_KIND_IDLE_V110, "score_int": 0, "reason_codes": ["empty"], "required_slots": [], "goal_id": ""}


def compute_executive_flags_v110(
    *,
    has_open_goal: bool,
    missing_slots: Sequence[str],
    chosen_kind: str,
    asked_needed_info_this_turn: bool,
    plan_step_index_before: int,
    plan_step_index_after: int,
    stall_detected: bool,
    overclarify: bool,
) -> List[str]:
    flags: List[str] = []
    if bool(has_open_goal) and str(chosen_kind) == EXEC_KIND_IDLE_V110:
        flags.append("PASSIVE_WITH_OPEN_GOAL")
    if bool(stall_detected):
        flags.append("STALL")
    if bool(overclarify):
        flags.append("OVERCLARIFY")
    if bool(has_open_goal) and (not list(missing_slots)) and (int(plan_step_index_after) <= int(plan_step_index_before)) and (not bool(asked_needed_info_this_turn)):
        flags.append("NO_PROGRESS")
    return sorted(set([str(x) for x in flags if str(x)]))


def compute_executive_score_v110(*, flags: Sequence[str], chosen_kind: str) -> int:
    # Deterministic: start at 100 and subtract for critical flags.
    score = 100
    fl = set([str(x) for x in flags if str(x)])
    if "PASSIVE_WITH_OPEN_GOAL" in fl:
        score -= 80
    if "STALL" in fl:
        score -= 60
    if "OVERCLARIFY" in fl:
        score -= 40
    if "NO_PROGRESS" in fl:
        score -= 30
    if str(chosen_kind) in {EXEC_KIND_CLOSE_GOAL_V110}:
        score = min(100, score + 5)
    return max(0, min(100, int(score)))


def decide_progress_and_repair_v110(
    *,
    has_open_goal: bool,
    missing_slots: Sequence[str],
    chosen_kind: str,
    flags: Sequence[str],
    prev_no_progress_count: int,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "progress_allowed_v110": bool,
        "repair_action_v110": str,
        "no_progress_count_next": int,
      }
    """
    fl = set([str(x) for x in flags if str(x)])
    ms = [str(x) for x in list(missing_slots) if isinstance(x, str) and str(x)]

    # Count consecutive NO_PROGRESS turns with open goal.
    no_progress = bool(has_open_goal and ("NO_PROGRESS" in fl or str(chosen_kind) == EXEC_KIND_IDLE_V110))
    npc = int(prev_no_progress_count) + 1 if no_progress else 0

    # S6: no passivity with open goal.
    if bool(has_open_goal) and ("PASSIVE_WITH_OPEN_GOAL" in fl):
        return {"progress_allowed_v110": False, "repair_action_v110": REPAIR_FORCE_NEXT_STEP_V110, "no_progress_count_next": int(npc)}

    # S8: over-clarify bound when we already have required info.
    if bool(has_open_goal) and (not ms) and ("OVERCLARIFY" in fl):
        return {"progress_allowed_v110": False, "repair_action_v110": REPAIR_FORCE_NEXT_STEP_V110, "no_progress_count_next": int(npc)}

    # S7: stall detector (2 consecutive no-progress turns).
    if bool(has_open_goal) and int(npc) >= 2:
        return {"progress_allowed_v110": False, "repair_action_v110": REPAIR_PLAN_REVISION_OR_ASK_V110, "no_progress_count_next": int(npc)}

    # Normal: progress is allowed if we're asking needed info or advancing/closing.
    if str(chosen_kind) in {EXEC_KIND_ADVANCE_PLAN_V110, EXEC_KIND_SUMMARIZE_AND_ADVANCE_V110, EXEC_KIND_CLOSE_GOAL_V110, EXEC_KIND_ASK_NEEDED_INFO_V110}:
        return {"progress_allowed_v110": True, "repair_action_v110": REPAIR_NONE_V110, "no_progress_count_next": int(npc)}

    return {"progress_allowed_v110": False, "repair_action_v110": REPAIR_FORCE_NEXT_STEP_V110, "no_progress_count_next": int(npc)}


def deterministic_choice_id_v110(*, conversation_id: str, user_turn_index: int, chosen_kind: str, goal_id: str) -> str:
    body = {"schema_version": 110, "conversation_id": str(conversation_id), "user_turn_index": int(user_turn_index), "chosen_kind": str(chosen_kind), "goal_id": str(goal_id)}
    return _stable_hash_obj(body)
