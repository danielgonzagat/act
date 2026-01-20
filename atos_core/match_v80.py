from __future__ import annotations

from typing import List

from .act import Act


def is_act_allowed_for_goal_kind(*, act: Act, goal_kind: str) -> bool:
    """
    Explicit (audit-first) match semantics for routing/enforcement:
      - If act.match has no valid goal_kinds list -> allowed (global).
      - If act.match.goal_kinds is a list -> allowed iff goal_kind in that list.
    """
    if act is None:
        return False
    mk = act.match if isinstance(getattr(act, "match", None), dict) else {}
    gks = mk.get("goal_kinds")
    if not isinstance(gks, list):
        return True
    allowed = {str(x) for x in gks if str(x)}
    return str(goal_kind or "") in allowed


def filter_concept_ids_for_goal_kind(*, store, goal_kind: str) -> List[str]:
    out: List[str] = []
    try:
        concept_acts = store.concept_acts()
    except Exception:
        concept_acts = []
    for act in concept_acts:
        if act is None:
            continue
        if is_act_allowed_for_goal_kind(act=act, goal_kind=str(goal_kind or "")):
            out.append(str(act.id))
    out.sort(key=str)
    return out

