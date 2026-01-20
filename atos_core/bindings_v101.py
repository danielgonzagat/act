from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def value_hash_v101(value: Any) -> str:
    """
    Hash for the binding *value* (pointer payload), not for ledger/evidence.
    """
    return sha256_hex(canonical_json_dumps(value).encode("utf-8"))


def binding_id_v101(*, binding_kind: str, value_hash: str) -> str:
    body = {"schema_version": 101, "binding_kind": str(binding_kind), "value_hash": str(value_hash)}
    return f"binding_v101_{_stable_hash_obj(body)}"


@dataclass(frozen=True)
class BindingV101:
    binding_id: str
    binding_kind: str
    value: Dict[str, Any]
    value_hash: str
    value_preview: str
    created_turn_index: int
    last_used_turn_index: int
    use_count: int
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 101,
            "kind": "binding_v101",
            "binding_id": str(self.binding_id),
            "binding_kind": str(self.binding_kind),
            "value": json.loads(canonical_json_dumps(self.value)) if isinstance(self.value, dict) else {},
            "value_hash": str(self.value_hash),
            "value_preview": str(self.value_preview),
            "created_turn_index": int(self.created_turn_index),
            "last_used_turn_index": int(self.last_used_turn_index),
            "use_count": int(self.use_count),
            "provenance": json.loads(canonical_json_dumps(self.provenance)) if isinstance(self.provenance, dict) else {},
        }


PRONOUNS_V101 = [
    # PT
    "isso",
    "aquilo",
    "isto",
    "este",
    "esta",
    "esse",
    "essa",
    # EN
    "this",
    "that",
    "it",
]


def extract_pronouns_v101(*, tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    for t in tokens:
        tt = str(t or "").strip().lower()
        if tt in set(PRONOUNS_V101):
            out.append(tt)
    return out


def infer_kind_hint_v101(*, tokens: Sequence[str]) -> str:
    """
    Deterministic kind hints based on explicit lexical cues.
    No fuzzy matching.
    """
    tset = set([str(t or "").strip().lower() for t in tokens if isinstance(t, str)])
    # Strong hints.
    if ("prioridade" in tset) or ("priority" in tset) or ("objetivo" in tset) or ("goal" in tset):
        return "goal"
    if ("plano" in tset) or ("plan" in tset) or ("passos" in tset) or ("steps" in tset) or ("step" in tset):
        return "plan"
    # Weak hints for shortening requests.
    if ("curto" in tset) or ("curta" in tset) or ("short" in tset) or ("shorter" in tset):
        return "plan"
    return ""


def resolve_reference_v101(
    *,
    pronoun: str,
    kind_hint: str,
    bindings_active: Sequence[Dict[str, Any]],
    current_user_turn_index: int,
    goals_active_ids: Sequence[str],
    margin: int = 50,
    max_candidates: int = 8,
) -> Dict[str, Any]:
    """
    Deterministic resolver:
      - candidates: last N bindings by last_used_turn_index desc (then id asc)
      - optional kind filtering if hint present
      - score = recency + kind_match + goal_active_bonus
      - resolved only if top1 >= top2 + margin (fail-closed otherwise)
    """
    pron = str(pronoun or "")
    hint = str(kind_hint or "")
    goal_active = set([str(x) for x in goals_active_ids if isinstance(x, str) and x])

    candidates0: List[Dict[str, Any]] = []
    for b in bindings_active:
        if not isinstance(b, dict):
            continue
        bid = str(b.get("binding_id") or "")
        bk = str(b.get("binding_kind") or "")
        if not bid or not bk:
            continue
        if hint and bk != hint:
            continue
        candidates0.append(dict(b))

    # If kind hint filtered everything out, treat as MISS (fail-closed).
    if hint and not candidates0:
        return {
            "status": "MISS",
            "pronoun": pron,
            "kind_hint": hint,
            "chosen_binding_id": "",
            "candidates": [],
            "reason": "no_candidates_for_kind_hint",
        }

    # Stable ordering: recency desc, then binding_id asc.
    candidates0.sort(key=lambda b: (-int(b.get("last_used_turn_index") or 0), str(b.get("binding_id") or "")))
    candidates0 = candidates0[: int(max_candidates)]

    scored: List[Dict[str, Any]] = []
    for b in candidates0:
        bid = str(b.get("binding_id") or "")
        bk = str(b.get("binding_kind") or "")
        last_used = int(b.get("last_used_turn_index") or 0)
        dist = int(current_user_turn_index) - int(last_used)
        if dist < 0:
            dist = 0
        recency = max(0, 1000 - min(999, int(dist)))
        kind_match = 100 if (hint and bk == hint) else 0
        goal_bonus = 10 if (bk == "goal" and bid and _binding_points_to_goal_id(b, goal_active)) else 0
        score = int(recency + kind_match + goal_bonus)
        scored.append(
            {
                "binding_id": bid,
                "score": int(score),
                "binding_kind": bk,
                "value_preview": str(b.get("value_preview") or ""),
                "reason": {"recency": int(recency), "kind_match": int(kind_match), "goal_active_bonus": int(goal_bonus)},
            }
        )

    scored.sort(key=lambda d: (-int(d.get("score") or 0), str(d.get("binding_id") or "")))
    if not scored:
        return {"status": "MISS", "pronoun": pron, "kind_hint": hint, "chosen_binding_id": "", "candidates": [], "reason": "no_candidates"}

    if len(scored) == 1:
        return {
            "status": "RESOLVED",
            "pronoun": pron,
            "kind_hint": hint,
            "chosen_binding_id": str(scored[0]["binding_id"]),
            "candidates": list(scored),
            "reason": "single_candidate",
        }

    top1 = int(scored[0].get("score") or 0)
    top2 = int(scored[1].get("score") or 0)
    if top1 >= int(top2 + int(margin)):
        return {
            "status": "RESOLVED",
            "pronoun": pron,
            "kind_hint": hint,
            "chosen_binding_id": str(scored[0]["binding_id"]),
            "candidates": list(scored),
            "reason": f"margin_ok:{int(margin)}",
        }

    return {
        "status": "AMBIGUOUS",
        "pronoun": pron,
        "kind_hint": hint,
        "chosen_binding_id": "",
        "candidates": list(scored),
        "reason": f"margin_too_small:{int(margin)}",
    }


def _binding_points_to_goal_id(binding: Dict[str, Any], active_goal_ids: set) -> bool:
    val = binding.get("value") if isinstance(binding.get("value"), dict) else {}
    gid = str(val.get("goal_id") or "") if isinstance(val, dict) else ""
    return bool(gid and gid in active_goal_ids)

