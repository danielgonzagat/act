from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _plan_event_sig_v104(*, prev_event_sig: str, event_body: Dict[str, Any]) -> str:
    payload = str(prev_event_sig or "") + canonical_json_dumps(event_body)
    return sha256_hex(payload.encode("utf-8"))


def plan_event_id_v104(event_sig: str) -> str:
    return f"plan_event_v104_{str(event_sig)}"


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


def _canon_candidate_v104(c: Any) -> Dict[str, Any]:
    if not isinstance(c, dict):
        return {}
    # Keep candidate schema stable; remove any accidental non-determinism.
    d = dict(c)
    # Canonicalize common fields.
    if "predicted_effects" in d and isinstance(d.get("predicted_effects"), dict):
        pe = d.get("predicted_effects") if isinstance(d.get("predicted_effects"), dict) else {}
        d["predicted_effects"] = {str(k): _canon_int(pe.get(k)) for k in sorted(pe.keys(), key=str)}
    if "dependencies" in d and isinstance(d.get("dependencies"), dict):
        dep = d.get("dependencies") if isinstance(d.get("dependencies"), dict) else {}
        dep2: Dict[str, Any] = {}
        for k in sorted(dep.keys(), key=str):
            v = dep.get(k)
            if isinstance(v, list):
                dep2[str(k)] = [str(x) for x in v if isinstance(x, str) and str(x)]
            elif isinstance(v, dict):
                dep2[str(k)] = {str(kk): dep.get(k).get(kk) for kk in sorted(v.keys(), key=str)}  # type: ignore[union-attr]
            else:
                dep2[str(k)] = v
        d["dependencies"] = dep2
    if "score_breakdown" in d and isinstance(d.get("score_breakdown"), dict):
        sb = d.get("score_breakdown") if isinstance(d.get("score_breakdown"), dict) else {}
        d["score_breakdown"] = {str(k): _canon_int(sb.get(k)) for k in sorted(sb.keys(), key=str)}
    # Canonicalize lists used in explain.
    if "concept_hit_names" in d:
        d["concept_hit_names"] = _canon_str_list(d.get("concept_hit_names"))
    # Ensure core ids are strings.
    for k in ["candidate_id", "plan_kind", "renderer_id", "variant_id", "notes"]:
        if k in d:
            d[k] = str(d.get(k) or "")
    return dict(d)


def _canon_candidates_v104(cands: Any) -> List[Dict[str, Any]]:
    if not isinstance(cands, list):
        return []
    out = []
    for c in cands:
        d = _canon_candidate_v104(c)
        if d:
            out.append(d)
    # Stable order (already sorted by the selector, but keep deterministic).
    out.sort(key=lambda d: (str(d.get("plan_kind") or ""), str(d.get("candidate_id") or "")))
    return list(out)


@dataclass(frozen=True)
class PlanEventV104:
    conversation_id: str
    ts_turn_index: int
    user_turn_id: str
    user_turn_index: int
    intent_id: str
    parse_sig: str
    objective_kind: str
    objective_id: str
    active_plan_state_before: Dict[str, Any]
    active_plan_state_after: Dict[str, Any]
    candidates_topk: List[Dict[str, Any]]
    selected_candidate_id: str
    selected_plan_kind: str
    notes: str
    concept_hit_names: List[str]
    created_step: int
    prev_event_sig: str

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 104,
            "kind": "plan_event_v104",
            "conversation_id": str(self.conversation_id),
            "ts_turn_index": int(self.ts_turn_index),
            "user_turn_id": str(self.user_turn_id or ""),
            "user_turn_index": int(self.user_turn_index),
            "intent_id": str(self.intent_id),
            "parse_sig": str(self.parse_sig),
            "objective_kind": str(self.objective_kind),
            "objective_id": str(self.objective_id),
            "active_plan_state_before": dict(self.active_plan_state_before) if isinstance(self.active_plan_state_before, dict) else {},
            "active_plan_state_after": dict(self.active_plan_state_after) if isinstance(self.active_plan_state_after, dict) else {},
            "candidates_topk": _canon_candidates_v104(self.candidates_topk),
            "selected_candidate_id": str(self.selected_candidate_id),
            "selected_plan_kind": str(self.selected_plan_kind),
            "notes": str(self.notes),
            "concept_hit_names": _canon_str_list(self.concept_hit_names),
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = _plan_event_sig_v104(prev_event_sig=str(self.prev_event_sig or ""), event_body=dict(body))
        return dict(body, prev_event_sig=str(self.prev_event_sig or ""), event_sig=str(sig), event_id=str(plan_event_id_v104(sig)))


def verify_plan_event_sig_v104(ev: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(ev, dict):
        return False, "plan_event_not_dict", {}
    prev_sig = str(ev.get("prev_event_sig") or "")
    want_sig = str(ev.get("event_sig") or "")
    body = dict(ev)
    body.pop("prev_event_sig", None)
    body.pop("event_sig", None)
    body.pop("event_id", None)
    got_sig = _plan_event_sig_v104(prev_event_sig=str(prev_sig), event_body=dict(body))
    if got_sig != want_sig:
        return False, "plan_event_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    eid = str(ev.get("event_id") or "")
    if eid != plan_event_id_v104(want_sig):
        return False, "plan_event_id_mismatch", {"want": plan_event_id_v104(want_sig), "got": str(eid)}
    return True, "ok", {}


def compute_plan_chain_hash_v104(plan_events: List[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in plan_events:
        if not isinstance(ev, dict):
            continue
        ids.append(str(ev.get("event_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))

