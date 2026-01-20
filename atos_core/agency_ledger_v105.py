from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex

AGENCY_KIND_ASK_CLARIFY_V105 = "ASK_CLARIFY"
AGENCY_KIND_CONTINUE_PLAN_V105 = "CONTINUE_PLAN"
AGENCY_KIND_PROPOSE_OPTIONS_V105 = "PROPOSE_OPTIONS"
AGENCY_KIND_EXECUTE_STEP_V105 = "EXECUTE_STEP"
AGENCY_KIND_CLOSE_GOAL_V105 = "CLOSE_GOAL"
AGENCY_KIND_IDLE_V105 = "IDLE"


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _agency_event_sig_v105(*, prev_event_sig: str, event_body: Dict[str, Any]) -> str:
    payload = str(prev_event_sig or "") + canonical_json_dumps(event_body)
    return sha256_hex(payload.encode("utf-8"))


def agency_event_id_v105(event_sig: str) -> str:
    return f"agency_event_v105_{str(event_sig)}"


def _canon_str_list(xs: Any) -> List[str]:
    if not isinstance(xs, list):
        return []
    out = [str(x) for x in xs if isinstance(x, str) and str(x)]
    return sorted(set(out), key=str)


def _canon_int(x: Any, *, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _canon_slots_state_v105(slots_state: Any) -> Dict[str, Any]:
    if not isinstance(slots_state, dict):
        return {"slots": {}, "missing_slots": [], "pending": {}}
    slots = slots_state.get("slots") if isinstance(slots_state.get("slots"), dict) else {}
    slots2: Dict[str, Any] = {}
    for k in sorted(slots.keys(), key=str):
        kk = str(k)
        v = slots.get(k)
        # Keep values JSON-safe; prefer strings for safety.
        if v is None:
            slots2[kk] = ""
        elif isinstance(v, (str, int, float, bool)):
            slots2[kk] = v
        else:
            slots2[kk] = str(v)

    missing = _canon_str_list(slots_state.get("missing_slots"))
    pending = slots_state.get("pending") if isinstance(slots_state.get("pending"), dict) else {}
    pending2: Dict[str, Any] = {
        "kind": str(pending.get("kind") or ""),
        "slot": str(pending.get("slot") or ""),
        "options": [],
    }
    opts = pending.get("options") if isinstance(pending.get("options"), list) else []
    opt_rows: List[Dict[str, Any]] = []
    for o in opts:
        if not isinstance(o, dict):
            continue
        opt_rows.append({"label": str(o.get("label") or ""), "plan_kind": str(o.get("plan_kind") or "")})
    opt_rows.sort(key=lambda d: (str(d.get("label") or ""), str(d.get("plan_kind") or "")))
    pending2["options"] = list(opt_rows)

    return {"slots": dict(slots2), "missing_slots": list(missing), "pending": dict(pending2)}


def _canon_candidate_action_v105(c: Any) -> Dict[str, Any]:
    if not isinstance(c, dict):
        return {}
    d = dict(c)
    out = {
        "kind": str(d.get("kind") or ""),
        "score_int": _canon_int(d.get("score_int")),
        "reason_codes": _canon_str_list(d.get("reason_codes")),
        "required_slots": _canon_str_list(d.get("required_slots")),
        "goal_id": str(d.get("goal_id") or ""),
        "plan_kind": str(d.get("plan_kind") or ""),
        "plan_candidate_id": str(d.get("plan_candidate_id") or ""),
        "choice_label": str(d.get("choice_label") or ""),
    }
    return dict(out)


def _canon_candidates_v105(cands: Any) -> List[Dict[str, Any]]:
    if not isinstance(cands, list):
        return []
    out: List[Dict[str, Any]] = []
    for c in cands:
        d = _canon_candidate_action_v105(c)
        if d.get("kind"):
            out.append(d)
    out.sort(key=lambda d: (-int(d.get("score_int") or 0), str(d.get("kind") or ""), str(d.get("goal_id") or ""), str(d.get("plan_candidate_id") or "")))
    return list(out)


@dataclass(frozen=True)
class AgencyEventV105:
    conversation_id: str
    ts_turn_index: int
    user_turn_id: str
    user_turn_index: int
    goal_id: str
    plan_turn_index_ref: int
    slots_state: Dict[str, Any]
    candidates_topk: List[Dict[str, Any]]
    chosen_kind: str
    chosen_reason_codes: List[str]
    chosen_candidate_id: str
    created_step: int
    prev_event_sig: str

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 105,
            "kind": "agency_event_v105",
            "conversation_id": str(self.conversation_id),
            "ts_turn_index": int(self.ts_turn_index),
            "user_turn_id": str(self.user_turn_id or ""),
            "user_turn_index": int(self.user_turn_index),
            "goal_id": str(self.goal_id or ""),
            "plan_turn_index_ref": int(self.plan_turn_index_ref),
            "slots_state": _canon_slots_state_v105(self.slots_state),
            "candidates_topk": _canon_candidates_v105(self.candidates_topk),
            "chosen_kind": str(self.chosen_kind or ""),
            "chosen_reason_codes": _canon_str_list(self.chosen_reason_codes),
            "chosen_candidate_id": str(self.chosen_candidate_id or ""),
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = _agency_event_sig_v105(prev_event_sig=str(self.prev_event_sig or ""), event_body=dict(body))
        return dict(body, prev_event_sig=str(self.prev_event_sig or ""), event_sig=str(sig), event_id=str(agency_event_id_v105(sig)))


def verify_agency_event_sig_v105(ev: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(ev, dict):
        return False, "agency_event_not_dict", {}
    prev_sig = str(ev.get("prev_event_sig") or "")
    want_sig = str(ev.get("event_sig") or "")
    body = dict(ev)
    body.pop("prev_event_sig", None)
    body.pop("event_sig", None)
    body.pop("event_id", None)
    got_sig = _agency_event_sig_v105(prev_event_sig=str(prev_sig), event_body=dict(body))
    if got_sig != want_sig:
        return False, "agency_event_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    want_id = agency_event_id_v105(want_sig)
    got_id = str(ev.get("event_id") or "")
    if got_id != want_id:
        return False, "agency_event_id_mismatch", {"want": str(want_id), "got": str(got_id)}
    return True, "ok", {}


def compute_agency_chain_hash_v105(agency_events: List[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in agency_events:
        if not isinstance(ev, dict):
            continue
        ids.append(str(ev.get("event_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


def fold_agency_ledger_v105(agency_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Replay/fold derived state from the agency ledger (slots + pending).
    Returns a deterministic snapshot-like dict (NOT including hashes).
    """
    by_goal: Dict[str, Dict[str, Any]] = {}
    for ev in agency_events:
        if not isinstance(ev, dict):
            continue
        gid = str(ev.get("goal_id") or "")
        if not gid:
            continue
        ss = ev.get("slots_state") if isinstance(ev.get("slots_state"), dict) else {}
        by_goal[gid] = _canon_slots_state_v105(ss)
        # Track last chosen kind for convenience (not used for verification logic).
        by_goal[gid]["last_chosen_kind"] = str(ev.get("chosen_kind") or "")
        by_goal[gid]["last_turn_index"] = _canon_int(ev.get("ts_turn_index"))
        by_goal[gid]["last_event_id"] = str(ev.get("event_id") or "")
    return {"schema_version": 105, "kind": "agency_registry_v105", "by_goal": {str(k): dict(by_goal[k]) for k in sorted(by_goal.keys(), key=str)}}


def agency_registry_snapshot_v105(agency_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    snap = fold_agency_ledger_v105(list(agency_events))
    sig = _stable_hash_obj(snap)
    return dict(snap, snapshot_sig=str(sig))


def render_agency_text_v105(snapshot: Dict[str, Any]) -> str:
    by_goal = snapshot.get("by_goal") if isinstance(snapshot.get("by_goal"), dict) else {}
    if not by_goal:
        return "AGENCY: (empty)"
    lines: List[str] = ["AGENCY:"]
    for gid in sorted(by_goal.keys(), key=str):
        st = by_goal.get(gid) if isinstance(by_goal.get(gid), dict) else {}
        slots = st.get("slots") if isinstance(st.get("slots"), dict) else {}
        missing = st.get("missing_slots") if isinstance(st.get("missing_slots"), list) else []
        pending = st.get("pending") if isinstance(st.get("pending"), dict) else {}
        lines.append(
            f"- goal_id={gid} missing={canonical_json_dumps([str(x) for x in missing if str(x)])} pending_kind={str(pending.get('kind') or '')} pending_slot={str(pending.get('slot') or '')} slots={canonical_json_dumps(slots)}"
        )
    return "\n".join(lines)


def render_explain_agency_text_v105(event: Dict[str, Any]) -> str:
    if not isinstance(event, dict):
        return "EXPLAIN_AGENCY: not_found"
    tid = str(event.get("user_turn_index") or "")
    gid = str(event.get("goal_id") or "")
    chosen_kind = str(event.get("chosen_kind") or "")
    rc = event.get("chosen_reason_codes") if isinstance(event.get("chosen_reason_codes"), list) else []
    rc2 = [str(x) for x in rc if str(x)]
    cands = event.get("candidates_topk") if isinstance(event.get("candidates_topk"), list) else []
    parts: List[str] = []
    for c in cands[:3]:
        if not isinstance(c, dict):
            continue
        parts.append(f"{str(c.get('kind') or '')}:{int(c.get('score_int') or 0)}")
    return "\n".join(
        [
            f"EXPLAIN_AGENCY: turn_index={tid} goal_id={gid} chosen={chosen_kind} reasons={canonical_json_dumps(sorted(set(rc2), key=str))}",
            "CANDIDATES: " + "; ".join(parts) if parts else "CANDIDATES: (none)",
        ]
    )


def render_trace_agency_text_v105(*, agency_events: List[Dict[str, Any]], until_turn_index: int) -> str:
    ids: List[str] = []
    for ev in agency_events:
        if not isinstance(ev, dict):
            continue
        try:
            ti = int(ev.get("ts_turn_index") or 0)
        except Exception:
            ti = 0
        if ti > int(until_turn_index):
            break
        ids.append(str(ev.get("event_id") or ""))
    return "TRACE_AGENCY: " + canonical_json_dumps(ids)


def lookup_agency_event_v105(*, agency_events: List[Dict[str, Any]], user_turn_index: int) -> Optional[Dict[str, Any]]:
    for ev in agency_events:
        if not isinstance(ev, dict):
            continue
        try:
            idx = int(ev.get("user_turn_index") or -1)
        except Exception:
            idx = -1
        if idx == int(user_turn_index):
            return dict(ev)
    return None

