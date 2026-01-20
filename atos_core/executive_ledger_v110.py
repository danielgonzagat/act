from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex


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


def _canon_candidate_action_v110(c: Any) -> Dict[str, Any]:
    if not isinstance(c, dict):
        return {}
    d = dict(c)
    d["kind"] = str(d.get("kind") or "")
    d["score_int"] = _canon_int(d.get("score_int"))
    d["reason_codes"] = _canon_str_list(d.get("reason_codes"))
    d["required_slots"] = _canon_str_list(d.get("required_slots"))
    d["goal_id"] = str(d.get("goal_id") or "")
    return d


def _canon_candidate_actions_v110(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for x in items:
        d = _canon_candidate_action_v110(x)
        if d:
            out.append(d)
    out.sort(key=lambda d: (-int(d.get("score_int") or 0), str(d.get("kind") or ""), str(d.get("goal_id") or "")))
    return list(out)


def _executive_event_sig_v110(*, prev_event_sig: str, event_body: Dict[str, Any]) -> str:
    payload = str(prev_event_sig or "") + canonical_json_dumps(event_body)
    return sha256_hex(payload.encode("utf-8"))


def executive_event_id_v110(event_sig: str) -> str:
    return f"executive_event_v110_{str(event_sig)}"


@dataclass(frozen=True)
class ExecutiveEventV110:
    conversation_id: str
    ts_turn_index: int
    user_turn_id: str
    user_turn_index: int
    goal_id: str
    goal_status: str
    plan_step_index_before: int
    plan_step_index_after: int
    missing_slots: List[str]
    pending_kind: str
    candidate_actions_topk: List[Dict[str, Any]]
    chosen_kind: str
    chosen_reason_codes: List[str]
    executive_score_v110: int
    executive_flags_v110: List[str]
    progress_allowed_v110: bool
    repair_action_v110: str
    created_step: int
    prev_event_sig: str

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 110,
            "kind": "executive_event_v110",
            "conversation_id": str(self.conversation_id),
            "ts_turn_index": int(self.ts_turn_index),
            "user_turn_id": str(self.user_turn_id or ""),
            "user_turn_index": int(self.user_turn_index),
            "goal_id": str(self.goal_id or ""),
            "goal_status": str(self.goal_status or ""),
            "plan_step_index_before": int(self.plan_step_index_before),
            "plan_step_index_after": int(self.plan_step_index_after),
            "missing_slots": _canon_str_list(self.missing_slots),
            "pending_kind": str(self.pending_kind or ""),
            "candidate_actions_topk": _canon_candidate_actions_v110(self.candidate_actions_topk),
            "chosen_kind": str(self.chosen_kind or ""),
            "chosen_reason_codes": _canon_str_list(self.chosen_reason_codes),
            "executive_score_v110": int(max(0, min(100, int(self.executive_score_v110)))),
            "executive_flags_v110": _canon_str_list(self.executive_flags_v110),
            "progress_allowed_v110": bool(self.progress_allowed_v110),
            "repair_action_v110": str(self.repair_action_v110 or ""),
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = _executive_event_sig_v110(prev_event_sig=str(self.prev_event_sig or ""), event_body=dict(body))
        return dict(body, prev_event_sig=str(self.prev_event_sig or ""), event_sig=str(sig), event_id=str(executive_event_id_v110(sig)))


def verify_executive_event_sig_v110(ev: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(ev, dict):
        return False, "executive_event_not_dict", {}
    prev_sig = str(ev.get("prev_event_sig") or "")
    want_sig = str(ev.get("event_sig") or "")
    body = dict(ev)
    body.pop("prev_event_sig", None)
    body.pop("event_sig", None)
    body.pop("event_id", None)
    got_sig = _executive_event_sig_v110(prev_event_sig=str(prev_sig), event_body=dict(body))
    if got_sig != want_sig:
        return False, "executive_event_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    eid = str(ev.get("event_id") or "")
    if eid != executive_event_id_v110(want_sig):
        return False, "executive_event_id_mismatch", {"want": executive_event_id_v110(want_sig), "got": str(eid)}
    return True, "ok", {}


def compute_executive_chain_hash_v110(executive_events: Sequence[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in executive_events:
        if not isinstance(ev, dict):
            continue
        ids.append(str(ev.get("event_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


def fold_executive_ledger_v110(executive_events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # Minimal snapshot: last turn + last goal + last plan step index.
    last_turn_index = -1
    last_goal_id = ""
    last_goal_status = ""
    last_step_after = 0
    flags_counts: Dict[str, int] = {}
    for ev in executive_events:
        if not isinstance(ev, dict):
            continue
        ti = _canon_int(ev.get("user_turn_index"))
        if ti >= 0 and ti >= last_turn_index:
            last_turn_index = int(ti)
            last_goal_id = str(ev.get("goal_id") or "")
            last_goal_status = str(ev.get("goal_status") or "")
            last_step_after = _canon_int(ev.get("plan_step_index_after"))
        for fl in _canon_str_list(ev.get("executive_flags_v110")):
            flags_counts[fl] = int(flags_counts.get(fl, 0)) + 1
    chain_hash = compute_executive_chain_hash_v110(list(executive_events))
    snap = {
        "schema_version": 110,
        "kind": "executive_registry_snapshot_v110",
        "executive_chain_hash_v110": str(chain_hash),
        "executive_events_total_v110": int(len([1 for ev in executive_events if isinstance(ev, dict)])),
        "last_user_turn_index": int(last_turn_index),
        "last_goal_id": str(last_goal_id),
        "last_goal_status": str(last_goal_status),
        "last_plan_step_index": int(last_step_after),
        "flags_counts": {str(k): int(flags_counts.get(k, 0)) for k in sorted(flags_counts.keys(), key=str)},
    }
    sem = dict(snap)
    sem.pop("snapshot_sig", None)
    snap["snapshot_sig"] = _stable_hash_obj(sem)
    return snap


def render_executive_text_v110(*, snapshot: Dict[str, Any]) -> str:
    if not isinstance(snapshot, dict):
        return "EXECUTIVE: (invalid)"
    last_goal = str(snapshot.get("last_goal_id") or "")
    last_step = _canon_int(snapshot.get("last_plan_step_index"))
    total = _canon_int(snapshot.get("executive_events_total_v110"))
    return f"EXECUTIVE: events={total} last_goal={last_goal} last_step_index={last_step}"


def lookup_executive_event_v110(*, executive_events: Sequence[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
    s = str(query or "").strip()
    if not s:
        return None
    try:
        qidx = int(s)
    except Exception:
        return None
    for ev in executive_events:
        if not isinstance(ev, dict):
            continue
        if _canon_int(ev.get("user_turn_index")) == int(qidx):
            return dict(ev)
    return None


def render_explain_executive_text_v110(*, executive_event: Dict[str, Any]) -> str:
    if not isinstance(executive_event, dict) or not executive_event:
        return "EXPLAIN_EXECUTIVE: not_found"
    uidx = _canon_int(executive_event.get("user_turn_index"))
    gid = str(executive_event.get("goal_id") or "")
    chosen = str(executive_event.get("chosen_kind") or "")
    ok = bool(executive_event.get("progress_allowed_v110", False))
    score = _canon_int(executive_event.get("executive_score_v110"))
    flags = _canon_str_list(executive_event.get("executive_flags_v110"))
    cands = executive_event.get("candidate_actions_topk") if isinstance(executive_event.get("candidate_actions_topk"), list) else []
    parts: List[str] = []
    for i, c in enumerate(cands[:3]):
        if not isinstance(c, dict):
            continue
        parts.append(
            f"{i+1}) {str(c.get('kind') or '')} score={_canon_int(c.get('score_int'))} req={json.dumps(_canon_str_list(c.get('required_slots')), ensure_ascii=False)}"
        )
    ranked = "; ".join(parts) if parts else "(none)"
    lines: List[str] = []
    lines.append(f"EXPLAIN_EXECUTIVE: user_turn_index={uidx} goal={gid} chosen={chosen} ok={str(ok).lower()} score={score}")
    lines.append(f"RANKED: {ranked}")
    lines.append("FLAGS: " + json.dumps(flags, ensure_ascii=False))
    lines.append(f"REPAIR: {str(executive_event.get('repair_action_v110') or '')}")
    return "\n".join(lines)


def render_trace_executive_text_v110(*, executive_events: Sequence[Dict[str, Any]], until_user_turn_index: int) -> str:
    out: List[Dict[str, Any]] = []
    for ev in executive_events:
        if not isinstance(ev, dict):
            continue
        if _canon_int(ev.get("user_turn_index")) <= int(until_user_turn_index):
            out.append(dict(ev))
    out.sort(key=lambda d: _canon_int(d.get("user_turn_index")))
    return json.dumps(out, ensure_ascii=False, sort_keys=True, indent=2)
