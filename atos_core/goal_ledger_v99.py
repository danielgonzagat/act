from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def goal_id_v99(*, conversation_id: str, ts_turn_index: int, text: str, parent_goal_id: str) -> str:
    body = {
        "schema_version": 99,
        "conversation_id": str(conversation_id),
        "ts_turn_index": int(ts_turn_index),
        "text": str(text),
        "parent_goal_id": str(parent_goal_id or ""),
    }
    return f"goal_v99_{_stable_hash_obj(body)}"


def _goal_event_sig_v99(*, prev_event_sig: str, event_body: Dict[str, Any]) -> str:
    """
    event_sig = sha256(prev_event_sig + canonical_json(event_body))
    event_body MUST NOT include prev_event_sig/event_sig/event_id to avoid cycles.
    """
    payload = str(prev_event_sig or "") + canonical_json_dumps(event_body)
    return sha256_hex(payload.encode("utf-8"))


def goal_event_id_v99(event_sig: str) -> str:
    return f"goal_event_v99_{str(event_sig)}"


@dataclass(frozen=True)
class GoalEventV99:
    conversation_id: str
    ts_turn_index: int
    op: str
    goal_id: str
    parent_goal_id: str
    priority: int
    status: str
    text: str
    cause_type: str
    cause_id: str
    created_step: int
    prev_event_sig: str

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 99,
            "kind": "goal_event_v99",
            "conversation_id": str(self.conversation_id),
            "ts_turn_index": int(self.ts_turn_index),
            "op": str(self.op),
            "goal_id": str(self.goal_id),
            "parent_goal_id": str(self.parent_goal_id or ""),
            "priority": int(self.priority),
            "status": str(self.status),
            "text": str(self.text),
            "cause": {"cause_type": str(self.cause_type), "cause_id": str(self.cause_id)},
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = _goal_event_sig_v99(prev_event_sig=str(self.prev_event_sig or ""), event_body=dict(body))
        return dict(body, prev_event_sig=str(self.prev_event_sig or ""), event_sig=str(sig), event_id=str(goal_event_id_v99(sig)))


def verify_goal_event_sig_v99(ev: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(ev, dict):
        return False, "goal_event_not_dict", {}
    prev_sig = str(ev.get("prev_event_sig") or "")
    want_sig = str(ev.get("event_sig") or "")
    body = dict(ev)
    body.pop("prev_event_sig", None)
    body.pop("event_sig", None)
    body.pop("event_id", None)
    got_sig = _goal_event_sig_v99(prev_event_sig=str(prev_sig), event_body=body)
    if got_sig != want_sig:
        return False, "goal_event_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    eid = str(ev.get("event_id") or "")
    if eid != goal_event_id_v99(want_sig):
        return False, "goal_event_id_mismatch", {"want": goal_event_id_v99(want_sig), "got": str(eid)}
    return True, "ok", {}


def compute_goal_chain_hash_v99(goal_events: List[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in goal_events:
        if not isinstance(ev, dict):
            continue
        ids.append(str(ev.get("event_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


def fold_goal_ledger_v99(goal_events: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
      active_by_id: goal_id -> goal_state (includes created_ts_turn_index)
      details: {created_ts_by_goal_id, goal_events_total}
    """
    created_ts: Dict[str, int] = {}
    all_by_id: Dict[str, Dict[str, Any]] = {}
    active_by_id: Dict[str, Dict[str, Any]] = {}

    for ev in goal_events:
        if not isinstance(ev, dict):
            continue
        op = str(ev.get("op") or "")
        gid = str(ev.get("goal_id") or "")
        if not gid:
            continue
        if op == "GOAL_ADD":
            try:
                ts = int(ev.get("ts_turn_index", 0))
            except Exception:
                ts = 0
            created_ts.setdefault(gid, int(ts))
            st = {
                "goal_id": str(gid),
                "parent_goal_id": str(ev.get("parent_goal_id") or ""),
                "priority": int(ev.get("priority") or 0),
                "status": "active",
                "text": str(ev.get("text") or ""),
                "created_ts_turn_index": int(created_ts.get(gid, int(ts))),
                "created_event_id": str(ev.get("event_id") or ""),
                "last_event_id": str(ev.get("event_id") or ""),
            }
            all_by_id[gid] = dict(st)
            active_by_id[gid] = dict(st)
        elif op in {"GOAL_UPDATE", "GOAL_DONE", "GOAL_RETRACT"}:
            prev = dict(all_by_id.get(gid) or {})
            if not prev:
                # Ignore updates for unknown goals; verifier will fail-closed if needed.
                continue
            prev["last_event_id"] = str(ev.get("event_id") or "")
            if op == "GOAL_UPDATE":
                if "text" in ev:
                    prev["text"] = str(ev.get("text") or prev.get("text") or "")
                if "priority" in ev:
                    try:
                        prev["priority"] = int(ev.get("priority") or prev.get("priority") or 0)
                    except Exception:
                        pass
                prev["status"] = "active"
                all_by_id[gid] = dict(prev)
                active_by_id[gid] = dict(prev)
            elif op == "GOAL_DONE":
                prev["status"] = "done"
                all_by_id[gid] = dict(prev)
                active_by_id.pop(gid, None)
            elif op == "GOAL_RETRACT":
                prev["status"] = "retracted"
                all_by_id[gid] = dict(prev)
                active_by_id.pop(gid, None)

    details = {"created_ts_by_goal_id": dict(created_ts), "goal_events_total": int(len(goal_events))}
    return dict(active_by_id), dict(details)


def goal_ledger_snapshot_v99(goal_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    active_by_id, _details = fold_goal_ledger_v99(list(goal_events))
    items: List[Dict[str, Any]] = []
    for gid in sorted(active_by_id.keys(), key=str):
        it = dict(active_by_id[gid])
        items.append(
            {
                "goal_id": str(it.get("goal_id") or ""),
                "parent_goal_id": str(it.get("parent_goal_id") or ""),
                "priority": int(it.get("priority") or 0),
                "status": str(it.get("status") or ""),
                "text": str(it.get("text") or ""),
                "created_ts_turn_index": int(it.get("created_ts_turn_index") or 0),
                "created_event_id": str(it.get("created_event_id") or ""),
                "last_event_id": str(it.get("last_event_id") or ""),
            }
        )
    items.sort(key=lambda d: (-int(d.get("priority") or 0), int(d.get("created_ts_turn_index") or 0), str(d.get("goal_id") or "")))
    snap = {"schema_version": 99, "kind": "goal_ledger_snapshot_v99", "goals_active": list(items)}
    snap_sig = _stable_hash_obj(snap)
    return dict(snap, snapshot_sig=str(snap_sig))

