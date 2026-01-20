from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _concept_event_sig_v103(*, prev_event_sig: str, event_body: Dict[str, Any]) -> str:
    """
    event_sig = sha256(prev_event_sig + canonical_json(event_body))
    event_body MUST NOT include prev_event_sig/event_sig/event_id to avoid cycles.
    """
    payload = str(prev_event_sig or "") + canonical_json_dumps(event_body)
    return sha256_hex(payload.encode("utf-8"))


def concept_event_id_v103(event_sig: str) -> str:
    return f"concept_event_v103_{str(event_sig)}"


@dataclass(frozen=True)
class ConceptEventV103:
    conversation_id: str
    ts_turn_index: int
    turn_id: str
    type: str
    payload: Dict[str, Any]
    created_step: int
    prev_event_sig: str

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 103,
            "kind": "concept_event_v103",
            "conversation_id": str(self.conversation_id),
            "ts_turn_index": int(self.ts_turn_index),
            "turn_id": str(self.turn_id or ""),
            "type": str(self.type),
            "payload": dict(self.payload),
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = _concept_event_sig_v103(prev_event_sig=str(self.prev_event_sig or ""), event_body=dict(body))
        return dict(body, prev_event_sig=str(self.prev_event_sig or ""), event_sig=str(sig), event_id=str(concept_event_id_v103(sig)))


def verify_concept_event_sig_v103(ev: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(ev, dict):
        return False, "concept_event_not_dict", {}
    prev_sig = str(ev.get("prev_event_sig") or "")
    want_sig = str(ev.get("event_sig") or "")
    body = dict(ev)
    body.pop("prev_event_sig", None)
    body.pop("event_sig", None)
    body.pop("event_id", None)
    got_sig = _concept_event_sig_v103(prev_event_sig=str(prev_sig), event_body=body)
    if got_sig != want_sig:
        return False, "concept_event_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    eid = str(ev.get("event_id") or "")
    if eid != concept_event_id_v103(want_sig):
        return False, "concept_event_id_mismatch", {"want": concept_event_id_v103(want_sig), "got": str(eid)}
    return True, "ok", {}


def compute_concept_chain_hash_v103(concept_events: List[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in concept_events:
        if not isinstance(ev, dict):
            continue
        ids.append(str(ev.get("event_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))

