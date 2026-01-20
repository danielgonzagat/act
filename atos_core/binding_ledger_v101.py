from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .bindings_v101 import BindingV101, binding_id_v101, value_hash_v101


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _binding_event_sig_v101(*, prev_event_sig: str, event_body: Dict[str, Any]) -> str:
    """
    event_sig = sha256(prev_event_sig + canonical_json(event_body))
    event_body MUST NOT include prev_event_sig/event_sig/event_id to avoid cycles.
    """
    payload = str(prev_event_sig or "") + canonical_json_dumps(event_body)
    return sha256_hex(payload.encode("utf-8"))


def binding_event_id_v101(event_sig: str) -> str:
    return f"binding_event_v101_{str(event_sig)}"


def binding_chain_hash_v101(binding_events: Sequence[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in binding_events:
        if not isinstance(ev, dict):
            continue
        ids.append(str(ev.get("event_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


@dataclass(frozen=True)
class BindingEventV101:
    conversation_id: str
    ts_logical: int
    turn_id: str
    type: str  # BIND_CREATE|BIND_RESOLVE|BIND_AMBIGUOUS|BIND_MISS|BIND_PRUNE
    binding_id: str
    binding_kind: str
    binding_value_hash: str
    binding_value: Dict[str, Any]
    binding_preview: str
    provenance: Dict[str, Any]
    evidence: Dict[str, Any]
    resolution: Dict[str, Any]
    created_step: int
    prev_event_sig: str

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 101,
            "kind": "binding_event_v101",
            "conversation_id": str(self.conversation_id),
            "ts_logical": int(self.ts_logical),
            "turn_id": str(self.turn_id),
            "type": str(self.type),
            "binding_id": str(self.binding_id or ""),
            "binding_kind": str(self.binding_kind or ""),
            "binding_value_hash": str(self.binding_value_hash or ""),
            "binding_value": json.loads(canonical_json_dumps(self.binding_value)) if isinstance(self.binding_value, dict) else {},
            "binding_preview": str(self.binding_preview or ""),
            "provenance": json.loads(canonical_json_dumps(self.provenance)) if isinstance(self.provenance, dict) else {},
            "evidence": json.loads(canonical_json_dumps(self.evidence)) if isinstance(self.evidence, dict) else {},
            "resolution": json.loads(canonical_json_dumps(self.resolution)) if isinstance(self.resolution, dict) else {},
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = _binding_event_sig_v101(prev_event_sig=str(self.prev_event_sig or ""), event_body=dict(body))
        return dict(body, prev_event_sig=str(self.prev_event_sig or ""), event_sig=str(sig), event_id=str(binding_event_id_v101(sig)))


def verify_binding_event_sig_v101(ev: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(ev, dict):
        return False, "binding_event_not_dict", {}
    prev_sig = str(ev.get("prev_event_sig") or "")
    want_sig = str(ev.get("event_sig") or "")
    if not want_sig:
        return False, "missing_binding_event_sig", {}
    body = dict(ev)
    body.pop("prev_event_sig", None)
    body.pop("event_sig", None)
    body.pop("event_id", None)
    got_sig = _binding_event_sig_v101(prev_event_sig=str(prev_sig), event_body=body)
    if got_sig != want_sig:
        return False, "binding_event_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    eid = str(ev.get("event_id") or "")
    if eid != binding_event_id_v101(want_sig):
        return False, "binding_event_id_mismatch", {"want": binding_event_id_v101(want_sig), "got": str(eid)}
    return True, "ok", {}


def fold_bindings_v101(binding_events: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Fold binding events into active bindings map by binding_id.
    Deterministic, fail-closed in verifier (here: best-effort).
    """
    active: Dict[str, Dict[str, Any]] = {}
    for ev in binding_events:
        if not isinstance(ev, dict):
            continue
        et = str(ev.get("type") or "")
        bid = str(ev.get("binding_id") or "")
        if et == "BIND_CREATE":
            if not bid:
                continue
            if bid in active:
                # Keep the first (deterministic); verifier can be stricter.
                continue
            active[bid] = {
                "binding_id": bid,
                "binding_kind": str(ev.get("binding_kind") or ""),
                "value": ev.get("binding_value") if isinstance(ev.get("binding_value"), dict) else {},
                "value_hash": str(ev.get("binding_value_hash") or ""),
                "value_preview": str(ev.get("binding_preview") or ""),
                "created_turn_index": int((ev.get("evidence") or {}).get("created_turn_index") or 0),
                "last_used_turn_index": int((ev.get("evidence") or {}).get("last_used_turn_index") or (ev.get("evidence") or {}).get("created_turn_index") or 0),
                "use_count": int((ev.get("evidence") or {}).get("use_count") or 0),
                "provenance": ev.get("provenance") if isinstance(ev.get("provenance"), dict) else {},
            }
        elif et == "BIND_RESOLVE":
            if bid and bid in active:
                ev_e = ev.get("evidence") if isinstance(ev.get("evidence"), dict) else {}
                if "last_used_turn_index" in ev_e:
                    active[bid]["last_used_turn_index"] = int(ev_e.get("last_used_turn_index") or active[bid].get("last_used_turn_index") or 0)
                if "use_count" in ev_e:
                    active[bid]["use_count"] = int(ev_e.get("use_count") or active[bid].get("use_count") or 0)
        elif et == "BIND_PRUNE":
            if bid:
                active.pop(bid, None)
        else:
            # AMBIGUOUS/MISS: no state change.
            continue
    return dict(active)


def binding_snapshot_v101(binding_events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    active = fold_bindings_v101(binding_events)
    items: List[Dict[str, Any]] = []
    for bid in sorted(active.keys(), key=str):
        b = active[bid]
        items.append(
            BindingV101(
                binding_id=str(b.get("binding_id") or ""),
                binding_kind=str(b.get("binding_kind") or ""),
                value=b.get("value") if isinstance(b.get("value"), dict) else {},
                value_hash=str(b.get("value_hash") or ""),
                value_preview=str(b.get("value_preview") or ""),
                created_turn_index=int(b.get("created_turn_index") or 0),
                last_used_turn_index=int(b.get("last_used_turn_index") or 0),
                use_count=int(b.get("use_count") or 0),
                provenance=b.get("provenance") if isinstance(b.get("provenance"), dict) else {},
            ).to_dict()
        )
    items.sort(key=lambda d: (-int(d.get("last_used_turn_index") or 0), str(d.get("binding_kind") or ""), str(d.get("binding_id") or "")))
    snap = {"schema_version": 101, "kind": "binding_snapshot_v101", "bindings_active": list(items)}
    sig = _stable_hash_obj(snap)
    return dict(snap, snapshot_sig=str(sig))


def make_binding_create_event_v101(
    *,
    conversation_id: str,
    ts_logical: int,
    turn_id: str,
    binding_kind: str,
    value: Dict[str, Any],
    value_preview: str,
    provenance: Dict[str, Any],
    created_turn_index: int,
    created_step: int,
    prev_event_sig: str,
) -> Dict[str, Any]:
    vhash = value_hash_v101(value)
    bid = binding_id_v101(binding_kind=str(binding_kind), value_hash=str(vhash))
    ev = BindingEventV101(
        conversation_id=str(conversation_id),
        ts_logical=int(ts_logical),
        turn_id=str(turn_id),
        type="BIND_CREATE",
        binding_id=str(bid),
        binding_kind=str(binding_kind),
        binding_value_hash=str(vhash),
        binding_value=dict(value),
        binding_preview=str(value_preview or ""),
        provenance=dict(provenance),
        evidence={"use_count": 0, "created_turn_index": int(created_turn_index), "last_used_turn_index": int(created_turn_index)},
        resolution={},
        created_step=int(created_step),
        prev_event_sig=str(prev_event_sig or ""),
    ).to_dict()
    return dict(ev)


def make_binding_resolve_event_v101(
    *,
    conversation_id: str,
    ts_logical: int,
    turn_id: str,
    pronoun: str,
    resolution: Dict[str, Any],
    chosen_binding_id: str,
    binding_kind: str,
    binding_value_hash: str,
    created_step: int,
    prev_event_sig: str,
    use_count: int,
    last_used_turn_index: int,
) -> Dict[str, Any]:
    ev = BindingEventV101(
        conversation_id=str(conversation_id),
        ts_logical=int(ts_logical),
        turn_id=str(turn_id),
        type="BIND_RESOLVE",
        binding_id=str(chosen_binding_id),
        binding_kind=str(binding_kind),
        binding_value_hash=str(binding_value_hash),
        binding_value={},
        binding_preview="",
        provenance={},
        evidence={"use_count": int(use_count), "last_used_turn_index": int(last_used_turn_index)},
        resolution={"pronoun": str(pronoun), **(dict(resolution) if isinstance(resolution, dict) else {})},
        created_step=int(created_step),
        prev_event_sig=str(prev_event_sig or ""),
    ).to_dict()
    return dict(ev)


def make_binding_ambiguous_event_v101(
    *,
    conversation_id: str,
    ts_logical: int,
    turn_id: str,
    pronoun: str,
    resolution: Dict[str, Any],
    created_step: int,
    prev_event_sig: str,
) -> Dict[str, Any]:
    ev = BindingEventV101(
        conversation_id=str(conversation_id),
        ts_logical=int(ts_logical),
        turn_id=str(turn_id),
        type="BIND_AMBIGUOUS",
        binding_id="",
        binding_kind="",
        binding_value_hash="",
        binding_value={},
        binding_preview="",
        provenance={},
        evidence={},
        resolution={"pronoun": str(pronoun), **(dict(resolution) if isinstance(resolution, dict) else {})},
        created_step=int(created_step),
        prev_event_sig=str(prev_event_sig or ""),
    ).to_dict()
    return dict(ev)


def make_binding_miss_event_v101(
    *,
    conversation_id: str,
    ts_logical: int,
    turn_id: str,
    pronoun: str,
    resolution: Dict[str, Any],
    created_step: int,
    prev_event_sig: str,
) -> Dict[str, Any]:
    ev = BindingEventV101(
        conversation_id=str(conversation_id),
        ts_logical=int(ts_logical),
        turn_id=str(turn_id),
        type="BIND_MISS",
        binding_id="",
        binding_kind="",
        binding_value_hash="",
        binding_value={},
        binding_preview="",
        provenance={},
        evidence={},
        resolution={"pronoun": str(pronoun), **(dict(resolution) if isinstance(resolution, dict) else {})},
        created_step=int(created_step),
        prev_event_sig=str(prev_event_sig or ""),
    ).to_dict()
    return dict(ev)


def make_binding_prune_event_v101(
    *,
    conversation_id: str,
    ts_logical: int,
    turn_id: str,
    binding_id: str,
    binding_kind: str,
    binding_value_hash: str,
    reason: str,
    created_step: int,
    prev_event_sig: str,
) -> Dict[str, Any]:
    ev = BindingEventV101(
        conversation_id=str(conversation_id),
        ts_logical=int(ts_logical),
        turn_id=str(turn_id),
        type="BIND_PRUNE",
        binding_id=str(binding_id),
        binding_kind=str(binding_kind),
        binding_value_hash=str(binding_value_hash),
        binding_value={},
        binding_preview="",
        provenance={},
        evidence={},
        resolution={"reason": str(reason)},
        created_step=int(created_step),
        prev_event_sig=str(prev_event_sig or ""),
    ).to_dict()
    return dict(ev)


def choose_prune_candidate_v101(active_bindings: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Deterministic LRU prune: oldest last_used_turn_index, then lowest use_count, then binding_id asc.
    """
    cands: List[Dict[str, Any]] = []
    for b in active_bindings:
        if not isinstance(b, dict):
            continue
        bid = str(b.get("binding_id") or "")
        if not bid:
            continue
        cands.append(dict(b))
    if not cands:
        return None
    cands.sort(
        key=lambda b: (
            int(b.get("last_used_turn_index") or 0),
            int(b.get("use_count") or 0),
            str(b.get("binding_id") or ""),
        )
    )
    return dict(cands[0])

