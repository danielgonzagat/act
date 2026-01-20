from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex


EXTERNAL_WORLD_ACTION_SEARCH_V111 = "SEARCH_PAST_DIALOGUE"
EXTERNAL_WORLD_ACTION_FETCH_V111 = "FETCH_CONTEXT"
EXTERNAL_WORLD_ACTION_OBSERVE_V111 = "OBSERVE_PAST_DIALOGUE"


EXTERNAL_WORLD_REASON_CODES_V111 = {
    "validator_failed_unresolved_reference",
    "validator_failed_goal_stall",
    "validator_failed_semantic_inconsistency",
    "validator_failed_fluency_contract",
}


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _canon_str(x: Any) -> str:
    return str(x) if isinstance(x, str) else str(x or "")


def _canon_int(x: Any, *, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _canon_json_obj(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_canon_json_obj(x) for x in obj]
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k in sorted([str(k) for k in obj.keys()], key=str):
            out[str(k)] = _canon_json_obj(obj.get(k))
        return dict(out)
    return str(obj)


def _external_world_event_sig_v111(*, prev_event_sig: str, event_body: Dict[str, Any]) -> str:
    payload = str(prev_event_sig or "") + canonical_json_dumps(event_body)
    return sha256_hex(payload.encode("utf-8"))


def external_world_event_id_v111(event_sig: str) -> str:
    return f"external_world_event_v111_{str(event_sig)}"


@dataclass(frozen=True)
class ExternalWorldEventV111:
    event_index: int
    turn_index: int
    action: str
    reason_code: str
    args: Dict[str, Any]
    result_summary: Dict[str, Any]
    prev_event_sig: str
    event_sig: str


def make_external_world_event_v111(
    *,
    event_index: int,
    turn_index: int,
    action: str,
    reason_code: str,
    args: Dict[str, Any],
    result_summary: Dict[str, Any],
    prev_event_sig: str,
) -> ExternalWorldEventV111:
    body = {
        "event_index": _canon_int(event_index, default=0),
        "turn_index": _canon_int(turn_index, default=0),
        "action": _canon_str(action),
        "reason_code": _canon_str(reason_code),
        "args": _canon_json_obj(args),
        "result_summary": _canon_json_obj(result_summary),
        "prev_event_sig": _canon_str(prev_event_sig),
    }
    sig = _external_world_event_sig_v111(prev_event_sig=str(prev_event_sig or ""), event_body=dict(body))
    return ExternalWorldEventV111(
        event_index=int(body["event_index"]),
        turn_index=int(body["turn_index"]),
        action=str(body["action"]),
        reason_code=str(body["reason_code"]),
        args=dict(body["args"]) if isinstance(body["args"], dict) else {},
        result_summary=dict(body["result_summary"]) if isinstance(body["result_summary"], dict) else {},
        prev_event_sig=str(body["prev_event_sig"]),
        event_sig=str(sig),
    )


def external_world_event_to_dict_v111(ev: ExternalWorldEventV111) -> Dict[str, Any]:
    return {
        "event_id": external_world_event_id_v111(str(ev.event_sig)),
        "event_index": int(ev.event_index),
        "turn_index": int(ev.turn_index),
        "action": str(ev.action),
        "reason_code": str(ev.reason_code),
        "args": _canon_json_obj(ev.args),
        "result_summary": _canon_json_obj(ev.result_summary),
        "prev_event_sig": str(ev.prev_event_sig),
        "event_sig": str(ev.event_sig),
    }


def compute_external_world_chain_hash_v111(events: Sequence[Dict[str, Any]]) -> str:
    sigs: List[str] = []
    for e in events:
        if isinstance(e, dict):
            sigs.append(str(e.get("event_sig") or ""))
    return _stable_hash_obj(sigs)


def verify_external_world_event_sig_chain_v111(events: Sequence[Dict[str, Any]]) -> Tuple[bool, str, Dict[str, Any]]:
    prev = ""
    prev_idx = -1
    for i, e in enumerate(events):
        if not isinstance(e, dict):
            return False, "external_world_event_not_dict", {"i": int(i)}
        try:
            idx = int(e.get("event_index", -1))
        except Exception:
            idx = -1
        if idx != i:
            return False, "external_world_event_index_mismatch", {"i": int(i), "event_index": idx}
        if prev_idx >= idx:
            return False, "external_world_event_index_not_monotonic", {"i": int(i)}
        prev_idx = idx

        action = str(e.get("action") or "")
        if action not in {EXTERNAL_WORLD_ACTION_SEARCH_V111, EXTERNAL_WORLD_ACTION_FETCH_V111, EXTERNAL_WORLD_ACTION_OBSERVE_V111}:
            return False, "invalid_external_world_action", {"i": int(i), "action": action}

        reason = str(e.get("reason_code") or "")
        if reason not in EXTERNAL_WORLD_REASON_CODES_V111:
            return False, "invalid_reason_code", {"i": int(i), "reason_code": reason}

        want_prev = str(e.get("prev_event_sig") or "")
        if want_prev != prev:
            return False, "external_world_prev_event_sig_mismatch", {"i": int(i), "want_prev": want_prev, "got_prev": prev}

        body = dict(e)
        # Canon body without event_id/event_sig (event_sig binds prev_event_sig and body fields).
        body.pop("event_id", None)
        body.pop("event_sig", None)
        # Remove any outer-chain fields if present.
        body.pop("prev_hash", None)
        body.pop("entry_hash", None)
        sig = _external_world_event_sig_v111(prev_event_sig=str(prev), event_body=body)
        if str(e.get("event_sig") or "") != sig:
            return False, "external_world_event_sig_mismatch", {"i": int(i), "want": str(e.get("event_sig") or ""), "got": sig}

        prev = sig
    return True, "ok", {"events_total": int(len(events))}
