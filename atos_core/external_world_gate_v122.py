from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .external_world_v122 import ExternalWorldFetchV122, ExternalWorldV122, ew_load_and_verify


EXTERNAL_WORLD_ACTION_SEARCH_V122 = "SEARCH_EXTERNAL_WORLD_V122"
EXTERNAL_WORLD_ACTION_FETCH_V122 = "FETCH_EXTERNAL_WORLD_V122"
EXTERNAL_WORLD_ACTION_OBSERVE_V122 = "OBSERVE_EXTERNAL_WORLD_V122"

EXTERNAL_WORLD_ACTIONS_V122 = {
    EXTERNAL_WORLD_ACTION_SEARCH_V122,
    EXTERNAL_WORLD_ACTION_FETCH_V122,
    EXTERNAL_WORLD_ACTION_OBSERVE_V122,
}

# Closed enum of reason codes (fail-closed).
EXTERNAL_WORLD_REASON_CODES_V122 = {
    "validator_failed_repair_impossible",
    "progress_blocked",
    "audit_mode_user_request",
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


def _external_world_event_sig_v122(*, prev_event_sig: str, event_body: Dict[str, Any]) -> str:
    payload = str(prev_event_sig or "") + canonical_json_dumps(event_body)
    return sha256_hex(payload.encode("utf-8"))


def external_world_event_id_v122(event_sig: str) -> str:
    return f"external_world_event_v122_{str(event_sig)}"


def _external_world_evidence_sig_v122(*, evidence_body: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(evidence_body).encode("utf-8"))


def external_world_evidence_id_v122(evidence_sig: str) -> str:
    return f"external_world_evidence_v122_{str(evidence_sig)}"


@dataclass(frozen=True)
class ExternalWorldEventV122:
    event_index: int
    turn_index: int
    action: str
    reason_code: str
    args: Dict[str, Any]
    result_summary: Dict[str, Any]
    evidence_ids: List[str]
    prev_event_sig: str
    event_sig: str


def make_external_world_event_v122(
    *,
    event_index: int,
    turn_index: int,
    action: str,
    reason_code: str,
    args: Dict[str, Any],
    result_summary: Dict[str, Any],
    evidence_ids: Sequence[str],
    prev_event_sig: str,
) -> ExternalWorldEventV122:
    body = {
        "event_index": _canon_int(event_index, default=0),
        "turn_index": _canon_int(turn_index, default=0),
        "action": _canon_str(action),
        "reason_code": _canon_str(reason_code),
        "args": _canon_json_obj(args),
        "result_summary": _canon_json_obj(result_summary),
        "evidence_ids": sorted([str(x) for x in (evidence_ids or []) if isinstance(x, str) and x]),
        "prev_event_sig": _canon_str(prev_event_sig),
    }
    sig = _external_world_event_sig_v122(prev_event_sig=str(prev_event_sig or ""), event_body=dict(body))
    return ExternalWorldEventV122(
        event_index=int(body["event_index"]),
        turn_index=int(body["turn_index"]),
        action=str(body["action"]),
        reason_code=str(body["reason_code"]),
        args=dict(body["args"]) if isinstance(body["args"], dict) else {},
        result_summary=dict(body["result_summary"]) if isinstance(body["result_summary"], dict) else {},
        evidence_ids=list(body["evidence_ids"]) if isinstance(body["evidence_ids"], list) else [],
        prev_event_sig=str(body["prev_event_sig"]),
        event_sig=str(sig),
    )


def external_world_event_to_dict_v122(ev: ExternalWorldEventV122) -> Dict[str, Any]:
    return {
        "event_id": external_world_event_id_v122(str(ev.event_sig)),
        "event_index": int(ev.event_index),
        "turn_index": int(ev.turn_index),
        "action": str(ev.action),
        "reason_code": str(ev.reason_code),
        "args": _canon_json_obj(ev.args),
        "result_summary": _canon_json_obj(ev.result_summary),
        "evidence_ids": list(ev.evidence_ids),
        "prev_event_sig": str(ev.prev_event_sig),
        "event_sig": str(ev.event_sig),
    }


def make_external_world_evidence_v122(*, kind: str, body: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"schema_version": 122, "kind": str(kind), "body": _canon_json_obj(body)}
    sig = _external_world_evidence_sig_v122(evidence_body=dict(payload))
    return {"evidence_id": external_world_evidence_id_v122(sig), "evidence_sig": str(sig), **dict(payload)}


def compute_external_world_chain_hash_v122(events: Sequence[Dict[str, Any]]) -> str:
    sigs: List[str] = []
    for e in events:
        if isinstance(e, dict):
            sigs.append(str(e.get("event_sig") or ""))
    return _stable_hash_obj(sigs)


def verify_external_world_event_sig_chain_v122(events: Sequence[Dict[str, Any]]) -> Tuple[bool, str, Dict[str, Any]]:
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
        if action not in EXTERNAL_WORLD_ACTIONS_V122:
            return False, "invalid_external_world_action", {"i": int(i), "action": action}

        reason = str(e.get("reason_code") or "")
        if reason not in EXTERNAL_WORLD_REASON_CODES_V122:
            return False, "invalid_reason_code", {"i": int(i), "reason_code": reason}

        want_prev = str(e.get("prev_event_sig") or "")
        if want_prev != prev:
            return False, "external_world_prev_event_sig_mismatch", {"i": int(i), "want_prev": want_prev, "got_prev": prev}

        body = dict(e)
        body.pop("event_id", None)
        body.pop("event_sig", None)
        body.pop("prev_hash", None)
        body.pop("entry_hash", None)
        sig = _external_world_event_sig_v122(prev_event_sig=str(prev), event_body=body)
        if str(e.get("event_sig") or "") != sig:
            return False, "external_world_event_sig_mismatch", {"i": int(i), "want": str(e.get("event_sig") or ""), "got": sig}
        prev = sig
    return True, "ok", {"events_total": int(len(events))}


def external_world_access_v122(
    *,
    allowed: bool,
    manifest_path: str,
    action: str,
    reason_code: str,
    args: Dict[str, Any],
    seed: int,
    turn_index: int,
    prev_event_sig: str,
    max_chars: int = 2000,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deterministic gating wrapper for the V122 unified ExternalWorld.

    Fail-closed:
      - if not allowed -> external_world_access_not_allowed
      - if reason_code invalid -> invalid_reason_code
      - if manifest invalid -> external_world_manifest_mismatch_v122
    """
    if not bool(allowed):
        raise ValueError("external_world_access_not_allowed")
    if str(reason_code) not in EXTERNAL_WORLD_REASON_CODES_V122:
        raise ValueError("invalid_reason_code")

    try:
        world = ew_load_and_verify(manifest_path=str(manifest_path))
    except Exception:
        raise ValueError("external_world_manifest_mismatch_v122")

    result_summary: Dict[str, Any] = {"seed": int(seed)}
    evidences: List[Dict[str, Any]] = []
    evidence_ids: List[str] = []

    if str(action) == EXTERNAL_WORLD_ACTION_SEARCH_V122:
        q = str((args or {}).get("query") or "")
        limit = int((args or {}).get("limit") or 3)
        source_filter = str((args or {}).get("source_filter") or "all")
        roles = (args or {}).get("roles") if isinstance((args or {}).get("roles"), list) else []
        hits = world.search(query=str(q), limit=int(limit), source_filter=str(source_filter), roles=[str(r) for r in roles if isinstance(r, str)])
        hit_view = [{"hit_id": h.hit_id, "source": h.source} for h in hits]
        result_summary.update({"query": str(q), "source_filter": str(source_filter), "hits_total": int(len(hits)), "hits": list(hit_view)})
        ev = make_external_world_evidence_v122(kind="external_world_search_v122", body={"query": str(q), "hits": list(hit_view)})
        evidences.append(dict(ev))
        evidence_ids.append(str(ev.get("evidence_id") or ""))
    elif str(action) == EXTERNAL_WORLD_ACTION_FETCH_V122:
        hit_id = str((args or {}).get("hit_id") or "")
        fetch: ExternalWorldFetchV122 = world.fetch(hit_id=str(hit_id), max_chars=int(max_chars))
        result_summary.update(
            {
                "hit_id": str(fetch.hit_id),
                "source": str(fetch.source),
                "doc_id": str(fetch.doc_id),
                "text_sha256": str(fetch.text_sha256),
                "truncated": bool(fetch.truncated),
                "offsets": dict(fetch.offsets),
            }
        )
        ev = make_external_world_evidence_v122(
            kind="external_world_fetch_v122",
            body={
                "hit_id": str(fetch.hit_id),
                "source": str(fetch.source),
                "doc_id": str(fetch.doc_id),
                "text_sha256": str(fetch.text_sha256),
                "offsets": dict(fetch.offsets),
                "max_chars": int(max_chars),
                "truncated": bool(fetch.truncated),
            },
        )
        evidences.append(dict(ev))
        evidence_ids.append(str(ev.get("evidence_id") or ""))
    elif str(action) == EXTERNAL_WORLD_ACTION_OBSERVE_V122:
        # Minimal observe-range support for dialogue_history (bounded).
        start_turn = int((args or {}).get("start_turn") or 0)
        end_turn = int((args or {}).get("end_turn") or 0)
        roles = (args or {}).get("roles") if isinstance((args or {}).get("roles"), list) else ["user"]
        limit = int((args or {}).get("limit") or 10)
        # Deterministic implementation via search+fetch is too expensive; do bounded fetch loop.
        turns: List[Dict[str, Any]] = []
        for idx in range(int(start_turn), int(end_turn) + 1):
            if len(turns) >= int(limit):
                break
            try:
                ft = world.fetch(hit_id="dlg:{i}".format(i=int(idx)), max_chars=int(max_chars))
            except Exception:
                continue
            if roles and str(ft.source) != "dialogue_history":
                continue
            turns.append({"hit_id": str(ft.hit_id), "text_sha256": str(ft.text_sha256), "offsets": dict(ft.offsets)})
        result_summary.update({"observed_total": int(len(turns)), "observed_hash": _stable_hash_obj(turns)})
        ev = make_external_world_evidence_v122(kind="external_world_observe_v122", body={"turns": list(turns)})
        evidences.append(dict(ev))
        evidence_ids.append(str(ev.get("evidence_id") or ""))
    else:
        raise ValueError("invalid_external_world_action")

    ev0 = make_external_world_event_v122(
        event_index=0,
        turn_index=int(turn_index),
        action=str(action),
        reason_code=str(reason_code),
        args=dict(args),
        result_summary=dict(result_summary),
        evidence_ids=list(evidence_ids),
        prev_event_sig=str(prev_event_sig or ""),
    )
    return [external_world_event_to_dict_v122(ev0)], list(evidences), dict(result_summary)

