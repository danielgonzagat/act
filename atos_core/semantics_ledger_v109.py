from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex


SEM_REPAIR_NONE_V109 = ""
SEM_REPAIR_CLARIFY_V109 = "clarify"
SEM_REPAIR_ASK_EXAMPLE_V109 = "ask_example"
SEM_REPAIR_CONFIRM_MEANING_V109 = "confirm_meaning"
SEM_REPAIR_UPDATE_CONCEPT_V109 = "update_concept"
SEM_REPAIR_REFUSE_SEMANTICALLY_V109 = "refuse_semantically"


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _canon_int(x: Any, *, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _canon_bool(x: Any) -> bool:
    return bool(x)


def _canon_str(x: Any) -> str:
    return str(x) if isinstance(x, str) else str(x or "")


def _canon_str_list(xs: Any) -> List[str]:
    if not isinstance(xs, list):
        return []
    out = [str(x) for x in xs if isinstance(x, str) and str(x)]
    return list(out)


def _canon_flags(flags: Any) -> Dict[str, bool]:
    if not isinstance(flags, dict):
        flags = {}
    out: Dict[str, bool] = {}
    for k in sorted([str(k) for k in flags.keys() if str(k)], key=str):
        out[str(k)] = _canon_bool(flags.get(k, False))
    return dict(out)


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


def _canon_concepts(concepts: Any) -> List[Dict[str, Any]]:
    if not isinstance(concepts, list):
        return []
    out: List[Dict[str, Any]] = []
    for c in concepts:
        if not isinstance(c, dict):
            continue
        out.append({"concept_id": _canon_str(c.get("concept_id") or ""), "name": _canon_str(c.get("name") or ""), "status": _canon_str(c.get("status") or "")})
    out.sort(key=lambda d: (str(d.get("concept_id") or ""), str(d.get("name") or "")))
    return list(out)


def _canon_candidates(cands: Any) -> List[Dict[str, Any]]:
    if not isinstance(cands, list):
        return []
    out: List[Dict[str, Any]] = []
    for c in cands:
        if not isinstance(c, dict):
            continue
        out.append(
            {
                "candidate_hash": _canon_str(c.get("candidate_hash") or ""),
                "semantic_score_v109": _canon_int(c.get("semantic_score_v109"), default=0),
                "semantic_flags_v109": _canon_flags(c.get("semantic_flags_v109")),
                "chosen": _canon_bool(c.get("chosen", False)),
            }
        )
    out.sort(key=lambda d: (-int(d.get("semantic_score_v109") or 0), str(d.get("candidate_hash") or "")))
    return list(out)


def _semantic_event_sig_v109(*, prev_event_sig: str, event_body: Dict[str, Any]) -> str:
    payload = str(prev_event_sig or "") + canonical_json_dumps(event_body)
    return sha256_hex(payload.encode("utf-8"))


def semantic_event_id_v109(event_sig: str) -> str:
    return f"semantic_event_v109_{str(event_sig)}"


@dataclass(frozen=True)
class SemanticEventV109:
    conversation_id: str
    user_turn_id: str
    user_turn_index: int
    assistant_turn_id: str
    assistant_turn_index: int
    input_hash: str
    output_hash: str
    active_concepts: List[Dict[str, Any]]
    triggered_concepts: List[str]
    decision: Dict[str, Any]
    semantic_score_v109: int
    flags_v109: Dict[str, Any]
    repair_action_v109: str
    progress_allowed_v109: bool
    created_step: int
    prev_event_sig: str

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 109,
            "kind": "semantic_event_v109",
            "conversation_id": _canon_str(self.conversation_id),
            "turn_index": int(self.user_turn_index),
            "speaker": "assistant",
            "user_turn_id": _canon_str(self.user_turn_id),
            "user_turn_index": int(self.user_turn_index),
            "assistant_turn_id": _canon_str(self.assistant_turn_id),
            "assistant_turn_index": int(self.assistant_turn_index),
            "ts_deterministic": deterministic_iso(step=int(self.created_step)),
            "input_hash": _canon_str(self.input_hash),
            "output_hash": _canon_str(self.output_hash),
            "active_concepts": _canon_concepts(self.active_concepts),
            "triggered_concepts": sorted(set(_canon_str_list(self.triggered_concepts)), key=str),
            "decision": _canon_json_obj(self.decision),
            "semantic_score_v109": max(0, min(100, int(self.semantic_score_v109))),
            "flags_v109": _canon_flags(self.flags_v109),
            "repair_action_v109": _canon_str(self.repair_action_v109),
            "progress_allowed_v109": bool(self.progress_allowed_v109),
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = _semantic_event_sig_v109(prev_event_sig=str(self.prev_event_sig or ""), event_body=dict(body))
        return dict(body, prev_event_sig=str(self.prev_event_sig or ""), event_sig=str(sig), event_id=str(semantic_event_id_v109(sig)))


def verify_semantic_event_sig_v109(ev: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(ev, dict):
        return False, "semantic_event_not_dict", {}
    prev_sig = str(ev.get("prev_event_sig") or "")
    want_sig = str(ev.get("event_sig") or "")
    body = dict(ev)
    body.pop("prev_event_sig", None)
    body.pop("event_sig", None)
    body.pop("event_id", None)
    got_sig = _semantic_event_sig_v109(prev_event_sig=str(prev_sig), event_body=dict(body))
    if got_sig != want_sig:
        return False, "semantic_event_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    want_id = semantic_event_id_v109(want_sig)
    got_id = str(ev.get("event_id") or "")
    if got_id != want_id:
        return False, "semantic_event_id_mismatch", {"want": str(want_id), "got": str(got_id)}
    return True, "ok", {}


def compute_semantic_chain_hash_v109(semantic_events: Sequence[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in semantic_events:
        if not isinstance(ev, dict):
            continue
        ids.append(str(ev.get("event_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


def fold_semantic_ledger_v109(semantic_events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    last: Dict[str, Any] = {}
    by_user_turn_index: Dict[str, Dict[str, Any]] = {}
    promise_action_active: str = ""
    for ev in semantic_events:
        if not isinstance(ev, dict):
            continue
        uidx = _canon_int(ev.get("user_turn_index"), default=0)
        decision = ev.get("decision") if isinstance(ev.get("decision"), dict) else {}
        if isinstance(decision, dict):
            if isinstance(decision.get("promise_action_open"), str) and str(decision.get("promise_action_open") or ""):
                promise_action_active = str(decision.get("promise_action_open") or "")
            if bool(decision.get("promise_action_clear", False)):
                promise_action_active = ""
        row = {
            "user_turn_index": int(uidx),
            "assistant_turn_index": _canon_int(ev.get("assistant_turn_index"), default=-1),
            "event_id": _canon_str(ev.get("event_id") or ""),
            "semantic_score_v109": _canon_int(ev.get("semantic_score_v109"), default=0),
            "progress_allowed_v109": _canon_bool(ev.get("progress_allowed_v109", False)),
            "repair_action_v109": _canon_str(ev.get("repair_action_v109") or ""),
        }
        by_user_turn_index[str(uidx)] = dict(row)
        last = dict(row)
    return {
        "schema_version": 109,
        "kind": "semantic_registry_v109",
        "by_user_turn_index": {str(k): dict(by_user_turn_index[k]) for k in sorted(by_user_turn_index.keys(), key=str)},
        "last": dict(last),
        "promise_action_active": str(promise_action_active),
    }


def semantic_registry_snapshot_v109(semantic_events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    snap = fold_semantic_ledger_v109(list(semantic_events))
    sig = _stable_hash_obj(snap)
    return dict(snap, snapshot_sig=str(sig))


def lookup_semantic_event_v109(*, semantic_events: Sequence[Dict[str, Any]], user_turn_index: int) -> Optional[Dict[str, Any]]:
    for ev in semantic_events:
        if not isinstance(ev, dict):
            continue
        if _canon_int(ev.get("user_turn_index"), default=-1) == int(user_turn_index):
            return dict(ev)
    return None


def _events_prefix(semantic_events: Sequence[Dict[str, Any]], *, until_user_turn_index_exclusive: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ev in semantic_events:
        if not isinstance(ev, dict):
            continue
        uidx = _canon_int(ev.get("user_turn_index"), default=0)
        if uidx >= int(until_user_turn_index_exclusive):
            break
        out.append(dict(ev))
    return list(out)


def render_semantics_text_v109(*, semantic_events: Sequence[Dict[str, Any]], until_user_turn_index_exclusive: int) -> str:
    prefix = _events_prefix(list(semantic_events), until_user_turn_index_exclusive=int(until_user_turn_index_exclusive))
    snap = semantic_registry_snapshot_v109(prefix)
    last = snap.get("last") if isinstance(snap.get("last"), dict) else {}
    if not prefix:
        return "SEMANTICS: (empty)"
    uidx = int(last.get("user_turn_index") or 0)
    score = int(last.get("semantic_score_v109") or 0)
    repair = str(last.get("repair_action_v109") or "")
    allowed = bool(last.get("progress_allowed_v109", False))
    return "SEMANTICS: last_turn={u} score={s} allowed={a} repair={r}".format(u=int(uidx), s=int(score), a="true" if allowed else "false", r=str(repair))


def render_explain_semantics_text_v109(*, semantic_events: Sequence[Dict[str, Any]], user_turn_index: int) -> str:
    ev = lookup_semantic_event_v109(semantic_events=list(semantic_events), user_turn_index=int(user_turn_index))
    if ev is None:
        return "EXPLAIN_SEMANTICS: (no_log)"
    score = int(ev.get("semantic_score_v109") or 0)
    allowed = bool(ev.get("progress_allowed_v109", False))
    repair = str(ev.get("repair_action_v109") or "")
    trig = ev.get("triggered_concepts") if isinstance(ev.get("triggered_concepts"), list) else []
    trig2 = [str(x) for x in trig if isinstance(x, str) and x]
    flags = ev.get("flags_v109") if isinstance(ev.get("flags_v109"), dict) else {}
    flags_on = [str(k) for k in sorted(flags.keys(), key=str) if bool(flags.get(k, False)) and str(k)]
    decision = ev.get("decision") if isinstance(ev.get("decision"), dict) else {}
    kind = str(decision.get("kind") or "")
    lines = [
        "EXPLAIN_SEMANTICS: turn={u} score={s} allowed={a} repair={r}".format(
            u=int(user_turn_index), s=int(score), a="true" if allowed else "false", r=str(repair)
        ),
        "DECISION: kind={k}".format(k=str(kind)),
        "TRIGGERED: " + (", ".join(trig2) if trig2 else "(none)"),
        "FLAGS: " + (", ".join(flags_on) if flags_on else "(none)"),
    ]
    return "\n".join(lines)


def render_trace_semantics_text_v109(*, semantic_events: Sequence[Dict[str, Any]], until_user_turn_index: int) -> str:
    ev = lookup_semantic_event_v109(semantic_events=list(semantic_events), user_turn_index=int(until_user_turn_index))
    if ev is None:
        return "TRACE_SEMANTICS: (no_log)"
    return canonical_json_dumps(dict(ev))

