from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex

DIALOGUE_REPAIR_NONE_V106 = ""
DIALOGUE_REPAIR_ASK_CLARIFY_V106 = "ask_clarify"
DIALOGUE_REPAIR_SUMMARIZE_CONFIRM_V106 = "summarize_confirm"
DIALOGUE_REPAIR_RESTATE_OFFER_OPTIONS_V106 = "restate_offer_options"


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _canon_int(x: Any, *, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _canon_str(x: Any) -> str:
    return str(x) if isinstance(x, str) else str(x or "")


def _canon_bool(x: Any) -> bool:
    return bool(x)


def _canon_flags(flags: Any) -> Dict[str, bool]:
    if not isinstance(flags, dict):
        flags = {}
    out: Dict[str, bool] = {}
    # Stable ordering of known flags; include unknowns deterministically too.
    known = [
        "too_dry",
        "too_long",
        "too_short",
        "abrupt_shift",
        "repetition",
        "asked_when_should_answer",
        "answered_when_should_ask",
    ]
    for k in known:
        out[str(k)] = _canon_bool(flags.get(k, False))
    for k in sorted([str(k) for k in flags.keys() if str(k) and str(k) not in set(known)], key=str):
        out[str(k)] = _canon_bool(flags.get(k, False))
    return dict(out)


def _canon_components(components: Any) -> Dict[str, int]:
    if not isinstance(components, dict):
        components = {}
    out = {
        "entity_consistency": _canon_int(components.get("entity_consistency"), default=0),
        "topic_continuity": _canon_int(components.get("topic_continuity"), default=0),
        "reference_resolution": _canon_int(components.get("reference_resolution"), default=0),
        "logical_consistency": _canon_int(components.get("logical_consistency"), default=0),
    }
    # clamp 0..100 deterministically
    for k in list(out.keys()):
        v = int(out.get(k) or 0)
        out[k] = max(0, min(100, int(v)))
    return dict(out)


def _canon_candidates(cands: Any) -> List[Dict[str, Any]]:
    if not isinstance(cands, list):
        return []
    out: List[Dict[str, Any]] = []
    for c in cands:
        if not isinstance(c, dict):
            continue
        out.append(
            {
                "cand_id": _canon_str(c.get("cand_id") or c.get("candidate_id") or ""),
                "text_sha256": _canon_str(c.get("text_sha256") or ""),
                "len_chars": _canon_int(c.get("len_chars"), default=0),
                "len_tokens": _canon_int(c.get("len_tokens"), default=0),
                "score": _canon_int(c.get("score"), default=0),
                "flags_digest": _canon_str(c.get("flags_digest") or ""),
            }
        )
    # Deterministic ranking for storage (best first).
    out.sort(key=lambda d: (-int(d.get("score") or 0), int(d.get("len_tokens") or 0), str(d.get("cand_id") or "")))
    return list(out)


def _dialogue_event_sig_v106(*, prev_event_sig: str, event_body: Dict[str, Any]) -> str:
    payload = str(prev_event_sig or "") + canonical_json_dumps(event_body)
    return sha256_hex(payload.encode("utf-8"))


def dialogue_event_id_v106(event_sig: str) -> str:
    return f"dialogue_event_v106_{str(event_sig)}"


@dataclass(frozen=True)
class DialogueEventV106:
    conversation_id: str
    user_turn_id: str
    user_turn_index: int
    assistant_turn_id: str
    assistant_turn_index: int
    coherence_score: int
    components: Dict[str, Any]
    flags: Dict[str, Any]
    candidates_topk: List[Dict[str, Any]]
    chosen_cand_id: str
    repair_action: str
    created_step: int
    prev_event_sig: str

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 106,
            "kind": "dialogue_event_v106",
            "conversation_id": _canon_str(self.conversation_id),
            "user_turn_id": _canon_str(self.user_turn_id),
            "user_turn_index": int(self.user_turn_index),
            "assistant_turn_id": _canon_str(self.assistant_turn_id),
            "assistant_turn_index": int(self.assistant_turn_index),
            "coherence_score": max(0, min(100, int(self.coherence_score))),
            "components": _canon_components(self.components),
            "flags": _canon_flags(self.flags),
            "candidates_topk": _canon_candidates(self.candidates_topk),
            "chosen_cand_id": _canon_str(self.chosen_cand_id),
            "repair_action": _canon_str(self.repair_action),
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = _dialogue_event_sig_v106(prev_event_sig=str(self.prev_event_sig or ""), event_body=dict(body))
        return dict(body, prev_event_sig=str(self.prev_event_sig or ""), event_sig=str(sig), event_id=str(dialogue_event_id_v106(sig)))


def verify_dialogue_event_sig_v106(ev: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(ev, dict):
        return False, "dialogue_event_not_dict", {}
    prev_sig = str(ev.get("prev_event_sig") or "")
    want_sig = str(ev.get("event_sig") or "")
    body = dict(ev)
    body.pop("prev_event_sig", None)
    body.pop("event_sig", None)
    body.pop("event_id", None)
    got_sig = _dialogue_event_sig_v106(prev_event_sig=str(prev_sig), event_body=dict(body))
    if got_sig != want_sig:
        return False, "dialogue_event_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    want_id = dialogue_event_id_v106(want_sig)
    got_id = str(ev.get("event_id") or "")
    if got_id != want_id:
        return False, "dialogue_event_id_mismatch", {"want": str(want_id), "got": str(got_id)}
    return True, "ok", {}


def compute_dialogue_chain_hash_v106(dialogue_events: Sequence[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in dialogue_events:
        if not isinstance(ev, dict):
            continue
        ids.append(str(ev.get("event_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


def fold_dialogue_ledger_v106(dialogue_events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Minimal derived state: last score/flags + last repair, and a per-turn index.
    This is NOT authoritative; it's a deterministic convenience snapshot.
    """
    by_user_turn_index: Dict[str, Dict[str, Any]] = {}
    last = {}
    for ev in dialogue_events:
        if not isinstance(ev, dict):
            continue
        try:
            uidx = int(ev.get("user_turn_index") or 0)
        except Exception:
            uidx = 0
        row = {
            "user_turn_index": int(uidx),
            "assistant_turn_index": _canon_int(ev.get("assistant_turn_index"), default=-1),
            "event_id": _canon_str(ev.get("event_id") or ""),
            "coherence_score": _canon_int(ev.get("coherence_score"), default=0),
            "repair_action": _canon_str(ev.get("repair_action") or ""),
            "chosen_cand_id": _canon_str(ev.get("chosen_cand_id") or ""),
        }
        by_user_turn_index[str(uidx)] = dict(row)
        last = dict(row)
    return {
        "schema_version": 106,
        "kind": "dialogue_registry_v106",
        "by_user_turn_index": {str(k): dict(by_user_turn_index[k]) for k in sorted(by_user_turn_index.keys(), key=str)},
        "last": dict(last),
    }


def dialogue_registry_snapshot_v106(dialogue_events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    snap = fold_dialogue_ledger_v106(list(dialogue_events))
    sig = _stable_hash_obj(snap)
    return dict(snap, snapshot_sig=str(sig))


def _dialogue_events_prefix(*, dialogue_events: Sequence[Dict[str, Any]], until_user_turn_index_exclusive: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ev in dialogue_events:
        if not isinstance(ev, dict):
            continue
        try:
            uidx = int(ev.get("user_turn_index") or 0)
        except Exception:
            uidx = 0
        if uidx >= int(until_user_turn_index_exclusive):
            break
        out.append(dict(ev))
    return list(out)


def lookup_dialogue_event_v106(*, dialogue_events: Sequence[Dict[str, Any]], user_turn_index: int) -> Optional[Dict[str, Any]]:
    for ev in dialogue_events:
        if not isinstance(ev, dict):
            continue
        try:
            uidx = int(ev.get("user_turn_index") or -1)
        except Exception:
            uidx = -1
        if uidx == int(user_turn_index):
            return dict(ev)
    return None


def render_dialogue_text_v106(*, dialogue_events: Sequence[Dict[str, Any]], until_user_turn_index_exclusive: int) -> str:
    prefix = _dialogue_events_prefix(dialogue_events=list(dialogue_events), until_user_turn_index_exclusive=int(until_user_turn_index_exclusive))
    snap = dialogue_registry_snapshot_v106(list(prefix))
    by_idx = snap.get("by_user_turn_index") if isinstance(snap.get("by_user_turn_index"), dict) else {}
    if not by_idx:
        return "DIALOGUE: (empty)"
    lines: List[str] = ["DIALOGUE:"]
    for k in sorted(by_idx.keys(), key=str):
        row = by_idx.get(k) if isinstance(by_idx.get(k), dict) else {}
        lines.append(
            f"- user_turn_index={k} coherence={int(row.get('coherence_score') or 0)} repair={str(row.get('repair_action') or '')} chosen={str(row.get('chosen_cand_id') or '')}"
        )
    return "\n".join(lines)


def render_explain_dialogue_text_v106(*, dialogue_events: Sequence[Dict[str, Any]], user_turn_index: int) -> str:
    ev = lookup_dialogue_event_v106(dialogue_events=list(dialogue_events), user_turn_index=int(user_turn_index))
    if not isinstance(ev, dict):
        return "EXPLAIN_DIALOGUE: not_found"
    parts: List[str] = []
    for c in (ev.get("candidates_topk") if isinstance(ev.get("candidates_topk"), list) else [])[:3]:
        if not isinstance(c, dict):
            continue
        parts.append(f"{str(c.get('cand_id') or '')}:{int(c.get('score') or 0)}")
    chosen = str(ev.get("chosen_cand_id") or "")
    repair = str(ev.get("repair_action") or "")
    comps = ev.get("components") if isinstance(ev.get("components"), dict) else {}
    return "\n".join(
        [
            f"EXPLAIN_DIALOGUE: user_turn_index={int(user_turn_index)} coherence={int(ev.get('coherence_score') or 0)} chosen={chosen} repair={repair}",
            f"COMPONENTS: entity={int(comps.get('entity_consistency') or 0)} topic={int(comps.get('topic_continuity') or 0)} ref={int(comps.get('reference_resolution') or 0)} logic={int(comps.get('logical_consistency') or 0)}",
            "CANDIDATES: " + "; ".join(parts) if parts else "CANDIDATES: (none)",
        ]
    )


def render_trace_dialogue_text_v106(*, dialogue_events: Sequence[Dict[str, Any]], until_user_turn_index: int) -> str:
    ids: List[str] = []
    for ev in dialogue_events:
        if not isinstance(ev, dict):
            continue
        try:
            uidx = int(ev.get("user_turn_index") or 0)
        except Exception:
            uidx = 0
        if uidx > int(until_user_turn_index):
            break
        ids.append(str(ev.get("event_id") or ""))
    return "TRACE_DIALOGUE: " + canonical_json_dumps(ids)

