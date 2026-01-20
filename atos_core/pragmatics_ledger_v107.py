from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex


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


def _canon_str_list(xs: Any) -> List[str]:
    if not isinstance(xs, list):
        return []
    out = [str(x) for x in xs if isinstance(x, str) and str(x)]
    return sorted(set(out), key=str)


def _canon_act(act: Any) -> Dict[str, Any]:
    if not isinstance(act, dict):
        return {"kind": "", "slots": {}, "confidence_tag": ""}
    slots = act.get("slots") if isinstance(act.get("slots"), dict) else {}
    slots2: Dict[str, Any] = {}
    for k in sorted(slots.keys(), key=str):
        kk = str(k)
        v = slots.get(k)
        if v is None:
            slots2[kk] = ""
        elif isinstance(v, (str, int, float, bool)):
            slots2[kk] = v
        else:
            slots2[kk] = str(v)
    return {
        "kind": _canon_str(act.get("kind") or ""),
        "slots": dict(slots2),
        "confidence_tag": _canon_str(act.get("confidence_tag") or ""),
    }


def _canon_dialogue_regime(reg: Any) -> Dict[str, str]:
    if not isinstance(reg, dict):
        reg = {}
    return {"prev": _canon_str(reg.get("prev") or ""), "next": _canon_str(reg.get("next") or "")}


def _canon_delta(delta: Any) -> Dict[str, Any]:
    if not isinstance(delta, dict):
        return {"updates": {}, "adds": [], "removes": []}
    updates = delta.get("updates") if isinstance(delta.get("updates"), dict) else {}
    updates2: Dict[str, Any] = {}
    for k in sorted(updates.keys(), key=str):
        updates2[str(k)] = updates.get(k)
    adds = _canon_str_list(delta.get("adds"))
    removes = _canon_str_list(delta.get("removes"))
    return {"updates": dict(updates2), "adds": list(adds), "removes": list(removes)}


def _canon_qset(q: Any) -> Dict[str, List[str]]:
    if not isinstance(q, dict):
        q = {}
    return {"opened": _canon_str_list(q.get("opened")), "closed": _canon_str_list(q.get("closed"))}


def _canon_commitments(c: Any) -> Dict[str, List[str]]:
    if not isinstance(c, dict):
        c = {}
    return {
        "opened": _canon_str_list(c.get("opened")),
        "closed": _canon_str_list(c.get("closed")),
        "violated": _canon_str_list(c.get("violated")),
    }


def _canon_topic_stack(ts: Any) -> Dict[str, Any]:
    if not isinstance(ts, dict):
        ts = {}
    return {
        "push": _canon_str_list(ts.get("push")),
        "pop": _canon_str_list(ts.get("pop")),
        "top_after": _canon_str(ts.get("top_after") or ""),
    }


def _canon_flags(flags: Any) -> Dict[str, bool]:
    if not isinstance(flags, dict):
        flags = {}
    out: Dict[str, bool] = {}
    for k in sorted([str(k) for k in flags.keys() if str(k)], key=str):
        out[str(k)] = _canon_bool(flags.get(k, False))
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
                "candidate_hash": _canon_str(c.get("candidate_hash") or ""),
                "coherence_score_v106": _canon_int(c.get("coherence_score_v106"), default=0),
                "pragmatics_score_v107": _canon_int(c.get("pragmatics_score_v107"), default=0),
                "flags_v106": _canon_flags(c.get("flags_v106")),
                "flags_v107": _canon_flags(c.get("flags_v107")),
                "length_chars": _canon_int(c.get("length_chars"), default=0),
                "repetition_metrics": c.get("repetition_metrics") if isinstance(c.get("repetition_metrics"), dict) else {},
            }
        )
    # Deterministic rank for storage.
    out.sort(key=lambda d: (-min(int(d.get("coherence_score_v106") or 0), int(d.get("pragmatics_score_v107") or 0)), str(d.get("candidate_hash") or "")))
    return list(out)


def _pragmatics_event_sig_v107(*, prev_event_sig: str, event_body: Dict[str, Any]) -> str:
    payload = str(prev_event_sig or "") + canonical_json_dumps(event_body)
    return sha256_hex(payload.encode("utf-8"))


def pragmatics_event_id_v107(event_sig: str) -> str:
    return f"pragmatics_event_v107_{str(event_sig)}"


@dataclass(frozen=True)
class PragmaticsEventV107:
    conversation_id: str
    event_index: int
    user_turn_id: str
    user_turn_index: int
    assistant_turn_id: str
    assistant_turn_index: int
    user_text_hash: str
    user_intent_act: Dict[str, Any]
    assistant_intent_act: Dict[str, Any]
    dialogue_regime: Dict[str, Any]
    dialogue_state_delta: Dict[str, Any]
    pending_questions: Dict[str, Any]
    commitments: Dict[str, Any]
    topic_stack: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    chosen_candidate_hash: str
    pragmatics_score: int
    flags_v107: Dict[str, Any]
    repair_action: Dict[str, Any]
    progress_blocked: bool
    blocked_ledgers: List[str]
    created_step: int
    prev_event_sig: str

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 107,
            "kind": "pragmatics_event_v107",
            "conversation_id": _canon_str(self.conversation_id),
            "event_index": int(self.event_index),
            "user_turn_id": _canon_str(self.user_turn_id),
            "user_turn_index": int(self.user_turn_index),
            "assistant_turn_id": _canon_str(self.assistant_turn_id),
            "assistant_turn_index": int(self.assistant_turn_index),
            "ts_deterministic": deterministic_iso(step=int(self.created_step)),
            "user_text_hash": _canon_str(self.user_text_hash),
            "user_intent_act": _canon_act(self.user_intent_act),
            "assistant_intent_act": _canon_act(self.assistant_intent_act),
            "dialogue_regime": _canon_dialogue_regime(self.dialogue_regime),
            "dialogue_state_delta": _canon_delta(self.dialogue_state_delta),
            "pending_questions": _canon_qset(self.pending_questions),
            "commitments": _canon_commitments(self.commitments),
            "topic_stack": _canon_topic_stack(self.topic_stack),
            "candidates": _canon_candidates(self.candidates),
            "chosen_candidate_hash": _canon_str(self.chosen_candidate_hash),
            "pragmatics_score": max(0, min(100, int(self.pragmatics_score))),
            "flags_v107": _canon_flags(self.flags_v107),
            "repair_action": self.repair_action if isinstance(self.repair_action, dict) else (None if not self.repair_action else {"kind": _canon_str(self.repair_action)}),
            "progress_blocked": bool(self.progress_blocked),
            "blocked_ledgers": _canon_str_list(self.blocked_ledgers),
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = _pragmatics_event_sig_v107(prev_event_sig=str(self.prev_event_sig or ""), event_body=dict(body))
        return dict(body, prev_event_sig=str(self.prev_event_sig or ""), event_sig=str(sig), event_id=str(pragmatics_event_id_v107(sig)))


def verify_pragmatics_event_sig_v107(ev: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(ev, dict):
        return False, "pragmatics_event_not_dict", {}
    prev_sig = str(ev.get("prev_event_sig") or "")
    want_sig = str(ev.get("event_sig") or "")
    body = dict(ev)
    body.pop("prev_event_sig", None)
    body.pop("event_sig", None)
    body.pop("event_id", None)
    got_sig = _pragmatics_event_sig_v107(prev_event_sig=str(prev_sig), event_body=dict(body))
    if got_sig != want_sig:
        return False, "pragmatics_event_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    want_id = pragmatics_event_id_v107(want_sig)
    got_id = str(ev.get("event_id") or "")
    if got_id != want_id:
        return False, "pragmatics_event_id_mismatch", {"want": str(want_id), "got": str(got_id)}
    return True, "ok", {}


def compute_pragmatics_chain_hash_v107(pragmatics_events: Sequence[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in pragmatics_events:
        if not isinstance(ev, dict):
            continue
        ids.append(str(ev.get("event_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


def fold_pragmatics_ledger_v107(pragmatics_events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pending: List[str] = []
    commits: List[str] = []
    topic_stack: List[str] = []
    current_regime = ""
    last_turn_id = ""
    for ev in pragmatics_events:
        if not isinstance(ev, dict):
            continue
        last_turn_id = str(ev.get("user_turn_id") or last_turn_id)
        reg = ev.get("dialogue_regime") if isinstance(ev.get("dialogue_regime"), dict) else {}
        current_regime = str(reg.get("next") or current_regime)

        pq = ev.get("pending_questions") if isinstance(ev.get("pending_questions"), dict) else {}
        opened = [str(x) for x in (pq.get("opened") if isinstance(pq.get("opened"), list) else []) if str(x)]
        closed = [str(x) for x in (pq.get("closed") if isinstance(pq.get("closed"), list) else []) if str(x)]
        for qid in opened:
            if qid not in pending:
                pending.append(str(qid))
        for qid in closed:
            if qid in pending:
                pending.remove(str(qid))

        cm = ev.get("commitments") if isinstance(ev.get("commitments"), dict) else {}
        cop = [str(x) for x in (cm.get("opened") if isinstance(cm.get("opened"), list) else []) if str(x)]
        ccl = [str(x) for x in (cm.get("closed") if isinstance(cm.get("closed"), list) else []) if str(x)]
        for cid in cop:
            if cid not in commits:
                commits.append(str(cid))
        for cid in ccl:
            if cid in commits:
                commits.remove(str(cid))

        ts = ev.get("topic_stack") if isinstance(ev.get("topic_stack"), dict) else {}
        push = [str(x) for x in (ts.get("push") if isinstance(ts.get("push"), list) else []) if str(x)]
        pop = [str(x) for x in (ts.get("pop") if isinstance(ts.get("pop"), list) else []) if str(x)]
        for t in push:
            topic_stack.append(str(t))
        for _t in pop:
            if topic_stack:
                topic_stack.pop()

    snap = {
        "schema_version": 107,
        "kind": "pragmatics_registry_v107",
        "last_turn_id": str(last_turn_id),
        "current_regime": str(current_regime),
        "topic_stack": list(topic_stack),
        "pending_questions": list(pending),
        "commitments": list(commits),
        "entities_mentioned": {},
        "facts_established": [],
        "user_model": {},
    }
    return dict(snap)


def pragmatics_registry_snapshot_v107(pragmatics_events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    snap = fold_pragmatics_ledger_v107(list(pragmatics_events))
    sig = _stable_hash_obj(snap)
    return dict(snap, snapshot_sig=str(sig))


def _events_prefix(*, pragmatics_events: Sequence[Dict[str, Any]], until_user_turn_index_exclusive: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ev in pragmatics_events:
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


def lookup_pragmatics_event_v107(*, pragmatics_events: Sequence[Dict[str, Any]], user_turn_index: int) -> Optional[Dict[str, Any]]:
    for ev in pragmatics_events:
        if not isinstance(ev, dict):
            continue
        try:
            uidx = int(ev.get("user_turn_index") or -1)
        except Exception:
            uidx = -1
        if uidx == int(user_turn_index):
            return dict(ev)
    return None


def render_pragmatics_text_v107(*, pragmatics_events: Sequence[Dict[str, Any]], until_user_turn_index_exclusive: int) -> str:
    prefix = _events_prefix(pragmatics_events=list(pragmatics_events), until_user_turn_index_exclusive=int(until_user_turn_index_exclusive))
    snap = pragmatics_registry_snapshot_v107(prefix)
    lines: List[str] = []
    lines.append(f"PRAGMATICS: regime={str(snap.get('current_regime') or '')} pending={len(list(snap.get('pending_questions') or []))}")
    for ev in prefix[-5:]:
        if not isinstance(ev, dict):
            continue
        lines.append(
            f"- turn={int(ev.get('user_turn_index') or 0)} score={int(ev.get('pragmatics_score') or 0)} repair={str((ev.get('repair_action') or {}) if isinstance(ev.get('repair_action'), dict) else ev.get('repair_action') or '')}"
        )
    return "\n".join(lines)


def render_explain_pragmatics_text_v107(*, pragmatics_events: Sequence[Dict[str, Any]], user_turn_index: int) -> str:
    ev = lookup_pragmatics_event_v107(pragmatics_events=list(pragmatics_events), user_turn_index=int(user_turn_index))
    if not isinstance(ev, dict):
        return "EXPLAIN_PRAGMATICS: not_found"
    uia = ev.get("user_intent_act") if isinstance(ev.get("user_intent_act"), dict) else {}
    aia = ev.get("assistant_intent_act") if isinstance(ev.get("assistant_intent_act"), dict) else {}
    reg = ev.get("dialogue_regime") if isinstance(ev.get("dialogue_regime"), dict) else {}
    ra = ev.get("repair_action") if isinstance(ev.get("repair_action"), dict) else {}
    blocked = bool(ev.get("progress_blocked", False))
    bl = ev.get("blocked_ledgers") if isinstance(ev.get("blocked_ledgers"), list) else []
    return (
        f"EXPLAIN_PRAGMATICS: turn={int(ev.get('user_turn_index') or 0)} score={int(ev.get('pragmatics_score') or 0)}\n"
        f"USER_INTENT={str(uia.get('kind') or '')} ASSISTANT_INTENT={str(aia.get('kind') or '')}\n"
        f"REGIME={str(reg.get('prev') or '')}->{str(reg.get('next') or '')} REPAIR={str(ra.get('kind') or '')}\n"
        f"BLOCKED={str(blocked).lower()} LEDGERS={','.join([str(x) for x in bl if str(x)])}"
    )


def render_trace_pragmatics_text_v107(*, pragmatics_events: Sequence[Dict[str, Any]], until_user_turn_index: int) -> str:
    prefix = _events_prefix(pragmatics_events=list(pragmatics_events), until_user_turn_index_exclusive=int(until_user_turn_index) + 1)
    h = compute_pragmatics_chain_hash_v107(prefix)
    last_id = ""
    if prefix:
        last_id = str(prefix[-1].get("event_id") or "")
    return f"TRACE_PRAGMATICS: until={int(until_user_turn_index)} last_event_id={last_id} chain_hash={h}"

