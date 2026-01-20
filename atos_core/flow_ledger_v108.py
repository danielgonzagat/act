from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex


FLOW_REPAIR_NONE_V108 = ""
FLOW_REPAIR_CLARIFY_REFERENCE_V108 = "clarify_reference"
FLOW_REPAIR_CLARIFY_PREFERENCE_V108 = "clarify_preference"
FLOW_REPAIR_REPHRASE_V108 = "rephrase"
FLOW_REPAIR_SUMMARIZE_CONFIRM_V108 = "summarize_confirm"
FLOW_REPAIR_REFUSE_V108 = "refuse"


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
                "flow_score_v108": _canon_int(c.get("flow_score_v108"), default=0),
                "flow_flags_v108": _canon_flags(c.get("flow_flags_v108")),
                "repetition_score": _canon_int(c.get("repetition_score"), default=0),
                "chosen": _canon_bool(c.get("chosen", False)),
            }
        )
    out.sort(
        key=lambda d: (
            -min(int(d.get("coherence_score_v106") or 0), int(d.get("pragmatics_score_v107") or 0), int(d.get("flow_score_v108") or 0)),
            str(d.get("candidate_hash") or ""),
        )
    )
    return list(out)


def _flow_event_sig_v108(*, prev_event_sig: str, event_body: Dict[str, Any]) -> str:
    payload = str(prev_event_sig or "") + canonical_json_dumps(event_body)
    return sha256_hex(payload.encode("utf-8"))


def flow_event_id_v108(event_sig: str) -> str:
    return f"flow_event_v108_{str(event_sig)}"


@dataclass(frozen=True)
class FlowEventV108:
    conversation_id: str
    user_turn_id: str
    user_turn_index: int
    assistant_turn_id: str
    assistant_turn_index: int
    user_text_hash: str
    assistant_text_hash: str
    coherence_score_v106: int
    coherence_flags_v106: Dict[str, Any]
    pragmatics_score_v107: int
    pragmatics_flags_v107: Dict[str, Any]
    intent_act_v107: Dict[str, Any]
    discourse_plan_v108: List[str]
    verbosity_mode_v108: str
    tone_mode_v108: str
    flow_memory_v108: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    selection_method_v108: str
    chosen_candidate_hash: str
    flow_score_v108: int
    flow_flags_v108: Dict[str, Any]
    progress_allowed_v108: bool
    repair_action_v108: str
    survival_rule_hits_v108: List[str]
    created_step: int
    prev_event_sig: str

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 108,
            "kind": "flow_event_v108",
            "conversation_id": _canon_str(self.conversation_id),
            "turn_index": int(self.user_turn_index),
            "speaker": "assistant",
            "user_turn_id": _canon_str(self.user_turn_id),
            "user_turn_index": int(self.user_turn_index),
            "assistant_turn_id": _canon_str(self.assistant_turn_id),
            "assistant_turn_index": int(self.assistant_turn_index),
            "ts_deterministic": deterministic_iso(step=int(self.created_step)),
            "user_text_hash": _canon_str(self.user_text_hash),
            "assistant_text_hash": _canon_str(self.assistant_text_hash),
            "coherence_score_v106": max(0, min(100, int(self.coherence_score_v106))),
            "coherence_flags_v106": _canon_flags(self.coherence_flags_v106),
            "pragmatics_score_v107": max(0, min(100, int(self.pragmatics_score_v107))),
            "pragmatics_flags_v107": _canon_flags(self.pragmatics_flags_v107),
            "intent_act_v107": _canon_json_obj(self.intent_act_v107),
            "discourse_plan_v108": _canon_str_list(self.discourse_plan_v108),
            "verbosity_mode_v108": _canon_str(self.verbosity_mode_v108),
            "tone_mode_v108": _canon_str(self.tone_mode_v108),
            "flow_memory_v108": _canon_json_obj(self.flow_memory_v108),
            "candidates": _canon_candidates(self.candidates),
            "selection_method_v108": _canon_str(self.selection_method_v108),
            "chosen_candidate_hash": _canon_str(self.chosen_candidate_hash),
            "flow_score_v108": max(0, min(100, int(self.flow_score_v108))),
            "flow_flags_v108": _canon_flags(self.flow_flags_v108),
            "progress_allowed_v108": bool(self.progress_allowed_v108),
            "repair_action_v108": _canon_str(self.repair_action_v108),
            "survival_rule_hits_v108": sorted(set(_canon_str_list(self.survival_rule_hits_v108)), key=str),
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = _flow_event_sig_v108(prev_event_sig=str(self.prev_event_sig or ""), event_body=dict(body))
        return dict(body, prev_event_sig=str(self.prev_event_sig or ""), event_sig=str(sig), event_id=str(flow_event_id_v108(sig)))


def verify_flow_event_sig_v108(ev: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(ev, dict):
        return False, "flow_event_not_dict", {}
    prev_sig = str(ev.get("prev_event_sig") or "")
    want_sig = str(ev.get("event_sig") or "")
    body = dict(ev)
    body.pop("prev_event_sig", None)
    body.pop("event_sig", None)
    body.pop("event_id", None)
    got_sig = _flow_event_sig_v108(prev_event_sig=str(prev_sig), event_body=dict(body))
    if got_sig != want_sig:
        return False, "flow_event_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    want_id = flow_event_id_v108(want_sig)
    got_id = str(ev.get("event_id") or "")
    if got_id != want_id:
        return False, "flow_event_id_mismatch", {"want": str(want_id), "got": str(got_id)}
    return True, "ok", {}


def compute_flow_chain_hash_v108(flow_events: Sequence[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in flow_events:
        if not isinstance(ev, dict):
            continue
        ids.append(str(ev.get("event_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


def fold_flow_ledger_v108(flow_events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    last: Dict[str, Any] = {}
    by_user_turn_index: Dict[str, Dict[str, Any]] = {}
    for ev in flow_events:
        if not isinstance(ev, dict):
            continue
        uidx = _canon_int(ev.get("user_turn_index"), default=0)
        row = {
            "user_turn_index": int(uidx),
            "assistant_turn_index": _canon_int(ev.get("assistant_turn_index"), default=-1),
            "event_id": _canon_str(ev.get("event_id") or ""),
            "flow_score_v108": _canon_int(ev.get("flow_score_v108"), default=0),
            "progress_allowed_v108": _canon_bool(ev.get("progress_allowed_v108", False)),
            "repair_action_v108": _canon_str(ev.get("repair_action_v108") or ""),
        }
        by_user_turn_index[str(uidx)] = dict(row)
        last = dict(row)
    return {
        "schema_version": 108,
        "kind": "flow_registry_v108",
        "by_user_turn_index": {str(k): dict(by_user_turn_index[k]) for k in sorted(by_user_turn_index.keys(), key=str)},
        "last": dict(last),
    }


def flow_registry_snapshot_v108(flow_events: Sequence[Dict[str, Any]], *, flow_memory_last: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    snap = fold_flow_ledger_v108(list(flow_events))
    if isinstance(flow_memory_last, dict):
        snap["flow_memory_last_v108"] = _canon_json_obj(flow_memory_last)
    sig = _stable_hash_obj(snap)
    return dict(snap, snapshot_sig=str(sig))


def _events_prefix(flow_events: Sequence[Dict[str, Any]], *, until_user_turn_index_exclusive: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ev in flow_events:
        if not isinstance(ev, dict):
            continue
        uidx = _canon_int(ev.get("user_turn_index"), default=0)
        if uidx >= int(until_user_turn_index_exclusive):
            break
        out.append(dict(ev))
    return list(out)


def lookup_flow_event_v108(*, flow_events: Sequence[Dict[str, Any]], user_turn_index: int) -> Optional[Dict[str, Any]]:
    for ev in flow_events:
        if not isinstance(ev, dict):
            continue
        if _canon_int(ev.get("user_turn_index"), default=-1) == int(user_turn_index):
            return dict(ev)
    return None


def render_flow_text_v108(*, flow_events: Sequence[Dict[str, Any]], until_user_turn_index_exclusive: int) -> str:
    prefix = _events_prefix(flow_events=list(flow_events), until_user_turn_index_exclusive=int(until_user_turn_index_exclusive))
    last_mem: Dict[str, Any] = {}
    if prefix:
        last = prefix[-1]
        fm = last.get("flow_memory_v108") if isinstance(last.get("flow_memory_v108"), dict) else {}
        last_mem = dict(fm)
    snap = flow_registry_snapshot_v108(prefix, flow_memory_last=dict(last_mem) if last_mem else None)
    mem = snap.get("flow_memory_last_v108") if isinstance(snap.get("flow_memory_last_v108"), dict) else {}
    discourse_plan = mem.get("recent_discourse_acts") if isinstance(mem.get("recent_discourse_acts"), list) else []
    episodes = mem.get("episodic_memory") if isinstance(mem.get("episodic_memory"), list) else []
    lines: List[str] = []
    lines.append("FLOW:")
    vm = str(mem.get("verbosity_mode_v108") or "")
    tm = str(mem.get("tone_mode_v108") or "")
    lines.append(f"verbosity={vm or 'UNKNOWN'} tone={tm or 'UNKNOWN'}")
    if discourse_plan:
        acts = [str(x) for x in discourse_plan if isinstance(x, str) and x]
        lines.append("discourse_acts=" + ",".join(acts[:12]))
    if episodes:
        lines.append("episodes:")
        for i, ep in enumerate(episodes[:8]):
            if not isinstance(ep, dict):
                continue
            tid = str(ep.get("episode_id") or "")
            topic = str(ep.get("topic_label") or "")
            sal = str(ep.get("salience") or "")
            lines.append(f"{i+1}) {topic} id={tid} sal={sal}")
    else:
        lines.append("episodes: (empty)")
    return "\n".join(lines).strip()


def render_explain_flow_text_v108(*, flow_events: Sequence[Dict[str, Any]], user_turn_index: int) -> str:
    ev = lookup_flow_event_v108(flow_events=list(flow_events), user_turn_index=int(user_turn_index))
    if not isinstance(ev, dict):
        return "NO FLOW LOG"
    score = _canon_int(ev.get("flow_score_v108"), default=0)
    pa = bool(ev.get("progress_allowed_v108", False))
    repair = str(ev.get("repair_action_v108") or "")
    flags = ev.get("flow_flags_v108") if isinstance(ev.get("flow_flags_v108"), dict) else {}
    crit = [str(k) for k in sorted(flags.keys(), key=str) if bool(flags.get(k, False)) and str(k)]
    lines: List[str] = []
    lines.append(f"FLOW_EXPLAIN: turn={int(user_turn_index)} score={score} progress_allowed={str(pa).lower()} repair={repair or '-'}")
    lines.append("FLAGS: " + (",".join(crit) if crit else "(none)"))
    cands = ev.get("candidates") if isinstance(ev.get("candidates"), list) else []
    parts: List[str] = []
    for i, c in enumerate(cands[:5]):
        if not isinstance(c, dict):
            continue
        ch = str(c.get("candidate_hash") or "")[:12]
        base = min(int(c.get("coherence_score_v106") or 0), int(c.get("pragmatics_score_v107") or 0), int(c.get("flow_score_v108") or 0))
        chosen = bool(c.get("chosen", False))
        parts.append(f"{i+1}){ch} s={base} chosen={str(chosen).lower()}")
    lines.append("CANDS: " + ("; ".join(parts) if parts else "(none)"))
    return "\n".join(lines).strip()


def render_trace_flow_text_v108(*, flow_events: Sequence[Dict[str, Any]], until_user_turn_index: int) -> str:
    prefix = _events_prefix(flow_events=list(flow_events), until_user_turn_index_exclusive=int(until_user_turn_index) + 1)
    payload = {"schema_version": 108, "kind": "flow_trace_v108", "events": list(prefix)}
    return canonical_json_dumps(payload)
