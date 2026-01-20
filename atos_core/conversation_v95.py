from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .intent_grammar_v93 import expected_learned_rule_id_v93, verify_learned_rule_sig_v93
from .intent_grammar_v95 import INTENT_FORGET_V95, INTENT_NOTE_V95, INTENT_RECALL_V95


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _safe_deepcopy(obj: Any) -> Any:
    try:
        return copy.deepcopy(obj)
    except Exception:
        if isinstance(obj, dict):
            return dict(obj)
        if isinstance(obj, list):
            return list(obj)
        return obj


def normalize_text_v95(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").strip()


def text_sig_v95(text: str) -> str:
    return sha256_hex(normalize_text_v95(text).encode("utf-8"))


def turn_id_v95(*, conversation_id: str, turn_index: int, role: str, text_sig: str) -> str:
    body = {
        "conversation_id": str(conversation_id),
        "turn_index": int(turn_index),
        "role": str(role),
        "text_sig": str(text_sig),
    }
    return f"turn_v95_{_stable_hash_obj(body)}"


@dataclass(frozen=True)
class TurnV95:
    conversation_id: str
    turn_index: int
    role: str  # "user" | "assistant"
    text: str
    created_step: int
    offset_us: int = 0
    objective_id: str = ""
    objective_kind: str = ""
    action_concept_id: str = ""
    eval_id: str = ""
    parse_sig: str = ""
    intent_id: str = ""
    matched_rule_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        text_norm = normalize_text_v95(self.text)
        sig = text_sig_v95(text_norm)
        tid = turn_id_v95(conversation_id=str(self.conversation_id), turn_index=int(self.turn_index), role=str(self.role), text_sig=str(sig))
        return {
            "kind": "turn_v95",
            "turn_id": str(tid),
            "conversation_id": str(self.conversation_id),
            "turn_index": int(self.turn_index),
            "role": str(self.role),
            "text": str(text_norm),
            "text_sig": str(sig),
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step), offset_us=int(self.offset_us)),
            "refs": {
                "objective_id": str(self.objective_id or ""),
                "objective_kind": str(self.objective_kind or ""),
                "action_concept_id": str(self.action_concept_id or ""),
                "eval_id": str(self.eval_id or ""),
                "parse_sig": str(self.parse_sig or ""),
                "intent_id": str(self.intent_id or ""),
                "matched_rule_id": str(self.matched_rule_id or ""),
            },
        }


def _canon_bindings(bindings: Any) -> Dict[str, Any]:
    if not isinstance(bindings, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in sorted(bindings.keys(), key=str):
        out[str(k)] = _safe_deepcopy(bindings.get(k))
    return out


def _canon_str_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for x in items:
        if isinstance(x, str) and x:
            out.append(x)
    return sorted(set(out))


def state_sig_v95(state_sem_sig: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(state_sem_sig).encode("utf-8"))


def state_id_v95(state_sig: str) -> str:
    return f"conversation_state_v95_{str(state_sig)}"


@dataclass(frozen=True)
class ConversationStateV95:
    conversation_id: str
    state_index: int
    prev_state_id: str
    active_goals: List[str]
    bindings: Dict[str, Any]
    tail_turn_ids: List[str]
    last_user_turn_id: str
    last_assistant_turn_id: str
    created_step: int
    last_step: int

    def to_dict(self) -> Dict[str, Any]:
        sem = {
            "schema_version": 95,
            "kind": "conversation_state_v95",
            "conversation_id": str(self.conversation_id),
            "state_index": int(self.state_index),
            "prev_state_id": str(self.prev_state_id or ""),
            "active_goals": _canon_str_list(self.active_goals),
            "bindings": _canon_bindings(self.bindings),
            "tail_turn_ids": [str(x) for x in self.tail_turn_ids if isinstance(x, str) and x],
            "last_user_turn_id": str(self.last_user_turn_id or ""),
            "last_assistant_turn_id": str(self.last_assistant_turn_id or ""),
            "created_step": int(self.created_step),
            "last_step": int(self.last_step),
            "invariants": {"schema_version": 95, "tail_k_fixed": True},
        }
        sig = state_sig_v95(sem)
        sid = state_id_v95(sig)
        return dict(sem, state_sig=str(sig), state_id=str(sid))


def action_plan_sig_v95(plan_sem_sig: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(plan_sem_sig).encode("utf-8"))


def action_plan_id_v95(plan_sig: str) -> str:
    return f"action_plan_v95_{str(plan_sig)}"


def _round6(x: Any) -> float:
    try:
        return float(round(float(x), 6))
    except Exception:
        return 0.0


def _canon_ranked_candidates_v95(items: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        act_id = str(it.get("act_id") or "")
        if not act_id:
            continue
        out.append({"act_id": act_id, "expected_success": _round6(it.get("expected_success")), "expected_cost": _round6(it.get("expected_cost"))})
    return list(out)


def _canon_attempted_actions_v95(items: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        act_id = str(it.get("act_id") or "")
        eval_id = str(it.get("eval_id") or "")
        if not act_id or not eval_id:
            continue
        out.append({"act_id": act_id, "eval_id": eval_id, "ok": bool(it.get("ok", False))})
    return list(out)


@dataclass(frozen=True)
class ActionPlanV95:
    conversation_id: str
    user_turn_id: str
    user_turn_index: int
    intent_id: str
    parse_sig: str
    objective_kind: str
    objective_id: str
    ranked_candidates: List[Dict[str, Any]]
    attempted_actions: List[Dict[str, Any]]
    chosen_action_id: str
    chosen_eval_id: str
    chosen_ok: bool
    notes: str
    created_step: int
    memory_read_ids: List[str]
    memory_write_event_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        sem = {
            "schema_version": 95,
            "kind": "action_plan_v95",
            "conversation_id": str(self.conversation_id),
            "user_turn_id": str(self.user_turn_id),
            "user_turn_index": int(self.user_turn_index),
            "intent_id": str(self.intent_id),
            "parse_sig": str(self.parse_sig),
            "objective_kind": str(self.objective_kind),
            "objective_id": str(self.objective_id),
            "ranked_candidates": _canon_ranked_candidates_v95(self.ranked_candidates),
            "attempted_actions": _canon_attempted_actions_v95(self.attempted_actions),
            "chosen_action_id": str(self.chosen_action_id),
            "chosen_eval_id": str(self.chosen_eval_id),
            "chosen_ok": bool(self.chosen_ok),
            "notes": str(self.notes),
            "memory_read_ids": _canon_str_list(self.memory_read_ids),
            "memory_write_event_ids": _canon_str_list(self.memory_write_event_ids),
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = action_plan_sig_v95(sem)
        pid = action_plan_id_v95(sig)
        return dict(sem, plan_sig=str(sig), plan_id=str(pid))


def memory_item_sig_v95(item_sem_sig: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(item_sem_sig).encode("utf-8"))


def memory_item_id_v95(memory_sig: str) -> str:
    return f"memory_v95_{str(memory_sig)}"


@dataclass(frozen=True)
class MemoryItemV95:
    conversation_id: str
    memory_text: str
    source_turn_id: str
    created_step: int

    def to_dict(self) -> Dict[str, Any]:
        sem = {
            "schema_version": 95,
            "kind": "memory_item_v95",
            "conversation_id": str(self.conversation_id),
            "memory_text": normalize_text_v95(self.memory_text),
            "source_turn_id": str(self.source_turn_id),
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = memory_item_sig_v95(sem)
        mid = memory_item_id_v95(sig)
        return dict(sem, memory_sig=str(sig), memory_id=str(mid))


def memory_event_sig_v95(event_sem_sig: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(event_sem_sig).encode("utf-8"))


def memory_event_id_v95(event_sig: str) -> str:
    return f"memory_event_v95_{str(event_sig)}"


@dataclass(frozen=True)
class MemoryEventV95:
    conversation_id: str
    event_kind: str  # "ADD" | "RETRACT"
    created_step: int
    source_turn_id: str
    memory_item: Optional[Dict[str, Any]] = None
    target_memory_id: str = ""
    retract_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        sem = {
            "schema_version": 95,
            "kind": "memory_event_v95",
            "conversation_id": str(self.conversation_id),
            "event_kind": str(self.event_kind),
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
            "source_turn_id": str(self.source_turn_id or ""),
            "memory_item": dict(self.memory_item) if isinstance(self.memory_item, dict) else None,
            "target_memory_id": str(self.target_memory_id or ""),
            "retract_reason": str(self.retract_reason or ""),
        }
        sig = memory_event_sig_v95(sem)
        eid = memory_event_id_v95(sig)
        return dict(sem, event_sig=str(sig), event_id=str(eid))


def render_recall_text_v95(active_items: Sequence[Dict[str, Any]]) -> str:
    items: List[Tuple[int, str, str]] = []
    for it in active_items:
        if not isinstance(it, dict):
            continue
        mid = str(it.get("memory_id") or "")
        mtext = str(it.get("memory_text") or "")
        try:
            cstep = int(it.get("created_step", -1))
        except Exception:
            cstep = -1
        if not mid:
            continue
        items.append((int(cstep), mid, mtext))
    items.sort(key=lambda t: (int(t[0]), str(t[1])))
    if not items:
        return "MEMORY: (empty)"
    lines: List[str] = ["MEMORY:"]
    for i, (_cs, mid, txt) in enumerate(items):
        lines.append(f"{i+1}) {mid} text={json.dumps(str(txt), ensure_ascii=False)}")
    return "\n".join(lines)


def render_note_ack_text_v95(memory_id: str) -> str:
    return f"MEMORY ADDED: {str(memory_id)}"


def render_forget_ack_text_v95(target_memory_id: str) -> str:
    return f"MEMORY RETRACTED: {str(target_memory_id)}"


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return iter(())
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_chained_jsonl_v95(path: str, entry: Dict[str, Any], *, prev_hash: Optional[str]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    body = dict(entry)
    body["prev_hash"] = prev_hash
    entry_hash = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    body["entry_hash"] = entry_hash
    with open(path, "a", encoding="utf-8") as f:
        f.write(canonical_json_dumps(body))
        f.write("\n")
    return entry_hash


def verify_chained_jsonl_v95(path: str) -> bool:
    prev: Optional[str] = None
    for row in _read_jsonl(path):
        row = dict(row)
        entry_hash = row.pop("entry_hash", None)
        if row.get("prev_hash") != prev:
            return False
        expected = sha256_hex(canonical_json_dumps(row).encode("utf-8"))
        if expected != entry_hash:
            return False
        prev = str(entry_hash)
    return True


def compute_transcript_hash_v95(turns: Sequence[Dict[str, Any]]) -> str:
    items: List[Dict[str, Any]] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        items.append({"turn_index": int(t.get("turn_index", 0)), "role": str(t.get("role") or ""), "text": str(t.get("text") or "")})
    items.sort(key=lambda r: int(r.get("turn_index", 0)))
    view = [{"role": str(r.get("role") or ""), "text": str(r.get("text") or "")} for r in items]
    return sha256_hex(canonical_json_dumps(view).encode("utf-8"))


def compute_state_chain_hash_v95(states: Sequence[Dict[str, Any]]) -> str:
    sigs: List[str] = []
    for s in states:
        if not isinstance(s, dict):
            continue
        sigs.append(str(s.get("state_sig") or ""))
    return sha256_hex(canonical_json_dumps(sigs).encode("utf-8"))


def compute_parse_chain_hash_v95(parse_events: Sequence[Dict[str, Any]]) -> str:
    sigs: List[str] = []
    for e in parse_events:
        if not isinstance(e, dict):
            continue
        payload = e.get("payload")
        if isinstance(payload, dict):
            sigs.append(str(payload.get("parse_sig") or ""))
    return sha256_hex(canonical_json_dumps(sigs).encode("utf-8"))


def compute_learned_chain_hash_v95(learned_events: Sequence[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in learned_events:
        if not isinstance(ev, dict):
            continue
        lr = ev.get("learned_rule")
        if isinstance(lr, dict):
            ids.append(str(lr.get("rule_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


def compute_plan_chain_hash_v95(plans: Sequence[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for p in plans:
        if not isinstance(p, dict):
            continue
        ids.append(str(p.get("plan_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


def compute_memory_chain_hash_v95(memory_events: Sequence[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in memory_events:
        if not isinstance(ev, dict):
            continue
        ids.append(str(ev.get("event_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


def _verify_parse_payload_sig_v95(payload: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    payload2 = dict(payload)
    got_sig = str(payload2.pop("parse_sig", "") or "")
    if not got_sig:
        return False, "missing_parse_sig", {}
    want_sig = sha256_hex(canonical_json_dumps(payload2).encode("utf-8"))
    if want_sig != got_sig:
        return False, "parse_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}

    if bool(payload.get("compound", False)):
        segs = payload.get("segments")
        if not isinstance(segs, list):
            return False, "compound_segments_not_list", {}
        for seg in segs:
            if not isinstance(seg, dict):
                return False, "segment_not_dict", {}
            sidx = int(seg.get("segment_index", -1))
            seg_parse = seg.get("segment_parse")
            if not isinstance(seg_parse, dict):
                return False, "segment_missing_parse", {"segment_index": int(sidx)}
            sp2 = dict(seg_parse)
            got2 = str(sp2.pop("parse_sig", "") or "")
            if not got2:
                return False, "segment_missing_parse_sig", {"segment_index": int(sidx)}
            want2 = sha256_hex(canonical_json_dumps(sp2).encode("utf-8"))
            if want2 != got2:
                return False, "segment_parse_sig_mismatch", {"segment_index": int(sidx)}

        want_ok = True
        for seg in segs:
            if not isinstance(seg, dict):
                want_ok = False
                break
            sp = seg.get("segment_parse")
            if not isinstance(sp, dict) or not bool(sp.get("parse_ok", False)):
                want_ok = False
                break
            missing = sp.get("missing_slots")
            if isinstance(missing, list) and missing:
                want_ok = False
                break
        if bool(payload.get("parse_ok", False)) != bool(want_ok):
            return False, "compound_parse_ok_inconsistent", {"want": bool(want_ok), "got": bool(payload.get("parse_ok", False))}

    return True, "ok", {}


def _verify_plan_sig_v95(plan: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    d = dict(plan)
    got_sig = str(d.pop("plan_sig", "") or "")
    got_id = str(d.pop("plan_id", "") or "")
    if not got_sig:
        return False, "missing_plan_sig", {}
    want_sig = sha256_hex(canonical_json_dumps(d).encode("utf-8"))
    if want_sig != got_sig:
        return False, "plan_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    want_id = action_plan_id_v95(got_sig)
    if want_id != got_id:
        return False, "plan_id_mismatch", {"want": str(want_id), "got": str(got_id)}
    return True, "ok", {}


def _verify_memory_item_sig_v95(item: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    d = dict(item)
    got_sig = str(d.pop("memory_sig", "") or "")
    got_id = str(d.pop("memory_id", "") or "")
    if not got_sig:
        return False, "missing_memory_sig", {}
    want_sig = sha256_hex(canonical_json_dumps(d).encode("utf-8"))
    if want_sig != got_sig:
        return False, "memory_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    want_id = memory_item_id_v95(got_sig)
    if want_id != got_id:
        return False, "memory_id_mismatch", {"want": str(want_id), "got": str(got_id)}
    return True, "ok", {}


def _verify_memory_event_sig_v95(ev: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    d = dict(ev)
    got_sig = str(d.pop("event_sig", "") or "")
    got_id = str(d.pop("event_id", "") or "")
    if not got_sig:
        return False, "missing_event_sig", {}
    want_sig = sha256_hex(canonical_json_dumps(d).encode("utf-8"))
    if want_sig != got_sig:
        return False, "event_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    want_id = memory_event_id_v95(got_sig)
    if want_id != got_id:
        return False, "event_id_mismatch", {"want": str(want_id), "got": str(got_id)}
    return True, "ok", {}


def _memory_active_after_step_v95(*, memory_events: Sequence[Dict[str, Any]], step: int) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Replay memory events up to and including created_step<=step.
    Returns (active_items_by_id, all_items_by_id).
    """
    all_items: Dict[str, Dict[str, Any]] = {}
    active: Dict[str, Dict[str, Any]] = {}
    for ev in memory_events:
        if not isinstance(ev, dict):
            continue
        try:
            cstep = int(ev.get("created_step", -1))
        except Exception:
            cstep = -1
        if cstep < 0 or cstep > int(step):
            continue
        if str(ev.get("event_kind") or "") == "ADD":
            mi = ev.get("memory_item")
            if isinstance(mi, dict):
                mid = str(mi.get("memory_id") or "")
                if mid:
                    all_items[mid] = dict(mi)
                    active[mid] = dict(mi)
        elif str(ev.get("event_kind") or "") == "RETRACT":
            tid = str(ev.get("target_memory_id") or "")
            if tid and tid in active:
                active.pop(tid, None)
    return dict(active), dict(all_items)


def verify_conversation_chain_v95(
    *,
    turns: Sequence[Dict[str, Any]],
    states: Sequence[Dict[str, Any]],
    parse_events: Sequence[Dict[str, Any]],
    trials: Sequence[Dict[str, Any]],
    learned_rule_events: Sequence[Dict[str, Any]],
    action_plans: Sequence[Dict[str, Any]],
    memory_events: Sequence[Dict[str, Any]],
    tail_k: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    by_index: Dict[int, Dict[str, Any]] = {}
    by_id: Dict[str, Dict[str, Any]] = {}
    max_idx = -1
    for t in turns:
        if not isinstance(t, dict):
            return False, "turn_not_dict", {}
        tid = str(t.get("turn_id") or "")
        if not tid:
            return False, "missing_turn_id", {}
        try:
            idx = int(t.get("turn_index", -1))
        except Exception:
            idx = -1
        if idx < 0:
            return False, "bad_turn_index", {"turn_id": tid}
        if idx in by_index:
            return False, "duplicate_turn_index", {"turn_index": int(idx)}
        by_index[int(idx)] = dict(t)
        by_id[tid] = dict(t)
        max_idx = max(max_idx, int(idx))
    for i in range(0, max_idx + 1):
        if i not in by_index:
            return False, "missing_turn_index", {"missing_turn_index": int(i)}
    for i in range(0, max_idx + 1):
        role = str(by_index[i].get("role") or "")
        want_role = "user" if (i % 2 == 0) else "assistant"
        if role != want_role:
            return False, "turn_role_mismatch", {"turn_index": int(i), "want": want_role, "got": role}

    parses_by_turn_id: Dict[str, Dict[str, Any]] = {}
    parse_order: List[Tuple[int, str]] = []
    for pe in parse_events:
        if not isinstance(pe, dict):
            return False, "parse_event_not_dict", {}
        tid = str(pe.get("turn_id") or "")
        if not tid:
            return False, "parse_event_missing_turn_id", {}
        try:
            tix = int(pe.get("turn_index", -1))
        except Exception:
            tix = -1
        if tix < 0:
            return False, "parse_event_bad_turn_index", {"turn_id": tid}
        payload = pe.get("payload")
        if not isinstance(payload, dict):
            return False, "parse_event_missing_payload", {"turn_id": tid}
        if tid in parses_by_turn_id:
            return False, "duplicate_parse_event_turn_id", {"turn_id": tid}
        parses_by_turn_id[tid] = dict(payload)
        parse_order.append((int(tix), tid))

        ok_sig, rsig, dsig = _verify_parse_payload_sig_v95(payload=dict(payload))
        if not ok_sig:
            d = dict(dsig)
            d["turn_id"] = str(tid)
            return False, str(rsig), d

    parse_order.sort(key=lambda x: (int(x[0]), str(x[1])))
    for i in range(0, max_idx + 1, 2):
        t = by_index[i]
        tid = str(t.get("turn_id") or "")
        if tid not in parses_by_turn_id:
            return False, "missing_parse_for_user_turn", {"turn_index": int(i), "turn_id": tid}
        pref = t.get("refs") if isinstance(t.get("refs"), dict) else {}
        want_parse_sig = str(pref.get("parse_sig") or "")
        want_intent_id = str(pref.get("intent_id") or "")
        want_rule_id = str(pref.get("matched_rule_id") or "")
        payload = parses_by_turn_id[tid]
        if want_parse_sig != str(payload.get("parse_sig") or ""):
            return False, "turn_parse_sig_mismatch", {"turn_index": int(i)}
        if want_intent_id != str(payload.get("intent_id") or ""):
            return False, "turn_intent_id_mismatch", {"turn_index": int(i)}
        if want_rule_id != str(payload.get("matched_rule_id") or ""):
            return False, "turn_matched_rule_id_mismatch", {"turn_index": int(i)}

    parse_turn_indices = [int(ix) for ix, _tid in parse_order]
    if parse_turn_indices != sorted(parse_turn_indices):
        return False, "parse_events_not_sorted", {}
    for ix in parse_turn_indices:
        if ix % 2 != 0:
            return False, "parse_event_on_non_user_turn", {"turn_index": int(ix)}

    # Learned rules integrity + mapping by teacher_turn_id.
    learned_by_teacher: Dict[str, Dict[str, Any]] = {}
    for ev in learned_rule_events:
        if not isinstance(ev, dict):
            return False, "learned_event_not_dict", {}
        teacher_turn_id = str(ev.get("teacher_turn_id") or "")
        if not teacher_turn_id:
            return False, "learned_missing_teacher_turn_id", {}
        if teacher_turn_id in learned_by_teacher:
            return False, "duplicate_learned_teacher_turn_id", {"teacher_turn_id": teacher_turn_id}
        lr = ev.get("learned_rule")
        if not isinstance(lr, dict):
            return False, "learned_missing_rule", {"teacher_turn_id": teacher_turn_id}
        want_rule_id = expected_learned_rule_id_v93(
            intent_id=str(lr.get("intent_id") or ""),
            pattern=lr.get("pattern") if isinstance(lr.get("pattern"), list) else [],
            required_slots=lr.get("required_slots") if isinstance(lr.get("required_slots"), list) else [],
        )
        got_rule_id = str(lr.get("rule_id") or "")
        if want_rule_id != got_rule_id:
            return False, "learned_rule_id_mismatch", {"teacher_turn_id": teacher_turn_id, "want": want_rule_id, "got": got_rule_id}
        ok_rs, rreason, rdetails = verify_learned_rule_sig_v93(dict(lr))
        if not ok_rs:
            d = dict(rdetails)
            d["teacher_turn_id"] = teacher_turn_id
            return False, str(rreason), d
        learned_by_teacher[teacher_turn_id] = dict(ev)

    # States: chain + tail correctness.
    if states:
        prev_state_id = ""
        prev_created_step = -1
        for i, s in enumerate(states):
            if not isinstance(s, dict):
                return False, "state_not_dict", {"index": int(i)}
            try:
                got_state_index = int(s.get("state_index", -1))
            except Exception:
                got_state_index = -1
            if got_state_index != int(i):
                return False, "state_index_not_incrementing", {"index": int(i), "got": s.get("state_index")}
            if i == 0:
                if str(s.get("prev_state_id") or "") not in ("", "None"):
                    return False, "genesis_prev_state_not_empty", {"got": s.get("prev_state_id")}
            else:
                if str(s.get("prev_state_id") or "") != str(prev_state_id):
                    return False, "prev_state_id_mismatch", {"index": int(i)}

            try:
                created_step = int(s.get("created_step", -1))
                last_step = int(s.get("last_step", -1))
            except Exception:
                return False, "bad_step_fields", {"index": int(i)}
            if created_step < 0 or last_step < 0:
                return False, "bad_step_fields", {"index": int(i)}
            if created_step < prev_created_step:
                return False, "created_step_not_monotonic", {"index": int(i)}
            if last_step < created_step:
                return False, "last_step_before_created_step", {"index": int(i)}

            s2 = dict(s)
            got_sig = str(s2.pop("state_sig", "") or "")
            got_state_id = str(s2.pop("state_id", "") or "")
            if not got_sig:
                return False, "missing_state_sig", {"index": int(i)}
            if got_state_id != state_id_v95(got_sig):
                return False, "state_id_mismatch", {"index": int(i)}
            want_sig = state_sig_v95(s2)
            if want_sig != got_sig:
                return False, "state_sig_mismatch", {"index": int(i)}

            last_user_tid = str(s.get("last_user_turn_id") or "")
            last_asst_tid = str(s.get("last_assistant_turn_id") or "")
            if last_user_tid not in by_id or last_asst_tid not in by_id:
                return False, "missing_last_turn_refs", {"index": int(i)}
            tu = by_id[last_user_tid]
            ta = by_id[last_asst_tid]
            if str(tu.get("role") or "") != "user" or str(ta.get("role") or "") != "assistant":
                return False, "last_turn_roles_wrong", {"index": int(i)}
            try:
                tu_idx = int(tu.get("turn_index", -1))
                ta_idx = int(ta.get("turn_index", -1))
            except Exception:
                return False, "last_turn_index_pair_invalid", {"index": int(i)}
            if tu_idx < 0 or ta_idx < 0 or (ta_idx != tu_idx + 1) or (tu_idx % 2 != 0):
                return False, "last_turn_index_pair_invalid", {"index": int(i)}

            tail = s.get("tail_turn_ids")
            tail = tail if isinstance(tail, list) else []
            tail2 = [str(x) for x in tail if isinstance(x, str) and x]
            if len(tail2) > int(tail_k):
                return False, "tail_too_long", {"index": int(i)}
            end_idx = int(ta_idx)
            start_idx = max(0, end_idx - (int(tail_k) - 1))
            expected_tail: List[str] = []
            for j in range(start_idx, end_idx + 1):
                expected_tail.append(str(by_index[int(j)].get("turn_id") or ""))
            if tail2 != expected_tail:
                return False, "tail_mismatch", {"index": int(i)}

            prev_state_id = str(got_state_id)
            prev_created_step = int(created_step)

    # Clarifying objectives must NOT create state.
    clarifying = {"COMM_ASK_CLARIFY", "COMM_CONFIRM"}
    state_user_turn_ids = {str(s.get("last_user_turn_id") or "") for s in states if isinstance(s, dict)}
    for tr in trials:
        if not isinstance(tr, dict):
            continue
        okind = str(tr.get("objective_kind") or "")
        if okind not in clarifying:
            continue
        utid = str(tr.get("user_turn_id") or "")
        if utid and utid in state_user_turn_ids:
            return False, "state_updated_on_clarification", {"user_turn_id": utid, "objective_kind": okind}

    # TEACH accept/reject invariants (based on parse payload).
    for i in range(0, max_idx + 1, 2):
        ut = by_index[i]
        utid = str(ut.get("turn_id") or "")
        payload = parses_by_turn_id.get(utid) if utid else None
        if not isinstance(payload, dict):
            continue
        if str(payload.get("intent_id") or "") != "INTENT_TEACH":
            continue
        teach_ok = bool(payload.get("teach_ok", False))
        learned_rule_id = str(payload.get("learned_rule_id") or "")
        if teach_ok:
            if utid not in learned_by_teacher:
                return False, "teach_ok_missing_learned_event", {"turn_id": utid}
            ev = learned_by_teacher[utid]
            lr = ev.get("learned_rule") if isinstance(ev.get("learned_rule"), dict) else {}
            if learned_rule_id and str(lr.get("rule_id") or "") != learned_rule_id:
                return False, "teach_learned_rule_id_mismatch", {"turn_id": utid}
            found_state = False
            for s in states:
                if not isinstance(s, dict):
                    continue
                if str(s.get("last_user_turn_id") or "") != utid:
                    continue
                bindings = s.get("bindings") if isinstance(s.get("bindings"), dict) else {}
                lra = bindings.get("learned_rules_active")
                lra_list = [str(x) for x in lra if isinstance(lra, list) and isinstance(x, str) and x]
                if learned_rule_id and learned_rule_id not in set(lra_list):
                    return False, "teach_state_missing_learned_rule", {"turn_id": utid}
                found_state = True
                break
            if not found_state:
                return False, "teach_ok_missing_state_update", {"turn_id": utid}
        else:
            if utid in learned_by_teacher:
                return False, "teach_reject_has_learned_event", {"turn_id": utid}

    # Memory events: validate sigs + replay active set + check state bindings.
    active_all_by_step_cache: Dict[int, Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]] = {}
    prev_ev_step = -1
    memory_by_event_id: Dict[str, Dict[str, Any]] = {}
    all_items_seen: Dict[str, Dict[str, Any]] = {}
    for ev in memory_events:
        if not isinstance(ev, dict):
            return False, "memory_event_not_dict", {}
        ok_es, ereason, edetails = _verify_memory_event_sig_v95(dict(ev))
        if not ok_es:
            return False, str(ereason), dict(edetails)
        eid = str(ev.get("event_id") or "")
        if eid in memory_by_event_id:
            return False, "duplicate_memory_event_id", {"event_id": eid}
        memory_by_event_id[eid] = dict(ev)
        try:
            cstep = int(ev.get("created_step", -1))
        except Exception:
            cstep = -1
        if cstep < 0:
            return False, "memory_event_bad_created_step", {"event_id": eid}
        if cstep < prev_ev_step:
            return False, "memory_events_not_monotonic", {"event_id": eid}
        prev_ev_step = int(cstep)
        if str(ev.get("event_kind") or "") == "ADD":
            mi = ev.get("memory_item")
            if not isinstance(mi, dict):
                return False, "memory_add_missing_item", {"event_id": eid}
            ok_mi, mreason, mdetails = _verify_memory_item_sig_v95(dict(mi))
            if not ok_mi:
                return False, str(mreason), dict(mdetails)
            mid = str(mi.get("memory_id") or "")
            if mid:
                all_items_seen[mid] = dict(mi)
        elif str(ev.get("event_kind") or "") == "RETRACT":
            tid = str(ev.get("target_memory_id") or "")
            if not tid:
                return False, "memory_retract_missing_target", {"event_id": eid}
            if tid not in all_items_seen:
                return False, "memory_retract_unknown_target", {"event_id": eid, "target_memory_id": tid}

    for s in states:
        if not isinstance(s, dict):
            continue
        try:
            sstep = int(s.get("created_step", -1))
        except Exception:
            sstep = -1
        if sstep < 0:
            return False, "state_bad_created_step", {}
        if sstep not in active_all_by_step_cache:
            active, all_items = _memory_active_after_step_v95(memory_events=memory_events, step=int(sstep))
            active_all_by_step_cache[int(sstep)] = (active, all_items)
        active_at, _all_at = active_all_by_step_cache[int(sstep)]
        bindings = s.get("bindings") if isinstance(s.get("bindings"), dict) else {}
        if "memory_active" not in bindings:
            return False, "memory_active_missing", {"state_id": str(s.get("state_id") or "")}
        mem_active = bindings.get("memory_active")
        if not isinstance(mem_active, list):
            return False, "memory_active_not_list", {"state_id": str(s.get("state_id") or ""), "got_type": str(type(mem_active).__name__)}
        mem_active_list: List[str] = []
        for x in mem_active:
            if not isinstance(x, str) or not x:
                return False, "memory_active_bad_item", {"state_id": str(s.get("state_id") or ""), "item": x}
            mem_active_list.append(str(x))
        mem_active_list2 = sorted(set(mem_active_list))
        if mem_active_list != mem_active_list2:
            return False, "memory_active_not_sorted_unique", {"state_id": str(s.get("state_id") or ""), "got": list(mem_active_list)}
        want_active = sorted(active_at.keys(), key=str)
        if mem_active_list2 != want_active:
            return False, "memory_active_mismatch", {"state_id": str(s.get("state_id") or ""), "want": want_active, "got": mem_active_list2}
        if "memory_active_count" not in bindings:
            return False, "memory_active_count_missing", {"state_id": str(s.get("state_id") or "")}
        try:
            cnt = int(bindings.get("memory_active_count", -1))
        except Exception:
            cnt = -1
        if cnt != int(len(want_active)):
            return False, "memory_active_count_mismatch", {"state_id": str(s.get("state_id") or ""), "want": int(len(want_active)), "got": int(cnt)}

    # Action plans: validate sigs and cross-check against turns + memory.
    plans_by_user_turn_id: Dict[str, Dict[str, Any]] = {}
    plan_user_indices: List[int] = []
    for p in action_plans:
        if not isinstance(p, dict):
            return False, "plan_not_dict", {}
        ok_ps, preason, pdetails = _verify_plan_sig_v95(dict(p))
        if not ok_ps:
            return False, str(preason), dict(pdetails)
        utid = str(p.get("user_turn_id") or "")
        if not utid:
            return False, "plan_missing_user_turn_id", {}
        if utid in plans_by_user_turn_id:
            return False, "duplicate_plan_user_turn_id", {"user_turn_id": utid}
        try:
            uidx = int(p.get("user_turn_index", -1))
        except Exception:
            uidx = -1
        if uidx < 0:
            return False, "plan_bad_user_turn_index", {"user_turn_id": utid}
        plans_by_user_turn_id[utid] = dict(p)
        plan_user_indices.append(int(uidx))

        ranked = p.get("ranked_candidates")
        ranked_list = ranked if isinstance(ranked, list) else []
        ranked_ids = [str(x.get("act_id") or "") for x in ranked_list if isinstance(x, dict)]
        if not ranked_ids:
            return False, "plan_ranked_candidates_empty", {"user_turn_id": utid}
        if str(p.get("chosen_action_id") or "") not in set(ranked_ids):
            return False, "plan_chosen_not_in_ranked", {"user_turn_id": utid}

        attempted = p.get("attempted_actions")
        attempted_list = attempted if isinstance(attempted, list) else []
        attempted_ids = [str(x.get("act_id") or "") for x in attempted_list if isinstance(x, dict)]
        if ranked_ids[: len(attempted_ids)] != attempted_ids:
            return False, "plan_attempted_not_prefix_of_ranked", {"user_turn_id": utid}
        chosen_eval_id = str(p.get("chosen_eval_id") or "")
        chosen_ok = bool(p.get("chosen_ok", False))
        if not attempted_list:
            return False, "plan_attempted_actions_empty", {"user_turn_id": utid}
        ok_indices = [i for i, a in enumerate(attempted_list) if isinstance(a, dict) and bool(a.get("ok", False))]
        if chosen_ok:
            if not ok_indices:
                return False, "plan_chosen_ok_true_but_no_ok_attempt", {"user_turn_id": utid}
            first_ok = int(ok_indices[0])
            a = attempted_list[first_ok]
            if str(a.get("act_id") or "") != str(p.get("chosen_action_id") or ""):
                return False, "plan_chosen_action_not_first_ok", {"user_turn_id": utid}
            if str(a.get("eval_id") or "") != chosen_eval_id:
                return False, "plan_chosen_eval_not_first_ok", {"user_turn_id": utid}
        else:
            if ok_indices:
                return False, "plan_chosen_ok_false_but_has_ok_attempt", {"user_turn_id": utid}

        # Memory cross-check for NOTE/RECALL/FORGET.
        mem_read = p.get("memory_read_ids")
        mem_write = p.get("memory_write_event_ids")
        mem_read_ids = _canon_str_list(mem_read)
        mem_write_ids = _canon_str_list(mem_write)
        if _canon_str_list(mem_read) != mem_read_ids or _canon_str_list(mem_write) != mem_write_ids:
            return False, "plan_memory_lists_not_canonical", {"user_turn_id": utid}

        intent_id = str(p.get("intent_id") or "")
        if intent_id == INTENT_NOTE_V95:
            parse_payload = parses_by_turn_id.get(utid, {})
            note_ok = bool(parse_payload.get("parse_ok", False))
            if note_ok and not mem_write_ids:
                return False, "note_missing_write_event_ids", {"user_turn_id": utid}
            if mem_write_ids:
                for wid in mem_write_ids:
                    if wid not in memory_by_event_id:
                        return False, "note_unknown_write_event_id", {"user_turn_id": utid, "event_id": wid}
                    ev = memory_by_event_id[wid]
                    if str(ev.get("event_kind") or "") != "ADD":
                        return False, "note_write_event_not_add", {"user_turn_id": utid, "event_id": wid}
        elif intent_id == INTENT_RECALL_V95:
            if mem_write_ids:
                return False, "recall_has_write_event_ids", {"user_turn_id": utid}
            # mem_read_ids must match active set at assistant turn step.
            assistant_turn = by_index.get(int(uidx) + 1, {})
            try:
                astep = int(assistant_turn.get("created_step", -1))
            except Exception:
                astep = -1
            if astep < 0:
                return False, "recall_missing_assistant_step", {"user_turn_id": utid}
            active_at, _all_at = _memory_active_after_step_v95(memory_events=memory_events, step=int(astep))
            want_ids = sorted(active_at.keys(), key=str)
            if mem_read_ids != want_ids:
                return False, "recall_memory_read_ids_mismatch", {"user_turn_id": utid, "want": want_ids, "got": mem_read_ids}
        elif intent_id == INTENT_FORGET_V95:
            assistant_turn = by_index.get(int(uidx) + 1, {})
            try:
                ustep = int(by_index.get(int(uidx), {}).get("created_step", -1))
            except Exception:
                ustep = -1
            active_before, _all_before = _memory_active_after_step_v95(memory_events=memory_events, step=max(ustep - 1, 0))
            if active_before:
                if not mem_write_ids:
                    return False, "forget_missing_write_event_ids", {"user_turn_id": utid}
                if not mem_read_ids:
                    return False, "forget_missing_memory_read_ids", {"user_turn_id": utid}
            if mem_write_ids:
                for wid in mem_write_ids:
                    if wid not in memory_by_event_id:
                        return False, "forget_unknown_write_event_id", {"user_turn_id": utid, "event_id": wid}
                    ev = memory_by_event_id[wid]
                    if str(ev.get("event_kind") or "") != "RETRACT":
                        return False, "forget_write_event_not_retract", {"user_turn_id": utid, "event_id": wid}
                    tid = str(ev.get("target_memory_id") or "")
                    if mem_read_ids and tid and tid not in set(mem_read_ids):
                        return False, "forget_target_not_in_memory_read_ids", {"user_turn_id": utid, "target_memory_id": tid}

    if plan_user_indices != sorted(plan_user_indices):
        return False, "plans_not_sorted_by_user_turn_index", {}

    for i in range(0, max_idx + 1, 2):
        utid = str(by_index[i].get("turn_id") or "")
        if utid not in plans_by_user_turn_id:
            return False, "missing_plan_for_user_turn", {"turn_index": int(i), "turn_id": utid}
        plan = plans_by_user_turn_id[utid]
        if int(plan.get("user_turn_index", -1)) != int(i):
            return False, "plan_user_turn_index_mismatch", {"turn_index": int(i), "turn_id": utid}
        pp = parses_by_turn_id.get(utid, {})
        if str(plan.get("intent_id") or "") != str(pp.get("intent_id") or ""):
            return False, "plan_intent_id_mismatch", {"turn_index": int(i)}
        if str(plan.get("parse_sig") or "") != str(pp.get("parse_sig") or ""):
            return False, "plan_parse_sig_mismatch", {"turn_index": int(i)}
        at = by_index[i + 1]
        aref = at.get("refs") if isinstance(at.get("refs"), dict) else {}
        if str(plan.get("objective_kind") or "") != str(aref.get("objective_kind") or ""):
            return False, "plan_objective_kind_mismatch", {"turn_index": int(i)}
        if str(plan.get("objective_id") or "") != str(aref.get("objective_id") or ""):
            return False, "plan_objective_id_mismatch", {"turn_index": int(i)}
        if str(plan.get("chosen_action_id") or "") != str(aref.get("action_concept_id") or ""):
            return False, "plan_action_id_mismatch", {"turn_index": int(i)}
        if str(plan.get("chosen_eval_id") or "") != str(aref.get("eval_id") or ""):
            return False, "plan_eval_id_mismatch", {"turn_index": int(i)}

    # RECALL output must match renderer based on memory_active at that time.
    for i in range(0, max_idx + 1, 2):
        ut = by_index[i]
        utid = str(ut.get("turn_id") or "")
        payload = parses_by_turn_id.get(utid, {})
        if str(payload.get("intent_id") or "") != INTENT_RECALL_V95:
            continue
        at = by_index[i + 1]
        try:
            astep = int(at.get("created_step", -1))
        except Exception:
            astep = -1
        if astep < 0:
            return False, "recall_missing_assistant_step", {"turn_id": utid}
        active_at, _all_at = _memory_active_after_step_v95(memory_events=memory_events, step=int(astep))
        want = render_recall_text_v95(list(active_at.values()))
        got = str(at.get("text") or "")
        if want != got:
            return False, "recall_text_mismatch", {"turn_id": utid, "want": want, "got": got}

    return True, "ok", {
        "turns_total": int(max_idx + 1),
        "states_total": int(len(states)),
        "parses_total": int(len(parse_events)),
        "learned_total": int(len(learned_rule_events)),
        "plans_total": int(len(action_plans)),
        "memory_events_total": int(len(memory_events)),
    }
