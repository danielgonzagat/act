from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .intent_grammar_v93 import expected_learned_rule_id_v93, verify_learned_rule_sig_v93


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


def normalize_text_v93(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").strip()


def text_sig_v93(text: str) -> str:
    return sha256_hex(normalize_text_v93(text).encode("utf-8"))


def turn_id_v93(*, conversation_id: str, turn_index: int, role: str, text_sig: str) -> str:
    body = {
        "conversation_id": str(conversation_id),
        "turn_index": int(turn_index),
        "role": str(role),
        "text_sig": str(text_sig),
    }
    return f"turn_v93_{_stable_hash_obj(body)}"


@dataclass(frozen=True)
class TurnV93:
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
        text_norm = normalize_text_v93(self.text)
        sig = text_sig_v93(text_norm)
        tid = turn_id_v93(
            conversation_id=str(self.conversation_id),
            turn_index=int(self.turn_index),
            role=str(self.role),
            text_sig=str(sig),
        )
        return {
            "kind": "turn_v93",
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


def state_sig_v93(state_sem_sig: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(state_sem_sig).encode("utf-8"))


def state_id_v93(state_sig: str) -> str:
    return f"conversation_state_v93_{str(state_sig)}"


@dataclass(frozen=True)
class ConversationStateV93:
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
            "schema_version": 93,
            "kind": "conversation_state_v93",
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
            "invariants": {"schema_version": 93, "tail_k_fixed": True},
        }
        sig = state_sig_v93(sem)
        sid = state_id_v93(sig)
        return dict(sem, state_sig=str(sig), state_id=str(sid))


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return iter(())
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_chained_jsonl_v93(path: str, entry: Dict[str, Any], *, prev_hash: Optional[str]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    body = dict(entry)
    body["prev_hash"] = prev_hash
    entry_hash = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    body["entry_hash"] = entry_hash
    with open(path, "a", encoding="utf-8") as f:
        f.write(canonical_json_dumps(body))
        f.write("\n")
    return entry_hash


def verify_chained_jsonl_v93(path: str) -> bool:
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


def compute_transcript_hash_v93(turns: Sequence[Dict[str, Any]]) -> str:
    items: List[Dict[str, Any]] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        items.append(
            {
                "turn_index": int(t.get("turn_index", 0) or 0),
                "role": str(t.get("role") or ""),
                "text": str(t.get("text") or ""),
            }
        )
    items.sort(key=lambda r: int(r.get("turn_index", 0) or 0))
    view = [{"role": str(r.get("role") or ""), "text": str(r.get("text") or "")} for r in items]
    return sha256_hex(canonical_json_dumps(view).encode("utf-8"))


def compute_state_chain_hash_v93(states: Sequence[Dict[str, Any]]) -> str:
    sigs: List[str] = []
    for s in states:
        if not isinstance(s, dict):
            continue
        sigs.append(str(s.get("state_sig") or ""))
    return sha256_hex(canonical_json_dumps(sigs).encode("utf-8"))


def compute_parse_chain_hash_v93(parse_events: Sequence[Dict[str, Any]]) -> str:
    sigs: List[str] = []
    for e in parse_events:
        if not isinstance(e, dict):
            continue
        payload = e.get("payload")
        if isinstance(payload, dict):
            sigs.append(str(payload.get("parse_sig") or ""))
    return sha256_hex(canonical_json_dumps(sigs).encode("utf-8"))


def compute_learned_chain_hash_v93(learned_events: Sequence[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in learned_events:
        if not isinstance(ev, dict):
            continue
        lr = ev.get("learned_rule")
        if isinstance(lr, dict):
            ids.append(str(lr.get("rule_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


def _verify_parse_payload_sig_v93(payload: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    payload2 = dict(payload)
    got_sig = str(payload2.pop("parse_sig", "") or "")
    if not got_sig:
        return False, "missing_parse_sig", {}
    want_sig = sha256_hex(canonical_json_dumps(payload2).encode("utf-8"))
    if want_sig != got_sig:
        return False, "parse_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}

    # Compound payload invariants (reuse V92 structure).
    if bool(payload.get("compound", False)):
        segs = payload.get("segments")
        if not isinstance(segs, list):
            return False, "compound_segments_not_list", {}
        for seg in segs:
            if not isinstance(seg, dict):
                return False, "segment_not_dict", {}
            sidx = int(seg.get("segment_index", -1) or -1)
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


def verify_conversation_chain_v93(
    *,
    turns: Sequence[Dict[str, Any]],
    states: Sequence[Dict[str, Any]],
    parse_events: Sequence[Dict[str, Any]],
    trials: Sequence[Dict[str, Any]],
    learned_rule_events: Sequence[Dict[str, Any]],
    tail_k: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Verify V93 invariants:
      - turns contiguous + alternating roles
      - Turn(user).refs parse_sig/intent_id/matched_rule_id match parse_events
      - parse_sig integrity (including nested segment parses when compound)
      - states hash-consistent + prev_state chain + tail correctness
      - COMM_ASK_CLARIFY / COMM_CONFIRM -> no state update
      - TEACH rejected -> no learned rule event and no state update
      - TEACH accepted -> learned rule event exists and state contains learned_rules_active including learned rule_id
      - learned rule integrity: rule_id derivation + rule_sig
    """
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

        ok_sig, rsig, dsig = _verify_parse_payload_sig_v93(payload=dict(payload))
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
        # Verify rule_id derivation.
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
            if got_state_id != state_id_v93(got_sig):
                return False, "state_id_mismatch", {"index": int(i)}
            want_sig = state_sig_v93(s2)
            if want_sig != got_sig:
                return False, "state_sig_mismatch", {"index": int(i)}

            # learned_rules_active must be sorted unique when present.
            bindings = s.get("bindings") if isinstance(s.get("bindings"), dict) else {}
            lra = bindings.get("learned_rules_active")
            if lra is not None:
                if not isinstance(lra, list):
                    return False, "learned_rules_active_not_list", {"index": int(i)}
                lra2 = [str(x) for x in lra if isinstance(x, str) and x]
                if lra2 != sorted(set(lra2)):
                    return False, "learned_rules_active_not_sorted_unique", {"index": int(i)}

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
            # State must be updated for this teach turn and include learned rule id.
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
            # No state update and no learned event.
            if utid in learned_by_teacher:
                return False, "teach_reject_has_learned_event", {"turn_id": utid}
            if utid in state_user_turn_ids:
                return False, "teach_reject_has_state_update", {"turn_id": utid}

    return True, "ok", {"turns_total": int(max_idx + 1), "states_total": int(len(states)), "parses_total": int(len(parse_events)), "learned_total": int(len(learned_rule_events))}

