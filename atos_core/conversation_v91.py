from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex


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


def normalize_text_v91(text: str) -> str:
    # Deterministic normalization without changing meaning for logs.
    return str(text or "").replace("\r\n", "\n").strip()


def text_sig_v91(text: str) -> str:
    return sha256_hex(normalize_text_v91(text).encode("utf-8"))


def turn_id_v91(*, conversation_id: str, turn_index: int, role: str, text_sig: str) -> str:
    body = {
        "conversation_id": str(conversation_id),
        "turn_index": int(turn_index),
        "role": str(role),
        "text_sig": str(text_sig),
    }
    return f"turn_v91_{_stable_hash_obj(body)}"


@dataclass(frozen=True)
class TurnV91:
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
        text_norm = normalize_text_v91(self.text)
        sig = text_sig_v91(text_norm)
        tid = turn_id_v91(
            conversation_id=str(self.conversation_id),
            turn_index=int(self.turn_index),
            role=str(self.role),
            text_sig=str(sig),
        )
        return {
            "kind": "turn_v91",
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


def state_sig_v91(state_sem_sig: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(state_sem_sig).encode("utf-8"))


def state_id_v91(state_sig: str) -> str:
    return f"conversation_state_v91_{str(state_sig)}"


@dataclass(frozen=True)
class ConversationStateV91:
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
            "schema_version": 1,
            "kind": "conversation_state_v91",
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
            "invariants": {"schema_version": 1, "tail_k_fixed": True},
        }
        sig = state_sig_v91(sem)
        sid = state_id_v91(sig)
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


def append_chained_jsonl_v91(path: str, entry: Dict[str, Any], *, prev_hash: Optional[str]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    body = dict(entry)
    body["prev_hash"] = prev_hash
    entry_hash = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    body["entry_hash"] = entry_hash
    with open(path, "a", encoding="utf-8") as f:
        f.write(canonical_json_dumps(body))
        f.write("\n")
    return entry_hash


def verify_chained_jsonl_v91(path: str) -> bool:
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


def compute_transcript_hash_v91(turns: Sequence[Dict[str, Any]]) -> str:
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


def compute_state_chain_hash_v91(states: Sequence[Dict[str, Any]]) -> str:
    sigs: List[str] = []
    for s in states:
        if not isinstance(s, dict):
            continue
        sigs.append(str(s.get("state_sig") or ""))
    return sha256_hex(canonical_json_dumps(sigs).encode("utf-8"))


def compute_parse_chain_hash_v91(parse_events: Sequence[Dict[str, Any]]) -> str:
    sigs: List[str] = []
    for e in parse_events:
        if not isinstance(e, dict):
            continue
        payload = e.get("payload")
        if isinstance(payload, dict):
            sigs.append(str(payload.get("parse_sig") or ""))
    return sha256_hex(canonical_json_dumps(sigs).encode("utf-8"))


def verify_conversation_chain_v91(
    *,
    turns: Sequence[Dict[str, Any]],
    states: Sequence[Dict[str, Any]],
    parse_events: Sequence[Dict[str, Any]],
    trials: Sequence[Dict[str, Any]],
    tail_k: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Verify V91 invariants (fail-closed, deterministic):
      - turns turn_index contiguous 0..N-1; roles alternate user/assistant
      - every user turn has parse_sig/intent_id refs matching parse_events
      - states are hash-consistent, chained by prev_state_id, and tail_turn_ids are the last K turns
      - for COMM_ASK_CLARIFY / COMM_CONFIRM trials: no state update is created for that turn
    """
    # Turns.
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

    # Parse events: map by turn_id.
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

        # Verify parse_sig matches payload hash (excluding parse_sig itself).
        payload2 = dict(payload)
        got_sig = str(payload2.pop("parse_sig", "") or "")
        if not got_sig:
            return False, "missing_parse_sig", {"turn_id": tid}
        want_sig = sha256_hex(canonical_json_dumps(payload2).encode("utf-8"))
        if want_sig != got_sig:
            return False, "parse_sig_mismatch", {"turn_id": tid, "want": want_sig, "got": got_sig}

    parse_order.sort(key=lambda x: (int(x[0]), str(x[1])))
    # Parse events must exist for every user turn (even indices).
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

    # Ensure parse_events are in increasing order by turn_index (user turns only).
    parse_turn_indices = [int(ix) for ix, _tid in parse_order]
    if parse_turn_indices != sorted(parse_turn_indices):
        return False, "parse_events_not_sorted", {}
    for ix in parse_turn_indices:
        if ix % 2 != 0:
            return False, "parse_event_on_non_user_turn", {"turn_index": int(ix)}

    # States: chain + tail correctness.
    if states:
        prev_state_id = ""
        prev_created_step = -1
        for i, s in enumerate(states):
            if not isinstance(s, dict):
                return False, "state_not_dict", {"index": int(i)}
            raw_state_index = s.get("state_index", None)
            try:
                got_state_index = int(raw_state_index) if raw_state_index is not None else -1
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

            raw_created_step = s.get("created_step", None)
            raw_last_step = s.get("last_step", None)
            try:
                created_step = int(raw_created_step) if raw_created_step is not None else -1
            except Exception:
                created_step = -1
            try:
                last_step = int(raw_last_step) if raw_last_step is not None else -1
            except Exception:
                last_step = -1
            if created_step < 0 or last_step < 0:
                return False, "bad_step_fields", {"index": int(i)}
            if created_step < prev_created_step:
                return False, "created_step_not_monotonic", {"index": int(i)}
            if last_step < created_step:
                return False, "last_step_before_created_step", {"index": int(i)}

            # Verify state_sig.
            s2 = dict(s)
            got_sig = str(s2.pop("state_sig", "") or "")
            got_state_id = str(s2.pop("state_id", "") or "")
            if not got_sig:
                return False, "missing_state_sig", {"index": int(i)}
            if got_state_id != state_id_v91(got_sig):
                return False, "state_id_mismatch", {"index": int(i)}
            want_sig = state_sig_v91(s2)
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
            except Exception:
                tu_idx = -1
            try:
                ta_idx = int(ta.get("turn_index", -1))
            except Exception:
                ta_idx = -1
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

    # State update suppression for clarifications/confirmations.
    clarifying = {"COMM_ASK_CLARIFY", "COMM_CONFIRM"}
    state_turn_ids = {str(s.get("last_user_turn_id") or "") for s in states if isinstance(s, dict)}
    for tr in trials:
        if not isinstance(tr, dict):
            continue
        okind = str(tr.get("objective_kind") or "")
        if okind not in clarifying:
            continue
        utid = str(tr.get("user_turn_id") or "")
        if utid and utid in state_turn_ids:
            return False, "state_updated_on_clarification", {"user_turn_id": utid, "objective_kind": okind}

    return True, "ok", {"turns_total": int(max_idx + 1), "states_total": int(len(states)), "parses_total": int(len(parse_events))}

