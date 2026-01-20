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


def normalize_text_v90(text: str) -> str:
    # Deterministic normalization without changing meaning for our DSL harness.
    return str(text or "").replace("\r\n", "\n").strip()


def text_sig_v90(text: str) -> str:
    return sha256_hex(normalize_text_v90(text).encode("utf-8"))


def turn_id_v90(*, conversation_id: str, turn_index: int, role: str, text_sig: str) -> str:
    body = {
        "conversation_id": str(conversation_id),
        "turn_index": int(turn_index),
        "role": str(role),
        "text_sig": str(text_sig),
    }
    return f"turn_v90_{_stable_hash_obj(body)}"


@dataclass(frozen=True)
class TurnV90:
    conversation_id: str
    turn_index: int
    role: str  # "user" | "assistant"
    text: str
    created_step: int
    offset_us: int = 0
    objective_id: str = ""
    action_concept_id: str = ""
    eval_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        text_norm = normalize_text_v90(self.text)
        sig = text_sig_v90(text_norm)
        tid = turn_id_v90(
            conversation_id=str(self.conversation_id),
            turn_index=int(self.turn_index),
            role=str(self.role),
            text_sig=str(sig),
        )
        return {
            "kind": "turn_v90",
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
                "action_concept_id": str(self.action_concept_id or ""),
                "eval_id": str(self.eval_id or ""),
            },
        }


def _canon_bindings(bindings: Any) -> Dict[str, Any]:
    if not isinstance(bindings, dict):
        return {}
    # Deterministic: sort keys, deep-copy JSON-safe leaves.
    out: Dict[str, Any] = {}
    for k in sorted(bindings.keys(), key=str):
        key = str(k)
        v = bindings.get(k)
        out[key] = _safe_deepcopy(v)
    return out


def _canon_str_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for x in items:
        if isinstance(x, str) and x:
            out.append(x)
    # unique+sorted
    return sorted(set(out))


def state_sig_v90(state_sem_sig: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(state_sem_sig).encode("utf-8"))


def state_id_v90(state_sig: str) -> str:
    return f"conversation_state_v90_{str(state_sig)}"


@dataclass(frozen=True)
class ConversationStateV90:
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
            "conversation_id": str(self.conversation_id),
            "state_index": int(self.state_index),
            "prev_state_id": str(self.prev_state_id or ""),
            "active_goals": _canon_str_list(self.active_goals),
            "bindings": _canon_bindings(self.bindings),
            "tail_turn_ids": list(self.tail_turn_ids),
            "last_user_turn_id": str(self.last_user_turn_id or ""),
            "last_assistant_turn_id": str(self.last_assistant_turn_id or ""),
            "created_step": int(self.created_step),
            "last_step": int(self.last_step),
            "invariants": {"schema_version": 1, "tail_k_fixed": True},
        }
        sig = state_sig_v90(sem)
        sid = state_id_v90(sig)
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


def append_chained_jsonl_v90(path: str, entry: Dict[str, Any], *, prev_hash: Optional[str]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    body = dict(entry)
    body["prev_hash"] = prev_hash
    entry_hash = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    body["entry_hash"] = entry_hash
    with open(path, "a", encoding="utf-8") as f:
        f.write(canonical_json_dumps(body))
        f.write("\n")
    return entry_hash


def verify_chained_jsonl_v90(path: str) -> bool:
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


def compute_transcript_hash_v90(turns: Sequence[Dict[str, Any]]) -> str:
    # Deterministic: sort by turn_index.
    items: List[Dict[str, Any]] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        items.append({"turn_index": int(t.get("turn_index", 0) or 0), "role": str(t.get("role") or ""), "text": str(t.get("text") or "")})
    items.sort(key=lambda r: int(r.get("turn_index", 0) or 0))
    view = [{"role": str(r.get("role") or ""), "text": str(r.get("text") or "")} for r in items]
    return sha256_hex(canonical_json_dumps(view).encode("utf-8"))


def compute_state_chain_hash_v90(states: Sequence[Dict[str, Any]]) -> str:
    sigs: List[str] = []
    for s in states:
        if not isinstance(s, dict):
            continue
        sigs.append(str(s.get("state_sig") or ""))
    return sha256_hex(canonical_json_dumps(sigs).encode("utf-8"))


def verify_conversation_chain_v90(
    *,
    turns: Sequence[Dict[str, Any]],
    states: Sequence[Dict[str, Any]],
    tail_k: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Verify state-chain invariants (fail-closed, deterministic).
    Returns (ok, reason, details).
    """
    tmap: Dict[str, Dict[str, Any]] = {}
    by_index: Dict[int, Dict[str, Any]] = {}
    for t in turns:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("turn_id") or "")
        if tid:
            tmap[tid] = dict(t)
        try:
            idx = int(t.get("turn_index", 0) or 0)
            by_index[idx] = dict(t)
        except Exception:
            pass

    if not states:
        return False, "missing_states", {}

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
        if int(got_state_index) != int(i):
            return False, "state_index_not_incrementing", {"index": int(i), "got": s.get("state_index")}
        if i == 0:
            if str(s.get("prev_state_id") or "") not in ("", "None"):
                return False, "genesis_prev_state_not_empty", {"got": s.get("prev_state_id")}
        else:
            if str(s.get("prev_state_id") or "") != str(prev_state_id):
                return False, "prev_state_id_mismatch", {"index": int(i), "want": str(prev_state_id), "got": s.get("prev_state_id")}

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
            return False, "bad_step_fields", {"index": int(i), "created_step": created_step, "last_step": last_step}
        if created_step < prev_created_step:
            return False, "created_step_not_monotonic", {"index": int(i), "prev": prev_created_step, "got": created_step}
        if last_step < created_step:
            return False, "last_step_before_created_step", {"index": int(i), "created_step": created_step, "last_step": last_step}

        # Verify state_sig.
        s2 = dict(s)
        got_sig = str(s2.pop("state_sig", "") or "")
        got_state_id = str(s2.pop("state_id", "") or "")
        if not got_sig:
            return False, "missing_state_sig", {"index": int(i)}
        if got_state_id != state_id_v90(got_sig):
            return False, "state_id_mismatch", {"index": int(i), "want": state_id_v90(got_sig), "got": got_state_id}
        want_sig = state_sig_v90(s2)
        if want_sig != got_sig:
            return False, "state_sig_mismatch", {"index": int(i), "want": want_sig, "got": got_sig}

        # Verify tail_turn_ids correspond to last K turns up to assistant turn at 2*i+1.
        last_user_tid = str(s.get("last_user_turn_id") or "")
        last_asst_tid = str(s.get("last_assistant_turn_id") or "")
        if last_user_tid not in tmap or last_asst_tid not in tmap:
            return False, "missing_last_turn_refs", {"index": int(i), "last_user_turn_id": last_user_tid, "last_assistant_turn_id": last_asst_tid}
        tu = tmap[last_user_tid]
        ta = tmap[last_asst_tid]
        if str(tu.get("role") or "") != "user" or str(ta.get("role") or "") != "assistant":
            return False, "last_turn_roles_wrong", {"index": int(i)}
        raw_tu_idx = tu.get("turn_index", None)
        raw_ta_idx = ta.get("turn_index", None)
        try:
            tu_idx = int(raw_tu_idx) if raw_tu_idx is not None else -1
        except Exception:
            tu_idx = -1
        try:
            ta_idx = int(raw_ta_idx) if raw_ta_idx is not None else -1
        except Exception:
            ta_idx = -1
        if int(tu_idx) != 2 * int(i):
            return False, "last_user_turn_index_wrong", {"index": int(i), "got": tu.get("turn_index")}
        if int(ta_idx) != 2 * int(i) + 1:
            return False, "last_assistant_turn_index_wrong", {"index": int(i), "got": ta.get("turn_index")}

        tail = s.get("tail_turn_ids")
        tail = tail if isinstance(tail, list) else []
        tail2 = [str(x) for x in tail if isinstance(x, str) and x]
        if len(tail2) > int(tail_k):
            return False, "tail_too_long", {"index": int(i), "len": int(len(tail2)), "tail_k": int(tail_k)}
        for tid in tail2:
            if tid not in tmap:
                return False, "tail_turn_missing", {"index": int(i), "turn_id": tid}

        end_idx = 2 * int(i) + 1
        start_idx = max(0, end_idx - (int(tail_k) - 1))
        expected_tail: List[str] = []
        for j in range(start_idx, end_idx + 1):
            tj = by_index.get(int(j))
            if tj is None:
                return False, "missing_turn_index", {"index": int(i), "missing_turn_index": int(j)}
            expected_tail.append(str(tj.get("turn_id") or ""))
        if tail2 != expected_tail:
            return False, "tail_mismatch", {"index": int(i), "want": expected_tail, "got": tail2}

        prev_state_id = str(got_state_id)
        prev_created_step = int(created_step)

    return True, "ok", {"states_total": int(len(states)), "turns_total": int(len(turns))}
