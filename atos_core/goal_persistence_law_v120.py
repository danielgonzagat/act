from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .act import canonical_json_dumps, sha256_hex
from .conversation_v96 import verify_chained_jsonl_v96
from .goal_ledger_v99 import verify_goal_event_sig_v99


FAIL_REASON_MISSING_GOAL_EVENTS_V120 = "missing_goal_events_v120"
FAIL_REASON_GOAL_EVENTS_CHAIN_INVALID_V120 = "goal_events_chain_invalid_v120"
FAIL_REASON_GOAL_EVENT_SIG_INVALID_V120 = "goal_event_sig_invalid_v120"
FAIL_REASON_GOAL_DONE_MISSING_ASSISTANT_TURN_V120 = "goal_done_missing_assistant_turn_v120"
FAIL_REASON_GOAL_DONE_MISSING_PROGRESS_PROOF_V120 = "goal_done_missing_progress_proof_v120"
FAIL_REASON_GOAL_DONE_BEFORE_HORIZON_V120 = "goal_done_before_horizon_v120"


@dataclass(frozen=True)
class GoalPersistenceResultV120:
    ok: bool
    reason: str
    details: Dict[str, Any]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _payload_view(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        payload = r.get("payload")
        if isinstance(payload, dict):
            out.append(dict(payload))
    return out


def _assistant_by_user_turn_index(conversation_turn_payloads: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    conversation_turns.jsonl turn_index is global across both roles.
    For a user turn at index i (even), the assistant reply (if present) is typically at i+1.
    """
    by_idx: Dict[int, Dict[str, Any]] = {}
    for p in conversation_turn_payloads:
        if not isinstance(p, dict):
            continue
        if str(p.get("role") or "") != "assistant":
            continue
        try:
            ti = int(p.get("turn_index", -1))
        except Exception:
            continue
        if ti >= 0 and ti not in by_idx:
            by_idx[int(ti)] = dict(p)
    return dict(by_idx)


_PROGRESS_RE = re.compile(r"(?P<step>\d+)\s*/\s*(?P<total>\d+)")


def _progress_proof_from_text(text: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Deterministic proof for a system-closed goal:
    expects a visible "<step>/<total>" marker in the assistant text.
    """
    m = _PROGRESS_RE.search(str(text or ""))
    if not m:
        return False, "missing_progress_marker", {}
    try:
        step = int(m.group("step"))
        total = int(m.group("total"))
    except Exception:
        return False, "bad_progress_marker", {}
    if total <= 0:
        return False, "bad_progress_total", {"step": int(step), "total": int(total)}
    return True, "ok", {"step": int(step), "total": int(total)}


def verify_goal_persistence_law_v120(*, run_dir: str, expected_autopilot_total_steps: Optional[int]) -> GoalPersistenceResultV120:
    """
    V120 goal persistence law (minimal, deterministic):
      - goal_events.jsonl must exist + be chain-valid (v96 chain).
      - every goal event must have a valid v99 event_sig/event_id.
      - any GOAL_DONE with cause_type=="system" must have a deterministic progress proof in the assistant text:
          * a "<step>/<total>" marker exists, and
          * step >= total, and
          * if expected_autopilot_total_steps is provided: total == expected.
    """
    rd = str(run_dir)
    goal_events_path = os.path.join(rd, "goal_events.jsonl")
    turns_path = os.path.join(rd, "conversation_turns.jsonl")

    if not os.path.exists(goal_events_path):
        return GoalPersistenceResultV120(ok=False, reason=FAIL_REASON_MISSING_GOAL_EVENTS_V120, details={"goal_events_path": str(goal_events_path)})
    if not bool(verify_chained_jsonl_v96(str(goal_events_path))):
        return GoalPersistenceResultV120(ok=False, reason=FAIL_REASON_GOAL_EVENTS_CHAIN_INVALID_V120, details={"goal_events_path": str(goal_events_path)})

    rows = _read_jsonl(goal_events_path)
    payloads = _payload_view(rows)
    bad_sig: List[Dict[str, Any]] = []
    for ev in payloads:
        ok_sig, reason_sig, det_sig = verify_goal_event_sig_v99(dict(ev))
        if not bool(ok_sig):
            bad_sig.append({"reason": str(reason_sig), "details": dict(det_sig)})
    if bad_sig:
        return GoalPersistenceResultV120(ok=False, reason=FAIL_REASON_GOAL_EVENT_SIG_INVALID_V120, details={"bad_sig": list(bad_sig)})

    # Build assistant turn lookup.
    turn_rows = _read_jsonl(turns_path)
    turn_payloads = _payload_view(turn_rows)
    assistant_by_idx = _assistant_by_user_turn_index(turn_payloads)

    violations: List[Dict[str, Any]] = []
    done_checked: List[Dict[str, Any]] = []
    expected_total = int(expected_autopilot_total_steps) if expected_autopilot_total_steps is not None else None

    for ev in payloads:
        if str(ev.get("op") or "") != "GOAL_DONE":
            continue
        cause = ev.get("cause") if isinstance(ev.get("cause"), dict) else {}
        cause_type = str(cause.get("cause_type") or "")
        if cause_type != "system":
            continue
        try:
            ts_turn_index = int(ev.get("ts_turn_index", -1))
        except Exception:
            ts_turn_index = -1
        assistant_turn = assistant_by_idx.get(int(ts_turn_index + 1)) if ts_turn_index >= 0 else None
        if not isinstance(assistant_turn, dict):
            violations.append({"reason": FAIL_REASON_GOAL_DONE_MISSING_ASSISTANT_TURN_V120, "ts_turn_index": int(ts_turn_index)})
            continue
        ok_pf, pf_reason, pf_details = _progress_proof_from_text(str(assistant_turn.get("text") or ""))
        if not bool(ok_pf):
            violations.append(
                {
                    "reason": FAIL_REASON_GOAL_DONE_MISSING_PROGRESS_PROOF_V120,
                    "ts_turn_index": int(ts_turn_index),
                    "proof_reason": str(pf_reason),
                }
            )
            continue
        step = int(pf_details.get("step", 0) or 0)
        total = int(pf_details.get("total", 0) or 0)
        if int(step) < int(total):
            violations.append(
                {
                    "reason": FAIL_REASON_GOAL_DONE_BEFORE_HORIZON_V120,
                    "ts_turn_index": int(ts_turn_index),
                    "step": int(step),
                    "total": int(total),
                }
            )
            continue
        if expected_total is not None and int(total) != int(expected_total):
            violations.append(
                {
                    "reason": FAIL_REASON_GOAL_DONE_BEFORE_HORIZON_V120,
                    "ts_turn_index": int(ts_turn_index),
                    "step": int(step),
                    "total": int(total),
                    "expected_total": int(expected_total),
                }
            )
            continue
        done_checked.append({"ts_turn_index": int(ts_turn_index), "step": int(step), "total": int(total)})

    ok = len(violations) == 0
    reason = "ok" if ok else str(violations[0].get("reason") or "violations")
    details = {
        "schema_version": 120,
        "kind": "goal_persistence_law_v120_result",
        "ok": bool(ok),
        "reason": str(reason),
        "expected_autopilot_total_steps": int(expected_total) if expected_total is not None else None,
        "system_goal_done_checked": list(done_checked),
        "violations": list(violations),
    }
    return GoalPersistenceResultV120(ok=bool(ok), reason=str(reason), details=dict(details))


def write_goal_persistence_summary_v120(*, run_dir: str, expected_autopilot_total_steps: Optional[int]) -> Dict[str, Any]:
    """
    Write-once summary artifact: goal_persistence_summary_v120.json (deterministic).
    """
    res = verify_goal_persistence_law_v120(run_dir=str(run_dir), expected_autopilot_total_steps=expected_autopilot_total_steps)
    body = dict(res.details)
    body["summary_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    path = os.path.join(str(run_dir), "goal_persistence_summary_v120.json")
    if os.path.exists(path):
        raise ValueError(f"worm_exists:{path}")
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        raise ValueError(f"tmp_exists:{tmp}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)
    return dict(body)
