from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps
from .binding_ledger_v101 import (
    binding_chain_hash_v101,
    binding_snapshot_v101,
    verify_binding_event_sig_v101,
)
from .conversation_v100 import no_hybridization_check_v100, verify_conversation_chain_v100


def _canon_str_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for x in items:
        if isinstance(x, str) and x:
            out.append(str(x))
    return sorted(set(out))


def render_bindings_text_v101(bindings_active: Sequence[Dict[str, Any]]) -> str:
    items: List[Dict[str, Any]] = []
    for b in bindings_active:
        if not isinstance(b, dict):
            continue
        bid = str(b.get("binding_id") or "")
        bk = str(b.get("binding_kind") or "")
        if not bid or not bk:
            continue
        items.append(dict(b))
    items.sort(key=lambda b: (-int(b.get("last_used_turn_index") or 0), str(b.get("binding_kind") or ""), str(b.get("binding_id") or "")))
    if not items:
        return "BINDINGS: (empty)"
    lines: List[str] = ["BINDINGS:"]
    for i, b in enumerate(items, start=1):
        preview = str(b.get("value_preview") or "")
        lines.append(
            f"{i}) id={str(b.get('binding_id') or '')} kind={str(b.get('binding_kind') or '')} use_count={int(b.get('use_count') or 0)} last_used_turn={int(b.get('last_used_turn_index') or 0)} preview={json.dumps(preview, ensure_ascii=False)}"
        )
    return "\n".join(lines)


def render_explain_binding_text_v101(binding: Dict[str, Any]) -> str:
    if not isinstance(binding, dict):
        return "BINDING: (invalid)"
    bid = str(binding.get("binding_id") or "")
    bk = str(binding.get("binding_kind") or "")
    value = binding.get("value") if isinstance(binding.get("value"), dict) else {}
    prov = binding.get("provenance") if isinstance(binding.get("provenance"), dict) else {}
    lines = [
        f"BINDING: {bid}",
        f"KIND={bk}",
        f"VALUE={canonical_json_dumps(value)}",
        f"VALUE_HASH={str(binding.get('value_hash') or '')}",
        f"PROVENANCE={canonical_json_dumps(prov)}",
        f"CREATED_TURN_INDEX={int(binding.get('created_turn_index') or 0)}",
        f"LAST_USED_TURN_INDEX={int(binding.get('last_used_turn_index') or 0)}",
        f"USE_COUNT={int(binding.get('use_count') or 0)}",
    ]
    return "\n".join(lines)


def render_trace_ref_text_v101(*, turn_id: str, binding_events: Sequence[Dict[str, Any]]) -> str:
    tid = str(turn_id or "")
    evs: List[Dict[str, Any]] = []
    for ev in binding_events:
        if not isinstance(ev, dict):
            continue
        if str(ev.get("turn_id") or "") != tid:
            continue
        et = str(ev.get("type") or "")
        if et not in {"BIND_RESOLVE", "BIND_AMBIGUOUS", "BIND_MISS"}:
            continue
        evs.append(dict(ev))
    evs.sort(key=lambda e: (int(e.get("ts_logical") or 0), str(e.get("event_id") or "")))
    if not evs:
        return f"TRACE_REF: {tid} (empty)"
    parts: List[str] = [f"TRACE_REF: {tid}"]
    for i, ev in enumerate(evs, start=1):
        res = ev.get("resolution") if isinstance(ev.get("resolution"), dict) else {}
        pron = str(res.get("pronoun") or "")
        chosen = str(res.get("chosen_binding_id") or ev.get("binding_id") or "")
        status = str(res.get("status") or "")
        parts.append(f"{i}) type={str(ev.get('type') or '')} pronoun={pron} status={status} chosen={chosen}")
    return "\n".join(parts)


def verify_conversation_chain_v101(
    *,
    turns: Sequence[Dict[str, Any]],
    states: Sequence[Dict[str, Any]],
    parse_events: Sequence[Dict[str, Any]],
    trials: Sequence[Dict[str, Any]],
    learned_rule_events: Sequence[Dict[str, Any]],
    action_plans: Sequence[Dict[str, Any]],
    memory_events: Sequence[Dict[str, Any]],
    belief_events: Sequence[Dict[str, Any]],
    evidence_events: Sequence[Dict[str, Any]],
    goal_events: Sequence[Dict[str, Any]],
    goal_snapshot: Dict[str, Any],
    discourse_events: Sequence[Dict[str, Any]],
    fragment_events: Sequence[Dict[str, Any]],
    binding_events: Sequence[Dict[str, Any]],
    binding_snapshot: Dict[str, Any],
    tail_k: int,
    repo_root: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    # Base invariants up to V100 (includes no-hybridization guard).
    ok0, reason0, details0 = verify_conversation_chain_v100(
        turns=turns,
        states=states,
        parse_events=parse_events,
        trials=trials,
        learned_rule_events=learned_rule_events,
        action_plans=action_plans,
        memory_events=memory_events,
        belief_events=belief_events,
        evidence_events=evidence_events,
        goal_events=goal_events,
        goal_snapshot=dict(goal_snapshot),
        discourse_events=discourse_events,
        fragment_events=fragment_events,
        tail_k=int(tail_k),
        repo_root=str(repo_root),
    )
    if not ok0:
        return False, str(reason0), dict(details0)

    # Redundant but explicit for V101: no-hybridization fail-closed.
    ok_h, reason_h, details_h = no_hybridization_check_v100(repo_root=str(repo_root))
    if not ok_h:
        return False, str(reason_h), dict(details_h)

    # Verify binding events sig-chain (inner) and basic invariants.
    prev_sig = ""
    prev_step = -1
    seen_binding_ids: Dict[str, bool] = {}
    binding_events_list: List[Dict[str, Any]] = []
    for i, ev in enumerate(list(binding_events)):
        if not isinstance(ev, dict):
            return False, "binding_event_not_dict", {"index": int(i)}
        ok_ev, breason, bdetails = verify_binding_event_sig_v101(dict(ev))
        if not ok_ev:
            return False, str(breason), dict(bdetails)
        if str(ev.get("prev_event_sig") or "") != str(prev_sig):
            return False, "binding_prev_event_sig_mismatch", {"index": int(i)}
        prev_sig = str(ev.get("event_sig") or "")
        try:
            cstep = int(ev.get("created_step", -1))
        except Exception:
            cstep = -1
        if cstep < 0:
            return False, "binding_event_bad_created_step", {"event_id": str(ev.get("event_id") or "")}
        if cstep < prev_step:
            return False, "binding_events_not_monotonic", {"event_id": str(ev.get("event_id") or "")}
        prev_step = int(cstep)

        et = str(ev.get("type") or "")
        bid = str(ev.get("binding_id") or "")
        if et == "BIND_CREATE":
            if not bid:
                return False, "binding_create_missing_binding_id", {"event_id": str(ev.get("event_id") or "")}
            seen_binding_ids[bid] = True
        elif et in {"BIND_RESOLVE", "BIND_PRUNE"}:
            if bid and bid not in seen_binding_ids:
                return False, "binding_unknown_id", {"binding_id": bid, "type": et}

        binding_events_list.append(dict(ev))

    # Snapshot must match fold/replay.
    expected_snapshot = binding_snapshot_v101(binding_events_list)
    if not isinstance(binding_snapshot, dict):
        return False, "binding_snapshot_not_dict", {}
    if canonical_json_dumps(dict(binding_snapshot)) != canonical_json_dumps(dict(expected_snapshot)):
        return False, "binding_snapshot_mismatch", {"want": str(expected_snapshot.get("snapshot_sig") or ""), "got": str(binding_snapshot.get("snapshot_sig") or "")}

    return True, "ok", {"binding_chain_hash": binding_chain_hash_v101(binding_events_list)}
