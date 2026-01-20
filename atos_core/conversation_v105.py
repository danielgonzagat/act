from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .agency_ledger_v105 import (
    agency_registry_snapshot_v105,
    compute_agency_chain_hash_v105,
    lookup_agency_event_v105,
    render_agency_text_v105,
    render_explain_agency_text_v105,
    render_trace_agency_text_v105,
    verify_agency_event_sig_v105,
)
from .conversation_v104 import verify_conversation_chain_v104
from .intent_grammar_v105 import INTENT_AGENCY_V105, INTENT_EXPLAIN_AGENCY_V105, INTENT_TRACE_AGENCY_V105


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _assistant_text_after_user_turn(*, turns: Sequence[Dict[str, Any]], user_turn_index: int) -> str:
    want_idx = int(user_turn_index + 1)
    for t in turns:
        if not isinstance(t, dict):
            continue
        if int(t.get("turn_index") or -1) == want_idx and str(t.get("role") or "") == "assistant":
            return str(t.get("text") or "")
    return ""


def _agency_events_prefix(*, agency_events: Sequence[Dict[str, Any]], until_ts_turn_index_exclusive: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ev in agency_events:
        if not isinstance(ev, dict):
            continue
        try:
            ti = int(ev.get("ts_turn_index") or 0)
        except Exception:
            ti = 0
        if ti >= int(until_ts_turn_index_exclusive):
            break
        out.append(dict(ev))
    return list(out)


def verify_conversation_chain_v105(
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
    style_events: Sequence[Dict[str, Any]],
    template_snapshot: Dict[str, Any],
    concept_events: Sequence[Dict[str, Any]],
    concept_snapshot: Dict[str, Any],
    plan_events: Sequence[Dict[str, Any]],
    plan_snapshot: Dict[str, Any],
    agency_events: Sequence[Dict[str, Any]],
    agency_snapshot: Dict[str, Any],
    tail_k: int,
    repo_root: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    ok0, r0, d0 = verify_conversation_chain_v104(
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
        binding_events=binding_events,
        binding_snapshot=dict(binding_snapshot),
        style_events=style_events,
        template_snapshot=dict(template_snapshot),
        concept_events=concept_events,
        concept_snapshot=dict(concept_snapshot),
        plan_events=plan_events,
        plan_snapshot=dict(plan_snapshot),
        tail_k=int(tail_k),
        repo_root=str(repo_root),
    )
    if not ok0:
        return False, str(r0), dict(d0)

    # Verify agency event sig chain.
    prev_sig = ""
    for i, ev in enumerate(list(agency_events)):
        if not isinstance(ev, dict):
            return False, "agency_event_not_dict", {"index": int(i)}
        ok_ev, rr, dd = verify_agency_event_sig_v105(dict(ev))
        if not ok_ev:
            return False, str(rr), dict(dd)
        if str(ev.get("prev_event_sig") or "") != str(prev_sig):
            return False, "agency_prev_event_sig_mismatch", {"index": int(i)}
        prev_sig = str(ev.get("event_sig") or "")

    # Verify derived snapshot matches replay.
    want_snap = agency_registry_snapshot_v105(list(agency_events))
    want_sem = dict(want_snap)
    got_sem = dict(agency_snapshot) if isinstance(agency_snapshot, dict) else {}
    if _stable_hash_obj(want_sem) != _stable_hash_obj(got_sem):
        return False, "agency_snapshot_mismatch", {"want_sig": _stable_hash_obj(want_sem), "got_sig": _stable_hash_obj(got_sem)}

    # Verify deterministic renderers for agency/explain_agency/trace_agency.
    for pe in list(parse_events):
        if not isinstance(pe, dict):
            continue
        tid = str(pe.get("turn_id") or "")
        idx = int(pe.get("turn_index") or 0)
        payload = pe.get("payload") if isinstance(pe.get("payload"), dict) else {}
        intent_id = str(payload.get("intent_id") or "")
        if intent_id not in {INTENT_AGENCY_V105, INTENT_EXPLAIN_AGENCY_V105, INTENT_TRACE_AGENCY_V105}:
            continue
        assistant_text = _assistant_text_after_user_turn(turns=list(turns), user_turn_index=int(idx))
        if intent_id == INTENT_AGENCY_V105:
            prefix = _agency_events_prefix(agency_events=list(agency_events), until_ts_turn_index_exclusive=int(idx))
            snap = agency_registry_snapshot_v105(list(prefix))
            want_text = render_agency_text_v105(dict(snap))
            if str(assistant_text) != str(want_text):
                return False, "agency_render_mismatch", {"turn_id": str(tid), "turn_index": int(idx)}
        elif intent_id == INTENT_EXPLAIN_AGENCY_V105:
            q = str(payload.get("query") or "")
            try:
                qidx = int(q)
            except Exception:
                qidx = -1
            ev = lookup_agency_event_v105(agency_events=list(agency_events), user_turn_index=int(qidx))
            want_text = render_explain_agency_text_v105(dict(ev) if isinstance(ev, dict) else {})
            if str(assistant_text) != str(want_text):
                return False, "explain_agency_render_mismatch", {"turn_id": str(tid), "turn_index": int(idx)}
        elif intent_id == INTENT_TRACE_AGENCY_V105:
            q = str(payload.get("query") or "")
            try:
                qidx = int(q)
            except Exception:
                qidx = -1
            want_text = render_trace_agency_text_v105(agency_events=list(agency_events), until_turn_index=int(qidx))
            if str(assistant_text) != str(want_text):
                return False, "trace_agency_render_mismatch", {"turn_id": str(tid), "turn_index": int(idx)}

    return True, "ok", {"agency_chain_hash_v105": compute_agency_chain_hash_v105(list(agency_events))}

