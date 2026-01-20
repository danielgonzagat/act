from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .conversation_v107 import verify_conversation_chain_v107
from .flow_engine_v108 import FLOW_THRESH_V108
from .flow_ledger_v108 import (
    FLOW_REPAIR_CLARIFY_REFERENCE_V108,
    FLOW_REPAIR_SUMMARIZE_CONFIRM_V108,
    compute_flow_chain_hash_v108,
    flow_registry_snapshot_v108,
    render_explain_flow_text_v108,
    render_flow_text_v108,
    render_trace_flow_text_v108,
    verify_flow_event_sig_v108,
)
from .intent_grammar_v108 import INTENT_EXPLAIN_FLOW_V108, INTENT_FLOW_V108, INTENT_TRACE_FLOW_V108


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


def _lookup_agency_event(*, agency_events: Sequence[Dict[str, Any]], user_turn_index: int) -> Optional[Dict[str, Any]]:
    for ev in agency_events:
        if not isinstance(ev, dict):
            continue
        if int(ev.get("user_turn_index") or -1) == int(user_turn_index):
            return dict(ev)
    return None


def verify_conversation_chain_v108(
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
    dialogue_events: Sequence[Dict[str, Any]],
    dialogue_snapshot: Dict[str, Any],
    pragmatics_events: Sequence[Dict[str, Any]],
    pragmatics_snapshot: Dict[str, Any],
    flow_events: Sequence[Dict[str, Any]],
    flow_snapshot: Dict[str, Any],
    tail_k: int,
    repo_root: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    ok0, r0, d0 = verify_conversation_chain_v107(
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
        agency_events=agency_events,
        agency_snapshot=dict(agency_snapshot),
        dialogue_events=dialogue_events,
        dialogue_snapshot=dict(dialogue_snapshot),
        pragmatics_events=pragmatics_events,
        pragmatics_snapshot=dict(pragmatics_snapshot),
        tail_k=int(tail_k),
        repo_root=str(repo_root),
    )
    if not ok0:
        return False, str(r0), dict(d0)

    # Verify flow inner sig chain.
    prev_sig = ""
    for i, ev in enumerate(list(flow_events)):
        if not isinstance(ev, dict):
            return False, "flow_event_not_dict", {"index": int(i)}
        ok_ev, rr, dd = verify_flow_event_sig_v108(dict(ev))
        if not ok_ev:
            return False, str(rr), dict(dd)
        if str(ev.get("prev_event_sig") or "") != str(prev_sig):
            return False, "flow_prev_event_sig_mismatch", {"index": int(i)}
        prev_sig = str(ev.get("event_sig") or "")

    # Verify derived snapshot matches replay (including last flow_memory when present).
    last_mem: Dict[str, Any] = {}
    if flow_events:
        last = flow_events[-1]
        fm = last.get("flow_memory_v108") if isinstance(last.get("flow_memory_v108"), dict) else {}
        last_mem = dict(fm)
    want_snap = flow_registry_snapshot_v108(list(flow_events), flow_memory_last=dict(last_mem) if last_mem else None)
    want_sem = dict(want_snap)
    got_sem = dict(flow_snapshot) if isinstance(flow_snapshot, dict) else {}
    if _stable_hash_obj(want_sem) != _stable_hash_obj(got_sem):
        return False, "flow_snapshot_mismatch", {"want_sig": _stable_hash_obj(want_sem), "got_sig": _stable_hash_obj(got_sem)}

    # Survival S4/S5: critical flow requires repair and blocks progress.
    plan_turns = set([int(pe.get("user_turn_index") or pe.get("ts_turn_index") or -1) for pe in plan_events if isinstance(pe, dict)])
    goal_turns = set([int(ge.get("ts_turn_index") or -1) for ge in goal_events if isinstance(ge, dict)])
    low_streak = 0
    for ev in list(flow_events):
        if not isinstance(ev, dict):
            continue
        uidx = int(ev.get("user_turn_index") or ev.get("turn_index") or 0)
        score = int(ev.get("flow_score_v108") or 0)
        flags = ev.get("flow_flags_v108") if isinstance(ev.get("flow_flags_v108"), dict) else {}
        repair = str(ev.get("repair_action_v108") or "")
        progress_allowed = bool(ev.get("progress_allowed_v108", False))
        critical = bool(
            score < int(FLOW_THRESH_V108)
            or bool(flags.get("ABRUPT_TOPIC_SHIFT", False))
            or bool(flags.get("REPETITION_LOOP", False))
            or bool(flags.get("TECHNICAL_MISMATCH", False))
            or bool(flags.get("UNRESOLVED_REFERENCE", False))
            or bool(flags.get("PENDING_QUESTION_AGED", False))
        )
        if critical:
            if progress_allowed:
                return False, "flow_progress_allowed_but_critical", {"user_turn_index": int(uidx)}
            if not repair:
                return False, "flow_survival_missing_repair", {"user_turn_index": int(uidx)}
            # S5 specialization
            if bool(flags.get("UNRESOLVED_REFERENCE", False)) and repair != FLOW_REPAIR_CLARIFY_REFERENCE_V108:
                return False, "flow_survival_wrong_repair_for_unresolved_reference", {"user_turn_index": int(uidx), "repair": str(repair)}
            if bool(flags.get("PENDING_QUESTION_AGED", False)) and repair != FLOW_REPAIR_SUMMARIZE_CONFIRM_V108:
                return False, "flow_survival_wrong_repair_for_pending_aged", {"user_turn_index": int(uidx), "repair": str(repair)}
            # Progress gating: no plan/goal and no agency progress actions.
            if int(uidx) in plan_turns:
                return False, "flow_progress_blocked_but_plan_event", {"user_turn_index": int(uidx)}
            if int(uidx) in goal_turns:
                return False, "flow_progress_blocked_but_goal_event", {"user_turn_index": int(uidx)}
            ae = _lookup_agency_event(agency_events=list(agency_events), user_turn_index=int(uidx))
            if isinstance(ae, dict):
                ck = str(ae.get("chosen_kind") or "")
                if ck in {"EXECUTE_STEP", "CLOSE_GOAL"}:
                    return False, "flow_progress_blocked_but_agency_progress", {"user_turn_index": int(uidx), "chosen_kind": str(ck)}

        if score < int(FLOW_THRESH_V108):
            low_streak += 1
            if low_streak >= 2 and not repair:
                return False, "flow_survival_missing_repair_streak", {"user_turn_index": int(uidx)}
        else:
            low_streak = 0

    # Verify deterministic renderers for flow/explain_flow/trace_flow.
    for pe in list(parse_events):
        if not isinstance(pe, dict):
            continue
        tid = str(pe.get("turn_id") or "")
        idx = int(pe.get("turn_index") or 0)
        payload = pe.get("payload") if isinstance(pe.get("payload"), dict) else {}
        intent_id = str(payload.get("intent_id") or "")
        if intent_id not in {INTENT_FLOW_V108, INTENT_EXPLAIN_FLOW_V108, INTENT_TRACE_FLOW_V108}:
            continue
        assistant_text = _assistant_text_after_user_turn(turns=list(turns), user_turn_index=int(idx))
        if intent_id == INTENT_FLOW_V108:
            want_text = render_flow_text_v108(flow_events=list(flow_events), until_user_turn_index_exclusive=int(idx))
            if str(assistant_text) != str(want_text):
                return False, "flow_render_mismatch", {"turn_id": str(tid), "turn_index": int(idx)}
        elif intent_id == INTENT_EXPLAIN_FLOW_V108:
            q = str(payload.get("query") or "")
            try:
                qidx = int(q)
            except Exception:
                qidx = -1
            want_text = render_explain_flow_text_v108(flow_events=list(flow_events), user_turn_index=int(qidx))
            if str(assistant_text) != str(want_text):
                return False, "explain_flow_render_mismatch", {"turn_id": str(tid), "turn_index": int(idx)}
        elif intent_id == INTENT_TRACE_FLOW_V108:
            q = str(payload.get("query") or "")
            try:
                qidx = int(q)
            except Exception:
                qidx = -1
            want_text = render_trace_flow_text_v108(flow_events=list(flow_events), until_user_turn_index=int(qidx))
            if str(assistant_text) != str(want_text):
                return False, "trace_flow_render_mismatch", {"turn_id": str(tid), "turn_index": int(idx)}

    dd = dict(d0)
    dd["flow_chain_hash_v108"] = str(compute_flow_chain_hash_v108(list(flow_events)))
    return True, "ok", dd

