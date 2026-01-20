from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .conversation_v108 import verify_conversation_chain_v108
from .intent_grammar_v109 import INTENT_EXPLAIN_SEMANTICS_V109, INTENT_SEMANTICS_V109, INTENT_TRACE_SEMANTICS_V109
from .semantics_engine_v109 import (
    FLAG_CONCEPT_MATCH_TOO_WEAK_V109,
    FLAG_CONTRADICTION_UNREPAIRED_V109,
    FLAG_REQUIRES_CLARIFICATION_V109,
    FLAG_TAUGHT_CONCEPT_NOT_REUSED_V109,
    SEMANTIC_THRESH_V109,
)
from .semantics_ledger_v109 import (
    compute_semantic_chain_hash_v109,
    render_explain_semantics_text_v109,
    render_semantics_text_v109,
    render_trace_semantics_text_v109,
    semantic_registry_snapshot_v109,
    verify_semantic_event_sig_v109,
)


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


def verify_conversation_chain_v109(
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
    semantic_events: Sequence[Dict[str, Any]],
    semantic_snapshot: Dict[str, Any],
    tail_k: int,
    repo_root: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    ok0, r0, d0 = verify_conversation_chain_v108(
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
        flow_events=flow_events,
        flow_snapshot=dict(flow_snapshot),
        tail_k=int(tail_k),
        repo_root=str(repo_root),
    )
    if not ok0:
        return False, str(r0), dict(d0)

    # Verify semantic inner sig chain.
    prev_sig = ""
    for i, ev in enumerate(list(semantic_events)):
        if not isinstance(ev, dict):
            return False, "semantic_event_not_dict", {"index": int(i)}
        ok_ev, rr, dd = verify_semantic_event_sig_v109(dict(ev))
        if not ok_ev:
            return False, str(rr), dict(dd)
        if str(ev.get("prev_event_sig") or "") != str(prev_sig):
            return False, "semantic_prev_event_sig_mismatch", {"index": int(i)}
        prev_sig = str(ev.get("event_sig") or "")

    # Verify derived snapshot matches replay.
    want_snap = semantic_registry_snapshot_v109(list(semantic_events))
    want_sem = dict(want_snap)
    got_sem = dict(semantic_snapshot) if isinstance(semantic_snapshot, dict) else {}
    if _stable_hash_obj(want_sem) != _stable_hash_obj(got_sem):
        return False, "semantic_snapshot_mismatch", {"want_sig": _stable_hash_obj(want_sem), "got_sig": _stable_hash_obj(got_sem)}

    # Survival S6/S7 (minimal): critical semantic flags must block progress and force repair.
    plan_turns = set([int(pe.get("user_turn_index") or pe.get("ts_turn_index") or -1) for pe in plan_events if isinstance(pe, dict)])
    goal_turns = set([int(ge.get("ts_turn_index") or -1) for ge in goal_events if isinstance(ge, dict)])
    critical_flags = {
        FLAG_TAUGHT_CONCEPT_NOT_REUSED_V109,
        FLAG_CONCEPT_MATCH_TOO_WEAK_V109,
        FLAG_CONTRADICTION_UNREPAIRED_V109,
        FLAG_REQUIRES_CLARIFICATION_V109,
    }
    for ev in list(semantic_events):
        if not isinstance(ev, dict):
            continue
        uidx = int(ev.get("user_turn_index") or ev.get("turn_index") or 0)
        score = int(ev.get("semantic_score_v109") or 0)
        flags = ev.get("flags_v109") if isinstance(ev.get("flags_v109"), dict) else {}
        repair = str(ev.get("repair_action_v109") or "")
        progress_allowed = bool(ev.get("progress_allowed_v109", False))
        critical = bool(score < int(SEMANTIC_THRESH_V109) or any(bool(flags.get(f, False)) for f in critical_flags))
        if critical:
            if progress_allowed:
                return False, "semantic_progress_allowed_but_critical", {"user_turn_index": int(uidx)}
            if not repair:
                return False, "semantic_survival_missing_repair", {"user_turn_index": int(uidx)}
            if int(uidx) in plan_turns:
                return False, "semantic_progress_blocked_but_plan_event", {"user_turn_index": int(uidx)}
            if int(uidx) in goal_turns:
                return False, "semantic_progress_blocked_but_goal_event", {"user_turn_index": int(uidx)}
            ae = _lookup_agency_event(agency_events=list(agency_events), user_turn_index=int(uidx))
            if isinstance(ae, dict):
                ck = str(ae.get("chosen_kind") or "")
                if ck in {"EXECUTE_STEP", "CLOSE_GOAL"}:
                    return False, "semantic_progress_blocked_but_agency_progress", {"user_turn_index": int(uidx), "chosen_kind": str(ck)}

    # Verify deterministic renderers for semantics/explain_semantics/trace_semantics.
    for pe in list(parse_events):
        if not isinstance(pe, dict):
            continue
        tid = str(pe.get("turn_id") or "")
        idx = int(pe.get("turn_index") or 0)
        payload = pe.get("payload") if isinstance(pe.get("payload"), dict) else {}
        intent_id = str(payload.get("intent_id") or "")
        if intent_id not in {INTENT_SEMANTICS_V109, INTENT_EXPLAIN_SEMANTICS_V109, INTENT_TRACE_SEMANTICS_V109}:
            continue
        assistant_text = _assistant_text_after_user_turn(turns=list(turns), user_turn_index=int(idx))
        if intent_id == INTENT_SEMANTICS_V109:
            want_text = render_semantics_text_v109(semantic_events=list(semantic_events), until_user_turn_index_exclusive=int(idx))
            if str(assistant_text) != str(want_text):
                return False, "semantic_render_mismatch", {"turn_id": str(tid), "turn_index": int(idx)}
        elif intent_id == INTENT_EXPLAIN_SEMANTICS_V109:
            q = str(payload.get("query") or "")
            try:
                qidx = int(q)
            except Exception:
                qidx = -1
            want_text = render_explain_semantics_text_v109(semantic_events=list(semantic_events), user_turn_index=int(qidx))
            if str(assistant_text) != str(want_text):
                return False, "explain_semantics_render_mismatch", {"turn_id": str(tid), "turn_index": int(idx)}
        elif intent_id == INTENT_TRACE_SEMANTICS_V109:
            q = str(payload.get("query") or "")
            try:
                qidx = int(q)
            except Exception:
                qidx = -1
            want_text = render_trace_semantics_text_v109(semantic_events=list(semantic_events), until_user_turn_index=int(qidx))
            if str(assistant_text) != str(want_text):
                return False, "trace_semantics_render_mismatch", {"turn_id": str(tid), "turn_index": int(idx)}

    dd = dict(d0)
    dd["semantic_chain_hash_v109"] = str(compute_semantic_chain_hash_v109(list(semantic_events)))
    return True, "ok", dd

