from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .agency_ledger_v105 import AGENCY_KIND_CLOSE_GOAL_V105, AGENCY_KIND_EXECUTE_STEP_V105
from .conversation_v109 import verify_conversation_chain_v109
from .executive_engine_v110 import (
    REPAIR_FORCE_NEXT_STEP_V110,
    REPAIR_PLAN_REVISION_OR_ASK_V110,
)
from .executive_ledger_v110 import (
    compute_executive_chain_hash_v110,
    fold_executive_ledger_v110,
    lookup_executive_event_v110,
    render_executive_text_v110,
    render_explain_executive_text_v110,
    render_trace_executive_text_v110,
    verify_executive_event_sig_v110,
)
from .intent_grammar_v110 import (
    INTENT_EXECUTIVE_V110,
    INTENT_EXPLAIN_EXECUTIVE_V110,
    INTENT_TRACE_EXECUTIVE_V110,
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


def verify_conversation_chain_v110(
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
    executive_events: Sequence[Dict[str, Any]],
    executive_snapshot: Dict[str, Any],
    tail_k: int,
    repo_root: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    ok0, r0, d0 = verify_conversation_chain_v109(
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
        semantic_events=semantic_events,
        semantic_snapshot=dict(semantic_snapshot),
        tail_k=int(tail_k),
        repo_root=str(repo_root),
    )
    if not ok0:
        return False, str(r0), dict(d0)

    # Verify executive inner sig chain.
    prev_sig = ""
    for i, ev in enumerate(list(executive_events)):
        if not isinstance(ev, dict):
            return False, "executive_event_not_dict", {"index": int(i)}
        ok_ev, rr, dd = verify_executive_event_sig_v110(dict(ev))
        if not ok_ev:
            return False, str(rr), dict(dd)
        if str(ev.get("prev_event_sig") or "") != str(prev_sig):
            return False, "executive_prev_event_sig_mismatch", {"index": int(i)}
        prev_sig = str(ev.get("event_sig") or "")

    # Verify derived snapshot matches replay.
    want_snap = fold_executive_ledger_v110(list(executive_events))
    got_snap = dict(executive_snapshot) if isinstance(executive_snapshot, dict) else {}
    want_sig = str(want_snap.get("snapshot_sig") or "")
    got_sig = str(got_snap.get("snapshot_sig") or "")
    if want_sig != got_sig or _stable_hash_obj(want_snap) != _stable_hash_obj(got_snap):
        return False, "executive_snapshot_mismatch", {"want_sig": str(want_sig), "got_sig": str(got_sig)}

    # Survival invariants S6/S7/S8: when progress_allowed=false, no goal/plan progress events and no agency progress.
    plan_turns = set([int(pe.get("user_turn_index") or pe.get("ts_turn_index") or -1) for pe in plan_events if isinstance(pe, dict)])
    goal_turns = set([int(ge.get("ts_turn_index") or -1) for ge in goal_events if isinstance(ge, dict)])

    no_progress_streak = 0
    for ev in list(executive_events):
        if not isinstance(ev, dict):
            continue
        uidx = int(ev.get("user_turn_index") or 0)
        goal_id = str(ev.get("goal_id") or "")
        has_open_goal = bool(goal_id)
        flags = ev.get("executive_flags_v110") if isinstance(ev.get("executive_flags_v110"), list) else []
        fl = set([str(x) for x in flags if isinstance(x, str) and str(x)])
        progress_allowed = bool(ev.get("progress_allowed_v110", False))
        repair = str(ev.get("repair_action_v110") or "")
        has_no_progress = bool(has_open_goal and ("NO_PROGRESS" in fl))
        would_streak = int(no_progress_streak) + 1 if has_no_progress else 0

        # S6/S8: hard block when explicitly marked passive/overclarify.
        if has_open_goal and ("PASSIVE_WITH_OPEN_GOAL" in fl or "OVERCLARIFY" in fl):
            if progress_allowed:
                return False, "executive_progress_allowed_but_passive", {"user_turn_index": int(uidx)}
            if not repair or str(repair) != str(REPAIR_FORCE_NEXT_STEP_V110):
                return False, "executive_missing_force_next_step", {"user_turn_index": int(uidx)}

        # S7: stall after 2 consecutive no-progress turns.
        if has_open_goal and int(would_streak) >= 2:
            if progress_allowed:
                return False, "executive_progress_allowed_but_stall", {"user_turn_index": int(uidx)}
            if not repair or str(repair) != str(REPAIR_PLAN_REVISION_OR_ASK_V110):
                return False, "executive_missing_plan_revision_or_ask", {"user_turn_index": int(uidx)}

        if not progress_allowed and has_open_goal:
            if int(uidx) in plan_turns:
                return False, "executive_progress_blocked_but_plan_event", {"user_turn_index": int(uidx)}
            if int(uidx) in goal_turns:
                return False, "executive_progress_blocked_but_goal_event", {"user_turn_index": int(uidx)}
            ae = _lookup_agency_event(agency_events=list(agency_events), user_turn_index=int(uidx))
            if isinstance(ae, dict):
                ck = str(ae.get("chosen_kind") or "")
                if ck in {AGENCY_KIND_EXECUTE_STEP_V105, AGENCY_KIND_CLOSE_GOAL_V105}:
                    return False, "executive_progress_blocked_but_agency_progress", {"user_turn_index": int(uidx), "chosen_kind": str(ck)}

        # Update streak (repair breaks the chain deterministically).
        if has_no_progress and not bool(repair):
            no_progress_streak = int(would_streak)
        else:
            no_progress_streak = 0

    # Verify deterministic renderers for executive/explain_executive/trace_executive.
    for pe in list(parse_events):
        if not isinstance(pe, dict):
            continue
        idx = int(pe.get("turn_index") or 0)
        payload = pe.get("payload") if isinstance(pe.get("payload"), dict) else {}
        intent_id = str(payload.get("intent_id") or "")
        if intent_id not in {INTENT_EXECUTIVE_V110, INTENT_EXPLAIN_EXECUTIVE_V110, INTENT_TRACE_EXECUTIVE_V110}:
            continue
        assistant_text = _assistant_text_after_user_turn(turns=list(turns), user_turn_index=int(idx))
        prefix_events = [dict(ev) for ev in executive_events if isinstance(ev, dict) and int(ev.get("user_turn_index") or -1) < int(idx)]
        if intent_id == INTENT_EXECUTIVE_V110:
            want_text = render_executive_text_v110(snapshot=fold_executive_ledger_v110(prefix_events))
            if str(assistant_text) != str(want_text):
                return False, "executive_render_mismatch", {"turn_index": int(idx)}
        elif intent_id == INTENT_EXPLAIN_EXECUTIVE_V110:
            q = str(payload.get("query") or "")
            ev0 = lookup_executive_event_v110(executive_events=list(prefix_events), query=str(q))
            want_text = render_explain_executive_text_v110(executive_event=dict(ev0) if isinstance(ev0, dict) else {})
            if str(assistant_text) != str(want_text):
                return False, "explain_executive_render_mismatch", {"turn_index": int(idx)}
        elif intent_id == INTENT_TRACE_EXECUTIVE_V110:
            q = str(payload.get("query") or "")
            try:
                qidx = int(q)
            except Exception:
                qidx = -1
            want_text = render_trace_executive_text_v110(executive_events=list(prefix_events), until_user_turn_index=int(qidx))
            if str(assistant_text) != str(want_text):
                return False, "trace_executive_render_mismatch", {"turn_index": int(idx)}

    dd = dict(d0)
    dd["executive_chain_hash_v110"] = str(compute_executive_chain_hash_v110(list(executive_events)))
    return True, "ok", dd
