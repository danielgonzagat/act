from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .conversation_v106 import verify_conversation_chain_v106
from .dialogue_engine_v106 import COHERENCE_THRESH_V106
from .intent_grammar_v107 import INTENT_EXPLAIN_PRAGMATICS_V107, INTENT_PRAGMATICS_V107, INTENT_TRACE_PRAGMATICS_V107
from .pragmatics_engine_v107 import PRAGMATICS_THRESH_V107
from .pragmatics_ledger_v107 import (
    compute_pragmatics_chain_hash_v107,
    pragmatics_registry_snapshot_v107,
    render_explain_pragmatics_text_v107,
    render_pragmatics_text_v107,
    render_trace_pragmatics_text_v107,
    verify_pragmatics_event_sig_v107,
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


def _lookup_dialogue_coherence_score(*, dialogue_events: Sequence[Dict[str, Any]], user_turn_index: int) -> Optional[int]:
    for ev in dialogue_events:
        if not isinstance(ev, dict):
            continue
        if int(ev.get("user_turn_index") or -1) == int(user_turn_index):
            try:
                return int(ev.get("coherence_score") or 0)
            except Exception:
                return 0
    return None


def _lookup_agency_event(*, agency_events: Sequence[Dict[str, Any]], user_turn_index: int) -> Optional[Dict[str, Any]]:
    for ev in agency_events:
        if not isinstance(ev, dict):
            continue
        if int(ev.get("user_turn_index") or -1) == int(user_turn_index):
            return dict(ev)
    return None


def verify_conversation_chain_v107(
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
    tail_k: int,
    repo_root: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    ok0, r0, d0 = verify_conversation_chain_v106(
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
        tail_k=int(tail_k),
        repo_root=str(repo_root),
    )
    if not ok0:
        return False, str(r0), dict(d0)

    # Verify pragmatics inner sig chain.
    prev_sig = ""
    for i, ev in enumerate(list(pragmatics_events)):
        if not isinstance(ev, dict):
            return False, "pragmatics_event_not_dict", {"index": int(i)}
        ok_ev, rr, dd = verify_pragmatics_event_sig_v107(dict(ev))
        if not ok_ev:
            return False, str(rr), dict(dd)
        if str(ev.get("prev_event_sig") or "") != str(prev_sig):
            return False, "pragmatics_prev_event_sig_mismatch", {"index": int(i)}
        prev_sig = str(ev.get("event_sig") or "")

    # Verify derived snapshot matches replay.
    want_snap = pragmatics_registry_snapshot_v107(list(pragmatics_events))
    want_sem = dict(want_snap)
    got_sem = dict(pragmatics_snapshot) if isinstance(pragmatics_snapshot, dict) else {}
    if _stable_hash_obj(want_sem) != _stable_hash_obj(got_sem):
        return False, "pragmatics_snapshot_mismatch", {"want_sig": _stable_hash_obj(want_sem), "got_sig": _stable_hash_obj(got_sem)}

    # Survival rule S1: pragmatics_score < thresh must have repair_action kind.
    for ev in list(pragmatics_events):
        if not isinstance(ev, dict):
            continue
        score = int(ev.get("pragmatics_score") or 0)
        ra = ev.get("repair_action") if isinstance(ev.get("repair_action"), dict) else {}
        ra_kind = str(ra.get("kind") or "")
        if score < int(PRAGMATICS_THRESH_V107) and not ra_kind:
            return False, "pragmatics_survival_missing_repair", {"user_turn_index": int(ev.get("user_turn_index") or 0)}

    # Survival rule S2: no 2 consecutive low (coherence OR pragmatics) without repair.
    low_streak = 0
    for ev in list(pragmatics_events):
        if not isinstance(ev, dict):
            continue
        uidx = int(ev.get("user_turn_index") or 0)
        p_score = int(ev.get("pragmatics_score") or 0)
        d_score = _lookup_dialogue_coherence_score(dialogue_events=list(dialogue_events), user_turn_index=int(uidx))
        if d_score is None:
            d_score = 0
        low = bool(int(p_score) < int(PRAGMATICS_THRESH_V107) or int(d_score) < int(COHERENCE_THRESH_V106))
        ra = ev.get("repair_action") if isinstance(ev.get("repair_action"), dict) else {}
        ra_kind = str(ra.get("kind") or "")
        if low:
            low_streak += 1
            if low_streak >= 2 and not ra_kind:
                return False, "pragmatics_survival_missing_repair_streak", {"user_turn_index": int(uidx)}
        else:
            low_streak = 0

    # Survival rule S3: if progress_blocked, do not emit plan events and do not execute agency progress actions.
    plan_turns = set([int(pe.get("user_turn_index") or pe.get("ts_turn_index") or -1) for pe in plan_events if isinstance(pe, dict)])
    goal_turns = set([int(ge.get("ts_turn_index") or -1) for ge in goal_events if isinstance(ge, dict)])
    for ev in list(pragmatics_events):
        if not isinstance(ev, dict):
            continue
        uidx = int(ev.get("user_turn_index") or 0)
        if bool(ev.get("progress_blocked", False)):
            if int(uidx) in plan_turns:
                return False, "pragmatics_progress_blocked_but_plan_event", {"user_turn_index": int(uidx)}
            if int(uidx) in goal_turns:
                return False, "pragmatics_progress_blocked_but_goal_event", {"user_turn_index": int(uidx)}
            ae = _lookup_agency_event(agency_events=list(agency_events), user_turn_index=int(uidx))
            if isinstance(ae, dict):
                ck = str(ae.get("chosen_kind") or "")
                if ck in {"EXECUTE_STEP", "CLOSE_GOAL"}:
                    return False, "pragmatics_progress_blocked_but_agency_progress", {"user_turn_index": int(uidx), "chosen_kind": str(ck)}

    # Verify deterministic renderers for pragmatics/explain_pragmatics/trace_pragmatics.
    for pe in list(parse_events):
        if not isinstance(pe, dict):
            continue
        tid = str(pe.get("turn_id") or "")
        idx = int(pe.get("turn_index") or 0)
        payload = pe.get("payload") if isinstance(pe.get("payload"), dict) else {}
        intent_id = str(payload.get("intent_id") or "")
        if intent_id not in {INTENT_PRAGMATICS_V107, INTENT_EXPLAIN_PRAGMATICS_V107, INTENT_TRACE_PRAGMATICS_V107}:
            continue
        assistant_text = _assistant_text_after_user_turn(turns=list(turns), user_turn_index=int(idx))
        if intent_id == INTENT_PRAGMATICS_V107:
            want_text = render_pragmatics_text_v107(pragmatics_events=list(pragmatics_events), until_user_turn_index_exclusive=int(idx))
            if str(assistant_text) != str(want_text):
                return False, "pragmatics_render_mismatch", {"turn_id": str(tid), "turn_index": int(idx)}
        elif intent_id == INTENT_EXPLAIN_PRAGMATICS_V107:
            q = str(payload.get("query") or "")
            try:
                qidx = int(q)
            except Exception:
                qidx = -1
            want_text = render_explain_pragmatics_text_v107(pragmatics_events=list(pragmatics_events), user_turn_index=int(qidx))
            if str(assistant_text) != str(want_text):
                return False, "explain_pragmatics_render_mismatch", {"turn_id": str(tid), "turn_index": int(idx)}
        elif intent_id == INTENT_TRACE_PRAGMATICS_V107:
            q = str(payload.get("query") or "")
            try:
                qidx = int(q)
            except Exception:
                qidx = -1
            want_text = render_trace_pragmatics_text_v107(pragmatics_events=list(pragmatics_events), until_user_turn_index=int(qidx))
            if str(assistant_text) != str(want_text):
                return False, "trace_pragmatics_render_mismatch", {"turn_id": str(tid), "turn_index": int(idx)}

    dd = dict(d0)
    dd["pragmatics_chain_hash_v107"] = str(compute_pragmatics_chain_hash_v107(list(pragmatics_events)))
    return True, "ok", dd

