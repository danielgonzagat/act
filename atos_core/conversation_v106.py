from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .conversation_v105 import verify_conversation_chain_v105
from .dialogue_engine_v106 import COHERENCE_THRESH_V106
from .dialogue_ledger_v106 import (
    compute_dialogue_chain_hash_v106,
    dialogue_registry_snapshot_v106,
    render_dialogue_text_v106,
    render_explain_dialogue_text_v106,
    render_trace_dialogue_text_v106,
    verify_dialogue_event_sig_v106,
)
from .intent_grammar_v106 import INTENT_DIALOGUE_V106, INTENT_EXPLAIN_DIALOGUE_V106, INTENT_TRACE_DIALOGUE_V106


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


def verify_conversation_chain_v106(
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
    tail_k: int,
    repo_root: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    ok0, r0, d0 = verify_conversation_chain_v105(
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
        tail_k=int(tail_k),
        repo_root=str(repo_root),
    )
    if not ok0:
        return False, str(r0), dict(d0)

    # Verify dialogue event sig chain.
    prev_sig = ""
    for i, ev in enumerate(list(dialogue_events)):
        if not isinstance(ev, dict):
            return False, "dialogue_event_not_dict", {"index": int(i)}
        ok_ev, rr, dd = verify_dialogue_event_sig_v106(dict(ev))
        if not ok_ev:
            return False, str(rr), dict(dd)
        if str(ev.get("prev_event_sig") or "") != str(prev_sig):
            return False, "dialogue_prev_event_sig_mismatch", {"index": int(i)}
        prev_sig = str(ev.get("event_sig") or "")

    # Verify derived snapshot matches replay.
    want_snap = dialogue_registry_snapshot_v106(list(dialogue_events))
    want_sem = dict(want_snap)
    got_sem = dict(dialogue_snapshot) if isinstance(dialogue_snapshot, dict) else {}
    if _stable_hash_obj(want_sem) != _stable_hash_obj(got_sem):
        return False, "dialogue_snapshot_mismatch", {"want_sig": _stable_hash_obj(want_sem), "got_sig": _stable_hash_obj(got_sem)}

    # Survival rule: cannot have two consecutive low scores without a repair action.
    low_streak = 0
    for ev in list(dialogue_events):
        if not isinstance(ev, dict):
            continue
        score = int(ev.get("coherence_score") or 0)
        repair = str(ev.get("repair_action") or "")
        if score < int(COHERENCE_THRESH_V106):
            low_streak += 1
            if low_streak >= 2 and not repair:
                return False, "dialogue_survival_missing_repair", {"user_turn_index": int(ev.get("user_turn_index") or 0)}
        else:
            low_streak = 0

    # Verify deterministic renderers for dialogue/explain_dialogue/trace_dialogue.
    for pe in list(parse_events):
        if not isinstance(pe, dict):
            continue
        tid = str(pe.get("turn_id") or "")
        idx = int(pe.get("turn_index") or 0)
        payload = pe.get("payload") if isinstance(pe.get("payload"), dict) else {}
        intent_id = str(payload.get("intent_id") or "")
        if intent_id not in {INTENT_DIALOGUE_V106, INTENT_EXPLAIN_DIALOGUE_V106, INTENT_TRACE_DIALOGUE_V106}:
            continue
        assistant_text = _assistant_text_after_user_turn(turns=list(turns), user_turn_index=int(idx))
        if intent_id == INTENT_DIALOGUE_V106:
            want_text = render_dialogue_text_v106(dialogue_events=list(dialogue_events), until_user_turn_index_exclusive=int(idx))
            if str(assistant_text) != str(want_text):
                return False, "dialogue_render_mismatch", {"turn_id": str(tid), "turn_index": int(idx)}
        elif intent_id == INTENT_EXPLAIN_DIALOGUE_V106:
            q = str(payload.get("query") or "")
            try:
                qidx = int(q)
            except Exception:
                qidx = -1
            want_text = render_explain_dialogue_text_v106(dialogue_events=list(dialogue_events), user_turn_index=int(qidx))
            if str(assistant_text) != str(want_text):
                return False, "explain_dialogue_render_mismatch", {"turn_id": str(tid), "turn_index": int(idx)}
        elif intent_id == INTENT_TRACE_DIALOGUE_V106:
            q = str(payload.get("query") or "")
            try:
                qidx = int(q)
            except Exception:
                qidx = -1
            want_text = render_trace_dialogue_text_v106(dialogue_events=list(dialogue_events), until_user_turn_index=int(qidx))
            if str(assistant_text) != str(want_text):
                return False, "trace_dialogue_render_mismatch", {"turn_id": str(tid), "turn_index": int(idx)}

    dd = dict(d0)
    dd["dialogue_chain_hash_v106"] = str(compute_dialogue_chain_hash_v106(list(dialogue_events)))
    return True, "ok", dd

