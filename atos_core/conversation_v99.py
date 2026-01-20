from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .conversation_v98 import render_explain_text_v98, verify_conversation_chain_v98
from .goal_ledger_v99 import compute_goal_chain_hash_v99, goal_ledger_snapshot_v99, verify_goal_event_sig_v99
from .intent_grammar_v99 import (
    INTENT_GOAL_ADD_V99,
    INTENT_GOAL_AUTO_V99,
    INTENT_GOAL_DONE_V99,
    INTENT_GOAL_LIST_V99,
    INTENT_GOAL_NEXT_V99,
)


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _canon_str_list(items: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(items, list):
        return out
    for x in items:
        if isinstance(x, str) and x:
            out.append(str(x))
    return sorted(set(out))


def render_explain_text_v99(plan: Dict[str, Any]) -> str:
    base = render_explain_text_v98(plan)
    goal_read = _canon_str_list(plan.get("goal_read_ids"))
    goal_write = _canon_str_list(plan.get("goal_write_ids"))
    prov = plan.get("provenance") if isinstance(plan.get("provenance"), dict) else {}
    goal_ids = _canon_str_list(prov.get("goal_ids"))
    line = (
        "GOAL_EFFECTS: "
        + f"goal_read_ids={json.dumps(goal_read, ensure_ascii=False)} "
        + f"goal_write_ids={json.dumps(goal_write, ensure_ascii=False)} "
        + f"provenance_goal_ids={json.dumps(goal_ids, ensure_ascii=False)}"
    )
    return "\n".join([base, line])


def render_goal_added_ack_text_v99(goal_id: str) -> str:
    return f"GOAL ADDED: {str(goal_id)}"


def render_goal_done_ack_text_v99(goal_id: str) -> str:
    return f"GOAL DONE: {str(goal_id)}"


def render_goals_text_v99(goals_active: Sequence[Dict[str, Any]]) -> str:
    items: List[Tuple[int, int, str, str, str]] = []
    for it in goals_active:
        if not isinstance(it, dict):
            continue
        gid = str(it.get("goal_id") or "")
        if not gid:
            continue
        try:
            pr = int(it.get("priority") or 0)
        except Exception:
            pr = 0
        try:
            created = int(it.get("created_ts_turn_index") or 0)
        except Exception:
            created = 0
        parent = str(it.get("parent_goal_id") or "")
        text = str(it.get("text") or "")
        items.append((int(pr), int(created), str(gid), str(parent), str(text)))
    items.sort(key=lambda t: (-int(t[0]), int(t[1]), str(t[2])))
    if not items:
        return "GOALS: (empty)"
    lines = ["GOALS:"]
    for i, (_pr, _created, gid, parent, text) in enumerate(items):
        lines.append(f"{i+1}) id={gid} priority={_pr} parent={parent} text={json.dumps(text, ensure_ascii=False)}")
    return "\n".join(lines)


def render_next_text_v99(tick_action: Dict[str, Any]) -> str:
    if not isinstance(tick_action, dict):
        return "NEXT: (invalid)"
    gid = str(tick_action.get("selected_goal_id") or "")
    eff = str(tick_action.get("effect") or "")
    sub = str(tick_action.get("subgoal_id") or "")
    if eff == "created_subgoal" and sub:
        return f"NEXT: goal={gid} action=create_subgoal subgoal={sub}"
    if eff == "no_active_goals":
        return "NEXT: no_active_goals"
    return f"NEXT: goal={gid} action=noop"


def render_auto_text_v99(tick_actions: Sequence[Dict[str, Any]], n: int) -> str:
    lines: List[str] = [f"AUTO: n={int(n)}"]
    for i, ta in enumerate(list(tick_actions)[: int(n)]):
        lines.append(f"{i+1}) {render_next_text_v99(dict(ta) if isinstance(ta, dict) else {})}")
    return "\n".join(lines)


def verify_conversation_chain_v99(
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
    tail_k: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    ok0, reason0, details0 = verify_conversation_chain_v98(
        turns=turns,
        states=states,
        parse_events=parse_events,
        trials=trials,
        learned_rule_events=learned_rule_events,
        action_plans=action_plans,
        memory_events=memory_events,
        belief_events=belief_events,
        evidence_events=evidence_events,
        tail_k=tail_k,
    )
    if not ok0:
        return False, str(reason0), dict(details0)

    # Verify goal events and inner chain.
    created_step_prev = -1
    prev_sig = ""
    seen_goal_ids: Dict[str, bool] = {}
    goal_events_list: List[Dict[str, Any]] = []
    for ev in goal_events:
        if not isinstance(ev, dict):
            return False, "goal_event_not_dict", {}
        ok_ev, greason, gdetails = verify_goal_event_sig_v99(dict(ev))
        if not ok_ev:
            return False, str(greason), dict(gdetails)
        try:
            cstep = int(ev.get("created_step", -1))
        except Exception:
            cstep = -1
        if cstep < 0:
            return False, "goal_event_bad_created_step", {"event_id": str(ev.get("event_id") or "")}
        if cstep < created_step_prev:
            return False, "goal_events_not_monotonic", {"event_id": str(ev.get("event_id") or "")}
        created_step_prev = int(cstep)

        if str(ev.get("prev_event_sig") or "") != str(prev_sig):
            return False, "goal_prev_event_sig_mismatch", {"event_id": str(ev.get("event_id") or "")}
        prev_sig = str(ev.get("event_sig") or "")

        op = str(ev.get("op") or "")
        gid = str(ev.get("goal_id") or "")
        if op == "GOAL_ADD":
            seen_goal_ids[gid] = True
        else:
            if gid not in seen_goal_ids:
                return False, "goal_unknown_id", {"goal_id": gid, "op": op}

        goal_events_list.append(dict(ev))

    expected_snapshot = goal_ledger_snapshot_v99(goal_events_list)
    if not isinstance(goal_snapshot, dict):
        return False, "goal_snapshot_not_dict", {}
    if canonical_json_dumps(dict(goal_snapshot)) != canonical_json_dumps(dict(expected_snapshot)):
        return False, "goal_snapshot_mismatch", {"want": str(expected_snapshot.get("snapshot_sig") or ""), "got": str(goal_snapshot.get("snapshot_sig") or "")}

    # Build helpers for turn->parse and plan lookup.
    by_index: Dict[int, Dict[str, Any]] = {}
    max_idx = -1
    for t in turns:
        if not isinstance(t, dict):
            continue
        try:
            idx = int(t.get("turn_index", -1))
        except Exception:
            idx = -1
        if idx >= 0:
            by_index[idx] = dict(t)
            if idx > max_idx:
                max_idx = int(idx)

    parses_by_turn_id: Dict[str, Dict[str, Any]] = {}
    for pe in parse_events:
        if not isinstance(pe, dict):
            continue
        tid = str(pe.get("turn_id") or "")
        payload = pe.get("payload") if isinstance(pe.get("payload"), dict) else {}
        if tid:
            parses_by_turn_id[tid] = dict(payload)

    plans_by_user_turn_id: Dict[str, Dict[str, Any]] = {}
    for p in action_plans:
        if not isinstance(p, dict):
            continue
        utid = str(p.get("user_turn_id") or "")
        if utid:
            plans_by_user_turn_id[utid] = dict(p)

    # Cross-check deterministic renderers for goal commands.
    for i in range(0, max_idx + 1, 2):
        ut = by_index.get(i)
        at = by_index.get(i + 1)
        if not isinstance(ut, dict) or not isinstance(at, dict):
            continue
        utid = str(ut.get("turn_id") or "")
        payload = parses_by_turn_id.get(utid)
        if not isinstance(payload, dict):
            continue
        intent_id = str(payload.get("intent_id") or "")
        plan = plans_by_user_turn_id.get(utid)
        if not isinstance(plan, dict):
            continue
        try:
            cutoff = int(plan.get("created_step", -1))
        except Exception:
            cutoff = -1
        if cutoff < 0:
            continue

        ge_at = [dict(ev) for ev in goal_events_list if isinstance(ev, dict) and int(ev.get("created_step", -1) or -1) <= int(cutoff)]
        snap_at = goal_ledger_snapshot_v99(ge_at)

        if intent_id == INTENT_GOAL_LIST_V99:
            want = render_goals_text_v99(snap_at.get("goals_active") if isinstance(snap_at.get("goals_active"), list) else [])
            if str(at.get("text") or "") != str(want):
                return False, "goals_text_mismatch", {"turn_id": utid}
        if intent_id == INTENT_GOAL_ADD_V99:
            gw = _canon_str_list(plan.get("goal_write_ids"))
            if len(gw) != 1:
                return False, "goal_add_plan_missing_goal_write_id", {"turn_id": utid}
            want = render_goal_added_ack_text_v99(gw[0])
            if str(at.get("text") or "") != str(want):
                return False, "goal_add_text_mismatch", {"turn_id": utid}
        if intent_id == INTENT_GOAL_DONE_V99:
            gw2 = _canon_str_list(plan.get("goal_write_ids"))
            if len(gw2) != 1:
                return False, "goal_done_plan_missing_goal_write_id", {"turn_id": utid}
            want = render_goal_done_ack_text_v99(gw2[0])
            if str(at.get("text") or "") != str(want):
                return False, "goal_done_text_mismatch", {"turn_id": utid}
        if intent_id == INTENT_GOAL_NEXT_V99:
            ta = plan.get("goal_tick_actions")
            ta_list = ta if isinstance(ta, list) else []
            want = render_next_text_v99(dict(ta_list[0]) if ta_list else {})
            if str(at.get("text") or "") != str(want):
                return False, "next_text_mismatch", {"turn_id": utid}
        if intent_id == INTENT_GOAL_AUTO_V99:
            ta2 = plan.get("goal_tick_actions")
            ta2_list = ta2 if isinstance(ta2, list) else []
            n = int(payload.get("n") or 0) if isinstance(payload.get("n"), int) else 0
            want = render_auto_text_v99([dict(x) for x in ta2_list if isinstance(x, dict)], int(n))
            if str(at.get("text") or "") != str(want):
                return False, "auto_text_mismatch", {"turn_id": utid}

    # Action plans: canonical goal lists and provenance union.
    for p in action_plans:
        if not isinstance(p, dict):
            continue
        utid = str(p.get("user_turn_id") or "")
        gread = _canon_str_list(p.get("goal_read_ids"))
        gwrite = _canon_str_list(p.get("goal_write_ids"))
        if _canon_str_list(p.get("goal_read_ids")) != gread or _canon_str_list(p.get("goal_write_ids")) != gwrite:
            return False, "plan_goal_lists_not_canonical", {"user_turn_id": utid}
        prov = p.get("provenance") if isinstance(p.get("provenance"), dict) else {}
        gids = _canon_str_list(prov.get("goal_ids"))
        want_union = sorted(set(list(gread) + list(gwrite)))
        if want_union != list(gids):
            return False, "plan_goal_ids_union_mismatch", {"user_turn_id": utid, "want": want_union, "got": list(gids)}

    d = dict(details0)
    d["goal_events_total"] = int(len(goal_events_list))
    d["goal_chain_hash"] = str(compute_goal_chain_hash_v99(goal_events_list))
    return True, "ok", d

