from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps
from .conversation_v101 import verify_conversation_chain_v101
from .discourse_templates_v102 import TemplateV102, base_templates_v102, render_templates_list_text_v102
from .intent_grammar_v100 import INTENT_DISCOURSE_V100
from .intent_grammar_v102 import (
    INTENT_EXPLAIN_STYLE_V102,
    INTENT_STYLE_PROFILE_V102,
    INTENT_TEMPLATES_V102,
    INTENT_TRACE_STYLE_V102,
)
from .style_ledger_v102 import compute_style_chain_hash_v102, fold_template_stats_v102, template_library_snapshot_v102, verify_style_event_sig_v102
from .style_profile_v102 import StyleProfileV102, coerce_style_profile_v102, default_style_profile_v102, render_style_profile_text_v102
from .style_selector_v102 import explain_style_text_v102


def _canon_str_list(items: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(items, list):
        return out
    for x in items:
        if isinstance(x, str) and x:
            out.append(str(x))
    return sorted(set(out))


def render_trace_style_text_v102(*, turn_id: str, style_events: Sequence[Dict[str, Any]]) -> str:
    tid = str(turn_id or "")
    for ev in style_events:
        if isinstance(ev, dict) and str(ev.get("turn_id") or "") == tid:
            return f"TRACE_STYLE: turn_id={tid} style_event_id={str(ev.get('event_id') or '')}"
    return f"TRACE_STYLE: turn_id={tid} style_event_id="


def render_dossier_text_v102() -> str:
    """
    Deterministic, audit-oriented dossier (V102). Adds style ledger invariants.
    """
    lines: List[str] = []
    lines.append("DOSSIER (V102):")
    lines.append("1) Record-keeping: WORM write-once files + append-only hash-chained JSONL logs.")
    lines.append("2) Replay: run verifier scripts on run_dir (e.g., scripts/verify_conversation_chain_v102.py --run_dir <RUN_DIR>).")
    lines.append("3) Determinismo: seed=0; canonical_json_dumps; explicit sorting; no wall-clock time.")
    lines.append("4) Style layer: style_events.jsonl (hash-chained + event_sig) with candidates TOP-K, critics, selection, and style_chain_hash.")
    lines.append("5) Fail-closed: if all style candidates fail critics, fallback safe response is emitted and logged.")
    lines.append("6) No-hybridization: verifier scans atos_core/ and scripts/ for forbidden ML/LLM imports.")
    return "\n".join(lines)


def verify_conversation_chain_v102(
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
    tail_k: int,
    repo_root: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    V102 invariants:
      - All V101 invariants must hold (backwards compatible).
      - style_events: signature chain ok; selected candidate matches assistant text; deterministic sorting.
      - template_snapshot matches fold(style_events).
      - Deterministic renderers for style_profile/templates/explain_style/trace_style.
    """
    ok0, reason0, details0 = verify_conversation_chain_v101(
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
        tail_k=int(tail_k),
        repo_root=str(repo_root),
    )
    if not ok0:
        return False, str(reason0), dict(details0)

    # Turn indices helpers.
    by_turn_id: Dict[str, Dict[str, Any]] = {}
    by_index: Dict[int, Dict[str, Any]] = {}
    max_idx = -1
    for t in turns:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("turn_id") or "")
        if tid:
            by_turn_id[tid] = dict(t)
        try:
            idx = int(t.get("turn_index", -1))
        except Exception:
            idx = -1
        if idx >= 0:
            by_index[idx] = dict(t)
            max_idx = max(max_idx, int(idx))

    parses_by_turn_id: Dict[str, Dict[str, Any]] = {}
    for pe in parse_events:
        if not isinstance(pe, dict):
            continue
        tid = str(pe.get("turn_id") or "")
        payload = pe.get("payload") if isinstance(pe.get("payload"), dict) else {}
        if tid:
            parses_by_turn_id[tid] = dict(payload)

    # State by user turn id (existing invariant: absent for clarifications/confirmations).
    state_by_user_turn_id: Dict[str, Dict[str, Any]] = {}
    for st in states:
        if not isinstance(st, dict):
            continue
        utid = str(st.get("last_user_turn_id") or "")
        if utid:
            state_by_user_turn_id[utid] = dict(st)

    # Verify style event sig chain and selection correctness.
    prev_sig = ""
    prev_after: Optional[Dict[str, Any]] = None
    for i, ev in enumerate(list(style_events)):
        if not isinstance(ev, dict):
            return False, "style_event_not_dict", {"index": int(i)}
        ok_ev, rreason, rdetails = verify_style_event_sig_v102(dict(ev))
        if not ok_ev:
            return False, str(rreason), dict(rdetails)
        if str(ev.get("prev_event_sig") or "") != str(prev_sig):
            return False, "style_prev_event_sig_mismatch", {"index": int(i)}
        prev_sig = str(ev.get("event_sig") or "")

        before = ev.get("style_profile_before") if isinstance(ev.get("style_profile_before"), dict) else {}
        after = ev.get("style_profile_after") if isinstance(ev.get("style_profile_after"), dict) else {}
        if prev_after is not None and canonical_json_dumps(dict(before)) != canonical_json_dumps(dict(prev_after)):
            return False, "style_profile_chain_mismatch", {"index": int(i)}
        prev_after = dict(after)

        turn_id = str(ev.get("turn_id") or "")
        at = by_turn_id.get(turn_id)
        if not isinstance(at, dict):
            return False, "style_turn_missing", {"turn_id": turn_id}
        if str(at.get("role") or "") != "assistant":
            return False, "style_turn_not_assistant", {"turn_id": turn_id}

        sel_id = str(ev.get("selected_candidate_id") or "")
        cands = ev.get("candidates_topk") if isinstance(ev.get("candidates_topk"), list) else []
        cand_by_id: Dict[str, Dict[str, Any]] = {}
        for c in cands:
            if isinstance(c, dict) and str(c.get("candidate_id") or ""):
                cand_by_id[str(c.get("candidate_id") or "")] = dict(c)
        if sel_id and sel_id not in cand_by_id:
            return False, "style_selected_candidate_missing", {"turn_id": turn_id}
        if sel_id:
            sel = cand_by_id[sel_id]
            got_sha = str(sel.get("text_sha256") or "")
            # recompute sha from assistant text (stored sha in candidate is stable hash, not file hash).
            # Candidate sha is computed by style_critics_v102.text_sha256_v102: sha256_hex(text_bytes).
            from .style_critics_v102 import text_sha256_v102  # local import to avoid cycles

            at_text = str(at.get("text") or "")
            at_sha = text_sha256_v102(at_text)
            if got_sha != at_sha:
                return False, "style_selected_text_sha_mismatch", {"turn_id": turn_id}

        # Verify candidates sorted deterministically by stored fields.
        def _rk(c: Dict[str, Any]) -> Tuple[Any, ...]:
            okc = bool(((c.get("critics") or {}).get("ok")) if isinstance(c.get("critics"), dict) else False)
            m = c.get("fluency_metrics") if isinstance(c.get("fluency_metrics"), dict) else {}
            return (
                0 if okc else 1,
                -float(c.get("fluency_score") or 0.0),
                int(m.get("words") or 0),
                str(c.get("template_id") or ""),
                str(c.get("text_sha256") or ""),
                str(c.get("candidate_id") or ""),
            )

        got_ids = [str(c.get("candidate_id") or "") for c in cands if isinstance(c, dict)]
        want_ids = [str(c.get("candidate_id") or "") for c in sorted([dict(x) for x in cands if isinstance(x, dict)], key=_rk)]
        if got_ids != want_ids:
            return False, "style_candidates_not_sorted", {"turn_id": turn_id}

        sel_meta = ev.get("selection") if isinstance(ev.get("selection"), dict) else {}
        method = str(sel_meta.get("method") or "")
        if method == "argmax":
            # must pick the first ok candidate.
            ok_ids = [cid for cid in want_ids if cid and bool(((cand_by_id.get(cid, {}).get("critics") or {}).get("ok")))]
            if ok_ids and sel_id and sel_id != ok_ids[0]:
                return False, "style_argmax_not_top1", {"turn_id": turn_id}
        elif method == "soft":
            # must pick top3[soft_index] among ok candidates.
            try:
                sidx = int(sel_meta.get("soft_index") or 0)
            except Exception:
                sidx = 0
            ok_ids = [cid for cid in want_ids if cid and bool(((cand_by_id.get(cid, {}).get("critics") or {}).get("ok")))]
            top3 = ok_ids[:3]
            if top3:
                want = top3[min(max(sidx, 0), len(top3) - 1)]
                if sel_id != want:
                    return False, "style_soft_selection_mismatch", {"turn_id": turn_id}
        elif method == "fallback":
            # ok
            pass
        elif method == "locked":
            # Selection was intentionally locked to the base/core text (no style override).
            pass
        else:
            return False, "style_unknown_selection_method", {"turn_id": turn_id, "method": method}

    # Template snapshot must match fold from events.
    templates = base_templates_v102()
    want_stats = fold_template_stats_v102(templates=list(templates), style_events=list(style_events))
    want_snap = template_library_snapshot_v102(templates=list(templates), style_events=list(style_events))
    if canonical_json_dumps(dict(template_snapshot)) != canonical_json_dumps(dict(want_snap)):
        return False, "template_snapshot_mismatch", {"want": str(want_snap.get("snapshot_sig") or ""), "got": str(template_snapshot.get("snapshot_sig") or "")}

    # Cross-check deterministic renderers for style introspection commands.
    # For each user turn, if intent is one of the style commands, assistant text must match renderer.
    for i in range(0, max_idx + 1, 2):
        ut = by_index.get(i)
        at = by_index.get(i + 1)
        if not isinstance(ut, dict) or not isinstance(at, dict):
            continue
        utid = str(ut.get("turn_id") or "")
        payload = parses_by_turn_id.get(utid)
        if not isinstance(payload, dict):
            continue
        iid = str(payload.get("intent_id") or "")
        if iid == INTENT_STYLE_PROFILE_V102:
            st = state_by_user_turn_id.get(utid, {})
            bnd = st.get("bindings") if isinstance(st.get("bindings"), dict) else {}
            prof = coerce_style_profile_v102(bnd.get("style_profile"))
            want = render_style_profile_text_v102(prof)
            if str(at.get("text") or "") != str(want):
                return False, "style_profile_text_mismatch", {"turn_id": utid}
        if iid == INTENT_TEMPLATES_V102:
            # `templates` must render the stats as-of that turn (prefix of style events),
            # because the assistant cannot "know" future uses.
            try:
                at_idx = int(at.get("turn_index", -1))
            except Exception:
                at_idx = -1
            prefix_style_events: List[Dict[str, Any]] = []
            if at_idx >= 0:
                for sev in style_events:
                    if not isinstance(sev, dict):
                        continue
                    sev_tid = str(sev.get("turn_id") or "")
                    sev_turn = by_turn_id.get(sev_tid)
                    if not isinstance(sev_turn, dict):
                        continue
                    try:
                        sev_idx = int(sev_turn.get("turn_index", -1))
                    except Exception:
                        sev_idx = -1
                    if sev_idx >= 0 and sev_idx < at_idx:
                        prefix_style_events.append(dict(sev))
            stats_now = fold_template_stats_v102(templates=list(templates), style_events=list(prefix_style_events))
            want = render_templates_list_text_v102(templates=list(templates), template_stats=dict(stats_now))
            if str(at.get("text") or "") != str(want):
                return False, "templates_text_mismatch", {"turn_id": utid}
        if iid == INTENT_EXPLAIN_STYLE_V102:
            turn_id = str(payload.get("turn_id") or "")
            target_ev = None
            for sev in style_events:
                if isinstance(sev, dict) and str(sev.get("turn_id") or "") == turn_id:
                    target_ev = dict(sev)
                    break
            if target_ev is None:
                return False, "explain_style_missing_event", {"turn_id": utid, "target_turn_id": turn_id}
            want = explain_style_text_v102(style_event=dict(target_ev))
            if str(at.get("text") or "") != str(want):
                return False, "explain_style_text_mismatch", {"turn_id": utid}
        if iid == INTENT_TRACE_STYLE_V102:
            turn_id = str(payload.get("turn_id") or "")
            want = render_trace_style_text_v102(turn_id=str(turn_id), style_events=list(style_events))
            if str(at.get("text") or "") != str(want):
                return False, "trace_style_text_mismatch", {"turn_id": utid}

    dd = dict(details0)
    dd["style_events_total"] = int(len(style_events))
    dd["style_chain_hash"] = str(compute_style_chain_hash_v102(style_events))
    dd["template_snapshot_sig"] = str(want_snap.get("snapshot_sig") or "")
    return True, "ok", dd
