from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .concept_engine_v103 import nearest_concepts_v103
from .concept_ledger_v103 import compute_concept_chain_hash_v103, verify_concept_event_sig_v103
from .concept_registry_v103 import (
    concept_library_snapshot_v103,
    concepts_list_view_v103,
    explain_concept_view_v103,
    fold_concept_ledger_v103,
    lookup_concept_id_by_name_v103,
)
from .conversation_v102 import verify_conversation_chain_v102
from .conversation_v96 import verify_chained_jsonl_v96
from .intent_grammar_v103 import (
    INTENT_CONCEPTS_V103,
    INTENT_EXPLAIN_CONCEPT_V103,
    INTENT_TRACE_CONCEPTS_V103,
)


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def render_concepts_text_v103(*, registry: Dict[str, Any]) -> str:
    items = concepts_list_view_v103(registry=dict(registry))
    if not items:
        return "CONCEPTS: (empty)"
    lines: List[str] = ["CONCEPTS:"]
    for i, it in enumerate(items):
        lines.append(
            f"{i+1}) id={it.get('concept_id')} name={it.get('name')} status={it.get('status')} usage={int(it.get('usage_count',0))} "
            f"toc={int(it.get('toc_pass',0))}/{int(it.get('toc_fail',0))} mdl_delta_bits={int(it.get('mdl_delta_bits',0))}"
        )
    return "\n".join(lines)


def render_explain_concept_text_v103(*, registry: Dict[str, Any], query: str) -> str:
    cid = lookup_concept_id_by_name_v103(registry=dict(registry), name_or_id=str(query))
    if not cid:
        return "EXPLAIN_CONCEPT: not_found"
    view = explain_concept_view_v103(registry=dict(registry), concept_id=str(cid))
    csg = view.get("csg") if isinstance(view.get("csg"), dict) else {}
    csv = view.get("csv") if isinstance(view.get("csv"), dict) else {}
    name = str(csg.get("name") or "")
    status = str(csv.get("status") or "")
    feats = []
    body = csg.get("body") if isinstance(csg.get("body"), dict) else {}
    if isinstance(body.get("features"), list):
        feats = [str(x) for x in body.get("features") if str(x)]
    pos_sigs = csv.get("pos_example_sigs") if isinstance(csv.get("pos_example_sigs"), list) else []
    neg_sigs = csv.get("neg_example_sigs") if isinstance(csv.get("neg_example_sigs"), list) else []
    lines: List[str] = []
    lines.append(f"EXPLAIN_CONCEPT: id={cid} name={name} status={status}")
    lines.append(f"FEATURES: {canonical_json_dumps(sorted(feats, key=str))}")
    lines.append(f"EVIDENCE_POS: {canonical_json_dumps(sorted([str(x) for x in pos_sigs if str(x)], key=str))}")
    lines.append(f"EVIDENCE_NEG: {canonical_json_dumps(sorted([str(x) for x in neg_sigs if str(x)], key=str))}")
    lines.append(
        "MDL: baseline_bits={b} model_bits={m} data_bits={d} delta_bits={dd}".format(
            b=int(csv.get("mdl_baseline_bits") or 0),
            m=int(csv.get("mdl_model_bits") or 0),
            d=int(csv.get("mdl_data_bits") or 0),
            dd=int(csv.get("mdl_delta_bits") or 0),
        )
    )
    lines.append(f"TOC: pass={int(csv.get('toc_pass') or 0)} fail={int(csv.get('toc_fail') or 0)}")
    return "\n".join(lines)


def _turn_index_by_turn_id(*, turns: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in turns:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("turn_id") or "")
        if not tid:
            continue
        try:
            idx = int(t.get("turn_index", -1))
        except Exception:
            idx = -1
        if idx >= 0:
            out[tid] = int(idx)
    return out


def render_trace_concepts_text_v103(
    *,
    registry_prefix: Dict[str, Any],
    concept_events: Sequence[Dict[str, Any]],
    turns: Sequence[Dict[str, Any]],
    target_turn_id: str,
) -> str:
    tid = str(target_turn_id or "")
    if not tid:
        return "TRACE_CONCEPTS: missing_turn_id"
    # Find target user text.
    text = ""
    for t in turns:
        if isinstance(t, dict) and str(t.get("turn_id") or "") == tid:
            text = str(t.get("text") or "")
            break
    matched: List[Dict[str, Any]] = []
    toc: List[Dict[str, Any]] = []
    for ev in concept_events:
        if not isinstance(ev, dict):
            continue
        if str(ev.get("turn_id") or "") != tid:
            continue
        et = str(ev.get("type") or "")
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        if et == "CONCEPT_MATCH":
            matched.append({"concept_id": str(payload.get("concept_id") or ""), "evidence": payload.get("evidence")})
        if et in {"CONCEPT_TOC_PASS", "CONCEPT_TOC_FAIL"}:
            toc.append({"type": et, "concept_id": str(payload.get("concept_id") or ""), "domain_distance": payload.get("domain_distance")})
    matched.sort(key=lambda d: str(d.get("concept_id") or ""))
    toc.sort(key=lambda d: (str(d.get("concept_id") or ""), str(d.get("type") or "")))

    c_by_id = registry_prefix.get("concepts_by_id") if isinstance(registry_prefix.get("concepts_by_id"), dict) else {}
    concepts = [dict(c_by_id[cid]) for cid in sorted(c_by_id.keys(), key=str)]
    nearest = nearest_concepts_v103(concepts=concepts, text=str(text), top_k=3)
    lines: List[str] = []
    lines.append(f"TRACE_CONCEPTS: turn_id={tid}")
    if matched:
        parts: List[str] = []
        for m in matched:
            evs = m.get("evidence")
            evs_list = [str(x) for x in evs] if isinstance(evs, list) else []
            parts.append(f"{m.get('concept_id')} evidence={canonical_json_dumps(sorted(evs_list, key=str))}")
        lines.append("MATCHED: " + "; ".join(parts))
    else:
        lines.append("MATCHED: (none)")
    if toc:
        parts2: List[str] = []
        for t in toc:
            parts2.append(f"{t.get('type')} {t.get('concept_id')} dist={t.get('domain_distance')}")
        lines.append("TOC: " + "; ".join(parts2))
    else:
        lines.append("TOC: (none)")
    if nearest:
        parts3: List[str] = []
        for n in nearest:
            parts3.append(f"{n.get('concept_id')} sim={n.get('similarity')}")
        lines.append("NEAREST: " + "; ".join(parts3))
    else:
        lines.append("NEAREST: (none)")
    return "\n".join(lines)


def verify_conversation_chain_v103(
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
    tail_k: int,
    repo_root: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    ok0, r0, d0 = verify_conversation_chain_v102(
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
        tail_k=int(tail_k),
        repo_root=str(repo_root),
    )
    if not ok0:
        return False, str(r0), dict(d0)

    # Verify concept event sig chain.
    prev_sig = ""
    for i, ev in enumerate(list(concept_events)):
        if not isinstance(ev, dict):
            return False, "concept_event_not_dict", {"index": int(i)}
        ok_ev, rr, dd = verify_concept_event_sig_v103(dict(ev))
        if not ok_ev:
            return False, str(rr), dict(dd)
        if str(ev.get("prev_event_sig") or "") != str(prev_sig):
            return False, "concept_prev_event_sig_mismatch", {"index": int(i)}
        prev_sig = str(ev.get("event_sig") or "")

    # Snapshot must match fold.
    want_snap = concept_library_snapshot_v103(concept_events=list(concept_events))
    if canonical_json_dumps(dict(concept_snapshot)) != canonical_json_dumps(dict(want_snap)):
        return False, "concept_snapshot_mismatch", {"want": str(want_snap.get("snapshot_sig") or ""), "got": str(concept_snapshot.get("snapshot_sig") or "")}

    # Render checks for concept introspection commands (prefix state).
    parses_by_turn_id: Dict[str, Dict[str, Any]] = {}
    for pe in parse_events:
        if not isinstance(pe, dict):
            continue
        tid = str(pe.get("turn_id") or "")
        payload = pe.get("payload") if isinstance(pe.get("payload"), dict) else {}
        if tid:
            parses_by_turn_id[tid] = dict(payload)

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

    turn_idx_by_id = _turn_index_by_turn_id(turns=turns)

    def _concept_events_prefix_for_assistant_turn(at_idx: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for ev in concept_events:
            if not isinstance(ev, dict):
                continue
            tid = str(ev.get("turn_id") or "")
            ti = int(turn_idx_by_id.get(tid, -1))
            if ti >= 0 and ti < int(at_idx):
                out.append(dict(ev))
        return out

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
        at_idx = int(at.get("turn_index", -1))
        prefix_events = _concept_events_prefix_for_assistant_turn(at_idx)
        reg_prefix, _det = fold_concept_ledger_v103(list(prefix_events))
        if iid == INTENT_CONCEPTS_V103:
            want = render_concepts_text_v103(registry=dict(reg_prefix))
            if str(at.get("text") or "") != str(want):
                return False, "concepts_text_mismatch", {"turn_id": utid}
        if iid == INTENT_EXPLAIN_CONCEPT_V103:
            q = str(payload.get("query") or "")
            want = render_explain_concept_text_v103(registry=dict(reg_prefix), query=str(q))
            if str(at.get("text") or "") != str(want):
                return False, "explain_concept_text_mismatch", {"turn_id": utid}
        if iid == INTENT_TRACE_CONCEPTS_V103:
            # turn_ref can be a turn_id or a numeric turn_index.
            tref = str(payload.get("turn_ref") or "")
            target_tid = ""
            if tref.isdigit():
                tix = int(tref)
                t = by_index.get(int(tix))
                if isinstance(t, dict):
                    target_tid = str(t.get("turn_id") or "")
            else:
                target_tid = str(tref)
            want = render_trace_concepts_text_v103(
                registry_prefix=dict(reg_prefix),
                concept_events=list(prefix_events),
                turns=turns,
                target_turn_id=str(target_tid),
            )
            if str(at.get("text") or "") != str(want):
                return False, "trace_concepts_text_mismatch", {"turn_id": utid}

    dd = dict(d0)
    dd["concept_events_total"] = int(len(list(concept_events)))
    dd["concept_chain_hash"] = str(compute_concept_chain_hash_v103(list(concept_events)))
    dd["concept_snapshot_sig"] = str(want_snap.get("snapshot_sig") or "")
    return True, "ok", dd

