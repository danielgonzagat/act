from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .act import canonical_json_dumps, sha256_hex
from .plan_ledger_v104 import compute_plan_chain_hash_v104


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _canon_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _canon_str_list(xs: Any) -> List[str]:
    if not isinstance(xs, list):
        return []
    out = [str(x) for x in xs if isinstance(x, str) and str(x)]
    out = sorted(set(out), key=str)
    return list(out)


def _canon_event_for_index(ev: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(ev)
    # Remove any outer-chain fields if present.
    d.pop("prev_hash", None)
    d.pop("entry_hash", None)
    return d


def fold_plan_ledger_v104(*, plan_events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    events = [dict(_canon_event_for_index(ev)) for ev in plan_events if isinstance(ev, dict)]
    by_user_turn_index: Dict[int, Dict[str, Any]] = {}
    active_state: Dict[str, Any] = {}
    for ev in events:
        uidx = _canon_int(ev.get("user_turn_index"))
        if uidx >= 0 and uidx not in by_user_turn_index:
            by_user_turn_index[int(uidx)] = dict(ev)
        # Last active_plan_state_after wins.
        aps = ev.get("active_plan_state_after") if isinstance(ev.get("active_plan_state_after"), dict) else {}
        if aps:
            active_state = dict(aps)
    chain_hash = compute_plan_chain_hash_v104(list(events))
    return {
        "schema_version": 104,
        "kind": "plan_registry_v104",
        "plan_chain_hash": str(chain_hash),
        "plan_events_total": int(len(events)),
        "active_plan_state": dict(active_state) if isinstance(active_state, dict) else {},
        "plans_by_user_turn_index": {str(k): dict(v) for k, v in sorted(by_user_turn_index.items(), key=lambda kv: int(kv[0]))},
    }


def plan_registry_snapshot_v104(*, plan_events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    reg = fold_plan_ledger_v104(plan_events=list(plan_events))
    sem = dict(reg)
    sem.pop("snapshot_sig", None)
    sig = _stable_hash_obj(sem)
    return dict(sem, snapshot_sig=str(sig))


def _parse_turn_ref(ref: str) -> Tuple[str, Optional[int]]:
    s = str(ref or "").strip()
    if not s:
        return "missing", None
    try:
        return "index", int(s)
    except Exception:
        return "unknown", None


def lookup_plan_event_v104(*, plan_events: Sequence[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
    kind, idx = _parse_turn_ref(str(query))
    if kind != "index" or idx is None or idx < 0:
        return None
    for ev in plan_events:
        if not isinstance(ev, dict):
            continue
        if _canon_int(ev.get("user_turn_index")) == int(idx):
            return dict(ev)
    return None


def render_plans_text_v104(*, registry: Dict[str, Any]) -> str:
    aps = registry.get("active_plan_state") if isinstance(registry.get("active_plan_state"), dict) else {}
    if not aps or not bool(aps.get("is_active", False)):
        return "PLANS: (none)"
    root = str(aps.get("root_plan_id") or "")
    pk = str(aps.get("plan_kind") or "")
    step_index = _canon_int(aps.get("step_index"))
    topic_sig = str(aps.get("topic_sig") or "")
    lines: List[str] = []
    lines.append(f"PLANS: active root={root} kind={pk} step_index={step_index}")
    lines.append(f"TOPIC_SIG: {topic_sig}")
    steps = aps.get("steps") if isinstance(aps.get("steps"), list) else []
    if steps:
        lines.append("STEPS:")
        for i, st in enumerate([str(x) for x in steps if isinstance(x, str) and x], start=1):
            lines.append(f"{i}) {st}")
    return "\n".join(lines)


def render_explain_plan_text_v104(*, plan_event: Dict[str, Any]) -> str:
    if not isinstance(plan_event, dict) or not plan_event:
        return "EXPLAIN_PLAN: not_found"
    uidx = _canon_int(plan_event.get("user_turn_index"))
    iid = str(plan_event.get("intent_id") or "")
    chosen_kind = str(plan_event.get("selected_plan_kind") or "")
    chosen_cid = str(plan_event.get("selected_candidate_id") or "")
    notes = str(plan_event.get("notes") or "")
    cands = plan_event.get("candidates_topk") if isinstance(plan_event.get("candidates_topk"), list) else []
    # Render A/B/C deterministically.
    parts: List[str] = []
    labels = ["A", "B", "C", "D", "E"]
    for i, c in enumerate(cands[:3]):
        if not isinstance(c, dict):
            continue
        sb = c.get("score_breakdown") if isinstance(c.get("score_breakdown"), dict) else {}
        total = _canon_int(sb.get("total"))
        pk = str(c.get("plan_kind") or "")
        cid = str(c.get("candidate_id") or "")
        parts.append(f"{labels[i]}) {pk} score={total} id={cid}")
    ranked = "; ".join(parts) if parts else "(none)"
    lines: List[str] = []
    lines.append(f"EXPLAIN_PLAN: user_turn_index={uidx} intent={iid} chosen={chosen_kind} ok=true")
    lines.append(f"RANKED: {ranked}")
    lines.append(f"CHOSEN: {chosen_kind} id={chosen_cid}")
    lines.append(f"NOTES: {notes}")
    hit_names = plan_event.get("concept_hit_names") if isinstance(plan_event.get("concept_hit_names"), list) else []
    lines.append("CONCEPT_HITS: " + (canonical_json_dumps(_canon_str_list(hit_names)) if hit_names else "[]"))
    return "\n".join(lines)


def render_trace_plans_text_v104(*, plan_event: Dict[str, Any]) -> str:
    if not isinstance(plan_event, dict) or not plan_event:
        return "TRACE_PLANS: not_found"
    # Stable JSON dump (canonical keys).
    body = dict(plan_event)
    return json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2)

