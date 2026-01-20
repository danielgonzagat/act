from __future__ import annotations

import ast
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .conversation_v99 import render_explain_text_v99, verify_conversation_chain_v99
from .discourse_ledger_v100 import (
    compute_discourse_chain_hash_v100,
    fluency_metrics_v100,
    fluency_score_v100,
    render_discourse_text_v100,
    render_dossier_text_v100,
    sha256_text_v100,
    verify_discourse_event_sig_v100,
)
from .fragment_library_v100 import (
    base_fragments_v100,
    compute_fragment_chain_hash_v100,
    verify_fragment_event_sig_v100,
)
from .intent_grammar_v94 import INTENT_EXPLAIN_V94
from .intent_grammar_v98 import INTENT_DOSSIER_V98
from .intent_grammar_v100 import INTENT_DISCOURSE_V100


def _round6(x: Any) -> float:
    try:
        return float(round(float(x), 6))
    except Exception:
        return 0.0


def _canon_str_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for x in items:
        if isinstance(x, str) and x:
            out.append(str(x))
    return sorted(set(out))


def render_explain_text_v100(plan: Dict[str, Any]) -> str:
    """
    Deterministic EXPLAIN (v100): extends v99 with discourse/fluency fields.
    No rationale beyond plan fields.
    """
    base = render_explain_text_v99(plan)

    disc = plan.get("discourse") if isinstance(plan.get("discourse"), dict) else {}
    sel_id = str(disc.get("selected_candidate_id") or "")
    sel_sha = str(disc.get("selected_text_sha256") or "")
    sel_score = _round6(disc.get("selected_fluency_score"))
    sel_frags = _canon_str_list(disc.get("selected_fragment_ids"))
    line_d1 = (
        "DISCOURSE: "
        + f"selected={sel_id} sha={sel_sha} score={sel_score:.6f} fragments={canonical_json_dumps(sel_frags)}"
    )

    topk = disc.get("candidates_topk") if isinstance(disc.get("candidates_topk"), list) else []
    parts: List[str] = []
    for i, it in enumerate(topk[:5], start=1):
        if not isinstance(it, dict):
            continue
        cid = str(it.get("candidate_id") or "")
        sha = str(it.get("text_sha256") or "")
        sc = _round6(it.get("fluency_score"))
        m = it.get("fluency_metrics") if isinstance(it.get("fluency_metrics"), dict) else {}
        rep = _round6(m.get("trigram_repeat_ratio"))
        words = int(m.get("words") or 0)
        parts.append(f"{i}){cid} score={sc:.6f} sha={sha} tri_rep={rep:.6f} words={words}")
    line_d2 = "DISCOURSE_TOPK: " + "; ".join(parts) if parts else "DISCOURSE_TOPK:"

    return "\n".join([str(base), line_d1, line_d2])


FORBIDDEN_IMPORTS_V100 = {
    "torch",
    "tensorflow",
    "jax",
    "transformers",
    "openai",
    "sentencepiece",
}


def no_hybridization_check_v100(*, repo_root: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Fail-closed guardrail: detect forbidden ML/LLM dependencies by AST import scan.
    Scans only `atos_core/` and `scripts/` .py files (not results/patches).
    """
    roots = [os.path.join(str(repo_root), "atos_core"), os.path.join(str(repo_root), "scripts")]
    hits: List[Dict[str, str]] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in sorted(list(filenames)):
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, repo_root).replace(os.sep, "/")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        src = f.read()
                    tree = ast.parse(src, filename=rel)
                except Exception:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            name = str(alias.name or "")
                            top = name.split(".")[0]
                            if top in FORBIDDEN_IMPORTS_V100:
                                hits.append({"file": rel, "import": name})
                    if isinstance(node, ast.ImportFrom):
                        mod = str(node.module or "")
                        top = mod.split(".")[0] if mod else ""
                        if top in FORBIDDEN_IMPORTS_V100:
                            hits.append({"file": rel, "import_from": mod})
    hits = sorted(hits, key=lambda d: (str(d.get("file") or ""), canonical_json_dumps(d)))
    if hits:
        return False, "forbidden_dependency_detected", {"hits": list(hits), "forbidden": sorted(list(FORBIDDEN_IMPORTS_V100))}
    return True, "ok", {"forbidden": sorted(list(FORBIDDEN_IMPORTS_V100))}


def verify_conversation_chain_v100(
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
    tail_k: int,
    repo_root: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    V100 invariants:
      - All V99 invariants must hold (backwards compatible).
      - discourse_events.jsonl: signature chain ok; selected candidate matches assistant text; best-by-score selection.
      - fragment_events: signature chain ok; fragment usage events match selected candidates; fragment ids exist.
      - EXPLAIN/DOSSIER/DISCOURSE outputs are deterministic renderers.
      - no-hybridization guardrail passes.
    """
    ok0, reason0, details0 = verify_conversation_chain_v99(
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
        tail_k=int(tail_k),
    )
    if not ok0:
        return False, str(reason0), dict(details0)

    ok_h, reason_h, details_h = no_hybridization_check_v100(repo_root=str(repo_root))
    if not ok_h:
        return False, str(reason_h), dict(details_h)

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
    plan_user_indices: List[int] = []
    for p in action_plans:
        if not isinstance(p, dict):
            continue
        utid = str(p.get("user_turn_id") or "")
        if utid:
            plans_by_user_turn_id[utid] = dict(p)
            try:
                plan_user_indices.append(int(p.get("user_turn_index", -1)))
            except Exception:
                pass
    if plan_user_indices != sorted(plan_user_indices):
        return False, "plans_not_sorted_by_user_turn_index", {}

    # Verify discourse event sig chain and selection correctness.
    prev_sig = ""
    prev_state_after: Optional[Dict[str, Any]] = None
    for i, ev in enumerate(list(discourse_events)):
        if not isinstance(ev, dict):
            continue
        ok_ev, rreason, rdetails = verify_discourse_event_sig_v100(dict(ev))
        if not ok_ev:
            return False, str(rreason), dict(rdetails)
        if str(ev.get("prev_event_sig") or "") != str(prev_sig):
            return False, "discourse_prev_event_sig_mismatch", {"index": int(i)}
        prev_sig = str(ev.get("event_sig") or "")

        before = ev.get("discourse_state_before") if isinstance(ev.get("discourse_state_before"), dict) else {}
        after = ev.get("discourse_state_after") if isinstance(ev.get("discourse_state_after"), dict) else {}
        if prev_state_after is not None and dict(before) != dict(prev_state_after):
            return False, "discourse_state_chain_mismatch", {"index": int(i)}
        prev_state_after = dict(after)

        turn_id = str(ev.get("turn_id") or "")
        at = by_turn_id.get(turn_id)
        if not isinstance(at, dict):
            return False, "discourse_turn_missing", {"turn_id": turn_id}
        if str(at.get("role") or "") != "assistant":
            return False, "discourse_turn_not_assistant", {"turn_id": turn_id}

        sel_id = str(ev.get("selected_candidate_id") or "")
        cands = ev.get("candidates_topk") if isinstance(ev.get("candidates_topk"), list) else []
        cand_by_id: Dict[str, Dict[str, Any]] = {}
        for c in cands:
            if isinstance(c, dict) and str(c.get("candidate_id") or ""):
                cand_by_id[str(c.get("candidate_id") or "")] = dict(c)
        if sel_id not in cand_by_id:
            return False, "selected_candidate_missing", {"turn_id": turn_id}
        sel = cand_by_id[sel_id]
        at_text = str(at.get("text") or "")
        at_sha = sha256_text_v100(at_text)
        if str(sel.get("text_sha256") or "") != str(at_sha):
            return False, "selected_text_sha_mismatch", {"turn_id": turn_id}

        # Recompute score + enforce sorted candidates and best selection.
        scored: List[Dict[str, Any]] = []
        for c in cands:
            if not isinstance(c, dict):
                continue
            text = str(c.get("text") or "")
            m = fluency_metrics_v100(text=text, response_kind=str((c.get("fluency_metrics") or {}).get("response_kind") or ""))
            s = fluency_score_v100(metrics=dict(m))
            if float(round(float(s), 6)) != float(round(float(c.get("fluency_score") or 0.0), 6)):
                return False, "fluency_score_mismatch", {"turn_id": turn_id, "candidate_id": str(c.get("candidate_id") or "")}
            scored.append(dict(c))

        def _rk(d: Dict[str, Any]) -> Tuple[Any, ...]:
            m2 = d.get("fluency_metrics") if isinstance(d.get("fluency_metrics"), dict) else {}
            return (
                -float(d.get("fluency_score") or 0.0),
                int(m2.get("words") or 0),
                int(m2.get("chars") or 0),
                str(d.get("text_sha256") or ""),
                str(d.get("candidate_id") or ""),
            )

        scored_sorted = sorted(list(scored), key=_rk)
        got_ids = [str(x.get("candidate_id") or "") for x in cands if isinstance(x, dict)]
        want_ids = [str(x.get("candidate_id") or "") for x in scored_sorted]
        if got_ids != want_ids:
            return False, "discourse_candidates_not_sorted", {"turn_id": turn_id}
        if want_ids and str(want_ids[0]) != str(sel_id):
            return False, "discourse_selected_not_best", {"turn_id": turn_id}

    # Verify fragment event chain and referential integrity.
    frag_defs = base_fragments_v100()
    frag_ids = {str(f.get("fragment_id") or "") for f in frag_defs if isinstance(f, dict)}
    frag_prev = ""
    for i, ev in enumerate(list(fragment_events)):
        if not isinstance(ev, dict):
            continue
        ok_ev, rreason, rdetails = verify_fragment_event_sig_v100(dict(ev))
        if not ok_ev:
            return False, str(rreason), dict(rdetails)
        if str(ev.get("prev_event_sig") or "") != str(frag_prev):
            return False, "fragment_prev_event_sig_mismatch", {"index": int(i)}
        frag_prev = str(ev.get("event_sig") or "")
        fid = str(ev.get("fragment_id") or "")
        if fid and fid not in frag_ids:
            return False, "unknown_fragment_id", {"fragment_id": fid}

    # Cross-check: if selected candidate used fragment_ids, they appear in plan provenance and in fragment USE events.
    frag_use_index: Dict[Tuple[str, str, str], bool] = {}
    for ev in fragment_events:
        if not isinstance(ev, dict):
            continue
        if str(ev.get("event_kind") or "") != "USE":
            continue
        key = (str(ev.get("turn_id") or ""), str(ev.get("candidate_id") or ""), str(ev.get("fragment_id") or ""))
        frag_use_index[key] = True

    for ev in discourse_events:
        if not isinstance(ev, dict):
            continue
        turn_id = str(ev.get("turn_id") or "")
        at = by_turn_id.get(turn_id, {})
        try:
            aidx = int(at.get("turn_index", -1))
        except Exception:
            aidx = -1
        ut = by_index.get(aidx - 1) if aidx > 0 else None
        if not isinstance(ut, dict):
            continue
        utid = str(ut.get("turn_id") or "")
        plan = plans_by_user_turn_id.get(utid)
        if not isinstance(plan, dict):
            continue
        sel_id = str(ev.get("selected_candidate_id") or "")
        cands = ev.get("candidates_topk") if isinstance(ev.get("candidates_topk"), list) else []
        sel: Optional[Dict[str, Any]] = None
        for c in cands:
            if isinstance(c, dict) and str(c.get("candidate_id") or "") == sel_id:
                sel = dict(c)
                break
        if sel is None:
            continue
        frags = _canon_str_list(sel.get("fragment_ids"))
        prov = plan.get("provenance") if isinstance(plan.get("provenance"), dict) else {}
        prov_frags = _canon_str_list(prov.get("fragment_ids_used"))
        if frags != prov_frags:
            return False, "plan_fragment_ids_mismatch", {"user_turn_id": utid}
        for fid in frags:
            if not frag_use_index.get((turn_id, sel_id, fid), False):
                return False, "missing_fragment_use_event", {"turn_id": turn_id, "fragment_id": fid}

    # Cross-check deterministic renderers for DISCOURSE and DOSSIER.
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
        if intent_id == INTENT_DISCOURSE_V100:
            # Find corresponding discourse event for this assistant turn.
            atid = str(at.get("turn_id") or "")
            want_state: Optional[Dict[str, Any]] = None
            for dev in discourse_events:
                if isinstance(dev, dict) and str(dev.get("turn_id") or "") == atid:
                    want_state = dev.get("discourse_state_before") if isinstance(dev.get("discourse_state_before"), dict) else {}
                    break
            if want_state is None:
                return False, "discourse_event_missing_for_discourse_turn", {"turn_id": utid}
            want = render_discourse_text_v100(dict(want_state))
            if str(at.get("text") or "") != str(want):
                return False, "discourse_text_mismatch", {"turn_id": utid}
        if intent_id == INTENT_DOSSIER_V98:
            want2 = render_dossier_text_v100()
            if str(at.get("text") or "") != str(want2):
                return False, "dossier_text_mismatch", {"turn_id": utid}
        if intent_id == INTENT_EXPLAIN_V94:
            # explain must match v100 renderer of the last non-explain plan.
            uidx = int(ut.get("turn_index") or 0)
            best: Optional[Dict[str, Any]] = None
            for p in action_plans:
                if not isinstance(p, dict):
                    continue
                try:
                    pi = int(p.get("user_turn_index", -1))
                except Exception:
                    pi = -1
                if pi < 0 or pi >= int(uidx):
                    continue
                if str(p.get("intent_id") or "") == INTENT_EXPLAIN_V94:
                    continue
                if best is None:
                    best = dict(p)
                else:
                    try:
                        if int(best.get("user_turn_index", -1)) < pi:
                            best = dict(p)
                    except Exception:
                        best = dict(p)
            if best is None:
                return False, "missing_explainable_plan_before_explain", {"turn_id": utid}
            want3 = render_explain_text_v100(best)
            if str(at.get("text") or "") != str(want3):
                return False, "explain_text_mismatch", {"turn_id": utid}

    dd = dict(details0)
    dd.update(dict(details_h))
    dd["discourse_events_total"] = int(len(discourse_events))
    dd["discourse_chain_hash"] = str(compute_discourse_chain_hash_v100(discourse_events))
    dd["fragment_events_total"] = int(len(fragment_events))
    dd["fragment_chain_hash"] = str(compute_fragment_chain_hash_v100(fragment_events))
    return True, "ok", dd
