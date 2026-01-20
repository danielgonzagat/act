from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .concept_engine_v103 import match_csg_v103
from .intent_grammar_v92 import tokenize_user_text_v92
from .semantics_ledger_v109 import (
    SEM_REPAIR_ASK_EXAMPLE_V109,
    SEM_REPAIR_CLARIFY_V109,
    SEM_REPAIR_CONFIRM_MEANING_V109,
    SEM_REPAIR_NONE_V109,
    SEM_REPAIR_UPDATE_CONCEPT_V109,
)


SEMANTIC_THRESH_V109 = 70


FLAG_TAUGHT_CONCEPT_NOT_REGISTERED_V109 = "TAUGHT_CONCEPT_NOT_REGISTERED"
FLAG_TAUGHT_CONCEPT_NOT_REUSED_V109 = "TAUGHT_CONCEPT_NOT_REUSED"
FLAG_CONCEPT_MATCH_TOO_WEAK_V109 = "CONCEPT_MATCH_TOO_WEAK"
FLAG_CONTRADICTION_UNREPAIRED_V109 = "CONTRADICTION_UNREPAIRED"
FLAG_REQUIRES_CLARIFICATION_V109 = "REQUIRES_CLARIFICATION"


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _norm_tokens(text: str) -> List[str]:
    toks = tokenize_user_text_v92(str(text or ""))
    return [str(t) for t in toks if str(t)]


def _contains_tokens_seq(hay: Sequence[str], needle: Sequence[str]) -> bool:
    h = [str(x) for x in hay]
    n = [str(x) for x in needle]
    if not n:
        return False
    for i in range(0, max(0, len(h) - len(n) + 1)):
        if h[i : i + len(n)] == n:
            return True
    return False


def detect_concept_query_v109(*, user_text: str, concept_names: Sequence[str]) -> str:
    """
    Deterministic detector for "is this <CONCEPT>?" style queries.
    Returns a canonical concept name from concept_names, or "".
    """
    raw = str(user_text or "")
    toks = _norm_tokens(raw)
    if not toks:
        return ""

    # Require at least one query marker ("?" or "isso/this/is/é") to avoid
    # accidental name mentions being treated as queries.
    has_qmark = "?" in raw
    markers = {"isso", "isto", "this", "is", "eh", "e", "é", "classifique", "classify"}
    has_marker = any(t in markers for t in toks)
    if not (has_qmark or has_marker):
        return ""

    best = ""
    best_key = None
    for name in concept_names:
        nm = str(name or "").strip()
        if not nm:
            continue
        ntoks = _norm_tokens(nm)
        if not ntoks:
            continue
        if _contains_tokens_seq(toks, ntoks) or set(ntoks).issubset(set(toks)):
            # Prefer longer (more tokens), then lexicographically.
            key = (-len(ntoks), nm)
            if best_key is None or key < best_key:
                best_key = key
                best = nm
    return str(best)


def _extract_promise_action(tokens: Sequence[str]) -> str:
    """
    Deterministic extraction of a promise/commitment action tail.
    Ex: "eu prometo que vou fazer deploy" -> "fazer deploy"
    """
    toks = [str(t) for t in tokens if str(t)]
    if "prometo" not in toks:
        return ""
    try:
        i = toks.index("prometo")
    except Exception:
        return ""
    tail = toks[i + 1 :]
    # Find "vou" and capture from there.
    if "vou" in tail:
        j = tail.index("vou")
        tail = tail[j + 1 :]
    # Drop leading "que"
    if tail and tail[0] == "que":
        tail = tail[1:]
    # Keep small tail only.
    tail2 = [t for t in tail[:8] if t not in {"eu", "voce", "você"}]
    return " ".join(tail2).strip()


def _extract_negation_action(tokens: Sequence[str]) -> str:
    """
    Deterministic extraction of a negation of intended action.
    Ex: "nao vou fazer deploy" -> "fazer deploy"
    """
    toks = [str(t) for t in tokens if str(t)]
    if "nao" not in toks or "vou" not in toks:
        return ""
    try:
        i = toks.index("vou")
    except Exception:
        return ""
    tail = toks[i + 1 :]
    if tail and tail[0] == "que":
        tail = tail[1:]
    tail2 = [t for t in tail[:8] if t not in {"eu", "voce", "você"}]
    return " ".join(tail2).strip()


def detect_contradiction_v109(*, user_text: str, promise_action_active: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Minimal contradiction detector: if a prior promise action exists and user
    now negates "nao vou <same action>", raise contradiction.
    """
    toks = _norm_tokens(str(user_text or ""))
    neg = _extract_negation_action(toks)
    prom = str(promise_action_active or "")
    if not prom or not neg:
        return False, {}
    if prom == neg:
        return True, {"kind": "contradiction", "promise_action": prom, "negation_action": neg}
    # Accept a deterministic "starts-with" match for robustness.
    if prom and neg and (prom.startswith(neg) or neg.startswith(prom)):
        return True, {"kind": "contradiction", "promise_action": prom, "negation_action": neg}
    return False, {}


def build_semantic_override_v109(
    *,
    user_text: str,
    intent_id: str,
    objective_kind: str,
    concept_registry_v103: Dict[str, Any],
    promise_action_active: str,
) -> Dict[str, Any]:
    """
    Determines whether semantics should override the response (repair or label).
    Returns a dict with keys:
      override_ok: bool
      objective_kind: str
      planned_text: str
      decision: dict
      flags: dict[str,bool]
      semantic_score: int
      progress_allowed: bool
      repair_action: str
    """
    intent_now = str(intent_id or "")
    if intent_now in {"INTENT_TEACH_CONCEPT", "INTENT_CONCEPTS", "INTENT_EXPLAIN_CONCEPT", "INTENT_TRACE_CONCEPTS"}:
        return {"override_ok": False}

    # Only override on otherwise-unknown inputs to avoid changing stable DSL semantics.
    if str(intent_now) != "INTENT_UNKNOWN":
        return {"override_ok": False}

    reg = dict(concept_registry_v103) if isinstance(concept_registry_v103, dict) else {}
    c_by_id = reg.get("concepts_by_id") if isinstance(reg.get("concepts_by_id"), dict) else {}
    csv_by_id = reg.get("csv_by_id") if isinstance(reg.get("csv_by_id"), dict) else {}
    names: List[str] = []
    by_name: Dict[str, str] = {}
    for cid in sorted(c_by_id.keys(), key=str):
        csg = c_by_id.get(cid) if isinstance(c_by_id.get(cid), dict) else {}
        st = csv_by_id.get(cid) if isinstance(csv_by_id.get(cid), dict) else {}
        if str(st.get("status") or "ALIVE") != "ALIVE":
            continue
        nm = str(csg.get("name") or "")
        if nm:
            names.append(nm)
            by_name[nm.lower()] = str(cid)
    names = sorted(set([str(n) for n in names if str(n)]), key=str)

    query_name = detect_concept_query_v109(user_text=str(user_text), concept_names=list(names))

    toks = _norm_tokens(str(user_text or ""))
    promise_action = _extract_promise_action(toks)
    contradiction, contradiction_details = detect_contradiction_v109(user_text=str(user_text), promise_action_active=str(promise_action_active or ""))

    # Promise (opening) is not a failure: just record in decision for state.
    if promise_action and not contradiction:
        return {
            "override_ok": False,
            "decision_hint": {"kind": "promise_open", "promise_action_open": str(promise_action)},
        }

    if contradiction:
        msg = "Você disse que iria {a}, mas agora disse que não. Qual vale?".format(a=str(contradiction_details.get("promise_action") or ""))
        return {
            "override_ok": True,
            "objective_kind": "COMM_CONFIRM",
            "planned_text": str(msg),
            "decision": dict(contradiction_details),
            "flags": {FLAG_CONTRADICTION_UNREPAIRED_V109: True},
            "semantic_score": 0,
            "progress_allowed": False,
            "repair_action": SEM_REPAIR_CONFIRM_MEANING_V109,
        }

    if query_name:
        cid = ""
        for nm in names:
            if nm.lower() == query_name.lower():
                cid = str(by_name.get(nm.lower(), ""))
                break
        csg = c_by_id.get(cid) if cid and isinstance(c_by_id.get(cid), dict) else {}
        m = match_csg_v103(csg=dict(csg), text=str(user_text))
        matched = bool(m.get("matched", False))
        evidence = list(m.get("evidence") or [])
        decision = {"kind": "concept_query", "concept_name": str(query_name), "concept_id": str(cid), "matched": bool(matched), "evidence": list(evidence)}
        if matched:
            return {
                "override_ok": True,
                "objective_kind": "COMM_RESPOND",
                "planned_text": str(query_name),
                "decision": dict(decision),
                "flags": {},
                "semantic_score": 100,
                "progress_allowed": True,
                "repair_action": SEM_REPAIR_NONE_V109,
            }
        # Concept exists but match is not strong enough: ask for example/clarify.
        return {
            "override_ok": True,
            "objective_kind": "COMM_CONFIRM",
            "planned_text": "Preciso de um exemplo ou mais contexto para avaliar {n}.".format(n=str(query_name)),
            "decision": dict(decision),
            "flags": {FLAG_CONCEPT_MATCH_TOO_WEAK_V109: True, FLAG_REQUIRES_CLARIFICATION_V109: True},
            "semantic_score": 0,
            "progress_allowed": False,
            "repair_action": SEM_REPAIR_ASK_EXAMPLE_V109 if evidence else SEM_REPAIR_CLARIFY_V109,
        }

    return {"override_ok": False}


def compute_semantic_metrics_v109(
    *,
    candidate_text: str,
    user_text: str,
    semantic_override: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes candidate-specific semantic score/flags.
    If semantic_override is provided, uses its decision/flags.
    Otherwise returns neutral metrics (score=100).
    """
    if not isinstance(semantic_override, dict) or not semantic_override.get("decision"):
        return {
            "semantic_score_v109": 100,
            "flags_v109": {},
            "repair_action_v109": SEM_REPAIR_NONE_V109,
            "progress_allowed_v109": True,
            "decision": {"kind": "none"},
        }

    decision = semantic_override.get("decision") if isinstance(semantic_override.get("decision"), dict) else {}
    flags = semantic_override.get("flags") if isinstance(semantic_override.get("flags"), dict) else {}
    repair = str(semantic_override.get("repair_action") or "")
    progress_allowed = bool(semantic_override.get("progress_allowed", False))
    score = int(semantic_override.get("semantic_score") or 0)

    kind = str(decision.get("kind") or "")
    if kind == "concept_query" and bool(decision.get("matched", False)):
        name = str(decision.get("concept_name") or "")
        if name and name.lower() in str(candidate_text or "").lower():
            return {"semantic_score_v109": 100, "flags_v109": dict(flags), "repair_action_v109": str(repair), "progress_allowed_v109": bool(progress_allowed), "decision": dict(decision)}
        # Missing required label in candidate: treat as critical.
        ff = dict(flags)
        ff[FLAG_TAUGHT_CONCEPT_NOT_REUSED_V109] = True
        return {"semantic_score_v109": 0, "flags_v109": dict(ff), "repair_action_v109": SEM_REPAIR_UPDATE_CONCEPT_V109, "progress_allowed_v109": False, "decision": dict(decision)}

    if kind == "concept_query" and not bool(decision.get("matched", False)):
        # Candidate should ask for example/clarification deterministically.
        txt = str(candidate_text or "")
        ok_q = "?" in txt or "exemplo" in txt.lower() or "contexto" in txt.lower()
        if ok_q:
            return {"semantic_score_v109": 80, "flags_v109": dict(flags), "repair_action_v109": str(repair), "progress_allowed_v109": False, "decision": dict(decision)}
        ff = dict(flags)
        ff[FLAG_CONCEPT_MATCH_TOO_WEAK_V109] = True
        return {"semantic_score_v109": 0, "flags_v109": dict(ff), "repair_action_v109": str(repair or SEM_REPAIR_ASK_EXAMPLE_V109), "progress_allowed_v109": False, "decision": dict(decision)}

    if kind == "contradiction":
        txt = str(candidate_text or "")
        ok_q = "?" in txt
        if ok_q:
            return {"semantic_score_v109": 90, "flags_v109": dict(flags), "repair_action_v109": str(repair or SEM_REPAIR_CONFIRM_MEANING_V109), "progress_allowed_v109": False, "decision": dict(decision)}
        ff = dict(flags)
        ff[FLAG_CONTRADICTION_UNREPAIRED_V109] = True
        return {"semantic_score_v109": 0, "flags_v109": dict(ff), "repair_action_v109": SEM_REPAIR_CONFIRM_MEANING_V109, "progress_allowed_v109": False, "decision": dict(decision)}

    return {"semantic_score_v109": max(0, min(100, int(score))), "flags_v109": dict(flags), "repair_action_v109": str(repair), "progress_allowed_v109": bool(progress_allowed), "decision": dict(decision)}

