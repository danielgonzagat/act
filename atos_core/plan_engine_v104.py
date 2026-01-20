from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


PLAN_KIND_DIRECT_ANSWER_V104 = "direct_answer"
PLAN_KIND_ASK_CLARIFY_V104 = "ask_clarify"
PLAN_KIND_PROPOSE_STEPS_V104 = "propose_steps"
PLAN_KIND_COMPARE_OPTIONS_V104 = "compare_options"
PLAN_KIND_SUMMARIZE_CONFIRM_V104 = "summarize_and_confirm"
PLAN_KIND_REFUSE_SAFE_V104 = "refuse_safe"


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


def _candidate_id_v104(body_sem_id: Dict[str, Any]) -> str:
    return f"plan_cand_v104_{_stable_hash_obj(body_sem_id)}"


def _topic_sig_v104(*, intent_id: str, slots: Dict[str, Any], user_text: str) -> str:
    # Topic signature: stable, does not include turn ids.
    body = {
        "schema_version": 104,
        "intent_id": str(intent_id or ""),
        "slots": {str(k): slots.get(k) for k in sorted(slots.keys(), key=str)} if isinstance(slots, dict) else {},
        # Include normalized user_text as a fallback so unknown intents still create a topic.
        "user_text": str(user_text or ""),
    }
    return _stable_hash_obj(body)


def _root_plan_id_v104(topic_sig: str) -> str:
    return f"plan_root_v104_{str(topic_sig)}"


def _concept_bonus_by_kind_v104(concept_hit_names: Sequence[str]) -> Dict[str, int]:
    """
    Deterministic mapping from concept hits to plan-kind bonuses.
    This is intentionally explicit and audit-friendly (no NLP heuristics).
    """
    names = set([str(n).strip().upper() for n in concept_hit_names if str(n).strip()])
    bonus: Dict[str, int] = {k: 0 for k in _ALL_PLAN_KINDS_V104()}
    if "USER_WANTS_PLAN" in names or "WANTS_PLAN" in names:
        bonus[PLAN_KIND_PROPOSE_STEPS_V104] += 40
    if "USER_WANTS_OPTIONS" in names or "WANTS_OPTIONS" in names or "OPTIONS" in names:
        bonus[PLAN_KIND_COMPARE_OPTIONS_V104] += 40
    if "USER_CONFUSED" in names or "CONFUSED" in names:
        bonus[PLAN_KIND_ASK_CLARIFY_V104] += 40
    return dict(bonus)


def _ALL_PLAN_KINDS_V104() -> List[str]:
    return [
        PLAN_KIND_DIRECT_ANSWER_V104,
        PLAN_KIND_ASK_CLARIFY_V104,
        PLAN_KIND_PROPOSE_STEPS_V104,
        PLAN_KIND_COMPARE_OPTIONS_V104,
        PLAN_KIND_SUMMARIZE_CONFIRM_V104,
        PLAN_KIND_REFUSE_SAFE_V104,
    ]


def _predicted_effects_v104(*, plan_kind: str, style_pref: Dict[str, Any], parse_ok: bool) -> Dict[str, int]:
    # Discrete (int) effects; deterministic.
    # 0..10 scale.
    pk = str(plan_kind or "")
    verbosity = str(style_pref.get("verbosity_preference") or "")
    wants_short = verbosity == "SHORT"
    wants_long = verbosity == "LONG"

    # defaults
    clarity = 5
    effort = 4
    risk = 2
    follow = 4
    vfit = 5

    if pk == PLAN_KIND_DIRECT_ANSWER_V104:
        clarity, effort, risk, follow = 6, 2, 3, 3
    elif pk == PLAN_KIND_ASK_CLARIFY_V104:
        clarity, effort, risk, follow = 9, 6, 1, 8
    elif pk == PLAN_KIND_PROPOSE_STEPS_V104:
        clarity, effort, risk, follow = 8, 5, 2, 4
    elif pk == PLAN_KIND_COMPARE_OPTIONS_V104:
        clarity, effort, risk, follow = 7, 5, 2, 5
    elif pk == PLAN_KIND_SUMMARIZE_CONFIRM_V104:
        clarity, effort, risk, follow = 7, 3, 1, 6
    elif pk == PLAN_KIND_REFUSE_SAFE_V104:
        clarity, effort, risk, follow = 3, 1, 1, 2

    if wants_short and pk in {PLAN_KIND_PROPOSE_STEPS_V104, PLAN_KIND_COMPARE_OPTIONS_V104}:
        vfit -= 2
    if wants_long and pk in {PLAN_KIND_PROPOSE_STEPS_V104}:
        vfit += 2
    if not bool(parse_ok) and pk == PLAN_KIND_DIRECT_ANSWER_V104:
        # If parse fails, direct answer is risky.
        risk += 3
        clarity -= 2

    vfit = max(0, min(10, int(vfit)))
    return {
        "clarity_gain": int(max(0, min(10, clarity))),
        "user_effort": int(max(0, min(10, effort))),
        "risk": int(max(0, min(10, risk))),
        "likely_followup": int(max(0, min(10, follow))),
        "verbosity_fit": int(vfit),
    }


def _score_candidate_v104(*, effects: Dict[str, int], bonus: int) -> Dict[str, int]:
    # Deterministic integer scoring; no learned weights.
    clarity = _canon_int(effects.get("clarity_gain"))
    effort = _canon_int(effects.get("user_effort"))
    risk = _canon_int(effects.get("risk"))
    vfit = _canon_int(effects.get("verbosity_fit"))
    follow = _canon_int(effects.get("likely_followup"))
    base = 100
    score = int(base + clarity * 10 + vfit * 3 - effort * 6 - risk * 12 - follow * 2 + int(bonus))
    return {
        "base": int(base),
        "bonus_concept": int(bonus),
        "clarity_gain_x10": int(clarity * 10),
        "verbosity_fit_x3": int(vfit * 3),
        "penalty_effort_x6": int(effort * 6),
        "penalty_risk_x12": int(risk * 12),
        "penalty_followup_x2": int(follow * 2),
        "total": int(score),
    }


def _candidate_sort_key_v104(c: Dict[str, Any]) -> Tuple[Any, ...]:
    sb = c.get("score_breakdown") if isinstance(c.get("score_breakdown"), dict) else {}
    total = int(sb.get("total") or 0)
    return (
        -int(total),
        str(c.get("plan_kind") or ""),
        str(c.get("candidate_id") or ""),
    )


def _is_missing_slots(parse: Dict[str, Any]) -> bool:
    ms = parse.get("missing_slots")
    if isinstance(ms, list) and ms:
        return True
    return False


def build_plan_candidates_v104(
    *,
    conversation_id: str,
    user_turn_id: str,
    user_turn_index: int,
    intent_id: str,
    parse_sig: str,
    parse: Dict[str, Any],
    user_text: str,
    concept_hit_names: Sequence[str],
    style_profile_dict: Dict[str, Any],
    active_plan_state_before: Dict[str, Any],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Build and select plan candidates (A/B/C).
    Returns a dict with stable schema:
      {
        "candidates_topk": [...],
        "selected": {...},
        "selection": {"method": "...", "notes": "..."},
        "active_plan_state_after": {...},
      }
    """
    iid = str(intent_id or "")
    parse_ok = bool(parse.get("parse_ok", False))
    missing_slots = _is_missing_slots(parse)

    slots = parse.get("slots") if isinstance(parse.get("slots"), dict) else {}
    topic_sig = _topic_sig_v104(intent_id=iid, slots=dict(slots), user_text=str(user_text))

    # Allow propose_steps only when we have a concrete target or plan-text path.
    has_plan_target = False
    if iid == "INTENT_PLAN_CREATE_V101":
        has_plan_target = bool(str(parse.get("target") or "").strip())

    bonuses = _concept_bonus_by_kind_v104(list(concept_hit_names))

    candidates: List[Dict[str, Any]] = []
    for pk in _ALL_PLAN_KINDS_V104():
        if pk == PLAN_KIND_PROPOSE_STEPS_V104 and not has_plan_target:
            # No concrete target => can't safely propose steps.
            continue

        # Basic gating: if missing slots, prefer clarify/confirm; still include other candidates for explainability.
        eff = _predicted_effects_v104(plan_kind=str(pk), style_pref=dict(style_profile_dict), parse_ok=bool(parse_ok and not missing_slots))
        bonus = int(bonuses.get(pk, 0))
        # Extra discrete bias from parse status.
        if missing_slots and pk == PLAN_KIND_ASK_CLARIFY_V104:
            bonus += 30
        if (not parse_ok) and pk == PLAN_KIND_REFUSE_SAFE_V104:
            bonus += 10

        sb = _score_candidate_v104(effects=dict(eff), bonus=int(bonus))
        body = {
            "schema_version": 104,
            "kind": "plan_candidate_v104",
            "plan_kind": str(pk),
            "renderer_id": "plan_engine_v104",
            "variant_id": "v104",
            "predicted_effects": dict(eff),
            "dependencies": {
                "concept_hit_names": _canon_str_list(list(concept_hit_names)),
                "style_profile_sig": str(style_profile_dict.get("style_profile_sig") or ""),
            },
            "score_breakdown": dict(sb),
            "notes": "score = base + clarity*10 + vfit*3 - effort*6 - risk*12 - follow*2 + concept_bonus",
        }
        cid = _candidate_id_v104(dict(body))
        candidates.append(dict(body, candidate_id=str(cid)))

    # Stable rank by score desc then kind/id.
    candidates_sorted = sorted(list(candidates), key=_candidate_sort_key_v104)
    top_k = int(max(1, min(int(top_k), 8)))
    candidates_topk = candidates_sorted[:top_k]
    chosen = dict(candidates_topk[0]) if candidates_topk else {"candidate_id": "", "plan_kind": PLAN_KIND_REFUSE_SAFE_V104}

    # Selection notes are deterministic (no randomness in V104).
    selection = {"method": "argmax", "notes": "argmax score_total; tie-break plan_kind asc; candidate_id asc"}

    # Update active plan state deterministically.
    active_before = dict(active_plan_state_before) if isinstance(active_plan_state_before, dict) else {}
    after = _update_active_plan_state_v104(
        before=active_before,
        topic_sig=str(topic_sig),
        selected_plan_kind=str(chosen.get("plan_kind") or ""),
        has_plan_target=bool(has_plan_target),
        plan_target=str(parse.get("target") or "").strip() if iid == "INTENT_PLAN_CREATE_V101" else "",
        user_turn_index=int(user_turn_index),
        conversation_id=str(conversation_id),
    )

    return {
        "candidates_topk": list(candidates_topk),
        "selected": dict(chosen),
        "selection": dict(selection),
        "active_plan_state_after": dict(after),
    }


def _update_active_plan_state_v104(
    *,
    before: Dict[str, Any],
    topic_sig: str,
    selected_plan_kind: str,
    has_plan_target: bool,
    plan_target: str,
    user_turn_index: int,
    conversation_id: str,
) -> Dict[str, Any]:
    """
    Minimal multi-turn persistence:
      - A root_plan_id is stable for a topic_sig.
      - step_index increments when the topic stays the same.
    """
    topic_sig = str(topic_sig or "")
    pk = str(selected_plan_kind or "")
    btopic = str(before.get("topic_sig") or "")
    same_topic = bool(topic_sig and btopic and topic_sig == btopic)

    if same_topic:
        root = str(before.get("root_plan_id") or _root_plan_id_v104(topic_sig))
        step_index = _canon_int(before.get("step_index")) + 1
        created_turn = _canon_int(before.get("created_turn_index"))
        steps = before.get("steps") if isinstance(before.get("steps"), list) else []
    else:
        root = _root_plan_id_v104(topic_sig)
        step_index = 0
        created_turn = int(user_turn_index)
        steps = []

    if pk == PLAN_KIND_PROPOSE_STEPS_V104 and has_plan_target and plan_target:
        # Deterministic 3-step skeleton (no external facts).
        steps = [
            f"Definir objetivo: {plan_target}",
            f"Executar ação principal para: {plan_target}",
            f"Verificar resultado de: {plan_target}",
        ]

    return {
        "schema_version": 104,
        "kind": "active_plan_state_v104",
        "conversation_id": str(conversation_id),
        "root_plan_id": str(root),
        "topic_sig": str(topic_sig),
        "plan_kind": str(pk),
        "step_index": int(step_index),
        "created_turn_index": int(created_turn),
        "last_turn_index": int(user_turn_index),
        "steps": [str(x) for x in steps if isinstance(x, str) and x],
        "is_active": bool(pk not in {PLAN_KIND_REFUSE_SAFE_V104}),
    }

