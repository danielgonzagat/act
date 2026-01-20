from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .flow_ledger_v108 import (
    FLOW_REPAIR_CLARIFY_PREFERENCE_V108,
    FLOW_REPAIR_CLARIFY_REFERENCE_V108,
    FLOW_REPAIR_NONE_V108,
    FLOW_REPAIR_REPHRASE_V108,
    FLOW_REPAIR_SUMMARIZE_CONFIRM_V108,
)


FLOW_THRESH_V108 = 70
FLOW_DELTA_ELITE_V108 = 3
FLOW_PENDING_AGE_TURNS_V108 = 2


DISCOURSE_ACT_OPEN_V108 = "OPEN"
DISCOURSE_ACT_ACK_V108 = "ACK"
DISCOURSE_ACT_ANSWER_V108 = "ANSWER"
DISCOURSE_ACT_CLARIFY_V108 = "CLARIFY"
DISCOURSE_ACT_REPAIR_V108 = "REPAIR"
DISCOURSE_ACT_PIVOT_SOFT_V108 = "PIVOT_SOFT"
DISCOURSE_ACT_SUMMARY_V108 = "SUMMARY"
DISCOURSE_ACT_NEXT_STEP_V108 = "NEXT_STEP"
DISCOURSE_ACT_CLOSE_V108 = "CLOSE"

DISCOURSE_ACTS_V108 = {
    DISCOURSE_ACT_OPEN_V108,
    DISCOURSE_ACT_ACK_V108,
    DISCOURSE_ACT_ANSWER_V108,
    DISCOURSE_ACT_CLARIFY_V108,
    DISCOURSE_ACT_REPAIR_V108,
    DISCOURSE_ACT_PIVOT_SOFT_V108,
    DISCOURSE_ACT_SUMMARY_V108,
    DISCOURSE_ACT_NEXT_STEP_V108,
    DISCOURSE_ACT_CLOSE_V108,
}


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _tokens(text: str) -> List[str]:
    s = str(text or "").lower()
    s = re.sub(r"[^a-z0-9_\\s]", " ", s)
    return [t for t in s.split() if t]


def _contains_any(text: str, words: Sequence[str]) -> bool:
    s = str(text or "").lower()
    for w in words:
        if w and w in s:
            return True
    return False


def discourse_plan_v108(*, objective_kind: str, pending_questions_active: Sequence[str], need_pivot: bool, need_ack: bool) -> List[str]:
    acts: List[str] = []
    if need_pivot:
        acts.append(DISCOURSE_ACT_PIVOT_SOFT_V108)
    if need_ack:
        acts.append(DISCOURSE_ACT_ACK_V108)
    if pending_questions_active:
        # Must handle clarifications first.
        acts.append(DISCOURSE_ACT_CLARIFY_V108)
    if str(objective_kind) in {"COMM_ASK_CLARIFY", "COMM_CONFIRM"}:
        acts.append(DISCOURSE_ACT_CLARIFY_V108)
    elif str(objective_kind) in {"COMM_SUMMARIZE"}:
        acts.append(DISCOURSE_ACT_SUMMARY_V108)
    elif str(objective_kind) in {"COMM_END"}:
        acts.append(DISCOURSE_ACT_CLOSE_V108)
    else:
        acts.append(DISCOURSE_ACT_ANSWER_V108)
    # deterministic de-dup keeping order
    out: List[str] = []
    seen = set()
    for a in acts:
        if a in DISCOURSE_ACTS_V108 and a not in seen:
            out.append(a)
            seen.add(a)
    return list(out)


def _verbosity_mode_v108(*, style_profile_dict: Dict[str, Any]) -> str:
    v = str((style_profile_dict or {}).get("verbosity_preference") or "").upper()
    if v == "SHORT":
        return "BRIEF"
    if v == "LONG":
        return "DETAILED"
    return "NORMAL"


def _tone_mode_v108(*, style_profile_dict: Dict[str, Any]) -> str:
    t = str((style_profile_dict or {}).get("tone_preference") or "").upper()
    if t == "FORMAL":
        return "CALM"
    if t == "INFORMAL":
        return "FRIENDLY"
    return "DIRECT"


def _pending_age_max_v108(pending_questions_active: Sequence[str], *, current_user_turn_index: int) -> int:
    """
    Deterministic age for pq ids that follow pq_v107_<turn_index>.
    Unknown formats count as age 0.
    """
    ages: List[int] = []
    for qid in pending_questions_active:
        s = str(qid or "")
        if not s:
            continue
        m = re.search(r"_(\\d+)$", s)
        if not m:
            ages.append(0)
            continue
        try:
            born = int(m.group(1))
        except Exception:
            born = int(current_user_turn_index)
        ages.append(max(0, int(current_user_turn_index) - int(born)))
    return max(ages) if ages else 0


def compute_flow_metrics_v108(
    *,
    candidate_text: str,
    user_text: str,
    objective_kind: str,
    binding_status: str,
    user_intent_act_v107: Dict[str, Any],
    pending_questions_active: Sequence[str],
    regime_prev: str,
    regime_next: str,
    style_profile_dict: Dict[str, Any],
    recent_assistant_texts: Sequence[str],
    recent_phrase_hashes: Sequence[str],
    recent_discourse_acts: Sequence[str],
    current_user_turn_index: int,
) -> Dict[str, Any]:
    text = str(candidate_text or "")
    toks = _tokens(text)
    words = len(toks)
    chars = len(text)

    verbosity_mode = _verbosity_mode_v108(style_profile_dict=dict(style_profile_dict))
    tone_mode = _tone_mode_v108(style_profile_dict=dict(style_profile_dict))

    # Repetition signals.
    cand_hash = sha256_hex(text.encode("utf-8"))
    prev_hash = str(recent_phrase_hashes[0]) if recent_phrase_hashes else ""
    repetition_exact = bool(prev_hash and cand_hash and cand_hash == prev_hash)

    # Simple ngram repetition: if candidate equals any of last 3 assistant texts.
    rep_loop = repetition_exact or bool(text and any(text == str(p) for p in list(recent_assistant_texts)[:3] if str(p)))
    repetition_score = 100 if rep_loop else 0

    # Topic/regime shift needs a soft pivot marker.
    need_pivot = bool(regime_prev and regime_next and str(regime_prev) != str(regime_next))
    has_pivot_marker = bool(re.search(r"\b(sobre|quanto a|mudando de assunto|indo pro ponto)\b", text.lower()))
    abrupt_topic_shift = bool(need_pivot and not has_pivot_marker and str(objective_kind) in {"COMM_RESPOND", "COMM_SUMMARIZE"})

    # If user signals confusion/frustration, require an ACK.
    user_text_l = str(user_text or "").lower()
    user_confused = bool(re.search(r"(não entendi|nao entendi|\?\?|não entendo|nao entendo|explica|explique)", user_text_l))
    need_ack = bool(user_confused)
    has_ack = bool(re.search(r"\b(entendi|ok|certo|beleza|vamos)\b", text.lower()))
    too_dry = bool(need_ack and not has_ack and chars <= 200 and str(objective_kind) in {"COMM_RESPOND", "COMM_CONFIRM"})

    # Length fit against verbosity mode.
    too_long = False
    too_short = False
    if verbosity_mode == "BRIEF":
        too_long = words >= 40
        too_short = words <= 2
    elif verbosity_mode == "DETAILED":
        too_short = words <= 6
    else:
        too_long = words >= 80
        too_short = words <= 2

    # Technical mismatch (very simple): user asked short/simple and response is code-like.
    user_wants_simple = bool(re.search(r"(curto|resumo|simples)", user_text_l))
    has_code = bool(re.search(r"[`{}\\[\\]=]", text))
    technical_mismatch = bool(user_wants_simple and has_code)

    # Reference handling: if binding status is ambiguous/miss, the reference is unresolved
    # until the user clarifies; treat as a repair turn (progress gating applies).
    unresolved_reference = bool(binding_status in {"AMBIGUOUS", "MISS"})

    # Pending question aging: if the same pending question lingers too long, force summarize+confirm.
    pending_age_max = _pending_age_max_v108(list(pending_questions_active), current_user_turn_index=int(current_user_turn_index))
    pending_aged = bool(pending_age_max > int(FLOW_PENDING_AGE_TURNS_V108))

    flags = {
        "TOO_DRY": bool(too_dry),
        "TOO_LONG_FOR_CONTEXT": bool(too_long),
        "TOO_SHORT_FOR_CONTEXT": bool(too_short),
        "ABRUPT_TOPIC_SHIFT": bool(abrupt_topic_shift),
        "REPETITION_LOOP": bool(rep_loop),
        "TECHNICAL_MISMATCH": bool(technical_mismatch),
        "UNRESOLVED_REFERENCE": bool(unresolved_reference),
        "PENDING_QUESTION_AGED": bool(pending_aged),
    }

    # Components 0..100
    length_fit = 100
    transition_fit = 100
    ack_fit = 100
    repetition_penalty = 0
    memory_alignment = 100
    reference_handling = 100

    if too_long:
        length_fit = 40 if verbosity_mode == "BRIEF" else 70
    if too_short:
        # In DETAILED mode, "too short" should penalize selection but not necessarily block progress.
        length_fit = min(length_fit, 75) if verbosity_mode == "DETAILED" else min(length_fit, 60)
    if abrupt_topic_shift:
        transition_fit = 40
    if need_ack and not has_ack:
        ack_fit = 40
    if rep_loop:
        repetition_penalty = 60
    if pending_questions_active and "?" not in text and str(objective_kind) in {"COMM_RESPOND", "COMM_SUMMARIZE"}:
        memory_alignment = 40
    if binding_status in {"AMBIGUOUS", "MISS"}:
        reference_handling = 0 if "?" not in text else 70
    if pending_aged:
        memory_alignment = min(memory_alignment, 30)

    # Score is the minimum of the structural components, minus repetition.
    base = min(length_fit, transition_fit, ack_fit, memory_alignment, reference_handling)
    base = max(0, min(100, int(base)))
    score = max(0, min(100, int(base - repetition_penalty)))

    # Repair decision (deterministic).
    repair = FLOW_REPAIR_NONE_V108
    hits: List[str] = []
    if score < int(FLOW_THRESH_V108) or any(bool(flags.get(k)) for k in ["ABRUPT_TOPIC_SHIFT", "REPETITION_LOOP", "TECHNICAL_MISMATCH", "UNRESOLVED_REFERENCE"]):
        hits.append("S4")
    if bool(flags.get("UNRESOLVED_REFERENCE")) or bool(flags.get("PENDING_QUESTION_AGED")):
        hits.append("S5")

    if bool(flags.get("UNRESOLVED_REFERENCE")):
        repair = FLOW_REPAIR_CLARIFY_REFERENCE_V108
    elif bool(flags.get("PENDING_QUESTION_AGED")):
        repair = FLOW_REPAIR_SUMMARIZE_CONFIRM_V108
    elif bool(flags.get("REPETITION_LOOP")) or bool(flags.get("TECHNICAL_MISMATCH")):
        repair = FLOW_REPAIR_REPHRASE_V108
    elif bool(flags.get("ABRUPT_TOPIC_SHIFT")):
        repair = FLOW_REPAIR_SUMMARIZE_CONFIRM_V108
    elif score < int(FLOW_THRESH_V108):
        repair = FLOW_REPAIR_CLARIFY_PREFERENCE_V108

    # Progress allowed only if no critical flags and score above threshold.
    critical = bool(
        score < int(FLOW_THRESH_V108)
        or bool(flags.get("ABRUPT_TOPIC_SHIFT"))
        or bool(flags.get("REPETITION_LOOP"))
        or bool(flags.get("TECHNICAL_MISMATCH"))
        or bool(flags.get("UNRESOLVED_REFERENCE"))
        or bool(flags.get("PENDING_QUESTION_AGED"))
    )
    progress_allowed = not critical

    return {
        "flow_score_v108": int(score),
        "components": {
            "length_fit": int(length_fit),
            "transition_fit": int(transition_fit),
            "ack_fit": int(ack_fit),
            "memory_alignment": int(memory_alignment),
            "reference_handling": int(reference_handling),
            "repetition_penalty": int(repetition_penalty),
        },
        "flags": dict(flags),
        "flags_digest": _stable_hash_obj(dict(flags)),
        "repetition_score": int(repetition_score),
        "verbosity_mode_v108": str(verbosity_mode),
        "tone_mode_v108": str(tone_mode),
        "discourse_plan_v108": discourse_plan_v108(
            objective_kind=str(objective_kind),
            pending_questions_active=list(pending_questions_active),
            need_pivot=bool(need_pivot),
            need_ack=bool(need_ack),
        ),
        "repair_action_v108": str(repair),
        "progress_allowed_v108": bool(progress_allowed),
        "survival_rule_hits_v108": sorted(set([str(x) for x in hits if str(x)]), key=str),
    }


def select_candidate_dsc_v108(
    *,
    evaluated_candidates: Sequence[Dict[str, Any]],
    seed: int,
    prev_flow_event_sig: str,
    selection_salt: str,
    delta: int = FLOW_DELTA_ELITE_V108,
) -> Dict[str, Any]:
    """
    Deterministic Soft Choice (DSC):
      - elite = candidates within (best_score - delta)
      - choose by stable hash of (prev_sig + elite hashes + seed + salt)
    """

    def _rk(c: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            -int(c.get("score_base") or 0),
            int(c.get("len_tokens") or 0),
            str(c.get("text_sha256") or ""),
            str(c.get("candidate_id") or c.get("cand_id") or ""),
        )

    cands = [dict(c) for c in evaluated_candidates if isinstance(c, dict)]
    cands_sorted = sorted(cands, key=_rk)
    if not cands_sorted:
        return {}
    best = int(cands_sorted[0].get("score_base") or 0)
    elite = [c for c in cands_sorted if int(c.get("score_base") or 0) >= int(best - int(delta))]
    elite = elite if elite else [dict(cands_sorted[0])]
    hashes = [str(c.get("candidate_hash") or c.get("text_sha256") or "") for c in elite]
    pick_sig = sha256_hex((str(prev_flow_event_sig or "") + canonical_json_dumps(hashes) + str(int(seed)) + str(selection_salt or "")).encode("utf-8"))
    pick = int(pick_sig[:8], 16) if pick_sig else 0
    idx = int(pick % max(1, len(elite)))
    chosen = dict(elite[idx])
    chosen["selection"] = {"method": "DSC", "delta": int(delta), "elite_n": int(len(elite)), "soft_index": int(idx), "pick_sig": str(pick_sig)}
    return chosen
