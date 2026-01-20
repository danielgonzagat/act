from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex

PRAGMATICS_THRESH_V107 = 60

# Regimes (MED) — discrete, deterministic.
REGIME_GREETING_V107 = "GREETING"
REGIME_TASK_QA_V107 = "TASK_QA"
REGIME_PLANNING_V107 = "PLANNING"
REGIME_INSTRUCTION_V107 = "INSTRUCTION"
REGIME_CHITCHAT_V107 = "CHITCHAT"
REGIME_REPAIR_V107 = "REPAIR"
REGIME_CLOSING_V107 = "CLOSING"
REGIME_META_V107 = "META"

# IntentActs (A-02) — discrete, deterministic.
INTENTACT_QUESTION_V107 = "QUESTION"
INTENTACT_REQUEST_V107 = "REQUEST"
INTENTACT_COMMAND_V107 = "COMMAND"
INTENTACT_ASSERTION_V107 = "ASSERTION"
INTENTACT_OPINION_V107 = "OPINION"
INTENTACT_PROMISE_ATTEMPT_V107 = "PROMISE_ATTEMPT"
INTENTACT_COMPLAINT_V107 = "COMPLAINT"
INTENTACT_META_SYSTEM_V107 = "META_SYSTEM"
INTENTACT_CHITCHAT_V107 = "CHITCHAT"
INTENTACT_UNKNOWN_V107 = "UNKNOWN"

# Assistant IntentActs — derived from metadata (not parsing assistant text).
ASSISTANTACT_ANSWER_V107 = "ANSWER"
ASSISTANTACT_CLARIFY_V107 = "CLARIFY"
ASSISTANTACT_REFUSE_V107 = "REFUSE"
ASSISTANTACT_CONFIRM_V107 = "CONFIRM"
ASSISTANTACT_SUMMARIZE_V107 = "SUMMARIZE"
ASSISTANTACT_CLOSE_V107 = "CLOSE"
ASSISTANTACT_CORRECT_V107 = "CORRECT"
ASSISTANTACT_OTHER_V107 = "OTHER"

CONFIDENCE_HIGH_V107 = "HIGH"
CONFIDENCE_MEDIUM_V107 = "MEDIUM"
CONFIDENCE_LOW_V107 = "LOW"

# Repair kinds (align with V106 where possible)
REPAIR_NONE_V107 = ""
REPAIR_ASK_CLARIFY_V107 = "ask_clarify"
REPAIR_SUMMARIZE_CONFIRM_V107 = "summarize_confirm"
REPAIR_RESTATE_OFFER_OPTIONS_V107 = "restate_offer_options"


def _strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _norm(text: str) -> str:
    return _strip_accents(str(text or "")).lower()


def tokenize_v107(text: str) -> List[str]:
    s = _norm(text)
    s = re.sub(r"[^a-z0-9_\\s\\?\\!\\+\\-\\=]", " ", s)
    toks = [t for t in s.split() if t]
    return list(toks)


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _contains_any(toks: Sequence[str], words: Sequence[str]) -> bool:
    ws = set([str(w) for w in words if str(w)])
    for t in toks:
        if str(t) in ws:
            return True
    return False


def _contains_deictic_pronoun_v107(toks: Sequence[str]) -> Optional[str]:
    # Deterministic, minimal deictics (PT/EN)
    deictics = ["isso", "aquilo", "isto", "este", "essa", "esse", "that", "this", "it"]
    for d in deictics:
        if d in set([str(t) for t in toks]):
            return str(d)
    return None


def classify_user_intent_act_v107(*, user_text: str, parse_intent_id: str) -> Dict[str, Any]:
    toks = tokenize_v107(user_text)
    raw = str(user_text or "")
    pid = str(parse_intent_id or "")

    # Strong signals from the existing deterministic parser.
    if pid and pid != "INTENT_UNKNOWN":
        return {"kind": INTENTACT_COMMAND_V107, "slots": {"parse_intent_id": pid}, "confidence_tag": CONFIDENCE_HIGH_V107}

    # Meta/system
    if toks and toks[0] in {"system", "sistema", "about", "manual"}:
        return {"kind": INTENTACT_META_SYSTEM_V107, "slots": {}, "confidence_tag": CONFIDENCE_HIGH_V107}

    # Greeting/chitchat
    if toks and toks[0] in {"oi", "ola", "hi", "hello"}:
        return {"kind": INTENTACT_CHITCHAT_V107, "slots": {"greeting": toks[0]}, "confidence_tag": CONFIDENCE_HIGH_V107}

    # Promise attempt / coercion.
    if _contains_any(toks, ["promete", "promise", "prometa"]):
        return {"kind": INTENTACT_PROMISE_ATTEMPT_V107, "slots": {}, "confidence_tag": CONFIDENCE_MEDIUM_V107}

    # Complaint/confusion.
    if ("??" in raw) or _contains_any(toks, ["nao", "entendi", "confuso", "confusa", "huh"]):
        if _contains_any(toks, ["entendi", "confuso", "confusa", "huh"]):
            return {"kind": INTENTACT_COMPLAINT_V107, "slots": {}, "confidence_tag": CONFIDENCE_MEDIUM_V107}

    # Question markers.
    if ("?" in raw) or _contains_any(toks, ["por", "que", "como", "why", "what", "qual", "quais", "quando"]):
        return {"kind": INTENTACT_QUESTION_V107, "slots": {}, "confidence_tag": CONFIDENCE_MEDIUM_V107}

    # Requests (soft).
    if _contains_any(toks, ["pode", "poderia", "please", "pls", "quero", "gostaria", "me", "ajuda"]):
        return {"kind": INTENTACT_REQUEST_V107, "slots": {}, "confidence_tag": CONFIDENCE_MEDIUM_V107}

    # Fallback: unknown.
    d = _contains_deictic_pronoun_v107(toks)
    slots = {"deictic": str(d)} if d else {}
    return {"kind": INTENTACT_UNKNOWN_V107, "slots": dict(slots), "confidence_tag": CONFIDENCE_LOW_V107}


def infer_assistant_intent_act_v107(*, objective_kind: str, planned_text: str, repair_action: str) -> Dict[str, Any]:
    ok = str(objective_kind or "")
    if str(repair_action or ""):
        return {"kind": ASSISTANTACT_CLARIFY_V107, "slots": {"repair_action": str(repair_action)}}
    if ok == "COMM_END":
        return {"kind": ASSISTANTACT_CLOSE_V107, "slots": {}}
    if ok == "COMM_SUMMARIZE":
        return {"kind": ASSISTANTACT_SUMMARIZE_V107, "slots": {}}
    if ok == "COMM_CONFIRM":
        return {"kind": ASSISTANTACT_CONFIRM_V107, "slots": {}}
    if ok == "COMM_ASK_CLARIFY":
        return {"kind": ASSISTANTACT_CLARIFY_V107, "slots": {}}
    if ok == "COMM_CORRECT":
        return {"kind": ASSISTANTACT_CORRECT_V107, "slots": {}}
    if ok == "COMM_ADMIT_UNKNOWN":
        # Treat as refuse/unknown in pragmatics.
        return {"kind": ASSISTANTACT_REFUSE_V107, "slots": {}}
    if ok == "COMM_RESPOND":
        # If we emit a question mark, it's a clarify; otherwise answer.
        if "?" in str(planned_text or ""):
            return {"kind": ASSISTANTACT_CLARIFY_V107, "slots": {}}
        return {"kind": ASSISTANTACT_ANSWER_V107, "slots": {}}
    return {"kind": ASSISTANTACT_OTHER_V107, "slots": {}}


def compute_repetition_metrics_v107(*, candidate_text: str, recent_assistant_texts: Sequence[str]) -> Dict[str, Any]:
    toks = tokenize_v107(candidate_text)
    bigrams = set([f"{toks[i]}_{toks[i+1]}" for i in range(0, max(0, len(toks) - 1))])
    trigrams = set([f"{toks[i]}_{toks[i+1]}_{toks[i+2]}" for i in range(0, max(0, len(toks) - 2))])
    recent = [tokenize_v107(t) for t in recent_assistant_texts if isinstance(t, str)]
    recent_bi = set([f"{rt[i]}_{rt[i+1]}" for rt in recent for i in range(0, max(0, len(rt) - 1))])
    recent_tri = set([f"{rt[i]}_{rt[i+1]}_{rt[i+2]}" for rt in recent for i in range(0, max(0, len(rt) - 2))])
    bi_overlap = 0.0
    tri_overlap = 0.0
    if bigrams:
        bi_overlap = float(len(bigrams.intersection(recent_bi))) / float(len(bigrams))
    if trigrams:
        tri_overlap = float(len(trigrams.intersection(recent_tri))) / float(len(trigrams))
    # Deterministic cycle heuristic: identical to either of last two assistant texts.
    last2 = [str(t) for t in list(recent_assistant_texts)[-2:] if isinstance(t, str)]
    cycle = bool(str(candidate_text or "") in set(last2))
    return {
        "bigrams_overlap_ratio": float(round(bi_overlap, 6)),
        "trigrams_overlap_ratio": float(round(tri_overlap, 6)),
        "cycle_detected": bool(cycle),
    }


def _is_unsafe_commitment_text_v107(text: str) -> bool:
    s = _norm(text)
    # Minimal "external commitment" patterns.
    bad = [
        "vou reservar",
        "irei reservar",
        "i will book",
        "i'll book",
        "vou comprar",
        "i will purchase",
        "i'll purchase",
    ]
    for b in bad:
        if str(b) in s:
            return True
    return False


def _count_words(toks: Sequence[str]) -> int:
    return int(len([t for t in toks if str(t)]))


def compute_pragmatics_metrics_v107(
    *,
    user_intent_act: Dict[str, Any],
    assistant_intent_act: Dict[str, Any],
    candidate_text: str,
    style_profile: Dict[str, Any],
    pending_questions_active: Sequence[str],
    recent_assistant_texts: Sequence[str],
    regime_prev: str,
    regime_next: str,
) -> Dict[str, Any]:
    u_kind = str((user_intent_act.get("kind") if isinstance(user_intent_act, dict) else "") or "")
    a_kind = str((assistant_intent_act.get("kind") if isinstance(assistant_intent_act, dict) else "") or "")
    toks = tokenize_v107(candidate_text)

    flags: Dict[str, bool] = {}

    # 1) intent_alignment_score
    intent_alignment = 100
    if u_kind == INTENTACT_QUESTION_V107:
        if a_kind not in {ASSISTANTACT_ANSWER_V107, ASSISTANTACT_CLARIFY_V107, ASSISTANTACT_REFUSE_V107}:
            intent_alignment = 0
            flags["intent_mismatch"] = True
    elif u_kind in {INTENTACT_COMMAND_V107, INTENTACT_REQUEST_V107}:
        if a_kind not in {ASSISTANTACT_ANSWER_V107, ASSISTANTACT_CLARIFY_V107, ASSISTANTACT_CONFIRM_V107, ASSISTANTACT_REFUSE_V107, ASSISTANTACT_CORRECT_V107}:
            intent_alignment = 0
            flags["intent_mismatch"] = True
    elif u_kind == INTENTACT_PROMISE_ATTEMPT_V107:
        # Must refuse or clarify, never commit externally.
        if _is_unsafe_commitment_text_v107(candidate_text):
            intent_alignment = 0
            flags["unsafe_commitment_attempt"] = True
        elif a_kind not in {ASSISTANTACT_REFUSE_V107, ASSISTANTACT_CLARIFY_V107, ASSISTANTACT_ANSWER_V107}:
            intent_alignment = 0
            flags["intent_mismatch"] = True
    elif u_kind == INTENTACT_UNKNOWN_V107:
        # Unknown cannot "pass": must clarify.
        if a_kind != ASSISTANTACT_CLARIFY_V107:
            intent_alignment = 0
            flags["requires_clarification"] = True
        else:
            flags["requires_clarification"] = True

    # 2) pending_progress_score
    pending_progress = 100
    if pending_questions_active:
        if a_kind != ASSISTANTACT_CLARIFY_V107:
            pending_progress = 0
            flags["pending_question_ignored"] = True

    # 3) commitment_safety_score
    commitment_safety = 100
    if _is_unsafe_commitment_text_v107(candidate_text):
        commitment_safety = 0
        flags["unsafe_commitment_attempt"] = True

    # 4) regime_consistency_score
    regime_consistency = 100
    if str(regime_prev or "") and str(regime_next or ""):
        if str(regime_prev) != str(regime_next) and str(regime_next) == REGIME_CHITCHAT_V107 and u_kind not in {INTENTACT_CHITCHAT_V107}:
            regime_consistency = 0
            flags["abrupt_regime_shift"] = True

    # 5) diversity_score
    rep = compute_repetition_metrics_v107(candidate_text=str(candidate_text), recent_assistant_texts=list(recent_assistant_texts))
    overlap = float(rep.get("trigrams_overlap_ratio") or rep.get("bigrams_overlap_ratio") or 0.0)
    diversity_score = max(0, min(100, int(round((1.0 - float(overlap)) * 100.0))))
    if bool(rep.get("cycle_detected")) or float(overlap) >= 0.85:
        flags["repetition_loop_detected"] = True

    # 6) verbosity_fit_score
    verbosity_fit = 100
    prof = style_profile if isinstance(style_profile, dict) else {}
    verbosity_pref = str(prof.get("verbosity_preference") or "MEDIUM")
    words = _count_words(toks)
    if verbosity_pref == "SHORT":
        if words > 18:
            verbosity_fit = 0
            flags["verbosity_mismatch"] = True
    elif verbosity_pref == "LONG":
        if words < 12:
            verbosity_fit = 50
            flags["verbosity_mismatch"] = True
    else:  # MEDIUM
        if words > 40:
            verbosity_fit = 50
            flags["verbosity_mismatch"] = True

    components = {
        "intent_alignment_score": int(intent_alignment),
        "pending_progress_score": int(pending_progress),
        "commitment_safety_score": int(commitment_safety),
        "regime_consistency_score": int(regime_consistency),
        "diversity_score": int(diversity_score),
        "verbosity_fit_score": int(verbosity_fit),
    }
    score = min(int(v) for v in components.values()) if components else 0
    score = max(0, min(100, int(score)))

    # Additional critical flag: unanswered question when user asked and we didn't answer nor clarify.
    if u_kind == INTENTACT_QUESTION_V107 and a_kind not in {ASSISTANTACT_ANSWER_V107, ASSISTANTACT_CLARIFY_V107, ASSISTANTACT_REFUSE_V107}:
        flags["unanswered_question"] = True

    flags_digest = _stable_hash_obj({k: bool(flags.get(k, False)) for k in sorted(flags.keys(), key=str)})
    return {
        "pragmatics_score": int(score),
        "components": dict(components),
        "flags": {str(k): bool(flags.get(k, False)) for k in sorted(flags.keys(), key=str)},
        "flags_digest": str(flags_digest),
        "repetition_metrics": dict(rep),
        "len_chars": int(len(str(candidate_text or ""))),
        "len_tokens": int(len(toks)),
    }


def decide_regime_next_v107(
    *,
    regime_prev: str,
    user_intent_act: Dict[str, Any],
    pending_questions_active: Sequence[str],
    repair_action_kind: str,
) -> str:
    if str(repair_action_kind or "") or list(pending_questions_active):
        return REGIME_REPAIR_V107
    uk = str((user_intent_act.get("kind") if isinstance(user_intent_act, dict) else "") or "")
    if uk == INTENTACT_META_SYSTEM_V107:
        return REGIME_META_V107
    if uk == INTENTACT_CHITCHAT_V107:
        return REGIME_CHITCHAT_V107
    if uk == INTENTACT_QUESTION_V107:
        return REGIME_TASK_QA_V107
    if uk in {INTENTACT_COMMAND_V107, INTENTACT_REQUEST_V107}:
        # Keep prior planning mode if already planning; otherwise instruction.
        return REGIME_PLANNING_V107 if str(regime_prev) == REGIME_PLANNING_V107 else REGIME_INSTRUCTION_V107
    return REGIME_TASK_QA_V107


def decide_repair_action_v107(*, pragmatics_metrics: Dict[str, Any], user_intent_act: Dict[str, Any]) -> str:
    flags = pragmatics_metrics.get("flags") if isinstance(pragmatics_metrics.get("flags"), dict) else {}
    u_kind = str((user_intent_act.get("kind") if isinstance(user_intent_act, dict) else "") or "")
    if bool(flags.get("requires_clarification")) or u_kind == INTENTACT_UNKNOWN_V107:
        return REPAIR_ASK_CLARIFY_V107
    if bool(flags.get("pending_question_ignored")):
        return REPAIR_ASK_CLARIFY_V107
    if bool(flags.get("unsafe_commitment_attempt")):
        return REPAIR_RESTATE_OFFER_OPTIONS_V107
    if bool(flags.get("abrupt_regime_shift")):
        return REPAIR_SUMMARIZE_CONFIRM_V107
    if int(pragmatics_metrics.get("pragmatics_score") or 0) < int(PRAGMATICS_THRESH_V107):
        return REPAIR_SUMMARIZE_CONFIRM_V107
    return REPAIR_NONE_V107


def critical_flags_v107(flags: Dict[str, Any]) -> List[str]:
    if not isinstance(flags, dict):
        return []
    critical = [
        "intent_mismatch",
        "unanswered_question",
        "pending_question_ignored",
        "unsafe_commitment_attempt",
        "requires_clarification",
    ]
    out = [k for k in critical if bool(flags.get(k, False))]
    return list(out)


def is_pragmatics_progress_blocked_v107(*, pragmatics_metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
    flags = pragmatics_metrics.get("flags") if isinstance(pragmatics_metrics.get("flags"), dict) else {}
    crit = critical_flags_v107(flags)
    if crit:
        return True, list(crit)
    if int(pragmatics_metrics.get("pragmatics_score") or 0) < int(PRAGMATICS_THRESH_V107):
        return True, ["low_pragmatics_score"]
    return False, []
