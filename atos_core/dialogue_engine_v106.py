from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .dialogue_ledger_v106 import (
    DIALOGUE_REPAIR_ASK_CLARIFY_V106,
    DIALOGUE_REPAIR_NONE_V106,
    DIALOGUE_REPAIR_RESTATE_OFFER_OPTIONS_V106,
    DIALOGUE_REPAIR_SUMMARIZE_CONFIRM_V106,
)


COHERENCE_THRESH_V106 = 60


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _tokens_v106(text: str) -> List[str]:
    s = str(text or "").lower()
    s = re.sub(r"[^a-z0-9_\\s]", " ", s)
    toks = [t for t in s.split() if t]
    return list(toks)


def _bigrams(toks: Sequence[str]) -> List[str]:
    out: List[str] = []
    for i in range(len(toks) - 1):
        out.append(str(toks[i]) + "\u241f" + str(toks[i + 1]))
    return out


def _ngram_repetition_ratio_v106(toks: Sequence[str], *, n: int) -> float:
    if n <= 1:
        return 0.0
    if len(toks) < n:
        return 0.0
    grams: List[str] = []
    for i in range(len(toks) - n + 1):
        grams.append("\u241f".join([str(x) for x in toks[i : i + n]]))
    if not grams:
        return 0.0
    uniq = len(set(grams))
    total = len(grams)
    if total <= 0:
        return 0.0
    return float(1.0 - (float(uniq) / float(total)))


def compute_metrics_v106(
    *,
    candidate_text: str,
    user_text: str,
    objective_kind: str,
    binding_status: str,
    recent_assistant_texts: Sequence[str],
    recent_topics: Sequence[str],
    current_topic: str,
) -> Dict[str, Any]:
    """
    Deterministic multi-turn dialogue coherence metric (MVP).
    Returns:
      {
        "coherence_score": int(0..100),
        "components": {... ints ...},
        "flags": {... bool ...},
        "len_chars": int,
        "len_tokens": int,
      }
    """
    text = str(candidate_text or "")
    toks = _tokens_v106(text)
    words = len(toks)
    chars = len(text)

    # Basic flags.
    too_long = words >= 60
    too_short = words <= 2
    too_dry = chars <= 8
    asked_when_should_answer = str(objective_kind) in {"COMM_RESPOND", "COMM_SUMMARIZE"} and "?" in text
    answered_when_should_ask = str(objective_kind) in {"COMM_ASK_CLARIFY", "COMM_CONFIRM"} and "?" not in text

    prev = str(recent_assistant_texts[0]) if recent_assistant_texts else ""
    repetition = bool(prev and text and text == prev)
    # internal repetition (trigram reuse)
    rep3 = _ngram_repetition_ratio_v106(toks, n=3)
    if rep3 >= 0.50:
        repetition = True

    # Abrupt topic shift (very simple): topic changed and no transition marker.
    prev_topic = str(recent_topics[0]) if recent_topics else ""
    abrupt_shift = bool(prev_topic and current_topic and prev_topic != current_topic and not re.search(r"\\b(sobre|indo|agora)\\b", text.lower()))

    flags = {
        "too_dry": bool(too_dry),
        "too_long": bool(too_long),
        "too_short": bool(too_short),
        "abrupt_shift": bool(abrupt_shift),
        "repetition": bool(repetition),
        "asked_when_should_answer": bool(asked_when_should_answer),
        "answered_when_should_ask": bool(answered_when_should_ask),
    }

    # Components 0..100
    entity = 100
    topic = 100
    ref = 100
    logic = 100

    if abrupt_shift:
        topic = 40
    if binding_status in {"AMBIGUOUS", "MISS"}:
        # If we have a reference problem and didn't ask, ref is very low.
        if "?" not in text:
            ref = 0
        else:
            ref = 60
    if answered_when_should_ask:
        ref = 0
    if asked_when_should_answer:
        logic = min(logic, 60)
    if repetition:
        logic = min(logic, 40)
        topic = min(topic, 60)
    if too_long:
        entity = min(entity, 80)
        logic = min(logic, 80)
    if too_short:
        topic = min(topic, 70)
        entity = min(entity, 70)

    components = {
        "entity_consistency": int(max(0, min(100, entity))),
        "topic_continuity": int(max(0, min(100, topic))),
        "reference_resolution": int(max(0, min(100, ref))),
        "logical_consistency": int(max(0, min(100, logic))),
    }

    score = int(
        round(
            0.3 * float(components["entity_consistency"])
            + 0.2 * float(components["topic_continuity"])
            + 0.2 * float(components["reference_resolution"])
            + 0.3 * float(components["logical_consistency"])
        )
    )
    score = max(0, min(100, int(score)))

    return {
        "coherence_score": int(score),
        "components": dict(components),
        "flags": dict(flags),
        "len_chars": int(chars),
        "len_tokens": int(words),
        "flags_digest": _stable_hash_obj(dict(flags)),
    }


def select_candidate_v106(
    *,
    evaluated_candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Deterministic selection among candidates. Assumes each candidate already has:
      - cand_id, score (int), len_tokens, text_sha256
    """

    def _rk(c: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            -int(c.get("score") or 0),
            int(c.get("len_tokens") or 0),
            str(c.get("text_sha256") or ""),
            str(c.get("cand_id") or ""),
        )

    cands = [dict(c) for c in evaluated_candidates if isinstance(c, dict)]
    cands_sorted = sorted(cands, key=_rk)
    return dict(cands_sorted[0]) if cands_sorted else {}


def decide_repair_action_v106(
    *,
    prev_coherence_scores: Sequence[int],
    selected_score: int,
    binding_status: str,
    has_active_goal: bool,
    objective_kind: str,
) -> str:
    """
    Survival rule (MVP):
      If previous turn score < THRESH and selected_score < THRESH => choose repair action.
      Otherwise: if the current objective is already a repair-type objective, record it.
    """
    prev_low = False
    if prev_coherence_scores:
        prev_low = int(prev_coherence_scores[0]) < int(COHERENCE_THRESH_V106)

    if str(objective_kind) in {"COMM_ASK_CLARIFY"}:
        return DIALOGUE_REPAIR_ASK_CLARIFY_V106
    if str(objective_kind) in {"COMM_SUMMARIZE"}:
        return DIALOGUE_REPAIR_SUMMARIZE_CONFIRM_V106
    if str(objective_kind) in {"COMM_CONFIRM"}:
        return DIALOGUE_REPAIR_RESTATE_OFFER_OPTIONS_V106

    if prev_low and int(selected_score) < int(COHERENCE_THRESH_V106):
        if str(binding_status) in {"AMBIGUOUS", "MISS"}:
            return DIALOGUE_REPAIR_ASK_CLARIFY_V106
        if bool(has_active_goal):
            return DIALOGUE_REPAIR_SUMMARIZE_CONFIRM_V106
        return DIALOGUE_REPAIR_RESTATE_OFFER_OPTIONS_V106

    return DIALOGUE_REPAIR_NONE_V106


def repair_text_v106(
    *,
    repair_action: str,
    user_text: str,
    active_goal_text: str,
) -> str:
    """
    Deterministic repair messages (used only when survival rule triggers).
    """
    ra = str(repair_action or "")
    if ra == DIALOGUE_REPAIR_ASK_CLARIFY_V106:
        ut = str(user_text or "").strip()
        # Keep it short and safe; do not invent.
        if ut:
            return f"Você pode esclarecer o que você quer dizer com: {ut}?"
        return "Você pode esclarecer o que você quer dizer?"
    if ra == DIALOGUE_REPAIR_SUMMARIZE_CONFIRM_V106:
        gt = str(active_goal_text or "").strip()
        if gt:
            return f"Resumo: objetivo ativo = {gt}. Você confirma que é isso que você quer?"
        return "Resumo: eu posso continuar, mas preciso confirmar o objetivo. Você confirma o que você quer fazer?"
    if ra == DIALOGUE_REPAIR_RESTATE_OFFER_OPTIONS_V106:
        return "Eu posso: (A) responder direto se for verificável; (B) pedir uma clarificação específica. Qual você prefere: A ou B?"
    return ""

