from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .style_profile_v102 import StyleProfileV102, VERBOSITY_LONG, VERBOSITY_SHORT


def _round6(x: Any) -> float:
    try:
        return float(round(float(x), 6))
    except Exception:
        return 0.0


def _tokens(text: str) -> List[str]:
    return [t for t in str(text or "").lower().split() if t]


def _trigrams(tokens: Sequence[str]) -> List[Tuple[str, str, str]]:
    toks = [str(t) for t in tokens if isinstance(t, str)]
    return list(zip(toks, toks[1:], toks[2:])) if len(toks) >= 3 else []


def _trigram_overlap_ratio(a: str, b: str) -> float:
    ta = set(_trigrams(_tokens(a)))
    tb = set(_trigrams(_tokens(b)))
    if not ta or not tb:
        return 0.0
    inter = len(ta.intersection(tb))
    denom = float(min(len(ta), len(tb)))
    return float(inter) / denom if denom > 0 else 0.0


def _prefix2(text: str) -> str:
    toks = _tokens(text)
    return " ".join(toks[:2])


def _critic_result(*, critic_id: str, status: str, score_delta: float, metrics: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "critic_id": str(critic_id),
        "status": str(status),
        "score_delta": _round6(score_delta),
        "metrics": dict(metrics) if isinstance(metrics, dict) else {},
        "reason": str(reason),
    }


def run_critics_v102(
    *,
    candidate_text: str,
    response_kind: str,
    style_profile: StyleProfileV102,
    binding_status: str,
    recent_assistant_texts: Sequence[str],
    recent_template_ids: Sequence[str],
    template_id: str,
) -> Dict[str, Any]:
    """
    Deterministic critics (fail-closed on clarity issues).
    Returns:
      {"ok": bool, "total_score_delta": float, "results": [...], "metrics": {...}}
    """
    text = str(candidate_text or "")
    rk = str(response_kind or "")
    bs = str(binding_status or "")

    results: List[Dict[str, Any]] = []
    total = 0.0

    toks = _tokens(text)
    words = len(toks)
    chars = len(text)
    lines = len(text.splitlines())

    # Critic: clarity for ambiguous/miss references.
    needs_q = (rk in {"clarify", "confirm"}) or (bs in {"AMBIGUOUS", "MISS"})
    has_q = "?" in text
    if needs_q and not has_q:
        results.append(_critic_result(critic_id="clarity_question", status="FAIL", score_delta=-1.0, metrics={"needs_question": True}, reason="missing_question_mark"))
        total -= 1.0
    else:
        results.append(_critic_result(critic_id="clarity_question", status="PASS", score_delta=0.0, metrics={"needs_question": bool(needs_q), "has_question": bool(has_q)}, reason="ok"))

    # Critic: length vs verbosity preference.
    if style_profile.verbosity_preference == VERBOSITY_SHORT and words > 24:
        delta = -0.10 * float(max(0, words - 24))
        results.append(_critic_result(critic_id="length_short", status="WARN", score_delta=delta, metrics={"words": int(words)}, reason="too_long_for_short"))
        total += float(delta)
    elif style_profile.verbosity_preference == VERBOSITY_LONG and words < 8:
        results.append(_critic_result(critic_id="length_long", status="WARN", score_delta=-0.20, metrics={"words": int(words)}, reason="too_short_for_long"))
        total -= 0.20
    else:
        results.append(_critic_result(critic_id="length", status="PASS", score_delta=0.0, metrics={"words": int(words)}, reason="ok"))

    # Critic: repetition with recent assistant turns (ngram overlap).
    recent = [str(x) for x in recent_assistant_texts if isinstance(x, str) and x]
    overlap_max = 0.0
    for prev in recent[:3]:
        overlap_max = max(overlap_max, _trigram_overlap_ratio(prev, text))
    if overlap_max >= 0.80 and len(recent) > 0:
        # Hard fail on near-duplicate.
        results.append(_critic_result(critic_id="repeat_trigram", status="FAIL", score_delta=-1.0, metrics={"overlap_max": _round6(overlap_max)}, reason="near_duplicate"))
        total -= 1.0
    else:
        penalty = -0.40 * float(overlap_max)
        status = "WARN" if overlap_max >= 0.40 else "PASS"
        results.append(_critic_result(critic_id="repeat_trigram", status=status, score_delta=penalty, metrics={"overlap_max": _round6(overlap_max)}, reason="ok"))
        total += float(penalty)

    # Critic: repeated prefix opener.
    pref = _prefix2(text)
    pref_prev = [_prefix2(x) for x in recent[:3] if isinstance(x, str)]
    repeats = sum(1 for p in pref_prev if p and p == pref)
    if repeats >= 2 and pref:
        results.append(_critic_result(critic_id="repeat_prefix", status="WARN", score_delta=-0.30, metrics={"prefix": pref, "repeats": int(repeats)}, reason="prefix_repeated"))
        total -= 0.30
    else:
        results.append(_critic_result(critic_id="repeat_prefix", status="PASS", score_delta=0.0, metrics={"prefix": pref, "repeats": int(repeats)}, reason="ok"))

    # Critic: template overuse.
    recent_t = [str(x) for x in recent_template_ids if isinstance(x, str) and x]
    consecutive = 0
    for t in reversed(recent_t):
        if t == str(template_id):
            consecutive += 1
        else:
            break
    if consecutive >= 2:
        results.append(_critic_result(critic_id="template_repeat", status="WARN", score_delta=-0.25, metrics={"consecutive": int(consecutive)}, reason="template_overused"))
        total -= 0.25
    else:
        results.append(_critic_result(critic_id="template_repeat", status="PASS", score_delta=0.0, metrics={"consecutive": int(consecutive)}, reason="ok"))

    ok = all(str(r.get("status") or "") != "FAIL" for r in results)
    metrics = {"words": int(words), "chars": int(chars), "lines": int(lines), "binding_status": bs, "response_kind": rk}
    return {"ok": bool(ok), "total_score_delta": _round6(total), "results": list(results), "metrics": dict(metrics)}


def fluency_score_v102(*, base_score: float, critics: Dict[str, Any]) -> float:
    """
    Deterministic aggregate score in [0, 1].
    """
    score = float(base_score)
    delta = float(critics.get("total_score_delta") or 0.0)
    score = score + delta
    if not bool(critics.get("ok", False)):
        score = min(score, 0.0)
    score = max(0.0, min(1.0, score))
    return _round6(score)


def text_sha256_v102(text: str) -> str:
    return sha256_hex(str(text or "").encode("utf-8"))


def candidate_sig_v102(body: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))

