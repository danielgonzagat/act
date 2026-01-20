from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex


def _norm_ws(s: str) -> str:
    return " ".join(str(s or "").strip().split())


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _extract_assistant_texts(transcript: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for t in transcript:
        if not isinstance(t, dict):
            continue
        role = str(t.get("role") or "")
        if role != "assistant":
            continue
        out.append(str(t.get("text") or ""))
    return out


def fluency_contract_v111(
    *,
    transcript: Sequence[Dict[str, Any]],
    most_common_reply_frac_max: float = 0.40,
    unique_reply_rate_min: float = 0.15,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Deterministic, binary-ish contract:
      - Limit exact repetition (most common normalized reply fraction).
      - Ensure reply diversity (unique reply rate).
      - Enforce "don't know" hygiene: repeated unknown without asking for info is a FAIL.
    """
    assistant = [_norm_ws(x) for x in _extract_assistant_texts(transcript)]
    total = len(assistant)
    if total == 0:
        return False, "no_assistant_texts", {"assistant_total": 0}

    counts: Dict[str, int] = {}
    for s in assistant:
        counts[s] = counts.get(s, 0) + 1
    most_common = max(counts.values()) if counts else 0
    most_common_frac = float(most_common) / float(total) if total else 1.0
    unique_rate = float(len(counts)) / float(total) if total else 0.0

    # "Don't know" repetition without asking.
    unknown_re = re.compile(r"\b(n[aã]o\s+sei|i\s+don'?t\s+know)\b", re.IGNORECASE)
    question_re = re.compile(r"\?")  # any question mark
    unknown_wo_q = 0
    for s in assistant:
        if unknown_re.search(s) and not question_re.search(s):
            unknown_wo_q += 1
    unknown_wo_q_frac = float(unknown_wo_q) / float(total) if total else 0.0

    details = {
        "assistant_total": int(total),
        "thresholds": {
            "most_common_reply_frac_max": float(round(float(most_common_reply_frac_max), 6)),
            "unique_reply_rate_min": float(round(float(unique_reply_rate_min), 6)),
        },
        "unique_replies": int(len(counts)),
        "most_common_reply": max(counts.items(), key=lambda kv: (int(kv[1]), str(kv[0])))[0] if counts else "",
        "most_common_reply_count": int(most_common),
        "most_common_reply_frac": float(round(most_common_frac, 6)),
        "unique_reply_rate": float(round(unique_rate, 6)),
        "unknown_without_question_count": int(unknown_wo_q),
        "unknown_without_question_frac": float(round(unknown_wo_q_frac, 6)),
        "contract_sig": _stable_hash_obj(
            {
                "assistant_total": int(total),
                "most_common_reply_frac": float(round(most_common_frac, 6)),
                "unique_reply_rate": float(round(unique_rate, 6)),
                "unknown_without_question_frac": float(round(unknown_wo_q_frac, 6)),
            }
        ),
    }

    if most_common_frac > float(most_common_reply_frac_max):
        return False, "most_common_reply_frac_too_high", dict(details)
    if unique_rate < float(unique_reply_rate_min):
        return False, "unique_reply_rate_too_low", dict(details)
    if unknown_wo_q >= 2:
        return False, "unknown_without_question_repeated", dict(details)

    return True, "ok", dict(details)
