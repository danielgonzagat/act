from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex


def _norm_ws(s: str) -> str:
    return " ".join(str(s or "").strip().split())


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _tokenize_v112(text: str) -> List[str]:
    t = _norm_ws(text).lower()
    if not t:
        return []
    return t.split()


def _ngrams(tokens: Sequence[str], n: int) -> List[str]:
    if n <= 0:
        return []
    toks = [str(x) for x in tokens if isinstance(x, str)]
    if len(toks) < n:
        return []
    out: List[str] = []
    for i in range(0, len(toks) - n + 1):
        out.append(" ".join(toks[i : i + n]))
    return out


def _repeated_ngram_rate(texts: Sequence[str], n: int) -> float:
    total = 0
    repeated = 0
    seen: Dict[str, int] = {}
    for t in texts:
        toks = _tokenize_v112(str(t))
        for ng in _ngrams(toks, int(n)):
            total += 1
            seen[ng] = seen.get(ng, 0) + 1
    for ng, c in seen.items():
        if not ng:
            continue
        if int(c) > 1:
            repeated += int(c) - 1
    return float(repeated) / float(total) if total else 0.0


def _count_scaffold_phrases(texts: Sequence[str]) -> Dict[str, int]:
    """
    Minimal, explicit list of high-scaffold openers/phrases. This is NOT a banlist.
    It is only a metric used for audit/triage (can be made learned in later versions).
    """
    phrases = [
        "vamos lá",
        "claro",
        "certo",
        "entendido",
        "ok",
        "posso ajudar",
        "como posso ajudar",
    ]
    counts: Dict[str, int] = {p: 0 for p in phrases}
    for t in texts:
        tt = _norm_ws(str(t)).lower()
        for p in phrases:
            if tt.startswith(p):
                counts[p] += 1
    # Drop zeros for compactness.
    return {k: int(v) for k, v in counts.items() if int(v) > 0}


def fluency_metrics_v112(*, transcript_view: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    assistant_texts: List[str] = []
    for r in transcript_view:
        if not isinstance(r, dict):
            continue
        if str(r.get("role") or "") != "assistant":
            continue
        assistant_texts.append(str(r.get("text") or ""))

    r3 = _repeated_ngram_rate(assistant_texts, 3)
    r4 = _repeated_ngram_rate(assistant_texts, 4)
    scaffold = _count_scaffold_phrases(assistant_texts)
    return {
        "assistant_total": int(len(assistant_texts)),
        "repeated_ngram_rate": {
            "n3": float(round(float(r3), 6)),
            "n4": float(round(float(r4), 6)),
        },
        "scaffold_prefix_counts": dict(scaffold),
        "metrics_sig": _stable_hash_obj({"n3": float(round(float(r3), 6)), "n4": float(round(float(r4), 6)), "scaffold": scaffold}),
    }


def fluency_contract_v112(
    *,
    transcript_view: Sequence[Dict[str, Any]],
    most_common_reply_frac_max: float = 0.35,
    unique_reply_rate_min: float = 0.18,
    repeated_ngram_rate_n4_max: float = 0.92,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    V112: fluency-as-survival gate (deterministic).
    Base = V111-style contract (diversity + unknown hygiene), plus a weak anti-template n-gram check.

    Deterministic V112 refinement:
      - Count "não sei"/"I don't know" as a hygiene violation only when it is the assistant's *leading clause*.
        This avoids false positives when quoting user text that contains "não sei" (e.g. "Comando inválido: ... diga não sei ...").
    """
    assistant = [_norm_ws(str(r.get("text") or "")) for r in transcript_view if isinstance(r, dict) and str(r.get("role") or "") == "assistant"]
    total = len(assistant)
    if total == 0:
        return False, "no_assistant_texts", {"assistant_total": 0}

    counts: Dict[str, int] = {}
    for s in assistant:
        counts[s] = counts.get(s, 0) + 1
    most_common = max(counts.values()) if counts else 0
    most_common_frac = float(most_common) / float(total) if total else 1.0
    unique_rate = float(len(counts)) / float(total) if total else 0.0

    unknown_start_re = re.compile(r"^\s*(n[aã]o\s+sei|i\s+don'?t\s+know)\b", re.IGNORECASE)
    question_re = re.compile(r"\?")  # any question mark
    unknown_wo_q = 0
    for s in assistant:
        if unknown_start_re.search(s) and not question_re.search(s):
            unknown_wo_q += 1
    unknown_wo_q_frac = float(unknown_wo_q) / float(total) if total else 0.0

    metrics = fluency_metrics_v112(transcript_view=transcript_view)
    details = {
        "v111_compat": {
            "assistant_total": int(total),
            "unique_replies": int(len(counts)),
            "most_common_reply": max(counts.items(), key=lambda kv: (int(kv[1]), str(kv[0])))[0] if counts else "",
            "most_common_reply_count": int(most_common),
            "most_common_reply_frac": float(round(float(most_common_frac), 6)),
            "unique_reply_rate": float(round(float(unique_rate), 6)),
            "unknown_without_question_count": int(unknown_wo_q),
            "unknown_without_question_frac": float(round(float(unknown_wo_q_frac), 6)),
        },
        "v112_metrics": dict(metrics),
        "thresholds": {
            "most_common_reply_frac_max": float(round(float(most_common_reply_frac_max), 6)),
            "unique_reply_rate_min": float(round(float(unique_reply_rate_min), 6)),
            "repeated_ngram_rate_n4_max": float(round(float(repeated_ngram_rate_n4_max), 6)),
        },
    }

    if most_common_frac > float(most_common_reply_frac_max):
        return False, "most_common_reply_frac_too_high", dict(details)
    if unique_rate < float(unique_reply_rate_min):
        return False, "unique_reply_rate_too_low", dict(details)
    if unknown_wo_q >= 2:
        return False, "unknown_without_question_repeated", dict(details)

    r4 = float(((metrics.get("repeated_ngram_rate") or {}).get("n4")) or 0.0)
    if r4 > float(repeated_ngram_rate_n4_max):
        return False, "repeated_ngram_rate_n4_too_high", dict(details)

    return True, "ok", dict(details)


@dataclass(frozen=True)
class FluencySurvivalAttemptV112:
    attempt_index: int
    seed_used: int
    ok: bool
    reason: str
    details: Dict[str, Any]


def fluency_survival_plan_v112(
    *,
    base_seed: int,
    max_attempts: int,
) -> List[int]:
    """
    Deterministic rewrite schedule: try base_seed, then bump seed by +1, +2, ...
    """
    out: List[int] = []
    for i in range(int(max_attempts)):
        out.append(int(base_seed) + int(i))
    return out


def summarize_fluency_fail_code_v112(reason: str) -> str:
    """
    Canonical short code for fail_catalog bucketing.
    """
    r = str(reason or "")
    if not r:
        return "unknown"
    r = re.sub(r"[^a-z0-9_]+", "_", r.lower()).strip("_")
    return r or "unknown"
