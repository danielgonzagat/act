from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .fluency_survival_v112 import fluency_contract_v112, fluency_metrics_v112


def _norm_ws(s: str) -> str:
    return " ".join(str(s or "").strip().split())


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _tokenize(text: str) -> List[str]:
    t = _norm_ws(text).lower()
    if not t:
        return []
    return t.split()


def _prefix2(text: str) -> str:
    toks = _tokenize(text)
    if not toks:
        return ""
    return " ".join(toks[:2])


def _max_consecutive_equal(items: Sequence[str]) -> Tuple[int, str]:
    best = 0
    best_item = ""
    cur = 0
    prev = None
    for it in items:
        if it and prev == it:
            cur += 1
        else:
            cur = 1 if it else 0
            prev = it
        if cur > best:
            best = cur
            best_item = it
    return int(best), str(best_item or "")


def fluency_contract_v118(
    *,
    transcript_view: Sequence[Dict[str, Any]],
    most_common_reply_frac_max: float = 0.35,
    unique_reply_rate_min: float = 0.18,
    repeated_ngram_rate_n4_max: float = 0.92,
    # V118 additions (explicit, deterministic; not a banlist):
    max_scaffold_prefix_frac: float = 0.60,
    max_consecutive_prefix2: int = 4,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    V118: strengthen fluency-as-survival with a few deterministic "adult dialogue" checks.

    This remains strictly non-ML:
      - no embeddings,
      - no probabilistic classifiers,
      - only explicit metrics + thresholds.
    """
    ok112, reason112, details112 = fluency_contract_v112(
        transcript_view=transcript_view,
        most_common_reply_frac_max=float(most_common_reply_frac_max),
        unique_reply_rate_min=float(unique_reply_rate_min),
        repeated_ngram_rate_n4_max=float(repeated_ngram_rate_n4_max),
    )
    if not ok112:
        # Preserve V112 reason codes for compatibility with existing fail catalogs.
        return False, str(reason112), dict(details112)

    assistant_texts: List[str] = []
    for r in transcript_view:
        if not isinstance(r, dict):
            continue
        if str(r.get("role") or "") != "assistant":
            continue
        assistant_texts.append(_norm_ws(str(r.get("text") or "")))
    total = int(len(assistant_texts))
    if total <= 0:
        return False, "no_assistant_texts", {"assistant_total": 0}

    # Reuse v112 metrics for transparency.
    metrics112 = fluency_metrics_v112(transcript_view=transcript_view)
    scaffold = (metrics112.get("scaffold_prefix_counts") if isinstance(metrics112, dict) else None) or {}
    if not isinstance(scaffold, dict):
        scaffold = {}

    worst_phrase = ""
    worst_frac = 0.0
    for p, c in scaffold.items():
        try:
            frac = float(int(c)) / float(total) if total else 0.0
        except Exception:
            frac = 0.0
        if frac > worst_frac or (abs(frac - worst_frac) < 1e-12 and str(p) < str(worst_phrase)):
            worst_frac = float(frac)
            worst_phrase = str(p)

    prefixes2 = [_prefix2(t) for t in assistant_texts]
    max_run, max_run_prefix = _max_consecutive_equal(prefixes2)

    details = {
        "schema_version": 118,
        "v112": dict(details112) if isinstance(details112, dict) else {},
        "v118_metrics": {
            "assistant_total": int(total),
            "scaffold_prefix_worst": {"phrase": str(worst_phrase), "frac": float(round(float(worst_frac), 6))},
            "prefix2_max_consecutive": {"n": int(max_run), "prefix2": str(max_run_prefix)},
        },
        "thresholds": {
            "max_scaffold_prefix_frac": float(round(float(max_scaffold_prefix_frac), 6)),
            "max_consecutive_prefix2": int(max_consecutive_prefix2),
        },
    }
    details["metrics_sig"] = _stable_hash_obj(details)

    # Deterministic checks (fail-closed).
    if worst_phrase and float(worst_frac) > float(max_scaffold_prefix_frac) and total >= 5:
        return False, "scaffold_prefix_overused", dict(details)
    if max_run >= int(max_consecutive_prefix2) and max_run_prefix:
        return False, "consecutive_prefix2_repeat", dict(details)

    # Avoid excessive "manual voice" leakage: if the assistant uses "PASS"/"FAIL" as the leading token too often.
    leading_re = re.compile(r"^\s*(pass|fail)\b", re.IGNORECASE)
    leading_counts = 0
    for t in assistant_texts:
        if leading_re.search(t):
            leading_counts += 1
    if total >= 8 and leading_counts >= 3:
        details2 = dict(details)
        details2["v118_metrics"] = dict(details2.get("v118_metrics") or {})
        details2["v118_metrics"]["leading_pass_fail_count"] = int(leading_counts)
        details2["metrics_sig"] = _stable_hash_obj(details2)
        return False, "over_manual_voice", dict(details2)

    return True, "ok", dict(details)


@dataclass(frozen=True)
class FluencyContractResultV118:
    ok: bool
    reason: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 118,
            "kind": "fluency_contract_result_v118",
            "ok": bool(self.ok),
            "reason": str(self.reason),
            "details": dict(self.details) if isinstance(self.details, dict) else {},
        }
        body["result_sig"] = _stable_hash_obj(body)
        return dict(body)

