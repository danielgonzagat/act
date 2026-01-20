from __future__ import annotations

import unicodedata
from typing import Any, Dict


INTENT_EXPLAIN_V94 = "INTENT_EXPLAIN"


def _strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _norm_prefix(text: str) -> str:
    return _strip_accents(str(text or "")).lower()


def is_explain_command_v94(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("explain") or s2.startswith("explique"))


def parse_explain_command_v94(user_text: str) -> Dict[str, Any]:
    """
    Parse EXPLAIN/EXPLIQUE command in raw text (before normalization of the main parser).
    Syntax (deterministic, fail-closed on extra tokens):
      explain
      explain:
      explique
      explique:
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    rest_raw = ""
    if s2.startswith("explain"):
        prefix = "explain"
        rest_raw = s[len("explain") :]
    elif s2.startswith("explique"):
        prefix = "explique"
        rest_raw = s[len("explique") :]
    else:
        return {"recognized": False}

    rest = str(rest_raw).strip()
    if rest.startswith(":"):
        rest = rest[1:].strip()
    if rest:
        # Fail-closed: do not accept extra payload in v94.
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "prefix": str(prefix), "extra_raw": str(rest_raw)}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix)}

