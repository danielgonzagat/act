from __future__ import annotations

import unicodedata
from typing import Any, Dict


INTENT_SYSTEM_V97 = "INTENT_SYSTEM"


def _strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _norm_prefix(text: str) -> str:
    return _strip_accents(str(text or "")).lower()


_SYSTEM_PREFIXES_V97 = ["system", "sistema", "manual", "about"]


def is_system_command_v97(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    for p in _SYSTEM_PREFIXES_V97:
        if not s2.startswith(p):
            continue
        if len(s2) == len(p):
            return True
        nxt = s2[len(p)]
        if nxt.isspace() or nxt == ":":
            return True
    return False


def parse_system_command_v97(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed, raw intercept):
      system / system:
      sistema / sistema:
      manual / manual:
      about / about:

    No payload allowed (whitespace-only ok).
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    rest_raw = ""
    for p in _SYSTEM_PREFIXES_V97:
        if not s2.startswith(p):
            continue
        # Avoid prefix-matching inside longer words.
        if len(s2) > len(p):
            nxt = s2[len(p)]
            if (not nxt.isspace()) and nxt != ":":
                continue
        prefix = p
        rest_raw = s[len(p) :]
        break
    if not prefix:
        return {"recognized": False}

    if rest_raw and (not rest_raw[0].isspace()) and (rest_raw[0] != ":"):
        return {"recognized": False}

    rest = str(rest_raw).strip()
    if rest.startswith(":"):
        rest = rest[1:].strip()
    if rest:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "prefix": str(prefix), "extra_raw": str(rest_raw)}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix)}

