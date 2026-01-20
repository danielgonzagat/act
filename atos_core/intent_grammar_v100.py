from __future__ import annotations

import unicodedata
from typing import Any, Dict


INTENT_DISCOURSE_V100 = "INTENT_DISCOURSE_V100"
INTENT_WHY_REF_V100 = "INTENT_WHY_REF_V100"


def _strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _norm_prefix(text: str) -> str:
    return _strip_accents(str(text or "")).lower()


def is_discourse_command_v100(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("discourse") or s2.startswith("discurso"))


def parse_discourse_command_v100(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      discourse
      discourse:   (no payload)
      discurso
      discurso:    (no payload)
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    if s2.startswith("discourse"):
        prefix = "discourse"
    elif s2.startswith("discurso"):
        prefix = "discurso"
    else:
        return {"recognized": False}
    rest_raw = s[len(prefix) :]
    if rest_raw and (not rest_raw[0].isspace()) and (rest_raw[0] != ":"):
        return {"recognized": False}
    rest = str(rest_raw).strip()
    if rest.startswith(":"):
        rest = rest[1:].strip()
    if rest:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "prefix": str(prefix), "extra_raw": str(rest_raw)}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix)}


def is_why_ref_command_v100(user_text: str) -> bool:
    """
    A deterministic, fail-closed "reference" variant of WHY:
      - porque isso / porque aquilo
      - why that / why this
    This is intercepted BEFORE the V98 WHY parser, so it doesn't change V98 semantics.
    """
    raw = str(user_text or "").strip()
    s2 = _norm_prefix(raw)
    if s2.startswith("porque "):
        rest = s2[len("porque ") :].strip()
        return rest in {"isso", "aquilo"}
    if s2.startswith("why "):
        rest = s2[len("why ") :].strip()
        return rest in {"that", "this"}
    return False


def parse_why_ref_command_v100(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      porque isso
      porque aquilo
      why that
      why this
    """
    raw = str(user_text or "").strip()
    s2 = _norm_prefix(raw)
    if s2.startswith("porque "):
        rest = s2[len("porque ") :].strip()
        if rest not in {"isso", "aquilo"}:
            return {"recognized": False}
        return {"recognized": True, "ok": True, "reason": "ok", "prefix": "porque", "ref_token": str(rest)}
    if s2.startswith("why "):
        rest = s2[len("why ") :].strip()
        if rest not in {"that", "this"}:
            return {"recognized": False}
        return {"recognized": True, "ok": True, "reason": "ok", "prefix": "why", "ref_token": str(rest)}
    return {"recognized": False}

