from __future__ import annotations

import unicodedata
from typing import Any, Dict


INTENT_NOTE_V95 = "INTENT_NOTE"
INTENT_RECALL_V95 = "INTENT_RECALL"
INTENT_FORGET_V95 = "INTENT_FORGET"


def _strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _norm_prefix(text: str) -> str:
    return _strip_accents(str(text or "")).lower()


def is_note_command_v95(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("note") or s2.startswith("nota"))


def parse_note_command_v95(user_text: str) -> Dict[str, Any]:
    """
    Parse NOTE/NOTA command in raw text (before normalization of the main parser).
    Syntax (deterministic, fail-closed):
      note: <text>
      nota: <text>
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    if s2.startswith("note"):
        prefix = "note"
        rest_raw = s[len("note") :]
    elif s2.startswith("nota"):
        prefix = "nota"
        rest_raw = s[len("nota") :]
    else:
        return {"recognized": False}

    if rest_raw and (not rest_raw[0].isspace()) and (rest_raw[0] != ":"):
        # Prefix is part of a larger token (e.g., "notebook").
        return {"recognized": False}

    rest = str(rest_raw).lstrip()
    if not rest.startswith(":"):
        return {"recognized": True, "ok": False, "reason": "missing_colon", "prefix": str(prefix), "extra_raw": str(rest_raw)}

    text = str(rest[1:]).strip()
    if not text:
        return {"recognized": True, "ok": False, "reason": "empty_text", "prefix": str(prefix)}

    return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix), "memory_text_raw": str(text)}


def is_recall_command_v95(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("recall") or s2.startswith("memoria"))


def parse_recall_command_v95(user_text: str) -> Dict[str, Any]:
    """
    Parse RECALL/MEMORIA command in raw text.
    Syntax:
      recall
      memoria
      memória
      (optional) "recall:" / "memoria:" with no payload
    Fail-closed on extra tokens/payload.
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    if s2.startswith("recall"):
        prefix = "recall"
        rest_raw = s[len("recall") :]
    elif s2.startswith("memoria"):
        prefix = "memoria"
        rest_raw = s[len("memoria") :]
    else:
        return {"recognized": False}

    if rest_raw and (not rest_raw[0].isspace()) and (rest_raw[0] != ":"):
        return {"recognized": False}

    rest = str(rest_raw).strip()
    if rest.startswith(":"):
        rest = rest[1:].strip()
    if rest:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "prefix": str(prefix), "extra_raw": str(rest_raw)}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix)}


def is_forget_command_v95(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("forget") or s2.startswith("esquece"))


def parse_forget_command_v95(user_text: str) -> Dict[str, Any]:
    """
    Parse FORGET/ESQUECE command in raw text.
    Syntax (fixed in V95):
      forget last
      esquece last
    Fail-closed on other formats (e.g., "forget:<id>") or extra tokens.
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    if s2.startswith("forget"):
        prefix = "forget"
        rest_raw = s[len("forget") :]
    elif s2.startswith("esquece"):
        prefix = "esquece"
        rest_raw = s[len("esquece") :]
    else:
        return {"recognized": False}

    if rest_raw and (not rest_raw[0].isspace()) and (rest_raw[0] != ":"):
        return {"recognized": False}

    rest = str(rest_raw).lstrip()
    if rest.startswith(":"):
        return {"recognized": True, "ok": False, "reason": "unsupported_format", "prefix": str(prefix), "extra_raw": str(rest_raw)}

    toks = [t for t in rest.split() if t]
    if not toks:
        return {"recognized": True, "ok": False, "reason": "missing_target", "prefix": str(prefix)}
    if toks[0].lower() != "last":
        return {"recognized": True, "ok": False, "reason": "bad_target", "prefix": str(prefix), "target": str(toks[0])}
    if len(toks) != 1:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "prefix": str(prefix), "extra_raw": str(rest_raw)}

    return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix), "target": "last"}

