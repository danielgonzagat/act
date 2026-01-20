from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List


INTENT_PRAGMATICS_V107 = "INTENT_PRAGMATICS_V107"
INTENT_EXPLAIN_PRAGMATICS_V107 = "INTENT_EXPLAIN_PRAGMATICS_V107"
INTENT_TRACE_PRAGMATICS_V107 = "INTENT_TRACE_PRAGMATICS_V107"


def _strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _norm(text: str) -> str:
    return _strip_accents(str(text or "")).lower()


def _tokenize(user_text: str) -> List[str]:
    s = _norm(user_text)
    s = re.sub(r"[^a-z0-9_\\s]", " ", s)
    toks = [t for t in s.split() if t]
    return list(toks)


def is_pragmatics_command_v107(user_text: str) -> bool:
    toks = _tokenize(user_text)
    return bool(toks and toks[0] in {"pragmatics", "pragmatica"} and len(toks) == 1)


def parse_pragmatics_command_v107(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks or toks[0] not in {"pragmatics", "pragmatica"}:
        return {"recognized": False}
    if len(toks) != 1:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "tokens": list(toks), "intent_id": INTENT_PRAGMATICS_V107}
    return {"recognized": True, "ok": True, "reason": "ok", "intent_id": INTENT_PRAGMATICS_V107, "prefix": str(toks[0])}


def is_explain_pragmatics_command_v107(user_text: str) -> bool:
    toks = _tokenize(user_text)
    if not toks:
        return False
    if toks[0] == "explain_pragmatics":
        return True
    if len(toks) >= 2 and toks[0] == "explain" and toks[1] in {"pragmatics", "pragmatica"}:
        return True
    if toks[0] in {"explique_pragmatics", "explique_pragmatica"}:
        return True
    if len(toks) >= 2 and toks[0] == "explique" and toks[1] in {"pragmatics", "pragmatica"}:
        return True
    return False


def parse_explain_pragmatics_command_v107(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks:
        return {"recognized": False}
    prefix = ""
    rest: List[str] = []
    if toks[0] == "explain_pragmatics":
        prefix = "explain_pragmatics"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "explain" and toks[1] in {"pragmatics", "pragmatica"}:
        prefix = "explain pragmatics"
        rest = toks[2:]
    elif toks[0] in {"explique_pragmatics", "explique_pragmatica"}:
        prefix = str(toks[0])
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "explique" and toks[1] in {"pragmatics", "pragmatica"}:
        prefix = "explique pragmatics"
        rest = toks[2:]
    else:
        return {"recognized": False}
    if len(rest) != 1:
        return {
            "recognized": True,
            "ok": False,
            "reason": "missing_turn_ref" if not rest else "extra_tokens",
            "intent_id": INTENT_EXPLAIN_PRAGMATICS_V107,
            "prefix": str(prefix),
            "tokens": list(toks),
        }
    q = str(rest[0]).strip()
    if not q:
        return {"recognized": True, "ok": False, "reason": "missing_turn_ref", "intent_id": INTENT_EXPLAIN_PRAGMATICS_V107, "prefix": str(prefix), "tokens": list(toks)}
    return {"recognized": True, "ok": True, "reason": "ok", "intent_id": INTENT_EXPLAIN_PRAGMATICS_V107, "prefix": str(prefix), "query": str(q)}


def is_trace_pragmatics_command_v107(user_text: str) -> bool:
    toks = _tokenize(user_text)
    if not toks:
        return False
    if toks[0] == "trace_pragmatics":
        return True
    if len(toks) >= 2 and toks[0] == "trace" and toks[1] in {"pragmatics", "pragmatica"}:
        return True
    return False


def parse_trace_pragmatics_command_v107(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks:
        return {"recognized": False}
    prefix = ""
    rest: List[str] = []
    if toks[0] == "trace_pragmatics":
        prefix = "trace_pragmatics"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "trace" and toks[1] in {"pragmatics", "pragmatica"}:
        prefix = "trace pragmatics"
        rest = toks[2:]
    else:
        return {"recognized": False}
    if len(rest) != 1:
        return {
            "recognized": True,
            "ok": False,
            "reason": "missing_turn_ref" if not rest else "extra_tokens",
            "intent_id": INTENT_TRACE_PRAGMATICS_V107,
            "prefix": str(prefix),
            "tokens": list(toks),
        }
    q = str(rest[0]).strip()
    if not q:
        return {"recognized": True, "ok": False, "reason": "missing_turn_ref", "intent_id": INTENT_TRACE_PRAGMATICS_V107, "prefix": str(prefix), "tokens": list(toks)}
    return {"recognized": True, "ok": True, "reason": "ok", "intent_id": INTENT_TRACE_PRAGMATICS_V107, "prefix": str(prefix), "query": str(q)}

