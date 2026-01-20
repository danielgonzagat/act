from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List


INTENT_SEMANTICS_V109 = "INTENT_SEMANTICS_V109"
INTENT_EXPLAIN_SEMANTICS_V109 = "INTENT_EXPLAIN_SEMANTICS_V109"
INTENT_TRACE_SEMANTICS_V109 = "INTENT_TRACE_SEMANTICS_V109"

INTENT_EXECUTIVE_V110 = "INTENT_EXECUTIVE_V110"
INTENT_EXPLAIN_EXECUTIVE_V110 = "INTENT_EXPLAIN_EXECUTIVE_V110"
INTENT_TRACE_EXECUTIVE_V110 = "INTENT_TRACE_EXECUTIVE_V110"


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


def is_semantics_command_v109(user_text: str) -> bool:
    toks = _tokenize(user_text)
    return bool(toks and toks[0] in {"semantics", "semantica", "semanticas"} and len(toks) == 1)


def parse_semantics_command_v109(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks or toks[0] not in {"semantics", "semantica", "semanticas"}:
        return {"recognized": False}
    if len(toks) != 1:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "tokens": list(toks), "intent_id": INTENT_SEMANTICS_V109}
    return {"recognized": True, "ok": True, "reason": "ok", "intent_id": INTENT_SEMANTICS_V109, "prefix": str(toks[0])}


def is_explain_semantics_command_v109(user_text: str) -> bool:
    toks = _tokenize(user_text)
    if not toks:
        return False
    if toks[0] == "explain_semantics":
        return True
    if len(toks) >= 2 and toks[0] == "explain" and toks[1] in {"semantics", "semantica", "semanticas"}:
        return True
    if toks[0] in {"explique_semantica", "explique_semantics"}:
        return True
    if len(toks) >= 2 and toks[0] == "explique" and toks[1] in {"semantica", "semantics", "semanticas"}:
        return True
    return False


def parse_explain_semantics_command_v109(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks:
        return {"recognized": False}
    prefix = ""
    rest: List[str] = []
    if toks[0] == "explain_semantics":
        prefix = "explain_semantics"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "explain" and toks[1] in {"semantics", "semantica", "semanticas"}:
        prefix = "explain semantics"
        rest = toks[2:]
    elif toks[0] in {"explique_semantica", "explique_semantics"}:
        prefix = str(toks[0])
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "explique" and toks[1] in {"semantica", "semantics", "semanticas"}:
        prefix = "explique semantics"
        rest = toks[2:]
    else:
        return {"recognized": False}

    if len(rest) != 1:
        return {
            "recognized": True,
            "ok": False,
            "reason": "missing_turn_ref" if not rest else "extra_tokens",
            "intent_id": INTENT_EXPLAIN_SEMANTICS_V109,
            "prefix": str(prefix),
            "tokens": list(toks),
        }
    q = str(rest[0]).strip()
    if not q:
        return {
            "recognized": True,
            "ok": False,
            "reason": "missing_turn_ref",
            "intent_id": INTENT_EXPLAIN_SEMANTICS_V109,
            "prefix": str(prefix),
            "tokens": list(toks),
        }
    return {"recognized": True, "ok": True, "reason": "ok", "intent_id": INTENT_EXPLAIN_SEMANTICS_V109, "prefix": str(prefix), "query": str(q)}


def is_trace_semantics_command_v109(user_text: str) -> bool:
    toks = _tokenize(user_text)
    if not toks:
        return False
    if toks[0] == "trace_semantics":
        return True
    if len(toks) >= 2 and toks[0] == "trace" and toks[1] in {"semantics", "semantica", "semanticas"}:
        return True
    return False


def parse_trace_semantics_command_v109(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks:
        return {"recognized": False}
    prefix = ""
    rest: List[str] = []
    if toks[0] == "trace_semantics":
        prefix = "trace_semantics"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "trace" and toks[1] in {"semantics", "semantica", "semanticas"}:
        prefix = "trace semantics"
        rest = toks[2:]
    else:
        return {"recognized": False}
    if len(rest) != 1:
        return {
            "recognized": True,
            "ok": False,
            "reason": "missing_turn_ref" if not rest else "extra_tokens",
            "intent_id": INTENT_TRACE_SEMANTICS_V109,
            "prefix": str(prefix),
            "tokens": list(toks),
        }
    q = str(rest[0]).strip()
    if not q:
        return {"recognized": True, "ok": False, "reason": "missing_turn_ref", "intent_id": INTENT_TRACE_SEMANTICS_V109, "prefix": str(prefix), "tokens": list(toks)}
    return {"recognized": True, "ok": True, "reason": "ok", "intent_id": INTENT_TRACE_SEMANTICS_V109, "prefix": str(prefix), "query": str(q)}


def is_executive_command_v110(user_text: str) -> bool:
    toks = _tokenize(user_text)
    return bool(toks and toks[0] in {"executive", "executivo"} and len(toks) == 1)


def parse_executive_command_v110(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks or toks[0] not in {"executive", "executivo"}:
        return {"recognized": False}
    if len(toks) != 1:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "tokens": list(toks), "intent_id": INTENT_EXECUTIVE_V110}
    return {"recognized": True, "ok": True, "reason": "ok", "intent_id": INTENT_EXECUTIVE_V110, "prefix": str(toks[0])}


def is_explain_executive_command_v110(user_text: str) -> bool:
    toks = _tokenize(user_text)
    if not toks:
        return False
    if toks[0] == "explain_executive":
        return True
    if len(toks) >= 2 and toks[0] == "explain" and toks[1] in {"executive", "executivo"}:
        return True
    if toks[0] in {"explique_executivo", "explique_executive"}:
        return True
    if len(toks) >= 2 and toks[0] == "explique" and toks[1] in {"executivo", "executive"}:
        return True
    return False


def parse_explain_executive_command_v110(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks:
        return {"recognized": False}
    prefix = ""
    rest: List[str] = []
    if toks[0] == "explain_executive":
        prefix = "explain_executive"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "explain" and toks[1] in {"executive", "executivo"}:
        prefix = "explain executive"
        rest = toks[2:]
    elif toks[0] in {"explique_executivo", "explique_executive"}:
        prefix = str(toks[0])
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "explique" and toks[1] in {"executivo", "executive"}:
        prefix = "explique executive"
        rest = toks[2:]
    else:
        return {"recognized": False}
    if len(rest) != 1:
        return {
            "recognized": True,
            "ok": False,
            "reason": "missing_turn_ref" if not rest else "extra_tokens",
            "intent_id": INTENT_EXPLAIN_EXECUTIVE_V110,
            "prefix": str(prefix),
            "tokens": list(toks),
        }
    q = str(rest[0]).strip()
    if not q:
        return {"recognized": True, "ok": False, "reason": "missing_turn_ref", "intent_id": INTENT_EXPLAIN_EXECUTIVE_V110, "prefix": str(prefix), "tokens": list(toks)}
    return {"recognized": True, "ok": True, "reason": "ok", "intent_id": INTENT_EXPLAIN_EXECUTIVE_V110, "prefix": str(prefix), "query": str(q)}


def is_trace_executive_command_v110(user_text: str) -> bool:
    toks = _tokenize(user_text)
    if not toks:
        return False
    if toks[0] == "trace_executive":
        return True
    if len(toks) >= 2 and toks[0] == "trace" and toks[1] in {"executive", "executivo"}:
        return True
    return False


def parse_trace_executive_command_v110(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks:
        return {"recognized": False}
    prefix = ""
    rest: List[str] = []
    if toks[0] == "trace_executive":
        prefix = "trace_executive"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "trace" and toks[1] in {"executive", "executivo"}:
        prefix = "trace executive"
        rest = toks[2:]
    else:
        return {"recognized": False}
    if len(rest) != 1:
        return {
            "recognized": True,
            "ok": False,
            "reason": "missing_turn_ref" if not rest else "extra_tokens",
            "intent_id": INTENT_TRACE_EXECUTIVE_V110,
            "prefix": str(prefix),
            "tokens": list(toks),
        }
    q = str(rest[0]).strip()
    if not q:
        return {"recognized": True, "ok": False, "reason": "missing_turn_ref", "intent_id": INTENT_TRACE_EXECUTIVE_V110, "prefix": str(prefix), "tokens": list(toks)}
    return {"recognized": True, "ok": True, "reason": "ok", "intent_id": INTENT_TRACE_EXECUTIVE_V110, "prefix": str(prefix), "query": str(q)}
