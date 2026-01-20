from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List


INTENT_AGENCY_V105 = "INTENT_AGENCY_V105"
INTENT_EXPLAIN_AGENCY_V105 = "INTENT_EXPLAIN_AGENCY_V105"
INTENT_TRACE_AGENCY_V105 = "INTENT_TRACE_AGENCY_V105"


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


def is_agency_command_v105(user_text: str) -> bool:
    toks = _tokenize(user_text)
    return bool(toks and toks[0] in {"agency", "agencia"} and len(toks) == 1)


def parse_agency_command_v105(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks or toks[0] not in {"agency", "agencia"}:
        return {"recognized": False}
    if len(toks) != 1:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "tokens": list(toks), "intent_id": INTENT_AGENCY_V105}
    return {"recognized": True, "ok": True, "reason": "ok", "intent_id": INTENT_AGENCY_V105, "prefix": str(toks[0])}


def is_explain_agency_command_v105(user_text: str) -> bool:
    toks = _tokenize(user_text)
    if not toks:
        return False
    if toks[0] == "explain_agency":
        return True
    if len(toks) >= 2 and toks[0] == "explain" and toks[1] == "agency":
        return True
    if toks[0] == "explique_agencia":
        return True
    if len(toks) >= 2 and toks[0] == "explique" and toks[1] == "agencia":
        return True
    return False


def parse_explain_agency_command_v105(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks:
        return {"recognized": False}
    prefix = ""
    rest: List[str] = []
    if toks[0] == "explain_agency":
        prefix = "explain_agency"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "explain" and toks[1] == "agency":
        prefix = "explain agency"
        rest = toks[2:]
    elif toks[0] == "explique_agencia":
        prefix = "explique_agencia"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "explique" and toks[1] == "agencia":
        prefix = "explique agencia"
        rest = toks[2:]
    else:
        return {"recognized": False}
    if len(rest) != 1:
        return {"recognized": True, "ok": False, "reason": "missing_turn_ref" if not rest else "extra_tokens", "intent_id": INTENT_EXPLAIN_AGENCY_V105, "prefix": str(prefix), "tokens": list(toks)}
    q = str(rest[0]).strip()
    if not q:
        return {"recognized": True, "ok": False, "reason": "missing_turn_ref", "intent_id": INTENT_EXPLAIN_AGENCY_V105, "prefix": str(prefix), "tokens": list(toks)}
    return {"recognized": True, "ok": True, "reason": "ok", "intent_id": INTENT_EXPLAIN_AGENCY_V105, "prefix": str(prefix), "query": str(q)}


def is_trace_agency_command_v105(user_text: str) -> bool:
    toks = _tokenize(user_text)
    if not toks:
        return False
    if toks[0] == "trace_agency":
        return True
    if len(toks) >= 2 and toks[0] == "trace" and toks[1] == "agency":
        return True
    if toks[0] == "trace_agencia":
        return True
    if len(toks) >= 2 and toks[0] == "trace" and toks[1] == "agencia":
        return True
    return False


def parse_trace_agency_command_v105(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks:
        return {"recognized": False}
    prefix = ""
    rest: List[str] = []
    if toks[0] == "trace_agency":
        prefix = "trace_agency"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "trace" and toks[1] == "agency":
        prefix = "trace agency"
        rest = toks[2:]
    elif toks[0] == "trace_agencia":
        prefix = "trace_agencia"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "trace" and toks[1] == "agencia":
        prefix = "trace agencia"
        rest = toks[2:]
    else:
        return {"recognized": False}
    if len(rest) != 1:
        return {"recognized": True, "ok": False, "reason": "missing_turn_ref" if not rest else "extra_tokens", "intent_id": INTENT_TRACE_AGENCY_V105, "prefix": str(prefix), "tokens": list(toks)}
    q = str(rest[0]).strip()
    if not q:
        return {"recognized": True, "ok": False, "reason": "missing_turn_ref", "intent_id": INTENT_TRACE_AGENCY_V105, "prefix": str(prefix), "tokens": list(toks)}
    return {"recognized": True, "ok": True, "reason": "ok", "intent_id": INTENT_TRACE_AGENCY_V105, "prefix": str(prefix), "query": str(q)}

