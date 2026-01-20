from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List


INTENT_PLANS_V104 = "INTENT_PLANS_V104"
INTENT_EXPLAIN_PLAN_V104 = "INTENT_EXPLAIN_PLAN_V104"
INTENT_TRACE_PLANS_V104 = "INTENT_TRACE_PLANS_V104"


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


def is_plans_command_v104(user_text: str) -> bool:
    toks = _tokenize(user_text)
    return bool(toks and toks[0] in {"plans", "planos"} and len(toks) == 1)


def parse_plans_command_v104(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks or toks[0] not in {"plans", "planos"}:
        return {"recognized": False}
    if len(toks) != 1:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "tokens": list(toks)}
    return {"recognized": True, "ok": True, "reason": "ok", "intent_id": INTENT_PLANS_V104, "prefix": str(toks[0])}


def is_explain_plan_command_v104(user_text: str) -> bool:
    toks = _tokenize(user_text)
    if not toks:
        return False
    if toks[0] == "explain_plan":
        return True
    if len(toks) >= 2 and toks[0] == "explain" and toks[1] == "plan":
        return True
    if toks[0] == "explique_plano":
        return True
    if len(toks) >= 2 and toks[0] == "explique" and toks[1] == "plano":
        return True
    return False


def parse_explain_plan_command_v104(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks:
        return {"recognized": False}
    rest: List[str] = []
    prefix = ""
    if toks[0] == "explain_plan":
        prefix = "explain_plan"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "explain" and toks[1] == "plan":
        prefix = "explain plan"
        rest = toks[2:]
    elif toks[0] == "explique_plano":
        prefix = "explique_plano"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "explique" and toks[1] == "plano":
        prefix = "explique plano"
        rest = toks[2:]
    else:
        return {"recognized": False}
    if len(rest) != 1:
        return {
            "recognized": True,
            "ok": False,
            "reason": "missing_turn_ref" if not rest else "extra_tokens",
            "intent_id": INTENT_EXPLAIN_PLAN_V104,
            "prefix": str(prefix),
            "tokens": list(toks),
        }
    q = str(rest[0]).strip()
    if not q:
        return {"recognized": True, "ok": False, "reason": "missing_turn_ref", "intent_id": INTENT_EXPLAIN_PLAN_V104, "prefix": str(prefix), "tokens": list(toks)}
    return {"recognized": True, "ok": True, "reason": "ok", "intent_id": INTENT_EXPLAIN_PLAN_V104, "prefix": str(prefix), "query": str(q)}


def is_trace_plans_command_v104(user_text: str) -> bool:
    toks = _tokenize(user_text)
    if not toks:
        return False
    if toks[0] == "trace_plans":
        return True
    if len(toks) >= 2 and toks[0] == "trace" and toks[1] == "plans":
        return True
    if toks[0] == "trace_planos":
        return True
    if len(toks) >= 2 and toks[0] == "trace" and toks[1] == "planos":
        return True
    return False


def parse_trace_plans_command_v104(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks:
        return {"recognized": False}
    rest: List[str] = []
    prefix = ""
    if toks[0] == "trace_plans":
        prefix = "trace_plans"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "trace" and toks[1] == "plans":
        prefix = "trace plans"
        rest = toks[2:]
    elif toks[0] == "trace_planos":
        prefix = "trace_planos"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "trace" and toks[1] == "planos":
        prefix = "trace planos"
        rest = toks[2:]
    else:
        return {"recognized": False}
    if len(rest) != 1:
        return {
            "recognized": True,
            "ok": False,
            "reason": "missing_turn_ref" if not rest else "extra_tokens",
            "intent_id": INTENT_TRACE_PLANS_V104,
            "prefix": str(prefix),
            "tokens": list(toks),
        }
    q = str(rest[0]).strip()
    if not q:
        return {"recognized": True, "ok": False, "reason": "missing_turn_ref", "intent_id": INTENT_TRACE_PLANS_V104, "prefix": str(prefix), "tokens": list(toks)}
    return {"recognized": True, "ok": True, "reason": "ok", "intent_id": INTENT_TRACE_PLANS_V104, "prefix": str(prefix), "query": str(q)}

