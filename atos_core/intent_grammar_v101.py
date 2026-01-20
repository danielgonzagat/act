from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List


INTENT_PLAN_CREATE_V101 = "INTENT_PLAN_CREATE_V101"
INTENT_PLAN_SHORTEN_REF_V101 = "INTENT_PLAN_SHORTEN_REF_V101"
INTENT_GOAL_PRIORITY_HIGH_REF_V101 = "INTENT_GOAL_PRIORITY_HIGH_REF_V101"
INTENT_BINDINGS_LIST_V101 = "INTENT_BINDINGS_LIST_V101"
INTENT_EXPLAIN_BINDING_V101 = "INTENT_EXPLAIN_BINDING_V101"
INTENT_TRACE_REF_V101 = "INTENT_TRACE_REF_V101"


def _strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _norm(text: str) -> str:
    return _strip_accents(str(text or "")).lower()


def _tokenize(user_text: str) -> List[str]:
    # Remove simple punctuation deterministically and split by whitespace.
    s = _norm(user_text)
    s = re.sub(r"[^a-z0-9_\\s]", " ", s)
    toks = [t for t in s.split() if t]
    return list(toks)


def is_bindings_command_v101(user_text: str) -> bool:
    toks = _tokenize(user_text)
    return bool(toks and toks[0] in {"bindings", "vinculos"} and len(toks) == 1)


def parse_bindings_command_v101(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks or toks[0] not in {"bindings", "vinculos"}:
        return {"recognized": False}
    if len(toks) != 1:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "tokens": list(toks)}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(toks[0])}


def is_explain_binding_command_v101(user_text: str) -> bool:
    toks = _tokenize(user_text)
    if not toks:
        return False
    if toks[0] == "explain_binding":
        return True
    if len(toks) >= 2 and toks[0] == "explain" and toks[1] == "binding":
        return True
    if toks[0] == "explique_vinculo":
        return True
    return False


def parse_explain_binding_command_v101(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks:
        return {"recognized": False}
    rest: List[str] = []
    prefix = ""
    if toks[0] == "explain_binding":
        prefix = "explain_binding"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "explain" and toks[1] == "binding":
        prefix = "explain binding"
        rest = toks[2:]
    elif toks[0] == "explique_vinculo":
        prefix = "explique_vinculo"
        rest = toks[1:]
    else:
        return {"recognized": False}
    if len(rest) != 1:
        return {"recognized": True, "ok": False, "reason": "missing_binding_id" if not rest else "extra_tokens", "prefix": str(prefix), "tokens": list(toks)}
    bid = str(rest[0]).strip()
    if not bid:
        return {"recognized": True, "ok": False, "reason": "missing_binding_id", "prefix": str(prefix), "tokens": list(toks)}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix), "binding_id": str(bid)}


def is_trace_ref_command_v101(user_text: str) -> bool:
    toks = _tokenize(user_text)
    if not toks:
        return False
    if toks[0] == "trace_ref":
        return True
    if len(toks) >= 2 and toks[0] == "trace" and toks[1] == "ref":
        return True
    return False


def parse_trace_ref_command_v101(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks:
        return {"recognized": False}
    rest: List[str] = []
    prefix = ""
    if toks[0] == "trace_ref":
        prefix = "trace_ref"
        rest = toks[1:]
    elif len(toks) >= 2 and toks[0] == "trace" and toks[1] == "ref":
        prefix = "trace ref"
        rest = toks[2:]
    else:
        return {"recognized": False}
    if len(rest) != 1:
        return {"recognized": True, "ok": False, "reason": "missing_turn_id" if not rest else "extra_tokens", "prefix": str(prefix), "tokens": list(toks)}
    tid = str(rest[0]).strip()
    if not tid:
        return {"recognized": True, "ok": False, "reason": "missing_turn_id", "prefix": str(prefix), "tokens": list(toks)}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix), "turn_id": str(tid)}


def is_plan_create_3_command_v101(user_text: str) -> bool:
    toks = _tokenize(user_text)
    # PT: crie (um) plano de 3 passos para X
    if toks[:7] == ["crie", "um", "plano", "de", "3", "passos", "para"] and len(toks) == 8:
        return True
    if toks[:6] == ["crie", "plano", "de", "3", "passos", "para"] and len(toks) == 7:
        return True
    # EN: create a 3 step plan for X
    if toks[:6] == ["create", "a", "3", "step", "plan", "for"] and len(toks) == 7:
        return True
    if toks[:6] == ["create", "a", "3", "steps", "plan", "for"] and len(toks) == 7:
        return True
    return False


def parse_plan_create_3_command_v101(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks:
        return {"recognized": False}
    target = ""
    if toks[:7] == ["crie", "um", "plano", "de", "3", "passos", "para"] and len(toks) == 8:
        target = str(toks[7])
    elif toks[:6] == ["crie", "plano", "de", "3", "passos", "para"] and len(toks) == 7:
        target = str(toks[6])
    elif toks[:6] == ["create", "a", "3", "step", "plan", "for"] and len(toks) == 7:
        target = str(toks[6])
    elif toks[:6] == ["create", "a", "3", "steps", "plan", "for"] and len(toks) == 7:
        target = str(toks[6])
    else:
        return {"recognized": False}
    if not target:
        return {"recognized": True, "ok": False, "reason": "missing_target", "tokens": list(toks)}
    return {"recognized": True, "ok": True, "reason": "ok", "target": str(target), "steps_total": 3, "tokens": list(toks)}


def is_shorten_ref_command_v101(user_text: str) -> bool:
    toks = _tokenize(user_text)
    # PT: (agora) faz isso mais curto
    if toks[:5] == ["agora", "faz", "isso", "mais", "curto"] and len(toks) == 5:
        return True
    if toks[:4] == ["faz", "isso", "mais", "curto"] and len(toks) == 4:
        return True
    # EN: make it shorter
    if toks[:3] == ["make", "it", "shorter"] and len(toks) == 3:
        return True
    return False


def parse_shorten_ref_command_v101(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks:
        return {"recognized": False}
    pron = ""
    if toks[:5] == ["agora", "faz", "isso", "mais", "curto"] and len(toks) == 5:
        pron = "isso"
    elif toks[:4] == ["faz", "isso", "mais", "curto"] and len(toks) == 4:
        pron = "isso"
    elif toks[:3] == ["make", "it", "shorter"] and len(toks) == 3:
        pron = "it"
    else:
        return {"recognized": False}
    return {"recognized": True, "ok": True, "reason": "ok", "pronoun": str(pron), "tokens": list(toks)}


def is_priority_high_ref_command_v101(user_text: str) -> bool:
    toks = _tokenize(user_text)
    # PT: (ok) (e) isso e prioridade alta
    if toks and toks[-2:] == ["prioridade", "alta"]:
        # require a pronoun earlier
        return ("isso" in toks) or ("aquilo" in toks) or ("isto" in toks)
    # EN: set this as high priority
    if toks[:4] == ["set", "this", "as", "high"] and "priority" in toks:
        return True
    return False


def parse_priority_high_ref_command_v101(user_text: str) -> Dict[str, Any]:
    toks = _tokenize(user_text)
    if not toks:
        return {"recognized": False}
    if toks and toks[-2:] == ["prioridade", "alta"]:
        pron = ""
        for p in ["isso", "aquilo", "isto", "esse", "este", "esta"]:
            if p in toks:
                pron = p
                break
        if not pron:
            return {"recognized": True, "ok": False, "reason": "missing_pronoun", "tokens": list(toks)}
        return {"recognized": True, "ok": True, "reason": "ok", "pronoun": str(pron), "priority": "high", "tokens": list(toks)}
    if toks[:4] == ["set", "this", "as", "high"] and "priority" in toks:
        return {"recognized": True, "ok": True, "reason": "ok", "pronoun": "this", "priority": "high", "tokens": list(toks)}
    return {"recognized": False}
