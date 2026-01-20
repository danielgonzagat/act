from __future__ import annotations

import unicodedata
from typing import Any, Dict


INTENT_STYLE_PROFILE_V102 = "INTENT_STYLE_PROFILE"
INTENT_TEMPLATES_V102 = "INTENT_TEMPLATES"
INTENT_EXPLAIN_STYLE_V102 = "INTENT_EXPLAIN_STYLE"
INTENT_TRACE_STYLE_V102 = "INTENT_TRACE_STYLE"


def _strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _norm_prefix(text: str) -> str:
    return _strip_accents(str(text or "")).lower().strip()


def is_style_profile_command_v102(user_text: str) -> bool:
    s = _norm_prefix(user_text)
    return bool(s in {"style_profile", "style profile", "perfil", "perfil de estilo", "estilo"})


def parse_style_profile_command_v102(user_text: str) -> Dict[str, Any]:
    s = _norm_prefix(user_text)
    if not is_style_profile_command_v102(s):
        return {"recognized": False}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": s.split(" ")[0] if s else "style_profile"}


def is_templates_command_v102(user_text: str) -> bool:
    s = _norm_prefix(user_text)
    return bool(s in {"templates", "modelos", "templates:", "modelos:"})


def parse_templates_command_v102(user_text: str) -> Dict[str, Any]:
    s = _norm_prefix(user_text)
    if not is_templates_command_v102(s):
        return {"recognized": False}
    # Fail-closed on payload.
    if ":" in s and s not in {"templates:", "modelos:"}:
        return {"recognized": True, "ok": False, "reason": "bad_syntax_payload_not_allowed", "prefix": s.split(":")[0]}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": s.split(":")[0]}


def is_explain_style_command_v102(user_text: str) -> bool:
    s = _norm_prefix(user_text)
    return bool(s.startswith("explain_style") or s.startswith("explain style") or s.startswith("explique_estilo") or s.startswith("explique estilo"))


def parse_explain_style_command_v102(user_text: str) -> Dict[str, Any]:
    raw = str(user_text or "").strip()
    s = _norm_prefix(raw)
    if not is_explain_style_command_v102(s):
        return {"recognized": False}
    # Accept forms: "explain_style <turn_id>" or "explique estilo <turn_id>".
    parts = s.replace("explain_style", "explain_style").replace("explain style", "explain_style").replace("explique_estilo", "explain_style").replace("explique estilo", "explain_style").split()
    if len(parts) == 1:
        return {"recognized": True, "ok": False, "reason": "missing_turn_id", "turn_id": ""}
    if len(parts) != 2:
        return {"recognized": True, "ok": False, "reason": "bad_arity", "turn_id": ""}
    turn_id = str(parts[1] or "")
    if not turn_id:
        return {"recognized": True, "ok": False, "reason": "missing_turn_id", "turn_id": ""}
    return {"recognized": True, "ok": True, "reason": "ok", "turn_id": turn_id}


def is_trace_style_command_v102(user_text: str) -> bool:
    s = _norm_prefix(user_text)
    return bool(s.startswith("trace_style") or s.startswith("trace style") or s.startswith("tracar_estilo") or s.startswith("tracar estilo"))


def parse_trace_style_command_v102(user_text: str) -> Dict[str, Any]:
    raw = str(user_text or "").strip()
    s = _norm_prefix(raw)
    if not is_trace_style_command_v102(s):
        return {"recognized": False}
    parts = s.replace("trace style", "trace_style").replace("tracar estilo", "trace_style").replace("tracar_estilo", "trace_style").split()
    if len(parts) == 1:
        return {"recognized": True, "ok": False, "reason": "missing_turn_id", "turn_id": ""}
    if len(parts) != 2:
        return {"recognized": True, "ok": False, "reason": "bad_arity", "turn_id": ""}
    turn_id = str(parts[1] or "")
    if not turn_id:
        return {"recognized": True, "ok": False, "reason": "missing_turn_id", "turn_id": ""}
    return {"recognized": True, "ok": True, "reason": "ok", "turn_id": turn_id}

