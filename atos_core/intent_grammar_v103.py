from __future__ import annotations

import unicodedata
from typing import Any, Dict


INTENT_STYLE_PROFILE_V102 = "INTENT_STYLE_PROFILE"
INTENT_TEMPLATES_V102 = "INTENT_TEMPLATES"
INTENT_EXPLAIN_STYLE_V102 = "INTENT_EXPLAIN_STYLE"
INTENT_TRACE_STYLE_V102 = "INTENT_TRACE_STYLE"

INTENT_TEACH_CONCEPT_V103 = "INTENT_TEACH_CONCEPT"
INTENT_CONCEPTS_V103 = "INTENT_CONCEPTS"
INTENT_EXPLAIN_CONCEPT_V103 = "INTENT_EXPLAIN_CONCEPT"
INTENT_TRACE_CONCEPTS_V103 = "INTENT_TRACE_CONCEPTS"


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


def is_concepts_command_v103(user_text: str) -> bool:
    s = _norm_prefix(user_text)
    return bool(s in {"concepts", "conceitos", "concepts:", "conceitos:"})


def parse_concepts_command_v103(user_text: str) -> Dict[str, Any]:
    s = _norm_prefix(user_text)
    if not is_concepts_command_v103(s):
        return {"recognized": False}
    # Fail-closed on payload.
    if ":" in s and s not in {"concepts:", "conceitos:"}:
        return {"recognized": True, "ok": False, "reason": "bad_syntax_payload_not_allowed"}
    return {"recognized": True, "ok": True, "reason": "ok"}


def is_explain_concept_command_v103(user_text: str) -> bool:
    s = _norm_prefix(user_text)
    return bool(
        s.startswith("explain_concept")
        or s.startswith("explain concept")
        or s.startswith("explique_conceito")
        or s.startswith("explique conceito")
    )


def parse_explain_concept_command_v103(user_text: str) -> Dict[str, Any]:
    raw = str(user_text or "").strip()
    s = _norm_prefix(raw)
    if not is_explain_concept_command_v103(s):
        return {"recognized": False}
    parts = (
        s.replace("explain concept", "explain_concept")
        .replace("explique conceito", "explain_concept")
        .replace("explique_conceito", "explain_concept")
        .split()
    )
    if len(parts) == 1:
        return {"recognized": True, "ok": False, "reason": "missing_query", "query": ""}
    if len(parts) != 2:
        return {"recognized": True, "ok": False, "reason": "bad_arity", "query": ""}
    q = str(parts[1] or "").strip()
    if not q:
        return {"recognized": True, "ok": False, "reason": "missing_query", "query": ""}
    return {"recognized": True, "ok": True, "reason": "ok", "query": q}


def is_trace_concepts_command_v103(user_text: str) -> bool:
    s = _norm_prefix(user_text)
    return bool(
        s.startswith("trace_concepts")
        or s.startswith("trace concepts")
        or s.startswith("tracar_conceitos")
        or s.startswith("tracar conceitos")
    )


def parse_trace_concepts_command_v103(user_text: str) -> Dict[str, Any]:
    raw = str(user_text or "").strip()
    s = _norm_prefix(raw)
    if not is_trace_concepts_command_v103(s):
        return {"recognized": False}
    parts = (
        s.replace("trace concepts", "trace_concepts")
        .replace("tracar conceitos", "trace_concepts")
        .replace("tracar_conceitos", "trace_concepts")
        .split()
    )
    if len(parts) == 1:
        return {"recognized": True, "ok": False, "reason": "missing_turn_ref", "turn_ref": ""}
    if len(parts) != 2:
        return {"recognized": True, "ok": False, "reason": "bad_arity", "turn_ref": ""}
    tr = str(parts[1] or "").strip()
    if not tr:
        return {"recognized": True, "ok": False, "reason": "missing_turn_ref", "turn_ref": ""}
    return {"recognized": True, "ok": True, "reason": "ok", "turn_ref": tr}


def is_teach_concept_command_v103(user_text: str) -> bool:
    s = _norm_prefix(user_text)
    return bool(s.startswith("teach_concept:") or s.startswith("ensine_conceito:"))


def parse_teach_concept_command_v103(user_text: str) -> Dict[str, Any]:
    raw = str(user_text or "").strip()
    s = _norm_prefix(raw)
    if not is_teach_concept_command_v103(s):
        return {"recognized": False}
    # Fail-closed raw parsing; accepts: teach_concept: NAME += TEXT (or -=)
    # We parse from raw to preserve '=' and punctuation in the example text.
    # NOTE: NAME and TEXT are trimmed; surrounding quotes on TEXT are optional and removed.
    lowered = _strip_accents(raw).lower().strip()
    # Find the first ':' to separate prefix.
    i = lowered.find(":")
    rest = raw[i + 1 :].strip() if i >= 0 else ""
    # Split by ' +=' or ' -=' (deterministic).
    op = "+=" if "+=" in rest else "-=" if "-=" in rest else ""
    if not op:
        return {"recognized": True, "ok": False, "reason": "missing_op", "name": "", "polarity": "", "text": ""}
    parts = rest.split(op, 1)
    if len(parts) != 2:
        return {"recognized": True, "ok": False, "reason": "bad_syntax", "name": "", "polarity": "", "text": ""}
    name = str(parts[0] or "").strip()
    txt = str(parts[1] or "").strip()
    if txt.startswith('\"') and txt.endswith('\"') and len(txt) >= 2:
        txt = txt[1:-1]
    if not name or not txt:
        return {"recognized": True, "ok": False, "reason": "empty_name_or_text", "name": name, "polarity": "", "text": txt}
    polarity = "+" if op == "+=" else "-"
    return {"recognized": True, "ok": True, "reason": "ok", "name": name, "polarity": polarity, "text": txt}
