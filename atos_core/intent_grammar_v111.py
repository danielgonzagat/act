from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .act import canonical_json_dumps, sha256_hex
from .conversation_v96 import normalize_text_v96


INTENT_EXTERNAL_WORLD_V111 = "INTENT_EXTERNAL_WORLD_V111"
INTENT_EXPLAIN_EXTERNAL_WORLD_V111 = "INTENT_EXPLAIN_EXTERNAL_WORLD_V111"
INTENT_TRACE_EXTERNAL_WORLD_V111 = "INTENT_TRACE_EXTERNAL_WORLD_V111"


def _norm(s: str) -> str:
    return normalize_text_v96(str(s or ""))


def is_external_world_command_v111(text: str) -> bool:
    t = _norm(text)
    return t in {"external_world", "mundo_externo"}


def is_explain_external_world_command_v111(text: str) -> bool:
    t = _norm(text)
    return t.startswith("explain_external_world")


def is_trace_external_world_command_v111(text: str) -> bool:
    t = _norm(text)
    return t.startswith("trace_external_world")


def parse_explain_external_world_command_v111(text: str) -> Dict[str, Any]:
    t = _norm(text)
    parts = t.split()
    if len(parts) == 2 and parts[0] == "explain_external_world":
        return {"intent_id": INTENT_EXPLAIN_EXTERNAL_WORLD_V111, "query": parts[1]}
    return {"intent_id": INTENT_EXPLAIN_EXTERNAL_WORLD_V111, "query": ""}


def parse_trace_external_world_command_v111(text: str) -> Dict[str, Any]:
    t = _norm(text)
    parts = t.split()
    if len(parts) == 2 and parts[0] == "trace_external_world":
        return {"intent_id": INTENT_TRACE_EXTERNAL_WORLD_V111, "query": parts[1]}
    return {"intent_id": INTENT_TRACE_EXTERNAL_WORLD_V111, "query": ""}


def parse_external_world_command_v111(text: str) -> Dict[str, Any]:
    if is_external_world_command_v111(text):
        return {"intent_id": INTENT_EXTERNAL_WORLD_V111}
    if is_explain_external_world_command_v111(text):
        return parse_explain_external_world_command_v111(text)
    if is_trace_external_world_command_v111(text):
        return parse_trace_external_world_command_v111(text)
    return {"intent_id": "", "error": "not_external_world_command"}

