from __future__ import annotations

import unicodedata
from typing import Any, Dict


INTENT_GOAL_ADD_V99 = "INTENT_GOAL_ADD"
INTENT_GOAL_LIST_V99 = "INTENT_GOAL_LIST"
INTENT_GOAL_DONE_V99 = "INTENT_GOAL_DONE"
INTENT_GOAL_NEXT_V99 = "INTENT_GOAL_NEXT"
INTENT_GOAL_AUTO_V99 = "INTENT_GOAL_AUTO"


def _strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _norm_prefix(text: str) -> str:
    return _strip_accents(str(text or "")).lower()


def is_goal_add_command_v99(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    if s2.startswith("goal"):
        rest = s[len("goal") :]
        return bool(rest and rest.lstrip().startswith(":"))
    return False


def parse_goal_add_command_v99(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      goal: <text>
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    if not s2.startswith("goal"):
        return {"recognized": False}
    rest_raw = s[len("goal") :]
    if rest_raw and (not rest_raw[0].isspace()) and (rest_raw[0] != ":"):
        return {"recognized": False}
    rest = str(rest_raw).lstrip()
    if not rest.startswith(":"):
        return {"recognized": True, "ok": False, "reason": "missing_colon", "prefix": "goal"}
    body = str(rest[1:]).strip()
    if not body:
        return {"recognized": True, "ok": False, "reason": "empty_goal_text", "prefix": "goal"}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": "goal", "goal_text": str(body)}


def is_goals_list_command_v99(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("goals"))


def parse_goals_list_command_v99(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      goals
      goals:   (no payload)
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    if not s2.startswith("goals"):
        return {"recognized": False}
    rest_raw = s[len("goals") :]
    if rest_raw and (not rest_raw[0].isspace()) and (rest_raw[0] != ":"):
        return {"recognized": False}
    rest = str(rest_raw).strip()
    if rest.startswith(":"):
        rest = rest[1:].strip()
    if rest:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "prefix": "goals", "extra_raw": str(rest_raw)}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": "goals"}


def is_done_command_v99(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("done"))


def parse_done_command_v99(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      done <goal_id>
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    if not s2.startswith("done"):
        return {"recognized": False}
    rest_raw = s[len("done") :]
    if rest_raw and (not rest_raw[0].isspace()):
        return {"recognized": False}
    rest = str(rest_raw).strip()
    if not rest:
        return {"recognized": True, "ok": False, "reason": "missing_goal_id", "prefix": "done"}
    toks = rest.split()
    if len(toks) != 1:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "prefix": "done", "extra_raw": str(rest_raw)}
    goal_id = str(toks[0]).strip()
    if not goal_id:
        return {"recognized": True, "ok": False, "reason": "missing_goal_id", "prefix": "done"}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": "done", "goal_id": str(goal_id)}


def is_next_command_v99(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("next"))


def parse_next_command_v99(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      next
      next:   (no payload)
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    if not s2.startswith("next"):
        return {"recognized": False}
    rest_raw = s[len("next") :]
    if rest_raw and (not rest_raw[0].isspace()) and (rest_raw[0] != ":"):
        return {"recognized": False}
    rest = str(rest_raw).strip()
    if rest.startswith(":"):
        rest = rest[1:].strip()
    if rest:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "prefix": "next", "extra_raw": str(rest_raw)}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": "next"}


def is_auto_command_v99(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("auto"))


def parse_auto_command_v99(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      auto <n>
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    if not s2.startswith("auto"):
        return {"recognized": False}
    rest_raw = s[len("auto") :]
    if rest_raw and (not rest_raw[0].isspace()):
        return {"recognized": False}
    rest = str(rest_raw).strip()
    if not rest:
        return {"recognized": True, "ok": False, "reason": "missing_n", "prefix": "auto"}
    toks = rest.split()
    if len(toks) != 1:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "prefix": "auto", "extra_raw": str(rest_raw)}
    n_raw = str(toks[0]).strip()
    if (not n_raw) or (not n_raw.isdigit()):
        return {"recognized": True, "ok": False, "reason": "bad_n", "prefix": "auto", "n_raw": str(n_raw)}
    n = int(n_raw)
    if n <= 0:
        return {"recognized": True, "ok": False, "reason": "bad_n", "prefix": "auto", "n_raw": str(n_raw)}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": "auto", "n": int(n)}

