from __future__ import annotations

import unicodedata
from typing import Any, Dict


INTENT_NOTE_V96 = "INTENT_NOTE"
INTENT_RECALL_V96 = "INTENT_RECALL"
INTENT_FORGET_V96 = "INTENT_FORGET"  # memory: "forget last"

INTENT_BELIEF_ADD_V96 = "INTENT_BELIEF_ADD"
INTENT_BELIEF_REVISE_V96 = "INTENT_BELIEF_REVISE"
INTENT_BELIEF_LIST_V96 = "INTENT_BELIEF_LIST"
INTENT_BELIEF_FORGET_V96 = "INTENT_BELIEF_FORGET"


def _strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _norm_prefix(text: str) -> str:
    return _strip_accents(str(text or "")).lower()


def is_note_command_v96(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("note") or s2.startswith("nota"))


def parse_note_command_v96(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
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
        return {"recognized": False}

    rest = str(rest_raw).lstrip()
    if not rest.startswith(":"):
        return {"recognized": True, "ok": False, "reason": "missing_colon", "prefix": str(prefix), "extra_raw": str(rest_raw)}

    text = str(rest[1:]).strip()
    if not text:
        return {"recognized": True, "ok": False, "reason": "empty_text", "prefix": str(prefix)}

    return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix), "memory_text_raw": str(text)}


def is_recall_command_v96(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("recall") or s2.startswith("memoria"))


def parse_recall_command_v96(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      recall
      memoria / memória
      (optional) recall: / memoria: with no payload
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


def is_belief_add_command_v96(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    if s2.startswith("belief"):
        rest = s[len("belief") :]
        return bool(rest and rest.lstrip().startswith(":"))
    if s2.startswith("crenca"):
        rest = s[len("crenca") :]
        return bool(rest and rest.lstrip().startswith(":"))
    return False


def _parse_key_value_after_colon(prefix: str, rest_raw: str) -> Dict[str, Any]:
    rest = str(rest_raw).lstrip()
    if not rest.startswith(":"):
        return {"recognized": True, "ok": False, "reason": "missing_colon", "prefix": str(prefix), "extra_raw": str(rest_raw)}
    body = str(rest[1:]).strip()
    if "=" not in body:
        return {"recognized": True, "ok": False, "reason": "missing_equals", "prefix": str(prefix), "body_raw": str(body)}
    k_raw, v_raw = body.split("=", 1)
    key = str(k_raw).strip()
    val = str(v_raw).strip()
    if not key:
        return {"recognized": True, "ok": False, "reason": "empty_key", "prefix": str(prefix), "body_raw": str(body)}
    if not val:
        return {"recognized": True, "ok": False, "reason": "empty_value", "prefix": str(prefix), "body_raw": str(body), "belief_key": str(key)}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix), "belief_key": str(key), "belief_value": str(val)}


def parse_belief_add_command_v96(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      belief: <key> = <value>
      crenca: <key> = <value>   (crença accepted via accent fold)
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    if s2.startswith("belief"):
        prefix = "belief"
        rest_raw = s[len("belief") :]
    elif s2.startswith("crenca"):
        prefix = "crenca"
        rest_raw = s[len("crenca") :]
    else:
        return {"recognized": False}

    # Avoid prefix as part of a larger token (e.g. "beliefs").
    if rest_raw and (not rest_raw[0].isspace()) and (rest_raw[0] != ":"):
        return {"recognized": False}

    return _parse_key_value_after_colon(str(prefix), str(rest_raw))


def is_belief_revise_command_v96(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    if s2.startswith("revise"):
        rest = s[len("revise") :]
        return bool(rest and rest.lstrip().startswith(":"))
    if s2.startswith("revisar"):
        rest = s[len("revisar") :]
        return bool(rest and rest.lstrip().startswith(":"))
    return False


def parse_belief_revise_command_v96(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      revise: <key> = <value>
      revisar: <key> = <value>
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    if s2.startswith("revise"):
        prefix = "revise"
        rest_raw = s[len("revise") :]
    elif s2.startswith("revisar"):
        prefix = "revisar"
        rest_raw = s[len("revisar") :]
    else:
        return {"recognized": False}

    if rest_raw and (not rest_raw[0].isspace()) and (rest_raw[0] != ":"):
        return {"recognized": False}

    return _parse_key_value_after_colon(str(prefix), str(rest_raw))


def is_beliefs_list_command_v96(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("beliefs") or s2.startswith("crencas"))


def parse_beliefs_list_command_v96(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      beliefs
      crencas / crenças
      (optional) beliefs: / crencas: with no payload
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    if s2.startswith("beliefs"):
        prefix = "beliefs"
        rest_raw = s[len("beliefs") :]
    elif s2.startswith("crencas"):
        prefix = "crencas"
        rest_raw = s[len("crencas") :]
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


def is_forget_command_v96(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("forget") or s2.startswith("esquece"))


def parse_forget_command_v96(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      forget last              -> memory
      esquece last             -> memory
      forget belief <key>      -> belief
      esquece crenca <key>     -> belief (crença accepted via accent fold)
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

    t0 = _norm_prefix(toks[0])
    if t0 == "last":
        if len(toks) != 1:
            return {"recognized": True, "ok": False, "reason": "extra_tokens", "prefix": str(prefix), "extra_raw": str(rest_raw)}
        return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix), "target_kind": "memory", "target": "last"}

    if t0 in ("belief", "crenca"):
        if len(toks) != 2:
            return {"recognized": True, "ok": False, "reason": "extra_tokens", "prefix": str(prefix), "extra_raw": str(rest_raw)}
        key = str(toks[1]).strip()
        if not key:
            return {"recognized": True, "ok": False, "reason": "empty_key", "prefix": str(prefix)}
        return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix), "target_kind": "belief", "belief_key": str(key)}

    return {"recognized": True, "ok": False, "reason": "bad_target", "prefix": str(prefix), "target": str(toks[0])}

