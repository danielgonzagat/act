from __future__ import annotations

import unicodedata
from typing import Any, Dict


INTENT_SYSTEM_V98 = "INTENT_SYSTEM"
INTENT_EVIDENCE_ADD_V98 = "INTENT_EVIDENCE_ADD"
INTENT_EVIDENCE_LIST_V98 = "INTENT_EVIDENCE_LIST"
INTENT_WHY_V98 = "INTENT_WHY"
INTENT_VERSIONS_V98 = "INTENT_VERSIONS"
INTENT_DOSSIER_V98 = "INTENT_DOSSIER"


def _strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _norm_prefix(text: str) -> str:
    return _strip_accents(str(text or "")).lower()


_SYSTEM_PREFIXES_V98 = ["system", "sistema", "manual", "about"]


def is_system_command_v98(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    for p in _SYSTEM_PREFIXES_V98:
        if not s2.startswith(p):
            continue
        if len(s2) == len(p):
            return True
        nxt = s2[len(p)]
        if nxt.isspace() or nxt == ":":
            return True
    return False


def parse_system_command_v98(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed, raw intercept):
      system / system:
      sistema / sistema:
      manual / manual:
      about / about:

    No payload allowed (whitespace-only ok).
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    rest_raw = ""
    for p in _SYSTEM_PREFIXES_V98:
        if not s2.startswith(p):
            continue
        # Avoid prefix-matching inside longer words.
        if len(s2) > len(p):
            nxt = s2[len(p)]
            if (not nxt.isspace()) and nxt != ":":
                continue
        prefix = p
        rest_raw = s[len(p) :]
        break
    if not prefix:
        return {"recognized": False}

    if rest_raw and (not rest_raw[0].isspace()) and (rest_raw[0] != ":"):
        return {"recognized": False}

    rest = str(rest_raw).strip()
    if rest.startswith(":"):
        rest = rest[1:].strip()
    if rest:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "prefix": str(prefix), "extra_raw": str(rest_raw)}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix)}


def is_evidence_add_command_v98(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    if s2.startswith("evidence"):
        rest = s[len("evidence") :]
        return bool(rest and rest.lstrip().startswith(":"))
    if s2.startswith("evidencia"):
        rest = s[len("evidencia") :]
        return bool(rest and rest.lstrip().startswith(":"))
    return False


def _parse_key_value_after_colon_v98(prefix: str, rest_raw: str) -> Dict[str, Any]:
    rest = str(rest_raw).lstrip()
    if not rest.startswith(":"):
        return {"recognized": True, "ok": False, "reason": "missing_colon", "prefix": str(prefix), "extra_raw": str(rest_raw)}
    body = str(rest[1:]).strip()
    if "=" not in body:
        return {"recognized": True, "ok": False, "reason": "missing_equals", "prefix": str(prefix), "body_raw": str(body)}
    parts = body.split("=", 1)
    key = str(parts[0]).strip()
    val = str(parts[1]).strip()
    if not key:
        return {"recognized": True, "ok": False, "reason": "empty_key", "prefix": str(prefix), "body_raw": str(body)}
    if not val:
        return {"recognized": True, "ok": False, "reason": "empty_value", "prefix": str(prefix), "body_raw": str(body)}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix), "key": str(key), "value": str(val)}


def parse_evidence_add_command_v98(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      evidence: <key> = <value>
      evidencia: <key> = <value>   (evidência accepted via accent fold)

    No other forms are accepted.
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    if s2.startswith("evidence"):
        prefix = "evidence"
        rest_raw = s[len("evidence") :]
    elif s2.startswith("evidencia"):
        prefix = "evidencia"
        rest_raw = s[len("evidencia") :]
    else:
        return {"recognized": False}
    if rest_raw and (not rest_raw[0].isspace()) and (rest_raw[0] != ":"):
        return {"recognized": False}
    kv = _parse_key_value_after_colon_v98(prefix, rest_raw)
    if not bool(kv.get("recognized", False)):
        return {"recognized": False}
    if not bool(kv.get("ok", False)):
        return dict(kv, evidence_key="", evidence_value="")
    return dict(kv, evidence_key=str(kv.get("key") or ""), evidence_value=str(kv.get("value") or ""), evidence_kind="OBSERVE")


def is_evidences_list_command_v98(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("evidences") or s2.startswith("evidencias"))


def parse_evidences_list_command_v98(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      evidences
      evidencias / evidências
      (optional) evidences: / evidencias: with no payload
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    if s2.startswith("evidences"):
        prefix = "evidences"
        rest_raw = s[len("evidences") :]
    elif s2.startswith("evidencias"):
        prefix = "evidencias"
        rest_raw = s[len("evidencias") :]
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


def is_why_command_v98(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("why") or s2.startswith("porque"))


def parse_why_command_v98(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      why <key>
      porque <key>
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    rest_raw = ""
    if s2.startswith("why"):
        prefix = "why"
        rest_raw = s[len("why") :]
    elif s2.startswith("porque"):
        prefix = "porque"
        rest_raw = s[len("porque") :]
    else:
        return {"recognized": False}
    if rest_raw and (not rest_raw[0].isspace()):
        return {"recognized": False}
    rest = str(rest_raw).strip()
    if not rest:
        return {"recognized": True, "ok": False, "reason": "missing_key", "prefix": str(prefix)}
    toks = rest.split()
    if len(toks) != 1:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "prefix": str(prefix), "extra_raw": str(rest_raw)}
    key = str(toks[0]).strip()
    if not key:
        return {"recognized": True, "ok": False, "reason": "missing_key", "prefix": str(prefix)}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix), "key": str(key)}


def is_versions_command_v98(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("versions") or s2.startswith("versoes"))


def parse_versions_command_v98(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      versions
      versoes / versões
      (optional) versions: / versoes: with no payload
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    if s2.startswith("versions"):
        prefix = "versions"
        rest_raw = s[len("versions") :]
    elif s2.startswith("versoes"):
        prefix = "versoes"
        rest_raw = s[len("versoes") :]
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


def is_dossier_command_v98(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("dossier") or s2.startswith("regulatory") or s2.startswith("compliance"))


def parse_dossier_command_v98(user_text: str) -> Dict[str, Any]:
    """
    Syntax (fail-closed):
      dossier
      regulatory
      compliance
      (optional) dossier: / regulatory: / compliance: with no payload
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    rest_raw = ""
    for p in ["dossier", "regulatory", "compliance"]:
        if not s2.startswith(p):
            continue
        if len(s2) > len(p):
            nxt = s2[len(p)]
            if (not nxt.isspace()) and nxt != ":":
                continue
        prefix = p
        rest_raw = s[len(p) :]
        break
    if not prefix:
        return {"recognized": False}

    if rest_raw and (not rest_raw[0].isspace()) and (rest_raw[0] != ":"):
        return {"recognized": False}

    rest = str(rest_raw).strip()
    if rest.startswith(":"):
        rest = rest[1:].strip()
    if rest:
        return {"recognized": True, "ok": False, "reason": "extra_tokens", "prefix": str(prefix), "extra_raw": str(rest_raw)}
    return {"recognized": True, "ok": True, "reason": "ok", "prefix": str(prefix)}
