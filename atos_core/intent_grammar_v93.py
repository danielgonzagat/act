from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .intent_grammar_v92 import _IGNORABLE_TOKENS_V92, _canonize_tokens_v92, tokenize_user_text_v92


INTENT_TEACH_V93 = "INTENT_TEACH"


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _norm_prefix(text: str) -> str:
    return _strip_accents(str(text or "")).lower()


def is_teach_command_v93(user_text: str) -> bool:
    s = str(user_text or "").lstrip()
    s2 = _norm_prefix(s)
    return bool(s2.startswith("teach:") or s2.startswith("ensine:"))


def parse_teach_command_v93(user_text: str) -> Dict[str, Any]:
    """
    Parse TEACH/ENSINE command in raw text (before normalization of the main parser).
    Syntax (deterministic):
      teach: <lhs> => <rhs>
      ensine: <lhs> => <rhs>
    """
    raw = str(user_text or "")
    s = raw.lstrip()
    s2 = _norm_prefix(s)
    prefix = ""
    if s2.startswith("teach:"):
        prefix = "teach"
        rest = s[len("teach:") :].strip()
    elif s2.startswith("ensine:"):
        prefix = "ensine"
        rest = s[len("ensine:") :].strip()
    else:
        return {"recognized": False}

    parts = rest.split("=>")
    if len(parts) != 2:
        return {
            "recognized": True,
            "ok": False,
            "reason": "bad_syntax_missing_arrow",
            "prefix": str(prefix),
            "lhs_raw": "",
            "rhs_raw": "",
        }
    lhs_raw = str(parts[0]).strip()
    rhs_raw = str(parts[1]).strip()
    if not lhs_raw:
        return {
            "recognized": True,
            "ok": False,
            "reason": "empty_lhs",
            "prefix": str(prefix),
            "lhs_raw": "",
            "rhs_raw": str(rhs_raw),
        }
    if not rhs_raw:
        return {
            "recognized": True,
            "ok": False,
            "reason": "empty_rhs",
            "prefix": str(prefix),
            "lhs_raw": str(lhs_raw),
            "rhs_raw": "",
        }
    return {
        "recognized": True,
        "ok": True,
        "reason": "ok",
        "prefix": str(prefix),
        "lhs_raw": str(lhs_raw),
        "rhs_raw": str(rhs_raw),
    }


def _is_ignorable_token_v93(tok: str) -> bool:
    return str(tok) in set(_IGNORABLE_TOKENS_V92)


def canonize_lhs_for_learned_rule_v93(lhs_raw: str) -> Dict[str, Any]:
    """
    Canonicalize LHS into literal-only pattern tokens, aligned with V92:
      - tokenization/number words via tokenize_user_text_v92
      - synonym canon via _canonize_tokens_v92
      - strip ignorable tokens only on prefix/suffix
    """
    tokens_raw_norm = tokenize_user_text_v92(str(lhs_raw))
    tokens_canon, canon_map_applied = _canonize_tokens_v92(tokens_raw_norm)

    pref_raw: List[str] = []
    pref_canon: List[str] = []
    suf_raw: List[str] = []
    suf_canon: List[str] = []

    tr = list(tokens_raw_norm)
    tc = list(tokens_canon)
    while tr and tc and _is_ignorable_token_v93(str(tc[0])):
        pref_raw.append(str(tr.pop(0)))
        pref_canon.append(str(tc.pop(0)))
    while tr and tc and _is_ignorable_token_v93(str(tc[-1])):
        suf_raw.insert(0, str(tr.pop(-1)))
        suf_canon.insert(0, str(tc.pop(-1)))

    return {
        "lhs_raw": str(lhs_raw),
        "lhs_tokens_raw_norm": list(tokens_raw_norm),
        "lhs_tokens_canon": list(tokens_canon),
        "canon_map_applied": list(canon_map_applied),
        "ignored_prefix_tokens_raw": list(pref_raw),
        "ignored_prefix_tokens": list(pref_canon),
        "ignored_suffix_tokens_raw": list(suf_raw),
        "ignored_suffix_tokens": list(suf_canon),
        "lhs_tokens_canon_stripped": list(tc),
        "lhs_tokens_raw_norm_stripped": list(tr),
    }


def expected_learned_rule_id_v93(*, intent_id: str, pattern: Sequence[Dict[str, str]], required_slots: Sequence[str]) -> str:
    body = {
        "schema_version": 93,
        "intent_id": str(intent_id),
        "pattern": [dict(p) for p in pattern],
        "required_slots": [str(x) for x in required_slots],
    }
    return f"INTENT_RULE_V93_LEARNED_{sha256_hex(canonical_json_dumps(body).encode('utf-8'))}"


@dataclass(frozen=True)
class IntentRuleV93:
    rule_id: str
    intent_id: str
    pattern: List[Dict[str, str]]
    required_slots: List[str]
    examples: List[str]
    rule_sig: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 93,
            "rule_id": str(self.rule_id),
            "intent_id": str(self.intent_id),
            "pattern": [dict(p) for p in self.pattern],
            "required_slots": [str(x) for x in self.required_slots],
            "examples": [str(x) for x in self.examples],
            "rule_sig": str(self.rule_sig),
        }


def make_learned_intent_rule_v93(*, intent_id: str, lhs_tokens_canon_stripped: Sequence[str]) -> IntentRuleV93:
    pattern = [{"t": "lit", "v": str(t)} for t in lhs_tokens_canon_stripped if isinstance(t, str) and t]
    required_slots: List[str] = []
    examples: List[str] = []
    rule_id = expected_learned_rule_id_v93(intent_id=str(intent_id), pattern=pattern, required_slots=required_slots)
    body = {
        "schema_version": 93,
        "rule_id": str(rule_id),
        "intent_id": str(intent_id),
        "pattern": [dict(p) for p in pattern],
        "required_slots": list(required_slots),
        "examples": list(examples),
    }
    sig = _stable_hash_obj(body)
    return IntentRuleV93(
        rule_id=str(rule_id),
        intent_id=str(intent_id),
        pattern=[dict(p) for p in pattern],
        required_slots=list(required_slots),
        examples=list(examples),
        rule_sig=str(sig),
    )


def verify_learned_rule_sig_v93(rule_dict: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    d = dict(rule_dict)
    got_sig = str(d.pop("rule_sig", "") or "")
    want_sig = sha256_hex(canonical_json_dumps(d).encode("utf-8"))
    if not got_sig:
        return False, "missing_rule_sig", {}
    if want_sig != got_sig:
        return False, "rule_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    return True, "ok", {}

