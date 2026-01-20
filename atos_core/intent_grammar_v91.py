from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import Act, canonical_json_dumps, deterministic_iso, sha256_hex


INTENT_SET_V91 = "INTENT_SET"
INTENT_GET_V91 = "INTENT_GET"
INTENT_ADD_V91 = "INTENT_ADD"
INTENT_SUMMARY_V91 = "INTENT_SUMMARY"
INTENT_END_V91 = "INTENT_END"
INTENT_UNKNOWN_V91 = "INTENT_UNKNOWN"


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _strip_accents(text: str) -> str:
    # Deterministic accent-folding (stdlib only).
    s = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


_NUM_WORDS: Dict[str, str] = {
    # English
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    # Portuguese (accentless canonical)
    "um": "1",
    "uma": "1",
    "dois": "2",
    "duas": "2",
    "tres": "3",
    "quatro": "4",
    "cinco": "5",
    "seis": "6",
    "sete": "7",
    "oito": "8",
    "nove": "9",
    "dez": "10",
    "onze": "11",
    "doze": "12",
    "treze": "13",
    "catorze": "14",
    "quatorze": "14",
    "quinze": "15",
    "dezesseis": "16",
    "dezessete": "17",
    "dezoito": "18",
    "dezenove": "19",
    "vinte": "20",
    # Common polite tokens we intentionally ignore are NOT removed (fail-closed).
}


def normalize_user_text_v91(text: str) -> str:
    """
    Deterministic normalization for intent parsing:
      - lowercase
      - accent fold
      - keep '=' and '+' as explicit tokens (surrounded by spaces)
      - remove simple punctuation by turning it into whitespace
    """
    s = _strip_accents(str(text or "")).lower()
    s = s.replace("=", " = ")
    s = s.replace("+", " + ")
    for ch in [".", ",", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "\"", "'"]:
        s = s.replace(ch, " ")
    s = " ".join(x for x in s.split(" ") if x)
    return s.strip()


def tokenize_user_text_v91(text: str) -> List[str]:
    toks = [t for t in normalize_user_text_v91(text).split(" ") if t]
    out: List[str] = []
    for t in toks:
        tt = str(t)
        out.append(_NUM_WORDS.get(tt, tt))
    return out


def lit(tok: str) -> Dict[str, str]:
    return {"t": "lit", "v": str(tok)}


def slot(name: str) -> Dict[str, str]:
    return {"t": "slot", "n": str(name)}


@dataclass(frozen=True)
class IntentRuleV91:
    rule_id: str
    intent_id: str
    pattern: List[Dict[str, str]]
    required_slots: List[str]
    examples: List[str]
    rule_sig: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "rule_id": str(self.rule_id),
            "intent_id": str(self.intent_id),
            "pattern": [dict(p) for p in self.pattern],
            "required_slots": [str(x) for x in self.required_slots],
            "examples": [str(x) for x in self.examples],
            "rule_sig": str(self.rule_sig),
        }


def make_intent_rule_v91(
    *,
    rule_id: str,
    intent_id: str,
    pattern: Sequence[Dict[str, str]],
    required_slots: Sequence[str],
    examples: Sequence[str],
) -> IntentRuleV91:
    body = {
        "schema_version": 1,
        "rule_id": str(rule_id),
        "intent_id": str(intent_id),
        "pattern": [dict(p) for p in pattern],
        "required_slots": [str(x) for x in required_slots],
        "examples": [str(x) for x in examples],
    }
    sig = _stable_hash_obj(body)
    return IntentRuleV91(
        rule_id=str(rule_id),
        intent_id=str(intent_id),
        pattern=[dict(p) for p in pattern],
        required_slots=[str(x) for x in required_slots],
        examples=[str(x) for x in examples],
        rule_sig=str(sig),
    )


def default_intent_rules_v91() -> List[IntentRuleV91]:
    """
    Deterministic, minimal intent grammar (EN+PT) with stable rule_ids.
    """
    rules: List[IntentRuleV91] = []

    # SET
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_SET_SET_TO",
            intent_id=INTENT_SET_V91,
            pattern=[lit("set"), slot("k"), lit("to"), slot("v")],
            required_slots=["k", "v"],
            examples=["set x to 4", "set y to four"],
        )
    )
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_SET_SET",
            intent_id=INTENT_SET_V91,
            pattern=[lit("set"), slot("k"), slot("v")],
            required_slots=["k", "v"],
            examples=["set x 4", "set y 8"],
        )
    )
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_SET_ASSIGN",
            intent_id=INTENT_SET_V91,
            pattern=[slot("k"), lit("="), slot("v")],
            required_slots=["k", "v"],
            examples=["x = 4", "y=8"],
        )
    )
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_SET_DEFINE_AS",
            intent_id=INTENT_SET_V91,
            pattern=[lit("define"), slot("k"), lit("as"), slot("v")],
            required_slots=["k", "v"],
            examples=["define x as 4", "define y as 8"],
        )
    )
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_SET_DEFINA_COMO",
            intent_id=INTENT_SET_V91,
            pattern=[lit("defina"), slot("k"), lit("como"), slot("v")],
            required_slots=["k", "v"],
            examples=["defina x como 4", "defina y como 8"],
        )
    )
    # Incomplete SET (missing value) -> ask clarify.
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_SET_SET_MISSING_V",
            intent_id=INTENT_SET_V91,
            pattern=[lit("set"), slot("k")],
            required_slots=["k", "v"],
            examples=["set x", "set y"],
        )
    )

    # GET
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_GET_GET",
            intent_id=INTENT_GET_V91,
            pattern=[lit("get"), slot("k")],
            required_slots=["k"],
            examples=["get x", "get y"],
        )
    )
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_GET_WHAT_IS",
            intent_id=INTENT_GET_V91,
            pattern=[lit("what"), lit("is"), slot("k")],
            required_slots=["k"],
            examples=["what is x", "what is y"],
        )
    )
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_GET_QUAL_E_O",
            intent_id=INTENT_GET_V91,
            pattern=[lit("qual"), lit("e"), lit("o"), slot("k")],
            required_slots=["k"],
            examples=["qual e o x", "qual e o y"],
        )
    )
    # Incomplete GET.
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_GET_GET_MISSING_K",
            intent_id=INTENT_GET_V91,
            pattern=[lit("get")],
            required_slots=["k"],
            examples=["get", "get ?"],
        )
    )

    # ADD
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_ADD_ADD_AND",
            intent_id=INTENT_ADD_V91,
            pattern=[lit("add"), slot("a"), lit("and"), slot("b")],
            required_slots=["a", "b"],
            examples=["add x and 10", "add 4 and 8"],
        )
    )
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_ADD_SUM_PLUS",
            intent_id=INTENT_ADD_V91,
            pattern=[lit("sum"), slot("a"), lit("+"), slot("b")],
            required_slots=["a", "b"],
            examples=["sum x + 10", "sum 4 + 8"],
        )
    )
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_ADD_SOME_E",
            intent_id=INTENT_ADD_V91,
            pattern=[lit("some"), slot("a"), lit("e"), slot("b")],
            required_slots=["a", "b"],
            examples=["some x e 10", "some 4 e 8"],
        )
    )
    # Incomplete ADD.
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_ADD_ADD_MISSING_B",
            intent_id=INTENT_ADD_V91,
            pattern=[lit("add"), slot("a")],
            required_slots=["a", "b"],
            examples=["add x", "add 4"],
        )
    )

    # SUMMARY
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_SUMMARY_SUMMARY",
            intent_id=INTENT_SUMMARY_V91,
            pattern=[lit("summary")],
            required_slots=[],
            examples=["summary", "summary please"],
        )
    )
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_SUMMARY_RESUMO",
            intent_id=INTENT_SUMMARY_V91,
            pattern=[lit("resumo")],
            required_slots=[],
            examples=["resumo", "resumo por favor"],
        )
    )
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_SUMMARY_SHOW_VARIABLES",
            intent_id=INTENT_SUMMARY_V91,
            pattern=[lit("show"), lit("variables")],
            required_slots=[],
            examples=["show variables", "show vars"],
        )
    )

    # END
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_END_END",
            intent_id=INTENT_END_V91,
            pattern=[lit("end")],
            required_slots=[],
            examples=["end", "end now"],
        )
    )
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_END_QUIT",
            intent_id=INTENT_END_V91,
            pattern=[lit("quit")],
            required_slots=[],
            examples=["quit", "quit now"],
        )
    )
    rules.append(
        make_intent_rule_v91(
            rule_id="INTENT_RULE_V91_END_FIM",
            intent_id=INTENT_END_V91,
            pattern=[lit("fim")],
            required_slots=[],
            examples=["fim", "fim."],
        )
    )

    rules.sort(key=lambda r: str(r.rule_id))
    return rules


def grammar_hash_v91(rules: Sequence[IntentRuleV91]) -> str:
    rows = [r.to_dict() for r in rules]
    rows.sort(key=lambda d: str(d.get("rule_id") or ""))
    return sha256_hex(canonical_json_dumps(rows).encode("utf-8"))


def make_intent_rule_act_v91(*, rule: IntentRuleV91, created_step: int = 0) -> Act:
    return Act(
        id=str(rule.rule_id),
        version=1,
        created_at=deterministic_iso(step=int(created_step)),
        kind="intent_rule_v91",  # type: ignore[assignment]
        match={},
        program=[],
        evidence={"intent_rule_v91": rule.to_dict()},
        cost={},
        deps=[],
        active=True,
    )


def default_intent_rule_acts_v91(*, created_step: int = 0) -> List[Act]:
    return [make_intent_rule_act_v91(rule=r, created_step=int(created_step)) for r in default_intent_rules_v91()]


def _match_pattern(pattern: Sequence[Dict[str, str]], tokens: Sequence[str]) -> Tuple[bool, Dict[str, str]]:
    if len(pattern) != len(tokens):
        return False, {}
    slots: Dict[str, str] = {}
    for p, t in zip(pattern, tokens):
        pt = str(p.get("t") or "")
        if pt == "lit":
            if str(p.get("v") or "") != str(t):
                return False, {}
        elif pt == "slot":
            name = str(p.get("n") or "")
            if not name:
                return False, {}
            slots[name] = str(t)
        else:
            return False, {}
    return True, slots


def parse_intent_v91(*, user_text: str, rules: Sequence[IntentRuleV91]) -> Dict[str, Any]:
    tokens = tokenize_user_text_v91(user_text)
    matches: List[Dict[str, Any]] = []
    for r in rules:
        ok, slots = _match_pattern(r.pattern, tokens)
        if not ok:
            continue
        missing = sorted({str(x) for x in r.required_slots if str(x) and str(x) not in slots})
        lit_count = sum(1 for p in r.pattern if str(p.get("t") or "") == "lit")
        match_len = int(len(r.pattern))
        matches.append(
            {
                "rule": r,
                "slots": dict(slots),
                "missing_slots": list(missing),
                "lit_count": int(lit_count),
                "match_len": int(match_len),
            }
        )

    # Deterministic best-match selection with explicit ambiguity detection.
    if not matches:
        sem = {
            "schema_version": 1,
            "intent_id": INTENT_UNKNOWN_V91,
            "slots": {},
            "missing_slots": [],
            "matched_rule_id": "",
            "parse_ok": False,
            "reason": "no_match",
            "tokens": list(tokens),
        }
        sig = _stable_hash_obj(sem)
        return dict(sem, parse_sig=str(sig))

    matches.sort(key=lambda m: (-int(m["lit_count"]), -int(m["match_len"]), str(m["rule"].rule_id)))
    best = matches[0]
    best_key = (int(best["lit_count"]), int(best["match_len"]))
    tied = [m for m in matches if (int(m["lit_count"]), int(m["match_len"])) == best_key]

    ambiguous_rule_ids = sorted({str(m["rule"].rule_id) for m in tied})
    ambiguous = len(ambiguous_rule_ids) > 1
    rbest: IntentRuleV91 = best["rule"]
    slots_best = dict(best["slots"])
    missing_best = list(best["missing_slots"])
    reason = "ok"
    parse_ok = (not ambiguous) and (not missing_best)
    if ambiguous:
        reason = "ambiguous"
        parse_ok = False
    elif missing_best:
        reason = "missing_slots"
        parse_ok = False

    sem = {
        "schema_version": 1,
        "intent_id": str(rbest.intent_id) if not ambiguous else INTENT_UNKNOWN_V91,
        "slots": dict(slots_best),
        "missing_slots": list(missing_best),
        "matched_rule_id": str(rbest.rule_id) if not ambiguous else "",
        "parse_ok": bool(parse_ok),
        "reason": str(reason),
        "tokens": list(tokens),
        "ambiguous_rule_ids": list(ambiguous_rule_ids) if ambiguous else [],
        "ambiguous_intents": [
            {"rule_id": str(m["rule"].rule_id), "intent_id": str(m["rule"].intent_id)}
            for m in sorted(tied, key=lambda mm: str(mm["rule"].rule_id))
        ]
        if ambiguous
        else [],
    }
    sig = _stable_hash_obj(sem)
    return dict(sem, parse_sig=str(sig))


def intent_grammar_snapshot_v91(rules: Sequence[IntentRuleV91]) -> Dict[str, Any]:
    rows = [r.to_dict() for r in rules]
    rows.sort(key=lambda d: str(d.get("rule_id") or ""))
    ghash = sha256_hex(canonical_json_dumps(rows).encode("utf-8"))
    return {"schema_version": 1, "grammar_hash": str(ghash), "rules": list(rows)}

