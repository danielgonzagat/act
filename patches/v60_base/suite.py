from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .metrics import distinct_n, loop_rate, repeat_ngram_rate, tokenize_text


CHAT_DIALOGUES_20X3: Tuple[Tuple[str, str, str], ...] = (
    ("Oi, quem é você?", "Explique em uma frase.", "Agora em inglês."),
    ("Hello, who are you?", "Explain in one sentence.", "Now in Portuguese."),
    ("Liste 3 regras do sistema.", "Por que você não usa gradiente?", "O que é um ato?"),
    ("What is MDL?", "How does MDL relate to compression?", "Give a tiny example."),
    ("Crie um mini diálogo sobre determinismo.", "Adicione mais um turno.", "Finalize."),
    ("Continue: the quick brown fox", "Agora em português.", "Agora com pontuação!"),
    ("Continue: um dois três", "Agora em inglês.", "Agora com números 123 e 42."),
    ("Defina 'estado explícito'.", "Por que é versionável?", "Dê um exemplo."),
    (
        "Explique a diferença entre runtime e learning aqui.",
        "Dê um exemplo de patch.",
        "E um exemplo de evidência.",
    ),
    ("Faça uma lista numerada com 5 itens.", "Agora remova o item 3.", "Agora ordene ao contrário."),
    ("Escreva uma frase curta e amigável.", "Agora mais formal.", "Agora mais técnica."),
    ("O que acontece se a repetição explode?", "Como você previne loops?", "Que métrica detecta isso?"),
    ("Me dê um exemplo de ato predictor.", "Me dê um exemplo de rewrite_rule.", "Me dê um exemplo de selector."),
    ("Explique n-grams em 1 linha.", "Agora em 2 linhas.", "Agora com uma analogia."),
    ("Pergunta: capital de Portugal?", "Pergunta: capital da França?", "Pergunta: capital do Brasil?"),
    ("Resuma: atos -> candidatos -> seleção -> atualização.", "Agora mais curto.", "Agora em inglês."),
    ("Escreva um micro-poema sobre atos.", "Sem repetir palavras.", "Agora com rima."),
    ("Diga algo sobre 'compressão'.", "Diga algo sobre 'fluência'.", "Conecte os dois."),
    ("Escreva uma frase com pontuação: ,.;:?!", "Agora sem pontuação.", "Agora com emoji textual :)"),
    (
        "Finalize com uma autodescrição concisa.",
        "Inclua uma restrição que você segue.",
        "Inclua uma segunda restrição.",
    ),
)


_WS_RE = re.compile(r"\s+", flags=re.UNICODE)
_LAST_USER_RE = re.compile(r"(?:^|\n)User:\s*(.*?)\nSystem:", flags=re.UNICODE | re.DOTALL)
_ALL_USERS_RE = re.compile(r"(?:^|\n)User:\s*([^\n]*)\nSystem:", flags=re.UNICODE)


# Utility / skill suite (deterministic validators, no world knowledge).
# Each task is validated on a single chosen turn (default: last turn).
SKILL_DIALOGUES_V0: Tuple[Dict[str, Any], ...] = (
    # Instruction following (exact output).
    {
        "task_id": "instr_ok",
        "dialogue": ("Responda exatamente: OK",),
        "validator_id": "exact_match",
        "expected_spec": {"text": "OK", "collapse_ws": True, "case_sensitive": True},
        "tags": ["instruction"],
    },
    {
        "task_id": "instr_exact_code_1",
        "dialogue": ("Retorne exatamente a string: PATO-123",),
        "validator_id": "exact_match",
        "expected_spec": {"text": "PATO-123", "collapse_ws": True, "case_sensitive": True},
        "tags": ["instruction"],
    },
    {
        "task_id": "instr_exact_code_2",
        "dialogue": ("Retorne exatamente a string: X_Y-Z",),
        "validator_id": "exact_match",
        "expected_spec": {"text": "X_Y-Z", "collapse_ws": True, "case_sensitive": True},
        "tags": ["instruction"],
    },
    # Format / JSON (strict, parseable, minimal schema).
    {
        "task_id": "json_ab_1",
        "dialogue": (
            'Responda com APENAS um JSON válido (sem texto extra) com chaves "a" e "b". '
            '"a" deve ser a string "x" e "b" deve ser o número 7.',
        ),
        "validator_id": "json_parse_keys",
        "expected_spec": {
            "required_keys": ["a", "b"],
            "types": {"a": "str", "b": "int"},
            "expected_values": {"a": "x", "b": 7},
        },
        "tags": ["format", "json", "instruction"],
    },
    {
        "task_id": "json_bool_1",
        "dialogue": (
            'Responda com APENAS um JSON válido (sem texto extra) com chaves "ok" e "n". '
            '"ok" deve ser true e "n" deve ser 3.',
        ),
        "validator_id": "json_parse_keys",
        "expected_spec": {
            "required_keys": ["ok", "n"],
            "types": {"ok": "bool", "n": "int"},
            "expected_values": {"ok": True, "n": 3},
        },
        "tags": ["format", "json", "instruction"],
    },
    # Arithmetic (no world knowledge; strict integer output only).
    {
        "task_id": "math_add_17_25",
        "dialogue": ("17 + 25 = ? (responda APENAS com o número inteiro)",),
        "validator_id": "int_exact",
        "expected_spec": {"value": 42, "strict": True},
        "tags": ["math", "format", "instruction"],
    },
    {
        "task_id": "math_mul_9_7",
        "dialogue": ("9 * 7 = ? (responda APENAS com o número inteiro)",),
        "validator_id": "int_exact",
        "expected_spec": {"value": 63, "strict": True},
        "tags": ["math", "format", "instruction"],
    },
    {
        "task_id": "math_sub_100_58",
        "dialogue": ("100 - 58 = ? (responda APENAS com o número inteiro)",),
        "validator_id": "int_exact",
        "expected_spec": {"value": 42, "strict": True},
        "tags": ["math", "format", "instruction"],
    },
    # Context factual (fact provided in-context; deterministic target).
    {
        "task_id": "ctx_password",
        "dialogue": (
            "Contexto: a senha é PINEAPPLE. Apenas memorize.",
            "Qual é a senha? Responda exatamente com a senha.",
        ),
        "validate_turn": 1,
        "validator_id": "exact_match",
        "expected_spec": {"text": "PINEAPPLE", "collapse_ws": True, "case_sensitive": True},
        "tags": ["factual", "state", "instruction"],
    },
    {
        "task_id": "ctx_year",
        "dialogue": (
            "Contexto: Joana nasceu em 1980. Apenas memorize.",
            "Em que ano Joana nasceu? Responda APENAS com o número inteiro.",
        ),
        "validate_turn": 1,
        "validator_id": "int_exact",
        "expected_spec": {"value": 1980, "strict": True},
        "tags": ["factual", "state", "math", "instruction"],
    },
    # State multi-turn (must carry state across turns).
    {
        "task_id": "state_code_alfa9",
        "dialogue": (
            "Memorize a palavra-código: ALFA9.",
            "Repita a palavra-código exatamente.",
        ),
        "validate_turn": 1,
        "validator_id": "exact_match",
        "expected_spec": {"text": "ALFA9", "collapse_ws": True, "case_sensitive": True},
        "tags": ["state", "instruction"],
    },
    {
        "task_id": "state_code_beta7",
        "dialogue": (
            "Memorize a palavra-código: BETA7.",
            "Repita a palavra-código exatamente.",
        ),
        "validate_turn": 1,
        "validator_id": "exact_match",
        "expected_spec": {"text": "BETA7", "collapse_ws": True, "case_sensitive": True},
        "tags": ["state", "instruction"],
    },
    # Contains token (allows extra text but must include exact token).
    {
        "task_id": "contains_token_foobar",
        "dialogue": (
            "A palavra-chave é FOOBAR. Responda em uma frase curta, mas inclua FOOBAR.",
        ),
        "validator_id": "contains_exact_token",
        "expected_spec": {"token": "FOOBAR", "case_sensitive": True},
        "tags": ["state", "instruction"],
    },
)

SKILL_DIALOGUES_V1_PARAPHRASE: Tuple[Dict[str, Any], ...] = (
    # Instruction following (paraphrases; exact output).
    {
        "task_id": "p_instr_ok",
        "dialogue": ("Responda somente com: OK",),
        "validator_id": "exact_match",
        "expected_spec": {"text": "OK", "collapse_ws": True, "case_sensitive": True},
        "tags": ["instruction"],
    },
    {
        "task_id": "p_instr_exact_code_1",
        "dialogue": ("Devolva exatamente: PATO-123",),
        "validator_id": "exact_match",
        "expected_spec": {"text": "PATO-123", "collapse_ws": True, "case_sensitive": True},
        "tags": ["instruction"],
    },
    {
        "task_id": "p_instr_exact_code_2",
        "dialogue": ("Responda apenas com: X_Y-Z",),
        "validator_id": "exact_match",
        "expected_spec": {"text": "X_Y-Z", "collapse_ws": True, "case_sensitive": True},
        "tags": ["instruction"],
    },
    # Format / JSON (paraphrases; strict, parseable, minimal schema).
    {
        "task_id": "p_json_ab_1",
        "dialogue": (
            'Devolva SOMENTE um JSON válido (sem texto extra) com chaves "a" e "b". '
            '"a" deve ser a string "x" e "b" deve ser o número 7.',
        ),
        "validator_id": "json_parse_keys",
        "expected_spec": {
            "required_keys": ["a", "b"],
            "types": {"a": "str", "b": "int"},
            "expected_values": {"a": "x", "b": 7},
        },
        "tags": ["format", "json", "instruction"],
    },
    {
        "task_id": "p_json_bool_1",
        "dialogue": (
            'Retorne SOMENTE um JSON válido (sem texto extra) com chaves "ok" e "n". '
            '"ok" deve ser true e "n" deve ser 3.',
        ),
        "validator_id": "json_parse_keys",
        "expected_spec": {
            "required_keys": ["ok", "n"],
            "types": {"ok": "bool", "n": "int"},
            "expected_values": {"ok": True, "n": 3},
        },
        "tags": ["format", "json", "instruction"],
    },
    # Arithmetic (paraphrases; strict integer output only).
    {
        "task_id": "p_math_add_12_30",
        "dialogue": ("Quanto é 12 + 30? Responda só o número.",),
        "validator_id": "int_exact",
        "expected_spec": {"value": 42, "strict": True},
        "tags": ["math", "format", "instruction"],
    },
    {
        "task_id": "p_math_mul_9_7",
        "dialogue": ("Calcule 9*7. Responda somente o número.",),
        "validator_id": "int_exact",
        "expected_spec": {"value": 63, "strict": True},
        "tags": ["math", "format", "instruction"],
    },
    {
        "task_id": "p_math_sub_100_58",
        "dialogue": ("Qual é o resultado de 100-58? Escreva apenas o número.",),
        "validator_id": "int_exact",
        "expected_spec": {"value": 42, "strict": True},
        "tags": ["math", "format", "instruction"],
    },
    # Context factual (paraphrases; fact provided in-context; deterministic target).
    {
        "task_id": "p_ctx_password",
        "dialogue": (
            "Contexto: a senha é PINEAPPLE. Memorize.",
            "Diga a senha. Responda somente com a senha.",
        ),
        "validate_turn": 1,
        "validator_id": "exact_match",
        "expected_spec": {"text": "PINEAPPLE", "collapse_ws": True, "case_sensitive": True},
        "tags": ["factual", "state", "instruction"],
    },
    {
        "task_id": "p_ctx_year",
        "dialogue": (
            "Contexto: Joana nasceu em 1980. Memorize.",
            "Qual é o ano de nascimento de Joana? Responda só o número.",
        ),
        "validate_turn": 1,
        "validator_id": "int_exact",
        "expected_spec": {"value": 1980, "strict": True},
        "tags": ["factual", "state", "math", "instruction"],
    },
    # State multi-turn (paraphrases; must carry state across turns).
    {
        "task_id": "p_state_code_alfa9",
        "dialogue": (
            "Guarde a palavra-código: ALFA9.",
            "Agora repita a palavra-código exatamente.",
        ),
        "validate_turn": 1,
        "validator_id": "exact_match",
        "expected_spec": {"text": "ALFA9", "collapse_ws": True, "case_sensitive": True},
        "tags": ["state", "instruction"],
    },
    {
        "task_id": "p_state_code_beta7",
        "dialogue": (
            "Guarde a palavra-código: BETA7.",
            "Repita o código exatamente.",
        ),
        "validate_turn": 1,
        "validator_id": "exact_match",
        "expected_spec": {"text": "BETA7", "collapse_ws": True, "case_sensitive": True},
        "tags": ["state", "instruction"],
    },
    # Contains token (paraphrase; allows extra text but must include exact token).
    {
        "task_id": "p_contains_token_foobar",
        "dialogue": ("Escreva uma frase curta e inclua a palavra FOOBAR.",),
        "validator_id": "contains_exact_token",
        "expected_spec": {"token": "FOOBAR", "case_sensitive": True},
        "tags": ["state", "instruction"],
    },
)


def _normalize_output(text: str, *, collapse_ws: bool) -> str:
    s = str(text or "").strip()
    if collapse_ws:
        s = _WS_RE.sub(" ", s)
    return s


def _short_snip(text: str, *, limit: int = 160) -> str:
    s = _normalize_output(text, collapse_ws=True)
    if len(s) <= int(limit):
        return s
    return s[: int(limit) - 1] + "…"


def _validate_exact_match(output: str, spec: Dict[str, Any]) -> Tuple[bool, str]:
    expected = str(spec.get("text", ""))
    collapse_ws = bool(spec.get("collapse_ws", True))
    case_sensitive = bool(spec.get("case_sensitive", True))
    out_n = _normalize_output(output, collapse_ws=collapse_ws)
    exp_n = _normalize_output(expected, collapse_ws=collapse_ws)
    if not case_sensitive:
        out_n = out_n.lower()
        exp_n = exp_n.lower()
    ok = bool(out_n == exp_n)
    return ok, "" if ok else f"exact_mismatch got={_short_snip(out_n)} expected={_short_snip(exp_n)}"


_INT_FULL_RE = re.compile(r"[-+]?\d+", flags=re.UNICODE)


def _validate_int_exact(output: str, spec: Dict[str, Any]) -> Tuple[bool, str]:
    expected = int(spec.get("value", 0) or 0)
    strict = bool(spec.get("strict", True))
    s = _normalize_output(output, collapse_ws=True)
    if strict:
        if re.fullmatch(_INT_FULL_RE, s) is None:
            return False, "int_not_strict"
        got = int(s)
    else:
        m = _INT_FULL_RE.search(s)
        if not m:
            return False, "int_missing"
        got = int(m.group(0))
    ok = bool(got == expected)
    return ok, "" if ok else f"int_mismatch got={got} expected={expected}"


def _validate_json_parse_keys(output: str, spec: Dict[str, Any]) -> Tuple[bool, str]:
    s = str(output or "").strip()
    required_keys = spec.get("required_keys") or []
    types = spec.get("types") or {}
    expected_values = spec.get("expected_values") or {}
    if not isinstance(required_keys, list) or not isinstance(types, dict) or not isinstance(expected_values, dict):
        return False, "invalid_spec"
    try:
        obj = json.loads(s)
    except Exception:
        return False, "json_parse_error"
    if not isinstance(obj, dict):
        return False, "json_not_object"
    for k in required_keys:
        if not isinstance(k, str):
            continue
        if k not in obj:
            return False, f"json_missing_key:{k}"
    for k, tname in types.items():
        if not isinstance(k, str) or not isinstance(tname, str):
            continue
        v = obj.get(k)
        if tname == "str":
            if not isinstance(v, str):
                return False, f"json_type_mismatch:{k}:str"
        elif tname == "int":
            if not isinstance(v, int) or isinstance(v, bool):
                return False, f"json_type_mismatch:{k}:int"
        elif tname == "bool":
            if not isinstance(v, bool):
                return False, f"json_type_mismatch:{k}:bool"
        else:
            return False, f"json_unknown_type:{k}:{tname}"
    for k, exp in expected_values.items():
        if not isinstance(k, str):
            continue
        if obj.get(k) != exp:
            return False, f"json_value_mismatch:{k}"
    return True, ""


def _validate_regex_fullmatch(output: str, spec: Dict[str, Any]) -> Tuple[bool, str]:
    pattern = str(spec.get("pattern") or "")
    flags = int(spec.get("flags", 0) or 0)
    s = str(output or "").strip()
    try:
        ok = re.fullmatch(pattern, s, flags=flags) is not None
    except re.error:
        return False, "regex_error"
    return ok, "" if ok else "regex_no_match"


def _validate_contains_exact_token(output: str, spec: Dict[str, Any]) -> Tuple[bool, str]:
    token = str(spec.get("token") or "")
    case_sensitive = bool(spec.get("case_sensitive", True))
    s = str(output or "")
    if not case_sensitive:
        s = s.lower()
        token = token.lower()
    ok = bool(token and (token in s))
    return ok, "" if ok else "token_missing"


_VALIDATORS: Dict[str, Callable[[str, Dict[str, Any]], Tuple[bool, str]]] = {
    "exact_match": _validate_exact_match,
    "int_exact": _validate_int_exact,
    "json_parse_keys": _validate_json_parse_keys,
    "regex_fullmatch": _validate_regex_fullmatch,
    "contains_exact_token": _validate_contains_exact_token,
}


def _plan_trace_for_task(task: Dict[str, Any], *, turn_idx: int) -> Dict[str, Any]:
    task_id = str(task.get("task_id") or "")
    dialogue = task.get("dialogue") or ()
    validate_turn = int(task.get("validate_turn", max(0, len(dialogue) - 1)) or 0)
    validator_id = str(task.get("validator_id") or "") if int(turn_idx) == int(validate_turn) else ""
    expected_spec = task.get("expected_spec") or {}

    expected_format = ""
    constraints: List[str] = []
    if validator_id == "exact_match":
        expected_format = "exact"
        constraints = ["exact_match", "no_extra_tokens"]
    elif validator_id == "int_exact":
        expected_format = "int"
        strict = bool(expected_spec.get("strict", True)) if isinstance(expected_spec, dict) else True
        constraints = ["int_exact", "digits_only" if strict else "int_extract"]
    elif validator_id == "json_parse_keys":
        expected_format = "json"
        keys = []
        if isinstance(expected_spec, dict):
            rk = expected_spec.get("required_keys") or []
            if isinstance(rk, list):
                keys = [str(k) for k in rk if isinstance(k, str)]
        if keys:
            constraints = ["json", "keys:" + ",".join(keys)]
        else:
            constraints = ["json"]
    elif validator_id == "regex_fullmatch":
        expected_format = "regex"
        constraints = ["regex_fullmatch"]
    elif validator_id == "contains_exact_token":
        expected_format = "contains"
        tok = ""
        if isinstance(expected_spec, dict):
            tok = str(expected_spec.get("token") or "")
        constraints = ["contains_exact_token"] + ([f"token:{tok}"] if tok else [])

    return {
        "task_id": task_id,
        "validator_id": validator_id,
        "expected_format": expected_format,
        "constraints": constraints,
    }


def reply_signature(text: str) -> str:
    s = text.strip()
    s = _WS_RE.sub(" ", s)
    s = s.lower()
    return s[:256]


def build_chat_prompt(history: Sequence[Dict[str, str]]) -> str:
    parts: List[str] = []
    for h in history:
        parts.append(f"User: {h['user']}\nSystem: {h.get('system','')}")
    return "\n".join(parts).rstrip() + "\n"


def _non_ws(tokens: Sequence[str]) -> List[str]:
    return [t for t in tokens if t and (not t.isspace()) and t != "<BOS>"]


def non_ws_tokens(tokens: Sequence[str]) -> List[str]:
    return _non_ws(tokens)


def last_user_text_from_prompt(prompt: str) -> str:
    matches = list(_LAST_USER_RE.finditer(prompt))
    if not matches:
        return ""
    return matches[-1].group(1)


def user_signature(text: str, *, k: int = 2) -> str:
    toks = _non_ws(tokenize_text(text))
    toks = toks[: max(0, int(k))]
    return " ".join(t.lower() for t in toks)


def user_signature_from_prompt(prompt: str, *, k: int = 2) -> str:
    return user_signature(last_user_text_from_prompt(prompt), k=k)


def user_signatures_from_prompt(prompt: str, *, k: int = 2) -> List[str]:
    out: List[str] = []
    for m in _ALL_USERS_RE.finditer(prompt):
        sig = user_signature(m.group(1), k=k)
        if sig:
            out.append(sig)
    return out


def prefix_k_signature(tokens: Sequence[str], *, k: int = 8) -> str:
    toks = _non_ws(tokens)[: max(0, int(k))]
    return " ".join(t.lower() for t in toks)


def prefix_k_dup_rate(prefix_sigs: Sequence[str]) -> float:
    total = len(prefix_sigs)
    if total <= 0:
        return 0.0
    uniq = len(set(prefix_sigs))
    return 1.0 - (uniq / total)


def template_ngram_dup_rate(
    reply_tokens: Sequence[Sequence[str]],
    *,
    n: int = 6,
    prefix_window: int = 32,
) -> float:
    n = int(n)
    prefix_window = int(prefix_window)
    if n <= 0 or prefix_window <= 0:
        return 0.0
    seen = set()
    repeats = 0
    total = 0
    for toks in reply_tokens:
        win = [t.lower() for t in _non_ws(toks)[:prefix_window]]
        if len(win) < n:
            continue
        for i in range(len(win) - n + 1):
            ng = tuple(win[i : i + n])
            total += 1
            if ng in seen:
                repeats += 1
            else:
                seen.add(ng)
    return (repeats / total) if total > 0 else 0.0


def cross_turn_signature_repeat_rate(dialogue_sigs: Sequence[Sequence[str]]) -> float:
    repeats = 0
    total = 0
    for sigs in dialogue_sigs:
        seen = set()
        for i, sig in enumerate(sigs):
            if i == 0:
                seen.add(sig)
                continue
            total += 1
            if sig in seen:
                repeats += 1
            seen.add(sig)
    return (repeats / total) if total > 0 else 0.0


def cross_turn_mode_repeat_rate(dialogue_modes: Sequence[Sequence[str]]) -> float:
    repeats = 0
    total = 0
    for modes in dialogue_modes:
        seen = set()
        for i, m in enumerate(modes):
            if i == 0:
                seen.add(m)
                continue
            total += 1
            if m in seen:
                repeats += 1
            seen.add(m)
    return (repeats / total) if total > 0 else 0.0


def suite_metrics_from_generations(
    *,
    all_gen_tokens: Sequence[str],
    reply_gen_tokens: Sequence[Sequence[str]],
    dialogue_reply_sigs: Sequence[Sequence[str]],
    dialogue_reply_modes: Sequence[Sequence[str]] = (),
    reply_sigs: Sequence[str],
    prefix_sigs: Sequence[str],
    reply_modes: Sequence[str] = (),
    reply_mode_sources: Sequence[str] = (),
    reply_policy_actions: Sequence[str] = (),
    reply_policy_coverages: Sequence[float] = (),
    prefix_k: int,
    template_ngram_n: int,
    template_prefix_window: int,
) -> Dict[str, Any]:
    total = max(1, len(all_gen_tokens))
    ws = sum(1 for t in all_gen_tokens if t.isspace())
    whitespace_ratio = ws / total

    non_ws_all = [t for t in all_gen_tokens if not t.isspace()]
    if len(non_ws_all) < 3:
        repeat3_global = 1.0
        loop_rate_global = 1.0
        distinct2_global = 0.0
    else:
        repeat3_global = repeat_ngram_rate(all_gen_tokens, 3, ignore_space=True)
        loop_rate_global = loop_rate(all_gen_tokens, n=3, window=128, ignore_space=True)
        distinct2_global = distinct_n(all_gen_tokens, 2, ignore_space=True)

    c = Counter(reply_sigs)
    most_common = max(c.values()) if c else 0
    unique_reply_rate = (len(c) / max(1, len(reply_sigs))) if reply_sigs else 0.0
    duplicate_reply_rate = 1.0 - unique_reply_rate
    most_common_reply_frac = (most_common / max(1, len(reply_sigs))) if reply_sigs else 0.0

    # Cross-turn / template metrics.
    prefix_k_dup = prefix_k_dup_rate(prefix_sigs)
    template_dup = template_ngram_dup_rate(
        reply_gen_tokens, n=template_ngram_n, prefix_window=template_prefix_window
    )
    cross_turn_rep = cross_turn_signature_repeat_rate(dialogue_reply_sigs)
    cross_turn_mode_rep = (
        cross_turn_mode_repeat_rate(dialogue_reply_modes) if dialogue_reply_modes else 0.0
    )

    reply_lengths = [len(t) for t in reply_gen_tokens]
    avg_len_tokens = sum(reply_lengths) / max(1, len(reply_lengths))

    mode_counts = Counter(str(m) for m in reply_modes) if reply_modes else Counter()
    mode_distribution = {
        k: (v / max(1, len(reply_modes))) for k, v in sorted(mode_counts.items())
    }

    mode_source_counts = Counter(str(s) for s in reply_mode_sources) if reply_mode_sources else Counter()
    policy_hits = int(mode_source_counts.get("policy", 0))
    policy_hit_rate = (policy_hits / max(1, len(reply_mode_sources))) if reply_mode_sources else 0.0

    policy_action_counts = (
        Counter(str(a) for a in reply_policy_actions) if reply_policy_actions else Counter()
    )
    explore = int(policy_action_counts.get("explore", 0))
    exploit = int(policy_action_counts.get("exploit", 0))
    policy_turns = int(explore + exploit)
    policy_explore_rate = explore / max(1, policy_turns) if policy_turns > 0 else 0.0
    policy_exploit_rate = exploit / max(1, policy_turns) if policy_turns > 0 else 0.0

    covs = [float(x) for x in reply_policy_coverages] if reply_policy_coverages else []
    policy_coverage_mean = (sum(covs) / len(covs)) if covs else 0.0

    return {
        "num_dialogues": float(len(dialogue_reply_sigs)),
        "num_replies": float(len(reply_sigs)),
        "avg_len_tokens": float(avg_len_tokens),
        "whitespace_ratio": float(whitespace_ratio),
        "repeat3_global": float(repeat3_global),
        "loop_rate_global": float(loop_rate_global),
        "distinct2_global": float(distinct2_global),
        "unique_reply_rate": float(unique_reply_rate),
        "duplicate_reply_rate": float(duplicate_reply_rate),
        "most_common_reply_frac": float(most_common_reply_frac),
        "prefix_k": float(prefix_k),
        "prefix_k_dup_rate": float(prefix_k_dup),
        "template_ngram_n": float(template_ngram_n),
        "template_prefix_window": float(template_prefix_window),
        "template_ngram_dup_rate": float(template_dup),
        "cross_turn_signature_repeat_rate": float(cross_turn_rep),
        "cross_turn_mode_repeat_rate": float(cross_turn_mode_rep),
        "mode_counts": dict(mode_counts),
        "mode_distribution": mode_distribution,
        "mode_source_counts": dict(mode_source_counts),
        "policy_hit_rate": float(policy_hit_rate),
        "policy_action_counts": dict(policy_action_counts),
        "policy_explore_rate": float(policy_explore_rate),
        "policy_exploit_rate": float(policy_exploit_rate),
        "policy_coverage_mean": float(policy_coverage_mean),
    }


def run_chat_suite(
    engine,
    *,
    dialogues: Sequence[Sequence[str]] = CHAT_DIALOGUES_20X3,
    max_new_tokens: int = 200,
    prefix_k: int = 8,
    template_ngram_n: int = 6,
    template_prefix_window: int = 32,
    csv: Any = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    transcripts: List[Dict[str, Any]] = []
    all_gen_tokens: List[str] = []
    reply_gen_tokens: List[List[str]] = []
    reply_sigs: List[str] = []
    prefix_sigs: List[str] = []
    dialogue_sigs: List[List[str]] = []
    reply_modes: List[str] = []
    dialogue_modes: List[List[str]] = []
    reply_mode_sources: List[str] = []
    dialogue_mode_sources: List[List[str]] = []
    reply_policy_actions: List[str] = []
    reply_policy_coverages: List[float] = []

    trace_tokens_total = 0
    trace_candidates_sum = 0
    trace_pred_matched_sum = 0
    trace_pred_emitted_sum = 0
    trace_pred_iterated_sum = 0
    trace_act_evals_sum = 0
    trace_scan_evals_sum = 0
    trace_active_set_size: int = 0
    trace_rewrite_rules_total: int = 0
    trace_selector_present: int = 0
    csv_step = 0

    for i, turns in enumerate(dialogues):
        history: List[Dict[str, str]] = []
        dial_sigs: List[str] = []
        dial_modes: List[str] = []
        dial_mode_sources: List[str] = []
        for j, user_msg in enumerate(turns):
            history.append({"user": user_msg, "system": ""})
            prompt = build_chat_prompt(history)
            out = engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                mode="greedy",
                dialogue_id=int(i),
                turn=int(j),
            )
            resp = out["text"][len(prompt) :]
            history[-1]["system"] = resp
            history[-1]["mode"] = str(out.get("mode") or "default")
            history[-1]["mode_source"] = str(out.get("mode_source") or "router")
            history[-1]["mode_policy_action"] = str(out.get("mode_policy_action") or "")
            history[-1]["policy_coverage"] = float(out.get("policy_coverage") or 0.0)
            history[-1]["user_sig"] = str(out.get("user_sig") or "")
            history[-1]["trace"] = dict(out.get("trace") or {})
            if csv is not None:
                try:
                    csv.observe_turn(
                        step=int(csv_step),
                        context_signature=f"chat␟d={i}␟t={j}",
                        trace=history[-1]["trace"],
                        utility_passed=None,
                        suite_kind="chat",
                        meta={"dialogue_id": int(i), "turn": int(j)},
                    )
                except Exception:
                    pass
                csv_step += 1

            sig = reply_signature(resp)
            reply_sigs.append(sig)
            dial_sigs.append(sig)

            m = str(out.get("mode") or "default")
            reply_modes.append(m)
            dial_modes.append(m)

            ms = str(out.get("mode_source") or "router")
            reply_mode_sources.append(ms)
            dial_mode_sources.append(ms)

            reply_policy_actions.append(str(out.get("mode_policy_action") or ""))
            reply_policy_coverages.append(float(out.get("policy_coverage") or 0.0))

            gen_toks = list(out["gen_tokens"])
            reply_gen_tokens.append(gen_toks)
            all_gen_tokens.extend(gen_toks)
            prefix_sigs.append(prefix_k_signature(gen_toks, k=prefix_k))

            tr = out.get("trace") or {}
            cand_post = tr.get("candidates_post_rewrite") or []
            pred_matched = tr.get("predictor_matched") or []
            pred_emitted = tr.get("predictor_emitted") or []
            pred_iterated = tr.get("predictor_iterated") or []
            if isinstance(cand_post, list) and isinstance(pred_matched, list) and isinstance(pred_emitted, list):
                n_tok = min(len(cand_post), len(pred_matched), len(pred_emitted))
                if n_tok > 0:
                    trace_tokens_total += int(n_tok)
                    trace_candidates_sum += int(sum(int(x) for x in cand_post[:n_tok]))
                    trace_pred_matched_sum += int(sum(int(x) for x in pred_matched[:n_tok]))
                    trace_pred_emitted_sum += int(sum(int(x) for x in pred_emitted[:n_tok]))
                    if isinstance(pred_iterated, list):
                        n2 = min(int(n_tok), int(len(pred_iterated)))
                        if n2 > 0:
                            trace_pred_iterated_sum += int(
                                sum(int(x) for x in pred_iterated[:n2])
                            )

                    rr_total = int(tr.get("rewrite_rules_total", 0) or 0)
                    sel_present = 1 if tr.get("selector_id") else 0
                    trace_act_evals_sum += int(sum(int(x) for x in pred_matched[:n_tok]))
                    trace_act_evals_sum += int(n_tok * rr_total)
                    trace_act_evals_sum += int(n_tok * sel_present)
                    if isinstance(pred_iterated, list) and n2 > 0:
                        trace_scan_evals_sum += int(sum(int(x) for x in pred_iterated[:n2]))
                        trace_scan_evals_sum += int(n2 * rr_total)
                        trace_scan_evals_sum += int(n2 * sel_present)

                    if trace_active_set_size <= 0:
                        trace_active_set_size = int(tr.get("active_set_size", 0) or 0)
                        trace_rewrite_rules_total = int(rr_total)
                        trace_selector_present = int(sel_present)

        dialogue_sigs.append(dial_sigs)
        dialogue_modes.append(dial_modes)
        dialogue_mode_sources.append(dial_mode_sources)
        full_text = build_chat_prompt(history)
        transcripts.append({"prompt_id": i, "turns": history, "full_text": full_text})

    metrics = suite_metrics_from_generations(
        all_gen_tokens=all_gen_tokens,
        reply_gen_tokens=reply_gen_tokens,
        dialogue_reply_sigs=dialogue_sigs,
        dialogue_reply_modes=dialogue_modes,
        reply_sigs=reply_sigs,
        prefix_sigs=prefix_sigs,
        reply_modes=reply_modes,
        reply_mode_sources=reply_mode_sources,
        reply_policy_actions=reply_policy_actions,
        reply_policy_coverages=reply_policy_coverages,
        prefix_k=prefix_k,
        template_ngram_n=template_ngram_n,
        template_prefix_window=template_prefix_window,
    )
    if trace_tokens_total > 0:
        metrics["trace_tokens_total"] = int(trace_tokens_total)
        metrics["candidates_considered_per_token_mean"] = float(
            trace_candidates_sum / trace_tokens_total
        )
        metrics["predictor_matched_per_token_mean"] = float(
            trace_pred_matched_sum / trace_tokens_total
        )
        metrics["predictor_emitted_per_token_mean"] = float(
            trace_pred_emitted_sum / trace_tokens_total
        )
        metrics["acts_considered_per_token_mean"] = float(
            trace_act_evals_sum / trace_tokens_total
        )
        metrics["search_steps_per_turn_mean"] = float(
            trace_act_evals_sum / max(1, len(reply_sigs))
        )
        if trace_pred_iterated_sum > 0:
            metrics["predictor_iterated_per_token_mean"] = float(
                trace_pred_iterated_sum / trace_tokens_total
            )
        if trace_scan_evals_sum > 0:
            metrics["scan_acts_considered_per_token_mean"] = float(
                trace_scan_evals_sum / trace_tokens_total
            )
            metrics["scan_steps_per_turn_mean"] = float(
                trace_scan_evals_sum / max(1, len(reply_sigs))
            )
        metrics["trace_active_set_size"] = int(trace_active_set_size)
        metrics["trace_rewrite_rules_total"] = int(trace_rewrite_rules_total)
        metrics["trace_selector_present"] = int(trace_selector_present)
    return transcripts, metrics


def run_skill_suite(
    engine,
    *,
    tasks: Sequence[Dict[str, Any]] = SKILL_DIALOGUES_V0,
    max_new_tokens: int = 200,
    csv: Any = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    transcripts: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    total_tasks = 0
    pass_count = 0

    cat_total: Counter = Counter()
    cat_pass: Counter = Counter()

    plan_turns_total = 0
    plan_turns_missing = 0
    contract_used_turns = 0
    contract_used_by_kind: Counter = Counter()
    csv_step = 0

    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or f"task_{i}")
        turns = task.get("dialogue") or ()
        if not isinstance(turns, (list, tuple)) or not turns:
            continue
        validate_turn = int(task.get("validate_turn", max(0, len(turns) - 1)) or 0)
        validate_turn = max(0, min(validate_turn, len(turns) - 1))
        validator_id = str(task.get("validator_id") or "")
        expected_spec = task.get("expected_spec") or {}
        tags = task.get("tags") or []
        if not isinstance(tags, list):
            tags = []

        total_tasks += 1
        for cat in ("instruction", "json", "math", "state"):
            if cat in tags:
                cat_total[cat] += 1

        history: List[Dict[str, Any]] = []
        for j, user_msg in enumerate(turns):
            history.append({"user": str(user_msg), "system": ""})
            prompt = build_chat_prompt(history)
            out = engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                mode="greedy",
                dialogue_id=int(i),
                turn=int(j),
            )
            resp = out["text"][len(prompt) :]
            history[-1]["system"] = resp
            history[-1]["mode"] = str(out.get("mode") or "default")
            history[-1]["mode_source"] = str(out.get("mode_source") or "router")
            history[-1]["mode_policy_action"] = str(out.get("mode_policy_action") or "")
            history[-1]["policy_coverage"] = float(out.get("policy_coverage") or 0.0)
            history[-1]["user_sig"] = str(out.get("user_sig") or "")

            tr = dict(out.get("trace") or {})
            plan_trace = _plan_trace_for_task(task, turn_idx=int(j))
            contract_meta = tr.get("instruction_contract")
            if isinstance(contract_meta, dict):
                plan_trace["contract"] = contract_meta
                if bool(contract_meta.get("used")):
                    contract_used_turns += 1
                    k = str(contract_meta.get("kind") or "")
                    if k:
                        contract_used_by_kind[k] += 1
            tr["plan_trace"] = plan_trace
            history[-1]["trace"] = tr

            plan_turns_total += 1
            if "plan_trace" not in tr:
                plan_turns_missing += 1

        full_text = build_chat_prompt(history)
        transcripts.append(
            {"prompt_id": i, "task_id": task_id, "turns": history, "full_text": full_text}
        )

        # Validate a single, deterministic turn per task.
        turn_rec = history[validate_turn] if validate_turn < len(history) else history[-1]
        out_text = str(turn_rec.get("system", ""))
        fn = _VALIDATORS.get(validator_id)
        if fn is None or not isinstance(expected_spec, dict):
            ok = False
            reason = "unknown_validator_or_spec"
        else:
            ok, reason = fn(out_text, expected_spec)

        if ok:
            pass_count += 1
            for cat in ("instruction", "json", "math", "state"):
                if cat in tags:
                    cat_pass[cat] += 1
        else:
            if len(failures) < 5:
                try:
                    exp_short = _short_snip(
                        json.dumps(expected_spec, ensure_ascii=False, sort_keys=True)
                    )
                except Exception:
                    exp_short = _short_snip(str(expected_spec))
                failures.append(
                    {
                        "task_id": task_id,
                        "turn": int(validate_turn),
                        "validator_id": validator_id,
                        "expected_spec": exp_short,
                        "output_snippet": _short_snip(out_text),
                        "reason": str(reason or ""),
                    }
                )

        if csv is not None:
            try:
                tr = turn_rec.get("trace") or {}
                if isinstance(tr, dict):
                    csv.observe_turn(
                        step=int(csv_step),
                        context_signature=f"skill␟task={task_id}␟turn={validate_turn}",
                        trace=tr,
                        utility_passed=bool(ok),
                        suite_kind="skill",
                        meta={"task_id": str(task_id), "validate_turn": int(validate_turn), "validator_id": str(validator_id)},
                    )
                    csv_step += 1
            except Exception:
                pass

    def _rate(ok: int, total: int) -> float:
        return float(ok / total) if int(total) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "total_tasks": int(total_tasks),
        "pass_count": int(pass_count),
        "pass_rate": _rate(pass_count, total_tasks),
        "instruction_pass_rate": _rate(
            int(cat_pass.get("instruction", 0)), int(cat_total.get("instruction", 0))
        ),
        "json_pass_rate": _rate(int(cat_pass.get("json", 0)), int(cat_total.get("json", 0))),
        "math_pass_rate": _rate(int(cat_pass.get("math", 0)), int(cat_total.get("math", 0))),
        "state_pass_rate": _rate(int(cat_pass.get("state", 0)), int(cat_total.get("state", 0))),
        "failures": list(failures),
        "plan_trace_turns_total": int(plan_turns_total),
        "plan_trace_missing_turns": int(plan_turns_missing),
        "contract_total_turns": int(plan_turns_total),
        "contract_used_turns": int(contract_used_turns),
        "contract_used_rate": _rate(contract_used_turns, plan_turns_total),
        "contract_used_by_kind": dict(
            sorted((str(k), int(v)) for k, v in contract_used_by_kind.items() if str(k))
        ),
    }
    return transcripts, metrics


def read_transcripts_jsonl(path: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def suite_metrics_from_transcripts(
    transcripts: Sequence[Dict[str, Any]],
    *,
    prefix_k: int = 8,
    template_ngram_n: int = 6,
    template_prefix_window: int = 32,
) -> Dict[str, Any]:
    all_gen_tokens: List[str] = []
    reply_gen_tokens: List[List[str]] = []
    reply_sigs: List[str] = []
    prefix_sigs: List[str] = []
    dialogue_sigs: List[List[str]] = []
    reply_modes: List[str] = []
    dialogue_modes: List[List[str]] = []
    reply_mode_sources: List[str] = []
    reply_policy_actions: List[str] = []
    reply_policy_coverages: List[float] = []

    trace_tokens_total = 0
    trace_candidates_sum = 0
    trace_pred_matched_sum = 0
    trace_pred_emitted_sum = 0
    trace_pred_iterated_sum = 0
    trace_act_evals_sum = 0
    trace_scan_evals_sum = 0
    trace_active_set_size: int = 0
    trace_rewrite_rules_total: int = 0
    trace_selector_present: int = 0

    for rec in transcripts:
        turns = rec.get("turns", [])
        dial_sigs: List[str] = []
        dial_modes: List[str] = []
        for t in turns:
            resp = str(t.get("system", ""))
            gen_toks = tokenize_text(resp)
            reply_gen_tokens.append(gen_toks)
            all_gen_tokens.extend(gen_toks)
            sig = reply_signature(resp)
            reply_sigs.append(sig)
            dial_sigs.append(sig)
            prefix_sigs.append(prefix_k_signature(gen_toks, k=prefix_k))
            m = str(t.get("mode") or "default")
            reply_modes.append(m)
            dial_modes.append(m)
            reply_mode_sources.append(str(t.get("mode_source") or "router"))
            reply_policy_actions.append(str(t.get("mode_policy_action") or ""))
            try:
                reply_policy_coverages.append(float(t.get("policy_coverage") or 0.0))
            except Exception:
                reply_policy_coverages.append(0.0)

            tr = t.get("trace") or {}
            if isinstance(tr, dict):
                cand_post = tr.get("candidates_post_rewrite") or []
                pred_matched = tr.get("predictor_matched") or []
                pred_emitted = tr.get("predictor_emitted") or []
                pred_iterated = tr.get("predictor_iterated") or []
                if (
                    isinstance(cand_post, list)
                    and isinstance(pred_matched, list)
                    and isinstance(pred_emitted, list)
                ):
                    n_tok = min(len(cand_post), len(pred_matched), len(pred_emitted))
                    if n_tok > 0:
                        trace_tokens_total += int(n_tok)
                        trace_candidates_sum += int(sum(int(x) for x in cand_post[:n_tok]))
                        trace_pred_matched_sum += int(
                            sum(int(x) for x in pred_matched[:n_tok])
                        )
                        trace_pred_emitted_sum += int(
                            sum(int(x) for x in pred_emitted[:n_tok])
                        )
                        if isinstance(pred_iterated, list):
                            n2 = min(int(n_tok), int(len(pred_iterated)))
                            if n2 > 0:
                                trace_pred_iterated_sum += int(
                                    sum(int(x) for x in pred_iterated[:n2])
                                )

                        rr_total = int(tr.get("rewrite_rules_total", 0) or 0)
                        sel_present = 1 if tr.get("selector_id") else 0
                        trace_act_evals_sum += int(sum(int(x) for x in pred_matched[:n_tok]))
                        trace_act_evals_sum += int(n_tok * rr_total)
                        trace_act_evals_sum += int(n_tok * sel_present)
                        if isinstance(pred_iterated, list) and n2 > 0:
                            trace_scan_evals_sum += int(
                                sum(int(x) for x in pred_iterated[:n2])
                            )
                            trace_scan_evals_sum += int(n2 * rr_total)
                            trace_scan_evals_sum += int(n2 * sel_present)

                        if trace_active_set_size <= 0:
                            trace_active_set_size = int(tr.get("active_set_size", 0) or 0)
                            trace_rewrite_rules_total = int(rr_total)
                            trace_selector_present = int(sel_present)
        if dial_sigs:
            dialogue_sigs.append(dial_sigs)
            dialogue_modes.append(dial_modes)

    metrics = suite_metrics_from_generations(
        all_gen_tokens=all_gen_tokens,
        reply_gen_tokens=reply_gen_tokens,
        dialogue_reply_sigs=dialogue_sigs,
        dialogue_reply_modes=dialogue_modes,
        reply_sigs=reply_sigs,
        prefix_sigs=prefix_sigs,
        reply_modes=reply_modes,
        reply_mode_sources=reply_mode_sources,
        reply_policy_actions=reply_policy_actions,
        reply_policy_coverages=reply_policy_coverages,
        prefix_k=prefix_k,
        template_ngram_n=template_ngram_n,
        template_prefix_window=template_prefix_window,
    )
    if trace_tokens_total > 0:
        metrics["trace_tokens_total"] = int(trace_tokens_total)
        metrics["candidates_considered_per_token_mean"] = float(
            trace_candidates_sum / trace_tokens_total
        )
        metrics["predictor_matched_per_token_mean"] = float(
            trace_pred_matched_sum / trace_tokens_total
        )
        metrics["predictor_emitted_per_token_mean"] = float(
            trace_pred_emitted_sum / trace_tokens_total
        )
        metrics["acts_considered_per_token_mean"] = float(
            trace_act_evals_sum / trace_tokens_total
        )
        metrics["search_steps_per_turn_mean"] = float(
            trace_act_evals_sum / max(1, len(reply_sigs))
        )
        if trace_pred_iterated_sum > 0:
            metrics["predictor_iterated_per_token_mean"] = float(
                trace_pred_iterated_sum / trace_tokens_total
            )
        if trace_scan_evals_sum > 0:
            metrics["scan_acts_considered_per_token_mean"] = float(
                trace_scan_evals_sum / trace_tokens_total
            )
            metrics["scan_steps_per_turn_mean"] = float(
                trace_scan_evals_sum / max(1, len(reply_sigs))
            )
        metrics["trace_active_set_size"] = int(trace_active_set_size)
        metrics["trace_rewrite_rules_total"] = int(trace_rewrite_rules_total)
        metrics["trace_selector_present"] = int(trace_selector_present)
    return metrics
