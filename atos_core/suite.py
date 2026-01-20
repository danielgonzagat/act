from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .metrics import distinct_n, loop_rate, repeat_ngram_rate, tokenize_text
from .validators import run_validator


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


def goal_id_for(goal_spec: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(goal_spec).encode("utf-8"))


V67_DIALOGUE_COMPILER_ID = "v67_dialogue_compiler"

_V67_A_RE = re.compile(r"(?i)\ba\s*=\s*(-?[0-9]+)\b", flags=re.UNICODE)
_V67_B_RE = re.compile(r"(?i)\bb\s*=\s*(-?[0-9]+)\b", flags=re.UNICODE)
_V67_SUM_RE = re.compile(r"(?i)\bsum\s*=\s*(-?[0-9]+)\b", flags=re.UNICODE)
_V67_PLAN_RE = re.compile(r"(?i)\bplan\s*=\s*([^;\s\.]+)", flags=re.UNICODE)


def _strip_edge_punct_v67(s: str) -> str:
    return str(s).strip().strip(".,;:!?()[]{}").strip()


def compile_dialogue_v67(dialogue: str) -> Dict[str, Any]:
    """
    Deterministic dialogue→expected_spec compiler for UTILITY_DIALOGUES_V67.

    Parses (a,b,plan) or (sum,b,plan) from the dialogue text and returns an
    expected_spec compatible with `plan_validator` (plus minimal meta fields).
    """
    u = str(dialogue or "").strip()
    if not u:
        raise ValueError("v67_empty_dialogue")

    m_plan = _V67_PLAN_RE.search(u)
    plan = _strip_edge_punct_v67(m_plan.group(1)) if m_plan else ""
    if not plan:
        raise ValueError("v67_missing_plan")

    m_a = _V67_A_RE.search(u)
    m_b = _V67_B_RE.search(u)
    m_sum = _V67_SUM_RE.search(u)

    if m_a and m_b:
        a = int(m_a.group(1))
        b = int(m_b.group(1))
        goal_spec = {"kind": "v67_sum_state_json", "a": int(a), "b": int(b), "plan": str(plan)}
        gid = goal_id_for(goal_spec)
        input_keys = ["goal_id", "plan", "a", "b"]
        ops = [
            {"fn": "add_int", "in": ["in2", "in3"], "out": "v0"},
            {"fn": "make_dict_goal_plan_ab", "in": ["in0", "in1", "v0", "in3"], "out": "v1"},
            {"fn": "json_canonical", "in": ["v1"], "out": "v2"},
        ]
        expected_obj = {"goal_id": gid, "plan": str(plan), "a": int(a) + int(b), "b": int(b)}
        expected_text = canonical_json_dumps(expected_obj)
        return {
            "compiler_id": V67_DIALOGUE_COMPILER_ID,
            "task_kind": "v67_sum_wrap",
            "goal_spec": dict(goal_spec),
            "goal_id": str(gid),
            "input_keys": list(input_keys),
            "inputs": {"goal_id": gid, "plan": str(plan), "a": int(a), "b": int(b)},
            "ops": list(ops),
            "return_var": "v2",
            "expected_output_text": str(expected_text),
            "required_keys": ["goal_id", "plan", "a", "b"],
            "expected_values": dict(expected_obj),
        }

    if m_sum and m_b:
        sum_value = int(m_sum.group(1))
        b = int(m_b.group(1))
        goal_spec = {
            "kind": "v67_wrap_state_json",
            "sum": int(sum_value),
            "b": int(b),
            "plan": str(plan),
        }
        gid = goal_id_for(goal_spec)
        input_keys = ["goal_id", "plan", "sum", "b"]
        ops = [
            {"fn": "make_dict_goal_plan_ab", "in": ["in0", "in1", "in2", "in3"], "out": "v0"},
            {"fn": "json_canonical", "in": ["v0"], "out": "v1"},
        ]
        expected_obj = {"goal_id": gid, "plan": str(plan), "a": int(sum_value), "b": int(b)}
        expected_text = canonical_json_dumps(expected_obj)
        return {
            "compiler_id": V67_DIALOGUE_COMPILER_ID,
            "task_kind": "v67_wrap_only",
            "goal_spec": dict(goal_spec),
            "goal_id": str(gid),
            "input_keys": list(input_keys),
            "inputs": {"goal_id": gid, "plan": str(plan), "sum": int(sum_value), "b": int(b)},
            "ops": list(ops),
            "return_var": "v1",
            "expected_output_text": str(expected_text),
            "required_keys": ["goal_id", "plan", "a", "b"],
            "expected_values": dict(expected_obj),
        }

    raise ValueError("v67_parse_missing_fields")


def _v66_plan_task(
    *,
    task_id: str,
    a: int,
    b: int,
    plan: str,
) -> Dict[str, Any]:
    goal_spec = {"kind": "v66_sum_state_json", "task_id": str(task_id), "a": int(a), "b": int(b), "plan": str(plan)}
    gid = goal_id_for(goal_spec)
    input_keys = ["goal_id", "plan", "a", "b"]
    ops = [
        {"fn": "add_int", "in": ["in2", "in3"], "out": "v0"},
        {"fn": "make_dict_goal_plan_ab", "in": ["in0", "in1", "v0", "in3"], "out": "v1"},
        {"fn": "json_canonical", "in": ["v1"], "out": "v2"},
    ]
    expected_obj = {"goal_id": gid, "plan": str(plan), "a": int(a) + int(b), "b": int(b)}
    expected_text = canonical_json_dumps(expected_obj)
    expected_spec = {
        "input_keys": list(input_keys),
        "inputs": {"goal_id": gid, "plan": str(plan), "a": int(a), "b": int(b)},
        "ops": list(ops),
        "return_var": "v2",
        "expected_output_text": str(expected_text),
        "required_keys": ["goal_id", "plan", "a", "b"],
        "expected_values": dict(expected_obj),
    }
    return {
        "task_id": str(task_id),
        "dialogue": (
            "Objetivo: retorne um JSON canônico (sem texto extra) com chaves goal_id, plan, a, b. "
            "a deve ser a+b e b deve ser b. Inclua goal_id e plan exatamente.",
        ),
        "validator_id": "plan_validator",
        "expected_spec": expected_spec,
        "goal_spec": goal_spec,
        "tags": ["utility", "plan", "state", "json"],
    }


def _v66_wrap_task(
    *,
    task_id: str,
    sum_value: int,
    b: int,
    plan: str,
) -> Dict[str, Any]:
    goal_spec = {
        "kind": "v66_wrap_state_json",
        "task_id": str(task_id),
        "sum": int(sum_value),
        "b": int(b),
        "plan": str(plan),
    }
    gid = goal_id_for(goal_spec)
    input_keys = ["goal_id", "plan", "sum", "b"]
    ops = [
        {"fn": "make_dict_goal_plan_ab", "in": ["in0", "in1", "in2", "in3"], "out": "v0"},
        {"fn": "json_canonical", "in": ["v0"], "out": "v1"},
    ]
    expected_obj = {"goal_id": gid, "plan": str(plan), "a": int(sum_value), "b": int(b)}
    expected_text = canonical_json_dumps(expected_obj)
    expected_spec = {
        "input_keys": list(input_keys),
        "inputs": {"goal_id": gid, "plan": str(plan), "sum": int(sum_value), "b": int(b)},
        "ops": list(ops),
        "return_var": "v1",
        "expected_output_text": str(expected_text),
        "required_keys": ["goal_id", "plan", "a", "b"],
        "expected_values": dict(expected_obj),
    }
    return {
        "task_id": str(task_id),
        "dialogue": (
            "Objetivo: retorne um JSON canônico (sem texto extra) com chaves goal_id, plan, a, b. "
            "Aqui a já é fornecido como sum, e b deve ser b. Inclua goal_id e plan exatamente.",
        ),
        "validator_id": "plan_validator",
        "expected_spec": expected_spec,
        "goal_spec": goal_spec,
        "tags": ["utility", "plan", "state", "json"],
    }


# Utility suite v66 (semantic pressure via deterministic plan/state/goal objects).
# Output is a canonical JSON object; validation uses plan simulation (no LLM, no heuristics).
UTILITY_DIALOGUES_V66: Tuple[Dict[str, Any], ...] = (
    _v66_plan_task(task_id="v66_sum_state_00", a=4, b=8, plan="add_int+wrap_goal_plan_state"),
    _v66_plan_task(task_id="v66_sum_state_01", a=17, b=25, plan="add_int+wrap_goal_plan_state"),
    _v66_plan_task(task_id="v66_sum_state_02", a=9, b=7, plan="add_int+wrap_goal_plan_state"),
    _v66_plan_task(task_id="v66_sum_state_03", a=12, b=30, plan="add_int+wrap_goal_plan_state"),
    _v66_plan_task(task_id="v66_sum_state_04", a=99, b=1, plan="add_int+wrap_goal_plan_state"),
    _v66_plan_task(task_id="v66_sum_state_05", a=0, b=42, plan="add_int+wrap_goal_plan_state"),
    _v66_wrap_task(task_id="v66_wrap_state_06", sum_value=42, b=0, plan="wrap_goal_plan_state_only"),
    _v66_wrap_task(task_id="v66_wrap_state_07", sum_value=3, b=2, plan="wrap_goal_plan_state_only"),
    _v66_wrap_task(task_id="v66_wrap_state_08", sum_value=3, b=1, plan="wrap_goal_plan_state_only"),
    _v66_wrap_task(task_id="v66_wrap_state_09", sum_value=42, b=39, plan="wrap_goal_plan_state_only"),
    _v66_wrap_task(task_id="v66_wrap_state_10", sum_value=13, b=7, plan="wrap_goal_plan_state_only"),
    _v66_wrap_task(task_id="v66_wrap_state_11", sum_value=42, b=32, plan="wrap_goal_plan_state_only"),
)


def _v67_sum_dialogue(*, task_id: str, a: int, b: int, plan: str) -> str:
    a = int(a)
    b = int(b)
    return (
        f"Tarefa {task_id}: a={a}; b={b}; plan={plan}. "
        "Retorne APENAS um JSON canônico (sem texto extra) com chaves goal_id, plan, a, b; "
        "onde a=a+b e b=b."
    )


def _v67_wrap_dialogue(*, task_id: str, sum_value: int, b: int, plan: str) -> str:
    sum_value = int(sum_value)
    b = int(b)
    return (
        f"Tarefa {task_id}: sum={sum_value}; b={b}; plan={plan}. "
        "Retorne APENAS um JSON canônico (sem texto extra) com chaves goal_id, plan, a, b; "
        "onde a=sum e b=b."
    )


# Utility suite v67 (dialogue-driven): expected_spec is derived deterministically from the dialogue text.
UTILITY_DIALOGUES_V67: Tuple[Dict[str, Any], ...] = (
    {
        "task_id": "v67_sum_state_00",
        "dialogue": (_v67_sum_dialogue(task_id="v67_sum_state_00", a=4, b=8, plan="add_int+wrap_goal_plan_state"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v67_sum_state_01",
        "dialogue": (_v67_sum_dialogue(task_id="v67_sum_state_01", a=17, b=25, plan="add_int+wrap_goal_plan_state"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v67_sum_state_02",
        "dialogue": (_v67_sum_dialogue(task_id="v67_sum_state_02", a=9, b=7, plan="add_int+wrap_goal_plan_state"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v67_sum_state_03",
        "dialogue": (_v67_sum_dialogue(task_id="v67_sum_state_03", a=12, b=30, plan="add_int+wrap_goal_plan_state"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v67_sum_state_04",
        "dialogue": (_v67_sum_dialogue(task_id="v67_sum_state_04", a=99, b=1, plan="add_int+wrap_goal_plan_state"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v67_sum_state_05",
        "dialogue": (_v67_sum_dialogue(task_id="v67_sum_state_05", a=0, b=42, plan="add_int+wrap_goal_plan_state"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v67_wrap_state_06",
        "dialogue": (_v67_wrap_dialogue(task_id="v67_wrap_state_06", sum_value=42, b=0, plan="wrap_goal_plan_state_only"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v67_wrap_state_07",
        "dialogue": (_v67_wrap_dialogue(task_id="v67_wrap_state_07", sum_value=3, b=2, plan="wrap_goal_plan_state_only"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v67_wrap_state_08",
        "dialogue": (_v67_wrap_dialogue(task_id="v67_wrap_state_08", sum_value=3, b=1, plan="wrap_goal_plan_state_only"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v67_wrap_state_09",
        "dialogue": (_v67_wrap_dialogue(task_id="v67_wrap_state_09", sum_value=42, b=39, plan="wrap_goal_plan_state_only"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v67_wrap_state_10",
        "dialogue": (_v67_wrap_dialogue(task_id="v67_wrap_state_10", sum_value=13, b=7, plan="wrap_goal_plan_state_only"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v67_wrap_state_11",
        "dialogue": (_v67_wrap_dialogue(task_id="v67_wrap_state_11", sum_value=42, b=32, plan="wrap_goal_plan_state_only"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
)


def _v68_sum_dialogue(*, a: int, b: int, plan: str) -> str:
    a = int(a)
    b = int(b)
    return (
        f"Missão: a={a}; b={b}; plan={plan}. "
        "Retorne APENAS um JSON canônico (sem texto extra) com chaves goal_id, plan, a, b; "
        "onde a=a+b e b=b."
    )


def _v68_wrap_dialogue(*, sum_value: int, b: int, plan: str) -> str:
    sum_value = int(sum_value)
    b = int(b)
    return (
        f"Missão: sum={sum_value}; b={b}; plan={plan}. "
        "Retorne APENAS um JSON canônico (sem texto extra) com chaves goal_id, plan, a, b; "
        "onde a=sum e b=b."
    )


# Utility suite v68: contract trigger MUST be metadata-only (plan_trace); dialogues intentionally break V67 regex triggers.
UTILITY_DIALOGUES_V68: Tuple[Dict[str, Any], ...] = (
    {
        "task_id": "v68_sum_state_00",
        "dialogue": (_v68_sum_dialogue(a=4, b=8, plan="add_int+wrap_goal_plan_state"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v68_sum_state_01",
        "dialogue": (_v68_sum_dialogue(a=17, b=25, plan="add_int+wrap_goal_plan_state"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v68_sum_state_02",
        "dialogue": (_v68_sum_dialogue(a=9, b=7, plan="add_int+wrap_goal_plan_state"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v68_sum_state_03",
        "dialogue": (_v68_sum_dialogue(a=12, b=30, plan="add_int+wrap_goal_plan_state"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v68_sum_state_04",
        "dialogue": (_v68_sum_dialogue(a=99, b=1, plan="add_int+wrap_goal_plan_state"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v68_sum_state_05",
        "dialogue": (_v68_sum_dialogue(a=0, b=42, plan="add_int+wrap_goal_plan_state"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v68_wrap_state_06",
        "dialogue": (_v68_wrap_dialogue(sum_value=42, b=0, plan="wrap_goal_plan_state_only"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v68_wrap_state_07",
        "dialogue": (_v68_wrap_dialogue(sum_value=3, b=2, plan="wrap_goal_plan_state_only"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v68_wrap_state_08",
        "dialogue": (_v68_wrap_dialogue(sum_value=3, b=1, plan="wrap_goal_plan_state_only"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v68_wrap_state_09",
        "dialogue": (_v68_wrap_dialogue(sum_value=42, b=39, plan="wrap_goal_plan_state_only"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v68_wrap_state_10",
        "dialogue": (_v68_wrap_dialogue(sum_value=13, b=7, plan="wrap_goal_plan_state_only"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
    },
    {
        "task_id": "v68_wrap_state_11",
        "dialogue": (_v68_wrap_dialogue(sum_value=42, b=32, plan="wrap_goal_plan_state_only"),),
        "validator_id": "plan_validator",
        "compiler_id": V67_DIALOGUE_COMPILER_ID,
        "tags": ["utility", "plan", "state", "json"],
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
    # Core deterministic validators (atos_core/validators.py).
    "instruction_following_validator": lambda out, spec: (
        lambda vr: (bool(vr.passed), str(vr.reason))
    )(run_validator("instruction_following_validator", out, spec)),
    "state_validator": lambda out, spec: (
        lambda vr: (bool(vr.passed), str(vr.reason))
    )(run_validator("state_validator", out, spec)),
    "plan_validator": lambda out, spec: (lambda vr: (bool(vr.passed), str(vr.reason)))(
        run_validator("plan_validator", out, spec)
    ),
}


def _plan_trace_for_task(task: Dict[str, Any], *, turn_idx: int) -> Dict[str, Any]:
    task_id = str(task.get("task_id") or "")
    dialogue = task.get("dialogue") or ()
    validate_turn = int(task.get("validate_turn", max(0, len(dialogue) - 1)) or 0)
    validator_id = str(task.get("validator_id") or "") if int(turn_idx) == int(validate_turn) else ""
    expected_spec = task.get("expected_spec") or {}
    compiler_id = str(task.get("compiler_id") or "")

    if (not isinstance(expected_spec, dict) or not expected_spec) and validator_id == "plan_validator":
        if compiler_id == V67_DIALOGUE_COMPILER_ID or str(task_id).startswith("v67_"):
            try:
                if isinstance(dialogue, (list, tuple)) and dialogue:
                    expected_spec = compile_dialogue_v67(str(dialogue[validate_turn]))
            except Exception:
                expected_spec = {}

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
    elif validator_id == "plan_validator":
        expected_format = "plan"
        constraints = ["plan_validator", "json_canonical"]
    elif validator_id == "state_validator":
        expected_format = "state"
        constraints = ["state_validator", "json"]
    elif validator_id == "instruction_following_validator":
        expected_format = "instruction"
        constraints = ["instruction_following_validator"]

    goal_id = ""
    if isinstance(expected_spec, dict) and str(expected_spec.get("goal_id") or ""):
        goal_id = str(expected_spec.get("goal_id") or "")
    else:
        goal_spec = task.get("goal_spec")
        if not isinstance(goal_spec, dict) or not goal_spec:
            goal_spec = expected_spec.get("goal_spec") if isinstance(expected_spec, dict) else None
        if isinstance(goal_spec, dict) and goal_spec:
            goal_id = goal_id_for(goal_spec)
    exp_sig = ""
    if isinstance(expected_spec, dict) and expected_spec:
        try:
            exp_sig = sha256_hex(canonical_json_dumps(expected_spec).encode("utf-8"))
        except Exception:
            exp_sig = ""

    return {
        "task_id": task_id,
        "compiler_id": str(compiler_id),
        "validator_id": validator_id,
        "expected_format": expected_format,
        "constraints": constraints,
        "goal_id": str(goal_id),
        "goal_active": bool(goal_id),
        "goal_progress": None,
        "goal_satisfied": None,
        "expected_spec_sig": str(exp_sig),
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
    goal_shadow_log_path: Optional[str] = None,
    goal_shadow_max_goals_per_turn: Optional[int] = None,
    goal_shadow_trace_log_path: Optional[str] = None,
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
                plan_trace=None,
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

            # Shadow-only: evaluate persistent goals after the normal chat turn and log,
            # without affecting token generation. Opt-in by providing a log path.
            if goal_shadow_log_path or goal_shadow_trace_log_path:
                try:
                    goals = []
                    try:
                        goals_fn = getattr(engine.store, "goal_acts", None)
                        if callable(goals_fn):
                            goals = list(goals_fn())
                    except Exception:
                        goals = []
                    goals = [g for g in goals if getattr(g, "active", True)]
                    # Deterministic scheduler: stable order by (-priority, id), then
                    # optional round-robin selection across turns.
                    def _goal_key(a: Any) -> Tuple[int, str]:
                        try:
                            ev = getattr(a, "evidence", None)
                            if not isinstance(ev, dict):
                                return (0, str(getattr(a, "id", "")))
                            goal = ev.get("goal")
                            if not isinstance(goal, dict):
                                return (0, str(getattr(a, "id", "")))
                            pr = int(goal.get("priority", 0) or 0)
                            return (-pr, str(getattr(a, "id", "")))
                        except Exception:
                            return (0, str(getattr(a, "id", "")))

                    goals.sort(key=_goal_key)

                    ctx_sig = f"chat␟d={i}␟t={j}"
                    total_goals = int(len(goals))
                    max_goals = goal_shadow_max_goals_per_turn
                    if max_goals is None:
                        max_goals_i = total_goals
                    else:
                        max_goals_i = max(0, int(max_goals))

                    # When limiting work, we round-robin (rotate) the goal list by a stable offset.
                    if total_goals > 0 and max_goals is not None and max_goals_i < total_goals:
                        start = int((int(i) * 1000 + int(j)) % total_goals)
                        ordered = [goals[(start + k) % total_goals] for k in range(total_goals)]
                    else:
                        ordered = list(goals)

                    selected_ids: set = set()
                    if total_goals > 0 and max_goals is not None and 0 < max_goals_i < total_goals:
                        selected_ids = {str(getattr(g, "id", "")) for g in ordered[:max_goals_i]}
                    else:
                        selected_ids = {str(getattr(g, "id", "")) for g in ordered}

                    capture_events = bool(max_goals is not None and max_goals_i <= 8)

                    log_rows: List[Dict[str, Any]] = []
                    trace_rows: List[Dict[str, Any]] = []
                    for g in ordered:
                        gid = str(getattr(g, "id", ""))
                        skipped = gid not in selected_ids

                        inputs: Dict[str, Any] = {}
                        try:
                            ev = getattr(g, "evidence", None)
                            if isinstance(ev, dict):
                                goal = ev.get("goal")
                                if isinstance(goal, dict) and isinstance(goal.get("inputs"), dict):
                                    inputs = dict(goal.get("inputs") or {})
                        except Exception:
                            inputs = {}

                        if skipped:
                            log_rows.append(
                                {
                                    "ctx_sig": str(ctx_sig),
                                    "goal_id": str(gid),
                                    "selected_concept_id": "",
                                    "goal_active": True,
                                    "goal_progress": None,
                                    "goal_satisfied": None,
                                    "ok": None,
                                    "reason": "skipped_by_scheduler",
                                    "skipped_by_scheduler": True,
                                }
                            )
                            if goal_shadow_trace_log_path:
                                trace_rows.append(
                                    {
                                        "ctx_sig": str(ctx_sig),
                                        "goal_id": str(gid),
                                        "skipped_by_scheduler": True,
                                        "inputs": dict(inputs),
                                        "output_text": "",
                                        "selected_concept_id": "",
                                        "events_sig": sha256_hex(b""),
                                        "program_sig": sha256_hex(b""),
                                        "events": [],
                                    }
                                )
                            continue

                        gr = engine.execute_goal(goal_act_id=str(gid), step=int(csv_step), max_depth=8)
                        tr = gr.get("trace") if isinstance(gr, dict) else {}
                        tr = tr if isinstance(tr, dict) else {}
                        meta = tr.get("concept_meta") if isinstance(tr.get("concept_meta"), dict) else {}
                        out_text = str(meta.get("output_text") or "")
                        selected_cid = str(tr.get("selected_concept_id") or "")
                        ok = bool(gr.get("ok", False))
                        reason = str(gr.get("reason") or "")

                        log_rows.append(
                            {
                                "ctx_sig": str(ctx_sig),
                                "goal_id": str(gid),
                                "selected_concept_id": str(selected_cid),
                                "goal_active": True,
                                "goal_progress": float(tr.get("goal_progress", 0.0) or 0.0),
                                "goal_satisfied": bool(tr.get("goal_satisfied", False)),
                                "ok": bool(ok),
                                "reason": str(reason),
                                "skipped_by_scheduler": False,
                            }
                        )

                        if goal_shadow_trace_log_path:
                            evs = gr.get("events") if isinstance(gr, dict) else []
                            evs = evs if isinstance(evs, list) else []
                            evs = [e for e in evs if isinstance(e, dict)]
                            evs_sig = sha256_hex(canonical_json_dumps(evs).encode("utf-8"))
                            row: Dict[str, Any] = {
                                "ctx_sig": str(ctx_sig),
                                "goal_id": str(gid),
                                "skipped_by_scheduler": False,
                                "inputs": dict(inputs),
                                "output_text": str(out_text),
                                "selected_concept_id": str(selected_cid),
                                "events_sig": str(evs_sig),
                                "program_sig": str(evs_sig),
                            }
                            if capture_events:
                                row["events"] = list(evs)
                            else:
                                row["events"] = []
                                row["events_truncated"] = True
                            trace_rows.append(row)

                    if goal_shadow_log_path:
                        os.makedirs(os.path.dirname(goal_shadow_log_path) or ".", exist_ok=True)
                        with open(goal_shadow_log_path, "a", encoding="utf-8") as f:
                            for row in log_rows:
                                f.write(
                                    json.dumps(
                                        row,
                                        ensure_ascii=False,
                                        sort_keys=True,
                                        separators=(",", ":"),
                                    )
                                )
                                f.write("\n")
                    if goal_shadow_trace_log_path and trace_rows:
                        os.makedirs(os.path.dirname(goal_shadow_trace_log_path) or ".", exist_ok=True)
                        with open(goal_shadow_trace_log_path, "a", encoding="utf-8") as f:
                            for row in trace_rows:
                                f.write(
                                    json.dumps(
                                        row,
                                        ensure_ascii=False,
                                        sort_keys=True,
                                        separators=(",", ":"),
                                    )
                                )
                                f.write("\n")
                except Exception:
                    pass

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
    goals_total = 0
    goals_satisfied = 0

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
        if (not isinstance(expected_spec, dict) or not expected_spec) and validator_id == "plan_validator":
            compiler_id = str(task.get("compiler_id") or "")
            if compiler_id == V67_DIALOGUE_COMPILER_ID or str(task_id).startswith("v67_"):
                try:
                    expected_spec = compile_dialogue_v67(str(turns[validate_turn]))
                except Exception:
                    expected_spec = {}
        tags = task.get("tags") or []
        if not isinstance(tags, list):
            tags = []

        goal_spec = task.get("goal_spec")
        if not isinstance(goal_spec, dict) or not goal_spec:
            goal_spec = expected_spec.get("goal_spec") if isinstance(expected_spec, dict) else None

        total_tasks += 1
        if isinstance(goal_spec, dict) and goal_spec:
            goals_total += 1
        for cat in ("instruction", "json", "math", "state"):
            if cat in tags:
                cat_total[cat] += 1

        history: List[Dict[str, Any]] = []
        for j, user_msg in enumerate(turns):
            history.append({"user": str(user_msg), "system": ""})
            plan_trace = _plan_trace_for_task(task, turn_idx=int(j))
            prompt = build_chat_prompt(history)
            out = engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                mode="greedy",
                dialogue_id=int(i),
                turn=int(j),
                plan_trace=plan_trace,
            )
            resp = out["text"][len(prompt) :]
            history[-1]["system"] = resp
            history[-1]["mode"] = str(out.get("mode") or "default")
            history[-1]["mode_source"] = str(out.get("mode_source") or "router")
            history[-1]["mode_policy_action"] = str(out.get("mode_policy_action") or "")
            history[-1]["policy_coverage"] = float(out.get("policy_coverage") or 0.0)
            history[-1]["user_sig"] = str(out.get("user_sig") or "")

            tr = dict(out.get("trace") or {})
            contract_meta = tr.get("instruction_contract")
            if isinstance(contract_meta, dict):
                if bool(contract_meta.get("used")):
                    contract_used_turns += 1
                    k = str(contract_meta.get("kind") or "")
                    if k:
                        contract_used_by_kind[k] += 1
            history[-1]["trace"] = tr

            plan_turns_total += 1
            if not isinstance(tr.get("plan_trace"), dict):
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
            if isinstance(goal_spec, dict) and goal_spec:
                goals_satisfied += 1
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
        "goals_total": int(goals_total),
        "goals_satisfied": int(goals_satisfied),
        "goals_satisfied_rate": _rate(goals_satisfied, goals_total),
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
