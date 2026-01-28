from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .metrics import distinct_n, loop_rate, repeat_ngram_rate, tokenize_text
from .validators import run_validator
from .semantic_ids import plan_id_for_expected_spec_sig, reference_id_v1
from .world_pressure_validators_v123 import validate_iac_v123


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


# Extra deterministic tasks used by the SOTA-ish packs (no world knowledge).
SKILL_DIALOGUES_V2_SOTA_EXTRA: Tuple[Dict[str, Any], ...] = (
    # Longer state retention across multiple turns.
    {
        "task_id": "state_long_delta8",
        "dialogue": (
            "Memorize a palavra-código: DELTA8.",
            "Responda exatamente: OK",
            "Responda exatamente: OK",
            "Repita a palavra-código exatamente.",
        ),
        "validate_turn": 3,
        "validator_id": "exact_match",
        "expected_spec": {"text": "DELTA8", "collapse_ws": True, "case_sensitive": True},
        "tags": ["state"],
    },
    # Clarification under ambiguity (must ask a question instead of guessing).
    {
        "task_id": "clarify_ambiguous_number_pt",
        "dialogue": ("Me dê o número. (Não especifiquei qual.)",),
        "validator_id": "regex_fullmatch",
        "expected_spec": {"pattern": r"(?is).*\bqual\b.*\?.*", "flags": 0},
        "tags": ["clarify"],
    },
    # Consistency / contradiction handling (deterministic rule).
    {
        "task_id": "consistency_contradiction_flag",
        "dialogue": (
            "Contexto: a senha é ALFA9. Apenas memorize.",
            "Agora diga que a senha é BETA7. Se isso contradiz o contexto, responda exatamente: CONTRADIÇÃO",
        ),
        "validate_turn": 1,
        "validator_id": "exact_match",
        "expected_spec": {"text": "CONTRADIÇÃO", "collapse_ws": True, "case_sensitive": True},
        "tags": ["consistency"],
    },
    {
        "task_id": "consistency_no_contradiction_branch",
        "dialogue": (
            "Contexto: a senha é GAMMA3. Apenas memorize.",
            "Agora, diga que a senha é GAMMA3. Se isso contradiz o contexto, responda exatamente: CONTRADIÇÃO; "
            "caso contrário responda exatamente com a senha.",
        ),
        "validate_turn": 1,
        "validator_id": "exact_match",
        "expected_spec": {"text": "GAMMA3", "collapse_ws": True, "case_sensitive": True},
        "tags": ["consistency"],
    },
    # Multi-step arithmetic (still deterministic and self-contained).
    {
        "task_id": "math_multistep_9_7",
        "dialogue": ("Calcule (9 + 7) * 2 - 10. Responda APENAS com o número inteiro.",),
        "validator_id": "int_exact",
        "expected_spec": {"value": 22, "strict": True},
        "tags": ["math"],
    },
)


# Deterministic long-memory tasks (closed world, no external knowledge).
# These force explicit memory state (via concept_csv + memory_facts) by separating
# store and recall across the pack ordering (so the secret is not present in the recall prompt).
SKILL_DIALOGUES_V3_MEMORY_LONG: Tuple[Dict[str, Any], ...] = (
    {
        "task_id": "memory_long_store_secret_v1",
        "dialogue": ("Memória: armazene o segredo. Responda exatamente: OK",),
        "validate_turn": 0,
        "validator_id": "exact_match",
        "expected_spec": {"text": "OK", "collapse_ws": True, "case_sensitive": True},
        # Generic concept-executor spec (executed by the runtime, not by token continuation).
        "concept_expected_spec": {
            "inputs": {"key": "mem_long_secret_v1", "value": "DELTA8"},
            "input_keys": ["key", "value"],
            "expected_for_validator": "OK",
        },
        "concept_validator_id": "text_exact",
        "tags": ["memory"],
    },
    {
        "task_id": "memory_long_recall_secret_v1",
        "dialogue": ("Memória: recupere o segredo e responda exatamente com ele (sem texto extra).",),
        "validate_turn": 0,
        "validator_id": "exact_match",
        "expected_spec": {"text": "DELTA8", "collapse_ws": True, "case_sensitive": True},
        "concept_expected_spec": {
            "inputs": {"key": "mem_long_secret_v1"},
            "input_keys": ["key"],
            "expected_for_validator": "DELTA8",
        },
        "concept_validator_id": "text_exact",
        "tags": ["memory"],
    },
)


# Long-dialogue coherence tasks (closed world; deterministic). These are designed to be
# run with a truncated prompt history (e.g., last 4 turns) + explicit `dialogue_state`
# so earlier facts/goals must persist beyond the visible prompt window.
SKILL_DIALOGUES_V4_DIALOGUE_LONG: Tuple[Dict[str, Any], ...] = (
    {
        "task_id": "dialogue_long_recall_name_pt_v1",
        "dialogue": (
            "Meu nome é CARLOS. Apenas memorize. Responda exatamente: OK",
            "Responda exatamente: OK",
            "Calcule 2 + 2. Responda APENAS com o número inteiro.",
            "Responda exatamente: OK",
            "Responda exatamente: OK",
            "Qual meu nome? Responda APENAS com ele.",
        ),
        "validate_turn": 5,
        "validator_id": "contains_exact_token",
        "expected_spec": {"token": "CARLOS", "case_sensitive": False},
        "tags": ["dialogue"],
    },
    {
        "task_id": "dialogue_long_resume_topic_en_v1",
        "dialogue": (
            "Let's talk about PROJECT ALPHA. Just remember. Reply exactly: OK",
            "Reply exactly: OK",
            "Compute (3 + 5) * 2. Reply with the integer only.",
            "Reply exactly: OK",
            "Reply exactly: OK",
            "Back to the previous topic: what project were we discussing? Reply with the project name only.",
        ),
        "validate_turn": 5,
        "validator_id": "contains_exact_token",
        "expected_spec": {"token": "PROJECT ALPHA", "case_sensitive": False},
        "tags": ["dialogue"],
    },
)


def goal_id_for(goal_spec: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(goal_spec).encode("utf-8"))


def _v69_long_sum_task(
    *,
    task_id: str,
    values: Sequence[int],
    b: int,
    plan: str,
    concept_min_depth: int = 2,
) -> Dict[str, Any]:
    vals = [int(x) for x in list(values)]
    goal_spec = {
        "kind": "v69_long_sum_state_json",
        "task_id": str(task_id),
        "plan": str(plan),
        "values": list(vals),
        "b": int(b),
    }
    gid = goal_id_for(goal_spec)
    input_keys = ["goal_id", "plan"] + [f"x{i}" for i in range(len(vals))] + ["b"]
    inputs: Dict[str, Any] = {"goal_id": str(gid), "plan": str(plan), "b": int(b)}
    for i, v in enumerate(vals):
        inputs[f"x{i}"] = int(v)

    # Ops: chain add_int over all x_i (>=2), then wrap + canonicalize.
    ops: List[Dict[str, Any]] = []
    if len(vals) < 2:
        raise ValueError("v69_long_sum_requires_at_least_2_values")

    # in0=goal_id, in1=plan, in2=x0, in3=x1, ... , in{len(vals)+1}=x{len(vals)-1}, last in = b
    v_prev = "v0"
    ops.append({"fn": "add_int", "in": ["in2", "in3"], "out": v_prev})
    for i in range(2, len(vals)):
        v_next = f"v{i-1}"
        ops.append({"fn": "add_int", "in": [v_prev, f"in{2 + i}"], "out": v_next})
        v_prev = v_next

    v_obj = f"v{len(vals)}"
    v_out = f"v{len(vals) + 1}"
    ops.append(
        {
            "fn": "make_dict_goal_plan_ab",
            "in": ["in0", "in1", v_prev, f"in{len(vals) + 2}"],
            "out": v_obj,
        }
    )
    ops.append({"fn": "json_canonical", "in": [v_obj], "out": v_out})

    expected_obj = {"goal_id": str(gid), "plan": str(plan), "a": int(sum(vals)), "b": int(b)}
    expected_text = canonical_json_dumps(expected_obj)
    expected_spec = {
        "compiler_id": "v69_long_sum_compiler",
        "task_kind": "v69_long_sum_wrap",
        "goal_spec": dict(goal_spec),
        "goal_id": str(gid),
        "input_keys": list(input_keys),
        "inputs": dict(inputs),
        "ops": list(ops),
        "return_var": str(v_out),
        "expected_output_text": str(expected_text),
        "required_keys": ["goal_id", "plan", "a", "b"],
        "expected_values": dict(expected_obj),
    }

    return {
        "task_id": str(task_id),
        "dialogue": (
            f"V69 long plan. Reply with the exact canonical JSON for goal_id={gid} (plan={plan}).",
        ),
        "validator_id": "plan_validator",
        "expected_spec": expected_spec,
        "concept_min_depth": int(concept_min_depth),
        "tags": ["utility", "plan", "math"],
    }


# Long-horizon deterministic plan tasks (closed world; no world knowledge). These force the
# system to sustain long multi-step execution and/or planning under the same SOTA regime.
SKILL_DIALOGUES_V5_LONG_HORIZON: Tuple[Dict[str, Any], ...] = (
    _v69_long_sum_task(
        task_id="v69_long_sum_21a",
        values=list(range(1, 22)),
        b=0,
        plan="long_sum_chain_21",
        concept_min_depth=2,
    ),
    _v69_long_sum_task(
        task_id="v69_long_sum_25a",
        values=list(range(1, 26)),
        b=0,
        plan="long_sum_chain_25",
        concept_min_depth=2,
    ),
)


# Longer-horizon deterministic plan tasks (>=50 steps). Used to stress multi-step execution
# beyond the 20–30 step range in sota_v7.
SKILL_DIALOGUES_V6_LONG_HORIZON: Tuple[Dict[str, Any], ...] = (
    _v69_long_sum_task(
        task_id="v69_long_sum_49a",
        values=list(range(1, 50)),
        b=0,
        plan="long_sum_chain_49",
        concept_min_depth=2,
    ),
    _v69_long_sum_task(
        task_id="v69_long_sum_75a",
        values=list(range(1, 76)),
        b=0,
        plan="long_sum_chain_75",
        concept_min_depth=2,
    ),
)


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


def _with_plan_concept_min_depth(
    tasks: Sequence[Dict[str, Any]],
    *,
    min_depth: int,
) -> Tuple[Dict[str, Any], ...]:
    """
    Deterministically enforce concept compositionality for plan_validator tasks:
    require concept_calls_max_depth >= min_depth at validation time.
    """
    md = int(min_depth or 0)
    if md < 0:
        md = 0
    out: List[Dict[str, Any]] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        t2 = dict(t)
        if str(t2.get("validator_id") or "") == "plan_validator":
            t2["concept_min_depth"] = int(md)
        out.append(t2)
    return tuple(out)


def _with_plan_iac_required(tasks: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    """
    Deterministically enforce IAC (Intention->Action->Consequence) as survival law for plan tasks:
    require explicit Goal + Plan objects to exist in the ActStore.
    """
    out: List[Dict[str, Any]] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        t2 = dict(t)
        if str(t2.get("validator_id") or "") == "plan_validator":
            t2["iac_required"] = True
            t2["hypothesis_required"] = True
        out.append(t2)
    return tuple(out)


def _with_plan_reference_required(tasks: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    """
    Deterministically enforce semantic reference bindings as survival law for plan tasks:
    require explicit token→object bindings (Reference ACTs) to exist in the ActStore.
    """
    out: List[Dict[str, Any]] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        t2 = dict(t)
        if str(t2.get("validator_id") or "") == "plan_validator":
            t2["reference_required"] = True
            # Within the goal scope, bind stable tokens to the current goal/plan objects.
            t2["reference_tokens"] = [
                {"token": "o objetivo", "target_kind": "goal"},
                {"token": "o plano", "target_kind": "plan"},
            ]
        out.append(t2)
    return tuple(out)


def _with_concept_policy_required(tasks: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    """
    Deterministically enforce concept-as-policy as survival law:
    tasks must select and execute an explicit concept_csv (dominant policy), with no global
    fallback "search" path.
    """
    out: List[Dict[str, Any]] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        t2 = dict(t)
        t2["concept_policy_required"] = True
        out.append(t2)
    return tuple(out)


def _with_plan_concept_csg_min_complexity(
    tasks: Sequence[Dict[str, Any]],
    *,
    min_nodes: int,
    min_edges: int,
) -> Tuple[Dict[str, Any], ...]:
    """
    Deterministically enforce "concepts as non-trivial subgraphs" for plan_validator tasks:
    require the executed concept_csv to have a CSG (v87) with at least `min_nodes` call-nodes and
    `min_edges` dataflow edges (i.e., not a wrapper-only deepwrap chain).
    """
    mn = int(min_nodes or 0)
    me = int(min_edges or 0)
    if mn < 0:
        mn = 0
    if me < 0:
        me = 0
    out: List[Dict[str, Any]] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        t2 = dict(t)
        if str(t2.get("validator_id") or "") == "plan_validator":
            t2["concept_csg_min_nodes"] = int(mn)
            t2["concept_csg_min_edges"] = int(me)
        out.append(t2)
    return tuple(out)


# Skill-suite packs used by training/eval loops. These are deterministic "validator packs"
# (no world knowledge) and can be used as a bottleneck (LOSS-AND) across categories.
SKILL_SUITE_PACKS: Dict[str, Tuple[Dict[str, Any], ...]] = {
    "v0": SKILL_DIALOGUES_V0,
    "v1_paraphrase": SKILL_DIALOGUES_V1_PARAPHRASE,
    "v68_plan": UTILITY_DIALOGUES_V68,
    # SOTA-ish pack: baseline + paraphrases + plan/state/json deterministic tasks.
    "sota_v1": SKILL_DIALOGUES_V0 + SKILL_DIALOGUES_V1_PARAPHRASE + UTILITY_DIALOGUES_V68,
    # Extended SOTA-ish pack: add ambiguity/consistency/long-state tasks.
    "sota_v2": SKILL_DIALOGUES_V0
    + SKILL_DIALOGUES_V1_PARAPHRASE
    + UTILITY_DIALOGUES_V68
    + SKILL_DIALOGUES_V2_SOTA_EXTRA,
    # SOTA-ish + concept compositionality: plan tasks require nested concept calls (CSV_CALL).
    "sota_v3": _with_plan_concept_min_depth(
        SKILL_DIALOGUES_V0
        + SKILL_DIALOGUES_V1_PARAPHRASE
        # Include both explicit-spec (v66) and dialogue-compiled (v68) plan families to force
        # cross-context transfer and prevent "single-family plan concepts" from surviving indefinitely.
        + UTILITY_DIALOGUES_V66
        + UTILITY_DIALOGUES_V68
        + SKILL_DIALOGUES_V2_SOTA_EXTRA,
        min_depth=1,
    ),
}

# SOTA-ish + compositionality + long-memory (store+recall) enforced as concept execution.
# Ordering: store task, then core pack, then recall task (so the secret is not present in recall prompt).
SKILL_SUITE_PACKS["sota_v4"] = (
    (SKILL_DIALOGUES_V3_MEMORY_LONG[0],)
    + SKILL_SUITE_PACKS["sota_v3"]
    + (SKILL_DIALOGUES_V3_MEMORY_LONG[1],)
)

# SOTA-ish + long-memory + deeper concept hierarchy: plan tasks require nested depth >=2.
SKILL_SUITE_PACKS["sota_v5"] = (
    (SKILL_DIALOGUES_V3_MEMORY_LONG[0],)
    + _with_plan_concept_min_depth(SKILL_SUITE_PACKS["sota_v3"], min_depth=2)
    + (SKILL_DIALOGUES_V3_MEMORY_LONG[1],)
)

# SOTA-ish + long-memory + deep concept hierarchy + long-dialogue coherence.
SKILL_SUITE_PACKS["sota_v6"] = (
    (SKILL_DIALOGUES_V3_MEMORY_LONG[0],)
    + _with_plan_concept_min_depth(SKILL_SUITE_PACKS["sota_v3"], min_depth=2)
    + SKILL_DIALOGUES_V4_DIALOGUE_LONG
    + (SKILL_DIALOGUES_V3_MEMORY_LONG[1],)
)

# SOTA-ish + long-memory + deep concept hierarchy + long-dialogue coherence + long-horizon plans.
# Ordering: store secret, then long-horizon plans (to excite planning/long reasoning), then core SOTA pack,
# then long-dialogue coherence, then recall secret.
SKILL_SUITE_PACKS["sota_v7"] = (
    (SKILL_DIALOGUES_V3_MEMORY_LONG[0],)
    + SKILL_DIALOGUES_V5_LONG_HORIZON
    + _with_plan_concept_min_depth(SKILL_SUITE_PACKS["sota_v3"], min_depth=2)
    + SKILL_DIALOGUES_V4_DIALOGUE_LONG
    + (SKILL_DIALOGUES_V3_MEMORY_LONG[1],)
)

# SOTA-ish + long-memory + deeper concept hierarchy + long-dialogue coherence + longer-horizon plans.
# This pushes:
# - multi-step execution (>50 steps)
# - nested concept execution depth >=3 (concept_min_depth).
SKILL_SUITE_PACKS["sota_v8"] = (
    (SKILL_DIALOGUES_V3_MEMORY_LONG[0],)
    + _with_plan_concept_min_depth(
        SKILL_DIALOGUES_V5_LONG_HORIZON + SKILL_DIALOGUES_V6_LONG_HORIZON + SKILL_SUITE_PACKS["sota_v3"],
        min_depth=3,
    )
    + SKILL_DIALOGUES_V4_DIALOGUE_LONG
    + (SKILL_DIALOGUES_V3_MEMORY_LONG[1],)
)

# SOTA v8 + world-pressure IAC law: plan tasks require Goal+Plan objects (not only correct output).
SKILL_SUITE_PACKS["sota_v9"] = _with_plan_iac_required(SKILL_SUITE_PACKS["sota_v8"])

# SOTA v9 + binding law: plan tasks require explicit Reference objects (token→goal/plan).
SKILL_SUITE_PACKS["sota_v10"] = _with_plan_reference_required(SKILL_SUITE_PACKS["sota_v9"])

# SOTA v10 + concept-as-policy law: no survival without selecting a concept as dominant policy.
SKILL_SUITE_PACKS["sota_v11"] = _with_concept_policy_required(SKILL_SUITE_PACKS["sota_v10"])

# SOTA v11 + semantic CSG law: plan tasks require non-trivial concept subgraphs (>=2 nodes, >=1 edge).
SKILL_SUITE_PACKS["sota_v12"] = _with_plan_concept_csg_min_complexity(
    SKILL_SUITE_PACKS["sota_v11"],
    min_nodes=2,
    min_edges=1,
)


def _xdom_v1_long_sum_task(
    *,
    task_id: str,
    values: Sequence[int],
    b: int,
    plan: str,
    concept_min_depth: int,
) -> Dict[str, Any]:
    """
    Cross-domain plan task (non-ARC): same logical primitives (add_int + canonical JSON),
    but different compiler_id/family_id so schema overlap with SOTA packs is 0.
    """
    vals = [int(x) for x in list(values)]
    goal_spec = {
        "kind": "xdom_v1_energy_balance_json",
        "task_id": str(task_id),
        "plan": str(plan),
        "values": list(vals),
        "b": int(b),
    }
    gid = goal_id_for(goal_spec)
    input_keys = ["goal_id", "plan"] + [f"x{i}" for i in range(len(vals))] + ["b"]
    inputs: Dict[str, Any] = {"goal_id": str(gid), "plan": str(plan), "b": int(b)}
    for i, v in enumerate(vals):
        inputs[f"x{i}"] = int(v)

    ops: List[Dict[str, Any]] = []
    if len(vals) < 2:
        raise ValueError("xdom_v1_long_sum_requires_at_least_2_values")
    v_prev = "v0"
    ops.append({"fn": "add_int", "in": ["in2", "in3"], "out": v_prev})
    for i in range(2, len(vals)):
        v_next = f"v{i-1}"
        ops.append({"fn": "add_int", "in": [v_prev, f"in{2 + i}"], "out": v_next})
        v_prev = v_next
    v_obj = f"v{len(vals)}"
    v_out = f"v{len(vals) + 1}"
    ops.append(
        {
            "fn": "make_dict_goal_plan_ab",
            "in": ["in0", "in1", v_prev, f"in{len(vals) + 2}"],
            "out": v_obj,
        }
    )
    ops.append({"fn": "json_canonical", "in": [v_obj], "out": v_out})

    expected_obj = {"goal_id": str(gid), "plan": str(plan), "a": int(sum(vals)), "b": int(b)}
    expected_text = canonical_json_dumps(expected_obj)
    expected_spec = {
        "task_kind": "xdom_v1_long_sum_wrap",
        "goal_spec": dict(goal_spec),
        "goal_id": str(gid),
        "input_keys": list(input_keys),
        "inputs": dict(inputs),
        "ops": list(ops),
        "return_var": str(v_out),
        "expected_output_text": str(expected_text),
        "required_keys": ["goal_id", "plan", "a", "b"],
        "expected_values": dict(expected_obj),
    }

    return {
        "task_id": str(task_id),
        # Critical for schema_overlap==0 in cross-domain tests: family_id includes compiler_id.
        "compiler_id": "xdom_v1_compiler",
        "dialogue": (
            f"XDOM v1. Energy-balance domain. Reply with the exact canonical JSON "
            f"for goal_id={gid} (plan={plan}).",
        ),
        "validator_id": "plan_validator",
        "expected_spec": expected_spec,
        "concept_min_depth": int(concept_min_depth),
        "tags": ["utility", "plan", "math", "xdom"],
    }


# Cross-domain suite (non-ARC): same logical primitives, different family ids (schema-disjoint).
# Keep it small and deterministic; used by CROSS_DOMAIN_CONCEPT_REUSE.
SKILL_SUITE_PACKS["xdom_v1"] = _with_plan_concept_csg_min_complexity(
    _with_concept_policy_required(
        _with_plan_concept_min_depth(
            (
                _xdom_v1_long_sum_task(
                    task_id="xdom_v1_sum_2a",
                    values=[7, 11],
                    b=3,
                    plan="xdom_energy_chain_2",
                    concept_min_depth=2,
                ),
                _xdom_v1_long_sum_task(
                    task_id="xdom_v1_sum_3a",
                    values=[3, 5, 9],
                    b=1,
                    plan="xdom_energy_chain_3",
                    concept_min_depth=2,
                ),
                _xdom_v1_long_sum_task(
                    task_id="xdom_v1_sum_5a",
                    values=[2, 4, 6, 8, 10],
                    b=0,
                    plan="xdom_energy_chain_5",
                    concept_min_depth=2,
                ),
                _xdom_v1_long_sum_task(
                    task_id="xdom_v1_sum_8a",
                    values=[1, 1, 2, 3, 5, 8, 13, 21],
                    b=34,
                    plan="xdom_energy_chain_8",
                    concept_min_depth=2,
                ),
            ),
            min_depth=2,
        )
    ),
    min_nodes=2,
    min_edges=1,
)


def _xdom_v2_sum_plus_offset_task(
    *,
    task_id: str,
    values: Sequence[int],
    b: int,
    plan: str,
    concept_min_depth: int,
) -> Dict[str, Any]:
    """
    Cross-domain plan task (non-ARC): compute (a=sum(values), b=sum(values)+b_input),
    then wrap + canonicalize. This differs from xdom_v1 where b is an independent input.
    """
    vals = [int(x) for x in list(values)]
    goal_spec = {
        "kind": "xdom_v2_sum_plus_offset_json",
        "task_id": str(task_id),
        "plan": str(plan),
        "values": list(vals),
        "b": int(b),
    }
    gid = goal_id_for(goal_spec)
    input_keys = ["goal_id", "plan"] + [f"x{i}" for i in range(len(vals))] + ["b"]
    inputs: Dict[str, Any] = {"goal_id": str(gid), "plan": str(plan), "b": int(b)}
    for i, v in enumerate(vals):
        inputs[f"x{i}"] = int(v)

    ops: List[Dict[str, Any]] = []
    if len(vals) < 2:
        raise ValueError("xdom_v2_sum_plus_offset_requires_at_least_2_values")

    # Sum chain over x_i.
    v_prev = "v0"
    ops.append({"fn": "add_int", "in": ["in2", "in3"], "out": v_prev})
    for i in range(2, len(vals)):
        v_next = f"v{i-1}"
        ops.append({"fn": "add_int", "in": [v_prev, f"in{2 + i}"], "out": v_next})
        v_prev = v_next

    # b_out = sum(values) + b_input
    v_b = f"v{len(vals)}"
    ops.append({"fn": "add_int", "in": [v_prev, f"in{len(vals) + 2}"], "out": v_b})

    v_obj = f"v{len(vals) + 1}"
    v_out = f"v{len(vals) + 2}"
    ops.append({"fn": "make_dict_goal_plan_ab", "in": ["in0", "in1", v_prev, v_b], "out": v_obj})
    ops.append({"fn": "json_canonical", "in": [v_obj], "out": v_out})

    expected_obj = {"goal_id": str(gid), "plan": str(plan), "a": int(sum(vals)), "b": int(sum(vals) + int(b))}
    expected_text = canonical_json_dumps(expected_obj)
    expected_spec = {
        "task_kind": "xdom_v2_sum_plus_offset_wrap",
        "goal_spec": dict(goal_spec),
        "goal_id": str(gid),
        "input_keys": list(input_keys),
        "inputs": dict(inputs),
        "ops": list(ops),
        "return_var": str(v_out),
        "expected_output_text": str(expected_text),
        "required_keys": ["goal_id", "plan", "a", "b"],
        "expected_values": dict(expected_obj),
    }

    return {
        "task_id": str(task_id),
        "compiler_id": "xdom_v2_compiler",
        "dialogue": (
            f"XDOM v2. Sum-plus-offset domain. Reply with the exact canonical JSON "
            f"for goal_id={gid} (plan={plan}).",
        ),
        "validator_id": "plan_validator",
        "expected_spec": expected_spec,
        "concept_min_depth": int(concept_min_depth),
        "tags": ["utility", "plan", "math", "xdom"],
    }


# Cross-domain suite v2 (schema-disjoint): b depends on a (sum), unlike xdom_v1.
SKILL_SUITE_PACKS["xdom_v2"] = _with_plan_concept_csg_min_complexity(
    _with_concept_policy_required(
        _with_plan_concept_min_depth(
            (
                _xdom_v2_sum_plus_offset_task(
                    task_id="xdom_v2_sumoff_2a",
                    values=[7, 11],
                    b=3,
                    plan="xdom_sum_plus_offset_2",
                    concept_min_depth=2,
                ),
                _xdom_v2_sum_plus_offset_task(
                    task_id="xdom_v2_sumoff_3a",
                    values=[3, 5, 9],
                    b=1,
                    plan="xdom_sum_plus_offset_3",
                    concept_min_depth=2,
                ),
                _xdom_v2_sum_plus_offset_task(
                    task_id="xdom_v2_sumoff_5a",
                    values=[2, 4, 6, 8, 10],
                    b=0,
                    plan="xdom_sum_plus_offset_5",
                    concept_min_depth=2,
                ),
                _xdom_v2_sum_plus_offset_task(
                    task_id="xdom_v2_sumoff_8a",
                    values=[1, 1, 2, 3, 5, 8, 13, 21],
                    b=34,
                    plan="xdom_sum_plus_offset_8",
                    concept_min_depth=2,
                ),
            ),
            min_depth=2,
        )
    ),
    min_nodes=2,
    min_edges=1,
)


def _xdom_v3_split_even_odd_task(
    *,
    task_id: str,
    values: Sequence[int],
    plan: str,
    concept_min_depth: int,
) -> Dict[str, Any]:
    """
    Cross-domain plan task (non-ARC): split a sequence into even/odd index sums:
      a = sum(values[0], values[2], ...), b = sum(values[1], values[3], ...)
    This induces different internal structure (two accumulators).
    """
    vals = [int(x) for x in list(values)]
    if len(vals) < 4 or (len(vals) % 2) != 0:
        raise ValueError("xdom_v3_split_even_odd_requires_even_len_ge4")

    goal_spec = {
        "kind": "xdom_v3_split_even_odd_json",
        "task_id": str(task_id),
        "plan": str(plan),
        "values": list(vals),
    }
    gid = goal_id_for(goal_spec)
    input_keys = ["goal_id", "plan"] + [f"x{i}" for i in range(len(vals))]
    inputs: Dict[str, Any] = {"goal_id": str(gid), "plan": str(plan)}
    for i, v in enumerate(vals):
        inputs[f"x{i}"] = int(v)

    # in0=goal_id, in1=plan, in2=x0, in3=x1, ...
    ops: List[Dict[str, Any]] = []
    # seed sums
    v_even = "v0"
    v_odd = "v1"
    ops.append({"fn": "add_int", "in": ["in2", "in4"], "out": v_even})
    ops.append({"fn": "add_int", "in": ["in3", "in5"], "out": v_odd})
    # accumulate remaining pairs
    out_idx = 2
    for i in range(6, 2 + len(vals), 2):
        v_next_even = f"v{out_idx}"
        v_next_odd = f"v{out_idx + 1}"
        ops.append({"fn": "add_int", "in": [v_even, f"in{i}"], "out": v_next_even})
        ops.append({"fn": "add_int", "in": [v_odd, f"in{i + 1}"], "out": v_next_odd})
        v_even = v_next_even
        v_odd = v_next_odd
        out_idx += 2

    v_obj = f"v{out_idx}"
    v_out = f"v{out_idx + 1}"
    ops.append({"fn": "make_dict_goal_plan_ab", "in": ["in0", "in1", v_even, v_odd], "out": v_obj})
    ops.append({"fn": "json_canonical", "in": [v_obj], "out": v_out})

    a_val = int(sum(vals[0::2]))
    b_val = int(sum(vals[1::2]))
    expected_obj = {"goal_id": str(gid), "plan": str(plan), "a": int(a_val), "b": int(b_val)}
    expected_text = canonical_json_dumps(expected_obj)
    expected_spec = {
        "task_kind": "xdom_v3_split_even_odd_wrap",
        "goal_spec": dict(goal_spec),
        "goal_id": str(gid),
        "input_keys": list(input_keys),
        "inputs": dict(inputs),
        "ops": list(ops),
        "return_var": str(v_out),
        "expected_output_text": str(expected_text),
        "required_keys": ["goal_id", "plan", "a", "b"],
        "expected_values": dict(expected_obj),
    }

    return {
        "task_id": str(task_id),
        "compiler_id": "xdom_v3_compiler",
        "dialogue": (
            f"XDOM v3. Split-sum domain (even/odd). Reply with the exact canonical JSON "
            f"for goal_id={gid} (plan={plan}).",
        ),
        "validator_id": "plan_validator",
        "expected_spec": expected_spec,
        "concept_min_depth": int(concept_min_depth),
        "tags": ["utility", "plan", "math", "xdom"],
    }


# Cross-domain suite v3 (schema-disjoint): two-accumulator split-sum structure.
SKILL_SUITE_PACKS["xdom_v3"] = _with_plan_concept_csg_min_complexity(
    _with_concept_policy_required(
        _with_plan_concept_min_depth(
            (
                _xdom_v3_split_even_odd_task(
                    task_id="xdom_v3_split_4a",
                    values=[7, 11, 3, 5],
                    plan="xdom_split_even_odd_4",
                    concept_min_depth=2,
                ),
                _xdom_v3_split_even_odd_task(
                    task_id="xdom_v3_split_6a",
                    values=[1, 2, 3, 5, 8, 13],
                    plan="xdom_split_even_odd_6",
                    concept_min_depth=2,
                ),
                _xdom_v3_split_even_odd_task(
                    task_id="xdom_v3_split_8a",
                    values=[1, 1, 2, 3, 5, 8, 13, 21],
                    plan="xdom_split_even_odd_8",
                    concept_min_depth=2,
                ),
                _xdom_v3_split_even_odd_task(
                    task_id="xdom_v3_split_10a",
                    values=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                    plan="xdom_split_even_odd_10",
                    concept_min_depth=2,
                ),
            ),
            min_depth=2,
        )
    ),
    min_nodes=2,
    min_edges=1,
)

def skill_suite_tasks_for_pack(pack: str) -> Tuple[Dict[str, Any], ...]:
    key = str(pack or "v0").strip().lower()
    aliases = {
        "v1": "v1_paraphrase",
        "paraphrase": "v1_paraphrase",
        "v68": "v68_plan",
        "plan": "v68_plan",
        "sota": "sota_v1",
        "sota_v2": "sota_v2",
        "sota_v3": "sota_v3",
        "sota_v4": "sota_v4",
        "sota_v5": "sota_v5",
        "sota_v6": "sota_v6",
        "sota_v7": "sota_v7",
        "sota_v8": "sota_v8",
        "sota_v9": "sota_v9",
        "sota_v10": "sota_v10",
        "sota_v11": "sota_v11",
        "sota_v12": "sota_v12",
        "xdom": "xdom_v1",
        "xdom_v1": "xdom_v1",
        "xdom_v2": "xdom_v2",
        "xdom_v3": "xdom_v3",
    }
    key = aliases.get(key, key)
    tasks = SKILL_SUITE_PACKS.get(key)
    if tasks is None:
        raise ValueError(f"unknown_skill_suite_pack:{pack}")
    return tasks


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

    def _normalize_constraint(c: str) -> str:
        s = str(c or "")
        if not s:
            return ""
        if s.startswith("token:"):
            return "token:*"
        if s.startswith("keys:"):
            raw = s[len("keys:") :]
            parts = [p.strip() for p in raw.split(",")] if raw else []
            parts = [p for p in parts if p]
            return f"keys:n={len(parts)}"
        return s

    constraints_norm = [_normalize_constraint(c) for c in constraints if str(c)]
    constraints_norm = [c for c in constraints_norm if str(c)]
    try:
        family_body = {
            "validator_id": str(validator_id),
            "expected_format": str(expected_format),
            "constraints": list(constraints_norm),
            "compiler_id": str(compiler_id),
            "goal_active": bool(goal_id),
        }
        family_id = "fam_" + sha256_hex(canonical_json_dumps(family_body).encode("utf-8"))[:16]
    except Exception:
        family_id = ""

    out = {
        "task_id": task_id,
        "compiler_id": str(compiler_id),
        "validator_id": validator_id,
        "expected_format": expected_format,
        "constraints": constraints,
        "family_id": str(family_id),
        "goal_id": str(goal_id),
        "goal_active": bool(goal_id),
        "goal_progress": None,
        "goal_satisfied": None,
        "expected_spec_sig": str(exp_sig),
    }
    # For plan_validator tasks, include the full expected_spec so the runtime can execute an
    # induced concept_csv (no hidden solver; execution must flow through explicit concept acts).
    if validator_id == "plan_validator" and isinstance(expected_spec, dict) and expected_spec:
        out["expected_spec"] = dict(expected_spec)
        try:
            out["concept_min_depth"] = int(task.get("concept_min_depth", 0) or 0)
        except Exception:
            out["concept_min_depth"] = 0
        try:
            out["concept_csg_min_nodes"] = int(task.get("concept_csg_min_nodes", 0) or 0)
        except Exception:
            out["concept_csg_min_nodes"] = 0
        try:
            out["concept_csg_min_edges"] = int(task.get("concept_csg_min_edges", 0) or 0)
        except Exception:
            out["concept_csg_min_edges"] = 0
    # Generic concept executor (non-plan tasks): when the harness provides an explicit
    # concept_expected_spec, attach it to the plan_trace for the validate turn so that the
    # runtime can execute a matching concept_csv ACT (no hidden solver).
    if int(turn_idx) == int(validate_turn):
        ces = task.get("concept_expected_spec")
        if isinstance(ces, dict) and ces:
            out["concept_expected_spec"] = dict(ces)
            out["concept_validator_id"] = str(task.get("concept_validator_id") or "")
            try:
                out["concept_min_depth"] = int(task.get("concept_min_depth", out.get("concept_min_depth", 0)) or 0)
            except Exception:
                out["concept_min_depth"] = int(out.get("concept_min_depth", 0) or 0)
            try:
                out["concept_csg_min_nodes"] = int(
                    task.get("concept_csg_min_nodes", out.get("concept_csg_min_nodes", 0)) or 0
                )
            except Exception:
                out["concept_csg_min_nodes"] = int(out.get("concept_csg_min_nodes", 0) or 0)
            try:
                out["concept_csg_min_edges"] = int(
                    task.get("concept_csg_min_edges", out.get("concept_csg_min_edges", 0)) or 0
                )
            except Exception:
                out["concept_csg_min_edges"] = int(out.get("concept_csg_min_edges", 0) or 0)
        # Reference bindings: optional, deterministic contract. This does not mutate state; it is
        # an explicit requirement that can be enforced by validators + maintained by ICS.
        if bool(task.get("reference_required", False)) or isinstance(task.get("reference_tokens"), list):
            out["reference_required"] = bool(task.get("reference_required", False))
            rts = task.get("reference_tokens") if isinstance(task.get("reference_tokens"), list) else []
            items: List[Dict[str, str]] = []
            for r in rts:
                if not isinstance(r, dict):
                    continue
                tok = str(r.get("token") or "")
                tk = str(r.get("target_kind") or "")
                if tok and tk:
                    items.append({"token": tok, "target_kind": tk})
            items.sort(key=lambda d: (str(d.get("target_kind") or ""), str(d.get("token") or "")))
            if items:
                out["reference_tokens"] = list(items)
        if bool(task.get("concept_policy_required", False)):
            out["concept_policy_required"] = True
    return out


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

    # Per-reply repetition/loop metrics (more stable and closer to dialogue quality than
    # concatenating all replies into a single stream).
    rep3_by_reply = [repeat_ngram_rate(toks, 3, ignore_space=True) for toks in reply_gen_tokens]
    loop_by_reply = [loop_rate(toks, n=3, window=128, ignore_space=True) for toks in reply_gen_tokens]
    rep3_reply_mean = (sum(rep3_by_reply) / len(rep3_by_reply)) if rep3_by_reply else 0.0
    loop_reply_mean = (sum(loop_by_reply) / len(loop_by_reply)) if loop_by_reply else 0.0
    rep3_reply_max = max(rep3_by_reply) if rep3_by_reply else 0.0
    loop_reply_max = max(loop_by_reply) if loop_by_reply else 0.0

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
        "repeat3_reply_mean": float(rep3_reply_mean),
        "loop_rate_reply_mean": float(loop_reply_mean),
        "repeat3_reply_max": float(rep3_reply_max),
        "loop_rate_reply_max": float(loop_reply_max),
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
            prompt_used = str(out.get("prompt") or prompt)
            resp = out["text"][len(prompt_used) :]
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
    prompt_history_k: int = 0,
    family_shuffle_seed: Optional[int] = None,
    family_shuffle_salt: int = 0,
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

    # Concept usage as survival law: for structured plan tasks, require the runtime to execute
    # an explicit concept_csv (not just emit the right text). This closes the escape route where
    # the system can survive without depending on induced semantic objects.
    concept_total = 0
    concept_used_turns = 0
    concept_ok_turns = 0
    concept_calls_total_sum = 0
    concept_calls_max_depth_sum = 0
    concept_composed_turns = 0
    concept_deep_turns = 0
    concept_very_deep_turns = 0
    concept_calls_max_depth_max = 0
    concept_csg_nodes_sum = 0
    concept_csg_edges_sum = 0
    concept_csg_count_turns = 0
    concept_csg_rich_turns = 0
    concept_csg_required_total = 0
    concept_csg_required_ok_turns = 0
    concept_min_depth_required_max = 0
    concept_nested_call_turns = 0
    concept_nested_call_ids: set = set()

    # Reference/binding as survival law (semantic persistence): require explicit token→object
    # bindings for certain tasks (even if the output matches).
    reference_required_total = 0
    reference_ok_turns = 0

    def _normalize_ref_token(token: str) -> str:
        return " ".join(str(token or "").strip().split()).lower()

    # Concept-as-policy law: require a concept executor to be selected (no global fallback).
    concept_policy_required_total = 0
    concept_policy_used_turns = 0

    # Concept usage (any task): used to prove cross-context semantic reuse beyond plan tasks.
    concept_any_used_turns = 0
    concept_any_ok_turns = 0
    concept_used_tags_by_id: Dict[str, set] = {}
    concept_usage_by_id: Dict[str, Dict[str, Any]] = {}

    plan_turns_total = 0
    plan_turns_missing = 0
    contract_used_turns = 0
    contract_used_by_kind: Counter = Counter()
    csv_step = 0

    family_id_map: Dict[str, str] = {}
    if family_shuffle_seed is not None:
        # Deterministic permutation over the latent family ids present in this task pack.
        # This is an ablation knob: if the system depends on stable latent families, shuffling
        # them should break convergence/performance.
        try:
            fids: set = set()
            for i0, task0 in enumerate(tasks):
                if not isinstance(task0, dict):
                    continue
                dialogue0 = task0.get("dialogue") or ()
                if not isinstance(dialogue0, (list, tuple)) or not dialogue0:
                    continue
                vt0 = int(task0.get("validate_turn", max(0, len(dialogue0) - 1)) or 0)
                vt0 = max(0, min(vt0, len(dialogue0) - 1))
                pt0 = _plan_trace_for_task(task0, turn_idx=int(vt0))
                if not isinstance(pt0, dict):
                    continue
                if not str(pt0.get("validator_id") or ""):
                    continue
                fid0 = str(pt0.get("family_id") or "")
                if fid0:
                    fids.add(fid0)
            ordered = sorted(fids, key=str)
            perm = sorted(
                ordered,
                key=lambda fid: sha256_hex(
                    f"{int(family_shuffle_seed)}:{int(family_shuffle_salt)}:{fid}".encode("utf-8")
                ),
            )
            if len(ordered) == len(perm) and ordered:
                family_id_map = {ordered[i]: perm[i] for i in range(len(ordered))}
        except Exception:
            family_id_map = {}

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
        for cat in ("instruction", "json", "math", "state", "plan", "clarify", "consistency", "memory", "dialogue"):
            if cat in tags:
                cat_total[cat] += 1

        history: List[Dict[str, Any]] = []
        for j, user_msg in enumerate(turns):
            history.append({"user": str(user_msg), "system": ""})
            plan_trace = _plan_trace_for_task(task, turn_idx=int(j))
            # Optional ablation: shuffle family ids (validate-turn contexts only).
            try:
                if family_id_map and isinstance(plan_trace, dict) and str(plan_trace.get("validator_id") or ""):
                    fid0 = str(plan_trace.get("family_id") or "")
                    if fid0 and fid0 in family_id_map:
                        plan_trace = dict(plan_trace, family_id=str(family_id_map[fid0]))
            except Exception:
                pass
            if isinstance(plan_trace, dict) and int(prompt_history_k) > 0:
                plan_trace = dict(plan_trace, prompt_history_k=int(prompt_history_k))
            if int(prompt_history_k) > 0 and len(history) > int(prompt_history_k):
                prompt = build_chat_prompt(history[-int(prompt_history_k) :])
            else:
                prompt = build_chat_prompt(history)
            out = engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                mode="greedy",
                dialogue_id=int(i),
                turn=int(j),
                plan_trace=plan_trace,
            )
            prompt_used = str(out.get("prompt") or prompt)
            resp = out["text"][len(prompt_used) :]
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

        # Concept executor requirement (survival law):
        # - plan_validator tasks always require explicit concept execution
        # - tasks may opt-in via task["concept_expected_spec"] (e.g., long-memory)
        concept_required = bool(str(validator_id) == "plan_validator" or bool(task.get("concept_expected_spec")))
        if concept_required:
            concept_total += 1
            min_depth = int(task.get("concept_min_depth", 0) or 0)
            if min_depth < 0:
                min_depth = 0
            concept_min_depth_required_max = max(int(concept_min_depth_required_max), int(min_depth))
            try:
                csg_req_nodes = int(task.get("concept_csg_min_nodes", 0) or 0)
            except Exception:
                csg_req_nodes = 0
            try:
                csg_req_edges = int(task.get("concept_csg_min_edges", 0) or 0)
            except Exception:
                csg_req_edges = 0
            csg_req_nodes = max(0, int(csg_req_nodes))
            csg_req_edges = max(0, int(csg_req_edges))
            tr0 = turn_rec.get("trace") if isinstance(turn_rec.get("trace"), dict) else {}
            cm = tr0.get("concept_executor") if isinstance(tr0.get("concept_executor"), dict) else {}
            used = bool(cm.get("used", False))
            ok2 = bool(cm.get("ok", False))
            try:
                calls_total = int(cm.get("concept_calls_total", 0) or 0)
            except Exception:
                calls_total = 0
            try:
                calls_max_depth = int(cm.get("concept_calls_max_depth", 0) or 0)
            except Exception:
                calls_max_depth = 0
            concept_calls_total_sum += int(max(0, calls_total))
            concept_calls_max_depth_sum += int(max(0, calls_max_depth))
            concept_calls_max_depth_max = max(int(concept_calls_max_depth_max), int(max(0, calls_max_depth)))
            if used:
                concept_used_turns += 1
            if ok2 and int(calls_max_depth) >= int(min_depth):
                concept_ok_turns += 1
            # Composed/deep rates are only meaningful when the concept executor is correct:
            # count nested depth only for successful concept execution (used + ok2).
            if used and ok2:
                if int(calls_max_depth) >= 1:
                    concept_composed_turns += 1
                if int(calls_max_depth) >= 2:
                    concept_deep_turns += 1
                if int(calls_max_depth) >= 3:
                    concept_very_deep_turns += 1

                # CSG richness: measure subgraph structure of the executed concept.
                # Prefer stored csg_v87; fallback to deriving from the program (CSV_CALL + bind dataflow).
                nodes = 0
                edges = 0
                try:
                    cid = str(cm.get("concept_id") or "")
                    act0 = engine.store.get_concept_act(cid) if cid else None
                    if act0 is not None:
                        ev0 = getattr(act0, "evidence", None)
                        ev0 = ev0 if isinstance(ev0, dict) else {}
                        meta0 = ev0.get("meta") if isinstance(ev0.get("meta"), dict) else {}
                        if isinstance(meta0, dict):
                            try:
                                nodes = int(meta0.get("csg_v87_nodes", 0) or 0)
                            except Exception:
                                nodes = 0
                            try:
                                edges = int(meta0.get("csg_v87_edges", 0) or 0)
                            except Exception:
                                edges = 0
                        csg0 = ev0.get("csg_v87") if isinstance(ev0.get("csg_v87"), dict) else None
                        if (nodes <= 0 and edges <= 0) and isinstance(csg0, dict):
                            nn = csg0.get("nodes") if isinstance(csg0.get("nodes"), list) else []
                            ee = csg0.get("edges") if isinstance(csg0.get("edges"), list) else []
                            nodes = int(len(nn))
                            edges = int(len(ee))
                        if nodes <= 0 and edges <= 0:
                            calls: List[Dict[str, Any]] = []
                            for ins in list(getattr(act0, "program", []) or []):
                                if str(getattr(ins, "op", "")) != "CSV_CALL":
                                    continue
                                args0 = getattr(ins, "args", {}) or {}
                                args0 = args0 if isinstance(args0, dict) else {}
                                outv = str(args0.get("out") or "")
                                bind0 = args0.get("bind") if isinstance(args0.get("bind"), dict) else {}
                                calls.append({"out": outv, "bind": dict(bind0)})
                            nodes = int(len(calls))
                            prod: Dict[str, int] = {}
                            for i0, c0 in enumerate(calls):
                                ov = str(c0.get("out") or "")
                                if ov:
                                    prod[ov] = int(i0)
                            e_cnt = 0
                            for j0, c0 in enumerate(calls):
                                bind0 = c0.get("bind") if isinstance(c0.get("bind"), dict) else {}
                                for v0 in bind0.values():
                                    vv = str(v0 or "")
                                    if not vv:
                                        continue
                                    i0 = prod.get(vv)
                                    if i0 is None or int(i0) == int(j0):
                                        continue
                                    e_cnt += 1
                            edges = int(e_cnt)
                except Exception:
                    nodes = 0
                    edges = 0

                concept_csg_nodes_sum += int(max(0, nodes))
                concept_csg_edges_sum += int(max(0, edges))
                concept_csg_count_turns += 1
                if int(nodes) >= 2 and int(edges) >= 1:
                    concept_csg_rich_turns += 1

                # Optional survival law: plan packs may require non-trivial CSG complexity.
                if int(csg_req_nodes) > 0 or int(csg_req_edges) > 0:
                    concept_csg_required_total += 1
                    ok_csg = bool(int(nodes) >= int(csg_req_nodes) and int(edges) >= int(csg_req_edges))
                    if ok_csg:
                        concept_csg_required_ok_turns += 1
                    if ok and (not ok_csg) and int(calls_max_depth) >= int(min_depth):
                        ok = False
                        reason = "concept_csg_too_shallow"
            elif int(csg_req_nodes) > 0 or int(csg_req_edges) > 0:
                # Requirement present but concept executor not ok/used: count as unmet.
                concept_csg_required_total += 1
            if ok and (not used):
                ok = False
                reason = "missing_concept_executor"
            elif ok and used and (not ok2):
                ok = False
                reason = "concept_executor_not_ok"
            elif ok and used and ok2 and int(calls_max_depth) < int(min_depth):
                ok = False
                reason = "concept_depth_too_shallow"

        # IAC requirement (world-pressure law): require explicit Goal+Plan objects to exist.
        # This can fail even when output is correct, closing the escape route "good text without semantics".
        if bool(task.get("iac_required", False)):
            goal_ok = False
            plan_ok = False
            eval_ok = False
            consequence_ok = False
            try:
                pt = turn_rec.get("trace") if isinstance(turn_rec.get("trace"), dict) else {}
                pt = pt.get("plan_trace") if isinstance(pt.get("plan_trace"), dict) else {}
            except Exception:
                pt = {}
            goal_id = str(pt.get("goal_id") or "")
            exp_sig = str(pt.get("expected_spec_sig") or "")
            plan_id = plan_id_for_expected_spec_sig(str(exp_sig))
            try:
                g = engine.store.get(str(goal_id)) if goal_id else None
                goal_ok = bool(
                    g is not None
                    and bool(getattr(g, "active", True))
                    and str(getattr(g, "kind", "")) == "goal"
                )
            except Exception:
                goal_ok = False
            try:
                p = engine.store.get(str(plan_id)) if plan_id else None
                plan_ok = bool(
                    p is not None
                    and bool(getattr(p, "active", True))
                    and str(getattr(p, "kind", "")) == "plan"
                )
            except Exception:
                plan_ok = False
            try:
                tr0 = turn_rec.get("trace") if isinstance(turn_rec.get("trace"), dict) else {}
                cm0 = tr0.get("concept_executor") if isinstance(tr0.get("concept_executor"), dict) else {}
                eval_ok = bool(cm0.get("used", False)) and bool(cm0.get("ok", False))
            except Exception:
                eval_ok = False
            consequence_ok = bool(ok)
            ok_iac, reason_iac = validate_iac_v123(
                goal_ok=bool(goal_ok),
                plan_ok=bool(plan_ok),
                eval_ok=bool(eval_ok),
                consequence_ok=bool(consequence_ok),
            )
            if ok and (not bool(ok_iac)):
                ok = False
                reason = str(reason_iac)

        # Reference requirement (binding law): require explicit token→object bindings to exist.
        # This can fail even when output is correct, closing the escape route "good text without bindings".
        if bool(task.get("reference_required", False)):
            reference_required_total += 1
            if bool(ok):
                try:
                    pt = turn_rec.get("trace") if isinstance(turn_rec.get("trace"), dict) else {}
                    pt = pt.get("plan_trace") if isinstance(pt.get("plan_trace"), dict) else {}
                except Exception:
                    pt = {}
                goal_id = str(pt.get("goal_id") or "")
                exp_sig = str(pt.get("expected_spec_sig") or "")
                plan_id = plan_id_for_expected_spec_sig(str(exp_sig))
                scope_sig = str(goal_id)
                ref_specs = pt.get("reference_tokens") if isinstance(pt.get("reference_tokens"), list) else None
                if not isinstance(ref_specs, list) or not ref_specs:
                    ref_specs = task.get("reference_tokens") if isinstance(task.get("reference_tokens"), list) else []
                ok_refs = True
                for rs in list(ref_specs):
                    if not isinstance(rs, dict):
                        continue
                    tok = str(rs.get("token") or "")
                    tk = str(rs.get("target_kind") or "")
                    if not tok or not tk:
                        continue
                    if tk == "goal":
                        want_id = str(goal_id)
                    elif tk == "plan":
                        want_id = str(plan_id)
                    else:
                        # Unknown binding kind ⇒ fail-closed.
                        ok_refs = False
                        reason = "reference_unknown_target_kind_v1"
                        break
                    rid = reference_id_v1(scope_sig=str(scope_sig), token=_normalize_ref_token(tok))
                    try:
                        ra = engine.store.get(str(rid)) if rid else None
                    except Exception:
                        ra = None
                    if ra is None or (not bool(getattr(ra, "active", True))) or str(getattr(ra, "kind", "")) != "reference":
                        ok_refs = False
                        reason = "reference_missing_v1"
                        break
                    evr = getattr(ra, "evidence", None)
                    evr = evr if isinstance(evr, dict) else {}
                    rr = evr.get("reference") if isinstance(evr.get("reference"), dict) else {}
                    if str(rr.get("target_kind") or "") != str(tk):
                        ok_refs = False
                        reason = "reference_kind_mismatch_v1"
                        break
                    if str(rr.get("target_id") or "") != str(want_id):
                        ok_refs = False
                        reason = "reference_target_mismatch_v1"
                        break
                if not bool(ok_refs):
                    ok = False
                else:
                    reference_ok_turns += 1

        # Concept-as-policy requirement: require an explicit concept to be selected/executed
        # (even if the output matches). This closes the escape route where tasks are solved by
        # raw token generation or other non-concept fallbacks.
        if bool(task.get("concept_policy_required", False)):
            concept_policy_required_total += 1
            try:
                trp = turn_rec.get("trace") if isinstance(turn_rec.get("trace"), dict) else {}
                cm_p = trp.get("concept_executor") if isinstance(trp.get("concept_executor"), dict) else {}
                used_p = bool(cm_p.get("used", False))
            except Exception:
                used_p = False
            if used_p:
                concept_policy_used_turns += 1
            if ok and (not used_p):
                ok = False
                reason = "missing_concept_policy_v1"

        # Hypothesis requirement: if a goal has a prior failure streak and we still fail now,
        # require at least one explicit hypothesis object for that goal.
        if bool(task.get("hypothesis_required", False)) and (not bool(ok)):
            try:
                pt = turn_rec.get("trace") if isinstance(turn_rec.get("trace"), dict) else {}
                pt = pt.get("plan_trace") if isinstance(pt.get("plan_trace"), dict) else {}
            except Exception:
                pt = {}
            goal_id = str(pt.get("goal_id") or "")
            streak = 0
            try:
                g = engine.store.get(str(goal_id)) if goal_id else None
                if g is not None and bool(getattr(g, "active", True)) and str(getattr(g, "kind", "")) == "goal":
                    ev = getattr(g, "evidence", None)
                    if isinstance(ev, dict):
                        gg = ev.get("goal")
                        if isinstance(gg, dict):
                            streak = int(gg.get("failure_streak", 0) or 0)
            except Exception:
                streak = 0
            if int(streak) >= 1:
                have_hyp = False
                try:
                    for h in engine.store.by_kind("hypothesis"):
                        evh = getattr(h, "evidence", None)
                        if not isinstance(evh, dict):
                            continue
                        hh = evh.get("hypothesis")
                        if not isinstance(hh, dict):
                            continue
                        if str(hh.get("goal_id") or "") == str(goal_id):
                            have_hyp = True
                            break
                except Exception:
                    have_hyp = False
                if not have_hyp:
                    ok = False
                    reason = "missing_hypothesis_v1"

        # Track concept reuse even when not required by the harness.
        try:
            tr0_any = turn_rec.get("trace") if isinstance(turn_rec.get("trace"), dict) else {}
            cm_any = tr0_any.get("concept_executor") if isinstance(tr0_any.get("concept_executor"), dict) else {}
            used_any = bool(cm_any.get("used", False))
            ok_any = bool(cm_any.get("ok", False))
            if used_any:
                concept_any_used_turns += 1
            if used_any and ok_any:
                concept_any_ok_turns += 1
            if used_any:
                ids: List[str] = []
                cid_main = str(cm_any.get("concept_id") or "")
                if cid_main:
                    ids.append(cid_main)
                call_ids = cm_any.get("concept_call_ids")
                if isinstance(call_ids, list):
                    for x in call_ids:
                        xs = str(x or "")
                        if xs:
                            ids.append(xs)
                nested_ids = cm_any.get("concept_nested_call_ids")
                if isinstance(nested_ids, list):
                    for x in nested_ids:
                        xs = str(x or "")
                        if xs:
                            ids.append(xs)
                    if nested_ids and ok_any:
                        concept_nested_call_turns += 1
                        for x in nested_ids:
                            xs = str(x or "")
                            if xs:
                                concept_nested_call_ids.add(xs)
                ids = sorted(set(ids))
                fam_id = ""
                try:
                    pt0 = tr0_any.get("plan_trace") if isinstance(tr0_any.get("plan_trace"), dict) else {}
                    fam_id = str(pt0.get("family_id") or "")
                except Exception:
                    fam_id = ""
                try:
                    calls_max_depth_any = int(cm_any.get("concept_calls_max_depth", 0) or 0)
                except Exception:
                    calls_max_depth_any = 0
                if ids:
                    for cid in ids:
                        if not cid:
                            continue
                        rec = concept_usage_by_id.get(cid)
                        if rec is None:
                            rec = {
                                "used_turns": 0,
                                "ok_turns": 0,
                                "calls_max_depth_max": 0,
                                "families": {},
                            }
                            concept_usage_by_id[cid] = rec
                        rec["used_turns"] = int(rec.get("used_turns", 0) or 0) + (1 if used_any else 0)
                        rec["ok_turns"] = int(rec.get("ok_turns", 0) or 0) + (1 if (used_any and ok_any) else 0)
                        rec["calls_max_depth_max"] = max(int(rec.get("calls_max_depth_max", 0) or 0), int(calls_max_depth_any))
                        fams = rec.get("families")
                        if not isinstance(fams, dict):
                            fams = {}
                            rec["families"] = fams
                        if fam_id:
                            frec = fams.get(fam_id)
                            if frec is None:
                                frec = {"used_turns": 0, "ok_turns": 0, "example_task_ids": set()}
                                fams[fam_id] = frec
                            frec["used_turns"] = int(frec.get("used_turns", 0) or 0) + (1 if used_any else 0)
                            frec["ok_turns"] = int(frec.get("ok_turns", 0) or 0) + (1 if (used_any and ok_any) else 0)
                            ex = frec.get("example_task_ids")
                            if not isinstance(ex, set):
                                ex = set()
                                frec["example_task_ids"] = ex
                            if str(task_id):
                                ex.add(str(task_id))
                cats_here = {
                    str(c)
                    for c in tags
                    if c
                    in {
                        "instruction",
                        "json",
                        "math",
                        "state",
                        "plan",
                        "clarify",
                        "consistency",
                        "memory",
                        "dialogue",
                        "agency",
                        "concept",
                    }
                }
                for cid in ids:
                    st = concept_used_tags_by_id.get(cid)
                    if st is None:
                        st = set()
                        concept_used_tags_by_id[cid] = st
                    st.update(cats_here)
        except Exception:
            pass

        if ok:
            pass_count += 1
            if isinstance(goal_spec, dict) and goal_spec:
                goals_satisfied += 1
            for cat in ("instruction", "json", "math", "state", "plan", "clarify", "consistency", "memory", "dialogue"):
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
        "concept_total": int(concept_total),
        "concept_used_turns": int(concept_used_turns),
        "concept_ok_turns": int(concept_ok_turns),
        "concept_used_rate": _rate(concept_used_turns, concept_total),
        "concept_pass_count": int(concept_ok_turns),
        "concept_pass_rate": _rate(concept_ok_turns, concept_total),
        "concept_calls_total_sum": int(concept_calls_total_sum),
        "concept_calls_max_depth_mean": float(concept_calls_max_depth_sum / concept_total) if concept_total > 0 else 0.0,
        "concept_calls_max_depth_max": int(concept_calls_max_depth_max),
        "concept_composed_turns": int(concept_composed_turns),
        "concept_composed_rate": _rate(concept_composed_turns, concept_total),
        "concept_deep_turns": int(concept_deep_turns),
        "concept_deep_rate": _rate(concept_deep_turns, concept_total),
        "concept_very_deep_turns": int(concept_very_deep_turns),
        "concept_very_deep_rate": _rate(concept_very_deep_turns, concept_total),
        "concept_csg_nodes_mean": float(concept_csg_nodes_sum / concept_csg_count_turns)
        if concept_csg_count_turns > 0
        else 0.0,
        "concept_csg_edges_mean": float(concept_csg_edges_sum / concept_csg_count_turns)
        if concept_csg_count_turns > 0
        else 0.0,
        "concept_csg_rich_turns": int(concept_csg_rich_turns),
        "concept_csg_rich_rate": _rate(concept_csg_rich_turns, concept_total),
        "concept_csg_required_total": int(concept_csg_required_total),
        "concept_csg_required_ok_turns": int(concept_csg_required_ok_turns),
        "concept_csg_required_pass_rate": _rate(concept_csg_required_ok_turns, concept_csg_required_total),
        "concept_min_depth_required_max": int(concept_min_depth_required_max),
        "concept_nested_call_turns": int(concept_nested_call_turns),
        "concept_nested_call_rate": _rate(concept_nested_call_turns, total_tasks),
        "concept_nested_call_ids_distinct": int(len(concept_nested_call_ids)),
        "concept_any_used_turns": int(concept_any_used_turns),
        "concept_any_ok_turns": int(concept_any_ok_turns),
        "concept_any_used_rate": _rate(concept_any_used_turns, total_tasks),
        "concept_any_ok_rate": _rate(concept_any_ok_turns, total_tasks),
        "concept_policy_required_total": int(concept_policy_required_total),
        "concept_policy_used_turns": int(concept_policy_used_turns),
        "concept_policy_pass_rate": _rate(concept_policy_used_turns, concept_policy_required_total),
        "concept_selected_as_policy_rate": _rate(concept_policy_used_turns, total_tasks),
        "reference_required_total": int(reference_required_total),
        "reference_ok_turns": int(reference_ok_turns),
        "reference_pass_rate": _rate(reference_ok_turns, reference_required_total),
        "instruction_total": int(cat_total.get("instruction", 0)),
        "instruction_pass_count": int(cat_pass.get("instruction", 0)),
        "instruction_pass_rate": _rate(
            int(cat_pass.get("instruction", 0)), int(cat_total.get("instruction", 0))
        ),
        "json_total": int(cat_total.get("json", 0)),
        "json_pass_count": int(cat_pass.get("json", 0)),
        "json_pass_rate": _rate(int(cat_pass.get("json", 0)), int(cat_total.get("json", 0))),
        "math_total": int(cat_total.get("math", 0)),
        "math_pass_count": int(cat_pass.get("math", 0)),
        "math_pass_rate": _rate(int(cat_pass.get("math", 0)), int(cat_total.get("math", 0))),
        "state_total": int(cat_total.get("state", 0)),
        "state_pass_count": int(cat_pass.get("state", 0)),
        "state_pass_rate": _rate(int(cat_pass.get("state", 0)), int(cat_total.get("state", 0))),
        "plan_total": int(cat_total.get("plan", 0)),
        "plan_pass_count": int(cat_pass.get("plan", 0)),
        "plan_pass_rate": _rate(int(cat_pass.get("plan", 0)), int(cat_total.get("plan", 0))),
        "clarify_total": int(cat_total.get("clarify", 0)),
        "clarify_pass_count": int(cat_pass.get("clarify", 0)),
        "clarify_pass_rate": _rate(
            int(cat_pass.get("clarify", 0)), int(cat_total.get("clarify", 0))
        ),
        "consistency_total": int(cat_total.get("consistency", 0)),
        "consistency_pass_count": int(cat_pass.get("consistency", 0)),
        "consistency_pass_rate": _rate(
            int(cat_pass.get("consistency", 0)), int(cat_total.get("consistency", 0))
        ),
        "memory_total": int(cat_total.get("memory", 0)),
        "memory_pass_count": int(cat_pass.get("memory", 0)),
        "memory_pass_rate": _rate(int(cat_pass.get("memory", 0)), int(cat_total.get("memory", 0))),
        "dialogue_total": int(cat_total.get("dialogue", 0)),
        "dialogue_pass_count": int(cat_pass.get("dialogue", 0)),
        "dialogue_pass_rate": _rate(int(cat_pass.get("dialogue", 0)), int(cat_total.get("dialogue", 0))),
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

    # Per-concept usage evidence (deterministic, compact): used by ICS to compute cross-context
    # fitness and perform ablation selection without relying on task_id-specific logic.
    try:
        usage_out: Dict[str, Any] = {}
        for cid in sorted(concept_usage_by_id.keys(), key=str):
            rec = concept_usage_by_id.get(cid) if isinstance(concept_usage_by_id.get(cid), dict) else {}
            fams = rec.get("families") if isinstance(rec.get("families"), dict) else {}
            fams_out: Dict[str, Any] = {}
            for fid in sorted(fams.keys(), key=str):
                frec = fams.get(fid) if isinstance(fams.get(fid), dict) else {}
                ex = frec.get("example_task_ids")
                if isinstance(ex, set):
                    ex_list = sorted([str(x) for x in ex if str(x)], key=str)[:8]
                elif isinstance(ex, list):
                    ex_list = sorted([str(x) for x in ex if str(x)], key=str)[:8]
                else:
                    ex_list = []
                fams_out[str(fid)] = {
                    "used_turns": int(frec.get("used_turns", 0) or 0),
                    "ok_turns": int(frec.get("ok_turns", 0) or 0),
                    "example_task_ids": list(ex_list),
                }
            usage_out[str(cid)] = {
                "used_turns": int(rec.get("used_turns", 0) or 0),
                "ok_turns": int(rec.get("ok_turns", 0) or 0),
                "calls_max_depth_max": int(rec.get("calls_max_depth_max", 0) or 0),
                "families": dict(fams_out),
            }
        if usage_out:
            metrics["concept_usage_by_id"] = dict(usage_out)
    except Exception:
        pass

    # Cross-tag semantic reuse: at least one concept should be exercised across different
    # utility categories (e.g., instruction+json, json+math) in the same run.
    cross_tag = [
        (cid, sorted(list(tags0))) for cid, tags0 in concept_used_tags_by_id.items() if len(tags0) >= 2
    ]

    def _is_seed_concept(cid: str) -> bool:
        c = str(cid or "")
        if c.startswith("concept_seed_") or c.startswith("goal_seed_"):
            return True
        if c.startswith("act_s000000_concept_"):
            return True
        try:
            act0 = engine.store.get_concept_act(c)
        except Exception:
            act0 = None
        if act0 is None or not isinstance(getattr(act0, "evidence", None), dict):
            return False
        name0 = str(act0.evidence.get("name") or "")
        return name0 in {"concept_seed_v0"}

    cross_tag.sort(key=lambda kv: (_is_seed_concept(str(kv[0])), -len(kv[1]), str(kv[0])))
    metrics["concept_cross_tag_reuse_count"] = int(len(cross_tag))
    if cross_tag:
        metrics["concept_cross_tag_reuse_example"] = {
            "concept_id": str(cross_tag[0][0]),
            "used_tags": list(cross_tag[0][1]),
        }

    # Cross-context reuse (birth vs usage): count concepts whose declared birth_tags are a strict
    # subset of the categories where they end up being used.
    cross_ctx = []
    for cid, used_tags in concept_used_tags_by_id.items():
        try:
            act = engine.store.get_concept_act(str(cid))
        except Exception:
            act = None
        if act is None or not isinstance(getattr(act, "evidence", None), dict):
            continue
        ev = act.evidence
        meta0 = ev.get("meta") if isinstance(ev.get("meta"), dict) else {}
        bt = meta0.get("birth_tags") if isinstance(meta0, dict) else None
        if not isinstance(bt, list) or not bt:
            continue
        birth_tags = {str(x) for x in bt if str(x)}
        extra = sorted([t for t in used_tags if t not in birth_tags])
        if extra:
            cross_ctx.append((cid, sorted(list(birth_tags)), sorted(list(used_tags)), extra))
    cross_ctx.sort(key=lambda t: (_is_seed_concept(str(t[0])), -len(t[3]), str(t[0])))
    metrics["concept_cross_context_reuse_count"] = int(len(cross_ctx))
    if cross_ctx:
        metrics["concept_cross_context_reuse_example"] = {
            "concept_id": str(cross_ctx[0][0]),
            "birth_tags": list(cross_ctx[0][1]),
            "used_tags": list(cross_ctx[0][2]),
            "extra_used_tags": list(cross_ctx[0][3]),
        }

    # Static concept depth (stored hierarchy): depth=0 is leaf (no CSV_CALL),
    # depth>=1 means calls other concepts, etc. This is independent of dynamic execution depth.
    try:
        memo_depth: Dict[str, int] = {}

        def _static_depth(concept_id: str, stack: set) -> int:
            cid = str(concept_id or "")
            if not cid:
                return 0
            if cid in memo_depth:
                return int(memo_depth[cid])
            if cid in stack:
                memo_depth[cid] = 0
                return 0
            try:
                act0 = engine.store.get_concept_act(cid)
            except Exception:
                act0 = None
            if act0 is None:
                memo_depth[cid] = 0
                return 0
            callees: List[str] = []
            for ins in list(getattr(act0, "program", []) or []):
                if str(getattr(ins, "op", "")) != "CSV_CALL":
                    continue
                args0 = getattr(ins, "args", {}) or {}
                if not isinstance(args0, dict):
                    args0 = {}
                callee = str(args0.get("concept_id") or "")
                if callee:
                    callees.append(callee)
            if not callees:
                d0 = 0
            else:
                st2 = set(stack)
                st2.add(cid)
                d0 = 1 + max(_static_depth(c, st2) for c in callees)
            memo_depth[cid] = int(d0)
            return int(d0)

        depths: List[int] = []
        for a in getattr(engine.store, "concept_acts", lambda: [])():
            depths.append(int(_static_depth(str(getattr(a, "id", "") or ""), set())))
        depths.sort()
        metrics["concept_static_depth_max"] = int(depths[-1]) if depths else 0
        metrics["concept_static_depth_median"] = int(depths[len(depths) // 2]) if depths else 0
        metrics["concept_static_depth_ge2_count"] = int(sum(1 for d in depths if int(d) >= 2))
        metrics["concept_static_depth_ge3_count"] = int(sum(1 for d in depths if int(d) >= 3))
    except Exception:
        pass
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
