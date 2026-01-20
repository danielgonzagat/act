from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .act import canonical_json_dumps
from .agent_v63 import V63Task, build_v63_tasks, build_v63_toolbox, extract_int, make_concept_act
from .act import Instruction


def build_v65_toolbox(
    *,
    step: int,
    store_hash_excl_semantic: str,
    overhead_bits: int = 1024,
) -> Dict[str, Any]:
    """
    Toolbox v65 = v63 toolbox + deterministic string primitives concepts.
    """
    tb = dict(build_v63_toolbox(step=step, store_hash_excl_semantic=store_hash_excl_semantic, overhead_bits=overhead_bits))

    # concat_str: str_concat(a,b) -> str
    tb["concat_str"] = make_concept_act(
        step=step + 10,
        store_hash_excl_semantic=store_hash_excl_semantic,
        title="v65_concat_str",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "a", "out": "a0"}),
            Instruction("CSV_GET_INPUT", {"name": "b", "out": "b0"}),
            Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["a0", "b0"], "out": "s"}),
            Instruction("CSV_RETURN", {"var": "s"}),
        ],
        interface={"input_schema": {"a": "str", "b": "str"}, "output_schema": {"value": "str"}, "validator_id": "text_exact"},
        overhead_bits=overhead_bits,
        meta={"builder": "agent_v65_toolbox"},
    )

    # int_to_str: int_to_str(n) -> str
    tb["int_to_str"] = make_concept_act(
        step=step + 11,
        store_hash_excl_semantic=store_hash_excl_semantic,
        title="v65_int_to_str",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "n", "out": "n0"}),
            Instruction("CSV_PRIMITIVE", {"fn": "int_to_str", "in": ["n0"], "out": "s"}),
            Instruction("CSV_RETURN", {"var": "s"}),
        ],
        interface={"input_schema": {"n": "int"}, "output_schema": {"value": "str"}, "validator_id": "text_exact"},
        overhead_bits=overhead_bits,
        meta={"builder": "agent_v65_toolbox"},
    )

    # concat3_str: str_concat(str_concat(a,b),c) -> str (mineable: 2 primitives)
    tb["concat3_str"] = make_concept_act(
        step=step + 12,
        store_hash_excl_semantic=store_hash_excl_semantic,
        title="v65_concat3_str",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "a", "out": "a0"}),
            Instruction("CSV_GET_INPUT", {"name": "b", "out": "b0"}),
            Instruction("CSV_GET_INPUT", {"name": "c", "out": "c0"}),
            Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["a0", "b0"], "out": "t"}),
            Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["t", "c0"], "out": "s"}),
            Instruction("CSV_RETURN", {"var": "s"}),
        ],
        interface={
            "input_schema": {"a": "str", "b": "str", "c": "str"},
            "output_schema": {"value": "str"},
            "validator_id": "text_exact",
        },
        overhead_bits=overhead_bits,
        meta={"builder": "agent_v65_toolbox"},
    )

    # format_prefix_int_suffix: prefix + int_to_str(n) + suffix (mineable: 3 primitives)
    tb["format_prefix_int_suffix"] = make_concept_act(
        step=step + 13,
        store_hash_excl_semantic=store_hash_excl_semantic,
        title="v65_format_prefix_int_suffix",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "prefix", "out": "p"}),
            Instruction("CSV_GET_INPUT", {"name": "n", "out": "n0"}),
            Instruction("CSV_GET_INPUT", {"name": "suffix", "out": "s0"}),
            Instruction("CSV_PRIMITIVE", {"fn": "int_to_str", "in": ["n0"], "out": "ns"}),
            Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["p", "ns"], "out": "t"}),
            Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["t", "s0"], "out": "out"}),
            Instruction("CSV_RETURN", {"var": "out"}),
        ],
        interface={
            "input_schema": {"prefix": "str", "n": "int", "suffix": "str"},
            "output_schema": {"value": "str"},
            "validator_id": "text_exact",
        },
        overhead_bits=overhead_bits,
        meta={"builder": "agent_v65_toolbox"},
    )

    return tb


@dataclass(frozen=True)
class V65Task:
    task_id: str
    category: str
    prompt_text: str
    kind: str
    args: Dict[str, Any]


def build_v65_tasks() -> List[V65Task]:
    """
    V65 tasks = v63 tasks + deterministic dialog/text tasks.
    Domains:
      A: parse, json, dialog
      B: math, plan
    """
    tasks: List[V65Task] = []

    # Start with v63 tasks (30).
    for t in build_v63_tasks():
        tasks.append(
            V65Task(
                task_id=str(t.task_id),
                category=str(t.category),
                prompt_text=str(t.prompt_text),
                kind=str(t.kind),
                args=dict(t.args),
            )
        )

    # Add dialog tasks (>=10): greet + format-int.
    greets: List[Tuple[str, str, str]] = [
        ("Olá, ", "Daniel", "!"),
        ("Olá, ", "ACT", "!"),
        ("Oi ", "PATO", "."),
        ("Bom dia, ", "Joana", "!"),
        ("Boa noite, ", "Mundo", "."),
        ("Saudações, ", "AGENTE", "!"),
        ("Ei, ", "VOCÊ", "!"),
        ("Olá, ", "Kairos", "."),
        ("Olá, ", "CSV", "!"),
        ("Olá, ", "ToC", "!"),
    ]
    for i, (prefix, name, suffix) in enumerate(greets):
        tasks.append(
            V65Task(
                task_id=f"v65_dialog_greet_{i:02d}",
                category="dialog",
                prompt_text="Construa uma saudação determinística por concatenação.",
                kind="dialog_greet",
                args={"prefix": str(prefix), "name": str(name), "suffix": str(suffix)},
            )
        )

    formats: List[Tuple[str, int, str]] = [
        ("Valor=", 7, "."),
        ("Valor=", 42, "."),
        ("N=", 0, ";"),
        ("Total=", 12, "."),
        ("S=", 99, "!"),
        ("X=", 5, "."),
        ("ID=", 123, "."),
        ("Z=", 100, "."),
        ("A=", 3, "."),
        ("B=", 14, "."),
    ]
    for i, (prefix, n, suffix) in enumerate(formats):
        tasks.append(
            V65Task(
                task_id=f"v65_dialog_format_int_{i:02d}",
                category="dialog",
                prompt_text="Formate prefix + inteiro + sufixo (texto exato).",
                kind="dialog_format_int",
                args={"prefix": str(prefix), "n": int(n), "suffix": str(suffix)},
            )
        )

    # Domain B: plan tasks that still use concat3_str (string assembly) to satisfy ToC.
    plan_concat: List[Tuple[str, str, str]] = [
        ("ID:", "ABC", "."),
        ("K=", "XYZ", ";"),
        ("TAG#", "PATO", "!"),
        ("REF:", "R2D2", "."),
        ("SLOT=", "A1", "."),
        ("SRC:", "NEWS", "."),
    ]
    for i, (a, b, c) in enumerate(plan_concat):
        tasks.append(
            V65Task(
                task_id=f"v65_plan_concat3_{i:02d}",
                category="plan",
                prompt_text="Plano: montar uma string por concatenação de 3 partes (texto exato).",
                kind="dialog_greet",
                args={"prefix": str(a), "name": str(b), "suffix": str(c)},
            )
        )

    # Domain B: math tasks with final textual output (forces int_to_str + concat/format in B).
    # Keep inputs small to avoid combinatorial explosion in planner bindings for str_concat.
    sums: List[Tuple[int, int, str, str]] = [
        (4, 8, "A soma é ", "."),
        (9, 7, "A soma é ", "."),
        (17, 25, "A soma é ", "."),
        (12, 30, "A soma é ", "."),
        (99, 1, "A soma é ", "."),
        (3, 4, "A soma é ", "."),
    ]
    for i, (a, b, prefix, suffix) in enumerate(sums):
        tasks.append(
            V65Task(
                task_id=f"v65_math_sum_sentence_{i:02d}",
                category="math",
                prompt_text="Some dois inteiros e retorne uma frase determinística (texto exato).",
                kind="sum_two_ints_sentence",
                args={"a": int(a), "b": int(b), "prefix": str(prefix), "suffix": str(suffix)},
            )
        )

    assert len(tasks) >= 40
    return tasks
