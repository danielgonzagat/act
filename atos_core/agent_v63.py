from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from .concepts import PRIMITIVE_OPS


def stable_act_id(prefix: str, body: Dict[str, Any]) -> str:
    return f"{prefix}{sha256_hex(canonical_json_dumps(body).encode('utf-8'))[:12]}"


def make_concept_act(
    *,
    step: int,
    store_hash_excl_semantic: str,
    title: str,
    program: Sequence[Instruction],
    interface: Dict[str, Any],
    overhead_bits: int = 1024,
    meta: Optional[Dict[str, Any]] = None,
) -> Act:
    ev = {
        "name": "concept_csv_v0",
        "interface": dict(interface),
        "meta": {
            "title": str(title),
            "trained_on_store_content_hash": str(store_hash_excl_semantic),
            **(dict(meta or {})),
        },
    }
    body = {
        "kind": "concept_csv",
        "version": 1,
        "match": {},
        "program": [ins.to_dict() for ins in program],
        "evidence": ev,
        "deps": [],
        "active": True,
    }
    act_id = stable_act_id("act_concept_csv_", body)
    return Act(
        id=act_id,
        version=1,
        created_at=deterministic_iso(step=int(step)),
        kind="concept_csv",
        match={},
        program=list(program),
        evidence=ev,
        cost={"overhead_bits": int(overhead_bits)},
        deps=[],
        active=True,
    )


def make_goal_act(
    *,
    step: int,
    store_hash_excl_semantic: str,
    title: str,
    concept_id: str,
    inputs: Dict[str, Any],
    expected: Any,
    priority: int = 10,
    overhead_bits: int = 1024,
) -> Act:
    ev = {
        "name": "goal_v0",
        "meta": {
            "title": str(title),
            "trained_on_store_content_hash": str(store_hash_excl_semantic),
        },
        "goal": {
            "priority": int(priority),
            "concept_id": str(concept_id),
            "inputs": dict(inputs),
            "expected": expected,
        },
    }
    body = {
        "kind": "goal",
        "version": 1,
        "match": {},
        "program": [],
        "evidence": ev,
        "deps": [],
        "active": True,
    }
    act_id = stable_act_id("act_goal_", body)
    return Act(
        id=act_id,
        version=1,
        created_at=deterministic_iso(step=int(step)),
        kind="goal",
        match={},
        program=[],
        evidence=ev,
        cost={"overhead_bits": int(overhead_bits)},
        deps=[],
        active=True,
    )


def _op(op_id: str) -> Any:
    spec_fn = PRIMITIVE_OPS.get(str(op_id))
    if spec_fn is None:
        raise KeyError(f"unknown_primitive:{op_id}")
    return spec_fn[1]


def extract_int(text: str, *, strip_one_zero: bool) -> int:
    scan = _op("scan_digits")
    strip0 = _op("strip_one_leading_zero")
    d2i = _op("digits_to_int")
    digits = scan(str(text))
    if bool(strip_one_zero):
        digits = strip0(digits)
    return int(d2i(digits))


def build_v63_toolbox(
    *,
    step: int,
    store_hash_excl_semantic: str,
    overhead_bits: int = 1024,
) -> Dict[str, Act]:
    """
    Deterministic toolbox of small concept_csv programs used by v63 agent loop.
    """
    # extract_int: scan_digits -> digits_to_int
    c_extract = make_concept_act(
        step=step,
        store_hash_excl_semantic=store_hash_excl_semantic,
        title="v63_extract_int",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "text", "out": "t"}),
            Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["t"], "out": "d"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "n"}),
            Instruction("CSV_RETURN", {"var": "n"}),
        ],
        interface={"input_schema": {"text": "str"}, "output_schema": {"value": "int"}, "validator_id": "int_value_exact"},
        overhead_bits=overhead_bits,
        meta={"builder": "agent_v63_toolbox"},
    )

    # extract_int_strip0: scan_digits -> strip_one_leading_zero -> digits_to_int
    c_extract_strip0 = make_concept_act(
        step=step + 1,
        store_hash_excl_semantic=store_hash_excl_semantic,
        title="v63_extract_int_strip0",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "text", "out": "t"}),
            Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["t"], "out": "d0"}),
            Instruction("CSV_PRIMITIVE", {"fn": "strip_one_leading_zero", "in": ["d0"], "out": "d"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "n"}),
            Instruction("CSV_RETURN", {"var": "n"}),
        ],
        interface={"input_schema": {"text": "str"}, "output_schema": {"value": "int"}, "validator_id": "int_value_exact"},
        overhead_bits=overhead_bits,
        meta={"builder": "agent_v63_toolbox"},
    )

    # add_int: add_int(a,b) -> int
    c_add = make_concept_act(
        step=step + 2,
        store_hash_excl_semantic=store_hash_excl_semantic,
        title="v63_add_int",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "a", "out": "a0"}),
            Instruction("CSV_GET_INPUT", {"name": "b", "out": "b0"}),
            Instruction("CSV_PRIMITIVE", {"fn": "add_int", "in": ["a0", "b0"], "out": "s"}),
            Instruction("CSV_RETURN", {"var": "s"}),
        ],
        interface={"input_schema": {"a": "int", "b": "int"}, "output_schema": {"value": "int"}, "validator_id": "int_value_exact"},
        overhead_bits=overhead_bits,
        meta={"builder": "agent_v63_toolbox"},
    )

    # json_ab: make_dict_ab(a,b) -> json_canonical(dict) -> str
    c_json_ab = make_concept_act(
        step=step + 3,
        store_hash_excl_semantic=store_hash_excl_semantic,
        title="v63_json_ab",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "a", "out": "a0"}),
            Instruction("CSV_GET_INPUT", {"name": "b", "out": "b0"}),
            Instruction("CSV_PRIMITIVE", {"fn": "make_dict_ab", "in": ["a0", "b0"], "out": "obj"}),
            Instruction("CSV_PRIMITIVE", {"fn": "json_canonical", "in": ["obj"], "out": "s"}),
            Instruction("CSV_RETURN", {"var": "s"}),
        ],
        interface={"input_schema": {"a": "int", "b": "int"}, "output_schema": {"value": "str"}, "validator_id": "json_ab_int_exact"},
        overhead_bits=overhead_bits,
        meta={"builder": "agent_v63_toolbox"},
    )

    return {
        "extract_int": c_extract,
        "extract_int_strip0": c_extract_strip0,
        "add_int": c_add,
        "json_ab": c_json_ab,
    }


@dataclass(frozen=True)
class V63Task:
    task_id: str
    category: str
    prompt_text: str
    kind: str
    args: Dict[str, Any]


def build_v63_tasks() -> List[V63Task]:
    tasks: List[V63Task] = []

    # A) Parsing/extraction/normalization (10)
    parse_texts = [
        "abc0123",
        "x9y7",
        "id=42",
        "ref0005Z",
        "A=0000",
        "user_77_end",
        "n=100",
        "t0099",
        "k=5",
        "zzz123zzz",
    ]
    for i, text in enumerate(parse_texts):
        strip0 = bool(i % 2 == 0)
        tasks.append(
            V63Task(
                task_id=f"v63_parse_{i:02d}",
                category="parse",
                prompt_text=f'Extraia o primeiro inteiro da string "{text}" e responda apenas o número.',
                kind="extract_int",
                args={"text": str(text), "strip0": bool(strip0)},
            )
        )

    # B) JSON (10)
    json_pairs: List[Tuple[int, int]] = [
        (17, 25),
        (9, 7),
        (1, 0),
        (42, 63),
        (5, 5),
        (3, 14),
        (99, 1),
        (8, 13),
        (0, 11),
        (12, 30),
    ]
    for i, (a, b) in enumerate(json_pairs):
        tasks.append(
            V63Task(
                task_id=f"v63_json_{i:02d}",
                category="json",
                prompt_text=f"Retorne um JSON canônico apenas com chaves a,b inteiros: a={a}, b={b}.",
                kind="json_ab",
                args={"a": int(a), "b": int(b)},
            )
        )

    # C) Arithmetic multi-step (7)
    sums: List[Tuple[str, str]] = [
        ("a=1", "b=2"),
        ("x09y", "z7"),
        ("id=10", "k=5"),
        ("m=0003", "n=0004"),
        ("p=12", "q=30"),
        ("u=99", "v=1"),
        ("foo42bar", "baz63"),
    ]
    for i, (ta, tb) in enumerate(sums):
        tasks.append(
            V63Task(
                task_id=f"v63_math_{i:02d}",
                category="math",
                prompt_text=f'Some os inteiros extraídos de "{ta}" e "{tb}" e responda apenas o número.',
                kind="sum_two_texts",
                args={"text_a": str(ta), "text_b": str(tb), "strip0_a": True, "strip0_b": False},
            )
        )

    # D) Planning explicit (3) — extract -> calc -> serialize
    plans: List[Tuple[str, str]] = [
        ("A=0002", "B=40"),
        ("x9", "y7"),
        ("foo17", "bar25"),
    ]
    for i, (ta, tb) in enumerate(plans):
        tasks.append(
            V63Task(
                task_id=f"v63_plan_{i:02d}",
                category="plan",
                prompt_text=(
                    f'Plano: extraia A de "{ta}", extraia B de "{tb}", compute S=A+B e responda JSON {{"a":S,"b":B}}.'
                ),
                kind="plan_json_sum",
                args={"text_a": str(ta), "text_b": str(tb), "strip0_a": True, "strip0_b": False},
            )
        )

    assert len(tasks) >= 30
    return tasks


def plan_task_steps(task: V63Task) -> List[Dict[str, Any]]:
    """
    Deterministic planner: returns a list of step specs, each with:
      - step_name
      - concept_key
      - inputs
      - expected (for validator)
      - expected_output_text
    """
    kind = str(task.kind)
    a: Any
    b: Any
    steps: List[Dict[str, Any]] = []

    if kind == "extract_int":
        text = str(task.args.get("text") or "")
        strip0 = bool(task.args.get("strip0", False))
        n = extract_int(text, strip_one_zero=strip0)
        steps.append(
            {
                "step_name": "extract_int",
                "concept_key": "extract_int_strip0" if strip0 else "extract_int",
                "inputs": {"text": str(text)},
                "expected": int(n),
                "expected_output_text": str(int(n)),
            }
        )
        return steps

    if kind == "json_ab":
        a = int(task.args.get("a", 0) or 0)
        b = int(task.args.get("b", 0) or 0)
        expected = {"a": int(a), "b": int(b)}
        steps.append(
            {
                "step_name": "json_ab",
                "concept_key": "json_ab",
                "inputs": {"a": int(a), "b": int(b)},
                "expected": dict(expected),
                "expected_output_text": canonical_json_dumps(expected),
            }
        )
        return steps

    if kind == "sum_two_texts":
        ta = str(task.args.get("text_a") or "")
        tb = str(task.args.get("text_b") or "")
        strip0_a = bool(task.args.get("strip0_a", False))
        strip0_b = bool(task.args.get("strip0_b", False))
        a = extract_int(ta, strip_one_zero=strip0_a)
        b = extract_int(tb, strip_one_zero=strip0_b)
        s = int(a) + int(b)
        steps.extend(
            [
                {
                    "step_name": "extract_a",
                    "concept_key": "extract_int_strip0" if strip0_a else "extract_int",
                    "inputs": {"text": str(ta)},
                    "expected": int(a),
                    "expected_output_text": str(int(a)),
                },
                {
                    "step_name": "extract_b",
                    "concept_key": "extract_int_strip0" if strip0_b else "extract_int",
                    "inputs": {"text": str(tb)},
                    "expected": int(b),
                    "expected_output_text": str(int(b)),
                },
                {
                    "step_name": "add",
                    "concept_key": "add_int",
                    "inputs": {"a": int(a), "b": int(b)},
                    "expected": int(s),
                    "expected_output_text": str(int(s)),
                },
            ]
        )
        return steps

    if kind == "plan_json_sum":
        ta = str(task.args.get("text_a") or "")
        tb = str(task.args.get("text_b") or "")
        strip0_a = bool(task.args.get("strip0_a", False))
        strip0_b = bool(task.args.get("strip0_b", False))
        a = extract_int(ta, strip_one_zero=strip0_a)
        b = extract_int(tb, strip_one_zero=strip0_b)
        s = int(a) + int(b)
        expected_json = {"a": int(s), "b": int(b)}
        steps.extend(
            [
                {
                    "step_name": "extract_a",
                    "concept_key": "extract_int_strip0" if strip0_a else "extract_int",
                    "inputs": {"text": str(ta)},
                    "expected": int(a),
                    "expected_output_text": str(int(a)),
                },
                {
                    "step_name": "extract_b",
                    "concept_key": "extract_int_strip0" if strip0_b else "extract_int",
                    "inputs": {"text": str(tb)},
                    "expected": int(b),
                    "expected_output_text": str(int(b)),
                },
                {
                    "step_name": "add",
                    "concept_key": "add_int",
                    "inputs": {"a": int(a), "b": int(b)},
                    "expected": int(s),
                    "expected_output_text": str(int(s)),
                },
                {
                    "step_name": "json",
                    "concept_key": "json_ab",
                    "inputs": {"a": int(s), "b": int(b)},
                    "expected": dict(expected_json),
                    "expected_output_text": canonical_json_dumps(expected_json),
                },
            ]
        )
        return steps

    raise ValueError(f"unknown_task_kind:{kind}")

