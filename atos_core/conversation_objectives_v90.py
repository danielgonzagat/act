from __future__ import annotations

from typing import Any, Dict

from .act import Act, Instruction, deterministic_iso
from .objective_v88 import OBJECTIVE_KIND_V88, ensure_objective_primitives_registered_v88


COMM_OBJECTIVES_V90 = (
    "COMM_RESPOND",
    "COMM_ASK_CLARIFY",
    "COMM_CONFIRM",
    "COMM_CORRECT",
    "COMM_SUMMARIZE",
    "COMM_ADMIT_UNKNOWN",
    "COMM_END",
)


def make_comm_objective_eq_text_v90(*, objective_id: str, objective_kind: str, created_step: int = 0) -> Act:
    """
    Deterministic communicative objective: verify that __output == expected.
    Returns ObjectiveVerdict-shaped dict via objective_eq_text_v88 primitive.
    """
    ensure_objective_primitives_registered_v88()
    prog = [
        Instruction("CSV_GET_INPUT", {"name": "__output", "out": "__output"}),
        Instruction("CSV_GET_INPUT", {"name": "expected", "out": "expected"}),
        Instruction("CSV_PRIMITIVE", {"fn": "objective_eq_text_v88", "in": ["__output", "expected"], "out": "verdict"}),
        Instruction("CSV_RETURN", {"var": "verdict"}),
    ]
    iface = {
        "input_schema": {"__output": "str", "expected": "str"},
        "output_schema": {"verdict": "dict"},
        "validator_id": "",
    }
    return Act(
        id=str(objective_id),
        version=1,
        created_at=deterministic_iso(step=int(created_step)),
        kind=OBJECTIVE_KIND_V88,  # type: ignore[assignment]
        match={},
        program=list(prog),
        evidence={
            "interface": dict(iface),
            "objective_v90": {"schema_version": 1, "objective_kind": str(objective_kind)},
        },
        cost={},
        deps=[],
        active=True,
    )


def comm_objective_ids_v90() -> Dict[str, str]:
    """
    Stable mapping objective_kind -> objective_act_id.
    """
    return {k: f"objective_v90_{k.lower()}" for k in COMM_OBJECTIVES_V90}

