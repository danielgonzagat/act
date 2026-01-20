from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .act import Instruction
from .csg_v87 import (
    CsvCsgLogsV87,
    append_chained_jsonl_v87,
    build_csg_concept_def_v87,
    canonicalize_csg_v87,
    csg_expand_v87,
    csg_hash_v87,
    csg_to_concept_program_v87,
    estimate_cost_v87,
    verify_chained_jsonl_v87,
)
from .store import ActStore


def canonicalize_csg_v130(csg: Dict[str, Any]) -> Dict[str, Any]:
    return canonicalize_csg_v87(csg)


def csg_hash_v130(csg: Dict[str, Any]) -> str:
    return csg_hash_v87(csg)


def estimate_cost_v130(csg: Dict[str, Any], store: ActStore) -> Dict[str, Any]:
    return estimate_cost_v87(csg, store)


def csg_expand_v130(csg: Dict[str, Any], store: ActStore) -> List[Dict[str, Any]]:
    return csg_expand_v87(csg, store)


def csg_to_concept_program_v130(csg: Dict[str, Any]) -> List[Instruction]:
    return csg_to_concept_program_v87(csg)


def append_chained_jsonl_v130(path: str, entry: Dict[str, Any], *, prev_hash: Optional[str]) -> str:
    return append_chained_jsonl_v87(path, entry, prev_hash=prev_hash)


def verify_chained_jsonl_v130(path: str) -> bool:
    return verify_chained_jsonl_v87(path)


CsvCsgLogsV130 = CsvCsgLogsV87


def build_csg_concept_def_v130(*, csg: Dict[str, Any], store: ActStore) -> Dict[str, Any]:
    # Versioned concept_id prefix to avoid collisions with earlier baselines.
    return build_csg_concept_def_v87(csg=csg, store=store, concept_id_prefix="concept_csv_v130_")
