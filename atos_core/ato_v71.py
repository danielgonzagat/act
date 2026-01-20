from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex

ATO_TYPES: Tuple[str, ...] = (
    "OBS",
    "STATE",
    "CONCEPT",
    "GOAL",
    "PLAN",
    "OPERATOR",
    "EVAL",
)


def ensure_ato_type(ato_type: str) -> str:
    t = str(ato_type or "")
    if t not in set(ATO_TYPES):
        raise ValueError(f"unknown_ato_type:{t}")
    return t


def stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def ato_struct_sig(ato_sem_sig: Dict[str, Any]) -> str:
    """
    Optional structural signature to freeze identity-relevant structure.
    Deterministic: sha256(canonical_json(structural_view)).
    """
    body = {
        "ato_type": str(ato_sem_sig.get("ato_type") or ""),
        "subgraph": ato_sem_sig.get("subgraph") if isinstance(ato_sem_sig.get("subgraph"), dict) else {},
        "slots": ato_sem_sig.get("slots") if isinstance(ato_sem_sig.get("slots"), dict) else {},
        "invariants": ato_sem_sig.get("invariants") if isinstance(ato_sem_sig.get("invariants"), dict) else {},
    }
    return stable_hash_obj(body)


def ato_sig(ato_sem_sig: Dict[str, Any]) -> str:
    return stable_hash_obj(ato_sem_sig)


def _sorted_evidence_refs(evidence_refs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Tuple[str, Dict[str, Any]]] = []
    for r in evidence_refs:
        if not isinstance(r, dict):
            continue
        try:
            key = canonical_json_dumps(r)
        except Exception:
            key = str(r)
        items.append((key, dict(r)))
    items.sort(key=lambda kv: str(kv[0]))
    return [v for _, v in items]


@dataclass(frozen=True)
class ATOv71:
    ato_id: str
    ato_type: str
    subgraph: Dict[str, Any] = field(default_factory=dict)
    slots: Dict[str, Any] = field(default_factory=dict)
    bindings: Dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    invariants: Dict[str, Any] = field(default_factory=dict)
    created_step: int = 0
    last_step: int = 0

    def __post_init__(self) -> None:
        ensure_ato_type(self.ato_type)
        object.__setattr__(self, "ato_id", str(self.ato_id or ""))
        object.__setattr__(self, "ato_type", str(self.ato_type or ""))
        object.__setattr__(self, "created_step", int(self.created_step or 0))
        object.__setattr__(self, "last_step", int(self.last_step or 0))
        try:
            object.__setattr__(self, "cost", float(self.cost or 0.0))
        except Exception:
            object.__setattr__(self, "cost", 0.0)

        # Deep-copy snapshots to avoid aliasing.
        for key in ("subgraph", "slots", "bindings", "invariants"):
            v = getattr(self, key)
            if not isinstance(v, dict):
                v = {}
            try:
                v2 = copy.deepcopy(v)
            except Exception:
                v2 = dict(v)
            object.__setattr__(self, key, v2 if isinstance(v2, dict) else {})

        ev = self.evidence_refs
        if not isinstance(ev, list):
            ev = []
        try:
            ev2 = copy.deepcopy(ev)
        except Exception:
            ev2 = list(ev)
        if not isinstance(ev2, list):
            ev2 = []
        ev3 = _sorted_evidence_refs([x for x in ev2 if isinstance(x, dict)])
        object.__setattr__(self, "evidence_refs", ev3)

    def to_dict(self, *, include_sig: bool = True, include_struct_sig: bool = False) -> Dict[str, Any]:
        body = {
            "ato_id": str(self.ato_id),
            "ato_type": str(self.ato_type),
            "subgraph": dict(self.subgraph),
            "slots": dict(self.slots),
            "bindings": dict(self.bindings),
            "cost": float(self.cost),
            "evidence_refs": list(self.evidence_refs),
            "invariants": dict(self.invariants),
            "created_step": int(self.created_step),
            "last_step": int(self.last_step),
        }
        if include_struct_sig:
            body["ato_struct_sig"] = ato_struct_sig(body)
        if include_sig:
            body["ato_sig"] = ato_sig(body)
        return body

