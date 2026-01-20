from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

ActKind = Literal[
    "predictor",
    "compressor",
    "selector",
    "rewrite_rule",
    "metric",
    "candidate_source",
    "mode_selector",
    "mode_policy",
    "memory_facts",
    "gate_table_ctxsig",
    "concept_csv",
    "goal",
]


def utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat()


_EPOCH = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)


def deterministic_iso(*, step: int, offset_us: int = 0) -> str:
    dt = _EPOCH + _dt.timedelta(seconds=int(step), microseconds=int(offset_us))
    return dt.isoformat()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Instruction:
    op: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        if not self.args:
            return {"op": self.op}
        return {"op": self.op, **self.args}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Instruction":
        d = dict(d)
        op = d.pop("op")
        return Instruction(op=op, args=d)


@dataclass
class Act:
    id: str
    version: int
    created_at: str
    kind: ActKind
    match: Dict[str, Any]
    program: List[Instruction]
    evidence: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    deps: List[str] = field(default_factory=list)
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "created_at": self.created_at,
            "kind": self.kind,
            "match": self.match,
            "program": [ins.to_dict() for ins in self.program],
            "evidence": self.evidence,
            "cost": self.cost,
            "deps": self.deps,
            "active": self.active,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Act":
        return Act(
            id=d["id"],
            version=int(d.get("version", 1)),
            created_at=d.get("created_at", utc_now_iso()),
            kind=d["kind"],
            match=dict(d.get("match", {})),
            program=[Instruction.from_dict(x) for x in d.get("program", [])],
            evidence=dict(d.get("evidence", {})),
            cost=dict(d.get("cost", {})),
            deps=list(d.get("deps", [])),
            active=bool(d.get("active", True)),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_dumps(self.to_dict()).encode("utf-8")

    def content_hash(self) -> str:
        return sha256_hex(self.canonical_bytes())


@dataclass(frozen=True)
class Patch:
    kind: Literal["ADD_ACT", "PRUNE_ACT", "MERGE_ACTS", "REWRITE_ACT"]
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "payload": self.payload}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Patch":
        return Patch(kind=d["kind"], payload=dict(d.get("payload", {})))


def estimate_table_cost_bits(
    table: Dict[str, Dict[str, int]],
    *,
    ctx_cost_bits: int = 16,
    edge_cost_bits: int = 8,
) -> int:
    contexts = len(table)
    edges = 0
    for nxt in table.values():
        edges += len(nxt)
    return contexts * ctx_cost_bits + edges * edge_cost_bits


def estimate_act_cost_bits(act: Act) -> int:
    overhead_bits = int(act.cost.get("overhead_bits", 1024))
    if act.kind == "predictor":
        table = act.evidence.get("table")
        if isinstance(table, dict):
            table_cost_bits = estimate_table_cost_bits(
                table,
                ctx_cost_bits=int(act.cost.get("ctx_cost_bits", 16)),
                edge_cost_bits=int(act.cost.get("edge_cost_bits", 8)),
            )
        else:
            table_cost_bits = 0
        return overhead_bits + table_cost_bits
    return overhead_bits
