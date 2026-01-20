from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .ato_v71 import ATOv71, ensure_ato_type, stable_hash_obj

EDGE_TYPES: Tuple[str, ...] = (
    "SUPPORTS",
    "DEPENDS_ON",
    "CALLS",
    "DERIVED_FROM",
    "USED_BY",
    "CAUSES",
)


def ensure_edge_type(edge_type: str) -> str:
    t = str(edge_type or "")
    if t not in set(EDGE_TYPES):
        raise ValueError(f"unknown_edge_type:{t}")
    return t


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(canonical_json_dumps(row))
        f.write("\n")


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return iter(())
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_chained_jsonl(path: str, entry: Dict[str, Any], *, prev_hash: Optional[str]) -> str:
    body = dict(entry)
    body["prev_hash"] = prev_hash
    entry_hash = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    body["entry_hash"] = entry_hash
    _append_jsonl(path, body)
    return entry_hash


def verify_chained_jsonl(path: str) -> bool:
    prev = None
    for row in _read_jsonl(path):
        row = dict(row)
        entry_hash = row.pop("entry_hash", None)
        if row.get("prev_hash") != prev:
            return False
        expected = sha256_hex(canonical_json_dumps(row).encode("utf-8"))
        if expected != entry_hash:
            return False
        prev = entry_hash
    return True


def mind_graph_sig(snapshot: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(snapshot).encode("utf-8"))


def ato_from_concept_registry_v70(concept: Dict[str, Any]) -> ATOv71:
    """
    Adapter (V70 -> V71): ConceptObjectV70 dict -> ATOv71(CONCEPT).
    """
    if not isinstance(concept, dict):
        raise ValueError("concept_not_dict")
    concept_id = str(concept.get("concept_id") or "")
    if not concept_id:
        raise ValueError("concept_missing_id")
    subgraph = {
        "interface_sig": str(concept.get("interface_sig") or ""),
        "program_sha256": str(concept.get("program_sha256") or ""),
        "program_len": int(concept.get("program_len", 0) or 0),
        "concept_state": str(concept.get("concept_state") or ""),
    }
    slots = concept.get("slots") if isinstance(concept.get("slots"), dict) else {}
    invariants = concept.get("invariants") if isinstance(concept.get("invariants"), dict) else {}
    evidence_refs = concept.get("evidence_refs") if isinstance(concept.get("evidence_refs"), list) else []
    ev2 = [x for x in evidence_refs if isinstance(x, dict)]
    created_step = int(concept.get("created_step", 0) or 0)
    last_step = int(concept.get("last_step", created_step) or created_step)
    cost = concept.get("cost", 0.0) or 0.0
    return ATOv71(
        ato_id=str(concept_id),
        ato_type="CONCEPT",
        subgraph=dict(subgraph),
        slots=dict(slots),
        bindings={},
        cost=float(cost),
        evidence_refs=list(ev2),
        invariants=dict(invariants),
        created_step=int(created_step),
        last_step=int(last_step),
    )


def _sorted_dict_list(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs: List[Tuple[str, Dict[str, Any]]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            k = canonical_json_dumps(it)
        except Exception:
            k = str(it)
        pairs.append((k, dict(it)))
    pairs.sort(key=lambda kv: str(kv[0]))
    return [v for _, v in pairs]


@dataclass
class MindGraphV71:
    run_dir: str
    nodes_path: str = field(init=False)
    edges_path: str = field(init=False)

    _nodes_prev_hash: Optional[str] = field(default=None, init=False)
    _edges_prev_hash: Optional[str] = field(default=None, init=False)
    _nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    _edge_sigs: Set[str] = field(default_factory=set, init=False)
    _edges: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        os.makedirs(self.run_dir, exist_ok=False)
        self.nodes_path = os.path.join(self.run_dir, "mind_nodes.jsonl")
        self.edges_path = os.path.join(self.run_dir, "mind_edges.jsonl")

    def add_node(self, *, step: int, ato: ATOv71, reason: str) -> Dict[str, Any]:
        if not isinstance(ato, ATOv71):
            raise ValueError("ato_must_be_ATOv71")
        ensure_ato_type(ato.ato_type)

        ato_dict = ato.to_dict(include_sig=True)
        ato_id = str(ato_dict.get("ato_id") or "")
        if not ato_id:
            raise ValueError("missing_ato_id")

        existing = self._nodes.get(ato_id)
        if isinstance(existing, dict) and str(existing.get("ato_sig") or "") == str(ato_dict.get("ato_sig") or ""):
            return dict(existing)

        try:
            snap = copy.deepcopy(ato_dict)
        except Exception:
            snap = dict(ato_dict)
        if not isinstance(snap, dict):
            snap = dict(ato_dict)
        self._nodes[ato_id] = snap

        self._nodes_prev_hash = append_chained_jsonl(
            self.nodes_path,
            {
                "time": deterministic_iso(step=int(step)),
                "step": int(step),
                "event": "NODE",
                "payload": {"reason": str(reason), "ato": dict(snap)},
            },
            prev_hash=self._nodes_prev_hash,
        )
        return dict(snap)

    def add_edge(
        self,
        *,
        step: int,
        src_ato_id: str,
        dst_ato_id: str,
        edge_type: str,
        evidence_refs: Sequence[Dict[str, Any]],
        reason: str,
    ) -> Dict[str, Any]:
        et = ensure_edge_type(edge_type)
        src = str(src_ato_id or "")
        dst = str(dst_ato_id or "")
        if not src or not dst:
            raise ValueError("missing_src_or_dst")
        if src not in self._nodes or dst not in self._nodes:
            raise ValueError("missing_endpoint_node")

        ev2 = _sorted_dict_list([x for x in evidence_refs if isinstance(x, dict)])
        edge_sem_sig = {"src": src, "dst": dst, "edge_type": et, "evidence_refs": list(ev2)}
        edge_sig = stable_hash_obj(edge_sem_sig)
        if edge_sig in self._edge_sigs:
            # Idempotent: do not append duplicates.
            return dict(edge_sem_sig, edge_sig=str(edge_sig))
        self._edge_sigs.add(edge_sig)

        edge = dict(edge_sem_sig, edge_sig=str(edge_sig))
        try:
            edge_snap = copy.deepcopy(edge)
        except Exception:
            edge_snap = dict(edge)
        if not isinstance(edge_snap, dict):
            edge_snap = dict(edge)
        self._edges.append(edge_snap)

        self._edges_prev_hash = append_chained_jsonl(
            self.edges_path,
            {
                "time": deterministic_iso(step=int(step)),
                "step": int(step),
                "event": "EDGE",
                "payload": {"reason": str(reason), "edge": dict(edge_snap)},
            },
            prev_hash=self._edges_prev_hash,
        )
        return dict(edge_snap)

    def verify_chains(self) -> Dict[str, bool]:
        return {
            "mind_nodes_chain_ok": bool(verify_chained_jsonl(self.nodes_path)),
            "mind_edges_chain_ok": bool(verify_chained_jsonl(self.edges_path)),
        }

    def snapshot_graph_state(self) -> Dict[str, Any]:
        nodes = [self._nodes[k] for k in sorted(self._nodes.keys())]
        edges = list(self._edges)
        edges.sort(
            key=lambda e: (
                str(e.get("src") or ""),
                str(e.get("dst") or ""),
                str(e.get("edge_type") or ""),
                str(e.get("edge_sig") or ""),
            )
        )
        return {"schema_version": 1, "nodes": list(nodes), "edges": list(edges)}

    def graph_sig(self) -> str:
        return mind_graph_sig(self.snapshot_graph_state())

