from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .act import Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from .store import ActStore


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _ensure_str_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for x in items:
        if isinstance(x, str) and x:
            out.append(x)
    # stable unique
    return sorted(set(out))


def _canon_goal_kinds(match: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(match) if isinstance(match, dict) else {}
    gk = m.get("goal_kinds")
    if isinstance(gk, list):
        m["goal_kinds"] = _ensure_str_list(gk)
    return m


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return iter(())
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_chained_jsonl_v87(path: str, entry: Dict[str, Any], *, prev_hash: Optional[str]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    body = dict(entry)
    body["prev_hash"] = prev_hash
    entry_hash = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    body["entry_hash"] = entry_hash
    with open(path, "a", encoding="utf-8") as f:
        f.write(canonical_json_dumps(body))
        f.write("\n")
    return entry_hash


def verify_chained_jsonl_v87(path: str) -> bool:
    prev: Optional[str] = None
    for row in _read_jsonl(path):
        row = dict(row)
        entry_hash = row.pop("entry_hash", None)
        if row.get("prev_hash") != prev:
            return False
        expected = sha256_hex(canonical_json_dumps(row).encode("utf-8"))
        if expected != entry_hash:
            return False
        prev = str(entry_hash)
    return True


def _canon_bind_map(bind_map: Any) -> Dict[str, str]:
    if not isinstance(bind_map, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in bind_map.items():
        ks = str(k)
        vs = str(v)
        if not ks or not vs:
            continue
        out[ks] = vs
    return {k: out[k] for k in sorted(out.keys())}


def _canon_nodes(nodes: Any) -> List[Dict[str, Any]]:
    if not isinstance(nodes, list):
        return []
    out: List[Dict[str, Any]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        act_id = str(n.get("act_id") or "")
        if not act_id:
            continue
        produces = str(n.get("produces") or n.get("out") or "")
        role = str(n.get("role") or "")
        node = {
            "act_id": act_id,
            "bind": _canon_bind_map(n.get("bind") if "bind" in n else n.get("bind_map")),
            "produces": produces,
        }
        if role:
            node["role"] = role
        out.append(node)
    return out


def _toposort_nodes(nodes: Sequence[Dict[str, Any]]) -> List[int]:
    """
    Deterministic topological sort using node_sig tie-breaks.
    Cycles fail-closed.
    """
    node_sigs: List[str] = [_stable_hash_obj(n) for n in nodes]
    produces_map: Dict[str, int] = {}
    for idx, n in enumerate(nodes):
        pv = str(n.get("produces") or "")
        if not pv:
            continue
        if pv in produces_map:
            raise ValueError(f"duplicate_produces_var:{pv}")
        produces_map[pv] = idx

    deps: List[set] = [set() for _ in nodes]
    outs: List[set] = [set() for _ in nodes]
    for j, n in enumerate(nodes):
        uses = n.get("bind")
        if not isinstance(uses, dict):
            continue
        for v in uses.values():
            vs = str(v)
            if not vs:
                continue
            i = produces_map.get(vs)
            if i is None:
                continue
            deps[j].add(i)
            outs[i].add(j)

    incoming = [len(d) for d in deps]
    ready: List[int] = [i for i, c in enumerate(incoming) if c == 0]
    ready.sort(key=lambda i: node_sigs[i])

    order: List[int] = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for j in sorted(outs[i], key=lambda x: node_sigs[x]):
            incoming[j] -= 1
            if incoming[j] == 0:
                ready.append(j)
                ready.sort(key=lambda x: node_sigs[x])

    if len(order) != len(nodes):
        raise ValueError("cycle_detected")
    return order


def _derive_edges(nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Build mapping var->producer index.
    produces_map: Dict[str, int] = {}
    for idx, n in enumerate(nodes):
        pv = str(n.get("produces") or "")
        if pv:
            produces_map[pv] = idx

    edges: List[Dict[str, Any]] = []
    for dst_idx, n in enumerate(nodes):
        bind = n.get("bind")
        if not isinstance(bind, dict):
            continue
        for var in bind.values():
            v = str(var)
            if not v:
                continue
            src_idx = produces_map.get(v)
            if src_idx is None:
                continue
            if int(src_idx) == int(dst_idx):
                continue
            edges.append({"src": int(src_idx), "dst": int(dst_idx), "var": str(v)})
    edges.sort(key=lambda e: (int(e["src"]), int(e["dst"]), str(e["var"])))
    return edges


def _infer_interface_from_nodes(nodes: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    produced: set = set()
    used: set = set()
    for n in nodes:
        pv = str(n.get("produces") or "")
        if pv:
            produced.add(pv)
        bind = n.get("bind")
        if isinstance(bind, dict):
            for v in bind.values():
                vs = str(v)
                if vs:
                    used.add(vs)
    inputs = sorted(used - produced)
    outputs = sorted(produced - used) if produced else []
    return inputs, outputs


def canonicalize_csg_v87(csg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonical CSG schema (V87):
      {
        "schema_version": 1,
        "nodes": [{"act_id","bind","produces","role"?}, ...],   # topologically sorted
        "edges": [{"src","dst","var"}, ...],                   # derived from bind/produces
        "interface": {
           "inputs": [...],
           "outputs": [...],
           "validator_ids": [...],
           "match": {...}
        }
      }
    """
    if not isinstance(csg, dict):
        raise ValueError("csg_not_dict")
    schema_version = int(csg.get("schema_version", 1) or 1)
    if schema_version != 1:
        raise ValueError(f"unsupported_csg_schema_version:{schema_version}")

    nodes_raw = _canon_nodes(csg.get("nodes"))
    if not nodes_raw:
        raise ValueError("empty_nodes")

    # Canonical order is deterministic topological sort.
    order = _toposort_nodes(nodes_raw)
    nodes: List[Dict[str, Any]] = [nodes_raw[i] for i in order]
    edges = _derive_edges(nodes)

    iface = csg.get("interface")
    iface = dict(iface) if isinstance(iface, dict) else {}
    inf_inputs, inf_outputs = _infer_interface_from_nodes(nodes)
    iface_inputs = _ensure_str_list(iface.get("inputs")) if "inputs" in iface else inf_inputs
    iface_outputs = _ensure_str_list(iface.get("outputs")) if "outputs" in iface else inf_outputs
    if "inputs" in iface and iface_inputs != inf_inputs:
        raise ValueError(f"interface_inputs_mismatch:want={iface_inputs}:got={inf_inputs}")
    if "outputs" in iface and iface_outputs != inf_outputs:
        raise ValueError(f"interface_outputs_mismatch:want={iface_outputs}:got={inf_outputs}")

    validator_ids = _ensure_str_list(iface.get("validator_ids"))
    match = _canon_goal_kinds(iface.get("match") if isinstance(iface.get("match"), dict) else {})

    iface_canon: Dict[str, Any] = {
        "inputs": list(iface_inputs),
        "outputs": list(iface_outputs),
        "validator_ids": list(validator_ids),
        "match": dict(match),
    }
    return {
        "schema_version": 1,
        "nodes": list(nodes),
        "edges": list(edges),
        "interface": iface_canon,
    }


def csg_hash_v87(csg: Dict[str, Any]) -> str:
    canon = canonicalize_csg_v87(csg)
    return sha256_hex(canonical_json_dumps(canon).encode("utf-8"))


def estimate_cost_v87(csg: Dict[str, Any], store: ActStore) -> Dict[str, Any]:
    canon = canonicalize_csg_v87(csg)
    nodes = canon.get("nodes") if isinstance(canon.get("nodes"), list) else []
    edges = canon.get("edges") if isinstance(canon.get("edges"), list) else []

    uniq_act_ids: List[str] = []
    seen: set = set()
    for n in nodes:
        if not isinstance(n, dict):
            continue
        aid = str(n.get("act_id") or "")
        if aid and aid not in seen:
            seen.add(aid)
            uniq_act_ids.append(aid)

    prog_len_total = 0
    for aid in sorted(uniq_act_ids):
        act = store.get_concept_act(str(aid))
        if act is None:
            continue
        prog_len_total += int(len(act.program or []))

    cost_units = int(len(nodes)) * 1000 + int(len(edges)) * 100 + int(prog_len_total)
    return {
        "schema_version": 1,
        "estimate_kind": "v87_simple",
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "act_program_len_total": int(prog_len_total),
        "cost_units": int(cost_units),
    }


def csg_expand_v87(csg: Dict[str, Any], store: ActStore) -> List[Dict[str, Any]]:
    """
    Deterministic linearization of a CSG for replay/audit.
    Returns a list of CALL-like steps: {"act_id","bind","produces","idx"}.
    """
    canon = canonicalize_csg_v87(csg)
    nodes = canon.get("nodes") if isinstance(canon.get("nodes"), list) else []
    steps: List[Dict[str, Any]] = []
    for idx, n in enumerate(nodes):
        if not isinstance(n, dict):
            continue
        aid = str(n.get("act_id") or "")
        if not aid:
            continue
        if store.get_concept_act(aid) is None:
            raise ValueError(f"missing_node_act:{aid}")
        steps.append(
            {
                "idx": int(idx),
                "act_id": str(aid),
                "bind": dict(n.get("bind") if isinstance(n.get("bind"), dict) else {}),
                "produces": str(n.get("produces") or ""),
                "role": str(n.get("role") or ""),
            }
        )
    return steps


def csg_to_concept_program_v87(csg: Dict[str, Any]) -> List[Instruction]:
    """
    Convert CSG nodes into a linear concept_csv program using CSV_CALL steps.
    The program assumes interface.inputs are provided as direct inputs with the same names.
    Returns the first output in interface.outputs via CSV_RETURN.
    """
    canon = canonicalize_csg_v87(csg)
    iface = canon.get("interface") if isinstance(canon.get("interface"), dict) else {}
    inputs = iface.get("inputs") if isinstance(iface.get("inputs"), list) else []
    outputs = iface.get("outputs") if isinstance(iface.get("outputs"), list) else []
    nodes = canon.get("nodes") if isinstance(canon.get("nodes"), list) else []

    prog: List[Instruction] = []
    for name in inputs:
        n = str(name)
        if not n:
            continue
        prog.append(Instruction("CSV_GET_INPUT", {"name": n, "out": n}))

    for n in nodes:
        if not isinstance(n, dict):
            continue
        callee = str(n.get("act_id") or "")
        bind = n.get("bind") if isinstance(n.get("bind"), dict) else {}
        out = str(n.get("produces") or "")
        prog.append(Instruction("CSV_CALL", {"concept_id": callee, "bind": dict(bind), "out": out}))

    ret_var = str(outputs[0]) if outputs else str(nodes[-1].get("produces") or "")
    if not ret_var:
        raise ValueError("missing_return_var")
    prog.append(Instruction("CSV_RETURN", {"var": ret_var}))
    return prog


@dataclass
class CsvCsgLogsV87:
    run_dir: str
    concepts_path: str
    evidence_path: str
    telemetry_path: str
    _concepts_prev: Optional[str] = None
    _evidence_prev: Optional[str] = None
    _telemetry_prev: Optional[str] = None

    @staticmethod
    def init(run_dir: str) -> "CsvCsgLogsV87":
        os.makedirs(run_dir, exist_ok=False)
        return CsvCsgLogsV87(
            run_dir=str(run_dir),
            concepts_path=os.path.join(run_dir, "concepts.jsonl"),
            evidence_path=os.path.join(run_dir, "concept_evidence.jsonl"),
            telemetry_path=os.path.join(run_dir, "concept_telemetry.jsonl"),
        )

    def append_concept_def(self, *, step: int, concept: Dict[str, Any]) -> str:
        self._concepts_prev = append_chained_jsonl_v87(
            self.concepts_path,
            {
                "time": deterministic_iso(step=int(step)),
                "step": int(step),
                "event": "CONCEPT_DEF",
                "payload": dict(concept),
            },
            prev_hash=self._concepts_prev,
        )
        return str(self._concepts_prev)

    def append_evidence(self, *, step: int, evidence: Dict[str, Any]) -> str:
        self._evidence_prev = append_chained_jsonl_v87(
            self.evidence_path,
            {
                "time": deterministic_iso(step=int(step)),
                "step": int(step),
                "event": "EVIDENCE",
                "payload": dict(evidence),
            },
            prev_hash=self._evidence_prev,
        )
        return str(self._evidence_prev)

    def append_telemetry(self, *, step: int, telemetry: Dict[str, Any]) -> str:
        self._telemetry_prev = append_chained_jsonl_v87(
            self.telemetry_path,
            {
                "time": deterministic_iso(step=int(step)),
                "step": int(step),
                "event": "TELEMETRY",
                "payload": dict(telemetry),
            },
            prev_hash=self._telemetry_prev,
        )
        return str(self._telemetry_prev)

    def verify_chains(self) -> Dict[str, Any]:
        return {
            "concepts_chain_ok": bool(verify_chained_jsonl_v87(self.concepts_path)),
            "evidence_chain_ok": bool(verify_chained_jsonl_v87(self.evidence_path)),
            "telemetry_chain_ok": bool(verify_chained_jsonl_v87(self.telemetry_path)),
        }


def build_csg_concept_def_v87(
    *,
    csg: Dict[str, Any],
    store: ActStore,
    concept_id_prefix: str = "concept_csv_v87_",
) -> Dict[str, Any]:
    """
    Build a deterministic concept definition (registry row) for a CSG-backed concept.
    """
    csg_canon = canonicalize_csg_v87(csg)
    csg_hash = sha256_hex(canonical_json_dumps(csg_canon).encode("utf-8"))
    iface = csg_canon.get("interface") if isinstance(csg_canon.get("interface"), dict) else {}
    match = iface.get("match") if isinstance(iface.get("match"), dict) else {}
    cost = estimate_cost_v87(csg_canon, store)
    concept_id = f"{str(concept_id_prefix)}{csg_hash}"
    concept = {
        "schema_version": 1,
        "concept_id": str(concept_id),
        "csg_hash": str(csg_hash),
        "csg": dict(csg_canon),
        "interface": dict(iface),
        "match": dict(match),
        "cost": dict(cost),
        "concept_sig": _stable_hash_obj(
            {
                "concept_id": str(concept_id),
                "csg_hash": str(csg_hash),
                "interface": dict(iface),
                "match": dict(match),
                "cost": dict(cost),
            }
        ),
    }
    return concept


def maybe_deepcopy(obj: Any) -> Any:
    try:
        return copy.deepcopy(obj)
    except Exception:
        if isinstance(obj, dict):
            return dict(obj)
        if isinstance(obj, list):
            return list(obj)
        return obj

