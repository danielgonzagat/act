"""
csv_composed_miner_v144.py - Mining Hierarchical Concepts from CSV_CALL Chains.

This module implements composition mining that detects patterns of CSV_CALL
chains and promotes them as hierarchical concepts, forcing semantic depth.

Design principles:
- Detect frequent CSV_CALL subgraphs across traces
- Promote compound concepts with PCC v2 (call_deps)
- Measure and enforce depth for LAW_DEPTH compliance
- Deterministic: same traces → same mined concepts

Key features:
- Subgraph extraction from traces with CSV_CALL events
- Frequency analysis with alpha-renaming normalization
- MDL-based promotion criteria
- PCC generation with call_deps for proof chain

Schema version: 144
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex


CSV_COMPOSED_MINER_SCHEMA_VERSION_V144 = 144


# ─────────────────────────────────────────────────────────────────────────────
# Call Graph Structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CallNode:
    """A node in the call graph representing a CSV_CALL."""
    callee_id: str
    bind_sig: str  # Signature of bind mapping
    depth: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "callee_id": str(self.callee_id),
            "bind_sig": str(self.bind_sig),
            "depth": int(self.depth),
        }


@dataclass(frozen=True)
class CallEdge:
    """An edge in the call graph representing dataflow."""
    from_node: int  # Index of source node
    to_node: int    # Index of target node
    var_name: str   # Variable name flowing

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_node": int(self.from_node),
            "to_node": int(self.to_node),
            "var_name": str(self.var_name),
        }


@dataclass
class CallSubgraph:
    """A subgraph of CSV_CALLs extracted from a trace."""
    nodes: Tuple[CallNode, ...]
    edges: Tuple[CallEdge, ...]
    root_concept_id: str
    max_depth: int
    trace_id: str
    context_id: str
    family_id: str

    def signature(self) -> str:
        """Compute normalized signature for deduplication."""
        # Alpha-rename: replace concrete callee IDs with generic placeholders
        callee_map: Dict[str, str] = {}
        normalized_nodes = []
        
        for node in self.nodes:
            if node.callee_id not in callee_map:
                callee_map[node.callee_id] = f"C{len(callee_map)}"
            normalized_nodes.append({
                "callee": callee_map[node.callee_id],
                "bind_sig": node.bind_sig,
                "depth": node.depth,
            })
        
        normalized_edges = [
            {"from": e.from_node, "to": e.to_node, "var": e.var_name}
            for e in self.edges
        ]
        
        sig_obj = {
            "nodes": normalized_nodes,
            "edges": normalized_edges,
        }
        return sha256_hex(canonical_json_dumps(sig_obj).encode("utf-8"))[:32]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(CSV_COMPOSED_MINER_SCHEMA_VERSION_V144),
            "kind": "call_subgraph_v144",
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "root_concept_id": str(self.root_concept_id),
            "max_depth": int(self.max_depth),
            "trace_id": str(self.trace_id),
            "context_id": str(self.context_id),
            "family_id": str(self.family_id),
            "signature": self.signature(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Subgraph Extraction from Traces
# ─────────────────────────────────────────────────────────────────────────────


def extract_call_subgraphs(
    trace_events: Sequence[Dict[str, Any]],
    *,
    trace_id: str,
    context_id: str = "",
    family_id: str = "",
) -> List[CallSubgraph]:
    """
    Extract call subgraphs from a trace's concept execution events.
    
    Looks for CSV_CALL events and builds subgraphs representing
    the call hierarchy.
    """
    # Find all CSV_CALL events
    call_events: List[Dict[str, Any]] = []
    for ev in trace_events:
        if not isinstance(ev, dict):
            continue
        op = str(ev.get("op", ""))
        if op == "CSV_CALL":
            call_events.append(ev)
    
    if not call_events:
        return []
    
    # Build call nodes
    nodes: List[CallNode] = []
    callee_to_idx: Dict[str, int] = {}
    
    for ev in call_events:
        callee = str(ev.get("callee", "") or ev.get("concept_id", ""))
        bind = ev.get("bind", {})
        bind_sig = sha256_hex(canonical_json_dumps(bind).encode("utf-8"))[:16] if bind else ""
        depth = int(ev.get("depth", 0) or 0)
        
        node = CallNode(callee_id=callee, bind_sig=bind_sig, depth=depth)
        nodes.append(node)
        callee_to_idx[callee] = len(nodes) - 1
    
    # Build edges (dataflow: output of one call → input of another)
    edges: List[CallEdge] = []
    var_producers: Dict[str, int] = {}  # var_name → node index that produced it
    
    for i, ev in enumerate(call_events):
        out_var = str(ev.get("out", ""))
        if out_var:
            var_producers[out_var] = i
        
        bind = ev.get("bind", {})
        if isinstance(bind, dict):
            for param, var in bind.items():
                var_str = str(var)
                if var_str in var_producers:
                    from_idx = var_producers[var_str]
                    edges.append(CallEdge(
                        from_node=int(from_idx),
                        to_node=int(i),
                        var_name=var_str,
                    ))
    
    if not nodes:
        return []
    
    # Find root (deepest node or first node)
    max_depth = max(n.depth for n in nodes)
    root_idx = 0
    for i, n in enumerate(nodes):
        if n.depth == 0:
            root_idx = i
            break
    
    root_concept_id = nodes[root_idx].callee_id if nodes else ""
    
    subgraph = CallSubgraph(
        nodes=tuple(nodes),
        edges=tuple(edges),
        root_concept_id=root_concept_id,
        max_depth=max_depth,
        trace_id=str(trace_id),
        context_id=str(context_id) or str(trace_id),
        family_id=str(family_id) or "default",
    )
    
    return [subgraph]


# ─────────────────────────────────────────────────────────────────────────────
# Composition Candidate
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ComposedConceptCandidate:
    """A candidate for a composed concept from mined subgraphs."""
    signature: str
    subgraphs: List[CallSubgraph]
    frequency: int
    contexts_distinct: int
    families_distinct: int
    max_depth: int
    avg_depth: float
    call_deps: List[str]  # Concept IDs this depends on
    
    # MDL metrics
    estimated_savings_bits: int = 0
    overhead_bits: int = 1024
    
    def utility_score(self) -> float:
        """Compute utility score for promotion decision."""
        if self.frequency == 0:
            return 0.0
        
        # Score = frequency * contexts * families / overhead
        import math
        freq_factor = math.log1p(self.frequency)
        context_factor = math.sqrt(self.contexts_distinct)
        family_factor = math.sqrt(self.families_distinct)
        depth_bonus = 1.0 + 0.2 * self.max_depth
        
        return float(freq_factor * context_factor * family_factor * depth_bonus)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(CSV_COMPOSED_MINER_SCHEMA_VERSION_V144),
            "kind": "composed_concept_candidate_v144",
            "signature": str(self.signature),
            "frequency": int(self.frequency),
            "contexts_distinct": int(self.contexts_distinct),
            "families_distinct": int(self.families_distinct),
            "max_depth": int(self.max_depth),
            "avg_depth": float(self.avg_depth),
            "call_deps": list(self.call_deps),
            "estimated_savings_bits": int(self.estimated_savings_bits),
            "overhead_bits": int(self.overhead_bits),
            "utility_score": float(self.utility_score()),
            "subgraph_count": len(self.subgraphs),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Composition Miner
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ComposedMinerConfig:
    """Configuration for composed concept mining."""
    min_frequency: int = 3
    min_contexts: int = 2
    min_families: int = 1
    min_depth: int = 1
    min_utility_score: float = 1.0
    max_candidates: int = 100
    overhead_bits: int = 1024
    bits_per_call: int = 128

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(CSV_COMPOSED_MINER_SCHEMA_VERSION_V144),
            "kind": "composed_miner_config_v144",
            "min_frequency": int(self.min_frequency),
            "min_contexts": int(self.min_contexts),
            "min_families": int(self.min_families),
            "min_depth": int(self.min_depth),
            "min_utility_score": float(self.min_utility_score),
            "max_candidates": int(self.max_candidates),
            "overhead_bits": int(self.overhead_bits),
            "bits_per_call": int(self.bits_per_call),
        }


def mine_composed_concepts(
    subgraphs: Sequence[CallSubgraph],
    *,
    config: Optional[ComposedMinerConfig] = None,
) -> List[ComposedConceptCandidate]:
    """
    Mine composed concept candidates from call subgraphs.
    
    Groups subgraphs by normalized signature, computes frequency
    and diversity metrics, and returns candidates passing thresholds.
    """
    cfg = config or ComposedMinerConfig()
    
    # Group by signature
    by_sig: Dict[str, List[CallSubgraph]] = {}
    for sg in subgraphs:
        sig = sg.signature()
        if sig not in by_sig:
            by_sig[sig] = []
        by_sig[sig].append(sg)
    
    candidates: List[ComposedConceptCandidate] = []
    
    for sig, sgs in by_sig.items():
        if len(sgs) < cfg.min_frequency:
            continue
        
        # Compute metrics
        contexts = set(sg.context_id for sg in sgs)
        families = set(sg.family_id for sg in sgs)
        depths = [sg.max_depth for sg in sgs]
        max_depth = max(depths) if depths else 0
        avg_depth = sum(depths) / len(depths) if depths else 0.0
        
        if len(contexts) < cfg.min_contexts:
            continue
        if len(families) < cfg.min_families:
            continue
        if max_depth < cfg.min_depth:
            continue
        
        # Extract call_deps from first subgraph (all have same structure)
        call_deps = list(sorted(set(n.callee_id for n in sgs[0].nodes)))
        
        # Estimate savings
        calls_count = len(sgs[0].nodes)
        estimated_savings = len(sgs) * calls_count * cfg.bits_per_call - cfg.overhead_bits
        
        cand = ComposedConceptCandidate(
            signature=str(sig),
            subgraphs=list(sgs),
            frequency=len(sgs),
            contexts_distinct=len(contexts),
            families_distinct=len(families),
            max_depth=int(max_depth),
            avg_depth=float(avg_depth),
            call_deps=call_deps,
            estimated_savings_bits=max(0, estimated_savings),
            overhead_bits=cfg.overhead_bits,
        )
        
        if cand.utility_score() >= cfg.min_utility_score:
            candidates.append(cand)
    
    # Sort by utility score descending
    candidates.sort(key=lambda c: c.utility_score(), reverse=True)
    
    return candidates[:cfg.max_candidates]


# ─────────────────────────────────────────────────────────────────────────────
# Materialize Composed Concept Act
# ─────────────────────────────────────────────────────────────────────────────


def materialize_composed_concept_act(
    candidate: ComposedConceptCandidate,
    *,
    step: int,
    store_content_hash: str,
    title: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Act:
    """
    Turn a mined ComposedConceptCandidate into a concept_csv Act.
    
    The resulting concept wraps the call chain with proper bindings
    and includes call_deps for PCC v2 proof chain.
    """
    if not candidate.subgraphs:
        raise ValueError("Candidate has no subgraphs")
    
    # Use first subgraph as template
    template = candidate.subgraphs[0]
    
    # Build program from call chain
    # This creates a wrapper concept that calls the component concepts
    program: List[Dict[str, Any]] = []
    
    # Input: generic inputs based on first node's bind
    input_schema: Dict[str, str] = {}
    output_schema: Dict[str, str] = {"result": "any"}
    
    var_counter = 0
    env_map: Dict[str, str] = {}  # Original var → new var
    
    for i, node in enumerate(template.nodes):
        # Create input instructions for first-level bindings
        if i == 0:
            # Assume first node takes inputs from outside
            for j, (param, _) in enumerate([(k, v) for k, v in {}]):  # Placeholder
                in_var = f"in{j}"
                input_schema[in_var] = "any"
                program.append({"op": "CSV_GET_INPUT", "args": {"name": in_var, "out": in_var}})
                env_map[in_var] = in_var
        
        # Create CSV_CALL instruction
        out_var = f"v{var_counter}"
        var_counter += 1
        
        bind: Dict[str, str] = {}
        # Build bind from edges pointing to this node
        for edge in template.edges:
            if edge.to_node == i:
                src_var = f"v{edge.from_node}" if edge.from_node >= 0 else edge.var_name
                bind[edge.var_name] = src_var
        
        program.append({
            "op": "CSV_CALL",
            "args": {
                "concept_id": node.callee_id,
                "bind": dict(bind),
                "out": out_var,
            },
        })
        env_map[node.callee_id] = out_var
    
    # Return last result
    last_var = f"v{var_counter - 1}" if var_counter > 0 else "result"
    program.append({"op": "CSV_RETURN", "args": {"var": last_var}})
    
    # Build interface
    interface = {
        "input_schema": input_schema if input_schema else {"in": "any"},
        "output_schema": output_schema,
        "validator_id": "",
    }
    
    # Build evidence with PCC v2 metadata
    ev = {
        "name": "concept_csv_composed_v144",
        "interface": interface,
        "meta": {
            "title": str(title),
            "mined_signature": str(candidate.signature),
            "frequency": int(candidate.frequency),
            "contexts_distinct": int(candidate.contexts_distinct),
            "families_distinct": int(candidate.families_distinct),
            "max_depth": int(candidate.max_depth),
            "utility_score": float(candidate.utility_score()),
            "trained_on_store_content_hash": str(store_content_hash),
            **(dict(meta) if meta else {}),
        },
        "pcc_v2": {
            "call_deps": list(candidate.call_deps),
            "depth": int(candidate.max_depth),
            "composition_kind": "mined_chain",
        },
    }
    
    body = {
        "kind": "concept_csv",
        "version": 1,
        "match": {},
        "program": program,
        "evidence": ev,
        "deps": list(candidate.call_deps),
        "active": True,
    }
    
    act_id = f"act_concept_csv_composed_{sha256_hex(canonical_json_dumps(body).encode('utf-8'))[:12]}"
    
    return Act(
        id=str(act_id),
        version=1,
        created_at=deterministic_iso(step=int(step)),
        kind="concept_csv",
        match={},
        program=[Instruction(d["op"], d.get("args", {})) for d in program],
        evidence=ev,
        cost={"overhead_bits": int(candidate.overhead_bits)},
        deps=list(candidate.call_deps),
        active=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full Mining Pipeline
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ComposedMiningResult:
    """Result of composed concept mining pipeline."""
    subgraphs_extracted: int
    candidates_found: int
    candidates_promoted: int
    promoted_acts: List[Act]
    config: ComposedMinerConfig
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(CSV_COMPOSED_MINER_SCHEMA_VERSION_V144),
            "kind": "composed_mining_result_v144",
            "subgraphs_extracted": int(self.subgraphs_extracted),
            "candidates_found": int(self.candidates_found),
            "candidates_promoted": len(self.promoted_acts),
            "promoted_act_ids": [a.id for a in self.promoted_acts],
            "config": self.config.to_dict(),
        }


def run_composed_mining_pipeline(
    traces: Sequence[Dict[str, Any]],
    *,
    step: int,
    store_content_hash: str,
    config: Optional[ComposedMinerConfig] = None,
    max_promotions: int = 10,
) -> ComposedMiningResult:
    """
    Run full composed concept mining pipeline.
    
    1. Extract call subgraphs from traces
    2. Mine candidates by signature frequency
    3. Promote top candidates as concept_csv Acts
    """
    cfg = config or ComposedMinerConfig()
    
    # Extract subgraphs from all traces
    all_subgraphs: List[CallSubgraph] = []
    for i, trace in enumerate(traces):
        if not isinstance(trace, dict):
            continue
        
        events = trace.get("events", [])
        if not isinstance(events, list):
            events = []
        
        trace_id = str(trace.get("trace_id", f"trace_{i}"))
        context_id = str(trace.get("context_id", trace_id))
        family_id = str(trace.get("family_id", "default"))
        
        sgs = extract_call_subgraphs(
            events,
            trace_id=trace_id,
            context_id=context_id,
            family_id=family_id,
        )
        all_subgraphs.extend(sgs)
    
    # Mine candidates
    candidates = mine_composed_concepts(all_subgraphs, config=cfg)
    
    # Promote top candidates
    promoted_acts: List[Act] = []
    for i, cand in enumerate(candidates[:max_promotions]):
        title = f"composed_concept_{cand.signature[:8]}"
        act = materialize_composed_concept_act(
            cand,
            step=step,
            store_content_hash=store_content_hash,
            title=title,
            meta={"promotion_rank": i},
        )
        promoted_acts.append(act)
    
    return ComposedMiningResult(
        subgraphs_extracted=len(all_subgraphs),
        candidates_found=len(candidates),
        candidates_promoted=len(promoted_acts),
        promoted_acts=promoted_acts,
        config=cfg,
    )
