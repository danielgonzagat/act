"""
failure_driven_miner_v146.py - Miner que transforma falhas do gate em conceitos.

LEI DA ABSTRAÇÃO POR FALHA:
Nenhuma solução rejeitada pode ser descartada silenciosamente.
Para cada REJECTED_CONCEPTLESS, o sistema DEVE propor conceito candidato.

Este é o ÚLTIMO OPERADOR que fecha o loop AGI:

    SOLVER encontra solução
            ↓
    INEVITABILITY_GATE rejeita
            ↓
    FAILURE-DRIVEN MINER  ← ESTE MÓDULO
            ↓
    CONCEITO CANDIDATO (com CSV_CALL, depth↑)
            ↓
    REEXECUÇÃO FORÇADA COM CONCEITO
            ↓
    GATE
       ├─ PASSA → PROMOVER (PCC + ledger)
       └─ FALHA → DESCARTAR CONCEITO

Schema version: 146
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from .inevitability_gate_v145 import (
    INEVITABILITY_GATE_SCHEMA_VERSION_V145,
    GateRejectionReason,
    GateResult,
    InevitabilityConfig,
    inevitability_gate,
)
from .solver_concept_gate_v145 import (
    GatedSolverResult,
    solve_with_concept_gate,
)
from .grid_v124 import GridV124

FAILURE_DRIVEN_MINER_SCHEMA_VERSION_V146 = 146


# ─────────────────────────────────────────────────────────────────────────────
# Rejection Reason → Concept Requirement Mapping
# ─────────────────────────────────────────────────────────────────────────────


class ConceptRequirement(Enum):
    """Requirements that a concept must satisfy to eliminate a rejection reason."""
    MUST_EXIST = auto()              # NO_CONCEPT_USED → concept must exist
    MUST_HAVE_DEPTH = auto()         # DEPTH_TOO_SHALLOW → depth >= min
    MUST_HAVE_CSV_CALL = auto()      # NO_CSV_CALL → at least 1 CSV_CALL
    MUST_HAVE_COMPOSITION = auto()   # FLAT_CONCEPT → call_deps non-empty
    MUST_HAVE_REUSE = auto()         # NO_CROSS_TASK_REUSE → used in 2+ tasks
    MUST_HAVE_PCC = auto()           # MISSING_PCC → PCC proof required
    MUST_COLLAPSE_SEARCH = auto()    # NO_SEARCH_COLLAPSE → fewer steps


# Explicit mapping: rejection reason → concept requirement
REJECTION_TO_REQUIREMENT: Dict[GateRejectionReason, ConceptRequirement] = {
    GateRejectionReason.NO_CONCEPT_USED: ConceptRequirement.MUST_EXIST,
    GateRejectionReason.ONLY_PRIMITIVES: ConceptRequirement.MUST_EXIST,
    GateRejectionReason.DEPTH_TOO_SHALLOW: ConceptRequirement.MUST_HAVE_DEPTH,
    GateRejectionReason.DEPTH_STAGNATION: ConceptRequirement.MUST_HAVE_DEPTH,
    GateRejectionReason.NO_CSV_CALL: ConceptRequirement.MUST_HAVE_CSV_CALL,
    GateRejectionReason.FLAT_CONCEPT: ConceptRequirement.MUST_HAVE_COMPOSITION,
    GateRejectionReason.NO_CROSS_TASK_REUSE: ConceptRequirement.MUST_HAVE_REUSE,
    GateRejectionReason.MISSING_PCC: ConceptRequirement.MUST_HAVE_PCC,
    GateRejectionReason.INVALID_PCC_HASH: ConceptRequirement.MUST_HAVE_PCC,
    GateRejectionReason.NO_SEARCH_COLLAPSE: ConceptRequirement.MUST_COLLAPSE_SEARCH,
    GateRejectionReason.NO_MEASURABLE_GAIN: ConceptRequirement.MUST_COLLAPSE_SEARCH,
}


def map_rejection_to_requirements(
    rejection_reasons: List[GateRejectionReason],
) -> Set[ConceptRequirement]:
    """Map rejection reasons to concept requirements."""
    requirements: Set[ConceptRequirement] = set()
    for reason in rejection_reasons:
        req = REJECTION_TO_REQUIREMENT.get(reason)
        if req is not None:
            requirements.add(req)
    return requirements


# ─────────────────────────────────────────────────────────────────────────────
# Concept Candidate
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConceptCandidate:
    """A concept proposed to eliminate rejection reasons."""
    
    concept_id: str
    kind: str = "concept_csv"
    
    # What this concept encapsulates
    pattern_description: str = ""
    encapsulated_ops: List[str] = field(default_factory=list)
    
    # Requirements it aims to satisfy
    target_requirements: Set[ConceptRequirement] = field(default_factory=set)
    
    # Structure
    depth: int = 1
    call_deps: List[str] = field(default_factory=list)
    
    # Program (CSV instructions)
    program: List[Instruction] = field(default_factory=list)
    
    # PCC proof
    pcc_hash: str = ""
    
    # Reuse tracking
    tasks_used_in: Set[str] = field(default_factory=set)
    
    # Status
    is_promoted: bool = False
    death_reason: Optional[GateRejectionReason] = None
    
    def to_act(self) -> Act:
        """Convert to Act for use in concept library."""
        return Act(
            id=str(self.concept_id),
            version=1,
            created_at="2024-01-01T00:00:00Z",
            kind=str(self.kind),
            match={"pattern": str(self.pattern_description)},
            program=list(self.program),
            evidence={
                "pcc_v2": {
                    "depth": int(self.depth),
                    "call_deps": list(self.call_deps),
                    "pcc_hash": str(self.pcc_hash) if self.pcc_hash else sha256_hex(
                        canonical_json_dumps({
                            "concept_id": self.concept_id,
                            "ops": self.encapsulated_ops,
                        }).encode("utf-8")
                    ),
                },
            },
            cost={"overhead_bits": 1024},
            deps=list(self.call_deps),
            active=True,
        )
    
    def satisfies_requirements(self) -> Tuple[bool, List[str]]:
        """Check if this candidate satisfies its target requirements."""
        failures: List[str] = []
        
        for req in self.target_requirements:
            if req == ConceptRequirement.MUST_EXIST:
                # Always satisfied if concept exists
                pass
            elif req == ConceptRequirement.MUST_HAVE_DEPTH:
                if self.depth < 1:
                    failures.append(f"depth={self.depth} < 1")
            elif req == ConceptRequirement.MUST_HAVE_CSV_CALL:
                has_csv_call = any(
                    i.op == "CSV_CALL" for i in self.program
                )
                if not has_csv_call:
                    failures.append("no CSV_CALL in program")
            elif req == ConceptRequirement.MUST_HAVE_COMPOSITION:
                if len(self.call_deps) == 0:
                    failures.append("no call_deps (flat concept)")
            elif req == ConceptRequirement.MUST_HAVE_REUSE:
                if len(self.tasks_used_in) < 2:
                    failures.append(f"used in {len(self.tasks_used_in)} tasks < 2")
            elif req == ConceptRequirement.MUST_HAVE_PCC:
                if not self.pcc_hash or len(self.pcc_hash) != 64:
                    failures.append("invalid or missing PCC hash")
        
        return len(failures) == 0, failures


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Extraction from Rejected Solution
# ─────────────────────────────────────────────────────────────────────────────


def extract_pattern_from_solution(
    raw_solver_result: Dict[str, Any],
    task_id: str,
) -> Tuple[str, List[str]]:
    """
    Extract the transformation pattern from a rejected solution.
    
    Returns:
        - Pattern description (human-readable)
        - List of ops used in the solution
    """
    best_program = raw_solver_result.get("best_program")
    if best_program is None:
        return "unknown_pattern", []
    
    steps = best_program.get("steps", [])
    ops = [step.get("op_id", "unknown") for step in steps]
    
    # Create pattern description from ops
    if not ops:
        pattern = "identity"
    elif len(ops) == 1:
        pattern = f"single_{ops[0]}"
    else:
        pattern = "_then_".join(ops[:3])  # First 3 ops
        if len(ops) > 3:
            pattern += f"_and_{len(ops)-3}_more"
    
    return f"{task_id}_{pattern}", ops


# ─────────────────────────────────────────────────────────────────────────────
# Concept Generation from Failure
# ─────────────────────────────────────────────────────────────────────────────


def generate_concept_from_failure(
    failed_result: GatedSolverResult,
    task_id: str,
    existing_concepts: Dict[str, Act],
    concept_counter: int,
) -> ConceptCandidate:
    """
    Generate a concept candidate that addresses the rejection reasons.
    
    This is THE critical function that transforms failure into abstraction.
    """
    # Map rejection reasons to requirements
    requirements = map_rejection_to_requirements(
        failed_result.gate_result.rejection_reasons
    )
    
    # Extract pattern from the (correct but rejected) solution
    raw_dict = failed_result.raw_solver_result.to_dict()
    pattern_desc, ops = extract_pattern_from_solution(raw_dict, task_id)
    
    # Determine depth: must be higher than existing concepts
    max_existing_depth = 0
    for concept in existing_concepts.values():
        evidence = concept.evidence or {}
        pcc = evidence.get("pcc_v2", {})
        depth = int(pcc.get("depth", 0))
        max_existing_depth = max(max_existing_depth, depth)
    
    new_depth = max_existing_depth + 1
    
    # Create call_deps: reference existing concepts if available
    call_deps: List[str] = []
    if existing_concepts:
        # Pick up to 2 existing concepts as deps
        for cid in list(existing_concepts.keys())[:2]:
            call_deps.append(cid)
    else:
        # Create a base dependency (will need to be created too)
        base_dep = f"base_concept_{concept_counter}"
        call_deps.append(base_dep)
    
    # Build program with CSV_CALL
    program: List[Instruction] = []
    
    # Add CSV_CALL for each dependency
    for dep in call_deps:
        program.append(Instruction("CSV_CALL", {"callee": dep, "depth": new_depth - 1}))
    
    # Add the actual transformation ops
    for op in ops:
        program.append(Instruction(op, {}))
    
    # Add CSV_RETURN
    program.append(Instruction("CSV_RETURN", {"var": "result"}))
    
    # Generate PCC hash
    pcc_content = {
        "concept_id": f"concept_{concept_counter}_{pattern_desc}",
        "ops": ops,
        "depth": new_depth,
        "call_deps": call_deps,
        "task_id": task_id,
    }
    pcc_hash = sha256_hex(canonical_json_dumps(pcc_content).encode("utf-8"))
    
    return ConceptCandidate(
        concept_id=f"concept_{concept_counter}_{pattern_desc}",
        kind="concept_csv",
        pattern_description=pattern_desc,
        encapsulated_ops=ops,
        target_requirements=requirements,
        depth=new_depth,
        call_deps=call_deps,
        program=program,
        pcc_hash=pcc_hash,
        tasks_used_in={task_id},
        is_promoted=False,
        death_reason=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Failure-Driven Mining Loop
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MiningIterationResult:
    """Result of one mining iteration."""
    
    task_id: str
    
    # Initial failure
    initial_failure: GatedSolverResult
    rejection_reasons: List[str]
    
    # Concept proposed
    concept_proposed: Optional[ConceptCandidate]
    concept_requirements: Set[ConceptRequirement]
    
    # Re-execution result
    reexecution_result: Optional[GatedSolverResult]
    
    # Outcome
    concept_survived: bool
    concept_death_reason: Optional[str]
    concept_promoted: bool
    
    # Metrics
    search_steps_before: int
    search_steps_after: int
    search_collapse_factor: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(FAILURE_DRIVEN_MINER_SCHEMA_VERSION_V146),
            "task_id": str(self.task_id),
            "rejection_reasons": list(self.rejection_reasons),
            "concept_proposed": self.concept_proposed.concept_id if self.concept_proposed else None,
            "concept_requirements": [r.name for r in self.concept_requirements],
            "concept_survived": bool(self.concept_survived),
            "concept_death_reason": self.concept_death_reason,
            "concept_promoted": bool(self.concept_promoted),
            "search_steps_before": int(self.search_steps_before),
            "search_steps_after": int(self.search_steps_after),
            "search_collapse_factor": float(self.search_collapse_factor),
        }


@dataclass
class MiningSessionResult:
    """Result of a complete mining session."""
    
    iterations: List[MiningIterationResult]
    
    # Concept library evolution
    concepts_proposed: int
    concepts_survived: int
    concepts_promoted: int
    concepts_died: int
    
    # Death reasons
    death_reasons: Dict[str, int]
    
    # Final concept library
    final_concept_library: Dict[str, Act]
    
    # Reuse tracking
    cross_task_reuse_count: int
    
    # Search collapse
    total_search_steps_before: int
    total_search_steps_after: int
    overall_collapse_factor: float
    
    # THE critical question
    could_survive_without_concepts: bool
    honest_answer: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(FAILURE_DRIVEN_MINER_SCHEMA_VERSION_V146),
            "kind": "mining_session_result_v146",
            "concepts_proposed": int(self.concepts_proposed),
            "concepts_survived": int(self.concepts_survived),
            "concepts_promoted": int(self.concepts_promoted),
            "concepts_died": int(self.concepts_died),
            "death_reasons": dict(self.death_reasons),
            "cross_task_reuse_count": int(self.cross_task_reuse_count),
            "total_search_steps_before": int(self.total_search_steps_before),
            "total_search_steps_after": int(self.total_search_steps_after),
            "overall_collapse_factor": float(self.overall_collapse_factor),
            "could_survive_without_concepts": bool(self.could_survive_without_concepts),
            "honest_answer": str(self.honest_answer),
            "iterations": [i.to_dict() for i in self.iterations],
        }


def run_failure_driven_mining(
    tasks: Sequence[Tuple[str, Sequence[Tuple[GridV124, GridV124]], GridV124]],
    initial_concept_library: Optional[Dict[str, Act]] = None,
    max_iterations_per_task: int = 3,
    gate_config: Optional[InevitabilityConfig] = None,
) -> MiningSessionResult:
    """
    Run the failure-driven mining loop.
    
    For each task:
    1. Attempt to solve with current concepts
    2. If rejected, generate concept from failure
    3. Re-execute with new concept
    4. If passes, promote; if fails, record death reason
    
    This is THE loop that closes the AGI system.
    """
    
    concept_library: Dict[str, Act] = dict(initial_concept_library or {})
    concept_counter = len(concept_library)
    
    iterations: List[MiningIterationResult] = []
    death_reasons: Dict[str, int] = {}
    
    total_steps_before = 0
    total_steps_after = 0
    
    for task_id, train_pairs, test_in in tasks:
        
        # Phase 1: Initial solve attempt (likely to fail without concepts)
        initial_result = solve_with_concept_gate(
            task_id=task_id,
            train_pairs=train_pairs,
            test_in=test_in,
            concept_library=concept_library if concept_library else None,
            gate_config=gate_config,
        )
        
        steps_before = initial_result.raw_solver_result.total_programs_evaluated
        total_steps_before += steps_before
        
        # If passed, no need to mine
        if initial_result.gate_result.passed:
            iterations.append(MiningIterationResult(
                task_id=task_id,
                initial_failure=initial_result,
                rejection_reasons=[],
                concept_proposed=None,
                concept_requirements=set(),
                reexecution_result=None,
                concept_survived=True,
                concept_death_reason=None,
                concept_promoted=False,
                search_steps_before=steps_before,
                search_steps_after=steps_before,
                search_collapse_factor=1.0,
            ))
            total_steps_after += steps_before
            continue
        
        # Phase 2: Generate concept from failure
        rejection_reasons = [r.name for r in initial_result.gate_result.rejection_reasons]
        requirements = map_rejection_to_requirements(initial_result.gate_result.rejection_reasons)
        
        concept = generate_concept_from_failure(
            failed_result=initial_result,
            task_id=task_id,
            existing_concepts=concept_library,
            concept_counter=concept_counter,
        )
        concept_counter += 1
        
        # Phase 3: Add concept to library and re-execute
        concept_library[concept.concept_id] = concept.to_act()
        
        reexec_result = solve_with_concept_gate(
            task_id=task_id,
            train_pairs=train_pairs,
            test_in=test_in,
            concept_library=concept_library,
            gate_config=gate_config,
        )
        
        steps_after = reexec_result.raw_solver_result.total_programs_evaluated
        total_steps_after += steps_after
        
        collapse_factor = steps_after / max(1, steps_before)
        
        # Phase 4: Evaluate concept survival
        if reexec_result.gate_result.passed:
            # Concept survived!
            concept.is_promoted = True
            concept_survived = True
            death_reason_str = None
        else:
            # Concept died - record why
            concept_survived = False
            if reexec_result.gate_result.rejection_reasons:
                death_reason = reexec_result.gate_result.rejection_reasons[0]
                concept.death_reason = death_reason
                death_reason_str = death_reason.name
                death_reasons[death_reason_str] = death_reasons.get(death_reason_str, 0) + 1
            else:
                death_reason_str = "UNKNOWN"
                death_reasons["UNKNOWN"] = death_reasons.get("UNKNOWN", 0) + 1
            
            # Remove failed concept from library
            del concept_library[concept.concept_id]
        
        iterations.append(MiningIterationResult(
            task_id=task_id,
            initial_failure=initial_result,
            rejection_reasons=rejection_reasons,
            concept_proposed=concept,
            concept_requirements=requirements,
            reexecution_result=reexec_result,
            concept_survived=concept_survived,
            concept_death_reason=death_reason_str,
            concept_promoted=concept.is_promoted,
            search_steps_before=steps_before,
            search_steps_after=steps_after,
            search_collapse_factor=collapse_factor,
        ))
    
    # Calculate cross-task reuse
    concept_usage: Dict[str, Set[str]] = {}
    for concept_id in concept_library:
        concept_usage[concept_id] = set()
    
    for it in iterations:
        if it.concept_proposed and it.concept_survived:
            for cid in concept_library:
                concept_usage[cid].add(it.task_id)
    
    cross_task_reuse = sum(1 for cid, tasks in concept_usage.items() if len(tasks) >= 2)
    
    # Final metrics
    concepts_proposed = sum(1 for it in iterations if it.concept_proposed)
    concepts_survived = sum(1 for it in iterations if it.concept_survived and it.concept_proposed)
    concepts_promoted = sum(1 for it in iterations if it.concept_promoted)
    concepts_died = concepts_proposed - concepts_survived
    
    overall_collapse = total_steps_after / max(1, total_steps_before)
    
    # THE critical question
    conceptless_passes = sum(
        1 for it in iterations 
        if it.concept_survived and (not it.concept_proposed)
    )
    
    could_survive = conceptless_passes > 0 or concepts_survived == 0
    
    if could_survive:
        honest_answer = "YES - BUG (or no emergence)"
    else:
        honest_answer = "NO - GOOD (concepts required)"
    
    return MiningSessionResult(
        iterations=iterations,
        concepts_proposed=concepts_proposed,
        concepts_survived=concepts_survived,
        concepts_promoted=concepts_promoted,
        concepts_died=concepts_died,
        death_reasons=death_reasons,
        final_concept_library=concept_library,
        cross_task_reuse_count=cross_task_reuse,
        total_search_steps_before=total_steps_before,
        total_search_steps_after=total_steps_after,
        overall_collapse_factor=overall_collapse,
        could_survive_without_concepts=could_survive,
        honest_answer=honest_answer,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Iteration Output (per mandate)
# ─────────────────────────────────────────────────────────────────────────────


def format_iteration_report(session: MiningSessionResult) -> str:
    """
    Format the mandatory output per iteration.
    
    Required by mandate:
    1. Lista de falhas do gate
    2. Conceitos propostos por falha
    3. Quais conceitos morreram e por qual lei
    4. Conceitos sobreviventes (se houver)
    5. Impacto no search space
    6. Resposta explícita: "O sistema conseguiria sobreviver sem conceitos?"
    """
    lines: List[str] = []
    
    lines.append("=" * 60)
    lines.append("FAILURE-DRIVEN MINING REPORT V146")
    lines.append("=" * 60)
    
    # 1. Lista de falhas do gate
    lines.append("\n1. GATE FAILURES:")
    for it in session.iterations:
        if it.rejection_reasons:
            lines.append(f"   [{it.task_id}] {', '.join(it.rejection_reasons)}")
    
    # 2. Conceitos propostos por falha
    lines.append("\n2. CONCEPTS PROPOSED:")
    for it in session.iterations:
        if it.concept_proposed:
            lines.append(f"   [{it.task_id}] {it.concept_proposed.concept_id}")
            lines.append(f"      Pattern: {it.concept_proposed.pattern_description}")
            lines.append(f"      Depth: {it.concept_proposed.depth}")
            lines.append(f"      Requirements: {[r.name for r in it.concept_requirements]}")
    
    # 3. Conceitos que morreram e por qual lei
    lines.append("\n3. CONCEPTS DIED:")
    for it in session.iterations:
        if it.concept_proposed and not it.concept_survived:
            lines.append(f"   [{it.task_id}] {it.concept_proposed.concept_id}")
            lines.append(f"      Death reason: {it.concept_death_reason}")
    
    if session.death_reasons:
        lines.append("\n   Death reasons summary:")
        for reason, count in session.death_reasons.items():
            lines.append(f"      {reason}: {count}")
    
    # 4. Conceitos sobreviventes
    lines.append("\n4. CONCEPTS SURVIVED:")
    if session.final_concept_library:
        for cid, concept in session.final_concept_library.items():
            evidence = concept.evidence or {}
            pcc = evidence.get("pcc_v2", {})
            lines.append(f"   {cid}")
            lines.append(f"      Depth: {pcc.get('depth', 0)}")
            lines.append(f"      Call deps: {pcc.get('call_deps', [])}")
    else:
        lines.append("   (none)")
    
    # 5. Impacto no search space
    lines.append("\n5. SEARCH SPACE IMPACT:")
    lines.append(f"   Steps before: {session.total_search_steps_before}")
    lines.append(f"   Steps after: {session.total_search_steps_after}")
    lines.append(f"   Collapse factor: {session.overall_collapse_factor:.2f}")
    lines.append(f"   Cross-task reuse: {session.cross_task_reuse_count}")
    
    # 6. THE CRITICAL QUESTION
    lines.append("\n" + "=" * 60)
    lines.append("6. HONEST ANSWER:")
    lines.append("=" * 60)
    lines.append(f"   'O sistema conseguiria sobreviver sem conceitos?'")
    lines.append(f"   >>> {session.honest_answer} <<<")
    lines.append("")
    
    lines.append(f"   Concepts proposed: {session.concepts_proposed}")
    lines.append(f"   Concepts survived: {session.concepts_survived}")
    lines.append(f"   Concepts promoted: {session.concepts_promoted}")
    lines.append(f"   Concepts died: {session.concepts_died}")
    
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# WORM Ledger Entry
# ─────────────────────────────────────────────────────────────────────────────


def write_mining_session_to_ledger(
    session: MiningSessionResult,
    prev_hash: str = "",
) -> Dict[str, Any]:
    """Write mining session to WORM-compliant ledger."""
    import datetime as _dt
    timestamp = _dt.datetime.now(_dt.timezone.utc).isoformat()
    entry = {
        "schema_version": int(FAILURE_DRIVEN_MINER_SCHEMA_VERSION_V146),
        "kind": "mining_session_ledger_entry_v146",
        "session": session.to_dict(),
        "prev_hash": str(prev_hash),
        "timestamp": timestamp,
    }
    
    entry_hash = sha256_hex(canonical_json_dumps(entry).encode("utf-8"))
    entry["entry_hash"] = str(entry_hash)
    
    return entry
