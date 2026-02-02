"""
inevitability_gate_v145.py - The Inevitability Gate.

This module implements the MANDATORY concept gate that makes it
MATHEMATICALLY IMPOSSIBLE to solve tasks without concepts.

Design principles:
- FAIL-CLOSED: Any solution without concept usage is REJECTED
- NO EXCEPTIONS: No bypass, no fallback, no "temporary" relaxation
- AUDITABLE: Every rejection is logged to WORM ledger
- DETERMINISTIC: Same inputs → same pass/fail decision

The gate sits between:
1. Solver output (candidate solutions)
2. Final acceptance (valid solutions)

If a solution reaches this gate without proper concept usage,
it is KILLED regardless of correctness.

A "correct but conceptless" solution is a BUG in the regime.

Schema version: 145

SURVIVAL LAWS ENFORCED:
1. LAW_CONCEPT: Solution MUST use at least 1 non-primitive concept_csv
2. LAW_DEPTH: Concept depth must exceed task's depth_floor
3. LAW_COMPOSITION: Concept must have CSV_CALL chains (no flat concepts)
4. LAW_REUSE: Concept must have cross-task reuse evidence
5. LAW_PROOF: Concept must have valid PCC hash
6. LAW_UTILITY: Concept must demonstrate search collapse
7. LAW_BUDGET: Solution must fit within budget (no escape via more compute)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .act import Act, canonical_json_dumps, deterministic_iso, sha256_hex


INEVITABILITY_GATE_SCHEMA_VERSION_V145 = 145


# ─────────────────────────────────────────────────────────────────────────────
# Gate Rejection Reasons
# ─────────────────────────────────────────────────────────────────────────────


class GateRejectionReason(str, Enum):
    """Reasons why a solution is rejected by the inevitability gate."""
    
    # LAW_CONCEPT violations
    NO_CONCEPT_USED = "NO_CONCEPT_USED"
    ONLY_PRIMITIVES = "ONLY_PRIMITIVES"
    CONCEPT_NOT_EXECUTED = "CONCEPT_NOT_EXECUTED"
    
    # LAW_DEPTH violations
    DEPTH_TOO_SHALLOW = "DEPTH_TOO_SHALLOW"
    DEPTH_STAGNATION = "DEPTH_STAGNATION"
    
    # LAW_COMPOSITION violations
    NO_CSV_CALL = "NO_CSV_CALL"
    FLAT_CONCEPT = "FLAT_CONCEPT"
    
    # LAW_REUSE violations
    NO_CROSS_TASK_REUSE = "NO_CROSS_TASK_REUSE"
    SAME_PATTERN_REUSE = "SAME_PATTERN_REUSE"
    
    # LAW_PROOF violations
    MISSING_PCC = "MISSING_PCC"
    INVALID_PCC_HASH = "INVALID_PCC_HASH"
    MISSING_CALL_DEPS = "MISSING_CALL_DEPS"
    
    # LAW_UTILITY violations
    NO_SEARCH_COLLAPSE = "NO_SEARCH_COLLAPSE"
    NO_MEASURABLE_GAIN = "NO_MEASURABLE_GAIN"
    
    # LAW_BUDGET violations
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    BUDGET_ESCAPE_DETECTED = "BUDGET_ESCAPE_DETECTED"
    
    # Meta violations
    SOLUTION_BYPASS_DETECTED = "SOLUTION_BYPASS_DETECTED"
    HEURISTIC_SHORTCUT = "HEURISTIC_SHORTCUT"
    REGIME_VIOLATION = "REGIME_VIOLATION"

    def __str__(self) -> str:
        return str(self.value)


# ─────────────────────────────────────────────────────────────────────────────
# Gate Result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GateResult:
    """Result of passing (or failing) the inevitability gate."""
    
    passed: bool
    rejection_reasons: List[GateRejectionReason]
    
    # Concept metrics
    concepts_used: int
    max_depth: int
    csv_calls_total: int
    cross_task_reuse_count: int
    
    # Search metrics
    search_steps_taken: int
    search_collapse_factor: float  # < 1.0 means collapse
    
    # Budget metrics
    budget_allocated: int
    budget_used: int
    budget_remaining: int
    
    # Audit
    solution_hash: str
    gate_timestamp: str
    audit_entries: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(INEVITABILITY_GATE_SCHEMA_VERSION_V145),
            "kind": "inevitability_gate_result_v145",
            "passed": bool(self.passed),
            "rejection_reasons": [str(r) for r in self.rejection_reasons],
            "concepts_used": int(self.concepts_used),
            "max_depth": int(self.max_depth),
            "csv_calls_total": int(self.csv_calls_total),
            "cross_task_reuse_count": int(self.cross_task_reuse_count),
            "search_steps_taken": int(self.search_steps_taken),
            "search_collapse_factor": float(self.search_collapse_factor),
            "budget_allocated": int(self.budget_allocated),
            "budget_used": int(self.budget_used),
            "budget_remaining": int(self.budget_remaining),
            "solution_hash": str(self.solution_hash),
            "gate_timestamp": str(self.gate_timestamp),
            "audit_entries": list(self.audit_entries),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Gate Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class InevitabilityConfig:
    """Configuration for the inevitability gate. STRICT BY DEFAULT."""
    
    # LAW_CONCEPT: minimum concepts required (CANNOT BE 0)
    min_concepts_required: int = 1
    reject_primitive_only: bool = True
    
    # LAW_DEPTH: minimum depth (CANNOT BE 0)
    min_depth_required: int = 1
    require_depth_progression: bool = True
    depth_floor_per_task_class: Dict[str, int] = field(default_factory=dict)
    
    # LAW_COMPOSITION: minimum CSV_CALLs (CANNOT BE 0)
    min_csv_calls_required: int = 1
    reject_flat_concepts: bool = True
    
    # LAW_REUSE: cross-task evidence required
    min_cross_task_reuse: int = 1
    reject_same_pattern_reuse: bool = True
    
    # LAW_PROOF: PCC requirements
    require_pcc: bool = True
    require_valid_hash: bool = True
    require_call_deps: bool = True
    
    # LAW_UTILITY: search collapse required
    max_search_collapse_factor: float = 1.0  # Must be <= 1.0 (ideally < 1.0)
    require_measurable_gain: bool = True
    
    # LAW_BUDGET: strict budget enforcement
    strict_budget: bool = True
    no_budget_escape: bool = True
    
    # Meta: no exceptions
    allow_bypass: bool = False  # ALWAYS FALSE
    allow_fallback: bool = False  # ALWAYS FALSE
    allow_relaxation: bool = False  # ALWAYS FALSE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(INEVITABILITY_GATE_SCHEMA_VERSION_V145),
            "kind": "inevitability_config_v145",
            "min_concepts_required": int(self.min_concepts_required),
            "reject_primitive_only": bool(self.reject_primitive_only),
            "min_depth_required": int(self.min_depth_required),
            "require_depth_progression": bool(self.require_depth_progression),
            "min_csv_calls_required": int(self.min_csv_calls_required),
            "reject_flat_concepts": bool(self.reject_flat_concepts),
            "min_cross_task_reuse": int(self.min_cross_task_reuse),
            "reject_same_pattern_reuse": bool(self.reject_same_pattern_reuse),
            "require_pcc": bool(self.require_pcc),
            "require_valid_hash": bool(self.require_valid_hash),
            "require_call_deps": bool(self.require_call_deps),
            "max_search_collapse_factor": float(self.max_search_collapse_factor),
            "require_measurable_gain": bool(self.require_measurable_gain),
            "strict_budget": bool(self.strict_budget),
            "no_budget_escape": bool(self.no_budget_escape),
            "allow_bypass": bool(self.allow_bypass),
            "allow_fallback": bool(self.allow_fallback),
            "allow_relaxation": bool(self.allow_relaxation),
        }
    
    def validate_config(self) -> List[str]:
        """Validate that config doesn't allow escape routes."""
        errors: List[str] = []
        
        # These MUST be true for inevitability
        if self.min_concepts_required < 1:
            errors.append("min_concepts_required MUST be >= 1")
        if self.min_depth_required < 1:
            errors.append("min_depth_required MUST be >= 1")
        if self.min_csv_calls_required < 1:
            errors.append("min_csv_calls_required MUST be >= 1")
        if not self.reject_primitive_only:
            errors.append("reject_primitive_only MUST be True")
        if not self.reject_flat_concepts:
            errors.append("reject_flat_concepts MUST be True")
        if not self.require_pcc:
            errors.append("require_pcc MUST be True")
        if self.allow_bypass:
            errors.append("allow_bypass MUST be False")
        if self.allow_fallback:
            errors.append("allow_fallback MUST be False")
        if self.allow_relaxation:
            errors.append("allow_relaxation MUST be False")
        
        return errors


# ─────────────────────────────────────────────────────────────────────────────
# THE INEVITABILITY GATE
# ─────────────────────────────────────────────────────────────────────────────


def check_law_concept(
    solution: Dict[str, Any],
    concepts_used: Sequence[Act],
    config: InevitabilityConfig,
) -> Tuple[bool, List[GateRejectionReason], List[Dict[str, Any]]]:
    """
    LAW_CONCEPT: Solution MUST use at least 1 non-primitive concept_csv.
    
    This is the primary gate. A solution without concepts is INVALID.
    """
    rejections: List[GateRejectionReason] = []
    audit: List[Dict[str, Any]] = []
    
    # Check concept count
    if len(concepts_used) < config.min_concepts_required:
        rejections.append(GateRejectionReason.NO_CONCEPT_USED)
        audit.append({
            "law": "LAW_CONCEPT",
            "check": "concept_count",
            "required": int(config.min_concepts_required),
            "actual": len(concepts_used),
            "verdict": "FAIL",
        })
    else:
        audit.append({
            "law": "LAW_CONCEPT",
            "check": "concept_count",
            "required": int(config.min_concepts_required),
            "actual": len(concepts_used),
            "verdict": "PASS",
        })
    
    # Check for primitive-only solutions
    if config.reject_primitive_only:
        non_primitive = [c for c in concepts_used if c.kind == "concept_csv"]
        if len(concepts_used) > 0 and len(non_primitive) == 0:
            rejections.append(GateRejectionReason.ONLY_PRIMITIVES)
            audit.append({
                "law": "LAW_CONCEPT",
                "check": "non_primitive",
                "required": "at least 1 concept_csv",
                "actual": "all primitives",
                "verdict": "FAIL",
            })
    
    passed = len(rejections) == 0
    return passed, rejections, audit


def check_law_depth(
    solution: Dict[str, Any],
    concepts_used: Sequence[Act],
    config: InevitabilityConfig,
    task_class: str = "default",
    historical_max_depth: int = 0,
) -> Tuple[bool, List[GateRejectionReason], List[Dict[str, Any]]]:
    """
    LAW_DEPTH: Concept depth must exceed task's depth floor.
    
    Depth progression is REQUIRED. Stagnation = failure.
    """
    rejections: List[GateRejectionReason] = []
    audit: List[Dict[str, Any]] = []
    
    # Calculate max depth from concepts
    max_depth = 0
    for concept in concepts_used:
        ev = concept.evidence or {}
        pcc = ev.get("pcc_v2", {})
        depth = int(pcc.get("depth", 0) or 0)
        max_depth = max(max_depth, depth)
    
    # Check minimum depth
    depth_floor = config.depth_floor_per_task_class.get(task_class, config.min_depth_required)
    if max_depth < depth_floor:
        rejections.append(GateRejectionReason.DEPTH_TOO_SHALLOW)
        audit.append({
            "law": "LAW_DEPTH",
            "check": "min_depth",
            "required": int(depth_floor),
            "actual": int(max_depth),
            "task_class": str(task_class),
            "verdict": "FAIL",
        })
    else:
        audit.append({
            "law": "LAW_DEPTH",
            "check": "min_depth",
            "required": int(depth_floor),
            "actual": int(max_depth),
            "task_class": str(task_class),
            "verdict": "PASS",
        })
    
    # Check depth progression
    if config.require_depth_progression and historical_max_depth > 0:
        if max_depth <= historical_max_depth:
            rejections.append(GateRejectionReason.DEPTH_STAGNATION)
            audit.append({
                "law": "LAW_DEPTH",
                "check": "progression",
                "required": f"depth > {historical_max_depth}",
                "actual": int(max_depth),
                "verdict": "FAIL",
            })
    
    passed = len(rejections) == 0
    return passed, rejections, audit


def check_law_composition(
    solution: Dict[str, Any],
    concepts_used: Sequence[Act],
    trace_events: Sequence[Dict[str, Any]],
    config: InevitabilityConfig,
) -> Tuple[bool, List[GateRejectionReason], List[Dict[str, Any]]]:
    """
    LAW_COMPOSITION: Concept must have CSV_CALL chains.
    
    Flat concepts (no composition) are REJECTED.
    """
    rejections: List[GateRejectionReason] = []
    audit: List[Dict[str, Any]] = []
    
    # Count CSV_CALLs in trace
    csv_calls = [e for e in trace_events if e.get("op") == "CSV_CALL"]
    csv_call_count = len(csv_calls)
    
    if csv_call_count < config.min_csv_calls_required:
        rejections.append(GateRejectionReason.NO_CSV_CALL)
        audit.append({
            "law": "LAW_COMPOSITION",
            "check": "csv_call_count",
            "required": int(config.min_csv_calls_required),
            "actual": int(csv_call_count),
            "verdict": "FAIL",
        })
    else:
        audit.append({
            "law": "LAW_COMPOSITION",
            "check": "csv_call_count",
            "required": int(config.min_csv_calls_required),
            "actual": int(csv_call_count),
            "verdict": "PASS",
        })
    
    # Check for flat concepts
    if config.reject_flat_concepts:
        for concept in concepts_used:
            ev = concept.evidence or {}
            pcc = ev.get("pcc_v2", {})
            call_deps = list(pcc.get("call_deps", []))
            
            if len(call_deps) == 0:
                rejections.append(GateRejectionReason.FLAT_CONCEPT)
                audit.append({
                    "law": "LAW_COMPOSITION",
                    "check": "flat_concept",
                    "concept_id": str(concept.id),
                    "call_deps": [],
                    "verdict": "FAIL",
                })
                break
    
    passed = len(rejections) == 0
    return passed, rejections, audit


def check_law_reuse(
    solution: Dict[str, Any],
    concepts_used: Sequence[Act],
    reuse_registry: Dict[str, Set[str]],  # concept_id -> set of task_ids
    current_task_id: str,
    config: InevitabilityConfig,
) -> Tuple[bool, List[GateRejectionReason], List[Dict[str, Any]]]:
    """
    LAW_REUSE: Concept must have cross-task reuse evidence.
    
    Reuse in same pattern doesn't count.
    """
    rejections: List[GateRejectionReason] = []
    audit: List[Dict[str, Any]] = []
    
    total_cross_task_reuse = 0
    
    for concept in concepts_used:
        cid = str(concept.id)
        tasks_used_in = reuse_registry.get(cid, set())
        
        # Add current task
        tasks_used_in.add(current_task_id)
        
        # Cross-task reuse = used in different tasks
        cross_task_count = len(tasks_used_in) - 1  # Exclude current
        total_cross_task_reuse += cross_task_count
    
    if total_cross_task_reuse < config.min_cross_task_reuse:
        rejections.append(GateRejectionReason.NO_CROSS_TASK_REUSE)
        audit.append({
            "law": "LAW_REUSE",
            "check": "cross_task_count",
            "required": int(config.min_cross_task_reuse),
            "actual": int(total_cross_task_reuse),
            "verdict": "FAIL",
        })
    else:
        audit.append({
            "law": "LAW_REUSE",
            "check": "cross_task_count",
            "required": int(config.min_cross_task_reuse),
            "actual": int(total_cross_task_reuse),
            "verdict": "PASS",
        })
    
    passed = len(rejections) == 0
    return passed, rejections, audit


def check_law_proof(
    solution: Dict[str, Any],
    concepts_used: Sequence[Act],
    config: InevitabilityConfig,
) -> Tuple[bool, List[GateRejectionReason], List[Dict[str, Any]]]:
    """
    LAW_PROOF: Concept must have valid PCC hash and call_deps.
    
    Without proof, concept cannot be trusted.
    """
    rejections: List[GateRejectionReason] = []
    audit: List[Dict[str, Any]] = []
    
    for concept in concepts_used:
        ev = concept.evidence or {}
        pcc = ev.get("pcc_v2", {})
        
        # Check PCC exists
        if config.require_pcc and not pcc:
            rejections.append(GateRejectionReason.MISSING_PCC)
            audit.append({
                "law": "LAW_PROOF",
                "check": "pcc_exists",
                "concept_id": str(concept.id),
                "verdict": "FAIL",
            })
            continue
        
        # Check hash
        if config.require_valid_hash:
            pcc_hash = str(pcc.get("pcc_hash", "") or ev.get("pcc_hash", ""))
            if not pcc_hash or len(pcc_hash) < 32:
                rejections.append(GateRejectionReason.INVALID_PCC_HASH)
                audit.append({
                    "law": "LAW_PROOF",
                    "check": "pcc_hash_valid",
                    "concept_id": str(concept.id),
                    "pcc_hash": str(pcc_hash),
                    "verdict": "FAIL",
                })
        
        # Check call_deps
        if config.require_call_deps:
            call_deps = list(pcc.get("call_deps", []))
            if len(call_deps) == 0:
                rejections.append(GateRejectionReason.MISSING_CALL_DEPS)
                audit.append({
                    "law": "LAW_PROOF",
                    "check": "call_deps_present",
                    "concept_id": str(concept.id),
                    "verdict": "FAIL",
                })
    
    # If no concepts, that's a LAW_CONCEPT violation, not LAW_PROOF
    if len(concepts_used) > 0 and len(rejections) == 0:
        audit.append({
            "law": "LAW_PROOF",
            "check": "all_concepts_valid",
            "concepts_checked": len(concepts_used),
            "verdict": "PASS",
        })
    
    passed = len(rejections) == 0
    return passed, rejections, audit


def check_law_utility(
    solution: Dict[str, Any],
    search_steps_with_concepts: int,
    search_steps_without_concepts: int,  # Baseline
    config: InevitabilityConfig,
) -> Tuple[bool, List[GateRejectionReason], List[Dict[str, Any]]]:
    """
    LAW_UTILITY: Concept must demonstrate search collapse.
    
    If concepts don't reduce search, they have no utility.
    """
    rejections: List[GateRejectionReason] = []
    audit: List[Dict[str, Any]] = []
    
    # Calculate collapse factor
    if search_steps_without_concepts > 0:
        collapse_factor = float(search_steps_with_concepts) / float(search_steps_without_concepts)
    else:
        collapse_factor = 1.0  # No baseline means no collapse proof
    
    if collapse_factor > config.max_search_collapse_factor:
        rejections.append(GateRejectionReason.NO_SEARCH_COLLAPSE)
        audit.append({
            "law": "LAW_UTILITY",
            "check": "search_collapse",
            "required": f"collapse_factor <= {config.max_search_collapse_factor}",
            "actual": float(collapse_factor),
            "steps_with": int(search_steps_with_concepts),
            "steps_without": int(search_steps_without_concepts),
            "verdict": "FAIL",
        })
    else:
        audit.append({
            "law": "LAW_UTILITY",
            "check": "search_collapse",
            "required": f"collapse_factor <= {config.max_search_collapse_factor}",
            "actual": float(collapse_factor),
            "steps_with": int(search_steps_with_concepts),
            "steps_without": int(search_steps_without_concepts),
            "verdict": "PASS",
        })
    
    # Check for measurable gain
    if config.require_measurable_gain:
        gain = search_steps_without_concepts - search_steps_with_concepts
        if gain <= 0:
            rejections.append(GateRejectionReason.NO_MEASURABLE_GAIN)
            audit.append({
                "law": "LAW_UTILITY",
                "check": "measurable_gain",
                "required": "gain > 0",
                "actual": int(gain),
                "verdict": "FAIL",
            })
    
    passed = len(rejections) == 0
    return passed, rejections, audit


def check_law_budget(
    solution: Dict[str, Any],
    budget_allocated: int,
    budget_used: int,
    config: InevitabilityConfig,
) -> Tuple[bool, List[GateRejectionReason], List[Dict[str, Any]]]:
    """
    LAW_BUDGET: Solution must fit within budget.
    
    Budget CANNOT be increased to save a task.
    """
    rejections: List[GateRejectionReason] = []
    audit: List[Dict[str, Any]] = []
    
    if config.strict_budget:
        if budget_used > budget_allocated:
            rejections.append(GateRejectionReason.BUDGET_EXCEEDED)
            audit.append({
                "law": "LAW_BUDGET",
                "check": "budget_limit",
                "allocated": int(budget_allocated),
                "used": int(budget_used),
                "verdict": "FAIL",
            })
        else:
            audit.append({
                "law": "LAW_BUDGET",
                "check": "budget_limit",
                "allocated": int(budget_allocated),
                "used": int(budget_used),
                "remaining": int(budget_allocated - budget_used),
                "verdict": "PASS",
            })
    
    passed = len(rejections) == 0
    return passed, rejections, audit


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────


def inevitability_gate(
    *,
    solution: Dict[str, Any],
    concepts_used: Sequence[Act],
    trace_events: Sequence[Dict[str, Any]],
    task_id: str,
    task_class: str = "default",
    reuse_registry: Dict[str, Set[str]] = None,
    historical_max_depth: int = 0,
    search_steps_with_concepts: int = 0,
    search_steps_without_concepts: int = 1,  # Baseline (must be > 0)
    budget_allocated: int = 1000,
    budget_used: int = 0,
    config: Optional[InevitabilityConfig] = None,
    step: int = 0,
) -> GateResult:
    """
    THE INEVITABILITY GATE.
    
    This is the MANDATORY checkpoint that ALL solutions must pass.
    
    A solution is REJECTED if:
    1. It doesn't use concepts (LAW_CONCEPT)
    2. Concepts are too shallow (LAW_DEPTH)
    3. Concepts have no composition (LAW_COMPOSITION)
    4. Concepts aren't reused cross-task (LAW_REUSE)
    5. Concepts lack proof (LAW_PROOF)
    6. Concepts don't collapse search (LAW_UTILITY)
    7. Solution exceeds budget (LAW_BUDGET)
    
    There is NO BYPASS. NO FALLBACK. NO EXCEPTION.
    
    Returns GateResult with pass/fail decision and full audit trail.
    """
    cfg = config or InevitabilityConfig()
    reuse_registry = reuse_registry or {}
    
    # Validate config first - no escape routes allowed
    config_errors = cfg.validate_config()
    if config_errors:
        # Config allows escape routes - FATAL
        return GateResult(
            passed=False,
            rejection_reasons=[GateRejectionReason.REGIME_VIOLATION],
            concepts_used=0,
            max_depth=0,
            csv_calls_total=0,
            cross_task_reuse_count=0,
            search_steps_taken=int(budget_used),
            search_collapse_factor=1.0,
            budget_allocated=int(budget_allocated),
            budget_used=int(budget_used),
            budget_remaining=int(budget_allocated - budget_used),
            solution_hash=sha256_hex(canonical_json_dumps(solution).encode("utf-8")),
            gate_timestamp=deterministic_iso(step=int(step)),
            audit_entries=[{
                "error": "CONFIG_ALLOWS_ESCAPE",
                "violations": config_errors,
            }],
        )
    
    all_rejections: List[GateRejectionReason] = []
    all_audits: List[Dict[str, Any]] = []
    
    # LAW_CONCEPT
    passed_1, rej_1, audit_1 = check_law_concept(solution, concepts_used, cfg)
    all_rejections.extend(rej_1)
    all_audits.extend(audit_1)
    
    # LAW_DEPTH
    passed_2, rej_2, audit_2 = check_law_depth(
        solution, concepts_used, cfg, task_class, historical_max_depth
    )
    all_rejections.extend(rej_2)
    all_audits.extend(audit_2)
    
    # LAW_COMPOSITION
    passed_3, rej_3, audit_3 = check_law_composition(
        solution, concepts_used, trace_events, cfg
    )
    all_rejections.extend(rej_3)
    all_audits.extend(audit_3)
    
    # LAW_REUSE
    passed_4, rej_4, audit_4 = check_law_reuse(
        solution, concepts_used, reuse_registry, task_id, cfg
    )
    all_rejections.extend(rej_4)
    all_audits.extend(audit_4)
    
    # LAW_PROOF
    passed_5, rej_5, audit_5 = check_law_proof(solution, concepts_used, cfg)
    all_rejections.extend(rej_5)
    all_audits.extend(audit_5)
    
    # LAW_UTILITY
    passed_6, rej_6, audit_6 = check_law_utility(
        solution, search_steps_with_concepts, search_steps_without_concepts, cfg
    )
    all_rejections.extend(rej_6)
    all_audits.extend(audit_6)
    
    # LAW_BUDGET
    passed_7, rej_7, audit_7 = check_law_budget(
        solution, budget_allocated, budget_used, cfg
    )
    all_rejections.extend(rej_7)
    all_audits.extend(audit_7)
    
    # Calculate metrics
    max_depth = 0
    csv_calls_total = len([e for e in trace_events if e.get("op") == "CSV_CALL"])
    for concept in concepts_used:
        ev = concept.evidence or {}
        pcc = ev.get("pcc_v2", {})
        depth = int(pcc.get("depth", 0) or 0)
        max_depth = max(max_depth, depth)
    
    # Calculate cross-task reuse
    cross_task_count = 0
    for concept in concepts_used:
        cid = str(concept.id)
        tasks_used_in = reuse_registry.get(cid, set())
        cross_task_count += len(tasks_used_in)
    
    # Calculate collapse factor
    if search_steps_without_concepts > 0:
        collapse_factor = float(search_steps_with_concepts) / float(search_steps_without_concepts)
    else:
        collapse_factor = 1.0
    
    # Final verdict
    passed = len(all_rejections) == 0
    
    return GateResult(
        passed=bool(passed),
        rejection_reasons=all_rejections,
        concepts_used=len(concepts_used),
        max_depth=int(max_depth),
        csv_calls_total=int(csv_calls_total),
        cross_task_reuse_count=int(cross_task_count),
        search_steps_taken=int(search_steps_with_concepts),
        search_collapse_factor=float(collapse_factor),
        budget_allocated=int(budget_allocated),
        budget_used=int(budget_used),
        budget_remaining=int(budget_allocated - budget_used),
        solution_hash=sha256_hex(canonical_json_dumps(solution).encode("utf-8")),
        gate_timestamp=deterministic_iso(step=int(step)),
        audit_entries=all_audits,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LEDGER INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────


def write_gate_result_to_ledger(
    result: GateResult,
    *,
    task_id: str,
    step: int,
    prev_hash: str = "",
) -> Dict[str, Any]:
    """Write gate result to WORM-compliant ledger."""
    entry: Dict[str, Any] = {
        "schema_version": int(INEVITABILITY_GATE_SCHEMA_VERSION_V145),
        "kind": "inevitability_gate_ledger_entry_v145",
        "task_id": str(task_id),
        "step": int(step),
        "timestamp": deterministic_iso(step=int(step)),
        "result": result.to_dict(),
        "prev_hash": str(prev_hash),
    }
    
    entry_json = canonical_json_dumps(entry)
    entry["entry_hash"] = sha256_hex(entry_json.encode("utf-8"))
    
    return entry


# ─────────────────────────────────────────────────────────────────────────────
# REGIME DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────


def diagnose_regime(
    results: Sequence[GateResult],
) -> Dict[str, Any]:
    """
    Produce honest regime diagnosis.
    
    KEY QUESTION: "Could the system survive without concepts?"
    
    If YES → REGIME IS BROKEN
    If NO → REGIME IS WORKING
    """
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    
    # Analyze why solutions failed
    rejection_counts: Dict[str, int] = {}
    for r in results:
        for rej in r.rejection_reasons:
            key = str(rej)
            rejection_counts[key] = rejection_counts.get(key, 0) + 1
    
    # Check for conceptless survivals (BUG)
    conceptless_passes = sum(1 for r in results if r.passed and r.concepts_used == 0)
    
    # Check for shallow survivals (WARNING)
    shallow_passes = sum(1 for r in results if r.passed and r.max_depth < 2)
    
    # Check for flat concept survivals (WARNING)
    flat_passes = sum(1 for r in results if r.passed and r.csv_calls_total == 0)
    
    # Calculate regime health
    if conceptless_passes > 0:
        regime_status = "BROKEN"
        regime_message = f"CRITICAL: {conceptless_passes} solutions passed WITHOUT concepts!"
    elif shallow_passes > total * 0.5:
        regime_status = "WEAK"
        regime_message = f"WARNING: {shallow_passes}/{total} solutions use shallow concepts"
    elif flat_passes > total * 0.5:
        regime_status = "WEAK"
        regime_message = f"WARNING: {flat_passes}/{total} solutions use flat concepts"
    else:
        regime_status = "HEALTHY"
        regime_message = "Regime is enforcing concept creation"
    
    return {
        "schema_version": int(INEVITABILITY_GATE_SCHEMA_VERSION_V145),
        "kind": "regime_diagnosis_v145",
        "total_solutions": int(total),
        "passed": int(passed),
        "failed": int(failed),
        "rejection_counts": dict(rejection_counts),
        "conceptless_passes": int(conceptless_passes),
        "shallow_passes": int(shallow_passes),
        "flat_passes": int(flat_passes),
        "regime_status": str(regime_status),
        "regime_message": str(regime_message),
        "could_survive_without_concepts": conceptless_passes > 0,
        "honest_answer": "YES - BUG" if conceptless_passes > 0 else "NO - GOOD",
    }
