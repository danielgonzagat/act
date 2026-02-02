"""
agi_regime_v144.py - AGI Regime: Survival Laws for Deterministic AGI.

This module implements the 7 Survival Laws that force AGI emergence by making
it impossible to solve tasks without concepts, planning, and composition.

Design principles:
- Survival Law: if a task requires utility, it CANNOT be solved without concepts
- Inevitability: no escape path - concept creation/usage is mandatory
- Deterministic: all checks are explicit, auditable, reproducible
- Fail-closed: UNKNOWN on ambiguity, FAIL on missing concept
- WORM-compliant: all decisions are logged with hashes

The 7 Survival Laws:
1. LAW_CONCEPT: No fallback - utility tasks require concept_csv execution
2. LAW_DEPTH: Plan tasks require concepts with depth >= concept_min_depth
3. LAW_COMPOSITION: Solver must use CSV_CALL chains, not just primitives
4. LAW_REUSE: Concepts must be used in multiple contexts to survive
5. LAW_PROOF: Concepts only enter the world if they pass PCC + hashes
6. LAW_UTILITY: Training bottleneck is utility, not fluency
7. LAW_BUDGET: Search cannot explode; must collapse via concepts

Schema version: 144
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .act import Act, canonical_json_dumps, deterministic_iso, sha256_hex


AGI_REGIME_SCHEMA_VERSION_V144 = 144


# ─────────────────────────────────────────────────────────────────────────────
# Survival Law Enumeration
# ─────────────────────────────────────────────────────────────────────────────


class SurvivalLaw(Enum):
    """The 7 Survival Laws that force AGI emergence."""
    LAW_CONCEPT = "LAW_CONCEPT"           # No fallback - require concept execution
    LAW_DEPTH = "LAW_DEPTH"               # Require concept depth >= min_depth
    LAW_COMPOSITION = "LAW_COMPOSITION"   # Require CSV_CALL chains
    LAW_REUSE = "LAW_REUSE"               # Cross-context reuse required
    LAW_PROOF = "LAW_PROOF"               # PCC + hashes required for promotion
    LAW_UTILITY = "LAW_UTILITY"           # Utility is bottleneck, not fluency
    LAW_BUDGET = "LAW_BUDGET"             # Search budget collapses via concepts


# ─────────────────────────────────────────────────────────────────────────────
# Concept Lifecycle States (for LAW_REUSE)
# ─────────────────────────────────────────────────────────────────────────────


class ConceptState(Enum):
    """Lifecycle states for concept survival."""
    CANDIDATE = "candidate"       # Newly mined, not yet proven
    PROMOTED = "promoted"         # Proven, passing PCC, cross-context reuse
    QUARANTINED = "quarantined"  # Failed recently, under probation
    PRUNED = "pruned"            # Dead - removed from system


@dataclass
class ConceptMetrics:
    """Metrics for concept lifecycle management."""
    concept_id: str
    state: ConceptState = ConceptState.CANDIDATE
    created_at_step: int = 0
    
    # Usage metrics
    total_uses: int = 0
    successful_uses: int = 0
    failed_uses: int = 0
    
    # Cross-context metrics (for LAW_REUSE)
    contexts_used: Set[str] = field(default_factory=set)
    families_used: Set[str] = field(default_factory=set)
    
    # Composition metrics (for LAW_COMPOSITION)
    max_call_depth: int = 0
    total_csv_calls: int = 0
    
    # Cost metrics
    total_cost_bits: int = 0
    avg_cost_bits: float = 0.0
    
    # Proof metrics (for LAW_PROOF)
    has_pcc: bool = False
    pcc_hash: str = ""
    proof_timestamp: str = ""
    
    # Survival metrics
    last_used_step: int = 0
    consecutive_failures: int = 0
    reuse_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(AGI_REGIME_SCHEMA_VERSION_V144),
            "concept_id": str(self.concept_id),
            "state": str(self.state.value),
            "created_at_step": int(self.created_at_step),
            "total_uses": int(self.total_uses),
            "successful_uses": int(self.successful_uses),
            "failed_uses": int(self.failed_uses),
            "contexts_used": list(sorted(self.contexts_used)),
            "families_used": list(sorted(self.families_used)),
            "max_call_depth": int(self.max_call_depth),
            "total_csv_calls": int(self.total_csv_calls),
            "total_cost_bits": int(self.total_cost_bits),
            "avg_cost_bits": float(self.avg_cost_bits),
            "has_pcc": bool(self.has_pcc),
            "pcc_hash": str(self.pcc_hash),
            "proof_timestamp": str(self.proof_timestamp),
            "last_used_step": int(self.last_used_step),
            "consecutive_failures": int(self.consecutive_failures),
            "reuse_score": float(self.reuse_score),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Survival Law Validators
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LawValidationResult:
    """Result of validating a survival law."""
    law: SurvivalLaw
    passed: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "law": str(self.law.value),
            "passed": bool(self.passed),
            "reason": str(self.reason),
            "details": dict(self.details),
        }


def validate_law_concept(
    *,
    trace: Dict[str, Any],
    task: Dict[str, Any],
) -> LawValidationResult:
    """
    LAW_CONCEPT: Utility tasks MUST execute a concept_csv.
    
    No global fallback allowed. If concept_policy_required=True and
    concept_executor.used=False, the task FAILS.
    """
    law = SurvivalLaw.LAW_CONCEPT
    
    # Check if this task requires concept policy
    concept_required = bool(task.get("concept_policy_required", False))
    if not concept_required:
        return LawValidationResult(
            law=law,
            passed=True,
            reason="concept_policy_not_required",
            details={"concept_policy_required": False},
        )
    
    # Check if concept was executed
    concept_exec = trace.get("concept_executor", {})
    if not isinstance(concept_exec, dict):
        concept_exec = {}
    
    used = bool(concept_exec.get("used", False))
    ok = bool(concept_exec.get("ok", False))
    
    if not used:
        return LawValidationResult(
            law=law,
            passed=False,
            reason="concept_not_executed",
            details={
                "concept_policy_required": True,
                "concept_executor_used": False,
                "violation": "LAW_CONCEPT requires concept execution for utility tasks",
            },
        )
    
    if not ok:
        return LawValidationResult(
            law=law,
            passed=False,
            reason="concept_execution_failed",
            details={
                "concept_policy_required": True,
                "concept_executor_used": True,
                "concept_executor_ok": False,
                "violation": "LAW_CONCEPT requires successful concept execution",
            },
        )
    
    return LawValidationResult(
        law=law,
        passed=True,
        reason="concept_executed_successfully",
        details={
            "concept_policy_required": True,
            "concept_executor_used": True,
            "concept_executor_ok": True,
        },
    )


def validate_law_depth(
    *,
    trace: Dict[str, Any],
    task: Dict[str, Any],
    min_depth: int = 1,
) -> LawValidationResult:
    """
    LAW_DEPTH: Plan tasks require concepts with call depth >= min_depth.
    
    This forces hierarchical composition - no flat primitive sequences.
    """
    law = SurvivalLaw.LAW_DEPTH
    
    # Check if this is a plan task that requires depth
    required_depth = int(task.get("concept_min_depth", 0) or 0)
    if required_depth <= 0:
        required_depth = int(min_depth)
    
    validator_id = str(task.get("validator_id", ""))
    if validator_id != "plan_validator" and required_depth <= 0:
        return LawValidationResult(
            law=law,
            passed=True,
            reason="depth_not_required",
            details={"concept_min_depth": 0},
        )
    
    # Get actual depth from trace
    concept_exec = trace.get("concept_executor", {})
    if not isinstance(concept_exec, dict):
        concept_exec = {}
    
    actual_depth = int(concept_exec.get("max_depth", 0) or 0)
    calls_total = int(concept_exec.get("calls_total", 0) or 0)
    
    if actual_depth < required_depth:
        return LawValidationResult(
            law=law,
            passed=False,
            reason="insufficient_depth",
            details={
                "required_depth": int(required_depth),
                "actual_depth": int(actual_depth),
                "calls_total": int(calls_total),
                "violation": f"LAW_DEPTH requires depth >= {required_depth}, got {actual_depth}",
            },
        )
    
    return LawValidationResult(
        law=law,
        passed=True,
        reason="depth_satisfied",
        details={
            "required_depth": int(required_depth),
            "actual_depth": int(actual_depth),
            "calls_total": int(calls_total),
        },
    )


def validate_law_composition(
    *,
    trace: Dict[str, Any],
    task: Dict[str, Any],
    min_csv_calls: int = 1,
) -> LawValidationResult:
    """
    LAW_COMPOSITION: Solver must use CSV_CALL chains, not just primitives.
    
    This forces modular decomposition and concept reuse.
    """
    law = SurvivalLaw.LAW_COMPOSITION
    
    # Check if composition is required
    required_calls = int(task.get("concept_min_csv_calls", 0) or 0)
    if required_calls <= 0:
        required_calls = int(min_csv_calls) if bool(task.get("concept_policy_required")) else 0
    
    if required_calls <= 0:
        return LawValidationResult(
            law=law,
            passed=True,
            reason="composition_not_required",
            details={"required_csv_calls": 0},
        )
    
    # Get actual calls from trace
    concept_exec = trace.get("concept_executor", {})
    if not isinstance(concept_exec, dict):
        concept_exec = {}
    
    actual_calls = int(concept_exec.get("calls_total", 0) or 0)
    
    if actual_calls < required_calls:
        return LawValidationResult(
            law=law,
            passed=False,
            reason="insufficient_composition",
            details={
                "required_csv_calls": int(required_calls),
                "actual_csv_calls": int(actual_calls),
                "violation": f"LAW_COMPOSITION requires >= {required_calls} CSV_CALLs, got {actual_calls}",
            },
        )
    
    return LawValidationResult(
        law=law,
        passed=True,
        reason="composition_satisfied",
        details={
            "required_csv_calls": int(required_calls),
            "actual_csv_calls": int(actual_calls),
        },
    )


def validate_law_budget(
    *,
    trace: Dict[str, Any],
    task: Dict[str, Any],
    max_search_steps: int = 1000,
) -> LawValidationResult:
    """
    LAW_BUDGET: Search cannot explode; must collapse via concept reuse.
    
    If search exceeds budget without finding a concept-based solution, FAIL.
    """
    law = SurvivalLaw.LAW_BUDGET
    
    search_steps = int(trace.get("search_steps", 0) or 0)
    budget = int(task.get("search_budget", max_search_steps) or max_search_steps)
    
    if search_steps > budget:
        return LawValidationResult(
            law=law,
            passed=False,
            reason="budget_exceeded",
            details={
                "search_steps": int(search_steps),
                "budget": int(budget),
                "violation": f"LAW_BUDGET: search exceeded budget ({search_steps} > {budget})",
            },
        )
    
    return LawValidationResult(
        law=law,
        passed=True,
        reason="within_budget",
        details={
            "search_steps": int(search_steps),
            "budget": int(budget),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Regime Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AGIRegimeConfig:
    """Configuration for AGI Regime enforcement."""
    
    # Which laws are enabled
    enable_law_concept: bool = True
    enable_law_depth: bool = True
    enable_law_composition: bool = True
    enable_law_reuse: bool = True
    enable_law_proof: bool = True
    enable_law_utility: bool = True
    enable_law_budget: bool = True
    
    # Law parameters
    default_min_depth: int = 1
    default_min_csv_calls: int = 1
    default_search_budget: int = 1000
    
    # Lifecycle parameters
    reuse_window_steps: int = 1000
    min_contexts_for_promotion: int = 2
    min_families_for_promotion: int = 2
    max_consecutive_failures: int = 3
    quarantine_duration_steps: int = 500
    
    # Utility bottleneck
    utility_weight: float = 1.0
    fluency_weight: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(AGI_REGIME_SCHEMA_VERSION_V144),
            "kind": "agi_regime_config_v144",
            "enable_law_concept": bool(self.enable_law_concept),
            "enable_law_depth": bool(self.enable_law_depth),
            "enable_law_composition": bool(self.enable_law_composition),
            "enable_law_reuse": bool(self.enable_law_reuse),
            "enable_law_proof": bool(self.enable_law_proof),
            "enable_law_utility": bool(self.enable_law_utility),
            "enable_law_budget": bool(self.enable_law_budget),
            "default_min_depth": int(self.default_min_depth),
            "default_min_csv_calls": int(self.default_min_csv_calls),
            "default_search_budget": int(self.default_search_budget),
            "reuse_window_steps": int(self.reuse_window_steps),
            "min_contexts_for_promotion": int(self.min_contexts_for_promotion),
            "min_families_for_promotion": int(self.min_families_for_promotion),
            "max_consecutive_failures": int(self.max_consecutive_failures),
            "quarantine_duration_steps": int(self.quarantine_duration_steps),
            "utility_weight": float(self.utility_weight),
            "fluency_weight": float(self.fluency_weight),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Regime Validator (Main Entry Point)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RegimeValidationResult:
    """Aggregated result of all survival law validations."""
    passed: bool
    laws_checked: int
    laws_passed: int
    laws_failed: int
    results: List[LawValidationResult]
    failure_reasons: List[str]
    config: AGIRegimeConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(AGI_REGIME_SCHEMA_VERSION_V144),
            "kind": "regime_validation_result_v144",
            "passed": bool(self.passed),
            "laws_checked": int(self.laws_checked),
            "laws_passed": int(self.laws_passed),
            "laws_failed": int(self.laws_failed),
            "results": [r.to_dict() for r in self.results],
            "failure_reasons": list(self.failure_reasons),
            "config": self.config.to_dict(),
        }


def validate_survival_laws(
    *,
    trace: Dict[str, Any],
    task: Dict[str, Any],
    config: Optional[AGIRegimeConfig] = None,
) -> RegimeValidationResult:
    """
    Validate all enabled survival laws for a task execution.
    
    This is the main entry point for regime enforcement.
    Returns a comprehensive result with all law validations.
    """
    cfg = config or AGIRegimeConfig()
    results: List[LawValidationResult] = []
    failure_reasons: List[str] = []
    
    # LAW_CONCEPT
    if cfg.enable_law_concept:
        r = validate_law_concept(trace=trace, task=task)
        results.append(r)
        if not r.passed:
            failure_reasons.append(f"{r.law.value}: {r.reason}")
    
    # LAW_DEPTH
    if cfg.enable_law_depth:
        r = validate_law_depth(
            trace=trace,
            task=task,
            min_depth=int(cfg.default_min_depth),
        )
        results.append(r)
        if not r.passed:
            failure_reasons.append(f"{r.law.value}: {r.reason}")
    
    # LAW_COMPOSITION
    if cfg.enable_law_composition:
        r = validate_law_composition(
            trace=trace,
            task=task,
            min_csv_calls=int(cfg.default_min_csv_calls),
        )
        results.append(r)
        if not r.passed:
            failure_reasons.append(f"{r.law.value}: {r.reason}")
    
    # LAW_BUDGET
    if cfg.enable_law_budget:
        r = validate_law_budget(
            trace=trace,
            task=task,
            max_search_steps=int(cfg.default_search_budget),
        )
        results.append(r)
        if not r.passed:
            failure_reasons.append(f"{r.law.value}: {r.reason}")
    
    laws_checked = len(results)
    laws_passed = sum(1 for r in results if r.passed)
    laws_failed = laws_checked - laws_passed
    passed = laws_failed == 0
    
    return RegimeValidationResult(
        passed=bool(passed),
        laws_checked=int(laws_checked),
        laws_passed=int(laws_passed),
        laws_failed=int(laws_failed),
        results=results,
        failure_reasons=failure_reasons,
        config=cfg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Concept Lifecycle Manager (for LAW_REUSE)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConceptLifecycleManager:
    """
    Manages concept lifecycle: candidate → promoted → quarantined → pruned.
    
    Implements LAW_REUSE: concepts must be used in multiple contexts to survive.
    """
    config: AGIRegimeConfig = field(default_factory=AGIRegimeConfig)
    concepts: Dict[str, ConceptMetrics] = field(default_factory=dict)
    current_step: int = 0

    def register_concept(
        self,
        *,
        concept_id: str,
        step: int,
        has_pcc: bool = False,
        pcc_hash: str = "",
    ) -> ConceptMetrics:
        """Register a new candidate concept."""
        metrics = ConceptMetrics(
            concept_id=str(concept_id),
            state=ConceptState.CANDIDATE,
            created_at_step=int(step),
            has_pcc=bool(has_pcc),
            pcc_hash=str(pcc_hash),
            proof_timestamp=deterministic_iso(step=int(step)) if has_pcc else "",
        )
        self.concepts[str(concept_id)] = metrics
        return metrics

    def record_usage(
        self,
        *,
        concept_id: str,
        step: int,
        success: bool,
        context_id: str,
        family_id: str,
        call_depth: int = 0,
        csv_calls: int = 0,
        cost_bits: int = 0,
    ) -> Optional[ConceptMetrics]:
        """Record a concept usage and update metrics."""
        cid = str(concept_id)
        if cid not in self.concepts:
            return None
        
        m = self.concepts[cid]
        m.total_uses += 1
        m.last_used_step = int(step)
        
        if success:
            m.successful_uses += 1
            m.consecutive_failures = 0
        else:
            m.failed_uses += 1
            m.consecutive_failures += 1
        
        m.contexts_used.add(str(context_id))
        m.families_used.add(str(family_id))
        
        if call_depth > m.max_call_depth:
            m.max_call_depth = int(call_depth)
        m.total_csv_calls += int(csv_calls)
        m.total_cost_bits += int(cost_bits)
        
        if m.total_uses > 0:
            m.avg_cost_bits = float(m.total_cost_bits) / float(m.total_uses)
        
        # Update reuse score
        m.reuse_score = self._compute_reuse_score(m)
        
        return m

    def _compute_reuse_score(self, m: ConceptMetrics) -> float:
        """Compute reuse score for concept lifecycle decisions."""
        if m.total_uses == 0:
            return 0.0
        
        success_rate = float(m.successful_uses) / float(m.total_uses)
        context_diversity = len(m.contexts_used)
        family_diversity = len(m.families_used)
        
        # Score = success_rate * sqrt(contexts) * sqrt(families)
        import math
        return float(success_rate * math.sqrt(context_diversity) * math.sqrt(family_diversity))

    def update_lifecycle(self, *, step: int) -> Dict[str, List[str]]:
        """
        Update concept lifecycle states based on survival laws.
        
        Returns dict with lists of concepts that changed state.
        """
        self.current_step = int(step)
        changes: Dict[str, List[str]] = {
            "promoted": [],
            "quarantined": [],
            "pruned": [],
            "recovered": [],
        }
        
        for cid, m in list(self.concepts.items()):
            old_state = m.state
            new_state = self._evaluate_state(m, step)
            
            if new_state != old_state:
                m.state = new_state
                if new_state == ConceptState.PROMOTED:
                    changes["promoted"].append(cid)
                elif new_state == ConceptState.QUARANTINED:
                    changes["quarantined"].append(cid)
                elif new_state == ConceptState.PRUNED:
                    changes["pruned"].append(cid)
                elif old_state == ConceptState.QUARANTINED and new_state == ConceptState.CANDIDATE:
                    changes["recovered"].append(cid)
        
        return changes

    def _evaluate_state(self, m: ConceptMetrics, step: int) -> ConceptState:
        """Evaluate what state a concept should be in based on metrics."""
        cfg = self.config
        
        # Check for pruning conditions
        if m.state == ConceptState.QUARANTINED:
            quarantine_end = m.last_used_step + cfg.quarantine_duration_steps
            if step > quarantine_end:
                # Still quarantined after duration → prune
                return ConceptState.PRUNED
        
        # Check for too many consecutive failures
        if m.consecutive_failures >= cfg.max_consecutive_failures:
            if m.state == ConceptState.QUARANTINED:
                return ConceptState.PRUNED
            return ConceptState.QUARANTINED
        
        # Check if unused for too long (LAW_REUSE)
        if m.state != ConceptState.PRUNED:
            steps_since_use = step - m.last_used_step
            if steps_since_use > cfg.reuse_window_steps:
                if m.state == ConceptState.QUARANTINED:
                    return ConceptState.PRUNED
                return ConceptState.QUARANTINED
        
        # Check for promotion conditions
        if m.state == ConceptState.CANDIDATE:
            contexts_ok = len(m.contexts_used) >= cfg.min_contexts_for_promotion
            families_ok = len(m.families_used) >= cfg.min_families_for_promotion
            proof_ok = m.has_pcc if cfg.enable_law_proof else True
            
            if contexts_ok and families_ok and proof_ok:
                return ConceptState.PROMOTED
        
        return m.state

    def get_active_concepts(self) -> List[str]:
        """Get list of active (candidate or promoted) concept IDs."""
        return [
            cid for cid, m in self.concepts.items()
            if m.state in (ConceptState.CANDIDATE, ConceptState.PROMOTED)
        ]

    def get_promoted_concepts(self) -> List[str]:
        """Get list of promoted concept IDs."""
        return [
            cid for cid, m in self.concepts.items()
            if m.state == ConceptState.PROMOTED
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(AGI_REGIME_SCHEMA_VERSION_V144),
            "kind": "concept_lifecycle_manager_v144",
            "current_step": int(self.current_step),
            "config": self.config.to_dict(),
            "concepts": {cid: m.to_dict() for cid, m in self.concepts.items()},
            "active_count": len(self.get_active_concepts()),
            "promoted_count": len(self.get_promoted_concepts()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Task Transformation (Apply Regime to Tasks)
# ─────────────────────────────────────────────────────────────────────────────


def apply_regime_to_tasks(
    tasks: Sequence[Dict[str, Any]],
    *,
    config: Optional[AGIRegimeConfig] = None,
) -> Tuple[Dict[str, Any], ...]:
    """
    Transform tasks to enforce AGI regime survival laws.
    
    This is the key function that "turns on" the regime by adding
    concept_policy_required, concept_min_depth, etc. to tasks.
    """
    cfg = config or AGIRegimeConfig()
    out: List[Dict[str, Any]] = []
    
    for t in tasks:
        if not isinstance(t, dict):
            continue
        
        t2 = dict(t)
        
        # LAW_CONCEPT: Force concept_policy_required for utility tasks
        if cfg.enable_law_concept:
            validator_id = str(t2.get("validator_id", ""))
            # Utility validators that require deterministic output
            utility_validators = {
                "plan_validator",
                "json_obj_exact",
                "text_exact",
                "grid_exact",
                "arc_grid_exact",
            }
            if validator_id in utility_validators:
                t2["concept_policy_required"] = True
        
        # LAW_DEPTH: Set minimum depth for plan tasks
        if cfg.enable_law_depth:
            if str(t2.get("validator_id", "")) == "plan_validator":
                current_depth = int(t2.get("concept_min_depth", 0) or 0)
                if current_depth < cfg.default_min_depth:
                    t2["concept_min_depth"] = int(cfg.default_min_depth)
        
        # LAW_COMPOSITION: Set minimum CSV_CALL count
        if cfg.enable_law_composition:
            if bool(t2.get("concept_policy_required")):
                current_calls = int(t2.get("concept_min_csv_calls", 0) or 0)
                if current_calls < cfg.default_min_csv_calls:
                    t2["concept_min_csv_calls"] = int(cfg.default_min_csv_calls)
        
        # LAW_BUDGET: Set search budget
        if cfg.enable_law_budget:
            if "search_budget" not in t2:
                t2["search_budget"] = int(cfg.default_search_budget)
        
        out.append(t2)
    
    return tuple(out)


# ─────────────────────────────────────────────────────────────────────────────
# Utility Bottleneck Loss (for LAW_UTILITY)
# ─────────────────────────────────────────────────────────────────────────────


def compute_regime_loss(
    *,
    validation_result: RegimeValidationResult,
    utility_pass_rate: float,
    fluency_score: float,
    config: Optional[AGIRegimeConfig] = None,
) -> Dict[str, Any]:
    """
    Compute loss with utility as bottleneck (LAW_UTILITY).
    
    Loss = utility_weight * (1 - utility_pass_rate) + fluency_weight * (1 - fluency_score)
    
    But if any survival law fails, loss = infinity (FAIL).
    """
    cfg = config or AGIRegimeConfig()
    
    # If any survival law failed, loss is infinite
    if not validation_result.passed:
        return {
            "schema_version": int(AGI_REGIME_SCHEMA_VERSION_V144),
            "kind": "regime_loss_v144",
            "loss": float("inf"),
            "utility_loss": float("inf"),
            "fluency_loss": float(1.0 - fluency_score),
            "survival_law_violation": True,
            "failure_reasons": list(validation_result.failure_reasons),
        }
    
    utility_loss = float(cfg.utility_weight) * (1.0 - float(utility_pass_rate))
    fluency_loss = float(cfg.fluency_weight) * (1.0 - float(fluency_score))
    total_loss = utility_loss + fluency_loss
    
    return {
        "schema_version": int(AGI_REGIME_SCHEMA_VERSION_V144),
        "kind": "regime_loss_v144",
        "loss": float(total_loss),
        "utility_loss": float(utility_loss),
        "fluency_loss": float(fluency_loss),
        "survival_law_violation": False,
        "utility_pass_rate": float(utility_pass_rate),
        "fluency_score": float(fluency_score),
    }


# ─────────────────────────────────────────────────────────────────────────────
# WORM-Compliant Ledger Integration
# ─────────────────────────────────────────────────────────────────────────────


def write_regime_validation_to_ledger(
    *,
    result: RegimeValidationResult,
    task_id: str,
    step: int,
    ledger_path: str,
    prev_hash: str = "",
) -> str:
    """
    Write regime validation result to WORM-compliant ledger.
    
    Returns the hash of the new ledger entry.
    """
    entry = {
        "schema_version": int(AGI_REGIME_SCHEMA_VERSION_V144),
        "kind": "regime_validation_ledger_entry_v144",
        "task_id": str(task_id),
        "step": int(step),
        "timestamp": deterministic_iso(step=int(step)),
        "result": result.to_dict(),
        "prev_hash": str(prev_hash),
    }
    
    entry_hash = sha256_hex(canonical_json_dumps(entry).encode("utf-8"))
    entry["entry_hash"] = str(entry_hash)
    
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    
    return str(entry_hash)


# ─────────────────────────────────────────────────────────────────────────────
# Regime Switch: The "turn on AGI regime" function
# ─────────────────────────────────────────────────────────────────────────────


def create_agi_regime_tasks(
    base_tasks: Sequence[Dict[str, Any]],
    *,
    regime_level: str = "full",
) -> Tuple[Dict[str, Any], ...]:
    """
    Create task pack with AGI regime enforced.
    
    regime_level:
    - "bootstrap": Minimal regime (LAW_CONCEPT only)
    - "intermediate": LAW_CONCEPT + LAW_DEPTH
    - "full": All 7 survival laws enabled
    
    This is THE SWITCH that changes the regime.
    """
    if regime_level == "bootstrap":
        config = AGIRegimeConfig(
            enable_law_concept=True,
            enable_law_depth=False,
            enable_law_composition=False,
            enable_law_reuse=False,
            enable_law_proof=False,
            enable_law_utility=True,
            enable_law_budget=True,
            default_min_depth=0,
            default_min_csv_calls=0,
        )
    elif regime_level == "intermediate":
        config = AGIRegimeConfig(
            enable_law_concept=True,
            enable_law_depth=True,
            enable_law_composition=False,
            enable_law_reuse=False,
            enable_law_proof=False,
            enable_law_utility=True,
            enable_law_budget=True,
            default_min_depth=1,
            default_min_csv_calls=0,
        )
    else:  # "full"
        config = AGIRegimeConfig(
            enable_law_concept=True,
            enable_law_depth=True,
            enable_law_composition=True,
            enable_law_reuse=True,
            enable_law_proof=True,
            enable_law_utility=True,
            enable_law_budget=True,
            default_min_depth=2,
            default_min_csv_calls=1,
        )
    
    return apply_regime_to_tasks(base_tasks, config=config)
