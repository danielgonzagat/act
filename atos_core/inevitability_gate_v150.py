"""
inevitability_gate_v150.py - The Inevitability Gate with Diversity Law.

Extension of V145 gate that integrates the V149 Diversity Law.

NEW SURVIVAL LAW ADDED:
8. LAW_DIVERSITY: Solution must satisfy diversity requirements

This ensures that:
- No single concept dominates (max 40% usage)
- Repeated calls are penalized
- Multiple concept lineages compete
- Ecosystem health is maintained

Schema version: 150
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .act import Act, canonical_json_dumps, deterministic_iso, sha256_hex
from .diversity_law_v149 import (
    ConceptEcosystem,
    DiversityConfig,
    check_solution_diversity,
    enforce_diversity_law,
    should_trigger_alternative_mining,
)
from .inevitability_gate_v145 import (
    INEVITABILITY_GATE_SCHEMA_VERSION_V145,
    GateRejectionReason,
    GateResult,
    InevitabilityConfig,
    check_law_budget,
    check_law_composition,
    check_law_concept,
    check_law_depth,
    check_law_proof,
    check_law_reuse,
    check_law_utility,
)


INEVITABILITY_GATE_SCHEMA_VERSION_V150 = 150


# ─────────────────────────────────────────────────────────────────────────────
# Extended Rejection Reasons
# ─────────────────────────────────────────────────────────────────────────────


class DiversityRejectionReason(str):
    """Rejection reasons specific to diversity violations."""
    
    # LAW_DIVERSITY violations
    DIVERSITY_VIOLATION = "DIVERSITY_VIOLATION"
    CONCEPT_DOMINANCE = "CONCEPT_DOMINANCE"
    EXCESSIVE_REPEATED_CALLS = "EXCESSIVE_REPEATED_CALLS"
    LOW_DIVERSITY_SCORE = "LOW_DIVERSITY_SCORE"
    SINGLE_LINEAGE_ONLY = "SINGLE_LINEAGE_ONLY"


# Alias for compatibility
GateRejectionReasonV150 = DiversityRejectionReason


# ─────────────────────────────────────────────────────────────────────────────
# Extended Gate Result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GateResultV150(GateResult):
    """Extended result with diversity metrics."""
    
    # Diversity metrics
    diversity_score: float = 0.0
    diversity_passed: bool = False
    ecosystem_diversity_score: float = 0.0
    alternative_mining_needed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["schema_version"] = int(INEVITABILITY_GATE_SCHEMA_VERSION_V150)
        base["kind"] = "inevitability_gate_result_v150"
        base["diversity_score"] = float(self.diversity_score)
        base["diversity_passed"] = bool(self.diversity_passed)
        base["ecosystem_diversity_score"] = float(self.ecosystem_diversity_score)
        base["alternative_mining_needed"] = bool(self.alternative_mining_needed)
        return base


# ─────────────────────────────────────────────────────────────────────────────
# Extended Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class InevitabilityConfigV150(InevitabilityConfig):
    """Extended configuration with diversity settings."""
    
    # LAW_DIVERSITY: diversity requirements
    enforce_diversity: bool = True
    diversity_config: DiversityConfig = field(default_factory=DiversityConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["schema_version"] = int(INEVITABILITY_GATE_SCHEMA_VERSION_V150)
        base["kind"] = "inevitability_config_v150"
        base["enforce_diversity"] = bool(self.enforce_diversity)
        base["diversity_config"] = {
            "max_concept_usage_pct": float(self.diversity_config.max_concept_usage_pct),
            "max_repeated_calls": int(self.diversity_config.max_repeated_calls),
            "repeated_call_penalty": float(self.diversity_config.repeated_call_penalty),
            "min_concept_families": int(self.diversity_config.min_concept_families),
            "similarity_threshold": float(self.diversity_config.similarity_threshold),
            "min_diversity_score": float(self.diversity_config.min_diversity_score),
            "enable_lineage_competition": bool(self.diversity_config.enable_lineage_competition),
            "min_lineages": int(self.diversity_config.min_lineages),
        }
        return base


# ─────────────────────────────────────────────────────────────────────────────
# LAW_DIVERSITY CHECK
# ─────────────────────────────────────────────────────────────────────────────


def check_law_diversity(
    solution: Dict[str, Any],
    concepts_used: Sequence[Act],
    ecosystem: ConceptEcosystem,
    config: InevitabilityConfigV150,
) -> tuple[bool, list[str], list[Dict[str, Any]]]:
    """
    LAW_DIVERSITY: Solution must satisfy diversity requirements.
    
    This is the 8th survival law, enforcing ecosystem health.
    """
    rejections: List[str] = []
    audit: List[Dict[str, Any]] = []
    
    if not config.enforce_diversity:
        audit.append({
            "law": "LAW_DIVERSITY",
            "check": "enforcement_status",
            "verdict": "SKIPPED",
            "reason": "diversity enforcement disabled",
        })
        return True, rejections, audit
    
    # Get concept IDs from Acts
    concept_ids = [str(c.id) for c in concepts_used]
    
    # Check solution diversity
    passed, diversity_audit = enforce_diversity_law(
        concept_ids, ecosystem, config.diversity_config
    )
    
    audit.append({
        "law": "LAW_DIVERSITY",
        "check": "solution_diversity",
        "passed": bool(passed),
        "score": float(diversity_audit.get("solution_check", {}).get("score", 0.0)),
        "violations": list(diversity_audit.get("solution_check", {}).get("violations", [])),
        "warnings": list(diversity_audit.get("solution_check", {}).get("warnings", [])),
    })
    
    if not passed:
        # Determine specific rejection reason
        violations = diversity_audit.get("solution_check", {}).get("violations", [])
        
        for v in violations:
            v_type = v.get("type", "")
            if v_type == "REPEATED_CALLS_EXCEEDED":
                rejections.append(DiversityRejectionReason.EXCESSIVE_REPEATED_CALLS)
            elif v_type in ("CONCEPT_DOMINANCE", "DOMINANT_CONCEPT_USED"):
                rejections.append(DiversityRejectionReason.CONCEPT_DOMINANCE)
            else:
                rejections.append(DiversityRejectionReason.DIVERSITY_VIOLATION)
        
        if len(rejections) == 0:
            # Generic diversity failure
            rejections.append(DiversityRejectionReason.LOW_DIVERSITY_SCORE)
    
    return passed, rejections, audit


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED INEVITABILITY GATE
# ─────────────────────────────────────────────────────────────────────────────


def pass_inevitability_gate_v150(
    solution: Dict[str, Any],
    concepts_used: Sequence[Act],
    trace_events: Sequence[Dict[str, Any]],
    reuse_registry: Optional[Dict[str, set]] = None,
    task_id: str = "unknown",
    task_class: str = "default",
    historical_max_depth: int = 0,
    search_steps_with_concepts: int = 0,
    search_steps_without_concepts: int = 1,
    budget_allocated: int = 1000,
    budget_used: int = 0,
    config: Optional[InevitabilityConfigV150] = None,
    ecosystem: Optional[ConceptEcosystem] = None,
    step: int = 0,
) -> GateResultV150:
    """
    THE INEVITABILITY GATE V150 - WITH DIVERSITY LAW.
    
    This extends V145 with the 8th survival law: LAW_DIVERSITY.
    
    A solution is REJECTED if:
    1. It doesn't use concepts (LAW_CONCEPT)
    2. Concepts are too shallow (LAW_DEPTH)
    3. Concepts have no composition (LAW_COMPOSITION)
    4. Concepts aren't reused cross-task (LAW_REUSE)
    5. Concepts lack proof (LAW_PROOF)
    6. Concepts don't collapse search (LAW_UTILITY)
    7. Solution exceeds budget (LAW_BUDGET)
    8. Solution violates diversity (LAW_DIVERSITY) [NEW]
    
    Returns GateResultV150 with pass/fail decision and full audit trail.
    """
    cfg = config or InevitabilityConfigV150()
    eco = ecosystem or ConceptEcosystem()
    reuse_registry = reuse_registry or {}
    
    # Validate config first
    config_errors = cfg.validate_config()
    if config_errors:
        return GateResultV150(
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
            diversity_score=0.0,
            diversity_passed=False,
            ecosystem_diversity_score=0.0,
            alternative_mining_needed=False,
        )
    
    all_rejections: List[GateRejectionReason] = []
    all_audits: List[Dict[str, Any]] = []
    
    # LAW_CONCEPT (Law 1)
    passed_1, rej_1, audit_1 = check_law_concept(solution, concepts_used, cfg)
    all_rejections.extend(rej_1)
    all_audits.extend(audit_1)
    
    # LAW_DEPTH (Law 2)
    passed_2, rej_2, audit_2 = check_law_depth(
        solution, concepts_used, cfg, task_class, historical_max_depth
    )
    all_rejections.extend(rej_2)
    all_audits.extend(audit_2)
    
    # LAW_COMPOSITION (Law 3)
    passed_3, rej_3, audit_3 = check_law_composition(
        solution, concepts_used, trace_events, cfg
    )
    all_rejections.extend(rej_3)
    all_audits.extend(audit_3)
    
    # LAW_REUSE (Law 4)
    passed_4, rej_4, audit_4 = check_law_reuse(
        solution, concepts_used, reuse_registry, task_id, cfg
    )
    all_rejections.extend(rej_4)
    all_audits.extend(audit_4)
    
    # LAW_PROOF (Law 5)
    passed_5, rej_5, audit_5 = check_law_proof(solution, concepts_used, cfg)
    all_rejections.extend(rej_5)
    all_audits.extend(audit_5)
    
    # LAW_UTILITY (Law 6)
    passed_6, rej_6, audit_6 = check_law_utility(
        solution, search_steps_with_concepts, search_steps_without_concepts, cfg
    )
    all_rejections.extend(rej_6)
    all_audits.extend(audit_6)
    
    # LAW_BUDGET (Law 7)
    passed_7, rej_7, audit_7 = check_law_budget(
        solution, budget_allocated, budget_used, cfg
    )
    all_rejections.extend(rej_7)
    all_audits.extend(audit_7)
    
    # LAW_DIVERSITY (Law 8) - NEW
    passed_8, rej_8, audit_8 = check_law_diversity(
        solution, concepts_used, eco, cfg
    )
    all_rejections.extend(rej_8)
    all_audits.extend(audit_8)
    
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
    
    # Diversity metrics
    diversity_score = 0.0
    for entry in all_audits:
        if entry.get("law") == "LAW_DIVERSITY" and "score" in entry:
            diversity_score = float(entry["score"])
            break
    
    # Check if alternative mining is needed
    mining_request = should_trigger_alternative_mining(eco, cfg.diversity_config)
    
    # Final verdict
    passed = len(all_rejections) == 0
    
    return GateResultV150(
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
        diversity_score=float(diversity_score),
        diversity_passed=bool(passed_8),
        ecosystem_diversity_score=float(eco.get_diversity_score()),
        alternative_mining_needed=bool(mining_request is not None),
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED REGIME DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────


def diagnose_regime_v150(
    results: Sequence[GateResultV150],
) -> Dict[str, Any]:
    """
    Extended regime diagnosis with diversity metrics.
    
    KEY QUESTIONS:
    1. "Could the system survive without concepts?" → Should be NO
    2. "Could one concept dominate?" → Should be NO
    """
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    
    # Analyze rejection reasons
    rejection_counts: Dict[str, int] = {}
    for r in results:
        for rej in r.rejection_reasons:
            key = str(rej)
            rejection_counts[key] = rejection_counts.get(key, 0) + 1
    
    # Conceptless survivals (BUG)
    conceptless_passes = sum(1 for r in results if r.passed and r.concepts_used == 0)
    
    # Shallow survivals (WARNING)
    shallow_passes = sum(1 for r in results if r.passed and r.max_depth < 2)
    
    # Flat concept survivals (WARNING)
    flat_passes = sum(1 for r in results if r.passed and r.csv_calls_total == 0)
    
    # Diversity failures (NEW)
    diversity_failures = sum(1 for r in results if not r.diversity_passed)
    avg_diversity_score = (
        sum(r.diversity_score for r in results) / total if total > 0 else 0.0
    )
    
    # Alternative mining needed
    mining_needed = sum(1 for r in results if r.alternative_mining_needed)
    
    # Calculate regime health
    if conceptless_passes > 0:
        regime_status = "BROKEN"
        regime_message = f"CRITICAL: {conceptless_passes} solutions passed WITHOUT concepts!"
    elif diversity_failures > total * 0.5:
        regime_status = "WEAK_DIVERSITY"
        regime_message = f"WARNING: {diversity_failures}/{total} solutions fail diversity"
    elif shallow_passes > total * 0.5:
        regime_status = "WEAK"
        regime_message = f"WARNING: {shallow_passes}/{total} solutions use shallow concepts"
    elif flat_passes > total * 0.5:
        regime_status = "WEAK"
        regime_message = f"WARNING: {flat_passes}/{total} solutions use flat concepts"
    elif mining_needed > total * 0.3:
        regime_status = "NEEDS_DIVERSIFICATION"
        regime_message = f"INFO: {mining_needed}/{total} solutions need alternative mining"
    else:
        regime_status = "HEALTHY"
        regime_message = "Regime is enforcing concept creation with diversity"
    
    return {
        "schema_version": int(INEVITABILITY_GATE_SCHEMA_VERSION_V150),
        "kind": "regime_diagnosis_v150",
        "total_solutions": int(total),
        "passed": int(passed),
        "failed": int(failed),
        "rejection_counts": dict(rejection_counts),
        "conceptless_passes": int(conceptless_passes),
        "shallow_passes": int(shallow_passes),
        "flat_passes": int(flat_passes),
        "diversity_failures": int(diversity_failures),
        "avg_diversity_score": float(avg_diversity_score),
        "mining_needed_count": int(mining_needed),
        "regime_status": str(regime_status),
        "regime_message": str(regime_message),
        "could_survive_without_concepts": conceptless_passes > 0,
        "could_one_concept_dominate": avg_diversity_score < 0.3,
        "honest_answer_concepts": "YES - BUG" if conceptless_passes > 0 else "NO - GOOD",
        "honest_answer_diversity": "YES - RISK" if avg_diversity_score < 0.3 else "NO - GOOD",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────


__all__ = [
    "INEVITABILITY_GATE_SCHEMA_VERSION_V150",
    "GateRejectionReasonV150",
    "GateResultV150",
    "InevitabilityConfigV150",
    "check_law_diversity",
    "pass_inevitability_gate_v150",
    "diagnose_regime_v150",
]
