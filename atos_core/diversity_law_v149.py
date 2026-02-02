"""
diversity_law_v149.py - Lei de Diversidade Conceitual

Impede que um único conceito domine o sistema AGI.

Princípios:
1. Nenhum conceito pode ser usado em mais de X% das soluções
2. Chamadas repetidas do mesmo conceito são penalizadas
3. O miner deve propor conceitos alternativos
4. Múltiplas "linhagens conceituais" devem competir
5. Conceitos muito similares devem ser fundidos ou um eliminado

AGI ≠ um conceito bom
AGI = ecossistema conceitual diverso

Schema version: 149
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

ARC_DIVERSITY_LAW_SCHEMA_VERSION_V149 = 149

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DiversityConfig:
    """Configuration for diversity enforcement."""
    
    # Maximum usage rate for any single concept
    max_concept_usage_pct: float = 40.0  # No concept can be used in >40% of tasks
    
    # Penalty for repeated calls within same solution
    repeated_call_penalty: float = 0.1  # 10% penalty per extra call
    max_repeated_calls: int = 3  # Hard limit on same concept calls in one solution
    
    # Minimum number of distinct concept "families"
    min_concept_families: int = 3
    
    # Similarity threshold for concept deduplication
    similarity_threshold: float = 0.9  # >90% similar = too similar
    
    # Minimum diversity score for a solution to be gate-accepted
    min_diversity_score: float = 0.5
    
    # Whether to force alternative mining when dominance detected
    force_alternative_mining: bool = True
    
    # Lineage competition settings
    enable_lineage_competition: bool = True
    min_lineages: int = 2  # At least 2 competing lineages
    lineage_tournament_size: int = 3  # Compare top 3 from each lineage


# ─────────────────────────────────────────────────────────────────────────────
# Concept Usage Tracking
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConceptUsageStats:
    """Statistics about concept usage across tasks."""
    
    concept_id: str
    total_calls: int = 0
    tasks_using: Set[str] = field(default_factory=set)
    call_depths: List[int] = field(default_factory=list)  # At what depth in program
    co_occurrence: Dict[str, int] = field(default_factory=dict)  # Other concepts used with
    
    def usage_rate(self, total_tasks: int) -> float:
        """Percentage of tasks using this concept."""
        if total_tasks == 0:
            return 0.0
        return len(self.tasks_using) / total_tasks * 100
    
    def avg_depth(self) -> float:
        """Average depth at which concept is called."""
        if not self.call_depths:
            return 0.0
        return sum(self.call_depths) / len(self.call_depths)


@dataclass
class ConceptEcosystem:
    """Tracks the ecosystem of concepts and their relationships."""
    
    concepts: Dict[str, ConceptUsageStats] = field(default_factory=dict)
    total_tasks: int = 0
    solved_tasks: int = 0
    
    # Lineage tracking: group concepts by origin
    lineages: Dict[str, Set[str]] = field(default_factory=dict)  # lineage_id -> concept_ids
    concept_lineage: Dict[str, str] = field(default_factory=dict)  # concept_id -> lineage_id
    
    # Diversity metrics
    dominance_warnings: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_usage(
        self,
        task_id: str,
        concept_id: str,
        depth: int = 0,
        other_concepts: Optional[List[str]] = None,
    ) -> None:
        """Record that a concept was used in a task."""
        if concept_id not in self.concepts:
            self.concepts[concept_id] = ConceptUsageStats(concept_id=concept_id)
        
        stats = self.concepts[concept_id]
        stats.total_calls += 1
        stats.tasks_using.add(task_id)
        stats.call_depths.append(depth)
        
        # Track co-occurrence
        if other_concepts:
            for other in other_concepts:
                if other != concept_id:
                    stats.co_occurrence[other] = stats.co_occurrence.get(other, 0) + 1
    
    def assign_lineage(self, concept_id: str, lineage_id: str) -> None:
        """Assign a concept to a lineage (family of related concepts)."""
        if lineage_id not in self.lineages:
            self.lineages[lineage_id] = set()
        self.lineages[lineage_id].add(concept_id)
        self.concept_lineage[concept_id] = lineage_id
    
    def get_dominant_concepts(self, config: DiversityConfig) -> List[str]:
        """Get concepts that exceed the usage threshold."""
        dominant = []
        for cid, stats in self.concepts.items():
            usage = stats.usage_rate(self.total_tasks)
            if usage > config.max_concept_usage_pct:
                dominant.append(cid)
                self.dominance_warnings.append({
                    "concept_id": cid,
                    "usage_rate": usage,
                    "threshold": config.max_concept_usage_pct,
                    "warning": "CONCEPT_DOMINANCE",
                })
        return dominant
    
    def get_diversity_score(self) -> float:
        """
        Calculate overall diversity score (0-1).
        
        Higher = more diverse, healthier ecosystem.
        """
        if not self.concepts:
            return 0.0
        
        # Factor 1: Number of active concepts
        n_concepts = len(self.concepts)
        concept_factor = min(1.0, n_concepts / 10)  # Normalize to 10 concepts
        
        # Factor 2: Usage distribution (entropy-like)
        usages = [stats.usage_rate(max(1, self.total_tasks)) for stats in self.concepts.values()]
        if usages:
            max_usage = max(usages)
            min_usage = min(usages)
            spread = (max_usage - min_usage) / max(1, max_usage)
            distribution_factor = 1.0 - (spread * 0.5)  # Penalize uneven spread
        else:
            distribution_factor = 0.0
        
        # Factor 3: Lineage diversity
        n_lineages = len(self.lineages)
        lineage_factor = min(1.0, n_lineages / 3)  # Normalize to 3 lineages
        
        # Weighted combination
        diversity_score = (
            concept_factor * 0.3 +
            distribution_factor * 0.4 +
            lineage_factor * 0.3
        )
        
        return diversity_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize ecosystem state."""
        return {
            "schema_version": int(ARC_DIVERSITY_LAW_SCHEMA_VERSION_V149),
            "total_tasks": int(self.total_tasks),
            "solved_tasks": int(self.solved_tasks),
            "n_concepts": len(self.concepts),
            "n_lineages": len(self.lineages),
            "diversity_score": float(self.get_diversity_score()),
            "concept_stats": {
                cid: {
                    "total_calls": int(stats.total_calls),
                    "n_tasks_using": len(stats.tasks_using),
                    "usage_rate_pct": float(stats.usage_rate(self.total_tasks)),
                    "avg_depth": float(stats.avg_depth()),
                }
                for cid, stats in self.concepts.items()
            },
            "lineages": {
                lid: list(sorted(cids))
                for lid, cids in self.lineages.items()
            },
            "dominance_warnings": list(self.dominance_warnings),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Concept Similarity
# ─────────────────────────────────────────────────────────────────────────────


def compute_concept_signature(
    concept_program: List[Dict[str, Any]],
) -> str:
    """
    Compute a signature for a concept's program structure.
    
    Used for similarity detection.
    """
    # Normalize: extract operation sequence
    ops = []
    for step in concept_program:
        op = str(step.get("op", step.get("op_id", "")))
        ops.append(op)
    
    # Create hash of normalized sequence
    sig = hashlib.sha256("|".join(ops).encode()).hexdigest()[:16]
    return sig


def compute_concept_similarity(
    concept_a: Dict[str, Any],
    concept_b: Dict[str, Any],
) -> float:
    """
    Compute similarity between two concepts (0-1).
    
    Based on:
    - Program structure similarity
    - Call dependency overlap
    - Task overlap
    """
    # Program similarity (Jaccard on operations)
    ops_a = set(
        str(s.get("op", s.get("op_id", "")))
        for s in concept_a.get("program", [])
    )
    ops_b = set(
        str(s.get("op", s.get("op_id", "")))
        for s in concept_b.get("program", [])
    )
    
    if not ops_a and not ops_b:
        program_sim = 1.0
    elif not ops_a or not ops_b:
        program_sim = 0.0
    else:
        intersection = len(ops_a & ops_b)
        union = len(ops_a | ops_b)
        program_sim = intersection / union if union > 0 else 0.0
    
    # Call dependency similarity
    deps_a = set(concept_a.get("call_deps", []))
    deps_b = set(concept_b.get("call_deps", []))
    
    if not deps_a and not deps_b:
        deps_sim = 1.0
    elif not deps_a or not deps_b:
        deps_sim = 0.0
    else:
        intersection = len(deps_a & deps_b)
        union = len(deps_a | deps_b)
        deps_sim = intersection / union if union > 0 else 0.0
    
    # Combined similarity
    similarity = program_sim * 0.7 + deps_sim * 0.3
    return similarity


def find_similar_concepts(
    concepts: List[Dict[str, Any]],
    threshold: float = 0.9,
) -> List[Tuple[str, str, float]]:
    """
    Find pairs of concepts that are too similar.
    
    Returns list of (concept_a_id, concept_b_id, similarity).
    """
    similar_pairs = []
    
    for i, ca in enumerate(concepts):
        for j, cb in enumerate(concepts):
            if i >= j:
                continue
            
            sim = compute_concept_similarity(ca, cb)
            if sim >= threshold:
                similar_pairs.append((
                    str(ca.get("id", ca.get("concept_id", f"concept_{i}"))),
                    str(cb.get("id", cb.get("concept_id", f"concept_{j}"))),
                    float(sim),
                ))
    
    return similar_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Diversity Gate Check
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DiversityCheckResult:
    """Result of checking a solution against diversity law."""
    
    passed: bool
    score: float
    violations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "score": float(self.score),
            "violations": list(self.violations),
            "warnings": list(self.warnings),
        }


def check_solution_diversity(
    solution_concepts: List[str],
    ecosystem: ConceptEcosystem,
    config: DiversityConfig,
) -> DiversityCheckResult:
    """
    Check if a solution's concept usage satisfies diversity law.
    
    Checks:
    1. No concept is used more than max_repeated_calls times
    2. Concepts used are not all from same lineage
    3. Overall diversity score meets threshold
    """
    violations = []
    warnings = []
    
    # Count concept usage in this solution
    usage_count = Counter(solution_concepts)
    
    # Check 1: Repeated calls
    for concept_id, count in usage_count.items():
        if count > config.max_repeated_calls:
            violations.append({
                "type": "REPEATED_CALLS_EXCEEDED",
                "concept_id": concept_id,
                "count": count,
                "max_allowed": config.max_repeated_calls,
            })
        elif count > 1:
            warnings.append({
                "type": "MULTIPLE_CALLS",
                "concept_id": concept_id,
                "count": count,
            })
    
    # Check 2: Lineage diversity (if enabled)
    if config.enable_lineage_competition:
        lineages_used = set()
        for cid in usage_count.keys():
            lid = ecosystem.concept_lineage.get(cid)
            if lid:
                lineages_used.add(lid)
        
        if len(lineages_used) == 1 and len(usage_count) >= 2:
            warnings.append({
                "type": "SINGLE_LINEAGE",
                "lineage": list(lineages_used)[0],
                "concepts": list(usage_count.keys()),
            })
    
    # Check 3: Dominant concept usage
    for concept_id in usage_count.keys():
        if concept_id in ecosystem.concepts:
            usage_rate = ecosystem.concepts[concept_id].usage_rate(ecosystem.total_tasks)
            if usage_rate > config.max_concept_usage_pct:
                warnings.append({
                    "type": "DOMINANT_CONCEPT_USED",
                    "concept_id": concept_id,
                    "usage_rate": usage_rate,
                    "threshold": config.max_concept_usage_pct,
                })
    
    # Calculate diversity score for this solution
    unique_concepts = len(usage_count)
    total_calls = sum(usage_count.values())
    
    if total_calls == 0:
        score = 0.0
    else:
        # Factor 1: Variety (more unique concepts = better)
        variety = min(1.0, unique_concepts / 3)
        
        # Factor 2: Distribution (even usage = better)
        if unique_concepts > 1:
            counts = list(usage_count.values())
            max_count = max(counts)
            avg_count = sum(counts) / len(counts)
            distribution = avg_count / max_count
        else:
            distribution = 1.0
        
        # Factor 3: Penalty for repeated calls
        penalty = 0.0
        for count in usage_count.values():
            if count > 1:
                penalty += (count - 1) * config.repeated_call_penalty
        penalty = min(0.5, penalty)  # Cap penalty at 50%
        
        score = (variety * 0.4 + distribution * 0.4) * (1 - penalty)
    
    passed = len(violations) == 0 and score >= config.min_diversity_score
    
    return DiversityCheckResult(
        passed=passed,
        score=score,
        violations=violations,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Alternative Mining Trigger
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AlternativeMiningRequest:
    """Request to mine alternative concepts when dominance detected."""
    
    trigger_reason: str
    dominant_concept_id: Optional[str] = None
    target_lineage: Optional[str] = None
    exclude_concepts: Set[str] = field(default_factory=set)
    target_tasks: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_reason": str(self.trigger_reason),
            "dominant_concept_id": str(self.dominant_concept_id) if self.dominant_concept_id else None,
            "target_lineage": str(self.target_lineage) if self.target_lineage else None,
            "exclude_concepts": list(sorted(self.exclude_concepts)),
            "n_target_tasks": len(self.target_tasks),
        }


def should_trigger_alternative_mining(
    ecosystem: ConceptEcosystem,
    config: DiversityConfig,
) -> Optional[AlternativeMiningRequest]:
    """
    Check if alternative concept mining should be triggered.
    
    Triggers:
    1. A concept exceeds usage threshold
    2. Too few lineages exist
    3. Overall diversity score is too low
    """
    # Check 1: Dominant concepts
    dominant = ecosystem.get_dominant_concepts(config)
    if dominant:
        return AlternativeMiningRequest(
            trigger_reason="CONCEPT_DOMINANCE",
            dominant_concept_id=dominant[0],
            exclude_concepts=set(dominant),
            target_tasks=set(),  # Mine for all tasks
        )
    
    # Check 2: Lineage diversity
    if config.enable_lineage_competition:
        if len(ecosystem.lineages) < config.min_lineages:
            return AlternativeMiningRequest(
                trigger_reason="INSUFFICIENT_LINEAGES",
                target_lineage=None,
                exclude_concepts=set(),
            )
    
    # Check 3: Overall diversity
    diversity_score = ecosystem.get_diversity_score()
    if diversity_score < config.min_diversity_score:
        return AlternativeMiningRequest(
            trigger_reason="LOW_DIVERSITY_SCORE",
            dominant_concept_id=None,
            exclude_concepts=set(),
        )
    
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Lineage Competition
# ─────────────────────────────────────────────────────────────────────────────


def run_lineage_tournament(
    ecosystem: ConceptEcosystem,
    config: DiversityConfig,
) -> Dict[str, Any]:
    """
    Run tournament between lineages to determine which concepts survive.
    
    This implements "concept natural selection" - lineages compete,
    and only the most useful concepts from each lineage survive.
    """
    results = {
        "lineages_competing": len(ecosystem.lineages),
        "tournament_size": config.lineage_tournament_size,
        "survivors": {},
        "eliminated": [],
    }
    
    if len(ecosystem.lineages) < 2:
        results["status"] = "insufficient_lineages"
        return results
    
    # For each lineage, rank concepts by utility
    for lineage_id, concept_ids in ecosystem.lineages.items():
        # Score concepts by: tasks solved, avg depth, uniqueness
        scored = []
        for cid in concept_ids:
            if cid not in ecosystem.concepts:
                continue
            stats = ecosystem.concepts[cid]
            
            # Score components
            usage_score = stats.usage_rate(ecosystem.total_tasks) / 100
            depth_score = min(1.0, stats.avg_depth() / 3)  # Normalize to depth 3
            
            # Penalize if too dominant
            if stats.usage_rate(ecosystem.total_tasks) > config.max_concept_usage_pct:
                penalty = 0.5
            else:
                penalty = 0.0
            
            total_score = (usage_score * 0.6 + depth_score * 0.4) * (1 - penalty)
            scored.append((cid, total_score))
        
        # Sort by score descending
        scored.sort(key=lambda x: -x[1])
        
        # Keep top N as survivors
        survivors = scored[:config.lineage_tournament_size]
        eliminated = scored[config.lineage_tournament_size:]
        
        results["survivors"][lineage_id] = [
            {"concept_id": cid, "score": score}
            for cid, score in survivors
        ]
        
        for cid, score in eliminated:
            results["eliminated"].append({
                "concept_id": cid,
                "lineage": lineage_id,
                "score": score,
            })
    
    results["status"] = "completed"
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Integration with Gate
# ─────────────────────────────────────────────────────────────────────────────


def enforce_diversity_law(
    solution_concepts: List[str],
    ecosystem: ConceptEcosystem,
    config: Optional[DiversityConfig] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Main entry point for diversity law enforcement.
    
    Returns (passed, audit_log).
    """
    cfg = config or DiversityConfig()
    
    audit = {
        "law": "DIVERSITY_LAW",
        "schema_version": int(ARC_DIVERSITY_LAW_SCHEMA_VERSION_V149),
    }
    
    # Check solution diversity
    check_result = check_solution_diversity(solution_concepts, ecosystem, cfg)
    
    audit["solution_check"] = check_result.to_dict()
    audit["ecosystem_diversity_score"] = float(ecosystem.get_diversity_score())
    
    # Check if alternative mining should be triggered
    mining_request = should_trigger_alternative_mining(ecosystem, cfg)
    if mining_request:
        audit["alternative_mining_needed"] = True
        audit["mining_request"] = mining_request.to_dict()
    else:
        audit["alternative_mining_needed"] = False
    
    passed = check_result.passed
    audit["verdict"] = "PASS" if passed else "FAIL"
    
    return passed, audit


# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────


__all__ = [
    "ARC_DIVERSITY_LAW_SCHEMA_VERSION_V149",
    "DiversityConfig",
    "ConceptUsageStats",
    "ConceptEcosystem",
    "DiversityCheckResult",
    "AlternativeMiningRequest",
    "compute_concept_signature",
    "compute_concept_similarity",
    "find_similar_concepts",
    "check_solution_diversity",
    "should_trigger_alternative_mining",
    "run_lineage_tournament",
    "enforce_diversity_law",
]
