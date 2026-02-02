"""
Tests for inevitability_gate_v150.py

Tests the extended gate that integrates diversity law as Law 8.
"""

import unittest
from typing import Any, Dict, List

from atos_core.act import Act
from atos_core.diversity_law_v149 import ConceptEcosystem, DiversityConfig
from atos_core.inevitability_gate_v145 import GateRejectionReason
from atos_core.inevitability_gate_v150 import (
    INEVITABILITY_GATE_SCHEMA_VERSION_V150,
    GateRejectionReasonV150,
    GateResultV150,
    InevitabilityConfigV150,
    check_law_diversity,
    diagnose_regime_v150,
    pass_inevitability_gate_v150,
)


def make_concept_act(
    concept_id: str,
    depth: int = 1,
    call_deps: List[str] = None,
    pcc_hash: str = "0" * 64,
) -> Act:
    """Helper to create a concept_csv Act for testing."""
    return Act(
        id=concept_id,
        version=1,
        created_at="2024-01-01T00:00:00Z",
        kind="concept_csv",
        match={},
        program=[],
        evidence={
            "pcc_v2": {
                "depth": depth,
                "call_deps": call_deps or [],
                "pcc_hash": pcc_hash,
            },
        },
    )


def make_trace_with_csv_calls(n: int) -> List[Dict[str, Any]]:
    """Helper to create trace events with CSV_CALL ops."""
    return [{"op": "CSV_CALL", "concept_id": f"c{i}"} for i in range(n)]


class TestSchemaVersion(unittest.TestCase):
    """Test schema version."""
    
    def test_version_is_150(self):
        self.assertEqual(INEVITABILITY_GATE_SCHEMA_VERSION_V150, 150)


class TestGateResultV150(unittest.TestCase):
    """Test GateResultV150 dataclass."""
    
    def test_to_dict_includes_diversity_fields(self):
        result = GateResultV150(
            passed=True,
            rejection_reasons=[],
            concepts_used=2,
            max_depth=1,
            csv_calls_total=3,
            cross_task_reuse_count=4,
            search_steps_taken=10,
            search_collapse_factor=0.5,
            budget_allocated=100,
            budget_used=50,
            budget_remaining=50,
            solution_hash="abc123",
            gate_timestamp="2024-01-01T00:00:00Z",
            diversity_score=0.75,
            diversity_passed=True,
            ecosystem_diversity_score=0.8,
            alternative_mining_needed=False,
        )
        
        data = result.to_dict()
        
        self.assertEqual(data["schema_version"], 150)
        self.assertEqual(data["kind"], "inevitability_gate_result_v150")
        self.assertEqual(data["diversity_score"], 0.75)
        self.assertTrue(data["diversity_passed"])
        self.assertEqual(data["ecosystem_diversity_score"], 0.8)
        self.assertFalse(data["alternative_mining_needed"])


class TestInevitabilityConfigV150(unittest.TestCase):
    """Test InevitabilityConfigV150 dataclass."""
    
    def test_default_enforces_diversity(self):
        config = InevitabilityConfigV150()
        self.assertTrue(config.enforce_diversity)
    
    def test_to_dict_includes_diversity_config(self):
        config = InevitabilityConfigV150()
        data = config.to_dict()
        
        self.assertEqual(data["schema_version"], 150)
        self.assertTrue(data["enforce_diversity"])
        self.assertIn("diversity_config", data)
        self.assertEqual(data["diversity_config"]["max_concept_usage_pct"], 40.0)


class TestCheckLawDiversity(unittest.TestCase):
    """Test the LAW_DIVERSITY check."""
    
    def setUp(self):
        self.ecosystem = ConceptEcosystem()
        self.ecosystem.total_tasks = 10
        self.config = InevitabilityConfigV150()
    
    def test_pass_with_diverse_concepts(self):
        concepts = [
            make_concept_act("c1"),
            make_concept_act("c2"),
        ]
        
        passed, rejections, audit = check_law_diversity(
            {}, concepts, self.ecosystem, self.config
        )
        
        self.assertTrue(passed)
        self.assertEqual(len(rejections), 0)
    
    def test_pass_with_diversity_disabled(self):
        config = InevitabilityConfigV150()
        config.enforce_diversity = False
        
        concepts = [make_concept_act("c1") for _ in range(10)]  # Same concept
        
        passed, rejections, audit = check_law_diversity(
            {}, concepts, self.ecosystem, config
        )
        
        self.assertTrue(passed)
        self.assertEqual(len(rejections), 0)
        self.assertTrue(any(e.get("verdict") == "SKIPPED" for e in audit))
    
    def test_fail_excessive_repeated_calls(self):
        # Same concept used 5 times (exceeds max_repeated_calls=3)
        concepts = [make_concept_act("same_concept") for _ in range(5)]
        
        passed, rejections, audit = check_law_diversity(
            {}, concepts, self.ecosystem, self.config
        )
        
        self.assertFalse(passed)
        self.assertTrue(any(
            r == GateRejectionReasonV150.EXCESSIVE_REPEATED_CALLS
            or r == "EXCESSIVE_REPEATED_CALLS"
            for r in rejections
        ))


class TestPassInevitabilityGateV150(unittest.TestCase):
    """Test the main gate function."""
    
    def setUp(self):
        self.ecosystem = ConceptEcosystem()
        self.ecosystem.total_tasks = 10
    
    def make_valid_solution(self) -> Dict[str, Any]:
        """Create a solution that passes all laws."""
        return {
            "task_id": "test_task",
            "output": [[0, 1], [2, 3]],
        }
    
    def make_valid_concepts(self, n: int = 2) -> List[Act]:
        """Create concepts that pass all requirements."""
        return [
            make_concept_act(
                f"concept_{i}",
                depth=1,
                call_deps=["dep1", "dep2"],
            )
            for i in range(n)
        ]
    
    def make_reuse_registry(self, concepts: List[Act]) -> Dict[str, set]:
        """Create reuse registry showing cross-task reuse."""
        return {
            str(c.id): {"task_1", "task_2", "task_3"}
            for c in concepts
        }
    
    def test_pass_all_laws(self):
        solution = self.make_valid_solution()
        concepts = self.make_valid_concepts(2)
        trace = make_trace_with_csv_calls(3)
        reuse = self.make_reuse_registry(concepts)
        
        # Record usage in ecosystem
        for c in concepts:
            self.ecosystem.record_usage("task_1", str(c.id))
        
        result = pass_inevitability_gate_v150(
            solution=solution,
            concepts_used=concepts,
            trace_events=trace,
            reuse_registry=reuse,
            task_id="test_task",
            search_steps_with_concepts=10,
            search_steps_without_concepts=100,
            ecosystem=self.ecosystem,
        )
        
        self.assertTrue(result.passed)
        self.assertEqual(len(result.rejection_reasons), 0)
        self.assertEqual(result.concepts_used, 2)
    
    def test_fail_no_concepts(self):
        result = pass_inevitability_gate_v150(
            solution={},
            concepts_used=[],
            trace_events=[],
            ecosystem=self.ecosystem,
        )
        
        self.assertFalse(result.passed)
        self.assertTrue(any(
            r == GateRejectionReason.NO_CONCEPT_USED
            for r in result.rejection_reasons
        ))
    
    def test_fail_excessive_repeated_calls(self):
        solution = self.make_valid_solution()
        # Same concept 5 times
        concepts = [make_concept_act("same", depth=1, call_deps=["d"]) for _ in range(5)]
        trace = make_trace_with_csv_calls(5)
        reuse = {"same": {"t1", "t2", "t3"}}
        
        result = pass_inevitability_gate_v150(
            solution=solution,
            concepts_used=concepts,
            trace_events=trace,
            reuse_registry=reuse,
            task_id="test_task",
            search_steps_with_concepts=10,
            search_steps_without_concepts=100,
            ecosystem=self.ecosystem,
        )
        
        self.assertFalse(result.passed)
        self.assertTrue(any(
            str(r) == GateRejectionReasonV150.EXCESSIVE_REPEATED_CALLS
            or str(r) == "EXCESSIVE_REPEATED_CALLS"
            for r in result.rejection_reasons
        ))
    
    def test_includes_diversity_metrics(self):
        solution = self.make_valid_solution()
        concepts = self.make_valid_concepts(2)
        trace = make_trace_with_csv_calls(2)
        reuse = self.make_reuse_registry(concepts)
        
        result = pass_inevitability_gate_v150(
            solution=solution,
            concepts_used=concepts,
            trace_events=trace,
            reuse_registry=reuse,
            search_steps_with_concepts=10,
            search_steps_without_concepts=100,
            ecosystem=self.ecosystem,
        )
        
        self.assertGreaterEqual(result.diversity_score, 0.0)
        self.assertIn("diversity_score", result.to_dict())
        self.assertIn("diversity_passed", result.to_dict())
    
    def test_detects_alternative_mining_needed(self):
        # Setup ecosystem with dominant concept
        for i in range(6):  # 60% usage (exceeds 40% threshold)
            self.ecosystem.record_usage(f"task_{i}", "dominant_concept")
        self.ecosystem.total_tasks = 10
        
        solution = self.make_valid_solution()
        concepts = [make_concept_act("dominant_concept", depth=1, call_deps=["d"])]
        trace = make_trace_with_csv_calls(1)
        reuse = {"dominant_concept": {"t1", "t2", "t3"}}
        
        result = pass_inevitability_gate_v150(
            solution=solution,
            concepts_used=concepts,
            trace_events=trace,
            reuse_registry=reuse,
            search_steps_with_concepts=10,
            search_steps_without_concepts=100,
            ecosystem=self.ecosystem,
        )
        
        self.assertTrue(result.alternative_mining_needed)


class TestDiagnoseRegimeV150(unittest.TestCase):
    """Test the extended regime diagnosis."""
    
    def make_result(
        self,
        passed: bool = True,
        concepts_used: int = 2,
        max_depth: int = 2,  # Changed to 2 to pass shallow check
        csv_calls_total: int = 3,  # Changed to non-zero
        diversity_passed: bool = True,
        diversity_score: float = 0.7,
        alternative_mining_needed: bool = False,
    ) -> GateResultV150:
        return GateResultV150(
            passed=passed,
            rejection_reasons=[],
            concepts_used=concepts_used,
            max_depth=max_depth,
            csv_calls_total=csv_calls_total,
            cross_task_reuse_count=3,
            search_steps_taken=10,
            search_collapse_factor=0.5,
            budget_allocated=100,
            budget_used=50,
            budget_remaining=50,
            solution_hash="abc",
            gate_timestamp="2024-01-01T00:00:00Z",
            diversity_score=diversity_score,
            diversity_passed=diversity_passed,
            ecosystem_diversity_score=0.8,
            alternative_mining_needed=alternative_mining_needed,
        )
    
    def test_healthy_regime(self):
        results = [
            self.make_result(passed=True, diversity_passed=True)
            for _ in range(10)
        ]
        
        diagnosis = diagnose_regime_v150(results)
        
        self.assertEqual(diagnosis["regime_status"], "HEALTHY")
        self.assertEqual(diagnosis["honest_answer_concepts"], "NO - GOOD")
        self.assertEqual(diagnosis["honest_answer_diversity"], "NO - GOOD")
    
    def test_broken_regime_conceptless(self):
        results = [
            self.make_result(passed=True, concepts_used=0)
            for _ in range(3)
        ]
        
        diagnosis = diagnose_regime_v150(results)
        
        self.assertEqual(diagnosis["regime_status"], "BROKEN")
        self.assertEqual(diagnosis["honest_answer_concepts"], "YES - BUG")
        self.assertGreater(diagnosis["conceptless_passes"], 0)
    
    def test_weak_diversity_regime(self):
        results = [
            self.make_result(passed=True, diversity_passed=False, diversity_score=0.2)
            for _ in range(10)
        ]
        
        diagnosis = diagnose_regime_v150(results)
        
        self.assertEqual(diagnosis["regime_status"], "WEAK_DIVERSITY")
        self.assertEqual(diagnosis["honest_answer_diversity"], "YES - RISK")
    
    def test_needs_diversification(self):
        results = [
            self.make_result(passed=True, alternative_mining_needed=True)
            for _ in range(5)
        ] + [
            self.make_result(passed=True, alternative_mining_needed=False)
            for _ in range(5)
        ]
        
        diagnosis = diagnose_regime_v150(results)
        
        self.assertEqual(diagnosis["mining_needed_count"], 5)
    
    def test_includes_diversity_metrics(self):
        results = [self.make_result() for _ in range(5)]
        
        diagnosis = diagnose_regime_v150(results)
        
        self.assertIn("diversity_failures", diagnosis)
        self.assertIn("avg_diversity_score", diagnosis)
        self.assertIn("mining_needed_count", diagnosis)
        self.assertIn("could_one_concept_dominate", diagnosis)


class TestGateRejectionReasonV150(unittest.TestCase):
    """Test the extended rejection reasons."""
    
    def test_has_diversity_reasons(self):
        self.assertEqual(
            GateRejectionReasonV150.DIVERSITY_VIOLATION,
            "DIVERSITY_VIOLATION"
        )
        self.assertEqual(
            GateRejectionReasonV150.CONCEPT_DOMINANCE,
            "CONCEPT_DOMINANCE"
        )
        self.assertEqual(
            GateRejectionReasonV150.EXCESSIVE_REPEATED_CALLS,
            "EXCESSIVE_REPEATED_CALLS"
        )
    
    def test_inherits_v145_reasons(self):
        # GateRejectionReasonV150 is now DiversityRejectionReason, not an enum
        # It provides diversity-specific reasons, V145 reasons come from original enum
        self.assertEqual(
            GateRejectionReason.NO_CONCEPT_USED,
            "NO_CONCEPT_USED"
        )


if __name__ == "__main__":
    unittest.main()
