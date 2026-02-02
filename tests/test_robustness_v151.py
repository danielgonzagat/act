"""
robustness_tests_v151.py - Anti-Regression Robustness Tests

These tests verify that the system FAILS when critical features are removed.

The key insight: If the system can solve tasks WITHOUT using concepts,
reuse, or CSV_CALL, then our emergence regime is broken.

Schema version: 151
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from atos_core.act import Act
from atos_core.diversity_law_v149 import ConceptEcosystem, DiversityConfig
from atos_core.inevitability_gate_v145 import (
    GateRejectionReason,
    GateResult,
    InevitabilityConfig,
    inevitability_gate,
)
from atos_core.inevitability_gate_v150 import (
    GateResultV150,
    InevitabilityConfigV150,
    diagnose_regime_v150,
    pass_inevitability_gate_v150,
)


ROBUSTNESS_TESTS_SCHEMA_VERSION_V151 = 151


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def make_valid_act(
    act_id: str,
    depth: int = 1,
    call_deps: Optional[List[str]] = None,
) -> Act:
    """Create a valid concept_csv Act that passes all checks."""
    return Act(
        id=act_id,
        version=1,
        created_at="2024-01-01T00:00:00Z",
        kind="concept_csv",
        match={},
        program=[],
        evidence={
            "pcc_v2": {
                "depth": depth,
                "call_deps": call_deps or ["dep1", "dep2"],
                "pcc_hash": "a" * 64,
            },
        },
    )


def make_primitive_act(act_id: str) -> Act:
    """Create a primitive Act (NOT concept_csv)."""
    return Act(
        id=act_id,
        version=1,
        created_at="2024-01-01T00:00:00Z",
        kind="primitive",  # NOT concept_csv
        match={},
        program=[],
        evidence={},
    )


def make_shallow_act(act_id: str) -> Act:
    """Create a concept with depth=0 (too shallow)."""
    return Act(
        id=act_id,
        version=1,
        created_at="2024-01-01T00:00:00Z",
        kind="concept_csv",
        match={},
        program=[],
        evidence={
            "pcc_v2": {
                "depth": 0,  # TOO SHALLOW
                "call_deps": [],
                "pcc_hash": "b" * 64,
            },
        },
    )


def make_flat_act(act_id: str) -> Act:
    """Create a concept with no call_deps (flat, no composition)."""
    return Act(
        id=act_id,
        version=1,
        created_at="2024-01-01T00:00:00Z",
        kind="concept_csv",
        match={},
        program=[],
        evidence={
            "pcc_v2": {
                "depth": 1,
                "call_deps": [],  # NO COMPOSITION
                "pcc_hash": "c" * 64,
            },
        },
    )


def make_reuse_registry(concept_ids: List[str], tasks_per_concept: int = 3) -> Dict[str, set]:
    """Create reuse registry showing cross-task reuse."""
    return {
        cid: {f"task_{i}" for i in range(tasks_per_concept)}
        for cid in concept_ids
    }


def make_trace_with_csv_calls(n: int) -> List[Dict[str, Any]]:
    """Create trace with n CSV_CALL events."""
    return [{"op": "CSV_CALL", "concept_id": f"c{i}"} for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# ROBUSTNESS TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestWithoutConceptsMustFail(unittest.TestCase):
    """
    Test that the system MUST FAIL when no concepts are provided.
    
    This is the core test: if this passes (gate rejects conceptless solutions),
    then the emergence regime is working.
    """
    
    def test_empty_concepts_rejected(self):
        """Gate MUST reject solution with zero concepts."""
        result = inevitability_gate(
            solution={"task": "test", "output": [[0]]},
            concepts_used=[],  # NO CONCEPTS
            trace_events=[],
            task_id="test_task",
        )
        
        self.assertFalse(result.passed, "Gate should REJECT solution without concepts")
        self.assertIn(
            GateRejectionReason.NO_CONCEPT_USED,
            result.rejection_reasons,
            "Rejection should be due to NO_CONCEPT_USED",
        )
    
    def test_only_primitives_rejected(self):
        """Gate MUST reject solution with only primitives (no concept_csv)."""
        primitives = [make_primitive_act(f"prim_{i}") for i in range(5)]
        
        result = inevitability_gate(
            solution={"task": "test"},
            concepts_used=primitives,
            trace_events=[],
            task_id="test_task",
        )
        
        self.assertFalse(result.passed, "Gate should REJECT primitive-only solutions")
        self.assertIn(
            GateRejectionReason.ONLY_PRIMITIVES,
            result.rejection_reasons,
            "Rejection should be due to ONLY_PRIMITIVES",
        )


class TestWithoutDepthMustFail(unittest.TestCase):
    """
    Test that the system MUST FAIL when concepts lack depth.
    
    Shallow concepts (depth=0) indicate no abstraction building.
    """
    
    def test_shallow_concepts_rejected(self):
        """Gate MUST reject solution with depth=0 concepts."""
        shallow = [make_shallow_act("shallow_1")]
        
        result = inevitability_gate(
            solution={"task": "test"},
            concepts_used=shallow,
            trace_events=[],
            task_id="test_task",
        )
        
        self.assertFalse(result.passed, "Gate should REJECT shallow concepts")
        # Note: depth=0 concepts may pass if they're "base concepts"
        # but they must have call_deps=[] in that case, caught by FLAT_CONCEPT


class TestWithoutCompositionMustFail(unittest.TestCase):
    """
    Test that the system MUST FAIL when concepts lack composition (CSV_CALL).
    
    Flat concepts with no call_deps indicate no concept reuse.
    """
    
    def test_flat_concepts_rejected(self):
        """Gate MUST reject solution with flat concepts (no call_deps)."""
        flat = [make_flat_act("flat_1")]
        trace_with_no_csv_calls = []  # No CSV_CALL in trace
        
        result = inevitability_gate(
            solution={"task": "test"},
            concepts_used=flat,
            trace_events=trace_with_no_csv_calls,
            task_id="test_task",
        )
        
        self.assertFalse(result.passed, "Gate should REJECT flat concepts")
        # Either NO_CSV_CALL or FLAT_CONCEPT depending on trace
        has_composition_violation = any(
            r in (GateRejectionReason.NO_CSV_CALL, GateRejectionReason.FLAT_CONCEPT)
            for r in result.rejection_reasons
        )
        self.assertTrue(has_composition_violation, "Should have composition violation")


class TestWithoutReuseMustFail(unittest.TestCase):
    """
    Test that the system MUST FAIL when concepts lack cross-task reuse.
    
    A concept used in only 1 task isn't really a "concept" - it's memorization.
    """
    
    def test_no_reuse_rejected(self):
        """Gate MUST reject solution when concept has no cross-task reuse."""
        concepts = [make_valid_act("concept_1")]
        trace = make_trace_with_csv_calls(2)
        
        # Empty reuse registry = no cross-task reuse
        no_reuse = {}
        
        result = inevitability_gate(
            solution={"task": "test"},
            concepts_used=concepts,
            trace_events=trace,
            reuse_registry=no_reuse,
            task_id="task_1",
        )
        
        self.assertFalse(result.passed, "Gate should REJECT without cross-task reuse")
        self.assertIn(
            GateRejectionReason.NO_CROSS_TASK_REUSE,
            result.rejection_reasons,
            "Rejection should be due to NO_CROSS_TASK_REUSE",
        )


class TestWithoutProofMustFail(unittest.TestCase):
    """
    Test that the system MUST FAIL when concepts lack valid PCC.
    
    PCC (Proof of Concept Creation) ensures concepts are legitimate.
    """
    
    def test_missing_pcc_rejected(self):
        """Gate MUST reject solution when concept has no PCC."""
        # Concept without proper evidence
        concept_no_pcc = Act(
            id="no_pcc",
            version=1,
            created_at="2024-01-01T00:00:00Z",
            kind="concept_csv",
            match={},
            program=[],
            evidence={},  # NO pcc_v2!
        )
        
        result = inevitability_gate(
            solution={"task": "test"},
            concepts_used=[concept_no_pcc],
            trace_events=[],
            task_id="test_task",
        )
        
        self.assertFalse(result.passed, "Gate should REJECT without PCC")
        self.assertIn(
            GateRejectionReason.MISSING_PCC,
            result.rejection_reasons,
            "Rejection should be due to MISSING_PCC",
        )


class TestWithoutUtilityMustFail(unittest.TestCase):
    """
    Test that the system MUST FAIL when concepts don't collapse search.
    
    If concepts don't reduce search steps, they have no utility.
    """
    
    def test_no_collapse_rejected(self):
        """Gate MUST reject solution when concepts don't reduce search."""
        concepts = [make_valid_act("concept_1")]
        trace = make_trace_with_csv_calls(2)
        reuse = make_reuse_registry(["concept_1"])
        
        result = inevitability_gate(
            solution={"task": "test"},
            concepts_used=concepts,
            trace_events=trace,
            reuse_registry=reuse,
            task_id="task_1",
            search_steps_with_concepts=100,  # Same as without
            search_steps_without_concepts=100,  # No improvement!
        )
        
        self.assertFalse(result.passed, "Gate should REJECT without search collapse")
        has_utility_violation = any(
            r in (GateRejectionReason.NO_SEARCH_COLLAPSE, GateRejectionReason.NO_MEASURABLE_GAIN)
            for r in result.rejection_reasons
        )
        self.assertTrue(has_utility_violation, "Should have utility violation")


class TestWithoutBudgetMustFail(unittest.TestCase):
    """
    Test that the system MUST FAIL when budget is exceeded.
    
    No budget escape = must stay within allocated budget.
    """
    
    def test_budget_exceeded_rejected(self):
        """Gate MUST reject solution when budget is exceeded."""
        concepts = [make_valid_act("concept_1")]
        trace = make_trace_with_csv_calls(2)
        reuse = make_reuse_registry(["concept_1"])
        
        result = inevitability_gate(
            solution={"task": "test"},
            concepts_used=concepts,
            trace_events=trace,
            reuse_registry=reuse,
            task_id="task_1",
            search_steps_with_concepts=10,
            search_steps_without_concepts=100,
            budget_allocated=50,
            budget_used=100,  # EXCEEDS allocated!
        )
        
        self.assertFalse(result.passed, "Gate should REJECT when budget exceeded")
        self.assertIn(
            GateRejectionReason.BUDGET_EXCEEDED,
            result.rejection_reasons,
            "Rejection should be due to BUDGET_EXCEEDED",
        )


class TestWithoutDiversityMustFail(unittest.TestCase):
    """
    Test that the system MUST FAIL when diversity requirements aren't met.
    
    This uses V150 gate with diversity law.
    """
    
    def test_excessive_repeated_calls_rejected(self):
        """Gate V150 MUST reject solution with excessive repeated calls."""
        # Same concept used 5 times (exceeds max_repeated_calls=3)
        concept = make_valid_act("same_concept")
        concepts = [concept] * 5
        trace = make_trace_with_csv_calls(5)
        reuse = {"same_concept": {"t1", "t2", "t3"}}
        
        ecosystem = ConceptEcosystem()
        ecosystem.total_tasks = 10
        
        result = pass_inevitability_gate_v150(
            solution={"task": "test"},
            concepts_used=concepts,
            trace_events=trace,
            reuse_registry=reuse,
            task_id="task_1",
            search_steps_with_concepts=10,
            search_steps_without_concepts=100,
            ecosystem=ecosystem,
        )
        
        self.assertFalse(result.passed, "Gate V150 should REJECT excessive repeated calls")
        self.assertFalse(result.diversity_passed, "Diversity check should fail")


class TestRegimeDiagnosisHonesty(unittest.TestCase):
    """
    Test that regime diagnosis is HONEST about failures.
    
    The key question: "Could the system survive without concepts?"
    """
    
    def make_conceptless_result(self) -> GateResultV150:
        """Create result that somehow passed WITHOUT concepts (BUG)."""
        return GateResultV150(
            passed=True,  # This shouldn't happen!
            rejection_reasons=[],
            concepts_used=0,  # NO CONCEPTS
            max_depth=0,
            csv_calls_total=0,
            cross_task_reuse_count=0,
            search_steps_taken=10,
            search_collapse_factor=0.5,
            budget_allocated=100,
            budget_used=50,
            budget_remaining=50,
            solution_hash="xxx",
            gate_timestamp="2024-01-01T00:00:00Z",
            diversity_score=0.0,
            diversity_passed=False,
            ecosystem_diversity_score=0.0,
            alternative_mining_needed=False,
        )
    
    def make_valid_result(self) -> GateResultV150:
        """Create valid result with concepts."""
        return GateResultV150(
            passed=True,
            rejection_reasons=[],
            concepts_used=2,
            max_depth=2,
            csv_calls_total=3,
            cross_task_reuse_count=4,
            search_steps_taken=10,
            search_collapse_factor=0.3,
            budget_allocated=100,
            budget_used=50,
            budget_remaining=50,
            solution_hash="xxx",
            gate_timestamp="2024-01-01T00:00:00Z",
            diversity_score=0.7,
            diversity_passed=True,
            ecosystem_diversity_score=0.8,
            alternative_mining_needed=False,
        )
    
    def test_detects_broken_regime(self):
        """Diagnosis MUST detect when conceptless solutions pass (BUG)."""
        # Simulate broken regime: solutions pass without concepts
        results = [self.make_conceptless_result() for _ in range(5)]
        
        diagnosis = diagnose_regime_v150(results)
        
        self.assertEqual(diagnosis["regime_status"], "BROKEN")
        self.assertTrue(diagnosis["could_survive_without_concepts"])
        self.assertEqual(diagnosis["honest_answer_concepts"], "YES - BUG")
    
    def test_detects_healthy_regime(self):
        """Diagnosis MUST report healthy when all solutions use concepts."""
        # Healthy regime: all solutions use concepts properly
        results = [self.make_valid_result() for _ in range(10)]
        
        diagnosis = diagnose_regime_v150(results)
        
        self.assertEqual(diagnosis["regime_status"], "HEALTHY")
        self.assertFalse(diagnosis["could_survive_without_concepts"])
        self.assertEqual(diagnosis["honest_answer_concepts"], "NO - GOOD")


class TestConfigValidation(unittest.TestCase):
    """
    Test that config validation catches escape routes.
    
    No config should allow bypassing the inevitability requirements.
    """
    
    def test_zero_concepts_not_allowed(self):
        """Config MUST NOT allow min_concepts_required=0."""
        config = InevitabilityConfig()
        config.min_concepts_required = 0
        
        errors = config.validate_config()
        
        self.assertTrue(len(errors) > 0, "Should have validation errors")
        self.assertTrue(any("min_concepts_required" in e for e in errors))
    
    def test_bypass_not_allowed(self):
        """Config MUST NOT allow bypass=True."""
        config = InevitabilityConfig()
        config.allow_bypass = True
        
        errors = config.validate_config()
        
        self.assertTrue(len(errors) > 0, "Should have validation errors")
        self.assertTrue(any("allow_bypass" in e for e in errors))
    
    def test_fallback_not_allowed(self):
        """Config MUST NOT allow fallback=True."""
        config = InevitabilityConfig()
        config.allow_fallback = True
        
        errors = config.validate_config()
        
        self.assertTrue(len(errors) > 0, "Should have validation errors")
        self.assertTrue(any("allow_fallback" in e for e in errors))
    
    def test_relaxation_not_allowed(self):
        """Config MUST NOT allow relaxation=True."""
        config = InevitabilityConfig()
        config.allow_relaxation = True
        
        errors = config.validate_config()
        
        self.assertTrue(len(errors) > 0, "Should have validation errors")
        self.assertTrue(any("allow_relaxation" in e for e in errors))


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Test: Running all robustness tests should PASS (meaning gate works)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RobustnessReport:
    """Report on robustness test results."""
    
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    
    # Each category
    without_concepts_protected: bool = False
    without_depth_protected: bool = False
    without_composition_protected: bool = False
    without_reuse_protected: bool = False
    without_proof_protected: bool = False
    without_utility_protected: bool = False
    without_budget_protected: bool = False
    without_diversity_protected: bool = False
    diagnosis_honest: bool = False
    config_validated: bool = False
    
    def is_robust(self) -> bool:
        """System is robust if all protections work."""
        return all([
            self.without_concepts_protected,
            self.without_depth_protected,
            self.without_composition_protected,
            self.without_reuse_protected,
            self.without_proof_protected,
            self.without_utility_protected,
            self.without_budget_protected,
            self.without_diversity_protected,
            self.diagnosis_honest,
            self.config_validated,
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(ROBUSTNESS_TESTS_SCHEMA_VERSION_V151),
            "total_tests": int(self.total_tests),
            "passed_tests": int(self.passed_tests),
            "failed_tests": int(self.failed_tests),
            "is_robust": bool(self.is_robust()),
            "protections": {
                "without_concepts": bool(self.without_concepts_protected),
                "without_depth": bool(self.without_depth_protected),
                "without_composition": bool(self.without_composition_protected),
                "without_reuse": bool(self.without_reuse_protected),
                "without_proof": bool(self.without_proof_protected),
                "without_utility": bool(self.without_utility_protected),
                "without_budget": bool(self.without_budget_protected),
                "without_diversity": bool(self.without_diversity_protected),
                "diagnosis_honest": bool(self.diagnosis_honest),
                "config_validated": bool(self.config_validated),
            },
        }


def run_robustness_suite() -> RobustnessReport:
    """
    Run all robustness tests and return report.
    
    Returns a report indicating if the system is protected against regressions.
    """
    import io
    import sys
    
    report = RobustnessReport()
    
    # Load and run all test classes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        TestWithoutConceptsMustFail,
        TestWithoutDepthMustFail,
        TestWithoutCompositionMustFail,
        TestWithoutReuseMustFail,
        TestWithoutProofMustFail,
        TestWithoutUtilityMustFail,
        TestWithoutBudgetMustFail,
        TestWithoutDiversityMustFail,
        TestRegimeDiagnosisHonesty,
        TestConfigValidation,
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests silently
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=0)
    result = runner.run(suite)
    
    report.total_tests = result.testsRun
    report.passed_tests = result.testsRun - len(result.failures) - len(result.errors)
    report.failed_tests = len(result.failures) + len(result.errors)
    
    # Check each protection category
    def all_passed(test_class):
        class_tests = loader.loadTestsFromTestCase(test_class)
        class_result = unittest.TestResult()
        class_tests.run(class_result)
        return class_result.wasSuccessful()
    
    report.without_concepts_protected = all_passed(TestWithoutConceptsMustFail)
    report.without_depth_protected = all_passed(TestWithoutDepthMustFail)
    report.without_composition_protected = all_passed(TestWithoutCompositionMustFail)
    report.without_reuse_protected = all_passed(TestWithoutReuseMustFail)
    report.without_proof_protected = all_passed(TestWithoutProofMustFail)
    report.without_utility_protected = all_passed(TestWithoutUtilityMustFail)
    report.without_budget_protected = all_passed(TestWithoutBudgetMustFail)
    report.without_diversity_protected = all_passed(TestWithoutDiversityMustFail)
    report.diagnosis_honest = all_passed(TestRegimeDiagnosisHonesty)
    report.config_validated = all_passed(TestConfigValidation)
    
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────


__all__ = [
    "ROBUSTNESS_TESTS_SCHEMA_VERSION_V151",
    "RobustnessReport",
    "run_robustness_suite",
    # Test classes for direct use
    "TestWithoutConceptsMustFail",
    "TestWithoutDepthMustFail",
    "TestWithoutCompositionMustFail",
    "TestWithoutReuseMustFail",
    "TestWithoutProofMustFail",
    "TestWithoutUtilityMustFail",
    "TestWithoutBudgetMustFail",
    "TestWithoutDiversityMustFail",
    "TestRegimeDiagnosisHonesty",
    "TestConfigValidation",
]


if __name__ == "__main__":
    unittest.main()
