"""
test_inevitability_v145.py - Tests that PROVE inevitability.

These tests MUST demonstrate that:
1. Solutions without concepts are REJECTED
2. Solutions with only primitives are REJECTED
3. Solutions with shallow concepts are REJECTED
4. Solutions with flat concepts are REJECTED
5. Solutions without cross-task reuse are REJECTED
6. Solutions without PCC proof are REJECTED
7. Solutions without search collapse are REJECTED
8. Solutions that exceed budget are REJECTED

If ANY of these tests can be bypassed, the regime is BROKEN.

Schema version: 145
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction
from atos_core.inevitability_gate_v145 import (
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
    diagnose_regime,
    inevitability_gate,
    write_gate_result_to_ledger,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def make_valid_concept(
    concept_id: str,
    *,
    depth: int = 2,
    call_deps: List[str] = None,
    pcc_hash: str = "a" * 64,
) -> Act:
    """Create a valid concept_csv Act that passes all laws."""
    call_deps = call_deps or ["dep_1", "dep_2"]
    
    return Act(
        id=str(concept_id),
        version=1,
        created_at="2024-01-01T00:00:00Z",
        kind="concept_csv",
        match={},
        program=[Instruction("CSV_RETURN", {"var": "result"})],
        evidence={
            "pcc_v2": {
                "depth": int(depth),
                "call_deps": list(call_deps),
                "pcc_hash": str(pcc_hash),
            },
        },
        cost={"overhead_bits": 1024},
        deps=list(call_deps),
        active=True,
    )


def make_primitive_act(act_id: str) -> Act:
    """Create a primitive Act (not concept_csv)."""
    return Act(
        id=str(act_id),
        version=1,
        created_at="2024-01-01T00:00:00Z",
        kind="primitive",  # NOT concept_csv
        match={},
        program=[Instruction("RETURN", {"value": 1})],
        evidence={},
        cost={"overhead_bits": 100},
        deps=[],
        active=True,
    )


def make_shallow_concept(concept_id: str) -> Act:
    """Create a shallow concept (depth=0)."""
    return Act(
        id=str(concept_id),
        version=1,
        created_at="2024-01-01T00:00:00Z",
        kind="concept_csv",
        match={},
        program=[Instruction("CSV_RETURN", {"var": "result"})],
        evidence={
            "pcc_v2": {
                "depth": 0,  # TOO SHALLOW
                "call_deps": ["dep"],
                "pcc_hash": "a" * 64,
            },
        },
        cost={"overhead_bits": 1024},
        deps=["dep"],
        active=True,
    )


def make_flat_concept(concept_id: str) -> Act:
    """Create a flat concept (no call_deps)."""
    return Act(
        id=str(concept_id),
        version=1,
        created_at="2024-01-01T00:00:00Z",
        kind="concept_csv",
        match={},
        program=[Instruction("CSV_RETURN", {"var": "result"})],
        evidence={
            "pcc_v2": {
                "depth": 2,
                "call_deps": [],  # NO COMPOSITION
                "pcc_hash": "a" * 64,
            },
        },
        cost={"overhead_bits": 1024},
        deps=[],
        active=True,
    )


def make_unproven_concept(concept_id: str) -> Act:
    """Create a concept without PCC proof."""
    return Act(
        id=str(concept_id),
        version=1,
        created_at="2024-01-01T00:00:00Z",
        kind="concept_csv",
        match={},
        program=[Instruction("CSV_RETURN", {"var": "result"})],
        evidence={},  # NO PCC
        cost={"overhead_bits": 1024},
        deps=[],
        active=True,
    )


def make_trace_with_csv_calls(count: int) -> List[Dict[str, Any]]:
    """Create trace events with CSV_CALLs."""
    return [
        {"op": "CSV_CALL", "callee": f"concept_{i}", "depth": i}
        for i in range(count)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# LAW_CONCEPT Tests - NO SOLUTION WITHOUT CONCEPTS
# ─────────────────────────────────────────────────────────────────────────────


class TestLawConcept(unittest.TestCase):
    """Tests that PROVE solutions without concepts are REJECTED."""

    def test_reject_no_concepts(self) -> None:
        """Solution with 0 concepts MUST be rejected."""
        solution = {"output": "result"}
        concepts: List[Act] = []
        config = InevitabilityConfig()
        
        passed, rejections, _ = check_law_concept(solution, concepts, config)
        
        self.assertFalse(passed)
        self.assertIn(GateRejectionReason.NO_CONCEPT_USED, rejections)

    def test_reject_only_primitives(self) -> None:
        """Solution with only primitives MUST be rejected."""
        solution = {"output": "result"}
        concepts = [make_primitive_act("prim_1"), make_primitive_act("prim_2")]
        config = InevitabilityConfig()
        
        passed, rejections, _ = check_law_concept(solution, concepts, config)
        
        self.assertFalse(passed)
        self.assertIn(GateRejectionReason.ONLY_PRIMITIVES, rejections)

    def test_accept_valid_concept(self) -> None:
        """Solution with valid concept_csv MUST pass."""
        solution = {"output": "result"}
        concepts = [make_valid_concept("concept_1")]
        config = InevitabilityConfig()
        
        passed, rejections, _ = check_law_concept(solution, concepts, config)
        
        self.assertTrue(passed)
        self.assertEqual(len(rejections), 0)


# ─────────────────────────────────────────────────────────────────────────────
# LAW_DEPTH Tests - NO SHALLOW CONCEPTS
# ─────────────────────────────────────────────────────────────────────────────


class TestLawDepth(unittest.TestCase):
    """Tests that PROVE shallow concepts are REJECTED."""

    def test_reject_shallow_concept(self) -> None:
        """Concept with depth < min MUST be rejected."""
        solution = {"output": "result"}
        concepts = [make_shallow_concept("shallow_1")]
        config = InevitabilityConfig(min_depth_required=1)
        
        passed, rejections, _ = check_law_depth(solution, concepts, config)
        
        self.assertFalse(passed)
        self.assertIn(GateRejectionReason.DEPTH_TOO_SHALLOW, rejections)

    def test_reject_depth_stagnation(self) -> None:
        """Concept with depth <= historical max MUST be rejected (if required)."""
        solution = {"output": "result"}
        concepts = [make_valid_concept("concept_1", depth=2)]
        config = InevitabilityConfig(require_depth_progression=True)
        
        passed, rejections, _ = check_law_depth(
            solution, concepts, config, historical_max_depth=3
        )
        
        self.assertFalse(passed)
        self.assertIn(GateRejectionReason.DEPTH_STAGNATION, rejections)

    def test_accept_deep_concept(self) -> None:
        """Concept with sufficient depth MUST pass."""
        solution = {"output": "result"}
        concepts = [make_valid_concept("concept_1", depth=3)]
        config = InevitabilityConfig(min_depth_required=2)
        
        passed, rejections, _ = check_law_depth(solution, concepts, config)
        
        self.assertTrue(passed)
        self.assertEqual(len(rejections), 0)


# ─────────────────────────────────────────────────────────────────────────────
# LAW_COMPOSITION Tests - NO FLAT CONCEPTS
# ─────────────────────────────────────────────────────────────────────────────


class TestLawComposition(unittest.TestCase):
    """Tests that PROVE flat concepts are REJECTED."""

    def test_reject_no_csv_calls(self) -> None:
        """Trace without CSV_CALLs MUST be rejected."""
        solution = {"output": "result"}
        concepts = [make_valid_concept("concept_1")]
        trace_events: List[Dict[str, Any]] = []  # NO CSV_CALLs
        config = InevitabilityConfig()
        
        passed, rejections, _ = check_law_composition(
            solution, concepts, trace_events, config
        )
        
        self.assertFalse(passed)
        self.assertIn(GateRejectionReason.NO_CSV_CALL, rejections)

    def test_reject_flat_concept(self) -> None:
        """Concept without call_deps MUST be rejected."""
        solution = {"output": "result"}
        concepts = [make_flat_concept("flat_1")]
        trace_events = make_trace_with_csv_calls(1)
        config = InevitabilityConfig()
        
        passed, rejections, _ = check_law_composition(
            solution, concepts, trace_events, config
        )
        
        self.assertFalse(passed)
        self.assertIn(GateRejectionReason.FLAT_CONCEPT, rejections)

    def test_accept_composed_concept(self) -> None:
        """Concept with CSV_CALLs and call_deps MUST pass."""
        solution = {"output": "result"}
        concepts = [make_valid_concept("concept_1", call_deps=["dep_1", "dep_2"])]
        trace_events = make_trace_with_csv_calls(2)
        config = InevitabilityConfig()
        
        passed, rejections, _ = check_law_composition(
            solution, concepts, trace_events, config
        )
        
        self.assertTrue(passed)
        self.assertEqual(len(rejections), 0)


# ─────────────────────────────────────────────────────────────────────────────
# LAW_REUSE Tests - NO SINGLE-TASK CONCEPTS
# ─────────────────────────────────────────────────────────────────────────────


class TestLawReuse(unittest.TestCase):
    """Tests that PROVE concepts without cross-task reuse are REJECTED."""

    def test_reject_no_cross_task_reuse(self) -> None:
        """Concept used in only one task MUST be rejected."""
        solution = {"output": "result"}
        concepts = [make_valid_concept("concept_1")]
        reuse_registry: Dict[str, Set[str]] = {}  # No prior reuse
        config = InevitabilityConfig(min_cross_task_reuse=1)
        
        passed, rejections, _ = check_law_reuse(
            solution, concepts, reuse_registry, "task_1", config
        )
        
        self.assertFalse(passed)
        self.assertIn(GateRejectionReason.NO_CROSS_TASK_REUSE, rejections)

    def test_accept_cross_task_reuse(self) -> None:
        """Concept reused across tasks MUST pass."""
        solution = {"output": "result"}
        concepts = [make_valid_concept("concept_1")]
        reuse_registry: Dict[str, Set[str]] = {
            "concept_1": {"task_0", "task_old"},  # Prior reuse
        }
        config = InevitabilityConfig(min_cross_task_reuse=1)
        
        passed, rejections, _ = check_law_reuse(
            solution, concepts, reuse_registry, "task_1", config
        )
        
        self.assertTrue(passed)
        self.assertEqual(len(rejections), 0)


# ─────────────────────────────────────────────────────────────────────────────
# LAW_PROOF Tests - NO UNPROVEN CONCEPTS
# ─────────────────────────────────────────────────────────────────────────────


class TestLawProof(unittest.TestCase):
    """Tests that PROVE concepts without PCC are REJECTED."""

    def test_reject_missing_pcc(self) -> None:
        """Concept without PCC MUST be rejected."""
        solution = {"output": "result"}
        concepts = [make_unproven_concept("unproven_1")]
        config = InevitabilityConfig()
        
        passed, rejections, _ = check_law_proof(solution, concepts, config)
        
        self.assertFalse(passed)
        self.assertIn(GateRejectionReason.MISSING_PCC, rejections)

    def test_reject_invalid_hash(self) -> None:
        """Concept with invalid hash MUST be rejected."""
        concept = make_valid_concept("concept_1", pcc_hash="short")  # Too short
        solution = {"output": "result"}
        config = InevitabilityConfig()
        
        passed, rejections, _ = check_law_proof(solution, [concept], config)
        
        self.assertFalse(passed)
        self.assertIn(GateRejectionReason.INVALID_PCC_HASH, rejections)

    def test_accept_proven_concept(self) -> None:
        """Concept with valid PCC MUST pass."""
        solution = {"output": "result"}
        concepts = [make_valid_concept("concept_1")]
        config = InevitabilityConfig()
        
        passed, rejections, _ = check_law_proof(solution, concepts, config)
        
        self.assertTrue(passed)
        self.assertEqual(len(rejections), 0)


# ─────────────────────────────────────────────────────────────────────────────
# LAW_UTILITY Tests - NO USELESS CONCEPTS
# ─────────────────────────────────────────────────────────────────────────────


class TestLawUtility(unittest.TestCase):
    """Tests that PROVE concepts without search collapse are REJECTED."""

    def test_reject_no_search_collapse(self) -> None:
        """Solution without search collapse MUST be rejected."""
        solution = {"output": "result"}
        config = InevitabilityConfig(max_search_collapse_factor=0.9)
        
        # Same steps with and without concepts = no collapse
        passed, rejections, _ = check_law_utility(
            solution,
            search_steps_with_concepts=100,
            search_steps_without_concepts=100,
            config=config,
        )
        
        self.assertFalse(passed)
        self.assertIn(GateRejectionReason.NO_SEARCH_COLLAPSE, rejections)

    def test_reject_no_measurable_gain(self) -> None:
        """Solution without measurable gain MUST be rejected."""
        solution = {"output": "result"}
        config = InevitabilityConfig(require_measurable_gain=True)
        
        # More steps with concepts = worse (no gain)
        passed, rejections, _ = check_law_utility(
            solution,
            search_steps_with_concepts=150,
            search_steps_without_concepts=100,
            config=config,
        )
        
        self.assertFalse(passed)
        self.assertIn(GateRejectionReason.NO_MEASURABLE_GAIN, rejections)

    def test_accept_search_collapse(self) -> None:
        """Solution with search collapse MUST pass."""
        solution = {"output": "result"}
        config = InevitabilityConfig(max_search_collapse_factor=1.0)
        
        # Fewer steps with concepts = collapse
        passed, rejections, _ = check_law_utility(
            solution,
            search_steps_with_concepts=50,
            search_steps_without_concepts=100,
            config=config,
        )
        
        self.assertTrue(passed)
        self.assertEqual(len(rejections), 0)


# ─────────────────────────────────────────────────────────────────────────────
# LAW_BUDGET Tests - NO BUDGET ESCAPE
# ─────────────────────────────────────────────────────────────────────────────


class TestLawBudget(unittest.TestCase):
    """Tests that PROVE solutions exceeding budget are REJECTED."""

    def test_reject_budget_exceeded(self) -> None:
        """Solution exceeding budget MUST be rejected."""
        solution = {"output": "result"}
        config = InevitabilityConfig(strict_budget=True)
        
        passed, rejections, _ = check_law_budget(
            solution,
            budget_allocated=100,
            budget_used=150,  # Over budget
            config=config,
        )
        
        self.assertFalse(passed)
        self.assertIn(GateRejectionReason.BUDGET_EXCEEDED, rejections)

    def test_accept_within_budget(self) -> None:
        """Solution within budget MUST pass."""
        solution = {"output": "result"}
        config = InevitabilityConfig(strict_budget=True)
        
        passed, rejections, _ = check_law_budget(
            solution,
            budget_allocated=100,
            budget_used=50,  # Under budget
            config=config,
        )
        
        self.assertTrue(passed)
        self.assertEqual(len(rejections), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Full Gate Tests - ALL LAWS COMBINED
# ─────────────────────────────────────────────────────────────────────────────


class TestInevitabilityGate(unittest.TestCase):
    """Tests for the full inevitability gate."""

    def test_gate_rejects_conceptless_solution(self) -> None:
        """Gate MUST reject solution without concepts."""
        result = inevitability_gate(
            solution={"output": "result"},
            concepts_used=[],
            trace_events=[],
            task_id="task_1",
            search_steps_with_concepts=100,
            search_steps_without_concepts=100,
        )
        
        self.assertFalse(result.passed)
        self.assertIn(GateRejectionReason.NO_CONCEPT_USED, result.rejection_reasons)

    def test_gate_rejects_primitive_only(self) -> None:
        """Gate MUST reject solution with only primitives."""
        result = inevitability_gate(
            solution={"output": "result"},
            concepts_used=[make_primitive_act("prim_1")],
            trace_events=[{"op": "CSV_CALL", "callee": "prim_1", "depth": 0}],
            task_id="task_1",
            search_steps_with_concepts=50,
            search_steps_without_concepts=100,
            reuse_registry={"prim_1": {"task_0"}},
        )
        
        self.assertFalse(result.passed)
        self.assertIn(GateRejectionReason.ONLY_PRIMITIVES, result.rejection_reasons)

    def test_gate_accepts_valid_solution(self) -> None:
        """Gate MUST accept solution meeting all laws."""
        concept = make_valid_concept("concept_1", depth=2, call_deps=["dep_1"])
        
        result = inevitability_gate(
            solution={"output": "result"},
            concepts_used=[concept],
            trace_events=[{"op": "CSV_CALL", "callee": "concept_1", "depth": 2}],
            task_id="task_1",
            reuse_registry={"concept_1": {"task_0", "task_old"}},
            search_steps_with_concepts=50,
            search_steps_without_concepts=100,
            budget_allocated=200,
            budget_used=50,
        )
        
        self.assertTrue(result.passed)
        self.assertEqual(len(result.rejection_reasons), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Config Validation Tests - NO ESCAPE ROUTES
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigValidation(unittest.TestCase):
    """Tests that PROVE config cannot allow escape routes."""

    def test_config_rejects_zero_concepts(self) -> None:
        """Config with min_concepts=0 MUST be rejected."""
        config = InevitabilityConfig(min_concepts_required=0)
        errors = config.validate_config()
        
        self.assertIn("min_concepts_required MUST be >= 1", errors)

    def test_config_rejects_zero_depth(self) -> None:
        """Config with min_depth=0 MUST be rejected."""
        config = InevitabilityConfig(min_depth_required=0)
        errors = config.validate_config()
        
        self.assertIn("min_depth_required MUST be >= 1", errors)

    def test_config_rejects_bypass(self) -> None:
        """Config allowing bypass MUST be rejected."""
        config = InevitabilityConfig(allow_bypass=True)
        errors = config.validate_config()
        
        self.assertIn("allow_bypass MUST be False", errors)

    def test_config_rejects_fallback(self) -> None:
        """Config allowing fallback MUST be rejected."""
        config = InevitabilityConfig(allow_fallback=True)
        errors = config.validate_config()
        
        self.assertIn("allow_fallback MUST be False", errors)

    def test_default_config_valid(self) -> None:
        """Default config MUST have no escape routes."""
        config = InevitabilityConfig()
        errors = config.validate_config()
        
        self.assertEqual(len(errors), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Regime Diagnosis Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRegimeDiagnosis(unittest.TestCase):
    """Tests for regime health diagnosis."""

    def test_diagnose_broken_regime(self) -> None:
        """Regime with conceptless passes is BROKEN."""
        results = [
            GateResult(
                passed=True,
                rejection_reasons=[],
                concepts_used=0,  # BUG: passed without concepts
                max_depth=0,
                csv_calls_total=0,
                cross_task_reuse_count=0,
                search_steps_taken=100,
                search_collapse_factor=1.0,
                budget_allocated=200,
                budget_used=100,
                budget_remaining=100,
                solution_hash="abc",
                gate_timestamp="2024-01-01T00:00:00Z",
            ),
        ]
        
        diagnosis = diagnose_regime(results)
        
        self.assertEqual(diagnosis["regime_status"], "BROKEN")
        self.assertTrue(diagnosis["could_survive_without_concepts"])
        self.assertEqual(diagnosis["honest_answer"], "YES - BUG")

    def test_diagnose_healthy_regime(self) -> None:
        """Regime with all proper passes is HEALTHY."""
        results = [
            GateResult(
                passed=True,
                rejection_reasons=[],
                concepts_used=2,
                max_depth=3,
                csv_calls_total=5,
                cross_task_reuse_count=3,
                search_steps_taken=50,
                search_collapse_factor=0.5,
                budget_allocated=200,
                budget_used=50,
                budget_remaining=150,
                solution_hash="abc",
                gate_timestamp="2024-01-01T00:00:00Z",
            ),
        ]
        
        diagnosis = diagnose_regime(results)
        
        self.assertEqual(diagnosis["regime_status"], "HEALTHY")
        self.assertFalse(diagnosis["could_survive_without_concepts"])
        self.assertEqual(diagnosis["honest_answer"], "NO - GOOD")


# ─────────────────────────────────────────────────────────────────────────────
# Ledger Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLedger(unittest.TestCase):
    """Tests for WORM-compliant ledger entries."""

    def test_ledger_entry_has_hash(self) -> None:
        """Ledger entry MUST have entry_hash."""
        result = GateResult(
            passed=False,
            rejection_reasons=[GateRejectionReason.NO_CONCEPT_USED],
            concepts_used=0,
            max_depth=0,
            csv_calls_total=0,
            cross_task_reuse_count=0,
            search_steps_taken=100,
            search_collapse_factor=1.0,
            budget_allocated=200,
            budget_used=100,
            budget_remaining=100,
            solution_hash="abc",
            gate_timestamp="2024-01-01T00:00:00Z",
        )
        
        entry = write_gate_result_to_ledger(result, task_id="task_1", step=0)
        
        self.assertIn("entry_hash", entry)
        self.assertEqual(len(entry["entry_hash"]), 64)
        self.assertEqual(entry["schema_version"], INEVITABILITY_GATE_SCHEMA_VERSION_V145)


# ─────────────────────────────────────────────────────────────────────────────
# Schema Version Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaVersion(unittest.TestCase):
    """Tests for schema version."""

    def test_schema_version_145(self) -> None:
        """Schema version MUST be 145."""
        self.assertEqual(INEVITABILITY_GATE_SCHEMA_VERSION_V145, 145)


if __name__ == "__main__":
    unittest.main()
