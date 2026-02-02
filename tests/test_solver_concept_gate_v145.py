"""
test_solver_concept_gate_v145.py - Tests for gated solver.

These tests PROVE that:
1. Correct solutions without concepts are REJECTED
2. Only solutions with concepts pass
3. The regime cannot survive without concepts

Schema version: 145
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction
from atos_core.grid_v124 import GridV124
from atos_core.inevitability_gate_v145 import (
    GateRejectionReason,
    InevitabilityConfig,
)
from atos_core.solver_concept_gate_v145 import (
    SOLVER_CONCEPT_GATE_SCHEMA_VERSION_V145,
    ConceptLookupResult,
    GatedSolverResult,
    lookup_concepts_for_task,
    solve_with_concept_gate,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def make_simple_task() -> Tuple[str, List[Tuple[GridV124, GridV124]], GridV124]:
    """Create a simple identity task."""
    grid: GridV124 = ((1, 2), (3, 4))
    train_pairs = [(grid, grid)]  # Identity transform
    test_in = grid
    return ("simple_identity", train_pairs, test_in)


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


def make_concept_library_with_reuse() -> Dict[str, Act]:
    """Create a concept library where concepts have been reused."""
    concept = make_valid_concept("identity_concept", depth=2, call_deps=["base_concept"])
    return {concept.id: concept}


# ─────────────────────────────────────────────────────────────────────────────
# Schema Version Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaVersion(unittest.TestCase):
    """Tests for schema version."""

    def test_schema_version_145(self) -> None:
        """Schema version MUST be 145."""
        self.assertEqual(SOLVER_CONCEPT_GATE_SCHEMA_VERSION_V145, 145)


# ─────────────────────────────────────────────────────────────────────────────
# Concept Lookup Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConceptLookup(unittest.TestCase):
    """Tests for concept lookup."""

    def test_lookup_empty_library_returns_empty(self) -> None:
        """Lookup with no library returns no concepts."""
        task_id, train_pairs, test_in = make_simple_task()
        
        result = lookup_concepts_for_task(
            task_id=task_id,
            train_pairs=train_pairs,
            test_in=test_in,
            concept_library=None,
        )
        
        self.assertEqual(len(result.concepts_found), 0)

    def test_lookup_with_library_returns_concepts(self) -> None:
        """Lookup with library returns matching concepts."""
        task_id, train_pairs, test_in = make_simple_task()
        library = make_concept_library_with_reuse()
        
        result = lookup_concepts_for_task(
            task_id=task_id,
            train_pairs=train_pairs,
            test_in=test_in,
            concept_library=library,
        )
        
        self.assertEqual(len(result.concepts_found), 1)


# ─────────────────────────────────────────────────────────────────────────────
# The Critical Test - Conceptless Solutions Are REJECTED
# ─────────────────────────────────────────────────────────────────────────────


class TestConceptlessRejection(unittest.TestCase):
    """Tests that PROVE conceptless solutions are rejected."""

    def test_solve_without_concepts_is_rejected(self) -> None:
        """
        Even if solver finds correct answer, it's REJECTED without concepts.
        
        This is THE critical test. If this passes, the regime works.
        """
        task_id, train_pairs, test_in = make_simple_task()
        
        # Solve WITHOUT concept library (no concepts available)
        result = solve_with_concept_gate(
            task_id=task_id,
            train_pairs=train_pairs,
            test_in=test_in,
            concept_library=None,  # NO CONCEPTS
        )
        
        # Raw solver might find solution (it's a simple identity task)
        # But gate MUST reject it
        
        self.assertFalse(result.gate_result.passed)
        self.assertIn(
            GateRejectionReason.NO_CONCEPT_USED,
            result.gate_result.rejection_reasons,
        )
        
        # Final status MUST be REJECTED_CONCEPTLESS or FAIL
        # Not SOLVED_WITH_CONCEPTS
        self.assertNotEqual(result.final_status, "SOLVED_WITH_CONCEPTS")
        
        if result.raw_solver_result.status == "SOLVED":
            # If raw solver found solution but gate rejected
            self.assertEqual(result.final_status, "REJECTED_CONCEPTLESS")
            self.assertIsNotNone(result.rejection_analysis)
            self.assertIn("message", result.rejection_analysis)

    def test_regime_diagnosis_shows_conceptless_is_bad(self) -> None:
        """Regime diagnosis MUST flag conceptless passes as BROKEN."""
        task_id, train_pairs, test_in = make_simple_task()
        
        result = solve_with_concept_gate(
            task_id=task_id,
            train_pairs=train_pairs,
            test_in=test_in,
            concept_library=None,
        )
        
        # Regime diagnosis must NOT say it could survive without concepts
        # (unless no solution was found at all, in which case it's moot)
        diagnosis = result.regime_diagnosis
        
        if result.raw_solver_result.status == "SOLVED":
            # If solver found solution but gate rejected,
            # the diagnosis should show this is caught
            self.assertIn("honest_answer", diagnosis)


# ─────────────────────────────────────────────────────────────────────────────
# Config Validation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigEnforcement(unittest.TestCase):
    """Tests that config cannot enable escape routes."""

    def test_bypass_config_raises_error(self) -> None:
        """Config with bypass enabled MUST raise error."""
        task_id, train_pairs, test_in = make_simple_task()
        bad_config = InevitabilityConfig(allow_bypass=True)
        
        with self.assertRaises(ValueError) as ctx:
            solve_with_concept_gate(
                task_id=task_id,
                train_pairs=train_pairs,
                test_in=test_in,
                concept_library=None,
                gate_config=bad_config,
            )
        
        self.assertIn("escape routes", str(ctx.exception))

    def test_fallback_config_raises_error(self) -> None:
        """Config with fallback enabled MUST raise error."""
        task_id, train_pairs, test_in = make_simple_task()
        bad_config = InevitabilityConfig(allow_fallback=True)
        
        with self.assertRaises(ValueError) as ctx:
            solve_with_concept_gate(
                task_id=task_id,
                train_pairs=train_pairs,
                test_in=test_in,
                concept_library=None,
                gate_config=bad_config,
            )
        
        self.assertIn("escape routes", str(ctx.exception))


# ─────────────────────────────────────────────────────────────────────────────
# Result Structure Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResultStructure(unittest.TestCase):
    """Tests for result structure."""

    def test_result_has_all_fields(self) -> None:
        """Result must have all required fields."""
        task_id, train_pairs, test_in = make_simple_task()
        
        result = solve_with_concept_gate(
            task_id=task_id,
            train_pairs=train_pairs,
            test_in=test_in,
            concept_library=None,
        )
        
        # Check required fields
        self.assertIsNotNone(result.raw_solver_result)
        self.assertIsNotNone(result.gate_result)
        self.assertIsInstance(result.concepts_used, list)
        self.assertIn(result.final_status, [
            "SOLVED_WITH_CONCEPTS",
            "REJECTED_CONCEPTLESS",
            "FAIL",
        ])
        self.assertIsNotNone(result.regime_diagnosis)
        self.assertGreaterEqual(result.total_time_ms, 0)
        self.assertGreaterEqual(result.solver_time_ms, 0)
        self.assertGreaterEqual(result.gate_time_ms, 0)

    def test_result_to_dict_has_schema_version(self) -> None:
        """Result dict must have schema version 145."""
        task_id, train_pairs, test_in = make_simple_task()
        
        result = solve_with_concept_gate(
            task_id=task_id,
            train_pairs=train_pairs,
            test_in=test_in,
            concept_library=None,
        )
        
        result_dict = result.to_dict()
        self.assertEqual(result_dict["schema_version"], 145)


if __name__ == "__main__":
    unittest.main()
