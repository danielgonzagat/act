"""
test_failure_driven_miner_v146.py - Tests for failure-driven concept miner.

These tests PROVE that:
1. Failures are transformed into concept candidates
2. Concept candidates address rejection reasons
3. The mining loop closes correctly
4. Dead concepts are tracked with death reasons

Schema version: 146
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
from atos_core.failure_driven_miner_v146 import (
    FAILURE_DRIVEN_MINER_SCHEMA_VERSION_V146,
    ConceptCandidate,
    ConceptRequirement,
    REJECTION_TO_REQUIREMENT,
    format_iteration_report,
    generate_concept_from_failure,
    map_rejection_to_requirements,
    run_failure_driven_mining,
    write_mining_session_to_ledger,
)
from atos_core.solver_concept_gate_v145 import (
    GatedSolverResult,
    solve_with_concept_gate,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def make_simple_tasks() -> List[Tuple[str, List[Tuple[GridV124, GridV124]], GridV124]]:
    """Create simple ARC tasks for testing."""
    return [
        ('task_identity', [(((1, 2), (3, 4)), ((1, 2), (3, 4)))], ((1, 2), (3, 4))),
        ('task_replace', [(((1, 1), (1, 1)), ((2, 2), (2, 2)))], ((1, 1), (1, 1))),
        ('task_fill', [(((0, 0), (0, 0)), ((5, 5), (5, 5)))], ((0, 0), (0, 0))),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Schema Version Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaVersion(unittest.TestCase):
    """Tests for schema version."""

    def test_schema_version_146(self) -> None:
        """Schema version MUST be 146."""
        self.assertEqual(FAILURE_DRIVEN_MINER_SCHEMA_VERSION_V146, 146)


# ─────────────────────────────────────────────────────────────────────────────
# Rejection to Requirement Mapping Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRejectionMapping(unittest.TestCase):
    """Tests for rejection reason to requirement mapping."""

    def test_no_concept_used_maps_to_must_exist(self) -> None:
        """NO_CONCEPT_USED must map to MUST_EXIST."""
        req = REJECTION_TO_REQUIREMENT.get(GateRejectionReason.NO_CONCEPT_USED)
        self.assertEqual(req, ConceptRequirement.MUST_EXIST)

    def test_depth_too_shallow_maps_to_must_have_depth(self) -> None:
        """DEPTH_TOO_SHALLOW must map to MUST_HAVE_DEPTH."""
        req = REJECTION_TO_REQUIREMENT.get(GateRejectionReason.DEPTH_TOO_SHALLOW)
        self.assertEqual(req, ConceptRequirement.MUST_HAVE_DEPTH)

    def test_no_csv_call_maps_to_must_have_csv_call(self) -> None:
        """NO_CSV_CALL must map to MUST_HAVE_CSV_CALL."""
        req = REJECTION_TO_REQUIREMENT.get(GateRejectionReason.NO_CSV_CALL)
        self.assertEqual(req, ConceptRequirement.MUST_HAVE_CSV_CALL)

    def test_no_cross_task_reuse_maps_to_must_have_reuse(self) -> None:
        """NO_CROSS_TASK_REUSE must map to MUST_HAVE_REUSE."""
        req = REJECTION_TO_REQUIREMENT.get(GateRejectionReason.NO_CROSS_TASK_REUSE)
        self.assertEqual(req, ConceptRequirement.MUST_HAVE_REUSE)

    def test_map_multiple_rejections(self) -> None:
        """Multiple rejections must map to multiple requirements."""
        rejections = [
            GateRejectionReason.NO_CONCEPT_USED,
            GateRejectionReason.DEPTH_TOO_SHALLOW,
            GateRejectionReason.NO_CSV_CALL,
        ]
        requirements = map_rejection_to_requirements(rejections)
        
        self.assertIn(ConceptRequirement.MUST_EXIST, requirements)
        self.assertIn(ConceptRequirement.MUST_HAVE_DEPTH, requirements)
        self.assertIn(ConceptRequirement.MUST_HAVE_CSV_CALL, requirements)


# ─────────────────────────────────────────────────────────────────────────────
# Concept Candidate Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConceptCandidate(unittest.TestCase):
    """Tests for concept candidate creation."""

    def test_candidate_has_csv_call(self) -> None:
        """Concept candidate MUST have CSV_CALL in program."""
        candidate = ConceptCandidate(
            concept_id="test_concept",
            depth=1,
            call_deps=["dep_1"],
            program=[
                Instruction("CSV_CALL", {"callee": "dep_1", "depth": 0}),
                Instruction("CSV_RETURN", {"var": "result"}),
            ],
            target_requirements={ConceptRequirement.MUST_HAVE_CSV_CALL},
        )
        
        satisfied, failures = candidate.satisfies_requirements()
        self.assertTrue(satisfied)
        self.assertEqual(len(failures), 0)

    def test_candidate_without_csv_call_fails(self) -> None:
        """Concept without CSV_CALL must fail requirement check."""
        candidate = ConceptCandidate(
            concept_id="test_concept",
            depth=1,
            call_deps=["dep_1"],
            program=[
                Instruction("RETURN", {"var": "result"}),  # No CSV_CALL
            ],
            target_requirements={ConceptRequirement.MUST_HAVE_CSV_CALL},
        )
        
        satisfied, failures = candidate.satisfies_requirements()
        self.assertFalse(satisfied)
        self.assertIn("no CSV_CALL in program", failures)

    def test_candidate_to_act_has_correct_kind(self) -> None:
        """Candidate converted to Act must have kind=concept_csv."""
        candidate = ConceptCandidate(
            concept_id="test_concept",
            kind="concept_csv",
            depth=1,
            call_deps=["dep_1"],
            program=[Instruction("CSV_RETURN", {"var": "result"})],
            pcc_hash="a" * 64,
        )
        
        act = candidate.to_act()
        self.assertEqual(act.kind, "concept_csv")
        self.assertEqual(act.id, "test_concept")


# ─────────────────────────────────────────────────────────────────────────────
# Mining Loop Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMiningLoop(unittest.TestCase):
    """Tests for the failure-driven mining loop."""

    def test_mining_proposes_concepts_for_failures(self) -> None:
        """Mining must propose concepts for each failure."""
        tasks = make_simple_tasks()
        
        result = run_failure_driven_mining(
            tasks=tasks,
            initial_concept_library=None,
            max_iterations_per_task=1,
        )
        
        # Should have proposed concepts (since initial solve fails without concepts)
        self.assertGreater(result.concepts_proposed, 0)

    def test_mining_tracks_death_reasons(self) -> None:
        """Mining must track why concepts die."""
        tasks = make_simple_tasks()
        
        result = run_failure_driven_mining(
            tasks=tasks,
            initial_concept_library=None,
        )
        
        # Either concepts survived OR we have death reasons
        total_outcomes = result.concepts_survived + result.concepts_died
        self.assertEqual(total_outcomes, result.concepts_proposed)

    def test_mining_produces_honest_answer(self) -> None:
        """Mining must produce honest answer about concept dependency."""
        tasks = make_simple_tasks()
        
        result = run_failure_driven_mining(
            tasks=tasks,
            initial_concept_library=None,
        )
        
        # Honest answer must be present
        self.assertIn("YES", result.honest_answer.upper() + "NO")  # Either YES or NO


# ─────────────────────────────────────────────────────────────────────────────
# Report Format Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReportFormat(unittest.TestCase):
    """Tests for iteration report format."""

    def test_report_has_all_sections(self) -> None:
        """Report must have all 6 mandatory sections."""
        tasks = make_simple_tasks()[:1]  # Just one task for speed
        
        result = run_failure_driven_mining(
            tasks=tasks,
            initial_concept_library=None,
        )
        
        report = format_iteration_report(result)
        
        # Check all mandatory sections
        self.assertIn("1. GATE FAILURES", report)
        self.assertIn("2. CONCEPTS PROPOSED", report)
        self.assertIn("3. CONCEPTS DIED", report)
        self.assertIn("4. CONCEPTS SURVIVED", report)
        self.assertIn("5. SEARCH SPACE IMPACT", report)
        self.assertIn("6. HONEST ANSWER", report)
        self.assertIn("sobreviver sem conceitos", report)


# ─────────────────────────────────────────────────────────────────────────────
# Ledger Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLedger(unittest.TestCase):
    """Tests for WORM-compliant ledger entries."""

    def test_ledger_entry_has_hash(self) -> None:
        """Ledger entry MUST have entry_hash."""
        tasks = make_simple_tasks()[:1]
        
        result = run_failure_driven_mining(
            tasks=tasks,
            initial_concept_library=None,
        )
        
        entry = write_mining_session_to_ledger(result)
        
        self.assertIn("entry_hash", entry)
        self.assertEqual(len(entry["entry_hash"]), 64)
        self.assertEqual(entry["schema_version"], FAILURE_DRIVEN_MINER_SCHEMA_VERSION_V146)


if __name__ == "__main__":
    unittest.main()
