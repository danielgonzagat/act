"""
test_cross_task_miner_v147.py - Tests for cross-task pattern miner.

Schema version: 147
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.grid_v124 import GridV124
from atos_core.cross_task_miner_v147 import (
    CROSS_TASK_MINER_SCHEMA_VERSION_V147,
    PatternSignature,
    classify_shape,
    extract_pattern_signature,
    format_cross_task_report,
    run_cross_task_mining,
    write_cross_task_mining_to_ledger,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures - Tasks with SHARED patterns
# ─────────────────────────────────────────────────────────────────────────────


def make_color_replace_tasks() -> List[Tuple[str, List[Tuple[GridV124, GridV124]], GridV124]]:
    """Create tasks that share the same pattern: replace one color with another."""
    return [
        # Task 1: Replace 1 with 2
        ('replace_1to2', [(((1, 1), (1, 1)), ((2, 2), (2, 2)))], ((1, 1), (1, 1))),
        # Task 2: Replace 3 with 4 (SAME PATTERN!)
        ('replace_3to4', [(((3, 3), (3, 3)), ((4, 4), (4, 4)))], ((3, 3), (3, 3))),
        # Task 3: Replace 5 with 6 (SAME PATTERN!)
        ('replace_5to6', [(((5, 5), (5, 5)), ((6, 6), (6, 6)))], ((5, 5), (5, 5))),
    ]


def make_identity_tasks() -> List[Tuple[str, List[Tuple[GridV124, GridV124]], GridV124]]:
    """Create identity tasks (output = input)."""
    return [
        ('identity_1', [(((1, 2), (3, 4)), ((1, 2), (3, 4)))], ((1, 2), (3, 4))),
        ('identity_2', [(((5, 6), (7, 8)), ((5, 6), (7, 8)))], ((5, 6), (7, 8))),
    ]


def make_mixed_tasks() -> List[Tuple[str, List[Tuple[GridV124, GridV124]], GridV124]]:
    """Create mix of different patterns."""
    return [
        # 3 color replace tasks (cluster)
        ('replace_a', [(((1, 1), (1, 1)), ((2, 2), (2, 2)))], ((1, 1), (1, 1))),
        ('replace_b', [(((3, 3), (3, 3)), ((4, 4), (4, 4)))], ((3, 3), (3, 3))),
        ('replace_c', [(((5, 5), (5, 5)), ((6, 6), (6, 6)))], ((5, 5), (5, 5))),
        # 2 identity tasks (cluster)
        ('identity_a', [(((1, 2), (3, 4)), ((1, 2), (3, 4)))], ((1, 2), (3, 4))),
        ('identity_b', [(((5, 6), (7, 8)), ((5, 6), (7, 8)))], ((5, 6), (7, 8))),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Schema Version Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaVersion(unittest.TestCase):
    """Tests for schema version."""

    def test_schema_version_147(self) -> None:
        """Schema version MUST be 147."""
        self.assertEqual(CROSS_TASK_MINER_SCHEMA_VERSION_V147, 147)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Signature Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPatternSignature(unittest.TestCase):
    """Tests for pattern signature extraction."""

    def test_classify_square(self) -> None:
        """Square grids must be classified as 'square'."""
        self.assertEqual(classify_shape(3, 3), "square")
        self.assertEqual(classify_shape(5, 5), "square")

    def test_classify_rect_h(self) -> None:
        """Tall grids must be classified as 'rect_h'."""
        self.assertEqual(classify_shape(5, 3), "rect_h")

    def test_classify_rect_w(self) -> None:
        """Wide grids must be classified as 'rect_w'."""
        self.assertEqual(classify_shape(3, 5), "rect_w")

    def test_signature_hashable(self) -> None:
        """Pattern signatures must be hashable for use as dict keys."""
        sig1 = PatternSignature(
            ops_tuple=("color_op",),
            input_shape_class="square",
            output_shape_class="square",
            color_change=True,
            size_change=False,
        )
        sig2 = PatternSignature(
            ops_tuple=("color_op",),
            input_shape_class="square",
            output_shape_class="square",
            color_change=True,
            size_change=False,
        )
        
        # Same signatures should be equal
        self.assertEqual(sig1, sig2)
        self.assertEqual(hash(sig1), hash(sig2))
        
        # Should work as dict key
        d = {sig1: "test"}
        self.assertEqual(d[sig2], "test")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Task Mining Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCrossTaskMining(unittest.TestCase):
    """Tests for cross-task mining."""

    def test_mining_finds_clusters(self) -> None:
        """Mining must find pattern clusters."""
        tasks = make_mixed_tasks()
        
        result = run_cross_task_mining(
            tasks=tasks,
            min_cluster_size=2,
        )
        
        # Should find at least one cluster
        clusters_with_multiple = [
            sig for sig, ids in result.pattern_clusters.items()
            if len(ids) >= 2
        ]
        self.assertGreater(len(clusters_with_multiple), 0)

    def test_mining_creates_concepts_for_clusters(self) -> None:
        """Mining must create concepts for clusters with 2+ tasks."""
        tasks = make_color_replace_tasks()  # 3 tasks with same pattern
        
        result = run_cross_task_mining(
            tasks=tasks,
            min_cluster_size=2,
        )
        
        # Should create at least one concept
        self.assertGreater(len(result.concepts_created), 0)

    def test_mining_produces_honest_answer(self) -> None:
        """Mining must produce honest answer."""
        tasks = make_identity_tasks()
        
        result = run_cross_task_mining(
            tasks=tasks,
            min_cluster_size=2,
        )
        
        # Honest answer must be present
        self.assertIn("YES", result.honest_answer.upper() + "NO")


# ─────────────────────────────────────────────────────────────────────────────
# Report Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReport(unittest.TestCase):
    """Tests for report formatting."""

    def test_report_has_all_sections(self) -> None:
        """Report must have all 6 sections."""
        tasks = make_identity_tasks()
        
        result = run_cross_task_mining(tasks=tasks)
        report = format_cross_task_report(result)
        
        self.assertIn("1. PATTERN CLUSTERS", report)
        self.assertIn("2. CONCEPTS CREATED", report)
        self.assertIn("3. CONCEPTS SURVIVED", report)
        self.assertIn("4. CONCEPTS DIED", report)
        self.assertIn("5. SEARCH SPACE IMPACT", report)
        self.assertIn("6. HONEST ANSWER", report)


# ─────────────────────────────────────────────────────────────────────────────
# Ledger Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLedger(unittest.TestCase):
    """Tests for WORM ledger."""

    def test_ledger_has_hash(self) -> None:
        """Ledger entry must have entry_hash."""
        tasks = make_identity_tasks()
        
        result = run_cross_task_mining(tasks=tasks)
        entry = write_cross_task_mining_to_ledger(result)
        
        self.assertIn("entry_hash", entry)
        self.assertEqual(len(entry["entry_hash"]), 64)
        self.assertEqual(entry["schema_version"], 147)


if __name__ == "__main__":
    unittest.main()
