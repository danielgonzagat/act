"""
Tests for parallel_planner_v142.py - Parallel Deterministic Planner for ARC.

These tests verify:
1. Determinism: same inputs → same outputs
2. Budget distribution correctness
3. Pruning heuristics
4. Log merging and WORM compliance
5. Ambiguity detection
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.parallel_planner_v142 import (
    PARALLEL_PLANNER_SCHEMA_VERSION_V142,
    ParallelPlannerV142,
    ParallelSearchConfigV142,
    ParallelSearchResultV142,
    ReachabilityStateV142,
    SearchBudgetV142,
    WorkerBudgetV142,
    WorkerLogEntryV142,
    WorkerResultV142,
    _detect_ambiguity_v142,
    _merge_worker_logs_v142,
    can_reach_target_v142,
    distribute_budget_v142,
    min_steps_shape_change_v142,
    min_steps_to_modify_v142,
    write_parallel_search_to_ledger_v142,
)


class TestSearchBudget(unittest.TestCase):
    """Tests for SearchBudgetV142 and budget distribution."""

    def test_budget_to_dict_schema_version(self) -> None:
        """Budget dict includes correct schema version."""
        budget = SearchBudgetV142(
            max_programs=1000,
            max_depth=4,
            max_expansions_per_worker=250,
            pruning_heuristic="reachability",
            seed=42,
        )
        d = budget.to_dict()
        self.assertEqual(d["schema_version"], PARALLEL_PLANNER_SCHEMA_VERSION_V142)
        self.assertEqual(d["max_programs"], 1000)
        self.assertEqual(d["seed"], 42)

    def test_distribute_budget_even_split(self) -> None:
        """Budget distribution splits operators evenly among workers."""
        budget = SearchBudgetV142(
            max_programs=1600,
            max_depth=4,
            max_expansions_per_worker=100,
            pruning_heuristic="reachability",
            seed=0,
        )

        slices = distribute_budget_v142(budget=budget, num_workers=4, operator_count=100)

        self.assertEqual(len(slices), 4)

        # Check no overlap in branch prefixes
        all_ranges = [(s.branch_prefix[0], s.branch_prefix[1]) for s in slices]
        for i, (start_i, end_i) in enumerate(all_ranges):
            for j, (start_j, end_j) in enumerate(all_ranges):
                if i != j:
                    # Ranges should not overlap
                    self.assertTrue(end_i <= start_j or end_j <= start_i, f"Overlap: {all_ranges}")

        # Check all operators covered
        covered = set()
        for s in slices:
            for op_idx in range(s.branch_prefix[0], s.branch_prefix[1]):
                covered.add(op_idx)
        self.assertEqual(covered, set(range(100)))

    def test_distribute_budget_more_workers_than_ops(self) -> None:
        """When more workers than operators, some workers get empty ranges."""
        budget = SearchBudgetV142(
            max_programs=100,
            max_depth=4,
            max_expansions_per_worker=100,
            pruning_heuristic="none",
            seed=0,
        )

        slices = distribute_budget_v142(budget=budget, num_workers=16, operator_count=4)

        self.assertEqual(len(slices), 16)

        # At most 4 workers should have non-empty ranges
        non_empty = [s for s in slices if s.branch_prefix[0] < s.branch_prefix[1]]
        self.assertLessEqual(len(non_empty), 4)

    def test_distribute_budget_deterministic(self) -> None:
        """Budget distribution is deterministic."""
        budget = SearchBudgetV142(
            max_programs=2000,
            max_depth=4,
            max_expansions_per_worker=500,
            pruning_heuristic="reachability",
            seed=123,
        )

        slices1 = distribute_budget_v142(budget=budget, num_workers=8, operator_count=50)
        slices2 = distribute_budget_v142(budget=budget, num_workers=8, operator_count=50)

        self.assertEqual(len(slices1), len(slices2))
        for s1, s2 in zip(slices1, slices2):
            self.assertEqual(s1.to_dict(), s2.to_dict())


class TestPruningHeuristics(unittest.TestCase):
    """Tests for pruning heuristics."""

    def test_min_steps_to_modify_grid(self) -> None:
        """Grid stage requires 1 step to modify."""
        self.assertEqual(min_steps_to_modify_v142("grid"), 1)

    def test_min_steps_to_modify_objset(self) -> None:
        """Objset stage requires 3 steps to modify grid."""
        self.assertEqual(min_steps_to_modify_v142("objset"), 3)

    def test_min_steps_shape_change_grow(self) -> None:
        """Shape growth from grid requires base + 1 step."""
        self.assertEqual(min_steps_shape_change_v142("grid", "grow"), 1)
        self.assertEqual(min_steps_shape_change_v142("patch", "grow"), 2)
        self.assertEqual(min_steps_shape_change_v142("objset", "grow"), 5)

    def test_can_reach_target_same_state(self) -> None:
        """Same state is always reachable."""
        self.assertTrue(
            can_reach_target_v142(
                current_shape=(3, 3),
                current_palette=(0, 1, 2),
                target_shape=(3, 3),
                target_palette=(0, 1, 2),
                stage="grid",
                steps_left=0,
            )
        )

    def test_can_reach_target_needs_modification(self) -> None:
        """Different palette with 0 steps left is unreachable."""
        self.assertFalse(
            can_reach_target_v142(
                current_shape=(3, 3),
                current_palette=(0, 1),
                target_shape=(3, 3),
                target_palette=(0, 1, 2),
                stage="grid",
                steps_left=0,
            )
        )

    def test_can_reach_target_shape_grow(self) -> None:
        """Growing shape requires sufficient steps."""
        # Need 1 step to grow from grid
        self.assertTrue(
            can_reach_target_v142(
                current_shape=(3, 3),
                current_palette=(0, 1),
                target_shape=(6, 6),
                target_palette=(0, 1),
                stage="grid",
                steps_left=1,
            )
        )
        self.assertFalse(
            can_reach_target_v142(
                current_shape=(3, 3),
                current_palette=(0, 1),
                target_shape=(6, 6),
                target_palette=(0, 1),
                stage="grid",
                steps_left=0,
            )
        )

    def test_reachability_state_key_deterministic(self) -> None:
        """ReachabilityStateV142 key is deterministic."""
        state1 = ReachabilityStateV142(shape=(3, 3), palette=(0, 1, 2), stage="grid")
        state2 = ReachabilityStateV142(shape=(3, 3), palette=(0, 1, 2), stage="grid")
        self.assertEqual(state1.key(), state2.key())

        state3 = ReachabilityStateV142(shape=(3, 4), palette=(0, 1, 2), stage="grid")
        self.assertNotEqual(state1.key(), state3.key())


class TestWorkerLogs(unittest.TestCase):
    """Tests for worker log structures and merging."""

    def test_log_entry_to_dict(self) -> None:
        """Log entry serializes correctly."""
        entry = WorkerLogEntryV142(
            timestamp_mono_ns=123456789,
            worker_id=5,
            event="test_event",
            data={"key": "value"},
        )
        d = entry.to_dict()
        self.assertEqual(d["timestamp_mono_ns"], 123456789)
        self.assertEqual(d["worker_id"], 5)
        self.assertEqual(d["event"], "test_event")
        self.assertEqual(d["data"]["key"], "value")

    def test_merge_worker_logs_deterministic_order(self) -> None:
        """Merged logs are sorted by timestamp, worker_id, event."""
        entry1 = WorkerLogEntryV142(timestamp_mono_ns=100, worker_id=1, event="a", data={})
        entry2 = WorkerLogEntryV142(timestamp_mono_ns=100, worker_id=0, event="a", data={})
        entry3 = WorkerLogEntryV142(timestamp_mono_ns=50, worker_id=2, event="b", data={})

        worker1 = WorkerResultV142(
            worker_id=1,
            budget=WorkerBudgetV142(worker_id=1, total_workers=2, programs_start=0, programs_end=100, depth_limit=4, seed_offset=0, branch_prefix=(0, 50)),
            programs_evaluated=0,
            best_program=None,
            best_loss=(999999, 999999),
            train_perfect_programs=[],
            log_entries=[entry1],
            status="ok",
            error_message="",
            wall_time_ms=0,
        )
        worker2 = WorkerResultV142(
            worker_id=0,
            budget=WorkerBudgetV142(worker_id=0, total_workers=2, programs_start=0, programs_end=100, depth_limit=4, seed_offset=0, branch_prefix=(50, 100)),
            programs_evaluated=0,
            best_program=None,
            best_loss=(999999, 999999),
            train_perfect_programs=[],
            log_entries=[entry2, entry3],
            status="ok",
            error_message="",
            wall_time_ms=0,
        )

        merged, hash_val = _merge_worker_logs_v142([worker1, worker2])

        # Should be sorted: timestamp 50 first, then timestamp 100 (worker 0 before 1)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["timestamp_mono_ns"], 50)
        self.assertEqual(merged[1]["timestamp_mono_ns"], 100)
        self.assertEqual(merged[1]["worker_id"], 0)
        self.assertEqual(merged[2]["timestamp_mono_ns"], 100)
        self.assertEqual(merged[2]["worker_id"], 1)

    def test_merge_worker_logs_hash_deterministic(self) -> None:
        """Merged log hash is deterministic."""
        entry = WorkerLogEntryV142(timestamp_mono_ns=100, worker_id=0, event="test", data={"x": 1})
        worker = WorkerResultV142(
            worker_id=0,
            budget=WorkerBudgetV142(worker_id=0, total_workers=1, programs_start=0, programs_end=100, depth_limit=4, seed_offset=0, branch_prefix=(0, 50)),
            programs_evaluated=0,
            best_program=None,
            best_loss=(999999, 999999),
            train_perfect_programs=[],
            log_entries=[entry],
            status="ok",
            error_message="",
            wall_time_ms=0,
        )

        _, hash1 = _merge_worker_logs_v142([worker])
        _, hash2 = _merge_worker_logs_v142([worker])

        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA256 hex


class TestAmbiguityDetection(unittest.TestCase):
    """Tests for ambiguity detection in train-perfect programs."""

    def test_no_ambiguity_single_program(self) -> None:
        """Single program has no ambiguity."""
        progs = [{"test_grid": [[1, 2], [3, 4]]}]
        amb, count = _detect_ambiguity_v142(train_perfect_programs=progs, limit=32)
        self.assertFalse(amb)
        self.assertEqual(count, 1)

    def test_no_ambiguity_same_output(self) -> None:
        """Multiple programs with same output have no ambiguity."""
        grid = [[1, 2], [3, 4]]
        progs = [
            {"test_grid": grid, "steps": [{"op_id": "a"}]},
            {"test_grid": grid, "steps": [{"op_id": "b"}]},
        ]
        amb, count = _detect_ambiguity_v142(train_perfect_programs=progs, limit=32)
        self.assertFalse(amb)
        self.assertEqual(count, 1)

    def test_ambiguity_different_outputs(self) -> None:
        """Programs with different outputs trigger ambiguity."""
        progs = [
            {"test_grid": [[1, 2], [3, 4]]},
            {"test_grid": [[5, 6], [7, 8]]},
        ]
        amb, count = _detect_ambiguity_v142(train_perfect_programs=progs, limit=32)
        self.assertTrue(amb)
        self.assertEqual(count, 2)

    def test_ambiguity_limit_respected(self) -> None:
        """Ambiguity detection respects limit."""
        progs = [
            {"test_grid": [[i, i], [i, i]]}
            for i in range(100)
        ]
        amb, count = _detect_ambiguity_v142(train_perfect_programs=progs, limit=10)
        # Should only look at first 10, which all have different outputs
        self.assertTrue(amb)
        self.assertEqual(count, 10)


class TestParallelPlanner(unittest.TestCase):
    """Tests for ParallelPlannerV142."""

    def test_config_to_dict(self) -> None:
        """Config serializes correctly."""
        config = ParallelSearchConfigV142(
            num_workers=8,
            max_programs=1000,
            max_depth=5,
            seed=42,
        )
        d = config.to_dict()
        self.assertEqual(d["schema_version"], PARALLEL_PLANNER_SCHEMA_VERSION_V142)
        self.assertEqual(d["num_workers"], 8)
        self.assertEqual(d["max_programs"], 1000)
        self.assertEqual(d["seed"], 42)

    def test_planner_search_returns_result(self) -> None:
        """Planner search returns a valid result structure."""
        config = ParallelSearchConfigV142(
            num_workers=2,
            max_programs=10,
            max_depth=2,
            seed=0,
        )
        planner = ParallelPlannerV142(config=config)

        # Simple test case
        train_pairs = [
            ([[0, 0], [0, 0]], [[1, 1], [1, 1]]),
        ]
        test_input = [[0, 0], [0, 0]]
        operators = [{"op_id": "replace_color", "kind": "builtin"}]

        result = planner.search(
            train_pairs=train_pairs,
            test_input=test_input,
            operators=operators,
        )

        self.assertIsInstance(result, ParallelSearchResultV142)
        self.assertIn(result.status, ["SOLVED", "UNKNOWN", "FAIL"])
        self.assertIsInstance(result.worker_results, list)
        self.assertGreaterEqual(len(result.merged_log_hash), 32)

    def test_planner_result_to_dict(self) -> None:
        """Planner result serializes correctly."""
        config = ParallelSearchConfigV142(num_workers=1, max_programs=5, max_depth=2, seed=0)
        planner = ParallelPlannerV142(config=config)

        train_pairs = [([[0]], [[1]])]
        test_input = [[0]]
        operators = []

        result = planner.search(
            train_pairs=train_pairs,
            test_input=test_input,
            operators=operators,
        )

        d = result.to_dict()
        self.assertEqual(d["schema_version"], PARALLEL_PLANNER_SCHEMA_VERSION_V142)
        self.assertIn("status", d)
        self.assertIn("worker_results", d)
        self.assertIn("merged_log_hash", d)


class TestLedgerIntegration(unittest.TestCase):
    """Tests for WORM-compliant ledger integration."""

    def test_write_to_ledger_creates_file(self) -> None:
        """Writing to ledger creates a valid JSONL entry."""
        config = ParallelSearchConfigV142(num_workers=1, max_programs=5)
        result = ParallelSearchResultV142(
            status="FAIL",
            failure_reason={"kind": "TEST"},
            best_program=None,
            predicted_grid=None,
            predicted_grids=[],
            candidate_steps=[],
            train_perfect_count=0,
            ambiguity_detected=False,
            total_programs_evaluated=0,
            worker_results=[],
            merged_log_hash="abc123",
            config=config,
            wall_time_ms=100,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = str(Path(tmpdir) / "test_ledger.jsonl")

            entry_hash = write_parallel_search_to_ledger_v142(
                result=result,
                task_id="test_task_001",
                ledger_path=ledger_path,
                prev_hash="",
            )

            # Check file was created
            self.assertTrue(Path(ledger_path).exists())

            # Check content
            with open(ledger_path, "r") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)

            entry = json.loads(lines[0])
            self.assertEqual(entry["schema_version"], PARALLEL_PLANNER_SCHEMA_VERSION_V142)
            self.assertEqual(entry["task_id"], "test_task_001")
            self.assertEqual(entry["entry_hash"], entry_hash)
            self.assertEqual(entry["prev_hash"], "")

    def test_write_to_ledger_chain_hashes(self) -> None:
        """Ledger entries chain hashes correctly."""
        config = ParallelSearchConfigV142(num_workers=1)

        def make_result(status: str) -> ParallelSearchResultV142:
            return ParallelSearchResultV142(
                status=status,
                failure_reason=None,
                best_program=None,
                predicted_grid=None,
                predicted_grids=[],
                candidate_steps=[],
                train_perfect_count=0,
                ambiguity_detected=False,
                total_programs_evaluated=0,
                worker_results=[],
                merged_log_hash="hash",
                config=config,
                wall_time_ms=0,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = str(Path(tmpdir) / "chain_ledger.jsonl")

            hash1 = write_parallel_search_to_ledger_v142(
                result=make_result("FAIL"),
                task_id="task_001",
                ledger_path=ledger_path,
                prev_hash="",
            )

            hash2 = write_parallel_search_to_ledger_v142(
                result=make_result("SOLVED"),
                task_id="task_002",
                ledger_path=ledger_path,
                prev_hash=hash1,
            )

            # Check chaining
            with open(ledger_path, "r") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)

            entry2 = json.loads(lines[1])
            self.assertEqual(entry2["prev_hash"], hash1)
            self.assertNotEqual(hash1, hash2)


class TestDeterminism(unittest.TestCase):
    """Tests for deterministic behavior."""

    def test_planner_deterministic_same_seed(self) -> None:
        """Same seed produces identical results."""
        config = ParallelSearchConfigV142(
            num_workers=2,
            max_programs=10,
            max_depth=2,
            seed=12345,
        )

        train_pairs = [([[0, 1], [1, 0]], [[1, 0], [0, 1]])]
        test_input = [[0, 1], [1, 0]]
        operators = [{"op_id": "rotate90"}, {"op_id": "reflect_h"}]

        planner1 = ParallelPlannerV142(config=config)
        result1 = planner1.search(train_pairs=train_pairs, test_input=test_input, operators=operators)

        planner2 = ParallelPlannerV142(config=config)
        result2 = planner2.search(train_pairs=train_pairs, test_input=test_input, operators=operators)

        # Core metrics should match
        self.assertEqual(result1.status, result2.status)
        self.assertEqual(result1.total_programs_evaluated, result2.total_programs_evaluated)
        self.assertEqual(result1.merged_log_hash, result2.merged_log_hash)

    def test_budget_distribution_deterministic(self) -> None:
        """Budget distribution is deterministic across multiple calls."""
        budget = SearchBudgetV142(
            max_programs=500,
            max_depth=3,
            max_expansions_per_worker=100,
            pruning_heuristic="cost_bound",
            seed=999,
        )

        results = []
        for _ in range(5):
            slices = distribute_budget_v142(budget=budget, num_workers=4, operator_count=20)
            results.append([s.to_dict() for s in slices])

        # All should be identical
        for r in results[1:]:
            self.assertEqual(r, results[0])


if __name__ == "__main__":
    unittest.main()
