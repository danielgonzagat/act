"""
Tests for arc_parallel_solver_v143.py - Parallel ARC Solver Integration.

These tests verify:
1. Correctness of parallel evaluation against arc_solver_v141
2. Determinism across runs
3. Proper handling of multi-step programs
4. Ambiguity detection
5. WORM-compliant ledger writing
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
from atos_core.arc_ops_v141 import OP_DEFS_V141
from atos_core.grid_v124 import GridV124
from atos_core.arc_parallel_solver_v143 import (
    ARC_PARALLEL_SOLVER_SCHEMA_VERSION_V143,
    EvalResultV143,
    ParallelSolverConfigV143,
    ParallelSolverResultV143,
    ProgramStepV143,
    ProgramV143,
    _apply_step_v143,
    _eval_program_v143,
    _propose_next_steps_v143,
    _slots_available_v143,
    solve_arc_task_parallel_v143,
    write_parallel_solve_to_ledger_v143,
)
from atos_core.arc_ops_v132 import StateV132


class TestProgramStructures(unittest.TestCase):
    """Tests for program step and program structures."""

    def test_program_step_to_dict(self) -> None:
        """ProgramStepV143 serializes correctly."""
        step = ProgramStepV143(op_id="rotate90", args={})
        d = step.to_dict()
        self.assertEqual(d["op_id"], "rotate90")
        self.assertEqual(d["args"], {})

    def test_program_step_with_args(self) -> None:
        """ProgramStepV143 with arguments serializes correctly."""
        step = ProgramStepV143(op_id="replace_color", args={"from_color": 1, "to_color": 2})
        d = step.to_dict()
        self.assertEqual(d["op_id"], "replace_color")
        self.assertEqual(d["args"]["from_color"], 1)
        self.assertEqual(d["args"]["to_color"], 2)

    def test_program_sig_deterministic(self) -> None:
        """ProgramV143 signature is deterministic."""
        steps = (
            ProgramStepV143(op_id="rotate90", args={}),
            ProgramStepV143(op_id="reflect_h", args={}),
        )
        prog1 = ProgramV143(steps=steps)
        prog2 = ProgramV143(steps=steps)
        self.assertEqual(prog1.program_sig(), prog2.program_sig())
        self.assertEqual(len(prog1.program_sig()), 32)

    def test_program_to_dict(self) -> None:
        """ProgramV143 serializes with schema version."""
        steps = (ProgramStepV143(op_id="rotate90", args={}),)
        prog = ProgramV143(steps=steps)
        d = prog.to_dict()
        self.assertEqual(d["schema_version"], ARC_PARALLEL_SOLVER_SCHEMA_VERSION_V143)
        self.assertEqual(len(d["steps"]), 1)
        self.assertIn("program_sig", d)


class TestSlotAvailability(unittest.TestCase):
    """Tests for slot availability detection."""

    def test_initial_state_only_grid(self) -> None:
        """Initial state only has grid slot."""
        grid: GridV124 = ((0, 0), (0, 0))
        state = StateV132(grid=grid)
        avail = _slots_available_v143(state)
        self.assertTrue(avail["grid"])
        self.assertFalse(avail["objset"])
        self.assertFalse(avail["obj"])
        self.assertFalse(avail["bbox"])
        self.assertFalse(avail["patch"])


class TestApplyStep(unittest.TestCase):
    """Tests for step application with caching."""

    def test_apply_rotate90(self) -> None:
        """rotate90 applies correctly."""
        grid: GridV124 = ((1, 2), (3, 4))
        state = StateV132(grid=grid)
        apply_cache: Dict[Tuple[Any, ...], StateV132] = {}
        grid_hash_cache: Dict[GridV124, str] = {}
        metrics: Dict[str, int] = {}

        result = _apply_step_v143(
            state=state,
            op_id="rotate90",
            args={},
            apply_cache=apply_cache,
            grid_hash_cache=grid_hash_cache,
            metrics=metrics,
        )

        # Rotate90 should transform [[1,2],[3,4]] -> [[3,1],[4,2]]
        self.assertEqual(result.grid, ((3, 1), (4, 2)))
        self.assertEqual(metrics["apply_cache_misses"], 1)

    def test_apply_cache_hit(self) -> None:
        """Second application uses cache."""
        grid: GridV124 = ((1, 2), (3, 4))
        state = StateV132(grid=grid)
        apply_cache: Dict[Tuple[Any, ...], StateV132] = {}
        grid_hash_cache: Dict[GridV124, str] = {}
        metrics: Dict[str, int] = {}

        result1 = _apply_step_v143(
            state=state,
            op_id="rotate90",
            args={},
            apply_cache=apply_cache,
            grid_hash_cache=grid_hash_cache,
            metrics=metrics,
        )
        result2 = _apply_step_v143(
            state=state,
            op_id="rotate90",
            args={},
            apply_cache=apply_cache,
            grid_hash_cache=grid_hash_cache,
            metrics=metrics,
        )

        self.assertEqual(result1.grid, result2.grid)
        self.assertEqual(metrics["apply_cache_misses"], 1)
        self.assertEqual(metrics["apply_cache_hits"], 1)

    def test_apply_replace_color(self) -> None:
        """replace_color applies correctly."""
        grid: GridV124 = ((1, 1), (2, 2))
        state = StateV132(grid=grid)
        apply_cache: Dict[Tuple[Any, ...], StateV132] = {}
        grid_hash_cache: Dict[GridV124, str] = {}
        metrics: Dict[str, int] = {}

        result = _apply_step_v143(
            state=state,
            op_id="replace_color",
            args={"from_color": 1, "to_color": 3},
            apply_cache=apply_cache,
            grid_hash_cache=grid_hash_cache,
            metrics=metrics,
        )

        self.assertEqual(result.grid, ((3, 3), (2, 2)))


class TestEvalProgram(unittest.TestCase):
    """Tests for program evaluation."""

    def test_eval_empty_program(self) -> None:
        """Empty program returns input unchanged."""
        train_pairs: List[Tuple[GridV124, GridV124]] = [
            (((1, 2), (3, 4)), ((1, 2), (3, 4))),
        ]
        test_in: GridV124 = ((1, 2), (3, 4))
        apply_cache: Dict[Tuple[Any, ...], StateV132] = {}
        grid_hash_cache: Dict[GridV124, str] = {}
        eval_cache: Dict[str, EvalResultV143] = {}
        metrics: Dict[str, int] = {}

        result = _eval_program_v143(
            steps=[],
            train_pairs=train_pairs,
            test_in=test_in,
            apply_cache=apply_cache,
            grid_hash_cache=grid_hash_cache,
            eval_cache=eval_cache,
            metrics=metrics,
        )

        self.assertTrue(result.ok_train)
        self.assertEqual(result.loss_shape, 0)
        self.assertEqual(result.loss_cells, 0)

    def test_eval_rotate90_program(self) -> None:
        """Program with rotate90 evaluates correctly."""
        # Input: [[1,2],[3,4]], Expected output: [[3,1],[4,2]] (rotate90)
        train_pairs: List[Tuple[GridV124, GridV124]] = [
            (((1, 2), (3, 4)), ((3, 1), (4, 2))),
        ]
        test_in: GridV124 = ((5, 6), (7, 8))
        apply_cache: Dict[Tuple[Any, ...], StateV132] = {}
        grid_hash_cache: Dict[GridV124, str] = {}
        eval_cache: Dict[str, EvalResultV143] = {}
        metrics: Dict[str, int] = {}

        steps = [ProgramStepV143(op_id="rotate90", args={})]
        result = _eval_program_v143(
            steps=steps,
            train_pairs=train_pairs,
            test_in=test_in,
            apply_cache=apply_cache,
            grid_hash_cache=grid_hash_cache,
            eval_cache=eval_cache,
            metrics=metrics,
        )

        self.assertTrue(result.ok_train)
        self.assertEqual(result.loss, (0, 0))
        # Test output should be rotate90 of [[5,6],[7,8]] = [[7,5],[8,6]]
        self.assertEqual(result.test_grid, ((7, 5), (8, 6)))

    def test_eval_cache_hit(self) -> None:
        """Same program uses eval cache."""
        train_pairs: List[Tuple[GridV124, GridV124]] = [
            (((1, 2), (3, 4)), ((3, 1), (4, 2))),
        ]
        test_in: GridV124 = ((5, 6), (7, 8))
        apply_cache: Dict[Tuple[Any, ...], StateV132] = {}
        grid_hash_cache: Dict[GridV124, str] = {}
        eval_cache: Dict[str, EvalResultV143] = {}
        metrics: Dict[str, int] = {}

        steps = [ProgramStepV143(op_id="rotate90", args={})]
        
        result1 = _eval_program_v143(
            steps=steps,
            train_pairs=train_pairs,
            test_in=test_in,
            apply_cache=apply_cache,
            grid_hash_cache=grid_hash_cache,
            eval_cache=eval_cache,
            metrics=metrics,
        )
        result2 = _eval_program_v143(
            steps=steps,
            train_pairs=train_pairs,
            test_in=test_in,
            apply_cache=apply_cache,
            grid_hash_cache=grid_hash_cache,
            eval_cache=eval_cache,
            metrics=metrics,
        )

        self.assertEqual(result1.vec_sig, result2.vec_sig)
        self.assertEqual(metrics["eval_cache_misses"], 1)
        self.assertEqual(metrics["eval_cache_hits"], 1)

    def test_eval_mismatch_detected(self) -> None:
        """Mismatch is correctly detected."""
        train_pairs: List[Tuple[GridV124, GridV124]] = [
            (((1, 2), (3, 4)), ((5, 5), (5, 5))),  # Wrong output
        ]
        test_in: GridV124 = ((1, 2), (3, 4))
        apply_cache: Dict[Tuple[Any, ...], StateV132] = {}
        grid_hash_cache: Dict[GridV124, str] = {}
        eval_cache: Dict[str, EvalResultV143] = {}
        metrics: Dict[str, int] = {}

        steps = [ProgramStepV143(op_id="rotate90", args={})]
        result = _eval_program_v143(
            steps=steps,
            train_pairs=train_pairs,
            test_in=test_in,
            apply_cache=apply_cache,
            grid_hash_cache=grid_hash_cache,
            eval_cache=eval_cache,
            metrics=metrics,
        )

        self.assertFalse(result.ok_train)
        self.assertGreater(result.loss_cells, 0)


class TestProposeNextSteps(unittest.TestCase):
    """Tests for next step proposal."""

    def test_propose_from_initial_state(self) -> None:
        """Proposals are generated from initial state."""
        grid: GridV124 = ((1, 2), (3, 4))
        state = StateV132(grid=grid)

        proposals = _propose_next_steps_v143(
            train_final_states=[state],
            operator_ids=["rotate90", "reflect_h", "replace_color"],
            palette_out=[0, 1, 2, 3],
            shapes_out=[(2, 2)],
            max_steps=32,
        )

        self.assertGreater(len(proposals), 0)
        op_ids = [p.op_id for p in proposals]
        self.assertIn("rotate90", op_ids)
        self.assertIn("reflect_h", op_ids)

    def test_propose_respects_max_steps(self) -> None:
        """Proposals respect max_steps limit."""
        grid: GridV124 = ((1, 2), (3, 4))
        state = StateV132(grid=grid)

        proposals = _propose_next_steps_v143(
            train_final_states=[state],
            operator_ids=list(OP_DEFS_V141.keys()),
            palette_out=[0, 1, 2, 3, 4, 5],
            shapes_out=[(2, 2), (3, 3)],
            max_steps=10,
        )

        self.assertLessEqual(len(proposals), 10)


class TestParallelSolverConfig(unittest.TestCase):
    """Tests for parallel solver configuration."""

    def test_config_to_dict(self) -> None:
        """Config serializes correctly."""
        config = ParallelSolverConfigV143(
            num_workers=8,
            max_programs_per_worker=100,
            max_depth=3,
            seed=42,
        )
        d = config.to_dict()
        self.assertEqual(d["schema_version"], ARC_PARALLEL_SOLVER_SCHEMA_VERSION_V143)
        self.assertEqual(d["num_workers"], 8)
        self.assertEqual(d["max_programs_per_worker"], 100)
        self.assertEqual(d["seed"], 42)


class TestParallelSolver(unittest.TestCase):
    """Tests for the parallel solver (end-to-end)."""

    def test_solve_simple_rotate90(self) -> None:
        """Parallel solver finds rotate90 solution."""
        # Task: rotate90
        train_pairs: List[Tuple[GridV124, GridV124]] = [
            (((1, 2), (3, 4)), ((3, 1), (4, 2))),
            (((5, 6), (7, 8)), ((7, 5), (8, 6))),
        ]
        test_in: GridV124 = ((0, 1), (2, 3))

        config = ParallelSolverConfigV143(
            num_workers=4,  # Need enough workers to cover rotate90's index
            max_programs_per_worker=100,
            max_depth=2,
            seed=0,
        )

        result = solve_arc_task_parallel_v143(
            train_pairs=train_pairs,
            test_in=test_in,
            config=config,
        )

        self.assertEqual(result.status, "SOLVED")
        self.assertTrue(result.train_perfect_count >= 1)
        # Expected test output: rotate90([[0,1],[2,3]]) = [[2,0],[3,1]]
        self.assertEqual(result.predicted_grid, [[2, 0], [3, 1]])

    def test_solve_replace_color(self) -> None:
        """Parallel solver finds replace_color solution."""
        # Task: replace 1 with 5
        train_pairs: List[Tuple[GridV124, GridV124]] = [
            (((1, 0), (0, 1)), ((5, 0), (0, 5))),
            (((1, 1), (1, 1)), ((5, 5), (5, 5))),
        ]
        test_in: GridV124 = ((0, 1), (1, 0))

        config = ParallelSolverConfigV143(
            num_workers=8,  # Need enough workers to cover replace_color's index
            max_programs_per_worker=200,
            max_depth=2,
            seed=0,
        )

        result = solve_arc_task_parallel_v143(
            train_pairs=train_pairs,
            test_in=test_in,
            config=config,
        )

        self.assertEqual(result.status, "SOLVED")
        self.assertEqual(result.predicted_grid, [[0, 5], [5, 0]])

    def test_solver_deterministic(self) -> None:
        """Parallel solver is deterministic with same seed."""
        train_pairs: List[Tuple[GridV124, GridV124]] = [
            (((1, 2), (3, 4)), ((3, 1), (4, 2))),
        ]
        test_in: GridV124 = ((5, 6), (7, 8))

        config = ParallelSolverConfigV143(
            num_workers=2,
            max_programs_per_worker=20,
            max_depth=2,
            seed=123,
        )

        result1 = solve_arc_task_parallel_v143(
            train_pairs=train_pairs,
            test_in=test_in,
            config=config,
        )
        result2 = solve_arc_task_parallel_v143(
            train_pairs=train_pairs,
            test_in=test_in,
            config=config,
        )

        self.assertEqual(result1.status, result2.status)
        self.assertEqual(result1.total_programs_evaluated, result2.total_programs_evaluated)
        self.assertEqual(result1.merged_log_hash, result2.merged_log_hash)

    def test_solver_fail_no_solution(self) -> None:
        """Parallel solver returns FAIL when no solution found."""
        # Task that requires multiple steps beyond budget
        train_pairs: List[Tuple[GridV124, GridV124]] = [
            (((0, 0), (0, 0)), ((9, 9, 9), (9, 9, 9), (9, 9, 9))),  # Impossible
        ]
        test_in: GridV124 = ((0, 0), (0, 0))

        config = ParallelSolverConfigV143(
            num_workers=1,
            max_programs_per_worker=10,
            max_depth=1,
            seed=0,
        )

        result = solve_arc_task_parallel_v143(
            train_pairs=train_pairs,
            test_in=test_in,
            config=config,
        )

        self.assertEqual(result.status, "FAIL")
        self.assertIsNotNone(result.failure_reason)
        self.assertEqual(result.failure_reason["kind"], "SEARCH_BUDGET_EXCEEDED")


class TestLedgerIntegration(unittest.TestCase):
    """Tests for WORM-compliant ledger integration."""

    def test_write_to_ledger(self) -> None:
        """Writing to ledger creates valid entry."""
        config = ParallelSolverConfigV143(num_workers=1)
        result = ParallelSolverResultV143(
            status="SOLVED",
            failure_reason=None,
            best_program={"steps": [], "program_sig": "abc"},
            predicted_grid=[[1, 2], [3, 4]],
            predicted_grids=[[[1, 2], [3, 4]]],
            candidate_programs=[{"steps": [], "program_sig": "abc"}],
            train_perfect_count=1,
            ambiguity_detected=False,
            total_programs_evaluated=100,
            worker_results=[],
            merged_log_hash="hash123",
            config=config,
            wall_time_ms=500,
            metrics={"workers_completed": 1},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = str(Path(tmpdir) / "test_ledger.jsonl")

            entry_hash = write_parallel_solve_to_ledger_v143(
                result=result,
                task_id="test_task_001",
                ledger_path=ledger_path,
                prev_hash="",
            )

            self.assertTrue(Path(ledger_path).exists())
            self.assertEqual(len(entry_hash), 64)

            with open(ledger_path, "r") as f:
                entry = json.loads(f.readline())
            
            self.assertEqual(entry["schema_version"], ARC_PARALLEL_SOLVER_SCHEMA_VERSION_V143)
            self.assertEqual(entry["task_id"], "test_task_001")
            self.assertEqual(entry["entry_hash"], entry_hash)

    def test_ledger_chain_hashes(self) -> None:
        """Ledger entries chain hashes correctly."""
        config = ParallelSolverConfigV143(num_workers=1)

        def make_result(status: str) -> ParallelSolverResultV143:
            return ParallelSolverResultV143(
                status=status,
                failure_reason=None,
                best_program=None,
                predicted_grid=None,
                predicted_grids=[],
                candidate_programs=[],
                train_perfect_count=0,
                ambiguity_detected=False,
                total_programs_evaluated=0,
                worker_results=[],
                merged_log_hash="hash",
                config=config,
                wall_time_ms=0,
                metrics={},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = str(Path(tmpdir) / "chain_ledger.jsonl")

            hash1 = write_parallel_solve_to_ledger_v143(
                result=make_result("FAIL"),
                task_id="task_001",
                ledger_path=ledger_path,
                prev_hash="",
            )

            hash2 = write_parallel_solve_to_ledger_v143(
                result=make_result("SOLVED"),
                task_id="task_002",
                ledger_path=ledger_path,
                prev_hash=hash1,
            )

            with open(ledger_path, "r") as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 2)
            entry2 = json.loads(lines[1])
            self.assertEqual(entry2["prev_hash"], hash1)
            self.assertNotEqual(hash1, hash2)


if __name__ == "__main__":
    unittest.main()
