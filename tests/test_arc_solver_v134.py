import re
import unittest
from pathlib import Path

from atos_core.arc_solver_v134 import SolveConfigV134, solve_arc_task_v134
from atos_core.arc_ops_v134 import apply_op_v134
from atos_core.arc_ops_v132 import StateV132
from atos_core.grid_v124 import GridV124


class TestArcSolverV134(unittest.TestCase):
    def test_repeat_grid_upscale_nn_solved(self) -> None:
        train_in: GridV124 = (
            (1, 2),
            (3, 4),
        )
        train_out: GridV124 = (
            (1, 1, 1, 2, 2, 2),
            (1, 1, 1, 2, 2, 2),
            (3, 3, 3, 4, 4, 4),
            (3, 3, 3, 4, 4, 4),
        )
        test_in: GridV124 = (
            (9, 8),
            (7, 6),
        )
        cfg = SolveConfigV134(max_depth=1, max_programs=200, trace_program_limit=20)
        res = solve_arc_task_v134(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)
        self.assertEqual(res.get("status"), "SOLVED")
        pred = res.get("predicted_grid")
        self.assertIsInstance(pred, list)
        self.assertEqual(len(pred), 4)
        self.assertEqual(len(pred[0]), 6)

    def test_repeat_grid_tile_solved(self) -> None:
        train_in: GridV124 = (
            (1, 2),
            (3, 4),
        )
        train_out: GridV124 = (
            (1, 2, 1, 2, 1, 2),
            (3, 4, 3, 4, 3, 4),
            (1, 2, 1, 2, 1, 2),
            (3, 4, 3, 4, 3, 4),
        )
        test_in: GridV124 = (
            (5, 6),
            (7, 8),
        )
        cfg = SolveConfigV134(max_depth=1, max_programs=200, trace_program_limit=20)
        res = solve_arc_task_v134(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)
        self.assertEqual(res.get("status"), "SOLVED")
        pred = res.get("predicted_grid")
        self.assertIsInstance(pred, list)
        self.assertEqual(len(pred), 4)
        self.assertEqual(len(pred[0]), 6)

    def test_bbox_by_color_extracts_bbox(self) -> None:
        g: GridV124 = (
            (0, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 1),
            (0, 0, 0, 0),
        )
        st = StateV132(grid=g)
        st2 = apply_op_v134(state=st, op_id="bbox_by_color", args={"color": 1})
        self.assertIsNotNone(st2.bbox)
        b = st2.bbox
        self.assertEqual(b.to_tuple(), (1, 1, 3, 4))

    def test_ambiguous_rule_fail_closed(self) -> None:
        train_in: GridV124 = (
            (1, 2),
            (2, 1),
        )
        train_out: GridV124 = (
            (2, 1),
            (1, 2),
        )
        test_in: GridV124 = (
            (1, 2),
            (3, 4),
        )
        cfg = SolveConfigV134(max_depth=1, max_programs=200, trace_program_limit=20)
        res = solve_arc_task_v134(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)
        self.assertEqual(res.get("status"), "UNKNOWN")
        fr = res.get("failure_reason")
        self.assertIsInstance(fr, dict)
        self.assertEqual(fr.get("kind"), "AMBIGUOUS_RULE")
        by_sol = res.get("predicted_grid_hash_by_solution")
        self.assertIsInstance(by_sol, dict)
        self.assertGreaterEqual(len(by_sol.keys()), 2)

    def test_invalid_grid_value_raises(self) -> None:
        train_in: GridV124 = (
            (0, 0),
            (0, 0),
        )
        train_out: GridV124 = (
            (0, 0),
            (0, 0),
        )
        test_in: GridV124 = (
            (0, 10),
            (0, 0),
        )
        cfg = SolveConfigV134(max_depth=1, max_programs=50, trace_program_limit=5)
        with self.assertRaises(ValueError):
            solve_arc_task_v134(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)


class TestArcAntiHackV134(unittest.TestCase):
    def test_no_task_id_branching_in_solver(self) -> None:
        # Minimal static guardrail: solver must not branch on task_id / paths.
        solver_path = Path(__file__).resolve().parent.parent / "atos_core" / "arc_solver_v134.py"
        txt = solver_path.read_text(encoding="utf-8")
        self.assertNotIn("task_id", txt)
        self.assertNotIn("Path(", txt)
        self.assertNotIn("glob(", txt)
        self.assertNotIn("rglob(", txt)
        # No hardcoded task hex ids.
        self.assertFalse(re.search(r"\\b[0-9a-f]{8}\\.json\\b", txt))


if __name__ == "__main__":
    unittest.main()

