import unittest


class TestArcSolverV136(unittest.TestCase):
    def test_pruning_soundness_minimal(self) -> None:
        from atos_core.arc_solver_v136 import SolveConfigV136, solve_arc_task_v136
        from atos_core.grid_v124 import grid_from_list_v124, grid_equal_v124

        # Identity should be solvable (empty program), pruning must not eliminate it.
        inp = grid_from_list_v124([[1, 0], [0, 1]])
        out = grid_from_list_v124([[1, 0], [0, 1]])
        res = solve_arc_task_v136(train_pairs=[(inp, out)], test_in=inp, config=SolveConfigV136(max_depth=1, max_programs=200))
        self.assertEqual(res["status"], "SOLVED")
        self.assertTrue(grid_equal_v124(out, grid_from_list_v124(res["predicted_grid"])))

    def test_fail_closed_ambiguous(self) -> None:
        from atos_core.arc_solver_v136 import SolveConfigV136, solve_arc_task_v136
        from atos_core.grid_v124 import grid_from_list_v124

        # Train pair is symmetric so multiple minimal transforms fit (reflect_h/reflect_v/rotate90/rotate270),
        # but they diverge on test_in. Must return UNKNOWN (fail-closed).
        train_in = grid_from_list_v124([[1, 0], [0, 1]])
        train_out = grid_from_list_v124([[0, 1], [1, 0]])
        test_in = grid_from_list_v124([[1, 2], [3, 4]])
        res = solve_arc_task_v136(
            train_pairs=[(train_in, train_out)], test_in=test_in, config=SolveConfigV136(max_depth=1, max_programs=500)
        )
        self.assertEqual(res["status"], "UNKNOWN")
        self.assertEqual(res["failure_reason"]["kind"], "AMBIGUOUS_RULE")

    def test_no_task_id_hacks_static(self) -> None:
        import re
        from pathlib import Path

        src = Path("atos_core/arc_solver_v136.py").read_text(encoding="utf-8")
        self.assertNotRegex(src, re.compile(r"\btask_id\b"))
        self.assertNotRegex(src, re.compile(r"\bPath\("))
        self.assertNotRegex(src, re.compile(r"\bglob\("))
        self.assertNotRegex(src, re.compile(r"\brglob\("))
        self.assertNotRegex(src, re.compile(r"[0-9a-f]{8}\.json"))


if __name__ == "__main__":
    unittest.main()
