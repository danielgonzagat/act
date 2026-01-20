import re
import unittest
from pathlib import Path


class TestArcSolverV137(unittest.TestCase):
    def test_propagate_color_translate_solved(self) -> None:
        from atos_core.arc_solver_v137 import SolveConfigV137, solve_arc_task_v137
        from atos_core.grid_v124 import grid_equal_v124, grid_from_list_v124

        # One-step closure propagation along +x.
        inp = grid_from_list_v124([[2, 0, 0, 0]])
        out = grid_from_list_v124([[2, 2, 2, 2]])
        res = solve_arc_task_v137(train_pairs=[(inp, out)], test_in=inp, config=SolveConfigV137(max_depth=1, max_programs=500))
        self.assertEqual(res["status"], "SOLVED")
        self.assertTrue(grid_equal_v124(out, grid_from_list_v124(res["predicted_grid"])))

    def test_fail_closed_ambiguous(self) -> None:
        from atos_core.arc_solver_v137 import SolveConfigV137, solve_arc_task_v137
        from atos_core.grid_v124 import grid_from_list_v124

        # Same as V136: multiple minimal transforms fit the train pair but diverge on test_in.
        train_in = grid_from_list_v124([[1, 0], [0, 1]])
        train_out = grid_from_list_v124([[0, 1], [1, 0]])
        test_in = grid_from_list_v124([[1, 2], [3, 4]])
        res = solve_arc_task_v137(train_pairs=[(train_in, train_out)], test_in=test_in, config=SolveConfigV137(max_depth=1, max_programs=800))
        self.assertEqual(res["status"], "UNKNOWN")
        self.assertEqual(res["failure_reason"]["kind"], "AMBIGUOUS_RULE")

    def test_no_task_id_hacks_static(self) -> None:
        src = Path("atos_core/arc_solver_v137.py").read_text(encoding="utf-8")
        self.assertNotRegex(src, re.compile(r"\btask_id\b"))
        self.assertNotRegex(src, re.compile(r"\bPath\("))
        self.assertNotRegex(src, re.compile(r"\bglob\("))
        self.assertNotRegex(src, re.compile(r"\brglob\("))
        self.assertNotRegex(src, re.compile(r"[0-9a-f]{8}\.json"))


if __name__ == "__main__":
    unittest.main()
