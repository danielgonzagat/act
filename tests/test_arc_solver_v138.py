from __future__ import annotations

import re
import unittest

from atos_core.arc_solver_v138 import SolveConfigV138, solve_arc_task_v138
from atos_core.grid_v124 import grid_from_list_v124


class TestArcSolverV138(unittest.TestCase):
    def test_unknown_includes_predicted_grids(self) -> None:
        # Construct a case where rotate90 and rotate270 are both minimal and consistent on train,
        # but diverge on test_in -> UNKNOWN with predicted_grids present.
        train_in = grid_from_list_v124([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
        train_out = grid_from_list_v124([[0, 0, 1], [0, 2, 0], [1, 0, 0]])
        test_in = grid_from_list_v124([[1, 0, 0], [0, 2, 0], [0, 0, 0]])

        res = solve_arc_task_v138(train_pairs=[(train_in, train_out)], test_in=test_in, config=SolveConfigV138(max_depth=1, max_programs=200))
        self.assertEqual(res["status"], "UNKNOWN")
        self.assertIn("predicted_grids", res)
        preds = res["predicted_grids"]
        self.assertIsInstance(preds, list)
        self.assertGreaterEqual(len(preds), 2)
        self.assertTrue(all(isinstance(x, dict) and "grid" in x and "grid_hash" in x for x in preds))

    def test_anti_hack_scan_solver_source(self) -> None:
        # Hard rule: core solver should not branch on task_id/paths/dataset strings.
        import inspect
        import atos_core.arc_solver_v138 as m

        src = inspect.getsource(m)
        banned_substrings = ["task_id", "Path(", "glob(", "rglob(", "/ARC-AGI"]
        for s in banned_substrings:
            self.assertNotIn(s, src, msg=f"banned_substring_found:{s}")
        banned_regex = [r"\bevaluation\b", r"\btraining\b"]
        for pat in banned_regex:
            self.assertIsNone(re.search(pat, src), msg=f"banned_pattern_found:{pat}")


if __name__ == "__main__":
    unittest.main()
