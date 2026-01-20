from __future__ import annotations

import unittest

from atos_core.arc_solver_v131 import solve_arc_task_v131
from atos_core.grid_v124 import grid_from_list_v124


class TestArcSolverV131(unittest.TestCase):
    def test_determinism_same_seed(self) -> None:
        inp = grid_from_list_v124([[1, 2], [3, 4]])
        out = grid_from_list_v124([[3, 1], [4, 2]])  # rotate90
        test_in = grid_from_list_v124([[9, 8], [7, 6]])

        r1 = solve_arc_task_v131(train_pairs=[(inp, out)], test_in=test_in, max_depth=1, max_programs=200)
        r2 = solve_arc_task_v131(train_pairs=[(inp, out)], test_in=test_in, max_depth=1, max_programs=200)
        self.assertEqual(r1.get("status"), "SOLVED")
        self.assertEqual(r2.get("status"), "SOLVED")
        self.assertEqual(r1.get("program_sig"), r2.get("program_sig"))
        self.assertEqual(r1.get("predicted_grid_hash"), r2.get("predicted_grid_hash"))

    def test_ambiguous_rule_fail_closed(self) -> None:
        # Train pair where both rotate90 and reflect_v map inp -> out (same MDL), but they diverge on test_in.
        inp = grid_from_list_v124([[1, 2], [3, 1]])
        out = grid_from_list_v124([[3, 1], [1, 2]])
        test_in = grid_from_list_v124([[1, 0], [2, 3]])

        r = solve_arc_task_v131(train_pairs=[(inp, out)], test_in=test_in, max_depth=1, max_programs=500)
        self.assertEqual(r.get("status"), "UNKNOWN")
        fr = r.get("failure_reason")
        self.assertIsInstance(fr, dict)
        self.assertEqual(fr.get("kind"), "AMBIGUOUS_RULE")
        hashes = r.get("predicted_grid_hash_by_solution")
        self.assertIsInstance(hashes, dict)
        self.assertGreaterEqual(len(hashes.keys()), 2)

    def test_invariant_violation_classification(self) -> None:
        # Invalid grid values must fail deterministically with an invariant failure, not a crash.
        bad_inp = ((10,),)  # out of allowed 0..9
        out = grid_from_list_v124([[0]])
        r = solve_arc_task_v131(train_pairs=[(bad_inp, out)], test_in=grid_from_list_v124([[0]]))
        self.assertEqual(r.get("status"), "FAIL")
        fr = r.get("failure_reason")
        self.assertIsInstance(fr, dict)
        self.assertEqual(fr.get("kind"), "INVARIANT_VIOLATION")


if __name__ == "__main__":
    unittest.main()

