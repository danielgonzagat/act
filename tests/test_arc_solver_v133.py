import unittest

from atos_core.arc_solver_v133 import SolveConfigV133, _infer_select_obj_args_v133, solve_arc_task_v133
from atos_core.grid_v124 import GridV124


class TestArcSolverV133(unittest.TestCase):
    def test_determinism_same_input(self) -> None:
        train_in: GridV124 = (
            (0, 0, 0),
            (0, 1, 0),
            (0, 0, 0),
        )
        train_out: GridV124 = (
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 0),
        )
        test_in: GridV124 = (
            (0, 0, 0),
            (1, 0, 0),
            (0, 0, 0),
        )
        cfg = SolveConfigV133(max_depth=2, max_programs=200, trace_program_limit=20)
        res1 = solve_arc_task_v133(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)
        res2 = solve_arc_task_v133(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)
        self.assertEqual(res1.get("status"), "SOLVED")
        self.assertEqual(res2.get("status"), "SOLVED")
        self.assertEqual(res1.get("program_sig"), res2.get("program_sig"))
        self.assertEqual(res1.get("predicted_grid_hash"), res2.get("predicted_grid_hash"))

    def test_ambiguous_rule_fail_closed(self) -> None:
        train_in: GridV124 = (
            (1, 2),
            (2, 1),
        )
        # rotate90(train_in) == rotate270(train_in) for this input, but train_out != train_in
        train_out: GridV124 = (
            (2, 1),
            (1, 2),
        )
        test_in: GridV124 = (
            (1, 2),
            (3, 4),
        )
        cfg = SolveConfigV133(max_depth=1, max_programs=200, trace_program_limit=20)
        res = solve_arc_task_v133(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)
        self.assertEqual(res.get("status"), "UNKNOWN")
        fr = res.get("failure_reason")
        self.assertIsInstance(fr, dict)
        self.assertEqual(fr.get("kind"), "AMBIGUOUS_RULE")
        by_sol = res.get("predicted_grid_hash_by_solution")
        self.assertIsInstance(by_sol, dict)
        self.assertGreaterEqual(len(by_sol.keys()), 2)

    def test_infer_select_obj_args_includes_area_max(self) -> None:
        inp: GridV124 = (
            (0, 0, 0, 0, 0),
            (0, 1, 1, 0, 0),
            (0, 1, 1, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 2),
        )
        # Change a single cell inside the 1-colored block so changed_mask intersects that object only.
        out: GridV124 = (
            (0, 0, 0, 0, 0),
            (0, 3, 1, 0, 0),
            (0, 1, 1, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 2),
        )
        specs = _infer_select_obj_args_v133(train_pairs=[(inp, out)], bg=0, max_rank=1)
        self.assertGreater(len(specs), 0)
        self.assertLessEqual(len(specs), 10)
        self.assertTrue(
            any(
                d.get("key") == "area" and d.get("order") == "max" and int(d.get("rank") or 0) == 0
                for d in specs
            )
        )

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
        cfg = SolveConfigV133(max_depth=1, max_programs=50, trace_program_limit=5)
        with self.assertRaises(ValueError):
            solve_arc_task_v133(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)


if __name__ == "__main__":
    unittest.main()

