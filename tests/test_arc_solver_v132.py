import unittest

from atos_core.arc_objects_v132 import BBoxV132, connected_components4_v132
from atos_core.arc_ops_v132 import StateV132, apply_op_v132
from atos_core.arc_solver_v132 import solve_arc_task_v132
from atos_core.grid_v124 import grid_from_list_v124


class TestArcSolverV132(unittest.TestCase):
    def test_cc4_deterministic(self) -> None:
        g = grid_from_list_v124(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 2],
                [0, 0, 0, 2],
            ]
        )
        os1 = connected_components4_v132(g, bg=0)
        os2 = connected_components4_v132(g, bg=0)
        self.assertEqual(os1.set_sig(), os2.set_sig())
        self.assertEqual(len(os1.objects), 2)
        # Largest area (2x2) first by canonical ordering
        self.assertEqual(os1.objects[0].color, 1)
        self.assertEqual(os1.objects[0].bbox.to_tuple(), (0, 1, 2, 3))

    def test_paint_rect_and_paste(self) -> None:
        base = grid_from_list_v124([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        st = StateV132(grid=base, bbox=BBoxV132(r0=1, c0=1, r1=3, c1=3))
        st2 = apply_op_v132(state=st, op_id="paint_rect", args={"color": 5})
        self.assertEqual(st2.grid, grid_from_list_v124([[0, 0, 0], [0, 5, 5], [0, 5, 5]]))

        # Crop a 2x2 patch and paste at (0,0)
        st3 = StateV132(grid=st2.grid, bbox=BBoxV132(r0=1, c0=1, r1=3, c1=3))
        st3 = apply_op_v132(state=st3, op_id="crop_bbox", args={})
        st3 = apply_op_v132(state=st3, op_id="paste", args={"top": 0, "left": 0, "transparent": 0})
        self.assertEqual(st3.grid, grid_from_list_v124([[5, 5, 0], [5, 5, 5], [0, 5, 5]]))

    def test_determinism_same_seed(self) -> None:
        # Identity task: empty program is unique minimal.
        inp = grid_from_list_v124([[1, 0], [0, 1]])
        out = grid_from_list_v124([[1, 0], [0, 1]])
        test_in = grid_from_list_v124([[2, 2], [2, 2]])
        r1 = solve_arc_task_v132(train_pairs=[(inp, out)], test_in=test_in, max_depth=1, max_programs=500)
        r2 = solve_arc_task_v132(train_pairs=[(inp, out)], test_in=test_in, max_depth=1, max_programs=500)
        self.assertEqual(r1["status"], "SOLVED")
        self.assertEqual(r2["status"], "SOLVED")
        self.assertEqual(r1["program_sig"], r2["program_sig"])
        self.assertEqual(r1["predicted_grid_hash"], r2["predicted_grid_hash"])

    def test_ambiguous_rule_fail_closed(self) -> None:
        # Training pair consistent with multiple 1-step transforms of equal cost.
        inp = grid_from_list_v124([[1, 2], [2, 1]])
        out = grid_from_list_v124([[2, 1], [1, 2]])
        test_in = grid_from_list_v124([[1, 2], [3, 4]])
        res = solve_arc_task_v132(train_pairs=[(inp, out)], test_in=test_in, max_depth=1, max_programs=5000)
        self.assertEqual(res["status"], "UNKNOWN")
        self.assertEqual(res["failure_reason"]["kind"], "AMBIGUOUS_RULE")
        self.assertIn("predicted_grid_hash_by_solution", res)

    def test_invariant_violation_classification(self) -> None:
        bad = ((10,),)  # value out of 0..9
        ok = grid_from_list_v124([[0]])
        res = solve_arc_task_v132(train_pairs=[(ok, ok)], test_in=bad, max_depth=1, max_programs=10)
        self.assertEqual(res["status"], "FAIL")
        self.assertEqual(res["failure_reason"]["kind"], "INVARIANT_VIOLATION")


if __name__ == "__main__":
    unittest.main()

