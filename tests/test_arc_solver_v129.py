from __future__ import annotations

import unittest
from typing import Any, Dict, List

from atos_core.arc_dsl_v129 import OP_DEFS_V129, apply_op_v129
from atos_core.arc_solver_v129 import solve_arc_task_v129
from atos_core.grid_v124 import GridV124, grid_equal_v124, grid_from_list_v124, translate_v124


def _apply_program_steps_v129(steps: List[Dict[str, Any]], g_in: GridV124) -> GridV124:
    env: Dict[str, Any] = {"g0": g_in}
    for st in steps:
        op = str(st.get("op_id") or "")
        in_vars = st.get("in_vars") if isinstance(st.get("in_vars"), list) else []
        out_var = str(st.get("out_var") or "")
        args = st.get("args") if isinstance(st.get("args"), dict) else {}
        ins = [env[str(v)] for v in in_vars]
        if op in OP_DEFS_V129:
            env[out_var] = apply_op_v129(op_id=op, inputs=ins, args=dict(args))
            continue
        if op == "translate":
            env[out_var] = translate_v124(ins[0], dx=int(args["dx"]), dy=int(args["dy"]), pad=int(args.get("pad", 0)))
            continue
        if op == "map_colors":
            g = ins[0]
            m_raw = args.get("mapping", {})
            if not isinstance(m_raw, dict):
                raise AssertionError("mapping_not_dict")
            m = {int(k): int(v) for k, v in m_raw.items()}
            env[out_var] = tuple(tuple(int(m.get(int(x), int(x))) for x in row) for row in g)
            continue
        raise AssertionError(f"unknown_op:{op}")
    out = env[str(steps[-1]["out_var"])]
    if not (isinstance(out, tuple) and (not out or isinstance(out[0], tuple))):
        raise AssertionError("output_not_grid")
    return out


class TestArcSolverV129(unittest.TestCase):
    def test_solved_rotation(self) -> None:
        train_pairs = [
            (
                grid_from_list_v124([[1, 0, 0], [0, 2, 0]]),
                grid_from_list_v124([[0, 1], [2, 0], [0, 0]]),
            )
        ]
        test_in = grid_from_list_v124([[0, 0, 6], [0, 7, 0]])
        want = grid_from_list_v124([[0, 0], [7, 0], [0, 6]])
        res = solve_arc_task_v129(train_pairs=train_pairs, test_in=test_in, max_depth=2)
        self.assertEqual(str(res.get("status")), "SOLVED")
        prog = res.get("program")
        self.assertTrue(isinstance(prog, dict))
        steps = prog.get("steps")
        self.assertTrue(isinstance(steps, list))
        got = _apply_program_steps_v129(list(steps), test_in)
        self.assertTrue(grid_equal_v124(got, want))

    def test_unknown_when_minimal_programs_diverge(self) -> None:
        # Ambiguity: in both train pairs, the target is simultaneously the smallest-area and the closest-to-top-left object,
        # so both selectors are consistent on training; on test input they differ.
        train_pairs = [
            (
                grid_from_list_v124(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
                grid_from_list_v124(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                grid_from_list_v124(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 1, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
                grid_from_list_v124(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 1, 0],
                        [0, 2, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            ),
        ]
        # Test: closest-to-top-left object is the large one (top-left-ish), smallest is rightmost.
        test_in = grid_from_list_v124(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        res = solve_arc_task_v129(train_pairs=train_pairs, test_in=test_in, max_depth=3)
        self.assertEqual(str(res.get("status")), "UNKNOWN")
        m = res.get("predicted_grid_hash_by_solution")
        self.assertTrue(isinstance(m, dict))
        self.assertGreaterEqual(len(m), 2)


if __name__ == "__main__":
    unittest.main()

