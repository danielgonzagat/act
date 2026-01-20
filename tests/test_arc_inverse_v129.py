from __future__ import annotations

import unittest
from typing import Any, Dict, List

from atos_core.arc_inverse_v129 import build_inverse_candidates_v129, inverse_propose_scale_up_v129, inverse_propose_symmetry_v129, inverse_propose_tile_v129
from atos_core.arc_dsl_v129 import OP_DEFS_V129, apply_op_v129
from atos_core.grid_v124 import GridV124, grid_equal_v124, grid_from_list_v124, translate_v124


def _apply_steps_v129(steps: List[Dict[str, Any]], g_in: GridV124) -> GridV124:
    env: Dict[str, Any] = {"gC": g_in}
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
        raise AssertionError(f"unknown_op:{op}")
    out = env[str(steps[-1]["out_var"])]
    if not (isinstance(out, tuple) and (not out or isinstance(out[0], tuple))):
        raise AssertionError("output_not_grid")
    return out


class TestArcInverseV129(unittest.TestCase):
    def test_inverse_symmetry(self) -> None:
        inp = grid_from_list_v124([[1, 0, 0], [0, 2, 0]])
        out = grid_from_list_v124([[0, 1], [2, 0], [0, 0]])
        cands = inverse_propose_symmetry_v129(inp=inp, out=out)
        self.assertTrue(cands)
        got = _apply_steps_v129(list(cands[0].steps), inp)
        self.assertTrue(grid_equal_v124(got, out))

    def test_inverse_scale_up(self) -> None:
        inp = grid_from_list_v124([[1, 2], [3, 4]])
        out = grid_from_list_v124(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ]
        )
        cands = inverse_propose_scale_up_v129(inp=inp, out=out)
        self.assertTrue(cands)
        got = _apply_steps_v129(list(cands[0].steps), inp)
        self.assertTrue(grid_equal_v124(got, out))

    def test_inverse_tile(self) -> None:
        inp = grid_from_list_v124([[1, 2], [3, 4]])
        out = grid_from_list_v124(
            [
                [1, 2, 1, 2, 1, 2],
                [3, 4, 3, 4, 3, 4],
                [1, 2, 1, 2, 1, 2],
                [3, 4, 3, 4, 3, 4],
            ]
        )
        cands = inverse_propose_tile_v129(inp=inp, out=out)
        self.assertTrue(cands)
        got = _apply_steps_v129(list(cands[0].steps), inp)
        self.assertTrue(grid_equal_v124(got, out))

    def test_inverse_crop_bbox(self) -> None:
        inp = grid_from_list_v124(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]
        )
        out = grid_from_list_v124([[1, 1], [1, 1]])
        cands, _trace = build_inverse_candidates_v129(inp=inp, out=out, selector_hypotheses=[])
        crop = [c for c in cands if c.op_name == "crop_bbox"]
        self.assertTrue(crop)
        got = _apply_steps_v129(list(crop[0].steps), inp)
        self.assertTrue(grid_equal_v124(got, out))


if __name__ == "__main__":
    unittest.main()
