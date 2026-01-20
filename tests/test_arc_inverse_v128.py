from __future__ import annotations

import unittest
from typing import Any, Dict, List

from atos_core.arc_delta_v128 import compute_delta_v128
from atos_core.arc_inverse_v128 import (
    inverse_propose_paint_mask_v128,
    inverse_propose_paste_v128,
    propose_bbox_exprs_v128,
    propose_mask_exprs_v128,
)
from atos_core.arc_dsl_v128 import OP_DEFS_V128, apply_op_v128
from atos_core.grid_v124 import GridV124, grid_equal_v124, grid_from_list_v124, translate_v124


def _apply_steps_v128(steps: List[Dict[str, Any]], g_in: GridV124) -> GridV124:
    env: Dict[str, Any] = {"gC": g_in}
    for st in steps:
        op = str(st.get("op_id") or "")
        in_vars = st.get("in_vars") if isinstance(st.get("in_vars"), list) else []
        out_var = str(st.get("out_var") or "")
        args = st.get("args") if isinstance(st.get("args"), dict) else {}
        ins = [env[str(v)] for v in in_vars]
        if op in OP_DEFS_V128:
            env[out_var] = apply_op_v128(op_id=op, inputs=ins, args=dict(args))
            continue
        if op == "map_colors":
            g = ins[0]
            m_raw = args.get("mapping", {})
            if not isinstance(m_raw, dict):
                raise AssertionError("mapping_not_dict")
            m = {int(k): int(v) for k, v in m_raw.items()}
            env[out_var] = tuple(tuple(int(m.get(int(x), int(x))) for x in row) for row in g)
            continue
        if op == "translate":
            env[out_var] = translate_v124(ins[0], dx=int(args["dx"]), dy=int(args["dy"]), pad=int(args.get("pad", 0)))
            continue
        raise AssertionError(f"unknown_op:{op}")
    out = env[str(steps[-1]["out_var"])]
    if not (isinstance(out, tuple) and (not out or isinstance(out[0], tuple))):
        raise AssertionError("output_not_grid")
    return out


class TestArcInverseV128(unittest.TestCase):
    def test_inverse_paint_mask_only_bg(self) -> None:
        inp = grid_from_list_v124(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        out = grid_from_list_v124(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 2, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        delta = compute_delta_v128(inp, out)
        bbox_exprs = propose_bbox_exprs_v128(bg_candidates=[0], color_candidates=[1, 2], selector_hypotheses=[], max_expand_delta=1)
        mask_exprs = propose_mask_exprs_v128(color_candidates=[1, 2], bbox_exprs=bbox_exprs[:8], selector_hypotheses=[])
        cands = inverse_propose_paint_mask_v128(inp=inp, out=out, delta=delta, mask_exprs=mask_exprs, bg_candidates=[0])
        self.assertTrue(cands)
        # Find an exact candidate (not just partial).
        found = False
        for c in cands:
            got = _apply_steps_v128(list(c.steps), inp)
            if grid_equal_v124(got, out):
                found = True
                break
        self.assertTrue(found)

    def test_inverse_paste(self) -> None:
        inp = grid_from_list_v124(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 3, 3, 0],
                [0, 0, 3, 3, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        out = grid_from_list_v124(
            [
                [3, 3, 0, 0, 0],
                [3, 3, 0, 0, 0],
                [0, 0, 3, 3, 0],
                [0, 0, 3, 3, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        delta = compute_delta_v128(inp, out)
        bbox_exprs = propose_bbox_exprs_v128(bg_candidates=[0], color_candidates=[3], selector_hypotheses=[], max_expand_delta=1)
        cands = inverse_propose_paste_v128(inp=inp, out=out, delta=delta, bbox_exprs=bbox_exprs, bg_candidates=[0])
        self.assertTrue(cands)
        got = _apply_steps_v128(list(cands[0].steps), inp)
        self.assertTrue(grid_equal_v124(got, out))


if __name__ == "__main__":
    unittest.main()

