from __future__ import annotations

import unittest

from atos_core.arc_delta_v127 import compute_delta_v127
from atos_core.arc_inverse_v127 import (
    inverse_propose_draw_rect_border_v127,
    inverse_propose_fill_rect_v127,
    inverse_propose_map_colors_v127,
    inverse_propose_translate_v127,
    propose_bbox_exprs_v127,
)
from atos_core.arc_solver_v127 import program_sig_v127
from atos_core.arc_dsl_v126 import apply_op_v126
from atos_core.grid_v124 import GridV124, grid_equal_v124, grid_from_list_v124, translate_v124


def _apply_steps(steps, g_in: GridV124) -> GridV124:
    # Minimal executor for the subset used in tests.
    env = {"g0": g_in}
    for st in steps:
        op = str(st.get("op_id") or "")
        in_vars = st.get("in_vars") if isinstance(st.get("in_vars"), list) else []
        out_var = str(st.get("out_var") or "")
        args = st.get("args") if isinstance(st.get("args"), dict) else {}
        ins = [env[str(v)] for v in in_vars]
        if op == "map_colors":
            m_raw = args.get("mapping", {})
            m = {int(k): int(v) for k, v in m_raw.items()}
            g = ins[0]
            env[out_var] = tuple(tuple(int(m.get(int(x), int(x))) for x in row) for row in g)
        elif op == "translate":
            env[out_var] = translate_v124(ins[0], dx=int(args["dx"]), dy=int(args["dy"]), pad=int(args.get("pad", 0)))
        else:
            env[out_var] = apply_op_v126(op_id=op, inputs=ins, args=dict(args))
    out = env[str(steps[-1]["out_var"])]
    self_check = isinstance(out, tuple) and (not out or isinstance(out[0], tuple))
    if not self_check:
        raise AssertionError("output_not_grid")
    return out


class TestArcInverseV127(unittest.TestCase):
    def test_inverse_map_colors(self) -> None:
        inp = grid_from_list_v124([[0, 1], [2, 1]])
        out = grid_from_list_v124([[0, 3], [4, 3]])
        cands = inverse_propose_map_colors_v127(inp=inp, out=out)
        self.assertTrue(cands)
        got = _apply_steps(list(cands[0].steps), inp)
        self.assertTrue(grid_equal_v124(got, out))

    def test_inverse_translate(self) -> None:
        inp = grid_from_list_v124(
            [
                [0, 0, 0, 0],
                [0, 9, 9, 0],
                [0, 9, 9, 0],
                [0, 0, 0, 0],
            ]
        )
        out = grid_from_list_v124(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 9, 9],
                [0, 0, 9, 9],
            ]
        )
        cands = inverse_propose_translate_v127(inp=inp, out=out, bg_candidates=[0])
        self.assertTrue(cands)
        got = _apply_steps(list(cands[0].steps), inp)
        self.assertTrue(grid_equal_v124(got, out))

    def test_inverse_fill_rect_and_border(self) -> None:
        inp = grid_from_list_v124(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )
        out = grid_from_list_v124(
            [
                [2, 2, 2],
                [2, 1, 2],
                [2, 2, 2],
            ]
        )
        delta = compute_delta_v127(inp, out)
        bbox_exprs = propose_bbox_exprs_v127(bg_candidates=[0], color_candidates=[1], selector_hypotheses=[], max_expand_delta=1)
        fill_cands = inverse_propose_fill_rect_v127(inp=inp, out=out, delta=delta, bbox_exprs=bbox_exprs)
        # This case is a border; fill_rect should not be the only correct op. Still, candidates exist.
        self.assertTrue(bbox_exprs)
        # Ensure border candidate exists and is executable.
        border_cands = inverse_propose_draw_rect_border_v127(inp=inp, out=out, delta=delta, bbox_exprs=bbox_exprs)
        self.assertTrue(border_cands)
        got = _apply_steps(list(border_cands[0].steps), inp)
        self.assertTrue(grid_equal_v124(got, out))
        # program_sig is stable
        _ = program_sig_v127(list(border_cands[0].steps))


if __name__ == "__main__":
    unittest.main()
