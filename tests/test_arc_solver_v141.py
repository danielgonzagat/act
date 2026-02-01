from __future__ import annotations

import re
import unittest
from pathlib import Path

from atos_core.arc_ops_v141 import apply_op_v141
from atos_core.arc_ops_v132 import StateV132
from atos_core.arc_solver_v141 import (
    SolveConfigV141,
    _csv_applicable_v141,
    _infer_direct_steps_v141,
    _resolve_concept_csg_binders_v141,
    solve_arc_task_v141,
)
from atos_core.grid_v124 import grid_from_list_v124


class TestArcSolverV141(unittest.TestCase):
    def test_determinism_same_seed(self) -> None:
        train_in = grid_from_list_v124([[1, 2], [3, 4]])
        train_out = grid_from_list_v124([[2, 1], [4, 3]])  # reflect_h
        test_in = grid_from_list_v124([[5, 6], [7, 8]])
        cfg = SolveConfigV141(max_depth=2, max_programs=200, trace_program_limit=5, max_ambiguous_outputs=8)
        r1 = solve_arc_task_v141(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)
        r2 = solve_arc_task_v141(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)
        self.assertEqual(r1.get("status"), "SOLVED")
        self.assertEqual(r1.get("predicted_grid_hash"), r2.get("predicted_grid_hash"))
        self.assertEqual(r1.get("program_sig"), r2.get("program_sig"))

    def test_ambiguous_rule_fail_closed(self) -> None:
        # reflect_h and reflect_v coincide on train, but diverge on test.
        train_in = grid_from_list_v124([[1, 2], [2, 1]])
        train_out = grid_from_list_v124([[2, 1], [1, 2]])  # both reflect_h and reflect_v
        test_in = grid_from_list_v124([[1, 2], [3, 4]])
        cfg = SolveConfigV141(max_depth=1, max_programs=100, trace_program_limit=5, max_ambiguous_outputs=8)
        r = solve_arc_task_v141(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)
        self.assertEqual(r.get("status"), "UNKNOWN")
        self.assertIsInstance(r.get("predicted_grids"), list)
        self.assertGreaterEqual(len(r.get("predicted_grids") or []), 2)

    def test_invariant_violation_classification(self) -> None:
        # Use raw tuple-of-tuples to bypass grid_from_list_v124 validation and
        # exercise solver's own invariant checks.
        train_in = ((10,),)  # type: ignore[assignment]
        train_out = grid_from_list_v124([[0]])
        test_in = grid_from_list_v124([[0]])
        cfg = SolveConfigV141(max_depth=1, max_programs=10, trace_program_limit=1, max_ambiguous_outputs=2)
        r = solve_arc_task_v141(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)
        self.assertEqual(r.get("status"), "FAIL")
        fr = r.get("failure_reason") or {}
        self.assertEqual(fr.get("kind"), "INVARIANT_VIOLATION")

    def test_cc4_nonbg_multicolor_groups_component(self) -> None:
        g = grid_from_list_v124([[1, 2], [0, 0]])
        st = StateV132(grid=g)
        st2 = apply_op_v141(state=st, op_id="cc4_nonbg_multicolor", args={"bg": 0})
        self.assertIsNotNone(st2.objset)
        objs = list(st2.objset.objects) if st2.objset is not None else []
        self.assertEqual(len(objs), 1)
        self.assertEqual(int(objs[0].area), 2)
        self.assertEqual(objs[0].bbox.to_tuple(), (0, 0, 1, 2))

    def test_cc4_color_bars_counts_components_per_color(self) -> None:
        # color 1 has 2 disconnected singletons; color 2 has 1 component.
        g = grid_from_list_v124([[1, 0, 1], [0, 2, 0]])
        st = StateV132(grid=g)
        st2 = apply_op_v141(state=st, op_id="cc4_color_bars", args={"bg": 0})
        self.assertEqual(st2.grid, grid_from_list_v124([[1, 1], [0, 2]]))
        st3 = apply_op_v141(state=st, op_id="cc4_color_bars", args={})
        self.assertEqual(st3.grid, grid_from_list_v124([[1, 1], [0, 2]]))

    def test_cc4_color_area_column_lists_color_by_area(self) -> None:
        # Disjoint monochrome objects (no multicolor adjacency).
        g = grid_from_list_v124([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2]])
        st = StateV132(grid=g)
        st2 = apply_op_v141(state=st, op_id="cc4_color_area_column", args={"bg": 0})
        self.assertEqual(st2.grid, grid_from_list_v124([[1], [1], [2], [2]]))
        st3 = apply_op_v141(state=st, op_id="cc4_color_area_column", args={})
        self.assertEqual(st3.grid, grid_from_list_v124([[1], [1], [2], [2]]))

    def test_cc4_color_area_column_infers_bg_by_mode(self) -> None:
        g = grid_from_list_v124([[9, 9, 9, 9], [9, 1, 9, 2], [9, 9, 9, 9]])
        st = StateV132(grid=g)
        st2 = apply_op_v141(state=st, op_id="cc4_color_area_column", args={})
        self.assertEqual(st2.grid, grid_from_list_v124([[1], [2]]))

    def test_cc4_nonbg_bfs_column_and_area_column_disambiguation(self) -> None:
        # Two colors touch; area-column should fail-closed, BFS-column should still work.
        g = grid_from_list_v124([[0, 0, 0], [1, 2, 0], [0, 0, 0]])
        st = StateV132(grid=g)
        st2 = apply_op_v141(state=st, op_id="cc4_nonbg_bfs_column", args={})
        self.assertEqual(st2.grid, grid_from_list_v124([[1], [2]]))
        with self.assertRaises(ValueError):
            apply_op_v141(state=st, op_id="cc4_color_area_column", args={})

    def test_color_counts_run_column_by_first_appearance(self) -> None:
        # Output is a 1-col run-length list of colors by their pixel counts, ordered by
        # first appearance in row-major order, excluding inferred background (mode).
        g = grid_from_list_v124(
            [
                [8, 8, 8, 8],
                [8, 1, 8, 8],
                [8, 2, 2, 8],
                [8, 3, 8, 8],
            ]
        )
        st = StateV132(grid=g)
        st2 = apply_op_v141(state=st, op_id="color_counts_run_column", args={})
        # bg=8; first appearance: 1,2,3 with counts 1,2,1
        self.assertEqual(st2.grid, grid_from_list_v124([[1], [2], [2], [3]]))

    def test_mask_nonbg(self) -> None:
        g = grid_from_list_v124([[0, 1], [0, 0]])
        st = StateV132(grid=g)
        st2 = apply_op_v141(state=st, op_id="mask_nonbg", args={"bg": 0})
        self.assertEqual(st2.patch, grid_from_list_v124([[0, 1], [0, 0]]))

    def test_symmetry_fill_h_and_v(self) -> None:
        # Fill from mirror into background only.
        g = grid_from_list_v124([[1, 0, 0]])
        st = StateV132(grid=g)
        st2 = apply_op_v141(state=st, op_id="symmetry_fill_h", args={"bg": 0})
        self.assertEqual(st2.grid, grid_from_list_v124([[1, 0, 1]]))

        g2 = grid_from_list_v124([[1], [0], [0]])
        st3 = StateV132(grid=g2)
        st4 = apply_op_v141(state=st3, op_id="symmetry_fill_v", args={"bg": 0})
        self.assertEqual(st4.grid, grid_from_list_v124([[1], [0], [1]]))

        g3 = grid_from_list_v124([[1, 0], [0, 0]])
        st5 = StateV132(grid=g3)
        st6 = apply_op_v141(state=st5, op_id="symmetry_fill_rot180", args={"bg": 0})
        self.assertEqual(st6.grid, grid_from_list_v124([[1, 0], [0, 1]]))

    def test_downsample_mode(self) -> None:
        g = grid_from_list_v124(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ]
        )
        st = StateV132(grid=g)
        st2 = apply_op_v141(state=st, op_id="downsample_mode", args={"sy": 2, "sx": 2})
        self.assertEqual(st2.grid, grid_from_list_v124([[1, 2], [3, 4]]))

    def test_resolve_concept_csg_binders_colors(self) -> None:
        train_in = grid_from_list_v124([[1, 0], [0, 0]])
        train_out = grid_from_list_v124([[2, 0], [0, 0]])
        test_in = grid_from_list_v124([[1, 0], [0, 0]])
        steps = [
            {"op_id": "mask_by_color", "args": {"color": {"bind": "diff_from_top1"}}},
            {"op_id": "paint_mask", "args": {"color": {"bind": "diff_to_top1"}, "mode": "overwrite"}},
        ]
        resolved = _resolve_concept_csg_binders_v141(steps=steps, train_pairs=[(train_in, train_out)], test_in=test_in)
        self.assertIsNotNone(resolved)
        rs = resolved or []
        self.assertEqual(rs[0]["args"]["color"], 1)
        self.assertEqual(rs[1]["args"]["color"], 2)

    def test_resolve_concept_csg_binders_colors_top2(self) -> None:
        # Two changed colors (tie by count) => top2 is the larger input color, by (count desc, color asc).
        train_in = grid_from_list_v124([[1, 2], [0, 0]])
        train_out = grid_from_list_v124([[3, 4], [0, 0]])
        test_in = grid_from_list_v124([[1, 2], [0, 0]])
        steps = [
            {"op_id": "mask_by_color", "args": {"color": {"bind": "diff_from_top2"}}},
            {"op_id": "paint_mask", "args": {"color": {"bind": "diff_to_top2"}, "mode": "overwrite"}},
        ]
        resolved = _resolve_concept_csg_binders_v141(steps=steps, train_pairs=[(train_in, train_out)], test_in=test_in)
        self.assertIsNotNone(resolved)
        rs = resolved or []
        self.assertEqual(rs[0]["args"]["color"], 2)
        self.assertEqual(rs[1]["args"]["color"], 4)

    def test_resolve_concept_csg_infer_color_in_out(self) -> None:
        train_in = grid_from_list_v124([[1, 0], [0, 0]])
        train_out = grid_from_list_v124([[2, 0], [0, 0]])
        test_in = grid_from_list_v124([[1, 0], [0, 0]])
        steps = [
            {"op_id": "mask_by_color", "args": {"color": {"infer": "color_in"}}},
            {"op_id": "paint_mask", "args": {"color": {"infer": "color_out"}, "mode": "overwrite"}},
        ]
        resolved = _resolve_concept_csg_binders_v141(steps=steps, train_pairs=[(train_in, train_out)], test_in=test_in)
        self.assertIsNotNone(resolved)
        rs = resolved or []
        self.assertEqual(rs[0]["args"]["color"], 1)
        self.assertEqual(rs[1]["args"]["color"], 2)

    def test_resolve_concept_csg_infer_color_map_diff(self) -> None:
        train_in = grid_from_list_v124([[1, 0], [0, 2]])
        train_out = grid_from_list_v124([[3, 0], [0, 4]])  # map 1->3, 2->4
        test_in = grid_from_list_v124([[1, 0], [0, 2]])
        steps = [{"op_id": "map_colors", "args": {"mapping": {"infer": "color_map_diff"}}}]
        resolved = _resolve_concept_csg_binders_v141(steps=steps, train_pairs=[(train_in, train_out)], test_in=test_in)
        self.assertIsNotNone(resolved)
        rs = resolved or []
        self.assertEqual(rs[0]["args"]["mapping"], {"1": 3, "2": 4})

    def test_resolve_concept_csg_binders_out_shape(self) -> None:
        train_in = grid_from_list_v124([[1, 0], [0, 0]])
        train_out = grid_from_list_v124([[2, 0, 0], [0, 0, 0]])
        test_in = grid_from_list_v124([[1, 0], [0, 0]])
        steps = [
            {
                "op_id": "new_canvas",
                "args": {"height": {"bind": "out_height"}, "width": {"bind": "out_width"}, "color": 0},
            },
        ]
        resolved = _resolve_concept_csg_binders_v141(steps=steps, train_pairs=[(train_in, train_out)], test_in=test_in)
        self.assertIsNotNone(resolved)
        rs = resolved or []
        self.assertEqual(rs[0]["args"]["height"], 2)
        self.assertEqual(rs[0]["args"]["width"], 3)

    def test_smear_nonbg(self) -> None:
        g = grid_from_list_v124([[1, 0, 0, 2], [0, 0, 3, 0]])
        st = StateV132(grid=g)
        r = apply_op_v141(state=st, op_id="smear_nonbg", args={"dir": "right", "bg": 0})
        self.assertEqual(r.grid, grid_from_list_v124([[1, 1, 1, 2], [0, 0, 3, 3]]))

    def test_direct_infer_paint_points_step(self) -> None:
        # Fixed point edits consistent across demos should be inferred as a direct concept_call.
        train_in1 = grid_from_list_v124([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        train_out1 = grid_from_list_v124([[1, 0, 0], [0, 2, 0], [0, 0, 0]])
        train_in2 = grid_from_list_v124([[3, 3, 3], [3, 3, 3], [3, 3, 3]])
        train_out2 = grid_from_list_v124([[1, 3, 3], [3, 2, 3], [3, 3, 3]])
        test_in = grid_from_list_v124([[9, 9, 9], [9, 9, 9], [9, 9, 9]])
        want = grid_from_list_v124([[1, 9, 9], [9, 2, 9], [9, 9, 9]])

        direct = _infer_direct_steps_v141(train_pairs=[(train_in1, train_out1), (train_in2, train_out2)], test_in=test_in)
        found = False
        for st in direct:
            if str(st.op_id or "") != "concept_call":
                continue
            args0 = st.args if isinstance(st.args, dict) else {}
            steps = args0.get("steps")
            if not isinstance(steps, list) or not steps:
                continue
            if not isinstance(steps[0], dict) or str(steps[0].get("op_id") or "") != "paint_points":
                continue
            st0 = StateV132(grid=test_in)
            for row in steps:
                if not isinstance(row, dict):
                    continue
                op_id = str(row.get("op_id") or "")
                a = row.get("args") if isinstance(row.get("args"), dict) else {}
                st0 = apply_op_v141(state=st0, op_id=op_id, args=dict(a))
            self.assertEqual(st0.grid, want)
            found = True
            break

        self.assertTrue(found)

    def test_builtin_cc4_patch_xform_paste_closure(self) -> None:
        # Minimal synthetic "duplicate object with patch xform" task:
        # - input: one monochrome rectangle object
        # - output: input + rotated copy pasted at a fixed location
        #
        # This exercises the builtin closure:
        #   cc4 -> select_obj -> obj_patch -> patch_rotate90 -> paste
        def _blank(h: int, w: int, v: int = 0) -> list[list[int]]:
            return [[int(v) for _ in range(int(w))] for _ in range(int(h))]

        def _place_rect(g: list[list[int]], *, top: int, left: int, h: int, w: int, color: int) -> None:
            for r in range(int(h)):
                for c in range(int(w)):
                    g[int(top) + int(r)][int(left) + int(c)] = int(color)

        def _rot90(p: list[list[int]]) -> list[list[int]]:
            # 90° clockwise rotation.
            return [list(row) for row in zip(*p[::-1])]

        def _paste(g: list[list[int]], p: list[list[int]], *, top: int, left: int, bg: int = 0) -> None:
            for r in range(len(p)):
                for c in range(len(p[0])):
                    if int(p[r][c]) != int(bg):
                        g[int(top) + int(r)][int(left) + int(c)] = int(p[r][c])

        H, W = 9, 9
        BG = 0
        C = 7
        target_top, target_left = 1, 2

        def _make_pair(obj_top: int, obj_left: int) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
            g_in = _blank(H, W, BG)
            _place_rect(g_in, top=obj_top, left=obj_left, h=2, w=3, color=C)  # non-square to force rotation
            g_out = [row[:] for row in g_in]
            patch = [row[obj_left : obj_left + 3] for row in g_in[obj_top : obj_top + 2]]
            _paste(g_out, _rot90(patch), top=target_top, left=target_left, bg=BG)
            return (grid_from_list_v124(g_in), grid_from_list_v124(g_out))

        train_pairs = [_make_pair(5, 5), _make_pair(6, 1)]
        test_in, want_out = _make_pair(2, 4)

        cfg = SolveConfigV141(
            max_depth=2,
            max_programs=800,
            trace_program_limit=10,
            max_ambiguous_outputs=8,
            abstraction_pressure=True,
            # Enable slot-progress moves so patch-space transforms are admissible before the first grid write.
            csv_allow_slot_progress=True,
        )

        r = solve_arc_task_v141(train_pairs=train_pairs, test_in=test_in, config=cfg)
        self.assertEqual(r.get("status"), "SOLVED")
        pred = r.get("predicted_grid")
        self.assertIsInstance(pred, list)
        self.assertEqual(grid_from_list_v124(pred), want_out)

        steps = r.get("program_steps") or []
        self.assertGreaterEqual(len(steps), 1)
        self.assertEqual(steps[0].get("op_id"), "concept_call")
        inner = (steps[0].get("args") or {}).get("steps") or []
        op_ids = [s.get("op_id") for s in inner if isinstance(s, dict)]
        self.assertTrue(
            any(
                op in set(op_ids)
                for op in {
                    "patch_rotate90",
                    "patch_rotate180",
                    "patch_rotate270",
                    "patch_reflect_h",
                    "patch_reflect_v",
                    "patch_transpose",
                }
            )
        )
        self.assertIn("paste", op_ids)

    def test_propagate_nonbg_translate_stride2(self) -> None:
        # Unlike smear_nonbg, propagate_nonbg_translate supports non-unit shifts (dx/dy).
        g = grid_from_list_v124([[1, 0, 0, 0, 0]])
        st = StateV132(grid=g)
        st2 = apply_op_v141(state=st, op_id="propagate_nonbg_translate", args={"dx": 2, "dy": 0, "pad": 0})
        self.assertEqual(st2.grid, grid_from_list_v124([[1, 0, 1, 0, 1]]))

    def test_repeat_grid_corners(self) -> None:
        # Sparse upscale: place the cell value only at the corners of each sy×sx block.
        g = grid_from_list_v124([[0, 7, 0], [7, 7, 7], [0, 7, 0]])
        st = StateV132(grid=g)
        st2 = apply_op_v141(state=st, op_id="repeat_grid", args={"mode": "corners", "sy": 3, "sx": 3, "bg": 0})
        want = grid_from_list_v124(
            [
                [0, 0, 0, 7, 0, 7, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 7, 0, 7, 0, 0, 0],
                [7, 0, 7, 7, 0, 7, 7, 0, 7],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [7, 0, 7, 7, 0, 7, 7, 0, 7],
                [0, 0, 0, 7, 0, 7, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 7, 0, 7, 0, 0, 0],
            ]
        )
        self.assertEqual(st2.grid, want)

    def test_relational_expand_eq_and_neq(self) -> None:
        g = grid_from_list_v124([[1, 2], [2, 1]])
        st = StateV132(grid=g)
        eq = apply_op_v141(state=st, op_id="relational_expand", args={"mode": "eq", "bg": 0})
        self.assertEqual(
            eq.grid,
            grid_from_list_v124(
                [
                    [1, 0, 0, 2],
                    [0, 1, 2, 0],
                    [0, 2, 1, 0],
                    [2, 0, 0, 1],
                ]
            ),
        )
        neq = apply_op_v141(state=st, op_id="relational_expand", args={"mode": "neq", "bg": 0})
        self.assertEqual(
            neq.grid,
            grid_from_list_v124(
                [
                    [0, 1, 2, 0],
                    [1, 0, 0, 2],
                    [2, 0, 0, 1],
                    [0, 2, 1, 0],
                ]
            ),
        )

    def test_uniform_line_expand_row_and_col_preference(self) -> None:
        # prefer=row: choose the first uniform row and embed the tiled grid into its band.
        g = grid_from_list_v124([[1, 2], [3, 3]])
        st = StateV132(grid=g)
        r = apply_op_v141(state=st, op_id="uniform_line_expand", args={"prefer": "row", "bg": 0})
        self.assertEqual(
            r.grid,
            grid_from_list_v124(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 2, 1, 2],
                    [3, 3, 3, 3],
                ]
            ),
        )

        # prefer=col: choose the first uniform column and embed the vertically tiled grid into its band.
        g2 = grid_from_list_v124([[4, 5], [4, 6]])
        st2 = StateV132(grid=g2)
        c = apply_op_v141(state=st2, op_id="uniform_line_expand", args={"prefer": "col", "bg": 0})
        self.assertEqual(
            c.grid,
            grid_from_list_v124(
                [
                    [4, 5, 0, 0],
                    [4, 6, 0, 0],
                    [4, 5, 0, 0],
                    [4, 6, 0, 0],
                ]
            ),
        )

    def test_mask_border_and_combinators(self) -> None:
        g = grid_from_list_v124(
            [
                [0, 1, 0],
                [2, 0, 2],
                [0, 1, 0],
            ]
        )
        st = StateV132(grid=g)

        mb = apply_op_v141(state=st, op_id="mask_border", args={})
        self.assertEqual(
            mb.patch,
            grid_from_list_v124(
                [
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                ]
            ),
        )

        mi = apply_op_v141(state=mb, op_id="mask_not", args={})
        self.assertEqual(
            mi.patch,
            grid_from_list_v124(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ]
            ),
        )

        mbg = apply_op_v141(state=st, op_id="mask_bg", args={"bg": 0})
        self.assertEqual(
            mbg.patch,
            grid_from_list_v124(
                [
                    [1, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1],
                ]
            ),
        )

        mc = apply_op_v141(state=mb, op_id="mask_and_color", args={"color": 1})
        self.assertEqual(
            mc.patch,
            grid_from_list_v124(
                [
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                ]
            ),
        )

        mn = apply_op_v141(state=mb, op_id="mask_and_nonbg", args={"bg": 0})
        self.assertEqual(
            mn.patch,
            grid_from_list_v124(
                [
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                ]
            ),
        )

    def test_builtin_csg_bbox_color_crop_solves_at_depth1(self) -> None:
        # Force a multi-step bbox/crop pipeline to be proposed as a single concept_call.
        # If crop_bbox_nonzero were used, the extra non-bg noise would expand the bbox and fail.
        train_pairs = [
            (
                grid_from_list_v124([[0, 3, 0], [0, 2, 0], [0, 2, 2]]),
                grid_from_list_v124([[2, 0], [2, 2]]),
            ),
            (
                grid_from_list_v124([[0, 0, 4, 0], [0, 2, 2, 0], [0, 0, 2, 0], [0, 0, 0, 0]]),
                grid_from_list_v124([[2, 2], [0, 2]]),
            ),
        ]
        test_in = grid_from_list_v124([[0, 9, 0], [0, 2, 0], [0, 2, 2]])
        want = grid_from_list_v124([[2, 0], [2, 2]])

        cfg = SolveConfigV141(max_depth=1, max_programs=4000, trace_program_limit=10, abstraction_pressure=True)
        r = solve_arc_task_v141(train_pairs=train_pairs, test_in=test_in, config=cfg)
        self.assertEqual(r.get("status"), "SOLVED")
        self.assertEqual(r.get("predicted_grid"), [list(row) for row in want])
        steps = r.get("program_steps") or []
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].get("op_id"), "concept_call")
        inner = [s.get("op_id") for s in (steps[0].get("args") or {}).get("steps") or []]
        self.assertTrue(inner)
        self.assertIn(inner[0], {"crop_bbox_by_color", "bbox_by_color", "cc4", "crop_cc4_select"})
        if inner[0] in {"bbox_by_color", "cc4"}:
            self.assertGreaterEqual(len(inner), 3)
            self.assertEqual(inner[-2:], ["crop_bbox", "commit_patch"])

    def test_best_diagonal_fill_solves_at_depth1(self) -> None:
        # best_diagonal_fill should be proposed as a single grid→grid step, then wrapped as a concept_call.
        train_pairs = [
            (
                grid_from_list_v124(
                    [
                        [0, 0, 5, 5],
                        [5, 0, 5, 5],
                        [5, 5, 0, 5],
                        [5, 5, 5, 0],
                    ]
                ),
                grid_from_list_v124(
                    [
                        [8, 0, 5, 5],
                        [5, 8, 5, 5],
                        [5, 5, 8, 5],
                        [5, 5, 5, 8],
                    ]
                ),
            ),
            (
                grid_from_list_v124(
                    [
                        [5, 5, 0, 0],
                        [5, 5, 0, 5],
                        [5, 0, 5, 5],
                        [0, 5, 5, 5],
                    ]
                ),
                grid_from_list_v124(
                    [
                        [5, 5, 0, 8],
                        [5, 5, 8, 5],
                        [5, 8, 5, 5],
                        [8, 5, 5, 5],
                    ]
                ),
            ),
        ]
        test_in = grid_from_list_v124(
            [
                [5, 0, 0, 5],
                [5, 5, 0, 5],
                [5, 5, 5, 0],
                [5, 5, 5, 5],
            ]
        )
        want = grid_from_list_v124(
            [
                [5, 8, 0, 5],
                [5, 5, 8, 5],
                [5, 5, 5, 8],
                [5, 5, 5, 5],
            ]
        )
        cfg = SolveConfigV141(max_depth=1, max_programs=2000, trace_program_limit=10, abstraction_pressure=True)
        r = solve_arc_task_v141(train_pairs=train_pairs, test_in=test_in, config=cfg)
        self.assertEqual(r.get("status"), "SOLVED")
        self.assertEqual(r.get("predicted_grid"), [list(row) for row in want])
        steps = r.get("program_steps") or []
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].get("op_id"), "concept_call")
        inner = [s.get("op_id") for s in (steps[0].get("args") or {}).get("steps") or []]
        self.assertEqual(inner, ["best_diagonal_fill"])

    def test_nest_by_color_area_solves_at_depth1(self) -> None:
        # nest_by_color_area should be inferred as a 1-step grid→grid op under abstraction pressure.
        train_pairs = [
            (
                grid_from_list_v124(
                    [
                        [0, 0, 0, 0, 2],
                        [0, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1],
                    ]
                ),
                grid_from_list_v124(
                    [
                        [1, 1, 1, 1],
                        [1, 2, 2, 1],
                        [1, 2, 2, 1],
                        [1, 1, 1, 1],
                    ]
                ),
            ),
            (
                grid_from_list_v124(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 4],
                        [0, 3, 3, 3, 3, 3, 0, 0],
                        [0, 3, 3, 3, 3, 3, 0, 0],
                        [0, 3, 3, 3, 3, 3, 0, 0],
                        [0, 3, 3, 3, 3, 3, 0, 0],
                        [0, 3, 3, 3, 3, 3, 0, 0],
                        [5, 5, 0, 0, 0, 0, 0, 0],
                        [5, 5, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                grid_from_list_v124(
                    [
                        [3, 3, 3, 3, 3],
                        [3, 5, 5, 5, 3],
                        [3, 5, 4, 5, 3],
                        [3, 5, 5, 5, 3],
                        [3, 3, 3, 3, 3],
                    ]
                ),
            ),
        ]
        test_in = grid_from_list_v124(
            [
                [0, 0, 0, 0, 7],
                [0, 6, 6, 6, 6],
                [0, 6, 6, 6, 6],
                [0, 6, 6, 6, 6],
                [0, 6, 6, 6, 6],
            ]
        )
        want = grid_from_list_v124(
            [
                [6, 6, 6, 6],
                [6, 7, 7, 6],
                [6, 7, 7, 6],
                [6, 6, 6, 6],
            ]
        )
        cfg = SolveConfigV141(max_depth=1, max_programs=4000, trace_program_limit=10, abstraction_pressure=True)
        r = solve_arc_task_v141(train_pairs=train_pairs, test_in=test_in, config=cfg)
        self.assertEqual(r.get("status"), "SOLVED")
        self.assertEqual(r.get("predicted_grid"), [list(row) for row in want])
        steps = r.get("program_steps") or []
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].get("op_id"), "concept_call")
        inner = [s.get("op_id") for s in (steps[0].get("args") or {}).get("steps") or []]
        self.assertEqual(inner, ["nest_by_color_area"])

    def test_quadrant_center_tile_solves_at_depth1(self) -> None:
        # quadrant_center_tile should be inferred as a 1-step grid→grid op, then wrapped as a concept_call.
        train_pairs = [
            (
                grid_from_list_v124(
                    [
                        [2, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [3, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                grid_from_list_v124(
                    [
                        [2, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 2, 0, 1, 0, 0],
                        [0, 0, 0, 5, 0, 0, 0],
                        [0, 0, 3, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [3, 0, 0, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                grid_from_list_v124(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 7, 0, 0, 4, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 8, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
                grid_from_list_v124(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 7, 0, 0, 4, 0],
                        [0, 7, 0, 4, 0, 0],
                        [0, 0, 5, 0, 0, 0],
                        [0, 8, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 8, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            ),
        ]
        test_in = grid_from_list_v124(
            [
                [0, 0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 3],
            ]
        )
        want = grid_from_list_v124(
            [
                [0, 0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0],
                [0, 0, 0, 5, 0, 0, 0],
                [0, 0, 1, 0, 3, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 3],
            ]
        )

        cfg = SolveConfigV141(max_depth=1, max_programs=4000, trace_program_limit=10, abstraction_pressure=True)
        r = solve_arc_task_v141(train_pairs=train_pairs, test_in=test_in, config=cfg)
        self.assertEqual(r.get("status"), "SOLVED")
        self.assertEqual(r.get("predicted_grid"), [list(row) for row in want])
        steps = r.get("program_steps") or []
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].get("op_id"), "concept_call")
        inner = [s.get("op_id") for s in (steps[0].get("args") or {}).get("steps") or []]
        self.assertEqual(inner, ["quadrant_center_tile"])

    def test_histogram_color_counts_solves_at_depth1(self) -> None:
        # histogram_color_counts should be inferred as a 1-step grid→grid op, then wrapped as a concept_call.
        train_pairs = [
            (
                grid_from_list_v124(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 2, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [3, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
                grid_from_list_v124(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0],
                        [1, 2, 3, 0, 0, 0],
                    ]
                ),
            ),
            (
                grid_from_list_v124(
                    [
                        [0, 4, 4, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [5, 0, 0, 0, 0, 0],
                        [5, 0, 0, 0, 0, 0],
                        [0, 4, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
                grid_from_list_v124(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [4, 0, 0, 0, 0, 0],
                        [4, 5, 0, 0, 0, 0],
                        [4, 5, 0, 0, 0, 0],
                    ]
                ),
            ),
        ]
        test_in = grid_from_list_v124(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [3, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        want = grid_from_list_v124(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0],
                [1, 2, 3, 0, 0, 0],
            ]
        )

        cfg = SolveConfigV141(max_depth=1, max_programs=4000, trace_program_limit=10, abstraction_pressure=True)
        r = solve_arc_task_v141(train_pairs=train_pairs, test_in=test_in, config=cfg)
        self.assertEqual(r.get("status"), "SOLVED")
        self.assertEqual(r.get("predicted_grid"), [list(row) for row in want])
        steps = r.get("program_steps") or []
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].get("op_id"), "concept_call")
        inner = [s.get("op_id") for s in (steps[0].get("args") or {}).get("steps") or []]
        self.assertEqual(inner, ["histogram_color_counts"])

    def test_csv_applicable_requires_strict_train_loss_reduction(self) -> None:
        train_in = grid_from_list_v124([[1, 0], [0, 0]])
        train_out = grid_from_list_v124([[0, 0], [0, 0]])
        st0 = StateV132(grid=train_in)

        ok = _csv_applicable_v141(
            steps=[{"op_id": "replace_color", "args": {"from_color": 1, "to_color": 0}}],
            train_final_states=[st0],
            train_pairs=[(train_in, train_out)],
            apply_cache={},
            grid_hash_cache={},
            metrics={},
            max_loss_delta=-1,
        )
        self.assertTrue(ok)

        bad = _csv_applicable_v141(
            steps=[{"op_id": "replace_color", "args": {"from_color": 2, "to_color": 0}}],
            train_final_states=[st0],
            train_pairs=[(train_in, train_out)],
            apply_cache={},
            grid_hash_cache={},
            metrics={},
            max_loss_delta=-1,
        )
        self.assertFalse(bad)

    def test_static_anti_hack_scan(self) -> None:
        # Solver/ops must not contain dataset/path/task_id conditionals.
        root = Path(__file__).resolve().parent.parent
        solver_src = (root / "atos_core" / "arc_solver_v141.py").read_text(encoding="utf-8")
        ops_src = (root / "atos_core" / "arc_ops_v141.py").read_text(encoding="utf-8")
        for src in (solver_src, ops_src):
            self.assertNotRegex(src, r"\btask_id\b")
            self.assertNotIn("Path(", src)
            self.assertNotIn("glob(", src)
            self.assertNotIn("rglob(", src)
            self.assertIsNone(re.search(r"\b[0-9a-f]{8}(?:\.json)?\b", src))
