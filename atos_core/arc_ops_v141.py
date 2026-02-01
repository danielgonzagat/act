from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple

from .arc_objects_v132 import BBoxV132, ObjectSetV132, ObjectV132
from .arc_ops_v132 import OpDefV132, StateV132
from .arc_ops_v137 import OP_DEFS_V137, apply_op_v137, step_cost_bits_v137
from .grid_v124 import (
    GridV124,
    grid_from_list_v124,
    grid_shape_v124,
    reflect_h_v124,
    reflect_v_v124,
    rotate180_v124,
    rotate270_v124,
    rotate90_v124,
    translate_v124,
)

ARC_OPS_SCHEMA_VERSION_V141 = 141


def _check_color_v141(c: int) -> int:
    cc = int(c)
    if cc < 0 or cc > 9:
        raise ValueError("color_out_of_range")
    return int(cc)


def _bbox_center_r2_v141(b: BBoxV132) -> int:
    r0, _, r1, _ = b.to_tuple()
    if int(r1) <= int(r0):
        return 0
    return int(r0 + r1 - 1)


def _bbox_center_c2_v141(b: BBoxV132) -> int:
    _, c0, _, c1 = b.to_tuple()
    if int(c1) <= int(c0):
        return 0
    return int(c0 + c1 - 1)


def _mode_color_in_cells_v141(g: GridV124, cells: Tuple[Tuple[int, int], ...]) -> int:
    counts: Dict[int, int] = {}
    for r, c in cells:
        vv = int(g[int(r)][int(c)])
        counts[vv] = int(counts.get(vv, 0)) + 1
    if not counts:
        return 0
    items = sorted(((int(col), int(n)) for col, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
    return int(items[0][0])


def connected_components4_nonbg_multicolor_v141(g: GridV124, *, bg: Optional[int] = None) -> ObjectSetV132:
    """
    Deterministic 4-neigh connected components over predicate (cell != bg), ignoring color.
    Each component becomes a single ObjectV132 whose cells may contain multiple colors.
    The stored ObjectV132.color is the mode color within the component (tie-break min id).
    """
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return ObjectSetV132(objects=tuple())
    bgc: Optional[int] = _check_color_v141(int(bg)) if bg is not None else None

    visited: Set[Tuple[int, int]] = set()
    objs: List[ObjectV132] = []

    for r in range(h):
        for c in range(w):
            rr = int(r)
            cc = int(c)
            if (rr, cc) in visited:
                continue
            if bgc is not None and int(g[rr][cc]) == int(bgc):
                continue
            if bgc is None:
                # bg=None means no bg filtering; treat all cells as foreground.
                pass
            q: List[Tuple[int, int]] = [(rr, cc)]
            visited.add((rr, cc))
            cells: List[Tuple[int, int]] = []
            rmin = rr
            cmin = cc
            rmax = rr
            cmax = cc
            while q:
                pr, pc = q.pop()
                cells.append((int(pr), int(pc)))
                rmin = min(int(rmin), int(pr))
                cmin = min(int(cmin), int(pc))
                rmax = max(int(rmax), int(pr))
                cmax = max(int(cmax), int(pc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(pr + dr)
                    nc = int(pc + dc)
                    if nr < 0 or nc < 0 or nr >= h or nc >= w:
                        continue
                    if (nr, nc) in visited:
                        continue
                    if bgc is not None and int(g[nr][nc]) == int(bgc):
                        continue
                    visited.add((nr, nc))
                    q.append((nr, nc))
            cells_sorted = tuple(sorted(cells, key=lambda rc: (int(rc[0]), int(rc[1]))))
            bbox = BBoxV132(r0=int(rmin), c0=int(cmin), r1=int(rmax + 1), c1=int(cmax + 1))
            rep_color = _mode_color_in_cells_v141(g, cells_sorted)
            objs.append(
                ObjectV132(
                    color=int(rep_color),
                    cells=cells_sorted,
                    bbox=bbox,
                    area=int(len(cells_sorted)),
                    width=int(bbox.width()),
                    height=int(bbox.height()),
                    bbox_center_r2=int(_bbox_center_r2_v141(bbox)),
                    bbox_center_c2=int(_bbox_center_c2_v141(bbox)),
                )
            )

    # Canonical ordering: area desc, bbox, color, cells
    objs.sort(key=lambda o: (-int(o.area), o.bbox.to_tuple(), int(o.color), o.cells))
    return ObjectSetV132(objects=tuple(objs))


OP_DEFS_V141 = dict(OP_DEFS_V137)
OP_DEFS_V141["cc4_nonbg_multicolor"] = OpDefV132(
    op_id="cc4_nonbg_multicolor",
    reads=("grid",),
    writes=("objset",),
    base_cost_bits=18,
)
OP_DEFS_V141["fill_enclosed_region"] = OpDefV132(
    op_id="fill_enclosed_region",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=24,
)
OP_DEFS_V141["cc4_color_bars"] = OpDefV132(
    op_id="cc4_color_bars",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=26,
)
OP_DEFS_V141["cc4_color_area_column"] = OpDefV132(
    op_id="cc4_color_area_column",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=28,
)
OP_DEFS_V141["cc4_nonbg_bfs_column"] = OpDefV132(
    op_id="cc4_nonbg_bfs_column",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=26,
)
OP_DEFS_V141["obj_patch"] = OpDefV132(op_id="obj_patch", reads=("grid", "obj"), writes=("patch",), base_cost_bits=18)
OP_DEFS_V141["transpose"] = OpDefV132(op_id="transpose", reads=("grid",), writes=("grid",), base_cost_bits=10)
OP_DEFS_V141["gravity"] = OpDefV132(op_id="gravity", reads=("grid",), writes=("grid",), base_cost_bits=22)
OP_DEFS_V141["mask_by_color"] = OpDefV132(op_id="mask_by_color", reads=("grid",), writes=("patch",), base_cost_bits=14)
OP_DEFS_V141["mask_bg"] = OpDefV132(op_id="mask_bg", reads=("grid",), writes=("patch",), base_cost_bits=12)
OP_DEFS_V141["mask_border"] = OpDefV132(op_id="mask_border", reads=("grid",), writes=("patch",), base_cost_bits=12)
OP_DEFS_V141["mask_nonbg"] = OpDefV132(op_id="mask_nonbg", reads=("grid",), writes=("patch",), base_cost_bits=14)
OP_DEFS_V141["mask_cross_center"] = OpDefV132(op_id="mask_cross_center", reads=("grid",), writes=("patch",), base_cost_bits=18)
OP_DEFS_V141["mask_outline"] = OpDefV132(op_id="mask_outline", reads=("patch",), writes=("patch",), base_cost_bits=14)
OP_DEFS_V141["mask_not"] = OpDefV132(op_id="mask_not", reads=("patch",), writes=("patch",), base_cost_bits=12)
OP_DEFS_V141["mask_and_color"] = OpDefV132(op_id="mask_and_color", reads=("grid", "patch"), writes=("patch",), base_cost_bits=14)
OP_DEFS_V141["mask_and_nonbg"] = OpDefV132(op_id="mask_and_nonbg", reads=("grid", "patch"), writes=("patch",), base_cost_bits=14)
OP_DEFS_V141["mask_dilate"] = OpDefV132(op_id="mask_dilate", reads=("patch",), writes=("patch",), base_cost_bits=16)
OP_DEFS_V141["mask_box_dilate"] = OpDefV132(op_id="mask_box_dilate", reads=("patch",), writes=("patch",), base_cost_bits=16)
OP_DEFS_V141["paint_mask"] = OpDefV132(op_id="paint_mask", reads=("grid", "patch"), writes=("grid",), base_cost_bits=16)
OP_DEFS_V141["paint_points"] = OpDefV132(op_id="paint_points", reads=("grid",), writes=("grid",), base_cost_bits=18)
OP_DEFS_V141["flood_fill"] = OpDefV132(op_id="flood_fill", reads=("grid", "patch"), writes=("grid",), base_cost_bits=22)
OP_DEFS_V141["symmetry_fill_h"] = OpDefV132(op_id="symmetry_fill_h", reads=("grid",), writes=("grid",), base_cost_bits=22)
OP_DEFS_V141["symmetry_fill_v"] = OpDefV132(op_id="symmetry_fill_v", reads=("grid",), writes=("grid",), base_cost_bits=22)
OP_DEFS_V141["symmetry_fill_rot180"] = OpDefV132(op_id="symmetry_fill_rot180", reads=("grid",), writes=("grid",), base_cost_bits=22)
OP_DEFS_V141["downsample_mode"] = OpDefV132(op_id="downsample_mode", reads=("grid",), writes=("grid",), base_cost_bits=22)
OP_DEFS_V141["relational_expand"] = OpDefV132(op_id="relational_expand", reads=("grid",), writes=("grid",), base_cost_bits=28)
OP_DEFS_V141["uniform_line_expand"] = OpDefV132(op_id="uniform_line_expand", reads=("grid",), writes=("grid",), base_cost_bits=26)
OP_DEFS_V141["smear_nonbg"] = OpDefV132(op_id="smear_nonbg", reads=("grid",), writes=("grid",), base_cost_bits=18)
OP_DEFS_V141["patch_rotate90"] = OpDefV132(op_id="patch_rotate90", reads=("patch",), writes=("patch",), base_cost_bits=10)
OP_DEFS_V141["patch_rotate180"] = OpDefV132(op_id="patch_rotate180", reads=("patch",), writes=("patch",), base_cost_bits=10)
OP_DEFS_V141["patch_rotate270"] = OpDefV132(op_id="patch_rotate270", reads=("patch",), writes=("patch",), base_cost_bits=10)
OP_DEFS_V141["patch_reflect_h"] = OpDefV132(op_id="patch_reflect_h", reads=("patch",), writes=("patch",), base_cost_bits=10)
OP_DEFS_V141["patch_reflect_v"] = OpDefV132(op_id="patch_reflect_v", reads=("patch",), writes=("patch",), base_cost_bits=10)
OP_DEFS_V141["patch_transpose"] = OpDefV132(op_id="patch_transpose", reads=("patch",), writes=("patch",), base_cost_bits=10)
OP_DEFS_V141["patch_translate"] = OpDefV132(op_id="patch_translate", reads=("patch",), writes=("patch",), base_cost_bits=12)
OP_DEFS_V141["crop_bbox_by_color"] = OpDefV132(op_id="crop_bbox_by_color", reads=("grid",), writes=("grid",), base_cost_bits=18)
OP_DEFS_V141["crop_cc4_select"] = OpDefV132(op_id="crop_cc4_select", reads=("grid",), writes=("grid",), base_cost_bits=28)
OP_DEFS_V141["tile_signature_fill"] = OpDefV132(op_id="tile_signature_fill", reads=("grid",), writes=("grid",), base_cost_bits=34)
OP_DEFS_V141["diagonal_fill"] = OpDefV132(op_id="diagonal_fill", reads=("grid",), writes=("grid",), base_cost_bits=18)
OP_DEFS_V141["best_diagonal_fill"] = OpDefV132(op_id="best_diagonal_fill", reads=("grid",), writes=("grid",), base_cost_bits=22)
OP_DEFS_V141["nest_by_color_area"] = OpDefV132(op_id="nest_by_color_area", reads=("grid",), writes=("grid",), base_cost_bits=30)
OP_DEFS_V141["quadrant_center_tile"] = OpDefV132(op_id="quadrant_center_tile", reads=("grid",), writes=("grid",), base_cost_bits=26)
OP_DEFS_V141["histogram_color_counts"] = OpDefV132(op_id="histogram_color_counts", reads=("grid",), writes=("grid",), base_cost_bits=24)
OP_DEFS_V141["color_counts_run_column"] = OpDefV132(op_id="color_counts_run_column", reads=("grid",), writes=("grid",), base_cost_bits=26)
OP_DEFS_V141["propagate_nonbg_translate"] = OpDefV132(op_id="propagate_nonbg_translate", reads=("grid",), writes=("grid",), base_cost_bits=22)


def step_cost_bits_v141(*, op_id: str, args: Dict[str, Any]) -> int:
    op = str(op_id)
    # Extend repeat_grid with a sparse upscale mode ("corners") without touching v134/v135/v137.
    # This is a generic grid→grid operator and remains deterministic and fail-closed.
    if op == "repeat_grid":
        mode = str(dict(args).get("mode") or "")
        if mode == "corners":
            od = OP_DEFS_V141.get(op)
            base = int(od.base_cost_bits) if od is not None else 24

            def _int_bits(x: int) -> int:
                xx = int(x)
                if xx == 0:
                    return 1
                return int(abs(xx).bit_length() + 1)

            a = dict(args)
            sy = int(a.get("sy", 1))
            sx = int(a.get("sx", 1))
            bg = int(a.get("bg", 0))
            extra = 0
            extra += 3  # mode tag richness (cell/grid/corners)
            extra += _int_bits(int(sy))
            extra += _int_bits(int(sx))
            extra += 4  # bg in 0..9
            return int(base + int(extra))
    if op == "relational_expand":
        od = OP_DEFS_V141.get(op)
        base = int(od.base_cost_bits) if od is not None else 24
        a = dict(args)
        extra = 0
        extra += 2  # mode tag (eq/neq)
        extra += 4  # bg in 0..9
        if "mode" not in a:
            extra += 2
        if "bg" not in a:
            extra += 2
        return int(base + int(extra))
    if op == "uniform_line_expand":
        od = OP_DEFS_V141.get(op)
        base = int(od.base_cost_bits) if od is not None else 24
        a = dict(args)
        extra = 0
        extra += 2  # prefer tag (row/col)
        extra += 4  # bg in 0..9
        if "prefer" not in a:
            extra += 2
        if "bg" not in a:
            extra += 2
        return int(base + int(extra))
    if op == "smear_nonbg":
        od = OP_DEFS_V141.get(op)
        base = int(od.base_cost_bits) if od is not None else 16
        a = dict(args)
        extra = 0
        extra += 2  # dir tag
        extra += 4  # bg in 0..9
        if "dir" not in a:
            extra += 2
        if "bg" not in a:
            extra += 2
        return int(base + int(extra))
    if op == "crop_cc4_select":
        od = OP_DEFS_V141.get(op)
        base = int(od.base_cost_bits) if od is not None else 24
        a = dict(args)
        extra = 0
        extra += 4  # bg in 0..9
        extra += 3  # key enum
        extra += 1  # order enum
        extra += 2  # rank small int
        if a.get("color_filter") is not None:
            extra += 4
        return int(base + int(extra))
    if op == "tile_signature_fill":
        od = OP_DEFS_V141.get(op)
        base = int(od.base_cost_bits) if od is not None else 28
        a = dict(args)
        tile_h = int(a.get("tile_h", 1))
        tile_w = int(a.get("tile_w", 1))
        bg = a.get("bg", None)
        mapping = a.get("mapping") if isinstance(a.get("mapping"), dict) else {}

        def _int_bits(x: int) -> int:
            xx = int(x)
            if xx == 0:
                return 1
            return int(abs(xx).bit_length() + 1)

        extra = 0
        extra += _int_bits(int(tile_h))
        extra += _int_bits(int(tile_w))
        if bg is not None:
            extra += 4
        extra += 6 * int(len(mapping))  # signature→color table
        return int(base + int(extra))
    if op == "diagonal_fill":
        od = OP_DEFS_V141.get(op)
        base = int(od.base_cost_bits) if od is not None else 18
        a = dict(args)
        k = int(a.get("k", 0))

        def _int_bits(x: int) -> int:
            xx = int(x)
            if xx == 0:
                return 1
            return int(abs(xx).bit_length() + 1)

        extra = 0
        extra += 2  # kind tag (main/anti)
        extra += _int_bits(int(k))
        extra += 4  # from_color in 0..9
        extra += 4  # to_color in 0..9
        return int(base + int(extra))
    if op == "best_diagonal_fill":
        od = OP_DEFS_V141.get(op)
        base = int(od.base_cost_bits) if od is not None else 22
        extra = 0
        extra += 4  # from_color in 0..9
        extra += 4  # to_color in 0..9
        return int(base + int(extra))
    if op == "nest_by_color_area":
        od = OP_DEFS_V141.get(op)
        base = int(od.base_cost_bits) if od is not None else 30
        extra = 0
        extra += 4  # bg in 0..9
        return int(base + int(extra))
    if op == "quadrant_center_tile":
        od = OP_DEFS_V141.get(op)
        base = int(od.base_cost_bits) if od is not None else 26
        a = dict(args)
        extra = 0
        if "bg" in a:
            extra += 4  # bg in 0..9
        extra += 4  # center_color in 0..9
        return int(base + int(extra))
    if op == "histogram_color_counts":
        od = OP_DEFS_V141.get(op)
        base = int(od.base_cost_bits) if od is not None else 24
        a = dict(args)
        extra = 0
        extra += 4  # bg in 0..9
        if "x0" in a:
            extra += 3  # small int (0..7 typical)
        return int(base + int(extra))
    if op == "color_counts_run_column":
        od = OP_DEFS_V141.get(op)
        base = int(od.base_cost_bits) if od is not None else 26
        a = dict(args)
        extra = 0
        if "bg" in a:
            extra += 4  # bg in 0..9
        return int(base + int(extra))
    if op == "propagate_nonbg_translate":
        od = OP_DEFS_V141.get(op)
        base = int(od.base_cost_bits) if od is not None else 22

        def _int_bits(x: int) -> int:
            xx = int(x)
            if xx == 0:
                return 1
            return int(abs(xx).bit_length() + 1)

        a = dict(args)
        dx = int(a.get("dx", 0))
        dy = int(a.get("dy", 0))
        pad = int(a.get("pad", 0))
        extra = 0
        extra += _int_bits(int(dx))
        extra += _int_bits(int(dy))
        extra += 4  # pad in 0..9
        return int(base + int(extra))
    if op == "paint_points":
        od = OP_DEFS_V141.get(op)
        base = int(od.base_cost_bits) if od is not None else 18

        def _int_bits(x: int) -> int:
            xx = int(x)
            if xx == 0:
                return 1
            return int(abs(xx).bit_length() + 1)

        a = dict(args)
        pts = a.get("points")
        mode = str(a.get("mode") or "overwrite")
        extra = 0
        extra += 2  # mode tag
        if mode == "only_bg":
            extra += 4  # bg in 0..9

        n = 0
        if isinstance(pts, list):
            for row in pts:
                if isinstance(row, dict):
                    extra += _int_bits(int(row.get("r", 0)))
                    extra += _int_bits(int(row.get("c", 0)))
                    extra += 4  # color in 0..9
                    n += 1
                elif isinstance(row, (list, tuple)) and len(row) == 3:
                    rr, cc, col = row
                    extra += _int_bits(int(rr))
                    extra += _int_bits(int(cc))
                    extra += 4
                    n += 1
        # Small fixed overhead for list framing.
        extra += min(12, int(n) + 2)
        return int(base + int(extra))
    if op == "patch_translate":
        od = OP_DEFS_V141.get(op)
        base = int(od.base_cost_bits) if od is not None else 12

        def _int_bits(x: int) -> int:
            xx = int(x)
            if xx == 0:
                return 1
            return int(abs(xx).bit_length() + 1)

        a = dict(args)
        dx = int(a.get("dx", 0))
        dy = int(a.get("dy", 0))
        pad = int(a.get("pad", 0))
        extra = 0
        extra += _int_bits(int(dx))
        extra += _int_bits(int(dy))
        extra += 4  # pad in 0..9
        return int(base + int(extra))
    if op in OP_DEFS_V137:
        return int(step_cost_bits_v137(op_id=str(op_id), args=dict(args)))
    od = OP_DEFS_V141.get(op)
    base = int(od.base_cost_bits) if od is not None else 24
    # Simple deterministic arg cost (colors are 0..9 => 4 bits).
    extra = 0
    if "bg" in args:
        extra += 4
    if "fill" in args:
        extra += 4
    if "dir" in args:
        extra += 2
    if "color" in args:
        extra += 4
    if "target_color" in args:
        extra += 4
    if "fill_color" in args:
        extra += 4
    if "steps" in args:
        extra += 2
    if "radius" in args:
        extra += 2
    if "sy" in args:
        extra += 2
    if "sx" in args:
        extra += 2
    if "mode" in args:
        extra += 2
    if "only_color" in args:
        extra += 4
    return int(base + int(extra))


def apply_op_v141(*, state: StateV132, op_id: str, args: Dict[str, Any]) -> StateV132:
    op = str(op_id)
    a = dict(args)
    if op == "repeat_grid":
        mode = str(a.get("mode") or "")
        if mode != "corners":
            return apply_op_v137(state=state, op_id=str(op_id), args=dict(args))
        sy = int(a.get("sy", 1))
        sx = int(a.get("sx", 1))
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        hi, wi = grid_shape_v124(state.grid)
        if hi <= 0 or wi <= 0:
            return state
        if sy <= 0 or sx <= 0:
            raise ValueError("repeat_grid_nonpositive_scale")
        out_h = int(hi * sy)
        out_w = int(wi * sx)
        out = [[int(bg) for _ in range(out_w)] for _ in range(out_h)]
        corners = [(0, 0), (0, int(sx - 1)), (int(sy - 1), 0), (int(sy - 1), int(sx - 1))]
        for r in range(int(hi)):
            for c in range(int(wi)):
                v = int(state.grid[r][c])
                br = int(r * sy)
                bc = int(c * sx)
                for dr, dc in corners:
                    rr = int(br + dr)
                    cc = int(bc + dc)
                    if 0 <= rr < out_h and 0 <= cc < out_w:
                        out[rr][cc] = int(v)
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)
    if op == "relational_expand":
        mode = str(a.get("mode") or "neq")
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state
        # Bound to keep search feasible; ARC tasks that need relational expansion are small.
        if int(h) > 10 or int(w) > 10:
            raise ValueError("relational_expand_too_large")
        out_h = int(h * h)
        out_w = int(w * w)
        out = [[int(bg) for _ in range(out_w)] for _ in range(out_h)]
        for r in range(h):
            for c in range(w):
                v = int(g[int(r)][int(c)])
                br = int(r * h)
                bc = int(c * w)
                for rr in range(h):
                    for cc in range(w):
                        vv = int(g[int(rr)][int(cc)])
                        cond = int(vv) == int(v) if mode == "eq" else int(vv) != int(v)
                        if cond:
                            out[int(br + rr)][int(bc + cc)] = int(v)
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)
    if op == "uniform_line_expand":
        prefer = str(a.get("prefer") or "row")
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state
        if int(h) > 10 or int(w) > 10:
            raise ValueError("uniform_line_expand_too_large")

        rows: List[int] = []
        for r in range(int(h)):
            v0 = int(g[int(r)][0])
            ok = True
            for c in range(int(w)):
                if int(g[int(r)][int(c)]) != int(v0):
                    ok = False
                    break
            if ok:
                rows.append(int(r))

        cols: List[int] = []
        for c in range(int(w)):
            v0 = int(g[0][int(c)])
            ok = True
            for r in range(int(h)):
                if int(g[int(r)][int(c)]) != int(v0):
                    ok = False
                    break
            if ok:
                cols.append(int(c))

        choose_row_first = str(prefer) != "col"
        chosen_row: Optional[int] = None
        chosen_col: Optional[int] = None
        if choose_row_first:
            if rows:
                chosen_row = int(rows[0])
            elif cols:
                chosen_col = int(cols[0])
        else:
            if cols:
                chosen_col = int(cols[0])
            elif rows:
                chosen_row = int(rows[0])

        if chosen_row is None and chosen_col is None:
            raise ValueError("uniform_line_expand_no_uniform_line")

        out_h = int(h * h)
        out_w = int(w * w)
        out = [[int(bg) for _ in range(out_w)] for _ in range(out_h)]
        if chosen_row is not None:
            dy = int(chosen_row * h)
            for r in range(int(h)):
                rr = int(dy + r)
                if rr < 0 or rr >= out_h:
                    continue
                for c in range(int(out_w)):
                    out[int(rr)][int(c)] = int(g[int(r)][int(c % w)])
        else:
            dx = int(chosen_col * w)
            for r in range(int(out_h)):
                for c in range(int(w)):
                    cc = int(dx + c)
                    if cc < 0 or cc >= out_w:
                        continue
                    out[int(r)][int(cc)] = int(g[int(r % h)][int(c)])
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)
    if op == "smear_nonbg":
        direction = str(a.get("dir") or "right")
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state
        out = [[int(g[int(r)][int(c)]) for c in range(int(w))] for r in range(int(h))]
        if direction == "right":
            for r in range(int(h)):
                last: Optional[int] = None
                for c in range(int(w)):
                    v = int(g[int(r)][int(c)])
                    if int(v) != int(bg):
                        last = int(v)
                    elif last is not None:
                        out[int(r)][int(c)] = int(last)
        elif direction == "left":
            for r in range(int(h)):
                last = None
                for c in range(int(w) - 1, -1, -1):
                    v = int(g[int(r)][int(c)])
                    if int(v) != int(bg):
                        last = int(v)
                    elif last is not None:
                        out[int(r)][int(c)] = int(last)
        elif direction == "down":
            for c in range(int(w)):
                last = None
                for r in range(int(h)):
                    v = int(g[int(r)][int(c)])
                    if int(v) != int(bg):
                        last = int(v)
                    elif last is not None:
                        out[int(r)][int(c)] = int(last)
        elif direction == "up":
            for c in range(int(w)):
                last = None
                for r in range(int(h) - 1, -1, -1):
                    v = int(g[int(r)][int(c)])
                    if int(v) != int(bg):
                        last = int(v)
                    elif last is not None:
                        out[int(r)][int(c)] = int(last)
        else:
            raise ValueError("smear_nonbg_bad_dir")
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

    if op == "crop_bbox_by_color":
        color = int(_check_color_v141(int(a.get("color", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state
        rmin = int(h)
        cmin = int(w)
        rmax = -1
        cmax = -1
        any_hit = False
        for r in range(h):
            for c in range(w):
                if int(g[int(r)][int(c)]) == int(color):
                    any_hit = True
                    rmin = min(int(rmin), int(r))
                    cmin = min(int(cmin), int(c))
                    rmax = max(int(rmax), int(r))
                    cmax = max(int(cmax), int(c))
        if not any_hit:
            raise ValueError("crop_bbox_by_color_empty")
        rr0 = int(rmin)
        cc0 = int(cmin)
        rr1 = int(rmax + 1)
        cc1 = int(cmax + 1)
        out = [[int(g[r][c]) for c in range(cc0, cc1)] for r in range(rr0, rr1)]
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

    if op == "crop_cc4_select":
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        key = str(a.get("key") or "area")
        order = str(a.get("order") or "max")
        rank = int(a.get("rank", 0))
        cf_raw = a.get("color_filter", None)
        cf = int(_check_color_v141(int(cf_raw))) if cf_raw is not None else None

        # Implement as a deterministic composition of existing object-centric ops.
        st = state
        st = apply_op_v141(state=st, op_id="cc4", args={"bg": int(bg)})
        st = apply_op_v141(
            state=st,
            op_id="select_obj",
            args={"key": str(key), "order": str(order), "rank": int(rank), "color_filter": (int(cf) if cf is not None else None)},
        )
        st = apply_op_v141(state=st, op_id="obj_bbox", args={})
        st = apply_op_v141(state=st, op_id="crop_bbox", args={})
        st = apply_op_v141(state=st, op_id="commit_patch", args={})
        return replace(st, objset=None, obj=None, bbox=None, patch=None)

    if op == "tile_signature_fill":
        tile_h = int(a.get("tile_h", 1))
        tile_w = int(a.get("tile_w", 1))
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        mapping_raw = a.get("mapping") if isinstance(a.get("mapping"), dict) else {}
        mapping: Dict[str, int] = {str(k): int(mapping_raw[k]) for k in sorted(mapping_raw.keys())}
        if tile_h <= 0 or tile_w <= 0:
            raise ValueError("tile_signature_fill_bad_tile")
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state
        if int(h) % int(tile_h) != 0 or int(w) % int(tile_w) != 0:
            raise ValueError("tile_signature_fill_shape_mismatch")

        from .act import canonical_json_dumps, sha256_hex

        def _tile_key(tile: List[List[int]]) -> str:
            # Canonicalize with an explicit background color: bg→0, other colors in sorted order.
            cols: Set[int] = set()
            for row in tile:
                for v in row:
                    vv = int(v)
                    if vv != int(bg):
                        cols.add(int(vv))
            order = sorted(int(x) for x in cols)
            ren: Dict[int, int] = {int(bg): 0}
            for i, c in enumerate(order):
                ren[int(c)] = int(i + 1)
            norm: List[List[int]] = []
            for row in tile:
                norm.append([int(ren.get(int(v), 0)) for v in row])
            body = {"kind": "tile_sig_v141", "tile": norm}
            return sha256_hex(canonical_json_dumps(body).encode("utf-8"))

        out = [[0 for _ in range(int(w))] for _ in range(int(h))]
        for r0 in range(0, int(h), int(tile_h)):
            for c0 in range(0, int(w), int(tile_w)):
                tile = [[int(g[int(r0 + rr)][int(c0 + cc)]) for cc in range(int(tile_w))] for rr in range(int(tile_h))]
                key = _tile_key(tile)
                if key not in mapping:
                    raise ValueError("tile_signature_fill_unknown_sig")
                col = int(_check_color_v141(int(mapping[key])))
                for rr in range(int(tile_h)):
                    for cc in range(int(tile_w)):
                        out[int(r0 + rr)][int(c0 + cc)] = int(col)
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)
    if op == "diagonal_fill":
        kind = str(a.get("kind") or "main")
        k = int(a.get("k", 0))
        from_color = int(_check_color_v141(int(a.get("from_color", 0))))
        to_color = int(_check_color_v141(int(a.get("to_color", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state
        out = [[int(g[int(r)][int(c)]) for c in range(int(w))] for r in range(int(h))]
        if kind == "main":
            # r - c == k
            for r in range(int(h)):
                c = int(r - k)
                if 0 <= c < int(w) and int(out[int(r)][int(c)]) == int(from_color):
                    out[int(r)][int(c)] = int(to_color)
        elif kind == "anti":
            # r + c == k
            for r in range(int(h)):
                c = int(k - r)
                if 0 <= c < int(w) and int(out[int(r)][int(c)]) == int(from_color):
                    out[int(r)][int(c)] = int(to_color)
        else:
            raise ValueError("diagonal_fill_bad_kind")
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)
    if op == "best_diagonal_fill":
        from_color = int(_check_color_v141(int(a.get("from_color", 0))))
        to_color = int(_check_color_v141(int(a.get("to_color", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state

        best_cells: Optional[List[Tuple[int, int]]] = None
        best_score = -1
        best_len = -1

        # Prefer the first diagonal in deterministic iteration order when scores tie:
        # main diagonals first (offset ascending), then anti diagonals (sum ascending).
        for kind in ("main", "anti"):
            if kind == "main":
                for k in range(-int(w - 1), int(h)):
                    cells: List[Tuple[int, int]] = []
                    score = 0
                    for r in range(int(h)):
                        c = int(r - k)
                        if 0 <= c < int(w):
                            cells.append((int(r), int(c)))
                            if int(g[int(r)][int(c)]) == int(from_color):
                                score += 1
                    if score > best_score or (score == best_score and int(len(cells)) > best_len):
                        best_score = int(score)
                        best_len = int(len(cells))
                        best_cells = cells
            else:
                for k in range(0, int(h + w - 1)):
                    cells = []
                    score = 0
                    for r in range(int(h)):
                        c = int(k - r)
                        if 0 <= c < int(w):
                            cells.append((int(r), int(c)))
                            if int(g[int(r)][int(c)]) == int(from_color):
                                score += 1
                    if score > best_score or (score == best_score and int(len(cells)) > best_len):
                        best_score = int(score)
                        best_len = int(len(cells))
                        best_cells = cells

        if not best_cells:
            return state

        out = [[int(g[int(r)][int(c)]) for c in range(int(w))] for r in range(int(h))]
        for r, c in best_cells:
            if int(out[int(r)][int(c)]) == int(from_color):
                out[int(r)][int(c)] = int(to_color)
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)
    if op == "nest_by_color_area":
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state

        # Max connected-component area per color (4-neigh), excluding background.
        # Also track the largest component bbox max-dimension; this defines the output square size.
        best_area_by_color: Dict[int, int] = {}
        seen: Set[Tuple[int, int]] = set()
        max_bbox_dim = 0
        for r0 in range(int(h)):
            for c0 in range(int(w)):
                rc = (int(r0), int(c0))
                if rc in seen:
                    continue
                col = int(g[int(r0)][int(c0)])
                if int(col) == int(bg):
                    continue
                stack: List[Tuple[int, int]] = [rc]
                seen.add(rc)
                area = 0
                rmin = int(r0)
                rmax = int(r0)
                cmin = int(c0)
                cmax = int(c0)
                while stack:
                    rr, cc = stack.pop()
                    area += 1
                    rmin = min(int(rmin), int(rr))
                    rmax = max(int(rmax), int(rr))
                    cmin = min(int(cmin), int(cc))
                    cmax = max(int(cmax), int(cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr = int(rr + dr)
                        nc = int(cc + dc)
                        if nr < 0 or nc < 0 or nr >= int(h) or nc >= int(w):
                            continue
                        nrc = (int(nr), int(nc))
                        if nrc in seen:
                            continue
                        if int(g[int(nr)][int(nc)]) != int(col):
                            continue
                        seen.add(nrc)
                        stack.append(nrc)
                max_bbox_dim = max(int(max_bbox_dim), int(max(int(rmax - rmin + 1), int(cmax - cmin + 1))))
                prev = int(best_area_by_color.get(int(col), 0))
                if int(area) > int(prev):
                    best_area_by_color[int(col)] = int(area)

        if not best_area_by_color or int(max_bbox_dim) <= 0:
            return state

        # Order colors by descending max-area, then by color id to make deterministic.
        ordered = sorted(best_area_by_color.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))
        colors = [int(c) for c, _a in ordered]
        size = int(max_bbox_dim)
        max_layers = int((int(size) + 1) // 2)
        colors = colors[: int(max_layers)]
        n = int(len(colors))
        if n <= 0:
            return state

        out = [[int(bg) for _ in range(int(size))] for _ in range(int(size))]
        for i, col in enumerate(colors):
            top = int(i)
            left = int(i)
            bottom = int(size - 1 - i)
            right = int(size - 1 - i)
            if i == int(n - 1):
                for rr in range(int(top), int(bottom) + 1):
                    for cc in range(int(left), int(right) + 1):
                        out[int(rr)][int(cc)] = int(col)
            else:
                for cc in range(int(left), int(right) + 1):
                    out[int(top)][int(cc)] = int(col)
                    out[int(bottom)][int(cc)] = int(col)
                for rr in range(int(top), int(bottom) + 1):
                    out[int(rr)][int(left)] = int(col)
                    out[int(rr)][int(right)] = int(col)

        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)
    if op == "quadrant_center_tile":
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state

        if "bg" in a:
            bg = int(_check_color_v141(int(a.get("bg", 0))))
        else:
            # Deterministic background heuristic: mode of the input grid.
            counts: Dict[int, int] = {}
            for r in range(int(h)):
                for c in range(int(w)):
                    v = int(g[int(r)][int(c)])
                    counts[int(v)] = int(counts.get(int(v), 0)) + 1
            bg = int(sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[0][0])

        center_color = int(_check_color_v141(int(a.get("center_color", 0))))

        # BBox of all non-bg cells.
        cells: List[Tuple[int, int, int]] = []
        for r in range(int(h)):
            for c in range(int(w)):
                v = int(g[int(r)][int(c)])
                if int(v) != int(bg):
                    cells.append((int(r), int(c), int(v)))
        if not cells:
            return state
        rmin = min(int(r) for r, _c, _v in cells)
        rmax = max(int(r) for r, _c, _v in cells)
        cmin = min(int(c) for _r, c, _v in cells)
        cmax = max(int(c) for _r, c, _v in cells)
        cr = int((int(rmin) + int(rmax)) // 2)
        cc = int((int(cmin) + int(cmax)) // 2)
        if not (0 <= int(cr) < int(h) and 0 <= int(cc) < int(w)):
            raise ValueError("quadrant_center_tile_center_oob")

        # One color per quadrant relative to center; ignore cells lying on center row/col.
        quad: Dict[str, int] = {}
        for r, c, v in cells:
            if int(r) == int(cr) or int(c) == int(cc):
                continue
            key = ("U" if int(r) < int(cr) else "D") + ("L" if int(c) < int(cc) else "R")
            prev = quad.get(str(key))
            if prev is not None and int(prev) != int(v):
                raise ValueError("quadrant_center_tile_ambiguous_quadrant")
            quad[str(key)] = int(v)

        out = [[int(g[int(r)][int(c)]) for c in range(int(w))] for r in range(int(h))]
        out[int(cr)][int(cc)] = int(center_color)

        for key, v in quad.items():
            dr = -1 if str(key)[0] == "U" else 1
            dc = -1 if str(key)[1] == "L" else 1
            rr = int(cr + dr)
            cc2 = int(cc + dc)
            if not (0 <= int(rr) < int(h) and 0 <= int(cc2) < int(w)):
                raise ValueError("quadrant_center_tile_target_oob")
            if int(out[int(rr)][int(cc2)]) != int(bg) and int(out[int(rr)][int(cc2)]) != int(v):
                raise ValueError("quadrant_center_tile_target_conflict")
            out[int(rr)][int(cc2)] = int(v)

        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)
    if op == "histogram_color_counts":
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        x0 = int(a.get("x0", 0))
        if int(x0) < 0:
            raise ValueError("histogram_color_counts_bad_x0")
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state
        if int(x0) >= int(w):
            raise ValueError("histogram_color_counts_x0_oob")

        counts: Dict[int, int] = {}
        for r in range(int(h)):
            for c in range(int(w)):
                v = int(g[int(r)][int(c)])
                if int(v) == int(bg):
                    continue
                counts[int(v)] = int(counts.get(int(v), 0)) + 1

        colors = sorted([int(c) for c in counts.keys()])
        out = [[int(bg) for _ in range(int(w))] for _ in range(int(h))]
        for i, col in enumerate(colors):
            cc = int(x0 + i)
            if int(cc) >= int(w):
                break
            n = int(counts.get(int(col), 0))
            if int(n) <= 0:
                continue
            if int(n) > int(h):
                raise ValueError("histogram_color_counts_too_tall")
            r0 = int(h - n)
            for rr in range(int(r0), int(h)):
                out[int(rr)][int(cc)] = int(col)
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)
    if op == "color_counts_run_column":
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return replace(state, objset=None, obj=None, bbox=None, patch=None)

        # Default background is the mode color of the entire grid (deterministic).
        bg_arg = a.get("bg")
        bg: int
        if bg_arg is None:
            counts0: Dict[int, int] = {}
            for r in range(int(h)):
                for c in range(int(w)):
                    v = int(g[int(r)][int(c)])
                    counts0[int(v)] = int(counts0.get(int(v), 0)) + 1
            items0 = sorted(((int(col), int(n)) for col, n in counts0.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
            bg = int(items0[0][0]) if items0 else 0
        else:
            bg = int(_check_color_v141(int(bg_arg)))

        # Colors ordered by first appearance (row-major), excluding bg.
        order: List[int] = []
        seen: Set[int] = set()
        for r in range(int(h)):
            for c in range(int(w)):
                v = int(g[int(r)][int(c)])
                if int(v) == int(bg):
                    continue
                if int(v) in seen:
                    continue
                seen.add(int(v))
                order.append(int(v))

        if not order:
            out = [[int(bg)]]
            return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

        counts: Dict[int, int] = {int(v): 0 for v in order}
        for r in range(int(h)):
            for c in range(int(w)):
                v = int(g[int(r)][int(c)])
                if int(v) == int(bg):
                    continue
                if int(v) in counts:
                    counts[int(v)] = int(counts.get(int(v), 0)) + 1

        col: List[int] = []
        for v in order:
            n = int(counts.get(int(v), 0))
            if n <= 0:
                continue
            col.extend([int(v)] * int(n))

        if not col:
            out = [[int(bg)]]
            return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

        out = [[int(v)] for v in col]
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)
    if op == "propagate_nonbg_translate":
        dx = int(a.get("dx", 0))
        dy = int(a.get("dy", 0))
        pad = int(_check_color_v141(int(a.get("pad", 0))))
        if int(dx) == 0 and int(dy) == 0:
            return replace(state, objset=None, obj=None, bbox=None, patch=None)
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return replace(state, objset=None, obj=None, bbox=None, patch=None)
        cur = [[int(g[int(r)][int(c)]) for c in range(int(w))] for r in range(int(h))]
        # Conservative deterministic bound.
        max_iters = int(h + w + 1)
        for _ in range(max_iters):
            changed = False
            nxt = [row[:] for row in cur]
            for r in range(int(h)):
                src_r = int(r - dy)
                if src_r < 0 or src_r >= int(h):
                    continue
                for c in range(int(w)):
                    if int(cur[int(r)][int(c)]) != int(pad):
                        continue
                    src_c = int(c - dx)
                    if src_c < 0 or src_c >= int(w):
                        continue
                    v = int(cur[int(src_r)][int(src_c)])
                    if int(v) == int(pad):
                        continue
                    nxt[int(r)][int(c)] = int(v)
                    changed = True
            if not changed:
                break
            cur = nxt
        return replace(state, grid=grid_from_list_v124(cur), objset=None, obj=None, bbox=None, patch=None)
    if op in OP_DEFS_V137:
        return apply_op_v137(state=state, op_id=str(op_id), args=dict(args))

    if op == "cc4_nonbg_multicolor":
        bg = a.get("bg")
        bgc = int(_check_color_v141(int(bg))) if bg is not None else None
        oset = connected_components4_nonbg_multicolor_v141(state.grid, bg=bgc)
        return replace(state, objset=oset, obj=None, bbox=None, patch=None)

    if op == "fill_enclosed_region":
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        fill = int(_check_color_v141(int(a.get("fill", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state
        # Mark background connected to the border (4-neigh).
        seen: Set[Tuple[int, int]] = set()
        q: List[Tuple[int, int]] = []
        for r in range(h):
            for c in (0, int(w - 1)):
                if int(g[int(r)][int(c)]) == int(bg) and (int(r), int(c)) not in seen:
                    seen.add((int(r), int(c)))
                    q.append((int(r), int(c)))
        for c in range(w):
            for r in (0, int(h - 1)):
                if int(g[int(r)][int(c)]) == int(bg) and (int(r), int(c)) not in seen:
                    seen.add((int(r), int(c)))
                    q.append((int(r), int(c)))
        while q:
            rr, cc = q.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(rr + dr)
                nc = int(cc + dc)
                if nr < 0 or nc < 0 or nr >= int(h) or nc >= int(w):
                    continue
                if (nr, nc) in seen:
                    continue
                if int(g[nr][nc]) != int(bg):
                    continue
                seen.add((nr, nc))
                q.append((nr, nc))
        out = [[int(g[r][c]) for c in range(w)] for r in range(h)]
        for r in range(h):
            for c in range(w):
                if int(out[r][c]) == int(bg) and (int(r), int(c)) not in seen:
                    out[r][c] = int(fill)
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

    if op == "cc4_color_bars":
        bg_raw = a.get("bg", None)
        if bg_raw is None:
            # Deterministic background inference: mode color (tie-break min id).
            counts: Dict[int, int] = {}
            g0 = state.grid
            h0, w0 = grid_shape_v124(g0)
            for r in range(h0):
                for c in range(w0):
                    vv = int(g0[int(r)][int(c)])
                    counts[vv] = int(counts.get(vv, 0)) + 1
            items = sorted(((int(col), int(n)) for col, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
            bg = int(items[0][0]) if items else 0
        else:
            bg = int(_check_color_v141(int(bg_raw)))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return replace(state, objset=None, obj=None, bbox=None, patch=None)
        colors: Set[int] = set()
        for r in range(h):
            for c in range(w):
                vv = int(g[int(r)][int(c)])
                if vv != int(bg):
                    colors.add(int(vv))
        cols = sorted(int(c) for c in colors)
        if not cols:
            return replace(state, grid=grid_from_list_v124([[int(bg)]]), objset=None, obj=None, bbox=None, patch=None)

        def count_cc4_for_color(col: int) -> int:
            visited: Set[Tuple[int, int]] = set()
            n = 0
            for rr in range(h):
                for cc in range(w):
                    if int(g[int(rr)][int(cc)]) != int(col):
                        continue
                    key = (int(rr), int(cc))
                    if key in visited:
                        continue
                    n += 1
                    q: List[Tuple[int, int]] = [key]
                    visited.add(key)
                    while q:
                        r0, c0 = q.pop()
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            r1 = int(r0 + dr)
                            c1 = int(c0 + dc)
                            if r1 < 0 or c1 < 0 or r1 >= int(h) or c1 >= int(w):
                                continue
                            if int(g[int(r1)][int(c1)]) != int(col):
                                continue
                            k2 = (int(r1), int(c1))
                            if k2 in visited:
                                continue
                            visited.add(k2)
                            q.append(k2)
            return int(n)

        counts: List[Tuple[int, int]] = [(int(c), int(count_cc4_for_color(int(c)))) for c in cols]
        counts.sort(key=lambda cn: (-int(cn[1]), int(cn[0])))
        maxn = max(int(n) for _c, n in counts) if counts else 0
        if maxn <= 0:
            return replace(state, grid=grid_from_list_v124([[int(bg)]]), objset=None, obj=None, bbox=None, patch=None)
        out: List[List[int]] = []
        for c, n in counts:
            row = [int(bg) for _ in range(int(maxn))]
            for i in range(int(n)):
                row[int(maxn - 1 - i)] = int(c)
            out.append(row)
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

    if op == "cc4_color_area_column":
        bg_raw = a.get("bg", None)
        if bg_raw is None:
            counts: Dict[int, int] = {}
            g0 = state.grid
            h0, w0 = grid_shape_v124(g0)
            for r in range(h0):
                for c in range(w0):
                    vv = int(g0[int(r)][int(c)])
                    counts[vv] = int(counts.get(vv, 0)) + 1
            items = sorted(((int(col), int(n)) for col, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
            bg = int(items[0][0]) if items else 0
        else:
            bg = int(_check_color_v141(int(bg_raw)))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return replace(state, objset=None, obj=None, bbox=None, patch=None)

        # This operator is intended for *disjoint* monochrome objects. If different non-bg colors
        # touch (forming multicolor components), it is structurally ambiguous; fail-closed.
        for rr in range(h):
            for cc in range(w):
                v0 = int(g[int(rr)][int(cc)])
                if v0 == int(bg):
                    continue
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    r2 = int(rr + dr)
                    c2 = int(cc + dc)
                    if r2 < 0 or c2 < 0 or r2 >= int(h) or c2 >= int(w):
                        continue
                    v1 = int(g[int(r2)][int(c2)])
                    if v1 == int(bg):
                        continue
                    if v1 != int(v0):
                        raise ValueError("multicolor_adjacent_nonbg")

        visited: Set[Tuple[int, int]] = set()
        objs: List[Tuple[int, int, int, int]] = []
        # Object record: (r0, c0, color, area) with bbox top-left sorting.
        for rr in range(h):
            for cc in range(w):
                col = int(g[int(rr)][int(cc)])
                if col == int(bg):
                    continue
                key = (int(rr), int(cc))
                if key in visited:
                    continue
                q: List[Tuple[int, int]] = [key]
                visited.add(key)
                area = 0
                r0 = int(rr)
                c0 = int(cc)
                while q:
                    r1, c1 = q.pop()
                    area += 1
                    r0 = min(int(r0), int(r1))
                    c0 = min(int(c0), int(c1))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        r2 = int(r1 + dr)
                        c2 = int(c1 + dc)
                        if r2 < 0 or c2 < 0 or r2 >= int(h) or c2 >= int(w):
                            continue
                        if int(g[int(r2)][int(c2)]) != int(col):
                            continue
                        k2 = (int(r2), int(c2))
                        if k2 in visited:
                            continue
                        visited.add(k2)
                        q.append(k2)
                objs.append((int(r0), int(c0), int(col), int(area)))

        if not objs:
            return replace(state, grid=grid_from_list_v124([[int(bg)]]), objset=None, obj=None, bbox=None, patch=None)

        objs.sort(key=lambda o: (int(o[0]), int(o[1]), int(o[2]), -int(o[3])))
        total = sum(int(area) for _r0, _c0, _col, area in objs)
        if total <= 0:
            return replace(state, grid=grid_from_list_v124([[int(bg)]]), objset=None, obj=None, bbox=None, patch=None)
        out: List[List[int]] = [[int(bg)] for _ in range(int(total))]
        i = 0
        for _r0, _c0, col, area in objs:
            for _ in range(int(area)):
                out[int(i)][0] = int(col)
                i += 1
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

    if op == "cc4_nonbg_bfs_column":
        bg_raw = a.get("bg", None)
        if bg_raw is None:
            counts: Dict[int, int] = {}
            g0 = state.grid
            h0, w0 = grid_shape_v124(g0)
            for r in range(h0):
                for c in range(w0):
                    vv = int(g0[int(r)][int(c)])
                    counts[vv] = int(counts.get(vv, 0)) + 1
            items = sorted(((int(col), int(n)) for col, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
            bg = int(items[0][0]) if items else 0
        else:
            bg = int(_check_color_v141(int(bg_raw)))

        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return replace(state, objset=None, obj=None, bbox=None, patch=None)

        visited: Set[Tuple[int, int]] = set()
        seq: List[int] = []

        # BFS over predicate (cell != bg), keeping the component discovery order deterministic:
        # starts are selected by row-major scan; within a BFS, neighbor order is (up, down, left, right).
        for rr in range(h):
            for cc in range(w):
                if int(g[int(rr)][int(cc)]) == int(bg):
                    continue
                start = (int(rr), int(cc))
                if start in visited:
                    continue
                q: List[Tuple[int, int]] = [start]
                visited.add(start)
                qi = 0
                while qi < len(q):
                    r0, c0 = q[qi]
                    qi += 1
                    seq.append(int(g[int(r0)][int(c0)]))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        r1 = int(r0 + dr)
                        c1 = int(c0 + dc)
                        if r1 < 0 or c1 < 0 or r1 >= int(h) or c1 >= int(w):
                            continue
                        if int(g[int(r1)][int(c1)]) == int(bg):
                            continue
                        key = (int(r1), int(c1))
                        if key in visited:
                            continue
                        visited.add(key)
                        q.append(key)

        if not seq:
            return replace(state, grid=grid_from_list_v124([[int(bg)]]), objset=None, obj=None, bbox=None, patch=None)
        out = [[int(v)] for v in seq]
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

    if op == "obj_patch":
        if state.obj is None:
            raise ValueError("missing_obj")
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        bbox = state.obj.bbox
        r0, c0, r1, c1 = (int(x) for x in bbox.to_tuple())
        hh = int(r1 - r0)
        ww = int(c1 - c0)
        if hh <= 0 or ww <= 0:
            raise ValueError("obj_patch_empty_bbox")
        keep = set((int(r), int(c)) for r, c in state.obj.cells)
        out = [[int(bg) for _ in range(ww)] for _ in range(hh)]
        for r, c in keep:
            rr = int(r - r0)
            cc = int(c - c0)
            if 0 <= rr < hh and 0 <= cc < ww:
                out[rr][cc] = int(state.grid[int(r)][int(c)])
        return replace(state, patch=grid_from_list_v124(out))

    if op == "transpose":
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state
        out = [[int(g[r][c]) for r in range(h)] for c in range(w)]
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

    if op == "gravity":
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        direction = str(a.get("dir") or "down")
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state
        out = [[int(bg) for _ in range(w)] for _ in range(h)]
        if direction in {"down", "up"}:
            for c in range(w):
                col = [int(g[r][c]) for r in range(h) if int(g[r][c]) != int(bg)]
                if direction == "down":
                    start = int(h - len(col))
                    for i, v in enumerate(col):
                        out[int(start + i)][int(c)] = int(v)
                else:
                    for i, v in enumerate(col):
                        out[int(i)][int(c)] = int(v)
        elif direction in {"right", "left"}:
            for r in range(h):
                row = [int(g[r][c]) for c in range(w) if int(g[r][c]) != int(bg)]
                if direction == "right":
                    start = int(w - len(row))
                    for i, v in enumerate(row):
                        out[int(r)][int(start + i)] = int(v)
                else:
                    for i, v in enumerate(row):
                        out[int(r)][int(i)] = int(v)
        else:
            raise ValueError("gravity_bad_dir")
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

    if op == "mask_by_color":
        color = int(_check_color_v141(int(a.get("color", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return replace(state, patch=grid_from_list_v124([]))
        out = [[1 if int(g[r][c]) == int(color) else 0 for c in range(w)] for r in range(h)]
        return replace(state, patch=grid_from_list_v124(out))

    if op == "mask_bg":
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return replace(state, patch=grid_from_list_v124([]))
        out = [[1 if int(g[r][c]) == int(bg) else 0 for c in range(w)] for r in range(h)]
        return replace(state, patch=grid_from_list_v124(out))

    if op == "mask_border":
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return replace(state, patch=grid_from_list_v124([]))
        out = [[0 for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if r == 0 or c == 0 or r == int(h - 1) or c == int(w - 1):
                    out[int(r)][int(c)] = 1
        return replace(state, patch=grid_from_list_v124(out))

    if op == "mask_nonbg":
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return replace(state, patch=grid_from_list_v124([]))
        out = [[1 if int(g[r][c]) != int(bg) else 0 for c in range(w)] for r in range(h)]
        return replace(state, patch=grid_from_list_v124(out))

    if op == "mask_cross_center":
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return replace(state, patch=grid_from_list_v124([]))
        row_counts = [0 for _ in range(h)]
        col_counts = [0 for _ in range(w)]
        for r in range(h):
            for c in range(w):
                if int(g[r][c]) != int(bg):
                    row_counts[r] += 1
                    col_counts[c] += 1
        max_r = max(row_counts) if row_counts else 0
        max_c = max(col_counts) if col_counts else 0
        if max_r <= 0 or max_c <= 0:
            raise ValueError("mask_cross_center_empty")
        rows = [int(r) for r, n in enumerate(row_counts) if int(n) == int(max_r)]
        cols = [int(c) for c, n in enumerate(col_counts) if int(n) == int(max_c)]
        out = [[0 for _ in range(w)] for _ in range(h)]
        for r in rows:
            for c in cols:
                out[int(r)][int(c)] = 1
        return replace(state, patch=grid_from_list_v124(out))

    if op == "mask_outline":
        if state.patch is None:
            raise ValueError("missing_patch")
        m = state.patch
        h, w = grid_shape_v124(m)
        if h <= 0 or w <= 0:
            return state
        out = [[0 for _ in range(w)] for _ in range(h)]
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for r in range(h):
            for c in range(w):
                if int(m[r][c]) == 0:
                    continue
                is_border = False
                for dr, dc in neigh:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if rr < 0 or rr >= h or cc < 0 or cc >= w:
                        is_border = True
                        break
                    if int(m[rr][cc]) == 0:
                        is_border = True
                        break
                if is_border:
                    out[r][c] = 1
        return replace(state, patch=grid_from_list_v124(out))

    if op == "mask_not":
        if state.patch is None:
            raise ValueError("missing_patch")
        m = state.patch
        h, w = grid_shape_v124(m)
        if h <= 0 or w <= 0:
            return state
        out = [[0 for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                out[int(r)][int(c)] = 0 if int(m[int(r)][int(c)]) != 0 else 1
        return replace(state, patch=grid_from_list_v124(out))

    if op == "mask_and_color":
        if state.patch is None:
            raise ValueError("missing_patch")
        color = int(_check_color_v141(int(a.get("color", 0))))
        g = state.grid
        m = state.patch
        h, w = grid_shape_v124(g)
        mh, mw = grid_shape_v124(m)
        if (h, w) != (mh, mw):
            raise ValueError("mask_and_color_shape_mismatch")
        out = [[0 for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if int(m[int(r)][int(c)]) == 0:
                    continue
                if int(g[int(r)][int(c)]) == int(color):
                    out[int(r)][int(c)] = 1
        return replace(state, patch=grid_from_list_v124(out))

    if op == "mask_and_nonbg":
        if state.patch is None:
            raise ValueError("missing_patch")
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        g = state.grid
        m = state.patch
        h, w = grid_shape_v124(g)
        mh, mw = grid_shape_v124(m)
        if (h, w) != (mh, mw):
            raise ValueError("mask_and_nonbg_shape_mismatch")
        out = [[0 for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if int(m[int(r)][int(c)]) == 0:
                    continue
                if int(g[int(r)][int(c)]) != int(bg):
                    out[int(r)][int(c)] = 1
        return replace(state, patch=grid_from_list_v124(out))

    if op == "mask_dilate":
        if state.patch is None:
            raise ValueError("missing_patch")
        steps = int(a.get("steps", 1))
        if steps < 0:
            raise ValueError("mask_dilate_steps_negative")
        cur = state.patch
        h, w = grid_shape_v124(cur)
        if h <= 0 or w <= 0:
            return state
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for _ in range(int(steps)):
            out = [[1 if int(cur[r][c]) != 0 else 0 for c in range(w)] for r in range(h)]
            for r in range(h):
                for c in range(w):
                    if int(cur[r][c]) != 0:
                        continue
                    for dr, dc in neigh:
                        rr = int(r + dr)
                        cc = int(c + dc)
                        if rr < 0 or rr >= h or cc < 0 or cc >= w:
                            continue
                        if int(cur[rr][cc]) != 0:
                            out[r][c] = 1
                            break
            cur = grid_from_list_v124(out)
        return replace(state, patch=cur)

    if op == "mask_box_dilate":
        if state.patch is None:
            raise ValueError("missing_patch")
        radius = int(a.get("radius", 1))
        if radius < 0:
            raise ValueError("mask_box_dilate_radius_negative")
        m = state.patch
        h, w = grid_shape_v124(m)
        if h <= 0 or w <= 0:
            return state
        r = int(radius)
        out = [[0 for _ in range(w)] for _ in range(h)]
        for rr in range(h):
            for cc in range(w):
                if int(m[rr][cc]) == 0:
                    continue
                for dr in range(-r, r + 1):
                    for dc in range(-r, r + 1):
                        nr = int(rr + dr)
                        nc = int(cc + dc)
                        if 0 <= nr < h and 0 <= nc < w:
                            out[nr][nc] = 1
        return replace(state, patch=grid_from_list_v124(out))

    if op == "paint_mask":
        if state.patch is None:
            raise ValueError("missing_patch")
        color = int(_check_color_v141(int(a.get("color", 0))))
        mode = str(a.get("mode") or "overwrite")
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        only_color_raw = a.get("only_color")
        only_color = int(_check_color_v141(int(only_color_raw))) if only_color_raw is not None else None
        g = state.grid
        m = state.patch
        h, w = grid_shape_v124(g)
        mh, mw = grid_shape_v124(m)
        if (h, w) != (mh, mw):
            raise ValueError("paint_mask_shape_mismatch")
        out = [[int(g[r][c]) for c in range(w)] for r in range(h)]
        for r in range(h):
            for c in range(w):
                if int(m[r][c]) != 0:
                    if mode == "overwrite":
                        out[r][c] = int(color)
                    elif mode == "only_bg":
                        if int(out[r][c]) == int(bg):
                            out[r][c] = int(color)
                    elif mode == "only_color":
                        if only_color is None:
                            raise ValueError("paint_mask_only_color_missing")
                        if int(out[r][c]) == int(only_color):
                            out[r][c] = int(color)
                    else:
                        raise ValueError(f"unknown_paint_mode:{mode}")
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

    if op == "paint_points":
        pts_raw = a.get("points")
        mode = str(a.get("mode") or "overwrite")
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if int(h) <= 0 or int(w) <= 0:
            return replace(state, objset=None, obj=None, bbox=None, patch=None)
        if not isinstance(pts_raw, list) or not pts_raw:
            raise ValueError("paint_points_missing_points")

        # Bound to keep search and direct inference feasible.
        if int(len(pts_raw)) > 128:
            raise ValueError("paint_points_too_many")

        pts: List[Tuple[int, int, int]] = []
        for row in pts_raw:
            if isinstance(row, dict):
                rr = int(row.get("r", 0))
                cc = int(row.get("c", 0))
                col = int(_check_color_v141(int(row.get("color", 0))))
                pts.append((int(rr), int(cc), int(col)))
            elif isinstance(row, (list, tuple)) and len(row) == 3:
                rr, cc, col0 = row
                col = int(_check_color_v141(int(col0)))
                pts.append((int(rr), int(cc), int(col)))
            else:
                raise ValueError("paint_points_bad_point")

        # Deterministic: require unique (r,c).
        seen_rc: Set[Tuple[int, int]] = set()
        for rr, cc, _col in pts:
            key = (int(rr), int(cc))
            if key in seen_rc:
                raise ValueError("paint_points_duplicate_rc")
            seen_rc.add(key)
            if rr < 0 or cc < 0 or rr >= int(h) or cc >= int(w):
                raise ValueError("paint_points_oob")

        out = [[int(g[int(r)][int(c)]) for c in range(int(w))] for r in range(int(h))]
        # Apply in canonical order for reproducibility.
        pts.sort(key=lambda t: (int(t[0]), int(t[1]), int(t[2])))
        for rr, cc, col in pts:
            if mode == "overwrite":
                out[int(rr)][int(cc)] = int(col)
            elif mode == "only_bg":
                if int(out[int(rr)][int(cc)]) == int(bg):
                    out[int(rr)][int(cc)] = int(col)
            else:
                raise ValueError(f"unknown_paint_points_mode:{mode}")
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

    if op == "flood_fill":
        if state.patch is None:
            raise ValueError("missing_patch")
        target_color = int(_check_color_v141(int(a.get("target_color", 0))))
        fill_color = int(_check_color_v141(int(a.get("fill_color", 0))))
        g = state.grid
        m = state.patch
        h, w = grid_shape_v124(g)
        mh, mw = grid_shape_v124(m)
        if (h, w) != (mh, mw):
            raise ValueError("flood_fill_shape_mismatch")
        out = [[int(g[r][c]) for c in range(w)] for r in range(h)]
        q: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        for r in range(h):
            for c in range(w):
                if int(m[r][c]) == 0:
                    continue
                if int(out[r][c]) != int(target_color):
                    continue
                if (int(r), int(c)) in seen:
                    continue
                seen.add((int(r), int(c)))
                q.append((int(r), int(c)))
        while q:
            rr, cc = q.pop()
            out[int(rr)][int(cc)] = int(fill_color)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(rr + dr)
                nc = int(cc + dc)
                if nr < 0 or nc < 0 or nr >= h or nc >= w:
                    continue
                if (nr, nc) in seen:
                    continue
                if int(out[nr][nc]) != int(target_color):
                    continue
                seen.add((nr, nc))
                q.append((nr, nc))
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

    if op == "symmetry_fill_h":
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state
        out = [[int(g[r][c]) for c in range(w)] for r in range(h)]
        for r in range(h):
            for c in range(w):
                if int(out[r][c]) != int(bg):
                    continue
                mc = int(w - 1 - c)
                out[r][c] = int(g[r][mc])
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

    if op == "symmetry_fill_v":
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state
        out = [[int(g[r][c]) for c in range(w)] for r in range(h)]
        for r in range(h):
            mr = int(h - 1 - r)
            for c in range(w):
                if int(out[r][c]) != int(bg):
                    continue
                out[r][c] = int(g[mr][c])
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

    if op == "symmetry_fill_rot180":
        bg = int(_check_color_v141(int(a.get("bg", 0))))
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state
        out = [[int(g[int(r)][int(c)]) for c in range(int(w))] for r in range(int(h))]
        for r in range(int(h)):
            rr = int(h - 1 - r)
            for c in range(int(w)):
                if int(out[int(r)][int(c)]) != int(bg):
                    continue
                cc = int(w - 1 - c)
                v = int(g[int(rr)][int(cc)])
                if int(v) != int(bg):
                    out[int(r)][int(c)] = int(v)
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

    if op == "downsample_mode":
        sy = int(a.get("sy", 1))
        sx = int(a.get("sx", 1))
        if sy <= 0 or sx <= 0:
            raise ValueError("downsample_factor_nonpositive")
        g = state.grid
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            return state
        if int(h) % int(sy) != 0 or int(w) % int(sx) != 0:
            raise ValueError("downsample_shape_not_divisible")
        oh = int(h // sy)
        ow = int(w // sx)
        out: List[List[int]] = [[0 for _ in range(ow)] for _ in range(oh)]
        for rr in range(oh):
            for cc in range(ow):
                counts: Dict[int, int] = {}
                for dr in range(sy):
                    for dc in range(sx):
                        v = int(g[int(rr * sy + dr)][int(cc * sx + dc)])
                        counts[v] = int(counts.get(v, 0)) + 1
                items = sorted(((int(col), int(n)) for col, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
                out[int(rr)][int(cc)] = int(items[0][0]) if items else 0
        return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)

    if op == "patch_rotate90":
        if state.patch is None:
            raise ValueError("missing_patch")
        return replace(state, patch=rotate90_v124(state.patch))
    if op == "patch_rotate180":
        if state.patch is None:
            raise ValueError("missing_patch")
        return replace(state, patch=rotate180_v124(state.patch))
    if op == "patch_rotate270":
        if state.patch is None:
            raise ValueError("missing_patch")
        return replace(state, patch=rotate270_v124(state.patch))
    if op == "patch_reflect_h":
        if state.patch is None:
            raise ValueError("missing_patch")
        return replace(state, patch=reflect_h_v124(state.patch))
    if op == "patch_reflect_v":
        if state.patch is None:
            raise ValueError("missing_patch")
        return replace(state, patch=reflect_v_v124(state.patch))
    if op == "patch_transpose":
        if state.patch is None:
            raise ValueError("missing_patch")
        p = state.patch
        h, w = grid_shape_v124(p)
        if h <= 0 or w <= 0:
            return state
        out = [[int(p[r][c]) for r in range(h)] for c in range(w)]
        return replace(state, patch=grid_from_list_v124(out))
    if op == "patch_translate":
        if state.patch is None:
            raise ValueError("missing_patch")
        dx = int(a.get("dx", 0))
        dy = int(a.get("dy", 0))
        pad = int(_check_color_v141(int(a.get("pad", 0))))
        return replace(state, patch=translate_v124(state.patch, dx=int(dx), dy=int(dy), pad=int(pad)))

    raise ValueError("unknown_op_v141")
