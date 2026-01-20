from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple

from .arc_objects_v132 import BBoxV132, ObjectSetV132, ObjectV132
from .arc_ops_v132 import OpDefV132, StateV132
from .arc_ops_v137 import OP_DEFS_V137, apply_op_v137, step_cost_bits_v137
from .grid_v124 import GridV124, grid_shape_v124

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


def step_cost_bits_v141(*, op_id: str, args: Dict[str, Any]) -> int:
    op = str(op_id)
    if op in OP_DEFS_V137:
        return int(step_cost_bits_v137(op_id=str(op_id), args=dict(args)))
    od = OP_DEFS_V141.get(op)
    base = int(od.base_cost_bits) if od is not None else 24
    # bg is a color in 0..9
    extra = 4 if "bg" in args else 0
    return int(base + int(extra))


def apply_op_v141(*, state: StateV132, op_id: str, args: Dict[str, Any]) -> StateV132:
    op = str(op_id)
    a = dict(args)
    if op in OP_DEFS_V137:
        return apply_op_v137(state=state, op_id=str(op_id), args=dict(args))

    if op == "cc4_nonbg_multicolor":
        bg = a.get("bg")
        bgc = int(_check_color_v141(int(bg))) if bg is not None else None
        oset = connected_components4_nonbg_multicolor_v141(state.grid, bg=bgc)
        return replace(state, objset=oset, obj=None, bbox=None, patch=None)

    raise ValueError("unknown_op_v141")

