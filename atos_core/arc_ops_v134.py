from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from .arc_objects_v132 import BBoxV132, _check_color_v132
from .arc_ops_v132 import OP_DEFS_V132, OpDefV132, StateV132, apply_op_v132, step_cost_bits_v132
from .grid_v124 import GridV124, grid_from_list_v124, grid_shape_v124

ARC_OPS_SCHEMA_VERSION_V134 = 134


def _int_cost_bits_v134(x: int) -> int:
    xx = int(x)
    if xx == 0:
        return 1
    return int(abs(xx).bit_length() + 1)


def _op_args_cost_bits_v134(op_id: str, args: Dict[str, Any]) -> int:
    op = str(op_id)
    a = dict(args)
    bits = 0
    if op == "bbox_by_color":
        bits += 4  # color in 0..9
        return int(bits)
    if op == "repeat_grid":
        mode = str(a.get("mode") or "")
        bits += 2  # mode tag (cell/grid)
        if mode == "cell":
            bits += _int_cost_bits_v134(int(a.get("sy", 1)))
            bits += _int_cost_bits_v134(int(a.get("sx", 1)))
        elif mode == "grid":
            bits += _int_cost_bits_v134(int(a.get("ry", 1)))
            bits += _int_cost_bits_v134(int(a.get("rx", 1)))
        else:
            bits += 8
        return int(bits)
    # Fallback: no extra cost.
    return int(bits)


OP_DEFS_V134 = dict(OP_DEFS_V132)
OP_DEFS_V134["bbox_by_color"] = OpDefV132(
    op_id="bbox_by_color",
    reads=("grid",),
    writes=("bbox",),
    base_cost_bits=10,
)
OP_DEFS_V134["repeat_grid"] = OpDefV132(
    op_id="repeat_grid",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=18,
)


def step_cost_bits_v134(*, op_id: str, args: Dict[str, Any]) -> int:
    op = str(op_id)
    if op in OP_DEFS_V132:
        return int(step_cost_bits_v132(op_id=str(op_id), args=dict(args)))
    od = OP_DEFS_V134.get(op)
    base = int(od.base_cost_bits) if od is not None else 24
    return int(base + _op_args_cost_bits_v134(op, dict(args)))


def _bbox_by_color_v134(g: GridV124, *, color: int) -> Optional[BBoxV132]:
    cc = _check_color_v132(int(color))
    h, w = grid_shape_v124(g)
    rmin = h
    cmin = w
    rmax = -1
    cmax = -1
    any_hit = False
    for r in range(h):
        for c in range(w):
            if int(g[r][c]) == int(cc):
                any_hit = True
                rmin = min(int(rmin), int(r))
                cmin = min(int(cmin), int(c))
                rmax = max(int(rmax), int(r))
                cmax = max(int(cmax), int(c))
    if not any_hit:
        return None
    return BBoxV132(r0=int(rmin), c0=int(cmin), r1=int(rmax + 1), c1=int(cmax + 1))


def _repeat_grid_cell_v134(g: GridV124, *, sy: int, sx: int) -> GridV124:
    hi, wi = grid_shape_v124(g)
    ssy = int(sy)
    ssx = int(sx)
    if ssy <= 0 or ssx <= 0:
        raise ValueError("repeat_grid_nonpositive_scale")
    out_h = int(hi * ssy)
    out_w = int(wi * ssx)
    out = [[0 for _ in range(out_w)] for _ in range(out_h)]
    for r in range(out_h):
        for c in range(out_w):
            out[r][c] = int(g[r // ssy][c // ssx])
    return grid_from_list_v124(out)


def _repeat_grid_tile_v134(g: GridV124, *, ry: int, rx: int) -> GridV124:
    hi, wi = grid_shape_v124(g)
    rry = int(ry)
    rrx = int(rx)
    if rry <= 0 or rrx <= 0:
        raise ValueError("repeat_grid_nonpositive_tile")
    out_h = int(hi * rry)
    out_w = int(wi * rrx)
    out = [[0 for _ in range(out_w)] for _ in range(out_h)]
    for r in range(out_h):
        for c in range(out_w):
            out[r][c] = int(g[r % hi][c % wi])
    return grid_from_list_v124(out)


def apply_op_v134(*, state: StateV132, op_id: str, args: Dict[str, Any]) -> StateV132:
    op = str(op_id)
    a = dict(args)
    if op in OP_DEFS_V132:
        return apply_op_v132(state=state, op_id=str(op_id), args=dict(args))

    if op == "bbox_by_color":
        color = int(a.get("color", 0))
        b = _bbox_by_color_v134(state.grid, color=int(color))
        if b is None:
            raise ValueError("bbox_by_color_empty")
        return replace(state, bbox=b, objset=None, obj=None, patch=None)

    if op == "repeat_grid":
        mode = str(a.get("mode") or "")
        if mode == "cell":
            sy = int(a.get("sy", 1))
            sx = int(a.get("sx", 1))
            g2 = _repeat_grid_cell_v134(state.grid, sy=int(sy), sx=int(sx))
        elif mode == "grid":
            ry = int(a.get("ry", 1))
            rx = int(a.get("rx", 1))
            g2 = _repeat_grid_tile_v134(state.grid, ry=int(ry), rx=int(rx))
        else:
            raise ValueError("repeat_grid_unknown_mode")
        return replace(state, grid=g2, objset=None, obj=None, bbox=None, patch=None)

    raise ValueError("unknown_op_v134")

