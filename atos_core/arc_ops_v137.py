from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

from .arc_ops_v132 import OpDefV132, StateV132
from .arc_ops_v135 import OP_DEFS_V135, apply_op_v135, step_cost_bits_v135
from .grid_v124 import GridV124, grid_from_list_v124, grid_shape_v124

ARC_OPS_SCHEMA_VERSION_V137 = 137


def _int_cost_bits_v137(x: int) -> int:
    xx = int(x)
    if xx == 0:
        return 1
    return int(abs(xx).bit_length() + 1)


def _op_args_cost_bits_v137(op_id: str, args: Dict[str, Any]) -> int:
    op = str(op_id)
    a = dict(args)
    if op == "propagate_color_translate":
        bits = 0
        bits += _int_cost_bits_v137(int(a.get("dx", 0)))
        bits += _int_cost_bits_v137(int(a.get("dy", 0)))
        bits += 4  # color in 0..9
        bits += 4  # pad in 0..9
        return int(bits)
    return 0


OP_DEFS_V137 = dict(OP_DEFS_V135)
OP_DEFS_V137["propagate_color_translate"] = OpDefV132(
    op_id="propagate_color_translate",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=22,
)


def step_cost_bits_v137(*, op_id: str, args: Dict[str, Any]) -> int:
    op = str(op_id)
    if op in OP_DEFS_V135:
        return int(step_cost_bits_v135(op_id=str(op_id), args=dict(args)))
    od = OP_DEFS_V137.get(op)
    base = int(od.base_cost_bits) if od is not None else 24
    return int(base + _op_args_cost_bits_v137(op, dict(args)))


def _propagate_color_translate_v137(
    g: GridV124,
    *,
    dx: int,
    dy: int,
    color: int,
    pad: int,
) -> GridV124:
    h, w = grid_shape_v124(g)
    if h <= 0 or w <= 0:
        return g
    ddx = int(dx)
    ddy = int(dy)
    if ddx == 0 and ddy == 0:
        return g
    cc = int(color)
    pp = int(pad)
    cur = [[int(g[r][c]) for c in range(w)] for r in range(h)]
    # A conservative deterministic iteration bound (finite grid, fixed dx/dy).
    max_iters = int(h + w + 1)
    for _ in range(max_iters):
        changed = False
        nxt = [row[:] for row in cur]
        for r in range(h):
            src_r = int(r - ddy)
            if src_r < 0 or src_r >= h:
                continue
            for c in range(w):
                if int(cur[r][c]) != pp:
                    continue
                src_c = int(c - ddx)
                if src_c < 0 or src_c >= w:
                    continue
                if int(cur[src_r][src_c]) == cc:
                    nxt[r][c] = int(cc)
                    changed = True
        if not changed:
            break
        cur = nxt
    return grid_from_list_v124(cur)


def apply_op_v137(*, state: StateV132, op_id: str, args: Dict[str, Any]) -> StateV132:
    op = str(op_id)
    a = dict(args)
    if op in OP_DEFS_V135:
        return apply_op_v135(state=state, op_id=str(op_id), args=dict(args))

    if op == "propagate_color_translate":
        dx = int(a.get("dx", 0))
        dy = int(a.get("dy", 0))
        color = int(a.get("color", 0))
        pad = int(a.get("pad", 0))
        g2 = _propagate_color_translate_v137(state.grid, dx=int(dx), dy=int(dy), color=int(color), pad=int(pad))
        return replace(state, grid=g2, objset=None, obj=None, bbox=None, patch=None)

    raise ValueError("unknown_op_v137")

