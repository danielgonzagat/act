from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

from .arc_ops_v132 import OpDefV132, StateV132
from .arc_ops_v134 import OP_DEFS_V134, apply_op_v134, step_cost_bits_v134
from .grid_v124 import GridV124, grid_from_list_v124, grid_shape_v124, translate_v124

ARC_OPS_SCHEMA_VERSION_V135 = 135


def _int_cost_bits_v135(x: int) -> int:
    xx = int(x)
    if xx == 0:
        return 1
    return int(abs(xx).bit_length() + 1)


def _op_args_cost_bits_v135(op_id: str, args: Dict[str, Any]) -> int:
    op = str(op_id)
    a = dict(args)
    if op == "overlay_self_translate":
        bits = 0
        bits += _int_cost_bits_v135(int(a.get("dx", 0)))
        bits += _int_cost_bits_v135(int(a.get("dy", 0)))
        bits += 4  # pad in 0..9
        return int(bits)
    return 0


OP_DEFS_V135 = dict(OP_DEFS_V134)
OP_DEFS_V135["overlay_self_translate"] = OpDefV132(
    op_id="overlay_self_translate",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=20,
)


def step_cost_bits_v135(*, op_id: str, args: Dict[str, Any]) -> int:
    op = str(op_id)
    if op in OP_DEFS_V134:
        return int(step_cost_bits_v134(op_id=str(op_id), args=dict(args)))
    od = OP_DEFS_V135.get(op)
    base = int(od.base_cost_bits) if od is not None else 24
    return int(base + _op_args_cost_bits_v135(op, dict(args)))


def _overlay_self_translate_v135(g: GridV124, *, dx: int, dy: int, pad: int) -> GridV124:
    g_shift = translate_v124(g, dx=int(dx), dy=int(dy), pad=int(pad))
    h, w = grid_shape_v124(g)
    out = [[int(g[r][c]) for c in range(w)] for r in range(h)]
    for r in range(h):
        for c in range(w):
            v = int(g_shift[r][c])
            if v != int(pad):
                out[r][c] = int(v)
    return grid_from_list_v124(out)


def apply_op_v135(*, state: StateV132, op_id: str, args: Dict[str, Any]) -> StateV132:
    op = str(op_id)
    a = dict(args)
    if op in OP_DEFS_V134:
        return apply_op_v134(state=state, op_id=str(op_id), args=dict(args))

    if op == "overlay_self_translate":
        dx = int(a.get("dx", 0))
        dy = int(a.get("dy", 0))
        pad = int(a.get("pad", 0))
        g2 = _overlay_self_translate_v135(state.grid, dx=int(dx), dy=int(dy), pad=int(pad))
        return replace(state, grid=g2, objset=None, obj=None, bbox=None, patch=None)

    raise ValueError("unknown_op_v135")

