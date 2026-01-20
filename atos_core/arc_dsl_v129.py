from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from .arc_dsl_v126 import BboxV126, ObjectV126, ObjectSetV126, apply_op_v126
from .arc_dsl_v128 import PaintModeV128, paint_mask_v128, select_object_v128
from .grid_v124 import (
    GridV124,
    grid_from_list_v124,
    grid_shape_v124,
    reflect_h_v124,
    reflect_v_v124,
    rotate180_v124,
    rotate270_v124,
    rotate90_v124,
)

ValueTypeV129 = Literal["GRID", "BBOX", "MASK", "COLOR", "INT", "OBJECT", "OBJECT_SET"]


def _check_color_v129(c: int) -> int:
    cc = int(c)
    if cc < 0 or cc > 9:
        raise ValueError("color_out_of_range")
    return cc


def transpose_v129(g: GridV124) -> GridV124:
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return g
    out: List[Tuple[int, ...]] = []
    for c in range(w):
        out.append(tuple(int(g[r][c]) for r in range(h)))
    return tuple(out)


def scale_up_v129(g: GridV124, *, k: int) -> GridV124:
    kk = int(k)
    if kk <= 0:
        raise ValueError("scale_k_nonpositive")
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return g
    out_rows: List[List[int]] = []
    for r in range(h):
        row_src = list(g[r])
        row_scaled: List[int] = []
        for x in row_src:
            row_scaled.extend([int(x)] * kk)
        for _ in range(kk):
            out_rows.append(list(row_scaled))
    return grid_from_list_v124(out_rows)


def tile_v129(g: GridV124, *, reps_h: int, reps_w: int) -> GridV124:
    rh = int(reps_h)
    rw = int(reps_w)
    if rh <= 0 or rw <= 0:
        raise ValueError("tile_reps_nonpositive")
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return g
    out_rows: List[List[int]] = []
    for _ in range(rh):
        for r in range(h):
            row: List[int] = []
            for _ in range(rw):
                row.extend([int(x) for x in g[r]])
            out_rows.append(row)
    return grid_from_list_v124(out_rows)


def mask_outline_v129(mask: GridV124) -> GridV124:
    h, w = grid_shape_v124(mask)
    if h == 0 or w == 0:
        return mask
    out: List[List[int]] = [[0 for _ in range(w)] for _ in range(h)]
    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(h):
        for c in range(w):
            if int(mask[r][c]) == 0:
                continue
            is_border = False
            for dr, dc in neigh:
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    is_border = True
                    break
                if int(mask[rr][cc]) == 0:
                    is_border = True
                    break
            if is_border:
                out[r][c] = 1
    return grid_from_list_v124(out)


def mask_dilate_v129(mask: GridV124, *, steps: int = 1) -> GridV124:
    s = int(steps)
    if s < 0:
        raise ValueError("dilate_steps_negative")
    cur = mask
    h, w = grid_shape_v124(cur)
    if h == 0 or w == 0:
        return cur
    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _ in range(s):
        out: List[List[int]] = [[int(cur[r][c]) for c in range(w)] for r in range(h)]
        for r in range(h):
            for c in range(w):
                if int(cur[r][c]) != 0:
                    continue
                for dr, dc in neigh:
                    rr = r + dr
                    cc = c + dc
                    if rr < 0 or rr >= h or cc < 0 or cc >= w:
                        continue
                    if int(cur[rr][cc]) != 0:
                        out[r][c] = 1
                        break
        cur = grid_from_list_v124(out)
    return cur


SelectorV129 = Literal[
    # v128 selectors
    "largest_area",
    "smallest_area",
    "leftmost",
    "rightmost",
    "topmost",
    "bottommost",
    "closest_to_center",
    # v129: corner proximity (general)
    "closest_to_tl",
    "closest_to_tr",
    "closest_to_bl",
    "closest_to_br",
]


def _scene_bbox_v129(oset: ObjectSetV126) -> BboxV126:
    r0 = None
    c0 = None
    r1 = None
    c1 = None
    for o in oset.objects:
        rr0, cc0, rr1, cc1 = o.bbox.to_tuple()
        r0 = int(rr0) if r0 is None else min(int(r0), int(rr0))
        c0 = int(cc0) if c0 is None else min(int(c0), int(cc0))
        r1 = int(rr1) if r1 is None else max(int(r1), int(rr1))
        c1 = int(cc1) if c1 is None else max(int(c1), int(cc1))
    if r0 is None or c0 is None or r1 is None or c1 is None:
        return BboxV126(r0=0, c0=0, r1=0, c1=0)
    return BboxV126(r0=int(r0), c0=int(c0), r1=int(r1), c1=int(c1))


def _bbox_center_v129(b: BboxV126) -> Tuple[int, int]:
    r0, c0, r1, c1 = b.to_tuple()
    return int((int(r0) + int(r1) - 1) // 2), int((int(c0) + int(c1) - 1) // 2)


def select_object_v129(obj_set: ObjectSetV126, *, selector: SelectorV129) -> ObjectV126:
    sel = str(selector)
    # Delegate to v128 selector when available.
    if sel in (
        "largest_area",
        "smallest_area",
        "leftmost",
        "rightmost",
        "topmost",
        "bottommost",
        "closest_to_center",
    ):
        return select_object_v128(obj_set, selector=sel)  # type: ignore[arg-type]

    if not obj_set.objects:
        raise ValueError("empty_object_set")

    scene = _scene_bbox_v129(obj_set)
    sr0, sc0, sr1, sc1 = scene.to_tuple()
    # Corner points are inclusive; use deterministic integer coords.
    corners: Dict[str, Tuple[int, int]] = {
        "closest_to_tl": (int(sr0), int(sc0)),
        "closest_to_tr": (int(sr0), max(int(sc0), int(sc1) - 1)),
        "closest_to_bl": (max(int(sr0), int(sr1) - 1), int(sc0)),
        "closest_to_br": (max(int(sr0), int(sr1) - 1), max(int(sc0), int(sc1) - 1)),
    }
    if sel not in corners:
        raise ValueError(f"unknown_selector:{sel}")
    cr, cc = corners[sel]

    def dist(o: ObjectV126) -> Tuple[int, Tuple[int, int, int, int], int, Tuple[Tuple[int, int], ...]]:
        orr, occ = _bbox_center_v129(o.bbox)
        d = abs(int(orr) - int(cr)) + abs(int(occ) - int(cc))
        # tie-break: bbox, color, cells
        return (int(d), o.bbox.to_tuple(), int(o.color), o.cells)

    objs = list(obj_set.objects)
    objs.sort(key=dist)
    return objs[0]


@dataclass(frozen=True)
class OpDefV129:
    op_id: str
    input_types: Tuple[ValueTypeV129, ...]
    output_type: ValueTypeV129
    cost_bits: int
    invariants: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_id": str(self.op_id),
            "input_types": [str(t) for t in self.input_types],
            "output_type": str(self.output_type),
            "cost_bits": int(self.cost_bits),
            "invariants": [str(s) for s in self.invariants],
        }


def _op_defs_base_v129() -> Dict[str, OpDefV129]:
    from .arc_dsl_v128 import OP_DEFS_V128

    base: Dict[str, OpDefV129] = {}
    for k, v in OP_DEFS_V128.items():
        base[str(k)] = OpDefV129(
            op_id=str(v.op_id),
            input_types=tuple(str(t) for t in v.input_types),  # type: ignore[arg-type]
            output_type=str(v.output_type),  # type: ignore[arg-type]
            cost_bits=int(v.cost_bits),
            invariants=tuple(str(s) for s in v.invariants),
        )
    return base


OP_DEFS_V129: Dict[str, OpDefV129] = _op_defs_base_v129()

# Symmetry / geometry (general, not ARC-specific).
OP_DEFS_V129["rotate90"] = OpDefV129(
    op_id="rotate90",
    input_types=("GRID",),
    output_type="GRID",
    cost_bits=12,
    invariants=("pure", "shape becomes (w,h)",),
)
OP_DEFS_V129["rotate180"] = OpDefV129(
    op_id="rotate180",
    input_types=("GRID",),
    output_type="GRID",
    cost_bits=12,
    invariants=("pure", "shape preserved",),
)
OP_DEFS_V129["rotate270"] = OpDefV129(
    op_id="rotate270",
    input_types=("GRID",),
    output_type="GRID",
    cost_bits=12,
    invariants=("pure", "shape becomes (w,h)",),
)
OP_DEFS_V129["reflect_h"] = OpDefV129(
    op_id="reflect_h",
    input_types=("GRID",),
    output_type="GRID",
    cost_bits=10,
    invariants=("pure", "horizontal reflection",),
)
OP_DEFS_V129["reflect_v"] = OpDefV129(
    op_id="reflect_v",
    input_types=("GRID",),
    output_type="GRID",
    cost_bits=10,
    invariants=("pure", "vertical reflection",),
)
OP_DEFS_V129["transpose"] = OpDefV129(
    op_id="transpose",
    input_types=("GRID",),
    output_type="GRID",
    cost_bits=12,
    invariants=("pure", "shape becomes (w,h)",),
)

# Scale / repetition (general).
OP_DEFS_V129["scale_up"] = OpDefV129(
    op_id="scale_up",
    input_types=("GRID",),
    output_type="GRID",
    cost_bits=20,
    invariants=("k>=1", "nearest-neighbor replication",),
)
OP_DEFS_V129["tile"] = OpDefV129(
    op_id="tile",
    input_types=("GRID",),
    output_type="GRID",
    cost_bits=20,
    invariants=("reps_h>=1", "reps_w>=1",),
)

# Mask morphology (general).
OP_DEFS_V129["mask_outline"] = OpDefV129(
    op_id="mask_outline",
    input_types=("MASK",),
    output_type="MASK",
    cost_bits=16,
    invariants=("4-neighborhood", "subset of mask",),
)
OP_DEFS_V129["mask_dilate"] = OpDefV129(
    op_id="mask_dilate",
    input_types=("MASK",),
    output_type="MASK",
    cost_bits=18,
    invariants=("4-neighborhood", "steps>=0",),
)


def apply_op_v129(*, op_id: str, inputs: Sequence[Any], args: Dict[str, Any]) -> Any:
    op = str(op_id)
    a = dict(args)

    if op == "rotate90":
        return rotate90_v124(inputs[0])
    if op == "rotate180":
        return rotate180_v124(inputs[0])
    if op == "rotate270":
        return rotate270_v124(inputs[0])
    if op == "reflect_h":
        return reflect_h_v124(inputs[0])
    if op == "reflect_v":
        return reflect_v_v124(inputs[0])
    if op == "transpose":
        return transpose_v129(inputs[0])
    if op == "scale_up":
        return scale_up_v129(inputs[0], k=int(a.get("k", 1)))
    if op == "tile":
        return tile_v129(inputs[0], reps_h=int(a.get("reps_h", 1)), reps_w=int(a.get("reps_w", 1)))
    if op == "mask_outline":
        return mask_outline_v129(inputs[0])
    if op == "mask_dilate":
        return mask_dilate_v129(inputs[0], steps=int(a.get("steps", 1)))

    # V128 overrides (paint_mask modes and selector).
    if op == "paint_mask":
        return paint_mask_v128(
            inputs[0],
            inputs[1],
            color=int(a["color"]),
            mode=str(a.get("mode", "overwrite")),  # type: ignore[arg-type]
            bg=int(a.get("bg", 0)),
            only_color=int(a["only_color"]) if "only_color" in a else None,
        )
    if op == "select_object":
        return select_object_v129(inputs[0], selector=str(a.get("selector") or ""))  # type: ignore[arg-type]

    # Fall back to v126 base ops (bbox/mask_by_color/fill_rect/draw_rect_border/flood_fill/overlay/paste/etc).
    return apply_op_v126(op_id=op, inputs=list(inputs), args=dict(a))
