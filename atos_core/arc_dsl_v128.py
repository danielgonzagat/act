from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from .arc_dsl_v126 import BboxV126, ObjectV126, ObjectSetV126, apply_op_v126
from .grid_v124 import GridV124, grid_from_list_v124, grid_shape_v124

ValueTypeV128 = Literal["GRID", "BBOX", "MASK", "COLOR", "INT", "OBJECT", "OBJECT_SET"]


def _check_color_v128(c: int) -> int:
    cc = int(c)
    if cc < 0 or cc > 9:
        raise ValueError("color_out_of_range")
    return cc


def mask_difference_v128(a: GridV124, b: GridV124) -> GridV124:
    ha, wa = grid_shape_v124(a)
    hb, wb = grid_shape_v124(b)
    if (ha, wa) != (hb, wb):
        raise ValueError("shape_mismatch")
    rows: List[List[int]] = []
    for r in range(ha):
        row: List[int] = []
        for c in range(wa):
            row.append(1 if int(a[r][c]) != int(b[r][c]) else 0)
        rows.append(row)
    return grid_from_list_v124(rows)


def mask_rect_v128(g: GridV124, bbox: BboxV126) -> GridV124:
    h, w = grid_shape_v124(g)
    r0, c0, r1, c1 = bbox.to_tuple()
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(int(h), int(r1))
    cc1 = min(int(w), int(c1))
    rows: List[List[int]] = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(rr0, rr1):
        for c in range(cc0, cc1):
            rows[r][c] = 1
    return grid_from_list_v124(rows)


def mask_of_object_v128(g: GridV124, o: ObjectV126) -> GridV124:
    h, w = grid_shape_v124(g)
    rows: List[List[int]] = [[0 for _ in range(w)] for _ in range(h)]
    for r, c in o.cells:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < h and 0 <= cc < w:
            rows[rr][cc] = 1
    return grid_from_list_v124(rows)


PaintModeV128 = Literal["overwrite", "only_bg", "only_color"]


def paint_mask_v128(
    g: GridV124,
    mask: GridV124,
    *,
    color: int,
    mode: PaintModeV128 = "overwrite",
    bg: int = 0,
    only_color: Optional[int] = None,
) -> GridV124:
    cc = _check_color_v128(int(color))
    hg, wg = grid_shape_v124(g)
    hm, wm = grid_shape_v124(mask)
    if (hg, wg) != (hm, wm):
        raise ValueError("mask_shape_mismatch")
    mm = str(mode)
    bgc = _check_color_v128(int(bg))
    onlyc = _check_color_v128(int(only_color)) if only_color is not None else None

    out: List[List[int]] = [[int(g[r][c]) for c in range(wg)] for r in range(hg)]
    for r in range(hg):
        for c in range(wg):
            if int(mask[r][c]) == 0:
                continue
            if mm == "overwrite":
                out[r][c] = int(cc)
                continue
            if mm == "only_bg":
                if int(out[r][c]) == int(bgc):
                    out[r][c] = int(cc)
                continue
            if mm == "only_color":
                if onlyc is None:
                    raise ValueError("only_color_missing")
                if int(out[r][c]) == int(onlyc):
                    out[r][c] = int(cc)
                continue
            raise ValueError(f"unknown_paint_mode:{mm}")
    return grid_from_list_v124(out)


@dataclass(frozen=True)
class OpDefV128:
    op_id: str
    input_types: Tuple[ValueTypeV128, ...]
    output_type: ValueTypeV128
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


def _op_defs_base_v128() -> Dict[str, OpDefV128]:
    from .arc_dsl_v126 import OP_DEFS_V126

    base: Dict[str, OpDefV128] = {}
    for k, v in OP_DEFS_V126.items():
        base[str(k)] = OpDefV128(
            op_id=str(v.op_id),
            input_types=tuple(str(t) for t in v.input_types),  # type: ignore[arg-type]
            output_type=str(v.output_type),  # type: ignore[arg-type]
            cost_bits=int(v.cost_bits),
            invariants=tuple(str(s) for s in v.invariants),
        )
    return base


OP_DEFS_V128: Dict[str, OpDefV128] = _op_defs_base_v128()

# Override/extend with new or strengthened ops.
OP_DEFS_V128["mask_difference"] = OpDefV128(
    op_id="mask_difference",
    input_types=("GRID", "GRID"),
    output_type="MASK",
    cost_bits=14,
    invariants=("same shape", "1 where a!=b else 0",),
)

OP_DEFS_V128["mask_rect"] = OpDefV128(
    op_id="mask_rect",
    input_types=("GRID", "BBOX"),
    output_type="MASK",
    cost_bits=16,
    invariants=("mask same shape as grid", "ones inside bbox",),
)

OP_DEFS_V128["mask_of_object"] = OpDefV128(
    op_id="mask_of_object",
    input_types=("GRID", "OBJECT"),
    output_type="MASK",
    cost_bits=18,
    invariants=("mask same shape as grid", "ones at object cells",),
)

# V128 paint_mask supports deterministic modes.
OP_DEFS_V128["paint_mask"] = OpDefV128(
    op_id="paint_mask",
    input_types=("GRID", "MASK"),
    output_type="GRID",
    cost_bits=20,
    invariants=("mask same shape as grid", "mode in {overwrite,only_bg,only_color}", "color in 0..9",),
)

# Extend select_object beyond v126, but keep the same op_id for compilation.
SelectorV128 = Literal[
    "largest_area",
    "smallest_area",
    "leftmost",
    "rightmost",
    "topmost",
    "bottommost",
    "closest_to_center",
]


def _area_of_bbox_v128(b: BboxV126) -> int:
    r0, c0, r1, c1 = b.to_tuple()
    return max(0, int(r1 - r0)) * max(0, int(c1 - c0))


def _scene_bbox_v128(oset: ObjectSetV126) -> BboxV126:
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


def _bbox_center_v128(b: BboxV126) -> Tuple[int, int]:
    r0, c0, r1, c1 = b.to_tuple()
    # Center in integer coords, deterministic rounding down.
    return int((int(r0) + int(r1) - 1) // 2), int((int(c0) + int(c1) - 1) // 2)


def select_object_v128(obj_set: ObjectSetV126, *, selector: SelectorV128) -> ObjectV126:
    if not obj_set.objects:
        raise ValueError("empty_object_set")
    sel = str(selector)
    objs = list(obj_set.objects)
    if sel == "largest_area":
        objs.sort(key=lambda o: (-_area_of_bbox_v128(o.bbox), o.bbox.to_tuple(), int(o.color), o.cells))
        return objs[0]
    if sel == "smallest_area":
        objs.sort(key=lambda o: (_area_of_bbox_v128(o.bbox), o.bbox.to_tuple(), int(o.color), o.cells))
        return objs[0]
    if sel == "leftmost":
        objs.sort(key=lambda o: (o.bbox.c0, o.bbox.r0, o.bbox.to_tuple(), int(o.color), o.cells))
        return objs[0]
    if sel == "rightmost":
        objs.sort(key=lambda o: (-int(o.bbox.c1), o.bbox.r0, o.bbox.to_tuple(), int(o.color), o.cells))
        return objs[0]
    if sel == "topmost":
        objs.sort(key=lambda o: (o.bbox.r0, o.bbox.c0, o.bbox.to_tuple(), int(o.color), o.cells))
        return objs[0]
    if sel == "bottommost":
        objs.sort(key=lambda o: (-int(o.bbox.r1), o.bbox.c0, o.bbox.to_tuple(), int(o.color), o.cells))
        return objs[0]
    if sel == "closest_to_center":
        scene = _scene_bbox_v128(obj_set)
        sr, sc = _bbox_center_v128(scene)

        def dist(o: ObjectV126) -> Tuple[int, int, int, int, Tuple[int, int, int, int], int]:
            orr, occ = _bbox_center_v128(o.bbox)
            d = abs(int(orr) - int(sr)) + abs(int(occ) - int(sc))
            return (int(d), _area_of_bbox_v128(o.bbox), int(o.bbox.r0), int(o.bbox.c0), o.bbox.to_tuple(), int(o.color))

        objs.sort(key=dist)
        return objs[0]
    raise ValueError(f"unknown_selector:{sel}")


OP_DEFS_V128["select_object"] = OpDefV128(
    op_id="select_object",
    input_types=("OBJECT_SET",),
    output_type="OBJECT",
    cost_bits=18,
    invariants=("selector in enum",),
)


def apply_op_v128(*, op_id: str, inputs: Sequence[Any], args: Dict[str, Any]) -> Any:
    op = str(op_id)
    a = dict(args)

    if op == "mask_difference":
        return mask_difference_v128(inputs[0], inputs[1])
    if op == "mask_rect":
        return mask_rect_v128(inputs[0], inputs[1])
    if op == "mask_of_object":
        return mask_of_object_v128(inputs[0], inputs[1])
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
        return select_object_v128(inputs[0], selector=str(a.get("selector") or ""))
    return apply_op_v126(op_id=op, inputs=list(inputs), args=dict(a))
