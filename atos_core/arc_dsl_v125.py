from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .grid_v124 import (
    ComponentV124,
    GridV124,
    bbox_nonzero_v124,
    connected_components4_v124,
    crop_v124,
    grid_from_list_v124,
    grid_shape_v124,
    mask_by_color_v124,
    overlay_v124,
)

ValueTypeV125 = Literal["GRID", "BBOX", "MASK", "COLOR", "INT", "OBJECT", "OBJECT_SET"]


@dataclass(frozen=True)
class BboxV125:
    r0: int
    c0: int
    r1: int
    c1: int

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return int(self.r0), int(self.c0), int(self.r1), int(self.c1)

    def to_dict(self) -> Dict[str, Any]:
        r0, c0, r1, c1 = self.to_tuple()
        return {"r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1)}


@dataclass(frozen=True)
class ObjectV125:
    # Minimal deterministic object: cells (sorted), color, bbox.
    cells: Tuple[Tuple[int, int], ...]
    color: int
    bbox: BboxV125

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cells": [{"r": int(r), "c": int(c)} for r, c in self.cells],
            "color": int(self.color),
            "bbox": self.bbox.to_dict(),
        }

    def object_sig(self) -> str:
        body = {"schema_version": 125, "kind": "object_v125", **self.to_dict()}
        return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


@dataclass(frozen=True)
class ObjectSetV125:
    objects: Tuple[ObjectV125, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"objects": [o.to_dict() for o in self.objects]}

    def set_sig(self) -> str:
        body = {"schema_version": 125, "kind": "object_set_v125", **self.to_dict()}
        return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _check_color(c: int) -> int:
    cc = int(c)
    if cc < 0 or cc > 9:
        raise ValueError("color_out_of_range")
    return cc


def _check_thickness(t: int) -> int:
    tt = int(t)
    if tt <= 0:
        raise ValueError("thickness_nonpositive")
    return tt


def bbox_from_tuple_v125(t: Tuple[int, int, int, int]) -> BboxV125:
    r0, c0, r1, c1 = (int(x) for x in t)
    return BboxV125(r0=int(r0), c0=int(c0), r1=int(r1), c1=int(c1))


def bbox_nonzero_v125(g: GridV124, *, bg: int = 0) -> BboxV125:
    return bbox_from_tuple_v125(bbox_nonzero_v124(g, bg=int(bg)))


def bbox_by_color_v125(g: GridV124, *, color: int) -> BboxV125:
    m = mask_by_color_v124(g, color=_check_color(int(color)))
    return bbox_from_tuple_v125(bbox_nonzero_v124(m, bg=0))


def crop_bbox_v125(g: GridV124, bbox: BboxV125) -> GridV124:
    r0, c0, r1, c1 = bbox.to_tuple()
    if r1 <= r0 or c1 <= c0:
        return tuple()
    return crop_v124(g, r0=int(r0), c0=int(c0), height=int(r1 - r0), width=int(c1 - c0))


def paint_mask_v125(g: GridV124, mask: GridV124, *, color: int) -> GridV124:
    cc = _check_color(int(color))
    hg, wg = grid_shape_v124(g)
    hm, wm = grid_shape_v124(mask)
    if (hg, wg) != (hm, wm):
        raise ValueError("mask_shape_mismatch")
    out: List[List[int]] = [[int(g[r][c]) for c in range(wg)] for r in range(hg)]
    for r in range(hg):
        for c in range(wg):
            if int(mask[r][c]) != 0:
                out[r][c] = int(cc)
    return grid_from_list_v124(out)


def fill_rect_v125(g: GridV124, bbox: BboxV125, *, color: int) -> GridV124:
    cc = _check_color(int(color))
    h, w = grid_shape_v124(g)
    r0, c0, r1, c1 = bbox.to_tuple()
    out: List[List[int]] = [[int(g[r][c]) for c in range(w)] for r in range(h)]
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h, int(r1))
    cc1 = min(w, int(c1))
    for r in range(rr0, rr1):
        for c in range(cc0, cc1):
            out[r][c] = int(cc)
    return grid_from_list_v124(out)


def draw_rect_border_v125(g: GridV124, bbox: BboxV125, *, color: int, thickness: int = 1) -> GridV124:
    cc = _check_color(int(color))
    t = _check_thickness(int(thickness))
    h, w = grid_shape_v124(g)
    r0, c0, r1, c1 = bbox.to_tuple()
    out: List[List[int]] = [[int(g[r][c]) for c in range(w)] for r in range(h)]
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h, int(r1))
    cc1 = min(w, int(c1))
    if rr1 <= rr0 or cc1 <= cc0:
        return grid_from_list_v124(out)

    for k in range(t):
        top = rr0 + k
        bot = rr1 - 1 - k
        left = cc0 + k
        right = cc1 - 1 - k
        if top > bot or left > right:
            break
        for c in range(left, right + 1):
            if 0 <= top < h and 0 <= c < w:
                out[top][c] = int(cc)
            if 0 <= bot < h and 0 <= c < w:
                out[bot][c] = int(cc)
        for r in range(top, bot + 1):
            if 0 <= r < h and 0 <= left < w:
                out[r][left] = int(cc)
            if 0 <= r < h and 0 <= right < w:
                out[r][right] = int(cc)
    return grid_from_list_v124(out)


def connected_components_v125(g: GridV124, *, color: Optional[int] = None) -> ObjectSetV125:
    comps = connected_components4_v124(g, color=int(color) if color is not None else None)
    objs: List[ObjectV125] = []
    for comp in comps:
        bbox = bbox_from_tuple_v125(comp.bbox)
        objs.append(ObjectV125(cells=tuple(comp.cells), color=int(comp.color), bbox=bbox))
    objs.sort(key=lambda o: (o.bbox.to_tuple(), int(o.color), o.cells))
    return ObjectSetV125(objects=tuple(objs))


SelectorV125 = Literal["largest_area", "smallest_area", "leftmost", "topmost"]


def _area_of_bbox(b: BboxV125) -> int:
    r0, c0, r1, c1 = b.to_tuple()
    return max(0, int(r1 - r0)) * max(0, int(c1 - c0))


def select_object_v125(obj_set: ObjectSetV125, *, selector: SelectorV125) -> ObjectV125:
    if not obj_set.objects:
        raise ValueError("empty_object_set")
    sel = str(selector)
    objs = list(obj_set.objects)
    if sel == "largest_area":
        objs.sort(key=lambda o: (-_area_of_bbox(o.bbox), o.bbox.to_tuple(), int(o.color), o.cells))
        return objs[0]
    if sel == "smallest_area":
        objs.sort(key=lambda o: (_area_of_bbox(o.bbox), o.bbox.to_tuple(), int(o.color), o.cells))
        return objs[0]
    if sel == "leftmost":
        objs.sort(key=lambda o: (o.bbox.c0, o.bbox.r0, o.bbox.to_tuple(), int(o.color), o.cells))
        return objs[0]
    if sel == "topmost":
        objs.sort(key=lambda o: (o.bbox.r0, o.bbox.c0, o.bbox.to_tuple(), int(o.color), o.cells))
        return objs[0]
    raise ValueError(f"unknown_selector:{sel}")


def overlay_grid_v125(base: GridV124, top: GridV124, *, transparent: int = 0) -> GridV124:
    return overlay_v124(base, top, transparent=_check_color(int(transparent)))


def value_to_canonical_json_v125(v: Any) -> Dict[str, Any]:
    # GridV124 is a tuple of tuple[int]; allow empty grid tuple().
    if isinstance(v, tuple) and (not v or isinstance(v[0], tuple)):
        # GridV124
        return {"type": "GRID", "grid": [list(r) for r in v]}  # type: ignore[arg-type]
    if isinstance(v, BboxV125):
        return {"type": "BBOX", **v.to_dict()}
    if isinstance(v, ObjectV125):
        return {"type": "OBJECT", **v.to_dict()}
    if isinstance(v, ObjectSetV125):
        return {"type": "OBJECT_SET", "objects": [o.to_dict() for o in v.objects]}
    if isinstance(v, int):
        # could be COLOR or INT; caller must disambiguate; keep as INT for hashing.
        return {"type": "INT", "value": int(v)}
    raise TypeError("unsupported_value_type_v125")


def value_sig_v125(v: Any) -> str:
    body = {"schema_version": 125, "kind": "dsl_value_v125", "value": value_to_canonical_json_v125(v)}
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


@dataclass(frozen=True)
class OpDefV125:
    op_id: str
    input_types: Tuple[ValueTypeV125, ...]
    output_type: ValueTypeV125
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


OP_DEFS_V125: Dict[str, OpDefV125] = {
    "bbox_nonzero": OpDefV125(
        op_id="bbox_nonzero",
        input_types=("GRID",),
        output_type="BBOX",
        cost_bits=12,
        invariants=("bg in 0..9",),
    ),
    "bbox_by_color": OpDefV125(
        op_id="bbox_by_color",
        input_types=("GRID",),
        output_type="BBOX",
        cost_bits=14,
        invariants=("color in 0..9",),
    ),
    "mask_by_color": OpDefV125(
        op_id="mask_by_color",
        input_types=("GRID",),
        output_type="MASK",
        cost_bits=14,
        invariants=("color in 0..9",),
    ),
    "crop_bbox": OpDefV125(
        op_id="crop_bbox",
        input_types=("GRID", "BBOX"),
        output_type="GRID",
        cost_bits=14,
        invariants=("bbox within bounds (empty ok -> empty grid)",),
    ),
    "fill_rect": OpDefV125(
        op_id="fill_rect",
        input_types=("GRID", "BBOX"),
        output_type="GRID",
        cost_bits=18,
        invariants=("color in 0..9",),
    ),
    "draw_rect_border": OpDefV125(
        op_id="draw_rect_border",
        input_types=("GRID", "BBOX"),
        output_type="GRID",
        cost_bits=20,
        invariants=("color in 0..9", "thickness>=1",),
    ),
    "paint_mask": OpDefV125(
        op_id="paint_mask",
        input_types=("GRID", "MASK"),
        output_type="GRID",
        cost_bits=18,
        invariants=("mask same shape as grid", "color in 0..9",),
    ),
    "overlay": OpDefV125(
        op_id="overlay",
        input_types=("GRID", "GRID"),
        output_type="GRID",
        cost_bits=18,
        invariants=("same shape", "transparent in 0..9",),
    ),
    "connected_components": OpDefV125(
        op_id="connected_components",
        input_types=("GRID",),
        output_type="OBJECT_SET",
        cost_bits=30,
        invariants=("4-neighborhood",),
    ),
    "select_object": OpDefV125(
        op_id="select_object",
        input_types=("OBJECT_SET",),
        output_type="OBJECT",
        cost_bits=16,
        invariants=("selector in enum",),
    ),
    "bbox_of_object": OpDefV125(
        op_id="bbox_of_object",
        input_types=("OBJECT",),
        output_type="BBOX",
        cost_bits=8,
        invariants=(),
    ),
}


def apply_op_v125(*, op_id: str, inputs: Sequence[Any], args: Dict[str, Any]) -> Any:
    op = str(op_id)
    a = dict(args)
    if op == "bbox_nonzero":
        g = inputs[0]
        return bbox_nonzero_v125(g, bg=int(a.get("bg", 0)))
    if op == "bbox_by_color":
        g = inputs[0]
        return bbox_by_color_v125(g, color=int(a["color"]))
    if op == "mask_by_color":
        g = inputs[0]
        return mask_by_color_v124(g, color=_check_color(int(a["color"])))
    if op == "crop_bbox":
        g = inputs[0]
        b = inputs[1]
        return crop_bbox_v125(g, b)
    if op == "fill_rect":
        g = inputs[0]
        b = inputs[1]
        return fill_rect_v125(g, b, color=int(a["color"]))
    if op == "draw_rect_border":
        g = inputs[0]
        b = inputs[1]
        return draw_rect_border_v125(g, b, color=int(a["color"]), thickness=int(a.get("thickness", 1)))
    if op == "paint_mask":
        g = inputs[0]
        m = inputs[1]
        return paint_mask_v125(g, m, color=int(a["color"]))
    if op == "overlay":
        g0 = inputs[0]
        g1 = inputs[1]
        return overlay_grid_v125(g0, g1, transparent=int(a.get("transparent", 0)))
    if op == "connected_components":
        g = inputs[0]
        if "color" in a:
            return connected_components_v125(g, color=int(a["color"]))
        return connected_components_v125(g)
    if op == "select_object":
        os = inputs[0]
        return select_object_v125(os, selector=str(a["selector"]))  # type: ignore[arg-type]
    if op == "bbox_of_object":
        o = inputs[0]
        return o.bbox
    raise ValueError(f"unknown_op:{op}")
