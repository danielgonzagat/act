from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .grid_v124 import (
    GridV124,
    bbox_nonzero_v124,
    connected_components4_v124,
    crop_v124,
    grid_from_list_v124,
    grid_shape_v124,
    mask_by_color_v124,
    overlay_v124,
)

ValueTypeV126 = Literal["GRID", "BBOX", "MASK", "COLOR", "INT", "OBJECT", "OBJECT_SET"]


@dataclass(frozen=True)
class BboxV126:
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
class ObjectV126:
    # Minimal deterministic object: cells (sorted), color, bbox.
    cells: Tuple[Tuple[int, int], ...]
    color: int
    bbox: BboxV126

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cells": [{"r": int(r), "c": int(c)} for r, c in self.cells],
            "color": int(self.color),
            "bbox": self.bbox.to_dict(),
        }

    def object_sig(self) -> str:
        body = {"schema_version": 126, "kind": "object_v126", **self.to_dict()}
        return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


@dataclass(frozen=True)
class ObjectSetV126:
    objects: Tuple[ObjectV126, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"objects": [o.to_dict() for o in self.objects]}

    def set_sig(self) -> str:
        body = {"schema_version": 126, "kind": "object_set_v126", **self.to_dict()}
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


def bbox_from_tuple_v126(t: Tuple[int, int, int, int]) -> BboxV126:
    r0, c0, r1, c1 = (int(x) for x in t)
    return BboxV126(r0=int(r0), c0=int(c0), r1=int(r1), c1=int(c1))


def bbox_nonzero_v126(g: GridV124, *, bg: int = 0) -> BboxV126:
    return bbox_from_tuple_v126(bbox_nonzero_v124(g, bg=int(bg)))


def bbox_by_color_v126(g: GridV124, *, color: int) -> BboxV126:
    m = mask_by_color_v124(g, color=_check_color(int(color)))
    return bbox_from_tuple_v126(bbox_nonzero_v124(m, bg=0))


def bbox_expand_v126(g: GridV124, bbox: BboxV126, *, delta: int) -> BboxV126:
    d = int(delta)
    if d < 0:
        raise ValueError("delta_negative")
    h, w = grid_shape_v124(g)
    r0, c0, r1, c1 = bbox.to_tuple()
    rr0 = max(0, int(r0) - d)
    cc0 = max(0, int(c0) - d)
    rr1 = min(int(h), int(r1) + d)
    cc1 = min(int(w), int(c1) + d)
    return BboxV126(r0=int(rr0), c0=int(cc0), r1=int(rr1), c1=int(cc1))


def crop_bbox_v126(g: GridV124, bbox: BboxV126) -> GridV124:
    r0, c0, r1, c1 = bbox.to_tuple()
    if r1 <= r0 or c1 <= c0:
        return tuple()
    return crop_v124(g, r0=int(r0), c0=int(c0), height=int(r1 - r0), width=int(c1 - c0))


def paint_mask_v126(g: GridV124, mask: GridV124, *, color: int) -> GridV124:
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


def fill_rect_v126(g: GridV124, bbox: BboxV126, *, color: int) -> GridV124:
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


def draw_rect_border_v126(g: GridV124, bbox: BboxV126, *, color: int, thickness: int = 1) -> GridV124:
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


def overlay_grid_v126(base: GridV124, top: GridV124, *, transparent: int = 0) -> GridV124:
    return overlay_v124(base, top, transparent=_check_color(int(transparent)))


def paste_v126(base: GridV124, patch: GridV124, *, top: int, left: int, transparent: int = 0) -> GridV124:
    tr = _check_color(int(transparent))
    h, w = grid_shape_v124(base)
    ph, pw = grid_shape_v124(patch)
    out: List[List[int]] = [[int(base[r][c]) for c in range(w)] for r in range(h)]
    rr0 = int(top)
    cc0 = int(left)
    for r in range(ph):
        for c in range(pw):
            rr = rr0 + r
            cc = cc0 + c
            if 0 <= rr < h and 0 <= cc < w:
                v = int(patch[r][c])
                if int(v) != int(tr):
                    out[rr][cc] = int(v)
    return grid_from_list_v124(out)


def flood_fill_v126(
    g: GridV124,
    seed_mask: GridV124,
    *,
    target_color: int,
    fill_color: int,
) -> GridV124:
    tg = _check_color(int(target_color))
    fc = _check_color(int(fill_color))
    h, w = grid_shape_v124(g)
    hm, wm = grid_shape_v124(seed_mask)
    if (h, w) != (hm, wm):
        raise ValueError("seed_mask_shape_mismatch")
    out: List[List[int]] = [[int(g[r][c]) for c in range(w)] for r in range(h)]
    q: deque[Tuple[int, int]] = deque()
    seen: set[Tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if int(seed_mask[r][c]) != 0:
                q.append((int(r), int(c)))
                seen.add((int(r), int(c)))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr = int(r + dr)
            cc = int(c + dc)
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                continue
            if (rr, cc) in seen:
                continue
            seen.add((rr, cc))
            if int(out[rr][cc]) == int(tg):
                out[rr][cc] = int(fc)
                q.append((rr, cc))
    return grid_from_list_v124(out)


def connected_components_v126(g: GridV124, *, color: Optional[int] = None) -> ObjectSetV126:
    comps = connected_components4_v124(g, color=int(color) if color is not None else None)
    objs: List[ObjectV126] = []
    for comp in comps:
        bbox = bbox_from_tuple_v126(comp.bbox)
        objs.append(ObjectV126(cells=tuple(comp.cells), color=int(comp.color), bbox=bbox))
    objs.sort(key=lambda o: (o.bbox.to_tuple(), int(o.color), o.cells))
    return ObjectSetV126(objects=tuple(objs))


SelectorV126 = Literal[
    "largest_area",
    "smallest_area",
    "leftmost",
    "rightmost",
    "topmost",
    "bottommost",
]


def _area_of_bbox(b: BboxV126) -> int:
    r0, c0, r1, c1 = b.to_tuple()
    return max(0, int(r1 - r0)) * max(0, int(c1 - c0))


def select_object_v126(obj_set: ObjectSetV126, *, selector: SelectorV126) -> ObjectV126:
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
    if sel == "rightmost":
        objs.sort(key=lambda o: (-int(o.bbox.c1), o.bbox.r0, o.bbox.to_tuple(), int(o.color), o.cells))
        return objs[0]
    if sel == "topmost":
        objs.sort(key=lambda o: (o.bbox.r0, o.bbox.c0, o.bbox.to_tuple(), int(o.color), o.cells))
        return objs[0]
    if sel == "bottommost":
        objs.sort(key=lambda o: (-int(o.bbox.r1), o.bbox.c0, o.bbox.to_tuple(), int(o.color), o.cells))
        return objs[0]
    raise ValueError(f"unknown_selector:{sel}")


def bbox_of_object_v126(o: ObjectV126) -> BboxV126:
    return o.bbox


@dataclass(frozen=True)
class OpDefV126:
    op_id: str
    input_types: Tuple[ValueTypeV126, ...]
    output_type: ValueTypeV126
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


OP_DEFS_V126: Dict[str, OpDefV126] = {
    "bbox_nonzero": OpDefV126(
        op_id="bbox_nonzero",
        input_types=("GRID",),
        output_type="BBOX",
        cost_bits=12,
        invariants=("bg in 0..9",),
    ),
    "bbox_by_color": OpDefV126(
        op_id="bbox_by_color",
        input_types=("GRID",),
        output_type="BBOX",
        cost_bits=14,
        invariants=("color in 0..9",),
    ),
    "bbox_expand": OpDefV126(
        op_id="bbox_expand",
        input_types=("GRID", "BBOX"),
        output_type="BBOX",
        cost_bits=10,
        invariants=("delta>=0", "clip to grid bounds",),
    ),
    "mask_by_color": OpDefV126(
        op_id="mask_by_color",
        input_types=("GRID",),
        output_type="MASK",
        cost_bits=14,
        invariants=("color in 0..9",),
    ),
    "crop_bbox": OpDefV126(
        op_id="crop_bbox",
        input_types=("GRID", "BBOX"),
        output_type="GRID",
        cost_bits=14,
        invariants=("bbox within bounds (empty ok -> empty grid)",),
    ),
    "fill_rect": OpDefV126(
        op_id="fill_rect",
        input_types=("GRID", "BBOX"),
        output_type="GRID",
        cost_bits=18,
        invariants=("color in 0..9",),
    ),
    "draw_rect_border": OpDefV126(
        op_id="draw_rect_border",
        input_types=("GRID", "BBOX"),
        output_type="GRID",
        cost_bits=20,
        invariants=("color in 0..9", "thickness>=1",),
    ),
    "paint_mask": OpDefV126(
        op_id="paint_mask",
        input_types=("GRID", "MASK"),
        output_type="GRID",
        cost_bits=18,
        invariants=("mask same shape as grid", "color in 0..9",),
    ),
    "flood_fill": OpDefV126(
        op_id="flood_fill",
        input_types=("GRID", "MASK"),
        output_type="GRID",
        cost_bits=28,
        invariants=("seed_mask same shape as grid", "4-neighborhood", "target_color in 0..9", "fill_color in 0..9",),
    ),
    "overlay": OpDefV126(
        op_id="overlay",
        input_types=("GRID", "GRID"),
        output_type="GRID",
        cost_bits=18,
        invariants=("same shape", "transparent in 0..9",),
    ),
    "paste": OpDefV126(
        op_id="paste",
        input_types=("GRID", "GRID"),
        output_type="GRID",
        cost_bits=24,
        invariants=("clip to bounds", "transparent in 0..9",),
    ),
    "connected_components": OpDefV126(
        op_id="connected_components",
        input_types=("GRID",),
        output_type="OBJECT_SET",
        cost_bits=30,
        invariants=("4-neighborhood",),
    ),
    "select_object": OpDefV126(
        op_id="select_object",
        input_types=("OBJECT_SET",),
        output_type="OBJECT",
        cost_bits=16,
        invariants=("selector in enum",),
    ),
    "bbox_of_object": OpDefV126(
        op_id="bbox_of_object",
        input_types=("OBJECT",),
        output_type="BBOX",
        cost_bits=8,
        invariants=(),
    ),
}


def apply_op_v126(*, op_id: str, inputs: Sequence[Any], args: Dict[str, Any]) -> Any:
    op = str(op_id)
    a = dict(args)

    if op == "bbox_nonzero":
        g = inputs[0]
        return bbox_nonzero_v126(g, bg=int(a.get("bg", 0)))
    if op == "bbox_by_color":
        g = inputs[0]
        return bbox_by_color_v126(g, color=int(a["color"]))
    if op == "bbox_expand":
        g = inputs[0]
        b = inputs[1]
        if not isinstance(b, BboxV126):
            raise ValueError("bbox_expand_requires_bbox")
        return bbox_expand_v126(g, b, delta=int(a.get("delta", 0)))
    if op == "mask_by_color":
        g = inputs[0]
        return mask_by_color_v124(g, color=int(a["color"]))
    if op == "crop_bbox":
        g = inputs[0]
        b = inputs[1]
        if not isinstance(b, BboxV126):
            raise ValueError("crop_bbox_requires_bbox")
        return crop_bbox_v126(g, b)
    if op == "fill_rect":
        g = inputs[0]
        b = inputs[1]
        if not isinstance(b, BboxV126):
            raise ValueError("fill_rect_requires_bbox")
        return fill_rect_v126(g, b, color=int(a["color"]))
    if op == "draw_rect_border":
        g = inputs[0]
        b = inputs[1]
        if not isinstance(b, BboxV126):
            raise ValueError("draw_rect_border_requires_bbox")
        return draw_rect_border_v126(g, b, color=int(a["color"]), thickness=int(a.get("thickness", 1)))
    if op == "paint_mask":
        g = inputs[0]
        m = inputs[1]
        return paint_mask_v126(g, m, color=int(a["color"]))
    if op == "flood_fill":
        g = inputs[0]
        m = inputs[1]
        return flood_fill_v126(g, m, target_color=int(a["target_color"]), fill_color=int(a["fill_color"]))
    if op == "overlay":
        base = inputs[0]
        top = inputs[1]
        return overlay_grid_v126(base, top, transparent=int(a.get("transparent", 0)))
    if op == "paste":
        base = inputs[0]
        patch = inputs[1]
        return paste_v126(
            base,
            patch,
            top=int(a.get("top", 0)),
            left=int(a.get("left", 0)),
            transparent=int(a.get("transparent", 0)),
        )
    if op == "connected_components":
        g = inputs[0]
        color = a.get("color")
        return connected_components_v126(g, color=int(color) if color is not None else None)
    if op == "select_object":
        os = inputs[0]
        if not isinstance(os, ObjectSetV126):
            raise ValueError("select_object_requires_object_set")
        return select_object_v126(os, selector=str(a["selector"]))  # type: ignore[arg-type]
    if op == "bbox_of_object":
        o = inputs[0]
        if not isinstance(o, ObjectV126):
            raise ValueError("bbox_of_object_requires_object")
        return bbox_of_object_v126(o)
    raise ValueError(f"unknown_op:{op}")


def value_to_canonical_json_v126(v: Any) -> Dict[str, Any]:
    # GridV124 is a tuple of tuple[int]; allow empty grid tuple().
    if isinstance(v, tuple) and (not v or isinstance(v[0], tuple)):
        return {"type": "GRID", "grid": [list(r) for r in v]}  # type: ignore[arg-type]
    if isinstance(v, BboxV126):
        return {"type": "BBOX", **v.to_dict()}
    if isinstance(v, ObjectV126):
        return {"type": "OBJECT", **v.to_dict()}
    if isinstance(v, ObjectSetV126):
        return {"type": "OBJECT_SET", "objects": [o.to_dict() for o in v.objects]}
    if isinstance(v, int):
        return {"type": "INT", "value": int(v)}
    raise ValueError("unsupported_value_type_v126")


def value_sig_v126(v: Any) -> str:
    body = {"schema_version": 126, "kind": "arc_value_v126", "value": value_to_canonical_json_v126(v)}
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))

