from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .grid_v124 import GridV124, connected_components4_v124, grid_shape_v124, unique_colors_v124

ARC_OBJECTS_SCHEMA_VERSION_V132 = 132


def _check_color_v132(c: int) -> int:
    cc = int(c)
    if cc < 0 or cc > 9:
        raise ValueError("color_out_of_range")
    return int(cc)


@dataclass(frozen=True)
class BBoxV132:
    # Half-open box: [r0,r1) x [c0,c1)
    r0: int
    c0: int
    r1: int
    c1: int

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return int(self.r0), int(self.c0), int(self.r1), int(self.c1)

    def to_dict(self) -> Dict[str, Any]:
        r0, c0, r1, c1 = self.to_tuple()
        return {"r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1)}

    def bbox_sig(self) -> str:
        body = {"schema_version": int(ARC_OBJECTS_SCHEMA_VERSION_V132), "kind": "bbox_v132", **self.to_dict()}
        return sha256_hex(canonical_json_dumps(body).encode("utf-8"))

    def height(self) -> int:
        r0, _, r1, _ = self.to_tuple()
        return max(0, int(r1 - r0))

    def width(self) -> int:
        _, c0, _, c1 = self.to_tuple()
        return max(0, int(c1 - c0))

    def area(self) -> int:
        return int(self.height() * self.width())


def bbox_union_v132(boxes: Iterable[BBoxV132]) -> Optional[BBoxV132]:
    r0: Optional[int] = None
    c0: Optional[int] = None
    r1: Optional[int] = None
    c1: Optional[int] = None
    any_box = False
    for b in boxes:
        any_box = True
        br0, bc0, br1, bc1 = b.to_tuple()
        if r0 is None:
            r0, c0, r1, c1 = int(br0), int(bc0), int(br1), int(bc1)
        else:
            r0 = min(int(r0), int(br0))
            c0 = min(int(c0), int(bc0))
            r1 = max(int(r1), int(br1))
            c1 = max(int(c1), int(bc1))
    if not any_box:
        return None
    return BBoxV132(r0=int(r0), c0=int(c0), r1=int(r1), c1=int(c1))


def delta_bbox_v132(inp: GridV124, out: GridV124) -> Optional[BBoxV132]:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if (hi, wi) != (ho, wo):
        return None
    if hi == 0 or wi == 0:
        return None
    rmin = hi
    cmin = wi
    rmax = -1
    cmax = -1
    for r in range(hi):
        for c in range(wi):
            if int(inp[r][c]) != int(out[r][c]):
                if r < rmin:
                    rmin = r
                if c < cmin:
                    cmin = c
                if r > rmax:
                    rmax = r
                if c > cmax:
                    cmax = c
    if rmax < 0:
        return None
    return BBoxV132(r0=int(rmin), c0=int(cmin), r1=int(rmax + 1), c1=int(cmax + 1))


@dataclass(frozen=True)
class MaskV132:
    # Deterministic sparse mask (shape is implicit in task; cells are sorted).
    cells: Tuple[Tuple[int, int], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"cells": [{"r": int(r), "c": int(c)} for r, c in self.cells]}

    def mask_sig(self) -> str:
        body = {"schema_version": int(ARC_OBJECTS_SCHEMA_VERSION_V132), "kind": "mask_v132", **self.to_dict()}
        return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


@dataclass(frozen=True)
class ObjectV132:
    color: int
    cells: Tuple[Tuple[int, int], ...]
    bbox: BBoxV132
    area: int
    width: int
    height: int
    bbox_center_r2: int
    bbox_center_c2: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "color": int(self.color),
            "cells": [{"r": int(r), "c": int(c)} for r, c in self.cells],
            "bbox": self.bbox.to_dict(),
            "area": int(self.area),
            "width": int(self.width),
            "height": int(self.height),
            "bbox_center_r2": int(self.bbox_center_r2),
            "bbox_center_c2": int(self.bbox_center_c2),
        }

    def object_sig(self) -> str:
        body = {"schema_version": int(ARC_OBJECTS_SCHEMA_VERSION_V132), "kind": "object_v132", **self.to_dict()}
        return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


@dataclass(frozen=True)
class ObjectSetV132:
    objects: Tuple[ObjectV132, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"objects": [o.to_dict() for o in self.objects]}

    def set_sig(self) -> str:
        body = {"schema_version": int(ARC_OBJECTS_SCHEMA_VERSION_V132), "kind": "object_set_v132", **self.to_dict()}
        return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _bbox_center_r2_v132(b: BBoxV132) -> int:
    r0, _, r1, _ = b.to_tuple()
    if r1 <= r0:
        return 0
    return int(r0 + r1 - 1)


def _bbox_center_c2_v132(b: BBoxV132) -> int:
    _, c0, _, c1 = b.to_tuple()
    if c1 <= c0:
        return 0
    return int(c0 + c1 - 1)


def connected_components4_v132(
    g: GridV124,
    *,
    colors: Optional[Sequence[int]] = None,
    bg: Optional[int] = None,
) -> ObjectSetV132:
    """
    Deterministic 4-neigh monochrome objects per color (no multi-color grouping).
    If bg is provided, that color is ignored.
    If colors is provided, only those colors are considered.
    """
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return ObjectSetV132(objects=tuple())

    bgc: Optional[int] = _check_color_v132(int(bg)) if bg is not None else None
    if colors is None:
        cols = unique_colors_v124(g)
    else:
        cols = [int(_check_color_v132(int(c))) for c in colors]
    cols_s = sorted(set(int(c) for c in cols))
    if bgc is not None:
        cols_s = [int(c) for c in cols_s if int(c) != int(bgc)]

    objs: List[ObjectV132] = []
    for c in cols_s:
        comps = connected_components4_v124(g, color=int(c))
        for comp in comps:
            cells = tuple(sorted(((int(r), int(cc)) for r, cc in comp.cells), key=lambda rc: (int(rc[0]), int(rc[1]))))
            r0, c0, r1, c1 = (int(x) for x in comp.bbox)
            bbox = BBoxV132(r0=int(r0), c0=int(c0), r1=int(r1), c1=int(c1))
            objs.append(
                ObjectV132(
                    color=int(c),
                    cells=cells,
                    bbox=bbox,
                    area=int(len(cells)),
                    width=int(bbox.width()),
                    height=int(bbox.height()),
                    bbox_center_r2=int(_bbox_center_r2_v132(bbox)),
                    bbox_center_c2=int(_bbox_center_c2_v132(bbox)),
                )
            )

    # Canonical object ordering (stable + documented):
    # area desc, bbox, color, cells
    objs.sort(key=lambda o: (-int(o.area), o.bbox.to_tuple(), int(o.color), o.cells))
    return ObjectSetV132(objects=tuple(objs))


def object_set_summary_v132(oset: ObjectSetV132, *, top_k: int = 5) -> Dict[str, Any]:
    objs = list(oset.objects)
    top = objs[: int(top_k)]
    return {
        "count": int(len(objs)),
        "top": [
            {"color": int(o.color), "area": int(o.area), "bbox": o.bbox.to_dict()}
            for o in top
        ],
    }
