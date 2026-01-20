from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .grid_v124 import GridV124, bbox_nonzero_v124, grid_from_list_v124, grid_shape_v124, unique_colors_v124

ARC_DELTA_SCHEMA_VERSION_V129 = 129


def _mask_where_v129(*, shape: Tuple[int, int], coords: Sequence[Tuple[int, int]]) -> GridV124:
    h, w = int(shape[0]), int(shape[1])
    rows: List[List[int]] = [[0 for _ in range(w)] for _ in range(h)]
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < h and 0 <= cc < w:
            rows[rr][cc] = 1
    return grid_from_list_v124(rows)


@dataclass(frozen=True)
class DeltaEvidenceV129:
    shape_in: Tuple[int, int]
    shape_out: Tuple[int, int]
    shape_changed: bool

    # When shape_out is an integer multiple of shape_in.
    ratio_h: Optional[int]
    ratio_w: Optional[int]

    changed_mask: GridV124
    changed_bbox: Tuple[int, int, int, int]
    changed_cells: int
    total_cells: int

    palette_in: Tuple[int, ...]
    palette_out: Tuple[int, ...]
    palette_added: Tuple[int, ...]
    palette_removed: Tuple[int, ...]

    out_colors_in_changed: Tuple[int, ...]
    in_colors_in_changed: Tuple[int, ...]
    out_color_counts_in_changed: Tuple[Tuple[int, int], ...]
    in_color_counts_in_changed: Tuple[Tuple[int, int], ...]

    # Decomposed evidence by color for changed cells.
    # Each entry: (color, mask_grid, bbox_tuple, cells_count)
    changed_out_color_masks: Tuple[Tuple[int, GridV124, Tuple[int, int, int, int], int], ...]
    changed_in_color_masks: Tuple[Tuple[int, GridV124, Tuple[int, int, int, int], int], ...]

    def to_dict(self) -> Dict[str, Any]:
        hi, wi = self.shape_in
        ho, wo = self.shape_out
        r0, c0, r1, c1 = self.changed_bbox
        out_masks: List[Dict[str, Any]] = []
        for color, mask, bb, cnt in self.changed_out_color_masks:
            rr0, cc0, rr1, cc1 = bb
            out_masks.append(
                {
                    "color": int(color),
                    "cells": int(cnt),
                    "bbox": {"r0": int(rr0), "c0": int(cc0), "r1": int(rr1), "c1": int(cc1)},
                    "mask": [list(r) for r in mask],
                }
            )
        in_masks: List[Dict[str, Any]] = []
        for color, mask, bb, cnt in self.changed_in_color_masks:
            rr0, cc0, rr1, cc1 = bb
            in_masks.append(
                {
                    "color": int(color),
                    "cells": int(cnt),
                    "bbox": {"r0": int(rr0), "c0": int(cc0), "r1": int(rr1), "c1": int(cc1)},
                    "mask": [list(r) for r in mask],
                }
            )
        return {
            "schema_version": int(ARC_DELTA_SCHEMA_VERSION_V129),
            "kind": "arc_delta_v129",
            "shape_in": {"h": int(hi), "w": int(wi)},
            "shape_out": {"h": int(ho), "w": int(wo)},
            "shape_changed": bool(self.shape_changed),
            "ratio_h": int(self.ratio_h) if self.ratio_h is not None else None,
            "ratio_w": int(self.ratio_w) if self.ratio_w is not None else None,
            "changed_mask": [list(r) for r in self.changed_mask],
            "changed_bbox": {"r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1)},
            "changed_cells": int(self.changed_cells),
            "total_cells": int(self.total_cells),
            "palette_in": [int(c) for c in self.palette_in],
            "palette_out": [int(c) for c in self.palette_out],
            "palette_added": [int(c) for c in self.palette_added],
            "palette_removed": [int(c) for c in self.palette_removed],
            "out_colors_in_changed": [int(c) for c in self.out_colors_in_changed],
            "in_colors_in_changed": [int(c) for c in self.in_colors_in_changed],
            "out_color_counts_in_changed": [{"color": int(c), "cells": int(n)} for c, n in self.out_color_counts_in_changed],
            "in_color_counts_in_changed": [{"color": int(c), "cells": int(n)} for c, n in self.in_color_counts_in_changed],
            "changed_out_color_masks": list(out_masks),
            "changed_in_color_masks": list(in_masks),
        }

    def delta_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


def _changed_mask_v129(inp: GridV124, out: GridV124) -> Tuple[GridV124, int]:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if (hi, wi) != (ho, wo) or hi == 0 or wi == 0:
        return tuple(), 0
    rows: List[List[int]] = []
    changed = 0
    for r in range(hi):
        row: List[int] = []
        for c in range(wi):
            d = 1 if int(inp[r][c]) != int(out[r][c]) else 0
            if d:
                changed += 1
            row.append(int(d))
        rows.append(row)
    return grid_from_list_v124(rows), int(changed)


def compute_delta_v129(inp: GridV124, out: GridV124) -> DeltaEvidenceV129:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    shape_changed = bool((hi, wi) != (ho, wo))

    ratio_h: Optional[int] = None
    ratio_w: Optional[int] = None
    if hi > 0 and wi > 0 and ho > 0 and wo > 0:
        if ho % hi == 0 and wo % wi == 0:
            ratio_h = int(ho // hi)
            ratio_w = int(wo // wi)

    mask, changed_cells = _changed_mask_v129(inp, out)
    changed_bbox = bbox_nonzero_v124(mask, bg=0) if mask else (0, 0, 0, 0)
    total_cells = int(hi * wi) if not shape_changed else 0

    pal_in = tuple(int(c) for c in unique_colors_v124(inp))
    pal_out = tuple(int(c) for c in unique_colors_v124(out))
    pal_added = tuple(int(c) for c in sorted(set(pal_out) - set(pal_in)))
    pal_removed = tuple(int(c) for c in sorted(set(pal_in) - set(pal_out)))

    out_changed: List[int] = []
    in_changed: List[int] = []
    out_coords_by_color: Dict[int, List[Tuple[int, int]]] = {}
    in_coords_by_color: Dict[int, List[Tuple[int, int]]] = {}
    if mask and not shape_changed:
        for r in range(hi):
            for c in range(wi):
                if int(mask[r][c]) == 0:
                    continue
                outc = int(out[r][c])
                inc = int(inp[r][c])
                out_changed.append(int(outc))
                in_changed.append(int(inc))
                out_coords_by_color.setdefault(int(outc), []).append((int(r), int(c)))
                in_coords_by_color.setdefault(int(inc), []).append((int(r), int(c)))

    out_colors_in_changed = tuple(int(c) for c in sorted(set(out_changed)))
    in_colors_in_changed = tuple(int(c) for c in sorted(set(in_changed)))

    out_counts: List[Tuple[int, int]] = []
    for c in sorted(out_coords_by_color.keys()):
        out_counts.append((int(c), int(len(out_coords_by_color[int(c)]))))
    in_counts: List[Tuple[int, int]] = []
    for c in sorted(in_coords_by_color.keys()):
        in_counts.append((int(c), int(len(in_coords_by_color[int(c)]))))

    out_masks: List[Tuple[int, GridV124, Tuple[int, int, int, int], int]] = []
    for c in sorted(out_coords_by_color.keys()):
        coords = out_coords_by_color[int(c)]
        m = _mask_where_v129(shape=(hi, wi), coords=coords)
        bb = bbox_nonzero_v124(m, bg=0) if m else (0, 0, 0, 0)
        out_masks.append((int(c), m, tuple(int(x) for x in bb), int(len(coords))))

    in_masks: List[Tuple[int, GridV124, Tuple[int, int, int, int], int]] = []
    for c in sorted(in_coords_by_color.keys()):
        coords = in_coords_by_color[int(c)]
        m = _mask_where_v129(shape=(hi, wi), coords=coords)
        bb = bbox_nonzero_v124(m, bg=0) if m else (0, 0, 0, 0)
        in_masks.append((int(c), m, tuple(int(x) for x in bb), int(len(coords))))

    return DeltaEvidenceV129(
        shape_in=(int(hi), int(wi)),
        shape_out=(int(ho), int(wo)),
        shape_changed=bool(shape_changed),
        ratio_h=int(ratio_h) if ratio_h is not None else None,
        ratio_w=int(ratio_w) if ratio_w is not None else None,
        changed_mask=mask,
        changed_bbox=tuple(int(x) for x in changed_bbox),
        changed_cells=int(changed_cells),
        total_cells=int(total_cells),
        palette_in=tuple(int(c) for c in pal_in),
        palette_out=tuple(int(c) for c in pal_out),
        palette_added=tuple(int(c) for c in pal_added),
        palette_removed=tuple(int(c) for c in pal_removed),
        out_colors_in_changed=out_colors_in_changed,
        in_colors_in_changed=in_colors_in_changed,
        out_color_counts_in_changed=tuple(out_counts),
        in_color_counts_in_changed=tuple(in_counts),
        changed_out_color_masks=tuple(out_masks),
        changed_in_color_masks=tuple(in_masks),
    )

