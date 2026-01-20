from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .grid_v124 import GridV124, bbox_nonzero_v124, grid_from_list_v124, grid_shape_v124, unique_colors_v124

ARC_DELTA_SCHEMA_VERSION_V127 = 127


@dataclass(frozen=True)
class DeltaEvidenceV127:
    shape_in: Tuple[int, int]
    shape_out: Tuple[int, int]
    shape_changed: bool

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

    def to_dict(self) -> Dict[str, Any]:
        hi, wi = self.shape_in
        ho, wo = self.shape_out
        r0, c0, r1, c1 = self.changed_bbox
        return {
            "schema_version": int(ARC_DELTA_SCHEMA_VERSION_V127),
            "kind": "arc_delta_v127",
            "shape_in": {"h": int(hi), "w": int(wi)},
            "shape_out": {"h": int(ho), "w": int(wo)},
            "shape_changed": bool(self.shape_changed),
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
        }

    def delta_sig(self) -> str:
        body = self.to_dict()
        return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _changed_mask_v127(inp: GridV124, out: GridV124) -> Tuple[GridV124, int]:
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


def compute_delta_v127(inp: GridV124, out: GridV124) -> DeltaEvidenceV127:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    shape_changed = bool((hi, wi) != (ho, wo))

    mask, changed_cells = _changed_mask_v127(inp, out)
    changed_bbox = bbox_nonzero_v124(mask, bg=0) if mask else (0, 0, 0, 0)
    total_cells = int(hi * wi) if not shape_changed else 0

    pal_in = tuple(int(c) for c in unique_colors_v124(inp))
    pal_out = tuple(int(c) for c in unique_colors_v124(out))
    pal_added = tuple(int(c) for c in sorted(set(pal_out) - set(pal_in)))
    pal_removed = tuple(int(c) for c in sorted(set(pal_in) - set(pal_out)))

    out_changed: List[int] = []
    in_changed: List[int] = []
    if mask and not shape_changed:
        for r in range(hi):
            for c in range(wi):
                if int(mask[r][c]) != 0:
                    out_changed.append(int(out[r][c]))
                    in_changed.append(int(inp[r][c]))
    out_colors_in_changed = tuple(int(c) for c in sorted(set(out_changed)))
    in_colors_in_changed = tuple(int(c) for c in sorted(set(in_changed)))

    return DeltaEvidenceV127(
        shape_in=(int(hi), int(wi)),
        shape_out=(int(ho), int(wo)),
        shape_changed=bool(shape_changed),
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
    )

