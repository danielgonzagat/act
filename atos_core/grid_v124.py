from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex

GridV124 = Tuple[Tuple[int, ...], ...]


def grid_from_list_v124(rows: Sequence[Sequence[int]]) -> GridV124:
    if not isinstance(rows, (list, tuple)):
        raise TypeError("grid_rows_not_sequence")
    out_rows: List[Tuple[int, ...]] = []
    w: Optional[int] = None
    for r in rows:
        if not isinstance(r, (list, tuple)):
            raise TypeError("grid_row_not_sequence")
        row: List[int] = []
        for x in r:
            if not isinstance(x, int):
                raise TypeError("grid_cell_not_int")
            if x < 0 or x > 9:
                raise ValueError("grid_cell_out_of_range")
            row.append(int(x))
        if w is None:
            w = int(len(row))
        elif int(len(row)) != int(w):
            raise ValueError("grid_non_rectangular")
        out_rows.append(tuple(row))
    if w is None:
        out_rows = []
    return tuple(out_rows)


def grid_to_list_v124(g: GridV124) -> List[List[int]]:
    return [list(r) for r in g]


def grid_shape_v124(g: GridV124) -> Tuple[int, int]:
    h = int(len(g))
    w = int(len(g[0])) if h > 0 else 0
    return h, w


def grid_hash_v124(g: GridV124) -> str:
    # Performance-critical: used heavily by ARC solver memoization.
    # Keep deterministic and collision-resistant, but avoid JSON encoding overhead.
    h, w = grid_shape_v124(g)
    buf = bytearray()
    buf.extend(b"grid_v124")
    # ARC grids are small (<256 in both dims), but clamp defensively.
    buf.append(int(h) & 0xFF)
    buf.append(int(w) & 0xFF)
    for row in g:
        for v in row:
            buf.append(int(v) & 0xFF)
    return sha256_hex(bytes(buf))


def grid_equal_v124(a: GridV124, b: GridV124) -> bool:
    return a == b


def rotate90_v124(g: GridV124) -> GridV124:
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return g
    out: List[Tuple[int, ...]] = []
    for c in range(w):
        out.append(tuple(g[h - 1 - r][c] for r in range(h)))
    return tuple(out)


def rotate180_v124(g: GridV124) -> GridV124:
    return rotate90_v124(rotate90_v124(g))


def rotate270_v124(g: GridV124) -> GridV124:
    return rotate90_v124(rotate180_v124(g))


def reflect_h_v124(g: GridV124) -> GridV124:
    return tuple(tuple(reversed(r)) for r in g)


def reflect_v_v124(g: GridV124) -> GridV124:
    return tuple(reversed(g))


def translate_v124(g: GridV124, *, dx: int, dy: int, pad: int = 0) -> GridV124:
    if pad < 0 or pad > 9:
        raise ValueError("pad_out_of_range")
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return g
    out: List[List[int]] = [[int(pad) for _ in range(w)] for _ in range(h)]
    for r in range(h):
        rr = r + int(dy)
        if rr < 0 or rr >= h:
            continue
        for c in range(w):
            cc = c + int(dx)
            if cc < 0 or cc >= w:
                continue
            out[rr][cc] = int(g[r][c])
    return grid_from_list_v124(out)


def pad_to_v124(g: GridV124, *, height: int, width: int, pad: int = 0) -> GridV124:
    if pad < 0 or pad > 9:
        raise ValueError("pad_out_of_range")
    h, w = grid_shape_v124(g)
    hh = int(height)
    ww = int(width)
    if hh < 0 or ww < 0:
        raise ValueError("pad_negative_shape")
    out: List[List[int]] = [[int(pad) for _ in range(ww)] for _ in range(hh)]
    for r in range(min(h, hh)):
        for c in range(min(w, ww)):
            out[r][c] = int(g[r][c])
    return grid_from_list_v124(out)


def crop_v124(g: GridV124, *, r0: int, c0: int, height: int, width: int) -> GridV124:
    h, w = grid_shape_v124(g)
    rr0 = int(r0)
    cc0 = int(c0)
    hh = int(height)
    ww = int(width)
    if hh < 0 or ww < 0:
        raise ValueError("crop_negative_shape")
    rr1 = rr0 + hh
    cc1 = cc0 + ww
    if rr0 < 0 or cc0 < 0 or rr1 > h or cc1 > w:
        raise ValueError("crop_out_of_bounds")
    return tuple(tuple(int(g[r][c]) for c in range(cc0, cc1)) for r in range(rr0, rr1))


def bbox_nonzero_v124(g: GridV124, *, bg: int = 0) -> Tuple[int, int, int, int]:
    if bg < 0 or bg > 9:
        raise ValueError("bg_out_of_range")
    h, w = grid_shape_v124(g)
    rmin = h
    cmin = w
    rmax = -1
    cmax = -1
    for r in range(h):
        for c in range(w):
            if int(g[r][c]) != int(bg):
                if r < rmin:
                    rmin = r
                if c < cmin:
                    cmin = c
                if r > rmax:
                    rmax = r
                if c > cmax:
                    cmax = c
    if rmax < 0:
        return 0, 0, 0, 0
    return int(rmin), int(cmin), int(rmax + 1), int(cmax + 1)


def crop_to_bbox_nonzero_v124(g: GridV124, *, bg: int = 0) -> GridV124:
    r0, c0, r1, c1 = bbox_nonzero_v124(g, bg=int(bg))
    if r1 <= r0 or c1 <= c0:
        return tuple()
    return crop_v124(g, r0=int(r0), c0=int(c0), height=int(r1 - r0), width=int(c1 - c0))


def unique_colors_v124(g: GridV124) -> List[int]:
    colors = {int(x) for row in g for x in row}
    return sorted(colors)


def count_colors_v124(g: GridV124) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for row in g:
        for x in row:
            out[int(x)] = int(out.get(int(x), 0)) + 1
    return {int(k): int(out[k]) for k in sorted(out.keys())}


def mask_by_color_v124(g: GridV124, *, color: int) -> GridV124:
    cc = int(color)
    if cc < 0 or cc > 9:
        raise ValueError("color_out_of_range")
    return tuple(tuple(1 if int(x) == cc else 0 for x in row) for row in g)


def overlay_v124(base: GridV124, top: GridV124, *, transparent: int = 0) -> GridV124:
    if transparent < 0 or transparent > 9:
        raise ValueError("transparent_out_of_range")
    hb, wb = grid_shape_v124(base)
    ht, wt = grid_shape_v124(top)
    if hb != ht or wb != wt:
        raise ValueError("overlay_shape_mismatch")
    out: List[Tuple[int, ...]] = []
    for r in range(hb):
        out.append(tuple(int(top[r][c]) if int(top[r][c]) != int(transparent) else int(base[r][c]) for c in range(wb)))
    return tuple(out)


@dataclass(frozen=True)
class ComponentV124:
    cells: Tuple[Tuple[int, int], ...]
    color: int
    bbox: Tuple[int, int, int, int]


def connected_components4_v124(g: GridV124, *, color: Optional[int] = None) -> List[ComponentV124]:
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return []
    want_color: Optional[int] = int(color) if color is not None else None
    if want_color is not None and (want_color < 0 or want_color > 9):
        raise ValueError("color_out_of_range")
    seen = [[False for _ in range(w)] for _ in range(h)]
    comps: List[ComponentV124] = []
    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0][c0]:
                continue
            col0 = int(g[r0][c0])
            if want_color is not None and col0 != int(want_color):
                seen[r0][c0] = True
                continue
            q: deque[Tuple[int, int]] = deque()
            q.append((r0, c0))
            seen[r0][c0] = True
            cells: List[Tuple[int, int]] = []
            rmin = r0
            cmin = c0
            rmax = r0
            cmax = c0
            while q:
                r, c = q.popleft()
                if int(g[r][c]) != col0:
                    continue
                cells.append((int(r), int(c)))
                if r < rmin:
                    rmin = r
                if c < cmin:
                    cmin = c
                if r > rmax:
                    rmax = r
                if c > cmax:
                    cmax = c
                for dr, dc in neigh:
                    rr = r + dr
                    cc = c + dc
                    if rr < 0 or rr >= h or cc < 0 or cc >= w:
                        continue
                    if seen[rr][cc]:
                        continue
                    if int(g[rr][cc]) != col0:
                        seen[rr][cc] = True
                        continue
                    seen[rr][cc] = True
                    q.append((rr, cc))
            if cells:
                cells_sorted = tuple(sorted(cells))
                bbox = (int(rmin), int(cmin), int(rmax + 1), int(cmax + 1))
                comps.append(ComponentV124(cells=cells_sorted, color=int(col0), bbox=bbox))
    # Deterministic ordering: by bbox, then color, then cells.
    comps.sort(key=lambda c: (c.bbox, int(c.color), c.cells))
    return comps


def detect_symmetry_v124(g: GridV124) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["reflect_h"] = bool(reflect_h_v124(g) == g)
    out["reflect_v"] = bool(reflect_v_v124(g) == g)
    out["rotate180"] = bool(rotate180_v124(g) == g)
    return out
