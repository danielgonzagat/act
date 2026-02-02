"""
cognitive_ops_selectors_v161.py - Seletores Semânticos

Esta expansão implementa todos os seletores semânticos autorizados para
alcançar ≥90% de acerto nos benchmarks ARC-AGI-1 e ARC-AGI-2.

Schema version: 161
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import math

GridV124 = List[List[int]]
COGNITIVE_OPS_SELECTORS_SCHEMA_VERSION_V161 = 161


def _grid_shape(g: GridV124) -> Tuple[int, int]:
    if not g:
        return (0, 0)
    return (len(g), len(g[0]) if g[0] else 0)


def _copy_grid(g: GridV124) -> GridV124:
    return [[int(c) for c in row] for row in g]


def _new_grid(rows: int, cols: int, fill: int = 0) -> GridV124:
    return [[fill for _ in range(cols)] for _ in range(rows)]


def _find_objects(g: GridV124, *, bg: int = 0) -> List[Set[Tuple[int, int]]]:
    h, w = _grid_shape(g)
    if h == 0 or w == 0:
        return []
    visited: Set[Tuple[int, int]] = set()
    objects: List[Set[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in visited or int(g[r][c]) == bg:
                continue
            component: Set[Tuple[int, int]] = set()
            queue = [(r, c)]
            visited.add((r, c))
            while queue:
                cr, cc = queue.pop(0)
                component.add((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and int(g[nr][nc]) != bg:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            objects.append(component)
    return objects


def _object_bbox(obj: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    if not obj:
        return (0, 0, 0, 0)
    rs = [r for r, c in obj]
    cs = [c for r, c in obj]
    return (min(rs), min(cs), max(rs), max(cs))


def _object_center(obj: Set[Tuple[int, int]]) -> Tuple[float, float]:
    if not obj:
        return (0.0, 0.0)
    return (sum(r for r, c in obj) / len(obj), sum(c for r, c in obj) / len(obj))


def _object_colors(g: GridV124, obj: Set[Tuple[int, int]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for r, c in obj:
        color = int(g[r][c])
        counts[color] = counts.get(color, 0) + 1
    return counts


def _object_dominant_color(g: GridV124, obj: Set[Tuple[int, int]]) -> int:
    counts = _object_colors(g, obj)
    return max(counts, key=lambda k: counts[k]) if counts else 0


def _extract_object_grid(g: GridV124, obj: Set[Tuple[int, int]], *, bg: int = 0) -> GridV124:
    if not obj:
        return [[]]
    min_r, min_c, max_r, max_c = _object_bbox(obj)
    h, w = max_r - min_r + 1, max_c - min_c + 1
    out = _new_grid(h, w, bg)
    for r, c in obj:
        out[r - min_r][c - min_c] = int(g[r][c])
    return out


def _object_signature(g: GridV124, obj: Set[Tuple[int, int]], *, bg: int = 0) -> str:
    return str(_extract_object_grid(g, obj, bg=bg))


@dataclass
class Selection:
    mask: GridV124
    objects: List[Set[Tuple[int, int]]]
    indices: List[int]
    
    def to_grid(self, g: GridV124, *, bg: int = 0) -> GridV124:
        h, w = _grid_shape(g)
        out = _new_grid(h, w, bg)
        for r in range(h):
            for c in range(w):
                if int(self.mask[r][c]) != 0:
                    out[r][c] = int(g[r][c])
        return out


def _create_selection_from_objects(g: GridV124, objects: List[Set[Tuple[int, int]]], selected_indices: List[int], *, bg: int = 0) -> Selection:
    h, w = _grid_shape(g)
    mask = _new_grid(h, w, bg)
    selected_objects = []
    for idx in selected_indices:
        if 0 <= idx < len(objects):
            obj = objects[idx]
            selected_objects.append(obj)
            for r, c in obj:
                mask[r][c] = 1
    return Selection(mask=mask, objects=selected_objects, indices=selected_indices)


def _create_selection_from_cells(g: GridV124, cells: Set[Tuple[int, int]], *, bg: int = 0) -> Selection:
    h, w = _grid_shape(g)
    mask = _new_grid(h, w, bg)
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            mask[r][c] = 1
    return Selection(mask=mask, objects=[cells] if cells else [], indices=[])


# SIZE-BASED SELECTORS

def select_largest_object(g: GridV124, *, bg: int = 0) -> Selection:
    objects = _find_objects(g, bg=bg)
    if not objects:
        h, w = _grid_shape(g)
        return Selection(mask=_new_grid(h, w, bg), objects=[], indices=[])
    largest_idx = max(range(len(objects)), key=lambda i: len(objects[i]))
    return _create_selection_from_objects(g, objects, [largest_idx], bg=bg)


def select_smallest_object(g: GridV124, *, bg: int = 0) -> Selection:
    objects = _find_objects(g, bg=bg)
    if not objects:
        h, w = _grid_shape(g)
        return Selection(mask=_new_grid(h, w, bg), objects=[], indices=[])
    smallest_idx = min(range(len(objects)), key=lambda i: len(objects[i]))
    return _create_selection_from_objects(g, objects, [smallest_idx], bg=bg)


def select_shape_by_size(g: GridV124, *, size: int, tolerance: int = 0, bg: int = 0) -> Selection:
    objects = _find_objects(g, bg=bg)
    selected = [i for i, obj in enumerate(objects) if abs(len(obj) - size) <= tolerance]
    return _create_selection_from_objects(g, objects, selected, bg=bg)


# COLOR-BASED SELECTORS

def select_by_color(g: GridV124, *, color: int, bg: int = 0) -> Selection:
    h, w = _grid_shape(g)
    cells: Set[Tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if int(g[r][c]) == color:
                cells.add((r, c))
    return _create_selection_from_cells(g, cells, bg=bg)


def select_most_frequent_color(g: GridV124, *, bg: int = 0) -> Selection:
    h, w = _grid_shape(g)
    counts: Dict[int, int] = {}
    for r in range(h):
        for c in range(w):
            color = int(g[r][c])
            if color != bg:
                counts[color] = counts.get(color, 0) + 1
    if not counts:
        return Selection(mask=_new_grid(h, w, bg), objects=[], indices=[])
    most_frequent = max(counts, key=lambda k: counts[k])
    return select_by_color(g, color=most_frequent, bg=bg)


def select_least_frequent_color(g: GridV124, *, bg: int = 0) -> Selection:
    h, w = _grid_shape(g)
    counts: Dict[int, int] = {}
    for r in range(h):
        for c in range(w):
            color = int(g[r][c])
            if color != bg:
                counts[color] = counts.get(color, 0) + 1
    if not counts:
        return Selection(mask=_new_grid(h, w, bg), objects=[], indices=[])
    least_frequent = min(counts, key=lambda k: counts[k])
    return select_by_color(g, color=least_frequent, bg=bg)


def select_outlier_color(g: GridV124, *, bg: int = 0) -> Selection:
    objects = _find_objects(g, bg=bg)
    if not objects:
        h, w = _grid_shape(g)
        return Selection(mask=_new_grid(h, w, bg), objects=[], indices=[])
    obj_colors = [_object_dominant_color(g, obj) for obj in objects]
    color_counts = Counter(obj_colors)
    selected = [i for i, color in enumerate(obj_colors) if color_counts[color] == 1]
    return _create_selection_from_objects(g, objects, selected, bg=bg)


# POSITION-BASED SELECTORS

def select_border_objects(g: GridV124, *, bg: int = 0) -> Selection:
    h, w = _grid_shape(g)
    objects = _find_objects(g, bg=bg)
    selected = [i for i, obj in enumerate(objects) if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in obj)]
    return _create_selection_from_objects(g, objects, selected, bg=bg)


def select_center_object(g: GridV124, *, bg: int = 0) -> Selection:
    h, w = _grid_shape(g)
    objects = _find_objects(g, bg=bg)
    if not objects:
        return Selection(mask=_new_grid(h, w, bg), objects=[], indices=[])
    center = (h / 2, w / 2)
    def dist(obj):
        cr, cc = _object_center(obj)
        return math.sqrt((cr - center[0]) ** 2 + (cc - center[1]) ** 2)
    closest_idx = min(range(len(objects)), key=lambda i: dist(objects[i]))
    return _create_selection_from_objects(g, objects, [closest_idx], bg=bg)


def select_corner_patterns(g: GridV124, *, corner: str = "all", size: int = 3, bg: int = 0) -> Selection:
    h, w = _grid_shape(g)
    cells: Set[Tuple[int, int]] = set()
    corners = []
    if corner in ["top_left", "all"]:
        corners.append((0, 0))
    if corner in ["top_right", "all"]:
        corners.append((0, w - size))
    if corner in ["bottom_left", "all"]:
        corners.append((h - size, 0))
    if corner in ["bottom_right", "all"]:
        corners.append((h - size, w - size))
    for cr, cc in corners:
        for r in range(max(0, cr), min(h, cr + size)):
            for c in range(max(0, cc), min(w, cc + size)):
                if int(g[r][c]) != bg:
                    cells.add((r, c))
    return _create_selection_from_cells(g, cells, bg=bg)


# SHAPE-BASED SELECTORS

def select_diagonal_line(g: GridV124, *, bg: int = 0, min_length: int = 3) -> Selection:
    objects = _find_objects(g, bg=bg)
    selected = []
    for i, obj in enumerate(objects):
        if len(obj) < min_length:
            continue
        cells = sorted(obj)
        if len(cells) >= 2:
            dr = cells[1][0] - cells[0][0]
            dc = cells[1][1] - cells[0][1]
            if abs(dr) == 1 and abs(dc) == 1:
                is_diagonal = all((cells[0][0] + j * dr, cells[0][1] + j * dc) in obj for j in range(len(cells)))
                if is_diagonal:
                    selected.append(i)
    return _create_selection_from_objects(g, objects, selected, bg=bg)


def select_straight_line(g: GridV124, *, bg: int = 0, min_length: int = 3) -> Selection:
    objects = _find_objects(g, bg=bg)
    selected = []
    for i, obj in enumerate(objects):
        if len(obj) < min_length:
            continue
        min_r, min_c, max_r, max_c = _object_bbox(obj)
        is_horizontal = (max_r == min_r) and (max_c - min_c + 1 == len(obj))
        is_vertical = (max_c == min_c) and (max_r - min_r + 1 == len(obj))
        if is_horizontal or is_vertical:
            selected.append(i)
    return _create_selection_from_objects(g, objects, selected, bg=bg)


def select_closed_shape(g: GridV124, *, bg: int = 0) -> Selection:
    objects = _find_objects(g, bg=bg)
    selected = []
    for i, obj in enumerate(objects):
        if len(obj) < 4:
            continue
        min_r, min_c, max_r, max_c = _object_bbox(obj)
        bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        obj_density = len(obj) / bbox_area if bbox_area > 0 else 1
        if obj_density < 0.7 and len(obj) >= 8:
            selected.append(i)
    return _create_selection_from_objects(g, objects, selected, bg=bg)


def select_hole(g: GridV124, *, bg: int = 0) -> Selection:
    h, w = _grid_shape(g)
    inverted = _new_grid(h, w, 0)
    for r in range(h):
        for c in range(w):
            if int(g[r][c]) == bg:
                inverted[r][c] = 1
    bg_components = _find_objects(inverted, bg=0)
    selected_cells: Set[Tuple[int, int]] = set()
    for comp in bg_components:
        if not any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
            selected_cells.update(comp)
    return _create_selection_from_cells(g, selected_cells, bg=bg)


def select_by_shape(g: GridV124, *, template: GridV124, bg: int = 0) -> Selection:
    objects = _find_objects(g, bg=bg)
    template_sig = str(template)
    selected = [i for i, obj in enumerate(objects) if str(_extract_object_grid(g, obj, bg=bg)) == template_sig]
    return _create_selection_from_objects(g, objects, selected, bg=bg)


# UNIQUENESS SELECTORS

def select_unique_object(g: GridV124, *, bg: int = 0) -> Selection:
    objects = _find_objects(g, bg=bg)
    if not objects:
        h, w = _grid_shape(g)
        return Selection(mask=_new_grid(h, w, bg), objects=[], indices=[])
    signatures = [_object_signature(g, obj, bg=bg) for obj in objects]
    sig_counts = Counter(signatures)
    selected = [i for i, sig in enumerate(signatures) if sig_counts[sig] == 1]
    if not selected:
        sizes = [len(obj) for obj in objects]
        size_counts = Counter(sizes)
        selected = [i for i, size in enumerate(sizes) if size_counts[size] == 1]
    return _create_selection_from_objects(g, objects, selected, bg=bg)


def select_duplicate_shapes(g: GridV124, *, bg: int = 0) -> Selection:
    objects = _find_objects(g, bg=bg)
    signatures = [_object_signature(g, obj, bg=bg) for obj in objects]
    sig_counts = Counter(signatures)
    selected = [i for i, sig in enumerate(signatures) if sig_counts[sig] > 1]
    return _create_selection_from_objects(g, objects, selected, bg=bg)


def select_unique_shape(g: GridV124, *, bg: int = 0) -> Selection:
    objects = _find_objects(g, bg=bg)
    def shape_signature(obj):
        obj_grid = _extract_object_grid(g, obj, bg=bg)
        h, w = _grid_shape(obj_grid)
        return str([[1 if obj_grid[r][c] != bg else 0 for c in range(w)] for r in range(h)])
    signatures = [shape_signature(obj) for obj in objects]
    sig_counts = Counter(signatures)
    selected = [i for i, sig in enumerate(signatures) if sig_counts[sig] == 1]
    return _create_selection_from_objects(g, objects, selected, bg=bg)


def select_pair_objects(g: GridV124, *, bg: int = 0) -> Selection:
    objects = _find_objects(g, bg=bg)
    signatures = [_object_signature(g, obj, bg=bg) for obj in objects]
    sig_counts = Counter(signatures)
    selected = [i for i, sig in enumerate(signatures) if sig_counts[sig] == 2]
    return _create_selection_from_objects(g, objects, selected, bg=bg)


# SPATIAL DIVISION SELECTORS

def select_top_half(g: GridV124, *, bg: int = 0) -> Selection:
    h, w = _grid_shape(g)
    cells: Set[Tuple[int, int]] = {(r, c) for r in range(h // 2) for c in range(w) if int(g[r][c]) != bg}
    return _create_selection_from_cells(g, cells, bg=bg)


def select_bottom_half(g: GridV124, *, bg: int = 0) -> Selection:
    h, w = _grid_shape(g)
    cells: Set[Tuple[int, int]] = {(r, c) for r in range(h // 2, h) for c in range(w) if int(g[r][c]) != bg}
    return _create_selection_from_cells(g, cells, bg=bg)


def select_left_side(g: GridV124, *, bg: int = 0) -> Selection:
    h, w = _grid_shape(g)
    cells: Set[Tuple[int, int]] = {(r, c) for r in range(h) for c in range(w // 2) if int(g[r][c]) != bg}
    return _create_selection_from_cells(g, cells, bg=bg)


def select_right_side(g: GridV124, *, bg: int = 0) -> Selection:
    h, w = _grid_shape(g)
    cells: Set[Tuple[int, int]] = {(r, c) for r in range(h) for c in range(w // 2, w) if int(g[r][c]) != bg}
    return _create_selection_from_cells(g, cells, bg=bg)


# PATTERN AND OUTLIER SELECTORS

def select_pattern_outliers(g: GridV124, *, bg: int = 0) -> Selection:
    objects = _find_objects(g, bg=bg)
    if len(objects) < 3:
        h, w = _grid_shape(g)
        return Selection(mask=_new_grid(h, w, bg), objects=[], indices=[])
    signatures = [_object_signature(g, obj, bg=bg) for obj in objects]
    sig_counts = Counter(signatures)
    if not sig_counts:
        h, w = _grid_shape(g)
        return Selection(mask=_new_grid(h, w, bg), objects=[], indices=[])
    pattern_sig = sig_counts.most_common(1)[0][0]
    selected = [i for i, sig in enumerate(signatures) if sig != pattern_sig]
    return _create_selection_from_objects(g, objects, selected, bg=bg)


def filter_noise_objects(g: GridV124, *, bg: int = 0, min_size: int = 3) -> Selection:
    objects = _find_objects(g, bg=bg)
    selected = [i for i, obj in enumerate(objects) if len(obj) >= min_size]
    return _create_selection_from_objects(g, objects, selected, bg=bg)


def select_background_region(g: GridV124, *, bg: int = 0) -> Selection:
    h, w = _grid_shape(g)
    inverted = _new_grid(h, w, 0)
    for r in range(h):
        for c in range(w):
            if int(g[r][c]) == bg:
                inverted[r][c] = 1
    bg_regions = _find_objects(inverted, bg=0)
    if not bg_regions:
        return Selection(mask=_new_grid(h, w, 0), objects=[], indices=[])
    largest = max(bg_regions, key=len)
    return _create_selection_from_cells(g, largest, bg=bg)


def select_by_orientation(g: GridV124, *, orientation: str = "horizontal", bg: int = 0) -> Selection:
    objects = _find_objects(g, bg=bg)
    selected = []
    for i, obj in enumerate(objects):
        min_r, min_c, max_r, max_c = _object_bbox(obj)
        height, width = max_r - min_r + 1, max_c - min_c + 1
        if orientation == "horizontal" and width > height:
            selected.append(i)
        elif orientation == "vertical" and height > width:
            selected.append(i)
        elif orientation == "diagonal" and height == width:
            selected.append(i)
    return _create_selection_from_objects(g, objects, selected, bg=bg)


def focus_on_symmetry_break(g: GridV124, *, bg: int = 0) -> Selection:
    h, w = _grid_shape(g)
    cells: Set[Tuple[int, int]] = set()
    for r in range(h):
        for c in range(w // 2):
            mirror_c = w - 1 - c
            if int(g[r][c]) != int(g[r][mirror_c]):
                if int(g[r][c]) != bg:
                    cells.add((r, c))
                if int(g[r][mirror_c]) != bg:
                    cells.add((r, mirror_c))
    for r in range(h // 2):
        mirror_r = h - 1 - r
        for c in range(w):
            if int(g[r][c]) != int(g[mirror_r][c]):
                if int(g[r][c]) != bg:
                    cells.add((r, c))
                if int(g[mirror_r][c]) != bg:
                    cells.add((mirror_r, c))
    return _create_selection_from_cells(g, cells, bg=bg)


# REGISTRY

SEMANTIC_SELECTORS_V161 = {
    "select_largest_object": select_largest_object,
    "select_smallest_object": select_smallest_object,
    "select_shape_by_size": select_shape_by_size,
    "select_by_color": select_by_color,
    "select_most_frequent_color": select_most_frequent_color,
    "select_least_frequent_color": select_least_frequent_color,
    "select_outlier_color": select_outlier_color,
    "select_border_objects": select_border_objects,
    "select_center_object": select_center_object,
    "select_corner_patterns": select_corner_patterns,
    "select_diagonal_line": select_diagonal_line,
    "select_straight_line": select_straight_line,
    "select_closed_shape": select_closed_shape,
    "select_hole": select_hole,
    "select_by_shape": select_by_shape,
    "select_unique_object": select_unique_object,
    "select_duplicate_shapes": select_duplicate_shapes,
    "select_unique_shape": select_unique_shape,
    "select_pair_objects": select_pair_objects,
    "select_top_half": select_top_half,
    "select_bottom_half": select_bottom_half,
    "select_left_side": select_left_side,
    "select_right_side": select_right_side,
    "select_pattern_outliers": select_pattern_outliers,
    "filter_noise_objects": filter_noise_objects,
    "select_background_region": select_background_region,
    "select_by_orientation": select_by_orientation,
    "focus_on_symmetry_break": focus_on_symmetry_break,
}


def get_semantic_selector_v161(name: str):
    return SEMANTIC_SELECTORS_V161.get(name)


def list_semantic_selectors_v161() -> List[str]:
    return list(SEMANTIC_SELECTORS_V161.keys())
