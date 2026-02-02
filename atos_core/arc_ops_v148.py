"""
arc_ops_v148.py - Extended Operators for ARC-AGI Solving

Extends arc_ops_v141 with additional operators critical for ARC-AGI-1/2:

1. Morphological operations (dilate, erode, open, close)
2. Symmetry detection (vertical, horizontal, diagonal, rotational)
3. Object grouping and merging
4. Grid rescaling
5. Feature extraction (color frequencies, object counts, pattern locations)
6. Advanced transformations (fractal tiling, pattern completion)

Schema version: 148
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from .arc_objects_v132 import BBoxV132, ObjectSetV132, ObjectV132
from .arc_ops_v132 import OpDefV132, StateV132
from .arc_ops_v141 import (
    ARC_OPS_SCHEMA_VERSION_V141,
    OP_DEFS_V141,
    apply_op_v141,
    step_cost_bits_v141,
)
from .grid_v124 import (
    GridV124,
    grid_equal_v124,
    grid_from_list_v124,
    grid_hash_v124,
    grid_shape_v124,
    unique_colors_v124,
)

ARC_OPS_SCHEMA_VERSION_V148 = 148

# ─────────────────────────────────────────────────────────────────────────────
# Grid Utility Functions
# ─────────────────────────────────────────────────────────────────────────────


def _check_color_v148(c: int) -> int:
    """Validate ARC color (0-9)."""
    cc = int(c)
    if cc < 0 or cc > 9:
        raise ValueError("color_out_of_range")
    return cc


def _new_grid_v148(rows: int, cols: int, fill: int = 0) -> GridV124:
    """Create a new grid with given dimensions."""
    return [[int(fill) for _ in range(int(cols))] for _ in range(int(rows))]


def _copy_grid_v148(g: GridV124) -> GridV124:
    """Deep copy a grid."""
    return [[int(c) for c in row] for row in g]


def _get_neighbors_4_v148(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    """Get 4-connected neighbors within bounds."""
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = int(r + dr), int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            neighbors.append((nr, nc))
    return neighbors


def _get_neighbors_8_v148(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    """Get 8-connected neighbors within bounds."""
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                neighbors.append((nr, nc))
    return neighbors


# ─────────────────────────────────────────────────────────────────────────────
# 1. Morphological Operations
# ─────────────────────────────────────────────────────────────────────────────


def _morpho_dilate_v148(g: GridV124, *, bg: int = 0, connectivity: int = 4) -> GridV124:
    """
    Morphological dilation: expand non-background regions.
    
    Each background cell adjacent to a non-background cell becomes non-background.
    """
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return _copy_grid_v148(g)
    
    bg = _check_color_v148(bg)
    out = _copy_grid_v148(g)
    get_neighbors = _get_neighbors_4_v148 if connectivity == 4 else _get_neighbors_8_v148
    
    for r in range(h):
        for c in range(w):
            if int(g[r][c]) == bg:
                # Check if any neighbor is non-bg
                for nr, nc in get_neighbors(r, c, h, w):
                    if int(g[nr][nc]) != bg:
                        out[r][c] = int(g[nr][nc])
                        break
    
    return out


def _morpho_erode_v148(g: GridV124, *, bg: int = 0, connectivity: int = 4) -> GridV124:
    """
    Morphological erosion: shrink non-background regions.
    
    Each non-background cell adjacent to a background cell becomes background.
    """
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return _copy_grid_v148(g)
    
    bg = _check_color_v148(bg)
    out = _copy_grid_v148(g)
    get_neighbors = _get_neighbors_4_v148 if connectivity == 4 else _get_neighbors_8_v148
    
    for r in range(h):
        for c in range(w):
            if int(g[r][c]) != bg:
                # Check if any neighbor is bg
                for nr, nc in get_neighbors(r, c, h, w):
                    if int(g[nr][nc]) == bg:
                        out[r][c] = bg
                        break
    
    return out


def _morpho_open_v148(g: GridV124, *, bg: int = 0, connectivity: int = 4) -> GridV124:
    """
    Morphological opening: erode then dilate.
    
    Removes small protrusions and noise.
    """
    eroded = _morpho_erode_v148(g, bg=bg, connectivity=connectivity)
    return _morpho_dilate_v148(eroded, bg=bg, connectivity=connectivity)


def _morpho_close_v148(g: GridV124, *, bg: int = 0, connectivity: int = 4) -> GridV124:
    """
    Morphological closing: dilate then erode.
    
    Fills small holes and gaps.
    """
    dilated = _morpho_dilate_v148(g, bg=bg, connectivity=connectivity)
    return _morpho_erode_v148(dilated, bg=bg, connectivity=connectivity)


def _morpho_skeleton_v148(g: GridV124, *, bg: int = 0) -> GridV124:
    """
    Morphological skeletonization: reduce shapes to 1-pixel wide skeleton.
    
    Useful for detecting structure patterns.
    """
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return _copy_grid_v148(g)
    
    bg = _check_color_v148(bg)
    
    # Simple thinning algorithm
    out = _copy_grid_v148(g)
    changed = True
    
    while changed:
        changed = False
        # Mark pixels for removal
        to_remove: List[Tuple[int, int]] = []
        
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if int(out[r][c]) == bg:
                    continue
                
                # Count non-bg neighbors
                neighbors = [
                    int(out[r - 1][c]) != bg,     # N
                    int(out[r - 1][c + 1]) != bg, # NE
                    int(out[r][c + 1]) != bg,     # E
                    int(out[r + 1][c + 1]) != bg, # SE
                    int(out[r + 1][c]) != bg,     # S
                    int(out[r + 1][c - 1]) != bg, # SW
                    int(out[r][c - 1]) != bg,     # W
                    int(out[r - 1][c - 1]) != bg, # NW
                ]
                
                n_neighbors = sum(neighbors)
                
                # Count transitions from 0 to 1
                transitions = 0
                for i in range(8):
                    if not neighbors[i] and neighbors[(i + 1) % 8]:
                        transitions += 1
                
                # Removal conditions (Zhang-Suen thinning simplified)
                if 2 <= n_neighbors <= 6 and transitions == 1:
                    to_remove.append((r, c))
        
        for r, c in to_remove:
            out[r][c] = bg
            changed = True
    
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Symmetry Detection
# ─────────────────────────────────────────────────────────────────────────────


def _detect_symmetry_h_v148(g: GridV124) -> bool:
    """Detect horizontal (left-right) symmetry."""
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return True
    
    for r in range(h):
        for c in range(w // 2):
            if int(g[r][c]) != int(g[r][w - 1 - c]):
                return False
    return True


def _detect_symmetry_v_v148(g: GridV124) -> bool:
    """Detect vertical (top-bottom) symmetry."""
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return True
    
    for r in range(h // 2):
        for c in range(w):
            if int(g[r][c]) != int(g[h - 1 - r][c]):
                return False
    return True


def _detect_symmetry_diag_v148(g: GridV124) -> bool:
    """Detect diagonal symmetry (transpose equals original)."""
    h, w = grid_shape_v124(g)
    if h != w:
        return False
    
    for r in range(h):
        for c in range(r + 1, w):
            if int(g[r][c]) != int(g[c][r]):
                return False
    return True


def _detect_symmetry_rot180_v148(g: GridV124) -> bool:
    """Detect 180-degree rotational symmetry."""
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return True
    
    for r in range(h):
        for c in range(w):
            if int(g[r][c]) != int(g[h - 1 - r][w - 1 - c]):
                return False
    return True


def _detect_symmetry_rot90_v148(g: GridV124) -> bool:
    """Detect 90-degree rotational symmetry (requires square grid)."""
    h, w = grid_shape_v124(g)
    if h != w:
        return False
    
    for r in range(h):
        for c in range(w):
            # 90° rotation: (r,c) -> (c, h-1-r)
            if int(g[r][c]) != int(g[c][h - 1 - r]):
                return False
    return True


def _get_symmetry_type_v148(g: GridV124) -> str:
    """
    Determine symmetry type of grid.
    
    Returns: "none", "h", "v", "hv", "diag", "rot90", "rot180"
    """
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return "full"
    
    sym_h = _detect_symmetry_h_v148(g)
    sym_v = _detect_symmetry_v_v148(g)
    sym_d = _detect_symmetry_diag_v148(g)
    sym_90 = _detect_symmetry_rot90_v148(g)
    sym_180 = _detect_symmetry_rot180_v148(g)
    
    if sym_90:
        return "rot90"
    if sym_d:
        return "diag"
    if sym_h and sym_v:
        return "hv"
    if sym_h:
        return "h"
    if sym_v:
        return "v"
    if sym_180:
        return "rot180"
    return "none"


def _symmetry_complete_h_v148(g: GridV124) -> GridV124:
    """Complete grid to have horizontal symmetry (mirror right from left)."""
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return _copy_grid_v148(g)
    
    out = _copy_grid_v148(g)
    for r in range(h):
        for c in range(w // 2 + 1, w):
            out[r][c] = int(out[r][w - 1 - c])
    return out


def _symmetry_complete_v_v148(g: GridV124) -> GridV124:
    """Complete grid to have vertical symmetry (mirror bottom from top)."""
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return _copy_grid_v148(g)
    
    out = _copy_grid_v148(g)
    for r in range(h // 2 + 1, h):
        for c in range(w):
            out[r][c] = int(out[h - 1 - r][c])
    return out


def _symmetry_complete_rot180_v148(g: GridV124) -> GridV124:
    """Complete grid to have 180° rotational symmetry."""
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return _copy_grid_v148(g)
    
    out = _copy_grid_v148(g)
    mid = (h * w) // 2
    
    for r in range(h):
        for c in range(w):
            idx = r * w + c
            if idx > mid:
                out[r][c] = int(out[h - 1 - r][w - 1 - c])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Object Grouping and Merging
# ─────────────────────────────────────────────────────────────────────────────


def _merge_adjacent_objects_v148(
    g: GridV124,
    *,
    bg: int = 0,
    merge_color: Optional[int] = None,
) -> GridV124:
    """
    Merge objects that are adjacent (touching).
    
    If merge_color is None, uses mode color of merged region.
    """
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return _copy_grid_v148(g)
    
    bg = _check_color_v148(bg)
    out = _copy_grid_v148(g)
    
    # Find connected components considering 8-connectivity
    visited: Set[Tuple[int, int]] = set()
    
    for r in range(h):
        for c in range(w):
            if (r, c) in visited or int(g[r][c]) == bg:
                continue
            
            # BFS to find all connected non-bg cells
            component: List[Tuple[int, int]] = []
            colors: List[int] = []
            queue = [(r, c)]
            visited.add((r, c))
            
            while queue:
                cr, cc = queue.pop(0)
                component.append((cr, cc))
                colors.append(int(g[cr][cc]))
                
                for nr, nc in _get_neighbors_8_v148(cr, cc, h, w):
                    if (nr, nc) not in visited and int(g[nr][nc]) != bg:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            
            # Determine color for merged region
            if merge_color is not None:
                fill = _check_color_v148(merge_color)
            else:
                # Mode color
                fill = Counter(colors).most_common(1)[0][0]
            
            # Apply merge
            for mr, mc in component:
                out[mr][mc] = fill
    
    return out


def _group_by_color_v148(g: GridV124, *, bg: int = 0) -> Dict[int, List[Tuple[int, int]]]:
    """Group cell positions by color."""
    h, w = grid_shape_v124(g)
    groups: Dict[int, List[Tuple[int, int]]] = {}
    
    for r in range(h):
        for c in range(w):
            color = int(g[r][c])
            if color != bg:
                if color not in groups:
                    groups[color] = []
                groups[color].append((r, c))
    
    return groups


def _fill_between_objects_v148(
    g: GridV124,
    *,
    bg: int = 0,
    fill_color: int = 1,
) -> GridV124:
    """Fill the space between objects (convex hull-ish)."""
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return _copy_grid_v148(g)
    
    bg = _check_color_v148(bg)
    fill_color = _check_color_v148(fill_color)
    out = _copy_grid_v148(g)
    
    # For each row, fill between leftmost and rightmost non-bg
    for r in range(h):
        left = -1
        right = -1
        for c in range(w):
            if int(g[r][c]) != bg:
                if left == -1:
                    left = c
                right = c
        
        if left != -1 and right != -1:
            for c in range(left, right + 1):
                if int(out[r][c]) == bg:
                    out[r][c] = fill_color
    
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Grid Rescaling
# ─────────────────────────────────────────────────────────────────────────────


def _scale_up_v148(g: GridV124, *, factor: int = 2) -> GridV124:
    """Scale grid up by integer factor (nearest neighbor)."""
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0 or factor < 1:
        return _copy_grid_v148(g)
    
    new_h = h * factor
    new_w = w * factor
    out = _new_grid_v148(new_h, new_w)
    
    for r in range(new_h):
        for c in range(new_w):
            out[r][c] = int(g[r // factor][c // factor])
    
    return out


def _scale_down_v148(g: GridV124, *, factor: int = 2, mode: str = "mode") -> GridV124:
    """
    Scale grid down by integer factor.
    
    mode: "mode" (most common), "min", "max", "first"
    """
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0 or factor < 1:
        return _copy_grid_v148(g)
    
    new_h = h // factor
    new_w = w // factor
    
    if new_h == 0 or new_w == 0:
        return _copy_grid_v148(g)
    
    out = _new_grid_v148(new_h, new_w)
    
    for r in range(new_h):
        for c in range(new_w):
            # Collect values in the block
            values = []
            for dr in range(factor):
                for dc in range(factor):
                    sr, sc = r * factor + dr, c * factor + dc
                    if sr < h and sc < w:
                        values.append(int(g[sr][sc]))
            
            if not values:
                continue
            
            if mode == "mode":
                out[r][c] = Counter(values).most_common(1)[0][0]
            elif mode == "min":
                out[r][c] = min(values)
            elif mode == "max":
                out[r][c] = max(values)
            else:  # first
                out[r][c] = values[0]
    
    return out


def _resize_v148(g: GridV124, *, new_h: int, new_w: int, bg: int = 0) -> GridV124:
    """Resize grid to specific dimensions, cropping or padding as needed."""
    h, w = grid_shape_v124(g)
    bg = _check_color_v148(bg)
    out = _new_grid_v148(new_h, new_w, bg)
    
    for r in range(min(h, new_h)):
        for c in range(min(w, new_w)):
            out[r][c] = int(g[r][c])
    
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────


def _count_colors_v148(g: GridV124) -> Dict[int, int]:
    """Count occurrences of each color."""
    h, w = grid_shape_v124(g)
    counts: Dict[int, int] = {}
    
    for r in range(h):
        for c in range(w):
            color = int(g[r][c])
            counts[color] = counts.get(color, 0) + 1
    
    return counts


def _most_frequent_color_v148(g: GridV124, *, exclude_bg: bool = True, bg: int = 0) -> int:
    """Get the most frequent color."""
    counts = _count_colors_v148(g)
    
    if exclude_bg and bg in counts:
        del counts[bg]
    
    if not counts:
        return bg
    
    return max(counts, key=lambda k: counts[k])


def _least_frequent_color_v148(g: GridV124, *, exclude_bg: bool = True, bg: int = 0) -> int:
    """Get the least frequent color (non-bg)."""
    counts = _count_colors_v148(g)
    
    if exclude_bg and bg in counts:
        del counts[bg]
    
    if not counts:
        return bg
    
    return min(counts, key=lambda k: counts[k])


def _count_objects_v148(g: GridV124, *, bg: int = 0) -> int:
    """Count number of connected components (objects)."""
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return 0
    
    bg = _check_color_v148(bg)
    visited: Set[Tuple[int, int]] = set()
    count = 0
    
    for r in range(h):
        for c in range(w):
            if (r, c) in visited or int(g[r][c]) == bg:
                continue
            
            # BFS
            queue = [(r, c)]
            visited.add((r, c))
            while queue:
                cr, cc = queue.pop(0)
                for nr, nc in _get_neighbors_4_v148(cr, cc, h, w):
                    if (nr, nc) not in visited and int(g[nr][nc]) != bg:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            
            count += 1
    
    return count


def _find_pattern_v148(g: GridV124, pattern: GridV124) -> List[Tuple[int, int]]:
    """Find all occurrences of pattern in grid. Returns top-left positions."""
    h, w = grid_shape_v124(g)
    ph, pw = grid_shape_v124(pattern)
    
    if ph > h or pw > w or ph == 0 or pw == 0:
        return []
    
    positions: List[Tuple[int, int]] = []
    
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            match = True
            for pr in range(ph):
                for pc in range(pw):
                    if int(g[r + pr][c + pc]) != int(pattern[pr][pc]):
                        match = False
                        break
                if not match:
                    break
            if match:
                positions.append((r, c))
    
    return positions


def _extract_unique_patterns_v148(
    g: GridV124,
    *,
    size: int = 3,
    bg: int = 0,
) -> List[GridV124]:
    """Extract all unique NxN patterns from grid."""
    h, w = grid_shape_v124(g)
    if size > h or size > w:
        return []
    
    patterns: List[GridV124] = []
    seen: Set[str] = set()
    
    for r in range(h - size + 1):
        for c in range(w - size + 1):
            # Extract pattern
            pattern = [[int(g[r + dr][c + dc]) for dc in range(size)] for dr in range(size)]
            
            # Check if contains non-bg
            has_nonbg = any(int(cell) != bg for row in pattern for cell in row)
            if not has_nonbg:
                continue
            
            # Hash pattern
            sig = str(pattern)
            if sig not in seen:
                seen.add(sig)
                patterns.append(pattern)
    
    return patterns


# ─────────────────────────────────────────────────────────────────────────────
# 6. Advanced Transformations
# ─────────────────────────────────────────────────────────────────────────────


def _fractal_tile_v148(g: GridV124, *, iterations: int = 1) -> GridV124:
    """
    Fractal tiling: for each non-bg cell, replace with scaled copy of pattern.
    
    This is a key pattern in many ARC tasks.
    """
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0 or iterations < 1:
        return _copy_grid_v148(g)
    
    result = _copy_grid_v148(g)
    
    for _ in range(iterations):
        old_h, old_w = grid_shape_v124(result)
        new_h = old_h * old_h
        new_w = old_w * old_w
        
        # Limit size
        if new_h > 30 or new_w > 30:
            break
        
        new_result = _new_grid_v148(new_h, new_w)
        
        for r in range(old_h):
            for c in range(old_w):
                if int(result[r][c]) != 0:
                    # Place copy of pattern
                    for pr in range(old_h):
                        for pc in range(old_w):
                            nr = r * old_h + pr
                            nc = c * old_w + pc
                            if nr < new_h and nc < new_w:
                                new_result[nr][nc] = int(result[pr][pc])
        
        result = new_result
    
    return result


def _complete_pattern_v148(
    g: GridV124,
    *,
    reference: GridV124,
    bg: int = 0,
) -> GridV124:
    """
    Pattern completion: fill missing parts based on reference pattern.
    
    Useful for tasks where partial patterns need completion.
    """
    h, w = grid_shape_v124(g)
    rh, rw = grid_shape_v124(reference)
    
    if h == 0 or w == 0 or rh == 0 or rw == 0:
        return _copy_grid_v148(g)
    
    bg = _check_color_v148(bg)
    out = _copy_grid_v148(g)
    
    # Find matching region
    positions = _find_pattern_v148(g, reference)
    
    if not positions:
        # Try to find partial match and complete
        for r in range(h):
            for c in range(w):
                # Check if this could be start of reference
                matches = 0
                for pr in range(min(rh, h - r)):
                    for pc in range(min(rw, w - c)):
                        if int(g[r + pr][c + pc]) == int(reference[pr][pc]):
                            matches += 1
                        elif int(g[r + pr][c + pc]) != bg:
                            matches = -1000
                            break
                    if matches < 0:
                        break
                
                # If partial match, complete
                if matches > 0:
                    for pr in range(min(rh, h - r)):
                        for pc in range(min(rw, w - c)):
                            if int(out[r + pr][c + pc]) == bg:
                                out[r + pr][c + pc] = int(reference[pr][pc])
    
    return out


def _xor_grids_v148(g1: GridV124, g2: GridV124, *, bg: int = 0) -> GridV124:
    """XOR two grids: output non-bg where only one input is non-bg."""
    h1, w1 = grid_shape_v124(g1)
    h2, w2 = grid_shape_v124(g2)
    h, w = max(h1, h2), max(w1, w2)
    
    bg = _check_color_v148(bg)
    out = _new_grid_v148(h, w, bg)
    
    for r in range(h):
        for c in range(w):
            v1 = int(g1[r][c]) if r < h1 and c < w1 else bg
            v2 = int(g2[r][c]) if r < h2 and c < w2 else bg
            
            is_fg1 = v1 != bg
            is_fg2 = v2 != bg
            
            if is_fg1 and not is_fg2:
                out[r][c] = v1
            elif is_fg2 and not is_fg1:
                out[r][c] = v2
    
    return out


def _and_grids_v148(g1: GridV124, g2: GridV124, *, bg: int = 0) -> GridV124:
    """AND two grids: output non-bg where both inputs are non-bg."""
    h1, w1 = grid_shape_v124(g1)
    h2, w2 = grid_shape_v124(g2)
    h, w = min(h1, h2), min(w1, w2)
    
    bg = _check_color_v148(bg)
    out = _new_grid_v148(h, w, bg)
    
    for r in range(h):
        for c in range(w):
            v1 = int(g1[r][c])
            v2 = int(g2[r][c])
            
            if v1 != bg and v2 != bg:
                out[r][c] = v1  # Use color from first grid
    
    return out


def _or_grids_v148(g1: GridV124, g2: GridV124, *, bg: int = 0) -> GridV124:
    """OR two grids: output non-bg where either input is non-bg."""
    h1, w1 = grid_shape_v124(g1)
    h2, w2 = grid_shape_v124(g2)
    h, w = max(h1, h2), max(w1, w2)
    
    bg = _check_color_v148(bg)
    out = _new_grid_v148(h, w, bg)
    
    for r in range(h):
        for c in range(w):
            v1 = int(g1[r][c]) if r < h1 and c < w1 else bg
            v2 = int(g2[r][c]) if r < h2 and c < w2 else bg
            
            if v1 != bg:
                out[r][c] = v1
            elif v2 != bg:
                out[r][c] = v2
    
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Apply Functions
# ─────────────────────────────────────────────────────────────────────────────


def apply_op_morpho_dilate_v148(s: StateV132, *, bg: int = 0, conn: int = 4) -> StateV132:
    """Apply morphological dilation."""
    g = s.grid
    new_g = _morpho_dilate_v148(g, bg=bg, connectivity=conn)
    return replace(s, grid=new_g)


def apply_op_morpho_erode_v148(s: StateV132, *, bg: int = 0, conn: int = 4) -> StateV132:
    """Apply morphological erosion."""
    g = s.grid
    new_g = _morpho_erode_v148(g, bg=bg, connectivity=conn)
    return replace(s, grid=new_g)


def apply_op_morpho_open_v148(s: StateV132, *, bg: int = 0, conn: int = 4) -> StateV132:
    """Apply morphological opening."""
    g = s.grid
    new_g = _morpho_open_v148(g, bg=bg, connectivity=conn)
    return replace(s, grid=new_g)


def apply_op_morpho_close_v148(s: StateV132, *, bg: int = 0, conn: int = 4) -> StateV132:
    """Apply morphological closing."""
    g = s.grid
    new_g = _morpho_close_v148(g, bg=bg, connectivity=conn)
    return replace(s, grid=new_g)


def apply_op_morpho_skeleton_v148(s: StateV132, *, bg: int = 0) -> StateV132:
    """Apply morphological skeletonization."""
    g = s.grid
    new_g = _morpho_skeleton_v148(g, bg=bg)
    return replace(s, grid=new_g)


def apply_op_symmetry_complete_h_v148(s: StateV132) -> StateV132:
    """Complete to horizontal symmetry."""
    g = s.grid
    new_g = _symmetry_complete_h_v148(g)
    return replace(s, grid=new_g)


def apply_op_symmetry_complete_v_v148(s: StateV132) -> StateV132:
    """Complete to vertical symmetry."""
    g = s.grid
    new_g = _symmetry_complete_v_v148(g)
    return replace(s, grid=new_g)


def apply_op_symmetry_complete_rot180_v148(s: StateV132) -> StateV132:
    """Complete to 180° rotational symmetry."""
    g = s.grid
    new_g = _symmetry_complete_rot180_v148(g)
    return replace(s, grid=new_g)


def apply_op_scale_up_v148(s: StateV132, *, factor: int = 2) -> StateV132:
    """Scale grid up."""
    g = s.grid
    new_g = _scale_up_v148(g, factor=factor)
    return replace(s, grid=new_g)


def apply_op_scale_down_v148(s: StateV132, *, factor: int = 2, mode: str = "mode") -> StateV132:
    """Scale grid down."""
    g = s.grid
    new_g = _scale_down_v148(g, factor=factor, mode=mode)
    return replace(s, grid=new_g)


def apply_op_merge_adjacent_v148(s: StateV132, *, bg: int = 0) -> StateV132:
    """Merge adjacent objects."""
    g = s.grid
    new_g = _merge_adjacent_objects_v148(g, bg=bg)
    return replace(s, grid=new_g)


def apply_op_fill_between_v148(s: StateV132, *, bg: int = 0, fill: int = 1) -> StateV132:
    """Fill between objects."""
    g = s.grid
    new_g = _fill_between_objects_v148(g, bg=bg, fill_color=fill)
    return replace(s, grid=new_g)


def apply_op_fractal_tile_v148(s: StateV132, *, iterations: int = 1) -> StateV132:
    """Apply fractal tiling."""
    g = s.grid
    new_g = _fractal_tile_v148(g, iterations=iterations)
    return replace(s, grid=new_g)


# ─────────────────────────────────────────────────────────────────────────────
# Operator Definitions
# ─────────────────────────────────────────────────────────────────────────────

# Start with V141 operators
OP_DEFS_V148 = dict(OP_DEFS_V141)

# Add new morphological operators
OP_DEFS_V148["morpho_dilate"] = OpDefV132(
    op_id="morpho_dilate",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=18,
)

OP_DEFS_V148["morpho_erode"] = OpDefV132(
    op_id="morpho_erode",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=18,
)

OP_DEFS_V148["morpho_open"] = OpDefV132(
    op_id="morpho_open",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=24,
)

OP_DEFS_V148["morpho_close"] = OpDefV132(
    op_id="morpho_close",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=24,
)

OP_DEFS_V148["morpho_skeleton"] = OpDefV132(
    op_id="morpho_skeleton",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=28,
)

# Add symmetry operators
OP_DEFS_V148["symmetry_complete_h"] = OpDefV132(
    op_id="symmetry_complete_h",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=16,
)

OP_DEFS_V148["symmetry_complete_v"] = OpDefV132(
    op_id="symmetry_complete_v",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=16,
)

OP_DEFS_V148["symmetry_complete_rot180"] = OpDefV132(
    op_id="symmetry_complete_rot180",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=18,
)

# Add scaling operators
OP_DEFS_V148["scale_up"] = OpDefV132(
    op_id="scale_up",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=20,
)

OP_DEFS_V148["scale_down"] = OpDefV132(
    op_id="scale_down",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=20,
)

# Add object manipulation operators
OP_DEFS_V148["merge_adjacent"] = OpDefV132(
    op_id="merge_adjacent",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=22,
)

OP_DEFS_V148["fill_between"] = OpDefV132(
    op_id="fill_between",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=20,
)

# Add fractal operator
OP_DEFS_V148["fractal_tile"] = OpDefV132(
    op_id="fractal_tile",
    reads=("grid",),
    writes=("grid",),
    base_cost_bits=26,
)


# ─────────────────────────────────────────────────────────────────────────────
# Apply Dispatcher
# ─────────────────────────────────────────────────────────────────────────────


def apply_op_v148(
    s: StateV132,
    op_id: str,
    args: Dict[str, Any],
) -> StateV132:
    """Apply an operator to state."""
    
    # First try V148 operators
    if op_id == "morpho_dilate":
        return apply_op_morpho_dilate_v148(s, bg=args.get("bg", 0), conn=args.get("conn", 4))
    elif op_id == "morpho_erode":
        return apply_op_morpho_erode_v148(s, bg=args.get("bg", 0), conn=args.get("conn", 4))
    elif op_id == "morpho_open":
        return apply_op_morpho_open_v148(s, bg=args.get("bg", 0), conn=args.get("conn", 4))
    elif op_id == "morpho_close":
        return apply_op_morpho_close_v148(s, bg=args.get("bg", 0), conn=args.get("conn", 4))
    elif op_id == "morpho_skeleton":
        return apply_op_morpho_skeleton_v148(s, bg=args.get("bg", 0))
    elif op_id == "symmetry_complete_h":
        return apply_op_symmetry_complete_h_v148(s)
    elif op_id == "symmetry_complete_v":
        return apply_op_symmetry_complete_v_v148(s)
    elif op_id == "symmetry_complete_rot180":
        return apply_op_symmetry_complete_rot180_v148(s)
    elif op_id == "scale_up":
        return apply_op_scale_up_v148(s, factor=args.get("factor", 2))
    elif op_id == "scale_down":
        return apply_op_scale_down_v148(s, factor=args.get("factor", 2), mode=args.get("mode", "mode"))
    elif op_id == "merge_adjacent":
        return apply_op_merge_adjacent_v148(s, bg=args.get("bg", 0))
    elif op_id == "fill_between":
        return apply_op_fill_between_v148(s, bg=args.get("bg", 0), fill=args.get("fill", 1))
    elif op_id == "fractal_tile":
        return apply_op_fractal_tile_v148(s, iterations=args.get("iterations", 1))
    
    # Fall back to V141
    return apply_op_v141(s, op_id, args)


def step_cost_bits_v148(op_id: str, args: Dict[str, Any]) -> int:
    """Get cost of an operator."""
    if op_id in OP_DEFS_V148:
        return int(OP_DEFS_V148[op_id].base_cost_bits)
    return step_cost_bits_v141(op_id, args)


# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "ARC_OPS_SCHEMA_VERSION_V148",
    "OP_DEFS_V148",
    "apply_op_v148",
    "step_cost_bits_v148",
    # Feature extraction
    "_count_colors_v148",
    "_most_frequent_color_v148",
    "_least_frequent_color_v148",
    "_count_objects_v148",
    "_find_pattern_v148",
    "_extract_unique_patterns_v148",
    # Symmetry detection
    "_detect_symmetry_h_v148",
    "_detect_symmetry_v_v148",
    "_detect_symmetry_diag_v148",
    "_detect_symmetry_rot90_v148",
    "_detect_symmetry_rot180_v148",
    "_get_symmetry_type_v148",
    # Grid operations
    "_xor_grids_v148",
    "_and_grids_v148",
    "_or_grids_v148",
]
