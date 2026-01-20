from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_objects_v132 import BBoxV132, ObjectSetV132, ObjectV132, connected_components4_v132
from .grid_v124 import (
    GridV124,
    crop_to_bbox_nonzero_v124,
    crop_v124,
    grid_from_list_v124,
    grid_shape_v124,
    pad_to_v124,
    reflect_h_v124,
    reflect_v_v124,
    rotate180_v124,
    rotate270_v124,
    rotate90_v124,
    translate_v124,
    unique_colors_v124,
)

ARC_OPS_SCHEMA_VERSION_V132 = 132


def _check_color_v132(c: int) -> int:
    cc = int(c)
    if cc < 0 or cc > 9:
        raise ValueError("color_out_of_range")
    return int(cc)


def _mode_color_v132(g: GridV124) -> int:
    counts: Dict[int, int] = {}
    for row in g:
        for x in row:
            counts[int(x)] = int(counts.get(int(x), 0)) + 1
    if not counts:
        return 0
    items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
    return int(items[0][0])


@dataclass(frozen=True)
class StateV132:
    grid: GridV124
    objset: Optional[ObjectSetV132] = None
    obj: Optional[ObjectV132] = None
    bbox: Optional[BBoxV132] = None
    patch: Optional[GridV124] = None

    def to_sig_dict(self) -> Dict[str, Any]:
        from .grid_v124 import grid_hash_v124

        return {
            "grid_hash": grid_hash_v124(self.grid),
            "objset_sig": str(self.objset.set_sig()) if self.objset is not None else "",
            "obj_sig": str(self.obj.object_sig()) if self.obj is not None else "",
            "bbox": self.bbox.to_dict() if self.bbox is not None else None,
            "patch_hash": grid_hash_v124(self.patch) if self.patch is not None else "",
        }


@dataclass(frozen=True)
class OpDefV132:
    op_id: str
    reads: Tuple[str, ...]
    writes: Tuple[str, ...]
    base_cost_bits: int


OP_DEFS_V132: Dict[str, OpDefV132] = {
    # Object-centric
    "cc4": OpDefV132(op_id="cc4", reads=("grid",), writes=("objset",), base_cost_bits=16),
    "select_obj": OpDefV132(op_id="select_obj", reads=("objset",), writes=("obj",), base_cost_bits=14),
    "obj_bbox": OpDefV132(op_id="obj_bbox", reads=("obj",), writes=("bbox",), base_cost_bits=8),
    "crop_bbox": OpDefV132(op_id="crop_bbox", reads=("grid", "bbox"), writes=("patch",), base_cost_bits=14),
    "commit_patch": OpDefV132(op_id="commit_patch", reads=("patch",), writes=("grid",), base_cost_bits=8),
    "new_canvas": OpDefV132(op_id="new_canvas", reads=tuple(), writes=("grid",), base_cost_bits=18),
    "paint_rect": OpDefV132(op_id="paint_rect", reads=("grid", "bbox"), writes=("grid",), base_cost_bits=16),
    "draw_rect_border": OpDefV132(op_id="draw_rect_border", reads=("grid", "bbox"), writes=("grid",), base_cost_bits=18),
    "paste": OpDefV132(op_id="paste", reads=("grid", "patch"), writes=("grid",), base_cost_bits=20),
    # General grid ops (reused; still general, no task-specific branching)
    "rotate90": OpDefV132(op_id="rotate90", reads=("grid",), writes=("grid",), base_cost_bits=10),
    "rotate180": OpDefV132(op_id="rotate180", reads=("grid",), writes=("grid",), base_cost_bits=10),
    "rotate270": OpDefV132(op_id="rotate270", reads=("grid",), writes=("grid",), base_cost_bits=10),
    "reflect_h": OpDefV132(op_id="reflect_h", reads=("grid",), writes=("grid",), base_cost_bits=10),
    "reflect_v": OpDefV132(op_id="reflect_v", reads=("grid",), writes=("grid",), base_cost_bits=10),
    "translate": OpDefV132(op_id="translate", reads=("grid",), writes=("grid",), base_cost_bits=14),
    "crop_bbox_nonzero": OpDefV132(op_id="crop_bbox_nonzero", reads=("grid",), writes=("grid",), base_cost_bits=12),
    "pad_to": OpDefV132(op_id="pad_to", reads=("grid",), writes=("grid",), base_cost_bits=14),
    "replace_color": OpDefV132(op_id="replace_color", reads=("grid",), writes=("grid",), base_cost_bits=16),
    "map_colors": OpDefV132(op_id="map_colors", reads=("grid",), writes=("grid",), base_cost_bits=18),
}


def _op_args_cost_bits_v132(op_id: str, args: Dict[str, Any]) -> int:
    # General deterministic parameter cost (no ARC-specific tuning):
    # - scalar params: +4 bits
    # - mapping: +8 bits/entry
    # - list params: +4 bits/item
    extra = 0
    for k in sorted(args.keys()):
        v = args[k]
        if k == "mapping" and isinstance(v, dict):
            extra += 8 * int(len(v))
            continue
        if isinstance(v, list):
            extra += 4 * int(len(v))
            continue
        # None / bool / int / str treated uniformly
        extra += 4
    # Slight penalty for enum-like selector richness.
    if op_id == "select_obj":
        extra += 2
    return int(extra)


def step_cost_bits_v132(*, op_id: str, args: Dict[str, Any]) -> int:
    od = OP_DEFS_V132.get(str(op_id))
    base = int(od.base_cost_bits) if od is not None else 24
    return int(base + _op_args_cost_bits_v132(str(op_id), dict(args)))


def _select_obj_v132(
    oset: ObjectSetV132,
    *,
    key: str,
    order: str,
    rank: int,
    color_filter: Optional[int],
    grid_shape: Tuple[int, int],
) -> ObjectV132:
    objs = list(oset.objects)
    if color_filter is not None:
        cf = _check_color_v132(int(color_filter))
        objs = [o for o in objs if int(o.color) == int(cf)]
    if not objs:
        raise ValueError("select_obj_empty")

    h, w = grid_shape
    center_r2 = int(h - 1)
    center_c2 = int(w - 1)

    def key_value(o: ObjectV132) -> int:
        if key == "area":
            return int(o.area)
        if key == "width":
            return int(o.width)
        if key == "height":
            return int(o.height)
        if key == "bbox_area":
            return int(o.width * o.height)
        if key == "top":
            return int(o.bbox.r0)
        if key == "left":
            return int(o.bbox.c0)
        if key == "bottom":
            return int(o.bbox.r1)
        if key == "right":
            return int(o.bbox.c1)
        if key == "color":
            return int(o.color)
        if key == "dist_center":
            return int(abs(int(o.bbox_center_r2) - center_r2) + abs(int(o.bbox_center_c2) - center_c2))
        raise ValueError("select_obj_unknown_key")

    rows: List[Tuple[int, Tuple[int, int, int, int], int, Tuple[Tuple[int, int], ...], ObjectV132]] = []
    for o in objs:
        rows.append((int(key_value(o)), o.bbox.to_tuple(), int(o.color), o.cells, o))
    reverse = str(order) == "max"
    rows.sort(key=lambda r: (int(r[0]), r[1], r[2], r[3]), reverse=bool(reverse))
    rr = int(rank)
    if rr < 0 or rr >= len(rows):
        raise ValueError("select_obj_rank_oob")
    return rows[rr][4]


def apply_op_v132(*, state: StateV132, op_id: str, args: Dict[str, Any]) -> StateV132:
    op = str(op_id)
    a = dict(args)
    if op == "cc4":
        bg = a.get("bg")
        colors = a.get("colors")
        colors_list: Optional[List[int]] = None
        if isinstance(colors, list):
            colors_list = [int(_check_color_v132(int(c))) for c in colors]
        oset = connected_components4_v132(state.grid, colors=colors_list, bg=int(bg) if bg is not None else None)
        return replace(state, objset=oset, obj=None, bbox=None)

    if op == "select_obj":
        if state.objset is None:
            raise ValueError("missing_objset")
        key = str(a.get("key") or "area")
        order = str(a.get("order") or "max")
        rank = int(a.get("rank") or 0)
        cf_raw = a.get("color_filter")
        cf = int(cf_raw) if cf_raw is not None else None
        h, w = grid_shape_v124(state.grid)
        o = _select_obj_v132(state.objset, key=key, order=order, rank=rank, color_filter=cf, grid_shape=(h, w))
        return replace(state, obj=o, bbox=None)

    if op == "obj_bbox":
        if state.obj is None:
            raise ValueError("missing_obj")
        return replace(state, bbox=state.obj.bbox)

    if op == "crop_bbox":
        if state.bbox is None:
            raise ValueError("missing_bbox")
        r0, c0, r1, c1 = state.bbox.to_tuple()
        h, w = grid_shape_v124(state.grid)
        if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
            raise ValueError("bbox_oob")
        patch = crop_v124(state.grid, r0=int(r0), c0=int(c0), height=int(r1 - r0), width=int(c1 - c0))
        return replace(state, patch=patch)

    if op == "commit_patch":
        if state.patch is None:
            raise ValueError("missing_patch")
        return replace(state, grid=state.patch, patch=None, objset=None, obj=None, bbox=None)

    if op == "new_canvas":
        hh = int(a["height"])
        ww = int(a["width"])
        cc = _check_color_v132(int(a.get("color", 0)))
        if hh < 0 or ww < 0:
            raise ValueError("new_canvas_negative_shape")
        g = grid_from_list_v124([[int(cc) for _ in range(ww)] for _ in range(hh)])
        return replace(state, grid=g, objset=None, obj=None, bbox=None, patch=None)

    if op == "paint_rect":
        if state.bbox is None:
            raise ValueError("missing_bbox")
        cc = _check_color_v132(int(a["color"]))
        h, w = grid_shape_v124(state.grid)
        r0, c0, r1, c1 = state.bbox.to_tuple()
        rr0 = max(0, int(r0))
        cc0 = max(0, int(c0))
        rr1 = min(int(h), int(r1))
        cc1 = min(int(w), int(c1))
        out: List[List[int]] = [[int(state.grid[r][c]) for c in range(w)] for r in range(h)]
        for r in range(rr0, rr1):
            for c in range(cc0, cc1):
                out[r][c] = int(cc)
        return replace(state, grid=grid_from_list_v124(out))

    if op == "draw_rect_border":
        if state.bbox is None:
            raise ValueError("missing_bbox")
        cc = _check_color_v132(int(a["color"]))
        t = int(a.get("thickness", 1))
        if t <= 0:
            raise ValueError("thickness_nonpositive")
        h, w = grid_shape_v124(state.grid)
        r0, c0, r1, c1 = state.bbox.to_tuple()
        rr0 = max(0, int(r0))
        cc0 = max(0, int(c0))
        rr1 = min(int(h), int(r1))
        cc1 = min(int(w), int(c1))
        out: List[List[int]] = [[int(state.grid[r][c]) for c in range(w)] for r in range(h)]
        if rr1 <= rr0 or cc1 <= cc0:
            return replace(state, grid=grid_from_list_v124(out))
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
        return replace(state, grid=grid_from_list_v124(out))

    if op == "paste":
        if state.patch is None:
            raise ValueError("missing_patch")
        top = int(a.get("top", 0))
        left = int(a.get("left", 0))
        tr = _check_color_v132(int(a.get("transparent", 0)))
        h, w = grid_shape_v124(state.grid)
        ph, pw = grid_shape_v124(state.patch)
        out: List[List[int]] = [[int(state.grid[r][c]) for c in range(w)] for r in range(h)]
        for r in range(ph):
            for c in range(pw):
                rr = int(top + r)
                cc = int(left + c)
                if 0 <= rr < h and 0 <= cc < w:
                    v = int(state.patch[r][c])
                    if v != int(tr):
                        out[rr][cc] = int(v)
        return replace(state, grid=grid_from_list_v124(out))

    # --- General grid-to-grid ops (no objectization) ---
    if op == "rotate90":
        return replace(state, grid=rotate90_v124(state.grid), objset=None, obj=None, bbox=None, patch=None)
    if op == "rotate180":
        return replace(state, grid=rotate180_v124(state.grid), objset=None, obj=None, bbox=None, patch=None)
    if op == "rotate270":
        return replace(state, grid=rotate270_v124(state.grid), objset=None, obj=None, bbox=None, patch=None)
    if op == "reflect_h":
        return replace(state, grid=reflect_h_v124(state.grid), objset=None, obj=None, bbox=None, patch=None)
    if op == "reflect_v":
        return replace(state, grid=reflect_v_v124(state.grid), objset=None, obj=None, bbox=None, patch=None)
    if op == "translate":
        return replace(
            state,
            grid=translate_v124(state.grid, dx=int(a["dx"]), dy=int(a["dy"]), pad=int(a.get("pad", 0))),
            objset=None,
            obj=None,
            bbox=None,
            patch=None,
        )
    if op == "crop_bbox_nonzero":
        return replace(
            state,
            grid=crop_to_bbox_nonzero_v124(state.grid, bg=int(a.get("bg", 0))),
            objset=None,
            obj=None,
            bbox=None,
            patch=None,
        )
    if op == "pad_to":
        return replace(
            state,
            grid=pad_to_v124(state.grid, height=int(a["height"]), width=int(a["width"]), pad=int(a.get("pad", 0))),
            objset=None,
            obj=None,
            bbox=None,
            patch=None,
        )
    if op == "replace_color":
        fc = _check_color_v132(int(a["from_color"]))
        tc = _check_color_v132(int(a["to_color"]))
        return replace(
            state,
            grid=tuple(tuple(int(tc) if int(x) == int(fc) else int(x) for x in row) for row in state.grid),
            objset=None,
            obj=None,
            bbox=None,
            patch=None,
        )
    if op == "map_colors":
        m_raw = a.get("mapping", {})
        if not isinstance(m_raw, dict):
            raise ValueError("mapping_not_dict")
        m: Dict[int, int] = {}
        for k, v in m_raw.items():
            kk = _check_color_v132(int(k))
            vv = _check_color_v132(int(v))
            m[kk] = vv
        return replace(
            state,
            grid=tuple(tuple(int(m.get(int(x), int(x))) for x in row) for row in state.grid),
            objset=None,
            obj=None,
            bbox=None,
            patch=None,
        )

    raise ValueError(f"unknown_op:{op}")


def _infer_color_mapping_v132(inp: GridV124, out: GridV124) -> Optional[Dict[str, int]]:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if hi != ho or wi != wo or hi == 0 or wi == 0:
        return None
    mapping: Dict[int, int] = {}
    for r in range(hi):
        for c in range(wi):
            a = int(inp[r][c])
            b = int(out[r][c])
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    return {str(k): int(mapping[k]) for k in sorted(mapping.keys())}


def _bg_candidates_v132(grids: Sequence[GridV124]) -> List[int]:
    out: List[int] = [0]
    for g in grids:
        h, w = grid_shape_v124(g)
        if h > 0 and w > 0:
            out.extend([int(g[0][0]), int(g[0][w - 1]), int(g[h - 1][0]), int(g[h - 1][w - 1])])
            out.append(_mode_color_v132(g))
    return sorted(set(int(x) for x in out))


def propose_step_variants_v132(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    max_translate_shift: int = 2,
) -> List[Dict[str, Any]]:
    """
    Deterministic parameter proposal for each operator (small but useful domains).
    Returns canonical step dicts: {"op_id":..., "args":{...}}.
    """
    train_inputs = [p[0] for p in train_pairs]
    train_outputs = [p[1] for p in train_pairs]

    colors: List[int] = []
    shapes: List[Tuple[int, int]] = []
    inferred_maps: List[Dict[str, int]] = []
    for inp, out in train_pairs:
        colors.extend(unique_colors_v124(inp))
        colors.extend(unique_colors_v124(out))
        shapes.append(grid_shape_v124(inp))
        shapes.append(grid_shape_v124(out))
        m = _infer_color_mapping_v132(inp, out)
        if m is not None:
            inferred_maps.append(m)
    colors.extend(unique_colors_v124(test_in))
    shapes.append(grid_shape_v124(test_in))

    colors_s = sorted(set(int(c) for c in colors))
    shapes_s = sorted(set((int(h), int(w)) for h, w in shapes))
    bgs = _bg_candidates_v132(list(train_inputs) + [test_in])
    inferred_maps = sorted(inferred_maps, key=lambda m: canonical_json_dumps(m))

    steps: List[Dict[str, Any]] = []

    # cc4: backgrounds candidates, colors="all" (empty list means all non-bg colors)
    for bg in bgs:
        steps.append({"op_id": "cc4", "args": {"bg": int(bg), "colors": []}})

    # select_obj: generic selectors (small set), with optional color_filter
    keys = ["area", "width", "height", "bbox_area", "top", "left", "bottom", "right", "color", "dist_center"]
    for key in keys:
        for order in ["min", "max"]:
            for rank in [0, 1, 2]:
                steps.append({"op_id": "select_obj", "args": {"key": str(key), "order": str(order), "rank": int(rank), "color_filter": None}})
                for cf in colors_s:
                    steps.append(
                        {"op_id": "select_obj", "args": {"key": str(key), "order": str(order), "rank": int(rank), "color_filter": int(cf)}}
                    )

    # obj_bbox, crop_bbox, commit_patch (no args)
    steps.append({"op_id": "obj_bbox", "args": {}})
    steps.append({"op_id": "crop_bbox", "args": {}})
    steps.append({"op_id": "commit_patch", "args": {}})

    # new_canvas: shapes from seen shapes, colors from bg candidates
    for h, w in shapes_s:
        for bg in bgs:
            steps.append({"op_id": "new_canvas", "args": {"height": int(h), "width": int(w), "color": int(bg)}})

    # paint_rect / draw_rect_border (colors from observed palette)
    for c in colors_s:
        steps.append({"op_id": "paint_rect", "args": {"color": int(c)}})
        steps.append({"op_id": "draw_rect_border", "args": {"color": int(c), "thickness": 1}})

    # paste positions: small offsets plus (0,0)
    offs: set[Tuple[int, int]] = set()
    for dy in range(-int(max_translate_shift), int(max_translate_shift) + 1):
        for dx in range(-int(max_translate_shift), int(max_translate_shift) + 1):
            offs.add((int(dy), int(dx)))
    # deterministic ordering
    for dy, dx in sorted(offs):
        steps.append({"op_id": "paste", "args": {"top": int(dy), "left": int(dx), "transparent": 0}})

    # Basic grid ops (no args)
    for op_id in ["rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v"]:
        steps.append({"op_id": str(op_id), "args": {}})

    # translate candidates: small offsets + inferred bbox shifts (bg=0 only; still general)
    trans: set[Tuple[int, int]] = set()
    for dy in range(-int(max_translate_shift), int(max_translate_shift) + 1):
        for dx in range(-int(max_translate_shift), int(max_translate_shift) + 1):
            trans.add((int(dy), int(dx)))
    for inp, out in train_pairs:
        try:
            hi, wi = grid_shape_v124(inp)
            ho, wo = grid_shape_v124(out)
            if (hi, wi) == (ho, wo) and hi > 0 and wi > 0:
                # bbox shift between nonzero bboxes (bg=0) as a candidate translation
                from .grid_v124 import bbox_nonzero_v124

                ir0, ic0, _, _ = bbox_nonzero_v124(inp, bg=0)
                or0, oc0, _, _ = bbox_nonzero_v124(out, bg=0)
                trans.add((int(or0 - ir0), int(oc0 - ic0)))
        except Exception:
            pass
    for dy, dx in sorted(trans):
        if dy == 0 and dx == 0:
            continue
        steps.append({"op_id": "translate", "args": {"dx": int(dx), "dy": int(dy), "pad": 0}})

    # crop_bbox_nonzero backgrounds
    for bg in bgs:
        steps.append({"op_id": "crop_bbox_nonzero", "args": {"bg": int(bg)}})

    # pad_to: shapes seen
    for h, w in shapes_s:
        steps.append({"op_id": "pad_to", "args": {"height": int(h), "width": int(w), "pad": 0}})

    # replace_color from palette
    for fc in colors_s:
        for tc in colors_s:
            if fc == tc:
                continue
            steps.append({"op_id": "replace_color", "args": {"from_color": int(fc), "to_color": int(tc)}})

    # map_colors from inferred mappings
    for m in inferred_maps:
        steps.append({"op_id": "map_colors", "args": {"mapping": dict(m)}})

    # Canonical ordering and dedup
    seen: set[str] = set()
    uniq: List[Dict[str, Any]] = []
    for st in steps:
        body = {"schema_version": int(ARC_OPS_SCHEMA_VERSION_V132), "kind": "arc_step_v132", "step": {"op_id": str(st["op_id"]), "args": st["args"]}}
        fp = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
        if fp in seen:
            continue
        seen.add(fp)
        uniq.append({"op_id": str(st["op_id"]), "args": dict(st["args"])})
    uniq.sort(key=lambda s: (str(s["op_id"]), canonical_json_dumps(s["args"])))
    return uniq

