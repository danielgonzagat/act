from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_objects_v132 import connected_components4_v132
from .arc_ops_v132 import StateV132
from .arc_ops_v135 import OP_DEFS_V135, apply_op_v135, step_cost_bits_v135
from .grid_v124 import (
    GridV124,
    bbox_nonzero_v124,
    crop_to_bbox_nonzero_v124,
    grid_equal_v124,
    grid_hash_v124,
    grid_shape_v124,
    pad_to_v124,
    unique_colors_v124,
)

ARC_SOLVER_SCHEMA_VERSION_V136 = 136


def _validate_grid_values_v136(g: GridV124) -> None:
    for row in g:
        for x in row:
            xx = int(x)
            if xx < 0 or xx > 9:
                raise ValueError("grid_cell_out_of_range")


@dataclass(frozen=True)
class ProgramStepV136:
    op_id: str
    args: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        a: Dict[str, Any] = {}
        for k in sorted(self.args.keys()):
            a[str(k)] = self.args[k]
        return {"op_id": str(self.op_id), "args": a}


@dataclass(frozen=True)
class ProgramV136:
    steps: Tuple[ProgramStepV136, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V136),
            "kind": "arc_program_v136",
            "steps": [s.to_dict() for s in self.steps],
        }

    def program_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


def _summarize_mismatch_v136(*, got: GridV124, want: GridV124) -> Dict[str, Any]:
    hg, wg = grid_shape_v124(got)
    hw, ww = grid_shape_v124(want)
    if (hg, wg) != (hw, ww):
        return {"kind": "shape_mismatch", "got": {"h": hg, "w": wg}, "want": {"h": hw, "w": ww}}
    diff = 0
    for r in range(hg):
        for c in range(wg):
            if int(got[r][c]) != int(want[r][c]):
                diff += 1
    return {"kind": "cell_mismatch", "diff_cells": int(diff), "total_cells": int(hg * wg)}


def _abstract_slots_after_steps_v136(steps: Sequence[ProgramStepV136]) -> Dict[str, bool]:
    avail: Dict[str, bool] = {"grid": True, "objset": False, "obj": False, "bbox": False, "patch": False}
    for st in steps:
        od = OP_DEFS_V135.get(str(st.op_id))
        if od is None:
            continue
        for r in od.reads:
            if not bool(avail.get(str(r), False)):
                return {"grid": True, "objset": False, "obj": False, "bbox": False, "patch": False, "invalid": True}
        for w in od.writes:
            avail[str(w)] = True
        # Model known invalidations (must match apply_op semantics).
        if str(st.op_id) == "commit_patch":
            # Concrete: grid := patch; patch cleared; obj/bbox cleared.
            avail["patch"] = False
            avail["objset"] = False
            avail["obj"] = False
            avail["bbox"] = False
        if str(st.op_id) == "new_canvas":
            avail["objset"] = False
            avail["obj"] = False
            avail["bbox"] = False
            avail["patch"] = False
        if str(st.op_id) in {
            "rotate90",
            "rotate180",
            "rotate270",
            "reflect_h",
            "reflect_v",
            "translate",
            "overlay_self_translate",
            "crop_bbox_nonzero",
            "pad_to",
            "replace_color",
            "map_colors",
            "repeat_grid",
        }:
            avail["objset"] = False
            avail["obj"] = False
            avail["bbox"] = False
            avail["patch"] = False
        if str(st.op_id) == "bbox_by_color":
            avail["objset"] = False
            avail["obj"] = False
            avail["patch"] = False
    return avail


def _stage_from_avail_v136(avail: Dict[str, bool]) -> str:
    if bool(avail.get("patch")):
        return "patch"
    if bool(avail.get("bbox")):
        return "bbox"
    if bool(avail.get("obj")):
        return "obj"
    if bool(avail.get("objset")):
        return "objset"
    return "grid"


def _min_steps_to_grid_stage_v136(stage: str) -> int:
    s = str(stage)
    if s == "grid":
        return 0
    if s == "patch":
        return 1  # commit_patch
    if s == "bbox":
        return 2  # crop_bbox + commit_patch (or paint->?? but stage reset needs commit_patch)
    if s == "obj":
        return 3  # obj_bbox + crop_bbox + commit_patch
    if s == "objset":
        return 4  # select_obj + obj_bbox + crop_bbox + commit_patch
    return 99


def _min_steps_to_grid_modify_v136(stage: str) -> int:
    s = str(stage)
    if s == "grid":
        return 1
    if s == "bbox":
        return 1  # paint_rect / draw_rect_border
    if s == "patch":
        return 1  # paste or commit_patch
    if s == "obj":
        return 2  # obj_bbox -> paint_rect
    if s == "objset":
        return 3  # select_obj -> obj_bbox -> paint_rect
    return 99


def _min_steps_to_shape_change_v136(stage: str, direction: str) -> int:
    s = str(stage)
    d = str(direction)
    if d == "grow":
        return int(_min_steps_to_grid_stage_v136(s)) + 1  # reach grid stage + pad_to/new_canvas
    if d == "shrink":
        if s == "patch":
            return 1  # commit_patch shrinks to patch
        if s == "bbox":
            return 2  # crop_bbox + commit_patch
        if s == "obj":
            return 3
        if s == "objset":
            return 4
        return 1  # crop_bbox_nonzero
    if d == "mixed":
        return int(_min_steps_to_grid_stage_v136(s)) + 2
    return 0


def _can_reach_shape_v136(
    *, stage: str, got_shape: Tuple[int, int], want_shape: Tuple[int, int], steps_left: int
) -> bool:
    hg, wg = int(got_shape[0]), int(got_shape[1])
    hw, ww = int(want_shape[0]), int(want_shape[1])
    if (hg, wg) == (hw, ww):
        return True
    if int(steps_left) <= 0:
        return False
    if hw >= hg and ww >= wg:
        needed = _min_steps_to_shape_change_v136(stage, "grow")
        return int(steps_left) >= int(needed)
    if hw <= hg and ww <= wg:
        needed = _min_steps_to_shape_change_v136(stage, "shrink")
        return int(steps_left) >= int(needed)
    needed = _min_steps_to_shape_change_v136(stage, "mixed")
    return int(steps_left) >= int(needed)


def _can_reach_palette_v136(
    *, stage: str, got_palette: Set[int], want_palette: Set[int], steps_left: int
) -> bool:
    if set(int(x) for x in got_palette) == set(int(x) for x in want_palette):
        return True
    if int(steps_left) <= 0:
        return False
    # Minimal, sound reachability: if we cannot modify the grid at all in remaining steps, palette cannot change.
    return int(steps_left) >= int(_min_steps_to_grid_modify_v136(stage))


def _last_cc4_bg_v136(steps: Sequence[ProgramStepV136]) -> Optional[int]:
    for st in reversed(list(steps)):
        if str(st.op_id) == "cc4":
            bg = st.args.get("bg")
            if bg is None:
                return None
            return int(bg)
    return None


def apply_program_v136(program: ProgramV136, g_in: GridV124) -> GridV124:
    _validate_grid_values_v136(g_in)
    st = StateV132(grid=g_in)
    for step in program.steps:
        st = apply_op_v135(state=st, op_id=str(step.op_id), args=dict(step.args))
        _validate_grid_values_v136(st.grid)
        if st.patch is not None:
            _validate_grid_values_v136(st.patch)
    return st.grid


def apply_program_state_v136(program: ProgramV136, g_in: GridV124) -> StateV132:
    _validate_grid_values_v136(g_in)
    st = StateV132(grid=g_in)
    for step in program.steps:
        st = apply_op_v135(state=st, op_id=str(step.op_id), args=dict(step.args))
        _validate_grid_values_v136(st.grid)
        if st.patch is not None:
            _validate_grid_values_v136(st.patch)
    return st


def _state_sig_dict_v136(st: StateV132) -> Dict[str, Any]:
    d: Dict[str, Any] = {"grid_hash": str(grid_hash_v124(st.grid))}
    if st.obj is not None:
        d["obj"] = {
            "color": int(st.obj.color),
            "bbox": tuple(int(x) for x in st.obj.bbox.to_tuple()),
            "area": int(st.obj.area),
        }
    if st.bbox is not None:
        d["bbox"] = tuple(int(x) for x in st.bbox.to_tuple())
    if st.patch is not None:
        ph, pw = grid_shape_v124(st.patch)
        d["patch_hash"] = str(grid_hash_v124(st.patch))
        d["patch_shape"] = {"h": int(ph), "w": int(pw)}
    return d


def _program_state_sig_v136(
    *,
    program: ProgramV136,
    train_in: Sequence[GridV124],
    test_in: GridV124,
    stage: str,
    last_cc4_bg: Optional[int],
) -> str:
    # Include only what can influence future execution/proposal. For non-object stages, cc4 bg
    # does not matter (cc4 must be re-run anyway), so omit to enable stronger pruning.
    bg_for_sig: Optional[int] = int(last_cc4_bg) if (stage in {"objset", "obj"} and last_cc4_bg is not None) else None
    train_rows: List[Dict[str, Any]] = []
    for idx, g in enumerate(list(train_in)):
        try:
            st = apply_program_state_v136(program, g)
            row = {"i": int(idx), "state": _state_sig_dict_v136(st)}
        except Exception as e:
            row = {"i": int(idx), "state": {"kind": "exception", "error_type": str(type(e).__name__), "error": str(e)[:120]}}
        train_rows.append(row)
    try:
        stt = apply_program_state_v136(program, test_in)
        test_state: Dict[str, Any] = _state_sig_dict_v136(stt)
    except Exception as e:
        test_state = {"kind": "exception", "error_type": str(type(e).__name__), "error": str(e)[:120]}

    body = {
        "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V136),
        "kind": "arc_program_state_sig_v136",
        "stage": str(stage),
        "cc4_bg": int(bg_for_sig) if bg_for_sig is not None else None,
        "train": train_rows,
        "test": test_state,
    }
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _infer_color_mapping_v136(inp: GridV124, out: GridV124) -> Optional[Dict[str, int]]:
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


def _mode_color_v136(g: GridV124) -> int:
    counts: Dict[int, int] = {}
    for row in g:
        for x in row:
            xx = int(x)
            counts[xx] = int(counts.get(xx, 0)) + 1
    ordered = sorted(((int(k), int(counts[k])) for k in counts.keys()), key=lambda kv: (-int(kv[1]), int(kv[0])))
    return int(ordered[0][0]) if ordered else 0


def _bg_candidates_v136(grids: Sequence[GridV124]) -> Tuple[int, ...]:
    out: List[int] = [0]
    for g in grids:
        h, w = grid_shape_v124(g)
        if h > 0 and w > 0:
            out.extend([int(g[0][0]), int(g[0][w - 1]), int(g[h - 1][0]), int(g[h - 1][w - 1])])
            out.append(int(_mode_color_v136(g)))
    return tuple(int(x) for x in sorted(set(int(x) for x in out)))


def _infer_repeat_grid_steps_v136(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[ProgramStepV136]:
    from .arc_solver_v134 import _infer_repeat_grid_steps_v134

    steps_v134 = _infer_repeat_grid_steps_v134(train_pairs)
    return [ProgramStepV136(op_id=str(s.op_id), args=dict(s.args)) for s in steps_v134]


def _infer_overlay_self_translate_steps_v136(
    *, train_pairs: Sequence[Tuple[GridV124, GridV124]], bg_candidates: Tuple[int, ...]
) -> List[ProgramStepV136]:
    if not train_pairs:
        return []
    shapes_in = {grid_shape_v124(inp) for inp, _ in train_pairs}
    shapes_out = {grid_shape_v124(out) for _, out in train_pairs}
    if len(shapes_in) != 1 or len(shapes_out) != 1:
        return []
    if list(shapes_in)[0] != list(shapes_out)[0]:
        return []
    h, w = list(shapes_in)[0]
    if all(grid_equal_v124(inp, out) for inp, out in train_pairs):
        return []
    steps: List[ProgramStepV136] = []
    for pad in bg_candidates:
        for dy in range(int(-(h - 1)), int(h)):
            for dx in range(int(-(w - 1)), int(w)):
                if dx == 0 and dy == 0:
                    continue
                step = ProgramStepV136(op_id="overlay_self_translate", args={"dx": int(dx), "dy": int(dy), "pad": int(pad)})
                ok = True
                for inp, out in train_pairs:
                    got = apply_program_v136(ProgramV136(steps=(step,)), inp)
                    if not grid_equal_v124(got, out):
                        ok = False
                        break
                if ok:
                    steps.append(step)
    steps.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
    seen: Set[str] = set()
    uniq: List[ProgramStepV136] = []
    for s in steps:
        sig = sha256_hex(canonical_json_dumps(s.to_dict()).encode("utf-8"))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(s)
    return uniq


def _infer_direct_steps_v136(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> List[ProgramStepV136]:
    direct: List[ProgramStepV136] = []

    for op_id in ["rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v"]:
        ok = True
        step = ProgramStepV136(op_id=str(op_id), args={})
        for inp, out in train_pairs:
            got = apply_program_v136(ProgramV136(steps=(step,)), inp)
            if not grid_equal_v124(got, out):
                ok = False
                break
        if ok:
            direct.append(step)

    mapping: Dict[str, int] = {}
    mapping_ok = True
    for inp, out in train_pairs:
        m = _infer_color_mapping_v136(inp, out)
        if m is None:
            mapping_ok = False
            break
        for k in m.keys():
            if k in mapping and int(mapping[k]) != int(m[k]):
                mapping_ok = False
                break
            mapping[k] = int(m[k])
        if not mapping_ok:
            break
    if mapping_ok and mapping:
        ok = True
        step = ProgramStepV136(op_id="map_colors", args={"mapping": {str(k): int(mapping[k]) for k in sorted(mapping.keys())}})
        for inp, out in train_pairs:
            got = apply_program_v136(ProgramV136(steps=(step,)), inp)
            if not grid_equal_v124(got, out):
                ok = False
                break
        if ok:
            direct.append(step)

    bgs = _bg_candidates_v136([p[0] for p in train_pairs] + [test_in])
    for bg in bgs:
        shift: Optional[Tuple[int, int]] = None
        consistent = True
        for inp, out in train_pairs:
            hi, wi = grid_shape_v124(inp)
            ho, wo = grid_shape_v124(out)
            if (hi, wi) != (ho, wo):
                consistent = False
                break
            ir0, ic0, _, _ = bbox_nonzero_v124(inp, bg=int(bg))
            or0, oc0, _, _ = bbox_nonzero_v124(out, bg=int(bg))
            dy = int(or0 - ir0)
            dx = int(oc0 - ic0)
            if shift is None:
                shift = (dy, dx)
            elif shift != (dy, dx):
                consistent = False
                break
        if not consistent or shift is None:
            continue
        dy, dx = shift
        if dy == 0 and dx == 0:
            continue
        step = ProgramStepV136(op_id="translate", args={"dx": int(dx), "dy": int(dy), "pad": int(bg)})
        ok = True
        for inp, out in train_pairs:
            got = apply_program_v136(ProgramV136(steps=(step,)), inp)
            if not grid_equal_v124(got, out):
                ok = False
                break
        if ok:
            direct.append(step)

    for bg in bgs:
        ok = True
        for inp, out in train_pairs:
            got = crop_to_bbox_nonzero_v124(inp, bg=int(bg))
            if not grid_equal_v124(got, out):
                ok = False
                break
        if ok:
            direct.append(ProgramStepV136(op_id="crop_bbox_nonzero", args={"bg": int(bg)}))

    shapes_out = sorted({grid_shape_v124(out) for _, out in train_pairs})
    for h, w in shapes_out:
        for bg in bgs:
            ok = True
            for inp, out in train_pairs:
                got = pad_to_v124(inp, height=int(h), width=int(w), pad=int(bg))
                if not grid_equal_v124(got, out):
                    ok = False
                    break
            if ok:
                direct.append(ProgramStepV136(op_id="pad_to", args={"height": int(h), "width": int(w), "pad": int(bg)}))

    direct.extend(_infer_repeat_grid_steps_v136(train_pairs))
    direct.extend(_infer_overlay_self_translate_steps_v136(train_pairs=train_pairs, bg_candidates=bgs))

    direct.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
    seen: Set[str] = set()
    out_steps: List[ProgramStepV136] = []
    for s in direct:
        sig = sha256_hex(canonical_json_dumps(s.to_dict()).encode("utf-8"))
        if sig in seen:
            continue
        seen.add(sig)
        out_steps.append(s)
    return out_steps


def _infer_select_obj_args_v136(
    *, train_pairs: Sequence[Tuple[GridV124, GridV124]], bg: int, max_rank: int = 1
) -> List[Dict[str, Any]]:
    from .arc_solver_v134 import _infer_select_obj_args_v134

    return _infer_select_obj_args_v134(train_pairs=train_pairs, bg=int(bg), max_rank=int(max_rank))


@dataclass(frozen=True)
class SolveConfigV136:
    max_depth: int = 4
    max_programs: int = 4000
    trace_program_limit: int = 80


def _propose_bbox_by_color_steps_v136(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[ProgramStepV136]:
    # Sound precondition pruning: bbox_by_color must not raise on any train input (and we also
    # require it to exist on all train inputs so that "missing color" cannot be a later failure).
    # Deterministic domain: intersection of colors present in every train input.
    if not train_pairs:
        return []
    colors_all: Optional[Set[int]] = None
    for inp, _ in train_pairs:
        cs = set(int(c) for c in unique_colors_v124(inp))
        colors_all = cs if colors_all is None else (colors_all & cs)
        if not colors_all:
            return []
    assert colors_all is not None
    return [ProgramStepV136(op_id="bbox_by_color", args={"color": int(c)}) for c in sorted(colors_all)]


def _cc4_nonempty_for_all_v136(*, grids: Sequence[GridV124], bg: int) -> bool:
    for g in grids:
        try:
            oset = connected_components4_v132(g, bg=int(bg))
        except Exception:
            return False
        if not getattr(oset, "objects", None):
            return False
        if len(oset.objects) <= 0:
            return False
    return True


def _crop_bbox_nonzero_is_noop_v136(*, train_in: Sequence[GridV124], bg: int) -> bool:
    for g in train_in:
        h, w = grid_shape_v124(g)
        r0, c0, r1, c1 = bbox_nonzero_v124(g, bg=int(bg))
        if int(r0) != 0 or int(c0) != 0 or int(r1) != int(h) or int(c1) != int(w):
            return False
    return True


def _pad_to_is_noop_v136(*, train_in_shapes: Sequence[Tuple[int, int]], height: int, width: int) -> bool:
    for h, w in train_in_shapes:
        if int(h) != int(height) or int(w) != int(width):
            return False
    return True


def _propose_next_steps_v136(
    *,
    steps_so_far: Sequence[ProgramStepV136],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    bg_candidates: Tuple[int, ...],
    shapes_out: Tuple[Tuple[int, int], ...],
    palette_out: Tuple[int, ...],
    direct_steps: Sequence[ProgramStepV136],
) -> List[ProgramStepV136]:
    avail = _abstract_slots_after_steps_v136(steps_so_far)
    if bool(avail.get("invalid", False)):
        return []
    out: List[ProgramStepV136] = []

    if not bool(avail.get("objset")) and not bool(avail.get("obj")) and not bool(avail.get("bbox")) and not bool(avail.get("patch")):
        out.extend(list(direct_steps))
        out.extend(_propose_bbox_by_color_steps_v136(train_pairs=train_pairs))
        train_in = [p[0] for p in train_pairs] + [test_in]
        train_in_shapes = [grid_shape_v124(g) for g in train_in]
        for bg in bg_candidates:
            if not _crop_bbox_nonzero_is_noop_v136(train_in=train_in, bg=int(bg)):
                out.append(ProgramStepV136(op_id="crop_bbox_nonzero", args={"bg": int(bg)}))
        if shapes_out:
            for h, w in shapes_out:
                for bg in bg_candidates:
                    # Skip pad_to no-ops (cannot be part of a minimal program).
                    if not _pad_to_is_noop_v136(train_in_shapes=train_in_shapes, height=int(h), width=int(w)):
                        out.append(ProgramStepV136(op_id="pad_to", args={"height": int(h), "width": int(w), "pad": int(bg)}))
                    out.append(ProgramStepV136(op_id="new_canvas", args={"height": int(h), "width": int(w), "color": int(bg)}))
        for bg in bg_candidates:
            if _cc4_nonempty_for_all_v136(grids=train_in, bg=int(bg)):
                out.append(ProgramStepV136(op_id="cc4", args={"bg": int(bg), "colors": []}))

    elif bool(avail.get("objset")) and not bool(avail.get("obj")):
        bg = _last_cc4_bg_v136(steps_so_far)
        bg = int(bg) if bg is not None else int(bg_candidates[0] if bg_candidates else 0)
        inferred = _infer_select_obj_args_v136(train_pairs=train_pairs, bg=int(bg), max_rank=1)
        if not inferred:
            inferred = [
                {"key": "area", "order": "max", "rank": 0, "color_filter": None},
                {"key": "area", "order": "min", "rank": 0, "color_filter": None},
                {"key": "dist_center", "order": "min", "rank": 0, "color_filter": None},
                {"key": "left", "order": "min", "rank": 0, "color_filter": None},
                {"key": "top", "order": "min", "rank": 0, "color_filter": None},
            ]
        for a in inferred:
            out.append(ProgramStepV136(op_id="select_obj", args=dict(a)))

    elif bool(avail.get("obj")) and not bool(avail.get("bbox")):
        out.append(ProgramStepV136(op_id="obj_bbox", args={}))

    elif bool(avail.get("bbox")) and not bool(avail.get("patch")):
        out.append(ProgramStepV136(op_id="crop_bbox", args={}))
        for c in palette_out:
            out.append(ProgramStepV136(op_id="paint_rect", args={"color": int(c)}))
            out.append(ProgramStepV136(op_id="draw_rect_border", args={"color": int(c), "thickness": 1}))

    elif bool(avail.get("patch")):
        out.append(ProgramStepV136(op_id="commit_patch", args={}))
        positions: Set[Tuple[int, int]] = {(0, 0)}
        from .arc_solver_v134 import _changed_cells_v134

        for inp, outg in train_pairs[:2]:
            cm = _changed_cells_v134(inp, outg)
            if cm:
                rs = [int(r) for r, _ in cm]
                cs = [int(c) for _, c in cm]
                positions.add((int(min(rs)), int(min(cs))))
        for top, left in sorted(positions):
            out.append(ProgramStepV136(op_id="paste", args={"top": int(top), "left": int(left), "transparent": 0}))

    out.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
    seen: Set[str] = set()
    uniq: List[ProgramStepV136] = []
    for s in out:
        ss = sha256_hex(canonical_json_dumps(s.to_dict()).encode("utf-8"))
        if ss in seen:
            continue
        seen.add(ss)
        uniq.append(s)
    return uniq


@dataclass(frozen=True)
class _EvalInfoV136:
    ok_train: bool
    loss: Tuple[int, int]
    mismatch_ex: Optional[Dict[str, Any]]
    got_shapes: Tuple[Tuple[int, int], ...]
    got_palettes: Tuple[Tuple[int, ...], ...]


def solve_arc_task_v136(
    *, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124, config: Optional[SolveConfigV136] = None
) -> Dict[str, Any]:
    cfg = config or SolveConfigV136()
    max_depth = int(cfg.max_depth)
    max_programs = int(cfg.max_programs)
    trace_program_limit = int(cfg.trace_program_limit)

    _validate_grid_values_v136(test_in)
    for inp, out in train_pairs:
        _validate_grid_values_v136(inp)
        _validate_grid_values_v136(out)

    all_grids = [test_in] + [p[0] for p in train_pairs] + [p[1] for p in train_pairs]
    bg_candidates = _bg_candidates_v136(all_grids)
    palette_all = sorted(set(int(c) for g in all_grids for c in unique_colors_v124(g)))
    palette_out = sorted(set(int(c) for _, out in train_pairs for c in unique_colors_v124(out)))
    shapes_out = sorted(set((int(h), int(w)) for _, out in train_pairs for h, w in [grid_shape_v124(out)]))

    direct_steps = _infer_direct_steps_v136(train_pairs=train_pairs, test_in=test_in)

    concept_trace: Dict[str, Any] = {
        "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V136),
        "kind": "concept_trace_v136",
        "bg_candidates": [int(x) for x in bg_candidates],
        "palette_all": [int(x) for x in palette_all],
        "palette_out": [int(x) for x in palette_out],
        "shapes_out": [{"h": int(h), "w": int(w)} for h, w in shapes_out],
        "direct_steps_count": int(len(direct_steps)),
    }
    try:
        for inp, _ in train_pairs[:3]:
            oset = connected_components4_v132(inp, bg=0)
            concept_trace.setdefault("obj_summaries_bg0", []).append({"count": int(len(oset.objects))})
    except Exception:
        pass

    want_shapes = [grid_shape_v124(out) for _, out in train_pairs]
    want_palettes = [set(int(c) for c in unique_colors_v124(out)) for _, out in train_pairs]

    eval_cache: Dict[str, _EvalInfoV136] = {}
    state_sig_cache: Dict[str, str] = {}

    def eval_program(program: ProgramV136) -> _EvalInfoV136:
        psig = program.program_sig()
        cached = eval_cache.get(psig)
        if cached is not None:
            return cached
        shape_mismatch = 0
        diff_cells = 0
        mismatch_ex: Optional[Dict[str, Any]] = None
        ok_all = True
        got_shapes: List[Tuple[int, int]] = []
        got_palettes: List[Tuple[int, ...]] = []
        for inp, want in train_pairs:
            try:
                got = apply_program_v136(program, inp)
            except Exception as e:
                ok_all = False
                shape_mismatch += 1
                diff_cells += 100000
                got_shapes.append(grid_shape_v124(inp))
                got_palettes.append(tuple(sorted(set(int(c) for c in unique_colors_v124(inp)))))
                if mismatch_ex is None:
                    mismatch_ex = {"kind": "exception", "error_type": str(type(e).__name__), "error": str(e)[:200]}
                continue
            got_shapes.append(grid_shape_v124(got))
            got_palettes.append(tuple(sorted(set(int(c) for c in unique_colors_v124(got)))))
            if not grid_equal_v124(got, want):
                ok_all = False
                mm = _summarize_mismatch_v136(got=got, want=want)
                if mm.get("kind") == "shape_mismatch":
                    shape_mismatch += 1
                    diff_cells += 100000
                else:
                    diff_cells += int(mm.get("diff_cells") or 0)
                if mismatch_ex is None:
                    mismatch_ex = mm
        info = _EvalInfoV136(
            ok_train=bool(ok_all),
            loss=(int(shape_mismatch), int(diff_cells)),
            mismatch_ex=mismatch_ex,
            got_shapes=tuple((int(h), int(w)) for h, w in got_shapes),
            got_palettes=tuple(tuple(int(x) for x in p) for p in got_palettes),
        )
        eval_cache[psig] = info
        return info

    start = ProgramV136(steps=tuple())
    start_sig = start.program_sig()
    heap: List[Tuple[int, Tuple[int, int], int, str, ProgramV136]] = []
    loss0 = eval_program(start).loss
    heapq.heappush(heap, (0, loss0, 0, start_sig, start))

    seen: Set[str] = {start_sig}
    best_cost_by_state_depth: Dict[str, Dict[int, int]] = {}
    tried = 0
    trace_programs: List[Dict[str, Any]] = []

    best_cost: Optional[int] = None
    best_programs: List[ProgramV136] = []

    pruned_by_shape = 0
    pruned_by_palette = 0
    pruned_by_no_grid_modify = 0
    pruned_by_dominated_state = 0

    # Seed state dominance table with the empty program state (depth=0, cost=0).
    try:
        start_avail = _abstract_slots_after_steps_v136(list(start.steps))
        start_stage = _stage_from_avail_v136(start_avail)
        start_cc4_bg = _last_cc4_bg_v136(list(start.steps))
        ssig = state_sig_cache.get(start_sig)
        if ssig is None:
            ssig = _program_state_sig_v136(
                program=start,
                train_in=[p[0] for p in train_pairs],
                test_in=test_in,
                stage=str(start_stage),
                last_cc4_bg=start_cc4_bg,
            )
            state_sig_cache[start_sig] = ssig
        best_cost_by_state_depth.setdefault(str(ssig), {})[0] = 0
    except Exception:
        # If state sig computation fails, skip dominance pruning for this task.
        best_cost_by_state_depth = {}

    while heap:
        cost_bits, loss, depth, psig, program = heapq.heappop(heap)
        tried += 1

        if tried <= trace_program_limit:
            info = eval_program(program)
            trace_programs.append(
                {
                    "program_sig": str(psig),
                    "cost_bits": int(cost_bits),
                    "depth": int(depth),
                    "ok_train": bool(info.ok_train),
                    "mismatch": info.mismatch_ex,
                    "steps": [s.to_dict() for s in program.steps],
                }
            )

        if best_cost is not None and int(cost_bits) > int(best_cost):
            break

        info = eval_program(program)
        if info.ok_train:
            if best_cost is None or int(cost_bits) < int(best_cost):
                best_cost = int(cost_bits)
                best_programs = [program]
            elif int(cost_bits) == int(best_cost):
                best_programs.append(program)
            continue

        if depth >= max_depth:
            continue
        if tried >= max_programs:
            break

        next_steps = _propose_next_steps_v136(
            steps_so_far=list(program.steps),
            train_pairs=train_pairs,
            test_in=test_in,
            bg_candidates=bg_candidates,
            shapes_out=tuple(shapes_out),
            palette_out=tuple(palette_out),
            direct_steps=direct_steps,
        )
        for st in next_steps:
            new_steps = tuple(list(program.steps) + [st])
            new_prog = ProgramV136(steps=new_steps)
            new_sig = new_prog.program_sig()
            if new_sig in seen:
                continue
            new_cost = int(cost_bits) + int(step_cost_bits_v135(op_id=str(st.op_id), args=dict(st.args)))
            avail = _abstract_slots_after_steps_v136(new_steps)
            if bool(avail.get("invalid", False)):
                continue

            depth2 = int(depth) + 1
            steps_left = int(max_depth - depth2)
            stage = _stage_from_avail_v136(avail)

            info2 = eval_program(new_prog)

            # Formal pruning: if we cannot change the grid anymore and still mismatch -> prune.
            if not info2.ok_train and int(steps_left) < int(_min_steps_to_grid_modify_v136(stage)):
                pruned_by_no_grid_modify += 1
                continue

            # Shape reachability pruning: if any train pair has wrong shape and it cannot be fixed in time -> prune.
            shape_ok = True
            for (gh, gw), (wh, ww) in zip(info2.got_shapes, want_shapes):
                if not _can_reach_shape_v136(
                    stage=str(stage), got_shape=(int(gh), int(gw)), want_shape=(int(wh), int(ww)), steps_left=int(steps_left)
                ):
                    shape_ok = False
                    break
            if not shape_ok:
                pruned_by_shape += 1
                continue

            # Palette reachability pruning (minimal, sound): if palette differs and we cannot modify grid in time -> prune.
            palette_ok = True
            for got_p, want_p in zip(info2.got_palettes, want_palettes):
                if not _can_reach_palette_v136(
                    stage=str(stage),
                    got_palette=set(int(x) for x in got_p),
                    want_palette=set(int(x) for x in want_p),
                    steps_left=int(steps_left),
                ):
                    palette_ok = False
                    break
            if not palette_ok:
                pruned_by_palette += 1
                continue

            # Dominance pruning by concrete state signature (sound): if we already reached the
            # same state with <= depth and <= cost, this program cannot be part of a minimal solution.
            if best_cost_by_state_depth:
                cc4_bg = _last_cc4_bg_v136(new_steps)
                ps = state_sig_cache.get(str(new_sig))
                if ps is None:
                    ps = _program_state_sig_v136(
                        program=new_prog,
                        train_in=[p[0] for p in train_pairs],
                        test_in=test_in,
                        stage=str(stage),
                        last_cc4_bg=cc4_bg,
                    )
                    state_sig_cache[str(new_sig)] = ps
                dom = best_cost_by_state_depth.get(str(ps), {})
                dominated = False
                for d0, c0 in dom.items():
                    if int(d0) <= int(depth2) and int(c0) <= int(new_cost):
                        dominated = True
                        break
                if dominated:
                    pruned_by_dominated_state += 1
                    continue
                prev = dom.get(int(depth2))
                if prev is None or int(new_cost) < int(prev):
                    dom[int(depth2)] = int(new_cost)
                    best_cost_by_state_depth[str(ps)] = dom

            seen.add(new_sig)
            heapq.heappush(heap, (int(new_cost), info2.loss, depth2, str(new_sig), new_prog))

    if best_cost is not None and best_programs:
        out_by_prog: Dict[str, str] = {}
        out_hashes: List[str] = []
        for p in best_programs:
            ph = p.program_sig()
            gout = apply_program_v136(p, test_in)
            gh = grid_hash_v124(gout)
            out_by_prog[str(ph)] = str(gh)
            out_hashes.append(str(gh))
        uniq = sorted(set(out_hashes))
        if len(uniq) == 1:
            predicted = apply_program_v136(best_programs[0], test_in)
            return {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V136),
                "kind": "arc_solver_result_v136",
                "status": "SOLVED",
                "failure_reason": None,
                "program_sig": str(best_programs[0].program_sig()),
                "program_cost_bits": int(best_cost),
                "predicted_grid": [list(r) for r in predicted],
                "predicted_grid_hash": str(grid_hash_v124(predicted)),
                "trace": {
                    "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V136),
                    "kind": "arc_trace_v136",
                    "concept_trace": concept_trace,
                    "trace_programs": trace_programs,
                    "tried": int(tried),
                    "max_programs": int(max_programs),
                    "max_depth": int(max_depth),
                    "min_cost_solutions": int(len(best_programs)),
                    "pruned_by_shape_reachability": int(pruned_by_shape),
                    "pruned_by_palette_reachability": int(pruned_by_palette),
                    "pruned_by_no_grid_modify_in_time": int(pruned_by_no_grid_modify),
                    "pruned_by_dominated_state": int(pruned_by_dominated_state),
                },
            }
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V136),
            "kind": "arc_solver_result_v136",
            "status": "UNKNOWN",
            "failure_reason": {
                "kind": "AMBIGUOUS_RULE",
                "details": {"min_cost_solutions": int(len(best_programs)), "predicted_grid_hashes": uniq},
            },
            "program_sig": "",
            "program_cost_bits": int(best_cost),
            "predicted_grid_hash": "",
            "predicted_grid_hash_by_solution": {str(k): str(out_by_prog[k]) for k in sorted(out_by_prog.keys())},
            "trace": {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V136),
                "kind": "arc_trace_v136",
                "concept_trace": concept_trace,
                "trace_programs": trace_programs,
                "tried": int(tried),
                "max_programs": int(max_programs),
                "max_depth": int(max_depth),
                "min_cost_solutions": int(len(best_programs)),
                "pruned_by_shape_reachability": int(pruned_by_shape),
                "pruned_by_palette_reachability": int(pruned_by_palette),
                "pruned_by_no_grid_modify_in_time": int(pruned_by_no_grid_modify),
                "pruned_by_dominated_state": int(pruned_by_dominated_state),
            },
        }

    if tried >= max_programs:
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V136),
            "kind": "arc_solver_result_v136",
            "status": "FAIL",
            "failure_reason": {
                "kind": "SEARCH_BUDGET_EXCEEDED",
                "details": {"candidates_tested": int(tried), "max_programs": int(max_programs), "max_depth": int(max_depth)},
            },
            "program_sig": "",
            "program_cost_bits": 0,
            "predicted_grid_hash": "",
            "trace": {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V136),
                "kind": "arc_trace_v136",
                "concept_trace": concept_trace,
                "trace_programs": trace_programs,
                "tried": int(tried),
                "max_programs": int(max_programs),
                "max_depth": int(max_depth),
                "min_cost_solutions": 0,
                "pruned_by_shape_reachability": int(pruned_by_shape),
                "pruned_by_palette_reachability": int(pruned_by_palette),
                "pruned_by_no_grid_modify_in_time": int(pruned_by_no_grid_modify),
                "pruned_by_dominated_state": int(pruned_by_dominated_state),
            },
        }

    return {
        "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V136),
        "kind": "arc_solver_result_v136",
        "status": "FAIL",
        "failure_reason": {
            "kind": "MISSING_OPERATOR",
            "details": {"search_exhausted": True, "candidates_tested": int(tried), "max_depth": int(max_depth)},
        },
        "program_sig": "",
        "program_cost_bits": 0,
        "predicted_grid_hash": "",
        "trace": {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V136),
            "kind": "arc_trace_v136",
            "concept_trace": concept_trace,
            "trace_programs": trace_programs,
            "tried": int(tried),
            "max_programs": int(max_programs),
            "max_depth": int(max_depth),
            "min_cost_solutions": 0,
            "pruned_by_shape_reachability": int(pruned_by_shape),
            "pruned_by_palette_reachability": int(pruned_by_palette),
            "pruned_by_no_grid_modify_in_time": int(pruned_by_no_grid_modify),
            "pruned_by_dominated_state": int(pruned_by_dominated_state),
        },
    }
