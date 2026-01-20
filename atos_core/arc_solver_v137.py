from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_objects_v132 import connected_components4_v132
from .arc_ops_v132 import StateV132
from .arc_ops_v137 import OP_DEFS_V137, apply_op_v137, step_cost_bits_v137
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

ARC_SOLVER_SCHEMA_VERSION_V137 = 137


def _validate_grid_values_v137(g: GridV124) -> None:
    for row in g:
        for x in row:
            xx = int(x)
            if xx < 0 or xx > 9:
                raise ValueError("grid_cell_out_of_range")


@dataclass(frozen=True)
class ProgramStepV137:
    op_id: str
    args: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        a: Dict[str, Any] = {}
        for k in sorted(self.args.keys()):
            a[str(k)] = self.args[k]
        return {"op_id": str(self.op_id), "args": a}


@dataclass(frozen=True)
class ProgramV137:
    steps: Tuple[ProgramStepV137, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V137),
            "kind": "arc_program_v137",
            "steps": [s.to_dict() for s in self.steps],
        }

    def program_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


def _summarize_mismatch_v137(*, got: GridV124, want: GridV124) -> Dict[str, Any]:
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


def _abstract_slots_after_steps_v137(steps: Sequence[ProgramStepV137]) -> Dict[str, bool]:
    avail: Dict[str, bool] = {"grid": True, "objset": False, "obj": False, "bbox": False, "patch": False}
    for st in steps:
        od = OP_DEFS_V137.get(str(st.op_id))
        if od is None:
            continue
        for r in od.reads:
            if not bool(avail.get(str(r), False)):
                return {"grid": True, "objset": False, "obj": False, "bbox": False, "patch": False, "invalid": True}
        for w in od.writes:
            avail[str(w)] = True
        # Model known invalidations (must match apply_op semantics).
        if str(st.op_id) == "commit_patch":
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
            "propagate_color_translate",
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


def _stage_from_avail_v137(avail: Dict[str, bool]) -> str:
    if bool(avail.get("patch")):
        return "patch"
    if bool(avail.get("bbox")):
        return "bbox"
    if bool(avail.get("obj")):
        return "obj"
    if bool(avail.get("objset")):
        return "objset"
    return "grid"


def _min_steps_to_grid_stage_v137(stage: str) -> int:
    s = str(stage)
    if s == "grid":
        return 0
    if s == "patch":
        return 1
    if s == "bbox":
        return 2
    if s == "obj":
        return 3
    if s == "objset":
        return 4
    return 99


def _min_steps_to_grid_modify_v137(stage: str) -> int:
    s = str(stage)
    if s == "grid":
        return 1
    if s == "bbox":
        return 1
    if s == "patch":
        return 1
    if s == "obj":
        return 2
    if s == "objset":
        return 3
    return 99


def _min_steps_to_shape_change_v137(stage: str, direction: str) -> int:
    s = str(stage)
    d = str(direction)
    if d == "grow":
        return int(_min_steps_to_grid_stage_v137(s)) + 1
    if d == "shrink":
        if s == "patch":
            return 1
        if s == "bbox":
            return 2
        if s == "obj":
            return 3
        if s == "objset":
            return 4
        return 1
    if d == "mixed":
        return int(_min_steps_to_grid_stage_v137(s)) + 2
    return 0


def _can_reach_shape_v137(
    *, stage: str, got_shape: Tuple[int, int], want_shape: Tuple[int, int], steps_left: int
) -> bool:
    hg, wg = int(got_shape[0]), int(got_shape[1])
    hw, ww = int(want_shape[0]), int(want_shape[1])
    if (hg, wg) == (hw, ww):
        return True
    if int(steps_left) <= 0:
        return False
    if hw >= hg and ww >= wg:
        needed = _min_steps_to_shape_change_v137(stage, "grow")
        return int(steps_left) >= int(needed)
    if hw <= hg and ww <= wg:
        needed = _min_steps_to_shape_change_v137(stage, "shrink")
        return int(steps_left) >= int(needed)
    needed = _min_steps_to_shape_change_v137(stage, "mixed")
    return int(steps_left) >= int(needed)


def _bg_candidates_v137(grids: Sequence[GridV124]) -> Tuple[int, ...]:
    from .arc_solver_v134 import _bg_candidates_v134

    return _bg_candidates_v134(grids)


def _infer_color_mapping_v137(inp: GridV124, out: GridV124) -> Optional[Dict[str, int]]:
    from .arc_solver_v134 import _infer_color_mapping_v134

    return _infer_color_mapping_v134(inp, out)


def _infer_repeat_grid_steps_v137(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[ProgramStepV137]:
    from .arc_solver_v134 import _infer_repeat_grid_steps_v134

    steps_v134 = _infer_repeat_grid_steps_v134(train_pairs)
    out: List[ProgramStepV137] = []
    for s in steps_v134:
        out.append(ProgramStepV137(op_id=str(s.op_id), args=dict(s.args)))
    return out


def _infer_overlay_self_translate_steps_v137(
    *, train_pairs: Sequence[Tuple[GridV124, GridV124]], bg_candidates: Sequence[int]
) -> List[ProgramStepV137]:
    from .arc_solver_v135 import _infer_overlay_self_translate_steps_v135

    steps_v135 = _infer_overlay_self_translate_steps_v135(train_pairs=train_pairs, bg_candidates=tuple(int(x) for x in bg_candidates))
    out: List[ProgramStepV137] = []
    for s in steps_v135:
        out.append(ProgramStepV137(op_id=str(s.op_id), args=dict(s.args)))
    return out


def _infer_propagate_color_translate_steps_v137(
    *, train_pairs: Sequence[Tuple[GridV124, GridV124]], bg_candidates: Sequence[int]
) -> List[ProgramStepV137]:
    # Deterministic small-domain inference: propose (color,dx,dy,pad) that exactly matches all train pairs.
    if not train_pairs:
        return []
    shapes = [grid_shape_v124(inp) for inp, _ in train_pairs]
    if not shapes:
        return []
    max_h = max(int(h) for h, _ in shapes)
    max_w = max(int(w) for _, w in shapes)
    max_shift = min(max(max_h, max_w) - 1, 6)
    if max_shift <= 0:
        return []

    palette_out: Set[int] = set()
    for _, out in train_pairs:
        palette_out |= set(int(c) for c in unique_colors_v124(out))
    colors = sorted(palette_out)

    def shift_order() -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for s in range(1, max_shift + 1):
            for dy in range(-s, s + 1):
                for dx in range(-s, s + 1):
                    if abs(dx) + abs(dy) != s:
                        continue
                    pairs.append((int(dx), int(dy)))
        return pairs

    shifts = shift_order()
    steps: List[ProgramStepV137] = []
    for pad in bg_candidates:
        for color in colors:
            if int(color) == int(pad):
                continue
            for dx, dy in shifts:
                step = ProgramStepV137(
                    op_id="propagate_color_translate",
                    args={"dx": int(dx), "dy": int(dy), "color": int(color), "pad": int(pad)},
                )
                ok = True
                prog = ProgramV137(steps=(step,))
                for inp, out in train_pairs:
                    got = apply_program_v137(prog, inp)
                    if not grid_equal_v124(got, out):
                        ok = False
                        break
                if ok:
                    steps.append(step)
    steps.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
    seen: Set[str] = set()
    uniq: List[ProgramStepV137] = []
    for s in steps:
        sig = sha256_hex(canonical_json_dumps(s.to_dict()).encode("utf-8"))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(s)
    return uniq


def _infer_direct_steps_v137(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> List[ProgramStepV137]:
    direct: List[ProgramStepV137] = []

    for op_id in ["rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v"]:
        ok = True
        step = ProgramStepV137(op_id=str(op_id), args={})
        for inp, out in train_pairs:
            got = apply_program_v137(ProgramV137(steps=(step,)), inp)
            if not grid_equal_v124(got, out):
                ok = False
                break
        if ok:
            direct.append(step)

    mapping: Dict[str, int] = {}
    mapping_ok = True
    for inp, out in train_pairs:
        m = _infer_color_mapping_v137(inp, out)
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
        step = ProgramStepV137(op_id="map_colors", args={"mapping": {str(k): int(mapping[k]) for k in sorted(mapping.keys())}})
        for inp, out in train_pairs:
            got = apply_program_v137(ProgramV137(steps=(step,)), inp)
            if not grid_equal_v124(got, out):
                ok = False
                break
        if ok:
            direct.append(step)

    bgs = _bg_candidates_v137([p[0] for p in train_pairs] + [test_in])
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
        step = ProgramStepV137(op_id="translate", args={"dx": int(dx), "dy": int(dy), "pad": int(bg)})
        ok = True
        for inp, out in train_pairs:
            got = apply_program_v137(ProgramV137(steps=(step,)), inp)
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
            direct.append(ProgramStepV137(op_id="crop_bbox_nonzero", args={"bg": int(bg)}))

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
                direct.append(ProgramStepV137(op_id="pad_to", args={"height": int(h), "width": int(w), "pad": int(bg)}))

    direct.extend(_infer_repeat_grid_steps_v137(train_pairs))
    direct.extend(_infer_overlay_self_translate_steps_v137(train_pairs=train_pairs, bg_candidates=bgs))
    direct.extend(_infer_propagate_color_translate_steps_v137(train_pairs=train_pairs, bg_candidates=bgs))

    direct.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
    seen: Set[str] = set()
    out_steps: List[ProgramStepV137] = []
    for s in direct:
        sig = sha256_hex(canonical_json_dumps(s.to_dict()).encode("utf-8"))
        if sig in seen:
            continue
        seen.add(sig)
        out_steps.append(s)
    return out_steps


def _infer_select_obj_args_v137(
    *, train_pairs: Sequence[Tuple[GridV124, GridV124]], bg: int, max_rank: int = 1
) -> List[Dict[str, Any]]:
    from .arc_solver_v134 import _infer_select_obj_args_v134

    return _infer_select_obj_args_v134(train_pairs=train_pairs, bg=int(bg), max_rank=int(max_rank))


@dataclass(frozen=True)
class SolveConfigV137:
    max_depth: int = 4
    max_programs: int = 4000
    trace_program_limit: int = 80


def _propose_bbox_by_color_steps_v137(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[ProgramStepV137]:
    if not train_pairs:
        return []
    colors_all: Optional[Set[int]] = None
    for inp, _ in train_pairs:
        cs = set(int(c) for c in unique_colors_v124(inp))
        colors_all = cs if colors_all is None else (colors_all & cs)
        if not colors_all:
            return []
    assert colors_all is not None
    return [ProgramStepV137(op_id="bbox_by_color", args={"color": int(c)}) for c in sorted(colors_all)]


def _cc4_nonempty_for_all_v137(*, grids: Sequence[GridV124], bg: int) -> bool:
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


def _crop_bbox_nonzero_is_noop_v137(*, train_in: Sequence[GridV124], bg: int) -> bool:
    for g in train_in:
        h, w = grid_shape_v124(g)
        r0, c0, r1, c1 = bbox_nonzero_v124(g, bg=int(bg))
        if int(r0) != 0 or int(c0) != 0 or int(r1) != int(h) or int(c1) != int(w):
            return False
    return True


def _pad_to_is_noop_v137(*, train_in_shapes: Sequence[Tuple[int, int]], height: int, width: int) -> bool:
    for h, w in train_in_shapes:
        if int(h) != int(height) or int(w) != int(width):
            return False
    return True


def _last_cc4_bg_v137(steps: Sequence[ProgramStepV137]) -> Optional[int]:
    for st in reversed(list(steps)):
        if str(st.op_id) == "cc4":
            a = dict(st.args)
            bg = a.get("bg")
            return int(bg) if bg is not None else 0
    return None


def _propose_next_steps_v137(
    *,
    steps_so_far: Sequence[ProgramStepV137],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    bg_candidates: Tuple[int, ...],
    shapes_out: Tuple[Tuple[int, int], ...],
    palette_out: Tuple[int, ...],
    direct_steps: Sequence[ProgramStepV137],
) -> List[ProgramStepV137]:
    avail = _abstract_slots_after_steps_v137(steps_so_far)
    if bool(avail.get("invalid", False)):
        return []
    out: List[ProgramStepV137] = []

    if not bool(avail.get("objset")) and not bool(avail.get("obj")) and not bool(avail.get("bbox")) and not bool(avail.get("patch")):
        out.extend(list(direct_steps))
        out.extend(_propose_bbox_by_color_steps_v137(train_pairs=train_pairs))
        train_in = [p[0] for p in train_pairs] + [test_in]
        train_in_shapes = [grid_shape_v124(g) for g in train_in]
        for bg in bg_candidates:
            if not _crop_bbox_nonzero_is_noop_v137(train_in=train_in, bg=int(bg)):
                out.append(ProgramStepV137(op_id="crop_bbox_nonzero", args={"bg": int(bg)}))
        if shapes_out:
            for h, w in shapes_out:
                for bg in bg_candidates:
                    if not _pad_to_is_noop_v137(train_in_shapes=train_in_shapes, height=int(h), width=int(w)):
                        out.append(ProgramStepV137(op_id="pad_to", args={"height": int(h), "width": int(w), "pad": int(bg)}))
                    out.append(ProgramStepV137(op_id="new_canvas", args={"height": int(h), "width": int(w), "color": int(bg)}))
        for bg in bg_candidates:
            if _cc4_nonempty_for_all_v137(grids=train_in, bg=int(bg)):
                out.append(ProgramStepV137(op_id="cc4", args={"bg": int(bg), "colors": []}))

    elif bool(avail.get("objset")) and not bool(avail.get("obj")):
        bg = _last_cc4_bg_v137(steps_so_far)
        bg = int(bg) if bg is not None else int(bg_candidates[0] if bg_candidates else 0)
        inferred = _infer_select_obj_args_v137(train_pairs=train_pairs, bg=int(bg), max_rank=1)
        if not inferred:
            inferred = [
                {"key": "area", "order": "max", "rank": 0, "color_filter": None},
                {"key": "area", "order": "min", "rank": 0, "color_filter": None},
                {"key": "dist_center", "order": "min", "rank": 0, "color_filter": None},
                {"key": "left", "order": "min", "rank": 0, "color_filter": None},
                {"key": "top", "order": "min", "rank": 0, "color_filter": None},
            ]
        for a in inferred:
            out.append(ProgramStepV137(op_id="select_obj", args=dict(a)))

    elif bool(avail.get("obj")) and not bool(avail.get("bbox")):
        out.append(ProgramStepV137(op_id="obj_bbox", args={}))

    elif bool(avail.get("bbox")) and not bool(avail.get("patch")):
        out.append(ProgramStepV137(op_id="crop_bbox", args={}))
        for c in palette_out:
            out.append(ProgramStepV137(op_id="paint_rect", args={"color": int(c)}))
            out.append(ProgramStepV137(op_id="draw_rect_border", args={"color": int(c), "thickness": 1}))

    elif bool(avail.get("patch")):
        out.append(ProgramStepV137(op_id="commit_patch", args={}))
        positions: Set[Tuple[int, int]] = {(0, 0)}
        from .arc_solver_v134 import _changed_cells_v134

        for inp, outg in train_pairs[:2]:
            cm = _changed_cells_v134(inp, outg)
            if cm:
                rs = [int(r) for r, _ in cm]
                cs = [int(c) for _, c in cm]
                positions.add((int(min(rs)), int(min(cs))))
        for top, left in sorted(positions):
            out.append(ProgramStepV137(op_id="paste", args={"top": int(top), "left": int(left), "transparent": 0}))

    out.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
    seen: Set[str] = set()
    uniq: List[ProgramStepV137] = []
    for s in out:
        ss = sha256_hex(canonical_json_dumps(s.to_dict()).encode("utf-8"))
        if ss in seen:
            continue
        seen.add(ss)
        uniq.append(s)
    return uniq


@dataclass(frozen=True)
class _EvalInfoV137:
    ok_train: bool
    loss: Tuple[int, int]
    mismatch_ex: Optional[Dict[str, Any]]


def _eval_on_train_v137(*, prog: ProgramV137, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> _EvalInfoV137:
    diff_sum = 0
    shape_pen = 0
    mismatch_ex: Optional[Dict[str, Any]] = None
    for inp, out in train_pairs:
        try:
            got = apply_program_v137(prog, inp)
        except Exception as e:
            return _EvalInfoV137(ok_train=False, loss=(10**9, 10**9), mismatch_ex={"kind": "exception", "error": str(e)})
        mm = _summarize_mismatch_v137(got=got, want=out)
        if mm["kind"] == "shape_mismatch":
            shape_pen += 100000
        else:
            diff_sum += int(mm.get("diff_cells") or 0)
        if mismatch_ex is None and (mm["kind"] != "cell_mismatch" or int(mm.get("diff_cells") or 0) != 0):
            mismatch_ex = mm
    ok = diff_sum == 0 and shape_pen == 0
    return _EvalInfoV137(ok_train=ok, loss=(int(shape_pen), int(diff_sum)), mismatch_ex=mismatch_ex)


def _apply_step_v137(state: StateV132, step: ProgramStepV137) -> StateV132:
    return apply_op_v137(state=state, op_id=str(step.op_id), args=dict(step.args))


def apply_program_v137(prog: ProgramV137, grid: GridV124) -> GridV124:
    state = StateV132(grid=grid)
    for step in prog.steps:
        state = _apply_step_v137(state, step)
    return state.grid


def _program_cost_bits_v137(steps: Sequence[ProgramStepV137]) -> int:
    total = 0
    for st in steps:
        total += int(step_cost_bits_v137(op_id=str(st.op_id), args=dict(st.args)))
    return int(total)


def _state_sig_v137(*, stage: str, state: StateV132, cc4_bg: Optional[int]) -> str:
    # Sound state signature for dominance pruning.
    h = {"grid_hash": grid_hash_v124(state.grid)}
    st = str(stage)
    if st in {"objset", "obj", "bbox", "patch"} and cc4_bg is not None:
        h["cc4_bg"] = int(cc4_bg)
    if state.bbox is not None:
        h["bbox"] = state.bbox.to_dict()
    if state.patch is not None:
        h["patch_hash"] = grid_hash_v124(state.patch)
    if state.obj is not None:
        h["obj_sig"] = str(state.obj.object_sig())
    if state.objset is not None:
        h["objset_sig"] = str(state.objset.set_sig())
    return sha256_hex(canonical_json_dumps(h).encode("utf-8"))


def solve_arc_task_v137(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124, config: SolveConfigV137) -> Dict[str, Any]:
    # Validate input invariants first (fail-closed).
    for inp, out in train_pairs:
        _validate_grid_values_v137(inp)
        _validate_grid_values_v137(out)
    _validate_grid_values_v137(test_in)

    bgs = _bg_candidates_v137([p[0] for p in train_pairs] + [test_in])
    shapes_out = tuple(sorted({grid_shape_v124(out) for _, out in train_pairs}))
    palette_out_set: Set[int] = set()
    for _, out in train_pairs:
        palette_out_set |= set(int(c) for c in unique_colors_v124(out))
    palette_out = tuple(sorted(palette_out_set))

    direct_steps = _infer_direct_steps_v137(train_pairs=train_pairs, test_in=test_in)

    max_depth = int(config.max_depth)
    max_programs = int(config.max_programs)
    trace_program_limit = int(config.trace_program_limit)

    # priority queue: (cost_bits, train_loss_tuple, depth, program_sig, steps)
    heap: List[Tuple[Tuple[int, int, int, str], Tuple[ProgramStepV137, ...]]] = []
    start = tuple()
    start_prog = ProgramV137(steps=start)
    start_cost = _program_cost_bits_v137(start)
    start_eval = _eval_on_train_v137(prog=start_prog, train_pairs=train_pairs)
    heapq.heappush(heap, ((int(start_cost), int(start_eval.loss[0]), int(start_eval.loss[1]), start_prog.program_sig()), start))

    best_by_state: Dict[str, Tuple[int, int]] = {}
    trace_programs: List[Dict[str, Any]] = []
    tried = 0
    min_cost_solutions: List[ProgramV137] = []
    min_cost_bits: Optional[int] = None

    pruned_by_shape_reachability = 0
    pruned_by_palette_reachability = 0  # reserved for future; must remain sound
    pruned_by_dominated_state = 0
    pruned_by_no_grid_modify_in_time = 0

    def record_trace(*, steps: Tuple[ProgramStepV137, ...], cost_bits: int, depth: int, ok_train: bool, mismatch: Optional[Dict[str, Any]]) -> None:
        if len(trace_programs) >= trace_program_limit:
            return
        trace_programs.append(
            {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V137),
                "kind": "arc_trace_program_v137",
                "program_sig": ProgramV137(steps=steps).program_sig(),
                "cost_bits": int(cost_bits),
                "depth": int(depth),
                "ok_train": bool(ok_train),
                "mismatch": mismatch,
                "steps": [s.to_dict() for s in steps],
            }
        )

    while heap and tried < max_programs:
        (_pri, steps) = heapq.heappop(heap)
        tried += 1
        depth = int(len(steps))
        prog = ProgramV137(steps=steps)
        cost_bits = _program_cost_bits_v137(steps)

        # Reachability pruning on shape (sound).
        stage = _stage_from_avail_v137(_abstract_slots_after_steps_v137(steps))
        steps_left = int(max_depth - depth)
        shapes_ok = True
        for (inp, out) in train_pairs:
            if not _can_reach_shape_v137(stage=stage, got_shape=grid_shape_v124(inp), want_shape=grid_shape_v124(out), steps_left=steps_left):
                shapes_ok = False
                break
        if not shapes_ok:
            pruned_by_shape_reachability += 1
            continue

        # Enforce a hard gate: if there are remaining steps, must be able to modify the grid within them.
        if int(steps_left) > 0:
            if int(_min_steps_to_grid_modify_v137(stage)) > int(steps_left):
                pruned_by_no_grid_modify_in_time += 1
                continue

        # Dominance pruning (sound): if we've seen equivalent state with <=(depth,cost), prune.
        try:
            # Evaluate on a canonical representative (test_in) to compute state signature.
            st = StateV132(grid=test_in)
            cc4_bg = None
            for s in steps:
                if str(s.op_id) == "cc4":
                    cc4_bg = int((s.args or {}).get("bg") or 0)
                st = _apply_step_v137(st, s)
            sig = _state_sig_v137(stage=stage, state=st, cc4_bg=cc4_bg)
            dom = best_by_state.get(sig)
            if dom is not None:
                dom_depth, dom_cost = dom
                if int(dom_depth) <= int(depth) and int(dom_cost) <= int(cost_bits):
                    pruned_by_dominated_state += 1
                    continue
            best_by_state[sig] = (int(min(dom[0], depth) if dom else depth), int(min(dom[1], cost_bits) if dom else cost_bits))
        except Exception:
            # If signature eval fails, do not prune (fail-open for pruning).
            pass

        ev = _eval_on_train_v137(prog=prog, train_pairs=train_pairs)
        record_trace(steps=steps, cost_bits=cost_bits, depth=depth, ok_train=ev.ok_train, mismatch=ev.mismatch_ex)

        if ev.ok_train:
            if min_cost_bits is None or int(cost_bits) < int(min_cost_bits):
                min_cost_bits = int(cost_bits)
                min_cost_solutions = [prog]
            elif int(cost_bits) == int(min_cost_bits):
                min_cost_solutions.append(prog)
            continue

        if depth >= max_depth:
            continue

        next_steps = _propose_next_steps_v137(
            steps_so_far=list(steps),
            train_pairs=train_pairs,
            test_in=test_in,
            bg_candidates=tuple(int(x) for x in bgs),
            shapes_out=tuple(shapes_out),
            palette_out=tuple(int(c) for c in palette_out),
            direct_steps=list(direct_steps),
        )

        for ns in next_steps:
            new_steps = steps + (ns,)
            new_prog = ProgramV137(steps=new_steps)
            new_cost = _program_cost_bits_v137(new_steps)
            new_ev = _eval_on_train_v137(prog=new_prog, train_pairs=train_pairs)
            pri = (int(new_cost), int(new_ev.loss[0]), int(new_ev.loss[1]), new_prog.program_sig())
            heapq.heappush(heap, (pri, new_steps))

    # Evaluate min-cost solutions on test_in (fail-closed ambiguity).
    if min_cost_bits is not None and min_cost_solutions:
        outputs: Dict[str, GridV124] = {}
        for prog in min_cost_solutions:
            got = apply_program_v137(prog, test_in)
            outputs[grid_hash_v124(got)] = got
        if len(outputs) == 1:
            out_grid = list(outputs.values())[0]
            return {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V137),
                "kind": "arc_solve_result_v137",
                "status": "SOLVED",
                "program_sig": str(min_cost_solutions[0].program_sig()),
                "program_cost_bits": int(min_cost_bits),
                "predicted_grid": [list(r) for r in out_grid],
                "predicted_grid_hash": grid_hash_v124(out_grid),
                "trace": {
                    "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V137),
                    "kind": "arc_trace_v137",
                    "max_depth": int(max_depth),
                    "max_programs": int(max_programs),
                    "tried": int(tried),
                    "min_cost_solutions": int(len(min_cost_solutions)),
                    "pruned_by_shape_reachability": int(pruned_by_shape_reachability),
                    "pruned_by_palette_reachability": int(pruned_by_palette_reachability),
                    "pruned_by_dominated_state": int(pruned_by_dominated_state),
                    "pruned_by_no_grid_modify_in_time": int(pruned_by_no_grid_modify_in_time),
                    "trace_programs": trace_programs,
                },
            }
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V137),
            "kind": "arc_solve_result_v137",
            "status": "UNKNOWN",
            "program_sig": "",
            "program_cost_bits": int(min_cost_bits or 0),
            "predicted_grid_hash": "",
            "failure_reason": {
                "kind": "AMBIGUOUS_RULE",
                "details": {"min_cost_solutions": int(len(min_cost_solutions)), "predicted_grid_hashes": sorted(outputs.keys())},
            },
            "trace": {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V137),
                "kind": "arc_trace_v137",
                "max_depth": int(max_depth),
                "max_programs": int(max_programs),
                "tried": int(tried),
                "min_cost_solutions": int(len(min_cost_solutions)),
                "pruned_by_shape_reachability": int(pruned_by_shape_reachability),
                "pruned_by_palette_reachability": int(pruned_by_palette_reachability),
                "pruned_by_dominated_state": int(pruned_by_dominated_state),
                "pruned_by_no_grid_modify_in_time": int(pruned_by_no_grid_modify_in_time),
                "trace_programs": trace_programs,
            },
        }

    # No solution found within budget.
    failure_kind = "SEARCH_BUDGET_EXCEEDED" if tried >= max_programs else "MISSING_OPERATOR"
    return {
        "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V137),
        "kind": "arc_solve_result_v137",
        "status": "FAIL",
        "program_sig": "",
        "program_cost_bits": 0,
        "predicted_grid_hash": "",
        "failure_reason": {
            "kind": str(failure_kind),
            "details": {
                "candidates_tested": int(tried),
                "max_depth": int(max_depth),
                "search_exhausted": bool(not heap),
            },
        },
        "trace": {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V137),
            "kind": "arc_trace_v137",
            "max_depth": int(max_depth),
            "max_programs": int(max_programs),
            "tried": int(tried),
            "min_cost_solutions": 0,
            "pruned_by_shape_reachability": int(pruned_by_shape_reachability),
            "pruned_by_palette_reachability": int(pruned_by_palette_reachability),
            "pruned_by_dominated_state": int(pruned_by_dominated_state),
            "pruned_by_no_grid_modify_in_time": int(pruned_by_no_grid_modify_in_time),
            "trace_programs": trace_programs,
        },
    }
