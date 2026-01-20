from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_objects_v132 import BBoxV132, ObjectV132, ObjectSetV132, connected_components4_v132
from .arc_ops_v137 import StateV132, apply_op_v137, step_cost_bits_v137
from .grid_v124 import (
    GridV124,
    bbox_nonzero_v124,
    crop_to_bbox_nonzero_v124,
    grid_equal_v124,
    grid_from_list_v124,
    grid_hash_v124,
    grid_shape_v124,
    pad_to_v124,
    unique_colors_v124,
)

ARC_SOLVER_SCHEMA_VERSION_V138 = 138


def _validate_grid_values_v138(g: GridV124) -> None:
    for row in g:
        for v in row:
            x = int(v)
            if x < 0 or x > 9:
                raise ValueError("grid_color_out_of_range")


def _summarize_mismatch_v138(*, got: GridV124, want: GridV124) -> Dict[str, Any]:
    hg, wg = grid_shape_v124(got)
    hw, ww = grid_shape_v124(want)
    if (hg, wg) != (hw, ww):
        return {"kind": "shape_mismatch", "got": {"h": int(hg), "w": int(wg)}, "want": {"h": int(hw), "w": int(ww)}}
    diff = 0
    for r in range(hg):
        for c in range(wg):
            if int(got[r][c]) != int(want[r][c]):
                diff += 1
    return {"kind": "cell_mismatch", "diff_cells": int(diff)}


def _stage_from_avail_v138(avail: Set[str]) -> str:
    # A small abstract stage for reachability pruning.
    if "grid" not in avail:
        return "none"
    # These stages are intentionally coarse; must remain sound.
    if "patch" in avail:
        return "patch"
    if "bbox" in avail:
        return "bbox"
    if "obj" in avail:
        return "obj"
    if "objset" in avail:
        return "objset"
    return "grid"


def _abstract_slots_after_steps_v138(steps: Sequence["ProgramStepV138"]) -> Set[str]:
    avail: Set[str] = {"grid"}
    for s in steps:
        op = str(s.op_id)
        if op == "cc4":
            avail.add("objset")
        elif op == "select_obj":
            if "objset" in avail:
                avail.add("obj")
        elif op == "obj_bbox":
            if "obj" in avail:
                avail.add("bbox")
        elif op in {"bbox_by_color"}:
            avail.add("bbox")
        elif op in {"crop_bbox"}:
            pass
        elif op in {"new_canvas"}:
            pass
        elif op in {"paint_rect", "draw_rect_border"}:
            pass
        elif op in {"pad_to"}:
            pass
        elif op in {"crop_bbox_nonzero"}:
            pass
        elif op in {"translate", "map_colors", "repeat_grid", "overlay_self_translate", "propagate_color_translate"}:
            pass
        elif op in {"paste"}:
            avail.add("patch")
        # If unknown op_id, do not add slots (fail-open for pruning).
    return avail


def _min_steps_to_grid_modify_v138(stage: str) -> int:
    # Minimal number of steps needed to modify grid from a given stage.
    st = str(stage)
    if st == "none":
        return 10**9
    # Any stage that has grid can apply a grid-writing op in 1 step.
    return 1


def _can_reach_shape_v138(*, stage: str, got_shape: Tuple[int, int], want_shape: Tuple[int, int], steps_left: int) -> bool:
    # Sound over-approx shape reachability for current operator inventory.
    if steps_left < 0:
        return False
    (hg, wg) = (int(got_shape[0]), int(got_shape[1]))
    (hw, ww) = (int(want_shape[0]), int(want_shape[1]))
    if (hg, wg) == (hw, ww):
        return True
    # With remaining steps, we can reach arbitrary (hw,ww) iff we can apply pad_to/new_canvas/crop.
    # In this solver family, pad_to and new_canvas are available at any stage with grid; crop_bbox needs bbox.
    if steps_left <= 0:
        return False
    st = str(stage)
    if st in {"grid", "objset", "obj", "bbox", "patch"}:
        # pad_to/new_canvas allow reaching any target shape.
        return True
    return False


@dataclass(frozen=True)
class ProgramStepV138:
    op_id: str
    args: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"op_id": str(self.op_id), "args": dict(self.args)}


@dataclass(frozen=True)
class ProgramV138:
    steps: Tuple[ProgramStepV138, ...]

    def program_sig(self) -> str:
        return sha256_hex(canonical_json_dumps({"steps": [s.to_dict() for s in self.steps]}).encode("utf-8"))


@dataclass(frozen=True)
class _EvalInfoV138:
    ok_train: bool
    loss: Tuple[int, int]
    mismatch_ex: Optional[Dict[str, Any]]


def _apply_step_v138(state: StateV132, step: ProgramStepV138) -> StateV132:
    return apply_op_v137(state=state, op_id=str(step.op_id), args=dict(step.args))


def apply_program_v138(prog: ProgramV138, grid: GridV124) -> GridV124:
    state = StateV132(grid=grid)
    for step in prog.steps:
        state = _apply_step_v138(state, step)
    return state.grid


def _program_cost_bits_v138(steps: Sequence[ProgramStepV138]) -> int:
    total = 0
    for st in steps:
        total += int(step_cost_bits_v137(op_id=str(st.op_id), args=dict(st.args)))
    return int(total)


def _state_sig_v138(*, stage: str, state: StateV132, cc4_bg: Optional[int]) -> str:
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


def _eval_on_train_v138(*, prog: ProgramV138, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> _EvalInfoV138:
    diff_sum = 0
    shape_pen = 0
    mismatch_ex: Optional[Dict[str, Any]] = None
    for inp, out in train_pairs:
        try:
            got = apply_program_v138(prog, inp)
        except Exception as e:
            return _EvalInfoV138(ok_train=False, loss=(10**9, 10**9), mismatch_ex={"kind": "exception", "error": str(e)})
        mm = _summarize_mismatch_v138(got=got, want=out)
        if mm["kind"] == "shape_mismatch":
            shape_pen += 100000
        else:
            diff_sum += int(mm.get("diff_cells") or 0)
        if mismatch_ex is None and (mm["kind"] != "cell_mismatch" or int(mm.get("diff_cells") or 0) != 0):
            mismatch_ex = mm
    ok = diff_sum == 0 and shape_pen == 0
    return _EvalInfoV138(ok_train=ok, loss=(int(shape_pen), int(diff_sum)), mismatch_ex=mismatch_ex)


def _bg_candidates_v138(grids: Sequence[GridV124]) -> Tuple[int, ...]:
    from .arc_solver_v134 import _bg_candidates_v134

    return _bg_candidates_v134(grids)


def _infer_color_mapping_v138(inp: GridV124, out: GridV124) -> Optional[Dict[str, int]]:
    from .arc_solver_v134 import _infer_color_mapping_v134

    return _infer_color_mapping_v134(inp, out)


def _infer_repeat_grid_steps_v138(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[ProgramStepV138]:
    from .arc_solver_v134 import _infer_repeat_grid_steps_v134

    steps_v134 = _infer_repeat_grid_steps_v134(train_pairs)
    out: List[ProgramStepV138] = []
    for s in steps_v134:
        out.append(ProgramStepV138(op_id=str(s.op_id), args=dict(s.args)))
    return out


def _infer_overlay_self_translate_steps_v138(
    *, train_pairs: Sequence[Tuple[GridV124, GridV124]], bg_candidates: Sequence[int]
) -> List[ProgramStepV138]:
    from .arc_solver_v135 import _infer_overlay_self_translate_steps_v135

    steps_v135 = _infer_overlay_self_translate_steps_v135(train_pairs=train_pairs, bg_candidates=tuple(int(x) for x in bg_candidates))
    out: List[ProgramStepV138] = []
    for s in steps_v135:
        out.append(ProgramStepV138(op_id=str(s.op_id), args=dict(s.args)))
    return out


def _infer_propagate_color_translate_steps_v138(
    *, train_pairs: Sequence[Tuple[GridV124, GridV124]], bg_candidates: Sequence[int]
) -> List[ProgramStepV138]:
    from .arc_solver_v137 import _infer_propagate_color_translate_steps_v137

    steps_v137 = _infer_propagate_color_translate_steps_v137(train_pairs=train_pairs, bg_candidates=tuple(int(x) for x in bg_candidates))
    out: List[ProgramStepV138] = []
    for s in steps_v137:
        out.append(ProgramStepV138(op_id=str(s.op_id), args=dict(s.args)))
    return out


def _infer_direct_steps_v138(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> List[ProgramStepV138]:
    direct: List[ProgramStepV138] = []

    for op_id in ["rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v"]:
        ok = True
        step = ProgramStepV138(op_id=str(op_id), args={})
        for inp, out in train_pairs:
            got = apply_program_v138(ProgramV138(steps=(step,)), inp)
            if not grid_equal_v124(got, out):
                ok = False
                break
        if ok:
            direct.append(step)

    mapping: Dict[str, int] = {}
    mapping_ok = True
    for inp, out in train_pairs:
        m = _infer_color_mapping_v138(inp, out)
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
        step = ProgramStepV138(op_id="map_colors", args={"mapping": {str(k): int(mapping[k]) for k in sorted(mapping.keys())}})
        for inp, out in train_pairs:
            got = apply_program_v138(ProgramV138(steps=(step,)), inp)
            if not grid_equal_v124(got, out):
                ok = False
                break
        if ok:
            direct.append(step)

    bgs = _bg_candidates_v138([p[0] for p in train_pairs] + [test_in])
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
        step = ProgramStepV138(op_id="translate", args={"dx": int(dx), "dy": int(dy), "pad": int(bg)})
        ok = True
        for inp, out in train_pairs:
            got = apply_program_v138(ProgramV138(steps=(step,)), inp)
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
            direct.append(ProgramStepV138(op_id="crop_bbox_nonzero", args={"bg": int(bg)}))

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
                direct.append(ProgramStepV138(op_id="pad_to", args={"height": int(h), "width": int(w), "pad": int(bg)}))

    direct.extend(_infer_repeat_grid_steps_v138(train_pairs))
    direct.extend(_infer_overlay_self_translate_steps_v138(train_pairs=train_pairs, bg_candidates=bgs))
    direct.extend(_infer_propagate_color_translate_steps_v138(train_pairs=train_pairs, bg_candidates=bgs))

    direct.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
    seen: Set[str] = set()
    out_steps: List[ProgramStepV138] = []
    for s in direct:
        sig = sha256_hex(canonical_json_dumps(s.to_dict()).encode("utf-8"))
        if sig in seen:
            continue
        seen.add(sig)
        out_steps.append(s)
    return out_steps


def _propose_bbox_by_color_steps_v138(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[ProgramStepV138]:
    if not train_pairs:
        return []
    colors_all: Optional[Set[int]] = None
    for inp, _ in train_pairs:
        cs = set(int(c) for c in unique_colors_v124(inp))
        colors_all = cs if colors_all is None else (colors_all & cs)
        if not colors_all:
            return []
    assert colors_all is not None
    return [ProgramStepV138(op_id="bbox_by_color", args={"color": int(c)}) for c in sorted(colors_all)]


def _cc4_nonempty_for_all_v138(*, grids: Sequence[GridV124], bg: int) -> bool:
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


def _crop_bbox_nonzero_is_noop_v138(*, train_in: Sequence[GridV124], bg: int) -> bool:
    for g in train_in:
        h, w = grid_shape_v124(g)
        r0, c0, r1, c1 = bbox_nonzero_v124(g, bg=int(bg))
        if int(r0) != 0 or int(c0) != 0 or int(r1) != int(h) or int(c1) != int(w):
            return False
    return True


def _pad_to_is_noop_v138(*, train_in_shapes: Sequence[Tuple[int, int]], height: int, width: int) -> bool:
    for h, w in train_in_shapes:
        if int(h) > int(height) or int(w) > int(width):
            return False
    return True


def _propose_next_steps_v138(
    *,
    steps_so_far: Sequence[ProgramStepV138],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    bg_candidates: Sequence[int],
    shapes_out: Sequence[Tuple[int, int]],
    palette_out: Sequence[int],
    direct_steps: Sequence[ProgramStepV138],
) -> List[ProgramStepV138]:
    avail = _abstract_slots_after_steps_v138(steps_so_far)
    stage = _stage_from_avail_v138(avail)

    train_in = [p[0] for p in train_pairs]
    train_in_shapes = [grid_shape_v124(g) for g in train_in]
    out_steps: List[ProgramStepV138] = []

    # direct steps first (strong constraints).
    for s in direct_steps:
        out_steps.append(s)

    # bbox_by_color can create bbox.
    if "bbox" not in avail:
        out_steps.extend(_propose_bbox_by_color_steps_v138(train_pairs=train_pairs))

    # object pipeline is only possible after cc4.
    if "objset" not in avail:
        for bg in bg_candidates:
            if not _cc4_nonempty_for_all_v138(grids=train_in + [test_in], bg=int(bg)):
                continue
            out_steps.append(ProgramStepV138(op_id="cc4", args={"bg": int(bg)}))

    if "objset" in avail and "obj" not in avail:
        # Conservative, small selector arg set (delegated).
        from .arc_solver_v134 import _infer_select_obj_args_v134

        args_list = _infer_select_obj_args_v134(train_pairs=train_pairs, bg=int(bg_candidates[0] if bg_candidates else 0), max_rank=1)
        for a in args_list:
            out_steps.append(ProgramStepV138(op_id="select_obj", args=dict(a)))

    if "obj" in avail and "bbox" not in avail:
        out_steps.append(ProgramStepV138(op_id="obj_bbox", args={}))

    # bbox-driven crops.
    if "bbox" in avail:
        out_steps.append(ProgramStepV138(op_id="crop_bbox", args={}))

    # grid-only ops always possible.
    for bg in bg_candidates:
        if not _crop_bbox_nonzero_is_noop_v138(train_in=train_in + [test_in], bg=int(bg)):
            out_steps.append(ProgramStepV138(op_id="crop_bbox_nonzero", args={"bg": int(bg)}))

    for (h, w) in shapes_out:
        for bg in bg_candidates:
            if _pad_to_is_noop_v138(train_in_shapes=train_in_shapes + [grid_shape_v124(test_in)], height=int(h), width=int(w)):
                continue
            out_steps.append(ProgramStepV138(op_id="pad_to", args={"height": int(h), "width": int(w), "pad": int(bg)}))

    # new_canvas: propose output shapes and bg candidates.
    for (h, w) in shapes_out:
        for bg in bg_candidates:
            out_steps.append(ProgramStepV138(op_id="new_canvas", args={"height": int(h), "width": int(w), "bg": int(bg)}))

    # paint/draw require bbox; if present, propose colors from palette_out.
    if "bbox" in avail:
        for c in palette_out:
            out_steps.append(ProgramStepV138(op_id="paint_rect", args={"color": int(c)}))
            out_steps.append(ProgramStepV138(op_id="draw_rect_border", args={"color": int(c), "thickness": 1}))

    # Dedup deterministically.
    out_steps.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
    seen: Set[str] = set()
    uniq: List[ProgramStepV138] = []
    for s in out_steps:
        sig = sha256_hex(canonical_json_dumps(s.to_dict()).encode("utf-8"))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(s)
    return uniq


@dataclass(frozen=True)
class SolveConfigV138:
    max_depth: int = 4
    max_programs: int = 4000
    trace_program_limit: int = 80
    max_ambiguous_outputs: int = 8


def solve_arc_task_v138(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124, config: SolveConfigV138) -> Dict[str, Any]:
    for inp, out in train_pairs:
        _validate_grid_values_v138(inp)
        _validate_grid_values_v138(out)
    _validate_grid_values_v138(test_in)

    bgs = _bg_candidates_v138([p[0] for p in train_pairs] + [test_in])
    shapes_out = tuple(sorted({grid_shape_v124(out) for _, out in train_pairs}))
    palette_out_set: Set[int] = set()
    for _, out in train_pairs:
        palette_out_set |= set(int(c) for c in unique_colors_v124(out))
    palette_out = tuple(sorted(palette_out_set))

    direct_steps = _infer_direct_steps_v138(train_pairs=train_pairs, test_in=test_in)

    max_depth = int(config.max_depth)
    max_programs = int(config.max_programs)
    trace_program_limit = int(config.trace_program_limit)

    heap: List[Tuple[Tuple[int, int, int, str], Tuple[ProgramStepV138, ...]]] = []
    start: Tuple[ProgramStepV138, ...] = tuple()
    start_prog = ProgramV138(steps=start)
    start_cost = _program_cost_bits_v138(start)
    start_eval = _eval_on_train_v138(prog=start_prog, train_pairs=train_pairs)
    heapq.heappush(heap, ((int(start_cost), int(start_eval.loss[0]), int(start_eval.loss[1]), start_prog.program_sig()), start))

    best_by_state: Dict[str, Tuple[int, int]] = {}
    trace_programs: List[Dict[str, Any]] = []
    tried = 0
    min_cost_solutions: List[ProgramV138] = []
    min_cost_bits: Optional[int] = None

    pruned_by_shape_reachability = 0
    pruned_by_palette_reachability = 0
    pruned_by_dominated_state = 0
    pruned_by_no_grid_modify_in_time = 0

    def record_trace(*, steps: Tuple[ProgramStepV138, ...], cost_bits: int, depth: int, ok_train: bool, mismatch: Optional[Dict[str, Any]]) -> None:
        if len(trace_programs) >= trace_program_limit:
            return
        trace_programs.append(
            {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V138),
                "kind": "arc_trace_program_v138",
                "program_sig": ProgramV138(steps=steps).program_sig(),
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
        prog = ProgramV138(steps=steps)
        cost_bits = _program_cost_bits_v138(steps)

        stage = _stage_from_avail_v138(_abstract_slots_after_steps_v138(steps))
        steps_left = int(max_depth - depth)

        shapes_ok = True
        for (inp, out) in train_pairs:
            if not _can_reach_shape_v138(stage=stage, got_shape=grid_shape_v124(inp), want_shape=grid_shape_v124(out), steps_left=steps_left):
                shapes_ok = False
                break
        if not shapes_ok:
            pruned_by_shape_reachability += 1
            continue

        if int(steps_left) > 0:
            if int(_min_steps_to_grid_modify_v138(stage)) > int(steps_left):
                pruned_by_no_grid_modify_in_time += 1
                continue

        try:
            st = StateV132(grid=test_in)
            cc4_bg = None
            for s in steps:
                if str(s.op_id) == "cc4":
                    cc4_bg = int((s.args or {}).get("bg") or 0)
                st = _apply_step_v138(st, s)
            sig = _state_sig_v138(stage=stage, state=st, cc4_bg=cc4_bg)
            dom = best_by_state.get(sig)
            if dom is not None:
                dom_depth, dom_cost = dom
                if int(dom_depth) <= int(depth) and int(dom_cost) <= int(cost_bits):
                    pruned_by_dominated_state += 1
                    continue
            best_by_state[sig] = (int(min(dom[0], depth) if dom else depth), int(min(dom[1], cost_bits) if dom else cost_bits))
        except Exception:
            pass

        ev = _eval_on_train_v138(prog=prog, train_pairs=train_pairs)
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

        next_steps = _propose_next_steps_v138(
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
            new_prog = ProgramV138(steps=new_steps)
            new_cost = _program_cost_bits_v138(new_steps)
            new_ev = _eval_on_train_v138(prog=new_prog, train_pairs=train_pairs)
            pri = (int(new_cost), int(new_ev.loss[0]), int(new_ev.loss[1]), new_prog.program_sig())
            heapq.heappush(heap, (pri, new_steps))

    if min_cost_bits is not None and min_cost_solutions:
        outputs: Dict[str, GridV124] = {}
        for prog in min_cost_solutions:
            got = apply_program_v138(prog, test_in)
            outputs[grid_hash_v124(got)] = got
        if len(outputs) == 1:
            out_grid = list(outputs.values())[0]
            return {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V138),
                "kind": "arc_solve_result_v138",
                "status": "SOLVED",
                "program_sig": str(min_cost_solutions[0].program_sig()),
                "program_cost_bits": int(min_cost_bits),
                "predicted_grid": [list(r) for r in out_grid],
                "predicted_grid_hash": grid_hash_v124(out_grid),
                "trace": {
                    "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V138),
                    "kind": "arc_trace_v138",
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
        # Ambiguous: include a deterministic small set of predicted grids for ARC-protocol multi-attempt scoring.
        hashes_sorted = sorted(outputs.keys())
        max_keep = int(max(1, int(config.max_ambiguous_outputs)))
        kept_hashes = hashes_sorted[:max_keep]
        predicted_grids = [{"grid_hash": str(h), "grid": [list(r) for r in outputs[h]]} for h in kept_hashes]
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V138),
            "kind": "arc_solve_result_v138",
            "status": "UNKNOWN",
            "program_sig": "",
            "program_cost_bits": int(min_cost_bits or 0),
            "predicted_grid_hash": "",
            "predicted_grids": predicted_grids,
            "failure_reason": {
                "kind": "AMBIGUOUS_RULE",
                "details": {
                    "min_cost_solutions": int(len(min_cost_solutions)),
                    "predicted_grid_hashes": hashes_sorted,
                    "predicted_grids_truncated": bool(len(hashes_sorted) > len(kept_hashes)),
                    "predicted_grids_kept": int(len(kept_hashes)),
                },
            },
            "trace": {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V138),
                "kind": "arc_trace_v138",
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

    failure_kind = "SEARCH_BUDGET_EXCEEDED" if tried >= max_programs else "MISSING_OPERATOR"
    return {
        "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V138),
        "kind": "arc_solve_result_v138",
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
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V138),
            "kind": "arc_trace_v138",
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

