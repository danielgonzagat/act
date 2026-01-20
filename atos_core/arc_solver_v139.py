from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_objects_v132 import connected_components4_v132
from .arc_ops_v132 import StateV132
from .arc_ops_v137 import apply_op_v137, step_cost_bits_v137
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

ARC_SOLVER_SCHEMA_VERSION_V139 = 139


def _validate_grid_values_v139(g: GridV124) -> None:
    for row in g:
        for v in row:
            x = int(v)
            if x < 0 or x > 9:
                raise ValueError("grid_color_out_of_range")


def _summarize_mismatch_v139(*, got: GridV124, want: GridV124) -> Dict[str, Any]:
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


def _stage_from_avail_v139(avail: Set[str]) -> str:
    if "grid" not in avail:
        return "none"
    if "patch" in avail:
        return "patch"
    if "bbox" in avail:
        return "bbox"
    if "obj" in avail:
        return "obj"
    if "objset" in avail:
        return "objset"
    return "grid"


def _abstract_slots_after_steps_v139(steps: Sequence["ProgramStepV139"]) -> Set[str]:
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
        elif op == "crop_bbox":
            avail.add("patch")
        elif op == "commit_patch":
            avail.discard("patch")
            avail.discard("objset")
            avail.discard("obj")
            avail.discard("bbox")
        elif op == "new_canvas":
            avail.discard("objset")
            avail.discard("obj")
            avail.discard("bbox")
            avail.discard("patch")
        elif op in {
            "rotate90",
            "rotate180",
            "rotate270",
            "reflect_h",
            "reflect_v",
            "translate",
            "overlay_self_translate",
            "propagate_color_translate",
            "repeat_grid",
            "crop_bbox_nonzero",
            "pad_to",
            "replace_color",
            "map_colors",
        }:
            avail.discard("objset")
            avail.discard("obj")
            avail.discard("bbox")
            avail.discard("patch")
        elif op == "bbox_by_color":
            avail.discard("objset")
            avail.discard("obj")
            avail.discard("patch")
    return avail


def _min_steps_to_grid_modify_v139(stage: str) -> int:
    if str(stage) == "none":
        return 10**9
    return 1


def _can_reach_shape_v139(*, stage: str, got_shape: Tuple[int, int], want_shape: Tuple[int, int], steps_left: int) -> bool:
    if steps_left < 0:
        return False
    (hg, wg) = (int(got_shape[0]), int(got_shape[1]))
    (hw, ww) = (int(want_shape[0]), int(want_shape[1]))
    if (hg, wg) == (hw, ww):
        return True
    if steps_left <= 0:
        return False
    st = str(stage)
    if st in {"grid", "objset", "obj", "bbox", "patch"}:
        return True
    return False


@dataclass(frozen=True)
class ProgramStepV139:
    op_id: str
    args: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"op_id": str(self.op_id), "args": {str(k): self.args[k] for k in sorted(self.args.keys())}}


@dataclass(frozen=True)
class ProgramV139:
    steps: Tuple[ProgramStepV139, ...]

    def program_sig(self) -> str:
        body = {"schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V139), "kind": "arc_program_v139", "steps": [s.to_dict() for s in self.steps]}
        return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


@dataclass(frozen=True)
class _EvalInfoV139:
    ok_train: bool
    loss: Tuple[int, int]
    mismatch_ex: Optional[Dict[str, Any]]
    vec_sig: str
    test_grid: GridV124


def _canonical_args_key_v139(args: Dict[str, Any]) -> str:
    a = {str(k): args[k] for k in sorted(args.keys())}
    return canonical_json_dumps(a)


def _grid_hash_cached_v139(*, g: GridV124, cache: Dict[GridV124, str]) -> str:
    hit = cache.get(g)
    if hit is not None:
        return str(hit)
    h = grid_hash_v124(g)
    cache[g] = str(h)
    return str(h)


def _state_sig_dict_cached_v139(*, state: StateV132, grid_hash_cache: Dict[GridV124, str]) -> Dict[str, Any]:
    return {
        "grid_hash": _grid_hash_cached_v139(g=state.grid, cache=grid_hash_cache),
        "objset_sig": str(state.objset.set_sig()) if state.objset is not None else "",
        "obj_sig": str(state.obj.object_sig()) if state.obj is not None else "",
        "bbox": state.bbox.to_dict() if state.bbox is not None else None,
        "patch_hash": _grid_hash_cached_v139(g=state.patch, cache=grid_hash_cache) if state.patch is not None else "",
    }


def _apply_step_cached_v139(
    *,
    state: StateV132,
    op_id: str,
    args: Dict[str, Any],
    apply_cache: Dict[str, StateV132],
    grid_hash_cache: Dict[GridV124, str],
    metrics: Dict[str, int],
) -> StateV132:
    st_key = canonical_json_dumps(_state_sig_dict_cached_v139(state=state, grid_hash_cache=grid_hash_cache))
    args_key = _canonical_args_key_v139(args)
    key = f"{st_key}|{str(op_id)}|{args_key}"
    hit = apply_cache.get(key)
    if hit is not None:
        metrics["apply_cache_hits"] = int(metrics.get("apply_cache_hits", 0)) + 1
        return hit
    metrics["apply_cache_misses"] = int(metrics.get("apply_cache_misses", 0)) + 1
    nxt = apply_op_v137(state=state, op_id=str(op_id), args=dict(args))
    apply_cache[key] = nxt
    return nxt


def _apply_program_state_cached_v139(
    *,
    grid: GridV124,
    steps: Sequence[ProgramStepV139],
    apply_cache: Dict[str, StateV132],
    grid_hash_cache: Dict[GridV124, str],
    metrics: Dict[str, int],
) -> StateV132:
    st = StateV132(grid=grid)
    for s in steps:
        st = _apply_step_cached_v139(
            state=st,
            op_id=str(s.op_id),
            args=dict(s.args),
            apply_cache=apply_cache,
            grid_hash_cache=grid_hash_cache,
            metrics=metrics,
        )
    return st


def apply_program_v139(prog: ProgramV139, grid: GridV124) -> GridV124:
    st = StateV132(grid=grid)
    for step in prog.steps:
        st = apply_op_v137(state=st, op_id=str(step.op_id), args=dict(step.args))
    return st.grid


def _program_cost_bits_v139(steps: Sequence[ProgramStepV139]) -> int:
    total = 0
    for st in steps:
        total += int(step_cost_bits_v137(op_id=str(st.op_id), args=dict(st.args)))
    return int(total)


def _infer_color_mapping_v139(inp: GridV124, out: GridV124) -> Optional[Dict[str, int]]:
    from .arc_solver_v134 import _infer_color_mapping_v134

    return _infer_color_mapping_v134(inp, out)


def _bg_candidates_v139(grids: Sequence[GridV124]) -> Tuple[int, ...]:
    from .arc_solver_v134 import _bg_candidates_v134

    return _bg_candidates_v134(grids)


def _infer_repeat_grid_steps_v139(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[ProgramStepV139]:
    from .arc_solver_v134 import _infer_repeat_grid_steps_v134

    steps_v134 = _infer_repeat_grid_steps_v134(train_pairs)
    return [ProgramStepV139(op_id=str(s.op_id), args=dict(s.args)) for s in steps_v134]


def _infer_overlay_self_translate_steps_v139(
    *, train_pairs: Sequence[Tuple[GridV124, GridV124]], bg_candidates: Sequence[int]
) -> List[ProgramStepV139]:
    from .arc_solver_v135 import _infer_overlay_self_translate_steps_v135

    steps_v135 = _infer_overlay_self_translate_steps_v135(train_pairs=train_pairs, bg_candidates=tuple(int(x) for x in bg_candidates))
    return [ProgramStepV139(op_id=str(s.op_id), args=dict(s.args)) for s in steps_v135]


def _infer_propagate_color_translate_steps_v139(
    *, train_pairs: Sequence[Tuple[GridV124, GridV124]], bg_candidates: Sequence[int]
) -> List[ProgramStepV139]:
    from .arc_solver_v137 import _infer_propagate_color_translate_steps_v137

    steps_v137 = _infer_propagate_color_translate_steps_v137(train_pairs=train_pairs, bg_candidates=tuple(int(x) for x in bg_candidates))
    return [ProgramStepV139(op_id=str(s.op_id), args=dict(s.args)) for s in steps_v137]


def _infer_direct_steps_v139(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> List[ProgramStepV139]:
    direct: List[ProgramStepV139] = []

    for op_id in ["rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v"]:
        ok = True
        step = ProgramStepV139(op_id=str(op_id), args={})
        for inp, out in train_pairs:
            got = apply_program_v139(ProgramV139(steps=(step,)), inp)
            if not grid_equal_v124(got, out):
                ok = False
                break
        if ok:
            direct.append(step)

    mapping: Dict[str, int] = {}
    mapping_ok = True
    for inp, out in train_pairs:
        m = _infer_color_mapping_v139(inp, out)
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
        step = ProgramStepV139(op_id="map_colors", args={"mapping": {str(k): int(mapping[k]) for k in sorted(mapping.keys())}})
        ok = True
        for inp, out in train_pairs:
            got = apply_program_v139(ProgramV139(steps=(step,)), inp)
            if not grid_equal_v124(got, out):
                ok = False
                break
        if ok:
            direct.append(step)

    bgs = _bg_candidates_v139([p[0] for p in train_pairs] + [test_in])
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
        step = ProgramStepV139(op_id="translate", args={"dx": int(dx), "dy": int(dy), "pad": int(bg)})
        ok = True
        for inp, out in train_pairs:
            got = apply_program_v139(ProgramV139(steps=(step,)), inp)
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
            direct.append(ProgramStepV139(op_id="crop_bbox_nonzero", args={"bg": int(bg)}))

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
                direct.append(ProgramStepV139(op_id="pad_to", args={"height": int(h), "width": int(w), "pad": int(bg)}))

    direct.extend(_infer_repeat_grid_steps_v139(train_pairs))
    direct.extend(_infer_overlay_self_translate_steps_v139(train_pairs=train_pairs, bg_candidates=bgs))
    direct.extend(_infer_propagate_color_translate_steps_v139(train_pairs=train_pairs, bg_candidates=bgs))

    direct.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
    seen: Set[str] = set()
    out_steps: List[ProgramStepV139] = []
    for s in direct:
        sig = sha256_hex(canonical_json_dumps(s.to_dict()).encode("utf-8"))
        if sig in seen:
            continue
        seen.add(sig)
        out_steps.append(s)
    return out_steps


def _propose_bbox_by_color_steps_v139(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[ProgramStepV139]:
    if not train_pairs:
        return []
    colors_all: Optional[Set[int]] = None
    for inp, _ in train_pairs:
        cs = set(int(c) for c in unique_colors_v124(inp))
        colors_all = cs if colors_all is None else (colors_all & cs)
        if not colors_all:
            return []
    assert colors_all is not None
    return [ProgramStepV139(op_id="bbox_by_color", args={"color": int(c)}) for c in sorted(colors_all)]


def _cc4_nonempty_for_all_v139(*, grids: Sequence[GridV124], bg: int) -> bool:
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


def _crop_bbox_nonzero_is_noop_v139(*, train_in: Sequence[GridV124], bg: int) -> bool:
    for g in train_in:
        h, w = grid_shape_v124(g)
        r0, c0, r1, c1 = bbox_nonzero_v124(g, bg=int(bg))
        if int(r0) != 0 or int(c0) != 0 or int(r1) != int(h) or int(c1) != int(w):
            return False
    return True


def _pad_to_is_noop_v139(*, train_in_shapes: Sequence[Tuple[int, int]], height: int, width: int) -> bool:
    for h, w in train_in_shapes:
        if int(h) > int(height) or int(w) > int(width):
            return False
    return True


def _propose_next_steps_v139(
    *,
    steps_so_far: Sequence[ProgramStepV139],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    bg_candidates: Sequence[int],
    shapes_out: Sequence[Tuple[int, int]],
    palette_out: Sequence[int],
    direct_steps: Sequence[ProgramStepV139],
) -> List[ProgramStepV139]:
    avail = _abstract_slots_after_steps_v139(steps_so_far)
    train_in = [p[0] for p in train_pairs]
    train_in_shapes = [grid_shape_v124(g) for g in train_in]
    out_steps: List[ProgramStepV139] = []

    for s in direct_steps:
        out_steps.append(s)

    if "bbox" not in avail:
        out_steps.extend(_propose_bbox_by_color_steps_v139(train_pairs=train_pairs))

    if "objset" not in avail:
        for bg in bg_candidates:
            if not _cc4_nonempty_for_all_v139(grids=train_in + [test_in], bg=int(bg)):
                continue
            out_steps.append(ProgramStepV139(op_id="cc4", args={"bg": int(bg)}))

    if "objset" in avail and "obj" not in avail:
        from .arc_solver_v134 import _infer_select_obj_args_v134

        args_list = _infer_select_obj_args_v134(train_pairs=train_pairs, bg=int(bg_candidates[0] if bg_candidates else 0), max_rank=1)
        for a in args_list:
            out_steps.append(ProgramStepV139(op_id="select_obj", args=dict(a)))

    if "obj" in avail and "bbox" not in avail:
        out_steps.append(ProgramStepV139(op_id="obj_bbox", args={}))

    if "bbox" in avail:
        out_steps.append(ProgramStepV139(op_id="crop_bbox", args={}))

    for bg in bg_candidates:
        if not _crop_bbox_nonzero_is_noop_v139(train_in=train_in + [test_in], bg=int(bg)):
            out_steps.append(ProgramStepV139(op_id="crop_bbox_nonzero", args={"bg": int(bg)}))

    for (h, w) in shapes_out:
        for bg in bg_candidates:
            if _pad_to_is_noop_v139(train_in_shapes=train_in_shapes + [grid_shape_v124(test_in)], height=int(h), width=int(w)):
                continue
            out_steps.append(ProgramStepV139(op_id="pad_to", args={"height": int(h), "width": int(w), "pad": int(bg)}))

    for (h, w) in shapes_out:
        for bg in bg_candidates:
            out_steps.append(ProgramStepV139(op_id="new_canvas", args={"height": int(h), "width": int(w), "bg": int(bg)}))

    if "bbox" in avail:
        for c in palette_out:
            out_steps.append(ProgramStepV139(op_id="paint_rect", args={"color": int(c)}))
            out_steps.append(ProgramStepV139(op_id="draw_rect_border", args={"color": int(c), "thickness": 1}))

    out_steps.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
    seen: Set[str] = set()
    uniq: List[ProgramStepV139] = []
    for s in out_steps:
        sig = sha256_hex(canonical_json_dumps(s.to_dict()).encode("utf-8"))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(s)
    return uniq


@dataclass(frozen=True)
class SolveConfigV139:
    max_depth: int = 4
    max_programs: int = 4000
    trace_program_limit: int = 80
    max_ambiguous_outputs: int = 8


def solve_arc_task_v139(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124, config: SolveConfigV139) -> Dict[str, Any]:
    # Fail-closed on invalid grids: return structured failure, do not throw.
    try:
        for inp, out in train_pairs:
            _validate_grid_values_v139(inp)
            _validate_grid_values_v139(out)
        _validate_grid_values_v139(test_in)
    except Exception as e:
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V139),
            "kind": "arc_solve_result_v139",
            "status": "FAIL",
            "program_sig": "",
            "program_cost_bits": 0,
            "predicted_grid_hash": "",
            "failure_reason": {"kind": "INVARIANT_VIOLATION", "details": {"error": str(e)}},
            "trace": {"schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V139), "kind": "arc_trace_v139", "tried": 0},
        }

    bgs = _bg_candidates_v139([p[0] for p in train_pairs] + [test_in])
    shapes_out = tuple(sorted({grid_shape_v124(out) for _, out in train_pairs}))
    palette_out_set: Set[int] = set()
    for _, out in train_pairs:
        palette_out_set |= set(int(c) for c in unique_colors_v124(out))
    palette_out = tuple(sorted(palette_out_set))

    direct_steps = _infer_direct_steps_v139(train_pairs=train_pairs, test_in=test_in)

    max_depth = int(config.max_depth)
    max_programs = int(config.max_programs)
    trace_program_limit = int(config.trace_program_limit)

    apply_cache: Dict[str, StateV132] = {}
    eval_cache: Dict[str, _EvalInfoV139] = {}
    grid_hash_cache: Dict[GridV124, str] = {}
    metrics: Dict[str, int] = {
        "apply_cache_hits": 0,
        "apply_cache_misses": 0,
        "eval_cache_hits": 0,
        "eval_cache_misses": 0,
    }

    def eval_program(steps: Tuple[ProgramStepV139, ...]) -> _EvalInfoV139:
        prog = ProgramV139(steps=steps)
        psig = prog.program_sig()
        cached = eval_cache.get(psig)
        if cached is not None:
            metrics["eval_cache_hits"] = int(metrics.get("eval_cache_hits", 0)) + 1
            return cached
        metrics["eval_cache_misses"] = int(metrics.get("eval_cache_misses", 0)) + 1

        # Determine stage for vec_sig (coarse, but includes slot signatures).
        avail = _abstract_slots_after_steps_v139(steps)
        stage = _stage_from_avail_v139(avail)
        cc4_bg: Optional[int] = None
        for st in steps:
            if str(st.op_id) == "cc4":
                cc4_bg = int((st.args or {}).get("bg") or 0)

        shape_pen = 0
        diff_sum = 0
        mismatch_ex: Optional[Dict[str, Any]] = None
        train_state_sigs: List[str] = []

        for inp, out in train_pairs:
            try:
                end_state = _apply_program_state_cached_v139(
                    grid=inp,
                    steps=steps,
                    apply_cache=apply_cache,
                    grid_hash_cache=grid_hash_cache,
                    metrics=metrics,
                )
                got = end_state.grid
            except Exception as e:
                # Prefix is not executable; extensions won't fix.
                ee = {"kind": "exception", "error": str(e)}
                cached_fail = _EvalInfoV139(ok_train=False, loss=(10**9, 10**9), mismatch_ex=ee, vec_sig="", test_grid=test_in)
                eval_cache[psig] = cached_fail
                return cached_fail

            mm = _summarize_mismatch_v139(got=got, want=out)
            if mm["kind"] == "shape_mismatch":
                shape_pen += 100000
            else:
                diff_sum += int(mm.get("diff_cells") or 0)
            if mismatch_ex is None and (mm["kind"] != "cell_mismatch" or int(mm.get("diff_cells") or 0) != 0):
                mismatch_ex = mm

            train_state_sigs.append(
                canonical_json_dumps(
                    {
                        "stage": str(stage),
                        "cc4_bg": int(cc4_bg) if cc4_bg is not None else -1,
                        "state": _state_sig_dict_cached_v139(state=end_state, grid_hash_cache=grid_hash_cache),
                    }
                )
            )

        ok_train = shape_pen == 0 and diff_sum == 0

        try:
            test_state = _apply_program_state_cached_v139(
                grid=test_in,
                steps=steps,
                apply_cache=apply_cache,
                grid_hash_cache=grid_hash_cache,
                metrics=metrics,
            )
        except Exception:
            # If program cannot be applied to test, treat as fail (still keep trace).
            test_state = StateV132(grid=test_in)

        test_state_sig = canonical_json_dumps(
            {
                "stage": str(stage),
                "cc4_bg": int(cc4_bg) if cc4_bg is not None else -1,
                "state": _state_sig_dict_cached_v139(state=test_state, grid_hash_cache=grid_hash_cache),
            }
        )
        vec_sig = sha256_hex(
            canonical_json_dumps(
                {
                    "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V139),
                    "kind": "arc_vec_state_sig_v139",
                    "stage": str(stage),
                    "cc4_bg": int(cc4_bg) if cc4_bg is not None else -1,
                    "train": list(train_state_sigs),
                    "test": str(test_state_sig),
                }
            ).encode("utf-8")
        )
        out = _EvalInfoV139(
            ok_train=bool(ok_train),
            loss=(int(shape_pen), int(diff_sum)),
            mismatch_ex=mismatch_ex,
            vec_sig=str(vec_sig),
            test_grid=test_state.grid,
        )
        eval_cache[psig] = out
        return out

    heap: List[Tuple[Tuple[int, int, int, int, str, str], Tuple[ProgramStepV139, ...]]] = []
    start: Tuple[ProgramStepV139, ...] = tuple()
    start_prog = ProgramV139(steps=start)
    start_cost = _program_cost_bits_v139(start)
    start_eval = eval_program(start)
    heapq.heappush(
        heap,
        (
            (int(start_cost), int(start_eval.loss[0]), int(start_eval.loss[1]), 0, str(start_eval.vec_sig), start_prog.program_sig()),
            start,
        ),
    )

    best_cost_by_vec_sig: Dict[str, int] = {}
    trace_programs: List[Dict[str, Any]] = []
    tried = 0
    min_cost_bits: Optional[int] = None
    min_cost_solution_sigs: List[str] = []

    pruned_by_shape_reachability = 0
    pruned_by_palette_reachability = 0
    pruned_by_dominated_state = 0
    pruned_by_no_grid_modify_in_time = 0
    expanded_states = 0
    frontier_max = 0

    def record_trace(*, steps: Tuple[ProgramStepV139, ...], cost_bits: int, depth: int, ok_train: bool, mismatch: Optional[Dict[str, Any]]) -> None:
        if len(trace_programs) >= trace_program_limit:
            return
        trace_programs.append(
            {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V139),
                "kind": "arc_trace_program_v139",
                "program_sig": ProgramV139(steps=steps).program_sig(),
                "cost_bits": int(cost_bits),
                "depth": int(depth),
                "ok_train": bool(ok_train),
                "mismatch": mismatch,
                "steps": [s.to_dict() for s in steps],
            }
        )

    while heap and tried < max_programs:
        frontier_max = max(frontier_max, len(heap))
        (_pri, steps) = heapq.heappop(heap)
        tried += 1
        expanded_states += 1

        depth = int(len(steps))
        cost_bits = _program_cost_bits_v139(steps)
        stage = _stage_from_avail_v139(_abstract_slots_after_steps_v139(steps))
        steps_left = int(max_depth - depth)

        shapes_ok = True
        for (inp, out) in train_pairs:
            if not _can_reach_shape_v139(stage=stage, got_shape=grid_shape_v124(inp), want_shape=grid_shape_v124(out), steps_left=steps_left):
                shapes_ok = False
                break
        if not shapes_ok:
            pruned_by_shape_reachability += 1
            continue
        if steps_left > 0 and int(_min_steps_to_grid_modify_v139(stage)) > int(steps_left):
            pruned_by_no_grid_modify_in_time += 1
            continue

        ev = eval_program(steps)
        record_trace(steps=steps, cost_bits=cost_bits, depth=depth, ok_train=ev.ok_train, mismatch=ev.mismatch_ex)
        if ev.mismatch_ex is not None and str(ev.mismatch_ex.get("kind") or "") == "exception":
            continue

        if ev.vec_sig:
            dom_cost = best_cost_by_vec_sig.get(str(ev.vec_sig))
            if dom_cost is not None and int(dom_cost) <= int(cost_bits):
                pruned_by_dominated_state += 1
                continue
            best_cost_by_vec_sig[str(ev.vec_sig)] = int(cost_bits)

        if ev.ok_train:
            prog_sig = ProgramV139(steps=steps).program_sig()
            if min_cost_bits is None or int(cost_bits) < int(min_cost_bits):
                min_cost_bits = int(cost_bits)
                min_cost_solution_sigs = [str(prog_sig)]
            elif int(cost_bits) == int(min_cost_bits):
                min_cost_solution_sigs.append(str(prog_sig))
            continue

        if depth >= max_depth:
            continue

        next_steps = _propose_next_steps_v139(
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
            new_cost = int(cost_bits + step_cost_bits_v137(op_id=str(ns.op_id), args=dict(ns.args)))
            new_ev = eval_program(new_steps)
            new_sig = ProgramV139(steps=new_steps).program_sig()
            pri = (int(new_cost), int(new_ev.loss[0]), int(new_ev.loss[1]), int(len(new_steps)), str(new_ev.vec_sig), str(new_sig))
            heapq.heappush(heap, (pri, new_steps))

    if min_cost_bits is not None and min_cost_solution_sigs:
        outputs: Dict[str, Dict[str, Any]] = {}
        for psig in min_cost_solution_sigs:
            entry = eval_cache.get(str(psig))
            if entry is None:
                continue
            out_grid = entry.test_grid
            gh = _grid_hash_cached_v139(g=out_grid, cache=grid_hash_cache)
            cur = outputs.get(gh)
            if cur is None or str(psig) < str(cur.get("min_program_sig") or ""):
                outputs[gh] = {"grid": out_grid, "min_program_sig": str(psig)}

        if len(outputs) == 1:
            out_grid = list(outputs.values())[0]["grid"]
            return {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V139),
                "kind": "arc_solve_result_v139",
                "status": "SOLVED",
                "program_sig": str(min_cost_solution_sigs[0]),
                "program_cost_bits": int(min_cost_bits),
                "predicted_grid": [list(r) for r in out_grid],
                "predicted_grid_hash": grid_hash_v124(out_grid),
                "trace": {
                    "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V139),
                    "kind": "arc_trace_v139",
                    "max_depth": int(max_depth),
                    "max_programs": int(max_programs),
                    "tried": int(tried),
                    "min_cost_solutions": int(len(min_cost_solution_sigs)),
                    "pruned_by_shape_reachability": int(pruned_by_shape_reachability),
                    "pruned_by_palette_reachability": int(pruned_by_palette_reachability),
                    "pruned_by_dominated_state": int(pruned_by_dominated_state),
                    "pruned_by_no_grid_modify_in_time": int(pruned_by_no_grid_modify_in_time),
                    "apply_cache_hits": int(metrics.get("apply_cache_hits", 0)),
                    "apply_cache_misses": int(metrics.get("apply_cache_misses", 0)),
                    "eval_cache_hits": int(metrics.get("eval_cache_hits", 0)),
                    "eval_cache_misses": int(metrics.get("eval_cache_misses", 0)),
                    "unique_states": int(len(best_cost_by_vec_sig)),
                    "expanded_states": int(expanded_states),
                    "frontier_max": int(frontier_max),
                    "trace_programs": trace_programs,
                },
            }

        ordered = sorted(outputs.items(), key=lambda kv: (str(kv[1].get("min_program_sig") or ""), str(kv[0])))
        max_keep = int(max(1, int(config.max_ambiguous_outputs)))
        kept = ordered[:max_keep]
        predicted_grids = [
            {
                "grid_hash": str(h),
                "program_sig": str(obj.get("min_program_sig") or ""),
                "grid": [list(r) for r in obj["grid"]],
            }
            for h, obj in kept
        ]
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V139),
            "kind": "arc_solve_result_v139",
            "status": "UNKNOWN",
            "program_sig": "",
            "program_cost_bits": int(min_cost_bits or 0),
            "predicted_grid_hash": "",
            "predicted_grids": predicted_grids,
            "failure_reason": {
                "kind": "AMBIGUOUS_RULE",
                "details": {
                    "min_cost_solutions": int(len(min_cost_solution_sigs)),
                    "predicted_grid_hashes": [str(h) for h, _ in ordered],
                    "predicted_grids_truncated": bool(len(ordered) > len(kept)),
                    "predicted_grids_kept": int(len(kept)),
                },
            },
            "trace": {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V139),
                "kind": "arc_trace_v139",
                "max_depth": int(max_depth),
                "max_programs": int(max_programs),
                "tried": int(tried),
                "min_cost_solutions": int(len(min_cost_solution_sigs)),
                "pruned_by_shape_reachability": int(pruned_by_shape_reachability),
                "pruned_by_palette_reachability": int(pruned_by_palette_reachability),
                "pruned_by_dominated_state": int(pruned_by_dominated_state),
                "pruned_by_no_grid_modify_in_time": int(pruned_by_no_grid_modify_in_time),
                "apply_cache_hits": int(metrics.get("apply_cache_hits", 0)),
                "apply_cache_misses": int(metrics.get("apply_cache_misses", 0)),
                "eval_cache_hits": int(metrics.get("eval_cache_hits", 0)),
                "eval_cache_misses": int(metrics.get("eval_cache_misses", 0)),
                "unique_states": int(len(best_cost_by_vec_sig)),
                "expanded_states": int(expanded_states),
                "frontier_max": int(frontier_max),
                "trace_programs": trace_programs,
            },
        }

    failure_kind = "SEARCH_BUDGET_EXCEEDED" if tried >= max_programs else "MISSING_OPERATOR"
    return {
        "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V139),
        "kind": "arc_solve_result_v139",
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
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V139),
            "kind": "arc_trace_v139",
            "max_depth": int(max_depth),
            "max_programs": int(max_programs),
            "tried": int(tried),
            "min_cost_solutions": 0,
            "pruned_by_shape_reachability": int(pruned_by_shape_reachability),
            "pruned_by_palette_reachability": int(pruned_by_palette_reachability),
            "pruned_by_dominated_state": int(pruned_by_dominated_state),
            "pruned_by_no_grid_modify_in_time": int(pruned_by_no_grid_modify_in_time),
            "apply_cache_hits": int(metrics.get("apply_cache_hits", 0)),
            "apply_cache_misses": int(metrics.get("apply_cache_misses", 0)),
            "eval_cache_hits": int(metrics.get("eval_cache_hits", 0)),
            "eval_cache_misses": int(metrics.get("eval_cache_misses", 0)),
            "unique_states": int(len(best_cost_by_vec_sig)),
            "expanded_states": int(expanded_states),
            "frontier_max": int(frontier_max),
            "trace_programs": trace_programs,
        },
    }
