from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_objects_v132 import connected_components4_v132
from .arc_ops_v132 import StateV132
from .arc_ops_v141 import OP_DEFS_V141, apply_op_v141, connected_components4_nonbg_multicolor_v141, step_cost_bits_v141
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

ARC_SOLVER_SCHEMA_VERSION_V141 = 141


def _validate_grid_values_v141(g: GridV124) -> None:
    for row in g:
        for v in row:
            x = int(v)
            if x < 0 or x > 9:
                raise ValueError("grid_color_out_of_range")


def _summarize_mismatch_v141(*, got: GridV124, want: GridV124) -> Dict[str, Any]:
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


def _abstract_slots_after_steps_v141(steps: Sequence["ProgramStepV141"]) -> Dict[str, bool]:
    avail: Dict[str, bool] = {"grid": True, "objset": False, "obj": False, "bbox": False, "patch": False}
    for st in steps:
        op_id = str(st.op_id)
        # Meta-ops (concept/macro calls) are treated as grid-modifying closures for reachability pruning.
        # Their inner steps are executed via _apply_step_cached_v141; here we only need conservative
        # invalidation so the stage machine remains sound.
        if op_id in {"macro_call", "concept_call"}:
            avail["objset"] = False
            avail["obj"] = False
            avail["bbox"] = False
            avail["patch"] = False
            continue
        od = OP_DEFS_V141.get(op_id)
        if od is None:
            continue
        for r in od.reads:
            if not bool(avail.get(str(r), False)):
                return {"grid": True, "objset": False, "obj": False, "bbox": False, "patch": False, "invalid": True}
        for w in od.writes:
            avail[str(w)] = True

        # Invalidations (must reflect apply_op semantics).
        if op_id == "commit_patch":
            avail["patch"] = False
            avail["objset"] = False
            avail["obj"] = False
            avail["bbox"] = False
        if op_id == "new_canvas":
            avail["objset"] = False
            avail["obj"] = False
            avail["bbox"] = False
            avail["patch"] = False
        if op_id in {
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
            avail["objset"] = False
            avail["obj"] = False
            avail["bbox"] = False
            avail["patch"] = False
        if op_id == "bbox_by_color":
            avail["objset"] = False
            avail["obj"] = False
            avail["patch"] = False
    return avail


def _stage_from_avail_v141(avail: Dict[str, bool]) -> str:
    if bool(avail.get("patch", False)):
        return "patch"
    if bool(avail.get("bbox", False)):
        return "bbox"
    if bool(avail.get("obj", False)):
        return "obj"
    if bool(avail.get("objset", False)):
        return "objset"
    return "grid"


def _min_steps_to_grid_stage_v141(stage: str) -> int:
    s = str(stage)
    if s == "grid":
        return 0
    if s == "patch":
        return 1  # commit_patch
    if s == "bbox":
        return 2  # crop_bbox + commit_patch
    if s == "obj":
        return 3  # obj_bbox + crop_bbox + commit_patch
    if s == "objset":
        return 4  # select_obj + obj_bbox + crop_bbox + commit_patch
    return 99


def _min_steps_to_grid_modify_v141(stage: str) -> int:
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


def _min_steps_to_shape_change_v141(stage: str, direction: str) -> int:
    s = str(stage)
    d = str(direction)
    if d == "grow":
        return int(_min_steps_to_grid_stage_v141(s)) + 1
    if d == "shrink":
        if s == "patch":
            return 1
        if s == "bbox":
            return 2
        if s == "obj":
            return 3
        if s == "objset":
            return 4
        return 1  # crop_bbox_nonzero
    if d == "mixed":
        return int(_min_steps_to_grid_stage_v141(s)) + 2
    return 0


def _can_reach_shape_v141(*, stage: str, got_shape: Tuple[int, int], want_shape: Tuple[int, int], steps_left: int) -> bool:
    hg, wg = int(got_shape[0]), int(got_shape[1])
    hw, ww = int(want_shape[0]), int(want_shape[1])
    if (hg, wg) == (hw, ww):
        return True
    if int(steps_left) <= 0:
        return False
    if hw >= hg and ww >= wg:
        return int(steps_left) >= int(_min_steps_to_shape_change_v141(stage, "grow"))
    if hw <= hg and ww <= wg:
        return int(steps_left) >= int(_min_steps_to_shape_change_v141(stage, "shrink"))
    return int(steps_left) >= int(_min_steps_to_shape_change_v141(stage, "mixed"))


def _can_reach_palette_v141(*, stage: str, got_palette: Set[int], want_palette: Set[int], steps_left: int) -> bool:
    if set(int(x) for x in got_palette) == set(int(x) for x in want_palette):
        return True
    if int(steps_left) <= 0:
        return False
    return int(steps_left) >= int(_min_steps_to_grid_modify_v141(stage))


@dataclass(frozen=True)
class ProgramStepV141:
    op_id: str
    args: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"op_id": str(self.op_id), "args": {str(k): self.args[k] for k in sorted(self.args.keys())}}


@dataclass(frozen=True)
class ProgramV141:
    steps: Tuple[ProgramStepV141, ...]

    def program_sig(self) -> str:
        body = {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
            "kind": "arc_program_v141",
            "steps": [s.to_dict() for s in self.steps],
        }
        return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


@dataclass(frozen=True)
class _EvalInfoV141:
    ok_train: bool
    loss: Tuple[int, int]
    mismatch_ex: Optional[Dict[str, Any]]
    vec_sig: str
    test_grid: GridV124
    got_shapes: Tuple[Tuple[int, int], ...]
    got_palettes: Tuple[Tuple[int, ...], ...]


def _canonical_args_key_v141(args: Dict[str, Any]) -> str:
    a = {str(k): args[k] for k in sorted(args.keys())}
    return canonical_json_dumps(a)


def _grid_hash_cached_v141(*, g: GridV124, cache: Dict[GridV124, str]) -> str:
    hit = cache.get(g)
    if hit is not None:
        return str(hit)
    h = grid_hash_v124(g)
    cache[g] = str(h)
    return str(h)


def _state_sig_dict_cached_v141(*, state: StateV132, grid_hash_cache: Dict[GridV124, str]) -> Dict[str, Any]:
    return {
        "grid_hash": _grid_hash_cached_v141(g=state.grid, cache=grid_hash_cache),
        "objset_sig": str(state.objset.set_sig()) if state.objset is not None else "",
        "obj_sig": str(state.obj.object_sig()) if state.obj is not None else "",
        "bbox": state.bbox.to_dict() if state.bbox is not None else None,
        "patch_hash": _grid_hash_cached_v141(g=state.patch, cache=grid_hash_cache) if state.patch is not None else "",
    }


def _apply_step_cached_v141(
    *,
    state: StateV132,
    op_id: str,
    args: Dict[str, Any],
    apply_cache: Dict[Tuple[Any, ...], StateV132],
    grid_hash_cache: Dict[GridV124, str],
    metrics: Dict[str, int],
) -> StateV132:
    op_id = str(op_id or "")
    if op_id in {"macro_call", "concept_call"}:
        steps_raw = args.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            raise ValueError("meta_call_missing_steps")
        st = state
        for row in steps_raw:
            if not isinstance(row, dict):
                raise ValueError("meta_call_bad_step")
            op2 = str(row.get("op_id") or "")
            if not op2:
                raise ValueError("meta_call_bad_step_op")
            if op2 in {"macro_call", "concept_call"}:
                raise ValueError("meta_call_nested_forbidden")
            a2 = row.get("args") if isinstance(row.get("args"), dict) else {}
            st = _apply_step_cached_v141(
                state=st,
                op_id=str(op2),
                args=dict(a2),
                apply_cache=apply_cache,
                grid_hash_cache=grid_hash_cache,
                metrics=metrics,
            )
        return st

    def freeze_json(x: Any) -> Any:
        if isinstance(x, dict):
            return tuple((str(k), freeze_json(x[k])) for k in sorted(x.keys()))
        if isinstance(x, list):
            return tuple(freeze_json(v) for v in x)
        return x

    bbox_sig: Optional[Tuple[int, int, int, int]] = None
    if state.bbox is not None:
        bbox_sig = (int(state.bbox.r0), int(state.bbox.c0), int(state.bbox.r1), int(state.bbox.c1))

    key = (
        "v141",
        _grid_hash_cached_v141(g=state.grid, cache=grid_hash_cache),
        str(state.objset.set_sig()) if state.objset is not None else "",
        str(state.obj.object_sig()) if state.obj is not None else "",
        bbox_sig,
        _grid_hash_cached_v141(g=state.patch, cache=grid_hash_cache) if state.patch is not None else "",
        str(op_id),
        freeze_json({str(k): args[k] for k in sorted(args.keys())}),
    )
    hit = apply_cache.get(key)
    if hit is not None:
        metrics["apply_cache_hits"] = int(metrics.get("apply_cache_hits", 0)) + 1
        return hit
    metrics["apply_cache_misses"] = int(metrics.get("apply_cache_misses", 0)) + 1
    nxt = apply_op_v141(state=state, op_id=str(op_id), args=dict(args))
    apply_cache[key] = nxt
    return nxt


def _eval_program_on_pairs_v141(
    *,
    program: ProgramV141,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    apply_cache: Dict[Tuple[Any, ...], StateV132],
    grid_hash_cache: Dict[GridV124, str],
    metrics: Dict[str, int],
) -> _EvalInfoV141:
    got_shapes: List[Tuple[int, int]] = []
    got_palettes: List[Tuple[int, ...]] = []
    mismatches: List[Dict[str, Any]] = []
    ok = True
    loss_cells = 0
    loss_shape = 0

    for inp, want in train_pairs:
        st = StateV132(grid=inp)
        got: GridV124 = inp
        try:
            for step in program.steps:
                st = _apply_step_cached_v141(
                    state=st,
                    op_id=str(step.op_id),
                    args=dict(step.args),
                    apply_cache=apply_cache,
                    grid_hash_cache=grid_hash_cache,
                    metrics=metrics,
                )
            got = st.grid
        except Exception as e:
            # Prefix is not executable; extensions won't fix. Treat as mismatch with
            # deterministic exception evidence.
            ok = False
            if mismatches:
                # keep first mismatch
                pass
            else:
                mismatches.append(
                    {
                        "kind": "exception",
                        "error": str(e),
                    }
                )
        got_shapes.append(tuple(int(x) for x in grid_shape_v124(got)))
        got_palettes.append(tuple(sorted(int(c) for c in unique_colors_v124(got))))
        if not grid_equal_v124(got, want):
            ok = False
            mm = _summarize_mismatch_v141(got=got, want=want)
            mismatches.append(mm)
            if mm.get("kind") == "shape_mismatch":
                loss_shape += 1
            elif mm.get("kind") == "cell_mismatch":
                loss_cells += int(mm.get("diff_cells") or 0)

    stt = StateV132(grid=test_in)
    test_grid: GridV124 = test_in
    try:
        for step in program.steps:
            stt = _apply_step_cached_v141(
                state=stt,
                op_id=str(step.op_id),
                args=dict(step.args),
                apply_cache=apply_cache,
                grid_hash_cache=grid_hash_cache,
                metrics=metrics,
            )
        test_grid = stt.grid
    except Exception:
        test_grid = test_in

    loss = (int(loss_shape), int(loss_cells))
    vec_obj = {
        "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
        "kind": "train_loss_vec_v141",
        "loss": {"shape": int(loss_shape), "cells": int(loss_cells)},
        "mismatch_ex": mismatches[0] if mismatches else None,
        "got_shapes": got_shapes,
        "got_palettes": got_palettes,
    }
    vec_sig = sha256_hex(canonical_json_dumps(vec_obj).encode("utf-8"))
    return _EvalInfoV141(
        ok_train=bool(ok),
        loss=loss,
        mismatch_ex=mismatches[0] if mismatches else None,
        vec_sig=str(vec_sig),
        test_grid=test_grid,
        got_shapes=tuple(got_shapes),
        got_palettes=tuple(got_palettes),
    )


def _mode_color_v141(g: GridV124) -> int:
    counts: Dict[int, int] = {}
    for row in g:
        for x in row:
            counts[int(x)] = int(counts.get(int(x), 0)) + 1
    if not counts:
        return 0
    items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
    return int(items[0][0])


def _infer_bg_candidates_v141(*, grids: Sequence[GridV124]) -> List[int]:
    if not grids:
        return [0]
    cand: Set[int] = set()
    for g in grids:
        h, w = grid_shape_v124(g)
        if h <= 0 or w <= 0:
            continue
        cand.add(_mode_color_v141(g))
        cand.add(int(g[0][0]))
        cand.add(int(g[0][w - 1]))
        cand.add(int(g[h - 1][0]))
        cand.add(int(g[h - 1][w - 1]))
    return sorted(set(int(x) for x in cand))


def _infer_shapes_out_v141(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[Tuple[int, int]]:
    shapes: Set[Tuple[int, int]] = set()
    for _, out in train_pairs:
        shapes.add(tuple(int(x) for x in grid_shape_v124(out)))
    return sorted(shapes)


def _infer_palette_out_v141(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[int]:
    cols: Set[int] = set()
    for _, out in train_pairs:
        for c in unique_colors_v124(out):
            cols.add(int(c))
    return sorted(cols)


def _infer_direct_steps_v141(
    *, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124
) -> List[ProgramStepV141]:
    from .arc_solver_v140 import _infer_direct_steps_v140

    steps_v140 = _infer_direct_steps_v140(train_pairs=train_pairs, test_in=test_in)
    out: List[ProgramStepV141] = []
    for s in steps_v140:
        out.append(ProgramStepV141(op_id=str(s.op_id), args=dict(s.args)))
    out.sort(key=lambda st: canonical_json_dumps(st.to_dict()))
    seen: Set[str] = set()
    uniq: List[ProgramStepV141] = []
    for st in out:
        sig = sha256_hex(canonical_json_dumps(st.to_dict()).encode("utf-8"))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(st)
    return uniq


def _propose_bbox_by_color_steps_v141(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[ProgramStepV141]:
    if not train_pairs:
        return []
    colors_all: Optional[Set[int]] = None
    for inp, _ in train_pairs:
        cs = set(int(c) for c in unique_colors_v124(inp))
        colors_all = cs if colors_all is None else (colors_all & cs)
        if not colors_all:
            return []
    assert colors_all is not None
    return [ProgramStepV141(op_id="bbox_by_color", args={"color": int(c)}) for c in sorted(colors_all)]


def _cc4_nonempty_for_all_v141(*, grids: Sequence[GridV124], bg: int) -> bool:
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


def _cc4_multicolor_nonempty_for_all_v141(*, grids: Sequence[GridV124], bg: int) -> bool:
    for g in grids:
        try:
            oset = connected_components4_nonbg_multicolor_v141(g, bg=int(bg))
        except Exception:
            return False
        if not getattr(oset, "objects", None):
            return False
        if len(oset.objects) <= 0:
            return False
    return True


def _grid_has_multicolor_component_v141(g: GridV124, *, bg: int) -> bool:
    # Deterministic: check if any (cell!=bg) CC contains >1 color.
    oset = connected_components4_nonbg_multicolor_v141(g, bg=int(bg))
    for o in getattr(oset, "objects", ()) or ():
        cols: Set[int] = set()
        for r, c in o.cells:
            cols.add(int(g[int(r)][int(c)]))
            if len(cols) > 1:
                return True
    return False


def _crop_bbox_nonzero_is_noop_v141(*, train_in: Sequence[GridV124], bg: int) -> bool:
    for g in train_in:
        h, w = grid_shape_v124(g)
        r0, c0, r1, c1 = bbox_nonzero_v124(g, bg=int(bg))
        if int(r0) != 0 or int(c0) != 0 or int(r1) != int(h) or int(c1) != int(w):
            return False
    return True


def _pad_to_is_noop_v141(*, train_in_shapes: Sequence[Tuple[int, int]], height: int, width: int) -> bool:
    for h, w in train_in_shapes:
        if int(h) > int(height) or int(w) > int(width):
            return False
    return True


def _propose_next_steps_v141(
    *,
    steps_so_far: Sequence[ProgramStepV141],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    bg_candidates: Sequence[int],
    shapes_out: Sequence[Tuple[int, int]],
    palette_out: Sequence[int],
    direct_steps: Sequence[ProgramStepV141],
) -> List[ProgramStepV141]:
    avail = _abstract_slots_after_steps_v141(steps_so_far)
    if bool(avail.get("invalid", False)):
        return []
    train_in = [p[0] for p in train_pairs]
    train_in_shapes = [grid_shape_v124(g) for g in train_in]
    out_steps: List[ProgramStepV141] = []

    for s in direct_steps:
        out_steps.append(s)

    if not bool(avail.get("bbox", False)):
        out_steps.extend(_propose_bbox_by_color_steps_v141(train_pairs=train_pairs))

    if not bool(avail.get("objset", False)):
        for bg in bg_candidates:
            # Always allow monochrome cc4 if nonempty.
            if _cc4_nonempty_for_all_v141(grids=train_in + [test_in], bg=int(bg)):
                out_steps.append(ProgramStepV141(op_id="cc4", args={"bg": int(bg)}))
            # Only propose multicolor cc when it yields multicolor components in at least one grid.
            any_multicolor = False
            for g in train_in + [test_in]:
                if _grid_has_multicolor_component_v141(g, bg=int(bg)):
                    any_multicolor = True
                    break
            if not any_multicolor:
                continue
            if not _cc4_multicolor_nonempty_for_all_v141(grids=train_in + [test_in], bg=int(bg)):
                continue
            out_steps.append(ProgramStepV141(op_id="cc4_nonbg_multicolor", args={"bg": int(bg)}))

    if bool(avail.get("objset", False)) and not bool(avail.get("obj", False)):
        from .arc_solver_v134 import _infer_select_obj_args_v134

        args_list = _infer_select_obj_args_v134(
            train_pairs=train_pairs, bg=int(bg_candidates[0] if bg_candidates else 0), max_rank=1
        )
        for a in args_list:
            out_steps.append(ProgramStepV141(op_id="select_obj", args=dict(a)))

    if bool(avail.get("obj", False)) and not bool(avail.get("bbox", False)):
        out_steps.append(ProgramStepV141(op_id="obj_bbox", args={}))

    if bool(avail.get("bbox", False)):
        out_steps.append(ProgramStepV141(op_id="crop_bbox", args={}))

    if bool(avail.get("patch", False)):
        # crop_bbox writes patch (not grid); commit_patch is required to make the crop effective.
        out_steps.append(ProgramStepV141(op_id="commit_patch", args={}))

    for bg in bg_candidates:
        if not _crop_bbox_nonzero_is_noop_v141(train_in=train_in + [test_in], bg=int(bg)):
            out_steps.append(ProgramStepV141(op_id="crop_bbox_nonzero", args={"bg": int(bg)}))

    for (h, w) in shapes_out:
        for bg in bg_candidates:
            if _pad_to_is_noop_v141(train_in_shapes=train_in_shapes + [grid_shape_v124(test_in)], height=int(h), width=int(w)):
                continue
            out_steps.append(ProgramStepV141(op_id="pad_to", args={"height": int(h), "width": int(w), "pad": int(bg)}))

    for (h, w) in shapes_out:
        for bg in bg_candidates:
            out_steps.append(ProgramStepV141(op_id="new_canvas", args={"height": int(h), "width": int(w), "color": int(bg)}))

    if bool(avail.get("bbox", False)):
        for c in palette_out:
            out_steps.append(ProgramStepV141(op_id="paint_rect", args={"color": int(c)}))
            out_steps.append(ProgramStepV141(op_id="draw_rect_border", args={"color": int(c), "thickness": 1}))

    out_steps.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
    seen: Set[str] = set()
    uniq: List[ProgramStepV141] = []
    for s in out_steps:
        sig = sha256_hex(canonical_json_dumps(s.to_dict()).encode("utf-8"))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(s)
    return uniq


def _norm_macro_templates_v141(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize macro/operator templates to a minimal internal schema:
      {macro_id, op_ids, support}

    Supports:
      - arc_macro_template_v143 (macro_id, op_ids, support)
      - arc_operator_template_v147 (operator_id/op_ids/support)  [treated as macro_id]
    """
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("kind") or "")
        macro_id = str(row.get("macro_id") or "")
        if not macro_id and kind == "arc_operator_template_v147":
            macro_id = str(row.get("operator_id") or "")
        if not macro_id:
            continue
        op_ids_raw = row.get("op_ids")
        if not isinstance(op_ids_raw, list) or not op_ids_raw:
            continue
        op_ids = [str(x) for x in op_ids_raw if str(x)]
        if not op_ids:
            continue
        support = int(row.get("support") or 0)
        if macro_id in seen:
            continue
        seen.add(macro_id)
        out.append({"macro_id": str(macro_id), "op_ids": list(op_ids), "support": int(support)})
    out.sort(
        key=lambda r: (-int(r.get("support") or 0), -len(list(r.get("op_ids") or [])), str(r.get("macro_id") or ""))
    )
    return out


def _floor_log2_int(n: int) -> int:
    x = int(n)
    if x <= 0:
        return 0
    return int(x.bit_length() - 1)


def _macro_call_cost_bits_v141(
    *,
    macro_id: str,
    steps: Sequence[Dict[str, Any]],
    support_by_macro_id: Dict[str, int],
) -> int:
    inner = 0
    for row in steps:
        if not isinstance(row, dict):
            continue
        op_id = str(row.get("op_id") or "")
        if not op_id or op_id in {"macro_call", "concept_call"}:
            continue
        args = row.get("args") if isinstance(row.get("args"), dict) else {}
        inner += int(step_cost_bits_v141(op_id=str(op_id), args=dict(args)))

    # MDL compression bonus: higher-support macros are cheaper to call (bounded).
    sup = int(support_by_macro_id.get(str(macro_id), 0))
    bonus = min(8, _floor_log2_int(max(1, sup)))
    return max(1, int(inner) - int(bonus))


def _step_cost_bits_total_v141(
    *,
    step: "ProgramStepV141",
    macro_support_by_id: Dict[str, int],
) -> int:
    op_id = str(step.op_id or "")
    if op_id == "macro_call":
        args = step.args if isinstance(step.args, dict) else {}
        mid = str(args.get("macro_id") or "")
        steps_raw = args.get("steps")
        steps_list = steps_raw if isinstance(steps_raw, list) else []
        return _macro_call_cost_bits_v141(macro_id=str(mid), steps=steps_list, support_by_macro_id=macro_support_by_id)
    if op_id == "concept_call":
        # Concept-call cost is carried by the concept template (compression). If missing, fall back to 10.
        args = step.args if isinstance(step.args, dict) else {}
        cb = args.get("cost_bits")
        if isinstance(cb, int):
            return max(1, int(cb))
        if isinstance(cb, str) and str(cb).isdigit():
            return max(1, int(str(cb)))
        return 10
    return int(step_cost_bits_v141(op_id=str(step.op_id), args=dict(step.args)))


def _instantiate_macro_steps_v141(
    *,
    macro_op_ids: Sequence[str],
    steps_prefix: Tuple["ProgramStepV141", ...],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    bg_candidates: Sequence[int],
    shapes_out: Sequence[Tuple[int, int]],
    palette_out: Sequence[int],
    direct_steps: Sequence["ProgramStepV141"],
    max_instantiations: int,
    max_branch_per_op: int,
) -> List[Tuple["ProgramStepV141", ...]]:
    """
    Deterministically instantiate a macro template (sequence of op_ids) into executable steps
    by choosing concrete args via the same per-step proposal policy used by the solver.

    This is a bounded search over args only (op_ids are fixed by the template).
    """
    want = [str(x) for x in macro_op_ids if str(x)]
    if not want:
        return []
    max_inst = max(1, int(max_instantiations))
    max_branch = max(1, int(max_branch_per_op))

    frontier: List[Tuple["ProgramStepV141", ...]] = [tuple()]
    for op in want:
        next_frontier: List[Tuple["ProgramStepV141", ...]] = []
        for inst in frontier:
            steps_so_far = tuple(list(steps_prefix) + list(inst))
            candidates = _propose_next_steps_v141(
                steps_so_far=steps_so_far,
                train_pairs=train_pairs,
                test_in=test_in,
                bg_candidates=bg_candidates,
                shapes_out=shapes_out,
                palette_out=palette_out,
                direct_steps=direct_steps,
            )
            matches = [c for c in candidates if str(c.op_id) == str(op)]
            matches.sort(
                key=lambda st: (
                    int(step_cost_bits_v141(op_id=str(st.op_id), args=dict(st.args))),
                    canonical_json_dumps(st.to_dict()),
                )
            )
            for st in matches[:max_branch]:
                next_frontier.append(tuple(list(inst) + [st]))
        if not next_frontier:
            return []

        # Prune instantiations deterministically by accumulated inner-step MDL.
        def _inst_cost_key(steps: Tuple["ProgramStepV141", ...]) -> Tuple[int, str]:
            cost = 0
            for st in steps:
                cost += int(step_cost_bits_v141(op_id=str(st.op_id), args=dict(st.args)))
            sig = sha256_hex(canonical_json_dumps([s.to_dict() for s in steps]).encode("utf-8"))
            return (int(cost), str(sig))

        next_frontier.sort(key=_inst_cost_key)
        frontier = next_frontier[:max_inst]

    return list(frontier)


@dataclass(frozen=True)
class SolveConfigV141:
    max_depth: int = 4
    max_programs: int = 4000
    trace_program_limit: int = 80
    max_ambiguous_outputs: int = 8
    max_next_steps: int = 128

    # Allow collecting multiple near-minimal solutions for auditing, while keeping the
    # decision rule fail-closed on minimal programs.
    solution_cost_slack_bits: int = 0

    # Learned closures: macro/operator templates (v143/v147) and concept templates (v146).
    macro_templates: Tuple[Dict[str, Any], ...] = tuple()
    concept_templates: Tuple[Dict[str, Any], ...] = tuple()

    # Pressure / gating flags (kept in config for determinism; the solver remains fail-closed).
    abstraction_pressure: bool = False
    macro_try_on_fail_only: bool = True
    enable_reachability_pruning: bool = True

    # Macro instantiation policy (bounded, deterministic).
    macro_propose_max_depth: int = 0
    macro_max_templates: int = 24
    macro_max_instantiations: int = 10
    macro_max_branch_per_op: int = 10

    # Optional stages (accepted for harness compatibility).
    enable_repair_stage: bool = True
    enable_residual_stage: bool = True
    enable_refine_stage: bool = True
    enable_point_patch_repair: bool = False
    point_patch_max_points: int = 12


def solve_arc_task_v141(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    config: SolveConfigV141,
) -> Dict[str, Any]:
    """
    Deterministic compositional search (best-first by MDL+train-loss) over a typed state machine.
    Fail-closed: if multiple minimal programs yield different outputs on test_in, return UNKNOWN.
    """
    # Fail-closed on invalid grids: return structured failure, do not throw.
    try:
        for inp, out in train_pairs:
            _validate_grid_values_v141(inp)
            _validate_grid_values_v141(out)
        _validate_grid_values_v141(test_in)
    except Exception as e:
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
            "kind": "arc_solver_result_v141",
            "status": "FAIL",
            "program_sig": "",
            "predicted_grid_hash": "",
            "failure_reason": {"kind": "INVARIANT_VIOLATION", "details": {"error": str(e)}},
            "trace": {
                "trace_programs": [],
                "apply_cache_hits": 0,
                "apply_cache_misses": 0,
                "eval_cache_hits": 0,
                "eval_cache_misses": 0,
                "pruned_by_shape_reachability": 0,
                "pruned_by_palette_reachability": 0,
                "pruned_by_no_grid_modify_in_time": 0,
                "pruned_by_exception_prefix": 0,
                "pruned_by_next_steps_cap": 0,
            },
        }

    bg_candidates = _infer_bg_candidates_v141(grids=[p[0] for p in train_pairs] + [test_in])
    shapes_out = _infer_shapes_out_v141(train_pairs=train_pairs)
    palette_out = _infer_palette_out_v141(train_pairs=train_pairs)
    direct_steps = _infer_direct_steps_v141(train_pairs=train_pairs, test_in=test_in)

    max_depth = int(config.max_depth)
    max_programs = int(config.max_programs)
    trace_program_limit = int(config.trace_program_limit)
    max_ambiguous_outputs = int(config.max_ambiguous_outputs)
    max_next_steps = int(config.max_next_steps)
    solution_cost_slack_bits = int(getattr(config, "solution_cost_slack_bits", 0) or 0)

    enable_reachability_pruning = bool(getattr(config, "enable_reachability_pruning", True))
    macro_propose_max_depth = int(getattr(config, "macro_propose_max_depth", 0) or 0)
    macro_max_templates = int(getattr(config, "macro_max_templates", 24) or 24)
    macro_max_instantiations = int(getattr(config, "macro_max_instantiations", 10) or 10)
    macro_max_branch_per_op = int(getattr(config, "macro_max_branch_per_op", 10) or 10)
    macro_try_on_fail_only = bool(getattr(config, "macro_try_on_fail_only", True))

    # Normalize macros (operators) once per task.
    raw_mt = getattr(config, "macro_templates", ()) or ()
    macro_templates_all = tuple([m for m in raw_mt if isinstance(m, dict)])
    macro_templates_norm = _norm_macro_templates_v141([dict(r) for r in macro_templates_all])
    macro_templates_norm = macro_templates_norm[: int(max(0, int(macro_max_templates)))]
    macro_support_by_id: Dict[str, int] = {
        str(r.get("macro_id") or ""): int(r.get("support") or 0) for r in macro_templates_norm if str(r.get("macro_id") or "")
    }

    # Memoization caches (pure, local).
    apply_cache: Dict[Tuple[Any, ...], StateV132] = {}
    eval_cache: Dict[str, _EvalInfoV141] = {}
    grid_hash_cache: Dict[GridV124, str] = {}
    metrics: Dict[str, int] = {
        "apply_cache_hits": 0,
        "apply_cache_misses": 0,
        "eval_cache_hits": 0,
        "eval_cache_misses": 0,
        "pruned_by_exception_prefix": 0,
        "pruned_by_next_steps_cap": 0,
    }

    def eval_program(steps: Tuple[ProgramStepV141, ...]) -> _EvalInfoV141:
        p = ProgramV141(steps=steps)
        psig = p.program_sig()
        cached = eval_cache.get(psig)
        if cached is not None:
            metrics["eval_cache_hits"] = int(metrics.get("eval_cache_hits", 0)) + 1
            return cached
        metrics["eval_cache_misses"] = int(metrics.get("eval_cache_misses", 0)) + 1
        ev = _eval_program_on_pairs_v141(
            program=p,
            train_pairs=train_pairs,
            test_in=test_in,
            apply_cache=apply_cache,
            grid_hash_cache=grid_hash_cache,
            metrics=metrics,
        )
        eval_cache[psig] = ev
        return ev

    def _solve_pass(*, allow_macros: bool) -> Dict[str, Any]:
        frontier: List[Tuple[Tuple[int, int, int, str], Tuple[ProgramStepV141, ...], str]] = []
        seen_programs: Set[str] = set()

        def push(steps: Tuple[ProgramStepV141, ...]) -> None:
            p = ProgramV141(steps=steps)
            sig = p.program_sig()
            if sig in seen_programs:
                return
            seen_programs.add(sig)
            # MDL cost (meta-ops handled explicitly; base ops via step_cost_bits_v141).
            cost = 0
            for st in steps:
                cost += int(_step_cost_bits_total_v141(step=st, macro_support_by_id=macro_support_by_id))
            ev = eval_program(steps)
            loss_shape, loss_cells = ev.loss
            key = (int(cost), int(loss_shape), int(loss_cells), str(sig))
            heapq.heappush(frontier, (key, steps, sig))

        push(tuple())

        trace_programs: List[Dict[str, Any]] = []
        trace_items: List[Tuple[Tuple[int, int, int, int, str], Dict[str, Any]]] = []
        programs_explored = 0
        best_cost: Optional[int] = None
        best_programs: List[Dict[str, Any]] = []

        pruned_shape = 0
        pruned_palette = 0
        pruned_no_grid_modify = 0

        while frontier and int(programs_explored) < int(max_programs):
            (cost_key, steps, sig) = heapq.heappop(frontier)
            programs_explored += 1
            depth = int(len(steps))
            ev = eval_program(steps)
            if int(trace_program_limit) > 0:
                tr_row = {
                    "program_sig": str(sig),
                    "depth": int(depth),
                    "cost_bits": int(cost_key[0]),
                    "loss": {"shape": int(ev.loss[0]), "cells": int(ev.loss[1])},
                    "ok_train": bool(ev.ok_train),
                    "mismatch_ex": ev.mismatch_ex,
                    "steps": [s.to_dict() for s in steps],
                }
                # Keep the best-N trace programs by (loss_shape, loss_cells, cost_bits, depth, sig).
                # This makes operator mining stable and useful (near-miss programs survive), rather than
                # storing just the first explored prefixes.
                tr_key = (int(ev.loss[0]), int(ev.loss[1]), int(cost_key[0]), int(depth), str(sig))
                if len(trace_items) < int(trace_program_limit):
                    trace_items.append((tr_key, tr_row))
                else:
                    worst_i = 0
                    worst_k = trace_items[0][0]
                    for i, (k, _row) in enumerate(trace_items):
                        if k > worst_k:
                            worst_k = k
                            worst_i = i
                    if tr_key < worst_k:
                        trace_items[worst_i] = (tr_key, tr_row)

            if ev.ok_train:
                if best_cost is None:
                    best_cost = int(cost_key[0])
                if int(cost_key[0]) <= int(best_cost) + int(solution_cost_slack_bits):
                    best_programs.append(
                        {
                            "program_sig": str(sig),
                            "cost_bits": int(cost_key[0]),
                            "steps": [s.to_dict() for s in steps],
                            "predicted_grid": [list(r) for r in ev.test_grid],
                            "predicted_grid_hash": _grid_hash_cached_v141(g=ev.test_grid, cache=grid_hash_cache),
                        }
                    )
                if len(best_programs) >= max_ambiguous_outputs:
                    break
                if frontier and best_cost is not None and int(frontier[0][0][0]) > int(best_cost) + int(solution_cost_slack_bits):
                    break
                continue

            if isinstance(ev.mismatch_ex, dict) and str(ev.mismatch_ex.get("kind") or "") == "exception":
                metrics["pruned_by_exception_prefix"] = int(metrics.get("pruned_by_exception_prefix", 0)) + 1
                continue

            if depth >= max_depth:
                continue

            avail = _abstract_slots_after_steps_v141(steps)
            stage = _stage_from_avail_v141(avail)
            steps_left = int(max_depth - depth)

            if bool(enable_reachability_pruning):
                ok_reach = True
                for want_shape in shapes_out:
                    for got_shape in ev.got_shapes:
                        if not _can_reach_shape_v141(
                            stage=stage, got_shape=got_shape, want_shape=want_shape, steps_left=steps_left
                        ):
                            pruned_shape += 1
                            ok_reach = False
                            break
                    if not ok_reach:
                        break
                if not ok_reach:
                    continue
                for want_palette in [set(int(c) for c in palette_out)]:
                    for got_pal in ev.got_palettes:
                        if not _can_reach_palette_v141(
                            stage=stage,
                            got_palette=set(int(x) for x in got_pal),
                            want_palette=want_palette,
                            steps_left=steps_left,
                        ):
                            pruned_palette += 1
                            ok_reach = False
                            break
                    if not ok_reach:
                        break
                if not ok_reach:
                    continue

                if _min_steps_to_grid_modify_v141(stage) > steps_left:
                    pruned_no_grid_modify += 1
                    continue

            next_steps: List[ProgramStepV141] = list(
                _propose_next_steps_v141(
                    steps_so_far=steps,
                    train_pairs=train_pairs,
                    test_in=test_in,
                    bg_candidates=bg_candidates,
                    shapes_out=shapes_out,
                    palette_out=palette_out,
                    direct_steps=direct_steps,
                )
            )

            macro_allowed_here = bool(allow_macros) and bool(macro_templates_norm) and int(depth) <= int(macro_propose_max_depth)
            if macro_allowed_here:
                for tmpl in macro_templates_norm:
                    mid = str(tmpl.get("macro_id") or "")
                    op_ids = tmpl.get("op_ids") if isinstance(tmpl.get("op_ids"), list) else []
                    insts = _instantiate_macro_steps_v141(
                        macro_op_ids=[str(x) for x in op_ids if str(x)],
                        steps_prefix=steps,
                        train_pairs=train_pairs,
                        test_in=test_in,
                        bg_candidates=bg_candidates,
                        shapes_out=shapes_out,
                        palette_out=palette_out,
                        direct_steps=direct_steps,
                        max_instantiations=int(macro_max_instantiations),
                        max_branch_per_op=int(macro_max_branch_per_op),
                    )
                    for inner_steps in insts:
                        next_steps.append(
                            ProgramStepV141(
                                op_id="macro_call",
                                args={"macro_id": str(mid), "steps": [s.to_dict() for s in inner_steps]},
                            )
                        )

            next_steps.sort(
                key=lambda st: (
                    int(_step_cost_bits_total_v141(step=st, macro_support_by_id=macro_support_by_id)),
                    canonical_json_dumps(st.to_dict()),
                )
            )
            if len(next_steps) > max_next_steps:
                metrics["pruned_by_next_steps_cap"] = int(metrics.get("pruned_by_next_steps_cap", 0)) + int(
                    len(next_steps) - max_next_steps
                )
                next_steps = next_steps[:max_next_steps]
            for ns in next_steps:
                push(tuple(list(steps) + [ns]))

        trace_items.sort(key=lambda kv: kv[0])
        trace_programs = [row for _k, row in trace_items]

        if best_cost is None or not best_programs:
            return {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
                "kind": "arc_solver_result_v141",
                "status": "FAIL",
                "program_sig": "",
                "predicted_grid_hash": "",
                "failure_reason": {"kind": "SEARCH_BUDGET_EXCEEDED" if frontier else "MISSING_OPERATOR", "details": {}},
                "trace": {
                    "trace_programs": trace_programs,
                    "apply_cache_hits": int(metrics.get("apply_cache_hits", 0)),
                    "apply_cache_misses": int(metrics.get("apply_cache_misses", 0)),
                    "eval_cache_hits": int(metrics.get("eval_cache_hits", 0)),
                    "eval_cache_misses": int(metrics.get("eval_cache_misses", 0)),
                    "pruned_by_shape_reachability": int(pruned_shape),
                    "pruned_by_palette_reachability": int(pruned_palette),
                    "pruned_by_no_grid_modify_in_time": int(pruned_no_grid_modify),
                    "pruned_by_exception_prefix": int(metrics.get("pruned_by_exception_prefix", 0)),
                    "pruned_by_next_steps_cap": int(metrics.get("pruned_by_next_steps_cap", 0)),
                },
            }

        # Fail-closed decision uses minimal-cost programs only.
        min_cost = min(int(bp.get("cost_bits") or 0) for bp in best_programs)
        minimal = [bp for bp in best_programs if int(bp.get("cost_bits") or 0) == int(min_cost)]

        outputs: Dict[str, Dict[str, Any]] = {}
        for bp in minimal:
            h = str(bp.get("predicted_grid_hash") or "")
            if h and h not in outputs:
                outputs[h] = bp

        if len(outputs) == 1:
            only = list(outputs.values())[0]
            steps_rows = only.get("steps") if isinstance(only.get("steps"), list) else []
            return {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
                "kind": "arc_solver_result_v141",
                "status": "SOLVED",
                "program_sig": str(only.get("program_sig") or ""),
                "program_cost_bits": int(only.get("cost_bits") or 0),
                "program_steps": list(steps_rows),
                "predicted_grid": only.get("predicted_grid"),
                "predicted_grid_hash": str(only.get("predicted_grid_hash") or ""),
                "trace": {
                    "trace_programs": trace_programs,
                    "apply_cache_hits": int(metrics.get("apply_cache_hits", 0)),
                    "apply_cache_misses": int(metrics.get("apply_cache_misses", 0)),
                    "eval_cache_hits": int(metrics.get("eval_cache_hits", 0)),
                    "eval_cache_misses": int(metrics.get("eval_cache_misses", 0)),
                    "pruned_by_shape_reachability": int(pruned_shape),
                    "pruned_by_palette_reachability": int(pruned_palette),
                    "pruned_by_no_grid_modify_in_time": int(pruned_no_grid_modify),
                    "pruned_by_exception_prefix": int(metrics.get("pruned_by_exception_prefix", 0)),
                    "pruned_by_next_steps_cap": int(metrics.get("pruned_by_next_steps_cap", 0)),
                },
            }

        outs_sorted = sorted(outputs.values(), key=lambda x: str(x.get("predicted_grid_hash") or ""))
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
            "kind": "arc_solver_result_v141",
            "status": "UNKNOWN",
            "predicted_grids": [o.get("predicted_grid") for o in outs_sorted],
            "predicted_grids_by_solution": [
                {
                    "program_sig": str(o.get("program_sig") or ""),
                    "program_cost_bits": int(o.get("cost_bits") or 0),
                    "predicted_grid_hash": str(o.get("predicted_grid_hash") or ""),
                }
                for o in outs_sorted
            ],
            "failure_reason": {"kind": "AMBIGUOUS_RULE", "details": {"solutions": int(len(outputs))}},
            "trace": {
                "trace_programs": trace_programs,
                "apply_cache_hits": int(metrics.get("apply_cache_hits", 0)),
                "apply_cache_misses": int(metrics.get("apply_cache_misses", 0)),
                "eval_cache_hits": int(metrics.get("eval_cache_hits", 0)),
                "eval_cache_misses": int(metrics.get("eval_cache_misses", 0)),
                "pruned_by_shape_reachability": int(pruned_shape),
                "pruned_by_palette_reachability": int(pruned_palette),
                "pruned_by_no_grid_modify_in_time": int(pruned_no_grid_modify),
                "pruned_by_exception_prefix": int(metrics.get("pruned_by_exception_prefix", 0)),
                "pruned_by_next_steps_cap": int(metrics.get("pruned_by_next_steps_cap", 0)),
            },
        }

    if macro_templates_norm and not bool(macro_try_on_fail_only):
        return _solve_pass(allow_macros=True)

    r0 = _solve_pass(allow_macros=False)
    if str(r0.get("status") or "") != "FAIL":
        return r0
    if not macro_templates_norm:
        return r0
    return _solve_pass(allow_macros=True)
