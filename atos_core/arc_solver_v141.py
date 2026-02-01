from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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

# When a train-perfect solution is found, the solver traditionally continues searching to
# detect ambiguity (multiple train-perfect programs with different test predictions).
# In practice this can dominate runtime even when a 1-step concept_call already solves train+test.
# We cap the post-solution probe deterministically to keep evaluation feasible under fail-closed.
# After finding a train-perfect program, probe a bounded number of additional programs to
# detect ambiguity among minimal-cost solutions. This must stay bounded to avoid the harness
# timing out *after* a solution exists (which would incorrectly surface as SEARCH_BUDGET_EXCEEDED).
#
# Operational rule: within this bounded probe window, if we observe multiple minimal-cost
# train-perfect programs with distinct test predictions, return UNKNOWN (fail-closed).
# If we observe only one distinct minimal-cost prediction within the window, return SOLVED.
#
# This is a deterministic compromise: it avoids "timing out after success" while still
# surfacing ambiguity when it is easy to detect.
POST_SOLUTION_PROBE_PROGRAMS_V141 = 32

# Best-effort trace snapshot for timeout/interrupt handling in the ARC harness.
#
# Some harness configurations enforce a hard per-task wall-time and abort via signals. In those
# cases the solver cannot return a structured FAIL with trace_programs. We keep a deterministic,
# periodically-updated snapshot so the harness can still write mining evidence for timed-out tasks.
_TRACE_PROGRAMS_SNAPSHOT_V141: List[Dict[str, Any]] = []
_TRACE_METRICS_SNAPSHOT_V141: Dict[str, Any] = {}

# Per-process normalization caches to avoid paying O(|bank|) cost for every task.
# These are deterministic memoization of pure computations; they do not affect solver semantics.
_MACRO_TEMPLATES_NORM_CACHE_V141: Dict[Tuple[int, int], Tuple[Tuple[Dict[str, Any], ...], Dict[str, int]]] = {}
_CONCEPT_TEMPLATES_NORM_CACHE_V141: Dict[int, Tuple[Dict[str, Any], ...]] = {}


def _get_macro_templates_norm_cached_v141(
    *,
    macro_templates_all: Sequence[Dict[str, Any]],
    macro_max_templates: int,
) -> Tuple[Tuple[Dict[str, Any], ...], Dict[str, int]]:
    key = (int(id(macro_templates_all)), int(max(0, int(macro_max_templates))))
    hit = _MACRO_TEMPLATES_NORM_CACHE_V141.get(key)
    if hit is not None:
        return hit

    norm = _norm_macro_templates_v141([dict(r) for r in macro_templates_all if isinstance(r, dict)])
    norm = norm[: int(key[1])]
    support_by_id: Dict[str, int] = {str(r.get("macro_id") or ""): int(r.get("support") or 0) for r in norm if str(r.get("macro_id") or "")}
    out = (tuple(dict(r) for r in norm), dict(support_by_id))
    if len(_MACRO_TEMPLATES_NORM_CACHE_V141) >= 8:
        # Keep bounded (deterministic): evict oldest insertion order (CPython preserves).
        _MACRO_TEMPLATES_NORM_CACHE_V141.pop(next(iter(_MACRO_TEMPLATES_NORM_CACHE_V141)))
    _MACRO_TEMPLATES_NORM_CACHE_V141[key] = out
    return out


def _get_concept_templates_norm_cached_v141(*, concept_templates_all: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    key = int(id(concept_templates_all))
    hit = _CONCEPT_TEMPLATES_NORM_CACHE_V141.get(key)
    if hit is not None:
        return hit

    norm = _norm_concept_templates_v141([dict(r) for r in concept_templates_all if isinstance(r, dict)])
    out = tuple(dict(r) for r in norm)
    if len(_CONCEPT_TEMPLATES_NORM_CACHE_V141) >= 8:
        _CONCEPT_TEMPLATES_NORM_CACHE_V141.pop(next(iter(_CONCEPT_TEMPLATES_NORM_CACHE_V141)))
    _CONCEPT_TEMPLATES_NORM_CACHE_V141[key] = out
    return out


@dataclass(frozen=True)
class _TaskFeatV141:
    shape_rel: str
    palette_rel: str
    delta_density: str

    def key(self) -> Tuple[str, str, str]:
        return (str(self.shape_rel), str(self.palette_rel), str(self.delta_density))

def _shape_rel_for_pairs_v141(pairs: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]]) -> str:
    if not pairs:
        return "unknown"
    if all(a == b for a, b in pairs):
        return "same"
    if all(b == (a[1], a[0]) for a, b in pairs):
        return "swap_hw"
    ratios: Set[Tuple[int, int]] = set()
    for (hi, wi), (ho, wo) in pairs:
        if hi <= 0 or wi <= 0 or ho <= 0 or wo <= 0:
            return "shape_change_mixed"
        if ho % hi != 0 or wo % wi != 0:
            return "shape_change_mixed"
        ratios.add((int(ho // hi), int(wo // wi)))
    if len(ratios) == 1:
        ry, rx = list(ratios)[0]
        return f"scale_integer:{int(ry)}x{int(rx)}"
    return "shape_change_mixed"


def _palette_rel_v141(p_in: Set[int], p_out: Set[int]) -> str:
    if p_out.issubset(p_in):
        if p_in == p_out:
            return "equal"
        return "subset"
    if p_in.issubset(p_out):
        return "superset"
    return "other"


def _delta_density_bin_v141(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> str:
    diffs: List[float] = []
    for inp, out in train_pairs:
        hi, wi = grid_shape_v124(inp)
        ho, wo = grid_shape_v124(out)
        if (int(hi), int(wi)) != (int(ho), int(wo)) or int(hi) <= 0 or int(wi) <= 0:
            continue
        diff = 0
        for r in range(int(hi)):
            for c in range(int(wi)):
                if int(inp[int(r)][int(c)]) != int(out[int(r)][int(c)]):
                    diff += 1
        diffs.append(float(diff) / float(max(1, int(hi) * int(wi))))
    if not diffs:
        return "n/a"
    avg = sum(diffs) / float(len(diffs))
    if avg <= 0.10:
        return "sparse<=0.10"
    if avg <= 0.30:
        return "local<=0.30"
    return "dense>0.30"


def _task_feat_from_train_pairs_v141(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> _TaskFeatV141:
    shapes: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    pin: Set[int] = set()
    pout: Set[int] = set()
    for inp, out in train_pairs:
        shapes.append((tuple(int(x) for x in grid_shape_v124(inp)), tuple(int(x) for x in grid_shape_v124(out))))
        pin |= set(int(x) for x in unique_colors_v124(inp))
        pout |= set(int(x) for x in unique_colors_v124(out))
    return _TaskFeatV141(
        shape_rel=str(_shape_rel_for_pairs_v141(shapes)),
        palette_rel=str(_palette_rel_v141(pin, pout)),
        delta_density=str(_delta_density_bin_v141(train_pairs)),
    )


def _colors_bin_v141(n: int) -> str:
    x = int(n)
    if x <= 2:
        return "c<=2"
    if x <= 4:
        return "c<=4"
    if x <= 6:
        return "c<=6"
    if x <= 9:
        return "c<=9"
    return "c=10"


def _dim_bin_v141(h: int, w: int) -> str:
    m = max(int(h), int(w))
    if m <= 5:
        return "d<=5"
    if m <= 10:
        return "d<=10"
    if m <= 20:
        return "d<=20"
    return "d>20"


def _task_feat_key_v2_from_train_pairs_v141(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> Tuple[str, ...]:
    shapes: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    pin: Set[int] = set()
    pout: Set[int] = set()
    in_dims: List[Tuple[int, int]] = []
    out_dims: List[Tuple[int, int]] = []
    for inp, out in train_pairs:
        si = tuple(int(x) for x in grid_shape_v124(inp))
        so = tuple(int(x) for x in grid_shape_v124(out))
        shapes.append((si, so))
        in_dims.append(si)
        out_dims.append(so)
        pin |= set(int(x) for x in unique_colors_v124(inp))
        pout |= set(int(x) for x in unique_colors_v124(out))

    hi = max((int(h) for h, _w in in_dims), default=0)
    wi = max((int(w) for _h, w in in_dims), default=0)
    ho = max((int(h) for h, _w in out_dims), default=0)
    wo = max((int(w) for _h, w in out_dims), default=0)

    return (
        str(_shape_rel_for_pairs_v141(shapes)),
        str(_palette_rel_v141(pin, pout)),
        str(_delta_density_bin_v141(train_pairs)),
        f"n{int(len(list(train_pairs)))}",
        f"in_{_colors_bin_v141(len(pin))}",
        f"out_{_colors_bin_v141(len(pout))}",
        f"in_{_dim_bin_v141(int(hi), int(wi))}",
        f"out_{_dim_bin_v141(int(ho), int(wo))}",
    )


def reset_trace_snapshot_v141() -> None:
    global _TRACE_METRICS_SNAPSHOT_V141, _TRACE_PROGRAMS_SNAPSHOT_V141
    _TRACE_PROGRAMS_SNAPSHOT_V141 = []
    # Keep a non-empty dict so the harness can always distinguish:
    #  - "no snapshot yet" vs "snapshot keys missing"
    # This is telemetry-only and does not affect solver semantics.
    _TRACE_METRICS_SNAPSHOT_V141 = {"stage": "reset"}


def get_trace_snapshot_v141() -> List[Dict[str, Any]]:
    return [dict(r) for r in _TRACE_PROGRAMS_SNAPSHOT_V141]


def get_trace_metrics_snapshot_v141() -> Dict[str, Any]:
    # Best-effort snapshot used by the harness when a hard wall-time aborts the solver.
    return dict(_TRACE_METRICS_SNAPSHOT_V141)


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

    def _invalid() -> Dict[str, bool]:
        return {"grid": True, "objset": False, "obj": False, "bbox": False, "patch": False, "invalid": True}

    def _apply_prim_to_avail(*, op_id: str, args: Dict[str, Any]) -> bool:
        od = OP_DEFS_V141.get(str(op_id))
        if od is None:
            return False
        for r in od.reads:
            if not bool(avail.get(str(r), False)):
                return False
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
            "transpose",
            "reflect_h",
            "reflect_v",
            "translate",
            "symmetry_fill_h",
            "symmetry_fill_v",
            "downsample_mode",
            "fill_enclosed_region",
            "gravity",
            "paint_mask",
            "flood_fill",
            "overlay_self_translate",
            "propagate_color_translate",
            "propagate_nonbg_translate",
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
        return True

    for st in steps:
        op_id = str(st.op_id)
        if op_id in {"macro_call", "concept_call"}:
            steps_raw = st.args.get("steps")
            if not isinstance(steps_raw, list) or not steps_raw:
                return _invalid()
            for row in steps_raw:
                if not isinstance(row, dict):
                    return _invalid()
                op2 = str(row.get("op_id") or "")
                if not op2 or op2 in {"macro_call", "concept_call"}:
                    return _invalid()
                a2 = row.get("args") if isinstance(row.get("args"), dict) else {}
                if not _apply_prim_to_avail(op_id=str(op2), args=dict(a2)):
                    return _invalid()
            continue

        if not _apply_prim_to_avail(op_id=str(op_id), args=dict(st.args)):
            return _invalid()
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
    train_final_states: Tuple[StateV132, ...]
    test_final_state: StateV132
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


def apply_program_steps_to_grid_v141(*, grid: GridV124, program_steps: Sequence[Dict[str, Any]]) -> GridV124:
    """
    Deterministically apply a serialized program (list of step dicts) to a grid and return the output grid.

    This is used by the ARC harness for multi-test tasks (ARC-AGI-2): solve once, then apply the same
    candidate program(s) to multiple test inputs without re-running search.
    """
    apply_cache: Dict[Tuple[Any, ...], StateV132] = {}
    grid_hash_cache: Dict[GridV124, str] = {}
    metrics: Dict[str, int] = {}

    st = StateV132(grid=grid)
    for row in program_steps:
        if not isinstance(row, dict):
            raise ValueError("bad_program_step_row")
        op_id = str(row.get("op_id") or "")
        if not op_id:
            raise ValueError("bad_program_step_op")
        args = row.get("args") if isinstance(row.get("args"), dict) else {}
        st = _apply_step_cached_v141(
            state=st,
            op_id=str(op_id),
            args=dict(args),
            apply_cache=apply_cache,
            grid_hash_cache=grid_hash_cache,
            metrics=metrics,
        )
    return st.grid


def _eval_program_on_pairs_v141(
    *,
    program: ProgramV141,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    apply_cache: Dict[Tuple[Any, ...], StateV132],
    grid_hash_cache: Dict[GridV124, str],
    metrics: Dict[str, int],
) -> _EvalInfoV141:
    avail0 = _abstract_slots_after_steps_v141(program.steps)
    stage0 = _stage_from_avail_v141(avail0)
    got_shapes: List[Tuple[int, int]] = []
    got_palettes: List[Tuple[int, ...]] = []
    train_state_sigs: List[Dict[str, Any]] = []
    train_final_states: List[StateV132] = []
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
        train_state_sigs.append(_state_sig_dict_cached_v141(state=st, grid_hash_cache=grid_hash_cache))
        train_final_states.append(st)
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
        stt = StateV132(grid=test_in)
        test_grid = test_in

    loss = (int(loss_shape), int(loss_cells))
    vec_obj = {
        "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
        "kind": "train_state_vec_v141",
        "stage": str(stage0),
        "state_sigs": train_state_sigs,
        "test_state_sig": _state_sig_dict_cached_v141(state=stt, grid_hash_cache=grid_hash_cache),
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
        train_final_states=tuple(train_final_states),
        test_final_state=stt,
        test_grid=test_grid,
        got_shapes=tuple(got_shapes),
        got_palettes=tuple(got_palettes),
    )


def _csv_applicable_v141(
    *,
    steps: Sequence[Dict[str, Any]],
    train_final_states: Sequence[StateV132],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    apply_cache: Dict[Tuple[Any, ...], StateV132],
    grid_hash_cache: Dict[GridV124, str],
    metrics: Dict[str, int],
    max_loss_delta: int = -1,
    allow_slot_progress: bool = False,
) -> bool:
    """
    Deterministic applicability test for a concrete concept (CSG/CSV) at the *current prefix state*.

    The concept is applicable iff, when applied to each training final state, it proves causal
    utility *now* by either:
      - strictly reducing total TRAIN grid loss (shape mismatches are heavily penalized), OR
      - (optionally) improving semantic readiness (slot penalty) without worsening TRAIN loss.

    This enforces: "no concept enters search unless it proves causal utility now", while still
    allowing bounded, deterministic enabling moves under concept-as-policy.
    """
    score = _csv_applicability_score_v141(
        steps=steps,
        train_final_states=train_final_states,
        train_pairs=train_pairs,
        apply_cache=apply_cache,
        grid_hash_cache=grid_hash_cache,
        metrics=metrics,
        allow_slot_progress=bool(allow_slot_progress),
    )
    if score is None:
        return False
    dl, ds = (int(score[0]), int(score[1]))
    return _csv_gate_ok_v141(dl=dl, ds=ds, delta_max=int(max_loss_delta), allow_slot_progress=bool(allow_slot_progress))


def _csv_gate_ok_v141(*, dl: int, ds: int, delta_max: int, allow_slot_progress: bool) -> bool:
    """
    Deterministic causal gate for semantic actions (concept/macro calls).

    Strict by design:
      - Prefer concepts that reduce TRAIN grid loss: dl <= delta_max (delta_max typically -1).

    Optionally allow *semantic readiness* moves:
      - dl == 0 and ds < 0 (missing-slot penalty decreased or slot signature changed)

    Those moves are only computed for non-grid-writing steps inside
    `_csv_applicability_score_v141`, so this does not permit "do nothing" grid edits.
    """
    if int(dl) <= int(delta_max):
        return True
    if bool(allow_slot_progress) and int(dl) == 0 and int(ds) < 0:
        return True
    return False


def _csv_applicability_score_v141(
    *,
    steps: Sequence[Dict[str, Any]],
    train_final_states: Sequence[StateV132],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    apply_cache: Dict[Tuple[Any, ...], StateV132],
    grid_hash_cache: Dict[GridV124, str],
    metrics: Dict[str, int],
    allow_slot_progress: bool = False,
) -> Optional[Tuple[int, int]]:
    """
    Deterministic applicability score for a concrete concept at the *current prefix state*.

    Returns (delta_loss_total, delta_slot_penalty_total) if the concept is admissible.

    Admissible here is intentionally *minimal*: this is a fast, deterministic "does it help right
    now?" check used to prevent search from being flooded with useless closures.

    Criteria:
      - total TRAIN loss must strictly decrease (delta_loss_total < 0), OR
      - total TRAIN loss is unchanged (delta_loss_total == 0) but semantic readiness improves
        (delta_slot_penalty_total < 0).

    NOTE: We do not require per-example monotonic improvement; multi-step solutions often need
    intermediate concepts that may temporarily worsen one example while improving others.
    """
    if not steps or not train_pairs or not train_final_states:
        return None

    writes_grid = False
    for row in steps:
        if not isinstance(row, dict):
            return None
        op0 = str(row.get("op_id") or "")
        if not op0:
            return None
        od0 = OP_DEFS_V141.get(str(op0))
        if od0 is None:
            return None
        if "grid" in set(str(w) for w in od0.writes):
            writes_grid = True

    def _slot_penalty(st: StateV132) -> int:
        # Semantic “readiness” proxy: missing internal slots can block many CSGs.
        # This is deterministic and domain-agnostic (it does not look at ARC-specific IDs).
        return int(
            (1 if st.objset is None else 0)
            + (1 if st.obj is None else 0)
            + (1 if st.bbox is None else 0)
            + (1 if st.patch is None else 0)
        )

    def _loss_cells(pred: GridV124, out: GridV124) -> int:
        hp, wp = grid_shape_v124(pred)
        ho, wo = grid_shape_v124(out)
        if (int(hp), int(wp)) != (int(ho), int(wo)):
            return int(10000 + abs(int(hp) - int(ho)) + abs(int(wp) - int(wo)))
        mism = 0
        for r in range(int(ho)):
            for c in range(int(wo)):
                if int(pred[int(r)][int(c)]) != int(out[int(r)][int(c)]):
                    mism += 1
        return int(mism)

    base_loss = 0
    after_loss = 0
    base_slot = 0
    after_slot = 0
    improved_ex = 0
    worsened_ex = 0
    slot_sig_changed = False
    slot_quality_total = 0
    for st0, (_inp, want) in zip(train_final_states, train_pairs):
        base_i = int(_loss_cells(st0.grid, want))
        base_s = int(_slot_penalty(st0))
        base_loss += int(base_i)
        base_slot += int(base_s)
        base_sig = (
            "" if st0.objset is None else str(st0.objset.set_sig()),
            "" if st0.obj is None else str(st0.obj.object_sig()),
            None if st0.bbox is None else tuple(int(x) for x in st0.bbox.to_tuple()),
            "" if st0.patch is None else _grid_hash_cached_v141(g=st0.patch, cache=grid_hash_cache),
        )
        st = st0
        try:
            for row in steps:
                op0 = str(row.get("op_id") or "")
                if not op0:
                    return False
                a0 = row.get("args") if isinstance(row.get("args"), dict) else {}
                st = _apply_step_cached_v141(
                    state=st,
                    op_id=str(op0),
                    args=dict(a0),
                    apply_cache=apply_cache,
                    grid_hash_cache=grid_hash_cache,
                    metrics=metrics,
                )
        except Exception:
            return None
        after_i = int(_loss_cells(st.grid, want))
        after_s = int(_slot_penalty(st))
        after_loss += int(after_i)
        after_slot += int(after_s)

        # Destructiveness guard (generic, deterministic):
        #
        # Many candidates can "improve" TRAIN loss by a tiny margin while rewriting large regions
        # of the grid. Those actions massively increase branching and are a primary driver of
        # SEARCH_BUDGET_EXCEEDED under abstraction pressure.
        #
        # Enforce a bounded change budget per-example relative to the current mismatch.
        # - If the grid already matches (base_i==0), the concept wouldn't be admitted anyway (dl<0 required).
        # - Only applies to shape-aligned cases (base_i/after_i < 10000).
        if bool(writes_grid) and int(base_i) > 0 and int(base_i) < 10000 and int(after_i) < 10000:
            hb, wb = grid_shape_v124(st0.grid)
            ha, wa = grid_shape_v124(st.grid)
            if (int(hb), int(wb)) == (int(ha), int(wa)) and int(hb) > 0 and int(wb) > 0:
                area = int(hb) * int(wb)
                changed = 0
                for r in range(int(hb)):
                    row0 = st0.grid[int(r)]
                    row1 = st.grid[int(r)]
                    for c in range(int(wb)):
                        if int(row0[int(c)]) != int(row1[int(c)]):
                            changed += 1
                # Allow proportional changes for relocation-style moves (often ~2×diff),
                # plus a small deterministic slack for intermediate steps.
                slack = max(8, int(area) // 20)
                allowed = min(int(area), int(base_i) * 3 + int(slack))
                if int(changed) > int(allowed):
                    metrics["csv_rejected_destructive"] = int(metrics.get("csv_rejected_destructive", 0)) + 1
                    return None
        if int(after_i) < int(base_i):
            improved_ex += 1
        elif int(after_i) > int(base_i):
            worsened_ex += 1
        if bool(allow_slot_progress) and not bool(slot_sig_changed):
            after_sig = (
                "" if st.objset is None else str(st.objset.set_sig()),
                "" if st.obj is None else str(st.obj.object_sig()),
                None if st.bbox is None else tuple(int(x) for x in st.bbox.to_tuple()),
                "" if st.patch is None else _grid_hash_cached_v141(g=st.patch, cache=grid_hash_cache),
            )
            if after_sig != base_sig:
                slot_sig_changed = True
        if bool(allow_slot_progress) and not bool(writes_grid):
            # Rank slot-progress moves by "relevance to the diff" so the policy doesn't
            # pick arbitrary builders (e.g., wrong mask_by_color) when dl==0.
            #
            # Deterministic + generic: uses only (grid, want) deltas from demonstrations.
            # This score is used only for ordering among dl==0 moves, not as a success signal.
            hs, ws = grid_shape_v124(st0.grid)
            ho, wo = grid_shape_v124(want)
            if (int(hs), int(ws)) == (int(ho), int(wo)) and int(hs) > 0 and int(ws) > 0:
                # Diff bbox in the current prefix grid.
                rmin = cmin = 10**9
                rmax = cmax = -1
                diff_cells = 0
                for r in range(int(hs)):
                    for c in range(int(ws)):
                        if int(st0.grid[int(r)][int(c)]) != int(want[int(r)][int(c)]):
                            diff_cells += 1
                            rmin = min(int(rmin), int(r))
                            cmin = min(int(cmin), int(c))
                            rmax = max(int(rmax), int(r))
                            cmax = max(int(cmax), int(c))
                if int(diff_cells) > 0 and int(rmax) >= 0:
                    # Patch relevance (only when patch is full-grid aligned).
                    if st.patch is not None:
                        hp, wp = grid_shape_v124(st.patch)
                        if (int(hp), int(wp)) == (int(hs), int(ws)):
                            overlap = 0
                            patch_size = 0
                            for r in range(int(hs)):
                                for c in range(int(ws)):
                                    if int(st.patch[int(r)][int(c)]) != 0:
                                        patch_size += 1
                                        if int(st0.grid[int(r)][int(c)]) != int(want[int(r)][int(c)]):
                                            overlap += 1
                            if int(patch_size) > 0 and int(overlap) > 0:
                                slot_quality_total += int(overlap) * 1000 - int(patch_size)
                    # BBox relevance (intersection with diff bbox).
                    if st.bbox is not None:
                        try:
                            br0, bc0, br1, bc1 = (int(x) for x in st.bbox.to_tuple())
                            br0 = max(0, min(int(br0), int(hs) - 1))
                            br1 = max(0, min(int(br1), int(hs) - 1))
                            bc0 = max(0, min(int(bc0), int(ws) - 1))
                            bc1 = max(0, min(int(bc1), int(ws) - 1))
                            if br0 <= br1 and bc0 <= bc1:
                                bbox_area = int((br1 - br0 + 1) * (bc1 - bc0 + 1))
                                ir0 = max(int(br0), int(rmin))
                                ic0 = max(int(bc0), int(cmin))
                                ir1 = min(int(br1), int(rmax))
                                ic1 = min(int(bc1), int(cmax))
                                if ir0 <= ir1 and ic0 <= ic1:
                                    inter_area = int((ir1 - ir0 + 1) * (ic1 - ic0 + 1))
                                    if int(bbox_area) > 0 and int(inter_area) > 0:
                                        slot_quality_total += int(inter_area) * 1000 - int(bbox_area)
                        except Exception:
                            pass

    dl = int(after_loss - base_loss)
    ds = int(after_slot - base_slot)
    # Net-improving gate: a semantic move must strictly reduce *total* TRAIN loss.
    #
    # IMPORTANT: Do not require per-example monotonic improvement here.
    # Many valid multi-step abstractions trade off examples temporarily; insisting on
    # "never worsen any example" collapses CSV applicability to ~0 and reintroduces
    # SEARCH_BUDGET_EXCEEDED as the dominant failure mode.
    if int(dl) < 0 and int(improved_ex) > 0:
        return (int(dl), int(ds))
    if (
        bool(allow_slot_progress)
        and not bool(writes_grid)
        and int(dl) == 0
        and int(worsened_ex) == 0
    ):
        # Learn-mode only: allow *true slot mutations* (including patch/object/bbox updates) even
        # when they do not reduce the missing-slot penalty (ds==0). This is required to avoid
        # getting stuck with an irrelevant patch (e.g., mask_border) that blocks later obj_patch,
        # and to allow patch transforms (patch_rotate/reflect/transpose) which keep slots but
        # change their contents.
        #
        # This remains deterministic and fail-closed:
        # - only non-grid-writing steps are eligible
        # - no train loss worsening is allowed
        # - at least one slot signature must actually change
        ds2 = int(ds)
        if int(slot_quality_total) > 0:
            ds2 = min(int(ds2), -int(max(1, int(slot_quality_total))))
        if int(ds2) < 0:
            return (0, int(ds2))
        if bool(slot_sig_changed):
            if int(slot_quality_total) > 0:
                return (0, -int(max(1, int(slot_quality_total))))
            return (0, -1)
    return None


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
    # ARC-style grids frequently use 0 as background even when it is absent from inputs.
    # Always include 0 to avoid making common pad/translate/new_canvas pipelines inexpressible.
    cand: Set[int] = {0}
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


def _resolve_concept_csg_binders_v141(
    *,
    steps: Sequence[Dict[str, Any]],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
) -> Optional[List[Dict[str, Any]]]:
    """
    Resolve deterministic binder expressions inside CSG steps.

    A binder is encoded as an args value like {"bind": "<name>"} and must resolve to a primitive,
    JSON-serializable value (int/str/bool/None). This keeps concept_call atomic (no arg branching)
    while allowing reusable "semantic slots" derived from demonstrations.
    """
    if not steps:
        return []

    train_in = [p[0] for p in train_pairs]
    train_out = [p[1] for p in train_pairs]
    bg_counts: Dict[int, int] = {}
    for g in train_in + train_out + [test_in]:
        for row in g:
            for x in row:
                bg_counts[int(x)] = int(bg_counts.get(int(x), 0)) + 1
    items_bg = sorted(((int(c), int(n)) for c, n in bg_counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
    bg = int(items_bg[0][0]) if items_bg else 0
    in_h, in_w = grid_shape_v124(test_in)

    def _only_nonbg_color(grids: Sequence[GridV124]) -> Optional[int]:
        """
        If every grid has exactly one non-bg color and it is consistent across grids, return it.

        This is a generic semantic slot used to bind color arguments without per-task heuristics.
        """
        c0: Optional[int] = None
        for g in grids:
            cols = [int(x) for x in unique_colors_v124(g) if int(x) != int(bg)]
            cols = sorted(set(int(x) for x in cols))
            if len(cols) != 1:
                return None
            v = int(cols[0])
            if c0 is None:
                c0 = int(v)
            elif int(v) != int(c0):
                return None
        return int(c0) if c0 is not None else None

    def _top1_nonbg(grids: Sequence[GridV124]) -> Optional[int]:
        counts: Dict[int, int] = {}
        for g in grids:
            for row in g:
                for x in row:
                    v = int(x)
                    if int(v) == int(bg):
                        continue
                    counts[int(v)] = int(counts.get(int(v), 0)) + 1
        if not counts:
            return None
        items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
        return int(items[0][0]) if items else None

    in_only_nonbg = _only_nonbg_color(train_in + [test_in])
    out_only_nonbg = _only_nonbg_color(train_out)
    in_top_nonbg = _top1_nonbg(train_in + [test_in])
    out_top_nonbg = _top1_nonbg(train_out)

    out_h: Optional[int] = None
    out_w: Optional[int] = None
    ok_out_shape = True
    for _, out in train_pairs:
        ho, wo = grid_shape_v124(out)
        if out_h is None:
            out_h = int(ho)
            out_w = int(wo)
        elif int(ho) != int(out_h) or int(wo) != int(out_w):
            ok_out_shape = False
            break
    if not ok_out_shape:
        out_h = None
        out_w = None

    diff_from: Dict[int, int] = {}
    diff_to: Dict[int, int] = {}
    diff_rmin: Optional[int] = None
    diff_cmin: Optional[int] = None
    diff_rmax: Optional[int] = None
    diff_cmax: Optional[int] = None
    ok_diff = True
    for inp, out in train_pairs:
        hi, wi = grid_shape_v124(inp)
        ho, wo = grid_shape_v124(out)
        if (int(hi), int(wi)) != (int(ho), int(wo)):
            ok_diff = False
            break
        rmin = cmin = 10**9
        rmax = cmax = -1
        for r in range(int(hi)):
            for c in range(int(wi)):
                vi = int(inp[int(r)][int(c)])
                vo = int(out[int(r)][int(c)])
                if vi == vo:
                    continue
                rmin = min(int(rmin), int(r))
                cmin = min(int(cmin), int(c))
                rmax = max(int(rmax), int(r))
                cmax = max(int(cmax), int(c))
                diff_from[int(vi)] = int(diff_from.get(int(vi), 0)) + 1
                diff_to[int(vo)] = int(diff_to.get(int(vo), 0)) + 1
        if rmax >= 0:
            if diff_rmin is None:
                diff_rmin, diff_cmin, diff_rmax, diff_cmax = int(rmin), int(cmin), int(rmax), int(cmax)
            elif (
                int(rmin) != int(diff_rmin)
                or int(cmin) != int(diff_cmin)
                or int(rmax) != int(diff_rmax)
                or int(cmax) != int(diff_cmax)
            ):
                diff_rmin = diff_cmin = diff_rmax = diff_cmax = None

    def _top1(counts: Dict[int, int]) -> Optional[int]:
        if not counts:
            return None
        items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
        return int(items[0][0]) if items else None

    def _topk(counts: Dict[int, int], k: int) -> List[int]:
        if not counts or int(k) <= 0:
            return []
        items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
        return [int(c) for c, _n in items[: int(k)]]

    diff_from_top = _topk(diff_from, 3) if ok_diff else []
    diff_to_top = _topk(diff_to, 3) if ok_diff else []
    diff_from_top1 = int(diff_from_top[0]) if len(diff_from_top) >= 1 else None
    diff_from_top2 = int(diff_from_top[1]) if len(diff_from_top) >= 2 else None
    diff_from_top3 = int(diff_from_top[2]) if len(diff_from_top) >= 3 else None
    diff_to_top1 = int(diff_to_top[0]) if len(diff_to_top) >= 1 else None
    diff_to_top2 = int(diff_to_top[1]) if len(diff_to_top) >= 2 else None
    diff_to_top3 = int(diff_to_top[2]) if len(diff_to_top) >= 3 else None

    def _bbox_of_color(g: GridV124, *, color: int) -> Optional[Tuple[int, int, int, int]]:
        r0 = c0 = 10**9
        r1 = c1 = -1
        for r, row in enumerate(g):
            for c, x in enumerate(row):
                if int(x) != int(color):
                    continue
                r0 = min(r0, int(r))
                c0 = min(c0, int(c))
                r1 = max(r1, int(r))
                c1 = max(c1, int(c))
        if r1 < 0:
            return None
        return (int(r0), int(c0), int(r1), int(c1))

    offset_dy_dx: Optional[Tuple[int, int]] = None
    if diff_from_top1 is not None and diff_to_top1 is not None:
        ok = True
        dy0: Optional[int] = None
        dx0: Optional[int] = None
        for inp, out in train_pairs:
            bi = _bbox_of_color(inp, color=int(diff_from_top1))
            bo = _bbox_of_color(out, color=int(diff_to_top1))
            if bi is None or bo is None:
                ok = False
                break
            dy = int(bo[0] - bi[0])
            dx = int(bo[1] - bi[1])
            if dy0 is None:
                dy0 = int(dy)
                dx0 = int(dx)
            elif int(dy) != int(dy0) or int(dx) != int(dx0):
                ok = False
                break
        if ok and dy0 is not None and dx0 is not None:
            offset_dy_dx = (int(dy0), int(dx0))

    def _infer_select_obj_spec() -> Tuple[Optional[str], Optional[str], Optional[int]]:
        # Only compute if any select_obj_* binder is used in steps.
        needs_key = needs_order = needs_rank = False
        for st0 in steps:
            st = st0 if isinstance(st0, dict) else {}
            if str(st.get("op_id") or "") != "select_obj":
                continue
            a0 = st.get("args") if isinstance(st.get("args"), dict) else {}
            for v in a0.values():
                if not (isinstance(v, dict) and "bind" in v):
                    continue
                b = str(v.get("bind") or "")
                if b == "select_obj_key":
                    needs_key = True
                elif b == "select_obj_order":
                    needs_order = True
                elif b == "select_obj_rank":
                    needs_rank = True
        if not (needs_key or needs_order or needs_rank):
            return (None, None, None)

        # Candidate key set mirrors arc_ops_v132.select_obj supported keys (kept small + deterministic).
        keys = ["area", "width", "height", "bbox_area", "top", "left", "bottom", "right", "color", "dist_center"]
        orders = ["max", "min"]
        ranks = [0, 1, 2]

        def _resolve_bind_primitive(nm0: str) -> Optional[Any]:
            nm = str(nm0)
            if nm == "bg":
                return int(bg)
            if nm == "in_height":
                return int(in_h)
            if nm == "in_width":
                return int(in_w)
            if nm == "out_height":
                return int(out_h) if out_h is not None else None
            if nm == "out_width":
                return int(out_w) if out_w is not None else None
            if nm == "diff_from_top1":
                return int(diff_from_top1) if diff_from_top1 is not None else None
            if nm == "diff_from_top2":
                return int(diff_from_top2) if diff_from_top2 is not None else None
            if nm == "diff_from_top3":
                return int(diff_from_top3) if diff_from_top3 is not None else None
            if nm == "diff_to_top1":
                return int(diff_to_top1) if diff_to_top1 is not None else None
            if nm == "diff_to_top2":
                return int(diff_to_top2) if diff_to_top2 is not None else None
            if nm == "diff_to_top3":
                return int(diff_to_top3) if diff_to_top3 is not None else None
            if nm == "diff_rmin":
                return int(diff_rmin) if diff_rmin is not None else None
            if nm == "diff_cmin":
                return int(diff_cmin) if diff_cmin is not None else None
            if nm == "diff_rmax":
                return int(diff_rmax) if diff_rmax is not None else None
            if nm == "diff_cmax":
                return int(diff_cmax) if diff_cmax is not None else None
            if nm == "offset_dy_from_to_bbox":
                return int(offset_dy_dx[0]) if offset_dy_dx is not None else None
            if nm == "offset_dx_from_to_bbox":
                return int(offset_dy_dx[1]) if offset_dy_dx is not None else None
            return None

        def _loss(pred: GridV124, out: GridV124) -> int:
            hp, wp = grid_shape_v124(pred)
            ho, wo = grid_shape_v124(out)
            if (int(hp), int(wp)) != (int(ho), int(wo)):
                return int(10000 + abs(int(hp) - int(ho)) + abs(int(wp) - int(wo)))
            mism = 0
            for r in range(int(ho)):
                for c in range(int(wo)):
                    if int(pred[int(r)][int(c)]) != int(out[int(r)][int(c)]):
                        mism += 1
            return int(mism)

        best_key = str(keys[0])
        best_order = str(orders[0])
        best_rank = int(ranks[0])
        best_total = 10**18

        keys_it = list(keys) if needs_key else [str(keys[0])]
        orders_it = list(orders) if needs_order else [str(orders[0])]
        ranks_it = list(ranks) if needs_rank else [int(ranks[0])]

        for key in keys_it:
            for order in orders_it:
                for rank in ranks_it:
                    # Resolve binders for this candidate key/spec.
                    resolved_steps: List[Dict[str, Any]] = []
                    ok_steps = True
                    for st0 in steps:
                        st = st0 if isinstance(st0, dict) else {}
                        op0 = str(st.get("op_id") or "")
                        if not op0 or op0 in {"macro_call", "concept_call"}:
                            ok_steps = False
                            break
                        a0 = st.get("args") if isinstance(st.get("args"), dict) else {}
                        a2: Dict[str, Any] = {}
                        for kk in sorted(a0.keys()):
                            vv0 = a0[kk]
                            if isinstance(vv0, dict) and "bind" in vv0:
                                bname = vv0.get("bind")
                                if bname is None:
                                    ok_steps = False
                                    break
                                if str(bname) == "select_obj_key":
                                    a2[str(kk)] = str(key)
                                    continue
                                if str(bname) == "select_obj_order":
                                    a2[str(kk)] = str(order)
                                    continue
                                if str(bname) == "select_obj_rank":
                                    a2[str(kk)] = int(rank)
                                    continue
                                bb = _resolve_bind_primitive(str(bname))
                                if bb is None:
                                    ok_steps = False
                                    break
                                a2[str(kk)] = bb
                            else:
                                a2[str(kk)] = vv0
                        if not ok_steps:
                            break
                        resolved_steps.append({"op_id": str(op0), "args": dict(a2)})
                    if not ok_steps:
                        continue

                    total = 0
                    ok_run = True
                    for inp, out in train_pairs:
                        try:
                            st = StateV132(grid=inp)
                            for rs in resolved_steps:
                                st = apply_op_v141(
                                    state=st,
                                    op_id=str(rs.get("op_id") or ""),
                                    args=dict(rs.get("args") or {}),
                                )
                            total += int(_loss(st.grid, out))
                        except Exception:
                            ok_run = False
                            break
                    if not ok_run:
                        continue
                    if int(total) < int(best_total):
                        best_total = int(total)
                        best_key = str(key)
                        best_order = str(order)
                        best_rank = int(rank)
        return (str(best_key), str(best_order), int(best_rank))

    select_obj_key, select_obj_order, select_obj_rank = _infer_select_obj_spec()

    def _resolve_bind(name: str) -> Optional[Any]:
        nm = str(name)
        if nm == "bg":
            return int(bg)
        if nm == "in_only_nonbg":
            return int(in_only_nonbg) if in_only_nonbg is not None else None
        if nm == "out_only_nonbg":
            return int(out_only_nonbg) if out_only_nonbg is not None else None
        if nm == "in_top_nonbg":
            return int(in_top_nonbg) if in_top_nonbg is not None else None
        if nm == "out_top_nonbg":
            return int(out_top_nonbg) if out_top_nonbg is not None else None
        if nm == "in_height":
            return int(in_h)
        if nm == "in_width":
            return int(in_w)
        if nm == "out_height":
            return int(out_h) if out_h is not None else None
        if nm == "out_width":
            return int(out_w) if out_w is not None else None
        if nm == "diff_from_top1":
            return int(diff_from_top1) if diff_from_top1 is not None else None
        if nm == "diff_from_top2":
            return int(diff_from_top2) if diff_from_top2 is not None else None
        if nm == "diff_from_top3":
            return int(diff_from_top3) if diff_from_top3 is not None else None
        if nm == "diff_to_top1":
            return int(diff_to_top1) if diff_to_top1 is not None else None
        if nm == "diff_to_top2":
            return int(diff_to_top2) if diff_to_top2 is not None else None
        if nm == "diff_to_top3":
            return int(diff_to_top3) if diff_to_top3 is not None else None
        if nm == "diff_rmin":
            return int(diff_rmin) if diff_rmin is not None else None
        if nm == "diff_cmin":
            return int(diff_cmin) if diff_cmin is not None else None
        if nm == "diff_rmax":
            return int(diff_rmax) if diff_rmax is not None else None
        if nm == "diff_cmax":
            return int(diff_cmax) if diff_cmax is not None else None
        if nm == "offset_dy_from_to_bbox":
            return int(offset_dy_dx[0]) if offset_dy_dx is not None else None
        if nm == "offset_dx_from_to_bbox":
            return int(offset_dy_dx[1]) if offset_dy_dx is not None else None
        if nm == "select_obj_key":
            return str(select_obj_key) if select_obj_key is not None else None
        if nm == "select_obj_order":
            return str(select_obj_order) if select_obj_order is not None else None
        if nm == "select_obj_rank":
            return int(select_obj_rank) if select_obj_rank is not None else None
        return None

    resolved: List[Dict[str, Any]] = []
    infer_slots_by_kind: Dict[str, List[Tuple[int, str]]] = {}
    for st0 in steps:
        st = st0 if isinstance(st0, dict) else {}
        op0 = str(st.get("op_id") or "")
        if not op0:
            return None
        if op0 in {"macro_call", "concept_call"}:
            return None
        a0 = st.get("args") if isinstance(st.get("args"), dict) else {}
        a2: Dict[str, Any] = {}
        for k in sorted(a0.keys()):
            v = a0[k]
            if isinstance(v, dict) and "bind" in v:
                bname = v.get("bind")
                if bname is None:
                    return None
                vv = _resolve_bind(str(bname))
                if vv is None:
                    return None
                if not (isinstance(vv, (int, str, bool)) or vv is None):
                    return None
                a2[str(k)] = vv
            elif isinstance(v, dict) and "infer" in v:
                kind = v.get("infer")
                if kind is None:
                    return None
                kn = str(kind)
                if not kn:
                    return None
                infer_slots_by_kind.setdefault(str(kn), []).append((len(resolved), str(k)))
                # Placeholder; filled after deterministic inference.
                a2[str(k)] = {"__infer__": str(kn)}
            else:
                a2[str(k)] = v
        resolved.append({"op_id": str(op0), "args": dict(a2)})

    if not infer_slots_by_kind:
        return resolved

    def _ranked_colors(grids: Sequence[GridV124]) -> List[int]:
        counts: Dict[int, int] = {}
        for g in grids:
            for row in g:
                for x in row:
                    counts[int(x)] = int(counts.get(int(x), 0)) + 1
        items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
        return [int(c) for c, _n in items]

    # Deterministic per-kind candidate domains.
    train_in_grids = [p[0] for p in train_pairs] + [test_in]
    train_out_grids = [p[1] for p in train_pairs]
    def _infer_color_map_diff() -> Optional[Dict[str, int]]:
        """
        Infer a deterministic color mapping from TRAIN diffs.

        Returns a partial mapping dict (keys are strings, values are ints), suitable for map_colors.
        Unspecified colors are identity-mapped by the op.

        Fail-closed: if any input color maps to multiple output colors across TRAIN pairs, return None.
        """
        m: Dict[int, int] = {}
        for inp, out in train_pairs:
            hi, wi = grid_shape_v124(inp)
            ho, wo = grid_shape_v124(out)
            if (int(hi), int(wi)) != (int(ho), int(wo)):
                return None
            for r in range(int(hi)):
                for c in range(int(wi)):
                    ic = int(inp[int(r)][int(c)])
                    oc = int(out[int(r)][int(c)])
                    if ic == oc:
                        continue
                    if ic in m and int(m[int(ic)]) != int(oc):
                        return None
                    m[int(ic)] = int(oc)
        if not m:
            return None
        return {str(int(k)): int(m[int(k)]) for k in sorted(m.keys())}

    domains: Dict[str, List[Any]] = {}
    for kn in sorted(infer_slots_by_kind.keys()):
        if kn == "color_in":
            dom = _ranked_colors(train_in_grids)
        elif kn == "color_out":
            dom = _ranked_colors(train_out_grids)
        elif kn == "color_any":
            dom = _ranked_colors(train_in_grids + train_out_grids)
        elif kn == "color_map_diff":
            m = _infer_color_map_diff()
            if m is None:
                return None
            dom = [dict(m)]
        elif kn in {"dx_small", "dy_small"}:
            # Small, generic translation deltas.
            #
            # Note: many ARC tasks require translations larger than ±2. We keep this domain modest
            # (±5) and enforce a hard total-combination guardrail below by shrinking other domains
            # deterministically when needed.
            dom = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        elif kn in {"left_small", "top_small"}:
            # Paste placements / small absolute offsets.
            #
            # The previous fixed domain [-5..5] is too weak for many ARC tasks where the correct
            # placement is far from the origin (e.g., paste at a bbox-aligned location). To keep
            # inference deterministic and bounded, we include:
            #   - the centered small window [-5..5] (useful as translation shorthand)
            #   - a few demo-derived absolute candidates (diff bbox mins, non-bg bbox mins)
            #
            # This preserves the "no search in binder inference" invariant while making many
            # CSG templates (crop+paste, obj_patch+paste) expressible under tight budgets.
            base = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
            derived: List[int] = [0]
            # Diff bbox mins per demo (shape-aligned only).
            for inp, out in train_pairs:
                if tuple(int(x) for x in grid_shape_v124(inp)) != tuple(int(x) for x in grid_shape_v124(out)):
                    continue
                h, w = grid_shape_v124(inp)
                rmin = None
                cmin = None
                for r in range(int(h)):
                    for c in range(int(w)):
                        if int(inp[int(r)][int(c)]) != int(out[int(r)][int(c)]):
                            rr = int(r)
                            cc = int(c)
                            rmin = rr if rmin is None else min(int(rmin), rr)
                            cmin = cc if cmin is None else min(int(cmin), cc)
                if rmin is not None and cmin is not None:
                    if kn == "top_small":
                        derived.append(int(rmin))
                    else:
                        derived.append(int(cmin))
            # Non-bg bbox mins under inferred bg (if any).
            try:
                from .grid_v124 import bbox_nonzero_v124

                for g in train_out_grids + train_in_grids:
                    try:
                        r0, c0, _r1, _c1 = bbox_nonzero_v124(g, bg=int(bg))
                    except Exception:
                        continue
                    if kn == "top_small":
                        derived.append(int(r0))
                    else:
                        derived.append(int(c0))
            except Exception:
                pass

            # Prefer derived anchors (+/-1) first, then fall back to the small window.
            anchors = sorted(set(int(x) for x in derived))
            cand: List[int] = []
            for v in anchors:
                cand.extend([int(v), int(v - 1), int(v + 1)])
            cand.extend(base)

            dom2: List[int] = []
            seen: Set[int] = set()
            for x in cand:
                xx = int(x)
                if xx in seen:
                    continue
                seen.add(xx)
                dom2.append(xx)
                if len(dom2) >= 21:
                    break
            dom = dom2
        else:
            return None
        # Keep domains small and deterministic; tie-break already encoded in ranking.
        if kn.startswith("color_"):
            dom = dom[:8]
        if not dom:
            return None
        domains[str(kn)] = list(dom)

    def _total_combos(dom: Dict[str, List[Any]]) -> int:
        total = 1
        for kn in sorted(dom.keys()):
            total *= max(1, int(len(dom.get(kn) or [])))
        return int(total)

    # Total combination guardrail (deterministic).
    #
    # If too large, shrink domains deterministically. This keeps binder inference usable for
    # translation+color concepts (critical to reduce search branching), without turning inference
    # into an unbounded search.
    total = _total_combos(domains)
    if total > 2000:
        # Prefer shrinking color domains first (generic: colors tend to be less informative than
        # spatial deltas once the task is shape-aligned).
        color_caps = [4, 3, 2, 1]
        for cap in color_caps:
            dom2: Dict[str, List[Any]] = {}
            for kn, vals in domains.items():
                if kn.startswith("color_"):
                    dom2[kn] = list(vals)[: int(cap)]
                else:
                    dom2[kn] = list(vals)
            if _total_combos(dom2) <= 2000:
                domains = dom2
                total = _total_combos(domains)
                break

    if total > 2000:
        # If still too large, shrink translation/placement domains next.
        xy_caps = [7, 5, 3]
        for cap in xy_caps:
            dom2 = {k: list(v) for k, v in domains.items()}
            for kn in sorted(dom2.keys()):
                if kn in {"dx_small", "dy_small", "left_small", "top_small"}:
                    vals = list(dom2[kn])
                    # Keep centered window deterministically.
                    if len(vals) > int(cap):
                        mid = len(vals) // 2
                        half = int(cap) // 2
                        lo = max(0, int(mid - half))
                        hi = min(len(vals), int(lo + cap))
                        dom2[kn] = vals[lo:hi]
            if _total_combos(dom2) <= 2000:
                domains = dom2
                total = _total_combos(domains)
                break

    if total > 2000:
        return None

    def _materialize(assign: Dict[str, Any]) -> List[Dict[str, Any]]:
        out_steps: List[Dict[str, Any]] = []
        for row in resolved:
            op_id = str(row.get("op_id") or "")
            a0 = row.get("args") if isinstance(row.get("args"), dict) else {}
            a2: Dict[str, Any] = {}
            for kk in sorted(a0.keys()):
                vv = a0[kk]
                if isinstance(vv, dict) and "__infer__" in vv:
                    kn2 = str(vv.get("__infer__") or "")
                    if kn2 not in assign:
                        raise ValueError("missing_infer_assign")
                    a2[str(kk)] = assign[str(kn2)]
                else:
                    a2[str(kk)] = vv
            out_steps.append({"op_id": str(op_id), "args": dict(a2)})
        return out_steps

    def _loss_cells(pred: GridV124, out: GridV124) -> int:
        hp, wp = grid_shape_v124(pred)
        ho, wo = grid_shape_v124(out)
        if (int(hp), int(wp)) != (int(ho), int(wo)):
            return int(10000 + abs(int(hp) - int(ho)) + abs(int(wp) - int(wo)))
        mism = 0
        for r in range(int(ho)):
            for c in range(int(wo)):
                if int(pred[int(r)][int(c)]) != int(out[int(r)][int(c)]):
                    mism += 1
        return int(mism)

    best_total = 10**18
    best_assign: Optional[Dict[str, Any]] = None

    kinds = sorted(domains.keys())
    import itertools

    for combo in itertools.product(*[domains[k] for k in kinds]):
        assign = {k: v for k, v in zip(kinds, combo)}
        try:
            steps2 = _materialize(assign)
        except Exception:
            continue
        total_loss = 0
        ok = True
        for inp, out in train_pairs:
            try:
                st = StateV132(grid=inp)
                for rs in steps2:
                    st = apply_op_v141(state=st, op_id=str(rs.get("op_id") or ""), args=dict(rs.get("args") or {}))
                total_loss += int(_loss_cells(st.grid, out))
            except Exception:
                ok = False
                break
        if not ok:
            continue
        if int(total_loss) < int(best_total):
            best_total = int(total_loss)
            best_assign = dict(assign)
        elif int(total_loss) == int(best_total) and best_assign is not None:
            # Deterministic tie-break: lexicographic over (kind, value).
            if tuple((k, assign[k]) for k in kinds) < tuple((k, best_assign[k]) for k in kinds):
                best_assign = dict(assign)

    if best_assign is None:
        return None
    return _materialize(best_assign)


def _infer_direct_steps_v141(
    *, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124
) -> List[ProgramStepV141]:
    from .arc_solver_v140 import _infer_direct_steps_v140

    def _wrap_single_step_as_concept_call(step: ProgramStepV141, *, kind: str) -> ProgramStepV141:
        """
        Under fail-closed concept-as-policy, train-perfect direct inferences MUST still satisfy
        require_concept_call. Wrap any single primitive step as a 1-step concept_call.
        """
        op0 = str(step.op_id or "")
        if op0 in {"concept_call", "macro_call"}:
            return step
        inner = [ProgramStepV141(op_id=str(op0), args=dict(step.args))]
        body = {"kind": str(kind), "steps": [s.to_dict() for s in inner]}
        cid = "csg_prim_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))[:16]
        cost = int(step_cost_bits_v141(op_id=str(op0), args=dict(step.args)))
        return ProgramStepV141(
            op_id="concept_call",
            args={
                "concept_id": str(cid),
                "cost_bits": max(1, int(cost) - 1),
                "op_ids": [str(op0)],
                "steps": [s.to_dict() for s in inner],
            },
        )

    def _infer_patch_paint_csg_calls() -> List[ProgramStepV141]:
        """
        Infer small, generic patch→mask→paint closures that exactly match all train pairs.

        Rationale: many ARC tasks are solved by a short causal chain like:
          mask_by_color(seed) -> (outline|dilate) -> paint_mask(paint)
        Searching these compositions explodes combinatorially under tight caps; when they
        are fully determined by demonstrations, we can propose them as a single concept_call.
        """
        if not train_pairs:
            return []
        # Require aligned shapes (these closures do not change shape).
        for inp, out in train_pairs:
            if tuple(int(x) for x in grid_shape_v124(inp)) != tuple(int(x) for x in grid_shape_v124(out)):
                return []

        # Score candidate seed/paint colors from diffs (generic; no task-specific hacks).
        seed_scores: Dict[int, int] = {}
        paint_scores: Dict[int, int] = {}
        for inp, out in train_pairs:
            h, w = grid_shape_v124(inp)
            for r in range(int(h)):
                for c in range(int(w)):
                    vi = int(inp[int(r)][int(c)])
                    vo = int(out[int(r)][int(c)])
                    if vi == vo:
                        continue
                    seed_scores[vi] = int(seed_scores.get(vi, 0)) + 2
                    paint_scores[vo] = int(paint_scores.get(vo, 0)) + 3
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        rr = int(r + dr)
                        cc = int(c + dc)
                        if rr < 0 or cc < 0 or rr >= int(h) or cc >= int(w):
                            continue
                        vnb = int(inp[int(rr)][int(cc)])
                        seed_scores[vnb] = int(seed_scores.get(vnb, 0)) + 1

        def _top_colors(scores: Dict[int, int], fallback: List[int], k: int) -> List[int]:
            items = sorted(((int(c), int(n)) for c, n in scores.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
            out: List[int] = [int(c) for c, _n in items[: int(k)]]
            for c in fallback:
                cc = int(c)
                if cc not in out:
                    out.append(int(cc))
            return out[: int(max(1, int(k) * 2))]

        pal_out = _infer_palette_out_v141(train_pairs=train_pairs)
        pal_in: List[int] = []
        for inp, _out in train_pairs:
            for c in unique_colors_v124(inp):
                pal_in.append(int(c))
        pal_in_s = sorted(set(int(c) for c in pal_in))

        seed_cands = _top_colors(seed_scores, fallback=pal_in_s, k=6)
        paint_cands = _top_colors(paint_scores, fallback=pal_out, k=6)

        # Candidate closure templates (bounded).
        templates: List[List[Tuple[str, Dict[str, Any]]]] = []
        templates.append([("mask_by_color", {}), ("mask_outline", {}), ("paint_mask", {})])
        templates.append([("mask_by_color", {}), ("mask_dilate", {"steps": 1}), ("paint_mask", {})])
        templates.append([("mask_by_color", {}), ("mask_dilate", {"steps": 2}), ("paint_mask", {})])
        templates.append([("mask_by_color", {}), ("mask_box_dilate", {"radius": 1}), ("paint_mask", {})])
        templates.append([("mask_by_color", {}), ("mask_box_dilate", {"radius": 2}), ("paint_mask", {})])

        out_steps: List[ProgramStepV141] = []
        from .act import canonical_json_dumps

        def _try_one(seed: int, paint: int, tmpl: List[Tuple[str, Dict[str, Any]]]) -> Optional[ProgramStepV141]:
            inner: List[ProgramStepV141] = []
            for op_id, base_args in tmpl:
                a = dict(base_args)
                if str(op_id) == "mask_by_color":
                    a["color"] = int(seed)
                if str(op_id) == "paint_mask":
                    a["color"] = int(paint)
                inner.append(ProgramStepV141(op_id=str(op_id), args=dict(a)))

            # Verify train exactness.
            for inp, want in train_pairs:
                st = StateV132(grid=inp, objset=None, obj=None, bbox=None, patch=None)
                try:
                    for stp in inner:
                        st = apply_op_v141(state=st, op_id=str(stp.op_id), args=dict(stp.args))
                except Exception:
                    return None
                if not grid_equal_v124(st.grid, want):
                    return None

            body = {"kind": "direct_csg_v141", "steps": [s.to_dict() for s in inner]}
            cid = "csg_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))[:16]
            cost = 0
            for stp in inner:
                cost += int(step_cost_bits_v141(op_id=str(stp.op_id), args=dict(stp.args)))
            return ProgramStepV141(
                op_id="concept_call",
                args={
                    "concept_id": str(cid),
                    "cost_bits": max(1, int(cost) - 1),
                    "op_ids": [str(s.op_id) for s in inner],
                    "steps": [s.to_dict() for s in inner],
                },
            )

        # Deterministic enumeration order.
        for seed in seed_cands:
            for paint in paint_cands:
                if int(seed) == int(paint):
                    continue
                for tmpl in templates:
                    st = _try_one(int(seed), int(paint), tmpl)
                    if st is not None:
                        out_steps.append(st)
                        if len(out_steps) >= 8:
                            break
                if len(out_steps) >= 8:
                    break
            if len(out_steps) >= 8:
                break

        out_steps.sort(key=lambda s: (int(s.args.get("cost_bits") or 0), canonical_json_dumps(s.to_dict())))
        return out_steps

    def _infer_synth_csg_calls(*, patch_calls: Sequence[ProgramStepV141]) -> List[ProgramStepV141]:
        """
        Small deterministic concept synthesizer: try to compose a few high-level, demo-derived
        closures (e.g., patch paint CSGs) with simple grid ops (e.g., replace_color, fill_enclosed_region)
        and return an exact train-matching program as a single concept_call.

        This targets SEARCH_BUDGET_EXCEEDED cases where a correct solution exists within the
        existing operator alphabet but cannot be found under compositional search caps.
        """
        if not train_pairs:
            return []

        # Keep this synthesis bounded and generic: only attempt on aligned-shape tasks.
        for inp, out in train_pairs:
            if tuple(int(x) for x in grid_shape_v124(inp)) != tuple(int(x) for x in grid_shape_v124(out)):
                return []

        from .act import canonical_json_dumps

        wants = [p[1] for p in train_pairs]

        def _loss_cells(got: GridV124, want: GridV124) -> Tuple[int, int]:
            if tuple(int(x) for x in grid_shape_v124(got)) != tuple(int(x) for x in grid_shape_v124(want)):
                return (1, 10**9)
            h, w = grid_shape_v124(want)
            diff = 0
            for r in range(int(h)):
                for c in range(int(w)):
                    if int(got[int(r)][int(c)]) != int(want[int(r)][int(c)]):
                        diff += 1
            return (0, int(diff))

        # Candidate high-level operations are represented as lists of primitive steps.
        op_cands: List[List[ProgramStepV141]] = []

        # Demo stats used to bound candidate arguments.
        diff_from_counts: Dict[int, int] = {}
        diff_to_counts: Dict[int, int] = {}
        in_counts: Dict[int, int] = {}
        out_counts: Dict[int, int] = {}
        for inp, out in train_pairs:
            for row in inp:
                for v in row:
                    in_counts[int(v)] = int(in_counts.get(int(v), 0)) + 1
            for row in out:
                for v in row:
                    out_counts[int(v)] = int(out_counts.get(int(v), 0)) + 1
            h, w = grid_shape_v124(inp)
            for r in range(int(h)):
                for c in range(int(w)):
                    vi = int(inp[int(r)][int(c)])
                    vo = int(out[int(r)][int(c)])
                    if vi == vo:
                        continue
                    diff_from_counts[int(vi)] = int(diff_from_counts.get(int(vi), 0)) + 1
                    diff_to_counts[int(vo)] = int(diff_to_counts.get(int(vo), 0)) + 1
        for row in test_in:
            for v in row:
                in_counts[int(v)] = int(in_counts.get(int(v), 0)) + 1

        def _top_k_colors(counts: Dict[int, int], k: int) -> List[int]:
            items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
            return [int(c) for c, _n in items[: int(k)]]

        bg_candidates = _infer_bg_candidates_v141(grids=[p[0] for p in train_pairs] + [test_in])
        palette_out = _infer_palette_out_v141(train_pairs=train_pairs)

        # Use already-inferred patch paint closures (bounded to <=8 in _infer_patch_paint_csg_calls).
        for st in patch_calls:
            if str(st.op_id) != "concept_call":
                continue
            steps_raw = st.args.get("steps")
            if not isinstance(steps_raw, list) or not steps_raw:
                continue
            inner: List[ProgramStepV141] = []
            ok = True
            for row in steps_raw:
                if not isinstance(row, dict):
                    ok = False
                    break
                op_id = str(row.get("op_id") or "")
                args = row.get("args") if isinstance(row.get("args"), dict) else {}
                if not op_id or op_id in {"macro_call", "concept_call"}:
                    ok = False
                    break
                inner.append(ProgramStepV141(op_id=str(op_id), args=dict(args)))
            if ok and inner:
                op_cands.append(inner)

        # Also propose a small set of *low-loss* patch paint closures (even if they don't solve alone).
        # This helps synthesize multi-step concepts where each sub-step is imperfect in isolation.
        seed_cands = _top_k_colors(diff_from_counts, 4) or _top_k_colors(in_counts, 4)
        paint_cands = _top_k_colors(diff_to_counts, 4) or _top_k_colors(out_counts, 4) or list(palette_out[:4])
        and_cands = _top_k_colors(out_counts, 3) or list(palette_out[:3])

        patch_templates: List[List[Tuple[str, Dict[str, Any]]]] = []
        patch_templates.append([("mask_by_color", {}), ("mask_outline", {}), ("paint_mask", {})])
        patch_templates.append([("mask_by_color", {}), ("mask_border", {}), ("paint_mask", {})])
        patch_templates.append([("mask_by_color", {}), ("mask_dilate", {"steps": 1}), ("paint_mask", {})])
        patch_templates.append([("mask_by_color", {}), ("mask_dilate", {"steps": 2}), ("paint_mask", {})])
        patch_templates.append([("mask_by_color", {}), ("mask_box_dilate", {"radius": 1}), ("paint_mask", {})])
        patch_templates.append([("mask_by_color", {}), ("mask_box_dilate", {"radius": 2}), ("paint_mask", {})])

        patch_templates_and: List[List[Tuple[str, Dict[str, Any]]]] = []
        patch_templates_and.append([("mask_by_color", {}), ("mask_dilate", {"steps": 1}), ("mask_and_color", {}), ("paint_mask", {})])
        patch_templates_and.append([("mask_by_color", {}), ("mask_dilate", {"steps": 2}), ("mask_and_color", {}), ("paint_mask", {})])
        patch_templates_and.append([("mask_by_color", {}), ("mask_box_dilate", {"radius": 1}), ("mask_and_color", {}), ("paint_mask", {})])

        scored_patch_ops: List[Tuple[Tuple[int, int, int, str], List[ProgramStepV141]]] = []
        for seed in seed_cands:
            for paint in paint_cands:
                if int(seed) == int(paint):
                    continue
                # basic templates
                for tmpl in patch_templates:
                    inner: List[ProgramStepV141] = []
                    for op_id, base_args in tmpl:
                        a = dict(base_args)
                        if str(op_id) == "mask_by_color":
                            a["color"] = int(seed)
                        if str(op_id) == "paint_mask":
                            a["color"] = int(paint)
                        inner.append(ProgramStepV141(op_id=str(op_id), args=dict(a)))
                    # score
                    ok = True
                    ls_sum = 0
                    lc_sum = 0
                    try:
                        for (inp, want) in train_pairs:
                            st = StateV132(grid=inp, objset=None, obj=None, bbox=None, patch=None)
                            for s in inner:
                                st = apply_op_v141(state=st, op_id=str(s.op_id), args=dict(s.args))
                            ls, lc = _loss_cells(st.grid, want)
                            ls_sum += int(ls)
                            lc_sum += int(lc) if int(ls) == 0 else 0
                    except Exception:
                        ok = False
                    if not ok:
                        continue
                    cost = 0
                    for s in inner:
                        cost += int(step_cost_bits_v141(op_id=str(s.op_id), args=dict(s.args)))
                    sig = sha256_hex(canonical_json_dumps([s.to_dict() for s in inner]).encode("utf-8"))
                    scored_patch_ops.append(((int(ls_sum), int(lc_sum), int(cost), str(sig)), inner))
                # templates with mask_and_color
                for and_col in and_cands:
                    for tmpl in patch_templates_and:
                        inner = []
                        for op_id, base_args in tmpl:
                            a = dict(base_args)
                            if str(op_id) == "mask_by_color":
                                a["color"] = int(seed)
                            if str(op_id) == "mask_and_color":
                                a["color"] = int(and_col)
                            if str(op_id) == "paint_mask":
                                a["color"] = int(paint)
                            inner.append(ProgramStepV141(op_id=str(op_id), args=dict(a)))
                        ok = True
                        ls_sum = 0
                        lc_sum = 0
                        try:
                            for (inp, want) in train_pairs:
                                st = StateV132(grid=inp, objset=None, obj=None, bbox=None, patch=None)
                                for s in inner:
                                    st = apply_op_v141(state=st, op_id=str(s.op_id), args=dict(s.args))
                                ls, lc = _loss_cells(st.grid, want)
                                ls_sum += int(ls)
                                lc_sum += int(lc) if int(ls) == 0 else 0
                        except Exception:
                            ok = False
                        if not ok:
                            continue
                        cost = 0
                        for s in inner:
                            cost += int(step_cost_bits_v141(op_id=str(s.op_id), args=dict(s.args)))
                        sig = sha256_hex(canonical_json_dumps([s.to_dict() for s in inner]).encode("utf-8"))
                        scored_patch_ops.append(((int(ls_sum), int(lc_sum), int(cost), str(sig)), inner))

        scored_patch_ops.sort(key=lambda kv: kv[0])
        for _k, ops in scored_patch_ops[:8]:
            op_cands.append(list(ops))

        # Add a small, demo-driven recolor domain (replace_color).
        from_cands = _top_k_colors(diff_from_counts, 4)
        to_cands = _top_k_colors(diff_to_counts, 4) or _top_k_colors(out_counts, 4)
        for fc in from_cands:
            for tc in to_cands:
                if int(fc) == int(tc):
                    continue
                op_cands.append([ProgramStepV141(op_id="replace_color", args={"from_color": int(fc), "to_color": int(tc)})])

        # Add a few fill_enclosed_region variants (already generic and cheap).
        fill_cands = _top_k_colors(diff_to_counts, 4) or list(palette_out[:4])
        for bg in list(bg_candidates[:2] if bg_candidates else [0]):
            for fill in fill_cands:
                if int(fill) == int(bg):
                    continue
                op_cands.append([ProgramStepV141(op_id="fill_enclosed_region", args={"bg": int(bg), "fill": int(fill)})])

        # Add simple bbox-driven drawing (generic, deterministic): bbox_by_color -> (paint_rect|draw_rect_border).
        # This targets tasks where the demonstration implies explicit geometry marking.
        bbox_cands = _top_k_colors(diff_from_counts, 4) or _top_k_colors(in_counts, 4) or list(palette_out[:4])
        draw_cands = _top_k_colors(diff_to_counts, 4) or _top_k_colors(out_counts, 4) or list(palette_out[:4])
        for bc in bbox_cands[:4]:
            for dc in draw_cands[:4]:
                op_cands.append(
                    [
                        ProgramStepV141(op_id="bbox_by_color", args={"color": int(bc)}),
                        ProgramStepV141(op_id="draw_rect_border", args={"color": int(dc), "thickness": 1}),
                    ]
                )
                op_cands.append(
                    [
                        ProgramStepV141(op_id="bbox_by_color", args={"color": int(bc)}),
                        ProgramStepV141(op_id="paint_rect", args={"color": int(dc)}),
                    ]
                )

        # Add a small set of global transforms and translations (generic, bounded).
        for op_id in ["rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v", "transpose"]:
            op_cands.append([ProgramStepV141(op_id=str(op_id), args={})])

        # Patch transforms (operate only when a patch exists). These are generic and enable
        # synthesis of reflected/rotated masks before painting/pasting.
        for op_id in [
            "patch_rotate90",
            "patch_rotate180",
            "patch_rotate270",
            "patch_reflect_h",
            "patch_reflect_v",
            "patch_transpose",
        ]:
            op_cands.append([ProgramStepV141(op_id=str(op_id), args={})])

        # Patch-mask refinements (operate only when a patch exists).
        for bg in list(bg_candidates[:2] if bg_candidates else [0]):
            op_cands.append([ProgramStepV141(op_id="mask_and_nonbg", args={"bg": int(bg)})])

        # Patch translation (operate only when a patch exists). Enables shifting masks/patches
        # without abusing paste (which shifts only at apply-time, not in patch coordinates).
        patch_shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dy, dx in patch_shifts:
            for pad in list(bg_candidates[:2] if bg_candidates else [0]):
                op_cands.append([ProgramStepV141(op_id="patch_translate", args={"dx": int(dx), "dy": int(dy), "pad": int(pad)})])

        # Paint the current patch as a mask, but only on background cells. This is a common
        # ARC pattern and avoids overpainting when the demos imply "fill empty only".
        for bg in list(bg_candidates[:2] if bg_candidates else [0]):
            for c in list(palette_out[:4] if palette_out else [1]):
                op_cands.append([ProgramStepV141(op_id="paint_mask", args={"color": int(c), "mode": "only_bg", "bg": int(bg)})])

        # Patch paste placement candidates (operate only when a patch exists).
        # Derived deterministically from train output geometry and change bboxes.
        try:
            from .grid_v124 import bbox_nonzero_v124

            paste_positions: Set[Tuple[int, int]] = {(0, 0)}
            for bg in list(bg_candidates[:2] if bg_candidates else [0]):
                for inp, out in train_pairs:
                    try:
                        r0, c0, _, _ = bbox_nonzero_v124(out, bg=int(bg))
                        paste_positions.add((int(r0), int(c0)))
                    except Exception:
                        pass
                    try:
                        hi, wi = grid_shape_v124(inp)
                        ho, wo = grid_shape_v124(out)
                        if (int(hi), int(wi)) != (int(ho), int(wo)) or int(hi) <= 0 or int(wi) <= 0:
                            continue
                        rmin = None
                        cmin = None
                        for r in range(int(hi)):
                            for c in range(int(wi)):
                                if int(inp[int(r)][int(c)]) != int(out[int(r)][int(c)]):
                                    rr = int(r)
                                    cc = int(c)
                                    rmin = rr if rmin is None else min(int(rmin), rr)
                                    cmin = cc if cmin is None else min(int(cmin), cc)
                        if rmin is not None and cmin is not None:
                            paste_positions.add((int(rmin), int(cmin)))
                    except Exception:
                        pass

            transparent_cands: List[int] = [0]
            if bg_candidates:
                transparent_cands.append(int(bg_candidates[0]))
            transparent_cands_s = sorted(set(int(x) for x in transparent_cands))
            for top, left in sorted(paste_positions)[:8]:
                for tr in transparent_cands_s:
                    op_cands.append([ProgramStepV141(op_id="paste", args={"top": int(top), "left": int(left), "transparent": int(tr)})])
        except Exception:
            pass

        shifts8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dy, dx in shifts8:
            for bg in list(bg_candidates[:2] if bg_candidates else [0]):
                op_cands.append([ProgramStepV141(op_id="translate", args={"dx": int(dx), "dy": int(dy), "pad": int(bg)})])

        try:
            from .grid_v124 import bbox_nonzero_v124

            derived: Set[Tuple[int, int, int]] = set()
            for inp, out in train_pairs:
                hi, wi = grid_shape_v124(inp)
                if tuple(int(x) for x in grid_shape_v124(out)) != (int(hi), int(wi)):
                    continue
                for bg in list(bg_candidates[:2] if bg_candidates else [0]):
                    try:
                        ir0, ic0, _, _ = bbox_nonzero_v124(inp, bg=int(bg))
                        or0, oc0, _, _ = bbox_nonzero_v124(out, bg=int(bg))
                        dy = int(or0 - ir0)
                        dx = int(oc0 - ic0)
                        if dy == 0 and dx == 0:
                            continue
                        derived.add((int(dy), int(dx), int(bg)))
                    except Exception:
                        continue
            for dy, dx, bg in sorted(derived)[:12]:
                op_cands.append([ProgramStepV141(op_id="translate", args={"dx": int(dx), "dy": int(dy), "pad": int(bg)})])
        except Exception:
            pass

        # Deterministic de-dup of op candidates.
        seen_ops: Set[str] = set()
        uniq_ops: List[List[ProgramStepV141]] = []
        for ops in op_cands:
            body = [s.to_dict() for s in ops]
            sig = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
            if sig in seen_ops:
                continue
            seen_ops.add(sig)
            uniq_ops.append(ops)

        # Beam search over a small number of composed operations.
        class _Node:
            __slots__ = ("steps", "train_states", "test_state", "loss", "cost_bits", "sig")

            def __init__(self, *, steps: List[ProgramStepV141], train_states: List[StateV132], test_state: StateV132, loss: Tuple[int, int], cost_bits: int):
                self.steps = steps
                self.train_states = train_states
                self.test_state = test_state
                self.loss = loss
                self.cost_bits = cost_bits
                self.sig = sha256_hex(canonical_json_dumps([s.to_dict() for s in steps]).encode("utf-8"))

        base_train_states = [StateV132(grid=inp, objset=None, obj=None, bbox=None, patch=None) for inp, _out in train_pairs]
        base_test_state = StateV132(grid=test_in, objset=None, obj=None, bbox=None, patch=None)

        base_loss_shape = 0
        base_loss_cells = 0
        for st0, want in zip(base_train_states, wants):
            ls, lc = _loss_cells(st0.grid, want)
            base_loss_shape += int(ls)
            base_loss_cells += int(lc) if int(ls) == 0 else 0
        beam: List[_Node] = [_Node(steps=[], train_states=base_train_states, test_state=base_test_state, loss=(int(base_loss_shape), int(base_loss_cells)), cost_bits=0)]

        max_ops = 4
        max_prims = 10
        beam_k = 32

        def _node_key(n: _Node) -> Tuple[int, int, int, str]:
            return (int(n.loss[0]), int(n.loss[1]), int(n.cost_bits), str(n.sig))

        for _depth in range(int(max_ops)):
            next_beam: List[_Node] = []
            for node in beam:
                for ops in uniq_ops:
                    if int(len(node.steps) + len(ops)) > int(max_prims):
                        continue
                    train_states = [st for st in node.train_states]
                    test_state = node.test_state
                    ok = True
                    try:
                        # Apply to each train state.
                        new_train: List[StateV132] = []
                        for st in train_states:
                            st2 = st
                            for s in ops:
                                st2 = apply_op_v141(state=st2, op_id=str(s.op_id), args=dict(s.args))
                            new_train.append(st2)
                        stt = test_state
                        for s in ops:
                            stt = apply_op_v141(state=stt, op_id=str(s.op_id), args=dict(s.args))
                        new_test = stt
                    except Exception:
                        ok = False
                    if not ok:
                        continue

                    loss_shape = 0
                    loss_cells = 0
                    for st2, want in zip(new_train, wants):
                        ls, lc = _loss_cells(st2.grid, want)
                        loss_shape += int(ls)
                        loss_cells += int(lc) if int(ls) == 0 else 0
                        if int(loss_shape) == 0 and int(loss_cells) == 0:
                            inner = list(node.steps) + list(ops)
                            # Pack into a single concept_call for fail-closed concept-as-policy.
                            body = {"kind": "direct_synth_csg_v141", "steps": [s.to_dict() for s in inner]}
                            cid = "csg_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))[:16]
                            cost = 0
                            for stp in inner:
                                cost += int(step_cost_bits_v141(op_id=str(stp.op_id), args=dict(stp.args)))
                            return [
                                ProgramStepV141(
                                    op_id="concept_call",
                                    args={
                                        "concept_id": str(cid),
                                        # Do not undercut primitive costs: preserving cost ties is
                                        # required for fail-closed ambiguity detection.
                                        "cost_bits": int(cost),
                                        "op_ids": [str(s.op_id) for s in inner],
                                        "steps": [s.to_dict() for s in inner],
                                    },
                                )
                            ]

                    cost_bits = int(node.cost_bits)
                    for stp in ops:
                        cost_bits += int(step_cost_bits_v141(op_id=str(stp.op_id), args=dict(stp.args)))
                    next_beam.append(
                        _Node(
                            steps=list(node.steps) + list(ops),
                            train_states=new_train,
                            test_state=new_test,
                            loss=(int(loss_shape), int(loss_cells)),
                            cost_bits=int(cost_bits),
                        )
                    )

            if not next_beam:
                break
            next_beam.sort(key=_node_key)
            # Keep deterministic diversity by key order only.
            uniq_next: List[_Node] = []
            seen: Set[str] = set()
            for n in next_beam:
                if n.sig in seen:
                    continue
                seen.add(n.sig)
                uniq_next.append(n)
                if len(uniq_next) >= int(beam_k):
                    break
            beam = uniq_next

        return []

    def _infer_tile_signature_fill_step() -> Optional[ProgramStepV141]:
        # Detect the common ARC pattern:
        # - output is a tiling of uniform-color blocks
        # - each input tile pattern maps deterministically to an output tile color
        #
        # This is generic (grid→grid), deterministic, and learns only from train pairs.
        if not train_pairs:
            return None
        # Require aligned shapes per example (tile mapping is positional).
        for inp, out in train_pairs:
            if tuple(int(x) for x in grid_shape_v124(inp)) != tuple(int(x) for x in grid_shape_v124(out)):
                return None

        out_shapes = [tuple(int(x) for x in grid_shape_v124(out)) for _inp, out in train_pairs]
        hs = [int(h) for h, _w in out_shapes]
        ws = [int(w) for _h, w in out_shapes]
        if any(int(h) <= 0 or int(w) <= 0 for h, w in out_shapes):
            return None

        def _small_divisors(n: int) -> List[int]:
            out: List[int] = []
            for d in range(1, min(10, int(n)) + 1):
                if int(n) % int(d) == 0:
                    out.append(int(d))
            return out

        cand_h: Optional[Set[int]] = None
        cand_w: Optional[Set[int]] = None
        for h in hs:
            ds = set(_small_divisors(int(h)))
            cand_h = ds if cand_h is None else (cand_h & ds)
        for w in ws:
            ds = set(_small_divisors(int(w)))
            cand_w = ds if cand_w is None else (cand_w & ds)
        if not cand_h or not cand_w:
            return None

        def _output_is_uniform_tiled(out_grid: GridV124, tile_h: int, tile_w: int) -> bool:
            h, w = grid_shape_v124(out_grid)
            if int(h) % int(tile_h) != 0 or int(w) % int(tile_w) != 0:
                return False
            for r0 in range(0, int(h), int(tile_h)):
                for c0 in range(0, int(w), int(tile_w)):
                    v0 = int(out_grid[int(r0)][int(c0)])
                    for rr in range(int(tile_h)):
                        for cc in range(int(tile_w)):
                            if int(out_grid[int(r0 + rr)][int(c0 + cc)]) != int(v0):
                                return False
            return True

        bg_candidates = _infer_bg_candidates_v141(grids=[p[0] for p in train_pairs] + [test_in])

        # Search candidate tile sizes and return the first one that yields a consistent
        # signature→color mapping and covers the test tiles.
        for bg in bg_candidates:
            bgc = int(bg)

            def _tile_key(tile: List[List[int]]) -> str:
                # Match arc_ops_v141.py::tile_signature_fill: canonicalize with bg→0, other colors sorted.
                cols: Set[int] = set()
                for row in tile:
                    for v in row:
                        vv = int(v)
                        if vv != int(bgc):
                            cols.add(int(vv))
                order = sorted(int(x) for x in cols)
                ren: Dict[int, int] = {int(bgc): 0}
                for i, c in enumerate(order):
                    ren[int(c)] = int(i + 1)
                norm: List[List[int]] = []
                for row in tile:
                    norm.append([int(ren.get(int(v), 0)) for v in row])
                body = {"kind": "tile_sig_v141", "tile": norm}
                return sha256_hex(canonical_json_dumps(body).encode("utf-8"))

            for tile_h in sorted(cand_h):
                for tile_w in sorted(cand_w):
                    if int(tile_h) == 1 and int(tile_w) == 1:
                        continue
                    ok = True
                    for _inp, out in train_pairs:
                        if not _output_is_uniform_tiled(out, int(tile_h), int(tile_w)):
                            ok = False
                            break
                    if not ok:
                        continue

                    mapping: Dict[str, int] = {}
                    for inp, out in train_pairs:
                        h, w = grid_shape_v124(inp)
                        for r0 in range(0, int(h), int(tile_h)):
                            for c0 in range(0, int(w), int(tile_w)):
                                tile = [
                                    [int(inp[int(r0 + rr)][int(c0 + cc)]) for cc in range(int(tile_w))]
                                    for rr in range(int(tile_h))
                                ]
                                k = _tile_key(tile)
                                col = int(out[int(r0)][int(c0)])
                                prev = mapping.get(str(k))
                                if prev is not None and int(prev) != int(col):
                                    ok = False
                                    break
                                mapping[str(k)] = int(col)
                            if not ok:
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    if len(mapping) > 64:
                        continue

                    ht, wt = grid_shape_v124(test_in)
                    if int(ht) % int(tile_h) != 0 or int(wt) % int(tile_w) != 0:
                        continue
                    covered = True
                    for r0 in range(0, int(ht), int(tile_h)):
                        for c0 in range(0, int(wt), int(tile_w)):
                            tile = [
                                [int(test_in[int(r0 + rr)][int(c0 + cc)]) for cc in range(int(tile_w))]
                                for rr in range(int(tile_h))
                            ]
                            if _tile_key(tile) not in mapping:
                                covered = False
                                break
                        if not covered:
                            break
                    if not covered:
                        continue

                    mapping_sorted = {k: int(mapping[k]) for k in sorted(mapping.keys())}
                    return ProgramStepV141(
                        op_id="tile_signature_fill",
                        args={"tile_h": int(tile_h), "tile_w": int(tile_w), "bg": int(bgc), "mapping": mapping_sorted},
                    )
        return None

    def _infer_downsample_mode_step() -> Optional[ProgramStepV141]:
        # Detect a strict integer downsample-by-mode from train pairs.
        if not train_pairs:
            return None
        sy: Optional[int] = None
        sx: Optional[int] = None
        for inp, out in train_pairs:
            hi, wi = grid_shape_v124(inp)
            ho, wo = grid_shape_v124(out)
            if int(ho) <= 0 or int(wo) <= 0:
                return None
            if int(hi) % int(ho) != 0 or int(wi) % int(wo) != 0:
                return None
            sy0 = int(hi // ho)
            sx0 = int(wi // wo)
            if sy is None:
                sy, sx = int(sy0), int(sx0)
            elif (int(sy), int(sx)) != (int(sy0), int(sx0)):
                return None
        if sy is None or sx is None:
            return None
        if int(sy) <= 1 and int(sx) <= 1:
            return None
        if int(sy) > 12 or int(sx) > 12:
            return None

        for inp, out in train_pairs:
            st0 = StateV132(grid=inp, objset=None, obj=None, bbox=None, patch=None)
            try:
                st1 = apply_op_v141(state=st0, op_id="downsample_mode", args={"sy": int(sy), "sx": int(sx)})
            except Exception:
                return None
            if not grid_equal_v124(st1.grid, out):
                return None
        return ProgramStepV141(op_id="downsample_mode", args={"sy": int(sy), "sx": int(sx)})

    def _infer_fill_enclosed_region_steps() -> List[ProgramStepV141]:
        # Detect fill_enclosed_region(bg, fill) that matches all train pairs.
        if not train_pairs:
            return []
        out: List[ProgramStepV141] = []
        bg_candidates = _infer_bg_candidates_v141(grids=[p[0] for p in train_pairs] + [test_in])
        fill_candidates = _infer_palette_out_v141(train_pairs=train_pairs)
        for bg in bg_candidates:
            for fill in fill_candidates:
                if int(fill) == int(bg):
                    continue
                ok = True
                for inp, want in train_pairs:
                    st0 = StateV132(grid=inp, objset=None, obj=None, bbox=None, patch=None)
                    try:
                        st1 = apply_op_v141(state=st0, op_id="fill_enclosed_region", args={"bg": int(bg), "fill": int(fill)})
                    except Exception:
                        ok = False
                        break
                    if not grid_equal_v124(st1.grid, want):
                        ok = False
                        break
                if ok:
                    out.append(ProgramStepV141(op_id="fill_enclosed_region", args={"bg": int(bg), "fill": int(fill)}))
        return out

    def _infer_symmetry_fill_steps() -> List[ProgramStepV141]:
        # Detect symmetry completion fills (h/v/rot180) with an explicit bg.
        if not train_pairs:
            return []
        out: List[ProgramStepV141] = []
        bg_candidates = _infer_bg_candidates_v141(grids=[p[0] for p in train_pairs] + [test_in])
        for op_id in ("symmetry_fill_h", "symmetry_fill_v", "symmetry_fill_rot180"):
            for bg in bg_candidates:
                ok = True
                for inp, want in train_pairs:
                    st0 = StateV132(grid=inp, objset=None, obj=None, bbox=None, patch=None)
                    try:
                        st1 = apply_op_v141(state=st0, op_id=str(op_id), args={"bg": int(bg)})
                    except Exception:
                        ok = False
                        break
                    if not grid_equal_v124(st1.grid, want):
                        ok = False
                        break
                if ok:
                    out.append(ProgramStepV141(op_id=str(op_id), args={"bg": int(bg)}))
        return out

    def _infer_best_diagonal_fill_step() -> Optional[ProgramStepV141]:
        # Detect deterministic diagonal marking tasks like:
        # - pick the diagonal (main or anti, any offset) with the highest count of from_color
        # - replace from_color→to_color along that diagonal
        #
        # This is generic (grid→grid), deterministic, and learned only from train pairs.
        if not train_pairs:
            return None
        for inp, out in train_pairs:
            if tuple(int(x) for x in grid_shape_v124(inp)) != tuple(int(x) for x in grid_shape_v124(out)):
                return None

        from_color: Optional[int] = None
        to_color: Optional[int] = None
        for inp, out in train_pairs:
            h, w = grid_shape_v124(inp)
            diffs: List[Tuple[int, int]] = []
            for r in range(int(h)):
                for c in range(int(w)):
                    if int(inp[int(r)][int(c)]) != int(out[int(r)][int(c)]):
                        diffs.append((int(r), int(c)))
            if not diffs:
                return None
            from_set = {int(inp[int(r)][int(c)]) for r, c in diffs}
            to_set = {int(out[int(r)][int(c)]) for r, c in diffs}
            if len(from_set) != 1 or len(to_set) != 1:
                return None
            fc = int(next(iter(from_set)))
            tc = int(next(iter(to_set)))
            if from_color is None:
                from_color = int(fc)
                to_color = int(tc)
            elif int(fc) != int(from_color) or int(tc) != int(to_color):
                return None

        if from_color is None or to_color is None:
            return None

        for inp, want in train_pairs:
            st0 = StateV132(grid=inp, objset=None, obj=None, bbox=None, patch=None)
            try:
                st1 = apply_op_v141(
                    state=st0,
                    op_id="best_diagonal_fill",
                    args={"from_color": int(from_color), "to_color": int(to_color)},
                )
            except Exception:
                return None
            if not grid_equal_v124(st1.grid, want):
                return None

        return ProgramStepV141(op_id="best_diagonal_fill", args={"from_color": int(from_color), "to_color": int(to_color)})

    def _infer_nest_by_color_area_step() -> Optional[ProgramStepV141]:
        # Detect tasks where the output is a nested-square CSG derived from input colors ordered by
        # max connected-component area (4-neigh). Generic and deterministic.
        if not train_pairs:
            return None
        bg_candidates = _infer_bg_candidates_v141(grids=[p[0] for p in train_pairs] + [test_in])
        for bg in bg_candidates:
            ok = True
            for inp, want in train_pairs:
                st0 = StateV132(grid=inp, objset=None, obj=None, bbox=None, patch=None)
                try:
                    st1 = apply_op_v141(state=st0, op_id="nest_by_color_area", args={"bg": int(bg)})
                except Exception:
                    ok = False
                    break
                if not grid_equal_v124(st1.grid, want):
                    ok = False
                    break
            if ok:
                return ProgramStepV141(op_id="nest_by_color_area", args={"bg": int(bg)})
        return None

    def _infer_quadrant_center_tile_step() -> Optional[ProgramStepV141]:
        # Detect tasks where sparse non-bg markers define a bbox, and the output adds a small
        # quadrant summary tile around the bbox center. Generic and deterministic.
        if not train_pairs:
            return None
        for inp, out in train_pairs:
            if tuple(int(x) for x in grid_shape_v124(inp)) != tuple(int(x) for x in grid_shape_v124(out)):
                return None

        bg_candidates = _infer_bg_candidates_v141(grids=[p[0] for p in train_pairs] + [test_in])
        for bg in bg_candidates:
            center_color: Optional[int] = None
            ok = True
            for inp, want in train_pairs:
                h, w = grid_shape_v124(inp)
                # bbox of non-bg in input under this bg
                rmin = None
                rmax = None
                cmin = None
                cmax = None
                for r in range(int(h)):
                    for c in range(int(w)):
                        if int(inp[int(r)][int(c)]) == int(bg):
                            continue
                        rmin = int(r) if rmin is None else min(int(rmin), int(r))
                        rmax = int(r) if rmax is None else max(int(rmax), int(r))
                        cmin = int(c) if cmin is None else min(int(cmin), int(c))
                        cmax = int(c) if cmax is None else max(int(cmax), int(c))
                if rmin is None or rmax is None or cmin is None or cmax is None:
                    ok = False
                    break
                cr = int((int(rmin) + int(rmax)) // 2)
                cc = int((int(cmin) + int(cmax)) // 2)
                cc_val = int(want[int(cr)][int(cc)])
                if center_color is None:
                    center_color = int(cc_val)
                elif int(cc_val) != int(center_color):
                    ok = False
                    break

                st0 = StateV132(grid=inp, objset=None, obj=None, bbox=None, patch=None)
                try:
                    st1 = apply_op_v141(
                        state=st0,
                        op_id="quadrant_center_tile",
                        args={"bg": int(bg), "center_color": int(center_color)},
                    )
                except Exception:
                    ok = False
                    break
                if not grid_equal_v124(st1.grid, want):
                    ok = False
                    break

            if ok and center_color is not None:
                return ProgramStepV141(op_id="quadrant_center_tile", args={"bg": int(bg), "center_color": int(center_color)})

        return None

    def _infer_histogram_color_counts_step() -> Optional[ProgramStepV141]:
        # Detect tasks where the output is a vertical histogram of non-bg color counts:
        # - output grid has same shape as input
        # - each non-bg color forms a bottom-aligned bar in columns starting at x0 (default 0)
        # - bar height equals the count of that color in the input
        if not train_pairs:
            return None
        for inp, out in train_pairs:
            if tuple(int(x) for x in grid_shape_v124(inp)) != tuple(int(x) for x in grid_shape_v124(out)):
                return None

        bg_candidates = _infer_bg_candidates_v141(grids=[p[0] for p in train_pairs] + [test_in])
        for bg in bg_candidates:
            for x0 in (0,):
                ok = True
                for inp, want in train_pairs:
                    st0 = StateV132(grid=inp, objset=None, obj=None, bbox=None, patch=None)
                    try:
                        st1 = apply_op_v141(state=st0, op_id="histogram_color_counts", args={"bg": int(bg), "x0": int(x0)})
                    except Exception:
                        ok = False
                        break
                    if not grid_equal_v124(st1.grid, want):
                        ok = False
                        break
                if ok:
                    return ProgramStepV141(op_id="histogram_color_counts", args={"bg": int(bg), "x0": int(x0)})
        return None

    def _infer_color_counts_run_column_step() -> Optional[ProgramStepV141]:
        """
        Detect tasks where the output is a single-column run-length list of non-bg colors,
        where each color is repeated by its pixel count in the input and ordered by first
        appearance in row-major order (background inferred as the mode color).

        This is deterministic, grid→grid, and requires exact agreement on all TRAIN pairs.
        """
        if not train_pairs:
            return None
        for _inp, out in train_pairs:
            h, w = grid_shape_v124(out)
            if int(h) <= 0 or int(w) != 1:
                return None
        ok = True
        for inp, want in train_pairs:
            st0 = StateV132(grid=inp, objset=None, obj=None, bbox=None, patch=None)
            try:
                st1 = apply_op_v141(state=st0, op_id="color_counts_run_column", args={})
            except Exception:
                ok = False
                break
            if not grid_equal_v124(st1.grid, want):
                ok = False
                break
        if not ok:
            return None
        return ProgramStepV141(op_id="color_counts_run_column", args={})

    def _infer_paint_points_step() -> Optional[ProgramStepV141]:
        """
        Detect tasks where the transformation is a fixed set of point edits (r,c,color) that is
        consistent across all TRAIN pairs, and therefore can be applied to TEST deterministically.

        This is intentionally conservative: it requires identical point sets and identical output
        colors for each point across all demos. It is generic (grid→grid), deterministic, and
        provides a "first semantic move" for many near-miss tasks where every broad mask/paint
        action worsens loss.
        """
        if not train_pairs:
            return None
        # paint_points does not change shape.
        for inp, out in train_pairs:
            if tuple(int(x) for x in grid_shape_v124(inp)) != tuple(int(x) for x in grid_shape_v124(out)):
                return None

        pts0: Optional[Dict[Tuple[int, int], int]] = None
        for inp, out in train_pairs:
            h, w = grid_shape_v124(inp)
            pts: Dict[Tuple[int, int], int] = {}
            for r in range(int(h)):
                for c in range(int(w)):
                    vi = int(inp[int(r)][int(c)])
                    vo = int(out[int(r)][int(c)])
                    if vi == vo:
                        continue
                    pts[(int(r), int(c))] = int(vo)
            if pts0 is None:
                pts0 = dict(pts)
            else:
                if set(pts.keys()) != set(pts0.keys()):
                    return None
                for k in pts0.keys():
                    if int(pts[k]) != int(pts0[k]):
                        return None

        if pts0 is None or not pts0:
            return None
        # Bound to keep the op and its MDL cost reasonable.
        if int(len(pts0)) > 64:
            return None

        points = [{"r": int(r), "c": int(c), "color": int(pts0[(int(r), int(c))])} for (r, c) in sorted(pts0.keys())]

        # Verify train exactness (defensive).
        for inp, want in train_pairs:
            st0 = StateV132(grid=inp, objset=None, obj=None, bbox=None, patch=None)
            try:
                st1 = apply_op_v141(state=st0, op_id="paint_points", args={"points": list(points), "mode": "overwrite"})
            except Exception:
                return None
            if not grid_equal_v124(st1.grid, want):
                return None

        return ProgramStepV141(op_id="paint_points", args={"points": list(points), "mode": "overwrite"})

    def _infer_spatial_mitm_csg_calls() -> List[ProgramStepV141]:
        """
        Bidirectional (meet-in-the-middle) direct inference for pure spatial transforms.

        This targets a common failure mode under tight depth caps: a solution exists as a short
        composition of invertible grid transforms (e.g., crop-free rotate/reflect/transpose and
        bbox-aligned translate), but the generic search cannot reach it.

        The result is wrapped as a concept_call so it stays usable under abstraction pressure
        (require_concept_call) without requiring a learned concept bank.
        """
        if not train_pairs:
            return []

        train_in = [p[0] for p in train_pairs]
        train_out = [p[1] for p in train_pairs]

        # Keep this bounded and deterministic.
        for g in train_in + train_out + [test_in]:
            h, w = grid_shape_v124(g)
            if int(h) > 30 or int(w) > 30:
                return []

        bg_candidates = _infer_bg_candidates_v141(grids=train_in + [test_in])

        base_ops: List[ProgramStepV141] = [
            ProgramStepV141(op_id="rotate90", args={}),
            ProgramStepV141(op_id="rotate180", args={}),
            ProgramStepV141(op_id="rotate270", args={}),
            ProgramStepV141(op_id="reflect_h", args={}),
            ProgramStepV141(op_id="reflect_v", args={}),
            ProgramStepV141(op_id="transpose", args={}),
        ]

        # Only include translations that are unambiguously implied by train demonstrations.
        trans_ops: List[ProgramStepV141] = []
        for bg in list(bg_candidates[:2] if bg_candidates else [0]):
            shift: Optional[Tuple[int, int]] = None
            consistent = True
            for inp, outg in train_pairs:
                if tuple(int(x) for x in grid_shape_v124(inp)) != tuple(int(x) for x in grid_shape_v124(outg)):
                    consistent = False
                    break
                ir0, ic0, _, _ = bbox_nonzero_v124(inp, bg=int(bg))
                or0, oc0, _, _ = bbox_nonzero_v124(outg, bg=int(bg))
                dy = int(or0 - ir0)
                dx = int(oc0 - ic0)
                if shift is None:
                    shift = (int(dy), int(dx))
                elif shift != (int(dy), int(dx)):
                    consistent = False
                    break
            if not consistent or shift is None:
                continue
            dy, dx = shift
            if int(dy) == 0 and int(dx) == 0:
                continue
            trans_ops.append(ProgramStepV141(op_id="translate", args={"dx": int(dx), "dy": int(dy), "pad": int(bg)}))

        ops: List[ProgramStepV141] = list(base_ops) + list(trans_ops)
        ops.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
        if not ops:
            return []

        def _apply_seq(g: GridV124, seq: Sequence[ProgramStepV141]) -> Optional[GridV124]:
            st = StateV132(grid=g, objset=None, obj=None, bbox=None, patch=None)
            try:
                for s in seq:
                    st = apply_op_v141(state=st, op_id=str(s.op_id), args=dict(s.args))
                return st.grid
            except Exception:
                return None

        def _seq_cost_bits(seq: Sequence[ProgramStepV141]) -> int:
            c = 0
            for s in seq:
                c += int(step_cost_bits_v141(op_id=str(s.op_id), args=dict(s.args)))
            return int(c)

        def _seq_js(seq: Sequence[ProgramStepV141]) -> str:
            return canonical_json_dumps([s.to_dict() for s in seq])

        def _invert_step(s: ProgramStepV141) -> Optional[ProgramStepV141]:
            op0 = str(s.op_id or "")
            if op0 == "rotate90":
                return ProgramStepV141(op_id="rotate270", args={})
            if op0 == "rotate270":
                return ProgramStepV141(op_id="rotate90", args={})
            if op0 == "rotate180":
                return ProgramStepV141(op_id="rotate180", args={})
            if op0 == "reflect_h":
                return ProgramStepV141(op_id="reflect_h", args={})
            if op0 == "reflect_v":
                return ProgramStepV141(op_id="reflect_v", args={})
            if op0 == "transpose":
                return ProgramStepV141(op_id="transpose", args={})
            if op0 == "translate":
                dx = int(s.args.get("dx", 0))
                dy = int(s.args.get("dy", 0))
                pad = int(s.args.get("pad", 0))
                return ProgramStepV141(op_id="translate", args={"dx": int(-dx), "dy": int(-dy), "pad": int(pad)})
            return None

        def _invert_seq(seq: Sequence[ProgramStepV141]) -> Optional[Tuple[ProgramStepV141, ...]]:
            inv: List[ProgramStepV141] = []
            for stp in reversed(list(seq)):
                x = _invert_step(stp)
                if x is None:
                    return None
                inv.append(x)
            return tuple(inv)

        # Enumerate all sequences up to max_half (including empty).
        max_total = 6
        max_half = int(max_total // 2)
        seqs: List[Tuple[ProgramStepV141, ...]] = [tuple()]
        cur: List[Tuple[ProgramStepV141, ...]] = [tuple()]
        for _ in range(int(max_half)):
            nxt: List[Tuple[ProgramStepV141, ...]] = []
            for seq in cur:
                for op in ops:
                    nxt.append(tuple(list(seq) + [op]))
            seqs.extend(nxt)
            cur = nxt

        # Map: intermediate signature -> best (cost, js, seq).
        f_best: Dict[Tuple[str, ...], Tuple[int, str, Tuple[ProgramStepV141, ...]]] = {}
        for seq in seqs:
            outs: List[str] = []
            ok = True
            for g in train_in:
                got = _apply_seq(g, seq)
                if got is None:
                    ok = False
                    break
                outs.append(str(grid_hash_v124(got)))
            if not ok:
                continue
            key = tuple(outs)
            cost = int(_seq_cost_bits(seq))
            js = str(_seq_js(seq))
            prev = f_best.get(key)
            if prev is None or (int(cost), str(js)) < (int(prev[0]), str(prev[1])):
                f_best[key] = (int(cost), str(js), tuple(seq))

        b_best: Dict[Tuple[str, ...], Tuple[int, str, Tuple[ProgramStepV141, ...]]] = {}
        for seq in seqs:
            inv = _invert_seq(seq)
            if inv is None:
                continue
            outs: List[str] = []
            ok = True
            for g in train_out:
                got = _apply_seq(g, inv)
                if got is None:
                    ok = False
                    break
                outs.append(str(grid_hash_v124(got)))
            if not ok:
                continue
            key = tuple(outs)
            cost = int(_seq_cost_bits(seq))
            js = str(_seq_js(seq))
            prev = b_best.get(key)
            if prev is None or (int(cost), str(js)) < (int(prev[0]), str(prev[1])):
                b_best[key] = (int(cost), str(js), tuple(seq))

        # Collect exact spatial programs from matches.
        candidates: List[Tuple[int, str, Tuple[ProgramStepV141, ...]]] = []
        for key, (c1, _js1, seq1) in f_best.items():
            hit = b_best.get(key)
            if hit is None:
                continue
            c2, _js2, seq2 = hit
            full = tuple(list(seq1) + list(seq2))
            if not full:
                continue
            ok = True
            for inp, want in train_pairs:
                got = _apply_seq(inp, full)
                if got is None or not grid_equal_v124(got, want):
                    ok = False
                    break
            if not ok:
                continue
            cost = int(c1) + int(c2)
            candidates.append((int(cost), str(_seq_js(full)), full))

        # Also consider spatial+map_colors: enumerate small spatial sequences and infer a consistent mapping.
        map_seq_cands: List[Tuple[int, str, Tuple[ProgramStepV141, ...]]] = []

        def _infer_mapping(got: GridV124, want: GridV124, mapping: Dict[int, int]) -> bool:
            if tuple(int(x) for x in grid_shape_v124(got)) != tuple(int(x) for x in grid_shape_v124(want)):
                return False
            h, w = grid_shape_v124(got)
            for r in range(int(h)):
                for c in range(int(w)):
                    vi = int(got[int(r)][int(c)])
                    vo = int(want[int(r)][int(c)])
                    prev = mapping.get(int(vi))
                    if prev is not None and int(prev) != int(vo):
                        return False
                    mapping[int(vi)] = int(vo)
            return True

        small_seqs = [s for s in seqs if int(len(s)) <= 3]
        for seq in small_seqs:
            mapping: Dict[int, int] = {}
            ok = True
            for inp, want in train_pairs:
                got = _apply_seq(inp, seq)
                if got is None or not _infer_mapping(got, want, mapping):
                    ok = False
                    break
            if not ok:
                continue
            mapping2 = {str(k): int(mapping[k]) for k in sorted(mapping.keys())}
            mstep = ProgramStepV141(op_id="map_colors", args={"mapping": dict(mapping2)})
            full = tuple(list(seq) + [mstep])
            ok2 = True
            for inp, want in train_pairs:
                got2 = _apply_seq(inp, full)
                if got2 is None or not grid_equal_v124(got2, want):
                    ok2 = False
                    break
            if not ok2:
                continue
            cost = int(_seq_cost_bits(seq)) + int(step_cost_bits_v141(op_id="map_colors", args={"mapping": dict(mapping2)}))
            map_seq_cands.append((int(cost), str(_seq_js(full)), full))

        candidates.extend(map_seq_cands)
        candidates.sort(key=lambda t: (int(t[0]), str(t[1])))

        out_steps: List[ProgramStepV141] = []
        for _cost, js, seq in candidates[:6]:
            if len(seq) <= 1:
                continue
            tag = "direct_mitm_v141:" + sha256_hex(str(js).encode("utf-8"))[:16]
            out_steps.append(_make_csg_call_v141(tag=str(tag), inner_steps=list(seq)))
        return out_steps

    steps_v140 = _infer_direct_steps_v140(train_pairs=train_pairs, test_in=test_in)
    out: List[ProgramStepV141] = []
    for s in steps_v140:
        out.append(_wrap_single_step_as_concept_call(ProgramStepV141(op_id=str(s.op_id), args=dict(s.args)), kind="direct_v140_prim_csg_v141"))
    # NOTE: Avoid expensive per-task direct synthesis (patch-paint and multi-step program induction)
    # in the solver hot path. Those mechanisms are useful in isolation but can dominate wall time
    # under strict per-task timeouts, preventing the main compositional search from exploring
    # anything (programs_explored≈0) and yielding empty timeout traces.
    #
    # Under the AGI regime, multi-step compression should come from the learned concept bank
    # (CSG/CSV) and from trace-driven induction in LEARN mode, not from heavy per-task enumeration.
    extra = _infer_tile_signature_fill_step()
    if extra is not None:
        out.append(_wrap_single_step_as_concept_call(extra, kind="direct_tile_signature_fill_csg_v141"))
    extra2 = _infer_downsample_mode_step()
    if extra2 is not None:
        out.append(_wrap_single_step_as_concept_call(extra2, kind="direct_downsample_mode_csg_v141"))
    out.extend(_wrap_single_step_as_concept_call(st, kind="direct_fill_enclosed_region_csg_v141") for st in _infer_fill_enclosed_region_steps())
    out.extend(_wrap_single_step_as_concept_call(st, kind="direct_symmetry_fill_csg_v141") for st in _infer_symmetry_fill_steps())
    extra3 = _infer_best_diagonal_fill_step()
    if extra3 is not None:
        out.append(_wrap_single_step_as_concept_call(extra3, kind="direct_best_diagonal_fill_csg_v141"))
    extra4 = _infer_nest_by_color_area_step()
    if extra4 is not None:
        out.append(_wrap_single_step_as_concept_call(extra4, kind="direct_nest_by_color_area_csg_v141"))
    extra5 = _infer_quadrant_center_tile_step()
    if extra5 is not None:
        out.append(_wrap_single_step_as_concept_call(extra5, kind="direct_quadrant_center_tile_csg_v141"))
    extra6 = _infer_histogram_color_counts_step()
    if extra6 is not None:
        out.append(_wrap_single_step_as_concept_call(extra6, kind="direct_histogram_color_counts_csg_v141"))
    extra6b = _infer_color_counts_run_column_step()
    if extra6b is not None:
        out.append(_wrap_single_step_as_concept_call(extra6b, kind="direct_color_counts_run_column_csg_v141"))
    extra7 = _infer_paint_points_step()
    if extra7 is not None:
        out.append(_wrap_single_step_as_concept_call(extra7, kind="direct_paint_points_csg_v141"))
    # Spatial MITM synthesis is intentionally disabled here for the same reason as above; it is
    # a structured search and belongs in learn-mode mining, not in per-task direct inference.
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


def _make_csg_call_v141(*, tag: str, inner_steps: Sequence[ProgramStepV141]) -> ProgramStepV141:
    # Deterministic, content-addressed concept wrapper for a small primitive subgraph.
    # This is used to propose generic multi-step closures under abstraction pressure.
    inner_dicts = [s.to_dict() for s in inner_steps]
    body = {"kind": "csg_call_v141", "tag": str(tag), "steps": list(inner_dicts)}
    cid = "csg_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))[:16]
    cost = 0
    op_ids: List[str] = []
    for st in inner_steps:
        op_ids.append(str(st.op_id))
        cost += int(step_cost_bits_v141(op_id=str(st.op_id), args=dict(st.args)))
    return ProgramStepV141(
        op_id="concept_call",
        args={
            "concept_id": str(cid),
            "cost_bits": max(1, int(cost) - 1),
            "op_ids": [str(x) for x in op_ids if str(x)],
            "steps": list(inner_dicts),
        },
    )


def _propose_builtin_csg_calls_v141(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    bg_candidates: Sequence[int],
) -> List[ProgramStepV141]:
    """
    Generic multi-step closures proposed as single concept_call steps.

    Goal: reduce depth/branching for common object/bbox pipelines without task-specific hacks.
    """
    train_in = [p[0] for p in train_pairs]
    out: List[ProgramStepV141] = []

    # bbox_by_color -> crop -> commit
    for st in _propose_bbox_by_color_steps_v141(train_pairs=train_pairs)[:10]:
        color = int(st.args.get("color", 0))
        inner = [
            ProgramStepV141(op_id="bbox_by_color", args={"color": int(color)}),
            ProgramStepV141(op_id="crop_bbox", args={}),
            ProgramStepV141(op_id="commit_patch", args={}),
        ]
        out.append(_make_csg_call_v141(tag=f"bbox_color_crop:{color}", inner_steps=inner))

    # cc4 -> select_obj -> bbox -> crop -> commit
    try:
        from .arc_solver_v134 import _infer_select_obj_args_v134

        # Prefer inferred selectors but keep a deterministic fallback.
        args_list = _infer_select_obj_args_v134(
            train_pairs=train_pairs,
            bg=int(bg_candidates[0] if bg_candidates else 0),
            max_rank=1,
        )
        if not args_list:
            args_list = [
                {"key": "area", "order": "max", "rank": 0, "color_filter": None},
                {"key": "area", "order": "min", "rank": 0, "color_filter": None},
                {"key": "bbox_area", "order": "max", "rank": 0, "color_filter": None},
                {"key": "bbox_area", "order": "min", "rank": 0, "color_filter": None},
                {"key": "width", "order": "max", "rank": 0, "color_filter": None},
                {"key": "height", "order": "max", "rank": 0, "color_filter": None},
            ]
        # Canonicalize and bound.
        args_list = [dict(a) for a in args_list if isinstance(a, dict)]
        args_list.sort(key=lambda a: canonical_json_dumps(a))
        args_list = args_list[:6]

        for bg in list(bg_candidates[:2] if bg_candidates else [0]):
            if not _cc4_nonempty_for_all_v141(grids=train_in + [test_in], bg=int(bg)):
                continue
            for sa in args_list:
                inner = [
                    ProgramStepV141(op_id="cc4", args={"bg": int(bg)}),
                    ProgramStepV141(op_id="select_obj", args=dict(sa)),
                    ProgramStepV141(op_id="obj_bbox", args={}),
                    ProgramStepV141(op_id="crop_bbox", args={}),
                    ProgramStepV141(op_id="commit_patch", args={}),
                ]
                tag = f"cc4_select_crop:bg={int(bg)}:{canonical_json_dumps(sa)}"
                out.append(_make_csg_call_v141(tag=tag, inner_steps=inner))

        # cc4 -> select_obj -> obj_patch -> patch_transform -> commit
        patch_ops = [
            "patch_rotate90",
            "patch_rotate180",
            "patch_rotate270",
            "patch_reflect_h",
            "patch_reflect_v",
            "patch_transpose",
        ]
        for bg in list(bg_candidates[:2] if bg_candidates else [0]):
            if not _cc4_nonempty_for_all_v141(grids=train_in + [test_in], bg=int(bg)):
                continue
            for sa in args_list:
                for pop in patch_ops:
                    inner = [
                        ProgramStepV141(op_id="cc4", args={"bg": int(bg)}),
                        ProgramStepV141(op_id="select_obj", args=dict(sa)),
                        ProgramStepV141(op_id="obj_patch", args={}),
                        ProgramStepV141(op_id=str(pop), args={}),
                        ProgramStepV141(op_id="commit_patch", args={}),
                    ]
                    tag = f"cc4_select_obj_patch:{str(pop)}:bg={int(bg)}:{canonical_json_dumps(sa)}"
                    out.append(_make_csg_call_v141(tag=tag, inner_steps=inner))

        # cc4_nonbg_multicolor -> select_obj -> bbox -> crop -> commit
        for bg in list(bg_candidates[:2] if bg_candidates else [0]):
            if not _cc4_multicolor_nonempty_for_all_v141(grids=train_in + [test_in], bg=int(bg)):
                continue
            for sa in args_list:
                inner = [
                    ProgramStepV141(op_id="cc4_nonbg_multicolor", args={"bg": int(bg)}),
                    ProgramStepV141(op_id="select_obj", args=dict(sa)),
                    ProgramStepV141(op_id="obj_bbox", args={}),
                    ProgramStepV141(op_id="crop_bbox", args={}),
                    ProgramStepV141(op_id="commit_patch", args={}),
                ]
                tag = f"cc4m_select_crop:bg={int(bg)}:{canonical_json_dumps(sa)}"
                out.append(_make_csg_call_v141(tag=tag, inner_steps=inner))

        # cc4_nonbg_multicolor -> select_obj -> obj_patch -> patch_transform -> commit
        for bg in list(bg_candidates[:2] if bg_candidates else [0]):
            if not _cc4_multicolor_nonempty_for_all_v141(grids=train_in + [test_in], bg=int(bg)):
                continue
            for sa in args_list:
                for pop in patch_ops:
                    inner = [
                        ProgramStepV141(op_id="cc4_nonbg_multicolor", args={"bg": int(bg)}),
                        ProgramStepV141(op_id="select_obj", args=dict(sa)),
                        ProgramStepV141(op_id="obj_patch", args={}),
                        ProgramStepV141(op_id=str(pop), args={}),
                        ProgramStepV141(op_id="commit_patch", args={}),
                    ]
                    tag = f"cc4m_select_obj_patch:{str(pop)}:bg={int(bg)}:{canonical_json_dumps(sa)}"
                    out.append(_make_csg_call_v141(tag=tag, inner_steps=inner))
    except Exception:
        pass

    # cc4 -> select_obj -> obj_patch -> paste (object relocation / duplication)
    #
    # Generic closure: many ARC-AGI tasks require extracting an object (via CC4 + select_obj),
    # turning it into a patch (obj_patch), and placing it elsewhere (paste). This pattern is
    # *not* ARC-specific: it is a basic gridworld transformation. We propose a small, bounded
    # set of placements derived deterministically from TRAIN diffs/bboxes.
    try:
        from .grid_v124 import bbox_nonzero_v124

        def _derived_paste_positions(*, bg: int) -> List[Tuple[int, int]]:
            # Do not include a fixed (0,0) fallback unless we truly have no evidence-derived
            # placements. Including (0,0) eagerly can create train-improving but wrong-test
            # hypotheses and wastes budget under strict, fail-closed evaluation.
            pos: Set[Tuple[int, int]] = set()
            for inp, outg in train_pairs:
                try:
                    r0, c0, _r1, _c1 = bbox_nonzero_v124(outg, bg=int(bg))
                    pos.add((int(r0), int(c0)))
                except Exception:
                    pass
                try:
                    hi, wi = grid_shape_v124(inp)
                    ho, wo = grid_shape_v124(outg)
                    if (int(hi), int(wi)) != (int(ho), int(wo)) or int(hi) <= 0 or int(wi) <= 0:
                        continue
                    rmin = None
                    cmin = None
                    for r in range(int(hi)):
                        for c in range(int(wi)):
                            if int(inp[int(r)][int(c)]) != int(outg[int(r)][int(c)]):
                                rr = int(r)
                                cc = int(c)
                                rmin = rr if rmin is None else min(int(rmin), rr)
                                cmin = cc if cmin is None else min(int(cmin), cc)
                    if rmin is not None and cmin is not None:
                        pos.add((int(rmin), int(cmin)))
                except Exception:
                    pass
            if not pos:
                pos.add((0, 0))
            return sorted(pos)

        from .arc_solver_v134 import _infer_select_obj_args_v134

        args_list = _infer_select_obj_args_v134(
            train_pairs=train_pairs,
            bg=int(bg_candidates[0] if bg_candidates else 0),
            max_rank=1,
        )
        if not args_list:
            args_list = [
                {"key": "area", "order": "max", "rank": 0, "color_filter": None},
                {"key": "bbox_area", "order": "max", "rank": 0, "color_filter": None},
                {"key": "height", "order": "max", "rank": 0, "color_filter": None},
                {"key": "width", "order": "max", "rank": 0, "color_filter": None},
            ]
        args_list = [dict(a) for a in args_list if isinstance(a, dict)]
        args_list.sort(key=lambda a: canonical_json_dumps(a))
        args_list = args_list[:4]

        for bg in list(bg_candidates[:2] if bg_candidates else [0]):
            if not _cc4_nonempty_for_all_v141(grids=train_in + [test_in], bg=int(bg)):
                continue
            pos_list = _derived_paste_positions(bg=int(bg))[:4]
            for sa in args_list:
                for top, left in pos_list:
                    inner = [
                        ProgramStepV141(op_id="cc4", args={"bg": int(bg)}),
                        ProgramStepV141(op_id="select_obj", args=dict(sa)),
                        ProgramStepV141(op_id="obj_patch", args={"bg": int(bg)}),
                        ProgramStepV141(op_id="paste", args={"top": int(top), "left": int(left), "transparent": int(bg)}),
                    ]
                    tag = f"cc4_select_obj_patch_paste:bg={int(bg)}:top={int(top)}:left={int(left)}:{canonical_json_dumps(sa)}"
                    out.append(_make_csg_call_v141(tag=tag, inner_steps=inner))
    except Exception:
        pass

    # cc4 -> select_obj -> obj_patch -> patch_transform -> paste (object duplication + transform)
    #
    # Generic closure: duplicate an object by extracting it as a patch, applying a small
    # patch-space transform, and pasting it elsewhere. This targets a common ARC-AGI pattern
    # and is still fully gated by the CSV applicability test downstream, so it cannot dominate
    # unless it causally improves TRAIN loss.
    try:
        patch_ops = [
            "patch_rotate90",
            "patch_rotate180",
            "patch_rotate270",
            "patch_reflect_h",
            "patch_reflect_v",
            "patch_transpose",
        ]

        # Keep candidate count bounded; the CSV gate will further prune.
        args_list_xform = list(args_list)[:2]

        for bg in list(bg_candidates[:2] if bg_candidates else [0]):
            pos_list = _derived_paste_positions(bg=int(bg))[:3]

            if _cc4_nonempty_for_all_v141(grids=train_in + [test_in], bg=int(bg)):
                for sa in args_list_xform:
                    for pop in patch_ops:
                        for top, left in pos_list:
                            inner = [
                                ProgramStepV141(op_id="cc4", args={"bg": int(bg)}),
                                ProgramStepV141(op_id="select_obj", args=dict(sa)),
                                ProgramStepV141(op_id="obj_patch", args={"bg": int(bg)}),
                                ProgramStepV141(op_id=str(pop), args={}),
                                ProgramStepV141(op_id="paste", args={"top": int(top), "left": int(left), "transparent": int(bg)}),
                            ]
                            tag = (
                                f"cc4_select_obj_patch_{str(pop)}_paste:"
                                f"bg={int(bg)}:top={int(top)}:left={int(left)}:{canonical_json_dumps(sa)}"
                            )
                            out.append(_make_csg_call_v141(tag=tag, inner_steps=inner))

            if _cc4_multicolor_nonempty_for_all_v141(grids=train_in + [test_in], bg=int(bg)):
                for sa in args_list_xform:
                    for pop in patch_ops:
                        for top, left in pos_list:
                            inner = [
                                ProgramStepV141(op_id="cc4_nonbg_multicolor", args={"bg": int(bg)}),
                                ProgramStepV141(op_id="select_obj", args=dict(sa)),
                                ProgramStepV141(op_id="obj_patch", args={"bg": int(bg)}),
                                ProgramStepV141(op_id=str(pop), args={}),
                                ProgramStepV141(op_id="paste", args={"top": int(top), "left": int(left), "transparent": int(bg)}),
                            ]
                            tag = (
                                f"cc4m_select_obj_patch_{str(pop)}_paste:"
                                f"bg={int(bg)}:top={int(top)}:left={int(left)}:{canonical_json_dumps(sa)}"
                            )
                            out.append(_make_csg_call_v141(tag=tag, inner_steps=inner))
    except Exception:
        pass

    # mask_by_color -> (mask_dilate | mask_box_dilate) -> paint_mask
    # Generic closure: common ARC pattern of expanding a seed color region and repainting it.
    # Candidates are derived deterministically from train-pair color diffs (not task-specific IDs).
    try:
        diff_counts: Dict[Tuple[int, int], int] = {}
        for inp, outg in train_pairs:
            hi, wi = grid_shape_v124(inp)
            ho, wo = grid_shape_v124(outg)
            if (hi, wi) != (ho, wo) or hi <= 0 or wi <= 0:
                continue
            for r in range(int(hi)):
                for c in range(int(wi)):
                    vi = int(inp[int(r)][int(c)])
                    vo = int(outg[int(r)][int(c)])
                    if vi == vo:
                        continue
                    k = (int(vi), int(vo))
                    diff_counts[k] = int(diff_counts.get(k, 0)) + 1
        pairs = sorted(diff_counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0][0]), int(kv[0][1])))
        # Bound to keep branching stable.
        for (mc, pc), _n in pairs[:8]:
            for mode in ("mask_dilate", "mask_box_dilate"):
                inner = [
                    ProgramStepV141(op_id="mask_by_color", args={"color": int(mc)}),
                    ProgramStepV141(op_id=str(mode), args={}),
                    ProgramStepV141(op_id="paint_mask", args={"color": int(pc)}),
                ]
                tag = f"dilate_paint:mc={int(mc)}:pc={int(pc)}:mode={str(mode)}"
                out.append(_make_csg_call_v141(tag=tag, inner_steps=inner))

            # mask_by_color(seed) -> outline(seed) -> paint_mask(paint)
            #
            # Generic closure: tasks that require painting only the border/outline of a region
            # (common in ARC-AGI-2). This is still fully parameterized by demo diffs and gated
            # downstream by the CSV applicability test, so it cannot dominate search unless it
            # causally improves TRAIN loss.
            inner_outline = [
                ProgramStepV141(op_id="mask_by_color", args={"color": int(mc)}),
                ProgramStepV141(op_id="mask_outline", args={}),
                ProgramStepV141(op_id="paint_mask", args={"color": int(pc)}),
            ]
            tag_outline = f"outline_paint:mc={int(mc)}:pc={int(pc)}"
            out.append(_make_csg_call_v141(tag=tag_outline, inner_steps=inner_outline))
    except Exception:
        pass

    # mask_by_color(seed) -> dilate -> mask_and_color(bg) -> paint_mask(paint)
    # Generic closure: paint background neighbors of a seed color. Useful for outlines/shadows.
    try:
        neigh_counts: Dict[Tuple[int, int], int] = {}
        for inp, outg in train_pairs:
            hi, wi = grid_shape_v124(inp)
            ho, wo = grid_shape_v124(outg)
            if (hi, wi) != (ho, wo) or hi <= 0 or wi <= 0:
                continue
            for r in range(int(hi)):
                for c in range(int(wi)):
                    vi = int(inp[int(r)][int(c)])
                    vo = int(outg[int(r)][int(c)])
                    if vi == vo:
                        continue
                    # Count neighbor evidence: changed cells tend to be adjacent to a seed color.
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        rr = int(r + dr)
                        cc = int(c + dc)
                        if rr < 0 or cc < 0 or rr >= int(hi) or cc >= int(wi):
                            continue
                        seed = int(inp[int(rr)][int(cc)])
                        k = (int(seed), int(vo))
                        neigh_counts[k] = int(neigh_counts.get(k, 0)) + 1
        pairs = sorted(neigh_counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0][0]), int(kv[0][1])))
        bg0 = int(bg_candidates[0]) if bg_candidates else 0
        for (seed, paint), _n in pairs[:8]:
            for mode in ("mask_dilate", "mask_box_dilate"):
                inner = [
                    ProgramStepV141(op_id="mask_by_color", args={"color": int(seed)}),
                    ProgramStepV141(op_id=str(mode), args={}),
                    ProgramStepV141(op_id="mask_and_color", args={"color": int(bg0)}),
                    ProgramStepV141(op_id="paint_mask", args={"color": int(paint)}),
                ]
                tag = f"neigh_bg_paint:seed={int(seed)}:paint={int(paint)}:bg={int(bg0)}:mode={str(mode)}"
                out.append(_make_csg_call_v141(tag=tag, inner_steps=inner))
    except Exception:
        pass

    # De-dup by concept_id and canonicalize ordering (cost, id).
    uniq: Dict[str, ProgramStepV141] = {}
    for s in out:
        cid = str(s.args.get("concept_id") or "")
        if not cid:
            continue
        if cid in uniq:
            continue
        uniq[cid] = s
    rows = list(uniq.values())
    rows.sort(key=lambda s: (int(s.args.get("cost_bits") or 0), str(s.args.get("concept_id") or "")))
    return rows


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
    # pad_to_v124 changes the grid iff (h,w)!=(height,width).
    # A previous version treated "input smaller than target" as a noop, which incorrectly
    # suppressed pad_to proposals for the common ARC pattern of embedding a smaller grid
    # into a larger canvas.
    for h, w in train_in_shapes:
        if (int(h), int(w)) != (int(height), int(width)):
            return False
    return True


def _infer_crop_cc4_select_args_v141(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    bg: int,
    max_rank: int = 1,
    max_specs: int = 8,
) -> List[Dict[str, Any]]:
    """
    Infer selector args for crop_cc4_select on *shape-changing* tasks.

    When output shape differs from input, overlap-based inference (changed mask) often fails.
    This routine scores candidate selectors by applying crop_cc4_select on train inputs and
    measuring train loss against train outputs.
    """
    if not train_pairs:
        return []
    # Only meaningful when output shape differs from input (crop/commit patterns).
    for inp, out in train_pairs:
        if tuple(int(x) for x in grid_shape_v124(inp)) != tuple(int(x) for x in grid_shape_v124(out)):
            break
    else:
        return []

    bgi = int(bg)
    keys = [
        "area",
        "width",
        "height",
        "bbox_area",
        "top",
        "left",
        "bottom",
        "right",
        "color",
        "dist_center",
    ]
    orders = ["min", "max"]

    def _loss(got: GridV124, want: GridV124) -> Tuple[int, int]:
        if tuple(int(x) for x in grid_shape_v124(got)) != tuple(int(x) for x in grid_shape_v124(want)):
            return (1, 10**9)
        h, w = grid_shape_v124(want)
        diff = 0
        for r in range(int(h)):
            for c in range(int(w)):
                if int(got[int(r)][int(c)]) != int(want[int(r)][int(c)]):
                    diff += 1
        return (0, int(diff))

    scored: List[Tuple[Tuple[int, int], str, Dict[str, Any]]] = []
    for key in keys:
        for order in orders:
            for rank in range(0, int(max_rank) + 1):
                spec: Dict[str, Any] = {
                    "bg": int(bgi),
                    "key": str(key),
                    "order": str(order),
                    "rank": int(rank),
                    "color_filter": None,
                }
                loss_shape = 0
                loss_cells = 0
                ok = True
                for inp, want in train_pairs:
                    st0 = StateV132(grid=inp, objset=None, obj=None, bbox=None, patch=None)
                    try:
                        st1 = apply_op_v141(state=st0, op_id="crop_cc4_select", args=dict(spec))
                    except Exception:
                        ok = False
                        break
                    ls, lc = _loss(st1.grid, want)
                    loss_shape += int(ls)
                    loss_cells += int(lc) if int(ls) == 0 else 0
                if not ok:
                    continue
                sig = canonical_json_dumps(spec)
                scored.append(((int(loss_shape), int(loss_cells)), str(sig), spec))

    if not scored:
        return []
    scored.sort(key=lambda x: (int(x[0][0]), int(x[0][1]), str(x[1])))
    best_loss = scored[0][0]
    keep = [spec for (loss, _sig, spec) in scored if loss == best_loss] if best_loss == (0, 0) else [spec for (_loss, _sig, spec) in scored[: int(max_specs)]]
    keep.sort(key=lambda d: canonical_json_dumps(d))
    return keep[: int(max_specs)]


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
    train_out = [p[1] for p in train_pairs]
    train_in_shapes = [grid_shape_v124(g) for g in train_in]
    train_out_shapes = [grid_shape_v124(g) for g in train_out]
    out_steps: List[ProgramStepV141] = []

    for s in direct_steps:
        out_steps.append(s)

    if not bool(avail.get("bbox", False)):
        out_steps.extend(_propose_bbox_by_color_steps_v141(train_pairs=train_pairs))
        # Direct crop by color bbox (grid→grid) to avoid needing bbox/patch intermediate stages.
        for st in _propose_bbox_by_color_steps_v141(train_pairs=train_pairs)[:10]:
            c = int(st.args.get("color", 0))
            if int(c) == 0:
                continue
            out_steps.append(ProgramStepV141(op_id="crop_bbox_by_color", args={"color": int(c)}))

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

    # One-shot object crop (grid→grid) as a primitive: cc4 + select_obj + bbox + crop + commit.
    # Bound aggressively to avoid branching blowups; selector args are deterministic and generic.
    if (
        bool(avail.get("grid", False))
        and not bool(avail.get("objset", False))
        and not bool(avail.get("obj", False))
        and not bool(avail.get("bbox", False))
        and not bool(avail.get("patch", False))
    ):
        from .arc_solver_v134 import _infer_select_obj_args_v134

        args_list = _infer_select_obj_args_v134(
            train_pairs=train_pairs, bg=int(bg_candidates[0] if bg_candidates else 0), max_rank=1
        )
        if not args_list:
            # Shape-changing tasks (crop/commit patterns) often have no meaningful changed-mask
            # overlap in the input frame; use crop-driven inference instead.
            args_list = _infer_crop_cc4_select_args_v141(
                train_pairs=train_pairs,
                bg=int(bg_candidates[0] if bg_candidates else 0),
                max_rank=1,
                max_specs=8,
            )
        if not args_list:
            args_list = [
                {"key": "area", "order": "max", "rank": 0, "color_filter": None},
                {"key": "area", "order": "min", "rank": 0, "color_filter": None},
                {"key": "bbox_area", "order": "max", "rank": 0, "color_filter": None},
                {"key": "bbox_area", "order": "min", "rank": 0, "color_filter": None},
                {"key": "width", "order": "max", "rank": 0, "color_filter": None},
                {"key": "height", "order": "max", "rank": 0, "color_filter": None},
            ]
        args_list = [dict(a) for a in args_list if isinstance(a, dict)]
        args_list.sort(key=lambda a: canonical_json_dumps(a))
        args_list = args_list[:6]

        for bg in list(bg_candidates[:2] if bg_candidates else [0]):
            for sa in args_list:
                a = dict(sa)
                a["bg"] = int(bg)
                out_steps.append(ProgramStepV141(op_id="crop_cc4_select", args=dict(a)))

    if bool(avail.get("objset", False)) and not bool(avail.get("obj", False)):
        from .arc_solver_v134 import _infer_select_obj_args_v134

        args_list = _infer_select_obj_args_v134(
            train_pairs=train_pairs, bg=int(bg_candidates[0] if bg_candidates else 0), max_rank=2
        )
        if not args_list:
            # Deterministic small fallback (no data-dependent heuristics): enough to unlock
            # object-centric pipelines when overlap inference is ambiguous.
            args_list = [
                {"key": "area", "order": "max", "rank": 0, "color_filter": None},
                {"key": "area", "order": "min", "rank": 0, "color_filter": None},
                {"key": "bbox_area", "order": "max", "rank": 0, "color_filter": None},
                {"key": "bbox_area", "order": "min", "rank": 0, "color_filter": None},
                {"key": "width", "order": "max", "rank": 0, "color_filter": None},
                {"key": "height", "order": "max", "rank": 0, "color_filter": None},
                {"key": "top", "order": "min", "rank": 0, "color_filter": None},
                {"key": "left", "order": "min", "rank": 0, "color_filter": None},
            ]
        for a in args_list:
            out_steps.append(ProgramStepV141(op_id="select_obj", args=dict(a)))

    if bool(avail.get("obj", False)) and not bool(avail.get("bbox", False)):
        out_steps.append(ProgramStepV141(op_id="obj_bbox", args={}))

    if bool(avail.get("obj", False)) and not bool(avail.get("patch", False)):
        for bg in bg_candidates:
            out_steps.append(ProgramStepV141(op_id="obj_patch", args={"bg": int(bg)}))

    if bool(avail.get("bbox", False)):
        out_steps.append(ProgramStepV141(op_id="crop_bbox", args={}))

    if bool(avail.get("patch", False)):
        # crop_bbox writes patch (not grid); commit_patch is required to make the crop effective.
        out_steps.append(ProgramStepV141(op_id="commit_patch", args={}))
        # Patch-level transforms (generic): allow reusing object crops with simple symmetries
        # before paste/commit. This reduces depth for many copy/transform tasks.
        out_steps.append(ProgramStepV141(op_id="patch_rotate90", args={}))
        out_steps.append(ProgramStepV141(op_id="patch_rotate180", args={}))
        out_steps.append(ProgramStepV141(op_id="patch_rotate270", args={}))
        out_steps.append(ProgramStepV141(op_id="patch_reflect_h", args={}))
        out_steps.append(ProgramStepV141(op_id="patch_reflect_v", args={}))
        out_steps.append(ProgramStepV141(op_id="patch_transpose", args={}))
        # Patch translation (mask/object movement within a fixed frame).
        # This is generic and reduces branching for "move-then-paint" pipelines.
        patch_shifts: Set[Tuple[int, int]] = {
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-2, 0),
            (2, 0),
            (0, -2),
            (0, 2),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        }
        try:
            from .grid_v124 import bbox_nonzero_v124

            for inp, out in train_pairs:
                hi, wi = grid_shape_v124(inp)
                if tuple(int(x) for x in grid_shape_v124(out)) != (int(hi), int(wi)):
                    continue
                for bg in (bg_candidates[:2] if bg_candidates else [0]):
                    try:
                        ir0, ic0, _, _ = bbox_nonzero_v124(inp, bg=int(bg))
                        or0, oc0, _, _ = bbox_nonzero_v124(out, bg=int(bg))
                        dy = int(or0 - ir0)
                        dx = int(oc0 - ic0)
                        if dy == 0 and dx == 0:
                            continue
                        patch_shifts.add((int(dy), int(dx)))
                    except Exception:
                        continue
        except Exception:
            pass
        pad_cands: List[int] = [0]
        if bg_candidates:
            pad_cands.append(int(bg_candidates[0]))
        for dy, dx in sorted(patch_shifts)[:12]:
            for pad in sorted(set(int(x) for x in pad_cands))[:2]:
                out_steps.append(
                    ProgramStepV141(
                        op_id="patch_translate",
                        args={"dx": int(dx), "dy": int(dy), "pad": int(pad)},
                    )
                )
        # Mask utilities (patch-as-mask).
        out_steps.append(ProgramStepV141(op_id="mask_outline", args={}))
        out_steps.append(ProgramStepV141(op_id="mask_not", args={}))
        out_steps.append(ProgramStepV141(op_id="mask_dilate", args={"steps": 1}))
        out_steps.append(ProgramStepV141(op_id="mask_dilate", args={"steps": 2}))
        out_steps.append(ProgramStepV141(op_id="mask_box_dilate", args={"radius": 1}))
        out_steps.append(ProgramStepV141(op_id="mask_box_dilate", args={"radius": 2}))
        for c in palette_out[:8]:
            out_steps.append(ProgramStepV141(op_id="mask_and_color", args={"color": int(c)}))
        for bg in (bg_candidates[:2] if bg_candidates else [0]):
            out_steps.append(ProgramStepV141(op_id="mask_and_nonbg", args={"bg": int(bg)}))
        # Paint/fill using the current patch as a mask/seed.
        for c in palette_out[:6]:
            out_steps.append(ProgramStepV141(op_id="paint_mask", args={"color": int(c)}))
        for bg in (bg_candidates[:2] if bg_candidates else [0]):
            for c in palette_out[:4]:
                out_steps.append(
                    ProgramStepV141(
                        op_id="paint_mask",
                        args={"color": int(c), "mode": "only_bg", "bg": int(bg)},
                    )
                )
        for bg in (bg_candidates[:1] if bg_candidates else [0]):
            for fc in palette_out[:6]:
                if int(fc) == int(bg):
                    continue
                out_steps.append(
                    ProgramStepV141(
                        op_id="flood_fill",
                        args={"target_color": int(bg), "fill_color": int(fc)},
                    )
                )

        # Derived paste positions (generic, deterministic):
        # v132 only proposes paste around the origin (±2), which leaves most copy/transform
        # tasks inexpressible under compositional search. Here we add a small, data-driven
        # set of candidate (top,left) placements derived from train outputs and change bboxes.
        #
        # This is NOT per-task hacking: the positions come from observed geometry in the
        # demonstration pairs and are bounded and canonicalized.
        from .grid_v124 import bbox_nonzero_v124

        paste_positions: Set[Tuple[int, int]] = {(0, 0)}
        for bg in (bg_candidates[:2] if bg_candidates else [0]):
            for inp, out in train_pairs:
                try:
                    r0, c0, _, _ = bbox_nonzero_v124(out, bg=int(bg))
                    paste_positions.add((int(r0), int(c0)))
                except Exception:
                    pass
                try:
                    hi, wi = grid_shape_v124(inp)
                    ho, wo = grid_shape_v124(out)
                    if (hi, wi) != (ho, wo) or hi <= 0 or wi <= 0:
                        continue
                    rmin = None
                    cmin = None
                    for r in range(int(hi)):
                        for c in range(int(wi)):
                            if int(inp[r][c]) != int(out[r][c]):
                                rr = int(r)
                                cc = int(c)
                                rmin = rr if rmin is None else min(int(rmin), rr)
                                cmin = cc if cmin is None else min(int(cmin), cc)
                    if rmin is not None and cmin is not None:
                        paste_positions.add((int(rmin), int(cmin)))
                except Exception:
                    pass

        transparent_cands: List[int] = [0]
        if bg_candidates:
            transparent_cands.append(int(bg_candidates[0]))
        transparent_cands_s = sorted(set(int(x) for x in transparent_cands))
        for top, left in sorted(paste_positions)[:16]:
            for tr in transparent_cands_s:
                out_steps.append(
                    ProgramStepV141(op_id="paste", args={"top": int(top), "left": int(left), "transparent": int(tr)})
                )

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

    # downsample_mode: if train pairs indicate consistent integer block downsampling.
    # Deterministic and generic; only propose small factors to avoid branching blowups.
    ds_sy: Optional[int] = None
    ds_sx: Optional[int] = None
    ok_ds = True
    for (hi, wi), (ho, wo) in zip(train_in_shapes, train_out_shapes):
        if int(ho) <= 0 or int(wo) <= 0:
            ok_ds = False
            break
        if int(hi) % int(ho) != 0 or int(wi) % int(wo) != 0:
            ok_ds = False
            break
        sy = int(hi // ho)
        sx = int(wi // wo)
        if ds_sy is None:
            ds_sy = int(sy)
            ds_sx = int(sx)
        elif int(sy) != int(ds_sy) or int(sx) != int(ds_sx):
            ok_ds = False
            break
    if ok_ds and ds_sy is not None and ds_sx is not None and (int(ds_sy), int(ds_sx)) in {(2, 2), (2, 3), (3, 2), (3, 3)}:
        out_steps.append(ProgramStepV141(op_id="downsample_mode", args={"sy": int(ds_sy), "sx": int(ds_sx)}))

    if bool(avail.get("bbox", False)):
        for c in palette_out:
            out_steps.append(ProgramStepV141(op_id="paint_rect", args={"color": int(c)}))
            out_steps.append(ProgramStepV141(op_id="draw_rect_border", args={"color": int(c), "thickness": 1}))

    # --- Generic proposal expansion (v132 variant enumerator) ---
    # The original v141 step proposer was intentionally conservative and only proposed some
    # grid ops when they were *direct* single-step solvers. This makes most ARC tasks
    # inexpressible under compositional search and leads to pervasive MISSING_OPERATOR /
    # SEARCH_BUDGET_EXCEEDED.
    #
    # Here we include a bounded, deterministic enumeration of generic operator variants,
    # then filter by typed slot availability (reads ⊆ avail). This is NOT per-task hacking:
    # it is a DSL-level expressivity + proposal-policy fix.
    try:
        from .arc_ops_v132 import propose_step_variants_v132

        # Reduce high-branching recolor enumeration using only information available
        # in the demonstration pairs (train_in → train_out). This is generic and
        # deterministic, and it prevents SEARCH_BUDGET_EXCEEDED from being dominated by
        # irrelevant color-pair candidates.
        #
        # NOTE: this does not change solver correctness; it only prunes proposal space.
        diff_from_counts: Dict[int, int] = {}
        diff_to_counts: Dict[int, int] = {}
        in_counts: Dict[int, int] = {}
        out_counts: Dict[int, int] = {}
        for inp, out in train_pairs:
            for row in inp:
                for v in row:
                    in_counts[int(v)] = int(in_counts.get(int(v), 0)) + 1
            for row in out:
                for v in row:
                    out_counts[int(v)] = int(out_counts.get(int(v), 0)) + 1
            hi, wi = grid_shape_v124(inp)
            ho, wo = grid_shape_v124(out)
            if (int(hi), int(wi)) != (int(ho), int(wo)) or int(hi) <= 0 or int(wi) <= 0:
                continue
            for r in range(int(hi)):
                for c in range(int(wi)):
                    vi = int(inp[int(r)][int(c)])
                    vo = int(out[int(r)][int(c)])
                    if vi == vo:
                        continue
                    diff_from_counts[int(vi)] = int(diff_from_counts.get(int(vi), 0)) + 1
                    diff_to_counts[int(vo)] = int(diff_to_counts.get(int(vo), 0)) + 1
        for row in test_in:
            for v in row:
                in_counts[int(v)] = int(in_counts.get(int(v), 0)) + 1

        def _top_k_colors(counts: Dict[int, int], k: int) -> List[int]:
            items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
            return [int(c) for c, _n in items[: int(k)]]

        # Prefer colors that actually change in the demos; otherwise fall back to the
        # most frequent palette colors (bounded).
        from_cands = _top_k_colors(diff_from_counts, 6) or _top_k_colors(in_counts, 6)
        to_cands = _top_k_colors(diff_to_counts, 6) or _top_k_colors(out_counts, 6) or list(palette_out[:6])
        # Always keep background candidates available (they are frequently involved in recolors).
        for bg in bg_candidates[:2]:
            if int(bg) not in from_cands:
                from_cands.append(int(bg))
            if int(bg) not in to_cands:
                to_cands.append(int(bg))
        from_set = set(int(c) for c in from_cands)
        to_set = set(int(c) for c in to_cands)

        # Keep the small enumerator domain; add larger, *derived* translations separately
        # to avoid blowing up the branching factor.
        variants = propose_step_variants_v132(train_pairs=train_pairs, test_in=test_in, max_translate_shift=2)
        for row in variants:
            if not isinstance(row, dict):
                continue
            op_id = str(row.get("op_id") or "")
            if not op_id or op_id in {"macro_call", "concept_call"}:
                continue
            # Prune recolor candidates to a small, data-driven domain.
            if op_id == "replace_color":
                args0 = row.get("args") if isinstance(row.get("args"), dict) else {}
                fc = int(args0.get("from_color", -1))
                tc = int(args0.get("to_color", -1))
                if int(fc) == int(tc):
                    continue
                if int(fc) not in from_set or int(tc) not in to_set:
                    continue
            # Avoid high-branching duplicates: v141 already proposes these via tighter, stage-aware rules.
            if op_id in {
                "cc4",
                "select_obj",
                "obj_bbox",
                "crop_bbox",
                "commit_patch",
                "new_canvas",
                "paint_rect",
                "draw_rect_border",
                "crop_bbox_nonzero",
                "pad_to",
            }:
                continue
            args = row.get("args") if isinstance(row.get("args"), dict) else {}
            od = OP_DEFS_V141.get(op_id)
            if od is None:
                continue
            ok = True
            for r in od.reads:
                if not bool(avail.get(str(r), False)):
                    ok = False
                    break
            if not ok:
                continue
            # Expand generic pad/transparent choices (bg candidates) for translate/paste.
            if op_id == "translate":
                pad_cands: List[int] = [int(args.get("pad", 0) or 0)]
                if bg_candidates:
                    pad_cands.append(int(bg_candidates[0]))
                for bg in sorted(set(int(x) for x in pad_cands)):
                    a2 = dict(args)
                    a2["pad"] = int(bg)
                    out_steps.append(ProgramStepV141(op_id="translate", args=dict(a2)))
                continue
            if op_id == "paste":
                tr_cands: List[int] = [int(args.get("transparent", 0) or 0)]
                if bg_candidates:
                    tr_cands.append(int(bg_candidates[0]))
                for bg in sorted(set(int(x) for x in tr_cands)):
                    a2 = dict(args)
                    a2["transparent"] = int(bg)
                    out_steps.append(ProgramStepV141(op_id="paste", args=dict(a2)))
                continue
            out_steps.append(ProgramStepV141(op_id=str(op_id), args=dict(args)))
    except Exception:
        # Fail-closed: if proposal expansion errors, fall back to the conservative set.
        pass

    # Derived large translations from bbox shift (generic, data-driven, deterministic).
    # This captures moves beyond the small ±2 enumerator without exploding branching.
    derived_trans: Set[Tuple[int, int, int]] = set()
    for inp, out in train_pairs:
        hi, wi = grid_shape_v124(inp)
        ho, wo = grid_shape_v124(out)
        if (hi, wi) != (ho, wo) or hi <= 0 or wi <= 0:
            continue
        for bg in bg_candidates:
            try:
                ir0, ic0, _, _ = bbox_nonzero_v124(inp, bg=int(bg))
                or0, oc0, _, _ = bbox_nonzero_v124(out, bg=int(bg))
                dy = int(or0 - ir0)
                dx = int(oc0 - ic0)
                if dy == 0 and dx == 0:
                    continue
                derived_trans.add((int(dy), int(dx), int(bg)))
            except Exception:
                continue
    for dy, dx, bg in sorted(derived_trans):
        out_steps.append(ProgramStepV141(op_id="translate", args={"dx": int(dx), "dy": int(dy), "pad": int(bg)}))
        # When a patch exists, the same inferred shift often applies to paste placement.
        if bool(avail.get("patch", False)):
            out_steps.append(
                ProgramStepV141(
                    op_id="paste",
                    args={"left": int(dx), "top": int(dy), "transparent": int(bg)},
                )
            )

    # Derived placement translations from output shape gaps (generic): when a pipeline pads to an
    # output shape and needs to re-center / align content, propose {top,center,bottom}×{left,center,right}.
    gap_trans: Set[Tuple[int, int, int]] = set()
    for (hi, wi), (ho, wo) in zip(train_in_shapes, train_out_shapes):
        if int(hi) <= 0 or int(wi) <= 0:
            continue
        dy_gap = int(ho - hi)
        dx_gap = int(wo - wi)
        if dy_gap < 0 or dx_gap < 0:
            continue
        dy_opts: List[int] = [0, int(dy_gap)]
        dx_opts: List[int] = [0, int(dx_gap)]
        if dy_gap % 2 == 0:
            dy_opts.append(int(dy_gap // 2))
        if dx_gap % 2 == 0:
            dx_opts.append(int(dx_gap // 2))
        for dy in sorted(set(int(x) for x in dy_opts)):
            for dx in sorted(set(int(x) for x in dx_opts)):
                for bg in (bg_candidates[:1] if bg_candidates else [0]):
                    gap_trans.add((int(dy), int(dx), int(bg)))
    for dy, dx, bg in sorted(gap_trans):
        out_steps.append(ProgramStepV141(op_id="translate", args={"dx": int(dx), "dy": int(dy), "pad": int(bg)}))
        if bool(avail.get("patch", False)):
            out_steps.append(
                ProgramStepV141(
                    op_id="paste",
                    args={"left": int(dx), "top": int(dy), "transparent": int(bg)},
                )
            )

    # fill_enclosed_region (holes): bounded bg/fill choices from observed palettes.
    fill_colors: Set[int] = set(int(c) for c in palette_out)
    for g in train_in + [test_in]:
        for c in unique_colors_v124(g):
            fill_colors.add(int(c))
    fill_colors_s = sorted(set(int(c) for c in fill_colors))
    for c in fill_colors_s:
        out_steps.append(ProgramStepV141(op_id="mask_by_color", args={"color": int(c)}))
    for bg in (bg_candidates[:2] if bg_candidates else [0]):
        out_steps.append(ProgramStepV141(op_id="mask_bg", args={"bg": int(bg)}))
    out_steps.append(ProgramStepV141(op_id="mask_border", args={}))
    for bg in bg_candidates:
        out_steps.append(ProgramStepV141(op_id="mask_nonbg", args={"bg": int(bg)}))
    for bg in bg_candidates:
        out_steps.append(ProgramStepV141(op_id="mask_cross_center", args={"bg": int(bg)}))
    for bg in bg_candidates:
        for fc in fill_colors_s:
            if int(fc) == int(bg):
                continue
            out_steps.append(ProgramStepV141(op_id="fill_enclosed_region", args={"bg": int(bg), "fill": int(fc)}))

    # cc4_color_bars: count 4-neigh connected components per color and render a right-aligned bar chart.
    # Generic and deterministic; helps tasks whose outputs are "component counts by color".
    out_steps.append(ProgramStepV141(op_id="cc4_color_bars", args={}))
    for bg in bg_candidates:
        out_steps.append(ProgramStepV141(op_id="cc4_color_bars", args={"bg": int(bg)}))

    # cc4_color_area_column: list objects (cc4 by same-color) as a single-column "color repeated by area",
    # ordered by top-left. Useful when outputs are tall 1-column summaries.
    if any(int(w) == 1 for _h, w in train_out_shapes):
        out_steps.append(ProgramStepV141(op_id="cc4_nonbg_bfs_column", args={}))
        out_steps.append(ProgramStepV141(op_id="cc4_color_area_column", args={}))
        for bg in bg_candidates:
            out_steps.append(ProgramStepV141(op_id="cc4_nonbg_bfs_column", args={"bg": int(bg)}))
            out_steps.append(ProgramStepV141(op_id="cc4_color_area_column", args={"bg": int(bg)}))

    # transpose (diagonal reflection / swap axes): a common ARC primitive missing from v132.
    out_steps.append(ProgramStepV141(op_id="transpose", args={}))

    # symmetry fill: fill background cells from their mirror counterparts.
    for bg in (bg_candidates if bg_candidates else [0]):
        out_steps.append(ProgramStepV141(op_id="symmetry_fill_h", args={"bg": int(bg)}))
        out_steps.append(ProgramStepV141(op_id="symmetry_fill_v", args={"bg": int(bg)}))
        out_steps.append(ProgramStepV141(op_id="symmetry_fill_rot180", args={"bg": int(bg)}))

    # fill_enclosed_region: fill background regions that are not connected to the border.
    # Generic and deterministic; propose only a bounded set of (bg, fill) candidates.
    for bg in (bg_candidates[:2] if bg_candidates else [0]):
        for fc in palette_out[:6]:
            if int(fc) == int(bg):
                continue
            out_steps.append(ProgramStepV141(op_id="fill_enclosed_region", args={"bg": int(bg), "fill": int(fc)}))

    # gravity: compact non-bg mass in a direction (generic).
    for bg in bg_candidates:
        for d in ("down", "up", "left", "right"):
            out_steps.append(ProgramStepV141(op_id="gravity", args={"dir": str(d), "bg": int(bg)}))

    # smear_nonbg: propagate the last seen non-background color across background runs.
    # Generic and deterministic; useful for many ARC line-fill tasks.
    for bg in (bg_candidates[:2] if bg_candidates else [0]):
        for d in ("right", "left", "down", "up"):
            out_steps.append(ProgramStepV141(op_id="smear_nonbg", args={"dir": str(d), "bg": int(bg)}))

    # --- Additional v141/v134 generic ops not covered by v132 enumerator ---
    # repeat_grid (mode=cell/grid/corners, small bounded factors)
    for mode in ("cell", "grid", "corners"):
        if mode == "cell":
            for sy in (1, 2, 3):
                for sx in (1, 2, 3):
                    if int(sy) == 1 and int(sx) == 1:
                        continue
                    out_steps.append(ProgramStepV141(op_id="repeat_grid", args={"mode": "cell", "sy": int(sy), "sx": int(sx)}))
        elif mode == "grid":
            for ry in (1, 2, 3):
                for rx in (1, 2, 3):
                    if int(ry) == 1 and int(rx) == 1:
                        continue
                    out_steps.append(ProgramStepV141(op_id="repeat_grid", args={"mode": "grid", "ry": int(ry), "rx": int(rx)}))
        else:
            # Sparse upscale: place each cell value on the corners of its sy×sx block.
            for sy in (3,):
                for sx in (3,):
                    for bg in (bg_candidates[:1] if bg_candidates else [0]):
                        out_steps.append(
                            ProgramStepV141(op_id="repeat_grid", args={"mode": "corners", "sy": int(sy), "sx": int(sx), "bg": int(bg)})
                        )

    # relational_expand: when outputs are (H*H, W*W), expand by pairwise equality/inequality.
    ok_rel = True
    for (hi, wi), (ho, wo) in zip(train_in_shapes, train_out_shapes):
        if int(hi) <= 0 or int(wi) <= 0:
            ok_rel = False
            break
        if int(ho) != int(hi) * int(hi) or int(wo) != int(wi) * int(wi):
            ok_rel = False
            break
    if ok_rel:
        for bg in (bg_candidates[:1] if bg_candidates else [0]):
            out_steps.append(ProgramStepV141(op_id="relational_expand", args={"mode": "neq", "bg": int(bg)}))
            out_steps.append(ProgramStepV141(op_id="relational_expand", args={"mode": "eq", "bg": int(bg)}))
        for bg in (bg_candidates[:2] if bg_candidates else [0]):
            out_steps.append(ProgramStepV141(op_id="uniform_line_expand", args={"prefer": "row", "bg": int(bg)}))
            out_steps.append(ProgramStepV141(op_id="uniform_line_expand", args={"prefer": "col", "bg": int(bg)}))

    # overlay_self_translate (bounded shifts, bg candidates as pad)
    shifts8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dy, dx in shifts8:
        for bg in bg_candidates:
            out_steps.append(ProgramStepV141(op_id="overlay_self_translate", args={"dx": int(dx), "dy": int(dy), "pad": int(bg)}))

    # propagate_color_translate (bounded shifts; top-4 output colors; bg candidates as pad)
    # Deterministic: choose frequent output colors from training outputs.
    color_counts: Dict[int, int] = {}
    for _inp, out in train_pairs:
        for row in out:
            for v in row:
                color_counts[int(v)] = int(color_counts.get(int(v), 0)) + 1
    color_ranked = sorted(list(color_counts.items()), key=lambda kv: (-int(kv[1]), int(kv[0])))
    top_colors = [int(c) for c, _n in color_ranked[:4]] if color_ranked else list(palette_out[:4])
    for dy, dx in shifts8:
        for col in top_colors:
            for bg in bg_candidates:
                out_steps.append(
                    ProgramStepV141(
                        op_id="propagate_color_translate",
                        args={"dx": int(dx), "dy": int(dy), "color": int(col), "pad": int(bg)},
                    )
                )

    # propagate_nonbg_translate (bounded shifts; bg candidates as pad)
    # Useful when multiple colors must propagate and choosing a single "color" would branch too much.
    for dy, dx in shifts8:
        for bg in bg_candidates:
            out_steps.append(ProgramStepV141(op_id="propagate_nonbg_translate", args={"dx": int(dx), "dy": int(dy), "pad": int(bg)}))

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


def _norm_concept_templates_v141(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize concept templates (v146) into a stable, minimal form:
      {concept_id, op_ids, cost_bits, support, signature?, steps?, support_task_feat_keys?, support_task_feat_keys_v2?}

    Supports:
      - arc_concept_template_v146: op_id-only templates (args chosen by instantiation policy)
      - arc_concept_csg_v148: concrete CSG templates carrying explicit inner steps (no arg branching)
      - arc_concept_csg_v149: CSG templates with deterministic binders in args (resolved at call-time)
    """
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for row0 in rows:
        row = row0 if isinstance(row0, dict) else {}
        if str(row.get("kind") or "") not in {"arc_concept_template_v146", "arc_concept_csg_v148", "arc_concept_csg_v149"} and "concept_id" not in row:
            continue
        cid = str(row.get("concept_id") or row.get("csv_id") or row.get("csg_id") or "")
        if not cid or cid in seen:
            continue
        steps_raw = row.get("steps")
        steps: List[Dict[str, Any]] = []
        if isinstance(steps_raw, list) and steps_raw:
            for st0 in steps_raw:
                st = st0 if isinstance(st0, dict) else {}
                op0 = str(st.get("op_id") or "")
                if not op0 or op0 in {"macro_call", "concept_call"}:
                    # Concrete concept CSG cannot contain nested meta calls (runtime forbids).
                    steps = []
                    break
                a0 = st.get("args") if isinstance(st.get("args"), dict) else {}
                steps.append({"op_id": str(op0), "args": {str(k): a0[k] for k in sorted(a0.keys())}})

        op_ids_raw = row.get("op_ids")
        op_ids: List[str] = []
        if isinstance(op_ids_raw, list) and op_ids_raw:
            op_ids = [str(x) for x in op_ids_raw if str(x)]
            op_ids = [x for x in op_ids if x not in {"macro_call", "concept_call"}]
        if not op_ids and steps:
            op_ids = [str(s.get("op_id") or "") for s in steps if str(s.get("op_id") or "")]
        if not op_ids:
            continue
        cost_bits = row.get("cost_bits")
        cb = 10
        if isinstance(cost_bits, int):
            cb = int(cost_bits)
        elif isinstance(cost_bits, str) and str(cost_bits).isdigit():
            cb = int(str(cost_bits))
        support = row.get("support")
        sup = 0
        if isinstance(support, int):
            sup = int(support)
        elif isinstance(support, str) and str(support).isdigit():
            sup = int(str(support))
        sig = row.get("signature") if isinstance(row.get("signature"), dict) else None
        support_task_feat_keys_raw = row.get("support_task_feat_keys")
        support_task_feat_keys: List[List[str]] = []
        if isinstance(support_task_feat_keys_raw, list) and support_task_feat_keys_raw:
            for fk0 in support_task_feat_keys_raw:
                if not isinstance(fk0, (list, tuple)) or len(fk0) != 3:
                    continue
                support_task_feat_keys.append([str(fk0[0]), str(fk0[1]), str(fk0[2])])
            support_task_feat_keys.sort()
        support_task_feat_keys_v2_raw = row.get("support_task_feat_keys_v2")
        support_task_feat_keys_v2: List[List[str]] = []
        if isinstance(support_task_feat_keys_v2_raw, list) and support_task_feat_keys_v2_raw:
            for fk0 in support_task_feat_keys_v2_raw:
                if not isinstance(fk0, (list, tuple)) or len(fk0) < 3:
                    continue
                support_task_feat_keys_v2.append([str(x) for x in fk0])
            support_task_feat_keys_v2.sort()
        seen.add(cid)
        out.append(
            {
                "concept_id": str(cid),
                "op_ids": list(op_ids),
                "cost_bits": int(cb),
                "support": int(sup),
                "signature": dict(sig) if isinstance(sig, dict) else None,
                "steps": list(steps) if steps else None,
                "support_task_feat_keys": list(support_task_feat_keys) if support_task_feat_keys else None,
                "support_task_feat_keys_v2": list(support_task_feat_keys_v2) if support_task_feat_keys_v2 else None,
            }
        )
    # Deterministic selection bias:
    # Under tight concept_max_templates, support-first sorting tends to crowd out longer
    # concept closures, forcing the solver to spend depth budget on primitive-like concepts
    # and blowing up search.
    #
    # We instead build a length-stratified, round-robin ordering:
    #   - within each length: higher support first (then cheaper MDL, then id)
    #   - across lengths: prefer longer templates early to enable compression
    #
    # This is generic (not task-specific) and preserves determinism.
    def _len_rr_order(rows_in: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_len: Dict[int, List[Dict[str, Any]]] = {}
        for row in rows_in:
            k = int(len(list(row.get("op_ids") or [])))
            by_len.setdefault(k, []).append(row)
        for k, rows_k in by_len.items():
            rows_k.sort(
                key=lambda r: (
                    -int(r.get("support") or 0),
                    int(r.get("cost_bits") or 10),
                    str(r.get("concept_id") or ""),
                )
            )
        lens = sorted(by_len.keys(), reverse=True)
        merged0: List[Dict[str, Any]] = []
        while True:
            emitted = False
            for k in lens:
                rows_k = by_len.get(k) or []
                if not rows_k:
                    continue
                merged0.append(rows_k.pop(0))
                emitted = True
            if not emitted:
                break
        return merged0

    # Prefer concrete CSG templates (explicit inner steps, no arg branching) ahead of
    # op_ids-only templates. This is generic: concrete closures reduce branching and
    # make abstraction_pressure effective under tight concept_max_templates.
    concrete = [r for r in out if isinstance(r.get("steps"), list) and bool(r.get("steps"))]
    abstract = [r for r in out if not (isinstance(r.get("steps"), list) and bool(r.get("steps")))]
    return _len_rr_order(concrete) + _len_rr_order(abstract)


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
    get_candidates: Optional[Callable[[Tuple["ProgramStepV141", ...]], Sequence["ProgramStepV141"]]] = None,
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
            if get_candidates is not None:
                candidates = list(get_candidates(steps_so_far))
            else:
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
            # Preserve the proposal ordering from _propose_next_steps_v141 (policy-rank + train-loss
            # prefilter when needed). Re-sorting by cost here destroys semantic ordering and can
            # drop the only viable arg instantiation under tight branch caps.
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


def _instantiate_concept_steps_v141(
    *,
    concept_op_ids: Sequence[str],
    steps_prefix: Tuple["ProgramStepV141", ...],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    bg_candidates: Sequence[int],
    shapes_out: Sequence[Tuple[int, int]],
    palette_out: Sequence[int],
    direct_steps: Sequence["ProgramStepV141"],
    max_instantiations: int,
    max_branch_per_op: int,
    get_candidates: Optional[Callable[[Tuple["ProgramStepV141", ...]], Sequence["ProgramStepV141"]]] = None,
) -> List[Tuple["ProgramStepV141", ...]]:
    """
    Deterministically instantiate a concept template (sequence of op_ids) into executable steps
    by choosing concrete args via the same per-step proposal policy used by the solver.
    """
    want = [str(x) for x in concept_op_ids if str(x)]
    want = [x for x in want if x not in {"macro_call", "concept_call"}]
    if not want:
        return []
    max_inst = max(1, int(max_instantiations))
    max_branch = max(1, int(max_branch_per_op))

    frontier: List[Tuple["ProgramStepV141", ...]] = [tuple()]
    for op in want:
        next_frontier: List[Tuple["ProgramStepV141", ...]] = []
        for inst in frontier:
            steps_so_far = tuple(list(steps_prefix) + list(inst))
            if get_candidates is not None:
                candidates = list(get_candidates(steps_so_far))
            else:
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
            # Preserve the proposal ordering from _propose_next_steps_v141 (policy-rank + train-loss
            # prefilter when needed). Re-sorting by cost here destroys semantic ordering and can
            # drop the only viable arg instantiation under tight branch caps.
            for st in matches[:max_branch]:
                next_frontier.append(tuple(list(inst) + [st]))
        if not next_frontier:
            return []

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
    enable_dominated_state_pruning: bool = True
    enable_concept_support_feat_ranking: bool = False

    # Macro instantiation policy (bounded, deterministic).
    macro_propose_max_depth: int = 0
    macro_max_templates: int = 24
    macro_max_instantiations: int = 10
    macro_max_branch_per_op: int = 10

    # Concept template instantiation policy (bounded, deterministic).
    # Default is permissive: concept calls are allowed at any outer depth.
    concept_propose_max_depth: int = 99
    concept_max_templates: int = 24
    concept_max_instantiations: int = 10
    concept_max_branch_per_op: int = 10

    # CSV applicability gate behavior:
    #
    # When enabled, the CSV gate may admit *non-grid-writing* semantic moves that do not reduce
    # immediate TRAIN grid loss but do improve semantic readiness (e.g., build obj/bbox/patch).
    #
    # This stays deterministic and fail-closed:
    # - the move must not worsen TRAIN loss,
    # - and must change the internal slot signature (or reduce the missing-slot penalty),
    # - and candidate branching remains strictly capped by concept-as-policy selection.
    #
    # Rationale: many valid multi-step solutions require enabling moves (slot builders / patch
    # transforms) before the first grid write reduces mismatch. Disallowing these can cause
    # csv_survivors==0 and immediate "MISSING_OPERATOR" even when the vocabulary is sufficient.
    csv_allow_slot_progress: bool = False

    # Optional: induce "concrete" Concept SubGraphs (CSG) from near-miss trace programs when search
    # would otherwise exhaust the budget. This is a deterministic, generic compression mechanism:
    # it wraps a deep (multi-step) trace prefix as a single concept_call with fixed args (no
    # internal branching), reducing effective depth before the search explodes.
    #
    # Disabled by default to preserve baseline behavior; enable explicitly for ARC runs.
    enable_trace_csg_induction: bool = False
    trace_csg_induction_first_pass_frac: float = 0.6
    trace_csg_induction_max_templates: int = 8
    trace_csg_induction_min_inner_steps: int = 3
    trace_csg_induction_max_inner_steps: int = 18
    trace_csg_induction_max_loss_cells: int = 60

    # Optional stages (accepted for harness compatibility).
    enable_repair_stage: bool = True
    enable_residual_stage: bool = True
    enable_refine_stage: bool = True
    enable_point_patch_repair: bool = False
    point_patch_max_points: int = 12


def _flatten_meta_steps_v141(steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flatten meta steps (macro_call / concept_call) into base primitive steps.

    Returned steps are dicts with {op_id, args}. Nested meta calls inside meta calls are not
    permitted by the runtime; this flattening is meant only for trace mining / CSG induction.
    """
    out: List[Dict[str, Any]] = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        op_id = str(st.get("op_id") or "")
        if not op_id:
            continue
        args = st.get("args") if isinstance(st.get("args"), dict) else {}
        if op_id in {"macro_call", "concept_call"}:
            inner = args.get("steps")
            if isinstance(inner, list):
                out.extend(_flatten_meta_steps_v141([x for x in inner if isinstance(x, dict)]))
            continue
        out.append({"op_id": str(op_id), "args": {str(k): args[k] for k in sorted(args.keys())}})
    return out


def _concept_id_for_concrete_steps_v141(steps: Sequence[Dict[str, Any]]) -> str:
    body = {
        "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
        "kind": "concept_csg_key_v141",
        "steps": [dict(s) for s in steps if isinstance(s, dict)],
    }
    cid = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return "csg_trace_" + str(cid)[:16]


def _induce_trace_csg_templates_v141(
    *,
    trace_programs: Sequence[Dict[str, Any]],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    max_templates: int,
    min_inner_steps: int,
    max_inner_steps: int,
    max_loss_cells: int,
) -> List[Dict[str, Any]]:
    """
    Deterministically induce "concrete" concept templates (CSG) from near-miss trace programs.

    A concrete template carries explicit inner steps (with args), so a concept_call becomes an
    atomic 1-step action that does not branch over args. This is the minimal operator needed to
    turn deep traces into reusable closures inside a task run (without per-task heuristics).
    """
    # Baseline (empty program) train loss: retained for diagnostics/experiments.
    # (Note: we do not require induced prefixes to improve baseline; many useful pipelines
    # are "setup" steps that only pay off after subsequent composition.)
    prog0 = ProgramV141(steps=tuple())
    info0 = _eval_program_on_pairs_v141(
        program=prog0,
        train_pairs=train_pairs,
        test_in=test_in,
        apply_cache={},
        grid_hash_cache={},
        metrics={},
    )
    base_loss = (int(info0.loss[0]), int(info0.loss[1]))

    def _bbox_of_color(g: GridV124, *, color: int) -> Optional[Tuple[int, int, int, int]]:
        r0 = c0 = 10**9
        r1 = c1 = -1
        for r, row in enumerate(g):
            for c, x in enumerate(row):
                if int(x) != int(color):
                    continue
                r0 = min(r0, int(r))
                c0 = min(c0, int(c))
                r1 = max(r1, int(r))
                c1 = max(c1, int(c))
        if r1 < 0:
            return None
        return (int(r0), int(c0), int(r1), int(c1))

    def _infer_color_candidates(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> Dict[str, List[int]]:
        train_in = [p[0] for p in train_pairs]
        train_out = [p[1] for p in train_pairs]
        bg = _infer_bg_candidates_v141(grids=train_in + train_out + [test_in])[0]
        in_colors: Set[int] = set()
        out_colors: Set[int] = set()
        for g in train_in + [test_in]:
            in_colors |= set(int(x) for x in unique_colors_v124(g))
        for g in train_out:
            out_colors |= set(int(x) for x in unique_colors_v124(g))
        # Deterministic ordering: prefer non-bg first, then bg.
        def order(cols: Set[int]) -> List[int]:
            xs = sorted(int(c) for c in cols)
            xs_nbg = [c for c in xs if int(c) != int(bg)]
            xs_bg = [c for c in xs if int(c) == int(bg)]
            return xs_nbg + xs_bg

        return {
            "bg": [int(bg)],
            "in": order(in_colors),
            "out": order(out_colors),
            "any": order(in_colors | out_colors),
        }

    def _infer_offset_candidates(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> List[Tuple[int, int]]:
        train_in = [p[0] for p in train_pairs]
        train_out = [p[1] for p in train_pairs]
        bg = _infer_bg_candidates_v141(grids=train_in + train_out + [test_in])[0]
        offsets: Set[Tuple[int, int]] = set()
        # Color-wise bbox offsets.
        colors_any: Set[int] = set()
        for g in train_in + train_out + [test_in]:
            colors_any |= set(int(x) for x in unique_colors_v124(g))
        colors_any = set(int(c) for c in colors_any if int(c) != int(bg))
        for (gin, gout) in train_pairs:
            for c in sorted(colors_any):
                bi = _bbox_of_color(gin, color=int(c))
                bo = _bbox_of_color(gout, color=int(c))
                if bi is None or bo is None:
                    continue
                hi = int(bi[2] - bi[0] + 1)
                wi = int(bi[3] - bi[1] + 1)
                ho = int(bo[2] - bo[0] + 1)
                wo = int(bo[3] - bo[1] + 1)
                if (hi, wi) != (ho, wo):
                    continue
                offsets.add((int(bo[0] - bi[0]), int(bo[1] - bi[1])))
        # Always include a small local neighborhood around 0 for safety.
        for dy in [-2, -1, 0, 1, 2]:
            for dx in [-2, -1, 0, 1, 2]:
                offsets.add((int(dy), int(dx)))
        out = sorted(list(offsets), key=lambda rc: (abs(int(rc[0])) + abs(int(rc[1])), int(rc[0]), int(rc[1])))
        return out[:40]

    def _optimize_flat_steps_v141(*, flat: Sequence[Dict[str, Any]]) -> Optional[Tuple[List[Dict[str, Any]], Tuple[int, int]]]:
        """
        Deterministically infer common args (colors, translations, rotations) to make a flat
        primitive-step sequence lower-loss (and ideally train-consistent). Returns concrete steps
        along with the resulting train loss (shape, cells).
        """
        if not flat:
            return None
        colors = _infer_color_candidates(train_pairs=train_pairs, test_in=test_in)
        offsets = _infer_offset_candidates(train_pairs=train_pairs, test_in=test_in)

        # Slots to optimize. Keep it small and generic.
        color_slots: List[Tuple[int, str]] = []
        translate_slots: List[Tuple[int, Tuple[int, int, int]]] = []  # (idx, (dy,dx,pad))
        rotate_slots: List[Tuple[int, int]] = []  # (idx, k)

        for i, st in enumerate(flat):
            if not isinstance(st, dict):
                continue
            op = str(st.get("op_id") or "")
            args = st.get("args") if isinstance(st.get("args"), dict) else {}
            if op == "patch_translate":
                try:
                    dy0 = int(args.get("dy"))
                    dx0 = int(args.get("dx"))
                    pad0 = int(args.get("pad") or 0)
                except Exception:
                    continue
                translate_slots.append((int(i), (int(dy0), int(dx0), int(pad0))))
                continue
            if op == "patch_rotate":
                try:
                    k0 = int(args.get("k"))
                except Exception:
                    continue
                rotate_slots.append((int(i), int(k0)))
                continue
            for k in sorted(args.keys()):
                v = args.get(k)
                if isinstance(v, bool) or v is None:
                    continue
                if not (isinstance(v, (int, str)) and str(v).lstrip("-").isdigit()):
                    continue
                kk = str(k)
                if kk == "bg" or kk == "color" or kk.endswith("_color") or kk in {"from_color", "to_color"}:
                    color_slots.append((int(i), str(kk)))

        # Limit combination blowups.
        max_combos = 192
        # Candidate values per slot.
        slot_values: List[List[Any]] = []
        slot_setters: List[Any] = []

        for (i, k) in color_slots:
            args0 = flat[i].get("args") if isinstance(flat[i].get("args"), dict) else {}
            v0 = args0.get(k)
            v0i = int(v0) if str(v0).lstrip("-").isdigit() else None
            # Use narrower domains when possible.
            domain_key = "any"
            if k in {"from_color", "target_color"}:
                domain_key = "in"
            elif k in {"to_color", "fill_color"}:
                domain_key = "out"
            elif k == "bg":
                domain_key = "bg"
            domain = list(colors.get(domain_key, colors["any"]))
            if v0i is not None and int(v0i) in domain:
                domain = [int(v0i)] + [int(x) for x in domain if int(x) != int(v0i)]
            domain = domain[:10]
            slot_values.append([int(x) for x in domain])

            def _mk_setter(ii: int, kk: str):
                def _set(steps: List[Dict[str, Any]], val: int) -> None:
                    steps[ii]["args"][kk] = int(val)

                return _set

            slot_setters.append(_mk_setter(int(i), str(k)))

        # Translate slots: treat (dy,dx) pair as a single slot, keep pad fixed.
        for (i, (dy0, dx0, pad0)) in translate_slots:
            dom = list(offsets)
            if (int(dy0), int(dx0)) in dom:
                dom = [(int(dy0), int(dx0))] + [(int(dy), int(dx)) for (dy, dx) in dom if (int(dy), int(dx)) != (int(dy0), int(dx0))]
            dom = dom[:20]
            slot_values.append(dom)

            def _mk_setter(ii: int, pad: int):
                def _set(steps: List[Dict[str, Any]], val: Tuple[int, int]) -> None:
                    dy, dx = val
                    steps[ii]["args"]["dy"] = int(dy)
                    steps[ii]["args"]["dx"] = int(dx)
                    steps[ii]["args"]["pad"] = int(pad)

                return _set

            slot_setters.append(_mk_setter(int(i), int(pad0)))

        for (i, k0) in rotate_slots:
            dom = [int(k0)] + [k for k in [0, 1, 2, 3] if int(k) != int(k0)]
            slot_values.append(dom)

            def _mk_setter(ii: int):
                def _set(steps: List[Dict[str, Any]], val: int) -> None:
                    steps[ii]["args"]["k"] = int(val)

                return _set

            slot_setters.append(_mk_setter(int(i)))

        if not slot_values:
            # Nothing to optimize.
            prog0 = ProgramV141(
                steps=tuple(
                    ProgramStepV141(op_id=str(st.get("op_id") or ""), args={str(k): (st.get("args") or {})[k] for k in sorted((st.get("args") or {}).keys())})
                    for st in flat
                    if isinstance(st, dict)
                )
            )
            info0 = _eval_program_on_pairs_v141(
                program=prog0,
                train_pairs=train_pairs,
                test_in=test_in,
                apply_cache={},
                grid_hash_cache={},
                metrics={},
            )
            return (list(flat), (int(info0.loss[0]), int(info0.loss[1])))

        # Enumerate a bounded set of assignments in deterministic order.
        assignments: List[List[Any]] = [[]]
        for vals in slot_values:
            nxt: List[List[Any]] = []
            for a in assignments:
                for v in vals:
                    nxt.append(a + [v])
                    if len(nxt) >= max_combos:
                        break
                if len(nxt) >= max_combos:
                    break
            assignments = nxt
            if len(assignments) >= max_combos:
                break

        apply_cache: Dict[Tuple[Any, ...], StateV132] = {}
        grid_hash_cache: Dict[GridV124, str] = {}
        metrics: Dict[str, int] = {}

        best: Optional[Tuple[int, int, int, str, List[Dict[str, Any]]]] = None
        for a in assignments:
            steps = [{"op_id": str(st.get("op_id") or ""), "args": dict(st.get("args") or {})} for st in flat]
            for setter, val in zip(slot_setters, a):
                setter(steps, val)
            prog = ProgramV141(
                steps=tuple(
                    ProgramStepV141(op_id=str(st.get("op_id") or ""), args={str(k): st["args"][k] for k in sorted(st["args"].keys())})
                    for st in steps
                )
            )
            info = _eval_program_on_pairs_v141(
                program=prog,
                train_pairs=train_pairs,
                test_in=test_in,
                apply_cache=apply_cache,
                grid_hash_cache=grid_hash_cache,
                metrics=metrics,
            )
            # Prefer cheaper programs; deterministic tiebreak by sig of steps.
            cost = 0
            for st in steps:
                cost += int(step_cost_bits_v141(op_id=str(st.get("op_id") or ""), args=dict(st.get("args") or {})))
            sig = sha256_hex(canonical_json_dumps(steps).encode("utf-8"))
            loss_shape, loss_cells = info.loss
            cand = (int(loss_shape), int(loss_cells), int(cost), str(sig), steps)
            if best is None or cand[:4] < best[:4]:
                best = cand
        if best is None:
            return None
        return (best[4], (int(best[0]), int(best[1])))

    # --- Prefix mining (generic, deterministic) ---------------------------------------------
    # Instead of wrapping entire near-miss traces (often too specific and arg-sensitive), mine
    # frequent *prefix* subgraphs shared across near-miss traces and promote them as callable
    # CSG closures. This targets the "how" crystallization described in SOTA INEVITÁVEL:
    # repeated pipelines become atomic concept_call steps.
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    prefix_stats: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for tp in trace_programs:
        if not isinstance(tp, dict):
            continue
        loss = tp.get("loss") if isinstance(tp.get("loss"), dict) else {}
        ls = int(loss.get("shape") or 0)
        lc = int(loss.get("cells") or 0)
        # Use only shape-consistent traces for v1. We intentionally do NOT require low loss:
        # early prefixes can still be useful if they improve over the empty baseline.
        if int(ls) != 0 or int(lc) < 0:
            continue
        steps_raw = tp.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            continue
        flat0 = _flatten_meta_steps_v141([s for s in steps_raw if isinstance(s, dict)])
        if int(len(flat0)) < int(min_inner_steps):
            continue
        max_L = min(int(max_inner_steps), int(len(flat0)))
        # Mine prefixes only (not all substrings) to bias toward "early collapse" of search.
        for L in range(int(min_inner_steps), int(max_L) + 1):
            pref = flat0[: int(L)]
            op_ids = tuple(str(st.get("op_id") or "") for st in pref if isinstance(st, dict))
            if not op_ids or any(not x for x in op_ids):
                continue
            rec = prefix_stats.get(op_ids)
            if rec is None:
                rec = {"count": 0, "examples": []}
                prefix_stats[op_ids] = rec
            rec["count"] = int(rec.get("count", 0)) + 1
            ex = rec.get("examples")
            if isinstance(ex, list) and len(ex) < 6:
                ex.append([dict(s) for s in pref if isinstance(s, dict)])

    ranked = sorted(
        prefix_stats.items(),
        key=lambda kv: (-int(kv[1].get("count", 0)), -int(len(kv[0])), canonical_json_dumps(list(kv[0]))),
    )

    for op_ids_key, rec in ranked:
        if len(out) >= int(max_templates):
            break
        examples = rec.get("examples") if isinstance(rec.get("examples"), list) else []
        best: Optional[Tuple[int, int, int, str, List[Dict[str, Any]]]] = None
        for ex in examples:
            if not isinstance(ex, list) or not ex:
                continue
            opt = _optimize_flat_steps_v141(flat=ex)
            if not opt:
                continue
            flat, (ls_opt, lc_opt) = opt
            # Only mine near-miss / solved prefixes: shape must match and cell loss must be bounded.
            # NOTE: max_loss_cells is part of the public SolveConfig (trace_csg_induction_max_loss_cells).
            # If we ignore it, the induced CSG pool becomes noisy and can crowd out useful proposals,
            # worsening SEARCH_BUDGET_EXCEEDED under tight caps.
            if int(ls_opt) != 0 or int(lc_opt) > int(max_loss_cells):
                continue
            inner_cost = 0
            inner_ops: List[str] = []
            for st in flat:
                op = str(st.get("op_id") or "")
                if not op:
                    continue
                inner_ops.append(op)
                args = st.get("args") if isinstance(st.get("args"), dict) else {}
                inner_cost += int(step_cost_bits_v141(op_id=str(op), args=dict(args)))
            if not inner_ops:
                continue
            sig = sha256_hex(canonical_json_dumps(flat).encode("utf-8"))
            cand = (int(ls_opt), int(lc_opt), int(inner_cost), str(sig), flat)
            if best is None or cand[:4] < best[:4]:
                best = cand
        if best is None:
            continue
        flat_best = best[4]
        cid = _concept_id_for_concrete_steps_v141(flat_best)
        if cid in seen:
            continue
        seen.add(cid)
        inner_ops = [str(st.get("op_id") or "") for st in flat_best if isinstance(st, dict) and str(st.get("op_id") or "")]
        inner_cost = 0
        for st in flat_best:
            if not isinstance(st, dict):
                continue
            op = str(st.get("op_id") or "")
            if not op:
                continue
            args = st.get("args") if isinstance(st.get("args"), dict) else {}
            inner_cost += int(step_cost_bits_v141(op_id=str(op), args=dict(args)))
        cost_bits = max(1, int(inner_cost) - 2)
        cost_bits = min(int(cost_bits), max(1, int(inner_cost) - 1))
        out.append(
            {
                "kind": "arc_concept_csg_v148",
                "schema_version": 148,
                "concept_id": str(cid),
                "op_ids": list(inner_ops),
                "steps": [dict(s) for s in flat_best],
                # Give induced templates an intentionally high support so they are proposed early
                # within this solve call (they are ephemeral and not persisted across tasks).
                "support": int(rec.get("count", 1) or 1) + 1_000_000,
                "cost_bits": int(cost_bits),
                "signature": {
                    "diff_kind": "TRACE_CSG_PREFIX_MINED",
                    "diff_bucket": "ANY",
                    "required_depth": 1,
                    "shape_ok": True,
                },
            }
        )

    return out


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
    reset_trace_snapshot_v141()
    t_start = time.monotonic()

    def _trace_stage(stage: str, **extra: Any) -> None:
        """
        Best-effort progress marker for the wall-time timeout harness.
        Must be cheap and deterministic.
        """
        global _TRACE_METRICS_SNAPSHOT_V141
        snap: Dict[str, Any] = {}
        try:
            if isinstance(_TRACE_METRICS_SNAPSHOT_V141, dict):
                snap = dict(_TRACE_METRICS_SNAPSHOT_V141)
        except Exception:
            snap = {}
        snap["stage"] = str(stage)
        snap["t_total_s"] = float(time.monotonic() - t_start)
        for k in sorted(extra.keys()):
            snap[str(k)] = extra[k]
        _TRACE_METRICS_SNAPSHOT_V141 = snap

    # Ensure the timeout harness always gets a non-empty snapshot even if a hard SIGALRM
    # lands extremely early (e.g., during validation or OS scheduling jitter).
    _trace_stage("start")
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
                "csv_tested": 0,
                "csv_rejected": 0,
            },
        }

    abstraction_pressure = bool(getattr(config, "abstraction_pressure", False))
    _trace_stage("validated", abstraction_pressure=bool(abstraction_pressure))

    t0 = time.monotonic()
    bg_candidates = _infer_bg_candidates_v141(grids=[p[0] for p in train_pairs] + [test_in])
    shapes_out = _infer_shapes_out_v141(train_pairs=train_pairs)
    palette_out = _infer_palette_out_v141(train_pairs=train_pairs)
    _trace_stage("signals_inferred", t_signals_s=float(time.monotonic() - t0))
    t1 = time.monotonic()
    direct_steps = _infer_direct_steps_v141(train_pairs=train_pairs, test_in=test_in)
    _trace_stage("direct_steps_inferred", t_direct_steps_s=float(time.monotonic() - t1))
    if bool(abstraction_pressure) and bool(getattr(config, "enable_fastpath", False)):
        # Under abstraction pressure, the solver can become "concept-as-policy": primitives are
        # wrapped/gated as concept_call and must show immediate TRAIN utility. Multi-step generic
        # closures (object/bbox/mask pipelines) can unblock tasks where no single primitive is
        # TRAIN-improving at depth=1.
        #
        # We keep this list bounded and deterministic: it is a task-level candidate library, not
        # a search explosion route.
        t2 = time.monotonic()
        extra_builtin = _propose_builtin_csg_calls_v141(train_pairs=train_pairs, test_in=test_in, bg_candidates=bg_candidates)
        extra_builtin.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
        # Keep the extra library bounded so it does not crowd out existing direct inference
        # hypotheses under the existing_calls cap in abstraction-pressure gating.
        extra_builtin = extra_builtin[:48]
        _trace_stage(
            "builtin_csg_proposed",
            t_builtin_csg_s=float(time.monotonic() - t2),
            builtin_csg_count=int(len(extra_builtin)),
        )

        merged = list(direct_steps) + list(extra_builtin)
        seen: Set[str] = set()
        uniq_steps: List[ProgramStepV141] = []
        for st in merged:
            try:
                sig = canonical_json_dumps(st.to_dict())
            except Exception:
                continue
            if sig in seen:
                continue
            seen.add(sig)
            uniq_steps.append(st)
        direct_steps = uniq_steps[:96]
    want_shapes_by_pair = [tuple(int(x) for x in grid_shape_v124(out)) for _inp, out in train_pairs]
    want_palettes_by_pair = [set(int(c) for c in unique_colors_v124(out)) for _inp, out in train_pairs]
    task_feat = _task_feat_from_train_pairs_v141(train_pairs=train_pairs)
    task_feat_key_v1 = task_feat.key()
    task_feat_key_v2 = _task_feat_key_v2_from_train_pairs_v141(train_pairs=train_pairs)

    max_depth = int(config.max_depth)
    max_programs = int(config.max_programs)
    trace_program_limit = int(config.trace_program_limit)
    max_ambiguous_outputs = int(config.max_ambiguous_outputs)
    max_next_steps = int(config.max_next_steps)
    solution_cost_slack_bits = int(getattr(config, "solution_cost_slack_bits", 0) or 0)

    enable_reachability_pruning = bool(getattr(config, "enable_reachability_pruning", True))
    enable_dominated_state_pruning = bool(getattr(config, "enable_dominated_state_pruning", True))
    enable_concept_support_feat_ranking = bool(getattr(config, "enable_concept_support_feat_ranking", False))
    macro_propose_max_depth = int(getattr(config, "macro_propose_max_depth", 0) or 0)
    macro_max_templates = int(getattr(config, "macro_max_templates", 24) or 24)
    macro_max_instantiations = int(getattr(config, "macro_max_instantiations", 10) or 10)
    macro_max_branch_per_op = int(getattr(config, "macro_max_branch_per_op", 10) or 10)
    macro_try_on_fail_only = bool(getattr(config, "macro_try_on_fail_only", True))
    # CSV applicability gate behavior:
    #
    # Under abstraction pressure (concept-as-policy), we must allow a bounded set of *enabling*
    # semantic moves (slot builders / patch transforms) to enter the frontier; otherwise many tasks
    # become inexpressible at fixed depth because no loss-reducing grid edit is possible before
    # internal slots exist. This manifests as `csv_survivors == 0` and immediate "MISSING_OPERATOR".
    #
    # The allowance remains deterministic and capped by concept selection; it does not relax
    # fail-closed evaluation.
    csv_allow_slot_progress = bool(getattr(config, "csv_allow_slot_progress", False))

    # Concepts are the intended compositional substrate: allow concept_call anywhere by default.
    concept_propose_max_depth = int(getattr(config, "concept_propose_max_depth", max_depth) or max_depth)
    concept_max_templates = int(getattr(config, "concept_max_templates", 24) or 24)
    concept_max_instantiations = int(getattr(config, "concept_max_instantiations", 10) or 10)
    concept_max_branch_per_op = int(getattr(config, "concept_max_branch_per_op", 10) or 10)

    enable_trace_csg_induction = bool(getattr(config, "enable_trace_csg_induction", False))
    trace_csg_induction_first_pass_frac = float(getattr(config, "trace_csg_induction_first_pass_frac", 0.6) or 0.6)
    trace_csg_induction_max_templates = int(getattr(config, "trace_csg_induction_max_templates", 8) or 8)
    trace_csg_induction_min_inner_steps = int(getattr(config, "trace_csg_induction_min_inner_steps", 3) or 3)
    trace_csg_induction_max_inner_steps = int(getattr(config, "trace_csg_induction_max_inner_steps", 18) or 18)
    trace_csg_induction_max_loss_cells = int(getattr(config, "trace_csg_induction_max_loss_cells", 60) or 60)

    enable_point_patch_repair = bool(getattr(config, "enable_point_patch_repair", False))
    point_patch_max_points = int(getattr(config, "point_patch_max_points", 12) or 12)

    # Normalize macros (operators): cached per-process by the identity of the loaded bank.
    t3 = time.monotonic()
    raw_mt = getattr(config, "macro_templates", ()) or ()
    macro_templates_all = raw_mt if isinstance(raw_mt, (list, tuple)) else ()
    macro_templates_norm, macro_support_by_id = _get_macro_templates_norm_cached_v141(
        macro_templates_all=macro_templates_all,
        macro_max_templates=int(macro_max_templates),
    )
    _trace_stage(
        "macros_normalized",
        t_norm_macros_s=float(time.monotonic() - t3),
        macro_templates_total=int(len(macro_templates_all)),
        macro_templates_norm=int(len(macro_templates_norm)),
    )

    # Normalize concepts: cached per-process by the identity of the loaded bank.
    t4 = time.monotonic()
    raw_ct = getattr(config, "concept_templates", ()) or ()
    concept_templates_all = raw_ct if isinstance(raw_ct, (list, tuple)) else ()
    concept_templates_norm: Sequence[Dict[str, Any]] = _get_concept_templates_norm_cached_v141(
        concept_templates_all=concept_templates_all
    )
    # Under abstraction pressure, treat only explicit CSGs (with concrete `steps`) as real concepts.
    # Templates that are only `op_ids` are macro-like and reintroduce branching via per-task arg
    # (re-)instantiation; they are not allowed to dominate concept-as-policy.
    concept_templates_norm_all: Sequence[Dict[str, Any]] = concept_templates_norm
    concept_templates_norm_steps_only: Sequence[Dict[str, Any]] = tuple(
        t
        for t in concept_templates_norm_all
        if isinstance(t, dict) and isinstance(t.get("steps"), list) and bool(t.get("steps"))
    )
    if bool(abstraction_pressure):
        concept_templates_norm = concept_templates_norm_steps_only
    _trace_stage(
        "concepts_normalized",
        t_norm_concepts_s=float(time.monotonic() - t4),
        concept_templates_total=int(len(concept_templates_all)),
        concept_templates_norm=int(len(concept_templates_norm)),
        concept_templates_norm_steps_only=int(len(concept_templates_norm_steps_only)),
    )
    # NOTE: do NOT globally truncate to concept_max_templates here.
    # Selection is performed per-state in the search loop based on applicability + policy rank.
    # Global truncation can permanently drop the relevant concept family for a task, forcing
    # primitive-only exploration and causing SEARCH_BUDGET_EXCEEDED.

    # Memoization caches (pure, local).
    apply_cache: Dict[Tuple[Any, ...], StateV132] = {}
    eval_cache: Dict[str, _EvalInfoV141] = {}
    grid_hash_cache: Dict[GridV124, str] = {}
    propose_cache: Dict[Tuple[Tuple[str, int], ...], Tuple[ProgramStepV141, ...]] = {}
    metrics: Dict[str, int] = {
        "apply_cache_hits": 0,
        "apply_cache_misses": 0,
        "eval_cache_hits": 0,
        "eval_cache_misses": 0,
        "propose_cache_hits": 0,
        "propose_cache_misses": 0,
        "pruned_by_exception_prefix": 0,
        "pruned_by_next_steps_cap": 0,
        "csv_tested": 0,
        "csv_rejected": 0,
        "csv_survivors": 0,
    }

    def _csv_gate_ok(*, dl: int, ds: int, delta_max: int) -> bool:
        return _csv_gate_ok_v141(
            dl=int(dl),
            ds=int(ds),
            delta_max=int(delta_max),
            allow_slot_progress=bool(csv_allow_slot_progress),
        )

    def _avail_sig_v141(steps: Tuple[ProgramStepV141, ...]) -> Tuple[Tuple[str, int], ...]:
        avail = _abstract_slots_after_steps_v141(steps)
        keys = ["grid", "objset", "obj", "bbox", "patch", "invalid"]
        out: List[Tuple[str, int]] = []
        for k in keys:
            if k in avail:
                out.append((str(k), 1 if bool(avail.get(str(k), False)) else 0))
        return tuple(out)

    def _get_candidates_cached_v141(steps: Tuple[ProgramStepV141, ...]) -> Tuple[ProgramStepV141, ...]:
        # _propose_next_steps_v141 depends only on abstract slot availability + task-level signals
        # (train_pairs/test_in/bg/palette/shapes/direct_steps). It does not inspect concrete state.
        # Caching by availability signature massively reduces inner-loop overhead and prevents
        # harness timeouts that would otherwise discard trace_programs.
        key = _avail_sig_v141(steps)
        hit = propose_cache.get(key)
        if hit is not None:
            metrics["propose_cache_hits"] = int(metrics.get("propose_cache_hits", 0)) + 1
            return hit
        metrics["propose_cache_misses"] = int(metrics.get("propose_cache_misses", 0)) + 1
        out = tuple(
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
        propose_cache[key] = out
        return out

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

    # Fast-path (deterministic, fail-closed): when abstraction_pressure is enabled, many ARC
    # tasks are solvable by a single primitive
    # grid→grid op wrapped as a concept_call. Evaluating all such 1-step candidates avoids
    # burning max_programs on deeper search before discovering the obvious rule.
    if bool(abstraction_pressure):
        # Root eval (empty prefix) is the reference for CSV applicability at depth=1.
        # We reuse its train_final_states so "applicable" is always defined w.r.t. the current
        # prefix state (here, just the raw inputs).
        ev0 = eval_program(tuple())
        direct_op_ids: Set[str] = set()
        for st in direct_steps:
            try:
                direct_op_ids.add(str(st.op_id))
            except Exception:
                continue
        # IMPORTANT: avoid calling the full proposal engine here.
        #
        # `_get_candidates_cached_v141(tuple())` scans the learned concept bank + CSV-gates
        # candidates; on hard ARC-AGI-2 tasks this can consume most of `task_timeout_s` before
        # best-first search explores anything (programs_explored≈1), which also starves trace
        # snapshots and prevents trace-driven CSG induction.
        #
        # The fast-path is only meant to catch obvious 1-step wins; keep it strictly bounded and
        # independent from the full bank scan.
        root_steps = tuple(direct_steps)
        # NOTE: This fast-path previously evaluated *all* concept_call candidates, which could
        # dominate wall time and cause hard timeouts at programs_explored≈1. Under abstraction
        # pressure we must be hypothesis-driven: only evaluate one-step candidates that are
        # CSV-applicable (TRAIN-improving) at the root prefix.
        fast_wrappers: List[ProgramStepV141] = []
        for st in root_steps:
            op0 = str(st.op_id or "")
            if not op0:
                continue
            # Keep already-synthesized concept_call candidates (direct inference, etc.).
            if op0 == "concept_call":
                fast_wrappers.append(st)
                continue
            # macro_call alone does not satisfy fail-closed concept-as-policy; ignore at fast-path.
            if op0 == "macro_call":
                continue
            # Keep fast-path strictly bounded: only wrap a small set of truly "one-step" primitives
            # (global transforms and simple spatial/color moves). Everything else is handled by
            # the main search + trace-driven concept induction.
            if op0 not in {
                "rotate90",
                "rotate180",
                "rotate270",
                "transpose",
                "reflect_h",
                "reflect_v",
                "translate",
                "crop_bbox_nonzero",
                "pad_to",
                "cc4_color_bars",
                "cc4_color_area_column",
                "cc4_nonbg_bfs_column",
                "crop_cc4_select",
                "relational_expand",
                "uniform_line_expand",
                "replace_color",
                "map_colors",
            } and op0 not in direct_op_ids:
                continue
            od = OP_DEFS_V141.get(op0)
            if od is None or "grid" not in set(str(w) for w in od.writes):
                continue
            inner_cost = int(step_cost_bits_v141(op_id=str(op0), args=dict(st.args)))
            cid_body = {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
                "kind": "fast_prim_csg_v141",
                "step": st.to_dict(),
            }
            cid = "csg_prim_" + sha256_hex(canonical_json_dumps(cid_body).encode("utf-8"))[:16]
            fast_wrappers.append(
                ProgramStepV141(
                    op_id="concept_call",
                    args={
                        "concept_id": str(cid),
                        "cost_bits": max(1, int(inner_cost) - 1),
                        "op_ids": [str(op0)],
                        "steps": [st.to_dict()],
                    },
                )
            )

        # Built-in multi-step CSG closures (generic, deterministic): propose a bounded set of
        # object/bbox pipelines as a single concept_call to reduce depth and avoid search blowups.
        fast_wrappers.extend(
            _propose_builtin_csg_calls_v141(train_pairs=train_pairs, test_in=test_in, bg_candidates=bg_candidates)[:24]
        )

        # De-dup and prefilter cheaply (deterministic): keep low-cost, higher-compression candidates.
        uniq_fast: Dict[str, ProgramStepV141] = {}
        for w in fast_wrappers:
            try:
                sig = canonical_json_dumps(w.to_dict())
            except Exception:
                continue
            if sig in uniq_fast:
                continue
            uniq_fast[sig] = w
        fast_wrappers = list(uniq_fast.values())
        # Under strong abstraction pressure, prioritize candidates that can plausibly fix
        # structural mismatches observed on TRAIN (e.g., shape changes) before purely cosmetic
        # low-cost hypotheses. This is deterministic and task-agnostic (derived only from pairs).
        needs_shape_change = any(
            tuple(int(x) for x in grid_shape_v124(inp)) != tuple(int(x) for x in grid_shape_v124(out))
            for inp, out in train_pairs
        )
        shape_ops: Set[str] = {
            "pad_to",
            "crop_bbox_nonzero",
            "crop_cc4_select",
            "cc4_color_bars",
            "cc4_color_area_column",
            "cc4_nonbg_bfs_column",
            "relational_expand",
            "uniform_line_expand",
            "transpose",
            "rotate90",
            "rotate180",
            "rotate270",
        }

        def _shape_priority(step: ProgramStepV141) -> int:
            if not bool(needs_shape_change):
                return 1
            a0 = step.args if isinstance(step.args, dict) else {}
            payload = a0.get("steps")
            if not isinstance(payload, list):
                return 1
            for row in payload:
                try:
                    if str(row.get("op_id") or "") in shape_ops:
                        return 0
                except Exception:
                    continue
            return 1

        fast_wrappers.sort(
            key=lambda w: (
                int(_shape_priority(w)),
                int((w.args.get("cost_bits") if isinstance(w.args, dict) else 0) or 0),
                -int(len((w.args.get("steps") if isinstance(w.args, dict) and isinstance(w.args.get("steps"), list) else []) or [])),
                str(w.args.get("concept_id") if isinstance(w.args, dict) else ""),
                canonical_json_dumps(w.to_dict()),
            )
        )
        # Keep fast-path strictly bounded: under abstraction pressure we rely on the CSV gate +
        # learned CSG bank to supply hypotheses, and spending too much wall-time here can trigger
        # task_timeout_s before the main search explores anything.
        cap_fast = 256
        if bool(abstraction_pressure):
            # Under strict concept-as-policy + tight timeouts, keep fast-path extremely small so
            # it doesn't consume the whole wall clock before the main search explores anything.
            cap_fast = 32
        fast_wrappers = fast_wrappers[: int(cap_fast)]

        gated_fast: List[Tuple[int, int, int, str, ProgramStepV141]] = []
        # Hard cap: do not spend the entire task timeout in CSV gating before the main search
        # even begins. Under abstraction pressure, the learned concept bank should do most of
        # the work; the fast-path is only to catch obvious one-step wins.
        fast_tested = 0
        fast_accepted = 0
        max_fast_tested = 256
        if bool(abstraction_pressure):
            # Pre-search is a fast-path only; under strict timeouts we must not spend most of the
            # wall clock validating one-step wrappers. The main search + learned CSG bank must do
            # the heavy lifting.
            max_fast_tested = 12

        for w in fast_wrappers:
            if int(fast_tested) >= int(max_fast_tested):
                break
            a0 = w.args if isinstance(w.args, dict) else {}
            steps_payload = a0.get("steps")
            if not isinstance(steps_payload, list) or not steps_payload:
                continue
            fast_tested += 1
            metrics["csv_tested"] = int(metrics.get("csv_tested", 0)) + 1
            score = _csv_applicability_score_v141(
                steps=steps_payload,
                train_final_states=ev0.train_final_states,
                train_pairs=train_pairs,
                apply_cache=apply_cache,
                grid_hash_cache=grid_hash_cache,
                metrics=metrics,
                allow_slot_progress=bool(csv_allow_slot_progress),
            )
            ok = False
            dl = 0
            ds = 0
            if score is not None:
                dl, ds = (int(score[0]), int(score[1]))
                ok = _csv_gate_ok(dl=int(dl), ds=int(ds), delta_max=-1)
            if not bool(ok):
                metrics["csv_rejected"] = int(metrics.get("csv_rejected", 0)) + 1
                continue
            cb = a0.get("cost_bits")
            cost_bits = int(cb) if isinstance(cb, int) else 0
            cid = str(a0.get("concept_id") or "")
            gated_fast.append((int(dl), int(ds), int(cost_bits), str(cid), w))
            fast_accepted += 1

        # Evaluate only a small number of the strongest CSV-admissible hypotheses.
        gated_fast.sort(key=lambda t: (int(t[0]), int(t[1]), int(t[2]), str(t[3])))
        keep_fast = 16
        if bool(abstraction_pressure):
            keep_fast = 8
        gated_fast = gated_fast[: int(keep_fast)]

        one_step_solutions: List[Dict[str, Any]] = []
        for _dl, _ds, _cb, _cid, wrapper in gated_fast:
            ev = eval_program((wrapper,))
            if not bool(ev.ok_train):
                continue
            p = ProgramV141(steps=(wrapper,))
            psig = p.program_sig()
            cost_bits = int(_step_cost_bits_total_v141(step=wrapper, macro_support_by_id=macro_support_by_id))
            one_step_solutions.append(
                {
                    "program_sig": str(psig),
                    "cost_bits": int(cost_bits),
                    "steps": [wrapper.to_dict()],
                    "predicted_grid": [list(r) for r in ev.test_grid],
                    "predicted_grid_hash": _grid_hash_cached_v141(g=ev.test_grid, cache=grid_hash_cache),
                }
            )

        if one_step_solutions:
            min_cost = min(int(s.get("cost_bits") or 0) for s in one_step_solutions)
            minimal = [s for s in one_step_solutions if int(s.get("cost_bits") or 0) == int(min_cost)]
            outputs: Dict[str, Dict[str, Any]] = {}
            for s in minimal:
                h = str(s.get("predicted_grid_hash") or "")
                if h and h not in outputs:
                    outputs[h] = s
            if len(outputs) == 1:
                only = list(outputs.values())[0]
                return {
                    "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
                    "kind": "arc_solver_result_v141",
                    "status": "SOLVED",
                    "program_sig": str(only.get("program_sig") or ""),
                    "program_cost_bits": int(only.get("cost_bits") or 0),
                    "program_steps": list(only.get("steps") or []),
                    "predicted_grid": only.get("predicted_grid"),
                    "predicted_grid_hash": str(only.get("predicted_grid_hash") or ""),
                    "trace": {
                        "trace_programs": [],
                        "apply_cache_hits": int(metrics.get("apply_cache_hits", 0)),
                        "apply_cache_misses": int(metrics.get("apply_cache_misses", 0)),
                        "eval_cache_hits": int(metrics.get("eval_cache_hits", 0)),
                        "eval_cache_misses": int(metrics.get("eval_cache_misses", 0)),
                        "pruned_by_shape_reachability": 0,
                        "pruned_by_palette_reachability": 0,
                        "pruned_by_no_grid_modify_in_time": 0,
                        "pruned_by_dominated_state": 0,
	                        "pruned_by_exception_prefix": int(metrics.get("pruned_by_exception_prefix", 0)),
	                        "pruned_by_next_steps_cap": int(metrics.get("pruned_by_next_steps_cap", 0)),
	                        "csv_tested": int(metrics.get("csv_tested", 0)),
	                        "csv_rejected": int(metrics.get("csv_rejected", 0)),
	                        "csv_survivors": int(metrics.get("csv_survivors", 0)),
	                    },
	                }

            outs_sorted = sorted(outputs.values(), key=lambda x: str(x.get("predicted_grid_hash") or ""))
            return {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
                "kind": "arc_solver_result_v141",
                "status": "UNKNOWN",
                "predicted_grids": [o.get("predicted_grid") for o in outs_sorted],
                "candidate_programs": [
                    {
                        "program_sig": str(o.get("program_sig") or ""),
                        "program_cost_bits": int(o.get("cost_bits") or 0),
                        "program_steps": list(o.get("steps") or []),
                        "predicted_grid": o.get("predicted_grid"),
                        "predicted_grid_hash": str(o.get("predicted_grid_hash") or ""),
                    }
                    for o in outs_sorted
                ],
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
                    "trace_programs": [],
                    "apply_cache_hits": int(metrics.get("apply_cache_hits", 0)),
                    "apply_cache_misses": int(metrics.get("apply_cache_misses", 0)),
                    "eval_cache_hits": int(metrics.get("eval_cache_hits", 0)),
                    "eval_cache_misses": int(metrics.get("eval_cache_misses", 0)),
                    "pruned_by_shape_reachability": 0,
                    "pruned_by_palette_reachability": 0,
                    "pruned_by_no_grid_modify_in_time": 0,
                    "pruned_by_dominated_state": 0,
	                    "pruned_by_exception_prefix": int(metrics.get("pruned_by_exception_prefix", 0)),
	                    "pruned_by_next_steps_cap": int(metrics.get("pruned_by_next_steps_cap", 0)),
	                    "csv_tested": int(metrics.get("csv_tested", 0)),
	                    "csv_rejected": int(metrics.get("csv_rejected", 0)),
	                    "csv_survivors": int(metrics.get("csv_survivors", 0)),
	                },
	            }

        # Fast-path templates (deterministic, bounded): attempt a small library of generic
        # multi-step programs wrapped as a single concept_call. This reduces branching for
        # common object- and bbox-centric tasks (borders, fills, crops, outlines) that are
        # otherwise discovered late in full search and frequently hit SEARCH_BUDGET_EXCEEDED.
        #
        # Templates are generic and parameterized only by observed palettes/shapes; there is
        # no per-task branching and all candidates must match ALL train pairs to be considered.
        palette_in: Set[int] = set()
        for g in [p[0] for p in train_pairs] + [test_in]:
            for row in g:
                for v in row:
                    palette_in.add(int(v))
        palette_all_s = sorted(set(int(c) for c in list(palette_out) + list(palette_in)))[:10]
        palette_out_s = list(palette_out[:6]) if palette_out else palette_all_s[:6]

        def _wrap_concept_template(inner: Sequence[ProgramStepV141]) -> ProgramStepV141:
            inner_steps = [s for s in inner if isinstance(s, ProgramStepV141)]
            steps_dicts = [s.to_dict() for s in inner_steps]
            inner_cost = 0
            op_ids: List[str] = []
            for s in inner_steps:
                op0 = str(s.op_id or "")
                if not op0 or op0 in {"macro_call", "concept_call"}:
                    continue
                op_ids.append(str(op0))
                inner_cost += int(step_cost_bits_v141(op_id=str(op0), args=dict(s.args)))
            bonus = min(6, _floor_log2_int(1 + int(len(inner_steps))))
            cost_bits = max(1, int(inner_cost) - int(bonus))
            sig = sha256_hex(canonical_json_dumps(steps_dicts).encode("utf-8"))[:16]
            return ProgramStepV141(
                op_id="concept_call",
                args={
                    "concept_id": "csg_tpl_" + str(sig),
                    "cost_bits": int(cost_bits),
                    "op_ids": list(op_ids),
                    "steps": steps_dicts,
                },
            )

        tpl_wrappers: List[ProgramStepV141] = []

        # bbox_by_color templates: border/fill/crop
        for c in palette_all_s:
            st_bbox = ProgramStepV141(op_id="bbox_by_color", args={"color": int(c)})
            tpl_wrappers.append(
                _wrap_concept_template(
                    [st_bbox, ProgramStepV141(op_id="crop_bbox", args={}), ProgramStepV141(op_id="commit_patch", args={})]
                )
            )
            for oc in palette_out_s:
                tpl_wrappers.append(
                    _wrap_concept_template(
                        [st_bbox, ProgramStepV141(op_id="draw_rect_border", args={"color": int(oc), "thickness": 1})]
                    )
                )
                tpl_wrappers.append(_wrap_concept_template([st_bbox, ProgramStepV141(op_id="paint_rect", args={"color": int(oc)})]))

        # Object-centric templates via cc4 + select_obj
        from .arc_solver_v134 import _infer_select_obj_args_v134

        for bg in (bg_candidates[:2] if bg_candidates else [0]):
            args_list = _infer_select_obj_args_v134(train_pairs=train_pairs, bg=int(bg), max_rank=2)
            if not args_list:
                args_list = [
                    {"key": "area", "order": "max", "rank": 0, "color_filter": None},
                    {"key": "area", "order": "min", "rank": 0, "color_filter": None},
                    {"key": "bbox_area", "order": "max", "rank": 0, "color_filter": None},
                    {"key": "bbox_area", "order": "min", "rank": 0, "color_filter": None},
                    {"key": "top", "order": "min", "rank": 0, "color_filter": None},
                    {"key": "left", "order": "min", "rank": 0, "color_filter": None},
                ]
            args_list = args_list[:12]
            st_cc4 = ProgramStepV141(op_id="cc4", args={"bg": int(bg)})
            for a in args_list:
                st_sel = ProgramStepV141(op_id="select_obj", args=dict(a))
                tpl_wrappers.append(
                    _wrap_concept_template(
                        [st_cc4, st_sel, ProgramStepV141(op_id="obj_patch", args={"bg": int(bg)}), ProgramStepV141(op_id="commit_patch", args={})]
                    )
                )
                for oc in palette_out_s:
                    tpl_wrappers.append(
                        _wrap_concept_template(
                            [
                                st_cc4,
                                st_sel,
                                ProgramStepV141(op_id="obj_bbox", args={}),
                                ProgramStepV141(op_id="draw_rect_border", args={"color": int(oc), "thickness": 1}),
                            ]
                        )
                    )
                    tpl_wrappers.append(
                        _wrap_concept_template(
                            [st_cc4, st_sel, ProgramStepV141(op_id="obj_bbox", args={}), ProgramStepV141(op_id="paint_rect", args={"color": int(oc)})]
                        )
                    )

        # Mask/outline templates: outline non-bg or a specific color, then paint.
        for bg in (bg_candidates[:2] if bg_candidates else [0]):
            st_mask_nb = ProgramStepV141(op_id="mask_nonbg", args={"bg": int(bg)})
            for oc in palette_out_s:
                tpl_wrappers.append(
                    _wrap_concept_template(
                        [st_mask_nb, ProgramStepV141(op_id="mask_outline", args={}), ProgramStepV141(op_id="paint_mask", args={"color": int(oc)})]
                    )
                )
        for c in palette_all_s[:8]:
            st_mask_c = ProgramStepV141(op_id="mask_by_color", args={"color": int(c)})
            for oc in palette_out_s:
                tpl_wrappers.append(
                    _wrap_concept_template(
                        [st_mask_c, ProgramStepV141(op_id="mask_outline", args={}), ProgramStepV141(op_id="paint_mask", args={"color": int(oc)})]
                    )
                )

        # Tile/Embed templates: repeat the whole grid (repeat_grid mode=grid) and embed into the
        # target canvas via pad_to + translate (top/center/bottom × left/center/right).
        #
        # This captures common ARC patterns where a small input tile is repeated along one axis
        # (or both) and placed inside a larger background canvas. Without a bounded deterministic
        # fast path, these are often discovered late in full search and hit SEARCH_BUDGET_EXCEEDED.
        train_in_shapes = sorted(set(tuple(int(x) for x in grid_shape_v124(inp)) for inp, _out in train_pairs))
        train_out_shapes = sorted(set(tuple(int(x) for x in grid_shape_v124(out)) for _inp, out in train_pairs))
        pad_colors: List[int] = sorted(set([0] + [int(b) for b in bg_candidates] + [int(c) for c in palette_out]))[:4]
        for (hi, wi) in train_in_shapes[:2]:
            if int(hi) <= 0 or int(wi) <= 0:
                continue
            for (ho, wo) in train_out_shapes[:2]:
                if int(ho) <= 0 or int(wo) <= 0:
                    continue
                for ry in (1, 2, 3):
                    for rx in (1, 2, 3):
                        if int(ry) == 1 and int(rx) == 1:
                            continue
                        hh = int(hi) * int(ry)
                        ww = int(wi) * int(rx)
                        if hh > int(ho) or ww > int(wo):
                            continue
                        gap_y = int(ho - hh)
                        gap_x = int(wo - ww)
                        dy_opts: List[int] = [0, int(gap_y)]
                        dx_opts: List[int] = [0, int(gap_x)]
                        if gap_y % 2 == 0:
                            dy_opts.append(int(gap_y // 2))
                        if gap_x % 2 == 0:
                            dx_opts.append(int(gap_x // 2))
                        st_tile = ProgramStepV141(op_id="repeat_grid", args={"mode": "grid", "ry": int(ry), "rx": int(rx)})
                        for pad in pad_colors:
                            st_pad = ProgramStepV141(op_id="pad_to", args={"height": int(ho), "width": int(wo), "pad": int(pad)})
                            for dy in sorted(set(int(x) for x in dy_opts)):
                                for dx in sorted(set(int(x) for x in dx_opts)):
                                    st_tr = ProgramStepV141(op_id="translate", args={"dx": int(dx), "dy": int(dy), "pad": int(pad)})
                                    tpl_wrappers.append(_wrap_concept_template([st_tile, st_pad, st_tr]))

        # Dedup and cap templates to keep overhead bounded.
        tpl_wrappers.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
        seen_tpl: Set[str] = set()
        uniq_tpl: List[ProgramStepV141] = []
        for w in tpl_wrappers:
            sig = sha256_hex(canonical_json_dumps(w.to_dict()).encode("utf-8"))
            if sig in seen_tpl:
                continue
            seen_tpl.add(sig)
            uniq_tpl.append(w)
        # Keep the fast-path library tightly bounded to avoid brute-force scanning dominating
        # wall time on hard tasks (which causes timeouts and prevents trace-driven induction).
        fast_k = min(96, int(max_next_steps))
        if bool(abstraction_pressure):
            # Under strict timeouts, evaluating many multi-step templates can consume ~all wall time.
            # This stage is only to catch obvious wins; the main search + induced CSGs must do the rest.
            fast_k = min(int(fast_k), 8)
        else:
            fast_k = max(32, int(fast_k))
        uniq_tpl = uniq_tpl[: int(fast_k)]

        multi_step_solutions: List[Dict[str, Any]] = []
        for w in uniq_tpl:
            # CSV v1 gate (root prefix): if a candidate doesn't strictly reduce TRAIN loss, do not
            # spend full eval_program() on it. This prevents pre-search from dominating timeouts.
            if bool(abstraction_pressure):
                a0 = w.args if isinstance(w.args, dict) else {}
                steps_payload = a0.get("steps")
                if isinstance(steps_payload, list) and steps_payload:
                    metrics["csv_tested"] = int(metrics.get("csv_tested", 0)) + 1
                    ok_csv = _csv_applicable_v141(
                        steps=steps_payload,
                        train_final_states=ev0.train_final_states,
                        train_pairs=train_pairs,
                        apply_cache=apply_cache,
                        grid_hash_cache=grid_hash_cache,
                        metrics=metrics,
                        max_loss_delta=-1,
                        allow_slot_progress=bool(csv_allow_slot_progress),
                    )
                    if not bool(ok_csv):
                        metrics["csv_rejected"] = int(metrics.get("csv_rejected", 0)) + 1
                        continue
                    metrics["csv_survivors"] = int(metrics.get("csv_survivors", 0)) + 1
            ev = eval_program((w,))
            if not bool(ev.ok_train):
                continue
            p = ProgramV141(steps=(w,))
            psig = p.program_sig()
            multi_step_solutions.append(
                {
                    "program_sig": str(psig),
                    "cost_bits": int(w.args.get("cost_bits") or 0),
                    "steps": [w.to_dict()],
                    "predicted_grid": [list(r) for r in ev.test_grid],
                    "predicted_grid_hash": _grid_hash_cached_v141(g=ev.test_grid, cache=grid_hash_cache),
                }
            )

        if multi_step_solutions:
            min_cost = min(int(s.get("cost_bits") or 0) for s in multi_step_solutions)
            minimal = [s for s in multi_step_solutions if int(s.get("cost_bits") or 0) == int(min_cost)]
            outputs: Dict[str, Dict[str, Any]] = {}
            for s in minimal:
                h = str(s.get("predicted_grid_hash") or "")
                if h and h not in outputs:
                    outputs[h] = s
            if len(outputs) == 1:
                only = list(outputs.values())[0]
                return {
                    "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
                    "kind": "arc_solver_result_v141",
                    "status": "SOLVED",
                    "program_sig": str(only.get("program_sig") or ""),
                    "program_cost_bits": int(only.get("cost_bits") or 0),
                    "program_steps": list(only.get("steps") or []),
                    "predicted_grid": only.get("predicted_grid"),
                    "predicted_grid_hash": str(only.get("predicted_grid_hash") or ""),
                    "trace": {
                        "trace_programs": [],
                        "apply_cache_hits": int(metrics.get("apply_cache_hits", 0)),
                        "apply_cache_misses": int(metrics.get("apply_cache_misses", 0)),
                        "eval_cache_hits": int(metrics.get("eval_cache_hits", 0)),
                        "eval_cache_misses": int(metrics.get("eval_cache_misses", 0)),
                        "pruned_by_shape_reachability": 0,
                        "pruned_by_palette_reachability": 0,
                        "pruned_by_no_grid_modify_in_time": 0,
                        "pruned_by_dominated_state": 0,
	                        "pruned_by_exception_prefix": int(metrics.get("pruned_by_exception_prefix", 0)),
	                        "pruned_by_next_steps_cap": int(metrics.get("pruned_by_next_steps_cap", 0)),
	                        "csv_tested": int(metrics.get("csv_tested", 0)),
	                        "csv_rejected": int(metrics.get("csv_rejected", 0)),
	                        "csv_survivors": int(metrics.get("csv_survivors", 0)),
	                    },
	                }

            outs_sorted = sorted(outputs.values(), key=lambda x: str(x.get("predicted_grid_hash") or ""))
            return {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
                "kind": "arc_solver_result_v141",
                "status": "UNKNOWN",
                "predicted_grids": [o.get("predicted_grid") for o in outs_sorted],
                "candidate_programs": [
                    {
                        "program_sig": str(o.get("program_sig") or ""),
                        "program_cost_bits": int(o.get("cost_bits") or 0),
                        "program_steps": list(o.get("steps") or []),
                        "predicted_grid": o.get("predicted_grid"),
                        "predicted_grid_hash": str(o.get("predicted_grid_hash") or ""),
                    }
                    for o in outs_sorted
                ],
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
                    "trace_programs": [],
                    "apply_cache_hits": int(metrics.get("apply_cache_hits", 0)),
                    "apply_cache_misses": int(metrics.get("apply_cache_misses", 0)),
                    "eval_cache_hits": int(metrics.get("eval_cache_hits", 0)),
                    "eval_cache_misses": int(metrics.get("eval_cache_misses", 0)),
                    "pruned_by_shape_reachability": 0,
                    "pruned_by_palette_reachability": 0,
                    "pruned_by_no_grid_modify_in_time": 0,
                    "pruned_by_dominated_state": 0,
	                    "pruned_by_exception_prefix": int(metrics.get("pruned_by_exception_prefix", 0)),
	                    "pruned_by_next_steps_cap": int(metrics.get("pruned_by_next_steps_cap", 0)),
	                    "csv_tested": int(metrics.get("csv_tested", 0)),
	                    "csv_rejected": int(metrics.get("csv_rejected", 0)),
	                    "csv_survivors": int(metrics.get("csv_survivors", 0)),
	                },
	            }

        # Bounded 2-step fast path (generic, deterministic): many ARC tasks require a small
        # composition of global/shape transforms (e.g., pad+rotate). When external banks
        # are loaded, full search can hit per-task timeouts before discovering these simple
        # compositions due to branching. Here we try a small, fixed budget of 2-step
        # concept_call programs before entering full best-first search.
        prim_wrappers: List[ProgramStepV141] = []
        for st in root_steps:
            op0 = str(st.op_id or "")
            if not op0 or op0 in {"macro_call", "concept_call"}:
                continue
            od = OP_DEFS_V141.get(op0)
            if od is None:
                continue
            if set(str(r) for r in od.reads) != {"grid"}:
                continue
            if set(str(w) for w in od.writes) != {"grid"}:
                continue
            inner_cost = int(step_cost_bits_v141(op_id=str(op0), args=dict(st.args)))
            prim_wrappers.append(
                ProgramStepV141(
                    op_id="concept_call",
                    args={
                        "concept_id": "csg_prim_" + str(op0),
                        "cost_bits": max(1, int(inner_cost) - 1),
                        "op_ids": [str(op0)],
                        "steps": [st.to_dict()],
                    },
                )
            )
        prim_wrappers.sort(
            key=lambda s: (
                int(s.args.get("cost_bits") or 0),
                canonical_json_dumps(s.to_dict()),
            )
        )
        tpl_cands: List[ProgramStepV141] = []
        if "uniq_tpl" in locals():
            try:
                tpl_cands = list(uniq_tpl)
            except Exception:
                tpl_cands = []
        tpl_cands.sort(
            key=lambda s: (
                int(s.args.get("cost_bits") or 0),
                canonical_json_dumps(s.to_dict()),
            )
        )

        # Keep the candidate set small and deterministic.
        #
        # Under abstraction pressure + strict timeouts, "proposal-time" can dominate the wall clock.
        # Prefer a smaller outer set to ensure the main best-first pass actually gets to explore.
        cand_outer: List[ProgramStepV141] = []
        prim_cap = 40
        if bool(abstraction_pressure):
            prim_cap = 12
        cand_outer.extend(prim_wrappers[: int(prim_cap)])
        builtin_cands = _propose_builtin_csg_calls_v141(train_pairs=train_pairs, test_in=test_in, bg_candidates=bg_candidates)
        cand_outer.extend(builtin_cands[:24])
        cand_outer.extend(tpl_cands[:20])
        # De-dup by full step signature.
        seen_outer: Set[str] = set()
        uniq_outer: List[ProgramStepV141] = []
        for s in cand_outer:
            sig = sha256_hex(canonical_json_dumps(s.to_dict()).encode("utf-8"))
            if sig in seen_outer:
                continue
            seen_outer.add(sig)
            uniq_outer.append(s)

        # Under abstraction pressure, we still want CSV as policy, but for short-horizon composition
        # we must allow a small set of *enabling* moves that may not individually reduce TRAIN loss.
        #
        # Otherwise, necessary 2-step concepts like rotate→translate (where rotate alone does not
        # improve loss) can never be discovered or applied as an atomic closure.
        if bool(abstraction_pressure):
            strict: List[Tuple[int, int, int, str, ProgramStepV141]] = []
            enable: List[Tuple[int, int, int, str, ProgramStepV141]] = []

            enabling_ops: Set[str] = {
                # global transforms / spatial alignment
                "transpose",
                "rotate90",
                "rotate180",
                "rotate270",
                "translate",
                "gravity",
                # shape adjust
                "pad_to",
                "crop_bbox_nonzero",
                "crop_bbox_by_color",
                "crop_bbox",
                "repeat_grid",
                "downsample_mode",
                # patch pipeline components
                "cc4",
                "cc4_nonbg_multicolor",
                "select_obj",
                "select_obj_by_color",
                "obj_patch",
                "commit_patch",
                "paste",
                "patch_rotate90",
                "patch_rotate180",
                "patch_rotate270",
                "patch_reflect_h",
                "patch_reflect_v",
                "patch_transpose",
                "patch_translate",
            }

            def _primary_op(step: ProgramStepV141) -> str:
                a0 = step.args if isinstance(step.args, dict) else {}
                steps_payload = a0.get("steps")
                if isinstance(steps_payload, list) and steps_payload and isinstance(steps_payload[0], dict):
                    return str(steps_payload[0].get("op_id") or "")
                op_ids = a0.get("op_ids")
                if isinstance(op_ids, list):
                    for x in op_ids:
                        if str(x):
                            return str(x)
                return str(step.op_id or "")

            for s in uniq_outer:
                a0 = s.args if isinstance(s.args, dict) else {}
                steps_payload = a0.get("steps")
                if not isinstance(steps_payload, list) or not steps_payload:
                    continue

                metrics["csv_tested"] = int(metrics.get("csv_tested", 0)) + 1
                score = _csv_applicability_score_v141(
                    steps=steps_payload,
                    train_final_states=ev0.train_final_states,
                    train_pairs=train_pairs,
                    apply_cache=apply_cache,
                    grid_hash_cache=grid_hash_cache,
                    metrics=metrics,
                    allow_slot_progress=bool(csv_allow_slot_progress),
                )
                ok = False
                dl = 0
                ds = 0
                if score is not None:
                    dl, ds = (int(score[0]), int(score[1]))
                    ok = _csv_gate_ok(dl=int(dl), ds=int(ds), delta_max=-1)
                cb = a0.get("cost_bits")
                cost_bits = int(cb) if isinstance(cb, int) else 0
                cid = str(a0.get("concept_id") or "")
                if bool(ok):
                    strict.append((int(dl), int(ds), int(cost_bits), str(cid), s))
                else:
                    # Keep only a tiny set of enabling ops for composition search.
                    # These are allowed even when not TRAIN-improving as single steps.
                    if _primary_op(s) in enabling_ops:
                        enable.append((int(dl), int(ds), int(cost_bits), str(cid), s))
                    else:
                        metrics["csv_rejected"] = int(metrics.get("csv_rejected", 0)) + 1

            strict.sort(key=lambda t: (int(t[0]), int(t[1]), int(t[2]), str(t[3])))
            enable.sort(key=lambda t: (int(t[2]), str(t[3])))

            # Keep the 2-step fast-path strictly bounded and deterministic.
            strict_cap = 10
            enable_cap = 6
            chosen = [s for _dl, _ds, _cb, _cid, s in strict[: int(strict_cap)]] + [s for _dl, _ds, _cb, _cid, s in enable[: int(enable_cap)]]
            # de-dup (stable)
            seen_sig: Set[str] = set()
            uniq_outer2: List[ProgramStepV141] = []
            for s in chosen:
                sig = sha256_hex(canonical_json_dumps(s.to_dict()).encode("utf-8"))
                if sig in seen_sig:
                    continue
                seen_sig.add(sig)
                uniq_outer2.append(s)
            uniq_outer = uniq_outer2

        two_step_solutions: List[Dict[str, Any]] = []
        # Brute forcing 1200 evals here can consume the entire per-task timeout and prevent
        # the main best-first search from exploring anything. Keep this small; any missing
        # multi-step pattern should be learned as a reusable CSG, not searched here.
        # Under abstraction pressure, keep this very small: any missing 2-step pattern should
        # be learned as a reusable CSG, not brute-forced here.
        # Under abstraction pressure we need enough room to discover atomic 2-step closures
        # (rotate→translate, crop→pad, etc.) without degenerating into brute force.
        two_step_budget = 120 if bool(abstraction_pressure) else 1200
        tried = 0
        for a in uniq_outer:
            for b in uniq_outer:
                tried += 1
                if tried > int(two_step_budget):
                    break
                ev = eval_program((a, b))
                if not bool(ev.ok_train):
                    continue
                p = ProgramV141(steps=(a, b))
                psig = p.program_sig()
                cost_bits = int(_step_cost_bits_total_v141(step=a, macro_support_by_id=macro_support_by_id)) + int(
                    _step_cost_bits_total_v141(step=b, macro_support_by_id=macro_support_by_id)
                )
                two_step_solutions.append(
                    {
                        "program_sig": str(psig),
                        "cost_bits": int(cost_bits),
                        "steps": [a.to_dict(), b.to_dict()],
                        "predicted_grid": [list(r) for r in ev.test_grid],
                        "predicted_grid_hash": _grid_hash_cached_v141(g=ev.test_grid, cache=grid_hash_cache),
                    }
                )
            if tried > int(two_step_budget):
                break

        if two_step_solutions:
            min_cost = min(int(s.get("cost_bits") or 0) for s in two_step_solutions)
            minimal = [s for s in two_step_solutions if int(s.get("cost_bits") or 0) == int(min_cost)]
            outputs2: Dict[str, Dict[str, Any]] = {}
            for s in minimal:
                h = str(s.get("predicted_grid_hash") or "")
                if h and h not in outputs2:
                    outputs2[h] = s
            if len(outputs2) == 1:
                only = list(outputs2.values())[0]
                return {
                    "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
                    "kind": "arc_solver_result_v141",
                    "status": "SOLVED",
                    "program_sig": str(only.get("program_sig") or ""),
                    "program_cost_bits": int(only.get("cost_bits") or 0),
                    "program_steps": list(only.get("steps") or []),
                    "predicted_grid": only.get("predicted_grid"),
                    "predicted_grid_hash": str(only.get("predicted_grid_hash") or ""),
                        "trace": {
                            "trace_programs": [],
                            "apply_cache_hits": int(metrics.get("apply_cache_hits", 0)),
                            "apply_cache_misses": int(metrics.get("apply_cache_misses", 0)),
                            "eval_cache_hits": int(metrics.get("eval_cache_hits", 0)),
                            "eval_cache_misses": int(metrics.get("eval_cache_misses", 0)),
                            "pruned_by_shape_reachability": 0,
                            "pruned_by_palette_reachability": 0,
                            "pruned_by_no_grid_modify_in_time": 0,
                            "pruned_by_dominated_state": 0,
	                            "pruned_by_exception_prefix": int(metrics.get("pruned_by_exception_prefix", 0)),
	                            "pruned_by_next_steps_cap": int(metrics.get("pruned_by_next_steps_cap", 0)),
	                            "csv_tested": int(metrics.get("csv_tested", 0)),
	                            "csv_rejected": int(metrics.get("csv_rejected", 0)),
	                            "csv_survivors": int(metrics.get("csv_survivors", 0)),
	                        },
	                    }

            outs_sorted = sorted(outputs2.values(), key=lambda x: str(x.get("predicted_grid_hash") or ""))
            return {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
                "kind": "arc_solver_result_v141",
                "status": "UNKNOWN",
                "predicted_grids": [o.get("predicted_grid") for o in outs_sorted],
                "candidate_programs": [
                    {
                        "program_sig": str(o.get("program_sig") or ""),
                        "program_cost_bits": int(o.get("cost_bits") or 0),
                        "program_steps": list(o.get("steps") or []),
                        "predicted_grid": o.get("predicted_grid"),
                        "predicted_grid_hash": str(o.get("predicted_grid_hash") or ""),
                    }
                    for o in outs_sorted
                ],
                "predicted_grids_by_solution": [
                    {
                        "program_sig": str(o.get("program_sig") or ""),
                        "program_cost_bits": int(o.get("cost_bits") or 0),
                        "predicted_grid_hash": str(o.get("predicted_grid_hash") or ""),
                    }
                    for o in outs_sorted
                ],
                "failure_reason": {"kind": "AMBIGUOUS_RULE", "details": {"solutions": int(len(outputs2))}},
                    "trace": {
                        "trace_programs": [],
                        "apply_cache_hits": int(metrics.get("apply_cache_hits", 0)),
                        "apply_cache_misses": int(metrics.get("apply_cache_misses", 0)),
                        "eval_cache_hits": int(metrics.get("eval_cache_hits", 0)),
                        "eval_cache_misses": int(metrics.get("eval_cache_misses", 0)),
                        "pruned_by_shape_reachability": 0,
                        "pruned_by_palette_reachability": 0,
                        "pruned_by_no_grid_modify_in_time": 0,
                        "pruned_by_dominated_state": 0,
	                        "pruned_by_exception_prefix": int(metrics.get("pruned_by_exception_prefix", 0)),
	                        "pruned_by_next_steps_cap": int(metrics.get("pruned_by_next_steps_cap", 0)),
	                        "csv_tested": int(metrics.get("csv_tested", 0)),
	                        "csv_rejected": int(metrics.get("csv_rejected", 0)),
	                        "csv_survivors": int(metrics.get("csv_survivors", 0)),
	                    },
	                }

    def _solve_pass(*, allow_macros: bool, max_programs_limit: Optional[int] = None) -> Dict[str, Any]:
        # Best-first over train mismatch, but with a generic one-step lookahead
        # for states that carry extra semantic slots (e.g., patch-as-mask).
        #
        # Priority tuple:
        #   (best_loss_shape, best_loss_cells, slot_penalty_total, loss_shape, loss_cells, cost_bits, depth, program_sig)
        #
        # `slot_penalty_total` breaks ties among equally-good loss states by preferring programs that
        # have constructed useful semantic slots (objset/obj/bbox/patch). This is critical under
        # abstraction pressure: many tasks require *enabling* semantic moves (dl==0, ds<0) before
        # any grid-writing action can reduce mismatch. Without this tie-breaker, the best-first
        # search can waste most of its budget chasing tiny loss reductions that do not unlock the
        # correct concept pipeline, leading to SEARCH_BUDGET_EXCEEDED.
        frontier: List[Tuple[Tuple[int, int, int, int, int, int, int, str], Tuple[ProgramStepV141, ...], str]] = []
        seen_programs: Set[str] = set()
        best_cost_by_vec_sig: Dict[str, int] = {}
        lookahead_loss_by_vec_sig: Dict[str, Tuple[int, int]] = {}
        concept_template_select_cache: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        concept_csg_resolve_cache: Dict[str, Optional[List[Dict[str, Any]]]] = {}
        concept_support_task_feat_cache: Dict[str, Tuple[Tuple[str, ...], ...]] = {}
        # Cache for CSV applicability decisions (vec_sig × candidate signature).
        # Value: (ok, delta_loss_total, delta_slot_total)
        csv_applicable_cache: Dict[Tuple[str, str], Tuple[bool, int, int]] = {}

        def _lookahead_grid_modify_loss(*, ev: _EvalInfoV141, depth: int) -> Optional[Tuple[int, int]]:
            if int(depth) >= int(max_depth):
                return None
            cached = lookahead_loss_by_vec_sig.get(str(ev.vec_sig))
            if cached is not None:
                return cached
            # Only compute lookahead when a semantic slot exists that can be committed to the grid.
            # This is generic and deterministic; it does not depend on task IDs or dataset priors.
            any_patch = any(st.patch is not None for st in ev.train_final_states)
            if not any_patch:
                return None

            candidates: List[ProgramStepV141] = [ProgramStepV141(op_id="commit_patch", args={})]
            for c in palette_out[:6]:
                candidates.append(ProgramStepV141(op_id="paint_mask", args={"color": int(c)}))

            best = (10**9, 10**9)
            for cand in candidates:
                ok_train = True
                loss_shape = 0
                loss_cells = 0
                for st0, (_inp, want) in zip(ev.train_final_states, train_pairs):
                    try:
                        st1 = _apply_step_cached_v141(
                            state=st0,
                            op_id=str(cand.op_id),
                            args=dict(cand.args),
                            apply_cache=apply_cache,
                            grid_hash_cache=grid_hash_cache,
                            metrics=metrics,
                        )
                        got = st1.grid
                    except Exception:
                        ok_train = False
                        loss_shape = 10**9
                        loss_cells = 10**9
                        break
                    if not grid_equal_v124(got, want):
                        ok_train = False
                        mm = _summarize_mismatch_v141(got=got, want=want)
                        if mm.get("kind") == "shape_mismatch":
                            loss_shape += 1
                        elif mm.get("kind") == "cell_mismatch":
                            loss_cells += int(mm.get("diff_cells") or 0)
                cand_loss = (int(loss_shape), int(loss_cells))
                if cand_loss < best:
                    best = cand_loss
                if ok_train:
                    best = (0, 0)
                    break

            if best[0] >= 10**9:
                return None
            lookahead_loss_by_vec_sig[str(ev.vec_sig)] = (int(best[0]), int(best[1]))
            return (int(best[0]), int(best[1]))

        def push(steps: Tuple[ProgramStepV141, ...]) -> None:
            p = ProgramV141(steps=steps)
            sig = p.program_sig()
            if sig in seen_programs:
                return
            seen_programs.add(sig)
            # MDL cost (meta-ops handled explicitly; base ops via step_cost_bits_v141).
            cost_bits = 0
            for st in steps:
                cost_bits += int(_step_cost_bits_total_v141(step=st, macro_support_by_id=macro_support_by_id))
            ev = eval_program(steps)
            loss_shape, loss_cells = ev.loss
            best_loss = (int(loss_shape), int(loss_cells))
            la = _lookahead_grid_modify_loss(ev=ev, depth=int(len(steps)))
            if la is not None and tuple(int(x) for x in la) < best_loss:
                best_loss = tuple(int(x) for x in la)
            slot_penalty_total = 0
            for st0 in ev.train_final_states:
                slot_penalty_total += int(
                    (1 if st0.objset is None else 0)
                    + (1 if st0.obj is None else 0)
                    + (1 if st0.bbox is None else 0)
                    + (1 if st0.patch is None else 0)
                )
            key = (
                int(best_loss[0]),
                int(best_loss[1]),
                int(slot_penalty_total),
                int(loss_shape),
                int(loss_cells),
                int(cost_bits),
                int(len(steps)),
                str(sig),
            )
            heapq.heappush(frontier, (key, steps, sig))

        _trace_stage("enter_search", t_pre_search_s=float(time.monotonic() - t_start))
        push(tuple())

        trace_programs: List[Dict[str, Any]] = []
        # Retain a diversified set of trace programs for mining: keep best-K per depth.
        #
        # Rationale: with abstraction pressure, the solver often finds very good 1-step near-misses
        # (low loss) that can crowd out deeper traces. Mining needs multi-step evidence to induce
        # reusable closures that actually reduce depth/branching.
        trace_items_by_depth: Dict[int, List[Tuple[Tuple[int, int, int, int, str], Dict[str, Any]]]] = {}
        trace_per_depth = 0
        if int(trace_program_limit) > 0:
            trace_per_depth = max(1, int(trace_program_limit) // max(1, int(max_depth) + 1))

        def _snapshot_trace_programs_v141() -> None:
            # Keep deterministic ordering (same as final trace_programs build).
            # Also snapshot key metrics so the harness can report causal gates even on hard timeouts.
            global _TRACE_METRICS_SNAPSHOT_V141, _TRACE_PROGRAMS_SNAPSHOT_V141
            items: List[Tuple[Tuple[int, int, int, int, str], Dict[str, Any]]] = []
            for bucket in trace_items_by_depth.values():
                items.extend(bucket)
            items.sort(key=lambda kv: kv[0])
            _TRACE_PROGRAMS_SNAPSHOT_V141 = [dict(row) for _k, row in items]
            snap: Dict[str, Any] = {}
            try:
                if isinstance(_TRACE_METRICS_SNAPSHOT_V141, dict):
                    snap = dict(_TRACE_METRICS_SNAPSHOT_V141)
            except Exception:
                snap = {}
            snap.update(
                {
                "programs_explored": int(programs_explored),
                # Keep wall-clock progress fresh even when the solver is interrupted mid-iteration.
                # The harness reads this snapshot on SIGALRM timeouts.
                "t_total_s": float(time.monotonic() - t_start),
                "apply_cache_hits": int(metrics.get("apply_cache_hits", 0)),
                "apply_cache_misses": int(metrics.get("apply_cache_misses", 0)),
                "eval_cache_hits": int(metrics.get("eval_cache_hits", 0)),
                "eval_cache_misses": int(metrics.get("eval_cache_misses", 0)),
                "propose_cache_hits": int(metrics.get("propose_cache_hits", 0)),
                "propose_cache_misses": int(metrics.get("propose_cache_misses", 0)),
                "pruned_by_shape_reachability": int(pruned_shape),
                "pruned_by_palette_reachability": int(pruned_palette),
                "pruned_by_no_grid_modify_in_time": int(pruned_no_grid_modify),
                "pruned_by_dominated_state": int(pruned_by_dominated_state),
	                "pruned_by_exception_prefix": int(metrics.get("pruned_by_exception_prefix", 0)),
	                "pruned_by_next_steps_cap": int(metrics.get("pruned_by_next_steps_cap", 0)),
	                "csv_tested": int(metrics.get("csv_tested", 0)),
	                "csv_rejected": int(metrics.get("csv_rejected", 0)),
	                "csv_survivors": int(metrics.get("csv_survivors", 0)),
	                }
	            )
            _TRACE_METRICS_SNAPSHOT_V141 = snap

        programs_explored = 0
        best_cost: Optional[int] = None
        best_programs: List[Dict[str, Any]] = []
        best_cost_probe_start_at: Optional[int] = None
        probe_window_exhausted = False

        pruned_shape = 0
        pruned_palette = 0
        pruned_no_grid_modify = 0
        pruned_by_dominated_state = 0

        limit = int(max_programs_limit) if max_programs_limit is not None else int(max_programs)
        while frontier and int(programs_explored) < int(limit):
            (prio_key, steps, sig) = heapq.heappop(frontier)
            programs_explored += 1
            if int(programs_explored) > 1 and (int(programs_explored) & 0x0F) == 0:
                _snapshot_trace_programs_v141()
            # Priority tuple:
            # (best_loss_shape, best_loss_cells, slot_penalty_total, loss_shape, loss_cells, cost_bits, depth, program_sig)
            cost_bits = int(prio_key[5])
            depth = int(prio_key[6])
            ev = eval_program(steps)

            if int(trace_program_limit) > 0:
                tr_row = {
                    "program_sig": str(sig),
                    "depth": int(depth),
                    "cost_bits": int(cost_bits),
                    "loss": {"shape": int(ev.loss[0]), "cells": int(ev.loss[1])},
                    "ok_train": bool(ev.ok_train),
                    "mismatch_ex": ev.mismatch_ex,
                    "steps": [s.to_dict() for s in steps],
                }
                # Keep the best-N trace programs by (loss_shape, loss_cells, cost_bits, depth, sig).
                # This makes operator mining stable and useful (near-miss programs survive), rather than
                # storing just the first explored prefixes.
                tr_key = (int(ev.loss[0]), int(ev.loss[1]), int(cost_bits), int(depth), str(sig))
                bucket = trace_items_by_depth.get(int(depth))
                if bucket is None:
                    bucket = []
                    trace_items_by_depth[int(depth)] = bucket
                if len(bucket) < int(trace_per_depth):
                    bucket.append((tr_key, tr_row))
                else:
                    worst_i = 0
                    worst_k = bucket[0][0]
                    for i, (k, _row) in enumerate(bucket):
                        if k > worst_k:
                            worst_k = k
                            worst_i = i
                    if tr_key < worst_k:
                        bucket[worst_i] = (tr_key, tr_row)
                if int(programs_explored) == 1 or (int(programs_explored) & 0x0F) == 0:
                    _snapshot_trace_programs_v141()

            # Dominated-state pruning (generic, deterministic): if two programs induce the same
            # train-loss vector signature, keep only the lowest-cost one.
            #
            # Note: keep this AFTER trace capture so operator mining has enough near-miss evidence.
            if bool(enable_dominated_state_pruning) and str(ev.vec_sig):
                dom_cost = best_cost_by_vec_sig.get(str(ev.vec_sig))
                if dom_cost is not None and int(dom_cost) <= int(cost_bits):
                    pruned_by_dominated_state += 1
                    continue
                best_cost_by_vec_sig[str(ev.vec_sig)] = int(cost_bits)

            if ev.ok_train:
                if best_cost is None or int(cost_bits) < int(best_cost):
                    best_cost = int(cost_bits)
                    best_cost_probe_start_at = int(programs_explored)
                    # Tighten the retained solution set when best_cost improves.
                    best_programs = [
                        p for p in best_programs if int(p.get("cost_bits") or 0) <= int(best_cost) + int(solution_cost_slack_bits)
                    ]
                if int(cost_bits) <= int(best_cost) + int(solution_cost_slack_bits):
                    best_programs.append(
                        {
                            "program_sig": str(sig),
                            "cost_bits": int(cost_bits),
                            "steps": [s.to_dict() for s in steps],
                            "predicted_grid": [list(r) for r in ev.test_grid],
                            "predicted_grid_hash": _grid_hash_cached_v141(g=ev.test_grid, cache=grid_hash_cache),
                        }
                    )
                if len(best_programs) >= max_ambiguous_outputs:
                    break
                if best_cost_probe_start_at is not None and int(programs_explored - best_cost_probe_start_at) >= int(
                    POST_SOLUTION_PROBE_PROGRAMS_V141
                ):
                    # Stop after a bounded probe window since the last best_cost improvement.
                    # This preserves determinism and still allows some ambiguity discovery,
                    # while avoiding worst-case full-budget scans.
                    probe_window_exhausted = True
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
            mm_kind = ""
            if isinstance(ev.mismatch_ex, dict):
                mm_kind = str(ev.mismatch_ex.get("kind") or "")

            if bool(enable_reachability_pruning):
                ok_reach = True
                # Per-pair reachability: output shapes/palettes can differ across training examples.
                # Using the cross-product over {got_shapes}×{want_shapes} is overly conservative and
                # prunes valid programs (common in ARC).
                for got_shape, want_shape in zip(ev.got_shapes, want_shapes_by_pair):
                    if not _can_reach_shape_v141(
                        stage=stage, got_shape=tuple(int(x) for x in got_shape), want_shape=tuple(int(x) for x in want_shape), steps_left=steps_left
                    ):
                        pruned_shape += 1
                        ok_reach = False
                        break
                if not ok_reach:
                    continue
                for got_pal, want_pal in zip(ev.got_palettes, want_palettes_by_pair):
                    if not _can_reach_palette_v141(
                        stage=stage,
                        got_palette=set(int(x) for x in got_pal),
                        want_palette=set(int(x) for x in want_pal),
                        steps_left=steps_left,
                    ):
                        pruned_palette += 1
                        ok_reach = False
                        break
                if not ok_reach:
                    continue

                if _min_steps_to_grid_modify_v141(stage) > steps_left:
                    pruned_no_grid_modify += 1
                    continue

            # Telemetry: in hard timeout regimes, most time can be spent inside proposal/gating/push
            # for the very first expanded program. Emit progress snapshots so SIGALRM timeouts are
            # attributable (no more "enter_search @ t_total_s=3s" ambiguity).
            t_propose0: Optional[float] = None
            if int(programs_explored) == 1:
                t_propose0 = float(time.monotonic())
                _trace_stage(
                    "propose_next_steps",
                    programs_explored=int(programs_explored),
                    depth=int(depth),
                    state_stage=str(stage),
                )
                _snapshot_trace_programs_v141()

            next_steps: List[ProgramStepV141] = list(_get_candidates_cached_v141(steps))

            if int(programs_explored) == 1:
                t1 = float(time.monotonic())
                _trace_stage(
                    "proposed_next_steps",
                    programs_explored=int(programs_explored),
                    depth=int(depth),
                    state_stage=str(stage),
                    proposed_next_steps_total=int(len(next_steps)),
                    t_propose_next_steps_s=float(t1 - float(t_propose0 or t1)),
                )
                _snapshot_trace_programs_v141()

            # Track CSV-gating wall time (proposal->wrapped list) for the first expanded program.
            t_gate0: Optional[float] = None
            if int(programs_explored) == 1:
                t_gate0 = float(time.monotonic())

            # Abstraction pressure: concept-call becomes the only permitted policy surface.
            # Replace primitive next-steps with concept_call wrappers (no global fallback).
            if bool(abstraction_pressure):
                def _prefilter_prim_steps_for_csv_gate_v141(
                    *,
                    prim_steps: Sequence[ProgramStepV141],
                    mm_kind: str,
                    avail: Dict[str, bool],
                    train_pairs: Sequence[Tuple[GridV124, GridV124]],
                    test_in: GridV124,
                    bg_candidates: Sequence[int],
                    palette_out: Sequence[int],
                ) -> List[ProgramStepV141]:
                    """
                    Deterministic prefilter to keep CSV gating bounded under abstraction pressure.

                    Rationale: _propose_next_steps_v141 is availability+task-signal driven and can
                    emit many argument-instantiated primitive candidates. Wrapping all of them as
                    concept_calls and running CSV applicability on each can dominate wall time and
                    cause hard timeouts before meaningful search (programs_explored≈1).

                    This prefilter:
                      - keeps stage-critical slot-building ops (objset/obj/bbox/patch) when needed
                      - restricts ops by mismatch kind (shape vs cell)
                      - restricts color arguments to a small, evidence-derived set
                      - hard caps the resulting candidate list deterministically
                    """

                    def _ranked_colors(grids: Sequence[GridV124], *, limit: int) -> List[int]:
                        counts: Dict[int, int] = {}
                        for g in grids:
                            for row in g:
                                for x in row:
                                    counts[int(x)] = int(counts.get(int(x), 0)) + 1
                        items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
                        return [int(c) for c, _n in items[: int(max(0, int(limit)))]]

                    bg0 = int(bg_candidates[0]) if bg_candidates else 0
                    train_in = [p[0] for p in train_pairs]
                    train_out = [p[1] for p in train_pairs]
                    colors_in = [c for c in _ranked_colors(train_in + [test_in], limit=10) if int(c) != int(bg0)]
                    colors_out = [c for c in _ranked_colors(train_out, limit=10) if int(c) != int(bg0)]

                    # Diff-derived colors (generic): colors that actually change across demos.
                    diff_from: Dict[int, int] = {}
                    diff_to: Dict[int, int] = {}
                    ok_diff = True
                    for inp, out in train_pairs:
                        hi, wi = grid_shape_v124(inp)
                        ho, wo = grid_shape_v124(out)
                        if (int(hi), int(wi)) != (int(ho), int(wo)):
                            ok_diff = False
                            break
                        for r in range(int(hi)):
                            for c in range(int(wi)):
                                vi = int(inp[int(r)][int(c)])
                                vo = int(out[int(r)][int(c)])
                                if int(vi) == int(vo):
                                    continue
                                diff_from[int(vi)] = int(diff_from.get(int(vi), 0)) + 1
                                diff_to[int(vo)] = int(diff_to.get(int(vo), 0)) + 1

                    def _topk(counts: Dict[int, int], k: int) -> List[int]:
                        if not counts or int(k) <= 0:
                            return []
                        items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
                        return [int(c) for c, _n in items[: int(k)]]

                    diff_from_top = _topk(diff_from, 3) if ok_diff else []
                    diff_to_top = _topk(diff_to, 3) if ok_diff else []

                    # Allowed color sets (small, deterministic).
                    allow_color_any: Set[int] = set([int(bg0)])
                    for c in (colors_in[:4] + colors_out[:4] + diff_from_top + diff_to_top + [int(x) for x in palette_out[:4]]):
                        allow_color_any.add(int(c))
                    allow_color_in: Set[int] = set(int(c) for c in (colors_in[:4] + diff_from_top))
                    allow_color_out: Set[int] = set(int(c) for c in (colors_out[:4] + diff_to_top + [int(x) for x in palette_out[:4]]))
                    allow_bg: Set[int] = set(int(x) for x in (list(bg_candidates[:2]) if bg_candidates else [int(bg0)]))

                    # Always keep critical slot-building ops when the slot is missing.
                    required_ops: Set[str] = set()
                    if not bool(avail.get("objset", False)):
                        required_ops |= {"cc4", "cc4_nonbg_multicolor"}
                    if bool(avail.get("objset", False)) and not bool(avail.get("obj", False)):
                        required_ops |= {"select_obj"}
                    if bool(avail.get("obj", False)) and not bool(avail.get("bbox", False)):
                        required_ops |= {"obj_bbox", "bbox_by_color"}
                    if bool(avail.get("obj", False)):
                        # Allow obj_patch even when a patch already exists.
                        # Rationale: in learn-mode we often get an irrelevant patch early (mask_*).
                        # If obj_patch is disallowed once patch is set, extract→paste pipelines can
                        # become inexpressible under concept-as-policy.
                        required_ops |= {"obj_patch"}
                    if bool(avail.get("bbox", False)) and not bool(avail.get("patch", False)):
                        required_ops |= {"crop_bbox", "crop_bbox_by_color"}
                    if bool(avail.get("patch", False)):
                        required_ops |= {"commit_patch", "paste", "paint_mask"}

                    global_xform_ops = {"rotate90", "rotate180", "rotate270", "transpose", "reflect_h", "reflect_v"}
                    shape_ops = {
                        "crop_bbox_nonzero",
                        "pad_to",
                        "new_canvas",
                        "repeat_grid",
                        "downsample_mode",
                        "quadrant_center_tile",
                        "nest_by_color_area",
                        "cc4_color_bars",
                        "cc4_color_area_column",
                        "cc4_nonbg_bfs_column",
                        "crop_cc4_select",
                        "relational_expand",
                    }
                    color_ops = {
                        "replace_color",
                        "map_colors",
                        "paint_rect",
                        "draw_rect_border",
                        "fill_enclosed_region",
                        "symmetry_fill_h",
                        "symmetry_fill_v",
                        "symmetry_fill_rot180",
                        "best_diagonal_fill",
                        "histogram_color_counts",
                        "paint_mask",
                        "flood_fill",
                        "smear_nonbg",
                    }
                    spatial_ops = {"translate", "paste", "gravity", "overlay_self_translate"}
                    slot_ops = {
                        "mask_by_color",
                        "mask_bg",
                        "mask_border",
                        "mask_nonbg",
                        "mask_cross_center",
                        "mask_outline",
                        "mask_dilate",
                        "mask_box_dilate",
                        "mask_majority_color",
                        "mask_centroid",
                        "mask_cc4_select",
                        "mask_bbox",
                        "mask_obj",
                    }

                    allowed_ops: Set[str] = set(required_ops) | set(slot_ops)
                    if str(mm_kind) == "shape_mismatch":
                        allowed_ops |= set(shape_ops) | set(global_xform_ops) | set(spatial_ops)
                    else:
                        # Default to cell_mismatch behavior (also covers mm_kind="").
                        allowed_ops |= set(global_xform_ops) | set(spatial_ops) | set(color_ops) | set(shape_ops)

                    def _colors_ok(op_id: str, args: Dict[str, Any]) -> bool:
                        if not isinstance(args, dict):
                            return True
                        # common color keys
                        if "color" in args and isinstance(args.get("color"), int):
                            v = int(args["color"])
                            if op_id in {"mask_by_color", "bbox_by_color", "crop_bbox_by_color"}:
                                return int(v) in allow_color_in
                            return int(v) in allow_color_out or int(v) in allow_color_any
                        if "from_color" in args and isinstance(args.get("from_color"), int):
                            if int(args["from_color"]) not in allow_color_in and int(args["from_color"]) not in allow_color_any:
                                return False
                        if "to_color" in args and isinstance(args.get("to_color"), int):
                            if int(args["to_color"]) not in allow_color_out and int(args["to_color"]) not in allow_color_any:
                                return False
                        if "bg" in args and isinstance(args.get("bg"), int):
                            if int(args["bg"]) not in allow_bg:
                                return False
                        if "pad" in args and isinstance(args.get("pad"), int):
                            if int(args["pad"]) not in allow_bg and int(args["pad"]) not in allow_color_any:
                                return False
                        if "fill" in args and isinstance(args.get("fill"), int):
                            if int(args["fill"]) not in allow_color_out and int(args["fill"]) not in allow_color_any:
                                return False
                        return True

                    filtered: List[ProgramStepV141] = []
                    for st in prim_steps:
                        op0 = str(st.op_id or "")
                        if not op0:
                            continue
                        if op0 not in allowed_ops:
                            continue
                        if not _colors_ok(op0, dict(st.args)):
                            continue
                        filtered.append(st)

                    # Ensure stage-critical slot-building ops are present even if the upstream
                    # proposer omitted them (abstraction pressure + CSV gating can otherwise
                    # degenerate to programs_explored≈1).
                    present_ops: Set[str] = {str(st.op_id or "") for st in filtered if str(st.op_id or "")}

                    def _add_min_step(op_id: str, args: Dict[str, Any]) -> None:
                        if not op_id or op_id not in allowed_ops:
                            return
                        if op_id in present_ops:
                            return
                        filtered.append(ProgramStepV141(op_id=str(op_id), args=dict(args)))
                        present_ops.add(str(op_id))

                    # Always include an object-set builder when missing.
                    if "cc4" in required_ops:
                        _add_min_step("cc4", {"bg": int(bg0)})
                    if "cc4_nonbg_multicolor" in required_ops:
                        _add_min_step("cc4_nonbg_multicolor", {"bg": int(bg0)})

                    # Ensure obj_patch is available when an object is selected, even if the
                    # upstream proposer omitted it due to an existing patch.
                    if bool(avail.get("obj", False)) and "obj_patch" in required_ops:
                        _add_min_step("obj_patch", {"bg": int(bg0)})

                    # Always include at least one patch builder when patch is missing, so
                    # slot-progress (dl==0, ds<0) can unblock later loss-reducing CSGs.
                    if not bool(avail.get("patch", False)) and "mask_nonbg" not in present_ops:
                        _add_min_step("mask_nonbg", {})

                    # Hard cap (deterministic): keep cheapest/highest-priority ops first.
                    filtered.sort(
                        key=lambda s: (
                            0 if str(s.op_id or "") in required_ops else 1,
                            int(step_cost_bits_v141(op_id=str(s.op_id or ""), args=dict(s.args))),
                            canonical_json_dumps(s.to_dict()),
                        )
                    )
                    return filtered[:32]

                prim_steps: List[ProgramStepV141] = []
                existing_calls: List[ProgramStepV141] = []
                for st in next_steps:
                    op0 = str(st.op_id or "")
                    if not op0 or op0 in {"macro_call", "concept_call"}:
                        if op0 == "concept_call":
                            existing_calls.append(st)
                        continue
                    prim_steps.append(st)

                # Strict eval policy under abstraction pressure:
                #
                # Historically we suppressed primitive wrappers at depth=0 to avoid spending most
                # wall time CSV-testing primitives before exploring meaningful programs.
                #
                # However, doing so also removes the only available ingredient to build *bounded*
                # CSV pair/triad closures (builder→writer, builder→mid→writer). Those closures are
                # still "concept_call as policy" (atomic, gated) while enabling the first semantic
                # action when no bank concept is immediately loss-reducing at the empty prefix.
                #
                # Therefore: keep a small, prefiltered primitive pool at depth=0 so CSV pair/triad
                # closures can be synthesized deterministically. The CSV gate still prevents
                # non-causal branching from entering the frontier.

                prim_steps = _prefilter_prim_steps_for_csv_gate_v141(
                    prim_steps=prim_steps,
                    mm_kind=str(mm_kind),
                    avail=dict(avail),
                    train_pairs=train_pairs,
                    test_in=test_in,
                    bg_candidates=bg_candidates,
                    palette_out=palette_out,
                )

                wrapped: List[ProgramStepV141] = []

                # Preserve already-proposed concept_call candidates (direct inference, etc.).
                # Under abstraction pressure, they must still pass the CSV gate at the current prefix.
                if existing_calls:
                    def _call_sort_key(st: ProgramStepV141) -> Tuple[int, int, str, str]:
                        a0 = st.args if isinstance(st.args, dict) else {}
                        cb = a0.get("cost_bits")
                        cost0 = int(cb) if isinstance(cb, int) else 0
                        steps_raw = a0.get("steps")
                        clen = int(len(steps_raw)) if isinstance(steps_raw, list) else 0
                        cid0 = str(a0.get("concept_id") or "")
                        return (int(cost0), -int(clen), str(cid0), canonical_json_dumps(st.to_dict()))

                    existing_calls.sort(key=_call_sort_key)
                    existing_calls = existing_calls[:64]
                    gated_existing: List[Tuple[int, int, int, str, ProgramStepV141]] = []
                    for cw in existing_calls:
                        a0 = cw.args if isinstance(cw.args, dict) else {}
                        cid0 = str(a0.get("concept_id") or "")
                        steps_payload = a0.get("steps")
                        if not cid0 or not isinstance(steps_payload, list) or not steps_payload:
                            continue
                        app_key = (str(ev.vec_sig), str(cid0))
                        if app_key not in csv_applicable_cache:
                            metrics["csv_tested"] = int(metrics.get("csv_tested", 0)) + 1
                            score = _csv_applicability_score_v141(
                                steps=steps_payload,
                                train_final_states=ev.train_final_states,
                                train_pairs=train_pairs,
                                apply_cache=apply_cache,
                                grid_hash_cache=grid_hash_cache,
                                metrics=metrics,
                                allow_slot_progress=bool(csv_allow_slot_progress),
                            )
                            ok = False
                            dl = 0
                            ds = 0
                            if score is not None:
                                dl, ds = (int(score[0]), int(score[1]))
                                ok = _csv_gate_ok(dl=int(dl), ds=int(ds), delta_max=-1)
                            if not bool(ok):
                                metrics["csv_rejected"] = int(metrics.get("csv_rejected", 0)) + 1
                            else:
                                metrics["csv_survivors"] = int(metrics.get("csv_survivors", 0)) + 1
                            csv_applicable_cache[app_key] = (bool(ok), int(dl), int(ds))
                        ok0, dl0, ds0 = csv_applicable_cache.get(app_key, (False, 0, 0))
                        if not bool(ok0):
                            continue
                        cb = a0.get("cost_bits")
                        cost_bits0 = int(cb) if isinstance(cb, int) else 0
                        gated_existing.append((int(dl0), int(ds0), int(cost_bits0), str(cid0), cw))
                    gated_existing.sort(key=lambda t: (int(t[0]), int(t[1]), int(t[2]), str(t[3])))
                    wrapped.extend([t[-1] for t in gated_existing[:16]])

                for st in prim_steps:
                    op0 = str(st.op_id or "")
                    inner_cost = int(step_cost_bits_v141(op_id=str(op0), args=dict(st.args)))
                    wrapped.append(
                        ProgramStepV141(
                            op_id="concept_call",
                            args={
                                "concept_id": "csg_prim_" + str(op0),
                                "cost_bits": max(1, int(inner_cost) - 1),
                                "op_ids": [str(op0)],
                                "steps": [st.to_dict()],
                            },
                        )
                    )

                # CSV applicability gate for primitive wrappers:
                # Under abstraction pressure, even 1-step "primitive-as-concept" candidates must
                # prove immediate causal utility on TRAIN at the current prefix state, or they
                # reintroduce branching and SEARCH_BUDGET_EXCEEDED.
                gated_wrapped: List[Tuple[int, int, int, str, ProgramStepV141]] = []
                for cw in wrapped:
                    steps_payload = cw.args.get("steps") if isinstance(cw.args, dict) else None
                    if not isinstance(steps_payload, list) or not steps_payload:
                        continue
                    steps_sig = "prim_" + sha256_hex(canonical_json_dumps(steps_payload).encode("utf-8"))[:16]
                    app_key = (str(ev.vec_sig), str(steps_sig))
                    if app_key not in csv_applicable_cache:
                        metrics["csv_tested"] = int(metrics.get("csv_tested", 0)) + 1
                        score = _csv_applicability_score_v141(
                            steps=steps_payload,
                            train_final_states=ev.train_final_states,
                            train_pairs=train_pairs,
                            apply_cache=apply_cache,
                            grid_hash_cache=grid_hash_cache,
                            metrics=metrics,
                            allow_slot_progress=bool(csv_allow_slot_progress),
                        )
                        ok = False
                        dl = 0
                        ds = 0
                        if score is not None:
                            dl, ds = (int(score[0]), int(score[1]))
                            ok = _csv_gate_ok(dl=int(dl), ds=int(ds), delta_max=-1)
                        if not bool(ok):
                            metrics["csv_rejected"] = int(metrics.get("csv_rejected", 0)) + 1
                        else:
                            metrics["csv_survivors"] = int(metrics.get("csv_survivors", 0)) + 1
                        csv_applicable_cache[app_key] = (bool(ok), int(dl), int(ds))
                    ok0, dl0, ds0 = csv_applicable_cache.get(app_key, (False, 0, 0))
                    if not bool(ok0):
                        continue
                    cb = cw.args.get("cost_bits") if isinstance(cw.args, dict) else None
                    cost_bits = int(cb) if isinstance(cb, int) else 0
                    gated_wrapped.append((int(dl0), int(ds0), int(cost_bits), str(steps_sig), cw))
                gated_wrapped.sort(key=lambda t: (int(t[0]), int(t[1]), int(t[2]), str(t[3])))
                # Keep branching very small under abstraction pressure, but preserve *readiness*
                # candidates (slot builders) alongside immediate loss-reducers.
                #
                # If we keep only "best delta_loss" primitives, the solver can get stuck trying
                # to write the grid without having built internal slots (obj/bbox/patch), leading
                # to SEARCH_BUDGET_EXCEEDED. This mix keeps both pressures alive deterministically.
                best_by_loss = list(gated_wrapped[:4])
                best_by_slot = sorted(gated_wrapped, key=lambda t: (int(t[1]), int(t[0]), int(t[2]), str(t[3])))[:4]
                picked: List[Tuple[int, int, int, str, ProgramStepV141]] = []
                seen_sig: Set[str] = set()
                for t in (best_by_loss + best_by_slot):
                    if str(t[3]) in seen_sig:
                        continue
                    picked.append(t)
                    seen_sig.add(str(t[3]))
                for t in gated_wrapped:
                    if len(picked) >= 8:
                        break
                    if str(t[3]) in seen_sig:
                        continue
                    picked.append(t)
                    seen_sig.add(str(t[3]))
                # Ensure critical slot-building moves are always present in the candidate set.
                #
                # Without this, cheap patch-builders (mask_*) can crowd out objectization builders
                # (cc4/select_obj/obj_patch), making extract→paste pipelines inexpressible under a
                # small fixed depth/budget.
                def _primary_op_id(w: ProgramStepV141) -> str:
                    a0 = w.args if isinstance(w.args, dict) else {}
                    steps0 = a0.get("steps")
                    if isinstance(steps0, list) and steps0 and isinstance(steps0[0], dict):
                        return str(steps0[0].get("op_id") or "")
                    return ""

                must_ops: List[str] = []
                if not bool(avail.get("objset", False)):
                    must_ops.append("cc4")
                if bool(avail.get("objset", False)) and not bool(avail.get("obj", False)):
                    must_ops.append("select_obj")
                if bool(avail.get("obj", False)):
                    # obj_patch may be needed even when a patch already exists (overwrite patch).
                    must_ops.append("obj_patch")
                if bool(avail.get("obj", False)) and not bool(avail.get("bbox", False)):
                    must_ops.append("obj_bbox")

                for op_need in must_ops:
                    if any(_primary_op_id(t[-1]) == str(op_need) for t in picked):
                        continue
                    for t in gated_wrapped:
                        if _primary_op_id(t[-1]) == str(op_need) and str(t[3]) not in seen_sig:
                            picked.append(t)
                            seen_sig.add(str(t[3]))
                            break

                picked.sort(key=lambda t: (int(t[0]), int(t[1]), int(t[2]), str(t[3])))
                wrapped = [t[-1] for t in picked]
                # Limited 2-step closure proposals (generic): when a patch exists, many solutions
                # require a patch transform followed by a patch-consuming grid edit. Bundling these
                # as a single concept_call reduces outer depth without introducing task-specific
                # heuristics.
                if bool(avail.get("patch", False)) and prim_steps:
                    patch_transforms: List[ProgramStepV141] = []
                    patch_consumers: List[ProgramStepV141] = []
                    for st in prim_steps:
                        od = OP_DEFS_V141.get(str(st.op_id or ""))
                        if od is None:
                            continue
                        reads = set(str(r) for r in od.reads)
                        writes = set(str(w) for w in od.writes)
                        if "patch" in reads and "patch" in writes and "grid" not in writes:
                            patch_transforms.append(st)
                        if "patch" in reads and "grid" in writes:
                            patch_consumers.append(st)
                    patch_transforms.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
                    patch_consumers.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
                    for a in patch_transforms[:2]:
                        for b in patch_consumers[:4]:
                            body = {"kind": "concept_pair_v141", "a": a.to_dict(), "b": b.to_dict()}
                            cid = "csg_pair_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))[:16]
                            ca = int(step_cost_bits_v141(op_id=str(a.op_id), args=dict(a.args)))
                            cb = int(step_cost_bits_v141(op_id=str(b.op_id), args=dict(b.args)))
                            wrapped.append(
                                ProgramStepV141(
                                    op_id="concept_call",
                                    args={
                                        "concept_id": str(cid),
                                        "cost_bits": max(1, int(ca) + int(cb) - 1),
                                        "op_ids": [str(a.op_id), str(b.op_id)],
                                        "steps": [a.to_dict(), b.to_dict()],
                                    },
                                )
                            )

                # Generic 2-step CSV closures (depth=0 only): allow a non-improving "slot-building"
                # step followed by a grid-writing step to be proposed as a single concept_call.
                #
                # Rationale: many ARC programs require internal structure setup (obj/bbox/patch) that
                # does not immediately reduce TRAIN cell loss, but the *pair* is causally useful and
                # should be selectable under abstraction pressure without reintroducing global search.
                if int(depth) == 0 and prim_steps:
                    builders: List[ProgramStepV141] = []
                    writers: List[ProgramStepV141] = []
                    for st in prim_steps:
                        od = OP_DEFS_V141.get(str(st.op_id or ""))
                        if od is None:
                            continue
                        writes = {str(w) for w in tuple(getattr(od, "writes", ()) or ()) if str(w)}
                        if "grid" in writes:
                            writers.append(st)
                        if "grid" not in writes and (writes & {"objset", "obj", "bbox", "patch"}):
                            builders.append(st)

                    # Evidence-derived ordering (generic, deterministic): prioritize colors that
                    # actually change across TRAIN pairs so depth-0 closures cover the relevant
                    # palette even when colors are not small integers.
                    diff_from_top: List[int] = []
                    diff_to_top: List[int] = []
                    try:
                        diff_from: Dict[int, int] = {}
                        diff_to: Dict[int, int] = {}
                        ok_diff = True
                        for inp, out in train_pairs:
                            hi, wi = grid_shape_v124(inp)
                            ho, wo = grid_shape_v124(out)
                            if (int(hi), int(wi)) != (int(ho), int(wo)):
                                ok_diff = False
                                break
                            for r in range(int(hi)):
                                for c in range(int(wi)):
                                    vi = int(inp[int(r)][int(c)])
                                    vo = int(out[int(r)][int(c)])
                                    if int(vi) == int(vo):
                                        continue
                                    diff_from[int(vi)] = int(diff_from.get(int(vi), 0)) + 1
                                    diff_to[int(vo)] = int(diff_to.get(int(vo), 0)) + 1
                        if ok_diff:
                            diff_from_top = [
                                int(c)
                                for c, _n in sorted(
                                    ((int(c), int(n)) for c, n in diff_from.items()),
                                    key=lambda cn: (-int(cn[1]), int(cn[0])),
                                )[:3]
                            ]
                            diff_to_top = [
                                int(c)
                                for c, _n in sorted(
                                    ((int(c), int(n)) for c, n in diff_to.items()),
                                    key=lambda cn: (-int(cn[1]), int(cn[0])),
                                )[:3]
                            ]
                    except Exception:
                        diff_from_top = []
                        diff_to_top = []

                    def _step_color_arg(st: ProgramStepV141) -> Optional[int]:
                        a0 = st.args if isinstance(st.args, dict) else {}
                        v0 = a0.get("color")
                        return int(v0) if isinstance(v0, int) else None

                    def _builder_sort_key(st: ProgramStepV141) -> Tuple[int, int, str]:
                        op0 = str(st.op_id or "")
                        c0 = _step_color_arg(st)
                        pr = 3
                        if op0 in {"mask_by_color", "bbox_by_color", "crop_bbox_by_color"}:
                            pr = 0 if (c0 is not None and int(c0) in set(int(x) for x in diff_from_top)) else 2
                        elif op0 in {"mask_nonbg", "mask_border", "mask_bg", "mask_majority_color"}:
                            pr = 1
                        cost0 = int(step_cost_bits_v141(op_id=str(op0), args=dict(st.args)))
                        return (int(pr), int(cost0), canonical_json_dumps(st.to_dict()))

                    def _writer_sort_key(st: ProgramStepV141) -> Tuple[int, int, str]:
                        op0 = str(st.op_id or "")
                        c0 = _step_color_arg(st)
                        pr = 3
                        if op0 in {"paint_mask", "paint_rect", "draw_rect_border", "replace_color", "fill_enclosed_region", "flood_fill"}:
                            pr = 0 if (c0 is not None and int(c0) in set(int(x) for x in diff_to_top)) else 1
                        cost0 = int(step_cost_bits_v141(op_id=str(op0), args=dict(st.args)))
                        return (int(pr), int(cost0), canonical_json_dumps(st.to_dict()))

                    builders.sort(key=_builder_sort_key)
                    writers.sort(key=_writer_sort_key)
                    # Builder-only CSV closures: allow selecting non-grid-writing slot builders
                    # as semantic moves (dl==0, ds<0). Without this, tasks that require internal
                    # structure setup (objset/obj/bbox/patch) can have *zero* CSV-admissible moves
                    # at depth=0 and the search never starts (programs_explored≈1).
                    for a in builders[:8]:
                        od = OP_DEFS_V141.get(str(a.op_id or ""))
                        if od is None:
                            continue
                        writes0 = {str(w) for w in tuple(getattr(od, "writes", ()) or ()) if str(w)}
                        if "grid" in writes0:
                            continue
                        # Only keep builders that actually add missing structure at this prefix.
                        if not any((slot in writes0 and not bool(avail.get(slot, False))) for slot in ("objset", "obj", "bbox", "patch")):
                            continue
                        body = {"kind": "csv_builder_v141", "a": a.to_dict()}
                        cid = "csg_csvbuilder_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))[:16]
                        app_key = (str(ev.vec_sig), str(cid))
                        if app_key not in csv_applicable_cache:
                            metrics["csv_tested"] = int(metrics.get("csv_tested", 0)) + 1
                            score = _csv_applicability_score_v141(
                                steps=[a.to_dict()],
                                train_final_states=ev.train_final_states,
                                train_pairs=train_pairs,
                                apply_cache=apply_cache,
                                grid_hash_cache=grid_hash_cache,
                                metrics=metrics,
                                allow_slot_progress=bool(csv_allow_slot_progress),
                            )
                            ok = False
                            dl = 0
                            ds = 0
                            if score is not None:
                                dl, ds = (int(score[0]), int(score[1]))
                                ok = _csv_gate_ok(dl=int(dl), ds=int(ds), delta_max=-1)
                            if not bool(ok):
                                metrics["csv_rejected"] = int(metrics.get("csv_rejected", 0)) + 1
                            csv_applicable_cache[app_key] = (bool(ok), int(dl), int(ds))
                        ok0, _dl0, _ds0 = csv_applicable_cache.get(app_key, (False, 0, 0))
                        if not bool(ok0):
                            continue
                        ca = int(step_cost_bits_v141(op_id=str(a.op_id), args=dict(a.args)))
                        wrapped.append(
                            ProgramStepV141(
                                op_id="concept_call",
                                args={
                                    "concept_id": str(cid),
                                    "cost_bits": max(1, int(ca) - 1),
                                    "op_ids": [str(a.op_id)],
                                    "steps": [a.to_dict()],
                                },
                            )
                        )
                    # Keep the pair-combination search modest to avoid dominating wall time.
                    for a in builders[:4]:
                        for b in writers[:6]:
                            body = {"kind": "csv_pair_v141", "a": a.to_dict(), "b": b.to_dict()}
                            cid = "csg_csvpair_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))[:16]
                            app_key = (str(ev.vec_sig), str(cid))
                            if app_key not in csv_applicable_cache:
                                metrics["csv_tested"] = int(metrics.get("csv_tested", 0)) + 1
                                score = _csv_applicability_score_v141(
                                    steps=[a.to_dict(), b.to_dict()],
                                    train_final_states=ev.train_final_states,
                                    train_pairs=train_pairs,
                                    apply_cache=apply_cache,
                                    grid_hash_cache=grid_hash_cache,
                                    metrics=metrics,
                                    allow_slot_progress=bool(csv_allow_slot_progress),
                                )
                                ok = False
                                dl = 0
                                ds = 0
                                if score is not None:
                                    dl, ds = (int(score[0]), int(score[1]))
                                    ok = _csv_gate_ok(dl=int(dl), ds=int(ds), delta_max=-1)
                                if not bool(ok):
                                    metrics["csv_rejected"] = int(metrics.get("csv_rejected", 0)) + 1
                                csv_applicable_cache[app_key] = (bool(ok), int(dl), int(ds))
                            ok0, _dl0, _ds0 = csv_applicable_cache.get(app_key, (False, 0, 0))
                            if not bool(ok0):
                                continue
                            ca = int(step_cost_bits_v141(op_id=str(a.op_id), args=dict(a.args)))
                            cb = int(step_cost_bits_v141(op_id=str(b.op_id), args=dict(b.args)))
                            wrapped.append(
                                ProgramStepV141(
                                    op_id="concept_call",
                                    args={
                                        "concept_id": str(cid),
                                        "cost_bits": max(1, int(ca) + int(cb) - 2),
                                        "op_ids": [str(a.op_id), str(b.op_id)],
                                        "steps": [a.to_dict(), b.to_dict()],
                                    },
                                )
                            )

                    # Generic 3-step CSV closures (depth=0 only): many ARC transformations require
                    # a short *pipeline* (build → transform → write) where intermediate steps may
                    # not reduce grid loss monotonically, but the *bundle* is causally useful.
                    #
                    # We enumerate a small, deterministic set of feasible triads from the current
                    # primitive candidate pool and subject them to the same CSV applicability gate.
                    mids: List[ProgramStepV141] = []
                    for st in prim_steps:
                        od = OP_DEFS_V141.get(str(st.op_id or ""))
                        if od is None:
                            continue
                        reads = {str(r) for r in tuple(getattr(od, "reads", ()) or ()) if str(r)}
                        writes = {str(w) for w in tuple(getattr(od, "writes", ()) or ()) if str(w)}
                        if "grid" in writes:
                            continue
                        # Prefer patch/bbox/object transforms (they often unlock 1-step commits).
                        if ("patch" in reads and "patch" in writes) or ("bbox" in reads and "patch" in writes) or ("obj" in reads and "patch" in writes):
                            mids.append(st)
                    mids.sort(key=lambda s: canonical_json_dumps(s.to_dict()))

                    def _feasible_chain(seq: Sequence[ProgramStepV141]) -> bool:
                        cur = {
                            "grid": True,
                            "objset": bool(avail.get("objset", False)),
                            "obj": bool(avail.get("obj", False)),
                            "bbox": bool(avail.get("bbox", False)),
                            "patch": bool(avail.get("patch", False)),
                        }
                        for st in seq:
                            od = OP_DEFS_V141.get(str(st.op_id or ""))
                            if od is None:
                                return False
                            reads = {str(r) for r in tuple(getattr(od, "reads", ()) or ()) if str(r)}
                            for r in reads:
                                if not bool(cur.get(str(r), False)):
                                    return False
                            writes = {str(w) for w in tuple(getattr(od, "writes", ()) or ()) if str(w)}
                            for w in writes:
                                cur[str(w)] = True
                        return True

                    triad_budget = 0
                    for a in builders[:4]:
                        for m in mids[:6]:
                            for b in writers[:6]:
                                if triad_budget >= 48:
                                    break
                                if not _feasible_chain([a, m, b]):
                                    continue
                                body = {"kind": "csv_triad_v141", "a": a.to_dict(), "m": m.to_dict(), "b": b.to_dict()}
                                cid = "csg_csvtriad_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))[:16]
                                app_key = (str(ev.vec_sig), str(cid))
                                if app_key not in csv_applicable_cache:
                                    metrics["csv_tested"] = int(metrics.get("csv_tested", 0)) + 1
                                    score = _csv_applicability_score_v141(
                                        steps=[a.to_dict(), m.to_dict(), b.to_dict()],
                                        train_final_states=ev.train_final_states,
                                        train_pairs=train_pairs,
                                        apply_cache=apply_cache,
                                        grid_hash_cache=grid_hash_cache,
                                        metrics=metrics,
                                        allow_slot_progress=bool(csv_allow_slot_progress),
                                    )
                                    ok = False
                                    dl = 0
                                    ds = 0
                                    if score is not None:
                                        dl, ds = (int(score[0]), int(score[1]))
                                        ok = _csv_gate_ok(dl=int(dl), ds=int(ds), delta_max=-1)
                                    if not bool(ok):
                                        metrics["csv_rejected"] = int(metrics.get("csv_rejected", 0)) + 1
                                    csv_applicable_cache[app_key] = (bool(ok), int(dl), int(ds))
                                ok0, _dl0, _ds0 = csv_applicable_cache.get(app_key, (False, 0, 0))
                                if not bool(ok0):
                                    continue
                                ca = int(step_cost_bits_v141(op_id=str(a.op_id), args=dict(a.args)))
                                cm = int(step_cost_bits_v141(op_id=str(m.op_id), args=dict(m.args)))
                                cb = int(step_cost_bits_v141(op_id=str(b.op_id), args=dict(b.args)))
                                wrapped.append(
                                    ProgramStepV141(
                                        op_id="concept_call",
                                        args={
                                            "concept_id": str(cid),
                                            "cost_bits": max(1, int(ca) + int(cm) + int(cb) - 4),
                                            "op_ids": [str(a.op_id), str(m.op_id), str(b.op_id)],
                                            "steps": [a.to_dict(), m.to_dict(), b.to_dict()],
                                        },
                                    )
                                )
                                triad_budget += 1
                            if triad_budget >= 48:
                                break
                        if triad_budget >= 48:
                            break

                # Built-in multi-step CSG closures (generic): object/bbox pipelines bundled as a
                # single concept_call. Propose them only at the grid stage to keep branching bounded.
                if str(stage) == "grid":
                    # Subject built-ins to the same CSV applicability gate so they cannot dominate
                    # search unless they are causally useful now (TRAIN-improving at the prefix).
                    builtin = _propose_builtin_csg_calls_v141(
                        train_pairs=train_pairs,
                        test_in=test_in,
                        bg_candidates=bg_candidates,
                    )
                    gated_builtin: List[Tuple[int, int, int, str, ProgramStepV141]] = []
                    for cw in builtin:
                        a0 = cw.args if isinstance(cw.args, dict) else {}
                        cid0 = str(a0.get("concept_id") or "")
                        steps_payload = a0.get("steps")
                        if not cid0 or not isinstance(steps_payload, list) or not steps_payload:
                            continue
                        app_key = (str(ev.vec_sig), str(cid0))
                        if app_key not in csv_applicable_cache:
                            metrics["csv_tested"] = int(metrics.get("csv_tested", 0)) + 1
                            score = _csv_applicability_score_v141(
                                steps=steps_payload,
                                train_final_states=ev.train_final_states,
                                train_pairs=train_pairs,
                                apply_cache=apply_cache,
                                grid_hash_cache=grid_hash_cache,
                                metrics=metrics,
                                allow_slot_progress=bool(csv_allow_slot_progress),
                            )
                            ok = False
                            dl = 0
                            ds = 0
                            if score is not None:
                                dl, ds = (int(score[0]), int(score[1]))
                                ok = _csv_gate_ok(dl=int(dl), ds=int(ds), delta_max=-1)
                            if not bool(ok):
                                metrics["csv_rejected"] = int(metrics.get("csv_rejected", 0)) + 1
                            csv_applicable_cache[app_key] = (bool(ok), int(dl), int(ds))
                        ok0, dl0, ds0 = csv_applicable_cache.get(app_key, (False, 0, 0))
                        if not bool(ok0):
                            continue
                        cb = a0.get("cost_bits")
                        cost_bits0 = int(cb) if isinstance(cb, int) else 0
                        gated_builtin.append((int(dl0), int(ds0), int(cost_bits0), str(cid0), cw))
                    gated_builtin.sort(key=lambda t: (int(t[0]), int(t[1]), int(t[2]), str(t[3])))
                    wrapped.extend([t[-1] for t in gated_builtin[:12]])

                # Prefix closures (depth=0 only): enable CSG concepts that require internal slots
                # (objset/patch/...) by bundling a *slot-building* primitive step in front of the
                # concept's concrete steps, forming one atomic concept_call that is still subject
                # to the strict CSV gate (must improve TRAIN loss).
                #
                # This is the minimal mechanism needed to escape the "no root actions" regime:
                # many mined CSGs start with select_obj/obj_bbox/mask_outline (requires slots), but
                # the strict gate forbids taking a standalone builder step. Prefix bundling keeps
                # concept-as-policy intact while letting useful high-level concepts enter at depth 0.
                # NOTE: prefix synthesis is expensive; run it only when the current root policy
                # surface is empty (i.e., no loss-reducing primitive/CSG closure survived).
                # Prefix closures are an optional escape hatch to bootstrap slot-dependent CSGs.
                #
                # IMPORTANT: keep this OFF by default in eval_arc. The binder resolution required
                # to make prefixed templates concrete can be expensive enough to consume the entire
                # per-task timeout before search even starts, yielding stage="proposed_next_steps"
                # and SEARCH_BUDGET_EXCEEDED with empty trace_programs.
                #
                # The intended long-term path is to mine loss-reducing, self-contained CSGs (with
                # internal slot builders) in learn mode and import them into the bank, so eval can
                # remain strict and fast.
                if (
                    bool(getattr(config, "enable_csv_prefix_closures", False))
                    and int(depth) == 0
                    and bool(concept_templates_norm)
                    and prim_steps
                    and not wrapped
                ):
                    # Candidate builders: must not write grid and must add at least one missing slot.
                    builders0: List[ProgramStepV141] = []
                    for st in prim_steps:
                        od = OP_DEFS_V141.get(str(st.op_id or ""))
                        if od is None:
                            continue
                        writes = {str(w) for w in tuple(getattr(od, "writes", ()) or ()) if str(w)}
                        if "grid" in writes:
                            continue
                        if not any((slot in writes and not bool(avail.get(slot, False))) for slot in ("objset", "obj", "bbox", "patch")):
                            continue
                        # Only keep builders feasible at this prefix (reads already satisfied).
                        reads = {str(r) for r in tuple(getattr(od, "reads", ()) or ()) if str(r)}
                        ok_reads = True
                        for r in reads:
                            if not bool(avail.get(str(r), False)):
                                ok_reads = False
                                break
                        if not ok_reads:
                            continue
                        builders0.append(st)
                    builders0.sort(
                        key=lambda s: (
                            int(step_cost_bits_v141(op_id=str(s.op_id or ""), args=dict(s.args))),
                            canonical_json_dumps(s.to_dict()),
                        )
                    )
                    builders0 = builders0[:2]

                    # Prefilter templates that are NOT feasible now but could become feasible after
                    # one builder (based on first-op reads).
                    cand_tmps: List[Tuple[int, int, int, str, Dict[str, Any]]] = []
                    for tmpl in concept_templates_norm:
                        cid0 = str(tmpl.get("concept_id") or "")
                        steps_payload0 = tmpl.get("steps")
                        if not cid0 or not isinstance(steps_payload0, list) or not steps_payload0:
                            continue
                        row0 = steps_payload0[0] if isinstance(steps_payload0[0], dict) else {}
                        op0 = str(row0.get("op_id") or "")
                        if not op0:
                            continue
                        od0 = OP_DEFS_V141.get(str(op0))
                        if od0 is None:
                            continue
                        reads0 = {str(r) for r in tuple(getattr(od0, "reads", ()) or ()) if str(r)}
                        missing0 = [r for r in sorted(reads0) if not bool(avail.get(str(r), False))]
                        if not missing0:
                            # Already feasible at this prefix; regular concept selection handles it.
                            continue
                        # Only consider templates missing one of the "internal" slots.
                        if not any(r in {"objset", "obj", "bbox", "patch"} for r in missing0):
                            continue
                        # Cheap ranking: prefer high support + low cost; deterministic tiebreak by id.
                        sup0 = int(tmpl.get("support") or 0)
                        cb0 = tmpl.get("cost_bits")
                        cost_bits0 = int(cb0) if isinstance(cb0, int) else 10
                        cand_tmps.append((-int(sup0), int(cost_bits0), int(len(steps_payload0)), str(cid0), tmpl))
                    cand_tmps.sort(key=lambda t: (int(t[0]), int(t[1]), -int(t[2]), str(t[3])))
                    cand_tmps = cand_tmps[:24]

                    prefix_resolve_cache: Dict[Tuple[str, str], Optional[List[Dict[str, Any]]]] = {}
                    added = 0
                    for a in builders0:
                        if added >= 4:
                            break
                        # Simulate availability after builder to filter templates quickly.
                        od_a = OP_DEFS_V141.get(str(a.op_id or ""))
                        if od_a is None:
                            continue
                        writes_a = {str(w) for w in tuple(getattr(od_a, "writes", ()) or ()) if str(w)}
                        avail2 = dict(avail)
                        for w in writes_a:
                            avail2[str(w)] = True

                        for _neg_sup, _cost, _lsteps, cid0, tmpl in cand_tmps:
                            if added >= 4:
                                break
                            steps_payload0 = tmpl.get("steps")
                            if not isinstance(steps_payload0, list) or not steps_payload0:
                                continue
                            row0 = steps_payload0[0] if isinstance(steps_payload0[0], dict) else {}
                            op0 = str(row0.get("op_id") or "")
                            if not op0:
                                continue
                            od0 = OP_DEFS_V141.get(str(op0))
                            if od0 is None:
                                continue
                            reads0 = {str(r) for r in tuple(getattr(od0, "reads", ()) or ()) if str(r)}
                            if any(not bool(avail2.get(str(r), False)) for r in reads0):
                                continue

                            # Build combined steps: [builder] + template steps.
                            inner: List[Dict[str, Any]] = [a.to_dict()] + [dict(x) for x in steps_payload0 if isinstance(x, dict)]
                            if not inner:
                                continue
                            # Determine binder presence.
                            has_binder = False
                            for row in inner:
                                if not isinstance(row, dict):
                                    continue
                                a0 = row.get("args") if isinstance(row.get("args"), dict) else {}
                                for v0 in a0.values():
                                    if isinstance(v0, dict) and ("bind" in v0 or "infer" in v0):
                                        has_binder = True
                                        break
                                if has_binder:
                                    break

                            body_sig = sha256_hex(canonical_json_dumps({"a": a.to_dict(), "cid": str(cid0)}).encode("utf-8"))[:16]
                            pcid = "csg_pref_" + str(body_sig)
                            app_key = (str(ev.vec_sig), str(pcid))
                            if app_key not in csv_applicable_cache:
                                # Resolve binders for the combined sequence (critical for select_obj).
                                key2 = (str(a.op_id or "") + ":" + canonical_json_dumps(a.to_dict()), str(cid0))
                                resolved_steps = prefix_resolve_cache.get(key2)
                                if resolved_steps is None and key2 not in prefix_resolve_cache:
                                    resolved_steps = _resolve_concept_csg_binders_v141(
                                        steps=inner, train_pairs=train_pairs, test_in=test_in
                                    )
                                    prefix_resolve_cache[key2] = list(resolved_steps) if resolved_steps is not None else None
                                    resolved_steps = prefix_resolve_cache.get(key2)
                                if resolved_steps is None:
                                    csv_applicable_cache[app_key] = (False, 0, 0)
                                else:
                                    metrics["csv_tested"] = int(metrics.get("csv_tested", 0)) + 1
                                    score = _csv_applicability_score_v141(
                                        steps=resolved_steps,
                                        train_final_states=ev.train_final_states,
                                        train_pairs=train_pairs,
                                        apply_cache=apply_cache,
                                        grid_hash_cache=grid_hash_cache,
                                        metrics=metrics,
                                        allow_slot_progress=bool(csv_allow_slot_progress),
                                    )
                                    ok = False
                                    dl = 0
                                    ds = 0
                                    if score is not None:
                                        dl, ds = (int(score[0]), int(score[1]))
                                        ok = _csv_gate_ok(dl=int(dl), ds=int(ds), delta_max=-1)
                                    if not bool(ok):
                                        metrics["csv_rejected"] = int(metrics.get("csv_rejected", 0)) + 1
                                    csv_applicable_cache[app_key] = (bool(ok), int(dl), int(ds))

                            ok0, _dl0, _ds0 = csv_applicable_cache.get(app_key, (False, 0, 0))
                            if not bool(ok0):
                                continue

                            # Cost model: compressed atomic call, but keep cost correlated with inner steps.
                            inner_cost = 0
                            for row in inner:
                                if not isinstance(row, dict):
                                    continue
                                opx = str(row.get("op_id") or "")
                                argsx = row.get("args") if isinstance(row.get("args"), dict) else {}
                                if not opx:
                                    continue
                                inner_cost += int(step_cost_bits_v141(op_id=str(opx), args=dict(argsx)))
                            cost_bits = max(1, int(inner_cost) - 3)
                            wrapped.append(
                                ProgramStepV141(
                                    op_id="concept_call",
                                    args={
                                        "concept_id": str(pcid),
                                        "cost_bits": int(cost_bits),
                                        "op_ids": [str(a.op_id)] + [str(op0)],
                                        "steps": prefix_resolve_cache.get((str(a.op_id or "") + ":" + canonical_json_dumps(a.to_dict()), str(cid0))) or inner,
                                    },
                                )
                            )
                            added += 1

                next_steps = wrapped

                if int(programs_explored) == 1:
                    t2 = float(time.monotonic())
                    _trace_stage(
                        "csv_gate_applied",
                        programs_explored=int(programs_explored),
                        depth=int(depth),
                        state_stage=str(stage),
                        csv_tested=int(metrics.get("csv_tested", 0)),
                        csv_rejected=int(metrics.get("csv_rejected", 0)),
                        next_steps_total=int(len(next_steps)),
                        t_csv_gate_s=float(t2 - float(t_gate0 or t2)),
                    )
                    _snapshot_trace_programs_v141()

            macro_allowed_here = bool(allow_macros) and bool(macro_templates_norm) and int(depth) <= int(macro_propose_max_depth)
            if macro_allowed_here:
                for tmpl in macro_templates_norm:
                    mid = str(tmpl.get("macro_id") or "")
                    op_ids = tmpl.get("op_ids") if isinstance(tmpl.get("op_ids"), list) else []
                    tmpl_len = int(len(op_ids))
                    max_inst = int(macro_max_instantiations)
                    max_branch = int(macro_max_branch_per_op)
                    if tmpl_len >= 6:
                        max_inst = max(2, int(max_inst) // 5)
                        max_branch = max(3, int(max_branch) // 3)
                    elif tmpl_len == 5:
                        max_inst = max(2, int(max_inst) // 3)
                        max_branch = max(4, int(max_branch) // 2)
                    elif tmpl_len == 4:
                        max_inst = max(3, int(max_inst) // 2)
                    insts = _instantiate_macro_steps_v141(
                        macro_op_ids=[str(x) for x in op_ids if str(x)],
                        steps_prefix=steps,
                        train_pairs=train_pairs,
                        test_in=test_in,
                        bg_candidates=bg_candidates,
                        shapes_out=shapes_out,
                        palette_out=palette_out,
                        direct_steps=direct_steps,
                        max_instantiations=int(max_inst),
                        max_branch_per_op=int(max_branch),
                        get_candidates=_get_candidates_cached_v141,
                    )
                    for inner_steps in insts:
                        inner_steps_dicts = [s.to_dict() for s in inner_steps]
                        # Under abstraction pressure, macro-derived closures must also pass the
                        # CSV applicability gate (TRAIN-improving at the current prefix).
                        if bool(abstraction_pressure):
                            steps_sig = "macro_" + sha256_hex(canonical_json_dumps(inner_steps_dicts).encode("utf-8"))[:16]
                            app_key = (str(ev.vec_sig), str(steps_sig))
                            if app_key not in csv_applicable_cache:
                                metrics["csv_tested"] = int(metrics.get("csv_tested", 0)) + 1
                                score = _csv_applicability_score_v141(
                                    steps=inner_steps_dicts,
                                    train_final_states=ev.train_final_states,
                                    train_pairs=train_pairs,
                                    apply_cache=apply_cache,
                                    grid_hash_cache=grid_hash_cache,
                                    metrics=metrics,
                                    allow_slot_progress=bool(csv_allow_slot_progress),
                                )
                                ok = False
                                dl = 0
                                ds = 0
                                if score is not None:
                                    dl, ds = (int(score[0]), int(score[1]))
                                    ok = _csv_gate_ok(dl=int(dl), ds=int(ds), delta_max=-1)
                                if not bool(ok):
                                    metrics["csv_rejected"] = int(metrics.get("csv_rejected", 0)) + 1
                                csv_applicable_cache[app_key] = (bool(ok), int(dl), int(ds))
                            ok0, _dl0, _ds0 = csv_applicable_cache.get(app_key, (False, 0, 0))
                            if not bool(ok0):
                                continue
                        if not bool(abstraction_pressure):
                            next_steps.append(
                                ProgramStepV141(
                                    op_id="macro_call",
                                    args={"macro_id": str(mid), "steps": list(inner_steps_dicts)},
                                )
                            )
                        # Bridge: allow learned operators (macros) to be used as explicit concepts.
                        # This makes concept_call a real compositional policy surface without needing
                        # a separate concept-template bank to "re-label" the same closure.
                        #
                        # Determinism: concept_id is derived from macro_id; cost uses the same MDL
                        # compression signal as macro_call (bounded by inner cost).
                        macro_cost = int(
                            _macro_call_cost_bits_v141(
                                macro_id=str(mid),
                                steps=list(inner_steps_dicts),
                                support_by_macro_id=macro_support_by_id,
                            )
                        )
                        next_steps.append(
                            ProgramStepV141(
                                op_id="concept_call",
                                args={
                                    "concept_id": "csg_" + str(mid)[:16],
                                    "cost_bits": max(1, int(macro_cost) - 1),
                                    "op_ids": [str(x) for x in op_ids if str(x)],
                                    "steps": list(inner_steps_dicts),
                                },
                            )
                        )

            concept_allowed_here = bool(concept_templates_norm) and int(depth) <= int(concept_propose_max_depth)
            if concept_allowed_here:
                def _tmpl_primary_op_id(tmpl: Dict[str, Any]) -> str:
                    steps_payload = tmpl.get("steps")
                    if isinstance(steps_payload, list) and steps_payload and isinstance(steps_payload[0], dict):
                        op0 = str(steps_payload[0].get("op_id") or "")
                        if op0:
                            return str(op0)
                    op_ids0 = tmpl.get("op_ids") if isinstance(tmpl.get("op_ids"), list) else []
                    for x in op_ids0:
                        if str(x):
                            return str(x)
                    return ""

                def _tmpl_first_op_feasible(op_id: str) -> bool:
                    od0 = OP_DEFS_V141.get(str(op_id) or "")
                    if od0 is None:
                        return False
                    for r0 in od0.reads:
                        if not bool(avail.get(str(r0), False)):
                            return False
                    return True

                def _tmpl_policy_rank(primary_op_id: str) -> int:
                    op = str(primary_op_id or "")
                    if not op:
                        return 9
                    # Stage-critical moves first (if the program has made those slots available).
                    if op == "commit_patch" and bool(avail.get("patch", False)):
                        return 0
                    if op in {
                        "patch_rotate90",
                        "patch_rotate180",
                        "patch_rotate270",
                        "patch_reflect_h",
                        "patch_reflect_v",
                        "patch_transpose",
                        "patch_translate",
                    } and bool(avail.get("patch", False)):
                        return 1
                    if op in {"mask_outline", "mask_dilate", "mask_box_dilate"} and bool(avail.get("patch", False)):
                        return 2
                    if op in {"mask_by_color", "mask_cross_center"} and not bool(avail.get("patch", False)):
                        return 2
                    if op == "crop_bbox" and bool(avail.get("bbox", False)) and not bool(avail.get("patch", False)):
                        return 0
                    if op == "obj_bbox" and bool(avail.get("obj", False)) and not bool(avail.get("bbox", False)):
                        return 1
                    if op == "obj_patch" and bool(avail.get("obj", False)) and not bool(avail.get("patch", False)):
                        return 1
                    if op == "select_obj" and bool(avail.get("objset", False)) and not bool(avail.get("obj", False)):
                        return 1
                    if op in {"cc4", "cc4_nonbg_multicolor"} and not bool(avail.get("objset", False)):
                        return 1
                    if op == "bbox_by_color" and not bool(avail.get("bbox", False)):
                        return 1

                    global_xform_ops = {"rotate90", "rotate180", "rotate270", "transpose", "reflect_h", "reflect_v"}
                    shape_ops = {
                        "crop_bbox_nonzero",
                        "pad_to",
                        "new_canvas",
                        "repeat_grid",
                        "downsample_mode",
                        "cc4_color_bars",
                        "cc4_color_area_column",
                        "cc4_nonbg_bfs_column",
                    }
                    color_ops = {
                        "replace_color",
                        "map_colors",
                        "paint_rect",
                        "draw_rect_border",
                        "fill_enclosed_region",
                        "symmetry_fill_h",
                        "symmetry_fill_v",
                        "paint_mask",
                        "flood_fill",
                    }
                    spatial_ops = {"translate", "paste", "gravity"}
                    special_ops = {"overlay_self_translate", "propagate_color_translate", "propagate_nonbg_translate"}

                    if str(mm_kind) == "shape_mismatch":
                        if op in shape_ops:
                            return 2
                        if op in global_xform_ops:
                            return 3
                        if op in spatial_ops:
                            return 4
                        if op in color_ops:
                            return 6
                        if op in special_ops:
                            return 7
                        return 8

                    if str(mm_kind) == "cell_mismatch":
                        if op in global_xform_ops:
                            return 2
                        if op in spatial_ops:
                            return 3
                        if op in color_ops:
                            return 4
                        if op in special_ops:
                            return 6
                        if op in shape_ops:
                            return 7
                        return 7

                    return 5

                def _select_concept_templates() -> List[Dict[str, Any]]:
                    # Under abstraction pressure, widen the candidate pool *before* the CSV gate,
                    # but keep it tightly bounded (CSV applicability is expensive).
                    #
                    # Rationale: the strict CSV gate + survivor cap keeps branching bounded, but we
                    # still need enough coverage so the "right" CSG has a chance to be considered.
                    pre_cap = int(max(0, int(concept_max_templates)))
                    if bool(abstraction_pressure):
                        # Keep the pre-gate pool modest: under tight timeouts we cannot afford to
                        # scan hundreds of CSGs per state. Scale it with the desired survivors.
                        pre_cap = max(int(pre_cap), max(32, 2 * int(concept_max_templates)))
                    task_key_v1 = task_feat_key_v1
                    task_key_v2 = task_feat_key_v2

                    def _task_feat_dist_v1(a: Tuple[str, str, str], b: Tuple[str, str, str]) -> int:
                        d = 0
                        if str(a[0]) != str(b[0]):
                            d += 2
                        if str(a[1]) != str(b[1]):
                            d += 1
                        if str(a[2]) != str(b[2]):
                            d += 1
                        return int(d)

                    def _task_feat_dist_v2(a: Tuple[str, ...], b: Tuple[str, ...]) -> int:
                        # Weighted, low-cardinality tuple match: earlier slots are more important.
                        # Tuple schema (v2): (shape_rel, palette_rel, delta_density, train_n, in_colors, out_colors, in_dim, out_dim)
                        w = (3, 2, 1, 1, 1, 1, 1, 1)
                        d = 0
                        n = min(int(len(a)), int(len(b)), int(len(w)))
                        for i in range(int(n)):
                            if str(a[i]) != str(b[i]):
                                d += int(w[int(i)])
                        # Penalize length mismatch (defensive).
                        d += abs(int(len(a)) - int(len(b)))
                        return int(d)

                    def _tmpl_task_dist(*, cid: str, tmpl: Dict[str, Any]) -> int:
                        if not bool(enable_concept_support_feat_ranking):
                            return 0
                        hit2 = concept_support_task_feat_cache.get(str(cid))
                        if hit2 is None:
                            feats: Set[Tuple[str, ...]] = set()
                            raw2 = tmpl.get("support_task_feat_keys_v2")
                            if isinstance(raw2, list):
                                for row in raw2[:64]:
                                    if not isinstance(row, (list, tuple)) or len(row) < 3:
                                        continue
                                    feats.add(tuple(str(x) for x in row))
                            if not feats:
                                raw = tmpl.get("support_task_feat_keys")
                                if isinstance(raw, list):
                                    # Deterministic cap to keep selection time bounded.
                                    for row in raw[:64]:
                                        if not isinstance(row, (list, tuple)) or len(row) != 3:
                                            continue
                                        feats.add((str(row[0]), str(row[1]), str(row[2])))
                            hit2 = tuple(sorted(feats))
                            concept_support_task_feat_cache[str(cid)] = hit2
                        if not hit2:
                            return 9
                        use_v2 = False
                        try:
                            use_v2 = int(len(hit2[0])) >= 8
                        except Exception:
                            use_v2 = False
                        best = 99
                        for fk in hit2:
                            if bool(use_v2):
                                best = min(int(best), int(_task_feat_dist_v2(task_key_v2, fk)))
                            else:
                                fk3 = (str(fk[0]) if len(fk) > 0 else "", str(fk[1]) if len(fk) > 1 else "", str(fk[2]) if len(fk) > 2 else "")
                                best = min(int(best), int(_task_feat_dist_v1(task_key_v1, fk3)))
                            if int(best) <= 0:
                                break
                        return int(best)
                    k = (
                        str(task_key_v2[0]),
                        str(task_key_v2[1]),
                        str(task_key_v2[2]),
                        str(task_key_v2[3]),
                        str(task_key_v2[4]),
                        str(task_key_v2[5]),
                        str(task_key_v2[6]),
                        str(task_key_v2[7]),
                        str(mm_kind),
                        str(stage),
                        1 if bool(avail.get("objset", False)) else 0,
                        1 if bool(avail.get("obj", False)) else 0,
                        1 if bool(avail.get("bbox", False)) else 0,
                        1 if bool(avail.get("patch", False)) else 0,
                        int(pre_cap),
                    )
                    hit = concept_template_select_cache.get(k)
                    if hit is not None:
                        return hit
                    cand: List[Tuple[int, int, int, int, int, int, int, int, str, Dict[str, Any]]] = []
                    for tmpl in concept_templates_norm:
                        cid0 = str(tmpl.get("concept_id") or "")
                        if not cid0:
                            continue
                        # Fast feasibility prefilter: CSG templates mined as mid-pipeline closures
                        # may require internal slots (objset/obj/bbox/patch). Skip them early if
                        # the current state cannot satisfy the declared requirements.
                        req0 = tmpl.get("inputs_required")
                        if isinstance(req0, list) and req0:
                            ok_req = True
                            for slot0 in req0:
                                s0 = str(slot0)
                                if s0 in {"grid", "orig"}:
                                    continue
                                if not bool(avail.get(s0, False)):
                                    ok_req = False
                                    break
                            if not bool(ok_req):
                                continue
                        op0 = _tmpl_primary_op_id(tmpl)
                        if not op0:
                            continue
                        steps_payload0 = tmpl.get("steps")
                        is_concrete = 0 if isinstance(steps_payload0, list) and bool(steps_payload0) else 1
                        if bool(abstraction_pressure) and int(is_concrete) != 0:
                            # Under abstraction pressure, op_id-only templates reintroduce
                            # arg branching and behave like weak macros. Prefer true CSGs.
                            continue
                        binder_count0 = 0
                        binder_score0 = 0
                        if isinstance(steps_payload0, list) and bool(steps_payload0):
                            for row0 in steps_payload0:
                                if not isinstance(row0, dict):
                                    continue
                                a0 = row0.get("args") if isinstance(row0.get("args"), dict) else {}
                                for v0 in a0.values():
                                    if isinstance(v0, dict) and ("bind" in v0 or "infer" in v0):
                                        binder_count0 += 1
                                        bnm = v0.get("bind") if "bind" in v0 else v0.get("infer")
                                        if bnm is not None and str(bnm) != "bg":
                                            binder_score0 += 1
                        if not _tmpl_first_op_feasible(op0):
                            continue
                        op_ids0 = tmpl.get("op_ids") if isinstance(tmpl.get("op_ids"), list) else []
                        tmpl_len = int(len(op_ids0))
                        sup0 = int(tmpl.get("support") or 0)
                        cb0 = tmpl.get("cost_bits")
                        cost_bits0 = 10
                        if isinstance(cb0, int):
                            cost_bits0 = int(cb0)
                        elif isinstance(cb0, str) and str(cb0).isdigit():
                            cost_bits0 = int(str(cb0))
                        util0 = 0
                        uv0 = tmpl.get("utility_train_loss_drop_mean")
                        if isinstance(uv0, (int, float)):
                            util0 = int(float(uv0) * 1000.0)
                        elif isinstance(uv0, str):
                            try:
                                util0 = int(float(str(uv0)) * 1000.0)
                            except Exception:
                                util0 = 0
                        cand.append(
                            (
                                int(_tmpl_policy_rank(str(op0))),
                                int(_tmpl_task_dist(cid=str(cid0), tmpl=tmpl)),
                                int(is_concrete),
                                -int(util0),
                                -int(sup0),
                                -int(binder_score0),
                                -int(tmpl_len),
                                int(cost_bits0),
                                str(cid0),
                                tmpl,
                            )
                        )
                    cand.sort(key=lambda x: x[:9])
                    # Diversify by primary op_id so tight caps don't crowd out the tail.
                    max_per_primary = 24
                    if bool(abstraction_pressure):
                        max_per_primary = 12
                    out2: List[Dict[str, Any]] = []
                    seen_cids: Set[str] = set()
                    per_primary: Dict[str, int] = {}
                    for row in cand:
                        tmpl = row[-1]
                        cid0 = str(tmpl.get("concept_id") or "")
                        if cid0 and cid0 in seen_cids:
                            continue
                        op0 = str(_tmpl_primary_op_id(tmpl) or "")
                        if int(per_primary.get(op0, 0)) >= int(max_per_primary):
                            continue
                        out2.append(tmpl)
                        if cid0:
                            seen_cids.add(cid0)
                        per_primary[op0] = int(per_primary.get(op0, 0)) + 1
                        if int(len(out2)) >= int(pre_cap):
                            break
                    if int(len(out2)) < int(pre_cap):
                        for row in cand:
                            tmpl = row[-1]
                            cid0 = str(tmpl.get("concept_id") or "")
                            if cid0 and cid0 in seen_cids:
                                continue
                            out2.append(tmpl)
                            if cid0:
                                seen_cids.add(cid0)
                            if int(len(out2)) >= int(pre_cap):
                                break
                    out = list(out2)[: int(pre_cap)]
                    concept_template_select_cache[k] = out
                    return out

                def _concrete_steps_feasible_v141(inner_steps: Sequence[Dict[str, Any]]) -> bool:
                    cur = {
                        "grid": True,
                        "objset": bool(avail.get("objset", False)),
                        "obj": bool(avail.get("obj", False)),
                        "bbox": bool(avail.get("bbox", False)),
                        "patch": bool(avail.get("patch", False)),
                    }
                    for row in inner_steps:
                        if not isinstance(row, dict):
                            return False
                        op0 = str(row.get("op_id") or "")
                        if not op0:
                            return False
                        od0 = OP_DEFS_V141.get(op0)
                        if od0 is None:
                            return False
                        for r in od0.reads:
                            if not bool(cur.get(str(r), False)):
                                return False
                        for w in od0.writes:
                            cur[str(w)] = True
                    return True

                max_applicable_templates = 0
                if bool(abstraction_pressure):
                    # Soft stop: once we have enough applicable concepts, stop scanning more
                    # templates (we will cap survivors anyway). This prevents "proposal time"
                    # from dominating task timeout.
                    # NOTE: Keep this high enough that we don't miss the relevant CSG family
                    # when many templates are locally applicable but only a few are truly useful.
                    # The downstream survivor cap (dl-ranked) still bounds branching.
                    max_applicable_templates = 24

                applicable_templates = 0
                for tmpl in _select_concept_templates():
                    cid = str(tmpl.get("concept_id") or "")
                    op_ids = tmpl.get("op_ids") if isinstance(tmpl.get("op_ids"), list) else []
                    tmpl_len = int(len(op_ids))
                    steps_payload = tmpl.get("steps")
                    cb = tmpl.get("cost_bits")
                    cost_bits = 10
                    if isinstance(cb, int):
                        cost_bits = int(cb)
                    elif isinstance(cb, str) and str(cb).isdigit():
                        cost_bits = int(str(cb))
                    # Concrete CSG templates (arc_concept_csg_v148) carry explicit steps with args:
                    # they do not branch over instantiations. Wrap as-is.
                    if isinstance(steps_payload, list) and steps_payload:
                        inner: List[Dict[str, Any]] = []
                        ok = True
                        has_binder = False
                        for row in steps_payload:
                            if not isinstance(row, dict):
                                ok = False
                                break
                            op0 = str(row.get("op_id") or "")
                            if not op0 or op0 in {"macro_call", "concept_call"}:
                                ok = False
                                break
                            a0 = row.get("args") if isinstance(row.get("args"), dict) else {}
                            a2: Dict[str, Any] = {str(k): a0[k] for k in sorted(a0.keys())}
                            for v in a2.values():
                                if isinstance(v, dict) and ("bind" in v or "infer" in v):
                                    has_binder = True
                                    break
                            inner.append({"op_id": str(op0), "args": dict(a2)})
                        if ok and inner:
                            inner2 = list(inner)
                            if has_binder:
                                if str(cid) in concept_csg_resolve_cache:
                                    resolved = concept_csg_resolve_cache.get(str(cid))
                                else:
                                    resolved = _resolve_concept_csg_binders_v141(
                                        steps=inner, train_pairs=train_pairs, test_in=test_in
                                    )
                                    concept_csg_resolve_cache[str(cid)] = list(resolved) if resolved is not None else None
                                    resolved = concept_csg_resolve_cache.get(str(cid))
                                if resolved is None:
                                    continue
                                inner2 = list(resolved)
                            if not _concrete_steps_feasible_v141(inner2):
                                continue
                            # CSV v1 gate: concept must be causally applicable NOW (strictly reduce TRAIN loss)
                            # or it is not permitted to enter the search frontier.
                            app_key = (str(ev.vec_sig), str(cid))
                            if app_key not in csv_applicable_cache:
                                metrics["csv_tested"] = int(metrics.get("csv_tested", 0)) + 1
                                app = tmpl.get("applicability") if isinstance(tmpl.get("applicability"), dict) else {}
                                delta_max = -1
                                if isinstance(app, dict):
                                    v = app.get("train_loss_delta_max")
                                    if isinstance(v, int):
                                        delta_max = int(v)
                                    elif isinstance(v, str) and str(v).lstrip("-").isdigit():
                                        delta_max = int(str(v))
                                score = _csv_applicability_score_v141(
                                    steps=inner2,
                                    train_final_states=ev.train_final_states,
                                    train_pairs=train_pairs,
                                    apply_cache=apply_cache,
                                    grid_hash_cache=grid_hash_cache,
                                    metrics=metrics,
                                    allow_slot_progress=bool(csv_allow_slot_progress),
                                )
                                ok = False
                                dl = 0
                                ds = 0
                                if score is not None:
                                    dl, ds = (int(score[0]), int(score[1]))
                                    ok = _csv_gate_ok(dl=int(dl), ds=int(ds), delta_max=int(delta_max))
                                if not bool(ok):
                                    metrics["csv_rejected"] = int(metrics.get("csv_rejected", 0)) + 1
                                csv_applicable_cache[app_key] = (bool(ok), int(dl), int(ds))
                            ok0, _dl0, _ds0 = csv_applicable_cache.get(app_key, (False, 0, 0))
                            if not bool(ok0):
                                continue
                            metrics["csv_survivors"] = int(metrics.get("csv_survivors", 0)) + 1
                            next_steps.append(
                                ProgramStepV141(
                                    op_id="concept_call",
                                    args={
                                        "concept_id": str(cid),
                                        "cost_bits": int(cost_bits),
                                        "op_ids": [str(x) for x in op_ids if str(x)],
                                        "steps": list(inner2),
                                    },
                                )
                            )
                            applicable_templates += 1
                            if int(max_applicable_templates) > 0 and int(applicable_templates) >= int(
                                max_applicable_templates
                            ):
                                break
                            continue
                    # Under abstraction pressure, prohibit non-concrete concept templates.
                    # Templates that only carry op_ids require re-instantiation/arg branching,
                    # which reintroduces global search as policy and prevents the "CSV as atom"
                    # regime from closing.
                    if bool(abstraction_pressure):
                        continue
                    # Instantiation budget: longer concepts compress depth but explode branching.
                    # Allocate fewer instantiations/arg branches to long templates so they remain
                    # selectable under tight caps (and don't crowd out all other next-steps).
                    max_inst = int(concept_max_instantiations)
                    max_branch = int(concept_max_branch_per_op)
                    if tmpl_len >= 5:
                        max_inst = max(2, int(max_inst) // 6)
                        max_branch = max(3, int(max_branch) // 3)
                    elif tmpl_len == 4:
                        max_inst = max(2, int(max_inst) // 4)
                        max_branch = max(4, int(max_branch) // 2)
                    elif tmpl_len == 3:
                        max_inst = max(3, int(max_inst) // 2)
                    # Under abstraction pressure, avoid turning op_id-only concepts into an inner
                    # arg-search: prefer concrete (steps-carrying) CSG templates and keep
                    # instantiation branching low and deterministic.
                    if bool(abstraction_pressure):
                        # Do not expand op_id-only templates under pressure: they would branch
                        # over args and defeat the purpose of concept_call as an atomic policy.
                        continue
                        max_inst = min(int(max_inst), 2)
                        max_branch = min(int(max_branch), 2)
                    insts = _instantiate_concept_steps_v141(
                        concept_op_ids=[str(x) for x in op_ids if str(x)],
                        steps_prefix=steps,
                        train_pairs=train_pairs,
                        test_in=test_in,
                        bg_candidates=bg_candidates,
                        shapes_out=shapes_out,
                        palette_out=palette_out,
                        direct_steps=direct_steps,
                        max_instantiations=int(max_inst),
                        max_branch_per_op=int(max_branch),
                        get_candidates=_get_candidates_cached_v141,
                    )
                    for inner_steps in insts:
                        next_steps.append(
                            ProgramStepV141(
                                op_id="concept_call",
                                args={
                                    "concept_id": str(cid),
                                    "cost_bits": int(cost_bits),
                                    "op_ids": [str(x) for x in op_ids if str(x)],
                                    "steps": [s.to_dict() for s in inner_steps],
                                },
                            )
                        )

            # Under abstraction pressure, enforce a small "CSV survivor" cap per-state.
            #
            # Rationale: after the strict CSV gate, many templates can still be applicable and
            # reduce TRAIN loss, but proposing all of them reintroduces combinatorial search.
            # We keep only the best few by (train_loss_delta, cost_bits, concept_id).
            if bool(abstraction_pressure):
                csteps = [st for st in next_steps if str(st.op_id or "") == "concept_call"]
                if csteps:
                    # Under strict concept-as-policy, keep the per-state branching factor small.
                    # This forces "hypothesis selection" to happen via applicability/utility,
                    # rather than via exponential search over many concepts.
                    cap = 4
                    if int(concept_max_templates) > 0:
                        cap = max(2, min(6, int(int(concept_max_templates) // 6) or 4))

                    ranked_c: List[Tuple[int, int, int, str, ProgramStepV141]] = []
                    for st in csteps:
                        args0 = st.args if isinstance(st.args, dict) else {}
                        cid0 = str(args0.get("concept_id") or "")
                        app_key = (str(ev.vec_sig), str(cid0))
                        _ok, dl0, ds0 = csv_applicable_cache.get(app_key, (False, 0, 0))
                        cb0 = args0.get("cost_bits")
                        cost0 = 10
                        if isinstance(cb0, int):
                            cost0 = int(cb0)
                        elif isinstance(cb0, str) and str(cb0).isdigit():
                            cost0 = int(str(cb0))
                        ranked_c.append((int(dl0), int(ds0), int(cost0), str(cid0), st))

                    ranked_c.sort(key=lambda x: x[:4])
                    best_dl = int(ranked_c[0][0])
                    # When no immediate loss-reducing semantic action exists (best_dl>=0),
                    # widen the survivor cap so the solver can explore multiple slot-building
                    # hypotheses (e.g., different mask_by_color colors) deterministically.
                    if int(best_dl) >= 0:
                        cap = max(int(cap), 12)
                    keep = [st for *_k, st in ranked_c[: int(min(int(cap), len(ranked_c)))]]
                    if len(keep) < len(csteps):
                        metrics["pruned_by_next_steps_cap"] = int(metrics.get("pruned_by_next_steps_cap", 0)) + int(
                            len(csteps) - len(keep)
                        )
                    next_steps = [st for st in next_steps if str(st.op_id or "") != "concept_call"] + list(keep)

            # Under abstraction pressure, enforce "concept-as-policy": if any concrete concept_call
            # is available, do not propose primitive (or macro) steps at that state.
            #
            # Rationale: without this, the search reverts to primitives after the first concept,
            # reintroducing global branching and SEARCH_BUDGET_EXCEEDED. A concept must be the
            # dominant policy whenever it is causally applicable (CSV gate already enforces
            # strict TRAIN-loss improvement).
            if bool(abstraction_pressure) and next_steps:
                csteps = [st for st in next_steps if str(st.op_id or "") == "concept_call"]
                if csteps:
                    next_steps = csteps

            # Partial-order reduction (generic, deterministic):
            # While a patch exists, restrict expansion to steps that read it.
            #
            # Rationale: any step that doesn't read patch (e.g., grid-only rotate/translate) commutes
            # with patch construction and would invalidate patch anyway; exploring it *before* the
            # patch is consumed explodes branching without increasing expressivity.
            if bool(avail.get("patch", False)) and next_steps:
                patch_readers: List[ProgramStepV141] = []

                def _step_reads_slots(step: ProgramStepV141) -> Set[str]:
                    op0 = str(step.op_id or "")
                    if op0 in {"macro_call", "concept_call"}:
                        reads: Set[str] = set()
                        steps_raw = step.args.get("steps")
                        if isinstance(steps_raw, list):
                            for row in steps_raw:
                                if not isinstance(row, dict):
                                    continue
                                op2 = str(row.get("op_id") or "")
                                od2 = OP_DEFS_V141.get(op2)
                                if od2 is None:
                                    continue
                                for r in od2.reads:
                                    reads.add(str(r))
                        return reads
                    od = OP_DEFS_V141.get(op0)
                    if od is None:
                        return set()
                    return set(str(r) for r in od.reads)

                for st in next_steps:
                    if "patch" in _step_reads_slots(st):
                        patch_readers.append(st)
                if patch_readers:
                    next_steps = patch_readers

            def _inner_primary_op_id(step: ProgramStepV141) -> str:
                op0 = str(step.op_id or "")
                if op0 in {"macro_call", "concept_call"}:
                    steps_raw = step.args.get("steps")
                    if isinstance(steps_raw, list) and steps_raw and isinstance(steps_raw[0], dict):
                        op2 = str(steps_raw[0].get("op_id") or "")
                        if op2:
                            return str(op2)
                    op_ids_raw = step.args.get("op_ids")
                    if isinstance(op_ids_raw, list) and op_ids_raw:
                        for x in op_ids_raw:
                            if str(x):
                                return str(x)
                return op0

            def _step_policy_rank(step: ProgramStepV141) -> int:
                op = _inner_primary_op_id(step)
                # Stage-critical moves first (if the program has made those slots available).
                rank = 5
                if op == "commit_patch" and bool(avail.get("patch", False)):
                    rank = 0
                elif op in {
                    "patch_rotate90",
                    "patch_rotate180",
                    "patch_rotate270",
                    "patch_reflect_h",
                    "patch_reflect_v",
                    "patch_transpose",
                    "patch_translate",
                } and bool(avail.get("patch", False)):
                    rank = 1
                elif op in {"mask_outline", "mask_dilate", "mask_box_dilate"} and bool(avail.get("patch", False)):
                    rank = 2
                elif op in {"mask_by_color", "mask_cross_center"} and not bool(avail.get("patch", False)):
                    rank = 2
                elif op == "crop_bbox" and bool(avail.get("bbox", False)) and not bool(avail.get("patch", False)):
                    rank = 0
                elif op == "obj_bbox" and bool(avail.get("obj", False)) and not bool(avail.get("bbox", False)):
                    rank = 1
                elif op == "obj_patch" and bool(avail.get("obj", False)) and not bool(avail.get("patch", False)):
                    rank = 1
                elif op == "select_obj" and bool(avail.get("objset", False)) and not bool(avail.get("obj", False)):
                    rank = 1
                elif op in {"cc4", "cc4_nonbg_multicolor"} and not bool(avail.get("objset", False)):
                    rank = 1
                elif op == "bbox_by_color" and not bool(avail.get("bbox", False)):
                    rank = 1

                global_xform_ops = {"rotate90", "rotate180", "rotate270", "transpose", "reflect_h", "reflect_v"}
                shape_ops = {
                    "crop_bbox_nonzero",
                    "pad_to",
                    "new_canvas",
                    "repeat_grid",
                    "downsample_mode",
                    "cc4_color_bars",
                    "cc4_color_area_column",
                    "cc4_nonbg_bfs_column",
                }
                color_ops = {
                    "replace_color",
                    "map_colors",
                    "paint_rect",
                    "draw_rect_border",
                    "fill_enclosed_region",
                    "symmetry_fill_h",
                    "symmetry_fill_v",
                    "paint_mask",
                    "flood_fill",
                }
                spatial_ops = {"translate", "paste", "gravity"}
                special_ops = {"overlay_self_translate", "propagate_color_translate", "propagate_nonbg_translate"}

                if mm_kind == "shape_mismatch":
                    if op in shape_ops:
                        rank = 2
                    elif op in global_xform_ops:
                        rank = 3
                    elif op in spatial_ops:
                        rank = 4
                    elif op in color_ops:
                        rank = 6
                    elif op in special_ops:
                        rank = 7
                    else:
                        rank = 8

                if mm_kind == "cell_mismatch":
                    # Always keep simple global transforms early: they solve many tasks in 1 step.
                    if op in global_xform_ops:
                        rank = 2
                    elif op in spatial_ops:
                        rank = 3
                    elif op in color_ops:
                        rank = 4
                    elif op in special_ops:
                        rank = 6
                    elif op in shape_ops:
                        rank = 7
                    else:
                        rank = 7

                # Under abstraction pressure, give a small deterministic boost to multi-step
                # non-primitive CSG concepts so they can crowd out weak primitive wrappers.
                if bool(abstraction_pressure) and str(step.op_id or "") == "concept_call":
                    a0 = step.args if isinstance(step.args, dict) else {}
                    cid = str(a0.get("concept_id") or "")
                    steps_raw = a0.get("steps")
                    if not cid.startswith("csg_prim_") and isinstance(steps_raw, list):
                        clen = int(len(steps_raw))
                        if clen >= 3:
                            rank = max(0, int(rank) - 1)

                return int(rank)

            def _one_step_train_loss(step: ProgramStepV141) -> Tuple[int, int, int]:
                # Generic incremental lookahead to keep the branching cap semantically meaningful.
                # Deterministic: apply the candidate step to the per-pair final states of the prefix.
                loss_shape = 0
                loss_cells = 0
                slot_pen = 0
                for st0, (_inp, want) in zip(ev.train_final_states, train_pairs):
                    try:
                        st1 = _apply_step_cached_v141(
                            state=st0,
                            op_id=str(step.op_id),
                            args=dict(step.args),
                            apply_cache=apply_cache,
                            grid_hash_cache=grid_hash_cache,
                            metrics=metrics,
                        )
                    except Exception:
                        return (10**9, 10**9, 10**9)
                    got = st1.grid
                    if not grid_equal_v124(got, want):
                        mm = _summarize_mismatch_v141(got=got, want=want)
                        if mm.get("kind") == "shape_mismatch":
                            loss_shape += 1
                        elif mm.get("kind") == "cell_mismatch":
                            loss_cells += int(mm.get("diff_cells") or 0)
                    slot_pen += int(
                        (1 if st1.objset is None else 0)
                        + (1 if st1.obj is None else 0)
                        + (1 if st1.bbox is None else 0)
                        + (1 if st1.patch is None else 0)
                    )
                return (int(loss_shape), int(loss_cells), int(slot_pen))

            def _compression_len(step: ProgramStepV141) -> int:
                if not bool(abstraction_pressure):
                    return 0
                op0 = str(step.op_id or "")
                if op0 not in {"macro_call", "concept_call"}:
                    return 0
                steps_raw = step.args.get("steps")
                if not isinstance(steps_raw, list):
                    return 0
                return int(len(steps_raw))

            def _step_writes_grid(step: ProgramStepV141) -> bool:
                op0 = str(step.op_id or "")
                if op0 in {"macro_call", "concept_call"}:
                    return True
                od0 = OP_DEFS_V141.get(op0)
                if od0 is None:
                    return False
                writes = tuple(getattr(od0, "writes", ()) or ())
                return "grid" in {str(x) for x in writes if str(x)}

            def _step_requires_train_improve(step: ProgramStepV141) -> bool:
                """
                Under abstraction pressure, we enforce *causal utility gates* on semantic actions
                (concept_call/macro_call): they must improve TRAIN loss at the current prefix.

                We intentionally do NOT require this for primitive grid-writing steps: many ARC
                solutions need intermediate state changes that do not monotonically reduce loss.
                Search still prefers improving primitives via ranking, but is allowed to explore
                non-improving intermediates when necessary.
                """
                op0 = str(step.op_id or "")
                return op0 in {"macro_call", "concept_call"}

            def _csv_cached_delta_v141(step: ProgramStepV141) -> Optional[Tuple[int, int]]:
                """
                Fast path: retrieve cached CSV applicability deltas for semantic steps.

                Under abstraction pressure, all concept/macro calls are supposed to be CSV-gated
                at generation time; ranking can then avoid re-evaluating TRAIN loss for ordering.
                """
                if not bool(abstraction_pressure):
                    return None
                op0 = str(step.op_id or "")
                if op0 not in {"macro_call", "concept_call"}:
                    return None
                a0 = step.args if isinstance(step.args, dict) else {}
                vec = str(ev.vec_sig)
                cid = str(a0.get("concept_id") or "")
                if cid:
                    v0 = csv_applicable_cache.get((str(vec), str(cid)))
                    if v0 is not None:
                        ok0, dl0, ds0 = v0
                        if bool(ok0):
                            return (int(dl0), int(ds0))
                steps_payload = a0.get("steps")
                if isinstance(steps_payload, list) and steps_payload:
                    h = sha256_hex(canonical_json_dumps(steps_payload).encode("utf-8"))[:16]
                    for pref in ("prim_", "macro_"):
                        v1 = csv_applicable_cache.get((str(vec), str(pref) + str(h)))
                        if v1 is not None:
                            ok1, dl1, ds1 = v1
                            if bool(ok1):
                                return (int(dl1), int(ds1))
                return None

            fast_ranked = False
            if bool(abstraction_pressure) and next_steps:
                all_semantic = all(str(st.op_id or "") in {"macro_call", "concept_call"} for st in next_steps)
                if bool(all_semantic):
                    ranked_fast: List[Tuple[int, int, int, int, int, str, ProgramStepV141]] = []
                    ok_all = True
                    for st in next_steps:
                        d = _csv_cached_delta_v141(st)
                        if d is None:
                            ok_all = False
                            break
                        ranked_fast.append(
                            (
                                int(_step_policy_rank(st)),
                                int(d[0]),
                                int(d[1]),
                                -int(_compression_len(st)),
                                int(_step_cost_bits_total_v141(step=st, macro_support_by_id=macro_support_by_id)),
                                canonical_json_dumps(st.to_dict()),
                                st,
                            )
                        )
                    if bool(ok_all):
                        ranked_fast.sort(key=lambda t: (int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4]), str(t[5])))
                        next_steps = [t[-1] for t in ranked_fast]
                        metrics["csv_fast_rank"] = int(metrics.get("csv_fast_rank", 0)) + 1
                        fast_ranked = True

            # Policy ranking always applies; when we hit the branching cap, use one-step train-loss
            # to avoid truncating away the only viable next move.
            if not bool(fast_ranked) and len(next_steps) > max_next_steps:
                # Under abstraction pressure, avoid recomputing expensive one-step loss for
                # already CSV-gated semantic steps (concept/macro calls). Rank them by cached
                # CSV deltas and only compute one-step loss for the remaining primitive candidates.
                if bool(abstraction_pressure):
                    sem_rank: List[Tuple[int, int, int, int, int, str, ProgramStepV141]] = []
                    prim: List[ProgramStepV141] = []
                    for st in next_steps:
                        op0 = str(st.op_id or "")
                        if op0 in {"macro_call", "concept_call"}:
                            d = _csv_cached_delta_v141(st)
                            if d is None:
                                prim.append(st)
                                continue
                            sem_rank.append(
                                (
                                    int(_step_policy_rank(st)),
                                    int(d[0]),
                                    int(d[1]),
                                    -int(_compression_len(st)),
                                    int(_step_cost_bits_total_v141(step=st, macro_support_by_id=macro_support_by_id)),
                                    canonical_json_dumps(st.to_dict()),
                                    st,
                                )
                            )
                        else:
                            prim.append(st)
                    sem_rank.sort(key=lambda t: (int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4]), str(t[5])))
                    sem_steps = [t[-1] for t in sem_rank]
                    if len(sem_steps) >= int(max_next_steps):
                        next_steps = sem_steps[: int(max_next_steps)]
                    else:
                        remain = int(max_next_steps) - int(len(sem_steps))
                        # Avoid scoring thousands of candidates with expensive evals: first prefilter by cheap,
                        # deterministic policy rank + cost, then compute one-step loss only for the top slice.
                        prim.sort(
                            key=lambda st: (
                                int(_step_policy_rank(st)),
                                -int(_compression_len(st)),
                                int(_step_cost_bits_total_v141(step=st, macro_support_by_id=macro_support_by_id)),
                                canonical_json_dumps(st.to_dict()),
                            )
                        )
                        pre_k = max(int(remain) * 2, int(remain) + 1)
                        cand_steps = prim[: int(pre_k)]
                        ranked: List[Tuple[Tuple[int, int, int], int, int, int, str, ProgramStepV141]] = []
                        for st in cand_steps:
                            ranked.append(
                                (
                                    _one_step_train_loss(st),
                                    int(_step_policy_rank(st)),
                                    -int(_compression_len(st)),
                                    int(_step_cost_bits_total_v141(step=st, macro_support_by_id=macro_support_by_id)),
                                    canonical_json_dumps(st.to_dict()),
                                    st,
                                )
                            )
                        ranked.sort(
                            key=lambda x: (
                                int(x[0][0]),
                                int(x[0][1]),
                                int(x[0][2]),
                                int(x[1]),
                                int(x[2]),
                                int(x[3]),
                                str(x[4]),
                            )
                        )
                        next_steps = sem_steps + [st for _loss, _r, _comp, _c, _js, st in ranked[: int(remain)]]
                    metrics["csv_semantic_split_rank"] = int(metrics.get("csv_semantic_split_rank", 0)) + 1
                else:
                    # Avoid scoring thousands of candidates with expensive evals: first prefilter by cheap,
                    # deterministic policy rank + cost, then compute one-step loss only for the top slice.
                    next_steps.sort(
                        key=lambda st: (
                            int(_step_policy_rank(st)),
                            -int(_compression_len(st)),
                            int(_step_cost_bits_total_v141(step=st, macro_support_by_id=macro_support_by_id)),
                            canonical_json_dumps(st.to_dict()),
                        )
                    )
                    # Keep the one-step-loss lookahead cheap: evaluating it is expensive (it applies the
                    # step to every train final-state). Using a very large prefilter slice can dominate
                    # wall time and starve the actual search, causing harness timeouts and trace loss.
                    pre_k = max(int(max_next_steps) * 2, int(max_next_steps) + 1)
                    cand_steps = next_steps[: int(pre_k)]
                    ranked = []
                    for st in cand_steps:
                        ranked.append(
                            (
                                _one_step_train_loss(st),
                                int(_step_policy_rank(st)),
                                -int(_compression_len(st)),
                                int(_step_cost_bits_total_v141(step=st, macro_support_by_id=macro_support_by_id)),
                                canonical_json_dumps(st.to_dict()),
                                st,
                            )
                        )
                    ranked.sort(
                        key=lambda x: (int(x[0][0]), int(x[0][1]), int(x[0][2]), int(x[1]), int(x[2]), int(x[3]), str(x[4]))
                    )
                    # Under abstraction pressure, forbid *semantic* actions (concept/macro calls) that
                    # do not improve TRAIN loss at the current prefix state.
                    if bool(abstraction_pressure):
                        base_slot = 0
                        for st0 in ev.train_final_states:
                            base_slot += int(
                                (1 if st0.objset is None else 0)
                                + (1 if st0.obj is None else 0)
                                + (1 if st0.bbox is None else 0)
                                + (1 if st0.patch is None else 0)
                            )
                        base_loss = (int(ev.loss[0]), int(ev.loss[1]), int(base_slot))
                        filtered = [t for t in ranked if (not _step_requires_train_improve(t[-1])) or (t[0] < base_loss)]
                        if filtered:
                            ranked = filtered
                    metrics["pruned_by_next_steps_cap"] = int(metrics.get("pruned_by_next_steps_cap", 0)) + int(
                        len(next_steps) - max_next_steps
                    )
                    next_steps = [st for _loss, _r, _comp, _c, _js, st in ranked[:max_next_steps]]
            elif not bool(fast_ranked):
                if bool(abstraction_pressure):
                    sem_rank: List[Tuple[int, int, int, int, int, str, ProgramStepV141]] = []
                    prim: List[ProgramStepV141] = []
                    for st in next_steps:
                        op0 = str(st.op_id or "")
                        if op0 in {"macro_call", "concept_call"}:
                            d = _csv_cached_delta_v141(st)
                            if d is None:
                                prim.append(st)
                                continue
                            sem_rank.append(
                                (
                                    int(_step_policy_rank(st)),
                                    int(d[0]),
                                    int(d[1]),
                                    -int(_compression_len(st)),
                                    int(_step_cost_bits_total_v141(step=st, macro_support_by_id=macro_support_by_id)),
                                    canonical_json_dumps(st.to_dict()),
                                    st,
                                )
                            )
                        else:
                            prim.append(st)
                    sem_rank.sort(key=lambda t: (int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4]), str(t[5])))
                    sem_steps = [t[-1] for t in sem_rank]
                    remain = int(max_next_steps) - int(len(sem_steps))
                    if remain <= 0:
                        next_steps = sem_steps[: int(max_next_steps)]
                    else:
                        ranked_p: List[Tuple[Tuple[int, int, int], int, int, int, str, ProgramStepV141]] = []
                        for st in prim:
                            ranked_p.append(
                                (
                                    _one_step_train_loss(st),
                                    int(_step_policy_rank(st)),
                                    -int(_compression_len(st)),
                                    int(_step_cost_bits_total_v141(step=st, macro_support_by_id=macro_support_by_id)),
                                    canonical_json_dumps(st.to_dict()),
                                    st,
                                )
                            )
                        ranked_p.sort(
                            key=lambda x: (
                                int(x[1]),
                                int(x[0][0]),
                                int(x[0][1]),
                                int(x[0][2]),
                                int(x[2]),
                                int(x[3]),
                                str(x[4]),
                            )
                        )
                        next_steps = sem_steps + [st for _loss, _r, _comp, _c, _js, st in ranked_p[: int(remain)]]
                    metrics["csv_semantic_split_rank"] = int(metrics.get("csv_semantic_split_rank", 0)) + 1
                else:
                    # Even when we don't exceed the branching cap, prefer steps that reduce TRAIN loss
                    # at the current prefix state. This keeps search "hypothesis-driven" rather than
                    # "try until timeout", and is critical under tight budgets.
                    ranked2: List[Tuple[Tuple[int, int, int], int, int, int, str, ProgramStepV141]] = []
                    for st in next_steps:
                        ranked2.append(
                            (
                                _one_step_train_loss(st),
                                int(_step_policy_rank(st)),
                                -int(_compression_len(st)),
                                int(_step_cost_bits_total_v141(step=st, macro_support_by_id=macro_support_by_id)),
                                canonical_json_dumps(st.to_dict()),
                                st,
                            )
                        )
                    ranked2.sort(
                        key=lambda x: (int(x[1]), int(x[0][0]), int(x[0][1]), int(x[0][2]), int(x[2]), int(x[3]), str(x[4]))
                    )
                    if bool(abstraction_pressure):
                        base_slot = 0
                        for st0 in ev.train_final_states:
                            base_slot += int(
                                (1 if st0.objset is None else 0)
                                + (1 if st0.obj is None else 0)
                                + (1 if st0.bbox is None else 0)
                                + (1 if st0.patch is None else 0)
                            )
                        base_loss = (int(ev.loss[0]), int(ev.loss[1]), int(base_slot))
                        filtered2 = [t for t in ranked2 if (not _step_requires_train_improve(t[-1])) or (t[0] < base_loss)]
                        ranked2 = filtered2 if filtered2 else ranked2
                    next_steps = [st for _loss, _r, _comp, _c, _js, st in ranked2]
            if len(next_steps) > max_next_steps:
                next_steps = next_steps[:max_next_steps]
            # Telemetry: if we timeout before exploring >1 program, it's often because evaluating
            # and pushing the root children consumes the whole wall-time cap. Track this deterministically.
            t_push0: Optional[float] = None
            pushed = 0
            if int(programs_explored) == 1:
                t_push0 = float(time.monotonic())
                _trace_stage(
                    "push_children",
                    programs_explored=int(programs_explored),
                    depth=int(depth),
                    state_stage=str(stage),
                    next_steps_total=int(len(next_steps)),
                )
                _snapshot_trace_programs_v141()
            for ns in next_steps:
                push(tuple(list(steps) + [ns]))
                pushed += 1
                if int(programs_explored) == 1 and int(pushed) in {1, 2, 4, 8, 12, 16, 24, 32}:
                    t3 = float(time.monotonic())
                    _trace_stage(
                        "push_children",
                        programs_explored=int(programs_explored),
                        depth=int(depth),
                        state_stage=str(stage),
                        next_steps_total=int(len(next_steps)),
                        pushed=int(pushed),
                        t_push_children_s=float(t3 - float(t_push0 or t3)),
                        eval_cache_misses=int(metrics.get("eval_cache_misses", 0)),
                        eval_cache_hits=int(metrics.get("eval_cache_hits", 0)),
                        apply_cache_hits=int(metrics.get("apply_cache_hits", 0)),
                        apply_cache_misses=int(metrics.get("apply_cache_misses", 0)),
                    )
                    _snapshot_trace_programs_v141()
            if int(programs_explored) == 1:
                t4 = float(time.monotonic())
                _trace_stage(
                    "push_children_done",
                    programs_explored=int(programs_explored),
                    depth=int(depth),
                    state_stage=str(stage),
                    next_steps_total=int(len(next_steps)),
                    pushed=int(pushed),
                    t_push_children_s=float(t4 - float(t_push0 or t4)),
                    eval_cache_misses=int(metrics.get("eval_cache_misses", 0)),
                    eval_cache_hits=int(metrics.get("eval_cache_hits", 0)),
                )
                _snapshot_trace_programs_v141()

        trace_items: List[Tuple[Tuple[int, int, int, int, str], Dict[str, Any]]] = []
        for bucket in trace_items_by_depth.values():
            trace_items.extend(bucket)
        trace_items.sort(key=lambda kv: kv[0])
        trace_programs = [row for _k, row in trace_items]
        global _TRACE_METRICS_SNAPSHOT_V141, _TRACE_PROGRAMS_SNAPSHOT_V141
        _TRACE_PROGRAMS_SNAPSHOT_V141 = [dict(r) for r in trace_programs]
        snap2: Dict[str, Any] = {}
        try:
            if isinstance(_TRACE_METRICS_SNAPSHOT_V141, dict):
                snap2 = dict(_TRACE_METRICS_SNAPSHOT_V141)
        except Exception:
            snap2 = {}
        snap2.update(
            {
                "programs_explored": int(programs_explored),
                "apply_cache_hits": int(metrics.get("apply_cache_hits", 0)),
                "apply_cache_misses": int(metrics.get("apply_cache_misses", 0)),
                "eval_cache_hits": int(metrics.get("eval_cache_hits", 0)),
                "eval_cache_misses": int(metrics.get("eval_cache_misses", 0)),
                "propose_cache_hits": int(metrics.get("propose_cache_hits", 0)),
                "propose_cache_misses": int(metrics.get("propose_cache_misses", 0)),
                "pruned_by_shape_reachability": int(pruned_shape),
                "pruned_by_palette_reachability": int(pruned_palette),
                "pruned_by_no_grid_modify_in_time": int(pruned_no_grid_modify),
                "pruned_by_dominated_state": int(pruned_by_dominated_state),
	                "pruned_by_exception_prefix": int(metrics.get("pruned_by_exception_prefix", 0)),
	                "pruned_by_next_steps_cap": int(metrics.get("pruned_by_next_steps_cap", 0)),
	                "csv_tested": int(metrics.get("csv_tested", 0)),
	                "csv_rejected": int(metrics.get("csv_rejected", 0)),
	                "csv_survivors": int(metrics.get("csv_survivors", 0)),
	            }
	        )
        _TRACE_METRICS_SNAPSHOT_V141 = snap2

        def _point_patch_repair_from_trace_v141() -> List[Dict[str, Any]]:
            """
            Deterministic point-patch repair for near-miss programs.

            If a candidate program induces TRAIN predictions that are close (<=K cells wrong) and
            the required point edits are consistent across all TRAIN demos, append a concept_call
            that applies `paint_points` to those (r,c,color) triples.

            This is a generic mechanism:
              - no per-task identifier usage
              - no cross-task lookup
              - only uses the demonstrations of the *current task*
            """
            if not bool(enable_point_patch_repair):
                return []
            if int(point_patch_max_points) <= 0:
                return []
            if not trace_programs:
                return []

            # Consider only the best near-miss traces (already sorted by loss/cost/depth).
            cand_rows = [tp for tp in trace_programs if isinstance(tp, dict)]
            cand_rows = cand_rows[:32]

            out: List[Dict[str, Any]] = []
            for tp in cand_rows:
                steps_js = tp.get("steps")
                if not isinstance(steps_js, list) or not steps_js:
                    continue
                steps: List[ProgramStepV141] = []
                ok = True
                for row in steps_js:
                    if not isinstance(row, dict):
                        ok = False
                        break
                    try:
                        steps.append(ProgramStepV141.from_dict(row))
                    except Exception:
                        ok = False
                        break
                if not ok:
                    continue

                ev0 = eval_program(tuple(steps))
                if not bool(ev0.ok_train) and not ev0.train_final_states:
                    continue

                pts0: Optional[Dict[Tuple[int, int], int]] = None
                too_many = False
                shape_bad = False
                for st, (_inp, want) in zip(ev0.train_final_states, train_pairs):
                    if tuple(int(x) for x in grid_shape_v124(st.grid)) != tuple(int(x) for x in grid_shape_v124(want)):
                        shape_bad = True
                        break
                    h, w = grid_shape_v124(want)
                    pts: Dict[Tuple[int, int], int] = {}
                    for r in range(int(h)):
                        for c in range(int(w)):
                            if int(st.grid[int(r)][int(c)]) != int(want[int(r)][int(c)]):
                                pts[(int(r), int(c))] = int(want[int(r)][int(c)])
                    if int(len(pts)) > int(point_patch_max_points):
                        too_many = True
                        break
                    if pts0 is None:
                        pts0 = dict(pts)
                    else:
                        if set(pts.keys()) != set(pts0.keys()):
                            pts0 = None
                            break
                        for k in pts0.keys():
                            if int(pts[k]) != int(pts0[k]):
                                pts0 = None
                                break
                    if pts0 is None:
                        break

                if shape_bad or too_many or pts0 is None or not pts0:
                    continue

                points = [{"r": int(r), "c": int(c), "color": int(pts0[(int(r), int(c))])} for (r, c) in sorted(pts0.keys())]
                patch_step = ProgramStepV141(op_id="paint_points", args={"points": list(points), "mode": "overwrite"})
                body = {"kind": "direct_point_patch_csg_v141", "steps": [patch_step.to_dict()]}
                cid = "csg_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))[:16]
                cost = int(step_cost_bits_v141(op_id="paint_points", args=dict(patch_step.args)))
                patch_call = ProgramStepV141(
                    op_id="concept_call",
                    args={
                        "concept_id": str(cid),
                        "cost_bits": max(1, int(cost) - 1),
                        "op_ids": ["paint_points"],
                        "steps": [patch_step.to_dict()],
                    },
                )

                # Verify TRAIN exactness after repair and produce TEST grid.
                ok_train = True
                for st, (_inp, want) in zip(ev0.train_final_states, train_pairs):
                    stp = st
                    try:
                        for row in patch_call.args.get("steps") or []:
                            stp = apply_op_v141(
                                state=stp,
                                op_id=str(row.get("op_id") or ""),
                                args=dict(row.get("args") or {}),
                            )
                    except Exception:
                        ok_train = False
                        break
                    if not grid_equal_v124(stp.grid, want):
                        ok_train = False
                        break
                if not ok_train:
                    continue

                # Apply to TEST.
                test_state = StateV132(grid=ev0.test_grid, objset=None, obj=None, bbox=None, patch=None)
                try:
                    for row in patch_call.args.get("steps") or []:
                        test_state = apply_op_v141(
                            state=test_state,
                            op_id=str(row.get("op_id") or ""),
                            args=dict(row.get("args") or {}),
                        )
                except Exception:
                    continue

                repaired_steps = tuple(list(steps) + [patch_call])
                p = ProgramV141(steps=repaired_steps)
                sig = p.program_sig()
                cost_bits = 0
                for stp in repaired_steps:
                    cost_bits += int(_step_cost_bits_total_v141(step=stp, macro_support_by_id=macro_support_by_id))
                out.append(
                    {
                        "program_sig": str(sig),
                        "cost_bits": int(cost_bits),
                        "steps": [s.to_dict() for s in repaired_steps],
                        "predicted_grid": [list(r) for r in test_state.grid],
                        "predicted_grid_hash": _grid_hash_cached_v141(g=test_state.grid, cache=grid_hash_cache),
                    }
                )

            # Deterministic de-dup by predicted grid hash.
            uniq: Dict[str, Dict[str, Any]] = {}
            for row in out:
                h = str(row.get("predicted_grid_hash") or "")
                if not h:
                    continue
                if h not in uniq:
                    uniq[h] = row
            return list(uniq.values())

        if best_cost is None or not best_programs:
            patched = _point_patch_repair_from_trace_v141()
            if patched:
                metrics["point_patch_repair_used"] = int(metrics.get("point_patch_repair_used", 0)) + 1
                best_programs = list(patched)
                best_cost = min(int(bp.get("cost_bits") or 0) for bp in best_programs)
            else:
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
                        "propose_cache_hits": int(metrics.get("propose_cache_hits", 0)),
                        "propose_cache_misses": int(metrics.get("propose_cache_misses", 0)),
                        "pruned_by_shape_reachability": int(pruned_shape),
                        "pruned_by_palette_reachability": int(pruned_palette),
                        "pruned_by_no_grid_modify_in_time": int(pruned_no_grid_modify),
                        "pruned_by_dominated_state": int(pruned_by_dominated_state),
                        "pruned_by_exception_prefix": int(metrics.get("pruned_by_exception_prefix", 0)),
                        "pruned_by_next_steps_cap": int(metrics.get("pruned_by_next_steps_cap", 0)),
                        "csv_tested": int(metrics.get("csv_tested", 0)),
                        "csv_rejected": int(metrics.get("csv_rejected", 0)),
                        "csv_survivors": int(metrics.get("csv_survivors", 0)),
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
                    "propose_cache_hits": int(metrics.get("propose_cache_hits", 0)),
                    "propose_cache_misses": int(metrics.get("propose_cache_misses", 0)),
                    "pruned_by_shape_reachability": int(pruned_shape),
                    "pruned_by_palette_reachability": int(pruned_palette),
                    "pruned_by_no_grid_modify_in_time": int(pruned_no_grid_modify),
                    "pruned_by_dominated_state": int(pruned_by_dominated_state),
                    "pruned_by_exception_prefix": int(metrics.get("pruned_by_exception_prefix", 0)),
                    "pruned_by_next_steps_cap": int(metrics.get("pruned_by_next_steps_cap", 0)),
	                    "post_solution_probe_programs": int(POST_SOLUTION_PROBE_PROGRAMS_V141),
	                    "probe_window_exhausted": bool(probe_window_exhausted),
	                    "csv_tested": int(metrics.get("csv_tested", 0)),
	                    "csv_rejected": int(metrics.get("csv_rejected", 0)),
	                    "csv_survivors": int(metrics.get("csv_survivors", 0)),
	                },
	            }

        outs_sorted = sorted(outputs.values(), key=lambda x: str(x.get("predicted_grid_hash") or ""))
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V141),
            "kind": "arc_solver_result_v141",
            "status": "UNKNOWN",
            "predicted_grids": [o.get("predicted_grid") for o in outs_sorted],
            "candidate_programs": [
                {
                    "program_sig": str(o.get("program_sig") or ""),
                    "program_cost_bits": int(o.get("cost_bits") or 0),
                    "program_steps": list(o.get("steps") or []),
                    "predicted_grid": o.get("predicted_grid"),
                    "predicted_grid_hash": str(o.get("predicted_grid_hash") or ""),
                }
                for o in outs_sorted
            ],
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
                "propose_cache_hits": int(metrics.get("propose_cache_hits", 0)),
                "propose_cache_misses": int(metrics.get("propose_cache_misses", 0)),
                "pruned_by_shape_reachability": int(pruned_shape),
                "pruned_by_palette_reachability": int(pruned_palette),
                "pruned_by_no_grid_modify_in_time": int(pruned_no_grid_modify),
                "pruned_by_dominated_state": int(pruned_by_dominated_state),
	                "pruned_by_exception_prefix": int(metrics.get("pruned_by_exception_prefix", 0)),
	                "pruned_by_next_steps_cap": int(metrics.get("pruned_by_next_steps_cap", 0)),
	                "csv_tested": int(metrics.get("csv_tested", 0)),
	                "csv_rejected": int(metrics.get("csv_rejected", 0)),
	                "csv_survivors": int(metrics.get("csv_survivors", 0)),
	            },
	        }

    def _maybe_induce_from_trace(res: Dict[str, Any]) -> int:
        nonlocal concept_templates_norm
        if not bool(enable_trace_csg_induction):
            return 0
        trace = res.get("trace") if isinstance(res.get("trace"), dict) else {}
        tps = trace.get("trace_programs")
        if not isinstance(tps, list) or not tps:
            return 0
        induced = _induce_trace_csg_templates_v141(
            trace_programs=[tp for tp in tps if isinstance(tp, dict)],
            train_pairs=train_pairs,
            test_in=test_in,
            max_templates=int(trace_csg_induction_max_templates),
            min_inner_steps=int(trace_csg_induction_min_inner_steps),
            max_inner_steps=int(trace_csg_induction_max_inner_steps),
            max_loss_cells=int(trace_csg_induction_max_loss_cells),
        )
        if not induced:
            return 0
        # Prepend induced templates so they are proposed before generic libraries.
        concept_templates_norm = _norm_concept_templates_v141(list(induced) + list(concept_templates_norm))
        return int(len(induced))

    def _with_trace_meta(res: Dict[str, Any], *, induced_total: int) -> Dict[str, Any]:
        if int(induced_total) <= 0:
            return res
        trace = res.get("trace") if isinstance(res.get("trace"), dict) else {}
        trace2 = dict(trace)
        trace2["trace_csg_induced_templates_total"] = int(induced_total)
        res2 = dict(res)
        res2["trace"] = trace2
        return res2

    if macro_templates_norm and not bool(macro_try_on_fail_only):
        if not bool(enable_trace_csg_induction):
            return _solve_pass(allow_macros=True)
        frac = min(0.9, max(0.1, float(trace_csg_induction_first_pass_frac)))
        pass1 = max(1, int(float(max_programs) * float(frac)))
        pass2 = max(1, int(max_programs - pass1))
        r1 = _solve_pass(allow_macros=True, max_programs_limit=int(pass1))
        if str(r1.get("status") or "") != "FAIL":
            return r1
        induced_n = 0
        fr = r1.get("failure_reason") if isinstance(r1.get("failure_reason"), dict) else {}
        if str(fr.get("kind") or "") == "SEARCH_BUDGET_EXCEEDED":
            induced_n = _maybe_induce_from_trace(r1)
        r2 = _solve_pass(allow_macros=True, max_programs_limit=int(pass2))
        return _with_trace_meta(r2, induced_total=int(induced_n))

    # No learned macros/operators available: optionally attempt a two-pass solve with trace-driven
    # concept induction, without exceeding the overall program budget.
    if not macro_templates_norm and bool(enable_trace_csg_induction):
        frac = min(0.9, max(0.1, float(trace_csg_induction_first_pass_frac)))
        pass1 = max(1, int(float(max_programs) * float(frac)))
        pass2 = max(1, int(max_programs - pass1))
        r1 = _solve_pass(allow_macros=False, max_programs_limit=int(pass1))
        if str(r1.get("status") or "") != "FAIL":
            return r1
        induced_n = 0
        fr = r1.get("failure_reason") if isinstance(r1.get("failure_reason"), dict) else {}
        if str(fr.get("kind") or "") == "SEARCH_BUDGET_EXCEEDED":
            induced_n = _maybe_induce_from_trace(r1)
        r2 = _solve_pass(allow_macros=False, max_programs_limit=int(pass2))
        return _with_trace_meta(r2, induced_total=int(induced_n))

    r0 = _solve_pass(allow_macros=False)
    if str(r0.get("status") or "") != "FAIL":
        return r0
    if not macro_templates_norm:
        return r0
    if not bool(enable_trace_csg_induction):
        return _solve_pass(allow_macros=True)
    frac = min(0.9, max(0.1, float(trace_csg_induction_first_pass_frac)))
    pass1 = max(1, int(float(max_programs) * float(frac)))
    pass2 = max(1, int(max_programs - pass1))
    r1 = _solve_pass(allow_macros=True, max_programs_limit=int(pass1))
    if str(r1.get("status") or "") != "FAIL":
        return r1
    induced_n = 0
    fr = r1.get("failure_reason") if isinstance(r1.get("failure_reason"), dict) else {}
    if str(fr.get("kind") or "") == "SEARCH_BUDGET_EXCEEDED":
        induced_n = _maybe_induce_from_trace(r1)
    r2 = _solve_pass(allow_macros=True, max_programs_limit=int(pass2))
    return _with_trace_meta(r2, induced_total=int(induced_n))
