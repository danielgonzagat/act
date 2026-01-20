from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_objects_v132 import connected_components4_v132
from .arc_ops_v132 import OP_DEFS_V132, StateV132, apply_op_v132, step_cost_bits_v132
from .grid_v124 import GridV124, grid_equal_v124, grid_hash_v124, grid_shape_v124, unique_colors_v124

ARC_SOLVER_SCHEMA_VERSION_V133 = 133


def _validate_grid_values_v133(g: GridV124) -> None:
    for row in g:
        for x in row:
            xx = int(x)
            if xx < 0 or xx > 9:
                raise ValueError("grid_cell_out_of_range")


@dataclass(frozen=True)
class ProgramStepV133:
    op_id: str
    args: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        a: Dict[str, Any] = {}
        for k in sorted(self.args.keys()):
            a[str(k)] = self.args[k]
        return {"op_id": str(self.op_id), "args": a}


@dataclass(frozen=True)
class ProgramV133:
    steps: Tuple[ProgramStepV133, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V133),
            "kind": "arc_program_v133",
            "steps": [s.to_dict() for s in self.steps],
        }

    def program_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


def _program_cost_bits_v133(program: ProgramV133) -> int:
    bits = 0
    for s in program.steps:
        bits += step_cost_bits_v132(op_id=str(s.op_id), args=dict(s.args))
    return int(bits)


def _summarize_mismatch_v133(*, got: GridV124, want: GridV124) -> Dict[str, Any]:
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


def _abstract_slots_after_steps_v133(steps: Sequence[ProgramStepV133]) -> Dict[str, bool]:
    avail: Dict[str, bool] = {"grid": True, "objset": False, "obj": False, "bbox": False, "patch": False}
    for st in steps:
        od = OP_DEFS_V132.get(str(st.op_id))
        if od is None:
            continue
        for r in od.reads:
            if not bool(avail.get(str(r), False)):
                return {"grid": True, "objset": False, "obj": False, "bbox": False, "patch": False, "invalid": True}
        for w in od.writes:
            avail[str(w)] = True
        if str(st.op_id) == "commit_patch":
            avail["patch"] = False
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
            "crop_bbox_nonzero",
            "pad_to",
            "replace_color",
            "map_colors",
        }:
            avail["objset"] = False
            avail["obj"] = False
            avail["bbox"] = False
            avail["patch"] = False
    return avail


def _last_cc4_bg_v133(steps: Sequence[ProgramStepV133]) -> Optional[int]:
    for st in reversed(list(steps)):
        if str(st.op_id) == "cc4":
            bg = st.args.get("bg")
            if bg is None:
                return None
            return int(bg)
    return None


def apply_program_v133(program: ProgramV133, g_in: GridV124) -> GridV124:
    _validate_grid_values_v133(g_in)
    st = StateV132(grid=g_in)
    for step in program.steps:
        st = apply_op_v132(state=st, op_id=str(step.op_id), args=dict(step.args))
        _validate_grid_values_v133(st.grid)
        if st.patch is not None:
            _validate_grid_values_v133(st.patch)
    return st.grid


def _infer_color_mapping_v133(inp: GridV124, out: GridV124) -> Optional[Dict[str, int]]:
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


def _mode_color_v133(g: GridV124) -> int:
    counts: Dict[int, int] = {}
    for row in g:
        for x in row:
            xx = int(x)
            counts[xx] = int(counts.get(xx, 0)) + 1
    ordered = sorted(((int(k), int(counts[k])) for k in counts.keys()), key=lambda kv: (-kv[1], kv[0]))
    return int(ordered[0][0]) if ordered else 0


def _bg_candidates_v133(grids: Sequence[GridV124]) -> Tuple[int, ...]:
    out: List[int] = [0]
    for g in grids:
        h, w = grid_shape_v124(g)
        if h > 0 and w > 0:
            out.extend([int(g[0][0]), int(g[0][w - 1]), int(g[h - 1][0]), int(g[h - 1][w - 1])])
            out.append(int(_mode_color_v133(g)))
    return tuple(int(x) for x in sorted(set(int(x) for x in out)))


def _changed_cells_v133(inp: GridV124, out: GridV124) -> Optional[Tuple[Tuple[int, int], ...]]:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if (hi, wi) != (ho, wo):
        return None
    cells: List[Tuple[int, int]] = []
    for r in range(hi):
        for c in range(wi):
            if int(inp[r][c]) != int(out[r][c]):
                cells.append((int(r), int(c)))
    cells.sort(key=lambda rc: (int(rc[0]), int(rc[1])))
    return tuple(cells)


def _infer_direct_steps_v133(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> List[ProgramStepV133]:
    # Inference is purely from train_pairs; candidates must be consistent across all pairs.
    direct: List[ProgramStepV133] = []

    # rotate/reflect candidates
    for op_id in ["rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v"]:
        ok = True
        for inp, out in train_pairs:
            got = apply_program_v133(ProgramV133(steps=(ProgramStepV133(op_id=op_id, args={}),)), inp)
            if not grid_equal_v124(got, out):
                ok = False
                break
        if ok:
            direct.append(ProgramStepV133(op_id=str(op_id), args={}))

    # map_colors (functional mapping)
    mapping: Dict[str, int] = {}
    mapping_ok = True
    for inp, out in train_pairs:
        m = _infer_color_mapping_v133(inp, out)
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
        # verify
        ok = True
        step = ProgramStepV133(op_id="map_colors", args={"mapping": {str(k): int(mapping[k]) for k in sorted(mapping.keys())}})
        for inp, out in train_pairs:
            got = apply_program_v133(ProgramV133(steps=(step,)), inp)
            if not grid_equal_v124(got, out):
                ok = False
                break
        if ok:
            direct.append(step)

    # translate (infer shifts from bbox_nonzero under bg candidates, verify)
    from .grid_v124 import bbox_nonzero_v124

    bgs = _bg_candidates_v133([p[0] for p in train_pairs] + [test_in])
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
        step = ProgramStepV133(op_id="translate", args={"dx": int(dx), "dy": int(dy), "pad": int(bg)})
        ok = True
        for inp, out in train_pairs:
            got = apply_program_v133(ProgramV133(steps=(step,)), inp)
            if not grid_equal_v124(got, out):
                ok = False
                break
        if ok:
            direct.append(step)

    # crop_bbox_nonzero candidates
    from .grid_v124 import crop_to_bbox_nonzero_v124

    for bg in bgs:
        ok = True
        for inp, out in train_pairs:
            got = crop_to_bbox_nonzero_v124(inp, bg=int(bg))
            if not grid_equal_v124(got, out):
                ok = False
                break
        if ok:
            direct.append(ProgramStepV133(op_id="crop_bbox_nonzero", args={"bg": int(bg)}))

    # pad_to candidates (exact match only)
    from .grid_v124 import pad_to_v124

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
                direct.append(ProgramStepV133(op_id="pad_to", args={"height": int(h), "width": int(w), "pad": int(bg)}))

    # Canonical order + dedup
    direct.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
    seen: Set[str] = set()
    out_steps: List[ProgramStepV133] = []
    for s in direct:
        sig = sha256_hex(canonical_json_dumps(s.to_dict()).encode("utf-8"))
        if sig in seen:
            continue
        seen.add(sig)
        out_steps.append(s)
    return out_steps


def _infer_select_obj_args_v133(
    *, train_pairs: Sequence[Tuple[GridV124, GridV124]], bg: int, max_rank: int = 1
) -> List[Dict[str, Any]]:
    """
    Infer a small set of selector args consistent with targets derived from changed_mask overlap.
    If inference fails, returns [] (caller should use a small fallback).
    """
    from .arc_ops_v132 import _select_obj_v132

    keys = ["area", "width", "height", "bbox_area", "top", "left", "bottom", "right", "color", "dist_center"]
    candidates_all: List[Dict[str, Any]] = []
    # Candidate selector space is still general, but we will filter by evidence.
    for key in keys:
        for order in ["min", "max"]:
            for rank in range(0, int(max_rank) + 1):
                candidates_all.append({"key": str(key), "order": str(order), "rank": int(rank), "color_filter": None})

    viable: Optional[Set[str]] = None
    for inp, out in train_pairs:
        cm = _changed_cells_v133(inp, out)
        if cm is None or not cm:
            return []
        cm_set = set(cm)
        oset = connected_components4_v132(inp, bg=int(bg))
        objs = list(oset.objects)
        if not objs:
            return []
        # Pick target object by max intersection with changed mask.
        best: Optional[Tuple[int, Tuple[int, int, int, int], int, Tuple[Tuple[int, int], ...]]] = None
        best_score = 0
        tied = False
        for o in objs:
            inter = 0
            for cell in o.cells:
                if cell in cm_set:
                    inter += 1
            if inter <= 0:
                continue
            if inter > best_score:
                best_score = int(inter)
                best = (int(o.color), o.bbox.to_tuple(), int(o.area), o.cells)
                tied = False
            elif inter == best_score and inter > 0:
                tied = True
        if best is None or tied:
            return []

        # Candidate color_filters: none or the target color (keeps domain small and evidence-driven).
        target_color = int(best[0])
        color_filters: List[Optional[int]] = [None, int(target_color)]
        # Expand candidate set with color_filter values.
        pair_candidates: List[Dict[str, Any]] = []
        for base in candidates_all:
            for cf in color_filters:
                d = dict(base)
                d["color_filter"] = int(cf) if cf is not None else None
                pair_candidates.append(d)

        ok_specs: Set[str] = set()
        h, w = grid_shape_v124(inp)
        for spec in pair_candidates:
            try:
                sel = _select_obj_v132(
                    oset,
                    key=str(spec["key"]),
                    order=str(spec["order"]),
                    rank=int(spec["rank"]),
                    color_filter=spec.get("color_filter"),
                    grid_shape=(int(h), int(w)),
                )
                sel_key = (int(sel.color), sel.bbox.to_tuple(), int(sel.area), sel.cells)
                if sel_key == best:
                    ok_specs.add(canonical_json_dumps(spec))
            except Exception:
                continue

        if viable is None:
            viable = set(ok_specs)
        else:
            viable = set(viable.intersection(ok_specs))
        if not viable:
            return []

    out_specs = [json.loads(s) for s in sorted(viable)]
    # Prefer smaller rank, then key, then order, then color_filter (deterministic).
    out_specs.sort(
        key=lambda d: (
            int(d.get("rank") or 0),
            str(d.get("key") or ""),
            str(d.get("order") or ""),
            "1" if d.get("color_filter") is None else "2",
            int(d.get("color_filter") or 0),
        )
    )
    return out_specs[:10]


@dataclass(frozen=True)
class SolveConfigV133:
    max_depth: int = 4
    max_programs: int = 4000
    trace_program_limit: int = 80


def _propose_next_steps_v133(
    *,
    steps_so_far: Sequence[ProgramStepV133],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    bg_candidates: Tuple[int, ...],
    shapes_out: Tuple[Tuple[int, int], ...],
    palette_out: Tuple[int, ...],
    direct_steps: Sequence[ProgramStepV133],
) -> List[ProgramStepV133]:
    avail = _abstract_slots_after_steps_v133(steps_so_far)
    if bool(avail.get("invalid", False)):
        return []

    out: List[ProgramStepV133] = []

    # Stage 0: operate on grid only (no derived slots). Keep candidate set small.
    if not bool(avail.get("objset")) and not bool(avail.get("obj")) and not bool(avail.get("bbox")) and not bool(avail.get("patch")):
        out.extend(list(direct_steps))
        # Minimal setup steps for shape-related tasks.
        for bg in bg_candidates:
            out.append(ProgramStepV133(op_id="crop_bbox_nonzero", args={"bg": int(bg)}))
        if shapes_out:
            for h, w in shapes_out:
                for bg in bg_candidates:
                    out.append(ProgramStepV133(op_id="pad_to", args={"height": int(h), "width": int(w), "pad": int(bg)}))
                    out.append(ProgramStepV133(op_id="new_canvas", args={"height": int(h), "width": int(w), "color": int(bg)}))
        # Allow objectization as an alternate branch.
        for bg in bg_candidates:
            out.append(ProgramStepV133(op_id="cc4", args={"bg": int(bg), "colors": []}))

    # Stage 1: choose an object (selectors inferred from train evidence for the chosen bg).
    elif bool(avail.get("objset")) and not bool(avail.get("obj")):
        bg = _last_cc4_bg_v133(steps_so_far)
        bg = int(bg) if bg is not None else int(bg_candidates[0] if bg_candidates else 0)
        inferred = _infer_select_obj_args_v133(train_pairs=train_pairs, bg=int(bg), max_rank=1)
        if not inferred:
            # Small deterministic fallback (no color_filter): reduces branching drastically vs V132.
            inferred = [
                {"key": "area", "order": "max", "rank": 0, "color_filter": None},
                {"key": "area", "order": "min", "rank": 0, "color_filter": None},
                {"key": "dist_center", "order": "min", "rank": 0, "color_filter": None},
                {"key": "left", "order": "min", "rank": 0, "color_filter": None},
                {"key": "top", "order": "min", "rank": 0, "color_filter": None},
            ]
        for a in inferred:
            out.append(ProgramStepV133(op_id="select_obj", args=dict(a)))

    # Stage 2: obj -> bbox
    elif bool(avail.get("obj")) and not bool(avail.get("bbox")):
        out.append(ProgramStepV133(op_id="obj_bbox", args={}))

    # Stage 3: bbox available (crop, paint, border)
    elif bool(avail.get("bbox")) and not bool(avail.get("patch")):
        out.append(ProgramStepV133(op_id="crop_bbox", args={}))
        for c in palette_out:
            out.append(ProgramStepV133(op_id="paint_rect", args={"color": int(c)}))
            out.append(ProgramStepV133(op_id="draw_rect_border", args={"color": int(c), "thickness": 1}))

    # Stage 4: patch available (commit/paste)
    elif bool(avail.get("patch")):
        out.append(ProgramStepV133(op_id="commit_patch", args={}))
        # Paste positions: derive from changed bbox of first pair (if any), plus (0,0).
        positions: Set[Tuple[int, int]] = {(0, 0)}
        for inp, outg in train_pairs[:2]:
            cm = _changed_cells_v133(inp, outg)
            if cm:
                rs = [int(r) for r, _ in cm]
                cs = [int(c) for _, c in cm]
                positions.add((int(min(rs)), int(min(cs))))
        for top, left in sorted(positions):
            out.append(ProgramStepV133(op_id="paste", args={"top": int(top), "left": int(left), "transparent": 0}))

    # Canonical ordering + dedup by step_sig
    out.sort(key=lambda s: canonical_json_dumps(s.to_dict()))
    seen: Set[str] = set()
    uniq: List[ProgramStepV133] = []
    for s in out:
        ss = sha256_hex(canonical_json_dumps(s.to_dict()).encode("utf-8"))
        if ss in seen:
            continue
        seen.add(ss)
        uniq.append(s)
    return uniq


def solve_arc_task_v133(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    config: Optional[SolveConfigV133] = None,
) -> Dict[str, Any]:
    cfg = config or SolveConfigV133()
    max_depth = int(cfg.max_depth)
    max_programs = int(cfg.max_programs)
    trace_program_limit = int(cfg.trace_program_limit)

    # Validate inputs early (fail deterministically).
    _validate_grid_values_v133(test_in)
    for inp, out in train_pairs:
        _validate_grid_values_v133(inp)
        _validate_grid_values_v133(out)

    # Evidence / concept trace (deterministic).
    all_grids = [test_in] + [p[0] for p in train_pairs] + [p[1] for p in train_pairs]
    bg_candidates = _bg_candidates_v133(all_grids)
    palette_all = sorted(set(int(c) for g in all_grids for c in unique_colors_v124(g)))
    palette_out = sorted(set(int(c) for _, out in train_pairs for c in unique_colors_v124(out)))
    shapes_out = sorted(set((int(h), int(w)) for _, out in train_pairs for h, w in [grid_shape_v124(out)]))

    direct_steps = _infer_direct_steps_v133(train_pairs=train_pairs, test_in=test_in)

    concept_trace: Dict[str, Any] = {
        "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V133),
        "kind": "concept_trace_v133",
        "bg_candidates": [int(x) for x in bg_candidates],
        "palette_all": [int(x) for x in palette_all],
        "palette_out": [int(x) for x in palette_out],
        "shapes_out": [{"h": int(h), "w": int(w)} for h, w in shapes_out],
        "obj_summaries_bg0": [],
    }
    try:
        for inp, _ in train_pairs[:3]:
            oset = connected_components4_v132(inp, bg=0)
            concept_trace["obj_summaries_bg0"].append({"count": int(len(oset.objects))})
    except Exception:
        pass

    def eval_program(program: ProgramV133) -> Tuple[bool, Tuple[int, int], Optional[Dict[str, Any]]]:
        # Returns (ok_train, loss_tuple, mismatch_example)
        shape_mismatch = 0
        diff_cells = 0
        mismatch_ex: Optional[Dict[str, Any]] = None
        ok_all = True
        for inp, want in train_pairs:
            try:
                got = apply_program_v133(program, inp)
            except Exception as e:
                ok_all = False
                shape_mismatch += 1
                diff_cells += 100000
                if mismatch_ex is None:
                    mismatch_ex = {
                        "kind": "exception",
                        "error_type": str(type(e).__name__),
                        "error": str(e)[:200],
                    }
                continue
            if not grid_equal_v124(got, want):
                ok_all = False
                mm = _summarize_mismatch_v133(got=got, want=want)
                if mm.get("kind") == "shape_mismatch":
                    shape_mismatch += 1
                    diff_cells += 100000
                else:
                    diff_cells += int(mm.get("diff_cells") or 0)
                if mismatch_ex is None:
                    mismatch_ex = mm
            else:
                # reward exact match by not increasing diff_cells
                pass
        return bool(ok_all), (int(shape_mismatch), int(diff_cells)), mismatch_ex

    # Search (deterministic best-first).
    start = ProgramV133(steps=tuple())
    start_sig = start.program_sig()

    heap: List[Tuple[int, Tuple[int, int], int, str, ProgramV133]] = []
    ok0, loss0, mm0 = eval_program(start)
    heapq.heappush(heap, (0, loss0, 0, start_sig, start))

    seen: Set[str] = {start_sig}
    tried = 0
    trace_programs: List[Dict[str, Any]] = []

    best_cost: Optional[int] = None
    best_programs: List[ProgramV133] = []

    while heap:
        cost_bits, loss, depth, psig, program = heapq.heappop(heap)
        tried += 1

        if tried <= trace_program_limit:
            ok_train, _, mismatch = eval_program(program)
            trace_programs.append(
                {
                    "program_sig": str(psig),
                    "cost_bits": int(cost_bits),
                    "depth": int(depth),
                    "ok_train": bool(ok_train),
                    "mismatch": mismatch,
                    "steps": [s.to_dict() for s in program.steps],
                }
            )

        if best_cost is not None and int(cost_bits) > int(best_cost):
            # No need to explore higher-cost programs once minimal cost solutions are fixed.
            break

        ok_train, _, _ = eval_program(program)
        if ok_train:
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

        next_steps = _propose_next_steps_v133(
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
            new_prog = ProgramV133(steps=new_steps)
            new_sig = new_prog.program_sig()
            if new_sig in seen:
                continue
            # Type discipline pruning by abstract slots (early).
            avail = _abstract_slots_after_steps_v133(new_steps)
            if bool(avail.get("invalid", False)):
                continue
            seen.add(new_sig)
            new_cost = int(cost_bits) + int(step_cost_bits_v132(op_id=str(st.op_id), args=dict(st.args)))
            ok_train2, loss2, _ = eval_program(new_prog)
            heapq.heappush(heap, (int(new_cost), loss2, int(depth) + 1, str(new_sig), new_prog))

    # Decide final status.
    if best_cost is not None and best_programs:
        # Fail-closed: if minimal programs disagree on test_in output -> UNKNOWN.
        out_by_prog: Dict[str, str] = {}
        out_hashes: List[str] = []
        for p in best_programs:
            ph = p.program_sig()
            gout = apply_program_v133(p, test_in)
            gh = grid_hash_v124(gout)
            out_by_prog[str(ph)] = str(gh)
            out_hashes.append(str(gh))
        uniq = sorted(set(out_hashes))
        if len(uniq) == 1:
            predicted = apply_program_v133(best_programs[0], test_in)
            return {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V133),
                "kind": "arc_solver_result_v133",
                "status": "SOLVED",
                "failure_reason": None,
                "program_sig": str(best_programs[0].program_sig()),
                "program_cost_bits": int(best_cost),
                "predicted_grid": [list(r) for r in predicted],
                "predicted_grid_hash": str(grid_hash_v124(predicted)),
                "trace": {
                    "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V133),
                    "kind": "arc_trace_v133",
                    "concept_trace": concept_trace,
                    "trace_programs": trace_programs,
                    "tried": int(tried),
                    "max_programs": int(max_programs),
                    "max_depth": int(max_depth),
                    "min_cost_solutions": int(len(best_programs)),
                },
            }
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V133),
            "kind": "arc_solver_result_v133",
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
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V133),
                "kind": "arc_trace_v133",
                "concept_trace": concept_trace,
                "trace_programs": trace_programs,
                "tried": int(tried),
                "max_programs": int(max_programs),
                "max_depth": int(max_depth),
                "min_cost_solutions": int(len(best_programs)),
            },
        }

    # No solution found.
    if tried >= max_programs:
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V133),
            "kind": "arc_solver_result_v133",
            "status": "FAIL",
            "failure_reason": {
                "kind": "SEARCH_BUDGET_EXCEEDED",
                "details": {"candidates_tested": int(tried), "max_programs": int(max_programs), "max_depth": int(max_depth)},
            },
            "program_sig": "",
            "program_cost_bits": 0,
            "predicted_grid_hash": "",
            "trace": {
                "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V133),
                "kind": "arc_trace_v133",
                "concept_trace": concept_trace,
                "trace_programs": trace_programs,
                "tried": int(tried),
                "max_programs": int(max_programs),
                "max_depth": int(max_depth),
                "min_cost_solutions": 0,
            },
        }

    return {
        "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V133),
        "kind": "arc_solver_result_v133",
        "status": "FAIL",
        "failure_reason": {
            "kind": "MISSING_OPERATOR",
            "details": {"search_exhausted": True, "candidates_tested": int(tried), "max_depth": int(max_depth)},
        },
        "program_sig": "",
        "program_cost_bits": 0,
        "predicted_grid_hash": "",
        "trace": {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V133),
            "kind": "arc_trace_v133",
            "concept_trace": concept_trace,
            "trace_programs": trace_programs,
            "tried": int(tried),
            "max_programs": int(max_programs),
            "max_depth": int(max_depth),
            "min_cost_solutions": 0,
        },
    }
