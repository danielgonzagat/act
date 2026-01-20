from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .grid_v124 import (
    GridV124,
    crop_to_bbox_nonzero_v124,
    grid_equal_v124,
    grid_hash_v124,
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

ARC_SOLVER_SCHEMA_VERSION_V131 = 131


def _validate_grid_values_v131(g: GridV124) -> None:
    for row in g:
        for x in row:
            xx = int(x)
            if xx < 0 or xx > 9:
                raise ValueError("grid_cell_out_of_range")


@dataclass(frozen=True)
class ProgramStepV131:
    op: str
    args: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"op": str(self.op)}
        for k in sorted(self.args.keys()):
            d[str(k)] = self.args[k]
        return d


@dataclass(frozen=True)
class ProgramV131:
    steps: Tuple[ProgramStepV131, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V131),
            "kind": "arc_program_v131",
            "steps": [s.to_dict() for s in self.steps],
        }

    def program_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


def _replace_color_v131(g: GridV124, *, from_color: int, to_color: int) -> GridV124:
    fc = int(from_color)
    tc = int(to_color)
    if fc < 0 or fc > 9 or tc < 0 or tc > 9:
        raise ValueError("color_out_of_range")
    return tuple(tuple(tc if int(x) == fc else int(x) for x in row) for row in g)


def _map_colors_v131(g: GridV124, *, mapping: Dict[str, int]) -> GridV124:
    m: Dict[int, int] = {}
    for k, v in mapping.items():
        kk = int(k)
        vv = int(v)
        if kk < 0 or kk > 9 or vv < 0 or vv > 9:
            raise ValueError("color_out_of_range")
        m[kk] = vv
    return tuple(tuple(int(m.get(int(x), int(x))) for x in row) for row in g)


def apply_program_v131(program: ProgramV131, g: GridV124) -> GridV124:
    _validate_grid_values_v131(g)
    cur = g
    for step in program.steps:
        op = str(step.op)
        a = dict(step.args)
        if op == "rotate90":
            cur = rotate90_v124(cur)
        elif op == "rotate180":
            cur = rotate180_v124(cur)
        elif op == "rotate270":
            cur = rotate270_v124(cur)
        elif op == "reflect_h":
            cur = reflect_h_v124(cur)
        elif op == "reflect_v":
            cur = reflect_v_v124(cur)
        elif op == "translate":
            cur = translate_v124(cur, dx=int(a["dx"]), dy=int(a["dy"]), pad=int(a.get("pad", 0)))
        elif op == "crop_bbox_nonzero":
            cur = crop_to_bbox_nonzero_v124(cur, bg=int(a.get("bg", 0)))
        elif op == "pad_to":
            cur = pad_to_v124(cur, height=int(a["height"]), width=int(a["width"]), pad=int(a.get("pad", 0)))
        elif op == "replace_color":
            cur = _replace_color_v131(cur, from_color=int(a["from_color"]), to_color=int(a["to_color"]))
        elif op == "map_colors":
            cur = _map_colors_v131(cur, mapping=dict(a.get("mapping", {})))
        else:
            raise ValueError(f"unknown_op:{op}")
        _validate_grid_values_v131(cur)
    return cur


def _program_cost_bits_v131(program: ProgramV131) -> int:
    # Deterministic MDL proxy:
    # - each step has a base cost
    # - each scalar param costs +4 bits
    # - mapping costs +8 bits per entry
    bits = 0
    for s in program.steps:
        bits += 10
        for k, v in s.args.items():
            if k == "mapping" and isinstance(v, dict):
                bits += 8 * int(len(v))
            else:
                bits += 4
    return int(bits)


def _infer_color_mapping_v131(inp: GridV124, out: GridV124) -> Optional[Dict[str, int]]:
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


def _bg_candidates_v131(grids: Sequence[GridV124]) -> List[int]:
    out: List[int] = [0]
    for g in grids:
        h, w = grid_shape_v124(g)
        if h > 0 and w > 0:
            out.extend([int(g[0][0]), int(g[0][w - 1]), int(g[h - 1][0]), int(g[h - 1][w - 1])])
    return sorted(set(int(x) for x in out))


def _op_variants_v131(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], max_translate_shift: int = 2) -> List[ProgramStepV131]:
    colors: List[int] = []
    shapes_out: List[Tuple[int, int]] = []
    inferred_maps: List[Dict[str, int]] = []
    train_inputs: List[GridV124] = []
    for inp, out in train_pairs:
        train_inputs.append(inp)
        colors.extend(unique_colors_v124(inp))
        colors.extend(unique_colors_v124(out))
        shapes_out.append(grid_shape_v124(out))
        m = _infer_color_mapping_v131(inp, out)
        if m is not None:
            inferred_maps.append(m)

    colors = sorted(set(int(c) for c in colors))
    shapes_out = sorted(set((int(h), int(w)) for h, w in shapes_out))
    inferred_maps = sorted(inferred_maps, key=lambda m: canonical_json_dumps(m))

    steps: List[ProgramStepV131] = []
    # Basic transforms
    for op in ["rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v"]:
        steps.append(ProgramStepV131(op=op, args={}))

    # crop_bbox_nonzero with plausible backgrounds
    for bg in _bg_candidates_v131(train_inputs):
        steps.append(ProgramStepV131(op="crop_bbox_nonzero", args={"bg": int(bg)}))

    # small translations (bounded deterministically)
    for dy in range(-int(max_translate_shift), int(max_translate_shift) + 1):
        for dx in range(-int(max_translate_shift), int(max_translate_shift) + 1):
            if dx == 0 and dy == 0:
                continue
            steps.append(ProgramStepV131(op="translate", args={"dx": int(dx), "dy": int(dy), "pad": 0}))

    # pad_to shapes seen in outputs
    for h, w in shapes_out:
        steps.append(ProgramStepV131(op="pad_to", args={"height": int(h), "width": int(w), "pad": 0}))

    # replace_color from observed colors
    for fc in colors:
        for tc in colors:
            if fc == tc:
                continue
            steps.append(ProgramStepV131(op="replace_color", args={"from_color": int(fc), "to_color": int(tc)}))

    # map_colors inferred from at least one training pair
    for m in inferred_maps:
        steps.append(ProgramStepV131(op="map_colors", args={"mapping": dict(m)}))

    # Deterministic ordering
    steps.sort(key=lambda s: (str(s.op), canonical_json_dumps(s.to_dict())))
    return steps


def _summarize_mismatch_v131(*, got: GridV124, want: GridV124) -> Dict[str, Any]:
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


def solve_arc_task_v131(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    max_depth: int = 3,
    max_programs: int = 2000,
    trace_program_limit: int = 50,
    max_translate_shift: int = 2,
) -> Dict[str, Any]:
    """
    ARC solver base (C1): deterministic compositional search over typed grid ops.
    FAIL-CLOSED: if multiple minimal programs disagree on test_in output -> UNKNOWN.
    """
    try:
        _validate_grid_values_v131(test_in)
        for inp, out in train_pairs:
            _validate_grid_values_v131(inp)
            _validate_grid_values_v131(out)
    except Exception as e:
        return {
            "schema_version": 131,
            "kind": "arc_solve_result_v131",
            "status": "FAIL",
            "failure_reason": {"kind": "INVARIANT_VIOLATION", "details": {"error": str(e)}},
        }

    if not train_pairs:
        return {
            "schema_version": 131,
            "kind": "arc_solve_result_v131",
            "status": "FAIL",
            "failure_reason": {"kind": "TYPE_MISMATCH", "details": {"error": "no_train_pairs"}},
        }

    ops = _op_variants_v131(train_pairs=list(train_pairs), max_translate_shift=int(max_translate_shift))
    step_fps: List[str] = [canonical_json_dumps(s.to_dict()) for s in ops]
    step_costs: List[int] = []
    for s in ops:
        step_costs.append(10 + sum(8 * int(len(v)) if k == "mapping" and isinstance(v, dict) else 4 for k, v in s.args.items()))

    def _child_sig(parent_sig: str, step_fp: str) -> str:
        return sha256_hex((str(parent_sig) + "|" + str(step_fp)).encode("utf-8"))

    root_sig = sha256_hex(b"arc_program_v131_root")
    frontier: List[Tuple[int, int, str, int, Tuple[ProgramStepV131, ...]]] = []
    push_idx = 0
    heapq.heappush(frontier, (0, 0, str(root_sig), push_idx, tuple()))

    seen: set[str] = set()
    best_cost: Optional[int] = None
    solutions: List[ProgramV131] = []

    tried = 0
    trace_programs: List[Dict[str, Any]] = []
    budget_exhausted = False

    while frontier:
        cost_bits, plen, sig, _, steps = heapq.heappop(frontier)
        if sig in seen:
            continue
        seen.add(sig)

        if best_cost is not None and int(cost_bits) > int(best_cost):
            break

        program = ProgramV131(steps=tuple(steps))

        ok_train = True
        mismatch: Optional[Dict[str, Any]] = None
        for inp, out in train_pairs:
            try:
                got = apply_program_v131(program, inp)
            except Exception as e:
                ok_train = False
                mismatch = {"kind": "apply_error", "error": str(e)}
                break
            if not grid_equal_v124(got, out):
                ok_train = False
                mismatch = _summarize_mismatch_v131(got=got, want=out)
                break

        if len(trace_programs) < int(trace_program_limit):
            trace_programs.append(
                {
                    "program_sig": program.program_sig(),
                    "cost_bits": int(cost_bits),
                    "depth": int(plen),
                    "steps": [s.to_dict() for s in program.steps],
                    "ok_train": bool(ok_train),
                    "mismatch": mismatch,
                }
            )

        tried += 1
        if tried >= int(max_programs):
            budget_exhausted = True
            break

        if ok_train:
            if best_cost is None:
                best_cost = int(cost_bits)
            if int(cost_bits) == int(best_cost):
                solutions.append(program)
            continue

        if plen >= int(max_depth):
            continue

        for step_idx, step in enumerate(ops):
            child_sig = _child_sig(str(sig), str(step_fps[step_idx]))
            if child_sig in seen:
                continue
            child_steps = tuple(list(steps) + [step])
            child_cost = int(cost_bits) + int(step_costs[step_idx])
            push_idx += 1
            heapq.heappush(frontier, (int(child_cost), int(plen + 1), str(child_sig), int(push_idx), child_steps))

    trace = {
        "schema_version": 131,
        "kind": "arc_trace_v131",
        "candidates_tested": int(tried),
        "candidates_kept_in_trace": int(len(trace_programs)),
        "trace_programs": list(trace_programs),
        "budget_exhausted": bool(budget_exhausted),
        "max_programs": int(max_programs),
        "max_depth": int(max_depth),
    }

    if solutions:
        sols_sorted = sorted(solutions, key=lambda p: p.program_sig())
        best_program = sols_sorted[0]
        outputs: List[Tuple[str, GridV124]] = []
        for p in sols_sorted:
            try:
                pred = apply_program_v131(p, test_in)
            except Exception as e:
                return {
                    "schema_version": 131,
                    "kind": "arc_solve_result_v131",
                    "status": "FAIL",
                    "failure_reason": {"kind": "INVARIANT_VIOLATION", "details": {"error": str(e)}},
                    "trace": dict(trace),
                }
            outputs.append((p.program_sig(), pred))

        pred_hashes = {sig: grid_hash_v124(g) for sig, g in outputs}
        out_hashes = sorted(set(pred_hashes.values()))
        if len(out_hashes) == 1:
            pred_grid = outputs[0][1]
            return {
                "schema_version": 131,
                "kind": "arc_solve_result_v131",
                "status": "SOLVED",
                "program": [s.to_dict() for s in best_program.steps],
                "program_sig": best_program.program_sig(),
                "program_cost_bits": int(_program_cost_bits_v131(best_program)),
                "predicted_grid_hash": grid_hash_v124(pred_grid),
                "predicted_output": [list(r) for r in pred_grid],
                "trace": dict(trace),
            }

        return {
            "schema_version": 131,
            "kind": "arc_solve_result_v131",
            "status": "UNKNOWN",
            "failure_reason": {
                "kind": "AMBIGUOUS_RULE",
                "details": {"solutions": int(len(solutions)), "distinct_predicted_outputs": int(len(out_hashes))},
            },
            "candidate_program_sigs": [p.program_sig() for p in sols_sorted],
            "predicted_grid_hash_by_solution": {str(k): str(pred_hashes[k]) for k in sorted(pred_hashes.keys())},
            "trace": dict(trace),
        }

    if budget_exhausted:
        return {
            "schema_version": 131,
            "kind": "arc_solve_result_v131",
            "status": "FAIL",
            "failure_reason": {
                "kind": "SEARCH_BUDGET_EXCEEDED",
                "details": {"candidates_tested": int(tried), "max_programs": int(max_programs), "max_depth": int(max_depth)},
            },
            "trace": dict(trace),
        }

    return {
        "schema_version": 131,
        "kind": "arc_solve_result_v131",
        "status": "FAIL",
        "failure_reason": {"kind": "MISSING_OPERATOR", "details": {"max_depth": int(max_depth)}},
        "trace": dict(trace),
    }

