from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .grid_v124 import (
    GridV124,
    bbox_nonzero_v124,
    crop_to_bbox_nonzero_v124,
    grid_equal_v124,
    grid_from_list_v124,
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

ARC_SOLVER_SCHEMA_VERSION_V124 = 124


@dataclass(frozen=True)
class ProgramStepV124:
    op: str
    args: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        if not self.args:
            return {"op": str(self.op)}
        d = {"op": str(self.op)}
        for k in sorted(self.args.keys()):
            d[str(k)] = self.args[k]
        return d


@dataclass(frozen=True)
class ProgramV124:
    steps: Tuple[ProgramStepV124, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V124), "steps": [s.to_dict() for s in self.steps]}

    def program_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


def _replace_color_v124(g: GridV124, *, from_color: int, to_color: int) -> GridV124:
    fc = int(from_color)
    tc = int(to_color)
    if fc < 0 or fc > 9 or tc < 0 or tc > 9:
        raise ValueError("color_out_of_range")
    return tuple(tuple(tc if int(x) == fc else int(x) for x in row) for row in g)


def _map_colors_v124(g: GridV124, *, mapping: Dict[str, int]) -> GridV124:
    m: Dict[int, int] = {}
    for k, v in mapping.items():
        kk = int(k)
        vv = int(v)
        if kk < 0 or kk > 9 or vv < 0 or vv > 9:
            raise ValueError("color_out_of_range")
        m[kk] = vv
    return tuple(tuple(int(m.get(int(x), int(x))) for x in row) for row in g)


def apply_program_v124(program: ProgramV124, g: GridV124) -> GridV124:
    cur = g
    for step in program.steps:
        op = str(step.op)
        a = dict(step.args)
        if op == "identity":
            cur = cur
        elif op == "rotate90":
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
            cur = _replace_color_v124(cur, from_color=int(a["from_color"]), to_color=int(a["to_color"]))
        elif op == "map_colors":
            cur = _map_colors_v124(cur, mapping=dict(a.get("mapping", {})))
        else:
            raise ValueError(f"unknown_op:{op}")
    return cur


def _program_cost_bits_v124(program: ProgramV124) -> int:
    # Minimal, explicit and deterministic MDL proxy:
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


def _infer_color_mapping_v124(inp: GridV124, out: GridV124) -> Optional[Dict[str, int]]:
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
    # canonicalize as str->int for json stability
    return {str(k): int(mapping[k]) for k in sorted(mapping.keys())}


def _op_variants_v124(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[ProgramStepV124]:
    # Deterministic parameter sets derived from training data (no task_id).
    colors: List[int] = []
    shapes_out: List[Tuple[int, int]] = []
    inferred_maps: List[Dict[str, int]] = []
    for inp, out in train_pairs:
        colors.extend(unique_colors_v124(inp))
        colors.extend(unique_colors_v124(out))
        shapes_out.append(grid_shape_v124(out))
        m = _infer_color_mapping_v124(inp, out)
        if m is not None:
            inferred_maps.append(m)
    colors = sorted(set(int(c) for c in colors))
    shapes_out = sorted(set((int(h), int(w)) for h, w in shapes_out))
    inferred_maps = sorted(inferred_maps, key=lambda m: canonical_json_dumps(m))

    steps: List[ProgramStepV124] = []
    # Identity + basic transforms
    steps.append(ProgramStepV124(op="identity", args={}))
    for op in ["rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v", "crop_bbox_nonzero"]:
        steps.append(ProgramStepV124(op=op, args={}))

    # Small translations (bounded, deterministic)
    for dy in [-2, -1, 0, 1, 2]:
        for dx in [-2, -1, 0, 1, 2]:
            if dx == 0 and dy == 0:
                continue
            steps.append(ProgramStepV124(op="translate", args={"dx": int(dx), "dy": int(dy), "pad": 0}))

    # pad_to shapes seen in outputs
    for h, w in shapes_out:
        steps.append(ProgramStepV124(op="pad_to", args={"height": int(h), "width": int(w), "pad": 0}))

    # replace_color from observed colors
    for fc in colors:
        for tc in colors:
            if fc == tc:
                continue
            steps.append(ProgramStepV124(op="replace_color", args={"from_color": int(fc), "to_color": int(tc)}))

    # map_colors inferred from at least one training pair
    for m in inferred_maps:
        steps.append(ProgramStepV124(op="map_colors", args={"mapping": dict(m)}))

    # Deterministic ordering
    steps.sort(key=lambda s: (str(s.op), canonical_json_dumps(s.to_dict())))
    return steps


def _summarize_mismatch_v124(*, got: GridV124, want: GridV124) -> Dict[str, Any]:
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


def solve_arc_task_v124(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    max_depth: int = 3,
    max_programs: int = 2000,
    trace_program_limit: int = 50,
) -> Dict[str, Any]:
    """
    Baseline compositional solver (honest, deterministic).
    Returns a dict with status SOLVED|UNKNOWN|FAIL + trace.
    """
    ops = _op_variants_v124(train_pairs=list(train_pairs))

    # Deterministic best-first search over programs by (cost_bits, length, program_sig).
    # Using a heap avoids quadratic frontier sorting while preserving stable tie-breaks.
    step_fps: List[str] = [canonical_json_dumps(s.to_dict()) for s in ops]
    step_costs: List[int] = []
    for s in ops:
        step_costs.append(10 + sum(8 * int(len(v)) if k == "mapping" and isinstance(v, dict) else 4 for k, v in s.args.items()))

    def _child_sig(parent_sig: str, step_fp: str) -> str:
        return sha256_hex((str(parent_sig) + "|" + str(step_fp)).encode("utf-8"))

    root_sig = sha256_hex(b"arc_program_v124_root")
    frontier: List[Tuple[int, int, str, int, Tuple[ProgramStepV124, ...]]] = []
    push_idx = 0
    heapq.heappush(frontier, (0, 0, str(root_sig), push_idx, tuple()))

    seen: set[str] = set()
    best: Optional[ProgramV124] = None
    best_cost: Optional[int] = None
    best_sigs: List[str] = []

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

        program = ProgramV124(steps=tuple(steps))

        # Evaluate on training pairs
        ok_train = True
        mismatch: Optional[Dict[str, Any]] = None
        for inp, out in train_pairs:
            try:
                got = apply_program_v124(program, inp)
            except Exception as e:
                ok_train = False
                mismatch = {"kind": "apply_error", "error": str(e)}
                break
            if not grid_equal_v124(got, out):
                ok_train = False
                mismatch = _summarize_mismatch_v124(got=got, want=out)
                break

        tried += 1
        if len(trace_programs) < int(trace_program_limit):
            trace_programs.append(
                {
                    "program_sig": str(sig),
                    "cost_bits": int(cost_bits),
                    "steps": [s.to_dict() for s in program.steps],
                    "ok_train": bool(ok_train),
                    "mismatch": dict(mismatch) if isinstance(mismatch, dict) else None,
                }
            )

        if ok_train:
            if best_cost is None or int(cost_bits) < int(best_cost):
                best = program
                best_cost = int(cost_bits)
                best_sigs = [str(sig)]
            elif int(cost_bits) == int(best_cost):
                best_sigs.append(str(sig))

        if tried >= int(max_programs):
            budget_exhausted = True
            break

        if best_cost is not None:
            # Do not expand after first solution; children would have higher cost anyway.
            continue

        if int(plen) >= int(max_depth):
            continue

        # Expand deterministically by appending each op variant.
        for step, step_fp, step_c in zip(ops, step_fps, step_costs):
            new_steps = tuple(list(steps) + [step])
            new_cost = int(cost_bits) + int(step_c)
            new_len = int(plen) + 1
            new_sig = _child_sig(str(sig), str(step_fp))
            if new_sig in seen:
                continue
            push_idx += 1
            heapq.heappush(frontier, (int(new_cost), int(new_len), str(new_sig), int(push_idx), new_steps))

    status = "FAIL"
    failure_reason: Dict[str, Any] = {}
    predicted: Optional[GridV124] = None
    if best is None:
        if budget_exhausted:
            status = "FAIL"
            failure_reason = {"kind": "SEARCH_BUDGET_EXCEEDED", "max_programs": int(max_programs), "max_depth": int(max_depth)}
        else:
            status = "FAIL"
            failure_reason = {"kind": "NO_CONSISTENT_PROGRAM", "max_depth": int(max_depth)}
    else:
        if len(best_sigs) > 1:
            status = "UNKNOWN"
            failure_reason = {"kind": "AMBIGUOUS_RULE", "solutions": list(sorted(set(best_sigs)))}
        else:
            status = "SOLVED"
            predicted = apply_program_v124(best, test_in)

    out: Dict[str, Any] = {
        "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V124),
        "status": str(status),
        "program": best.to_dict() if best is not None and len(best_sigs) == 1 else None,
        "program_cost_bits": int(best_cost) if best_cost is not None and len(best_sigs) == 1 else None,
        "predicted_grid": [list(r) for r in predicted] if predicted is not None else None,
        "predicted_grid_hash": grid_hash_v124(predicted) if predicted is not None else "",
        "failure_reason": dict(failure_reason) if failure_reason else None,
        "trace": {
            "programs_tried": int(tried),
            "budget_exhausted": bool(budget_exhausted),
            "trace_programs": list(trace_programs),
        },
    }
    out["result_sig"] = sha256_hex(canonical_json_dumps(out).encode("utf-8"))
    return out


def diagnose_missing_operator_v124(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> Dict[str, Any]:
    """
    Heuristic, deterministic diagnostic (not a solver):
    derives coarse "operator gaps" from train I/O deltas.
    """
    shape_change = False
    color_change = False
    out_new_colors = set()
    in_colors = set()
    for inp, out in train_pairs:
        if grid_shape_v124(inp) != grid_shape_v124(out):
            shape_change = True
        in_colors.update(unique_colors_v124(inp))
        out_colors = set(unique_colors_v124(out))
        out_new_colors.update(out_colors - set(unique_colors_v124(inp)))
        if set(unique_colors_v124(inp)) != out_colors:
            color_change = True
    gaps: List[str] = []
    if shape_change:
        gaps.append("shape_transform_needed")
    if color_change:
        gaps.append("color_transform_needed")
    if out_new_colors:
        gaps.append("introduces_new_colors")
    return {"gaps": gaps, "out_new_colors": [int(c) for c in sorted(out_new_colors)], "in_colors": [int(c) for c in sorted(in_colors)]}
