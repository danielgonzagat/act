from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_objects_v132 import connected_components4_v132, object_set_summary_v132
from .arc_ops_v132 import OP_DEFS_V132, StateV132, apply_op_v132, propose_step_variants_v132, step_cost_bits_v132
from .grid_v124 import GridV124, grid_equal_v124, grid_hash_v124, grid_shape_v124, unique_colors_v124

ARC_SOLVER_SCHEMA_VERSION_V132 = 132


def _validate_grid_values_v132(g: GridV124) -> None:
    for row in g:
        for x in row:
            xx = int(x)
            if xx < 0 or xx > 9:
                raise ValueError("grid_cell_out_of_range")


@dataclass(frozen=True)
class ProgramStepV132:
    op_id: str
    args: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"op_id": str(self.op_id)}
        a: Dict[str, Any] = {}
        for k in sorted(self.args.keys()):
            a[str(k)] = self.args[k]
        d["args"] = a
        return d


@dataclass(frozen=True)
class ProgramV132:
    steps: Tuple[ProgramStepV132, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V132),
            "kind": "arc_program_v132",
            "steps": [s.to_dict() for s in self.steps],
        }

    def program_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


def _program_cost_bits_v132(program: ProgramV132) -> int:
    bits = 0
    for s in program.steps:
        bits += step_cost_bits_v132(op_id=str(s.op_id), args=dict(s.args))
    return int(bits)


def _summarize_mismatch_v132(*, got: GridV124, want: GridV124) -> Dict[str, Any]:
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


def _abstract_slots_after_steps_v132(steps: Sequence[ProgramStepV132]) -> Dict[str, bool]:
    # Abstract interpretation: which slots are available (non-None) assuming ops succeed.
    avail: Dict[str, bool] = {"grid": True, "objset": False, "obj": False, "bbox": False, "patch": False}
    for st in steps:
        od = OP_DEFS_V132.get(str(st.op_id))
        if od is None:
            continue
        # reads must be available for type-correct programs
        for r in od.reads:
            if not bool(avail.get(str(r), False)):
                return {"grid": True, "objset": False, "obj": False, "bbox": False, "patch": False, "invalid": True}
        for w in od.writes:
            avail[str(w)] = True
        # commit_patch clears patch in the concrete state; keep abstract patch available false to reduce branching.
        if str(st.op_id) == "commit_patch":
            avail["patch"] = False
        # new_canvas resets derived slots
        if str(st.op_id) == "new_canvas":
            avail["objset"] = False
            avail["obj"] = False
            avail["bbox"] = False
            avail["patch"] = False
        # grid-to-grid ops reset derived slots
        if str(st.op_id) in {"rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v", "translate", "crop_bbox_nonzero", "pad_to", "replace_color", "map_colors"}:
            avail["objset"] = False
            avail["obj"] = False
            avail["bbox"] = False
            avail["patch"] = False
    return avail


def apply_program_v132(program: ProgramV132, g_in: GridV124) -> GridV124:
    _validate_grid_values_v132(g_in)
    st = StateV132(grid=g_in)
    for step in program.steps:
        st = apply_op_v132(state=st, op_id=str(step.op_id), args=dict(step.args))
        _validate_grid_values_v132(st.grid)
        if st.patch is not None:
            _validate_grid_values_v132(st.patch)
    return st.grid


def solve_arc_task_v132(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    max_depth: int = 4,
    max_programs: int = 4000,
    trace_program_limit: int = 80,
    max_translate_shift: int = 2,
) -> Dict[str, Any]:
    """
    ARC solver v132: object-centric, stateful typed ops with deterministic MDL search.
    FAIL-CLOSED: multiple minimal-cost solutions that diverge on test_in -> UNKNOWN.
    """
    try:
        _validate_grid_values_v132(test_in)
        for inp, out in train_pairs:
            _validate_grid_values_v132(inp)
            _validate_grid_values_v132(out)
    except Exception as e:
        return {
            "schema_version": 132,
            "kind": "arc_solve_result_v132",
            "status": "FAIL",
            "failure_reason": {"kind": "INVARIANT_VIOLATION", "details": {"error": str(e)}},
        }

    if not train_pairs:
        return {
            "schema_version": 132,
            "kind": "arc_solve_result_v132",
            "status": "FAIL",
            "failure_reason": {"kind": "TYPE_MISMATCH", "details": {"error": "no_train_pairs"}},
        }

    # Concept trace (audit only): palette, bg candidates, object summaries (bg=0)
    palette: List[int] = []
    shapes_out: List[Tuple[int, int]] = []
    for inp, out in train_pairs:
        palette.extend(unique_colors_v124(inp))
        palette.extend(unique_colors_v124(out))
        shapes_out.append(grid_shape_v124(out))
    palette.extend(unique_colors_v124(test_in))
    palette_s = sorted(set(int(c) for c in palette))

    obj_summaries: List[Dict[str, Any]] = []
    for inp, _ in train_pairs:
        oset = connected_components4_v132(inp, bg=0)
        obj_summaries.append(object_set_summary_v132(oset, top_k=5))

    # Precompute step variants and costs.
    step_variants = propose_step_variants_v132(train_pairs=list(train_pairs), test_in=test_in, max_translate_shift=int(max_translate_shift))
    steps: List[ProgramStepV132] = []
    step_fps: List[str] = []
    step_costs: List[int] = []
    for st in step_variants:
        ps = ProgramStepV132(op_id=str(st["op_id"]), args=dict(st.get("args", {})))
        steps.append(ps)
        step_fps.append(canonical_json_dumps(ps.to_dict()))
        step_costs.append(step_cost_bits_v132(op_id=str(ps.op_id), args=dict(ps.args)))

    def _child_sig(parent_sig: str, step_fp: str) -> str:
        return sha256_hex((str(parent_sig) + "|" + str(step_fp)).encode("utf-8"))

    root_sig = sha256_hex(b"arc_program_v132_root")
    frontier: List[Tuple[int, int, str, int, Tuple[ProgramStepV132, ...]]] = []
    push_idx = 0
    heapq.heappush(frontier, (0, 0, str(root_sig), push_idx, tuple()))

    seen: set[str] = set()
    best_cost: Optional[int] = None
    solutions: List[ProgramV132] = []

    tried = 0
    budget_exhausted = False
    trace_programs: List[Dict[str, Any]] = []

    while frontier:
        cost_bits, plen, sig, _, prefix_steps = heapq.heappop(frontier)
        if sig in seen:
            continue
        seen.add(sig)

        if best_cost is not None and int(cost_bits) > int(best_cost):
            break

        program = ProgramV132(steps=tuple(prefix_steps))

        ok_train = True
        mismatch: Optional[Dict[str, Any]] = None
        for inp, out in train_pairs:
            try:
                got = apply_program_v132(program, inp)
            except Exception as e:
                ok_train = False
                mismatch = {"kind": "apply_error", "error": str(e)}
                break
            if not grid_equal_v124(got, out):
                ok_train = False
                mismatch = _summarize_mismatch_v132(got=got, want=out)
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

        # Type discipline: only expand with steps whose reads are available abstractly.
        avail = _abstract_slots_after_steps_v132(prefix_steps)
        if bool(avail.get("invalid", False)):
            continue

        for step_idx, step in enumerate(steps):
            od = OP_DEFS_V132.get(str(step.op_id))
            if od is None:
                continue
            if any(not bool(avail.get(str(r), False)) for r in od.reads):
                continue
            child_sig = _child_sig(str(sig), str(step_fps[step_idx]))
            if child_sig in seen:
                continue
            child_steps = tuple(list(prefix_steps) + [step])
            child_cost = int(cost_bits) + int(step_costs[step_idx])
            push_idx += 1
            heapq.heappush(frontier, (int(child_cost), int(plen + 1), str(child_sig), int(push_idx), child_steps))

    trace = {
        "schema_version": 132,
        "kind": "arc_trace_v132",
        "candidates_tested": int(tried),
        "candidates_kept_in_trace": int(len(trace_programs)),
        "trace_programs": list(trace_programs),
        "budget_exhausted": bool(budget_exhausted),
        "max_programs": int(max_programs),
        "max_depth": int(max_depth),
        "step_variants_total": int(len(steps)),
        "step_variants_by_op": {
            op: int(sum(1 for s in steps if str(s.op_id) == op)) for op in sorted(set(str(s.op_id) for s in steps))
        },
    }

    concept_trace = {
        "schema_version": 132,
        "kind": "arc_concept_trace_v132",
        "palette": [int(c) for c in palette_s],
        "train_obj_summaries_bg0": list(obj_summaries),
    }

    if solutions:
        sols_sorted = sorted(solutions, key=lambda p: p.program_sig())
        best_program = sols_sorted[0]
        outputs: List[Tuple[str, GridV124]] = []
        for p in sols_sorted:
            pred = apply_program_v132(p, test_in)
            outputs.append((p.program_sig(), pred))
        pred_hashes = {sig: grid_hash_v124(g) for sig, g in outputs}
        out_hashes = sorted(set(pred_hashes.values()))
        if len(out_hashes) == 1:
            pred_grid = outputs[0][1]
            return {
                "schema_version": 132,
                "kind": "arc_solve_result_v132",
                "status": "SOLVED",
                "program": [s.to_dict() for s in best_program.steps],
                "program_sig": best_program.program_sig(),
                "program_cost_bits": int(_program_cost_bits_v132(best_program)),
                "predicted_grid_hash": grid_hash_v124(pred_grid),
                "predicted_output": [list(r) for r in pred_grid],
                "concept_trace": dict(concept_trace),
                "trace": dict(trace),
            }
        return {
            "schema_version": 132,
            "kind": "arc_solve_result_v132",
            "status": "UNKNOWN",
            "failure_reason": {
                "kind": "AMBIGUOUS_RULE",
                "details": {"solutions": int(len(solutions)), "distinct_predicted_outputs": int(len(out_hashes))},
            },
            "candidate_program_sigs": [p.program_sig() for p in sols_sorted],
            "predicted_grid_hash_by_solution": {str(k): str(pred_hashes[k]) for k in sorted(pred_hashes.keys())},
            "concept_trace": dict(concept_trace),
            "trace": dict(trace),
        }

    if budget_exhausted:
        return {
            "schema_version": 132,
            "kind": "arc_solve_result_v132",
            "status": "FAIL",
            "failure_reason": {
                "kind": "SEARCH_BUDGET_EXCEEDED",
                "details": {"candidates_tested": int(tried), "max_programs": int(max_programs), "max_depth": int(max_depth)},
            },
            "concept_trace": dict(concept_trace),
            "trace": dict(trace),
        }

    return {
        "schema_version": 132,
        "kind": "arc_solve_result_v132",
        "status": "FAIL",
        "failure_reason": {"kind": "MISSING_OPERATOR", "details": {"max_depth": int(max_depth)}},
        "concept_trace": dict(concept_trace),
        "trace": dict(trace),
    }

