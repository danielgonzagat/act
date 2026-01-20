from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_dsl_v125 import (
    BboxV125,
    ObjectSetV125,
    OpDefV125,
    OP_DEFS_V125,
    apply_op_v125,
    bbox_nonzero_v125,
)
from .grid_v124 import (
    GridV124,
    crop_to_bbox_nonzero_v124,
    grid_equal_v124,
    grid_hash_v124,
    grid_shape_v124,
    reflect_h_v124,
    reflect_v_v124,
    rotate180_v124,
    rotate270_v124,
    rotate90_v124,
    translate_v124,
    unique_colors_v124,
)

ARC_SOLVER_SCHEMA_VERSION_V125 = 125


@dataclass(frozen=True)
class ProgramStepV125:
    op_id: str
    in_vars: Tuple[str, ...]
    out_var: str
    args: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"op_id": str(self.op_id), "in_vars": list(self.in_vars), "out_var": str(self.out_var)}
        for k in sorted(self.args.keys()):
            d[str(k)] = self.args[k]
        return d


@dataclass(frozen=True)
class ProgramV125:
    steps: Tuple[ProgramStepV125, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V125), "steps": [s.to_dict() for s in self.steps]}

    def program_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


def _is_grid_op(op_id: str) -> bool:
    # Grid-to-grid transforms and edits.
    return op_id in {
        "identity",
        "rotate90",
        "rotate180",
        "rotate270",
        "reflect_h",
        "reflect_v",
        "translate",
        "crop_bbox_nonzero",
        "crop_bbox",
        "fill_rect",
        "draw_rect_border",
        "paint_mask",
        "overlay",
        "pad_to",
        "replace_color",
        "map_colors",
    }


def _step_cost_bits_v125(step: ProgramStepV125) -> int:
    op = str(step.op_id)
    if op in OP_DEFS_V125:
        base = int(OP_DEFS_V125[op].cost_bits)
    else:
        base = 10
    extra = 0
    for k, v in step.args.items():
        if k == "mapping" and isinstance(v, dict):
            extra += 8 * int(len(v))
        else:
            extra += 4
    # Referencing more inputs costs slightly (explicit causal wiring).
    extra += 2 * max(0, int(len(step.in_vars)) - 1)
    return int(base + extra)


def _program_cost_bits_v125(steps: Sequence[ProgramStepV125]) -> int:
    return int(sum(_step_cost_bits_v125(s) for s in steps))


def _infer_color_mapping_v125(inp: GridV124, out: GridV124) -> Optional[Dict[str, int]]:
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


def _summarize_mismatch_v125(*, got: GridV124, want: GridV124) -> Dict[str, Any]:
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


def _apply_step_v125(*, env: Dict[str, Any], step: ProgramStepV125) -> None:
    op = str(step.op_id)
    if op == "identity":
        env[str(step.out_var)] = env[str(step.in_vars[0])]
        return
    if op == "rotate90":
        env[str(step.out_var)] = rotate90_v124(env[str(step.in_vars[0])])
        return
    if op == "rotate180":
        env[str(step.out_var)] = rotate180_v124(env[str(step.in_vars[0])])
        return
    if op == "rotate270":
        env[str(step.out_var)] = rotate270_v124(env[str(step.in_vars[0])])
        return
    if op == "reflect_h":
        env[str(step.out_var)] = reflect_h_v124(env[str(step.in_vars[0])])
        return
    if op == "reflect_v":
        env[str(step.out_var)] = reflect_v_v124(env[str(step.in_vars[0])])
        return
    if op == "translate":
        g = env[str(step.in_vars[0])]
        env[str(step.out_var)] = translate_v124(g, dx=int(step.args["dx"]), dy=int(step.args["dy"]), pad=int(step.args.get("pad", 0)))
        return
    if op == "crop_bbox_nonzero":
        g = env[str(step.in_vars[0])]
        env[str(step.out_var)] = crop_to_bbox_nonzero_v124(g, bg=int(step.args.get("bg", 0)))
        return
    if op in OP_DEFS_V125:
        ins = [env[str(vn)] for vn in step.in_vars]
        env[str(step.out_var)] = apply_op_v125(op_id=str(op), inputs=ins, args=dict(step.args))
        return
    if op == "replace_color":
        g = env[str(step.in_vars[0])]
        fc = int(step.args["from_color"])
        tc = int(step.args["to_color"])
        env[str(step.out_var)] = tuple(tuple(tc if int(x) == fc else int(x) for x in row) for row in g)
        return
    if op == "map_colors":
        g = env[str(step.in_vars[0])]
        m_raw = step.args.get("mapping", {})
        if not isinstance(m_raw, dict):
            raise ValueError("mapping_not_dict")
        m: Dict[int, int] = {int(k): int(v) for k, v in m_raw.items()}
        env[str(step.out_var)] = tuple(tuple(int(m.get(int(x), int(x))) for x in row) for row in g)
        return
    if op == "pad_to":
        g = env[str(step.in_vars[0])]
        hh = int(step.args["height"])
        ww = int(step.args["width"])
        pad = int(step.args.get("pad", 0))
        out: List[List[int]] = [[int(pad) for _ in range(ww)] for _ in range(hh)]
        h, w = grid_shape_v124(g)
        for r in range(min(h, hh)):
            for c in range(min(w, ww)):
                out[r][c] = int(g[r][c])
        env[str(step.out_var)] = tuple(tuple(int(x) for x in row) for row in out)
        return
    raise ValueError(f"unknown_op:{op}")


def apply_program_v125(program: ProgramV125, g_in: GridV124) -> GridV124:
    env: Dict[str, Any] = {"g0": g_in}
    for step in program.steps:
        _apply_step_v125(env=env, step=step)
    # Output is the latest grid value in env (by last assignment).
    # Convention: the last step must output a GRID, or program has no effect (identity).
    if not program.steps:
        return g_in
    out = env[str(program.steps[-1].out_var)]
    if not (isinstance(out, tuple) and (not out or isinstance(out[0], tuple))):
        raise ValueError("program_output_not_grid")
    return out  # type: ignore[return-value]


def _op_variants_v125(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[Tuple[str, Dict[str, Any]]]:
    colors_in: List[int] = []
    colors_out: List[int] = []
    shapes_out: List[Tuple[int, int]] = []
    inferred_maps: List[Dict[str, int]] = []
    for inp, out in train_pairs:
        colors_in.extend(unique_colors_v124(inp))
        colors_out.extend(unique_colors_v124(out))
        shapes_out.append(grid_shape_v124(out))
        m = _infer_color_mapping_v125(inp, out)
        if m is not None:
            inferred_maps.append(m)
    colors_in_s = sorted(set(int(c) for c in colors_in))
    colors_out_s = sorted(set(int(c) for c in colors_out))
    colors_all = sorted(set(colors_in_s + colors_out_s))
    shapes_out = sorted(set((int(h), int(w)) for h, w in shapes_out))
    inferred_maps = sorted(inferred_maps, key=lambda m: canonical_json_dumps(m))

    vars_list: List[Tuple[str, Dict[str, Any]]] = []
    # Basic transforms
    for op in ["identity", "rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v", "crop_bbox_nonzero"]:
        vars_list.append((op, {}))

    # Translations (bounded)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            vars_list.append(("translate", {"dx": int(dx), "dy": int(dy), "pad": 0}))

    # pad_to shapes in outputs
    # Only include when at least one training pair changes shape.
    if any(grid_shape_v124(inp) != grid_shape_v124(out) for inp, out in train_pairs):
        for h, w in shapes_out:
            vars_list.append(("pad_to", {"height": int(h), "width": int(w), "pad": 0}))

    # Replace color
    for fc in colors_in_s:
        for tc in colors_out_s:
            if fc == tc:
                continue
            vars_list.append(("replace_color", {"from_color": int(fc), "to_color": int(tc)}))

    # Map colors inferred
    for m in inferred_maps:
        vars_list.append(("map_colors", {"mapping": dict(m)}))

    # Latent + edit operators
    # Bounding boxes/masks are typically extracted from existing colors in the input.
    bg = 0
    for c in colors_in_s:
        if int(c) == int(bg):
            continue
        vars_list.append(("bbox_by_color", {"color": int(c)}))
        vars_list.append(("mask_by_color", {"color": int(c)}))
    vars_list.append(("bbox_nonzero", {"bg": 0}))

    # Painting uses colors that appear in outputs (including new colors).
    for c in colors_out_s:
        vars_list.append(("fill_rect", {"color": int(c)}))
        vars_list.append(("draw_rect_border", {"color": int(c), "thickness": 1}))
        vars_list.append(("paint_mask", {"color": int(c)}))

    vars_list.append(("connected_components", {}))
    # Component extraction by explicit color (exclude likely background when possible).
    for c in colors_in_s:
        if int(c) == int(bg):
            continue
        vars_list.append(("connected_components", {"color": int(c)}))
    for sel in ["largest_area", "smallest_area", "leftmost", "topmost"]:
        vars_list.append(("select_object", {"selector": str(sel)}))
    vars_list.append(("bbox_of_object", {}))

    # Deterministic ordering
    vars_list.sort(key=lambda oa: (str(oa[0]), canonical_json_dumps(oa[1])))
    return vars_list


def _type_of_var_name_v125(name: str) -> str:
    if name.startswith("g"):
        return "GRID"
    if name.startswith("bb"):
        return "BBOX"
    if name.startswith("m"):
        return "MASK"
    if name.startswith("os"):
        return "OBJECT_SET"
    if name.startswith("o"):
        return "OBJECT"
    return "UNKNOWN"


def solve_arc_task_v125(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    max_depth: int = 4,
    max_programs: int = 8000,
    trace_program_limit: int = 80,
) -> Dict[str, Any]:
    """
    Compositional, typed-ish solver with explicit latent extraction and structural edits.
    Deterministic; FAIL-CLOSED on ambiguity.
    """
    op_variants = _op_variants_v125(train_pairs=list(train_pairs))

    def _step_fp(step: ProgramStepV125) -> str:
        return canonical_json_dumps(step.to_dict())

    def _child_sig(parent_sig: str, step_fp: str) -> str:
        return sha256_hex((str(parent_sig) + "|" + str(step_fp)).encode("utf-8"))

    # --- Stage 0: template-driven synthesis (general patterns) ---
    def _eval_program(program: ProgramV125) -> Tuple[bool, Optional[Dict[str, Any]]]:
        if program.steps:
            last_var = program.steps[-1].out_var
            if _type_of_var_name_v125(str(last_var)) != "GRID":
                return False, {"kind": "incomplete_program_no_grid_output", "last_var": str(last_var)}
        for inp, out in train_pairs:
            try:
                got = apply_program_v125(program, inp)
            except Exception as e:
                return False, {"kind": "apply_error", "error": str(e)}
            if not grid_equal_v124(got, out):
                return False, _summarize_mismatch_v125(got=got, want=out)
        return True, None

    # Derive colors to parameterize templates deterministically.
    colors_in: List[int] = []
    colors_out: List[int] = []
    for inp, out in train_pairs:
        colors_in.extend(unique_colors_v124(inp))
        colors_out.extend(unique_colors_v124(out))
    colors_in_s = sorted(set(int(c) for c in colors_in))
    colors_out_s = sorted(set(int(c) for c in colors_out))
    bg = 0

    templates: List[ProgramV125] = []
    # Single-step transforms
    for op in ["identity", "rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v", "crop_bbox_nonzero"]:
        templates.append(ProgramV125(steps=(ProgramStepV125(op_id=str(op), in_vars=("g0",), out_var="g1", args={}),)))
    # Single-step replace/map
    for fc in colors_in_s:
        for tc in colors_out_s:
            if fc == tc:
                continue
            templates.append(
                ProgramV125(
                    steps=(ProgramStepV125(op_id="replace_color", in_vars=("g0",), out_var="g1", args={"from_color": int(fc), "to_color": int(tc)}),)
                )
            )
    # Latent bbox -> border/fill
    templates.append(ProgramV125(steps=(ProgramStepV125(op_id="bbox_nonzero", in_vars=("g0",), out_var="bb1", args={"bg": int(bg)}),)))
    for c_sel in [c for c in colors_in_s if int(c) != int(bg)]:
        templates.append(ProgramV125(steps=(ProgramStepV125(op_id="bbox_by_color", in_vars=("g0",), out_var="bb1", args={"color": int(c_sel)}),)))
    for c_paint in colors_out_s:
        # bbox_nonzero -> draw_border / fill
        templates.append(
            ProgramV125(
                steps=(
                    ProgramStepV125(op_id="bbox_nonzero", in_vars=("g0",), out_var="bb1", args={"bg": int(bg)}),
                    ProgramStepV125(op_id="draw_rect_border", in_vars=("g0", "bb1"), out_var="g2", args={"color": int(c_paint), "thickness": 1}),
                )
            )
        )
        templates.append(
            ProgramV125(
                steps=(
                    ProgramStepV125(op_id="bbox_nonzero", in_vars=("g0",), out_var="bb1", args={"bg": int(bg)}),
                    ProgramStepV125(op_id="fill_rect", in_vars=("g0", "bb1"), out_var="g2", args={"color": int(c_paint)}),
                )
            )
        )
        for c_sel in [c for c in colors_in_s if int(c) != int(bg)]:
            templates.append(
                ProgramV125(
                    steps=(
                        ProgramStepV125(op_id="bbox_by_color", in_vars=("g0",), out_var="bb1", args={"color": int(c_sel)}),
                        ProgramStepV125(op_id="draw_rect_border", in_vars=("g0", "bb1"), out_var="g2", args={"color": int(c_paint), "thickness": 1}),
                    )
                )
            )
            templates.append(
                ProgramV125(
                    steps=(
                        ProgramStepV125(op_id="bbox_by_color", in_vars=("g0",), out_var="bb1", args={"color": int(c_sel)}),
                        ProgramStepV125(op_id="fill_rect", in_vars=("g0", "bb1"), out_var="g2", args={"color": int(c_paint)}),
                    )
                )
            )
            for c_border in colors_out_s:
                if int(c_border) == int(c_paint):
                    continue
                templates.append(
                    ProgramV125(
                        steps=(
                            ProgramStepV125(op_id="bbox_by_color", in_vars=("g0",), out_var="bb1", args={"color": int(c_sel)}),
                            ProgramStepV125(op_id="fill_rect", in_vars=("g0", "bb1"), out_var="g2", args={"color": int(c_paint)}),
                            ProgramStepV125(op_id="draw_rect_border", in_vars=("g2", "bb1"), out_var="g3", args={"color": int(c_border), "thickness": 1}),
                        )
                    )
                )

    # Connected components by explicit color -> bbox -> fill
    for c_sel in [c for c in colors_in_s if int(c) != int(bg)]:
        for sel in ["largest_area", "smallest_area", "leftmost", "topmost"]:
            for c_paint in colors_out_s:
                templates.append(
                    ProgramV125(
                        steps=(
                            ProgramStepV125(op_id="connected_components", in_vars=("g0",), out_var="os1", args={"color": int(c_sel)}),
                            ProgramStepV125(op_id="select_object", in_vars=("os1",), out_var="o2", args={"selector": str(sel)}),
                            ProgramStepV125(op_id="bbox_of_object", in_vars=("o2",), out_var="bb3", args={}),
                            ProgramStepV125(op_id="fill_rect", in_vars=("g0", "bb3"), out_var="g4", args={"color": int(c_paint)}),
                        )
                    )
                )

    # Sort templates by MDL proxy (cost, len, program_sig).
    templates.sort(key=lambda p: (int(_program_cost_bits_v125(p.steps)), int(len(p.steps)), str(p.program_sig())))

    best: Optional[ProgramV125] = None
    best_cost: Optional[int] = None
    best_sigs: List[str] = []
    best_program_json: Optional[Dict[str, Any]] = None

    tried = 0
    trace_programs: List[Dict[str, Any]] = []
    budget_exhausted = False

    for program in templates:
        sig = program.program_sig()
        cost_bits = int(_program_cost_bits_v125(program.steps))
        if best_cost is not None and cost_bits > int(best_cost):
            break
        ok_train, mismatch = _eval_program(program)
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
                best_program_json = program.to_dict()
            elif int(cost_bits) == int(best_cost):
                best_sigs.append(str(sig))

    # --- Stage 1: fallback best-first search (if templates didn't solve) ---
    if best is None:
        root_sig = sha256_hex(b"arc_program_v125_root")
        frontier: List[Tuple[int, int, str, int, Tuple[ProgramStepV125, ...], Tuple[Tuple[str, str], ...]]] = []
        push_idx = 0
        heapq.heappush(frontier, (0, 0, str(root_sig), push_idx, tuple(), (("g0", "GRID"),)))
        seen: set[str] = set()

        while frontier:
            cost_bits, plen, sig, _, steps, var_types = heapq.heappop(frontier)
            if sig in seen:
                continue
            seen.add(sig)
            program = ProgramV125(steps=tuple(steps))
            ok_train, mismatch = _eval_program(program)
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
                    best_program_json = program.to_dict()
                elif int(cost_bits) == int(best_cost):
                    best_sigs.append(str(sig))
            if tried >= int(max_programs):
                budget_exhausted = True
                break
            if int(plen) >= int(max_depth):
                continue
            if best_cost is not None:
                continue

            grid_vars = [vn for vn, t in var_types if t == "GRID"]
            cur_grid_var = grid_vars[-1] if grid_vars else "g0"
            for op_id, args in op_variants:
                op = str(op_id)
                in_vars: List[str] = []
                if op in {"identity", "rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v", "translate", "crop_bbox_nonzero", "pad_to", "replace_color", "map_colors"}:
                    in_vars = [str(cur_grid_var)]
                    out_prefix = "g"
                    out_type = "GRID"
                elif op in {"bbox_nonzero", "bbox_by_color"}:
                    in_vars = [str(cur_grid_var)]
                    out_prefix = "bb"
                    out_type = "BBOX"
                elif op in {"mask_by_color"}:
                    in_vars = [str(cur_grid_var)]
                    out_prefix = "m"
                    out_type = "MASK"
                elif op in {"connected_components"}:
                    in_vars = [str(cur_grid_var)]
                    out_prefix = "os"
                    out_type = "OBJECT_SET"
                elif op in {"select_object"}:
                    os_vars = [vn for vn, t in var_types if t == "OBJECT_SET"]
                    if not os_vars:
                        continue
                    in_vars = [str(os_vars[-1])]
                    out_prefix = "o"
                    out_type = "OBJECT"
                elif op in {"bbox_of_object"}:
                    o_vars = [vn for vn, t in var_types if t == "OBJECT"]
                    if not o_vars:
                        continue
                    in_vars = [str(o_vars[-1])]
                    out_prefix = "bb"
                    out_type = "BBOX"
                elif op in {"crop_bbox", "fill_rect", "draw_rect_border"}:
                    bb_vars = [vn for vn, t in var_types if t == "BBOX"]
                    if not bb_vars:
                        continue
                    in_vars = [str(cur_grid_var), str(bb_vars[-1])]
                    out_prefix = "g"
                    out_type = "GRID"
                elif op in {"paint_mask"}:
                    m_vars = [vn for vn, t in var_types if t == "MASK"]
                    if not m_vars:
                        continue
                    in_vars = [str(cur_grid_var), str(m_vars[-1])]
                    out_prefix = "g"
                    out_type = "GRID"
                elif op in {"overlay"}:
                    g_vars = [vn for vn, t in var_types if t == "GRID"]
                    if len(g_vars) < 2:
                        continue
                    in_vars = [str(g_vars[-2]), str(g_vars[-1])]
                    out_prefix = "g"
                    out_type = "GRID"
                else:
                    continue

                out_var = f"{out_prefix}{len(steps) + 1}"
                step = ProgramStepV125(op_id=str(op), in_vars=tuple(in_vars), out_var=str(out_var), args=dict(args))
                step_bits = int(_step_cost_bits_v125(step))
                new_cost = int(cost_bits) + int(step_bits)
                new_len = int(plen) + 1
                new_sig = _child_sig(str(sig), _step_fp(step))
                push_idx += 1
                new_steps = tuple(list(steps) + [step])
                new_var_types = tuple(list(var_types) + [(str(out_var), str(out_type))])
                heapq.heappush(frontier, (int(new_cost), int(new_len), str(new_sig), int(push_idx), new_steps, new_var_types))

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
        if len(set(best_sigs)) > 1:
            status = "UNKNOWN"
            failure_reason = {"kind": "AMBIGUOUS_RULE", "solutions": list(sorted(set(best_sigs)))}
        else:
            status = "SOLVED"
            predicted = apply_program_v125(best, test_in)

    out: Dict[str, Any] = {
        "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V125),
        "status": str(status),
        "program": dict(best_program_json) if best_program_json is not None and len(set(best_sigs)) == 1 else None,
        "program_cost_bits": int(best_cost) if best_cost is not None and len(set(best_sigs)) == 1 else None,
        "predicted_grid": [list(r) for r in predicted] if predicted is not None else None,
        "predicted_grid_hash": grid_hash_v124(predicted) if predicted is not None else "",
        "failure_reason": dict(failure_reason) if failure_reason else None,
        "trace": {"programs_tried": int(tried), "budget_exhausted": bool(budget_exhausted), "trace_programs": list(trace_programs)},
    }
    out["result_sig"] = sha256_hex(canonical_json_dumps(out).encode("utf-8"))
    return out


def diagnose_missing_operator_v125(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> Dict[str, Any]:
    """
    Deterministic diagnostic (not a solver): derive coarse operator gaps.
    Produces gap tags suitable for an operator backlog.
    """
    gaps: List[str] = []
    shape_change = False
    out_new_colors = set()
    in_colors = set()
    for inp, out in train_pairs:
        if grid_shape_v124(inp) != grid_shape_v124(out):
            shape_change = True
        in_colors.update(unique_colors_v124(inp))
        out_colors = set(unique_colors_v124(out))
        out_new_colors |= (out_colors - set(unique_colors_v124(inp)))
    if shape_change:
        gaps.append("shape_change_present")
    if out_new_colors:
        gaps.append("introduces_new_colors")
    # If outputs contain nonzero bbox that differs from input bbox, likely needs bbox/rect ops.
    try:
        b_in = [bbox_nonzero_v125(inp).to_tuple() for inp, _ in train_pairs]
        b_out = [bbox_nonzero_v125(out).to_tuple() for _, out in train_pairs]
        if b_in != b_out:
            gaps.append("bbox_transform_or_draw_needed")
    except Exception:
        pass
    gaps = sorted(set(str(g) for g in gaps))
    return {
        "schema_version": 125,
        "kind": "arc_operator_gap_diag_v125",
        "gaps": gaps,
        "out_new_colors": [int(c) for c in sorted(out_new_colors)],
        "in_colors": [int(c) for c in sorted(in_colors)],
    }
