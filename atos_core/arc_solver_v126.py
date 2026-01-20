from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_concepts_v126 import ConceptV126, concept_trace_v126, propose_concepts_v126
from .arc_dsl_v126 import OP_DEFS_V126, ObjectSetV126, apply_op_v126
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

ARC_SOLVER_SCHEMA_VERSION_V126 = 126


@dataclass(frozen=True)
class ProgramStepV126:
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
class ProgramV126:
    steps: Tuple[ProgramStepV126, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V126), "steps": [s.to_dict() for s in self.steps]}

    def program_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


@dataclass(frozen=True)
class PlanCandidateV126:
    plan_kind: str
    params: Tuple[Tuple[str, Any], ...]
    concepts_used: Tuple[str, ...]
    program: ProgramV126

    def to_dict(self) -> Dict[str, Any]:
        p: Dict[str, Any] = {}
        for k, v in sorted(self.params, key=lambda kv: str(kv[0])):
            p[str(k)] = v
        return {
            "schema_version": 126,
            "kind": "arc_plan_candidate_v126",
            "plan_kind": str(self.plan_kind),
            "params": p,
            "concepts_used": list(self.concepts_used),
            "program": self.program.to_dict(),
        }

    def plan_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


def _step_cost_bits_v126(step: ProgramStepV126) -> int:
    op = str(step.op_id)
    if op in OP_DEFS_V126:
        base = int(OP_DEFS_V126[op].cost_bits)
    elif op in {"identity", "rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v"}:
        base = 8
    elif op in {"translate"}:
        base = 12
    elif op in {"crop_bbox_nonzero"}:
        base = 10
    elif op in {"replace_color", "map_colors"}:
        base = 16
    else:
        base = 20
    extra = 0
    for k, v in step.args.items():
        if k == "mapping" and isinstance(v, dict):
            extra += 8 * int(len(v))
        else:
            extra += 4
    extra += 2 * max(0, int(len(step.in_vars)) - 1)
    return int(base + extra)


def _program_cost_bits_v126(steps: Sequence[ProgramStepV126]) -> int:
    return int(sum(_step_cost_bits_v126(s) for s in steps))


def _summarize_mismatch_v126(*, got: GridV124, want: GridV124) -> Dict[str, Any]:
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


def _infer_color_mapping_v126(inp: GridV124, out: GridV124) -> Optional[Dict[str, int]]:
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


def _apply_step_v126(*, env: Dict[str, Any], step: ProgramStepV126) -> None:
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
        env[str(step.out_var)] = translate_v124(
            g, dx=int(step.args["dx"]), dy=int(step.args["dy"]), pad=int(step.args.get("pad", 0))
        )
        return
    if op == "crop_bbox_nonzero":
        g = env[str(step.in_vars[0])]
        env[str(step.out_var)] = crop_to_bbox_nonzero_v124(g, bg=int(step.args.get("bg", 0)))
        return
    if op in OP_DEFS_V126:
        ins = [env[str(vn)] for vn in step.in_vars]
        env[str(step.out_var)] = apply_op_v126(op_id=str(op), inputs=ins, args=dict(step.args))
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
    raise ValueError(f"unknown_op:{op}")


def apply_program_v126(program: ProgramV126, g_in: GridV124) -> GridV124:
    env: Dict[str, Any] = {"g0": g_in}
    for step in program.steps:
        _apply_step_v126(env=env, step=step)
    if not program.steps:
        return g_in
    out = env[str(program.steps[-1].out_var)]
    if not (isinstance(out, tuple) and (not out or isinstance(out[0], tuple))):
        raise ValueError("program_output_not_grid")
    return out  # type: ignore[return-value]


def _type_of_var_name_v126(name: str) -> str:
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


def _bbox_source_steps_from_concept_v126(concept: ConceptV126) -> Optional[Tuple[List[ProgramStepV126], str]]:
    if str(concept.concept_kind) != "BboxSource":
        return None
    params = {k: v for k, v in concept.params}
    kind = str(params.get("kind") or "")
    if kind == "bbox_nonzero":
        bg = int(params.get("bg", 0))
        return ([ProgramStepV126(op_id="bbox_nonzero", in_vars=("g0",), out_var="bb1", args={"bg": int(bg)})], "bb1")
    if kind == "bbox_by_color":
        color = int(params.get("color", 0))
        return (
            [ProgramStepV126(op_id="bbox_by_color", in_vars=("g0",), out_var="bb1", args={"color": int(color)})],
            "bb1",
        )
    if kind == "bbox_of_selected_object":
        selector = str(params.get("selector") or "")
        color = int(params.get("color", 0))
        steps = [
            ProgramStepV126(op_id="connected_components", in_vars=("g0",), out_var="os1", args={"color": int(color)}),
            ProgramStepV126(op_id="select_object", in_vars=("os1",), out_var="o2", args={"selector": str(selector)}),
            ProgramStepV126(op_id="bbox_of_object", in_vars=("o2",), out_var="bb3", args={}),
        ]
        return (steps, "bb3")
    return None


def _enumerate_plan_candidates_v126(
    *, concepts: Sequence[ConceptV126], train_pairs: Sequence[Tuple[GridV124, GridV124]]
) -> List[PlanCandidateV126]:
    # Gather palettes
    colors_in: List[int] = []
    colors_out: List[int] = []
    inferred_maps: List[Dict[str, int]] = []
    for inp, out in train_pairs:
        colors_in.extend(unique_colors_v124(inp))
        colors_out.extend(unique_colors_v124(out))
        m = _infer_color_mapping_v126(inp, out)
        if m is not None:
            inferred_maps.append(m)
    colors_in_s = sorted(set(int(c) for c in colors_in))
    colors_out_s = sorted(set(int(c) for c in colors_out))
    inferred_maps = sorted(inferred_maps, key=lambda m: canonical_json_dumps(m))

    # Bbox sources proposed by concepts.
    bbox_concepts = [c for c in concepts if str(c.concept_kind) == "BboxSource"]
    bbox_concepts.sort(key=lambda c: str(c.concept_sig()))

    plans: List[PlanCandidateV126] = []

    # --- PlanKind: global transforms ---
    for op in ["identity", "rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v", "crop_bbox_nonzero"]:
        p = ProgramV126(steps=(ProgramStepV126(op_id=str(op), in_vars=("g0",), out_var="g1", args={}),))
        plans.append(PlanCandidateV126(plan_kind="global_transform", params=(("op", str(op)),), concepts_used=tuple(), program=p))

    # bounded translations
    for dy in [-2, -1, 0, 1, 2]:
        for dx in [-2, -1, 0, 1, 2]:
            if dx == 0 and dy == 0:
                continue
            p = ProgramV126(
                steps=(
                    ProgramStepV126(op_id="translate", in_vars=("g0",), out_var="g1", args={"dx": int(dx), "dy": int(dy), "pad": 0}),
                )
            )
            plans.append(
                PlanCandidateV126(plan_kind="translate", params=(("dx", int(dx)), ("dy", int(dy))), concepts_used=tuple(), program=p)
            )

            # overlay shift (duplicate)
            p2 = ProgramV126(
                steps=(
                    ProgramStepV126(op_id="translate", in_vars=("g0",), out_var="g1", args={"dx": int(dx), "dy": int(dy), "pad": 0}),
                    ProgramStepV126(op_id="overlay", in_vars=("g0", "g1"), out_var="g2", args={"transparent": 0}),
                )
            )
            plans.append(
                PlanCandidateV126(
                    plan_kind="overlay_shift",
                    params=(("dx", int(dx)), ("dy", int(dy)), ("transparent", 0)),
                    concepts_used=tuple(),
                    program=p2,
                )
            )

    # --- PlanKind: color replace/map ---
    for fc in colors_in_s:
        for tc in colors_out_s:
            if fc == tc:
                continue
            p = ProgramV126(
                steps=(ProgramStepV126(op_id="replace_color", in_vars=("g0",), out_var="g1", args={"from_color": int(fc), "to_color": int(tc)}),)
            )
            plans.append(
                PlanCandidateV126(
                    plan_kind="replace_color",
                    params=(("from_color", int(fc)), ("to_color", int(tc))),
                    concepts_used=tuple(),
                    program=p,
                )
            )

    for m in inferred_maps:
        p = ProgramV126(steps=(ProgramStepV126(op_id="map_colors", in_vars=("g0",), out_var="g1", args={"mapping": dict(m)}),))
        plans.append(PlanCandidateV126(plan_kind="map_colors", params=(("mapping", dict(m)),), concepts_used=tuple(), program=p))

    # --- PlanKind: bbox edits ---
    for bbox_concept in bbox_concepts:
        steps_bbox, bb_var = _bbox_source_steps_from_concept_v126(bbox_concept) or (None, "")
        if steps_bbox is None:
            continue
        bb_sig = str(bbox_concept.concept_sig())

        for c in colors_out_s:
            # draw border
            steps = list(steps_bbox) + [ProgramStepV126(op_id="draw_rect_border", in_vars=("g0", bb_var), out_var="g2", args={"color": int(c), "thickness": 1})]
            plans.append(
                PlanCandidateV126(
                    plan_kind="bbox_border",
                    params=(("color", int(c)),),
                    concepts_used=(bb_sig,),
                    program=ProgramV126(steps=tuple(steps)),
                )
            )
            # fill rect
            steps = list(steps_bbox) + [ProgramStepV126(op_id="fill_rect", in_vars=("g0", bb_var), out_var="g2", args={"color": int(c)})]
            plans.append(
                PlanCandidateV126(
                    plan_kind="bbox_fill",
                    params=(("color", int(c)),),
                    concepts_used=(bb_sig,),
                    program=ProgramV126(steps=tuple(steps)),
                )
            )

        # expand bbox + border
        for d in [1, 2]:
            for c in colors_out_s:
                steps = list(steps_bbox) + [
                    ProgramStepV126(op_id="bbox_expand", in_vars=("g0", bb_var), out_var="bbx2", args={"delta": int(d)}),
                    ProgramStepV126(op_id="draw_rect_border", in_vars=("g0", "bbx2"), out_var="g3", args={"color": int(c), "thickness": 1}),
                ]
                plans.append(
                    PlanCandidateV126(
                        plan_kind="bbox_expand_border",
                        params=(("delta", int(d)), ("color", int(c))),
                        concepts_used=(bb_sig,),
                        program=ProgramV126(steps=tuple(steps)),
                    )
                )

        # fill then border with potentially different colors
        for c_fill in colors_out_s:
            for c_border in colors_out_s:
                if c_fill == c_border:
                    continue
                steps = list(steps_bbox) + [
                    ProgramStepV126(op_id="fill_rect", in_vars=("g0", bb_var), out_var="g2", args={"color": int(c_fill)}),
                    ProgramStepV126(op_id="draw_rect_border", in_vars=("g2", bb_var), out_var="g3", args={"color": int(c_border), "thickness": 1}),
                ]
                plans.append(
                    PlanCandidateV126(
                        plan_kind="bbox_fill_then_border",
                        params=(("fill_color", int(c_fill)), ("border_color", int(c_border))),
                        concepts_used=(bb_sig,),
                        program=ProgramV126(steps=tuple(steps)),
                    )
                )

        # crop + paste to origin (duplicate)
        for c_sel in colors_in_s:
            if c_sel == 0:
                continue
            # Ensure bbox_by_color uses c_sel (avoid huge connected-components fanout here)
            if not (str({k: v for k, v in bbox_concept.params}.get("kind") or "") == "bbox_by_color" and int({k: v for k, v in bbox_concept.params}.get("color", -1)) == int(c_sel)):
                continue
            steps = [
                ProgramStepV126(op_id="bbox_by_color", in_vars=("g0",), out_var="bb1", args={"color": int(c_sel)}),
                ProgramStepV126(op_id="crop_bbox", in_vars=("g0", "bb1"), out_var="g2", args={}),
                ProgramStepV126(op_id="paste", in_vars=("g0", "g2"), out_var="g3", args={"top": 0, "left": 0, "transparent": 0}),
            ]
            plans.append(
                PlanCandidateV126(
                    plan_kind="paste_duplicate_origin",
                    params=(("color", int(c_sel)), ("top", 0), ("left", 0)),
                    concepts_used=(bb_sig,),
                    program=ProgramV126(steps=tuple(steps)),
                )
            )

    # --- PlanKind: flood fill from seed mask ---
    # seed colors from input, fill colors from output.
    for seed_c in colors_in_s:
        if int(seed_c) == 0:
            continue
        for fill_c in colors_out_s:
            if int(fill_c) == int(seed_c):
                continue
            steps = [
                ProgramStepV126(op_id="mask_by_color", in_vars=("g0",), out_var="m1", args={"color": int(seed_c)}),
                ProgramStepV126(op_id="flood_fill", in_vars=("g0", "m1"), out_var="g2", args={"target_color": 0, "fill_color": int(fill_c)}),
            ]
            plans.append(
                PlanCandidateV126(
                    plan_kind="flood_fill_from_seed",
                    params=(("seed_color", int(seed_c)), ("target_color", 0), ("fill_color", int(fill_c))),
                    concepts_used=tuple(),
                    program=ProgramV126(steps=tuple(steps)),
                )
            )

    # Deterministic ordering: MDL proxy then plan_sig.
    plans.sort(key=lambda p: (int(_program_cost_bits_v126(p.program.steps)), int(len(p.program.steps)), str(p.plan_sig())))
    return plans


def solve_arc_task_v126(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    max_depth: int = 5,
    max_programs: int = 12000,
    trace_limit: int = 120,
) -> Dict[str, Any]:
    """
    Two-level solver:
      - plan enumeration (concept-guided, typed)
      - compilation to a DSL program and validation on train pairs
    Deterministic; FAIL-CLOSED on ambiguity.
    """

    concepts = propose_concepts_v126(train_pairs=train_pairs, test_in=test_in)
    concept_trace = concept_trace_v126(concepts)

    # Plan enumeration and evaluation.
    plan_candidates = _enumerate_plan_candidates_v126(concepts=concepts, train_pairs=train_pairs)

    # Cache apply_program results deterministically (in-memory; per-run only).
    apply_cache: Dict[Tuple[str, str], GridV124] = {}

    def _eval_program(program: ProgramV126, inp: GridV124) -> GridV124:
        ps = str(program.program_sig())
        gh = str(grid_hash_v124(inp))
        key = (ps, gh)
        if key in apply_cache:
            return apply_cache[key]
        got = apply_program_v126(program, inp)
        apply_cache[key] = got
        return got

    def _ok_on_train(program: ProgramV126) -> Tuple[bool, Optional[Dict[str, Any]]]:
        if program.steps:
            last_var = program.steps[-1].out_var
            if _type_of_var_name_v126(str(last_var)) != "GRID":
                return False, {"kind": "incomplete_program_no_grid_output", "last_var": str(last_var)}
        for inp, out in train_pairs:
            try:
                got = _eval_program(program, inp)
            except Exception as e:
                return False, {"kind": "apply_error", "error": str(e)}
            if not grid_equal_v124(got, out):
                return False, _summarize_mismatch_v126(got=got, want=out)
        return True, None

    tried = 0
    trace_plans: List[Dict[str, Any]] = []
    best_cost: Optional[int] = None
    best_programs: List[ProgramV126] = []
    best_sigs: List[str] = []
    best_plan_sigs: List[str] = []

    for cand in plan_candidates:
        cost_bits = int(_program_cost_bits_v126(cand.program.steps))
        if best_cost is not None and cost_bits > int(best_cost):
            break
        ok, mismatch = _ok_on_train(cand.program)
        tried += 1
        if len(trace_plans) < int(trace_limit):
            trace_plans.append(
                {
                    "plan_sig": str(cand.plan_sig()),
                    "plan_kind": str(cand.plan_kind),
                    "program_sig": str(cand.program.program_sig()),
                    "cost_bits": int(cost_bits),
                    "concepts_used": list(cand.concepts_used),
                    "ok_train": bool(ok),
                    "mismatch": dict(mismatch) if isinstance(mismatch, dict) else None,
                }
            )
        if ok:
            if best_cost is None or int(cost_bits) < int(best_cost):
                best_cost = int(cost_bits)
                best_programs = [cand.program]
                best_sigs = [str(cand.program.program_sig())]
                best_plan_sigs = [str(cand.plan_sig())]
            elif int(cost_bits) == int(best_cost):
                best_programs.append(cand.program)
                best_sigs.append(str(cand.program.program_sig()))
                best_plan_sigs.append(str(cand.plan_sig()))

    # Fallback Stage: typed best-first search (limited) if plan enumeration didn't find anything.
    budget_exhausted = False
    search_trace: List[Dict[str, Any]] = []

    if best_cost is None:
        root_sig = sha256_hex(b"arc_program_v126_root")
        frontier: List[Tuple[int, int, str, int, Tuple[ProgramStepV126, ...], Tuple[Tuple[str, str], ...]]] = []
        push_idx = 0
        heapq.heappush(frontier, (0, 0, str(root_sig), push_idx, tuple(), (("g0", "GRID"),)))
        seen: set[str] = set()

        # Precompute op variants deterministically from palettes (bounded).
        colors_in: List[int] = []
        colors_out: List[int] = []
        for inp, out in train_pairs:
            colors_in.extend(unique_colors_v124(inp))
            colors_out.extend(unique_colors_v124(out))
        colors_in_s = sorted(set(int(c) for c in colors_in))
        colors_out_s = sorted(set(int(c) for c in colors_out))

        op_variants: List[Tuple[str, Dict[str, Any]]] = []
        for op in ["identity", "rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v", "crop_bbox_nonzero"]:
            op_variants.append((op, {}))
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                op_variants.append(("translate", {"dx": int(dx), "dy": int(dy), "pad": 0}))
        for fc in colors_in_s:
            for tc in colors_out_s:
                if fc == tc:
                    continue
                op_variants.append(("replace_color", {"from_color": int(fc), "to_color": int(tc)}))
        # DSL ops
        bg = 0
        for c in colors_in_s:
            if int(c) == int(bg):
                continue
            op_variants.append(("bbox_by_color", {"color": int(c)}))
            op_variants.append(("mask_by_color", {"color": int(c)}))
            op_variants.append(("connected_components", {"color": int(c)}))
        op_variants.append(("bbox_nonzero", {"bg": 0}))
        for c in colors_out_s:
            op_variants.append(("fill_rect", {"color": int(c)}))
            op_variants.append(("draw_rect_border", {"color": int(c), "thickness": 1}))
            op_variants.append(("paint_mask", {"color": int(c)}))
            op_variants.append(("flood_fill", {"target_color": 0, "fill_color": int(c)}))
        for d in [1, 2]:
            op_variants.append(("bbox_expand", {"delta": int(d)}))
        op_variants.append(("crop_bbox", {}))
        op_variants.append(("overlay", {"transparent": 0}))
        op_variants.append(("paste", {"top": 0, "left": 0, "transparent": 0}))
        for sel in ["largest_area", "smallest_area", "leftmost", "rightmost", "topmost", "bottommost"]:
            op_variants.append(("select_object", {"selector": str(sel)}))
        op_variants.append(("bbox_of_object", {}))
        op_variants.sort(key=lambda oa: (str(oa[0]), canonical_json_dumps(oa[1])))

        def _step_fp(step: ProgramStepV126) -> str:
            return canonical_json_dumps(step.to_dict())

        def _child_sig(parent_sig: str, step_fp: str) -> str:
            return sha256_hex((str(parent_sig) + "|" + str(step_fp)).encode("utf-8"))

        while frontier:
            cost_bits, plen, sig, _, steps, var_types = heapq.heappop(frontier)
            if sig in seen:
                continue
            seen.add(sig)
            program = ProgramV126(steps=tuple(steps))
            ok_train, mismatch = _ok_on_train(program)
            tried += 1
            if len(search_trace) < int(trace_limit):
                search_trace.append(
                    {
                        "program_sig": str(program.program_sig()),
                        "cost_bits": int(cost_bits),
                        "steps": [s.to_dict() for s in program.steps],
                        "ok_train": bool(ok_train),
                        "mismatch": dict(mismatch) if isinstance(mismatch, dict) else None,
                    }
                )
            if ok_train:
                best_cost = int(cost_bits)
                best_programs = [program]
                best_sigs = [str(program.program_sig())]
                best_plan_sigs = []
                break
            if tried >= int(max_programs):
                budget_exhausted = True
                break
            if int(plen) >= int(max_depth):
                continue

            grid_vars = [vn for vn, t in var_types if t == "GRID"]
            cur_grid_var = grid_vars[-1] if grid_vars else "g0"

            for op_id, args in op_variants:
                op = str(op_id)
                in_vars: List[str] = []
                out_prefix = ""
                out_type = ""

                if op in {"identity", "rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v", "translate", "crop_bbox_nonzero", "replace_color", "map_colors"}:
                    in_vars = [str(cur_grid_var)]
                    out_prefix, out_type = "g", "GRID"
                elif op in {"bbox_nonzero", "bbox_by_color"}:
                    in_vars = [str(cur_grid_var)]
                    out_prefix, out_type = "bb", "BBOX"
                elif op in {"bbox_expand"}:
                    bb_vars = [vn for vn, t in var_types if t == "BBOX"]
                    if not bb_vars:
                        continue
                    in_vars = [str(cur_grid_var), str(bb_vars[-1])]
                    out_prefix, out_type = "bb", "BBOX"
                elif op in {"mask_by_color"}:
                    in_vars = [str(cur_grid_var)]
                    out_prefix, out_type = "m", "MASK"
                elif op in {"connected_components"}:
                    in_vars = [str(cur_grid_var)]
                    out_prefix, out_type = "os", "OBJECT_SET"
                elif op in {"select_object"}:
                    os_vars = [vn for vn, t in var_types if t == "OBJECT_SET"]
                    if not os_vars:
                        continue
                    in_vars = [str(os_vars[-1])]
                    out_prefix, out_type = "o", "OBJECT"
                elif op in {"bbox_of_object"}:
                    o_vars = [vn for vn, t in var_types if t == "OBJECT"]
                    if not o_vars:
                        continue
                    in_vars = [str(o_vars[-1])]
                    out_prefix, out_type = "bb", "BBOX"
                elif op in {"crop_bbox", "fill_rect", "draw_rect_border"}:
                    bb_vars = [vn for vn, t in var_types if t == "BBOX"]
                    if not bb_vars:
                        continue
                    in_vars = [str(cur_grid_var), str(bb_vars[-1])]
                    out_prefix, out_type = "g", "GRID"
                elif op in {"paint_mask", "flood_fill"}:
                    m_vars = [vn for vn, t in var_types if t == "MASK"]
                    if not m_vars:
                        continue
                    in_vars = [str(cur_grid_var), str(m_vars[-1])]
                    out_prefix, out_type = "g", "GRID"
                elif op in {"overlay", "paste"}:
                    g_vars = [vn for vn, t in var_types if t == "GRID"]
                    if len(g_vars) < 2:
                        continue
                    in_vars = [str(g_vars[-2]), str(g_vars[-1])]
                    out_prefix, out_type = "g", "GRID"
                else:
                    continue

                out_var = f"{out_prefix}{len(steps) + 1}"
                step = ProgramStepV126(op_id=str(op), in_vars=tuple(in_vars), out_var=str(out_var), args=dict(args))
                new_cost = int(cost_bits) + int(_step_cost_bits_v126(step))
                new_len = int(plen) + 1
                new_sig = _child_sig(str(sig), _step_fp(step))
                push_idx += 1
                new_steps = tuple(list(steps) + [step])
                new_var_types = tuple(list(var_types) + [(str(out_var), str(out_type))])
                heapq.heappush(frontier, (int(new_cost), int(new_len), str(new_sig), int(push_idx), new_steps, new_var_types))

    status = "FAIL"
    failure_reason: Dict[str, Any] = {}
    predicted: Optional[GridV124] = None
    best_program_json: Optional[Dict[str, Any]] = None

    if best_cost is None:
        if budget_exhausted:
            failure_reason = {"kind": "SEARCH_BUDGET_EXCEEDED", "max_programs": int(max_programs), "max_depth": int(max_depth)}
        else:
            failure_reason = {"kind": "NO_CONSISTENT_PROGRAM", "max_depth": int(max_depth)}
        status = "FAIL"
    else:
        uniq_best_sigs = sorted(set(str(s) for s in best_sigs))
        if len(uniq_best_sigs) > 1:
            # Disambiguate using *test_in only* (never test output): if all minimum-MDL programs
            # produce the same predicted grid, it's safe to emit that prediction deterministically.
            pred_hash_by_sig: Dict[str, str] = {}
            pred_grid_by_hash: Dict[str, GridV124] = {}
            for program in sorted(best_programs, key=lambda p: str(p.program_sig())):
                sig = str(program.program_sig())
                pred = _eval_program(program, test_in)
                ph = str(grid_hash_v124(pred))
                pred_hash_by_sig[sig] = ph
                if ph not in pred_grid_by_hash:
                    pred_grid_by_hash[ph] = pred
            uniq_pred_hashes = sorted(set(pred_hash_by_sig.values()))
            if len(uniq_pred_hashes) == 1:
                status = "SOLVED"
                chosen_sig = sorted(pred_hash_by_sig.keys())[0]
                chosen_program = next(p for p in best_programs if str(p.program_sig()) == chosen_sig)
                best_program_json = chosen_program.to_dict()
                predicted = pred_grid_by_hash[uniq_pred_hashes[0]]
                failure_reason = {}
            else:
                status = "UNKNOWN"
                failure_reason = {
                    "kind": "AMBIGUOUS_RULE",
                    "solutions": list(uniq_best_sigs),
                    "predicted_grid_hash_by_solution": {k: pred_hash_by_sig[k] for k in sorted(pred_hash_by_sig.keys())},
                }
        else:
            status = "SOLVED"
            program = best_programs[0]
            best_program_json = program.to_dict()
            predicted = _eval_program(program, test_in)

    plan_trace = {
        "schema_version": 126,
        "kind": "arc_plan_trace_v126",
        "plans_tried": int(tried),
        "trace_plans": list(trace_plans),
        "fallback_search_trace": list(search_trace) if search_trace else None,
        "best_plan_sigs": list(sorted(set(best_plan_sigs))) if best_plan_sigs else None,
    }

    out: Dict[str, Any] = {
        "schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V126),
        "status": str(status),
        "program": dict(best_program_json) if best_program_json is not None and status == "SOLVED" else None,
        "program_cost_bits": int(best_cost) if best_cost is not None and status == "SOLVED" else None,
        "predicted_grid": [list(r) for r in predicted] if predicted is not None else None,
        "predicted_grid_hash": grid_hash_v124(predicted) if predicted is not None else "",
        "failure_reason": dict(failure_reason) if failure_reason else None,
        "concept_trace": dict(concept_trace),
        "plan_trace": dict(plan_trace),
    }
    out["result_sig"] = sha256_hex(canonical_json_dumps(out).encode("utf-8"))
    return out


def diagnose_missing_operator_v126(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> Dict[str, Any]:
    """
    Deterministic coarse diagnostic (not a solver): derive operator-gap tags.
    Intended for an operator backlog; must not depend on task_id.
    """
    gaps: List[str] = []
    shape_change = False
    out_new_colors = set()
    for inp, out in train_pairs:
        if grid_shape_v124(inp) != grid_shape_v124(out):
            shape_change = True
        out_colors = set(unique_colors_v124(out))
        out_new_colors |= (out_colors - set(unique_colors_v124(inp)))
    if shape_change:
        gaps.append("shape_change_present")
    if out_new_colors:
        gaps.append("introduces_new_colors")
    gaps = sorted(set(str(g) for g in gaps))
    return {
        "schema_version": 126,
        "kind": "arc_operator_gap_diag_v126",
        "gaps": gaps,
        "out_new_colors": [int(c) for c in sorted(out_new_colors)],
    }
