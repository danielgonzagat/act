from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_delta_v128 import DeltaEvidenceV128, compute_delta_v128
from .arc_dsl_v128 import OP_DEFS_V128, apply_op_v128
from .arc_inverse_v128 import (
    BboxExprV128,
    InverseCandidateV128,
    MaskExprV128,
    inverse_propose_draw_rect_border_v128,
    inverse_propose_fill_rect_v128,
    inverse_propose_map_colors_v128,
    inverse_propose_overlay_v128,
    inverse_propose_paint_mask_v128,
    inverse_propose_paste_v128,
    inverse_propose_translate_v128,
    propose_bbox_exprs_v128,
    propose_mask_exprs_v128,
)
from .arc_selector_v128 import SelectorHypothesisV128, infer_selector_hypotheses_v128
from .grid_v124 import GridV124, grid_equal_v124, grid_hash_v124, grid_shape_v124, unique_colors_v124

ARC_SOLVER_SCHEMA_VERSION_V128 = 128


def _mode_color_v128(g: GridV124) -> int:
    counts: Dict[int, int] = {}
    for row in g:
        for x in row:
            counts[int(x)] = int(counts.get(int(x), 0)) + 1
    if not counts:
        return 0
    items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
    return int(items[0][0])


def _canonical_step_v128(step: Dict[str, Any]) -> Dict[str, Any]:
    op_id = str(step.get("op_id") or "")
    in_vars = [str(v) for v in (step.get("in_vars") or [])]
    out_var = str(step.get("out_var") or "")
    args_raw = step.get("args") if isinstance(step.get("args"), dict) else {}
    args: Dict[str, Any] = {}
    for k in sorted(args_raw.keys()):
        args[str(k)] = args_raw[k]
    return {"op_id": op_id, "in_vars": list(in_vars), "out_var": out_var, "args": args}


def program_sig_v128(steps: Sequence[Dict[str, Any]]) -> str:
    body = {"schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V128), "kind": "arc_program_v128", "steps": [_canonical_step_v128(s) for s in steps]}
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _step_cost_bits_v128(step: Dict[str, Any]) -> int:
    op = str(step.get("op_id") or "")
    if op in OP_DEFS_V128:
        base = int(OP_DEFS_V128[op].cost_bits)
    elif op == "map_colors":
        base = 16
    elif op == "translate":
        base = 12
    else:
        base = 24
    extra = 0
    args = step.get("args") if isinstance(step.get("args"), dict) else {}
    for k, v in args.items():
        if k == "mapping" and isinstance(v, dict):
            extra += 8 * int(len(v))
        else:
            extra += 4
    in_vars = step.get("in_vars") if isinstance(step.get("in_vars"), list) else []
    extra += 2 * max(0, int(len(in_vars)) - 1)
    return int(base + extra)


def _program_cost_bits_v128(steps: Sequence[Dict[str, Any]]) -> int:
    return int(sum(_step_cost_bits_v128(_canonical_step_v128(s)) for s in steps))


def _apply_program_v128(*, steps: Sequence[Dict[str, Any]], inp: GridV124, out_var: str) -> GridV124:
    from .grid_v124 import translate_v124

    env: Dict[str, Any] = {"g0": inp}
    for raw in steps:
        st = _canonical_step_v128(raw)
        op = str(st["op_id"])
        ins = [env[str(v)] for v in st["in_vars"]]
        args = dict(st["args"])
        if op in OP_DEFS_V128:
            env[str(st["out_var"])] = apply_op_v128(op_id=op, inputs=ins, args=args)
            continue
        if op == "map_colors":
            g = ins[0]
            m_raw = args.get("mapping", {})
            if not isinstance(m_raw, dict):
                raise ValueError("mapping_not_dict")
            m: Dict[int, int] = {int(k): int(v) for k, v in m_raw.items()}
            env[str(st["out_var"])] = tuple(tuple(int(m.get(int(x), int(x))) for x in row) for row in g)
            continue
        if op == "translate":
            g = ins[0]
            env[str(st["out_var"])] = translate_v124(g, dx=int(args["dx"]), dy=int(args["dy"]), pad=int(args.get("pad", 0)))
            continue
        raise ValueError(f"unknown_op:{op}")
    out = env.get(str(out_var))
    if not (isinstance(out, tuple) and (not out or isinstance(out[0], tuple))):
        raise ValueError("program_output_not_grid")
    return out  # type: ignore[return-value]


def _mismatch_cells_v128(a: GridV124, b: GridV124) -> int:
    ha, wa = grid_shape_v124(a)
    hb, wb = grid_shape_v124(b)
    if (ha, wa) != (hb, wb):
        return int(ha * wa + hb * wb)
    mism = 0
    for r in range(ha):
        for c in range(wa):
            if int(a[r][c]) != int(b[r][c]):
                mism += 1
    return int(mism)


def _palette_gap_v128(g: GridV124, target: GridV124) -> int:
    have = set(int(c) for c in unique_colors_v124(g))
    want = set(int(c) for c in unique_colors_v124(target))
    return int(len(want - have))


def _objective_key_v128(*, residual_cells: int, residual_palette_gap: int, cost_bits: int, tie_sig: str) -> Tuple[int, int, int, str]:
    return int(residual_cells), int(residual_palette_gap), int(cost_bits), str(tie_sig)


def _rebase_steps_block_v128(
    *,
    block_steps: Sequence[Dict[str, Any]],
    current_var: str,
    suffix: str,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Rebase a candidate block that uses placeholder var `gC` (current grid) into a program env:
      - gC -> current_var
      - all other vars (except g0 and current_var) get a deterministic suffix.
    Returns (rebased_steps, rebased_last_out_var).
    """
    mapping: Dict[str, str] = {"gC": str(current_var), "g0": "g0"}

    def map_var(v: str) -> str:
        vv = str(v)
        if vv in mapping:
            return mapping[vv]
        mapping[vv] = str(vv) + str(suffix)
        return mapping[vv]

    rebased: List[Dict[str, Any]] = []
    for raw in block_steps:
        st = _canonical_step_v128(raw)
        in_vars = [map_var(v) for v in st["in_vars"]]
        out_var = map_var(st["out_var"])
        rebased.append({"op_id": str(st["op_id"]), "in_vars": in_vars, "out_var": out_var, "args": dict(st["args"])})
    last_out = rebased[-1]["out_var"] if rebased else str(current_var)
    return rebased, str(last_out)


@dataclass(frozen=True)
class _StateV128:
    depth: int
    steps: Tuple[Dict[str, Any], ...]
    current_var: str
    cost_bits: int
    train_outs: Tuple[GridV124, ...]
    test_out: GridV124
    residual_cells_per_pair: Tuple[int, ...]
    residual_palette_gap_per_pair: Tuple[int, ...]

    def residual_cells_total(self) -> int:
        return int(sum(int(x) for x in self.residual_cells_per_pair))

    def residual_palette_gap_total(self) -> int:
        return int(sum(int(x) for x in self.residual_palette_gap_per_pair))

    def program_sig(self) -> str:
        return program_sig_v128(self.steps)


def _pick_focus_pair_v128(residual_cells_per_pair: Sequence[int]) -> int:
    # Deterministic: pick max residual; tie by lower index.
    best_i = 0
    best_v = -1
    for i, v in enumerate(residual_cells_per_pair):
        vv = int(v)
        if vv > best_v:
            best_v = vv
            best_i = int(i)
    return int(best_i)


def _bg_candidates_v128(*, grids: Sequence[GridV124]) -> List[int]:
    bgs: List[int] = [0]
    for g in grids:
        bgs.append(_mode_color_v128(g))
    return sorted(set(int(x) for x in bgs))


def _generate_candidates_for_state_v128(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    train_curr: Sequence[GridV124],
    test_curr: GridV124,
    residual_cells_per_pair: Sequence[int],
    beam_width: int,
) -> Tuple[List[InverseCandidateV128], Dict[str, Any]]:
    """
    Generate candidate step-blocks (as InverseCandidateV128 with placeholder gC) from the current residual state.
    Candidate set is built from a deterministic focus pair, then filtered by applicability across all train inputs.
    """
    if not train_pairs:
        return [], {"kind": "empty_train_pairs"}

    def _apply_block_placeholder(steps: Sequence[Dict[str, Any]], g_cur: GridV124) -> None:
        from .grid_v124 import translate_v124

        env: Dict[str, Any] = {"gC": g_cur}
        for raw in steps:
            st = _canonical_step_v128(raw)
            op = str(st["op_id"])
            ins = [env[str(v)] for v in st["in_vars"]]
            args = dict(st["args"])
            out_var = str(st["out_var"])
            if op in OP_DEFS_V128:
                env[out_var] = apply_op_v128(op_id=op, inputs=ins, args=args)
                continue
            if op == "map_colors":
                g = ins[0]
                m_raw = args.get("mapping", {})
                if not isinstance(m_raw, dict):
                    raise ValueError("mapping_not_dict")
                m: Dict[int, int] = {int(k): int(v) for k, v in m_raw.items()}
                env[out_var] = tuple(tuple(int(m.get(int(x), int(x))) for x in row) for row in g)
                continue
            if op == "translate":
                g = ins[0]
                env[out_var] = translate_v124(g, dx=int(args["dx"]), dy=int(args["dy"]), pad=int(args.get("pad", 0)))
                continue
            raise ValueError(f"unknown_op:{op}")

    focus_i = _pick_focus_pair_v128(residual_cells_per_pair)
    inp0 = train_curr[int(focus_i)]
    out0 = train_pairs[int(focus_i)][1]
    delta0 = compute_delta_v128(inp0, out0)

    # Compute selector hypotheses and derived expressions on the current residual world.
    curr_pairs = list(zip(list(train_curr), [o for _i, (_in, o) in enumerate(train_pairs)]))
    selector_hyps = infer_selector_hypotheses_v128(train_pairs=curr_pairs)

    bg_candidates = _bg_candidates_v128(grids=[inp0, out0, test_curr])
    color_candidates: List[int] = []
    for g in list(train_curr) + [out0, test_curr]:
        color_candidates.extend(unique_colors_v124(g))
    color_candidates = sorted(set(int(c) for c in color_candidates))

    bbox_exprs = propose_bbox_exprs_v128(
        bg_candidates=bg_candidates,
        color_candidates=color_candidates,
        selector_hypotheses=selector_hyps,
        max_expand_delta=2,
    )
    mask_exprs = propose_mask_exprs_v128(
        color_candidates=color_candidates,
        bbox_exprs=bbox_exprs[:18],
        selector_hypotheses=selector_hyps,
    )

    cands: List[InverseCandidateV128] = []

    # 1) mapping candidates consistent across all pairs (at this state)
    mapping_sig = ""
    mapping_cand: Optional[InverseCandidateV128] = None
    ok_map = True
    for i, (cur, (_inp, out)) in enumerate(zip(train_curr, train_pairs)):
        c0 = inverse_propose_map_colors_v128(inp=cur, out=out)
        if not c0:
            ok_map = False
            break
        m_raw = c0[0].steps[0].get("args", {}).get("mapping", {})
        sig = canonical_json_dumps(m_raw)
        if mapping_sig and sig != mapping_sig:
            ok_map = False
            break
        mapping_sig = sig
        mapping_cand = c0[0]
    if ok_map and mapping_cand is not None:
        cands.append(mapping_cand)

    # 2) translate candidates with common (dx,dy,pad) across all pairs
    common_keys: Optional[set[Tuple[int, int, int]]] = None
    examples: Dict[Tuple[int, int, int], InverseCandidateV128] = {}
    for cur, (_inp, out) in zip(train_curr, train_pairs):
        cc = inverse_propose_translate_v128(inp=cur, out=out, bg_candidates=bg_candidates)
        keys: set[Tuple[int, int, int]] = set()
        for c in cc:
            args = c.steps[0].get("args", {})
            key = (int(args.get("dx", 0)), int(args.get("dy", 0)), int(args.get("pad", 0)))
            keys.add(key)
            examples[key] = c
        if common_keys is None:
            common_keys = set(keys)
        else:
            common_keys &= set(keys)
    if common_keys:
        for k in sorted(common_keys):
            cands.append(examples[k])

    # 3) bbox-based atoms
    cands.extend(inverse_propose_fill_rect_v128(inp=inp0, out=out0, delta=delta0, bbox_exprs=bbox_exprs))
    cands.extend(inverse_propose_draw_rect_border_v128(inp=inp0, out=out0, delta=delta0, bbox_exprs=bbox_exprs))

    # 4) paint_mask (partial candidates ok)
    cands.extend(
        inverse_propose_paint_mask_v128(
            inp=inp0,
            out=out0,
            delta=delta0,
            mask_exprs=mask_exprs,
            bg_candidates=bg_candidates,
            max_candidates=64,
        )
    )

    # 5) paste and overlay (partial ok)
    cands.extend(
        inverse_propose_paste_v128(
            inp=inp0,
            out=out0,
            delta=delta0,
            bbox_exprs=bbox_exprs,
            bg_candidates=bg_candidates,
            max_candidates=48,
        )
    )
    cands.extend(inverse_propose_overlay_v128(inp=inp0, out=out0, delta=delta0, bg_candidates=bg_candidates, max_candidates=24))

    # Dedup and sort by (cost_bits, candidate_sig)
    seen: set[str] = set()
    uniq: List[InverseCandidateV128] = []
    for c in cands:
        sig = str(c.candidate_sig())
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(c)
    uniq.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))

    # Filter by applicability across all train inputs (must execute without exception).
    applicable: List[InverseCandidateV128] = []
    for c in uniq:
        ok = True
        for cur in train_curr:
            try:
                _apply_block_placeholder(list(c.steps), cur)
            except Exception:
                ok = False
                break
        if ok:
            applicable.append(c)
        if len(applicable) >= int(beam_width):
            break

    trace = {
        "schema_version": 128,
        "kind": "arc_candidate_generation_trace_v128",
        "focus_pair_index": int(focus_i),
        "delta0": delta0.to_dict(),
        "bg_candidates": [int(x) for x in bg_candidates],
        "selector_hypotheses": [{"sig": h.hypothesis_sig(), **h.to_dict()} for h in selector_hyps],
        "bbox_exprs_top": [e.to_dict() for e in bbox_exprs[:12]],
        "mask_exprs_top": [e.to_dict() for e in mask_exprs[:12]],
        "candidates_top": [c.to_dict() for c in applicable[:20]],
    }
    return applicable, trace


def solve_arc_task_v128(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> Dict[str, Any]:
    """
    V128 solver: delta decomposition + inverse proposers (partial ok) + selector induction + bounded search (depth<=3).
    No family templates; no task_id usage; train_pairs + test_in only.
    """
    plan_trace: Dict[str, Any] = {"schema_version": 128, "kind": "arc_plan_trace_v128"}
    if not train_pairs:
        return {"schema_version": 128, "kind": "arc_solve_result_v128", "status": "FAIL", "failure_reason": {"kind": "EMPTY_TRAIN"}, "plan_trace": plan_trace}

    # Search hyperparameters (deterministic, small).
    max_depth = 3
    beam_width = 48
    max_states_per_depth = 64

    train_inputs = [inp for inp, _out in train_pairs]
    train_targets = [out for _inp, out in train_pairs]

    # Initial state: identity.
    init_steps: Tuple[Dict[str, Any], ...] = tuple()
    init_current_var = "g0"
    init_cost = 0
    init_train_outs = tuple(train_inputs)
    init_test_out = test_in
    init_res_cells = tuple(_mismatch_cells_v128(inp, out) for inp, out in zip(init_train_outs, train_targets))
    init_res_pal = tuple(_palette_gap_v128(inp, out) for inp, out in zip(init_train_outs, train_targets))
    init_state = _StateV128(
        depth=0,
        steps=init_steps,
        current_var=str(init_current_var),
        cost_bits=int(init_cost),
        train_outs=init_train_outs,
        test_out=init_test_out,
        residual_cells_per_pair=init_res_cells,
        residual_palette_gap_per_pair=init_res_pal,
    )

    # Beam over states by objective.
    beam: List[_StateV128] = [init_state]
    seen_state_sigs: set[str] = set()
    solutions: List[_StateV128] = []

    for depth in range(max_depth + 1):
        # Collect solutions at this depth.
        for st in beam:
            if st.residual_cells_total() == 0 and st.residual_palette_gap_total() == 0:
                solutions.append(st)
        if depth == max_depth:
            break
        next_states: List[_StateV128] = []
        for st_idx, st in enumerate(beam):
            if st.residual_cells_total() == 0 and st.residual_palette_gap_total() == 0:
                continue
            cands, cand_trace = _generate_candidates_for_state_v128(
                train_pairs=train_pairs,
                train_curr=list(st.train_outs),
                test_curr=st.test_out,
                residual_cells_per_pair=list(st.residual_cells_per_pair),
                beam_width=int(beam_width),
            )
            if "candidate_traces" not in plan_trace:
                plan_trace["candidate_traces"] = []
            if len(plan_trace["candidate_traces"]) < 6:
                plan_trace["candidate_traces"].append({"depth": int(depth), "state_index": int(st_idx), "trace": cand_trace})

            for j, cand in enumerate(cands):
                suffix = f"_d{depth}_i{st_idx}_c{j}"
                rebased_steps, new_current_var = _rebase_steps_block_v128(block_steps=cand.steps, current_var=st.current_var, suffix=suffix)
                new_steps = list(st.steps) + rebased_steps
                new_cost = int(_program_cost_bits_v128(new_steps))

                # Evaluate on all train inputs; enforce monotonic residual improvement.
                new_train_outs: List[GridV124] = []
                ok_apply = True
                for inp in train_inputs:
                    try:
                        got = _apply_program_v128(steps=new_steps, inp=inp, out_var=new_current_var)
                    except Exception:
                        ok_apply = False
                        break
                    new_train_outs.append(got)
                if not ok_apply:
                    continue

                new_res_cells = [int(_mismatch_cells_v128(g, t)) for g, t in zip(new_train_outs, train_targets)]
                new_res_pal = [int(_palette_gap_v128(g, t)) for g, t in zip(new_train_outs, train_targets)]
                if sum(new_res_cells) >= st.residual_cells_total():
                    continue
                if any(int(n) > int(o) for n, o in zip(new_res_cells, st.residual_cells_per_pair)):
                    continue

                try:
                    new_test_out = _apply_program_v128(steps=new_steps, inp=test_in, out_var=new_current_var)
                except Exception:
                    continue

                prog_sig = program_sig_v128(new_steps)
                if prog_sig in seen_state_sigs:
                    continue
                seen_state_sigs.add(prog_sig)

                next_states.append(
                    _StateV128(
                        depth=int(depth + 1),
                        steps=tuple(_canonical_step_v128(s) for s in new_steps),
                        current_var=str(new_current_var),
                        cost_bits=int(new_cost),
                        train_outs=tuple(new_train_outs),
                        test_out=new_test_out,
                        residual_cells_per_pair=tuple(int(x) for x in new_res_cells),
                        residual_palette_gap_per_pair=tuple(int(x) for x in new_res_pal),
                    )
                )

        # prune to top states by objective
        next_states.sort(
            key=lambda st: _objective_key_v128(
                residual_cells=st.residual_cells_total(),
                residual_palette_gap=st.residual_palette_gap_total(),
                cost_bits=st.cost_bits,
                tie_sig=st.program_sig(),
            )
        )
        beam = next_states[: int(max_states_per_depth)]
        if not beam:
            break

    if not solutions:
        return {
            "schema_version": 128,
            "kind": "arc_solve_result_v128",
            "status": "FAIL",
            "failure_reason": {"kind": "NO_PROGRAM_FOUND", "notes": "no depth<=3 program satisfied all train pairs"},
            "plan_trace": plan_trace,
        }

    # Minimal solutions by program_cost_bits.
    solutions.sort(key=lambda st: (int(st.cost_bits), str(st.program_sig())))
    min_cost = int(solutions[0].cost_bits)
    min_solutions = [st for st in solutions if int(st.cost_bits) == int(min_cost)]

    uniq: Dict[str, _StateV128] = {}
    for st in min_solutions:
        sig = st.program_sig()
        uniq[sig] = st
    min_prog_sigs = sorted(uniq.keys())

    outputs: Dict[str, str] = {}
    predicted: Optional[GridV124] = None
    for sig in min_prog_sigs:
        st = uniq[sig]
        gh = grid_hash_v124(st.test_out)
        outputs[sig] = str(gh)
        if predicted is None:
            predicted = st.test_out
        else:
            if grid_hash_v124(predicted) != gh:
                predicted = None

    if predicted is None:
        return {
            "schema_version": 128,
            "kind": "arc_solve_result_v128",
            "status": "UNKNOWN",
            "failure_reason": {"kind": "AMBIGUOUS_RULE", "minimal_programs": list(min_prog_sigs)},
            "program_sigs": list(min_prog_sigs),
            "predicted_grid_hash_by_solution": {str(k): str(outputs[k]) for k in sorted(outputs.keys())},
            "plan_trace": plan_trace,
        }

    chosen_sig = min_prog_sigs[0]
    chosen = uniq[chosen_sig]
    prog = {"schema_version": 128, "kind": "arc_program_v128", "steps": list(chosen.steps)}
    return {
        "schema_version": 128,
        "kind": "arc_solve_result_v128",
        "status": "SOLVED",
        "program": prog,
        "program_sig": str(chosen_sig),
        "program_cost_bits": int(min_cost),
        "program_sigs": list(min_prog_sigs),
        "predicted_grid_hash": str(grid_hash_v124(predicted)),
        "plan_trace": plan_trace,
    }


def diagnose_missing_operator_v128(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> Dict[str, Any]:
    gaps: List[str] = []
    shape_change = False
    out_new_colors = set()
    multi_changed_colors = False
    for inp, out in train_pairs:
        if grid_shape_v124(inp) != grid_shape_v124(out):
            shape_change = True
        d = compute_delta_v128(inp, out)
        if len(d.out_colors_in_changed) > 1:
            multi_changed_colors = True
        out_colors = set(unique_colors_v124(out))
        out_new_colors |= (out_colors - set(unique_colors_v124(inp)))
    if shape_change:
        gaps.append("shape_change_present")
    if out_new_colors:
        gaps.append("introduces_new_colors")
    if multi_changed_colors:
        gaps.append("multi_color_delta_present")
    gaps = sorted(set(str(g) for g in gaps))
    return {
        "schema_version": 128,
        "kind": "arc_operator_gap_diag_v128",
        "gaps": gaps,
        "out_new_colors": [int(c) for c in sorted(out_new_colors)],
    }
