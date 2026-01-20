from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_delta_v127 import DeltaEvidenceV127, compute_delta_v127
from .arc_dsl_v126 import OP_DEFS_V126, apply_op_v126
from .arc_inverse_v127 import (
    BboxExprV127,
    InverseCandidateV127,
    inverse_propose_draw_rect_border_v127,
    inverse_propose_fill_rect_v127,
    inverse_propose_map_colors_v127,
    inverse_propose_translate_v127,
    propose_bbox_exprs_v127,
)
from .arc_selector_v127 import SelectorHypothesisV127, infer_selector_hypotheses_v127
from .grid_v124 import GridV124, grid_equal_v124, grid_hash_v124, grid_shape_v124, unique_colors_v124

ARC_SOLVER_SCHEMA_VERSION_V127 = 127


def _mode_color_v127(g: GridV124) -> int:
    counts: Dict[int, int] = {}
    for row in g:
        for x in row:
            counts[int(x)] = int(counts.get(int(x), 0)) + 1
    if not counts:
        return 0
    items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
    return int(items[0][0])


def _bg_candidates_v127(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> List[int]:
    bgs: List[int] = [0]
    for inp, _out in train_pairs:
        bgs.append(_mode_color_v127(inp))
    bgs.append(_mode_color_v127(test_in))
    return sorted(set(int(x) for x in bgs))


def _canonical_step_v127(step: Dict[str, Any]) -> Dict[str, Any]:
    op_id = str(step.get("op_id") or "")
    in_vars = [str(v) for v in (step.get("in_vars") or [])]
    out_var = str(step.get("out_var") or "")
    args_raw = step.get("args") if isinstance(step.get("args"), dict) else {}
    args: Dict[str, Any] = {}
    for k in sorted(args_raw.keys()):
        args[str(k)] = args_raw[k]
    return {"op_id": op_id, "in_vars": list(in_vars), "out_var": out_var, "args": args}


def program_sig_v127(steps: Sequence[Dict[str, Any]]) -> str:
    body = {"schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V127), "kind": "arc_program_v127", "steps": [_canonical_step_v127(s) for s in steps]}
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _apply_program_steps_v127(steps: Sequence[Dict[str, Any]], g_in: GridV124) -> GridV124:
    from .grid_v124 import translate_v124

    env: Dict[str, Any] = {"g0": g_in}
    for raw in steps:
        st = _canonical_step_v127(raw)
        op = str(st["op_id"])
        ins = [env[str(v)] for v in st["in_vars"]]
        out_var = str(st["out_var"])
        args = dict(st["args"])
        if op in OP_DEFS_V126:
            env[out_var] = apply_op_v126(op_id=op, inputs=ins, args=args)
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
            env[out_var] = translate_v124(
                g,
                dx=int(args["dx"]),
                dy=int(args["dy"]),
                pad=int(args.get("pad", 0)),
            )
            continue
        raise ValueError(f"unknown_op:{op}")
    if not steps:
        return g_in
    out = env[str(_canonical_step_v127(steps[-1])["out_var"])]
    if not (isinstance(out, tuple) and (not out or isinstance(out[0], tuple))):
        raise ValueError("program_output_not_grid")
    return out  # type: ignore[return-value]


def _program_cost_bits_v127(steps: Sequence[Dict[str, Any]]) -> int:
    # Mirror v126 costs, but keep versioned.
    cost = 0
    for raw in steps:
        st = _canonical_step_v127(raw)
        op = str(st["op_id"])
        if op in OP_DEFS_V126:
            base = int(OP_DEFS_V126[op].cost_bits)
        elif op in {"map_colors"}:
            base = 16
        elif op in {"translate"}:
            base = 12
        else:
            base = 20
        extra = 0
        for k, v in st["args"].items():
            if k == "mapping" and isinstance(v, dict):
                extra += 8 * int(len(v))
            else:
                extra += 4
        extra += 2 * max(0, int(len(st["in_vars"])) - 1)
        cost += int(base + extra)
    return int(cost)


def _dedup_candidates_v127(cands: Sequence[InverseCandidateV127]) -> List[InverseCandidateV127]:
    seen: set[str] = set()
    out: List[InverseCandidateV127] = []
    for c in cands:
        sig = str(c.candidate_sig())
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    out.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out


def _infer_global_mapping_candidate_v127(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> Optional[InverseCandidateV127]:
    mapping_sig = ""
    cand0: Optional[InverseCandidateV127] = None
    for inp, out in train_pairs:
        cands = inverse_propose_map_colors_v127(inp=inp, out=out)
        if not cands:
            return None
        c = cands[0]
        m = c.steps[0].get("args", {}).get("mapping", {})
        sig = canonical_json_dumps(m)
        if mapping_sig and sig != mapping_sig:
            return None
        mapping_sig = sig
        cand0 = c
    return cand0


def _infer_common_translate_candidates_v127(
    train_pairs: Sequence[Tuple[GridV124, GridV124]], bg_candidates: Sequence[int]
) -> List[InverseCandidateV127]:
    common: Optional[set[Tuple[int, int, int]]] = None
    examples: Dict[Tuple[int, int, int], InverseCandidateV127] = {}
    for inp, out in train_pairs:
        cands = inverse_propose_translate_v127(inp=inp, out=out, bg_candidates=bg_candidates)
        keys: set[Tuple[int, int, int]] = set()
        for c in cands:
            args = c.steps[0].get("args", {})
            key = (int(args.get("dx", 0)), int(args.get("dy", 0)), int(args.get("pad", 0)))
            keys.add(key)
            examples[key] = c
        if common is None:
            common = set(keys)
        else:
            common &= set(keys)
    if not common:
        return []
    out_cands = [examples[k] for k in sorted(common)]
    return _dedup_candidates_v127(out_cands)


def _build_bbox_exprs_v127(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    bg_candidates: Sequence[int],
) -> Tuple[List[BboxExprV127], List[SelectorHypothesisV127]]:
    selector_hyps = infer_selector_hypotheses_v127(train_pairs=train_pairs)
    selector_hyp_dicts: List[Dict[str, Any]] = []
    for h in selector_hyps:
        selector_hyp_dicts.append({"color_filter": h.color_filter, "selector": str(h.selector)})
    color_candidates: List[int] = []
    for inp, _out in train_pairs:
        color_candidates.extend(unique_colors_v124(inp))
    color_candidates.extend(unique_colors_v124(test_in))
    bbox_exprs = propose_bbox_exprs_v127(
        bg_candidates=list(bg_candidates),
        color_candidates=sorted(set(int(c) for c in color_candidates)),
        selector_hypotheses=selector_hyp_dicts,
        max_expand_delta=2,
    )
    return bbox_exprs, selector_hyps


def _generate_atom_candidates_v127(
    *, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124
) -> Tuple[List[InverseCandidateV127], Dict[str, Any]]:
    """
    Generate a small, deterministic candidate set of 1-step "atom programs" derived via inverse_propose,
    filtering only for applicability across all train inputs (not for full match).
    """
    if not train_pairs:
        return [], {"kind": "empty_train_pairs"}
    bg_candidates = _bg_candidates_v127(train_pairs=train_pairs, test_in=test_in)
    bbox_exprs, selector_hyps = _build_bbox_exprs_v127(train_pairs=train_pairs, test_in=test_in, bg_candidates=bg_candidates)

    delta0 = compute_delta_v127(train_pairs[0][0], train_pairs[0][1])
    # Candidate set built from first pair's delta; then we filter for applicability across all inputs.
    cands: List[InverseCandidateV127] = []

    mc = _infer_global_mapping_candidate_v127(train_pairs)
    if mc is not None:
        cands.append(mc)
    cands.extend(_infer_common_translate_candidates_v127(train_pairs, bg_candidates))
    cands.extend(inverse_propose_fill_rect_v127(inp=train_pairs[0][0], out=train_pairs[0][1], delta=delta0, bbox_exprs=bbox_exprs))
    cands.extend(
        inverse_propose_draw_rect_border_v127(inp=train_pairs[0][0], out=train_pairs[0][1], delta=delta0, bbox_exprs=bbox_exprs)
    )

    cands = _dedup_candidates_v127(cands)

    # Filter by applicability: must execute on every train input without error.
    applicable: List[InverseCandidateV127] = []
    for c in cands:
        ok = True
        for inp, _out in train_pairs:
            try:
                _ = _apply_program_steps_v127(c.steps, inp)
            except Exception:
                ok = False
                break
        if ok:
            applicable.append(c)

    trace = {
        "schema_version": 127,
        "kind": "arc_atom_generation_trace_v127",
        "bg_candidates": [int(x) for x in bg_candidates],
        "delta0": delta0.to_dict(),
        "selector_hypotheses": [{"sig": h.hypothesis_sig(), **h.to_dict()} for h in selector_hyps],
        "bbox_exprs_top": [e.to_dict() for e in bbox_exprs[:12]],
        "candidates_top": [c.to_dict() for c in applicable[:20]],
    }
    return applicable, trace


def _compose_steps_v127(step1: Sequence[Dict[str, Any]], step2: Sequence[Dict[str, Any]], *, in_var_for_step2: str) -> List[Dict[str, Any]]:
    # Rebase step2 vars: g0 -> in_var_for_step2; other vars get suffix "_2".
    suffix = "_2"
    mapping: Dict[str, str] = {"g0": str(in_var_for_step2)}

    def map_var(v: str) -> str:
        if v in mapping:
            return mapping[v]
        mapping[v] = str(v) + suffix
        return mapping[v]

    rebased: List[Dict[str, Any]] = []
    for raw in step2:
        st = _canonical_step_v127(raw)
        in_vars = [map_var(str(v)) for v in st["in_vars"]]
        out_var = map_var(str(st["out_var"]))
        rebased.append({"op_id": str(st["op_id"]), "in_vars": in_vars, "out_var": out_var, "args": dict(st["args"])})
    return list(step1) + rebased


def solve_arc_task_v127(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> Dict[str, Any]:
    """
    V127 solver: inverse-propose (abduction) + selector induction + typed composition (depth 1/2).
    No templates by family/task_id. Train_pairs + test_in only.
    """
    plan_trace: Dict[str, Any] = {"schema_version": 127, "kind": "arc_plan_trace_v127"}

    atom_cands, atom_trace = _generate_atom_candidates_v127(train_pairs=train_pairs, test_in=test_in)
    plan_trace["atom_trace"] = dict(atom_trace)

    solutions: List[Tuple[int, List[Dict[str, Any]]]] = []

    # Depth 1: direct candidates
    for c in atom_cands:
        ok = True
        for inp, out in train_pairs:
            got = _apply_program_steps_v127(c.steps, inp)
            if not grid_equal_v124(got, out):
                ok = False
                break
        if ok:
            steps = [_canonical_step_v127(s) for s in c.steps]
            solutions.append((int(_program_cost_bits_v127(steps)), steps))

    # Depth 2: compose two atom candidates, guided by residual deltas
    if not solutions:
        max_first = 24
        max_second = 24
        for c1 in atom_cands[:max_first]:
            mids: List[Tuple[GridV124, GridV124]] = []
            ok_mid = True
            for inp, out in train_pairs:
                try:
                    mid = _apply_program_steps_v127(c1.steps, inp)
                except Exception:
                    ok_mid = False
                    break
                mids.append((mid, out))
            if not ok_mid:
                continue
            atom2, atom2_trace = _generate_atom_candidates_v127(train_pairs=mids, test_in=_apply_program_steps_v127(c1.steps, test_in))
            # keep a small trace sample for the first few c1
            if "depth2_traces" not in plan_trace:
                plan_trace["depth2_traces"] = []
            if len(plan_trace["depth2_traces"]) < 5:
                plan_trace["depth2_traces"].append(
                    {
                        "step1_candidate_sig": str(c1.candidate_sig()),
                        "step2_atom_trace": atom2_trace,
                    }
                )
            out_var1 = str(_canonical_step_v127(c1.steps[-1])["out_var"])
            for c2 in atom2[:max_second]:
                combined = _compose_steps_v127(c1.steps, c2.steps, in_var_for_step2=out_var1)
                ok = True
                for inp, out in train_pairs:
                    got = _apply_program_steps_v127(combined, inp)
                    if not grid_equal_v124(got, out):
                        ok = False
                        break
                if not ok:
                    continue
                steps = [_canonical_step_v127(s) for s in combined]
                solutions.append((int(_program_cost_bits_v127(steps)), steps))

    if not solutions:
        fr = {"kind": "NO_PROGRAM_FOUND", "notes": "no depth<=2 program satisfied all train pairs"}
        return {
            "schema_version": 127,
            "kind": "arc_solve_result_v127",
            "status": "FAIL",
            "failure_reason": fr,
            "plan_trace": plan_trace,
        }

    # Keep minimal-cost programs that satisfy train.
    solutions.sort(key=lambda t: (int(t[0]), program_sig_v127(t[1])))
    min_cost = int(solutions[0][0])
    min_solutions = [s for s in solutions if int(s[0]) == int(min_cost)]
    # Dedup by program_sig
    uniq: Dict[str, List[Dict[str, Any]]] = {}
    for _cost, steps in min_solutions:
        sig = program_sig_v127(steps)
        uniq[sig] = steps
    min_prog_sigs = sorted(uniq.keys())

    outputs: Dict[str, str] = {}
    predicted: Optional[GridV124] = None
    for sig in min_prog_sigs:
        steps = uniq[sig]
        got = _apply_program_steps_v127(steps, test_in)
        gh = grid_hash_v124(got)
        outputs[sig] = str(gh)
        if predicted is None:
            predicted = got
        else:
            if grid_hash_v124(predicted) != gh:
                predicted = None

    if predicted is None:
        return {
            "schema_version": 127,
            "kind": "arc_solve_result_v127",
            "status": "UNKNOWN",
            "failure_reason": {"kind": "AMBIGUOUS_RULE", "minimal_programs": list(min_prog_sigs)},
            "program_sigs": list(min_prog_sigs),
            "predicted_grid_hash_by_solution": {str(k): str(outputs[k]) for k in sorted(outputs.keys())},
            "plan_trace": plan_trace,
        }

    pred_hash = grid_hash_v124(predicted)
    chosen_sig = min_prog_sigs[0]
    chosen_steps = uniq[chosen_sig]
    prog = {"schema_version": 127, "kind": "arc_program_v127", "steps": list(chosen_steps)}
    return {
        "schema_version": 127,
        "kind": "arc_solve_result_v127",
        "status": "SOLVED",
        "program": prog,
        "program_sig": str(program_sig_v127(chosen_steps)),
        "program_cost_bits": int(min_cost),
        "program_sigs": list(min_prog_sigs),
        "predicted_grid_hash": str(pred_hash),
        "plan_trace": plan_trace,
    }


def diagnose_missing_operator_v127(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> Dict[str, Any]:
    gaps: List[str] = []
    shape_change = False
    out_new_colors = set()
    multi_changed_colors = False
    for inp, out in train_pairs:
        if grid_shape_v124(inp) != grid_shape_v124(out):
            shape_change = True
        d = compute_delta_v127(inp, out)
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
        "schema_version": 127,
        "kind": "arc_operator_gap_diag_v127",
        "gaps": gaps,
        "out_new_colors": [int(c) for c in sorted(out_new_colors)],
    }

