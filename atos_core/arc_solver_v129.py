from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_dsl_v129 import OP_DEFS_V129, apply_op_v129
from .arc_inverse_v129 import InverseCandidateV129, build_inverse_candidates_v129
from .arc_selector_v129 import infer_selector_hypotheses_v129
from .grid_v124 import GridV124, grid_hash_v124, grid_shape_v124, unique_colors_v124

ARC_SOLVER_SCHEMA_VERSION_V129 = 129


def _canonical_step_v129(step: Dict[str, Any]) -> Dict[str, Any]:
    op_id = str(step.get("op_id") or "")
    in_vars = [str(v) for v in (step.get("in_vars") or [])]
    out_var = str(step.get("out_var") or "")
    args_raw = step.get("args") if isinstance(step.get("args"), dict) else {}
    args: Dict[str, Any] = {}
    for k in sorted(args_raw.keys()):
        args[str(k)] = args_raw[k]
    return {"op_id": op_id, "in_vars": list(in_vars), "out_var": out_var, "args": args}


def program_sig_v129(steps: Sequence[Dict[str, Any]]) -> str:
    body = {"schema_version": int(ARC_SOLVER_SCHEMA_VERSION_V129), "kind": "arc_program_v129", "steps": [_canonical_step_v129(s) for s in steps]}
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _step_cost_bits_v129(step: Dict[str, Any]) -> int:
    op = str(step.get("op_id") or "")
    if op in OP_DEFS_V129:
        base = int(OP_DEFS_V129[op].cost_bits)
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


def _program_cost_bits_v129(steps: Sequence[Dict[str, Any]]) -> int:
    return int(sum(_step_cost_bits_v129(_canonical_step_v129(s)) for s in steps))


def _apply_program_v129(*, steps: Sequence[Dict[str, Any]], inp: GridV124, out_var: str) -> GridV124:
    from .grid_v124 import translate_v124

    env: Dict[str, Any] = {"g0": inp}
    for raw in steps:
        st = _canonical_step_v129(raw)
        op = str(st["op_id"])
        ins = [env[str(v)] for v in st["in_vars"]]
        args = dict(st["args"])
        if op in OP_DEFS_V129:
            env[str(st["out_var"])] = apply_op_v129(op_id=op, inputs=ins, args=args)
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


def _mismatch_cells_v129(a: GridV124, b: GridV124) -> int:
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


def _palette_gap_v129(g: GridV124, target: GridV124) -> int:
    have = set(int(c) for c in unique_colors_v124(g))
    want = set(int(c) for c in unique_colors_v124(target))
    return int(len(want - have))


def _objective_key_v129(
    *,
    residual_cells: int,
    residual_palette_gap: int,
    cost_bits: int,
    tie_sig: str,
) -> Tuple[int, int, int, str]:
    return int(residual_cells), int(residual_palette_gap), int(cost_bits), str(tie_sig)


def _rebase_steps_block_v129(
    *,
    block_steps: Sequence[Dict[str, Any]],
    current_var: str,
    suffix: str,
) -> Tuple[List[Dict[str, Any]], str]:
    mapping: Dict[str, str] = {"gC": str(current_var), "g0": "g0"}

    def map_var(v: str) -> str:
        vv = str(v)
        if vv in mapping:
            return mapping[vv]
        mapping[vv] = str(vv) + str(suffix)
        return mapping[vv]

    rebased: List[Dict[str, Any]] = []
    for raw in block_steps:
        st = _canonical_step_v129(raw)
        in_vars = [map_var(v) for v in st["in_vars"]]
        out_var = map_var(st["out_var"])
        rebased.append({"op_id": str(st["op_id"]), "in_vars": in_vars, "out_var": out_var, "args": dict(st["args"])})
    last_out = rebased[-1]["out_var"] if rebased else str(current_var)
    return rebased, str(last_out)


@dataclass(frozen=True)
class _StateV129:
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
        return program_sig_v129(self.steps)


def _pick_focus_pair_v129(residual_cells_per_pair: Sequence[int]) -> int:
    best_i = 0
    best_v = -1
    for i, v in enumerate(residual_cells_per_pair):
        vv = int(v)
        if vv > best_v:
            best_v = vv
            best_i = int(i)
    return int(best_i)


def _bg_candidates_v129(*, grids: Sequence[GridV124]) -> List[int]:
    bgs: List[int] = [0]
    for g in grids:
        h, w = grid_shape_v124(g)
        if h > 0 and w > 0:
            bgs.extend([int(g[0][0]), int(g[0][w - 1]), int(g[h - 1][0]), int(g[h - 1][w - 1])])
    return sorted(set(int(x) for x in bgs))


def _apply_block_placeholder_v129(*, steps: Sequence[Dict[str, Any]], g_cur: GridV124) -> GridV124:
    from .grid_v124 import translate_v124

    env: Dict[str, Any] = {"gC": g_cur}
    for raw in steps:
        st = _canonical_step_v129(raw)
        op = str(st["op_id"])
        ins = [env[str(v)] for v in st["in_vars"]]
        args = dict(st["args"])
        out_var = str(st["out_var"])
        if op in OP_DEFS_V129:
            env[out_var] = apply_op_v129(op_id=op, inputs=ins, args=args)
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
    out = env.get(str(steps[-1]["out_var"])) if steps else g_cur
    if not (isinstance(out, tuple) and (not out or isinstance(out[0], tuple))):
        raise ValueError("block_output_not_grid")
    return out  # type: ignore[return-value]


def solve_arc_task_v129(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    max_depth: int = 4,
    beam_width: int = 48,
    max_states_per_depth: int = 64,
    max_step_regress_total: int = 6,
    max_step_regress_per_pair: int = 3,
) -> Dict[str, Any]:
    if not train_pairs:
        return {"schema_version": 129, "kind": "arc_solve_result_v129", "status": "FAIL", "failure_reason": {"kind": "NO_TRAIN_PAIRS"}}

    train_inputs = [p[0] for p in train_pairs]
    train_targets = [p[1] for p in train_pairs]

    init_train_outs = list(train_inputs)
    init_test_out = test_in

    init_res_cells = [int(_mismatch_cells_v129(g, t)) for g, t in zip(init_train_outs, train_targets)]
    init_res_pal = [int(_palette_gap_v129(g, t)) for g, t in zip(init_train_outs, train_targets)]

    init_state = _StateV129(
        depth=0,
        steps=tuple(),
        current_var="g0",
        cost_bits=0,
        train_outs=tuple(init_train_outs),
        test_out=init_test_out,
        residual_cells_per_pair=tuple(int(x) for x in init_res_cells),
        residual_palette_gap_per_pair=tuple(int(x) for x in init_res_pal),
    )

    plan_trace: Dict[str, Any] = {"schema_version": 129, "kind": "arc_plan_trace_v129", "candidate_traces": []}

    beam: List[_StateV129] = [init_state]
    seen_state_sigs: set[str] = set()
    solutions: List[_StateV129] = []

    for depth in range(max_depth + 1):
        for st in beam:
            if st.residual_cells_total() == 0 and st.residual_palette_gap_total() == 0:
                solutions.append(st)
        if depth == max_depth:
            break

        next_states: List[_StateV129] = []
        for st_idx, st in enumerate(beam):
            if st.residual_cells_total() == 0 and st.residual_palette_gap_total() == 0:
                continue

            focus_i = _pick_focus_pair_v129(list(st.residual_cells_per_pair))
            inp0 = st.train_outs[int(focus_i)]
            out0 = train_targets[int(focus_i)]

            curr_pairs = list(zip(list(st.train_outs), train_targets))
            selector_hyps = infer_selector_hypotheses_v129(train_pairs=curr_pairs)

            cands, cand_trace = build_inverse_candidates_v129(inp=inp0, out=out0, selector_hypotheses=selector_hyps)

            # Rank candidates by benefit on the focus pair per cost (deterministic).
            prev_focus = int(st.residual_cells_per_pair[int(focus_i)])
            ranked: List[Tuple[int, int, str, InverseCandidateV129]] = []
            for cand in cands:
                try:
                    got = _apply_block_placeholder_v129(steps=list(cand.steps), g_cur=inp0)
                except Exception:
                    continue
                new_focus = int(_mismatch_cells_v129(got, out0))
                benefit = int(prev_focus - new_focus)
                ratio = int((benefit * 1024) // (int(cand.cost_bits) + 1))
                ranked.append((int(-ratio), int(cand.cost_bits), str(cand.candidate_sig()), cand))
            ranked.sort(key=lambda t: (int(t[0]), int(t[1]), str(t[2])))
            cands_ranked = [t[3] for t in ranked[: int(beam_width)]]

            if len(plan_trace["candidate_traces"]) < 8:
                plan_trace["candidate_traces"].append(
                    {
                        "depth": int(depth),
                        "state_index": int(st_idx),
                        "focus_pair": int(focus_i),
                        "cand_trace": cand_trace,
                        "candidates_considered": int(len(cands)),
                        "candidates_ranked": int(len(cands_ranked)),
                    }
                )

            for j, cand in enumerate(cands_ranked):
                suffix = f"_d{depth}_i{st_idx}_c{j}"
                rebased_steps, new_current_var = _rebase_steps_block_v129(block_steps=cand.steps, current_var=st.current_var, suffix=suffix)
                new_steps = list(st.steps) + rebased_steps
                new_cost = int(_program_cost_bits_v129(new_steps))

                new_train_outs: List[GridV124] = []
                ok_apply = True
                for inp in train_inputs:
                    try:
                        got = _apply_program_v129(steps=new_steps, inp=inp, out_var=new_current_var)
                    except Exception:
                        ok_apply = False
                        break
                    new_train_outs.append(got)
                if not ok_apply:
                    continue

                new_res_cells = [int(_mismatch_cells_v129(g, t)) for g, t in zip(new_train_outs, train_targets)]
                new_res_pal = [int(_palette_gap_v129(g, t)) for g, t in zip(new_train_outs, train_targets)]

                # Planning with bounded regression (general, deterministic).
                if sum(new_res_cells) > st.residual_cells_total() + int(max_step_regress_total):
                    continue
                if any(int(n) > int(o) + int(max_step_regress_per_pair) for n, o in zip(new_res_cells, st.residual_cells_per_pair)):
                    continue

                try:
                    new_test_out = _apply_program_v129(steps=new_steps, inp=test_in, out_var=new_current_var)
                except Exception:
                    continue

                prog_sig = program_sig_v129(new_steps)
                if prog_sig in seen_state_sigs:
                    continue
                seen_state_sigs.add(prog_sig)

                next_states.append(
                    _StateV129(
                        depth=int(depth + 1),
                        steps=tuple(_canonical_step_v129(s) for s in new_steps),
                        current_var=str(new_current_var),
                        cost_bits=int(new_cost),
                        train_outs=tuple(new_train_outs),
                        test_out=new_test_out,
                        residual_cells_per_pair=tuple(int(x) for x in new_res_cells),
                        residual_palette_gap_per_pair=tuple(int(x) for x in new_res_pal),
                    )
                )

        next_states.sort(
            key=lambda st: _objective_key_v129(
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
            "schema_version": 129,
            "kind": "arc_solve_result_v129",
            "status": "FAIL",
            "failure_reason": {"kind": "NO_PROGRAM_FOUND", "notes": "no depth<=max_depth program satisfied all train pairs"},
            "plan_trace": plan_trace,
        }

    solutions.sort(key=lambda st: (int(st.cost_bits), str(st.program_sig())))
    min_cost = int(solutions[0].cost_bits)
    min_solutions = [st for st in solutions if int(st.cost_bits) == int(min_cost)]

    uniq: Dict[str, _StateV129] = {}
    for st in min_solutions:
        uniq[st.program_sig()] = st
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
            "schema_version": 129,
            "kind": "arc_solve_result_v129",
            "status": "UNKNOWN",
            "failure_reason": {"kind": "AMBIGUOUS_RULE", "minimal_programs": list(min_prog_sigs)},
            "program_sigs": list(min_prog_sigs),
            "predicted_grid_hash_by_solution": {str(k): str(outputs[k]) for k in sorted(outputs.keys())},
            "plan_trace": plan_trace,
        }

    chosen_sig = min_prog_sigs[0]
    chosen = uniq[chosen_sig]
    prog = {"schema_version": 129, "kind": "arc_program_v129", "steps": list(chosen.steps)}
    return {
        "schema_version": 129,
        "kind": "arc_solve_result_v129",
        "status": "SOLVED",
        "program_sig": str(chosen_sig),
        "program": prog,
        "predicted": [list(r) for r in predicted],
        "predicted_grid_hash": str(grid_hash_v124(predicted)),
        "plan_trace": plan_trace,
    }
