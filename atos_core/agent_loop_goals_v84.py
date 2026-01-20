from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .agent_loop_v80 import run_goal_spec_v80
from .goal_act_v75 import goal_sig_v75, goal_v75_is_satisfied, goal_v75_update_from_run, list_goal_acts_v75
from .goal_spec_v72 import GoalSpecV72
from .mine_promote_v74 import (
    extract_rep_steps,
    materialize_composed_act_v74,
    mine_candidates_v74,
    mutate_bindings_plus1_numeric,
)
from .pcc_v84 import build_certificate_v84, verify_pcc_v84
from .store import ActStore
from .trace_v73 import TraceV73, trace_from_agent_loop_v72


def _fail(msg: str) -> None:
    raise ValueError(msg)


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"path_exists:{path}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_canon(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


_EPOCH = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)


def goal_created_step_v75(goal_act) -> int:
    """
    Deterministic created_step derived from goal_act.created_at (deterministic_iso).
    """
    try:
        ts = str(getattr(goal_act, "created_at", "") or "")
        if not ts:
            return 0
        dt = _dt.datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return int((dt - _EPOCH).total_seconds())
    except Exception:
        return 0


def run_dir_sha256_v84(*, run_dir: str) -> str:
    mg_dir = os.path.join(str(run_dir), "mind_graph")
    nodes_path = os.path.join(mg_dir, "mind_nodes.jsonl")
    edges_path = os.path.join(mg_dir, "mind_edges.jsonl")
    nodes_sha = sha256_file(nodes_path)
    edges_sha = sha256_file(edges_path)
    return _sha256_canon({"mind_nodes_sha256": str(nodes_sha), "mind_edges_sha256": str(edges_sha)})


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(canonical_json_dumps(row))
        f.write("\n")


def _write_json_once(path: str, obj: Any) -> None:
    ensure_absent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)


def goal_spec_v72_from_goal_act_v75(goal_act) -> Tuple[Optional[GoalSpecV72], str]:
    if goal_act is None or str(getattr(goal_act, "kind", "")) != "goal_v75":
        return None, "not_goal_v75_act"
    ev = goal_act.evidence if isinstance(goal_act.evidence, dict) else {}
    goal = ev.get("goal") if isinstance(ev.get("goal"), dict) else {}

    goal_kind = str(goal.get("goal_kind") or "")
    bindings = goal.get("bindings") if isinstance(goal.get("bindings"), dict) else {}
    output_key = str(goal.get("output_key") or "")
    expected = goal.get("expected")
    validator_id = str(goal.get("validator_id") or "text_exact")
    if not output_key:
        return None, "missing_output_key"
    return (
        GoalSpecV72(
            goal_kind=str(goal_kind),
            bindings={str(k): bindings.get(k) for k in sorted(bindings.keys(), key=str)},
            output_key=str(output_key),
            expected=expected,
            validator_id=str(validator_id),
            created_step=0,
        ),
        "ok",
    )


def goal_kind_from_goal_act_v75(goal_act) -> str:
    if goal_act is None or str(getattr(goal_act, "kind", "")) != "goal_v75":
        return ""
    ev = goal_act.evidence if isinstance(goal_act.evidence, dict) else {}
    goal = ev.get("goal") if isinstance(ev.get("goal"), dict) else {}
    return str(goal.get("goal_kind") or "")


def _find_subpath_start(path: Sequence[str], subpath: Sequence[str]) -> Optional[int]:
    p = [str(x) for x in path]
    sp = [str(x) for x in subpath]
    if not sp or len(sp) > len(p):
        return None
    for i in range(0, len(p) - len(sp) + 1):
        if p[i : i + len(sp)] == sp:
            return int(i)
    return None


def _trace_contains_subpath(trace: TraceV73, subpath: Sequence[str]) -> bool:
    try:
        return _find_subpath_start(trace.acts_path(), subpath) is not None
    except Exception:
        return False


def _candidate_goal_kinds_supported(cand, traces_ok: Sequence[TraceV73]) -> List[str]:
    kinds: List[str] = []
    for tr in traces_ok:
        if not isinstance(tr, TraceV73):
            continue
        if _trace_contains_subpath(tr, cand.subpath):
            k = str(tr.goal_kind or "")
            if k and k not in kinds:
                kinds.append(k)
    kinds.sort(key=str)
    return kinds


def _execute_prefix_state(
    *,
    store_base: ActStore,
    trace: TraceV73,
    upto_idx: int,
    seed: int = 0,
) -> Dict[str, Any]:
    if int(upto_idx) <= 0:
        return dict(trace.bindings)

    from .engine import Engine, EngineConfig

    vars_state: Dict[str, Any] = dict(trace.bindings)
    engine = Engine(store_base, seed=int(seed), config=EngineConfig(enable_contracts=False))
    steps = list(trace.steps)
    for i, st in enumerate(steps[: int(upto_idx)]):
        bm = st.bind_map if isinstance(st.bind_map, dict) else {}
        inps: Dict[str, Any] = {}
        for slot in sorted(bm.keys(), key=str):
            vn = str(bm.get(slot) or "")
            inps[str(slot)] = vars_state.get(vn)
        out = engine.execute_concept_csv(
            concept_act_id=str(st.concept_id),
            inputs=dict(inps),
            expected=None,
            step=int(i),
            max_depth=8,
            max_events=512,
            validate_output=False,
        )
        meta = out.get("meta") if isinstance(out, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        out_text = str(meta.get("output_text") or out.get("output") or "")
        vars_state[str(st.produces)] = out_text
    return dict(vars_state)


def _expected_for_steps(
    *,
    store_base: ActStore,
    steps: Sequence,
    start_state: Dict[str, Any],
    seed: int = 0,
) -> str:
    from .mine_promote_v74 import execute_steps_expected_output

    return execute_steps_expected_output(store_base=store_base, steps=steps, bindings=start_state, seed=int(seed))


def _build_vector_specs_v84(
    *,
    store_base: ActStore,
    act_candidate,
    cand,
    traces_ok: Sequence[TraceV73],
    kinds: Sequence[str],
    seed: int,
    max_base_vectors: int = 3,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    Build deterministic vector_specs WITH explicit goal_kind and goal_kind coverage.
    Requires >=1 vector per kind in `kinds` and >=3 vectors total (incl. extra).
    """
    input_keys: List[str] = []
    if isinstance(act_candidate.evidence, dict):
        iface = act_candidate.evidence.get("interface") if isinstance(act_candidate.evidence.get("interface"), dict) else {}
        iface = iface if isinstance(iface, dict) else {}
        in_schema = iface.get("input_schema") if isinstance(iface.get("input_schema"), dict) else {}
        input_keys = [str(k) for k in sorted(in_schema.keys(), key=str)]

    support_set = set(str(x) for x in (cand.contexts or []) if str(x))
    support_traces = [t for t in traces_ok if isinstance(t, TraceV73) and str(t.context_id) in support_set]
    support_traces.sort(key=lambda t: str(t.trace_sig()))
    if not support_traces:
        return None, "insufficient_support_traces"

    kinds2 = [str(k) for k in kinds if str(k)]
    kinds2 = sorted(set(kinds2), key=str)
    if not kinds2:
        return None, "insufficient_vector_goal_kind_coverage_global_kind_coverage_fail"

    chosen_by_kind: Dict[str, TraceV73] = {}
    for gk in kinds2:
        cands = [t for t in support_traces if str(t.goal_kind) == str(gk) and _trace_contains_subpath(t, cand.subpath)]
        cands.sort(key=lambda t: str(t.trace_sig()))
        if not cands:
            return None, "insufficient_vector_goal_kind_coverage_global_kind_coverage_fail"
        chosen_by_kind[str(gk)] = cands[0]

    used_trace_sigs = set()
    base_vectors: List[Dict[str, Any]] = []
    for gk in kinds2:
        tr = chosen_by_kind[str(gk)]
        used_trace_sigs.add(str(tr.trace_sig()))
        start = _find_subpath_start(tr.acts_path(), cand.subpath)
        if start is None:
            return None, "insufficient_vector_goal_kind_coverage_global_kind_coverage_fail"
        state0 = _execute_prefix_state(store_base=store_base, trace=tr, upto_idx=int(start), seed=int(seed))
        sub_steps = list(tr.steps)[int(start) : int(start) + int(len(cand.subpath))]
        exp = _expected_for_steps(store_base=store_base, steps=sub_steps, start_state=state0, seed=int(seed))
        inputs = {k: state0.get(k) for k in input_keys}
        ctx_sig = _sha256_canon({"goal_kind": str(gk), "ctx": str(tr.context_id), "sub_sig": str(cand.sub_sig), "inputs": inputs})
        base_vectors.append(
            {
                "context_id": f"{gk}:{tr.context_id}:{ctx_sig}",
                "goal_kind": str(gk),
                "inputs": dict(inputs),
                "expected": str(exp),
            }
        )

    for tr in support_traces:
        if len(base_vectors) >= int(max_base_vectors):
            break
        if str(tr.trace_sig()) in used_trace_sigs:
            continue
        start = _find_subpath_start(tr.acts_path(), cand.subpath)
        if start is None:
            continue
        gk = str(tr.goal_kind or "")
        if not gk:
            continue
        state0 = _execute_prefix_state(store_base=store_base, trace=tr, upto_idx=int(start), seed=int(seed))
        sub_steps = list(tr.steps)[int(start) : int(start) + int(len(cand.subpath))]
        exp = _expected_for_steps(store_base=store_base, steps=sub_steps, start_state=state0, seed=int(seed))
        inputs = {k: state0.get(k) for k in input_keys}
        ctx_sig = _sha256_canon({"goal_kind": str(gk), "ctx": str(tr.context_id), "sub_sig": str(cand.sub_sig), "inputs": inputs})
        base_vectors.append(
            {
                "context_id": f"{gk}:{tr.context_id}:{ctx_sig}",
                "goal_kind": str(gk),
                "inputs": dict(inputs),
                "expected": str(exp),
            }
        )
        used_trace_sigs.add(str(tr.trace_sig()))

    base_kind = kinds2[0]
    base_trace = chosen_by_kind[base_kind]
    start0 = _find_subpath_start(base_trace.acts_path(), cand.subpath)
    if start0 is None:
        return None, "insufficient_vector_goal_kind_coverage_global_kind_coverage_fail"
    mutated_bindings = mutate_bindings_plus1_numeric(bindings=dict(base_trace.bindings), key_preference=["x", "y"])
    trace_mut = TraceV73(
        context_id=str(base_trace.context_id),
        goal_sig=str(base_trace.goal_sig),
        goal_id=str(base_trace.goal_id),
        goal_kind=str(base_trace.goal_kind),
        bindings=dict(mutated_bindings),
        output_key=str(base_trace.output_key),
        expected=base_trace.expected,
        validator_id=str(base_trace.validator_id),
        steps=list(base_trace.steps),
        outcome=dict(base_trace.outcome),
        cost_units=dict(base_trace.cost_units),
    )
    state_mut = _execute_prefix_state(store_base=store_base, trace=trace_mut, upto_idx=int(start0), seed=int(seed))
    sub_steps0 = list(base_trace.steps)[int(start0) : int(start0) + int(len(cand.subpath))]
    exp_mut = _expected_for_steps(store_base=store_base, steps=sub_steps0, start_state=state_mut, seed=int(seed))
    inputs_mut = {k: state_mut.get(k) for k in input_keys}
    extra_ctx = _sha256_canon({"extra": True, "goal_kind": str(base_trace.goal_kind), "sub_sig": str(cand.sub_sig), "inputs": inputs_mut})
    base_vectors.append(
        {
            "context_id": f"extra:{base_trace.goal_kind}:{extra_ctx}",
            "goal_kind": str(base_trace.goal_kind),
            "inputs": dict(inputs_mut),
            "expected": str(exp_mut),
        }
    )

    uniq: Dict[str, Dict[str, Any]] = {}
    for v in base_vectors:
        gk = str(v.get("goal_kind") or "")
        inp = v.get("inputs") if isinstance(v.get("inputs"), dict) else {}
        exp = str(v.get("expected") or "")
        sig = _sha256_canon({"goal_kind": str(gk), "inputs": {str(k): inp.get(k) for k in sorted(inp.keys(), key=str)}, "expected": str(exp)})
        if sig not in uniq:
            uniq[sig] = v
    vectors = [uniq[k] for k in sorted(uniq.keys(), key=str)]

    present_kinds = sorted(set(str(v.get("goal_kind") or "") for v in vectors if str(v.get("goal_kind") or "")), key=str)
    for gk in kinds2:
        if str(gk) not in set(present_kinds):
            return None, "insufficient_vector_goal_kind_coverage_global_kind_coverage_fail"
    if len(vectors) < 3:
        return None, "insufficient_vector_specs_after_dedup"

    return list(vectors), "ok"


def _compression_points_from_events(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    idxs: List[int] = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        if str(r.get("event_kind") or "") == "promotion_attempt" and str(r.get("decision") or "") == "promoted":
            idxs.append(int(i))

    for i, ev_idx in enumerate(idxs):
        before = None
        for r in reversed(list(rows[:ev_idx])):
            if not isinstance(r, dict):
                continue
            if str(r.get("event_kind") or "") != "goal_attempt":
                continue
            before = int(r.get("steps_total", 0) or 0)
            break
        after = None
        for r in list(rows[ev_idx + 1 :]):
            if not isinstance(r, dict):
                continue
            if str(r.get("event_kind") or "") != "goal_attempt":
                continue
            after = int(r.get("steps_total", 0) or 0)
            break
        if before is None or after is None:
            continue
        points.append(
            {
                "promotion_index": int(i),
                "steps_before": int(before),
                "steps_after": int(after),
                "delta_steps": int(int(before) - int(after)),
            }
        )
    return points


def run_goals_v84(
    *,
    store: ActStore,
    seed: int,
    out_dir: str,
    max_rounds: int = 10,
    max_goals_per_round: int = 1,
    enable_promotion: bool = True,
    promotion_budget_bits: int = 2048,
    promotion_min_traces: int = 2,
    promotion_top_k: int = 8,
    max_promotions_per_run: int = 3,
    promotion_kind_diversity_min: int = 1,
) -> Dict[str, Any]:
    """
    V84: V83 loop + PCC exec-code-bound verification (fail-closed).
    """
    ensure_absent(out_dir)
    os.makedirs(out_dir, exist_ok=False)

    events_path = os.path.join(out_dir, "goals_v84_events.jsonl")
    ensure_absent(events_path)

    traces: List[TraceV73] = []
    attempts_total = 0
    promoted_total = 0
    used_bits = 0

    promotions_dir = os.path.join(out_dir, "promotion")
    promotions_path = os.path.join(promotions_dir, "v84_promotions.jsonl")
    candidates_dir = os.path.join(out_dir, "candidates")
    traces_path = os.path.join(out_dir, "traces_v84.json")
    mined_path = os.path.join(out_dir, "mined_candidates_v84.json")
    curve_path = os.path.join(out_dir, "compression_curve.json")

    store_hash_init = str(store.content_hash())
    step_ctr = 0
    events_buf: List[Dict[str, Any]] = []
    window_start_idx = 0

    def _emit_event(row: Dict[str, Any]) -> None:
        nonlocal step_ctr
        body = dict(row)
        body["created_at"] = deterministic_iso(step=int(step_ctr))
        _append_jsonl(events_path, body)
        events_buf.append(dict(body))
        step_ctr += 1

    def _emit_promo_row(row: Dict[str, Any]) -> None:
        os.makedirs(promotions_dir, exist_ok=True)
        _append_jsonl(promotions_path, row)

    for r in range(0, int(max_rounds)):
        goals = list_goal_acts_v75(store)
        pending = [g for g in goals if not goal_v75_is_satisfied(g)]
        pending.sort(key=lambda a: (int(goal_created_step_v75(a)), str(getattr(a, "id", ""))))

        if not pending:
            break

        to_run = pending[: int(max_goals_per_round)]
        skipped = pending[int(max_goals_per_round) :]

        for g in skipped:
            store_hash_before = str(store.content_hash())
            _emit_event(
                {
                    "round": int(r),
                    "event_kind": "goal_skipped",
                    "promotion_index": -1,
                    "goal_id": str(getattr(g, "id", "")),
                    "goal_sig": str(goal_sig_v75(g)),
                    "goal_kind": str(goal_kind_from_goal_act_v75(g)),
                    "goal_created_step": int(goal_created_step_v75(g)),
                    "decision": "skipped",
                    "reason": "max_goals_per_round",
                    "plan_sig": "",
                    "trace_sig": "",
                    "steps_total": 0,
                    "run_dir_sha256": "",
                    "candidate_id": "",
                    "certificate_sig": "",
                    "overhead_bits": 0,
                    "gain_bits_est": 0,
                    "used_bits_after": int(used_bits),
                    "store_hash_before": str(store_hash_before),
                    "store_hash_after": str(store_hash_before),
                }
            )

        for g in to_run:
            store_hash_before = str(store.content_hash())
            goal_spec, reason = goal_spec_v72_from_goal_act_v75(g)
            goal_kind = str(goal_kind_from_goal_act_v75(g))
            goal_step = int(goal_created_step_v75(g))
            if goal_spec is None:
                _emit_event(
                    {
                        "round": int(r),
                        "event_kind": "goal_attempt",
                        "promotion_index": -1,
                        "goal_id": str(getattr(g, "id", "")),
                        "goal_sig": str(goal_sig_v75(g)),
                        "goal_kind": str(goal_kind),
                        "goal_created_step": int(goal_step),
                        "decision": "failed",
                        "reason": str(reason),
                        "plan_sig": "",
                        "trace_sig": "",
                        "steps_total": 0,
                        "run_dir_sha256": "",
                        "candidate_id": "",
                        "certificate_sig": "",
                        "overhead_bits": 0,
                        "gain_bits_est": 0,
                        "used_bits_after": int(used_bits),
                        "store_hash_before": str(store_hash_before),
                        "store_hash_after": str(store_hash_before),
                    }
                )
                continue

            attempts_total += 1
            gid = str(getattr(g, "id", "") or "")
            attempt_dir = os.path.join(out_dir, f"round{int(r):02d}", f"goal_{gid[:24]}")
            ensure_absent(attempt_dir)
            os.makedirs(attempt_dir, exist_ok=False)

            res = run_goal_spec_v80(goal_spec=goal_spec, store=store, seed=int(seed), out_dir=attempt_dir)
            ok = bool(res.get("ok", False))
            plan = res.get("plan") if isinstance(res.get("plan"), dict) else {}
            plan_sig = str(plan.get("plan_sig") or "")
            tr = trace_from_agent_loop_v72(goal_spec=goal_spec, result=res)
            traces.append(tr)
            trace_sig = str(tr.trace_sig())
            run_sha = run_dir_sha256_v84(run_dir=attempt_dir)

            updated = goal_v75_update_from_run(
                act=g,
                run_res=res,
                trace_sig=str(trace_sig),
                run_dir_sha256=str(run_sha),
                step=int(step_ctr),
            )
            store.add(updated)

            store_hash_after = str(store.content_hash())
            _emit_event(
                {
                    "round": int(r),
                    "event_kind": "goal_attempt",
                    "promotion_index": -1,
                    "goal_id": str(getattr(g, "id", "")),
                    "goal_sig": str(goal_sig_v75(g)),
                    "goal_kind": str(goal_kind),
                    "goal_created_step": int(goal_step),
                    "decision": "satisfied" if ok else "active",
                    "reason": str(res.get("reason") or ""),
                    "plan_sig": str(plan_sig),
                    "trace_sig": str(trace_sig),
                    "steps_total": int(len(tr.steps)),
                    "run_dir_sha256": str(run_sha),
                    "candidate_id": "",
                    "certificate_sig": "",
                    "overhead_bits": 0,
                    "gain_bits_est": 0,
                    "used_bits_after": int(used_bits),
                    "store_hash_before": str(store_hash_before),
                    "store_hash_after": str(store_hash_after),
                }
            )

        if not bool(enable_promotion):
            continue
        if int(promoted_total) >= int(max_promotions_per_run):
            continue

        traces_ok = [t for t in traces[window_start_idx:] if isinstance(t, TraceV73) and bool(t.outcome.get("ok", False))]
        traces_ok.sort(key=lambda t: str(t.trace_sig()))
        if len(traces_ok) < int(promotion_min_traces):
            continue

        mined, mined_dbg = mine_candidates_v74(
            traces=traces_ok,
            max_k=6,
            min_support=2,
            top_k=int(promotion_top_k),
        )
        if not os.path.exists(mined_path):
            _write_json_once(
                mined_path,
                {
                    "schema_version": 1,
                    "round": int(r),
                    "window_start_idx": int(window_start_idx),
                    "traces_ok": int(len(traces_ok)),
                    "candidates": [c.to_dict() for c in mined],
                    "debug": dict(mined_dbg),
                },
            )

        if not mined:
            continue
        if int(used_bits) >= int(promotion_budget_bits):
            continue

        if not os.path.exists(candidates_dir):
            ensure_absent(candidates_dir)
            os.makedirs(candidates_dir, exist_ok=False)
        if not os.path.exists(promotions_dir):
            ensure_absent(promotions_dir)
            os.makedirs(promotions_dir, exist_ok=False)

        traces_by_sig = {str(t.trace_sig()): t for t in traces_ok}

        for cand_idx, cand in enumerate(mined):
            promotion_index = int(promoted_total)
            store_hash_before = str(store.content_hash())

            kinds = _candidate_goal_kinds_supported(cand, traces_ok)
            kinds = sorted(set(str(k) for k in kinds if str(k)), key=str)
            required_kinds = int(promotion_kind_diversity_min) if int(promotion_index) == 0 else 1
            if len(kinds) < int(required_kinds):
                _emit_event(
                    {
                        "round": int(r),
                        "event_kind": "promotion_attempt",
                        "promotion_index": int(promotion_index),
                        "goal_id": "",
                        "goal_sig": "",
                        "goal_kind": "",
                        "goal_created_step": 0,
                        "decision": "skipped",
                        "reason": "insufficient_goal_kind_diversity",
                        "plan_sig": "",
                        "trace_sig": "",
                        "steps_total": 0,
                        "run_dir_sha256": "",
                        "candidate_id": "",
                        "certificate_sig": "",
                        "overhead_bits": 0,
                        "gain_bits_est": int(cand.gain_bits_est),
                        "used_bits_after": int(used_bits),
                        "store_hash_before": str(store_hash_before),
                        "store_hash_after": str(store_hash_before),
                    }
                )
                _emit_promo_row(
                    {
                        "created_at": deterministic_iso(step=10_000 + int(step_ctr) + int(cand_idx)),
                        "candidate_id": "",
                        "certificate_sig": "",
                        "gain_bits_est": int(cand.gain_bits_est),
                        "overhead_bits": 0,
                        "decision": "skipped",
                        "reason": "insufficient_goal_kind_diversity",
                        "store_hash_before": str(store_hash_before),
                        "store_hash_after": str(store_hash_before),
                        "goal_kinds_supported": list(kinds),
                        "required_goal_kind_diversity": int(required_kinds),
                    }
                )
                continue

            rep_steps = extract_rep_steps(
                traces_by_sig=traces_by_sig,
                rep_trace_sig=str(cand.rep_trace_sig),
                start_idx=int(cand.start_idx),
                subpath_len=int(len(cand.subpath)),
            )
            act, _dbg = materialize_composed_act_v74(
                store_base=store,
                steps=rep_steps,
                support_contexts=int(cand.support_contexts),
                contexts=list(cand.contexts),
                seed_step=0,
            )

            act.match = {"goal_kinds": list(kinds)}
            overhead_bits = int((act.cost or {}).get("overhead_bits", 1024) or 1024)

            if store.get(str(act.id)) is not None:
                _emit_event(
                    {
                        "round": int(r),
                        "event_kind": "promotion_attempt",
                        "promotion_index": int(promotion_index),
                        "goal_id": "",
                        "goal_sig": "",
                        "goal_kind": "",
                        "goal_created_step": 0,
                        "decision": "skipped",
                        "reason": "already_in_store",
                        "plan_sig": "",
                        "trace_sig": "",
                        "steps_total": 0,
                        "run_dir_sha256": "",
                        "candidate_id": str(act.id),
                        "certificate_sig": "",
                        "overhead_bits": int(overhead_bits),
                        "gain_bits_est": int(cand.gain_bits_est),
                        "used_bits_after": int(used_bits),
                        "store_hash_before": str(store_hash_before),
                        "store_hash_after": str(store_hash_before),
                    }
                )
                _emit_promo_row(
                    {
                        "created_at": deterministic_iso(step=11_000 + int(step_ctr) + int(cand_idx)),
                        "candidate_id": str(act.id),
                        "certificate_sig": "",
                        "gain_bits_est": int(cand.gain_bits_est),
                        "overhead_bits": int(overhead_bits),
                        "decision": "skipped",
                        "reason": "already_in_store",
                        "store_hash_before": str(store_hash_before),
                        "store_hash_after": str(store_hash_before),
                    }
                )
                continue

            if int(used_bits) + int(overhead_bits) > int(promotion_budget_bits):
                _emit_event(
                    {
                        "round": int(r),
                        "event_kind": "promotion_attempt",
                        "promotion_index": int(promotion_index),
                        "goal_id": "",
                        "goal_sig": "",
                        "goal_kind": "",
                        "goal_created_step": 0,
                        "decision": "skipped",
                        "reason": "budget_exceeded",
                        "plan_sig": "",
                        "trace_sig": "",
                        "steps_total": 0,
                        "run_dir_sha256": "",
                        "candidate_id": str(act.id),
                        "certificate_sig": "",
                        "overhead_bits": int(overhead_bits),
                        "gain_bits_est": int(cand.gain_bits_est),
                        "used_bits_after": int(used_bits),
                        "store_hash_before": str(store_hash_before),
                        "store_hash_after": str(store_hash_before),
                    }
                )
                _emit_promo_row(
                    {
                        "created_at": deterministic_iso(step=12_000 + int(step_ctr) + int(cand_idx)),
                        "candidate_id": str(act.id),
                        "certificate_sig": "",
                        "gain_bits_est": int(cand.gain_bits_est),
                        "overhead_bits": int(overhead_bits),
                        "decision": "skipped",
                        "reason": "budget_exceeded",
                        "store_hash_before": str(store_hash_before),
                        "store_hash_after": str(store_hash_before),
                    }
                )
                continue

            vectors, v_reason = _build_vector_specs_v84(
                store_base=store,
                act_candidate=act,
                cand=cand,
                traces_ok=traces_ok,
                kinds=kinds,
                seed=int(seed),
            )
            if vectors is None:
                _emit_event(
                    {
                        "round": int(r),
                        "event_kind": "promotion_attempt",
                        "promotion_index": int(promotion_index),
                        "goal_id": "",
                        "goal_sig": "",
                        "goal_kind": "",
                        "goal_created_step": 0,
                        "decision": "skipped",
                        "reason": str(v_reason),
                        "plan_sig": "",
                        "trace_sig": "",
                        "steps_total": 0,
                        "run_dir_sha256": "",
                        "candidate_id": str(act.id),
                        "certificate_sig": "",
                        "overhead_bits": int(overhead_bits),
                        "gain_bits_est": int(cand.gain_bits_est),
                        "used_bits_after": int(used_bits),
                        "store_hash_before": str(store_hash_before),
                        "store_hash_after": str(store_hash_before),
                    }
                )
                _emit_promo_row(
                    {
                        "created_at": deterministic_iso(step=13_000 + int(step_ctr) + int(cand_idx)),
                        "candidate_id": str(act.id),
                        "certificate_sig": "",
                        "gain_bits_est": int(cand.gain_bits_est),
                        "overhead_bits": int(overhead_bits),
                        "decision": "skipped",
                        "reason": str(v_reason),
                        "store_hash_before": str(store_hash_before),
                        "store_hash_after": str(store_hash_before),
                    }
                )
                continue

            mined_from = {
                "trace_sigs": [str(t.trace_sig()) for t in sorted(traces_ok, key=lambda t: str(t.trace_sig()))],
                "goal_kinds": [str(x) for x in sorted(set(str(t.goal_kind) for t in traces_ok), key=str)],
                "goal_kinds_distinct": int(len(set(str(t.goal_kind) for t in traces_ok))),
                "candidate": {"sub_sig": str(cand.sub_sig), "subpath": [str(x) for x in cand.subpath], "goal_kinds_supported": list(kinds)},
            }

            cert = build_certificate_v84(candidate_act=act, store_base=store, mined_from=mined_from, vector_specs=vectors, seed=int(seed))
            ok_pcc, reason_pcc, details_pcc = verify_pcc_v84(candidate_act=act, certificate=cert, store_base=store, seed=int(seed))
            cert_sig = str(cert.get("certificate_sig") or "")
            if not ok_pcc:
                _emit_event(
                    {
                        "round": int(r),
                        "event_kind": "promotion_attempt",
                        "promotion_index": int(promotion_index),
                        "goal_id": "",
                        "goal_sig": "",
                        "goal_kind": "",
                        "goal_created_step": 0,
                        "decision": "skipped",
                        "reason": f"pcc_fail:{reason_pcc}",
                        "plan_sig": "",
                        "trace_sig": "",
                        "steps_total": 0,
                        "run_dir_sha256": "",
                        "candidate_id": str(act.id),
                        "certificate_sig": str(cert_sig),
                        "overhead_bits": int(overhead_bits),
                        "gain_bits_est": int(cand.gain_bits_est),
                        "used_bits_after": int(used_bits),
                        "store_hash_before": str(store_hash_before),
                        "store_hash_after": str(store_hash_before),
                    }
                )
                _emit_promo_row(
                    {
                        "created_at": deterministic_iso(step=14_000 + int(step_ctr) + int(cand_idx)),
                        "candidate_id": str(act.id),
                        "certificate_sig": str(cert_sig),
                        "gain_bits_est": int(cand.gain_bits_est),
                        "overhead_bits": int(overhead_bits),
                        "decision": "skipped",
                        "reason": f"pcc_fail:{reason_pcc}",
                        "store_hash_before": str(store_hash_before),
                        "store_hash_after": str(store_hash_before),
                        "details": dict(details_pcc) if isinstance(details_pcc, dict) else {},
                    }
                )
                continue

            cand_k = int(promoted_total)
            act_path = os.path.join(candidates_dir, f"candidate_{cand_k:03d}_act.json")
            cert_path = os.path.join(candidates_dir, f"candidate_{cand_k:03d}_certificate_v2.json")
            _write_json_once(act_path, act.to_dict())
            _write_json_once(cert_path, cert)

            store_hash_before_add = str(store.content_hash())
            store.add(act)
            store_hash_after_add = str(store.content_hash())
            used_bits += int(overhead_bits)
            promoted_total += 1

            _emit_event(
                {
                    "round": int(r),
                    "event_kind": "promotion_attempt",
                    "promotion_index": int(promotion_index),
                    "goal_id": "",
                    "goal_sig": "",
                    "goal_kind": "",
                    "goal_created_step": 0,
                    "decision": "promoted",
                    "reason": "ok",
                    "plan_sig": "",
                    "trace_sig": "",
                    "steps_total": 0,
                    "run_dir_sha256": "",
                    "candidate_id": str(act.id),
                    "certificate_sig": str(cert_sig),
                    "overhead_bits": int(overhead_bits),
                    "gain_bits_est": int(cand.gain_bits_est),
                    "used_bits_after": int(used_bits),
                    "store_hash_before": str(store_hash_before_add),
                    "store_hash_after": str(store_hash_after_add),
                }
            )
            _emit_promo_row(
                {
                    "created_at": deterministic_iso(step=15_000 + int(step_ctr) + int(cand_idx)),
                    "candidate_id": str(act.id),
                    "certificate_sig": str(cert_sig),
                    "gain_bits_est": int(cand.gain_bits_est),
                    "overhead_bits": int(overhead_bits),
                    "decision": "promoted",
                    "reason": "ok",
                    "store_hash_before": str(store_hash_before_add),
                    "store_hash_after": str(store_hash_after_add),
                }
            )

            window_start_idx = int(len(traces))
            break

    traces_sorted = sorted(traces, key=lambda t: str(t.trace_sig()))
    _write_json_once(traces_path, {"schema_version": 1, "traces": [t.to_canonical_dict(include_sig=True) for t in traces_sorted]})
    points = _compression_points_from_events(events_buf)
    curve_core = {"schema_version": 1, "points": list(points), "curve_sig": _sha256_canon(points)}
    _write_json_once(curve_path, curve_core)

    store_hash_final = str(store.content_hash())
    return {
        "schema_version": 1,
        "seed": int(seed),
        "store_hash_init": str(store_hash_init),
        "store_hash_final": str(store_hash_final),
        "goals_total": int(len(list_goal_acts_v75(store))),
        "goals_satisfied": int(len([g for g in list_goal_acts_v75(store) if goal_v75_is_satisfied(g)])),
        "attempts_total": int(attempts_total),
        "traces_total": int(len(traces)),
        "promoted_total": int(promoted_total),
        "used_bits": int(used_bits),
        "budget_bits": int(promotion_budget_bits),
        "artifacts": {
            "goals_v84_events_jsonl_sha256": sha256_file(events_path),
            "traces_v84_json_sha256": sha256_file(traces_path),
            "mined_candidates_v84_json_sha256": sha256_file(mined_path) if os.path.exists(mined_path) else "",
            "v84_promotions_jsonl_sha256": sha256_file(promotions_path) if os.path.exists(promotions_path) else "",
            "compression_curve_json_sha256": sha256_file(curve_path),
        },
    }

