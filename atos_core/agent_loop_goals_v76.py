from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .agent_loop_v72 import run_goal_spec_v72
from .goal_act_v75 import goal_sig_v75, goal_v75_is_satisfied, goal_v75_update_from_run, list_goal_acts_v75
from .goal_spec_v72 import GoalSpecV72
from .mine_promote_v74 import (
    extract_rep_steps,
    materialize_composed_act_v74,
    mine_candidates_v74,
    mutate_bindings_plus1_numeric,
)
from .pcc_v74 import build_certificate_v2, verify_pcc_v2
from .store import ActStore
from .trace_v73 import TraceV73


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


def run_dir_sha256_v76(*, run_dir: str) -> str:
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


def _write_json(path: str, obj: Any) -> None:
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


def _find_subpath_start(path: Sequence[str], subpath: Sequence[str]) -> Optional[int]:
    p = [str(x) for x in path]
    sp = [str(x) for x in subpath]
    if not sp or len(sp) > len(p):
        return None
    for i in range(0, len(p) - len(sp) + 1):
        if p[i : i + len(sp)] == sp:
            return int(i)
    return None


def _execute_prefix_state(
    *,
    store_base: ActStore,
    trace: TraceV73,
    upto_idx: int,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Execute trace.steps[0:upto_idx] starting from trace.bindings to recover vars_state at subpath start.
    """
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


def _build_vector_specs_v76(
    *,
    store_base: ActStore,
    act_candidate,
    cand,
    traces_ok: Sequence[TraceV73],
    seed: int,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    Deterministic vector_specs builder (>=3):
      - >=2 vectors from distinct support contexts
      - +1 extra deterministic mutation
    """
    input_keys: List[str] = []
    if isinstance(act_candidate.evidence, dict):
        iface = act_candidate.evidence.get("interface") if isinstance(act_candidate.evidence.get("interface"), dict) else {}
        iface = iface if isinstance(iface, dict) else {}
        in_schema = iface.get("input_schema") if isinstance(iface.get("input_schema"), dict) else {}
        input_keys = [str(k) for k in sorted(in_schema.keys(), key=str)]

    support_set = set(str(x) for x in (cand.contexts or []) if str(x))
    support_traces = [t for t in traces_ok if str(t.context_id) in support_set]
    support_traces.sort(key=lambda t: str(t.trace_sig()))
    if len(support_traces) < 2:
        return None, "insufficient_support_traces"

    vector_specs: List[Dict[str, Any]] = []
    for st in support_traces:
        start = _find_subpath_start(st.acts_path(), cand.subpath)
        if start is None:
            continue
        state0 = _execute_prefix_state(store_base=store_base, trace=st, upto_idx=int(start), seed=int(seed))
        sub_steps = list(st.steps)[int(start) : int(start) + int(len(cand.subpath))]
        exp = _expected_for_steps(store_base=store_base, steps=sub_steps, start_state=state0, seed=int(seed))
        inputs = {k: state0.get(k) for k in input_keys}
        ctx_sig = _sha256_canon({"ctx": str(st.context_id), "sub_sig": str(cand.sub_sig), "inputs": inputs})
        vector_specs.append({"context_id": f"{st.context_id}:{ctx_sig}", "inputs": dict(inputs), "expected": str(exp)})

    vector_specs.sort(key=lambda v: _sha256_canon({"inputs": v.get("inputs", {}), "expected": v.get("expected")}))
    vector_specs = vector_specs[:3]

    base = support_traces[0]
    start0 = _find_subpath_start(base.acts_path(), cand.subpath) or 0
    mutated_bindings = mutate_bindings_plus1_numeric(bindings=dict(base.bindings), key_preference=["x", "y"])
    state_mut = _execute_prefix_state(
        store_base=store_base,
        trace=TraceV73(
            context_id=str(base.context_id),
            goal_sig=str(base.goal_sig),
            goal_id=str(base.goal_id),
            goal_kind=str(base.goal_kind),
            bindings=dict(mutated_bindings),
            output_key=str(base.output_key),
            expected=base.expected,
            validator_id=str(base.validator_id),
            steps=list(base.steps),
            outcome=dict(base.outcome),
            cost_units=dict(base.cost_units),
        ),
        upto_idx=int(start0),
        seed=int(seed),
    )
    sub_steps0 = list(base.steps)[int(start0) : int(start0) + int(len(cand.subpath))]
    exp_mut = _expected_for_steps(store_base=store_base, steps=sub_steps0, start_state=state_mut, seed=int(seed))
    inputs_mut = {k: state_mut.get(k) for k in input_keys}
    extra_ctx = _sha256_canon({"extra": True, "sub_sig": str(cand.sub_sig), "inputs": inputs_mut})
    vector_specs.append({"context_id": f"extra:{extra_ctx}", "inputs": dict(inputs_mut), "expected": str(exp_mut)})

    # Deduplicate by expected_sig (stable).
    uniq: Dict[str, Dict[str, Any]] = {}
    for vs in vector_specs:
        sig = _sha256_canon({"inputs": vs.get("inputs", {}), "expected": vs.get("expected")})
        if sig not in uniq:
            uniq[sig] = vs
    vector_specs = [uniq[k] for k in sorted(uniq.keys(), key=str)]
    if len(vector_specs) < 3:
        return None, "insufficient_vector_specs_after_dedup"

    return list(vector_specs), "ok"


def run_goals_v76(
    *,
    store: ActStore,
    seed: int,
    out_dir: str,
    max_rounds: int = 6,
    max_goals_per_round: int = 1,
    enable_promotion: bool = True,
    promotion_budget_bits: int = 1024,
    promotion_min_traces: int = 2,
    promotion_top_k: int = 8,
) -> Dict[str, Any]:
    """
    Online promotion inside the deterministic goal scheduler loop:
      goal attempts -> accumulate ok traces -> mine/promote (PCC v2 fail-closed) -> continue with updated store.
    """
    ensure_absent(out_dir)
    os.makedirs(out_dir, exist_ok=False)

    events_path = os.path.join(out_dir, "goals_v76_events.jsonl")
    ensure_absent(events_path)

    traces: List[TraceV73] = []
    attempts_total = 0
    promoted_total = 0
    used_bits = 0

    mined_path = os.path.join(out_dir, "mined_candidates_v76.json")
    candidates_dir = os.path.join(out_dir, "candidates")
    promo_dir = os.path.join(out_dir, "promotion")
    promos_path = os.path.join(promo_dir, "v76_promotions.jsonl")

    store_hash_init = str(store.content_hash())
    step_ctr = 0

    for r in range(0, int(max_rounds)):
        goals = list_goal_acts_v75(store)
        pending = [g for g in goals if not goal_v75_is_satisfied(g)]
        pending.sort(key=lambda a: str(getattr(a, "id", "")))

        if not pending:
            break

        to_run = pending[: int(max_goals_per_round)]
        skipped = pending[int(max_goals_per_round) :]

        for g in skipped:
            store_hash_before = str(store.content_hash())
            row = {
                "created_at": deterministic_iso(step=int(step_ctr)),
                "round": int(r),
                "event_kind": "goal_skipped",
                "goal_id": str(getattr(g, "id", "")),
                "goal_sig": str(goal_sig_v75(g)),
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
            _append_jsonl(events_path, row)
            step_ctr += 1

        for g in to_run:
            store_hash_before = str(store.content_hash())
            goal_spec, reason = goal_spec_v72_from_goal_act_v75(g)
            if goal_spec is None:
                row = {
                    "created_at": deterministic_iso(step=int(step_ctr)),
                    "round": int(r),
                    "event_kind": "goal_attempt",
                    "goal_id": str(getattr(g, "id", "")),
                    "goal_sig": str(goal_sig_v75(g)),
                    "decision": "active",
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
                _append_jsonl(events_path, row)
                step_ctr += 1
                continue

            gid_short = str(getattr(g, "id", ""))[:24]
            run_subdir = os.path.join(out_dir, f"round{int(r):02d}", f"goal_{gid_short}")
            ensure_absent(run_subdir)
            os.makedirs(run_subdir, exist_ok=False)

            res = run_goal_spec_v72(goal_spec=goal_spec, store=store, seed=int(seed), out_dir=run_subdir)
            attempts_total += 1

            plan = res.get("plan") if isinstance(res.get("plan"), dict) else {}
            plan = plan if isinstance(plan, dict) else {}
            raw_steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []

            # Build TraceV73 compatible object (minimal fields).
            from .trace_v73 import trace_from_agent_loop_v72

            tr = trace_from_agent_loop_v72(goal_spec=goal_spec, result=res)
            traces.append(tr)
            tr_sig = str(tr.trace_sig())

            run_sha = run_dir_sha256_v76(run_dir=run_subdir)
            updated = goal_v75_update_from_run(
                act=g,
                run_res=res,
                trace_sig=str(tr_sig),
                run_dir_sha256=str(run_sha),
                step=int(step_ctr),
            )
            store.add(updated)

            plan_sig = str(plan.get("plan_sig") or "")
            steps_total = int(len(raw_steps))
            satisfied = goal_v75_is_satisfied(updated)
            decision = "satisfied" if satisfied else "active"
            reason2 = ""
            if not satisfied:
                reason2 = str(res.get("reason") or "")
                if not reason2:
                    final = res.get("final") if isinstance(res.get("final"), dict) else {}
                    final = final if isinstance(final, dict) else {}
                    v = final.get("validator") if isinstance(final.get("validator"), dict) else {}
                    reason2 = str(v.get("reason") or "")

            store_hash_after = str(store.content_hash())
            row = {
                "created_at": deterministic_iso(step=int(step_ctr)),
                "round": int(r),
                "event_kind": "goal_attempt",
                "goal_id": str(updated.id),
                "goal_sig": str(goal_sig_v75(updated)),
                "decision": str(decision),
                "reason": str(reason2),
                "plan_sig": str(plan_sig),
                "trace_sig": str(tr_sig),
                "steps_total": int(steps_total),
                "run_dir_sha256": str(run_sha),
                "candidate_id": "",
                "certificate_sig": "",
                "overhead_bits": 0,
                "gain_bits_est": 0,
                "used_bits_after": int(used_bits),
                "store_hash_before": str(store_hash_before),
                "store_hash_after": str(store_hash_after),
            }
            _append_jsonl(events_path, row)
            step_ctr += 1

        # Online promotion (between rounds).
        if not bool(enable_promotion):
            continue
        if promoted_total >= 1:
            continue
        if int(used_bits) >= int(promotion_budget_bits):
            continue

        traces_ok = [t for t in traces if bool(t.outcome.get("ok", False))]
        if len(traces_ok) < int(promotion_min_traces):
            continue

        # Mine multi-candidates deterministically.
        mined, mined_debug = mine_candidates_v74(traces=traces_ok, max_k=6, min_support=2, top_k=int(promotion_top_k))
        if not os.path.exists(mined_path):
            _write_json(
                mined_path,
                {
                    "schema_version": 1,
                    "debug": dict(mined_debug),
                    "candidates": [c.to_dict() for c in mined],
                },
            )

        if not mined:
            # Still log a promotion_attempt for audit (skipped).
            store_hash_before = str(store.content_hash())
            row = {
                "created_at": deterministic_iso(step=int(step_ctr)),
                "round": int(r),
                "event_kind": "promotion_attempt",
                "goal_id": "",
                "goal_sig": "",
                "decision": "skipped",
                "reason": "no_mined_candidates",
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
            _append_jsonl(events_path, row)
            step_ctr += 1
            continue

        ensure_absent(candidates_dir) if not os.path.exists(candidates_dir) else None
        if not os.path.exists(candidates_dir):
            os.makedirs(candidates_dir, exist_ok=False)
        ensure_absent(promo_dir) if not os.path.exists(promo_dir) else None
        if not os.path.exists(promo_dir):
            os.makedirs(promo_dir, exist_ok=False)
        if not os.path.exists(promos_path):
            ensure_absent(promos_path)

        traces_by_sig = {str(t.trace_sig()): t for t in traces_ok}

        # Try candidates in mined order (already deterministically ranked).
        for idx, cand in enumerate(mined):
            store_hash_before = str(store.content_hash())

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
            overhead_bits = int((act.cost or {}).get("overhead_bits", 1024) or 1024)

            vector_specs, v_reason = _build_vector_specs_v76(
                store_base=store,
                act_candidate=act,
                cand=cand,
                traces_ok=traces_ok,
                seed=int(seed),
            )
            if vector_specs is None:
                row = {
                    "created_at": deterministic_iso(step=int(step_ctr)),
                    "round": int(r),
                    "event_kind": "promotion_attempt",
                    "goal_id": "",
                    "goal_sig": "",
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
                _append_jsonl(events_path, row)
                _append_jsonl(
                    promos_path,
                    {
                        "created_at": deterministic_iso(step=1000 + idx),
                        "candidate_id": str(act.id),
                        "certificate_sig": "",
                        "gain_bits_est": int(cand.gain_bits_est),
                        "overhead_bits": int(overhead_bits),
                        "decision": "skipped",
                        "reason": str(v_reason),
                        "store_hash_before": str(store_hash_before),
                        "store_hash_after": str(store_hash_before),
                    },
                )
                step_ctr += 1
                continue

            mined_from = {
                "trace_sigs": [str(t.trace_sig()) for t in sorted(traces_ok, key=lambda t: str(t.trace_sig()))],
                "contexts": [str(x) for x in sorted(set(str(t.context_id) for t in traces_ok), key=str)],
                "contexts_distinct": int(len(set(str(t.context_id) for t in traces_ok))),
                "candidate": {"sub_sig": str(cand.sub_sig), "subpath": [str(x) for x in cand.subpath]},
            }

            cert = build_certificate_v2(candidate_act=act, store_base=store, mined_from=mined_from, vector_specs=vector_specs, seed=int(seed))
            ok_pcc, reason_pcc, details_pcc = verify_pcc_v2(candidate_act=act, certificate=cert, store_base=store, seed=int(seed))
            if not ok_pcc:
                row = {
                    "created_at": deterministic_iso(step=int(step_ctr)),
                    "round": int(r),
                    "event_kind": "promotion_attempt",
                    "goal_id": "",
                    "goal_sig": "",
                    "decision": "skipped",
                    "reason": f"pcc_fail:{reason_pcc}",
                    "plan_sig": "",
                    "trace_sig": "",
                    "steps_total": 0,
                    "run_dir_sha256": "",
                    "candidate_id": str(act.id),
                    "certificate_sig": str(cert.get("certificate_sig") or ""),
                    "overhead_bits": int(overhead_bits),
                    "gain_bits_est": int(cand.gain_bits_est),
                    "used_bits_after": int(used_bits),
                    "store_hash_before": str(store_hash_before),
                    "store_hash_after": str(store_hash_before),
                }
                _append_jsonl(events_path, row)
                _append_jsonl(
                    promos_path,
                    {
                        "created_at": deterministic_iso(step=1100 + idx),
                        "candidate_id": str(act.id),
                        "certificate_sig": str(cert.get("certificate_sig") or ""),
                        "gain_bits_est": int(cand.gain_bits_est),
                        "overhead_bits": int(overhead_bits),
                        "decision": "skipped",
                        "reason": f"pcc_fail:{reason_pcc}",
                        "store_hash_before": str(store_hash_before),
                        "store_hash_after": str(store_hash_before),
                    },
                )
                step_ctr += 1
                continue

            # Persist candidate_000 artifacts (audit-first).
            if idx == 0:
                act_path = os.path.join(candidates_dir, "candidate_000_act.json")
                cert_path = os.path.join(candidates_dir, "candidate_000_certificate_v2.json")
                if not os.path.exists(act_path):
                    _write_json(act_path, act.to_dict())
                if not os.path.exists(cert_path):
                    _write_json(cert_path, cert)

            cert_sig = str(cert.get("certificate_sig") or "")
            decision = "skipped"
            reason = ""
            store_hash_after = store_hash_before
            used_after = int(used_bits)

            if int(used_bits) + int(overhead_bits) > int(promotion_budget_bits):
                decision = "skipped"
                reason = "budget_exceeded"
            else:
                decision = "promoted"
                reason = "pcc_ok_under_budget"
                store.add(act)
                used_bits = int(used_bits) + int(overhead_bits)
                used_after = int(used_bits)
                promoted_total += 1
                store_hash_after = str(store.content_hash())

            row = {
                "created_at": deterministic_iso(step=int(step_ctr)),
                "round": int(r),
                "event_kind": "promotion_attempt",
                "goal_id": "",
                "goal_sig": "",
                "decision": str(decision),
                "reason": str(reason),
                "plan_sig": "",
                "trace_sig": "",
                "steps_total": 0,
                "run_dir_sha256": "",
                "candidate_id": str(act.id),
                "certificate_sig": str(cert_sig),
                "overhead_bits": int(overhead_bits),
                "gain_bits_est": int(cand.gain_bits_est),
                "used_bits_after": int(used_after),
                "store_hash_before": str(store_hash_before),
                "store_hash_after": str(store_hash_after),
            }
            _append_jsonl(events_path, row)
            _append_jsonl(
                promos_path,
                {
                    "created_at": deterministic_iso(step=1200 + idx),
                    "candidate_id": str(act.id),
                    "certificate_sig": str(cert_sig),
                    "gain_bits_est": int(cand.gain_bits_est),
                    "overhead_bits": int(overhead_bits),
                    "decision": str(decision),
                    "reason": str(reason),
                    "store_hash_before": str(store_hash_before),
                    "store_hash_after": str(store_hash_after),
                },
            )
            step_ctr += 1

            if decision == "promoted":
                break

            # Stop after first successfully verified candidate is evaluated for MVP determinism.
            if idx == 0 and promoted_total >= 1:
                break

        # Stop attempting further promotions (MVP: at most one per run).
        if promoted_total >= 1:
            continue

    # Persist traces for mining/replay.
    traces_sorted = sorted([t.to_canonical_dict(include_sig=True) for t in traces], key=lambda d: str(d.get("trace_sig") or ""))
    traces_path = os.path.join(out_dir, "traces_v76.json")
    ensure_absent(traces_path)
    _write_json(traces_path, {"schema_version": 1, "traces": list(traces_sorted)})

    # Collect artifact hashes (only if files exist).
    artifacts: Dict[str, Any] = {
        "goals_v76_events_jsonl_sha256": sha256_file(events_path),
        "traces_v76_json_sha256": sha256_file(traces_path),
    }
    if os.path.exists(mined_path):
        artifacts["mined_candidates_v76_json_sha256"] = sha256_file(mined_path)
    if os.path.exists(promos_path):
        artifacts["v76_promotions_jsonl_sha256"] = sha256_file(promos_path)

    return {
        "schema_version": 1,
        "seed": int(seed),
        "store_hash_init": str(store_hash_init),
        "store_hash_final": str(store.content_hash()),
        "goals_total": int(len(list_goal_acts_v75(store))),
        "goals_satisfied": int(len([g for g in list_goal_acts_v75(store) if goal_v75_is_satisfied(g)])),
        "attempts_total": int(attempts_total),
        "traces_total": int(len(traces_sorted)),
        "promoted_total": int(promoted_total),
        "budget_bits": int(promotion_budget_bits),
        "used_bits": int(used_bits),
        "artifacts": dict(artifacts),
        "__internal_traces": list(traces),
    }

