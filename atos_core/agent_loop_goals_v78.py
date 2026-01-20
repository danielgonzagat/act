from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import Act, canonical_json_dumps, deterministic_iso, sha256_hex
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


def _is_act_allowed_for_goal_kind(*, act: Act, goal_kind: str) -> bool:
    """
    Explicit (non-heuristic) filtering based on Act.match metadata.

    If act.match contains {"goal_kinds": [...]}, allow only when goal_kind is present.
    Missing/invalid match means "global" (allowed).
    """
    if act is None:
        return False
    mk = act.match if isinstance(getattr(act, "match", None), dict) else {}
    gks = mk.get("goal_kinds")
    if not isinstance(gks, list):
        return True
    allowed = [str(x) for x in gks if str(x)]
    return str(goal_kind or "") in set(allowed)


def _store_view_for_goal_kind(*, store: ActStore, goal_kind: str) -> ActStore:
    """
    Build a deterministic store view where concept_csv acts can be gated by match.goal_kinds.
    This avoids hidden heuristics: selection is driven solely by explicit metadata.
    """
    view = ActStore()
    for act_id in sorted((store.acts or {}).keys(), key=str):
        act = store.acts.get(act_id)
        if act is None:
            continue
        snap = Act.from_dict(act.to_dict())
        if str(getattr(snap, "kind", "")) == "concept_csv":
            if not _is_act_allowed_for_goal_kind(act=snap, goal_kind=str(goal_kind or "")):
                snap.active = False
        view.add(snap)
    view.next_id_int = int(getattr(store, "next_id_int", 1) or 1)
    return view


def run_dir_sha256_v78(*, run_dir: str) -> str:
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


def _build_vector_specs_v78(
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


def _compression_points_from_events(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    idxs: List[int] = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        if str(r.get("event_kind") or "") == "promotion_attempt" and str(r.get("decision") or "") == "promoted":
            idxs.append(int(i))

    for i, ev_idx in enumerate(idxs):
        # steps_before = last goal_attempt before promotion
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


def run_goals_v78(
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
    V78: deterministic goal loop with multi online promotions + curriculum scheduler (created_step ASC),
    and explicit cross-goal_kind transfer gating for the first promotion.
    """
    ensure_absent(out_dir)
    os.makedirs(out_dir, exist_ok=False)

    events_path = os.path.join(out_dir, "goals_v78_events.jsonl")
    ensure_absent(events_path)

    traces: List[TraceV73] = []
    attempts_total = 0
    promoted_total = 0
    used_bits = 0

    promotions_dir = os.path.join(out_dir, "promotion")
    promotions_path = os.path.join(promotions_dir, "v78_promotions.jsonl")
    candidates_dir = os.path.join(out_dir, "candidates")
    traces_path = os.path.join(out_dir, "traces_v78.json")
    mined_path = os.path.join(out_dir, "mined_candidates_v78.json")
    curve_path = os.path.join(out_dir, "compression_curve.json")

    store_hash_init = str(store.content_hash())
    step_ctr = 0
    events_buf: List[Dict[str, Any]] = []
    mining_attempts: List[Dict[str, Any]] = []

    # Trace window for hierarchical mining: only mine on traces since last promotion.
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
                )
                continue

            gid_short = str(getattr(g, "id", ""))[:24]
            run_subdir = os.path.join(out_dir, f"round{int(r):02d}", f"goal_{gid_short}")
            ensure_absent(run_subdir)
            os.makedirs(run_subdir, exist_ok=False)

            store_view = _store_view_for_goal_kind(store=store, goal_kind=str(goal_kind))
            res = run_goal_spec_v72(goal_spec=goal_spec, store=store_view, seed=int(seed), out_dir=run_subdir)
            attempts_total += 1

            tr = trace_from_agent_loop_v72(goal_spec=goal_spec, result=res)
            traces.append(tr)
            tr_sig = str(tr.trace_sig())

            run_sha = run_dir_sha256_v78(run_dir=run_subdir)
            updated = goal_v75_update_from_run(
                act=g,
                run_res=res,
                trace_sig=str(tr_sig),
                run_dir_sha256=str(run_sha),
                step=int(step_ctr),
            )
            store.add(updated)

            plan = res.get("plan") if isinstance(res.get("plan"), dict) else {}
            plan = plan if isinstance(plan, dict) else {}
            raw_steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
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
            _emit_event(
                {
                    "round": int(r),
                    "event_kind": "goal_attempt",
                    "promotion_index": -1,
                    "goal_id": str(updated.id),
                    "goal_sig": str(goal_sig_v75(updated)),
                    "goal_kind": str(goal_kind),
                    "goal_created_step": int(goal_step),
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
            )

        # Online promotion between rounds (at most one promotion per round, but multiple per run).
        if not bool(enable_promotion):
            continue
        if int(promoted_total) >= int(max_promotions_per_run):
            continue
        if int(used_bits) >= int(promotion_budget_bits):
            continue

        # Hierarchical mining window.
        window_traces = traces[int(window_start_idx) :]
        traces_ok = [t for t in window_traces if bool(t.outcome.get("ok", False))]
        if len(traces_ok) < int(promotion_min_traces):
            continue

        mined, mined_debug = mine_candidates_v74(traces=traces_ok, max_k=3, min_support=2, top_k=int(promotion_top_k))

        # Log candidate goal_kinds_supported for audit.
        mined_logged: List[Dict[str, Any]] = []
        for cand in mined:
            kinds = _candidate_goal_kinds_supported(cand, traces_ok)
            d = cand.to_dict()
            d["goal_kinds_supported"] = list(kinds)
            d["goal_kinds_supported_count"] = int(len(kinds))
            mined_logged.append(d)

        mining_attempts.append(
            {
                "promotion_index": int(promoted_total),
                "round": int(r),
                "required_goal_kind_diversity": int(promotion_kind_diversity_min if int(promoted_total) == 0 else 1),
                "window": {
                    "trace_sigs": [str(t.trace_sig()) for t in sorted(traces_ok, key=lambda t: str(t.trace_sig()))],
                    "goal_kinds": [str(x) for x in sorted(set(str(t.goal_kind) for t in traces_ok), key=str)],
                    "goal_kinds_distinct": int(len(set(str(t.goal_kind) for t in traces_ok))),
                },
                "debug": dict(mined_debug),
                "candidates": list(mined_logged),
            }
        )

        if not mined:
            store_hash_before = str(store.content_hash())
            _emit_event(
                {
                    "round": int(r),
                    "event_kind": "promotion_attempt",
                    "promotion_index": int(promoted_total),
                    "goal_id": "",
                    "goal_sig": "",
                    "goal_kind": "",
                    "goal_created_step": 0,
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
            )
            continue

        # Prepare dirs for promotion artifacts.
        if not os.path.exists(candidates_dir):
            ensure_absent(candidates_dir)
            os.makedirs(candidates_dir, exist_ok=False)
        if not os.path.exists(promotions_dir):
            ensure_absent(promotions_dir)
            os.makedirs(promotions_dir, exist_ok=False)

        traces_by_sig = {str(t.trace_sig()): t for t in traces_ok}

        # Candidate loop (deterministic mined order).
        for cand_idx, cand in enumerate(mined):
            promotion_index = int(promoted_total)
            store_hash_before = str(store.content_hash())

            kinds = _candidate_goal_kinds_supported(cand, traces_ok)
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

            # Restrict act applicability by explicit goal_kind metadata (audit-first).
            if kinds:
                act.match = {"goal_kinds": list(kinds)}
            overhead_bits = int((act.cost or {}).get("overhead_bits", 1024) or 1024)

            # Avoid re-promoting identical act.id.
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

            vector_specs, v_reason = _build_vector_specs_v78(
                store_base=store,
                act_candidate=act,
                cand=cand,
                traces_ok=traces_ok,
                seed=int(seed),
            )
            if vector_specs is None:
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
            cert = build_certificate_v2(candidate_act=act, store_base=store, mined_from=mined_from, vector_specs=vector_specs, seed=int(seed))
            ok_pcc, reason_pcc, details_pcc = verify_pcc_v2(candidate_act=act, certificate=cert, store_base=store, seed=int(seed))
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
                        "certificate_sig": str(cert.get("certificate_sig") or ""),
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
                        "certificate_sig": str(cert.get("certificate_sig") or ""),
                        "gain_bits_est": int(cand.gain_bits_est),
                        "overhead_bits": int(overhead_bits),
                        "decision": "skipped",
                        "reason": f"pcc_fail:{reason_pcc}",
                        "store_hash_before": str(store_hash_before),
                        "store_hash_after": str(store_hash_before),
                    }
                )
                continue

            # Promote.
            cert_sig = str(cert.get("certificate_sig") or "")
            store.add(act)
            used_bits = int(used_bits) + int(overhead_bits)
            store_hash_after = str(store.content_hash())

            # Persist candidate artifacts for this promotion index (write-once).
            act_path = os.path.join(candidates_dir, f"candidate_{promotion_index:03d}_act.json")
            cert_path = os.path.join(candidates_dir, f"candidate_{promotion_index:03d}_certificate_v2.json")
            _write_json_once(act_path, act.to_dict())
            _write_json_once(cert_path, cert)

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
                    "reason": "pcc_ok_under_budget",
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
                    "store_hash_after": str(store_hash_after),
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
                    "reason": "pcc_ok_under_budget",
                    "store_hash_before": str(store_hash_before),
                    "store_hash_after": str(store_hash_after),
                    "goal_kinds_supported": list(kinds),
                }
            )

            promoted_total += 1
            window_start_idx = int(len(traces))
            break

        # end candidate loop

    # Persist traces / mined_candidates / curve (write-once).
    traces_sorted = sorted([t.to_canonical_dict(include_sig=True) for t in traces], key=lambda d: str(d.get("trace_sig") or ""))
    _write_json_once(traces_path, {"schema_version": 1, "traces": list(traces_sorted)})
    _write_json_once(mined_path, {"schema_version": 1, "mining_attempts": list(mining_attempts)})

    points = _compression_points_from_events(events_buf)
    curve_core = {"schema_version": 1, "points": list(points)}
    curve_core["curve_sig"] = _sha256_canon({"points": list(points)})
    _write_json_once(curve_path, curve_core)

    # Collect artifact hashes.
    artifacts: Dict[str, Any] = {
        "goals_v78_events_jsonl_sha256": sha256_file(events_path),
        "traces_v78_json_sha256": sha256_file(traces_path),
        "mined_candidates_v78_json_sha256": sha256_file(mined_path),
        "compression_curve_json_sha256": sha256_file(curve_path),
    }
    if os.path.exists(promotions_path):
        artifacts["v78_promotions_jsonl_sha256"] = sha256_file(promotions_path)
    for k in range(0, int(promoted_total)):
        ap = os.path.join(candidates_dir, f"candidate_{k:03d}_act.json")
        cp = os.path.join(candidates_dir, f"candidate_{k:03d}_certificate_v2.json")
        if os.path.exists(ap):
            artifacts[f"candidate_{k:03d}_act_json_sha256"] = sha256_file(ap)
        if os.path.exists(cp):
            artifacts[f"candidate_{k:03d}_certificate_v2_json_sha256"] = sha256_file(cp)

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
        "promotion_kind_diversity_min": int(promotion_kind_diversity_min),
        "compression_curve": dict(curve_core),
        "artifacts": dict(artifacts),
        "__internal_traces": list(traces),
    }
