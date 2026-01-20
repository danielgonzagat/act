#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.agent_loop_v72 import run_goal_spec_v72
from atos_core.goal_spec_v72 import GoalSpecV72
from atos_core.mine_promote_v74 import (
    extract_rep_steps,
    materialize_composed_act_v74,
    mine_candidates_v74,
    mutate_bindings_plus1_numeric,
)
from atos_core.pcc_v74 import build_certificate_v2, verify_pcc_v2
from atos_core.store import ActStore
from atos_core.trace_v73 import TraceV73, trace_from_agent_loop_v72


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()


def sha256_canon(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"ERROR: path already exists: {path}")


def write_json(path: str, obj: Any) -> str:
    ensure_absent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)
    return sha256_file(path)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    ensure_absent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(canonical_json_dumps(r))
            f.write("\n")
    os.replace(tmp, path)
    return sha256_file(path)


def make_concept_act(
    *,
    act_id: str,
    input_schema: Dict[str, str],
    output_schema: Dict[str, str],
    validator_id: str,
    program: List[Instruction],
) -> Act:
    return Act(
        id=str(act_id),
        version=1,
        created_at=deterministic_iso(step=0),
        kind="concept_csv",
        match={},
        program=list(program),
        evidence={
            "interface": {
                "input_schema": dict(input_schema),
                "output_schema": dict(output_schema),
                "validator_id": str(validator_id),
            }
        },
        cost={},
        deps=[],
        active=True,
    )


def _eval_from_run(res: Dict[str, Any]) -> Dict[str, Any]:
    plan = res.get("plan") if isinstance(res.get("plan"), dict) else {}
    plan = plan if isinstance(plan, dict) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    final = res.get("final") if isinstance(res.get("final"), dict) else {}
    final = final if isinstance(final, dict) else {}
    graph = res.get("graph") if isinstance(res.get("graph"), dict) else {}
    graph = graph if isinstance(graph, dict) else {}
    chains = graph.get("chains") if isinstance(graph.get("chains"), dict) else {}
    return {
        "ok": bool(res.get("ok", False)),
        "plan_sig": str(plan.get("plan_sig") or ""),
        "steps_total": int(len(steps)),
        "got": str(final.get("got") or ""),
        "expected": final.get("expected"),
        "graph_sig": str(graph.get("graph_sig") or ""),
        "chains": dict(chains) if isinstance(chains, dict) else {},
    }


def _trace_json(tr: TraceV73) -> Dict[str, Any]:
    return tr.to_canonical_dict(include_sig=True)


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
    from atos_core.mine_promote_v74 import execute_steps_expected_output

    if int(upto_idx) <= 0:
        return dict(trace.bindings)

    # Reconstruct state by repeatedly executing prefixes, keeping a running env.
    # We reuse execute_steps_expected_output semantics but need the intermediate env, so do it here.
    from atos_core.engine import Engine, EngineConfig

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
    from atos_core.mine_promote_v74 import execute_steps_expected_output

    return execute_steps_expected_output(store_base=store_base, steps=steps, bindings=start_state, seed=int(seed))


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    store = ActStore()

    # Base micro-world: same semantics as V73/V72.
    normalize_x_id = "concept_v72_normalize_x_v0"
    normalize_y_id = "concept_v72_normalize_y_v0"
    add_nx_ny_id = "concept_v72_add_nx_ny_v0"

    store.add(
        make_concept_act(
            act_id=normalize_x_id,
            input_schema={"x": "str"},
            output_schema={"nx": "str"},
            validator_id="text_exact",
            program=[
                Instruction("CSV_GET_INPUT", {"name": "x", "out": "x"}),
                Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["x"], "out": "d"}),
                Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "i"}),
                Instruction("CSV_PRIMITIVE", {"fn": "int_to_digits", "in": ["i"], "out": "nx"}),
                Instruction("CSV_RETURN", {"var": "nx"}),
            ],
        )
    )
    store.add(
        make_concept_act(
            act_id=normalize_y_id,
            input_schema={"y": "str"},
            output_schema={"ny": "str"},
            validator_id="text_exact",
            program=[
                Instruction("CSV_GET_INPUT", {"name": "y", "out": "y"}),
                Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["y"], "out": "d"}),
                Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "i"}),
                Instruction("CSV_PRIMITIVE", {"fn": "int_to_digits", "in": ["i"], "out": "ny"}),
                Instruction("CSV_RETURN", {"var": "ny"}),
            ],
        )
    )
    store.add(
        make_concept_act(
            act_id=add_nx_ny_id,
            input_schema={"nx": "str", "ny": "str"},
            output_schema={"sum": "str"},
            validator_id="text_exact",
            program=[
                Instruction("CSV_GET_INPUT", {"name": "nx", "out": "nx"}),
                Instruction("CSV_GET_INPUT", {"name": "ny", "out": "ny"}),
                Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["nx"], "out": "dx"}),
                Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["ny"], "out": "dy"}),
                Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["dx"], "out": "ix"}),
                Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["dy"], "out": "iy"}),
                Instruction("CSV_PRIMITIVE", {"fn": "add_int", "in": ["ix", "iy"], "out": "sum_i"}),
                Instruction("CSV_PRIMITIVE", {"fn": "int_to_digits", "in": ["sum_i"], "out": "sum"}),
                Instruction("CSV_RETURN", {"var": "sum"}),
            ],
        )
    )

    store_hash_base = store.content_hash()

    # Generate >=3 contexts.
    goals = [
        GoalSpecV72(goal_kind="v74_sum_norm", bindings={"x": "0004", "y": "0008"}, output_key="sum", expected="12", validator_id="text_exact", created_step=0),
        GoalSpecV72(goal_kind="v74_sum_norm", bindings={"x": "0010", "y": "0003"}, output_key="sum", expected="13", validator_id="text_exact", created_step=0),
        GoalSpecV72(goal_kind="v74_sum_norm", bindings={"x": "0007", "y": "0005"}, output_key="sum", expected="12", validator_id="text_exact", created_step=0),
    ]

    before_evals: Dict[str, Any] = {}
    traces: List[TraceV73] = []

    for i, g in enumerate(goals):
        gdir = os.path.join(out_dir, f"goal{i+1}_before")
        ensure_absent(gdir)
        os.makedirs(gdir, exist_ok=False)
        res = run_goal_spec_v72(goal_spec=g, store=store, seed=int(seed), out_dir=gdir)
        if not bool(res.get("ok", False)):
            _fail(f"ERROR: goal{i+1}_before not ok")
        before_evals[f"goal{i+1}"] = _eval_from_run(res)
        traces.append(trace_from_agent_loop_v72(goal_spec=g, result=res))

    steps_before = int(before_evals.get("goal1", {}).get("steps_total", 0) or 0)
    if steps_before < 2:
        _fail(f"ERROR: expected >=2 steps before mining, got={steps_before}")

    # Persist traces.
    traces_sorted = sorted([_trace_json(t) for t in traces], key=lambda d: str(d.get("trace_sig") or ""))
    traces_path = os.path.join(out_dir, "traces_v74.json")
    traces_sha = write_json(traces_path, {"schema_version": 1, "traces": list(traces_sorted)})

    # Mine multi-candidates.
    mined, mined_debug = mine_candidates_v74(traces=traces, max_k=6, min_support=2, top_k=8)
    mined_path = os.path.join(out_dir, "mined_candidates_v74.json")
    mined_sha = write_json(
        mined_path,
        {
            "schema_version": 1,
            "debug": dict(mined_debug),
            "candidates": [c.to_dict() for c in mined],
        },
    )
    if not mined:
        _fail("ERROR: expected mined candidates >=1")

    # Promotion under budget, PCC v2 fail-closed.
    promo_dir = os.path.join(out_dir, "promotion")
    ensure_absent(promo_dir)
    os.makedirs(promo_dir, exist_ok=False)
    decisions_path = os.path.join(promo_dir, "v74_promotions.jsonl")

    budget_bits = 1024  # promote at most one concept in this smoke (demonstrates budgeted skipping deterministically)
    used_bits = 0

    traces_by_sig = {str(t.trace_sig()): t for t in traces}
    promoted: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    candidate_artifacts: List[Dict[str, Any]] = []

    for idx, cand in enumerate(mined):
        store_hash_before = str(store.content_hash())

        rep_steps = extract_rep_steps(
            traces_by_sig=traces_by_sig,
            rep_trace_sig=str(cand.rep_trace_sig),
            start_idx=int(cand.start_idx),
            subpath_len=int(len(cand.subpath)),
        )
        act, act_dbg = materialize_composed_act_v74(
            store_base=store,
            steps=rep_steps,
            support_contexts=int(cand.support_contexts),
            contexts=list(cand.contexts),
            seed_step=0,
        )
        overhead_bits = int((act.cost or {}).get("overhead_bits", 1024) or 1024)

        # Build vector_specs:
        # - >=2 from support traces
        # - +1 extra deterministic mutation
        support_traces = [t for t in traces if str(t.context_id) in set(cand.contexts)]
        support_traces.sort(key=lambda t: str(t.trace_sig()))
        if len(support_traces) < 2:
            decision = {"decision": "skipped", "reason": "insufficient_support_traces"}
            skipped.append({"candidate_id": str(act.id), **decision})
            decision_rows.append(
                {
                    "created_at": deterministic_iso(step=100 + idx),
                    "candidate_id": str(act.id),
                    "certificate_sig": "",
                    "gain_bits_est": int(cand.gain_bits_est),
                    "overhead_bits": int(overhead_bits),
                    "decision": "skipped",
                    "reason": str(decision["reason"]),
                    "store_hash_before": str(store_hash_before),
                    "store_hash_after": str(store_hash_before),
                }
            )
            continue

        input_keys = []
        if isinstance(act.evidence, dict):
            iface = act.evidence.get("interface") if isinstance(act.evidence.get("interface"), dict) else {}
            iface = iface if isinstance(iface, dict) else {}
            in_schema = iface.get("input_schema") if isinstance(iface.get("input_schema"), dict) else {}
            input_keys = [str(k) for k in sorted(in_schema.keys(), key=str)]

        vector_specs: List[Dict[str, Any]] = []
        for st in support_traces:
            start = _find_subpath_start(st.acts_path(), cand.subpath)
            if start is None:
                continue
            state0 = _execute_prefix_state(store_base=store, trace=st, upto_idx=int(start), seed=int(seed))
            sub_steps = list(st.steps)[int(start) : int(start) + int(len(cand.subpath))]
            exp = _expected_for_steps(store_base=store, steps=sub_steps, start_state=state0, seed=int(seed))
            inputs = {k: state0.get(k) for k in input_keys}
            ctx_sig = sha256_canon({"ctx": str(st.context_id), "sub_sig": str(cand.sub_sig), "inputs": inputs})
            vector_specs.append({"context_id": f"{st.context_id}:{ctx_sig}", "inputs": dict(inputs), "expected": str(exp)})

        vector_specs.sort(key=lambda v: sha256_canon({"inputs": v.get("inputs", {}), "expected": v.get("expected")}))
        vector_specs = vector_specs[:3]  # deterministic cap for performance (still >=2)

        # Extra deterministic mutation based on the first support trace.
        base = support_traces[0]
        start0 = _find_subpath_start(base.acts_path(), cand.subpath) or 0
        mutated_bindings = mutate_bindings_plus1_numeric(bindings=dict(base.bindings), key_preference=["x", "y"])
        state_mut = _execute_prefix_state(
            store_base=store,
            trace=TraceV73(
                context_id=str(base.context_id),
                goal_sig=str(base.goal_sig),
                goal_id=str(base.goal_id),
                goal_kind=str(base.goal_kind),
                bindings=dict(mutated_bindings),
                output_key=str(base.output_key),
                expected=str(base.expected),
                validator_id=str(base.validator_id),
                steps=list(base.steps),
                outcome=dict(base.outcome),
                cost_units=dict(base.cost_units),
            ),
            upto_idx=int(start0),
            seed=int(seed),
        )
        sub_steps0 = list(base.steps)[int(start0) : int(start0) + int(len(cand.subpath))]
        exp_mut = _expected_for_steps(store_base=store, steps=sub_steps0, start_state=state_mut, seed=int(seed))
        inputs_mut = {k: state_mut.get(k) for k in input_keys}
        extra_ctx = sha256_canon({"extra": True, "sub_sig": str(cand.sub_sig), "inputs": inputs_mut})
        vector_specs.append({"context_id": f"extra:{extra_ctx}", "inputs": dict(inputs_mut), "expected": str(exp_mut)})

        # Deduplicate by expected_sig (stable).
        uniq: Dict[str, Dict[str, Any]] = {}
        for vs in vector_specs:
            sig = sha256_canon({"inputs": vs.get("inputs", {}), "expected": vs.get("expected")})
            if sig not in uniq:
                uniq[sig] = vs
        vector_specs = [uniq[k] for k in sorted(uniq.keys(), key=str)]
        if len(vector_specs) < 3:
            _fail("ERROR: insufficient vector_specs after dedup (need >=3)")

        mined_from = {
            "trace_sigs": [str(t.trace_sig()) for t in sorted(traces, key=lambda t: str(t.trace_sig()))],
            "contexts": [str(x) for x in sorted(set(str(t.context_id) for t in traces), key=str)],
            "contexts_distinct": int(len(set(str(t.context_id) for t in traces))),
            "candidate": {"sub_sig": str(cand.sub_sig), "subpath": [str(x) for x in cand.subpath]},
        }

        cert = build_certificate_v2(candidate_act=act, store_base=store, mined_from=mined_from, vector_specs=vector_specs, seed=int(seed))
        ok_pcc, reason_pcc, details_pcc = verify_pcc_v2(candidate_act=act, certificate=cert, store_base=store, seed=int(seed))
        if not ok_pcc:
            _fail(f"ERROR: PCC v2 verify failed: {reason_pcc}: {details_pcc}")

        # Persist candidate artifacts WORM.
        cand_dir = os.path.join(out_dir, "candidates")
        if not os.path.exists(cand_dir):
            os.makedirs(cand_dir, exist_ok=False)
        act_path = os.path.join(cand_dir, f"candidate_{idx:03d}_act.json")
        cert_path = os.path.join(cand_dir, f"candidate_{idx:03d}_certificate_v2.json")
        act_sha = write_json(act_path, act.to_dict())
        cert_sha = write_json(cert_path, cert)
        candidate_artifacts.append(
            {
                "idx": int(idx),
                "candidate_id": str(act.id),
                "gain_bits_est": int(cand.gain_bits_est),
                "overhead_bits": int(overhead_bits),
                "act_sha256": str(act_sha),
                "certificate_sha256": str(cert_sha),
                "certificate_sig": str(cert.get("certificate_sig") or ""),
            }
        )

        decision = "skipped"
        reason = ""
        cert_sig = str(cert.get("certificate_sig") or "")
        if int(used_bits) + int(overhead_bits) > int(budget_bits):
            decision = "skipped"
            reason = "budget_exceeded"
        else:
            decision = "promoted"
            reason = "pcc_ok_under_budget"
            store.add(act)
            used_bits += int(overhead_bits)
            promoted.append({"candidate_id": str(act.id), "certificate_sig": cert_sig, "gain_bits_est": int(cand.gain_bits_est)})

        store_hash_after = str(store.content_hash())
        if decision != "promoted":
            store_hash_after = store_hash_before
            skipped.append({"candidate_id": str(act.id), "certificate_sig": cert_sig, "reason": str(reason)})

        decision_rows.append(
            {
                "created_at": deterministic_iso(step=200 + idx),
                "candidate_id": str(act.id),
                "certificate_sig": str(cert_sig),
                "gain_bits_est": int(cand.gain_bits_est),
                "overhead_bits": int(overhead_bits),
                "decision": str(decision),
                "reason": str(reason),
                "store_hash_before": str(store_hash_before),
                "store_hash_after": str(store_hash_after),
            }
        )

    if not promoted:
        _fail("ERROR: expected at least 1 promoted candidate")

    promotions_sha = write_jsonl(decisions_path, decision_rows)

    # Re-run goal1 to prove plan compression.
    after_dir = os.path.join(out_dir, "goal1_after")
    ensure_absent(after_dir)
    os.makedirs(after_dir, exist_ok=False)
    res_after = run_goal_spec_v72(goal_spec=goals[0], store=store, seed=int(seed), out_dir=after_dir)
    if not bool(res_after.get("ok", False)):
        _fail("ERROR: goal1_after not ok")
    after_eval = _eval_from_run(res_after)

    steps_after = int(after_eval.get("steps_total", 0) or 0)
    if steps_after >= steps_before:
        _fail(f"ERROR: expected plan compression: before={steps_before} after={steps_after}")

    eval_before_path = os.path.join(out_dir, "eval_before.json")
    eval_before_sha = write_json(eval_before_path, {"schema_version": 1, "before": dict(before_evals)})
    eval_after_path = os.path.join(out_dir, "eval_after.json")
    eval_after_sha = write_json(eval_after_path, {"schema_version": 1, "after": {"goal1": dict(after_eval)}})

    return {
        "seed": int(seed),
        "store": {"hash_base": str(store_hash_base), "hash_after": str(store.content_hash())},
        "traces": {"traces_total": int(len(traces_sorted)), "trace_sigs": [str(t.get("trace_sig") or "") for t in traces_sorted]},
        "mining": {"mined_total": int(len(mined)), "top_candidate": mined[0].to_dict() if mined else {}},
        "promotion": {
            "budget_bits": int(budget_bits),
            "used_bits": int(used_bits),
            "promoted_total": int(len(promoted)),
            "skipped_total": int(len(skipped)),
            "promoted": list(promoted),
            "promotions_jsonl_sha256": str(promotions_sha),
        },
        "before": {"goal1": dict(before_evals.get("goal1", {})), "goal2": dict(before_evals.get("goal2", {})), "goal3": dict(before_evals.get("goal3", {}))},
        "after": {"goal1": dict(after_eval)},
        "delta": {"steps_before": int(steps_before), "steps_after": int(steps_after), "delta_steps": int(steps_before - steps_after)},
        "artifacts": {
            "traces_v74_json": {"sha256": str(traces_sha)},
            "mined_candidates_v74_json": {"sha256": str(mined_sha)},
            "promotions_jsonl": {"sha256": str(promotions_sha)},
            "eval_before_json": {"sha256": str(eval_before_sha)},
            "eval_after_json": {"sha256": str(eval_after_sha)},
            "candidates": list(candidate_artifacts),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", default="results/run_smoke_mine_promote_pcc_v74")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_base = str(args.out_base)
    seed = int(args.seed)

    results: Dict[str, Any] = {"seed": seed, "tries": {}}
    sigs: List[Tuple[str, int, int]] = []
    summary_shas: List[str] = []

    for t in (1, 2):
        out_dir = f"{out_base}_try{t}"
        ensure_absent(out_dir)
        os.makedirs(out_dir, exist_ok=False)

        ev = smoke_try(out_dir=out_dir, seed=seed)
        eval_path = os.path.join(out_dir, "eval.json")
        eval_sha = write_json(eval_path, ev)

        core = {
            "seed": int(seed),
            "steps_before": int(ev.get("delta", {}).get("steps_before") if isinstance(ev.get("delta"), dict) else 0),
            "steps_after": int(ev.get("delta", {}).get("steps_after") if isinstance(ev.get("delta"), dict) else 0),
            "delta_steps": int(ev.get("delta", {}).get("delta_steps") if isinstance(ev.get("delta"), dict) else 0),
            "promoted_total": int(ev.get("promotion", {}).get("promoted_total") if isinstance(ev.get("promotion"), dict) else 0),
            "certificate_sig": str(
                (ev.get("promotion", {}).get("promoted", [{}])[0].get("certificate_sig") if isinstance(ev.get("promotion"), dict) and ev.get("promotion", {}).get("promoted") else "")
            ),
            "sha256_eval_json": str(eval_sha),
        }
        summary_sha = sha256_text(canonical_json_dumps(core))
        smoke = {"summary": core, "determinism": {"summary_sha256": str(summary_sha)}}
        smoke_path = os.path.join(out_dir, "smoke_summary.json")
        smoke_sha = write_json(smoke_path, smoke)

        sigs.append((str(core["certificate_sig"]), int(core["steps_before"]), int(core["steps_after"])))
        summary_shas.append(str(summary_sha))

        results["tries"][f"try{t}"] = {
            "out_dir": out_dir,
            "eval_json": {"path": eval_path, "sha256": eval_sha},
            "smoke_summary_json": {"path": smoke_path, "sha256": smoke_sha},
            "summary_sha256": summary_sha,
        }

    determinism_ok = bool(
        len(sigs) == 2 and sigs[0] == sigs[1] and len(summary_shas) == 2 and summary_shas[0] == summary_shas[1]
    )
    if not determinism_ok:
        _fail(f"ERROR: determinism mismatch: sigs={sigs} summary_shas={summary_shas}")
    results["determinism"] = {"ok": True, "summary_sha256": summary_shas[0], "certificate_sig": sigs[0][0], "steps_before": sigs[0][1], "steps_after": sigs[0][2]}
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

