#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps
from atos_core.concepts import ConceptInterface, ConceptPolicies, ConceptRegistry, concept_id_for
from atos_core.csv_miner import mine_csv_candidates
from atos_core.suite import UTILITY_DIALOGUES_V66
from atos_core.validators import run_validator


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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
    os.replace(tmp, path)
    return sha256_file(path)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> str:
    ensure_absent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(canonical_json_dumps(r))
            f.write("\n")
    os.replace(tmp, path)
    return sha256_file(path)


def _task_exec_record(task: Dict[str, Any]) -> Dict[str, Any]:
    task_id = str(task.get("task_id") or "")
    spec = task.get("expected_spec") or {}
    if not isinstance(spec, dict):
        raise ValueError("missing_expected_spec")
    input_keys = spec.get("input_keys") or []
    inputs = spec.get("inputs") or {}
    ops = spec.get("ops") or []
    return_var = str(spec.get("return_var") or "")
    expected_out = str(spec.get("expected_output_text") or "")
    if not isinstance(input_keys, list) or not isinstance(inputs, dict) or not isinstance(ops, list) or not return_var or not expected_out:
        raise ValueError("bad_expected_spec")

    vr = run_validator("plan_validator", expected_out, spec)
    if not bool(vr.passed):
        raise ValueError(f"baseline_invalid:{vr.reason}")

    events: List[Dict[str, Any]] = []
    for idx, name in enumerate([str(k) for k in input_keys]):
        events.append({"t": "INS", "op": "CSV_GET_INPUT", "name": str(name), "out": f"in{idx}"})
    for op in ops:
        if not isinstance(op, dict):
            raise ValueError("bad_op")
        events.append(
            {
                "t": "INS",
                "op": "CSV_PRIMITIVE",
                "fn": str(op.get("fn") or ""),
                "in": list(op.get("in") or []),
                "out": str(op.get("out") or ""),
            }
        )
    events.append({"t": "INS", "op": "CSV_RETURN", "var": str(return_var)})

    return {
        "ctx_sig": f"v66_utility␟task={task_id}",
        "inputs": dict(inputs),
        "events": events,
        "utility_passed": True,
        "expected_output_text": expected_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_tasks", type=int, default=12)
    ap.add_argument("--top_k_candidates", type=int, default=8)
    ap.add_argument("--promote_max", type=int, default=4)
    args = ap.parse_args()

    out_dir = str(args.out)
    ensure_absent(out_dir)
    os.makedirs(out_dir, exist_ok=False)

    tasks = list(UTILITY_DIALOGUES_V66)[: max(1, int(args.limit_tasks))]

    # (1) Generate deterministic exec traces (JSONL).
    exec_rows: List[Dict[str, Any]] = []
    for t in tasks:
        try:
            exec_rows.append(_task_exec_record(t))
        except Exception as e:
            _fail(f"ERROR: failed to build exec record for task {t.get('task_id')}: {e}")

    csv_exec_path = os.path.join(out_dir, "csv_exec_v66.jsonl")
    csv_exec_sha = write_jsonl(csv_exec_path, exec_rows)

    # (2) Mine candidates.
    candidates = mine_csv_candidates(
        csv_exec_path,
        min_ops=2,
        max_ops=6,
        bits_per_op=128,
        overhead_bits=1024,
        max_examples_per_candidate=50,
    )
    mined_candidates_path = os.path.join(out_dir, "mined_candidates.json")
    mined_candidates_sha = write_json(mined_candidates_path, [c.to_dict() for c in candidates])

    # (3) Promote living concepts under "utility as law".
    reg_dir = os.path.join(out_dir, "concept_registry_v66")
    pol = ConceptPolicies(
        ema_alpha=0.4,
        prune_min_calls=4,
        prune_min_lifetime_steps=4,
        prune_fail_streak=2,
        prune_u_threshold=0.4,
        prune_s_threshold=0.25,
    )
    reg = ConceptRegistry(run_dir=reg_dir)

    births_total = 0
    promoted_ids: List[str] = []
    promote_max = max(1, int(args.promote_max))
    top_k = max(1, int(args.top_k_candidates))

    for cand in candidates[:top_k]:
        # Law: only promote candidates that have an explicit, deterministic validator for utility.
        if str(cand.validator_id) != "plan_validator":
            continue
        iface = ConceptInterface(
            input_schema=dict(cand.input_schema),
            output_schema={"value": str(cand.output_type)},
            validator_id=str(cand.validator_id),
            preconditions={},
            postconditions={},
        )
        subgraph_ref = {
            "kind": "csv_ops_v0",
            "input_keys": list(cand.input_schema.keys()),
            "ops": list(cand.ops),
            "return_var": str(cand.ops[-1].get("out") or ""),
        }
        cid = concept_id_for(subgraph_ref, iface)
        if reg.get(cid) is not None:
            continue
        c = reg.define(
            step=10 + births_total,
            subgraph_ref=subgraph_ref,
            interface=iface,
            policies=pol,
            birth_reason="v66_promote_by_utility",
            birth_prior={"u_ema": float(max(0.0, cand.utility_pass_rate)), "k_ema": 1.0},
        )
        births_total += 1
        promoted_ids.append(str(c.id))
        if len(promoted_ids) >= promote_max:
            break

    if births_total < 2:
        _fail("ERROR: expected >=2 promoted concepts (need at least 2 distinct programs for pruning demo)")

    # Index promoted concepts by input key set for deterministic selection per task.
    by_inputs: Dict[Tuple[str, ...], str] = {}
    for cid in promoted_ids:
        c = reg.get(cid)
        if c is None:
            continue
        keys = tuple(sorted(list(c.interface.input_schema.keys())))
        if keys not in by_inputs:
            by_inputs[keys] = str(c.id)

    # (4) Compositional CALL (demo): wrapper calls the best sum-concept once (nested call_depth).
    base_id = promoted_ids[0]
    base = reg.get(base_id)
    if base is None:
        _fail("ERROR: missing base concept")
    wrapper = reg.define(
        step=200,
        subgraph_ref={"kind": "concept_call_v0", "callee_concept_id": str(base.id)},
        interface=base.interface,
        policies=pol,
        birth_reason="v66_wrapper_call_demo",
        birth_prior={"u_ema": float(base.u_ema), "k_ema": float(base.k_ema)},
    )

    # (5) Execute reuse with invariance fallback (baseline output is expected_output_text).
    calls_total = 0
    fallbacks_total = 0
    mismatches_total = 0
    pass_before = 0
    pass_after = 0

    # Wrapper CALL demo (nested call_depth): execute once on the first task if compatible.
    spec_demo = tasks[0].get("expected_spec") if isinstance(tasks[0].get("expected_spec"), dict) else {}
    if isinstance(spec_demo, dict):
        try:
            out_demo, vr_demo, _ = reg.call(
                step=250,
                concept_id=str(wrapper.id),
                inputs=dict(spec_demo.get("inputs") or {}),
                expected=dict(spec_demo),
                context_signature="v66_wrapper_demo",
                call_depth=0,
                baseline_cost=3.0,
                contract_active=False,
            )
            calls_total += 1
            if not bool(vr_demo.passed):
                _fail(f"ERROR: wrapper demo call failed: {vr_demo.reason}")
        except Exception as e:
            _fail(f"ERROR: wrapper demo call exception: {e}")

    for i, t in enumerate(tasks):
        task_id = str(t.get("task_id") or f"task_{i}")
        spec = t.get("expected_spec") if isinstance(t.get("expected_spec"), dict) else {}
        if not isinstance(spec, dict):
            _fail(f"ERROR: missing expected_spec for task {task_id}")
        baseline_out = str(spec.get("expected_output_text") or "")
        if not baseline_out:
            _fail(f"ERROR: missing expected_output_text for task {task_id}")

        # Baseline utility check (deterministic).
        vr_base = run_validator("plan_validator", baseline_out, spec)
        if bool(vr_base.passed):
            pass_before += 1

        inputs = dict(spec.get("inputs") or {})
        key = tuple(sorted(list(inputs.keys())))
        pick = by_inputs.get(key)
        if not pick:
            mismatches_total += 1
            fallbacks_total += 1
            out_s = baseline_out
        else:
            out, vr, _ = reg.call(
                step=300 + i,
                concept_id=str(pick),
                inputs=dict(inputs),
                expected=dict(spec),
                context_signature=f"v66_reuse␟task={task_id}",
                call_depth=0,
                baseline_cost=3.0,
                contract_active=False,
            )
            calls_total += 1
            out_s = "" if out is None else str(out)

        if out_s != baseline_out:
            mismatches_total += 1
            fallbacks_total += 1
            out_s = baseline_out

        vr_after = run_validator("plan_validator", out_s, spec)
        if bool(vr_after.passed):
            pass_after += 1

    # (6) Prune demonstration: misapply second concept until it is deactivated.
    pruned_total = 0
    stress_id = promoted_ids[1]
    stress = reg.get(stress_id)
    if stress is None:
        _fail("ERROR: missing stress concept")
    stress_key = tuple(sorted(list(stress.interface.input_schema.keys())))
    spec_in: Dict[str, Any] = {}
    spec_wrong: Dict[str, Any] = {}
    for t in tasks:
        s = t.get("expected_spec")
        if not isinstance(s, dict):
            continue
        inp = s.get("inputs")
        if not isinstance(inp, dict):
            continue
        k = tuple(sorted(list(inp.keys())))
        if k == stress_key and not spec_in:
            spec_in = dict(s)
    for t in tasks:
        s = t.get("expected_spec")
        if not isinstance(s, dict):
            continue
        if not spec_in:
            continue
        if str(s.get("expected_output_text") or "") != str(spec_in.get("expected_output_text") or ""):
            spec_wrong = dict(s)
            break
    if not spec_in or not spec_wrong:
        _fail("ERROR: could not pick specs for prune demo")
    for k in range(8):
        step = 400 + k
        out, vr, _ = reg.call(
            step=int(step),
            concept_id=str(stress.id),
            inputs=dict(spec_in.get("inputs") or {}),
            expected=dict(spec_wrong),
            context_signature="v66_force_prune",
            call_depth=0,
            baseline_cost=3.0,
            contract_active=False,
        )
        calls_total += 1
        if bool(vr.passed):
            _fail("ERROR: expected stress call to fail")
        if not bool(stress.alive):
            pruned_total = 1
            break

    if pruned_total < 1:
        _fail("ERROR: expected >=1 concept pruned")

    chains_ok = reg.verify_chains()
    promotion_chain_ok = bool(all(bool(v) for v in chains_ok.values()))
    invariance_ok = bool(mismatches_total == 0)

    utility_before = float(pass_before / max(1, len(tasks)))
    utility_after = float(pass_after / max(1, len(tasks)))

    # Top concepts snapshot.
    top_concepts: List[Dict[str, Any]] = []
    alive = list(reg.alive_concepts())
    alive.sort(key=lambda c: (-float(c.u_ema), float(c.k_ema), str(c.id)))
    for c in alive[:5]:
        top_concepts.append(
            {
                "concept_id": str(c.id),
                "alive": bool(c.alive),
                "calls_total": int(c.calls_total),
                "u_ema": float(c.u_ema),
                "k_ema": float(c.k_ema),
                "s_t": float(c.s_t),
                "contexts_distinct": int(c.contexts_distinct()),
                "validator_id": str(c.interface.validator_id),
                "subgraph_kind": str(c.subgraph_ref.get("kind") or ""),
            }
        )

    summary = {
        "seed": int(args.seed),
        "tasks_total": int(len(tasks)),
        "births_total": int(births_total),
        "promoted_total": int(len(promoted_ids)),
        "calls_total": int(calls_total),
        "pruned_total": int(pruned_total),
        "fallbacks_total": int(fallbacks_total),
        "mismatches_total": int(mismatches_total),
        "invariance_ok": bool(invariance_ok),
        "promotion_chain_ok": bool(promotion_chain_ok),
        "utility_before": float(utility_before),
        "utility_after": float(utility_after),
        "chains_ok": dict(chains_ok),
        "top_concepts": list(top_concepts),
        "artifacts": {
            "csv_exec_v66_jsonl": str(csv_exec_path),
            "mined_candidates_json": str(mined_candidates_path),
            "concepts_jsonl": str(reg.concepts_path),
            "concept_evidence_jsonl": str(reg.evidence_path),
            "concept_telemetry_jsonl": str(reg.telemetry_path),
        },
        "sha256": {
            "csv_exec_v66_jsonl": str(csv_exec_sha),
            "mined_candidates_json": str(mined_candidates_sha),
            "concepts_jsonl": str(sha256_file(reg.concepts_path)),
            "concept_evidence_jsonl": str(sha256_file(reg.evidence_path)),
            "concept_telemetry_jsonl": str(sha256_file(reg.telemetry_path)),
        },
    }

    summary_path = os.path.join(out_dir, "summary.json")
    summary_sha = write_json(summary_path, {"summary": summary})
    out = {"out_dir": out_dir, "summary": summary}
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
