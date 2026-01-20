#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

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


def sha256_text(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_tasks", type=int, default=12)
    args = ap.parse_args()

    out_dir = str(args.out)
    ensure_absent(out_dir)
    os.makedirs(out_dir, exist_ok=False)

    tasks = list(UTILITY_DIALOGUES_V66)[: max(1, int(args.limit_tasks))]
    exec_rows: List[Dict[str, Any]] = []

    for t in tasks:
        task_id = str(t.get("task_id") or "")
        spec = t.get("expected_spec") or {}
        if not isinstance(spec, dict):
            _fail(f"ERROR: missing expected_spec for task {task_id}")
        input_keys = spec.get("input_keys") or []
        inputs = spec.get("inputs") or {}
        ops = spec.get("ops") or []
        return_var = str(spec.get("return_var") or "")
        expected_out = str(spec.get("expected_output_text") or "")
        if not isinstance(input_keys, list) or not isinstance(inputs, dict) or not isinstance(ops, list) or not return_var or not expected_out:
            _fail(f"ERROR: bad expected_spec for task {task_id}")

        vr = run_validator("plan_validator", expected_out, spec)
        if not bool(vr.passed):
            _fail(f"ERROR: baseline expected output does not validate for task {task_id}: {vr.reason}")

        events: List[Dict[str, Any]] = []
        for idx, name in enumerate([str(k) for k in input_keys]):
            events.append({"t": "INS", "op": "CSV_GET_INPUT", "name": str(name), "out": f"in{idx}"})
        for op in ops:
            if not isinstance(op, dict):
                _fail(f"ERROR: bad op in task {task_id}")
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

        exec_rows.append(
            {
                "ctx_sig": f"v66_utility␟task={task_id}",
                "inputs": dict(inputs),
                "events": events,
                "utility_passed": True,
                "expected_output_text": expected_out,
            }
        )

    csv_exec_path = os.path.join(out_dir, "csv_exec_v66.jsonl")
    csv_exec_sha = write_jsonl(csv_exec_path, exec_rows)

    # Mine candidates (includes utility bonus if utility_passed provided).
    candidates = mine_csv_candidates(
        csv_exec_path,
        min_ops=2,
        max_ops=6,
        bits_per_op=128,
        overhead_bits=1024,
        max_examples_per_candidate=20,
    )
    mined_candidates_path = os.path.join(out_dir, "mined_candidates.json")
    mined_candidates_sha = write_json(mined_candidates_path, [c.to_dict() for c in candidates])

    # Living concepts registry (WORM, chained JSONL).
    registry_dir = os.path.join(out_dir, "concept_registry_v66")
    pol = ConceptPolicies(
        ema_alpha=0.5,
        prune_min_calls=4,
        prune_min_lifetime_steps=4,
        prune_fail_streak=2,
        prune_u_threshold=0.4,
        prune_s_threshold=0.25,
    )
    reg = ConceptRegistry(run_dir=registry_dir)

    births_total = 0
    promoted_ids: List[str] = []

    # Promote top-N plan-capable candidates (deterministic).
    for cand in candidates:
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
            step=1 + births_total,
            subgraph_ref=subgraph_ref,
            interface=iface,
            policies=pol,
            birth_reason="v66_miner_promote",
            birth_prior={"u_ema": float(max(0.0, cand.utility_pass_rate)), "k_ema": 1.0},
        )
        births_total += 1
        promoted_ids.append(str(c.id))
        if len(promoted_ids) >= 3:
            break

    if births_total < 1:
        _fail("ERROR: expected >=1 promoted living concept")

    # Deterministic fusion demo: define two equivalent subgraphs and merge.
    # (Equivalent here ignores scan_overhead_units for engine_turn_subgraph_v0.)
    iface_gate = ConceptInterface(
        input_schema={"turn_sig": "str"},
        output_schema={"executed_predictor_act_ids": "list[str]"},
        validator_id="list_contains_all_str",
        preconditions={},
        postconditions={},
    )
    g1 = reg.define(
        step=100,
        subgraph_ref={
            "kind": "engine_turn_subgraph_v0",
            "executed_predictor_act_ids": ["act_dummy_0"],
            "rewrite_rule_hit_ids": [],
            "scan_overhead_units": 1,
        },
        interface=iface_gate,
        policies=pol,
        birth_reason="v66_merge_demo",
        birth_prior={"u_ema": 0.0, "k_ema": 0.0},
    )
    g2 = reg.define(
        step=101,
        subgraph_ref={
            "kind": "engine_turn_subgraph_v0",
            "executed_predictor_act_ids": ["act_dummy_0"],
            "rewrite_rule_hit_ids": [],
            "scan_overhead_units": 999,
        },
        interface=iface_gate,
        policies=pol,
        birth_reason="v66_merge_demo",
        birth_prior={"u_ema": 0.0, "k_ema": 0.0},
    )
    merged_total = reg.merge_equivalents(step=102, reason="v66_equiv_sig_demo")

    # Compositional CALL demo: wrapper calls best promoted concept (nested call_depth).
    base_id = promoted_ids[0]
    base = reg.get(base_id)
    if base is None:
        _fail("ERROR: missing base concept")
    wrapper = reg.define(
        step=110,
        subgraph_ref={"kind": "concept_call_v0", "callee_concept_id": str(base.id)},
        interface=base.interface,
        policies=pol,
        birth_reason="v66_auto_wrapper",
        birth_prior={"u_ema": float(base.u_ema), "k_ema": float(base.k_ema)},
    )

    # Execute calls and force at least one pruning deterministically.
    calls_total = 0
    pruned_total = 0

    t0 = tasks[0]
    t1 = tasks[1] if len(tasks) > 1 else tasks[0]
    spec0 = t0.get("expected_spec") if isinstance(t0.get("expected_spec"), dict) else {}
    spec1 = t1.get("expected_spec") if isinstance(t1.get("expected_spec"), dict) else {}
    if not isinstance(spec0, dict) or not isinstance(spec1, dict):
        _fail("ERROR: missing v66 expected_spec")

    # Call wrapper once (nested call_depth>0 present in logs).
    out0, vr0, _ = reg.call(
        step=200,
        concept_id=str(wrapper.id),
        inputs=dict(spec0.get("inputs") or {}),
        expected=dict(spec0),
        context_signature="smoke_v66",
        call_depth=0,
        baseline_cost=3.0,
        contract_active=False,
    )
    calls_total += 1
    if not bool(vr0.passed):
        _fail(f"ERROR: wrapper call must pass: {vr0.reason}")

    # Force failures to demonstrate pruning (misapply spec1 expected to spec0 output).
    target = reg.get(str(base.id))
    if target is None:
        _fail("ERROR: missing target concept")
    for i in range(6):
        step = 210 + i
        out, vr, _ = reg.call(
            step=int(step),
            concept_id=str(target.id),
            inputs=dict(spec0.get("inputs") or {}),
            expected=dict(spec1),
            context_signature="smoke_v66_force_prune",
            call_depth=0,
            baseline_cost=3.0,
            contract_active=False,
        )
        calls_total += 1
        if bool(vr.passed):
            _fail("ERROR: expected forced prune call to fail")
        if not bool(target.alive):
            pruned_total = 1
            break

    if pruned_total < 1:
        _fail("ERROR: expected >=1 concept pruned")

    chains_ok = reg.verify_chains()

    summary = {
        "ok": True,
        "seed": int(args.seed),
        "tasks_total": int(len(tasks)),
        "births_total": int(births_total),
        "calls_total": int(calls_total),
        "promoted_total": int(len(promoted_ids)),
        "pruned_total": int(pruned_total),
        "merged_total": int(merged_total),
        "chains_ok": dict(chains_ok),
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

    # Determinism proof: hash only stable content (exclude out_dir-dependent paths).
    det_payload = {
        "seed": int(summary["seed"]),
        "tasks_total": int(summary["tasks_total"]),
        "births_total": int(summary["births_total"]),
        "calls_total": int(summary["calls_total"]),
        "promoted_total": int(summary["promoted_total"]),
        "pruned_total": int(summary["pruned_total"]),
        "merged_total": int(summary["merged_total"]),
        "chains_ok": dict(summary["chains_ok"]),
        "sha256": dict(summary["sha256"]),
    }
    summary["determinism"] = {
        "summary_sha256": sha256_text(canonical_json_dumps(det_payload)),
        "trace_sha256": str(summary["sha256"]["concept_telemetry_jsonl"]),
    }

    smoke_summary_path = os.path.join(out_dir, "smoke_summary.json")
    write_json(smoke_summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
