#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.agent_v63 import build_v63_toolbox
from atos_core.csv_miner import materialize_concept_act_from_candidate, mine_csv_candidates
from atos_core.ethics import validate_act_for_promotion
from atos_core.proof import act_body_sha256_placeholder, build_concept_pcc_certificate_v2, verify_concept_pcc_v2
from atos_core.store import ActStore


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"ERROR: path already exists: {path}")


def stable_act_id(prefix: str, body: Dict[str, Any]) -> str:
    return f"{prefix}{sha256_hex(canonical_json_dumps(body).encode('utf-8'))[:12]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_run", required=True)
    ap.add_argument("--out", required=True, help="WORM out dir (must not exist)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ensure_absent(args.out)
    os.makedirs(args.out, exist_ok=False)

    # (1) Determinism: run agent_loop twice with the same seed and compare hashes.
    run_a = os.path.join(args.out, "run_a")
    run_b = os.path.join(args.out, "run_b")
    ensure_absent(run_a)
    ensure_absent(run_b)

    cmd_base = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "agent_loop_v63.py"),
        "--acts_run",
        str(args.acts_run),
        "--seed",
        str(int(args.seed)),
        "--max_steps",
        "512",
        "--max_depth",
        "8",
        "--max_events_per_step",
        "128",
    ]
    subprocess.run(cmd_base + ["--out", run_a], check=True)
    subprocess.run(cmd_base + ["--out", run_b], check=True)

    trace_a = os.path.join(run_a, "traces", "agent_trace_v63.jsonl")
    trace_b = os.path.join(run_b, "traces", "agent_trace_v63.jsonl")
    sum_a = os.path.join(run_a, "summary.json")
    sum_b = os.path.join(run_b, "summary.json")
    if not (os.path.exists(trace_a) and os.path.exists(trace_b) and os.path.exists(sum_a) and os.path.exists(sum_b)):
        _fail("FAIL: agent_loop did not produce expected artifacts")

    h_trace_a = sha256_file(trace_a)
    h_trace_b = sha256_file(trace_b)
    h_sum_a = sha256_file(sum_a)
    h_sum_b = sha256_file(sum_b)
    if h_trace_a != h_trace_b or h_sum_a != h_sum_b:
        _fail("FAIL: determinism check failed (hash mismatch across identical runs)")

    # (2) Trace schema sanity.
    try:
        with open(trace_a, "r", encoding="utf-8") as f:
            first = json.loads(next(iter(f)).strip())
    except Exception:
        _fail("FAIL: could not read/parse agent_trace_v63.jsonl")

    required = [
        "ctx_sig",
        "step_id",
        "goal_id",
        "inputs",
        "output_text",
        "expected_output_text",
        "selected_concept_id",
        "events_sig",
        "events",
        "ok",
        "reason",
    ]
    missing = [k for k in required if k not in first]
    if missing:
        _fail(f"FAIL: trace schema missing keys: {missing}")

    # (3) Miner produces >=2 candidates from INS events in the trace.
    cands = mine_csv_candidates(trace_a, min_ops=2, max_ops=6, bits_per_op=128, overhead_bits=1024)
    if len(cands) < 2:
        _fail(f"FAIL: expected >=2 mined candidates, got {len(cands)}")

    # (4) PCC v2: wrapper CALL chain passes, then fails on callee tamper.
    base_acts = os.path.join(args.acts_run, "acts.jsonl")
    if not os.path.exists(base_acts):
        _fail(f"FAIL: missing base acts.jsonl: {base_acts}")
    store = ActStore.load_jsonl(base_acts)
    store_hash_excl = store.content_hash(exclude_kinds=["gate_table_ctxsig", "concept_csv", "goal"])
    toolbox = build_v63_toolbox(step=1, store_hash_excl_semantic=store_hash_excl, overhead_bits=1024)
    for a in toolbox.values():
        if store.get(a.id) is None:
            store.add(a)

    top = cands[0]
    concept1 = materialize_concept_act_from_candidate(
        top,
        step=10,
        store_content_hash_excluding_semantic=store_hash_excl,
        title="smoke_v63_mined_0",
        overhead_bits=1024,
        meta={"builder": "smoke_v63", "trace_file": trace_a},
    )
    vecs: List[Dict[str, Any]] = []
    uniq: set = set()
    for ex in top.examples:
        sig = str(ex.get("expected_sig") or "")
        if not sig or sig in uniq:
            continue
        uniq.add(sig)
        vecs.append(
            {
                "inputs": dict(ex.get("inputs") or {}),
                "expected": ex.get("expected"),
                "expected_output_text": str(ex.get("expected_output_text") or ""),
            }
        )
        if len(vecs) >= 3:
            break
    if len(vecs) < 3:
        _fail("FAIL: mined candidate did not have >=3 unique vectors")

    ethics = validate_act_for_promotion(concept1)
    if not bool(ethics.ok):
        _fail(f"FAIL: ethics rejected mined concept: {ethics.reason}:{ethics.violated_laws}")
    cert1 = build_concept_pcc_certificate_v2(
        concept1,
        store=store,
        mined_from={"trace_file": trace_a, "kind": "smoke_v63"},
        test_vectors=list(vecs),
        ethics_verdict=ethics.to_dict(),
        uncertainty_policy="no_ic",
    )
    concept1.evidence.setdefault("certificate_v2", cert1)
    concept1.evidence["certificate_v2"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(concept1)
    v1 = verify_concept_pcc_v2(concept1, store)
    if not bool(v1.ok):
        _fail(f"FAIL: PCC v2 verify should pass for callee, got {v1.reason}:{v1.details}")

    iface = concept1.evidence.get("interface") if isinstance(concept1.evidence, dict) else {}
    iface = iface if isinstance(iface, dict) else {}
    in_schema = iface.get("input_schema") if isinstance(iface.get("input_schema"), dict) else {}
    in_schema = dict(in_schema)
    out_schema = iface.get("output_schema") if isinstance(iface.get("output_schema"), dict) else {}
    out_schema = dict(out_schema)
    validator_id = str(iface.get("validator_id") or "")
    in_keys = sorted(list(in_schema.keys()))

    prog: List[Instruction] = []
    bind: Dict[str, str] = {}
    for idx, name in enumerate(in_keys):
        prog.append(Instruction("CSV_GET_INPUT", {"name": str(name), "out": f"in{idx}"}))
        bind[str(name)] = f"in{idx}"
    prog.append(Instruction("CSV_CALL", {"concept_id": str(concept1.id), "out": "out0", "bind": dict(bind)}))
    prog.append(Instruction("CSV_RETURN", {"var": "out0"}))

    wrapper = Act(
        id="",
        version=1,
        created_at=deterministic_iso(step=20),
        kind="concept_csv",
        match={},
        program=list(prog),
        evidence={
            "name": "concept_csv_v0",
            "interface": {"input_schema": dict(in_schema), "output_schema": dict(out_schema), "validator_id": str(validator_id)},
            "meta": {"title": "wrapper_smoke_v63"},
        },
        cost={"overhead_bits": 1024},
        deps=[],
        active=True,
    )
    body_w = {
        "kind": "concept_csv",
        "version": 1,
        "match": {},
        "program": [ins.to_dict() for ins in wrapper.program],
        "evidence": wrapper.evidence,
        "deps": [],
        "active": True,
    }
    wrapper.id = stable_act_id("act_concept_csv_", body_w)

    store2 = ActStore(acts=dict(store.acts), next_id_int=int(store.next_id_int))
    store2.add(concept1)
    store2.add(wrapper)

    cert_w = build_concept_pcc_certificate_v2(
        wrapper,
        store=store2,
        mined_from={"kind": "wrapper_smoke_v63", "callee_id": str(concept1.id)},
        test_vectors=list(vecs),
        ethics_verdict=validate_act_for_promotion(wrapper).to_dict(),
        uncertainty_policy="no_ic",
    )
    wrapper.evidence.setdefault("certificate_v2", cert_w)
    wrapper.evidence["certificate_v2"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(wrapper)
    vw = verify_concept_pcc_v2(wrapper, store2)
    if not bool(vw.ok):
        _fail(f"FAIL: PCC v2 verify should pass for wrapper, got {vw.reason}:{vw.details}")

    # Tamper callee program: change one primitive; wrapper verify must fail on callee hash mismatch.
    tampered = ActStore(acts=dict(store2.acts), next_id_int=int(store2.next_id_int))
    callee_t = tampered.get(concept1.id)
    if callee_t is None:
        _fail("FAIL: callee missing in tampered store")
    for idx, ins in enumerate(list(callee_t.program)):
        if str(ins.op) == "CSV_PRIMITIVE":
            new_args = dict(ins.args or {})
            new_args["fn"] = "strip_one_leading_zero"
            callee_t.program[idx] = Instruction("CSV_PRIMITIVE", new_args)
            break

    vt = verify_concept_pcc_v2(wrapper, tampered)
    if bool(vt.ok) or str(vt.reason) != "callee_program_sha256_mismatch":
        _fail(f"FAIL: expected callee_program_sha256_mismatch, got {vt.reason}:{vt.details}")

    out = {
        "ok": True,
        "determinism": {"trace_sha256": h_trace_a, "summary_sha256": h_sum_a},
        "candidates_total": int(len(cands)),
        "top_candidate_sig": str(top.candidate_sig),
    }
    out_path = os.path.join(args.out, "smoke_result.json")
    ensure_absent(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

