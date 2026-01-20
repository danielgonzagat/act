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

from atos_core.act import Act, canonical_json_dumps, sha256_hex
from atos_core.csv_miner import materialize_concept_act_from_candidate, mine_csv_candidates
from atos_core.ethics import validate_act_for_promotion
from atos_core.proof import act_body_sha256_placeholder, build_concept_pcc_certificate_v2, verify_concept_pcc_v2
from atos_core.store import ActStore
from atos_core.toc import detect_duplicate, toc_eval, verify_concept_toc_v1


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


def _domain_from_ctx_sig(ctx_sig: str) -> str:
    s = str(ctx_sig)
    if "task=v63_parse_" in s or "task=v63_json_" in s or "task=v65_dialog_" in s:
        return "A"
    if "task=v63_math_" in s or "task=v63_plan_" in s or "task=v65_math_" in s or "task=v65_plan_" in s:
        return "B"
    return "B"


def _unique_vectors_by_domain(examples: List[Dict[str, Any]], *, domain: str, min_vectors: int = 3) -> List[Dict[str, Any]]:
    exs = [e for e in examples if isinstance(e, dict)]
    exs.sort(key=lambda e: (str(e.get("ctx_sig") or ""), str(e.get("expected_sig") or "")))
    uniq: set = set()
    out: List[Dict[str, Any]] = []
    for ex in exs:
        if _domain_from_ctx_sig(str(ex.get("ctx_sig") or "")) != str(domain):
            continue
        sig = str(ex.get("expected_sig") or "")
        if not sig or sig in uniq:
            continue
        uniq.add(sig)
        out.append(
            {
                "inputs": dict(ex.get("inputs") or {}),
                "expected": ex.get("expected"),
                "expected_output_text": str(ex.get("expected_output_text") or ""),
            }
        )
        if len(out) >= int(min_vectors):
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_run", required=True)
    ap.add_argument("--out", required=True, help="WORM out dir (must not exist)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ensure_absent(args.out)
    os.makedirs(args.out, exist_ok=False)

    # (1) Determinism: run agent_loop_v65 twice with the same seed and compare hashes.
    run_a = os.path.join(args.out, "run_a")
    run_b = os.path.join(args.out, "run_b")
    ensure_absent(run_a)
    ensure_absent(run_b)

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "agent_loop_v65.py"),
        "--acts_run",
        str(args.acts_run),
        "--seed",
        str(int(args.seed)),
        "--planner_max_depth",
        "8",
        "--planner_max_expansions",
        "8000",
        "--max_events_per_step",
        "128",
    ]
    subprocess.run(cmd + ["--out", run_a], check=True)
    subprocess.run(cmd + ["--out", run_b], check=True)

    trace_a = os.path.join(run_a, "traces", "agent_trace_v65.jsonl")
    trace_b = os.path.join(run_b, "traces", "agent_trace_v65.jsonl")
    sum_a = os.path.join(run_a, "summary.json")
    sum_b = os.path.join(run_b, "summary.json")
    if not (os.path.exists(trace_a) and os.path.exists(trace_b) and os.path.exists(sum_a) and os.path.exists(sum_b)):
        _fail("FAIL: agent_loop_v65 did not produce expected artifacts")

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
        _fail("FAIL: could not read/parse agent_trace_v65.jsonl")

    required = [
        "ctx_sig",
        "step_id",
        "inputs",
        "output_text",
        "expected_output_text",
        "selected_concept_id",
        "program_sig",
        "events_sig",
        "events",
        "ok",
        "reason",
    ]
    missing = [k for k in required if k not in first]
    if missing:
        _fail(f"FAIL: trace schema missing keys: {missing}")

    # (3) Miner produces >=2 candidates from INS events in the trace.
    cands = mine_csv_candidates(
        trace_a,
        min_ops=2,
        max_ops=6,
        bits_per_op=128,
        overhead_bits=1024,
        max_examples_per_candidate=200,
    )
    if len(cands) < 2:
        _fail(f"FAIL: expected >=2 mined candidates, got {len(cands)}")

    # Pick a candidate that has vectors in both domains (ToC).
    top = None
    vecA: List[Dict[str, Any]] = []
    vecB: List[Dict[str, Any]] = []
    for c in cands:
        vA = _unique_vectors_by_domain(list(c.examples), domain="A", min_vectors=3)
        vB = _unique_vectors_by_domain(list(c.examples), domain="B", min_vectors=3)
        if len(vA) >= 3 and len(vB) >= 3:
            top = c
            vecA = vA
            vecB = vB
            break
    if top is None:
        _fail("FAIL: no candidate had >=3 vectors in both domains A and B")

    base_acts = os.path.join(args.acts_run, "acts.jsonl")
    if not os.path.exists(base_acts):
        _fail(f"FAIL: missing base acts.jsonl: {base_acts}")
    store = ActStore.load_jsonl(base_acts)
    store_hash_excl = store.content_hash(exclude_kinds=["gate_table_ctxsig", "concept_csv", "goal"])

    concept1 = materialize_concept_act_from_candidate(
        top,
        step=10,
        store_content_hash_excluding_semantic=store_hash_excl,
        title="smoke_v65_mined_0",
        overhead_bits=1024,
        meta={"builder": "smoke_v65", "trace_file": trace_a},
    )
    ethics = validate_act_for_promotion(concept1)
    if not bool(ethics.ok):
        _fail(f"FAIL: ethics rejected mined concept: {ethics.reason}:{ethics.violated_laws}")

    store2 = ActStore(acts=dict(store.acts), next_id_int=int(store.next_id_int))
    store2.add(concept1)
    toc = toc_eval(
        concept_act=concept1,
        vectors_A=list(vecA),
        vectors_B=list(vecB),
        store=store2,
        domain_A="A",
        domain_B="B",
        min_vectors_per_domain=3,
    )
    if not bool(toc.get("pass_A", False)) or not bool(toc.get("pass_B", False)):
        _fail("FAIL: expected ToC eval to pass for both domains on concept1")

    cert1 = build_concept_pcc_certificate_v2(
        concept1,
        store=store2,
        mined_from={"trace_file": trace_a, "kind": "smoke_v65"},
        test_vectors=list(vecA[:3]),
        ethics_verdict=ethics.to_dict(),
        uncertainty_policy="no_ic",
        toc_v1=dict(toc),
    )
    concept1.evidence.setdefault("certificate_v2", cert1)
    concept1.evidence["certificate_v2"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(concept1)

    v1 = verify_concept_pcc_v2(concept1, store2)
    if not bool(v1.ok):
        _fail(f"FAIL: PCC v2 verify should pass, got {v1.reason}:{v1.details}")
    tv1 = verify_concept_toc_v1(concept1, store=store2)
    if not bool(tv1.ok):
        _fail(f"FAIL: ToC verify should pass, got {tv1.reason}:{tv1.details}")

    # Create a failing ToC certificate (domain B vectors missing) and prove it is rejected.
    concept_fail = materialize_concept_act_from_candidate(
        top,
        step=11,
        store_content_hash_excluding_semantic=store_hash_excl,
        title="smoke_v65_mined_fail",
        overhead_bits=1024,
        meta={"builder": "smoke_v65_fail", "trace_file": trace_a},
    )
    store3 = ActStore(acts=dict(store.acts), next_id_int=int(store.next_id_int))
    store3.add(concept_fail)
    toc_fail = dict(toc)
    toc_fail["vectors_B"] = []
    toc_fail["pass_B"] = False

    cert_fail = build_concept_pcc_certificate_v2(
        concept_fail,
        store=store3,
        mined_from={"trace_file": trace_a, "kind": "smoke_v65_fail"},
        test_vectors=list(vecA[:3]),
        ethics_verdict=ethics.to_dict(),
        uncertainty_policy="no_ic",
        toc_v1=dict(toc_fail),
    )
    concept_fail.evidence.setdefault("certificate_v2", cert_fail)
    concept_fail.evidence["certificate_v2"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(concept_fail)

    pv_fail = verify_concept_pcc_v2(concept_fail, store3)
    toc_verdict_fail = verify_concept_toc_v1(concept_fail, store=store3)
    if bool(pv_fail.ok) and bool(toc_verdict_fail.ok):
        _fail("FAIL: concept_fail should not pass ToC gating (must not be promoted)")
    if bool(toc_verdict_fail.ok):
        _fail("FAIL: expected ToC verify to fail when vectors_B missing")

    # Duplicate detection: clone with same program/interface but different id must be flagged.
    clone = Act.from_dict(concept1.to_dict())
    clone.id = stable_act_id(
        "act_concept_csv_clone_",
        {"clone_of": str(concept1.id), "program": [i.to_dict() for i in clone.program], "iface": clone.evidence.get("interface", {})},
    )
    dup = detect_duplicate(clone, existing=[concept1], similarity_threshold=0.95)
    if dup is None:
        _fail("FAIL: expected detect_duplicate to flag clone")

    out: Dict[str, Any] = {
        "ok": True,
        "determinism": {"trace_sha256": h_trace_a, "summary_sha256": h_sum_a},
        "candidates_total": int(len(cands)),
        "picked_candidate_sig": str(top.candidate_sig),
        "toc_gate": {"pass_ok": True, "fail_ok": True},
        "duplicate_detection": {"dup": dup},
    }

    smoke_path = os.path.join(args.out, "smoke_summary.json")
    ensure_absent(smoke_path)
    with open(smoke_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

