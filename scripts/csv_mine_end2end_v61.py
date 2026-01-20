#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction, Patch, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.concepts import PRIMITIVE_OPS
from atos_core.csv_miner import CsvCandidate, materialize_concept_act_from_candidate, mine_csv_candidates
from atos_core.engine import Engine, EngineConfig
from atos_core.ethics import validate_act_for_promotion
from atos_core.ledger import Ledger
from atos_core.proof import (
    act_body_sha256_placeholder,
    build_concept_pcc_certificate_v2,
    verify_concept_pcc_v2,
)
from atos_core.store import ActStore
from atos_core.suite import CHAT_DIALOGUES_20X3, run_chat_suite


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


def stable_act_id(prefix: str, body: Dict[str, Any]) -> str:
    return f"{prefix}{sha256_hex(canonical_json_dumps(body).encode('utf-8'))[:12]}"


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ensure_absent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(canonical_json_dumps(r))
            f.write("\n")
    os.replace(tmp, path)


def write_promoted_acts_preserve_order(
    *, base_acts_path: str, out_acts_path: str, appended_acts: Sequence[Act]
) -> str:
    with open(base_acts_path, "rb") as f:
        base_bytes = f.read()
    if base_bytes and not base_bytes.endswith(b"\n"):
        base_bytes += b"\n"
    tail = b"".join(canonical_json_dumps(a.to_dict()).encode("utf-8") + b"\n" for a in appended_acts)
    tmp = out_acts_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(base_bytes)
        f.write(tail)
    os.replace(tmp, out_acts_path)
    return sha256_file(out_acts_path)


def transcript_hash(transcripts: Sequence[Dict[str, Any]]) -> str:
    full = "\n".join(str(t.get("full_text") or "") for t in transcripts)
    return sha256_hex(full.encode("utf-8"))


def make_concept_act(
    *,
    step: int,
    store_hash_excl_semantic: str,
    title: str,
    program: Sequence[Instruction],
    interface: Dict[str, Any],
    overhead_bits: int = 1024,
    meta: Optional[Dict[str, Any]] = None,
) -> Act:
    ev = {
        "name": "concept_csv_v0",
        "interface": dict(interface),
        "meta": {
            "title": str(title),
            "trained_on_store_content_hash": str(store_hash_excl_semantic),
            **(dict(meta or {})),
        },
    }
    body = {
        "kind": "concept_csv",
        "version": 1,
        "match": {},
        "program": [ins.to_dict() for ins in program],
        "evidence": ev,
        "deps": [],
        "active": True,
    }
    act_id = stable_act_id("act_concept_csv_", body)
    return Act(
        id=act_id,
        version=1,
        created_at=deterministic_iso(step=int(step)),
        kind="concept_csv",
        match={},
        program=list(program),
        evidence=ev,
        cost={"overhead_bits": int(overhead_bits)},
        deps=[],
        active=True,
    )


def make_goal_act(
    *,
    step: int,
    store_hash_excl_semantic: str,
    title: str,
    concept_id: str,
    inputs: Dict[str, Any],
    expected: Any,
    priority: int,
    overhead_bits: int = 1024,
) -> Act:
    ev = {
        "name": "goal_v0",
        "meta": {
            "title": str(title),
            "trained_on_store_content_hash": str(store_hash_excl_semantic),
        },
        "goal": {
            "priority": int(priority),
            "concept_id": str(concept_id),
            "inputs": dict(inputs),
            "expected": expected,
        },
    }
    body = {
        "kind": "goal",
        "version": 1,
        "match": {},
        "program": [],
        "evidence": ev,
        "deps": [],
        "active": True,
    }
    act_id = stable_act_id("act_goal_", body)
    return Act(
        id=act_id,
        version=1,
        created_at=deterministic_iso(step=int(step)),
        kind="goal",
        match={},
        program=[],
        evidence=ev,
        cost={"overhead_bits": int(overhead_bits)},
        deps=[],
        active=True,
    )


def _inputs_sig(inputs: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(inputs).encode("utf-8"))


def _trace_program_sig(events: List[Dict[str, Any]]) -> str:
    # Stable per-record signature of the trace events blob.
    return sha256_hex(canonical_json_dumps(events).encode("utf-8"))


def _expected_sig(output_text: str) -> str:
    return sha256_hex(str(output_text).encode("utf-8"))


def _unique_test_vectors_from_examples(
    examples: Sequence[Dict[str, Any]], *, min_vectors: int = 3
) -> List[Dict[str, Any]]:
    uniq: set = set()
    out: List[Dict[str, Any]] = []
    for ex in examples:
        if not isinstance(ex, dict):
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patch_diff", default="")
    ap.add_argument("--freeze_path", default="")
    ap.add_argument("--episodes_per_pattern", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--chat_dialogues", type=int, default=2)
    ap.add_argument("--top_k_candidates", type=int, default=4)
    ap.add_argument("--max_total_overhead_bits", type=int, default=4096)
    ap.add_argument("--concept_overhead_bits", type=int, default=1024)
    ap.add_argument("--goal_shadow_max_goals_per_turn", type=int, default=1)
    args = ap.parse_args()

    ensure_absent(args.out)
    if args.freeze_path:
        ensure_absent(args.freeze_path)
    os.makedirs(args.out, exist_ok=False)

    base_acts = os.path.join(args.acts_run, "acts.jsonl")
    if not os.path.exists(base_acts):
        _fail(f"ERROR: missing base acts.jsonl: {base_acts}")
    base_acts_sha256 = sha256_file(base_acts)

    base_store = ActStore.load_jsonl(base_acts)
    store_hash_excl = base_store.content_hash(exclude_kinds=["gate_table_ctxsig", "concept_csv", "goal"])

    # (1) Build two deterministic "inline" concepts (executed via Engine) to generate INS traces.
    # These are NOT promoted; they exist only to generate real engine traces for mining.
    concept_a = make_concept_act(
        step=10,
        store_hash_excl_semantic=store_hash_excl,
        title="trace_pattern_extract_int_v61",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "text", "out": "t"}),
            Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["t"], "out": "d"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "n"}),
            Instruction("CSV_RETURN", {"var": "n"}),
        ],
        interface={
            "input_schema": {"text": "str"},
            "output_schema": {"value": "int"},
            "validator_id": "int_value_exact",
        },
        overhead_bits=int(args.concept_overhead_bits),
        meta={"builder": "trace_pattern_v61"},
    )

    concept_b = make_concept_act(
        step=11,
        store_hash_excl_semantic=store_hash_excl,
        title="trace_pattern_json_ab_v61",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
            Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
            Instruction("CSV_PRIMITIVE", {"fn": "make_dict_ab", "in": ["a", "b"], "out": "d"}),
            Instruction("CSV_PRIMITIVE", {"fn": "json_canonical", "in": ["d"], "out": "j"}),
            Instruction("CSV_RETURN", {"var": "j"}),
        ],
        interface={
            "input_schema": {"a": "int", "b": "int"},
            "output_schema": {"value": "str"},
            "validator_id": "json_ab_int_exact",
        },
        overhead_bits=int(args.concept_overhead_bits),
        meta={"builder": "trace_pattern_v61"},
    )

    # Store for trace execution (base acts + temp trace concepts).
    store_trace = ActStore(acts=dict(base_store.acts), next_id_int=int(base_store.next_id_int))
    store_trace.add(concept_a)
    store_trace.add(concept_b)
    engine_trace = Engine(store_trace, seed=int(args.seed), config=EngineConfig())

    traces_dir = os.path.join(args.out, "traces")
    os.makedirs(traces_dir, exist_ok=False)
    csv_exec_path = os.path.join(traces_dir, "csv_exec.jsonl")
    ensure_absent(csv_exec_path)

    # (2) Execute mini-suite via engine.execute_concept_csv to capture INS events.
    episodes = max(3, int(args.episodes_per_pattern))
    trace_rows: List[Dict[str, Any]] = []

    # Pattern A episodes.
    _, fn_scan = PRIMITIVE_OPS["scan_digits"]
    _, fn_d2i = PRIMITIVE_OPS["digits_to_int"]
    for i in range(episodes):
        text = f"abc{i}123"
        digits = fn_scan(text)
        exp_int = int(fn_d2i(digits))
        inputs = {"text": str(text)}
        r = engine_trace.execute_concept_csv(
            concept_act_id=str(concept_a.id),
            inputs=dict(inputs),
            expected=exp_int,
            step=int(i),
            max_depth=8,
        )
        meta = r.get("meta") if isinstance(r, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        out_text = str(meta.get("output_text") or "")
        evs = r.get("events") if isinstance(r, dict) else []
        evs = evs if isinstance(evs, list) else []
        ctx_sig = f"engine␟extract_int␟i={i}"
        trace_rows.append(
            {
                "run_id": str(args.out),
                "ctx_sig": str(ctx_sig),
                "goal_id": f"engine_goal_extract_int_{i}",
                "program_sig": _trace_program_sig([e for e in evs if isinstance(e, dict)]),
                "events": [e for e in evs if isinstance(e, dict)],
                "inputs": dict(inputs),
                "inputs_sig": _inputs_sig(inputs),
                "output_text": str(out_text),
                "output_sig": _expected_sig(out_text),
            }
        )

    # Pattern B episodes.
    for i in range(episodes):
        a = int(40 + i)
        b = int(2 + i)
        inputs = {"a": int(a), "b": int(b)}
        expected = {"a": int(a), "b": int(b)}
        r = engine_trace.execute_concept_csv(
            concept_act_id=str(concept_b.id),
            inputs=dict(inputs),
            expected=expected,
            step=int(100 + i),
            max_depth=8,
        )
        meta = r.get("meta") if isinstance(r, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        out_text = str(meta.get("output_text") or "")
        evs = r.get("events") if isinstance(r, dict) else []
        evs = evs if isinstance(evs, list) else []
        ctx_sig = f"engine␟json_ab␟i={i}"
        trace_rows.append(
            {
                "run_id": str(args.out),
                "ctx_sig": str(ctx_sig),
                "goal_id": f"engine_goal_json_ab_{i}",
                "program_sig": _trace_program_sig([e for e in evs if isinstance(e, dict)]),
                "events": [e for e in evs if isinstance(e, dict)],
                "inputs": dict(inputs),
                "inputs_sig": _inputs_sig(inputs),
                "output_text": str(out_text),
                "output_sig": _expected_sig(out_text),
            }
        )

    write_jsonl(csv_exec_path, trace_rows)
    csv_exec_sha256 = sha256_file(csv_exec_path)

    # (3) Mine candidates from INS traces and rank deterministically.
    candidates = mine_csv_candidates(csv_exec_path, min_ops=2, max_ops=6, bits_per_op=128, overhead_bits=1024)
    mined_candidates_path = os.path.join(args.out, "mined_candidates.json")
    ensure_absent(mined_candidates_path)
    with open(mined_candidates_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"candidates": [c.to_dict() for c in candidates]}, ensure_ascii=False, indent=2))
    if len(candidates) < 2:
        _fail(f"ERROR: miner produced <2 candidates (got {len(candidates)})")

    top_k = max(2, int(args.top_k_candidates))
    top_cands = candidates[:top_k]

    # (4) Materialize top-K concepts and attach PCC (v2).
    materialized: List[Act] = []
    rejected: List[Dict[str, Any]] = []
    for idx, cand in enumerate(top_cands):
        title = f"mined_concept_v61_rank{idx:02d}"
        concept = materialize_concept_act_from_candidate(
            cand,
            step=200 + idx,
            store_content_hash_excluding_semantic=store_hash_excl,
            title=title,
            overhead_bits=int(args.concept_overhead_bits),
            meta={
                "builder": "csv_miner_v61",
                "trace_file_sha256": str(csv_exec_sha256),
                "trace_file": str(csv_exec_path),
            },
        )
        # Test vectors from miner examples (>=3, deterministic, unique by expected_sig).
        test_vectors = _unique_test_vectors_from_examples(cand.examples, min_vectors=3)
        if len(test_vectors) < 3:
            rejected.append(
                {"candidate_sig": str(cand.candidate_sig), "reason": "not_enough_test_vectors", "need": 3}
            )
            continue

        ethics = validate_act_for_promotion(concept)
        if not bool(ethics.ok):
            rejected.append(
                {
                    "candidate_sig": str(cand.candidate_sig),
                    "reason": "ethics_fail_closed",
                    "ethics": ethics.to_dict(),
                }
            )
            continue

        cert = build_concept_pcc_certificate_v2(
            concept,
            store=base_store,
            mined_from={
                "trace_file": str(csv_exec_path),
                "trace_file_sha256": str(csv_exec_sha256),
                "store_hash_excluding_semantic": str(store_hash_excl),
                "seed": int(args.seed),
                "candidate_sig": str(cand.candidate_sig),
                "ctx_sigs": [
                    str(x.get("ctx_sig") or "") for x in cand.examples[:5] if isinstance(x, dict)
                ],
            },
            test_vectors=test_vectors,
            ethics_verdict=ethics.to_dict(),
            uncertainty_policy="no_ic",
        )
        concept.evidence.setdefault("certificate_v2", cert)
        try:
            concept.evidence["certificate_v2"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(concept)
        except Exception:
            pass

        proof_v = verify_concept_pcc_v2(concept, base_store)
        if not bool(proof_v.ok):
            rejected.append(
                {
                    "candidate_sig": str(cand.candidate_sig),
                    "reason": "pcc_verify_failed",
                    "pcc": proof_v.to_dict(),
                }
            )
            continue
        materialized.append(concept)

    if len(materialized) < 2:
        _fail(f"ERROR: expected >=2 materialized concepts, got {len(materialized)}")

    # (5) Create one wrapper concept with CSV_CALL to the first int concept (PCC v2 with call_deps).
    callee_int: Optional[Act] = None
    for c in materialized:
        ev = c.evidence if isinstance(c.evidence, dict) else {}
        iface = ev.get("interface") if isinstance(ev, dict) else {}
        iface = iface if isinstance(iface, dict) else {}
        if str(iface.get("validator_id") or "") == "int_value_exact":
            callee_int = c
            break
    if callee_int is None:
        _fail("ERROR: missing mined int concept for wrapper")

    callee_iface = (callee_int.evidence.get("interface") if isinstance(callee_int.evidence, dict) else {}) or {}
    callee_iface = callee_iface if isinstance(callee_iface, dict) else {}
    in_schema = callee_iface.get("input_schema") if isinstance(callee_iface.get("input_schema"), dict) else {}
    in_schema = dict(in_schema)
    out_schema = callee_iface.get("output_schema") if isinstance(callee_iface.get("output_schema"), dict) else {}
    out_schema = dict(out_schema)
    validator_id = str(callee_iface.get("validator_id") or "")
    in_keys = sorted(list(in_schema.keys()))

    wrapper_prog: List[Instruction] = []
    bind: Dict[str, str] = {}
    for idx, name in enumerate(in_keys):
        wrapper_prog.append(Instruction("CSV_GET_INPUT", {"name": str(name), "out": f"in{idx}"}))
        bind[str(name)] = f"in{idx}"
    wrapper_prog.append(
        Instruction(
            "CSV_CALL",
            {
                "concept_id": str(callee_int.id),
                "out": "out0",
                "bind": dict(bind),
            },
        )
    )
    wrapper_prog.append(Instruction("CSV_RETURN", {"var": "out0"}))

    wrapper = make_concept_act(
        step=400,
        store_hash_excl_semantic=store_hash_excl,
        title="wrapper_call_extract_int_v61",
        program=wrapper_prog,
        interface={
            "input_schema": dict(in_schema),
            "output_schema": dict(out_schema),
            "validator_id": str(validator_id),
        },
        overhead_bits=int(args.concept_overhead_bits),
        meta={"builder": "wrapper_v61", "callee_id": str(callee_int.id)},
    )

    # Reuse test vectors from the callee's PCC (>=3) for the wrapper.
    callee_cert = (
        callee_int.evidence.get("certificate_v2") if isinstance(callee_int.evidence, dict) else {}
    )
    callee_cert = callee_cert if isinstance(callee_cert, dict) else {}
    callee_vecs = callee_cert.get("test_vectors") if isinstance(callee_cert.get("test_vectors"), list) else []
    callee_vecs = list(callee_vecs)
    wrapper_vecs = callee_vecs[:3]
    if len(wrapper_vecs) < 3:
        _fail("ERROR: callee did not have >=3 vectors for wrapper")

    wrapper_ethics = validate_act_for_promotion(wrapper)
    if not bool(wrapper_ethics.ok):
        _fail(f"ERROR: wrapper ethics promotion failed: {wrapper_ethics.reason}:{wrapper_ethics.violated_laws}")

    store_for_wrapper = ActStore(acts=dict(base_store.acts), next_id_int=int(base_store.next_id_int))
    for c in materialized:
        store_for_wrapper.add(c)
    store_for_wrapper.add(wrapper)

    wrapper_cert = build_concept_pcc_certificate_v2(
        wrapper,
        store=store_for_wrapper,
        mined_from={
            "trace_file": str(csv_exec_path),
            "trace_file_sha256": str(csv_exec_sha256),
            "store_hash_excluding_semantic": str(store_hash_excl),
            "seed": int(args.seed),
            "kind": "wrapper_manual_v61",
            "callee_id": str(callee_int.id),
        },
        test_vectors=list(wrapper_vecs),
        ethics_verdict=wrapper_ethics.to_dict(),
        uncertainty_policy="no_ic",
    )
    wrapper.evidence.setdefault("certificate_v2", wrapper_cert)
    try:
        wrapper.evidence["certificate_v2"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(wrapper)
    except Exception:
        pass
    wrapper_proof = verify_concept_pcc_v2(wrapper, store_for_wrapper)
    if not bool(wrapper_proof.ok):
        _fail(f"ERROR: wrapper PCC v2 verify failed: {wrapper_proof.reason}:{wrapper_proof.details}")

    # (6) Budgeted promotion (deterministic) over concept acts: materialized + wrapper.
    max_bits = max(0, int(args.max_total_overhead_bits))
    used_bits = 0
    promoted_concepts: List[Act] = []
    promotion_rejections: List[Dict[str, Any]] = []

    for c in list(materialized) + [wrapper]:
        ob = int(c.cost.get("overhead_bits", 1024))
        if used_bits + ob > max_bits:
            promotion_rejections.append(
                {"act_id": str(c.id), "kind": "concept_csv", "reason": "budget_exceeded", "overhead_bits": ob}
            )
            continue
        used_bits += ob
        promoted_concepts.append(c)

    if len(promoted_concepts) < 2:
        _fail("ERROR: budget caused <2 concepts to be promoted")

    # (7) Create goals (>=3 per promoted concept), explicit concept_id for determinism.
    goals: List[Act] = []
    goal_rows_expected: List[Dict[str, Any]] = []
    goal_step = 500
    for c in promoted_concepts:
        c_ev = c.evidence if isinstance(c.evidence, dict) else {}
        cert = c_ev.get("certificate_v2") if isinstance(c_ev.get("certificate_v2"), dict) else {}
        cert = cert if isinstance(cert, dict) else {}
        vecs = cert.get("test_vectors") if isinstance(cert.get("test_vectors"), list) else []
        vecs = list(vecs)
        if len(vecs) < 3:
            _fail(f"ERROR: promoted concept missing >=3 vectors: {c.id}")
        for vi, v in enumerate(vecs[:3]):
            if not isinstance(v, dict):
                continue
            inputs = v.get("inputs") if isinstance(v.get("inputs"), dict) else {}
            expected = v.get("expected")
            expected_text = str(v.get("expected_output_text") or "")
            title = f"goal_v61_{c.id}_v{vi}"
            g = make_goal_act(
                step=goal_step,
                store_hash_excl_semantic=store_hash_excl,
                title=title,
                concept_id=str(c.id),
                inputs=dict(inputs),
                expected=expected,
                priority=10,
            )
            goal_step += 1
            goals.append(g)
            goal_rows_expected.append(
                {
                    "goal_id": str(g.id),
                    "concept_id": str(c.id),
                    "inputs": dict(inputs),
                    "expected": expected,
                    "expected_output_text": expected_text,
                }
            )

    # (8) Promote append-only: base + concepts + goals; preserve order; hash-chained ledger.
    promo_dir = os.path.join(args.out, "promotion")
    os.makedirs(promo_dir, exist_ok=False)
    acts_promoted = os.path.join(promo_dir, "acts_promoted.jsonl")

    appended = list(promoted_concepts) + list(goals)
    promoted_sha256 = write_promoted_acts_preserve_order(
        base_acts_path=base_acts, out_acts_path=acts_promoted, appended_acts=appended
    )

    promotion_ledger_path = os.path.join(promo_dir, "promotion_ledger.jsonl")
    ledger = Ledger(path=promotion_ledger_path)
    for idx, a in enumerate(appended):
        patch = Patch(kind="ADD_ACT", payload={"act_id": str(a.id), "kind": str(a.kind)})
        ledger.append(
            step=int(idx),
            patch=patch,
            acts_hash=str(promoted_sha256),
            metrics={"promotion": True, "act_id": str(a.id), "kind": str(a.kind)},
            snapshot_path=None,
        )
    promotion_chain_ok = bool(ledger.verify_chain())

    promotion_manifest = {
        "base_acts_path": str(base_acts),
        "base_acts_sha256": str(base_acts_sha256),
        "store_hash_excluding_semantic": str(store_hash_excl),
        "csv_exec_path": str(csv_exec_path),
        "csv_exec_sha256": str(csv_exec_sha256),
        "mined_candidates_total": int(len(candidates)),
        "top_k_candidates": int(top_k),
        "materialized_concepts_total": int(len(materialized)),
        "rejected_materialization": list(rejected),
        "promotion_budget": {"max_total_overhead_bits": int(max_bits), "used_bits": int(used_bits)},
        "promoted_concept_ids": [str(c.id) for c in promoted_concepts],
        "promotion_rejections": list(promotion_rejections),
        "goal_ids": [str(g.id) for g in goals],
        "promotion_chain_ok": bool(promotion_chain_ok),
    }
    promotion_manifest_path = os.path.join(promo_dir, "promotion_manifest.json")
    ensure_absent(promotion_manifest_path)
    with open(promotion_manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(promotion_manifest, ensure_ascii=False, indent=2))

    # (9) From-store via goal execution; check invariance against expected baseline.
    store2 = ActStore.load_jsonl(acts_promoted)
    engine2 = Engine(store2, seed=int(args.seed), config=EngineConfig())

    mismatch_goals = 0
    call_depth_max = 0
    ethics_passed = 0
    ic_count = 0
    from_store_rows: List[Dict[str, Any]] = []

    for i, row in enumerate(goal_rows_expected):
        gid = str(row["goal_id"])
        r = engine2.execute_goal(goal_act_id=gid, step=i, max_depth=8)
        tr = r.get("trace") if isinstance(r, dict) else {}
        tr = tr if isinstance(tr, dict) else {}
        meta = tr.get("concept_meta") if isinstance(tr.get("concept_meta"), dict) else {}
        eth2 = meta.get("ethics") if isinstance(meta.get("ethics"), dict) else {}
        unc2 = meta.get("uncertainty") if isinstance(meta.get("uncertainty"), dict) else {}
        if bool(eth2.get("ok", True)):
            ethics_passed += 1
        if str(unc2.get("mode_out") or "") == "IC":
            ic_count += 1
        evs = r.get("events") if isinstance(r, dict) else []
        if isinstance(evs, list):
            for ev in evs:
                if isinstance(ev, dict):
                    call_depth_max = max(call_depth_max, int(ev.get("depth", 0) or 0))

        out_text = str(meta.get("output_text") or "")
        expected_text = str(row.get("expected_output_text") or "")
        if out_text != expected_text:
            mismatch_goals += 1
        from_store_rows.append(
            {
                "goal_id": str(gid),
                "ok": bool(r.get("ok", False)),
                "output_text": out_text,
                "expected_output_text": expected_text,
                "selected_concept_id": str(tr.get("selected_concept_id") or ""),
                "ethics": eth2,
                "uncertainty": unc2,
            }
        )

    reuse = sum(1 for r in from_store_rows if str(r.get("selected_concept_id") or ""))
    reuse_rate = float(reuse / max(1, len(from_store_rows)))

    # (10) Goal-shadow scheduler in chat loop (telemetry only, must not change tokens).
    goal_shadow_path = os.path.join(traces_dir, "goal_shadow.jsonl")
    chat_dialogues = CHAT_DIALOGUES_20X3[: max(0, int(args.chat_dialogues))]
    engine_chat_a = Engine(store2, seed=int(args.seed), config=EngineConfig())
    base_transcripts, _ = run_chat_suite(
        engine_chat_a,
        dialogues=chat_dialogues,
        max_new_tokens=int(args.max_new_tokens),
        prefix_k=8,
        template_ngram_n=6,
        template_prefix_window=32,
        csv=None,
        goal_shadow_log_path=None,
    )
    chat_hash_base = transcript_hash(base_transcripts)

    engine_chat_b = Engine(store2, seed=int(args.seed), config=EngineConfig())
    shadow_transcripts, _ = run_chat_suite(
        engine_chat_b,
        dialogues=chat_dialogues,
        max_new_tokens=int(args.max_new_tokens),
        prefix_k=8,
        template_ngram_n=6,
        template_prefix_window=32,
        csv=None,
        goal_shadow_log_path=goal_shadow_path,
        goal_shadow_max_goals_per_turn=int(args.goal_shadow_max_goals_per_turn),
    )
    chat_hash_shadow = transcript_hash(shadow_transcripts)
    goal_shadow_invariance_ok = bool(chat_hash_shadow == chat_hash_base)

    goal_shadow_lines = 0
    try:
        with open(goal_shadow_path, "r", encoding="utf-8") as f:
            for _ in f:
                goal_shadow_lines += 1
    except Exception:
        goal_shadow_lines = 0

    gain_bits_est_total = 0
    for c in promoted_concepts:
        try:
            ev = c.evidence if isinstance(c.evidence, dict) else {}
            meta = ev.get("meta") if isinstance(ev, dict) else {}
            if not isinstance(meta, dict):
                meta = {}
            gain_bits_est_total += int(meta.get("gain_bits_est") or 0)
        except Exception:
            pass

    summary = {
        "seed": int(args.seed),
        "mismatch_goals": int(mismatch_goals),
        "goals_total": int(len(goal_rows_expected)),
        "csv_invariance_ok": bool(mismatch_goals == 0),
        "mined_candidates_total": int(len(candidates)),
        "promoted_concepts_total": int(len(promoted_concepts)),
        "gain_bits_est_total": int(gain_bits_est_total),
        "reuse_rate": float(reuse_rate),
        "call_depth_max": int(call_depth_max),
        "ethics_checks_passed": int(ethics_passed),
        "uncertainty_ic_count": int(ic_count),
        "promotion_chain_ok": bool(promotion_chain_ok),
        "goal_shadow_invariance_ok": bool(goal_shadow_invariance_ok),
        "goal_shadow_lines": int(goal_shadow_lines),
        "chat_hash_base": str(chat_hash_base),
        "chat_hash_shadow": str(chat_hash_shadow),
    }

    summary_csv = os.path.join(args.out, "summary.csv")
    ensure_absent(summary_csv)
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write(
            "seed,goals_total,mismatch_goals,csv_invariance_ok,mined_candidates_total,promoted_concepts_total,gain_bits_est_total,reuse_rate,call_depth_max,ethics_checks_passed,uncertainty_ic_count,promotion_chain_ok,goal_shadow_invariance_ok,goal_shadow_lines,chat_hash_base,chat_hash_shadow\n"
        )
        f.write(
            f"{summary['seed']},{summary['goals_total']},{summary['mismatch_goals']},{int(summary['csv_invariance_ok'])},{summary['mined_candidates_total']},{summary['promoted_concepts_total']},{summary['gain_bits_est_total']},{summary['reuse_rate']},{summary['call_depth_max']},{summary['ethics_checks_passed']},{summary['uncertainty_ic_count']},{int(summary['promotion_chain_ok'])},{int(summary['goal_shadow_invariance_ok'])},{summary['goal_shadow_lines']},{summary['chat_hash_base']},{summary['chat_hash_shadow']}\n"
        )

    summary_json = os.path.join(args.out, "summary.json")
    ensure_absent(summary_json)
    with open(summary_json, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "summary": summary,
                    "promotion_manifest": promotion_manifest,
                    "promotion_rejections": promotion_rejections,
                    "from_store": from_store_rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if args.freeze_path:
        sha: Dict[str, str] = {
            "base_acts_jsonl": str(base_acts_sha256),
            "csv_exec_jsonl": str(csv_exec_sha256),
            "mined_candidates_json": str(sha256_file(mined_candidates_path)),
            "acts_promoted_jsonl": str(promoted_sha256),
            "promotion_ledger_jsonl": str(sha256_file(promotion_ledger_path)),
            "promotion_manifest_json": str(sha256_file(promotion_manifest_path)),
            "summary_csv": str(sha256_file(summary_csv)),
            "summary_json": str(sha256_file(summary_json)),
            "goal_shadow_jsonl": str(sha256_file(goal_shadow_path)) if os.path.exists(goal_shadow_path) else "",
        }
        if args.patch_diff and os.path.exists(args.patch_diff):
            sha["patch_diff"] = str(sha256_file(args.patch_diff))

        freeze = {
            "name": "V61_ENGINE_TRACES_MULTI_PCC_V2_SCHEDULER",
            "acts_source_run": str(args.acts_run),
            "out_dir": str(args.out),
            "commands": [" ".join(sys.argv)],
            "verify_chain": bool(promotion_chain_ok),
            "sha256": sha,
            "summary": summary,
        }
        with open(args.freeze_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(freeze, ensure_ascii=False, indent=2))

    print(json.dumps({"summary": summary, "out_dir": str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
