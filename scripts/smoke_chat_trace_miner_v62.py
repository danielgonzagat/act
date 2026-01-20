#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.concepts import PRIMITIVE_OPS
from atos_core.csv_miner import materialize_concept_act_from_candidate, mine_csv_candidates
from atos_core.engine import Engine, EngineConfig
from atos_core.ethics import validate_act_for_promotion
from atos_core.proof import act_body_sha256_placeholder, build_concept_pcc_certificate_v2, verify_concept_pcc_v2
from atos_core.store import ActStore
from atos_core.suite import CHAT_DIALOGUES_20X3, run_chat_suite


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"ERROR: path already exists: {path}")


def transcript_hash(transcripts: Sequence[Dict[str, Any]]) -> str:
    full = "\n".join(str(t.get("full_text") or "") for t in transcripts)
    return sha256_hex(full.encode("utf-8"))


def stable_act_id(prefix: str, body: Dict[str, Any]) -> str:
    return f"{prefix}{sha256_hex(canonical_json_dumps(body).encode('utf-8'))[:12]}"


def make_concept_act(
    *,
    step: int,
    title: str,
    program: Sequence[Instruction],
    interface: Dict[str, Any],
) -> Act:
    ev = {"name": "concept_csv_v0", "interface": dict(interface), "meta": {"title": str(title)}}
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
        cost={"overhead_bits": 1024},
        deps=[],
        active=True,
    )


def make_goal_act(
    *,
    step: int,
    title: str,
    concept_id: str,
    inputs: Dict[str, Any],
    expected: Any,
    priority: int = 10,
) -> Act:
    ev = {
        "name": "goal_v0",
        "meta": {"title": str(title)},
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
        cost={"overhead_bits": 1024},
        deps=[],
        active=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ensure_absent(args.out)
    os.makedirs(args.out, exist_ok=False)

    base_acts = os.path.join(args.acts_run, "acts.jsonl")
    if not os.path.exists(base_acts):
        _fail(f"FAIL: missing acts.jsonl: {base_acts}")
    base_store = ActStore.load_jsonl(base_acts)

    traces_dir = os.path.join(args.out, "traces")
    os.makedirs(traces_dir, exist_ok=False)
    goal_shadow_log_path = os.path.join(traces_dir, "goal_shadow.jsonl")
    goal_shadow_trace_log_path = os.path.join(traces_dir, "goal_shadow_trace.jsonl")

    # Seed store with two concepts + 6 goals (3 per concept) to generate mineable traces.
    concept_a = make_concept_act(
        step=1,
        title="seed_extract_int",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "text", "out": "t"}),
            Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["t"], "out": "d"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "n"}),
            Instruction("CSV_RETURN", {"var": "n"}),
        ],
        interface={"input_schema": {"text": "str"}, "output_schema": {"value": "int"}, "validator_id": "int_value_exact"},
    )
    concept_b = make_concept_act(
        step=2,
        title="seed_json_ab",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
            Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
            Instruction("CSV_PRIMITIVE", {"fn": "make_dict_ab", "in": ["a", "b"], "out": "d"}),
            Instruction("CSV_PRIMITIVE", {"fn": "json_canonical", "in": ["d"], "out": "j"}),
            Instruction("CSV_RETURN", {"var": "j"}),
        ],
        interface={"input_schema": {"a": "int", "b": "int"}, "output_schema": {"value": "str"}, "validator_id": "json_ab_int_exact"},
    )

    store = ActStore(acts=dict(base_store.acts), next_id_int=int(base_store.next_id_int))
    store.add(concept_a)
    store.add(concept_b)

    _, fn_scan = PRIMITIVE_OPS["scan_digits"]
    _, fn_d2i = PRIMITIVE_OPS["digits_to_int"]
    goals: List[Act] = []
    for i, text in enumerate(["abc0123", "id=42", "x9y7"]):
        digits = fn_scan(text)
        exp_int = int(fn_d2i(digits))
        goals.append(make_goal_act(step=10 + i, title=f"g_int_{i}", concept_id=str(concept_a.id), inputs={"text": text}, expected=exp_int))
    for i, (a, b) in enumerate([(40, 2), (41, 3), (42, 4)]):
        goals.append(make_goal_act(step=20 + i, title=f"g_json_{i}", concept_id=str(concept_b.id), inputs={"a": a, "b": b}, expected={"a": a, "b": b}))
    for g in goals:
        store.add(g)

    # Chat invariance (tokens) baseline vs shadow+schedule.
    dialogues = CHAT_DIALOGUES_20X3[:4]
    engine_a = Engine(store, seed=0, config=EngineConfig())
    t_base, _ = run_chat_suite(engine_a, dialogues=dialogues, max_new_tokens=20, goal_shadow_log_path=None)
    h_base = transcript_hash(t_base)

    engine_b = Engine(store, seed=0, config=EngineConfig())
    t_shadow, _ = run_chat_suite(
        engine_b,
        dialogues=dialogues,
        max_new_tokens=20,
        goal_shadow_log_path=goal_shadow_log_path,
        goal_shadow_trace_log_path=goal_shadow_trace_log_path,
        goal_shadow_max_goals_per_turn=1,
    )
    h_shadow = transcript_hash(t_shadow)
    if h_base != h_shadow:
        _fail("FAIL: chat transcript hash changed with goal shadow scheduler")

    # Log schema: one line per goal with {ctx_sig, goal_id, skipped_by_scheduler}, and at least 1 skipped.
    saw_skipped = False
    seen_fields = 0
    try:
        with open(goal_shadow_log_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "ctx_sig" in row and "goal_id" in row and "skipped_by_scheduler" in row:
                    seen_fields += 1
                if bool(row.get("skipped_by_scheduler", False)):
                    saw_skipped = True
        if seen_fields <= 0:
            _fail("FAIL: goal_shadow log missing required fields")
        if not saw_skipped:
            _fail("FAIL: expected at least one skipped_by_scheduler=true record")
    except Exception:
        _fail("FAIL: could not read/parse goal_shadow log")

    # Miner accepts INS from goal-shadow trace log and produces >=2 candidates.
    cands = mine_csv_candidates(goal_shadow_trace_log_path, min_ops=2, max_ops=6)
    if len(cands) < 2:
        _fail(f"FAIL: expected >=2 mined candidates from goal shadow trace, got {len(cands)}")

    # PCC v2 tamper detection (callee_program_sha256_mismatch).
    cand_int = None
    for c in cands:
        if str(getattr(c, "validator_id", "")) == "int_value_exact":
            cand_int = c
            break
    if cand_int is None:
        _fail("FAIL: expected an int_value_exact candidate from chat goal-shadow trace")

    top = cand_int
    callee = materialize_concept_act_from_candidate(
        top,
        step=100,
        store_content_hash_excluding_semantic="dummy",
        title="mined_from_chat_trace",
        overhead_bits=1024,
    )
    ethics = validate_act_for_promotion(callee)
    if not bool(ethics.ok):
        _fail(f"FAIL: ethics rejected mined callee: {ethics.reason}")
    vecs: List[Dict[str, Any]] = []
    uniq: set = set()
    for ex in top.examples:
        if not isinstance(ex, dict):
            continue
        sig = str(ex.get("expected_sig") or "")
        if not sig or sig in uniq:
            continue
        uniq.add(sig)
        vecs.append(
            {"inputs": dict(ex.get("inputs") or {}), "expected": ex.get("expected"), "expected_output_text": str(ex.get("expected_output_text") or "")}
        )
        if len(vecs) >= 3:
            break
    if len(vecs) < 3:
        _fail("FAIL: not enough test vectors for PCC v2")

    store_p = ActStore()
    store_p.add(callee)
    cert = build_concept_pcc_certificate_v2(callee, store=store_p, mined_from={"trace": goal_shadow_trace_log_path}, test_vectors=vecs, ethics_verdict=ethics.to_dict(), uncertainty_policy="no_ic")
    callee.evidence.setdefault("certificate_v2", cert)
    callee.evidence["certificate_v2"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(callee)
    v = verify_concept_pcc_v2(callee, store_p)
    if not bool(v.ok):
        _fail(f"FAIL: callee PCC v2 should verify, got {v.reason}:{v.details}")

    wrapper = Act(
        id="",
        version=1,
        created_at=deterministic_iso(step=101),
        kind="concept_csv",
        match={},
        program=[
            Instruction("CSV_GET_INPUT", {"name": "text", "out": "t"}),
            Instruction("CSV_CALL", {"concept_id": str(callee.id), "out": "n", "bind": {"text": "t"}}),
            Instruction("CSV_RETURN", {"var": "n"}),
        ],
        evidence={
            "name": "concept_csv_v0",
            "interface": {"input_schema": {"text": "str"}, "output_schema": {"value": "int"}, "validator_id": "int_value_exact"},
            "meta": {"title": "wrapper_v62_smoke"},
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

    store_w = ActStore()
    store_w.add(callee)
    store_w.add(wrapper)
    cert_w = build_concept_pcc_certificate_v2(wrapper, store=store_w, mined_from={"kind": "wrapper_smoke"}, test_vectors=vecs, ethics_verdict=validate_act_for_promotion(wrapper).to_dict(), uncertainty_policy="no_ic")
    wrapper.evidence.setdefault("certificate_v2", cert_w)
    wrapper.evidence["certificate_v2"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(wrapper)
    vw = verify_concept_pcc_v2(wrapper, store_w)
    if not bool(vw.ok):
        _fail(f"FAIL: wrapper PCC v2 should verify, got {vw.reason}:{vw.details}")

    # Tamper callee program to force mismatch.
    tampered_store = ActStore(acts=dict(store_w.acts), next_id_int=int(store_w.next_id_int))
    callee_t = tampered_store.get(callee.id)
    if callee_t is None:
        _fail("FAIL: callee missing for tamper")
    for idx, ins in enumerate(list(callee_t.program)):
        if str(ins.op) == "CSV_PRIMITIVE":
            new_args = dict(ins.args or {})
            new_args["fn"] = "strip_one_leading_zero"
            callee_t.program[idx] = Instruction("CSV_PRIMITIVE", new_args)
            break
    vt = verify_concept_pcc_v2(wrapper, tampered_store)
    if bool(vt.ok) or str(vt.reason) != "callee_program_sha256_mismatch":
        _fail(f"FAIL: expected callee_program_sha256_mismatch, got {vt.reason}:{vt.details}")

    print(json.dumps({"ok": True, "candidates": len(cands), "chat_hash": h_base}, ensure_ascii=False))


if __name__ == "__main__":
    main()
