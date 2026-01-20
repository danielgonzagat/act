#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.csv_miner import materialize_concept_act_from_candidate, mine_csv_candidates
from atos_core.engine import Engine, EngineConfig
from atos_core.ethics import validate_act_for_promotion
from atos_core.proof import (
    act_body_sha256_placeholder,
    build_concept_pcc_certificate_v2,
    verify_concept_pcc_v2,
)
from atos_core.store import ActStore
from atos_core.suite import CHAT_DIALOGUES_20X3, run_chat_suite


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"ERROR: path already exists: {path}")


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ensure_absent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(canonical_json_dumps(r))
            f.write("\n")
    os.replace(tmp, path)
    return sha256_hex(open(path, "rb").read())


def _transcript_hash(transcripts: List[Dict[str, Any]]) -> str:
    full = "\n".join(str(t.get("full_text") or "") for t in transcripts)
    return sha256_hex(full.encode("utf-8"))


def make_concept_const(*, title: str, text: str) -> Act:
    iface = {"input_schema": {}, "output_schema": {"value": "str"}, "validator_id": ""}
    ev = {"name": "concept_csv_v0", "interface": iface, "meta": {"title": str(title)}}
    body = {
        "kind": "concept_csv",
        "version": 1,
        "match": {},
        "program": [
            Instruction("CSV_CONST", {"out": "s", "value": str(text)}).to_dict(),
            Instruction("CSV_RETURN", {"var": "s"}).to_dict(),
        ],
        "evidence": ev,
        "deps": [],
        "active": True,
    }
    act_id = f"act_concept_csv_{sha256_hex(canonical_json_dumps(body).encode('utf-8'))[:12]}"
    return Act(
        id=act_id,
        version=1,
        created_at=deterministic_iso(step=0),
        kind="concept_csv",
        match={},
        program=[
            Instruction("CSV_CONST", {"out": "s", "value": str(text)}),
            Instruction("CSV_RETURN", {"var": "s"}),
        ],
        evidence=ev,
        cost={"overhead_bits": 1024},
        deps=[],
        active=True,
    )


def stable_act_id(prefix: str, body: Dict[str, Any]) -> str:
    return f"{prefix}{sha256_hex(canonical_json_dumps(body).encode('utf-8'))[:12]}"


def make_goal_act(*, concept_id: str, title: str, expected: Any) -> Act:
    ev = {
        "name": "goal_v0",
        "meta": {"title": str(title)},
        "goal": {"priority": 10, "concept_id": str(concept_id), "inputs": {}, "expected": expected},
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
        created_at=deterministic_iso(step=0),
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
    ap.add_argument("--acts_run", required=True, help="Base run with acts.jsonl for chat-suite smoke")
    ap.add_argument("--out", required=True, help="WORM out dir (must not exist)")
    args = ap.parse_args()

    ensure_absent(args.out)
    os.makedirs(args.out, exist_ok=False)

    traces_dir = os.path.join(args.out, "traces")
    os.makedirs(traces_dir, exist_ok=False)

    # (1) Miner mines from real engine-style INS events.
    csv_exec = os.path.join(traces_dir, "csv_exec.jsonl")
    rows: List[Dict[str, Any]] = []
    for i, text in enumerate(["abc0123", "x9y7", "id=42"]):
        # Simulated engine.execute_concept_csv INS trace.
        events = [
            {"t": "INS", "op": "CSV_GET_INPUT", "name": "text", "out": "t"},
            {"t": "INS", "op": "CSV_PRIMITIVE", "fn": "scan_digits", "in": ["t"], "out": "d"},
            {"t": "INS", "op": "CSV_PRIMITIVE", "fn": "digits_to_int", "in": ["d"], "out": "n"},
            {"t": "INS", "op": "CSV_RETURN", "var": "n"},
        ]
        inputs = {"text": str(text)}
        rows.append(
            {
                "run_id": str(args.out),
                "ctx_sig": f"engine_ins␟i={i}",
                "goal_id": f"g{i}",
                "program_sig": sha256_hex(canonical_json_dumps(events).encode("utf-8")),
                "events": events,
                "inputs": dict(inputs),
                "inputs_sig": sha256_hex(canonical_json_dumps(inputs).encode("utf-8")),
                "output_text": "",
                "output_sig": "",
            }
        )
    write_jsonl(csv_exec, rows)
    cands = mine_csv_candidates(csv_exec, min_ops=2, max_ops=6)
    if not cands:
        _fail("FAIL: miner produced 0 candidates")
    top = cands[0]
    fns = [op.get("fn") for op in top.ops]
    if fns != ["scan_digits", "digits_to_int"]:
        _fail(f"FAIL: expected ops scan_digits->digits_to_int, got {fns}")

    # (2) Multi-promotion under budget is deterministic (promote until max bits).
    store = ActStore()
    concept1 = materialize_concept_act_from_candidate(
        top,
        step=1,
        store_content_hash_excluding_semantic="dummy",
        title="mined_extract_int_v61_smoke",
        overhead_bits=1024,
        meta={"builder": "smoke_v61"},
    )
    # Create a second concept candidate by reusing the same program but different id/title.
    concept2 = Act.from_dict(concept1.to_dict())
    concept2.evidence = dict(concept2.evidence)
    meta2 = dict(concept2.evidence.get("meta") or {})
    meta2["title"] = "mined_extract_int_v61_smoke_2"
    concept2.evidence["meta"] = meta2
    body2 = concept2.to_dict()
    concept2.id = stable_act_id("act_concept_csv_", {k: body2[k] for k in ("kind", "version", "match", "program", "evidence", "deps", "active")})

    concepts_ranked = [concept1, concept2]

    def _budget_select(max_bits: int) -> List[str]:
        used = 0
        kept: List[str] = []
        for c in concepts_ranked:
            ob = int(c.cost.get("overhead_bits", 1024))
            if used + ob > int(max_bits):
                continue
            used += ob
            kept.append(str(c.id))
        return kept

    kept1 = _budget_select(1024)
    kept2 = _budget_select(2048)
    if len(kept1) != 1 or len(kept2) != 2:
        _fail(f"FAIL: budget selection unexpected kept1={kept1} kept2={kept2}")
    if kept1 != _budget_select(1024) or kept2 != _budget_select(2048):
        _fail("FAIL: budget selection not deterministic")

    # (3) PCC v2 passes for wrapper (CALL chain) and fails on callee tamper.
    ethics = validate_act_for_promotion(concept1)
    if not bool(ethics.ok):
        _fail(f"FAIL: ethics rejected concept: {ethics.reason}")
    vecs = []
    for ex in top.examples[:3]:
        vecs.append(
            {
                "inputs": dict(ex.get("inputs") or {}),
                "expected": ex.get("expected"),
                "expected_output_text": str(ex.get("expected_output_text") or ""),
            }
        )
    cert1 = build_concept_pcc_certificate_v2(
        concept1,
        store=store,
        mined_from={"trace_file": csv_exec},
        test_vectors=vecs,
        ethics_verdict=ethics.to_dict(),
        uncertainty_policy="no_ic",
    )
    concept1.evidence.setdefault("certificate_v2", cert1)
    concept1.evidence["certificate_v2"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(concept1)
    v1 = verify_concept_pcc_v2(concept1, store)
    if not bool(v1.ok):
        _fail(f"FAIL: PCC v2 verify should pass for callee, got {v1.reason}:{v1.details}")

    wrapper = Act(
        id="",
        version=1,
        created_at=deterministic_iso(step=2),
        kind="concept_csv",
        match={},
        program=[
            Instruction("CSV_GET_INPUT", {"name": "text", "out": "t"}),
            Instruction("CSV_CALL", {"concept_id": str(concept1.id), "out": "n", "bind": {"text": "t"}}),
            Instruction("CSV_RETURN", {"var": "n"}),
        ],
        evidence={
            "name": "concept_csv_v0",
            "interface": {"input_schema": {"text": "str"}, "output_schema": {"value": "int"}, "validator_id": "int_value_exact"},
            "meta": {"title": "wrapper_smoke_v61"},
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

    store4 = ActStore()
    store4.add(concept1)
    store4.add(wrapper)
    cert_w = build_concept_pcc_certificate_v2(
        wrapper,
        store=store4,
        mined_from={"kind": "wrapper_smoke"},
        test_vectors=vecs,
        ethics_verdict=validate_act_for_promotion(wrapper).to_dict(),
        uncertainty_policy="no_ic",
    )
    wrapper.evidence.setdefault("certificate_v2", cert_w)
    wrapper.evidence["certificate_v2"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(wrapper)
    vw = verify_concept_pcc_v2(wrapper, store4)
    if not bool(vw.ok):
        _fail(f"FAIL: PCC v2 verify should pass for wrapper, got {vw.reason}:{vw.details}")

    # Tamper callee program: change one op; wrapper verify must fail on callee hash mismatch.
    tampered_store = ActStore(acts=dict(store4.acts), next_id_int=int(store4.next_id_int))
    callee_t = tampered_store.get(concept1.id)
    if callee_t is None:
        _fail("FAIL: callee missing in tampered_store")
    # Replace the first primitive fn to change program_sha256 deterministically.
    for idx, ins in enumerate(list(callee_t.program)):
        if str(ins.op) == "CSV_PRIMITIVE":
            new_args = dict(ins.args or {})
            new_args["fn"] = "strip_one_leading_zero"
            callee_t.program[idx] = Instruction("CSV_PRIMITIVE", new_args)
            break
    vt = verify_concept_pcc_v2(wrapper, tampered_store)
    if bool(vt.ok) or str(vt.reason) != "callee_program_sha256_mismatch":
        _fail(f"FAIL: expected callee_program_sha256_mismatch, got {vt.reason}:{vt.details}")

    # (4) Scheduler invariance: chat output hash must be identical with/without goal shadow.
    base_acts = os.path.join(args.acts_run, "acts.jsonl")
    if not os.path.exists(base_acts):
        _fail(f"FAIL: missing acts.jsonl: {base_acts}")
    chat_store = ActStore.load_jsonl(base_acts)
    chat_store.add(make_concept_const(title="goal_const", text="X"))
    goal1 = make_goal_act(concept_id=list(chat_store.by_kind("concept_csv"))[-1].id, title="g1", expected="X")
    goal2 = make_goal_act(concept_id=list(chat_store.by_kind("concept_csv"))[-1].id, title="g2", expected="X")
    chat_store.add(goal1)
    chat_store.add(goal2)

    engine_a = Engine(chat_store, seed=0, config=EngineConfig())
    t1, _ = run_chat_suite(engine_a, dialogues=CHAT_DIALOGUES_20X3[:1], max_new_tokens=20, goal_shadow_log_path=None)
    h1 = _transcript_hash(t1)

    goal_shadow_path = os.path.join(traces_dir, "goal_shadow.jsonl")
    engine_b = Engine(chat_store, seed=0, config=EngineConfig())
    t2, _ = run_chat_suite(
        engine_b,
        dialogues=CHAT_DIALOGUES_20X3[:1],
        max_new_tokens=20,
        goal_shadow_log_path=goal_shadow_path,
        goal_shadow_max_goals_per_turn=1,
    )
    h2 = _transcript_hash(t2)
    if h1 != h2:
        _fail("FAIL: goal shadow scheduler changed chat transcript hash")
    try:
        saw_skip = False
        with open(goal_shadow_path, "r", encoding="utf-8") as f:
            for line in f:
                if '"kind":"skip"' in line:
                    saw_skip = True
                    break
        if not saw_skip:
            _fail("FAIL: expected at least one scheduler skip record in goal_shadow log")
    except Exception:
        _fail("FAIL: could not read goal_shadow log for scheduler check")

    # (5) Ethics fail-closed blocks LO-02/LO-06 (synthetic).
    concept_bad = make_concept_const(title="sentience", text="EU SOU CONSCIENTE")
    store_bad = ActStore()
    store_bad.add(concept_bad)
    eng_bad = Engine(store_bad, seed=0, config=EngineConfig())
    out_bad = eng_bad.execute_concept_csv(concept_act_id=concept_bad.id, inputs={}, expected=None, step=0)
    meta_bad = out_bad.get("meta") if isinstance(out_bad, dict) else {}
    meta_bad = meta_bad if isinstance(meta_bad, dict) else {}
    eth_bad = meta_bad.get("ethics") if isinstance(meta_bad.get("ethics"), dict) else {}
    if bool(eth_bad.get("ok", True)):
        _fail(f"FAIL: expected ethics fail-closed, got {eth_bad}")

    print(json.dumps({"ok": True, "top_candidate_sig": top.candidate_sig}, ensure_ascii=False))


if __name__ == "__main__":
    main()
