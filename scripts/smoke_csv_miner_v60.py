#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.csv_miner import materialize_concept_act_from_candidate, mine_csv_candidates
from atos_core.engine import Engine, EngineConfig
from atos_core.ethics import validate_act_for_promotion
from atos_core.proof import (
    act_body_sha256_placeholder,
    build_concept_pcc_certificate_v1,
    verify_concept_pcc,
)
from atos_core.store import ActStore


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
        program=[Instruction("CSV_CONST", {"out": "s", "value": str(text)}), Instruction("CSV_RETURN", {"var": "s"})],
        evidence=ev,
        cost={"overhead_bits": 1024},
        deps=[],
        active=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="WORM out dir (must not exist)")
    args = ap.parse_args()

    ensure_absent(args.out)
    os.makedirs(args.out, exist_ok=False)

    # (1) Create a tiny trace file and ensure miner finds scan_digits->digits_to_int.
    traces_dir = os.path.join(args.out, "traces")
    os.makedirs(traces_dir, exist_ok=False)
    csv_exec = os.path.join(traces_dir, "csv_exec.jsonl")

    rows: List[Dict[str, Any]] = []
    for i, text in enumerate(["abc0123", "x9y7", "id=42"]):
        events = [
            {"t": "GET_INPUT", "name": "text", "out": "t"},
            {"t": "PRIMITIVE", "fn": "scan_digits", "in": ["t"], "out": "d"},
            {"t": "PRIMITIVE", "fn": "digits_to_int", "in": ["d"], "out": "n"},
            {"t": "RETURN", "var": "n"},
        ]
        rows.append(
            {
                "run_id": str(args.out),
                "ctx_sig": f"inline␟i={i}",
                "goal_id": f"g{i}",
                "program_sig": sha256_hex(canonical_json_dumps(events).encode("utf-8")),
                "events": events,
                "inputs": {"text": str(text)},
                "inputs_sig": sha256_hex(str(text).encode("utf-8")),
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

    # (2) PCC verify ok for mined candidate.
    store = ActStore()
    concept = materialize_concept_act_from_candidate(
        top,
        step=1,
        store_content_hash_excluding_semantic="dummy",
        title="mined_extract_int_v60_smoke",
    )
    ethics = validate_act_for_promotion(concept)
    if not bool(ethics.ok):
        _fail(f"FAIL: ethics rejected mined concept: {ethics.reason}")
    test_vectors = []
    for ex in top.examples[:3]:
        test_vectors.append(
            {
                "inputs": dict(ex.get("inputs") or {}),
                "expected": ex.get("expected"),
                "expected_output_text": str(ex.get("expected_output_text") or ""),
            }
        )
    cert = build_concept_pcc_certificate_v1(
        concept,
        mined_from={"trace_file_sha256": sha256_hex(open(csv_exec, "rb").read())},
        test_vectors=test_vectors,
        ethics_verdict=ethics.to_dict(),
    )
    concept.evidence.setdefault("certificate_v1", cert)
    concept.evidence["certificate_v1"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(concept)
    v = verify_concept_pcc(concept, store)
    if not bool(v.ok):
        _fail(f"FAIL: PCC verify should pass, got {v.reason}:{v.details}")

    # (3) Tamper test: change expected_output_text -> verify must fail.
    tampered = Act.from_dict(concept.to_dict())
    cert2 = tampered.evidence.get("certificate_v1", {})
    if isinstance(cert2, dict) and isinstance(cert2.get("test_vectors"), list) and cert2["test_vectors"]:
        cert2["test_vectors"][0]["expected_output_text"] = "999999"
    v2 = verify_concept_pcc(tampered, store)
    if bool(v2.ok):
        _fail("FAIL: tampered PCC should fail")

    # (4) Promotion rejects concept that downgrades to IC (strong claim without evidence).
    concept_ic = make_concept_const(title="strong_claim", text="COM CERTEZA PARIS")
    store2 = ActStore()
    store2.add(concept_ic)
    eng = Engine(store2, seed=0, config=EngineConfig())
    out = eng.execute_concept_csv(concept_act_id=concept_ic.id, inputs={}, expected=None, step=0)
    meta = out.get("meta") if isinstance(out, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    unc = meta.get("uncertainty") if isinstance(meta.get("uncertainty"), dict) else {}
    if str(unc.get("mode_out") or "") != "IC":
        _fail(f"FAIL: expected IR->IC downgrade, got {unc}")

    # (5) Ethics fail-closed blocks LO-02/LO-06 (synthetic).
    concept_bad = make_concept_const(title="sentience", text="EU SOU CONSCIENTE")
    store3 = ActStore()
    store3.add(concept_bad)
    eng2 = Engine(store3, seed=0, config=EngineConfig())
    out_bad = eng2.execute_concept_csv(concept_act_id=concept_bad.id, inputs={}, expected=None, step=0)
    meta_bad = out_bad.get("meta") if isinstance(out_bad, dict) else {}
    meta_bad = meta_bad if isinstance(meta_bad, dict) else {}
    eth_bad = meta_bad.get("ethics") if isinstance(meta_bad.get("ethics"), dict) else {}
    if bool(eth_bad.get("ok", True)):
        _fail(f"FAIL: expected ethics fail-closed, got {eth_bad}")

    print(json.dumps({"ok": True, "top_candidate_sig": top.candidate_sig}, ensure_ascii=False))


if __name__ == "__main__":
    main()

