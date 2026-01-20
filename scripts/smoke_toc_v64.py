#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.ethics import validate_act_for_promotion
from atos_core.proof import (
    act_body_sha256_placeholder,
    build_concept_pcc_certificate_v2,
    certificate_body_sha256,
    certificate_sha256,
    verify_concept_pcc_v2,
)
from atos_core.store import ActStore
from atos_core.toc import detect_duplicate, toc_eval, verify_concept_toc_v1


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"ERROR: path already exists: {path}")


def stable_act_id(prefix: str, body: Dict[str, Any]) -> str:
    return f"{prefix}{sha256_hex(canonical_json_dumps(body).encode('utf-8'))[:12]}"


def make_extract_int_concept(*, step: int, title: str) -> Act:
    iface = {"input_schema": {"text": "str"}, "output_schema": {"value": "int"}, "validator_id": "int_value_exact"}
    ev = {"name": "concept_csv_v0", "interface": dict(iface), "meta": {"title": str(title)}}
    body = {
        "kind": "concept_csv",
        "version": 1,
        "match": {},
        "program": [
            Instruction("CSV_GET_INPUT", {"name": "text", "out": "t"}).to_dict(),
            Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["t"], "out": "d"}).to_dict(),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "n"}).to_dict(),
            Instruction("CSV_RETURN", {"var": "n"}).to_dict(),
        ],
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
        program=[
            Instruction("CSV_GET_INPUT", {"name": "text", "out": "t"}),
            Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["t"], "out": "d"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "n"}),
            Instruction("CSV_RETURN", {"var": "n"}),
        ],
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
        _fail(f"missing acts.jsonl: {base_acts}")
    store = ActStore.load_jsonl(base_acts)

    # Callee concept: should pass ToC across A/B (both are strings but different "domains").
    c = make_extract_int_concept(step=1, title="toc_pass")
    store.add(c)

    ethics = validate_act_for_promotion(c)
    if not bool(ethics.ok):
        _fail(f"ethics rejected concept: {ethics.reason}:{ethics.violated_laws}")

    vecA = [
        {"inputs": {"text": "id=42"}, "expected": 42, "expected_output_text": "42"},
        {"inputs": {"text": "abc0123"}, "expected": 123, "expected_output_text": "123"},
        {"inputs": {"text": "x9y7"}, "expected": 9, "expected_output_text": "9"},
    ]
    vecB = [
        {"inputs": {"text": "foo17bar"}, "expected": 17, "expected_output_text": "17"},
        {"inputs": {"text": "bar25"}, "expected": 25, "expected_output_text": "25"},
        {"inputs": {"text": "A=0005"}, "expected": 5, "expected_output_text": "5"},
    ]

    cert = build_concept_pcc_certificate_v2(
        c,
        store=store,
        mined_from={"kind": "smoke_toc_v64"},
        test_vectors=list(vecA),
        ethics_verdict=ethics.to_dict(),
        uncertainty_policy="no_ic",
    )
    toc = toc_eval(concept_act=c, vectors_A=vecA, vectors_B=vecB, store=store, domain_A="A", domain_B="B")
    cert["toc_v1"] = toc
    cert["hashes"]["certificate_body_sha256"] = certificate_body_sha256(cert)
    cert["hashes"]["certificate_sha256"] = certificate_sha256(cert)
    c.evidence.setdefault("certificate_v2", cert)
    c.evidence["certificate_v2"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(c)

    pcc = verify_concept_pcc_v2(c, store)
    if not bool(pcc.ok):
        _fail(f"PCC v2 should pass, got {pcc.reason}:{pcc.details}")
    tv = verify_concept_toc_v1(c, store=store)
    if not bool(tv.ok):
        _fail(f"ToC should pass, got {tv.reason}:{tv.details}")

    # Fail case: pass A but fail B due to missing vectors_B (treated as no transfer).
    c2 = make_extract_int_concept(step=2, title="toc_fail_missing_B")
    store.add(c2)
    ethics2 = validate_act_for_promotion(c2)
    cert2 = build_concept_pcc_certificate_v2(
        c2,
        store=store,
        mined_from={"kind": "smoke_toc_v64"},
        test_vectors=list(vecA),
        ethics_verdict=ethics2.to_dict(),
        uncertainty_policy="no_ic",
    )
    toc2 = toc_eval(concept_act=c2, vectors_A=vecA, vectors_B=[], store=store, domain_A="A", domain_B="B")
    cert2["toc_v1"] = toc2
    cert2["hashes"]["certificate_body_sha256"] = certificate_body_sha256(cert2)
    cert2["hashes"]["certificate_sha256"] = certificate_sha256(cert2)
    c2.evidence.setdefault("certificate_v2", cert2)
    c2.evidence["certificate_v2"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(c2)
    tv2 = verify_concept_toc_v1(c2, store=store)
    if bool(tv2.ok):
        _fail("Expected ToC to fail when vectors_B are missing")

    # Clone detection: same program/interface but different id should be duplicate.
    clone = Act.from_dict(c.to_dict())
    clone.evidence = dict(clone.evidence)
    clone_meta = dict(clone.evidence.get("meta") or {})
    clone_meta["title"] = "clone_variant_meta_only"
    clone.evidence["meta"] = clone_meta
    body_clone = {
        "kind": "concept_csv",
        "version": 1,
        "match": {},
        "program": [ins.to_dict() for ins in clone.program],
        "evidence": clone.evidence,
        "deps": [],
        "active": True,
    }
    clone.id = stable_act_id("act_concept_csv_", body_clone)
    dup = detect_duplicate(clone, existing=[c], similarity_threshold=0.95)
    if dup is None:
        _fail("Expected duplicate detection to flag clone")

    out = {
        "ok": True,
        "toc_pass": tv.to_dict(),
        "toc_fail": tv2.to_dict(),
        "duplicate": dup,
    }
    out_path = os.path.join(args.out, "smoke_result.json")
    ensure_absent(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
