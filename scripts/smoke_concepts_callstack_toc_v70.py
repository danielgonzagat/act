#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.concept_registry_v70 import (
    ConceptRegistryV70,
    interface_sig_from_act,
    program_sha256_from_act,
)
from atos_core.engine import Engine, EngineConfig
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


def _concept_sig_from_fields(*, concept_id: str, interface_sig: str, program_sha256: str) -> str:
    return sha256_canon(
        {"concept_id": str(concept_id), "interface_sig": str(interface_sig), "program_sha256": str(program_sha256)}
    )


def _assert_call_event(event: Dict[str, Any], *, store: ActStore) -> None:
    for k in (
        "concept_id",
        "concept_sig",
        "interface_sig",
        "program_sha256",
        "bindings",
        "bindings_sig",
        "call_depth",
        "return",
        "return_sig",
        "ok",
        "cost",
        "evidence_refs",
    ):
        if k not in event:
            _fail(f"ERROR: missing concept_call field: {k}")

    if not isinstance(event.get("bindings"), dict):
        _fail("ERROR: concept_call.bindings_not_dict")
    if not isinstance(event.get("return"), dict):
        _fail("ERROR: concept_call.return_not_dict")
    if not isinstance(event.get("evidence_refs"), list):
        _fail("ERROR: concept_call.evidence_refs_not_list")

    bindings_sig = str(event.get("bindings_sig") or "")
    want_bind_sig = sha256_canon(event.get("bindings"))
    if bindings_sig != want_bind_sig:
        _fail(f"ERROR: bindings_sig_mismatch: want={want_bind_sig} got={bindings_sig}")

    return_sig = str(event.get("return_sig") or "")
    want_ret_sig = sha256_canon(event.get("return"))
    if return_sig != want_ret_sig:
        _fail(f"ERROR: return_sig_mismatch: want={want_ret_sig} got={return_sig}")

    cid = str(event.get("concept_id") or "")
    act = store.get_concept_act(cid)
    if act is None:
        _fail(f"ERROR: concept_call.concept_act_not_found: {cid}")

    want_iface = interface_sig_from_act(act)
    want_prog = program_sha256_from_act(act)
    if str(event.get("interface_sig") or "") != want_iface:
        _fail("ERROR: interface_sig_mismatch")
    if str(event.get("program_sha256") or "") != want_prog:
        _fail("ERROR: program_sha256_mismatch")

    want_csig = _concept_sig_from_fields(concept_id=cid, interface_sig=want_iface, program_sha256=want_prog)
    if str(event.get("concept_sig") or "") != want_csig:
        _fail("ERROR: concept_sig_mismatch")


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    # Build a small deterministic store with concept_csv acts only.
    store = ActStore()

    iface_xy = {"x": "str", "y": "str"}
    out_schema = {"out": "str"}

    # Callee: normalize a single string number -> canonical digits (no leading zeros).
    normalize_id = "concept_v70_normalize_x_v0"
    normalize = make_concept_act(
        act_id=normalize_id,
        input_schema={"x": "str"},
        output_schema=out_schema,
        validator_id="text_exact",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "x", "out": "x"}),
            Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["x"], "out": "d"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "i"}),
            Instruction("CSV_PRIMITIVE", {"fn": "int_to_digits", "in": ["i"], "out": "out"}),
            Instruction("CSV_RETURN", {"var": "out"}),
        ],
    )
    store.add(normalize)

    # Candidate that FAILS transfer: ignores y (passes A, fails B).
    fail_id = "concept_v70_fail_transfer_v0"
    fail_transfer = make_concept_act(
        act_id=fail_id,
        input_schema=iface_xy,
        output_schema=out_schema,
        validator_id="text_exact",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "x", "out": "x"}),
            Instruction("CSV_CALL", {"concept_id": normalize_id, "bind": {"x": "x"}, "out": "nx"}),
            Instruction("CSV_RETURN", {"var": "nx"}),
        ],
    )
    store.add(fail_transfer)

    # Candidate that PASSES ToC: add two numbers using nested normalization calls.
    good_id = "concept_v70_add_xy_v0"
    good = make_concept_act(
        act_id=good_id,
        input_schema=iface_xy,
        output_schema=out_schema,
        validator_id="text_exact",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "x", "out": "x"}),
            Instruction("CSV_GET_INPUT", {"name": "y", "out": "y"}),
            Instruction("CSV_CALL", {"concept_id": normalize_id, "bind": {"x": "x"}, "out": "nx"}),
            Instruction("CSV_CALL", {"concept_id": normalize_id, "bind": {"x": "y"}, "out": "ny"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["nx"], "out": "ix"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["ny"], "out": "iy"}),
            Instruction("CSV_PRIMITIVE", {"fn": "add_int", "in": ["ix", "iy"], "out": "sum"}),
            Instruction("CSV_PRIMITIVE", {"fn": "int_to_digits", "in": ["sum"], "out": "out"}),
            Instruction("CSV_RETURN", {"var": "out"}),
        ],
    )
    store.add(good)

    # Clone twin: same structure/interface, different id.
    clone_id = "concept_v70_add_xy_clone_v0"
    clone = make_concept_act(
        act_id=clone_id,
        input_schema=iface_xy,
        output_schema=out_schema,
        validator_id="text_exact",
        program=list(good.program),
    )
    store.add(clone)

    # Registry (append-only/WORM).
    registry_dir = os.path.join(out_dir, "registry")
    reg = ConceptRegistryV70(registry_dir, toc_fail_threshold=2, similarity_threshold=0.95)

    vectors_A = [
        {"inputs": {"x": "007", "y": "0"}, "expected": "7", "expected_output_text": "7"},
        {"inputs": {"x": "0004", "y": "0"}, "expected": "4", "expected_output_text": "4"},
        {"inputs": {"x": "42", "y": "0"}, "expected": "42", "expected_output_text": "42"},
    ]
    vectors_B = [
        {"inputs": {"x": "4", "y": "8"}, "expected": "12", "expected_output_text": "12"},
        {"inputs": {"x": "09", "y": "1"}, "expected": "10", "expected_output_text": "10"},
        {"inputs": {"x": "100", "y": "23"}, "expected": "123", "expected_output_text": "123"},
    ]

    # (1) Pass A, fail B => NOT promoted.
    r1 = reg.attempt_promote_with_toc(
        step=1,
        candidate=fail_transfer,
        store=store,
        vectors_A=vectors_A,
        vectors_B=vectors_B,
        domain_A="A_normalize",
        domain_B="B_add",
        existing_for_duplicates=[good],
    )
    toc1 = r1.get("toc") if isinstance(r1.get("toc"), dict) else {}
    if not bool(toc1.get("pass_A", False)):
        _fail("ERROR: expected fail_transfer pass_A==True")
    if bool(toc1.get("pass_B", False)):
        _fail("ERROR: expected fail_transfer pass_B==False")
    c1 = r1.get("concept") if isinstance(r1.get("concept"), dict) else {}
    if str(c1.get("concept_state") or "") == "ACTIVE":
        _fail("ERROR: fail_transfer_should_not_be_active")

    # (4) GC on repeated transfer failure (threshold=2).
    r2 = reg.attempt_promote_with_toc(
        step=2,
        candidate=fail_transfer,
        store=store,
        vectors_A=vectors_A,
        vectors_B=vectors_B,
        domain_A="A_normalize",
        domain_B="B_add",
        existing_for_duplicates=[good],
    )
    c2 = r2.get("concept") if isinstance(r2.get("concept"), dict) else {}
    if str(c2.get("concept_state") or "") != "DEAD":
        _fail("ERROR: expected fail_transfer DEAD after toc_fail_threshold")

    # (3) ToC OK => ACTIVE.
    r_good = reg.attempt_promote_with_toc(
        step=3,
        candidate=good,
        store=store,
        vectors_A=vectors_A,
        vectors_B=vectors_B,
        domain_A="A_normalize",
        domain_B="B_add",
        existing_for_duplicates=[],
    )
    c_good = r_good.get("concept") if isinstance(r_good.get("concept"), dict) else {}
    if str(c_good.get("concept_state") or "") != "ACTIVE":
        _fail("ERROR: expected good concept ACTIVE")

    # (2) Clone detection blocks twin.
    r_clone = reg.attempt_promote_with_toc(
        step=4,
        candidate=clone,
        store=store,
        vectors_A=vectors_A,
        vectors_B=vectors_B,
        domain_A="A_normalize",
        domain_B="B_add",
        existing_for_duplicates=[good],
    )
    c_clone = r_clone.get("concept") if isinstance(r_clone.get("concept"), dict) else {}
    if str(c_clone.get("concept_state") or "") == "ACTIVE":
        _fail("ERROR: expected clone NOT ACTIVE")
    dup = r_clone.get("duplicate") if isinstance(r_clone.get("duplicate"), dict) else {}
    if str(dup.get("reason") or "") not in {"duplicate_exact", "duplicate_similar"}:
        _fail(f"ERROR: expected duplicate_* reason, got: {dup}")
    if str(dup.get("other_id") or "") != str(good_id):
        _fail("ERROR: expected duplicate other_id == good_id")

    chains = reg.verify_chains()
    if not (bool(chains.get("concept_registry_chain_ok")) and bool(chains.get("concept_registry_evidence_chain_ok"))):
        _fail(f"ERROR: registry_chain_verify_failed: {chains}")

    # (4) CALL/RETURN semantic trace with call stack, sigs, anti-aliasing.
    engine = Engine(store, seed=int(seed), config=EngineConfig(enable_contracts=False))
    bindings = {"x": "0004", "y": "0008"}
    out = engine.execute_concept_csv(concept_act_id=good_id, inputs=bindings, expected="12", step=5)
    trace = out.get("trace") if isinstance(out, dict) else None
    trace = trace if isinstance(trace, dict) else {}
    calls = trace.get("concept_calls")
    if not isinstance(calls, list) or not calls:
        _fail("ERROR: missing trace.concept_calls")

    # anti-aliasing: mutate bindings after execution must not affect snapshots.
    bindings["x"] = "MUTATED"
    bindings["y"] = "MUTATED"

    depths: List[int] = []
    for ev in calls:
        if not isinstance(ev, dict):
            _fail("ERROR: concept_calls element not dict")
        _assert_call_event(ev, store=store)
        depths.append(int(ev.get("call_depth", -1)))

    if min(depths) != 0:
        _fail(f"ERROR: expected min call_depth == 0, got {depths}")
    if max(depths) < 1:
        _fail(f"ERROR: expected nested call_depth >=1, got {depths}")

    # Anti-aliasing check for bindings snapshot at root call.
    root = min((ev for ev in calls if isinstance(ev, dict)), key=lambda e: int(e.get("call_depth", 999)))
    root_bind = root.get("bindings") if isinstance(root.get("bindings"), dict) else {}
    if str(root_bind.get("x") or "") != "0004" or str(root_bind.get("y") or "") != "0008":
        _fail("ERROR: bindings_snapshot_mutated_via_aliasing")

    summary = {
        "seed": int(seed),
        "registry": {
            "fail_transfer_state_after_try2": str(c2.get("concept_state") or ""),
            "good_state": str(c_good.get("concept_state") or ""),
            "clone_state": str(c_clone.get("concept_state") or ""),
            "clone_duplicate_reason": str(dup.get("reason") or ""),
        },
        "trace": {
            "concept_calls_total": int(len(calls)),
            "call_depths": list(depths),
            "sample_root_bindings_sig": str(root.get("bindings_sig") or ""),
            "sample_root_return_sig": str(root.get("return_sig") or ""),
        },
        "chains": dict(chains),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", default="results/run_smoke_concepts_callstack_toc_v70")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_base = str(args.out_base)
    seed = int(args.seed)

    results: Dict[str, Any] = {"seed": seed, "tries": {}}
    summary_shas: List[str] = []

    for t in (1, 2):
        out_dir = f"{out_base}_try{t}"
        ensure_absent(out_dir)
        os.makedirs(out_dir, exist_ok=False)

        summ = smoke_try(out_dir=out_dir, seed=seed)
        eval_path = os.path.join(out_dir, "eval.json")
        eval_sha = write_json(eval_path, summ)

        core = dict(summ)
        core["sha256_eval_json"] = str(eval_sha)
        summary_sha = sha256_text(canonical_json_dumps(core))
        smoke = {"summary": core, "determinism": {"summary_sha256": str(summary_sha)}}
        smoke_path = os.path.join(out_dir, "smoke_summary.json")
        smoke_sha = write_json(smoke_path, smoke)

        summary_shas.append(str(summary_sha))
        results["tries"][f"try{t}"] = {
            "out_dir": out_dir,
            "eval_json": {"path": eval_path, "sha256": eval_sha},
            "smoke_summary_json": {"path": smoke_path, "sha256": smoke_sha},
            "summary_sha256": summary_sha,
        }

    if not (len(summary_shas) == 2 and summary_shas[0] == summary_shas[1]):
        _fail(f"ERROR: determinism mismatch: {summary_shas}")
    results["determinism"] = {"ok": True, "summary_sha256": summary_shas[0]}
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

