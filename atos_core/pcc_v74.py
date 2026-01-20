from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import Act, Instruction, canonical_json_dumps, sha256_hex
from .concept_registry_v70 import interface_sig_from_act, program_sha256_from_act
from .engine import Engine, EngineConfig
from .store import ActStore


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _safe_deepcopy(obj: Any) -> Any:
    try:
        return copy.deepcopy(obj)
    except Exception:
        if isinstance(obj, dict):
            return dict(obj)
        if isinstance(obj, list):
            return list(obj)
        return obj


def certificate_sig_v2(cert: Dict[str, Any]) -> str:
    body = dict(cert) if isinstance(cert, dict) else {}
    body.pop("certificate_sig", None)
    return _stable_hash_obj(body)


def _extract_call_deps_from_program(program: Sequence[Instruction]) -> List[str]:
    callees: List[str] = []
    for ins in program:
        if str(getattr(ins, "op", "")) != "CSV_CALL":
            continue
        args = getattr(ins, "args", None)
        args = args if isinstance(args, dict) else {}
        cid = str(args.get("concept_id") or "")
        if cid and cid not in callees:
            callees.append(cid)
    callees.sort(key=str)
    return callees


def _clone_store_with_candidate(*, store_base: ActStore, candidate_act: Act) -> ActStore:
    store2 = ActStore()
    for act_id in sorted(store_base.acts.keys(), key=str):
        act = store_base.acts[act_id]
        store2.add(Act.from_dict(act.to_dict()))
    store2.add(Act.from_dict(candidate_act.to_dict()))
    return store2


def build_certificate_v2(
    *,
    candidate_act: Act,
    store_base: ActStore,
    mined_from: Dict[str, Any],
    vector_specs: Sequence[Dict[str, Any]],
    seed: int = 0,
) -> Dict[str, Any]:
    """
    PCC v2 (V74): proof-carrying certificate with call_deps supply-chain integrity.

    vector_specs: each item minimally:
      {"inputs": {...}, "expected": ..., "context_id": "..."}  (context_id optional)
    """
    if str(getattr(candidate_act, "kind", "")) != "concept_csv":
        raise ValueError("candidate_wrong_kind")

    store_hash_base = str(store_base.content_hash())

    # Candidate integrity fields.
    iface = {}
    if isinstance(candidate_act.evidence, dict):
        iface = candidate_act.evidence.get("interface") if isinstance(candidate_act.evidence.get("interface"), dict) else {}
    iface = iface if isinstance(iface, dict) else {}
    candidate_interface = {
        "input_schema": dict(iface.get("input_schema", {})),
        "output_schema": dict(iface.get("output_schema", {})),
        "validator_id": str(iface.get("validator_id") or ""),
    }
    cand_iface_sig = interface_sig_from_act(candidate_act)
    cand_prog_sha = program_sha256_from_act(candidate_act)

    # call_deps: direct CSV_CALL dependencies (deterministic order).
    call_deps: List[Dict[str, Any]] = []
    for dep_id in _extract_call_deps_from_program(candidate_act.program or []):
        dep = store_base.get_concept_act(dep_id)
        if dep is None:
            raise ValueError(f"missing_call_dep:{dep_id}")
        call_deps.append(
            {
                "concept_id": str(dep_id),
                "interface_sig": str(interface_sig_from_act(dep)),
                "program_sha256": str(program_sha256_from_act(dep)),
            }
        )

    # Build test_vectors by executing candidate deterministically.
    store_exec = _clone_store_with_candidate(store_base=store_base, candidate_act=candidate_act)
    engine = Engine(store_exec, seed=int(seed), config=EngineConfig(enable_contracts=False))

    enriched: List[Dict[str, Any]] = []
    for spec in vector_specs:
        if not isinstance(spec, dict):
            continue
        inputs = spec.get("inputs") if isinstance(spec.get("inputs"), dict) else {}
        expected = spec.get("expected")
        ctx = str(spec.get("context_id") or "")
        expected_sig = _stable_hash_obj(
            {"inputs": {str(k): inputs.get(k) for k in sorted(inputs.keys(), key=str)}, "expected": expected}
        )
        enriched.append({"context_id": ctx, "inputs": dict(inputs), "expected": expected, "expected_sig": str(expected_sig)})

    # Stable order by expected_sig.
    enriched.sort(key=lambda v: str(v.get("expected_sig") or ""))

    test_vectors: List[Dict[str, Any]] = []
    ethics_ok_all = True
    ic_count = 0
    ok_count = 0

    for i, v in enumerate(enriched):
        inputs = v.get("inputs") if isinstance(v.get("inputs"), dict) else {}
        expected = v.get("expected")
        out = engine.execute_concept_csv(
            concept_act_id=str(candidate_act.id),
            inputs={str(k): inputs.get(k) for k in sorted(inputs.keys(), key=str)},
            expected=expected,
            step=int(i),
            max_depth=8,
            max_events=512,
            validate_output=True,
        )
        meta = out.get("meta") if isinstance(out, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        got = str(meta.get("output_text") or out.get("output") or "")
        output_sig = str(meta.get("output_sig") or "")
        ok = bool(meta.get("ok", False))
        validator = meta.get("validator") if isinstance(meta.get("validator"), dict) else {}
        ethics = meta.get("ethics") if isinstance(meta.get("ethics"), dict) else {}
        uncertainty = meta.get("uncertainty") if isinstance(meta.get("uncertainty"), dict) else {}

        ethics_ok = bool(ethics.get("ok", False))
        ethics_ok_all = bool(ethics_ok_all and ethics_ok)
        if str(uncertainty.get("mode_out") or "") == "IC":
            ic_count += 1
        if ok:
            ok_count += 1

        test_vectors.append(
            {
                "context_id": str(v.get("context_id") or ""),
                "inputs": {str(k): inputs.get(k) for k in sorted(inputs.keys(), key=str)},
                "expected": expected,
                "expected_sig": str(v.get("expected_sig") or ""),
                "got": str(got),
                "ok": bool(ok),
                "output_sig": str(output_sig),
                "validator": _safe_deepcopy(validator),
                "ethics": _safe_deepcopy(ethics),
                "uncertainty": _safe_deepcopy(uncertainty),
            }
        )

    cert: Dict[str, Any] = {
        "schema_version": 2,
        "certificate_kind": "pcc_v74_certificate_v2",
        "store_hash_base": str(store_hash_base),
        "mined_from": _safe_deepcopy(mined_from) if isinstance(mined_from, dict) else {},
        "candidate": {
            "act_id": str(candidate_act.id),
            "kind": str(candidate_act.kind),
            "interface": dict(candidate_interface),
            "interface_sig": str(cand_iface_sig),
            "program_sha256": str(cand_prog_sha),
            "program_len": int(len(candidate_act.program or [])),
        },
        "call_deps": list(call_deps),
        "test_vectors": list(test_vectors),
        "ethics_verdict": {"ok": bool(ethics_ok_all), "mode": "fail_closed"},
        "uncertainty_verdict": {"ic_count": int(ic_count), "ok": bool(int(ic_count) == 0), "mode": "fail_closed"},
        "stats": {"vectors_total": int(len(test_vectors)), "vectors_ok": int(ok_count)},
    }
    cert["certificate_sig"] = certificate_sig_v2(cert)
    return cert


def verify_pcc_v2(
    *,
    candidate_act: Act,
    certificate: Dict[str, Any],
    store_base: ActStore,
    seed: int = 0,
) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(certificate, dict):
        return False, "certificate_not_dict", {}
    if int(certificate.get("schema_version", 0) or 0) != 2:
        return False, "bad_schema_version", {}
    if str(certificate.get("certificate_kind") or "") != "pcc_v74_certificate_v2":
        return False, "bad_certificate_kind", {}

    # certificate_sig integrity.
    want_sig = str(certificate.get("certificate_sig") or "")
    got_sig = certificate_sig_v2(certificate)
    if want_sig != got_sig:
        return False, "certificate_sig_mismatch", {"want": want_sig, "got": got_sig}

    # store_hash_base integrity.
    want_store = str(certificate.get("store_hash_base") or "")
    got_store = str(store_base.content_hash())
    if want_store != got_store:
        return False, "store_hash_base_mismatch", {"want": want_store, "got": got_store}

    # Candidate integrity.
    cand = certificate.get("candidate") if isinstance(certificate.get("candidate"), dict) else {}
    failures: List[Dict[str, Any]] = []
    if str(cand.get("act_id") or "") != str(candidate_act.id):
        failures.append({"reason": "candidate_id_mismatch"})
    if str(cand.get("kind") or "") and str(cand.get("kind") or "") != str(candidate_act.kind):
        failures.append({"reason": "candidate_kind_mismatch"})

    want_iface_sig = str(cand.get("interface_sig") or "")
    want_prog_sha = str(cand.get("program_sha256") or "")
    got_iface_sig = interface_sig_from_act(candidate_act)
    got_prog_sha = program_sha256_from_act(candidate_act)
    if want_iface_sig != got_iface_sig:
        failures.append({"reason": "interface_sig_mismatch", "want": want_iface_sig, "got": got_iface_sig})
    if want_prog_sha != got_prog_sha:
        failures.append({"reason": "program_sha256_mismatch", "want": want_prog_sha, "got": got_prog_sha})
    if int(cand.get("program_len", 0) or 0) != int(len(candidate_act.program or [])):
        failures.append({"reason": "program_len_mismatch"})

    # call_deps integrity against store_base.
    deps = certificate.get("call_deps") if isinstance(certificate.get("call_deps"), list) else None
    if deps is None:
        failures.append({"reason": "missing_call_deps"})
        deps = []

    # Compare to computed call deps from candidate program to prevent omission.
    want_dep_ids = [d.get("concept_id") for d in deps if isinstance(d, dict)]
    want_dep_ids = [str(x) for x in want_dep_ids if str(x)]
    want_dep_ids_sorted = sorted(set(want_dep_ids), key=str)
    got_dep_ids_sorted = _extract_call_deps_from_program(candidate_act.program or [])
    if want_dep_ids_sorted != got_dep_ids_sorted:
        failures.append({"reason": "call_deps_list_mismatch", "want": want_dep_ids_sorted, "got": got_dep_ids_sorted})

    for dep in deps:
        if not isinstance(dep, dict):
            failures.append({"reason": "bad_call_dep"})
            continue
        cid = str(dep.get("concept_id") or "")
        if not cid:
            failures.append({"reason": "call_dep_missing_concept_id"})
            continue
        dep_act = store_base.get_concept_act(cid)
        if dep_act is None:
            failures.append({"reason": "call_dep_missing_in_store", "concept_id": cid})
            continue
        want_isig = str(dep.get("interface_sig") or "")
        want_psha = str(dep.get("program_sha256") or "")
        got_isig = interface_sig_from_act(dep_act)
        got_psha = program_sha256_from_act(dep_act)
        if want_isig != got_isig:
            failures.append({"reason": "call_dep_interface_sig_mismatch", "concept_id": cid})
        if want_psha != got_psha:
            failures.append({"reason": "call_dep_program_sha256_mismatch", "concept_id": cid})

    # Re-execute test_vectors deterministically.
    tvs = certificate.get("test_vectors") if isinstance(certificate.get("test_vectors"), list) else []
    if int(len(tvs)) < 3:
        failures.append({"reason": "not_enough_test_vectors", "got": int(len(tvs))})

    store_exec = _clone_store_with_candidate(store_base=store_base, candidate_act=candidate_act)
    engine = Engine(store_exec, seed=int(seed), config=EngineConfig(enable_contracts=False))

    ok_count = 0
    ic_count = 0
    for i, tv in enumerate(tvs):
        if not isinstance(tv, dict):
            failures.append({"idx": int(i), "reason": "test_vector_not_dict"})
            continue
        inputs = tv.get("inputs") if isinstance(tv.get("inputs"), dict) else None
        if inputs is None:
            failures.append({"idx": int(i), "reason": "bad_vector_inputs"})
            continue
        expected = tv.get("expected")
        expected_sig = _stable_hash_obj({"inputs": {str(k): inputs.get(k) for k in sorted(inputs.keys(), key=str)}, "expected": expected})
        if str(tv.get("expected_sig") or "") != str(expected_sig):
            failures.append({"idx": int(i), "reason": "expected_sig_mismatch"})
            continue

        out = engine.execute_concept_csv(
            concept_act_id=str(candidate_act.id),
            inputs={str(k): inputs.get(k) for k in sorted(inputs.keys(), key=str)},
            expected=expected,
            step=int(i),
            max_depth=8,
            max_events=512,
            validate_output=True,
        )
        meta = out.get("meta") if isinstance(out, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        got = str(meta.get("output_text") or out.get("output") or "")
        got_ok = bool(meta.get("ok", False))
        got_sig = str(meta.get("output_sig") or "")
        validator = meta.get("validator") if isinstance(meta.get("validator"), dict) else {}
        ethics = meta.get("ethics") if isinstance(meta.get("ethics"), dict) else {}
        uncertainty = meta.get("uncertainty") if isinstance(meta.get("uncertainty"), dict) else {}

        if str(uncertainty.get("mode_out") or "") == "IC":
            ic_count += 1

        if got != str(tv.get("got") or ""):
            failures.append({"idx": int(i), "reason": "got_mismatch"})
            continue
        if got_ok != bool(tv.get("ok", False)):
            failures.append({"idx": int(i), "reason": "ok_mismatch"})
            continue
        if got_sig != str(tv.get("output_sig") or ""):
            failures.append({"idx": int(i), "reason": "output_sig_mismatch"})
            continue
        want_v = tv.get("validator") if isinstance(tv.get("validator"), dict) else {}
        if bool(validator.get("passed", False)) != bool(want_v.get("passed", False)):
            failures.append({"idx": int(i), "reason": "validator_pass_mismatch"})
            continue
        if not bool(validator.get("passed", False)):
            failures.append({"idx": int(i), "reason": "validator_failed"})
            continue
        if not bool(ethics.get("ok", False)):
            failures.append({"idx": int(i), "reason": "ethics_fail_closed"})
            continue

        ok_count += 1

    if int(ic_count) != 0:
        failures.append({"reason": "uncertainty_ic_fail_closed", "ic_count": int(ic_count)})

    ok = not failures and ok_count == len(tvs)
    details = {"vectors_total": int(len(tvs)), "vectors_ok": int(ok_count), "ic_count": int(ic_count), "failures": list(failures)}
    return bool(ok), "ok" if ok else "pcc_verify_failed", details

