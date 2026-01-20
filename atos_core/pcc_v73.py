from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence, Tuple

from .act import Act, canonical_json_dumps, sha256_hex
from .concept_registry_v70 import interface_sig_from_act, program_sha256_from_act
from .engine import Engine, EngineConfig
from .store import ActStore
from .trace_v73 import TraceV73


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


def certificate_sig_v1(cert: Dict[str, Any]) -> str:
    body = dict(cert) if isinstance(cert, dict) else {}
    body.pop("certificate_sig", None)
    return _stable_hash_obj(body)


def _clone_store_with_candidate(*, store_base: ActStore, candidate_act: Act) -> ActStore:
    """
    Create an isolated store containing base acts + candidate act.
    Deterministic and avoids mutating caller state during verification.
    """
    store2 = ActStore()
    for act_id in sorted(store_base.acts.keys(), key=str):
        act = store_base.acts[act_id]
        store2.add(Act.from_dict(act.to_dict()))
    store2.add(Act.from_dict(candidate_act.to_dict()))
    return store2


def build_certificate_v1(
    *,
    candidate_act: Act,
    traces: Sequence[TraceV73],
    store_base: ActStore,
    seed: int = 0,
) -> Dict[str, Any]:
    iface_sig = interface_sig_from_act(candidate_act)
    prog_sha = program_sha256_from_act(candidate_act)
    store_hash_base = store_base.content_hash()

    store_exec = _clone_store_with_candidate(store_base=store_base, candidate_act=candidate_act)
    engine = Engine(store_exec, seed=int(seed), config=EngineConfig(enable_contracts=False))

    traces_sorted = sorted([t for t in traces if isinstance(t, TraceV73)], key=lambda t: str(t.trace_sig()))
    trace_sigs = [str(t.trace_sig()) for t in traces_sorted]
    contexts = sorted(set(str(t.context_id) for t in traces_sorted if str(t.context_id)))

    test_vectors: List[Dict[str, Any]] = []
    ethics_ok_all = True
    ic_count = 0

    for i, tr in enumerate(traces_sorted):
        inputs = tr.bindings if isinstance(tr.bindings, dict) else {}
        expected = tr.expected
        res = engine.execute_concept_csv(
            concept_act_id=str(candidate_act.id),
            inputs={str(k): inputs.get(k) for k in sorted(inputs.keys(), key=str)},
            expected=expected,
            step=int(i),
            max_depth=8,
            max_events=512,
            validate_output=True,
        )
        meta = res.get("meta") if isinstance(res, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        ethics = meta.get("ethics") if isinstance(meta.get("ethics"), dict) else {}
        uncertainty = meta.get("uncertainty") if isinstance(meta.get("uncertainty"), dict) else {}
        validator = meta.get("validator") if isinstance(meta.get("validator"), dict) else {}

        ok = bool(meta.get("ok", False))
        got = str(meta.get("output_text") or res.get("output") or "")
        output_sig = str(meta.get("output_sig") or "")

        ethics_ok = bool(ethics.get("ok", False))
        ethics_ok_all = bool(ethics_ok_all and ethics_ok)
        if str(uncertainty.get("mode_out") or "") == "IC":
            ic_count += 1

        tv = {
            "trace_sig": str(tr.trace_sig()),
            "context_id": str(tr.context_id),
            "inputs": {str(k): inputs.get(k) for k in sorted(inputs.keys(), key=str)},
            "expected": expected,
            "got": str(got),
            "ok": bool(ok),
            "output_sig": str(output_sig),
            "validator": _safe_deepcopy(validator),
            "ethics": _safe_deepcopy(ethics),
            "uncertainty": _safe_deepcopy(uncertainty),
        }
        test_vectors.append(tv)

    cert: Dict[str, Any] = {
        "schema_version": 1,
        "certificate_kind": "pcc_v73_certificate_v1",
        "store_hash_base": str(store_hash_base),
        "mined_from": {"trace_sigs": list(trace_sigs), "contexts_distinct": int(len(contexts))},
        "candidate": {
            "act_id": str(candidate_act.id),
            "kind": str(candidate_act.kind),
            "interface": _safe_deepcopy(
                (candidate_act.evidence or {}).get("interface") if isinstance(candidate_act.evidence, dict) else {}
            ),
            "interface_sig": str(iface_sig),
            "program_sha256": str(prog_sha),
            "program_len": int(len(candidate_act.program or [])),
        },
        "test_vectors": list(test_vectors),
        "ethics_verdict": {"ok": bool(ethics_ok_all), "mode": "fail_closed"},
        "uncertainty_verdict": {"ic_count": int(ic_count), "ok": bool(ic_count == 0), "mode": "fail_closed"},
    }
    cert["certificate_sig"] = certificate_sig_v1(cert)
    return cert


def verify_pcc_v73(
    *,
    candidate_act: Act,
    certificate: Dict[str, Any],
    store_base: ActStore,
    seed: int = 0,
) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(certificate, dict):
        return False, "certificate_not_dict", {}
    if int(certificate.get("schema_version", 0) or 0) != 1:
        return False, "bad_schema_version", {}
    if str(certificate.get("certificate_kind") or "") != "pcc_v73_certificate_v1":
        return False, "bad_certificate_kind", {}

    # Signature integrity.
    want_sig = str(certificate.get("certificate_sig") or "")
    got_sig = certificate_sig_v1(certificate)
    if want_sig != got_sig:
        return False, "certificate_sig_mismatch", {"want": want_sig, "got": got_sig}

    # Store base hash integrity.
    want_store_hash = str(certificate.get("store_hash_base") or "")
    got_store_hash = str(store_base.content_hash())
    if want_store_hash != got_store_hash:
        return False, "store_hash_base_mismatch", {"want": want_store_hash, "got": got_store_hash}

    # Candidate integrity.
    cand = certificate.get("candidate") if isinstance(certificate.get("candidate"), dict) else {}
    want_iface_sig = str(cand.get("interface_sig") or "")
    want_prog_sha = str(cand.get("program_sha256") or "")
    got_iface_sig = interface_sig_from_act(candidate_act)
    got_prog_sha = program_sha256_from_act(candidate_act)
    if want_iface_sig != got_iface_sig:
        return False, "interface_sig_mismatch", {"want": want_iface_sig, "got": got_iface_sig}
    if want_prog_sha != got_prog_sha:
        return False, "program_sha256_mismatch", {"want": want_prog_sha, "got": got_prog_sha}
    if str(cand.get("act_id") or "") != str(candidate_act.id):
        return False, "candidate_id_mismatch", {"want": str(cand.get("act_id") or ""), "got": str(candidate_act.id)}

    # Re-execute vectors deterministically and compare.
    store_exec = _clone_store_with_candidate(store_base=store_base, candidate_act=candidate_act)
    engine = Engine(store_exec, seed=int(seed), config=EngineConfig(enable_contracts=False))

    tvs = certificate.get("test_vectors") if isinstance(certificate.get("test_vectors"), list) else []
    failures: List[Dict[str, Any]] = []
    ok_count = 0
    ic_count = 0
    ethics_ok_all = True

    for i, tv in enumerate(tvs):
        if not isinstance(tv, dict):
            failures.append({"idx": int(i), "reason": "test_vector_not_dict"})
            continue
        inputs = tv.get("inputs") if isinstance(tv.get("inputs"), dict) else {}
        expected = tv.get("expected")
        res = engine.execute_concept_csv(
            concept_act_id=str(candidate_act.id),
            inputs={str(k): inputs.get(k) for k in sorted(inputs.keys(), key=str)},
            expected=expected,
            step=int(i),
            max_depth=8,
            max_events=512,
            validate_output=True,
        )
        meta = res.get("meta") if isinstance(res, dict) else {}
        meta = meta if isinstance(meta, dict) else {}

        got = str(meta.get("output_text") or res.get("output") or "")
        got_ok = bool(meta.get("ok", False))
        got_sig2 = str(meta.get("output_sig") or "")
        ethics = meta.get("ethics") if isinstance(meta.get("ethics"), dict) else {}
        uncertainty = meta.get("uncertainty") if isinstance(meta.get("uncertainty"), dict) else {}
        validator = meta.get("validator") if isinstance(meta.get("validator"), dict) else {}

        ethics_ok = bool(ethics.get("ok", False))
        ethics_ok_all = bool(ethics_ok_all and ethics_ok)
        if str(uncertainty.get("mode_out") or "") == "IC":
            ic_count += 1

        if got != str(tv.get("got") or ""):
            failures.append({"idx": int(i), "reason": "got_mismatch", "want": str(tv.get("got") or ""), "got": got})
            continue
        if got_ok != bool(tv.get("ok", False)):
            failures.append({"idx": int(i), "reason": "ok_mismatch", "want": bool(tv.get("ok", False)), "got": got_ok})
            continue
        if got_sig2 != str(tv.get("output_sig") or ""):
            failures.append({"idx": int(i), "reason": "output_sig_mismatch", "want": str(tv.get("output_sig") or ""), "got": got_sig2})
            continue
        # Validator pass is part of the contract.
        want_v = tv.get("validator") if isinstance(tv.get("validator"), dict) else {}
        if bool(validator.get("passed", False)) != bool(want_v.get("passed", False)):
            failures.append({"idx": int(i), "reason": "validator_pass_mismatch"})
            continue

        ok_count += 1

    if not bool(ethics_ok_all):
        failures.append({"reason": "ethics_fail_closed"})
    if int(ic_count) != 0:
        failures.append({"reason": "uncertainty_ic_fail_closed", "ic_count": int(ic_count)})

    ok = (not failures) and (ok_count == len(tvs))
    details = {"vectors_total": int(len(tvs)), "vectors_ok": int(ok_count), "failures": list(failures), "ic_count": int(ic_count)}
    return bool(ok), "ok" if ok else "pcc_verify_failed", details

