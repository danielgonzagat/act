from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .act import Act, canonical_json_dumps, sha256_hex
from .engine import Engine, EngineConfig
from .ethics import validate_act_for_promotion
from .store import ActStore
from .validators import run_validator


@dataclass(frozen=True)
class ProofVerdict:
    ok: bool
    reason: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": bool(self.ok), "reason": str(self.reason), "details": dict(self.details)}


def _sha256_text(s: str) -> str:
    return sha256_hex(str(s).encode("utf-8"))


def program_sha256(act: Act) -> str:
    prog = [ins.to_dict() for ins in (act.program or [])]
    return sha256_hex(canonical_json_dumps(prog).encode("utf-8"))


def certificate_body_sha256(cert: Dict[str, Any]) -> str:
    body = dict(cert)
    body.pop("hashes", None)
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def certificate_sha256(cert: Dict[str, Any]) -> str:
    c = copy.deepcopy(cert)
    hashes = c.get("hashes")
    if not isinstance(hashes, dict):
        hashes = {}
        c["hashes"] = hashes
    hashes["certificate_sha256"] = ""
    # act_body_sha256 is computed with placeholder semantics; exclude it from the cert hash to
    # avoid circularity (it is verified independently).
    hashes["act_body_sha256"] = ""
    return sha256_hex(canonical_json_dumps(c).encode("utf-8"))


def act_body_sha256_placeholder(act: Act) -> str:
    d = act.to_dict()
    ev = d.get("evidence")
    if isinstance(ev, dict):
        cert = ev.get("certificate_v1")
        if isinstance(cert, dict):
            hashes = cert.get("hashes")
            if isinstance(hashes, dict):
                hashes = dict(hashes)
                hashes["act_body_sha256"] = ""
                cert = dict(cert)
                cert["hashes"] = hashes
                ev = dict(ev)
                ev["certificate_v1"] = cert
                d["evidence"] = ev
    return sha256_hex(canonical_json_dumps(d).encode("utf-8"))


def build_concept_pcc_certificate_v1(
    act: Act,
    *,
    mined_from: Dict[str, Any],
    test_vectors: List[Dict[str, Any]],
    ethics_verdict: Dict[str, Any],
    uncertainty_policy: str = "no_ic",
) -> Dict[str, Any]:
    """
    Build a deterministic Proof-Carrying Concept certificate (PCC) with stable hashes.
    """
    iface = {}
    ev = act.evidence if isinstance(act.evidence, dict) else {}
    iface = ev.get("interface") if isinstance(ev, dict) else {}
    iface = iface if isinstance(iface, dict) else {}
    interface = {
        "input_schema": dict(iface.get("input_schema", {})),
        "output_schema": dict(iface.get("output_schema", {})),
        "validator_id": str(iface.get("validator_id") or ""),
        "iface_sig": _sha256_text(
            canonical_json_dumps(
                {
                    "in": iface.get("input_schema", {}),
                    "out": iface.get("output_schema", {}),
                    "validator_id": iface.get("validator_id", ""),
                }
            )
        ),
    }

    cert: Dict[str, Any] = {
        "schema_version": 1,
        "mined_from": dict(mined_from),
        "interface": interface,
        "test_vectors": list(test_vectors),
        "validator_results": [],
        "ethics_verdict": dict(ethics_verdict),
        "uncertainty_policy": str(uncertainty_policy),
        "hashes": {},
    }

    # Pre-fill deterministic hashes (placeholders for act_body_sha256).
    hashes: Dict[str, Any] = {
        "program_sha256": str(program_sha256(act)),
        "certificate_body_sha256": str(certificate_body_sha256(cert)),
        "certificate_sha256": "",
        "act_body_sha256": "",
    }
    cert["hashes"] = hashes
    hashes["certificate_sha256"] = str(certificate_sha256(cert))
    return cert


def verify_concept_pcc(act: Act, store: ActStore) -> ProofVerdict:
    if str(getattr(act, "kind", "")) != "concept_csv":
        return ProofVerdict(False, "wrong_kind", {"kind": str(getattr(act, "kind", ""))})

    ev = act.evidence if isinstance(act.evidence, dict) else {}
    cert = ev.get("certificate_v1") if isinstance(ev, dict) else None
    if not isinstance(cert, dict):
        return ProofVerdict(False, "missing_certificate", {})
    if int(cert.get("schema_version", 0) or 0) != 1:
        return ProofVerdict(False, "bad_schema_version", {"schema_version": cert.get("schema_version")})

    hashes = cert.get("hashes")
    if not isinstance(hashes, dict):
        return ProofVerdict(False, "missing_hashes", {})

    want_prog = str(hashes.get("program_sha256") or "")
    got_prog = program_sha256(act)
    if want_prog != got_prog:
        return ProofVerdict(False, "program_sha256_mismatch", {"want": want_prog, "got": got_prog})

    want_body = str(hashes.get("certificate_body_sha256") or "")
    got_body = certificate_body_sha256(cert)
    if want_body != got_body:
        return ProofVerdict(False, "certificate_body_sha256_mismatch", {"want": want_body, "got": got_body})

    want_cert = str(hashes.get("certificate_sha256") or "")
    got_cert = certificate_sha256(cert)
    if want_cert != got_cert:
        return ProofVerdict(False, "certificate_sha256_mismatch", {"want": want_cert, "got": got_cert})

    want_act = str(hashes.get("act_body_sha256") or "")
    got_act = act_body_sha256_placeholder(act)
    if want_act and want_act != got_act:
        return ProofVerdict(False, "act_body_sha256_mismatch", {"want": want_act, "got": got_act})

    # Ethics must pass for promotion.
    ethics = validate_act_for_promotion(act)
    if not bool(ethics.ok):
        return ProofVerdict(False, "ethics_fail_closed", ethics.to_dict())

    iface = cert.get("interface") if isinstance(cert.get("interface"), dict) else {}
    validator_id = str(iface.get("validator_id") or "")
    if not validator_id:
        return ProofVerdict(False, "missing_validator_id", {})

    test_vectors = cert.get("test_vectors")
    if not isinstance(test_vectors, list) or len(test_vectors) < 1:
        return ProofVerdict(False, "missing_test_vectors", {})

    # Verify vectors by executing the concept with the exact store + act present.
    store2 = ActStore(acts=dict(store.acts), next_id_int=int(store.next_id_int))
    if store2.get(act.id) is None:
        store2.add(act)
    engine = Engine(store2, seed=0, config=EngineConfig())

    results: List[Dict[str, Any]] = []
    for vec in test_vectors:
        if not isinstance(vec, dict):
            return ProofVerdict(False, "bad_vector", {"vector": vec})
        inputs = vec.get("inputs")
        if not isinstance(inputs, dict):
            return ProofVerdict(False, "bad_vector_inputs", {"vector": vec})
        expected = vec.get("expected")
        expected_text = str(vec.get("expected_output_text") or "")

        out = engine.execute_concept_csv(concept_act_id=act.id, inputs=dict(inputs), expected=expected, step=0)
        meta = out.get("meta") if isinstance(out, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        ok = bool(meta.get("ok", False))
        out_text = str(meta.get("output_text") or "")
        eth = meta.get("ethics") if isinstance(meta.get("ethics"), dict) else {}
        unc = meta.get("uncertainty") if isinstance(meta.get("uncertainty"), dict) else {}
        unc_mode = str(unc.get("mode_out") or "")

        vres = run_validator(validator_id, out_text, expected)
        results.append(
            {
                "ok": ok,
                "out_text": out_text,
                "expected_text": expected_text,
                "validator_passed": bool(vres.passed),
                "validator_reason": str(vres.reason),
                "ethics_ok": bool(eth.get("ok", True)),
                "uncertainty_mode_out": unc_mode,
            }
        )
        if out_text != expected_text:
            return ProofVerdict(False, "vector_output_text_mismatch", {"result": results[-1]})
        if not bool(vres.passed):
            return ProofVerdict(False, "vector_validator_failed", {"result": results[-1]})
        if not bool(eth.get("ok", True)):
            return ProofVerdict(False, "vector_ethics_failed", {"result": results[-1]})
        if unc_mode == "IC":
            return ProofVerdict(False, "vector_uncertainty_ic", {"result": results[-1]})

    return ProofVerdict(True, "ok", {"vectors_verified": len(results)})
