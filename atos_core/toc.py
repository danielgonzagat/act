from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .act import Act, canonical_json_dumps, sha256_hex
from .engine import Engine, EngineConfig
from .validators import run_validator


def compute_interface_sig(act: Act) -> str:
    ev = act.evidence if isinstance(act.evidence, dict) else {}
    iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
    iface = iface if isinstance(iface, dict) else {}
    body = {
        "in": iface.get("input_schema", {}),
        "out": iface.get("output_schema", {}),
        "validator_id": str(iface.get("validator_id") or ""),
    }
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def compute_program_sig(act: Act) -> str:
    prog = [ins.to_dict() for ins in (act.program or [])]
    return sha256_hex(canonical_json_dumps(prog).encode("utf-8"))


def op_token_set(act: Act) -> List[str]:
    toks: List[str] = []
    for ins in act.program or []:
        op = str(getattr(ins, "op", "") or "")
        args = getattr(ins, "args", {}) or {}
        if op == "CSV_PRIMITIVE":
            fn = str(args.get("fn") or "")
            toks.append(f"{op}:{fn}")
        elif op == "CSV_CALL":
            toks.append("CSV_CALL")
        else:
            toks.append(op)
    toks.sort()
    return toks


def similarity(a: Act, b: Act) -> float:
    """
    Deterministic, cheap similarity to detect near-duplicates.
    Jaccard over (program op tokens + interface fields).
    """
    a_set = set(op_token_set(a))
    b_set = set(op_token_set(b))
    # Interface contributes as tokens too.
    for act, s in ((a, a_set), (b, b_set)):
        ev = act.evidence if isinstance(act.evidence, dict) else {}
        iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
        iface = iface if isinstance(iface, dict) else {}
        inp = iface.get("input_schema") if isinstance(iface.get("input_schema"), dict) else {}
        out = iface.get("output_schema") if isinstance(iface.get("output_schema"), dict) else {}
        for k in sorted(list(inp.keys())):
            s.add(f"in:{k}:{inp.get(k)}")
        for k in sorted(list(out.keys())):
            s.add(f"out:{k}:{out.get(k)}")
        s.add(f"validator:{str(iface.get('validator_id') or '')}")

    if not a_set and not b_set:
        return 1.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return float(inter / max(1, union))


@dataclass(frozen=True)
class ToCVerdict:
    ok: bool
    reason: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": bool(self.ok), "reason": str(self.reason), "details": dict(self.details)}


def _exec_vectors(
    *,
    act: Act,
    vectors: Sequence[Dict[str, Any]],
    store,
) -> Tuple[bool, List[Dict[str, Any]]]:
    engine = Engine(store, seed=0, config=EngineConfig())
    iface_sig = compute_interface_sig(act)
    prog_sig = compute_program_sig(act)
    results: List[Dict[str, Any]] = []
    for vec in vectors:
        inputs = vec.get("inputs") if isinstance(vec.get("inputs"), dict) else None
        expected = vec.get("expected")
        expected_text = str(vec.get("expected_output_text") or "")
        if inputs is None:
            results.append({"ok": False, "reason": "bad_vector_inputs"})
            continue
        out = engine.execute_concept_csv(concept_act_id=str(act.id), inputs=dict(inputs), expected=expected, step=0)
        meta = out.get("meta") if isinstance(out, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        ok = bool(meta.get("ok", False))
        out_text = str(meta.get("output_text") or "")
        eth = meta.get("ethics") if isinstance(meta.get("ethics"), dict) else {}
        unc = meta.get("uncertainty") if isinstance(meta.get("uncertainty"), dict) else {}
        unc_mode = str(unc.get("mode_out") or "")

        # Also enforce validator deterministically (defense-in-depth).
        cert_iface = act.evidence.get("interface") if isinstance(act.evidence, dict) else {}
        cert_iface = cert_iface if isinstance(cert_iface, dict) else {}
        validator_id = str(cert_iface.get("validator_id") or "")
        vres = run_validator(validator_id, out_text, expected) if validator_id else None

        results.append(
            {
                "ok": bool(ok),
                "out_text": str(out_text),
                "expected_text": str(expected_text),
                "ethics_ok": bool(eth.get("ok", True)),
                "uncertainty_mode_out": str(unc_mode),
                "validator_id": str(validator_id),
                "validator_passed": bool(vres.passed) if vres is not None else True,
                "validator_reason": str(vres.reason) if vres is not None else "skipped",
                "iface_sig": str(iface_sig),
                "program_sig": str(prog_sig),
            }
        )

    all_ok = all(
        bool(r.get("ok", False))
        and bool(r.get("ethics_ok", True))
        and bool(r.get("validator_passed", True))
        and str(r.get("out_text") or "") == str(r.get("expected_text") or "")
        and str(r.get("uncertainty_mode_out") or "") != "IC"
        for r in results
    )
    return bool(all_ok), results


def toc_eval(
    *,
    concept_act: Act,
    vectors_A: Sequence[Dict[str, Any]],
    vectors_B: Sequence[Dict[str, Any]],
    store,
    domain_A: str,
    domain_B: str,
    min_vectors_per_domain: int = 3,
) -> Dict[str, Any]:
    """
    Deterministic transfer-of-composition evaluation across two domains.
    """
    iface_sig = compute_interface_sig(concept_act)
    prog_sig = compute_program_sig(concept_act)

    a_ok = False
    b_ok = False
    a_results: List[Dict[str, Any]] = []
    b_results: List[Dict[str, Any]] = []

    if len(list(vectors_A)) >= int(min_vectors_per_domain):
        a_ok, a_results = _exec_vectors(act=concept_act, vectors=vectors_A, store=store)
    if len(list(vectors_B)) >= int(min_vectors_per_domain):
        b_ok, b_results = _exec_vectors(act=concept_act, vectors=vectors_B, store=store)

    return {
        "schema_version": 1,
        "domains": [str(domain_A), str(domain_B)],
        "toc_required": True,
        "iface_sig": str(iface_sig),
        "program_sig": str(prog_sig),
        "vectors_A": list(vectors_A),
        "vectors_B": list(vectors_B),
        "pass_A": bool(a_ok),
        "pass_B": bool(b_ok),
        "details": {
            "domain_A": str(domain_A),
            "domain_B": str(domain_B),
            "min_vectors_per_domain": int(min_vectors_per_domain),
            "got_vectors_A": int(len(list(vectors_A))),
            "got_vectors_B": int(len(list(vectors_B))),
            "results_A": list(a_results),
            "results_B": list(b_results),
        },
    }


def verify_concept_toc_v1(concept_act: Act, *, store) -> ToCVerdict:
    ev = concept_act.evidence if isinstance(concept_act.evidence, dict) else {}
    cert = ev.get("certificate_v2") if isinstance(ev.get("certificate_v2"), dict) else None
    if not isinstance(cert, dict):
        return ToCVerdict(False, "missing_certificate_v2", {})
    toc = cert.get("toc_v1")
    if not isinstance(toc, dict):
        return ToCVerdict(False, "missing_toc_v1", {})
    if int(toc.get("schema_version", 0) or 0) != 1:
        return ToCVerdict(False, "bad_toc_schema_version", {"schema_version": toc.get("schema_version")})

    want_iface = str(toc.get("iface_sig") or "")
    want_prog = str(toc.get("program_sig") or "")
    got_iface = compute_interface_sig(concept_act)
    got_prog = compute_program_sig(concept_act)
    if want_iface and want_iface != got_iface:
        return ToCVerdict(False, "iface_sig_mismatch", {"want": want_iface, "got": got_iface})
    if want_prog and want_prog != got_prog:
        return ToCVerdict(False, "program_sig_mismatch", {"want": want_prog, "got": got_prog})

    domains = toc.get("domains") if isinstance(toc.get("domains"), list) else []
    if len(domains) != 2:
        return ToCVerdict(False, "bad_domains", {"domains": domains})
    domain_A = str(domains[0])
    domain_B = str(domains[1])

    vectors_A = toc.get("vectors_A") if isinstance(toc.get("vectors_A"), list) else []
    vectors_B = toc.get("vectors_B") if isinstance(toc.get("vectors_B"), list) else []
    min_vecs = int(toc.get("details", {}).get("min_vectors_per_domain", 3) or 3) if isinstance(toc.get("details"), dict) else 3

    toc2 = toc_eval(
        concept_act=concept_act,
        vectors_A=vectors_A,
        vectors_B=vectors_B,
        store=store,
        domain_A=domain_A,
        domain_B=domain_B,
        min_vectors_per_domain=min_vecs,
    )
    if not bool(toc2.get("pass_A", False)):
        return ToCVerdict(False, "toc_domain_A_failed", {"toc": toc2})
    if not bool(toc2.get("pass_B", False)):
        return ToCVerdict(False, "toc_domain_B_failed", {"toc": toc2})
    return ToCVerdict(True, "ok", {"toc": toc2})


def detect_duplicate(
    candidate: Act,
    *,
    existing: Sequence[Act],
    similarity_threshold: float = 0.95,
) -> Optional[Dict[str, Any]]:
    cand_iface = compute_interface_sig(candidate)
    cand_prog = compute_program_sig(candidate)
    for other in existing:
        if str(other.id) == str(candidate.id):
            continue
        if compute_interface_sig(other) == cand_iface and compute_program_sig(other) == cand_prog:
            return {
                "reason": "duplicate_exact",
                "other_id": str(other.id),
                "similarity": 1.0,
            }
        sim = similarity(candidate, other)
        if sim >= float(similarity_threshold):
            return {
                "reason": "duplicate_similar",
                "other_id": str(other.id),
                "similarity": float(sim),
            }
    return None

