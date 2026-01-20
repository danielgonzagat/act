from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import Act, canonical_json_dumps, sha256_hex
from .logic_v86 import enumerate_assignments, ensure_bool_primitives_registered, logic_canon, logic_eval, render_output01, truth_table, truth_table_sha256
from .pcc_v74 import certificate_sig_v2
from .pcc_v85 import build_certificate_v85, verify_pcc_v85
from .store import ActStore


def _stable_hash(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _norm_str_list(val: Any) -> List[str]:
    if not isinstance(val, list):
        return []
    out: List[str] = []
    seen = set()
    for x in val:
        s = str(x or "")
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    out.sort(key=str)
    return out


def _is_logic_eligible_act(act: Act) -> bool:
    mk = act.match if isinstance(getattr(act, "match", None), dict) else {}
    gks = mk.get("goal_kinds")
    gks2 = gks if isinstance(gks, list) else []
    for gk in gks2:
        if str(gk or "").startswith("v86_logic_"):
            return True
    return False


def _render_to_out01(out: bool) -> str:
    return "1" if bool(out) else "0"


def _norm_render(render: Dict[str, Any]) -> Dict[str, Any]:
    rk = str(render.get("kind") or "")
    if rk == "raw01":
        return {"kind": "raw01"}
    if rk == "prefix01":
        return {"kind": "prefix01", "prefix": str(render.get("prefix") or "")}
    raise ValueError("bad_render")


def _norm_domain_values(domain_values: Any) -> List[str]:
    if not isinstance(domain_values, list):
        raise ValueError("bad_domain_values")
    vals = [str(v) for v in domain_values if str(v)]
    vals = sorted(set(vals), key=str)
    if vals != ["0", "1"]:
        raise ValueError("bad_domain_values")
    return list(vals)


def build_certificate_v86(
    *,
    candidate_act: Act,
    store_base: ActStore,
    mined_from: Dict[str, Any],
    vector_specs: Sequence[Dict[str, Any]],
    seed: int = 0,
    logic_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    V86: PCC wrapper that extends V85 with an optional finite-domain logic proof block.

    logic_spec (optional) shape:
      {
        "vars": [...],
        "domain_values": ["0","1"],
        "expr": <logic AST>,
        "render": {"kind":"raw01"} | {"kind":"prefix01","prefix":"BOOL="},
      }
    """
    ensure_bool_primitives_registered()

    cert = build_certificate_v85(
        candidate_act=candidate_act,
        store_base=store_base,
        mined_from=mined_from,
        vector_specs=vector_specs,
        seed=int(seed),
    )

    if logic_spec is None:
        return cert

    vars_raw = logic_spec.get("vars") if isinstance(logic_spec, dict) else None
    domain_raw = logic_spec.get("domain_values") if isinstance(logic_spec, dict) else None
    expr_raw = logic_spec.get("expr") if isinstance(logic_spec, dict) else None
    render_raw = logic_spec.get("render") if isinstance(logic_spec, dict) else None

    vars_norm = _norm_str_list(vars_raw)
    if not vars_norm:
        raise ValueError("bad_logic_vars")
    domain_values = _norm_domain_values(domain_raw)
    expr_canon = logic_canon(expr_raw)
    render = _norm_render(render_raw if isinstance(render_raw, dict) else {})

    tt = truth_table(vars=vars_norm, domain_values=domain_values, expr=expr_canon, render=render)
    tt_sha = truth_table_sha256(tt=tt)

    body = {
        "schema_version": 1,
        "vars": list(vars_norm),
        "domain_values": list(domain_values),
        "expr": dict(expr_canon),
        "render": dict(render),
        "truth_table_sha256": str(tt_sha),
    }
    cert["logic_binding"] = {**dict(body), "binding_sig": _stable_hash(body)}
    cert["certificate_sig"] = certificate_sig_v2(cert)
    return cert


def verify_pcc_v86(
    *,
    candidate_act: Act,
    certificate: Dict[str, Any],
    store_base: ActStore,
    seed: int = 0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    V86: verify V85 + (optional/required) finite-domain logic proof.
    """
    ensure_bool_primitives_registered()

    ok, reason, details = verify_pcc_v85(candidate_act=candidate_act, certificate=certificate, store_base=store_base, seed=int(seed))
    if not bool(ok):
        return False, str(reason), dict(details)

    lb = certificate.get("logic_binding") if isinstance(certificate.get("logic_binding"), dict) else None
    require_logic = bool(isinstance(lb, dict)) or bool(_is_logic_eligible_act(candidate_act))
    if require_logic and not isinstance(lb, dict):
        return False, "missing_logic_binding", {}
    if not require_logic:
        return True, "ok", dict(details)

    # logic_binding integrity.
    if int(lb.get("schema_version", 0) or 0) != 1:
        return False, "bad_logic_schema_version", {}

    want_sig = str(lb.get("binding_sig") or "")
    body = dict(lb)
    body.pop("binding_sig", None)
    got_sig = _stable_hash(body)
    if want_sig != got_sig:
        return False, "logic_binding_sig_mismatch", {"want": want_sig, "got": got_sig}

    vars_norm = _norm_str_list(body.get("vars"))
    if not vars_norm:
        return False, "bad_logic_vars", {}
    domain_values = _norm_domain_values(body.get("domain_values"))
    expr_canon = logic_canon(body.get("expr"))
    render = _norm_render(body.get("render") if isinstance(body.get("render"), dict) else {})
    want_tt_sha = str(body.get("truth_table_sha256") or "")
    if not want_tt_sha:
        return False, "missing_truth_table_sha256", {}
    got_tt = truth_table(vars=vars_norm, domain_values=domain_values, expr=expr_canon, render=render)
    got_tt_sha = truth_table_sha256(tt=got_tt)
    if want_tt_sha != got_tt_sha:
        return False, "logic_truth_table_hash_mismatch", {"want": want_tt_sha, "got": got_tt_sha}

    # Exhaustive coverage over assignments: require >=1 vector per assignment.
    tvs = certificate.get("test_vectors") if isinstance(certificate.get("test_vectors"), list) else []
    tvs2 = [tv for tv in tvs if isinstance(tv, dict)]

    expected_by_key: Dict[str, str] = {}
    for a in enumerate_assignments(vars=vars_norm, domain_values=domain_values):
        out_b = logic_eval(expr_canon, a)
        out01 = _render_to_out01(out_b)
        exp = render_output01(out01=out01, render=dict(render))
        key = _stable_hash({"inputs": {str(k): str(a.get(k)) for k in vars_norm}})
        expected_by_key[str(key)] = str(exp)

    present: Dict[str, Dict[str, Any]] = {}
    counterexample = None

    for tv in tvs2:
        inps = tv.get("inputs") if isinstance(tv.get("inputs"), dict) else {}
        sub = {str(k): inps.get(str(k)) for k in vars_norm}
        if any(str(sub.get(k) or "") == "" for k in vars_norm):
            continue
        key = _stable_hash({"inputs": {str(k): str(sub.get(k)) for k in vars_norm}})
        present[str(key)] = {"inputs": {str(k): str(sub.get(k)) for k in vars_norm}}
        want_exp = expected_by_key.get(str(key))
        got_exp = str(tv.get("expected") if tv.get("expected") is not None else "")
        if want_exp is not None and got_exp != str(want_exp) and counterexample is None:
            counterexample = {"inputs": {str(k): str(sub.get(k)) for k in vars_norm}, "want": str(want_exp), "got": str(got_exp)}

    missing_keys = [k for k in sorted(expected_by_key.keys(), key=str) if k not in present]
    if missing_keys:
        missing = [present.get(k) for k in missing_keys if k in present]
        # present[k] doesn't exist for missing; build deterministically from expected_by_key by enumerating.
        miss_inputs: List[Dict[str, Any]] = []
        for a in enumerate_assignments(vars=vars_norm, domain_values=domain_values):
            key = _stable_hash({"inputs": {str(k): str(a.get(k)) for k in vars_norm}})
            if key in set(missing_keys):
                miss_inputs.append({"inputs": {str(k): str(a.get(k)) for k in vars_norm}})
        return False, "logic_vector_coverage_incomplete", {"missing": miss_inputs, "missing_total": int(len(miss_inputs))}

    if counterexample is not None:
        return False, "logic_expected_mismatch", dict(counterexample)

    out_details = dict(details) if isinstance(details, dict) else {}
    out_details["logic_bound"] = True
    out_details["logic_truth_table_sha256"] = str(want_tt_sha)
    out_details["logic_vars"] = list(vars_norm)
    out_details["logic_domain_values"] = list(domain_values)
    out_details["logic_render"] = dict(render)
    out_details["logic_vectors_total"] = int(len(tvs2))
    return True, "ok", out_details

