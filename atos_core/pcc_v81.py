from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .act import Act, canonical_json_dumps, sha256_hex
from .pcc_v74 import build_certificate_v2, certificate_sig_v2, verify_pcc_v2
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


def _match_goal_kinds_raw_from_act(act: Act) -> Any:
    mk = act.match if isinstance(getattr(act, "match", None), dict) else {}
    return mk.get("goal_kinds")


def build_certificate_v81(
    *,
    candidate_act: Act,
    store_base: ActStore,
    mined_from: Dict[str, Any],
    vector_specs: Sequence[Dict[str, Any]],
    seed: int = 0,
) -> Dict[str, Any]:
    """
    V81: PCC v2 wrapper that *binds* match.goal_kinds to the certificate.

    - Calls PCC v2 builder.
    - Normalizes mined_from fields deterministically.
    - Injects `match_binding` and recomputes `certificate_sig`.
    """
    mf = dict(mined_from) if isinstance(mined_from, dict) else {}

    # Normalize mined_from critical fields.
    mf_trace_sigs = _norm_str_list(mf.get("trace_sigs"))
    mf_goal_kinds = _norm_str_list(mf.get("goal_kinds"))
    mf_goal_kinds_distinct = int(len(mf_goal_kinds))

    cand = mf.get("candidate") if isinstance(mf.get("candidate"), dict) else {}
    cand = dict(cand)
    cand_supported = _norm_str_list(cand.get("goal_kinds_supported"))
    cand["goal_kinds_supported"] = list(cand_supported)
    mf["candidate"] = cand

    mf["trace_sigs"] = list(mf_trace_sigs)
    mf["goal_kinds"] = list(mf_goal_kinds)
    mf["goal_kinds_distinct"] = int(mf_goal_kinds_distinct)

    cert = build_certificate_v2(
        candidate_act=candidate_act,
        store_base=store_base,
        mined_from=mf,
        vector_specs=vector_specs,
        seed=int(seed),
    )

    # Match binding block.
    raw = _match_goal_kinds_raw_from_act(candidate_act)
    raw_is_list = isinstance(raw, list)
    raw_list = [str(x or "") for x in raw] if isinstance(raw, list) else []
    candidate_match_goal_kinds = [s for s in raw_list if s]
    candidate_match_goal_kinds_norm = sorted(set(candidate_match_goal_kinds), key=str)

    binding_body = {
        "candidate_match_goal_kinds": list(candidate_match_goal_kinds_norm),
        "candidate_match_goal_kinds_raw": list(raw_list),
        "candidate_match_goal_kinds_is_list": bool(raw_is_list),
        "cert_goal_kinds_supported": list(cand_supported),
    }
    binding_sig = _stable_hash(binding_body)
    cert["match_binding"] = {**binding_body, "binding_sig": str(binding_sig)}

    # IMPORTANT: certificate_sig must include match_binding for integrity.
    cert["certificate_sig"] = certificate_sig_v2(cert)
    return cert


def verify_pcc_v81(
    *,
    candidate_act: Act,
    certificate: Dict[str, Any],
    store_base: ActStore,
    seed: int = 0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    V81: PCC v2 verifier + match-binding enforcement (fail-closed).
    """
    ok, reason, details = verify_pcc_v2(candidate_act=candidate_act, certificate=certificate, store_base=store_base, seed=int(seed))
    if not bool(ok):
        return False, str(reason), dict(details)

    # Require match_binding.
    mb = certificate.get("match_binding") if isinstance(certificate.get("match_binding"), dict) else None
    if not isinstance(mb, dict):
        return False, "missing_match_binding", {}

    # Verify match_binding integrity.
    want_binding_sig = str(mb.get("binding_sig") or "")
    body = dict(mb)
    body.pop("binding_sig", None)
    got_binding_sig = _stable_hash(body)
    if want_binding_sig != got_binding_sig:
        return False, "binding_sig_mismatch", {"want": want_binding_sig, "got": got_binding_sig}

    # Rule A: promoted candidates must have explicit non-empty match.goal_kinds list with no empty strings.
    raw = _match_goal_kinds_raw_from_act(candidate_act)
    if not isinstance(raw, list) or len(raw) == 0:
        return False, "missing_match_goal_kinds", {"match_goal_kinds": raw}
    raw_list = [str(x or "") for x in raw]
    if any(not s for s in raw_list):
        return False, "missing_match_goal_kinds", {"match_goal_kinds": raw_list}

    match_norm = sorted(set(raw_list), key=str)

    # Rule B: candidate match must equal certified supported kinds.
    cert_supported = mb.get("cert_goal_kinds_supported")
    cert_norm = _norm_str_list(cert_supported)
    if match_norm != cert_norm:
        return False, "match_goal_kinds_mismatch", {"candidate_match_goal_kinds": list(match_norm), "cert_goal_kinds_supported": list(cert_norm)}

    # Rule C: cert_supported must be consistent with mined_from.goal_kinds (if present).
    mf = certificate.get("mined_from") if isinstance(certificate.get("mined_from"), dict) else {}
    mf_goal_kinds = _norm_str_list(mf.get("goal_kinds"))
    if "goal_kinds" in mf and not mf_goal_kinds:
        return False, "mined_from_goal_kinds_inconsistent", {"mined_from_goal_kinds": mf.get("goal_kinds")}
    if mf_goal_kinds:
        want_distinct = int(mf.get("goal_kinds_distinct", 0) or 0)
        if want_distinct != int(len(mf_goal_kinds)):
            return False, "mined_from_goal_kinds_inconsistent", {"want": want_distinct, "got": int(len(mf_goal_kinds)), "goal_kinds": list(mf_goal_kinds)}
        if not set(cert_norm).issubset(set(mf_goal_kinds)):
            return False, "mined_from_goal_kinds_inconsistent", {"cert_goal_kinds_supported": list(cert_norm), "mined_from_goal_kinds": list(mf_goal_kinds)}

    return True, "ok", {"match_bound": True, "match_goal_kinds": list(match_norm), "cert_goal_kinds_supported": list(cert_norm)}

