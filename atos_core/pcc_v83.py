from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from .act import Act, canonical_json_dumps, sha256_hex
from .pcc_v74 import certificate_sig_v2
from .pcc_v82 import build_certificate_v82, verify_pcc_v82
from .store import ActStore

ENGINE_ID_V83 = "EngineV80"


def _stable_hash(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def build_certificate_v83(
    *,
    candidate_act: Act,
    store_base: ActStore,
    mined_from: Dict[str, Any],
    vector_specs: Sequence[Dict[str, Any]],
    seed: int = 0,
) -> Dict[str, Any]:
    """
    V83: PCC wrapper that is:
      - match-bound (V81)
      - goal_kind-exec-bound (V82)
      - exec-env-bound (V83): binds store_hash_base + engine_id + seed.
    """
    cert = build_certificate_v82(
        candidate_act=candidate_act,
        store_base=store_base,
        mined_from=mined_from,
        vector_specs=vector_specs,
        seed=int(seed),
    )

    body = {
        "store_hash_base": str(store_base.content_hash()),
        "engine_id": str(ENGINE_ID_V83),
        "seed": int(seed),
    }
    cert["exec_env_binding"] = {**dict(body), "binding_sig": _stable_hash(body)}

    cert["certificate_sig"] = certificate_sig_v2(cert)
    return cert


def verify_pcc_v83(
    *,
    candidate_act: Act,
    certificate: Dict[str, Any],
    store_base: ActStore,
    seed: int = 0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    V83: verify PCC V82 + exec-env-binding (fail-closed).
    """
    ok, reason, details = verify_pcc_v82(candidate_act=candidate_act, certificate=certificate, store_base=store_base, seed=int(seed))
    if not bool(ok):
        # Normalize store mismatch reason for V83 smoke/audit.
        if str(reason) == "store_hash_base_mismatch":
            return False, "store_hash_mismatch", dict(details)
        return False, str(reason), dict(details)

    eb = certificate.get("exec_env_binding") if isinstance(certificate.get("exec_env_binding"), dict) else None
    if not isinstance(eb, dict):
        return False, "missing_exec_env_binding", {}

    want_sig = str(eb.get("binding_sig") or "")
    body = dict(eb)
    body.pop("binding_sig", None)
    got_sig = _stable_hash(body)
    if want_sig != got_sig:
        return False, "exec_env_binding_sig_mismatch", {"want": want_sig, "got": got_sig}

    # store_hash_base must bind to current store_base state.
    want_store = str(eb.get("store_hash_base") or "")
    got_store = str(store_base.content_hash())
    if want_store != got_store:
        return False, "store_hash_mismatch", {"want": want_store, "got": got_store}

    # engine_id must match the executor identity.
    got_engine_id = str(eb.get("engine_id") or "")
    if got_engine_id != str(ENGINE_ID_V83):
        return False, "engine_id_mismatch", {"want": str(ENGINE_ID_V83), "got": got_engine_id}

    # seed must match the effective seed used for verification.
    want_seed = int(seed)
    cert_seed_raw = eb.get("seed")
    try:
        cert_seed = int(cert_seed_raw)
    except Exception:
        return False, "seed_mismatch", {"want": want_seed, "got": cert_seed_raw}
    if cert_seed != want_seed:
        return False, "seed_mismatch", {"want": want_seed, "got": cert_seed}

    out_details = dict(details) if isinstance(details, dict) else {}
    out_details["exec_env_bound"] = True
    out_details["exec_env_binding"] = {
        "store_hash_base": str(want_store),
        "engine_id": str(got_engine_id),
        "seed": int(cert_seed),
    }
    return True, "ok", out_details

