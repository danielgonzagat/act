from __future__ import annotations

import platform
import sys
from typing import Any, Dict, List, Sequence, Tuple

from .act import Act, canonical_json_dumps, sha256_hex
from .pcc_v74 import certificate_sig_v2
from .pcc_v84 import build_certificate_v84, verify_pcc_v84
from .store import ActStore


def _stable_hash(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _compute_exec_toolchain_sha256_v85() -> str:
    """
    Deterministic toolchain manifest hash:
      - python_implementation (platform.python_implementation())
      - python_version (platform.python_version())
      - sys_platform (sys.platform)
      - machine (platform.machine())
      - installed_packages via importlib.metadata.distributions():
          list of {"name": lower(name), "version": version}
          dedup by (name, version), sorted by (name, version)
    """
    try:
        from importlib import metadata as importlib_metadata  # py3.9+
    except Exception:
        importlib_metadata = None  # type: ignore[assignment]

    pkgs: List[Dict[str, Any]] = []
    seen = set()
    if importlib_metadata is not None:
        try:
            dists = list(importlib_metadata.distributions())
        except Exception:
            dists = []
        for d in dists:
            try:
                md = getattr(d, "metadata", None)
                name = ""
                if md is not None:
                    try:
                        name = str(md.get("Name") or md.get("name") or "")
                    except Exception:
                        name = ""
                if not name:
                    continue
                name_l = str(name).strip().lower()
                version = ""
                try:
                    version = str(getattr(d, "version", "") or "")
                except Exception:
                    version = ""
                key = (name_l, version)
                if key in seen:
                    continue
                seen.add(key)
                pkgs.append({"name": name_l, "version": version})
            except Exception:
                continue

    pkgs.sort(key=lambda r: (str(r.get("name") or ""), str(r.get("version") or "")))

    manifest = {
        "python_implementation": str(platform.python_implementation()),
        "python_version": str(platform.python_version()),
        "sys_platform": str(sys.platform),
        "machine": str(platform.machine()),
        "installed_packages": list(pkgs),
    }
    return sha256_hex(canonical_json_dumps(manifest).encode("utf-8"))


def build_certificate_v85(
    *,
    candidate_act: Act,
    store_base: ActStore,
    mined_from: Dict[str, Any],
    vector_specs: Sequence[Dict[str, Any]],
    seed: int = 0,
) -> Dict[str, Any]:
    """
    V85: PCC wrapper that extends V84 with exec-toolchain binding.
    """
    cert = build_certificate_v84(
        candidate_act=candidate_act,
        store_base=store_base,
        mined_from=mined_from,
        vector_specs=vector_specs,
        seed=int(seed),
    )

    eb = cert.get("exec_env_binding") if isinstance(cert.get("exec_env_binding"), dict) else None
    if not isinstance(eb, dict):
        raise ValueError("missing_exec_env_binding")

    body = dict(eb)
    body.pop("binding_sig", None)
    body["exec_toolchain_sha256"] = _compute_exec_toolchain_sha256_v85()
    # Optional, debug-friendly: keep python_version explicit (also bound by binding_sig).
    body["python_version"] = str(platform.python_version())

    cert["exec_env_binding"] = {**body, "binding_sig": _stable_hash(body)}
    cert["certificate_sig"] = certificate_sig_v2(cert)
    return cert


def verify_pcc_v85(
    *,
    candidate_act: Act,
    certificate: Dict[str, Any],
    store_base: ActStore,
    seed: int = 0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    V85: verify V84 + exec-toolchain binding (fail-closed).
    """
    ok, reason, details = verify_pcc_v84(candidate_act=candidate_act, certificate=certificate, store_base=store_base, seed=int(seed))
    if not bool(ok):
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

    want_toolchain = str(eb.get("exec_toolchain_sha256") or "")
    if not want_toolchain:
        return False, "missing_exec_toolchain_sha256", {}
    got_toolchain = _compute_exec_toolchain_sha256_v85()
    if want_toolchain != got_toolchain:
        return False, "exec_toolchain_hash_mismatch", {"want": want_toolchain, "got": got_toolchain}

    out_details = dict(details) if isinstance(details, dict) else {}
    out_details["exec_toolchain_bound"] = True
    out_details["exec_toolchain_sha256"] = str(got_toolchain)
    return True, "ok", out_details

