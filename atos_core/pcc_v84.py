from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Sequence, Tuple

from .act import Act, canonical_json_dumps, sha256_hex
from .pcc_v74 import certificate_sig_v2
from .pcc_v83 import ENGINE_ID_V83, build_certificate_v83, verify_pcc_v83
from .store import ActStore


def _stable_hash(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _sha256_file_bytes(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _compute_exec_code_sha256_v84() -> str:
    """
    Deterministic manifest hash of executor code:
      - all *.py under atos_core/
      - exclude __pycache__ dirs
      - manifest entries sorted by relpath (repo-root relative, '/' separators)
      - exec_code_sha256 = sha256(canonical_json_dumps(manifest))
    """
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
    atos_dir = os.path.join(repo_root, "atos_core")
    files: List[str] = []
    for root, dirnames, filenames in os.walk(atos_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not str(fn).endswith(".py"):
                continue
            files.append(os.path.join(root, fn))

    entries: List[Dict[str, Any]] = []
    for p in sorted(files, key=lambda x: os.path.relpath(x, repo_root).replace(os.sep, "/")):
        rel = os.path.relpath(p, repo_root).replace(os.sep, "/")
        entries.append({"relpath": str(rel), "sha256_file_bytes": str(_sha256_file_bytes(p))})

    return sha256_hex(canonical_json_dumps(entries).encode("utf-8"))


def build_certificate_v84(
    *,
    candidate_act: Act,
    store_base: ActStore,
    mined_from: Dict[str, Any],
    vector_specs: Sequence[Dict[str, Any]],
    seed: int = 0,
) -> Dict[str, Any]:
    """
    V84: PCC wrapper that is:
      - match-bound (V81)
      - goal_kind-exec-bound (V82)
      - exec-env-bound (V83): store_hash_base + engine_id + seed
      - exec-code-bound (V84): binds exec_code_sha256 (manifest of atos_core/*.py)
    """
    cert = build_certificate_v83(
        candidate_act=candidate_act,
        store_base=store_base,
        mined_from=mined_from,
        vector_specs=vector_specs,
        seed=int(seed),
    )

    exec_code_sha256 = _compute_exec_code_sha256_v84()

    body = {
        "store_hash_base": str(store_base.content_hash()),
        "engine_id": str(ENGINE_ID_V83),
        "seed": int(seed),
        "exec_code_sha256": str(exec_code_sha256),
    }
    cert["exec_env_binding"] = {**dict(body), "binding_sig": _stable_hash(body)}
    cert["certificate_sig"] = certificate_sig_v2(cert)
    return cert


def verify_pcc_v84(
    *,
    candidate_act: Act,
    certificate: Dict[str, Any],
    store_base: ActStore,
    seed: int = 0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    V84: verify PCC v83 + exec-code-binding (fail-closed).
    """
    ok, reason, details = verify_pcc_v83(candidate_act=candidate_act, certificate=certificate, store_base=store_base, seed=int(seed))
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

    want_exec_code = str(eb.get("exec_code_sha256") or "")
    if not want_exec_code:
        return False, "missing_exec_code_sha256", {}
    got_exec_code = _compute_exec_code_sha256_v84()
    if want_exec_code != got_exec_code:
        return False, "exec_code_hash_mismatch", {"want": want_exec_code, "got": got_exec_code}

    out_details = dict(details) if isinstance(details, dict) else {}
    out_details["exec_code_bound"] = True
    out_details["exec_code_sha256"] = str(got_exec_code)
    return True, "ok", out_details

