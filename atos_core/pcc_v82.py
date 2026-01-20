from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence, Tuple

from .act import Act, canonical_json_dumps, sha256_hex
from .engine_v80 import EngineV80
from .pcc_v74 import certificate_sig_v2
from .pcc_v81 import build_certificate_v81, verify_pcc_v81
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


def _safe_deepcopy(obj: Any) -> Any:
    try:
        return copy.deepcopy(obj)
    except Exception:
        if isinstance(obj, dict):
            return dict(obj)
        if isinstance(obj, list):
            return list(obj)
        return obj


def _clone_store_with_candidate(*, store_base: ActStore, candidate_act: Act) -> ActStore:
    store2 = ActStore()
    for act_id in sorted(store_base.acts.keys(), key=str):
        act = store_base.acts[act_id]
        store2.add(Act.from_dict(act.to_dict()))
    store2.add(Act.from_dict(candidate_act.to_dict()))
    return store2


def build_certificate_v82(
    *,
    candidate_act: Act,
    store_base: ActStore,
    mined_from: Dict[str, Any],
    vector_specs: Sequence[Dict[str, Any]],
    seed: int = 0,
) -> Dict[str, Any]:
    """
    V82: PCC wrapper that is match-bound (V81) AND goal_kind-exec-bound:
      - vector_specs must include explicit goal_kind
      - certificate records goal_kind coverage + binds goal_kind to each test_vector
    """
    if not isinstance(vector_specs, (list, tuple)):
        raise ValueError("vector_specs_not_sequence")

    norm_specs: List[Dict[str, Any]] = []
    ctx_to_kind: Dict[str, str] = {}
    vector_goal_kinds_raw: List[str] = []

    for i, spec in enumerate(vector_specs):
        if not isinstance(spec, dict):
            raise ValueError("vector_spec_not_dict")
        gk = str(spec.get("goal_kind") or "")
        if not gk:
            raise ValueError("missing_vector_goal_kind")
        inputs = spec.get("inputs") if isinstance(spec.get("inputs"), dict) else {}
        expected = spec.get("expected")
        expected_sig = _stable_hash(
            {"inputs": {str(k): inputs.get(k) for k in sorted(inputs.keys(), key=str)}, "expected": expected}
        )
        ctx = str(spec.get("context_id") or "")
        if not ctx:
            ctx = f"v82:{gk}:{expected_sig}:{int(i)}"
        # Ensure deterministic uniqueness of context_id.
        if ctx in ctx_to_kind:
            ctx = f"{ctx}:{int(i)}"
        ctx_to_kind[str(ctx)] = str(gk)

        vector_goal_kinds_raw.append(str(gk))
        norm_specs.append({**dict(spec), "context_id": str(ctx), "goal_kind": str(gk)})

    cert = build_certificate_v81(
        candidate_act=candidate_act,
        store_base=store_base,
        mined_from=mined_from,
        vector_specs=norm_specs,
        seed=int(seed),
    )

    # Bind each test_vector to its goal_kind (by context_id).
    tvs = cert.get("test_vectors") if isinstance(cert.get("test_vectors"), list) else []
    tvs2: List[Dict[str, Any]] = []
    for tv in tvs:
        if not isinstance(tv, dict):
            continue
        ctx = str(tv.get("context_id") or "")
        gk = str(ctx_to_kind.get(ctx, "") or "")
        tv2 = dict(tv)
        tv2["goal_kind"] = str(gk)
        tvs2.append(tv2)
    cert["test_vectors"] = list(tvs2)

    # Record goal_kind exec binding / coverage.
    vector_goal_kinds_norm = _norm_str_list(vector_goal_kinds_raw)
    binding_body = {
        "vector_goal_kinds_raw": [str(x or "") for x in vector_goal_kinds_raw],
        "vector_goal_kinds_norm": list(vector_goal_kinds_norm),
        "vector_goal_kinds_distinct": int(len(vector_goal_kinds_norm)),
    }
    cert["goal_kind_exec_binding"] = {**binding_body, "binding_sig": _stable_hash(binding_body)}

    # Recompute certificate_sig to include goal_kind_exec_binding and augmented test_vectors.
    cert["certificate_sig"] = certificate_sig_v2(cert)
    return cert


def verify_pcc_v82(
    *,
    candidate_act: Act,
    certificate: Dict[str, Any],
    store_base: ActStore,
    seed: int = 0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    V82: verify PCC v2 + match-binding (V81) + goal_kind-exec-binding (V82).
    """
    ok, reason, details = verify_pcc_v81(candidate_act=candidate_act, certificate=certificate, store_base=store_base, seed=int(seed))
    if not bool(ok):
        return False, str(reason), dict(details)

    gkeb = certificate.get("goal_kind_exec_binding") if isinstance(certificate.get("goal_kind_exec_binding"), dict) else None
    if not isinstance(gkeb, dict):
        return False, "missing_goal_kind_exec_binding", {}

    want_sig = str(gkeb.get("binding_sig") or "")
    body = dict(gkeb)
    body.pop("binding_sig", None)
    got_sig = _stable_hash(body)
    if want_sig != got_sig:
        return False, "goal_kind_exec_binding_sig_mismatch", {"want": want_sig, "got": got_sig}

    mb = certificate.get("match_binding") if isinstance(certificate.get("match_binding"), dict) else {}
    cert_supported = _norm_str_list(mb.get("cert_goal_kinds_supported"))
    if not cert_supported:
        return False, "missing_cert_goal_kinds_supported", {"match_binding": dict(mb)}

    tvs = certificate.get("test_vectors") if isinstance(certificate.get("test_vectors"), list) else []
    tvs2 = [tv for tv in tvs if isinstance(tv, dict)]
    vector_goal_kinds = _norm_str_list([tv.get("goal_kind") for tv in tvs2])

    missing = [k for k in cert_supported if k not in set(vector_goal_kinds)]
    if missing:
        return False, "missing_vector_goal_kind_coverage", {"missing_goal_kinds": list(missing), "present_goal_kinds": list(vector_goal_kinds)}

    # Re-execute vectors under match-enforced EngineV80 with explicit goal_kind.
    store_exec = _clone_store_with_candidate(store_base=store_base, candidate_act=candidate_act)
    engine = EngineV80(store_exec, seed=int(seed))

    for i, tv in enumerate(tvs2):
        gk = str(tv.get("goal_kind") or "")
        if not gk:
            return False, "missing_vector_goal_kind", {"idx": int(i)}
        inputs = tv.get("inputs") if isinstance(tv.get("inputs"), dict) else {}
        expected = tv.get("expected")
        exp_text = str("" if expected is None else expected)

        out = engine.execute_concept_csv(
            concept_act_id=str(candidate_act.id),
            inputs={str(k): inputs.get(k) for k in sorted(inputs.keys(), key=str)},
            goal_kind=str(gk),
            expected=None,
            step=0,
            max_depth=8,
            max_events=256,
            validate_output=False,
        )
        meta = out.get("meta") if isinstance(out, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        if not bool(meta.get("ok", False)):
            callee_meta = meta.get("callee_meta") if isinstance(meta.get("callee_meta"), dict) else {}
            return (
                False,
                "vector_exec_failed",
                {
                    "idx": int(i),
                    "failing_goal_kind": str(gk),
                    "concept_id": str(meta.get("concept_id") or candidate_act.id),
                    "meta_reason": str(meta.get("reason") or ""),
                    "callee_meta_reason": str(callee_meta.get("reason") or "") if isinstance(callee_meta, dict) else "",
                    "context_id": str(tv.get("context_id") or ""),
                    "expected_sig": str(tv.get("expected_sig") or ""),
                },
            )

        got_text = str(meta.get("output_text") or out.get("output") or "")
        if got_text != exp_text:
            return (
                False,
                "vector_output_mismatch",
                {
                    "idx": int(i),
                    "failing_goal_kind": str(gk),
                    "expected": str(exp_text),
                    "got": str(got_text),
                    "context_id": str(tv.get("context_id") or ""),
                    "expected_sig": str(tv.get("expected_sig") or ""),
                },
            )

    return True, "ok", {"match_bound": True, "exec_bound": True, "coverage_ok": True, "vector_goal_kinds_norm": list(vector_goal_kinds)}

