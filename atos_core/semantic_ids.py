from __future__ import annotations

from typing import Any, Dict

from .act import canonical_json_dumps, sha256_hex


def plan_id_for_expected_spec_sig(expected_spec_sig: str) -> str:
    sig = str(expected_spec_sig or "").strip()
    if not sig:
        return ""
    return f"plan_{sig[:16]}"


def hypothesis_id_v1(*, goal_id: str, reason: str, context: Dict[str, Any]) -> str:
    body = {"goal_id": str(goal_id or ""), "reason": str(reason or ""), "context": dict(context)}
    sig = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return f"hyp_{sig[:16]}"


def reference_id_v1(*, scope_sig: str, token: str) -> str:
    body = {"scope": str(scope_sig or ""), "token": str(token or "")}
    sig = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return f"ref_{sig[:16]}"

