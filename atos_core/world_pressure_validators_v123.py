from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .external_world_gate_v122 import (
    EXTERNAL_WORLD_ACTION_SEARCH_V122,
    compute_external_world_chain_hash_v122,
    external_world_access_v122,
    verify_external_world_event_sig_chain_v122,
)

WORLD_PRESSURE_SCHEMA_VERSION_V123 = 123

# Closed set of V123 reason codes (hard gate).
HISTORICAL_REGRESSION_WITHOUT_CAUSAL_DIFF_V123 = "historical_regression_without_causal_diff_v123"
HISTORICAL_WORLD_IGNORED_V123 = "historical_world_ignored_v123"
REUSE_REQUIRED_BUT_NOT_ATTEMPTED_V123 = "reuse_required_but_not_attempted_v123"
IAC_MISSING_GOAL_V123 = "iac_missing_goal_v123"
IAC_MISSING_PLAN_V123 = "iac_missing_plan_v123"
IAC_MISSING_EVAL_V123 = "iac_missing_eval_v123"
IAC_MISSING_CONSEQUENCE_V123 = "iac_missing_consequence_v123"
EXHAUSTED_BUT_WORLD_HAS_VIABLE_CANDIDATE_V123 = "exhausted_but_world_has_viable_candidate_v123"
WORLD_EVIDENCE_MISSING_OR_INVALID_V123 = "world_evidence_missing_or_invalid_v123"

WORLD_PRESSURE_REASON_CODES_V123 = {
    HISTORICAL_REGRESSION_WITHOUT_CAUSAL_DIFF_V123,
    HISTORICAL_WORLD_IGNORED_V123,
    REUSE_REQUIRED_BUT_NOT_ATTEMPTED_V123,
    IAC_MISSING_GOAL_V123,
    IAC_MISSING_PLAN_V123,
    IAC_MISSING_EVAL_V123,
    IAC_MISSING_CONSEQUENCE_V123,
    EXHAUSTED_BUT_WORLD_HAS_VIABLE_CANDIDATE_V123,
    WORLD_EVIDENCE_MISSING_OR_INVALID_V123,
}


def _ensure_absent(path: str) -> None:
    if os.path.exists(path):
        raise ValueError(f"worm_exists:{path}")
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        raise ValueError(f"tmp_exists:{tmp}")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_once_json(path: str, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)


def _write_jsonl_x(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    _ensure_absent(path)
    with open(path, "x", encoding="utf-8") as f:
        for r in rows:
            f.write(canonical_json_dumps(r))
            f.write("\n")


def fail_signature_v123(*, validator_name: str, reason_code: str, context: Dict[str, Any]) -> str:
    """
    Deterministic failure signature for anti-regression lookup.
    """
    body = {
        "schema_version": int(WORLD_PRESSURE_SCHEMA_VERSION_V123),
        "validator_name": str(validator_name),
        "reason_code": str(reason_code),
        "context": dict(context),
    }
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def should_consult_world_v123(*, reason_code: str, repeated_count: int, is_exhaustion: bool) -> bool:
    if bool(is_exhaustion):
        return True
    return int(repeated_count) >= 1


@dataclass(frozen=True)
class ExternalWorldConsultV123:
    query: str
    hits_total: int
    evidence_ids: List[str]
    external_world_chain_hash_v122: str
    result_summary: Dict[str, Any]
    events: List[Dict[str, Any]]
    evidences: List[Dict[str, Any]]


def consult_external_world_v123(
    *,
    manifest_path: str,
    query: str,
    seed: int,
    turn_index: int,
    prev_event_sig: str,
    out_dir: str,
    allowed: bool,
    reason_code: str,
    limit: int = 3,
    source_filter: str = "engineering_doc",
    artifact_prefix: str = "external_world",
) -> ExternalWorldConsultV123:
    """
    Consult the unified ExternalWorld via the V122 gate and write WORM artifacts under out_dir.
    """
    evs, evids, result_summary = external_world_access_v122(
        allowed=bool(allowed),
        manifest_path=str(manifest_path),
        action=str(EXTERNAL_WORLD_ACTION_SEARCH_V122),
        reason_code=str(reason_code),
        args={"query": str(query), "limit": int(limit), "source_filter": str(source_filter), "roles": []},
        seed=int(seed),
        turn_index=int(turn_index),
        prev_event_sig=str(prev_event_sig or ""),
    )

    ap = str(artifact_prefix or "external_world")
    events_path = os.path.join(str(out_dir), f"{ap}_events_v123.jsonl")
    evidence_path = os.path.join(str(out_dir), f"{ap}_evidence_v123.jsonl")
    snap_path = os.path.join(str(out_dir), f"{ap}_registry_snapshot_v123.json")
    _write_jsonl_x(events_path, list(evs))
    _write_jsonl_x(evidence_path, list(evids))

    ok_sig, reason_sig, _ = verify_external_world_event_sig_chain_v122(list(evs))
    if not bool(ok_sig):
        raise ValueError(f"external_world_event_sig_chain_fail:{reason_sig}")
    chain_hash = compute_external_world_chain_hash_v122(list(evs))

    snap = {
        "schema_version": int(WORLD_PRESSURE_SCHEMA_VERSION_V123),
        "kind": "external_world_registry_snapshot_v123",
        "events_total": int(len(evs)),
        "external_world_chain_hash_v122": str(chain_hash),
        "sha256": {
            "events_jsonl": _sha256_file(events_path),
            "evidence_jsonl": _sha256_file(evidence_path),
        },
    }
    snap["snapshot_sig"] = sha256_hex(canonical_json_dumps(snap).encode("utf-8"))
    _write_once_json(snap_path, snap)

    hits_total = int(result_summary.get("hits_total") or 0)
    evidence_ids = list(result_summary.get("evidence_ids") or [])
    if not evidence_ids:
        # Derive evidence IDs from the evidence objects.
        evidence_ids = [str(e.get("evidence_id") or "") for e in evids if isinstance(e, dict)]
        evidence_ids = [x for x in evidence_ids if x]
    return ExternalWorldConsultV123(
        query=str(query),
        hits_total=int(hits_total),
        evidence_ids=list(evidence_ids),
        external_world_chain_hash_v122=str(chain_hash),
        result_summary=dict(result_summary),
        events=list(evs),
        evidences=list(evids),
    )


def validate_iac_v123(
    *,
    goal_ok: bool,
    plan_ok: bool,
    eval_ok: bool,
    consequence_ok: bool,
) -> Tuple[bool, str]:
    """
    Hard IAC gate: Intention -> Action -> Consequence must exist.
    """
    if not bool(goal_ok):
        return False, IAC_MISSING_GOAL_V123
    if not bool(plan_ok):
        return False, IAC_MISSING_PLAN_V123
    if not bool(eval_ok):
        return False, IAC_MISSING_EVAL_V123
    if not bool(consequence_ok):
        return False, IAC_MISSING_CONSEQUENCE_V123
    return True, "ok"


def validate_historical_regression_v123(
    *,
    repeated: bool,
    world_hits_total: int,
    causal_diff_present: bool,
) -> Tuple[bool, str]:
    """
    If a failure repeats and the world contains any evidence hits, a causal diff is required.
    """
    if bool(repeated) and int(world_hits_total) > 0 and not bool(causal_diff_present):
        return False, HISTORICAL_REGRESSION_WITHOUT_CAUSAL_DIFF_V123
    return True, "ok"


def validate_exhaustion_with_world_v123(
    *,
    exhausted: bool,
    world_hits_total: int,
) -> Tuple[bool, str]:
    """
    If exhaustion is declared and the world has any hits for the query, treat as not-proved (fail).
    """
    if bool(exhausted) and int(world_hits_total) > 0:
        return False, EXHAUSTED_BUT_WORLD_HAS_VIABLE_CANDIDATE_V123
    return True, "ok"


def validate_reuse_required_v123(
    *,
    world_hits_total: int,
    reuse_attempted: bool,
) -> Tuple[bool, str]:
    """
    Hard gate: if the world indicates a viable candidate exists, require an explicit reuse attempt
    before declaring failure/exhaustion.

    In V123 minimal form, "reuse_attempted" is a structured boolean set by the caller (runtime).
    """
    if int(world_hits_total) > 0 and not bool(reuse_attempted):
        return False, REUSE_REQUIRED_BUT_NOT_ATTEMPTED_V123
    return True, "ok"
