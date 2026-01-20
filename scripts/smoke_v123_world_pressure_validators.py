#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.external_world_gate_v122 import EXTERNAL_WORLD_ACTION_SEARCH_V122, external_world_access_v122
from atos_core.world_pressure_validators_v123 import (
    EXHAUSTED_BUT_WORLD_HAS_VIABLE_CANDIDATE_V123,
    HISTORICAL_REGRESSION_WITHOUT_CAUSAL_DIFF_V123,
    REUSE_REQUIRED_BUT_NOT_ATTEMPTED_V123,
    consult_external_world_v123,
    validate_exhaustion_with_world_v123,
    validate_historical_regression_v123,
    validate_iac_v123,
    validate_reuse_required_v123,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ensure_absent(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"worm_exists:{path}")


def _write_once_json(path: Path, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise SystemExit(f"tmp_exists:{tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _run_one(*, out_dir: Path, seed: int, manifest: str) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    neg: Dict[str, Dict[str, Any]] = {}
    try:
        external_world_access_v122(
            allowed=False,
            manifest_path=str(manifest),
            action=str(EXTERNAL_WORLD_ACTION_SEARCH_V122),
            reason_code="progress_blocked",
            args={"query": "x", "limit": 1, "source_filter": "engineering_doc", "roles": []},
            seed=int(seed),
            turn_index=0,
            prev_event_sig="",
        )
        neg["access_not_allowed"] = {"ok": True, "reason": "unexpected_ok"}
    except Exception as e:
        neg["access_not_allowed"] = {"ok": False, "reason": str(e)}

    try:
        external_world_access_v122(
            allowed=True,
            manifest_path=str(manifest),
            action=str(EXTERNAL_WORLD_ACTION_SEARCH_V122),
            reason_code="bad_reason_code",
            args={"query": "x", "limit": 1, "source_filter": "engineering_doc", "roles": []},
            seed=int(seed),
            turn_index=0,
            prev_event_sig="",
        )
        neg["invalid_reason_code"] = {"ok": True, "reason": "unexpected_ok"}
    except Exception as e:
        neg["invalid_reason_code"] = {"ok": False, "reason": str(e)}

    consult = consult_external_world_v123(
        manifest_path=str(manifest),
        query="fluência como lei física",
        seed=int(seed),
        turn_index=0,
        prev_event_sig="",
        out_dir=str(out_dir),
        allowed=True,
        reason_code="progress_blocked",
        limit=3,
        source_filter="engineering_doc",
        artifact_prefix="external_world_smoke",
    )

    ok_exh, reason_exh = validate_exhaustion_with_world_v123(exhausted=True, world_hits_total=int(consult.hits_total))
    ok_ar, reason_ar = validate_historical_regression_v123(repeated=True, world_hits_total=int(consult.hits_total), causal_diff_present=False)
    ok_ar2, reason_ar2 = validate_historical_regression_v123(repeated=True, world_hits_total=int(consult.hits_total), causal_diff_present=True)
    ok_iac, reason_iac = validate_iac_v123(goal_ok=False, plan_ok=True, eval_ok=True, consequence_ok=True)
    ok_reuse, reason_reuse = validate_reuse_required_v123(world_hits_total=int(consult.hits_total), reuse_attempted=False)

    eval_obj = {
        "schema_version": 123,
        "seed": int(seed),
        "manifest_sha256": _sha256_file(Path(manifest)),
        "negative": dict(neg),
        "consult": {
            "hits_total": int(consult.hits_total),
            "external_world_chain_hash_v122": str(consult.external_world_chain_hash_v122),
            "events_total": int(len(consult.events)),
            "evidence_total": int(len(consult.evidences)),
        },
        "checks": {
            "exhaustion_with_hits": {"ok": bool(ok_exh), "reason": str(reason_exh)},
            "anti_regression_no_causal_diff": {"ok": bool(ok_ar), "reason": str(reason_ar)},
            "anti_regression_with_causal_diff": {"ok": bool(ok_ar2), "reason": str(reason_ar2)},
            "iac_missing_goal": {"ok": bool(ok_iac), "reason": str(reason_iac)},
            "reuse_required": {"ok": bool(ok_reuse), "reason": str(reason_reuse)},
        },
        "sha256": {
            "events_jsonl": _sha256_file(out_dir / "external_world_smoke_events_v123.jsonl"),
            "evidence_jsonl": _sha256_file(out_dir / "external_world_smoke_evidence_v123.jsonl"),
            "snapshot_json": _sha256_file(out_dir / "external_world_smoke_registry_snapshot_v123.json"),
        },
    }
    _write_once_json(out_dir / "eval.json", eval_obj)

    core = {
        "schema_version": 123,
        "seed": int(seed),
        "manifest_sha256": str(eval_obj["manifest_sha256"]),
        "hits_total": int(consult.hits_total),
        "neg_access_not_allowed": str(neg["access_not_allowed"]["reason"]),
        "neg_invalid_reason_code": str(neg["invalid_reason_code"]["reason"]),
        "expect_exhaustion_reason": str(EXHAUSTED_BUT_WORLD_HAS_VIABLE_CANDIDATE_V123),
        "expect_anti_regression_reason": str(HISTORICAL_REGRESSION_WITHOUT_CAUSAL_DIFF_V123),
        "expect_reuse_reason": str(REUSE_REQUIRED_BUT_NOT_ATTEMPTED_V123),
    }
    summary_sha256 = sha256_hex(canonical_json_dumps(core).encode("utf-8"))
    smoke_summary = {"core": core, "summary_sha256": str(summary_sha256)}
    _write_once_json(out_dir / "smoke_summary.json", smoke_summary)
    return {"eval": eval_obj, "smoke_summary": smoke_summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--manifest", default="external_world_v122/manifest_v122.json")
    args = ap.parse_args()

    seed = int(args.seed)
    out_base = Path(str(args.out_base))
    manifest = str(args.manifest)

    out1 = Path(str(out_base) + "_try1")
    out2 = Path(str(out_base) + "_try2")
    r1 = _run_one(out_dir=out1, seed=seed, manifest=manifest)
    r2 = _run_one(out_dir=out2, seed=seed, manifest=manifest)

    e1 = r1["eval"]
    e2 = r2["eval"]
    if canonical_json_dumps(e1) != canonical_json_dumps(e2):
        raise SystemExit("determinism_failed:eval_json")
    s1 = r1["smoke_summary"]
    s2 = r2["smoke_summary"]
    if canonical_json_dumps(s1) != canonical_json_dumps(s2):
        raise SystemExit("determinism_failed:smoke_summary")

    out = {
        "ok": True,
        "determinism_ok": True,
        "summary_sha256": str(s1["summary_sha256"]),
        "try1_dir": str(out1),
        "try2_dir": str(out2),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
