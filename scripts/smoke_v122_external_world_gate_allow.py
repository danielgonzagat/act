#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.external_world_gate_v122 import (
    EXTERNAL_WORLD_ACTION_FETCH_V122,
    compute_external_world_chain_hash_v122,
    external_world_access_v122,
    verify_external_world_event_sig_chain_v122,
)
from atos_core.external_world_v122 import ew_load_and_verify


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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once_json(path: Path, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise SystemExit(f"tmp_exists:{tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _write_events_jsonl(path: Path, events: List[Dict[str, Any]]) -> None:
    _ensure_absent(path)
    with open(path, "x", encoding="utf-8") as f:
        for e in events:
            f.write(canonical_json_dumps(e))
            f.write("\n")


def _write_evidence_jsonl(path: Path, evs: List[Dict[str, Any]]) -> None:
    _ensure_absent(path)
    with open(path, "x", encoding="utf-8") as f:
        for e in evs:
            f.write(canonical_json_dumps(e))
            f.write("\n")


def _first_doc_chunk_hit_id(manifest_path: str) -> str:
    world = ew_load_and_verify(manifest_path=str(manifest_path))
    # Load the first chunk id deterministically by reading the chunks file.
    m = _load_json(Path(str(manifest_path)))
    root = Path(str(manifest_path)).resolve().parent
    chunks_rel = str((m.get("paths") or {}).get("engineering_doc_chunks_jsonl") or "")
    chunks_path = (root / chunks_rel).resolve()
    with open(chunks_path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
    if not first:
        raise SystemExit("empty_engineering_chunks")
    obj = json.loads(first)
    cid = str(obj.get("chunk_id") or "")
    if not cid:
        raise SystemExit("missing_chunk_id_first_line")
    # verify fetch works
    _ = world.fetch(hit_id="doc:" + cid, max_chars=10)
    return "doc:" + cid


def _tamper_manifest_fail(manifest_path: Path, tamper_dir: Path) -> Dict[str, Any]:
    _ensure_absent(tamper_dir)
    tamper_dir.mkdir(parents=True, exist_ok=False)
    mp2 = tamper_dir / manifest_path.name
    shutil.copyfile(str(manifest_path), str(mp2))
    m = _load_json(mp2)
    if not isinstance(m, dict):
        raise SystemExit("tamper_manifest_not_dict")
    sha = m.get("sha256") if isinstance(m.get("sha256"), dict) else {}
    # deterministically tamper one sha key
    sha["engineering_doc_plain_v122_txt"] = "0" * 64
    m["sha256"] = dict(sha)
    # recompute manifest_sig deterministically
    body = dict(m)
    body.pop("manifest_sig", None)
    m["manifest_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    mp2.write_text(json.dumps(m, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ok = False
    reason = ""
    try:
        _ = ew_load_and_verify(manifest_path=str(mp2))
        ok = True
    except ValueError as e:
        reason = str(e)
    return {"ok": bool(ok), "reason": str(reason)}


def _run_try(*, manifest: str, out_dir: Path, seed: int) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    hit_id = _first_doc_chunk_hit_id(str(manifest))
    events, evidences, summary = external_world_access_v122(
        allowed=True,
        manifest_path=str(manifest),
        action=EXTERNAL_WORLD_ACTION_FETCH_V122,
        reason_code="progress_blocked",
        args={"hit_id": str(hit_id)},
        seed=int(seed),
        turn_index=0,
        prev_event_sig="",
        max_chars=800,
    )

    events_path = out_dir / "external_world_events_v122.jsonl"
    evidence_path = out_dir / "external_world_evidence_v122.jsonl"
    _write_events_jsonl(events_path, list(events))
    _write_evidence_jsonl(evidence_path, list(evidences))

    ok_chain, reason_chain, details_chain = verify_external_world_event_sig_chain_v122(list(events))
    if not ok_chain:
        raise SystemExit("sig_chain_fail:" + str(reason_chain))
    chain_hash = compute_external_world_chain_hash_v122(list(events))
    snap = {
        "schema_version": 122,
        "kind": "external_world_registry_snapshot_v122",
        "events_total": int(len(events)),
        "external_world_chain_hash_v122": str(chain_hash),
    }
    snap_path = out_dir / "external_world_registry_snapshot_v122.json"
    _write_once_json(snap_path, snap)

    return {
        "events_total": int(len(events)),
        "external_world_chain_hash_v122": str(chain_hash),
        "sha256": {
            "events_jsonl": _sha256_file(events_path),
            "evidence_jsonl": _sha256_file(evidence_path),
            "snapshot_json": _sha256_file(snap_path),
        },
        "result_summary": dict(summary),
        "chain_verify": {"ok": bool(ok_chain), "reason": str(reason_chain), "details": dict(details_chain)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()

    seed = int(args.seed)
    out_base = Path(str(args.out_base))
    out1 = Path(str(out_base) + "_try1")
    out2 = Path(str(out_base) + "_try2")

    r1 = _run_try(manifest=str(args.manifest), out_dir=out1, seed=seed)
    r2 = _run_try(manifest=str(args.manifest), out_dir=out2, seed=seed)

    if canonical_json_dumps(r1) != canonical_json_dumps(r2):
        raise SystemExit("determinism_failed:try_payload")

    tamper_dir = Path(str(out_base) + "_tamper")
    tamper = _tamper_manifest_fail(Path(str(args.manifest)).resolve(), tamper_dir)
    if tamper["reason"] != "external_world_manifest_mismatch_v122":
        raise SystemExit("negative_failed:manifest_mismatch_reason")

    core = {
        "schema_version": 122,
        "seed": int(seed),
        "try": dict(r1),
        "negative_manifest_tamper": dict(tamper),
    }
    summary_sha256 = sha256_hex(canonical_json_dumps(core).encode("utf-8"))
    out = {
        "ok": True,
        "determinism_ok": True,
        "summary_sha256": str(summary_sha256),
        "core": core,
        "try1_dir": str(out1),
        "try2_dir": str(out2),
        "tamper_dir": str(tamper_dir),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
