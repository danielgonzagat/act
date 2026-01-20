from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Sequence

from .conversation_loop_v110 import run_conversation_v110
from .conversation_v96 import verify_chained_jsonl_v96
from .external_world_ledger_v111 import compute_external_world_chain_hash_v111, verify_external_world_event_sig_chain_v111
from .fluency_contract_v111 import fluency_contract_v111


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def run_conversation_v111(
    *,
    user_turn_texts: Sequence[str],
    out_dir: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Wrapper over V110:
      - Runs V110 conversation pipeline.
      - Adds V111 read-only artifacts (external_world_events ledger placeholder + fluency contract report).
      - Produces freeze_manifest_v111.json (WORM) + summary_v111.json (WORM).
    """
    run_conversation_v110(user_turn_texts=list(user_turn_texts), out_dir=str(out_dir), seed=int(seed))

    ext_events_path = os.path.join(str(out_dir), "external_world_events.jsonl")
    ext_snapshot_path = os.path.join(str(out_dir), "external_world_registry_snapshot_v111.json")
    fluency_path = os.path.join(str(out_dir), "fluency_contract_v111.json")
    manifest_path = os.path.join(str(out_dir), "freeze_manifest_v111.json")
    summary_path = os.path.join(str(out_dir), "summary_v111.json")

    # External world ledger must exist (empty by default).
    if os.path.exists(ext_events_path):
        _fail(f"worm_exists:{ext_events_path}")
    with open(ext_events_path, "x", encoding="utf-8") as f:
        f.write("")

    # Verify empty ledger ok.
    ext_events: list = []
    ok_sig, reason_sig, _ = verify_external_world_event_sig_chain_v111(ext_events)
    if not ok_sig or str(reason_sig) != "ok":
        _fail(f"external_world_sig_chain_fail:{reason_sig}")
    ext_chain_hash = compute_external_world_chain_hash_v111(ext_events)
    if os.path.exists(ext_snapshot_path):
        _fail(f"worm_exists:{ext_snapshot_path}")
    with open(ext_snapshot_path, "x", encoding="utf-8") as f:
        f.write(json.dumps({"schema_version": 111, "events_total": 0, "external_world_chain_hash_v111": ext_chain_hash}, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")

    # Fluency contract on transcript.
    transcript_path = os.path.join(str(out_dir), "transcript.jsonl")
    transcript: list = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            transcript.append(json.loads(line))
    ok_fc, reason_fc, details_fc = fluency_contract_v111(transcript=transcript)
    if os.path.exists(fluency_path):
        _fail(f"worm_exists:{fluency_path}")
    with open(fluency_path, "x", encoding="utf-8") as f:
        f.write(json.dumps({"ok": bool(ok_fc), "reason": str(reason_fc), "details": dict(details_fc)}, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")

    # Freeze manifest v111 (WORM).
    if os.path.exists(manifest_path):
        _fail(f"worm_exists:{manifest_path}")
    sha256 = {
        "v110_freeze_manifest_v110_json": _sha256_file(os.path.join(str(out_dir), "freeze_manifest_v110.json")),
        "external_world_events_jsonl": _sha256_file(ext_events_path),
        "external_world_registry_snapshot_v111_json": _sha256_file(ext_snapshot_path),
        "fluency_contract_v111_json": _sha256_file(fluency_path),
    }
    with open(manifest_path, "x", encoding="utf-8") as f:
        f.write(json.dumps({"schema_version": 111, "kind": "freeze_manifest_v111", "sha256": sha256}, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")

    # Summary v111 (WORM).
    if os.path.exists(summary_path):
        _fail(f"worm_exists:{summary_path}")
    ledger_hash = _sha256_file(manifest_path)
    with open(summary_path, "x", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "schema_version": 111,
                    "seed": int(seed),
                    "external_world_chain_hash_v111": str(ext_chain_hash),
                    "fluency_contract_ok_v111": bool(ok_fc),
                    "fluency_contract_reason_v111": str(reason_fc),
                    "ledger_hash": str(ledger_hash),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        f.write("\n")

    return {"ok": True, "out_dir": str(out_dir), "ledger_hash": str(ledger_hash)}

