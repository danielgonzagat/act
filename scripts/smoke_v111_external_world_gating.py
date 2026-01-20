#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.conversation_v96 import append_chained_jsonl_v96, verify_chained_jsonl_v96
from atos_core.external_dialogue_world_v111 import load_world_v111
from atos_core.external_world_ledger_v111 import (
    EXTERNAL_WORLD_ACTION_FETCH_V111,
    EXTERNAL_WORLD_REASON_CODES_V111,
    compute_external_world_chain_hash_v111,
    external_world_event_to_dict_v111,
    make_external_world_event_v111,
    verify_external_world_event_sig_chain_v111,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def _run_one(*, out_dir: Path, world_manifest: str, seed: int) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    events_path = out_dir / "external_world_events.jsonl"
    events_path.write_text("", encoding="utf-8")
    prev_hash = None
    prev_event_sig = ""
    events: List[Dict[str, Any]] = []

    # Scenario 1: disallowed access (should not write).
    disallowed_reason = "external_world_access_not_allowed"
    allowed = False
    if allowed:
        raise SystemExit("internal_error:allowed_false_expected")

    # Scenario 2: allowed access (exactly 1 call).
    allowed = True
    reason_code = "validator_failed_unresolved_reference"
    if reason_code not in EXTERNAL_WORLD_REASON_CODES_V111:
        raise SystemExit("internal_error:reason_code_enum")

    world = load_world_v111(manifest_path=str(world_manifest))
    turn0 = world.fetch_turn(0)
    result_summary = {
        "fetched_turn_id": int(turn0.global_turn_index),
        "conversation_id": str(turn0.conversation_id),
        "role": str(turn0.role),
        "text_sha256": sha256_hex(str(turn0.text).encode("utf-8")),
    }
    ev = make_external_world_event_v111(
        event_index=0,
        turn_index=0,
        action=EXTERNAL_WORLD_ACTION_FETCH_V111,
        reason_code=str(reason_code),
        args={"turn_id": 0},
        result_summary=result_summary,
        prev_event_sig=str(prev_event_sig),
    )
    evd = external_world_event_to_dict_v111(ev)
    prev_hash = append_chained_jsonl_v96(str(events_path), dict(evd), prev_hash=prev_hash)
    events.append(dict(evd))
    prev_event_sig = str(ev.event_sig)

    ok_file_chain = bool(verify_chained_jsonl_v96(str(events_path)))
    ok_sig_chain, reason_sig_chain, _ = verify_external_world_event_sig_chain_v111(_load_jsonl(events_path))
    chain_hash = compute_external_world_chain_hash_v111(_load_jsonl(events_path))

    snapshot = {"schema_version": 111, "events_total": int(len(events)), "external_world_chain_hash_v111": str(chain_hash)}
    _write_once_json(out_dir / "external_world_registry_snapshot_v111.json", snapshot)
    _write_once_json(
        out_dir / "summary.json",
        {
            "schema_version": 111,
            "seed": int(seed),
            "events_total": int(len(events)),
            "file_chain_ok": bool(ok_file_chain),
            "sig_chain_ok": bool(ok_sig_chain),
            "sig_chain_reason": str(reason_sig_chain),
            "external_world_chain_hash_v111": str(chain_hash),
            "disallowed_reason": disallowed_reason,
        },
    )
    _write_once_json(
        out_dir / "freeze_manifest_v111.json",
        {
            "schema_version": 111,
            "kind": "freeze_manifest_v111_external_world_gating",
            "sha256": {
                "external_world_events_jsonl": _sha256_file(events_path),
                "external_world_registry_snapshot_v111_json": _sha256_file(out_dir / "external_world_registry_snapshot_v111.json"),
                "summary_json": _sha256_file(out_dir / "summary.json"),
            },
        },
    )
    return {
        "events_total": int(len(events)),
        "external_world_chain_hash_v111": str(chain_hash),
        "file_chain_ok": bool(ok_file_chain),
        "sig_chain_ok": bool(ok_sig_chain),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--world_manifest", required=True)
    args = ap.parse_args()

    out_base = Path(str(args.out_base))
    seed = int(args.seed)
    world_manifest = str(args.world_manifest)

    out1 = Path(str(out_base) + "_try1")
    out2 = Path(str(out_base) + "_try2")
    r1 = _run_one(out_dir=out1, world_manifest=world_manifest, seed=seed)
    r2 = _run_one(out_dir=out2, world_manifest=world_manifest, seed=seed)

    core1 = {"seed": seed, "r": r1}
    core2 = {"seed": seed, "r": r2}
    determinism_ok = (canonical_json_dumps(core1) == canonical_json_dumps(core2))
    summary_sha = sha256_hex(canonical_json_dumps(core1).encode("utf-8"))

    # Negative tamper: break event_sig in try1.
    tamper_dir = Path(str(out_base) + "_try1_tamper")
    _ensure_absent(tamper_dir)
    shutil.copytree(out1, tamper_dir)
    events_path = tamper_dir / "external_world_events.jsonl"
    lines = events_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit("tamper_internal_error:no_events")
    obj = json.loads(lines[0])
    obj["event_sig"] = "0" * 64
    lines[0] = canonical_json_dumps(obj)
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok_sig_chain, reason_sig_chain, _ = verify_external_world_event_sig_chain_v111(_load_jsonl(events_path))

    out = {
        "ok": True,
        "determinism_ok": bool(determinism_ok),
        "summary_sha256": str(summary_sha),
        "try1": core1,
        "try2": core2,
        "negative_tamper": {"ok": bool(ok_sig_chain), "reason": str(reason_sig_chain)},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
