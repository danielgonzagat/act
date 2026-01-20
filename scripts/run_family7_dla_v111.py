#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.conversation_loop_v110 import run_conversation_v110
from atos_core.external_world_ledger_v111 import compute_external_world_chain_hash_v111, verify_external_world_event_sig_chain_v111
from atos_core.fluency_contract_v111 import fluency_contract_v111


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


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
        _fail(f"worm_exists:{path}")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_conversation_run(run_dir: Path) -> Dict[str, Any]:
    paths = {
        "transcript": run_dir / "transcript.jsonl",
        "verify_chain_v110": run_dir / "verify_chain_v110.json",
    }
    transcript_rows = _load_jsonl(paths["transcript"])
    transcript_view: List[Dict[str, Any]] = []
    for r in transcript_rows:
        if not isinstance(r, dict):
            continue
        payload = r.get("payload")
        if not isinstance(payload, dict):
            continue
        transcript_view.append({"role": str(payload.get("role") or ""), "text": str(payload.get("text") or "")})
    return {
        "transcript": list(transcript_view),
        "verify_chain_v110": json.loads(paths["verify_chain_v110"].read_text(encoding="utf-8"))
        if paths["verify_chain_v110"].exists()
        else {},
    }


def _write_once_json(path: Path, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        _fail(f"tmp_exists:{tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _write_once_text(path: Path, text: str) -> None:
    _ensure_absent(path)
    path.write_text(str(text), encoding="utf-8")


def _compute_freeze_manifest_v111(*, task_dir: Path, sha256_paths: Dict[str, str]) -> Dict[str, Any]:
    sha256: Dict[str, str] = {}
    rel_paths: Dict[str, str] = {}
    for k, p in sorted(sha256_paths.items(), key=lambda kv: str(kv[0])):
        fp = Path(p)
        try:
            rel_paths[str(k)] = str(fp.relative_to(task_dir))
        except Exception:
            rel_paths[str(k)] = str(fp.name)
        if fp.exists():
            sha256[str(k)] = _sha256_file(fp)
    manifest = {
        "schema_version": 111,
        "kind": "freeze_manifest_v111",
        "sha256": sha256,
        "sha256_paths": dict(rel_paths),
    }
    # ledger_hash is sha256 of this manifest file bytes (computed by caller after write).
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--max_tasks", type=int, default=10)
    args = ap.parse_args()

    seed = int(args.seed)
    tasks_path = Path(str(args.tasks))
    out_dir = Path(str(args.out))
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    tasks: List[Dict[str, Any]] = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    if not tasks:
        _fail("empty_tasks")

    max_tasks = min(int(args.max_tasks), len(tasks))
    tasks = tasks[:max_tasks]

    results: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or f"task_{i}")
        user_turns = task.get("user_turns") if isinstance(task.get("user_turns"), list) else []
        user_turn_texts = [str(x) for x in user_turns if isinstance(x, str)]
        task_subdir = out_dir / f"task_{i:03d}"
        run_conversation_v110(user_turn_texts=user_turn_texts, out_dir=str(task_subdir), seed=int(seed))

        # External world ledger (must exist, even if empty).
        ext_events_path = task_subdir / "external_world_events.jsonl"
        if ext_events_path.exists():
            _fail(f"worm_exists:{ext_events_path}")
        ext_events_path.write_text("", encoding="utf-8")
        ext_snapshot_path = task_subdir / "external_world_registry_snapshot_v111.json"
        _write_once_json(ext_snapshot_path, {"schema_version": 111, "kind": "external_world_registry_snapshot_v111", "events_total": 0})

        # Load v110 verify output (conversation loop writes this deterministically).
        run = _load_conversation_run(task_subdir)
        verify_v110 = run.get("verify_chain_v110") if isinstance(run.get("verify_chain_v110"), dict) else {}
        ok_chain = bool(verify_v110.get("ok"))
        reason_chain = "ok" if ok_chain else "verify_chain_v110_failed"
        details_chain = dict(verify_v110)

        # Fluency contract (V111) on transcript.
        ok_fc, reason_fc, details_fc = fluency_contract_v111(transcript=run["transcript"])
        fluency_path = task_subdir / "fluency_contract_v111.json"
        _write_once_json(fluency_path, {"ok": bool(ok_fc), "reason": str(reason_fc), "details": dict(details_fc)})

        # External world ledger is empty -> chain hash is stable (empty list).
        ext_events = _load_jsonl(ext_events_path)
        ok_ext_sig, reason_ext_sig, _ = verify_external_world_event_sig_chain_v111(ext_events)
        ext_chain_hash = compute_external_world_chain_hash_v111(ext_events)

        # Freeze manifest V111 (task-local).
        freeze_path = task_subdir / "freeze_manifest_v111.json"
        sha256_paths = {
            "v110_summary_json": str(task_subdir / "summary.json"),
            "v110_freeze_manifest_v110_json": str(task_subdir / "freeze_manifest_v110.json"),
            "task_eval_json": str(task_subdir / "eval.json"),
            "fluency_contract_v111_json": str(fluency_path),
            "external_world_events_jsonl": str(ext_events_path),
            "external_world_registry_snapshot_v111_json": str(ext_snapshot_path),
        }
        freeze = _compute_freeze_manifest_v111(task_dir=task_subdir, sha256_paths=sha256_paths)
        _write_once_json(freeze_path, freeze)
        ledger_hash = _sha256_file(freeze_path)

        # Per-task eval.
        eval_obj = {
            "schema_version": 111,
            "task_id": task_id,
            "ok_conversation_chain_v110": bool(ok_chain),
            "reason_conversation_chain_v110": str(reason_chain),
            "ok_fluency_contract_v111": bool(ok_fc),
            "reason_fluency_contract_v111": str(reason_fc),
            "external_world_events_total": int(len(ext_events)),
            "external_world_chain_hash_v111": str(ext_chain_hash),
            "external_world_sig_chain_ok": bool(ok_ext_sig),
            "external_world_sig_chain_reason": str(reason_ext_sig),
            "ledger_hash": str(ledger_hash),
        }
        eval_path = task_subdir / "eval.json"
        _write_once_json(eval_path, eval_obj)

        results.append(
            {
                "task_index": int(i),
                "task_id": task_id,
                "run_dir": f"task_{i:03d}",
                "ok": bool(ok_chain and ok_fc),
                "ledger_hash": str(ledger_hash),
                "external_world_chain_hash_v111": str(ext_chain_hash),
            }
        )

    # Aggregate eval.
    agg = {
        "schema_version": 111,
        "seed": int(seed),
        "tasks_total": int(len(results)),
        "tasks_ok": int(sum(1 for r in results if bool(r.get("ok")))),
        "results": list(results),
        "aggregate_sig": sha256_hex(canonical_json_dumps({"seed": int(seed), "results": results}).encode("utf-8")),
    }
    _write_once_json(out_dir / "eval.json", agg)

    # Minimal summary for determinism in smoke.
    summary = {
        "schema_version": 111,
        "seed": int(seed),
        "tasks_total": int(len(results)),
        "tasks_ok": int(sum(1 for r in results if bool(r.get("ok")))),
        "eval_sha256": _sha256_file(out_dir / "eval.json"),
    }
    _write_once_json(out_dir / "summary.json", summary)

    print(json.dumps({"ok": True, "out_dir": str(out_dir), "summary": summary}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
