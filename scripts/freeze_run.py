#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.ledger import Ledger


_SNAP_RE = re.compile(r"^step(\d+)_acts\.jsonl$")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_latest_snapshot(run_dir: str) -> Optional[Tuple[int, str]]:
    snaps = os.path.join(run_dir, "snapshots")
    if not os.path.isdir(snaps):
        return None
    best: Tuple[int, str] = (-1, "")
    for name in os.listdir(snaps):
        m = _SNAP_RE.match(name)
        if not m:
            continue
        try:
            step = int(m.group(1))
        except Exception:
            continue
        path = os.path.join(snaps, name)
        if step > best[0]:
            best = (step, path)
    return best if best[0] >= 0 else None


def _git(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run dir under artifacts/ (contains ledger/snapshots).")
    ap.add_argument("--out", default="", help="Output manifest path (default: <run>/freeze_manifest.json).")
    args = ap.parse_args()

    run_dir = os.path.abspath(str(args.run))
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run_dir_not_found:{run_dir}")

    ledger_path = os.path.join(run_dir, "ledger.jsonl")
    snaps = _find_latest_snapshot(run_dir)
    acts_path = snaps[1] if snaps else os.path.join(run_dir, "acts.jsonl")
    if not os.path.exists(acts_path):
        raise SystemExit(f"acts_not_found:{acts_path}")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git_rev = _git(["git", "-C", repo_root, "rev-parse", "HEAD"])
    git_status = _git(["git", "-C", repo_root, "status", "--porcelain=v1"])
    git_dirty = bool(git_status.strip()) if isinstance(git_status, str) else None

    ledger_ok = None
    if os.path.exists(ledger_path):
        try:
            ledger_ok = bool(Ledger(ledger_path).verify_chain())
        except Exception:
            ledger_ok = False

    manifest_path = str(args.out).strip() or os.path.join(run_dir, "freeze_manifest.json")

    data: Dict[str, Any] = {
        "run_dir": run_dir,
        "acts_path": os.path.relpath(acts_path, run_dir),
        "acts_sha256": _sha256_file(acts_path),
        "snapshot_step": int(snaps[0]) if snaps else None,
        "ledger_present": bool(os.path.exists(ledger_path)),
        "ledger_chain_ok": ledger_ok,
        "ledger_sha256": _sha256_file(ledger_path) if os.path.exists(ledger_path) else None,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "git": {
            "repo_root": repo_root,
            "rev": git_rev,
            "dirty": git_dirty,
        },
    }

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))
        f.write("\n")
    print(manifest_path)


if __name__ == "__main__":
    main()

