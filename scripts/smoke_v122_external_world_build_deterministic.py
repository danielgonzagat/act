#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex


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


def _run_builder(*, conv: str, rtf: str, out_dir: Path) -> Dict[str, Any]:
    if out_dir.exists():
        mp = out_dir / "manifest_v122.json"
        if not mp.exists():
            raise SystemExit("existing_out_dir_missing_manifest_v122:" + str(out_dir))
        return _load_json(mp)
    cmd = [
        sys.executable,
        "scripts/build_external_world_v122.py",
        "--conversations_input",
        str(conv),
        "--rtf_input",
        str(rtf),
        "--out",
        str(out_dir),
    ]
    env = dict(os.environ)
    p = subprocess.run(cmd, env=env, cwd=str(Path(__file__).resolve().parent.parent), capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("builder_failed:\nSTDOUT:\n{out}\nSTDERR:\n{err}".format(out=p.stdout, err=p.stderr))
    return _load_json(out_dir / "manifest_v122.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations_input", required=True)
    ap.add_argument("--rtf_input", required=True)
    ap.add_argument("--out1", required=True)
    ap.add_argument("--out2", required=True)
    args = ap.parse_args()

    out1 = Path(str(args.out1)).resolve()
    out2 = Path(str(args.out2)).resolve()

    if out2.exists():
        raise SystemExit(f"worm_exists:{out2}")

    m1 = _run_builder(conv=str(args.conversations_input), rtf=str(args.rtf_input), out_dir=out1)
    m2 = _run_builder(conv=str(args.conversations_input), rtf=str(args.rtf_input), out_dir=out2)

    # Determinism check: compare sha256 entries + manifest_sig.
    sha1 = dict(m1.get("sha256") or {}) if isinstance(m1.get("sha256"), dict) else {}
    sha2 = dict(m2.get("sha256") or {}) if isinstance(m2.get("sha256"), dict) else {}
    if canonical_json_dumps(sha1) != canonical_json_dumps(sha2):
        raise SystemExit("determinism_failed:sha256_map")
    c1 = dict(m1.get("counts") or {}) if isinstance(m1.get("counts"), dict) else {}
    c2 = dict(m2.get("counts") or {}) if isinstance(m2.get("counts"), dict) else {}
    if canonical_json_dumps(c1) != canonical_json_dumps(c2):
        raise SystemExit("determinism_failed:counts")

    # Also compare file sha256 for the three output files.
    dh1 = out1 / str((m1.get("paths") or {}).get("dialogue_history_canonical_jsonl") or "")
    dh2 = out2 / str((m2.get("paths") or {}).get("dialogue_history_canonical_jsonl") or "")
    if _sha256_file(dh1) != _sha256_file(dh2):
        raise SystemExit("determinism_failed:dialogue_jsonl_sha")

    core = {
        "schema_version": 122,
        "determinism_ok": True,
        "content_sig": sha256_hex(canonical_json_dumps({"sha256": sha1, "counts": c1}).encode("utf-8")),
        "sha256": dict(sha1),
        "out1": str(out1),
        "out2": str(out2),
    }
    summary_sha256 = sha256_hex(canonical_json_dumps(core).encode("utf-8"))
    print(json.dumps({"ok": True, "summary_sha256": summary_sha256, "core": core}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
