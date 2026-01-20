#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from typing import Any, Dict, List, Optional, Tuple


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _parse_arg_from_command(cmd: str, flag: str) -> Optional[str]:
    try:
        toks = shlex.split(cmd)
    except Exception:
        toks = cmd.split()
    for i, t in enumerate(toks):
        if t == flag and i + 1 < len(toks):
            return str(toks[i + 1])
    return None


def _maybe_join(base: str, p: str) -> str:
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base, p))


def _derive_sha256_paths(freeze: Dict[str, Any]) -> Dict[str, str]:
    """
    Project-style derived paths for common freeze keys.
    """
    out_dir = str(freeze.get("out_dir") or "")
    acts_source_run = str(freeze.get("acts_source_run") or "")
    sha = freeze.get("sha256")
    sha = sha if isinstance(sha, dict) else {}

    # Base (acts) is conventionally stored as acts_source_run/acts.jsonl.
    paths: Dict[str, str] = {}
    if "base_acts_jsonl" in sha:
        paths["base_acts_jsonl"] = os.path.join(acts_source_run, "acts.jsonl")

    # Common run-local artifacts.
    for k in list(sha.keys()):
        if k == "goal_shadow_jsonl":
            paths[k] = os.path.join(out_dir, "traces", "goal_shadow.jsonl")
        elif k == "goal_shadow_trace_jsonl":
            paths[k] = os.path.join(out_dir, "traces", "goal_shadow_trace.jsonl")
        elif k == "csv_exec_jsonl":
            paths[k] = os.path.join(out_dir, "traces", "csv_exec.jsonl")
        elif k == "mined_candidates_json":
            paths[k] = os.path.join(out_dir, "mined_candidates.json")
        elif k == "acts_promoted_jsonl":
            paths[k] = os.path.join(out_dir, "promotion", "acts_promoted.jsonl")
        elif k == "promotion_ledger_jsonl":
            paths[k] = os.path.join(out_dir, "promotion", "promotion_ledger.jsonl")
        elif k == "promotion_manifest_json":
            paths[k] = os.path.join(out_dir, "promotion", "promotion_manifest.json")
        elif k == "summary_csv":
            paths[k] = os.path.join(out_dir, "summary.csv")
        elif k == "summary_json":
            paths[k] = os.path.join(out_dir, "summary.json")

    # patch_diff is typically passed via CLI.
    cmd0 = ""
    cmds = freeze.get("commands")
    if isinstance(cmds, list) and cmds:
        cmd0 = str(cmds[0])
    patch_diff = _parse_arg_from_command(cmd0, "--patch_diff") or ""
    if "patch_diff" in sha and patch_diff:
        paths["patch_diff"] = str(patch_diff)

    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True)
    args = ap.parse_args()

    freeze_path = str(args.freeze)
    if not os.path.exists(freeze_path):
        _fail(f"missing_freeze:{freeze_path}")

    with open(freeze_path, "r", encoding="utf-8") as f:
        freeze = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(freeze_path)) or "."

    sha = freeze.get("sha256")
    sha = sha if isinstance(sha, dict) else {}

    derived = _derive_sha256_paths(freeze)
    missing_paths: List[Dict[str, Any]] = []
    hash_mismatches: List[Dict[str, Any]] = []
    invariant_failures: List[Dict[str, Any]] = []
    warnings: List[str] = []

    # (i) Validate referenced directories exist.
    out_dir = str(freeze.get("out_dir") or "")
    acts_source_run = str(freeze.get("acts_source_run") or "")
    if out_dir:
        out_abs = _maybe_join(base_dir, out_dir)
        if not os.path.exists(out_abs):
            missing_paths.append({"key": "out_dir", "path": out_dir})
    if acts_source_run:
        acts_abs = _maybe_join(base_dir, acts_source_run)
        if not os.path.exists(acts_abs):
            missing_paths.append({"key": "acts_source_run", "path": acts_source_run})

    # (ii) Validate derived sha256 entries.
    for key, want_hash in sorted(sha.items(), key=lambda kv: str(kv[0])):
        want = str(want_hash or "")
        if not want:
            continue
        p = derived.get(str(key), "")
        if not p:
            warnings.append(f"no_path_mapping_for_sha_key:{key}")
            continue
        p_abs = _maybe_join(base_dir, str(p))
        if not os.path.exists(p_abs):
            missing_paths.append({"key": str(key), "path": str(p)})
            continue
        got = _sha256_file(p_abs)
        if got != want:
            hash_mismatches.append({"key": str(key), "path": str(p), "want": want, "got": got})

    # (iii) Invariants.
    if "verify_chain" in freeze and not bool(freeze.get("verify_chain", False)):
        invariant_failures.append({"key": "verify_chain", "want": True, "got": bool(freeze.get("verify_chain"))})

    summary = freeze.get("summary")
    if isinstance(summary, dict):
        if "promotion_chain_ok" in summary and not bool(summary.get("promotion_chain_ok", False)):
            invariant_failures.append({"key": "summary.promotion_chain_ok", "want": True, "got": bool(summary.get("promotion_chain_ok"))})
        if "mismatch_goals" in summary and int(summary.get("mismatch_goals", 0) or 0) != 0:
            invariant_failures.append({"key": "summary.mismatch_goals", "want": 0, "got": int(summary.get("mismatch_goals", 0) or 0)})
        if "uncertainty_ic_count" in summary and int(summary.get("uncertainty_ic_count", 0) or 0) != 0:
            invariant_failures.append({"key": "summary.uncertainty_ic_count", "want": 0, "got": int(summary.get("uncertainty_ic_count", 0) or 0)})

    ok = (not missing_paths) and (not hash_mismatches) and (not invariant_failures)
    out = {
        "ok": bool(ok),
        "freeze_path": str(freeze_path),
        "missing_paths": list(missing_paths),
        "hash_mismatches": list(hash_mismatches),
        "invariant_failures": list(invariant_failures),
        "warnings": list(warnings),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

