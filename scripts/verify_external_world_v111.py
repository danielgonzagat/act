#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_u64_le_list(path: str) -> List[int]:
    data = open(path, "rb").read()
    if len(data) % 8 != 0:
        _fail(f"offsets_size_not_multiple_of_8:{path}:{len(data)}")
    out: List[int] = []
    for i in range(0, len(data), 8):
        out.append(int(struct.unpack("<Q", data[i : i + 8])[0]))
    return out


def _join(world_root: str, rel: str) -> str:
    return os.path.normpath(os.path.join(world_root, str(rel)))


def _fetch_line_by_offset(canon_path: str, offset: int) -> str:
    with open(canon_path, "rb") as f:
        f.seek(int(offset))
        b = f.readline()
    try:
        return b.decode("utf-8")
    except Exception:
        return b.decode("utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    manifest_path = str(args.manifest)
    if not os.path.exists(manifest_path):
        _fail(f"missing_manifest:{manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    world_root = os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))

    want_sha = manifest.get("sha256") if isinstance(manifest.get("sha256"), dict) else {}
    paths = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}

    canon_rel = str(paths.get("canonical_jsonl") or "")
    offsets_rel = str(paths.get("offsets_bin") or "")
    conv_index_rel = str(paths.get("conversations_index_json") or "")
    raw_rel = str(paths.get("raw_copy") or "")
    if not (canon_rel and offsets_rel and conv_index_rel and raw_rel):
        _fail("manifest_missing_paths")

    canon_path = _join(world_root, canon_rel)
    offsets_path = _join(world_root, offsets_rel)
    conv_index_path = _join(world_root, conv_index_rel)
    raw_path = _join(world_root, raw_rel)

    for p in [canon_path, offsets_path, conv_index_path, raw_path]:
        if not os.path.exists(p):
            _fail(f"missing_path:{p}")

    got = {
        "raw_copy": _sha256_file(raw_path),
        "canonical_jsonl": _sha256_file(canon_path),
        "offsets_bin": _sha256_file(offsets_path),
        "conversations_index_json": _sha256_file(conv_index_path),
    }

    mismatches: List[Dict[str, Any]] = []
    for k in sorted(got.keys()):
        want = str(want_sha.get(k) or "")
        if want and got[k] != want:
            mismatches.append({"key": k, "want": want, "got": got[k]})
    if mismatches:
        _fail("sha256_mismatch:" + json.dumps(mismatches, ensure_ascii=False, sort_keys=True))

    turns_total = int(counts.get("turns_total") or 0)
    if turns_total <= 0:
        _fail("invalid_turns_total")

    offsets = _read_u64_le_list(offsets_path)
    if len(offsets) != turns_total:
        _fail(f"offsets_len_mismatch:want={turns_total}:got={len(offsets)}")

    # Deterministic sample checks for offsets -> line -> JSON.
    sample_idxs: List[int] = []
    for i in range(min(10, turns_total)):
        sample_idxs.append(i)
    mid = turns_total // 2
    for i in range(max(0, mid - 3), min(turns_total, mid + 3)):
        sample_idxs.append(i)
    for i in range(max(0, turns_total - 10), turns_total):
        sample_idxs.append(i)
    sample_idxs = sorted(set(sample_idxs))

    parsed_fail: List[Dict[str, Any]] = []
    prev_idx = -1
    for idx in sample_idxs:
        if idx <= prev_idx:
            _fail("sample_not_monotonic_internal_error")
        prev_idx = idx
        line = _fetch_line_by_offset(canon_path, offsets[idx]).strip("\n")
        try:
            obj = json.loads(line)
        except Exception as e:
            parsed_fail.append({"idx": idx, "err": str(e)})
            continue
        if int(obj.get("global_turn_index", -1)) != int(idx):
            parsed_fail.append({"idx": idx, "reason": "global_turn_index_mismatch", "got": obj.get("global_turn_index")})
    if parsed_fail:
        _fail("offsets_parse_fail:" + json.dumps(parsed_fail, ensure_ascii=False, sort_keys=True))

    # Verify conversations_index ranges are within bounds and deterministic ordering.
    with open(conv_index_path, "r", encoding="utf-8") as f:
        conv_index = json.load(f)
    convs = conv_index.get("conversations") if isinstance(conv_index.get("conversations"), list) else []
    if not convs:
        _fail("empty_conversations_index")
    last_id = ""
    for c in convs:
        if not isinstance(c, dict):
            _fail("invalid_conversations_index_entry")
        cid = str(c.get("conversation_id") or "")
        if not cid:
            _fail("missing_conversation_id")
        if last_id and cid < last_id:
            _fail("conversations_index_not_sorted")
        last_id = cid
        s_raw = c.get("start_turn", -1)
        e_raw = c.get("end_turn", -1)
        s = int(s_raw) if s_raw is not None else -1
        e = int(e_raw) if e_raw is not None else -1
        if s < 0 or e < s or e >= turns_total:
            _fail(f"conversation_range_invalid:{cid}:{s}:{e}:{turns_total}")

    out = {
        "ok": True,
        "reason": "ok",
        "manifest": manifest_path,
        "world_root": world_root,
        "sha256": got,
        "turns_total": turns_total,
        "conversations_total": int(counts.get("conversations_total") or 0),
        "samples_checked": len(sample_idxs),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
