#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import struct
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# Ensure repo root is on sys.path (scripts/ is sys.path[0] when invoked directly).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Reuse project canonicalization/hashing utilities for determinism.
from atos_core.act import canonical_json_dumps, sha256_hex


CHUNK_CHARS = 1024 * 1024


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


def _iso_utc_from_epoch_seconds(sec: int) -> str:
    # Deterministic; no wall-clock.
    dt = _dt.datetime.fromtimestamp(int(sec), tz=_dt.timezone.utc)
    # Drop microseconds to avoid float-rounding drift from the export.
    dt = dt.replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def _iter_json_array(path: str) -> Iterator[Any]:
    """
    Streaming JSON array parser using stdlib json.JSONDecoder.raw_decode.
    Handles very large arrays without loading the whole file.
    """
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        # Seek to start of array.
        while True:
            ch = f.read(1)
            if ch == "":
                _fail("json_array_missing_open_bracket")
            if ch.isspace():
                continue
            if ch != "[":
                _fail(f"json_array_expected_open_bracket_got:{repr(ch)}")
            break

        buf = ""
        while True:
            if not buf:
                chunk = f.read(CHUNK_CHARS)
                if chunk == "":
                    _fail("json_array_unexpected_eof")
                buf += chunk

            # Skip whitespace and commas.
            i = 0
            while i < len(buf) and buf[i].isspace():
                i += 1
            if i:
                buf = buf[i:]
            if buf.startswith(","):
                buf = buf[1:]
                continue
            if buf.startswith("]"):
                return

            try:
                obj, idx = dec.raw_decode(buf)
            except json.JSONDecodeError:
                chunk = f.read(CHUNK_CHARS)
                if chunk == "":
                    _fail("json_array_decode_error_eof")
                buf += chunk
                continue

            yield obj
            buf = buf[idx:]


def _extract_text_from_message_content(content: Any) -> str:
    if not isinstance(content, dict):
        return ""
    ctype = str(content.get("content_type") or "")
    if ctype == "text":
        parts = content.get("parts")
        if isinstance(parts, list):
            out_parts: List[str] = []
            for p in parts:
                if isinstance(p, str):
                    out_parts.append(p)
                else:
                    # Preserve non-string parts deterministically.
                    out_parts.append(canonical_json_dumps(p))
            return "\n".join(out_parts)
        # Fallback: keep full content if parts missing.
        return canonical_json_dumps(content)
    # Non-text content: serialize deterministically.
    return canonical_json_dumps(content)


def _canon_role(role: Any) -> str:
    r = str(role or "")
    if r in ("user", "assistant", "system", "tool"):
        return r
    return "unknown"


def _canon_children(children: Any) -> List[str]:
    if not isinstance(children, list):
        return []
    out: List[str] = []
    for c in children:
        if isinstance(c, str) and c:
            out.append(str(c))
    return sorted(set(out))


def _safe_int(x: Any) -> Optional[int]:
    try:
        # Export uses float seconds; cast via int for determinism.
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None


def _extract_messages_from_conversation(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns deterministic list of message records for one conversation.
    Each record carries enough info to build the canonical turn line.
    """
    mapping = conv.get("mapping")
    if not isinstance(mapping, dict):
        return []

    out: List[Dict[str, Any]] = []
    seen_ids: set = set()
    seen_no_id: set = set()
    for mid, node in mapping.items():
        if not isinstance(mid, str) or not mid:
            continue
        if not isinstance(node, dict):
            continue
        msg = node.get("message")
        if not isinstance(msg, dict):
            continue
        author = msg.get("author")
        author = author if isinstance(author, dict) else {}
        role = _canon_role(author.get("role"))

        content = msg.get("content")
        text = _extract_text_from_message_content(content)

        create_time = _safe_int(msg.get("create_time"))
        ts = _iso_utc_from_epoch_seconds(int(create_time)) if create_time is not None else None

        parent_id = node.get("parent") if isinstance(node.get("parent"), str) else None
        children_ids = _canon_children(node.get("children"))

        # Dedup within conversation only (deterministic + bounded memory).
        if mid in seen_ids:
            continue
        if mid:
            seen_ids.add(mid)
        else:
            key = (ts or "", role, sha256_hex(text.encode("utf-8")))
            if key in seen_no_id:
                continue
            seen_no_id.add(key)

        out.append(
            {
                "message_id": str(mid),
                "parent_message_id": str(parent_id or ""),
                "children_message_ids": list(children_ids),
                "role": str(role),
                "text": str(text),
                "timestamp": ts,
                "create_time": create_time if create_time is not None else -1,
            }
        )

    out.sort(key=lambda d: (int(d.get("create_time", -1)), str(d.get("message_id") or "")))
    return out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_once_bytes(path: str, data: bytes) -> None:
    # WORM: write-once.
    with open(path, "xb") as f:
        f.write(data)


def _write_once_text(path: str, text: str) -> None:
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


def _write_or_verify_file(path: str, *, write_fn, expected_sha256: str) -> None:
    if os.path.exists(path):
        got = _sha256_file(path)
        if got != expected_sha256:
            _fail(f"worm_path_exists_hash_mismatch:{path}:want={expected_sha256}:got={got}")
        return
    write_fn()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    input_path = str(args.input)
    out_root = str(args.out)
    if not os.path.exists(input_path):
        _fail(f"missing_input:{input_path}")

    raw_dir = os.path.join(out_root, "dialogue_history_raw")
    canon_dir = os.path.join(out_root, "dialogue_history_canonical")
    manifest_dir = os.path.join(out_root, "manifests")
    _ensure_dir(raw_dir)
    _ensure_dir(canon_dir)
    _ensure_dir(manifest_dir)

    input_sha = _sha256_file(input_path)
    raw_copy_name = f"conversations_{input_sha[:16]}.json"
    raw_copy_path = os.path.join(raw_dir, raw_copy_name)

    # Copy raw (idempotent by content-addressed name).
    if not os.path.exists(raw_copy_path):
        tmp = raw_copy_path + ".tmp"
        if os.path.exists(tmp):
            _fail(f"tmp_exists:{tmp}")
        shutil.copyfile(input_path, tmp)
        os.replace(tmp, raw_copy_path)
    else:
        if _sha256_file(raw_copy_path) != input_sha:
            _fail(f"raw_copy_sha_mismatch:{raw_copy_path}")

    canon_jsonl_path = os.path.join(canon_dir, "dialogue_history_canonical_v111.jsonl")
    offsets_path = os.path.join(canon_dir, "offsets_v111.bin")
    conv_index_path = os.path.join(canon_dir, "conversations_index_v111.json")
    manifest_path = os.path.join(manifest_dir, "world_manifest_v111.json")

    # If manifest already exists, verify it and exit idempotently.
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            want = str(m.get("sha256", {}).get("canonical_jsonl") or "")
            if want and os.path.exists(canon_jsonl_path):
                got = _sha256_file(canon_jsonl_path)
                if got != want:
                    _fail(f"manifest_mismatch:canonical_jsonl_sha:want={want}:got={got}")
            print(json.dumps({"ok": True, "reason": "already_built", "manifest": manifest_path}, indent=2, sort_keys=True))
            return
        except Exception as e:
            _fail(f"manifest_read_error:{e}")

    # Build canonical jsonl + offsets + conversation index (write-once).
    if any(os.path.exists(p) for p in [canon_jsonl_path, offsets_path, conv_index_path]):
        _fail("canonical_paths_exist_without_manifest (WORM)")

    conv_ranges: Dict[str, Dict[str, Any]] = {}
    global_turn_index = 0

    # Write canonical and offsets in lockstep.
    with open(canon_jsonl_path, "xb") as canon_f, open(offsets_path, "xb") as off_f:
        for conv in _iter_json_array(input_path):
            if not isinstance(conv, dict):
                continue
            conv_id = str(conv.get("id") or "")
            if not conv_id:
                continue
            title = conv.get("title")
            title_s = str(title) if isinstance(title, str) and title else None

            msgs = _extract_messages_from_conversation(conv)
            if not msgs:
                continue

            start = global_turn_index
            turn_in_conv = 0

            for msg in msgs:
                offset = canon_f.tell()
                off_f.write(struct.pack("<Q", int(offset)))

                line_obj = {
                    "global_turn_index": int(global_turn_index),
                    "conversation_id": str(conv_id),
                    "conversation_title": title_s,
                    "turn_in_conversation": int(turn_in_conv),
                    "timestamp": msg.get("timestamp"),
                    "role": str(msg.get("role") or "unknown"),
                    "text": str(msg.get("text") or ""),
                    "message_id": str(msg.get("message_id") or ""),
                    "parent_message_id": str(msg.get("parent_message_id") or ""),
                    "children_message_ids": list(msg.get("children_message_ids") or []),
                    "source": "chatgpt_export_conversations.json",
                }

                b = (canonical_json_dumps(line_obj) + "\n").encode("utf-8")
                canon_f.write(b)

                global_turn_index += 1
                turn_in_conv += 1

            end = global_turn_index - 1
            conv_ranges[str(conv_id)] = {
                "conversation_id": str(conv_id),
                "title": title_s,
                "start_turn": int(start),
                "end_turn": int(end),
                "turns_total": int(end - start + 1),
            }

    # Write conversations index (deterministic ordering).
    conv_index = {"schema_version": 111, "conversations": [conv_ranges[k] for k in sorted(conv_ranges.keys())]}
    _write_once_text(conv_index_path, json.dumps(conv_index, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    # Compute hashes and write manifest (WORM).
    sha = {
        "raw_conversations_json": input_sha,
        "raw_copy": _sha256_file(raw_copy_path),
        "canonical_jsonl": _sha256_file(canon_jsonl_path),
        "offsets_bin": _sha256_file(offsets_path),
        "conversations_index_json": _sha256_file(conv_index_path),
    }
    if sha["raw_copy"] != sha["raw_conversations_json"]:
        _fail("raw_copy_sha_mismatch_postcopy")

    manifest = {
        "schema_version": 111,
        "kind": "external_dialogue_world_v111",
        "source": {"input_path": input_path, "input_sha256": input_sha},
        "paths": {
            "raw_copy": os.path.relpath(raw_copy_path, out_root).replace(os.sep, "/"),
            "canonical_jsonl": os.path.relpath(canon_jsonl_path, out_root).replace(os.sep, "/"),
            "offsets_bin": os.path.relpath(offsets_path, out_root).replace(os.sep, "/"),
            "conversations_index_json": os.path.relpath(conv_index_path, out_root).replace(os.sep, "/"),
        },
        "sha256": dict(sha),
        "counts": {"turns_total": int(global_turn_index), "conversations_total": int(len(conv_ranges))},
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_once_text(manifest_path, manifest_text)

    print(
        json.dumps(
            {
                "ok": True,
                "manifest": manifest_path,
                "sha256": manifest["sha256"],
                "counts": manifest["counts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
