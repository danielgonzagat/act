#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Ensure repo root is on sys.path (scripts/ is sys.path[0] when invoked directly).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
    dt = _dt.datetime.fromtimestamp(int(sec), tz=_dt.timezone.utc)
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


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        # Export uses float seconds; cast via int for determinism.
        return int(float(x))
    except Exception:
        return None


def _canon_role(role: Any) -> str:
    r = str(role or "")
    if r in ("user", "assistant", "system", "tool"):
        return r
    return "unknown"


def _normalize_text(text: str) -> str:
    # Minimal and deterministic: normalize line endings only.
    t = str(text or "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return t


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
                    out_parts.append(canonical_json_dumps(p))
            return "\n".join(out_parts)
        return canonical_json_dumps(content)
    return canonical_json_dumps(content)


def _resolve_input_path(user_supplied: str) -> Tuple[str, List[str]]:
    tried: List[str] = []

    def _try(p: str) -> Optional[str]:
        pp = os.path.expanduser(str(p))
        tried.append(pp)
        if os.path.exists(pp):
            return pp
        return None

    if user_supplied:
        got = _try(user_supplied)
        if got:
            return got, tried

    # Known default locations (accented/unaccented).
    defaults = [
        "/Users/danielpenin/Desktop/HISTÓRICO/conversations.json",
        "/Users/danielpenin/Desktop/HISTORICO/conversations.json",
        os.path.expanduser("~/Desktop/HISTÓRICO/conversations.json"),
        os.path.expanduser("~/Desktop/HISTORICO/conversations.json"),
    ]
    for p in defaults:
        got = _try(p)
        if got:
            return got, tried

    # Deterministic fallback search (sorted).
    desktop = os.path.expanduser("~/Desktop")
    tried.append(desktop + "/* (search)")
    candidates: List[str] = []
    if os.path.isdir(desktop):
        for root, dirs, files in os.walk(desktop):
            dirs.sort()
            files.sort()
            if "conversations.json" in files:
                candidates.append(os.path.join(root, "conversations.json"))
    candidates = sorted(set(candidates))
    for p in candidates:
        got = _try(p)
        if got:
            return got, tried

    _fail("missing_input_conversations_json_tried:" + json.dumps(tried, ensure_ascii=False, sort_keys=True))
    raise AssertionError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="", help="Path to conversations.json (ChatGPT export).")
    ap.add_argument("--out", required=True, help="External world output root (e.g., external_world).")
    args = ap.parse_args()

    input_path, tried = _resolve_input_path(str(args.input or ""))
    out_root = str(args.out)
    os.makedirs(out_root, exist_ok=True)

    input_sha = _sha256_file(input_path)

    raw_dir = os.path.join(out_root, "dialogue_history_raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_copy_name = "conversations_v113_{h}.json".format(h=input_sha[:16])
    raw_copy_path = os.path.join(raw_dir, raw_copy_name)
    if not os.path.exists(raw_copy_path):
        tmp = raw_copy_path + ".tmp"
        if os.path.exists(tmp):
            _fail(f"tmp_exists:{tmp}")
        shutil.copyfile(input_path, tmp)
        os.replace(tmp, raw_copy_path)
    else:
        if _sha256_file(raw_copy_path) != input_sha:
            _fail(f"raw_copy_sha_mismatch:{raw_copy_path}")

    canon_path = os.path.join(out_root, "dialogue_history_canonical_v113.jsonl")
    manifest_path = os.path.join(out_root, "dialogue_history_canonical_v113_manifest.json")
    if os.path.exists(canon_path) or os.path.exists(manifest_path):
        _fail("worm_exists_output_paths")

    # Collect all message records for global temporal ordering.
    messages: List[Dict[str, Any]] = []
    conversation_ids: set = set()
    missing_time = 0
    unknown_role = 0

    # Dedup guard: (conversation_id, message_id) must not conflict.
    seen_mid: Dict[Tuple[str, str], Tuple[int, str]] = {}
    dup_dropped = 0

    for conv in _iter_json_array(input_path):
        if not isinstance(conv, dict):
            continue
        conv_id = str(conv.get("id") or "")
        if not conv_id:
            continue
        conversation_ids.add(conv_id)
        mapping = conv.get("mapping")
        if not isinstance(mapping, dict):
            continue
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
            if role == "unknown":
                unknown_role += 1
            content = msg.get("content")
            text = _normalize_text(_extract_text_from_message_content(content))

            create_time = _safe_int(msg.get("create_time"))
            if create_time is None:
                missing_time += 1
            ct_sort = int(create_time) if create_time is not None else -1
            ts = _iso_utc_from_epoch_seconds(ct_sort) if create_time is not None else ""

            key = (conv_id, mid)
            text_hash = sha256_hex(text.encode("utf-8"))
            if key in seen_mid:
                prev_ct, prev_th = seen_mid[key]
                if prev_ct == ct_sort and prev_th == text_hash:
                    dup_dropped += 1
                    continue
                _fail("duplicate_message_id_conflict:" + canonical_json_dumps({"conversation_id": conv_id, "message_id": mid}))
            seen_mid[key] = (ct_sort, text_hash)

            messages.append(
                {
                    "create_time": int(ct_sort),
                    "conversation_id": str(conv_id),
                    "message_id": str(mid),
                    "timestamp": str(ts),
                    "role": str(role),
                    "text": str(text),
                }
            )

    messages.sort(key=lambda d: (int(d.get("create_time") or -1), str(d.get("conversation_id") or ""), str(d.get("message_id") or "")))

    # Write canonical JSONL + compute offsets in-memory for internal checks only.
    turns_total = 0
    with open(canon_path, "xb") as f:
        for idx, m in enumerate(messages):
            line_obj = {
                "global_turn_index": int(idx),
                "conversation_id": str(m.get("conversation_id") or ""),
                "message_id": str(m.get("message_id") or ""),
                "timestamp": str(m.get("timestamp") or ""),
                "role": str(m.get("role") or "unknown"),
                "text": str(m.get("text") or ""),
                "source": "chatgpt_export_v113",
            }
            f.write((canonical_json_dumps(line_obj) + "\n").encode("utf-8"))
            turns_total += 1

    out_sha = _sha256_file(canon_path)
    raw_copy_sha = _sha256_file(raw_copy_path)
    if raw_copy_sha != input_sha:
        _fail("raw_copy_sha_mismatch_postcopy")

    manifest = {
        "schema_version": 113,
        "kind": "chatgpt_export_canonical_v113",
        "source": {
            "input_path": str(input_path),
            "input_sha256": str(input_sha),
            "paths_tried": list(tried),
        },
        "paths": {
            "raw_copy": os.path.relpath(raw_copy_path, out_root).replace(os.sep, "/"),
            "canonical_jsonl": os.path.relpath(canon_path, out_root).replace(os.sep, "/"),
        },
        "sha256": {"input": str(input_sha), "raw_copy": str(raw_copy_sha), "canonical_jsonl": str(out_sha)},
        "counts": {
            "turns_total": int(turns_total),
            "conversations_total": int(len(conversation_ids)),
            "missing_timestamp_total": int(missing_time),
            "unknown_role_total": int(unknown_role),
            "duplicates_dropped_total": int(dup_dropped),
        },
    }
    with open(manifest_path, "x", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "ok": True,
                "reason": "built",
                "out": {"canonical_jsonl": canon_path, "manifest": manifest_path},
                "sha256": dict(manifest["sha256"]),
                "counts": dict(manifest["counts"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

