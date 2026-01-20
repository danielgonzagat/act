#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Ensure repo root is on sys.path (scripts/ is sys.path[0] when invoked directly).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex


CHUNK_CHARS = 1024 * 1024

# Default chunking for the engineering doc (plain text chars).
DOC_CHUNK_CHARS = 4000


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _ensure_absent(path: Path) -> None:
    if path.exists():
        _fail(f"worm_exists:{path}")


def _sha256_file(path: Path) -> str:
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


def _iter_json_array(path: Path) -> Iterator[Any]:
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


def _safe_int_epoch(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
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


def _find_desktop_inputs() -> Tuple[Optional[Path], Optional[Path], List[str]]:
    """
    Deterministic discovery of:
      - Conversations.json / conversations.json (any case) under ~/Desktop (maxdepth 3)
      - Projeto ACT.rtf (maxdepth 2)
    Returns: (conversations_path, rtf_path, attempts)
    """
    home = Path(os.path.expanduser("~"))
    desktop = home / "Desktop"
    attempts: List[str] = []

    conv_candidates: List[Path] = []
    if desktop.is_dir():
        # Deterministic os.walk: sorted dirs/files.
        for root, dirs, files in os.walk(str(desktop)):
            rel_depth = Path(root).relative_to(desktop).parts
            if len(rel_depth) > 3:
                # prune deeper
                dirs[:] = []
                continue
            dirs.sort()
            files.sort()
            for fn in files:
                if fn.lower() == "conversations.json":
                    conv_candidates.append(Path(root) / fn)
    conv_candidates = sorted(set([p.resolve() for p in conv_candidates]), key=lambda p: str(p))
    if conv_candidates:
        attempts.append("found_conversations_candidates=" + json.dumps([str(p) for p in conv_candidates], ensure_ascii=False, sort_keys=True))
    conv_path = conv_candidates[0] if conv_candidates else None

    rtf_candidates: List[Path] = []
    if desktop.is_dir():
        for root, dirs, files in os.walk(str(desktop)):
            rel_depth = Path(root).relative_to(desktop).parts
            if len(rel_depth) > 2:
                dirs[:] = []
                continue
            dirs.sort()
            files.sort()
            for fn in files:
                if fn.lower() == "projeto act.rtf":
                    rtf_candidates.append(Path(root) / fn)
    rtf_candidates = sorted(set([p.resolve() for p in rtf_candidates]), key=lambda p: str(p))
    if rtf_candidates:
        attempts.append("found_rtf_candidates=" + json.dumps([str(p) for p in rtf_candidates], ensure_ascii=False, sort_keys=True))
    rtf_path = rtf_candidates[0] if rtf_candidates else None

    return conv_path, rtf_path, attempts


def _rtf_to_text_v122(data: bytes) -> str:
    """
    Minimal deterministic RTF -> plain text converter (stdlib only).
    Focus: preserve textual content; ignore formatting.
    """
    b = data
    n = len(b)
    i = 0
    out_chars: List[str] = []
    # group stack: skip flag
    skip_stack: List[bool] = [False]

    def _in_skip() -> bool:
        return bool(skip_stack[-1]) if skip_stack else False

    def _push(skip: bool) -> None:
        skip_stack.append(bool(skip))

    def _pop() -> None:
        if len(skip_stack) > 1:
            skip_stack.pop()

    def _append(s: str) -> None:
        if _in_skip():
            return
        out_chars.append(s)

    while i < n:
        ch = chr(b[i])
        if ch == "{":
            _push(skip_stack[-1])
            i += 1
            continue
        if ch == "}":
            _pop()
            i += 1
            continue
        if ch == "\\":
            i += 1
            if i >= n:
                break
            nxt = chr(b[i])
            # hex escape \'hh
            if nxt == "'":
                if i + 2 < n:
                    hh = b[i + 1 : i + 3]
                    try:
                        byte = int(hh.decode("ascii"), 16)
                        _append(chr(byte))
                    except Exception:
                        pass
                    i += 3
                    continue
                i += 1
                continue
            # control symbol like \{ \} \\ \~
            if nxt in "{}\\":  # escaped literal
                _append(nxt)
                i += 1
                continue
            if nxt == "~":
                _append(" ")
                i += 1
                continue
            if nxt == "*":
                # ignorable destination: mark this group as skipped
                if skip_stack:
                    skip_stack[-1] = True
                i += 1
                continue
            # control word
            start = i
            while i < n and chr(b[i]).isalpha():
                i += 1
            cw = bytes(b[start:i]).decode("ascii", errors="ignore")
            # optional numeric param
            sign = 1
            if i < n and chr(b[i]) == "-":
                sign = -1
                i += 1
            num_start = i
            while i < n and chr(b[i]).isdigit():
                i += 1
            num: Optional[int] = None
            if i > num_start:
                try:
                    num = sign * int(bytes(b[num_start:i]).decode("ascii"))
                except Exception:
                    num = None
            # delimiter: optional space
            if i < n and chr(b[i]) == " ":
                i += 1

            # destinations to skip
            if cw in {"fonttbl", "colortbl", "stylesheet", "info", "pict", "object", "datastore"}:
                if skip_stack:
                    skip_stack[-1] = True
                continue

            if cw in {"par", "line"}:
                _append("\n")
                continue
            if cw == "tab":
                _append("\t")
                continue
            if cw == "emdash":
                _append("—")
                continue
            if cw == "endash":
                _append("–")
                continue
            if cw == "u" and num is not None:
                # \uN? : unicode codepoint, followed by one fallback char to skip.
                try:
                    cp = int(num)
                    if cp < 0:
                        cp = (cp + 0x10000) % 0x10000
                    # Avoid emitting lone surrogate codepoints (not UTF-8 encodable).
                    if 0xD800 <= cp <= 0xDFFF:
                        _append("\uFFFD")
                    else:
                        _append(chr(cp))
                except Exception:
                    pass
                # skip one fallback char if present
                if i < n:
                    i += 1
                continue
            # other control words ignored
            continue
        # regular char
        if not _in_skip():
            if ch == "\n" or ch == "\r":
                # RTF rarely has literal newlines; treat as newline.
                out_chars.append("\n")
            else:
                out_chars.append(ch)
        i += 1

    txt = "".join(out_chars)
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    # Deterministic safety: replace any remaining surrogate codepoints.
    txt = "".join([("\uFFFD" if 0xD800 <= ord(ch) <= 0xDFFF else ch) for ch in txt])
    return txt


def _write_once_text(path: Path, text: str) -> None:
    _ensure_absent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        _fail(f"tmp_exists:{tmp}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def _write_once_json(path: Path, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        _fail(f"tmp_exists:{tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _build_dialogue_history_canonical_v122(
    *,
    conversations_path: Path,
    out_jsonl: Path,
) -> Dict[str, Any]:
    """
    Build canonical dialogue JSONL in deterministic global temporal order using an external sort.
    Streaming: messages are streamed into `sort` stdin; then read back sorted.
    """
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    env["LANG"] = "C"

    sort_cmd = [
        "/usr/bin/sort" if os.path.exists("/usr/bin/sort") else "sort",
        "-t",
        "\t",
        "-k1,1n",
        "-k2,2",
        "-k3,3",
        "-s",
    ]

    p = subprocess.Popen(
        sort_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=False,
    )
    assert p.stdin is not None
    assert p.stdout is not None

    conversations_total = 0
    unknown_role_total = 0
    missing_timestamp_total = 0
    duplicates_dropped_total = 0
    messages_written_to_sort = 0

    # Stream records into sort stdin.
    for conv in _iter_json_array(conversations_path):
        if not isinstance(conv, dict):
            continue
        conv_id = str(conv.get("id") or conv.get("conversation_id") or "")
        if not conv_id:
            continue
        conversations_total += 1
        title = str(conv.get("title") or "")
        conv_version = conv.get("version")
        default_model = str(conv.get("default_model_slug") or "")

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
                unknown_role_total += 1

            content = msg.get("content")
            text = _normalize_text(_extract_text_from_message_content(content))

            ct = _safe_int_epoch(msg.get("create_time"))
            if ct is None:
                missing_timestamp_total += 1
            ct_sort = int(ct) if ct is not None else -1
            ts = _iso_utc_from_epoch_seconds(ct_sort) if ct is not None else ""

            # model hint from message metadata (best-effort)
            md = msg.get("metadata")
            md = md if isinstance(md, dict) else {}
            model_slug = str(md.get("model_slug") or default_model or "")

            rec = {
                "create_time": int(ct_sort),
                "conversation_id": str(conv_id),
                "message_id": str(mid),
                "timestamp": str(ts),
                "role": str(role),
                "text": str(text),
                "meta": {
                    "title": str(title),
                    "model": str(model_slug),
                    "version": int(conv_version) if isinstance(conv_version, int) else None,
                },
            }

            # Tab-separated prefix keys (JSON escapes tabs/newlines).
            line = "{ct}\t{cid}\t{mid}\t{js}\n".format(
                ct=str(int(ct_sort)),
                cid=str(conv_id),
                mid=str(mid),
                js=canonical_json_dumps(rec),
            ).encode("utf-8")
            try:
                p.stdin.write(line)
                messages_written_to_sort += 1
            except BrokenPipeError:
                err = p.stderr.read().decode("utf-8", errors="replace") if p.stderr else ""
                _fail("sort_broken_pipe:" + err[:200])

    p.stdin.close()

    # Now read sorted output and write canonical JSONL.
    _ensure_absent(out_jsonl)
    turns_total = 0

    last_mid_key: Optional[Tuple[str, str]] = None
    last_mid_ct: int = -999999999
    last_mid_text_hash: str = ""

    last_time_text_key: Optional[Tuple[int, str, str, str]] = None
    last_time_text_mid_key: Optional[Tuple[str, str]] = None

    with open(out_jsonl, "xb") as out_f:
        for raw_line in p.stdout:
            try:
                parts = raw_line.decode("utf-8").rstrip("\n").split("\t", 3)
            except Exception:
                _fail("sorted_line_decode_error")
            if len(parts) != 4:
                _fail("sorted_line_bad_fields")
            _ct_s, cid, mid, js = parts
            try:
                rec = json.loads(js)
            except Exception:
                _fail("sorted_line_json_decode_error")
            if str(rec.get("conversation_id") or "") != str(cid) or str(rec.get("message_id") or "") != str(mid):
                _fail("sorted_record_key_mismatch")

            ct_sort = int(rec.get("create_time") or -1)
            role = str(rec.get("role") or "unknown")
            text = str(rec.get("text") or "")
            text_hash = sha256_hex(text.encode("utf-8"))

            mid_key = (str(cid), str(mid))
            if last_mid_key is not None and mid_key == last_mid_key:
                if int(ct_sort) == int(last_mid_ct) and str(text_hash) == str(last_mid_text_hash):
                    duplicates_dropped_total += 1
                    continue
                _fail("duplicate_message_id_conflict:" + canonical_json_dumps({"conversation_id": cid, "message_id": mid}))
            last_mid_key = mid_key
            last_mid_ct = int(ct_sort)
            last_mid_text_hash = str(text_hash)

            time_text_key = (int(ct_sort), str(cid), str(role), str(text_hash))
            if last_time_text_key is not None and time_text_key == last_time_text_key and last_time_text_mid_key is not None and mid_key != last_time_text_mid_key:
                # Same (timestamp,text) within conversation/role -> drop as exact duplicate.
                duplicates_dropped_total += 1
                continue
            last_time_text_key = time_text_key
            last_time_text_mid_key = mid_key

            line_obj = {
                "turn_index": int(turns_total),
                "conversation_id": str(cid),
                "message_id": str(mid),
                "timestamp": str(rec.get("timestamp") or ""),
                "role": str(role),
                "text": str(text),
                "meta": rec.get("meta") if isinstance(rec.get("meta"), dict) else {},
                "source": "chatgpt_export_v122",
            }
            out_f.write((canonical_json_dumps(line_obj) + "\n").encode("utf-8"))
            turns_total += 1

    stderr = p.stderr.read().decode("utf-8", errors="replace") if p.stderr else ""
    rc = p.wait()
    if rc != 0:
        _fail("sort_failed_rc={rc} stderr={err}".format(rc=rc, err=stderr[:500]))

    return {
        "turns_total": int(turns_total),
        "conversations_total": int(conversations_total),
        "unknown_role_total": int(unknown_role_total),
        "missing_timestamp_total": int(missing_timestamp_total),
        "duplicates_dropped_total": int(duplicates_dropped_total),
        "messages_written_to_sort": int(messages_written_to_sort),
    }


def _build_engineering_doc_v122(*, rtf_path: Path, out_plain: Path, out_chunks: Path) -> Dict[str, Any]:
    data = rtf_path.read_bytes()
    plain = _rtf_to_text_v122(data)

    # Minimal deterministic normalization: normalize line endings and strip trailing spaces.
    lines = [ln.rstrip() for ln in plain.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    plain_norm = "\n".join(lines).strip() + "\n"
    _write_once_text(out_plain, plain_norm)

    doc_sha = sha256_hex(plain_norm.encode("utf-8"))
    chunks_total = 0
    chars_total = len(plain_norm)

    _ensure_absent(out_chunks)
    with open(out_chunks, "x", encoding="utf-8") as f:
        off = 0
        chunk_index = 0
        while off < len(plain_norm):
            chunk_text = plain_norm[off : off + int(DOC_CHUNK_CHARS)]
            if not chunk_text:
                break
            sha = sha256_hex(chunk_text.encode("utf-8"))
            chunk_id = "Projeto_ACT:{doc}:{idx:06d}".format(doc=str(doc_sha)[:16], idx=int(chunk_index))
            obj = {
                "doc": "Projeto_ACT",
                "chunk_id": str(chunk_id),
                "heading": "",
                "text": str(chunk_text),
                "offset_start": int(off),
                "offset_end": int(off + len(chunk_text)),
                "sha256_text": str(sha),
            }
            f.write(canonical_json_dumps(obj))
            f.write("\n")
            chunks_total += 1
            off += len(chunk_text)
            chunk_index += 1

    return {
        "doc_sha256": str(doc_sha),
        "doc_chars_total": int(chars_total),
        "chunks_total": int(chunks_total),
        "chunk_chars": int(DOC_CHUNK_CHARS),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations_input", default="", help="Path to Conversations.json / conversations.json (ChatGPT export).")
    ap.add_argument("--rtf_input", default="", help="Path to Projeto ACT.rtf.")
    ap.add_argument("--out", required=True, help="Output directory (WORM).")
    args = ap.parse_args()

    out_dir = Path(str(args.out)).resolve()
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    # Resolve inputs (deterministic discovery).
    conv_path: Optional[Path] = Path(str(args.conversations_input)).expanduser().resolve() if str(args.conversations_input or "") else None
    rtf_path: Optional[Path] = Path(str(args.rtf_input)).expanduser().resolve() if str(args.rtf_input or "") else None
    attempts: List[str] = []
    if conv_path is not None and conv_path.exists():
        attempts.append("conversations_input_arg=" + str(conv_path))
    else:
        conv_found, _, a = _find_desktop_inputs()
        attempts += list(a)
        conv_path = conv_found
    if rtf_path is not None and rtf_path.exists():
        attempts.append("rtf_input_arg=" + str(rtf_path))
    else:
        _, rtf_found, a = _find_desktop_inputs()
        attempts += list(a)
        rtf_path = rtf_found

    if conv_path is None or not conv_path.exists():
        _fail("missing_conversations_json_v122 attempts=" + json.dumps(attempts, ensure_ascii=False, sort_keys=True))
    if rtf_path is None or not rtf_path.exists():
        _fail("missing_projeto_act_rtf_v122 attempts=" + json.dumps(attempts, ensure_ascii=False, sort_keys=True))

    conv_sha = _sha256_file(conv_path)
    rtf_sha = _sha256_file(rtf_path)

    dialogue_jsonl = out_dir / "dialogue_history_canonical_v122.jsonl"
    eng_plain = out_dir / "engineering_doc_plain_v122.txt"
    eng_chunks = out_dir / "engineering_doc_chunks_v122.jsonl"
    manifest_path = out_dir / "manifest_v122.json"

    counts_dialogue = _build_dialogue_history_canonical_v122(conversations_path=conv_path, out_jsonl=dialogue_jsonl)
    counts_doc = _build_engineering_doc_v122(rtf_path=rtf_path, out_plain=eng_plain, out_chunks=eng_chunks)

    out_sha = {
        "dialogue_history_canonical_v122_jsonl": _sha256_file(dialogue_jsonl),
        "engineering_doc_plain_v122_txt": _sha256_file(eng_plain),
        "engineering_doc_chunks_v122_jsonl": _sha256_file(eng_chunks),
    }
    manifest = {
        "schema_version": 122,
        "kind": "external_world_unified_v122",
        "inputs": {
            "conversations_json_path": str(conv_path),
            "conversations_json_sha256": str(conv_sha),
            "projeto_act_rtf_path": str(rtf_path),
            "projeto_act_rtf_sha256": str(rtf_sha),
            "paths_tried": list(attempts),
        },
        "paths": {
            "dialogue_history_canonical_jsonl": str(dialogue_jsonl.name),
            "engineering_doc_plain_txt": str(eng_plain.name),
            "engineering_doc_chunks_jsonl": str(eng_chunks.name),
        },
        "sha256": dict(out_sha),
        "counts": {
            "dialogue": dict(counts_dialogue),
            "engineering_doc": dict(counts_doc),
        },
        "command_line": {"argv": [str(x) for x in sys.argv]},
    }
    manifest["manifest_sig"] = sha256_hex(canonical_json_dumps(manifest).encode("utf-8"))
    _write_once_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "ok": True,
                "reason": "built",
                "out_dir": str(out_dir),
                "manifest_path": str(manifest_path),
                "sha256": dict(out_sha),
                "counts": dict(manifest.get("counts") or {}),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
