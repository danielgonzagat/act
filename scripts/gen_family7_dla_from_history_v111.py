#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex


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
        _fail("offsets_size_not_multiple_of_8")
    out: List[int] = []
    for i in range(0, len(data), 8):
        out.append(int(struct.unpack("<Q", data[i : i + 8])[0]))
    return out


def _world_paths_from_manifest(manifest_path: str) -> Dict[str, str]:
    mp = str(manifest_path)
    with open(mp, "r", encoding="utf-8") as f:
        m = json.load(f)
    world_root = os.path.dirname(os.path.dirname(os.path.abspath(mp)))
    paths = m.get("paths") if isinstance(m.get("paths"), dict) else {}
    out: Dict[str, str] = {}
    for k in ["canonical_jsonl", "offsets_bin", "conversations_index_json"]:
        rel = str(paths.get(k) or "")
        if not rel:
            _fail(f"manifest_missing_path:{k}")
        out[k] = os.path.normpath(os.path.join(world_root, rel))
    out["world_root"] = world_root
    out["manifest_sha256"] = _sha256_file(mp)
    out["manifest_path"] = mp
    return out


def _fetch_turn_texts_user_only(
    *,
    canon_path: str,
    offsets: List[int],
    start_turn: int,
    end_turn: int,
    max_user_turns: int,
) -> List[str]:
    out: List[str] = []
    with open(canon_path, "rb") as f:
        for idx in range(int(start_turn), int(end_turn) + 1):
            if idx < 0 or idx >= len(offsets):
                break
            f.seek(int(offsets[idx]))
            line = f.readline()
            try:
                obj = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            if str(obj.get("role") or "") != "user":
                continue
            txt = str(obj.get("text") or "")
            if txt:
                out.append(txt)
                if len(out) >= int(max_user_turns):
                    break
    return list(out)


def _is_safe_user_turn_text_v111(text: str) -> bool:
    """
    Deterministic filter to avoid UI/structured payload turns that can break
    baseline parsers (we still preserve chaos via real turns + adversarial injections).
    """
    t = str(text or "").strip()
    if not t:
        return False
    # Keep only reasonably-sized natural turns for the initial v111 smoke.
    if len(t) > 500:
        return False
    t0 = t.lstrip()
    if t0.startswith("{") or t0.startswith("["):
        return False
    bad_substrings = [
        "content_type",
        "asset_pointer",
        "file-service://",
        "multimodal_text",
        "image_asset_pointer",
    ]
    for s in bad_substrings:
        if s in t0:
            return False
    if t0.count("\n") > 6:
        return False
    return True


def _inject_adversarial_turns_v111(user_turns: Sequence[str]) -> List[str]:
    """
    Deterministic, minimal injection (no dependency on content).
    """
    out = list([str(x) for x in user_turns if isinstance(x, str)])
    inject = [
        ("ok", 3),
        ("continua", 7),
        ("isso", 11),
        ("na verdade era X, não Y", 17),
        ("não invente nada", 23),
        ("ok", 29),
    ]
    for text, pos in sorted(inject, key=lambda t: int(t[1])):
        i = int(pos)
        if i < 0:
            continue
        if i > len(out):
            i = len(out)
        out.insert(i, str(text))
    return list(out)


def _make_task(task_kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = dict(payload)
    body["schema_version"] = 111
    body["task_kind"] = str(task_kind)
    task_id = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return dict(body, task_id=f"family7_dla_v111_{task_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world_manifest", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seed = int(args.seed)
    out_path = str(args.out)
    if os.path.exists(out_path):
        _fail(f"worm_exists:{out_path}")

    paths = _world_paths_from_manifest(str(args.world_manifest))
    canon_path = paths["canonical_jsonl"]
    offsets_path = paths["offsets_bin"]
    conv_index_path = paths["conversations_index_json"]
    if not (os.path.exists(canon_path) and os.path.exists(offsets_path) and os.path.exists(conv_index_path)):
        _fail("world_paths_missing")

    offsets = _read_u64_le_list(offsets_path)

    with open(conv_index_path, "r", encoding="utf-8") as f:
        conv_index = json.load(f)
    convs = conv_index.get("conversations") if isinstance(conv_index.get("conversations"), list) else []
    if not convs:
        _fail("empty_conversations_index")

    # Pick deterministic conversations by total turns DESC, then conversation_id ASC.
    cands: List[Dict[str, Any]] = []
    for c in convs:
        if not isinstance(c, dict):
            continue
        turns_total = int(c.get("turns_total") or 0)
        if turns_total < 200:
            continue
        cid = str(c.get("conversation_id") or "")
        if not cid:
            continue
        cands.append(dict(c))
    cands.sort(key=lambda d: (-int(d.get("turns_total") or 0), str(d.get("conversation_id") or "")))
    if len(cands) < 2:
        _fail("not_enough_large_conversations")

    tasks: List[Dict[str, Any]] = []
    for c in cands[:2]:
        cid = str(c.get("conversation_id") or "")
        s = int(c.get("start_turn") or 0)
        e = int(c.get("end_turn") or 0)
        user_turns = _fetch_turn_texts_user_only(canon_path=canon_path, offsets=offsets, start_turn=s, end_turn=e, max_user_turns=40)
        safe_turns = [t for t in user_turns if _is_safe_user_turn_text_v111(t)]
        if len(safe_turns) < 6:
            continue

        # Inject a synthetic goal + make dialogue long under minimal user driving, plus a small sample of real user turns.
        goal_turn = "goal: autopilot demo outcome=complete constraints=deterministic deadline=60"
        # Keep a small, deterministic slice of real turns to preserve chaos source.
        real_sample = safe_turns[:6]
        minimal = ["ok"] * 60
        turns = [goal_turn] + real_sample + _inject_adversarial_turns_v111(minimal)

        tasks.append(
            _make_task(
                "family7_dla_task_v111",
                {
                    "seed": int(seed),
                    "world_manifest": str(paths["manifest_path"]),
                    "conversation_id": cid,
                    "window": {"start_turn": int(s), "end_turn": int(e)},
                    "user_turns": list(turns),
                    "injection": {"real_user_sample_turns": int(len(real_sample)), "minimal_ok_turns": int(len(minimal))},
                },
            )
        )

    if len(tasks) < 2:
        _fail("failed_to_build_two_tasks")

    # Write tasks jsonl (WORM).
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "x", encoding="utf-8") as f:
        for t in tasks:
            f.write(canonical_json_dumps(t))
            f.write("\n")

    manifest_out = os.path.splitext(out_path)[0] + "_manifest.json"
    if os.path.exists(manifest_out):
        _fail(f"worm_exists:{manifest_out}")

    tasks_sha = _sha256_file(out_path)
    manifest = {
        "schema_version": 111,
        "kind": "family7_dla_tasks_v111",
        "seed": int(seed),
        "paths": {"tasks_jsonl": out_path, "world_manifest": str(paths["manifest_path"])},
        "sha256": {"tasks_jsonl": tasks_sha, "world_manifest": str(paths["manifest_sha256"])},
        "tasks_total": int(len(tasks)),
    }
    with open(manifest_out, "x", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")

    print(
        json.dumps(
            {"ok": True, "tasks": out_path, "tasks_sha256": tasks_sha, "manifest": manifest_out, "tasks_total": len(tasks)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
