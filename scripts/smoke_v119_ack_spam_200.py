#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.conversation_loop_v110 import run_conversation_v110
from atos_core.discourse_variants_v119 import prefix2_from_text_v119
from atos_core.fluency_contract_v118 import fluency_contract_v118


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


def _load_jsonl_payload_view(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                continue
            out.append(dict(payload))
    return out


def _transcript_view(path: Path) -> List[Dict[str, Any]]:
    rows = _load_jsonl_payload_view(path)
    out: List[Dict[str, Any]] = []
    for r in rows:
        role = str(r.get("role") or "")
        text = str(r.get("text") or "")
        if role in {"user", "assistant"}:
            out.append({"role": role, "text": text})
    return out


def _assistant_prefix2s(transcript_view: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for r in transcript_view:
        if not isinstance(r, dict):
            continue
        if str(r.get("role") or "") != "assistant":
            continue
        out.append(prefix2_from_text_v119(str(r.get("text") or "")))
    return out


def _max_consecutive_equal(items: Sequence[str]) -> Tuple[int, str]:
    best = 0
    best_item = ""
    cur = 0
    prev = None
    for it in items:
        if it and prev == it:
            cur += 1
        else:
            cur = 1 if it else 0
            prev = it
        if cur > best:
            best = cur
            best_item = it
    return int(best), str(best_item or "")


def _write_once_json(path: Path, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        _fail(f"tmp_exists:{tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _smoke_once(*, out_dir: Path, seed: int, turns: int) -> Dict[str, Any]:
    _ensure_absent(out_dir)

    # A deterministic stress pattern: repeated "ok" turns, with an explicit end so the run terminates.
    user_turns = ["ok"] * int(turns) + ["end now"]

    run_conversation_v110(
        user_turn_texts=list(user_turns),
        out_dir=str(out_dir),
        seed=int(seed),
        discourse_variants_v119_enabled=True,
    )

    transcript_path = out_dir / "transcript.jsonl"
    tv = _transcript_view(transcript_path)
    ok_flu, reason_flu, details_flu = fluency_contract_v118(transcript_view=list(tv))

    prefix2s = _assistant_prefix2s(tv)
    max_run, max_prefix2 = _max_consecutive_equal(prefix2s)

    eval_obj = {
        "schema_version": 119,
        "kind": "smoke_v119_ack_spam_200_eval",
        "seed": int(seed),
        "turns_user_ok": int(turns),
        "assistant_prefix2_max_consecutive": {"n": int(max_run), "prefix2": str(max_prefix2)},
        "fluency": {"ok": bool(ok_flu), "reason": str(reason_flu), "details_sig": str(details_flu.get("metrics_sig") or "")},
    }
    eval_obj["eval_sig"] = sha256_hex(canonical_json_dumps(eval_obj).encode("utf-8"))
    _write_once_json(out_dir / "eval.json", eval_obj)

    # Deterministic summary core.
    core = {
        "seed": int(seed),
        "turns_user_ok": int(turns),
        "fluency_ok": bool(ok_flu),
        "fluency_reason": str(reason_flu),
        "assistant_prefix2_max_consecutive": {"n": int(max_run), "prefix2": str(max_prefix2)},
        "eval_sha256": _sha256_file(out_dir / "eval.json"),
    }
    summary_sha256 = sha256_hex(canonical_json_dumps(core).encode("utf-8"))
    summary = {"schema_version": 119, "kind": "smoke_v119_ack_spam_200_summary", "core": dict(core), "summary_sha256": str(summary_sha256)}
    _write_once_json(out_dir / "smoke_summary.json", summary)

    if not bool(ok_flu):
        _fail(f"fluency_fail:{reason_flu}")
    if int(max_run) >= 4 and str(max_prefix2):
        _fail(f"consecutive_prefix2_repeat_detected:{max_prefix2}")

    return {"core": dict(core), "summary_sha256": str(summary_sha256)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--turns", type=int, default=200)
    args = ap.parse_args()

    out_base = str(args.out_base)
    seed = int(args.seed)
    turns = int(args.turns)

    r1 = _smoke_once(out_dir=Path(out_base + "_try1"), seed=seed, turns=turns)
    r2 = _smoke_once(out_dir=Path(out_base + "_try2"), seed=seed, turns=turns)

    if canonical_json_dumps(r1["core"]) != canonical_json_dumps(r2["core"]):
        _fail("determinism_core_mismatch")
    if str(r1["summary_sha256"]) != str(r2["summary_sha256"]):
        _fail("determinism_summary_sha256_mismatch")

    print(
        json.dumps(
            {
                "ok": True,
                "determinism_ok": True,
                "summary_sha256": str(r1["summary_sha256"]),
                "turns_user_ok": int(turns),
                "out_try1": str(out_base + "_try1"),
                "out_try2": str(out_base + "_try2"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
