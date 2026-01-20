#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_loop_v91 import run_conversation_v91
from atos_core.conversation_v91 import verify_conversation_chain_v91


def sha256_file(path: str) -> str:
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


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"ERROR: path already exists (WORM): {path}")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_payloads(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in _read_jsonl(path):
        payload = r.get("payload")
        if isinstance(payload, dict):
            out.append(dict(payload))
    return out


def _extract_parse_events(path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for r in _read_jsonl(path):
        if not isinstance(r, dict):
            continue
        events.append(
            {
                "turn_id": str(r.get("turn_id") or ""),
                "turn_index": int(r.get("turn_index") or 0),
                "payload": dict(r.get("payload") or {}) if isinstance(r.get("payload"), dict) else {},
            }
        )
    return events


def _extract_trials_for_verify(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in _read_jsonl(path):
        if not isinstance(r, dict):
            continue
        out.append(
            {
                "objective_kind": str(r.get("objective_kind") or ""),
                "user_turn_id": str(r.get("user_turn_id") or r.get("turn_id") or ""),
            }
        )
    return out


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    # Fixed deterministic conversation with EN+PT paraphrases.
    user_turns = [
        "Set x to four",
        "defina y como 8",
        "sum x + 10",
        "add last_answer and three",
        "what is z?",
        "set z 10",
        "some x e z",
        "show variables",
        "set w",
        "w = 2",
        "blorp",
        "fim",
    ]
    res = run_conversation_v91(user_turn_texts=list(user_turns), out_dir=str(out_dir), seed=int(seed))

    turns = _extract_payloads(res["paths"]["turns_jsonl"])  # type: ignore[index]
    states = _extract_payloads(res["paths"]["states_jsonl"])  # type: ignore[index]
    parse_events = _extract_parse_events(res["paths"]["parses_jsonl"])  # type: ignore[index]
    trials = _extract_trials_for_verify(res["paths"]["trials_jsonl"])  # type: ignore[index]

    # Negative test: corrupt a user-turn parse_sig and ensure verify fails (fail-closed).
    if not turns:
        _fail("ERROR: missing turns payloads for negative test")
    bad_turns = [dict(t) for t in turns]
    # First user turn is turn_index 0.
    bad_turns[0] = dict(bad_turns[0])
    refs = bad_turns[0].get("refs")
    refs = dict(refs) if isinstance(refs, dict) else {}
    refs["parse_sig"] = "0" * 64
    bad_turns[0]["refs"] = refs
    ok_bad, reason_bad, _details_bad = verify_conversation_chain_v91(
        turns=bad_turns, states=states, parse_events=parse_events, trials=trials, tail_k=6
    )
    if ok_bad:
        _fail("ERROR: expected verify_conversation_chain_v91 to fail on corrupted turn.refs.parse_sig")

    core = {
        "store_hash": str(res.get("store_hash") or ""),
        "transcript_hash": str(res.get("transcript_hash") or ""),
        "state_chain_hash": str(res.get("state_chain_hash") or ""),
        "parse_chain_hash": str(res.get("parse_chain_hash") or ""),
        "ledger_hash": str(res.get("ledger_hash") or ""),
        "summary_sha256": str(res.get("summary_sha256") or ""),
        "negative_test": {"ok": True, "reason": str(reason_bad)},
        "sha256_summary_json": sha256_file(res["paths"]["summary_json"]),  # type: ignore[index]
    }
    return dict(res, core=core)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_base = str(args.out_base)
    seed = int(args.seed)

    out1 = f"{out_base}_try1"
    out2 = f"{out_base}_try2"
    ensure_absent(out1)
    ensure_absent(out2)

    r1 = smoke_try(out_dir=out1, seed=seed)
    r2 = smoke_try(out_dir=out2, seed=seed)

    keys = ["store_hash", "transcript_hash", "state_chain_hash", "parse_chain_hash", "ledger_hash", "summary_sha256"]
    for k in keys:
        if str(r1.get(k) or "") != str(r2.get(k) or ""):
            _fail(f"ERROR: determinism mismatch for {k}: try1={r1.get(k)} try2={r2.get(k)}")

    out = {
        "ok": True,
        "seed": int(seed),
        "determinism_ok": True,
        "determinism": {k: str(r1.get(k) or "") for k in keys},
        "negative_test": dict(r1.get("core", {}).get("negative_test", {})),
        "try1": {"out_dir": str(out1), "summary_sha256": str(r1.get("summary_sha256") or "")},
        "try2": {"out_dir": str(out2), "summary_sha256": str(r2.get("summary_sha256") or "")},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

