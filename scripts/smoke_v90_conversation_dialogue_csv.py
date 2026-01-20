#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_loop_v90 import run_conversation_v90
from atos_core.conversation_v90 import verify_conversation_chain_v90


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


def _read_payloads(path: str, *, payload_key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            payload = obj.get(payload_key)
            if isinstance(payload, dict):
                rows.append(dict(payload))
    return rows


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    # Fixed deterministic conversation (DSL harness).
    user_turns = [
        "SET x 4",
        "SET y 8",
        "ADD x y",
        "ADD last_answer 3",
        "GET z",
        "SET z 10",
        "ADD x z",
        "SUMMARY",
        "END",
    ]
    res = run_conversation_v90(user_turn_texts=list(user_turns), out_dir=str(out_dir), seed=int(seed))

    # Negative test: corrupt a state field in memory and ensure verify fails.
    states = _read_payloads(res["paths"]["states_jsonl"], payload_key="payload")  # type: ignore[index]
    turns = _read_payloads(res["paths"]["turns_jsonl"], payload_key="payload")  # type: ignore[index]
    if not states or not turns:
        _fail("ERROR: missing states/turns payloads for negative test")
    bad_states = [dict(s) for s in states]
    bad_states[0] = dict(bad_states[0])
    bad_states[0]["state_index"] = 999  # deterministic corruption
    ok_bad, reason_bad, _details_bad = verify_conversation_chain_v90(turns=turns, states=bad_states, tail_k=6)
    if ok_bad:
        _fail("ERROR: expected verify_conversation_chain_v90 to fail on corrupted state_index")

    core = {
        "store_hash": str(res.get("store_hash") or ""),
        "transcript_hash": str(res.get("transcript_hash") or ""),
        "state_chain_hash": str(res.get("state_chain_hash") or ""),
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

    # Determinism: store_hash, transcript_hash, state_chain_hash, ledger_hash must match.
    keys = ["store_hash", "transcript_hash", "state_chain_hash", "ledger_hash", "summary_sha256"]
    for k in keys:
        if str(r1.get(k) or "") != str(r2.get(k) or ""):
            _fail(f"ERROR: determinism mismatch for {k}: try1={r1.get(k)} try2={r2.get(k)}")

    out = {
        "ok": True,
        "seed": int(seed),
        "determinism": {
            "ok": True,
            "store_hash": str(r1.get("store_hash") or ""),
            "transcript_hash": str(r1.get("transcript_hash") or ""),
            "state_chain_hash": str(r1.get("state_chain_hash") or ""),
            "ledger_hash": str(r1.get("ledger_hash") or ""),
            "summary_sha256": str(r1.get("summary_sha256") or ""),
        },
        "negative_test": dict(r1.get("core", {}).get("negative_test", {})),
        "try1": {"out_dir": str(out1), "summary_sha256": str(r1.get("summary_sha256") or "")},
        "try2": {"out_dir": str(out2), "summary_sha256": str(r2.get("summary_sha256") or "")},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

