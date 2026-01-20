#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.engine import Engine
from atos_core.store import ActStore
from atos_core.suite import SKILL_DIALOGUES_V0, run_skill_suite


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def transcripts_text(transcripts: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(str(r.get("full_text", "")) for r in transcripts)


def first_plan_trace(transcripts: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for rec in transcripts:
        turns = rec.get("turns", [])
        if not isinstance(turns, list):
            continue
        for t in turns:
            if not isinstance(t, dict):
                continue
            tr = t.get("trace") or {}
            if not isinstance(tr, dict):
                continue
            pt = tr.get("plan_trace")
            if isinstance(pt, dict):
                return pt
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run dir containing acts.jsonl (read-only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    acts_path = os.path.join(args.run, "acts.jsonl")
    store = ActStore.load_jsonl(acts_path)
    engine = Engine(store, seed=args.seed)

    transcripts, metrics = run_skill_suite(
        engine, tasks=SKILL_DIALOGUES_V0, max_new_tokens=args.max_new_tokens
    )
    txt = transcripts_text(transcripts)
    out: Dict[str, Any] = {
        "run": str(args.run),
        "seed": int(args.seed),
        "max_new_tokens": int(args.max_new_tokens),
        "sha256_transcript_text": sha256_text(txt),
        "plan_trace_sample": first_plan_trace(transcripts),
        **dict(metrics),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

