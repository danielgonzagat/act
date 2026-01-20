#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.metrics import tokenize_text
from atos_core.suite import non_ws_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run dir containing transcripts.jsonl")
    ap.add_argument("--prompt_id", type=int, default=0)
    ap.add_argument("--turn", type=int, default=0, help="0-based turn index in the transcript")
    ap.add_argument("--k", type=int, default=24, help="How many non-ws lower tokens to print")
    args = ap.parse_args()

    transcripts_path = os.path.join(args.run, "transcripts.jsonl")
    with open(transcripts_path, "r", encoding="utf-8") as f:
        rec: Dict[str, Any] | None = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if int(r.get("prompt_id", -1)) == int(args.prompt_id):
                rec = r
                break
    if rec is None:
        raise SystemExit(f"prompt_id={args.prompt_id} not found in {transcripts_path}")

    turns = rec.get("turns", [])
    if not isinstance(turns, list) or args.turn < 0 or args.turn >= len(turns):
        raise SystemExit(f"turn={args.turn} out of range for prompt_id={args.prompt_id}")

    reply = str(turns[args.turn].get("system", ""))
    toks = tokenize_text(reply)
    non = non_ws_tokens(toks)
    lower = [t.lower() for t in non]

    out = {
        "run": os.path.normpath(args.run),
        "prompt_id": int(args.prompt_id),
        "turn": int(args.turn),
        "first_non_ws_lower": lower[: int(args.k)],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

