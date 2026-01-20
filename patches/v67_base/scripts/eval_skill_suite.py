#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.engine import Engine, EngineConfig
from atos_core.store import ActStore
from atos_core.suite import (
    SKILL_DIALOGUES_V0,
    SKILL_DIALOGUES_V1_PARAPHRASE,
    UTILITY_DIALOGUES_V66,
    run_skill_suite,
    suite_metrics_from_transcripts,
)


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
    ap.add_argument("--enable_contracts", action="store_true", help="Enable deterministic instruction contracts.")
    ap.add_argument(
        "--utility_suite_version",
        choices=["v0", "v1", "v66"],
        default="v0",
        help="Which deterministic utility suite to run (v0=baseline, v1=paraphrase, v66=utility-as-law).",
    )
    args = ap.parse_args()

    acts_path = os.path.join(args.run, "acts.jsonl")
    store = ActStore.load_jsonl(acts_path)
    engine = Engine(store, seed=args.seed, config=EngineConfig(enable_contracts=bool(args.enable_contracts)))

    if str(args.utility_suite_version) == "v0":
        tasks = SKILL_DIALOGUES_V0
    elif str(args.utility_suite_version) == "v1":
        tasks = SKILL_DIALOGUES_V1_PARAPHRASE
    else:
        tasks = UTILITY_DIALOGUES_V66
    transcripts, metrics = run_skill_suite(
        engine, tasks=tasks, max_new_tokens=args.max_new_tokens
    )
    cost_metrics = suite_metrics_from_transcripts(transcripts)
    txt = transcripts_text(transcripts)
    out: Dict[str, Any] = {
        "run": str(args.run),
        "seed": int(args.seed),
        "max_new_tokens": int(args.max_new_tokens),
        "utility_suite_version": str(args.utility_suite_version),
        "sha256_transcript_text": sha256_text(txt),
        "plan_trace_sample": first_plan_trace(transcripts),
        "cost": {k: cost_metrics.get(k) for k in sorted(cost_metrics.keys())},
        **dict(metrics),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
