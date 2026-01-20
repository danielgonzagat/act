#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.csg_v130 import verify_chained_jsonl_v130


def _fail(reason: str, details: Dict[str, Any]) -> None:
    out = {"ok": False, "reason": str(reason), "details": dict(details)}
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))
    raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    run_dir = str(args.run_dir)
    if not os.path.isdir(run_dir):
        _fail("run_dir_not_found", {"run_dir": run_dir})

    concepts = os.path.join(run_dir, "concepts.jsonl")
    evidence = os.path.join(run_dir, "concept_evidence.jsonl")
    telemetry = os.path.join(run_dir, "concept_telemetry.jsonl")
    for p in [concepts, evidence, telemetry]:
        if not os.path.exists(p):
            _fail("missing_required_log", {"path": str(p)})

    ok_concepts = bool(verify_chained_jsonl_v130(concepts))
    ok_evidence = bool(verify_chained_jsonl_v130(evidence))
    ok_telemetry = bool(verify_chained_jsonl_v130(telemetry))
    if not (ok_concepts and ok_evidence and ok_telemetry):
        _fail(
            "hash_chain_failed",
            {
                "concepts_chain_ok": ok_concepts,
                "evidence_chain_ok": ok_evidence,
                "telemetry_chain_ok": ok_telemetry,
            },
        )

    out = {
        "ok": True,
        "run_dir": str(run_dir),
        "concepts_chain_ok": True,
        "evidence_chain_ok": True,
        "telemetry_chain_ok": True,
    }
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

