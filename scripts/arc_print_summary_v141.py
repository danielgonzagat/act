#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_json", help="Path to run_dir/summary.json")
    args = ap.parse_args()

    p = Path(args.summary_json).resolve()
    if not p.is_file():
        raise SystemExit(f"missing_file:{p}")

    s = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(s, dict):
        raise SystemExit("bad_summary_json")

    out: Dict[str, Any] = {
        "tasks_total": int(s.get("tasks_total") or 0),
        "tasks_solved_by_k": s.get("tasks_solved_by_k") if isinstance(s.get("tasks_solved_by_k"), dict) else {},
        "failure_counts": s.get("failure_counts") if isinstance(s.get("failure_counts"), dict) else {},
        "program_usage": s.get("program_usage") if isinstance(s.get("program_usage"), dict) else {},
        "concept_bank_in": str(s.get("concept_templates_path") or ""),
        "macro_bank_in": str(s.get("macro_templates_path") or ""),
    }
    print(_stable_json(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

