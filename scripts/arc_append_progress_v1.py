#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_summary(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "summary.json"
    if not p.is_file():
        return {}
    raw = _read_json(p)
    return raw if isinstance(raw, dict) else {}


def _load_eval(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "eval.json"
    if not p.is_file():
        return {}
    raw = _read_json(p)
    return raw if isinstance(raw, dict) else {}


def _load_omega_state(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    raw = _read_json(path)
    return raw if isinstance(raw, dict) else {}


def _omega_metrics(omega: Dict[str, Any]) -> Dict[str, Any]:
    if not omega:
        return {}
    banned_fams = omega.get("banned_task_families") if isinstance(omega.get("banned_task_families"), list) else []
    banned_rsig = (
        omega.get("banned_residual_signatures") if isinstance(omega.get("banned_residual_signatures"), list) else []
    )
    concepts = omega.get("concepts") if isinstance(omega.get("concepts"), list) else []
    operators = omega.get("operators") if isinstance(omega.get("operators"), list) else []
    promoted_concepts = 0
    for c in concepts:
        if isinstance(c, dict) and str(c.get("state") or "") == "promoted":
            promoted_concepts += 1
    promoted_ops = 0
    for o in operators:
        if isinstance(o, dict) and str(o.get("state") or "") == "promoted":
            promoted_ops += 1
    return {
        "runs_total": int(omega.get("runs_total") or 0),
        "state_sha": str(omega.get("state_sha") or ""),
        "prev_state_sha": str(omega.get("prev_state_sha") or ""),
        "strict_burns_total": int(omega.get("strict_burns_total") or 0),
        "dead_clusters_total": int(omega.get("dead_clusters_total") or 0),
        "max_depth_cap": int(omega.get("max_depth_cap") or 0),
        "max_programs_cap": int(omega.get("max_programs_cap") or 0),
        "banned_task_families_total": int(len([x for x in banned_fams if str(x)])),
        "banned_residual_signatures_total": int(len([x for x in banned_rsig if str(x)])),
        "promoted_concepts_total": int(promoted_concepts),
        "promoted_operators_total": int(promoted_ops),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Append deterministic ARC loop progress.jsonl entries (WORM-safe append).")
    ap.add_argument("--progress", required=True, help="Path to progress.jsonl (created if missing; appended if exists).")
    ap.add_argument("--stage", required=True, help="Stage tag (e.g., base, omega_base, macro_run, omega_macro).")
    ap.add_argument("--run_dir", required=True, help="ARC run dir containing summary.json/eval.json (v141).")
    ap.add_argument("--omega_state", default="", help="Optional omega_state_v2*.json to include (v2).")
    ap.add_argument("--concept_bank", default="", help="Optional concept bank jsonl path (for bookkeeping).")
    ap.add_argument("--macro_bank", default="", help="Optional macro/operator bank jsonl path (for bookkeeping).")
    args = ap.parse_args()

    run_dir = Path(str(args.run_dir)).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"missing_run_dir:{run_dir}")

    progress_path = Path(str(args.progress)).resolve()
    omega_state_path = Path(str(args.omega_state)).resolve() if str(args.omega_state).strip() else None

    summary = _load_summary(run_dir)
    eval_obj = _load_eval(run_dir)
    omega = _load_omega_state(omega_state_path)

    # Keep the record small and audit-friendly; embed only key metrics.
    rec: Dict[str, Any] = {
        "kind": "arc_progress_v1",
        "schema_version": 1,
        "stage": str(args.stage),
        "run_dir": str(run_dir),
        "run_summary": {
            "tasks_total": int(summary.get("tasks_total") or 0),
            "tasks_solved_by_k": summary.get("tasks_solved_by_k") if isinstance(summary.get("tasks_solved_by_k"), dict) else {},
            "failure_counts": summary.get("failure_counts") if isinstance(summary.get("failure_counts"), dict) else {},
            "program_usage": summary.get("program_usage") if isinstance(summary.get("program_usage"), dict) else {},
            "max_depth": int(summary.get("max_depth") or 0),
            "max_programs": int(summary.get("max_programs") or 0),
            "abstraction_pressure": bool(summary.get("abstraction_pressure") or False),
            "macro_try_on_fail_only": bool(summary.get("macro_try_on_fail_only") or False),
            "macro_templates_total": int(summary.get("macro_templates_total") or 0),
            "concept_templates_total": int(summary.get("concept_templates_total") or 0),
            "summary_sig": str(summary.get("summary_sig") or ""),
        },
        "eval_sig": str(eval_obj.get("eval_sig") or ""),
        "banks": {
            "concept_bank": str(args.concept_bank or ""),
            "macro_bank": str(args.macro_bank or ""),
        },
        "omega_state": _omega_metrics(omega),
    }

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(_stable_json(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
