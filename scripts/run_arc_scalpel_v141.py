#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _ensure_absent(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"worm_exists:{path}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_once_json(path: Path, obj: Any) -> None:
    _ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise SystemExit(f"tmp_exists:{tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _write_text_x(path: Path, text: str) -> None:
    _ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


def _excluded_dir_parts_v141() -> set:
    return {
        ".git",
        "__pycache__",
        ".pycache",
        "results",
        "external_world",
        "external_world_v122",
        "external_world_v122_try2",
        "external_world_v122_try3",
        "external_world_v122_try4",
        "external_world_v122_try5",
        "external_world_v122_try6",
    }


def _repo_snapshot_sha256_v141(*, root: Path, exclude_paths: Sequence[Path]) -> str:
    excluded = _excluded_dir_parts_v141()
    excludes = [p.resolve() for p in exclude_paths]
    rows: List[Dict[str, Any]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in excluded for part in p.parts):
            continue
        rp = p.resolve()
        if any(str(rp).startswith(str(ex)) for ex in excludes):
            continue
        rel = p.relative_to(root).as_posix()
        rows.append({"path": str(rel), "sha256": _sha256_file(p)})
    rows.sort(key=lambda r: str(r["path"]))
    body = {"schema_version": 141, "kind": "repo_snapshot_v141", "files": rows}
    from atos_core.act import canonical_json_dumps, sha256_hex

    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _sanitize_task_id(task_id: str) -> str:
    s = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(task_id)])
    return s or "task"


def _grid_equal(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if list(ra) != list(rb):
            return False
    return True


def _benchmark_profile_default_trials(profile: str) -> int:
    p = str(profile)
    if p == "ARC_AGI2_PROFILE":
        return 2
    return 3  # ARC_AGI1_PROFILE default


def _score_test_case_all_k_v141(
    *,
    solver_res: Dict[str, Any],
    want_grid: Optional[Sequence[Sequence[int]]],
    max_trials: int,
) -> Dict[str, Any]:
    status = str(solver_res.get("status") or "")
    if want_grid is None:
        fk = ""
        fr = solver_res.get("failure_reason")
        if isinstance(fr, dict):
            fk = str(fr.get("kind") or "")
        return {
            "solver_status": status if status in {"SOLVED", "UNKNOWN"} else "FAIL",
            "scored": False,
            "failure_kind": fk,
            "scores_by_k": {},
        }

    # Candidate outputs from solver.
    preds = solver_res.get("predicted_grids")
    if isinstance(preds, list) and preds:
        candidates = [p for p in preds if isinstance(p, list)]
    else:
        pred = solver_res.get("predicted_grid")
        candidates = [pred] if isinstance(pred, list) else []

    scores_by_k: Dict[str, Any] = {}
    for k in (1, 2, 3):
        kk = int(k)
        if kk <= 0:
            continue
        if kk > int(max_trials):
            continue
        ok = False
        for cand in candidates[:kk]:
            if _grid_equal(cand, want_grid):
                ok = True
                break
        scores_by_k[str(kk)] = {"k": int(kk), "ok": bool(ok)}
    # failure_kind for aggregates uses solver status/failure_reason.
    fk = ""
    fr = solver_res.get("failure_reason")
    if isinstance(fr, dict):
        fk = str(fr.get("kind") or "")
    return {
        "solver_status": status if status in {"SOLVED", "UNKNOWN"} else "FAIL",
        "scored": True,
        "failure_kind": fk,
        "scores_by_k": scores_by_k,
    }


def _summarize_scores_v141(rows: List[Dict[str, Any]], *, max_trials: int) -> Dict[str, Any]:
    total = int(len(rows))
    tasks_solved_by_k: Dict[str, int] = {str(k): 0 for k in range(1, int(max_trials) + 1)}
    for r in rows:
        s = r.get("scoring") if isinstance(r.get("scoring"), dict) else {}
        by_k = s.get("scores_by_k") if isinstance(s.get("scores_by_k"), dict) else {}
        # For unscored tasks (e.g. internal sample suites with no test outputs),
        # count SOLVED based on solver_status for all k.
        if not bool(s.get("scored")):
            if str(s.get("solver_status") or "") == "SOLVED":
                for k in tasks_solved_by_k.keys():
                    tasks_solved_by_k[str(k)] = int(tasks_solved_by_k[str(k)]) + 1
            continue
        for k in tasks_solved_by_k.keys():
            ok = False
            item = by_k.get(k) if isinstance(by_k.get(k), dict) else {}
            ok = bool(item.get("ok")) if isinstance(item, dict) else False
            if ok:
                tasks_solved_by_k[str(k)] = int(tasks_solved_by_k[str(k)]) + 1
    solved_kmax = int(tasks_solved_by_k.get(str(max_trials), 0))
    solve_rate = float(solved_kmax) / float(total) if total else 0.0
    return {"tasks_total": int(total), "tasks_solved_by_k": tasks_solved_by_k, "solve_rate_kmax": float(solve_rate)}


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl_x(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        for row in rows:
            f.write(_stable_json(row) + "\n")


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _make_diag_report_v141(*, summary_obj: Dict[str, Any], eval_obj: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# ARC_DIAG_REPORT_v141")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(_stable_json(summary_obj))
    lines.append("")
    lines.append("## Failure Counts")
    lines.append("")
    fc = eval_obj.get("failure_counts") if isinstance(eval_obj.get("failure_counts"), dict) else {}
    lines.append(_stable_json(fc))
    lines.append("")
    return "\n".join(lines) + "\n"


def _outputs_manifest_v141(*, run_dir: Path) -> Dict[str, Any]:
    # Deterministic listing of key artifacts (sha256).
    per_task_dir = run_dir / "per_task"
    per_task_files: List[str] = []
    if per_task_dir.is_dir():
        for p in per_task_dir.iterdir():
            if p.is_file() and p.name.endswith(".json"):
                per_task_files.append(p.name)
    per_task_files.sort()
    manifest: Dict[str, Any] = {
        "schema_version": 141,
        "kind": "outputs_manifest_v141",
        "files": [],
    }
    files: List[Dict[str, Any]] = []

    def add(rel: str) -> None:
        p = run_dir / rel
        if not p.exists():
            return
        files.append({"path": str(rel), "sha256": _sha256_file(p)})

    add("summary.json")
    add("eval.json")
    add("isolation_check_v141.json")
    add("trace_candidates.jsonl")
    add("per_task_manifest.jsonl")
    add("ARC_DIAG_REPORT_v141.md")
    add("input/arc_manifest_v141.json")
    add("input/arc_tasks_canonical_v141.jsonl")
    for fn in per_task_files:
        add(f"per_task/{fn}")
    files.sort(key=lambda r: str(r["path"]))
    manifest["files"] = files
    from atos_core.act import canonical_json_dumps, sha256_hex

    manifest["outputs_manifest_sig"] = sha256_hex(canonical_json_dumps(manifest).encode("utf-8"))
    return manifest


def _run_once_v141(
    *,
    out_dir: Path,
    arc_root: str,
    split: Optional[str],
    limit: int,
    seed: int,
    max_depth: int,
    max_programs: int,
    max_trials: int,
    benchmark_profile: str,
) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    root = Path(__file__).resolve().parent.parent
    before_sig = _repo_snapshot_sha256_v141(root=root, exclude_paths=[out_dir])

    input_dir = out_dir / "input"
    input_jsonl = input_dir / "arc_tasks_canonical_v141.jsonl"
    input_manifest = input_dir / "arc_manifest_v141.json"

    from atos_core.arc_loader_v141 import iter_canonical_tasks_v141, write_arc_canonical_jsonl_v141

    manifest_obj = write_arc_canonical_jsonl_v141(
        arc_root=str(arc_root),
        split=str(split) if split is not None else None,
        limit=int(limit),
        out_jsonl=input_jsonl,
        out_manifest=input_manifest,
    )

    from atos_core.arc_solver_v141 import SolveConfigV141, solve_arc_task_v141

    cfg = SolveConfigV141(max_depth=int(max_depth), max_programs=int(max_programs), trace_program_limit=80, max_ambiguous_outputs=int(max_trials))

    per_task_rows: List[Dict[str, Any]] = []
    trace_candidates_rows: List[Dict[str, Any]] = []
    scoring_rows: List[Dict[str, Any]] = []

    for task in iter_canonical_tasks_v141(str(input_jsonl)):
        solver_res = solve_arc_task_v141(train_pairs=list(task.train_pairs), test_in=task.test_pairs[0][0], config=cfg)
        # Scoring against first test output only (ARC tasks have 1 test by default).
        want_grid = task.test_pairs[0][1]
        scoring = _score_test_case_all_k_v141(solver_res=solver_res, want_grid=want_grid, max_trials=int(max_trials))
        row = {
            "schema_version": 141,
            "kind": "arc_per_task_v141",
            "task": task.to_dict(),
            "solver_results": [solver_res],
            "scoring": scoring,
        }
        per_task_rows.append(row)

        # Per-task manifest entry (small).
        solver_status = str(solver_res.get("status") or "")
        fk = str(scoring.get("failure_kind") or "")
        per_task_manifest_row = {
            "schema_version": 141,
            "kind": "arc_per_task_manifest_row_v141",
            "task_id": str(task.task_id),
            "status": solver_status,
            "failure_kind": fk,
            "program_sig": str(solver_res.get("program_sig") or ""),
            "predicted_grid_hash": str(solver_res.get("predicted_grid_hash") or ""),
        }
        scoring_rows.append(per_task_manifest_row)

        trace = solver_res.get("trace") if isinstance(solver_res.get("trace"), dict) else {}
        for tp in trace.get("trace_programs", []) if isinstance(trace.get("trace_programs"), list) else []:
            if not isinstance(tp, dict):
                continue
            trace_candidates_rows.append(
                {
                    "schema_version": 141,
                    "kind": "arc_trace_candidate_v141",
                    "task_id": str(task.task_id),
                    **{str(k): tp.get(k) for k in sorted(tp.keys())},
                }
            )

        # Write each task row under per_task/
        per_task_dir = out_dir / "per_task"
        per_task_dir.mkdir(parents=True, exist_ok=True)
        safe = _sanitize_task_id(task.task_id)
        out_path = per_task_dir / f"{safe}.json"
        _ensure_absent(out_path)
        out_path.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Write aggregate jsonl artifacts.
    _write_jsonl_x(out_dir / "per_task_manifest.jsonl", scoring_rows)
    _write_jsonl_x(out_dir / "trace_candidates.jsonl", trace_candidates_rows)

    # Aggregate eval + summary.
    failure_counts: Dict[str, int] = {}
    for r in scoring_rows:
        st = str(r.get("status") or "")
        if st == "SOLVED":
            continue
        fk = str(r.get("failure_kind") or "")
        if fk:
            failure_counts[fk] = int(failure_counts.get(fk, 0)) + 1

    score_summary = _summarize_scores_v141(per_task_rows, max_trials=int(max_trials))
    eval_obj: Dict[str, Any] = {
        "schema_version": 141,
        "kind": "arc_eval_v141",
        "failure_counts": {k: int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "tasks_total": int(score_summary["tasks_total"]),
        "tasks_solved_by_k": score_summary["tasks_solved_by_k"],
    }
    from atos_core.act import canonical_json_dumps, sha256_hex

    eval_obj["eval_sig"] = sha256_hex(canonical_json_dumps(eval_obj).encode("utf-8"))
    _write_once_json(out_dir / "eval.json", eval_obj)

    summary_obj: Dict[str, Any] = {
        "schema_version": 141,
        "kind": "arc_summary_v141",
        "arc_root": str(arc_root),
        "split": str(split or ""),
        "limit": int(limit),
        "seed": int(seed),
        "max_depth": int(max_depth),
        "max_programs": int(max_programs),
        "benchmark_profile": str(benchmark_profile),
        "max_trials": int(max_trials),
        "tasks_total": int(score_summary["tasks_total"]),
        "tasks_solved_by_k": score_summary["tasks_solved_by_k"],
        "failure_counts": {k: int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "eval_sig": str(eval_obj["eval_sig"]),
    }
    summary_obj["summary_sig"] = sha256_hex(canonical_json_dumps(summary_obj).encode("utf-8"))
    _write_once_json(out_dir / "summary.json", summary_obj)

    # Diag report is deterministic function of summary+eval.
    _write_text_x(out_dir / "ARC_DIAG_REPORT_v141.md", _make_diag_report_v141(summary_obj=summary_obj, eval_obj=eval_obj))

    # Isolation check
    after_sig = _repo_snapshot_sha256_v141(root=root, exclude_paths=[out_dir])
    isolation = {
        "schema_version": 141,
        "kind": "isolation_check_v141",
        "repo_snapshot_before": str(before_sig),
        "repo_snapshot_after": str(after_sig),
        "ok": bool(str(before_sig) == str(after_sig)),
    }
    _write_once_json(out_dir / "isolation_check_v141.json", isolation)
    if not isolation["ok"]:
        raise SystemExit("isolation_violation_v141")

    outputs_manifest = _outputs_manifest_v141(run_dir=out_dir)
    _write_once_json(out_dir / "outputs_manifest.json", outputs_manifest)

    return {
        "out_dir": str(out_dir),
        "arc_manifest_sig": str(manifest_obj.get("manifest_sig") or ""),
        "summary_sig": str(summary_obj.get("summary_sig") or ""),
        "outputs_manifest_sig": str(outputs_manifest.get("outputs_manifest_sig") or ""),
        "tasks_total": int(score_summary["tasks_total"]),
        "tasks_solved_by_k": score_summary["tasks_solved_by_k"],
        "failure_counts": {k: int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "determinism_ok": True,
        "isolation_ok": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc_root", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--max_programs", type=int, default=4000)
    ap.add_argument("--benchmark_profile", default="ARC_AGI1_PROFILE")
    ap.add_argument("--max_trials", type=int, default=None)
    ap.add_argument("--out_base", required=True)
    args = ap.parse_args()

    out_base = Path(str(args.out_base))
    try1 = Path(str(out_base) + "_try1")
    try2 = Path(str(out_base) + "_try2")

    _ensure_absent(try1)
    _ensure_absent(try2)

    max_trials = int(args.max_trials) if args.max_trials is not None else _benchmark_profile_default_trials(str(args.benchmark_profile))

    r1 = _run_once_v141(
        out_dir=try1,
        arc_root=str(args.arc_root),
        split=str(args.split) if args.split else None,
        limit=int(args.limit),
        seed=int(args.seed),
        max_depth=int(args.max_depth),
        max_programs=int(args.max_programs),
        max_trials=int(max_trials),
        benchmark_profile=str(args.benchmark_profile),
    )
    r2 = _run_once_v141(
        out_dir=try2,
        arc_root=str(args.arc_root),
        split=str(args.split) if args.split else None,
        limit=int(args.limit),
        seed=int(args.seed),
        max_depth=int(args.max_depth),
        max_programs=int(args.max_programs),
        max_trials=int(max_trials),
        benchmark_profile=str(args.benchmark_profile),
    )

    determinism_ok = bool(r1.get("summary_sig") == r2.get("summary_sig") and r1.get("outputs_manifest_sig") == r2.get("outputs_manifest_sig"))
    if not determinism_ok:
        raise SystemExit("determinism_mismatch_v141")

    final = {
        "ok": True,
        "determinism_ok": bool(determinism_ok),
        "try1": r1,
        "try2": r2,
    }
    print(_stable_json(final))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
