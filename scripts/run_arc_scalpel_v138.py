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


def _excluded_dir_parts_v138() -> set:
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


def _repo_snapshot_sha256_v138(*, root: Path, exclude_paths: Sequence[Path]) -> str:
    excluded = _excluded_dir_parts_v138()
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
    body = {"schema_version": 138, "kind": "repo_snapshot_v138", "files": rows}
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


def _build_report_markdown_v138(*, eval_obj: Dict[str, Any], backlog: Sequence[Dict[str, Any]]) -> str:
    total = int(eval_obj.get("tasks_total") or 0)
    solved = int(eval_obj.get("tasks_solved") or 0)
    unknown = int(eval_obj.get("tasks_unknown") or 0)
    failed = int(eval_obj.get("tasks_failed") or 0)
    failures = eval_obj.get("failure_counts")
    failures = failures if isinstance(failures, dict) else {}
    top = sorted(((str(k), int(failures[k])) for k in failures.keys()), key=lambda kv: (-int(kv[1]), str(kv[0])))[:15]

    lines: List[str] = []
    lines.append("# ARC_DIAG_REPORT_v138")
    lines.append("")
    lines.append("## Solve rate (scored vs test outputs; max_guesses=2)")
    lines.append(f"- tasks_total={total} solved={solved} unknown={unknown} failed={failed}")
    if total:
        lines.append(f"- solve_rate={solved/total:.3f}")
    lines.append("")
    lines.append("## Top failures (scoring failure_kind)")
    if not top:
        lines.append("- (none)")
    else:
        for k, n in top:
            lines.append(f"- {k}: {n}")
    lines.append("")
    lines.append("## Backlog (operator gaps) — propostas gerais")
    if not backlog:
        lines.append("- (none)")
    else:
        for item in backlog:
            lines.append(f"### {item['name']}")
            lines.append(f"- signature: `{item['signature']}`")
            lines.append(f"- invariants: {item['invariants']}")
            lines.append(f"- examples: {item['examples']}")
            lines.append(f"- covers: {item['covers']}")
            lines.append("")
    return "\n".join(lines)


def _derive_backlog_v138(*, failure_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if "MISSING_OPERATOR" in failure_counts:
        out.append(
            {
                "name": "general shape-changing operators (gap)",
                "signature": "(GRID)->GRID (scale/tile/resize or crop/paste families) with inverse propose",
                "invariants": "Determinístico; tipado; parâmetros inferidos de demonstrações.",
                "examples": "scale_cell, tile_repeat, reflect/rotate, crop/pad/paste combos.",
                "covers": "MISSING_OPERATOR clusters with shape_change_mixed / scale_integer.",
            }
        )
    if "TEST_OUTPUT_MISMATCH" in failure_counts:
        out.append(
            {
                "name": "reduce overfit / ambiguity (gap)",
                "signature": "scoring-only: prefer smaller hypothesis set; or add more invariants to prune",
                "invariants": "No test leakage; only train evidence; fail-closed ambiguity.",
                "examples": "stronger palette/shape reachability; object role induction from train pairs.",
                "covers": "cases where train-consistent program is not correct on test.",
            }
        )
    return out[:10]


def _build_outputs_manifest_v138(*, out_dir: Path) -> Dict[str, Any]:
    per_task_dir = out_dir / "per_task"
    per_task_files = [p for p in per_task_dir.glob("*.json") if p.is_file()]
    per_task_files.sort(key=lambda p: p.name)

    def rel(p: Path) -> str:
        return p.relative_to(out_dir).as_posix()

    files: List[Dict[str, Any]] = []
    fixed = [
        out_dir / "summary.json",
        out_dir / "smoke_summary.json",
        out_dir / "eval.json",
        out_dir / "per_task_manifest.jsonl",
        out_dir / "trace_candidates.jsonl",
        out_dir / "ARC_DIAG_REPORT_v138.md",
        out_dir / "isolation_check_v138.json",
        out_dir / "input" / "arc_manifest_v138.json",
        out_dir / "input" / "arc_tasks_canonical_v138.jsonl",
    ]
    for p in fixed:
        files.append({"path": rel(p), "sha256": _sha256_file(p)})
    for p in per_task_files:
        files.append({"path": rel(p), "sha256": _sha256_file(p)})

    body = {"schema_version": 138, "kind": "arc_outputs_manifest_v138", "files": files}
    from atos_core.act import canonical_json_dumps, sha256_hex

    body["manifest_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return body


def _score_one_test_case_v138(
    *, solver_res: Dict[str, Any], want_grid: Optional[Sequence[Sequence[int]]], max_guesses: int
) -> Dict[str, Any]:
    status = str(solver_res.get("status") or "")
    if want_grid is None:
        # Unscored: fall back to solver status only (used for internal sample/synth where test outputs are omitted).
        fk = ""
        fr = solver_res.get("failure_reason")
        if isinstance(fr, dict):
            fk = str(fr.get("kind") or "")
        return {
            "status": status if status in {"SOLVED", "UNKNOWN"} else "FAIL",
            "failure_kind": fk,
            "attempts_used": 1,
            "solver_status": status,
            "scored": False,
        }
    if status == "SOLVED":
        pred = solver_res.get("predicted_grid")
        ok = isinstance(pred, list) and _grid_equal(pred, want_grid)
        return {
            "status": "SOLVED" if ok else "FAIL",
            "failure_kind": "" if ok else "TEST_OUTPUT_MISMATCH",
            "attempts_used": 1,
            "solver_status": "SOLVED",
            "scored": True,
        }
    if status == "UNKNOWN":
        preds: List[Sequence[Sequence[int]]] = []
        for item in solver_res.get("predicted_grids", []) if isinstance(solver_res.get("predicted_grids"), list) else []:
            if isinstance(item, dict) and isinstance(item.get("grid"), list):
                preds.append(item.get("grid"))  # type: ignore[arg-type]
        ok = False
        used = 0
        for g in preds[: int(max_guesses)]:
            used += 1
            if _grid_equal(g, want_grid):
                ok = True
                break
        return {
            "status": "SOLVED" if ok else "UNKNOWN",
            "failure_kind": "" if ok else "AMBIGUOUS_RULE",
            "attempts_used": int(used) if used else int(max_guesses),
            "solver_status": "UNKNOWN",
            "scored": True,
        }
    fk = ""
    fr = solver_res.get("failure_reason")
    if isinstance(fr, dict):
        fk = str(fr.get("kind") or "")
    return {"status": "FAIL", "failure_kind": fk or "FAIL", "attempts_used": 1, "solver_status": "FAIL", "scored": True}


def _run_one(
    *,
    arc_root: str,
    split: str,
    limit: int,
    seed: int,
    out_dir: Path,
    max_guesses: int,
    max_depth: int,
    max_programs: int,
) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONPYCACHEPREFIX"] = str(out_dir / ".pycache")

    from atos_core.act import canonical_json_dumps, sha256_hex
    from atos_core.arc_loader_v138 import iter_canonical_tasks_v138, write_arc_canonical_jsonl_v138
    from atos_core.arc_solver_v138 import SolveConfigV138, solve_arc_task_v138

    repo_root = Path(__file__).resolve().parent.parent
    snap_before = _repo_snapshot_sha256_v138(root=repo_root, exclude_paths=[out_dir])

    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    canon_jsonl = input_dir / "arc_tasks_canonical_v138.jsonl"
    manifest_path = input_dir / "arc_manifest_v138.json"
    write_arc_canonical_jsonl_v138(
        arc_root=str(arc_root),
        split=str(split),
        limit=int(limit),
        out_jsonl=canon_jsonl,
        out_manifest=manifest_path,
    )

    per_task_dir = out_dir / "per_task"
    per_task_dir.mkdir(parents=True, exist_ok=False)

    per_task_manifest_path = out_dir / "per_task_manifest.jsonl"
    trace_candidates_path = out_dir / "trace_candidates.jsonl"
    _ensure_absent(per_task_manifest_path)
    _ensure_absent(trace_candidates_path)

    tasks_total = 0
    tasks_solved = 0
    tasks_unknown = 0
    tasks_failed = 0
    failure_counts: Dict[str, int] = {}

    per_task_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    cfg = SolveConfigV138(
        max_depth=int(max_depth),
        max_programs=int(max_programs),
        trace_program_limit=80,
        max_ambiguous_outputs=max(8, int(max_guesses)),
    )

    for task in iter_canonical_tasks_v138(str(canon_jsonl)):
        tasks_total += 1
        # Solve each test input independently; scoring uses test outputs only after solve.
        test_case_results: List[Dict[str, Any]] = []
        solver_results: List[Dict[str, Any]] = []
        for test_in, test_out in task.test_pairs:
            res = solve_arc_task_v138(train_pairs=list(task.train_pairs), test_in=test_in, config=cfg)
            solver_results.append(res)
            want_grid = [list(r) for r in test_out] if test_out is not None else None
            test_case_results.append(_score_one_test_case_v138(solver_res=res, want_grid=want_grid, max_guesses=int(max_guesses)))

        # Aggregate task status.
        if all(r["status"] == "SOLVED" for r in test_case_results):
            status = "SOLVED"
        elif any(r["status"] == "FAIL" for r in test_case_results):
            status = "FAIL"
        else:
            status = "UNKNOWN"

        if status == "SOLVED":
            tasks_solved += 1
        elif status == "UNKNOWN":
            tasks_unknown += 1
        else:
            tasks_failed += 1

        failure_kind = ""
        if status != "SOLVED":
            # deterministic priority among failure kinds
            kinds: List[str] = []
            for r in test_case_results:
                k = str(r.get("failure_kind") or "")
                if k:
                    kinds.append(k)
            pri = ["TEST_OUTPUT_MISMATCH", "SEARCH_BUDGET_EXCEEDED", "MISSING_OPERATOR", "AMBIGUOUS_RULE", "FAIL"]
            for k in pri:
                if k in kinds:
                    failure_kind = k
                    break
            if not failure_kind and kinds:
                failure_kind = sorted(kinds)[0]
        if status == "FAIL" and failure_kind:
            failure_counts[failure_kind] = int(failure_counts.get(failure_kind, 0)) + 1

        per_task_obj = {
            "kind": "arc_per_task_v138",
            "schema_version": 138,
            "task": task.to_dict(),
            "solver_results": solver_results,
            "scoring": {
                "schema_version": 138,
                "max_guesses": int(max_guesses),
                "status": str(status),
                "failure_kind": str(failure_kind),
                "test_case_results": test_case_results,
            },
        }
        task_path = per_task_dir / f"{_sanitize_task_id(task.task_id)}.json"
        _write_once_json(task_path, per_task_obj)

        row = {
            "schema_version": 138,
            "kind": "arc_per_task_manifest_row_v138",
            "task_id": str(task.task_id),
            "status": str(status),
            "failure_kind": str(failure_kind),
            "solver_statuses": [str(r.get("status") or "") for r in solver_results],
            "max_guesses": int(max_guesses),
        }
        per_task_rows.append(row)

        # Trace candidates (sample) for audit: include solver trace_programs only.
        for idx, res in enumerate(solver_results):
            trace = res.get("trace") if isinstance(res.get("trace"), dict) else {}
            for tp in trace.get("trace_programs", []) if isinstance(trace.get("trace_programs"), list) else []:
                if isinstance(tp, dict):
                    trace_rows.append(
                        {
                            "schema_version": 138,
                            "kind": "arc_trace_candidate_v138",
                            "task_id": str(task.task_id),
                            "test_index": int(idx),
                            "program_sig": str(tp.get("program_sig") or ""),
                            "cost_bits": int(tp.get("cost_bits") or 0),
                            "depth": int(tp.get("depth") or 0),
                            "ok_train": bool(tp.get("ok_train") or False),
                            "mismatch_kind": str(((tp.get("mismatch") or {}) if isinstance(tp.get("mismatch"), dict) else {}).get("kind") or ""),
                            "steps": tp.get("steps") if isinstance(tp.get("steps"), list) else [],
                        }
                    )

    # Write manifests
    with open(per_task_manifest_path, "x", encoding="utf-8") as f:
        for row in per_task_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    with open(trace_candidates_path, "x", encoding="utf-8") as f:
        for row in trace_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    eval_obj: Dict[str, Any] = {
        "schema_version": 138,
        "kind": "arc_eval_v138",
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "solve_rate": float(float(tasks_solved) / float(tasks_total) if tasks_total else 0.0),
        "failure_counts": {str(k): int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "max_guesses": int(max_guesses),
    }
    eval_obj["eval_sig"] = sha256_hex(canonical_json_dumps(eval_obj).encode("utf-8"))
    _write_once_json(out_dir / "eval.json", eval_obj)

    summary_obj: Dict[str, Any] = {
        "schema_version": 138,
        "kind": "arc_summary_v138",
        "arc_root": str(arc_root),
        "split": str(split),
        "limit": int(limit),
        "seed": int(seed),
        "max_guesses": int(max_guesses),
        "max_depth": int(max_depth),
        "max_programs": int(max_programs),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "failure_counts": {str(k): int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "eval_sig": str(eval_obj["eval_sig"]),
    }
    summary_obj["summary_sig"] = sha256_hex(canonical_json_dumps(summary_obj).encode("utf-8"))
    _write_once_json(out_dir / "summary.json", summary_obj)
    _write_once_json(out_dir / "smoke_summary.json", {"schema_version": 138, "kind": "arc_smoke_summary_v138", "summary_sha256": summary_obj["summary_sig"]})

    backlog = _derive_backlog_v138(failure_counts=failure_counts)
    report_md = _build_report_markdown_v138(eval_obj=eval_obj, backlog=backlog)
    _write_text_x(out_dir / "ARC_DIAG_REPORT_v138.md", report_md)

    snap_after = _repo_snapshot_sha256_v138(root=repo_root, exclude_paths=[out_dir])
    isolation = {
        "schema_version": 138,
        "kind": "isolation_check_v138",
        "repo_root": str(repo_root),
        "snapshot_before": str(snap_before),
        "snapshot_after": str(snap_after),
        "ok": bool(snap_before == snap_after),
    }
    _write_once_json(out_dir / "isolation_check_v138.json", isolation)
    if not bool(isolation["ok"]):
        raise SystemExit("isolation_failed")

    outputs_manifest = _build_outputs_manifest_v138(out_dir=out_dir)
    _write_once_json(out_dir / "outputs_manifest.json", outputs_manifest)

    return {
        "schema_version": 138,
        "kind": "arc_run_result_v138",
        "out_dir": str(out_dir),
        "summary_sha256": str(summary_obj["summary_sig"]),
        "outputs_manifest_sig": str(outputs_manifest["manifest_sig"]),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "failure_counts": {str(k): int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "isolation_ok": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc_root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--max_guesses", type=int, default=2)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--max_programs", type=int, default=4000)
    args = ap.parse_args(argv)

    out_base = Path(str(args.out_base))
    out_try1 = Path(str(out_base) + "_try1")
    out_try2 = Path(str(out_base) + "_try2")

    r1 = _run_one(
        arc_root=str(args.arc_root),
        split=str(args.split),
        limit=int(args.limit),
        seed=int(args.seed),
        out_dir=out_try1,
        max_guesses=int(args.max_guesses),
        max_depth=int(args.max_depth),
        max_programs=int(args.max_programs),
    )
    r2 = _run_one(
        arc_root=str(args.arc_root),
        split=str(args.split),
        limit=int(args.limit),
        seed=int(args.seed),
        out_dir=out_try2,
        max_guesses=int(args.max_guesses),
        max_depth=int(args.max_depth),
        max_programs=int(args.max_programs),
    )

    determinism_ok = str(r1.get("summary_sha256")) == str(r2.get("summary_sha256")) and str(r1.get("outputs_manifest_sig")) == str(
        r2.get("outputs_manifest_sig")
    )
    if not determinism_ok:
        raise SystemExit("determinism_failed")

    out = {"ok": True, "determinism_ok": True, "try1": r1, "try2": r2}
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
