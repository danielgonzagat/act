#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


def _excluded_dir_parts_v136() -> set:
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


def _repo_snapshot_sha256_v136(*, root: Path, exclude_paths: Sequence[Path]) -> str:
    excluded = _excluded_dir_parts_v136()
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
    body = {"schema_version": 136, "kind": "repo_snapshot_v136", "files": rows}
    from atos_core.act import canonical_json_dumps, sha256_hex

    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _sanitize_task_id(task_id: str) -> str:
    s = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(task_id)])
    return s or "task"


def _build_report_markdown_v136(*, eval_obj: Dict[str, Any], backlog: Sequence[Dict[str, Any]]) -> str:
    total = int(eval_obj.get("tasks_total") or 0)
    solved = int(eval_obj.get("tasks_solved") or 0)
    unknown = int(eval_obj.get("tasks_unknown") or 0)
    failed = int(eval_obj.get("tasks_failed") or 0)
    failures = eval_obj.get("failure_counts")
    failures = failures if isinstance(failures, dict) else {}
    top = sorted(((str(k), int(failures[k])) for k in failures.keys()), key=lambda kv: (-int(kv[1]), str(kv[0])))[:15]

    lines: List[str] = []
    lines.append("# ARC_DIAG_REPORT_v136")
    lines.append("")
    lines.append("## Solve rate")
    lines.append(f"- tasks_total={total} solved={solved} unknown={unknown} failed={failed}")
    if total:
        lines.append(f"- solve_rate={solved/total:.3f}")
    lines.append("")
    lines.append("## Top failures (failure_reason.kind)")
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


def _derive_backlog_v136(*, failure_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if "MISSING_OPERATOR" in failure_counts:
        out.append(
            {
                "name": "conditional paste / pattern replication (operator gap)",
                "signature": "(GRID, MASK|RULE) -> GRID",
                "invariants": "Determinístico; sem task-specific branching; compilável em passos.",
                "examples": "Para cada célula, escolher entre patch A e patch B conforme cor/condição.",
                "covers": "MISSING_OPERATOR em tasks de expansão condicional (ex: bloco 3x3 por célula).",
            }
        )
    return out[:10]


def _build_outputs_manifest_v136(*, out_dir: Path) -> Dict[str, Any]:
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
        out_dir / "ARC_DIAG_REPORT_v136.md",
        out_dir / "isolation_check_v136.json",
        out_dir / "input" / "arc_manifest_v136.json",
        out_dir / "input" / "arc_tasks_canonical_v136.jsonl",
    ]
    for p in fixed:
        files.append({"path": rel(p), "sha256": _sha256_file(p)})
    for p in per_task_files:
        files.append({"path": rel(p), "sha256": _sha256_file(p)})

    body = {"schema_version": 136, "kind": "arc_outputs_manifest_v136", "files": files}
    from atos_core.act import canonical_json_dumps, sha256_hex

    body["manifest_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return body


def _run_one(*, arc_root: str, split: str, limit: int, seed: int, out_dir: Path) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONPYCACHEPREFIX"] = str(out_dir / ".pycache")

    from atos_core.act import canonical_json_dumps, sha256_hex
    from atos_core.arc_loader_v136 import iter_canonical_tasks_v136, write_arc_canonical_jsonl_v136
    from atos_core.arc_solver_v136 import SolveConfigV136, solve_arc_task_v136

    repo_root = Path(__file__).resolve().parent.parent
    snap_before = _repo_snapshot_sha256_v136(root=repo_root, exclude_paths=[out_dir])

    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    canon_jsonl = input_dir / "arc_tasks_canonical_v136.jsonl"
    manifest_path = input_dir / "arc_manifest_v136.json"
    write_arc_canonical_jsonl_v136(
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

    cfg = SolveConfigV136(max_depth=4, max_programs=4000, trace_program_limit=80)

    for task in iter_canonical_tasks_v136(str(canon_jsonl)):
        tasks_total += 1
        res = solve_arc_task_v136(train_pairs=list(task.train_pairs), test_in=task.test_in, config=cfg)
        status = str(res.get("status") or "")
        failure_kind = ""
        fr = res.get("failure_reason")
        if isinstance(fr, dict):
            failure_kind = str(fr.get("kind") or "")
        if status == "SOLVED":
            tasks_solved += 1
        elif status == "UNKNOWN":
            tasks_unknown += 1
        else:
            tasks_failed += 1
            if failure_kind:
                failure_counts[failure_kind] = int(failure_counts.get(failure_kind, 0)) + 1

        per_task_obj = {"kind": "arc_per_task_v136", "result": res, "task": task.to_dict()}
        task_path = per_task_dir / f"{_sanitize_task_id(task.task_id)}.json"
        _write_once_json(task_path, per_task_obj)

        row = {
            "schema_version": 136,
            "kind": "arc_per_task_manifest_row_v136",
            "task_id": str(task.task_id),
            "status": str(status),
            "failure_kind": str(failure_kind),
            "program_sig": str(res.get("program_sig") or ""),
            "program_cost_bits": int(res.get("program_cost_bits") or 0),
            "predicted_grid_hash": str(res.get("predicted_grid_hash") or ""),
        }
        per_task_rows.append(row)

        trace = res.get("trace") if isinstance(res.get("trace"), dict) else {}
        for tp in trace.get("trace_programs", []) if isinstance(trace.get("trace_programs"), list) else []:
            if isinstance(tp, dict):
                trace_rows.append(
                    {
                        "schema_version": 136,
                        "kind": "arc_trace_candidate_v136",
                        "task_id": str(task.task_id),
                        "program_sig": str(tp.get("program_sig") or ""),
                        "cost_bits": int(tp.get("cost_bits") or 0),
                        "depth": int(tp.get("depth") or 0),
                        "ok_train": bool(tp.get("ok_train")),
                        "mismatch": tp.get("mismatch"),
                    }
                )

    with open(per_task_manifest_path, "x", encoding="utf-8") as f:
        for row in per_task_rows:
            f.write(canonical_json_dumps(row) + "\n")
    with open(trace_candidates_path, "x", encoding="utf-8") as f:
        for row in trace_rows:
            f.write(canonical_json_dumps(row) + "\n")

    eval_obj: Dict[str, Any] = {
        "schema_version": 136,
        "kind": "arc_eval_v136",
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "failure_counts": {str(k): int(failure_counts[k]) for k in sorted(failure_counts.keys())},
    }
    eval_obj["eval_sig"] = sha256_hex(canonical_json_dumps(eval_obj).encode("utf-8"))
    _write_once_json(out_dir / "eval.json", eval_obj)

    summary_obj: Dict[str, Any] = {
        "schema_version": 136,
        "kind": "arc_summary_v136",
        "arc_root": str(Path(str(arc_root)).resolve()),
        "split": str(split),
        "limit": int(limit),
        "seed": int(seed),
        "eval_sig": str(eval_obj["eval_sig"]),
    }
    summary_obj["summary_sig"] = sha256_hex(canonical_json_dumps(summary_obj).encode("utf-8"))
    _write_once_json(out_dir / "summary.json", summary_obj)

    smoke_summary = {"schema_version": 136, "kind": "arc_smoke_summary_v136", "summary_sha256": str(summary_obj["summary_sig"])}
    _write_once_json(out_dir / "smoke_summary.json", smoke_summary)

    backlog = _derive_backlog_v136(failure_counts={str(k): int(failure_counts[k]) for k in failure_counts.keys()})
    report_text = _build_report_markdown_v136(eval_obj=eval_obj, backlog=backlog)
    report_path = out_dir / "ARC_DIAG_REPORT_v136.md"
    _write_text_x(report_path, report_text + "\n")

    snap_after = _repo_snapshot_sha256_v136(root=repo_root, exclude_paths=[out_dir])
    isolation = {
        "schema_version": 136,
        "kind": "isolation_check_v136",
        "repo_root": str(repo_root),
        "snapshot_before": str(snap_before),
        "snapshot_after": str(snap_after),
        "ok": bool(snap_before == snap_after),
    }
    _write_once_json(out_dir / "isolation_check_v136.json", isolation)
    if not bool(isolation["ok"]):
        raise SystemExit("isolation_failed")

    outputs_manifest = _build_outputs_manifest_v136(out_dir=out_dir)
    _write_once_json(out_dir / "outputs_manifest.json", outputs_manifest)

    return {
        "schema_version": 136,
        "kind": "arc_run_result_v136",
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
    args = ap.parse_args(argv)

    out_base = Path(str(args.out_base))
    out_try1 = Path(str(out_base) + "_try1")
    out_try2 = Path(str(out_base) + "_try2")

    r1 = _run_one(arc_root=str(args.arc_root), split=str(args.split), limit=int(args.limit), seed=int(args.seed), out_dir=out_try1)
    r2 = _run_one(arc_root=str(args.arc_root), split=str(args.split), limit=int(args.limit), seed=int(args.seed), out_dir=out_try2)

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
