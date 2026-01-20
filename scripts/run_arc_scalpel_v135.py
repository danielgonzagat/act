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


def _excluded_dir_parts_v135() -> set:
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


def _repo_snapshot_sha256_v135(*, root: Path, exclude_paths: Sequence[Path]) -> str:
    excluded = _excluded_dir_parts_v135()
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
    body = {"schema_version": 135, "kind": "repo_snapshot_v135", "files": rows}
    from atos_core.act import canonical_json_dumps, sha256_hex

    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _sanitize_task_id(task_id: str) -> str:
    s = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(task_id)])
    return s or "task"


def _build_report_markdown_v135(*, eval_obj: Dict[str, Any], backlog: Sequence[Dict[str, Any]]) -> str:
    total = int(eval_obj.get("tasks_total") or 0)
    solved = int(eval_obj.get("tasks_solved") or 0)
    unknown = int(eval_obj.get("tasks_unknown") or 0)
    failed = int(eval_obj.get("tasks_failed") or 0)
    failures = eval_obj.get("failure_counts")
    failures = failures if isinstance(failures, dict) else {}
    top = sorted(((str(k), int(failures[k])) for k in failures.keys()), key=lambda kv: (-int(kv[1]), str(kv[0])))[:15]

    lines: List[str] = []
    lines.append("# ARC_DIAG_REPORT_v135")
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


def _derive_backlog_v135(*, failure_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    out.append(
        {
            "name": "overlay_self_translate(dx,dy,pad)",
            "signature": "GRID -> GRID",
            "invariants": "Determinístico; shape preservada; overlay não escreve onde shift==pad.",
            "examples": "União do grid original com cópia transladada (sem memória extra).",
            "covers": "MISSING_OPERATOR de união/overlay via deslocamento.",
        }
    )
    if "MISSING_OPERATOR" in failure_counts:
        out.append(
            {
                "name": "mask/region fill (floodfill / interior fill)",
                "signature": "(GRID[, MASK|SEED]) -> MASK or GRID",
                "invariants": "Determinístico; 4-neigh; sem heurística por task.",
                "examples": "Pintar interior de contorno mantendo borda.",
                "covers": "Tasks de interior fill (MISSING_OPERATOR).",
            }
        )
    return out[:10]


def _build_outputs_manifest_v135(*, out_dir: Path) -> Dict[str, Any]:
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
        out_dir / "ARC_DIAG_REPORT_v135.md",
        out_dir / "isolation_check_v135.json",
        out_dir / "input" / "arc_manifest_v135.json",
        out_dir / "input" / "arc_tasks_canonical_v135.jsonl",
    ]
    for p in fixed:
        files.append({"path": rel(p), "sha256": _sha256_file(p)})
    for p in per_task_files:
        files.append({"path": rel(p), "sha256": _sha256_file(p)})

    body = {"schema_version": 135, "kind": "arc_outputs_manifest_v135", "files": files}
    from atos_core.act import canonical_json_dumps, sha256_hex

    body["manifest_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return body


def _run_one(*, arc_root: str, split: str, limit: int, seed: int, out_dir: Path) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONPYCACHEPREFIX"] = str(out_dir / ".pycache")

    from atos_core.act import canonical_json_dumps, sha256_hex
    from atos_core.arc_loader_v135 import iter_canonical_tasks_v135, write_arc_canonical_jsonl_v135
    from atos_core.arc_solver_v135 import SolveConfigV135, solve_arc_task_v135

    repo_root = Path(__file__).resolve().parent.parent
    snap_before = _repo_snapshot_sha256_v135(root=repo_root, exclude_paths=[out_dir])

    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    canon_jsonl = input_dir / "arc_tasks_canonical_v135.jsonl"
    manifest_path = input_dir / "arc_manifest_v135.json"
    write_arc_canonical_jsonl_v135(
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

    cfg = SolveConfigV135(max_depth=4, max_programs=4000, trace_program_limit=80)

    for task in iter_canonical_tasks_v135(str(canon_jsonl)):
        tasks_total += 1
        res = solve_arc_task_v135(train_pairs=list(task.train_pairs), test_in=task.test_in, config=cfg)
        status = str(res.get("status") or "")
        failure_kind = ""
        fr = res.get("failure_reason")
        if isinstance(fr, dict):
            failure_kind = str(fr.get("kind") or "")
        if status == "SOLVED":
            tasks_solved += 1
        elif status == "UNKNOWN":
            tasks_unknown += 1
            failure_counts[failure_kind or "UNKNOWN"] = int(failure_counts.get(failure_kind or "UNKNOWN", 0)) + 1
        else:
            tasks_failed += 1
            failure_counts[failure_kind or "FAIL"] = int(failure_counts.get(failure_kind or "FAIL", 0)) + 1

        task_id = str(task.task_id)
        safe_id = _sanitize_task_id(task_id)
        per_task_path = per_task_dir / f"{safe_id}.json"
        per_task_obj = {
            "schema_version": 135,
            "kind": "arc_per_task_v135",
            "task": task.to_dict(),
            "result": res,
        }
        _write_once_json(per_task_path, per_task_obj)

        per_task_rows.append(
            {
                "task_id": str(task_id),
                "status": str(status),
                "failure_kind": str(failure_kind),
                "program_sig": str(res.get("program_sig") or ""),
                "program_cost_bits": int(res.get("program_cost_bits") or 0),
            }
        )

        tr = res.get("trace")
        if isinstance(tr, dict):
            for row in tr.get("trace_programs") or []:
                if isinstance(row, dict):
                    trace_rows.append({"task_id": str(task_id), "row": row})

    with open(per_task_manifest_path, "x", encoding="utf-8") as f:
        for r in per_task_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    with open(trace_candidates_path, "x", encoding="utf-8") as f:
        for r in trace_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    eval_obj: Dict[str, Any] = {
        "schema_version": 135,
        "kind": "arc_eval_v135",
        "seed": int(seed),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "failure_counts": {str(k): int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "sha256": {
            "arc_manifest_json": _sha256_file(manifest_path),
            "arc_canonical_jsonl": _sha256_file(canon_jsonl),
            "per_task_manifest_jsonl": _sha256_file(per_task_manifest_path),
            "trace_candidates_jsonl": _sha256_file(trace_candidates_path),
        },
    }
    eval_obj["eval_sig"] = sha256_hex(canonical_json_dumps(eval_obj).encode("utf-8"))
    _write_once_json(out_dir / "eval.json", eval_obj)

    solved_rate = float(tasks_solved) / float(tasks_total) if tasks_total else 0.0
    summary_obj: Dict[str, Any] = {
        "schema_version": 135,
        "kind": "arc_summary_v135",
        "seed": int(seed),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "solve_rate": float(solved_rate),
        "failure_counts": {str(k): int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "eval_sha256": _sha256_file(out_dir / "eval.json"),
        "arc_root": str(arc_root),
        "split": str(split),
    }
    summary_obj["summary_sha256"] = sha256_hex(canonical_json_dumps(summary_obj).encode("utf-8"))
    _write_once_json(out_dir / "summary.json", summary_obj)

    smoke_summary = {
        "schema_version": 135,
        "kind": "arc_smoke_summary_v135",
        "summary_sha256": str(summary_obj["summary_sha256"]),
        "eval_sha256": str(summary_obj["eval_sha256"]),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "solve_rate": float(solved_rate),
    }
    smoke_summary["smoke_sig"] = sha256_hex(canonical_json_dumps(smoke_summary).encode("utf-8"))
    _write_once_json(out_dir / "smoke_summary.json", smoke_summary)

    backlog = _derive_backlog_v135(failure_counts=failure_counts)
    report = _build_report_markdown_v135(eval_obj=eval_obj, backlog=backlog)
    _write_text_x(out_dir / "ARC_DIAG_REPORT_v135.md", report + "\n")

    snap_after = _repo_snapshot_sha256_v135(root=repo_root, exclude_paths=[out_dir])
    iso_obj = {
        "schema_version": 135,
        "kind": "isolation_check_v135",
        "ok": bool(snap_before == snap_after),
        "repo_snapshot_before": str(snap_before),
        "repo_snapshot_after": str(snap_after),
    }
    _write_once_json(out_dir / "isolation_check_v135.json", iso_obj)

    out_manifest = _build_outputs_manifest_v135(out_dir=out_dir)
    _write_once_json(out_dir / "outputs_manifest.json", out_manifest)

    return {
        "out_dir": str(out_dir),
        "summary_sha256": str(summary_obj["summary_sha256"]),
        "outputs_manifest_sig": str(out_manifest.get("manifest_sig") or ""),
        "isolation_ok": bool(iso_obj["ok"]),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "solve_rate": float(solved_rate),
        "eval_sha256": str(summary_obj["eval_sha256"]),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc_root", required=True)
    ap.add_argument("--split", default="sample")
    ap.add_argument("--limit", type=int, default=999999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_base", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)

    out_base = Path(str(args.out_base)).resolve()
    out_try1 = Path(str(out_base) + "_try1")
    out_try2 = Path(str(out_base) + "_try2")
    _ensure_absent(out_try1)
    _ensure_absent(out_try2)

    try1 = _run_one(arc_root=str(args.arc_root), split=str(args.split), limit=int(args.limit), seed=int(args.seed), out_dir=out_try1)
    try2 = _run_one(arc_root=str(args.arc_root), split=str(args.split), limit=int(args.limit), seed=int(args.seed), out_dir=out_try2)

    determinism_ok = bool(try1["summary_sha256"] == try2["summary_sha256"]) and bool(try1["outputs_manifest_sig"] == try2["outputs_manifest_sig"])
    ok = determinism_ok and bool(try1["isolation_ok"]) and bool(try2["isolation_ok"])

    print(
        json.dumps(
            {
                "schema_version": 135,
                "kind": "arc_scalpel_run_v135",
                "ok": bool(ok),
                "determinism_ok": bool(determinism_ok),
                "arc_root": str(args.arc_root),
                "split": str(args.split),
                "limit": int(args.limit),
                "seed": int(args.seed),
                "try1": try1,
                "try2": try2,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

