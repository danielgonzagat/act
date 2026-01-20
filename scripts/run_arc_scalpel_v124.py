#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Prevent any bytecode writes outside run dirs.
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


def _excluded_dir_parts_v124() -> set:
    return {".git", "__pycache__", ".pycache", "results"}


def _repo_snapshot_sha256_v124(*, root: Path, exclude_paths: Sequence[Path]) -> str:
    excluded_parts = _excluded_dir_parts_v124()
    excludes = [p.resolve() for p in exclude_paths]
    rows: List[Dict[str, Any]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in excluded_parts for part in p.parts):
            continue
        rp = p.resolve()
        if any(str(rp).startswith(str(ex)) for ex in excludes):
            continue
        rel = p.relative_to(root).as_posix()
        rows.append({"path": str(rel), "sha256": _sha256_file(p)})
    rows.sort(key=lambda r: str(r["path"]))
    body = {"schema_version": 124, "kind": "repo_snapshot_v124", "files": rows}
    from atos_core.act import canonical_json_dumps, sha256_hex

    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _sanitize_task_id(task_id: str) -> str:
    s = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(task_id)])
    return s or "task"


def _build_report_markdown_v124(*, eval_obj: Dict[str, Any]) -> str:
    total = int(eval_obj.get("tasks_total") or 0)
    solved = int(eval_obj.get("tasks_solved") or 0)
    unknown = int(eval_obj.get("tasks_unknown") or 0)
    failed = int(eval_obj.get("tasks_failed") or 0)
    failures = eval_obj.get("failure_counts")
    failures = failures if isinstance(failures, dict) else {}
    top = sorted(((str(k), int(failures[k])) for k in failures.keys()), key=lambda kv: (-int(kv[1]), str(kv[0])))[:10]

    lines: List[str] = []
    lines.append("# ARC_DIAG_REPORT_v124")
    lines.append("")
    lines.append("## Solve rate")
    lines.append(f"- tasks_total={total} solved={solved} unknown={unknown} failed={failed}")
    if total:
        lines.append(f"- solve_rate={solved/total:.3f}")
    lines.append("")
    lines.append("## Top failures")
    if not top:
        lines.append("- (none)")
    else:
        for k, n in top:
            lines.append(f"- {k}: {n}")
    lines.append("")
    lines.append("## Backlog (operator gaps)")
    lines.append("- Para cada falha recorrente, priorizar operador geral (sem hacks por task_id) com: assinatura, invariantes e exemplos mínimos.")
    lines.append("- Se a falha dominante for `shape_transform_needed`, considerar operadores gerais de crop/pad/bbox/resize.")
    lines.append("- Se a falha dominante for `color_transform_needed`, considerar operadores gerais de mapeamento de cores e pintura por máscara/componente.")
    lines.append("- Se a falha dominante for `AMBIGUOUS_RULE`, adicionar critério determinístico de desambiguação (ex.: preferir programa com menor mudança estrutural) ou manter FAIL-CLOSED.")
    lines.append("")
    return "\n".join(lines)


def _build_outputs_manifest_v124(*, out_dir: Path) -> Dict[str, Any]:
    per_task_dir = out_dir / "per_task"
    per_task_files = [p for p in per_task_dir.glob("*.json") if p.is_file()]
    per_task_files.sort(key=lambda p: p.name)

    def rel(p: Path) -> str:
        return p.relative_to(out_dir).as_posix()

    entries: List[Dict[str, Any]] = []
    for fixed in [
        out_dir / "summary.json",
        out_dir / "eval.json",
        out_dir / "isolation_check_v124.json",
        out_dir / "arc_task_events_v124.jsonl",
        out_dir / "ARC_DIAG_REPORT_v124.md",
        out_dir / "input" / "arc_manifest_v124.json",
        out_dir / "input" / "arc_tasks_canonical_v124.jsonl",
    ]:
        entries.append({"path": rel(fixed), "sha256": _sha256_file(fixed)})
    for p in per_task_files:
        entries.append({"path": rel(p), "sha256": _sha256_file(p)})

    body = {"schema_version": 124, "kind": "arc_outputs_manifest_v124", "files": entries}
    from atos_core.act import canonical_json_dumps, sha256_hex

    body["manifest_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return body


def _write_task_events_jsonl_v124(*, out_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    _ensure_absent(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from atos_core.act import canonical_json_dumps

    with open(out_path, "x", encoding="utf-8") as f:
        for r in rows:
            f.write(canonical_json_dumps(r))
            f.write("\n")


def _run_one(*, arc_root: str, split: str, limit: int, seed: int, out_dir: Path) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONPYCACHEPREFIX"] = str(out_dir / ".pycache")

    from atos_core.act import canonical_json_dumps, sha256_hex
    from atos_core.arc_loader_v124 import iter_canonical_tasks_v124, write_arc_canonical_jsonl_v124
    from atos_core.arc_solver_v124 import diagnose_missing_operator_v124, solve_arc_task_v124

    repo_root = Path(__file__).resolve().parent.parent
    snap_before = _repo_snapshot_sha256_v124(root=repo_root, exclude_paths=[out_dir])

    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    canon_jsonl = input_dir / "arc_tasks_canonical_v124.jsonl"
    manifest = write_arc_canonical_jsonl_v124(
        arc_root=str(arc_root), out_jsonl_path=str(canon_jsonl), limit=int(limit), split=str(split)
    )
    manifest_path = input_dir / "arc_manifest_v124.json"
    _write_once_json(manifest_path, manifest)

    per_task_dir = out_dir / "per_task"
    per_task_dir.mkdir(parents=True, exist_ok=False)

    task_events: List[Dict[str, Any]] = []
    prev_hash = ""

    tasks_total = 0
    tasks_solved = 0
    tasks_unknown = 0
    tasks_failed = 0
    failure_counts: Dict[str, int] = {}

    for task in iter_canonical_tasks_v124(str(canon_jsonl)):
        tasks_total += 1
        solve = solve_arc_task_v124(train_pairs=list(task.train_pairs), test_in=task.test_in)
        status = str(solve.get("status") or "FAIL")
        if status == "SOLVED":
            tasks_solved += 1
        elif status == "UNKNOWN":
            tasks_unknown += 1
        else:
            tasks_failed += 1

        failure_kind = ""
        fr = solve.get("failure_reason")
        if isinstance(fr, dict):
            failure_kind = str(fr.get("kind") or "")
        if status != "SOLVED":
            failure_counts[failure_kind or "unknown_failure_kind"] = int(
                failure_counts.get(failure_kind or "unknown_failure_kind", 0)
            ) + 1

        diag = diagnose_missing_operator_v124(train_pairs=list(task.train_pairs))
        per = {
            "schema_version": 124,
            "kind": "arc_task_result_v124",
            "task_id": str(task.task_id),
            "input": task.to_dict(),
            "result": dict(solve),
            "diagnostic": dict(diag),
        }
        per["per_task_sig"] = sha256_hex(canonical_json_dumps(per).encode("utf-8"))

        task_path = per_task_dir / (_sanitize_task_id(task.task_id) + ".json")
        _write_once_json(task_path, per)

        program_sig = ""
        prog = solve.get("program")
        if isinstance(prog, dict):
            program_sig = sha256_hex(canonical_json_dumps(prog).encode("utf-8"))

        ev_body = {
            "schema_version": 124,
            "kind": "arc_task_event_v124",
            "task_id": str(task.task_id),
            "status": str(status),
            "program_sig": str(program_sig),
            "predicted_grid_hash": str(solve.get("predicted_grid_hash") or ""),
            "per_task_sha256": _sha256_file(task_path),
        }
        entry_hash = sha256_hex(canonical_json_dumps({"prev_hash": prev_hash, "body": ev_body}).encode("utf-8"))
        row = {"prev_hash": str(prev_hash), "body": ev_body, "entry_hash": str(entry_hash)}
        task_events.append(row)
        prev_hash = str(entry_hash)

    task_events_path = out_dir / "arc_task_events_v124.jsonl"
    _write_task_events_jsonl_v124(out_path=task_events_path, rows=task_events)
    chain_hash = sha256_hex(canonical_json_dumps([r["entry_hash"] for r in task_events]).encode("utf-8"))

    eval_obj = {
        "schema_version": 124,
        "kind": "arc_eval_v124",
        "seed": int(seed),
        "arc_root": str(arc_root),
        "split": str(split),
        "limit": int(limit),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "failure_counts": {str(k): int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "chain": {"arc_task_events_sha256": _sha256_file(task_events_path), "arc_task_chain_hash_v124": str(chain_hash)},
        "sha256": {"arc_canonical_jsonl": _sha256_file(canon_jsonl), "arc_manifest_json": _sha256_file(manifest_path)},
    }
    eval_obj["eval_sig"] = sha256_hex(canonical_json_dumps(eval_obj).encode("utf-8"))
    eval_path = out_dir / "eval.json"
    _write_once_json(eval_path, eval_obj)

    report_md = _build_report_markdown_v124(eval_obj=dict(eval_obj)) + "\n"
    report_path = out_dir / "ARC_DIAG_REPORT_v124.md"
    _write_text_x(report_path, report_md)
    report_sha256 = _sha256_file(report_path)
    report_sig = sha256_hex(
        canonical_json_dumps({"schema_version": 124, "kind": "arc_diag_report_sig_v124", "sha256": report_sha256}).encode("utf-8")
    )

    summary = {
        "schema_version": 124,
        "kind": "arc_summary_v124",
        "seed": int(seed),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "solve_rate": float(tasks_solved / tasks_total) if tasks_total else 0.0,
        "eval_sha256": _sha256_file(eval_path),
        "arc_task_chain_hash_v124": str(chain_hash),
        "report_sha256": str(report_sha256),
        "report_sig": str(report_sig),
    }
    summary["summary_sig"] = sha256_hex(canonical_json_dumps(summary).encode("utf-8"))
    summary_path = out_dir / "summary.json"
    _write_once_json(summary_path, summary)

    snap_after = _repo_snapshot_sha256_v124(root=repo_root, exclude_paths=[out_dir])
    isolation = {
        "schema_version": 124,
        "kind": "arc_isolation_check_v124",
        "excluded_dir_parts": sorted(list(_excluded_dir_parts_v124())),
        "repo_snapshot_before": str(snap_before),
        "repo_snapshot_after": str(snap_after),
        "ok": bool(snap_before == snap_after),
    }
    isolation["isolation_sig"] = sha256_hex(canonical_json_dumps(isolation).encode("utf-8"))
    _write_once_json(out_dir / "isolation_check_v124.json", isolation)
    if not isolation["ok"]:
        raise SystemExit("isolation_failed:repo_snapshot_changed")

    outputs_manifest = _build_outputs_manifest_v124(out_dir=out_dir)
    _write_once_json(out_dir / "outputs_manifest.json", outputs_manifest)

    return {"eval": eval_obj, "summary": summary, "outputs_manifest": outputs_manifest}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc_root", required=True)
    ap.add_argument("--split", default="")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_base", required=True, help="Base path; script writes _try1 and _try2 and fails if they exist.")
    args = ap.parse_args()

    out_base = Path(str(args.out_base))
    out1 = Path(str(out_base) + "_try1")
    out2 = Path(str(out_base) + "_try2")
    _ensure_absent(out1)
    _ensure_absent(out2)

    r1 = _run_one(arc_root=str(args.arc_root), split=str(args.split), limit=int(args.limit), seed=int(args.seed), out_dir=out1)
    r2 = _run_one(arc_root=str(args.arc_root), split=str(args.split), limit=int(args.limit), seed=int(args.seed), out_dir=out2)

    from atos_core.act import canonical_json_dumps

    if canonical_json_dumps(r1["outputs_manifest"]) != canonical_json_dumps(r2["outputs_manifest"]):
        raise SystemExit("determinism_failed:outputs_manifest")

    out = {
        "ok": True,
        "determinism_ok": True,
        "try1_dir": str(out1),
        "try2_dir": str(out2),
        "summary_sha256": _sha256_file(out1 / "summary.json"),
        "eval_sha256": _sha256_file(out1 / "eval.json"),
        "report_try1": str(out1 / "ARC_DIAG_REPORT_v124.md"),
        "report_try2": str(out2 / "ARC_DIAG_REPORT_v124.md"),
        "outputs_manifest_try1": str(out1 / "outputs_manifest.json"),
        "outputs_manifest_try2": str(out2 / "outputs_manifest.json"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

