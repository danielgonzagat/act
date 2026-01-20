#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def _excluded_dir_parts_v129() -> set:
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


def _repo_snapshot_sha256_v129(*, root: Path) -> str:
    excluded = _excluded_dir_parts_v129()
    rows: List[Dict[str, Any]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in excluded for part in p.parts):
            continue
        rel = p.relative_to(root).as_posix()
        rows.append({"path": str(rel), "sha256": _sha256_file(p)})
    rows.sort(key=lambda r: str(r["path"]))
    body = {"schema_version": 129, "kind": "repo_snapshot_v129", "files": rows}
    from atos_core.act import canonical_json_dumps, sha256_hex

    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _sanitize_task_id(task_id: str) -> str:
    s = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(task_id)])
    return s or "task"


def _run_one(*, arc_root: str, split: str, limit: int, seed: int, out_dir: Path) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONPYCACHEPREFIX"] = str(out_dir / ".pycache")

    from atos_core.act import canonical_json_dumps, sha256_hex
    from atos_core.arc_loader_v129 import iter_canonical_tasks_v129, write_arc_canonical_jsonl_v129
    from atos_core.arc_solver_v129 import solve_arc_task_v129

    repo_root = Path(__file__).resolve().parent.parent
    snap_before = _repo_snapshot_sha256_v129(root=repo_root)

    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    canon_jsonl = input_dir / "arc_tasks_canonical_v129.jsonl"
    manifest = write_arc_canonical_jsonl_v129(
        arc_root=str(arc_root), out_jsonl_path=str(canon_jsonl), limit=int(limit), split=str(split)
    )
    manifest_path = input_dir / "arc_manifest_v129.json"
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

    for task in iter_canonical_tasks_v129(str(canon_jsonl)):
        tasks_total += 1
        solve = solve_arc_task_v129(train_pairs=list(task.train_pairs), test_in=task.test_in)
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
            k = failure_kind or "unknown_failure_kind"
            failure_counts[k] = int(failure_counts.get(k, 0)) + 1

        per = {
            "schema_version": 129,
            "kind": "arc_task_result_v129",
            "task_id": str(task.task_id),
            "input": task.to_dict(),
            "result": dict(solve),
        }
        per["per_task_sig"] = sha256_hex(canonical_json_dumps(per).encode("utf-8"))
        task_path = per_task_dir / (_sanitize_task_id(task.task_id) + ".json")
        _write_once_json(task_path, per)

        program_sig = ""
        if isinstance(solve.get("program_sig"), str):
            program_sig = str(solve.get("program_sig") or "")

        ev_body = {
            "schema_version": 129,
            "kind": "arc_task_event_v129",
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

    task_events_path = out_dir / "arc_task_events_v129.jsonl"
    _ensure_absent(task_events_path)
    from atos_core.act import canonical_json_dumps as _cjd

    with open(task_events_path, "x", encoding="utf-8") as f:
        for r in task_events:
            f.write(_cjd(r))
            f.write("\n")
    chain_hash = sha256_hex(_cjd([r["entry_hash"] for r in task_events]).encode("utf-8"))

    eval_obj = {
        "schema_version": 129,
        "kind": "arc_eval_v129",
        "seed": int(seed),
        "arc_root": str(arc_root),
        "split": str(split),
        "limit": int(limit),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "failure_counts": {str(k): int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "chain": {"arc_task_events_sha256": _sha256_file(task_events_path), "arc_task_chain_hash_v129": str(chain_hash)},
        "sha256": {"arc_canonical_jsonl": _sha256_file(canon_jsonl), "arc_manifest_json": _sha256_file(manifest_path)},
    }
    eval_obj["eval_sig"] = sha256_hex(_cjd(eval_obj).encode("utf-8"))
    eval_path = out_dir / "eval.json"
    _write_once_json(eval_path, eval_obj)

    summary = {
        "schema_version": 129,
        "kind": "arc_summary_v129",
        "seed": int(seed),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "solve_rate": float(tasks_solved / tasks_total) if tasks_total else 0.0,
        "eval_sha256": _sha256_file(eval_path),
        "arc_task_chain_hash_v129": str(chain_hash),
    }
    summary["summary_sig"] = sha256_hex(_cjd(summary).encode("utf-8"))
    _write_once_json(out_dir / "summary.json", summary)

    snap_after = _repo_snapshot_sha256_v129(root=repo_root)
    isolation = {
        "schema_version": 129,
        "kind": "arc_isolation_check_v129",
        "excluded_dir_parts": sorted(list(_excluded_dir_parts_v129())),
        "repo_snapshot_before": str(snap_before),
        "repo_snapshot_after": str(snap_after),
        "ok": bool(snap_before == snap_after),
    }
    isolation["isolation_sig"] = sha256_hex(_cjd(isolation).encode("utf-8"))
    _write_once_json(out_dir / "isolation_check_v129.json", isolation)
    if not isolation["ok"]:
        raise SystemExit("isolation_failed:repo_snapshot_changed")

    return {"eval": eval_obj, "summary": summary}


def _build_report_markdown_v129(*, eval_obj: Dict[str, Any]) -> str:
    total = int(eval_obj.get("tasks_total") or 0)
    solved = int(eval_obj.get("tasks_solved") or 0)
    unknown = int(eval_obj.get("tasks_unknown") or 0)
    failed = int(eval_obj.get("tasks_failed") or 0)

    failures = eval_obj.get("failure_counts")
    failures = failures if isinstance(failures, dict) else {}
    top_fail = sorted(((str(k), int(failures[k])) for k in failures.keys()), key=lambda kv: (-int(kv[1]), str(kv[0])))[:10]

    lines: List[str] = []
    lines.append("# ARC_DIAG_REPORT_v129")
    lines.append("")
    lines.append("## Solve rate")
    lines.append(f"- tasks_total={total} solved={solved} unknown={unknown} failed={failed}")
    lines.append(f"- solve_rate={(float(solved/total) if total else 0.0):.3f}")
    lines.append("")
    lines.append("## Top failures")
    if not top_fail:
        lines.append("- (none)")
    else:
        for k, n in top_fail:
            lines.append(f"- {k}: {n}")
    lines.append("")
    lines.append("## Anti-hack compliance")
    lines.append("- Solver NÃO usa task_id para decidir regras; task_id só aparece em logs.")
    lines.append("- Somente train_pairs + test_in entram na abdução/plano/compilação (nunca test_output).")
    lines.append("- Abdução por operador (inverse_propose_*) + indução de seletor são registradas no trace por task.")
    lines.append("- Ambiguidade mínima => UNKNOWN (fail-closed).")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc_root", required=True)
    ap.add_argument("--split", default="sample")
    ap.add_argument("--limit", type=int, default=999999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_base", required=True, help="Base path; script writes _try1 and _try2 and fails if they exist.")
    args = ap.parse_args()

    out_base = Path(str(args.out_base))
    out1 = Path(str(out_base) + "_try1")
    out2 = Path(str(out_base) + "_try2")
    _ensure_absent(out1)
    _ensure_absent(out2)

    r1 = _run_one(arc_root=str(args.arc_root), split=str(args.split), limit=int(args.limit), seed=int(args.seed), out_dir=out1)
    rep1 = _build_report_markdown_v129(eval_obj=r1["eval"])
    _write_text_x(out1 / "ARC_DIAG_REPORT_v129.md", rep1)

    r2 = _run_one(arc_root=str(args.arc_root), split=str(args.split), limit=int(args.limit), seed=int(args.seed), out_dir=out2)
    rep2 = _build_report_markdown_v129(eval_obj=r2["eval"])
    _write_text_x(out2 / "ARC_DIAG_REPORT_v129.md", rep2)

    same = True
    for rel in [
        "eval.json",
        "summary.json",
        "input/arc_manifest_v129.json",
        "input/arc_tasks_canonical_v129.jsonl",
        "arc_task_events_v129.jsonl",
        "isolation_check_v129.json",
        "ARC_DIAG_REPORT_v129.md",
    ]:
        p1 = out1 / rel
        p2 = out2 / rel
        if _sha256_file(p1) != _sha256_file(p2):
            same = False
            break

    out = {
        "schema_version": 129,
        "kind": "arc_scalpel_run_v129",
        "ok": bool(same),
        "seed": int(args.seed),
        "out_try1": str(out1),
        "out_try2": str(out2),
        "sha256": {"try1_summary": _sha256_file(out1 / "summary.json"), "try2_summary": _sha256_file(out2 / "summary.json")},
    }
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))
    if not same:
        raise SystemExit("determinism_failed")


if __name__ == "__main__":
    main()
