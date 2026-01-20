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


def _excluded_dir_parts_v127() -> set:
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
    }


def _repo_snapshot_sha256_v127(*, root: Path) -> str:
    excluded = _excluded_dir_parts_v127()
    rows: List[Dict[str, Any]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in excluded for part in p.parts):
            continue
        rel = p.relative_to(root).as_posix()
        rows.append({"path": str(rel), "sha256": _sha256_file(p)})
    rows.sort(key=lambda r: str(r["path"]))
    body = {"schema_version": 127, "kind": "repo_snapshot_v127", "files": rows}
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
    from atos_core.arc_loader_v127 import iter_canonical_tasks_v127, write_arc_canonical_jsonl_v127
    from atos_core.arc_solver_v127 import diagnose_missing_operator_v127, solve_arc_task_v127

    repo_root = Path(__file__).resolve().parent.parent
    snap_before = _repo_snapshot_sha256_v127(root=repo_root)

    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    canon_jsonl = input_dir / "arc_tasks_canonical_v127.jsonl"
    manifest = write_arc_canonical_jsonl_v127(arc_root=str(arc_root), out_jsonl_path=str(canon_jsonl), limit=int(limit))
    manifest_path = input_dir / "arc_manifest_v127.json"
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
    gap_counts: Dict[str, int] = {}

    for task in iter_canonical_tasks_v127(str(canon_jsonl)):
        tasks_total += 1
        solve = solve_arc_task_v127(train_pairs=list(task.train_pairs), test_in=task.test_in)
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

        diag = diagnose_missing_operator_v127(train_pairs=list(task.train_pairs))
        if isinstance(diag, dict) and isinstance(diag.get("gaps"), list):
            for g in diag.get("gaps") or []:
                gg = str(g)
                gap_counts[gg] = int(gap_counts.get(gg, 0)) + 1

        per = {
            "schema_version": 127,
            "kind": "arc_task_result_v127",
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
            "schema_version": 127,
            "kind": "arc_task_event_v127",
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

    task_events_path = out_dir / "arc_task_events_v127.jsonl"
    _ensure_absent(task_events_path)
    with open(task_events_path, "x", encoding="utf-8") as f:
        for r in task_events:
            f.write(canonical_json_dumps(r))
            f.write("\n")
    chain_hash = sha256_hex(canonical_json_dumps([r["entry_hash"] for r in task_events]).encode("utf-8"))

    eval_obj = {
        "schema_version": 127,
        "kind": "arc_eval_v127",
        "seed": int(seed),
        "arc_root": str(arc_root),
        "split": str(split),
        "limit": int(limit),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "failure_counts": {str(k): int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "gap_counts": {str(k): int(gap_counts[k]) for k in sorted(gap_counts.keys())},
        "chain": {"arc_task_events_sha256": _sha256_file(task_events_path), "arc_task_chain_hash_v127": str(chain_hash)},
        "sha256": {"arc_canonical_jsonl": _sha256_file(canon_jsonl), "arc_manifest_json": _sha256_file(manifest_path)},
    }
    eval_obj["eval_sig"] = sha256_hex(canonical_json_dumps(eval_obj).encode("utf-8"))
    eval_path = out_dir / "eval.json"
    _write_once_json(eval_path, eval_obj)

    summary = {
        "schema_version": 127,
        "kind": "arc_summary_v127",
        "seed": int(seed),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "solve_rate": float(tasks_solved / tasks_total) if tasks_total else 0.0,
        "eval_sha256": _sha256_file(eval_path),
        "arc_task_chain_hash_v127": str(chain_hash),
    }
    summary["summary_sig"] = sha256_hex(canonical_json_dumps(summary).encode("utf-8"))
    _write_once_json(out_dir / "summary.json", summary)

    snap_after = _repo_snapshot_sha256_v127(root=repo_root)
    isolation = {
        "schema_version": 127,
        "kind": "arc_isolation_check_v127",
        "excluded_dir_parts": sorted(list(_excluded_dir_parts_v127())),
        "repo_snapshot_before": str(snap_before),
        "repo_snapshot_after": str(snap_after),
        "ok": bool(snap_before == snap_after),
    }
    isolation["isolation_sig"] = sha256_hex(canonical_json_dumps(isolation).encode("utf-8"))
    _write_once_json(out_dir / "isolation_check_v127.json", isolation)
    if not isolation["ok"]:
        raise SystemExit("isolation_failed:repo_snapshot_changed")

    return {"eval": eval_obj, "summary": summary}


def _build_report_markdown_v127(*, eval_obj: Dict[str, Any]) -> str:
    total = int(eval_obj.get("tasks_total") or 0)
    solved = int(eval_obj.get("tasks_solved") or 0)
    unknown = int(eval_obj.get("tasks_unknown") or 0)
    failed = int(eval_obj.get("tasks_failed") or 0)

    failures = eval_obj.get("failure_counts")
    failures = failures if isinstance(failures, dict) else {}
    top_fail = sorted(((str(k), int(failures[k])) for k in failures.keys()), key=lambda kv: (-int(kv[1]), str(kv[0])))[:10]

    gaps = eval_obj.get("gap_counts")
    gaps = gaps if isinstance(gaps, dict) else {}
    top_gaps = sorted(((str(k), int(gaps[k])) for k in gaps.keys()), key=lambda kv: (-int(kv[1]), str(kv[0])))[:10]

    backlog: List[Tuple[str, str, str]] = []
    if "shape_change_present" in gaps:
        backlog.append(("resize_nearest", "resize_nearest(grid,new_h,new_w)->grid", "determinístico; sem inventar; inverse_propose necessário"))
    if "multi_color_delta_present" in gaps:
        backlog.append(("mask_difference", "mask_difference(inp,out)->mask", "puro; determinístico; útil para abdução de passos residuais"))

    lines: List[str] = []
    lines.append("# ARC_DIAG_REPORT_v127")
    lines.append("")
    lines.append("## Solve rate")
    lines.append(f"- tasks_total={total} solved={solved} unknown={unknown} failed={failed}")
    if total:
        lines.append(f"- solve_rate={solved/total:.3f}")
    lines.append("")
    lines.append("## Top failures")
    if not top_fail:
        lines.append("- (none)")
    else:
        for k, n in top_fail:
            lines.append(f"- {k}: {n}")
    lines.append("")
    lines.append("## Top operator-gap signals (diagnostic)")
    if not top_gaps:
        lines.append("- (none)")
    else:
        for k, n in top_gaps:
            lines.append(f"- {k}: {n}")
    lines.append("")
    lines.append("## Backlog (general operators)")
    lines.append("Cada item abaixo é um operador geral (não ARC-específico) com assinatura e invariantes.")
    if not backlog:
        lines.append("- (no backlog items inferred from this run)")
    else:
        for name, sig, inv in backlog:
            lines.append(f"- `{name}`: `{sig}`")
            lines.append(f"  - invariantes: {inv}")
    lines.append("")
    lines.append("## Anti-hack compliance")
    lines.append("- Solver NÃO usa task_id para decidir regras; task_id só aparece em logs.")
    lines.append("- Somente train_pairs + test_in entram na abdução/plano/compilação (nunca test_output).")
    lines.append("- Abdução por operador (`inverse_propose_*`) + indução de seletor são registradas no trace por task.")
    lines.append("- Ambiguidade mínima => UNKNOWN (fail-closed).")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc_root", required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_base", required=True)
    args = ap.parse_args()

    out_base = Path(str(args.out_base))
    out1 = Path(str(out_base) + "_try1")
    out2 = Path(str(out_base) + "_try2")
    _ensure_absent(out1)
    _ensure_absent(out2)

    r1 = _run_one(arc_root=str(args.arc_root), split=str(args.split), limit=int(args.limit), seed=int(args.seed), out_dir=out1)
    r2 = _run_one(arc_root=str(args.arc_root), split=str(args.split), limit=int(args.limit), seed=int(args.seed), out_dir=out2)

    from atos_core.act import canonical_json_dumps, sha256_hex

    if canonical_json_dumps(r1["eval"]) != canonical_json_dumps(r2["eval"]):
        raise SystemExit("determinism_failed:eval_json")
    if canonical_json_dumps(r1["summary"]) != canonical_json_dumps(r2["summary"]):
        raise SystemExit("determinism_failed:summary_json")

    report_md = _build_report_markdown_v127(eval_obj=dict(r1["eval"]))
    report_path1 = out1 / "ARC_DIAG_REPORT_v127.md"
    report_path2 = out2 / "ARC_DIAG_REPORT_v127.md"
    _write_text_x(report_path1, report_md + "\n")
    _write_text_x(report_path2, report_md + "\n")

    out = {
        "ok": True,
        "determinism_ok": True,
        "try1_dir": str(out1),
        "try2_dir": str(out2),
        "summary_sha256": sha256_hex(canonical_json_dumps(r1["summary"]).encode("utf-8")),
        "eval_sha256": str(r1["summary"].get("eval_sha256") or ""),
        "report_sha256": sha256_hex(report_md.encode("utf-8")),
        "report_path_try1": str(report_path1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

