#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _excluded_dir_parts_v132() -> set:
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


def _repo_snapshot_sha256_v132(*, root: Path, exclude_paths: Sequence[Path]) -> str:
    excluded = _excluded_dir_parts_v132()
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
    body = {"schema_version": 132, "kind": "repo_snapshot_v132", "files": rows}
    from atos_core.act import canonical_json_dumps, sha256_hex

    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _sanitize_task_id(task_id: str) -> str:
    s = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(task_id)])
    return s or "task"


def _build_report_markdown_v132(*, eval_obj: Dict[str, Any], backlog: Sequence[Dict[str, Any]]) -> str:
    total = int(eval_obj.get("tasks_total") or 0)
    solved = int(eval_obj.get("tasks_solved") or 0)
    unknown = int(eval_obj.get("tasks_unknown") or 0)
    failed = int(eval_obj.get("tasks_failed") or 0)
    failures = eval_obj.get("failure_counts")
    failures = failures if isinstance(failures, dict) else {}
    top = sorted(((str(k), int(failures[k])) for k in failures.keys()), key=lambda kv: (-int(kv[1]), str(kv[0])))[:15]

    lines: List[str] = []
    lines.append("# ARC_DIAG_REPORT_v132")
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


def _derive_backlog_v132(*, failure_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    # Deterministic, general operator proposals based on common failure kinds.
    # Diagnostic only; solver must not branch on these.
    ordered = sorted(((str(k), int(failure_counts[k])) for k in failure_counts.keys()), key=lambda kv: (-kv[1], kv[0]))
    dominant = ordered[0][0] if ordered else ""
    out: List[Dict[str, Any]] = []

    out.append(
        {
            "name": "connected_components4(grid[, colors, bg]) -> object_set",
            "signature": "(GRID[, [COLOR], COLOR]) -> OBJECT_SET",
            "invariants": "Determinístico; 4-neigh; objetos ordenados por (area desc, bbox, cor).",
            "examples": "Separar blobs e selecionar alvo por área/posição.",
            "covers": "MISSING_OPERATOR em tarefas object-centric.",
        }
    )
    out.append(
        {
            "name": "select_object(object_set, key, order, rank[, color_filter]) -> object",
            "signature": "(OBJECT_SET, KEY, ORDER, INT[, COLOR]) -> OBJECT",
            "invariants": "Key/order/rank são enums; desempate determinístico por bbox/cor/cells.",
            "examples": "Escolher menor/maior/mais próximo do centro.",
            "covers": "Regime 4/5 (variável latente: alvo) sem heurística por task.",
        }
    )
    out.append(
        {
            "name": "paint_rect(grid, bbox, color) -> grid",
            "signature": "(GRID, BBOX, COLOR) -> GRID",
            "invariants": "Não muda shape; pinta apenas bbox; cores 0..9.",
            "examples": "Preencher bbox do alvo com cor constante.",
            "covers": "Edição estrutural governada por bbox.",
        }
    )
    out.append(
        {
            "name": "paste(base, patch, at=(r,c)[, transparent]) -> grid",
            "signature": "(GRID, GRID, (INT,INT)[, COLOR]) -> GRID",
            "invariants": "Determinístico; não muda shape; transparent opcional.",
            "examples": "Recortar patch e colar em offset derivado do delta.",
            "covers": "Composição causal (crop->paste).",
        }
    )
    if dominant == "SEARCH_BUDGET_EXCEEDED":
        out.append(
            {
                "name": "inverse_propose(op, inp, out) -> small_candidates",
                "signature": "(OP, GRID, GRID) -> [PARAMS]",
                "invariants": "Somente train pairs; candidatos ordenados; reduz branching.",
                "examples": "Inferir bbox do delta e offsets plausíveis.",
                "covers": "SEARCH_BUDGET_EXCEEDED por explosão combinatória.",
            }
        )
    return out[:10]


def _build_outputs_manifest_v132(*, out_dir: Path) -> Dict[str, Any]:
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
        out_dir / "ARC_DIAG_REPORT_v132.md",
        out_dir / "isolation_check_v132.json",
        out_dir / "input" / "arc_manifest_v132.json",
        out_dir / "input" / "arc_tasks_canonical_v132.jsonl",
    ]
    for p in fixed:
        files.append({"path": rel(p), "sha256": _sha256_file(p)})
    for p in per_task_files:
        files.append({"path": rel(p), "sha256": _sha256_file(p)})

    body = {"schema_version": 132, "kind": "arc_outputs_manifest_v132", "files": files}
    from atos_core.act import canonical_json_dumps, sha256_hex

    body["manifest_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return body


def _run_one(*, arc_root: str, split: str, limit: int, seed: int, out_dir: Path) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONPYCACHEPREFIX"] = str(out_dir / ".pycache")

    from atos_core.act import canonical_json_dumps, sha256_hex
    from atos_core.arc_loader_v132 import iter_canonical_tasks_v132, write_arc_canonical_jsonl_v132
    from atos_core.arc_solver_v132 import solve_arc_task_v132

    repo_root = Path(__file__).resolve().parent.parent
    snap_before = _repo_snapshot_sha256_v132(root=repo_root, exclude_paths=[out_dir])

    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    canon_jsonl = input_dir / "arc_tasks_canonical_v132.jsonl"
    manifest = write_arc_canonical_jsonl_v132(
        arc_root=str(arc_root), out_jsonl_path=str(canon_jsonl), limit=int(limit), split=str(split)
    )
    manifest_path = input_dir / "arc_manifest_v132.json"
    _write_once_json(manifest_path, manifest)

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

    for task in iter_canonical_tasks_v132(str(canon_jsonl)):
        tasks_total += 1
        res = solve_arc_task_v132(train_pairs=list(task.train_pairs), test_in=task.test_in)
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
            "schema_version": 132,
            "kind": "arc_per_task_v132",
            "task_id": str(task_id),
            "task": task.to_dict(),
            "result": dict(res),
        }
        _write_once_json(per_task_path, per_task_obj)

        row = {
            "schema_version": 132,
            "kind": "arc_per_task_manifest_row_v132",
            "task_id": str(task_id),
            "status": str(status),
            "failure_kind": str(failure_kind),
            "program_sig": str(res.get("program_sig") or ""),
            "program_cost_bits": int(res.get("program_cost_bits") or 0),
            "predicted_grid_hash": str(res.get("predicted_grid_hash") or ""),
            "per_task_sha256": _sha256_file(per_task_path),
        }
        row["row_sig"] = sha256_hex(canonical_json_dumps(row).encode("utf-8"))
        per_task_rows.append(row)

        tr = res.get("trace")
        if isinstance(tr, dict):
            tps = tr.get("trace_programs")
            if isinstance(tps, list):
                for i, tp in enumerate(tps[:10]):
                    if not isinstance(tp, dict):
                        continue
                    trace_rows.append(
                        {
                            "schema_version": 132,
                            "kind": "arc_candidate_trace_row_v132",
                            "task_id": str(task_id),
                            "candidate_index": int(i),
                            "program_sig": str(tp.get("program_sig") or ""),
                            "cost_bits": int(tp.get("cost_bits") or 0),
                            "depth": int(tp.get("depth") or 0),
                            "ok_train": bool(tp.get("ok_train") or False),
                            "mismatch": tp.get("mismatch"),
                        }
                    )

    with open(per_task_manifest_path, "x", encoding="utf-8") as f:
        for row in per_task_rows:
            f.write(canonical_json_dumps(row))
            f.write("\n")
    with open(trace_candidates_path, "x", encoding="utf-8") as f:
        for row in trace_rows:
            row["row_sig"] = sha256_hex(canonical_json_dumps(row).encode("utf-8"))
            f.write(canonical_json_dumps(row))
            f.write("\n")

    eval_obj: Dict[str, Any] = {
        "schema_version": 132,
        "kind": "arc_eval_v132",
        "arc_root": str(arc_root),
        "split": str(split),
        "limit": int(limit),
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
        "schema_version": 132,
        "kind": "arc_summary_v132",
        "seed": int(seed),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "solve_rate": float(solved_rate),
        "eval_sha256": _sha256_file(out_dir / "eval.json"),
    }
    summary_obj["summary_sig"] = sha256_hex(canonical_json_dumps(summary_obj).encode("utf-8"))
    summary_obj["summary_sha256"] = sha256_hex(canonical_json_dumps(summary_obj).encode("utf-8"))
    _write_once_json(out_dir / "summary.json", summary_obj)

    smoke_summary = {
        "schema_version": 132,
        "kind": "arc_smoke_summary_v132",
        "summary_sha256": str(summary_obj["summary_sha256"]),
        "eval_sha256": str(summary_obj["eval_sha256"]),
    }
    _write_once_json(out_dir / "smoke_summary.json", smoke_summary)

    backlog = _derive_backlog_v132(failure_counts=dict(eval_obj["failure_counts"]))
    report_text = _build_report_markdown_v132(eval_obj=eval_obj, backlog=backlog)
    report_path = out_dir / "ARC_DIAG_REPORT_v132.md"
    _write_text_x(report_path, report_text + "\n")

    snap_after = _repo_snapshot_sha256_v132(root=repo_root, exclude_paths=[out_dir])
    isolation = {
        "schema_version": 132,
        "kind": "arc_isolation_check_v132",
        "ok": bool(snap_before == snap_after),
        "repo_snapshot_before": str(snap_before),
        "repo_snapshot_after": str(snap_after),
        "excluded_dir_parts": sorted(_excluded_dir_parts_v132()),
    }
    _write_once_json(out_dir / "isolation_check_v132.json", isolation)

    outputs_manifest = _build_outputs_manifest_v132(out_dir=out_dir)
    _write_once_json(out_dir / "outputs_manifest.json", outputs_manifest)

    return {
        "out_dir": str(out_dir),
        "summary_sha256": str(summary_obj["summary_sha256"]),
        "eval_sha256": str(summary_obj["eval_sha256"]),
        "outputs_manifest_sig": str(outputs_manifest["manifest_sig"]),
        "isolation_ok": bool(isolation["ok"]),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "solve_rate": float(solved_rate),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc_root", default="data/arc_v124_sample")
    ap.add_argument("--split", default="sample")
    ap.add_argument("--limit", type=int, default=999999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_base", required=True)
    args = ap.parse_args()

    out_base = Path(str(args.out_base))
    out_try1 = out_base.with_name(out_base.name + "_try1")
    out_try2 = out_base.with_name(out_base.name + "_try2")

    # Ensure WORM at directory level.
    _ensure_absent(out_try1)
    _ensure_absent(out_try2)

    one = _run_one(
        arc_root=str(args.arc_root),
        split=str(args.split),
        limit=int(args.limit),
        seed=int(args.seed),
        out_dir=out_try1,
    )
    two = _run_one(
        arc_root=str(args.arc_root),
        split=str(args.split),
        limit=int(args.limit),
        seed=int(args.seed),
        out_dir=out_try2,
    )

    determinism_ok = (str(one["summary_sha256"]) == str(two["summary_sha256"])) and (
        str(one["outputs_manifest_sig"]) == str(two["outputs_manifest_sig"])
    )
    ok = bool(determinism_ok) and bool(one["isolation_ok"]) and bool(two["isolation_ok"])

    out = {
        "ok": bool(ok),
        "determinism_ok": bool(determinism_ok),
        "arc_root": str(args.arc_root),
        "split": str(args.split),
        "limit": int(args.limit),
        "seed": int(args.seed),
        "try1": dict(one),
        "try2": dict(two),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

