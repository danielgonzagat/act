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


def _excluded_dir_parts_v131() -> set:
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


def _repo_snapshot_sha256_v131(*, root: Path, exclude_paths: Sequence[Path]) -> str:
    excluded = _excluded_dir_parts_v131()
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
    body = {"schema_version": 131, "kind": "repo_snapshot_v131", "files": rows}
    from atos_core.act import canonical_json_dumps, sha256_hex

    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _sanitize_task_id(task_id: str) -> str:
    s = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(task_id)])
    return s or "task"


def _build_report_markdown_v131(*, eval_obj: Dict[str, Any], backlog: Sequence[Dict[str, Any]]) -> str:
    total = int(eval_obj.get("tasks_total") or 0)
    solved = int(eval_obj.get("tasks_solved") or 0)
    unknown = int(eval_obj.get("tasks_unknown") or 0)
    failed = int(eval_obj.get("tasks_failed") or 0)
    failures = eval_obj.get("failure_counts")
    failures = failures if isinstance(failures, dict) else {}
    top = sorted(((str(k), int(failures[k])) for k in failures.keys()), key=lambda kv: (-int(kv[1]), str(kv[0])))[:15]

    lines: List[str] = []
    lines.append("# ARC_DIAG_REPORT_v131")
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


def _derive_backlog_v131(*, failure_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    # Deterministic, general operator proposals based on common failure kinds.
    # This is a diagnostic only; solver must not branch on these.
    ordered = sorted(((str(k), int(failure_counts[k])) for k in failure_counts.keys()), key=lambda kv: (-kv[1], kv[0]))
    dominant = ordered[0][0] if ordered else ""
    out: List[Dict[str, Any]] = []

    # Always include a small, general backlog (no task-specific patterns).
    out.append(
        {
            "name": "connected_components4(grid[, color]) -> object_set",
            "signature": "(GRID[, COLOR]) -> OBJECT_SET",
            "invariants": "Determinístico; 4-neigh; objetos ordenados por bbox/cor; cores 0..9.",
            "examples": "Ex: separar 2 blobs por cor e selecionar maior/menor por área.",
            "covers": "MISSING_OPERATOR / AMBIGUOUS_RULE em tarefas de seleção de objeto e edição localizada.",
        }
    )
    out.append(
        {
            "name": "paint_rect(grid, bbox, color[, mode]) -> grid",
            "signature": "(GRID, BBOX, COLOR[, MODE]) -> GRID",
            "invariants": "Não altera shape; pinta apenas dentro do bbox; cores 0..9.",
            "examples": "Ex: desenhar retângulo preenchido onde delta mudou.",
            "covers": "MISSING_OPERATOR quando o delta é retangular e envolve preenchimento/borda.",
        }
    )
    out.append(
        {
            "name": "draw_rect_border(grid, bbox, color, thickness=1) -> grid",
            "signature": "(GRID, BBOX, COLOR[, INT]) -> GRID",
            "invariants": "Não altera shape; desenha contorno; thickness pequeno e determinístico.",
            "examples": "Ex: borda do bbox do objeto não-zero.",
            "covers": "MISSING_OPERATOR em tarefas de borda/contorno.",
        }
    )
    out.append(
        {
            "name": "paste(base, patch, at=(r,c)[, transparent]) -> grid",
            "signature": "(GRID, GRID, (INT,INT)[, COLOR]) -> GRID",
            "invariants": "Determinístico; não muda shape; recorta patch se necessário; transparent opcional.",
            "examples": "Ex: copiar um subgrid e colar em offset.",
            "covers": "MISSING_OPERATOR em tarefas de composição/duplicação local.",
        }
    )
    out.append(
        {
            "name": "symmetry_detect(grid) -> {axis, kind}",
            "signature": "(GRID) -> SYMMETRY_HYP",
            "invariants": "Somente diagnóstico/hyp; retorna evidência (mismatch count) e eixos candidatos.",
            "examples": "Ex: detectar se o output é reflexo do input.",
            "covers": "MISSING_OPERATOR em tarefas de flip/rotate/reflect parametrizado.",
        }
    )

    if dominant == "SEARCH_BUDGET_EXCEEDED":
        out.append(
            {
                "name": "inverse_propose(op, inp, out) -> small_candidates",
                "signature": "(OP, GRID, GRID) -> [PARAMS]",
                "invariants": "Só usa train pairs; candidatos ordenados; sem task_id; reduz branching.",
                "examples": "Ex: inferir mapping de cores funcional; inferir bbox da máscara de delta.",
                "covers": "SEARCH_BUDGET_EXCEEDED por explosão combinatória.",
            }
        )
    return out[:10]


def _build_outputs_manifest_v131(*, out_dir: Path) -> Dict[str, Any]:
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
        out_dir / "ARC_DIAG_REPORT_v131.md",
        out_dir / "isolation_check_v131.json",
        out_dir / "input" / "arc_manifest_v131.json",
        out_dir / "input" / "arc_tasks_canonical_v131.jsonl",
    ]
    for p in fixed:
        files.append({"path": rel(p), "sha256": _sha256_file(p)})
    for p in per_task_files:
        files.append({"path": rel(p), "sha256": _sha256_file(p)})

    body = {"schema_version": 131, "kind": "arc_outputs_manifest_v131", "files": files}
    from atos_core.act import canonical_json_dumps, sha256_hex

    body["manifest_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return body


def _run_one(*, arc_root: str, split: str, limit: int, seed: int, out_dir: Path) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONPYCACHEPREFIX"] = str(out_dir / ".pycache")

    from atos_core.act import canonical_json_dumps, sha256_hex
    from atos_core.arc_loader_v131 import iter_canonical_tasks_v131, write_arc_canonical_jsonl_v131
    from atos_core.arc_solver_v131 import solve_arc_task_v131

    repo_root = Path(__file__).resolve().parent.parent
    snap_before = _repo_snapshot_sha256_v131(root=repo_root, exclude_paths=[out_dir])

    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    canon_jsonl = input_dir / "arc_tasks_canonical_v131.jsonl"
    manifest = write_arc_canonical_jsonl_v131(
        arc_root=str(arc_root), out_jsonl_path=str(canon_jsonl), limit=int(limit), split=str(split)
    )
    manifest_path = input_dir / "arc_manifest_v131.json"
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

    with open(per_task_manifest_path, "x", encoding="utf-8") as per_f, open(
        trace_candidates_path, "x", encoding="utf-8"
    ) as trace_f:
        for task in iter_canonical_tasks_v131(str(canon_jsonl)):
            tasks_total += 1
            solve = solve_arc_task_v131(train_pairs=list(task.train_pairs), test_in=task.test_in)
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
                "schema_version": 131,
                "kind": "arc_task_result_v131",
                "task_id": str(task.task_id),
                "input": task.to_dict(),
                "result": dict(solve),
            }
            per["per_task_sig"] = sha256_hex(canonical_json_dumps(per).encode("utf-8"))
            task_path = per_task_dir / (_sanitize_task_id(task.task_id) + ".json")
            _write_once_json(task_path, per)

            pm_row = {
                "schema_version": 131,
                "kind": "arc_per_task_manifest_row_v131",
                "task_id": str(task.task_id),
                "status": str(status),
                "failure_kind": str(failure_kind),
                "program_sig": str(solve.get("program_sig") or ""),
                "program_cost_bits": int(solve.get("program_cost_bits") or 0),
                "predicted_grid_hash": str(solve.get("predicted_grid_hash") or ""),
                "per_task_sha256": _sha256_file(task_path),
            }
            pm_row["row_sig"] = sha256_hex(canonical_json_dumps(pm_row).encode("utf-8"))
            per_f.write(canonical_json_dumps(pm_row))
            per_f.write("\n")

            tr = solve.get("trace") if isinstance(solve.get("trace"), dict) else {}
            trace_programs = tr.get("trace_programs") if isinstance(tr.get("trace_programs"), list) else []
            for i, cand in enumerate(trace_programs):
                if not isinstance(cand, dict):
                    continue
                row = {
                    "schema_version": 131,
                    "kind": "arc_candidate_trace_row_v131",
                    "task_id": str(task.task_id),
                    "candidate_index": int(i),
                    "program_sig": str(cand.get("program_sig") or ""),
                    "cost_bits": int(cand.get("cost_bits") or 0),
                    "depth": int(cand.get("depth") or 0),
                    "ok_train": bool(cand.get("ok_train")),
                    "mismatch": cand.get("mismatch"),
                }
                row["row_sig"] = sha256_hex(canonical_json_dumps(row).encode("utf-8"))
                trace_f.write(canonical_json_dumps(row))
                trace_f.write("\n")

    eval_obj = {
        "schema_version": 131,
        "kind": "arc_eval_v131",
        "seed": int(seed),
        "arc_root": str(arc_root),
        "split": str(split),
        "limit": int(limit),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "failure_counts": {str(k): int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "sha256": {
            "arc_canonical_jsonl": _sha256_file(canon_jsonl),
            "arc_manifest_json": _sha256_file(manifest_path),
            "per_task_manifest_jsonl": _sha256_file(per_task_manifest_path),
            "trace_candidates_jsonl": _sha256_file(trace_candidates_path),
        },
    }
    eval_obj["eval_sig"] = sha256_hex(canonical_json_dumps(eval_obj).encode("utf-8"))
    eval_path = out_dir / "eval.json"
    _write_once_json(eval_path, eval_obj)

    summary = {
        "schema_version": 131,
        "kind": "arc_summary_v131",
        "seed": int(seed),
        "tasks_total": int(tasks_total),
        "tasks_solved": int(tasks_solved),
        "tasks_unknown": int(tasks_unknown),
        "tasks_failed": int(tasks_failed),
        "solve_rate": float(tasks_solved / tasks_total) if tasks_total else 0.0,
        "eval_sha256": _sha256_file(eval_path),
    }
    from atos_core.act import sha256_hex as _sha

    summary_sha256 = _sha(canonical_json_dumps(summary).encode("utf-8"))
    summary["summary_sha256"] = str(summary_sha256)
    summary["summary_sig"] = sha256_hex(canonical_json_dumps(summary).encode("utf-8"))
    _write_once_json(out_dir / "summary.json", summary)
    _write_once_json(out_dir / "smoke_summary.json", {"summary_sha256": str(summary_sha256), "eval_sha256": _sha256_file(eval_path)})

    backlog = _derive_backlog_v131(failure_counts=failure_counts)
    report_text = _build_report_markdown_v131(eval_obj=eval_obj, backlog=backlog)
    _write_text_x(out_dir / "ARC_DIAG_REPORT_v131.md", report_text)

    snap_after = _repo_snapshot_sha256_v131(root=repo_root, exclude_paths=[out_dir])
    isolation = {
        "schema_version": 131,
        "kind": "arc_isolation_check_v131",
        "excluded_dir_parts": sorted(list(_excluded_dir_parts_v131())),
        "repo_snapshot_before": str(snap_before),
        "repo_snapshot_after": str(snap_after),
        "ok": bool(str(snap_before) == str(snap_after)),
    }
    _write_once_json(out_dir / "isolation_check_v131.json", isolation)

    outputs_manifest = _build_outputs_manifest_v131(out_dir=out_dir)
    _write_once_json(out_dir / "outputs_manifest.json", outputs_manifest)

    return {
        "out_dir": str(out_dir),
        "summary_sha256": str(summary_sha256),
        "eval_sha256": _sha256_file(eval_path),
        "isolation_ok": bool(isolation["ok"]),
        "outputs_manifest_sig": str(outputs_manifest.get("manifest_sig") or ""),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc_root", default="data/arc_v124_sample")
    ap.add_argument("--split", default="sample")
    ap.add_argument("--limit", type=int, default=999999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_base", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)

    out_base = Path(str(args.out_base))
    out_try1 = Path(str(out_base) + "_try1")
    out_try2 = Path(str(out_base) + "_try2")
    _ensure_absent(out_try1)
    _ensure_absent(out_try2)

    r1 = _run_one(arc_root=str(args.arc_root), split=str(args.split), limit=int(args.limit), seed=int(args.seed), out_dir=out_try1)
    r2 = _run_one(arc_root=str(args.arc_root), split=str(args.split), limit=int(args.limit), seed=int(args.seed), out_dir=out_try2)

    ok = bool(r1["summary_sha256"] == r2["summary_sha256"]) and bool(
        r1["outputs_manifest_sig"] == r2["outputs_manifest_sig"]
    )
    if not ok:
        raise SystemExit("determinism_mismatch_v131")

    if not bool(r1["isolation_ok"]) or not bool(r2["isolation_ok"]):
        raise SystemExit("isolation_failed_v131")

    out = {
        "ok": True,
        "determinism_ok": True,
        "schema_version": 131,
        "out_try1": str(out_try1),
        "out_try2": str(out_try2),
        "summary_sha256": str(r1["summary_sha256"]),
        "eval_sha256_try1": str(r1["eval_sha256"]),
        "eval_sha256_try2": str(r2["eval_sha256"]),
        "outputs_manifest_sig": str(r1["outputs_manifest_sig"]),
    }
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()

