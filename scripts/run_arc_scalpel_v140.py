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


def _excluded_dir_parts_v140() -> set:
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


def _repo_snapshot_sha256_v140(*, root: Path, exclude_paths: Sequence[Path]) -> str:
    excluded = _excluded_dir_parts_v140()
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
    body = {"schema_version": 140, "kind": "repo_snapshot_v140", "files": rows}
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


def _score_test_case_all_k_v140(
    *, solver_res: Dict[str, Any], want_grid: Optional[Sequence[Sequence[int]]]
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
            "k": {"1": {"status": status}, "2": {"status": status}, "3": {"status": status}},
        }

    preds: List[Sequence[Sequence[int]]] = []
    if status == "SOLVED":
        pred = solver_res.get("predicted_grid")
        if isinstance(pred, list):
            preds = [pred]  # type: ignore[list-item]
    elif status == "UNKNOWN":
        pg = solver_res.get("predicted_grids")
        if isinstance(pg, list):
            for item in pg:
                if isinstance(item, dict) and isinstance(item.get("grid"), list):
                    preds.append(item["grid"])  # type: ignore[list-item]

    def solved_at_k(k: int) -> bool:
        kk = int(k)
        for g in preds[:kk]:
            if _grid_equal(g, want_grid):
                return True
        return False

    out_k: Dict[str, Any] = {}
    for kk in (1, 2, 3):
        ok = solved_at_k(kk)
        if status == "FAIL":
            out_k[str(kk)] = {"status": "FAIL", "attempts_used": 1}
        elif ok:
            out_k[str(kk)] = {"status": "SOLVED", "attempts_used": min(int(kk), max(1, len(preds)))}
        else:
            out_k[str(kk)] = {"status": "UNKNOWN" if status == "UNKNOWN" else "FAIL", "attempts_used": min(int(kk), max(1, len(preds)))}

    failure_kind = ""
    if not solved_at_k(1):
        if status == "UNKNOWN":
            failure_kind = "AMBIGUOUS_RULE"
        elif status == "SOLVED":
            failure_kind = "TEST_OUTPUT_MISMATCH"
        else:
            fr = solver_res.get("failure_reason")
            if isinstance(fr, dict):
                failure_kind = str(fr.get("kind") or "")
            failure_kind = failure_kind or "FAIL"

    return {
        "solver_status": status,
        "scored": True,
        "failure_kind": failure_kind,
        "k": out_k,
    }


def _build_report_markdown_v140(*, eval_obj: Dict[str, Any], clusters: Sequence[Dict[str, Any]], backlog: Sequence[Dict[str, Any]]) -> str:
    total = int(eval_obj.get("tasks_total") or 0)
    solved_k = eval_obj.get("tasks_solved_by_k")
    solved_k = solved_k if isinstance(solved_k, dict) else {}
    failures = eval_obj.get("failure_counts")
    failures = failures if isinstance(failures, dict) else {}
    top_fail = sorted(((str(k), int(failures[k])) for k in failures.keys()), key=lambda kv: (-int(kv[1]), str(kv[0])))[:15]

    lines: List[str] = []
    lines.append("# ARC_DIAG_REPORT_v140")
    lines.append("")
    lines.append("## Solve rate (scored vs test outputs)")
    lines.append(f"- tasks_total={total}")
    for kk in ("1", "2", "3"):
        s = int(solved_k.get(kk, 0) or 0)
        r = (float(s) / float(total)) if total else 0.0
        lines.append(f"- solved@{kk}={s} solve_rate@{kk}={r:.3f}")
    lines.append("")
    lines.append("## Top failures (official_k failure_kind)")
    if not top_fail:
        lines.append("- (none)")
    else:
        for k, n in top_fail:
            lines.append(f"- {k}: {n}")
    lines.append("")
    lines.append("## Failure clusters (structural signature; no task_id)")
    if not clusters:
        lines.append("- (none)")
    else:
        for c in clusters[:10]:
            lines.append(f"- count={c['count']} key={c['key']} example_failure={c['failure_kind']}")
    lines.append("")
    lines.append("## Backlog (general operator/concept gaps)")
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


def _derive_backlog_v140(*, failure_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if "MISSING_OPERATOR" in failure_counts:
        out.append(
            {
                "name": "shape-changing operators with inverse propose (gap)",
                "signature": "(GRID)->GRID (scale/tile/resize families)",
                "invariants": "Determinístico; parâmetros inferidos de demonstrações; tipado.",
                "examples": "scale_cell, tile_repeat, resize_nearest integer factors.",
                "covers": "MISSING_OPERATOR clusters with shape_change.",
            }
        )
    if "SEARCH_BUDGET_EXCEEDED" in failure_counts:
        out.append(
            {
                "name": "state-space search scaling (gap)",
                "signature": "memoization + dominance + constraint propagation",
                "invariants": "No task_id; only train evidence; deterministic ordering.",
                "examples": "cache apply_op, cache eval, prune dominated states.",
                "covers": "SEARCH_BUDGET_EXCEEDED clusters.",
            }
        )
    return out[:10]


def _structural_signature_v140(task_obj: Dict[str, Any]) -> str:
    try:
        train = task_obj.get("train_pairs") or []
        if not isinstance(train, list) or not train:
            return "no_train"
        inp0 = train[0].get("in_grid")
        out0 = train[0].get("out_grid")
        hi = len(inp0) if isinstance(inp0, list) else 0
        wi = len(inp0[0]) if isinstance(inp0, list) and inp0 and isinstance(inp0[0], list) else 0
        ho = len(out0) if isinstance(out0, list) else 0
        wo = len(out0[0]) if isinstance(out0, list) and out0 and isinstance(out0[0], list) else 0
        shape_rel = "same" if (hi, wi) == (ho, wo) else "diff"
        # palette relation on first pair only (cheap; deterministic).
        pal_in: Set[int] = set()
        pal_out: Set[int] = set()
        if isinstance(inp0, list):
            for r in inp0:
                if isinstance(r, list):
                    for x in r:
                        pal_in.add(int(x))
        if isinstance(out0, list):
            for r in out0:
                if isinstance(r, list):
                    for x in r:
                        pal_out.add(int(x))
        if pal_out == pal_in:
            pal_rel = "equal"
        elif pal_out.issubset(pal_in):
            pal_rel = "subset"
        elif pal_in.issubset(pal_out):
            pal_rel = "superset"
        else:
            pal_rel = "mixed"
        # delta density bucket (first pair).
        diff = 0
        total = max(1, min(hi, ho) * min(wi, wo))
        if isinstance(inp0, list) and isinstance(out0, list) and hi == ho and wi == wo:
            for r in range(hi):
                for c in range(wi):
                    if int(inp0[r][c]) != int(out0[r][c]):
                        diff += 1
        ratio = float(diff) / float(total)
        if ratio <= 0.05:
            dens = "sparse"
        elif ratio <= 0.3:
            dens = "local"
        else:
            dens = "dense"
        return f"shape={shape_rel}|pal={pal_rel}|dens={dens}"
    except Exception:
        return "sig_error"


def _build_outputs_manifest_v140(*, out_dir: Path) -> Dict[str, Any]:
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
        out_dir / "ARC_DIAG_REPORT_v140.md",
        out_dir / "isolation_check_v140.json",
        out_dir / "input" / "arc_manifest_v140.json",
        out_dir / "input" / "arc_tasks_canonical_v140.jsonl",
    ]
    for p in fixed:
        files.append({"path": rel(p), "sha256": _sha256_file(p)})
    for p in per_task_files:
        files.append({"path": rel(p), "sha256": _sha256_file(p)})

    body = {"schema_version": 140, "kind": "arc_outputs_manifest_v140", "files": files}
    from atos_core.act import canonical_json_dumps, sha256_hex

    body["manifest_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return body


def _run_one(
    *,
    arc_root: str,
    split: str,
    limit: int,
    seed: int,
    out_dir: Path,
    benchmark_profile: str,
    max_trials: int,
    max_depth: int,
    max_programs: int,
) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONPYCACHEPREFIX"] = str(out_dir / ".pycache")

    from atos_core.act import canonical_json_dumps, sha256_hex
    from atos_core.arc_loader_v140 import iter_canonical_tasks_v140, write_arc_canonical_jsonl_v140
    from atos_core.arc_solver_v140 import SolveConfigV140, solve_arc_task_v140

    repo_root = Path(__file__).resolve().parent.parent
    snap_before = _repo_snapshot_sha256_v140(root=repo_root, exclude_paths=[out_dir])

    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    canon_jsonl = input_dir / "arc_tasks_canonical_v140.jsonl"
    manifest_path = input_dir / "arc_manifest_v140.json"
    write_arc_canonical_jsonl_v140(
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
    solved_by_k = {"1": 0, "2": 0, "3": 0}
    unknown_by_k = {"1": 0, "2": 0, "3": 0}
    failed_by_k = {"1": 0, "2": 0, "3": 0}
    failure_counts: Dict[str, int] = {}
    clusters: Dict[str, int] = {}

    per_task_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    cfg = SolveConfigV140(
        max_depth=int(max_depth),
        max_programs=int(max_programs),
        trace_program_limit=80,
        max_ambiguous_outputs=16,
    )

    for task in iter_canonical_tasks_v140(str(canon_jsonl)):
        tasks_total += 1

        test_case_results: List[Dict[str, Any]] = []
        solver_results: List[Dict[str, Any]] = []
        for test_in, test_out in task.test_pairs:
            res = solve_arc_task_v140(train_pairs=list(task.train_pairs), test_in=test_in, config=cfg)
            solver_results.append(res)
            want_grid = [list(r) for r in test_out] if test_out is not None else None
            test_case_results.append(_score_test_case_all_k_v140(solver_res=res, want_grid=want_grid))

        # Aggregate per k.
        for kk in ("1", "2", "3"):
            if all(r["k"][kk]["status"] == "SOLVED" for r in test_case_results):
                solved_by_k[kk] = int(solved_by_k[kk]) + 1
            elif any(r["k"][kk]["status"] == "FAIL" for r in test_case_results):
                failed_by_k[kk] = int(failed_by_k[kk]) + 1
            else:
                unknown_by_k[kk] = int(unknown_by_k[kk]) + 1

        official_k = str(int(max_trials))
        status = "SOLVED" if all(r["k"][official_k]["status"] == "SOLVED" for r in test_case_results) else (
            "FAIL" if any(r["k"][official_k]["status"] == "FAIL" for r in test_case_results) else "UNKNOWN"
        )

        failure_kind = ""
        if status != "SOLVED":
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

        sig = _structural_signature_v140(task.to_dict())
        if status != "SOLVED":
            clusters[f"{failure_kind}|{sig}"] = int(clusters.get(f"{failure_kind}|{sig}", 0)) + 1

        per_task_obj = {
            "kind": "arc_per_task_v140",
            "schema_version": 140,
            "task": task.to_dict(),
            "solver_results": solver_results,
            "scoring": {
                "schema_version": 140,
                "benchmark_profile": str(benchmark_profile),
                "max_trials": int(max_trials),
                "status": str(status),
                "failure_kind": str(failure_kind),
                "test_case_results": test_case_results,
            },
        }
        task_path = per_task_dir / f"{_sanitize_task_id(task.task_id)}.json"
        _write_once_json(task_path, per_task_obj)

        per_task_rows.append(
            {
                "schema_version": 140,
                "kind": "arc_per_task_manifest_row_v140",
                "task_id": str(task.task_id),
                "status": str(status),
                "failure_kind": str(failure_kind),
                "benchmark_profile": str(benchmark_profile),
                "max_trials": int(max_trials),
                "solver_statuses": [str(r.get("status") or "") for r in solver_results],
            }
        )

        for idx, res in enumerate(solver_results):
            trace = res.get("trace") if isinstance(res.get("trace"), dict) else {}
            for tp in trace.get("trace_programs", []) if isinstance(trace.get("trace_programs"), list) else []:
                if isinstance(tp, dict):
                    trace_rows.append(
                        {
                            "schema_version": 140,
                            "kind": "arc_trace_candidate_v140",
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

    with open(per_task_manifest_path, "x", encoding="utf-8") as f:
        for row in per_task_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    with open(trace_candidates_path, "x", encoding="utf-8") as f:
        for row in trace_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    clusters_top = sorted(((k, int(v)) for k, v in clusters.items()), key=lambda kv: (-int(kv[1]), str(kv[0])))[:30]
    clusters_rows = [{"key": str(k), "count": int(v), "failure_kind": str(k).split("|", 1)[0]} for k, v in clusters_top]

    eval_obj: Dict[str, Any] = {
        "schema_version": 140,
        "kind": "arc_eval_v140",
        "tasks_total": int(tasks_total),
        "tasks_solved_by_k": {str(k): int(solved_by_k[k]) for k in ("1", "2", "3")},
        "tasks_unknown_by_k": {str(k): int(unknown_by_k[k]) for k in ("1", "2", "3")},
        "tasks_failed_by_k": {str(k): int(failed_by_k[k]) for k in ("1", "2", "3")},
        "solve_rate_by_k": {
            str(k): float(float(solved_by_k[k]) / float(tasks_total) if tasks_total else 0.0) for k in ("1", "2", "3")
        },
        "benchmark_profile": str(benchmark_profile),
        "max_trials": int(max_trials),
        "official_k": int(max_trials),
        "failure_counts": {str(k): int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "clusters_top": clusters_rows,
        "max_depth": int(max_depth),
        "max_programs": int(max_programs),
    }
    eval_obj["eval_sig"] = sha256_hex(canonical_json_dumps(eval_obj).encode("utf-8"))
    _write_once_json(out_dir / "eval.json", eval_obj)

    summary_obj: Dict[str, Any] = {
        "schema_version": 140,
        "kind": "arc_summary_v140",
        "arc_root": str(arc_root),
        "split": str(split),
        "limit": int(limit),
        "seed": int(seed),
        "benchmark_profile": str(benchmark_profile),
        "max_trials": int(max_trials),
        "max_depth": int(max_depth),
        "max_programs": int(max_programs),
        "tasks_total": int(tasks_total),
        "tasks_solved_by_k": {str(k): int(solved_by_k[k]) for k in ("1", "2", "3")},
        "failure_counts": {str(k): int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "eval_sig": str(eval_obj["eval_sig"]),
    }
    summary_obj["summary_sig"] = sha256_hex(canonical_json_dumps(summary_obj).encode("utf-8"))
    _write_once_json(out_dir / "summary.json", summary_obj)
    _write_once_json(out_dir / "smoke_summary.json", {"schema_version": 140, "kind": "arc_smoke_summary_v140", "summary_sha256": summary_obj["summary_sig"]})

    backlog = _derive_backlog_v140(failure_counts=failure_counts)
    report_md = _build_report_markdown_v140(eval_obj=eval_obj, clusters=clusters_rows, backlog=backlog)
    _write_text_x(out_dir / "ARC_DIAG_REPORT_v140.md", report_md)

    snap_after = _repo_snapshot_sha256_v140(root=repo_root, exclude_paths=[out_dir])
    isolation = {
        "schema_version": 140,
        "kind": "isolation_check_v140",
        "repo_root": str(repo_root),
        "snapshot_before": str(snap_before),
        "snapshot_after": str(snap_after),
        "ok": bool(snap_before == snap_after),
    }
    _write_once_json(out_dir / "isolation_check_v140.json", isolation)
    if not bool(isolation["ok"]):
        raise SystemExit("isolation_failed")

    outputs_manifest = _build_outputs_manifest_v140(out_dir=out_dir)
    _write_once_json(out_dir / "outputs_manifest.json", outputs_manifest)

    return {
        "schema_version": 140,
        "kind": "arc_run_result_v140",
        "out_dir": str(out_dir),
        "summary_sha256": str(summary_obj["summary_sig"]),
        "outputs_manifest_sig": str(outputs_manifest["manifest_sig"]),
        "tasks_total": int(tasks_total),
        "tasks_solved_by_k": {str(k): int(solved_by_k[k]) for k in ("1", "2", "3")},
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
    ap.add_argument("--benchmark_profile", default="ARC_AGI1_PROFILE")
    ap.add_argument("--max_trials", type=int, default=0)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--max_programs", type=int, default=4000)
    args = ap.parse_args(argv)

    profile = str(args.benchmark_profile)
    mt = int(args.max_trials)
    effective_trials = int(mt) if int(mt) > 0 else int(_benchmark_profile_default_trials(profile))
    if effective_trials < 1 or effective_trials > 3:
        raise SystemExit("invalid_max_trials")

    out_base = Path(str(args.out_base))
    out_try1 = Path(str(out_base) + "_try1")
    out_try2 = Path(str(out_base) + "_try2")

    r1 = _run_one(
        arc_root=str(args.arc_root),
        split=str(args.split),
        limit=int(args.limit),
        seed=int(args.seed),
        out_dir=out_try1,
        benchmark_profile=profile,
        max_trials=int(effective_trials),
        max_depth=int(args.max_depth),
        max_programs=int(args.max_programs),
    )
    r2 = _run_one(
        arc_root=str(args.arc_root),
        split=str(args.split),
        limit=int(args.limit),
        seed=int(args.seed),
        out_dir=out_try2,
        benchmark_profile=profile,
        max_trials=int(effective_trials),
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
