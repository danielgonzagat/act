#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _install_sigint_sigterm_keyboard_interrupt() -> None:
    def handler(signum: int, frame: Any) -> None:  # type: ignore[override]
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _ensure_absent(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"worm_exists:{path}")


def _stable_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_once_text(path: Path, text: str) -> None:
    _ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


def _write_once_json(path: Path, obj: Any) -> None:
    _ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise SystemExit(f"tmp_exists:{tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _iter_per_task_files(per_task_dir: Path) -> Iterable[Path]:
    for p in sorted(per_task_dir.glob("*.json.json")):
        if p.is_file():
            yield p


def _copy_omega_events_subset(*, base_run_dir: Path, out_dir: Path, task_ids: Sequence[str]) -> Optional[str]:
    """
    Preserve born-from-failure provenance for downstream operator mining.

    Deep subset runs are synthesized runs (not produced by the main harness) and therefore do not
    naturally emit omega_events_v2.jsonl. We copy the base run's Ω events for the selected task_ids
    so miners can attach stable origin_clusters (rsig_*/family) deterministically.
    """
    src = (base_run_dir / "omega_events_v2.jsonl").resolve()
    if not src.is_file():
        return None
    want = {str(t) for t in (task_ids or []) if str(t)}
    if not want:
        return None

    out_path = (out_dir / "omega_events_v2.jsonl").resolve()
    if out_path.exists():
        raise SystemExit(f"worm_exists:{out_path}")
    out_lines: List[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        tid = str(obj.get("task_id") or "")
        if tid and tid in want:
            out_lines.append(line)
    _write_once_text(out_path, "\n".join(out_lines) + ("\n" if out_lines else ""))
    return str(out_path)


def _min_trace_loss(per_task_obj: Dict[str, Any]) -> Tuple[int, int]:
    # Return (min_loss_shape, min_loss_cells) over trace_programs, or (999,999) if absent.
    solver_results = per_task_obj.get("solver_results")
    if not isinstance(solver_results, list) or not solver_results:
        return (999, 999)
    sr0 = solver_results[0]
    if not isinstance(sr0, dict):
        return (999, 999)
    trace = sr0.get("trace")
    if not isinstance(trace, dict):
        return (999, 999)
    tps = trace.get("trace_programs")
    if not isinstance(tps, list) or not tps:
        return (999, 999)
    best = (999, 999)
    for row in tps:
        if not isinstance(row, dict):
            continue
        loss = row.get("loss")
        if not isinstance(loss, dict):
            continue
        ls = int(loss.get("shape") or 0)
        lc = int(loss.get("cells") or 0)
        if (ls, lc) < best:
            best = (ls, lc)
    return best


def _best_shape0_seed_steps(per_task_obj: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Extract a single warm-start seed program from a base run per_task object:
    choose the best trace_program with loss.shape==0 under a deterministic total order.
    """
    solver_results = per_task_obj.get("solver_results")
    if not isinstance(solver_results, list) or not solver_results:
        return None
    sr0 = solver_results[0]
    if not isinstance(sr0, dict):
        return None
    trace = sr0.get("trace")
    if not isinstance(trace, dict):
        return None
    tps = trace.get("trace_programs")
    if not isinstance(tps, list) or not tps:
        return None

    best_key: Optional[Tuple[int, int, int, str]] = None
    best_steps: Optional[List[Dict[str, Any]]] = None
    for row in tps:
        if not isinstance(row, dict):
            continue
        loss = row.get("loss")
        if not isinstance(loss, dict):
            continue
        if int(loss.get("shape") or 0) != 0:
            continue
        lc = int(loss.get("cells") or 0)
        depth = int(row.get("depth") or 0)
        cost = int(row.get("cost_bits") or 0)
        steps = row.get("steps")
        if not isinstance(steps, list):
            continue
        # Deterministic tiebreaker: canonical JSON of steps.
        steps_sig = _stable_json(steps)
        key = (int(lc), int(depth), int(cost), str(steps_sig))
        if best_key is None or key < best_key:
            best_key = key
            best_steps = [s for s in steps if isinstance(s, dict)]
    if not best_steps:
        return None
    return list(best_steps)


def _topk_shape0_seed_steps(per_task_obj: Dict[str, Any], *, k: int) -> List[List[Dict[str, Any]]]:
    """
    Extract up to K warm-start seed programs from a base run per_task object:
    choose the best trace_programs with loss.shape==0 under a deterministic total order.
    """
    kk = int(k)
    if kk <= 0:
        kk = 1

    solver_results = per_task_obj.get("solver_results")
    if not isinstance(solver_results, list) or not solver_results:
        return []
    sr0 = solver_results[0]
    if not isinstance(sr0, dict):
        return []
    trace = sr0.get("trace")
    if not isinstance(trace, dict):
        return []
    tps = trace.get("trace_programs")
    if not isinstance(tps, list) or not tps:
        return []

    rows: List[Tuple[Tuple[int, int, int, str], List[Dict[str, Any]]]] = []
    for row in tps:
        if not isinstance(row, dict):
            continue
        loss = row.get("loss")
        if not isinstance(loss, dict):
            continue
        if int(loss.get("shape") or 0) != 0:
            continue
        lc = int(loss.get("cells") or 0)
        depth = int(row.get("depth") or 0)
        cost = int(row.get("cost_bits") or 0)
        steps = row.get("steps")
        if not isinstance(steps, list):
            continue
        steps_norm = [s for s in steps if isinstance(s, dict)]
        if not steps_norm:
            continue
        steps_sig = _stable_json(steps_norm)
        key = (int(lc), int(depth), int(cost), str(steps_sig))
        rows.append((key, steps_norm))

    rows.sort(key=lambda x: x[0])
    out: List[List[Dict[str, Any]]] = []
    seen_sigs: set[str] = set()
    for _, st in rows:
        if len(out) >= kk:
            break
        sig = _stable_json(st)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        out.append(st)
    return out


@dataclass(frozen=True)
class _TaskPick:
    task_id: str
    failure_kind: str
    solver_status: str
    min_loss_shape: int
    min_loss_cells: int


def _load_failed_near_miss_tasks(*, base_run_dir: Path) -> List[_TaskPick]:
    per_task_dir = base_run_dir / "per_task"
    if not per_task_dir.is_dir():
        raise SystemExit(f"missing_per_task_dir:{per_task_dir}")

    picks: List[_TaskPick] = []
    for p in _iter_per_task_files(per_task_dir):
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            continue
        scoring = obj.get("scoring")
        if not isinstance(scoring, dict):
            continue
        solver_status = str(scoring.get("solver_status") or "")
        failure_kind = str(scoring.get("failure_kind") or "")
        if solver_status != "FAIL":
            continue
        # Focus on near-miss failures where shape is already reachable to maximize the chance that
        # a deeper run yields new full solutions and reusable macro traces.
        if failure_kind not in {"SEARCH_BUDGET_EXCEEDED", "MISSING_OPERATOR", "MISSING_ABSTRACTION"}:
            continue
        ls, lc = _min_trace_loss(obj)
        if int(ls) != 0:
            continue
        # Per-task files are named like "<task_id>.json.json" where task_id already ends with ".json".
        # Strip only the last ".json" suffix to recover the canonical task_id (keeps ".json").
        name = str(p.name)
        task_id = name
        if name.endswith(".json.json"):
            task_id = name[: -len(".json")]
        picks.append(
            _TaskPick(
                task_id=str(task_id),
                failure_kind=str(failure_kind),
                solver_status=str(solver_status),
                min_loss_shape=int(ls),
                min_loss_cells=int(lc),
            )
        )
    picks.sort(key=lambda t: (int(t.min_loss_cells), str(t.task_id)))
    return picks


def _solve_one(
    task: Tuple[str, Any, Any],
    *,
    max_depth: int,
    max_programs: int,
    solution_cost_slack_bits: int,
    timeout_s: int,
    enable_point_patch_repair: bool,
    point_patch_max_points: int,
) -> Dict[str, Any]:
    task_id, arc_task, seed_steps_raw = task
    from atos_core.arc_solver_v141 import SolveConfigV141, solve_arc_task_v141

    class _SolveTimeout(BaseException):
        pass

    # Per-task wall-time guard (prevents long-tail stalls).
    alarm_seconds = int(timeout_s)
    if alarm_seconds < 0:
        alarm_seconds = 0

    old_handler = None

    def _alarm_handler(signum: int, frame: Any) -> None:  # type: ignore[override]
        raise _SolveTimeout(f"timeout_s:{alarm_seconds}")

    # NOTE: SolveConfigV141 currently does not support warm-start seed programs. We keep the
    # seed_steps extraction pipeline (selection.json) for audit/debug, but do not apply it here.
    _ = seed_steps_raw

    cfg = SolveConfigV141(
        max_depth=int(max_depth),
        max_programs=int(max_programs),
        trace_program_limit=200,
        max_ambiguous_outputs=3,
        solution_cost_slack_bits=int(solution_cost_slack_bits),
        macro_templates=tuple(),
        macro_try_on_fail_only=False,
        # For deep operator induction, prefer exposing "unreachable-under-caps" prefixes in trace_programs.
        enable_reachability_pruning=False,
        enable_repair_stage=True,
        enable_residual_stage=True,
        enable_refine_stage=True,
        enable_point_patch_repair=bool(enable_point_patch_repair),
        point_patch_max_points=int(point_patch_max_points),
    )
    # NOTE: ARC-AGI canonical tasks always have one test input; the test output may exist for scoring-only.
    try:
        if alarm_seconds > 0:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(alarm_seconds)
        res = solve_arc_task_v141(train_pairs=list(arc_task.train_pairs), test_in=arc_task.test_pairs[0][0], config=cfg)
        return {"task_id": str(task_id), "solver_res": res}
    except _SolveTimeout:
        return {
            "task_id": str(task_id),
            "solver_res": {
                "schema_version": 141,
                "kind": "arc_solver_result_v141",
                "status": "FAIL",
                "failure_reason": {"kind": "TIMEOUT", "details": {"timeout_s": int(alarm_seconds)}},
                "trace": {"trace_programs": []},
            },
        }
    finally:
        if alarm_seconds > 0:
            try:
                signal.alarm(0)
            except Exception:
                pass
            if old_handler is not None:
                try:
                    signal.signal(signal.SIGALRM, old_handler)
                except Exception:
                    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_run_dir", required=True, help="A *_try1 dir produced by run_arc_scalpel_v141.py")
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=24, help="Max number of failed near-miss tasks to deep-run")
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--max_programs", type=int, default=12000)
    ap.add_argument("--solution_cost_slack_bits", type=int, default=16)
    ap.add_argument("--timeout_s", type=int, default=600, help="Per-task wall-time limit (0 disables)")
    ap.add_argument(
        "--seed_k",
        type=int,
        default=4,
        help=(
            "Record up to K best shape==0 seed programs per task in selection.json (audit/debug only; "
            "current SolveConfigV141 does not support warm-start seeds)."
        ),
    )
    ap.add_argument("--enable_point_patch_repair", action="store_true")
    ap.add_argument("--point_patch_max_points", type=int, default=18)
    args = ap.parse_args()

    _install_sigint_sigterm_keyboard_interrupt()

    base_run_dir = Path(str(args.base_run_dir)).resolve()
    out_dir = Path(str(args.out_base) + "_try1").resolve()
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    input_dir = out_dir / "input"
    per_task_dir = out_dir / "per_task"
    input_dir.mkdir(parents=True, exist_ok=False)
    per_task_dir.mkdir(parents=True, exist_ok=False)

    # Load candidate tasks from the base run (failures only).
    picks = _load_failed_near_miss_tasks(base_run_dir=base_run_dir)
    if not picks:
        # WORM-friendly: write a minimal summary and exit 0 so orchestration scripts can skip.
        _write_once_text(out_dir / "progress.log", "")
        _write_once_json(
            input_dir / "selection.json",
            {
                "schema_version": 144,
                "kind": "arc_deep_selection_v144",
                "base_run_dir": str(base_run_dir),
                "selected": [],
                "note": "no_failed_near_miss_tasks",
            },
        )
        summary = {
            "schema_version": 144,
            "kind": "arc_deep_summary_v144",
            "base_run_dir": str(base_run_dir),
            "tasks_total": 0,
            "tasks_solved": 0,
            "failure_counts": {},
            "max_depth": int(args.max_depth),
            "max_programs": int(args.max_programs),
            "solution_cost_slack_bits": int(args.solution_cost_slack_bits),
            "note": "no_failed_near_miss_tasks",
        }
        _write_once_json(out_dir / "summary.json", summary)
        print(f"[done] summary: {out_dir}/summary.json", file=sys.stderr)
        return
    picks = picks[: int(max(1, int(args.limit)))]

    # Load canonical tasks from the base run input to ensure identical task parsing.
    base_input = base_run_dir / "input" / "arc_tasks_canonical_v141.jsonl"
    if not base_input.is_file():
        raise SystemExit(f"missing_base_input:{base_input}")

    from atos_core.arc_loader_v141 import ArcTaskV141, iter_canonical_tasks_v141
    from atos_core.act import canonical_json_dumps

    task_map: Dict[str, ArcTaskV141] = {}
    for t in iter_canonical_tasks_v141(str(base_input)):
        task_map[str(t.task_id)] = t

    selected: List[Tuple[str, ArcTaskV141, Any]] = []
    missing: List[str] = []
    for p in picks:
        tt = task_map.get(str(p.task_id))
        if tt is None:
            missing.append(str(p.task_id))
            continue
        # Load warm-start seed steps from base per_task file.
        per_task_path = (base_run_dir / "per_task" / f"{p.task_id}.json").resolve()
        seed_steps: Any = None
        if per_task_path.is_file():
            base_obj = json.loads(per_task_path.read_text(encoding="utf-8"))
            if isinstance(base_obj, dict):
                seeds = _topk_shape0_seed_steps(base_obj, k=int(args.seed_k))
                if seeds:
                    seed_steps = seeds
        selected.append((str(p.task_id), tt, seed_steps))
    if missing:
        raise SystemExit(f"missing_tasks_in_base_input:{len(missing)}")
    if not selected:
        raise SystemExit("no_selected_tasks")

    # Write selected tasks canonical jsonl (WORM).
    sel_jsonl = input_dir / "arc_tasks_canonical_v141.jsonl"
    _ensure_absent(sel_jsonl)
    with open(sel_jsonl, "x", encoding="utf-8") as f:
        for task_id, t, _ in selected:
            f.write(canonical_json_dumps(t.to_dict()) + "\n")

    _write_once_json(
        input_dir / "selection.json",
        {
            "schema_version": 144,
            "kind": "arc_deep_selection_v144",
            "base_run_dir": str(base_run_dir),
            "selected": [
                {
                    "task_id": str(p.task_id),
                    "failure_kind": str(p.failure_kind),
                    "min_loss_shape": int(p.min_loss_shape),
                    "min_loss_cells": int(p.min_loss_cells),
                }
                for p in picks
            ],
        },
    )
    _write_once_text(input_dir / "base_input_sha256.txt", _sha256_file(base_input) + "\n")

    # Copy Ω provenance (born-from-failure) for the selected tasks into this deep run dir so
    # operator miners can attach stable origin_clusters.
    _copy_omega_events_subset(base_run_dir=base_run_dir, out_dir=out_dir, task_ids=[tid for tid, _t, _s in selected])

    progress_path = out_dir / "progress.log"
    _write_once_text(progress_path, "")

    jobs = int(args.jobs)
    if jobs <= 0:
        jobs = int(os.cpu_count() or 1)
    if jobs < 1:
        jobs = 1

    started = time.monotonic()
    done = 0
    solved = 0
    failures: Dict[str, int] = {}

    def log_progress(obj: Dict[str, Any]) -> None:
        line = _stable_json(obj)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, file=sys.stderr, flush=True)

    log_progress(
        {
            "done": 0,
            "elapsed_s": 0.0,
            "last_status": "",
            "last_task": "",
            "solved": 0,
            "total": int(len(selected)),
            "phase": "start",
        }
    )

    # Parallel solve across selected tasks.
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = []
        for task_id, t, seed_steps in selected:
            futs.append(
                ex.submit(
                    _solve_one,
                    (task_id, t, seed_steps),
                    max_depth=int(args.max_depth),
                    max_programs=int(args.max_programs),
                    solution_cost_slack_bits=int(args.solution_cost_slack_bits),
                    timeout_s=int(args.timeout_s),
                    enable_point_patch_repair=bool(args.enable_point_patch_repair),
                    point_patch_max_points=int(args.point_patch_max_points),
                )
            )

        for fut in as_completed(futs):
            item = fut.result()
            task_id = str(item.get("task_id") or "")
            solver_res = item.get("solver_res")
            if not isinstance(solver_res, dict):
                continue
            done += 1
            status = str(solver_res.get("status") or "")
            if status == "SOLVED":
                solved += 1
            else:
                fk = str((solver_res.get("failure_reason") or {}).get("kind") or "FAIL")
                failures[fk] = int(failures.get(fk, 0)) + 1

            # Store a minimal per-task envelope compatible with macro mining.
            per_task_obj = {
                "kind": "arc_per_task_v141",
                "schema_version": 141,
                "task": task_id,
                "solver_results": [solver_res],
                "scoring": {"solver_status": status},
            }
            # Match run_arc_scalpel_v141.py naming: task_id already ends with ".json",
            # so appending ".json" yields "<id>.json.json".
            out_path = per_task_dir / f"{task_id}.json"
            _ensure_absent(out_path)
            out_path.write_text(_stable_json(per_task_obj) + "\n", encoding="utf-8")

            log_progress(
                {
                    "done": int(done),
                    "elapsed_s": float(time.monotonic() - started),
                    "last_status": str(status),
                    "last_task": str(task_id),
                    "solved": int(solved),
                    "total": int(len(selected)),
                }
            )

    summary = {
        "schema_version": 144,
        "kind": "arc_deep_summary_v144",
        "base_run_dir": str(base_run_dir),
        "tasks_total": int(len(selected)),
        "tasks_solved": int(solved),
        "failure_counts": dict(sorted(failures.items(), key=lambda kv: str(kv[0]))),
        "max_depth": int(args.max_depth),
        "max_programs": int(args.max_programs),
        "solution_cost_slack_bits": int(args.solution_cost_slack_bits),
        "selected_input_sha256": _sha256_file(sel_jsonl),
    }
    _write_once_json(out_dir / "summary.json", summary)

    print(f"[done] summary: {out_dir}/summary.json", file=sys.stderr)


if __name__ == "__main__":
    main()
