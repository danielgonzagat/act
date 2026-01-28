#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class _ArcTaskTimeoutV141(BaseException):
    pass


_WORKER_CFG_PAYLOAD_V141: Optional[Dict[str, Any]] = None


def _worker_init_v141(cfg_payload: Dict[str, Any]) -> None:
    global _WORKER_CFG_PAYLOAD_V141
    _WORKER_CFG_PAYLOAD_V141 = dict(cfg_payload)


def _install_sigint_sigterm_keyboard_interrupt_v141() -> None:
    def handler(signum: int, frame: Any) -> None:  # type: ignore[override]
        # Ensure follow-up signals do not interrupt cleanup paths (atexit, shutdown hooks).
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _snapshot_processpool_workers_v141(ex: concurrent.futures.ProcessPoolExecutor) -> List[Any]:
    procs = getattr(ex, "_processes", None)
    if not isinstance(procs, dict):
        return []
    return [p for p in procs.values() if p is not None]


def _terminate_processpool_workers_v141(ex: concurrent.futures.ProcessPoolExecutor, *, procs_snapshot: Optional[List[Any]] = None) -> None:
    """
    Best-effort termination of worker processes.

    Required for watchdog mode (no_progress_timeout_s): we must not hang on shutdown
    with still-running tasks, and we must not leak orphan workers.
    """
    procs = list(procs_snapshot or _snapshot_processpool_workers_v141(ex))
    if not procs:
        return

    # Phase 1: terminate.
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass

    # Phase 2: wait briefly for exit.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        alive_any = False
        for p in procs:
            try:
                if p.is_alive():
                    alive_any = True
                    break
            except Exception:
                continue
        if not alive_any:
            break
        time.sleep(0.05)

    # Phase 3: kill remaining.
    for p in procs:
        try:
            if not p.is_alive():
                continue
        except Exception:
            # If we cannot check liveness, still attempt a hard kill by pid.
            pass
        try:
            p.kill()
            continue
        except Exception:
            pass
        try:
            if getattr(p, "pid", None):
                os.kill(int(p.pid), signal.SIGKILL)
        except Exception:
            pass


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


def _write_crash_artifact_v141(*, out_dir: Path, stage: str, err: BaseException) -> None:
    """
    Best-effort crash artifact for debugging long-running ARC loops.

    This is WORM-safe (write-once) and only created on exceptions; it does not affect determinism
    in successful runs.
    """
    try:
        body = {
            "schema_version": 141,
            "kind": "arc_run_crash_v141",
            "stage": str(stage),
            "error": {
                "type": str(err.__class__.__name__),
                "message": str(err),
            },
            "traceback": traceback.format_exc().splitlines()[-200:],
        }
        p = out_dir / "CRASH_v141.json"
        if not p.exists():
            _write_once_json(p, body)
    except Exception:
        # Never let crash reporting mask the original exception.
        pass


def _excluded_dir_parts_v141() -> set:
    return {
        ".git",
        "__pycache__",
        ".pycache",
        "artifacts",
        "results",
        "tmp",
        "external_world",
        "external_world_v122",
        "external_world_v122_try2",
        "external_world_v122_try3",
        "external_world_v122_try4",
        "external_world_v122_try5",
        "external_world_v122_try6",
    }


def _repo_snapshot_sha256_v141(*, root: Path, exclude_paths: Sequence[Path]) -> str:
    excluded = _excluded_dir_parts_v141()
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
    body = {"schema_version": 141, "kind": "repo_snapshot_v141", "files": rows}
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


def _score_test_case_all_k_v141(
    *,
    solver_res: Dict[str, Any],
    want_grid: Optional[Sequence[Sequence[int]]],
    max_trials: int,
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
            "scores_by_k": {},
        }

    # Candidate outputs from solver.
    preds = solver_res.get("predicted_grids")
    if isinstance(preds, list) and preds:
        candidates = [p for p in preds if isinstance(p, list)]
    else:
        pred = solver_res.get("predicted_grid")
        candidates = [pred] if isinstance(pred, list) else []

    scores_by_k: Dict[str, Any] = {}
    any_ok = False
    for k in (1, 2, 3):
        kk = int(k)
        if kk <= 0:
            continue
        if kk > int(max_trials):
            continue
        ok = False
        for cand in candidates[:kk]:
            if _grid_equal(cand, want_grid):
                ok = True
                break
        scores_by_k[str(kk)] = {"k": int(kk), "ok": bool(ok)}
        any_ok = bool(any_ok) or bool(ok)
    # failure_kind for aggregates uses solver status/failure_reason.
    fk = ""
    fr = solver_res.get("failure_reason")
    if isinstance(fr, dict):
        fk = str(fr.get("kind") or "")
    # If the solver produced outputs but none match the test target, mark explicitly.
    if not any_ok and not fk and candidates:
        fk = "WRONG_OUTPUT"
    return {
        "solver_status": status if status in {"SOLVED", "UNKNOWN"} else "FAIL",
        "scored": True,
        "failure_kind": fk,
        "scores_by_k": scores_by_k,
    }


def _summarize_scores_v141(rows: List[Dict[str, Any]], *, max_trials: int) -> Dict[str, Any]:
    total = int(len(rows))
    tasks_solved_by_k: Dict[str, int] = {str(k): 0 for k in range(1, int(max_trials) + 1)}
    for r in rows:
        s = r.get("scoring") if isinstance(r.get("scoring"), dict) else {}
        by_k = s.get("scores_by_k") if isinstance(s.get("scores_by_k"), dict) else {}
        # For unscored tasks (e.g. internal sample suites with no test outputs),
        # count SOLVED based on solver_status for all k.
        if not bool(s.get("scored")):
            if str(s.get("solver_status") or "") == "SOLVED":
                for k in tasks_solved_by_k.keys():
                    tasks_solved_by_k[str(k)] = int(tasks_solved_by_k[str(k)]) + 1
            continue
        for k in tasks_solved_by_k.keys():
            ok = False
            item = by_k.get(k) if isinstance(by_k.get(k), dict) else {}
            ok = bool(item.get("ok")) if isinstance(item, dict) else False
            if ok:
                tasks_solved_by_k[str(k)] = int(tasks_solved_by_k[str(k)]) + 1
    solved_kmax = int(tasks_solved_by_k.get(str(max_trials), 0))
    solve_rate = float(solved_kmax) / float(total) if total else 0.0
    return {"tasks_total": int(total), "tasks_solved_by_k": tasks_solved_by_k, "solve_rate_kmax": float(solve_rate)}


def _summarize_program_usage_v141(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Lightweight "hierarchy / concept usage" telemetry (deterministic).

    This is intentionally simple and audit-friendly: it only inspects the solver's chosen
    unique train-perfect program (status == SOLVED) and counts concept/macro calls.
    """
    solved = 0
    solved_with_concept = 0
    solved_with_multistep_concept = 0
    solved_with_macro = 0

    concept_calls = 0
    multistep_concept_calls = 0
    macro_calls = 0

    total_steps = 0
    total_cost_bits = 0

    for row in rows:
        solver_results = row.get("solver_results")
        if not isinstance(solver_results, list) or not solver_results:
            continue
        sr0 = solver_results[0] if isinstance(solver_results[0], dict) else {}
        if str(sr0.get("status") or "") != "SOLVED":
            continue

        solved += 1
        total_cost_bits += int(sr0.get("program_cost_bits") or 0)

        steps = sr0.get("program_steps")
        if not isinstance(steps, list):
            steps = []
        total_steps += int(len(steps))

        any_concept = False
        any_multistep_concept = False
        any_macro = False

        for st in steps:
            if not isinstance(st, dict):
                continue
            op_id = str(st.get("op_id") or "")
            if op_id == "concept_call":
                any_concept = True
                concept_calls += 1
                args = st.get("args") if isinstance(st.get("args"), dict) else {}
                op_ids = args.get("op_ids")
                if isinstance(op_ids, list):
                    n = len([x for x in op_ids if str(x)])
                    if n > 1:
                        any_multistep_concept = True
                        multistep_concept_calls += 1
            elif op_id == "macro_call":
                any_macro = True
                macro_calls += 1

        if any_concept:
            solved_with_concept += 1
        if any_multistep_concept:
            solved_with_multistep_concept += 1
        if any_macro:
            solved_with_macro += 1

    avg_steps = float(total_steps) / float(solved) if solved else 0.0
    avg_cost_bits = float(total_cost_bits) / float(solved) if solved else 0.0
    hur = float(solved_with_multistep_concept) / float(solved) if solved else 0.0
    return {
        "solved_unique": int(solved),
        "solved_with_concept_call": int(solved_with_concept),
        "solved_with_multistep_concept_call": int(solved_with_multistep_concept),
        "solved_with_macro_call": int(solved_with_macro),
        "concept_call_steps_total": int(concept_calls),
        "concept_call_multistep_steps_total": int(multistep_concept_calls),
        "macro_call_steps_total": int(macro_calls),
        "avg_program_steps_solved": float(avg_steps),
        "avg_program_cost_bits_solved": float(avg_cost_bits),
        "hierarchical_utilization_ratio": float(hur),
    }


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl_x(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        for row in rows:
            f.write(_stable_json(row) + "\n")


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _solve_one_task_worker_v141(args: Tuple[Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Worker for parallel task solving.

    Must remain top-level (picklable) and avoid repo mutations.
    """
    payload, cfg_payload = args
    task_id = str(payload.get("task_id") or "")
    train_pairs = payload.get("train_pairs")
    test_in = payload.get("test_in")
    if not isinstance(train_pairs, (list, tuple)) or not isinstance(test_in, (list, tuple)):
        raise RuntimeError("bad_task_payload_v141")

    from atos_core.arc_solver_v141 import SolveConfigV141, solve_arc_task_v141

    macro_templates: Tuple[Dict[str, Any], ...] = tuple()
    mt = cfg_payload.get("macro_templates")
    if isinstance(mt, list):
        macro_templates = tuple([m for m in mt if isinstance(m, dict)])

    concept_templates: Tuple[Dict[str, Any], ...] = tuple()
    ct = cfg_payload.get("concept_templates")
    if isinstance(ct, list):
        concept_templates = tuple([c for c in ct if isinstance(c, dict)])

    timeout_s = float(cfg_payload.get("task_timeout_s") or 0.0)
    if timeout_s < 0.0:
        timeout_s = 0.0

    enable_point_patch_repair = bool(cfg_payload.get("enable_point_patch_repair", False))
    point_patch_max_points = int(cfg_payload.get("point_patch_max_points", 12))
    if point_patch_max_points < 0:
        point_patch_max_points = 0
    enable_repair_stage = bool(cfg_payload.get("enable_repair_stage", True))
    enable_residual_stage = bool(cfg_payload.get("enable_residual_stage", True))
    enable_refine_stage = bool(cfg_payload.get("enable_refine_stage", True))
    macro_try_on_fail_only = bool(cfg_payload.get("macro_try_on_fail_only", True))
    abstraction_pressure = bool(cfg_payload.get("abstraction_pressure", False))
    enable_reachability_pruning = bool(cfg_payload.get("enable_reachability_pruning", True))

    cfg = SolveConfigV141(
        max_depth=int(cfg_payload.get("max_depth", 4)),
        max_programs=int(cfg_payload.get("max_programs", 4000)),
        trace_program_limit=int(cfg_payload.get("trace_program_limit", 80)),
        max_ambiguous_outputs=int(cfg_payload.get("max_ambiguous_outputs", 3)),
        solution_cost_slack_bits=int(cfg_payload.get("solution_cost_slack_bits", 0)),
        macro_templates=macro_templates,
        concept_templates=concept_templates,
        abstraction_pressure=bool(abstraction_pressure),
        macro_try_on_fail_only=bool(macro_try_on_fail_only),
        enable_reachability_pruning=bool(enable_reachability_pruning),
        macro_propose_max_depth=int(cfg_payload.get("macro_propose_max_depth", 0)),
        macro_max_templates=int(cfg_payload.get("macro_max_templates", 24)),
        macro_max_instantiations=int(cfg_payload.get("macro_max_instantiations", 10)),
        macro_max_branch_per_op=int(cfg_payload.get("macro_max_branch_per_op", 10)),
        enable_repair_stage=bool(enable_repair_stage),
        enable_residual_stage=bool(enable_residual_stage),
        enable_refine_stage=bool(enable_refine_stage),
        enable_point_patch_repair=bool(enable_point_patch_repair),
        point_patch_max_points=int(point_patch_max_points),
    )

    def _exception_result(err: BaseException) -> Dict[str, Any]:
        return {
            "schema_version": 141,
            "kind": "arc_solver_result_v141",
            "status": "FAIL",
            "program_sig": "",
            "predicted_grid_hash": "",
            "failure_reason": {
                "kind": "EXCEPTION",
                "details": {"type": str(err.__class__.__name__), "message": str(err)},
            },
            "trace": {"exception_type": str(err.__class__.__name__), "exception": str(err)},
        }

    if timeout_s > 0.0:
        import signal

        def _handler(_signum: int, _frame: Any) -> None:  # type: ignore[valid-type]
            raise _ArcTaskTimeoutV141()

        prev = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
        try:
            solver_res = solve_arc_task_v141(train_pairs=list(train_pairs), test_in=test_in, config=cfg)
        except _ArcTaskTimeoutV141:
            solver_res = {
                "schema_version": 141,
                "kind": "arc_solver_result_v141",
                "status": "FAIL",
                "program_sig": "",
                "predicted_grid_hash": "",
                "failure_reason": {"kind": "SEARCH_BUDGET_EXCEEDED", "details": {"timeout_s": float(timeout_s)}},
                "trace": {"timeout_s": float(timeout_s)},
            }
        except Exception as e:
            solver_res = _exception_result(e)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, prev)
    else:
        try:
            solver_res = solve_arc_task_v141(train_pairs=list(train_pairs), test_in=test_in, config=cfg)
        except Exception as e:
            solver_res = _exception_result(e)
    return {"task_id": task_id, "solver_res": solver_res}


def _solve_one_task_worker_inited_v141(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker entrypoint using a process-global cfg payload (initializer).

    This avoids pickling large concept/macro banks for every submitted task when running with many jobs.
    """
    cfg_payload = _WORKER_CFG_PAYLOAD_V141 or {}
    return _solve_one_task_worker_v141((payload, cfg_payload))


def _make_diag_report_v141(*, summary_obj: Dict[str, Any], eval_obj: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# ARC_DIAG_REPORT_v141")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(_stable_json(summary_obj))
    lines.append("")
    lines.append("## Failure Counts")
    lines.append("")
    fc = eval_obj.get("failure_counts") if isinstance(eval_obj.get("failure_counts"), dict) else {}
    lines.append(_stable_json(fc))
    lines.append("")
    return "\n".join(lines) + "\n"


def _outputs_manifest_v141(*, run_dir: Path) -> Dict[str, Any]:
    # Deterministic listing of key artifacts (sha256).
    per_task_dir = run_dir / "per_task"
    per_task_files: List[str] = []
    if per_task_dir.is_dir():
        for p in per_task_dir.iterdir():
            if p.is_file() and p.name.endswith(".json"):
                per_task_files.append(p.name)
    per_task_files.sort()
    manifest: Dict[str, Any] = {
        "schema_version": 141,
        "kind": "outputs_manifest_v141",
        "files": [],
    }
    files: List[Dict[str, Any]] = []

    def add(rel: str) -> None:
        p = run_dir / rel
        if not p.exists():
            return
        files.append({"path": str(rel), "sha256": _sha256_file(p)})

    add("summary.json")
    add("eval.json")
    add("isolation_check_v141.json")
    add("OMEGA_DEAD_v2.json")
    add("omega_events_v2.jsonl")
    add("trace_candidates.jsonl")
    add("per_task_manifest.jsonl")
    add("ARC_DIAG_REPORT_v141.md")
    add("input/arc_manifest_v141.json")
    add("input/arc_tasks_canonical_v141.jsonl")
    add("input/arc_macro_templates_v143.jsonl")
    add("input/arc_concept_templates_v146.jsonl")
    add("input/omega_state_v2.json")
    for fn in per_task_files:
        add(f"per_task/{fn}")
    files.sort(key=lambda r: str(r["path"]))
    manifest["files"] = files
    from atos_core.act import canonical_json_dumps, sha256_hex

    manifest["outputs_manifest_sig"] = sha256_hex(canonical_json_dumps(manifest).encode("utf-8"))
    return manifest


def _run_once_v141(
    *,
    out_dir: Path,
    arc_root: str,
    split: Optional[str],
    limit: int,
    seed: int,
    jobs: int,
    max_depth: int,
    max_programs: int,
    solution_cost_slack_bits: int,
    max_trials: int,
    benchmark_profile: str,
    macro_templates_path: Optional[str],
    concept_templates_path: Optional[str],
    task_timeout_s: float,
    no_progress_timeout_s: float,
    resume: bool,
    enable_repair_stage: bool,
    enable_residual_stage: bool,
    enable_refine_stage: bool,
    macro_try_on_fail_only: bool,
    macro_propose_max_depth: int,
    macro_max_templates: int,
    macro_max_instantiations: int,
    macro_max_branch_per_op: int,
    abstraction_pressure: bool,
    enable_reachability_pruning: bool,
    enable_point_patch_repair: bool,
    point_patch_max_points: int,
    omega_enabled: bool,
    omega_state_in: Optional[str],
) -> Dict[str, Any]:
    resume = bool(resume)
    macro_propose_max_depth = max(0, int(macro_propose_max_depth))
    macro_max_templates = max(0, int(macro_max_templates))
    macro_max_instantiations = max(0, int(macro_max_instantiations))
    macro_max_branch_per_op = max(0, int(macro_max_branch_per_op))
    if resume:
        if not out_dir.is_dir():
            raise SystemExit(f"resume_missing_out_dir_v141:{out_dir}")
        if (out_dir / "summary.json").exists():
            raise SystemExit(f"resume_already_completed_v141:{out_dir}")
    else:
        _ensure_absent(out_dir)
        out_dir.mkdir(parents=True, exist_ok=False)
    _install_sigint_sigterm_keyboard_interrupt_v141()

    root = Path(__file__).resolve().parent.parent
    before_sig = _repo_snapshot_sha256_v141(root=root, exclude_paths=[out_dir])

    input_dir = out_dir / "input"
    input_jsonl = input_dir / "arc_tasks_canonical_v141.jsonl"
    input_manifest = input_dir / "arc_manifest_v141.json"
    input_macro_templates = input_dir / "arc_macro_templates_v143.jsonl"
    input_concept_templates = input_dir / "arc_concept_templates_v146.jsonl"
    input_omega_state = input_dir / "omega_state_v2.json"

    from atos_core.arc_loader_v141 import iter_canonical_tasks_v141, write_arc_canonical_jsonl_v141

    if resume:
        if not input_jsonl.is_file():
            raise SystemExit(f"resume_missing_input_jsonl_v141:{input_jsonl}")
        if not input_manifest.is_file():
            raise SystemExit(f"resume_missing_input_manifest_v141:{input_manifest}")
        manifest_obj = _read_json(input_manifest)
        if not isinstance(manifest_obj, dict):
            raise SystemExit(f"resume_bad_manifest_v141:{input_manifest}")
        # Minimal config match guard: resume must target the same dataset slice.
        # Note: arc_manifest_v141 intentionally does not include arc_root path (to avoid absolute paths).
        want = {
            "split": str(split) if split is not None else None,
            "limit": int(limit),
            "seed": int(seed),
            "selection_mode": "shuffled" if int(limit) > 0 else "sorted",
        }
        got = {
            "split": manifest_obj.get("split") if "split" in manifest_obj else None,
            "limit": int(manifest_obj.get("limit") or 0),
            "seed": (manifest_obj.get("seed") if "seed" in manifest_obj else None),
            "selection_mode": str(manifest_obj.get("selection_mode") or "sorted"),
        }
        if want != got:
            raise SystemExit(f"resume_manifest_mismatch_v141:want={want},got={got}")
    else:
        manifest_obj = write_arc_canonical_jsonl_v141(
            arc_root=str(arc_root),
            split=str(split) if split is not None else None,
            limit=int(limit),
            seed=int(seed),
            out_jsonl=input_jsonl,
            out_manifest=input_manifest,
        )

    from atos_core.arc_solver_v141 import SolveConfigV141, solve_arc_task_v141

    macro_templates_rows: List[Dict[str, Any]] = []
    macro_templates_sig = ""
    if resume:
        # Resume uses the WORM input copy if present (to avoid config drift).
        if input_macro_templates.is_file():
            macro_templates_sig = _sha256_file(input_macro_templates)
            if macro_templates_path:
                mt_path = Path(str(macro_templates_path))
                if not mt_path.is_file():
                    raise SystemExit(f"missing_macro_templates:{mt_path}")
                if _sha256_file(mt_path) != macro_templates_sig:
                    raise SystemExit("resume_macro_templates_mismatch_v141")
            for line in input_macro_templates.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    macro_templates_rows.append(obj)
            macro_templates_rows.sort(
                key=lambda r: (str(r.get("macro_id") or r.get("operator_id") or ""), _stable_json(r))
            )
        else:
            if macro_templates_path:
                raise SystemExit("resume_cannot_add_macro_templates_v141")
    else:
        if macro_templates_path:
            mt_path = Path(str(macro_templates_path))
            if not mt_path.is_file():
                raise SystemExit(f"missing_macro_templates:{mt_path}")
            _ensure_absent(input_macro_templates)
            input_macro_templates.parent.mkdir(parents=True, exist_ok=True)
            input_macro_templates.write_text(mt_path.read_text(encoding="utf-8"), encoding="utf-8")
            macro_templates_sig = _sha256_file(input_macro_templates)
            for line in input_macro_templates.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    macro_templates_rows.append(obj)
            macro_templates_rows.sort(
                key=lambda r: (str(r.get("macro_id") or r.get("operator_id") or ""), _stable_json(r))
            )

    concept_templates_rows: List[Dict[str, Any]] = []
    concept_templates_sig = ""
    if resume:
        if input_concept_templates.is_file():
            concept_templates_sig = _sha256_file(input_concept_templates)
            if concept_templates_path:
                ct_path = Path(str(concept_templates_path))
                if not ct_path.is_file():
                    raise SystemExit(f"missing_concept_templates:{ct_path}")
                if _sha256_file(ct_path) != concept_templates_sig:
                    raise SystemExit("resume_concept_templates_mismatch_v141")
            for line in input_concept_templates.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    concept_templates_rows.append(obj)
            concept_templates_rows.sort(key=lambda r: (str(r.get("concept_id") or ""), _stable_json(r)))
        else:
            if concept_templates_path:
                raise SystemExit("resume_cannot_add_concept_templates_v141")
    else:
        if concept_templates_path:
            ct_path = Path(str(concept_templates_path))
            if not ct_path.is_file():
                raise SystemExit(f"missing_concept_templates:{ct_path}")
            _ensure_absent(input_concept_templates)
            input_concept_templates.parent.mkdir(parents=True, exist_ok=True)
            input_concept_templates.write_text(ct_path.read_text(encoding="utf-8"), encoding="utf-8")
            concept_templates_sig = _sha256_file(input_concept_templates)
            for line in input_concept_templates.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    concept_templates_rows.append(obj)
            concept_templates_rows.sort(key=lambda r: (str(r.get("concept_id") or ""), _stable_json(r)))

    omega_state_obj = None
    omega_state_sig = ""
    max_depth_eff = int(max_depth)
    max_programs_eff = int(max_programs)
    if bool(omega_enabled):
        from atos_core.omega_v2 import OmegaStateV2, apply_omega_caps

        if resume:
            if input_omega_state.is_file():
                omega_state_sig = _sha256_file(input_omega_state)
                if omega_state_in:
                    op = Path(str(omega_state_in))
                    if not op.is_file():
                        raise SystemExit(f"missing_omega_state_in:{op}")
                    if _sha256_file(op) != omega_state_sig:
                        raise SystemExit("resume_omega_state_mismatch_v2")
                omega_state_obj = OmegaStateV2.from_path(input_omega_state)
            else:
                if omega_state_in:
                    raise SystemExit("resume_cannot_add_omega_state_v2")
        else:
            if omega_state_in:
                op = Path(str(omega_state_in))
                if not op.is_file():
                    raise SystemExit(f"missing_omega_state_in:{op}")
                _ensure_absent(input_omega_state)
                input_omega_state.parent.mkdir(parents=True, exist_ok=True)
                input_omega_state.write_text(op.read_text(encoding="utf-8"), encoding="utf-8")
                omega_state_sig = _sha256_file(input_omega_state)
                omega_state_obj = OmegaStateV2.from_path(input_omega_state)

        max_depth_eff, max_programs_eff = apply_omega_caps(
            want_max_depth=int(max_depth), want_max_programs=int(max_programs), state=omega_state_obj
        )

    cfg = SolveConfigV141(
        max_depth=int(max_depth_eff),
        max_programs=int(max_programs_eff),
        trace_program_limit=80,
        max_ambiguous_outputs=int(max_trials),
        solution_cost_slack_bits=int(solution_cost_slack_bits),
        macro_templates=tuple(macro_templates_rows),
        concept_templates=tuple(concept_templates_rows),
        abstraction_pressure=bool(abstraction_pressure),
        macro_try_on_fail_only=bool(macro_try_on_fail_only),
        enable_reachability_pruning=bool(enable_reachability_pruning),
        macro_propose_max_depth=int(macro_propose_max_depth),
        macro_max_templates=int(macro_max_templates),
        macro_max_instantiations=int(macro_max_instantiations),
        macro_max_branch_per_op=int(macro_max_branch_per_op),
        enable_repair_stage=bool(enable_repair_stage),
        enable_residual_stage=bool(enable_residual_stage),
        enable_refine_stage=bool(enable_refine_stage),
        enable_point_patch_repair=bool(enable_point_patch_repair),
        point_patch_max_points=int(point_patch_max_points),
    )
    tasks = list(iter_canonical_tasks_v141(str(input_jsonl)))
    omega_banned_tasks: List[str] = []
    omega_banned_families: List[str] = []
    if bool(omega_enabled) and omega_state_obj is not None:
        from atos_core.omega_v2 import arc_task_family_id

        banned = {str(x) for x in getattr(omega_state_obj, "banned_task_families", ()) if str(x)}
        omega_banned_families = sorted(list(banned))
        if banned:
            kept = []
            for t in tasks:
                fam = arc_task_family_id(t.to_dict())
                if fam in banned:
                    omega_banned_tasks.append(str(t.task_id))
                    continue
                kept.append(t)
            tasks = kept
        if not tasks:
            # Ω death: no reachable tasks remain under the permanently destroyed future subspace.
            _write_once_json(
                out_dir / "OMEGA_DEAD_v2.json",
                {
                    "schema_version": 2,
                    "kind": "omega_dead_v2",
                    "omega_state_in": str(omega_state_in or ""),
                    "omega_state_sig": str(omega_state_sig or ""),
                    "banned_task_families_total": int(len(omega_banned_families)),
                    "banned_task_families": omega_banned_families,
                    "limit": int(limit),
                    "split": str(split or ""),
                },
            )
            raise SystemExit("omega_dead_v2:no_reachable_tasks")
    want_by_id: Dict[str, Optional[Sequence[Sequence[int]]]] = {str(t.task_id): t.test_pairs[0][1] for t in tasks}
    tasks_by_id: Dict[str, Any] = {str(t.task_id): t for t in tasks}

    per_task_rows: List[Dict[str, Any]] = []
    trace_candidates_rows: List[Dict[str, Any]] = []
    scoring_rows: List[Dict[str, Any]] = []

    progress_path = out_dir / "progress.log"
    if not progress_path.exists():
        progress_path.write_text("", encoding="utf-8")

    per_task_dir = out_dir / "per_task"
    per_task_dir.mkdir(parents=True, exist_ok=True)

    def per_task_path(task_id: str) -> Path:
        safe = _sanitize_task_id(task_id)
        return per_task_dir / f"{safe}.json"

    def write_per_task_row(*, task_id: str, solver_res: Dict[str, Any], scoring: Dict[str, Any]) -> None:
        task = tasks_by_id.get(str(task_id))
        if task is None:
            return
        out_path = per_task_path(str(task.task_id))
        if out_path.exists():
            return
        row = {
            "schema_version": 141,
            "kind": "arc_per_task_v141",
            "task": task.to_dict(),
            "solver_results": [solver_res],
            "scoring": scoring,
        }
        _write_once_json(out_path, row)

    jobs_n = int(jobs)
    if jobs_n <= 0:
        jobs_n = int(os.cpu_count() or 1)
    if jobs_n < 1:
        jobs_n = 1

    # Parallel solve across tasks (pure, no shared state).
    solver_results: List[Dict[str, Any]] = []
    scoring_by_id: Dict[str, Dict[str, Any]] = {}
    failures_so_far: Dict[str, int] = {}
    solved_kmax = 0
    total_tasks = int(len(tasks))
    started = time.monotonic()
    last_progress = time.monotonic()

    # Resume support: count already completed tasks from per_task/ (WORM) without rerunning them.
    done = 0
    if resume:
        for t in tasks:
            if per_task_path(str(t.task_id)).exists():
                done += 1

    def log_progress(obj: Dict[str, Any]) -> None:
        line = _stable_json(obj)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, file=sys.stderr, flush=True)

    if jobs_n == 1:
        import signal

        def _timeout_result() -> Dict[str, Any]:
            return {
                "schema_version": 141,
                "kind": "arc_solver_result_v141",
                "status": "FAIL",
                "program_sig": "",
                "predicted_grid_hash": "",
                "failure_reason": {"kind": "SEARCH_BUDGET_EXCEEDED", "details": {"timeout_s": float(task_timeout_s)}},
                "trace": {"timeout_s": float(task_timeout_s)},
            }

        def _exception_result(err: BaseException) -> Dict[str, Any]:
            return {
                "schema_version": 141,
                "kind": "arc_solver_result_v141",
                "status": "FAIL",
                "program_sig": "",
                "predicted_grid_hash": "",
                "failure_reason": {"kind": "EXCEPTION", "details": {"type": str(err.__class__.__name__), "message": str(err)}},
                "trace": {"exception_type": str(err.__class__.__name__), "exception": str(err)},
            }

        def _handler(_signum: int, _frame: Any) -> None:  # type: ignore[valid-type]
            raise _ArcTaskTimeoutV141()

        prev_handler = signal.signal(signal.SIGALRM, _handler)
        for i, task in enumerate(tasks):
            if float(task_timeout_s) > 0.0:
                signal.setitimer(signal.ITIMER_REAL, float(task_timeout_s))
            try:
                solver_res = solve_arc_task_v141(
                    train_pairs=list(task.train_pairs),
                    test_in=task.test_pairs[0][0],
                    config=cfg,
                )
            except _ArcTaskTimeoutV141:
                solver_res = _timeout_result()
            except Exception as e:
                solver_res = _exception_result(e)
            finally:
                if float(task_timeout_s) > 0.0:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
            item = {"task_id": str(task.task_id), "solver_res": solver_res}
            solver_results.append(item)
            scoring = _score_test_case_all_k_v141(
                solver_res=solver_res, want_grid=want_by_id.get(str(task.task_id)), max_trials=int(max_trials)
            )
            scoring_by_id[str(task.task_id)] = scoring
            write_per_task_row(task_id=str(task.task_id), solver_res=solver_res, scoring=scoring)
            fk = str(scoring.get("failure_kind") or "")
            failures_so_far[fk] = int(failures_so_far.get(fk, 0)) + 1
            by_k = scoring.get("scores_by_k") if isinstance(scoring.get("scores_by_k"), dict) else {}
            ok_item = by_k.get(str(int(max_trials))) if isinstance(by_k.get(str(int(max_trials))), dict) else {}
            if bool(ok_item.get("ok")):
                solved_kmax += 1
            done += 1
            last_progress = time.monotonic()
            log_progress(
                {
                    "done": int(done),
                    "total": int(total_tasks),
                    "solved_kmax": int(solved_kmax),
                    "last_task_id": str(task.task_id),
                    "last_status": str(solver_res.get("status") or ""),
                    "last_failure": str(fk),
                    "elapsed_s": float(time.monotonic() - started),
                }
            )
        signal.signal(signal.SIGALRM, prev_handler)
    else:
        cfg_payload = {
            "max_depth": int(max_depth_eff),
            "max_programs": int(max_programs_eff),
            "trace_program_limit": 80,
            "max_ambiguous_outputs": int(max_trials),
            "solution_cost_slack_bits": int(solution_cost_slack_bits),
            "macro_templates": list(macro_templates_rows),
            "concept_templates": list(concept_templates_rows),
            "task_timeout_s": float(task_timeout_s),
            "enable_repair_stage": bool(enable_repair_stage),
            "enable_residual_stage": bool(enable_residual_stage),
            "enable_refine_stage": bool(enable_refine_stage),
            "macro_try_on_fail_only": bool(macro_try_on_fail_only),
            "macro_propose_max_depth": int(macro_propose_max_depth),
            "macro_max_templates": int(macro_max_templates),
            "macro_max_instantiations": int(macro_max_instantiations),
            "macro_max_branch_per_op": int(macro_max_branch_per_op),
            "abstraction_pressure": bool(abstraction_pressure),
            "enable_reachability_pruning": bool(enable_reachability_pruning),
            "enable_point_patch_repair": bool(enable_point_patch_repair),
            "point_patch_max_points": int(point_patch_max_points),
        }
        task_payloads: List[Dict[str, Any]] = []
        for task in tasks:
            if resume and per_task_path(str(task.task_id)).exists():
                continue
            task_payloads.append(
                {"task_id": str(task.task_id), "train_pairs": list(task.train_pairs), "test_in": task.test_pairs[0][0]}
            )

        ex = concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs_n,
            initializer=_worker_init_v141,
            initargs=(cfg_payload,),
        )
        try:
            def _exception_result(err: BaseException) -> Dict[str, Any]:
                return {
                    "schema_version": 141,
                    "kind": "arc_solver_result_v141",
                    "status": "FAIL",
                    "program_sig": "",
                    "predicted_grid_hash": "",
                    "failure_reason": {
                        "kind": "EXCEPTION",
                        "details": {"type": str(err.__class__.__name__), "message": str(err)},
                    },
                    "trace": {"exception_type": str(err.__class__.__name__), "exception": str(err)},
                }

            fut_by_task_id: Dict[concurrent.futures.Future, str] = {}
            for task_payload in task_payloads:
                task_id = str(task_payload.get("task_id") or "") if isinstance(task_payload, dict) else ""
                fut = ex.submit(_solve_one_task_worker_inited_v141, task_payload)
                fut_by_task_id[fut] = task_id
            pending = set(fut_by_task_id.keys())
            no_progress = float(no_progress_timeout_s or 0.0)
            stalled = False

            while pending:
                if no_progress > 0.0:
                    done_set, pending = concurrent.futures.wait(
                        pending, timeout=1.0, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    if not done_set:
                        if (time.monotonic() - last_progress) > no_progress:
                            stalled = True
                            break
                        continue
                    futs = list(done_set)
                else:
                    # Default behavior: block until futures complete.
                    fut = next(iter(concurrent.futures.as_completed(list(pending))))
                    pending.remove(fut)
                    futs = [fut]

                for fut in futs:
                    task_id_hint = str(fut_by_task_id.get(fut, "") or "")
                    try:
                        item = fut.result()
                    except Exception as e:
                        # Do not abort the entire run due to a single worker crash; record it as a task-level FAIL.
                        item = {"task_id": task_id_hint, "solver_res": _exception_result(e)}
                    solver_results.append(item)
                    task_id = str(item.get("task_id") or "")
                    solver_res = item.get("solver_res") if isinstance(item.get("solver_res"), dict) else {}
                    scoring = _score_test_case_all_k_v141(
                        solver_res=solver_res, want_grid=want_by_id.get(task_id), max_trials=int(max_trials)
                    )
                    scoring_by_id[task_id] = scoring
                    write_per_task_row(task_id=task_id, solver_res=solver_res, scoring=scoring)
                    fk = str(scoring.get("failure_kind") or "")
                    failures_so_far[fk] = int(failures_so_far.get(fk, 0)) + 1
                    by_k = scoring.get("scores_by_k") if isinstance(scoring.get("scores_by_k"), dict) else {}
                    ok_item = by_k.get(str(int(max_trials))) if isinstance(by_k.get(str(int(max_trials))), dict) else {}
                    if bool(ok_item.get("ok")):
                        solved_kmax += 1
                    done += 1
                    last_progress = time.monotonic()
                    log_progress(
                        {
                            "done": int(done),
                            "total": int(total_tasks),
                            "solved_kmax": int(solved_kmax),
                            "last_task_id": str(task_id),
                            "last_status": str(solver_res.get("status") or ""),
                            "last_failure": str(fk),
                            "elapsed_s": float(time.monotonic() - started),
                        }
                    )

            if stalled:
                # Structural watchdog: no completed tasks for a long window.
                # This does not change solver behavior; it only prevents the harness from hanging forever.
                pending_task_ids = sorted({str(fut_by_task_id.get(f, "") or "") for f in pending if f in fut_by_task_id})
                _write_once_json(
                    out_dir / "WATCHDOG_v141.json",
                    {
                        "schema_version": 141,
                        "kind": "arc_no_progress_watchdog_v141",
                        "no_progress_timeout_s": float(no_progress),
                        "seconds_since_last_progress": float(time.monotonic() - last_progress),
                        "done": int(done),
                        "total": int(total_tasks),
                        "pending_task_ids": pending_task_ids,
                    },
                )

                # Stop workers and mark remaining tasks as SEARCH_BUDGET_EXCEEDED (ontological failure cone).
                procs_snapshot = _snapshot_processpool_workers_v141(ex)
                ex.shutdown(wait=False, cancel_futures=True)
                _terminate_processpool_workers_v141(ex, procs_snapshot=procs_snapshot)
                for task_id in pending_task_ids:
                    if not task_id:
                        continue
                    if per_task_path(task_id).exists():
                        continue
                    solver_res = {
                        "schema_version": 141,
                        "kind": "arc_solver_result_v141",
                        "status": "FAIL",
                        "program_sig": "",
                        "predicted_grid_hash": "",
                        "failure_reason": {
                            "kind": "SEARCH_BUDGET_EXCEEDED",
                            "details": {
                                "type": "NO_PROGRESS_WATCHDOG",
                                "no_progress_timeout_s": float(no_progress),
                                "seconds_since_last_progress": float(time.monotonic() - last_progress),
                            },
                        },
                        "trace": {"watchdog": True},
                    }
                    scoring = _score_test_case_all_k_v141(
                        solver_res=solver_res, want_grid=want_by_id.get(task_id), max_trials=int(max_trials)
                    )
                    write_per_task_row(task_id=task_id, solver_res=solver_res, scoring=scoring)
                    scoring_by_id[task_id] = scoring
                    solver_results.append({"task_id": task_id, "solver_res": solver_res})
                    done += 1
                pending.clear()
        except BaseException:
            procs_snapshot = _snapshot_processpool_workers_v141(ex)
            ex.shutdown(wait=False, cancel_futures=True)
            # Best-effort terminate any workers to avoid orphans on interrupts.
            _terminate_processpool_workers_v141(ex, procs_snapshot=procs_snapshot)
            raise
        else:
            # If the no-progress watchdog fired we must not wait for long-running workers;
            # the run has already been finalized by writing SEARCH_BUDGET_EXCEEDED rows.
            if stalled:
                ex.shutdown(wait=False, cancel_futures=True)
            else:
                ex.shutdown(wait=True, cancel_futures=False)

    # Index results for deterministic task association.
    res_by_id: Dict[str, Dict[str, Any]] = {str(r.get("task_id") or ""): r for r in solver_results}

    for task in tasks:
        out_path = per_task_path(str(task.task_id))
        if out_path.exists():
            obj = _read_json(out_path)
            if isinstance(obj, dict):
                per_task_rows.append(obj)
                continue

        item = res_by_id.get(str(task.task_id))
        if item is None:
            # Never crash the run due to missing worker output; record as a task-level failure and continue.
            solver_res = {
                "schema_version": 141,
                "kind": "arc_solver_result_v141",
                "status": "FAIL",
                "program_sig": "",
                "predicted_grid_hash": "",
                "failure_reason": {
                    "kind": "EXCEPTION",
                    "details": {"type": "MISSING_SOLVER_RESULT", "message": f"missing_solver_result_v141:{task.task_id}"},
                },
                "trace": {"exception_type": "MISSING_SOLVER_RESULT", "exception": f"missing_solver_result_v141:{task.task_id}"},
            }
            scoring = _score_test_case_all_k_v141(
                solver_res=solver_res, want_grid=task.test_pairs[0][1], max_trials=int(max_trials)
            )
            scoring_by_id[str(task.task_id)] = scoring
            item = {"task_id": str(task.task_id), "solver_res": solver_res}
            res_by_id[str(task.task_id)] = item
        solver_res = item.get("solver_res")
        if not isinstance(solver_res, dict):
            raise RuntimeError(f"bad_solver_result_v141:{task.task_id}")
        scoring = scoring_by_id.get(str(task.task_id)) or _score_test_case_all_k_v141(
            solver_res=solver_res, want_grid=task.test_pairs[0][1], max_trials=int(max_trials)
        )
        row = {
            "schema_version": 141,
            "kind": "arc_per_task_v141",
            "task": task.to_dict(),
            "solver_results": [solver_res],
            "scoring": scoring,
        }
        per_task_rows.append(row)

        write_per_task_row(task_id=str(task.task_id), solver_res=solver_res, scoring=scoring)

    # Now that per_task rows are complete, generate per-task manifest and trace candidates deterministically.
    for row in per_task_rows:
        if not isinstance(row, dict):
            continue
        task_obj = row.get("task") if isinstance(row.get("task"), dict) else {}
        task_id = str(task_obj.get("task_id") or "")
        solver_results_row = row.get("solver_results")
        sr0 = solver_results_row[0] if isinstance(solver_results_row, list) and solver_results_row and isinstance(solver_results_row[0], dict) else {}
        scoring = row.get("scoring") if isinstance(row.get("scoring"), dict) else {}

        solver_status = str(sr0.get("status") or "")
        fk = str(scoring.get("failure_kind") or "")
        scoring_rows.append(
            {
                "schema_version": 141,
                "kind": "arc_per_task_manifest_row_v141",
                "task_id": str(task_id),
                "status": solver_status,
                "failure_kind": fk,
                "program_sig": str(sr0.get("program_sig") or ""),
                "predicted_grid_hash": str(sr0.get("predicted_grid_hash") or ""),
            }
        )

        trace = sr0.get("trace") if isinstance(sr0.get("trace"), dict) else {}
        tps = trace.get("trace_programs", []) if isinstance(trace.get("trace_programs"), list) else []
        tps_dicts = [tp for tp in tps if isinstance(tp, dict)]
        tps_dicts.sort(key=lambda d: (str(d.get("program_sig") or ""), _stable_json(d)))
        for tp in tps_dicts:
            trace_candidates_rows.append(
                {
                    "schema_version": 141,
                    "kind": "arc_trace_candidate_v141",
                    "task_id": str(task_id),
                    **{str(k): tp.get(k) for k in sorted(tp.keys())},
                }
            )

    omega_events: List[Dict[str, Any]] = []
    omega_fail_total = 0
    if bool(omega_enabled):
        from atos_core.omega_v2 import arc_task_context_id, arc_task_family_id

        for row in per_task_rows:
            if not isinstance(row, dict):
                continue
            task_obj = row.get("task") if isinstance(row.get("task"), dict) else {}
            task_id = str(task_obj.get("task_id") or "")
            scoring = row.get("scoring") if isinstance(row.get("scoring"), dict) else {}
            solver_results_row = row.get("solver_results")
            sr0 = (
                solver_results_row[0]
                if isinstance(solver_results_row, list) and solver_results_row and isinstance(solver_results_row[0], dict)
                else {}
            )

            family_id = arc_task_family_id(task_obj)
            context_id = arc_task_context_id(task_obj)

            by_k = scoring.get("scores_by_k") if isinstance(scoring.get("scores_by_k"), dict) else {}
            ok_item = by_k.get(str(int(max_trials))) if isinstance(by_k.get(str(int(max_trials))), dict) else {}
            episode_success = bool(ok_item.get("ok"))
            scoring_failure_kind = str(scoring.get("failure_kind") or "")

            fr = sr0.get("failure_reason") if isinstance(sr0.get("failure_reason"), dict) else {}
            solver_failure_kind = str(fr.get("kind") or "")
            details = fr.get("details") if isinstance(fr.get("details"), dict) else {}
            tmpl = details.get("concept_template") if isinstance(details.get("concept_template"), dict) else None
            residual_sig = details.get("residual_signature") if isinstance(details.get("residual_signature"), dict) else None

            concept_calls: List[str] = []
            macro_calls: List[str] = []
            program_cost_bits: Optional[int] = None
            program_depth: Optional[int] = None
            if episode_success:
                steps = sr0.get("program_steps")
                if isinstance(steps, list):
                    program_depth = int(len(steps))
                    for st in steps:
                        if not isinstance(st, dict):
                            continue
                        op_id = str(st.get("op_id") or "")
                        if op_id == "concept_call":
                            args = st.get("args") if isinstance(st.get("args"), dict) else {}
                            cid = str(args.get("concept_id") or "")
                            if cid:
                                concept_calls.append(cid)
                        elif op_id == "macro_call":
                            args = st.get("args") if isinstance(st.get("args"), dict) else {}
                            mid = str(args.get("macro_id") or "")
                            if mid:
                                macro_calls.append(mid)
                pcb = sr0.get("program_cost_bits")
                if isinstance(pcb, int):
                    program_cost_bits = int(pcb)
            concept_calls = sorted(list(dict.fromkeys([c for c in concept_calls if c])))
            macro_calls = sorted(list(dict.fromkeys([m for m in macro_calls if m])))

            if not episode_success:
                omega_fail_total += 1

            omega_events.append(
                {
                    "schema_version": 2,
                    "kind": "omega_event_v2",
                    "task_id": str(task_id),
                    "task_family_id": str(family_id),
                    "task_context_id": str(context_id),
                    "task": dict(task_obj),
                    "episode_success": bool(episode_success),
                    "scoring_failure_kind": str(scoring_failure_kind),
                    "solver_failure_kind": str(solver_failure_kind),
                    "residual_signature": dict(residual_sig) if isinstance(residual_sig, dict) else None,
                    "concept_template": dict(tmpl) if isinstance(tmpl, dict) else None,
                    "concept_calls_solution": concept_calls,
                    "macro_calls_solution": macro_calls,
                    "program_cost_bits": int(program_cost_bits) if isinstance(program_cost_bits, int) else None,
                    "program_depth": int(program_depth) if isinstance(program_depth, int) else None,
                }
            )
        omega_events.sort(key=lambda r: str(r.get("task_id") or ""))
        _write_jsonl_x(out_dir / "omega_events_v2.jsonl", omega_events)

    # Write aggregate jsonl artifacts.
    _write_jsonl_x(out_dir / "per_task_manifest.jsonl", scoring_rows)
    _write_jsonl_x(out_dir / "trace_candidates.jsonl", trace_candidates_rows)

    # Aggregate eval + summary.
    failure_counts: Dict[str, int] = {}
    for r in scoring_rows:
        st = str(r.get("status") or "")
        if st == "SOLVED":
            continue
        fk = str(r.get("failure_kind") or "")
        if fk:
            failure_counts[fk] = int(failure_counts.get(fk, 0)) + 1

    score_summary = _summarize_scores_v141(per_task_rows, max_trials=int(max_trials))
    program_usage = _summarize_program_usage_v141(per_task_rows)
    eval_obj: Dict[str, Any] = {
        "schema_version": 141,
        "kind": "arc_eval_v141",
        "failure_counts": {k: int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "tasks_total": int(score_summary["tasks_total"]),
        "tasks_solved_by_k": score_summary["tasks_solved_by_k"],
        "program_usage": dict(program_usage),
    }
    from atos_core.act import canonical_json_dumps, sha256_hex

    eval_obj["eval_sig"] = sha256_hex(canonical_json_dumps(eval_obj).encode("utf-8"))
    _write_once_json(out_dir / "eval.json", eval_obj)

    summary_obj: Dict[str, Any] = {
        "schema_version": 141,
        "kind": "arc_summary_v141",
        "arc_root": str(arc_root),
        "split": str(split or ""),
        "limit": int(limit),
        "seed": int(seed),
        "max_depth": int(max_depth_eff),
        "max_programs": int(max_programs_eff),
        # Solver config (needed for Ω promotion gates + ablation to mirror the run deterministically).
        "trace_program_limit": 80,
        "solution_cost_slack_bits": int(solution_cost_slack_bits),
        "abstraction_pressure": bool(abstraction_pressure),
        "max_depth_requested": int(max_depth),
        "max_programs_requested": int(max_programs),
        "task_timeout_s": float(task_timeout_s),
        "no_progress_timeout_s": float(no_progress_timeout_s or 0.0),
        "resume": bool(resume),
        "benchmark_profile": str(benchmark_profile),
        "max_trials": int(max_trials),
        "macro_templates_path": str(macro_templates_path or ""),
        "concept_templates_path": str(concept_templates_path or ""),
        "macro_templates_total": int(len(macro_templates_rows)),
        "concept_templates_total": int(len(concept_templates_rows)),
        "macro_templates_sig": str(macro_templates_sig),
        "concept_templates_sig": str(concept_templates_sig),
        "omega_enabled": bool(omega_enabled),
        "omega_state_in": str(omega_state_in or ""),
        "omega_state_sig": str(omega_state_sig or ""),
        "omega_caps": (
            {
                "max_depth_cap": int(getattr(omega_state_obj, "max_depth_cap", 0) or 0),
                "max_programs_cap": int(getattr(omega_state_obj, "max_programs_cap", 0) or 0),
                "dead_clusters_total": int(getattr(omega_state_obj, "dead_clusters_total", 0) or 0),
                "state_sha": str(getattr(omega_state_obj, "state_sha", "") or ""),
            }
            if omega_state_obj is not None
            else None
        ),
        "omega_banned_task_families_total": int(len(omega_banned_families)),
        "omega_banned_tasks_filtered": int(len(omega_banned_tasks)),
        "omega_failures_total": int(omega_fail_total),
        "macro_try_on_fail_only": bool(macro_try_on_fail_only),
        "enable_reachability_pruning": bool(enable_reachability_pruning),
        "macro_propose_max_depth": int(macro_propose_max_depth),
        "macro_max_templates": int(macro_max_templates),
        "macro_max_instantiations": int(macro_max_instantiations),
        "macro_max_branch_per_op": int(macro_max_branch_per_op),
        "enable_repair_stage": bool(enable_repair_stage),
        "enable_residual_stage": bool(enable_residual_stage),
        "enable_refine_stage": bool(enable_refine_stage),
        "enable_point_patch_repair": bool(enable_point_patch_repair),
        "point_patch_max_points": int(point_patch_max_points),
        "tasks_total": int(score_summary["tasks_total"]),
        "tasks_solved_by_k": score_summary["tasks_solved_by_k"],
        "failure_counts": {k: int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "program_usage": dict(program_usage),
        "eval_sig": str(eval_obj["eval_sig"]),
    }
    summary_obj["summary_sig"] = sha256_hex(canonical_json_dumps(summary_obj).encode("utf-8"))
    _write_once_json(out_dir / "summary.json", summary_obj)

    # Diag report is deterministic function of summary+eval.
    _write_text_x(out_dir / "ARC_DIAG_REPORT_v141.md", _make_diag_report_v141(summary_obj=summary_obj, eval_obj=eval_obj))

    # Isolation check
    after_sig = _repo_snapshot_sha256_v141(root=root, exclude_paths=[out_dir])
    isolation = {
        "schema_version": 141,
        "kind": "isolation_check_v141",
        "repo_snapshot_before": str(before_sig),
        "repo_snapshot_after": str(after_sig),
        "ok": bool(str(before_sig) == str(after_sig)),
    }
    _write_once_json(out_dir / "isolation_check_v141.json", isolation)
    if not isolation["ok"]:
        raise SystemExit("isolation_violation_v141")

    outputs_manifest = _outputs_manifest_v141(run_dir=out_dir)
    _write_once_json(out_dir / "outputs_manifest.json", outputs_manifest)

    return {
        "out_dir": str(out_dir),
        "arc_manifest_sig": str(manifest_obj.get("manifest_sig") or ""),
        "macro_templates_sig": str(macro_templates_sig),
        "concept_templates_sig": str(concept_templates_sig),
        "summary_sig": str(summary_obj.get("summary_sig") or ""),
        "outputs_manifest_sig": str(outputs_manifest.get("outputs_manifest_sig") or ""),
        "tasks_total": int(score_summary["tasks_total"]),
        "tasks_solved_by_k": score_summary["tasks_solved_by_k"],
        "failure_counts": {k: int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "determinism_ok": True,
        "isolation_ok": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc_root", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    # jobs<=0 => auto (os.cpu_count()).
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--max_programs", type=int, default=4000)
    ap.add_argument(
        "--solution_cost_slack_bits",
        type=int,
        default=16,
        help="Collect additional train-perfect solutions within this MDL window to reduce overfitting when train pairs do not disambiguate.",
    )
    ap.add_argument("--benchmark_profile", default="ARC_AGI1_PROFILE")
    ap.add_argument("--max_trials", type=int, default=None)
    ap.add_argument("--macro_templates", default=None)
    ap.add_argument("--concept_templates", default=None)
    ap.add_argument(
        "--tries",
        type=int,
        default=2,
        help="How many deterministic tries to run (1 = faster iteration, 2 = full determinism check).",
    )
    ap.add_argument(
        "--task_timeout_s",
        type=float,
        default=0.0,
        help="Per-task wall-time limit for fast iteration (0 disables). Only allowed with --tries 1.",
    )
    ap.add_argument(
        "--no_progress_timeout_s",
        type=float,
        default=0.0,
        help="Harness watchdog: if no tasks complete for this many seconds, mark remaining tasks as SEARCH_BUDGET_EXCEEDED and finalize the run (0 disables).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume an incomplete run in the existing out_dir (WORM: never overwrites; only writes missing per_task and final summary artifacts).",
    )
    ap.add_argument(
        "--enable_point_patch_repair",
        action="store_true",
        help="Enable deterministic constant point-patch repair on very small near-misses (still train-consistent).",
    )
    ap.add_argument(
        "--point_patch_max_points",
        type=int,
        default=12,
        help="Max points for constant point-patch repair (only used when enabled).",
    )
    ap.add_argument(
        "--disable_repair_stage",
        action="store_true",
        help="Disable the near-miss repair stage (useful for 'abstraction pressure' runs).",
    )
    ap.add_argument(
        "--disable_residual_stage",
        action="store_true",
        help="Disable the residual refinement stage (useful for 'abstraction pressure' runs).",
    )
    ap.add_argument(
        "--disable_refine_stage",
        action="store_true",
        help="Disable the refine stage (useful for 'abstraction pressure' runs).",
    )
    ap.add_argument(
        "--disable_reachability_pruning",
        action="store_true",
        help="Disable shape/palette/no-grid-modify reachability pruning (useful for operator discovery under tight caps).",
    )
    ap.add_argument(
        "--macro_try_on_fail_only",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: baseline-first; only try macros if baseline fails. 0: include macros in the main search.",
    )
    ap.add_argument(
        "--macro_propose_max_depth",
        type=int,
        default=0,
        help="Allow macro_call proposals up to this search depth (0 = root-only).",
    )
    ap.add_argument(
        "--macro_max_templates",
        type=int,
        default=24,
        help="Max macro templates to consider per state when macro proposals are enabled.",
    )
    ap.add_argument(
        "--macro_max_instantiations",
        type=int,
        default=10,
        help="Max arg instantiations per template when proposing macro_call.",
    )
    ap.add_argument(
        "--macro_max_branch_per_op",
        type=int,
        default=10,
        help="Max arg branches per op_id within macro instantiation.",
    )
    ap.add_argument(
        "--abstraction_pressure",
        action="store_true",
        help="Enable abstraction-or-die pressure: prefer closure (concept_call) as soon as shape matches and classify shape-correct budget failures as MISSING_CONCEPT.",
    )
    ap.add_argument(
        "--omega",
        action="store_true",
        help="Enable Ω (destructive ontological memory): on failure without *new* concept_template emission, future search optionality is reduced in the next omega_state update.",
    )
    ap.add_argument(
        "--omega_state_in",
        default="",
        help="Optional omega_state_v1.json to cap max_depth/max_programs (Ω is inescapable if this is chained across runs).",
    )
    ap.add_argument("--out_base", required=True)
    args = ap.parse_args()

    tries = int(args.tries)
    if tries not in (1, 2):
        raise SystemExit("bad_tries_v141")

    task_timeout_s = float(args.task_timeout_s or 0.0)
    if tries >= 2 and task_timeout_s > 0.0:
        raise SystemExit("task_timeout_requires_tries1_v141")

    no_progress_timeout_s = float(args.no_progress_timeout_s or 0.0)
    if tries >= 2 and no_progress_timeout_s > 0.0:
        raise SystemExit("no_progress_timeout_requires_tries1_v141")

    resume = bool(args.resume)
    if resume and tries != 1:
        raise SystemExit("resume_requires_tries1_v141")

    out_base = Path(str(args.out_base))
    try1 = Path(str(out_base) + "_try1")
    try2 = Path(str(out_base) + "_try2")

    if not resume:
        _ensure_absent(try1)
        if tries >= 2:
            _ensure_absent(try2)

    max_trials = int(args.max_trials) if args.max_trials is not None else _benchmark_profile_default_trials(str(args.benchmark_profile))

    try:
        r1 = _run_once_v141(
            out_dir=try1,
            arc_root=str(args.arc_root),
            split=str(args.split) if args.split else None,
            limit=int(args.limit),
            seed=int(args.seed),
            jobs=int(args.jobs),
            max_depth=int(args.max_depth),
            max_programs=int(args.max_programs),
            solution_cost_slack_bits=int(args.solution_cost_slack_bits),
            max_trials=int(max_trials),
            benchmark_profile=str(args.benchmark_profile),
            macro_templates_path=str(args.macro_templates) if args.macro_templates else None,
            concept_templates_path=str(args.concept_templates) if args.concept_templates else None,
            task_timeout_s=float(task_timeout_s),
            no_progress_timeout_s=float(no_progress_timeout_s),
            resume=bool(resume),
            enable_repair_stage=bool(not bool(args.disable_repair_stage)),
            enable_residual_stage=bool(not bool(args.disable_residual_stage)),
            enable_refine_stage=bool(not bool(args.disable_refine_stage)),
            macro_try_on_fail_only=bool(int(args.macro_try_on_fail_only) != 0),
            macro_propose_max_depth=int(args.macro_propose_max_depth),
            macro_max_templates=int(args.macro_max_templates),
            macro_max_instantiations=int(args.macro_max_instantiations),
            macro_max_branch_per_op=int(args.macro_max_branch_per_op),
            abstraction_pressure=bool(args.abstraction_pressure),
            enable_reachability_pruning=not bool(args.disable_reachability_pruning),
            enable_point_patch_repair=bool(args.enable_point_patch_repair),
            point_patch_max_points=int(args.point_patch_max_points),
            omega_enabled=bool(args.omega),
            omega_state_in=str(args.omega_state_in or "").strip() or None,
        )
    except BaseException as e:
        _write_crash_artifact_v141(out_dir=try1, stage="try1", err=e)
        raise

    r2: Dict[str, Any] = {}
    determinism_ok = True
    if tries >= 2:
        try:
            r2 = _run_once_v141(
                out_dir=try2,
                arc_root=str(args.arc_root),
                split=str(args.split) if args.split else None,
                limit=int(args.limit),
                seed=int(args.seed),
                jobs=int(args.jobs),
                max_depth=int(args.max_depth),
                max_programs=int(args.max_programs),
                solution_cost_slack_bits=int(args.solution_cost_slack_bits),
                max_trials=int(max_trials),
                benchmark_profile=str(args.benchmark_profile),
                macro_templates_path=str(args.macro_templates) if args.macro_templates else None,
                concept_templates_path=str(args.concept_templates) if args.concept_templates else None,
                task_timeout_s=float(task_timeout_s),
                no_progress_timeout_s=float(no_progress_timeout_s),
                resume=bool(resume),
                enable_repair_stage=bool(not bool(args.disable_repair_stage)),
                enable_residual_stage=bool(not bool(args.disable_residual_stage)),
                enable_refine_stage=bool(not bool(args.disable_refine_stage)),
                macro_try_on_fail_only=bool(int(args.macro_try_on_fail_only) != 0),
                macro_propose_max_depth=int(args.macro_propose_max_depth),
                macro_max_templates=int(args.macro_max_templates),
                macro_max_instantiations=int(args.macro_max_instantiations),
                macro_max_branch_per_op=int(args.macro_max_branch_per_op),
                abstraction_pressure=bool(args.abstraction_pressure),
                enable_reachability_pruning=not bool(args.disable_reachability_pruning),
                enable_point_patch_repair=bool(args.enable_point_patch_repair),
                point_patch_max_points=int(args.point_patch_max_points),
                omega_enabled=bool(args.omega),
                omega_state_in=str(args.omega_state_in or "").strip() or None,
            )
        except BaseException as e:
            _write_crash_artifact_v141(out_dir=try2, stage="try2", err=e)
            raise
        determinism_ok = bool(
            r1.get("summary_sig") == r2.get("summary_sig") and r1.get("outputs_manifest_sig") == r2.get("outputs_manifest_sig")
        )
        if not determinism_ok:
            raise SystemExit("determinism_mismatch_v141")

    final: Dict[str, Any] = {"ok": True, "tries": int(tries), "determinism_ok": bool(determinism_ok), "try1": r1}
    if tries >= 2:
        final["try2"] = r2
    print(_stable_json(final))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
