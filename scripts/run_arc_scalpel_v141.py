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
        # Datasets are treated as external inputs for isolation purposes. Including them in the
        # repo snapshot can produce false isolation violations if a dataset is newly added or
        # updated (e.g. downloading ARC-AGI-2) while keeping code unchanged.
        "ARC-AGI",
        "arc-agi-2",
        "ARC-AGI-2",
        "data",
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


def _grid_shape_v141(g: Sequence[Sequence[int]]) -> Tuple[int, int]:
    h = int(len(g))
    w = int(len(g[0])) if h > 0 else 0
    return (int(h), int(w))


def _unique_colors_v141(g: Sequence[Sequence[int]]) -> set:
    out = set()
    for row in g:
        for v in row:
            out.add(int(v))
    return out


def _shape_rel_for_pairs_v141(pairs: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]]) -> str:
    if not pairs:
        return "unknown"
    if all(a == b for a, b in pairs):
        return "same"
    if all(b == (a[1], a[0]) for a, b in pairs):
        return "swap_hw"
    ratios = set()
    for (hi, wi), (ho, wo) in pairs:
        if hi <= 0 or wi <= 0 or ho <= 0 or wo <= 0:
            return "shape_change_mixed"
        if ho % hi != 0 or wo % wi != 0:
            return "shape_change_mixed"
        ratios.add((int(ho // hi), int(wo // wi)))
    if len(ratios) == 1:
        ry, rx = list(ratios)[0]
        return f"scale_integer:{int(ry)}x{int(rx)}"
    return "shape_change_mixed"


def _palette_rel_v141(p_in: set, p_out: set) -> str:
    if p_out.issubset(p_in):
        if p_in == p_out:
            return "equal"
        return "subset"
    if p_in.issubset(p_out):
        return "superset"
    return "other"


def _delta_density_bin_v141(train_pairs: Sequence[Tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]]) -> str:
    diffs: List[float] = []
    for inp, out in train_pairs:
        hi, wi = _grid_shape_v141(inp)
        ho, wo = _grid_shape_v141(out)
        if (int(hi), int(wi)) != (int(ho), int(wo)) or int(hi) <= 0 or int(wi) <= 0:
            continue
        diff = 0
        for r in range(int(hi)):
            for c in range(int(wi)):
                if int(inp[int(r)][int(c)]) != int(out[int(r)][int(c)]):
                    diff += 1
        diffs.append(float(diff) / float(max(1, int(hi) * int(wi))))
    if not diffs:
        return "n/a"
    avg = sum(diffs) / float(len(diffs))
    if avg <= 0.10:
        return "sparse<=0.10"
    if avg <= 0.30:
        return "local<=0.30"
    return "dense>0.30"


def _task_feat_key_from_train_pairs_v141(
    *, train_pairs: Sequence[Tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]]
) -> Tuple[str, str, str]:
    shapes: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    pin: set = set()
    pout: set = set()
    for inp, out in train_pairs:
        shapes.append((_grid_shape_v141(inp), _grid_shape_v141(out)))
        pin |= _unique_colors_v141(inp)
        pout |= _unique_colors_v141(out)
    return (
        str(_shape_rel_for_pairs_v141(shapes)),
        str(_palette_rel_v141(pin, pout)),
        str(_delta_density_bin_v141(train_pairs)),
    )


def _colors_bin_v141(n: int) -> str:
    x = int(n)
    if x <= 2:
        return "c<=2"
    if x <= 4:
        return "c<=4"
    if x <= 6:
        return "c<=6"
    if x <= 9:
        return "c<=9"
    return "c=10"


def _dim_bin_v141(h: int, w: int) -> str:
    m = max(int(h), int(w))
    if m <= 5:
        return "d<=5"
    if m <= 10:
        return "d<=10"
    if m <= 20:
        return "d<=20"
    return "d>20"


def _task_feat_key_v2_from_train_pairs_v141(
    *, train_pairs: Sequence[Tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]]
) -> Tuple[str, ...]:
    shapes: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    pin: set = set()
    pout: set = set()
    in_dims: List[Tuple[int, int]] = []
    out_dims: List[Tuple[int, int]] = []
    for inp, out in train_pairs:
        si = _grid_shape_v141(inp)
        so = _grid_shape_v141(out)
        shapes.append((si, so))
        in_dims.append(tuple(int(x) for x in si))
        out_dims.append(tuple(int(x) for x in so))
        pin |= _unique_colors_v141(inp)
        pout |= _unique_colors_v141(out)

    hi = max((int(h) for h, _w in in_dims), default=0)
    wi = max((int(w) for _h, w in in_dims), default=0)
    ho = max((int(h) for h, _w in out_dims), default=0)
    wo = max((int(w) for _h, w in out_dims), default=0)

    return (
        str(_shape_rel_for_pairs_v141(shapes)),
        str(_palette_rel_v141(pin, pout)),
        str(_delta_density_bin_v141(train_pairs)),
        f"n{int(len(list(train_pairs)))}",
        f"in_{_colors_bin_v141(len(pin))}",
        f"out_{_colors_bin_v141(len(pout))}",
        f"in_{_dim_bin_v141(int(hi), int(wi))}",
        f"out_{_dim_bin_v141(int(ho), int(wo))}",
    )


def _enrich_concept_support_task_feat_keys_v141(*, concept_templates_rows: List[Dict[str, Any]], arc_root: str) -> None:
    """
    Add a cheap, training-only feature signature for each concept template's support tasks.

    This enables deterministic concept retrieval ranking inside the solver without giving it
    any direct access to the filesystem/dataset at runtime.
    """
    # Prefer the current arc_root, but allow cross-root concept banks (ARC-AGI1 + ARC-AGI2).
    train_dirs: List[Path] = []
    root_primary = (Path(str(arc_root)) / "data" / "training").resolve()
    train_dirs.append(root_primary)
    for alt in ("ARC-AGI", "arc-agi-2", "ARC-AGI-2"):
        p = (Path(str(alt)) / "data" / "training").resolve()
        if p not in train_dirs:
            train_dirs.append(p)

    feat_cache_v1: Dict[str, Tuple[str, str, str]] = {}
    feat_cache_v2: Dict[str, Tuple[str, ...]] = {}

    def _read_train_pairs(tid: str) -> Optional[List[Tuple[List[List[int]], List[List[int]]]]]:
        if tid in feat_cache_v2:
            return None
        for train_dir in train_dirs:
            if not train_dir.is_dir():
                continue
            p = (train_dir / str(tid)).resolve()
            if not str(p).startswith(str(train_dir)):
                continue
            if not p.is_file():
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            train_raw = obj.get("train")
            if not isinstance(train_raw, list) or not train_raw:
                continue
            out: List[Tuple[List[List[int]], List[List[int]]]] = []
            for row in train_raw:
                if not isinstance(row, dict):
                    continue
                gi = row.get("input")
                go = row.get("output")
                if not isinstance(gi, list) or not isinstance(go, list):
                    continue
                out.append((gi, go))
            if not out:
                continue
            return out
        return None

    for tmpl in concept_templates_rows:
        if not isinstance(tmpl, dict):
            continue
        has_v1 = isinstance(tmpl.get("support_task_feat_keys"), list)
        has_v2 = isinstance(tmpl.get("support_task_feat_keys_v2"), list)
        if has_v1 and has_v2:
            continue
        stids = tmpl.get("support_task_ids") if isinstance(tmpl.get("support_task_ids"), list) else []
        feats_v1 = set()
        feats_v2 = set()
        # Deterministic cap for runtime/memory.
        for tid0 in stids[:64]:
            tid = str(tid0 or "").strip().replace("\\", "/").lstrip("/")
            if not tid:
                continue
            if not tid.endswith(".json"):
                tid = tid + ".json"
            if tid not in feat_cache_v2:
                pairs = _read_train_pairs(tid)
                if pairs is None:
                    continue
                feat_cache_v1[tid] = _task_feat_key_from_train_pairs_v141(train_pairs=pairs)
                feat_cache_v2[tid] = _task_feat_key_v2_from_train_pairs_v141(train_pairs=pairs)
            if tid in feat_cache_v1:
                feats_v1.add(feat_cache_v1[tid])
            if tid in feat_cache_v2:
                feats_v2.add(feat_cache_v2[tid])
        if (not has_v1) and feats_v1:
            tmpl["support_task_feat_keys"] = [list(x) for x in sorted(feats_v1)]
        if (not has_v2) and feats_v2:
            tmpl["support_task_feat_keys_v2"] = [list(x) for x in sorted(feats_v2)]


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
    require_concept_call: bool = False,
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

    def _has_concept_call(steps: Any) -> bool:
        if not isinstance(steps, list):
            return False
        for st in steps:
            if not isinstance(st, dict):
                continue
            if str(st.get("op_id") or "") == "concept_call":
                return True
        return False

    # Candidate outputs from solver.
    #
    # IMPORTANT: keep positional alignment with per-candidate steps (if present). Do NOT
    # filter out placeholders here; skip invalid candidates at comparison time.
    preds = solver_res.get("predicted_grids")
    if isinstance(preds, list) and preds:
        candidates = list(preds)
    else:
        pred = solver_res.get("predicted_grid")
        candidates = [pred] if pred is not None else []

    # Candidate program steps aligned with candidate outputs (when available).
    #
    # NOTE: In FAIL-CLOSED mode, the solver may return status=UNKNOWN with a correct
    # candidate output, but without top-level program_steps. In that case we must
    # enforce require_concept_call using per-candidate steps.
    steps_by_cand: List[Any] = []
    cps = solver_res.get("candidate_program_steps")
    if isinstance(cps, list) and cps:
        steps_by_cand = list(cps)
    else:
        cand_programs = solver_res.get("candidate_programs")
        if isinstance(cand_programs, list) and cand_programs:
            for cp in cand_programs:
                if not isinstance(cp, dict):
                    steps_by_cand.append(None)
                    continue
                steps_by_cand.append(cp.get("program_steps"))
        else:
            ps = solver_res.get("program_steps")
            if isinstance(ps, list) and ps:
                steps_by_cand = [ps]

    scores_by_k: Dict[str, Any] = {}
    any_ok = False
    any_ok_ignoring_concept_gate = False
    for k in (1, 2, 3):
        kk = int(k)
        if kk <= 0:
            continue
        if kk > int(max_trials):
            continue
        ok = False
        for i, cand in enumerate(candidates[:kk]):
            if not isinstance(cand, list) or not _grid_equal(cand, want_grid):
                continue
            any_ok_ignoring_concept_gate = True
            if bool(require_concept_call):
                # Enforce that the *matching candidate* is backed by concept_call.
                steps = steps_by_cand[i] if i < len(steps_by_cand) else None
                if not _has_concept_call(steps):
                    continue
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
    if bool(require_concept_call) and not any_ok and any_ok_ignoring_concept_gate:
        # Fail-closed: output correctness does not count as SOLVED if concept usage is absent.
        fk = "NO_CONCEPT_CALL"
    return {
        "solver_status": status if status in {"SOLVED", "UNKNOWN"} else "FAIL",
        "scored": True,
        "failure_kind": fk,
        "scores_by_k": scores_by_k,
    }


def _score_task_all_tests_all_k_v141(
    *,
    solver_results: Sequence[Dict[str, Any]],
    want_grids: Sequence[Optional[Sequence[Sequence[int]]]],
    max_trials: int,
    require_concept_call: bool = False,
) -> Dict[str, Any]:
    """
    Score an ARC task with multiple test cases.

    Task is considered solved only if *all* test outputs are correct under the same k.
    This is stricter and avoids self-deception on ARC-AGI datasets with multiple test inputs.
    """
    if not want_grids or any(w is None for w in want_grids):
        # Unscored mode (e.g. internal suites without test outputs): fall back to solver status only.
        status0 = ""
        if solver_results:
            status0 = str(solver_results[0].get("status") or "")
        fk0 = ""
        fr = solver_results[0].get("failure_reason") if solver_results and isinstance(solver_results[0], dict) else {}
        if isinstance(fr, dict):
            fk0 = str(fr.get("kind") or "")
        return {
            "solver_status": status0 if status0 in {"SOLVED", "UNKNOWN"} else "FAIL",
            "scored": False,
            "failure_kind": fk0,
            "scores_by_k": {},
        }

    if len(want_grids) <= 1:
        # Single-test tasks: keep legacy scoring (k means "any of first k candidate outputs").
        return _score_test_case_all_k_v141(
            solver_res=solver_results[0] if solver_results else {},
            want_grid=want_grids[0] if want_grids else None,
            max_trials=int(max_trials),
            require_concept_call=bool(require_concept_call),
        )

    # Multi-test tasks: require *one* candidate program to solve *all* tests.
    #
    # IMPORTANT: k means "try the first k candidate programs" and candidate index i is shared
    # across tests. It is invalid to solve test0 with candidate0 and test1 with candidate1.
    def _cands(sr: Dict[str, Any]) -> List[Any]:
        preds = sr.get("predicted_grids")
        if isinstance(preds, list) and preds:
            # Keep alignment; placeholders can be null/invalid.
            return list(preds)
        pred = sr.get("predicted_grid")
        return [pred] if pred is not None else []

    def _candidate_steps(sr: Dict[str, Any]) -> List[Any]:
        cps = sr.get("candidate_program_steps")
        if isinstance(cps, list) and cps:
            return list(cps)
        ps = sr.get("program_steps")
        if isinstance(ps, list) and ps:
            return [ps]
        return []

    cands_by_test: List[List[Any]] = []
    for sr in list(solver_results):
        if not isinstance(sr, dict):
            sr = {}
        cands_by_test.append(_cands(sr))
    if not cands_by_test or any(not isinstance(c, list) or not c for c in cands_by_test):
        fk0 = ""
        fr0 = solver_results[0].get("failure_reason") if solver_results and isinstance(solver_results[0], dict) else {}
        if isinstance(fr0, dict):
            fk0 = str(fr0.get("kind") or "")
        return {
            "solver_status": "FAIL",
            "scored": True,
            "failure_kind": fk0,
            "scores_by_k": {str(k): {"k": int(k), "ok": False} for k in range(1, int(max_trials) + 1)},
        }

    # Candidate programs are shared across tests, so length must be consistent.
    cand_len = min(len(c) for c in cands_by_test)
    steps_by_cand = _candidate_steps(solver_results[0] if solver_results else {})
    if bool(require_concept_call) and steps_by_cand:
        cand_len = min(int(cand_len), int(len(steps_by_cand)))

    def _has_concept_call(steps: Any) -> bool:
        if not isinstance(steps, list):
            return False
        for st in steps:
            if not isinstance(st, dict):
                continue
            if str(st.get("op_id") or "") == "concept_call":
                return True
        return False

    scores_by_k: Dict[str, Any] = {}
    for kk in range(1, int(max_trials) + 1):
        ok = False
        for i in range(min(int(kk), int(cand_len))):
            if bool(require_concept_call) and steps_by_cand:
                if not _has_concept_call(steps_by_cand[i] if i < len(steps_by_cand) else None):
                    continue
            ok_i = True
            for cand_grid, want in zip(cands_by_test, want_grids):
                got = cand_grid[i] if i < len(cand_grid) else None
                if want is None or not isinstance(got, list) or not _grid_equal(got, want):
                    ok_i = False
                    break
            if ok_i:
                ok = True
                break
        scores_by_k[str(int(kk))] = {"k": int(kk), "ok": bool(ok)}

    any_ok = bool(scores_by_k.get(str(int(max_trials)), {}).get("ok"))
    fk = ""
    if not any_ok:
        # Prefer NO_CONCEPT_CALL if there exists a fully-correct candidate output but it lacks concept_call.
        if bool(require_concept_call) and steps_by_cand:
            for i in range(int(cand_len)):
                ok_i = True
                for cand_grid, want in zip(cands_by_test, want_grids):
                    got = cand_grid[i] if i < len(cand_grid) else None
                    if want is None or not isinstance(got, list) or not _grid_equal(got, want):
                        ok_i = False
                        break
                if ok_i and not _has_concept_call(steps_by_cand[i] if i < len(steps_by_cand) else None):
                    fk = "NO_CONCEPT_CALL"
                    break
        if not fk:
            fr0 = solver_results[0].get("failure_reason") if solver_results and isinstance(solver_results[0], dict) else {}
            if isinstance(fr0, dict):
                fk = str(fr0.get("kind") or "")

    return {
        "solver_status": "SOLVED" if any_ok else "FAIL",
        "scored": True,
        "failure_kind": str(fk),
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

    This is intentionally simple and audit-friendly:
      - For multi-test tasks, it inspects the solver's unique chosen program (status == SOLVED).
      - For single-test tasks, the solver may return status == UNKNOWN (ambiguous) while scoring
        still marks k=1 as correct (due to candidate_programs). In that case, this picks the
        lowest-cost matching candidate program for usage accounting, so metrics reflect the
        actually-scored solution surface.
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
        scoring = row.get("scoring") if isinstance(row.get("scoring"), dict) else {}
        scores_by_k = scoring.get("scores_by_k") if isinstance(scoring.get("scores_by_k"), dict) else {}
        # Count only k=1-scored solves (unique output under gates). This matches tasks_solved_by_k["1"].
        k1 = scores_by_k.get("1") if isinstance(scores_by_k.get("1"), dict) else {}
        if not bool(k1.get("ok")):
            continue
        solver_results = row.get("solver_results")
        if not isinstance(solver_results, list) or not solver_results:
            continue
        sr0 = solver_results[0] if isinstance(solver_results[0], dict) else {}
        status0 = str(sr0.get("status") or "")

        steps: List[Dict[str, Any]] = []
        cost_bits = 0
        if status0 == "SOLVED":
            cost_bits = int(sr0.get("program_cost_bits") or 0)
            raw_steps = sr0.get("program_steps")
            if isinstance(raw_steps, list):
                steps = [st for st in raw_steps if isinstance(st, dict)]
        elif status0 == "UNKNOWN":
            # Single-test ambiguity: pick the lowest-cost candidate that matches the ground-truth output.
            task_obj = row.get("task") if isinstance(row.get("task"), dict) else {}
            test = task_obj.get("test") if isinstance(task_obj.get("test"), list) else []
            want = None
            if test and isinstance(test[0], dict) and isinstance(test[0].get("output"), list):
                want = test[0].get("output")
            cands = sr0.get("candidate_programs") if isinstance(sr0.get("candidate_programs"), list) else []
            best = None
            for cand in cands:
                if not isinstance(cand, dict):
                    continue
                pred = cand.get("predicted_grid")
                if want is None or pred != want:
                    continue
                cb = int(cand.get("program_cost_bits") or 0)
                psig = str(cand.get("program_sig") or "")
                raw_steps = cand.get("program_steps")
                if not isinstance(raw_steps, list):
                    continue
                st = [x for x in raw_steps if isinstance(x, dict)]
                key = (int(cb), psig)
                if best is None or key < best[0]:
                    best = (key, int(cb), st)
            if best is None:
                continue
            cost_bits = int(best[1])
            steps = list(best[2])
        else:
            continue

        solved += 1
        total_cost_bits += int(cost_bits)

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
    test_ins = payload.get("test_ins")
    if test_ins is None:
        # Back-compat with older payloads (single-test tasks).
        test_ins = [payload.get("test_in")]
    if not isinstance(train_pairs, (list, tuple)) or not isinstance(test_ins, list):
        raise RuntimeError("bad_task_payload_v141")
    test_ins2 = [ti for ti in test_ins if isinstance(ti, (list, tuple))]
    if not test_ins2:
        raise RuntimeError("bad_task_payload_v141")

    from atos_core.arc_solver_v141 import (
        SolveConfigV141,
        apply_program_steps_to_grid_v141,
        get_trace_metrics_snapshot_v141,
        get_trace_snapshot_v141,
        reset_trace_snapshot_v141,
        solve_arc_task_v141,
    )
    from atos_core.grid_v124 import grid_hash_v124

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
    csv_allow_slot_progress = bool(cfg_payload.get("csv_allow_slot_progress", False))
    enable_reachability_pruning = bool(cfg_payload.get("enable_reachability_pruning", True))
    concept_support_feat_ranking = bool(cfg_payload.get("enable_concept_support_feat_ranking", False))
    trace_csg_induction = bool(cfg_payload.get("trace_csg_induction", False))
    trace_csg_first_pass_frac = float(cfg_payload.get("trace_csg_first_pass_frac", 0.6) or 0.6)
    ensemble = bool(cfg_payload.get("ensemble", False))

    cfg = SolveConfigV141(
        max_depth=int(cfg_payload.get("max_depth", 4)),
        max_programs=int(cfg_payload.get("max_programs", 4000)),
        trace_program_limit=int(cfg_payload.get("trace_program_limit", 80)),
        max_ambiguous_outputs=int(cfg_payload.get("max_ambiguous_outputs", 3)),
        solution_cost_slack_bits=int(cfg_payload.get("solution_cost_slack_bits", 0)),
        macro_templates=macro_templates,
        concept_templates=concept_templates,
        abstraction_pressure=bool(abstraction_pressure),
        csv_allow_slot_progress=bool(csv_allow_slot_progress),
        macro_try_on_fail_only=bool(macro_try_on_fail_only),
        enable_reachability_pruning=bool(enable_reachability_pruning),
        enable_concept_support_feat_ranking=bool(concept_support_feat_ranking),
        macro_propose_max_depth=int(cfg_payload.get("macro_propose_max_depth", 0)),
        macro_max_templates=int(cfg_payload.get("macro_max_templates", 24)),
        macro_max_instantiations=int(cfg_payload.get("macro_max_instantiations", 10)),
        macro_max_branch_per_op=int(cfg_payload.get("macro_max_branch_per_op", 10)),
        concept_propose_max_depth=int(cfg_payload.get("concept_propose_max_depth", 99)),
        concept_max_templates=int(cfg_payload.get("concept_max_templates", 24)),
        concept_max_instantiations=int(cfg_payload.get("concept_max_instantiations", 10)),
        concept_max_branch_per_op=int(cfg_payload.get("concept_max_branch_per_op", 10)),
        enable_trace_csg_induction=bool(trace_csg_induction),
        trace_csg_induction_first_pass_frac=float(trace_csg_first_pass_frac),
        enable_repair_stage=bool(enable_repair_stage),
        enable_residual_stage=bool(enable_residual_stage),
        enable_refine_stage=bool(enable_refine_stage),
        enable_point_patch_repair=bool(enable_point_patch_repair),
        point_patch_max_points=int(point_patch_max_points),
    )

    from dataclasses import replace

    def _extract_candidate_programs(res: Any) -> List[Dict[str, Any]]:
        cand_programs: List[Dict[str, Any]] = []
        if not isinstance(res, dict):
            return cand_programs

        cps = res.get("candidate_programs")
        if isinstance(cps, list):
            for row in cps:
                if not isinstance(row, dict):
                    continue
                steps = row.get("program_steps")
                if not isinstance(steps, list) or not steps:
                    continue
                cand_programs.append(
                    {
                        "program_sig": str(row.get("program_sig") or ""),
                        "program_cost_bits": int(row.get("program_cost_bits") or 0),
                        "program_steps": [dict(s) for s in steps if isinstance(s, dict)],
                        "predicted_grid": row.get("predicted_grid"),
                        "predicted_grid_hash": str(row.get("predicted_grid_hash") or ""),
                    }
                )

        if not cand_programs:
            ps = res.get("program_steps")
            if isinstance(ps, list) and ps:
                cand_programs.append(
                    {
                        "program_sig": str(res.get("program_sig") or ""),
                        "program_cost_bits": int(res.get("program_cost_bits") or 0),
                        "program_steps": [dict(s) for s in ps if isinstance(s, dict)],
                        "predicted_grid": res.get("predicted_grid"),
                        "predicted_grid_hash": str(res.get("predicted_grid_hash") or ""),
                    }
                )

        return cand_programs

    def _merge_candidate_programs(
        cands_by_variant: Sequence[Sequence[Dict[str, Any]]],
        *,
        max_total: int,
    ) -> List[Dict[str, Any]]:
        want_total = max(0, int(max_total))
        if want_total <= 0:
            return []
        max_len = 0
        for cands in list(cands_by_variant):
            if isinstance(cands, list):
                max_len = max(max_len, int(len(cands)))

        out: List[Dict[str, Any]] = []
        seen: set = set()

        for rank in range(int(max_len)):
            for cands in list(cands_by_variant):
                if not isinstance(cands, list):
                    continue
                if rank >= len(cands):
                    continue
                cand = cands[rank]
                if not isinstance(cand, dict):
                    continue
                sig = str(cand.get("program_sig") or "")
                if not sig:
                    steps0 = cand.get("program_steps")
                    if isinstance(steps0, list) and steps0:
                        sig = "steps:" + hashlib.sha256(_stable_json(steps0).encode("utf-8")).hexdigest()
                if sig and sig in seen:
                    continue
                if sig:
                    seen.add(sig)
                out.append(cand)
                if len(out) >= int(want_total):
                    return out

        return out

    def _ensemble_cfgs(base_cfg: SolveConfigV141) -> List[SolveConfigV141]:
        if not bool(ensemble):
            return [base_cfg]

        variants: List[SolveConfigV141] = [base_cfg]
        variants.append(replace(base_cfg, abstraction_pressure=not bool(base_cfg.abstraction_pressure)))
        variants.append(
            replace(
                base_cfg,
                macro_try_on_fail_only=False,
                macro_propose_max_depth=max(int(getattr(base_cfg, "macro_propose_max_depth", 0) or 0), 2),
            )
        )

        # Use at most K variants (K=max_trials); scoring only considers the first K candidates.
        variants = variants[: max(1, int(base_cfg.max_ambiguous_outputs))]

        # De-dupe identical configs (deterministic).
        uniq: List[SolveConfigV141] = []
        seen_keys: set = set()
        for v in variants:
            k = (
                int(v.max_depth),
                int(v.max_programs),
                int(v.max_ambiguous_outputs),
                bool(v.abstraction_pressure),
                bool(v.macro_try_on_fail_only),
                int(v.macro_propose_max_depth),
                bool(v.enable_reachability_pruning),
                bool(v.enable_trace_csg_induction),
                float(v.trace_csg_induction_first_pass_frac),
                bool(v.enable_repair_stage),
                bool(v.enable_residual_stage),
                bool(v.enable_refine_stage),
                bool(v.enable_point_patch_repair),
                int(v.point_patch_max_points),
            )
            if k in seen_keys:
                continue
            seen_keys.add(k)
            uniq.append(v)
        variants = uniq

        # When no wall-time cap is provided, split max_programs across variants to keep
        # total search budget roughly constant.
        if float(timeout_s) <= 0.0 and len(variants) > 1:
            total = int(base_cfg.max_programs)
            n = int(len(variants))
            q = max(1, int(total // n))
            rem = int(total - (q * n))
            out: List[SolveConfigV141] = []
            for i, v in enumerate(variants):
                budget = int(q + (1 if i < rem else 0))
                out.append(replace(v, max_programs=max(1, int(budget))))
            variants = out

        return variants

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

    solver_results_by_test: List[Dict[str, Any]] = []

    def _timeout_result() -> Dict[str, Any]:
        tps = get_trace_snapshot_v141()
        tms = get_trace_metrics_snapshot_v141()
        trace_obj: Dict[str, Any] = {"timeout_s": float(timeout_s), "trace_programs": list(tps)}
        if isinstance(tms, dict):
            # Include key causal-gate telemetry even when the harness aborts on wall-time.
            for k in sorted(tms.keys()):
                trace_obj[str(k)] = tms[k]
        return {
            "schema_version": 141,
            "kind": "arc_solver_result_v141",
            "status": "FAIL",
            "program_sig": "",
            "predicted_grid_hash": "",
            "failure_reason": {"kind": "SEARCH_BUDGET_EXCEEDED", "details": {"timeout_s": float(timeout_s)}},
            "trace": dict(trace_obj),
        }

    if timeout_s > 0.0:
        import signal

        def _handler(_signum: int, _frame: Any) -> None:  # type: ignore[valid-type]
            raise _ArcTaskTimeoutV141()

        prev = signal.signal(signal.SIGALRM, _handler)
        try:
            if len(test_ins2) > 1:
                # Multi-test ARC tasks (ARC-AGI-2): solve once (using test0) then apply the same
                # candidate program(s) to all test inputs to avoid N× repeated search.
                base_res: Dict[str, Any] = {}
                cand_lists: List[List[Dict[str, Any]]] = []
                seen_cands: set = set()
                deadline = time.monotonic() + float(timeout_s)
                for vcfg in _ensemble_cfgs(cfg):
                    remaining = float(deadline - time.monotonic())
                    if remaining <= 0.0:
                        break
                    signal.setitimer(signal.ITIMER_REAL, float(remaining))
                    reset_trace_snapshot_v141()
                    try:
                        res = solve_arc_task_v141(train_pairs=list(train_pairs), test_in=test_ins2[0], config=vcfg)
                    except _ArcTaskTimeoutV141:
                        res = _timeout_result()
                    except Exception as e:
                        res = _exception_result(e)
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0.0)
                    if not base_res and isinstance(res, dict):
                        base_res = dict(res)
                    cands = _extract_candidate_programs(res)
                    cand_lists.append(list(cands))
                    for cand in cands:
                        if not isinstance(cand, dict):
                            continue
                        sig0 = str(cand.get("program_sig") or "")
                        if not sig0:
                            steps0 = cand.get("program_steps")
                            if isinstance(steps0, list) and steps0:
                                sig0 = "steps:" + hashlib.sha256(_stable_json(steps0).encode("utf-8")).hexdigest()
                        if sig0:
                            seen_cands.add(str(sig0))
                    # Portfolio early-stop: if a variant solved train, stop spending wall time on other variants.
                    if isinstance(res, dict) and str(res.get("status") or "") == "SOLVED":
                        base_res = dict(res)
                        cand_lists = [list(cands)]
                        break
                    # Stop once we already have enough distinct candidates for scoring.
                    if len(seen_cands) >= int(cfg.max_ambiguous_outputs):
                        break
                if not base_res:
                    base_res = _timeout_result()

                cand_programs = _merge_candidate_programs(cand_lists, max_total=int(cfg.max_ambiguous_outputs))
                cand_steps: List[List[Dict[str, Any]]] = [
                    [dict(s) for s in c.get("program_steps") if isinstance(s, dict)] for c in cand_programs
                ]

                for ti in test_ins2:
                    # Build per-test predicted outputs aligned by candidate index.
                    predicted_grids: List[Any] = []
                    predicted_grid_hash = ""
                    predicted_grid: Any = None
                    if cand_steps:
                        for steps in cand_steps:
                            try:
                                out_g = apply_program_steps_to_grid_v141(grid=ti, program_steps=steps)
                                out = [list(r) for r in out_g]
                                predicted_grids.append(out)
                            except Exception:
                                predicted_grids.append(None)
                        if isinstance(predicted_grids[0], list):
                            predicted_grid = predicted_grids[0]
                            try:
                                predicted_grid_hash = str(grid_hash_v124(tuple(tuple(int(v) for v in row) for row in predicted_grid)))
                            except Exception:
                                predicted_grid_hash = ""

                    sr = dict(base_res) if isinstance(base_res, dict) else {}
                    # Override test-dependent output fields.
                    if predicted_grid is not None:
                        sr["predicted_grid"] = predicted_grid
                        sr["predicted_grid_hash"] = str(predicted_grid_hash)
                    if predicted_grids:
                        sr["predicted_grids"] = list(predicted_grids)
                    if cand_steps:
                        sr["candidate_program_steps"] = [list(s) for s in cand_steps]
                        sr["program_steps"] = list(cand_steps[0])
                        if cand_programs:
                            sr["program_sig"] = str(cand_programs[0].get("program_sig") or "")
                            sr["program_cost_bits"] = int(cand_programs[0].get("program_cost_bits") or 0)
                    solver_results_by_test.append(dict(sr))
            else:
                for test_in in test_ins2:
                    base_res: Dict[str, Any] = {}
                    cand_lists: List[List[Dict[str, Any]]] = []
                    seen_cands: set = set()
                    deadline = time.monotonic() + float(timeout_s)
                    for vcfg in _ensemble_cfgs(cfg):
                        remaining = float(deadline - time.monotonic())
                        if remaining <= 0.0:
                            break
                        signal.setitimer(signal.ITIMER_REAL, float(remaining))
                        reset_trace_snapshot_v141()
                        try:
                            res = solve_arc_task_v141(train_pairs=list(train_pairs), test_in=test_in, config=vcfg)
                        except _ArcTaskTimeoutV141:
                            res = _timeout_result()
                        except Exception as e:
                            res = _exception_result(e)
                        finally:
                            signal.setitimer(signal.ITIMER_REAL, 0.0)
                        if not base_res and isinstance(res, dict):
                            base_res = dict(res)
                        cands = _extract_candidate_programs(res)
                        cand_lists.append(list(cands))
                        for cand in cands:
                            if not isinstance(cand, dict):
                                continue
                            sig0 = str(cand.get("program_sig") or "")
                            if not sig0:
                                steps0 = cand.get("program_steps")
                                if isinstance(steps0, list) and steps0:
                                    sig0 = "steps:" + hashlib.sha256(_stable_json(steps0).encode("utf-8")).hexdigest()
                            if sig0:
                                seen_cands.add(str(sig0))
                        if isinstance(res, dict) and str(res.get("status") or "") == "SOLVED":
                            base_res = dict(res)
                            cand_lists = [list(cands)]
                            break
                        if len(seen_cands) >= int(cfg.max_ambiguous_outputs):
                            break
                    if not base_res:
                        base_res = _timeout_result()

                    cand_programs = _merge_candidate_programs(cand_lists, max_total=int(cfg.max_ambiguous_outputs))
                    sr = dict(base_res) if isinstance(base_res, dict) else {}
                    if cand_programs:
                        predicted_grids: List[Any] = []
                        cand_steps: List[List[Dict[str, Any]]] = []
                        for cp in cand_programs:
                            steps = cp.get("program_steps") if isinstance(cp.get("program_steps"), list) else []
                            steps2 = [dict(s) for s in steps if isinstance(s, dict)]
                            cand_steps.append(list(steps2))
                            pg = cp.get("predicted_grid")
                            if not isinstance(pg, list) and steps2:
                                try:
                                    out_g = apply_program_steps_to_grid_v141(grid=test_in, program_steps=steps2)
                                    pg = [list(r) for r in out_g]
                                except Exception:
                                    pg = None
                            predicted_grids.append(pg if isinstance(pg, list) else None)

                        predicted_grid_hash = ""
                        predicted_grid: Any = None
                        if predicted_grids and isinstance(predicted_grids[0], list):
                            predicted_grid = predicted_grids[0]
                            predicted_grid_hash = str(cand_programs[0].get("predicted_grid_hash") or "")
                            if predicted_grid_hash == "":
                                try:
                                    predicted_grid_hash = str(
                                        grid_hash_v124(tuple(tuple(int(v) for v in row) for row in predicted_grid))
                                    )
                                except Exception:
                                    predicted_grid_hash = ""

                        if predicted_grid is not None:
                            sr["predicted_grid"] = predicted_grid
                            sr["predicted_grid_hash"] = str(predicted_grid_hash)
                        if predicted_grids:
                            sr["predicted_grids"] = list(predicted_grids)
                        if cand_steps:
                            sr["candidate_program_steps"] = [list(s) for s in cand_steps]
                            sr["program_steps"] = list(cand_steps[0])
                            sr["program_sig"] = str(cand_programs[0].get("program_sig") or "")
                            sr["program_cost_bits"] = int(cand_programs[0].get("program_cost_bits") or 0)
                            sr["candidate_programs"] = [
                                {
                                    "program_sig": str(cp.get("program_sig") or ""),
                                    "program_cost_bits": int(cp.get("program_cost_bits") or 0),
                                    "program_steps": list(cp.get("program_steps") or []),
                                    "predicted_grid": (predicted_grids[i] if i < len(predicted_grids) else None),
                                    "predicted_grid_hash": str(cp.get("predicted_grid_hash") or ""),
                                }
                                for i, cp in enumerate(cand_programs)
                            ]

                    solver_results_by_test.append(dict(sr))
        finally:
            signal.signal(signal.SIGALRM, prev)
    else:
        if len(test_ins2) > 1:
            base_res: Dict[str, Any] = {}
            cand_lists: List[List[Dict[str, Any]]] = []
            seen_cands: set = set()
            for vcfg in _ensemble_cfgs(cfg):
                reset_trace_snapshot_v141()
                try:
                    res = solve_arc_task_v141(train_pairs=list(train_pairs), test_in=test_ins2[0], config=vcfg)
                except Exception as e:
                    res = _exception_result(e)
                if not base_res and isinstance(res, dict):
                    base_res = dict(res)
                cands = _extract_candidate_programs(res)
                cand_lists.append(list(cands))
                for cand in cands:
                    if not isinstance(cand, dict):
                        continue
                    sig0 = str(cand.get("program_sig") or "")
                    if not sig0:
                        steps0 = cand.get("program_steps")
                        if isinstance(steps0, list) and steps0:
                            sig0 = "steps:" + hashlib.sha256(_stable_json(steps0).encode("utf-8")).hexdigest()
                    if sig0:
                        seen_cands.add(str(sig0))
                if isinstance(res, dict) and str(res.get("status") or "") == "SOLVED":
                    base_res = dict(res)
                    cand_lists = [list(cands)]
                    break
                if len(seen_cands) >= int(cfg.max_ambiguous_outputs):
                    break
            if not base_res:
                base_res = _exception_result(RuntimeError("missing_base_res_v141"))

            cand_programs = _merge_candidate_programs(cand_lists, max_total=int(cfg.max_ambiguous_outputs))
            cand_steps: List[List[Dict[str, Any]]] = [
                [dict(s) for s in c.get("program_steps") if isinstance(s, dict)] for c in cand_programs
            ]

            for ti in test_ins2:
                predicted_grids: List[Any] = []
                predicted_grid_hash = ""
                predicted_grid: Any = None
                if cand_steps:
                    for steps in cand_steps:
                        try:
                            out_g = apply_program_steps_to_grid_v141(grid=ti, program_steps=steps)
                            predicted_grids.append([list(r) for r in out_g])
                        except Exception:
                            predicted_grids.append(None)
                    if isinstance(predicted_grids[0], list):
                        predicted_grid = predicted_grids[0]
                        try:
                            predicted_grid_hash = str(grid_hash_v124(tuple(tuple(int(v) for v in row) for row in predicted_grid)))
                        except Exception:
                            predicted_grid_hash = ""

                sr = dict(base_res) if isinstance(base_res, dict) else {}
                if predicted_grid is not None:
                    sr["predicted_grid"] = predicted_grid
                    sr["predicted_grid_hash"] = str(predicted_grid_hash)
                if predicted_grids:
                    sr["predicted_grids"] = list(predicted_grids)
                if cand_steps:
                    sr["candidate_program_steps"] = [list(s) for s in cand_steps]
                    sr["program_steps"] = list(cand_steps[0])
                    if cand_programs:
                        sr["program_sig"] = str(cand_programs[0].get("program_sig") or "")
                        sr["program_cost_bits"] = int(cand_programs[0].get("program_cost_bits") or 0)
                solver_results_by_test.append(dict(sr))
        else:
            for test_in in test_ins2:
                base_res: Dict[str, Any] = {}
                cand_lists: List[List[Dict[str, Any]]] = []
                seen_cands: set = set()
                for vcfg in _ensemble_cfgs(cfg):
                    try:
                        reset_trace_snapshot_v141()
                        res = solve_arc_task_v141(train_pairs=list(train_pairs), test_in=test_in, config=vcfg)
                    except Exception as e:
                        res = _exception_result(e)
                    if not base_res and isinstance(res, dict):
                        base_res = dict(res)
                    cands = _extract_candidate_programs(res)
                    cand_lists.append(list(cands))
                    for cand in cands:
                        if not isinstance(cand, dict):
                            continue
                        sig0 = str(cand.get("program_sig") or "")
                        if not sig0:
                            steps0 = cand.get("program_steps")
                            if isinstance(steps0, list) and steps0:
                                sig0 = "steps:" + hashlib.sha256(_stable_json(steps0).encode("utf-8")).hexdigest()
                        if sig0:
                            seen_cands.add(str(sig0))
                    if isinstance(res, dict) and str(res.get("status") or "") == "SOLVED":
                        base_res = dict(res)
                        cand_lists = [list(cands)]
                        break
                    if len(seen_cands) >= int(cfg.max_ambiguous_outputs):
                        break
                if not base_res:
                    base_res = _exception_result(RuntimeError("missing_base_res_v141"))

                cand_programs = _merge_candidate_programs(cand_lists, max_total=int(cfg.max_ambiguous_outputs))
                sr = dict(base_res) if isinstance(base_res, dict) else {}
                if cand_programs:
                    predicted_grids2: List[Any] = []
                    cand_steps2: List[List[Dict[str, Any]]] = []
                    for cp in cand_programs:
                        steps = cp.get("program_steps") if isinstance(cp.get("program_steps"), list) else []
                        steps2 = [dict(s) for s in steps if isinstance(s, dict)]
                        cand_steps2.append(list(steps2))
                        pg = cp.get("predicted_grid")
                        if not isinstance(pg, list) and steps2:
                            try:
                                out_g = apply_program_steps_to_grid_v141(grid=test_in, program_steps=steps2)
                                pg = [list(r) for r in out_g]
                            except Exception:
                                pg = None
                        predicted_grids2.append(pg if isinstance(pg, list) else None)

                    predicted_grid_hash = ""
                    predicted_grid: Any = None
                    if predicted_grids2 and isinstance(predicted_grids2[0], list):
                        predicted_grid = predicted_grids2[0]
                        predicted_grid_hash = str(cand_programs[0].get("predicted_grid_hash") or "")
                        if predicted_grid_hash == "":
                            try:
                                predicted_grid_hash = str(
                                    grid_hash_v124(tuple(tuple(int(v) for v in row) for row in predicted_grid))
                                )
                            except Exception:
                                predicted_grid_hash = ""

                    if predicted_grid is not None:
                        sr["predicted_grid"] = predicted_grid
                        sr["predicted_grid_hash"] = str(predicted_grid_hash)
                    if predicted_grids2:
                        sr["predicted_grids"] = list(predicted_grids2)
                    if cand_steps2:
                        sr["candidate_program_steps"] = [list(s) for s in cand_steps2]
                        sr["program_steps"] = list(cand_steps2[0])
                        sr["program_sig"] = str(cand_programs[0].get("program_sig") or "")
                        sr["program_cost_bits"] = int(cand_programs[0].get("program_cost_bits") or 0)
                        sr["candidate_programs"] = [
                            {
                                "program_sig": str(cp.get("program_sig") or ""),
                                "program_cost_bits": int(cp.get("program_cost_bits") or 0),
                                "program_steps": list(cp.get("program_steps") or []),
                                "predicted_grid": (predicted_grids2[i] if i < len(predicted_grids2) else None),
                                "predicted_grid_hash": str(cp.get("predicted_grid_hash") or ""),
                            }
                            for i, cp in enumerate(cand_programs)
                        ]

                solver_results_by_test.append(dict(sr))

    return {"task_id": task_id, "solver_results": solver_results_by_test}


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
    task_ids: Optional[Sequence[str]],
    seed: int,
    jobs: int,
    max_depth: int,
    max_programs: int,
    solution_cost_slack_bits: int,
    max_trials: int,
    benchmark_profile: str,
    macro_templates_path: Optional[str],
    concept_templates_path: Optional[str],
    require_concept_call: bool,
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
    concept_propose_max_depth: int,
    concept_max_templates: int,
    concept_max_instantiations: int,
    concept_max_branch_per_op: int,
    abstraction_pressure: bool,
    csv_allow_slot_progress: bool,
    concept_support_feat_ranking: bool,
    trace_csg_induction: bool,
    trace_csg_first_pass_frac: float,
    enable_reachability_pruning: bool,
    enable_point_patch_repair: bool,
    point_patch_max_points: int,
    ensemble: bool,
    omega_enabled: bool,
    omega_state_in: Optional[str],
) -> Dict[str, Any]:
    resume = bool(resume)
    require_concept_call = bool(require_concept_call)
    ensemble = bool(ensemble)
    csv_allow_slot_progress = bool(csv_allow_slot_progress)
    concept_support_feat_ranking = bool(concept_support_feat_ranking)
    macro_propose_max_depth = max(0, int(macro_propose_max_depth))
    macro_max_templates = max(0, int(macro_max_templates))
    macro_max_instantiations = max(0, int(macro_max_instantiations))
    macro_max_branch_per_op = max(0, int(macro_max_branch_per_op))
    concept_propose_max_depth = max(0, int(concept_propose_max_depth))
    concept_max_templates = max(0, int(concept_max_templates))
    concept_max_instantiations = max(0, int(concept_max_instantiations))
    concept_max_branch_per_op = max(0, int(concept_max_branch_per_op))
    trace_csg_induction = bool(trace_csg_induction)
    trace_csg_first_pass_frac = float(trace_csg_first_pass_frac)
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

    task_ids_norm: List[str] = []
    if task_ids is not None:
        for x in task_ids:
            s = str(x or "").strip().replace("\\", "/").lstrip("/")
            if not s:
                continue
            if not s.endswith(".json"):
                s = s + ".json"
            task_ids_norm.append(str(s))

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
            "limit": int(len(task_ids_norm) if task_ids_norm else int(limit)),
            "seed": int(seed),
            "selection_mode": "explicit" if task_ids_norm else ("shuffled" if int(limit) > 0 else "sorted"),
            "task_ids": list(task_ids_norm),
        }
        got = {
            "split": manifest_obj.get("split") if "split" in manifest_obj else None,
            "limit": int(manifest_obj.get("limit") or 0),
            "seed": (manifest_obj.get("seed") if "seed" in manifest_obj else None),
            "selection_mode": str(manifest_obj.get("selection_mode") or "sorted"),
            "task_ids": list(manifest_obj.get("task_ids") or []),
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
            task_ids=(task_ids_norm if task_ids_norm else None),
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

    if bool(concept_support_feat_ranking):
        _enrich_concept_support_task_feat_keys_v141(concept_templates_rows=concept_templates_rows, arc_root=str(arc_root))

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
        csv_allow_slot_progress=bool(csv_allow_slot_progress),
        macro_try_on_fail_only=bool(macro_try_on_fail_only),
        enable_reachability_pruning=bool(enable_reachability_pruning),
        enable_concept_support_feat_ranking=bool(concept_support_feat_ranking),
        macro_propose_max_depth=int(macro_propose_max_depth),
        macro_max_templates=int(macro_max_templates),
        macro_max_instantiations=int(macro_max_instantiations),
        macro_max_branch_per_op=int(macro_max_branch_per_op),
        concept_propose_max_depth=int(concept_propose_max_depth),
        concept_max_templates=int(concept_max_templates),
        concept_max_instantiations=int(concept_max_instantiations),
        concept_max_branch_per_op=int(concept_max_branch_per_op),
        enable_trace_csg_induction=bool(trace_csg_induction),
        trace_csg_induction_first_pass_frac=float(trace_csg_first_pass_frac),
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
    want_by_id: Dict[str, List[Optional[Sequence[Sequence[int]]]]] = {
        str(t.task_id): [out for _inp, out in t.test_pairs] for t in tasks
    }
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

    def write_per_task_row(
        *, task_id: str, solver_results: Sequence[Dict[str, Any]], scoring: Dict[str, Any]
    ) -> None:
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
            "solver_results": [dict(r) for r in solver_results if isinstance(r, dict)],
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
        from atos_core.arc_solver_v141 import (
            get_trace_metrics_snapshot_v141,
            get_trace_snapshot_v141,
            reset_trace_snapshot_v141,
        )

        def _timeout_result() -> Dict[str, Any]:
            tps = get_trace_snapshot_v141()
            tms = get_trace_metrics_snapshot_v141()
            trace_obj: Dict[str, Any] = {"timeout_s": float(task_timeout_s), "trace_programs": list(tps)}
            if isinstance(tms, dict):
                for k in sorted(tms.keys()):
                    trace_obj[str(k)] = tms[k]
            return {
                "schema_version": 141,
                "kind": "arc_solver_result_v141",
                "status": "FAIL",
                "program_sig": "",
                "predicted_grid_hash": "",
                "failure_reason": {"kind": "SEARCH_BUDGET_EXCEEDED", "details": {"timeout_s": float(task_timeout_s)}},
                "trace": dict(trace_obj),
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
            solver_results_by_test: List[Dict[str, Any]] = []
            for test_idx, (test_in, _want) in enumerate(task.test_pairs):
                if float(task_timeout_s) > 0.0:
                    signal.setitimer(signal.ITIMER_REAL, float(task_timeout_s))
                reset_trace_snapshot_v141()
                try:
                    solver_res = solve_arc_task_v141(
                        train_pairs=list(task.train_pairs),
                        test_in=test_in,
                        config=cfg,
                    )
                except _ArcTaskTimeoutV141:
                    solver_res = _timeout_result()
                except Exception as e:
                    solver_res = _exception_result(e)
                finally:
                    if float(task_timeout_s) > 0.0:
                        signal.setitimer(signal.ITIMER_REAL, 0.0)
                solver_results_by_test.append(dict(solver_res))

            item = {"task_id": str(task.task_id), "solver_results": solver_results_by_test}
            solver_results.append(item)
            scoring = _score_task_all_tests_all_k_v141(
                solver_results=solver_results_by_test,
                want_grids=want_by_id.get(str(task.task_id)) or [],
                max_trials=int(max_trials),
                require_concept_call=bool(require_concept_call),
            )
            scoring_by_id[str(task.task_id)] = scoring
            write_per_task_row(task_id=str(task.task_id), solver_results=solver_results_by_test, scoring=scoring)
            fk = str(scoring.get("failure_kind") or "")
            failures_so_far[fk] = int(failures_so_far.get(fk, 0)) + 1
            by_k = scoring.get("scores_by_k") if isinstance(scoring.get("scores_by_k"), dict) else {}
            ok_item = by_k.get(str(int(max_trials))) if isinstance(by_k.get(str(int(max_trials))), dict) else {}
            if bool(ok_item.get("ok")):
                solved_kmax += 1
            done += 1
            last_progress = time.monotonic()
            last_status = str(scoring.get("solver_status") or "")
            log_progress(
                {
                    "done": int(done),
                    "total": int(total_tasks),
                    "solved_kmax": int(solved_kmax),
                    "last_task_id": str(task.task_id),
                    "last_status": str(last_status),
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
            "require_concept_call": bool(require_concept_call),
            "task_timeout_s": float(task_timeout_s),
            "enable_repair_stage": bool(enable_repair_stage),
            "enable_residual_stage": bool(enable_residual_stage),
            "enable_refine_stage": bool(enable_refine_stage),
            "macro_try_on_fail_only": bool(macro_try_on_fail_only),
            "macro_propose_max_depth": int(macro_propose_max_depth),
            "macro_max_templates": int(macro_max_templates),
            "macro_max_instantiations": int(macro_max_instantiations),
            "macro_max_branch_per_op": int(macro_max_branch_per_op),
            "concept_propose_max_depth": int(concept_propose_max_depth),
            "concept_max_templates": int(concept_max_templates),
            "concept_max_instantiations": int(concept_max_instantiations),
            "concept_max_branch_per_op": int(concept_max_branch_per_op),
            "abstraction_pressure": bool(abstraction_pressure),
            "enable_concept_support_feat_ranking": bool(concept_support_feat_ranking),
            # learn mode: allow slot-building semantic moves (dl==0, ds<0) so trace_programs
            # contain non-empty pipelines for downstream CSG mining. eval_arc keeps this False.
            "csv_allow_slot_progress": bool(csv_allow_slot_progress),
            "trace_csg_induction": bool(trace_csg_induction),
            "trace_csg_first_pass_frac": float(trace_csg_first_pass_frac),
            "enable_reachability_pruning": bool(enable_reachability_pruning),
            "enable_point_patch_repair": bool(enable_point_patch_repair),
            "point_patch_max_points": int(point_patch_max_points),
            "ensemble": bool(ensemble),
        }
        task_payloads: List[Dict[str, Any]] = []
        for task in tasks:
            if resume and per_task_path(str(task.task_id)).exists():
                continue
            task_payloads.append(
                {
                    "task_id": str(task.task_id),
                    "train_pairs": list(task.train_pairs),
                    "test_ins": [inp for inp, _out in task.test_pairs],
                }
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
                        item = {"task_id": task_id_hint, "solver_results": [_exception_result(e)]}
                    solver_results.append(item)
                    task_id = str(item.get("task_id") or "")
                    solver_results_by_test = item.get("solver_results")
                    if not isinstance(solver_results_by_test, list) or not solver_results_by_test:
                        sr0 = item.get("solver_res") if isinstance(item.get("solver_res"), dict) else {}
                        solver_results_by_test = [sr0]
                    solver_results_by_test2 = [sr for sr in solver_results_by_test if isinstance(sr, dict)]
                    if not solver_results_by_test2:
                        solver_results_by_test2 = [_exception_result(RuntimeError("missing_solver_results_v141"))]
                    scoring = _score_task_all_tests_all_k_v141(
                        solver_results=solver_results_by_test2,
                        want_grids=want_by_id.get(task_id) or [],
                        max_trials=int(max_trials),
                        require_concept_call=bool(require_concept_call),
                    )
                    scoring_by_id[task_id] = scoring
                    write_per_task_row(task_id=task_id, solver_results=solver_results_by_test2, scoring=scoring)
                    fk = str(scoring.get("failure_kind") or "")
                    failures_so_far[fk] = int(failures_so_far.get(fk, 0)) + 1
                    by_k = scoring.get("scores_by_k") if isinstance(scoring.get("scores_by_k"), dict) else {}
                    ok_item = by_k.get(str(int(max_trials))) if isinstance(by_k.get(str(int(max_trials))), dict) else {}
                    if bool(ok_item.get("ok")):
                        solved_kmax += 1
                    done += 1
                    last_progress = time.monotonic()
                    last_status = str(scoring.get("solver_status") or "")
                    log_progress(
                        {
                            "done": int(done),
                            "total": int(total_tasks),
                            "solved_kmax": int(solved_kmax),
                            "last_task_id": str(task_id),
                            "last_status": str(last_status),
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
                    solver_res0 = {
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
                    want_grids = want_by_id.get(task_id) or []
                    solver_results_by_test = [dict(solver_res0) for _ in range(max(1, int(len(want_grids))))]
                    scoring = _score_task_all_tests_all_k_v141(
                        solver_results=solver_results_by_test,
                        want_grids=want_grids,
                        max_trials=int(max_trials),
                        require_concept_call=bool(require_concept_call),
                    )
                    write_per_task_row(task_id=task_id, solver_results=solver_results_by_test, scoring=scoring)
                    scoring_by_id[task_id] = scoring
                    solver_results.append({"task_id": task_id, "solver_results": solver_results_by_test})
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
            solver_res0 = {
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
            want_grids = [out for _inp, out in task.test_pairs]
            solver_results_by_test = [dict(solver_res0) for _ in range(max(1, int(len(want_grids))))]
            scoring = _score_task_all_tests_all_k_v141(
                solver_results=solver_results_by_test,
                want_grids=want_grids,
                max_trials=int(max_trials),
                require_concept_call=bool(require_concept_call),
            )
            scoring_by_id[str(task.task_id)] = scoring
            item = {"task_id": str(task.task_id), "solver_results": solver_results_by_test}
            res_by_id[str(task.task_id)] = item
        solver_results_row = item.get("solver_results")
        if not isinstance(solver_results_row, list) or not solver_results_row:
            sr0 = item.get("solver_res") if isinstance(item.get("solver_res"), dict) else {}
            solver_results_row = [sr0]
        solver_results_by_test2 = [sr for sr in solver_results_row if isinstance(sr, dict)]
        if not solver_results_by_test2:
            raise RuntimeError(f"bad_solver_result_v141:{task.task_id}")
        scoring = scoring_by_id.get(str(task.task_id)) or _score_task_all_tests_all_k_v141(
            solver_results=solver_results_by_test2,
            want_grids=[out for _inp, out in task.test_pairs],
            max_trials=int(max_trials),
            require_concept_call=bool(require_concept_call),
        )
        row = {
            "schema_version": 141,
            "kind": "arc_per_task_v141",
            "task": task.to_dict(),
            "solver_results": [dict(r) for r in solver_results_by_test2 if isinstance(r, dict)],
            "scoring": scoring,
        }
        per_task_rows.append(row)

        write_per_task_row(task_id=str(task.task_id), solver_results=solver_results_by_test2, scoring=scoring)

    # Now that per_task rows are complete, generate per-task manifest and trace candidates deterministically.
    for row in per_task_rows:
        if not isinstance(row, dict):
            continue
        task_obj = row.get("task") if isinstance(row.get("task"), dict) else {}
        task_id = str(task_obj.get("task_id") or "")
        solver_results_row = row.get("solver_results")
        sr_list = [sr for sr in solver_results_row if isinstance(sr, dict)] if isinstance(solver_results_row, list) else []
        sr0 = sr_list[0] if sr_list else {}
        scoring = row.get("scoring") if isinstance(row.get("scoring"), dict) else {}

        solver_status = str(scoring.get("solver_status") or (sr0.get("status") or ""))
        fk = str(scoring.get("failure_kind") or "")
        program_sigs_by_test = [str(sr.get("program_sig") or "") for sr in sr_list]
        predicted_hashes_by_test = [str(sr.get("predicted_grid_hash") or "") for sr in sr_list]
        scoring_rows.append(
            {
                "schema_version": 141,
                "kind": "arc_per_task_manifest_row_v141",
                "task_id": str(task_id),
                "status": solver_status,
                "failure_kind": fk,
                "program_sig": str(sr0.get("program_sig") or ""),
                "predicted_grid_hash": str(sr0.get("predicted_grid_hash") or ""),
                "program_sigs_by_test": program_sigs_by_test,
                "predicted_grid_hashes_by_test": predicted_hashes_by_test,
            }
        )

        for test_index, sr in enumerate(sr_list):
            trace = sr.get("trace") if isinstance(sr.get("trace"), dict) else {}
            tps = trace.get("trace_programs", []) if isinstance(trace.get("trace_programs"), list) else []
            tps_dicts = [tp for tp in tps if isinstance(tp, dict)]
            tps_dicts.sort(key=lambda d: (str(d.get("program_sig") or ""), _stable_json(d)))
            for tp in tps_dicts:
                trace_candidates_rows.append(
                    {
                        "schema_version": 141,
                        "kind": "arc_trace_candidate_v141",
                        "task_id": str(task_id),
                        "test_index": int(test_index),
                        **{str(k): tp.get(k) for k in sorted(tp.keys())},
                    }
                )

            # Also record the final program itself as a trace candidate (when available).
            #
            # Rationale:
            # - In strict concept-as-policy regimes, many tasks solve via a single high-level concept_call
            #   with an internal multi-step CSG body. The solver does not always populate trace_programs
            #   for successful runs (to keep per_task logs bounded), but we still need a deterministic
            #   corpus of concrete programs for downstream CSG mining in learn-mode.
            # - ARC eval remains read-only; emitting trace candidates does not mutate banks.
            #
            # Only emit for solved tasks to avoid polluting mining with arbitrary wrong outputs.
            if str(solver_status) == "SOLVED":
                final_steps = sr.get("program_steps")
                if isinstance(final_steps, list) and final_steps:
                    trace_candidates_rows.append(
                        {
                            "schema_version": 141,
                            "kind": "arc_trace_candidate_v141",
                            "task_id": str(task_id),
                            "test_index": int(test_index),
                            "program_sig": str(sr.get("program_sig") or ""),
                            "cost_bits": int(sr.get("program_cost_bits") or 0),
                            "loss": {"shape": 0, "cells": 0},
                            "steps": [dict(s) for s in final_steps if isinstance(s, dict)],
                            "source": "final_program",
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
            sr_list = [sr for sr in solver_results_row if isinstance(sr, dict)] if isinstance(solver_results_row, list) else []
            sr0 = sr_list[0] if sr_list else {}

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
                for sr in sr_list:
                    steps = sr.get("program_steps")
                    if isinstance(steps, list):
                        program_depth = max(int(program_depth or 0), int(len(steps)))
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
                    pcb = sr.get("program_cost_bits")
                    if isinstance(pcb, int):
                        program_cost_bits = max(int(program_cost_bits or 0), int(pcb))
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
        fk = str(r.get("failure_kind") or "")
        if fk:
            failure_counts[fk] = int(failure_counts.get(fk, 0)) + 1

    csv_tested_total = 0
    csv_rejected_total = 0
    for row in per_task_rows:
        if not isinstance(row, dict):
            continue
        solver_results_row = row.get("solver_results")
        sr_list = [sr for sr in solver_results_row if isinstance(sr, dict)] if isinstance(solver_results_row, list) else []
        sr0 = sr_list[0] if sr_list else {}
        trace = sr0.get("trace") if isinstance(sr0.get("trace"), dict) else {}
        csv_tested_total += int(trace.get("csv_tested") or 0)
        csv_rejected_total += int(trace.get("csv_rejected") or 0)
    csv_survival_count = max(0, int(csv_tested_total) - int(csv_rejected_total))
    csv_rejection_rate: Optional[float] = None
    if int(csv_tested_total) > 0:
        csv_rejection_rate = float(int(csv_rejected_total)) / float(int(csv_tested_total))

    score_summary = _summarize_scores_v141(per_task_rows, max_trials=int(max_trials))
    program_usage = _summarize_program_usage_v141(per_task_rows)
    eval_obj: Dict[str, Any] = {
        "schema_version": 141,
        "kind": "arc_eval_v141",
        "require_concept_call": bool(require_concept_call),
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
        "concept_support_feat_ranking": bool(concept_support_feat_ranking),
        "max_depth_requested": int(max_depth),
        "max_programs_requested": int(max_programs),
        "task_timeout_s": float(task_timeout_s),
        "no_progress_timeout_s": float(no_progress_timeout_s or 0.0),
        "resume": bool(resume),
        "benchmark_profile": str(benchmark_profile),
        "require_concept_call": bool(require_concept_call),
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
        "csv_tested_total": int(csv_tested_total),
        "csv_rejected_total": int(csv_rejected_total),
        "csv_survival_count": int(csv_survival_count),
        "csv_rejection_rate": float(csv_rejection_rate) if csv_rejection_rate is not None else None,
        "macro_try_on_fail_only": bool(macro_try_on_fail_only),
        "enable_reachability_pruning": bool(enable_reachability_pruning),
        "trace_csg_induction": bool(trace_csg_induction),
        "trace_csg_first_pass_frac": float(trace_csg_first_pass_frac),
        "macro_propose_max_depth": int(macro_propose_max_depth),
        "macro_max_templates": int(macro_max_templates),
        "macro_max_instantiations": int(macro_max_instantiations),
        "macro_max_branch_per_op": int(macro_max_branch_per_op),
        "concept_propose_max_depth": int(concept_propose_max_depth),
        "concept_max_templates": int(concept_max_templates),
        "concept_max_instantiations": int(concept_max_instantiations),
        "concept_max_branch_per_op": int(concept_max_branch_per_op),
        "enable_repair_stage": bool(enable_repair_stage),
        "enable_residual_stage": bool(enable_residual_stage),
        "enable_refine_stage": bool(enable_refine_stage),
        "enable_point_patch_repair": bool(enable_point_patch_repair),
        "point_patch_max_points": int(point_patch_max_points),
        "ensemble": bool(ensemble),
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
    isolation_ok = bool(isolation["ok"])
    if not isolation_ok:
        # Do not abort the run: record the violation and continue so automation can
        # consume the produced artifacts (summary/eval/per_task). The caller can
        # treat isolation_ok=false as a hard failure if desired.
        print(
            f"WARNING: isolation_violation_v141 (repo snapshot changed during run; see {out_dir / 'isolation_check_v141.json'})",
            file=sys.stderr,
            flush=True,
        )

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
        "isolation_ok": bool(isolation_ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["learn", "eval_arc"],
        default="learn",
        help=(
            "Execution mode guardrails. "
            "learn: allow learning-oriented features (only when explicitly enabled). "
            "eval_arc: ARC auditor mode (read-only): forbids any in-run induction/mutation features."
        ),
    )
    ap.add_argument("--arc_root", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--task_ids",
        default="",
        help=(
            "Optional comma-separated list of task_id filenames to run (e.g., 'd35bdbdc.json'). "
            "Overrides --limit/--seed dataset slice selection for WORM input manifest generation."
        ),
    )
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
        "--require_concept_call",
        type=int,
        default=0,
        choices=[0, 1],
        help="1: treat output-correct solutions without concept_call as FAIL (validator-like).",
    )
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
        "--ensemble",
        type=int,
        default=0,
        choices=[0, 1],
        help="1: run a small deterministic config ensemble per task and merge candidates (improves k-trials surface).",
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
        "--concept_propose_max_depth",
        type=int,
        default=99,
        help="Allow concept_call proposals up to this search depth (default: any depth).",
    )
    ap.add_argument(
        "--concept_max_templates",
        type=int,
        default=24,
        help="Max concept templates to consider per state when concept proposals are enabled.",
    )
    ap.add_argument(
        "--concept_max_instantiations",
        type=int,
        default=10,
        help="Max arg instantiations per concept template when proposing concept_call.",
    )
    ap.add_argument(
        "--concept_max_branch_per_op",
        type=int,
        default=10,
        help="Max arg branches per op_id within concept instantiation.",
    )
    ap.add_argument(
        "--abstraction_pressure",
        action="store_true",
        help="Enable abstraction-or-die pressure: prefer closure (concept_call) as soon as shape matches and classify shape-correct budget failures as MISSING_CONCEPT.",
    )
    ap.add_argument(
        "--concept_support_feat_ranking",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable training-only support-feature ranking for concept template retrieval (requires support_task_ids in the concept bank).",
    )
    ap.add_argument(
        "--trace_csg_induction",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable deterministic trace→CSG induction retry pass (collapses deep near-miss traces into atomic concept_call steps).",
    )
    ap.add_argument(
        "--trace_csg_first_pass_frac",
        type=float,
        default=0.6,
        help="Fraction of max_programs allocated to the first pass before inducing CSG from traces (rest is used for the retry pass).",
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

    # AUTO benchmark profile: if the user didn't explicitly pass --benchmark_profile, infer from arc_root.
    # This avoids accidentally evaluating ARC-AGI-2 runs under ARC-AGI-1 trial rules.
    if "--benchmark_profile" not in sys.argv:
        ar = str(getattr(args, "arc_root", "") or "").lower()
        if "agi2" in ar or "arc-agi-2" in ar or "arc_agi2" in ar:
            args.benchmark_profile = "ARC_AGI2_PROFILE"

    # MODE=EVAL_ARC is a hard guardrail required by the project docs:
    # ARC is an external auditor; evaluation must be read-only and must not persist new structure.
    #
    # Note: trace→CSG induction is permitted in eval mode because it is strictly:
    #   - per-task, in-memory only (no bank mutation / no WORM writes beyond the run artifacts)
    #   - deterministic given (seed, code, flags)
    #   - expressivity-preserving (no new primitives; it only wraps existing primitive prefixes)
    if str(getattr(args, "mode", "learn")) == "eval_arc":
        if bool(args.omega):
            raise SystemExit("eval_arc_forbids_omega_v141")

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

    task_ids: List[str] = []
    raw_task_ids = str(getattr(args, "task_ids", "") or "").strip()
    if raw_task_ids:
        for x in raw_task_ids.split(","):
            s = str(x or "").strip().replace("\\", "/").lstrip("/")
            if not s:
                continue
            if not s.endswith(".json"):
                s = s + ".json"
            task_ids.append(str(s))

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
            task_ids=(task_ids if task_ids else None),
            seed=int(args.seed),
            jobs=int(args.jobs),
            max_depth=int(args.max_depth),
            max_programs=int(args.max_programs),
            solution_cost_slack_bits=int(args.solution_cost_slack_bits),
            max_trials=int(max_trials),
            benchmark_profile=str(args.benchmark_profile),
            macro_templates_path=str(args.macro_templates) if args.macro_templates else None,
            concept_templates_path=str(args.concept_templates) if args.concept_templates else None,
            require_concept_call=bool(int(args.require_concept_call) != 0),
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
            concept_propose_max_depth=int(args.concept_propose_max_depth),
            concept_max_templates=int(args.concept_max_templates),
            concept_max_instantiations=int(args.concept_max_instantiations),
            concept_max_branch_per_op=int(args.concept_max_branch_per_op),
            abstraction_pressure=bool(args.abstraction_pressure),
            # NOTE: under abstraction pressure (concept-as-policy), the solver must be able to
            # perform bounded *enabling* semantic moves (slot builders / patch transforms) before
            # the first grid write reduces mismatch, otherwise many tasks become inexpressible at
            # fixed depth and fail immediately with csv_survivors==0 (MISSING_OPERATOR).
            csv_allow_slot_progress=bool(str(getattr(args, "mode", "learn")) == "learn")
            or bool(args.abstraction_pressure),
            concept_support_feat_ranking=bool(int(args.concept_support_feat_ranking) != 0),
            trace_csg_induction=bool(int(args.trace_csg_induction) != 0),
            trace_csg_first_pass_frac=float(args.trace_csg_first_pass_frac),
            enable_reachability_pruning=not bool(args.disable_reachability_pruning),
            enable_point_patch_repair=bool(args.enable_point_patch_repair),
            point_patch_max_points=int(args.point_patch_max_points),
            ensemble=bool(int(args.ensemble) != 0),
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
                task_ids=(task_ids if task_ids else None),
                seed=int(args.seed),
                jobs=int(args.jobs),
                max_depth=int(args.max_depth),
                max_programs=int(args.max_programs),
                solution_cost_slack_bits=int(args.solution_cost_slack_bits),
                max_trials=int(max_trials),
                benchmark_profile=str(args.benchmark_profile),
                macro_templates_path=str(args.macro_templates) if args.macro_templates else None,
                concept_templates_path=str(args.concept_templates) if args.concept_templates else None,
                require_concept_call=bool(int(args.require_concept_call) != 0),
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
                concept_propose_max_depth=int(args.concept_propose_max_depth),
                concept_max_templates=int(args.concept_max_templates),
                concept_max_instantiations=int(args.concept_max_instantiations),
                concept_max_branch_per_op=int(args.concept_max_branch_per_op),
                abstraction_pressure=bool(args.abstraction_pressure),
                csv_allow_slot_progress=bool(str(getattr(args, "mode", "learn")) == "learn")
                or bool(args.abstraction_pressure),
                concept_support_feat_ranking=bool(int(args.concept_support_feat_ranking) != 0),
                trace_csg_induction=bool(int(args.trace_csg_induction) != 0),
                trace_csg_first_pass_frac=float(args.trace_csg_first_pass_frac),
                enable_reachability_pruning=not bool(args.disable_reachability_pruning),
                enable_point_patch_repair=bool(args.enable_point_patch_repair),
                point_patch_max_points=int(args.point_patch_max_points),
                ensemble=bool(int(args.ensemble) != 0),
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
