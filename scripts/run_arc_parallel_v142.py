#!/usr/bin/env python3
"""
run_arc_parallel_v142.py - Parallel ARC solver runner with deterministic search.

This script runs the ARC solver using parallel deterministic search across
multiple CPU cores while maintaining full auditability and WORM compliance.

Usage:
    python3 scripts/run_arc_parallel_v142.py \
        --arc_root ARC-AGI \
        --split training \
        --limit 100 \
        --jobs 16 \
        --max_programs 2000 \
        --max_depth 4 \
        --seed 0 \
        --out_base results/run_arc_parallel_v142

Schema version: 142
"""

from __future__ import annotations

import argparse
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

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.parallel_planner_v142 import (
    PARALLEL_PLANNER_SCHEMA_VERSION_V142,
    ParallelPlannerV142,
    ParallelSearchConfigV142,
    ParallelSearchResultV142,
    write_parallel_search_to_ledger_v142,
)

RUN_ARC_PARALLEL_SCHEMA_VERSION_V142 = 142


def _install_sigint_sigterm_handler() -> None:
    """Install signal handlers for graceful shutdown."""

    def handler(signum: int, frame: Any) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _ensure_absent(path: Path) -> None:
    """WORM compliance: ensure file does not already exist."""
    if path.exists():
        raise SystemExit(f"worm_exists:{path}")


def _sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_once_json(path: Path, obj: Any) -> None:
    """Write JSON file once (WORM compliant)."""
    _ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise SystemExit(f"tmp_exists:{tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _sanitize_task_id(task_id: str) -> str:
    """Sanitize task ID for use in filenames."""
    s = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(task_id)])
    return s or "task"


def _grid_shape(g: Sequence[Sequence[int]]) -> Tuple[int, int]:
    """Get grid shape."""
    h = int(len(g))
    w = int(len(g[0])) if h > 0 else 0
    return (int(h), int(w))


def _unique_colors(g: Sequence[Sequence[int]]) -> Tuple[int, ...]:
    """Get unique colors in a grid."""
    out = set()
    for row in g:
        for v in row:
            out.add(int(v))
    return tuple(sorted(out))


def _load_arc_task(task_path: Path) -> Dict[str, Any]:
    """Load an ARC task from JSON file."""
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_train_pairs(task: Dict[str, Any]) -> List[Tuple[List[List[int]], List[List[int]]]]:
    """Extract training pairs from task data."""
    train_raw = task.get("train", [])
    pairs = []
    for row in train_raw:
        if isinstance(row, dict):
            inp = row.get("input")
            out = row.get("output")
            if isinstance(inp, list) and isinstance(out, list):
                pairs.append((inp, out))
    return pairs


def _extract_test_cases(task: Dict[str, Any]) -> List[Tuple[List[List[int]], Optional[List[List[int]]]]]:
    """Extract test cases from task data."""
    test_raw = task.get("test", [])
    cases = []
    for row in test_raw:
        if isinstance(row, dict):
            inp = row.get("input")
            out = row.get("output")  # May be None for evaluation
            if isinstance(inp, list):
                cases.append((inp, out if isinstance(out, list) else None))
    return cases


def _load_operator_bank(bank_path: Optional[Path]) -> List[Dict[str, Any]]:
    """Load operator/concept bank from JSONL file."""
    if bank_path is None or not bank_path.exists():
        return []
    ops = []
    with open(bank_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ops.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return ops


def _get_builtin_operators() -> List[Dict[str, Any]]:
    """Get builtin ARC operators from the solver."""
    # Import here to avoid circular imports
    from atos_core.arc_ops_v141 import OP_DEFS_V141

    ops = []
    for op_id, op_def in OP_DEFS_V141.items():
        ops.append(
            {
                "op_id": str(op_id),
                "reads": list(op_def.reads),
                "writes": list(op_def.writes),
                "base_cost_bits": int(op_def.base_cost_bits),
                "kind": "builtin_op",
            }
        )
    return ops


def _grid_equal(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> bool:
    """Check if two grids are equal."""
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if list(ra) != list(rb):
            return False
    return True


def run_parallel_solver(
    *,
    arc_root: str,
    split: str,
    limit: int,
    jobs: int,
    max_programs: int,
    max_depth: int,
    seed: int,
    out_base: str,
    concept_bank_path: Optional[str] = None,
    macro_bank_path: Optional[str] = None,
    tries: int = 2,
    detect_ambiguity: bool = True,
) -> Dict[str, Any]:
    """
    Run the parallel ARC solver on a dataset split.

    Args:
        arc_root: Path to ARC dataset root.
        split: Dataset split ("training" or "evaluation").
        limit: Maximum number of tasks to process (0 = all).
        jobs: Number of parallel workers.
        max_programs: Maximum programs to evaluate per task.
        max_depth: Maximum program depth.
        seed: Random seed for determinism.
        out_base: Output directory base path.
        concept_bank_path: Optional path to concept bank JSONL.
        macro_bank_path: Optional path to macro bank JSONL.
        tries: Number of prediction attempts per test case.
        detect_ambiguity: Whether to return UNKNOWN on ambiguous solutions.

    Returns:
        Summary dict with results.
    """
    start_time = time.monotonic()

    # Setup paths
    arc_root_path = Path(arc_root)
    data_dir = arc_root_path / "data" / split
    if not data_dir.is_dir():
        raise SystemExit(f"data_dir_not_found:{data_dir}")

    out_dir = Path(out_base)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load operator banks
    operators = _get_builtin_operators()
    if concept_bank_path:
        concept_ops = _load_operator_bank(Path(concept_bank_path))
        operators.extend(concept_ops)
    if macro_bank_path:
        macro_ops = _load_operator_bank(Path(macro_bank_path))
        operators.extend(macro_ops)

    # Setup parallel planner
    config = ParallelSearchConfigV142(
        num_workers=int(jobs),
        max_programs=int(max_programs),
        max_depth=int(max_depth),
        seed=int(seed),
        detect_ambiguity=bool(detect_ambiguity),
        enable_worm_logging=True,
    )
    planner = ParallelPlannerV142(config=config)

    # Find tasks
    task_files = sorted(data_dir.glob("*.json"))
    if int(limit) > 0:
        task_files = task_files[: int(limit)]

    print(f"[run_arc_parallel_v142] tasks={len(task_files)} jobs={jobs} max_programs={max_programs}")

    # Process tasks
    results: List[Dict[str, Any]] = []
    ledger_path = out_dir / "parallel_search_ledger.jsonl"
    prev_hash = ""

    solved_count = 0
    unknown_count = 0
    fail_count = 0
    failure_kinds: Dict[str, int] = {}

    for task_idx, task_path in enumerate(task_files):
        task_id = task_path.stem
        safe_task_id = _sanitize_task_id(task_id)

        try:
            task_data = _load_arc_task(task_path)
            train_pairs = _extract_train_pairs(task_data)
            test_cases = _extract_test_cases(task_data)

            if not train_pairs:
                print(f"  [{task_idx+1}/{len(task_files)}] {task_id}: skip (no train)")
                continue

            # Compute target features for pruning
            target_shapes = [_grid_shape(out) for _, out in train_pairs]
            target_palettes = [_unique_colors(out) for _, out in train_pairs]

            # Use mode shape/palette as target
            from collections import Counter

            shape_counts = Counter(target_shapes)
            palette_counts = Counter(target_palettes)
            target_shape = shape_counts.most_common(1)[0][0] if shape_counts else None
            target_palette = palette_counts.most_common(1)[0][0] if palette_counts else None

            task_start = time.monotonic()

            # Run parallel search
            test_results: List[Dict[str, Any]] = []
            task_status = "SOLVED"

            for test_idx, (test_input, test_output) in enumerate(test_cases):
                result = planner.search(
                    train_pairs=train_pairs,
                    test_input=test_input,
                    operators=operators,
                    target_shape=target_shape,
                    target_palette=target_palette,
                )

                # Score against ground truth if available
                correct = False
                if test_output is not None and result.predicted_grid is not None:
                    correct = _grid_equal(result.predicted_grid, test_output)

                # Write to ledger
                prev_hash = write_parallel_search_to_ledger_v142(
                    result=result,
                    task_id=f"{task_id}_test{test_idx}",
                    ledger_path=str(ledger_path),
                    prev_hash=prev_hash,
                )

                test_result = {
                    "test_index": int(test_idx),
                    "status": str(result.status),
                    "correct": bool(correct) if test_output is not None else None,
                    "programs_evaluated": int(result.total_programs_evaluated),
                    "wall_time_ms": int(result.wall_time_ms),
                    "ambiguity_detected": bool(result.ambiguity_detected),
                    "failure_reason": result.failure_reason,
                }
                test_results.append(test_result)

                if result.status != "SOLVED":
                    task_status = result.status

            task_time_ms = int((time.monotonic() - task_start) * 1000)

            # Update counters
            if task_status == "SOLVED":
                solved_count += 1
            elif task_status == "UNKNOWN":
                unknown_count += 1
            else:
                fail_count += 1
                if test_results:
                    fr = test_results[0].get("failure_reason")
                    if isinstance(fr, dict):
                        kind = str(fr.get("kind", "UNKNOWN"))
                        failure_kinds[kind] = failure_kinds.get(kind, 0) + 1

            results.append(
                {
                    "task_id": str(task_id),
                    "status": str(task_status),
                    "test_results": test_results,
                    "wall_time_ms": int(task_time_ms),
                }
            )

            print(
                f"  [{task_idx+1}/{len(task_files)}] {task_id}: {task_status} "
                f"({task_time_ms}ms, {sum(r.get('programs_evaluated', 0) for r in test_results)} progs)"
            )

        except Exception as e:
            print(f"  [{task_idx+1}/{len(task_files)}] {task_id}: ERROR - {e}")
            traceback.print_exc()
            fail_count += 1
            failure_kinds["EXCEPTION"] = failure_kinds.get("EXCEPTION", 0) + 1
            results.append(
                {
                    "task_id": str(task_id),
                    "status": "FAIL",
                    "error": str(e),
                    "test_results": [],
                    "wall_time_ms": 0,
                }
            )

    total_time_ms = int((time.monotonic() - start_time) * 1000)

    # Build summary
    summary = {
        "schema_version": int(RUN_ARC_PARALLEL_SCHEMA_VERSION_V142),
        "kind": "arc_parallel_run_summary_v142",
        "arc_root": str(arc_root),
        "split": str(split),
        "config": config.to_dict(),
        "tasks_total": len(task_files),
        "solved": int(solved_count),
        "unknown": int(unknown_count),
        "fail": int(fail_count),
        "failure_kinds": dict(failure_kinds),
        "wall_time_ms": int(total_time_ms),
        "ledger_path": str(ledger_path),
        "ledger_hash": str(prev_hash),
    }

    # Write summary
    summary_path = out_dir / "summary.json"
    _write_once_json(summary_path, summary)

    # Write detailed results
    results_path = out_dir / "results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"\n[run_arc_parallel_v142] Done: solved={solved_count} unknown={unknown_count} fail={fail_count}")
    print(f"  Total time: {total_time_ms}ms")
    print(f"  Summary: {summary_path}")
    print(f"  Ledger: {ledger_path}")

    return summary


def main() -> None:
    """Main entry point."""
    _install_sigint_sigterm_handler()

    parser = argparse.ArgumentParser(description="Run parallel ARC solver")
    parser.add_argument("--arc_root", type=str, default="ARC-AGI", help="Path to ARC dataset root")
    parser.add_argument("--split", type=str, default="training", help="Dataset split (training/evaluation)")
    parser.add_argument("--limit", type=int, default=0, help="Max tasks (0=all)")
    parser.add_argument("--jobs", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--max_programs", type=int, default=2000, help="Max programs per task")
    parser.add_argument("--max_depth", type=int, default=4, help="Max program depth")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out_base", type=str, required=True, help="Output directory")
    parser.add_argument("--concept_bank", type=str, default=None, help="Path to concept bank JSONL")
    parser.add_argument("--macro_bank", type=str, default=None, help="Path to macro bank JSONL")
    parser.add_argument("--tries", type=int, default=2, help="Prediction attempts per test")
    parser.add_argument("--detect_ambiguity", action="store_true", default=True, help="Return UNKNOWN on ambiguity")
    parser.add_argument("--no_detect_ambiguity", action="store_false", dest="detect_ambiguity")

    args = parser.parse_args()

    summary = run_parallel_solver(
        arc_root=args.arc_root,
        split=args.split,
        limit=args.limit,
        jobs=args.jobs,
        max_programs=args.max_programs,
        max_depth=args.max_depth,
        seed=args.seed,
        out_base=args.out_base,
        concept_bank_path=args.concept_bank,
        macro_bank_path=args.macro_bank,
        tries=args.tries,
        detect_ambiguity=args.detect_ambiguity,
    )

    # Print JSON summary for scripting
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
