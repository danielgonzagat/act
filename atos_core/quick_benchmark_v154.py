#!/usr/bin/env python
"""
quick_benchmark_v154.py - Quick parallel benchmark for ARC solver baseline

Measures the real baseline of arc_solver_v141 without concept requirements.
Uses multiprocessing for speed.

Usage:
    python -m atos_core.quick_benchmark_v154 --tasks 100 --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Worker function (must be at module level for pickling)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_task_worker(args: Tuple[str, List, Any, int, int]) -> Dict[str, Any]:
    """Worker function to solve a single task."""
    task_id, train_pairs, test_in, max_depth, max_programs = args
    
    from atos_core.arc_solver_v141 import solve_arc_task_v141, SolveConfigV141
    
    config = SolveConfigV141(
        max_depth=max_depth,
        max_programs=max_programs,
        abstraction_pressure=False,  # No concept requirement for baseline
    )
    
    t0 = time.monotonic()
    result = solve_arc_task_v141(
        train_pairs=train_pairs,
        test_in=test_in,
        config=config,
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    
    return {
        "task_id": task_id,
        "status": result.get("status", "FAIL"),
        "failure_kind": result.get("failure_reason", {}).get("kind", ""),
        "elapsed_ms": elapsed_ms,
        "program_len": len(result.get("program", [])) if isinstance(result.get("program"), list) else 0,
    }


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    total_tasks: int
    solved: int
    accuracy_pct: float
    
    failure_breakdown: Dict[str, int]
    avg_time_ms: float
    total_time_s: float
    
    # Config
    max_depth: int
    max_programs: int
    workers: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "solved": self.solved,
            "accuracy_pct": self.accuracy_pct,
            "failure_breakdown": self.failure_breakdown,
            "avg_time_ms": self.avg_time_ms,
            "total_time_s": self.total_time_s,
            "config": {
                "max_depth": self.max_depth,
                "max_programs": self.max_programs,
                "workers": self.workers,
            },
        }
    
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "BENCHMARK RESULT",
            "=" * 50,
            f"Tasks: {self.total_tasks}",
            f"Solved: {self.solved} ({self.accuracy_pct:.1f}%)",
            "",
            "Failure Breakdown:",
        ]
        for reason, count in sorted(self.failure_breakdown.items(), key=lambda x: -x[1]):
            lines.append(f"  {reason}: {count}")
        lines.extend([
            "",
            f"Avg time per task: {self.avg_time_ms:.0f}ms",
            f"Total time: {self.total_time_s:.1f}s",
            "",
            f"Config: depth={self.max_depth}, programs={self.max_programs}, workers={self.workers}",
            "=" * 50,
        ])
        return "\n".join(lines)


def run_benchmark(
    dataset_path: str,
    num_tasks: int = 100,
    max_depth: int = 4,
    max_programs: int = 4000,
    workers: int = 16,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run parallel benchmark on ARC tasks."""
    
    from atos_core.arc_evaluation_harness_v148 import load_arc_tasks
    
    # Load tasks
    tasks = load_arc_tasks(dataset_path, include_training=True, include_evaluation=False)
    tasks = tasks[:num_tasks]
    
    if verbose:
        print(f"Loaded {len(tasks)} tasks")
        print(f"Config: depth={max_depth}, programs={max_programs}, workers={workers}")
        print()
    
    # Prepare arguments
    work_items = []
    for task in tasks:
        train_pairs = [
            (tuple(tuple(c for c in row) for row in inp),
             tuple(tuple(c for c in row) for row in out))
            for inp, out in task.train_pairs
        ]
        test_in = tuple(tuple(c for c in row) for row in task.test_pairs[0][0])
        
        work_items.append((task.task_id, train_pairs, test_in, max_depth, max_programs))
    
    # Run in parallel
    t0 = time.monotonic()
    
    with multiprocessing.Pool(workers) as pool:
        results = pool.map(_solve_task_worker, work_items)
    
    total_time = time.monotonic() - t0
    
    # Aggregate results
    solved = sum(1 for r in results if r["status"] == "SOLVED")
    
    failure_breakdown: Dict[str, int] = {}
    total_elapsed = 0
    
    for r in results:
        total_elapsed += r["elapsed_ms"]
        if r["status"] != "SOLVED":
            kind = r["failure_kind"] or "UNKNOWN"
            failure_breakdown[kind] = failure_breakdown.get(kind, 0) + 1
        
        if verbose and r["status"] == "SOLVED":
            print(f"  ✓ {r['task_id']} (depth={r['program_len']}, {r['elapsed_ms']}ms)")
    
    return BenchmarkResult(
        total_tasks=len(tasks),
        solved=solved,
        accuracy_pct=(solved / len(tasks) * 100) if tasks else 0.0,
        failure_breakdown=failure_breakdown,
        avg_time_ms=total_elapsed / len(tasks) if tasks else 0.0,
        total_time_s=total_time,
        max_depth=max_depth,
        max_programs=max_programs,
        workers=workers,
    )


def main():
    parser = argparse.ArgumentParser(description="Quick ARC solver benchmark")
    parser.add_argument("--dataset", default="/workspaces/act/ARC-AGI/data", help="Path to ARC dataset")
    parser.add_argument("--tasks", type=int, default=100, help="Number of tasks to test")
    parser.add_argument("--depth", type=int, default=4, help="Max search depth")
    parser.add_argument("--programs", type=int, default=4000, help="Max programs per task")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    result = run_benchmark(
        dataset_path=args.dataset,
        num_tasks=args.tasks,
        max_depth=args.depth,
        max_programs=args.programs,
        workers=args.workers,
    )
    
    print(result)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
