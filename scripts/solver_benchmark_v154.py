#!/usr/bin/env python3
"""
solver_benchmark_v154.py - Benchmark the arc_solver_v141 to diagnose baseline performance.

This script measures:
1. What percentage of tasks can be solved with different budgets
2. What are the main failure modes
3. How long each task takes

Schema version: 154
"""

import sys
import time
import json
from collections import Counter
from typing import Any, Dict, List, Tuple

# Add parent to path
sys.path.insert(0, '/workspaces/act')

from atos_core.arc_solver_v141 import solve_arc_task_v141, SolveConfigV141
from atos_core.arc_evaluation_harness_v148 import load_arc_tasks

def benchmark_solver(
    num_tasks: int = 100,
    max_depth: int = 4,
    max_programs: int = 2000,
    timeout_per_task: float = 5.0,
) -> Dict[str, Any]:
    """Run benchmark on solver."""
    
    tasks = load_arc_tasks('/workspaces/act/ARC-AGI/data', include_training=True, include_evaluation=False)
    tasks = tasks[:num_tasks]
    
    config = SolveConfigV141(
        max_depth=max_depth,
        max_programs=max_programs,
        abstraction_pressure=False,
    )
    
    results = {
        "solved": [],
        "failed": [],
        "failure_reasons": Counter(),
        "times": [],
    }
    
    for i, task in enumerate(tasks):
        # Convert to tuples
        train_pairs = [
            (tuple(tuple(c for c in row) for row in inp),
             tuple(tuple(c for c in row) for row in out))
            for inp, out in task.train_pairs
        ]
        test_in = tuple(tuple(c for c in row) for row in task.test_pairs[0][0])
        
        t0 = time.time()
        result = solve_arc_task_v141(
            train_pairs=train_pairs,
            test_in=test_in,
            config=config,
        )
        elapsed = time.time() - t0
        results["times"].append(elapsed)
        
        status = result.get("status", "FAIL")
        
        if status == "SOLVED":
            results["solved"].append(task.task_id)
            print(f"[{i+1}/{num_tasks}] {task.task_id}: SOLVED ({elapsed:.2f}s)")
        else:
            reason = result.get("failure_reason", {}).get("kind", "UNKNOWN")
            results["failed"].append(task.task_id)
            results["failure_reasons"][reason] += 1
            if i < 10 or (i+1) % 20 == 0:
                print(f"[{i+1}/{num_tasks}] {task.task_id}: FAILED ({reason}) ({elapsed:.2f}s)")
    
    # Summary
    total = len(tasks)
    solved = len(results["solved"])
    
    print("\n" + "=" * 60)
    print(f"SOLVER BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Config: depth={max_depth}, programs={max_programs}")
    print(f"Tasks: {num_tasks}")
    print(f"Solved: {solved}/{total} = {solved/total*100:.1f}%")
    print(f"Avg time: {sum(results['times'])/len(results['times']):.2f}s")
    print(f"\nFailure reasons:")
    for reason, count in results["failure_reasons"].most_common():
        print(f"  {reason}: {count} ({count/total*100:.1f}%)")
    
    return {
        "config": {"depth": max_depth, "programs": max_programs},
        "total": total,
        "solved": solved,
        "accuracy_pct": solved/total*100,
        "failure_reasons": dict(results["failure_reasons"]),
        "avg_time_s": sum(results["times"])/len(results["times"]),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--programs", type=int, default=2000)
    args = parser.parse_args()
    
    result = benchmark_solver(
        num_tasks=args.tasks,
        max_depth=args.depth,
        max_programs=args.programs,
    )
    
    # Save results
    with open(f"/workspaces/act/benchmark_d{args.depth}_p{args.programs}_t{args.tasks}.json", "w") as f:
        json.dump(result, f, indent=2)
