"""
arc_evaluation_harness_v148.py - ARC-AGI Evaluation Harness

Comprehensive evaluation framework for measuring solver performance on
ARC-AGI-1 and ARC-AGI-2 datasets.

Features:
- Parallel execution across all available cores (16 threads)
- Detailed metrics: accuracy, programs tested, depth, cost
- Breakdown by task difficulty/type
- JSON + Markdown report generation
- Integration-ready for CI pipelines
- Cross-task concept reuse tracking

Schema version: 148
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .grid_v124 import GridV124, grid_equal_v124

ARC_EVALUATION_HARNESS_SCHEMA_VERSION_V148 = 148

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ARCTask:
    """A single ARC task with training and test examples."""
    task_id: str
    train_pairs: List[Tuple[GridV124, GridV124]]
    test_pairs: List[Tuple[GridV124, GridV124]]  # input, expected_output
    source_file: str
    dataset: str  # "training" or "evaluation"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "n_train": len(self.train_pairs),
            "n_test": len(self.test_pairs),
            "dataset": str(self.dataset),
        }


@dataclass
class TaskResult:
    """Result of solving a single task."""
    task_id: str
    solved: bool
    ambiguous: bool
    timeout: bool
    error: Optional[str]
    
    # Metrics
    programs_tested: int
    solution_depth: int
    solution_cost_bits: int
    elapsed_ms: int
    
    # Solution details
    predicted_outputs: List[GridV124]
    expected_outputs: List[GridV124]
    match_per_test: List[bool]
    
    # Concept tracking
    concepts_used: List[str]
    concept_reuse_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "solved": bool(self.solved),
            "ambiguous": bool(self.ambiguous),
            "timeout": bool(self.timeout),
            "error": str(self.error) if self.error else None,
            "programs_tested": int(self.programs_tested),
            "solution_depth": int(self.solution_depth),
            "solution_cost_bits": int(self.solution_cost_bits),
            "elapsed_ms": int(self.elapsed_ms),
            "match_per_test": [bool(m) for m in self.match_per_test],
            "concepts_used": list(self.concepts_used),
            "concept_reuse_count": int(self.concept_reuse_count),
        }


@dataclass
class EvaluationMetrics:
    """Aggregated metrics from evaluation run."""
    total_tasks: int
    solved_tasks: int
    ambiguous_tasks: int
    timeout_tasks: int
    error_tasks: int
    
    # Rates
    accuracy_pct: float
    ambiguity_rate_pct: float
    timeout_rate_pct: float
    error_rate_pct: float
    
    # Program metrics
    avg_programs_tested: float
    avg_solution_depth: float
    avg_solution_cost_bits: float
    avg_elapsed_ms: float
    
    # Concept metrics
    total_concepts_used: int
    unique_concepts_used: int
    avg_reuse_per_concept: float
    tasks_using_concepts: int
    
    # Per-dataset breakdown
    training_accuracy_pct: float
    evaluation_accuracy_pct: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": int(self.total_tasks),
            "solved_tasks": int(self.solved_tasks),
            "ambiguous_tasks": int(self.ambiguous_tasks),
            "timeout_tasks": int(self.timeout_tasks),
            "error_tasks": int(self.error_tasks),
            "accuracy_pct": float(self.accuracy_pct),
            "ambiguity_rate_pct": float(self.ambiguity_rate_pct),
            "timeout_rate_pct": float(self.timeout_rate_pct),
            "error_rate_pct": float(self.error_rate_pct),
            "avg_programs_tested": float(self.avg_programs_tested),
            "avg_solution_depth": float(self.avg_solution_depth),
            "avg_solution_cost_bits": float(self.avg_solution_cost_bits),
            "avg_elapsed_ms": float(self.avg_elapsed_ms),
            "total_concepts_used": int(self.total_concepts_used),
            "unique_concepts_used": int(self.unique_concepts_used),
            "avg_reuse_per_concept": float(self.avg_reuse_per_concept),
            "tasks_using_concepts": int(self.tasks_using_concepts),
            "training_accuracy_pct": float(self.training_accuracy_pct),
            "evaluation_accuracy_pct": float(self.evaluation_accuracy_pct),
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    schema_version: int = ARC_EVALUATION_HARNESS_SCHEMA_VERSION_V148
    timestamp: str = ""
    solver_version: str = ""
    git_commit: str = ""
    
    # Configuration
    dataset_path: str = ""
    num_workers: int = 16
    max_programs_per_task: int = 100000
    max_depth: int = 6
    timeout_per_task_ms: int = 60000
    
    # Results
    metrics: Optional[EvaluationMetrics] = None
    task_results: List[TaskResult] = field(default_factory=list)
    unsolved_task_ids: List[str] = field(default_factory=list)
    ambiguous_task_ids: List[str] = field(default_factory=list)
    
    # Timing
    total_elapsed_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "timestamp": str(self.timestamp),
            "solver_version": str(self.solver_version),
            "git_commit": str(self.git_commit),
            "config": {
                "dataset_path": str(self.dataset_path),
                "num_workers": int(self.num_workers),
                "max_programs_per_task": int(self.max_programs_per_task),
                "max_depth": int(self.max_depth),
                "timeout_per_task_ms": int(self.timeout_per_task_ms),
            },
            "metrics": self.metrics.to_dict() if self.metrics else {},
            "task_results": [r.to_dict() for r in self.task_results],
            "unsolved_task_ids": list(self.unsolved_task_ids),
            "ambiguous_task_ids": list(self.ambiguous_task_ids),
            "total_elapsed_ms": int(self.total_elapsed_ms),
        }
    
    def to_markdown(self) -> str:
        """Generate markdown summary report."""
        lines = [
            f"# ARC-AGI Evaluation Report V{self.schema_version}",
            "",
            f"**Timestamp:** {self.timestamp}",
            f"**Solver Version:** {self.solver_version}",
            f"**Git Commit:** {self.git_commit[:8] if self.git_commit else 'unknown'}",
            "",
            "## Configuration",
            "",
            f"- **Workers:** {self.num_workers}",
            f"- **Max Programs per Task:** {self.max_programs_per_task:,}",
            f"- **Max Depth:** {self.max_depth}",
            f"- **Timeout per Task:** {self.timeout_per_task_ms:,}ms",
            "",
            "## Results Summary",
            "",
        ]
        
        if self.metrics:
            m = self.metrics
            lines.extend([
                "| Metric | Value |",
                "|--------|-------|",
                f"| Total Tasks | {m.total_tasks} |",
                f"| **Solved Tasks** | **{m.solved_tasks}** |",
                f"| **Accuracy** | **{m.accuracy_pct:.2f}%** |",
                f"| Ambiguous | {m.ambiguous_tasks} ({m.ambiguity_rate_pct:.1f}%) |",
                f"| Timeout | {m.timeout_tasks} ({m.timeout_rate_pct:.1f}%) |",
                f"| Error | {m.error_tasks} ({m.error_rate_pct:.1f}%) |",
                "",
                "### Per-Dataset Accuracy",
                "",
                f"- **Training:** {m.training_accuracy_pct:.2f}%",
                f"- **Evaluation:** {m.evaluation_accuracy_pct:.2f}%",
                "",
                "### Program Metrics (solved tasks only)",
                "",
                f"- Avg Programs Tested: {m.avg_programs_tested:,.1f}",
                f"- Avg Solution Depth: {m.avg_solution_depth:.2f}",
                f"- Avg Solution Cost: {m.avg_solution_cost_bits:.1f} bits",
                f"- Avg Time per Task: {m.avg_elapsed_ms:,.0f}ms",
                "",
                "### Concept Usage",
                "",
                f"- Tasks Using Concepts: {m.tasks_using_concepts}/{m.total_tasks}",
                f"- Unique Concepts: {m.unique_concepts_used}",
                f"- Total Concept Calls: {m.total_concepts_used}",
                f"- Avg Reuse per Concept: {m.avg_reuse_per_concept:.2f}",
                "",
            ])
        
        lines.extend([
            f"**Total Elapsed:** {self.total_elapsed_ms / 1000:.1f}s",
            "",
            "## Unsolved Tasks",
            "",
            f"Total: {len(self.unsolved_task_ids)}",
            "",
        ])
        
        if self.unsolved_task_ids[:20]:
            lines.append("```")
            for tid in self.unsolved_task_ids[:20]:
                lines.append(tid)
            if len(self.unsolved_task_ids) > 20:
                lines.append(f"... and {len(self.unsolved_task_ids) - 20} more")
            lines.append("```")
        
        lines.extend([
            "",
            "---",
            "",
            "## Critical Question",
            "",
            "**Would the system survive without concepts?**",
            "",
        ])
        
        if self.metrics and self.metrics.tasks_using_concepts == 0:
            lines.append("⚠️ **YES** - System doesn't use concepts. FAILURE MODE.")
        elif self.metrics and self.metrics.tasks_using_concepts < self.metrics.total_tasks * 0.5:
            lines.append(f"⚠️ **PARTIAL** - Only {self.metrics.tasks_using_concepts}/{self.metrics.total_tasks} tasks use concepts.")
        else:
            lines.append("✅ **NO** - Concepts are required for most solutions.")
        
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────


def load_arc_tasks(
    dataset_path: str,
    include_training: bool = True,
    include_evaluation: bool = True,
) -> List[ARCTask]:
    """
    Load all ARC tasks from the dataset directory.
    
    Expected structure:
    dataset_path/
        training/
            *.json
        evaluation/
            *.json
    """
    tasks: List[ARCTask] = []
    base_path = Path(dataset_path)
    
    subdirs = []
    if include_training:
        subdirs.append(("training", base_path / "training"))
    if include_evaluation:
        subdirs.append(("evaluation", base_path / "evaluation"))
    
    for dataset_name, subdir in subdirs:
        if not subdir.exists():
            continue
        
        for json_file in sorted(subdir.glob("*.json")):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                
                task_id = json_file.stem
                
                # Parse training pairs
                train_pairs: List[Tuple[GridV124, GridV124]] = []
                for pair in data.get("train", []):
                    inp = [[int(c) for c in row] for row in pair["input"]]
                    out = [[int(c) for c in row] for row in pair["output"]]
                    train_pairs.append((inp, out))
                
                # Parse test pairs
                test_pairs: List[Tuple[GridV124, GridV124]] = []
                for pair in data.get("test", []):
                    inp = [[int(c) for c in row] for row in pair["input"]]
                    out = [[int(c) for c in row] for row in pair["output"]]
                    test_pairs.append((inp, out))
                
                tasks.append(ARCTask(
                    task_id=task_id,
                    train_pairs=train_pairs,
                    test_pairs=test_pairs,
                    source_file=str(json_file),
                    dataset=dataset_name,
                ))
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
    
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Solver wrapper
# ─────────────────────────────────────────────────────────────────────────────


def _solve_single_task(
    task: ARCTask,
    max_programs: int,
    max_depth: int,
    timeout_ms: int,
    concept_library: Optional[Dict[str, Any]] = None,
) -> TaskResult:
    """
    Solve a single ARC task using the parallel solver.
    
    Returns TaskResult with all metrics.
    """
    start_time = time.monotonic()
    
    try:
        # Import solver here to avoid circular imports
        from .arc_parallel_solver_v143 import (
            ParallelSolverConfigV143,
            solve_arc_task_parallel_v143,
        )
        
        config = ParallelSolverConfigV143(
            num_workers=4,  # Use 4 workers per task (16 total / 4 parallel tasks)
            max_programs_per_worker=max_programs // 4,
            max_depth=max_depth,
            enable_reachability_pruning=True,
        )
        
        # Prepare inputs
        train_pairs = task.train_pairs
        
        predicted_outputs: List[GridV124] = []
        expected_outputs: List[GridV124] = []
        match_per_test: List[bool] = []
        total_programs = 0
        best_depth = 0
        best_cost = 0
        concepts_used: List[str] = []
        
        # Solve for each test case
        all_solved = True
        for test_in, test_expected in task.test_pairs:
            result = solve_arc_task_parallel_v143(
                train_pairs=train_pairs,
                test_in=test_in,
                config=config,
            )
            
            expected_outputs.append(test_expected)
            total_programs += result.total_programs_evaluated
            
            if result.status == "success" and result.test_output is not None:
                predicted_outputs.append(result.test_output)
                is_match = grid_equal_v124(result.test_output, test_expected)
                match_per_test.append(is_match)
                
                if result.winning_program:
                    best_depth = max(best_depth, len(result.winning_program.steps))
                    best_cost = max(best_cost, result.winning_cost_bits)
                
                # Track concepts from solution
                if result.winning_program:
                    for step in result.winning_program.steps:
                        if step.op_id.startswith("concept_"):
                            concepts_used.append(step.op_id)
                
                if not is_match:
                    all_solved = False
            else:
                predicted_outputs.append([])  # Empty grid for failed prediction
                match_per_test.append(False)
                all_solved = False
        
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        
        return TaskResult(
            task_id=task.task_id,
            solved=all_solved,
            ambiguous=False,  # TODO: detect ambiguity from solver
            timeout=elapsed_ms >= timeout_ms,
            error=None,
            programs_tested=total_programs,
            solution_depth=best_depth,
            solution_cost_bits=best_cost,
            elapsed_ms=elapsed_ms,
            predicted_outputs=predicted_outputs,
            expected_outputs=expected_outputs,
            match_per_test=match_per_test,
            concepts_used=concepts_used,
            concept_reuse_count=len(concepts_used),
        )
        
    except Exception as e:
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        return TaskResult(
            task_id=task.task_id,
            solved=False,
            ambiguous=False,
            timeout=False,
            error=str(e) + "\n" + traceback.format_exc(),
            programs_tested=0,
            solution_depth=0,
            solution_cost_bits=0,
            elapsed_ms=elapsed_ms,
            predicted_outputs=[],
            expected_outputs=[out for _, out in task.test_pairs],
            match_per_test=[False] * len(task.test_pairs),
            concepts_used=[],
            concept_reuse_count=0,
        )


def _solve_task_wrapper(args: Tuple) -> Dict[str, Any]:
    """Wrapper for parallel execution."""
    task_dict, max_programs, max_depth, timeout_ms = args
    
    # Reconstruct ARCTask from dict
    task = ARCTask(
        task_id=task_dict["task_id"],
        train_pairs=task_dict["train_pairs"],
        test_pairs=task_dict["test_pairs"],
        source_file=task_dict["source_file"],
        dataset=task_dict["dataset"],
    )
    
    result = _solve_single_task(task, max_programs, max_depth, timeout_ms)
    return result.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────


def compute_metrics(results: List[TaskResult]) -> EvaluationMetrics:
    """Compute aggregated metrics from task results."""
    total = len(results)
    if total == 0:
        return EvaluationMetrics(
            total_tasks=0, solved_tasks=0, ambiguous_tasks=0, timeout_tasks=0, error_tasks=0,
            accuracy_pct=0.0, ambiguity_rate_pct=0.0, timeout_rate_pct=0.0, error_rate_pct=0.0,
            avg_programs_tested=0.0, avg_solution_depth=0.0, avg_solution_cost_bits=0.0, avg_elapsed_ms=0.0,
            total_concepts_used=0, unique_concepts_used=0, avg_reuse_per_concept=0.0, tasks_using_concepts=0,
            training_accuracy_pct=0.0, evaluation_accuracy_pct=0.0,
        )
    
    solved = sum(1 for r in results if r.solved)
    ambiguous = sum(1 for r in results if r.ambiguous)
    timeout = sum(1 for r in results if r.timeout)
    errors = sum(1 for r in results if r.error is not None)
    
    # Program metrics (only from solved tasks)
    solved_results = [r for r in results if r.solved]
    if solved_results:
        avg_programs = sum(r.programs_tested for r in solved_results) / len(solved_results)
        avg_depth = sum(r.solution_depth for r in solved_results) / len(solved_results)
        avg_cost = sum(r.solution_cost_bits for r in solved_results) / len(solved_results)
        avg_time = sum(r.elapsed_ms for r in solved_results) / len(solved_results)
    else:
        avg_programs = avg_depth = avg_cost = avg_time = 0.0
    
    # Concept metrics
    all_concepts: List[str] = []
    tasks_with_concepts = 0
    for r in results:
        if r.concepts_used:
            all_concepts.extend(r.concepts_used)
            tasks_with_concepts += 1
    
    unique_concepts = set(all_concepts)
    total_concept_calls = len(all_concepts)
    avg_reuse = (total_concept_calls / len(unique_concepts)) if unique_concepts else 0.0
    
    # Per-dataset accuracy (placeholder - need dataset info in results)
    training_solved = sum(1 for r in results if r.solved and "train" in r.task_id.lower())
    training_total = sum(1 for r in results if "train" in r.task_id.lower())
    eval_solved = sum(1 for r in results if r.solved and "train" not in r.task_id.lower())
    eval_total = sum(1 for r in results if "train" not in r.task_id.lower())
    
    return EvaluationMetrics(
        total_tasks=total,
        solved_tasks=solved,
        ambiguous_tasks=ambiguous,
        timeout_tasks=timeout,
        error_tasks=errors,
        accuracy_pct=(solved / total * 100) if total > 0 else 0.0,
        ambiguity_rate_pct=(ambiguous / total * 100) if total > 0 else 0.0,
        timeout_rate_pct=(timeout / total * 100) if total > 0 else 0.0,
        error_rate_pct=(errors / total * 100) if total > 0 else 0.0,
        avg_programs_tested=avg_programs,
        avg_solution_depth=avg_depth,
        avg_solution_cost_bits=avg_cost,
        avg_elapsed_ms=avg_time,
        total_concepts_used=total_concept_calls,
        unique_concepts_used=len(unique_concepts),
        avg_reuse_per_concept=avg_reuse,
        tasks_using_concepts=tasks_with_concepts,
        training_accuracy_pct=(training_solved / training_total * 100) if training_total > 0 else 0.0,
        evaluation_accuracy_pct=(eval_solved / eval_total * 100) if eval_total > 0 else 0.0,
    )


def run_evaluation(
    dataset_path: str,
    *,
    num_workers: int = 16,
    max_programs_per_task: int = 100000,
    max_depth: int = 6,
    timeout_per_task_ms: int = 60000,
    include_training: bool = True,
    include_evaluation: bool = True,
    task_limit: Optional[int] = None,
    solver_version: str = "arc_parallel_solver_v143",
    output_json: Optional[str] = None,
    output_markdown: Optional[str] = None,
    verbose: bool = True,
) -> EvaluationReport:
    """
    Run full evaluation on ARC dataset.
    
    Args:
        dataset_path: Path to ARC dataset (containing training/ and evaluation/)
        num_workers: Number of parallel workers (default 16)
        max_programs_per_task: Max programs to test per task
        max_depth: Max program depth
        timeout_per_task_ms: Timeout per task in milliseconds
        include_training: Include training set
        include_evaluation: Include evaluation set
        task_limit: Limit number of tasks (for testing)
        solver_version: Solver identifier
        output_json: Path for JSON output
        output_markdown: Path for Markdown output
        verbose: Print progress
    
    Returns:
        EvaluationReport with all metrics and results
    """
    start_time = time.monotonic()
    
    # Get git commit
    git_commit = ""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except:
        pass
    
    # Load tasks
    if verbose:
        print(f"Loading ARC tasks from {dataset_path}...")
    
    tasks = load_arc_tasks(
        dataset_path,
        include_training=include_training,
        include_evaluation=include_evaluation,
    )
    
    if task_limit:
        tasks = tasks[:task_limit]
    
    if verbose:
        print(f"Loaded {len(tasks)} tasks")
        print(f"  Training: {sum(1 for t in tasks if t.dataset == 'training')}")
        print(f"  Evaluation: {sum(1 for t in tasks if t.dataset == 'evaluation')}")
    
    # Prepare task dicts for serialization
    task_dicts = [
        {
            "task_id": t.task_id,
            "train_pairs": t.train_pairs,
            "test_pairs": t.test_pairs,
            "source_file": t.source_file,
            "dataset": t.dataset,
        }
        for t in tasks
    ]
    
    # Run parallel evaluation
    if verbose:
        print(f"\nRunning evaluation with {num_workers} workers...")
        print(f"Max programs per task: {max_programs_per_task:,}")
        print(f"Max depth: {max_depth}")
        print(f"Timeout per task: {timeout_per_task_ms:,}ms\n")
    
    results: List[TaskResult] = []
    completed = 0
    
    # Process tasks in parallel
    # Use 4 concurrent tasks, each with 4 internal workers = 16 total cores
    concurrent_tasks = max(1, num_workers // 4)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=concurrent_tasks) as executor:
        args_list = [
            (task_dict, max_programs_per_task, max_depth, timeout_per_task_ms)
            for task_dict in task_dicts
        ]
        
        future_to_task = {
            executor.submit(_solve_task_wrapper, args): args[0]["task_id"]
            for args in args_list
        }
        
        for future in concurrent.futures.as_completed(future_to_task):
            task_id = future_to_task[future]
            completed += 1
            
            try:
                result_dict = future.result()
                result = TaskResult(
                    task_id=result_dict["task_id"],
                    solved=result_dict["solved"],
                    ambiguous=result_dict["ambiguous"],
                    timeout=result_dict["timeout"],
                    error=result_dict["error"],
                    programs_tested=result_dict["programs_tested"],
                    solution_depth=result_dict["solution_depth"],
                    solution_cost_bits=result_dict["solution_cost_bits"],
                    elapsed_ms=result_dict["elapsed_ms"],
                    predicted_outputs=[],  # Not serialized back
                    expected_outputs=[],
                    match_per_test=result_dict["match_per_test"],
                    concepts_used=result_dict["concepts_used"],
                    concept_reuse_count=result_dict["concept_reuse_count"],
                )
                results.append(result)
                
                if verbose:
                    status = "✓" if result.solved else ("⏱" if result.timeout else "✗")
                    print(f"[{completed}/{len(tasks)}] {task_id}: {status} ({result.elapsed_ms}ms, {result.programs_tested} programs)")
                    
            except Exception as e:
                if verbose:
                    print(f"[{completed}/{len(tasks)}] {task_id}: ERROR - {e}")
                results.append(TaskResult(
                    task_id=task_id,
                    solved=False,
                    ambiguous=False,
                    timeout=False,
                    error=str(e),
                    programs_tested=0,
                    solution_depth=0,
                    solution_cost_bits=0,
                    elapsed_ms=0,
                    predicted_outputs=[],
                    expected_outputs=[],
                    match_per_test=[],
                    concepts_used=[],
                    concept_reuse_count=0,
                ))
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    total_elapsed_ms = int((time.monotonic() - start_time) * 1000)
    
    # Build report
    report = EvaluationReport(
        timestamp=datetime.now().isoformat(),
        solver_version=solver_version,
        git_commit=git_commit,
        dataset_path=dataset_path,
        num_workers=num_workers,
        max_programs_per_task=max_programs_per_task,
        max_depth=max_depth,
        timeout_per_task_ms=timeout_per_task_ms,
        metrics=metrics,
        task_results=sorted(results, key=lambda r: r.task_id),
        unsolved_task_ids=[r.task_id for r in results if not r.solved],
        ambiguous_task_ids=[r.task_id for r in results if r.ambiguous],
        total_elapsed_ms=total_elapsed_ms,
    )
    
    # Save outputs
    if output_json:
        with open(output_json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        if verbose:
            print(f"\nJSON report saved to: {output_json}")
    
    if output_markdown:
        with open(output_markdown, "w") as f:
            f.write(report.to_markdown())
        if verbose:
            print(f"Markdown report saved to: {output_markdown}")
    
    if verbose:
        print("\n" + "=" * 60)
        print(report.to_markdown())
    
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Quick evaluation for CI
# ─────────────────────────────────────────────────────────────────────────────


def run_quick_evaluation(
    dataset_path: str,
    task_limit: int = 50,
    **kwargs,
) -> EvaluationReport:
    """
    Run quick evaluation on a subset of tasks for CI.
    """
    return run_evaluation(
        dataset_path,
        task_limit=task_limit,
        verbose=True,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Command-line interface for evaluation harness."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ARC-AGI Evaluation Harness V148",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--dataset", "-d",
        default="/workspaces/act/ARC-AGI/data",
        help="Path to ARC dataset",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=16,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--max-programs", "-p",
        type=int,
        default=100000,
        help="Max programs per task",
    )
    parser.add_argument(
        "--max-depth", "-D",
        type=int,
        default=6,
        help="Max program depth",
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=60000,
        help="Timeout per task (ms)",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of tasks",
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Only evaluate training set",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Only evaluate evaluation set",
    )
    parser.add_argument(
        "--output-json", "-o",
        default=None,
        help="Output JSON file",
    )
    parser.add_argument(
        "--output-markdown", "-m",
        default=None,
        help="Output Markdown file",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        dataset_path=args.dataset,
        num_workers=args.workers,
        max_programs_per_task=args.max_programs,
        max_depth=args.max_depth,
        timeout_per_task_ms=args.timeout,
        include_training=not args.evaluation_only,
        include_evaluation=not args.training_only,
        task_limit=args.limit,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
