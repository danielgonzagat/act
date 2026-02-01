"""
Parallel Deterministic Planner v142 for ACT ARC Solver.

This module implements a deterministic planning loop that distributes search budget
across multiple CPU cores while maintaining full auditability and WORM compliance.

Design principles:
- Deterministic: same seed + inputs → same outputs (parallel or sequential).
- Fail-closed: UNKNOWN on ambiguity/timeout rather than guessing.
- Auditable: separate logs per worker, merged into a single WORM-compliant ledger.
- Explicit budget: no hidden timeouts; budget is divided among workers.

Schema version: 142
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .act import canonical_json_dumps, sha256_hex

PARALLEL_PLANNER_SCHEMA_VERSION_V142 = 142


# -----------------------------------------------------------------------------
# Deterministic budget distribution
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchBudgetV142:
    """Explicit budget for deterministic search."""

    max_programs: int
    max_depth: int
    max_expansions_per_worker: int
    pruning_heuristic: str  # "reachability" | "cost_bound" | "none"
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(PARALLEL_PLANNER_SCHEMA_VERSION_V142),
            "max_programs": int(self.max_programs),
            "max_depth": int(self.max_depth),
            "max_expansions_per_worker": int(self.max_expansions_per_worker),
            "pruning_heuristic": str(self.pruning_heuristic),
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class WorkerBudgetV142:
    """Per-worker budget slice for parallel search."""

    worker_id: int
    total_workers: int
    programs_start: int
    programs_end: int
    depth_limit: int
    seed_offset: int
    branch_prefix: Tuple[int, ...]  # Deterministic branch assignment

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": int(self.worker_id),
            "total_workers": int(self.total_workers),
            "programs_start": int(self.programs_start),
            "programs_end": int(self.programs_end),
            "depth_limit": int(self.depth_limit),
            "seed_offset": int(self.seed_offset),
            "branch_prefix": list(self.branch_prefix),
        }


def distribute_budget_v142(
    *,
    budget: SearchBudgetV142,
    num_workers: int,
    operator_count: int,
) -> List[WorkerBudgetV142]:
    """
    Distribute search budget across workers deterministically.

    Strategy: divide the operator index space into contiguous slices.
    Each worker explores programs whose first operator index falls in its slice.
    This ensures no overlap and deterministic merge.
    """
    n = max(1, int(num_workers))
    op_count = max(1, int(operator_count))
    total_progs = int(budget.max_programs)
    progs_per_worker = max(1, total_progs // n)

    # Divide first-operator index space.
    ops_per_worker = max(1, (op_count + n - 1) // n)

    slices: List[WorkerBudgetV142] = []
    for i in range(n):
        op_start = i * ops_per_worker
        op_end = min(op_count, (i + 1) * ops_per_worker)
        if op_start >= op_count:
            # More workers than operators; this worker gets no work.
            op_start = op_count
            op_end = op_count

        slices.append(
            WorkerBudgetV142(
                worker_id=int(i),
                total_workers=int(n),
                programs_start=i * progs_per_worker,
                programs_end=min(total_progs, (i + 1) * progs_per_worker),
                depth_limit=int(budget.max_depth),
                seed_offset=int(budget.seed) + i * 1000003,
                branch_prefix=(int(op_start), int(op_end)),
            )
        )
    return slices


# -----------------------------------------------------------------------------
# Worker result and log structures
# -----------------------------------------------------------------------------


@dataclass
class WorkerLogEntryV142:
    """Single log entry for auditability."""

    timestamp_mono_ns: int
    worker_id: int
    event: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_mono_ns": int(self.timestamp_mono_ns),
            "worker_id": int(self.worker_id),
            "event": str(self.event),
            "data": dict(self.data),
        }


@dataclass
class WorkerResultV142:
    """Result from a single worker's search."""

    worker_id: int
    budget: WorkerBudgetV142
    programs_evaluated: int
    best_program: Optional[Dict[str, Any]]
    best_loss: Tuple[int, int]
    train_perfect_programs: List[Dict[str, Any]]
    log_entries: List[WorkerLogEntryV142]
    status: str  # "ok" | "budget_exhausted" | "error"
    error_message: str
    wall_time_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(PARALLEL_PLANNER_SCHEMA_VERSION_V142),
            "worker_id": int(self.worker_id),
            "budget": self.budget.to_dict(),
            "programs_evaluated": int(self.programs_evaluated),
            "best_program": self.best_program,
            "best_loss": list(self.best_loss),
            "train_perfect_programs": list(self.train_perfect_programs),
            "log_entries": [e.to_dict() for e in self.log_entries],
            "status": str(self.status),
            "error_message": str(self.error_message),
            "wall_time_ms": int(self.wall_time_ms),
        }


# -----------------------------------------------------------------------------
# Pruning heuristics (deterministic)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ReachabilityStateV142:
    """Abstract state for reachability pruning."""

    shape: Tuple[int, int]
    palette: Tuple[int, ...]
    stage: str  # "grid" | "objset" | "obj" | "bbox" | "patch"

    def key(self) -> str:
        return sha256_hex(
            canonical_json_dumps(
                {
                    "shape": list(self.shape),
                    "palette": list(self.palette),
                    "stage": str(self.stage),
                }
            ).encode("utf-8")
        )[:16]


def min_steps_to_modify_v142(stage: str) -> int:
    """Minimum steps needed to modify the grid from a given stage."""
    s = str(stage)
    if s == "grid":
        return 1
    if s == "patch":
        return 1  # commit_patch
    if s == "bbox":
        return 1
    if s == "obj":
        return 2
    if s == "objset":
        return 3
    return 99


def min_steps_shape_change_v142(stage: str, direction: str) -> int:
    """Minimum steps to change grid shape."""
    s = str(stage)
    d = str(direction)
    base = 0
    if s == "patch":
        base = 1
    elif s == "bbox":
        base = 2
    elif s == "obj":
        base = 3
    elif s == "objset":
        base = 4

    if d == "grow":
        return base + 1  # pad_to or repeat_grid
    if d == "shrink":
        return base + 1  # crop
    return base + 2


def can_reach_target_v142(
    *,
    current_shape: Tuple[int, int],
    current_palette: Tuple[int, ...],
    target_shape: Tuple[int, int],
    target_palette: Tuple[int, ...],
    stage: str,
    steps_left: int,
) -> bool:
    """Deterministic reachability check for pruning."""
    hc, wc = int(current_shape[0]), int(current_shape[1])
    ht, wt = int(target_shape[0]), int(target_shape[1])
    cp = set(int(x) for x in current_palette)
    tp = set(int(x) for x in target_palette)

    if int(steps_left) <= 0:
        return (hc, wc) == (ht, wt) and cp == tp

    # Shape reachability
    if (hc, wc) != (ht, wt):
        if ht > hc or wt > wc:
            if int(steps_left) < int(min_steps_shape_change_v142(stage, "grow")):
                return False
        elif ht < hc or wt < wc:
            if int(steps_left) < int(min_steps_shape_change_v142(stage, "shrink")):
                return False

    # Palette reachability
    if cp != tp:
        if int(steps_left) < int(min_steps_to_modify_v142(stage)):
            return False

    return True


# -----------------------------------------------------------------------------
# Parallel search coordinator
# -----------------------------------------------------------------------------


@dataclass
class ParallelSearchConfigV142:
    """Configuration for parallel deterministic search."""

    num_workers: int = 16
    max_programs: int = 2000
    max_depth: int = 4
    max_expansions_per_worker: int = 500
    pruning_heuristic: str = "reachability"
    seed: int = 0
    # Fail-closed: return UNKNOWN if multiple distinct train-perfect solutions exist.
    detect_ambiguity: bool = True
    ambiguity_probe_limit: int = 32
    # WORM compliance
    enable_worm_logging: bool = True
    log_every_n_programs: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(PARALLEL_PLANNER_SCHEMA_VERSION_V142),
            "num_workers": int(self.num_workers),
            "max_programs": int(self.max_programs),
            "max_depth": int(self.max_depth),
            "max_expansions_per_worker": int(self.max_expansions_per_worker),
            "pruning_heuristic": str(self.pruning_heuristic),
            "seed": int(self.seed),
            "detect_ambiguity": bool(self.detect_ambiguity),
            "ambiguity_probe_limit": int(self.ambiguity_probe_limit),
            "enable_worm_logging": bool(self.enable_worm_logging),
            "log_every_n_programs": int(self.log_every_n_programs),
        }


@dataclass
class ParallelSearchResultV142:
    """Aggregated result from parallel search."""

    status: str  # "SOLVED" | "UNKNOWN" | "FAIL"
    failure_reason: Optional[Dict[str, Any]]
    best_program: Optional[Dict[str, Any]]
    predicted_grid: Optional[List[List[int]]]
    predicted_grids: List[List[List[int]]]
    candidate_steps: List[List[Dict[str, Any]]]
    train_perfect_count: int
    ambiguity_detected: bool
    total_programs_evaluated: int
    worker_results: List[WorkerResultV142]
    merged_log_hash: str
    config: ParallelSearchConfigV142
    wall_time_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(PARALLEL_PLANNER_SCHEMA_VERSION_V142),
            "status": str(self.status),
            "failure_reason": self.failure_reason,
            "best_program": self.best_program,
            "predicted_grid": self.predicted_grid,
            "predicted_grids": list(self.predicted_grids),
            "candidate_steps": list(self.candidate_steps),
            "train_perfect_count": int(self.train_perfect_count),
            "ambiguity_detected": bool(self.ambiguity_detected),
            "total_programs_evaluated": int(self.total_programs_evaluated),
            "worker_results": [r.to_dict() for r in self.worker_results],
            "merged_log_hash": str(self.merged_log_hash),
            "config": self.config.to_dict(),
            "wall_time_ms": int(self.wall_time_ms),
        }


def _merge_worker_logs_v142(worker_results: Sequence[WorkerResultV142]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Merge worker logs into a single deterministic sequence.

    Merge order: by (timestamp_mono_ns, worker_id, event) for determinism.
    Returns (merged_entries, hash_of_merged).
    """
    all_entries: List[Tuple[int, int, str, Dict[str, Any]]] = []
    for wr in worker_results:
        for entry in wr.log_entries:
            all_entries.append(
                (
                    int(entry.timestamp_mono_ns),
                    int(entry.worker_id),
                    str(entry.event),
                    entry.to_dict(),
                )
            )

    # Deterministic sort
    all_entries.sort(key=lambda x: (int(x[0]), int(x[1]), str(x[2])))

    merged = [e[3] for e in all_entries]
    merged_hash = sha256_hex(canonical_json_dumps(merged).encode("utf-8"))
    return merged, merged_hash


def _detect_ambiguity_v142(
    *,
    train_perfect_programs: Sequence[Dict[str, Any]],
    limit: int,
) -> Tuple[bool, int]:
    """
    Detect if multiple train-perfect programs produce distinct test outputs.

    Returns (ambiguity_detected, distinct_test_output_count).
    """
    if len(train_perfect_programs) <= 1:
        return False, len(train_perfect_programs)

    # Collect distinct test output signatures
    test_sigs: Set[str] = set()
    for prog in train_perfect_programs[:limit]:
        test_grid = prog.get("test_grid")
        if test_grid is not None:
            sig = sha256_hex(canonical_json_dumps(test_grid).encode("utf-8"))[:32]
            test_sigs.add(sig)

    return len(test_sigs) > 1, len(test_sigs)


# -----------------------------------------------------------------------------
# Worker task function (runs in subprocess)
# -----------------------------------------------------------------------------


def _worker_search_task_v142(
    *,
    worker_budget: Dict[str, Any],
    task_data: Dict[str, Any],
    operators: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Worker search task executed in a subprocess.

    This function is serializable and deterministic.

    Note: Log timestamps are deterministic counters, not wall-clock time,
    to ensure reproducibility across runs.
    """
    import time as _time

    start_ms = int(_time.monotonic() * 1000)
    # Use a deterministic counter for log timestamps instead of monotonic_ns
    # This ensures log hashes are reproducible across runs
    log_counter = [0]  # Mutable for closure

    wb = WorkerBudgetV142(
        worker_id=int(worker_budget.get("worker_id", 0)),
        total_workers=int(worker_budget.get("total_workers", 1)),
        programs_start=int(worker_budget.get("programs_start", 0)),
        programs_end=int(worker_budget.get("programs_end", 0)),
        depth_limit=int(worker_budget.get("depth_limit", 4)),
        seed_offset=int(worker_budget.get("seed_offset", 0)),
        branch_prefix=tuple(int(x) for x in worker_budget.get("branch_prefix", (0, 0))),
    )

    log_entries: List[WorkerLogEntryV142] = []
    log_every_n = int(config.get("log_every_n_programs", 100))

    def _log(event: str, data: Dict[str, Any]) -> None:
        # Use deterministic counter instead of wall-clock for reproducibility
        log_counter[0] += 1
        log_entries.append(
            WorkerLogEntryV142(
                timestamp_mono_ns=int(log_counter[0] * 1000),  # Deterministic sequence
                worker_id=int(wb.worker_id),
                event=str(event),
                data=dict(data),
            )
        )

    _log("worker_start", {"budget": wb.to_dict()})

    # Extract task data
    train_pairs = task_data.get("train_pairs", [])
    test_input = task_data.get("test_input")
    target_shape = tuple(int(x) for x in task_data.get("target_shape", (0, 0)))
    target_palette = tuple(int(x) for x in task_data.get("target_palette", ()))

    # Filter operators to this worker's branch
    op_start, op_end = wb.branch_prefix
    if int(op_start) >= int(op_end) or int(op_start) >= len(operators):
        # No operators for this worker
        _log("worker_done", {"reason": "no_operators"})
        return WorkerResultV142(
            worker_id=int(wb.worker_id),
            budget=wb,
            programs_evaluated=0,
            best_program=None,
            best_loss=(999999, 999999),
            train_perfect_programs=[],
            log_entries=log_entries,
            status="ok",
            error_message="",
            wall_time_ms=int(_time.monotonic() * 1000 - start_ms),
        ).to_dict()

    my_ops = operators[int(op_start) : int(op_end)]
    _log("operators_assigned", {"count": len(my_ops), "range": [int(op_start), int(op_end)]})

    # Search state
    programs_evaluated = 0
    max_programs = int(wb.programs_end - wb.programs_start)
    best_program: Optional[Dict[str, Any]] = None
    best_loss = (999999, 999999)
    train_perfect: List[Dict[str, Any]] = []

    # Deterministic search: enumerate programs by depth
    # For now, implement a simple enumerative search
    # TODO: integrate with arc_solver_v141 search logic

    try:
        # Placeholder: would integrate with actual solver
        # This is where the real search would happen
        _log("search_start", {"max_programs": int(max_programs), "depth_limit": int(wb.depth_limit)})

        # The actual search would enumerate programs here
        # For the skeleton, we just log completion
        _log("search_done", {"programs_evaluated": int(programs_evaluated)})

    except Exception as e:
        _log("worker_error", {"error": str(e)})
        return WorkerResultV142(
            worker_id=int(wb.worker_id),
            budget=wb,
            programs_evaluated=int(programs_evaluated),
            best_program=best_program,
            best_loss=best_loss,
            train_perfect_programs=train_perfect,
            log_entries=log_entries,
            status="error",
            error_message=str(e),
            wall_time_ms=int(_time.monotonic() * 1000 - start_ms),
        ).to_dict()

    _log("worker_done", {"programs_evaluated": int(programs_evaluated), "train_perfect_count": len(train_perfect)})

    return WorkerResultV142(
        worker_id=int(wb.worker_id),
        budget=wb,
        programs_evaluated=int(programs_evaluated),
        best_program=best_program,
        best_loss=best_loss,
        train_perfect_programs=train_perfect,
        log_entries=log_entries,
        status="ok" if int(programs_evaluated) < int(max_programs) else "budget_exhausted",
        error_message="",
        wall_time_ms=int(_time.monotonic() * 1000 - start_ms),
    ).to_dict()


# -----------------------------------------------------------------------------
# Main parallel search coordinator
# -----------------------------------------------------------------------------


class ParallelPlannerV142:
    """
    Parallel deterministic planner for ARC tasks.

    Distributes search budget across multiple workers and merges results
    while maintaining determinism and auditability.
    """

    def __init__(self, config: Optional[ParallelSearchConfigV142] = None) -> None:
        self.config = config or ParallelSearchConfigV142()

    def search(
        self,
        *,
        train_pairs: Sequence[Tuple[Any, Any]],
        test_input: Any,
        operators: Sequence[Dict[str, Any]],
        target_shape: Optional[Tuple[int, int]] = None,
        target_palette: Optional[Tuple[int, ...]] = None,
    ) -> ParallelSearchResultV142:
        """
        Run parallel deterministic search.

        Args:
            train_pairs: List of (input_grid, output_grid) training examples.
            test_input: Test input grid.
            operators: List of available operator definitions.
            target_shape: Expected output shape (for pruning).
            target_palette: Expected output palette (for pruning).

        Returns:
            ParallelSearchResultV142 with aggregated results.
        """
        import time as _time

        start_ms = int(_time.monotonic() * 1000)

        # Build search budget
        budget = SearchBudgetV142(
            max_programs=int(self.config.max_programs),
            max_depth=int(self.config.max_depth),
            max_expansions_per_worker=int(self.config.max_expansions_per_worker),
            pruning_heuristic=str(self.config.pruning_heuristic),
            seed=int(self.config.seed),
        )

        # Distribute budget
        worker_budgets = distribute_budget_v142(
            budget=budget,
            num_workers=int(self.config.num_workers),
            operator_count=len(operators),
        )

        # Prepare task data (serializable)
        task_data = {
            "train_pairs": [
                {"input": list(list(r) for r in inp), "output": list(list(r) for r in out)}
                for inp, out in train_pairs
            ],
            "test_input": list(list(r) for r in test_input) if test_input else None,
            "target_shape": list(target_shape) if target_shape else [0, 0],
            "target_palette": list(target_palette) if target_palette else [],
        }

        # Prepare operators (serializable)
        ops_list = [dict(op) for op in operators]

        # Run workers
        worker_results: List[WorkerResultV142] = []

        # Use ProcessPoolExecutor for true parallelism
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(self.config.num_workers)) as executor:
            futures = []
            for wb in worker_budgets:
                future = executor.submit(
                    _worker_search_task_v142,
                    worker_budget=wb.to_dict(),
                    task_data=task_data,
                    operators=ops_list,
                    config=self.config.to_dict(),
                )
                futures.append(future)

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result_dict = future.result()
                    wr = WorkerResultV142(
                        worker_id=int(result_dict.get("worker_id", 0)),
                        budget=WorkerBudgetV142(
                            worker_id=int(result_dict.get("budget", {}).get("worker_id", 0)),
                            total_workers=int(result_dict.get("budget", {}).get("total_workers", 1)),
                            programs_start=int(result_dict.get("budget", {}).get("programs_start", 0)),
                            programs_end=int(result_dict.get("budget", {}).get("programs_end", 0)),
                            depth_limit=int(result_dict.get("budget", {}).get("depth_limit", 4)),
                            seed_offset=int(result_dict.get("budget", {}).get("seed_offset", 0)),
                            branch_prefix=tuple(
                                int(x) for x in result_dict.get("budget", {}).get("branch_prefix", (0, 0))
                            ),
                        ),
                        programs_evaluated=int(result_dict.get("programs_evaluated", 0)),
                        best_program=result_dict.get("best_program"),
                        best_loss=tuple(int(x) for x in result_dict.get("best_loss", (999999, 999999))),
                        train_perfect_programs=list(result_dict.get("train_perfect_programs", [])),
                        log_entries=[
                            WorkerLogEntryV142(
                                timestamp_mono_ns=int(e.get("timestamp_mono_ns", 0)),
                                worker_id=int(e.get("worker_id", 0)),
                                event=str(e.get("event", "")),
                                data=dict(e.get("data", {})),
                            )
                            for e in result_dict.get("log_entries", [])
                        ],
                        status=str(result_dict.get("status", "ok")),
                        error_message=str(result_dict.get("error_message", "")),
                        wall_time_ms=int(result_dict.get("wall_time_ms", 0)),
                    )
                    worker_results.append(wr)
                except Exception as e:
                    # Worker failed; create error result
                    worker_results.append(
                        WorkerResultV142(
                            worker_id=-1,
                            budget=WorkerBudgetV142(
                                worker_id=-1,
                                total_workers=int(self.config.num_workers),
                                programs_start=0,
                                programs_end=0,
                                depth_limit=int(self.config.max_depth),
                                seed_offset=0,
                                branch_prefix=(0, 0),
                            ),
                            programs_evaluated=0,
                            best_program=None,
                            best_loss=(999999, 999999),
                            train_perfect_programs=[],
                            log_entries=[],
                            status="error",
                            error_message=str(e),
                            wall_time_ms=0,
                        )
                    )

        # Sort worker results by worker_id for determinism
        worker_results.sort(key=lambda r: int(r.worker_id))

        # Merge logs
        merged_logs, merged_log_hash = _merge_worker_logs_v142(worker_results)

        # Aggregate results
        total_evaluated = sum(int(r.programs_evaluated) for r in worker_results)
        all_train_perfect: List[Dict[str, Any]] = []
        for r in worker_results:
            all_train_perfect.extend(r.train_perfect_programs)

        # Find best program across all workers
        best_program: Optional[Dict[str, Any]] = None
        best_loss = (999999, 999999)
        for r in worker_results:
            if r.best_program is not None and r.best_loss < best_loss:
                best_loss = r.best_loss
                best_program = r.best_program

        # Detect ambiguity
        ambiguity_detected, distinct_count = _detect_ambiguity_v142(
            train_perfect_programs=all_train_perfect,
            limit=int(self.config.ambiguity_probe_limit),
        )

        # Determine final status
        status = "FAIL"
        failure_reason: Optional[Dict[str, Any]] = None
        predicted_grid: Optional[List[List[int]]] = None
        predicted_grids: List[List[List[int]]] = []
        candidate_steps: List[List[Dict[str, Any]]] = []

        if all_train_perfect:
            if ambiguity_detected and self.config.detect_ambiguity:
                status = "UNKNOWN"
                failure_reason = {
                    "kind": "AMBIGUITY_DETECTED",
                    "distinct_test_outputs": int(distinct_count),
                    "train_perfect_count": len(all_train_perfect),
                }
            else:
                status = "SOLVED"
                # Extract predictions from train-perfect programs
                for prog in all_train_perfect[:2]:  # Keep top 2 candidates
                    tg = prog.get("test_grid")
                    if tg is not None:
                        predicted_grids.append(list(list(int(c) for c in r) for r in tg))
                    steps = prog.get("steps")
                    if isinstance(steps, list):
                        candidate_steps.append(list(steps))
                if predicted_grids:
                    predicted_grid = predicted_grids[0]
        else:
            status = "FAIL"
            failure_reason = {
                "kind": "SEARCH_BUDGET_EXCEEDED",
                "total_evaluated": int(total_evaluated),
                "max_programs": int(self.config.max_programs),
            }

        wall_time_ms = int(_time.monotonic() * 1000 - start_ms)

        return ParallelSearchResultV142(
            status=str(status),
            failure_reason=failure_reason,
            best_program=best_program,
            predicted_grid=predicted_grid,
            predicted_grids=predicted_grids,
            candidate_steps=candidate_steps,
            train_perfect_count=len(all_train_perfect),
            ambiguity_detected=bool(ambiguity_detected),
            total_programs_evaluated=int(total_evaluated),
            worker_results=worker_results,
            merged_log_hash=str(merged_log_hash),
            config=self.config,
            wall_time_ms=int(wall_time_ms),
        )


# -----------------------------------------------------------------------------
# Integration with existing solver
# -----------------------------------------------------------------------------


def parallel_solve_arc_task_v142(
    *,
    train_pairs: Sequence[Tuple[Any, Any]],
    test_inputs: Sequence[Any],
    operators: Sequence[Dict[str, Any]],
    config: Optional[ParallelSearchConfigV142] = None,
) -> Dict[str, Any]:
    """
    Solve an ARC task using parallel deterministic search.

    This is the main entry point for integration with the existing harness.
    """
    cfg = config or ParallelSearchConfigV142()
    planner = ParallelPlannerV142(config=cfg)

    results: List[Dict[str, Any]] = []
    for test_idx, test_input in enumerate(test_inputs):
        result = planner.search(
            train_pairs=train_pairs,
            test_input=test_input,
            operators=operators,
        )
        results.append(
            {
                "test_index": int(test_idx),
                "result": result.to_dict(),
            }
        )

    return {
        "schema_version": int(PARALLEL_PLANNER_SCHEMA_VERSION_V142),
        "kind": "parallel_arc_solve_result_v142",
        "config": cfg.to_dict(),
        "test_results": results,
    }


# -----------------------------------------------------------------------------
# Ledger integration for WORM compliance
# -----------------------------------------------------------------------------


def write_parallel_search_to_ledger_v142(
    *,
    result: ParallelSearchResultV142,
    task_id: str,
    ledger_path: str,
    prev_hash: str = "",
) -> str:
    """
    Write parallel search result to WORM-compliant ledger.

    Returns the hash of the new ledger entry.
    """
    entry = {
        "schema_version": int(PARALLEL_PLANNER_SCHEMA_VERSION_V142),
        "kind": "parallel_search_ledger_entry_v142",
        "task_id": str(task_id),
        "result": result.to_dict(),
        "prev_hash": str(prev_hash),
    }

    entry_hash = sha256_hex(canonical_json_dumps(entry).encode("utf-8"))
    entry["entry_hash"] = str(entry_hash)

    # Append to ledger file
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    return str(entry_hash)
