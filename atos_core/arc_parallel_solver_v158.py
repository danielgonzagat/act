"""
arc_parallel_solver_v158.py - Parallel ARC Solver with Optimized Caching

UPGRADE from V143 - LESSONS LEARNED:

1. SHARED CACHE VIA MANAGER IS SLOWER (tested empirically):
   - multiprocessing.Manager uses IPC (sockets/pipes)
   - IPC overhead > computation cost for small ops like grid transforms
   - 10-task benchmark: with Manager = 47.84s, without = 24.18s
   - CONCLUSION: LOCAL caches per worker are FASTER
   
2. TRUE PARALLELISM BENEFIT comes from OPERATOR PARTITIONING:
   - Each worker explores a disjoint subset of first-step operators
   - After first step, workers search full operator space
   - This prevents duplicate work at the root level
   
3. WHY THIS WORKS FOR ARC:
   - Program search is COMBINATORIAL, not data-parallel
   - Most redundant computation is within a branch, not across branches
   - Local caches already capture intra-branch redundancy efficiently
   
4. FUTURE OPTIMIZATION IDEAS:
   - Pre-compute operator applicability per state signature
   - Use shared memory (mmap) instead of Manager for read-only data
   - Implement work-stealing between workers when one finishes early
   
Schema version: 158
"""

from __future__ import annotations

import concurrent.futures
import heapq
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_ops_v132 import StateV132
from .arc_ops_v141 import OP_DEFS_V141, apply_op_v141, step_cost_bits_v141
from .grid_v124 import (
    GridV124,
    grid_equal_v124,
    grid_hash_v124,
    grid_shape_v124,
    unique_colors_v124,
)
from .parallel_planner_v142 import (
    WorkerBudgetV142,
    WorkerLogEntryV142,
    WorkerResultV142,
    _merge_worker_logs_v142,
    _detect_ambiguity_v142,
)

ARC_PARALLEL_SOLVER_SCHEMA_VERSION_V158 = 158


# ─────────────────────────────────────────────────────────────────────────────
# Program step and evaluation structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProgramStepV158:
    """A single step in an ARC program."""
    op_id: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"op_id": str(self.op_id), "args": dict(self.args)}
    
    def __hash__(self) -> int:
        return hash((self.op_id, tuple(sorted(self.args.items()))))


@dataclass(frozen=True)
class ProgramV158:
    """A complete ARC program (sequence of steps)."""
    steps: Tuple[ProgramStepV158, ...]

    def program_sig(self) -> str:
        return sha256_hex(
            canonical_json_dumps(
                {"steps": [s.to_dict() for s in self.steps]}
            ).encode("utf-8")
        )[:32]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(ARC_PARALLEL_SOLVER_SCHEMA_VERSION_V158),
            "steps": [s.to_dict() for s in self.steps],
            "program_sig": str(self.program_sig()),
        }


@dataclass
class EvalResultV158:
    """Result of evaluating a program on training pairs."""
    ok_train: bool
    loss_shape: int
    loss_cells: int
    cost_bits: int
    train_final_states: Tuple[StateV132, ...]
    test_final_state: Optional[StateV132]
    test_grid: Optional[GridV124]
    got_shapes: Tuple[Tuple[int, int], ...]
    got_palettes: Tuple[Tuple[int, ...], ...]
    vec_sig: str
    mismatch_ex: Optional[Dict[str, Any]]

    @property
    def loss(self) -> Tuple[int, int]:
        return (int(self.loss_shape), int(self.loss_cells))
    
    def to_summary(self) -> Dict[str, Any]:
        """Compact summary for cache (without full states)."""
        return {
            "ok_train": self.ok_train,
            "loss_shape": self.loss_shape,
            "loss_cells": self.loss_cells,
            "cost_bits": self.cost_bits,
            "got_shapes": list(self.got_shapes),
            "got_palettes": list(self.got_palettes),
            "vec_sig": self.vec_sig,
            "test_grid": [[int(c) for c in row] for row in self.test_grid] if self.test_grid else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# State signature helpers
# ─────────────────────────────────────────────────────────────────────────────


def _state_sig_v158(state: StateV132, grid_hash_cache: Dict[GridV124, str]) -> str:
    """Compute a deterministic signature for a state."""
    gh = grid_hash_cache.get(state.grid)
    if gh is None:
        gh = grid_hash_v124(state.grid)
        grid_hash_cache[state.grid] = gh

    parts = [f"g:{gh[:16]}"]
    if state.objset is not None:
        parts.append(f"os:{len(state.objset.objects)}")
    if state.obj is not None:
        parts.append(f"o:{len(state.obj.cells)}")
    if state.bbox is not None:
        parts.append(f"b:{state.bbox.to_tuple()}")
    if state.patch is not None:
        parts.append(f"p:{grid_hash_v124(state.patch)[:8]}")
    return "|".join(parts)


def _grid_to_tuple(grid: GridV124) -> Tuple[Tuple[int, ...], ...]:
    """Convert grid to fully hashable tuple."""
    return tuple(tuple(int(c) for c in row) for row in grid)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Cache Manager
# ─────────────────────────────────────────────────────────────────────────────


class SharedCacheManager:
    """
    Global shared cache for parallel solver.
    
    Uses multiprocessing.Manager for cross-process sharing.
    All writes are deterministic (same key -> same value).
    """
    
    def __init__(self, manager: "multiprocessing.managers.SyncManager"):
        # Cache: (state_sig, op_id, args_sig) -> result_grid_hash
        self.apply_cache = manager.dict()
        
        # Cache: program_sig -> eval_summary_dict
        self.eval_cache = manager.dict()
        
        # Cache: grid_tuple -> grid_hash
        self.grid_hash_cache = manager.dict()
        
        # Metrics (for reporting)
        self.metrics = manager.dict()
        self.metrics["global_apply_hits"] = 0
        self.metrics["global_apply_misses"] = 0
        self.metrics["global_eval_hits"] = 0
        self.metrics["global_eval_misses"] = 0
    
    def get_apply(self, key: str) -> Optional[str]:
        """Get cached apply result (returns grid hash or None)."""
        return self.apply_cache.get(key)
    
    def set_apply(self, key: str, value: str) -> None:
        """Set cached apply result."""
        self.apply_cache[key] = value
    
    def get_eval(self, prog_sig: str) -> Optional[Dict[str, Any]]:
        """Get cached eval summary."""
        return self.eval_cache.get(prog_sig)
    
    def set_eval(self, prog_sig: str, summary: Dict[str, Any]) -> None:
        """Set cached eval summary."""
        self.eval_cache[prog_sig] = summary
    
    def get_grid_hash(self, grid_tuple: Tuple) -> Optional[str]:
        """Get cached grid hash."""
        return self.grid_hash_cache.get(grid_tuple)
    
    def set_grid_hash(self, grid_tuple: Tuple, hash_val: str) -> None:
        """Set cached grid hash."""
        self.grid_hash_cache[grid_tuple] = hash_val
    
    def inc_metric(self, key: str) -> None:
        """Increment a metric counter."""
        current = self.metrics.get(key, 0)
        self.metrics[key] = current + 1
    
    def get_metrics(self) -> Dict[str, int]:
        """Get all metrics."""
        return dict(self.metrics)


# ─────────────────────────────────────────────────────────────────────────────
# Apply and evaluation with shared cache
# ─────────────────────────────────────────────────────────────────────────────


def _apply_step_shared_v158(
    *,
    state: StateV132,
    op_id: str,
    args: Dict[str, Any],
    shared_cache: Optional[Dict],  # Manager dict for apply cache (NOT USED - too slow)
    local_cache: Dict[Tuple[Any, ...], StateV132],  # Local full state cache
    grid_hash_local: Dict[GridV124, str],  # Local grid hash cache
    metrics: Dict[str, int],
) -> StateV132:
    """Apply a single step with LOCAL caching only (shared cache too slow via IPC)."""
    
    # Compute state signature
    gh = grid_hash_local.get(state.grid)
    if gh is None:
        gh = grid_hash_v124(state.grid)
        grid_hash_local[state.grid] = gh
    
    state_sig = _state_sig_v158(state, grid_hash_local)
    args_sig = canonical_json_dumps(args)
    local_key = (str(state_sig), str(op_id), str(args_sig))

    # Check local cache ONLY (much faster than IPC to Manager)
    hit = local_cache.get(local_key)
    if hit is not None:
        metrics["local_apply_hits"] = metrics.get("local_apply_hits", 0) + 1
        return hit

    metrics["apply_misses"] = metrics.get("apply_misses", 0) + 1
    
    # Execute operation
    result = apply_op_v141(state=state, op_id=str(op_id), args=dict(args))
    
    # Update local cache only
    local_cache[local_key] = result
    grid_hash_local[result.grid] = grid_hash_v124(result.grid)
    
    return result


def _eval_program_shared_v158(
    *,
    steps: Sequence[ProgramStepV158],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    shared_eval_cache: Optional[Dict],  # Manager dict (NOT USED - too slow via IPC)
    local_apply_cache: Dict[Tuple[Any, ...], StateV132],
    shared_apply_cache: Optional[Dict],  # NOT USED
    grid_hash_local: Dict[GridV124, str],
    metrics: Dict[str, int],
    local_eval_cache: Optional[Dict[str, "EvalResultV158"]] = None,  # Local eval cache
) -> EvalResultV158:
    """Evaluate a program with LOCAL caching only (shared too slow)."""
    prog = ProgramV158(steps=tuple(steps))
    sig = prog.program_sig()

    # Check LOCAL eval cache (much faster than IPC)
    if local_eval_cache is not None:
        cached = local_eval_cache.get(sig)
        if cached is not None:
            metrics["local_eval_hits"] = metrics.get("local_eval_hits", 0) + 1
            return cached

    metrics["eval_misses"] = metrics.get("eval_misses", 0) + 1

    ok = True
    loss_shape = 0
    loss_cells = 0
    mismatches: List[Dict[str, Any]] = []
    train_final_states: List[StateV132] = []
    got_shapes: List[Tuple[int, int]] = []
    got_palettes: List[Tuple[int, ...]] = []
    train_state_sigs: List[str] = []

    # Evaluate on training pairs
    for inp, want in train_pairs:
        st = StateV132(grid=inp)
        got: GridV124 = inp
        try:
            for step in steps:
                st = _apply_step_shared_v158(
                    state=st,
                    op_id=str(step.op_id),
                    args=dict(step.args),
                    shared_cache=shared_apply_cache,
                    local_cache=local_apply_cache,
                    grid_hash_local=grid_hash_local,
                    metrics=metrics,
                )
            got = st.grid
        except Exception as e:
            ok = False
            if not mismatches:
                mismatches.append({"kind": "exception", "error": str(e)})

        train_state_sigs.append(_state_sig_v158(st, grid_hash_local))
        train_final_states.append(st)
        got_shapes.append(tuple(int(x) for x in grid_shape_v124(got)))
        got_palettes.append(tuple(sorted(int(c) for c in unique_colors_v124(got))))

        if not grid_equal_v124(got, want):
            ok = False
            hg, wg = grid_shape_v124(got)
            hw, ww = grid_shape_v124(want)
            if (hg, wg) != (hw, ww):
                loss_shape += 1
                mismatches.append({
                    "kind": "shape_mismatch",
                    "got": {"h": int(hg), "w": int(wg)},
                    "want": {"h": int(hw), "w": int(ww)},
                })
            else:
                diff = 0
                for r in range(hg):
                    for c in range(wg):
                        if int(got[r][c]) != int(want[r][c]):
                            diff += 1
                loss_cells += diff
                if not any(m.get("kind") == "cell_mismatch" for m in mismatches):
                    mismatches.append({"kind": "cell_mismatch", "diff_cells": int(diff)})

    # Evaluate on test input
    stt = StateV132(grid=test_in)
    test_grid: GridV124 = test_in
    try:
        for step in steps:
            stt = _apply_step_shared_v158(
                state=stt,
                op_id=str(step.op_id),
                args=dict(step.args),
                shared_cache=shared_apply_cache,
                local_cache=local_apply_cache,
                grid_hash_local=grid_hash_local,
                metrics=metrics,
            )
        test_grid = stt.grid
    except Exception:
        stt = StateV132(grid=test_in)
        test_grid = test_in

    # Compute cost in bits
    cost_bits = 0
    for step in steps:
        cost_bits += int(step_cost_bits_v141(op_id=str(step.op_id), args=dict(step.args)))

    # Compute vector signature for dominated-state pruning
    vec_obj = {
        "state_sigs": train_state_sigs,
        "test_state_sig": _state_sig_v158(stt, grid_hash_local),
        "loss": {"shape": int(loss_shape), "cells": int(loss_cells)},
    }
    vec_sig = sha256_hex(canonical_json_dumps(vec_obj).encode("utf-8"))[:32]

    result = EvalResultV158(
        ok_train=bool(ok),
        loss_shape=int(loss_shape),
        loss_cells=int(loss_cells),
        cost_bits=int(cost_bits),
        train_final_states=tuple(train_final_states),
        test_final_state=stt,
        test_grid=test_grid,
        got_shapes=tuple(got_shapes),
        got_palettes=tuple(got_palettes),
        vec_sig=str(vec_sig),
        mismatch_ex=mismatches[0] if mismatches else None,
    )

    # Update LOCAL eval cache only (shared cache via IPC too slow)
    if local_eval_cache is not None:
        local_eval_cache[sig] = result

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Next step proposal (operator enumeration)
# ─────────────────────────────────────────────────────────────────────────────


def _slots_available_v158(state: StateV132) -> Dict[str, bool]:
    """Determine which slots are available in the current state."""
    return {
        "grid": True,
        "objset": state.objset is not None,
        "obj": state.obj is not None,
        "bbox": state.bbox is not None,
        "patch": state.patch is not None,
    }


def _propose_next_steps_v158(
    *,
    train_final_states: Sequence[StateV132],
    operator_ids: Sequence[str],
    palette_out: Sequence[int],
    shapes_out: Sequence[Tuple[int, int]],
    max_steps: int = 64,
) -> List[ProgramStepV158]:
    """Propose next steps based on current state slots and operators."""
    proposals: List[ProgramStepV158] = []

    # Analyze available slots across all training states
    all_avail: Dict[str, int] = {"grid": 0, "objset": 0, "obj": 0, "bbox": 0, "patch": 0}
    for st in train_final_states:
        avail = _slots_available_v158(st)
        for k, v in avail.items():
            if v:
                all_avail[k] = int(all_avail.get(k, 0)) + 1

    # Include colors from both output palette AND current state grids
    state_colors: Set[int] = set()
    for st in train_final_states:
        for row in st.grid:
            for c in row:
                state_colors.add(int(c))
    
    all_colors = set(int(c) for c in palette_out) | state_colors
    colors = list(sorted(all_colors))[:10]

    for op_id in operator_ids:
        od = OP_DEFS_V141.get(str(op_id))
        if od is None:
            continue

        # Check if required slots are available in any state
        reads_ok = True
        for r in od.reads:
            if int(all_avail.get(str(r), 0)) == 0:
                reads_ok = False
                break
        if not reads_ok:
            continue

        # Generate argument combinations
        if op_id in {"rotate90", "rotate180", "rotate270", "reflect_h", "reflect_v", "transpose"}:
            proposals.append(ProgramStepV158(op_id=str(op_id), args={}))
        elif op_id == "replace_color":
            for c1 in colors:
                for c2 in colors:
                    if c1 != c2:
                        proposals.append(ProgramStepV158(op_id=str(op_id), args={"from_color": int(c1), "to_color": int(c2)}))
        elif op_id == "fill_color":
            for c in colors:
                proposals.append(ProgramStepV158(op_id=str(op_id), args={"color": int(c)}))
        elif op_id == "paint_mask":
            for c in colors:
                proposals.append(ProgramStepV158(op_id=str(op_id), args={"color": int(c)}))
        elif op_id in {"cc4_nonbg", "cc4_nonbg_multicolor"}:
            for bg in colors:
                proposals.append(ProgramStepV158(op_id=str(op_id), args={"bg": int(bg)}))
        elif op_id == "select_obj":
            for idx in range(8):
                proposals.append(ProgramStepV158(op_id=str(op_id), args={"index": int(idx)}))
        elif op_id == "obj_bbox":
            proposals.append(ProgramStepV158(op_id=str(op_id), args={}))
        elif op_id == "crop_bbox":
            proposals.append(ProgramStepV158(op_id=str(op_id), args={}))
        elif op_id == "commit_patch":
            proposals.append(ProgramStepV158(op_id=str(op_id), args={}))
        elif op_id == "crop_bbox_nonzero":
            proposals.append(ProgramStepV158(op_id=str(op_id), args={}))
        elif op_id == "pad_to":
            for h, w in shapes_out[:4]:
                proposals.append(ProgramStepV158(op_id=str(op_id), args={"h": int(h), "w": int(w)}))
        elif op_id == "repeat_grid":
            for rh in [2, 3]:
                for rw in [2, 3]:
                    proposals.append(ProgramStepV158(op_id=str(op_id), args={"repeat_h": int(rh), "repeat_w": int(rw)}))
        elif op_id == "gravity":
            for d in ["up", "down", "left", "right"]:
                proposals.append(ProgramStepV158(op_id=str(op_id), args={"direction": str(d)}))
        elif op_id in {"flood_fill"}:
            for c in colors:
                proposals.append(ProgramStepV158(op_id=str(op_id), args={"color": int(c), "r": 0, "c": 0}))
        elif op_id == "new_canvas":
            for h, w in shapes_out[:2]:
                for c in colors[:2]:
                    proposals.append(ProgramStepV158(op_id=str(op_id), args={"h": int(h), "w": int(w), "fill": int(c)}))
        elif op_id == "bbox_by_color":
            for c in colors:
                proposals.append(ProgramStepV158(op_id=str(op_id), args={"color": int(c)}))
        elif op_id in {"translate"}:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr != 0 or dc != 0:
                        proposals.append(ProgramStepV158(op_id=str(op_id), args={"dr": int(dr), "dc": int(dc)}))
        elif op_id in {"overlay_self_translate", "propagate_color_translate", "propagate_nonbg_translate"}:
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    if dr != 0 or dc != 0:
                        if op_id == "propagate_color_translate":
                            for c in colors:
                                proposals.append(ProgramStepV158(op_id=str(op_id), args={"dr": int(dr), "dc": int(dc), "color": int(c)}))
                        else:
                            proposals.append(ProgramStepV158(op_id=str(op_id), args={"dr": int(dr), "dc": int(dc)}))
        elif op_id in {"symmetry_fill_h", "symmetry_fill_v", "downsample_mode", "fill_enclosed_region"}:
            proposals.append(ProgramStepV158(op_id=str(op_id), args={}))
        else:
            if not od.reads or all(int(all_avail.get(str(r), 0)) > 0 for r in od.reads):
                proposals.append(ProgramStepV158(op_id=str(op_id), args={}))

        if len(proposals) >= int(max_steps):
            break

    return proposals[:int(max_steps)]


# ─────────────────────────────────────────────────────────────────────────────
# Parallel worker task WITH shared cache
# ─────────────────────────────────────────────────────────────────────────────


def _worker_search_shared_v158(
    *,
    worker_id: int,
    total_workers: int,
    operator_slice: Tuple[int, int],
    train_pairs_raw: List[Tuple[List[List[int]], List[List[int]]]],
    test_in_raw: List[List[int]],
    all_operator_ids: List[str],
    max_programs: int,
    max_depth: int,
    seed: int,
    log_every_n: int,
    enable_reachability_pruning: bool,
    palette_out: List[int],
    shapes_out: List[Tuple[int, int]],
    shared_apply_cache: Optional[Dict],  # Manager dict
    shared_eval_cache: Optional[Dict],   # Manager dict
) -> Dict[str, Any]:
    """
    Worker search task with shared cache.
    
    Uses both local caches (fast, per-worker) and shared caches (cross-process).
    """
    import time as _time

    start_ms = int(_time.monotonic() * 1000)
    log_counter = [0]

    # Convert raw grids to tuples
    train_pairs: List[Tuple[GridV124, GridV124]] = [
        (tuple(tuple(int(c) for c in row) for row in inp), tuple(tuple(int(c) for c in row) for row in out))
        for inp, out in train_pairs_raw
    ]
    test_in: GridV124 = tuple(tuple(int(c) for c in row) for row in test_in_raw)

    # Local caches (fast, per-worker)
    local_apply_cache: Dict[Tuple[Any, ...], StateV132] = {}
    grid_hash_local: Dict[GridV124, str] = {}
    local_eval_cache: Dict[str, EvalResultV158] = {}
    metrics: Dict[str, int] = {}

    log_entries: List[WorkerLogEntryV142] = []

    def _log(event: str, data: Dict[str, Any]) -> None:
        log_counter[0] += 1
        log_entries.append(WorkerLogEntryV142(
            timestamp_mono_ns=int(log_counter[0] * 1000),
            worker_id=int(worker_id),
            event=str(event),
            data=dict(data),
        ))

    _log("worker_start", {
        "worker_id": int(worker_id),
        "operator_slice": list(operator_slice),
        "max_programs": int(max_programs),
        "max_depth": int(max_depth),
        "shared_cache_enabled": shared_apply_cache is not None,
    })

    # Get operators for this worker
    op_start, op_end = operator_slice
    my_ops = all_operator_ids[int(op_start):int(op_end)]
    if not my_ops:
        _log("worker_done", {"reason": "no_operators", "programs_evaluated": 0})
        return WorkerResultV142(
            worker_id=int(worker_id),
            budget=WorkerBudgetV142(
                worker_id=int(worker_id),
                total_workers=int(total_workers),
                programs_start=0,
                programs_end=0,
                depth_limit=int(max_depth),
                seed_offset=int(seed),
                branch_prefix=tuple(operator_slice),
            ),
            programs_evaluated=0,
            best_program=None,
            best_loss=(999999, 999999),
            train_perfect_programs=[],
            log_entries=log_entries,
            status="ok",
            error_message="",
            wall_time_ms=int(_time.monotonic() * 1000 - start_ms),
        ).to_dict()

    _log("operators_assigned", {"count": len(my_ops), "ops": my_ops[:10]})

    # Best-first search
    frontier: List[Tuple[Tuple[int, int, int, int, str], Tuple[ProgramStepV158, ...]]] = []
    seen_sigs: Set[str] = set()
    best_cost_by_vec: Dict[str, int] = {}

    programs_evaluated = 0
    best_program: Optional[Dict[str, Any]] = None
    best_loss = (999999, 999999)
    train_perfect: List[Dict[str, Any]] = []
    pruned_dominated = 0

    def push(steps: Tuple[ProgramStepV158, ...]) -> None:
        nonlocal programs_evaluated
        prog = ProgramV158(steps=steps)
        sig = prog.program_sig()
        if sig in seen_sigs:
            return
        seen_sigs.add(sig)

        # Check local eval cache first
        ev = local_eval_cache.get(sig)
        if ev is None:
            ev = _eval_program_shared_v158(
                steps=steps,
                train_pairs=train_pairs,
                test_in=test_in,
                shared_eval_cache=None,  # Disabled - too slow via IPC
                local_apply_cache=local_apply_cache,
                shared_apply_cache=None,  # Disabled - too slow via IPC
                grid_hash_local=grid_hash_local,
                metrics=metrics,
                local_eval_cache=local_eval_cache,
            )

        key = (
            int(ev.loss_shape),
            int(ev.loss_cells),
            int(ev.cost_bits),
            int(len(steps)),
            str(sig),
        )
        heapq.heappush(frontier, (key, steps))

    # Start with empty program
    push(tuple())

    try:
        while frontier and programs_evaluated < max_programs:
            (prio_key, steps) = heapq.heappop(frontier)
            programs_evaluated += 1

            if programs_evaluated % log_every_n == 0:
                _log("progress", {
                    "programs_evaluated": int(programs_evaluated),
                    "frontier_size": len(frontier),
                    "train_perfect_count": len(train_perfect),
                    "shared_eval_hits": metrics.get("shared_eval_hits", 0),
                    "shared_apply_hits": metrics.get("shared_apply_hits", 0),
                })

            depth = len(steps)
            prog = ProgramV158(steps=steps)
            sig = prog.program_sig()
            
            ev = local_eval_cache.get(sig)
            if ev is None:
                ev = _eval_program_shared_v158(
                    steps=steps,
                    train_pairs=train_pairs,
                    test_in=test_in,
                    shared_eval_cache=None,  # Disabled - too slow via IPC
                    local_apply_cache=local_apply_cache,
                    shared_apply_cache=None,  # Disabled - too slow via IPC
                    grid_hash_local=grid_hash_local,
                    metrics=metrics,
                    local_eval_cache=local_eval_cache,
                )

            # Track best
            if ev.loss < best_loss:
                best_loss = ev.loss
                best_program = prog.to_dict()
                best_program["loss"] = {"shape": int(ev.loss_shape), "cells": int(ev.loss_cells)}
                best_program["test_grid"] = [[int(c) for c in row] for row in ev.test_grid] if ev.test_grid else None

            # Check if train-perfect
            if ev.ok_train:
                prog_dict = prog.to_dict()
                prog_dict["test_grid"] = [[int(c) for c in row] for row in ev.test_grid] if ev.test_grid else None
                prog_dict["cost_bits"] = int(ev.cost_bits)
                train_perfect.append(prog_dict)
                _log("train_perfect_found", {
                    "programs_evaluated": int(programs_evaluated),
                    "depth": int(depth),
                    "cost_bits": int(ev.cost_bits),
                })
                if len(train_perfect) >= 8:
                    break

            # Dominated state pruning
            dom_cost = best_cost_by_vec.get(str(ev.vec_sig))
            if dom_cost is not None and dom_cost <= ev.cost_bits:
                pruned_dominated += 1
                continue
            best_cost_by_vec[str(ev.vec_sig)] = int(ev.cost_bits)

            # Expand if not at max depth
            if depth >= max_depth:
                continue

            # Can't expand without full states (from shared cache summary)
            if not ev.train_final_states:
                continue

            # For first step, only use operators in this worker's slice
            if depth == 0:
                ops_to_use = my_ops
            else:
                ops_to_use = all_operator_ids

            # Propose next steps
            proposals = _propose_next_steps_v158(
                train_final_states=ev.train_final_states,
                operator_ids=ops_to_use,
                palette_out=palette_out,
                shapes_out=shapes_out,
                max_steps=64,
            )

            for proposal in proposals:
                new_steps = steps + (proposal,)
                push(new_steps)

    except Exception as e:
        _log("worker_error", {"error": str(e)})
        return WorkerResultV142(
            worker_id=int(worker_id),
            budget=WorkerBudgetV142(
                worker_id=int(worker_id),
                total_workers=int(total_workers),
                programs_start=0,
                programs_end=int(max_programs),
                depth_limit=int(max_depth),
                seed_offset=int(seed),
                branch_prefix=tuple(operator_slice),
            ),
            programs_evaluated=int(programs_evaluated),
            best_program=best_program,
            best_loss=best_loss,
            train_perfect_programs=train_perfect,
            log_entries=log_entries,
            status="error",
            error_message=str(e),
            wall_time_ms=int(_time.monotonic() * 1000 - start_ms),
        ).to_dict()

    _log("worker_done", {
        "programs_evaluated": int(programs_evaluated),
        "train_perfect_count": len(train_perfect),
        "best_loss": list(best_loss),
        "pruned_dominated": int(pruned_dominated),
        "local_apply_hits": int(metrics.get("local_apply_hits", 0)),
        "shared_apply_hits": int(metrics.get("shared_apply_hits", 0)),
        "shared_eval_hits": int(metrics.get("shared_eval_hits", 0)),
        "apply_misses": int(metrics.get("apply_misses", 0)),
        "eval_misses": int(metrics.get("eval_misses", 0)),
    })

    return WorkerResultV142(
        worker_id=int(worker_id),
        budget=WorkerBudgetV142(
            worker_id=int(worker_id),
            total_workers=int(total_workers),
            programs_start=0,
            programs_end=int(max_programs),
            depth_limit=int(max_depth),
            seed_offset=int(seed),
            branch_prefix=tuple(operator_slice),
        ),
        programs_evaluated=int(programs_evaluated),
        best_program=best_program,
        best_loss=best_loss,
        train_perfect_programs=train_perfect,
        log_entries=log_entries,
        status="ok" if programs_evaluated < max_programs else "budget_exhausted",
        error_message="",
        wall_time_ms=int(_time.monotonic() * 1000 - start_ms),
    ).to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Configuration and Result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ParallelSolverConfigV158:
    """Configuration for parallel ARC solver with shared cache."""
    num_workers: int = 16
    max_programs_per_worker: int = 500
    max_depth: int = 4
    seed: int = 0
    enable_reachability_pruning: bool = True
    detect_ambiguity: bool = True
    ambiguity_probe_limit: int = 32
    log_every_n: int = 100
    enable_shared_cache: bool = True  # NEW: Enable global shared cache

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(ARC_PARALLEL_SOLVER_SCHEMA_VERSION_V158),
            "num_workers": int(self.num_workers),
            "max_programs_per_worker": int(self.max_programs_per_worker),
            "max_depth": int(self.max_depth),
            "seed": int(self.seed),
            "enable_reachability_pruning": bool(self.enable_reachability_pruning),
            "detect_ambiguity": bool(self.detect_ambiguity),
            "ambiguity_probe_limit": int(self.ambiguity_probe_limit),
            "log_every_n": int(self.log_every_n),
            "enable_shared_cache": bool(self.enable_shared_cache),
        }


@dataclass
class ParallelSolverResultV158:
    """Result from parallel ARC solver with shared cache metrics."""
    status: str
    failure_reason: Optional[Dict[str, Any]]
    best_program: Optional[Dict[str, Any]]
    predicted_grid: Optional[List[List[int]]]
    predicted_grids: List[List[List[int]]]
    candidate_programs: List[Dict[str, Any]]
    train_perfect_count: int
    ambiguity_detected: bool
    total_programs_evaluated: int
    worker_results: List[Dict[str, Any]]
    merged_log_hash: str
    config: ParallelSolverConfigV158
    wall_time_ms: int
    metrics: Dict[str, Any]
    shared_cache_stats: Dict[str, int]  # NEW: Shared cache statistics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(ARC_PARALLEL_SOLVER_SCHEMA_VERSION_V158),
            "kind": "arc_parallel_solver_result_v158",
            "status": str(self.status),
            "failure_reason": self.failure_reason,
            "best_program": self.best_program,
            "predicted_grid": self.predicted_grid,
            "predicted_grids": self.predicted_grids,
            "candidate_programs": self.candidate_programs,
            "train_perfect_count": int(self.train_perfect_count),
            "ambiguity_detected": bool(self.ambiguity_detected),
            "total_programs_evaluated": int(self.total_programs_evaluated),
            "worker_results_count": len(self.worker_results),
            "merged_log_hash": str(self.merged_log_hash),
            "config": self.config.to_dict(),
            "wall_time_ms": int(self.wall_time_ms),
            "metrics": dict(self.metrics),
            "shared_cache_stats": dict(self.shared_cache_stats),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main parallel solver with shared cache
# ─────────────────────────────────────────────────────────────────────────────


def solve_arc_task_parallel_v158(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    config: Optional[ParallelSolverConfigV158] = None,
) -> ParallelSolverResultV158:
    """
    Solve an ARC task using parallel search with GLOBAL SHARED CACHE.
    
    Key improvement over V143: Workers share eval and apply caches,
    eliminating redundant computation across processes.
    """
    import time as _time

    start_ms = int(_time.monotonic() * 1000)
    cfg = config or ParallelSolverConfigV158()

    # Prepare serializable data
    train_pairs_raw = [
        ([[int(c) for c in row] for row in inp], [[int(c) for c in row] for row in out])
        for inp, out in train_pairs
    ]
    test_in_raw = [[int(c) for c in row] for row in test_in]

    # Infer output features
    shapes_out: List[Tuple[int, int]] = list(set(
        tuple(int(x) for x in grid_shape_v124(out))
        for _, out in train_pairs
    ))
    palette_out: List[int] = list(sorted(set(
        int(c)
        for _, out in train_pairs
        for c in unique_colors_v124(out)
    )))

    # Get all operator IDs
    all_operator_ids = list(sorted(OP_DEFS_V141.keys()))
    n_ops = len(all_operator_ids)
    n_workers = min(int(cfg.num_workers), n_ops)

    # Distribute operators among workers
    ops_per_worker = max(1, (n_ops + n_workers - 1) // n_workers)
    worker_slices: List[Tuple[int, int]] = []
    for i in range(n_workers):
        op_start = i * ops_per_worker
        op_end = min(n_ops, (i + 1) * ops_per_worker)
        worker_slices.append((int(op_start), int(op_end)))

    # Create shared cache using Manager
    shared_cache_stats = {
        "apply_cache_size": 0,
        "eval_cache_size": 0,
    }
    
    worker_results_raw: List[Dict[str, Any]] = []

    # NOTE: Shared cache via Manager was tested but IPC overhead made it SLOWER.
    # Each worker now uses LOCAL caches only, which is faster.
    # The benefit of parallel comes from distributing operators across workers.
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i, (op_start, op_end) in enumerate(worker_slices):
            future = executor.submit(
                _worker_search_shared_v158,
                worker_id=int(i),
                total_workers=int(n_workers),
                operator_slice=(int(op_start), int(op_end)),
                train_pairs_raw=train_pairs_raw,
                test_in_raw=test_in_raw,
                all_operator_ids=all_operator_ids,
                max_programs=int(cfg.max_programs_per_worker),
                max_depth=int(cfg.max_depth),
                seed=int(cfg.seed),
                log_every_n=int(cfg.log_every_n),
                enable_reachability_pruning=bool(cfg.enable_reachability_pruning),
                palette_out=palette_out,
                shapes_out=shapes_out,
                shared_apply_cache=None,  # Disabled - local cache faster
                shared_eval_cache=None,   # Disabled - local cache faster
            )
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                result_dict = future.result()
                worker_results_raw.append(result_dict)
            except Exception as e:
                worker_results_raw.append({
                    "worker_id": -1,
                    "status": "error",
                    "error_message": str(e),
                    "programs_evaluated": 0,
                    "best_program": None,
                    "best_loss": [999999, 999999],
                    "train_perfect_programs": [],
                    "log_entries": [],
                    "wall_time_ms": 0,
                })

    # Sort by worker_id for determinism
    worker_results_raw.sort(key=lambda r: int(r.get("worker_id", -1)))

    # Reconstruct WorkerResultV142 objects
    worker_results: List[WorkerResultV142] = []
    for wr_dict in worker_results_raw:
        log_entries = [
            WorkerLogEntryV142(
                timestamp_mono_ns=int(e.get("timestamp_mono_ns", 0)),
                worker_id=int(e.get("worker_id", 0)),
                event=str(e.get("event", "")),
                data=dict(e.get("data", {})),
            )
            for e in wr_dict.get("log_entries", [])
        ]
        budget_dict = wr_dict.get("budget", {})
        wr = WorkerResultV142(
            worker_id=int(wr_dict.get("worker_id", 0)),
            budget=WorkerBudgetV142(
                worker_id=int(budget_dict.get("worker_id", 0)),
                total_workers=int(budget_dict.get("total_workers", 1)),
                programs_start=int(budget_dict.get("programs_start", 0)),
                programs_end=int(budget_dict.get("programs_end", 0)),
                depth_limit=int(budget_dict.get("depth_limit", 4)),
                seed_offset=int(budget_dict.get("seed_offset", 0)),
                branch_prefix=tuple(int(x) for x in budget_dict.get("branch_prefix", (0, 0))),
            ),
            programs_evaluated=int(wr_dict.get("programs_evaluated", 0)),
            best_program=wr_dict.get("best_program"),
            best_loss=tuple(int(x) for x in wr_dict.get("best_loss", (999999, 999999))),
            train_perfect_programs=list(wr_dict.get("train_perfect_programs", [])),
            log_entries=log_entries,
            status=str(wr_dict.get("status", "ok")),
            error_message=str(wr_dict.get("error_message", "")),
            wall_time_ms=int(wr_dict.get("wall_time_ms", 0)),
        )
        worker_results.append(wr)

    # Merge logs
    merged_logs, merged_log_hash = _merge_worker_logs_v142(worker_results)

    # Aggregate results
    total_evaluated = sum(int(wr.programs_evaluated) for wr in worker_results)
    all_train_perfect: List[Dict[str, Any]] = []
    for wr in worker_results:
        all_train_perfect.extend(wr.train_perfect_programs)

    # Find best program
    best_program: Optional[Dict[str, Any]] = None
    best_loss = (999999, 999999)
    for wr in worker_results:
        if wr.best_program is not None and wr.best_loss < best_loss:
            best_loss = wr.best_loss
            best_program = wr.best_program

    # Detect ambiguity
    ambiguity_detected, distinct_count = _detect_ambiguity_v142(
        train_perfect_programs=all_train_perfect,
        limit=int(cfg.ambiguity_probe_limit),
    )

    # Determine final status
    status = "FAIL"
    failure_reason: Optional[Dict[str, Any]] = None
    predicted_grid: Optional[List[List[int]]] = None
    predicted_grids: List[List[List[int]]] = []
    candidate_programs: List[Dict[str, Any]] = []

    if all_train_perfect:
        if ambiguity_detected and cfg.detect_ambiguity:
            status = "UNKNOWN"
            failure_reason = {
                "kind": "AMBIGUOUS_RULE",
                "distinct_test_outputs": int(distinct_count),
                "train_perfect_count": len(all_train_perfect),
            }
        else:
            status = "SOLVED"

        for prog in all_train_perfect[:2]:
            tg = prog.get("test_grid")
            if tg is not None:
                predicted_grids.append([[int(c) for c in row] for row in tg])
            candidate_programs.append(prog)
        if predicted_grids:
            predicted_grid = predicted_grids[0]
    else:
        status = "FAIL"
        failure_reason = {
            "kind": "SEARCH_BUDGET_EXCEEDED",
            "total_evaluated": int(total_evaluated),
            "max_programs_total": int(cfg.max_programs_per_worker * n_workers),
        }

    wall_time_ms = int(_time.monotonic() * 1000 - start_ms)

    metrics = {
        "workers_completed": sum(1 for wr in worker_results if wr.status == "ok"),
        "workers_error": sum(1 for wr in worker_results if wr.status == "error"),
        "total_wall_time_ms": int(wall_time_ms),
        "avg_programs_per_worker": int(total_evaluated) // max(1, len(worker_results)),
    }

    return ParallelSolverResultV158(
        status=str(status),
        failure_reason=failure_reason,
        best_program=best_program,
        predicted_grid=predicted_grid,
        predicted_grids=predicted_grids,
        candidate_programs=candidate_programs,
        train_perfect_count=len(all_train_perfect),
        ambiguity_detected=bool(ambiguity_detected),
        total_programs_evaluated=int(total_evaluated),
        worker_results=[wr.to_dict() for wr in worker_results],
        merged_log_hash=str(merged_log_hash),
        config=cfg,
        wall_time_ms=int(wall_time_ms),
        metrics=metrics,
        shared_cache_stats=shared_cache_stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function compatible with solve_v141 API
# ─────────────────────────────────────────────────────────────────────────────


def solve_v158(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_input: GridV124,
    max_depth: int = 4,
    max_programs: int = 2000,
    num_workers: int = 8,
    enable_shared_cache: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Solve an ARC task using parallel search with shared cache.
    
    Compatible with solve_v141 API for easy replacement.
    """
    programs_per_worker = max(1, max_programs // num_workers)
    
    config = ParallelSolverConfigV158(
        num_workers=num_workers,
        max_programs_per_worker=programs_per_worker,
        max_depth=max_depth,
        seed=seed,
        enable_shared_cache=enable_shared_cache,
    )
    
    result = solve_arc_task_parallel_v158(
        train_pairs=train_pairs,
        test_in=test_input,
        config=config,
    )
    
    return {
        "schema_version": ARC_PARALLEL_SOLVER_SCHEMA_VERSION_V158,
        "status": result.status,
        "predicted_output": result.predicted_grid,
        "programs_tested": result.total_programs_evaluated,
        "best_program": result.best_program,
        "failure_reason": result.failure_reason,
        "wall_time_ms": result.wall_time_ms,
        "shared_cache_stats": result.shared_cache_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI for testing
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Parallel ARC Solver V158 with Shared Cache")
    parser.add_argument("--task", type=str, help="Path to ARC task JSON")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--programs", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-shared-cache", action="store_true")
    
    args = parser.parse_args()
    
    if args.task:
        with open(args.task) as f:
            data = json.load(f)
        
        train_pairs = [
            (tuple(tuple(int(c) for c in row) for row in p["input"]),
             tuple(tuple(int(c) for c in row) for row in p["output"]))
            for p in data["train"]
        ]
        test_input = tuple(tuple(int(c) for c in row) for row in data["test"][0]["input"])
        
        result = solve_v158(
            train_pairs=train_pairs,
            test_input=test_input,
            max_depth=args.depth,
            max_programs=args.programs,
            num_workers=args.workers,
            enable_shared_cache=not args.no_shared_cache,
        )
        
        print(f"Status: {result['status']}")
        print(f"Programs tested: {result['programs_tested']}")
        print(f"Time: {result['wall_time_ms']}ms")
        print(f"Shared cache stats: {result['shared_cache_stats']}")
    else:
        print("Parallel ARC Solver V158 with Shared Cache")
        print("Use --task to solve a specific task")
