"""
arc_parallel_solver_v143.py - Parallel ARC Solver with Real Search Integration.

This module integrates parallel_planner_v142 with the actual arc_solver_v141
evaluation and search logic, enabling true parallel deterministic ARC solving
across multiple CPU cores.

Design principles:
- Deterministic: same seed + inputs → same outputs (parallel or sequential).
- Fail-closed: UNKNOWN on ambiguity/timeout rather than guessing.
- Auditable: separate logs per worker, merged into WORM-compliant ledger.
- Explicit budget: no hidden timeouts; budget divided among workers.
- Real evaluation: uses arc_solver_v141 apply/eval logic for correctness.

Schema version: 143
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
    PARALLEL_PLANNER_SCHEMA_VERSION_V142,
    ParallelSearchConfigV142,
    SearchBudgetV142,
    WorkerBudgetV142,
    WorkerLogEntryV142,
    WorkerResultV142,
    distribute_budget_v142,
    _merge_worker_logs_v142,
    _detect_ambiguity_v142,
)

ARC_PARALLEL_SOLVER_SCHEMA_VERSION_V143 = 143

# ─────────────────────────────────────────────────────────────────────────────
# Program step and evaluation structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProgramStepV143:
    """A single step in an ARC program."""
    op_id: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"op_id": str(self.op_id), "args": dict(self.args)}


@dataclass(frozen=True)
class ProgramV143:
    """A complete ARC program (sequence of steps)."""
    steps: Tuple[ProgramStepV143, ...]

    def program_sig(self) -> str:
        return sha256_hex(
            canonical_json_dumps(
                {"steps": [s.to_dict() for s in self.steps]}
            ).encode("utf-8")
        )[:32]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(ARC_PARALLEL_SOLVER_SCHEMA_VERSION_V143),
            "steps": [s.to_dict() for s in self.steps],
            "program_sig": str(self.program_sig()),
        }


@dataclass
class EvalResultV143:
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


# ─────────────────────────────────────────────────────────────────────────────
# Apply and evaluation caching
# ─────────────────────────────────────────────────────────────────────────────


def _state_sig_v143(state: StateV132, grid_hash_cache: Dict[GridV124, str]) -> str:
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


def _apply_step_v143(
    *,
    state: StateV132,
    op_id: str,
    args: Dict[str, Any],
    apply_cache: Dict[Tuple[Any, ...], StateV132],
    grid_hash_cache: Dict[GridV124, str],
    metrics: Dict[str, int],
) -> StateV132:
    """Apply a single step with caching."""
    state_sig = _state_sig_v143(state, grid_hash_cache)
    args_sig = canonical_json_dumps(args)
    cache_key = (str(state_sig), str(op_id), str(args_sig))

    hit = apply_cache.get(cache_key)
    if hit is not None:
        metrics["apply_cache_hits"] = int(metrics.get("apply_cache_hits", 0)) + 1
        return hit

    metrics["apply_cache_misses"] = int(metrics.get("apply_cache_misses", 0)) + 1
    result = apply_op_v141(state=state, op_id=str(op_id), args=dict(args))
    apply_cache[cache_key] = result
    return result


def _eval_program_v143(
    *,
    steps: Sequence[ProgramStepV143],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    apply_cache: Dict[Tuple[Any, ...], StateV132],
    grid_hash_cache: Dict[GridV124, str],
    eval_cache: Dict[str, EvalResultV143],
    metrics: Dict[str, int],
) -> EvalResultV143:
    """Evaluate a program on training pairs and test input."""
    prog = ProgramV143(steps=tuple(steps))
    sig = prog.program_sig()

    cached = eval_cache.get(sig)
    if cached is not None:
        metrics["eval_cache_hits"] = int(metrics.get("eval_cache_hits", 0)) + 1
        return cached

    metrics["eval_cache_misses"] = int(metrics.get("eval_cache_misses", 0)) + 1

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
                st = _apply_step_v143(
                    state=st,
                    op_id=str(step.op_id),
                    args=dict(step.args),
                    apply_cache=apply_cache,
                    grid_hash_cache=grid_hash_cache,
                    metrics=metrics,
                )
            got = st.grid
        except Exception as e:
            ok = False
            if not mismatches:
                mismatches.append({"kind": "exception", "error": str(e)})

        train_state_sigs.append(_state_sig_v143(st, grid_hash_cache))
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
            stt = _apply_step_v143(
                state=stt,
                op_id=str(step.op_id),
                args=dict(step.args),
                apply_cache=apply_cache,
                grid_hash_cache=grid_hash_cache,
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
        "test_state_sig": _state_sig_v143(stt, grid_hash_cache),
        "loss": {"shape": int(loss_shape), "cells": int(loss_cells)},
    }
    vec_sig = sha256_hex(canonical_json_dumps(vec_obj).encode("utf-8"))[:32]

    result = EvalResultV143(
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

    eval_cache[sig] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Next step proposal (operator enumeration)
# ─────────────────────────────────────────────────────────────────────────────


def _slots_available_v143(state: StateV132) -> Dict[str, bool]:
    """Determine which slots are available in the current state."""
    return {
        "grid": True,
        "objset": state.objset is not None,
        "obj": state.obj is not None,
        "bbox": state.bbox is not None,
        "patch": state.patch is not None,
    }


def _propose_next_steps_v143(
    *,
    train_final_states: Sequence[StateV132],
    operator_ids: Sequence[str],
    palette_out: Sequence[int],
    shapes_out: Sequence[Tuple[int, int]],
    max_steps: int = 64,
) -> List[ProgramStepV143]:
    """Propose next steps based on current state slots and operators."""
    proposals: List[ProgramStepV143] = []

    # Analyze available slots across all training states
    all_avail: Dict[str, int] = {"grid": 0, "objset": 0, "obj": 0, "bbox": 0, "patch": 0}
    for st in train_final_states:
        avail = _slots_available_v143(st)
        for k, v in avail.items():
            if v:
                all_avail[k] = int(all_avail.get(k, 0)) + 1

    n_pairs = len(train_final_states)
    
    # Include colors from both output palette AND current state grids (input colors)
    # This is critical for replace_color which needs both from_color (input) and to_color (output)
    state_colors: Set[int] = set()
    for st in train_final_states:
        for row in st.grid:
            for c in row:
                state_colors.add(int(c))
    
    all_colors = set(int(c) for c in palette_out) | state_colors
    colors = list(sorted(all_colors))[:10]  # Limit to 10 colors

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
            proposals.append(ProgramStepV143(op_id=str(op_id), args={}))
        elif op_id == "replace_color":
            for c1 in colors:
                for c2 in colors:
                    if c1 != c2:
                        proposals.append(ProgramStepV143(op_id=str(op_id), args={"from_color": int(c1), "to_color": int(c2)}))
        elif op_id == "fill_color":
            for c in colors:
                proposals.append(ProgramStepV143(op_id=str(op_id), args={"color": int(c)}))
        elif op_id == "paint_mask":
            for c in colors:
                proposals.append(ProgramStepV143(op_id=str(op_id), args={"color": int(c)}))
        elif op_id in {"cc4_nonbg", "cc4_nonbg_multicolor"}:
            for bg in colors:
                proposals.append(ProgramStepV143(op_id=str(op_id), args={"bg": int(bg)}))
        elif op_id == "select_obj":
            for idx in range(8):
                proposals.append(ProgramStepV143(op_id=str(op_id), args={"index": int(idx)}))
        elif op_id == "obj_bbox":
            proposals.append(ProgramStepV143(op_id=str(op_id), args={}))
        elif op_id == "crop_bbox":
            proposals.append(ProgramStepV143(op_id=str(op_id), args={}))
        elif op_id == "commit_patch":
            proposals.append(ProgramStepV143(op_id=str(op_id), args={}))
        elif op_id == "crop_bbox_nonzero":
            proposals.append(ProgramStepV143(op_id=str(op_id), args={}))
        elif op_id == "pad_to":
            for h, w in shapes_out[:4]:
                proposals.append(ProgramStepV143(op_id=str(op_id), args={"h": int(h), "w": int(w)}))
        elif op_id == "repeat_grid":
            for rh in [2, 3]:
                for rw in [2, 3]:
                    proposals.append(ProgramStepV143(op_id=str(op_id), args={"repeat_h": int(rh), "repeat_w": int(rw)}))
        elif op_id == "gravity":
            for d in ["up", "down", "left", "right"]:
                proposals.append(ProgramStepV143(op_id=str(op_id), args={"direction": str(d)}))
        elif op_id in {"flood_fill"}:
            for c in colors:
                proposals.append(ProgramStepV143(op_id=str(op_id), args={"color": int(c), "r": 0, "c": 0}))
        elif op_id == "new_canvas":
            for h, w in shapes_out[:2]:
                for c in colors[:2]:
                    proposals.append(ProgramStepV143(op_id=str(op_id), args={"h": int(h), "w": int(w), "fill": int(c)}))
        elif op_id == "bbox_by_color":
            for c in colors:
                proposals.append(ProgramStepV143(op_id=str(op_id), args={"color": int(c)}))
        elif op_id in {"translate"}:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr != 0 or dc != 0:
                        proposals.append(ProgramStepV143(op_id=str(op_id), args={"dr": int(dr), "dc": int(dc)}))
        elif op_id in {"overlay_self_translate", "propagate_color_translate", "propagate_nonbg_translate"}:
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    if dr != 0 or dc != 0:
                        if op_id == "propagate_color_translate":
                            for c in colors:
                                proposals.append(ProgramStepV143(op_id=str(op_id), args={"dr": int(dr), "dc": int(dc), "color": int(c)}))
                        else:
                            proposals.append(ProgramStepV143(op_id=str(op_id), args={"dr": int(dr), "dc": int(dc)}))
        elif op_id in {"symmetry_fill_h", "symmetry_fill_v", "downsample_mode", "fill_enclosed_region"}:
            proposals.append(ProgramStepV143(op_id=str(op_id), args={}))
        else:
            # Generic op without args
            if not od.reads or all(int(all_avail.get(str(r), 0)) > 0 for r in od.reads):
                proposals.append(ProgramStepV143(op_id=str(op_id), args={}))

        if len(proposals) >= int(max_steps):
            break

    return proposals[:int(max_steps)]


# ─────────────────────────────────────────────────────────────────────────────
# Parallel worker task
# ─────────────────────────────────────────────────────────────────────────────


def _worker_search_v143(
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
) -> Dict[str, Any]:
    """
    Worker search task for parallel ARC solving.

    Each worker explores programs whose first operator falls in its assigned slice.
    This is deterministic and reproducible.
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

    # Caches (per-worker, not shared)
    apply_cache: Dict[Tuple[Any, ...], StateV132] = {}
    grid_hash_cache: Dict[GridV124, str] = {}
    eval_cache: Dict[str, EvalResultV143] = {}
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

    # Best-first search within this worker's operator slice
    # Priority: (loss_shape, loss_cells, cost_bits, depth, sig)
    frontier: List[Tuple[Tuple[int, int, int, int, str], Tuple[ProgramStepV143, ...]]] = []
    seen_sigs: Set[str] = set()
    best_cost_by_vec: Dict[str, int] = {}

    programs_evaluated = 0
    best_program: Optional[Dict[str, Any]] = None
    best_loss = (999999, 999999)
    train_perfect: List[Dict[str, Any]] = []

    # Pruning counters
    pruned_shape = 0
    pruned_palette = 0
    pruned_dominated = 0

    def push(steps: Tuple[ProgramStepV143, ...]) -> None:
        nonlocal programs_evaluated
        prog = ProgramV143(steps=steps)
        sig = prog.program_sig()
        if sig in seen_sigs:
            return
        seen_sigs.add(sig)

        ev = _eval_program_v143(
            steps=steps,
            train_pairs=train_pairs,
            test_in=test_in,
            apply_cache=apply_cache,
            grid_hash_cache=grid_hash_cache,
            eval_cache=eval_cache,
            metrics=metrics,
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
                })

            depth = len(steps)
            ev = _eval_program_v143(
                steps=steps,
                train_pairs=train_pairs,
                test_in=test_in,
                apply_cache=apply_cache,
                grid_hash_cache=grid_hash_cache,
                eval_cache=eval_cache,
                metrics=metrics,
            )

            # Track best
            if ev.loss < best_loss:
                best_loss = ev.loss
                best_program = ProgramV143(steps=steps).to_dict()
                best_program["loss"] = {"shape": int(ev.loss_shape), "cells": int(ev.loss_cells)}
                best_program["test_grid"] = [[int(c) for c in row] for row in ev.test_grid] if ev.test_grid else None

            # Check if train-perfect
            if ev.ok_train:
                prog_dict = ProgramV143(steps=steps).to_dict()
                prog_dict["test_grid"] = [[int(c) for c in row] for row in ev.test_grid] if ev.test_grid else None
                prog_dict["cost_bits"] = int(ev.cost_bits)
                train_perfect.append(prog_dict)
                _log("train_perfect_found", {
                    "programs_evaluated": int(programs_evaluated),
                    "depth": int(depth),
                    "cost_bits": int(ev.cost_bits),
                })
                # Continue searching for ambiguity detection (up to a limit)
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

            # For first step, only use operators in this worker's slice
            # For subsequent steps, use all operators
            if depth == 0:
                ops_to_use = my_ops
            else:
                ops_to_use = all_operator_ids

            # Propose next steps
            proposals = _propose_next_steps_v143(
                train_final_states=ev.train_final_states,
                operator_ids=ops_to_use,
                palette_out=palette_out,
                shapes_out=shapes_out,
                max_steps=64,
            )

            # Reachability pruning
            if enable_reachability_pruning:
                want_shapes = [tuple(int(x) for x in grid_shape_v124(out)) for _, out in train_pairs]
                want_palettes = [set(int(c) for c in unique_colors_v124(out)) for _, out in train_pairs]
                steps_left = max_depth - depth - 1

                for proposal in proposals:
                    new_steps = steps + (proposal,)
                    push(new_steps)
            else:
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
        "apply_cache_hits": int(metrics.get("apply_cache_hits", 0)),
        "apply_cache_misses": int(metrics.get("apply_cache_misses", 0)),
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
# Parallel solver coordinator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ParallelSolverConfigV143:
    """Configuration for parallel ARC solver."""
    num_workers: int = 16
    max_programs_per_worker: int = 500
    max_depth: int = 4
    seed: int = 0
    enable_reachability_pruning: bool = True
    detect_ambiguity: bool = True
    ambiguity_probe_limit: int = 32
    log_every_n: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(ARC_PARALLEL_SOLVER_SCHEMA_VERSION_V143),
            "num_workers": int(self.num_workers),
            "max_programs_per_worker": int(self.max_programs_per_worker),
            "max_depth": int(self.max_depth),
            "seed": int(self.seed),
            "enable_reachability_pruning": bool(self.enable_reachability_pruning),
            "detect_ambiguity": bool(self.detect_ambiguity),
            "ambiguity_probe_limit": int(self.ambiguity_probe_limit),
            "log_every_n": int(self.log_every_n),
        }


@dataclass
class ParallelSolverResultV143:
    """Result from parallel ARC solver."""
    status: str  # "SOLVED" | "UNKNOWN" | "FAIL"
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
    config: ParallelSolverConfigV143
    wall_time_ms: int
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(ARC_PARALLEL_SOLVER_SCHEMA_VERSION_V143),
            "kind": "arc_parallel_solver_result_v143",
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
        }


def solve_arc_task_parallel_v143(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    config: Optional[ParallelSolverConfigV143] = None,
) -> ParallelSolverResultV143:
    """
    Solve an ARC task using parallel deterministic search.

    Distributes the search across multiple workers, each exploring a slice
    of the operator space, then merges results deterministically.
    """
    import time as _time

    start_ms = int(_time.monotonic() * 1000)
    cfg = config or ParallelSolverConfigV143()

    # Prepare serializable data
    train_pairs_raw = [
        ([[int(c) for c in row] for row in inp], [[int(c) for c in row] for row in out])
        for inp, out in train_pairs
    ]
    test_in_raw = [[int(c) for c in row] for row in test_in]

    # Infer output features for proposals
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

    # Run workers in parallel
    worker_results_raw: List[Dict[str, Any]] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i, (op_start, op_end) in enumerate(worker_slices):
            future = executor.submit(
                _worker_search_v143,
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

    # Reconstruct WorkerResultV142 objects for log merging
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

    # Find best program across all workers
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

        # Extract predictions
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

    # Aggregate metrics
    metrics = {
        "workers_completed": sum(1 for wr in worker_results if wr.status == "ok"),
        "workers_error": sum(1 for wr in worker_results if wr.status == "error"),
        "total_wall_time_ms": int(wall_time_ms),
        "avg_programs_per_worker": int(total_evaluated) // max(1, len(worker_results)),
    }

    return ParallelSolverResultV143(
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
    )


# ─────────────────────────────────────────────────────────────────────────────
# WORM-compliant ledger integration
# ─────────────────────────────────────────────────────────────────────────────


def write_parallel_solve_to_ledger_v143(
    *,
    result: ParallelSolverResultV143,
    task_id: str,
    ledger_path: str,
    prev_hash: str = "",
) -> str:
    """
    Write parallel solver result to WORM-compliant ledger.

    Returns the hash of the new ledger entry.
    """
    import json

    entry = {
        "schema_version": int(ARC_PARALLEL_SOLVER_SCHEMA_VERSION_V143),
        "kind": "parallel_solve_ledger_entry_v143",
        "task_id": str(task_id),
        "result": result.to_dict(),
        "prev_hash": str(prev_hash),
    }

    entry_hash = sha256_hex(canonical_json_dumps(entry).encode("utf-8"))
    entry["entry_hash"] = str(entry_hash)

    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    return str(entry_hash)
