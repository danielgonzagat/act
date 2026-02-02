"""
agi_loop_v144.py - The Closed-Loop AGI Engine.

Implements the deterministic AGI loop:
Plan → Execute → Validate → Repair → Mine → Promote → Re-run

This is the core engine that enforces the 7 Survival Laws and
guarantees concept creation/usage as a survival condition.

Design principles:
- Closed loop with mandatory validation checkpoints
- If task survives without concept creation → inject failure
- Mining occurs WITHIN the loop, not as a post-process
- Promotion decisions are deterministic and auditable
- All state transitions recorded for WORM ledger

Schema version: 144
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from .agi_regime_v144 import (
    AGIRegimeConfig,
    ConceptLifecycleManager,
    ConceptMetrics,
    ConceptState,
    LawValidationResult,
    RegimeValidationResult,
    SurvivalLaw,
    compute_regime_loss,
    validate_survival_laws,
)
from .csv_composed_miner_v144 import (
    CallSubgraph,
    ComposedMinerConfig,
    ComposedMiningResult,
    extract_call_subgraphs,
    mine_composed_concepts,
    run_composed_mining_pipeline,
)


AGI_LOOP_SCHEMA_VERSION_V144 = 144


# ─────────────────────────────────────────────────────────────────────────────
# Loop Phase Definitions
# ─────────────────────────────────────────────────────────────────────────────


class LoopPhase(str, Enum):
    """Phases of the AGI loop."""
    PLAN = "plan"
    EXECUTE = "execute"
    VALIDATE = "validate"
    REPAIR = "repair"
    MINE = "mine"
    PROMOTE = "promote"
    RERUN = "rerun"
    COMPLETE = "complete"
    FAILED = "failed"

    def __str__(self) -> str:
        return str(self.value)


class LoopExitReason(str, Enum):
    """Reasons for loop exit."""
    SUCCESS = "success"
    MAX_ITERATIONS = "max_iterations"
    VALIDATION_FAILED = "validation_failed"
    REGIME_VIOLATION = "regime_violation"
    NO_REPAIR_POSSIBLE = "no_repair_possible"
    TIMEOUT = "timeout"
    EXTERNAL_ABORT = "external_abort"

    def __str__(self) -> str:
        return str(self.value)


# ─────────────────────────────────────────────────────────────────────────────
# Loop State
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LoopIteration:
    """Record of one loop iteration."""
    iteration: int
    phase: LoopPhase
    task_id: str
    regime_result: Optional[RegimeValidationResult] = None
    execution_result: Optional[Dict[str, Any]] = None
    mining_result: Optional[ComposedMiningResult] = None
    concepts_promoted: List[str] = field(default_factory=list)
    repairs_attempted: int = 0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(AGI_LOOP_SCHEMA_VERSION_V144),
            "kind": "loop_iteration_v144",
            "iteration": int(self.iteration),
            "phase": str(self.phase),
            "task_id": str(self.task_id),
            "regime_result": self.regime_result.to_dict() if self.regime_result else None,
            "execution_result": dict(self.execution_result) if self.execution_result else None,
            "mining_result": self.mining_result.to_dict() if self.mining_result else None,
            "concepts_promoted": list(self.concepts_promoted),
            "repairs_attempted": int(self.repairs_attempted),
            "timestamp": str(self.timestamp),
        }


@dataclass
class LoopState:
    """Full state of the AGI loop."""
    task_id: str
    current_phase: LoopPhase
    current_iteration: int
    max_iterations: int
    iterations: List[LoopIteration] = field(default_factory=list)
    
    # Accumulated state
    total_concepts_created: int = 0
    total_concepts_promoted: int = 0
    total_repairs: int = 0
    traces_accumulated: List[Dict[str, Any]] = field(default_factory=list)
    
    # Lifecycle manager
    lifecycle_manager: Optional[ConceptLifecycleManager] = None
    
    # Exit conditions
    exit_reason: Optional[LoopExitReason] = None
    final_success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(AGI_LOOP_SCHEMA_VERSION_V144),
            "kind": "loop_state_v144",
            "task_id": str(self.task_id),
            "current_phase": str(self.current_phase),
            "current_iteration": int(self.current_iteration),
            "max_iterations": int(self.max_iterations),
            "iterations": [it.to_dict() for it in self.iterations],
            "total_concepts_created": int(self.total_concepts_created),
            "total_concepts_promoted": int(self.total_concepts_promoted),
            "total_repairs": int(self.total_repairs),
            "exit_reason": str(self.exit_reason) if self.exit_reason else None,
            "final_success": bool(self.final_success),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loop Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AGILoopConfig:
    """Configuration for the AGI loop engine."""
    max_iterations: int = 10
    max_repairs_per_iteration: int = 3
    
    # Regime settings
    regime_level: str = "full"  # "bootstrap" | "intermediate" | "full"
    min_concept_depth: int = 1
    require_composition: bool = True
    require_reuse: bool = False  # Start relaxed
    
    # Mining settings
    mining_config: ComposedMinerConfig = field(default_factory=ComposedMinerConfig)
    max_promotions_per_iteration: int = 5
    
    # Validation settings
    require_pcc: bool = True
    require_hash_proof: bool = True
    
    # Execution settings
    timeout_ms: int = 60000
    parallel_workers: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(AGI_LOOP_SCHEMA_VERSION_V144),
            "kind": "agi_loop_config_v144",
            "max_iterations": int(self.max_iterations),
            "max_repairs_per_iteration": int(self.max_repairs_per_iteration),
            "regime_level": str(self.regime_level),
            "min_concept_depth": int(self.min_concept_depth),
            "require_composition": bool(self.require_composition),
            "require_reuse": bool(self.require_reuse),
            "mining_config": self.mining_config.to_dict(),
            "max_promotions_per_iteration": int(self.max_promotions_per_iteration),
            "require_pcc": bool(self.require_pcc),
            "require_hash_proof": bool(self.require_hash_proof),
            "timeout_ms": int(self.timeout_ms),
            "parallel_workers": int(self.parallel_workers),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Phase Handlers
# ─────────────────────────────────────────────────────────────────────────────


def phase_plan(
    state: LoopState,
    task: Dict[str, Any],
    config: AGILoopConfig,
) -> Tuple[LoopPhase, Dict[str, Any]]:
    """
    PLAN phase: Generate execution plan from task.
    
    The plan MUST include concept_csv operations if regime requires.
    """
    plan: Dict[str, Any] = {
        "task_id": str(task.get("id", state.task_id)),
        "inputs": dict(task.get("inputs", {})),
        "expected_outputs": dict(task.get("expected_outputs", {})),
        "operations": [],
    }
    
    # If regime requires concepts, plan must include concept operations
    if config.regime_level in ("intermediate", "full"):
        plan["operations"].append({
            "kind": "concept_lookup",
            "match": task.get("match", {}),
        })
        plan["operations"].append({
            "kind": "concept_execute",
            "require_depth": config.min_concept_depth,
        })
    
    return LoopPhase.EXECUTE, plan


def phase_execute(
    state: LoopState,
    plan: Dict[str, Any],
    task: Dict[str, Any],
    concept_store: Sequence[Act],
    config: AGILoopConfig,
    executor: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Tuple[LoopPhase, Dict[str, Any]]:
    """
    EXECUTE phase: Run the plan with concept_csv operations.
    
    Collects execution trace for mining.
    """
    trace: Dict[str, Any] = {
        "trace_id": f"trace_{state.task_id}_{state.current_iteration}",
        "context_id": str(task.get("context_id", state.task_id)),
        "family_id": str(task.get("family_id", "default")),
        "events": [],
        "concepts_used": [],
        "outputs": {},
        "success": False,
    }
    
    # If no executor provided, simulate concept lookup and execution
    if executor is None:
        # Find matching concepts
        match_criteria = task.get("match", {})
        matching_concepts: List[Act] = []
        
        for concept in concept_store:
            if not concept.active:
                continue
            if concept.kind != "concept_csv":
                continue
            # Simple match: check if concept's match is subset of criteria
            concept_match = concept.match or {}
            if all(concept_match.get(k) == v for k, v in match_criteria.items() if k in concept_match):
                matching_concepts.append(concept)
        
        # Record concept lookup event
        trace["events"].append({
            "op": "CONCEPT_LOOKUP",
            "match": dict(match_criteria),
            "found": len(matching_concepts),
            "concept_ids": [c.id for c in matching_concepts[:5]],
        })
        
        # If concepts found, simulate execution
        for concept in matching_concepts[:3]:  # Execute top 3
            # Record lifecycle usage
            if state.lifecycle_manager:
                state.lifecycle_manager.record_usage(
                    concept.id,
                    context_id=str(trace["context_id"]),
                )
            
            # Record execution event
            ev = concept.evidence or {}
            pcc = ev.get("pcc_v2", {})
            depth = int(pcc.get("depth", 0) or 0)
            call_deps = list(pcc.get("call_deps", []))
            
            trace["events"].append({
                "op": "CSV_CALL",
                "concept_id": str(concept.id),
                "callee": str(concept.id),
                "depth": int(depth),
                "call_deps": call_deps,
                "bind": {},
                "out": f"out_{concept.id[:8]}",
            })
            trace["concepts_used"].append(str(concept.id))
        
        # Check if we got any output
        if matching_concepts:
            trace["outputs"] = {"result": "simulated_output"}
            trace["success"] = True
        else:
            trace["outputs"] = {}
            trace["success"] = False
    else:
        # Use provided executor
        result = executor(plan, task, concept_store)
        trace.update(dict(result) if result else {})
    
    # Record trace
    state.traces_accumulated.append(trace)
    
    return LoopPhase.VALIDATE, trace


def phase_validate(
    state: LoopState,
    execution_result: Dict[str, Any],
    config: AGILoopConfig,
) -> Tuple[LoopPhase, RegimeValidationResult]:
    """
    VALIDATE phase: Check survival laws against execution result.
    
    This is where the regime is enforced.
    """
    # Build regime config from loop config
    regime_config = AGIRegimeConfig(
        default_min_depth=config.min_concept_depth,
    )
    
    # Build task dict for validation
    task = {
        "concept_policy_required": True,
        "concept_min_depth": config.min_concept_depth,
        "concept_min_csv_calls": 1 if config.require_composition else 0,
    }
    
    # Build trace from execution result
    trace = {
        "concept_executor": {
            "used": bool(execution_result.get("concepts_used")),
            "ok": bool(execution_result.get("success", False)),
            "calls_total": len(execution_result.get("concepts_used", [])),
            "max_depth": int(execution_result.get("max_depth", 0) or 0),
        },
        "search_steps": int(execution_result.get("search_steps", 0) or 0),
    }
    
    # Validate against survival laws
    result = validate_survival_laws(
        trace=trace,
        task=task,
        config=regime_config,
    )
    
    # Compute loss
    loss = compute_regime_loss(
        validation_result=result,
        utility_pass_rate=1.0 if execution_result.get("success") else 0.0,
        fluency_score=1.0,
    )
    
    # Decide next phase
    if result.passed and math.isfinite(loss.get("loss", float("inf"))):
        return LoopPhase.MINE, result
    elif state.iterations and len([it for it in state.iterations if it.repairs_attempted > 0]) < config.max_repairs_per_iteration:
        return LoopPhase.REPAIR, result
    else:
        return LoopPhase.MINE, result  # Mine even on failure to learn


def phase_repair(
    state: LoopState,
    validation_result: RegimeValidationResult,
    task: Dict[str, Any],
    config: AGILoopConfig,
) -> Tuple[LoopPhase, Dict[str, Any]]:
    """
    REPAIR phase: Attempt to fix validation failures.
    
    Strategies:
    1. If LAW_CONCEPT failed: Force concept creation
    2. If LAW_DEPTH failed: Promote deeper concepts
    3. If LAW_COMPOSITION failed: Create composed concepts
    """
    state.total_repairs += 1
    
    repair_actions: List[Dict[str, Any]] = []
    
    for law_result in validation_result.results:
        if law_result.passed:
            continue
        
        if law_result.law == SurvivalLaw.LAW_CONCEPT:
            repair_actions.append({
                "action": "force_concept_creation",
                "reason": str(law_result.reason),
                "task_match": dict(task.get("match", {})),
            })
        
        elif law_result.law == SurvivalLaw.LAW_DEPTH:
            details = law_result.details or {}
            repair_actions.append({
                "action": "promote_deeper_concepts",
                "required_depth": int(details.get("required_depth", 1)),
                "current_depth": int(details.get("max_depth", 0)),
            })
        
        elif law_result.law == SurvivalLaw.LAW_COMPOSITION:
            repair_actions.append({
                "action": "create_composed_concept",
                "reason": str(law_result.reason),
            })
    
    repair_result: Dict[str, Any] = {
        "repair_actions": repair_actions,
        "repairs_count": len(repair_actions),
    }
    
    if repair_actions:
        return LoopPhase.PLAN, repair_result  # Re-plan with repairs
    else:
        return LoopPhase.FAILED, repair_result  # No repair possible


def phase_mine(
    state: LoopState,
    config: AGILoopConfig,
    step: int,
    store_content_hash: str,
) -> Tuple[LoopPhase, ComposedMiningResult]:
    """
    MINE phase: Extract patterns from accumulated traces.
    
    Mining happens WITHIN the loop, enabling concept promotion
    before re-running.
    """
    # Run composed concept mining
    result = run_composed_mining_pipeline(
        state.traces_accumulated,
        step=int(step),
        store_content_hash=str(store_content_hash),
        config=config.mining_config,
        max_promotions=config.max_promotions_per_iteration,
    )
    
    state.total_concepts_created += result.candidates_promoted
    
    return LoopPhase.PROMOTE, result


def phase_promote(
    state: LoopState,
    mining_result: ComposedMiningResult,
    concept_store: List[Act],
    config: AGILoopConfig,
) -> Tuple[LoopPhase, List[str]]:
    """
    PROMOTE phase: Add mined concepts to store and lifecycle manager.
    
    Only promotes concepts that pass all regime checks.
    """
    promoted_ids: List[str] = []
    
    for act in mining_result.promoted_acts:
        # Register with lifecycle manager
        if state.lifecycle_manager:
            ev = act.evidence or {}
            pcc = ev.get("pcc_v2", {})
            
            state.lifecycle_manager.register_concept(
                concept_id=str(act.id),
                depth=int(pcc.get("depth", 0) or 0),
                call_deps=list(pcc.get("call_deps", [])),
                meta={"promoted_in_iteration": state.current_iteration},
            )
        
        # Add to store
        concept_store.append(act)
        promoted_ids.append(str(act.id))
        state.total_concepts_promoted += 1
    
    # Decide next phase
    if promoted_ids and state.current_iteration < state.max_iterations:
        return LoopPhase.RERUN, promoted_ids
    elif state.current_iteration >= state.max_iterations:
        return LoopPhase.COMPLETE, promoted_ids
    else:
        return LoopPhase.COMPLETE, promoted_ids


def phase_rerun(
    state: LoopState,
    promoted_concepts: List[str],
    config: AGILoopConfig,
) -> Tuple[LoopPhase, Dict[str, Any]]:
    """
    RERUN phase: Prepare for next iteration with new concepts.
    
    This closes the loop by going back to PLAN with enriched store.
    """
    rerun_context: Dict[str, Any] = {
        "iteration": state.current_iteration + 1,
        "new_concepts": list(promoted_concepts),
        "total_concepts": state.total_concepts_promoted,
        "reason": "concepts_promoted",
    }
    
    state.current_iteration += 1
    
    return LoopPhase.PLAN, rerun_context


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop Engine
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AGILoopResult:
    """Final result of AGI loop execution."""
    success: bool
    exit_reason: LoopExitReason
    iterations_completed: int
    concepts_created: int
    concepts_promoted: int
    final_state: LoopState
    traces: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(AGI_LOOP_SCHEMA_VERSION_V144),
            "kind": "agi_loop_result_v144",
            "success": bool(self.success),
            "exit_reason": str(self.exit_reason),
            "iterations_completed": int(self.iterations_completed),
            "concepts_created": int(self.concepts_created),
            "concepts_promoted": int(self.concepts_promoted),
            "final_state": self.final_state.to_dict(),
            "trace_count": len(self.traces),
        }


def run_agi_loop(
    task: Dict[str, Any],
    *,
    concept_store: List[Act],
    config: Optional[AGILoopConfig] = None,
    step: int = 0,
    store_content_hash: str = "",
    executor: Optional[Callable[..., Dict[str, Any]]] = None,
) -> AGILoopResult:
    """
    Run the full AGI loop on a task.
    
    The loop enforces survival laws and guarantees that:
    1. Tasks cannot survive without concepts
    2. Concepts must be created if none exist
    3. Mining happens within the loop
    4. Promotion decisions are deterministic
    
    Returns final result with full audit trail.
    """
    cfg = config or AGILoopConfig()
    
    task_id = str(task.get("id", f"task_{sha256_hex(canonical_json_dumps(task).encode('utf-8'))[:8]}"))
    
    # Initialize state
    state = LoopState(
        task_id=task_id,
        current_phase=LoopPhase.PLAN,
        current_iteration=0,
        max_iterations=cfg.max_iterations,
        lifecycle_manager=ConceptLifecycleManager(),
    )
    
    # Initialize lifecycle with existing concepts
    for concept in concept_store:
        if concept.kind == "concept_csv" and concept.active:
            ev = concept.evidence or {}
            pcc = ev.get("pcc_v2", {})
            if state.lifecycle_manager:
                state.lifecycle_manager.register_concept(
                    concept_id=str(concept.id),
                    step=step,
                    has_pcc=bool(pcc),
                    pcc_hash=str(ev.get("pcc_hash", "")),
                )
    
    # Main loop
    current_data: Any = None
    
    while state.current_iteration < cfg.max_iterations:
        iteration = LoopIteration(
            iteration=state.current_iteration,
            phase=state.current_phase,
            task_id=task_id,
            timestamp=deterministic_iso(step=step + state.current_iteration),
        )
        
        # Phase dispatch
        if state.current_phase == LoopPhase.PLAN:
            next_phase, current_data = phase_plan(state, task, cfg)
            state.current_phase = next_phase
        
        elif state.current_phase == LoopPhase.EXECUTE:
            next_phase, current_data = phase_execute(
                state, current_data, task, concept_store, cfg, executor
            )
            iteration.execution_result = dict(current_data) if current_data else None
            state.current_phase = next_phase
        
        elif state.current_phase == LoopPhase.VALIDATE:
            next_phase, current_data = phase_validate(state, current_data, cfg)
            iteration.regime_result = current_data
            state.current_phase = next_phase
            
            # Check for success
            if current_data.passed:
                state.final_success = True
        
        elif state.current_phase == LoopPhase.REPAIR:
            iteration.repairs_attempted += 1
            next_phase, current_data = phase_repair(state, current_data, task, cfg)
            state.current_phase = next_phase
            
            if next_phase == LoopPhase.FAILED:
                state.exit_reason = LoopExitReason.NO_REPAIR_POSSIBLE
                break
        
        elif state.current_phase == LoopPhase.MINE:
            next_phase, current_data = phase_mine(state, cfg, step, store_content_hash)
            iteration.mining_result = current_data
            state.current_phase = next_phase
        
        elif state.current_phase == LoopPhase.PROMOTE:
            next_phase, promoted = phase_promote(state, current_data, concept_store, cfg)
            iteration.concepts_promoted = list(promoted)
            state.current_phase = next_phase
            
            if next_phase == LoopPhase.COMPLETE:
                break
        
        elif state.current_phase == LoopPhase.RERUN:
            next_phase, current_data = phase_rerun(state, current_data, cfg)
            state.current_phase = next_phase
            # Continue to next iteration
        
        elif state.current_phase == LoopPhase.COMPLETE:
            state.exit_reason = LoopExitReason.SUCCESS
            break
        
        elif state.current_phase == LoopPhase.FAILED:
            if not state.exit_reason:
                state.exit_reason = LoopExitReason.VALIDATION_FAILED
            break
        
        else:
            # Unknown phase
            state.exit_reason = LoopExitReason.EXTERNAL_ABORT
            break
        
        state.iterations.append(iteration)
    
    # Determine final exit reason
    if state.exit_reason is None:
        if state.current_iteration >= cfg.max_iterations:
            state.exit_reason = LoopExitReason.MAX_ITERATIONS
        elif state.final_success:
            state.exit_reason = LoopExitReason.SUCCESS
        else:
            state.exit_reason = LoopExitReason.VALIDATION_FAILED
    
    return AGILoopResult(
        success=state.final_success,
        exit_reason=state.exit_reason,
        iterations_completed=state.current_iteration + 1,
        concepts_created=state.total_concepts_created,
        concepts_promoted=state.total_concepts_promoted,
        final_state=state,
        traces=list(state.traces_accumulated),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch Processing
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BatchLoopResult:
    """Result of batch AGI loop execution."""
    tasks_processed: int
    tasks_succeeded: int
    tasks_failed: int
    total_concepts_promoted: int
    individual_results: List[AGILoopResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(AGI_LOOP_SCHEMA_VERSION_V144),
            "kind": "batch_loop_result_v144",
            "tasks_processed": int(self.tasks_processed),
            "tasks_succeeded": int(self.tasks_succeeded),
            "tasks_failed": int(self.tasks_failed),
            "total_concepts_promoted": int(self.total_concepts_promoted),
            "success_rate": float(self.tasks_succeeded / max(1, self.tasks_processed)),
            "results_count": len(self.individual_results),
        }


def run_agi_loop_batch(
    tasks: Sequence[Dict[str, Any]],
    *,
    concept_store: List[Act],
    config: Optional[AGILoopConfig] = None,
    step: int = 0,
    store_content_hash: str = "",
    executor: Optional[Callable[..., Dict[str, Any]]] = None,
) -> BatchLoopResult:
    """
    Run AGI loop on a batch of tasks.
    
    Concepts promoted in earlier tasks are available to later tasks,
    enabling transfer learning within the batch.
    """
    cfg = config or AGILoopConfig()
    
    results: List[AGILoopResult] = []
    succeeded = 0
    total_promoted = 0
    
    for i, task in enumerate(tasks):
        result = run_agi_loop(
            task,
            concept_store=concept_store,  # Shared, mutable store
            config=cfg,
            step=step + i * cfg.max_iterations,
            store_content_hash=store_content_hash,
            executor=executor,
        )
        results.append(result)
        
        if result.success:
            succeeded += 1
        total_promoted += result.concepts_promoted
    
    return BatchLoopResult(
        tasks_processed=len(tasks),
        tasks_succeeded=succeeded,
        tasks_failed=len(tasks) - succeeded,
        total_concepts_promoted=total_promoted,
        individual_results=results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LEDGER Integration
# ─────────────────────────────────────────────────────────────────────────────


def write_loop_result_to_ledger(
    result: AGILoopResult,
    *,
    step: int,
    store_content_hash: str,
) -> Dict[str, Any]:
    """
    Create WORM-compliant ledger entry for loop result.
    """
    entry: Dict[str, Any] = {
        "schema_version": int(AGI_LOOP_SCHEMA_VERSION_V144),
        "kind": "agi_loop_ledger_entry_v144",
        "timestamp": deterministic_iso(step=int(step)),
        "step": int(step),
        "store_content_hash": str(store_content_hash),
        "result": result.to_dict(),
    }
    
    # Compute entry hash
    entry_json = canonical_json_dumps(entry)
    entry["entry_hash"] = sha256_hex(entry_json.encode("utf-8"))
    
    return entry
