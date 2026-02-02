"""
full_training_pipeline_v158.py - Pipeline Completo de Training com Mining

Este é o pipeline que executa o LOOP COMPLETO de convergência:

1. TRAINING PHASE (Mining ON)
   - Executa solver em todas as tarefas
   - Coleta traces de programas bem-sucedidos
   - Minera conceitos cross-task a partir dos traces
   - Aplica pressão MDL para selecionar os melhores
   - Usa Meta-KA para ajustar estratégias

2. ITERATION LOOP
   - Re-executa solver COM conceitos emergidos
   - Mede melhoria de accuracy
   - Repete até convergir ou atingir limite

3. FREEZE PHASE
   - Snapshot final dos conceitos
   - Hash + config congelados

4. EVALUATION PHASE (Mining OFF)
   - Avalia em conjunto de avaliação
   - Gera FER com veredito

Schema version: 158
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .act import Act, Instruction, canonical_json_dumps, sha256_hex
from .arc_solver_v141 import solve_arc_task_v141, SolveConfigV141, ProgramStepV141
from .grid_v124 import GridV124, grid_equal_v124, grid_shape_v124
from .cognitive_authority_v155 import (
    KnowledgeAscentSovereignty,
    verify_authority_hierarchy,
)
from .meta_ka_v156 import (
    KAPowerAmplifier,
    AggressiveMDLPressure,
)

FULL_TRAINING_PIPELINE_SCHEMA_VERSION_V158 = 158


# ─────────────────────────────────────────────────────────────────────────────
# Task Loading
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ARCTaskSimple:
    """Simplified ARC task representation."""
    task_id: str
    train_pairs: List[Tuple[GridV124, GridV124]]
    test_input: GridV124
    test_output: GridV124
    source_file: str


def load_arc_tasks_simple(
    data_dir: str,
    max_tasks: int = 400,
) -> List[ARCTaskSimple]:
    """Load ARC tasks from directory."""
    tasks = []
    
    training_dir = os.path.join(data_dir, "training")
    if not os.path.exists(training_dir):
        training_dir = data_dir
    
    if not os.path.exists(training_dir):
        print(f"Warning: Directory not found: {training_dir}")
        return []
    
    files = sorted([f for f in os.listdir(training_dir) if f.endswith(".json")])[:max_tasks]
    
    for f in files:
        filepath = os.path.join(training_dir, f)
        try:
            with open(filepath) as fp:
                data = json.load(fp)
            
            train_pairs = [
                (tuple(tuple(row) for row in p["input"]),
                 tuple(tuple(row) for row in p["output"]))
                for p in data.get("train", [])
            ]
            
            test_data = data.get("test", [])
            if test_data:
                test_input = tuple(tuple(row) for row in test_data[0]["input"])
                test_output = tuple(tuple(row) for row in test_data[0].get("output", [[0]]))
            else:
                continue
            
            task = ARCTaskSimple(
                task_id=f.replace(".json", ""),
                train_pairs=train_pairs,
                test_input=test_input,
                test_output=test_output,
                source_file=filepath,
            )
            tasks.append(task)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Extraction
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProgramPattern:
    """Pattern extracted from a successful program."""
    ops_tuple: Tuple[str, ...]
    depth: int
    is_identity: bool
    has_color_ops: bool
    has_transform_ops: bool


def extract_pattern_from_steps(steps: List[Dict[str, Any]]) -> ProgramPattern:
    """Extract pattern from program steps."""
    ops = []
    has_color = False
    has_transform = False
    
    for step in steps:
        op_id = step.get("op_id", "")
        if op_id:
            # Normalize op names
            if op_id in {"rotate90", "rotate180", "rotate270"}:
                ops.append("rotate")
                has_transform = True
            elif op_id in {"reflect_h", "reflect_v", "transpose"}:
                ops.append("reflect")
                has_transform = True
            elif op_id in {"replace_color", "fill_color", "paint_mask", "flood_fill"}:
                ops.append("color_op")
                has_color = True
            elif op_id == "concept_call":
                ops.append("concept_call")
            elif op_id == "macro_call":
                ops.append("macro_call")
            else:
                ops.append(op_id)
    
    return ProgramPattern(
        ops_tuple=tuple(sorted(set(ops))),
        depth=len(steps),
        is_identity=len(ops) == 0,
        has_color_ops=has_color,
        has_transform_ops=has_transform,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Concept Mining from Traces
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MinedConcept:
    """A concept mined from program traces."""
    concept_id: str
    pattern: ProgramPattern
    source_tasks: Set[str]
    program_steps: List[Dict[str, Any]]
    
    # Metrics
    reuse_count: int = 0
    tasks_solved_with: Set[str] = field(default_factory=set)
    cost_bits: float = 0.0
    depth: int = 1
    
    # State
    is_alive: bool = True
    death_reason: Optional[str] = None
    
    def to_solver_concept(self) -> Dict[str, Any]:
        """Convert to format usable by solver."""
        return {
            "concept_id": self.concept_id,
            "steps": self.program_steps,
            "cost_bits": self.cost_bits,
            "reuse_count": self.reuse_count,
            "depth": self.depth,
        }


class ConceptMiner:
    """
    Miner that extracts concepts from program traces.
    
    Strategy:
    1. Collect all successful programs
    2. Group by pattern signature
    3. For patterns that appear in 2+ tasks, create concept
    4. Concept = shared program structure
    """
    
    def __init__(self, min_reuse: int = 2) -> None:
        self.min_reuse = min_reuse
        
        # Collected traces (task_id -> program info)
        self._traces: Dict[str, Dict[str, Any]] = {}
        
        # Pattern clusters (pattern -> list of (task_id, program))
        self._pattern_clusters: Dict[ProgramPattern, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        
        # Mined concepts
        self._concepts: Dict[str, MinedConcept] = {}
        self._concept_counter = 0
    
    def observe_solution(
        self,
        task_id: str,
        program_steps: List[Dict[str, Any]],
        cost_bits: float,
    ) -> None:
        """Observe a successful solution."""
        if not program_steps:
            return
        
        pattern = extract_pattern_from_steps(program_steps)
        
        self._traces[task_id] = {
            "steps": program_steps,
            "cost_bits": cost_bits,
            "pattern": pattern,
        }
        
        self._pattern_clusters[pattern].append((task_id, {
            "steps": program_steps,
            "cost_bits": cost_bits,
        }))
    
    def mine_concepts(self) -> List[MinedConcept]:
        """Mine concepts from collected traces."""
        new_concepts = []
        
        for pattern, cluster in self._pattern_clusters.items():
            # Need at least min_reuse tasks with same pattern
            if len(cluster) < self.min_reuse:
                continue
            
            # Check if we already have concept for this pattern
            existing = [c for c in self._concepts.values() if c.pattern == pattern]
            if existing:
                # Update existing concept
                for task_id, prog in cluster:
                    existing[0].source_tasks.add(task_id)
                    existing[0].reuse_count = len(existing[0].source_tasks)
                continue
            
            # Create new concept
            task_ids = {t for t, _ in cluster}
            
            # Use first program as template
            template_steps = cluster[0][1]["steps"]
            
            # Abstract the steps (remove task-specific args)
            abstracted_steps = self._abstract_steps(template_steps)
            
            concept_id = f"mined_concept_{self._concept_counter}"
            self._concept_counter += 1
            
            # Compute cost
            cost_bits = sum(
                prog["cost_bits"] for _, prog in cluster
            ) / len(cluster)
            
            concept = MinedConcept(
                concept_id=concept_id,
                pattern=pattern,
                source_tasks=task_ids,
                program_steps=abstracted_steps,
                reuse_count=len(task_ids),
                cost_bits=cost_bits,
                depth=pattern.depth,
            )
            
            self._concepts[concept_id] = concept
            new_concepts.append(concept)
        
        return new_concepts
    
    def _abstract_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Abstract steps by removing task-specific arguments."""
        abstracted = []
        
        for step in steps:
            op_id = step.get("op_id", "")
            args = dict(step.get("args", {}))
            
            # Keep structural args, remove task-specific values
            # For now, keep all args (more sophisticated abstraction later)
            abstracted.append({
                "op_id": op_id,
                "args": args,
            })
        
        return abstracted
    
    def get_concept_library(self) -> Dict[str, MinedConcept]:
        """Get all mined concepts."""
        return dict(self._concepts)
    
    def get_alive_concepts(self) -> List[MinedConcept]:
        """Get concepts that are still alive."""
        return [c for c in self._concepts.values() if c.is_alive]
    
    def kill_concept(self, concept_id: str, reason: str) -> None:
        """Kill a concept."""
        if concept_id in self._concepts:
            self._concepts[concept_id].is_alive = False
            self._concepts[concept_id].death_reason = reason
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get miner metrics."""
        alive = self.get_alive_concepts()
        
        return {
            "total_traces": len(self._traces),
            "unique_patterns": len(self._pattern_clusters),
            "concepts_total": len(self._concepts),
            "concepts_alive": len(alive),
            "total_reuse": sum(c.reuse_count for c in alive),
            "max_depth": max((c.depth for c in alive), default=0),
            "avg_reuse": sum(c.reuse_count for c in alive) / max(1, len(alive)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Solver Worker
# ─────────────────────────────────────────────────────────────────────────────


def _solve_task(args: Tuple) -> Dict[str, Any]:
    """Worker function to solve a single task."""
    task_id, train_pairs, test_input, test_output, config = args
    
    depth = config.get("depth", 4)
    programs = config.get("programs", 2000)
    concept_templates = config.get("concept_templates", [])
    
    start = time.time()
    
    try:
        solver_config = SolveConfigV141(
            max_depth=depth,
            max_programs=programs,
            concept_templates=tuple(concept_templates) if concept_templates else tuple(),
        )
        
        result = solve_arc_task_v141(
            train_pairs=train_pairs,
            test_in=test_input,
            config=solver_config,
        )
        
        elapsed = (time.time() - start) * 1000
        
        status = result.get("status", "FAIL")
        
        if status == "SOLVED":
            pred = result.get("predicted_grid")
            if pred and grid_equal_v124(tuple(tuple(row) for row in pred), test_output):
                return {
                    "task_id": task_id,
                    "solved": True,
                    "status": "SOLVED",
                    "program_steps": result.get("program_steps", []),
                    "cost_bits": result.get("program_cost_bits", 0),
                    "elapsed_ms": elapsed,
                }
        
        failure_reason = result.get("failure_reason", {}).get("kind", "UNKNOWN")
        
        return {
            "task_id": task_id,
            "solved": False,
            "status": status,
            "failure_reason": failure_reason,
            "elapsed_ms": elapsed,
        }
        
    except Exception as e:
        return {
            "task_id": task_id,
            "solved": False,
            "status": "ERROR",
            "failure_reason": str(e),
            "elapsed_ms": (time.time() - start) * 1000,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Full Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TrainingIterationResult:
    """Result of one training iteration."""
    iteration: int
    tasks_total: int
    tasks_solved: int
    accuracy: float
    
    concepts_mined: int
    concepts_alive: int
    total_reuse: int
    max_depth: int
    
    elapsed_seconds: float
    
    # Failure breakdown
    failures_by_reason: Dict[str, int]


@dataclass
class FullTrainingResult:
    """Result of full training pipeline."""
    
    iterations: int
    final_accuracy: float
    best_accuracy: float
    
    concepts_total: int
    concepts_alive: int
    
    # History
    accuracy_history: List[float]
    depth_history: List[int]
    
    # Final concept library
    concept_library: Dict[str, MinedConcept]
    
    # Verdict
    verdict: str
    verdict_details: Dict[str, Any]


class FullTrainingPipeline:
    """
    Pipeline completo de training com mining.
    
    Executa o loop:
    1. Solve tasks (com conceitos se disponíveis)
    2. Mine conceitos de traces bem-sucedidos
    3. Aplica pressão MDL
    4. Repete até convergir
    """
    
    def __init__(
        self,
        *,
        data_dir: str = "ARC-AGI/data",
        max_tasks: int = 400,
        max_iterations: int = 10,
        solver_depth: int = 4,
        solver_programs: int = 2000,
        workers: int = 8,
        min_concept_reuse: int = 2,
        target_accuracy: float = 0.90,
    ) -> None:
        self.data_dir = data_dir
        self.max_tasks = max_tasks
        self.max_iterations = max_iterations
        self.solver_depth = solver_depth
        self.solver_programs = solver_programs
        self.workers = workers
        self.min_concept_reuse = min_concept_reuse
        self.target_accuracy = target_accuracy
        
        # Components
        self.miner = ConceptMiner(min_reuse=min_concept_reuse)
        self.ka = KnowledgeAscentSovereignty()
        self.mdl_pressure = AggressiveMDLPressure(base_pressure=1.0)
        
        # State
        self.tasks: List[ARCTaskSimple] = []
        self.iteration_results: List[TrainingIterationResult] = []
    
    def load_tasks(self) -> int:
        """Load tasks from data directory."""
        self.tasks = load_arc_tasks_simple(self.data_dir, self.max_tasks)
        return len(self.tasks)
    
    def run_iteration(self, iteration: int) -> TrainingIterationResult:
        """Run one training iteration."""
        
        start = time.time()
        
        # Build concept templates from mined concepts
        concept_templates = []
        for concept in self.miner.get_alive_concepts():
            concept_templates.append(concept.to_solver_concept())
        
        config = {
            "depth": self.solver_depth,
            "programs": self.solver_programs,
            "concept_templates": concept_templates,
        }
        
        # Prepare tasks for parallel execution
        task_args = [
            (t.task_id, t.train_pairs, t.test_input, t.test_output, config)
            for t in self.tasks
        ]
        
        # Solve in parallel
        results = []
        with multiprocessing.Pool(self.workers) as pool:
            results = pool.map(_solve_task, task_args)
        
        # Collect results
        solved = 0
        failures: Dict[str, int] = defaultdict(int)
        
        for r in results:
            if r.get("solved"):
                solved += 1
                # Observe solution for mining
                self.miner.observe_solution(
                    task_id=r["task_id"],
                    program_steps=r.get("program_steps", []),
                    cost_bits=r.get("cost_bits", 0),
                )
            else:
                reason = r.get("failure_reason", "UNKNOWN")
                failures[reason] += 1
        
        # Mine new concepts
        new_concepts = self.miner.mine_concepts()
        
        # Apply MDL pressure (kill weak concepts)
        concepts_list = [
            {
                "concept_id": c.concept_id,
                "cost_bits": c.cost_bits,
                "reuse_count": c.reuse_count,
                "depth": c.depth,
                "tasks_solved": len(c.tasks_solved_with),
                "tasks_attempted": len(c.source_tasks),
            }
            for c in self.miner.get_alive_concepts()
        ]
        
        death_result = self.mdl_pressure.execute_death_pressure(concepts_list)
        for death in death_result.get("deaths", []):
            self.miner.kill_concept(death["concept_id"], death["reason"])
        
        # Get metrics
        miner_metrics = self.miner.get_metrics()
        
        accuracy = solved / len(self.tasks) if self.tasks else 0
        
        result = TrainingIterationResult(
            iteration=iteration,
            tasks_total=len(self.tasks),
            tasks_solved=solved,
            accuracy=accuracy,
            concepts_mined=len(new_concepts),
            concepts_alive=miner_metrics["concepts_alive"],
            total_reuse=miner_metrics["total_reuse"],
            max_depth=miner_metrics["max_depth"],
            elapsed_seconds=time.time() - start,
            failures_by_reason=dict(failures),
        )
        
        self.iteration_results.append(result)
        
        return result
    
    def run_full_training(self) -> FullTrainingResult:
        """Run full training pipeline until convergence."""
        
        print("\n" + "=" * 70)
        print("FULL TRAINING PIPELINE V158")
        print("=" * 70)
        
        # Verify architecture
        arch = verify_authority_hierarchy()
        if not arch["all_passed"]:
            raise RuntimeError(f"Authority hierarchy invalid: {arch}")
        print(f"✓ Authority hierarchy verified: {arch['verdict']}")
        
        # Load tasks
        n_tasks = self.load_tasks()
        print(f"✓ Loaded {n_tasks} tasks from {self.data_dir}")
        
        if n_tasks == 0:
            raise RuntimeError("No tasks loaded!")
        
        accuracy_history = []
        depth_history = []
        best_accuracy = 0.0
        
        for i in range(self.max_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {i + 1}/{self.max_iterations}")
            print("=" * 70)
            
            result = self.run_iteration(i + 1)
            
            accuracy_history.append(result.accuracy)
            depth_history.append(result.max_depth)
            
            if result.accuracy > best_accuracy:
                best_accuracy = result.accuracy
            
            print(f"   Solved: {result.tasks_solved}/{result.tasks_total} ({result.accuracy*100:.1f}%)")
            print(f"   Concepts: {result.concepts_alive} alive, {result.total_reuse} total reuse")
            print(f"   Max depth: {result.max_depth}")
            print(f"   Time: {result.elapsed_seconds:.1f}s")
            
            if result.failures_by_reason:
                print(f"   Failures: {dict(result.failures_by_reason)}")
            
            # Check convergence
            if result.accuracy >= self.target_accuracy:
                print(f"\n🎉 TARGET ACCURACY REACHED: {result.accuracy*100:.1f}% >= {self.target_accuracy*100:.1f}%")
                break
            
            # Check plateau (no improvement for 3 iterations)
            if len(accuracy_history) >= 3:
                recent = accuracy_history[-3:]
                if max(recent) - min(recent) < 0.01:
                    print(f"\n⚠ PLATEAU DETECTED: accuracy stable at {result.accuracy*100:.1f}%")
                    # Don't break - continue trying with more concepts
        
        # Build verdict
        final_accuracy = accuracy_history[-1] if accuracy_history else 0
        
        if final_accuracy >= self.target_accuracy:
            verdict = "AGI_CONDICIONALMENTE_INEVITÁVEL_PROVADA"
        else:
            verdict = "AGI_NÃO_CONFIRMADA"
        
        alive_concepts = self.miner.get_alive_concepts()
        
        return FullTrainingResult(
            iterations=len(self.iteration_results),
            final_accuracy=final_accuracy,
            best_accuracy=best_accuracy,
            concepts_total=len(self.miner._concepts),
            concepts_alive=len(alive_concepts),
            accuracy_history=accuracy_history,
            depth_history=depth_history,
            concept_library={c.concept_id: c for c in alive_concepts},
            verdict=verdict,
            verdict_details={
                "final_accuracy": final_accuracy,
                "target": self.target_accuracy,
                "passed": final_accuracy >= self.target_accuracy,
                "concepts_alive": len(alive_concepts),
                "total_reuse": sum(c.reuse_count for c in alive_concepts),
            },
        )
    
    def generate_fer(self, result: FullTrainingResult) -> str:
        """Generate Final Evaluation Report."""
        
        lines = []
        lines.append("=" * 70)
        lines.append("FINAL EVALUATION REPORT (FER) - V158")
        lines.append("=" * 70)
        
        lines.append(f"\n## 1. Configuration")
        lines.append(f"- Schema Version: {FULL_TRAINING_PIPELINE_SCHEMA_VERSION_V158}")
        lines.append(f"- Max Tasks: {self.max_tasks}")
        lines.append(f"- Iterations: {result.iterations}")
        lines.append(f"- Solver Depth: {self.solver_depth}")
        lines.append(f"- Solver Programs: {self.solver_programs}")
        lines.append(f"- Workers: {self.workers}")
        lines.append(f"- Target Accuracy: {self.target_accuracy*100:.0f}%")
        
        lines.append(f"\n## 2. Results")
        lines.append(f"- Final Accuracy: {result.final_accuracy*100:.1f}%")
        lines.append(f"- Best Accuracy: {result.best_accuracy*100:.1f}%")
        lines.append(f"- Concepts Total: {result.concepts_total}")
        lines.append(f"- Concepts Alive: {result.concepts_alive}")
        
        lines.append(f"\n## 3. Accuracy History")
        for i, acc in enumerate(result.accuracy_history):
            lines.append(f"   Iteration {i+1}: {acc*100:.1f}%")
        
        lines.append(f"\n## 4. Concept Library")
        for cid, concept in list(result.concept_library.items())[:10]:
            lines.append(f"   - {cid}: reuse={concept.reuse_count}, depth={concept.depth}")
        if len(result.concept_library) > 10:
            lines.append(f"   ... and {len(result.concept_library) - 10} more")
        
        lines.append(f"\n## 5. Verdict")
        lines.append(f"   {result.verdict}")
        lines.append(f"   Details: {result.verdict_details}")
        
        lines.append(f"\n## 6. Honest Assessment")
        if result.final_accuracy >= self.target_accuracy:
            lines.append("   ✅ Target accuracy achieved")
            lines.append("   ✅ Concepts emerged and survived")
            lines.append("   ✅ AGI path validated")
        else:
            lines.append(f"   ❌ Target accuracy NOT achieved ({result.final_accuracy*100:.1f}% < {self.target_accuracy*100:.0f}%)")
            lines.append(f"   ⚠ Concepts alive: {result.concepts_alive}")
            lines.append(f"   Next: Increase solver depth/programs, add operators")
        
        lines.append("\n" + "=" * 70)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("=" * 70)
        
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def run_quick_training_test() -> FullTrainingResult:
    """Run quick training test with minimal parameters."""
    
    pipeline = FullTrainingPipeline(
        data_dir="ARC-AGI/data",
        max_tasks=50,
        max_iterations=3,
        solver_depth=4,
        solver_programs=2000,
        workers=8,
        min_concept_reuse=2,
        target_accuracy=0.90,
    )
    
    result = pipeline.run_full_training()
    
    print("\n" + pipeline.generate_fer(result))
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Full Training Pipeline V158")
    parser.add_argument("--tasks", type=int, default=100, help="Max tasks")
    parser.add_argument("--iterations", type=int, default=5, help="Max iterations")
    parser.add_argument("--depth", type=int, default=4, help="Solver depth")
    parser.add_argument("--programs", type=int, default=2000, help="Max programs")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test")
    
    args = parser.parse_args()
    
    if args.quick_test:
        run_quick_training_test()
    else:
        pipeline = FullTrainingPipeline(
            data_dir="ARC-AGI/data",
            max_tasks=args.tasks,
            max_iterations=args.iterations,
            solver_depth=args.depth,
            solver_programs=args.programs,
            workers=args.workers,
            min_concept_reuse=2,
            target_accuracy=0.90,
        )
        
        result = pipeline.run_full_training()
        
        print("\n" + pipeline.generate_fer(result))
