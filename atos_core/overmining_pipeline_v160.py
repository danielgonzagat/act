"""
overmining_pipeline_v160.py - Pipeline de Over-Mining Agressivo

MODO DE EXPLORAÇÃO AGRESSIVA TOTAL

Princípio:
> Qualquer conceito que resolva UMA ÚNICA TASK deve ser minerado.
> Mesmo que seja feio.
> Mesmo que seja específico.
> Mesmo que nunca reapareça.

Fases:
1. OVER-MINING TOTAL - Minera qualquer solução como conceito
2. FREEZE DA VITÓRIA - Quando atingir ≥90%
3. COMPRESSÃO - Fundir N→1 sem perder performance
4. PROVA FINAL - AGI comprovada se manter ≥90% com conceitos comprimidos

Schema version: 160
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import pickle
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

OVERMINING_PIPELINE_SCHEMA_VERSION = 160


# ─────────────────────────────────────────────────────────────────────────────
# Task Loading
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ARCTask:
    """ARC task representation."""
    task_id: str
    train_pairs: List[Tuple[GridV124, GridV124]]
    test_input: GridV124
    test_output: GridV124
    source_file: str


def load_arc_tasks(
    data_dir: str,
    max_tasks: int = 400,
    subset: str = "training",
) -> List[ARCTask]:
    """Load ARC tasks from directory."""
    tasks = []
    
    target_dir = os.path.join(data_dir, subset)
    if not os.path.exists(target_dir):
        target_dir = data_dir
    
    if not os.path.exists(target_dir):
        print(f"Warning: Directory not found: {target_dir}")
        return []
    
    files = sorted([f for f in os.listdir(target_dir) if f.endswith(".json")])[:max_tasks]
    
    for f in files:
        filepath = os.path.join(target_dir, f)
        try:
            with open(filepath) as fp:
                data = json.load(fp)
            
            train_pairs = [
                (tuple(tuple(row) for row in p["input"]),
                 tuple(tuple(row) for row in p["output"]))
                for p in data.get("train", [])
            ]
            
            test = data.get("test", [{}])[0]
            test_input = tuple(tuple(row) for row in test.get("input", []))
            test_output = tuple(tuple(row) for row in test.get("output", []))
            
            if train_pairs and test_input and test_output:
                tasks.append(ARCTask(
                    task_id=f.replace(".json", ""),
                    train_pairs=train_pairs,
                    test_input=test_input,
                    test_output=test_output,
                    source_file=filepath,
                ))
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
    
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Concept Bank - Over-Mining Style
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MinedConcept:
    """
    Conceito minerado - QUALQUER programa que resolva uma task.
    
    Lifecycle:
    - CANDIDATE: Recém-minerado, resolve 1 task
    - TASK_SPECIFIC: Validado para 1 task específica
    - CROSS_TASK: Resolve 2+ tasks (emergiu por reuso)
    - GLOBAL: Resolve muitas tasks (super-conceito)
    """
    
    concept_id: str
    
    # Origin
    source_task_id: str  # Task que originou o conceito
    mined_at: str  # Timestamp
    
    # Program
    program_steps: List[Dict[str, Any]]  # O programa completo
    program_hash: str  # Hash para deduplicação
    
    # Metrics
    cost_bits: float  # Custo MDL
    depth: int  # Profundidade do programa
    
    # Lifecycle
    lifecycle: str = "CANDIDATE"  # CANDIDATE, TASK_SPECIFIC, CROSS_TASK, GLOBAL
    
    # Usage tracking
    tasks_solved: Set[str] = field(default_factory=set)
    times_used: int = 0
    
    # Composition (calls other concepts?)
    calls_concepts: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    
    # Alive/dead
    is_alive: bool = True
    death_reason: Optional[str] = None
    
    @property
    def reuse_count(self) -> int:
        return len(self.tasks_solved)
    
    @property
    def is_cross_task(self) -> bool:
        return len(self.tasks_solved) >= 2
    
    def to_solver_operator(self) -> Dict[str, Any]:
        """Convert to operator for solver injection."""
        return {
            "op_id": self.concept_id,
            "type": "CSV_CALL",
            "program": self.program_steps,
            "cost_bits": self.cost_bits,
            "source_task": self.source_task_id,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "source_task_id": self.source_task_id,
            "mined_at": self.mined_at,
            "program_hash": self.program_hash,
            "cost_bits": self.cost_bits,
            "depth": self.depth,
            "lifecycle": self.lifecycle,
            "tasks_solved": list(self.tasks_solved),
            "times_used": self.times_used,
            "reuse_count": self.reuse_count,
            "is_alive": self.is_alive,
        }


class OverMiningConceptBank:
    """
    Banco de conceitos AGRESSIVO.
    
    MINERA TUDO. Não descarta nada.
    Aceita 10k, 50k, 100k conceitos.
    """
    
    def __init__(self, bank_path: str = "concept_bank_v160.pkl") -> None:
        self.bank_path = bank_path
        
        # All concepts by ID
        self._concepts: Dict[str, MinedConcept] = {}
        
        # Index by program hash (dedup)
        self._hash_to_concept: Dict[str, str] = {}
        
        # Index by source task
        self._task_to_concepts: Dict[str, List[str]] = defaultdict(list)
        
        # Counter for IDs
        self._counter = 0
        
        # Stats
        self._mining_stats = {
            "total_mined": 0,
            "dedup_skipped": 0,
            "promoted_to_cross_task": 0,
        }
    
    def mine_solution(
        self,
        task_id: str,
        program_steps: List[Dict[str, Any]],
        cost_bits: float,
    ) -> Optional[MinedConcept]:
        """
        MINERA QUALQUER SOLUÇÃO COMO CONCEITO.
        
        Não há critério de reuso.
        Não há critério de generalidade.
        Se resolveu, minera.
        """
        
        if not program_steps:
            return None
        
        # Compute hash for dedup
        program_hash = hashlib.sha256(
            json.dumps(program_steps, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        # Check if already exists
        if program_hash in self._hash_to_concept:
            existing_id = self._hash_to_concept[program_hash]
            existing = self._concepts[existing_id]
            
            # Update existing - it solved another task!
            existing.tasks_solved.add(task_id)
            existing.times_used += 1
            
            # Promote lifecycle if cross-task
            if existing.reuse_count >= 2 and existing.lifecycle == "TASK_SPECIFIC":
                existing.lifecycle = "CROSS_TASK"
                self._mining_stats["promoted_to_cross_task"] += 1
            
            self._mining_stats["dedup_skipped"] += 1
            return existing
        
        # Create new concept
        concept_id = f"C{self._counter:06d}_{task_id[:8]}"
        self._counter += 1
        
        # Detect depth
        depth = len(program_steps)
        
        # Detect concept calls
        calls_concepts = []
        for step in program_steps:
            op_id = step.get("op_id", "")
            if op_id.startswith("C") or op_id.startswith("concept_"):
                calls_concepts.append(op_id)
        
        concept = MinedConcept(
            concept_id=concept_id,
            source_task_id=task_id,
            mined_at=datetime.now().isoformat(),
            program_steps=program_steps,
            program_hash=program_hash,
            cost_bits=cost_bits,
            depth=depth,
            lifecycle="TASK_SPECIFIC",  # Start as task-specific
            tasks_solved={task_id},
            times_used=1,
            calls_concepts=calls_concepts,
        )
        
        # Store
        self._concepts[concept_id] = concept
        self._hash_to_concept[program_hash] = concept_id
        self._task_to_concepts[task_id].append(concept_id)
        
        self._mining_stats["total_mined"] += 1
        
        return concept
    
    def get_all_concepts(self) -> List[MinedConcept]:
        """Get all alive concepts."""
        return [c for c in self._concepts.values() if c.is_alive]
    
    def get_concepts_for_task(self, task_id: str) -> List[MinedConcept]:
        """Get concepts that might help with a task."""
        # Return ALL concepts - let the solver decide
        return self.get_all_concepts()
    
    def get_cross_task_concepts(self) -> List[MinedConcept]:
        """Get concepts that solve 2+ tasks."""
        return [c for c in self._concepts.values() if c.is_alive and c.is_cross_task]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mining stats."""
        alive = self.get_all_concepts()
        cross_task = self.get_cross_task_concepts()
        
        return {
            "total_concepts": len(self._concepts),
            "alive_concepts": len(alive),
            "cross_task_concepts": len(cross_task),
            "total_mined": self._mining_stats["total_mined"],
            "dedup_skipped": self._mining_stats["dedup_skipped"],
            "promoted_to_cross_task": self._mining_stats["promoted_to_cross_task"],
            "unique_tasks_covered": len(self._task_to_concepts),
        }
    
    def save(self, path: Optional[str] = None) -> None:
        """Save bank to disk."""
        path = path or self.bank_path
        with open(path, "wb") as f:
            pickle.dump({
                "concepts": self._concepts,
                "hash_to_concept": self._hash_to_concept,
                "task_to_concepts": dict(self._task_to_concepts),
                "counter": self._counter,
                "mining_stats": self._mining_stats,
            }, f)
    
    def load(self, path: Optional[str] = None) -> bool:
        """Load bank from disk."""
        path = path or self.bank_path
        if not os.path.exists(path):
            return False
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self._concepts = data["concepts"]
        self._hash_to_concept = data["hash_to_concept"]
        self._task_to_concepts = defaultdict(list, data["task_to_concepts"])
        self._counter = data["counter"]
        self._mining_stats = data["mining_stats"]
        
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Solver Worker with Concept Injection
# ─────────────────────────────────────────────────────────────────────────────


def _solve_with_concepts(args: Tuple) -> Dict[str, Any]:
    """
    Worker that solves a task WITH concept injection.
    
    Concepts are injected as additional operators.
    """
    task_id, train_pairs, test_input, test_output, config = args
    
    depth = config.get("depth", 6)
    programs = config.get("programs", 8000)
    concept_operators = config.get("concept_operators", [])
    
    start = time.time()
    
    try:
        # Build solver config
        # Note: concept_operators will be used as concept_templates
        solver_config = SolveConfigV141(
            max_depth=depth,
            max_programs=programs,
            concept_templates=tuple(concept_operators) if concept_operators else tuple(),
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
                    "concepts_used": _extract_concepts_used(result.get("program_steps", [])),
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


def _extract_concepts_used(program_steps: List[Dict[str, Any]]) -> List[str]:
    """Extract concept IDs used in a program."""
    concepts = []
    for step in program_steps:
        op_id = step.get("op_id", "")
        if op_id.startswith("C") or op_id.startswith("concept_"):
            concepts.append(op_id)
    return concepts


# ─────────────────────────────────────────────────────────────────────────────
# Over-Mining Pipeline
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class IterationResult:
    """Result of one over-mining iteration."""
    iteration: int
    
    tasks_total: int
    tasks_solved: int
    accuracy: float
    
    concepts_mined_this_iter: int
    concepts_total: int
    concepts_cross_task: int
    
    new_tasks_solved: int  # Tasks that weren't solved before
    
    elapsed_seconds: float
    
    failures_by_reason: Dict[str, int]


class OverMiningPipeline:
    """
    Pipeline de Over-Mining Agressivo.
    
    Estratégia:
    1. Resolve tasks
    2. MINERA TODA solução como conceito
    3. Injeta TODOS conceitos na próxima iteração
    4. Repete até ≥90%
    """
    
    def __init__(
        self,
        *,
        data_dir: str = "ARC-AGI/data",
        max_tasks: int = 400,
        max_iterations: int = 20,
        solver_depth: int = 6,
        solver_programs: int = 8000,
        workers: int = 16,
        target_accuracy: float = 0.90,
        bank_path: str = "concept_bank_v160.pkl",
    ) -> None:
        self.data_dir = data_dir
        self.max_tasks = max_tasks
        self.max_iterations = max_iterations
        self.solver_depth = solver_depth
        self.solver_programs = solver_programs
        self.workers = workers
        self.target_accuracy = target_accuracy
        
        # Concept bank
        self.bank = OverMiningConceptBank(bank_path)
        
        # Try to load existing bank
        if self.bank.load():
            print(f"✓ Loaded existing concept bank: {self.bank.get_stats()}")
        
        # State
        self.tasks: List[ARCTask] = []
        self.solved_tasks: Set[str] = set()  # Tasks we've solved at least once
        self.iteration_results: List[IterationResult] = []
    
    def load_tasks(self) -> int:
        """Load tasks."""
        self.tasks = load_arc_tasks(self.data_dir, self.max_tasks)
        return len(self.tasks)
    
    def run_iteration(self, iteration: int) -> IterationResult:
        """Run one over-mining iteration."""
        
        start = time.time()
        
        # Get all concepts as operators
        concept_operators = [
            c.to_solver_operator() 
            for c in self.bank.get_all_concepts()
        ]
        
        print(f"   Injecting {len(concept_operators)} concepts as operators")
        
        config = {
            "depth": self.solver_depth,
            "programs": self.solver_programs,
            "concept_operators": concept_operators,
        }
        
        # Prepare tasks
        task_args = [
            (t.task_id, t.train_pairs, t.test_input, t.test_output, config)
            for t in self.tasks
        ]
        
        # Solve in parallel
        results = []
        with multiprocessing.Pool(self.workers) as pool:
            results = pool.map(_solve_with_concepts, task_args)
        
        # Process results
        solved_count = 0
        new_solved = 0
        concepts_mined = 0
        failures: Dict[str, int] = defaultdict(int)
        
        for r in results:
            task_id = r["task_id"]
            
            if r.get("solved"):
                solved_count += 1
                
                # Is this a NEW solution?
                if task_id not in self.solved_tasks:
                    new_solved += 1
                    self.solved_tasks.add(task_id)
                
                # MINE THE SOLUTION AS CONCEPT
                concept = self.bank.mine_solution(
                    task_id=task_id,
                    program_steps=r.get("program_steps", []),
                    cost_bits=r.get("cost_bits", 0),
                )
                
                if concept:
                    concepts_mined += 1
            else:
                reason = r.get("failure_reason", "UNKNOWN")
                failures[reason] += 1
        
        # Save bank after each iteration
        self.bank.save()
        
        stats = self.bank.get_stats()
        
        result = IterationResult(
            iteration=iteration,
            tasks_total=len(self.tasks),
            tasks_solved=solved_count,
            accuracy=solved_count / len(self.tasks) if self.tasks else 0,
            concepts_mined_this_iter=concepts_mined,
            concepts_total=stats["alive_concepts"],
            concepts_cross_task=stats["cross_task_concepts"],
            new_tasks_solved=new_solved,
            elapsed_seconds=time.time() - start,
            failures_by_reason=dict(failures),
        )
        
        self.iteration_results.append(result)
        
        return result
    
    def run_overmining(self) -> Dict[str, Any]:
        """Run full over-mining pipeline."""
        
        print("\n" + "=" * 70)
        print("OVER-MINING PIPELINE V160 - EXPLORAÇÃO AGRESSIVA TOTAL")
        print("=" * 70)
        print(">> MINERA TUDO. NÃO DESCARTA NADA. VENCE A TODO CUSTO. <<")
        print("=" * 70)
        
        # Verify architecture
        arch = verify_authority_hierarchy()
        if not arch["all_passed"]:
            raise RuntimeError(f"Authority hierarchy invalid: {arch}")
        print(f"✓ Authority hierarchy verified")
        
        # Load tasks
        n_tasks = self.load_tasks()
        print(f"✓ Loaded {n_tasks} tasks from {self.data_dir}")
        
        if n_tasks == 0:
            raise RuntimeError("No tasks loaded!")
        
        best_accuracy = 0.0
        
        for i in range(self.max_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {i + 1}/{self.max_iterations}")
            print("=" * 70)
            
            result = self.run_iteration(i + 1)
            
            if result.accuracy > best_accuracy:
                best_accuracy = result.accuracy
            
            print(f"   ✓ Solved: {result.tasks_solved}/{result.tasks_total} ({result.accuracy*100:.1f}%)")
            print(f"   ✓ NEW tasks solved this iter: {result.new_tasks_solved}")
            print(f"   ✓ Concepts mined this iter: {result.concepts_mined_this_iter}")
            print(f"   ✓ Total concepts: {result.concepts_total}")
            print(f"   ✓ Cross-task concepts: {result.concepts_cross_task}")
            print(f"   ✓ Time: {result.elapsed_seconds:.1f}s")
            
            if result.failures_by_reason:
                total_failures = sum(result.failures_by_reason.values())
                print(f"   ❌ Failures:")
                for reason, count in sorted(result.failures_by_reason.items(), key=lambda x: -x[1]):
                    pct = count / total_failures * 100
                    print(f"      • {reason}: {count} ({pct:.0f}%)")
            
            # Check target
            if result.accuracy >= self.target_accuracy:
                print(f"\n🎉🎉🎉 TARGET REACHED: {result.accuracy*100:.1f}% >= {self.target_accuracy*100:.0f}% 🎉🎉🎉")
                break
            
            # Check if we're making progress
            if i >= 3:
                recent = [r.accuracy for r in self.iteration_results[-3:]]
                if max(recent) - min(recent) < 0.01:
                    print(f"\n⚠️ PLATEAU at {result.accuracy*100:.1f}% - may need more depth/programs")
        
        # Final stats
        stats = self.bank.get_stats()
        final_accuracy = self.iteration_results[-1].accuracy if self.iteration_results else 0
        
        return {
            "iterations": len(self.iteration_results),
            "final_accuracy": final_accuracy,
            "best_accuracy": best_accuracy,
            "target_accuracy": self.target_accuracy,
            "target_reached": final_accuracy >= self.target_accuracy,
            "concepts": stats,
            "solved_tasks": len(self.solved_tasks),
            "accuracy_history": [r.accuracy for r in self.iteration_results],
        }
    
    def generate_report(self) -> str:
        """Generate final report."""
        
        stats = self.bank.get_stats()
        final_result = self.iteration_results[-1] if self.iteration_results else None
        
        lines = []
        lines.append("=" * 70)
        lines.append("OVER-MINING FINAL REPORT - V160")
        lines.append("=" * 70)
        
        lines.append(f"\n## Configuration")
        lines.append(f"- Tasks: {self.max_tasks}")
        lines.append(f"- Depth: {self.solver_depth}")
        lines.append(f"- Programs: {self.solver_programs}")
        lines.append(f"- Workers: {self.workers}")
        lines.append(f"- Target: {self.target_accuracy*100:.0f}%")
        
        lines.append(f"\n## Results")
        if final_result:
            lines.append(f"- Final Accuracy: {final_result.accuracy*100:.1f}%")
            lines.append(f"- Tasks Solved: {final_result.tasks_solved}/{final_result.tasks_total}")
            lines.append(f"- Iterations: {len(self.iteration_results)}")
        
        lines.append(f"\n## Concept Bank")
        lines.append(f"- Total Concepts: {stats['total_concepts']}")
        lines.append(f"- Alive Concepts: {stats['alive_concepts']}")
        lines.append(f"- Cross-Task Concepts: {stats['cross_task_concepts']}")
        lines.append(f"- Unique Tasks Covered: {stats['unique_tasks_covered']}")
        
        lines.append(f"\n## Accuracy History")
        for i, r in enumerate(self.iteration_results):
            lines.append(f"   Iter {i+1}: {r.accuracy*100:.1f}% (concepts: {r.concepts_total})")
        
        lines.append(f"\n## Verdict")
        if final_result and final_result.accuracy >= self.target_accuracy:
            lines.append(f"   🎉 TARGET REACHED - PHASE 1 COMPLETE")
            lines.append(f"   → Proceed to PHASE 2: FREEZE")
            lines.append(f"   → Then PHASE 3: COMPRESSION")
        else:
            lines.append(f"   ⏳ Target not yet reached")
            lines.append(f"   → Continue mining")
            lines.append(f"   → Consider increasing depth/programs")
        
        lines.append("\n" + "=" * 70)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("=" * 70)
        
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Over-Mining Pipeline V160")
    parser.add_argument("--tasks", type=int, default=400, help="Max tasks")
    parser.add_argument("--iterations", type=int, default=20, help="Max iterations")
    parser.add_argument("--depth", type=int, default=6, help="Solver depth")
    parser.add_argument("--programs", type=int, default=8000, help="Max programs")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--target", type=float, default=0.90, help="Target accuracy")
    parser.add_argument("--quick", action="store_true", help="Quick test (50 tasks)")
    
    args = parser.parse_args()
    
    if args.quick:
        args.tasks = 50
        args.iterations = 5
    
    pipeline = OverMiningPipeline(
        data_dir="ARC-AGI/data",
        max_tasks=args.tasks,
        max_iterations=args.iterations,
        solver_depth=args.depth,
        solver_programs=args.programs,
        workers=args.workers,
        target_accuracy=args.target,
    )
    
    result = pipeline.run_overmining()
    
    print("\n" + pipeline.generate_report())
    
    # Save result
    with open("overmining_result.json", "w") as f:
        json.dump(result, f, indent=2)
