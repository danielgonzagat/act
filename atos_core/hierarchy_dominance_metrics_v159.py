"""
hierarchy_dominance_metrics_v159.py - Métricas de Dominância de Hierarquia

Instrumentação para observar se a hierarquia de conceitos está DOMINANDO.

Métricas rastreadas:
1. CSV_CALL usage: número de soluções que usam CSV_CALL
2. Concept call depth: profundidade máxima de CSV_CALL por solução
3. Concept composition: quantas vezes um conceito chama outro conceito
4. Reuse distribution: reuse_count de cada conceito
5. Conceptual solutions: tasks resolvidas com ≤2 passos conceituais

Schema version: 159
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import os
from datetime import datetime


HIERARCHY_DOMINANCE_SCHEMA_VERSION = 159


@dataclass
class SolutionTrace:
    """Trace de uma solução individual."""
    task_id: str
    solved: bool
    
    # Steps info
    total_steps: int = 0
    primitive_steps: int = 0  # Passos que usam operadores primitivos
    csv_call_steps: int = 0   # Passos que usam CSV_CALL
    
    # Depth of CSV_CALL
    max_csv_call_depth: int = 0  # Máximo nesting de CSV_CALL
    
    # Concepts used
    concepts_used: List[str] = field(default_factory=list)
    concept_call_chain: List[str] = field(default_factory=list)  # C1 -> C2 -> C3
    
    # Effective depth
    effective_depth: int = 0  # Steps necessários (menor = melhor)
    
    def is_conceptual_solution(self, max_steps: int = 2) -> bool:
        """Retorna True se a solução usa ≤max_steps passos conceituais."""
        return self.solved and self.csv_call_steps > 0 and self.effective_depth <= max_steps


@dataclass
class ConceptDominanceStats:
    """Stats de um conceito individual."""
    concept_id: str
    
    # Usage
    tasks_used_in: Set[str] = field(default_factory=set)
    times_called: int = 0
    
    # Composition
    concepts_it_calls: Set[str] = field(default_factory=set)  # Conceitos que este chama
    called_by_concepts: Set[str] = field(default_factory=set)  # Conceitos que chamam este
    
    # Effectiveness
    tasks_solved_with: Set[str] = field(default_factory=set)  # Tasks resolvidas quando usado
    
    @property
    def reuse_count(self) -> int:
        return len(self.tasks_used_in)
    
    @property
    def composition_depth(self) -> int:
        """Profundidade na hierarquia (0 = primitivo, 1+ = composto)."""
        if not self.concepts_it_calls:
            return 0
        return 1  # Simplified - could recurse
    
    @property
    def solve_rate(self) -> float:
        """Taxa de sucesso quando usado."""
        if not self.tasks_used_in:
            return 0.0
        return len(self.tasks_solved_with) / len(self.tasks_used_in)


@dataclass
class HierarchyDominanceReport:
    """Relatório completo de dominância de hierarquia."""
    
    iteration: int
    timestamp: str
    
    # Global counts
    total_tasks: int = 0
    tasks_solved: int = 0
    
    # CSV_CALL dominance
    solutions_with_csv_call: int = 0
    solutions_without_csv_call: int = 0  # Primitivas puras
    
    # Depth metrics
    max_csv_call_depth_seen: int = 0
    avg_csv_call_depth: float = 0.0
    
    # Conceptual solutions (≤2 passos)
    conceptual_solutions: int = 0
    
    # Concept stats
    total_concepts_used: int = 0
    concepts_with_reuse_gt_1: int = 0
    concepts_with_reuse_gt_3: int = 0
    
    # Composition
    concepts_calling_concepts: int = 0  # Conceitos que chamam outros conceitos
    max_composition_chain: int = 0  # Maior cadeia C1->C2->C3
    
    # Per-concept details
    concept_details: List[Dict[str, Any]] = field(default_factory=list)
    
    # Example solutions
    example_conceptual_solutions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": HIERARCHY_DOMINANCE_SCHEMA_VERSION,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "summary": {
                "total_tasks": self.total_tasks,
                "tasks_solved": self.tasks_solved,
                "accuracy": self.tasks_solved / max(1, self.total_tasks),
            },
            "csv_call_dominance": {
                "solutions_with_csv_call": self.solutions_with_csv_call,
                "solutions_without_csv_call": self.solutions_without_csv_call,
                "csv_call_rate": self.solutions_with_csv_call / max(1, self.tasks_solved),
            },
            "depth_metrics": {
                "max_csv_call_depth": self.max_csv_call_depth_seen,
                "avg_csv_call_depth": self.avg_csv_call_depth,
            },
            "conceptual_dominance": {
                "conceptual_solutions": self.conceptual_solutions,
                "conceptual_rate": self.conceptual_solutions / max(1, self.tasks_solved),
            },
            "concept_reuse": {
                "total_concepts_used": self.total_concepts_used,
                "with_reuse_gt_1": self.concepts_with_reuse_gt_1,
                "with_reuse_gt_3": self.concepts_with_reuse_gt_3,
            },
            "composition": {
                "concepts_calling_concepts": self.concepts_calling_concepts,
                "max_composition_chain": self.max_composition_chain,
            },
            "concept_details": self.concept_details[:10],  # Top 10
            "example_solutions": self.example_conceptual_solutions[:5],  # Top 5
        }


class HierarchyDominanceTracker:
    """
    Rastreador de dominância de hierarquia.
    
    Uso:
        tracker = HierarchyDominanceTracker()
        
        for task_result in results:
            tracker.observe_solution(task_result)
        
        report = tracker.generate_report(iteration=1)
        tracker.print_report(report)
    """
    
    def __init__(self) -> None:
        self._traces: List[SolutionTrace] = []
        self._concept_stats: Dict[str, ConceptDominanceStats] = defaultdict(
            lambda: ConceptDominanceStats(concept_id="")
        )
        self._iteration = 0
    
    def reset(self) -> None:
        """Reset para nova iteração."""
        self._traces = []
        # Mantém concept_stats para acumular
    
    def observe_solution(
        self,
        task_id: str,
        solved: bool,
        program_steps: List[Dict[str, Any]],
        cost_bits: float = 0.0,
    ) -> SolutionTrace:
        """Observa uma solução e extrai métricas de hierarquia."""
        
        trace = SolutionTrace(
            task_id=task_id,
            solved=solved,
            total_steps=len(program_steps),
        )
        
        if not program_steps:
            self._traces.append(trace)
            return trace
        
        # Analisa steps
        csv_call_count = 0
        max_depth = 0
        current_depth = 0
        concepts_used = []
        
        for step in program_steps:
            op_id = step.get("op_id", "")
            args = step.get("args", {})
            
            # Detecta CSV_CALL
            if "CSV_CALL" in op_id or op_id.startswith("csv_call"):
                csv_call_count += 1
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                
                # Extrai conceito chamado
                concept_id = args.get("concept_id") or args.get("name") or op_id
                concepts_used.append(concept_id)
                
            elif op_id.startswith("mined_concept_") or op_id.startswith("concept_"):
                # Conceito sendo usado diretamente
                csv_call_count += 1
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                concepts_used.append(op_id)
                
            else:
                # Operador primitivo
                trace.primitive_steps += 1
                current_depth = 0  # Reset depth on primitive
        
        trace.csv_call_steps = csv_call_count
        trace.max_csv_call_depth = max_depth
        trace.concepts_used = concepts_used
        trace.effective_depth = len(program_steps)
        
        # Atualiza stats de conceitos
        for concept_id in concepts_used:
            if not self._concept_stats[concept_id].concept_id:
                self._concept_stats[concept_id].concept_id = concept_id
            
            stats = self._concept_stats[concept_id]
            stats.tasks_used_in.add(task_id)
            stats.times_called += 1
            
            if solved:
                stats.tasks_solved_with.add(task_id)
        
        # Detecta composição (conceito chamando conceito)
        if len(concepts_used) >= 2:
            for i in range(len(concepts_used) - 1):
                caller = concepts_used[i]
                callee = concepts_used[i + 1]
                
                self._concept_stats[caller].concepts_it_calls.add(callee)
                self._concept_stats[callee].called_by_concepts.add(caller)
        
        trace.concept_call_chain = concepts_used
        
        self._traces.append(trace)
        return trace
    
    def generate_report(self, iteration: int) -> HierarchyDominanceReport:
        """Gera relatório de dominância."""
        
        report = HierarchyDominanceReport(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
        )
        
        # Global counts
        report.total_tasks = len(self._traces)
        report.tasks_solved = sum(1 for t in self._traces if t.solved)
        
        # CSV_CALL dominance
        for trace in self._traces:
            if trace.solved:
                if trace.csv_call_steps > 0:
                    report.solutions_with_csv_call += 1
                else:
                    report.solutions_without_csv_call += 1
        
        # Depth metrics
        depths = [t.max_csv_call_depth for t in self._traces if t.solved and t.csv_call_steps > 0]
        if depths:
            report.max_csv_call_depth_seen = max(depths)
            report.avg_csv_call_depth = sum(depths) / len(depths)
        
        # Conceptual solutions
        report.conceptual_solutions = sum(
            1 for t in self._traces if t.is_conceptual_solution(max_steps=2)
        )
        
        # Concept stats
        used_concepts = [s for s in self._concept_stats.values() if s.tasks_used_in]
        report.total_concepts_used = len(used_concepts)
        report.concepts_with_reuse_gt_1 = sum(1 for c in used_concepts if c.reuse_count > 1)
        report.concepts_with_reuse_gt_3 = sum(1 for c in used_concepts if c.reuse_count > 3)
        
        # Composition
        report.concepts_calling_concepts = sum(1 for c in used_concepts if c.concepts_it_calls)
        
        # Max composition chain
        max_chain = 0
        for trace in self._traces:
            if trace.concept_call_chain:
                max_chain = max(max_chain, len(trace.concept_call_chain))
        report.max_composition_chain = max_chain
        
        # Per-concept details (sorted by reuse)
        sorted_concepts = sorted(used_concepts, key=lambda c: c.reuse_count, reverse=True)
        report.concept_details = [
            {
                "concept_id": c.concept_id,
                "reuse_count": c.reuse_count,
                "times_called": c.times_called,
                "solve_rate": round(c.solve_rate, 2),
                "calls_other_concepts": len(c.concepts_it_calls),
                "called_by_others": len(c.called_by_concepts),
            }
            for c in sorted_concepts[:10]
        ]
        
        # Example conceptual solutions
        conceptual_traces = [t for t in self._traces if t.is_conceptual_solution(max_steps=2)]
        report.example_conceptual_solutions = [
            {
                "task_id": t.task_id,
                "total_steps": t.total_steps,
                "csv_call_steps": t.csv_call_steps,
                "concepts_used": t.concepts_used[:5],
                "effective_depth": t.effective_depth,
            }
            for t in conceptual_traces[:5]
        ]
        
        return report
    
    def print_report(self, report: HierarchyDominanceReport) -> None:
        """Imprime relatório formatado."""
        
        print("\n" + "=" * 70)
        print(f"HIERARCHY DOMINANCE REPORT - Iteration {report.iteration}")
        print("=" * 70)
        
        print(f"\n📊 SUMMARY")
        print(f"   Total tasks: {report.total_tasks}")
        print(f"   Solved: {report.tasks_solved} ({report.tasks_solved/max(1,report.total_tasks)*100:.1f}%)")
        
        print(f"\n🎯 CSV_CALL DOMINANCE")
        print(f"   Solutions WITH CSV_CALL: {report.solutions_with_csv_call}")
        print(f"   Solutions WITHOUT (primitives only): {report.solutions_without_csv_call}")
        csv_rate = report.solutions_with_csv_call / max(1, report.tasks_solved)
        print(f"   CSV_CALL Rate: {csv_rate*100:.1f}%")
        
        if csv_rate < 0.1:
            print(f"   ⚠️  HIERARCHY NOT DOMINATING - CSV_CALL rate too low")
        elif csv_rate < 0.5:
            print(f"   🔶 HIERARCHY EMERGING - some CSV_CALL usage")
        else:
            print(f"   ✅ HIERARCHY DOMINATING - majority uses CSV_CALL")
        
        print(f"\n📈 DEPTH METRICS")
        print(f"   Max CSV_CALL depth: {report.max_csv_call_depth_seen}")
        print(f"   Avg CSV_CALL depth: {report.avg_csv_call_depth:.2f}")
        
        print(f"\n🧠 CONCEPTUAL SOLUTIONS (≤2 steps)")
        print(f"   Count: {report.conceptual_solutions}")
        concept_rate = report.conceptual_solutions / max(1, report.tasks_solved)
        print(f"   Rate: {concept_rate*100:.1f}%")
        
        print(f"\n🔄 CONCEPT REUSE")
        print(f"   Concepts used: {report.total_concepts_used}")
        print(f"   With reuse > 1: {report.concepts_with_reuse_gt_1}")
        print(f"   With reuse > 3: {report.concepts_with_reuse_gt_3}")
        
        print(f"\n🔗 COMPOSITION")
        print(f"   Concepts calling concepts: {report.concepts_calling_concepts}")
        print(f"   Max call chain length: {report.max_composition_chain}")
        
        if report.concept_details:
            print(f"\n📋 TOP CONCEPTS BY REUSE")
            for c in report.concept_details[:5]:
                print(f"   • {c['concept_id']}: reuse={c['reuse_count']}, solve_rate={c['solve_rate']}")
        
        if report.example_conceptual_solutions:
            print(f"\n✨ EXAMPLE CONCEPTUAL SOLUTIONS")
            for sol in report.example_conceptual_solutions[:3]:
                print(f"   • {sol['task_id']}: {sol['csv_call_steps']} CSV_CALLs, depth={sol['effective_depth']}")
        
        # Verdict
        print(f"\n" + "=" * 70)
        print("DOMINANCE VERDICT:")
        
        if csv_rate >= 0.5 and report.concepts_with_reuse_gt_3 >= 3:
            print("   ✅ HIERARCHY IS DOMINATING")
            print("   Concepts are solving multiple tasks with composition")
        elif csv_rate >= 0.2 or report.concepts_with_reuse_gt_1 >= 2:
            print("   🔶 HIERARCHY IS EMERGING")
            print("   Some concepts appearing, but not yet dominant")
        else:
            print("   ❌ HIERARCHY NOT YET ACTIVE")
            print("   Search is still primitive-dominated")
        
        print("=" * 70)


def save_dominance_report(report: HierarchyDominanceReport, filepath: str) -> None:
    """Salva relatório em arquivo JSON."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(report.to_dict(), f, indent=2)


def load_dominance_report(filepath: str) -> Dict[str, Any]:
    """Carrega relatório de arquivo JSON."""
    with open(filepath) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Integration helper
# ─────────────────────────────────────────────────────────────────────────────


def analyze_training_results(
    results: List[Dict[str, Any]],
    iteration: int = 1,
) -> HierarchyDominanceReport:
    """
    Analisa resultados de treinamento e gera relatório de dominância.
    
    Args:
        results: Lista de dicts com {task_id, solved, program_steps, ...}
        iteration: Número da iteração
    
    Returns:
        HierarchyDominanceReport
    """
    tracker = HierarchyDominanceTracker()
    
    for r in results:
        tracker.observe_solution(
            task_id=r.get("task_id", "unknown"),
            solved=r.get("solved", False),
            program_steps=r.get("program_steps", []),
            cost_bits=r.get("cost_bits", 0.0),
        )
    
    return tracker.generate_report(iteration)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_hierarchy_dominance_tracker():
    """Test básico do tracker."""
    
    tracker = HierarchyDominanceTracker()
    
    # Solução com CSV_CALL
    tracker.observe_solution(
        task_id="task_001",
        solved=True,
        program_steps=[
            {"op_id": "mined_concept_1", "args": {}},
            {"op_id": "primitive_op", "args": {}},
        ],
    )
    
    # Solução com múltiplos CSV_CALLs
    tracker.observe_solution(
        task_id="task_002",
        solved=True,
        program_steps=[
            {"op_id": "mined_concept_1", "args": {}},  # Reuse!
            {"op_id": "mined_concept_2", "args": {}},
        ],
    )
    
    # Solução primitiva
    tracker.observe_solution(
        task_id="task_003",
        solved=True,
        program_steps=[
            {"op_id": "rotate", "args": {}},
            {"op_id": "flip", "args": {}},
        ],
    )
    
    # Falha
    tracker.observe_solution(
        task_id="task_004",
        solved=False,
        program_steps=[],
    )
    
    report = tracker.generate_report(iteration=1)
    
    assert report.total_tasks == 4
    assert report.tasks_solved == 3
    assert report.solutions_with_csv_call == 2
    assert report.solutions_without_csv_call == 1
    assert report.concepts_with_reuse_gt_1 >= 1  # mined_concept_1 used twice
    
    tracker.print_report(report)
    
    print("\n✅ test_hierarchy_dominance_tracker PASSED")


if __name__ == "__main__":
    test_hierarchy_dominance_tracker()
