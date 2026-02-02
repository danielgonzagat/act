"""
transfer_domain_v152.py - Non-ARC Transfer Domain for Generalization Testing

This module implements a micro-domain OUTSIDE of ARC to prove that:
1. The system generalizes beyond ARC
2. Concepts transfer to different task types
3. The emergence regime works universally

DOMAIN: Sequence Transformation (Seq2Seq)
- Input: List of integers
- Output: Transformed list of integers
- Rules: Arithmetic, positional, filtering operations

This is INTENTIONALLY DIFFERENT from ARC (grids) to test true generalization.

Schema version: 152
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


TRANSFER_DOMAIN_SCHEMA_VERSION_V152 = 152


# ─────────────────────────────────────────────────────────────────────────────
# Seq2Seq Task Definition
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Seq2SeqExample:
    """A single input-output example."""
    input: List[int]
    output: List[int]


@dataclass
class Seq2SeqTask:
    """A sequence transformation task with train/test examples."""
    
    task_id: str
    task_type: str  # Category of transformation
    description: str
    train_examples: List[Seq2SeqExample]
    test_examples: List[Seq2SeqExample]
    difficulty: int = 1  # 1=easy, 2=medium, 3=hard
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "task_type": str(self.task_type),
            "description": str(self.description),
            "difficulty": int(self.difficulty),
            "train": [
                {"input": list(e.input), "output": list(e.output)}
                for e in self.train_examples
            ],
            "test": [
                {"input": list(e.input), "output": list(e.output)}
                for e in self.test_examples
            ],
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Seq2SeqTask":
        return Seq2SeqTask(
            task_id=str(d["task_id"]),
            task_type=str(d["task_type"]),
            description=str(d.get("description", "")),
            difficulty=int(d.get("difficulty", 1)),
            train_examples=[
                Seq2SeqExample(input=list(e["input"]), output=list(e["output"]))
                for e in d["train"]
            ],
            test_examples=[
                Seq2SeqExample(input=list(e["input"]), output=list(e["output"]))
                for e in d["test"]
            ],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task Generators (Programmatic)
# ─────────────────────────────────────────────────────────────────────────────


def generate_add_constant_task(constant: int = 1, seed: int = 42) -> Seq2SeqTask:
    """Generate: output[i] = input[i] + constant"""
    rng = random.Random(seed)
    
    examples = []
    for i in range(5):
        inp = [rng.randint(0, 10) for _ in range(rng.randint(3, 7))]
        out = [x + constant for x in inp]
        examples.append(Seq2SeqExample(input=inp, output=out))
    
    return Seq2SeqTask(
        task_id=f"add_constant_{constant}",
        task_type="arithmetic_elementwise",
        description=f"Add {constant} to each element",
        train_examples=examples[:3],
        test_examples=examples[3:],
        difficulty=1,
    )


def generate_multiply_constant_task(constant: int = 2, seed: int = 42) -> Seq2SeqTask:
    """Generate: output[i] = input[i] * constant"""
    rng = random.Random(seed)
    
    examples = []
    for i in range(5):
        inp = [rng.randint(0, 5) for _ in range(rng.randint(3, 6))]
        out = [x * constant for x in inp]
        examples.append(Seq2SeqExample(input=inp, output=out))
    
    return Seq2SeqTask(
        task_id=f"multiply_constant_{constant}",
        task_type="arithmetic_elementwise",
        description=f"Multiply each element by {constant}",
        train_examples=examples[:3],
        test_examples=examples[3:],
        difficulty=1,
    )


def generate_reverse_task(seed: int = 42) -> Seq2SeqTask:
    """Generate: output = reverse(input)"""
    rng = random.Random(seed)
    
    examples = []
    for i in range(5):
        inp = [rng.randint(0, 9) for _ in range(rng.randint(3, 8))]
        out = list(reversed(inp))
        examples.append(Seq2SeqExample(input=inp, output=out))
    
    return Seq2SeqTask(
        task_id="reverse",
        task_type="positional",
        description="Reverse the sequence",
        train_examples=examples[:3],
        test_examples=examples[3:],
        difficulty=1,
    )


def generate_filter_even_task(seed: int = 42) -> Seq2SeqTask:
    """Generate: output = filter(is_even, input)"""
    rng = random.Random(seed)
    
    examples = []
    for i in range(5):
        inp = [rng.randint(0, 20) for _ in range(rng.randint(5, 10))]
        out = [x for x in inp if x % 2 == 0]
        if len(out) == 0:  # Ensure at least one even
            inp.append(rng.randint(0, 10) * 2)
            out = [x for x in inp if x % 2 == 0]
        examples.append(Seq2SeqExample(input=inp, output=out))
    
    return Seq2SeqTask(
        task_id="filter_even",
        task_type="filtering",
        description="Keep only even numbers",
        train_examples=examples[:3],
        test_examples=examples[3:],
        difficulty=1,
    )


def generate_filter_odd_task(seed: int = 42) -> Seq2SeqTask:
    """Generate: output = filter(is_odd, input)"""
    rng = random.Random(seed)
    
    examples = []
    for i in range(5):
        inp = [rng.randint(0, 20) for _ in range(rng.randint(5, 10))]
        out = [x for x in inp if x % 2 == 1]
        if len(out) == 0:  # Ensure at least one odd
            inp.append(rng.randint(0, 10) * 2 + 1)
            out = [x for x in inp if x % 2 == 1]
        examples.append(Seq2SeqExample(input=inp, output=out))
    
    return Seq2SeqTask(
        task_id="filter_odd",
        task_type="filtering",
        description="Keep only odd numbers",
        train_examples=examples[:3],
        test_examples=examples[3:],
        difficulty=1,
    )


def generate_sort_ascending_task(seed: int = 42) -> Seq2SeqTask:
    """Generate: output = sorted(input)"""
    rng = random.Random(seed)
    
    examples = []
    for i in range(5):
        inp = [rng.randint(0, 20) for _ in range(rng.randint(4, 8))]
        out = sorted(inp)
        examples.append(Seq2SeqExample(input=inp, output=out))
    
    return Seq2SeqTask(
        task_id="sort_ascending",
        task_type="ordering",
        description="Sort in ascending order",
        train_examples=examples[:3],
        test_examples=examples[3:],
        difficulty=1,
    )


def generate_sort_descending_task(seed: int = 42) -> Seq2SeqTask:
    """Generate: output = sorted(input, reverse=True)"""
    rng = random.Random(seed)
    
    examples = []
    for i in range(5):
        inp = [rng.randint(0, 20) for _ in range(rng.randint(4, 8))]
        out = sorted(inp, reverse=True)
        examples.append(Seq2SeqExample(input=inp, output=out))
    
    return Seq2SeqTask(
        task_id="sort_descending",
        task_type="ordering",
        description="Sort in descending order",
        train_examples=examples[:3],
        test_examples=examples[3:],
        difficulty=1,
    )


def generate_cumsum_task(seed: int = 42) -> Seq2SeqTask:
    """Generate: output[i] = sum(input[:i+1]) (cumulative sum)"""
    rng = random.Random(seed)
    
    examples = []
    for i in range(5):
        inp = [rng.randint(1, 5) for _ in range(rng.randint(3, 6))]
        out = []
        cumsum = 0
        for x in inp:
            cumsum += x
            out.append(cumsum)
        examples.append(Seq2SeqExample(input=inp, output=out))
    
    return Seq2SeqTask(
        task_id="cumulative_sum",
        task_type="aggregation",
        description="Compute cumulative sum",
        train_examples=examples[:3],
        test_examples=examples[3:],
        difficulty=2,
    )


def generate_diff_task(seed: int = 42) -> Seq2SeqTask:
    """Generate: output[i] = input[i+1] - input[i] (differences)"""
    rng = random.Random(seed)
    
    examples = []
    for i in range(5):
        inp = [rng.randint(0, 10) for _ in range(rng.randint(4, 7))]
        out = [inp[i+1] - inp[i] for i in range(len(inp)-1)]
        examples.append(Seq2SeqExample(input=inp, output=out))
    
    return Seq2SeqTask(
        task_id="differences",
        task_type="aggregation",
        description="Compute differences between consecutive elements",
        train_examples=examples[:3],
        test_examples=examples[3:],
        difficulty=2,
    )


def generate_unique_task(seed: int = 42) -> Seq2SeqTask:
    """Generate: output = unique(input) preserving order"""
    rng = random.Random(seed)
    
    examples = []
    for i in range(5):
        # Generate with duplicates
        base = [rng.randint(0, 5) for _ in range(rng.randint(3, 5))]
        inp = base + [rng.choice(base) for _ in range(rng.randint(1, 3))]
        rng.shuffle(inp)
        
        # Unique preserving order
        seen = set()
        out = []
        for x in inp:
            if x not in seen:
                seen.add(x)
                out.append(x)
        
        examples.append(Seq2SeqExample(input=inp, output=out))
    
    return Seq2SeqTask(
        task_id="unique",
        task_type="filtering",
        description="Remove duplicates, preserving order",
        train_examples=examples[:3],
        test_examples=examples[3:],
        difficulty=2,
    )


def generate_double_first_last_task(seed: int = 42) -> Seq2SeqTask:
    """Generate: Double the first and last elements."""
    rng = random.Random(seed)
    
    examples = []
    for i in range(5):
        inp = [rng.randint(1, 5) for _ in range(rng.randint(3, 6))]
        out = list(inp)
        out[0] *= 2
        out[-1] *= 2
        examples.append(Seq2SeqExample(input=inp, output=out))
    
    return Seq2SeqTask(
        task_id="double_first_last",
        task_type="positional",
        description="Double the first and last elements",
        train_examples=examples[:3],
        test_examples=examples[3:],
        difficulty=2,
    )


def generate_swap_first_last_task(seed: int = 42) -> Seq2SeqTask:
    """Generate: Swap first and last elements."""
    rng = random.Random(seed)
    
    examples = []
    for i in range(5):
        inp = [rng.randint(0, 9) for _ in range(rng.randint(3, 7))]
        out = list(inp)
        out[0], out[-1] = out[-1], out[0]
        examples.append(Seq2SeqExample(input=inp, output=out))
    
    return Seq2SeqTask(
        task_id="swap_first_last",
        task_type="positional",
        description="Swap first and last elements",
        train_examples=examples[:3],
        test_examples=examples[3:],
        difficulty=2,
    )


# Hard tasks (combining operations)
def generate_reverse_then_cumsum_task(seed: int = 42) -> Seq2SeqTask:
    """Generate: cumsum(reverse(input))"""
    rng = random.Random(seed)
    
    examples = []
    for i in range(5):
        inp = [rng.randint(1, 5) for _ in range(rng.randint(3, 5))]
        reversed_inp = list(reversed(inp))
        out = []
        cumsum = 0
        for x in reversed_inp:
            cumsum += x
            out.append(cumsum)
        examples.append(Seq2SeqExample(input=inp, output=out))
    
    return Seq2SeqTask(
        task_id="reverse_then_cumsum",
        task_type="composition",
        description="Reverse then compute cumulative sum",
        train_examples=examples[:3],
        test_examples=examples[3:],
        difficulty=3,
    )


def generate_filter_then_sort_task(seed: int = 42) -> Seq2SeqTask:
    """Generate: sort(filter_even(input))"""
    rng = random.Random(seed)
    
    examples = []
    for i in range(5):
        inp = [rng.randint(0, 15) for _ in range(rng.randint(6, 10))]
        # Ensure at least 2 even numbers
        evens_needed = 2 - sum(1 for x in inp if x % 2 == 0)
        if evens_needed > 0:
            inp.extend([rng.randint(0, 5) * 2 for _ in range(evens_needed)])
        out = sorted([x for x in inp if x % 2 == 0])
        examples.append(Seq2SeqExample(input=inp, output=out))
    
    return Seq2SeqTask(
        task_id="filter_even_then_sort",
        task_type="composition",
        description="Filter even numbers then sort",
        train_examples=examples[:3],
        test_examples=examples[3:],
        difficulty=3,
    )


def generate_double_odds_zero_evens_task(seed: int = 42) -> Seq2SeqTask:
    """Generate: Double odds, replace evens with 0."""
    rng = random.Random(seed)
    
    examples = []
    for i in range(5):
        inp = [rng.randint(0, 10) for _ in range(rng.randint(4, 7))]
        out = [x * 2 if x % 2 == 1 else 0 for x in inp]
        examples.append(Seq2SeqExample(input=inp, output=out))
    
    return Seq2SeqTask(
        task_id="double_odds_zero_evens",
        task_type="conditional",
        description="Double odd numbers, replace even with 0",
        train_examples=examples[:3],
        test_examples=examples[3:],
        difficulty=3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full Task Suite
# ─────────────────────────────────────────────────────────────────────────────


def generate_all_seq2seq_tasks(base_seed: int = 42) -> List[Seq2SeqTask]:
    """Generate all Seq2Seq tasks for transfer testing."""
    generators = [
        # Easy (difficulty 1)
        (generate_add_constant_task, {"constant": 1}),
        (generate_add_constant_task, {"constant": 3}),
        (generate_multiply_constant_task, {"constant": 2}),
        (generate_multiply_constant_task, {"constant": 3}),
        (generate_reverse_task, {}),
        (generate_filter_even_task, {}),
        (generate_filter_odd_task, {}),
        (generate_sort_ascending_task, {}),
        (generate_sort_descending_task, {}),
        
        # Medium (difficulty 2)
        (generate_cumsum_task, {}),
        (generate_diff_task, {}),
        (generate_unique_task, {}),
        (generate_double_first_last_task, {}),
        (generate_swap_first_last_task, {}),
        
        # Hard (difficulty 3) - compositions
        (generate_reverse_then_cumsum_task, {}),
        (generate_filter_then_sort_task, {}),
        (generate_double_odds_zero_evens_task, {}),
    ]
    
    tasks = []
    for i, (gen_func, kwargs) in enumerate(generators):
        task = gen_func(seed=base_seed + i, **kwargs)
        tasks.append(task)
    
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Seq2Seq Operators (Primitives)
# ─────────────────────────────────────────────────────────────────────────────


def op_add(seq: List[int], constant: int) -> List[int]:
    """Add constant to each element."""
    return [x + constant for x in seq]


def op_multiply(seq: List[int], constant: int) -> List[int]:
    """Multiply each element by constant."""
    return [x * constant for x in seq]


def op_reverse(seq: List[int]) -> List[int]:
    """Reverse sequence."""
    return list(reversed(seq))


def op_filter_even(seq: List[int]) -> List[int]:
    """Keep only even numbers."""
    return [x for x in seq if x % 2 == 0]


def op_filter_odd(seq: List[int]) -> List[int]:
    """Keep only odd numbers."""
    return [x for x in seq if x % 2 == 1]


def op_sort_asc(seq: List[int]) -> List[int]:
    """Sort ascending."""
    return sorted(seq)


def op_sort_desc(seq: List[int]) -> List[int]:
    """Sort descending."""
    return sorted(seq, reverse=True)


def op_cumsum(seq: List[int]) -> List[int]:
    """Cumulative sum."""
    out = []
    cumsum = 0
    for x in seq:
        cumsum += x
        out.append(cumsum)
    return out


def op_diff(seq: List[int]) -> List[int]:
    """Consecutive differences."""
    return [seq[i+1] - seq[i] for i in range(len(seq)-1)]


def op_unique(seq: List[int]) -> List[int]:
    """Remove duplicates preserving order."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def op_double_first_last(seq: List[int]) -> List[int]:
    """Double first and last elements."""
    if len(seq) == 0:
        return seq
    out = list(seq)
    out[0] *= 2
    if len(out) > 1:
        out[-1] *= 2
    return out


def op_swap_first_last(seq: List[int]) -> List[int]:
    """Swap first and last elements."""
    if len(seq) < 2:
        return seq
    out = list(seq)
    out[0], out[-1] = out[-1], out[0]
    return out


def op_conditional_double_odds_zero_evens(seq: List[int]) -> List[int]:
    """Double odds, replace evens with 0."""
    return [x * 2 if x % 2 == 1 else 0 for x in seq]


# Operator registry for Seq2Seq
SEQ2SEQ_OPS = {
    "add": op_add,
    "multiply": op_multiply,
    "reverse": op_reverse,
    "filter_even": op_filter_even,
    "filter_odd": op_filter_odd,
    "sort_asc": op_sort_asc,
    "sort_desc": op_sort_desc,
    "cumsum": op_cumsum,
    "diff": op_diff,
    "unique": op_unique,
    "double_first_last": op_double_first_last,
    "swap_first_last": op_swap_first_last,
    "conditional_dble_odds_zero_evens": op_conditional_double_odds_zero_evens,
}


# ─────────────────────────────────────────────────────────────────────────────
# Transfer Evaluation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TransferResult:
    """Result of testing transfer to Seq2Seq domain."""
    
    task_id: str
    task_type: str
    difficulty: int
    solved: bool
    predicted_output: Optional[List[int]] = None
    expected_output: Optional[List[int]] = None
    concepts_used: List[str] = field(default_factory=list)
    search_steps: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "task_type": str(self.task_type),
            "difficulty": int(self.difficulty),
            "solved": bool(self.solved),
            "predicted_output": list(self.predicted_output) if self.predicted_output else None,
            "expected_output": list(self.expected_output) if self.expected_output else None,
            "concepts_used": list(self.concepts_used),
            "search_steps": int(self.search_steps),
        }


@dataclass
class TransferReport:
    """Full report on transfer testing."""
    
    total_tasks: int = 0
    solved: int = 0
    accuracy_pct: float = 0.0
    
    # By difficulty
    easy_solved: int = 0
    easy_total: int = 0
    medium_solved: int = 0
    medium_total: int = 0
    hard_solved: int = 0
    hard_total: int = 0
    
    # By task type
    by_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Concept usage
    concepts_used_total: int = 0
    unique_concepts_used: int = 0
    
    # Detailed results
    results: List[TransferResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(TRANSFER_DOMAIN_SCHEMA_VERSION_V152),
            "total_tasks": int(self.total_tasks),
            "solved": int(self.solved),
            "accuracy_pct": float(self.accuracy_pct),
            "by_difficulty": {
                "easy": {"solved": int(self.easy_solved), "total": int(self.easy_total)},
                "medium": {"solved": int(self.medium_solved), "total": int(self.medium_total)},
                "hard": {"solved": int(self.hard_solved), "total": int(self.hard_total)},
            },
            "by_type": {k: dict(v) for k, v in self.by_type.items()},
            "concepts_used_total": int(self.concepts_used_total),
            "unique_concepts_used": int(self.unique_concepts_used),
            "results": [r.to_dict() for r in self.results],
        }
    
    def to_markdown(self) -> str:
        """Generate markdown summary."""
        lines = [
            "# Transfer Domain Test Report (Seq2Seq)",
            "",
            f"**Schema Version:** {TRANSFER_DOMAIN_SCHEMA_VERSION_V152}",
            f"**Total Tasks:** {self.total_tasks}",
            f"**Solved:** {self.solved}",
            f"**Accuracy:** {self.accuracy_pct:.1f}%",
            "",
            "## By Difficulty",
            "",
            f"- Easy: {self.easy_solved}/{self.easy_total}",
            f"- Medium: {self.medium_solved}/{self.medium_total}",
            f"- Hard: {self.hard_solved}/{self.hard_total}",
            "",
            "## By Task Type",
            "",
        ]
        
        for task_type, stats in self.by_type.items():
            lines.append(f"- {task_type}: {stats.get('solved', 0)}/{stats.get('total', 0)}")
        
        lines.extend([
            "",
            "## Concept Usage",
            "",
            f"- Total calls: {self.concepts_used_total}",
            f"- Unique concepts: {self.unique_concepts_used}",
            "",
        ])
        
        return "\n".join(lines)


def run_transfer_evaluation(
    tasks: List[Seq2SeqTask],
    solve_func: Callable[[Seq2SeqTask], TransferResult],
) -> TransferReport:
    """
    Run transfer evaluation on Seq2Seq tasks.
    
    solve_func should be a function that takes a task and returns TransferResult.
    """
    report = TransferReport()
    report.total_tasks = len(tasks)
    
    all_concepts_used = []
    
    for task in tasks:
        result = solve_func(task)
        report.results.append(result)
        
        if result.solved:
            report.solved += 1
            
            if task.difficulty == 1:
                report.easy_solved += 1
            elif task.difficulty == 2:
                report.medium_solved += 1
            else:
                report.hard_solved += 1
        
        # Count by difficulty
        if task.difficulty == 1:
            report.easy_total += 1
        elif task.difficulty == 2:
            report.medium_total += 1
        else:
            report.hard_total += 1
        
        # Count by type
        if task.task_type not in report.by_type:
            report.by_type[task.task_type] = {"solved": 0, "total": 0}
        report.by_type[task.task_type]["total"] += 1
        if result.solved:
            report.by_type[task.task_type]["solved"] += 1
        
        # Track concepts
        all_concepts_used.extend(result.concepts_used)
    
    report.accuracy_pct = (report.solved / max(1, report.total_tasks)) * 100
    report.concepts_used_total = len(all_concepts_used)
    report.unique_concepts_used = len(set(all_concepts_used))
    
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────


__all__ = [
    "TRANSFER_DOMAIN_SCHEMA_VERSION_V152",
    "Seq2SeqExample",
    "Seq2SeqTask",
    "TransferResult",
    "TransferReport",
    "generate_all_seq2seq_tasks",
    "run_transfer_evaluation",
    "SEQ2SEQ_OPS",
    # Individual generators
    "generate_add_constant_task",
    "generate_multiply_constant_task",
    "generate_reverse_task",
    "generate_filter_even_task",
    "generate_filter_odd_task",
    "generate_sort_ascending_task",
    "generate_sort_descending_task",
    "generate_cumsum_task",
    "generate_diff_task",
    "generate_unique_task",
    "generate_double_first_last_task",
    "generate_swap_first_last_task",
    "generate_reverse_then_cumsum_task",
    "generate_filter_then_sort_task",
    "generate_double_odds_zero_evens_task",
    # Operators
    "op_add",
    "op_multiply",
    "op_reverse",
    "op_filter_even",
    "op_filter_odd",
    "op_sort_asc",
    "op_sort_desc",
    "op_cumsum",
    "op_diff",
    "op_unique",
    "op_double_first_last",
    "op_swap_first_last",
    "op_conditional_double_odds_zero_evens",
]
