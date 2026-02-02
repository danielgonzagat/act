"""
Tests for transfer_domain_v152.py

Tests the Seq2Seq micro-domain for transfer testing outside ARC.
"""

import unittest
from typing import Any, Dict, List

from atos_core.transfer_domain_v152 import (
    TRANSFER_DOMAIN_SCHEMA_VERSION_V152,
    Seq2SeqExample,
    Seq2SeqTask,
    TransferReport,
    TransferResult,
    generate_add_constant_task,
    generate_all_seq2seq_tasks,
    generate_cumsum_task,
    generate_diff_task,
    generate_double_first_last_task,
    generate_double_odds_zero_evens_task,
    generate_filter_even_task,
    generate_filter_odd_task,
    generate_filter_then_sort_task,
    generate_multiply_constant_task,
    generate_reverse_task,
    generate_reverse_then_cumsum_task,
    generate_sort_ascending_task,
    generate_sort_descending_task,
    generate_swap_first_last_task,
    generate_unique_task,
    op_add,
    op_cumsum,
    op_diff,
    op_double_first_last,
    op_filter_even,
    op_filter_odd,
    op_multiply,
    op_reverse,
    op_sort_asc,
    op_sort_desc,
    op_swap_first_last,
    op_unique,
    run_transfer_evaluation,
    SEQ2SEQ_OPS,
)


class TestSchemaVersion(unittest.TestCase):
    """Test schema version."""
    
    def test_version_is_152(self):
        self.assertEqual(TRANSFER_DOMAIN_SCHEMA_VERSION_V152, 152)


class TestSeq2SeqExample(unittest.TestCase):
    """Test Seq2SeqExample dataclass."""
    
    def test_creation(self):
        ex = Seq2SeqExample(input=[1, 2, 3], output=[2, 3, 4])
        self.assertEqual(ex.input, [1, 2, 3])
        self.assertEqual(ex.output, [2, 3, 4])


class TestSeq2SeqTask(unittest.TestCase):
    """Test Seq2SeqTask dataclass."""
    
    def test_to_dict(self):
        task = Seq2SeqTask(
            task_id="test_task",
            task_type="test_type",
            description="A test task",
            train_examples=[
                Seq2SeqExample(input=[1, 2], output=[2, 4]),
            ],
            test_examples=[
                Seq2SeqExample(input=[3, 4], output=[6, 8]),
            ],
            difficulty=2,
        )
        
        data = task.to_dict()
        
        self.assertEqual(data["task_id"], "test_task")
        self.assertEqual(data["task_type"], "test_type")
        self.assertEqual(data["difficulty"], 2)
        self.assertEqual(len(data["train"]), 1)
        self.assertEqual(len(data["test"]), 1)
    
    def test_from_dict(self):
        data = {
            "task_id": "test",
            "task_type": "arithmetic",
            "description": "Test",
            "difficulty": 1,
            "train": [{"input": [1], "output": [2]}],
            "test": [{"input": [3], "output": [4]}],
        }
        
        task = Seq2SeqTask.from_dict(data)
        
        self.assertEqual(task.task_id, "test")
        self.assertEqual(len(task.train_examples), 1)
        self.assertEqual(len(task.test_examples), 1)


class TestOperators(unittest.TestCase):
    """Test individual operators."""
    
    def test_op_add(self):
        self.assertEqual(op_add([1, 2, 3], 5), [6, 7, 8])
        self.assertEqual(op_add([0], 0), [0])
    
    def test_op_multiply(self):
        self.assertEqual(op_multiply([1, 2, 3], 2), [2, 4, 6])
        self.assertEqual(op_multiply([5], 3), [15])
    
    def test_op_reverse(self):
        self.assertEqual(op_reverse([1, 2, 3]), [3, 2, 1])
        self.assertEqual(op_reverse([1]), [1])
    
    def test_op_filter_even(self):
        self.assertEqual(op_filter_even([1, 2, 3, 4, 5]), [2, 4])
        self.assertEqual(op_filter_even([1, 3, 5]), [])
    
    def test_op_filter_odd(self):
        self.assertEqual(op_filter_odd([1, 2, 3, 4, 5]), [1, 3, 5])
        self.assertEqual(op_filter_odd([2, 4, 6]), [])
    
    def test_op_sort_asc(self):
        self.assertEqual(op_sort_asc([3, 1, 2]), [1, 2, 3])
        self.assertEqual(op_sort_asc([1]), [1])
    
    def test_op_sort_desc(self):
        self.assertEqual(op_sort_desc([1, 3, 2]), [3, 2, 1])
    
    def test_op_cumsum(self):
        self.assertEqual(op_cumsum([1, 2, 3]), [1, 3, 6])
        self.assertEqual(op_cumsum([5]), [5])
    
    def test_op_diff(self):
        self.assertEqual(op_diff([1, 3, 6]), [2, 3])
        self.assertEqual(op_diff([5, 5, 5]), [0, 0])
    
    def test_op_unique(self):
        self.assertEqual(op_unique([1, 2, 1, 3, 2]), [1, 2, 3])
        self.assertEqual(op_unique([1, 1, 1]), [1])
    
    def test_op_double_first_last(self):
        self.assertEqual(op_double_first_last([1, 2, 3]), [2, 2, 6])
        self.assertEqual(op_double_first_last([5]), [10])
    
    def test_op_swap_first_last(self):
        self.assertEqual(op_swap_first_last([1, 2, 3]), [3, 2, 1])
        self.assertEqual(op_swap_first_last([1, 2]), [2, 1])


class TestOperatorRegistry(unittest.TestCase):
    """Test operator registry."""
    
    def test_has_all_operators(self):
        expected = [
            "add", "multiply", "reverse", "filter_even", "filter_odd",
            "sort_asc", "sort_desc", "cumsum", "diff", "unique",
            "double_first_last", "swap_first_last",
        ]
        for op_name in expected:
            self.assertIn(op_name, SEQ2SEQ_OPS)
    
    def test_operators_are_callable(self):
        for op_name, op_func in SEQ2SEQ_OPS.items():
            self.assertTrue(callable(op_func))


class TestTaskGenerators(unittest.TestCase):
    """Test task generators."""
    
    def test_generate_add_constant(self):
        task = generate_add_constant_task(constant=5, seed=123)
        
        self.assertEqual(task.task_type, "arithmetic_elementwise")
        self.assertEqual(task.difficulty, 1)
        self.assertEqual(len(task.train_examples), 3)
        self.assertEqual(len(task.test_examples), 2)
        
        # Verify correctness
        for ex in task.train_examples + task.test_examples:
            expected = [x + 5 for x in ex.input]
            self.assertEqual(ex.output, expected)
    
    def test_generate_multiply_constant(self):
        task = generate_multiply_constant_task(constant=3, seed=123)
        
        for ex in task.train_examples + task.test_examples:
            expected = [x * 3 for x in ex.input]
            self.assertEqual(ex.output, expected)
    
    def test_generate_reverse(self):
        task = generate_reverse_task(seed=123)
        
        for ex in task.train_examples + task.test_examples:
            expected = list(reversed(ex.input))
            self.assertEqual(ex.output, expected)
    
    def test_generate_filter_even(self):
        task = generate_filter_even_task(seed=123)
        
        for ex in task.train_examples + task.test_examples:
            expected = [x for x in ex.input if x % 2 == 0]
            self.assertEqual(ex.output, expected)
    
    def test_generate_filter_odd(self):
        task = generate_filter_odd_task(seed=123)
        
        for ex in task.train_examples + task.test_examples:
            expected = [x for x in ex.input if x % 2 == 1]
            self.assertEqual(ex.output, expected)
    
    def test_generate_sort_ascending(self):
        task = generate_sort_ascending_task(seed=123)
        
        for ex in task.train_examples + task.test_examples:
            expected = sorted(ex.input)
            self.assertEqual(ex.output, expected)
    
    def test_generate_sort_descending(self):
        task = generate_sort_descending_task(seed=123)
        
        for ex in task.train_examples + task.test_examples:
            expected = sorted(ex.input, reverse=True)
            self.assertEqual(ex.output, expected)
    
    def test_generate_cumsum(self):
        task = generate_cumsum_task(seed=123)
        
        for ex in task.train_examples + task.test_examples:
            expected = []
            cumsum = 0
            for x in ex.input:
                cumsum += x
                expected.append(cumsum)
            self.assertEqual(ex.output, expected)
    
    def test_generate_diff(self):
        task = generate_diff_task(seed=123)
        
        for ex in task.train_examples + task.test_examples:
            expected = [ex.input[i+1] - ex.input[i] for i in range(len(ex.input)-1)]
            self.assertEqual(ex.output, expected)
    
    def test_generate_unique(self):
        task = generate_unique_task(seed=123)
        
        for ex in task.train_examples + task.test_examples:
            seen = set()
            expected = []
            for x in ex.input:
                if x not in seen:
                    seen.add(x)
                    expected.append(x)
            self.assertEqual(ex.output, expected)
    
    def test_generate_double_first_last(self):
        task = generate_double_first_last_task(seed=123)
        
        for ex in task.train_examples + task.test_examples:
            expected = list(ex.input)
            expected[0] *= 2
            expected[-1] *= 2
            self.assertEqual(ex.output, expected)
    
    def test_generate_swap_first_last(self):
        task = generate_swap_first_last_task(seed=123)
        
        for ex in task.train_examples + task.test_examples:
            expected = list(ex.input)
            expected[0], expected[-1] = expected[-1], expected[0]
            self.assertEqual(ex.output, expected)


class TestHardTasks(unittest.TestCase):
    """Test hard (composition) tasks."""
    
    def test_generate_reverse_then_cumsum(self):
        task = generate_reverse_then_cumsum_task(seed=123)
        
        self.assertEqual(task.difficulty, 3)
        self.assertEqual(task.task_type, "composition")
        
        for ex in task.train_examples + task.test_examples:
            reversed_inp = list(reversed(ex.input))
            expected = []
            cumsum = 0
            for x in reversed_inp:
                cumsum += x
                expected.append(cumsum)
            self.assertEqual(ex.output, expected)
    
    def test_generate_filter_then_sort(self):
        task = generate_filter_then_sort_task(seed=123)
        
        self.assertEqual(task.difficulty, 3)
        
        for ex in task.train_examples + task.test_examples:
            expected = sorted([x for x in ex.input if x % 2 == 0])
            self.assertEqual(ex.output, expected)
    
    def test_generate_double_odds_zero_evens(self):
        task = generate_double_odds_zero_evens_task(seed=123)
        
        self.assertEqual(task.difficulty, 3)
        
        for ex in task.train_examples + task.test_examples:
            expected = [x * 2 if x % 2 == 1 else 0 for x in ex.input]
            self.assertEqual(ex.output, expected)


class TestGenerateAllTasks(unittest.TestCase):
    """Test generating all tasks."""
    
    def test_generates_correct_number(self):
        tasks = generate_all_seq2seq_tasks()
        
        # 9 easy + 5 medium + 3 hard = 17
        self.assertEqual(len(tasks), 17)
    
    def test_difficulty_distribution(self):
        tasks = generate_all_seq2seq_tasks()
        
        easy = sum(1 for t in tasks if t.difficulty == 1)
        medium = sum(1 for t in tasks if t.difficulty == 2)
        hard = sum(1 for t in tasks if t.difficulty == 3)
        
        self.assertEqual(easy, 9)
        self.assertEqual(medium, 5)
        self.assertEqual(hard, 3)
    
    def test_all_tasks_have_examples(self):
        tasks = generate_all_seq2seq_tasks()
        
        for task in tasks:
            self.assertGreater(len(task.train_examples), 0)
            self.assertGreater(len(task.test_examples), 0)


class TestTransferResult(unittest.TestCase):
    """Test TransferResult dataclass."""
    
    def test_to_dict(self):
        result = TransferResult(
            task_id="test_task",
            task_type="test_type",
            difficulty=2,
            solved=True,
            predicted_output=[1, 2, 3],
            expected_output=[1, 2, 3],
            concepts_used=["concept_a"],
            search_steps=10,
        )
        
        data = result.to_dict()
        
        self.assertEqual(data["task_id"], "test_task")
        self.assertTrue(data["solved"])
        self.assertEqual(data["concepts_used"], ["concept_a"])


class TestTransferReport(unittest.TestCase):
    """Test TransferReport dataclass."""
    
    def test_to_dict(self):
        report = TransferReport(
            total_tasks=10,
            solved=7,
            accuracy_pct=70.0,
        )
        
        data = report.to_dict()
        
        self.assertEqual(data["schema_version"], 152)
        self.assertEqual(data["total_tasks"], 10)
        self.assertEqual(data["accuracy_pct"], 70.0)
    
    def test_to_markdown(self):
        report = TransferReport(
            total_tasks=10,
            solved=7,
            accuracy_pct=70.0,
            easy_solved=5,
            easy_total=5,
            medium_solved=2,
            medium_total=3,
            hard_solved=0,
            hard_total=2,
        )
        
        md = report.to_markdown()
        
        self.assertIn("70.0%", md)
        self.assertIn("Easy: 5/5", md)
        self.assertIn("Medium: 2/3", md)


class TestRunTransferEvaluation(unittest.TestCase):
    """Test the transfer evaluation runner."""
    
    def mock_solver(self, task: Seq2SeqTask) -> TransferResult:
        """Mock solver that always succeeds for easy tasks."""
        test_ex = task.test_examples[0]
        solved = task.difficulty == 1  # Only solve easy tasks
        
        return TransferResult(
            task_id=task.task_id,
            task_type=task.task_type,
            difficulty=task.difficulty,
            solved=solved,
            predicted_output=test_ex.output if solved else None,
            expected_output=test_ex.output,
            concepts_used=["mock_concept"] if solved else [],
            search_steps=10,
        )
    
    def test_evaluation_runs(self):
        tasks = generate_all_seq2seq_tasks()
        report = run_transfer_evaluation(tasks, self.mock_solver)
        
        self.assertEqual(report.total_tasks, 17)
        # Only easy tasks (9) should be solved
        self.assertEqual(report.solved, 9)
        self.assertEqual(report.easy_solved, 9)
        self.assertEqual(report.medium_solved, 0)
        self.assertEqual(report.hard_solved, 0)
    
    def test_tracks_concepts(self):
        tasks = generate_all_seq2seq_tasks()[:3]  # Just a few
        report = run_transfer_evaluation(tasks, self.mock_solver)
        
        self.assertGreater(report.concepts_used_total, 0)


if __name__ == "__main__":
    unittest.main()
