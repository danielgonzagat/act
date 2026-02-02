"""
Tests for arc_evaluation_harness_v148.py
"""

import unittest
import tempfile
import json
import os

from atos_core.arc_evaluation_harness_v148 import (
    ARCTask,
    TaskResult,
    EvaluationMetrics,
    EvaluationReport,
    load_arc_tasks,
    compute_metrics,
)


class TestARCTask(unittest.TestCase):
    """Tests for ARCTask dataclass."""
    
    def test_task_creation(self):
        """Create ARCTask with all fields."""
        task = ARCTask(
            task_id="test123",
            train_pairs=[([[0, 1], [1, 0]], [[1, 0], [0, 1]])],
            test_pairs=[([[0, 0], [0, 0]], [[0, 0], [0, 0]])],
            source_file="/path/to/test123.json",
            dataset="training",
        )
        
        self.assertEqual(task.task_id, "test123")
        self.assertEqual(len(task.train_pairs), 1)
        self.assertEqual(len(task.test_pairs), 1)
        self.assertEqual(task.dataset, "training")
    
    def test_task_to_dict(self):
        """Convert task to dict."""
        task = ARCTask(
            task_id="abc",
            train_pairs=[([[1]], [[2]])],
            test_pairs=[([[3]], [[4]])],
            source_file="abc.json",
            dataset="evaluation",
        )
        d = task.to_dict()
        
        self.assertEqual(d["task_id"], "abc")
        self.assertEqual(d["n_train"], 1)
        self.assertEqual(d["n_test"], 1)
        self.assertEqual(d["dataset"], "evaluation")


class TestTaskResult(unittest.TestCase):
    """Tests for TaskResult dataclass."""
    
    def test_result_creation(self):
        """Create TaskResult with all fields."""
        result = TaskResult(
            task_id="test123",
            solved=True,
            ambiguous=False,
            timeout=False,
            error=None,
            programs_tested=1000,
            solution_depth=3,
            solution_cost_bits=50,
            elapsed_ms=500,
            predicted_outputs=[[[1, 2], [3, 4]]],
            expected_outputs=[[[1, 2], [3, 4]]],
            match_per_test=[True],
            concepts_used=["concept_a"],
            concept_reuse_count=1,
        )
        
        self.assertTrue(result.solved)
        self.assertEqual(result.programs_tested, 1000)
        self.assertEqual(len(result.concepts_used), 1)
    
    def test_result_to_dict(self):
        """Convert result to dict."""
        result = TaskResult(
            task_id="xyz",
            solved=False,
            ambiguous=False,
            timeout=True,
            error=None,
            programs_tested=5000,
            solution_depth=0,
            solution_cost_bits=0,
            elapsed_ms=60000,
            predicted_outputs=[],
            expected_outputs=[[[0]]],
            match_per_test=[False],
            concepts_used=[],
            concept_reuse_count=0,
        )
        d = result.to_dict()
        
        self.assertEqual(d["task_id"], "xyz")
        self.assertFalse(d["solved"])
        self.assertTrue(d["timeout"])


class TestEvaluationMetrics(unittest.TestCase):
    """Tests for metrics computation."""
    
    def test_compute_metrics_empty(self):
        """Handle empty results."""
        metrics = compute_metrics([])
        
        self.assertEqual(metrics.total_tasks, 0)
        self.assertEqual(metrics.accuracy_pct, 0.0)
    
    def test_compute_metrics_all_solved(self):
        """All tasks solved."""
        results = [
            TaskResult(
                task_id=f"task{i}",
                solved=True,
                ambiguous=False,
                timeout=False,
                error=None,
                programs_tested=100,
                solution_depth=2,
                solution_cost_bits=30,
                elapsed_ms=100,
                predicted_outputs=[],
                expected_outputs=[],
                match_per_test=[True],
                concepts_used=["concept_x"],
                concept_reuse_count=1,
            )
            for i in range(10)
        ]
        
        metrics = compute_metrics(results)
        
        self.assertEqual(metrics.total_tasks, 10)
        self.assertEqual(metrics.solved_tasks, 10)
        self.assertEqual(metrics.accuracy_pct, 100.0)
        self.assertEqual(metrics.tasks_using_concepts, 10)
    
    def test_compute_metrics_mixed(self):
        """Mixed solved/unsolved tasks."""
        results = [
            TaskResult(
                task_id="solved1",
                solved=True,
                ambiguous=False,
                timeout=False,
                error=None,
                programs_tested=50,
                solution_depth=2,
                solution_cost_bits=25,
                elapsed_ms=200,
                predicted_outputs=[],
                expected_outputs=[],
                match_per_test=[True],
                concepts_used=["concept_a", "concept_b"],
                concept_reuse_count=2,
            ),
            TaskResult(
                task_id="unsolved1",
                solved=False,
                ambiguous=False,
                timeout=True,
                error=None,
                programs_tested=1000,
                solution_depth=0,
                solution_cost_bits=0,
                elapsed_ms=60000,
                predicted_outputs=[],
                expected_outputs=[],
                match_per_test=[False],
                concepts_used=[],
                concept_reuse_count=0,
            ),
        ]
        
        metrics = compute_metrics(results)
        
        self.assertEqual(metrics.total_tasks, 2)
        self.assertEqual(metrics.solved_tasks, 1)
        self.assertEqual(metrics.accuracy_pct, 50.0)
        self.assertEqual(metrics.timeout_tasks, 1)
        self.assertEqual(metrics.tasks_using_concepts, 1)


class TestEvaluationReport(unittest.TestCase):
    """Tests for report generation."""
    
    def test_report_to_dict(self):
        """Convert report to dict."""
        metrics = EvaluationMetrics(
            total_tasks=100,
            solved_tasks=50,
            ambiguous_tasks=5,
            timeout_tasks=10,
            error_tasks=2,
            accuracy_pct=50.0,
            ambiguity_rate_pct=5.0,
            timeout_rate_pct=10.0,
            error_rate_pct=2.0,
            avg_programs_tested=500.0,
            avg_solution_depth=3.0,
            avg_solution_cost_bits=40.0,
            avg_elapsed_ms=1000.0,
            total_concepts_used=75,
            unique_concepts_used=5,
            avg_reuse_per_concept=15.0,
            tasks_using_concepts=45,
            training_accuracy_pct=55.0,
            evaluation_accuracy_pct=45.0,
        )
        
        report = EvaluationReport(
            timestamp="2024-01-01T00:00:00",
            solver_version="test_solver",
            git_commit="abc123",
            dataset_path="/test/path",
            metrics=metrics,
        )
        
        d = report.to_dict()
        
        self.assertEqual(d["schema_version"], 148)
        self.assertEqual(d["metrics"]["total_tasks"], 100)
        self.assertEqual(d["metrics"]["accuracy_pct"], 50.0)
    
    def test_report_to_markdown(self):
        """Generate markdown report."""
        metrics = EvaluationMetrics(
            total_tasks=10,
            solved_tasks=5,
            ambiguous_tasks=0,
            timeout_tasks=2,
            error_tasks=0,
            accuracy_pct=50.0,
            ambiguity_rate_pct=0.0,
            timeout_rate_pct=20.0,
            error_rate_pct=0.0,
            avg_programs_tested=100.0,
            avg_solution_depth=2.0,
            avg_solution_cost_bits=30.0,
            avg_elapsed_ms=500.0,
            total_concepts_used=10,
            unique_concepts_used=2,
            avg_reuse_per_concept=5.0,
            tasks_using_concepts=4,
            training_accuracy_pct=60.0,
            evaluation_accuracy_pct=40.0,
        )
        
        report = EvaluationReport(
            timestamp="2024-01-01",
            solver_version="v148",
            metrics=metrics,
            unsolved_task_ids=["task1", "task2"],
        )
        
        md = report.to_markdown()
        
        self.assertIn("# ARC-AGI Evaluation Report", md)
        self.assertIn("**Accuracy** | **50.00%**", md)
        self.assertIn("Critical Question", md)


class TestLoadTasks(unittest.TestCase):
    """Tests for task loading."""
    
    def test_load_from_temp_dir(self):
        """Load tasks from temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create training directory
            training_dir = os.path.join(tmpdir, "training")
            os.makedirs(training_dir)
            
            # Create a test task file
            task_data = {
                "train": [
                    {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
                ],
                "test": [
                    {"input": [[0, 0], [0, 0]], "output": [[1, 1], [1, 1]]},
                ],
            }
            
            with open(os.path.join(training_dir, "test_task.json"), "w") as f:
                json.dump(task_data, f)
            
            # Load tasks
            tasks = load_arc_tasks(tmpdir, include_training=True, include_evaluation=False)
            
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].task_id, "test_task")
            self.assertEqual(tasks[0].dataset, "training")
            self.assertEqual(len(tasks[0].train_pairs), 1)
            self.assertEqual(len(tasks[0].test_pairs), 1)
    
    def test_load_actual_arc_dataset(self):
        """Load actual ARC dataset (if available)."""
        arc_path = "/workspaces/act/ARC-AGI/data"
        
        if os.path.exists(arc_path):
            tasks = load_arc_tasks(
                arc_path,
                include_training=True,
                include_evaluation=True,
            )
            
            # Should have 800 tasks (400 training + 400 evaluation)
            self.assertEqual(len(tasks), 800)
            
            # Check dataset distribution
            training_count = sum(1 for t in tasks if t.dataset == "training")
            eval_count = sum(1 for t in tasks if t.dataset == "evaluation")
            
            self.assertEqual(training_count, 400)
            self.assertEqual(eval_count, 400)


if __name__ == "__main__":
    unittest.main()
