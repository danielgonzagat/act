"""
Tests for diversity_law_v149.py

Tests the diversity enforcement mechanisms that prevent
concept monopoly and ensure a healthy concept ecosystem.
"""

import unittest
from typing import Any, Dict, List

from atos_core.diversity_law_v149 import (
    ARC_DIVERSITY_LAW_SCHEMA_VERSION_V149,
    AlternativeMiningRequest,
    ConceptEcosystem,
    ConceptUsageStats,
    DiversityCheckResult,
    DiversityConfig,
    check_solution_diversity,
    compute_concept_signature,
    compute_concept_similarity,
    enforce_diversity_law,
    find_similar_concepts,
    run_lineage_tournament,
    should_trigger_alternative_mining,
)


class TestSchemaVersion(unittest.TestCase):
    """Test schema version."""
    
    def test_version_is_149(self):
        self.assertEqual(ARC_DIVERSITY_LAW_SCHEMA_VERSION_V149, 149)


class TestDiversityConfig(unittest.TestCase):
    """Test DiversityConfig defaults."""
    
    def test_default_values(self):
        config = DiversityConfig()
        self.assertEqual(config.max_concept_usage_pct, 40.0)
        self.assertEqual(config.repeated_call_penalty, 0.1)
        self.assertEqual(config.max_repeated_calls, 3)
        self.assertEqual(config.min_concept_families, 3)
        self.assertEqual(config.similarity_threshold, 0.9)
        self.assertEqual(config.min_diversity_score, 0.5)
        self.assertTrue(config.force_alternative_mining)
        self.assertTrue(config.enable_lineage_competition)
        self.assertEqual(config.min_lineages, 2)
        self.assertEqual(config.lineage_tournament_size, 3)


class TestConceptUsageStats(unittest.TestCase):
    """Test ConceptUsageStats tracking."""
    
    def test_usage_rate_calculation(self):
        stats = ConceptUsageStats(concept_id="test_concept")
        stats.tasks_using.add("task_1")
        stats.tasks_using.add("task_2")
        
        # 2 tasks using out of 10 total = 20%
        self.assertAlmostEqual(stats.usage_rate(10), 20.0)
        
        # 2 tasks using out of 4 total = 50%
        self.assertAlmostEqual(stats.usage_rate(4), 50.0)
    
    def test_usage_rate_zero_tasks(self):
        stats = ConceptUsageStats(concept_id="test")
        self.assertEqual(stats.usage_rate(0), 0.0)
    
    def test_avg_depth_calculation(self):
        stats = ConceptUsageStats(concept_id="test")
        stats.call_depths = [0, 1, 2, 1]
        self.assertEqual(stats.avg_depth(), 1.0)
    
    def test_avg_depth_empty(self):
        stats = ConceptUsageStats(concept_id="test")
        self.assertEqual(stats.avg_depth(), 0.0)


class TestConceptEcosystem(unittest.TestCase):
    """Test ConceptEcosystem tracking and metrics."""
    
    def setUp(self):
        self.ecosystem = ConceptEcosystem()
    
    def test_record_usage_creates_stats(self):
        self.ecosystem.record_usage("task_1", "concept_a", depth=1)
        
        self.assertIn("concept_a", self.ecosystem.concepts)
        self.assertEqual(self.ecosystem.concepts["concept_a"].total_calls, 1)
        self.assertIn("task_1", self.ecosystem.concepts["concept_a"].tasks_using)
    
    def test_record_usage_accumulates(self):
        self.ecosystem.record_usage("task_1", "concept_a", depth=0)
        self.ecosystem.record_usage("task_2", "concept_a", depth=1)
        
        self.assertEqual(self.ecosystem.concepts["concept_a"].total_calls, 2)
        self.assertEqual(len(self.ecosystem.concepts["concept_a"].tasks_using), 2)
    
    def test_record_usage_tracks_co_occurrence(self):
        self.ecosystem.record_usage(
            "task_1", "concept_a", depth=0,
            other_concepts=["concept_b", "concept_c"]
        )
        
        stats = self.ecosystem.concepts["concept_a"]
        self.assertEqual(stats.co_occurrence.get("concept_b"), 1)
        self.assertEqual(stats.co_occurrence.get("concept_c"), 1)
    
    def test_assign_lineage(self):
        self.ecosystem.assign_lineage("concept_a", "lineage_color")
        self.ecosystem.assign_lineage("concept_b", "lineage_color")
        self.ecosystem.assign_lineage("concept_c", "lineage_shape")
        
        self.assertEqual(len(self.ecosystem.lineages), 2)
        self.assertIn("concept_a", self.ecosystem.lineages["lineage_color"])
        self.assertIn("concept_b", self.ecosystem.lineages["lineage_color"])
        self.assertIn("concept_c", self.ecosystem.lineages["lineage_shape"])
    
    def test_get_dominant_concepts(self):
        # Setup: 10 tasks total
        self.ecosystem.total_tasks = 10
        
        # Concept A used in 5/10 tasks = 50% (above 40% threshold)
        for i in range(5):
            self.ecosystem.record_usage(f"task_{i}", "concept_a")
        
        # Concept B used in 3/10 tasks = 30% (below threshold)
        for i in range(3):
            self.ecosystem.record_usage(f"task_{i}", "concept_b")
        
        config = DiversityConfig()
        dominant = self.ecosystem.get_dominant_concepts(config)
        
        self.assertIn("concept_a", dominant)
        self.assertNotIn("concept_b", dominant)
    
    def test_diversity_score_empty(self):
        self.assertEqual(self.ecosystem.get_diversity_score(), 0.0)
    
    def test_diversity_score_single_concept(self):
        self.ecosystem.total_tasks = 1
        self.ecosystem.record_usage("task_1", "concept_a")
        
        # Single concept: low diversity
        score = self.ecosystem.get_diversity_score()
        self.assertLess(score, 0.5)
    
    def test_diversity_score_multiple_concepts(self):
        self.ecosystem.total_tasks = 10
        
        # 5 different concepts, evenly distributed
        for i, cid in enumerate(["c1", "c2", "c3", "c4", "c5"]):
            self.ecosystem.record_usage(f"task_{i % 10}", cid)
            self.ecosystem.assign_lineage(cid, f"lineage_{i % 3}")
        
        score = self.ecosystem.get_diversity_score()
        self.assertGreater(score, 0.3)
    
    def test_to_dict_serialization(self):
        self.ecosystem.total_tasks = 5
        self.ecosystem.record_usage("task_1", "concept_a")
        self.ecosystem.assign_lineage("concept_a", "lineage_1")
        
        data = self.ecosystem.to_dict()
        
        self.assertEqual(data["schema_version"], 149)
        self.assertEqual(data["total_tasks"], 5)
        self.assertEqual(data["n_concepts"], 1)
        self.assertEqual(data["n_lineages"], 1)
        self.assertIn("diversity_score", data)
        self.assertIn("concept_stats", data)


class TestConceptSimilarity(unittest.TestCase):
    """Test concept similarity computation."""
    
    def test_compute_concept_signature(self):
        program = [
            {"op": "color_swap"},
            {"op": "rotate"},
        ]
        
        sig = compute_concept_signature(program)
        self.assertEqual(len(sig), 16)  # 16-char hex
    
    def test_identical_programs_same_signature(self):
        program_a = [{"op": "rotate"}, {"op": "flip"}]
        program_b = [{"op": "rotate"}, {"op": "flip"}]
        
        sig_a = compute_concept_signature(program_a)
        sig_b = compute_concept_signature(program_b)
        
        self.assertEqual(sig_a, sig_b)
    
    def test_different_programs_different_signature(self):
        program_a = [{"op": "rotate"}]
        program_b = [{"op": "flip"}]
        
        sig_a = compute_concept_signature(program_a)
        sig_b = compute_concept_signature(program_b)
        
        self.assertNotEqual(sig_a, sig_b)
    
    def test_compute_similarity_identical(self):
        concept_a = {"program": [{"op": "a"}, {"op": "b"}]}
        concept_b = {"program": [{"op": "a"}, {"op": "b"}]}
        
        sim = compute_concept_similarity(concept_a, concept_b)
        self.assertEqual(sim, 1.0)
    
    def test_compute_similarity_no_overlap(self):
        concept_a = {"program": [{"op": "a"}, {"op": "b"}], "call_deps": ["x"]}
        concept_b = {"program": [{"op": "c"}, {"op": "d"}], "call_deps": ["y"]}
        
        sim = compute_concept_similarity(concept_a, concept_b)
        # No program overlap (0.0 * 0.7) + no deps overlap (0.0 * 0.3) = 0.0
        self.assertEqual(sim, 0.0)
    
    def test_compute_similarity_partial_overlap(self):
        concept_a = {"program": [{"op": "a"}, {"op": "b"}, {"op": "c"}]}
        concept_b = {"program": [{"op": "a"}, {"op": "d"}]}
        
        sim = compute_concept_similarity(concept_a, concept_b)
        # Jaccard: 1 / 4 = 0.25, weighted 0.7 → ~0.175
        self.assertGreater(sim, 0.1)
        self.assertLess(sim, 0.5)
    
    def test_find_similar_concepts(self):
        concepts = [
            {"concept_id": "c1", "program": [{"op": "a"}, {"op": "b"}]},
            {"concept_id": "c2", "program": [{"op": "a"}, {"op": "b"}]},  # Same as c1
            {"concept_id": "c3", "program": [{"op": "x"}, {"op": "y"}]},  # Different
        ]
        
        pairs = find_similar_concepts(concepts, threshold=0.9)
        
        self.assertEqual(len(pairs), 1)
        self.assertIn("c1", pairs[0])
        self.assertIn("c2", pairs[0])


class TestDiversityCheck(unittest.TestCase):
    """Test check_solution_diversity."""
    
    def setUp(self):
        self.config = DiversityConfig()
        self.ecosystem = ConceptEcosystem()
        self.ecosystem.total_tasks = 10
    
    def test_empty_solution(self):
        result = check_solution_diversity([], self.ecosystem, self.config)
        
        self.assertEqual(result.score, 0.0)
        self.assertEqual(len(result.violations), 0)
    
    def test_single_concept_ok(self):
        result = check_solution_diversity(["concept_a"], self.ecosystem, self.config)
        
        self.assertEqual(len(result.violations), 0)
        self.assertGreater(result.score, 0)
    
    def test_repeated_calls_warning(self):
        # 2 calls of same concept = warning
        result = check_solution_diversity(
            ["concept_a", "concept_a"],
            self.ecosystem,
            self.config
        )
        
        self.assertEqual(len(result.violations), 0)
        self.assertTrue(any(w["type"] == "MULTIPLE_CALLS" for w in result.warnings))
    
    def test_excessive_repeated_calls_violation(self):
        # More than max_repeated_calls (default 3)
        result = check_solution_diversity(
            ["concept_a"] * 5,
            self.ecosystem,
            self.config
        )
        
        self.assertEqual(len(result.violations), 1)
        self.assertEqual(result.violations[0]["type"], "REPEATED_CALLS_EXCEEDED")
        self.assertFalse(result.passed)
    
    def test_diverse_solution_passes(self):
        # Multiple different concepts
        result = check_solution_diversity(
            ["concept_a", "concept_b", "concept_c"],
            self.ecosystem,
            self.config
        )
        
        self.assertEqual(len(result.violations), 0)
        self.assertTrue(result.passed)
        self.assertGreater(result.score, 0.5)
    
    def test_dominant_concept_warning(self):
        # Setup ecosystem with dominant concept
        for i in range(6):  # 60% usage
            self.ecosystem.record_usage(f"task_{i}", "dominant_concept")
        
        result = check_solution_diversity(
            ["dominant_concept"],
            self.ecosystem,
            self.config
        )
        
        self.assertTrue(any(
            w["type"] == "DOMINANT_CONCEPT_USED"
            for w in result.warnings
        ))
    
    def test_single_lineage_warning(self):
        # Setup concepts in same lineage
        self.ecosystem.assign_lineage("concept_a", "lineage_1")
        self.ecosystem.assign_lineage("concept_b", "lineage_1")
        
        result = check_solution_diversity(
            ["concept_a", "concept_b"],
            self.ecosystem,
            self.config
        )
        
        self.assertTrue(any(
            w["type"] == "SINGLE_LINEAGE"
            for w in result.warnings
        ))


class TestAlternativeMining(unittest.TestCase):
    """Test alternative mining trigger."""
    
    def setUp(self):
        self.config = DiversityConfig()
        self.ecosystem = ConceptEcosystem()
        self.ecosystem.total_tasks = 10
    
    def test_no_mining_needed_healthy_ecosystem(self):
        # Multiple concepts, none dominant, multiple lineages
        for i, cid in enumerate(["c1", "c2", "c3", "c4"]):
            for j in range(2):
                self.ecosystem.record_usage(f"task_{i*2+j}", cid)
            self.ecosystem.assign_lineage(cid, f"lineage_{i % 3}")
        
        request = should_trigger_alternative_mining(self.ecosystem, self.config)
        
        self.assertIsNone(request)
    
    def test_mining_triggered_by_dominance(self):
        # One concept dominates
        for i in range(6):  # 60% usage
            self.ecosystem.record_usage(f"task_{i}", "dominant_concept")
        
        request = should_trigger_alternative_mining(self.ecosystem, self.config)
        
        self.assertIsNotNone(request)
        self.assertEqual(request.trigger_reason, "CONCEPT_DOMINANCE")
        self.assertEqual(request.dominant_concept_id, "dominant_concept")
    
    def test_mining_triggered_by_insufficient_lineages(self):
        # Only one lineage
        for cid in ["c1", "c2", "c3"]:
            self.ecosystem.record_usage(f"task_{cid}", cid)
            self.ecosystem.assign_lineage(cid, "single_lineage")
        
        request = should_trigger_alternative_mining(self.ecosystem, self.config)
        
        self.assertIsNotNone(request)
        self.assertEqual(request.trigger_reason, "INSUFFICIENT_LINEAGES")


class TestLineageTournament(unittest.TestCase):
    """Test lineage tournament competition."""
    
    def setUp(self):
        self.config = DiversityConfig()
        self.ecosystem = ConceptEcosystem()
        self.ecosystem.total_tasks = 20
    
    def test_insufficient_lineages(self):
        # Only one lineage
        self.ecosystem.assign_lineage("c1", "lineage_1")
        
        results = run_lineage_tournament(self.ecosystem, self.config)
        
        self.assertEqual(results["status"], "insufficient_lineages")
    
    def test_tournament_with_two_lineages(self):
        # Setup two lineages with multiple concepts
        for i, cid in enumerate(["c1", "c2", "c3"]):
            for j in range(3):
                self.ecosystem.record_usage(f"task_{i*3+j}", cid, depth=i)
            self.ecosystem.assign_lineage(cid, "lineage_A")
        
        for i, cid in enumerate(["c4", "c5", "c6"]):
            for j in range(2):
                self.ecosystem.record_usage(f"task_{10+i*2+j}", cid, depth=i)
            self.ecosystem.assign_lineage(cid, "lineage_B")
        
        results = run_lineage_tournament(self.ecosystem, self.config)
        
        self.assertEqual(results["status"], "completed")
        self.assertIn("lineage_A", results["survivors"])
        self.assertIn("lineage_B", results["survivors"])


class TestEnforceDiversityLaw(unittest.TestCase):
    """Test main enforce_diversity_law entry point."""
    
    def test_passing_solution(self):
        ecosystem = ConceptEcosystem()
        ecosystem.total_tasks = 10
        
        passed, audit = enforce_diversity_law(
            ["concept_a", "concept_b"],
            ecosystem
        )
        
        self.assertTrue(passed)
        self.assertEqual(audit["verdict"], "PASS")
        self.assertEqual(audit["law"], "DIVERSITY_LAW")
        self.assertEqual(audit["schema_version"], 149)
    
    def test_failing_solution(self):
        ecosystem = ConceptEcosystem()
        ecosystem.total_tasks = 10
        
        # Too many repeated calls
        passed, audit = enforce_diversity_law(
            ["concept_a"] * 10,
            ecosystem
        )
        
        self.assertFalse(passed)
        self.assertEqual(audit["verdict"], "FAIL")
        self.assertGreater(len(audit["solution_check"]["violations"]), 0)
    
    def test_audit_includes_mining_status(self):
        ecosystem = ConceptEcosystem()
        ecosystem.total_tasks = 10
        
        _, audit = enforce_diversity_law(["c1"], ecosystem)
        
        self.assertIn("alternative_mining_needed", audit)


class TestDiversityCheckResult(unittest.TestCase):
    """Test DiversityCheckResult serialization."""
    
    def test_to_dict(self):
        result = DiversityCheckResult(
            passed=True,
            score=0.75,
            violations=[],
            warnings=[{"type": "TEST_WARNING"}],
        )
        
        data = result.to_dict()
        
        self.assertTrue(data["passed"])
        self.assertEqual(data["score"], 0.75)
        self.assertEqual(len(data["violations"]), 0)
        self.assertEqual(len(data["warnings"]), 1)


class TestAlternativeMiningRequest(unittest.TestCase):
    """Test AlternativeMiningRequest serialization."""
    
    def test_to_dict(self):
        request = AlternativeMiningRequest(
            trigger_reason="CONCEPT_DOMINANCE",
            dominant_concept_id="bad_concept",
            exclude_concepts={"bad_concept", "also_bad"},
            target_tasks={"task_1", "task_2", "task_3"},
        )
        
        data = request.to_dict()
        
        self.assertEqual(data["trigger_reason"], "CONCEPT_DOMINANCE")
        self.assertEqual(data["dominant_concept_id"], "bad_concept")
        self.assertEqual(len(data["exclude_concepts"]), 2)
        self.assertEqual(data["n_target_tasks"], 3)


if __name__ == "__main__":
    unittest.main()
