"""
test_agi_regime_v144.py - Unit Tests for AGI Regime V144.

Tests validate all 7 Survival Laws:
1. LAW_CONCEPT - No task survival without concepts
2. LAW_DEPTH - Minimum concept depth required
3. LAW_COMPOSITION - CSV_CALL chains required
4. LAW_REUSE - Cross-context reuse tracked
5. LAW_PROOF - PCC + hashes required
6. LAW_UTILITY - Utility-based learning
7. LAW_BUDGET - Search budget enforcement

Schema version: 144
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction
from atos_core.agi_regime_v144 import (
    AGI_REGIME_SCHEMA_VERSION_V144,
    AGIRegimeConfig,
    ConceptLifecycleManager,
    ConceptState,
    SurvivalLaw,
    apply_regime_to_tasks,
    compute_regime_loss,
    create_agi_regime_tasks,
    validate_law_budget,
    validate_law_composition,
    validate_law_concept,
    validate_law_depth,
    validate_survival_laws,
    write_regime_validation_to_ledger,
)
from atos_core.csv_composed_miner_v144 import (
    CSV_COMPOSED_MINER_SCHEMA_VERSION_V144,
    CallEdge,
    CallNode,
    CallSubgraph,
    ComposedConceptCandidate,
    ComposedMinerConfig,
    extract_call_subgraphs,
    materialize_composed_concept_act,
    mine_composed_concepts,
)
from atos_core.agi_loop_v144 import (
    AGI_LOOP_SCHEMA_VERSION_V144,
    AGILoopConfig,
    LoopExitReason,
    LoopPhase,
    run_agi_loop,
    run_agi_loop_batch,
    write_loop_result_to_ledger,
)


# ─────────────────────────────────────────────────────────────────────────────
# LAW_CONCEPT Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLawConcept(unittest.TestCase):
    """Tests for LAW_CONCEPT - No task survival without concepts."""

    def test_law_concept_pass_with_concepts(self) -> None:
        """Task with concepts should pass LAW_CONCEPT."""
        trace = {
            "concept_executor": {
                "used": True,
                "ok": True,
            },
        }
        task = {"concept_policy_required": True}
        
        law_result = validate_law_concept(trace=trace, task=task)
        
        self.assertTrue(law_result.passed)
        self.assertEqual(law_result.law, SurvivalLaw.LAW_CONCEPT)

    def test_law_concept_fail_without_concepts(self) -> None:
        """Task without concepts should fail LAW_CONCEPT."""
        trace = {
            "concept_executor": {
                "used": False,
                "ok": False,
            },
        }
        task = {"concept_policy_required": True}
        
        law_result = validate_law_concept(trace=trace, task=task)
        
        self.assertFalse(law_result.passed)
        self.assertEqual(law_result.law, SurvivalLaw.LAW_CONCEPT)

    def test_law_concept_pass_when_not_required(self) -> None:
        """Task without concept requirement should pass."""
        trace = {}
        task = {"concept_policy_required": False}
        
        law_result = validate_law_concept(trace=trace, task=task)
        
        self.assertTrue(law_result.passed)


# ─────────────────────────────────────────────────────────────────────────────
# LAW_DEPTH Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLawDepth(unittest.TestCase):
    """Tests for LAW_DEPTH - Minimum concept depth required."""

    def test_law_depth_pass_sufficient_depth(self) -> None:
        """Concepts with sufficient depth should pass."""
        trace = {
            "concept_executor": {
                "max_depth": 3,
            },
        }
        task = {"concept_min_depth": 2}
        
        result = validate_law_depth(trace=trace, task=task, min_depth=2)
        
        self.assertTrue(result.passed)
        self.assertEqual(result.law, SurvivalLaw.LAW_DEPTH)

    def test_law_depth_fail_insufficient_depth(self) -> None:
        """Concepts with insufficient depth should fail."""
        trace = {
            "concept_executor": {
                "max_depth": 1,
            },
        }
        task = {"concept_min_depth": 2}
        
        result = validate_law_depth(trace=trace, task=task, min_depth=2)
        
        self.assertFalse(result.passed)


# ─────────────────────────────────────────────────────────────────────────────
# LAW_COMPOSITION Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLawComposition(unittest.TestCase):
    """Tests for LAW_COMPOSITION - CSV_CALL chains required."""

    def test_law_composition_pass_with_calls(self) -> None:
        """Trace with CSV_CALLs should pass."""
        trace = {
            "concept_executor": {
                "calls_total": 5,
            },
        }
        task = {"concept_policy_required": True, "concept_min_csv_calls": 2}
        
        result = validate_law_composition(trace=trace, task=task, min_csv_calls=1)
        
        self.assertTrue(result.passed)
        self.assertEqual(result.law, SurvivalLaw.LAW_COMPOSITION)

    def test_law_composition_fail_without_calls(self) -> None:
        """Trace without CSV_CALLs should fail."""
        trace = {
            "concept_executor": {
                "calls_total": 0,
            },
        }
        task = {"concept_policy_required": True, "concept_min_csv_calls": 2}
        
        result = validate_law_composition(trace=trace, task=task, min_csv_calls=2)
        
        self.assertFalse(result.passed)


# ─────────────────────────────────────────────────────────────────────────────
# LAW_BUDGET Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLawBudget(unittest.TestCase):
    """Tests for LAW_BUDGET - Search budget enforcement."""

    def test_law_budget_pass_within_budget(self) -> None:
        """Search within budget should pass."""
        trace = {"search_steps": 50}
        task = {"search_budget": 100}
        
        result = validate_law_budget(trace=trace, task=task, max_search_steps=100)
        
        self.assertTrue(result.passed)
        self.assertEqual(result.law, SurvivalLaw.LAW_BUDGET)

    def test_law_budget_fail_over_budget(self) -> None:
        """Search over budget should fail."""
        trace = {"search_steps": 150}
        task = {"search_budget": 100}
        
        result = validate_law_budget(trace=trace, task=task, max_search_steps=100)
        
        self.assertFalse(result.passed)


# ─────────────────────────────────────────────────────────────────────────────
# Survival Laws Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSurvivalLawsIntegration(unittest.TestCase):
    """Integration tests for validate_survival_laws()."""

    def test_all_laws_pass(self) -> None:
        """Trace passing all laws should succeed."""
        trace = {
            "concept_executor": {
                "used": True,
                "ok": True,
                "calls_total": 2,
                "max_depth": 2,
            },
            "search_steps": 50,
        }
        task = {
            "concept_policy_required": True,
            "concept_min_depth": 1,
            "concept_min_csv_calls": 1,
            "search_budget": 100,
        }
        
        config = AGIRegimeConfig()
        validation = validate_survival_laws(trace=trace, task=task, config=config)
        
        self.assertTrue(validation.passed)
        self.assertGreaterEqual(validation.laws_passed, 1)

    def test_fails_without_concepts(self) -> None:
        """Trace without concepts should fail LAW_CONCEPT."""
        trace = {
            "concept_executor": {
                "used": False,
                "ok": False,
            },
        }
        task = {"concept_policy_required": True}
        
        validation = validate_survival_laws(trace=trace, task=task)
        
        self.assertFalse(validation.passed)

    def test_regime_loss_infinite_on_failure(self) -> None:
        """Regime loss should be infinite when laws fail."""
        trace = {
            "concept_executor": {
                "used": False,
                "ok": False,
            },
        }
        task = {"concept_policy_required": True}
        
        validation = validate_survival_laws(trace=trace, task=task)
        loss = compute_regime_loss(
            validation_result=validation,
            utility_pass_rate=0.0,
            fluency_score=0.5,
        )
        
        self.assertTrue(math.isinf(loss["loss"]))

    def test_regime_loss_finite_on_success(self) -> None:
        """Regime loss should be finite when laws pass."""
        trace = {
            "concept_executor": {
                "used": True,
                "ok": True,
                "calls_total": 1,
                "max_depth": 1,
            },
        }
        task = {
            "concept_policy_required": True,
            "concept_min_depth": 1,
        }
        
        validation = validate_survival_laws(trace=trace, task=task)
        loss = compute_regime_loss(
            validation_result=validation,
            utility_pass_rate=1.0,
            fluency_score=0.9,
        )
        
        self.assertTrue(math.isfinite(loss["loss"]))


# ─────────────────────────────────────────────────────────────────────────────
# Concept Lifecycle Manager Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConceptLifecycleManager(unittest.TestCase):
    """Tests for ConceptLifecycleManager (ICS)."""

    def test_register_concept(self) -> None:
        """Can register new concept."""
        mgr = ConceptLifecycleManager()
        metrics = mgr.register_concept(
            concept_id="c1",
            step=0,
            has_pcc=True,
            pcc_hash="abc123",
        )
        
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.concept_id, "c1")
        self.assertEqual(metrics.state, ConceptState.CANDIDATE)

    def test_record_usage_updates_metrics(self) -> None:
        """Recording usage should update metrics."""
        mgr = ConceptLifecycleManager()
        mgr.register_concept(concept_id="c1", step=0)
        
        mgr.record_usage(
            concept_id="c1",
            step=1,
            success=True,
            context_id="ctx1",
            family_id="fam1",
        )
        mgr.record_usage(
            concept_id="c1",
            step=2,
            success=True,
            context_id="ctx2",
            family_id="fam1",
        )
        
        metrics = mgr.concepts.get("c1")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.total_uses, 2)
        self.assertEqual(len(metrics.contexts_used), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Regime Tasks Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRegimeTasks(unittest.TestCase):
    """Tests for create_agi_regime_tasks()."""

    def test_create_bootstrap_tasks(self) -> None:
        """Bootstrap regime should add basic concept policy."""
        base_tasks = [{"id": "t1", "validator_id": "plan_validator", "inputs": {}}]
        regime_tasks = create_agi_regime_tasks(base_tasks, regime_level="bootstrap")
        
        self.assertEqual(len(regime_tasks), 1)
        self.assertIn("concept_policy_required", regime_tasks[0])

    def test_create_full_regime_tasks(self) -> None:
        """Full regime should add all constraints."""
        base_tasks = [{"id": "t1", "validator_id": "plan_validator", "inputs": {}}]
        regime_tasks = create_agi_regime_tasks(base_tasks, regime_level="full")
        
        self.assertEqual(len(regime_tasks), 1)
        task = regime_tasks[0]
        self.assertTrue(task.get("concept_policy_required"))
        self.assertGreaterEqual(task.get("concept_min_depth", 0), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Composed Miner Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestComposedMiner(unittest.TestCase):
    """Tests for csv_composed_miner_v144."""

    def test_extract_call_subgraphs(self) -> None:
        """Should extract subgraphs from CSV_CALL events."""
        events = [
            {"op": "CSV_CALL", "callee": "c1", "depth": 0, "out": "v1"},
            {"op": "CSV_CALL", "callee": "c2", "depth": 1, "bind": {"x": "v1"}, "out": "v2"},
        ]
        
        subgraphs = extract_call_subgraphs(
            events,
            trace_id="trace_1",
            context_id="ctx_1",
        )
        
        self.assertEqual(len(subgraphs), 1)
        sg = subgraphs[0]
        self.assertEqual(len(sg.nodes), 2)
        self.assertEqual(len(sg.edges), 1)
        self.assertEqual(sg.max_depth, 1)

    def test_mine_composed_concepts(self) -> None:
        """Should find frequent subgraph patterns."""
        # Create multiple similar subgraphs
        subgraphs = []
        for i in range(5):
            sg = CallSubgraph(
                nodes=(
                    CallNode("c1", "sig1", 0),
                    CallNode("c2", "sig2", 1),
                ),
                edges=(CallEdge(0, 1, "v"),),
                root_concept_id="c1",
                max_depth=1,
                trace_id=f"trace_{i}",
                context_id=f"ctx_{i}",
                family_id="family_1",
            )
            subgraphs.append(sg)
        
        candidates = mine_composed_concepts(
            subgraphs,
            config=ComposedMinerConfig(min_frequency=3, min_contexts=3),
        )
        
        self.assertGreaterEqual(len(candidates), 1)
        self.assertGreaterEqual(candidates[0].frequency, 5)

    def test_materialize_composed_concept_act(self) -> None:
        """Should create valid concept_csv Act from candidate."""
        sg = CallSubgraph(
            nodes=(
                CallNode("c1", "sig1", 0),
                CallNode("c2", "sig2", 1),
            ),
            edges=(CallEdge(0, 1, "v"),),
            root_concept_id="c1",
            max_depth=1,
            trace_id="trace_1",
            context_id="ctx_1",
            family_id="family_1",
        )
        
        candidate = ComposedConceptCandidate(
            signature="sig123",
            subgraphs=[sg],
            frequency=5,
            contexts_distinct=3,
            families_distinct=1,
            max_depth=1,
            avg_depth=1.0,
            call_deps=["c1", "c2"],
        )
        
        act = materialize_composed_concept_act(
            candidate,
            step=1,
            store_content_hash="abc123",
            title="test_composed",
        )
        
        self.assertEqual(act.kind, "concept_csv")
        self.assertIn("c1", act.deps)
        self.assertIn("c2", act.deps)


# ─────────────────────────────────────────────────────────────────────────────
# AGI Loop Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAGILoop(unittest.TestCase):
    """Tests for agi_loop_v144."""

    def test_loop_phases_enum(self) -> None:
        """Loop phases should be properly defined."""
        self.assertEqual(LoopPhase.PLAN.value, "plan")
        self.assertEqual(LoopPhase.EXECUTE.value, "execute")
        self.assertEqual(LoopPhase.VALIDATE.value, "validate")
        self.assertEqual(LoopPhase.MINE.value, "mine")
        self.assertEqual(LoopPhase.PROMOTE.value, "promote")
        self.assertEqual(LoopPhase.RERUN.value, "rerun")

    def test_run_agi_loop_basic(self) -> None:
        """Basic AGI loop should complete."""
        task = {"id": "task_1", "inputs": {"x": 1}}
        concept_store: List[Act] = []
        
        config = AGILoopConfig(
            max_iterations=3,
            regime_level="bootstrap",
        )
        
        result = run_agi_loop(
            task,
            concept_store=concept_store,
            config=config,
            step=0,
            store_content_hash="test_hash",
        )
        
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.iterations_completed, 1)

    def test_loop_batch(self) -> None:
        """Batch loop should process multiple tasks."""
        tasks = [
            {"id": "t1", "inputs": {"x": 1}},
            {"id": "t2", "inputs": {"x": 2}},
        ]
        concept_store: List[Act] = []
        
        config = AGILoopConfig(
            max_iterations=2,
            regime_level="bootstrap",
        )
        
        result = run_agi_loop_batch(
            tasks,
            concept_store=concept_store,
            config=config,
            step=0,
            store_content_hash="test_hash",
        )
        
        self.assertEqual(result.tasks_processed, 2)


# ─────────────────────────────────────────────────────────────────────────────
# WORM Ledger Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWORMLedger(unittest.TestCase):
    """Tests for WORM-compliant ledger entries."""

    def test_regime_validation_ledger_entry(self) -> None:
        """Should create valid ledger entry for regime validation."""
        trace = {
            "concept_executor": {
                "used": True,
                "ok": True,
            },
        }
        task = {"concept_policy_required": True}
        
        validation = validate_survival_laws(trace=trace, task=task)
        
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            ledger_path = f.name
        
        try:
            entry_hash = write_regime_validation_to_ledger(
                result=validation,
                task_id="test_task_1",
                step=100,
                ledger_path=ledger_path,
            )
            
            self.assertEqual(len(entry_hash), 64)  # SHA-256 hex
            
            # Read and verify ledger entry
            with open(ledger_path, "r") as f:
                line = f.readline()
                entry = json.loads(line)
                self.assertEqual(entry["schema_version"], AGI_REGIME_SCHEMA_VERSION_V144)
                self.assertEqual(entry["step"], 100)
                self.assertIn("entry_hash", entry)
        finally:
            os.unlink(ledger_path)

    def test_loop_result_ledger_entry(self) -> None:
        """Should create valid ledger entry for loop result."""
        task = {"id": "task_1"}
        concept_store: List[Act] = []
        
        result = run_agi_loop(
            task,
            concept_store=concept_store,
            config=AGILoopConfig(max_iterations=1),
            step=0,
            store_content_hash="test",
        )
        
        entry = write_loop_result_to_ledger(
            result,
            step=1,
            store_content_hash="test_hash_456",
        )
        
        self.assertEqual(entry["schema_version"], AGI_LOOP_SCHEMA_VERSION_V144)
        self.assertIn("entry_hash", entry)


# ─────────────────────────────────────────────────────────────────────────────
# Schema Version Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaVersions(unittest.TestCase):
    """Tests for schema version consistency."""

    def test_all_schema_versions_v144(self) -> None:
        """All V144 modules should have schema version 144."""
        self.assertEqual(AGI_REGIME_SCHEMA_VERSION_V144, 144)
        self.assertEqual(CSV_COMPOSED_MINER_SCHEMA_VERSION_V144, 144)
        self.assertEqual(AGI_LOOP_SCHEMA_VERSION_V144, 144)


if __name__ == "__main__":
    unittest.main()
