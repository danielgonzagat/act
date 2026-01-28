from __future__ import annotations

import tempfile
import unittest
from typing import Any, Dict, Optional

from atos_core.learn import KAAbsoluteTrainer, TrainConfig
from atos_core.suite import run_skill_suite


class _FakeEngine:
    def __init__(self, *, response: str, include_plan_trace: bool) -> None:
        self._response = str(response)
        self._include_plan_trace = bool(include_plan_trace)

    def generate(  # noqa: ANN201
        self,
        *,
        prompt: str,
        max_new_tokens: int,  # noqa: ARG002
        mode: str,  # noqa: ARG002
        dialogue_id: int,  # noqa: ARG002
        turn: int,  # noqa: ARG002
        plan_trace: Optional[Dict[str, Any]] = None,
        **kwargs: Any,  # noqa: ANN401,ARG002
    ) -> Dict[str, Any]:
        trace: Dict[str, Any] = {}
        if self._include_plan_trace and isinstance(plan_trace, dict):
            trace["plan_trace"] = dict(plan_trace)
        return {
            "text": str(prompt) + self._response,
            "trace": trace,
            "mode": "default",
            "mode_source": "router",
            "mode_policy_action": "",
            "policy_coverage": 0.0,
            "user_sig": "",
        }


class UtilityBottleneckLossTests(unittest.TestCase):
    def test_fallback_to_pass_rate_when_no_totals(self) -> None:
        cfg = TrainConfig(steps=1, window=1, propose_every=1)
        with tempfile.TemporaryDirectory() as td:
            trainer = KAAbsoluteTrainer(data_path="unused.txt", out_dir=td, config=cfg)
            comp, loss, terms = trainer._utility_bottleneck_loss({"utility_pass_rate": 0.25})
            self.assertEqual(comp, "pass_rate")
            self.assertAlmostEqual(loss, 0.75, places=9)
            self.assertIn("pass_rate_loss", terms)

    def test_bottleneck_max_across_categories(self) -> None:
        cfg = TrainConfig(steps=1, window=1, propose_every=1)
        with tempfile.TemporaryDirectory() as td:
            trainer = KAAbsoluteTrainer(data_path="unused.txt", out_dir=td, config=cfg)
            gen = {
                "utility_instruction_total": 2,
                "utility_instruction_pass_rate": 1.0,
                "utility_json_total": 5,
                "utility_json_pass_rate": 0.6,  # loss 0.4
                "utility_math_total": 0,
                "utility_state_total": 1,
                "utility_state_pass_rate": 0.9,  # loss 0.1
                "utility_goals_total": 0,
            }
            comp, loss, terms = trainer._utility_bottleneck_loss(gen)
            self.assertEqual(comp, "json")
            self.assertAlmostEqual(loss, 0.4, places=9)
            self.assertAlmostEqual(float(terms.get("json_loss") or 0.0), 0.4, places=9)

    def test_plan_trace_missing_included_when_plan_tasks_present(self) -> None:
        cfg = TrainConfig(steps=1, window=1, propose_every=1)
        with tempfile.TemporaryDirectory() as td:
            trainer = KAAbsoluteTrainer(data_path="unused.txt", out_dir=td, config=cfg)
            gen = {
                "utility_plan_total": 1,
                "utility_plan_pass_rate": 1.0,
                "utility_plan_trace_turns_total": 10,
                "utility_plan_trace_missing_turns": 2,
                "utility_instruction_total": 1,
                "utility_instruction_pass_rate": 1.0,
            }
            comp, loss, _terms = trainer._utility_bottleneck_loss(gen)
            self.assertEqual(comp, "plan_trace_missing")
            self.assertAlmostEqual(loss, 0.2, places=9)

    def test_bottleneck_includes_clarify_and_consistency_categories(self) -> None:
        cfg = TrainConfig(steps=1, window=1, propose_every=1)
        with tempfile.TemporaryDirectory() as td:
            trainer = KAAbsoluteTrainer(data_path="unused.txt", out_dir=td, config=cfg)
            gen = {
                "utility_instruction_total": 1,
                "utility_instruction_pass_rate": 1.0,
                "utility_clarify_total": 2,
                "utility_clarify_pass_rate": 0.5,  # loss 0.5
                "utility_consistency_total": 2,
                "utility_consistency_pass_rate": 0.0,  # loss 1.0 (worst)
            }
            comp, loss, terms = trainer._utility_bottleneck_loss(gen)
            self.assertEqual(comp, "consistency")
            self.assertAlmostEqual(loss, 1.0, places=9)
            self.assertAlmostEqual(
                float(terms.get("consistency_loss") or 0.0), 1.0, places=9
            )

    def test_bottleneck_includes_memory_category(self) -> None:
        cfg = TrainConfig(steps=1, window=1, propose_every=1)
        with tempfile.TemporaryDirectory() as td:
            trainer = KAAbsoluteTrainer(data_path="unused.txt", out_dir=td, config=cfg)
            gen = {
                "utility_instruction_total": 1,
                "utility_instruction_pass_rate": 1.0,
                "utility_memory_total": 2,
                "utility_memory_pass_rate": 0.0,  # loss 1.0 (worst)
            }
            comp, loss, terms = trainer._utility_bottleneck_loss(gen)
            self.assertEqual(comp, "memory")
            self.assertAlmostEqual(loss, 1.0, places=9)
            self.assertAlmostEqual(float(terms.get("memory_loss") or 0.0), 1.0, places=9)


class SkillSuiteTotalsTests(unittest.TestCase):
    def test_run_skill_suite_emits_totals_and_plan_counts(self) -> None:
        engine = _FakeEngine(response="OK", include_plan_trace=False)
        tasks = (
            {
                "task_id": "t1",
                "dialogue": ("Responda exatamente: OK",),
                "validator_id": "exact_match",
                "expected_spec": {"text": "OK", "collapse_ws": True, "case_sensitive": True},
                "tags": ["instruction"],
            },
            {
                "task_id": "t2",
                "dialogue": ("Responda exatamente: OK",),
                "validator_id": "exact_match",
                "expected_spec": {"text": "OK", "collapse_ws": True, "case_sensitive": True},
                "tags": ["plan", "json"],
            },
            {
                "task_id": "t3",
                "dialogue": ("Responda exatamente: OK",),
                "validator_id": "exact_match",
                "expected_spec": {"text": "OK", "collapse_ws": True, "case_sensitive": True},
                "tags": ["clarify"],
            },
            {
                "task_id": "t4",
                "dialogue": ("Responda exatamente: OK",),
                "validator_id": "exact_match",
                "expected_spec": {"text": "OK", "collapse_ws": True, "case_sensitive": True},
                "tags": ["consistency"],
            },
        )

        _transcripts, metrics = run_skill_suite(engine, tasks=tasks, max_new_tokens=8)
        self.assertEqual(int(metrics.get("total_tasks") or 0), 4)
        self.assertEqual(int(metrics.get("instruction_total") or 0), 1)
        self.assertEqual(int(metrics.get("plan_total") or 0), 1)
        self.assertEqual(int(metrics.get("json_total") or 0), 1)
        self.assertEqual(int(metrics.get("clarify_total") or 0), 1)
        self.assertEqual(int(metrics.get("consistency_total") or 0), 1)
        self.assertAlmostEqual(float(metrics.get("plan_pass_rate") or 0.0), 1.0, places=9)
        self.assertGreaterEqual(int(metrics.get("plan_trace_missing_turns") or 0), 1)


if __name__ == "__main__":
    unittest.main()
