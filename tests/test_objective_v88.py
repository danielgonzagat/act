from __future__ import annotations

import unittest

from atos_core.goal_act_v75 import make_goal_act_v75
from atos_core.goal_objective_v88 import evaluate_goal_success_v88
from atos_core.objective_v88 import (
    ObjectiveVerdictV88,
    execute_objective_csv_v88,
    make_objective_eq_text_act_v88,
    normalize_objective_verdict_v88,
)
from atos_core.store import ActStore


class TestObjectiveVerdictV88(unittest.TestCase):
    def test_normalize_bool(self) -> None:
        v = normalize_objective_verdict_v88(True)
        self.assertEqual(v.ok, True)
        self.assertEqual(v.score, 1)
        self.assertEqual(v.reason, "")
        self.assertEqual(v.details, {})

        v2 = normalize_objective_verdict_v88(False)
        self.assertEqual(v2.ok, False)
        self.assertEqual(v2.score, 0)

    def test_normalize_dict(self) -> None:
        raw = {"ok": True, "score": 7, "reason": "x", "details": {"a": 1}}
        v = normalize_objective_verdict_v88(raw)
        self.assertEqual(v, ObjectiveVerdictV88(ok=True, score=7, reason="x", details={"a": 1}))

        # Defaults are deterministic.
        v2 = normalize_objective_verdict_v88({"ok": True})
        self.assertEqual(v2.ok, True)
        self.assertEqual(v2.score, 1)

    def test_normalize_invalid_type(self) -> None:
        v = normalize_objective_verdict_v88("oops")
        self.assertEqual(v.ok, False)
        self.assertEqual(v.score, 0)
        self.assertIn("invalid_objective_output_type", v.reason)


class TestObjectiveExecutionV88(unittest.TestCase):
    def test_execute_objective_eq_text(self) -> None:
        store = ActStore()
        obj = make_objective_eq_text_act_v88(created_step=0, match={})
        store.add(obj)

        ok_v = execute_objective_csv_v88(
            store=store,
            seed=0,
            objective_act_id=str(obj.id),
            inputs={"__output": "abc", "expected": "abc"},
            step=0,
            goal_kind="v88_demo",
        )
        self.assertTrue(ok_v.ok)
        self.assertEqual(ok_v.score, 1)

        bad_v = execute_objective_csv_v88(
            store=store,
            seed=0,
            objective_act_id=str(obj.id),
            inputs={"__output": "abc", "expected": "xyz"},
            step=0,
            goal_kind="v88_demo",
        )
        self.assertFalse(bad_v.ok)
        self.assertEqual(bad_v.score, 0)


class TestGoalIntegrationV88(unittest.TestCase):
    def test_goal_with_objective_uses_objective(self) -> None:
        store = ActStore()
        obj = make_objective_eq_text_act_v88(created_step=0, match={})
        store.add(obj)

        goal = make_goal_act_v75(
            goal_kind="v88_demo",
            bindings={"x": "0004"},
            output_key="sum",
            expected="12",
            validator_id="text_exact",
            created_step=0,
        )
        goal.evidence["goal"]["objective_act_id"] = str(obj.id)
        store.add(goal)

        verdict, dbg = evaluate_goal_success_v88(
            store=store, seed=0, goal_act=goal, goal_output="12", step=0
        )
        self.assertTrue(verdict.ok)
        self.assertTrue(bool(dbg.get("used_objective", False)))
        self.assertEqual(str(dbg.get("objective_act_id") or ""), str(obj.id))

    def test_goal_without_objective_uses_legacy(self) -> None:
        store = ActStore()
        goal = make_goal_act_v75(
            goal_kind="v88_demo",
            bindings={},
            output_key="sum",
            expected="12",
            validator_id="text_exact",
            created_step=0,
        )
        store.add(goal)

        verdict, dbg = evaluate_goal_success_v88(
            store=store, seed=0, goal_act=goal, goal_output="12", step=0
        )
        self.assertTrue(verdict.ok)
        self.assertFalse(bool(dbg.get("used_objective", True)))

        verdict2, _ = evaluate_goal_success_v88(
            store=store, seed=0, goal_act=goal, goal_output="13", step=0
        )
        self.assertFalse(verdict2.ok)


if __name__ == "__main__":
    unittest.main()

