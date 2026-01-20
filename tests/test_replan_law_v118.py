from __future__ import annotations

import unittest
from typing import List, Optional

from atos_core.replan_law_v118 import (
    PlanCandidateV118,
    REPLAN_REASON_EXHAUSTED_PLANS_V118,
    REPLAN_REASON_OK_V118,
    REPLAN_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V118,
    replan_until_satisfies_v118,
)


class TestReplanLawV118(unittest.TestCase):
    def test_first_fails_second_passes(self) -> None:
        plans: List[PlanCandidateV118] = [
            PlanCandidateV118(plan_id="planA_direct_answer", plan_cost=1.0, plan_sem_sig={"k": "A"}),
            PlanCandidateV118(plan_id="planB_ask_clarify", plan_cost=1.0, plan_sem_sig={"k": "B"}),
        ]

        def next_plan() -> Optional[PlanCandidateV118]:
            return plans.pop(0) if plans else None

        def exec_plan(p: PlanCandidateV118):
            if p.plan_id == "planA_direct_answer":
                return False, True, "direct_answer_failed"
            return True, True, ""

        res = replan_until_satisfies_v118(next_plan=next_plan, exec_plan=exec_plan, max_attempts=8)
        self.assertTrue(res.ok)
        self.assertEqual(res.reason, REPLAN_REASON_OK_V118)
        self.assertEqual(res.attempts_total, 2)
        self.assertEqual(res.attempts[-1].plan_id, "planB_ask_clarify")
        self.assertTrue(res.attempts[-1].eval_satisfies)
        self.assertTrue(res.attempts[-1].dialogue_survival_ok)

    def test_exhausted_plans(self) -> None:
        plans: List[PlanCandidateV118] = [
            PlanCandidateV118(plan_id="planA", plan_cost=1.0, plan_sem_sig={"k": "A"}),
        ]

        def next_plan() -> Optional[PlanCandidateV118]:
            return plans.pop(0) if plans else None

        def exec_plan(_p: PlanCandidateV118):
            return False, True, "failed"

        res = replan_until_satisfies_v118(next_plan=next_plan, exec_plan=exec_plan, max_attempts=8)
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, REPLAN_REASON_EXHAUSTED_PLANS_V118)
        self.assertEqual(res.attempts_total, 1)

    def test_budget_exhausted(self) -> None:
        plans: List[PlanCandidateV118] = [
            PlanCandidateV118(plan_id="planA", plan_cost=1.0, plan_sem_sig={"k": "A"}),
            PlanCandidateV118(plan_id="planB", plan_cost=1.0, plan_sem_sig={"k": "B"}),
            PlanCandidateV118(plan_id="planC", plan_cost=1.0, plan_sem_sig={"k": "C"}),
            PlanCandidateV118(plan_id="planD", plan_cost=1.0, plan_sem_sig={"k": "D"}),
        ]

        def next_plan() -> Optional[PlanCandidateV118]:
            return plans.pop(0) if plans else None

        def exec_plan(_p: PlanCandidateV118):
            return False, True, "failed"

        res = replan_until_satisfies_v118(next_plan=next_plan, exec_plan=exec_plan, max_attempts=3)
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, REPLAN_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V118)
        self.assertEqual(res.attempts_total, 3)


if __name__ == "__main__":
    unittest.main()

