from __future__ import annotations

import unittest

from atos_core.goal_policy_v89 import (
    CandidateScoreV89,
    apply_anti_repeat_v89,
    rank_candidates_v89,
    should_abandon_goal_v89,
    should_cooldown_edge_v89,
)
from atos_core.goal_supports_v89 import SupportClaimV89, SupportStatsV89, expected_cost_smoothing_v89, posterior_mean_beta_v89


class TestSupportStatsV89(unittest.TestCase):
    def test_support_stats_posterior_mean_beta_v89(self) -> None:
        # Prior 0.5, strength 2 => alpha0=1, beta0=1.
        m0 = posterior_mean_beta_v89(prior_success=0.5, prior_strength=2, successes=0, attempts=0)
        self.assertAlmostEqual(m0, 0.5, places=9)

        # 1 success / 1 attempt => (1+1)/(1+1+1)=2/3.
        m1 = posterior_mean_beta_v89(prior_success=0.5, prior_strength=2, successes=1, attempts=1)
        self.assertAlmostEqual(m1, 2.0 / 3.0, places=9)

        # Prior 0.9, strength 2 => alpha0=1.8, beta0=0.2; 0/1 => 1.8/(2+1)=0.6
        m2 = posterior_mean_beta_v89(prior_success=0.9, prior_strength=2, successes=0, attempts=1)
        self.assertAlmostEqual(m2, 0.6, places=9)

    def test_expected_cost_smoothing_v89(self) -> None:
        # prior_cost=1.0, strength=2; cost_sum=3.0 attempts=1 => (2+3)/(3)=5/3
        c = expected_cost_smoothing_v89(prior_cost=1.0, prior_strength=2, cost_sum=3.0, attempts=1)
        self.assertAlmostEqual(c, 5.0 / 3.0, places=9)


class TestCandidateRankingV89(unittest.TestCase):
    def test_candidate_ranking_deterministic_v89(self) -> None:
        claim = SupportClaimV89(goal_id="g0", prior_success=0.5, prior_strength=2, prior_cost=1.0, note="")
        stats = SupportStatsV89(
            attempts=0,
            successes=0,
            failures=0,
            cost_sum=0.0,
            last_step=-1,
            expected_success=0.5,
            expected_cost=1.0,
        )
        c1 = CandidateScoreV89(concept_key="b", score=1.0, claim=claim, stats=stats)
        c2 = CandidateScoreV89(concept_key="a", score=1.0, claim=claim, stats=stats)
        ranked = rank_candidates_v89([c1, c2])
        self.assertEqual([c.concept_key for c in ranked], ["a", "b"])


class TestPruningV89(unittest.TestCase):
    def test_cooldown_prunes_edge_v89(self) -> None:
        stats = SupportStatsV89(
            attempts=5,
            successes=0,
            failures=5,
            cost_sum=5.0,
            last_step=10,
            expected_success=0.1,
            expected_cost=1.0,
        )
        self.assertTrue(should_cooldown_edge_v89(stats=stats))

    def test_goal_abandonment_thresholds_v89(self) -> None:
        ok, r = should_abandon_goal_v89(attempts=49, cost_total=49.9)
        self.assertFalse(ok)

        ok2, r2 = should_abandon_goal_v89(attempts=50, cost_total=0.0)
        self.assertTrue(ok2)
        self.assertEqual(r2, "max_attempts_per_goal")

        ok3, r3 = should_abandon_goal_v89(attempts=0, cost_total=50.0)
        self.assertTrue(ok3)
        self.assertEqual(r3, "max_cost_per_goal")

    def test_loop_anti_repeat_v89(self) -> None:
        edge = "edge0"
        repeats = 0
        prev_edge = ""
        # 1..3 repeats should not trigger; 4th should trigger (limit=3).
        for i in range(1, 4):
            repeats, trig = apply_anti_repeat_v89(
                prev_edge_key=prev_edge,
                prev_repeats=repeats,
                cur_edge_key=edge,
                last_attempt_ok=False,
            )
            prev_edge = edge
            self.assertFalse(trig, msg=f"unexpected trigger at i={i}")
        repeats, trig2 = apply_anti_repeat_v89(
            prev_edge_key=prev_edge,
            prev_repeats=repeats,
            cur_edge_key=edge,
            last_attempt_ok=False,
        )
        self.assertTrue(trig2)


if __name__ == "__main__":
    unittest.main()

