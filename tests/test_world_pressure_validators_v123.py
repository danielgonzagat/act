import unittest

from atos_core.world_pressure_validators_v123 import (
    EXHAUSTED_BUT_WORLD_HAS_VIABLE_CANDIDATE_V123,
    HISTORICAL_REGRESSION_WITHOUT_CAUSAL_DIFF_V123,
    IAC_MISSING_CONSEQUENCE_V123,
    IAC_MISSING_EVAL_V123,
    IAC_MISSING_GOAL_V123,
    IAC_MISSING_PLAN_V123,
    REUSE_REQUIRED_BUT_NOT_ATTEMPTED_V123,
    fail_signature_v123,
    validate_exhaustion_with_world_v123,
    validate_historical_regression_v123,
    validate_iac_v123,
    validate_reuse_required_v123,
)


class TestWorldPressureValidatorsV123(unittest.TestCase):
    def test_iac_missing_goal(self) -> None:
        ok, reason = validate_iac_v123(goal_ok=False, plan_ok=True, eval_ok=True, consequence_ok=True)
        self.assertFalse(ok)
        self.assertEqual(reason, IAC_MISSING_GOAL_V123)

    def test_iac_missing_plan(self) -> None:
        ok, reason = validate_iac_v123(goal_ok=True, plan_ok=False, eval_ok=True, consequence_ok=True)
        self.assertFalse(ok)
        self.assertEqual(reason, IAC_MISSING_PLAN_V123)

    def test_iac_missing_eval(self) -> None:
        ok, reason = validate_iac_v123(goal_ok=True, plan_ok=True, eval_ok=False, consequence_ok=True)
        self.assertFalse(ok)
        self.assertEqual(reason, IAC_MISSING_EVAL_V123)

    def test_iac_missing_consequence(self) -> None:
        ok, reason = validate_iac_v123(goal_ok=True, plan_ok=True, eval_ok=True, consequence_ok=False)
        self.assertFalse(ok)
        self.assertEqual(reason, IAC_MISSING_CONSEQUENCE_V123)

    def test_historical_regression_requires_causal_diff(self) -> None:
        ok, reason = validate_historical_regression_v123(repeated=True, world_hits_total=1, causal_diff_present=False)
        self.assertFalse(ok)
        self.assertEqual(reason, HISTORICAL_REGRESSION_WITHOUT_CAUSAL_DIFF_V123)

    def test_exhaustion_with_world_hits_fails(self) -> None:
        ok, reason = validate_exhaustion_with_world_v123(exhausted=True, world_hits_total=2)
        self.assertFalse(ok)
        self.assertEqual(reason, EXHAUSTED_BUT_WORLD_HAS_VIABLE_CANDIDATE_V123)

    def test_reuse_required_when_world_has_hits(self) -> None:
        ok, reason = validate_reuse_required_v123(world_hits_total=1, reuse_attempted=False)
        self.assertFalse(ok)
        self.assertEqual(reason, REUSE_REQUIRED_BUT_NOT_ATTEMPTED_V123)

    def test_fail_signature_deterministic(self) -> None:
        s1 = fail_signature_v123(validator_name="v", reason_code="r", context={"a": 1, "b": 2})
        s2 = fail_signature_v123(validator_name="v", reason_code="r", context={"a": 1, "b": 2})
        self.assertEqual(s1, s2)
