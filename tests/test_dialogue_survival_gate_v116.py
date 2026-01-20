from __future__ import annotations

import unittest

from atos_core.dialogue_survival_gate_v116 import (
    DIALOGUE_SURVIVAL_REASON_CONTRADICTION_V116,
    DIALOGUE_SURVIVAL_REASON_FLUENCY_FAIL_V116,
    DIALOGUE_SURVIVAL_REASON_OK_V116,
    DIALOGUE_SURVIVAL_REASON_UNRESOLVED_REFERENCE_V116,
    decide_dialogue_survival_v116,
)


class TestDialogueSurvivalGateV116(unittest.TestCase):
    def test_pass(self) -> None:
        dec = decide_dialogue_survival_v116(
            fluency_ok=True,
            fluency_reason="ok",
            fluency_details={},
            unresolved_reference_final=0,
            contradiction_flags_total=0,
        )
        self.assertTrue(dec.ok)
        self.assertEqual(dec.reason_code, DIALOGUE_SURVIVAL_REASON_OK_V116)

    def test_fail_fluency(self) -> None:
        dec = decide_dialogue_survival_v116(
            fluency_ok=False,
            fluency_reason="most_common_reply_frac_too_high",
            fluency_details={"x": 1},
            unresolved_reference_final=0,
            contradiction_flags_total=0,
        )
        self.assertFalse(dec.ok)
        self.assertEqual(dec.reason_code, DIALOGUE_SURVIVAL_REASON_FLUENCY_FAIL_V116)

    def test_fail_unresolved_reference(self) -> None:
        dec = decide_dialogue_survival_v116(
            fluency_ok=True,
            fluency_reason="ok",
            fluency_details={},
            unresolved_reference_final=1,
            contradiction_flags_total=0,
        )
        self.assertFalse(dec.ok)
        self.assertEqual(dec.reason_code, DIALOGUE_SURVIVAL_REASON_UNRESOLVED_REFERENCE_V116)

    def test_fail_contradiction(self) -> None:
        dec = decide_dialogue_survival_v116(
            fluency_ok=True,
            fluency_reason="ok",
            fluency_details={},
            unresolved_reference_final=0,
            contradiction_flags_total=2,
        )
        self.assertFalse(dec.ok)
        self.assertEqual(dec.reason_code, DIALOGUE_SURVIVAL_REASON_CONTRADICTION_V116)

