from __future__ import annotations

import unittest

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.discourse_variants_v119 import (
    choose_clarify_variant_v119,
    prefix2_from_text_v119,
)
from atos_core.fluency_contract_v118 import fluency_contract_v118


class TestDiscourseVariantsV119(unittest.TestCase):
    def test_choose_deterministic(self) -> None:
        ctx_sig = sha256_hex(canonical_json_dumps({"k": "ctx", "n": 1}).encode("utf-8"))
        c1 = choose_clarify_variant_v119(context_sig=ctx_sig, last_prefix2="", attempt_index=0)
        c2 = choose_clarify_variant_v119(context_sig=ctx_sig, last_prefix2="", attempt_index=0)
        self.assertEqual(c1.choice_sig, c2.choice_sig)
        self.assertEqual(c1.text, c2.text)

    def test_avoids_prefix2_collision_when_possible(self) -> None:
        ctx_sig = sha256_hex(canonical_json_dumps({"k": "ctx", "n": 2}).encode("utf-8"))
        c1 = choose_clarify_variant_v119(context_sig=ctx_sig, last_prefix2="", attempt_index=0)
        lp2 = prefix2_from_text_v119(c1.text)
        c2 = choose_clarify_variant_v119(context_sig=ctx_sig, last_prefix2=lp2, attempt_index=0)
        self.assertNotEqual(prefix2_from_text_v119(c2.text), lp2)

    def test_bullet_collision_avoided(self) -> None:
        # Find a context_sig that naturally selects a variant whose first token is "não".
        found_ctx = ""
        for i in range(500):
            ctx_sig = sha256_hex(canonical_json_dumps({"search": i}).encode("utf-8"))
            c = choose_clarify_variant_v119(context_sig=ctx_sig, last_prefix2="", attempt_index=0)
            toks = str(c.text).strip().lower().split()
            if toks and toks[0] == "não":
                found_ctx = str(ctx_sig)
                break
        self.assertTrue(found_ctx, "could not find a 'não' starter within bounded search")
        # Now force bullet-style previous prefix2, which should avoid choosing a "não ..." starter.
        c2 = choose_clarify_variant_v119(context_sig=found_ctx, last_prefix2="- não", attempt_index=0)
        toks2 = str(c2.text).strip().lower().split()
        self.assertTrue(toks2)
        self.assertNotEqual(toks2[0], "não")

    def test_ack_spam_sequence_passes_fluency_contract(self) -> None:
        transcript = []
        last_p2 = ""
        for i in range(80):
            transcript.append({"role": "user", "text": "ok"})
            ctx_sig = sha256_hex(canonical_json_dumps({"turn": i}).encode("utf-8"))
            c = choose_clarify_variant_v119(context_sig=ctx_sig, last_prefix2=last_p2, attempt_index=0)
            transcript.append({"role": "assistant", "text": c.text})
            last_p2 = prefix2_from_text_v119(c.text)
        ok, reason, _details = fluency_contract_v118(transcript_view=transcript)
        self.assertTrue(ok, reason)


if __name__ == "__main__":
    unittest.main()

