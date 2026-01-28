from __future__ import annotations

import unittest

from atos_core.act import deterministic_iso
from atos_core.engine import Engine, EngineConfig
from atos_core.learn import _make_unigram_act
from atos_core.store import ActStore


class DecoderFluencyLayerTests(unittest.TestCase):
    def test_freeform_blocks_doc_markers_when_configured(self) -> None:
        store = ActStore()
        uni = _make_unigram_act(act_id="act_uni_test", created_at=deterministic_iso(step=0))
        # Make "<DOC>" the most likely next token unless the fluency layer blocks it.
        tbl = uni.evidence["table"][""]
        tbl["<DOC>"] = 10
        tbl["OK"] = 9
        store.add(uni)

        engine = Engine(
            store,
            seed=0,
            config=EngineConfig(
                decoder_fluency_block_token_regex=r"</?DOC>",
                decoder_fluency_block_penalty=1e6,
            ),
        )
        out = engine.generate(
            prompt="User: hi\nSystem:\n",
            max_new_tokens=1,
            mode="greedy",
            dialogue_id=0,
            turn=0,
            plan_trace=None,
        )
        self.assertEqual(list(out.get("gen_tokens") or [None])[0], "OK")

