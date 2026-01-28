from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

from atos_core.act import Patch
from atos_core.engine import EngineConfig
from atos_core.learn import KAAbsoluteTrainer, TrainConfig


class _FakeTrainer(KAAbsoluteTrainer):
    def __init__(self, *, out_dir: str, config: TrainConfig, eval_map: Dict[str, Dict[str, Any]]):
        super().__init__(data_path="unused.txt", out_dir=out_dir, config=config)
        self._eval_map = dict(eval_map)

    def _eval_online_window(
        self,
        store,  # noqa: ANN001
        tokens: Sequence[str],
        *,
        start: int,
        starts: Optional[Sequence[int]] = None,
        length: int,
        engine_config: EngineConfig,
        patch: Optional[Patch] = None,
    ) -> Dict[str, Any]:
        if patch is None:
            return dict(self._eval_map["base"])
        key = str(patch.payload.get("act_id") or "unknown")
        return dict(self._eval_map[key])


class RepairModeSelectionTests(unittest.TestCase):
    def test_select_patch_requires_repair_progress(self) -> None:
        cfg = TrainConfig(
            steps=100,
            val_tokens=10,
            seed=0,
            mode="demo",
            selection_mode="weighted",
            fluency_lambda=1.0,
            fluency_lambda_schedule="constant",
            repair_bottleneck_tol=1e-9,
        )

        base = {
            "nll_bits": 100.0,
            "nll_bits_windows": [100.0],
            "cost_bits": 0,
            "gen": {
                "repeat3_global": 0.0,
                "loop_rate_global": 0.5,
                "whitespace_ratio": 0.0,
                "duplicate_reply_rate": 0.0,
                "most_common_reply_frac": 0.0,
                "prefix_k_dup_rate": 0.0,
                "template_ngram_dup_rate": 0.0,
                "cross_turn_signature_repeat_rate": 0.0,
                "utility_pass_rate": 0.0,
            },
        }

        # bad keeps the bottleneck unchanged (no repair progress) but would otherwise be attractive.
        bad = {
            "nll_bits": 100.0,
            "nll_bits_windows": [100.0],
            "cost_bits": 0,
            "gen": dict(base["gen"]),
        }

        # good reduces the bottleneck below target, but has worse gain, so it would be rejected without repair.
        good = {
            "nll_bits": 110.0,
            "nll_bits_windows": [110.0],
            "cost_bits": 0,
            "gen": {**dict(base["gen"]), "loop_rate_global": 0.2},
        }

        with tempfile.TemporaryDirectory() as td:
            trainer = _FakeTrainer(out_dir=td, config=cfg, eval_map={"base": base, "bad": bad, "good": good})
            engine = SimpleNamespace(config=EngineConfig())
            tokens = ["x"] * 100
            patches = [
                Patch(kind="REWRITE_ACT", payload={"act_id": "bad", "act": {"kind": "rewrite_rule"}}),
                Patch(kind="REWRITE_ACT", payload={"act_id": "good", "act": {"kind": "rewrite_rule"}}),
            ]

            chosen = trainer._select_patch(
                step=10,
                engine=engine,
                tokens=tokens,
                patches=patches,
                divergence=False,
                repair_mode=True,
                repair_target_bottleneck=0.3,
            )
            self.assertIsNotNone(chosen)
            patch, meta = chosen or (None, {})
            self.assertEqual(patch.payload.get("act_id"), "good")
            self.assertTrue(bool(meta.get("repair_mode")))
            self.assertTrue(bool(meta.get("repair_needed")))
            self.assertAlmostEqual(float(meta.get("repair_target_bottleneck") or 0.0), 0.3, places=9)


if __name__ == "__main__":
    unittest.main()

