from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

from atos_core.act import Patch
from atos_core.engine import EngineConfig
from atos_core.learn import KAAbsoluteTrainer, TrainConfig


class _EngineStub:
    def __init__(self) -> None:
        self.config = EngineConfig()

    def vocab(self) -> list[str]:
        return [f"t{i}" for i in range(16)]


class _FakeTrainer(KAAbsoluteTrainer):
    def __init__(
        self,
        *,
        out_dir: str,
        config: TrainConfig,
        online_map: Dict[str, Dict[str, Any]],
        holdout_map: Dict[str, Dict[str, Any]],
    ):
        super().__init__(data_path="unused.txt", out_dir=out_dir, config=config)
        self._online_map = dict(online_map)
        self._holdout_map = dict(holdout_map)

    def _eval_online_window(  # type: ignore[override]
        self,
        store,  # noqa: ANN001
        tokens: Sequence[str],
        *,
        start: int,  # noqa: ARG002
        starts: Optional[Sequence[int]] = None,  # noqa: ARG002
        length: int,  # noqa: ARG002
        engine_config: EngineConfig,  # noqa: ARG002
        patch: Optional[Patch] = None,
    ) -> Dict[str, Any]:
        if patch is None:
            return dict(self._online_map["base"])
        key = str(patch.payload.get("act_id") or "unknown")
        return dict(self._online_map[key])

    def _eval_static_nll_windows(  # type: ignore[override]
        self,
        store,  # noqa: ANN001
        tokens: Sequence[str],
        *,
        start: int,  # noqa: ARG002
        starts: Optional[Sequence[int]] = None,  # noqa: ARG002
        length: int,  # noqa: ARG002
        engine_config: EngineConfig,  # noqa: ARG002
        patch: Optional[Patch] = None,
    ) -> Dict[str, Any]:
        if patch is None:
            return dict(self._holdout_map["base"])
        key = str(patch.payload.get("act_id") or "unknown")
        return dict(self._holdout_map[key])


class SurvivalSelectionTests(unittest.TestCase):
    def test_survival_selection_prefers_holdout_bottleneck_improvement(self) -> None:
        cfg = TrainConfig(
            steps=100,
            val_tokens=10,
            seed=0,
            mode="demo",
            selection_mode="survival",
            fluency_lambda=1.0,
            fluency_lambda_schedule="constant",
            holdout_frac=0.1,
            holdout_eval_windows=2,
            holdout_eval_tokens=10,
            nll_eval_windows=2,
            utility_weight=0.0,
        )

        base_gen = {
            "repeat3_global": 0.0,
            "loop_rate_global": 0.0,
            "whitespace_ratio": 0.0,
            "duplicate_reply_rate": 0.0,
            "most_common_reply_frac": 0.0,
            "prefix_k_dup_rate": 0.0,
            "template_ngram_dup_rate": 0.0,
            "cross_turn_signature_repeat_rate": 0.0,
            "utility_pass_rate": 1.0,  # keep utility out of the bottleneck
        }

        online = {
            "base": {"nll_bits": 100.0, "nll_bits_windows": [100.0, 100.0], "cost_bits": 0, "gen": dict(base_gen)},
            # Improves online NLL more, but does not improve holdout.
            "online_better": {
                "nll_bits": 90.0,
                "nll_bits_windows": [90.0, 90.0],
                "cost_bits": 0,
                "gen": dict(base_gen),
            },
            # Improves holdout (and slightly improves online NLL), should win survival selection.
            "holdout_better": {
                "nll_bits": 95.0,
                "nll_bits_windows": [95.0, 95.0],
                "cost_bits": 0,
                "gen": dict(base_gen),
            },
        }

        holdout = {
            "base": {"nll_bits": 100.0, "nll_bits_windows": [100.0, 100.0]},
            "online_better": {"nll_bits": 100.0, "nll_bits_windows": [100.0, 100.0]},
            "holdout_better": {"nll_bits": 80.0, "nll_bits_windows": [80.0, 80.0]},
        }

        with tempfile.TemporaryDirectory() as td:
            trainer = _FakeTrainer(out_dir=td, config=cfg, online_map=online, holdout_map=holdout)
            trainer._holdout_tokens = ["h"] * 100  # type: ignore[attr-defined]
            engine = _EngineStub()
            tokens = ["x"] * 100
            patches = [
                Patch(kind="REWRITE_ACT", payload={"act_id": "online_better", "act": {"kind": "predictor"}}),
                Patch(kind="REWRITE_ACT", payload={"act_id": "holdout_better", "act": {"kind": "predictor"}}),
            ]

            chosen = trainer._select_patch(step=10, engine=engine, tokens=tokens, patches=patches)
            self.assertIsNotNone(chosen)
            patch, meta = chosen or (None, {})
            self.assertEqual(patch.payload.get("act_id"), "holdout_better")
            self.assertEqual(str(meta.get("selection_mode") or ""), "survival")
            self.assertTrue(bool(meta.get("holdout_gate")))


if __name__ == "__main__":
    unittest.main()

