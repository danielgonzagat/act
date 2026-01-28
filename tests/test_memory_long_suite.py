from __future__ import annotations

import os
import tempfile
import unittest

from atos_core.engine import Engine, EngineConfig
from atos_core.learn import KAAbsoluteTrainer, TrainConfig
from atos_core.suite import SKILL_DIALOGUES_V3_MEMORY_LONG, run_skill_suite


class MemoryLongSuiteTests(unittest.TestCase):
    def test_memory_long_store_and_recall_pass_with_concept_and_memory(self) -> None:
        cfg = TrainConfig(steps=1, window=1, propose_every=1)
        with tempfile.TemporaryDirectory() as td:
            data_path = os.path.join(td, "data.txt")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("x")
            trainer = KAAbsoluteTrainer(data_path=data_path, out_dir=td, config=cfg)
            trainer._init_acts()
            engine = Engine(trainer.store, seed=0, config=EngineConfig(enable_contracts=False))
            _t, metrics = run_skill_suite(engine, tasks=SKILL_DIALOGUES_V3_MEMORY_LONG, max_new_tokens=16)
            self.assertEqual(int(metrics.get("total_tasks") or 0), 2)
            self.assertEqual(int(metrics.get("pass_count") or 0), 2)
            self.assertAlmostEqual(float(metrics.get("pass_rate") or 0.0), 1.0, places=9)
            self.assertEqual(int(metrics.get("memory_total") or 0), 2)
            self.assertAlmostEqual(float(metrics.get("memory_pass_rate") or 0.0), 1.0, places=9)
            self.assertEqual(int(metrics.get("concept_total") or 0), 2)
            self.assertAlmostEqual(float(metrics.get("concept_pass_rate") or 0.0), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()

