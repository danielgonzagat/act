from __future__ import annotations

import tempfile
import unittest

from atos_core.learn import KAAbsoluteTrainer, TrainConfig


class ConceptCsvMiningTests(unittest.TestCase):
    def test_mine_and_promote_concept_csv_adds_and_prunes_to_budget(self) -> None:
        cfg = TrainConfig(
            steps=1,
            window=1,
            propose_every=1,
            seed=0,
            concept_csv_mining_enabled=True,
            concept_csv_mining_top_k=8,
            concept_csv_mining_max_new_per_window=2,
            concept_csv_budget=1,
            concept_csv_overhead_bits=0,
        )
        with tempfile.TemporaryDirectory() as td:
            trainer = KAAbsoluteTrainer(data_path="unused.txt", out_dir=td, config=cfg)
            meta = trainer._mine_and_promote_concept_csv(step=100)

            self.assertTrue(bool(meta.get("enabled", False)))
            self.assertEqual(int(meta.get("added") or 0), 2)
            # Newer trainer protects mined concepts that are required by active concept_csv deps.
            # In that case, budget pruning is infeasible and must fail-open (skip pruning).
            gc = meta.get("gc", {}) if isinstance(meta.get("gc"), dict) else {}
            self.assertEqual(int(gc.get("budget") or 0), 1)
            self.assertEqual(int(gc.get("total") or 0), 2)
            self.assertEqual(int(gc.get("pruned") or 0), 0)
            self.assertTrue(bool(gc.get("skipped", False)))
            self.assertEqual(str(gc.get("reason") or ""), "budget_infeasible_due_to_deps")

            acts = trainer.store.by_kind("concept_csv")
            self.assertEqual(len(acts), 2)
            for a in acts:
                ev = a.evidence if isinstance(a.evidence, dict) else {}
                self.assertEqual(str(ev.get("name") or ""), "concept_csv_mined_train_v0")


if __name__ == "__main__":
    unittest.main()
