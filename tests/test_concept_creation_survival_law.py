from __future__ import annotations

import os
import tempfile
import unittest

from atos_core.learn import KAAbsoluteTrainer, TrainConfig


class ConceptCreationSurvivalLawTests(unittest.TestCase):
    def test_concept_crisis_hard_fails_when_no_new_concepts_possible(self) -> None:
        # Minimal token stream.
        with tempfile.TemporaryDirectory() as td:
            data_path = os.path.join(td, "data.txt")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("a a a a a\n")

            # Force a deterministic plateau quickly and require concept creation to escape.
            cfg = TrainConfig(
                steps=3,
                window=1,
                propose_every=999999,  # avoid patch-eval overhead; concept crisis must be independent of propose cadence
                seed=0,
                mode="pure",
                selection_mode="survival",
                # Utility pack contains plan_validator tasks => concept_total > 0.
                utility_weight=1.0,
                skill_suite_pack="sota_v2",
                skill_suite_max_new_tokens=16,
                fluency_gen_tokens=8,
                # Plateau should trigger fast (make "improvement" impossible).
                survival_plateau_windows=1,
                survival_improve_tol=1e9,
                # Concept creation law thresholds.
                concept_csv_mining_enabled=True,
                concept_csv_mining_top_k=0,  # enabled but cannot create concepts
                concept_csv_mining_max_new_per_window=0,
                survival_concept_no_add_windows=1,
                survival_concept_reuse_stall_windows=1,
                survival_concept_hard_fail_windows=1,
            )

            trainer = KAAbsoluteTrainer(data_path=str(data_path), out_dir=str(td), config=cfg)
            with self.assertRaises(RuntimeError) as ctx:
                trainer.train()
            self.assertIn("concept crisis hard-fail", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
