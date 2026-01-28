from __future__ import annotations

import tempfile
import unittest

from atos_core.act import Act, Instruction, deterministic_iso
from atos_core.learn import KAAbsoluteTrainer, TrainConfig
from atos_core.store import ActStore


class ConceptDependencyRepairTests(unittest.TestCase):
    def test_repair_reactivates_inactive_called_concept(self) -> None:
        store = ActStore()

        leaf = Act(
            id="act_concept_csv_mined_leaf_v0",
            version=1,
            created_at=deterministic_iso(step=0),
            kind="concept_csv",
            match={},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
                Instruction("CSV_RETURN", {"var": "a"}),
            ],
            evidence={
                "name": "concept_csv_mined_train_v0",
                "interface": {
                    "input_schema": {"a": "int"},
                    "output_schema": {"value": "int"},
                    "validator_id": "int_value_exact",
                },
                "meta": {"gain_bits_est": 10, "contexts_distinct": 1, "count": 1, "birth_tags": ["math"]},
            },
            cost={"overhead_bits": 1},
            deps=[],
            active=False,
        )
        store.add(leaf)

        parent = Act(
            id="concept_parent_calls_leaf_v0",
            version=1,
            created_at=deterministic_iso(step=0, offset_us=1),
            kind="concept_csv",
            match={},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
                Instruction("CSV_CALL", {"concept_id": str(leaf.id), "bind": {"a": "a"}, "out": "v0"}),
                Instruction("CSV_RETURN", {"var": "v0"}),
            ],
            evidence={
                "name": "concept_parent_v0",
                "interface": {
                    "input_schema": {"a": "int"},
                    "output_schema": {"value": "int"},
                    "validator_id": "int_value_exact",
                },
                "meta": {"birth_tags": ["math"]},
            },
            cost={"overhead_bits": 1},
            deps=[],
            active=True,
        )
        store.add(parent)

        cfg = TrainConfig(steps=1, window=1, concept_csv_budget=1)
        with tempfile.TemporaryDirectory() as td:
            trainer = KAAbsoluteTrainer(data_path="unused.txt", out_dir=td, config=cfg)
            trainer.store = store

            # Before repair, the called leaf is inactive (concept_not_found at runtime).
            self.assertFalse(bool(trainer.store.get(str(leaf.id)).active))

            meta = trainer._repair_concept_csv_dependencies(step=0)
            self.assertTrue(bool(meta.get("enabled", False)))
            self.assertGreaterEqual(int(meta.get("reactivated") or 0), 1)
            self.assertTrue(bool(trainer.store.get(str(leaf.id)).active))

            # Budget prune must not prune the required leaf (it is a dependency of an active concept).
            other = Act(
                id="act_concept_csv_mined_other_v0",
                version=1,
                created_at=deterministic_iso(step=0, offset_us=2),
                kind="concept_csv",
                match={},
                program=[
                    Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
                    Instruction("CSV_RETURN", {"var": "a"}),
                ],
                evidence={
                    "name": "concept_csv_mined_train_v0",
                    "interface": {
                        "input_schema": {"a": "int"},
                        "output_schema": {"value": "int"},
                        "validator_id": "int_value_exact",
                    },
                    "meta": {"gain_bits_est": 0, "contexts_distinct": 1, "count": 1, "birth_tags": ["math"]},
                },
                cost={"overhead_bits": 1},
                deps=[],
                active=True,
            )
            trainer.store.add(other)
            pr = trainer._concept_csv_budget_prune(step=0)
            self.assertTrue(bool(pr.get("enabled", False)))
            self.assertTrue(bool(trainer.store.get(str(leaf.id)).active))


if __name__ == "__main__":
    unittest.main()

