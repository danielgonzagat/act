from __future__ import annotations

import tempfile
import unittest
from typing import Dict, List

from atos_core.act import Act, Instruction, deterministic_iso
from atos_core.learn import KAAbsoluteTrainer, TrainConfig
from atos_core.mine_promote_v74 import materialize_deep_wrapper_act_v74
from atos_core.store import ActStore


class DeepwrapDepth3Tests(unittest.TestCase):
    def test_deepwrap_v74_can_raise_static_depth_to_3_without_ok_traces(self) -> None:
        # Build a minimal plan_validator concept chain of static depth=2:
        # leaf (depth0) <- wrap1 (depth1) <- wrap2 (depth2).
        store_base = ActStore()
        leaf = Act(
            id="concept_leaf_plan_v0",
            version=1,
            created_at=deterministic_iso(step=0),
            kind="concept_csv",
            match={},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "x", "out": "x"}),
                Instruction("CSV_RETURN", {"var": "x"}),
            ],
            evidence={
                "interface": {
                    "input_schema": {"x": "str"},
                    "output_schema": {"value": "str"},
                    "validator_id": "plan_validator",
                },
                "meta": {"birth_tags": ["plan"]},
            },
            cost={"overhead_bits": 0},
            deps=[],
            active=True,
        )
        store_base.add(leaf)
        wrap1, _ = materialize_deep_wrapper_act_v74(store_base=store_base, inner_concept_id=str(leaf.id), overhead_bits=0, seed_step=0)
        store_base.add(wrap1)
        wrap2, _ = materialize_deep_wrapper_act_v74(store_base=store_base, inner_concept_id=str(wrap1.id), overhead_bits=0, seed_step=0)
        store_base.add(wrap2)

        cfg = TrainConfig(
            steps=1,
            window=1,
            propose_every=1,
            seed=0,
            # Enable composed induction so _mine_and_promote_composed_concept_csv_v74 runs.
            concept_csv_composed_enabled=True,
            # Use a pack that enforces concept_min_depth>=3 on plan tasks.
            skill_suite_pack="sota_v8",
            # Keep deepwrap able to add at least one act per window.
            concept_csv_composed_max_new_per_window=1,
            concept_csv_overhead_bits=0,
        )
        with tempfile.TemporaryDirectory() as td:
            trainer = KAAbsoluteTrainer(data_path="unused.txt", out_dir=td, config=cfg)
            # Seed trainer store with the depth2 chain.
            trainer.store.add(leaf)
            trainer.store.add(wrap1)
            trainer.store.add(wrap2)

            meta = trainer._mine_and_promote_composed_concept_csv_v74(step=100, plan_tasks=(), force_new=False)

            # We should early-exit due to lack of ok traces, but deepwrap must still run and add a depth3 wrapper.
            self.assertTrue(bool(meta.get("enabled", False)))
            self.assertIn("deepwrap_v74", meta)
            dw = meta["deepwrap_v74"]
            self.assertEqual(int(dw.get("required_depth") or 0), 3)
            self.assertGreaterEqual(int(dw.get("added") or 0), 1)

            # Verify that trainer store now contains a concept with static depth >=3.
            memo: Dict[str, int] = {}

            def _static_depth(cid: str, stack: set) -> int:
                if cid in memo:
                    return int(memo[cid])
                if cid in stack:
                    memo[cid] = 0
                    return 0
                act = trainer.store.get_concept_act(cid)
                if act is None:
                    memo[cid] = 0
                    return 0
                callees: List[str] = []
                for ins in list(getattr(act, "program", []) or []):
                    if str(getattr(ins, "op", "")) != "CSV_CALL":
                        continue
                    args0 = getattr(ins, "args", {}) or {}
                    if not isinstance(args0, dict):
                        args0 = {}
                    callee = str(args0.get("concept_id") or "")
                    if callee:
                        callees.append(callee)
                if not callees:
                    d0 = 0
                else:
                    st2 = set(stack)
                    st2.add(cid)
                    d0 = 1 + max(_static_depth(c, st2) for c in callees)
                memo[cid] = int(d0)
                return int(d0)

            depths = [
                _static_depth(str(a.id), set())
                for a in trainer.store.by_kind("concept_csv")
                if str(getattr(a, "active", True))
            ]
            self.assertGreaterEqual(max(depths or [0]), 3)


if __name__ == "__main__":
    unittest.main()

