from __future__ import annotations

import os
import tempfile

from atos_core.learn import KAAbsoluteTrainer, TrainConfig


def test_full_plan_miner_materializes_call_based_concepts_with_deepwrapped_plan_ops() -> None:
    cfg = TrainConfig(
        steps=1,
        window=1,
        propose_every=1,
        seed=0,
        # Enable concept mining; full-plan materialization happens during mining when plan packs are present.
        concept_csv_mining_enabled=True,
        # Keep overhead low for test speed; semantics enforced elsewhere.
        concept_csv_overhead_bits=0,
        # Use the suite pack that requires deep plan concepts + non-trivial CSG.
        skill_suite_pack="sota_v12",
    )
    # Optional knobs are read via getattr; TrainConfig may not declare them as dataclass fields.
    setattr(cfg, "concept_csv_mining_full_plan_max_new_per_window", 8)
    setattr(cfg, "concept_csv_plan_op_deepwrap_max_new_per_window", 64)

    with tempfile.TemporaryDirectory() as td:
        data_path = os.path.join(td, "data.txt")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write("x\n")

        trainer = KAAbsoluteTrainer(data_path=str(data_path), out_dir=td, config=cfg)
        trainer._init_acts()
        trainer._mine_and_promote_concept_csv(step=100)

        full_calls = []
        for a in trainer.store.by_kind("concept_csv"):
            ev = a.evidence if isinstance(getattr(a, "evidence", None), dict) else {}
            meta = ev.get("meta") if isinstance(ev.get("meta"), dict) else {}
            if str(meta.get("builder") or "") != "concept_csv_full_plan_calls_v2":
                continue
            full_calls.append(a)

        assert full_calls, "expected at least one full-plan call-based concept_csv"

        a0 = full_calls[0]
        ops = [str(getattr(ins, "op", "") or "") for ins in list(getattr(a0, "program", []) or [])]
        assert "CSV_CALL" in ops
        assert "CSV_PRIMITIVE" not in ops

        deps = [str(x) for x in list(getattr(a0, "deps", []) or []) if str(x)]
        assert any(d.startswith("concept_v74_deepwrap_") for d in deps)
