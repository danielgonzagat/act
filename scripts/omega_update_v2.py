#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.omega_v2 import OmegaParamsV2, OmegaStateV2, update_omega_state_v2


def main() -> int:
    ap = argparse.ArgumentParser(description="Ω update v2 (destructive future bans + promoted concept/operator gate).")
    ap.add_argument("--prev_state", default="", help="Previous omega_state_v2.json (optional).")
    ap.add_argument("--run_dir", required=True, help="ARC run dir containing omega_events_v2.jsonl.")
    ap.add_argument(
        "--events_path",
        default="",
        help="Optional override for events jsonl path (must live under --run_dir; default: <run_dir>/omega_events_v2.jsonl).",
    )
    ap.add_argument("--concept_bank", default="", help="Concept bank jsonl used for promotion bookkeeping (optional).")
    ap.add_argument("--macro_bank", default="", help="Macro/operator bank jsonl used for operator promotion bookkeeping (optional).")
    ap.add_argument(
        "--induction_log",
        default="",
        help="Optional concept induction log jsonl (from arc_induce_concept_templates_v146.py --induction_log).",
    )
    ap.add_argument("--base_max_depth", type=int, default=4, help="Baseline max_depth (used only if prev_state is empty).")
    ap.add_argument("--base_max_programs", type=int, default=4000, help="Baseline max_programs (used only if prev_state is empty).")
    ap.add_argument("--out", required=True, help="WORM output omega_state_v2.json path.")

    ap.add_argument("--cluster_k", type=int, default=3)
    ap.add_argument("--cluster_cooldown_runs", type=int, default=1)
    ap.add_argument("--cluster_max_induce_attempts", type=int, default=3)

    ap.add_argument("--min_max_programs", type=int, default=200)
    ap.add_argument("--burn_programs_per_dead_cluster", type=int, default=50)
    ap.add_argument("--min_max_depth", type=int, default=2)
    ap.add_argument("--burn_depth_every_n_dead_clusters", type=int, default=10)

    ap.add_argument("--promote_min_support", type=int, default=2)
    ap.add_argument("--promote_min_used_solved", type=int, default=2)
    ap.add_argument("--promote_min_contexts", type=int, default=2)
    ap.add_argument(
        "--require_concept_call_after_runs",
        type=int,
        default=0,
        help="After this run index (1-based), treat solved episodes without any concept_call as FAIL for Ω.",
    )
    ap.add_argument(
        "--require_promoted_concept_call_after_runs",
        type=int,
        default=0,
        help="After this run index (1-based), if any promoted concepts exist, require solved episodes to use one.",
    )

    args = ap.parse_args()

    run_dir = Path(str(args.run_dir)).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"missing_run_dir:{run_dir}")
    events_path = (
        Path(str(args.events_path)).resolve() if str(args.events_path).strip() else (run_dir / "omega_events_v2.jsonl")
    )
    if events_path.parent != run_dir:
        raise SystemExit(f"events_path_must_be_under_run_dir:events={events_path},run_dir={run_dir}")
    if not events_path.is_file():
        raise SystemExit(f"missing_omega_events_v2:{events_path}")

    prev_state_path = Path(str(args.prev_state)).resolve() if str(args.prev_state).strip() else None
    prev_state = OmegaStateV2.from_path(prev_state_path) if prev_state_path is not None else None

    concept_bank_path = Path(str(args.concept_bank)).resolve() if str(args.concept_bank).strip() else None
    if concept_bank_path is not None and not concept_bank_path.is_file():
        raise SystemExit(f"missing_concept_bank:{concept_bank_path}")

    macro_bank_path = Path(str(args.macro_bank)).resolve() if str(args.macro_bank).strip() else None
    if macro_bank_path is not None and not macro_bank_path.is_file():
        raise SystemExit(f"missing_macro_bank:{macro_bank_path}")

    induction_log_path = Path(str(args.induction_log)).resolve() if str(args.induction_log).strip() else None
    if induction_log_path is not None and not induction_log_path.is_file():
        raise SystemExit(f"missing_induction_log:{induction_log_path}")

    params = OmegaParamsV2(
        cluster_k=int(args.cluster_k),
        cluster_cooldown_runs=int(args.cluster_cooldown_runs),
        cluster_max_induce_attempts=int(args.cluster_max_induce_attempts),
        min_max_programs=int(args.min_max_programs),
        burn_programs_per_dead_cluster=int(args.burn_programs_per_dead_cluster),
        min_max_depth=int(args.min_max_depth),
        burn_depth_every_n_dead_clusters=int(args.burn_depth_every_n_dead_clusters),
        promote_min_support=int(args.promote_min_support),
        promote_min_used_solved=int(args.promote_min_used_solved),
        promote_min_contexts=int(args.promote_min_contexts),
        require_concept_call_after_runs=int(args.require_concept_call_after_runs),
        require_promoted_concept_call_after_runs=int(args.require_promoted_concept_call_after_runs),
    )

    st, info = update_omega_state_v2(
        prev=prev_state,
        events_path=events_path,
        concept_bank_path=concept_bank_path,
        macro_bank_path=macro_bank_path,
        induction_log_path=induction_log_path,
        base_max_depth=int(args.base_max_depth),
        base_max_programs=int(args.base_max_programs),
        params=params if prev_state is None else None,
    )

    out_path = Path(str(args.out)).resolve()
    st.write_worm(out_path)
    print(json.dumps({"ok": True, "out": str(out_path), **info}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
