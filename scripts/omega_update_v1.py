#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.omega_v1 import OmegaParamsV1, OmegaStateV1, count_omega_burns_from_events, update_omega_state


def main() -> int:
    ap = argparse.ArgumentParser(description="Ω update (destructive ontological memory): derive next omega_state_v1 from a run's omega_events.")
    ap.add_argument("--prev_state", default="", help="Previous omega_state_v1.json (optional).")
    ap.add_argument("--run_dir", required=True, help="ARC run dir containing omega_events_v1.jsonl.")
    ap.add_argument("--base_max_depth", type=int, default=4, help="Baseline max_depth (used only if prev_state is empty).")
    ap.add_argument("--base_max_programs", type=int, default=4000, help="Baseline max_programs (used only if prev_state is empty).")
    ap.add_argument("--out", required=True, help="WORM output omega_state_v1.json path.")
    ap.add_argument("--min_max_programs", type=int, default=200)
    ap.add_argument("--burn_programs_per_failure", type=int, default=10)
    ap.add_argument("--min_max_depth", type=int, default=2)
    ap.add_argument("--burn_depth_every_n_failures", type=int, default=200)
    args = ap.parse_args()

    run_dir = Path(str(args.run_dir)).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"missing_run_dir:{run_dir}")
    events_path = run_dir / "omega_events_v1.jsonl"
    if not events_path.is_file():
        raise SystemExit(f"missing_omega_events:{events_path}")

    prev_state_path = Path(str(args.prev_state)).resolve() if str(args.prev_state).strip() else None
    prev_state = OmegaStateV1.from_path(prev_state_path) if prev_state_path is not None else None

    params = OmegaParamsV1(
        min_max_programs=int(args.min_max_programs),
        burn_programs_per_failure=int(args.burn_programs_per_failure),
        min_max_depth=int(args.min_max_depth),
        burn_depth_every_n_failures=int(args.burn_depth_every_n_failures),
    )
    burns_in_run = count_omega_burns_from_events(events_path)
    st = update_omega_state(
        prev=prev_state,
        burns_in_run=int(burns_in_run),
        base_max_depth=int(args.base_max_depth),
        base_max_programs=int(args.base_max_programs),
        params=params if prev_state is None else None,
    )

    out_path = Path(str(args.out)).resolve()
    st.write_worm(out_path)
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(out_path),
                "burns_in_run": int(burns_in_run),
                "burns_total": int(st.burns_total),
                "max_depth_cap": int(st.max_depth_cap),
                "max_programs_cap": int(st.max_programs_cap),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
