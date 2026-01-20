#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.external_world_gate_v122 import EXTERNAL_WORLD_ACTION_SEARCH_V122, external_world_access_v122


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    manifest = str(args.manifest)

    ok1 = False
    reason1 = ""
    try:
        external_world_access_v122(
            allowed=False,
            manifest_path=manifest,
            action=EXTERNAL_WORLD_ACTION_SEARCH_V122,
            reason_code="progress_blocked",
            args={"query": "x", "limit": 1, "source_filter": "engineering_doc"},
            seed=0,
            turn_index=0,
            prev_event_sig="",
        )
        ok1 = True
    except ValueError as e:
        reason1 = str(e)

    ok2 = False
    reason2 = ""
    try:
        external_world_access_v122(
            allowed=True,
            manifest_path=manifest,
            action=EXTERNAL_WORLD_ACTION_SEARCH_V122,
            reason_code="invalid_reason_code_x",
            args={"query": "x", "limit": 1, "source_filter": "engineering_doc"},
            seed=0,
            turn_index=0,
            prev_event_sig="",
        )
        ok2 = True
    except ValueError as e:
        reason2 = str(e)

    out = {
        "ok": True,
        "negative_tests": {
            "access_not_allowed": {"ok": bool(ok1), "reason": str(reason1)},
            "invalid_reason_code": {"ok": bool(ok2), "reason": str(reason2)},
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

