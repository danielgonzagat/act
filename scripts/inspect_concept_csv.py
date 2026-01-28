#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act
from atos_core.store import ActStore


_SNAP_RE = re.compile(r"^step(\d+)_acts\.jsonl$")


def _find_latest_snapshot(run_dir: str) -> Optional[str]:
    snaps = os.path.join(run_dir, "snapshots")
    if not os.path.isdir(snaps):
        return None
    best: Tuple[int, str] = (-1, "")
    for name in os.listdir(snaps):
        m = _SNAP_RE.match(name)
        if not m:
            continue
        try:
            step = int(m.group(1))
        except Exception:
            continue
        path = os.path.join(snaps, name)
        if step > best[0]:
            best = (step, path)
    return best[1] if best[0] >= 0 else None


def _resolve_acts_path(*, run: Optional[str], acts: Optional[str]) -> str:
    if acts:
        p = str(acts)
        if not os.path.exists(p):
            raise SystemExit(f"acts_not_found:{p}")
        return p
    if not run:
        raise SystemExit("must_pass --run or --acts")
    run_dir = str(run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run_dir_not_found:{run_dir}")
    snap = _find_latest_snapshot(run_dir)
    if snap:
        return snap
    p = os.path.join(run_dir, "acts.jsonl")
    if os.path.exists(p):
        return p
    raise SystemExit(f"no_acts_found_in_run:{run_dir}")


def _call_deps(act: Act) -> List[str]:
    callees: List[str] = []
    for ins in list(getattr(act, "program", []) or []):
        if str(getattr(ins, "op", "")) != "CSV_CALL":
            continue
        args = getattr(ins, "args", {}) or {}
        if not isinstance(args, dict):
            continue
        cid = str(args.get("concept_id") or "")
        if cid and cid not in callees:
            callees.append(cid)
    callees.sort()
    return callees


def _static_depth(*, store: ActStore, concept_id: str, memo: Dict[str, int], stack: set) -> int:
    cid = str(concept_id or "")
    if not cid:
        return 0
    if cid in memo:
        return int(memo[cid])
    if cid in stack:
        memo[cid] = 0
        return 0
    act = store.get(cid)  # traverse inactive too (for repair diagnostics)
    if act is None or str(getattr(act, "kind", "")) != "concept_csv":
        memo[cid] = 0
        return 0
    callees = _call_deps(act)
    if not callees:
        memo[cid] = 0
        return 0
    st2 = set(stack)
    st2.add(cid)
    d = 1 + max(_static_depth(store=store, concept_id=c, memo=memo, stack=st2) for c in callees)
    memo[cid] = int(d)
    return int(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="Training run dir (uses latest snapshot by default).")
    ap.add_argument("--acts", help="Path to acts snapshot jsonl.")
    ap.add_argument("--concept_id", required=True, help="concept_csv act id to inspect")
    ap.add_argument("--max_program_ops", type=int, default=200)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    acts_path = _resolve_acts_path(run=args.run, acts=args.acts)
    store = ActStore.load_jsonl(acts_path)
    cid = str(args.concept_id)
    act = store.get(cid)
    if act is None:
        raise SystemExit(f"concept_not_found:{cid}")
    if str(getattr(act, "kind", "")) != "concept_csv":
        raise SystemExit(f"not_concept_csv:{cid}:{getattr(act, 'kind', '')}")

    ev = act.evidence if isinstance(act.evidence, dict) else {}
    iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
    meta = ev.get("meta") if isinstance(ev.get("meta"), dict) else {}
    memo: Dict[str, int] = {}
    depth = _static_depth(store=store, concept_id=cid, memo=memo, stack=set())
    callees = _call_deps(act)

    out: Dict[str, Any] = {
        "acts_path": str(acts_path),
        "concept_id": str(cid),
        "active": bool(getattr(act, "active", False)),
        "version": int(getattr(act, "version", 1) or 1),
        "created_at": str(getattr(act, "created_at", "")),
        "overhead_bits": int(getattr(act, "cost", {}).get("overhead_bits", 0) or 0),
        "interface": dict(iface),
        "meta": dict(meta),
        "static_depth": int(depth),
        "call_deps": list(callees),
        "program": [
            ins.to_dict()
            for ins in list(getattr(act, "program", []) or [])[: int(args.max_program_ops)]
        ],
        "program_truncated": int(len(list(getattr(act, "program", []) or []))) > int(args.max_program_ops),
    }
    if bool(args.json):
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print(json.dumps({k: out[k] for k in ("concept_id", "active", "static_depth", "call_deps")}, ensure_ascii=False))
    print(json.dumps({"interface": out["interface"]}, ensure_ascii=False))
    if out.get("meta"):
        print(json.dumps({"meta": out["meta"]}, ensure_ascii=False))
    print("program:")
    for ins in out["program"]:
        print(json.dumps(ins, ensure_ascii=False))
    if out.get("program_truncated"):
        print(f"... (truncated to {int(args.max_program_ops)} ops)")


if __name__ == "__main__":
    main()

