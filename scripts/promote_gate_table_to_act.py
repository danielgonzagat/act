#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Patch, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.ethics import validate_act_for_promotion
from atos_core.ledger import Ledger
from atos_core.store import ActStore


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def ensure_out_dir_absent(out_dir: str) -> None:
    if os.path.exists(out_dir):
        _fail(f"ERROR: --out already exists: {out_dir}")


def find_scenario_summary_path(gate_table_json_path: str) -> Optional[str]:
    cand = os.path.join(os.path.dirname(gate_table_json_path), "scenario_summary.json")
    return cand if os.path.exists(cand) else None


def extract_saved_metrics_from_scenario_summary(
    scenario_summary_path: Optional[str],
) -> Tuple[Optional[float], Optional[float]]:
    if not scenario_summary_path:
        return None, None
    try:
        obj = json.loads(open(scenario_summary_path, "r", encoding="utf-8").read())
        pct_saved_real = (
            obj.get("active_real", {})
            .get("real_cost", {})
            .get("pct_saved_real")
        )
        pct_saved_compare = (
            obj.get("active_compare", {})
            .get("force_gate", {})
            .get("pct_saved")
        )
        return (
            float(pct_saved_real) if pct_saved_real is not None else None,
            float(pct_saved_compare) if pct_saved_compare is not None else None,
        )
    except Exception:
        return None, None


def normalize_gate_table(table: Any) -> Dict[str, List[str]]:
    if not isinstance(table, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for ctx_sig, ids in table.items():
        if not isinstance(ctx_sig, str) or not ctx_sig:
            continue
        if not isinstance(ids, list) or not ids:
            continue
        cleaned: List[str] = []
        for x in ids:
            if not isinstance(x, str):
                continue
            x = str(x)
            if not x:
                continue
            cleaned.append(x)
        if not cleaned:
            continue
        uniq = sorted(set(cleaned))
        out[str(ctx_sig)] = uniq
    return out


def gate_table_edge_count(table: Dict[str, List[str]]) -> int:
    return int(sum(len(v) for v in table.values()))


def make_gate_table_act(
    *,
    store_hash_excluding_gate_tables: str,
    base_acts_sha256: str,
    gate_table_json_path: str,
    gate_table_json_sha256: str,
    gate_table_meta: Dict[str, Any],
    gate_table: Dict[str, List[str]],
    pct_saved_real: Optional[float],
    pct_saved_compare: Optional[float],
    scenario_summary_path: Optional[str],
) -> Act:
    meta: Dict[str, Any] = {}
    meta.update({k: gate_table_meta.get(k) for k in sorted(gate_table_meta.keys())})
    meta.update(
        {
            "builder": str(gate_table_meta.get("builder") or ""),
            "top_k": int(gate_table_meta.get("top_k", 0) or 0),
            "min_ctx_occ": int(gate_table_meta.get("min_ctx_occ", 0) or 0),
            "min_winner_rate": float(gate_table_meta.get("min_winner_rate", 0.0) or 0.0),
            "topcand_n": int(gate_table_meta.get("topcand_n", 0) or 0),
            "trained_on_store_content_hash": str(store_hash_excluding_gate_tables),
            "trained_on_store_content_hash_excluding_kinds": ["gate_table_ctxsig"],
            "trained_on_acts_jsonl_sha256": str(base_acts_sha256),
            "gate_table_json_path": str(gate_table_json_path),
            "gate_table_json_sha256": str(gate_table_json_sha256),
            "scenario_summary_path": str(scenario_summary_path or ""),
            "pct_saved_real": pct_saved_real,
            "pct_saved_compare": pct_saved_compare,
            "table_ctx_sigs": int(len(gate_table)),
            "table_edges": int(gate_table_edge_count(gate_table)),
        }
    )

    evidence = {
        "name": "gate_table_ctxsig_v0",
        "meta": meta,
        "table": gate_table,
    }

    body = {
        "kind": "gate_table_ctxsig",
        "match": {"type": "always"},
        "program": [],
        "evidence": evidence,
        "deps": [],
        "active": True,
    }
    act_id = "act_gate_table_ctxsig_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))[:12]
    return Act(
        id=act_id,
        version=1,
        created_at=deterministic_iso(step=0, offset_us=570000),
        kind="gate_table_ctxsig",
        match={"type": "always"},
        program=[],
        evidence=evidence,
        cost={"overhead_bits": 512},
        deps=[],
        active=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_run", required=True, help="Run dir containing acts.jsonl (base).")
    ap.add_argument("--gate_table_json", required=True, help="Path to active gate-table JSON.")
    ap.add_argument("--out", required=True, help="New WORM run dir to write updated acts.jsonl.")
    args = ap.parse_args()

    acts_run = str(args.acts_run)
    gate_table_json_path = str(args.gate_table_json)
    out_dir = str(args.out)

    ensure_out_dir_absent(out_dir)
    os.makedirs(out_dir, exist_ok=False)

    acts_jsonl = os.path.join(acts_run, "acts.jsonl")
    if not os.path.exists(acts_jsonl):
        _fail(f"ERROR: acts.jsonl not found: {acts_jsonl}")
    if not os.path.exists(gate_table_json_path):
        _fail(f"ERROR: gate table JSON not found: {gate_table_json_path}")

    base_acts_sha256 = sha256_file(acts_jsonl)
    gate_table_json_sha256 = sha256_file(gate_table_json_path)

    store = ActStore.load_jsonl(acts_jsonl)
    store_hash_excl = store.content_hash(exclude_kinds=["gate_table_ctxsig"])

    raw = json.loads(open(gate_table_json_path, "r", encoding="utf-8").read())
    gate_table_meta = raw.get("meta") if isinstance(raw, dict) else {}
    gate_table = normalize_gate_table(raw.get("table") if isinstance(raw, dict) else {})

    scenario_summary_path = find_scenario_summary_path(gate_table_json_path)
    pct_saved_real, pct_saved_compare = extract_saved_metrics_from_scenario_summary(
        scenario_summary_path
    )

    gate_act = make_gate_table_act(
        store_hash_excluding_gate_tables=store_hash_excl,
        base_acts_sha256=base_acts_sha256,
        gate_table_json_path=gate_table_json_path,
        gate_table_json_sha256=gate_table_json_sha256,
        gate_table_meta=gate_table_meta if isinstance(gate_table_meta, dict) else {},
        gate_table=gate_table,
        pct_saved_real=pct_saved_real,
        pct_saved_compare=pct_saved_compare,
        scenario_summary_path=scenario_summary_path,
    )

    ethics = validate_act_for_promotion(gate_act)
    if not bool(ethics.ok):
        _fail(
            f"ERROR: ethics_fail_closed:promote_gate_table:{ethics.reason}:{ethics.violated_laws}"
        )

    if store.get(gate_act.id) is not None:
        _fail(f"ERROR: gate-table act id collision: {gate_act.id}")
    store.add(gate_act)

    out_acts_jsonl = os.path.join(out_dir, "acts.jsonl")
    # Preserve base file order for auditability: copy base acts.jsonl as-is and append the new ACT.
    with open(acts_jsonl, "rb") as f:
        base_bytes = f.read()
    if base_bytes and not base_bytes.endswith(b"\n"):
        base_bytes += b"\n"
    gate_line = canonical_json_dumps(gate_act.to_dict()).encode("utf-8") + b"\n"
    tmp = out_acts_jsonl + ".tmp"
    with open(tmp, "wb") as f:
        f.write(base_bytes)
        f.write(gate_line)
    os.replace(tmp, out_acts_jsonl)
    out_acts_sha256 = sha256_file(out_acts_jsonl)

    promotion_ledger_path = os.path.join(out_dir, "promotion_ledger.jsonl")
    ledger = Ledger(path=promotion_ledger_path)
    patch = Patch(
        kind="ADD_ACT",
        payload={
            "act_id": str(gate_act.id),
            "act_kind": str(gate_act.kind),
            "trained_on_store_content_hash": str(store_hash_excl),
            "source_acts_run": str(acts_run),
            "source_acts_jsonl_sha256": str(base_acts_sha256),
            "source_gate_table_json_path": str(gate_table_json_path),
            "source_gate_table_json_sha256": str(gate_table_json_sha256),
        },
    )
    entry = ledger.append(
        step=0,
        patch=patch,
        acts_hash=store.content_hash(),
        metrics={
            "promotion": {
                "store_content_hash_excluding_gate_tables": str(store_hash_excl),
                "base_acts_jsonl_sha256": str(base_acts_sha256),
                "gate_table_json_sha256": str(gate_table_json_sha256),
                "out_acts_jsonl_sha256": str(out_acts_sha256),
            }
        },
        snapshot_path="acts.jsonl",
    )

    ok = ledger.verify_chain()
    manifest = {
        "name": "PROMOTE_GATE_TABLE_TO_ACT_V57",
        "out_dir": str(out_dir),
        "source_acts_run": str(acts_run),
        "source_gate_table_json": str(gate_table_json_path),
        "gate_table_act_id": str(gate_act.id),
        "store_content_hash_excluding_gate_tables": str(store_hash_excl),
        "verify_chain": bool(ok),
        "promotion_ledger_last_hash": str(entry.get("entry_hash") or ""),
        "sha256": {
            "source_acts_jsonl": str(base_acts_sha256),
            "source_gate_table_json": str(gate_table_json_sha256),
            "out_acts_jsonl": str(out_acts_sha256),
            "promotion_ledger_jsonl": str(sha256_file(promotion_ledger_path)),
        },
    }
    manifest_path = os.path.join(out_dir, "promotion_manifest.json")
    with open(manifest_path + ".tmp", "w", encoding="utf-8") as f:
        f.write(canonical_json_dumps(manifest))
        f.write("\n")
    os.replace(manifest_path + ".tmp", manifest_path)

    print(
        canonical_json_dumps(
            {
                "out_dir": str(out_dir),
                "gate_table_act_id": str(gate_act.id),
                "sha256_out_acts_jsonl": str(out_acts_sha256),
                "sha256_in_gate_table_json": str(gate_table_json_sha256),
                "verify_chain": bool(ok),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
