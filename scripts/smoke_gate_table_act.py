#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.engine import Engine, EngineConfig
from atos_core.store import ActStore
from atos_core.suite import CHAT_DIALOGUES_20X3, run_chat_suite


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def transcripts_text(transcripts: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(str(r.get("full_text", "")) for r in transcripts)


def normalize_gate_table(table: Any) -> Dict[str, List[str]]:
    if not isinstance(table, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for ctx_sig, ids in table.items():
        if not isinstance(ctx_sig, str) or not ctx_sig:
            continue
        if not isinstance(ids, list) or not ids:
            continue
        cleaned = [str(x) for x in ids if isinstance(x, str) and str(x)]
        if not cleaned:
            continue
        out[str(ctx_sig)] = sorted(set(cleaned))
    return out


def make_gate_table_act(
    *,
    trained_on_store_hash: str,
    gate_table_json_sha256: str,
    gate_table_meta: Dict[str, Any],
    gate_table: Dict[str, List[str]],
) -> Act:
    meta = {
        "builder": str(gate_table_meta.get("builder") or ""),
        "top_k": int(gate_table_meta.get("top_k", 0) or 0),
        "min_ctx_occ": int(gate_table_meta.get("min_ctx_occ", 0) or 0),
        "min_winner_rate": float(gate_table_meta.get("min_winner_rate", 0.0) or 0.0),
        "topcand_n": int(gate_table_meta.get("topcand_n", 0) or 0),
        "trained_on_store_content_hash": str(trained_on_store_hash),
        "trained_on_store_content_hash_excluding_kinds": ["gate_table_ctxsig"],
        "gate_table_json_sha256": str(gate_table_json_sha256),
        "table_ctx_sigs": int(len(gate_table)),
    }
    evidence = {"name": "gate_table_ctxsig_v0", "meta": meta, "table": gate_table}
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
        created_at=deterministic_iso(step=0, offset_us=571000),
        kind="gate_table_ctxsig",
        match={"type": "always"},
        program=[],
        evidence=evidence,
        cost={"overhead_bits": 512},
        deps=[],
        active=True,
    )


def extract_gate_meta(transcripts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    for d in transcripts:
        turns = d.get("turns") or []
        if not isinstance(turns, list):
            continue
        for t in turns:
            if not isinstance(t, dict):
                continue
            tr = t.get("trace") or {}
            if isinstance(tr, dict) and isinstance(tr.get("gate_table_act"), dict):
                return dict(tr.get("gate_table_act") or {})
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_run", required=True)
    ap.add_argument("--gate_table_json", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--dialogues", type=int, default=10, help="How many CHAT_DIALOGUES_20X3 items to use.")
    args = ap.parse_args()

    acts_jsonl = os.path.join(str(args.acts_run), "acts.jsonl")
    if not os.path.exists(acts_jsonl):
        raise SystemExit(f"ERROR: missing acts.jsonl: {acts_jsonl}")

    gate_path = str(args.gate_table_json)
    if not os.path.exists(gate_path):
        raise SystemExit(f"ERROR: missing gate table JSON: {gate_path}")

    raw = json.loads(open(gate_path, "r", encoding="utf-8").read())
    gate_table_meta = raw.get("meta") if isinstance(raw, dict) else {}
    gate_table = normalize_gate_table(raw.get("table") if isinstance(raw, dict) else {})
    gate_table_sha = sha256_file(gate_path)

    base_store = ActStore.load_jsonl(acts_jsonl)
    base_hash = base_store.content_hash(exclude_kinds=["gate_table_ctxsig"])

    gate_act_ok = make_gate_table_act(
        trained_on_store_hash=base_hash,
        gate_table_json_sha256=gate_table_sha,
        gate_table_meta=gate_table_meta if isinstance(gate_table_meta, dict) else {},
        gate_table=gate_table,
    )

    gate_store = ActStore.load_jsonl(acts_jsonl)
    gate_store.add(gate_act_ok)

    dialogues = CHAT_DIALOGUES_20X3[: max(1, int(args.dialogues))]

    eng_base = Engine(base_store, seed=int(args.seed), config=EngineConfig())
    tx_base, m_base = run_chat_suite(eng_base, dialogues=dialogues, max_new_tokens=int(args.max_new_tokens))
    h_base = sha256_text(transcripts_text(tx_base))
    cost_base = float(m_base.get("scan_acts_considered_per_token_mean") or 0.0)

    eng_gate = Engine(
        gate_store, seed=int(args.seed), config=EngineConfig(use_gate_table_act=True)
    )
    tx_gate, m_gate = run_chat_suite(eng_gate, dialogues=dialogues, max_new_tokens=int(args.max_new_tokens))
    h_gate = sha256_text(transcripts_text(tx_gate))
    cost_gate = float(m_gate.get("scan_acts_considered_per_token_mean") or 0.0)

    if h_gate != h_base:
        raise SystemExit("FAIL: compatible gate-table act changed output hash")
    if cost_gate > cost_base:
        raise SystemExit("FAIL: compatible gate-table act did not reduce scan cost")

    # Incompatible act: explicit selection should NO-OP.
    gate_act_bad = make_gate_table_act(
        trained_on_store_hash="bogus_hash",
        gate_table_json_sha256=gate_table_sha,
        gate_table_meta=gate_table_meta if isinstance(gate_table_meta, dict) else {},
        gate_table=gate_table,
    )
    bad_store = ActStore.load_jsonl(acts_jsonl)
    bad_store.add(gate_act_bad)
    eng_bad = Engine(
        bad_store,
        seed=int(args.seed),
        config=EngineConfig(gate_table_act_id=str(gate_act_bad.id)),
    )
    tx_bad, m_bad = run_chat_suite(eng_bad, dialogues=dialogues, max_new_tokens=int(args.max_new_tokens))
    h_bad = sha256_text(transcripts_text(tx_bad))
    if h_bad != h_base:
        raise SystemExit("FAIL: incompatible gate-table act changed output hash")
    meta_bad = extract_gate_meta(tx_bad)
    if str(meta_bad.get("reason") or "") != "hash_mismatch":
        raise SystemExit(f"FAIL: expected reason=hash_mismatch, got {meta_bad}")
    if bool(meta_bad.get("used")):
        raise SystemExit("FAIL: incompatible gate-table act should not be used")

    # Incompatible act (auto selection): must NO-OP and provide explicit diagnostics.
    auto_store = ActStore.load_jsonl(acts_jsonl)
    auto_store.add(gate_act_bad)
    eng_auto = Engine(
        auto_store,
        seed=int(args.seed),
        config=EngineConfig(use_gate_table_act=True),
    )
    tx_auto, _m_auto = run_chat_suite(
        eng_auto, dialogues=dialogues, max_new_tokens=int(args.max_new_tokens)
    )
    h_auto = sha256_text(transcripts_text(tx_auto))
    if h_auto != h_base:
        raise SystemExit("FAIL: hash_mismatch_auto changed output hash")
    meta_auto = extract_gate_meta(tx_auto)
    if str(meta_auto.get("reason") or "") != "hash_mismatch_auto":
        raise SystemExit(f"FAIL: expected reason=hash_mismatch_auto, got {meta_auto}")
    if bool(meta_auto.get("used")):
        raise SystemExit("FAIL: hash_mismatch_auto should not be used")
    if int(meta_auto.get("rejected_count") or 0) < 1:
        raise SystemExit(f"FAIL: expected rejected_count>=1, got {meta_auto}")

    print(
        canonical_json_dumps(
            {
                "ok": True,
                "seed": int(args.seed),
                "sha256_baseline": str(h_base),
                "sha256_gate_ok": str(h_gate),
                "sha256_gate_bad": str(h_bad),
                "sha256_gate_auto_bad": str(h_auto),
                "scan_cost_baseline": float(cost_base),
                "scan_cost_gate_ok": float(cost_gate),
                "gate_bad_trace": meta_bad,
                "gate_auto_bad_trace": meta_auto,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
