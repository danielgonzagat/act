#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.engine import Engine, EngineConfig
from atos_core.store import ActStore
from atos_core.suite import CHAT_DIALOGUES_20X3, run_chat_suite

SEP = "\u241f"


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


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def ensure_out_dir_absent(out_dir: str) -> None:
    if os.path.exists(out_dir):
        raise SystemExit(f"ERROR: --out already exists: {out_dir}")


def transcripts_text(transcripts: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(str(r.get("full_text", "")) for r in transcripts)


def compare_transcripts_by_full_text(
    a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]
) -> Tuple[int, int, List[Dict[str, Any]]]:
    total = min(len(a), len(b))
    mismatches = 0
    examples: List[Dict[str, Any]] = []
    for i in range(total):
        ta = str(a[i].get("full_text", ""))
        tb = str(b[i].get("full_text", ""))
        if ta != tb:
            mismatches += 1
            if len(examples) < 3:
                examples.append(
                    {
                        "index": int(i),
                        "a_sha256": sha256_text(ta),
                        "b_sha256": sha256_text(tb),
                        "a_snip": ta[:120],
                        "b_snip": tb[:120],
                    }
                )
    return mismatches, total, examples


def expand_dialogues(
    dialogues: Sequence[Sequence[str]], *, repeats: int
) -> Tuple[Tuple[str, ...], ...]:
    r = max(1, int(repeats))
    base: List[Tuple[str, ...]] = [tuple(d) for d in dialogues]
    out: List[Tuple[str, ...]] = []
    for _ in range(r):
        out.extend(base)
    return tuple(out)


def extract_gate_act_trace_meta(transcripts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    for d in transcripts:
        turns = d.get("turns") or []
        if not isinstance(turns, list):
            continue
        for t in turns:
            if not isinstance(t, dict):
                continue
            tr = t.get("trace") or {}
            if not isinstance(tr, dict):
                continue
            meta = tr.get("gate_table_act")
            if isinstance(meta, dict):
                return dict(meta)
    return {}


def extract_gate_table_from_store(store: ActStore) -> Tuple[Optional[str], Dict[str, List[str]], Dict[str, Any]]:
    base_hash = store.content_hash(exclude_kinds=["gate_table_ctxsig"])
    act = store.best_gate_table_for_hash(base_hash)
    if act is None:
        return None, {}, {"reason": "no_compatible_gate_table_act", "store_hash": base_hash}
    ev = act.evidence if isinstance(act.evidence, dict) else {}
    table = ev.get("table") if isinstance(ev, dict) else {}
    meta = ev.get("meta") if isinstance(ev, dict) else {}
    if not isinstance(table, dict):
        table = {}
    out: Dict[str, List[str]] = {}
    for ctx_sig, ids in table.items():
        if not isinstance(ctx_sig, str) or not ctx_sig:
            continue
        if not isinstance(ids, list):
            continue
        out[str(ctx_sig)] = [str(x) for x in ids if isinstance(x, str) and str(x)]
    return str(act.id), out, {"store_hash": base_hash, "meta": meta}


def gate_table_used_rate(
    *, transcripts: Sequence[Dict[str, Any]], table: Dict[str, List[str]]
) -> Tuple[int, int]:
    covered = 0
    total = 0
    for d in transcripts:
        turns = d.get("turns") or []
        if not isinstance(turns, list):
            continue
        for t in turns:
            if not isinstance(t, dict):
                continue
            mode = str(t.get("mode") or "default")
            tr = t.get("trace") or {}
            if not isinstance(tr, dict):
                continue
            cks = tr.get("context_keys") or []
            if not isinstance(cks, list):
                continue
            for ck in cks:
                if not isinstance(ck, str) or not ck:
                    continue
                total += 1
                if f"{mode}{SEP}{ck}" in table:
                    covered += 1
    return covered, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_base_run", required=True, help="Base run dir with acts.jsonl (no gate-table act).")
    ap.add_argument("--acts_gate_run", required=True, help="Run dir with acts.jsonl including gate-table act.")
    ap.add_argument("--out", required=True, help="WORM output dir under results/.")
    ap.add_argument("--seeds", default="0,1", help="Comma-separated seeds (default: 0,1).")
    ap.add_argument("--router_modes", default="off,on", help="Comma-separated router modes: off,on.")
    ap.add_argument("--suite_repeats", type=int, default=5, help="Repeat CHAT_DIALOGUES_20X3 N times.")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    args = ap.parse_args()

    out_dir = str(args.out)
    ensure_out_dir_absent(out_dir)
    os.makedirs(out_dir, exist_ok=False)

    base_acts_jsonl = os.path.join(str(args.acts_base_run), "acts.jsonl")
    gate_acts_jsonl = os.path.join(str(args.acts_gate_run), "acts.jsonl")
    if not os.path.exists(base_acts_jsonl):
        raise SystemExit(f"ERROR: missing {base_acts_jsonl}")
    if not os.path.exists(gate_acts_jsonl):
        raise SystemExit(f"ERROR: missing {gate_acts_jsonl}")

    dialogues = expand_dialogues(CHAT_DIALOGUES_20X3, repeats=int(args.suite_repeats))

    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip() != ""]
    router_modes = [str(x).strip() for x in str(args.router_modes).split(",") if str(x).strip() != ""]
    if not seeds:
        raise SystemExit("ERROR: empty --seeds")
    if not router_modes:
        raise SystemExit("ERROR: empty --router_modes")

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for rm in router_modes:
            router_on = rm.lower() in {"on", "true", "1", "yes"}

            base_store = ActStore.load_jsonl(base_acts_jsonl)
            base_engine = Engine(
                base_store,
                seed=int(seed),
                config=EngineConfig(router_live_enabled=bool(router_on)),
            )
            base_transcripts, base_metrics = run_chat_suite(
                base_engine, dialogues=dialogues, max_new_tokens=int(args.max_new_tokens)
            )
            base_text = transcripts_text(base_transcripts)
            base_hash = sha256_text(base_text)
            base_cost = float(base_metrics.get("scan_acts_considered_per_token_mean") or 0.0)
            tokens_total = int(base_metrics.get("trace_tokens_total") or 0)

            gate_store = ActStore.load_jsonl(gate_acts_jsonl)
            gate_engine = Engine(
                gate_store,
                seed=int(seed),
                config=EngineConfig(
                    router_live_enabled=bool(router_on),
                    use_gate_table_act=True,
                ),
            )
            gate_transcripts, gate_metrics = run_chat_suite(
                gate_engine, dialogues=dialogues, max_new_tokens=int(args.max_new_tokens)
            )
            gate_text = transcripts_text(gate_transcripts)
            gate_hash = sha256_text(gate_text)

            mismatches, total_dialogues, examples = compare_transcripts_by_full_text(
                base_transcripts, gate_transcripts
            )
            divergence_rate = float(mismatches / max(1, total_dialogues))

            gate_cost = float(gate_metrics.get("scan_acts_considered_per_token_mean") or 0.0)
            pct_saved_real = float((base_cost - gate_cost) / base_cost) if base_cost > 0 else 0.0

            gate_act_id, gate_table, gate_table_meta = extract_gate_table_from_store(gate_store)
            used_cov, used_total = gate_table_used_rate(transcripts=base_transcripts, table=gate_table)
            used_rate = float(used_cov / max(1, used_total))

            trace_meta = extract_gate_act_trace_meta(gate_transcripts)
            gate_used = bool(trace_meta.get("used")) if isinstance(trace_meta, dict) else False
            acts_hash_match = bool(
                str(trace_meta.get("reason") or "") == "ok"
                and str(trace_meta.get("trained_on_store_content_hash") or "")
                and str(trace_meta.get("trained_on_store_content_hash") or "")
                == str(trace_meta.get("store_content_hash") or "")
            )

            rows.append(
                {
                    "seed": int(seed),
                    "router_mode": "on" if router_on else "off",
                    "tokens_total": int(tokens_total),
                    "divergence_rate_full_text": float(divergence_rate),
                    "mismatch_dialogues": int(mismatches),
                    "total_dialogues": int(total_dialogues),
                    "pct_saved_real": float(pct_saved_real),
                    "avg_scan_cost_baseline_per_token_mean": float(base_cost),
                    "avg_scan_cost_gate_per_token_mean": float(gate_cost),
                    "used_rate_est": float(used_rate),
                    "gate_act_used": bool(gate_used),
                    "gate_act_id": str(trace_meta.get("act_id") or gate_act_id or ""),
                    "acts_hash_match": bool(acts_hash_match),
                    "hash_baseline": str(base_hash),
                    "hash_gate_from_store": str(gate_hash),
                    "mismatch_examples": examples,
                    "gate_table_meta": gate_table_meta,
                }
            )

    summary = {
        "name": "V57_GATE_TABLE_ACT_EVAL",
        "acts": {
            "base_acts_jsonl": base_acts_jsonl,
            "gate_acts_jsonl": gate_acts_jsonl,
            "sha256_base_acts_jsonl": sha256_file(base_acts_jsonl),
            "sha256_gate_acts_jsonl": sha256_file(gate_acts_jsonl),
        },
        "results": rows,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path + ".tmp", "w", encoding="utf-8") as f:
        f.write(canonical_json(summary))
        f.write("\n")
    os.replace(summary_path + ".tmp", summary_path)

    csv_path = os.path.join(out_dir, "summary.csv")
    fields = [
        "seed",
        "router_mode",
        "tokens_total",
        "divergence_rate_full_text",
        "pct_saved_real",
        "avg_scan_cost_baseline_per_token_mean",
        "avg_scan_cost_gate_per_token_mean",
        "used_rate_est",
        "gate_act_used",
        "gate_act_id",
        "acts_hash_match",
        "hash_baseline",
        "hash_gate_from_store",
    ]
    with open(csv_path + ".tmp", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    os.replace(csv_path + ".tmp", csv_path)

    print(canonical_json({"out_dir": out_dir, "summary_json": summary_path, "summary_csv": csv_path}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

