#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.csv_integration import CSVLoopConfig, CSVLoopIntegration
from atos_core.concepts import ConceptRegistry
from atos_core.engine import Engine, EngineConfig
from atos_core.store import ActStore
from atos_core.suite import CHAT_DIALOGUES_20X3, SKILL_DIALOGUES_V0, run_chat_suite, run_skill_suite


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def transcripts_text(transcripts: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(str(r.get("full_text", "")) for r in transcripts)


def shannon_entropy(counts: Dict[str, int]) -> float:
    total = float(sum(int(v) for v in counts.values()))
    if total <= 0:
        return 0.0
    import math

    ent = 0.0
    for v in counts.values():
        p = float(v) / total
        if p > 0:
            ent -= p * math.log(p, 2)
    return float(ent)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def summarize_csv_run(*, run_dir: str, registry: ConceptRegistry) -> Dict[str, Any]:
    concepts_path = os.path.join(run_dir, "concepts.jsonl")
    evidence_path = os.path.join(run_dir, "concept_evidence.jsonl")
    telemetry_path = os.path.join(run_dir, "concept_telemetry.jsonl")

    births = 0
    if os.path.exists(concepts_path):
        for row in load_jsonl(concepts_path):
            if row.get("event") == "DEFINE":
                births += 1

    prunes = 0
    if os.path.exists(evidence_path):
        for row in load_jsonl(evidence_path):
            if row.get("event") == "PRUNE":
                prunes += 1

    call_rows = 0
    pass_rows = 0
    cost_used_sum = 0.0
    cost_base_sum = 0.0
    type_counts: Dict[str, int] = {}
    concept_call_counts: Dict[str, int] = {}
    if os.path.exists(telemetry_path):
        for row in load_jsonl(telemetry_path):
            ev = str(row.get("event") or "")
            if ev.startswith("CALL"):
                call_rows += 1
                if bool(row.get("validator_passed")):
                    pass_rows += 1
                cost_used_sum += float(row.get("cost_used") or 0.0)
                cost_base_sum += float(row.get("baseline_cost") or 0.0)
                cid = row.get("concept_id")
                if isinstance(cid, str) and cid:
                    concept_call_counts[cid] = concept_call_counts.get(cid, 0) + 1

            ct = row.get("concept_type")
            if isinstance(ct, str) and ct:
                type_counts[ct] = type_counts.get(ct, 0) + 1

    pass_rate = float(pass_rows / call_rows) if call_rows > 0 else 0.0
    avg_cost_base = float(cost_base_sum / call_rows) if call_rows > 0 else 0.0
    avg_cost_used = float(cost_used_sum / call_rows) if call_rows > 0 else 0.0
    avg_cost_saved = float(avg_cost_base - avg_cost_used)

    entropy_types = shannon_entropy(type_counts)
    monopoly = None
    if concept_call_counts:
        total_calls = sum(int(v) for v in concept_call_counts.values())
        top_cid, top_n = sorted(concept_call_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        share = float(top_n) / float(max(1, total_calls))
        if share >= 0.8:
            monopoly = {"concept_id": str(top_cid), "share": float(share)}

    alive = [c for c in registry.concepts() if bool(c.alive)]

    # Category-theory checks (MVP): only unary_pipeline concepts participate; none expected here.
    composition = {"identity_error": 0, "assoc_error": 0, "checked": 0}

    return {
        "births": int(births),
        "prunes": int(prunes),
        "alive_count": int(len(alive)),
        "totals": {
            "calls": int(call_rows),
            "pass_rate": float(pass_rate),
            "avg_cost_baseline": float(avg_cost_base),
            "avg_cost_used": float(avg_cost_used),
            "avg_cost_saved": float(avg_cost_saved),
        },
        "types": {
            "entropy_types": float(entropy_types),
            "entropy_collapse": bool(entropy_types < 0.5 and sum(type_counts.values()) >= 10),
            "type_counts": dict(sorted((k, int(v)) for k, v in type_counts.items())),
        },
        "alarms": {"monopoly": monopoly},
        "composition": composition,
        "chains": registry.verify_chains(),
        "paths": {
            "concepts_jsonl": concepts_path,
            "concept_evidence_jsonl": evidence_path,
            "concept_telemetry_jsonl": telemetry_path,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Existing run dir containing acts.jsonl (read-only)")
    ap.add_argument("--out", required=True, help="New WORM out dir for CSV logs (must not exist)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--suite", choices=["chat", "skill"], default="chat")
    ap.add_argument("--enable_contracts", action="store_true")
    args = ap.parse_args()

    acts_path = os.path.join(str(args.run), "acts.jsonl")
    store = ActStore.load_jsonl(acts_path)

    # Baseline (CSV off)
    eng0 = Engine(store, seed=int(args.seed), config=EngineConfig(enable_contracts=bool(args.enable_contracts)))
    if str(args.suite) == "skill":
        base_transcripts, base_metrics = run_skill_suite(
            eng0, tasks=SKILL_DIALOGUES_V0, max_new_tokens=int(args.max_new_tokens), csv=None
        )
    else:
        base_transcripts, base_metrics = run_chat_suite(
            eng0,
            dialogues=CHAT_DIALOGUES_20X3,
            max_new_tokens=int(args.max_new_tokens),
            prefix_k=8,
            template_ngram_n=6,
            template_prefix_window=32,
            csv=None,
        )
    base_text = transcripts_text(base_transcripts)
    base_hash = sha256_text(base_text)

    # Shadow (CSV on, no output influence)
    out_dir = str(args.out)
    if os.path.exists(out_dir):
        raise SystemExit(f"--out already exists (WORM): {out_dir}")
    registry = ConceptRegistry(run_dir=out_dir)
    csv = CSVLoopIntegration(
        registry=registry,
        config=CSVLoopConfig(
            mode="shadow",
            birth_min_count=5,
            birth_window_size=200,
            birth_min_pass_rate=0.0,
            birth_min_avg_cost=1.0,
            candidate_prefix_k=32,
            enable_empty_concept=True,
            enable_top1_winner_concept=True,
            enable_exec_prefix_concept=True,
            eval_stress=True,
            eval_best=True,
        ),
    )

    eng1 = Engine(store, seed=int(args.seed), config=EngineConfig(enable_contracts=bool(args.enable_contracts)))
    if str(args.suite) == "skill":
        shadow_transcripts, shadow_metrics = run_skill_suite(
            eng1, tasks=SKILL_DIALOGUES_V0, max_new_tokens=int(args.max_new_tokens), csv=csv
        )
    else:
        shadow_transcripts, shadow_metrics = run_chat_suite(
            eng1,
            dialogues=CHAT_DIALOGUES_20X3,
            max_new_tokens=int(args.max_new_tokens),
            prefix_k=8,
            template_ngram_n=6,
            template_prefix_window=32,
            csv=csv,
        )
    shadow_text = transcripts_text(shadow_transcripts)
    shadow_hash = sha256_text(shadow_text)
    invariance_ok = bool(base_hash == shadow_hash)
    if not invariance_ok:
        raise SystemExit("OUTPUT_INVARIANCE_FAILED: sha256(baseline)!=sha256(shadow)")

    summary: Dict[str, Any] = {
        "seed": int(args.seed),
        "suite": str(args.suite),
        "max_new_tokens": int(args.max_new_tokens),
        "acts_source_run": str(args.run),
        "out_dir": out_dir,
        "output_invariance": {
            "sha256_transcript_text_baseline": base_hash,
            "sha256_transcript_text_shadow": shadow_hash,
            "ok": bool(invariance_ok),
        },
        "baseline_suite_metrics": dict(base_metrics),
        "shadow_suite_metrics": dict(shadow_metrics),
        "csv": summarize_csv_run(run_dir=out_dir, registry=registry),
    }

    # Reproducibility hashes for the new run dir.
    summary["sha256"] = {
        "acts_jsonl": sha256_file(acts_path),
        "concepts_jsonl": sha256_file(os.path.join(out_dir, "concepts.jsonl")),
        "concept_evidence_jsonl": sha256_file(os.path.join(out_dir, "concept_evidence.jsonl")),
        "concept_telemetry_jsonl": sha256_file(os.path.join(out_dir, "concept_telemetry.jsonl")),
    }

    summary_path = os.path.join(out_dir, "concept_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

