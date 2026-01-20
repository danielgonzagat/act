#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.concepts import ConceptRegistry
from atos_core.csv_integration import (
    CSVLoopConfig,
    CSVLoopIntegration,
)
from atos_core.engine import Engine, EngineConfig
from atos_core.store import ActStore
from atos_core.suite import CHAT_DIALOGUES_20X3, SKILL_DIALOGUES_V0, run_chat_suite, run_skill_suite

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


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


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


def summarize_csv_logs(*, run_dir: str, registry: ConceptRegistry) -> Dict[str, Any]:
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
    role_counts: Dict[str, int] = {}
    fallback_counts: Dict[str, int] = {}

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

                inputs = row.get("inputs") or {}
                if isinstance(inputs, dict):
                    role = str(inputs.get("csv_role") or "")
                    if role:
                        role_counts[role] = role_counts.get(role, 0) + 1
                    fb = str(inputs.get("gate_fallback_reason") or "")
                    if fb:
                        fallback_counts[fb] = fallback_counts.get(fb, 0) + 1

            ct = row.get("concept_type")
            if isinstance(ct, str) and ct:
                type_counts[ct] = type_counts.get(ct, 0) + 1

    pass_rate = float(pass_rows / call_rows) if call_rows > 0 else 0.0
    avg_cost_base = float(cost_base_sum / call_rows) if call_rows > 0 else 0.0
    avg_cost_used = float(cost_used_sum / call_rows) if call_rows > 0 else 0.0
    avg_cost_saved = float(avg_cost_base - avg_cost_used)
    pct_saved = float(avg_cost_saved / avg_cost_base) if avg_cost_base > 0 else 0.0

    entropy_types = shannon_entropy(type_counts)
    monopoly = None
    if concept_call_counts:
        total_calls = sum(int(v) for v in concept_call_counts.values())
        top_cid, top_n = sorted(concept_call_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        share = float(top_n) / float(max(1, total_calls))
        if share >= 0.8:
            monopoly = {"concept_id": str(top_cid), "share": float(share)}

    alive = [c for c in registry.concepts() if bool(c.alive)]
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
            "pct_saved": float(pct_saved),
        },
        "types": {
            "entropy_types": float(entropy_types),
            "entropy_collapse": bool(entropy_types < 0.5 and sum(type_counts.values()) >= 10),
            "type_counts": dict(sorted((k, int(v)) for k, v in type_counts.items())),
        },
        "alarms": {"monopoly": monopoly},
        "composition": composition,
        "roles": dict(sorted((k, int(v)) for k, v in role_counts.items())),
        "gate_fallback_reasons": dict(sorted((k, int(v)) for k, v in fallback_counts.items())),
        "chains": registry.verify_chains(),
        "paths": {
            "concepts_jsonl": concepts_path,
            "concept_evidence_jsonl": evidence_path,
            "concept_telemetry_jsonl": telemetry_path,
        },
    }


@dataclass
class CtxGateStats:
    tokens_total: int = 0
    tokens_covered: int = 0
    tokens_fallback: int = 0
    winners_in_allowed: int = 0
    allowed_size_sum: int = 0
    fallback_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        cov = float(self.tokens_covered / max(1, self.tokens_total))
        win_cov = float(self.winners_in_allowed / max(1, self.tokens_total))
        win_cov_on_cov = float(self.winners_in_allowed / max(1, self.tokens_covered))
        mean_allowed = float(self.allowed_size_sum / max(1, self.tokens_covered))
        return {
            "tokens_total": int(self.tokens_total),
            "coverage_tokens": int(self.tokens_covered),
            "fallback_tokens": int(self.tokens_fallback),
            "coverage_rate": float(cov),
            "winner_in_allowed_tokens": int(self.winners_in_allowed),
            "winner_in_allowed_rate": float(win_cov),
            "winner_in_allowed_rate_on_covered": float(win_cov_on_cov),
            "allowed_size_mean": float(mean_allowed),
            "fallback_reasons": dict(sorted((k, int(v)) for k, v in self.fallback_reasons.items())),
        }


def _iter_ctx_sig_winners_chat(
    transcripts: Sequence[Dict[str, Any]],
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
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
            winners = tr.get("selected_source_act_ids") or []
            if not isinstance(cks, list) or not isinstance(winners, list):
                continue
            n = min(len(cks), len(winners))
            for i in range(n):
                ck = cks[i]
                win = winners[i]
                if not isinstance(ck, str) or not ck:
                    continue
                if not isinstance(win, str) or not win:
                    continue
                if win in {"__engine__", "__unknown__", "__contract__"}:
                    continue
                out.append((f"{mode}{SEP}{ck}", str(win)))
    return out


def build_ctx_sig_gate_table_chat(
    *,
    transcripts: Sequence[Dict[str, Any]],
    top_k: int,
) -> Dict[str, List[str]]:
    top_k = max(1, int(top_k))
    events = _iter_ctx_sig_winners_chat(transcripts)
    counts: Dict[str, Dict[str, int]] = {}
    for ctx_sig, win in events:
        row = counts.get(ctx_sig)
        if row is None:
            row = {}
            counts[ctx_sig] = row
        row[win] = int(row.get(win, 0)) + 1

    table: Dict[str, List[str]] = {}
    for ctx_sig, row in counts.items():
        ordered = sorted(row.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        table[str(ctx_sig)] = [str(k) for k, _v in ordered[:top_k]]
    return table


def ctx_gate_stats_chat(
    *, transcripts: Sequence[Dict[str, Any]], table: Dict[str, List[str]]
) -> CtxGateStats:
    st = CtxGateStats()
    for ctx_sig, win in _iter_ctx_sig_winners_chat(transcripts):
        st.tokens_total += 1
        allowed = table.get(ctx_sig)
        if isinstance(allowed, list) and allowed:
            st.tokens_covered += 1
            st.allowed_size_sum += int(len(allowed))
            if win in set(str(x) for x in allowed if isinstance(x, str)):
                st.winners_in_allowed += 1
        else:
            st.tokens_fallback += 1
            reason = "missing_ctx" if ctx_sig not in table else "empty_allowed"
            st.fallback_reasons[reason] = int(st.fallback_reasons.get(reason, 0)) + 1
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Existing run dir containing acts.jsonl (read-only)")
    ap.add_argument("--out", required=True, help="New WORM out dir for CSV logs (must not exist)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--suite", choices=["chat", "skill"], default="chat")
    ap.add_argument("--enable_contracts", action="store_true")
    ap.add_argument("--no_router", action="store_true", help="Disable macro_router usage (baseline high mode)")
    ap.add_argument("--active_gate_top_k", type=int, default=2)
    ap.add_argument("--active_gate_mode", choices=["compare", "real"], default="compare")
    args = ap.parse_args()

    acts_path = os.path.join(str(args.run), "acts.jsonl")
    store = ActStore.load_jsonl(acts_path)

    out_dir = str(args.out)
    if os.path.exists(out_dir):
        raise SystemExit(f"--out already exists (WORM): {out_dir}")

    base_cfg = EngineConfig(
        enable_contracts=bool(args.enable_contracts),
        disable_macro_router=bool(args.no_router),
    )

    # 1) Baseline (CSV off)
    eng0 = Engine(store, seed=int(args.seed), config=base_cfg)
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

    # 2) Shadow (CSV on, must be invariant)
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

    eng1 = Engine(store, seed=int(args.seed), config=base_cfg)
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

    # 3) Build a ctx_sig -> topK predictors gate table from the baseline trace.
    gate_top_k = max(1, int(args.active_gate_top_k))
    gate_table: Dict[str, List[str]] = {}
    gate_table_path = ""
    gate_stats_base = CtxGateStats()
    gate_stats_compare = CtxGateStats()
    gate_stats_real = CtxGateStats()
    if str(args.suite) == "chat":
        gate_table = build_ctx_sig_gate_table_chat(transcripts=base_transcripts, top_k=int(gate_top_k))
        gate_stats_base = ctx_gate_stats_chat(transcripts=base_transcripts, table=gate_table)
        gate_table_path = os.path.join(out_dir, f"active_gate_table_ctxsig_top{int(gate_top_k)}.json")
        with open(gate_table_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"top_k": int(gate_top_k), "table": gate_table},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            f.write("\n")

    # 4) Active gate compare (invariant): compute baseline + forced gate per token and fall back
    # to baseline on mismatch, while measuring would-save.
    compare_cfg = EngineConfig(
        enable_contracts=bool(args.enable_contracts),
        disable_macro_router=bool(args.no_router),
        force_predictor_ids_by_ctx_sig=gate_table if gate_table else None,
        force_gate_debug_compare=True,
    )
    eng2 = Engine(store, seed=int(args.seed), config=compare_cfg)

    if str(args.suite) == "skill":
        compare_transcripts, compare_metrics = run_skill_suite(
            eng2, tasks=SKILL_DIALOGUES_V0, max_new_tokens=int(args.max_new_tokens), csv=None
        )
    else:
        compare_transcripts, compare_metrics = run_chat_suite(
            eng2,
            dialogues=CHAT_DIALOGUES_20X3,
            max_new_tokens=int(args.max_new_tokens),
            prefix_k=8,
            template_ngram_n=6,
            template_prefix_window=32,
            csv=None,
        )
        gate_stats_compare = ctx_gate_stats_chat(transcripts=compare_transcripts, table=gate_table)
    compare_text = transcripts_text(compare_transcripts)
    compare_hash = sha256_text(compare_text)
    invariance_ok_compare = bool(compare_hash == base_hash)
    if not invariance_ok_compare:
        raise SystemExit("ACTIVE_COMPARE_INVARIANCE_FAILED: sha256(baseline)!=sha256(active_compare)")

    def force_gate_compare_stats_chat(transcripts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        tokens_total = 0
        mismatch_tokens = 0
        fallback_tokens = 0
        used_tokens = 0
        base_iter_sum = 0
        gate_iter_sum = 0
        overhead_sum = 0

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
                m = tr.get("force_gate_mismatch") or []
                fb = tr.get("force_gate_fallback") or []
                used = tr.get("force_gate_used") or []
                base_it = tr.get("force_gate_baseline_predictors_iterated") or []
                gate_it = tr.get("force_gate_predictors_iterated") or []
                if not all(isinstance(x, list) for x in [m, fb, used, base_it, gate_it]):
                    continue
                n = min(len(m), len(fb), len(used), len(base_it), len(gate_it))
                if n <= 0:
                    continue

                rr_total = int(tr.get("rewrite_rules_total", 0) or 0)
                sel_present = 1 if tr.get("selector_id") else 0
                overhead = int(rr_total + sel_present)

                tokens_total += n
                mismatch_tokens += sum(int(bool(m[i])) for i in range(n))
                fallback_tokens += sum(int(bool(fb[i])) for i in range(n))
                used_tokens += sum(int(bool(used[i])) for i in range(n))
                base_iter_sum += sum(int(base_it[i] or 0) for i in range(n))
                gate_iter_sum += sum(int(gate_it[i] or 0) for i in range(n))
                overhead_sum += overhead * n

        base_cost_sum = float(base_iter_sum + overhead_sum)
        gate_cost_sum = float(gate_iter_sum + overhead_sum)
        base_cost_mean = float(base_cost_sum / max(1, tokens_total))
        gate_cost_mean = float(gate_cost_sum / max(1, tokens_total))
        saved = float(base_cost_mean - gate_cost_mean)
        pct_saved = float(saved / base_cost_mean) if base_cost_mean > 0 else 0.0

        return {
            "tokens_total": int(tokens_total),
            "mismatch_tokens": int(mismatch_tokens),
            "mismatch_rate": float(mismatch_tokens / max(1, tokens_total)),
            "fallback_tokens": int(fallback_tokens),
            "fallback_rate": float(fallback_tokens / max(1, tokens_total)),
            "used_tokens": int(used_tokens),
            "used_rate": float(used_tokens / max(1, tokens_total)),
            "avg_scan_cost_baseline_per_token_mean": float(base_cost_mean),
            "avg_scan_cost_gate_per_token_mean": float(gate_cost_mean),
            "avg_scan_cost_would_save": float(saved),
            "pct_saved": float(pct_saved),
        }

    compare_force_gate = {}
    if str(args.suite) == "chat":
        compare_force_gate = force_gate_compare_stats_chat(compare_transcripts)

    def _scan_cost(m: Dict[str, Any]) -> float:
        v = m.get("scan_acts_considered_per_token_mean")
        if v is None:
            return 0.0
        return float(v)

    baseline_scan = _scan_cost(base_metrics)
    compare_scan = _scan_cost(compare_metrics)

    # Optional: active-real run (economia real, divergência permitida).
    real_transcripts: List[Dict[str, Any]] = []
    real_metrics: Dict[str, Any] = {}
    real_hash = ""
    real_div = {}
    real_cost = {}
    if str(args.active_gate_mode) == "real":
        real_cfg = EngineConfig(
            enable_contracts=bool(args.enable_contracts),
            disable_macro_router=bool(args.no_router),
            force_predictor_ids_by_ctx_sig=gate_table if gate_table else None,
        )
        eng3 = Engine(store, seed=int(args.seed), config=real_cfg)
        if str(args.suite) == "skill":
            real_transcripts, real_metrics = run_skill_suite(
                eng3, tasks=SKILL_DIALOGUES_V0, max_new_tokens=int(args.max_new_tokens), csv=None
            )
        else:
            real_transcripts, real_metrics = run_chat_suite(
                eng3,
                dialogues=CHAT_DIALOGUES_20X3,
                max_new_tokens=int(args.max_new_tokens),
                prefix_k=8,
                template_ngram_n=6,
                template_prefix_window=32,
                csv=None,
            )
            gate_stats_real = ctx_gate_stats_chat(transcripts=real_transcripts, table=gate_table)
        real_text = transcripts_text(real_transcripts)
        real_hash = sha256_text(real_text)

        mismatch_count, mismatch_total, mismatch_examples = compare_transcripts_by_full_text(
            base_transcripts, real_transcripts
        )
        real_div = {
            "sha256_transcript_text_active_real": real_hash,
            "mismatch_count_full_text": int(mismatch_count),
            "mismatch_total_full_text": int(mismatch_total),
            "mismatch_examples": mismatch_examples,
        }
        real_scan = _scan_cost(real_metrics)
        saved_real = float(baseline_scan - real_scan)
        pct_saved_real = float(saved_real / baseline_scan) if baseline_scan > 0 else 0.0
        real_cost = {
            "avg_scan_cost_baseline_per_token_mean": float(baseline_scan),
            "avg_scan_cost_active_real_per_token_mean": float(real_scan),
            "avg_scan_cost_saved": float(saved_real),
            "pct_saved": float(pct_saved_real),
        }

    summary: Dict[str, Any] = {
        "seed": int(args.seed),
        "suite": str(args.suite),
        "max_new_tokens": int(args.max_new_tokens),
        "acts_source_run": str(args.run),
        "out_dir": out_dir,
        "flags": {
            "enable_contracts": bool(args.enable_contracts),
            "no_router": bool(args.no_router),
            "active_gate_top_k": int(gate_top_k),
            "active_gate_mode": str(args.active_gate_mode),
        },
        "output_invariance": {
            "sha256_transcript_text_baseline": base_hash,
            "sha256_transcript_text_shadow": shadow_hash,
            "sha256_transcript_text_active_compare": compare_hash,
            "ok": bool(invariance_ok),
            "active_compare_ok": bool(invariance_ok_compare),
        },
        "baseline_suite_metrics": dict(base_metrics),
        "shadow_suite_metrics": dict(shadow_metrics),
        "active_compare_suite_metrics": dict(compare_metrics),
        "active_compare_force_gate": dict(compare_force_gate),
        "active_compare_cost": {
            "avg_scan_cost_baseline_per_token_mean": float(baseline_scan),
            "avg_scan_cost_active_compare_per_token_mean": float(compare_scan),
            "avg_scan_cost_saved": float(baseline_scan - compare_scan),
            "pct_saved": float((baseline_scan - compare_scan) / baseline_scan) if baseline_scan > 0 else 0.0,
        },
        "active_gate": {
            "top_k": int(gate_top_k),
            "table_entries": int(len(gate_table)),
            "table_path": str(gate_table_path),
            "table_sha256": sha256_file(gate_table_path) if gate_table_path else "",
            "baseline_stats": gate_stats_base.to_dict(),
            "active_compare_stats": gate_stats_compare.to_dict(),
            **({"active_real_stats": gate_stats_real.to_dict()} if real_transcripts else {}),
        },
        **({"active_real_suite_metrics": dict(real_metrics)} if real_transcripts else {}),
        **({"active_real_divergence": dict(real_div)} if real_div else {}),
        **({"active_real_cost": dict(real_cost)} if real_cost else {}),
        "csv": summarize_csv_logs(run_dir=out_dir, registry=registry),
    }

    summary["sha256"] = {
        "acts_jsonl": sha256_file(acts_path),
        "concepts_jsonl": sha256_file(os.path.join(out_dir, "concepts.jsonl")),
        "concept_evidence_jsonl": sha256_file(os.path.join(out_dir, "concept_evidence.jsonl")),
        "concept_telemetry_jsonl": sha256_file(os.path.join(out_dir, "concept_telemetry.jsonl")),
        **({"active_gate_table_json": sha256_file(gate_table_path)} if gate_table_path else {}),
    }

    summary_path = os.path.join(out_dir, "concept_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
