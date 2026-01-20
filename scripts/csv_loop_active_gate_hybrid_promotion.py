#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.concepts import ConceptRegistry
from atos_core.csv_integration import CSVLoopConfig, CSVLoopIntegration
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


def _iter_ctx_sig_winners_chat(transcripts: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
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


def collect_ctx_sig_winner_counts_chat(
    transcripts: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    occ: Dict[str, int] = {}
    for ctx_sig, win in _iter_ctx_sig_winners_chat(transcripts):
        occ[ctx_sig] = int(occ.get(ctx_sig, 0) + 1)
        row = counts.get(ctx_sig)
        if row is None:
            row = {}
            counts[ctx_sig] = row
        row[str(win)] = int(row.get(str(win), 0)) + 1
    return counts, occ


def build_winners_table(
    *,
    winner_counts_by_ctx_sig: Dict[str, Dict[str, int]],
    occ_by_ctx_sig: Dict[str, int],
    top_k: int,
    min_ctx_occ: int,
    min_winner_rate: float,
    predictor_ids: set,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    top_k = max(1, int(top_k))
    min_ctx_occ = max(1, int(min_ctx_occ))
    min_winner_rate = float(min_winner_rate)

    kept: Dict[str, List[str]] = {}
    removed_occ = 0
    removed_rate = 0
    removed_empty = 0
    total_ctx = 0

    for ctx_sig, win_row in winner_counts_by_ctx_sig.items():
        total_ctx += 1
        occ = int(occ_by_ctx_sig.get(ctx_sig, 0) or 0)
        if occ < min_ctx_occ:
            removed_occ += 1
            continue
        items = [
            (str(k), int(v))
            for k, v in win_row.items()
            if isinstance(k, str) and str(k) in predictor_ids
        ]
        items.sort(key=lambda kv: (-int(kv[1]), str(kv[0])))
        allowed = [str(k) for k, _v in items[:top_k]]
        if not allowed:
            removed_empty += 1
            continue
        total = int(sum(int(v) for _k, v in items))
        cov = float(sum(int(win_row.get(a, 0)) for a in allowed)) / float(max(1, total))
        if cov < min_winner_rate:
            removed_rate += 1
            continue
        kept[str(ctx_sig)] = allowed

    meta = {
        "builder": "winners",
        "top_k": int(top_k),
        "min_ctx_occ": int(min_ctx_occ),
        "min_winner_rate": float(min_winner_rate),
        "ctx_sigs_total": int(total_ctx),
        "ctx_sigs_kept": int(len(kept)),
        "ctx_sigs_removed_min_occ": int(removed_occ),
        "ctx_sigs_removed_min_winner_rate": int(removed_rate),
        "ctx_sigs_removed_empty": int(removed_empty),
    }
    return kept, meta


def extract_mismatch_examples(
    transcripts: Sequence[Dict[str, Any]], *, limit: int
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in transcripts:
        did = int(d.get("prompt_id", -1) or -1)
        turns = d.get("turns") or []
        if not isinstance(turns, list):
            continue
        for tidx, t in enumerate(turns):
            if not isinstance(t, dict):
                continue
            tr = t.get("trace") or {}
            if not isinstance(tr, dict):
                continue
            exs = tr.get("force_gate_mismatch_examples") or []
            if not isinstance(exs, list) or not exs:
                continue
            sel_toks = tr.get("selected_tokens") or []
            for ex in exs:
                if not isinstance(ex, dict):
                    continue
                idx = int(ex.get("token_index", 0) or 0)
                prev = []
                if isinstance(sel_toks, list) and idx > 0:
                    prev = [str(x) for x in sel_toks[max(0, idx - 10) : idx] if isinstance(x, str)]
                out.append(
                    {
                        "dialogue_id": int(did),
                        "turn": int(tidx),
                        "token_index": int(idx),
                        "ctx_sig": str(ex.get("ctx_sig") or ""),
                        "baseline_token": str(ex.get("baseline_token") or ""),
                        "gate_token": str(ex.get("gate_token") or ""),
                        "fingerprint_prev_tokens": prev,
                        "debug": ex,
                    }
                )
                if len(out) >= int(limit):
                    return out
    return out


def classify_mismatch(ex: Dict[str, Any]) -> str:
    dbg = ex.get("debug") or {}
    if not isinstance(dbg, dict):
        return "unknown"
    if list(dbg.get("rewrite_rule_hits_baseline") or []) != list(dbg.get("rewrite_rule_hits_gate") or []):
        return "rewrite_rule_delta"
    base_exec = set(str(x) for x in (dbg.get("baseline_executed_predictor_ids") or []) if isinstance(x, str))
    gate_exec = set(str(x) for x in (dbg.get("gate_executed_predictor_ids") or []) if isinstance(x, str))
    if sorted(base_exec - gate_exec):
        return "missing_competitor_exec"
    base_tok = str(dbg.get("baseline_token") or "")
    gate_top = dbg.get("gate_top_candidates") or []
    if base_tok and isinstance(gate_top, list):
        if any(isinstance(c, dict) and str(c.get("token") or "") == base_tok for c in gate_top):
            return "present_but_loses"
    return "missing_candidate"


def build_hybrid_default_table(
    *,
    winner_counts_by_ctx_sig: Dict[str, Dict[str, int]],
    occ_by_ctx_sig: Dict[str, int],
    topcand_counts_by_ctx_sig: Dict[str, Dict[str, int]],
    top_k: int,
    min_ctx_occ: int,
    min_winner_rate: float,
    predictor_ids: set,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    top_k = max(1, int(top_k))
    min_ctx_occ = max(1, int(min_ctx_occ))
    min_winner_rate = float(min_winner_rate)

    kept: Dict[str, List[str]] = {}
    removed_occ = 0
    removed_rate = 0
    removed_empty = 0
    total_ctx = 0

    for ctx_sig, win_row in winner_counts_by_ctx_sig.items():
        total_ctx += 1
        occ = int(occ_by_ctx_sig.get(ctx_sig, 0) or 0)
        if occ < min_ctx_occ:
            removed_occ += 1
            continue

        winners = [
            (str(k), int(v))
            for k, v in win_row.items()
            if isinstance(k, str) and str(k) in predictor_ids
        ]
        winners.sort(key=lambda kv: (-int(kv[1]), str(kv[0])))
        topcand_row = topcand_counts_by_ctx_sig.get(ctx_sig) or {}
        topcands = [
            (str(k), int(v))
            for k, v in topcand_row.items()
            if isinstance(k, str) and str(k) in predictor_ids
        ]
        topcands.sort(key=lambda kv: (-int(kv[1]), str(kv[0])))

        allowed: List[str] = []
        # Always include the top-1 winner for this ctx_sig (keeps coverage high).
        if winners:
            allowed.append(str(winners[0][0]))
        # Add a competitor-aware source when available (from mismatch top-candidates).
        if topcands:
            pid = str(topcands[0][0])
            if pid and pid not in set(allowed):
                allowed.append(pid)
        # Fill remaining slots by winner frequency, then competitor frequency.
        for pid, _v in winners:
            if len(allowed) >= top_k:
                break
            if pid not in set(allowed):
                allowed.append(pid)
        for pid, _v in topcands:
            if len(allowed) >= top_k:
                break
            if pid not in set(allowed):
                allowed.append(pid)

        allowed = allowed[:top_k]
        if not allowed:
            removed_empty += 1
            continue

        total = int(sum(int(v) for _k, v in winners))
        cov = float(sum(int(win_row.get(a, 0)) for a in allowed)) / float(max(1, total))
        if cov < min_winner_rate:
            removed_rate += 1
            continue
        kept[str(ctx_sig)] = list(allowed)

    meta = {
        "builder": "hybrid",
        "top_k": int(top_k),
        "min_ctx_occ": int(min_ctx_occ),
        "min_winner_rate": float(min_winner_rate),
        "ctx_sigs_total": int(total_ctx),
        "ctx_sigs_kept": int(len(kept)),
        "ctx_sigs_removed_min_occ": int(removed_occ),
        "ctx_sigs_removed_min_winner_rate": int(removed_rate),
        "ctx_sigs_removed_empty": int(removed_empty),
    }
    return kept, meta


def force_gate_compare_stats_chat(transcripts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tokens_total = 0
    mismatch_tokens = 0
    fallback_tokens = 0
    used_tokens = 0
    base_iter_sum = 0
    gate_iter_sum = 0
    overhead_sum = 0
    base_iter_used_sum = 0
    gate_iter_used_sum = 0
    overhead_used_sum = 0

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
            overhead_sum += overhead * n

            for i in range(n):
                mi = int(bool(m[i]))
                fbi = int(bool(fb[i]))
                ui = int(bool(used[i]))
                mismatch_tokens += mi
                fallback_tokens += fbi
                used_tokens += ui

                bi = int(base_it[i] or 0)
                gi = int(gate_it[i] or 0)
                base_iter_sum += bi
                gate_iter_sum += gi
                if ui:
                    base_iter_used_sum += bi
                    gate_iter_used_sum += gi
                    overhead_used_sum += overhead

    base_cost_sum = float(base_iter_sum + overhead_sum)
    gate_cost_sum = float(gate_iter_sum + overhead_sum)
    base_cost_mean = float(base_cost_sum / max(1, tokens_total))
    gate_cost_mean = float(gate_cost_sum / max(1, tokens_total))
    saved = float(base_cost_mean - gate_cost_mean)
    pct_saved = float(saved / base_cost_mean) if base_cost_mean > 0 else 0.0

    base_used_sum = float(base_iter_used_sum + overhead_used_sum)
    gate_used_sum = float(gate_iter_used_sum + overhead_used_sum)
    base_used_mean = float(base_used_sum / max(1, used_tokens))
    gate_used_mean = float(gate_used_sum / max(1, used_tokens))
    saved_used = float(base_used_mean - gate_used_mean)
    pct_saved_used = float(saved_used / base_used_mean) if base_used_mean > 0 else 0.0

    return {
        "tokens_total": int(tokens_total),
        "used_tokens": int(used_tokens),
        "used_rate": float(used_tokens / max(1, tokens_total)),
        "mismatch_tokens": int(mismatch_tokens),
        "mismatch_rate": float(mismatch_tokens / max(1, tokens_total)),
        "mismatch_rate_on_used": float(mismatch_tokens / max(1, used_tokens)),
        "fallback_tokens": int(fallback_tokens),
        "fallback_rate": float(fallback_tokens / max(1, tokens_total)),
        "avg_scan_cost_baseline_per_token_mean": float(base_cost_mean),
        "avg_scan_cost_gate_per_token_mean": float(gate_cost_mean),
        "avg_scan_cost_would_save": float(saved),
        "pct_saved": float(pct_saved),
        "avg_scan_cost_baseline_on_used_per_token_mean": float(base_used_mean),
        "avg_scan_cost_gate_on_used_per_token_mean": float(gate_used_mean),
        "avg_scan_cost_would_save_on_used": float(saved_used),
        "pct_saved_on_used": float(pct_saved_used),
    }


def _scan_cost(m: Dict[str, Any]) -> float:
    v = m.get("scan_acts_considered_per_token_mean")
    if v is None:
        return 0.0
    return float(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Existing run dir containing acts.jsonl (read-only)")
    ap.add_argument("--out", required=True, help="New WORM out dir (must not exist)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--suite_repeats", type=int, default=5, help="Repeat CHAT_DIALOGUES_20X3 deterministically")
    ap.add_argument("--router_modes", default="off,on", help="Comma list: off,on")

    # Default operational config (v55 recommendation)
    ap.add_argument("--active_gate_top_k", type=int, default=2)
    ap.add_argument("--active_gate_min_ctx_occ", type=int, default=2)
    ap.add_argument("--active_gate_min_winner_rate", type=float, default=0.99)
    ap.add_argument("--active_gate_topcand_n", type=int, default=5)

    ap.add_argument("--diag_mismatch_topn", type=int, default=5)
    ap.add_argument("--diag_mismatch_cap", type=int, default=5)
    args = ap.parse_args()

    out_dir = str(args.out)
    if os.path.exists(out_dir):
        raise SystemExit(f"WORM_VIOLATION: --out already exists: {out_dir}")
    os.makedirs(out_dir, exist_ok=False)

    acts_path = os.path.join(str(args.run), "acts.jsonl")
    if not os.path.exists(acts_path):
        raise SystemExit(f"Missing acts.jsonl at: {acts_path}")
    store = ActStore.load_jsonl(acts_path)
    predictor_ids = {str(a.id) for a in store.active() if getattr(a, "kind", "") == "predictor"}

    dialogues = expand_dialogues(CHAT_DIALOGUES_20X3, repeats=int(args.suite_repeats))

    router_modes = []
    for part in str(args.router_modes).split(","):
        part = part.strip().lower()
        if part in {"off", "on"}:
            router_modes.append(part)
    if not router_modes:
        raise SystemExit("--router_modes produced no valid modes (expected off,on)")

    rows: List[Dict[str, Any]] = []

    for rm in router_modes:
        router_live_enabled = True if rm == "on" else False
        scenario_dir = os.path.join(out_dir, f"router_{rm}")

        base_cfg = EngineConfig(
            enable_contracts=False,
            router_live_enabled=bool(router_live_enabled),
            disable_macro_router=False,
        )

        # 1) Baseline (CSV off)
        eng0 = Engine(store, seed=int(args.seed), config=base_cfg)
        base_transcripts, base_metrics = run_chat_suite(
            eng0,
            dialogues=dialogues,
            max_new_tokens=int(args.max_new_tokens),
            prefix_k=8,
            template_ngram_n=6,
            template_prefix_window=32,
            csv=None,
        )
        base_text = transcripts_text(base_transcripts)
        base_hash = sha256_text(base_text)

        # 2) Shadow (CSV on) — WORM logs in scenario_dir, invariance required.
        registry = ConceptRegistry(run_dir=scenario_dir)
        csv_loop = CSVLoopIntegration(
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
        shadow_transcripts, shadow_metrics = run_chat_suite(
            eng1,
            dialogues=dialogues,
            max_new_tokens=int(args.max_new_tokens),
            prefix_k=8,
            template_ngram_n=6,
            template_prefix_window=32,
            csv=csv_loop,
        )
        shadow_hash = sha256_text(transcripts_text(shadow_transcripts))
        if shadow_hash != base_hash:
            raise SystemExit(f"OUTPUT_INVARIANCE_FAILED: baseline!=shadow (router_mode={rm})")

        winner_counts, occ_by_ctx_sig = collect_ctx_sig_winner_counts_chat(base_transcripts)

        # 3) Diagnostic compare with winners-table: capture mismatch examples → topcand_counts
        winners_table, winners_meta = build_winners_table(
            winner_counts_by_ctx_sig=winner_counts,
            occ_by_ctx_sig=occ_by_ctx_sig,
            top_k=int(args.active_gate_top_k),
            min_ctx_occ=int(args.active_gate_min_ctx_occ),
            min_winner_rate=float(args.active_gate_min_winner_rate),
            predictor_ids=predictor_ids,
        )
        diag_cfg = EngineConfig(
            enable_contracts=False,
            router_live_enabled=bool(router_live_enabled),
            disable_macro_router=False,
            force_predictor_ids_by_ctx_sig=winners_table if winners_table else None,
            force_gate_debug_compare=True,
            force_gate_debug_capture_mismatch_topn=int(args.diag_mismatch_topn),
            force_gate_debug_capture_mismatch_cap=int(args.diag_mismatch_cap),
        )
        eng_diag = Engine(store, seed=int(args.seed), config=diag_cfg)
        diag_transcripts, diag_metrics = run_chat_suite(
            eng_diag,
            dialogues=dialogues,
            max_new_tokens=int(args.max_new_tokens),
            prefix_k=8,
            template_ngram_n=6,
            template_prefix_window=32,
            csv=None,
        )
        diag_hash = sha256_text(transcripts_text(diag_transcripts))
        if diag_hash != base_hash:
            raise SystemExit(f"DIAG_INVARIANCE_FAILED: baseline!=diag_compare (router_mode={rm})")

        mismatch_examples = extract_mismatch_examples(diag_transcripts, limit=200)
        cats: Dict[str, int] = {}
        topcand_counts: Dict[str, Dict[str, int]] = {}
        for ex in mismatch_examples:
            c = classify_mismatch(ex)
            cats[c] = int(cats.get(c, 0) + 1)
            dbg = ex.get("debug") or {}
            if not isinstance(dbg, dict):
                continue
            ctx_sig = str(ex.get("ctx_sig") or "")
            if not ctx_sig:
                continue
            tops = dbg.get("baseline_top_candidates") or []
            if not isinstance(tops, list):
                continue
            row = topcand_counts.get(ctx_sig)
            if row is None:
                row = {}
                topcand_counts[ctx_sig] = row
            for cnd in tops[: max(0, int(args.active_gate_topcand_n))]:
                if not isinstance(cnd, dict):
                    continue
                src = str(cnd.get("source_act") or "")
                if not src or src not in predictor_ids:
                    continue
                row[src] = int(row.get(src, 0) + 1)

        mismatch_report = {
            "router_mode": rm,
            "sha256_transcript_text": diag_hash,
            "winners_table_meta": winners_meta,
            "force_gate": force_gate_compare_stats_chat(diag_transcripts),
            "category_counts": dict(sorted((k, int(v)) for k, v in cats.items())),
            "examples": mismatch_examples[:50],
        }
        with open(os.path.join(scenario_dir, "mismatch_report.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(mismatch_report, ensure_ascii=False, indent=2))
            f.write("\n")

        # 4) Default gate table (hybrid) — required artifact name.
        gate_table, gate_meta = build_hybrid_default_table(
            winner_counts_by_ctx_sig=winner_counts,
            occ_by_ctx_sig=occ_by_ctx_sig,
            topcand_counts_by_ctx_sig=topcand_counts,
            top_k=int(args.active_gate_top_k),
            min_ctx_occ=int(args.active_gate_min_ctx_occ),
            min_winner_rate=float(args.active_gate_min_winner_rate),
            predictor_ids=predictor_ids,
        )
        gate_meta["topcand_n"] = int(args.active_gate_topcand_n)
        gate_meta["diag_mismatch_tokens_winners_table"] = int(
            mismatch_report["force_gate"].get("mismatch_tokens", 0) or 0
        )
        gate_meta["diag_mismatch_category_counts"] = mismatch_report["category_counts"]

        table_name = "active_gate_table_ctxsig_default_hybrid_top2_minocc2_minwin0p99.json"
        table_path = os.path.join(scenario_dir, table_name)
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"meta": gate_meta, "table": gate_table},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            f.write("\n")
        table_sha256 = sha256_file(table_path)

        # 5) Active compare (invariant)
        compare_cfg = EngineConfig(
            enable_contracts=False,
            router_live_enabled=bool(router_live_enabled),
            disable_macro_router=False,
            force_predictor_ids_by_ctx_sig=gate_table if gate_table else None,
            force_gate_debug_compare=True,
        )
        eng_cmp = Engine(store, seed=int(args.seed), config=compare_cfg)
        cmp_transcripts, cmp_metrics = run_chat_suite(
            eng_cmp,
            dialogues=dialogues,
            max_new_tokens=int(args.max_new_tokens),
            prefix_k=8,
            template_ngram_n=6,
            template_prefix_window=32,
            csv=None,
        )
        cmp_hash = sha256_text(transcripts_text(cmp_transcripts))
        if cmp_hash != base_hash:
            raise SystemExit(f"ACTIVE_COMPARE_INVARIANCE_FAILED: baseline!=compare (router_mode={rm})")
        compare_force_gate = force_gate_compare_stats_chat(cmp_transcripts)

        # 6) Active real (should be ~0 divergence)
        real_cfg = EngineConfig(
            enable_contracts=False,
            router_live_enabled=bool(router_live_enabled),
            disable_macro_router=False,
            force_predictor_ids_by_ctx_sig=gate_table if gate_table else None,
        )
        eng_real = Engine(store, seed=int(args.seed), config=real_cfg)
        real_transcripts, real_metrics = run_chat_suite(
            eng_real,
            dialogues=dialogues,
            max_new_tokens=int(args.max_new_tokens),
            prefix_k=8,
            template_ngram_n=6,
            template_prefix_window=32,
            csv=None,
        )
        real_hash = sha256_text(transcripts_text(real_transcripts))
        mismatch_count, mismatch_total, mismatch_examples = compare_transcripts_by_full_text(
            base_transcripts, real_transcripts
        )
        div_rate = float(mismatch_count / max(1, mismatch_total))

        baseline_scan = _scan_cost(base_metrics)
        real_scan = _scan_cost(real_metrics)
        saved_real = float(baseline_scan - real_scan)
        pct_saved_real = float(saved_real / baseline_scan) if baseline_scan > 0 else 0.0

        row = {
            "seed": int(args.seed),
            "router_mode": str(rm),
            "router_live_enabled": bool(router_live_enabled),
            "suite_dialogues": int(len(dialogues)),
            "suite_repeats": int(args.suite_repeats),
            "tokens_total": int(base_metrics.get("trace_tokens_total", 0) or 0),
            "hashes": {
                "baseline": str(base_hash),
                "shadow": str(shadow_hash),
                "active_compare": str(cmp_hash),
                "active_real": str(real_hash),
            },
            "invariance": {
                "baseline_vs_shadow_ok": True,
                "baseline_vs_active_compare_ok": True,
            },
            "gate_table": {
                "path": str(table_path),
                "sha256": str(table_sha256),
                "meta": dict(gate_meta),
                "entries": int(len(gate_table)),
            },
            "active_compare": {
                "force_gate": dict(compare_force_gate),
            },
            "active_real": {
                "divergence_rate_full_text": float(div_rate),
                "mismatch_count_full_text": int(mismatch_count),
                "mismatch_total_full_text": int(mismatch_total),
                "mismatch_examples": mismatch_examples,
                "real_cost": {
                    "avg_scan_cost_baseline_per_token_mean": float(baseline_scan),
                    "avg_scan_cost_active_real_per_token_mean": float(real_scan),
                    "pct_saved_real": float(pct_saved_real),
                },
            },
            "baseline_suite_metrics": dict(base_metrics),
            "shadow_suite_metrics": dict(shadow_metrics),
            "active_compare_suite_metrics": dict(cmp_metrics),
            "active_real_suite_metrics": dict(real_metrics),
            "csv_chains_ok": registry.verify_chains(),
        }
        with open(os.path.join(scenario_dir, "scenario_summary.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, indent=2))
            f.write("\n")
        rows.append(row)

    # Consolidated summary for this seed (router OFF/ON)
    summary = {
        "seed": int(args.seed),
        "acts_source_run": str(args.run),
        "out_dir": str(out_dir),
        "max_new_tokens": int(args.max_new_tokens),
        "suite_repeats": int(args.suite_repeats),
        "default_gate_config": {
            "builder": "hybrid",
            "top_k": int(args.active_gate_top_k),
            "min_ctx_occ": int(args.active_gate_min_ctx_occ),
            "min_winner_rate": float(args.active_gate_min_winner_rate),
            "topcand_n": int(args.active_gate_topcand_n),
        },
        "rows": rows,
        "sha256": {"acts_jsonl": sha256_file(acts_path)},
    }
    summary_path = os.path.join(out_dir, "hybrid_promotion_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n")

    csv_path = os.path.join(out_dir, "hybrid_promotion_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "seed",
                "router_mode",
                "tokens_total",
                "used_rate",
                "mismatch_rate_compare",
                "mismatch_rate_on_used_compare",
                "pct_saved_compare_would",
                "divergence_rate_full_text_real",
                "pct_saved_real",
            ]
        )
        for r in rows:
            fg = (r.get("active_compare") or {}).get("force_gate") or {}
            real = r.get("active_real") or {}
            w.writerow(
                [
                    int(r.get("seed") or 0),
                    str(r.get("router_mode") or ""),
                    int(r.get("tokens_total") or 0),
                    float(fg.get("used_rate") or 0.0),
                    float(fg.get("mismatch_rate") or 0.0),
                    float(fg.get("mismatch_rate_on_used") or 0.0),
                    float(fg.get("pct_saved") or 0.0),
                    float(real.get("divergence_rate_full_text") or 0.0),
                    float(((real.get("real_cost") or {}).get("pct_saved_real")) or 0.0),
                ]
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

