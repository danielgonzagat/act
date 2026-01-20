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

    alive = [c for c in registry.concepts() if bool(c.alive)]

    return {
        "births": int(births),
        "prunes": int(prunes),
        "alive_count": int(len(alive)),
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


def _iter_chat_tokens(
    transcripts: Sequence[Dict[str, Any]],
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
            mode = str(t.get("mode") or "default")
            tr = t.get("trace") or {}
            if not isinstance(tr, dict):
                continue
            cks = tr.get("context_keys") or []
            winners = tr.get("selected_source_act_ids") or []
            execs = tr.get("executed_predictor_ids") or []
            sels = tr.get("selected_tokens") or []
            if not isinstance(cks, list) or not isinstance(winners, list) or not isinstance(execs, list):
                continue
            n = min(len(cks), len(winners), len(execs))
            for i in range(n):
                ck = cks[i]
                win = winners[i]
                ex = execs[i]
                tok = sels[i] if (isinstance(sels, list) and i < len(sels)) else ""
                if not isinstance(ck, str) or not ck:
                    continue
                if not isinstance(win, str) or not win:
                    continue
                if win in {"__engine__", "__unknown__", "__contract__"}:
                    continue
                exec_ids: List[str] = []
                if isinstance(ex, list):
                    exec_ids = [str(x) for x in ex if isinstance(x, str) and str(x)]
                out.append(
                    {
                        "dialogue_id": int(did),
                        "turn": int(tidx),
                        "token_index": int(i),
                        "ctx_sig": f"{mode}{SEP}{ck}",
                        "winner_act_id": str(win),
                        "selected_token": str(tok) if isinstance(tok, str) else "",
                        "executed_predictor_ids": exec_ids,
                    }
                )
    return out


def ctx_gate_stats_chat(
    *, tok_recs: Sequence[Dict[str, Any]], table: Dict[str, List[str]]
) -> CtxGateStats:
    st = CtxGateStats()
    for r in tok_recs:
        ctx_sig = str(r.get("ctx_sig") or "")
        win = str(r.get("winner_act_id") or "")
        if not ctx_sig or not win:
            continue
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
            st.fallback_reasons[reason] = int(st.fallback_reasons.get(reason, 0) + 1)
    return st


def collect_counts(
    *, tok_recs: Sequence[Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]], Dict[str, int]]:
    winners: Dict[str, Dict[str, int]] = {}
    executed: Dict[str, Dict[str, int]] = {}
    occ: Dict[str, int] = {}
    for r in tok_recs:
        ctx_sig = str(r.get("ctx_sig") or "")
        win = str(r.get("winner_act_id") or "")
        if not ctx_sig or not win:
            continue
        occ[ctx_sig] = int(occ.get(ctx_sig, 0) + 1)

        roww = winners.get(ctx_sig)
        if roww is None:
            roww = {}
            winners[ctx_sig] = roww
        roww[win] = int(roww.get(win, 0) + 1)

        ex = r.get("executed_predictor_ids") or []
        if isinstance(ex, list):
            rowe = executed.get(ctx_sig)
            if rowe is None:
                rowe = {}
                executed[ctx_sig] = rowe
            for pid in ex:
                if not isinstance(pid, str) or not pid:
                    continue
                rowe[pid] = int(rowe.get(pid, 0) + 1)
    return winners, executed, occ


def parse_top_k_list(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return [int(x) for x in out if int(x) > 0]


def parse_builder_list(s: str) -> List[str]:
    allowed = {"winners", "executed", "topcand", "hybrid"}
    out: List[str] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if part in allowed:
            out.append(part)
    return out


def safe_float_token(x: float) -> str:
    s = f"{float(x):.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p").replace("-", "m")


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


def build_gate_table(
    *,
    builder: str,
    top_k: int,
    min_ctx_occ: int,
    min_winner_rate: float,
    predictor_ids: set,
    occ_by_ctx_sig: Dict[str, int],
    winner_counts_by_ctx_sig: Dict[str, Dict[str, int]],
    executed_counts_by_ctx_sig: Dict[str, Dict[str, int]],
    topcand_counts_by_ctx_sig: Dict[str, Dict[str, int]],
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    builder = str(builder)
    top_k = max(1, int(top_k))
    min_ctx_occ = max(1, int(min_ctx_occ))
    min_winner_rate = float(min_winner_rate)

    kept: Dict[str, List[str]] = {}
    removed_occ = 0
    removed_rate = 0
    removed_empty = 0
    total_ctx = 0
    kept_occ_sum = 0
    kept_unique_sources_sum = 0
    kept_topk_rate_sum = 0.0

    for ctx_sig, occ in occ_by_ctx_sig.items():
        total_ctx += 1
        if int(occ) < min_ctx_occ:
            removed_occ += 1
            continue

        base_counts: Dict[str, int] = {}
        if builder == "winners":
            base_counts = dict(winner_counts_by_ctx_sig.get(ctx_sig) or {})
        elif builder == "executed":
            base_counts = dict(executed_counts_by_ctx_sig.get(ctx_sig) or {})
        elif builder == "topcand":
            base_counts = dict(topcand_counts_by_ctx_sig.get(ctx_sig) or {})
            if not base_counts:
                base_counts = dict(winner_counts_by_ctx_sig.get(ctx_sig) or {})
        elif builder == "hybrid":
            base_counts = dict(winner_counts_by_ctx_sig.get(ctx_sig) or {})
            extra = topcand_counts_by_ctx_sig.get(ctx_sig) or {}
            for k, v in extra.items():
                base_counts[str(k)] = int(base_counts.get(str(k), 0) + int(v))

        items = [
            (str(k), int(v))
            for k, v in base_counts.items()
            if isinstance(k, str) and str(k) in predictor_ids
        ]
        items.sort(key=lambda kv: (-int(kv[1]), str(kv[0])))
        allowed = [str(k) for k, _v in items[:top_k]]
        if not allowed:
            removed_empty += 1
            continue

        win_row = winner_counts_by_ctx_sig.get(ctx_sig) or {}
        win_total = int(sum(int(v) for _k, v in win_row.items()))
        cov = float(sum(int(win_row.get(a, 0)) for a in allowed)) / float(max(1, win_total))
        if cov < min_winner_rate:
            removed_rate += 1
            continue

        kept[str(ctx_sig)] = allowed
        kept_occ_sum += int(occ)
        kept_unique_sources_sum += int(len(items))
        kept_topk_rate_sum += float(cov)

    kept_ctx = int(len(kept))
    meta = {
        "builder": builder,
        "top_k": int(top_k),
        "min_ctx_occ": int(min_ctx_occ),
        "min_winner_rate": float(min_winner_rate),
        "ctx_sigs_total": int(total_ctx),
        "ctx_sigs_kept": int(kept_ctx),
        "ctx_sigs_removed_min_occ": int(removed_occ),
        "ctx_sigs_removed_min_winner_rate": int(removed_rate),
        "ctx_sigs_removed_empty": int(removed_empty),
        "kept_occ_sum": int(kept_occ_sum),
        "kept_unique_sources_mean": float(kept_unique_sources_sum / max(1, kept_ctx)),
        "kept_topk_coverage_rate_mean": float(kept_topk_rate_sum / max(1, kept_ctx)),
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
                        "allowed_size": int(ex.get("allowed_size") or 0),
                        "baseline_predictors_iterated": int(ex.get("baseline_predictors_iterated") or 0),
                        "gate_predictors_iterated": int(ex.get("gate_predictors_iterated") or 0),
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
    missing_exec = base_exec - gate_exec
    if missing_exec:
        return "missing_competitor_exec"
    base_tok = str(dbg.get("baseline_token") or "")
    gate_top = dbg.get("gate_top_candidates") or []
    if base_tok and isinstance(gate_top, list):
        if any(isinstance(c, dict) and str(c.get("token") or "") == base_tok for c in gate_top):
            return "present_but_loses"
    return "missing_candidate"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Existing run dir containing acts.jsonl (read-only)")
    ap.add_argument("--out", required=True, help="New WORM out dir for CSV logs (must not exist)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--suite", choices=["chat"], default="chat")
    ap.add_argument("--enable_contracts", action="store_true")
    ap.add_argument("--no_router", action="store_true")
    ap.add_argument("--active_gate_top_k_list", default="1,2,3,4,6,8")
    ap.add_argument("--active_gate_builders", default="winners,executed,topcand,hybrid")
    ap.add_argument("--active_gate_min_ctx_occ", type=int, default=2)
    ap.add_argument("--active_gate_min_winner_rate", type=float, default=0.99)
    ap.add_argument("--active_gate_topcand_n", type=int, default=5)
    ap.add_argument("--target_mismatch_rate", type=float, default=0.001)
    ap.add_argument("--target_divergence_dialogues_rate", type=float, default=0.05)
    ap.add_argument(
        "--mismatch_diag_table_path",
        default="results/run_csv_loop_v54_sweep_seed0_try2/active_gate_table_ctxsig_top2_minocc2_minwin0p99.json",
    )
    ap.add_argument("--debug_mismatch_topn", type=int, default=5)
    ap.add_argument("--debug_mismatch_cap", type=int, default=50)
    args = ap.parse_args()

    acts_path = os.path.join(str(args.run), "acts.jsonl")
    if not os.path.exists(acts_path):
        raise SystemExit(f"Missing acts.jsonl at: {acts_path}")

    store = ActStore.load_jsonl(acts_path)
    predictor_ids = {str(a.id) for a in store.active() if getattr(a, "kind", "") == "predictor"}

    out_dir = str(args.out)
    if os.path.exists(out_dir):
        raise SystemExit(f"WORM_VIOLATION: --out already exists: {out_dir}")

    # 1) Baseline (CSV off)
    base_cfg = EngineConfig(
        enable_contracts=bool(args.enable_contracts),
        disable_macro_router=bool(args.no_router),
    )
    eng0 = Engine(store, seed=int(args.seed), config=base_cfg)
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

    tok_recs = _iter_chat_tokens(base_transcripts)
    winner_counts, executed_counts, occ_by_ctx_sig = collect_counts(tok_recs=tok_recs)

    # 2) Shadow (CSV on): lifecycle + invariance
    registry = ConceptRegistry(run_dir=out_dir)
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
        dialogues=CHAT_DIALOGUES_20X3,
        max_new_tokens=int(args.max_new_tokens),
        prefix_k=8,
        template_ngram_n=6,
        template_prefix_window=32,
        csv=csv_loop,
    )
    shadow_hash = sha256_text(transcripts_text(shadow_transcripts))
    if base_hash != shadow_hash:
        raise SystemExit("OUTPUT_INVARIANCE_FAILED: sha256(baseline)!=sha256(shadow)")

    # 3) Mismatch diagnosis from v54 table (compare+fallback; invariance required)
    mismatch_report: Dict[str, Any] = {"enabled": False}
    topcand_counts: Dict[str, Dict[str, int]] = {}
    diag_table_path = str(args.mismatch_diag_table_path or "")
    if diag_table_path and os.path.exists(diag_table_path):
        with open(diag_table_path, "r", encoding="utf-8") as f:
            d = json.load(f)
            diag_table = d.get("table") or {}
        diag_cfg = EngineConfig(
            enable_contracts=bool(args.enable_contracts),
            disable_macro_router=bool(args.no_router),
            force_predictor_ids_by_ctx_sig=diag_table if diag_table else None,
            force_gate_debug_compare=True,
            force_gate_debug_capture_mismatch_topn=int(args.debug_mismatch_topn),
            force_gate_debug_capture_mismatch_cap=int(args.debug_mismatch_cap),
        )
        eng_diag = Engine(store, seed=int(args.seed), config=diag_cfg)
        diag_transcripts, diag_metrics = run_chat_suite(
            eng_diag,
            dialogues=CHAT_DIALOGUES_20X3,
            max_new_tokens=int(args.max_new_tokens),
            prefix_k=8,
            template_ngram_n=6,
            template_prefix_window=32,
            csv=None,
        )
        diag_hash = sha256_text(transcripts_text(diag_transcripts))
        if diag_hash != base_hash:
            raise SystemExit("DIAG_INVARIANCE_FAILED: sha256(baseline)!=sha256(active_compare_diag)")

        mismatch_examples = extract_mismatch_examples(diag_transcripts, limit=50)
        cats: Dict[str, int] = {}
        for ex in mismatch_examples:
            c = classify_mismatch(ex)
            cats[c] = int(cats.get(c, 0) + 1)
            dbg = ex.get("debug") or {}
            ctx_sig = str(ex.get("ctx_sig") or "")
            if not isinstance(dbg, dict) or not ctx_sig:
                continue
            tops = dbg.get("baseline_top_candidates") or []
            if isinstance(tops, list):
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
            "enabled": True,
            "source": {"table_path": diag_table_path, "table_sha256": sha256_file(diag_table_path)},
            "sha256_transcript_text": str(diag_hash),
            "suite_metrics": dict(diag_metrics),
            "force_gate": force_gate_compare_stats_chat(diag_transcripts),
            "examples": mismatch_examples,
            "category_counts": dict(sorted((k, int(v)) for k, v in cats.items())),
        }
        with open(os.path.join(out_dir, "mismatch_report.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(mismatch_report, ensure_ascii=False, indent=2))
            f.write("\n")

    builders = parse_builder_list(str(args.active_gate_builders))
    if not builders:
        raise SystemExit("--active_gate_builders produced no valid builders")
    top_k_list = sorted(set(parse_top_k_list(str(args.active_gate_top_k_list))))
    if not top_k_list:
        raise SystemExit("--active_gate_top_k_list produced no valid ints")

    # 4) Sweep compare for builder+K (invariance required)
    sweep: List[Dict[str, Any]] = []
    for builder in builders:
        for k in top_k_list:
            gate_table, gate_meta = build_gate_table(
                builder=str(builder),
                top_k=int(k),
                min_ctx_occ=int(args.active_gate_min_ctx_occ),
                min_winner_rate=float(args.active_gate_min_winner_rate),
                predictor_ids=predictor_ids,
                occ_by_ctx_sig=occ_by_ctx_sig,
                winner_counts_by_ctx_sig=winner_counts,
                executed_counts_by_ctx_sig=executed_counts,
                topcand_counts_by_ctx_sig=topcand_counts,
            )
            token = safe_float_token(float(args.active_gate_min_winner_rate))
            table_path = os.path.join(
                out_dir,
                f"active_gate_table_ctxsig_{builder}_top{int(k)}_minocc{int(args.active_gate_min_ctx_occ)}_minwin{token}.json",
            )
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

            base_table_stats = ctx_gate_stats_chat(tok_recs=tok_recs, table=gate_table).to_dict()

            compare_cfg = EngineConfig(
                enable_contracts=bool(args.enable_contracts),
                disable_macro_router=bool(args.no_router),
                force_predictor_ids_by_ctx_sig=gate_table if gate_table else None,
                force_gate_debug_compare=True,
            )
            eng_cmp = Engine(store, seed=int(args.seed), config=compare_cfg)
            cmp_transcripts, cmp_metrics = run_chat_suite(
                eng_cmp,
                dialogues=CHAT_DIALOGUES_20X3,
                max_new_tokens=int(args.max_new_tokens),
                prefix_k=8,
                template_ngram_n=6,
                template_prefix_window=32,
                csv=None,
            )
            cmp_hash = sha256_text(transcripts_text(cmp_transcripts))
            if cmp_hash != base_hash:
                raise SystemExit(
                    f"ACTIVE_COMPARE_INVARIANCE_FAILED: builder={builder} K={k} sha256(baseline)!=sha256(active_compare)"
                )
            compare_force_gate = force_gate_compare_stats_chat(cmp_transcripts)
            sweep.append(
                {
                    "builder": str(builder),
                    "k": int(k),
                    "gate_table": {
                        "table_entries": int(len(gate_table)),
                        "table_path": str(table_path),
                        "table_sha256": sha256_file(table_path),
                        "meta": dict(gate_meta),
                        "baseline_stats": base_table_stats,
                    },
                    "active_compare": {
                        "sha256_transcript_text": str(cmp_hash),
                        "suite_metrics": dict(cmp_metrics),
                        "force_gate": dict(compare_force_gate),
                    },
                }
            )

    # 5) Active-real for 3 points per builder (min/max/compare-recommended)
    baseline_scan = _scan_cost(base_metrics)
    for builder in builders:
        rows = [r for r in sweep if r.get("builder") == builder]
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: int(r.get("k") or 0))
        min_k = int(rows_sorted[0]["k"])
        max_k = int(rows_sorted[-1]["k"])
        rec_k = min_k
        # recommend: min mismatch_rate then max pct_saved.
        cand = []
        for r in rows:
            fg = (r.get("active_compare") or {}).get("force_gate") or {}
            cand.append((float(fg.get("mismatch_rate") or 0.0), -float(fg.get("pct_saved") or 0.0), int(r["k"])))
        cand.sort(key=lambda x: (x[0], x[1], x[2]))
        if cand:
            rec_k = int(cand[0][2])

        ks_for_real = []
        for k in [min_k, rec_k, max_k]:
            if int(k) not in ks_for_real:
                ks_for_real.append(int(k))

        for r in rows:
            if int(r.get("k") or 0) not in set(ks_for_real):
                continue
            table_path = str((r.get("gate_table") or {}).get("table_path") or "")
            gate_table: Dict[str, List[str]] = {}
            if table_path and os.path.exists(table_path):
                with open(table_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    gate_table = d.get("table") or {}

            real_cfg = EngineConfig(
                enable_contracts=bool(args.enable_contracts),
                disable_macro_router=bool(args.no_router),
                force_predictor_ids_by_ctx_sig=gate_table if gate_table else None,
            )
            eng_real = Engine(store, seed=int(args.seed), config=real_cfg)
            real_transcripts, real_metrics = run_chat_suite(
                eng_real,
                dialogues=CHAT_DIALOGUES_20X3,
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
            real_scan = _scan_cost(real_metrics)
            saved_real = float(baseline_scan - real_scan)
            pct_saved_real = float(saved_real / baseline_scan) if baseline_scan > 0 else 0.0
            r["active_real"] = {
                "sha256_transcript_text": str(real_hash),
                "suite_metrics": dict(real_metrics),
                "divergence": {
                    "mismatch_count_full_text": int(mismatch_count),
                    "mismatch_total_full_text": int(mismatch_total),
                    "mismatch_rate_full_text": float(div_rate),
                    "mismatch_examples": mismatch_examples,
                },
                "real_cost": {
                    "avg_scan_cost_baseline_per_token_mean": float(baseline_scan),
                    "avg_scan_cost_active_real_per_token_mean": float(real_scan),
                    "avg_scan_cost_saved": float(saved_real),
                    "pct_saved": float(pct_saved_real),
                },
            }

    # 6) Global recommendation across builder+K
    target_mr = float(args.target_mismatch_rate)
    target_div = float(args.target_divergence_dialogues_rate)
    best: Optional[Dict[str, Any]] = None
    for r in sweep:
        fg = (r.get("active_compare") or {}).get("force_gate") or {}
        mr = float(fg.get("mismatch_rate") or 0.0)
        ur = float(fg.get("used_rate") or 0.0)
        ps = float(fg.get("pct_saved") or 0.0)
        if ur <= 0.0:
            continue
        ok = mr <= target_mr
        if "active_real" in r:
            div = float(((r["active_real"].get("divergence") or {}).get("mismatch_rate_full_text")) or 0.0)
            ok = ok and (div <= target_div)
        if not ok:
            continue
        cand = (ps, ur, -mr, str(r.get("builder") or ""), -int(r.get("k") or 0))
        if best is None or cand > best.get("_cand", ()):
            best = dict(r)
            best["_cand"] = cand

    recommendation = {
        "targets": {"mismatch_rate_le": float(target_mr), "divergence_dialogues_rate_le": float(target_div)},
        "config": None,
    }
    if best is not None:
        recommendation["config"] = {
            "builder": str(best.get("builder") or ""),
            "top_k": int(best.get("k") or 0),
            "min_ctx_occ": int(args.active_gate_min_ctx_occ),
            "min_winner_rate": float(args.active_gate_min_winner_rate),
            "topcand_n": int(args.active_gate_topcand_n),
        }

    # 7) Write summary + CSV
    summary = {
        "seed": int(args.seed),
        "suite": str(args.suite),
        "max_new_tokens": int(args.max_new_tokens),
        "acts_source_run": str(args.run),
        "out_dir": str(out_dir),
        "flags": {
            "enable_contracts": bool(args.enable_contracts),
            "no_router": bool(args.no_router),
            "active_gate_top_k_list": str(args.active_gate_top_k_list),
            "active_gate_builders": str(args.active_gate_builders),
            "active_gate_min_ctx_occ": int(args.active_gate_min_ctx_occ),
            "active_gate_min_winner_rate": float(args.active_gate_min_winner_rate),
            "active_gate_topcand_n": int(args.active_gate_topcand_n),
            "mismatch_diag_table_path": str(args.mismatch_diag_table_path),
        },
        "hashes": {
            "sha256_transcript_text_baseline": str(base_hash),
            "sha256_transcript_text_shadow": str(shadow_hash),
        },
        "output_invariance": {"baseline_vs_shadow_ok": True},
        "baseline_suite_metrics": dict(base_metrics),
        "shadow_suite_metrics": dict(shadow_metrics),
        "csv": summarize_csv_logs(run_dir=out_dir, registry=registry),
        "mismatch_report": mismatch_report,
        "sweep": sweep,
        "recommendation": recommendation,
        "sha256": {"acts_jsonl": sha256_file(acts_path)},
    }

    summary_path = os.path.join(out_dir, "active_gate_builder_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n")

    csv_path = os.path.join(out_dir, "active_gate_builder_summary.csv")
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "builder",
                    "k",
                    "table_entries",
                    "used_rate",
                    "mismatch_rate",
                    "mismatch_rate_on_used",
                    "would_save_pct",
                    "real_divergence_rate",
                    "real_pct_saved",
                ]
            )
            for r in sweep:
                fg = ((r.get("active_compare") or {}).get("force_gate") or {}) if r else {}
                real_div = ""
                real_saved = ""
                if "active_real" in r:
                    real_div = str(
                        float(((r["active_real"].get("divergence") or {}).get("mismatch_rate_full_text")) or 0.0)
                    )
                    real_saved = str(
                        float(((r["active_real"].get("real_cost") or {}).get("pct_saved")) or 0.0)
                    )
                w.writerow(
                    [
                        str(r.get("builder") or ""),
                        int(r.get("k") or 0),
                        int(((r.get("gate_table") or {}).get("table_entries")) or 0),
                        float(fg.get("used_rate") or 0.0),
                        float(fg.get("mismatch_rate") or 0.0),
                        float(fg.get("mismatch_rate_on_used") or 0.0),
                        float(fg.get("pct_saved") or 0.0),
                        real_div,
                        real_saved,
                    ]
                )
    except Exception:
        pass

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
