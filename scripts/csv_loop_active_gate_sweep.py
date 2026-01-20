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
            st.fallback_reasons[reason] = int(st.fallback_reasons.get(reason, 0) + 1)
    return st


def collect_ctx_sig_winner_counts_chat(
    transcripts: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for ctx_sig, win in _iter_ctx_sig_winners_chat(transcripts):
        row = counts.get(ctx_sig)
        if row is None:
            row = {}
            counts[ctx_sig] = row
        row[str(win)] = int(row.get(str(win), 0)) + 1
    return counts


def build_gate_table_filtered(
    *,
    counts_by_ctx_sig: Dict[str, Dict[str, int]],
    top_k: int,
    min_ctx_occ: int,
    min_winner_rate: float,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    top_k = max(1, int(top_k))
    min_ctx_occ = max(1, int(min_ctx_occ))
    min_winner_rate = float(min_winner_rate)

    kept: Dict[str, List[str]] = {}
    removed_occ = 0
    removed_rate = 0
    total_ctx = 0
    kept_occ_sum = 0
    kept_unique_winners_sum = 0
    kept_top1_rate_sum = 0.0
    kept_topk_rate_sum = 0.0

    for ctx_sig, row in counts_by_ctx_sig.items():
        total_ctx += 1
        items = sorted(row.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        total = int(sum(int(v) for _k, v in items))
        if total < min_ctx_occ:
            removed_occ += 1
            continue
        allowed = [str(k) for k, _v in items[:top_k]]
        if not allowed:
            removed_rate += 1
            continue
        cov = float(sum(int(row.get(a, 0)) for a in allowed)) / float(max(1, total))
        top1_rate = float(int(items[0][1])) / float(max(1, total)) if items else 0.0
        if cov < min_winner_rate:
            removed_rate += 1
            continue
        kept[str(ctx_sig)] = allowed
        kept_occ_sum += total
        kept_unique_winners_sum += int(len(row))
        kept_top1_rate_sum += float(top1_rate)
        kept_topk_rate_sum += float(cov)

    kept_ctx = int(len(kept))
    meta = {
        "top_k": int(top_k),
        "min_ctx_occ": int(min_ctx_occ),
        "min_winner_rate": float(min_winner_rate),
        "ctx_sigs_total": int(total_ctx),
        "ctx_sigs_kept": int(kept_ctx),
        "ctx_sigs_removed_min_occ": int(removed_occ),
        "ctx_sigs_removed_min_winner_rate": int(removed_rate),
        "kept_occ_sum": int(kept_occ_sum),
        "kept_unique_winners_mean": float(kept_unique_winners_sum / max(1, kept_ctx)),
        "kept_top1_rate_mean": float(kept_top1_rate_sum / max(1, kept_ctx)),
        "kept_topk_coverage_rate_mean": float(kept_topk_rate_sum / max(1, kept_ctx)),
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


def _scan_cost(m: Dict[str, Any]) -> float:
    v = m.get("scan_acts_considered_per_token_mean")
    if v is None:
        return 0.0
    return float(v)


def safe_float_token(x: float) -> str:
    s = f"{float(x):.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p").replace("-", "m")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Existing run dir containing acts.jsonl (read-only)")
    ap.add_argument("--out", required=True, help="New WORM out dir for CSV logs (must not exist)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--suite", choices=["chat", "skill"], default="chat")
    ap.add_argument("--enable_contracts", action="store_true")
    ap.add_argument("--no_router", action="store_true", help="Disable macro_router usage (baseline high mode)")

    ap.add_argument("--active_gate_top_k_list", default="1,2,3,4,6,8")
    ap.add_argument("--active_gate_mode", choices=["compare", "real", "both"], default="compare")
    ap.add_argument("--active_gate_min_ctx_occ", type=int, default=1)
    ap.add_argument("--active_gate_min_winner_rate", type=float, default=0.0)

    ap.add_argument("--target_mismatch_rate", type=float, default=0.001, help="Target mismatch_rate (compare)")
    ap.add_argument(
        "--target_divergence_dialogues_rate",
        type=float,
        default=0.05,
        help="Target divergence rate in active_real (mismatch_count_full_text/total)",
    )
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

    # 2) Shadow (CSV on, must be invariant) — lifecycle logs
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
    if str(args.suite) == "skill":
        shadow_transcripts, shadow_metrics = run_skill_suite(
            eng1, tasks=SKILL_DIALOGUES_V0, max_new_tokens=int(args.max_new_tokens), csv=csv_loop
        )
    else:
        shadow_transcripts, shadow_metrics = run_chat_suite(
            eng1,
            dialogues=CHAT_DIALOGUES_20X3,
            max_new_tokens=int(args.max_new_tokens),
            prefix_k=8,
            template_ngram_n=6,
            template_prefix_window=32,
            csv=csv_loop,
        )
    shadow_text = transcripts_text(shadow_transcripts)
    shadow_hash = sha256_text(shadow_text)
    if base_hash != shadow_hash:
        raise SystemExit("OUTPUT_INVARIANCE_FAILED: sha256(baseline)!=sha256(shadow)")

    top_k_list = parse_top_k_list(str(args.active_gate_top_k_list))
    if not top_k_list:
        raise SystemExit("--active_gate_top_k_list produced no valid ints")
    top_k_list = sorted(set(int(x) for x in top_k_list))

    sweep: List[Dict[str, Any]] = []
    counts_by_ctx_sig: Dict[str, Dict[str, int]] = {}
    if str(args.suite) == "chat":
        counts_by_ctx_sig = collect_ctx_sig_winner_counts_chat(base_transcripts)

    # 3) Sweep compare for all Ks
    for k in top_k_list:
        gate_table: Dict[str, List[str]] = {}
        gate_table_meta: Dict[str, Any] = {}
        table_path = ""
        base_table_stats: Optional[Dict[str, Any]] = None

        if str(args.suite) == "chat":
            gate_table, gate_table_meta = build_gate_table_filtered(
                counts_by_ctx_sig=counts_by_ctx_sig,
                top_k=int(k),
                min_ctx_occ=int(args.active_gate_min_ctx_occ),
                min_winner_rate=float(args.active_gate_min_winner_rate),
            )
            token = safe_float_token(float(args.active_gate_min_winner_rate))
            table_path = os.path.join(
                out_dir,
                f"active_gate_table_ctxsig_top{int(k)}_minocc{int(args.active_gate_min_ctx_occ)}_minwin{token}.json",
            )
            with open(table_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"meta": gate_table_meta, "table": gate_table},
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
                f.write("\n")
            base_table_stats = ctx_gate_stats_chat(transcripts=base_transcripts, table=gate_table).to_dict()

        compare_cfg = EngineConfig(
            enable_contracts=bool(args.enable_contracts),
            disable_macro_router=bool(args.no_router),
            force_predictor_ids_by_ctx_sig=gate_table if gate_table else None,
            force_gate_debug_compare=True,
        )
        eng_cmp = Engine(store, seed=int(args.seed), config=compare_cfg)
        if str(args.suite) == "skill":
            cmp_transcripts, cmp_metrics = run_skill_suite(
                eng_cmp, tasks=SKILL_DIALOGUES_V0, max_new_tokens=int(args.max_new_tokens), csv=None
            )
        else:
            cmp_transcripts, cmp_metrics = run_chat_suite(
                eng_cmp,
                dialogues=CHAT_DIALOGUES_20X3,
                max_new_tokens=int(args.max_new_tokens),
                prefix_k=8,
                template_ngram_n=6,
                template_prefix_window=32,
                csv=None,
            )
        cmp_text = transcripts_text(cmp_transcripts)
        cmp_hash = sha256_text(cmp_text)
        if cmp_hash != base_hash:
            raise SystemExit(
                f"ACTIVE_COMPARE_INVARIANCE_FAILED: K={k} sha256(baseline)!=sha256(active_compare)"
            )

        compare_force_gate = {}
        if str(args.suite) == "chat":
            compare_force_gate = force_gate_compare_stats_chat(cmp_transcripts)

        sweep.append(
            {
                "k": int(k),
                "gate_table": {
                    "table_entries": int(len(gate_table)),
                    "table_path": str(table_path),
                    "table_sha256": sha256_file(table_path) if table_path else "",
                    "meta": dict(gate_table_meta),
                    **({"baseline_stats": base_table_stats} if base_table_stats is not None else {}),
                },
                "active_compare": {
                    "sha256_transcript_text": str(cmp_hash),
                    "suite_metrics": dict(cmp_metrics),
                    "force_gate": dict(compare_force_gate),
                },
            }
        )

    # 4) Active-real (optional): decide which Ks to run
    do_real = str(args.active_gate_mode) in {"real", "both"} and str(args.suite) == "chat"
    ks_for_real: List[int] = []
    if do_real:
        if str(args.active_gate_mode) == "real":
            ks_for_real = [int(x) for x in top_k_list]
        else:
            # Run real on a small, informative subset:
            # - compare-recommended K (min mismatch, then max savings),
            # - min/max K for trade-off curve,
            # - and any K that already meets the mismatch target.
            candidates = []
            for row in sweep:
                fg = (row.get("active_compare") or {}).get("force_gate") or {}
                mr = float(fg.get("mismatch_rate") or 0.0)
                ur = float(fg.get("used_rate") or 0.0)
                ps = float(fg.get("pct_saved") or 0.0)
                if ur <= 0.0:
                    continue
                candidates.append((mr, -ps, int(row["k"])))
            candidates.sort(key=lambda kv: (kv[0], kv[1], kv[2]))
            rec_k = int(candidates[0][2]) if candidates else int(top_k_list[0])

            ks_for_real = [int(rec_k), int(min(top_k_list)), int(max(top_k_list))]
            for mr, _neg_ps, k in candidates:
                if mr <= float(args.target_mismatch_rate) and int(k) not in ks_for_real:
                    ks_for_real.append(int(k))
            # Hard cap to keep the demo fast/deterministic.
            ks_for_real = ks_for_real[:5]

    for row in sweep:
        k = int(row["k"])
        if k not in set(ks_for_real):
            continue
        table_info = row.get("gate_table") or {}
        table_path = str(table_info.get("table_path") or "")
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
        real_text = transcripts_text(real_transcripts)
        real_hash = sha256_text(real_text)

        mismatch_count, mismatch_total, mismatch_examples = compare_transcripts_by_full_text(
            base_transcripts, real_transcripts
        )
        div_rate = float(mismatch_count / max(1, mismatch_total))
        baseline_scan = _scan_cost(base_metrics)
        real_scan = _scan_cost(real_metrics)
        saved_real = float(baseline_scan - real_scan)
        pct_saved_real = float(saved_real / baseline_scan) if baseline_scan > 0 else 0.0

        real_gate_stats = ctx_gate_stats_chat(transcripts=real_transcripts, table=gate_table).to_dict()
        row["active_real"] = {
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
            "gate_stats": dict(real_gate_stats),
        }

    # 5) Recommendation (simple Pareto-ish)
    target_mr = float(args.target_mismatch_rate)
    target_div = float(args.target_divergence_dialogues_rate)
    best = None
    for row in sweep:
        fg = (row.get("active_compare") or {}).get("force_gate") or {}
        mr = float(fg.get("mismatch_rate") or 0.0)
        ur = float(fg.get("used_rate") or 0.0)
        ps = float(fg.get("pct_saved") or 0.0)
        if ur <= 0.0:
            continue
        ok = mr <= target_mr
        if "active_real" in row:
            div = float(((row["active_real"].get("divergence") or {}).get("mismatch_rate_full_text")) or 0.0)
            ok = ok and (div <= target_div)
        if not ok:
            continue
        cand = (ps, ur, -mr, -int(row["k"]))
        if best is None or cand > best[0]:
            best = (cand, row)

    if best is None:
        # Fallback: lowest mismatch_rate, then highest pct_saved.
        ordered = []
        for row in sweep:
            fg = (row.get("active_compare") or {}).get("force_gate") or {}
            mr = float(fg.get("mismatch_rate") or 0.0)
            ps = float(fg.get("pct_saved") or 0.0)
            ordered.append((mr, -ps, int(row["k"]), row))
        ordered.sort(key=lambda x: (x[0], x[1], x[2]))
        rec_row = ordered[0][3] if ordered else None
    else:
        rec_row = best[1]

    recommendation = {
        "targets": {
            "mismatch_rate_le": float(target_mr),
            "divergence_dialogues_rate_le": float(target_div),
        },
        "config": {
            "top_k": int(rec_row["k"]) if rec_row is not None else None,
            "min_ctx_occ": int(args.active_gate_min_ctx_occ),
            "min_winner_rate": float(args.active_gate_min_winner_rate),
        },
    }

    # 6) Write summary + optional CSV
    summary = {
        "seed": int(args.seed),
        "suite": str(args.suite),
        "max_new_tokens": int(args.max_new_tokens),
        "acts_source_run": str(args.run),
        "out_dir": str(out_dir),
        "flags": {
            "enable_contracts": bool(args.enable_contracts),
            "no_router": bool(args.no_router),
            "active_gate_mode": str(args.active_gate_mode),
            "active_gate_top_k_list": str(args.active_gate_top_k_list),
            "active_gate_min_ctx_occ": int(args.active_gate_min_ctx_occ),
            "active_gate_min_winner_rate": float(args.active_gate_min_winner_rate),
        },
        "hashes": {
            "sha256_transcript_text_baseline": str(base_hash),
            "sha256_transcript_text_shadow": str(shadow_hash),
        },
        "output_invariance": {"baseline_vs_shadow_ok": True},
        "baseline_suite_metrics": dict(base_metrics),
        "shadow_suite_metrics": dict(shadow_metrics),
        "csv": summarize_csv_logs(run_dir=out_dir, registry=registry),
        "sweep": sweep,
        "recommendation": recommendation,
        "sha256": {
            "acts_jsonl": sha256_file(acts_path),
        },
    }

    summary_path = os.path.join(out_dir, "active_gate_sweep_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n")

    csv_path = os.path.join(out_dir, "active_gate_sweep_summary.csv")
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "k",
                    "table_entries",
                    "baseline_coverage_rate",
                    "compare_used_rate",
                    "compare_mismatch_rate",
                    "compare_mismatch_rate_on_used",
                    "compare_pct_saved_would",
                    "real_divergence_rate",
                    "real_pct_saved",
                ]
            )
            for row in sweep:
                k = int(row["k"])
                ti = row.get("gate_table") or {}
                base_cov = None
                if isinstance(ti.get("baseline_stats"), dict):
                    base_cov = float(ti["baseline_stats"].get("coverage_rate") or 0.0)
                fg = ((row.get("active_compare") or {}).get("force_gate") or {}) if row else {}
                real_div = ""
                real_saved = ""
                if "active_real" in row:
                    real_div = str(
                        float(((row["active_real"].get("divergence") or {}).get("mismatch_rate_full_text")) or 0.0)
                    )
                    real_saved = str(
                        float(((row["active_real"].get("real_cost") or {}).get("pct_saved")) or 0.0)
                    )
                w.writerow(
                    [
                        k,
                        int(ti.get("table_entries") or 0),
                        float(base_cov or 0.0),
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
