#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.engine import Engine, EngineConfig
from atos_core.metrics import tokenize_text
from atos_core.store import ActStore
from atos_core.suite import CHAT_DIALOGUES_20X3, run_chat_suite


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def transcripts_text(transcripts: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(str(r.get("full_text", "")) for r in transcripts)


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _require_aligned_list(
    tr: Dict[str, Any], *, key: str, want_len: int, where: str
) -> List[Any]:
    v = tr.get(key)
    if not isinstance(v, list):
        raise ValueError(f"{where}: trace[{key}] is not a list")
    if len(v) != int(want_len):
        raise ValueError(f"{where}: len(trace[{key}])={len(v)} != {int(want_len)}")
    return v


def sum_trace_metric(transcripts: Sequence[Dict[str, Any]], *, metric_key: str) -> Tuple[int, int]:
    total = 0
    tokens = 0
    for rec in transcripts:
        pid = rec.get("prompt_id")
        turns = rec.get("turns", [])
        if not isinstance(turns, list):
            continue
        for turn_idx, t in enumerate(turns):
            if not isinstance(t, dict):
                continue
            tr = t.get("trace") or {}
            if not isinstance(tr, dict):
                continue
            winners = tr.get("selected_source_act_ids") or []
            if not isinstance(winners, list):
                continue
            L = int(len(winners))
            where = f"prompt_id={pid} turn={turn_idx}"
            vals = _require_aligned_list(tr, key=metric_key, want_len=L, where=where)
            total += int(sum(_safe_int(x) for x in vals))
            tokens += L
    return total, tokens


def first_token_diff(
    baseline: Sequence[Dict[str, Any]], gate: Sequence[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    base_by: Dict[int, Dict[str, Any]] = {}
    gate_by: Dict[int, Dict[str, Any]] = {}
    for rec in baseline:
        try:
            base_by[int(rec.get("prompt_id", -1))] = rec
        except Exception:
            continue
    for rec in gate:
        try:
            gate_by[int(rec.get("prompt_id", -1))] = rec
        except Exception:
            continue

    all_ids = sorted(set(base_by.keys()) | set(gate_by.keys()))
    for pid in all_ids:
        b = base_by.get(pid)
        g = gate_by.get(pid)
        if b is None or g is None:
            return {
                "prompt_id": int(pid),
                "turn": 0,
                "token_index": 0,
                "baseline_token": "<MISSING_PROMPT>",
                "gate_token": "<MISSING_PROMPT>",
            }

        b_turns = b.get("turns", [])
        g_turns = g.get("turns", [])
        if not isinstance(b_turns, list) or not isinstance(g_turns, list):
            continue

        n_turns = max(len(b_turns), len(g_turns))
        for turn_idx in range(n_turns):
            bt = b_turns[turn_idx] if turn_idx < len(b_turns) else {}
            gt = g_turns[turn_idx] if turn_idx < len(g_turns) else {}
            b_sys = str(bt.get("system", "")) if isinstance(bt, dict) else ""
            g_sys = str(gt.get("system", "")) if isinstance(gt, dict) else ""
            b_toks = tokenize_text(b_sys)
            g_toks = tokenize_text(g_sys)
            m = min(len(b_toks), len(g_toks))
            for i in range(m):
                if b_toks[i] != g_toks[i]:
                    return {
                        "prompt_id": int(pid),
                        "turn": int(turn_idx),
                        "token_index": int(i),
                        "baseline_token": str(b_toks[i]),
                        "gate_token": str(g_toks[i]),
                    }
            if len(b_toks) != len(g_toks):
                b_tok = str(b_toks[m]) if m < len(b_toks) else "<END>"
                g_tok = str(g_toks[m]) if m < len(g_toks) else "<END>"
                return {
                    "prompt_id": int(pid),
                    "turn": int(turn_idx),
                    "token_index": int(m),
                    "baseline_token": b_tok,
                    "gate_token": g_tok,
                }
    return None


def collect_gate_live_metrics(
    transcripts: Sequence[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    total_tokens = 0
    covered = 0
    winner_ok = 0
    fastpath = 0
    fallbacks = 0
    mismatches = 0

    it_sum = 0
    matched_sum = 0
    emitted_sum = 0
    dbg_base_it_sum = 0
    dbg_base_mat_sum = 0
    dbg_base_emit_sum = 0

    reason_counts: Counter = Counter()
    examples: List[Dict[str, Any]] = []

    for rec in transcripts:
        pid = rec.get("prompt_id")
        turns = rec.get("turns", [])
        if not isinstance(turns, list):
            continue
        for turn_idx, t in enumerate(turns):
            if not isinstance(t, dict):
                continue
            tr = t.get("trace") or {}
            if not isinstance(tr, dict):
                continue

            winners = tr.get("selected_source_act_ids") or []
            if not isinstance(winners, list):
                continue
            L = int(len(winners))
            if L <= 0:
                continue

            where = f"prompt_id={pid} turn={turn_idx}"
            allowed = _require_aligned_list(
                tr, key="router_live_allowed_predictor_ids", want_len=L, where=where
            )
            used = _require_aligned_list(tr, key="router_live_used", want_len=L, where=where)
            fallback = _require_aligned_list(tr, key="router_live_fallback", want_len=L, where=where)
            reasons = _require_aligned_list(
                tr, key="router_live_fallback_reason", want_len=L, where=where
            )
            mismatch = _require_aligned_list(tr, key="router_live_mismatch", want_len=L, where=where)
            it = _require_aligned_list(tr, key="predictor_iterated", want_len=L, where=where)
            matched = _require_aligned_list(tr, key="predictor_matched", want_len=L, where=where)
            emitted = _require_aligned_list(tr, key="predictor_emitted", want_len=L, where=where)
            ctx_keys = _require_aligned_list(tr, key="context_keys", want_len=L, where=where)
            dbg_base_tok = _require_aligned_list(
                tr, key="router_live_debug_baseline_token", want_len=L, where=where
            )
            dbg_gate_tok = _require_aligned_list(
                tr, key="router_live_debug_gate_token", want_len=L, where=where
            )
            dbg_base_it = _require_aligned_list(
                tr, key="baseline_predictors_iterated", want_len=L, where=where
            )
            dbg_base_mat = _require_aligned_list(
                tr, key="baseline_predictors_matched", want_len=L, where=where
            )
            dbg_base_emit = _require_aligned_list(
                tr, key="baseline_predictors_emitted", want_len=L, where=where
            )

            for i in range(L):
                total_tokens += 1

                it_sum += _safe_int(it[i])
                matched_sum += _safe_int(matched[i])
                emitted_sum += _safe_int(emitted[i])
                dbg_base_it_sum += _safe_int(dbg_base_it[i])
                dbg_base_mat_sum += _safe_int(dbg_base_mat[i])
                dbg_base_emit_sum += _safe_int(dbg_base_emit[i])

                a: List[str] = []
                if isinstance(allowed[i], list):
                    a = [str(x) for x in allowed[i] if isinstance(x, str) and x]
                if a:
                    covered += 1
                    if isinstance(winners[i], str) and winners[i] in set(a):
                        winner_ok += 1

                if _safe_int(used[i]) == 1:
                    fastpath += 1
                if _safe_int(fallback[i]) == 1:
                    fallbacks += 1
                    r = str(reasons[i] or "")
                    reason_counts[r] += 1

                is_mm = _safe_int(mismatch[i]) == 1
                if is_mm:
                    mismatches += 1
                    if len(examples) < 3:
                        examples.append(
                            {
                                "prompt_id": pid,
                                "turn": int(turn_idx),
                                "token_index": int(i),
                                "mode": str(t.get("mode") or "default"),
                                "ctx_key": str(ctx_keys[i]),
                                "baseline_token": str(dbg_base_tok[i]),
                                "gate_token": str(dbg_gate_tok[i]),
                                "fallback_reason": str(reasons[i] or ""),
                            }
                        )

    metrics: Dict[str, Any] = {
        "tokens": int(total_tokens),
        "coverage_rate": float(covered / max(1, total_tokens)),
        "winner_in_allowed_rate": float(winner_ok / max(1, total_tokens)),
        "live_fastpath_rate": float(fastpath / max(1, total_tokens)),
        "live_fallback_rate": float(fallbacks / max(1, total_tokens)),
        "live_mismatch_rate": float(mismatches / max(1, total_tokens)),
        "mismatch_count": int(mismatches),
        "fallback_reason_counts": {k: int(v) for k, v in sorted(reason_counts.items()) if k},
        "predictors_iterated_sum": int(it_sum),
        "predictors_iterated_per_token_mean": float(it_sum / max(1, total_tokens)),
        "predictors_matched_sum": int(matched_sum),
        "predictors_matched_per_token_mean": float(matched_sum / max(1, total_tokens)),
        "predictors_emitted_sum": int(emitted_sum),
        "predictors_emitted_per_token_mean": float(emitted_sum / max(1, total_tokens)),
        "debug_baseline_predictors_iterated_sum": int(dbg_base_it_sum),
        "debug_baseline_predictors_iterated_per_token_mean": float(
            dbg_base_it_sum / max(1, total_tokens)
        ),
        "debug_baseline_predictors_matched_sum": int(dbg_base_mat_sum),
        "debug_baseline_predictors_matched_per_token_mean": float(
            dbg_base_mat_sum / max(1, total_tokens)
        ),
        "debug_baseline_predictors_emitted_sum": int(dbg_base_emit_sum),
        "debug_baseline_predictors_emitted_per_token_mean": float(
            dbg_base_emit_sum / max(1, total_tokens)
        ),
    }
    return metrics, examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run dir containing acts.jsonl (read-only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--enable_contracts", action="store_true", help="Enable deterministic instruction contracts.")
    args = ap.parse_args()

    acts_path = os.path.join(args.run, "acts.jsonl")
    store = ActStore.load_jsonl(acts_path)

    base_engine = Engine(store, seed=args.seed, config=EngineConfig(enable_contracts=bool(args.enable_contracts)))
    gate_proof_engine = Engine(
        store,
        seed=args.seed,
        config=EngineConfig(
            router_live_enabled=True,
            router_live_debug_compare=True,
            enable_contracts=bool(args.enable_contracts),
        ),
    )
    gate_live_engine = Engine(
        store,
        seed=args.seed,
        config=EngineConfig(
            router_live_enabled=True,
            router_live_debug_compare=False,
            enable_contracts=bool(args.enable_contracts),
        ),
    )

    base_transcripts, _ = run_chat_suite(
        base_engine,
        dialogues=CHAT_DIALOGUES_20X3,
        max_new_tokens=args.max_new_tokens,
        prefix_k=8,
        template_ngram_n=6,
        template_prefix_window=32,
    )
    gate_proof_transcripts, _ = run_chat_suite(
        gate_proof_engine,
        dialogues=CHAT_DIALOGUES_20X3,
        max_new_tokens=args.max_new_tokens,
        prefix_k=8,
        template_ngram_n=6,
        template_prefix_window=32,
    )
    gate_live_transcripts, _ = run_chat_suite(
        gate_live_engine,
        dialogues=CHAT_DIALOGUES_20X3,
        max_new_tokens=args.max_new_tokens,
        prefix_k=8,
        template_ngram_n=6,
        template_prefix_window=32,
    )

    base_txt = transcripts_text(base_transcripts)
    gate_proof_txt = transcripts_text(gate_proof_transcripts)
    gate_live_txt = transcripts_text(gate_live_transcripts)

    base_sha = sha256_text(base_txt)
    gate_proof_sha = sha256_text(gate_proof_txt)
    gate_live_sha = sha256_text(gate_live_txt)

    if gate_proof_sha != base_sha:
        diff = first_token_diff(base_transcripts, gate_proof_transcripts)
        print(
            json.dumps(
                {
                    "error": "sha_mismatch_gate_proof",
                    "sha256_baseline": base_sha,
                    "sha256_gate_proof": gate_proof_sha,
                    "first_diff": diff,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1)

    if gate_live_sha != base_sha:
        diff = first_token_diff(base_transcripts, gate_live_transcripts)
        print(
            json.dumps(
                {
                    "error": "sha_mismatch_gate_live",
                    "sha256_baseline": base_sha,
                    "sha256_gate_live": gate_live_sha,
                    "first_diff": diff,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1)

    base_it_sum, base_tokens = sum_trace_metric(base_transcripts, metric_key="predictor_iterated")
    base_mat_sum, _ = sum_trace_metric(base_transcripts, metric_key="predictor_matched")
    base_emit_sum, _ = sum_trace_metric(base_transcripts, metric_key="predictor_emitted")

    proof_metrics, proof_examples = collect_gate_live_metrics(gate_proof_transcripts)
    live_metrics, live_examples = collect_gate_live_metrics(gate_live_transcripts)

    def _skip_rate(base: int, cur: int) -> float:
        if base <= 0:
            return 0.0
        return float((base - cur) / base)

    proof_it_sum = int(proof_metrics.get("predictors_iterated_sum", 0) or 0)
    proof_mat_sum = int(proof_metrics.get("predictors_matched_sum", 0) or 0)
    proof_emit_sum = int(proof_metrics.get("predictors_emitted_sum", 0) or 0)

    live_it_sum = int(live_metrics.get("predictors_iterated_sum", 0) or 0)
    live_mat_sum = int(live_metrics.get("predictors_matched_sum", 0) or 0)
    live_emit_sum = int(live_metrics.get("predictors_emitted_sum", 0) or 0)

    out: Dict[str, Any] = {
        "run": str(args.run),
        "seed": int(args.seed),
        "max_new_tokens": int(args.max_new_tokens),
        "sha256_transcript_text": {
            "baseline": str(base_sha),
            "gate_proof": str(gate_proof_sha),
            "gate_live": str(gate_live_sha),
        },
        "baseline": {
            "tokens": int(base_tokens),
            "predictors_iterated_sum": int(base_it_sum),
            "predictors_iterated_per_token_mean": float(base_it_sum / max(1, base_tokens)),
            "predictors_matched_sum": int(base_mat_sum),
            "predictors_matched_per_token_mean": float(base_mat_sum / max(1, base_tokens)),
            "predictors_emitted_sum": int(base_emit_sum),
            "predictors_emitted_per_token_mean": float(base_emit_sum / max(1, base_tokens)),
        },
        "gate_proof": {
            **proof_metrics,
            "mismatch_examples": proof_examples,
            "live_would_skip_rate_iterated": float(_skip_rate(int(base_it_sum), int(proof_it_sum))),
            "live_would_skip_rate_matched": float(_skip_rate(int(base_mat_sum), int(proof_mat_sum))),
            "live_would_skip_rate_emitted": float(_skip_rate(int(base_emit_sum), int(proof_emit_sum))),
        },
        "gate_live": {
            **live_metrics,
            "mismatch_examples": live_examples,
            "live_would_skip_rate_iterated": float(_skip_rate(int(base_it_sum), int(live_it_sum))),
            "live_would_skip_rate_matched": float(_skip_rate(int(base_mat_sum), int(live_mat_sum))),
            "live_would_skip_rate_emitted": float(_skip_rate(int(base_emit_sum), int(live_emit_sum))),
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
