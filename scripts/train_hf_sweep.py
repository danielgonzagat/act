#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.hf_corpus import build_hf_corpus
from atos_core.learn import KAAbsoluteTrainer, TrainConfig
from atos_core.suite import SKILL_SUITE_PACKS


def _default_corpus_path(source: str, split: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in source)
    safe = safe.strip("_") or "hf"
    return os.path.join("data", "raw", "hf_corpus", f"{safe}__{split}.txt")


def _is_sota_pack_at_least(pack: str, *, min_version: int) -> bool:
    p = str(pack or "").strip().lower()
    if not p.startswith("sota_v"):
        return False
    try:
        v = int(p.split("sota_v", 1)[1])
    except Exception:
        return False
    return int(v) >= int(min_version)


def _parse_steps_list(raw: str) -> List[int]:
    out: List[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("empty --steps_list")
    return out


def _read_report_last(report_path: str) -> Dict[str, Any]:
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        if isinstance(rep, list) and rep:
            last = rep[-1]
            return dict(last) if isinstance(last, dict) else {}
    except Exception:
        return {}
    return {}


def _summ_row(out_dir: str, last: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "out_dir": str(out_dir),
        "step": int(last.get("step", 0) or 0),
        "gain_horizon_steps": int(last.get("gain_horizon_steps", 0) or 0),
        "nll_bits_mean": float(last.get("nll_bits_mean", float("nan"))),
        "nll_bits_ema": float(last.get("nll_bits_ema", float("nan"))),
        "holdout_enabled": bool(last.get("holdout_enabled", False)),
        "holdout_frac": float(last.get("holdout_frac", 0.0) or 0.0),
        "holdout_nll_bits_mean": (
            float(last.get("holdout_nll_bits_mean", float("nan")))
            if last.get("holdout_nll_bits_mean") is not None
            else float("nan")
        ),
        "holdout_nll_ratio": (
            float(last.get("holdout_nll_ratio", float("nan")))
            if last.get("holdout_nll_ratio") is not None
            else float("nan")
        ),
        "mdl_net_est_bits": float(last.get("mdl_net_est_bits", float("nan"))),
        "model_cost_bits": int(last.get("model_cost_bits", 0) or 0),
        "vocab_size": int(last.get("vocab_size", 0) or 0),
        "num_acts": int(last.get("num_acts", 0) or 0),
        "adds": int(last.get("adds", 0) or 0),
        "merges": int(last.get("merges", 0) or 0),
        "prunes": int(last.get("prunes", 0) or 0),
        "fluency_lambda_effective": float(last.get("fluency_lambda_effective", float("nan"))),
        "repeat3_global": float(last.get("repeat3_global", float("nan"))),
        "loop_rate_global": float(last.get("loop_rate_global", float("nan"))),
        "utility_pass_rate": float(last.get("utility_pass_rate", float("nan"))),
        "tokens_per_s": float(last.get("tokens_per_s", float("nan"))),
        "acts_hash": str(last.get("acts_hash", "")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="HF dataset/model id or URL (owner/repo)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--dataset", default=None, help="Override resolved dataset id (owner/repo)")
    ap.add_argument("--config", dest="name", default=None, help="HF dataset config name")
    ap.add_argument("--corpus_out", default=None, help="Where to write the flat text corpus")
    ap.add_argument("--target_bytes", type=int, default=20_000_000)
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--max_chars_per_example", type=int, default=8_000)
    ap.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--shuffle_buffer", type=int, default=10_000)
    ap.add_argument("--prepare_force", action="store_true")

    ap.add_argument("--out_root", required=True, help="Artifacts root dir (recommended: artifacts/...)")
    ap.add_argument("--steps_list", default="500,1000,2000,5000")
    ap.add_argument(
        "--gain_horizon_steps",
        type=int,
        default=0,
        help="If 0, defaults to max(steps_list). If >0, used for patch MDL gain estimation.",
    )
    ap.add_argument(
        "--nll_eval_windows",
        type=int,
        default=3,
        help="Cross-context pressure: evaluate online NLL on N deterministic windows and use the mean.",
    )
    ap.add_argument("--holdout_frac", type=float, default=0.0)
    ap.add_argument("--holdout_eval_windows", type=int, default=0)
    ap.add_argument("--holdout_eval_tokens", type=int, default=0)
    ap.add_argument("--survival_plateau_windows", type=int, default=3)
    ap.add_argument("--survival_improve_tol", type=float, default=1e-4)
    ap.add_argument("--survival_hard_fail_windows", type=int, default=0)
    ap.add_argument("--survival_no_abstraction_windows", type=int, default=6)
    ap.add_argument("--survival_reuse_min", type=float, default=0.01)
    ap.add_argument("--survival_no_reuse_windows", type=int, default=6)
    ap.add_argument("--survival_concept_no_add_windows", type=int, default=3)
    ap.add_argument("--survival_concept_reuse_stall_windows", type=int, default=3)
    ap.add_argument("--survival_concept_reuse_tol", type=float, default=1e-6)
    ap.add_argument("--survival_concept_hard_fail_windows", type=int, default=3)
    ap.add_argument("--survival_concept_composed_no_add_windows", type=int, default=3)
    ap.add_argument("--survival_concept_composed_rate_stall_windows", type=int, default=3)
    ap.add_argument("--survival_concept_composed_rate_tol", type=float, default=1e-6)
    ap.add_argument("--survival_concept_composed_hard_fail_windows", type=int, default=3)
    ap.add_argument("--survival_concept_deep_rate_stall_windows", type=int, default=3)
    ap.add_argument("--survival_concept_deep_rate_tol", type=float, default=1e-6)
    ap.add_argument("--survival_concept_deep_hard_fail_windows", type=int, default=3)
    ap.add_argument("--survival_concept_very_deep_rate_stall_windows", type=int, default=3)
    ap.add_argument("--survival_concept_very_deep_rate_tol", type=float, default=1e-6)
    ap.add_argument("--survival_concept_very_deep_hard_fail_windows", type=int, default=3)
    ap.add_argument("--repair_max_windows", type=int, default=4)
    ap.add_argument("--repair_bottleneck_tol", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=0, help="If 0, auto: clamp(steps//20, 50..10000)")
    ap.add_argument("--mode", choices=["demo", "pure"], default="pure")
    ap.add_argument("--selection_mode", choices=["weighted", "bottleneck", "survival"], default="weighted")
    ap.add_argument("--fluency_lambda", type=float, default=20000.0)
    ap.add_argument(
        "--fluency_lambda_schedule",
        choices=["constant", "linear_warmup"],
        default="linear_warmup",
    )
    ap.add_argument(
        "--fluency_warmup_frac",
        type=float,
        default=1.0,
        help="For linear warmup: λ_eff = λ * clamp(step/(steps*frac),0..1).",
    )
    ap.add_argument("--utility_weight", type=float, default=0.0)
    ap.add_argument("--skill_suite_max_new_tokens", type=int, default=128)
    ap.add_argument(
        "--skill_suite_pack",
        choices=sorted(SKILL_SUITE_PACKS.keys()),
        default="v0",
        help="Which deterministic validator pack to use for utility/skill metrics (try: sota_v3).",
    )
    ap.add_argument("--agency_suite_enabled", action="store_true")
    ap.add_argument("--agency_suite_max_tasks", type=int, default=6)
    ap.add_argument("--agency_suite_min_steps", type=int, default=2)
    ap.add_argument("--agency_suite_base_max_program_len", type=int, default=6)
    ap.add_argument("--agency_suite_planner_max_depth", type=int, default=6)
    ap.add_argument("--agency_suite_planner_max_expansions", type=int, default=256)
    ap.add_argument("--enable_contracts", action="store_true")
    ap.add_argument("--concept_csv_mining_enabled", action="store_true")
    ap.add_argument("--concept_csv_mining_top_k", type=int, default=4)
    ap.add_argument("--concept_csv_mining_max_new_per_window", type=int, default=2)
    ap.add_argument(
        "--concept_csv_mining_max_ops",
        type=int,
        default=6,
        help="Max primitive-op length for concept_csv mining (increase for long-horizon plan packs).",
    )
    ap.add_argument("--concept_csv_composed_enabled", action="store_true")
    ap.add_argument("--concept_csv_composed_base_max_program_len", type=int, default=6)
    ap.add_argument("--concept_csv_composed_planner_max_depth", type=int, default=6)
    ap.add_argument("--concept_csv_composed_max_k", type=int, default=6)
    ap.add_argument("--concept_csv_composed_min_support", type=int, default=2)
    ap.add_argument("--concept_csv_composed_top_k", type=int, default=4)
    ap.add_argument("--concept_csv_composed_max_new_per_window", type=int, default=1)
    ap.add_argument("--concept_csv_budget", type=int, default=16)
    ap.add_argument("--concept_csv_overhead_bits", type=int, default=1024)
    ap.add_argument("--divergence_after_no_patch_windows", type=int, default=3)
    ap.add_argument("--divergence_after_no_growth_windows", type=int, default=2)
    ap.add_argument("--divergence_lambda_scale", type=float, default=0.1)
    ap.add_argument("--divergence_fluency_slack", type=float, default=0.05)
    # Decoder fluency projection layer (CPU-only; deterministic).
    ap.add_argument("--decoder_fluency_no_repeat_ngram", type=int, default=3)
    ap.add_argument(
        "--decoder_fluency_prompt_ngram_block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Seed decoder no-repeat n-grams with prompt n-grams (freeform turns only).",
    )
    ap.add_argument(
        "--decoder_fluency_min_new_tokens_before_eos_freeform",
        type=int,
        default=16,
        help="Freeform turns: minimum new tokens before EOS (0=disable; strict turns unaffected).",
    )
    ap.add_argument(
        "--decoder_fluency_block_token_regex",
        default=r"</?DOC>|_task_|_template_|zs_opt",
        help="Soft block regex for freeform turns to avoid dataset markup leakage (empty disables).",
    )
    ap.add_argument("--decoder_fluency_block_penalty", type=float, default=1e6)
    ap.add_argument("--force_runs", action="store_true", help="Overwrite existing run dirs")
    args = ap.parse_args()

    steps_list = _parse_steps_list(str(args.steps_list))
    horizon_steps = int(args.gain_horizon_steps)
    if horizon_steps <= 0:
        horizon_steps = max(int(x) for x in steps_list)
    os.makedirs(args.out_root, exist_ok=True)

    corpus_path = args.corpus_out or _default_corpus_path(str(args.source), str(args.split))
    meta = build_hf_corpus(
        source=str(args.source),
        out_path=str(corpus_path),
        split=str(args.split),
        dataset=(str(args.dataset) if args.dataset else None),
        name=(str(args.name) if args.name else None),
        streaming=bool(args.streaming),
        shuffle=bool(args.shuffle),
        seed=int(args.seed),
        shuffle_buffer=int(args.shuffle_buffer),
        target_bytes=int(args.target_bytes),
        max_examples=int(args.max_examples),
        max_chars_per_example=int(args.max_chars_per_example),
        force=bool(args.prepare_force),
    )
    print(f"[hf_corpus] ready: {meta.get('out_path')} ({meta.get('bytes_written')} bytes)")

    summary: List[Dict[str, Any]] = []
    for steps in steps_list:
        steps_i = int(steps)
        run_dir = os.path.join(args.out_root, f"steps_{steps_i:06d}")
        report_path = os.path.join(run_dir, "report.json")

        if os.path.exists(run_dir) and not bool(args.force_runs):
            last = _read_report_last(report_path)
            summary.append(_summ_row(run_dir, last))
            continue

        if os.path.exists(run_dir) and bool(args.force_runs):
            # Safe-ish overwrite: keep under artifacts/ (ignored) and delete is not necessary.
            pass

        if int(args.window) > 0:
            window = int(args.window)
        else:
            window = max(50, min(10_000, steps_i // 20))
        window = max(1, min(window, steps_i))

        cfg = TrainConfig(
            steps=int(steps_i),
            gain_horizon_steps=int(horizon_steps),
            nll_eval_windows=int(args.nll_eval_windows),
            holdout_frac=float(args.holdout_frac),
            holdout_eval_windows=int(args.holdout_eval_windows),
            holdout_eval_tokens=int(args.holdout_eval_tokens),
            survival_plateau_windows=int(args.survival_plateau_windows),
            survival_improve_tol=float(args.survival_improve_tol),
            survival_hard_fail_windows=int(args.survival_hard_fail_windows),
            survival_no_abstraction_windows=int(args.survival_no_abstraction_windows),
            survival_reuse_min=float(args.survival_reuse_min),
            survival_no_reuse_windows=int(args.survival_no_reuse_windows),
            survival_concept_no_add_windows=int(args.survival_concept_no_add_windows),
            survival_concept_reuse_stall_windows=int(args.survival_concept_reuse_stall_windows),
            survival_concept_reuse_tol=float(args.survival_concept_reuse_tol),
            survival_concept_hard_fail_windows=int(args.survival_concept_hard_fail_windows),
            survival_concept_composed_no_add_windows=int(args.survival_concept_composed_no_add_windows),
            survival_concept_composed_rate_stall_windows=int(args.survival_concept_composed_rate_stall_windows),
            survival_concept_composed_rate_tol=float(args.survival_concept_composed_rate_tol),
            survival_concept_composed_hard_fail_windows=int(args.survival_concept_composed_hard_fail_windows),
            survival_concept_deep_rate_stall_windows=int(args.survival_concept_deep_rate_stall_windows),
            survival_concept_deep_rate_tol=float(args.survival_concept_deep_rate_tol),
            survival_concept_deep_hard_fail_windows=int(args.survival_concept_deep_hard_fail_windows),
            survival_concept_very_deep_rate_stall_windows=int(args.survival_concept_very_deep_rate_stall_windows),
            survival_concept_very_deep_rate_tol=float(args.survival_concept_very_deep_rate_tol),
            survival_concept_very_deep_hard_fail_windows=int(args.survival_concept_very_deep_hard_fail_windows),
            repair_max_windows=int(args.repair_max_windows),
            repair_bottleneck_tol=float(args.repair_bottleneck_tol),
            seed=int(args.seed),
            window=int(window),
            propose_every=int(window),
            mode=str(args.mode),
            selection_mode=str(args.selection_mode),
            fluency_lambda=float(args.fluency_lambda),
            fluency_lambda_schedule=str(args.fluency_lambda_schedule),
            fluency_warmup_frac=float(args.fluency_warmup_frac),
            divergence_after_no_patch_windows=int(args.divergence_after_no_patch_windows),
            divergence_after_no_growth_windows=int(args.divergence_after_no_growth_windows),
            divergence_lambda_scale=float(args.divergence_lambda_scale),
            divergence_fluency_slack=float(args.divergence_fluency_slack),
            decoder_fluency_no_repeat_ngram=int(args.decoder_fluency_no_repeat_ngram),
            decoder_fluency_prompt_ngram_block=bool(args.decoder_fluency_prompt_ngram_block),
            decoder_fluency_min_new_tokens_before_eos_freeform=int(
                args.decoder_fluency_min_new_tokens_before_eos_freeform
            ),
            decoder_fluency_block_token_regex=str(args.decoder_fluency_block_token_regex or ""),
            decoder_fluency_block_penalty=float(args.decoder_fluency_block_penalty),
            utility_weight=float(args.utility_weight),
            skill_suite_max_new_tokens=int(args.skill_suite_max_new_tokens),
            skill_suite_pack=str(args.skill_suite_pack),
            agency_suite_enabled=bool(
                args.agency_suite_enabled or _is_sota_pack_at_least(str(args.skill_suite_pack), min_version=3)
            ),
            agency_suite_max_tasks=int(args.agency_suite_max_tasks),
            agency_suite_min_steps=int(args.agency_suite_min_steps),
            agency_suite_base_max_program_len=int(args.agency_suite_base_max_program_len),
            agency_suite_planner_max_depth=int(args.agency_suite_planner_max_depth),
            agency_suite_planner_max_expansions=int(args.agency_suite_planner_max_expansions),
            enable_contracts=bool(args.enable_contracts),
            concept_csv_mining_enabled=bool(args.concept_csv_mining_enabled),
            concept_csv_mining_top_k=int(args.concept_csv_mining_top_k),
            concept_csv_mining_max_new_per_window=int(args.concept_csv_mining_max_new_per_window),
            concept_csv_mining_max_ops=int(args.concept_csv_mining_max_ops),
            concept_csv_composed_enabled=bool(
                args.concept_csv_composed_enabled or _is_sota_pack_at_least(str(args.skill_suite_pack), min_version=3)
            ),
            concept_csv_composed_base_max_program_len=int(args.concept_csv_composed_base_max_program_len),
            concept_csv_composed_planner_max_depth=int(args.concept_csv_composed_planner_max_depth),
            concept_csv_composed_max_k=int(args.concept_csv_composed_max_k),
            concept_csv_composed_min_support=int(args.concept_csv_composed_min_support),
            concept_csv_composed_top_k=int(args.concept_csv_composed_top_k),
            concept_csv_composed_max_new_per_window=int(args.concept_csv_composed_max_new_per_window),
            concept_csv_budget=int(args.concept_csv_budget),
            concept_csv_overhead_bits=int(args.concept_csv_overhead_bits),
        )
        trainer = KAAbsoluteTrainer(data_path=str(corpus_path), out_dir=str(run_dir), config=cfg)
        trainer.train()

        last = _read_report_last(report_path)
        summary.append(_summ_row(run_dir, last))

    out_path = os.path.join(args.out_root, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n")
    print(out_path)


if __name__ == "__main__":
    main()
