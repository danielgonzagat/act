#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.hf_corpus import build_hf_corpus
from atos_core.hf_corpus import resolve_hf_source
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
    ap.add_argument("--prepare_only", action="store_true")
    ap.add_argument(
        "--on_the_fly",
        action="store_true",
        help="Stream HF dataset on-the-fly during training (no local flat corpus).",
    )

    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument(
        "--max_hours",
        type=float,
        default=0.0,
        help="Optional wall-clock budget (hours). If >0, training stops gracefully after this time.",
    )
    ap.add_argument(
        "--gain_horizon_steps",
        type=int,
        default=0,
        help="If >0, used as the long-horizon token budget for patch MDL gain estimation.",
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
    ap.add_argument(
        "--resume_acts_path",
        default="",
        help="Optional acts.jsonl snapshot to resume from (preserves learned acts; adds missing seeds).",
    )
    ap.add_argument("--out", required=True, help="Output dir (use artifacts/... to stay ignored)")
    ap.add_argument("--window", type=int, default=10_000)
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
    ap.add_argument("--concept_csv_composed_overhead_scale", type=float, default=0.6)
    ap.add_argument("--concept_csv_deepwrap_overhead_scale", type=float, default=0.6)
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
    ap.add_argument(
        "--stream_buffer_tokens",
        type=int,
        default=200_000,
        help="Streaming mode: bounded in-memory token buffer used for eval/patch selection.",
    )
    ap.add_argument(
        "--stream_holdout_tokens",
        type=int,
        default=0,
        help="Streaming mode: static holdout token budget (0=auto from holdout eval config).",
    )
    args = ap.parse_args()

    corpus_path = args.corpus_out or _default_corpus_path(str(args.source), str(args.split))
    dataset_id = str(args.dataset) if args.dataset else ""
    if not dataset_id:
        resolved = resolve_hf_source(str(args.source))
        dataset_id = str(resolved.dataset_id)

    if not bool(args.on_the_fly):
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
        if bool(args.prepare_only):
            return

    os.makedirs(args.out, exist_ok=True)
    cfg = TrainConfig(
        steps=int(args.steps),
        max_seconds=(float(args.max_hours) * 3600.0 if float(args.max_hours) > 0.0 else 0.0),
        gain_horizon_steps=int(args.gain_horizon_steps),
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
        resume_acts_path=str(args.resume_acts_path or ""),
        window=int(args.window),
        propose_every=int(args.window),
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
        concept_csv_composed_overhead_scale=float(args.concept_csv_composed_overhead_scale),
        concept_csv_deepwrap_overhead_scale=float(args.concept_csv_deepwrap_overhead_scale),
        hf_dataset_id=str(dataset_id) if bool(args.on_the_fly) else "",
        hf_dataset_split=str(args.split),
        hf_dataset_name=str(args.name or ""),
        hf_streaming=bool(args.streaming),
        hf_shuffle=bool(args.shuffle),
        hf_shuffle_buffer=int(args.shuffle_buffer),
        hf_max_chars_per_example=int(args.max_chars_per_example),
        stream_buffer_tokens=int(args.stream_buffer_tokens),
        stream_holdout_tokens=int(args.stream_holdout_tokens),
    )
    data_path = str(corpus_path)
    if bool(args.on_the_fly):
        data_path = "__hf_stream__"
    trainer = KAAbsoluteTrainer(data_path=str(data_path), out_dir=str(args.out), config=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
