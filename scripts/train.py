#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.learn import KAAbsoluteTrainer, TrainConfig
from atos_core.suite import SKILL_SUITE_PACKS


def ensure_sample_dataset(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    base = textwrap.dedent(
        """
        Oi! Este é um texto de amostra para treinar um sistema cognitivo discreto.
        A ideia é simples: contar padrões e escolher próximas palavras sem usar redes neurais.
        Regras: nada de gradiente, nada de pesos treináveis; apenas atos editáveis.

        Hello! This is a mixed Portuguese/English sample corpus.
        We want predictable repetition, varied structure, and enough symbols to learn n-grams.
        The system should learn to continue phrases like "Oi, quem é você?" and "What is this?".
        ---
        """
    ).strip() + "\n"
    target_bytes = 1_200_000
    blocks = []
    for i in range(200):
        blocks.append(base)
        blocks.append(f"Variação {i}: Olá mundo, passo {i}, seed 0, determinismo.\n")
        blocks.append(f"Variation {i}: hello world, step {i}, seed 0, determinism.\n")
        blocks.append("Pergunta: qual é a capital de Portugal? Resposta: Lisboa.\n")
        blocks.append("Question: what is the capital of France? Answer: Paris.\n")
        blocks.append("Resumo: atos -> candidatos -> seleção -> atualização.\n")
        blocks.append("Summary: acts -> candidates -> selection -> update.\n")
    text = "".join(blocks)
    buf = (text * ((target_bytes // len(text)) + 2))[:target_bytes]
    with open(path, "w", encoding="utf-8") as f:
        f.write(buf)


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
    ap.add_argument("--data", required=True, help="Path to plain-text dataset")
    ap.add_argument("--steps", type=int, default=200_000)
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
    ap.add_argument("--out", required=True, help="Output dir (e.g., results/run1)")
    ap.add_argument("--window", type=int, default=10_000)
    ap.add_argument("--mode", choices=["demo", "pure"], default="demo")
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
    ap.add_argument(
        "--utility_weight",
        type=float,
        default=0.0,
        help="Weight for utility penalty in patch score (default 0.0 = shadow/logging only).",
    )
    ap.add_argument(
        "--skill_suite_max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens for the utility suite during training (eval scripts remain canonical).",
    )
    ap.add_argument(
        "--skill_suite_pack",
        choices=sorted(SKILL_SUITE_PACKS.keys()),
        default="v0",
        help="Which deterministic validator pack to use for utility/skill metrics (try: sota_v4).",
    )
    ap.add_argument(
        "--agency_suite_enabled",
        action="store_true",
        help="Enable deterministic agency suite (planner+executor) as a utility bottleneck dimension.",
    )
    ap.add_argument("--agency_suite_max_tasks", type=int, default=6)
    ap.add_argument("--agency_suite_min_steps", type=int, default=2)
    ap.add_argument("--agency_suite_base_max_program_len", type=int, default=6)
    ap.add_argument("--agency_suite_planner_max_depth", type=int, default=6)
    ap.add_argument("--agency_suite_planner_max_expansions", type=int, default=256)
    ap.add_argument(
        "--enable_contracts",
        action="store_true",
        help="Enable deterministic instruction contracts during generation (default OFF).",
    )
    ap.add_argument(
        "--concept_csv_mining_enabled",
        action="store_true",
        help="Enable deterministic concept_csv induction (INDUCE_CONCEPT via CSV mining).",
    )
    ap.add_argument("--concept_csv_mining_top_k", type=int, default=4)
    ap.add_argument("--concept_csv_mining_max_new_per_window", type=int, default=2)
    ap.add_argument(
        "--concept_csv_mining_max_ops",
        type=int,
        default=6,
        help="Max primitive-op length for concept_csv mining (increase for long-horizon plan packs).",
    )
    ap.add_argument(
        "--concept_csv_composed_enabled",
        action="store_true",
        help="Enable composed concept induction (v74): promote CSV_CALL concepts from multi-step traces.",
    )
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
    args = ap.parse_args()

    ensure_sample_dataset(args.data)
    os.makedirs(args.out, exist_ok=True)

    cfg = TrainConfig(
        steps=args.steps,
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
        seed=args.seed,
        window=args.window,
        propose_every=args.window,
        mode=args.mode,
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
    )
    trainer = KAAbsoluteTrainer(data_path=args.data, out_dir=args.out, config=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
