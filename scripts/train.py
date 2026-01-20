#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.learn import KAAbsoluteTrainer, TrainConfig


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to plain-text dataset")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="Output dir (e.g., results/run1)")
    ap.add_argument("--window", type=int, default=10_000)
    ap.add_argument("--mode", choices=["demo", "pure"], default="demo")
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
        "--enable_contracts",
        action="store_true",
        help="Enable deterministic instruction contracts during generation (default OFF).",
    )
    args = ap.parse_args()

    ensure_sample_dataset(args.data)
    os.makedirs(args.out, exist_ok=True)

    cfg = TrainConfig(
        steps=args.steps,
        seed=args.seed,
        window=args.window,
        propose_every=args.window,
        mode=args.mode,
        utility_weight=float(args.utility_weight),
        skill_suite_max_new_tokens=int(args.skill_suite_max_new_tokens),
        enable_contracts=bool(args.enable_contracts),
    )
    trainer = KAAbsoluteTrainer(data_path=args.data, out_dir=args.out, config=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
