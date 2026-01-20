#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import estimate_act_cost_bits
from atos_core.engine import Engine
from atos_core.ledger import Ledger
from atos_core.metrics import tokenize_text
from atos_core.store import ActStore
from atos_core.suite import CHAT_DIALOGUES_20X3, run_chat_suite


def eval_nll_bits(engine: Engine, tokens: List[str], *, length: int = 20_000) -> float:
    ctx = ["<BOS>"] * (engine.config.max_order - 1)
    nll = 0.0
    for i in range(length):
        tok = tokens[i % len(tokens)]
        lp = engine.logprob_next(context=ctx, token=tok)
        nll += -lp / 0.6931471805599453  # ln(2)
        ctx.append(tok)
        ctx = ctx[-(engine.config.max_order - 1) :]
    return nll / length


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data/sample_text.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    acts_path = os.path.join(args.run, "acts.jsonl")
    report_path = os.path.join(args.run, "report.json")
    transcripts_path = os.path.join(args.run, "transcripts.jsonl")
    ledger_path = os.path.join(args.run, "ledger.jsonl")

    store = ActStore.load_jsonl(acts_path)
    engine = Engine(store, seed=args.seed)

    with open(args.data, "r", encoding="utf-8") as f:
        tokens = tokenize_text(f.read())
    nll_bits_mean = eval_nll_bits(engine, tokens, length=min(20_000, max(1, len(tokens))))

    # Summarize report if present.
    report_summary: Dict[str, Any] = {}
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        if rep:
            report_summary = {"first": rep[0], "last": rep[-1], "num_windows": len(rep)}

    transcripts, fluency = run_chat_suite(
        engine,
        dialogues=CHAT_DIALOGUES_20X3,
        max_new_tokens=args.max_new_tokens,
        prefix_k=8,
        template_ngram_n=6,
        template_prefix_window=32,
    )
    with open(transcripts_path, "w", encoding="utf-8") as f:
        for rec in transcripts:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    eval_out = {
        "nll_bits_mean_eval": nll_bits_mean,
        "model_cost_bits": sum(estimate_act_cost_bits(a) for a in store.active()),
        "verify_chain": Ledger(ledger_path).verify_chain() if os.path.exists(ledger_path) else False,
        "fluency": fluency,
        "report_summary": report_summary,
    }
    print(json.dumps(eval_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
