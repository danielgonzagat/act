#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.engine import Engine
from atos_core.store import ActStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run dir (e.g., results/run1)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acts_path = os.path.join(args.run, "acts.jsonl")
    store = ActStore.load_jsonl(acts_path)
    engine = Engine(store, seed=args.seed)
    out = engine.generate(prompt=args.prompt, max_new_tokens=args.max_new_tokens, mode=args.mode)
    print(out["text"])


if __name__ == "__main__":
    main()
