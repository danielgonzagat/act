#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.hf_corpus import build_hf_corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="HF dataset/model id or URL (owner/repo)")
    ap.add_argument("--out", required=True, help="Output corpus path (txt)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--dataset", default=None, help="Override resolved dataset id (owner/repo)")
    ap.add_argument("--config", dest="name", default=None, help="HF dataset config name")
    ap.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle_buffer", type=int, default=10_000)
    ap.add_argument("--target_bytes", type=int, default=20_000_000)
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--max_chars_per_example", type=int, default=8_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    meta = build_hf_corpus(
        source=str(args.source),
        out_path=str(args.out),
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
        force=bool(args.force),
    )
    print(meta["out_path"])


if __name__ == "__main__":
    main()

