#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.engine import Engine
from atos_core.metrics import is_space, tokenize_text
from atos_core.store import ActStore
from atos_core.suite import CHAT_DIALOGUES_20X3, build_chat_prompt, non_ws_tokens


def _topk(cands: Dict[str, Any], k: int) -> List[Tuple[str, float]]:
    items = sorted(
        ((tok, float(c.score)) for tok, c in cands.items()),
        key=lambda kv: (-kv[1], kv[0]),
    )
    return items[:k]


def _best_alt_score(
    cands: Dict[str, Any],
    *,
    exclude_lower: set[str],
    exclude_space: bool,
) -> float:
    best = float("-inf")
    for tok, c in cands.items():
        if tok == "<EOS>":
            continue
        if exclude_space and is_space(tok):
            continue
        tl = str(tok).lower()
        if tl in exclude_lower:
            continue
        best = max(best, float(c.score))
    return best


def _apply_rewrite_rules(
    engine: Engine,
    *,
    context: List[str],
    candidates: Dict[str, Any],
    penalties: Dict[str, Any],
    include_antitemplate: bool,
) -> Dict[str, Any]:
    c = candidates
    for rr in engine._rewrite_rules:
        if not include_antitemplate and rr.evidence.get("name") == "anti_template_v2":
            continue
        st = engine.vm.run(
            rr,
            context=context,
            tables=engine._tables,
            vocab=engine.vocab(),
            initial_candidates=c,
            penalties=penalties,
            mode="emit_only",
        )
        c = st.candidates
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run dir containing acts.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--dialogue_id", type=int, default=0)
    ap.add_argument("--turn", type=int, default=0, help="0-based user turn within the dialogue")
    ap.add_argument("--exclude_space_alt", action="store_true", help="Exclude whitespace for best-alt")
    ap.add_argument("--template_start_token", default="and", help="Lowercased token to calibrate as template-start")
    ap.add_argument(
        "--template_gap_stage",
        choices=["guardrails", "all"],
        default="guardrails",
        help="Which candidate stage to use for template-start gap computation",
    )
    ap.add_argument(
        "--ban_token",
        action="append",
        default=[],
        help="Override/extend banned tokens (repeatable); compared as lower()",
    )
    args = ap.parse_args()

    acts_path = os.path.join(args.run, "acts.jsonl")
    store = ActStore.load_jsonl(acts_path)
    engine = Engine(store, seed=args.seed)

    user_msg = CHAT_DIALOGUES_20X3[int(args.dialogue_id)][int(args.turn)]
    prompt = build_chat_prompt([{"user": user_msg, "system": ""}])

    prompt_tokens = tokenize_text(prompt)
    context: List[str] = ["<BOS>"] * (engine.config.max_order - 1)
    for t in prompt_tokens[-(engine.config.max_order - 1) :]:
        context.append(t)
    context = context[-(engine.config.max_order - 1) :]

    out_tokens: List[str] = list(prompt_tokens)
    gen_tokens: List[str] = []
    history_ngrams: List[Tuple[str, ...]] = []
    history_ngram_set = set()

    # Pull current anti-template evidence (for banned tokens) if present.
    banned_tokens_lower: set[str] = set()
    for act in store.active():
        if act.kind == "rewrite_rule" and act.evidence.get("name") == "anti_template_v2":
            banned_tokens_lower = {str(t).lower() for t in act.evidence.get("ban_tokens", [])}
            break
    if args.ban_token:
        banned_tokens_lower |= {str(t).lower() for t in args.ban_token if t is not None}

    template_start = str(args.template_start_token).lower()
    max_gap_ban = 0.0
    max_gap_ban_detail: Dict[str, Any] | None = None
    max_gap_template = 0.0
    max_gap_template_detail: Dict[str, Any] | None = None

    rows: List[Dict[str, Any]] = []
    for step in range(int(args.steps)):
        # penalties view (must match Engine.generate)
        filtered = [x for x in (out_tokens + gen_tokens) if x not in {"<BOS>"}]
        recent = filtered[-engine.config.repetition_recent :]
        penalties = {
            "recent_tokens": recent,
            "history_ngrams": set(history_ngram_set),
            "gen_tokens": list(gen_tokens),
        }

        pre = engine._emit_candidates(context=context, penalties=penalties)

        # stage: after guardrails (exclude anti_template)
        after_guard = _apply_rewrite_rules(
            engine,
            context=context,
            candidates=pre,
            penalties=penalties,
            include_antitemplate=False,
        )
        # stage: after all rewrite rules (current engine behavior)
        after_all = _apply_rewrite_rules(
            engine,
            context=context,
            candidates=pre,
            penalties=penalties,
            include_antitemplate=True,
        )

        # EOS guardrail (match Engine.generate)
        if step < int(engine.config.min_new_tokens_before_eos) and "<EOS>" in after_all:
            after_all["<EOS>"].score -= 1e6

        # select next token via selector act (match Engine.generate)
        nxt = None
        if engine._selector is not None:
            st = engine.vm.run(
                engine._selector,
                context=context,
                tables=engine._tables,
                vocab=engine.vocab(),
                initial_candidates=after_all,
                penalties=penalties,
                mode="select",
            )
            nxt = st.selected
        if nxt is None:
            ordered = sorted(
                after_all.values(), key=lambda c: (-c.score, c.token, c.source_act)
            )
            nxt = ordered[0].token if ordered else None

        # calibration gaps:
        # - banned-token gap is computed on the *guardrails-only* stage (no anti_template)
        # - template-start gap can be computed on guardrails-only or full stage
        best_alt = _best_alt_score(
            after_guard,
            exclude_lower=banned_tokens_lower,
            exclude_space=bool(args.exclude_space_alt),
        )

        if best_alt != float("-inf") and banned_tokens_lower:
            for tok, cand in after_guard.items():
                tl = str(tok).lower()
                if tl in banned_tokens_lower:
                    gap = float(cand.score) - best_alt
                    if gap > max_gap_ban:
                        max_gap_ban = gap
                        max_gap_ban_detail = {
                            "step": step,
                            "token": tok,
                            "token_score": float(cand.score),
                            "best_alt_score": float(best_alt),
                        }

        tpl_stage = after_guard if args.template_gap_stage == "guardrails" else after_all
        if template_start in tpl_stage:
            cand = tpl_stage[template_start]
            best_alt_tpl = _best_alt_score(
                tpl_stage,
                exclude_lower=banned_tokens_lower | {template_start},
                exclude_space=bool(args.exclude_space_alt),
            )
            if best_alt_tpl != float("-inf"):
                gap = float(cand.score) - float(best_alt_tpl)
                if gap > max_gap_template:
                    max_gap_template = gap
                    max_gap_template_detail = {
                        "step": step,
                        "token": template_start,
                        "token_score": float(cand.score),
                        "best_alt_score": float(best_alt_tpl),
                        "stage": str(args.template_gap_stage),
                    }

        # collect row
        gen_non_ws = [t.lower() for t in non_ws_tokens(gen_tokens)]
        rows.append(
            {
                "step": step,
                "gen_non_ws_pos": int(len(gen_non_ws)),
                "selected": nxt,
                "top_pre": _topk(pre, int(args.topk)),
                "top_after_guardrails": _topk(after_guard, int(args.topk)),
                "top_after_all_rr": _topk(after_all, int(args.topk)),
            }
        )

        if nxt is None or nxt == "<EOS>":
            break

        # advance state like Engine.generate (no observe)
        gen_tokens.append(nxt)
        filtered = [x for x in (out_tokens + gen_tokens) if x not in {"<BOS>"}]
        n = int(engine.config.cycle_ngram)
        if len(filtered) >= n:
            ng = tuple(filtered[-n:])
            history_ngrams.append(ng)
            history_ngram_set.add(ng)
            if len(history_ngrams) > int(engine.config.cycle_history):
                old = history_ngrams.pop(0)
                if old not in history_ngrams:
                    history_ngram_set.discard(old)

        context.append(nxt)
        context = context[-(engine.config.max_order - 1) :]

    w_ban_token = float(max(0.0, max_gap_ban) + 2.0)
    w_ban_prefix_seq = float(max(0.0, max_gap_template) + 2.0)

    out = {
        "run": os.path.normpath(args.run),
        "seed": int(args.seed),
        "probe": {
            "dialogue_id": int(args.dialogue_id),
            "turn": int(args.turn),
            "user_msg": user_msg,
            "prompt": prompt,
        },
        "params": {
            "steps": int(args.steps),
            "topk": int(args.topk),
            "exclude_space_alt": bool(args.exclude_space_alt),
            "template_start_token": template_start,
            "template_gap_stage": str(args.template_gap_stage),
            "banned_tokens_lower": sorted(banned_tokens_lower),
        },
        "calibration": {
            "max_gap_ban": float(max_gap_ban),
            "max_gap_ban_detail": max_gap_ban_detail,
            "max_gap_template_start": float(max_gap_template),
            "max_gap_template_detail": max_gap_template_detail,
            "suggest": {
                "w_ban_token": float(w_ban_token),
                "w_ban_prefix_seq": float(w_ban_prefix_seq),
            },
        },
        "trace": rows,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
