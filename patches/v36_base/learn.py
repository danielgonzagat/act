from __future__ import annotations

import json
import math
import os
import time
import copy
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from .act import (
    Act,
    Instruction,
    Patch,
    canonical_json_dumps,
    deterministic_iso,
    estimate_act_cost_bits,
    sha256_hex,
)
from .engine import Engine, EngineConfig, ctx_key
from .ledger import Ledger
from .metrics import (
    rss_bytes_best_effort,
    safe_log2,
    loop_rate,
    repeat_ngram_rate,
    tokenize_text,
)
from .suite import (
    CHAT_DIALOGUES_20X3,
    non_ws_tokens,
    prefix_k_signature,
    reply_signature,
    run_chat_suite,
    run_skill_suite,
    user_signature,
)
from .store import ActStore


def det_act_id(*, step: int, name: str, idx: int = 0) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in name)
    return f"act_s{int(step):06d}_{safe}_i{int(idx)}"


def _table_stats(table: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    ctx = int(len(table))
    edges = 0
    for nxt in table.values():
        edges += int(len(nxt))
    return {"n_ctx": ctx, "n_edges": int(edges)}


def _union_sum_tables(
    a: Dict[str, Dict[str, int]], b: Dict[str, Dict[str, int]]
) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for k, nxt in a.items():
        out[k] = {t: int(c) for t, c in nxt.items()}
    for k, nxt in b.items():
        dst = out.get(k)
        if dst is None:
            out[k] = {t: int(c) for t, c in nxt.items()}
            continue
        for t, c in nxt.items():
            dst[t] = int(dst.get(t, 0)) + int(c)
    return out


def _apply_budget_topk(
    table: Dict[str, Dict[str, int]],
    *,
    max_contexts: int,
    max_next_per_ctx: int,
) -> Dict[str, Dict[str, int]]:
    if max_next_per_ctx > 0:
        for k, nxt in list(table.items()):
            items = [(t, int(c)) for t, c in nxt.items()]
            items.sort(key=lambda kv: (-kv[1], kv[0]))
            items = items[: int(max_next_per_ctx)]
            table[k] = {t: int(c) for t, c in items}

    if max_contexts > 0 and len(table) > int(max_contexts):
        scored: List[Tuple[int, str]] = []
        for k, nxt in table.items():
            total = int(sum(int(c) for c in nxt.values()))
            scored.append((total, k))
        scored.sort(key=lambda kv: (-kv[0], kv[1]))
        keep_keys = [k for _, k in scored[: int(max_contexts)]]
        table = {k: table[k] for k in keep_keys}

    return table


def _reset_fifo_state_for_table(table: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    # Deterministic initialization for FIFO budget metadata.
    # - ctx_fifo: contexts ordered by (total_count asc, key lex)
    # - next_fifo: tokens per context ordered by (count asc, token lex)
    ctx_items: List[Tuple[int, str]] = []
    next_fifo: Dict[str, List[str]] = {}
    for k, nxt in table.items():
        total = int(sum(int(c) for c in nxt.values()))
        ctx_items.append((total, k))
        toks = [(int(c), str(t)) for t, c in nxt.items()]
        toks.sort(key=lambda kv: (kv[0], kv[1]))
        next_fifo[k] = [t for _, t in toks]
    ctx_items.sort(key=lambda kv: (kv[0], kv[1]))
    ctx_fifo = [k for _, k in ctx_items]
    return {"ctx_fifo": ctx_fifo, "next_fifo": next_fifo}


def _merge_fifo_state_for_table(
    table: Dict[str, Dict[str, int]],
    *,
    keep_ctx_fifo: Any,
    keep_next_fifo: Any,
    other_ctx_fifo: Any,
    other_next_fifo: Any,
) -> Dict[str, Any]:
    want_ctx = set(table.keys())

    ctx_fifo: List[str] = []
    seen_ctx: set[str] = set()

    def extend_ctx(seq: Any) -> None:
        if not isinstance(seq, list):
            return
        for k in seq:
            if not isinstance(k, str):
                continue
            if k not in want_ctx:
                continue
            if k in seen_ctx:
                continue
            seen_ctx.add(k)
            ctx_fifo.append(k)

    extend_ctx(keep_ctx_fifo)
    extend_ctx(other_ctx_fifo)

    # Ensure all contexts in table are present (deterministic fallback).
    missing_ctx = sorted(want_ctx - seen_ctx)
    ctx_fifo.extend(missing_ctx)

    next_fifo: Dict[str, List[str]] = {}

    def load_next(src: Any) -> Dict[str, Any]:
        return src if isinstance(src, dict) else {}

    a = load_next(keep_next_fifo)
    b = load_next(other_next_fifo)

    # Start with keep's ordering.
    for k, seq in a.items():
        if not isinstance(k, str) or k not in want_ctx:
            continue
        if not isinstance(seq, list):
            continue
        next_fifo[k] = [t for t in seq if isinstance(t, str)]

    # Append other's ordering (stable union).
    for k, seq in b.items():
        if not isinstance(k, str) or k not in want_ctx:
            continue
        if not isinstance(seq, list):
            continue
        base = next_fifo.get(k)
        if base is None:
            next_fifo[k] = [t for t in seq if isinstance(t, str)]
            continue
        have = set(base)
        for t in seq:
            if not isinstance(t, str) or t in have:
                continue
            have.add(t)
            base.append(t)

    # Filter to tokens present in table; ensure coverage deterministically.
    for k, nxt in table.items():
        fifo = next_fifo.get(k, [])
        present = set(str(t) for t in nxt.keys())
        fifo = [t for t in fifo if t in present]
        missing = sorted(present - set(fifo))
        fifo.extend(missing)
        next_fifo[k] = fifo

    return {"ctx_fifo": ctx_fifo, "next_fifo": next_fifo}


def _make_unigram_act(*, act_id: str, created_at: str) -> Act:
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="predictor",
        match={"type": "ngram", "n": 1},
        program=[
            Instruction("MATCH_NGRAM", {"n": 1}),
            Instruction("EMIT_CANDIDATES_TOPK", {"table_id": "tbl_uni", "k": 64, "n": 1}),
            Instruction("SCORE_CANDIDATES", {"method": "logprob", "alpha": 0.5}),
        ],
        evidence={
            "name": "unigram",
            "n": 1,
            "table_id": "tbl_uni",
            "table": {"": {}},
            "allow_new_contexts": False,
            "allow_new_tokens": True,
            "max_contexts": 1,
            "max_next_per_ctx": 4096,
            "evict_policy": "fifo",
        },
        cost={"overhead_bits": 1024, "ctx_cost_bits": 0, "edge_cost_bits": 4},
        deps=[],
    )


def _make_rewrite_rule_act(*, act_id: str, created_at: str) -> Act:
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="rewrite_rule",
        match={"type": "always"},
        program=[
            Instruction("APPLY_PENALTY", {"kind": "repetition", "strength": 1.5}),
            Instruction("APPLY_PENALTY", {"kind": "ngram_cycle", "n": 3, "strength": 3.0}),
        ],
        evidence={"name": "fluency_guardrails_v1"},
        cost={"overhead_bits": 512},
        deps=[],
    )


def _make_mode_selector_act(*, act_id: str, created_at: str, mode: str) -> Act:
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="mode_selector",
        match={"type": "always"},
        program=[Instruction("SET_MODE", {"mode": str(mode)})],
        evidence={"name": "mode_selector_v1", "mode": str(mode)},
        cost={"overhead_bits": 128},
        deps=[],
    )


def _make_mode_policy_act(*, act_id: str, created_at: str) -> Act:
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="mode_policy",
        match={"type": "always"},
        program=[],
        evidence={
            "name": "mode_policy_v1",
            # user prompt signature width (non-ws tokens, lower)
            "k": 2,
            # min evidence before a mode is eligible for selection
            "min_trials": 1,
            # user_sig -> mode -> {trials, pen_sum}
            "table": {},
        },
        cost={"overhead_bits": 512},
        deps=[],
    )


def _make_anti_template_rewrite_rule_act(*, act_id: str, created_at: str) -> Act:
    # v0.2.1: penalize early reply templates (tags/boilerplate) deterministically.
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="rewrite_rule",
        match={"type": "always"},
        program=[
            Instruction("APPLY_PENALTY", {"kind": "anti_template_v2"}),
        ],
        evidence={
            "name": "anti_template_v2",
            "k": 8,
            # Back-compat defaults (used when mode config missing).
            "prefix_window": 96,
            "w_ban_token": 2.0,
            "ban_tokens": ["User:", "System:", "Pessoa:", "Sistema:"],
            "w_ban_prefix_seq": 2.0,
            "ban_prefix_seqs": [],
            # v0.2.6: suffix n-gram blocking anywhere within an early window.
            "ngram_prefix_window": 32,
            "w_ban_ngram_seq": 2.5,
            "ban_ngram_seqs": [],
            # v0.2.7: KA-TEMPLATE (deterministically refresh ban_ngram_seqs per window from suite transcripts).
            "ka_template": {
                "enabled": True,
                "n": 6,
                "prefix_window": 32,
                "top_k": 32,
                "min_count": 2,
            },
            # v0.2.3: condition anti-template by internal MODE.
            "modes": {
                "default": {
                    "prefix_window": 96,
                    "w_ban_token": 2.0,
                    "ban_tokens": ["User:", "System:", "Pessoa:", "Sistema:"],
                    "w_ban_prefix_seq": 2.0,
                    "ban_prefix_seqs": [],
                },
                "list": {
                    # Allow list markers; keep only chat-tag suppression.
                    "prefix_window": 96,
                    "w_ban_token": 2.0,
                    "ban_tokens": ["User:", "System:", "Pessoa:", "Sistema:"],
                    "w_ban_prefix_seq": 2.1096372878054203,
                    "ban_prefix_seqs": [
                        ["->"],
                        ["includes", "lists", "and", "\"what", "is"],
                        ["breaks."],
                        ["and", "\"what", "is", "a", "collection"],
                        ["abc", "abd", "abc"],
                        ["-", "it", "also", "includes"],
                        ["the", "next", "token"],
                    ],
                },
                "definition": {
                    # Push away from dataset-format continuation templates.
                    "prefix_window": 96,
                    "w_ban_token": 3.0,
                    "ban_tokens": ["User:", "System:", "Pessoa:", "Sistema:", "abf."],
                    "w_ban_prefix_seq": 3.0,
                    "ban_prefix_seqs": [
                        ["abc", "abd", "abc"],
                        ["also", "includes", "lists"],
                        ["-", "it", "also", "includes"],
                        ["includes", "lists", "and", "\"what", "is"],
                        ["breaks."],
                        ["and", "apply", "repetition", "penalties"],
                        ["and", "\"what", "is"],
                        ["the", "next", "token"],
                        ["->"],
                        ["0,", "determinism."],
                    ],
                },
                "explanation": {
                    # Similar to definition, slightly weaker.
                    "prefix_window": 96,
                    "w_ban_token": 2.5,
                    "ban_tokens": ["User:", "System:", "Pessoa:", "Sistema:", "abf."],
                    "w_ban_prefix_seq": 2.5,
                    "ban_prefix_seqs": [
                        ["abc", "abd", "abc"],
                        ["-", "it", "also", "includes"],
                        ["includes", "lists", "and", "\"what", "is"],
                        ["breaks."],
                        ["brown", "fox", "jumps", "over"],
                        ["and", "apply", "repetition", "penalties"],
                        ["and", "\"what", "is"],
                        ["the", "next", "token"],
                        ["->"],
                        ["0,", "determinism."],
                    ],
                },
            },
        },
        cost={"overhead_bits": 512},
        deps=[],
    )


def _make_fact_memory_act(*, act_id: str, created_at: str) -> Act:
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="memory_facts",
        match={"type": "always"},
        program=[],
        evidence={
            "name": "fact_memory_v0",
            "enabled": True,
            "shadow_mode": True,
            # Budget / governance.
            "max_facts": 256,
            "max_evidence_turns": 8,
            # Extraction constraints (keep memory clean in v0).
            "max_answer_tokens": 3,
            "require_capital_question": True,
            # Scaffold filters (substring, lower).
            "ban_keywords": [
                "regras:",
                "rules:",
                "nada de",
                "gradiente",
                "sgd",
                "adam",
                "embedding",
                "rede neural",
                "neural",
                "neurais",
                "summary:",
                "resumo:",
                "acts ->",
                "atos ->",
                "candidates ->",
                "-> atualização",
                "the next token",
                "brown fox",
                "abf.",
            ],
            # fact_key -> {value, confidence, evidence_turns, last_seen_step, count}
            "table": {},
        },
        cost={"overhead_bits": 512},
        deps=[],
    )


def _make_macro_library_act(*, act_id: str, created_at: str) -> Act:
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="compressor",
        match={"type": "always"},
        program=[],
        evidence={
            "name": "macro_library_v0",
            "enabled": True,
            "shadow_mode": True,
            # Governance / budgets.
            "budget": 128,
            "max_new_per_window": 8,
            # Mining parameters (act-sequence n-grams from traces).
            "n_min": 3,
            "n_max": 6,
            "min_count": 4,
            "max_tokens_per_reply": 96,
            # Mining parameters (predictor co-activation sets).
            "set_min_size": 2,
            "set_min_count": 4,
            "transition_top_k": 32,
            # macro_id -> {type, pattern, deps, n|k, count, last_seen_step, gain_*}
            "macros": {},
        },
        cost={"overhead_bits": 512},
        deps=[],
    )


def _make_macro_router_act(*, act_id: str, created_at: str) -> Act:
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="compressor",
        match={"type": "always"},
        program=[],
        evidence={
            "name": "macro_router_v0",
            "enabled": True,
            "shadow_mode": True,
            # Budget / governance.
            "max_contexts": 8192,
            "top_k": 2,
            # ctx_sig -> {"predictors": [...], "counts": {...}, "total": int}
            "table": {},
        },
        cost={"overhead_bits": 512},
        deps=[],
    )


def _make_instruction_contract_act(*, act_id: str, created_at: str) -> Act:
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="candidate_source",
        match={"type": "always"},
        program=[],
        evidence={
            "name": "instruction_contract_v0",
            "enabled": True,
            "shadow_mode": True,
        },
        cost={"overhead_bits": 512},
        deps=[],
    )


def _make_selector_act(*, act_id: str, created_at: str) -> Act:
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="selector",
        match={"type": "always"},
        program=[Instruction("SELECT_NEXT", {"mode": "greedy"})],
        evidence={"name": "selector_greedy_v1"},
        cost={"overhead_bits": 256},
        deps=[],
    )


def build_ngram_table(
    tokens: Sequence[str],
    *,
    n: int,
    prev_is_space: Optional[bool] = None,
    max_next: int = 64,
    min_ctx_total: int = 2,
    min_next_count: int = 1,
) -> Dict[str, Dict[str, int]]:
    if n <= 0:
        return {}
    ctx_len = max(0, n - 1)
    table: Dict[str, Dict[str, int]] = {}
    context: List[str] = ["<BOS>"] * ctx_len
    for tok in tokens:
        if ctx_len:
            prev = context[-1]
            if prev_is_space is not None:
                want = bool(prev_is_space)
                if prev.isspace() != want:
                    context = (context + [tok])[-ctx_len:]
                    continue
            key = ctx_key(context)
        else:
            key = ""
        nxt = table.get(key)
        if nxt is None:
            nxt = {}
            table[key] = nxt
        nxt[tok] = int(nxt.get(tok, 0)) + 1
        if ctx_len:
            context = (context + [tok])[-ctx_len:]

    pruned: Dict[str, Dict[str, int]] = {}
    for key, nxt in table.items():
        total = sum(nxt.values())
        if total < min_ctx_total:
            continue
        items = [(t, c) for t, c in nxt.items() if c >= min_next_count]
        if not items:
            continue
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        items = items[:max_next]
        pruned[key] = {t: int(c) for t, c in items}
    return pruned


def _make_bigram_space_act(
    *,
    act_id: str,
    table: Dict[str, Dict[str, int]],
    created_at: str,
    allow_new_contexts: bool,
    allow_new_tokens: bool,
    max_contexts: int,
    max_next_per_ctx: int,
    evict_policy: str = "fifo",
) -> Act:
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="predictor",
        match={"type": "ngram", "n": 2, "prev_is_space": True},
        program=[
            Instruction("MATCH_NGRAM", {"n": 2}),
            Instruction("EMIT_CANDIDATES_TOPK", {"table_id": "tbl_bi_space", "k": 64, "n": 2}),
            Instruction("SCORE_CANDIDATES", {"method": "logprob", "alpha": 0.5}),
        ],
        evidence={
            "name": "bigram_space",
            "n": 2,
            "table_id": "tbl_bi_space",
            "table": table,
            "allow_new_contexts": bool(allow_new_contexts),
            "allow_new_tokens": bool(allow_new_tokens),
            "max_contexts": int(max_contexts),
            "max_next_per_ctx": int(max_next_per_ctx),
            "evict_policy": str(evict_policy),
        },
        cost={"overhead_bits": 1024, "ctx_cost_bits": 16, "edge_cost_bits": 8},
        deps=[],
    )


def _make_bigram_nonspace_act(
    *,
    act_id: str,
    table: Dict[str, Dict[str, int]],
    created_at: str,
    allow_new_contexts: bool,
    allow_new_tokens: bool,
    max_contexts: int,
    max_next_per_ctx: int,
    evict_policy: str = "fifo",
) -> Act:
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="predictor",
        match={"type": "ngram", "n": 2, "prev_is_space": False},
        program=[
            Instruction("MATCH_NGRAM", {"n": 2}),
            Instruction("EMIT_CANDIDATES_TOPK", {"table_id": "tbl_bi_nonspace", "k": 64, "n": 2}),
            Instruction("SCORE_CANDIDATES", {"method": "logprob", "alpha": 0.5}),
        ],
        evidence={
            "name": "bigram_nonspace",
            "n": 2,
            "table_id": "tbl_bi_nonspace",
            "table": table,
            "allow_new_contexts": bool(allow_new_contexts),
            "allow_new_tokens": bool(allow_new_tokens),
            "max_contexts": int(max_contexts),
            "max_next_per_ctx": int(max_next_per_ctx),
            "evict_policy": str(evict_policy),
        },
        cost={"overhead_bits": 1024, "ctx_cost_bits": 16, "edge_cost_bits": 8},
        deps=[],
    )


def _make_trigram_act(
    *,
    act_id: str,
    table: Dict[str, Dict[str, int]],
    created_at: str,
    allow_new_contexts: bool,
    allow_new_tokens: bool,
    max_contexts: int,
    max_next_per_ctx: int,
    evict_policy: str = "fifo",
) -> Act:
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="predictor",
        match={"type": "ngram", "n": 3},
        program=[
            Instruction("MATCH_NGRAM", {"n": 3}),
            Instruction("EMIT_CANDIDATES_TOPK", {"table_id": "tbl_tri", "k": 64, "n": 3}),
            Instruction("SCORE_CANDIDATES", {"method": "logprob", "alpha": 0.5}),
        ],
        evidence={
            "name": "trigram",
            "n": 3,
            "table_id": "tbl_tri",
            "table": table,
            "allow_new_contexts": bool(allow_new_contexts),
            "allow_new_tokens": bool(allow_new_tokens),
            "max_contexts": int(max_contexts),
            "max_next_per_ctx": int(max_next_per_ctx),
            "evict_policy": str(evict_policy),
        },
        cost={"overhead_bits": 1024, "ctx_cost_bits": 20, "edge_cost_bits": 10},
        deps=[],
    )


def _make_fourgram_act(
    *,
    act_id: str,
    table: Dict[str, Dict[str, int]],
    created_at: str,
    allow_new_contexts: bool,
    allow_new_tokens: bool,
    max_contexts: int,
    max_next_per_ctx: int,
    evict_policy: str = "fifo",
) -> Act:
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="predictor",
        match={"type": "ngram", "n": 4},
        program=[
            Instruction("MATCH_NGRAM", {"n": 4}),
            Instruction("EMIT_CANDIDATES_TOPK", {"table_id": "tbl_4", "k": 64, "n": 4}),
            Instruction("SCORE_CANDIDATES", {"method": "logprob", "alpha": 0.5}),
        ],
        evidence={
            "name": "fourgram",
            "n": 4,
            "table_id": "tbl_4",
            "table": table,
            "allow_new_contexts": bool(allow_new_contexts),
            "allow_new_tokens": bool(allow_new_tokens),
            "max_contexts": int(max_contexts),
            "max_next_per_ctx": int(max_next_per_ctx),
            "evict_policy": str(evict_policy),
        },
        cost={"overhead_bits": 1024, "ctx_cost_bits": 22, "edge_cost_bits": 12},
        deps=[],
    )


def _make_bigram_merged_act(
    *,
    act_id: str,
    table: Dict[str, Dict[str, int]],
    deps: List[str],
    created_at: str,
    allow_new_contexts: bool,
    allow_new_tokens: bool,
    max_contexts: int,
    max_next_per_ctx: int,
    evict_policy: str = "fifo",
) -> Act:
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="predictor",
        match={"type": "ngram", "n": 2},
        program=[
            Instruction("MATCH_NGRAM", {"n": 2}),
            Instruction("EMIT_CANDIDATES_TOPK", {"table_id": "tbl_bi", "k": 64, "n": 2}),
            Instruction("SCORE_CANDIDATES", {"method": "logprob", "alpha": 0.5}),
        ],
        evidence={
            "name": "bigram_merged",
            "n": 2,
            "table_id": "tbl_bi",
            "table": table,
            "allow_new_contexts": bool(allow_new_contexts),
            "allow_new_tokens": bool(allow_new_tokens),
            "max_contexts": int(max_contexts),
            "max_next_per_ctx": int(max_next_per_ctx),
            "evict_policy": str(evict_policy),
        },
        cost={"overhead_bits": 1024, "ctx_cost_bits": 16, "edge_cost_bits": 8},
        deps=list(deps),
    )


@dataclass
class TrainConfig:
    steps: int = 200_000
    seed: int = 0
    window: int = 10_000
    propose_every: int = 10_000
    val_tokens: int = 4000
    mode: Literal["demo", "pure"] = "demo"
    fluency_lambda: float = 20000.0
    fluency_prompt: str = "Oi, quem é você?\n"
    fluency_prompts: Tuple[str, str, str] = (
        "Oi, quem é você?\n",
        "Hello, who are you?\n",
        "What is MDL?\n",
    )
    fluency_gen_tokens: int = 200
    suite_prefix_k: int = 8
    suite_template_ngram_n: int = 6
    suite_template_prefix_window: int = 32
    # Utility suite weight in patch score (shadow default: 0.0). Set >0 to optimize pass_rate.
    utility_weight: float = 0.0
    # Utility suite token budget during training (evaluation scripts use their own max_new_tokens).
    skill_suite_max_new_tokens: int = 128
    # Enable deterministic instruction contracts during generation (default OFF).
    enable_contracts: bool = False


class KAAbsoluteTrainer:
    def __init__(self, *, data_path: str, out_dir: str, config: TrainConfig):
        self.data_path = data_path
        self.out_dir = out_dir
        self.config = config

        os.makedirs(self.out_dir, exist_ok=True)
        self.acts_path = os.path.join(self.out_dir, "acts.jsonl")
        self.ledger_path = os.path.join(self.out_dir, "ledger.jsonl")
        self.report_path = os.path.join(self.out_dir, "report.json")
        self.snapshots_dir = os.path.join(self.out_dir, "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)

        self.store = ActStore()
        self.ledger = Ledger(self.ledger_path)

        self._adds = 0
        self._merges = 0
        self._prunes = 0

    def _init_acts(self) -> None:
        self.store.add(
            _make_unigram_act(
                act_id=det_act_id(step=0, name="unigram", idx=0),
                created_at=deterministic_iso(step=0, offset_us=0),
            )
        )
        self.store.add(
            _make_mode_policy_act(
                act_id=det_act_id(step=0, name="mode_policy_v1", idx=0),
                created_at=deterministic_iso(step=0, offset_us=1),
            )
        )
        self.store.add(
            _make_mode_selector_act(
                act_id=det_act_id(step=0, name="mode_definition_v1", idx=0),
                created_at=deterministic_iso(step=0, offset_us=2),
                mode="definition",
            )
        )
        self.store.add(
            _make_mode_selector_act(
                act_id=det_act_id(step=0, name="mode_explanation_v1", idx=0),
                created_at=deterministic_iso(step=0, offset_us=3),
                mode="explanation",
            )
        )
        self.store.add(
            _make_mode_selector_act(
                act_id=det_act_id(step=0, name="mode_list_v1", idx=0),
                created_at=deterministic_iso(step=0, offset_us=4),
                mode="list",
            )
        )
        self.store.add(
            _make_rewrite_rule_act(
                act_id=det_act_id(step=0, name="fluency_guardrails_v1", idx=0),
                created_at=deterministic_iso(step=0, offset_us=5),
            )
        )
        self.store.add(
            _make_anti_template_rewrite_rule_act(
                act_id=det_act_id(step=0, name="anti_template_v2", idx=0),
                created_at=deterministic_iso(step=0, offset_us=6),
            )
        )
        self.store.add(
            _make_selector_act(
                act_id=det_act_id(step=0, name="selector_greedy_v1", idx=0),
                created_at=deterministic_iso(step=0, offset_us=7),
            )
        )
        self.store.add(
            _make_fact_memory_act(
                act_id=det_act_id(step=0, name="fact_memory_v0", idx=0),
                created_at=deterministic_iso(step=0, offset_us=8),
            )
        )
        self.store.add(
            _make_macro_library_act(
                act_id=det_act_id(step=0, name="macro_library_v0", idx=0),
                created_at=deterministic_iso(step=0, offset_us=9),
            )
        )
        self.store.add(
            _make_macro_router_act(
                act_id=det_act_id(step=0, name="macro_router_v0", idx=0),
                created_at=deterministic_iso(step=0, offset_us=10),
            )
        )
        self.store.add(
            _make_instruction_contract_act(
                act_id=det_act_id(step=0, name="instruction_contract_v0", idx=0),
                created_at=deterministic_iso(step=0, offset_us=11),
            )
        )

    def _load_tokens(self) -> List[str]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            text = f.read()
        toks = tokenize_text(text)
        if not toks:
            raise ValueError("Empty dataset after tokenization.")
        return toks

    def _model_cost_bits(self) -> int:
        return sum(estimate_act_cost_bits(a) for a in self.store.active())

    def _eval_nll_bits(
        self, engine: Engine, tokens: Sequence[str], *, start: int, length: int
    ) -> float:
        cfg = engine.config
        ctx: List[str] = ["<BOS>"] * (cfg.max_order - 1)
        nll_bits = 0.0
        end = start + length
        for i in range(start, end):
            tok = tokens[i % len(tokens)]
            lp = engine.logprob_next(context=ctx, token=tok)
            nll_bits += -lp / math.log(2)
            ctx.append(tok)
            ctx = ctx[-(cfg.max_order - 1) :]
        return nll_bits

    def _eval_chat_harness_metrics(self, engine: Engine) -> Dict[str, float]:
        _transcripts, metrics = run_chat_suite(
            engine,
            dialogues=CHAT_DIALOGUES_20X3,
            max_new_tokens=self.config.fluency_gen_tokens,
            prefix_k=self.config.suite_prefix_k,
            template_ngram_n=self.config.suite_template_ngram_n,
            template_prefix_window=self.config.suite_template_prefix_window,
        )
        return metrics

    def _eval_skill_suite_metrics(self, engine: Engine) -> Dict[str, Any]:
        _transcripts, metrics = run_skill_suite(
            engine,
            max_new_tokens=self.config.skill_suite_max_new_tokens,
        )
        return dict(metrics)

    def _get_mode_policy_act(self) -> Optional[Act]:
        acts = self.store.by_kind("mode_policy")
        return acts[0] if acts else None

    def _get_fact_memory_act(self) -> Optional[Act]:
        for act in self.store.by_kind("memory_facts"):
            ev = act.evidence
            if isinstance(ev, dict) and ev.get("name") == "fact_memory_v0":
                return act
        return None

    def _get_macro_library_act(self) -> Optional[Act]:
        for act in self.store.by_kind("compressor"):
            ev = act.evidence
            if isinstance(ev, dict) and ev.get("name") == "macro_library_v0":
                return act
        return None

    def _get_macro_router_act(self) -> Optional[Act]:
        for act in self.store.by_kind("compressor"):
            ev = act.evidence
            if isinstance(ev, dict) and ev.get("name") == "macro_router_v0":
                return act
        return None

    def _get_anti_template_act(self) -> Optional[Act]:
        for act in self.store.by_kind("rewrite_rule"):
            ev = act.evidence
            if isinstance(ev, dict) and ev.get("name") == "anti_template_v2":
                return act
        return None

    def _update_macro_library_from_transcripts(
        self, transcripts: Sequence[Dict[str, Any]], *, step: int
    ) -> Dict[str, Any]:
        act = self._get_macro_library_act()
        if act is None:
            return {"enabled": False}
        ev = act.evidence
        if not isinstance(ev, dict) or not bool(ev.get("enabled", False)):
            return {"enabled": False}

        macros = ev.setdefault("macros", {})
        if not isinstance(macros, dict):
            macros = {}
            ev["macros"] = macros

        budget = int(ev.get("budget", 128) or 128)
        max_new = int(ev.get("max_new_per_window", 8) or 8)
        n_min = int(ev.get("n_min", 3) or 3)
        n_max = int(ev.get("n_max", 6) or 6)
        min_count = int(ev.get("min_count", 4) or 4)
        max_tokens = int(ev.get("max_tokens_per_reply", 96) or 96)
        set_min_size = int(ev.get("set_min_size", 2) or 2)
        set_min_count = int(ev.get("set_min_count", 4) or 4)
        transition_top_k = int(ev.get("transition_top_k", 32) or 32)

        if (
            budget <= 0
            or max_new <= 0
            or n_min <= 0
            or n_max < n_min
            or min_count <= 1
            or set_min_size <= 0
            or set_min_count <= 1
            or transition_top_k < 0
        ):
            return {"enabled": True, "updated": False, "reason": "invalid_params"}

        def sym_cost_bits(sym: str) -> int:
            b = sym.encode("utf-8", errors="ignore")
            return 8 * (len(b) + 1)

        def macro_entry_cost_bits(ent: Dict[str, Any]) -> int:
            return 8 * len(canonical_json_dumps(ent).encode("utf-8")) + 128

        # Extract causal traces (per token) from suite transcripts.
        seqs_all: List[List[str]] = []
        seq_pairs_mine: List[Tuple[List[str], List[str]]] = []
        set_tokens_all: List[Tuple[str, ...]] = []
        set_counts: Counter = Counter()
        set_contexts: Dict[Tuple[str, ...], set] = {}
        transition_counts: Counter = Counter()
        rewrite_hit_counts: Counter = Counter()

        for rec in transcripts:
            turns = rec.get("turns", [])
            for t in turns:
                mode = str(t.get("mode") or "default")
                tr = t.get("trace") or {}
                if not isinstance(tr, dict):
                    continue
                ctx_keys0 = tr.get("context_keys") or []
                if not isinstance(ctx_keys0, list):
                    ctx_keys0 = []

                sel_ids = tr.get("selected_source_act_ids")
                if not isinstance(sel_ids, list):
                    sel_ids = tr.get("selected_act_ids") or []
                if not isinstance(sel_ids, list):
                    sel_ids = []
                Ls = int(len(sel_ids))
                if ctx_keys0:
                    Ls = min(int(Ls), int(len(ctx_keys0)))
                if max_tokens > 0:
                    Ls = min(int(Ls), int(max_tokens))

                seq: List[str] = []
                seq_ctx: List[str] = []
                for i in range(int(Ls)):
                    aid = sel_ids[i]
                    if not isinstance(aid, str) or (not aid) or aid.startswith("__"):
                        continue
                    seq.append(str(aid))
                    ck = ctx_keys0[i] if i < len(ctx_keys0) else ""
                    if isinstance(ck, str) and ck:
                        seq_ctx.append(f"{mode}\u241f{ck}")
                    else:
                        seq_ctx.append("")
                if max_tokens > 0:
                    seq = seq[:max_tokens]
                if seq:
                    seqs_all.append(seq)
                    if len(seq) >= n_min:
                        seq_pairs_mine.append((seq, seq_ctx))

                # Predictor co-activation sets (token-aligned).
                exec_by_tok = tr.get("executed_predictor_ids") or []
                sel_toks = tr.get("selected_tokens") or []
                if not isinstance(exec_by_tok, list):
                    exec_by_tok = []
                if not isinstance(sel_toks, list):
                    sel_toks = []
                L = min(len(exec_by_tok), len(sel_toks)) if sel_toks else len(exec_by_tok)
                if max_tokens > 0:
                    L = min(int(L), int(max_tokens))
                prev_set: Optional[Tuple[str, ...]] = None
                for i in range(int(L)):
                    ids = exec_by_tok[i]
                    if not isinstance(ids, list):
                        continue
                    uniq = sorted(
                        {
                            str(x)
                            for x in ids
                            if isinstance(x, str) and x and (not x.startswith("__"))
                        }
                    )
                    if len(uniq) < set_min_size:
                        continue
                    st = tuple(uniq)
                    set_tokens_all.append(st)
                    set_counts[st] += 1
                    ck = ctx_keys0[i] if i < len(ctx_keys0) else ""
                    if isinstance(ck, str) and ck:
                        ctx_sig = f"{mode}\u241f{ck}"
                        s = set_contexts.get(st)
                        if s is None:
                            s = set()
                            set_contexts[st] = s
                        s.add(ctx_sig)
                    if prev_set is not None:
                        transition_counts[(prev_set, st)] += 1
                    prev_set = st

                # Rewrite-rule hits (cheap v0 signal).
                rr_hits = tr.get("rewrite_rule_hit_ids") or []
                if isinstance(rr_hits, list):
                    Lh = min(len(rr_hits), len(sel_toks)) if sel_toks else len(rr_hits)
                    if max_tokens > 0:
                        Lh = min(int(Lh), int(max_tokens))
                    for hits in rr_hits[:Lh]:
                        if not isinstance(hits, list):
                            continue
                        for hid in hits:
                            if isinstance(hid, str) and hid:
                                rewrite_hit_counts[hid] += 1

        if not seqs_all and not set_tokens_all:
            ev["last_update_step"] = int(step)
            return {
                "enabled": True,
                "shadow_mode": bool(ev.get("shadow_mode", True)),
                "updated": False,
                "reason": "no_traces",
                "total_macros": int(len(macros)),
            }

        # Helper: compress selected-act sequences using seq-macros (greedy, deterministic).
        def _seq_patterns_from_macros(m: Dict[str, Any]) -> List[Tuple[int, Tuple[str, ...], str]]:
            pats: List[Tuple[int, Tuple[str, ...], str]] = []
            for mid, ent in m.items():
                if not isinstance(ent, dict):
                    continue
                if str(ent.get("type", "")) != "seq":
                    continue
                pat = ent.get("pattern")
                if not isinstance(pat, list) or not pat:
                    continue
                toks = tuple(str(x) for x in pat if x is not None)
                if len(toks) < 2:
                    continue
                pats.append((len(toks), toks, str(mid)))
            pats.sort(key=lambda x: (-x[0], x[2], x[1]))
            return pats

        def _compress_seq_bits(
            seqs: Sequence[Sequence[str]], patterns: Sequence[Tuple[int, Tuple[str, ...], str]]
        ) -> Tuple[int, int, int, int]:
            bits = 0
            symbols_after = 0
            covered = 0
            hits = 0
            for seq in seqs:
                i = 0
                while i < len(seq):
                    matched = False
                    for plen, pat, mid in patterns:
                        if i + plen <= len(seq) and tuple(seq[i : i + plen]) == pat:
                            bits += sym_cost_bits(mid)
                            symbols_after += 1
                            covered += int(plen)
                            hits += 1
                            i += plen
                            matched = True
                            break
                    if not matched:
                        bits += sym_cost_bits(str(seq[i]))
                        symbols_after += 1
                        i += 1
            return int(bits), int(symbols_after), int(covered), int(hits)

        # Helper: compress executed-predictor sets using set-macros (exact match).
        def _set_map_from_macros(m: Dict[str, Any]) -> Dict[Tuple[str, ...], str]:
            out: Dict[Tuple[str, ...], str] = {}
            for mid, ent in m.items():
                if not isinstance(ent, dict):
                    continue
                if str(ent.get("type", "")) != "set":
                    continue
                pat = ent.get("pattern")
                if not isinstance(pat, list) or not pat:
                    continue
                toks = tuple(sorted(str(x) for x in pat if x is not None))
                if len(toks) < 1:
                    continue
                out[toks] = str(mid)
            return out

        def _compress_set_bits(
            sets: Sequence[Tuple[str, ...]], set_map: Dict[Tuple[str, ...], str]
        ) -> Tuple[int, int, int, int]:
            bits = 0
            symbols_before = 0
            symbols_after = 0
            hits = 0
            for st in sets:
                symbols_before += int(len(st))
                mid = set_map.get(st)
                if mid is None:
                    for pid in st:
                        bits += sym_cost_bits(pid)
                    symbols_after += int(len(st))
                else:
                    bits += sym_cost_bits(mid)
                    symbols_after += 1
                    hits += 1
            return int(bits), int(symbols_before), int(symbols_after), int(hits)

        # Existing macros by type.
        existing_seq_patterns = _seq_patterns_from_macros(macros)
        existing_set_map = _set_map_from_macros(macros)

        # Raw trace bits (baseline, no macros).
        raw_seq_bits = sum(sym_cost_bits(sym) for seq in seqs_all for sym in seq)
        raw_seq_symbols = sum(len(seq) for seq in seqs_all)
        raw_set_bits = sum(sym_cost_bits(pid) for st in set_tokens_all for pid in st)
        raw_set_symbols = sum(len(st) for st in set_tokens_all)
        trace_mdl_before_bits_est = int(raw_seq_bits + raw_set_bits)

        # Baseline compressed bits with current macro set.
        base_seq_bits, base_seq_symbols_after, base_seq_cov, base_seq_hits = _compress_seq_bits(
            seqs_all, existing_seq_patterns
        )
        base_set_bits, base_set_symbols_before, base_set_symbols_after, base_set_hits = _compress_set_bits(
            set_tokens_all, existing_set_map
        )

        # Mine candidate sequential patterns (+ distinct contexts).
        seq_ngram_counts: Counter = Counter()
        seq_ngram_contexts: Dict[Tuple[str, ...], set] = {}
        for seq, seq_ctx in seq_pairs_mine:
            L = int(len(seq))
            for n in range(n_min, n_max + 1):
                if L < n:
                    continue
                for i in range(int(L - n + 1)):
                    ng = tuple(seq[i : i + n])
                    seq_ngram_counts[ng] += 1
                    if i < len(seq_ctx) and seq_ctx[i]:
                        s = seq_ngram_contexts.get(ng)
                        if s is None:
                            s = set()
                            seq_ngram_contexts[ng] = s
                        s.add(str(seq_ctx[i]))
        seq_items = [(ng, int(c)) for ng, c in seq_ngram_counts.items() if int(c) >= min_count]
        seq_items.sort(key=lambda kv: (-int(kv[1]) * (len(kv[0]) - 1), -int(kv[1]), kv[0]))

        # Mine candidate predictor-set patterns.
        set_items = [(st, int(c)) for st, c in set_counts.items() if int(c) >= set_min_count]
        set_items.sort(key=lambda kv: (-int(kv[1]) * (len(kv[0]) - 1), -int(kv[1]), kv[0]))

        # Update existing macro counts from this window's mined counts.
        updated_existing = 0
        for mid, ent in macros.items():
            if not isinstance(ent, dict):
                continue
            typ = str(ent.get("type", ""))
            pat = ent.get("pattern")
            if not isinstance(pat, list):
                continue
            ctx_n = 0
            if typ == "seq":
                key = tuple(str(x) for x in pat if x is not None)
                c = int(seq_ngram_counts.get(key, 0) or 0)
                ctx_n = int(len(seq_ngram_contexts.get(key, set())))
            elif typ == "set":
                key2 = tuple(sorted(str(x) for x in pat if x is not None))
                c = int(set_counts.get(key2, 0) or 0)
                ctx_n = int(len(set_contexts.get(key2, set())))
            else:
                continue
            if c > 0:
                ent["count"] = int(ent.get("count", 0) or 0) + int(c)
                ent["last_seen_step"] = int(step)
                if ctx_n > 0:
                    ent["contexts_distinct"] = max(int(ent.get("contexts_distinct", 0) or 0), int(ctx_n))
                updated_existing += 1

        # Greedy selection: pick up to max_new macros with positive *real* MDL gain vs current macros.
        added = 0
        added_seq = 0
        added_set = 0
        candidates_seq = list(seq_items)
        candidates_set = list(set_items)
        cur_seq_patterns = list(existing_seq_patterns)
        cur_set_map = dict(existing_set_map)
        cur_seq_bits = int(base_seq_bits)
        cur_set_bits = int(base_set_bits)

        def _new_seq_entry(
            mid: str, pat: Tuple[str, ...], c: int, gain_bits: int, ctx_n: int
        ) -> Dict[str, Any]:
            return {
                "id": mid,
                "type": "seq",
                "pattern": list(pat),
                "deps": sorted(set(pat)),
                "n": int(len(pat)),
                "count": int(c),
                "last_seen_step": int(step),
                "contexts_distinct": int(max(0, int(ctx_n))),
                "gain_real_bits_est": int(gain_bits),
            }

        def _new_set_entry(
            mid: str, pat: Tuple[str, ...], c: int, gain_bits: int, ctx_n: int
        ) -> Dict[str, Any]:
            return {
                "id": mid,
                "type": "set",
                "pattern": list(pat),
                "deps": sorted(set(pat)),
                "k": int(len(pat)),
                "count": int(c),
                "last_seen_step": int(step),
                "contexts_distinct": int(max(0, int(ctx_n))),
                "gain_real_bits_est": int(gain_bits),
            }

        for _ in range(int(max_new)):
            best: Optional[Tuple[int, int, str, Any, Any, Any]] = None
            # (gain_bits, count, kind, pat, new_seq_patterns, new_set_map)

            for pat, c in candidates_seq:
                raw = ("seq|" + "|".join(pat)).encode("utf-8")
                mid = f"mseq_{sha256_hex(raw)[:12]}"
                if mid in macros:
                    continue
                pats2 = list(cur_seq_patterns)
                pats2.append((len(pat), tuple(pat), mid))
                pats2.sort(key=lambda x: (-x[0], x[2], x[1]))
                new_seq_bits, _sym_after, _cov, _hits = _compress_seq_bits(seqs_all, pats2)
                delta = int(cur_seq_bits - new_seq_bits)
                if delta <= 0:
                    continue
                ctx_n = int(len(seq_ngram_contexts.get(tuple(pat), set())))
                entry = _new_seq_entry(mid, tuple(pat), int(c), int(delta), int(ctx_n))
                cost = macro_entry_cost_bits(entry)
                gain = int(delta - cost)
                if gain <= 0:
                    continue
                cand_key = (gain, int(c), "seq", tuple(pat))
                if best is None or cand_key > (best[0], best[1], best[2], best[3]):
                    best = (gain, int(c), "seq", tuple(pat), pats2, None)

            for st, c in candidates_set:
                raw = ("set|" + "|".join(st)).encode("utf-8")
                mid = f"mset_{sha256_hex(raw)[:12]}"
                if mid in macros:
                    continue
                if st in cur_set_map:
                    continue
                map2 = dict(cur_set_map)
                map2[tuple(st)] = mid
                new_set_bits, _b, _a, _h = _compress_set_bits(set_tokens_all, map2)
                delta = int(cur_set_bits - new_set_bits)
                if delta <= 0:
                    continue
                ctx_n = int(len(set_contexts.get(tuple(st), set())))
                entry = _new_set_entry(mid, tuple(st), int(c), int(delta), int(ctx_n))
                cost = macro_entry_cost_bits(entry)
                gain = int(delta - cost)
                if gain <= 0:
                    continue
                cand_key = (gain, int(c), "set", tuple(st))
                if best is None or cand_key > (best[0], best[1], best[2], best[3]):
                    best = (gain, int(c), "set", tuple(st), None, map2)

            if best is None:
                break

            gain, c, kind, pat, pats2, map2 = best
            if kind == "seq":
                raw = ("seq|" + "|".join(pat)).encode("utf-8")
                mid = f"mseq_{sha256_hex(raw)[:12]}"
                ctx_n = int(len(seq_ngram_contexts.get(tuple(pat), set())))
                entry = _new_seq_entry(mid, tuple(pat), int(c), int(gain), int(ctx_n))
                macros[mid] = entry
                cur_seq_patterns = list(pats2) if pats2 is not None else cur_seq_patterns
                cur_seq_bits, _sym_after, _cov, _hits = _compress_seq_bits(seqs_all, cur_seq_patterns)
                added_seq += 1
            else:
                raw = ("set|" + "|".join(pat)).encode("utf-8")
                mid = f"mset_{sha256_hex(raw)[:12]}"
                ctx_n = int(len(set_contexts.get(tuple(pat), set())))
                entry = _new_set_entry(mid, tuple(pat), int(c), int(gain), int(ctx_n))
                macros[mid] = entry
                cur_set_map = dict(map2) if map2 is not None else cur_set_map
                cur_set_bits, _b, _a, _h = _compress_set_bits(set_tokens_all, cur_set_map)
                added_set += 1
            added += 1

        # Evict to budget deterministically by utility (gain_real_bits_est, count, id).
        evicted = 0
        if budget > 0 and len(macros) > budget:
            ranked: List[Tuple[int, int, str]] = []
            for mid, ent in macros.items():
                if not isinstance(ent, dict):
                    continue
                ranked.append(
                    (
                        int(ent.get("gain_real_bits_est", 0) or 0),
                        int(ent.get("count", 0) or 0),
                        str(mid),
                    )
                )
            ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))
            keep = {mid for _g, _c, mid in ranked[:budget]}
            for mid in list(macros.keys()):
                if mid not in keep:
                    macros.pop(mid, None)
                    evicted += 1

        # Update top transitions (explicit, deterministic) for audit/debug.
        top_transitions: List[Dict[str, Any]] = []
        if transition_top_k > 0 and transition_counts:
            trans_items = [
                (a, b, int(c)) for (a, b), c in transition_counts.items() if int(c) >= 2
            ]
            trans_items.sort(key=lambda x: (-x[2], x[0], x[1]))
            for a, b, c in trans_items[:transition_top_k]:
                top_transitions.append({"from": list(a), "to": list(b), "count": int(c)})
        ev["top_set_transitions"] = top_transitions

        # Recompute final compression stats & explicit MDL.
        final_seq_patterns = _seq_patterns_from_macros(macros)
        final_set_map = _set_map_from_macros(macros)
        seq_bits_after, seq_symbols_after, seq_cov, seq_hits = _compress_seq_bits(
            seqs_all, final_seq_patterns
        )
        set_bits_after, set_symbols_before, set_symbols_after, set_hits = _compress_set_bits(
            set_tokens_all, final_set_map
        )

        macro_cost_bits_total = 0
        for ent in macros.values():
            if not isinstance(ent, dict):
                continue
            macro_cost_bits_total += macro_entry_cost_bits(ent)

        trace_mdl_after_bits_est = int(seq_bits_after + set_bits_after)
        trace_mdl_gain_bits_est = int(trace_mdl_before_bits_est - trace_mdl_after_bits_est)
        trace_mdl_gain_real_bits_est = int(
            trace_mdl_before_bits_est - (trace_mdl_after_bits_est + macro_cost_bits_total)
        )

        cleared = False
        if trace_mdl_gain_real_bits_est <= 0 and macros:
            macros.clear()
            ev["top_set_transitions"] = []
            cleared = True
            macro_cost_bits_total = 0
            final_seq_patterns = []
            final_set_map = {}
            seq_bits_after, seq_symbols_after, seq_cov, seq_hits = _compress_seq_bits(
                seqs_all, final_seq_patterns
            )
            set_bits_after, set_symbols_before, set_symbols_after, set_hits = _compress_set_bits(
                set_tokens_all, final_set_map
            )
            trace_mdl_after_bits_est = int(seq_bits_after + set_bits_after)
            trace_mdl_gain_bits_est = int(trace_mdl_before_bits_est - trace_mdl_after_bits_est)
            trace_mdl_gain_real_bits_est = int(
                trace_mdl_before_bits_est - (trace_mdl_after_bits_est + macro_cost_bits_total)
            )

        # Dependency depth (for future macro-in-macro).
        memo: Dict[str, int] = {}

        def _depth(mid: str, stack: set) -> int:
            if mid in memo:
                return memo[mid]
            if mid in stack:
                return 1
            ent = macros.get(mid)
            if not isinstance(ent, dict):
                memo[mid] = 1
                return 1
            pat = ent.get("pattern")
            if not isinstance(pat, list):
                memo[mid] = 1
                return 1
            stack.add(mid)
            d = 1
            for sym in pat:
                s = str(sym)
                if s in macros:
                    d = max(d, 1 + _depth(s, stack))
            stack.remove(mid)
            memo[mid] = int(d)
            return int(d)

        depths = [_depth(str(mid), set()) for mid in macros.keys()]
        depths.sort()
        depth_max = int(depths[-1]) if depths else 0
        depth_med = int(depths[len(depths) // 2]) if depths else 0

        top_macros: List[Dict[str, Any]] = []
        ranked_top: List[Tuple[int, int, str, Dict[str, Any]]] = []
        for mid, ent in macros.items():
            if not isinstance(ent, dict):
                continue
            ranked_top.append(
                (
                    int(ent.get("gain_real_bits_est", 0) or 0),
                    int(ent.get("count", 0) or 0),
                    str(mid),
                    ent,
                )
            )
        ranked_top.sort(key=lambda x: (-x[0], -x[1], x[2]))
        for g, c, mid, ent in ranked_top[:10]:
            top_macros.append(
                {
                    "id": mid,
                    "type": str(ent.get("type", "")),
                    "count": int(c),
                    "gain_real_bits_est": int(g),
                    "contexts_distinct": int(ent.get("contexts_distinct", 0) or 0),
                    "pattern": list(ent.get("pattern") or []),
                }
            )

        rr_top = []
        if rewrite_hit_counts:
            rr_items = [(rid, int(c)) for rid, c in rewrite_hit_counts.items()]
            rr_items.sort(key=lambda x: (-x[1], x[0]))
            rr_top = [{"id": rid, "count": int(c)} for rid, c in rr_items[:10]]

        total_symbols_before = int(raw_seq_symbols + raw_set_symbols)
        total_symbols_after = int(seq_symbols_after + set_symbols_after)
        num_turns = int(len(seqs_all))
        avg_before = float(total_symbols_before / max(1, num_turns))
        avg_after = float(total_symbols_after / max(1, num_turns))

        macro_hit_rate_seq = float((seq_cov / raw_seq_symbols) if raw_seq_symbols > 0 else 0.0)
        macro_hit_rate_set = float((set_hits / len(set_tokens_all)) if set_tokens_all else 0.0)

        ev["last_update_step"] = int(step)

        return {
            "enabled": True,
            "shadow_mode": bool(ev.get("shadow_mode", True)),
            "updated": True,
            "cleared_for_net_mdl": bool(cleared),
            "added": int(added),
            "added_seq": int(added_seq),
            "added_set": int(added_set),
            "updated_existing": int(updated_existing),
            "evicted": int(evicted),
            "total_macros": int(len(macros)),
            "unique_seq_patterns": int(len(seq_ngram_counts)),
            "unique_set_patterns": int(len(set_counts)),
            "trace_mdl_before_bits_est": int(trace_mdl_before_bits_est),
            "trace_mdl_after_bits_est": int(trace_mdl_after_bits_est),
            "trace_mdl_gain_bits_est": int(trace_mdl_gain_bits_est),
            "macro_library_cost_bits_est": int(macro_cost_bits_total),
            "trace_mdl_gain_real_bits_est": int(trace_mdl_gain_real_bits_est),
            "raw_seq_symbols": int(raw_seq_symbols),
            "raw_set_symbols": int(raw_set_symbols),
            "compressed_seq_symbols": int(seq_symbols_after),
            "compressed_set_symbols": int(set_symbols_after),
            "macro_hit_rate_seq": float(macro_hit_rate_seq),
            "macro_hit_rate_set": float(macro_hit_rate_set),
            "macro_hits_seq": int(seq_hits),
            "macro_hits_set": int(set_hits),
            "avg_trace_len_before": float(avg_before),
            "avg_trace_len_after": float(avg_after),
            "depth_compositional_max": int(depth_max),
            "depth_compositional_median": int(depth_med),
            "top_macros": top_macros,
            "top_rewrite_rule_hits": rr_top,
            "top_set_transitions": top_transitions[:10],
        }

    def _update_macro_router_from_transcripts(
        self, transcripts: Sequence[Dict[str, Any]], *, step: int
    ) -> Dict[str, Any]:
        act = self._get_macro_router_act()
        if act is None:
            return {"enabled": False}
        ev = act.evidence
        if not isinstance(ev, dict) or not bool(ev.get("enabled", False)):
            return {"enabled": False}

        table = ev.setdefault("table", {})
        if not isinstance(table, dict):
            table = {}
            ev["table"] = table

        max_contexts = int(ev.get("max_contexts", 8192) or 8192)
        top_k = int(ev.get("top_k", 4) or 4)
        if max_contexts <= 0 or top_k <= 0:
            return {"enabled": True, "updated": False, "reason": "invalid_params"}

        # ctx_sig -> predictor_id -> count (winner-based; gating should preserve the winner)
        ctx_counts: Dict[str, Counter] = {}
        total_tokens = 0
        for rec in transcripts:
            turns = rec.get("turns", [])
            for t in turns:
                mode = str(t.get("mode") or "default")
                tr = t.get("trace") or {}
                if not isinstance(tr, dict):
                    continue
                ctx_keys = tr.get("context_keys") or []
                winners = tr.get("selected_source_act_ids") or []
                sel_toks = tr.get("selected_tokens") or []
                if not isinstance(ctx_keys, list) or not isinstance(winners, list):
                    continue
                if not isinstance(sel_toks, list):
                    sel_toks = []
                L = min(len(ctx_keys), len(winners))
                if sel_toks:
                    L = min(int(L), int(len(sel_toks)))
                for i in range(int(L)):
                    ck = ctx_keys[i]
                    win = winners[i]
                    if not isinstance(ck, str) or not isinstance(win, str):
                        continue
                    ctx_sig = f"{mode}\u241f{ck}"
                    c = ctx_counts.get(ctx_sig)
                    if c is None:
                        c = Counter()
                        ctx_counts[ctx_sig] = c
                    if win and (not win.startswith("__")):
                        c[win] += 1
                        total_tokens += 1

        added_ctx = 0
        updated_ctx = 0
        for ctx_sig, c in ctx_counts.items():
            entry = table.get(ctx_sig)
            if entry is None or not isinstance(entry, dict):
                entry = {"counts": {}, "predictors": [], "total": 0}
                table[ctx_sig] = entry
                added_ctx += 1
            counts0 = entry.get("counts")
            if not isinstance(counts0, dict):
                counts0 = {}
                entry["counts"] = counts0
            # merge counts
            for pid, cnt in c.items():
                counts0[pid] = int(counts0.get(pid, 0) or 0) + int(cnt)
            entry["total"] = int(sum(int(v) for v in counts0.values()))
            items = [(pid, int(cnt)) for pid, cnt in counts0.items()]
            items.sort(key=lambda kv: (-kv[1], kv[0]))
            entry["predictors"] = [pid for pid, _cnt in items[:top_k]]
            updated_ctx += 1

        # Budget eviction (deterministic): drop lowest-total contexts.
        evicted = 0
        if max_contexts > 0 and len(table) > max_contexts:
            scored: List[Tuple[int, str]] = []
            for k, v in table.items():
                if not isinstance(v, dict):
                    continue
                scored.append((int(v.get("total", 0) or 0), str(k)))
            scored.sort(key=lambda x: (x[0], x[1]))
            while len(table) > max_contexts and scored:
                _tot, k = scored.pop(0)
                table.pop(k, None)
                evicted += 1

        # Shadow "would-have" metrics: predictor evals saved if we gated by router table.
        baseline_eval = 0
        router_eval = 0
        saved = 0
        fastpath = 0
        fallback = 0
        covered = 0
        total = 0
        winner_in_allowed = 0

        for rec in transcripts:
            turns = rec.get("turns", [])
            for t in turns:
                mode = str(t.get("mode") or "default")
                tr = t.get("trace") or {}
                if not isinstance(tr, dict):
                    continue
                ctx_keys = tr.get("context_keys") or []
                exec_by_tok = tr.get("executed_predictor_ids") or []
                pred_iter = tr.get("predictor_iterated") or []
                sel_toks = tr.get("selected_tokens") or []
                winners = tr.get("selected_source_act_ids") or []
                if not isinstance(ctx_keys, list) or not isinstance(exec_by_tok, list) or not isinstance(pred_iter, list):
                    continue
                if not isinstance(sel_toks, list):
                    sel_toks = []
                if not isinstance(winners, list):
                    winners = []
                L = min(len(ctx_keys), len(exec_by_tok), len(pred_iter))
                if sel_toks:
                    L = min(int(L), int(len(sel_toks)))
                if winners:
                    L = min(int(L), int(len(winners)))
                for i in range(int(L)):
                    ck = ctx_keys[i]
                    ids = exec_by_tok[i]
                    m = pred_iter[i]
                    win = winners[i] if i < len(winners) else None
                    if not isinstance(ck, str) or not isinstance(ids, list):
                        continue
                    try:
                        base = int(m)
                    except Exception:
                        base = 0
                    base = max(0, base)
                    ctx_sig = f"{mode}\u241f{ck}"
                    entry = table.get(ctx_sig)
                    allowed: List[str] = []
                    if isinstance(entry, dict):
                        allowed = entry.get("predictors") or []
                    if isinstance(allowed, list):
                        allowed = [str(x) for x in allowed if isinstance(x, str) and x]
                    else:
                        allowed = []
                    total += 1
                    baseline_eval += base
                    if allowed:
                        covered += 1
                        if isinstance(win, str) and win in set(allowed):
                            winner_in_allowed += 1
                        exec_set = {str(x) for x in ids if isinstance(x, str) and x}
                        if exec_set.intersection(set(allowed)):
                            fastpath += 1
                            r = int(len(allowed))
                            router_eval += r
                            if base > r:
                                saved += int(base - r)
                        else:
                            # Would fall back to full scan if gating yields no candidates.
                            fallback += 1
                            router_eval += base
                    else:
                        router_eval += base

        ev["last_update_step"] = int(step)
        return {
            "enabled": True,
            "shadow_mode": bool(ev.get("shadow_mode", True)),
            "updated": True,
            "added_contexts": int(added_ctx),
            "updated_contexts": int(updated_ctx),
            "evicted": int(evicted),
            "total_contexts": int(len(table)),
            "total_tokens": int(total_tokens),
            "shadow_metrics": {
                "tokens": int(total),
                "predictors_evaluated_baseline_per_token_mean": float(
                    baseline_eval / max(1, total)
                ),
                "predictors_evaluated_router_per_token_mean": float(router_eval / max(1, total)),
                "estimated_act_evals_saved": int(saved),
                "would_skip_rate": float((saved / max(1, baseline_eval)) if baseline_eval > 0 else 0.0),
                "fastpath_rate": float((fastpath / max(1, total)) if total > 0 else 0.0),
                "fallback_rate": float((fallback / max(1, total)) if total > 0 else 0.0),
                "coverage_rate": float((covered / max(1, total)) if total > 0 else 0.0),
                "winner_in_allowed_rate": float(
                    (winner_in_allowed / max(1, total)) if total > 0 else 0.0
                ),
            },
        }

    def _update_anti_template_from_transcripts(
        self,
        transcripts: Sequence[Dict[str, Any]],
        *,
        step: int,
        template_ngram_n: int,
        template_prefix_window: int,
    ) -> Dict[str, Any]:
        act = self._get_anti_template_act()
        if act is None:
            return {"enabled": False}
        ev = act.evidence
        if not isinstance(ev, dict):
            return {"enabled": False}

        ka = ev.get("ka_template")
        if not isinstance(ka, dict):
            return {"enabled": False}
        if not bool(ka.get("enabled", False)):
            return {"enabled": False}

        n = int(ka.get("n", template_ngram_n) or template_ngram_n)
        prefix_window = int(
            ka.get("prefix_window", template_prefix_window) or template_prefix_window
        )
        top_k = int(ka.get("top_k", 32) or 32)
        min_count = int(ka.get("min_count", 2) or 2)

        if n <= 0 or prefix_window <= 0 or top_k <= 0 or min_count <= 1:
            return {
                "enabled": True,
                "updated": False,
                "reason": "invalid_params",
                "n": int(n),
                "prefix_window": int(prefix_window),
                "top_k": int(top_k),
                "min_count": int(min_count),
            }

        counts: Counter = Counter()
        total_ngrams = 0
        for rec in transcripts:
            turns = rec.get("turns", [])
            for t in turns:
                resp = str(t.get("system", ""))
                toks = tokenize_text(resp)
                win = [x.lower() for x in non_ws_tokens(toks)[:prefix_window]]
                if len(win) < n:
                    continue
                for i in range(len(win) - n + 1):
                    ng = tuple(win[i : i + n])
                    counts[ng] += 1
                    total_ngrams += 1

        items = [(ng, c) for ng, c in counts.items() if int(c) >= min_count]
        items.sort(key=lambda kv: (-int(kv[1]), kv[0]))
        selected = items[:top_k]

        ev["ban_ngram_seqs"] = [[str(x) for x in ng] for ng, _c in selected]
        ev["ngram_prefix_window"] = int(prefix_window)
        ev["ka_template_last_update_step"] = int(step)

        top = None
        if selected:
            top_ng, top_c = selected[0]
            top = {"seq": [str(x) for x in top_ng], "count": int(top_c)}

        return {
            "enabled": True,
            "updated": True,
            "n": int(n),
            "prefix_window": int(prefix_window),
            "top_k": int(top_k),
            "min_count": int(min_count),
            "unique_ngrams": int(len(counts)),
            "total_ngrams": int(total_ngrams),
            "selected": int(len(selected)),
            "top": top,
        }

    def _update_fact_memory_from_transcripts(
        self, transcripts: Sequence[Dict[str, Any]], *, step: int
    ) -> Dict[str, Any]:
        act = self._get_fact_memory_act()
        if act is None:
            return {"enabled": False}
        ev = act.evidence
        if not isinstance(ev, dict) or not bool(ev.get("enabled", False)):
            return {"enabled": False}

        table = ev.setdefault("table", {})
        if not isinstance(table, dict):
            table = {}
            ev["table"] = table

        max_facts = int(ev.get("max_facts", 256) or 256)
        max_ev_turns = int(ev.get("max_evidence_turns", 8) or 8)
        max_ans_tokens = int(ev.get("max_answer_tokens", 3) or 3)
        require_capital = bool(ev.get("require_capital_question", True))
        ban_keywords = [
            str(x).lower() for x in (ev.get("ban_keywords") or []) if x is not None
        ]

        anti = self._get_anti_template_act()
        anti_ngram_seqs: List[List[str]] = []
        anti_ban_tokens: set = set()
        if anti is not None and isinstance(anti.evidence, dict):
            anti_ban_tokens = {
                str(x).lower()
                for x in (anti.evidence.get("ban_tokens") or [])
                if x is not None
            }
            for seq in (anti.evidence.get("ban_ngram_seqs") or []):
                if not isinstance(seq, list):
                    continue
                toks = [str(x).lower() for x in seq if x is not None]
                if toks:
                    anti_ngram_seqs.append(toks)

        ws_re = re.compile(r"\s+", flags=re.UNICODE)

        def _norm_space(s: str) -> str:
            return ws_re.sub(" ", s.strip())

        def _strip_edge_punct(s: str) -> str:
            return (
                s.strip()
                .strip("“”\"'`")
                .strip(".,;:!?()[]{}")
                .strip()
            )

        qa_patterns = [
            ("pt", re.compile(r"pergunta:\s*(.*?)\s*resposta:\s*([^\n\r.?!]+)", re.I)),
            ("en", re.compile(r"question:\s*(.*?)\s*answer:\s*([^\n\r.?!]+)", re.I)),
        ]

        def _entity_after_capital(text: str) -> str:
            toks = non_ws_tokens(tokenize_text(text))
            toks_l = [t.lower() for t in toks]
            if "capital" not in toks_l:
                return ""
            i = toks_l.index("capital")
            j = i + 1
            while j < len(toks_l) and toks_l[j] in {"de", "do", "da", "of", "the"}:
                j += 1
            ent = toks[j : j + 3]
            ent = [_strip_edge_punct(x) for x in ent if x]
            ent = [x for x in ent if x]
            return " ".join(x.lower() for x in ent)

        def _first_answer_token(resp: str) -> str:
            toks = non_ws_tokens(tokenize_text(resp))
            if not toks:
                return ""
            return _strip_edge_punct(str(toks[0]))

        def _contains_banned_ngrams(tokens_l: List[str]) -> bool:
            for seq in anti_ngram_seqs:
                n = len(seq)
                if n <= 0 or len(tokens_l) < n:
                    continue
                for i in range(len(tokens_l) - n + 1):
                    if tokens_l[i : i + n] == seq:
                        return True
            return False

        def _is_scaffold_text(s: str) -> bool:
            s_l = s.lower()
            return any(k in s_l for k in ban_keywords)

        added = 0
        updated = 0
        skipped_scaffold = 0
        skipped_conflict = 0
        skipped_invalid = 0

        for rec in transcripts:
            prompt_id = int(rec.get("prompt_id", -1))
            turns = rec.get("turns", [])
            for turn_idx, t in enumerate(turns):
                user_msg = str(t.get("user", ""))
                resp = str(t.get("system", ""))
                mode = str(t.get("mode") or "default")
                user_sig = str(t.get("user_sig") or "")

                if not resp:
                    continue

                pairs: List[Tuple[str, str]] = []
                resp_norm = _norm_space(resp)

                # Strategy 1: explicit QA markers inside system response.
                for _lang, pat in qa_patterns:
                    for m in pat.finditer(resp_norm):
                        q = _strip_edge_punct(_norm_space(m.group(1)))
                        a = _strip_edge_punct(_norm_space(m.group(2)))
                        if q and a:
                            pairs.append((q, a))

                # Strategy 2: capital question in user, short answer in system.
                if not pairs:
                    ent = _entity_after_capital(user_msg)
                    if ent:
                        ans = ""
                        m_pt = re.search(
                            r"resposta:\s*([^\n\r.?!]+)", resp_norm, flags=re.I
                        )
                        m_en = re.search(r"answer:\s*([^\n\r.?!]+)", resp_norm, flags=re.I)
                        if m_pt:
                            ans = _strip_edge_punct(_norm_space(m_pt.group(1)))
                        elif m_en:
                            ans = _strip_edge_punct(_norm_space(m_en.group(1)))
                        else:
                            # Shadow-mode safety: only infer an unlabeled answer if the whole
                            # response is already extremely short (avoid scaffold contamination).
                            resp_toks = non_ws_tokens(tokenize_text(resp))
                            if len(resp_toks) <= max_ans_tokens:
                                ans = _first_answer_token(resp)
                        if ans:
                            pairs.append((f"capital of {ent}", ans))

                for q, a in pairs:
                    if require_capital and ("capital" not in q.lower()):
                        skipped_invalid += 1
                        continue
                    if not q or not a:
                        skipped_invalid += 1
                        continue
                    if _is_scaffold_text(q) or _is_scaffold_text(a):
                        skipped_scaffold += 1
                        continue

                    a_toks = [x.lower() for x in non_ws_tokens(tokenize_text(a))]
                    if not a_toks or len(a_toks) > max_ans_tokens:
                        skipped_invalid += 1
                        continue
                    # Prefer proper-noun short answers (e.g., Lisboa/Paris/Brasília); avoid scaffolds.
                    if not a or (not a[0].isalpha()) or (not a[0].isupper()):
                        skipped_invalid += 1
                        continue
                    if any(x in anti_ban_tokens for x in a_toks):
                        skipped_scaffold += 1
                        continue
                    if any(x in {"->", "-", "summary:", "resumo:"} for x in a_toks):
                        skipped_scaffold += 1
                        continue

                    ent = _entity_after_capital(q)
                    if not ent:
                        skipped_invalid += 1
                        continue
                    fact_key = f"capital_of:{ent}"

                    cand_tokens = [
                        x.lower()
                        for x in non_ws_tokens(tokenize_text(fact_key + " " + a))
                    ]
                    if _contains_banned_ngrams(cand_tokens):
                        skipped_scaffold += 1
                        continue

                    entry = table.get(fact_key)
                    if entry is None:
                        entry = {
                            "value": a,
                            "confidence": 0.5,
                            "evidence_turns": [],
                            "last_seen_step": int(step),
                            "count": 0,
                        }
                        table[fact_key] = entry
                        added += 1

                    if not isinstance(entry, dict):
                        skipped_invalid += 1
                        continue

                    prev_val = str(entry.get("value", ""))
                    if prev_val and prev_val.lower() != a.lower():
                        skipped_conflict += 1
                        continue

                    entry["value"] = a
                    entry["count"] = int(entry.get("count", 0)) + 1
                    cnt = int(entry["count"])
                    entry["confidence"] = float(1.0 - (0.5**cnt))
                    entry["last_seen_step"] = int(step)
                    ev_turns = entry.get("evidence_turns")
                    if not isinstance(ev_turns, list):
                        ev_turns = []
                        entry["evidence_turns"] = ev_turns
                    ev_turns.append(
                        {
                            "step": int(step),
                            "prompt_id": int(prompt_id),
                            "turn": int(turn_idx),
                            "mode": mode,
                            "user_sig": user_sig,
                        }
                    )
                    while len(ev_turns) > max_ev_turns:
                        ev_turns.pop(0)
                    updated += 1

        if max_facts > 0 and len(table) > max_facts:
            ranked: List[Tuple[float, int, str]] = []
            for k, v in table.items():
                if not isinstance(v, dict):
                    continue
                ranked.append(
                    (
                        float(v.get("confidence", 0.0)),
                        int(v.get("count", 0)),
                        str(k),
                    )
                )
            ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))
            keep = {k for _c, _n, k in ranked[:max_facts]}
            for k in list(table.keys()):
                if k not in keep:
                    table.pop(k, None)

        ev["last_update_step"] = int(step)

        top_facts: List[Dict[str, Any]] = []
        ranked2: List[Tuple[float, int, str, Dict[str, Any]]] = []
        for k, v in table.items():
            if not isinstance(v, dict):
                continue
            ranked2.append(
                (float(v.get("confidence", 0.0)), int(v.get("count", 0)), str(k), v)
            )
        ranked2.sort(key=lambda x: (-x[0], -x[1], x[2]))
        for conf, cnt, key, v in ranked2[:10]:
            top_facts.append(
                {
                    "fact_key": key,
                    "value": str(v.get("value", "")),
                    "confidence": float(conf),
                    "count": int(cnt),
                    "last_seen_step": int(v.get("last_seen_step", 0) or 0),
                }
            )

        return {
            "enabled": True,
            "shadow_mode": bool(ev.get("shadow_mode", True)),
            "added": int(added),
            "updated": int(updated),
            "skipped_scaffold": int(skipped_scaffold),
            "skipped_conflict": int(skipped_conflict),
            "skipped_invalid": int(skipped_invalid),
            "total_facts": int(len(table)),
            "top_facts": top_facts,
        }

    def _update_mode_policy_from_transcripts(
        self,
        transcripts: Sequence[Dict[str, Any]],
        *,
        step: int,
        prefix_k: int,
        template_ngram_n: int,
        template_prefix_window: int,
    ) -> Dict[str, Any]:
        act = self._get_mode_policy_act()
        if act is None:
            return {"enabled": False}

        k_user = int(act.evidence.get("k", 2))
        table = act.evidence.setdefault("table", {})
        if not isinstance(table, dict):
            table = {}
            act.evidence["table"] = table

        infos: List[Dict[str, Any]] = []
        reply_sigs: List[str] = []
        prefix_sigs: List[str] = []

        # Deterministic streaming n-gram tracking (same ordering as transcripts).
        seen_template_ngrams: set = set()

        for rec in transcripts:
            turns = rec.get("turns", [])
            dial_seen = set()
            for t in turns:
                user_msg = str(t.get("user", ""))
                mode = str(t.get("mode") or "default")
                resp = str(t.get("system", ""))

                user_sig = user_signature(user_msg, k=k_user)
                toks = tokenize_text(resp)
                r_sig = reply_signature(resp)
                p_sig = prefix_k_signature(toks, k=prefix_k)

                ws = sum(1 for x in toks if x.isspace())
                ws_ratio = ws / max(1, len(toks))
                rep3 = repeat_ngram_rate(toks, 3, ignore_space=True)
                lp = loop_rate(toks, n=3, window=128, ignore_space=True)

                cross_turn_repeat = 1.0 if r_sig in dial_seen else 0.0
                dial_seen.add(r_sig)

                # Template duplication on a prefix-window, streaming global.
                win = [x.lower() for x in non_ws_tokens(toks)[: int(template_prefix_window)]]
                n = int(template_ngram_n)
                if n <= 0 or len(win) < n:
                    template_dup = 0.0
                else:
                    total = len(win) - n + 1
                    rep = 0
                    for i in range(total):
                        ng = tuple(win[i : i + n])
                        if ng in seen_template_ngrams:
                            rep += 1
                        else:
                            seen_template_ngrams.add(ng)
                    template_dup = rep / total if total > 0 else 0.0

                infos.append(
                    {
                        "user_sig": user_sig,
                        "mode": mode,
                        "reply_sig": r_sig,
                        "prefix_sig": p_sig,
                        "ws_ratio": float(ws_ratio),
                        "repeat3": float(rep3),
                        "loop": float(lp),
                        "template_dup": float(template_dup),
                        "cross_turn_repeat": float(cross_turn_repeat),
                    }
                )
                reply_sigs.append(r_sig)
                prefix_sigs.append(p_sig)

        reply_counts = Counter(reply_sigs)
        prefix_counts = Counter(prefix_sigs)

        updates = 0
        for info in infos:
            user_sig = str(info["user_sig"])
            mode = str(info["mode"])
            dup_reply = 1.0 if reply_counts.get(str(info["reply_sig"]), 0) > 1 else 0.0
            dup_prefix = 1.0 if prefix_counts.get(str(info["prefix_sig"]), 0) > 1 else 0.0

            pen = (
                float(info["repeat3"])
                + float(info["loop"])
                + float(info["ws_ratio"])
                + float(dup_reply)
                + float(dup_prefix)
                + float(info["template_dup"])
                + float(info["cross_turn_repeat"])
            )

            row = table.setdefault(user_sig, {})
            if not isinstance(row, dict):
                row = {}
                table[user_sig] = row
            st = row.setdefault(mode, {"trials": 0, "pen_sum": 0.0})
            if not isinstance(st, dict):
                st = {"trials": 0, "pen_sum": 0.0}
                row[mode] = st
            st["trials"] = int(st.get("trials", 0)) + 1
            st["pen_sum"] = float(st.get("pen_sum", 0.0)) + float(pen)
            updates += 1

        act.evidence["last_update_step"] = int(step)
        return {
            "enabled": True,
            "updates": int(updates),
            "table_size": int(len(table)),
            "k_user": int(k_user),
        }

    def _eval_online_window(
        self,
        store: ActStore,
        tokens: Sequence[str],
        *,
        start: int,
        length: int,
        engine_config: EngineConfig,
        patch: Optional[Patch] = None,
    ) -> Dict[str, Any]:
        store_copy: ActStore = copy.deepcopy(store)
        if patch is not None:
            self._apply_patch_to_store(store_copy, patch)
        engine = Engine(store_copy, seed=self.config.seed, config=engine_config)
        ctx: List[str] = ["<BOS>"] * (engine.config.max_order - 1)
        nll_bits = 0.0
        for i in range(int(length)):
            tok = tokens[(start + i) % len(tokens)]
            lp = engine.logprob_next(context=ctx, token=tok)
            nll_bits += -lp / math.log(2)
            engine.observe(context=ctx, token=tok)
            ctx.append(tok)
            ctx = ctx[-(engine.config.max_order - 1) :]

        cost_bits = sum(estimate_act_cost_bits(a) for a in store_copy.active())
        gen: Dict[str, Any] = dict(self._eval_chat_harness_metrics(engine))
        util = self._eval_skill_suite_metrics(engine)
        for k, v in util.items():
            gen[f"utility_{k}"] = v
        return {"nll_bits": nll_bits, "cost_bits": cost_bits, "gen": gen}

    def _apply_patch(self, patch: Patch, *, count: bool) -> Dict[str, Any]:
        added: List[str] = []
        prev_active: Dict[str, bool] = {}

        def _prune(act_id: str) -> None:
            act = self.store.get(act_id)
            if act is None:
                return
            prev_active[act_id] = bool(act.active)
            self.store.prune(act_id)

        kind = patch.kind
        payload = patch.payload
        if kind == "ADD_ACT":
            for a in payload.get("acts", []):
                act = Act.from_dict(a)
                self.store.add(act)
                added.append(act.id)
                if count:
                    self._adds += 1
        elif kind == "PRUNE_ACT":
            for act_id in payload.get("act_ids", []):
                _prune(act_id)
                if count:
                    self._prunes += 1
        elif kind == "MERGE_ACTS":
            if "a_id" in payload and "b_id" in payload:
                a_id = str(payload["a_id"])
                b_id = str(payload["b_id"])
                keep_id = str(payload.get("keep_id") or payload.get("keep") or a_id)
                merge_step = int(payload.get("step", 0))
                policy = dict(payload.get("policy", {}) or {})

                act_a = self.store.get(a_id)
                act_b = self.store.get(b_id)
                if act_a is None or act_b is None:
                    raise ValueError("MERGE_ACTS missing source acts")

                act_keep = self.store.get(keep_id)
                if act_keep is None:
                    raise ValueError("MERGE_ACTS missing keep act")

                # Read explicit evidence tables (source of truth).
                tab_a = act_a.evidence.get("table", {})
                tab_b = act_b.evidence.get("table", {})
                if not isinstance(tab_a, dict) or not isinstance(tab_b, dict):
                    raise ValueError("MERGE_ACTS expects predictor evidence.table dicts")

                hash_a_before = act_a.content_hash()
                hash_b_before = act_b.content_hash()
                stats_a = _table_stats(tab_a)
                stats_b = _table_stats(tab_b)

                merged_table = _union_sum_tables(tab_a, tab_b)

                max_contexts = int(policy.get("max_contexts", act_keep.evidence.get("max_contexts", 0) or 0) or 0)
                max_next = int(
                    policy.get(
                        "max_next_per_ctx", act_keep.evidence.get("max_next_per_ctx", 0) or 0
                    )
                    or 0
                )
                merged_table = _apply_budget_topk(
                    merged_table, max_contexts=max_contexts, max_next_per_ctx=max_next
                )
                evict_policy = str(policy.get("evict_policy", act_keep.evidence.get("evict_policy", "")))
                if not evict_policy:
                    evict_policy = "fifo"

                # Rewrite keep act to represent the merged structure.
                before = act_keep.to_dict()
                act_keep.version = int(act_keep.version) + 1
                act_keep.created_at = deterministic_iso(step=merge_step, offset_us=0)
                if "match" in policy and isinstance(policy["match"], dict):
                    act_keep.match = dict(policy["match"])
                if "merged_name" in policy:
                    act_keep.evidence["name"] = str(policy["merged_name"])
                act_keep.evidence["table"] = merged_table
                if "allow_new_contexts" in policy:
                    act_keep.evidence["allow_new_contexts"] = bool(policy["allow_new_contexts"])
                if "allow_new_tokens" in policy:
                    act_keep.evidence["allow_new_tokens"] = bool(policy["allow_new_tokens"])
                if max_contexts:
                    act_keep.evidence["max_contexts"] = int(max_contexts)
                if max_next:
                    act_keep.evidence["max_next_per_ctx"] = int(max_next)
                if "evict_policy" in policy:
                    act_keep.evidence["evict_policy"] = str(policy["evict_policy"])
                else:
                    act_keep.evidence["evict_policy"] = str(evict_policy)
                if str(act_keep.evidence.get("evict_policy")) == "count_lex":
                    act_keep.evidence.pop("ctx_fifo", None)
                    act_keep.evidence.pop("next_fifo", None)
                act_keep.deps = [a_id, b_id]

                # Prune the non-kept act(s).
                for act_id in (a_id, b_id):
                    if act_id != keep_id:
                        _prune(act_id)
                        if count:
                            self._prunes += 1

                if count:
                    self._merges += 1

                merged_hash = act_keep.content_hash()
                merged_stats = _table_stats(merged_table)
                audit = {
                    "hash_a_before": hash_a_before,
                    "hash_b_before": hash_b_before,
                    "hash_merged_after": merged_hash,
                    "n_ctx_a": int(stats_a["n_ctx"]),
                    "n_ctx_b": int(stats_b["n_ctx"]),
                    "n_ctx_merged": int(merged_stats["n_ctx"]),
                    "n_edges_merged": int(merged_stats["n_edges"]),
                    "budget": {
                        "max_contexts": int(max_contexts),
                        "max_next_per_ctx": int(max_next),
                        "evict_policy": str(act_keep.evidence.get("evict_policy", "")),
                        "policy": policy,
                    },
                }
                return {
                    "added": added,
                    "prev_active": prev_active,
                    "rewritten": (keep_id, before),
                    "meta": {"merge_audit": audit},
                }

            if "add" in payload:
                act = Act.from_dict(payload["add"])
                self.store.add(act)
                added.append(act.id)
                if count:
                    self._merges += 1
                for act_id in payload.get("prune", []):
                    _prune(act_id)
                    if count:
                        self._prunes += 1
            elif "rewrite" in payload:
                rw = payload["rewrite"]
                act_id = str(rw["act_id"])
                act = self.store.get(act_id)
                if act is None:
                    return {"added": added, "prev_active": prev_active, "rewritten": None}
                before = act.to_dict()
                after = Act.from_dict(dict(rw["act"])).to_dict()
                self.store.acts[act_id] = Act.from_dict(after)
                if count:
                    self._merges += 1
                for pid in payload.get("prune", []):
                    _prune(pid)
                    if count:
                        self._prunes += 1
                return {"added": added, "prev_active": prev_active, "rewritten": (act_id, before)}
            else:
                raise ValueError("MERGE_ACTS payload must contain 'add' or 'rewrite'")
        elif kind == "REWRITE_ACT":
            act_id = str(payload["act_id"])
            act = self.store.get(act_id)
            if act is None:
                return {"added": added, "prev_active": prev_active, "rewritten": None}
            payload_act = dict(payload["act"])
            before = act.to_dict()
            after = Act.from_dict(payload_act).to_dict()
            self.store.acts[act_id] = Act.from_dict(after)
            return {"added": added, "prev_active": prev_active, "rewritten": (act_id, before)}
        else:
            raise ValueError(f"Unknown patch kind: {kind}")

        return {"added": added, "prev_active": prev_active, "rewritten": None}

    @staticmethod
    def _apply_patch_to_store(store: ActStore, patch: Patch) -> None:
        kind = patch.kind
        payload = patch.payload
        if kind == "ADD_ACT":
            for a in payload.get("acts", []):
                store.add(Act.from_dict(a))
        elif kind == "PRUNE_ACT":
            for act_id in payload.get("act_ids", []):
                store.prune(str(act_id))
        elif kind == "MERGE_ACTS":
            if "a_id" in payload and "b_id" in payload:
                a_id = str(payload["a_id"])
                b_id = str(payload["b_id"])
                keep_id = str(payload.get("keep_id") or payload.get("keep") or a_id)
                policy = dict(payload.get("policy", {}) or {})

                act_a = store.get(a_id)
                act_b = store.get(b_id)
                act_keep = store.get(keep_id)
                if act_a is None or act_b is None or act_keep is None:
                    return
                tab_a = act_a.evidence.get("table", {})
                tab_b = act_b.evidence.get("table", {})
                if not isinstance(tab_a, dict) or not isinstance(tab_b, dict):
                    return
                merged_table = _union_sum_tables(tab_a, tab_b)
                max_contexts = int(
                    policy.get("max_contexts", act_keep.evidence.get("max_contexts", 0) or 0) or 0
                )
                max_next = int(
                    policy.get(
                        "max_next_per_ctx", act_keep.evidence.get("max_next_per_ctx", 0) or 0
                    )
                    or 0
                )
                merged_table = _apply_budget_topk(
                    merged_table, max_contexts=max_contexts, max_next_per_ctx=max_next
                )
                evict_policy = str(policy.get("evict_policy", act_keep.evidence.get("evict_policy", "")))
                if not evict_policy:
                    evict_policy = "fifo"

                act_keep.version = int(act_keep.version) + 1
                act_keep.created_at = deterministic_iso(step=int(payload.get("step", 0)), offset_us=0)
                if "match" in policy and isinstance(policy["match"], dict):
                    act_keep.match = dict(policy["match"])
                if "merged_name" in policy:
                    act_keep.evidence["name"] = str(policy["merged_name"])
                act_keep.evidence["table"] = merged_table
                if "allow_new_contexts" in policy:
                    act_keep.evidence["allow_new_contexts"] = bool(policy["allow_new_contexts"])
                if "allow_new_tokens" in policy:
                    act_keep.evidence["allow_new_tokens"] = bool(policy["allow_new_tokens"])
                if max_contexts:
                    act_keep.evidence["max_contexts"] = int(max_contexts)
                if max_next:
                    act_keep.evidence["max_next_per_ctx"] = int(max_next)
                if "evict_policy" in policy:
                    act_keep.evidence["evict_policy"] = str(policy["evict_policy"])
                else:
                    act_keep.evidence["evict_policy"] = str(evict_policy)
                if str(act_keep.evidence.get("evict_policy")) == "count_lex":
                    act_keep.evidence.pop("ctx_fifo", None)
                    act_keep.evidence.pop("next_fifo", None)
                act_keep.deps = [a_id, b_id]

                for act_id in (a_id, b_id):
                    if act_id != keep_id:
                        store.prune(str(act_id))
                return

            if "add" in payload:
                store.add(Act.from_dict(payload["add"]))
                for act_id in payload.get("prune", []):
                    store.prune(str(act_id))
            elif "rewrite" in payload:
                rw = payload["rewrite"]
                act_id = str(rw["act_id"])
                store.acts[act_id] = Act.from_dict(dict(rw["act"]))
                for act_id in payload.get("prune", []):
                    store.prune(str(act_id))
            else:
                raise ValueError("MERGE_ACTS payload must contain 'add' or 'rewrite'")
        elif kind == "REWRITE_ACT":
            act_id = str(payload["act_id"])
            store.acts[act_id] = Act.from_dict(dict(payload["act"]))
        else:
            raise ValueError(f"Unknown patch kind: {kind}")

    def _revert_patch(self, undo: Dict[str, Any]) -> None:
        for act_id in undo.get("added", []):
            self.store.remove(act_id)
        for act_id, was_active in undo.get("prev_active", {}).items():
            act = self.store.get(act_id)
            if act is not None:
                act.active = bool(was_active)
        rewritten = undo.get("rewritten")
        if rewritten:
            act_id, before = rewritten
            self.store.acts[act_id] = Act.from_dict(before)

    def _find_act_by_name(self, name: str) -> Optional[Act]:
        for act in self.store.active():
            if act.evidence.get("name") == name:
                return act
        return None

    def _propose_patches(self, *, step: int, tokens: Sequence[str]) -> List[Patch]:
        patches: List[Patch] = []

        has_bigram_space = self._find_act_by_name("bigram_space") is not None
        has_bigram_nonspace = self._find_act_by_name("bigram_nonspace") is not None
        has_bigram_merged = self._find_act_by_name("bigram_merged") is not None
        has_trigram = self._find_act_by_name("trigram") is not None
        has_fourgram = self._find_act_by_name("fourgram") is not None
        prefill = self.config.mode == "demo"
        learnable = self.config.mode == "pure"

        if step >= 10_000 and (not has_bigram_space and not has_bigram_nonspace and not has_bigram_merged):
            evict = "count_lex" if self.config.mode == "pure" else "fifo"
            if prefill:
                t_space = build_ngram_table(
                    tokens, n=2, prev_is_space=True, max_next=64, min_ctx_total=2
                )
                t_non = build_ngram_table(
                    tokens, n=2, prev_is_space=False, max_next=64, min_ctx_total=2
                )
            else:
                t_space = {}
                t_non = {}
            a1 = _make_bigram_space_act(
                act_id=det_act_id(step=step, name="bigram_space", idx=0),
                table=t_space,
                created_at=deterministic_iso(step=step, offset_us=0),
                allow_new_contexts=learnable,
                allow_new_tokens=learnable,
                max_contexts=20_000,
                max_next_per_ctx=32,
                evict_policy=evict,
            ).to_dict()
            a2 = _make_bigram_nonspace_act(
                act_id=det_act_id(step=step, name="bigram_nonspace", idx=0),
                table=t_non,
                created_at=deterministic_iso(step=step, offset_us=1),
                allow_new_contexts=learnable,
                allow_new_tokens=learnable,
                max_contexts=20_000,
                max_next_per_ctx=32,
                evict_policy=evict,
            ).to_dict()
            if self.config.mode == "pure":
                if a1.get("evidence", {}).get("table"):
                    raise RuntimeError("PURE violation: bigram_space prefill detected")
                if a2.get("evidence", {}).get("table"):
                    raise RuntimeError("PURE violation: bigram_nonspace prefill detected")
            patches.append(Patch(kind="ADD_ACT", payload={"acts": [a1, a2]}))

        if step >= 40_000 and (not has_trigram):
            evict = "count_lex" if self.config.mode == "pure" else "fifo"
            if prefill:
                t_tri = build_ngram_table(tokens, n=3, max_next=64, min_ctx_total=2)
            else:
                t_tri = {}
            a = _make_trigram_act(
                act_id=det_act_id(step=step, name="trigram", idx=0),
                table=t_tri,
                created_at=deterministic_iso(step=step, offset_us=0),
                allow_new_contexts=learnable,
                allow_new_tokens=learnable,
                max_contexts=30_000,
                max_next_per_ctx=24,
                evict_policy=evict,
            ).to_dict()
            if self.config.mode == "pure" and a.get("evidence", {}).get("table"):
                raise RuntimeError("PURE violation: trigram prefill detected")
            patches.append(Patch(kind="ADD_ACT", payload={"acts": [a]}))

        if step >= 80_000 and (has_bigram_space and has_bigram_nonspace and not has_bigram_merged):
            a_space = self._find_act_by_name("bigram_space")
            a_non = self._find_act_by_name("bigram_nonspace")
            if a_space and a_non:
                if self.config.mode == "pure":
                    # PURE: merge via explicit evidence transfer (union-sum), without prefill.
                    max_ctx = int(a_space.evidence.get("max_contexts", 0) or 0) + int(
                        a_non.evidence.get("max_contexts", 0) or 0
                    )
                    max_next = max(
                        int(a_space.evidence.get("max_next_per_ctx", 0) or 0),
                        int(a_non.evidence.get("max_next_per_ctx", 0) or 0),
                    )
                    patches.append(
                        Patch(
                            kind="MERGE_ACTS",
                            payload={
                                "a_id": a_space.id,
                                "b_id": a_non.id,
                                "keep_id": a_space.id,
                                "step": int(step),
                                "policy": {
                                    "merged_name": "bigram_merged",
                                    "match": {"type": "ngram", "n": 2},
                                    "allow_new_contexts": True,
                                    "allow_new_tokens": True,
                                    "max_contexts": int(max_ctx),
                                    "max_next_per_ctx": int(max_next),
                                    "evict_policy": "count_lex",
                                },
                            },
                        )
                    )
                else:
                    # DEMO: merge by materializing a combined table (evidence transfer).
                    t = {}
                    t.update(a_space.evidence.get("table", {}))
                    t.update(a_non.evidence.get("table", {}))
                    merged_act = _make_bigram_merged_act(
                        act_id=det_act_id(step=step, name="bigram_merged", idx=0),
                        table=t,
                        deps=[a_space.id, a_non.id],
                        created_at=deterministic_iso(step=step, offset_us=0),
                        allow_new_contexts=False,
                        allow_new_tokens=False,
                        max_contexts=40_000,
                        max_next_per_ctx=32,
                        evict_policy="fifo",
                    )
                    patches.append(
                        Patch(
                            kind="MERGE_ACTS",
                            payload={"add": merged_act.to_dict(), "prune": [a_space.id, a_non.id]},
                        )
                    )

        if step >= 120_000 and (not has_fourgram):
            evict = "count_lex" if self.config.mode == "pure" else "fifo"
            if prefill:
                t_4 = build_ngram_table(tokens, n=4, max_next=48, min_ctx_total=2)
            else:
                t_4 = {}
            a = _make_fourgram_act(
                act_id=det_act_id(step=step, name="fourgram", idx=0),
                table=t_4,
                created_at=deterministic_iso(step=step, offset_us=0),
                allow_new_contexts=learnable,
                allow_new_tokens=learnable,
                max_contexts=20_000,
                max_next_per_ctx=16,
                evict_policy=evict,
            ).to_dict()
            if self.config.mode == "pure" and a.get("evidence", {}).get("table"):
                raise RuntimeError("PURE violation: fourgram prefill detected")
            patches.append(Patch(kind="ADD_ACT", payload={"acts": [a]}))

        return patches

    def _select_patch(
        self, *, step: int, engine: Engine, tokens: Sequence[str], patches: List[Patch]
    ) -> Optional[Tuple[Patch, Dict[str, Any]]]:
        if not patches:
            return None

        start = (step * 7919) % max(1, len(tokens))
        base_eval = self._eval_online_window(
            self.store,
            tokens,
            start=start,
            length=self.config.val_tokens,
            engine_config=engine.config,
        )
        base_nll = float(base_eval["nll_bits"])
        base_cost = int(base_eval["cost_bits"])
        base_gen = dict(base_eval["gen"])

        best: Optional[Tuple[Patch, Dict[str, Any]]] = None
        best_score = float("-inf")

        def fluency_penalty(gen: Dict[str, Any]) -> float:
            return (
                float(gen["repeat3_global"])
                + float(gen["loop_rate_global"])
                + float(gen["whitespace_ratio"])
                + float(gen["duplicate_reply_rate"])
                + float(gen["most_common_reply_frac"])
                + float(gen.get("prefix_k_dup_rate", 0.0))
                + float(gen.get("template_ngram_dup_rate", 0.0))
                + float(gen.get("cross_turn_signature_repeat_rate", 0.0))
            )

        def utility_penalty(gen: Dict[str, Any]) -> float:
            try:
                pass_rate = float(gen.get("utility_pass_rate", 0.0) or 0.0)
            except Exception:
                pass_rate = 0.0
            if pass_rate != pass_rate:
                pass_rate = 0.0
            pass_rate = max(0.0, min(1.0, pass_rate))
            return 1.0 - pass_rate

        base_pen = fluency_penalty(base_gen)
        base_util_pen = utility_penalty(base_gen)
        lam = float(self.config.fluency_lambda)
        util_w = float(self.config.utility_weight)
        base_score = (-lam * base_pen) - (util_w * base_util_pen)

        for patch in patches:
            cand_eval = self._eval_online_window(
                self.store,
                tokens,
                start=start,
                length=self.config.val_tokens,
                engine_config=engine.config,
                patch=patch,
            )
            cand_nll = float(cand_eval["nll_bits"])
            cand_cost = int(cand_eval["cost_bits"])
            cand_gen = dict(cand_eval["gen"])

            # Estimate gain on a longer horizon than the validation slice.
            horizon = max(1, int(self.config.steps - step))
            base_nll_per_tok = base_nll / self.config.val_tokens
            cand_nll_per_tok = cand_nll / self.config.val_tokens
            data_gain_bits = (base_nll_per_tok - cand_nll_per_tok) * horizon
            cost_delta_bits = cand_cost - base_cost
            gain = data_gain_bits - cost_delta_bits

            cand_pen = fluency_penalty(cand_gen)
            cand_util_pen = utility_penalty(cand_gen)
            score = gain - lam * cand_pen - (util_w * cand_util_pen)

            # Accept criteria (v0.2.1):
            # - Non-merge: must improve the *relative* objective vs. baseline slice
            #   score_rel_base = -λ*base_pen
            #   score_rel_cand = gain_bits - λ*cand_pen
            # - Merge: behavior-preserving (no NLL/penalty regression on the slice).
            if patch.kind == "MERGE_ACTS":
                if cand_nll_per_tok > base_nll_per_tok + 1e-9:
                    continue
                if cand_pen > base_pen + 1e-9:
                    continue
                if gain <= 0.0:
                    continue
            else:
                if score <= base_score + 1e-9:
                    continue

            # Fluency hard caps (defensive).
            if cand_gen["loop_rate_global"] > 0.95:
                continue
            if cand_gen["repeat3_global"] > 0.95:
                continue

            if score > best_score + 1e-9:
                best_score = score
                best = (
                    patch,
                    {
                        "gain_bits": gain,
                        "score": score,
                        "base_score": base_score,
                        "score_improvement": score - base_score,
                        "fluency_lambda": lam,
                        "utility_weight": util_w,
                        "base_penalty_sum": base_pen,
                        "cand_penalty_sum": cand_pen,
                        "base_utility_penalty": base_util_pen,
                        "cand_utility_penalty": cand_util_pen,
                        "horizon_tokens": horizon,
                        "base": {"nll_bits": base_nll, "cost_bits": base_cost, **base_gen},
                        "cand": {"nll_bits": cand_nll, "cost_bits": cand_cost, **cand_gen},
                    },
                )

        return best

    def train(self) -> None:
        self._init_acts()
        tokens = self._load_tokens()

        engine = Engine(
            self.store,
            seed=self.config.seed,
            config=EngineConfig(enable_contracts=bool(self.config.enable_contracts)),
        )

        report: List[Dict[str, Any]] = []
        nll_sum_bits = 0.0
        nll_cum_bits = 0.0
        nll_ema = float("nan")
        win_t0 = time.time()

        ctx: List[str] = ["<BOS>"] * (engine.config.max_order - 1)
        for step in range(1, int(self.config.steps) + 1):
            idx = (step - 1) % len(tokens)
            if idx == 0 and step > 1:
                ctx = ["<BOS>"] * (engine.config.max_order - 1)
            tok = tokens[idx]

            lp = engine.logprob_next(context=ctx, token=tok)
            nll_bits = -lp / math.log(2)
            nll_sum_bits += nll_bits
            nll_ema = nll_bits if nll_ema != nll_ema else (0.98 * nll_ema + 0.02 * nll_bits)

            engine.observe(context=ctx, token=tok)
            ctx.append(tok)
            ctx = ctx[-(engine.config.max_order - 1) :]

            if step % self.config.window == 0:
                elapsed = max(1e-9, time.time() - win_t0)
                toks_per_s = self.config.window / elapsed
                win_t0 = time.time()

                mean_nll = nll_sum_bits / self.config.window
                nll_cum_bits += nll_sum_bits
                patch: Optional[Patch] = None
                patch_meta: Optional[Dict[str, Any]] = None
                if step % self.config.propose_every == 0:
                    proposals = self._propose_patches(step=step, tokens=tokens)
                    choice = self._select_patch(
                        step=step, engine=engine, tokens=tokens, patches=proposals
                    )
                    if choice is not None:
                        patch, patch_meta = choice
                        apply_res = self._apply_patch(patch, count=True)
                        extra = apply_res.get("meta")
                        if extra:
                            if patch_meta is None:
                                patch_meta = {}
                            patch_meta.update(dict(extra))
                        engine.rebuild_cache()

                V = max(1, len(engine.vocab()) + 1)
                baseline = step * safe_log2(V)
                cost_bits = self._model_cost_bits()
                mdl_total = nll_cum_bits + cost_bits
                mdl_net = baseline - mdl_total

                transcripts, gen = run_chat_suite(
                    engine,
                    dialogues=CHAT_DIALOGUES_20X3,
                    max_new_tokens=self.config.fluency_gen_tokens,
                    prefix_k=self.config.suite_prefix_k,
                    template_ngram_n=self.config.suite_template_ngram_n,
                    template_prefix_window=self.config.suite_template_prefix_window,
                )
                policy_meta = self._update_mode_policy_from_transcripts(
                    transcripts,
                    step=step,
                    prefix_k=self.config.suite_prefix_k,
                    template_ngram_n=self.config.suite_template_ngram_n,
                    template_prefix_window=self.config.suite_template_prefix_window,
                )
                if policy_meta and policy_meta.get("enabled"):
                    if patch_meta is None:
                        patch_meta = {}
                    patch_meta["mode_policy_update"] = dict(policy_meta)

                template_meta = self._update_anti_template_from_transcripts(
                    transcripts,
                    step=step,
                    template_ngram_n=self.config.suite_template_ngram_n,
                    template_prefix_window=self.config.suite_template_prefix_window,
                )
                if template_meta and template_meta.get("enabled"):
                    if patch_meta is None:
                        patch_meta = {}
                    patch_meta["ka_template_update"] = dict(template_meta)

                shadow_transcripts = transcripts
                if step == int(self.config.steps):
                    post_transcripts, post_gen = run_chat_suite(
                        engine,
                        dialogues=CHAT_DIALOGUES_20X3,
                        max_new_tokens=self.config.fluency_gen_tokens,
                        prefix_k=self.config.suite_prefix_k,
                        template_ngram_n=self.config.suite_template_ngram_n,
                        template_prefix_window=self.config.suite_template_prefix_window,
                    )
                    shadow_transcripts = post_transcripts
                    if patch_meta is None:
                        patch_meta = {}
                    patch_meta["post_update_suite"] = {
                        "enabled": True,
                        "trace_tokens_total_pre": int(gen.get("trace_tokens_total") or 0),
                        "trace_tokens_total_post": int(post_gen.get("trace_tokens_total") or 0),
                    }
                    gen = post_gen

                _skill_transcripts, util_metrics = run_skill_suite(
                    engine,
                    max_new_tokens=self.config.skill_suite_max_new_tokens,
                )
                util_log = dict(util_metrics)
                if step != int(self.config.steps):
                    util_log.pop("failures", None)
                for k, v in util_log.items():
                    gen[f"utility_{k}"] = v

                memory_meta = self._update_fact_memory_from_transcripts(transcripts, step=step)
                if memory_meta and memory_meta.get("enabled"):
                    if patch_meta is None:
                        patch_meta = {}
                    mem_log = dict(memory_meta)
                    if step != int(self.config.steps):
                        mem_log.pop("top_facts", None)
                    patch_meta["memory_update"] = mem_log

                macro_meta = self._update_macro_library_from_transcripts(shadow_transcripts, step=step)
                if macro_meta and macro_meta.get("enabled"):
                    if patch_meta is None:
                        patch_meta = {}
                    mac_log = dict(macro_meta)
                    if step != int(self.config.steps):
                        mac_log.pop("top_macros", None)
                        mac_log.pop("top_rewrite_rule_hits", None)
                        mac_log.pop("top_set_transitions", None)
                    patch_meta["macro_update"] = mac_log

                router_meta = self._update_macro_router_from_transcripts(shadow_transcripts, step=step)
                if router_meta and router_meta.get("enabled"):
                    if patch_meta is None:
                        patch_meta = {}
                    rlog = dict(router_meta)
                    shadow = rlog.pop("shadow_metrics", None)
                    patch_meta["router_update"] = rlog
                    if shadow is not None:
                        patch_meta["router_shadow_metrics"] = shadow

                # Snapshot the full explicit state (acts + evidence/counts) every window.
                snap_path = os.path.join(self.snapshots_dir, f"step{step:06d}_acts.jsonl")
                self.store.save_jsonl(snap_path)
                acts_hash = self.store.content_hash()

                # Ledger: WORM chain + hashes; patch is structural-only (or null).
                self.ledger.append(
                    step=step,
                    patch=patch,
                    acts_hash=acts_hash,
                    metrics={"patch_meta": patch_meta},
                    snapshot_path=os.path.relpath(snap_path, self.out_dir),
                )

                row = {
                    "step": step,
                    "nll_bits_mean": mean_nll,
                    "nll_bits_ema": nll_ema,
                    "nll_bits_cum": nll_cum_bits,
                    "mdl_total_est_bits": mdl_total,
                    "mdl_net_est_bits": mdl_net,
                    "model_cost_bits": cost_bits,
                    "vocab_size": V,
                    "mode": self.config.mode,
                    "fluency_lambda": float(self.config.fluency_lambda),
                    "num_acts": len(self.store.active()),
                    "adds": self._adds,
                    "merges": self._merges,
                    "prunes": self._prunes,
                    "patch_kind": patch.kind if patch else None,
                    "repeat3_global": gen["repeat3_global"],
                    "loop_rate_global": gen["loop_rate_global"],
                    "distinct2_global": gen["distinct2_global"],
                    "unique_reply_rate": gen["unique_reply_rate"],
                    "duplicate_reply_rate": gen["duplicate_reply_rate"],
                    "most_common_reply_frac": gen["most_common_reply_frac"],
                    "prefix_k": gen.get("prefix_k"),
                    "prefix_k_dup_rate": gen.get("prefix_k_dup_rate"),
                    "template_ngram_n": gen.get("template_ngram_n"),
                    "template_prefix_window": gen.get("template_prefix_window"),
                    "template_ngram_dup_rate": gen.get("template_ngram_dup_rate"),
                    "cross_turn_signature_repeat_rate": gen.get("cross_turn_signature_repeat_rate"),
                    "cross_turn_mode_repeat_rate": gen.get("cross_turn_mode_repeat_rate"),
                    "mode_distribution": gen.get("mode_distribution"),
                    "mode_counts": gen.get("mode_counts"),
                    "mode_source_counts": gen.get("mode_source_counts"),
                    "policy_hit_rate": gen.get("policy_hit_rate"),
                    "policy_action_counts": gen.get("policy_action_counts"),
                    "policy_explore_rate": gen.get("policy_explore_rate"),
                    "policy_exploit_rate": gen.get("policy_exploit_rate"),
                    "policy_coverage_mean": gen.get("policy_coverage_mean"),
                    "utility_weight": float(self.config.utility_weight),
                    "utility_total_tasks": gen.get("utility_total_tasks"),
                    "utility_pass_count": gen.get("utility_pass_count"),
                    "utility_pass_rate": gen.get("utility_pass_rate"),
                    "utility_instruction_pass_rate": gen.get("utility_instruction_pass_rate"),
                    "utility_json_pass_rate": gen.get("utility_json_pass_rate"),
                    "utility_math_pass_rate": gen.get("utility_math_pass_rate"),
                    "utility_state_pass_rate": gen.get("utility_state_pass_rate"),
                    "utility_plan_trace_missing_turns": gen.get(
                        "utility_plan_trace_missing_turns"
                    ),
                    "mode_policy_table_size": policy_meta.get("table_size") if policy_meta else None,
                    "mode_policy_k_user": policy_meta.get("k_user") if policy_meta else None,
                    "memory_total_facts": memory_meta.get("total_facts") if memory_meta else None,
                    "memory_added": memory_meta.get("added") if memory_meta else None,
                    "memory_updated": memory_meta.get("updated") if memory_meta else None,
                    "memory_skipped_scaffold": memory_meta.get("skipped_scaffold") if memory_meta else None,
                    "memory_skipped_conflict": memory_meta.get("skipped_conflict") if memory_meta else None,
                    "memory_skipped_invalid": memory_meta.get("skipped_invalid") if memory_meta else None,
                    "trace_tokens_total": gen.get("trace_tokens_total"),
                    "candidates_considered_per_token_mean": gen.get(
                        "candidates_considered_per_token_mean"
                    ),
                    "predictor_matched_per_token_mean": gen.get(
                        "predictor_matched_per_token_mean"
                    ),
                    "predictor_emitted_per_token_mean": gen.get(
                        "predictor_emitted_per_token_mean"
                    ),
                    "acts_considered_per_token_mean": gen.get("acts_considered_per_token_mean"),
                    "search_steps_per_turn_mean": gen.get("search_steps_per_turn_mean"),
                    "avg_trace_len_before": macro_meta.get("avg_trace_len_before")
                    if macro_meta
                    else None,
                    "avg_trace_len_after": macro_meta.get("avg_trace_len_after")
                    if macro_meta
                    else None,
                    "macro_hit_rate_seq": macro_meta.get("macro_hit_rate_seq") if macro_meta else None,
                    "macro_hit_rate_set": macro_meta.get("macro_hit_rate_set") if macro_meta else None,
                    "macro_hits_seq": macro_meta.get("macro_hits_seq") if macro_meta else None,
                    "macro_hits_set": macro_meta.get("macro_hits_set") if macro_meta else None,
                    "trace_mdl_before_bits_est": macro_meta.get("trace_mdl_before_bits_est")
                    if macro_meta
                    else None,
                    "trace_mdl_after_bits_est": macro_meta.get("trace_mdl_after_bits_est")
                    if macro_meta
                    else None,
                    "trace_mdl_gain_bits_est": macro_meta.get("trace_mdl_gain_bits_est")
                    if macro_meta
                    else None,
                    "trace_mdl_gain_real_bits_est": macro_meta.get("trace_mdl_gain_real_bits_est")
                    if macro_meta
                    else None,
                    "total_macros": macro_meta.get("total_macros") if macro_meta else None,
                    "macro_added": macro_meta.get("added") if macro_meta else None,
                    "macro_updated": macro_meta.get("updated_existing") if macro_meta else None,
                    "macro_evicted": macro_meta.get("evicted") if macro_meta else None,
                    "macro_library_cost_bits_est": macro_meta.get("macro_library_cost_bits_est")
                    if macro_meta
                    else None,
                    "depth_compositional_max": macro_meta.get("depth_compositional_max")
                    if macro_meta
                    else None,
                    "depth_compositional_median": macro_meta.get("depth_compositional_median")
                    if macro_meta
                    else None,
                    "router_predictors_evaluated_baseline_per_token_mean": (
                        router_meta.get("shadow_metrics", {}).get(
                            "predictors_evaluated_baseline_per_token_mean"
                        )
                        if router_meta
                        else None
                    ),
                    "router_predictors_evaluated_per_token_mean": (
                        router_meta.get("shadow_metrics", {}).get(
                            "predictors_evaluated_router_per_token_mean"
                        )
                        if router_meta
                        else None
                    ),
                    "router_estimated_act_evals_saved": (
                        router_meta.get("shadow_metrics", {}).get("estimated_act_evals_saved")
                        if router_meta
                        else None
                    ),
                    "router_would_skip_rate": (
                        router_meta.get("shadow_metrics", {}).get("would_skip_rate")
                        if router_meta
                        else None
                    ),
                    "router_fastpath_rate": (
                        router_meta.get("shadow_metrics", {}).get("fastpath_rate")
                        if router_meta
                        else None
                    ),
                    "router_fallback_rate": (
                        router_meta.get("shadow_metrics", {}).get("fallback_rate")
                        if router_meta
                        else None
                    ),
                    "router_coverage_rate": (
                        router_meta.get("shadow_metrics", {}).get("coverage_rate")
                        if router_meta
                        else None
                    ),
                    "avg_len_tokens": gen["avg_len_tokens"],
                    "whitespace_ratio": gen["whitespace_ratio"],
                    "tokens_per_s": toks_per_s,
                    "rss_bytes": rss_bytes_best_effort(),
                    "acts_hash": acts_hash,
                }
                if memory_meta and step == int(self.config.steps):
                    row["memory_top_facts"] = memory_meta.get("top_facts")
                report.append(row)

                # Reset window accumulators
                nll_sum_bits = 0.0
                self._adds = self._merges = self._prunes = 0

        # Final artifacts
        self.store.save_jsonl(self.acts_path)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
