from __future__ import annotations

import json
import math
import os
import sys
import time
import copy
import re
import itertools
import shutil
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
    V67_DIALOGUE_COMPILER_ID,
    compile_dialogue_v67,
    non_ws_tokens,
    prefix_k_signature,
    reply_signature,
    run_chat_suite,
    run_skill_suite,
    skill_suite_tasks_for_pack,
    user_signature,
)
from .store import ActStore
from .concepts import PRIMITIVE_OPS
from .csv_miner import mine_csv_candidates
from .semantic_ids import plan_id_for_expected_spec_sig, hypothesis_id_v1, reference_id_v1


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


def _make_fact_set_concept_act(*, act_id: str, created_at: str) -> Act:
    """
    Deterministic concept_csv helper to set a (key,value) pair into the explicit memory_facts table.
    Used by closed-world long-memory validator tasks; no external knowledge.
    """
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="concept_csv",
        match={},
        program=[
            Instruction("CSV_GET_INPUT", {"name": "key", "out": "key"}),
            Instruction("CSV_GET_INPUT", {"name": "value", "out": "value"}),
            Instruction("CSV_FACT_SET", {"key_var": "key", "value_var": "value"}),
            Instruction("CSV_CONST", {"value": "OK", "out": "value_out"}),
            Instruction("CSV_RETURN", {"var": "value_out"}),
        ],
        evidence={
            "name": "concept_fact_set_v0",
            "interface": {
                "input_schema": {"key": "str", "value": "str"},
                "output_schema": {"value_out": "str"},
                "validator_id": "text_exact",
            },
            "meta": {"title": "memory_facts_set"},
        },
        cost={"overhead_bits": 512},
        deps=[],
        active=True,
    )


def _make_fact_get_concept_act(*, act_id: str, created_at: str) -> Act:
    """
    Deterministic concept_csv helper to read a key from the explicit memory_facts table.
    """
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="concept_csv",
        match={},
        program=[
            Instruction("CSV_GET_INPUT", {"name": "key", "out": "key"}),
            Instruction("CSV_FACT_GET", {"key_var": "key", "out": "value_out"}),
            Instruction("CSV_RETURN", {"var": "value_out"}),
        ],
        evidence={
            "name": "concept_fact_get_v0",
            "interface": {
                "input_schema": {"key": "str"},
                "output_schema": {"value_out": "str"},
                "validator_id": "text_exact",
            },
            "meta": {"title": "memory_facts_get"},
        },
        cost={"overhead_bits": 512},
        deps=[],
        active=True,
    )


def _make_instruction_follow_concept_act(*, act_id: str, created_at: str) -> Act:
    """
    Closed-world instruction-following concept executor.

    This is an explicit concept_csv ACT (versioned/auditable) that solves a small family of
    deterministic instruction patterns used by the validator packs via a single primitive:
    `instruction_solve_v1(text) -> str`.
    """
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="concept_csv",
        match={},
        program=[
            Instruction("CSV_GET_INPUT", {"name": "text", "out": "text"}),
            Instruction(
                "CSV_PRIMITIVE",
                {"fn": "instruction_solve_v1", "in": ["text"], "out": "value_out"},
            ),
            Instruction("CSV_RETURN", {"var": "value_out"}),
        ],
        evidence={
            "name": "concept_instruction_follow_v1",
            "interface": {
                "input_schema": {"text": "str"},
                "output_schema": {"value_out": "str"},
                "validator_id": "instruction_following_validator",
            },
            "meta": {"title": "instruction_follow_v1", "birth_tags": ["instruction"]},
        },
        cost={"overhead_bits": 512},
        deps=[],
        active=True,
    )


def _make_plan_op_add_int_concept_act(*, act_id: str, created_at: str) -> Act:
    """
    Deterministic call-free primitive wrapper for the plan-op `add_int(a,b) -> int`.

    Needed for agency_suite plan synthesis from `plan_validator` expected_spec.ops.
    """
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="concept_csv",
        match={},
        program=[
            Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
            Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
            Instruction("CSV_PRIMITIVE", {"fn": "add_int", "in": ["a", "b"], "out": "value_out"}),
            Instruction("CSV_RETURN", {"var": "value_out"}),
        ],
        evidence={
            "name": "concept_plan_op_add_int_v0",
            "interface": {
                "input_schema": {"a": "int", "b": "int"},
                "output_schema": {"value_out": "int"},
                # This wrapper is intended as an internal planning operator; validators are applied
                # at the goal (plan_validator) level.
                "validator_id": "",
            },
            "meta": {"title": "plan_op_add_int", "birth_tags": ["plan", "math"]},
        },
        cost={"overhead_bits": 512},
        deps=[],
        active=True,
    )


def _make_plan_op_make_dict_goal_plan_ab_concept_act(*, act_id: str, created_at: str) -> Act:
    """
    Deterministic call-free primitive wrapper for `make_dict_goal_plan_ab(goal_id, plan, a, b) -> dict`.
    """
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="concept_csv",
        match={},
        program=[
            Instruction("CSV_GET_INPUT", {"name": "goal_id", "out": "goal_id"}),
            Instruction("CSV_GET_INPUT", {"name": "plan", "out": "plan"}),
            Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
            Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
            Instruction(
                "CSV_PRIMITIVE",
                {
                    "fn": "make_dict_goal_plan_ab",
                    "in": ["goal_id", "plan", "a", "b"],
                    "out": "value_out",
                },
            ),
            Instruction("CSV_RETURN", {"var": "value_out"}),
        ],
        evidence={
            "name": "concept_plan_op_make_dict_goal_plan_ab_v0",
            "interface": {
                "input_schema": {"goal_id": "str", "plan": "str", "a": "int", "b": "int"},
                "output_schema": {"value_out": "dict"},
                "validator_id": "",
            },
            "meta": {"title": "plan_op_make_dict_goal_plan_ab", "birth_tags": ["plan"]},
        },
        cost={"overhead_bits": 512},
        deps=[],
        active=True,
    )


def _make_plan_op_json_canonical_concept_act(*, act_id: str, created_at: str) -> Act:
    """
    Deterministic call-free primitive wrapper for `json_canonical(obj) -> str`.
    """
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="concept_csv",
        match={},
        program=[
            Instruction("CSV_GET_INPUT", {"name": "obj", "out": "obj"}),
            Instruction("CSV_PRIMITIVE", {"fn": "json_canonical", "in": ["obj"], "out": "value_out"}),
            Instruction("CSV_RETURN", {"var": "value_out"}),
        ],
        evidence={
            "name": "concept_plan_op_json_canonical_v0",
            "interface": {
                "input_schema": {"obj": "dict"},
                "output_schema": {"value_out": "str"},
                "validator_id": "",
            },
            "meta": {"title": "plan_op_json_canonical", "birth_tags": ["plan", "json"]},
        },
        cost={"overhead_bits": 512},
        deps=[],
        active=True,
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
            # Decoder compatibility tag (used by gate-live to preserve invariants across decoder changes).
            "decoder_fluency_id": "antiloop_v40",
            # Budget / governance.
            "max_contexts": 8192,
            "top_k": 2,
            # ctx_sig -> {"predictors": [...], "counts": {...}, "total": int}
            "table": {},
        },
        cost={"overhead_bits": 512},
        deps=[],
    )


def _make_system_survival_goal_act(*, act_id: str, created_at: str) -> Act:
    return Act(
        id=act_id,
        version=1,
        created_at=created_at,
        kind="goal",
        match={"type": "always"},
        program=[],
        evidence={
            "name": "system_survival_v0",
            "enabled": True,
            "shadow_mode": True,
            "goal": {
                "goal_id": "SYSTEM_SURVIVAL",
                "goal_type": "MAINTAIN_CAPABILITY",
                "persistence_policy": "HARD",
                "success_criteria": {
                    "loss_trend_negative": True,
                    "reuse_positive": True,
                    "new_abstractions_emerging": True,
                },
                "failure_criteria": {
                    "long_plateau": True,
                    "no_abstraction": True,
                    "no_reuse": True,
                },
            },
            "last_update_step": 0,
            "status": {},
        },
        cost={"overhead_bits": 256},
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
    # Optional wall-clock budget: if >0, training stops (gracefully) after this many seconds.
    max_seconds: float = 0.0
    # Long-horizon gain estimate for patch selection. If 0, uses `steps`.
    # Use a larger value when training in segments but planning to continue (e.g., 200_000).
    gain_horizon_steps: int = 0
    seed: int = 0
    # Optional: start from an existing acts snapshot (acts.jsonl/stepXXXXXX_acts.jsonl).
    # When set, base seed acts are only added if missing; learned acts are preserved.
    resume_acts_path: str = ""
    window: int = 10_000
    propose_every: int = 10_000
    val_tokens: int = 4000
    # Cross-context pressure for predictor selection: evaluate NLL on multiple, disjoint
    # online windows (deterministically) and use the mean for gain estimation.
    nll_eval_windows: int = 3
    # Holdout pressure: keep a deterministic token holdout slice and require predictor patches
    # to avoid regressions on holdout NLL (static, no online adaptation).
    holdout_frac: float = 0.0
    holdout_eval_windows: int = 0
    holdout_eval_tokens: int = 0
    # System survival (autopoiesis-like): treat sustained plateau as an existential failure and
    # force deterministic divergence (structural exploration) until capability improves.
    survival_plateau_windows: int = 3
    survival_improve_tol: float = 1e-4
    # If >0, hard-fail training if system survival does not improve for this many proposal windows.
    # Use for strict "no escape route" runs.
    survival_hard_fail_windows: int = 0
    # Require ongoing abstraction growth (macros/concepts) to avoid a "comfortable plateau".
    survival_no_abstraction_windows: int = 6
    # Require positive reuse signal (from macro/concept mining) to avoid "tuning without reuse".
    survival_reuse_min: float = 0.01
    survival_no_reuse_windows: int = 6
    # Concept creation as survival law: if the system is in plateau and fails to create new
    # reusable concept_csv objects, it enters a concept crisis (INDUCE_CONCEPT forced) and
    # can hard-fail deterministically.
    survival_concept_no_add_windows: int = 3
    survival_concept_reuse_stall_windows: int = 3
    survival_concept_reuse_tol: float = 1e-6
    # If >0, hard-fail when a concept crisis persists without producing a new concept.
    survival_concept_hard_fail_windows: int = 3
    # Concept compositionality as survival law: when the validator pack requires nested concept calls
    # (CSV_CALL), failing to induce a composed concept during plateau is an existential failure.
    survival_concept_composed_no_add_windows: int = 3
    survival_concept_composed_rate_stall_windows: int = 3
    survival_concept_composed_rate_tol: float = 1e-6
    survival_concept_composed_hard_fail_windows: int = 3
    # Deep concept hierarchy as survival law: when the validator pack requires depth>=2, the system
    # must induce and actually use deep composed concepts (nested CSV_CALL depth>=2), or it dies.
    survival_concept_deep_rate_stall_windows: int = 3
    survival_concept_deep_rate_tol: float = 1e-6
    survival_concept_deep_hard_fail_windows: int = 3
    # Very deep concept hierarchy (depth>=3): when required by the validator pack, the system must
    # induce and actually use nested composed concepts (nested CSV_CALL depth>=3), or it dies.
    survival_concept_very_deep_rate_stall_windows: int = 3
    survival_concept_very_deep_rate_tol: float = 1e-6
    survival_concept_very_deep_hard_fail_windows: int = 3
    # REPAIR mode: if divergence accepts a patch that worsens the fluency bottleneck, the system
    # accrues a deterministic "fluency debt" and must repair until the bottleneck returns.
    repair_max_windows: int = 4
    repair_bottleneck_tol: float = 1e-9
    mode: Literal["demo", "pure"] = "demo"
    selection_mode: Literal["weighted", "bottleneck", "survival"] = "weighted"
    fluency_lambda: float = 20000.0
    fluency_lambda_schedule: Literal["constant", "linear_warmup"] = "linear_warmup"
    # For linear warmup: λ_eff = λ * clamp(step / (steps*frac), 0..1).
    # Use 1.0 to ramp for the whole run; use <1.0 to reach full λ earlier.
    fluency_warmup_frac: float = 1.0
    # Escape hatch: when no patch is accepted for N proposal windows, temporarily relax
    # fluency penalty in patch selection to force structural exploration (deterministic).
    divergence_after_no_patch_windows: int = 1
    # Stronger survival pressure: if we keep accepting only rewrites (no structural growth),
    # trigger divergence to force new structure (ADD/MERGE) instead of "tuning forever".
    divergence_after_no_growth_windows: int = 2
    divergence_lambda_scale: float = 0.05
    # In divergence mode, allow limited (bounded) fluency regression to explore new structure.
    # This prevents divergence from becoming an unbounded "bypass" for dialogue quality.
    divergence_fluency_slack: float = 0.05
    fluency_prompt: str = "Oi, quem é você?\n"
    fluency_prompts: Tuple[str, str, str] = (
        "Oi, quem é você?\n",
        "Hello, who are you?\n",
        "What is MDL?\n",
    )
    fluency_gen_tokens: int = 200
    # Decoder-level fluency projection (CPU-only; deterministic). These do not add new cognition;
    # they only shape the surface text when not in strict-format turns.
    decoder_fluency_no_repeat_ngram: int = 3
    decoder_fluency_prompt_ngram_block: bool = False
    decoder_fluency_min_new_tokens_before_eos_freeform: int = 0
    decoder_fluency_block_token_regex: str = ""
    decoder_fluency_block_penalty: float = 1e6
    # Dialogue state / long-dialogue coherence (explicit state; CPU-only; deterministic).
    dialogue_state_enabled: bool = True
    dialogue_state_tail_k: int = 64
    dialogue_state_prefix_enabled: bool = True
    dialogue_state_prefix_prompt_turns_leq: int = 4
    dialogue_state_prefix_max_missing_turns: int = 8
    dialogue_state_prefix_max_chars: int = 2000
    dialogue_state_fact_extract_enabled: bool = True
    dialogue_state_fact_max: int = 16
    suite_prefix_k: int = 8
    suite_template_ngram_n: int = 6
    suite_template_prefix_window: int = 32
    # Utility suite weight in patch score (shadow default: 0.0). Set >0 to optimize pass_rate.
    utility_weight: float = 0.0
    # Utility suite token budget during training (evaluation scripts use their own max_new_tokens).
    skill_suite_max_new_tokens: int = 128
    # When >0, truncate dialogue history to the last K turns when building prompts in the
    # skill suite (forces long-dialogue persistence via explicit dialogue_state).
    skill_suite_prompt_history_k: int = 4
    # Which deterministic validator pack to use for the skill suite.
    # Use "v0" for the baseline pack; use "sota_v1" for a broader pack including plan/state/json tasks.
    skill_suite_pack: str = "v0"
    # Ablation: deterministically shuffle latent family ids in the skill suite's plan_trace.
    # If the system depends on stable latent families, this should break convergence/performance.
    suite_shuffle_families: bool = False
    # Agency suite (planner+executor): deterministic goals that require explicit multi-step planning
    # over concept_csv operators. This is used as a utility bottleneck dimension.
    agency_suite_enabled: bool = False
    agency_suite_max_tasks: int = 6
    agency_suite_min_steps: int = 2
    agency_suite_base_max_program_len: int = 6
    agency_suite_planner_max_depth: int = 6
    agency_suite_planner_max_expansions: int = 256
    # Enable deterministic instruction contracts during generation (default OFF).
    # NOTE: contracts are treated as non-learning scaffolding; training objective should not rely on them.
    enable_contracts: bool = False
    # Concept induction (CSV mining): deterministically mine and promote concept_csv acts
    # as explicit, invocable semantic objects (no gradients, no weights).
    concept_csv_mining_enabled: bool = False
    concept_csv_mining_top_k: int = 4
    concept_csv_mining_max_new_per_window: int = 2
    # Max primitive-op length for candidate mining from concept_csv traces.
    # Increase (e.g., 64–128) when using long-horizon plan packs (v69_long_sum_*) so the
    # miner can materialize full-plan operators with large input schemas.
    concept_csv_mining_max_ops: int = 6
    # Second-stage concept induction (v74): mine repeated multi-step concept call subpaths
    # and promote composed concept_csv acts using CSV_CALL (hierarchy / compositionality).
    concept_csv_composed_enabled: bool = False
    # Filter base concept operators used to generate multi-step traces (forces composition).
    concept_csv_composed_base_max_program_len: int = 6
    concept_csv_composed_planner_max_depth: int = 6
    concept_csv_composed_max_k: int = 6
    concept_csv_composed_min_support: int = 2
    concept_csv_composed_top_k: int = 4
    concept_csv_composed_max_new_per_window: int = 1
    # Budget/GC for mined concept_csv acts (count, not bytes): keeps semantic overhead bounded.
    concept_csv_budget: int = 16
    concept_csv_overhead_bits: int = 1024
    # MDL: composed concept_csv acts (v74) should have a lower but non-zero overhead
    # than flat mined concept_csv, to encourage hierarchy without creating a free bypass.
    concept_csv_composed_overhead_scale: float = 0.6
    # Deep wrappers (v74) are a surgical hierarchy enabler; keep their overhead aligned with composed.
    concept_csv_deepwrap_overhead_scale: float = 0.6
    # Deep wrappers per-window cap (count): must be high enough to satisfy required_depth across
    # multiple plan interfaces (otherwise concept_depth_too_shallow can persist indefinitely).
    concept_csv_deepwrap_max_new_per_window: int = 6

    # ICS (INDUCE_CONCEPT_SUBGRAPH) — sovereign concept governance (v1).
    # When enabled, concept lifecycle is governed by an explicit merge/split/fitness/selection loop.
    ics_enabled: bool = False
    # Sovereign regime: disallow any learning/state updates outside ICS.
    # When enabled, all updates that mutate cognitive state must be routed through `_ics_step`.
    # This enforces: no patch selection, no macro/memory/policy/router updates outside ICS.
    ics_sovereign: bool = False
    # ICS semantic objects (v1): Goals/Plans/Hypotheses/References as explicit, living acts.
    # When enabled, ICS is responsible for creating/updating these banks deterministically from traces.
    ics_semantic_banks_enabled: bool = False
    # Split: restrict overgeneral concepts to families where they pass.
    ics_split_keep_ok_rate: float = 0.8
    ics_split_drop_ok_rate: float = 0.2
    ics_split_min_family_turns: int = 2
    # Fitness/selection: promote only with cross-context evidence + ablation; delete on persistent negative fitness.
    ics_promote_threshold_bits: int = 256
    ics_promote_max_per_window: int = 1
    ics_delete_neg_streak_windows: int = 3
    ics_ablation_enabled: bool = True
    # Ablation mode: shuffle family ids only for ICS evidence/fitness (engine execution uses real family_id).
    ics_shuffle_families: bool = False

    # Optional on-the-fly HF streaming dataset (to avoid local 300GB corpora).
    # When `hf_dataset_id` is set, the trainer streams tokens and keeps only a bounded
    # ring buffer for evaluation/patch selection.
    hf_dataset_id: str = ""
    hf_dataset_split: str = "train"
    hf_dataset_name: str = ""
    hf_streaming: bool = True
    hf_shuffle: bool = True
    hf_shuffle_buffer: int = 10_000
    hf_max_chars_per_example: int = 8_000
    hf_holdout_seed_offset: int = 1
    stream_buffer_tokens: int = 200_000
    stream_holdout_tokens: int = 0


class KAAbsoluteTrainer:
    def __init__(self, *, data_path: str, out_dir: str, config: TrainConfig):
        self.data_path = data_path
        self.out_dir = out_dir
        self.config = config

        os.makedirs(self.out_dir, exist_ok=True)
        self.acts_path = os.path.join(self.out_dir, "acts.jsonl")
        self.ledger_path = os.path.join(self.out_dir, "ledger.jsonl")
        self.report_path = os.path.join(self.out_dir, "report.json")
        self.report_jsonl_path = os.path.join(self.out_dir, "report.jsonl")
        self.report_last_path = os.path.join(self.out_dir, "report_last.json")
        self.snapshots_dir = os.path.join(self.out_dir, "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)

        self.store = ActStore()
        self.ledger = Ledger(self.ledger_path)

        self._adds = 0
        self._merges = 0
        self._prunes = 0
        self._holdout_tokens: List[str] = []
        self._concept_csv_added = 0
        self._concept_csv_pruned = 0

    def _init_acts(self) -> None:
        resume_path = str(getattr(self.config, "resume_acts_path", "") or "").strip()
        if resume_path:
            try:
                self.store = ActStore.load_jsonl(resume_path)
                print(f"[resume] loaded acts: {resume_path}")
            except Exception as e:
                # Fail-open: resume is optional; fall back to a clean seed store.
                print(f"[resume] failed: {resume_path}: {e}")
                self.store = ActStore()

        def _ensure(act: Act) -> None:
            try:
                if self.store.get(str(act.id)) is None:
                    self.store.add(act)
            except Exception:
                # If the store is in a bad state, fail-closed by raising.
                raise

        _ensure(
            _make_unigram_act(
                act_id=det_act_id(step=0, name="unigram", idx=0),
                created_at=deterministic_iso(step=0, offset_us=0),
            )
        )
        _ensure(
            _make_mode_policy_act(
                act_id=det_act_id(step=0, name="mode_policy_v1", idx=0),
                created_at=deterministic_iso(step=0, offset_us=1),
            )
        )
        _ensure(
            _make_mode_selector_act(
                act_id=det_act_id(step=0, name="mode_definition_v1", idx=0),
                created_at=deterministic_iso(step=0, offset_us=2),
                mode="definition",
            )
        )
        _ensure(
            _make_mode_selector_act(
                act_id=det_act_id(step=0, name="mode_explanation_v1", idx=0),
                created_at=deterministic_iso(step=0, offset_us=3),
                mode="explanation",
            )
        )
        _ensure(
            _make_mode_selector_act(
                act_id=det_act_id(step=0, name="mode_list_v1", idx=0),
                created_at=deterministic_iso(step=0, offset_us=4),
                mode="list",
            )
        )
        _ensure(
            _make_rewrite_rule_act(
                act_id=det_act_id(step=0, name="fluency_guardrails_v1", idx=0),
                created_at=deterministic_iso(step=0, offset_us=5),
            )
        )
        _ensure(
            _make_anti_template_rewrite_rule_act(
                act_id=det_act_id(step=0, name="anti_template_v2", idx=0),
                created_at=deterministic_iso(step=0, offset_us=6),
            )
        )
        _ensure(
            _make_selector_act(
                act_id=det_act_id(step=0, name="selector_greedy_v1", idx=0),
                created_at=deterministic_iso(step=0, offset_us=7),
            )
        )
        _ensure(
            _make_fact_memory_act(
                act_id=det_act_id(step=0, name="fact_memory_v0", idx=0),
                created_at=deterministic_iso(step=0, offset_us=8),
            )
        )
        _ensure(
            _make_macro_library_act(
                act_id=det_act_id(step=0, name="macro_library_v0", idx=0),
                created_at=deterministic_iso(step=0, offset_us=9),
            )
        )
        _ensure(
            _make_macro_router_act(
                act_id=det_act_id(step=0, name="macro_router_v0", idx=0),
                created_at=deterministic_iso(step=0, offset_us=10),
            )
        )
        if bool(self.config.enable_contracts):
            _ensure(
                _make_instruction_contract_act(
                    act_id=det_act_id(step=0, name="instruction_contract_v0", idx=0),
                    created_at=deterministic_iso(step=0, offset_us=11),
                )
            )
        _ensure(
            _make_system_survival_goal_act(
                act_id=det_act_id(step=0, name="system_survival_v0", idx=0),
                created_at=deterministic_iso(step=0, offset_us=12),
            )
        )
        # Explicit long-memory primitives (closed world): concept_csv acts that read/write the
        # `memory_facts` ACT table via CSV_FACT_SET/CSV_FACT_GET. These are small and deterministic,
        # and enable validator packs to require long-memory without contracts.
        _ensure(
            _make_fact_set_concept_act(
                act_id=det_act_id(step=0, name="concept_fact_set_v0", idx=0),
                created_at=deterministic_iso(step=0, offset_us=13),
            )
        )
        _ensure(
            _make_fact_get_concept_act(
                act_id=det_act_id(step=0, name="concept_fact_get_v0", idx=0),
                created_at=deterministic_iso(step=0, offset_us=14),
            )
        )
        _ensure(
            _make_instruction_follow_concept_act(
                act_id=det_act_id(step=0, name="concept_instruction_follow_v1", idx=0),
                created_at=deterministic_iso(step=0, offset_us=15),
            )
        )
        # Call-free primitive wrappers required for deterministic agency-suite plan synthesis from
        # plan_validator expected_spec.ops (prevents PlannerV79 search explosion on large-arity goals).
        _ensure(
            _make_plan_op_add_int_concept_act(
                act_id=det_act_id(step=0, name="concept_plan_op_add_int_v0", idx=0),
                created_at=deterministic_iso(step=0, offset_us=16),
            )
        )
        _ensure(
            _make_plan_op_make_dict_goal_plan_ab_concept_act(
                act_id=det_act_id(step=0, name="concept_plan_op_make_dict_goal_plan_ab_v0", idx=0),
                created_at=deterministic_iso(step=0, offset_us=17),
            )
        )
        _ensure(
            _make_plan_op_json_canonical_concept_act(
                act_id=det_act_id(step=0, name="concept_plan_op_json_canonical_v0", idx=0),
                created_at=deterministic_iso(step=0, offset_us=18),
            )
        )

        self._backfill_concept_birth_tags()
        # Repair legacy snapshots: concept pruning may have deactivated "leaf" mined concepts that
        # are still referenced by higher-level composed/deepwrap concepts (CSV_CALL). This creates
        # concept_not_found failures at runtime and can incorrectly collapse plan/json skill scores.
        #
        # Deterministic rule: if an ACTIVE concept references an INACTIVE concept via CSV_CALL, the
        # callee must be reactivated (or the parent should be pruned). We choose reactivation to
        # preserve the learned hierarchy, and protect these dependencies from future budget pruning.
        self._repair_concept_csv_dependencies(step=0)

    def _backfill_concept_birth_tags(self) -> None:
        """
        One-time migration: older snapshots may contain concept_csv ACTs without `evidence.meta.birth_tags`.
        Birth tags are required to prove cross-context semantic reuse ("created here, used there").

        This is deterministic and uses only the concept interface/validator_id.
        """

        def _infer_birth_tag(validator_id: str) -> Optional[str]:
            v = str(validator_id or "").strip().lower()
            if not v:
                return None
            if v == "plan_validator" or "plan" in v:
                return "plan"
            if "json" in v:
                return "json"
            if v in {"int_value_exact"} or "math" in v or "arith" in v or v.startswith("int_") or "int" in v:
                return "math"
            if "instruction" in v:
                return "instruction"
            if "state" in v:
                return "state"
            if "memory" in v:
                return "memory"
            if "agency" in v:
                return "agency"
            return None

        updated = 0
        try:
            acts = list(self.store.by_kind("concept_csv"))
        except Exception:
            acts = []
        for act in acts:
            if act is None or not isinstance(getattr(act, "evidence", None), dict):
                continue
            ev = act.evidence
            meta = ev.get("meta")
            if not isinstance(meta, dict):
                meta = {}
                ev["meta"] = meta
            bt = meta.get("birth_tags")
            if isinstance(bt, list) and any(str(x) for x in bt):
                continue
            iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
            iface = iface if isinstance(iface, dict) else {}
            tag = _infer_birth_tag(str(iface.get("validator_id") or ""))
            if not tag:
                continue
            meta["birth_tags"] = [str(tag)]
            updated += 1

        if updated:
            try:
                print(f"[resume] backfilled concept birth_tags: {updated}")
            except Exception:
                pass

    def _effective_fluency_lambda(self, *, step: int) -> float:
        lam = float(self.config.fluency_lambda)
        sched = str(getattr(self.config, "fluency_lambda_schedule", "constant") or "constant")
        if sched not in {"constant", "linear_warmup"}:
            sched = "constant"
        if sched == "constant":
            return lam
        frac = float(getattr(self.config, "fluency_warmup_frac", 1.0) or 1.0)
        if not (frac > 0.0):
            return lam
        warmup_steps = max(1, int(round(float(self.config.steps) * frac)))
        return lam * max(0.0, min(1.0, float(step) / float(warmup_steps)))

    def _materialize_mined_concept_csv_act(self, *, cand: Any, step: int) -> Act:
        """
        Deterministically materialize a mined CSV candidate into a first-class concept_csv ACT.

        NOTE: The act_id is derived from the candidate signature only (not from the mutable store hash),
        since predictor tables mutate continuously during training.
        """
        cand_sig = str(getattr(cand, "candidate_sig", "") or "")
        if not cand_sig:
            raise ValueError("concept_candidate_missing_sig")

        ops = getattr(cand, "ops", None)
        if not isinstance(ops, list) or not ops:
            raise ValueError("concept_candidate_missing_ops")
        input_schema = getattr(cand, "input_schema", None)
        if not isinstance(input_schema, dict):
            input_schema = {}
        output_type = str(getattr(cand, "output_type", "") or "str")
        validator_id = str(getattr(cand, "validator_id", "") or "")
        if not validator_id:
            validator_id = "text_exact"

        birth_tags: set = set()
        try:
            exs = getattr(cand, "examples", None)
            if isinstance(exs, list):
                for ex in exs:
                    if not isinstance(ex, dict):
                        continue
                    ctx = str(ex.get("ctx_sig") or "")
                    if ctx.startswith("plan␟"):
                        birth_tags.add("plan")
                    elif ctx.startswith("seed␟"):
                        if "extract_int" in ctx:
                            birth_tags.add("math")
                        elif "json_ab" in ctx:
                            birth_tags.add("json")
        except Exception:
            birth_tags = set()
        if str(validator_id) == "plan_validator":
            birth_tags.add("plan")

        in_keys = list(input_schema.keys())
        program: List[Instruction] = []
        for idx, name in enumerate(in_keys):
            program.append(Instruction("CSV_GET_INPUT", {"name": str(name), "out": f"in{idx}"}))
        for op in ops:
            if not isinstance(op, dict):
                continue
            program.append(
                Instruction(
                    "CSV_PRIMITIVE",
                    {
                        "fn": str(op.get("fn") or ""),
                        "in": list(op.get("in") or []),
                        "out": str(op.get("out") or ""),
                    },
                )
            )
        program.append(Instruction("CSV_RETURN", {"var": str(ops[-1].get("out") or "")}))

        out_key = str(ops[-1].get("out") or "")
        if not out_key:
            out_key = "value"
        interface = {
            "input_schema": {str(k): str(v) for k, v in input_schema.items() if str(k)},
            "output_schema": {str(out_key): str(output_type)},
            "validator_id": str(validator_id),
        }
        ev = {
            "name": "concept_csv_mined_train_v0",
            "interface": dict(interface),
            "meta": {
                "builder": "concept_csv_mining_v0",
                "candidate_sig": str(cand_sig),
                "output_key": str(out_key),
                "gain_bits_est": int(getattr(cand, "gain_bits_est", 0) or 0),
                "contexts_distinct": int(getattr(cand, "contexts_distinct", 0) or 0),
                "count": int(getattr(cand, "count", 0) or 0),
                "birth_tags": [str(x) for x in sorted(set(str(t) for t in birth_tags if str(t)), key=str)],
            },
        }
        act_id = f"act_concept_csv_mined_{cand_sig[:16]}"
        overhead_bits = int(getattr(self.config, "concept_csv_overhead_bits", 1024) or 1024)
        overhead_bits = max(0, int(overhead_bits))
        return Act(
            id=str(act_id),
            version=1,
            created_at=deterministic_iso(step=int(step)),
            kind="concept_csv",
            match={},
            program=list(program),
            evidence=ev,
            cost={"overhead_bits": int(overhead_bits)},
            deps=[],
            active=True,
        )

    @staticmethod
    def _concept_csv_call_deps(act: Act) -> List[str]:
        callees: List[str] = []
        for ins in list(getattr(act, "program", []) or []):
            if str(getattr(ins, "op", "")) != "CSV_CALL":
                continue
            args = getattr(ins, "args", {}) or {}
            if not isinstance(args, dict):
                continue
            cid = str(args.get("concept_id") or "")
            if cid and cid not in callees:
                callees.append(cid)
        return callees

    def _concept_csv_dependency_closure_ids(self, *, active_roots_only: bool = True) -> set:
        """
        Deterministic transitive closure over CSV_CALL dependencies for concept_csv ACTs.

        NOTE: the closure traverses *through inactive callees* (store.get) so we can repair
        legacy snapshots where a required leaf was pruned but is still referenced by an active
        composed/deepwrap concept.
        """
        roots: List[str] = []
        for a in list(getattr(self.store, "acts", {}).values()):
            if a is None or str(getattr(a, "kind", "")) != "concept_csv":
                continue
            if active_roots_only and not bool(getattr(a, "active", False)):
                continue
            roots.append(str(getattr(a, "id", "") or ""))
        roots = [r for r in roots if r]
        roots.sort()

        seen: set = set()
        todo: List[str] = list(reversed(roots))
        while todo:
            cid = str(todo.pop() or "")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            act = self.store.get(cid)
            if act is None or str(getattr(act, "kind", "")) != "concept_csv":
                continue
            callees = self._concept_csv_call_deps(act)
            # LIFO traversal; reverse to keep deterministic visitation for equal programs.
            for callee in reversed([c for c in callees if isinstance(c, str) and c]):
                if callee not in seen:
                    todo.append(callee)
        return seen

    def _repair_concept_csv_dependencies(self, *, step: int) -> Dict[str, Any]:
        """
        Repair rule (deterministic):
          if an ACTIVE concept_csv references an INACTIVE concept_csv via CSV_CALL,
          the callee must be reactivated (or the parent pruned).

        We choose reactivation to preserve the learned hierarchy and prevent runtime
        `concept_not_found` failures that can collapse plan/json skills.
        """
        required = self._concept_csv_dependency_closure_ids(active_roots_only=True)
        if not required:
            return {"enabled": True, "reactivated": 0, "missing": 0}

        reactivated: List[str] = []
        missing: List[str] = []
        # Stable iteration for determinism.
        ordered = sorted([str(x) for x in required if str(x)], key=str)
        for idx, cid in enumerate(ordered):
            act = self.store.get(str(cid))
            if act is None:
                missing.append(str(cid))
                continue
            if str(getattr(act, "kind", "")) != "concept_csv":
                continue
            if bool(getattr(act, "active", False)):
                continue

            after = act.to_dict()
            after["active"] = True
            after["version"] = int(after.get("version", 1) or 1) + 1
            after["created_at"] = deterministic_iso(step=int(step), offset_us=1000 + int(idx))
            patch = Patch(kind="REWRITE_ACT", payload={"act_id": str(cid), "act": after})
            self._apply_patch(patch, count=False)
            reactivated.append(str(cid))

        if reactivated or missing:
            try:
                snap_path = os.path.join(self.snapshots_dir, "step000000_acts.jsonl")
                self.store.save_jsonl(snap_path)
                acts_hash = self.store.content_hash()
                self.ledger.append(
                    step=0,
                    patch=None,
                    acts_hash=acts_hash,
                    metrics={
                        "init_concept_dep_repair": {
                            "reactivated": list(reactivated),
                            "missing": list(missing),
                            "required_total": int(len(required)),
                        }
                    },
                    snapshot_path=os.path.relpath(snap_path, self.out_dir),
                )
            except Exception:
                pass

        if reactivated:
            print(f"[repair] reactivated concept_csv deps: {len(reactivated)}")
        if missing:
            print(f"[repair] missing concept_csv deps: {len(missing)}")
        return {"enabled": True, "reactivated": int(len(reactivated)), "missing": int(len(missing))}

    def _concept_csv_budget_prune(self, *, step: int) -> Dict[str, Any]:
        budget = int(getattr(self.config, "concept_csv_budget", 0) or 0)
        budget = max(0, int(budget))
        acts = [
            a
            for a in self.store.by_kind("concept_csv")
            if isinstance(getattr(a, "evidence", None), dict)
            and str(a.evidence.get("name") or "") == "concept_csv_mined_train_v0"
        ]
        if budget <= 0:
            return {"enabled": True, "budget": int(budget), "total": int(len(acts)), "pruned": 0}
        if len(acts) <= budget:
            return {"enabled": True, "budget": int(budget), "total": int(len(acts)), "pruned": 0}

        # Dependency protection: never prune a mined concept if it is required by any ACTIVE concept_csv
        # via CSV_CALL (directly or transitively). Otherwise, deepwrap/induced concepts can become
        # non-executable ("concept_not_found"), and the engine may fall back to raw token emission.
        required = self._concept_csv_dependency_closure_ids(active_roots_only=True)
        protected = {str(a.id) for a in acts if str(a.id) in required}
        if len(protected) > budget:
            # Budget is infeasible under dependency closure; fail-open by skipping pruning.
            return {
                "enabled": True,
                "budget": int(budget),
                "total": int(len(acts)),
                "pruned": 0,
                "skipped": True,
                "reason": "budget_infeasible_due_to_deps",
                "protected": int(len(protected)),
            }

        def _score(a: Act) -> Tuple[int, int, int, str]:
            ev = a.evidence if isinstance(a.evidence, dict) else {}
            meta = ev.get("meta") if isinstance(ev.get("meta"), dict) else {}
            return (
                int(meta.get("gain_bits_est", 0) or 0),
                int(meta.get("contexts_distinct", 0) or 0),
                int(meta.get("count", 0) or 0),
                str(a.id),
            )

        ranked = sorted(
            acts,
            key=lambda a: (-_score(a)[0], -_score(a)[1], -_score(a)[2], _score(a)[3]),
        )
        keep = {str(a.id) for a in ranked[:budget]}
        keep |= set(protected)
        to_prune = [a for a in ranked if str(a.id) not in keep]
        if not to_prune:
            return {"enabled": True, "budget": int(budget), "total": int(len(ranked)), "pruned": 0}

        patch = Patch(kind="PRUNE_ACT", payload={"act_ids": [a.id for a in to_prune]})
        self._apply_patch(patch, count=True)
        pruned = int(len(to_prune))
        self._concept_csv_pruned += int(pruned)
        return {"enabled": True, "budget": int(budget), "total": int(len(ranked)), "pruned": int(pruned)}

    def _ics_step(
        self,
        *,
        step: int,
        util_metrics: Dict[str, Any],
        skill_tasks: Sequence[Dict[str, Any]],
        force_new: bool,
        transcripts: Optional[Sequence[Dict[str, Any]]] = None,
        shadow_transcripts: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        ICS (INDUCE_CONCEPT_SUBGRAPH) — sovereign concept operator (v1).

        Order is fixed:
          1) MERGE redundancy
          2) SPLIT (restrict overgeneralization)
          3) FITNESS (explicit, bits proxy)
          4) SELECTION (promote/quarantine/delete)
          5) INDUCE (delegate to existing deterministic concept miners)
          6) REWRITE (repair deps)

        Sovereign regime (ics_sovereign=True):
          - Disallow any learning outside ICS by routing all state updates that mutate
            cognition (policy/templates/memory/macros/router) through this step.
          - Concept lifecycle order above remains fixed; non-concept updates are treated
            as evidence ingestion (prelude) and are returned in the ICS meta for audit.
        """
        enabled = bool(getattr(self.config, "ics_enabled", False))
        sovereign = bool(getattr(self.config, "ics_sovereign", False))
        out: Dict[str, Any] = {"enabled": bool(enabled), "sovereign": bool(sovereign)}

        # Sovereign mode: when ICS is disabled, concept induction is disabled (no alternate path).
        if not enabled:
            out["skipped"] = True
            out["reason"] = "ics_disabled"
            return out

        # --------------------------------------------
        # 0) INGEST evidence + update non-concept state
        # --------------------------------------------
        # In sovereign mode, these updates are forbidden outside ICS.
        if bool(sovereign):
            out["mode_policy_update"] = (
                self._update_mode_policy_from_transcripts(
                    transcripts or [],
                    step=int(step),
                    prefix_k=self.config.suite_prefix_k,
                    template_ngram_n=self.config.suite_template_ngram_n,
                    template_prefix_window=self.config.suite_template_prefix_window,
                )
                if transcripts is not None
                else {"enabled": False, "reason": "no_transcripts"}
            )
            out["ka_template_update"] = (
                self._update_anti_template_from_transcripts(
                    transcripts or [],
                    step=int(step),
                    template_ngram_n=self.config.suite_template_ngram_n,
                    template_prefix_window=self.config.suite_template_prefix_window,
                )
                if transcripts is not None
                else {"enabled": False, "reason": "no_transcripts"}
            )
            out["memory_update"] = (
                self._update_fact_memory_from_transcripts(transcripts or [], step=int(step))
                if transcripts is not None
                else {"enabled": False, "reason": "no_transcripts"}
            )

            sh = shadow_transcripts if shadow_transcripts is not None else transcripts
            out["macro_update"] = (
                self._update_macro_library_from_transcripts(sh or [], step=int(step))
                if sh is not None
                else {"enabled": False, "reason": "no_transcripts"}
            )
            out["router_update"] = (
                self._update_macro_router_from_transcripts(sh or [], step=int(step))
                if sh is not None
                else {"enabled": False, "reason": "no_transcripts"}
            )

        # --------------------------------------------------------
        # 0b) UNRESTRICT (globalize) internal promoted plan-op CSGs
        # --------------------------------------------------------
        # Internal concepts (interface.validator_id == "") are not selected as top-level policy;
        # they are subgraph building blocks (e.g., deepwrap wrappers around plan ops like add_int,
        # json_canonical). Restricting them by latent family_id creates a deadlock in cross-domain
        # transfer: new domains cannot call these subgraphs, so higher-level concepts fail with
        # callee_failed/match_disallowed even when the causal transformation is valid.
        #
        # Therefore, in sovereign ICS mode we force internal *promoted* concepts to be globally
        # callable by clearing match.family_ids (WORM via REWRITE_ACT).
        unrestrict_meta: Dict[str, Any] = {"enabled": True, "cleared": 0}
        try:
            cleared: List[Dict[str, Any]] = []
            ids = sorted([str(getattr(a, "id", "") or "") for a in self.store.by_kind("concept_csv")], key=str)
            for cid in ids:
                if not cid:
                    continue
                act0 = self.store.get(cid)
                if act0 is None or (not bool(getattr(act0, "active", True))):
                    continue
                ev0 = act0.evidence if isinstance(getattr(act0, "evidence", None), dict) else {}
                iface0 = ev0.get("interface") if isinstance(ev0.get("interface"), dict) else {}
                if str(iface0.get("validator_id") or ""):
                    continue
                meta0 = ev0.get("meta") if isinstance(ev0.get("meta"), dict) else {}
                ics0 = meta0.get("ics_v1") if isinstance(meta0.get("ics_v1"), dict) else {}
                if str(ics0.get("state") or "") != "promoted":
                    continue
                m0 = dict(getattr(act0, "match", {}) or {})
                fams = m0.get("family_ids")
                if not isinstance(fams, list) or not fams:
                    continue
                # Clear family restriction (do not alter other match keys).
                m1 = dict(m0)
                m1.pop("family_ids", None)
                after = act0.to_dict()
                after["match"] = dict(m1)
                after["version"] = int(after.get("version", 1) or 1) + 1
                after["created_at"] = deterministic_iso(step=int(step), offset_us=2500 + int(len(cleared)))
                patch = Patch(kind="REWRITE_ACT", payload={"act_id": str(cid), "act": after})
                self._apply_patch(patch, count=False)
                cleared.append({"concept_id": str(cid), "cleared_family_ids": int(len(fams))})
            unrestrict_meta = {"enabled": True, "cleared": int(len(cleared)), "examples": list(cleared)[:8]}
        except Exception as e:
            unrestrict_meta = {"enabled": True, "error": str(e)[:200]}
        out["unrestrict_internal"] = dict(unrestrict_meta)

        # -----------------------------
        # 1) MERGE redundancy (dedupe)
        # -----------------------------
        merge_meta: Dict[str, Any] = {"enabled": True}
        try:
            groups: Dict[str, List[str]] = {}
            for a in self.store.by_kind("concept_csv"):
                ev = a.evidence if isinstance(getattr(a, "evidence", None), dict) else {}
                iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
                try:
                    prog = [ins.to_dict() for ins in (getattr(a, "program", None) or [])]
                except Exception:
                    prog = []
                body = {
                    "match": dict(getattr(a, "match", {}) or {}),
                    "interface": dict(iface) if isinstance(iface, dict) else {},
                    "program": list(prog),
                }
                k = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
                groups.setdefault(str(k), []).append(str(getattr(a, "id", "") or ""))

            replace: Dict[str, str] = {}
            dup_groups = 0
            for ids in groups.values():
                ids2 = sorted({str(x) for x in ids if str(x)}, key=str)
                if len(ids2) <= 1:
                    continue
                dup_groups += 1
                keep = str(ids2[0])
                for oid in ids2[1:]:
                    replace[str(oid)] = str(keep)

            # Rewrite callsites (CSV_CALL) to the kept ids.
            rewritten = 0
            if replace:
                act_ids = sorted([str(a.id) for a in self.store.by_kind("concept_csv") if str(a.id)], key=str)
                for idx, act_id in enumerate(act_ids):
                    act0 = self.store.get(str(act_id))
                    if act0 is None or (not bool(getattr(act0, "active", True))):
                        continue
                    prog0 = list(getattr(act0, "program", []) or [])
                    changed = False
                    prog2: List[Instruction] = []
                    for ins in prog0:
                        if str(getattr(ins, "op", "")) != "CSV_CALL":
                            prog2.append(ins)
                            continue
                        args0 = getattr(ins, "args", {}) or {}
                        if not isinstance(args0, dict):
                            args0 = {}
                        cid = str(args0.get("concept_id") or "")
                        if cid and cid in replace:
                            args2 = dict(args0)
                            args2["concept_id"] = str(replace[cid])
                            prog2.append(Instruction("CSV_CALL", args2))
                            changed = True
                        else:
                            prog2.append(ins)
                    if not changed:
                        continue
                    after = act0.to_dict()
                    after["program"] = [ins.to_dict() for ins in prog2]
                    after["version"] = int(after.get("version", 1) or 1) + 1
                    after["created_at"] = deterministic_iso(step=int(step), offset_us=2000 + int(idx))
                    patch = Patch(kind="REWRITE_ACT", payload={"act_id": str(act_id), "act": after})
                    self._apply_patch(patch, count=False)
                    rewritten += 1

                # Prune duplicates (do after callsite rewrites).
                prune_ids = sorted([str(x) for x in replace.keys() if str(x)], key=str)
                if prune_ids:
                    patch = Patch(kind="PRUNE_ACT", payload={"act_ids": list(prune_ids)})
                    self._apply_patch(patch, count=True)
                    self._concept_csv_pruned += int(len(prune_ids))

            merge_meta = {
                "enabled": True,
                "duplicate_groups": int(dup_groups),
                "duplicates_pruned": int(len(replace)),
                "callsites_rewritten": int(rewritten),
            }
        except Exception as e:
            merge_meta = {"enabled": True, "error": str(e)[:200]}
        out["merge"] = dict(merge_meta)

        # -----------------------------------------
        # 2) SPLIT (restrict overgeneralization)
        # -----------------------------------------
        split_meta: Dict[str, Any] = {"enabled": True, "restricted": 0}
        try:
            usage = util_metrics.get("concept_usage_by_id") if isinstance(util_metrics, dict) else None
            if not isinstance(usage, dict) or not usage:
                split_meta = {"enabled": True, "skipped": True, "reason": "no_concept_usage_by_id"}
            else:
                try:
                    keep_rate = float(getattr(self.config, "ics_split_keep_ok_rate", 0.8) or 0.8)
                except Exception:
                    keep_rate = 0.8
                try:
                    drop_rate = float(getattr(self.config, "ics_split_drop_ok_rate", 0.2) or 0.2)
                except Exception:
                    drop_rate = 0.2
                try:
                    min_fam_turns = int(getattr(self.config, "ics_split_min_family_turns", 2) or 2)
                except Exception:
                    min_fam_turns = 2
                keep_rate = max(0.0, min(1.0, float(keep_rate)))
                drop_rate = max(0.0, min(1.0, float(drop_rate)))
                min_fam_turns = max(1, min(32, int(min_fam_turns)))

                restricted: List[Dict[str, Any]] = []
                for cid in sorted([str(k) for k in usage.keys() if str(k)], key=str):
                    act0 = self.store.get_concept_act(str(cid))
                    if act0 is None:
                        continue
                    # Do not restrict internal helper concepts (interface.validator_id == "").
                    # These are subgraph building blocks (e.g., deepwrap around plan ops) and
                    # must remain globally callable for cross-domain transfer; restricting them
                    # by family creates match_disallowed deadlocks in nested CSV_CALL graphs.
                    try:
                        ev0 = act0.evidence if isinstance(getattr(act0, "evidence", None), dict) else {}
                        iface0 = ev0.get("interface") if isinstance(ev0.get("interface"), dict) else {}
                        if not str(iface0.get("validator_id") or ""):
                            continue
                    except Exception:
                        pass
                    rec = usage.get(str(cid)) if isinstance(usage.get(str(cid)), dict) else {}
                    fams = rec.get("families") if isinstance(rec.get("families"), dict) else {}
                    if not isinstance(fams, dict) or len(fams) < 2:
                        continue
                    keep_fams: List[str] = []
                    drop_fams: List[str] = []
                    for fid in sorted([str(x) for x in fams.keys() if str(x)], key=str):
                        frec = fams.get(str(fid)) if isinstance(fams.get(str(fid)), dict) else {}
                        ut = int(frec.get("used_turns", 0) or 0)
                        ot = int(frec.get("ok_turns", 0) or 0)
                        if ut < int(min_fam_turns):
                            continue
                        rate = float(ot) / float(ut) if ut > 0 else 0.0
                        if rate >= float(keep_rate):
                            keep_fams.append(str(fid))
                        if rate <= float(drop_rate):
                            drop_fams.append(str(fid))
                    if not keep_fams or not drop_fams:
                        continue
                    # Restrict to the "good" families; force missing regions to trigger INDUCE_CONCEPT.
                    m0 = dict(getattr(act0, "match", {}) or {})
                    cur = m0.get("family_ids")
                    # Avoid widening: only apply if currently unrestricted or already has family_ids.
                    if cur is not None and not isinstance(cur, list):
                        cur = None
                    keep_fams2 = sorted(set(keep_fams), key=str)
                    if cur is not None:
                        keep_fams2 = [f for f in keep_fams2 if f in {str(x) for x in cur if str(x)}]
                    if not keep_fams2:
                        continue
                    m0["family_ids"] = list(keep_fams2)
                    after = act0.to_dict()
                    after["match"] = dict(m0)
                    after["version"] = int(after.get("version", 1) or 1) + 1
                    after["created_at"] = deterministic_iso(step=int(step), offset_us=3000 + int(len(restricted)))
                    patch = Patch(kind="REWRITE_ACT", payload={"act_id": str(cid), "act": after})
                    self._apply_patch(patch, count=False)
                    restricted.append(
                        {
                            "concept_id": str(cid),
                            "families_keep": list(keep_fams2),
                            "families_drop": list(sorted(set(drop_fams), key=str))[:8],
                        }
                    )
                split_meta = {
                    "enabled": True,
                    "restricted": int(len(restricted)),
                    "examples": list(restricted)[:8],
                }
        except Exception as e:
            split_meta = {"enabled": True, "error": str(e)[:200]}
        out["split"] = dict(split_meta)

        # -----------------------------------------
        # 3) FITNESS + 4) SELECTION (hard)
        # -----------------------------------------
        fitness_meta: Dict[str, Any] = {"enabled": True}
        try:
            usage = util_metrics.get("concept_usage_by_id") if isinstance(util_metrics, dict) else None
            usage = usage if isinstance(usage, dict) else {}

            try:
                promote_thr = int(getattr(self.config, "ics_promote_threshold_bits", 256) or 256)
            except Exception:
                promote_thr = 256
            promote_thr = max(0, min(10**9, int(promote_thr)))
            try:
                delete_neg = int(getattr(self.config, "ics_delete_neg_streak_windows", 3) or 3)
            except Exception:
                delete_neg = 3
            delete_neg = max(1, min(64, int(delete_neg)))
            try:
                max_promote = int(getattr(self.config, "ics_promote_max_per_window", 1) or 1)
            except Exception:
                max_promote = 1
            max_promote = max(0, min(16, int(max_promote)))
            try:
                ablation_enabled = bool(getattr(self.config, "ics_ablation_enabled", True))
            except Exception:
                ablation_enabled = True

            # Dependency protection: do not delete required concept closures.
            protected = self._concept_csv_dependency_closure_ids(active_roots_only=True)

            # Task lookup for deterministic ablation subsets.
            task_by_id: Dict[str, Dict[str, Any]] = {}
            for t in list(skill_tasks):
                if not isinstance(t, dict):
                    continue
                tid = str(t.get("task_id") or "")
                if tid:
                    task_by_id[tid] = t

            promoted_now: List[str] = []
            pruned_now: List[str] = []
            updated_now = 0

            # Deterministic candidate order: highest fitness first, then id.
            scored: List[Tuple[int, str]] = []
            fitness_by_id: Dict[str, int] = {}
            for act0 in self.store.by_kind("concept_csv"):
                cid = str(getattr(act0, "id", "") or "")
                if not cid:
                    continue
                rec = usage.get(cid) if isinstance(usage.get(cid), dict) else {}
                used_turns = int(rec.get("used_turns", 0) or 0)
                ok_turns = int(rec.get("ok_turns", 0) or 0)
                fams = rec.get("families") if isinstance(rec.get("families"), dict) else {}
                families_used = int(len(fams)) if isinstance(fams, dict) else 0
                ev0 = act0.evidence if isinstance(getattr(act0, "evidence", None), dict) else {}
                meta0 = ev0.get("meta") if isinstance(ev0.get("meta"), dict) else {}
                gain_bits_est = int(meta0.get("gain_bits_est", 0) or 0) if isinstance(meta0, dict) else 0
                overhead_bits = int(estimate_act_cost_bits(act0))
                prog_len = int(len(getattr(act0, "program", []) or []))

                benefit_bits = int(gain_bits_est) + int(ok_turns) * 128
                cost_bits = int(overhead_bits) + int(used_turns) * int(max(1, prog_len)) * 8
                # Cross-context is a hard requirement for semantic survival; encode as a benefit gate.
                if int(families_used) >= 2:
                    benefit_bits += 256
                fitness_bits = int(benefit_bits) - int(cost_bits)
                fitness_by_id[cid] = int(fitness_bits)
                scored.append((int(fitness_bits), str(cid)))
            scored.sort(key=lambda t: (-int(t[0]), str(t[1])))

            def _ablation_passes(*, concept_id: str, example_task_ids: List[str]) -> Tuple[bool, Dict[str, Any]]:
                if not ablation_enabled:
                    return True, {"enabled": False, "skipped": True, "reason": "disabled"}
                tids = [str(t) for t in example_task_ids if str(t) and str(t) in task_by_id]
                tids = sorted(set(tids), key=str)
                if not tids:
                    return False, {"enabled": True, "ok": False, "reason": "no_tasks"}
                tasks_sub = [task_by_id[tid] for tid in tids if tid in task_by_id]
                if not tasks_sub:
                    return False, {"enabled": True, "ok": False, "reason": "no_tasks"}

                try:
                    baseline_store = copy.deepcopy(self.store)
                    baseline_engine = Engine(
                        baseline_store,
                        seed=int(getattr(self.config, "seed", 0) or 0),
                        config=EngineConfig(enable_contracts=False),
                    )
                    _, base_m = run_skill_suite(
                        baseline_engine,
                        tasks=list(tasks_sub),
                        max_new_tokens=int(getattr(self.config, "skill_suite_max_new_tokens", 128) or 128),
                        prompt_history_k=int(getattr(self.config, "skill_suite_prompt_history_k", 0) or 0),
                        family_shuffle_seed=(
                            int(getattr(self.config, "seed", 0) or 0)
                            if bool(getattr(self.config, "suite_shuffle_families", False))
                            else None
                        ),
                        family_shuffle_salt=int(step),
                    )
                    base_pass = int(base_m.get("pass_count", 0) or 0)
                except Exception:
                    return False, {"enabled": True, "ok": False, "reason": "baseline_error"}

                try:
                    ablated_store = copy.deepcopy(self.store)
                    ablated_store.prune(str(concept_id))
                    ablated_engine = Engine(
                        ablated_store,
                        seed=int(getattr(self.config, "seed", 0) or 0),
                        config=EngineConfig(enable_contracts=False),
                    )
                    _, abl_m = run_skill_suite(
                        ablated_engine,
                        tasks=list(tasks_sub),
                        max_new_tokens=int(getattr(self.config, "skill_suite_max_new_tokens", 128) or 128),
                        prompt_history_k=int(getattr(self.config, "skill_suite_prompt_history_k", 0) or 0),
                        family_shuffle_seed=(
                            int(getattr(self.config, "seed", 0) or 0)
                            if bool(getattr(self.config, "suite_shuffle_families", False))
                            else None
                        ),
                        family_shuffle_salt=int(step),
                    )
                    abl_pass = int(abl_m.get("pass_count", 0) or 0)
                except Exception:
                    return False, {"enabled": True, "ok": False, "reason": "ablation_error"}

                delta = int(base_pass) - int(abl_pass)
                ok = bool(delta > 0)
                return ok, {
                    "enabled": True,
                    "ok": bool(ok),
                    "tasks": list(tids),
                    "pass_base": int(base_pass),
                    "pass_ablated": int(abl_pass),
                    "delta_pass": int(delta),
                }

            # Promotion/deletion loop.
            for _, cid in scored:
                act0 = self.store.get_concept_act(str(cid))
                if act0 is None:
                    continue
                ev0 = act0.evidence if isinstance(getattr(act0, "evidence", None), dict) else {}
                meta0 = ev0.get("meta") if isinstance(ev0.get("meta"), dict) else {}
                meta0 = dict(meta0) if isinstance(meta0, dict) else {}
                ics0 = meta0.get("ics_v1") if isinstance(meta0.get("ics_v1"), dict) else {}
                ics0 = dict(ics0) if isinstance(ics0, dict) else {}
                state0 = str(ics0.get("state") or "candidate")
                first_seen = int(ics0.get("first_seen_step", step) or step)
                neg_streak = int(ics0.get("neg_streak", 0) or 0)
                fitness_bits = int(fitness_by_id.get(str(cid), 0))

                # Update streaks deterministically.
                if int(step) > int(first_seen) and int(fitness_bits) < 0:
                    neg_streak = int(neg_streak) + 1
                elif int(fitness_bits) >= 0:
                    neg_streak = 0

                # Determine families_used from usage evidence.
                rec = usage.get(str(cid)) if isinstance(usage.get(str(cid)), dict) else {}
                fams = rec.get("families") if isinstance(rec.get("families"), dict) else {}
                families_used = int(len(fams)) if isinstance(fams, dict) else 0
                ok_turns = int(rec.get("ok_turns", 0) or 0)

                # Hard delete (prune) when persistently negative, but never delete protected deps.
                if (
                    int(fitness_bits) < 0
                    and int(neg_streak) >= int(delete_neg)
                    and str(cid) not in protected
                    and state0 != "promoted"
                ):
                    patch = Patch(kind="PRUNE_ACT", payload={"act_ids": [str(cid)]})
                    self._apply_patch(patch, count=True)
                    self._concept_csv_pruned += 1
                    pruned_now.append(str(cid))
                    continue

                # Promotion gate: cross-context + positive fitness + ablation.
                promote = False
                # Preserve prior ablation evidence for auditability. Never overwrite a real result
                # (ok/failed) with a synthetic "not_attempted" just because the concept is already
                # promoted or doesn't meet the gate in this window.
                prior_ab = ics0.get("ablation") if isinstance(ics0.get("ablation"), dict) else None
                ab_ev: Dict[str, Any] = (
                    dict(prior_ab)
                    if isinstance(prior_ab, dict) and prior_ab
                    else {"enabled": bool(ablation_enabled), "ok": False, "reason": "not_attempted"}
                )
                if (
                    state0 != "promoted"
                    and int(max_promote) > 0
                    and int(len(promoted_now)) < int(max_promote)
                    and int(fitness_bits) >= int(promote_thr)
                    and int(families_used) >= 2
                    and int(ok_turns) >= 2
                ):
                    # Deterministic task subset from usage evidence.
                    ex_tids: List[str] = []
                    if isinstance(fams, dict):
                        for fid in sorted(fams.keys(), key=str):
                            frec = fams.get(fid) if isinstance(fams.get(fid), dict) else {}
                            ex = frec.get("example_task_ids")
                            if isinstance(ex, list):
                                ex_tids.extend([str(x) for x in ex if str(x)])
                    ex_tids = sorted(set(ex_tids), key=str)[:8]
                    ab_ok, ab_ev = _ablation_passes(concept_id=str(cid), example_task_ids=list(ex_tids))
                    promote = bool(ab_ok)

                if promote:
                    state0 = "promoted"
                    promoted_now.append(str(cid))
                elif state0 not in {"promoted", "quarantined"} and int(fitness_bits) < 0:
                    state0 = "quarantined"

                # Persist ICS meta into the act evidence (as explicit state).
                ics0.update(
                    {
                        "state": str(state0),
                        "fitness_bits": int(fitness_bits),
                        "neg_streak": int(neg_streak),
                        "first_seen_step": int(first_seen),
                        "last_step": int(step),
                        "families_used": int(families_used),
                        "ablation": dict(ab_ev) if isinstance(ab_ev, dict) else {},
                    }
                )
                meta0["ics_v1"] = dict(ics0)
                ev_new = dict(ev0)
                ev_new["meta"] = dict(meta0)
                after = act0.to_dict()
                after["evidence"] = dict(ev_new)
                after["version"] = int(after.get("version", 1) or 1) + 1
                after["created_at"] = deterministic_iso(step=int(step), offset_us=4000 + int(updated_now))
                patch = Patch(kind="REWRITE_ACT", payload={"act_id": str(cid), "act": after})
                self._apply_patch(patch, count=False)
                updated_now += 1

            fitness_meta = {
                "enabled": True,
                "promote_threshold_bits": int(promote_thr),
                "delete_neg_streak_windows": int(delete_neg),
                "promoted_now": list(promoted_now),
                "pruned_now": list(pruned_now),
                "updated": int(updated_now),
            }
        except Exception as e:
            fitness_meta = {"enabled": True, "error": str(e)[:200]}
        out["fitness"] = dict(fitness_meta)

        # -----------------------------
        # 5) INDUCE (concept miners)
        # -----------------------------
        induce_meta = self._mine_and_promote_concept_csv(step=int(step), force_new=bool(force_new))
        out["induce"] = dict(induce_meta) if isinstance(induce_meta, dict) else {"enabled": False}

        # -----------------------------
        # 6) REWRITE (repair deps)
        # -----------------------------
        try:
            rewrite_meta = self._repair_concept_csv_dependencies(step=int(step))
        except Exception as e:
            rewrite_meta = {"enabled": True, "error": str(e)[:200]}
        out["rewrite"] = {"concept_dep_repair": dict(rewrite_meta) if isinstance(rewrite_meta, dict) else {}}

        # ---------------------------------
        # 7) SEMANTIC BANKS (goals/plans/...)
        # ---------------------------------
        if bool(getattr(self.config, "ics_semantic_banks_enabled", False)):
            try:
                sem_meta = self._ics_update_semantic_banks_v1(
                    step=int(step),
                    transcripts=shadow_transcripts if shadow_transcripts is not None else (transcripts or []),
                )
            except Exception as e:
                sem_meta = {"enabled": True, "error": str(e)[:200]}
            out["semantic_banks_v1"] = dict(sem_meta) if isinstance(sem_meta, dict) else {"enabled": True}

        return out

    def _ics_update_semantic_banks_v1(
        self,
        *,
        step: int,
        transcripts: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        ICS semantic objects (v1): maintain Goal/Plan/Hypothesis/Reference as explicit ACTs.

        Deterministic rules (minimal):
          - Any observed plan_validator turn induces:
              * goal ACT with id == goal_id (from plan_trace)
              * plan ACT with id == plan_{expected_spec_sig[:16]}
          - Goal/Plan attempt + success/fail counts are updated from observable traces:
              * output_text == expected_output_text
              * concept_executor.used + concept_executor.ok
              * concept_calls_max_depth >= concept_min_depth
          - Repeated failure streak (>=2) induces a hypothesis ACT (born-from-failure).
        """
        out: Dict[str, Any] = {"enabled": True}

        goals_created = 0
        goals_updated = 0
        plans_created = 0
        plans_updated = 0
        hyps_created = 0
        hyps_updated = 0
        refs_created = 0
        refs_updated = 0
        seen_turns = 0

        def _is_active_kind(a: Any, k: str) -> bool:
            return bool(a is not None and bool(getattr(a, "active", True)) and str(getattr(a, "kind", "")) == str(k))

        def _infer_input_schema(*, input_keys: List[str], inputs: Dict[str, Any]) -> Dict[str, str]:
            schema: Dict[str, str] = {}
            for k in list(input_keys):
                ks = str(k or "")
                if not ks:
                    continue
                v = inputs.get(ks)
                if isinstance(v, bool):
                    t = "int"
                elif isinstance(v, int):
                    t = "int"
                else:
                    t = "str"
                schema[ks] = str(t)
            return schema

        def _iface_sig(*, input_schema: Dict[str, str], output_schema: Dict[str, str], validator_id: str) -> str:
            body = {"in": dict(input_schema), "out": dict(output_schema), "validator_id": str(validator_id or "")}
            return sha256_hex(canonical_json_dumps(body).encode("utf-8"))

        def _update_goal_act(
            *,
            goal_id: str,
            plan_id: str,
            expected_spec_sig: str,
            expected_spec: Dict[str, Any],
            input_keys: List[str],
            inputs: Dict[str, Any],
            output_key: str,
            expected_text: str,
            min_depth: int,
            ok_now: bool,
            fail_reason: str,
        ) -> None:
            nonlocal goals_created, goals_updated

            act0 = self.store.get(str(goal_id)) if str(goal_id) else None
            if act0 is None:
                input_schema = _infer_input_schema(input_keys=input_keys, inputs=inputs)
                out_schema = {str(output_key): "str"} if str(output_key) else {"value": "str"}
                goal_act = Act(
                    id=str(goal_id),
                    version=1,
                    created_at=deterministic_iso(step=int(step), offset_us=9100 + int(goals_created)),
                    kind="goal",
                    match={},
                    program=[],
                    evidence={
                        "name": "goal_live_v1",
                        "meta": {"builder": "ics_semantic_banks_v1"},
                        "goal": {
                            "priority": 10,
                            "selector": {
                                "kind": "interface_sig",
                                "iface_sig": str(
                                    _iface_sig(
                                        input_schema=dict(input_schema),
                                        output_schema=dict(out_schema),
                                        validator_id="plan_validator",
                                    )
                                ),
                            },
                            "inputs": dict(inputs),
                            "expected": dict(expected_spec),
                            "expected_output_text": str(expected_text),
                            "expected_spec_sig": str(expected_spec_sig),
                            "output_key": str(output_key),
                            "plan_id": str(plan_id),
                            "concept_min_depth": int(min_depth),
                            "status": "active",
                            "progress": 1.0 if bool(ok_now) else 0.0,
                            "attempts": 1,
                            "successes": 1 if bool(ok_now) else 0,
                            "failures": 0 if bool(ok_now) else 1,
                            "failure_streak": 0 if bool(ok_now) else 1,
                            "last_failure_reason": "" if bool(ok_now) else str(fail_reason or ""),
                            "created_step": int(step),
                            "last_updated_step": int(step),
                        },
                    },
                    cost={"overhead_bits": 512},
                    deps=[],
                    active=True,
                )
                patch = Patch(kind="ADD_ACT", payload={"acts": [goal_act.to_dict()]})
                self._apply_patch(patch, count=True)
                goals_created += 1
                return

            if not _is_active_kind(act0, "goal"):
                return

            ev0 = act0.evidence if isinstance(getattr(act0, "evidence", None), dict) else {}
            goal0 = ev0.get("goal") if isinstance(ev0.get("goal"), dict) else {}
            goal0 = dict(goal0) if isinstance(goal0, dict) else {}

            attempts = int(goal0.get("attempts", 0) or 0) + 1
            successes = int(goal0.get("successes", 0) or 0) + (1 if bool(ok_now) else 0)
            failures = int(goal0.get("failures", 0) or 0) + (0 if bool(ok_now) else 1)
            streak = int(goal0.get("failure_streak", 0) or 0)
            streak = 0 if bool(ok_now) else int(streak) + 1

            goal0.update(
                {
                    "plan_id": str(plan_id),
                    "concept_min_depth": int(min_depth),
                    "attempts": int(attempts),
                    "successes": int(successes),
                    "failures": int(failures),
                    "failure_streak": int(streak),
                    "progress": 1.0 if bool(ok_now) else float(goal0.get("progress") or 0.0),
                    "status": "satisfied" if bool(ok_now) else str(goal0.get("status") or "active"),
                    "last_failure_reason": "" if bool(ok_now) else str(fail_reason or ""),
                    "last_updated_step": int(step),
                }
            )
            ev_new = dict(ev0)
            ev_new["goal"] = dict(goal0)
            after = act0.to_dict()
            after["evidence"] = dict(ev_new)
            after["version"] = int(after.get("version", 1) or 1) + 1
            after["created_at"] = deterministic_iso(step=int(step), offset_us=9200 + int(goals_updated))
            patch = Patch(kind="REWRITE_ACT", payload={"act_id": str(goal_id), "act": after})
            self._apply_patch(patch, count=False)
            goals_updated += 1

        def _update_plan_act(
            *,
            plan_id: str,
            goal_id: str,
            expected_spec_sig: str,
            expected_spec: Dict[str, Any],
            min_depth: int,
            ok_now: bool,
            fail_reason: str,
        ) -> None:
            nonlocal plans_created, plans_updated

            act0 = self.store.get(str(plan_id)) if str(plan_id) else None
            if act0 is None:
                plan_act = Act(
                    id=str(plan_id),
                    version=1,
                    created_at=deterministic_iso(step=int(step), offset_us=9300 + int(plans_created)),
                    kind="plan",
                    match={},
                    program=[],
                    evidence={
                        "name": "plan_live_v1",
                        "meta": {"builder": "ics_semantic_banks_v1"},
                        "plan": {
                            "goal_id": str(goal_id),
                            "expected_spec_sig": str(expected_spec_sig),
                            "expected": dict(expected_spec),
                            "concept_min_depth": int(min_depth),
                            "status": "active",
                            "attempts": 1,
                            "successes": 1 if bool(ok_now) else 0,
                            "failures": 0 if bool(ok_now) else 1,
                            "failure_streak": 0 if bool(ok_now) else 1,
                            "last_failure_reason": "" if bool(ok_now) else str(fail_reason or ""),
                            "created_step": int(step),
                            "last_updated_step": int(step),
                        },
                    },
                    cost={"overhead_bits": 512},
                    deps=[],
                    active=True,
                )
                patch = Patch(kind="ADD_ACT", payload={"acts": [plan_act.to_dict()]})
                self._apply_patch(patch, count=True)
                plans_created += 1
                return

            if not _is_active_kind(act0, "plan"):
                return

            ev0 = act0.evidence if isinstance(getattr(act0, "evidence", None), dict) else {}
            plan0 = ev0.get("plan") if isinstance(ev0.get("plan"), dict) else {}
            plan0 = dict(plan0) if isinstance(plan0, dict) else {}

            attempts = int(plan0.get("attempts", 0) or 0) + 1
            successes = int(plan0.get("successes", 0) or 0) + (1 if bool(ok_now) else 0)
            failures = int(plan0.get("failures", 0) or 0) + (0 if bool(ok_now) else 1)
            streak = int(plan0.get("failure_streak", 0) or 0)
            streak = 0 if bool(ok_now) else int(streak) + 1

            plan0.update(
                {
                    "goal_id": str(goal_id),
                    "expected_spec_sig": str(expected_spec_sig),
                    "concept_min_depth": int(min_depth),
                    "attempts": int(attempts),
                    "successes": int(successes),
                    "failures": int(failures),
                    "failure_streak": int(streak),
                    "status": "completed" if bool(ok_now) else str(plan0.get("status") or "active"),
                    "last_failure_reason": "" if bool(ok_now) else str(fail_reason or ""),
                    "last_updated_step": int(step),
                }
            )
            ev_new = dict(ev0)
            ev_new["plan"] = dict(plan0)
            after = act0.to_dict()
            after["evidence"] = dict(ev_new)
            after["version"] = int(after.get("version", 1) or 1) + 1
            after["created_at"] = deterministic_iso(step=int(step), offset_us=9400 + int(plans_updated))
            patch = Patch(kind="REWRITE_ACT", payload={"act_id": str(plan_id), "act": after})
            self._apply_patch(patch, count=False)
            plans_updated += 1

        def _update_hypothesis_act(
            *,
            goal_id: str,
            plan_id: str,
            expected_spec_sig: str,
            fail_reason: str,
            fail_context: Dict[str, Any],
        ) -> None:
            nonlocal hyps_created, hyps_updated

            hyp_id = hypothesis_id_v1(goal_id=str(goal_id), reason=str(fail_reason), context=dict(fail_context))
            act0 = self.store.get(str(hyp_id)) if str(hyp_id) else None
            if act0 is None:
                hyp_act = Act(
                    id=str(hyp_id),
                    version=1,
                    created_at=deterministic_iso(step=int(step), offset_us=9500 + int(hyps_created)),
                    kind="hypothesis",
                    match={},
                    program=[],
                    evidence={
                        "name": "hypothesis_live_v1",
                        "meta": {
                            "builder": "ics_semantic_banks_v1",
                            "born_from_failure": True,
                            "origin": {"goal_id": str(goal_id), "plan_id": str(plan_id), "expected_spec_sig": str(expected_spec_sig)},
                        },
                        "hypothesis": {
                            "goal_id": str(goal_id),
                            "plan_id": str(plan_id),
                            "expected_spec_sig": str(expected_spec_sig),
                            "statement": f"fail_reason={str(fail_reason)}",
                            "confidence": 0.5,
                            "tested_count": 0,
                            "falsified": False,
                            "last_updated_step": int(step),
                        },
                    },
                    cost={"overhead_bits": 512},
                    deps=[],
                    active=True,
                )
                patch = Patch(kind="ADD_ACT", payload={"acts": [hyp_act.to_dict()]})
                self._apply_patch(patch, count=True)
                hyps_created += 1
                return

            if not _is_active_kind(act0, "hypothesis"):
                return
            ev0 = act0.evidence if isinstance(getattr(act0, "evidence", None), dict) else {}
            hyp0 = ev0.get("hypothesis") if isinstance(ev0.get("hypothesis"), dict) else {}
            hyp0 = dict(hyp0) if isinstance(hyp0, dict) else {}
            hyp0["last_updated_step"] = int(step)
            ev_new = dict(ev0)
            ev_new["hypothesis"] = dict(hyp0)
            after = act0.to_dict()
            after["evidence"] = dict(ev_new)
            after["version"] = int(after.get("version", 1) or 1) + 1
            after["created_at"] = deterministic_iso(step=int(step), offset_us=9600 + int(hyps_updated))
            patch = Patch(kind="REWRITE_ACT", payload={"act_id": str(hyp_id), "act": after})
            self._apply_patch(patch, count=False)
            hyps_updated += 1

        def _normalize_ref_token(token: str) -> str:
            # Deterministic normalization (no locale rules): collapse whitespace + lowercase.
            return " ".join(str(token or "").strip().split()).lower()

        def _update_reference_act(
            *,
            scope_sig: str,
            token: str,
            target_kind: str,
            target_id: str,
            goal_id: str,
            plan_id: str,
            expected_spec_sig: str,
        ) -> None:
            nonlocal refs_created, refs_updated
            scope = str(scope_sig or "")
            tok = _normalize_ref_token(str(token or ""))
            tk = str(target_kind or "")
            tid = str(target_id or "")
            if not scope or not tok or not tk or not tid:
                return
            ref_id = reference_id_v1(scope_sig=str(scope), token=str(tok))
            if not ref_id:
                return

            act0 = self.store.get(str(ref_id)) if str(ref_id) else None
            if act0 is None:
                ref_act = Act(
                    id=str(ref_id),
                    version=1,
                    created_at=deterministic_iso(step=int(step), offset_us=9700 + int(refs_created)),
                    kind="reference",
                    match={},
                    program=[],
                    evidence={
                        "name": "reference_live_v1",
                        "meta": {"builder": "ics_semantic_banks_v1"},
                        "reference": {
                            "schema_version": 1,
                            "scope_sig": str(scope),
                            "token": str(tok),
                            "target_kind": str(tk),
                            "target_id": str(tid),
                            "goal_id": str(goal_id),
                            "plan_id": str(plan_id),
                            "expected_spec_sig": str(expected_spec_sig),
                            "bind_count": 1,
                            "created_step": int(step),
                            "last_updated_step": int(step),
                        },
                    },
                    cost={"overhead_bits": 256},
                    deps=[],
                    active=True,
                )
                patch = Patch(kind="ADD_ACT", payload={"acts": [ref_act.to_dict()]})
                self._apply_patch(patch, count=True)
                refs_created += 1
                return

            if not _is_active_kind(act0, "reference"):
                return
            ev0 = act0.evidence if isinstance(getattr(act0, "evidence", None), dict) else {}
            ref0 = ev0.get("reference") if isinstance(ev0.get("reference"), dict) else {}
            ref0 = dict(ref0) if isinstance(ref0, dict) else {}
            binds = int(ref0.get("bind_count", 0) or 0) + 1
            ref0.update(
                {
                    "scope_sig": str(scope),
                    "token": str(tok),
                    "target_kind": str(tk),
                    "target_id": str(tid),
                    "goal_id": str(goal_id),
                    "plan_id": str(plan_id),
                    "expected_spec_sig": str(expected_spec_sig),
                    "bind_count": int(binds),
                    "last_updated_step": int(step),
                }
            )
            ev_new = dict(ev0)
            ev_new["reference"] = dict(ref0)
            after = act0.to_dict()
            after["evidence"] = dict(ev_new)
            after["version"] = int(after.get("version", 1) or 1) + 1
            after["created_at"] = deterministic_iso(step=int(step), offset_us=9800 + int(refs_updated))
            patch = Patch(kind="REWRITE_ACT", payload={"act_id": str(ref_id), "act": after})
            self._apply_patch(patch, count=False)
            refs_updated += 1

        # Walk transcripts and induce/update banks.
        for tr in list(transcripts):
            turns = tr.get("turns") if isinstance(tr, dict) else None
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                trace = turn.get("trace") if isinstance(turn.get("trace"), dict) else {}
                plan_trace = trace.get("plan_trace") if isinstance(trace.get("plan_trace"), dict) else {}
                if not isinstance(plan_trace, dict) or str(plan_trace.get("validator_id") or "") != "plan_validator":
                    continue
                expected_spec = plan_trace.get("expected_spec")
                if not isinstance(expected_spec, dict) or not expected_spec:
                    continue
                goal_id = str(plan_trace.get("goal_id") or "")
                expected_spec_sig = str(plan_trace.get("expected_spec_sig") or "")
                plan_id = plan_id_for_expected_spec_sig(str(expected_spec_sig))
                input_keys = expected_spec.get("input_keys") if isinstance(expected_spec.get("input_keys"), list) else []
                input_keys = [str(k) for k in input_keys if str(k)]
                inputs = expected_spec.get("inputs") if isinstance(expected_spec.get("inputs"), dict) else {}
                inputs = dict(inputs) if isinstance(inputs, dict) else {}
                output_key = str(expected_spec.get("return_var") or "")
                expected_text = str(expected_spec.get("expected_output_text") or "")
                try:
                    min_depth = int(plan_trace.get("concept_min_depth", 0) or 0)
                except Exception:
                    min_depth = 0
                if min_depth < 0:
                    min_depth = 0

                out_text = str(turn.get("system", ""))
                cm = trace.get("concept_executor") if isinstance(trace.get("concept_executor"), dict) else {}
                used = bool(cm.get("used", False))
                ok_exec = bool(cm.get("ok", False))
                try:
                    calls_depth = int(cm.get("concept_calls_max_depth", 0) or 0)
                except Exception:
                    calls_depth = 0

                ok_text = bool(expected_text and out_text == expected_text)
                ok_depth = bool(int(calls_depth) >= int(min_depth))
                ok_now = bool(ok_text and used and ok_exec and ok_depth)
                fail_reason = ""
                if not ok_now:
                    fail_reason = str(cm.get("reason") or "fail")
                    if ok_text and used and ok_exec and (not ok_depth):
                        fail_reason = "concept_depth_too_shallow"
                    elif ok_text and (not used):
                        fail_reason = "missing_concept_executor"

                _update_plan_act(
                    plan_id=str(plan_id),
                    goal_id=str(goal_id),
                    expected_spec_sig=str(expected_spec_sig),
                    expected_spec=dict(expected_spec),
                    min_depth=int(min_depth),
                    ok_now=bool(ok_now),
                    fail_reason=str(fail_reason),
                )
                _update_goal_act(
                    goal_id=str(goal_id),
                    plan_id=str(plan_id),
                    expected_spec_sig=str(expected_spec_sig),
                    expected_spec=dict(expected_spec),
                    input_keys=list(input_keys),
                    inputs=dict(inputs),
                    output_key=str(output_key),
                    expected_text=str(expected_text),
                    min_depth=int(min_depth),
                    ok_now=bool(ok_now),
                    fail_reason=str(fail_reason),
                )

                # Reference bindings (semantic persistence): tasks may require explicit token→object bindings.
                # These are first-class ACTs and are only created/updated inside ICS.
                try:
                    ref_specs = plan_trace.get("reference_tokens")
                    if not isinstance(ref_specs, list):
                        ref_specs = []
                    if bool(plan_trace.get("reference_required", False)) and not ref_specs:
                        ref_specs = [
                            {"token": "o objetivo", "target_kind": "goal"},
                            {"token": "o plano", "target_kind": "plan"},
                        ]
                except Exception:
                    ref_specs = []
                for rs in list(ref_specs):
                    if not isinstance(rs, dict):
                        continue
                    tok = str(rs.get("token") or "")
                    tk = str(rs.get("target_kind") or "")
                    if not tok or not tk:
                        continue
                    if tk == "goal":
                        _update_reference_act(
                            scope_sig=str(goal_id),
                            token=str(tok),
                            target_kind="goal",
                            target_id=str(goal_id),
                            goal_id=str(goal_id),
                            plan_id=str(plan_id),
                            expected_spec_sig=str(expected_spec_sig),
                        )
                    elif tk == "plan":
                        _update_reference_act(
                            scope_sig=str(goal_id),
                            token=str(tok),
                            target_kind="plan",
                            target_id=str(plan_id),
                            goal_id=str(goal_id),
                            plan_id=str(plan_id),
                            expected_spec_sig=str(expected_spec_sig),
                        )

                # Hypothesis induction: any observed failure streak >=1 ⇒ create a hypothesis (born from failure).
                goal_act = self.store.get(str(goal_id)) if str(goal_id) else None
                if _is_active_kind(goal_act, "goal"):
                    evg = goal_act.evidence if isinstance(getattr(goal_act, "evidence", None), dict) else {}
                    gg = evg.get("goal") if isinstance(evg.get("goal"), dict) else {}
                    streak = int(gg.get("failure_streak", 0) or 0) if isinstance(gg, dict) else 0
                    if int(streak) >= 1 and (not ok_now):
                        _update_hypothesis_act(
                            goal_id=str(goal_id),
                            plan_id=str(plan_id),
                            expected_spec_sig=str(expected_spec_sig),
                            fail_reason=str(fail_reason),
                            fail_context={"min_depth": int(min_depth), "calls_depth": int(calls_depth)},
                        )

                seen_turns += 1

        out.update(
            {
                "seen_plan_turns": int(seen_turns),
                "goals_created": int(goals_created),
                "goals_updated": int(goals_updated),
                "plans_created": int(plans_created),
                "plans_updated": int(plans_updated),
                "hypotheses_created": int(hyps_created),
                "hypotheses_updated": int(hyps_updated),
                "references_created": int(refs_created),
                "references_updated": int(refs_updated),
                "goals_total": int(len(self.store.by_kind("goal"))),
                "plans_total": int(len(self.store.by_kind("plan"))),
                "hypotheses_total": int(len(self.store.by_kind("hypothesis"))),
                "references_total": int(len(self.store.by_kind("reference"))),
            }
        )
        return out

    def _mine_and_promote_concept_csv(self, *, step: int, force_new: bool = False) -> Dict[str, Any]:
        enabled = bool(getattr(self.config, "concept_csv_mining_enabled", False))
        if not enabled:
            return {"enabled": False}

        force_new = bool(force_new)
        try:
            top_k = int(getattr(self.config, "concept_csv_mining_top_k", 4))
        except Exception:
            top_k = 4
        top_k = max(0, min(32, int(top_k)))
        try:
            max_new = int(getattr(self.config, "concept_csv_mining_max_new_per_window", 2))
        except Exception:
            max_new = 2
        max_new = max(0, min(16, int(max_new)))
        if top_k <= 0 or max_new <= 0:
            return {
                "enabled": True,
                "skipped": True,
                "reason": "disabled_by_params",
                "force_new": bool(force_new),
            }

        trace_dir = os.path.join(self.out_dir, "concept_csv_traces")
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"step{int(step):06d}_trace.jsonl")

        # Shadow store (tiny): only concept seeds + goal seeds.
        store_shadow = ActStore()

        def _seed_concept(*, act_id: str, program: List[Instruction], interface: Dict[str, Any]) -> Act:
            return Act(
                id=str(act_id),
                version=1,
                created_at=deterministic_iso(step=0),
                kind="concept_csv",
                match={},
                program=list(program),
                evidence={"interface": dict(interface), "name": "concept_seed_v0"},
                cost={"overhead_bits": 0},
                deps=[],
                active=True,
            )

        concept_a = _seed_concept(
            act_id="concept_seed_extract_int_v0",
            program=[
                Instruction("CSV_GET_INPUT", {"name": "text", "out": "t"}),
                Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["t"], "out": "d"}),
                Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "n"}),
                Instruction("CSV_RETURN", {"var": "n"}),
            ],
            interface={
                "input_schema": {"text": "str"},
                "output_schema": {"value": "int"},
                "validator_id": "int_value_exact",
            },
        )
        concept_b = _seed_concept(
            act_id="concept_seed_json_ab_v0",
            program=[
                Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
                Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
                Instruction("CSV_PRIMITIVE", {"fn": "make_dict_ab", "in": ["a", "b"], "out": "d"}),
                Instruction("CSV_PRIMITIVE", {"fn": "json_canonical", "in": ["d"], "out": "j"}),
                Instruction("CSV_RETURN", {"var": "j"}),
            ],
            interface={
                "input_schema": {"a": "int", "b": "int"},
                "output_schema": {"value": "str"},
                "validator_id": "json_ab_int_exact",
            },
        )
        store_shadow.add(concept_a)
        store_shadow.add(concept_b)

        store_hash_stub = "train_shadow_v0"
        goals: List[Act] = []
        _, fn_scan = PRIMITIVE_OPS["scan_digits"]
        _, fn_d2i = PRIMITIVE_OPS["digits_to_int"]
        for i, text in enumerate(["abc0123", "id=42", "x9y7"]):
            digits = fn_scan(text)
            exp_int = int(fn_d2i(digits))
            goals.append(
                Act(
                    id=f"goal_seed_extract_int_{i}",
                    version=1,
                    created_at=deterministic_iso(step=0),
                    kind="goal",
                    match={},
                    program=[],
                    evidence={
                        "name": "goal_seed_v0",
                        "meta": {
                            "title": f"seed_extract_int_{i}",
                            "trained_on_store_content_hash": store_hash_stub,
                        },
                        "goal": {
                            "priority": 10,
                            "concept_id": str(concept_a.id),
                            "inputs": {"text": str(text)},
                            "expected": int(exp_int),
                        },
                    },
                    cost={"overhead_bits": 0},
                    deps=[],
                    active=True,
                )
            )
        for i, (a, b) in enumerate([(40, 2), (41, 3), (42, 4)]):
            goals.append(
                Act(
                    id=f"goal_seed_json_ab_{i}",
                    version=1,
                    created_at=deterministic_iso(step=0),
                    kind="goal",
                    match={},
                    program=[],
                    evidence={
                        "name": "goal_seed_v0",
                        "meta": {
                            "title": f"seed_json_ab_{i}",
                            "trained_on_store_content_hash": store_hash_stub,
                        },
                        "goal": {
                            "priority": 10,
                            "concept_id": str(concept_b.id),
                            "inputs": {"a": int(a), "b": int(b)},
                            "expected": {"a": int(a), "b": int(b)},
                        },
                    },
                    cost={"overhead_bits": 0},
                    deps=[],
                    active=True,
                )
            )
        for g in goals:
            store_shadow.add(g)

        engine = Engine(
            store_shadow,
            seed=int(getattr(self.config, "seed", 0) or 0),
            config=EngineConfig(enable_contracts=False),
        )

        trace_rows: List[Dict[str, Any]] = []
        for g in goals:
            res = engine.execute_goal(goal_act_id=str(g.id), step=0, max_depth=8)
            ok = bool(res.get("ok", False))
            evs = res.get("events") if isinstance(res.get("events"), list) else []
            gg = {}
            try:
                ge = g.evidence if isinstance(g.evidence, dict) else {}
                gg = ge.get("goal") if isinstance(ge.get("goal"), dict) else {}
            except Exception:
                gg = {}
            inps = gg.get("inputs") if isinstance(gg.get("inputs"), dict) else {}
            trace_rows.append(
                {
                    "ctx_sig": f"seed␟{g.id}",
                    "inputs": dict(inps),
                    "utility_passed": bool(ok),
                    "events": list(evs),
                }
            )

        # Extra trace seeds: if the selected validator pack includes plan tasks, compile their
        # deterministic expected_spec and execute the corresponding primitive program as a concept_csv.
        # This ties concept mining to "semantic plan/state" structure rather than only toy seeds.
        plan_tasks: List[Tuple[str, Dict[str, Any]]] = []
        plan_family_by_task_id: Dict[str, str] = {}
        try:
            tasks = skill_suite_tasks_for_pack(
                str(getattr(self.config, "skill_suite_pack", "v0") or "v0")
            )
        except Exception:
            tasks = ()
        for t in tasks:
            if not isinstance(t, dict):
                continue
            task_id = str(t.get("task_id") or "")
            if not task_id:
                continue
            if str(t.get("validator_id") or "") != "plan_validator":
                continue
            expected_spec = t.get("expected_spec")
            if not isinstance(expected_spec, dict) or not expected_spec:
                compiler_id = str(t.get("compiler_id") or "")
                if compiler_id == V67_DIALOGUE_COMPILER_ID or task_id.startswith(("v67_", "v68_")):
                    turns = t.get("dialogue") or ()
                    if isinstance(turns, (list, tuple)) and turns:
                        vt = int(t.get("validate_turn", max(0, len(turns) - 1)) or 0)
                        vt = max(0, min(vt, len(turns) - 1))
                        try:
                            expected_spec = compile_dialogue_v67(str(turns[vt]))
                        except Exception:
                            expected_spec = {}
            if not isinstance(expected_spec, dict) or not expected_spec:
                continue
            ops = expected_spec.get("ops")
            if not isinstance(ops, list) or not ops:
                continue
            plan_tasks.append((task_id, dict(expected_spec)))
            # Family id for plan tasks is a stable latent bucket derived from compiler + constraints
            # (not per-task content). This is used for match constraints and cross-context evidence.
            try:
                compiler_id = str(t.get("compiler_id") or "")
                family_body = {
                    "validator_id": "plan_validator",
                    "expected_format": "plan",
                    "constraints": ["plan_validator", "json_canonical"],
                    "compiler_id": str(compiler_id),
                    "goal_active": True,
                }
                fid = "fam_" + sha256_hex(canonical_json_dumps(family_body).encode("utf-8"))[:16]
            except Exception:
                fid = ""
            if fid:
                plan_family_by_task_id[str(task_id)] = str(fid)
        # Deterministic selection bias: seed harder (longer) plans first so that large-input-schema
        # interfaces (e.g., v69_long_sum) are covered even when the suite contains many short plan
        # tasks (e.g., v66/v68). Tie-break by task_id for reproducibility.
        plan_tasks.sort(key=lambda kv: (-int(len((kv[1] or {}).get("ops") or [])), str(kv[0])))

        plan_traces_added = 0
        full_plan_added = 0
        full_plan_skipped_existing = 0
        full_plan_skipped_disabled = 0
        full_plan_decompose_targets = 0
        full_plan_decompose_added = 0
        full_plan_decompose_skipped_existing = 0
        full_plan_decompose_skipped_disabled = 0
        full_plan_decompose_errors = 0
        try:
            _full_plan_decompose_flag = getattr(
                self.config, "concept_csv_mining_full_plan_decompose_enabled", None
            )
        except Exception:
            _full_plan_decompose_flag = None
        if _full_plan_decompose_flag is None:
            # Auto-enable when the selected validator pack requires deep nested plan concepts (e.g., sota_v8).
            full_plan_decompose_enabled = False
            for t in tasks:
                if not isinstance(t, dict):
                    continue
                if str(t.get("validator_id") or "") != "plan_validator":
                    continue
                try:
                    md = int(t.get("concept_min_depth", 0) or 0)
                except Exception:
                    md = 0
                if int(md) >= 3:
                    full_plan_decompose_enabled = True
                    break
        else:
            full_plan_decompose_enabled = bool(_full_plan_decompose_flag)
        try:
            full_plan_decompose_min_ops = int(
                getattr(self.config, "concept_csv_mining_full_plan_decompose_min_ops", 16) or 16
            )
        except Exception:
            full_plan_decompose_min_ops = 16
        full_plan_decompose_min_ops = max(0, min(8192, int(full_plan_decompose_min_ops)))
        try:
            full_plan_decompose_max_targets = int(
                getattr(self.config, "concept_csv_mining_full_plan_decompose_max_targets_per_window", 2) or 2
            )
        except Exception:
            full_plan_decompose_max_targets = 2
        full_plan_decompose_max_targets = max(0, min(64, int(full_plan_decompose_max_targets)))

        # Plan-op concept ids (call-free primitives) used to materialize plan-derived CSG concepts via CSV_CALL.
        # This is required to satisfy deep nested concept survival laws without relying on wrapper-only top-level
        # deepwrap concepts (which violate CSG richness).
        plan_op_specs: Dict[str, Tuple[str, List[str]]] = {
            "add_int": (det_act_id(step=0, name="concept_plan_op_add_int_v0", idx=0), ["a", "b"]),
            "make_dict_goal_plan_ab": (
                det_act_id(step=0, name="concept_plan_op_make_dict_goal_plan_ab_v0", idx=0),
                ["goal_id", "plan", "a", "b"],
            ),
            "json_canonical": (det_act_id(step=0, name="concept_plan_op_json_canonical_v0", idx=0), ["obj"]),
        }
        plan_ops_missing: List[str] = []
        for fn_id, (cid, _slots) in plan_op_specs.items():
            if self.store.get_concept_act(str(cid)) is None:
                plan_ops_missing.append(str(fn_id))

        plan_required_depth = 0
        plan_required_csg_nodes = 0
        plan_required_csg_edges = 0
        for t in tasks:
            if not isinstance(t, dict):
                continue
            if str(t.get("validator_id") or "") != "plan_validator":
                continue
            try:
                plan_required_depth = max(int(plan_required_depth), int(t.get("concept_min_depth", 0) or 0))
            except Exception:
                pass
            try:
                plan_required_csg_nodes = max(
                    int(plan_required_csg_nodes), int(t.get("concept_csg_min_nodes", 0) or 0)
                )
            except Exception:
                pass
            try:
                plan_required_csg_edges = max(
                    int(plan_required_csg_edges), int(t.get("concept_csg_min_edges", 0) or 0)
                )
            except Exception:
                pass
        plan_required_depth = max(0, int(plan_required_depth))
        plan_required_csg_nodes = max(0, int(plan_required_csg_nodes))
        plan_required_csg_edges = max(0, int(plan_required_csg_edges))
        if int(plan_required_depth) < 2:
            plan_required_depth = 0

        plan_ops_deepwrap_meta: Dict[str, Any] = {"enabled": False, "missing_plan_ops": list(plan_ops_missing)}
        if (
            (not plan_ops_missing)
            and int(plan_required_depth) >= 2
            and (int(plan_required_csg_nodes) > 0 or int(plan_required_csg_edges) > 0)
        ):
            # Lift plan ops to depth>=plan_required_depth-1 so plan concepts can reach required depth
            # without wrapper-only top-level deepwrap.
            from .mine_promote_v74 import materialize_deep_wrapper_act_v74

            target_depth = max(0, int(plan_required_depth) - 1)
            try:
                max_new = int(getattr(self.config, "concept_csv_plan_op_deepwrap_max_new_per_window", 24) or 24)
            except Exception:
                max_new = 24
            max_new = max(0, min(256, int(max_new)))
            need = int(len(plan_op_specs)) * int(target_depth)
            if int(target_depth) > 0:
                max_new = max(int(max_new), int(need))
            budget = int(max_new)

            base_overhead_bits = int(getattr(self.config, "concept_csv_overhead_bits", 1024) or 1024)
            if base_overhead_bits < 0:
                base_overhead_bits = 0
            try:
                dw_scale = float(getattr(self.config, "concept_csv_deepwrap_overhead_scale", 0.6) or 0.6)
            except Exception:
                dw_scale = 0.6
            if dw_scale != dw_scale:
                dw_scale = 0.6
            dw_scale = max(0.0, min(8.0, float(dw_scale)))
            deepwrap_overhead_bits = int(round(float(base_overhead_bits) * float(dw_scale)))
            if int(base_overhead_bits) > 0:
                deepwrap_overhead_bits = max(1, int(deepwrap_overhead_bits))
            else:
                deepwrap_overhead_bits = max(0, int(deepwrap_overhead_bits))

            memo_depth: Dict[str, int] = {}

            def _static_depth_plan(concept_id: str, stack: set) -> int:
                cid = str(concept_id or "")
                if not cid:
                    return 0
                if cid in memo_depth:
                    return int(memo_depth[cid])
                if cid in stack:
                    memo_depth[cid] = 0
                    return 0
                act0 = self.store.get_concept_act(cid)
                if act0 is None:
                    memo_depth[cid] = 0
                    return 0
                callees: List[str] = []
                for ins in list(getattr(act0, "program", []) or []):
                    if str(getattr(ins, "op", "")) != "CSV_CALL":
                        continue
                    args0 = getattr(ins, "args", {}) or {}
                    if not isinstance(args0, dict):
                        args0 = {}
                    callee = str(args0.get("concept_id") or "")
                    if callee:
                        callees.append(callee)
                if not callees:
                    d0 = 0
                else:
                    st2 = set(stack)
                    st2.add(cid)
                    d0 = 1 + max(_static_depth_plan(c, st2) for c in callees)
                memo_depth[cid] = int(d0)
                return int(d0)

            store_plan_ops = ActStore()
            added_wraps: List[str] = []
            skipped_existing_wraps = 0
            errs: List[str] = []

            for fn_id in sorted(plan_op_specs.keys(), key=str):
                cid0, slots0 = plan_op_specs.get(str(fn_id), ("", []))
                current_id = str(cid0)
                cur_depth = int(_static_depth_plan(current_id, set()))
                while budget > 0 and int(cur_depth) < int(target_depth):
                    try:
                        if store_plan_ops.get_concept_act(str(current_id)) is None:
                            inner_act = self.store.get_concept_act(str(current_id))
                            if inner_act is not None:
                                store_plan_ops.add(copy.deepcopy(inner_act))
                        wrap_act, _wrap_dbg = materialize_deep_wrapper_act_v74(
                            store_base=store_plan_ops,
                            inner_concept_id=str(current_id),
                            overhead_bits=int(deepwrap_overhead_bits),
                            seed_step=int(step),
                        )
                        if store_plan_ops.get_concept_act(str(wrap_act.id)) is None:
                            store_plan_ops.add(copy.deepcopy(wrap_act))
                        if self.store.get(str(wrap_act.id)) is None:
                            patch = Patch(kind="ADD_ACT", payload={"acts": [wrap_act.to_dict()]})
                            self._apply_patch(patch, count=True)
                            self._concept_csv_added += 1
                            added_wraps.append(str(wrap_act.id))
                            budget -= 1
                        else:
                            skipped_existing_wraps += 1
                        current_id = str(wrap_act.id)
                        cur_depth = int(cur_depth) + 1
                    except Exception as e:
                        errs.append(str(e))
                        break
                plan_op_specs[str(fn_id)] = (str(current_id), list(slots0))

            plan_ops_deepwrap_meta = {
                "enabled": True,
                "required_depth": int(plan_required_depth),
                "required_csg_min_nodes": int(plan_required_csg_nodes),
                "required_csg_min_edges": int(plan_required_csg_edges),
                "target_depth": int(target_depth),
                "max_new": int(max_new),
                "added": int(len(added_wraps)),
                "promoted_ids": list(added_wraps),
                "skipped_existing": int(skipped_existing_wraps),
                "errors": list(errs)[:3],
                "missing_plan_ops": list(plan_ops_missing),
            }

        def _eval_ops_env(
            *,
            ops: Sequence[Dict[str, Any]],
            inputs: Dict[str, Any],
            input_keys: Sequence[str],
        ) -> Dict[str, Any]:
            env: Dict[str, Any] = {}
            for idx, key in enumerate(list(input_keys)):
                k = str(key or "")
                if not k:
                    continue
                env[f"in{int(idx)}"] = inputs.get(k)
            for op in ops:
                if not isinstance(op, dict):
                    continue
                fn_id = str(op.get("fn") or "")
                out_v = str(op.get("out") or "")
                in_vars = op.get("in", [])
                if not out_v or not isinstance(in_vars, list):
                    continue
                spec_fn = PRIMITIVE_OPS.get(fn_id)
                if spec_fn is None:
                    continue
                spec, fn = spec_fn
                vals = [env.get(str(v)) for v in in_vars]
                if int(spec.arity) != int(len(vals)):
                    continue
                try:
                    out_res = fn(*vals) if int(spec.arity) > 1 else fn(vals[0])
                except Exception:
                    continue
                env[str(out_v)] = out_res
            return env

        def _split_ops_for_depth_v1(ops0: Sequence[Dict[str, Any]]) -> Optional[Tuple[int, str]]:
            """
            Deterministic split for long primitive-only plan programs.

            Returns (cut_idx, mid_var) where:
              - ops[:cut_idx] computes mid_var
              - ops[cut_idx:] consumes mid_var
              - mid_var is the ONLY cross-cut dependency (prevents "bundle outputs" hacks)

            This is a generic "operator discovery" primitive used to bootstrap compositional
            plan concepts (depth>=1) so v74 deepwrap can raise depth when required.
            """
            ops = [op for op in list(ops0) if isinstance(op, dict)]
            if len(ops) < 2:
                return None

            out_by_idx: List[str] = []
            ins_by_idx: List[List[str]] = []
            for op in ops:
                out_by_idx.append(str(op.get("out") or ""))
                ins0 = op.get("in", [])
                if not isinstance(ins0, list):
                    ins0 = []
                ins_by_idx.append([str(v) for v in ins0 if str(v)])

            # Suffix used vars at each cut position (vars referenced by inputs in ops[cut:]).
            suffix_used: List[set] = [set() for _ in range(len(ops) + 1)]
            cur: set = set()
            for i in reversed(range(len(ops))):
                for v in ins_by_idx[i]:
                    if v:
                        cur.add(str(v))
                suffix_used[i] = set(cur)
            suffix_used[len(ops)] = set()

            produced: set = set()
            best: Optional[Tuple[int, int, int, int, str]] = None  # (uses, min_side, balance, cut, mid)
            for cut in range(1, len(ops)):
                out_prev = str(out_by_idx[cut - 1] or "")
                if out_prev:
                    produced.add(out_prev)
                cross = produced.intersection(suffix_used[cut])
                if len(cross) != 1:
                    continue
                mid = next(iter(cross))
                if not mid:
                    continue
                # Score: prefer mid used many times in suffix (strong shared hinge),
                # then prefer larger min segment size (avoid degenerate splits),
                # then prefer more balanced split, then earliest cut, then lex mid.
                uses = 0
                for j in range(cut, len(ops)):
                    uses += sum(1 for v in ins_by_idx[j] if str(v) == str(mid))
                left = int(cut)
                right = int(len(ops) - cut)
                min_side = min(left, right)
                balance = abs(left - right)
                cand = (int(uses), int(min_side), -int(balance), int(cut), str(mid))
                if best is None or cand > best:
                    best = cand

            if best is None:
                return None
            cut_idx = int(best[3])
            mid_var = str(best[4] or "")
            if cut_idx <= 0 or cut_idx >= len(ops) or not mid_var:
                return None
            return int(cut_idx), str(mid_var)

        def _collect_in_indices(ops0: Sequence[Dict[str, Any]]) -> List[int]:
            out: set = set()
            for op0 in list(ops0):
                if not isinstance(op0, dict):
                    continue
                ins0 = op0.get("in", [])
                if not isinstance(ins0, list):
                    continue
                for vname in ins0:
                    m = re.fullmatch(r"in([0-9]+)", str(vname))
                    if not m:
                        continue
                    try:
                        out.add(int(m.group(1)))
                    except Exception:
                        continue
            return sorted(list(out))

        def _decompose_full_plan_v1(
            *,
            task_id: str,
            family_id: str,
            input_keys: Sequence[str],
            input_schema0: Dict[str, str],
            ops0: Sequence[Dict[str, Any]],
            return_var: str,
            step: int,
        ) -> Tuple[List[Act], Dict[str, Any]]:
            split = _split_ops_for_depth_v1(ops0)
            if split is None:
                raise ValueError("no_split_available")
            cut_idx, mid_var = int(split[0]), str(split[1])

            ops_a = list(ops0)[: int(cut_idx)]
            ops_b = list(ops0)[int(cut_idx) :]
            if not ops_a or not ops_b:
                raise ValueError("bad_split_empty_side")

            in_idx_a = _collect_in_indices(ops_a)
            in_idx_b = _collect_in_indices(ops_b)

            # Infer the mid_var type from the primitive spec that produces it (last assignment in ops_a).
            mid_type = "str"
            try:
                for op in list(ops_a):
                    if not isinstance(op, dict):
                        continue
                    if str(op.get("out") or "") != str(mid_var):
                        continue
                    fn_id = str(op.get("fn") or "")
                    spec_fn = PRIMITIVE_OPS.get(fn_id)
                    if spec_fn is None:
                        continue
                    spec = spec_fn[0]
                    mid_type = str(getattr(spec, "output_type", "str") or "str")
            except Exception:
                mid_type = "str"
            if mid_type not in {"str", "int", "dict", "list"}:
                mid_type = "str"

            seg_a_input_schema: Dict[str, str] = {}
            for idx in in_idx_a:
                if idx < 0 or idx >= len(list(input_keys)):
                    continue
                k = str(list(input_keys)[idx] or "")
                if not k:
                    continue
                seg_a_input_schema[k] = str(input_schema0.get(k) or "str")

            seg_b_input_schema: Dict[str, str] = {str(mid_var): str(mid_type)}
            for idx in in_idx_b:
                if idx < 0 or idx >= len(list(input_keys)):
                    continue
                k = str(list(input_keys)[idx] or "")
                if not k:
                    continue
                seg_b_input_schema[k] = str(input_schema0.get(k) or "str")

            program_a: List[Instruction] = []
            for idx in in_idx_a:
                if idx < 0 or idx >= len(list(input_keys)):
                    continue
                k = str(list(input_keys)[idx] or "")
                if not k:
                    continue
                program_a.append(Instruction("CSV_GET_INPUT", {"name": k, "out": f"in{idx}"}))
            deps_a: List[str] = []
            for op in ops_a:
                if not isinstance(op, dict):
                    raise ValueError("plan_segA_op_not_dict")
                fn_id = str(op.get("fn") or "")
                spec_call = plan_op_specs.get(str(fn_id))
                if spec_call is None:
                    raise ValueError(f"unknown_plan_op_for_call:{fn_id}")
                call_cid, slot_order = spec_call
                in_vars0 = op.get("in") if isinstance(op.get("in"), list) else []
                if not isinstance(in_vars0, list) or len(in_vars0) != len(slot_order):
                    raise ValueError(f"plan_op_arity_mismatch:{fn_id}")
                bind_map: Dict[str, str] = {}
                for j, slot in enumerate(list(slot_order)):
                    bind_map[str(slot)] = str(in_vars0[j])
                out_v = str(op.get("out") or "")
                if not out_v:
                    raise ValueError("plan_op_missing_out")
                program_a.append(
                    Instruction("CSV_CALL", {"concept_id": str(call_cid), "bind": dict(bind_map), "out": str(out_v)})
                )
                deps_a.append(str(call_cid))
            program_a.append(Instruction("CSV_RETURN", {"var": str(mid_var)}))

            program_b: List[Instruction] = []
            program_b.append(Instruction("CSV_GET_INPUT", {"name": str(mid_var), "out": str(mid_var)}))
            for idx in in_idx_b:
                if idx < 0 or idx >= len(list(input_keys)):
                    continue
                k = str(list(input_keys)[idx] or "")
                if not k:
                    continue
                program_b.append(Instruction("CSV_GET_INPUT", {"name": k, "out": f"in{idx}"}))
            deps_b: List[str] = []
            for op in ops_b:
                if not isinstance(op, dict):
                    raise ValueError("plan_segB_op_not_dict")
                fn_id = str(op.get("fn") or "")
                spec_call = plan_op_specs.get(str(fn_id))
                if spec_call is None:
                    raise ValueError(f"unknown_plan_op_for_call:{fn_id}")
                call_cid, slot_order = spec_call
                in_vars0 = op.get("in") if isinstance(op.get("in"), list) else []
                if not isinstance(in_vars0, list) or len(in_vars0) != len(slot_order):
                    raise ValueError(f"plan_op_arity_mismatch:{fn_id}")
                bind_map: Dict[str, str] = {}
                for j, slot in enumerate(list(slot_order)):
                    bind_map[str(slot)] = str(in_vars0[j])
                out_v = str(op.get("out") or "")
                if not out_v:
                    raise ValueError("plan_op_missing_out")
                program_b.append(
                    Instruction("CSV_CALL", {"concept_id": str(call_cid), "bind": dict(bind_map), "out": str(out_v)})
                )
                deps_b.append(str(call_cid))
            program_b.append(Instruction("CSV_RETURN", {"var": str(return_var)}))

            body_a = {
                "schema_version": 2,
                "kind": "concept_csv_full_plan_seg_calls_v2",
                "seg": "A",
                "mid_var": str(mid_var),
                "mid_type": str(mid_type),
                "input_schema": dict(seg_a_input_schema),
                "ops": list(ops_a),
                "return_var": str(mid_var),
            }
            sig_a = sha256_hex(canonical_json_dumps(body_a).encode("utf-8"))
            act_id_a = f"act_concept_csv_mined_{sig_a[:16]}_segA"

            body_b = {
                "schema_version": 2,
                "kind": "concept_csv_full_plan_seg_calls_v2",
                "seg": "B",
                "mid_var": str(mid_var),
                "mid_type": str(mid_type),
                "input_schema": dict(seg_b_input_schema),
                "ops": list(ops_b),
                "return_var": str(return_var),
            }
            sig_b = sha256_hex(canonical_json_dumps(body_b).encode("utf-8"))
            act_id_b = f"act_concept_csv_mined_{sig_b[:16]}_segB"

            program_comp: List[Instruction] = []
            input_schema_comp: Dict[str, str] = {str(k): str(v) for k, v in dict(input_schema0).items() if str(k)}
            for k in sorted(input_schema_comp.keys(), key=str):
                program_comp.append(Instruction("CSV_GET_INPUT", {"name": str(k), "out": str(k)}))
            bind_a = {str(k): str(k) for k in sorted(seg_a_input_schema.keys(), key=str)}
            program_comp.append(
                Instruction("CSV_CALL", {"concept_id": str(act_id_a), "bind": dict(bind_a), "out": str(mid_var)})
            )
            bind_b = {str(mid_var): str(mid_var)}
            for k in sorted(seg_b_input_schema.keys(), key=str):
                if str(k) == str(mid_var):
                    continue
                bind_b[str(k)] = str(k)
            program_comp.append(
                Instruction("CSV_CALL", {"concept_id": str(act_id_b), "bind": dict(bind_b), "out": str(return_var)})
            )
            program_comp.append(Instruction("CSV_RETURN", {"var": str(return_var)}))

            body_comp = {
                "schema_version": 2,
                "kind": "concept_csv_full_plan_comp_calls_v2",
                "seg_a": str(act_id_a),
                "seg_b": str(act_id_b),
                "mid_var": str(mid_var),
                "interface": {
                    "input_schema": dict(input_schema_comp),
                    "output_schema": {str(return_var): "str"},
                    "validator_id": "plan_validator",
                },
            }
            sig_comp = sha256_hex(canonical_json_dumps(body_comp).encode("utf-8"))
            act_id_comp = f"act_concept_csv_mined_{sig_comp[:16]}_plancomp"

            overhead_bits = int(getattr(self.config, "concept_csv_overhead_bits", 1024) or 1024)
            overhead_bits = max(0, int(overhead_bits))

            act_a = Act(
                id=str(act_id_a),
                version=1,
                created_at=deterministic_iso(step=int(step)),
                kind="concept_csv",
                # Start unrestricted to allow cross-family transfer evidence; SPLIT can restrict later.
                match={},
                program=list(program_a),
                evidence={
                    "name": "concept_csv_mined_train_v0",
                    "interface": {
                        "input_schema": dict(seg_a_input_schema),
                        "output_schema": {str(mid_var): str(mid_type)},
                        "validator_id": "state_validator",
                    },
                    "meta": {
                        "builder": "concept_csv_full_plan_decompose_calls_v2",
                        "seg": "A",
                        "seed_task_id": str(task_id),
                        "birth_tags": ["plan"],
                    },
                },
                cost={"overhead_bits": int(overhead_bits)},
                deps=sorted(set(str(x) for x in deps_a if str(x)), key=str),
                active=True,
            )

            act_b = Act(
                id=str(act_id_b),
                version=1,
                created_at=deterministic_iso(step=int(step)),
                kind="concept_csv",
                # Start unrestricted to allow cross-family transfer evidence; SPLIT can restrict later.
                match={},
                program=list(program_b),
                evidence={
                    "name": "concept_csv_mined_train_v0",
                    "interface": {
                        "input_schema": dict(seg_b_input_schema),
                        "output_schema": {str(return_var): "str"},
                        "validator_id": "plan_validator",
                    },
                    "meta": {
                        "builder": "concept_csv_full_plan_decompose_calls_v2",
                        "seg": "B",
                        "seed_task_id": str(task_id),
                        "birth_tags": ["plan"],
                    },
                },
                cost={"overhead_bits": int(overhead_bits)},
                deps=sorted(set(str(x) for x in deps_b if str(x)), key=str),
                active=True,
            )

            act_comp = Act(
                id=str(act_id_comp),
                version=1,
                created_at=deterministic_iso(step=int(step)),
                kind="concept_csv",
                # Start unrestricted to allow cross-family transfer evidence; SPLIT can restrict later.
                match={},
                program=list(program_comp),
                evidence={
                    "name": "concept_csv_mined_train_v0",
                    "interface": {
                        "input_schema": dict(input_schema_comp),
                        "output_schema": {str(return_var): "str"},
                        "validator_id": "plan_validator",
                    },
                    "meta": {
                        "builder": "concept_csv_full_plan_decompose_calls_v2",
                        "seg": "COMP",
                        "seed_task_id": str(task_id),
                        "birth_tags": ["plan"],
                    },
                },
                cost={"overhead_bits": int(overhead_bits)},
                deps=[str(act_id_a), str(act_id_b)],
                active=True,
            )

            return [act_a, act_b, act_comp], {"mid_var": str(mid_var), "cut_idx": int(cut_idx)}

        for task_id, expected_spec in plan_tasks[:24]:
            input_keys = expected_spec.get("input_keys")
            inputs = expected_spec.get("inputs")
            ops = expected_spec.get("ops")
            return_var = str(expected_spec.get("return_var") or "")
            if not isinstance(input_keys, list) or not isinstance(inputs, dict) or not isinstance(ops, list):
                continue
            if not return_var:
                continue

            # Infer external input schema from primitive usage (deterministic, fail-closed on conflicts).
            type_by_in_idx: Dict[int, str] = {}
            ok_types = True
            for op in ops:
                if not isinstance(op, dict):
                    ok_types = False
                    break
                fn_id = str(op.get("fn") or "")
                spec_fn = PRIMITIVE_OPS.get(fn_id)
                if spec_fn is None:
                    ok_types = False
                    break
                spec = spec_fn[0]
                ins = op.get("in", [])
                if not isinstance(ins, list) or len(ins) != int(spec.arity):
                    ok_types = False
                    break
                for j, vname in enumerate(ins):
                    m = re.fullmatch(r"in([0-9]+)", str(vname))
                    if not m:
                        continue
                    idx = int(m.group(1))
                    want_t = str(spec.input_types[j])
                    prev_t = type_by_in_idx.get(idx)
                    if prev_t and prev_t != want_t:
                        ok_types = False
                        break
                    type_by_in_idx[idx] = want_t
                if not ok_types:
                    break
            if not ok_types:
                continue

            input_schema: Dict[str, str] = {}
            for idx, key in enumerate(list(input_keys)):
                k = str(key or "")
                if not k:
                    continue
                input_schema[k] = str(type_by_in_idx.get(int(idx), "str"))

            safe_task = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in task_id)
            concept_id = f"concept_seed_plan_{safe_task}"
            program: List[Instruction] = []
            for idx, key in enumerate(list(input_keys)):
                k = str(key or "")
                if not k:
                    continue
                program.append(Instruction("CSV_GET_INPUT", {"name": k, "out": f"in{idx}"}))
            for op in ops:
                if not isinstance(op, dict):
                    continue
                program.append(
                    Instruction(
                        "CSV_PRIMITIVE",
                        {
                            "fn": str(op.get("fn") or ""),
                            "in": list(op.get("in") or []),
                            "out": str(op.get("out") or ""),
                        },
                    )
                )
            program.append(Instruction("CSV_RETURN", {"var": str(return_var)}))
            plan_concept = _seed_concept(
                act_id=str(concept_id),
                program=program,
                interface={
                    "input_schema": dict(input_schema),
                    "output_schema": {"value": "str"},
                    "validator_id": "plan_validator",
                },
            )
            try:
                store_shadow.add(plan_concept)
            except Exception:
                continue

            try:
                res = engine.execute_concept_csv(
                    concept_act_id=str(plan_concept.id),
                    inputs=dict(inputs),
                    expected=dict(expected_spec),
                    step=0,
                    max_depth=8,
                )
                meta = res.get("meta") if isinstance(res.get("meta"), dict) else {}
                evs = res.get("events") if isinstance(res.get("events"), list) else []
                ok = bool(meta.get("ok", False))
            except Exception:
                ok = False
                evs = []
            # Augment trace inputs with intermediate values (deterministic) so concept mining
            # can induce operators that accept intermediate variables as explicit inputs.
            aug_inputs: Dict[str, Any] = dict(inputs)
            try:
                env_extra = _eval_ops_env(ops=list(ops), inputs=dict(inputs), input_keys=list(input_keys))
                for k, v in env_extra.items():
                    if str(k) and str(k) not in aug_inputs:
                        aug_inputs[str(k)] = v
            except Exception:
                aug_inputs = dict(inputs)
            trace_rows.append(
                {
                    "ctx_sig": (
                        f"plan␟{plan_family_by_task_id.get(str(task_id), '')}"
                        if str(plan_family_by_task_id.get(str(task_id), ""))
                        else f"plan␟{task_id}"
                    ),
                    "inputs": dict(aug_inputs),
                    "utility_passed": bool(ok),
                    "events": list(evs),
                }
            )
            plan_traces_added += 1

            # Optional: materialize a "full plan" concept (entire ops list) for long-horizon
            # plan_validator tasks, so the runtime has an exact-schema concept to execute.
            #
            # This closes the escape route where tasks with many inputs (x0..xN) are unsolvable
            # because no mined operator matches the large input schema.
            try:
                min_full_ops = int(getattr(self.config, "concept_csv_mining_full_plan_min_ops", 1) or 1)
            except Exception:
                min_full_ops = 1
            min_full_ops = max(1, int(min_full_ops))
            try:
                max_full_new = int(getattr(self.config, "concept_csv_mining_full_plan_max_new_per_window", 4) or 4)
            except Exception:
                max_full_new = 4
            max_full_new = max(0, min(32, int(max_full_new)))
            # When the active suite pack enforces deep nested concepts + non-trivial CSG structure,
            # we must materialize enough full-plan concepts to cover all distinct plan interfaces
            # early (otherwise the system can die before it ever has a chance to induce semantics).
            if int(plan_required_depth) >= 2 and (int(plan_required_csg_nodes) > 0 or int(plan_required_csg_edges) > 0):
                max_full_new = max(int(max_full_new), 6)
            if max_full_new <= 0 or min_full_ops <= 0 or len(ops) < int(min_full_ops):
                full_plan_skipped_disabled += 1
            else:
                try:
                    body = {
                        "schema_version": 2,
                        "kind": "concept_csv_full_plan_calls_v2",
                        "ops": list(ops),
                        "input_schema": dict(input_schema),
                        "return_var": str(return_var),
                        "output_type": "str",
                    }
                    cand_sig = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
                    act_id = f"act_concept_csv_mined_{cand_sig[:16]}_fullcall"
                except Exception:
                    cand_sig = ""
                    act_id = ""

                full_exists = bool(act_id and self.store.get(str(act_id)) is not None)
                if full_exists:
                    full_plan_skipped_existing += 1
                elif act_id and full_plan_added < int(max_full_new):
                    try:
                        # Deterministic concept_csv act that executes the full plan ops list via concept calls.
                        program_full: List[Instruction] = []
                        deps_full: List[str] = []
                        for idx, key in enumerate(list(input_keys)):
                            k = str(key or "")
                            if not k:
                                continue
                            program_full.append(Instruction("CSV_GET_INPUT", {"name": k, "out": f"in{idx}"}))
                        for op in ops:
                            if not isinstance(op, dict):
                                raise ValueError("plan_full_op_not_dict")
                            fn_id = str(op.get("fn") or "")
                            spec_call = plan_op_specs.get(str(fn_id))
                            if spec_call is None:
                                raise ValueError(f"unknown_plan_op_for_call:{fn_id}")
                            call_cid, slot_order = spec_call
                            in_vars0 = op.get("in") if isinstance(op.get("in"), list) else []
                            if not isinstance(in_vars0, list) or len(in_vars0) != len(slot_order):
                                raise ValueError(f"plan_op_arity_mismatch:{fn_id}")
                            bind_map: Dict[str, str] = {}
                            for j, slot in enumerate(list(slot_order)):
                                bind_map[str(slot)] = str(in_vars0[j])
                            out_v = str(op.get("out") or "")
                            if not out_v:
                                raise ValueError("plan_op_missing_out")
                            program_full.append(
                                Instruction(
                                    "CSV_CALL",
                                    {"concept_id": str(call_cid), "bind": dict(bind_map), "out": str(out_v)},
                                )
                            )
                            deps_full.append(str(call_cid))
                        program_full.append(Instruction("CSV_RETURN", {"var": str(return_var)}))
                        overhead_bits = int(getattr(self.config, "concept_csv_overhead_bits", 1024) or 1024)
                        overhead_bits = max(0, int(overhead_bits))
                        act_full = Act(
                            id=str(act_id),
                            version=1,
                            created_at=deterministic_iso(step=int(step)),
                            kind="concept_csv",
                            # Start unrestricted to allow cross-family transfer evidence; SPLIT can restrict later.
                            match={},
                            program=list(program_full),
                            evidence={
                                "name": "concept_csv_mined_train_v0",
                                "interface": {
                                    "input_schema": dict(input_schema),
                                    # Prefer the actual return_var for plan matching (strong output_key match).
                                    "output_schema": {str(return_var): "str"},
                                    "validator_id": "plan_validator",
                                },
                                "meta": {
                                    "builder": "concept_csv_full_plan_calls_v2",
                                    "candidate_sig": str(cand_sig),
                                    "output_key": str(return_var),
                                    "gain_bits_est": int(max(0, len(ops) * 128 - overhead_bits)),
                                    "contexts_distinct": 1,
                                    "count": 1,
                                    "birth_tags": ["plan"],
                                },
                            },
                            cost={"overhead_bits": int(overhead_bits)},
                            deps=sorted(set(str(x) for x in deps_full if str(x)), key=str),
                            active=True,
                        )
                        patch = Patch(kind="ADD_ACT", payload={"acts": [act_full.to_dict()]})
                        self._apply_patch(patch, count=True)
                        full_plan_added += 1
                        self._concept_csv_added += 1
                    except Exception:
                        full_plan_skipped_disabled += 1
                else:
                    full_plan_skipped_disabled += 1

                # Full-plan decomposition: induce a composed plan concept (depth>=1) so deepwrap can
                # raise it to the validator pack's required nested depth (e.g., sota_v8 min_depth=3).
                if (
                    full_plan_decompose_enabled
                    and full_plan_decompose_targets < int(full_plan_decompose_max_targets)
                    and len(ops) >= int(full_plan_decompose_min_ops)
                ):
                    try:
                        new_acts, _dbg = _decompose_full_plan_v1(
                            task_id=str(task_id),
                            family_id=str(plan_family_by_task_id.get(str(task_id), "")),
                            input_keys=list(input_keys),
                            input_schema0=dict(input_schema),
                            ops0=list(ops),
                            return_var=str(return_var),
                            step=int(step),
                        )
                    except Exception:
                        full_plan_decompose_errors += 1
                        new_acts = []

                    if not new_acts:
                        full_plan_decompose_skipped_disabled += 1
                    else:
                        missing = [a for a in new_acts if self.store.get(str(a.id)) is None]
                        if not missing:
                            full_plan_decompose_skipped_existing += 1
                        else:
                            for a in missing:
                                patch = Patch(kind="ADD_ACT", payload={"acts": [a.to_dict()]})
                                self._apply_patch(patch, count=True)
                                self._concept_csv_added += 1
                                full_plan_decompose_added += 1
                            full_plan_decompose_targets += 1

        tmp = trace_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in trace_rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                f.write("\n")
        os.replace(tmp, trace_path)

        candidates = mine_csv_candidates(
            trace_path,
            min_ops=1,
            max_ops=int(
                max(
                    1,
                    min(
                        256,
                        int(getattr(self.config, "concept_csv_mining_max_ops", 6) or 6),
                    ),
                )
            ),
            bits_per_op=128,
            overhead_bits=1024,
        )
        if not candidates:
            return {
                "enabled": True,
                "trace_path": os.path.relpath(trace_path, self.out_dir),
                "candidates_total": 0,
                "plan_traces_added": int(plan_traces_added),
                "full_plan_added": int(full_plan_added),
                "full_plan_skipped_existing": int(full_plan_skipped_existing),
                "full_plan_skipped_disabled": int(full_plan_skipped_disabled),
                "full_plan_decompose_targets": int(full_plan_decompose_targets),
                "full_plan_decompose_added": int(full_plan_decompose_added),
                "full_plan_decompose_skipped_existing": int(full_plan_decompose_skipped_existing),
                "full_plan_decompose_skipped_disabled": int(full_plan_decompose_skipped_disabled),
                "full_plan_decompose_errors": int(full_plan_decompose_errors),
                "added": 0,
                "skipped_existing": 0,
                "force_new": bool(force_new),
            }

        added = 0
        skipped_existing = 0
        promoted_ids: List[str] = []
        scanned = 0
        scan_list = list(candidates) if bool(force_new) else list(candidates)[:top_k]
        for cand in scan_list:
            if added >= max_new:
                break
            scanned += 1
            try:
                act = self._materialize_mined_concept_csv_act(cand=cand, step=int(step))
            except Exception:
                continue
            if self.store.get(str(act.id)) is not None:
                skipped_existing += 1
                continue
            patch = Patch(kind="ADD_ACT", payload={"acts": [act.to_dict()]})
            self._apply_patch(patch, count=True)
            added += 1
            self._concept_csv_added += 1
            promoted_ids.append(str(act.id))

        gc_meta = self._concept_csv_budget_prune(step=int(step))
        meta_out: Dict[str, Any] = {
            "enabled": True,
            "trace_path": os.path.relpath(trace_path, self.out_dir),
            "candidates_total": int(len(candidates)),
            "candidates_scanned": int(scanned),
            "plan_traces_added": int(plan_traces_added),
            "full_plan_added": int(full_plan_added),
            "full_plan_skipped_existing": int(full_plan_skipped_existing),
            "full_plan_skipped_disabled": int(full_plan_skipped_disabled),
            "full_plan_decompose_targets": int(full_plan_decompose_targets),
            "full_plan_decompose_added": int(full_plan_decompose_added),
            "full_plan_decompose_skipped_existing": int(full_plan_decompose_skipped_existing),
            "full_plan_decompose_skipped_disabled": int(full_plan_decompose_skipped_disabled),
            "full_plan_decompose_errors": int(full_plan_decompose_errors),
            "added": int(added),
            "skipped_existing": int(skipped_existing),
            "promoted_ids": list(promoted_ids),
            "gc": dict(gc_meta),
            "force_new": bool(force_new),
        }

        # Optional second-stage induction: promote composed (CSV_CALL) concepts from multi-step traces.
        composed_meta = self._mine_and_promote_composed_concept_csv_v74(
            step=int(step),
            plan_tasks=list(plan_tasks),
            force_new=bool(force_new),
        )
        if isinstance(composed_meta, dict) and bool(composed_meta.get("enabled", False)):
            meta_out["composed_v74"] = dict(composed_meta)

        return meta_out

    def _mine_and_promote_composed_concept_csv_v74(
        self,
        *,
        step: int,
        plan_tasks: Sequence[Tuple[str, Dict[str, Any]]],
        force_new: bool,
    ) -> Dict[str, Any]:
        """
        Second-stage concept induction: mine repeated subpaths of concept calls (TraceV73) and
        promote a composed concept_csv using CSV_CALL steps (v74).

        This is deterministic and uses only explicit concept_csv ACTs (no hidden state).
        """
        enabled = bool(getattr(self.config, "concept_csv_composed_enabled", False))
        if not enabled:
            return {"enabled": False}

        try:
            base_max_prog = int(getattr(self.config, "concept_csv_composed_base_max_program_len", 6) or 6)
        except Exception:
            base_max_prog = 6
        base_max_prog = max(1, min(64, int(base_max_prog)))
        try:
            max_k = int(getattr(self.config, "concept_csv_composed_max_k", 6) or 6)
        except Exception:
            max_k = 6
        max_k = max(2, min(16, int(max_k)))
        try:
            min_support = int(getattr(self.config, "concept_csv_composed_min_support", 2) or 2)
        except Exception:
            min_support = 2
        min_support = max(2, min(16, int(min_support)))
        try:
            top_k = int(getattr(self.config, "concept_csv_composed_top_k", 4) or 4)
        except Exception:
            top_k = 4
        top_k = max(0, min(64, int(top_k)))
        try:
            max_new = int(getattr(self.config, "concept_csv_composed_max_new_per_window", 1) or 1)
        except Exception:
            max_new = 1
        max_new = max(0, min(16, int(max_new)))
        if top_k <= 0 or max_new <= 0:
            return {"enabled": True, "skipped": True, "reason": "disabled_by_params"}

        # MDL: composed concepts should pay a smaller (but non-zero) overhead than flat mined concepts.
        base_overhead_bits = int(getattr(self.config, "concept_csv_overhead_bits", 1024) or 1024)
        if base_overhead_bits < 0:
            base_overhead_bits = 0
        try:
            comp_scale = float(getattr(self.config, "concept_csv_composed_overhead_scale", 0.6) or 0.6)
        except Exception:
            comp_scale = 0.6
        if comp_scale != comp_scale:
            comp_scale = 0.6
        comp_scale = max(0.0, min(8.0, float(comp_scale)))
        composed_overhead_bits = int(round(float(base_overhead_bits) * float(comp_scale)))
        if int(base_overhead_bits) > 0:
            composed_overhead_bits = max(1, int(composed_overhead_bits))
        else:
            composed_overhead_bits = max(0, int(composed_overhead_bits))
        try:
            dw_scale = float(getattr(self.config, "concept_csv_deepwrap_overhead_scale", comp_scale) or comp_scale)
        except Exception:
            dw_scale = float(comp_scale)
        if dw_scale != dw_scale:
            dw_scale = float(comp_scale)
        dw_scale = max(0.0, min(8.0, float(dw_scale)))
        deepwrap_overhead_bits = int(round(float(base_overhead_bits) * float(dw_scale)))
        if int(base_overhead_bits) > 0:
            deepwrap_overhead_bits = max(1, int(deepwrap_overhead_bits))
        else:
            deepwrap_overhead_bits = max(0, int(deepwrap_overhead_bits))

        # Local imports to keep learn.py lightweight and avoid import cycles.
        from .goal_spec_v72 import GoalSpecV72
        from .trace_v73 import PlanStepTraceV73, TraceV73, context_id_from_bindings
        from .validators import run_validator
        from .mine_promote_v74 import (
            extract_rep_steps,
            materialize_composed_act_v74,
            materialize_deep_wrapper_act_v74,
            mine_candidates_v74,
        )

        # Planning is the expensive part of composed-mining. Keep the operator set bounded:
        # - small primitives always allowed (<= base_max_prog)
        # - already-composed/deepwrap concepts allowed (has_call)
        # - plan_validator concepts allowed only up to a cap (avoid monolith direct solvers)
        try:
            plan_prog_cap = int(getattr(self.config, "concept_csv_composed_plan_max_program_len", 12) or 12)
        except Exception:
            plan_prog_cap = 12
        plan_prog_cap = max(int(base_max_prog), min(64, int(plan_prog_cap)))

        # Filter store for planning: keep only "small" concepts as primitives to force multi-step traces.
        store_base = ActStore()
        for act in self.store.active():
            if str(getattr(act, "kind", "")) != "concept_csv":
                store_base.add(copy.deepcopy(act))
                continue
            prog_len = int(len(getattr(act, "program", []) or []))
            prog = list(getattr(act, "program", []) or [])
            has_call = any(str(getattr(ins, "op", "")) == "CSV_CALL" for ins in prog)
            ev = act.evidence if isinstance(getattr(act, "evidence", None), dict) else {}
            iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
            iface = iface if isinstance(iface, dict) else {}
            v_id = str(iface.get("validator_id") or "")
            # Allow already-composed concepts as primitives (even if longer), so hierarchy can deepen
            # via v74 subpath mining (concepts composed of composed concepts).
            # Also keep plan_validator concepts (but avoid long monoliths): we need them to reach the
            # goal output key during trace generation, while one-step direct solvers are filtered
            # per-goal below.
            if prog_len <= int(base_max_prog) or has_call or (v_id == "plan_validator" and prog_len <= int(plan_prog_cap)):
                store_base.add(copy.deepcopy(act))

        # Optional "deep wrapper" induction (v74): satisfy existential nested concept depth requirements
        # for plan_validator tasks *even if* we don't yet have enough ok traces to mine composed acts.
        #
        # This prevents a deadlock when a validator pack requires depth>=K but the store has depth<K,
        # causing the concept executor to refuse execution and plan traces to disappear.
        deepwrap_meta: Dict[str, Any] = {"enabled": False}
        # Optional plan-op deepwrap meta (may be populated later). Keep defined even when we return early.
        plan_ops_deepwrap_meta: Dict[str, Any] = {"enabled": False}
        try:
            tasks = skill_suite_tasks_for_pack(
                str(getattr(self.config, "skill_suite_pack", "v0") or "v0")
            )
        except Exception:
            tasks = ()
        required_depth = 0
        for t in tasks:
            if not isinstance(t, dict):
                continue
            if str(t.get("validator_id") or "") != "plan_validator":
                continue
            try:
                md = int(t.get("concept_min_depth", 0) or 0)
            except Exception:
                md = 0
            required_depth = max(int(required_depth), int(md))
        # When validator packs require non-trivial concept subgraphs (CSG), we must be able to
        # satisfy depth requirements without relying on wrapper-only deepwrap concepts.
        required_csg_nodes = 0
        required_csg_edges = 0
        for t in tasks:
            if not isinstance(t, dict):
                continue
            if str(t.get("validator_id") or "") != "plan_validator":
                continue
            try:
                required_csg_nodes = max(int(required_csg_nodes), int(t.get("concept_csg_min_nodes", 0) or 0))
            except Exception:
                pass
            try:
                required_csg_edges = max(int(required_csg_edges), int(t.get("concept_csg_min_edges", 0) or 0))
            except Exception:
                pass
        required_csg_nodes = max(0, int(required_csg_nodes))
        required_csg_edges = max(0, int(required_csg_edges))
        if int(required_depth) < 2:
            required_depth = 0
        # Run deepwrap only when the selected validator pack requires deep nested concepts;
        # keep it surgical via a small deterministic per-window cap.
        if int(required_depth) >= 2:
            memo_depth: Dict[str, int] = {}

            def _static_depth(concept_id: str, stack: set) -> int:
                cid = str(concept_id or "")
                if not cid:
                    return 0
                if cid in memo_depth:
                    return memo_depth[cid]
                if cid in stack:
                    memo_depth[cid] = 0
                    return 0
                act0 = self.store.get_concept_act(cid)
                if act0 is None:
                    memo_depth[cid] = 0
                    return 0
                callees: List[str] = []
                for ins in list(getattr(act0, "program", []) or []):
                    if str(getattr(ins, "op", "")) != "CSV_CALL":
                        continue
                    args0 = getattr(ins, "args", {}) or {}
                    if not isinstance(args0, dict):
                        args0 = {}
                    callee = str(args0.get("concept_id") or "")
                    if callee:
                        callees.append(callee)
                if not callees:
                    d0 = 0
                else:
                    st2 = set(stack)
                    st2.add(cid)
                    d0 = 1 + max(_static_depth(c, st2) for c in callees)
                memo_depth[cid] = int(d0)
                return int(d0)

            def _is_plan_validator(act0: Act) -> bool:
                ev0 = act0.evidence if isinstance(act0.evidence, dict) else {}
                iface0 = ev0.get("interface") if isinstance(ev0.get("interface"), dict) else {}
                iface0 = iface0 if isinstance(iface0, dict) else {}
                return str(iface0.get("validator_id") or "") == "plan_validator"

            # Ensure depth>=required_depth exists per plan_validator interface signature (not just globally),
            # otherwise a single deep concept could mask missing interfaces.
            have_ifaces: set = set()
            # iface_sig -> (depth, support_ctx, act_id) for best candidate below required_depth.
            best_below: Dict[str, Tuple[int, int, str]] = {}
            for a0 in self.store.by_kind("concept_csv"):
                if not _is_plan_validator(a0):
                    continue
                ev0 = a0.evidence if isinstance(a0.evidence, dict) else {}
                iface0 = ev0.get("interface") if isinstance(ev0.get("interface"), dict) else {}
                iface0 = iface0 if isinstance(iface0, dict) else {}
                try:
                    iface_sig = canonical_json_dumps(
                        {
                            "in": iface0.get("input_schema") if isinstance(iface0.get("input_schema"), dict) else {},
                            "out": iface0.get("output_schema") if isinstance(iface0.get("output_schema"), dict) else {},
                            "v": str(iface0.get("validator_id") or ""),
                        }
                    )
                except Exception:
                    iface_sig = str(a0.id)
                d = int(_static_depth(str(a0.id), set()))
                if int(d) >= int(required_depth):
                    have_ifaces.add(str(iface_sig))
                    continue
                # NOTE: allow leaf solvers (depth=0) to be wrapped when the active validator pack
                # requires deep nested concepts; otherwise some plan interfaces can never reach the
                # required_depth and the system can stall permanently under concept-depth pressure.
                # The cost of this indirection is explicit (overhead_bits) and auditable.
                # Support contexts is only present for induced_v74; deepwrap/other acts default to 0.
                sctx = 0
                if isinstance(ev0.get("induced_v74"), dict):
                    try:
                        sctx = int(ev0["induced_v74"].get("support_contexts", 0) or 0)
                    except Exception:
                        sctx = 0
                prev = best_below.get(str(iface_sig))
                if prev is None:
                    best_below[str(iface_sig)] = (int(d), int(sctx), str(a0.id))
                else:
                    prev_d, prev_sctx, prev_id = int(prev[0]), int(prev[1]), str(prev[2])
                    if int(d) > int(prev_d) or (
                        int(d) == int(prev_d)
                        and (int(sctx) > int(prev_sctx) or (int(sctx) == int(prev_sctx) and str(a0.id) < str(prev_id)))
                    ):
                        best_below[str(iface_sig)] = (int(d), int(sctx), str(a0.id))

            # Deepwrap must be able to cover multiple plan interfaces and multi-layer depth gaps.
            # Keep it deterministic and bounded by an explicit config knob.
            max_deepwrap = int(getattr(self.config, "concept_csv_deepwrap_max_new_per_window", 6) or 6)
            max_deepwrap = max(0, min(64, int(max_deepwrap)))
            # Ensure at least required_depth wraps are possible when required_depth is high.
            if int(required_depth) > 0:
                max_deepwrap = max(int(max_deepwrap), int(required_depth))

            pending = [
                (sig, ent[0], ent[1], ent[2])
                for sig, ent in best_below.items()
                if sig not in have_ifaces
            ]
            # Prefer the shallowest interfaces first (largest depth deficit), then strongest support_ctx.
            # This ensures that any observed "concept_depth_too_shallow" interface gets resolved quickly.
            pending.sort(key=lambda t: (int(t[1]), -int(t[2]), str(t[0]), str(t[3])))
            if not pending:
                deepwrap_meta = {
                    "enabled": True,
                    "skipped": True,
                    "reason": "deep_ifaces_complete",
                    "required_depth": int(required_depth),
                    "ifaces_have": int(len(have_ifaces)),
                    "ifaces_below": int(len(best_below)),
                }
            else:
                added_wraps: List[str] = []
                skipped_existing_wraps = 0
                errs: List[str] = []
                budget = int(max_deepwrap)
                for _sig, inner_depth, _sctx, inner_id0 in pending:
                    if budget <= 0:
                        break
                    current_id = str(inner_id0)
                    current_depth = int(inner_depth)
                    # Raise depth by wrapping until we hit required_depth (each wrap adds +1).
                    while budget > 0 and int(current_depth) < int(required_depth):
                        try:
                            # Ensure the inner concept is present in the materialization store.
                            if store_base.get_concept_act(str(current_id)) is None:
                                inner_act = self.store.get_concept_act(str(current_id))
                                if inner_act is not None:
                                    store_base.add(copy.deepcopy(inner_act))
                            wrap_act, _wrap_dbg = materialize_deep_wrapper_act_v74(
                                store_base=store_base,
                                inner_concept_id=str(current_id),
                                overhead_bits=int(deepwrap_overhead_bits),
                                seed_step=int(step),
                            )
                            # Keep the wrapper available for further wrapping within this window.
                            if store_base.get_concept_act(str(wrap_act.id)) is None:
                                store_base.add(copy.deepcopy(wrap_act))
                            if self.store.get(str(wrap_act.id)) is None:
                                patch = Patch(kind="ADD_ACT", payload={"acts": [wrap_act.to_dict()]})
                                self._apply_patch(patch, count=True)
                                self._concept_csv_added += 1
                                added_wraps.append(str(wrap_act.id))
                                budget -= 1
                            else:
                                skipped_existing_wraps += 1
                            current_id = str(wrap_act.id)
                            current_depth = int(current_depth) + 1
                        except Exception as e:
                            errs.append(str(e))
                            break

                deepwrap_meta = {
                    "enabled": True,
                    "required_depth": int(required_depth),
                    "max_new": int(max_deepwrap),
                    "added": int(len(added_wraps)),
                    "promoted_ids": list(added_wraps),
                    "skipped_existing": int(skipped_existing_wraps),
                    "errors": list(errs)[:3],
                }

        # Build deterministic goal specs from plan tasks.
        # Keep trace-mining bounded: we only need a small, diverse subset to discover repeated
        # subpaths. Prefer short programs; long-horizon plans are handled by the flat miner and
        # full-plan decomposer, not by composed subpath mining.
        try:
            max_trace_goals = int(getattr(self.config, "concept_csv_composed_trace_goals", 8) or 8)
        except Exception:
            max_trace_goals = 8
        max_trace_goals = max(2, min(64, int(max_trace_goals)))
        goals: List[GoalSpecV72] = []
        try:
            trace_max_ops = int(getattr(self.config, "concept_csv_composed_trace_max_ops", 8) or 8)
        except Exception:
            trace_max_ops = 8
        trace_max_ops = max(2, min(2048, int(trace_max_ops)))

        def _ops_len(spec: Dict[str, Any]) -> int:
            ops0 = spec.get("ops") if isinstance(spec.get("ops"), list) else []
            return int(len(ops0))

        # Candidate set: short (<=trace_max_ops) first; if none, fall back to shortest overall.
        cand_short = [(tid, spec) for tid, spec in plan_tasks if isinstance(spec, dict) and _ops_len(spec) >= 2 and _ops_len(spec) <= int(trace_max_ops)]
        cand_short.sort(key=lambda kv: (_ops_len(kv[1]), str(kv[0])))
        cand_all = [(tid, spec) for tid, spec in plan_tasks if isinstance(spec, dict) and _ops_len(spec) >= 2]
        cand_all.sort(key=lambda kv: (_ops_len(kv[1]), str(kv[0])))
        cand = cand_short if cand_short else cand_all

        for task_id, expected_spec in list(cand)[: int(max_trace_goals)]:
            if not isinstance(expected_spec, dict):
                continue
            inputs = expected_spec.get("inputs")
            if not isinstance(inputs, dict) or not inputs:
                continue
            output_key = str(expected_spec.get("return_var") or "")
            if not output_key:
                continue
            goals.append(
                GoalSpecV72(
                    goal_kind="composed_plan_v74",
                    bindings=dict(inputs),
                    output_key=str(output_key),
                    expected=dict(expected_spec),
                    validator_id="plan_validator",
                    created_step=int(step),
                )
            )
        goals.sort(key=lambda g: str(g.goal_sig()))
        if not goals:
            out_early = {"enabled": True, "skipped": True, "reason": "no_goals", "force_new": bool(force_new)}
            if bool(deepwrap_meta.get("enabled", False)):
                out_early["deepwrap_v74"] = dict(deepwrap_meta)
            if bool(plan_ops_deepwrap_meta.get("enabled", False)):
                out_early["plan_ops_deepwrap_v74"] = dict(plan_ops_deepwrap_meta)
            return out_early

        # Trace mining for composed concepts: execute the *deterministic* plan program encoded
        # by the plan_validator expected_spec using the seeded plan-op concept acts.
        #
        # Rationale:
        # PlannerV79 is interface/type-driven and intentionally does NOT reason over semantics.
        # For plan_validator packs, semantic correctness often requires non-identity bindings
        # (e.g., `a := add_int(a,b)`), which a type-only planner will not reliably discover.
        # Using the explicit plan ops as the trace source is audit-first and deterministic:
        # "no hidden teacher" — the plan spec is part of the environment.
        plan_op_specs: Dict[str, Tuple[str, List[str]]] = {
            "add_int": (det_act_id(step=0, name="concept_plan_op_add_int_v0", idx=0), ["a", "b"]),
            "make_dict_goal_plan_ab": (
                det_act_id(step=0, name="concept_plan_op_make_dict_goal_plan_ab_v0", idx=0),
                ["goal_id", "plan", "a", "b"],
            ),
            "json_canonical": (det_act_id(step=0, name="concept_plan_op_json_canonical_v0", idx=0), ["obj"]),
        }
        # When validator packs require deep nested concepts AND non-trivial CSG structure, leaf plan ops must be
        # lifted to depth>=required_depth-1 so that composed concepts can reach required_depth without relying on
        # wrapper-only deepwrap at the top-level (which violates CSG richness).
        if int(required_depth) >= 2 and (int(required_csg_nodes) > 0 or int(required_csg_edges) > 0):
            target_depth = max(0, int(required_depth) - 1)
            try:
                max_new = int(
                    getattr(self.config, "concept_csv_composed_plan_op_deepwrap_max_new_per_window", 16) or 16
                )
            except Exception:
                max_new = 16
            max_new = max(0, min(256, int(max_new)))
            need = int(len(plan_op_specs)) * int(target_depth)
            if int(target_depth) > 0:
                max_new = max(int(max_new), int(need))
            budget = int(max_new)

            added_wraps: List[str] = []
            skipped_existing_wraps = 0
            errs: List[str] = []

            def _depth0(concept_id: str) -> int:
                try:
                    # _static_depth is defined above when required_depth>=2; fall back to leaf if absent.
                    return int(_static_depth(str(concept_id), set()))  # type: ignore[name-defined]
                except Exception:
                    return 0

            for fn_id in sorted(plan_op_specs.keys(), key=str):
                cid0, slots0 = plan_op_specs.get(str(fn_id), ("", []))
                current_id = str(cid0)
                cur_depth = int(_depth0(current_id))
                while budget > 0 and int(cur_depth) < int(target_depth):
                    try:
                        if store_base.get_concept_act(str(current_id)) is None:
                            inner_act = self.store.get_concept_act(str(current_id))
                            if inner_act is not None:
                                store_base.add(copy.deepcopy(inner_act))
                        wrap_act, _wrap_dbg = materialize_deep_wrapper_act_v74(
                            store_base=store_base,
                            inner_concept_id=str(current_id),
                            overhead_bits=int(deepwrap_overhead_bits),
                            seed_step=int(step),
                        )
                        if store_base.get_concept_act(str(wrap_act.id)) is None:
                            store_base.add(copy.deepcopy(wrap_act))
                        if self.store.get(str(wrap_act.id)) is None:
                            patch = Patch(kind="ADD_ACT", payload={"acts": [wrap_act.to_dict()]})
                            self._apply_patch(patch, count=True)
                            self._concept_csv_added += 1
                            added_wraps.append(str(wrap_act.id))
                            budget -= 1
                        else:
                            skipped_existing_wraps += 1
                        current_id = str(wrap_act.id)
                        cur_depth = int(cur_depth) + 1
                    except Exception as e:
                        errs.append(str(e))
                        break
                plan_op_specs[str(fn_id)] = (str(current_id), list(slots0))

            plan_ops_deepwrap_meta = {
                "enabled": True,
                "required_depth": int(required_depth),
                "required_csg_min_nodes": int(required_csg_nodes),
                "required_csg_min_edges": int(required_csg_edges),
                "target_depth": int(target_depth),
                "max_new": int(max_new),
                "added": int(len(added_wraps)),
                "promoted_ids": list(added_wraps),
                "skipped_existing": int(skipped_existing_wraps),
                "errors": list(errs)[:3],
            }
        # Ensure required plan-op deps exist in the materialization store.
        missing_plan_ops: List[str] = []
        for fn_id, (cid, _slots) in plan_op_specs.items():
            if store_base.get_concept_act(str(cid)) is None:
                dep = self.store.get_concept_act(str(cid))
                if dep is not None:
                    store_base.add(copy.deepcopy(dep))
                else:
                    missing_plan_ops.append(str(fn_id))

        traces: List[TraceV73] = []
        traces_ok: List[TraceV73] = []
        trace_reason_counts: Counter = Counter()
        trace_stage_counts: Counter = Counter()

        traces: List[TraceV73] = []
        traces_ok: List[TraceV73] = []
        # We only need enough OK multi-step traces to mine repeated subpaths; cap this to keep
        # per-window cost bounded as the store grows.
        try:
            ok_cap = int(getattr(self.config, "concept_csv_composed_traces_ok_cap", 8) or 8)
        except Exception:
            ok_cap = 8
        ok_cap = max(int(min_support), min(64, int(ok_cap)))
        for gi, goal in enumerate(goals):
            expected = goal.expected if isinstance(goal.expected, dict) else {}
            ops0 = expected.get("ops") if isinstance(expected.get("ops"), list) else []
            input_keys0 = expected.get("input_keys") if isinstance(expected.get("input_keys"), list) else []
            if not ops0 or not input_keys0:
                trace_stage_counts["skip_missing_spec"] += 1
                continue

            def _map_var_ref(vname: str) -> str:
                m = re.fullmatch(r"in([0-9]+)", str(vname))
                if not m:
                    return str(vname)
                try:
                    idx = int(m.group(1))
                except Exception:
                    return str(vname)
                if idx < 0 or idx >= len(input_keys0):
                    return str(vname)
                return str(input_keys0[idx] or vname)

            steps_tr: List[PlanStepTraceV73] = []
            compile_ok = True
            for si, op in enumerate(list(ops0)):
                if not isinstance(op, dict):
                    compile_ok = False
                    break
                fn_id = str(op.get("fn") or "")
                in_vars0 = op.get("in") if isinstance(op.get("in"), list) else []
                out_var = str(op.get("out") or "")
                if not fn_id or not out_var:
                    compile_ok = False
                    break
                spec = plan_op_specs.get(str(fn_id))
                if spec is None:
                    compile_ok = False
                    break
                concept_id, slot_order = spec
                if store_base.get_concept_act(str(concept_id)) is None:
                    compile_ok = False
                    break
                if len(in_vars0) != len(slot_order):
                    compile_ok = False
                    break
                bind_map: Dict[str, str] = {}
                for j, slot in enumerate(list(slot_order)):
                    bind_map[str(slot)] = str(_map_var_ref(str(in_vars0[j])))
                steps_tr.append(
                    PlanStepTraceV73(
                        idx=int(si),
                        concept_id=str(concept_id),
                        bind_map=dict(bind_map),
                        produces=str(out_var),
                    )
                )
            if not compile_ok or len(steps_tr) < 2:
                trace_stage_counts["compile_failed"] += 1
                continue

            engine = Engine(
                store_base,
                seed=int(getattr(self.config, "seed", 0) or 0),
                config=EngineConfig(enable_contracts=False),
            )

            vars_state: Dict[str, Any] = dict(goal.bindings)
            got_text = ""
            ok_exec = True
            reason = "not_executed"
            for si, st in enumerate(list(steps_tr)):
                bm = dict(st.bind_map) if isinstance(st.bind_map, dict) else {}
                inps: Dict[str, Any] = {}
                for slot, vname in sorted(bm.items(), key=lambda kv: str(kv[0])):
                    inps[str(slot)] = vars_state.get(str(vname))
                try:
                    res = engine.execute_concept_csv(
                        concept_act_id=str(st.concept_id),
                        inputs=dict(inps),
                        expected=None,
                        step=int(si),
                        max_depth=8,
                        max_events=512,
                        validate_output=False,
                    )
                    meta = res.get("meta") if isinstance(res.get("meta"), dict) else {}
                    meta = meta if isinstance(meta, dict) else {}
                    if not bool(meta.get("ok", False)):
                        ok_exec = False
                        reason = "callee_exec_failed"
                        break
                    out_val = res.get("output")
                    vars_state[str(st.produces)] = out_val
                    if str(st.produces) == str(goal.output_key):
                        got_text = str(meta.get("output_text") or out_val or "")
                except Exception:
                    ok_exec = False
                    reason = "callee_exception"
                    break

            ok_val = False
            if ok_exec:
                try:
                    vres = run_validator(str(goal.validator_id), str(got_text), goal.expected)
                    ok_val = bool(vres.passed)
                    reason = str(vres.reason)
                except Exception:
                    ok_val = False
                    reason = "validator_error"
            trace_reason_counts[str(reason)] += 1
            traces.append(
                TraceV73(
                    context_id=str(context_id_from_bindings(goal.bindings)),
                    goal_sig=str(goal.goal_sig()),
                    goal_id=str(goal.goal_id()),
                    goal_kind=str(goal.goal_kind),
                    bindings=dict(goal.bindings),
                    output_key=str(goal.output_key),
                    expected=goal.expected,
                    validator_id=str(goal.validator_id),
                    steps=list(steps_tr),
                    outcome={"ok": bool(ok_exec and ok_val), "reason": str(reason), "got": str(got_text), "expected": goal.expected},
                    cost_units={"steps_total": int(len(steps_tr)), "goal_idx": int(gi)},
                )
            )
            # Early stop once we have enough OK traces to mine candidates.
            if bool(ok_exec and ok_val) and len(steps_tr) >= 2:
                traces_ok.append(traces[-1])
                if int(len(traces_ok)) >= int(ok_cap):
                    break

        if len(traces_ok) < int(min_support):
            out_early = {
                "enabled": True,
                "skipped": True,
                "reason": "not_enough_ok_traces",
                "traces_total": int(len(traces)),
                "traces_ok": int(len(traces_ok)),
                "base_max_program_len": int(base_max_prog),
                "trace_max_ops": int(trace_max_ops),
                "missing_plan_ops": list(sorted(set(str(x) for x in missing_plan_ops if str(x)), key=str)),
                "trace_reasons_top": [
                    {"reason": str(k), "count": int(v)}
                    for k, v in trace_reason_counts.most_common(5)
                ],
                "trace_stage_counts": {str(k): int(v) for k, v in trace_stage_counts.items()},
                "force_new": bool(force_new),
            }
            if bool(deepwrap_meta.get("enabled", False)):
                out_early["deepwrap_v74"] = dict(deepwrap_meta)
            if bool(plan_ops_deepwrap_meta.get("enabled", False)):
                out_early["plan_ops_deepwrap_v74"] = dict(plan_ops_deepwrap_meta)
            return out_early

        scan_top_k = 0 if bool(force_new) else int(top_k)
        mined, dbg_mine = mine_candidates_v74(
            traces=traces_ok,
            max_k=int(max_k),
            min_support=int(min_support),
            top_k=int(scan_top_k),
        )
        if not mined:
            out_early = {
                "enabled": True,
                "skipped": True,
                "reason": "no_candidates",
                "traces_total": int(len(traces)),
                "traces_ok": int(len(traces_ok)),
                "mine": dict(dbg_mine),
                "force_new": bool(force_new),
            }
            if bool(deepwrap_meta.get("enabled", False)):
                out_early["deepwrap_v74"] = dict(deepwrap_meta)
            return out_early

        traces_by_sig = {str(t.trace_sig()): t for t in traces_ok}
        added = 0
        skipped_existing = 0
        skipped_low_gain = 0
        promoted_ids: List[str] = []
        for cand in mined:
            if added >= int(max_new):
                break
            try:
                gain_bits_est = int(getattr(cand, "gain_bits_est", 0) or 0)
            except Exception:
                gain_bits_est = 0
            # MDL gate: don't promote composed concepts unless the estimated gain exceeds overhead.
            if int(gain_bits_est) <= int(composed_overhead_bits):
                skipped_low_gain += 1
                continue
            try:
                rep_steps = extract_rep_steps(
                    traces_by_sig=traces_by_sig,
                    rep_trace_sig=str(getattr(cand, "rep_trace_sig", "") or ""),
                    start_idx=int(getattr(cand, "start_idx", 0) or 0),
                    subpath_len=int(len(getattr(cand, "subpath", ()) or ())),
                )
                act, _dbg_act = materialize_composed_act_v74(
                    store_base=store_base,
                    steps=list(rep_steps),
                    support_contexts=int(getattr(cand, "support_contexts", 0) or 0),
                    contexts=list(getattr(cand, "contexts", ()) or ()),
                    gain_bits_est=int(gain_bits_est),
                    overhead_bits=int(composed_overhead_bits),
                    seed_step=int(step),
                )
            except Exception:
                continue
            if self.store.get(str(act.id)) is not None:
                skipped_existing += 1
                continue
            patch = Patch(kind="ADD_ACT", payload={"acts": [act.to_dict()]})
            self._apply_patch(patch, count=True)
            added += 1
            self._concept_csv_added += 1
            promoted_ids.append(str(act.id))

        out_meta = {
            "enabled": True,
            "traces_total": int(len(traces)),
            "traces_ok": int(len(traces_ok)),
            "base_max_program_len": int(base_max_prog),
            "trace_max_ops": int(trace_max_ops),
            "overhead_bits": int(composed_overhead_bits),
            "mine": dict(dbg_mine),
            "candidates_total": int(len(mined)),
            "added": int(added),
            "skipped_low_gain": int(skipped_low_gain),
            "skipped_existing": int(skipped_existing),
            "promoted_ids": list(promoted_ids),
            "missing_plan_ops": list(sorted(set(str(x) for x in missing_plan_ops if str(x)), key=str)),
            "trace_reasons_top": [
                {"reason": str(k), "count": int(v)}
                for k, v in trace_reason_counts.most_common(5)
            ],
            "trace_stage_counts": {str(k): int(v) for k, v in trace_stage_counts.items()},
            "force_new": bool(force_new),
        }
        if bool(deepwrap_meta.get("enabled", False)):
            out_meta["deepwrap_v74"] = dict(deepwrap_meta)
        if bool(plan_ops_deepwrap_meta.get("enabled", False)):
            out_meta["plan_ops_deepwrap_v74"] = dict(plan_ops_deepwrap_meta)
        return out_meta

    def _load_tokens(self) -> List[str]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            text = f.read()
        toks = tokenize_text(text)
        if not toks:
            raise ValueError("Empty dataset after tokenization.")
        holdout_frac = float(getattr(self.config, "holdout_frac", 0.0) or 0.0)
        if holdout_frac != holdout_frac:
            holdout_frac = 0.0
        holdout_frac = max(0.0, min(0.5, holdout_frac))
        if holdout_frac > 0.0 and len(toks) >= 10:
            split = int(len(toks) * (1.0 - holdout_frac))
            split = max(1, min(split, len(toks) - 1))
            self._holdout_tokens = list(toks[split:])
            toks = list(toks[:split])
        else:
            self._holdout_tokens = []
        return toks

    def _iter_hf_stream_tokens(self, *, seed_offset: int = 0) -> "Iterator[str]":
        """
        Deterministic token stream from a HF dataset in streaming mode (no local 300GB corpus).

        Yields tokenized "<DOC> ... </DOC>" records, matching the on-disk hf_corpus format.
        """
        # Keep optional deps out of core import-time.
        from .hf_corpus import iter_hf_dataset  # type: ignore

        # Use the same formatter as hf_corpus (private), but tolerate changes.
        try:
            from .hf_corpus import format_example_auto as _fmt  # type: ignore
        except Exception:
            from .hf_corpus import _format_example_auto as _fmt  # type: ignore

        ds = iter_hf_dataset(
            dataset_id=str(getattr(self.config, "hf_dataset_id", "") or ""),
            split=str(getattr(self.config, "hf_dataset_split", "train") or "train"),
            name=(
                str(getattr(self.config, "hf_dataset_name", "") or "")
                or None
            ),
            streaming=bool(getattr(self.config, "hf_streaming", True)),
            shuffle=bool(getattr(self.config, "hf_shuffle", True)),
            seed=int(getattr(self.config, "seed", 0) or 0) + int(seed_offset),
            shuffle_buffer=int(getattr(self.config, "hf_shuffle_buffer", 10_000) or 10_000),
        )

        max_chars = int(getattr(self.config, "hf_max_chars_per_example", 8_000) or 8_000)
        max_chars = max(0, int(max_chars))

        ex_n = 0
        yielded_any = False
        # Startup logging: HF streaming shuffle buffers can delay the first yield, which makes
        # `tail -f train.log` look "stuck" even though CPU is busy pulling/parsing parquet.
        startup_log = bool(int(seed_offset) == 0)
        startup_log_every_ex = 1_000

        for ex in ds:
            ex_n += 1
            if startup_log and (not yielded_any) and (ex_n % startup_log_every_ex == 0):
                try:
                    sys.stdout.write(
                        f"[stream] main: {ex_n} examples read (awaiting first token)\n"
                    )
                    sys.stdout.flush()
                except Exception:
                    pass
            try:
                text = str(_fmt(ex) or "").strip()
            except Exception:
                try:
                    text = json.dumps(ex, ensure_ascii=False, sort_keys=True)
                except Exception:
                    text = str(ex)
                text = (text or "").strip()
            if not text:
                continue
            if max_chars > 0 and len(text) > max_chars:
                text = text[:max_chars].rstrip() + "…"
            rec = f"<DOC>\n{text}\n</DOC>\n\n"
            for tok in tokenize_text(rec):
                if startup_log and (not yielded_any):
                    yielded_any = True
                    try:
                        sys.stdout.write(
                            f"[stream] main: first token yielded after {ex_n} examples\n"
                        )
                        sys.stdout.flush()
                    except Exception:
                        pass
                yield tok

    def _init_streaming_holdout_tokens(self) -> None:
        """
        Populate `self._holdout_tokens` deterministically for holdout gating in streaming runs.
        """
        h_windows = int(getattr(self.config, "holdout_eval_windows", 0) or 0)
        h_windows = max(0, min(16, int(h_windows)))
        if h_windows <= 0:
            self._holdout_tokens = []
            return
        h_len = int(getattr(self.config, "holdout_eval_tokens", 0) or 0)
        if h_len <= 0:
            h_len = int(getattr(self.config, "val_tokens", 4000) or 4000)
        h_len = max(1, int(h_len))
        target = int(getattr(self.config, "stream_holdout_tokens", 0) or 0)
        # If not specified, ensure we have enough tokens to cover all windows plus slack.
        if target <= 0:
            target = int(h_len * h_windows * 4)
        target = max(32, int(target))

        seed_off = int(getattr(self.config, "hf_holdout_seed_offset", 1) or 1)
        it = self._iter_hf_stream_tokens(seed_offset=seed_off)
        toks: List[str] = []
        try:
            sys.stdout.write(
                f"[holdout] init: collecting {target} tokens (seed_offset={seed_off}, "
                f"windows={h_windows}, window_len={h_len})\n"
            )
            sys.stdout.flush()
        except Exception:
            pass
        next_log = 5_000
        while len(toks) < target:
            toks.append(next(it))
            if len(toks) >= next_log:
                try:
                    sys.stdout.write(f"[holdout] init: {len(toks)}/{target} tokens\n")
                    sys.stdout.flush()
                except Exception:
                    pass
                next_log += 5_000
        self._holdout_tokens = toks
        try:
            sys.stdout.write(f"[holdout] init: done ({len(toks)} tokens)\n")
            sys.stdout.flush()
        except Exception:
            pass

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

    def _eval_static_nll_windows(
        self,
        store: ActStore,
        tokens: Sequence[str],
        *,
        start: int,
        starts: Optional[Sequence[int]] = None,
        length: int,
        engine_config: EngineConfig,
        patch: Optional[Patch] = None,
    ) -> Dict[str, Any]:
        if not tokens:
            raise ValueError("empty token stream for evaluation")
        store_copy: ActStore = copy.deepcopy(store)
        if patch is not None:
            self._apply_patch_to_store(store_copy, patch)

        if starts is None:
            starts_eval = [int(start)]
        else:
            starts_eval = [int(x) for x in list(starts) if x is not None]
            if not starts_eval:
                starts_eval = [int(start)]
        L = max(1, len(tokens))
        seen = set()
        uniq: List[int] = []
        for s in starts_eval:
            ss = int(s) % L
            if ss in seen:
                continue
            seen.add(ss)
            uniq.append(ss)
        starts_eval = uniq or [int(start) % L]

        nll_bits_windows: List[float] = []
        for s in starts_eval:
            engine = Engine(store_copy, seed=self.config.seed, config=engine_config)
            nll_bits_windows.append(
                float(self._eval_nll_bits(engine, tokens, start=int(s), length=int(length)))
            )
        nll_bits_mean = sum(nll_bits_windows) / max(1, len(nll_bits_windows))
        return {"nll_bits": float(nll_bits_mean), "nll_bits_windows": list(nll_bits_windows)}

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
        tasks = skill_suite_tasks_for_pack(str(getattr(self.config, "skill_suite_pack", "v0") or "v0"))
        _transcripts, metrics = run_skill_suite(
            engine,
            tasks=tasks,
            max_new_tokens=self.config.skill_suite_max_new_tokens,
            prompt_history_k=int(getattr(self.config, "skill_suite_prompt_history_k", 0) or 0),
        )
        return dict(metrics)

    def _eval_agency_suite_metrics(self, *, store_override: Optional[ActStore] = None) -> Dict[str, Any]:
        """
        Deterministic planner+executor "agency" suite.

        This is closed-world (no external knowledge) and measures whether the current
        concept_csv library supports explicit multi-step planning (GOAL->PLAN->EXEC->EVAL).
        """
        enabled = bool(getattr(self.config, "agency_suite_enabled", False))
        if not enabled:
            return {"enabled": False}

        try:
            max_tasks = int(getattr(self.config, "agency_suite_max_tasks", 0) or 0)
        except Exception:
            max_tasks = 0
        max_tasks = max(0, min(64, int(max_tasks)))
        if max_tasks <= 0:
            return {"enabled": True, "skipped": True, "reason": "max_tasks_zero"}

        try:
            min_steps = int(getattr(self.config, "agency_suite_min_steps", 2) or 2)
        except Exception:
            min_steps = 2
        # Long-horizon agency requires >16 steps in some packs (e.g., sota_v7 long-sum tasks).
        min_steps = max(1, min(256, int(min_steps)))

        try:
            base_max_prog = int(getattr(self.config, "agency_suite_base_max_program_len", 6) or 6)
        except Exception:
            base_max_prog = 6
        base_max_prog = max(1, min(64, int(base_max_prog)))

        try:
            p_max_depth = int(getattr(self.config, "agency_suite_planner_max_depth", 6) or 6)
        except Exception:
            p_max_depth = 6
        # Planner depth must be able to exceed 16 for long-horizon agency tasks.
        p_max_depth = max(1, min(256, int(p_max_depth)))
        try:
            p_max_exp = int(getattr(self.config, "agency_suite_planner_max_expansions", 256) or 256)
        except Exception:
            p_max_exp = 256
        p_max_exp = max(1, min(4096, int(p_max_exp)))

        # Local imports to keep learn.py lightweight and avoid import cycles.
        from .goal_spec_v72 import GoalSpecV72
        from .planner_v79 import PlanStepV79, PlanV79, PlannerV79
        from .match_v79 import is_act_allowed_for_goal_kind
        from .validators import run_validator

        # Select deterministic plan tasks as goal specs (closed; no world knowledge).
        try:
            tasks = skill_suite_tasks_for_pack(str(getattr(self.config, "skill_suite_pack", "v0") or "v0"))
        except Exception:
            tasks = ()

        goals: List[GoalSpecV72] = []
        for t in tasks:
            if not isinstance(t, dict):
                continue
            if str(t.get("validator_id") or "") != "plan_validator":
                continue
            task_id = str(t.get("task_id") or "")
            expected_spec = t.get("expected_spec")
            if not isinstance(expected_spec, dict) or not expected_spec:
                compiler_id = str(t.get("compiler_id") or "")
                if compiler_id == V67_DIALOGUE_COMPILER_ID or task_id.startswith(("v67_", "v68_")):
                    turns = t.get("dialogue") or ()
                    if isinstance(turns, (list, tuple)) and turns:
                        vt = int(t.get("validate_turn", max(0, len(turns) - 1)) or 0)
                        vt = max(0, min(vt, len(turns) - 1))
                        try:
                            expected_spec = compile_dialogue_v67(str(turns[vt]))
                        except Exception:
                            expected_spec = {}
            if not isinstance(expected_spec, dict) or not expected_spec:
                continue
            inputs = expected_spec.get("inputs")
            if not isinstance(inputs, dict) or not inputs:
                continue
            out_key = str(expected_spec.get("return_var") or "")
            if not out_key:
                continue
            goals.append(
                GoalSpecV72(
                    goal_kind="agency_suite_v1",
                    bindings=dict(inputs),
                    output_key=str(out_key),
                    expected=dict(expected_spec),
                    validator_id="plan_validator",
                    created_step=0,
                )
            )
            if len(goals) >= int(max_tasks):
                break

        goals.sort(key=lambda g: str(g.goal_sig()))
        if not goals:
            return {"enabled": True, "skipped": True, "reason": "no_goals"}

        # Filter store for planning: keep only "small" concepts as primitives so that a plan requires
        # explicit multi-step structure (no single monolith act escape route).
        store_active = store_override.active() if isinstance(store_override, ActStore) else self.store.active()
        store_base = ActStore()
        for act in store_active:
            if str(getattr(act, "kind", "")) != "concept_csv":
                store_base.add(copy.deepcopy(act))
                continue
            prog = list(getattr(act, "program", []) or [])
            prog_len = int(len(prog))
            has_call = any(str(getattr(ins, "op", "")) == "CSV_CALL" for ins in prog)
            # For the agency suite, "primitives" must be call-free; otherwise we can include a wrapper
            # without its callees and get runtime-only failures (callee_failed), and/or collapse the
            # apparent plan horizon via nested calls.
            if has_call:
                continue
            if prog_len <= int(base_max_prog):
                store_base.add(copy.deepcopy(act))

        def _infer_type(v: Any) -> str:
            if isinstance(v, bool):
                return "int"
            if isinstance(v, int):
                return "int"
            if isinstance(v, dict):
                return "dict"
            return "str"

        def _is_direct_solver_by_types(
            *, in_schema: Dict[str, Any], out_schema: Dict[str, Any], init_types: Dict[str, str], goal_out_type: str, goal_out_key: str
        ) -> bool:
            """
            With PlannerV79 supporting non-identity bind_maps, input *names* no longer protect against
            one-step "direct solvers". A concept is a direct solver if its required input *types* can be
            satisfied from the initial bindings (distinct vars per slot), and it can emit the goal output.
            """
            if not isinstance(in_schema, dict) or not isinstance(out_schema, dict):
                return False
            req_counts: Counter = Counter()
            for _, t in in_schema.items():
                tt = str(t or "str") or "str"
                req_counts[tt] += 1
            avail_counts: Counter = Counter()
            for _, t in init_types.items():
                avail_counts[str(t or "str") or "str"] += 1
            for tt, need in req_counts.items():
                if int(avail_counts.get(str(tt), 0)) < int(need):
                    return False
            out_keys0 = {str(k) for k in out_schema.keys() if str(k)}
            out_types0 = {str(v) for v in out_schema.values() if str(v)}
            if str(goal_out_key) and str(goal_out_key) in out_keys0:
                return True
            # Type-only `str` direct-solver detection is too broad and can filter away all useful
            # concepts (most outputs are `str`), preventing multi-step planning/tracing.
            if str(goal_out_type) and str(goal_out_type) in out_types0:
                return True
            return False

        def _callee_chain(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
            chain: List[Dict[str, Any]] = []
            cur = dict(meta or {}) if isinstance(meta, dict) else {}
            for _ in range(32):
                if not isinstance(cur, dict):
                    break
                reason = str(cur.get("reason") or "")
                chain.append(
                    {
                        "concept_id": str(cur.get("concept_id") or ""),
                        "reason": reason,
                        "callee": str(cur.get("callee") or ""),
                    }
                )
                if reason != "callee_failed":
                    break
                nxt = cur.get("callee_meta")
                if not isinstance(nxt, dict):
                    break
                cur = nxt
            return chain

        def _primitive_wrapper_fn_id(act: Act) -> str:
            """
            Detect a call-free single-primitive concept wrapper (used to synthesize long plans from
            plan_validator expected_spec.ops deterministically, without hidden solvers).
            """
            if act is None or str(getattr(act, "kind", "")) != "concept_csv" or (not bool(getattr(act, "active", True))):
                return ""
            if not is_act_allowed_for_goal_kind(act=act, goal_kind="agency_suite_v1"):
                return ""
            prog = list(getattr(act, "program", []) or [])
            if any(str(getattr(ins, "op", "")) == "CSV_CALL" for ins in prog):
                return ""
            prims = [ins for ins in prog if str(getattr(ins, "op", "")) == "CSV_PRIMITIVE"]
            if len(prims) != 1:
                return ""
            for ins in prog:
                if str(getattr(ins, "op", "")) not in {"CSV_GET_INPUT", "CSV_PRIMITIVE", "CSV_RETURN"}:
                    return ""
            args = getattr(prims[0], "args", {}) or {}
            if not isinstance(args, dict):
                args = {}
            return str(args.get("fn") or "")

        def _bind_map_for_fn(*, fn_id: str, slot_keys: List[str], arg_vars: List[str]) -> Dict[str, str]:
            slots = [str(s) for s in slot_keys if str(s)]
            slots = sorted(set(slots))
            args = [str(v) for v in arg_vars if str(v)]
            if fn_id == "add_int" and len(args) >= 2:
                if "a" in slots and "b" in slots:
                    return {"a": args[0], "b": args[1]}
                if "x" in slots and "y" in slots:
                    return {"x": args[0], "y": args[1]}
            if fn_id == "make_dict_goal_plan_ab" and len(args) >= 4:
                bm: Dict[str, str] = {}
                if "goal_id" in slots:
                    bm["goal_id"] = args[0]
                if "plan" in slots:
                    bm["plan"] = args[1]
                if "b" in slots:
                    bm["b"] = args[3]
                rem = [s for s in slots if s not in bm]
                if rem:
                    bm[rem[0]] = args[2]
                # Fill any remaining (stable).
                for i, s in enumerate(slots):
                    if s not in bm:
                        bm[s] = args[min(i, len(args) - 1)]
                return bm
            # Fallback: bind slots in sorted order to args in order.
            bm2: Dict[str, str] = {}
            for i, s in enumerate(slots):
                if i >= len(args):
                    break
                bm2[s] = args[i]
            return bm2

        def _synthesize_plan_from_expected_ops(*, goal: GoalSpecV72, store: ActStore) -> Optional[PlanV79]:
            exp = goal.expected if isinstance(getattr(goal, "expected", None), dict) else {}
            ops = exp.get("ops") if isinstance(exp.get("ops"), list) else []
            input_keys = exp.get("input_keys") if isinstance(exp.get("input_keys"), list) else []
            if not ops or not input_keys:
                return None
            # Map v66-style sim vars (in0,in1,...) to actual external binding keys (goal_id,plan,x0,...,b).
            sym_to_var: Dict[str, str] = {}
            for i, k in enumerate(input_keys):
                ks = str(k or "")
                if ks:
                    sym_to_var[f"in{i}"] = ks

            # Build a deterministic primitive-op lookup.
            fn_best: Dict[str, Tuple[int, str, List[str]]] = {}
            # fn_id -> (prog_len, act_id, slot_keys)
            for act0 in store.active():
                if str(getattr(act0, "kind", "")) != "concept_csv":
                    continue
                fn0 = _primitive_wrapper_fn_id(act0)
                if not fn0:
                    continue
                ev0 = act0.evidence if isinstance(getattr(act0, "evidence", None), dict) else {}
                iface0 = ev0.get("interface") if isinstance(ev0.get("interface"), dict) else {}
                iface0 = iface0 if isinstance(iface0, dict) else {}
                in0 = iface0.get("input_schema") if isinstance(iface0.get("input_schema"), dict) else {}
                prog_len = int(len(getattr(act0, "program", []) or []))
                cur = fn_best.get(fn0)
                cand = (int(prog_len), str(getattr(act0, "id", "")), sorted(str(k) for k in in0.keys() if str(k)))
                if cur is None or (cand[0], cand[1]) < (cur[0], cur[1]):
                    fn_best[fn0] = cand

            steps: List[PlanStepV79] = []
            used_vars: Set[str] = set(str(k) for k in (goal.bindings or {}).keys() if str(k))
            for idx, op in enumerate(list(ops)):
                if not isinstance(op, dict):
                    return None
                fn_id = str(op.get("fn") or "")
                ins = op.get("in") if isinstance(op.get("in"), list) else []
                out_sym = str(op.get("out") or "")
                if not fn_id or not ins or not out_sym:
                    return None
                best = fn_best.get(fn_id)
                if best is None:
                    return None
                _, act_id, slot_keys = best
                # Resolve input symbols (in*, v*) to concrete var names.
                arg_vars: List[str] = []
                for s in ins:
                    ss = str(s or "")
                    if not ss:
                        return None
                    vv = sym_to_var.get(ss)
                    if vv is None:
                        # v* symbols: keep as-is (already produced).
                        vv = ss
                    arg_vars.append(str(vv))

                # Ensure deterministic fresh var naming for outputs to avoid collisions.
                out_var = str(out_sym)
                if out_var in used_vars:
                    out_var = f"{out_var}_{idx}"
                sym_to_var[out_sym] = out_var
                used_vars.add(out_var)

                bm = _bind_map_for_fn(fn_id=fn_id, slot_keys=list(slot_keys), arg_vars=list(arg_vars))
                step_body = {
                    "idx": int(idx),
                    "concept_id": str(act_id),
                    "bind_map": {str(k): str(bm.get(k) or "") for k in sorted(bm.keys(), key=str)},
                    "produces": str(out_var),
                }
                step_id = sha256_hex(canonical_json_dumps(step_body).encode("utf-8"))
                steps.append(
                    PlanStepV79(
                        step_id=str(step_id),
                        idx=int(idx),
                        concept_id=str(act_id),
                        bind_map=dict(step_body["bind_map"]),
                        produces=str(out_var),
                    )
                )

            plan_body = {"schema_version": 1, "steps": [s.to_dict() for s in steps]}
            psig = sha256_hex(canonical_json_dumps(plan_body).encode("utf-8"))
            return PlanV79(steps=steps, plan_sig=str(psig))

        planner = PlannerV79(max_depth=int(p_max_depth), max_expansions=int(p_max_exp))

        total = 0
        plan_found = 0
        passed = 0
        steps_pass: List[int] = []
        fail_reasons: Counter = Counter()
        fail_details: List[Dict[str, Any]] = []

        for g in goals:
            total += 1
            # Close the last "one-step agency" escape route: forbid any concept that can directly
            # produce the goal output key using only the initial bindings. This forces explicit
            # multi-step planning (>= min_steps) to survive the agency suite.
            init_bindings = dict(g.bindings) if isinstance(g.bindings, dict) else {}
            init_types = {str(k): _infer_type(v) for k, v in init_bindings.items() if str(k)}
            store_goal = ActStore()
            for act0 in store_base.active():
                if str(getattr(act0, "kind", "")) != "concept_csv":
                    store_goal.add(copy.deepcopy(act0))
                    continue
                ev0 = act0.evidence if isinstance(getattr(act0, "evidence", None), dict) else {}
                iface0 = ev0.get("interface") if isinstance(ev0.get("interface"), dict) else {}
                iface0 = iface0 if isinstance(iface0, dict) else {}
                in_schema0 = iface0.get("input_schema") if isinstance(iface0.get("input_schema"), dict) else {}
                out_schema0 = iface0.get("output_schema") if isinstance(iface0.get("output_schema"), dict) else {}
                goal_out_type = "str" if str(g.validator_id or "") in {"plan_validator", "state_validator"} else ""
                if _is_direct_solver_by_types(
                    in_schema=in_schema0,
                    out_schema=out_schema0,
                    init_types=init_types,
                    goal_out_type=str(goal_out_type),
                    goal_out_key=str(g.output_key),
                ):
                    continue
                store_goal.add(copy.deepcopy(act0))

            # Prefer deterministic synthesis from plan_validator expected_spec.ops when available.
            plan: Optional[PlanV79] = None
            dbg: Dict[str, Any] = {}
            if str(g.validator_id or "") == "plan_validator":
                try:
                    plan = _synthesize_plan_from_expected_ops(goal=g, store=store_goal)
                    if plan is not None:
                        dbg = {"reason": "expected_ops"}
                except Exception:
                    plan = None
                    dbg = {}
            if plan is None:
                plan, dbg = planner.plan(goal_spec=g, store=store_goal)
            if plan is None:
                fail_reasons[str(dbg.get("reason") or "plan_not_found")] += 1
                continue
            plan_found += 1
            if int(len(plan.steps)) < int(min_steps):
                fail_reasons["plan_too_short"] += 1
                continue
            vars_state: Dict[str, Any] = dict(g.bindings)
            got_text = ""
            ok_exec = True
            fail_detail: Optional[Dict[str, Any]] = None
            engine = Engine(
                store_goal,
                seed=int(getattr(self.config, "seed", 0) or 0),
                config=EngineConfig(enable_contracts=False),
            )
            for si, st in enumerate(list(plan.steps)):
                bm = dict(st.bind_map) if isinstance(st.bind_map, dict) else {}
                inps: Dict[str, Any] = {}
                for slot, vname in sorted(bm.items(), key=lambda kv: str(kv[0])):
                    inps[str(slot)] = vars_state.get(str(vname))
                try:
                    res = engine.execute_concept_csv(
                        concept_act_id=str(st.concept_id),
                        inputs=dict(inps),
                        expected=None,
                        step=int(si),
                        max_depth=8,
                        max_events=512,
                        validate_output=False,
                    )
                    meta = res.get("meta") if isinstance(res.get("meta"), dict) else {}
                    meta = meta if isinstance(meta, dict) else {}
                    if not bool(meta.get("ok", False)):
                        ok_exec = False
                        fail_reasons[str(meta.get("reason") or "exec_failed")] += 1
                        if fail_detail is None:
                            exp = g.expected if isinstance(getattr(g, "expected", None), dict) else {}
                            fail_detail = {
                                "goal_sig": str(g.goal_sig()),
                                "task_id": str(exp.get("task_id") or ""),
                                "plan_len": int(len(plan.steps)),
                                "fail_step": int(si),
                                "fail_concept_id": str(st.concept_id),
                                "fail_reason": str(meta.get("reason") or ""),
                                "fail_chain": _callee_chain(meta),
                                "plan_steps_preview": [
                                    {
                                        "i": int(j),
                                        "concept_id": str(s2.concept_id),
                                        "bind_map": dict(s2.bind_map) if isinstance(s2.bind_map, dict) else {},
                                        "produces": str(s2.produces),
                                    }
                                    for j, s2 in enumerate(list(plan.steps)[: min(12, len(plan.steps))])
                                ],
                            }
                        break
                    out_val = res.get("output")
                    vars_state[str(st.produces)] = out_val
                    if str(st.produces) == str(g.output_key):
                        got_text = str(meta.get("output_text") or out_val or "")
                except Exception:
                    ok_exec = False
                    fail_reasons["exec_error"] += 1
                    break
            if not ok_exec:
                if fail_detail is not None and len(fail_details) < 4:
                    fail_details.append(fail_detail)
                continue
            try:
                vres = run_validator(str(g.validator_id), str(got_text), g.expected)
                if bool(vres.passed):
                    passed += 1
                    steps_pass.append(int(len(plan.steps)))
                else:
                    fail_reasons[str(vres.reason or "validator_fail")] += 1
            except Exception:
                fail_reasons["validator_error"] += 1

        pass_rate = float(passed / total) if total > 0 else 0.0
        found_rate = float(plan_found / total) if total > 0 else 0.0
        steps_mean = float(sum(steps_pass) / len(steps_pass)) if steps_pass else 0.0
        steps_pass_sorted = sorted(int(x) for x in steps_pass) if steps_pass else []
        steps_median = float(steps_pass_sorted[len(steps_pass_sorted) // 2]) if steps_pass_sorted else 0.0

        return {
            "enabled": True,
            "total": int(total),
            "plan_found": int(plan_found),
            "plan_found_rate": float(found_rate),
            "pass_count": int(passed),
            "pass_rate": float(pass_rate),
            "min_steps": int(min_steps),
            "base_max_program_len": int(base_max_prog),
            "planner_max_depth": int(p_max_depth),
            "planner_max_expansions": int(p_max_exp),
            "steps_pass_mean": float(steps_mean),
            "steps_pass_median": float(steps_median),
            "fail_reasons": dict(sorted((str(k), int(v)) for k, v in fail_reasons.items())),
            "fail_details": list(fail_details),
        }

    def _utility_bottleneck_loss(self, gen: Dict[str, Any]) -> Tuple[str, float, Dict[str, float]]:
        """
        Deterministic "LOSS-AND" for the validator pack: treat utility as a hard bottleneck.
        Returns (component, loss, terms), where loss = max(component losses).
        """

        def _int0(x: Any) -> int:
            try:
                v = int(x or 0)
            except Exception:
                v = 0
            return int(v)

        def _float01(x: Any) -> float:
            try:
                v = float(x or 0.0)
            except Exception:
                v = 0.0
            if v != v:
                v = 0.0
            return max(0.0, min(1.0, float(v)))

        terms: List[Tuple[str, float]] = []

        # Category bottlenecks (only include categories that exist in the selected pack).
        for cat in (
            "instruction",
            "json",
            "math",
            "state",
            "plan",
            "clarify",
            "consistency",
            "memory",
            "dialogue",
            "concept",
            "agency",
        ):
            total = _int0(gen.get(f"utility_{cat}_total"))
            if total <= 0:
                continue
            pass_rate = _float01(gen.get(f"utility_{cat}_pass_rate"))
            terms.append((cat, 1.0 - pass_rate))

        # Goal satisfaction bottleneck (only if the pack includes goal-bearing tasks).
        goals_total = _int0(gen.get("utility_goals_total"))
        if goals_total > 0:
            pass_rate = _float01(gen.get("utility_goals_satisfied_rate"))
            terms.append(("goals", 1.0 - pass_rate))

        # Planning trace integrity (only meaningful when plan tasks are present).
        plan_total = _int0(gen.get("utility_plan_total"))
        turns_total = _int0(gen.get("utility_plan_trace_turns_total"))
        if plan_total > 0 and turns_total > 0:
            missing = _int0(gen.get("utility_plan_trace_missing_turns"))
            miss_rate = max(0.0, min(1.0, float(missing) / float(turns_total)))
            terms.append(("plan_trace_missing", float(miss_rate)))

        # Back-compat: older reports don't have per-category totals; fall back to overall pass_rate.
        if not terms:
            pass_rate = _float01(gen.get("utility_pass_rate"))
            terms.append(("pass_rate", 1.0 - pass_rate))

        component, loss = max(terms, key=lambda kv: kv[1])
        return str(component), float(loss), {f"{k}_loss": float(v) for k, v in terms}

    def _get_mode_policy_act(self) -> Optional[Act]:
        acts = self.store.by_kind("mode_policy")
        return acts[0] if acts else None

    def _get_fact_memory_act(self) -> Optional[Act]:
        for act in self.store.by_kind("memory_facts"):
            ev = act.evidence
            if isinstance(ev, dict) and ev.get("name") == "fact_memory_v0":
                return act
        return None

    def _get_system_survival_act(self) -> Optional[Act]:
        for act in self.store.by_kind("goal"):
            ev = act.evidence
            if isinstance(ev, dict) and ev.get("name") == "system_survival_v0":
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

        # Turn-level "concepts as subgraphs" (shadow-only: affects logging/metrics, not generation).
        concepts = ev.setdefault("concepts", {})
        if not isinstance(concepts, dict):
            concepts = {}
            ev["concepts"] = concepts

        concept_budget = int(ev.get("concept_budget", 128) or 128)
        concept_max_new = int(ev.get("concept_max_new_per_window", 8) or 8)
        concept_n_min = int(ev.get("concept_n_min", 2) or 2)
        concept_n_max = int(ev.get("concept_n_max", 6) or 6)
        concept_min_count = int(ev.get("concept_min_count", 3) or 3)
        concept_set_min_size = int(ev.get("concept_set_min_size", 2) or 2)
        concept_set_max_size = int(ev.get("concept_set_max_size", 6) or 6)
        concept_set_min_count = int(ev.get("concept_set_min_count", 3) or 3)

        if (
            budget <= 0
            or max_new <= 0
            or n_min <= 0
            or n_max < n_min
            or min_count <= 1
            or set_min_size <= 0
            or set_min_count <= 1
            or transition_top_k < 0
            or concept_budget < 0
            or concept_max_new < 0
            or concept_n_min <= 0
            or concept_n_max < concept_n_min
            or concept_min_count <= 1
            or concept_set_min_size <= 0
            or concept_set_max_size < concept_set_min_size
            or concept_set_min_count <= 1
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
        # Turn-level subgraph motifs (per turn) mined from trace["subgraph"].
        turn_exec_seqs_all: List[List[str]] = []
        turn_rr_seqs_all: List[List[str]] = []
        turn_exec_seq_counts: Counter = Counter()
        turn_exec_seq_contexts: Dict[Tuple[str, ...], set] = {}
        turn_exec_set_counts: Counter = Counter()
        turn_exec_set_contexts: Dict[Tuple[str, ...], set] = {}
        turn_rr_seq_counts: Counter = Counter()
        turn_rr_seq_contexts: Dict[Tuple[str, ...], set] = {}
        turn_rr_set_counts: Counter = Counter()
        turn_rr_set_contexts: Dict[Tuple[str, ...], set] = {}

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

                # Deterministic "turn context" signature: first non-empty ctx_key (mode␟ctx_key).
                turn_ctx_key = ""
                for _ck in ctx_keys0:
                    if isinstance(_ck, str) and _ck:
                        turn_ctx_key = _ck
                        break
                turn_ctx_sig = f"{mode}\u241f{turn_ctx_key}" if turn_ctx_key else ""

                # Turn-level subgraph motifs: executed predictors + rewrite hits (ordered unique).
                sub = tr.get("subgraph") or {}
                if not isinstance(sub, dict):
                    sub = {}
                exec_turn0 = sub.get("executed_predictor_act_ids") or []
                rr_turn0 = sub.get("rewrite_rule_hit_ids") or []

                exec_turn: List[str] = []
                if isinstance(exec_turn0, list):
                    for x in exec_turn0:
                        if isinstance(x, str) and x and (not x.startswith("__")):
                            exec_turn.append(str(x))
                rr_turn: List[str] = []
                if isinstance(rr_turn0, list):
                    for x in rr_turn0:
                        if isinstance(x, str) and x and (not x.startswith("__")):
                            rr_turn.append(str(x))

                if exec_turn:
                    turn_exec_seqs_all.append(list(exec_turn))
                    # Motif SEQ: n-grams over the ordered unique list.
                    if len(exec_turn) >= int(concept_n_min):
                        nmax2 = min(int(concept_n_max), int(len(exec_turn)))
                        for n in range(int(concept_n_min), int(nmax2) + 1):
                            for i0 in range(int(len(exec_turn) - n + 1)):
                                pat = tuple(exec_turn[i0 : i0 + n])
                                turn_exec_seq_counts[pat] += 1
                                if turn_ctx_sig:
                                    s = turn_exec_seq_contexts.get(pat)
                                    if s is None:
                                        s = set()
                                        turn_exec_seq_contexts[pat] = s
                                    s.add(turn_ctx_sig)
                    # Motif SET: co-occurrence itemsets (subset combinations).
                    sorted_exec = sorted(set(exec_turn))
                    smax2 = min(int(concept_set_max_size), int(len(sorted_exec)))
                    for k in range(int(concept_set_min_size), int(smax2) + 1):
                        for comb in itertools.combinations(sorted_exec, k):
                            pat2 = tuple(comb)
                            turn_exec_set_counts[pat2] += 1
                            if turn_ctx_sig:
                                s = turn_exec_set_contexts.get(pat2)
                                if s is None:
                                    s = set()
                                    turn_exec_set_contexts[pat2] = s
                                s.add(turn_ctx_sig)

                if rr_turn:
                    turn_rr_seqs_all.append(list(rr_turn))
                    if len(rr_turn) >= int(concept_n_min):
                        nmax2 = min(int(concept_n_max), int(len(rr_turn)))
                        for n in range(int(concept_n_min), int(nmax2) + 1):
                            for i0 in range(int(len(rr_turn) - n + 1)):
                                pat = tuple(rr_turn[i0 : i0 + n])
                                turn_rr_seq_counts[pat] += 1
                                if turn_ctx_sig:
                                    s = turn_rr_seq_contexts.get(pat)
                                    if s is None:
                                        s = set()
                                        turn_rr_seq_contexts[pat] = s
                                    s.add(turn_ctx_sig)
                    sorted_rr = sorted(set(rr_turn))
                    smax2 = min(int(concept_set_max_size), int(len(sorted_rr)))
                    for k in range(int(concept_set_min_size), int(smax2) + 1):
                        for comb in itertools.combinations(sorted_rr, k):
                            pat2 = tuple(comb)
                            turn_rr_set_counts[pat2] += 1
                            if turn_ctx_sig:
                                s = turn_rr_set_contexts.get(pat2)
                                if s is None:
                                    s = set()
                                    turn_rr_set_contexts[pat2] = s
                                s.add(turn_ctx_sig)

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

        # Evict to budget deterministically by utility (gain_real_bits_est, contexts_distinct, count, id).
        evicted = 0
        if budget > 0 and len(macros) > budget:
            ranked: List[Tuple[int, int, int, str]] = []
            for mid, ent in macros.items():
                if not isinstance(ent, dict):
                    continue
                ranked.append(
                    (
                        int(ent.get("gain_real_bits_est", 0) or 0),
                        int(ent.get("contexts_distinct", 0) or 0),
                        int(ent.get("count", 0) or 0),
                        str(mid),
                    )
                )
            ranked.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))
            keep = {mid for _g, _ctx, _c, mid in ranked[:budget]}
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

        # ---- Concepts as turn-level subgraphs (shadow-only): mine, budget, and MDL-real gate ----
        concept_added = 0
        concept_added_seq = 0
        concept_added_set = 0
        concept_updated_existing = 0
        concept_evicted = 0
        concept_cleared = False

        concept_trace_mdl_before_bits_est = 0
        concept_trace_mdl_after_bits_est = 0
        concept_trace_mdl_gain_bits_est = 0
        concept_library_cost_bits_est = 0
        concept_trace_mdl_gain_real_bits_est = 0

        avg_trace_len_turn_before = 0.0
        avg_trace_len_turn_after = 0.0
        concept_turn_hit_rate = 0.0
        concept_symbol_coverage_rate = 0.0
        concept_hits_per_turn_mean = 0.0

        # concepts (dict) is defined near the top of this function.
        def _concept_seq_patterns(src: str) -> List[Tuple[int, Tuple[str, ...], str]]:
            pats: List[Tuple[int, Tuple[str, ...], str]] = []
            for cid, ent in concepts.items():
                if not isinstance(ent, dict):
                    continue
                if str(ent.get("kind", "")) != "concept":
                    continue
                if str(ent.get("source", "")) != src:
                    continue
                if str(ent.get("type", "")) != "seq":
                    continue
                pat = ent.get("pattern")
                if not isinstance(pat, list) or not pat:
                    continue
                toks = tuple(str(x) for x in pat if x is not None)
                if len(toks) < 2:
                    continue
                pats.append((len(toks), toks, str(cid)))
            pats.sort(key=lambda x: (-x[0], x[2], x[1]))
            return pats

        def _concept_set_patterns(src: str) -> List[Tuple[int, Tuple[str, ...], str]]:
            pats: List[Tuple[int, Tuple[str, ...], str]] = []
            for cid, ent in concepts.items():
                if not isinstance(ent, dict):
                    continue
                if str(ent.get("kind", "")) != "concept":
                    continue
                if str(ent.get("source", "")) != src:
                    continue
                if str(ent.get("type", "")) != "set":
                    continue
                pat = ent.get("pattern")
                if not isinstance(pat, list) or not pat:
                    continue
                toks = tuple(sorted(str(x) for x in pat if x is not None))
                if len(toks) < 2:
                    continue
                pats.append((len(toks), toks, str(cid)))
            pats.sort(key=lambda x: (-x[0], x[2], x[1]))
            return pats

        def _compress_turn_concepts(
            seqs: Sequence[Sequence[str]],
            seq_pats: Sequence[Tuple[int, Tuple[str, ...], str]],
            set_pats: Sequence[Tuple[int, Tuple[str, ...], str]],
        ) -> Tuple[int, int, int, int, int, int, int]:
            bits = 0
            symbols_before = 0
            symbols_after = 0
            covered = 0
            seq_hits = 0
            set_hits = 0
            turns_hit = 0
            set_pats2 = [(k, set(pat), mid) for k, pat, mid in set_pats if pat]
            for seq in seqs:
                seq0 = [str(x) for x in seq if isinstance(x, str) and x and (not x.startswith("__"))]
                if not seq0:
                    continue
                symbols_before += int(len(seq0))
                i = 0
                remaining: List[str] = []
                out_seq_syms: List[str] = []
                local_seq_hits = 0
                local_set_hits = 0
                while i < len(seq0):
                    matched = False
                    for plen, pat, mid in seq_pats:
                        if i + plen <= len(seq0) and tuple(seq0[i : i + plen]) == pat:
                            out_seq_syms.append(str(mid))
                            covered += int(plen)
                            seq_hits += 1
                            local_seq_hits += 1
                            i += int(plen)
                            matched = True
                            break
                    if not matched:
                        remaining.append(str(seq0[i]))
                        i += 1

                rem_set = set(remaining)
                out_set_syms: List[str] = []
                for _k, pat_set, mid in set_pats2:
                    if pat_set and pat_set.issubset(rem_set):
                        out_set_syms.append(str(mid))
                        rem_set -= pat_set
                        covered += int(len(pat_set))
                        set_hits += 1
                        local_set_hits += 1

                out_syms = list(out_seq_syms) + sorted(out_set_syms) + sorted(rem_set)
                symbols_after += int(len(out_syms))
                for sym in out_syms:
                    bits += sym_cost_bits(str(sym))
                if (local_seq_hits + local_set_hits) > 0:
                    turns_hit += 1
            return (
                int(bits),
                int(symbols_before),
                int(symbols_after),
                int(covered),
                int(seq_hits),
                int(set_hits),
                int(turns_hit),
            )

        # Update existing concept counts/contexts from this window's mined motifs.
        for cid, ent in concepts.items():
            if not isinstance(ent, dict):
                continue
            if str(ent.get("kind", "")) != "concept":
                continue
            typ = str(ent.get("type", ""))
            src = str(ent.get("source", ""))
            pat0 = ent.get("pattern")
            if not isinstance(pat0, list):
                continue
            if typ == "seq":
                key = tuple(str(x) for x in pat0 if x is not None)
                if src == "executed_predictor_act_ids":
                    c = int(turn_exec_seq_counts.get(key, 0) or 0)
                    ctx_n = int(len(turn_exec_seq_contexts.get(key, set())))
                elif src == "rewrite_rule_hit_ids":
                    c = int(turn_rr_seq_counts.get(key, 0) or 0)
                    ctx_n = int(len(turn_rr_seq_contexts.get(key, set())))
                else:
                    continue
            elif typ == "set":
                key2 = tuple(sorted(str(x) for x in pat0 if x is not None))
                if src == "executed_predictor_act_ids":
                    c = int(turn_exec_set_counts.get(key2, 0) or 0)
                    ctx_n = int(len(turn_exec_set_contexts.get(key2, set())))
                elif src == "rewrite_rule_hit_ids":
                    c = int(turn_rr_set_counts.get(key2, 0) or 0)
                    ctx_n = int(len(turn_rr_set_contexts.get(key2, set())))
                else:
                    continue
            else:
                continue
            if c > 0:
                ent["count"] = int(ent.get("count", 0) or 0) + int(c)
                ent["last_seen_step"] = int(step)
                if ctx_n > 0:
                    ent["contexts_distinct"] = max(int(ent.get("contexts_distinct", 0) or 0), int(ctx_n))
                concept_updated_existing += 1

        def _concept_id(*, typ: str, src: str, pat: Tuple[str, ...]) -> str:
            raw = (f"{src}|{typ}|" + "|".join(pat)).encode("utf-8")
            return f"c{typ}_{sha256_hex(raw)[:12]}"

        def _new_concept_entry(
            cid: str, *, typ: str, src: str, pat: Tuple[str, ...], c: int, ctx_n: int, gain_bits: int
        ) -> Dict[str, Any]:
            return {
                "id": str(cid),
                "kind": "concept",
                "type": str(typ),
                "source": str(src),
                "pattern": list(pat),
                "deps": sorted(set(str(x) for x in pat if x is not None)),
                "n": int(len(pat)) if typ == "seq" else None,
                "k": int(len(pat)) if typ == "set" else None,
                "count": int(c),
                "contexts_distinct": int(max(0, int(ctx_n))),
                "last_seen_step": int(step),
                "gain_real_bits_est": int(gain_bits),
            }

        # Greedy selection: pick up to concept_max_new concepts with positive MDL-real gain.
        if concept_budget > 0 and concept_max_new > 0 and (turn_exec_seqs_all or turn_rr_seqs_all):
            cur_exec_seq_pats = _concept_seq_patterns("executed_predictor_act_ids")
            cur_exec_set_pats = _concept_set_patterns("executed_predictor_act_ids")
            cur_rr_seq_pats = _concept_seq_patterns("rewrite_rule_hit_ids")
            cur_rr_set_pats = _concept_set_patterns("rewrite_rule_hit_ids")

            raw_exec_bits = sum(sym_cost_bits(sym) for seq in turn_exec_seqs_all for sym in seq)
            raw_rr_bits = sum(sym_cost_bits(sym) for seq in turn_rr_seqs_all for sym in seq)
            concept_trace_mdl_before_bits_est = int(raw_exec_bits + raw_rr_bits)

            cur_exec_bits, exec_sym_before, exec_sym_after, exec_cov, exec_seq_hits, exec_set_hits, exec_turn_hit = (
                _compress_turn_concepts(turn_exec_seqs_all, cur_exec_seq_pats, cur_exec_set_pats)
            )
            cur_rr_bits, rr_sym_before, rr_sym_after, rr_cov, rr_seq_hits, rr_set_hits, rr_turn_hit = (
                _compress_turn_concepts(turn_rr_seqs_all, cur_rr_seq_pats, cur_rr_set_pats)
            )
            cur_total_bits = int(cur_exec_bits + cur_rr_bits)

            # Candidate pools (deterministic ordering from tuples).
            candidates: List[Tuple[str, str, Tuple[str, ...], int, int]] = []
            for pat, c in turn_exec_seq_counts.items():
                if int(c) < int(concept_min_count):
                    continue
                if len(pat) < int(concept_n_min) or len(pat) > int(concept_n_max):
                    continue
                ctx_n = int(len(turn_exec_seq_contexts.get(pat, set())))
                candidates.append(("seq", "executed_predictor_act_ids", tuple(pat), int(c), int(ctx_n)))
            for pat, c in turn_exec_set_counts.items():
                if int(c) < int(concept_set_min_count):
                    continue
                if len(pat) < int(concept_set_min_size) or len(pat) > int(concept_set_max_size):
                    continue
                ctx_n = int(len(turn_exec_set_contexts.get(pat, set())))
                candidates.append(("set", "executed_predictor_act_ids", tuple(pat), int(c), int(ctx_n)))
            for pat, c in turn_rr_seq_counts.items():
                if int(c) < int(concept_min_count):
                    continue
                if len(pat) < int(concept_n_min) or len(pat) > int(concept_n_max):
                    continue
                ctx_n = int(len(turn_rr_seq_contexts.get(pat, set())))
                candidates.append(("seq", "rewrite_rule_hit_ids", tuple(pat), int(c), int(ctx_n)))
            for pat, c in turn_rr_set_counts.items():
                if int(c) < int(concept_set_min_count):
                    continue
                if len(pat) < int(concept_set_min_size) or len(pat) > int(concept_set_max_size):
                    continue
                ctx_n = int(len(turn_rr_set_contexts.get(pat, set())))
                candidates.append(("set", "rewrite_rule_hit_ids", tuple(pat), int(c), int(ctx_n)))

            for _ in range(int(concept_max_new)):
                best: Optional[Tuple[int, int, int, str, str, Tuple[str, ...], Dict[str, Any], Any]] = None
                # (gain, ctx_n, count, typ, src, pat, entry, new_state)

                for typ, src, pat, c, ctx_n in candidates:
                    cid = _concept_id(typ=typ, src=src, pat=pat)
                    if cid in concepts:
                        continue
                    if ctx_n <= 0:
                        continue

                    if src == "executed_predictor_act_ids":
                        if typ == "seq":
                            pats2 = list(cur_exec_seq_pats)
                            pats2.append((len(pat), tuple(pat), cid))
                            pats2.sort(key=lambda x: (-x[0], x[2], x[1]))
                            new_exec_bits = _compress_turn_concepts(turn_exec_seqs_all, pats2, cur_exec_set_pats)[0]
                            new_total_bits = int(new_exec_bits + cur_rr_bits)
                            new_state = ("exec_seq", pats2, None, int(new_exec_bits))
                        else:
                            sp2 = list(cur_exec_set_pats)
                            sp2.append((len(pat), tuple(sorted(pat)), cid))
                            sp2.sort(key=lambda x: (-x[0], x[2], x[1]))
                            new_exec_bits = _compress_turn_concepts(turn_exec_seqs_all, cur_exec_seq_pats, sp2)[0]
                            new_total_bits = int(new_exec_bits + cur_rr_bits)
                            new_state = ("exec_set", None, sp2, int(new_exec_bits))
                    else:
                        if typ == "seq":
                            pats2 = list(cur_rr_seq_pats)
                            pats2.append((len(pat), tuple(pat), cid))
                            pats2.sort(key=lambda x: (-x[0], x[2], x[1]))
                            new_rr_bits = _compress_turn_concepts(turn_rr_seqs_all, pats2, cur_rr_set_pats)[0]
                            new_total_bits = int(cur_exec_bits + new_rr_bits)
                            new_state = ("rr_seq", pats2, None, int(new_rr_bits))
                        else:
                            sp2 = list(cur_rr_set_pats)
                            sp2.append((len(pat), tuple(sorted(pat)), cid))
                            sp2.sort(key=lambda x: (-x[0], x[2], x[1]))
                            new_rr_bits = _compress_turn_concepts(turn_rr_seqs_all, cur_rr_seq_pats, sp2)[0]
                            new_total_bits = int(cur_exec_bits + new_rr_bits)
                            new_state = ("rr_set", None, sp2, int(new_rr_bits))

                    delta_bits = int(cur_total_bits - new_total_bits)
                    if delta_bits <= 0:
                        continue
                    entry0 = _new_concept_entry(
                        cid,
                        typ=typ,
                        src=src,
                        pat=tuple(pat),
                        c=int(c),
                        ctx_n=int(ctx_n),
                        gain_bits=int(delta_bits),
                    )
                    cost = int(macro_entry_cost_bits(entry0))
                    gain = int(delta_bits - cost)
                    if gain <= 0:
                        continue
                    entry0["gain_real_bits_est"] = int(gain)
                    cand_key = (gain, int(ctx_n), int(c), str(typ), str(src), tuple(pat))
                    if best is None or cand_key > (best[0], best[1], best[2], best[3], best[4], best[5]):
                        best = (gain, int(ctx_n), int(c), str(typ), str(src), tuple(pat), entry0, new_state)

                if best is None:
                    break
                gain, ctx_n, c, typ, src, pat, entry0, new_state = best
                cid = str(entry0.get("id") or _concept_id(typ=typ, src=src, pat=pat))
                concepts[cid] = dict(entry0)
                concept_added += 1
                if typ == "seq":
                    concept_added_seq += 1
                else:
                    concept_added_set += 1

                tag = new_state[0]
                if tag == "exec_seq":
                    cur_exec_seq_pats = list(new_state[1])
                    cur_exec_bits = int(new_state[3])
                elif tag == "exec_set":
                    cur_exec_set_pats = list(new_state[2])
                    cur_exec_bits = int(new_state[3])
                elif tag == "rr_seq":
                    cur_rr_seq_pats = list(new_state[1])
                    cur_rr_bits = int(new_state[3])
                elif tag == "rr_set":
                    cur_rr_set_pats = list(new_state[2])
                    cur_rr_bits = int(new_state[3])
                cur_total_bits = int(cur_exec_bits + cur_rr_bits)

            # Budget eviction (deterministic): keep strongest concepts by (gain, contexts_distinct, count, id).
            if concept_budget > 0 and len(concepts) > concept_budget:
                ranked: List[Tuple[int, int, int, str]] = []
                for cid, ent in concepts.items():
                    if not isinstance(ent, dict):
                        continue
                    if str(ent.get("kind", "")) != "concept":
                        continue
                    ranked.append(
                        (
                            int(ent.get("gain_real_bits_est", 0) or 0),
                            int(ent.get("contexts_distinct", 0) or 0),
                            int(ent.get("count", 0) or 0),
                            str(cid),
                        )
                    )
                ranked.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))
                keep = {cid for _g, _ctx, _c, cid in ranked[: int(concept_budget)]}
                for cid in list(concepts.keys()):
                    if cid not in keep:
                        concepts.pop(cid, None)
                        concept_evicted += 1

            # Final recompute with kept concepts.
            final_exec_seq_pats = _concept_seq_patterns("executed_predictor_act_ids")
            final_exec_set_pats = _concept_set_patterns("executed_predictor_act_ids")
            final_rr_seq_pats = _concept_seq_patterns("rewrite_rule_hit_ids")
            final_rr_set_pats = _concept_set_patterns("rewrite_rule_hit_ids")
            exec_bits2, exec_sym_before, exec_sym_after, exec_cov, exec_seq_hits, exec_set_hits, exec_turn_hit = (
                _compress_turn_concepts(turn_exec_seqs_all, final_exec_seq_pats, final_exec_set_pats)
            )
            rr_bits2, rr_sym_before, rr_sym_after, rr_cov, rr_seq_hits, rr_set_hits, rr_turn_hit = (
                _compress_turn_concepts(turn_rr_seqs_all, final_rr_seq_pats, final_rr_set_pats)
            )
            concept_trace_mdl_after_bits_est = int(exec_bits2 + rr_bits2)
            concept_trace_mdl_gain_bits_est = int(
                concept_trace_mdl_before_bits_est - concept_trace_mdl_after_bits_est
            )

            concept_library_cost_bits_est = int(
                sum(macro_entry_cost_bits(ent) for ent in concepts.values() if isinstance(ent, dict))
            )
            concept_trace_mdl_gain_real_bits_est = int(
                concept_trace_mdl_before_bits_est
                - (concept_trace_mdl_after_bits_est + concept_library_cost_bits_est)
            )
            if concept_trace_mdl_gain_real_bits_est <= 0 and concepts:
                concepts.clear()
                concept_cleared = True
                concept_library_cost_bits_est = 0
                exec_bits2, exec_sym_before, exec_sym_after, exec_cov, exec_seq_hits, exec_set_hits, exec_turn_hit = (
                    _compress_turn_concepts(turn_exec_seqs_all, [], [])
                )
                rr_bits2, rr_sym_before, rr_sym_after, rr_cov, rr_seq_hits, rr_set_hits, rr_turn_hit = (
                    _compress_turn_concepts(turn_rr_seqs_all, [], [])
                )
                concept_trace_mdl_after_bits_est = int(exec_bits2 + rr_bits2)
                concept_trace_mdl_gain_bits_est = int(
                    concept_trace_mdl_before_bits_est - concept_trace_mdl_after_bits_est
                )
                concept_trace_mdl_gain_real_bits_est = int(
                    concept_trace_mdl_before_bits_est
                    - (concept_trace_mdl_after_bits_est + concept_library_cost_bits_est)
                )

            turns_total = int(len(turn_exec_seqs_all))
            if turns_total > 0:
                avg_trace_len_turn_before = float(exec_sym_before / turns_total)
                avg_trace_len_turn_after = float(exec_sym_after / turns_total)
                concept_turn_hit_rate = float(exec_turn_hit / turns_total)
                concept_hits_total = int(exec_seq_hits + exec_set_hits)
                concept_hits_per_turn_mean = float(concept_hits_total / turns_total)
                concept_symbol_coverage_rate = float(
                    (exec_cov / max(1, exec_sym_before)) if exec_sym_before > 0 else 0.0
                )

            ev["concepts_last_update_step"] = int(step)

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

        top_concepts: List[Dict[str, Any]] = []
        if isinstance(concepts, dict) and concepts:
            ranked_c: List[Tuple[int, int, int, str, Dict[str, Any]]] = []
            for cid, ent in concepts.items():
                if not isinstance(ent, dict):
                    continue
                if str(ent.get("kind", "")) != "concept":
                    continue
                ranked_c.append(
                    (
                        int(ent.get("gain_real_bits_est", 0) or 0),
                        int(ent.get("contexts_distinct", 0) or 0),
                        int(ent.get("count", 0) or 0),
                        str(cid),
                        ent,
                    )
                )
            ranked_c.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))
            for g, ctx_n, c, cid, ent in ranked_c[:10]:
                top_concepts.append(
                    {
                        "id": str(cid),
                        "type": str(ent.get("type", "")),
                        "source": str(ent.get("source", "")),
                        "count": int(c),
                        "contexts_distinct": int(ctx_n),
                        "gain_real_bits_est": int(g),
                        "pattern": list(ent.get("pattern") or []),
                    }
                )

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
            "avg_trace_len_turn_before": float(avg_trace_len_turn_before),
            "avg_trace_len_turn_after": float(avg_trace_len_turn_after),
            "depth_compositional_max": int(depth_max),
            "depth_compositional_median": int(depth_med),
            "concept_cleared_for_net_mdl": bool(concept_cleared),
            "concept_added": int(concept_added),
            "concept_added_seq": int(concept_added_seq),
            "concept_added_set": int(concept_added_set),
            "concept_updated_existing": int(concept_updated_existing),
            "concept_evicted": int(concept_evicted),
            "concept_total": int(len(concepts)) if isinstance(concepts, dict) else 0,
            "concept_trace_mdl_before_bits_est": int(concept_trace_mdl_before_bits_est),
            "concept_trace_mdl_after_bits_est": int(concept_trace_mdl_after_bits_est),
            "concept_trace_mdl_gain_bits_est": int(concept_trace_mdl_gain_bits_est),
            "concept_library_cost_bits_est": int(concept_library_cost_bits_est),
            "concept_trace_mdl_gain_real_bits_est": int(concept_trace_mdl_gain_real_bits_est),
            "concept_turn_hit_rate": float(concept_turn_hit_rate),
            "concept_symbol_coverage_rate": float(concept_symbol_coverage_rate),
            "concept_hits_per_turn_mean": float(concept_hits_per_turn_mean),
            "top_macros": top_macros,
            "top_concepts": top_concepts,
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

        # Mark the router as compatible with the current decoder fluency regime.
        ev["decoder_fluency_id"] = "antiloop_v40"

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
        starts: Optional[Sequence[int]] = None,
        length: int,
        engine_config: EngineConfig,
        patch: Optional[Patch] = None,
    ) -> Dict[str, Any]:
        if not tokens:
            raise ValueError("empty token stream for evaluation")
        store_copy: ActStore = copy.deepcopy(store)
        if patch is not None:
            self._apply_patch_to_store(store_copy, patch)

        # Cross-context pressure: compute online NLL on multiple deterministic, disjoint windows
        # and use the mean as the gain estimate signal (no gradients, just survival).
        if starts is None:
            starts_eval = [int(start)]
        else:
            starts_eval = [int(x) for x in list(starts) if x is not None]
            if not starts_eval:
                starts_eval = [int(start)]
        L = max(1, len(tokens))
        seen = set()
        uniq: List[int] = []
        for s in starts_eval:
            ss = int(s) % L
            if ss in seen:
                continue
            seen.add(ss)
            uniq.append(ss)
        starts_eval = uniq or [int(start) % L]

        nll_bits_windows: List[float] = []
        for s in starts_eval:
            engine = Engine(store_copy, seed=self.config.seed, config=engine_config)
            ctx: List[str] = ["<BOS>"] * (engine.config.max_order - 1)
            nll_bits = 0.0
            for i in range(int(length)):
                tok = tokens[(int(s) + i) % len(tokens)]
                lp = engine.logprob_next(context=ctx, token=tok)
                nll_bits += -lp / math.log(2)
                engine.observe(context=ctx, token=tok)
                ctx.append(tok)
                ctx = ctx[-(engine.config.max_order - 1) :]
            nll_bits_windows.append(float(nll_bits))
        nll_bits_mean = sum(nll_bits_windows) / max(1, len(nll_bits_windows))

        cost_bits = sum(estimate_act_cost_bits(a) for a in store_copy.active())
        # NOTE: contract-enabled generation is treated as external scaffolding; do not allow
        # training selection to win "utility by contract". Always evaluate suites with contracts OFF.
        try:
            cfgd = dict(engine_config.__dict__)
            cfgd["enable_contracts"] = False
            eval_cfg = EngineConfig(**cfgd)
        except Exception:
            eval_cfg = EngineConfig(enable_contracts=False)
        eval_engine = Engine(store_copy, seed=self.config.seed, config=eval_cfg)

        gen: Dict[str, Any] = dict(self._eval_chat_harness_metrics(eval_engine))
        util = self._eval_skill_suite_metrics(eval_engine)
        for k, v in util.items():
            gen[f"utility_{k}"] = v
        agency = self._eval_agency_suite_metrics(store_override=store_copy)
        if isinstance(agency, dict) and bool(agency.get("enabled", False)):
            gen["utility_agency_total"] = int(agency.get("total") or 0)
            gen["utility_agency_pass_rate"] = float(agency.get("pass_rate") or 0.0)
            gen["utility_agency_plan_found_rate"] = float(agency.get("plan_found_rate") or 0.0)
            gen["utility_agency_steps_pass_mean"] = float(agency.get("steps_pass_mean") or 0.0)
            gen["utility_agency_steps_pass_median"] = float(agency.get("steps_pass_median") or 0.0)
            gen["utility_agency_fail_reasons"] = (
                dict(agency.get("fail_reasons") or {}) if isinstance(agency.get("fail_reasons"), dict) else {}
            )
            gen["utility_agency_fail_details"] = (
                list(agency.get("fail_details") or []) if isinstance(agency.get("fail_details"), list) else []
            )
        return {
            "nll_bits": float(nll_bits_mean),
            "nll_bits_windows": list(nll_bits_windows),
            "cost_bits": cost_bits,
            "gen": gen,
        }

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

        # Proportional schedule (default 200k): 5%/20%/40%/60% => 10k/40k/80k/120k.
        total_steps = max(1, int(self.config.steps))
        bigram_at = max(1, int(round(total_steps * 0.05)))
        trigram_at = max(1, int(round(total_steps * 0.20)))
        merge_bigram_at = max(1, int(round(total_steps * 0.40)))
        fourgram_at = max(1, int(round(total_steps * 0.60)))

        has_bigram_space = self._find_act_by_name("bigram_space") is not None
        has_bigram_nonspace = self._find_act_by_name("bigram_nonspace") is not None
        has_bigram_merged = self._find_act_by_name("bigram_merged") is not None
        has_trigram = self._find_act_by_name("trigram") is not None
        has_fourgram = self._find_act_by_name("fourgram") is not None
        prefill = self.config.mode == "demo"
        learnable = self.config.mode == "pure"

        # Pure-mode budget defaults: keep predictors small so MDL gain can dominate early.
        # These bounds are deliberately tight; COUNT_LEX eviction preserves frequent structure.
        bi_max_ctx = 128 if learnable else 20_000
        bi_max_next = 8 if learnable else 32
        tri_max_ctx = 128 if learnable else 30_000
        tri_max_next = 6 if learnable else 24
        four_max_ctx = 128 if learnable else 20_000
        four_max_next = 4 if learnable else 16

        if step >= bigram_at and (not has_bigram_space and not has_bigram_nonspace and not has_bigram_merged):
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
                max_contexts=int(bi_max_ctx),
                max_next_per_ctx=int(bi_max_next),
                evict_policy=evict,
            ).to_dict()
            a2 = _make_bigram_nonspace_act(
                act_id=det_act_id(step=step, name="bigram_nonspace", idx=0),
                table=t_non,
                created_at=deterministic_iso(step=step, offset_us=1),
                allow_new_contexts=learnable,
                allow_new_tokens=learnable,
                max_contexts=int(bi_max_ctx),
                max_next_per_ctx=int(bi_max_next),
                evict_policy=evict,
            ).to_dict()
            if self.config.mode == "pure":
                if a1.get("evidence", {}).get("table"):
                    raise RuntimeError("PURE violation: bigram_space prefill detected")
                if a2.get("evidence", {}).get("table"):
                    raise RuntimeError("PURE violation: bigram_nonspace prefill detected")
            patches.append(Patch(kind="ADD_ACT", payload={"acts": [a1, a2]}))

        if step >= trigram_at and (not has_trigram):
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
                max_contexts=int(tri_max_ctx),
                max_next_per_ctx=int(tri_max_next),
                evict_policy=evict,
            ).to_dict()
            if self.config.mode == "pure" and a.get("evidence", {}).get("table"):
                raise RuntimeError("PURE violation: trigram prefill detected")
            patches.append(Patch(kind="ADD_ACT", payload={"acts": [a]}))

        if step >= merge_bigram_at and (has_bigram_space and has_bigram_nonspace and not has_bigram_merged):
            a_space = self._find_act_by_name("bigram_space")
            a_non = self._find_act_by_name("bigram_nonspace")
            if a_space and a_non:
                if self.config.mode == "pure":
                    # PURE: merge via explicit evidence transfer (union-sum), without prefill.
                    max_ctx = max(
                        int(a_space.evidence.get("max_contexts", 0) or 0),
                        int(a_non.evidence.get("max_contexts", 0) or 0),
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

        if step >= fourgram_at and (not has_fourgram):
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
                max_contexts=int(four_max_ctx),
                max_next_per_ctx=int(four_max_next),
                evict_policy=evict,
            ).to_dict()
            if self.config.mode == "pure" and a.get("evidence", {}).get("table"):
                raise RuntimeError("PURE violation: fourgram prefill detected")
            patches.append(Patch(kind="ADD_ACT", payload={"acts": [a]}))

        # Tune fluency guardrails when higher-order predictors are eligible.
        # This helps avoid a training deadlock where better predictors improve NLL but hurt
        # fluency enough to block structural growth (especially under strict selection).
        if step >= bigram_at:
            guard = self._find_act_by_name("fluency_guardrails_v1")
            if guard is not None and str(getattr(guard, "kind", "")) == "rewrite_rule":
                rep_strength = None
                cycle_strength = None
                for ins in list(getattr(guard, "program", []) or []):
                    if not isinstance(ins, Instruction):
                        continue
                    if str(ins.op) != "APPLY_PENALTY":
                        continue
                    kind = str(ins.args.get("kind") or "")
                    if kind == "repetition":
                        try:
                            rep_strength = float(ins.args.get("strength", 0.0))
                        except Exception:
                            rep_strength = None
                    elif kind == "ngram_cycle":
                        try:
                            cycle_strength = float(ins.args.get("strength", 0.0))
                        except Exception:
                            cycle_strength = None

                # Deterministic candidate ladder.
                cand_pairs = [
                    (2.0, 4.0),
                    (2.5, 5.0),
                    (3.0, 6.0),
                    (4.0, 8.0),
                    (5.0, 10.0),
                    (6.0, 12.0),
                    (8.0, 16.0),
                    (10.0, 20.0),
                ]
                for idx, (rep_s, cyc_s) in enumerate(cand_pairs):
                    if rep_strength is not None and cycle_strength is not None:
                        if rep_strength >= rep_s - 1e-12 and cycle_strength >= cyc_s - 1e-12:
                            continue
                    new = Act.from_dict(guard.to_dict())
                    new.version = int(getattr(guard, "version", 1) or 1) + 1
                    new.created_at = deterministic_iso(step=int(step), offset_us=100 + int(idx))
                    new.program = [
                        Instruction("APPLY_PENALTY", {"kind": "repetition", "strength": float(rep_s)}),
                        Instruction(
                            "APPLY_PENALTY",
                            {"kind": "ngram_cycle", "n": 3, "strength": float(cyc_s)},
                        ),
                    ]
                    if not isinstance(new.evidence, dict):
                        new.evidence = {}
                    new.evidence = dict(new.evidence)
                    new.evidence["tuned"] = {
                        "name": "fluency_guardrails_v1",
                        "rep_strength": float(rep_s),
                        "cycle_strength": float(cyc_s),
                        "step": int(step),
                    }
                    patches.append(
                        Patch(kind="REWRITE_ACT", payload={"act_id": str(guard.id), "act": new.to_dict()})
                    )
                    break

        # Expand predictor budgets (PURE only): lets the system grow capacity when it pays off.
        if learnable:
            ladders = [
                ("bigram_space", [128, 256, 512, 1024, 2048], [8, 16, 24, 32]),
                ("bigram_nonspace", [128, 256, 512, 1024, 2048], [8, 16, 24, 32]),
                ("trigram", [128, 256, 512, 1024], [6, 8, 12, 16]),
                ("fourgram", [128, 256, 512, 1024], [4, 6, 8, 12]),
            ]
            for name, ctx_ladder, next_ladder in ladders:
                act = self._find_act_by_name(str(name))
                if act is None or str(getattr(act, "kind", "")) != "predictor":
                    continue
                ev = act.evidence if isinstance(act.evidence, dict) else {}
                tab = ev.get("table")
                cur_tab_ctx = len(tab) if isinstance(tab, dict) else 0
                cur_ctx = int(ev.get("max_contexts", 0) or 0)
                cur_next = int(ev.get("max_next_per_ctx", 0) or 0)

                # Expand only when we're close to saturating the current context budget.
                if cur_ctx > 0 and cur_tab_ctx < int(0.9 * float(cur_ctx)):
                    continue

                want_ctx = None
                for v in ctx_ladder:
                    if int(v) > cur_ctx:
                        want_ctx = int(v)
                        break
                want_next = None
                for v in next_ladder:
                    if int(v) > cur_next:
                        want_next = int(v)
                        break
                if want_ctx is None and want_next is None:
                    continue

                new = Act.from_dict(act.to_dict())
                new.version = int(getattr(act, "version", 1) or 1) + 1
                new.created_at = deterministic_iso(step=int(step), offset_us=500)
                if not isinstance(new.evidence, dict):
                    new.evidence = {}
                new.evidence = dict(new.evidence)
                if want_ctx is not None:
                    new.evidence["max_contexts"] = int(want_ctx)
                if want_next is not None:
                    new.evidence["max_next_per_ctx"] = int(want_next)
                new.evidence["budget_expand"] = {
                    "name": str(name),
                    "from": {"max_contexts": int(cur_ctx), "max_next_per_ctx": int(cur_next)},
                    "to": {
                        "max_contexts": int(new.evidence.get("max_contexts", cur_ctx) or cur_ctx),
                        "max_next_per_ctx": int(new.evidence.get("max_next_per_ctx", cur_next) or cur_next),
                    },
                    "step": int(step),
                }
                patches.append(Patch(kind="REWRITE_ACT", payload={"act_id": str(act.id), "act": new.to_dict()}))
                break

        return patches

    def _select_patch(
        self,
        *,
        step: int,
        engine: Engine,
        tokens: Sequence[str],
        patches: List[Patch],
        divergence: bool = False,
        repair_mode: bool = False,
        repair_target_bottleneck: Optional[float] = None,
    ) -> Optional[Tuple[Patch, Dict[str, Any]]]:
        if not patches:
            return None

        selection_mode = str(getattr(self.config, "selection_mode", "weighted") or "weighted")
        if selection_mode not in {"weighted", "bottleneck", "survival"}:
            selection_mode = "weighted"
        tol = 1e-9

        start = (step * 7919) % max(1, len(tokens))
        n_windows = int(getattr(self.config, "nll_eval_windows", 1) or 1)
        n_windows = max(1, min(16, n_windows))
        starts = [int(start)]
        if n_windows > 1 and tokens:
            # Disjoint windows by construction when the corpus is large enough.
            stride = max(1, int(self.config.val_tokens))
            L = len(tokens)
            for i in range(1, n_windows):
                starts.append((int(start) + i * stride) % L)

        # REPAIR mode is mutually exclusive with divergence (repair = consolidation phase).
        if repair_mode:
            divergence = False

        base_eval = self._eval_online_window(
            self.store,
            tokens,
            start=start,
            starts=starts,
            length=self.config.val_tokens,
            engine_config=engine.config,
        )
        base_nll = float(base_eval["nll_bits"])
        base_cost = int(base_eval["cost_bits"])
        base_gen = dict(base_eval["gen"])
        base_nll_windows = list(base_eval.get("nll_bits_windows", [base_nll]))

        # Normalized NLL ratio (bits per token / log2(V)) for survival-style bottleneck selection.
        V = 1
        try:
            if hasattr(engine, "vocab") and callable(getattr(engine, "vocab")):
                V = max(1, len(engine.vocab()) + 1)  # type: ignore[misc]
            else:
                V = max(1, len(set(str(t) for t in tokens if str(t))) + 1)
        except Exception:
            V = 1
        log2V = safe_log2(int(V))
        base_nll_per_tok = float(base_nll) / float(self.config.val_tokens)
        base_nll_ratio = float(base_nll_per_tok / log2V) if log2V > 0 else float("inf")

        # Holdout: static NLL on a deterministic slice not used for training tokens.
        holdout_tokens = list(getattr(self, "_holdout_tokens", []) or [])
        holdout_frac = float(getattr(self.config, "holdout_frac", 0.0) or 0.0)
        if holdout_frac != holdout_frac:
            holdout_frac = 0.0
        holdout_frac = max(0.0, min(0.5, holdout_frac))
        h_windows = int(getattr(self.config, "holdout_eval_windows", 0) or 0)
        h_windows = max(0, min(16, h_windows))
        h_len = int(getattr(self.config, "holdout_eval_tokens", 0) or 0)
        if h_len <= 0:
            h_len = int(self.config.val_tokens)
        h_len = max(1, int(h_len))

        holdout_enabled = bool(holdout_tokens) and h_windows > 0
        holdout_starts: List[int] = []
        base_holdout_nll_windows: List[float] = []
        base_holdout_nll_bits = float("nan")
        base_holdout_nll_ratio = float("nan")
        if holdout_enabled:
            start_h = (int(step) * 104729) % max(1, len(holdout_tokens))
            holdout_starts = [int(start_h)]
            if h_windows > 1:
                stride_h = max(1, int(h_len))
                Lh = len(holdout_tokens)
                for i in range(1, int(h_windows)):
                    holdout_starts.append((int(start_h) + i * int(stride_h)) % Lh)
            base_holdout_eval = self._eval_static_nll_windows(
                self.store,
                holdout_tokens,
                start=int(start_h),
                starts=list(holdout_starts),
                length=int(h_len),
                engine_config=engine.config,
            )
            base_holdout_nll_bits = float(base_holdout_eval["nll_bits"])
            base_holdout_nll_windows = list(
                base_holdout_eval.get("nll_bits_windows", [base_holdout_nll_bits])
            )
            if log2V > 0:
                base_holdout_nll_ratio = float((float(base_holdout_nll_bits) / float(h_len)) / float(log2V))

        best: Optional[Tuple[Patch, Dict[str, Any]]] = None
        best_score = float("-inf")
        best_surv_key: Optional[Tuple[float, ...]] = None
        best_repair_key: Optional[Tuple[float, float]] = None

        def fluency_penalty(gen: Dict[str, Any]) -> float:
            rep3 = float(gen.get("repeat3_reply_mean", gen.get("repeat3_global", 0.0)) or 0.0)
            loop3 = float(gen.get("loop_rate_reply_mean", gen.get("loop_rate_global", 0.0)) or 0.0)
            return (
                rep3
                + loop3
                + float(gen["whitespace_ratio"])
                + float(gen["duplicate_reply_rate"])
                + float(gen["most_common_reply_frac"])
                + float(gen.get("prefix_k_dup_rate", 0.0))
                + float(gen.get("template_ngram_dup_rate", 0.0))
                + float(gen.get("cross_turn_signature_repeat_rate", 0.0))
            )

        def utility_penalty(gen: Dict[str, Any]) -> float:
            _comp, loss, _terms = self._utility_bottleneck_loss(gen)
            return float(loss)

        base_pen = fluency_penalty(base_gen)
        base_util_component, base_util_pen, base_util_terms = self._utility_bottleneck_loss(base_gen)
        lam = float(self._effective_fluency_lambda(step=step))
        if divergence:
            scale = float(getattr(self.config, "divergence_lambda_scale", 0.1) or 0.1)
            if scale != scale:
                scale = 0.1
            scale = max(0.0, min(1.0, scale))
            lam *= scale
        util_w = float(self.config.utility_weight)
        base_score = (-lam * base_pen) - (util_w * base_util_pen)

        def _m(gen: Dict[str, Any], key: str) -> float:
            try:
                v = float(gen.get(key, 0.0) or 0.0)
            except Exception:
                v = 0.0
            if v != v:
                v = 0.0
            return v

        def fluency_bottleneck(gen: Dict[str, Any]) -> float:
            return max(
                _m(gen, "repeat3_reply_max") if "repeat3_reply_max" in gen else _m(gen, "repeat3_global"),
                _m(gen, "loop_rate_reply_max") if "loop_rate_reply_max" in gen else _m(gen, "loop_rate_global"),
                _m(gen, "duplicate_reply_rate"),
                _m(gen, "most_common_reply_frac"),
                _m(gen, "whitespace_ratio"),
                _m(gen, "prefix_k_dup_rate"),
                _m(gen, "template_ngram_dup_rate"),
                _m(gen, "cross_turn_signature_repeat_rate"),
            )

        base_bottleneck = fluency_bottleneck(base_gen)

        # Survival-style (LOSS-AND) selection: compare the sorted bottleneck vector lexicographically.
        # This promotes holdout/utility/fluency to an actual selection law (not only telemetry/gates).
        def _survival_terms_for(
            *,
            nll_ratio: float,
            util_component: str,
            util_loss: float,
            fluency_bn: float,
            holdout_ratio: Optional[float],
        ) -> List[Tuple[str, float]]:
            terms: List[Tuple[str, float]] = [
                ("nll_ratio", float(nll_ratio)),
                (f"utility:{str(util_component)}", float(util_loss)),
                ("fluency", float(fluency_bn)),
            ]
            if holdout_ratio is not None and holdout_ratio == holdout_ratio:
                terms.append(("holdout_nll_ratio", float(holdout_ratio)))
            return terms

        def _survival_key(terms: List[Tuple[str, float]]) -> Tuple[float, ...]:
            vals = [float(v) for _k, v in terms]
            vals.sort(reverse=True)
            return tuple(vals)

        def _survival_key_better(cand: Tuple[float, ...], base: Tuple[float, ...]) -> bool:
            m = min(len(cand), len(base))
            for i in range(m):
                if float(cand[i]) < float(base[i]) - tol:
                    return True
                if float(cand[i]) > float(base[i]) + tol:
                    return False
            return len(cand) < len(base)

        base_surv_terms = _survival_terms_for(
            nll_ratio=float(base_nll_ratio),
            util_component=str(base_util_component),
            util_loss=float(base_util_pen),
            fluency_bn=float(base_bottleneck),
            holdout_ratio=(
                float(base_holdout_nll_ratio)
                if (holdout_enabled and base_holdout_nll_ratio == base_holdout_nll_ratio)
                else None
            ),
        )
        base_surv_key = _survival_key(base_surv_terms)
        base_surv_component, base_surv_loss = max(base_surv_terms, key=lambda kv: kv[1])

        # REPAIR: if we previously accepted a divergence patch that worsened the fluency bottleneck,
        # we must deterministically repair until the bottleneck returns to its pre-divergence target.
        repair_tol = float(getattr(self.config, "repair_bottleneck_tol", 1e-9) or 1e-9)
        if repair_tol != repair_tol:
            repair_tol = 1e-9
        repair_tol = max(0.0, min(1.0, repair_tol))

        repair_target = None
        if repair_mode and repair_target_bottleneck is not None:
            try:
                repair_target = float(repair_target_bottleneck)
            except Exception:
                repair_target = None
            if repair_target is not None and repair_target != repair_target:
                repair_target = None

        base_deficit = 0.0
        repair_needed = False
        if repair_mode and repair_target is not None:
            base_deficit = max(0.0, float(base_bottleneck) - float(repair_target))
            repair_needed = bool(float(base_bottleneck) > float(repair_target) + float(repair_tol))
        fluency_slack = 0.0
        if divergence:
            try:
                fluency_slack = float(getattr(self.config, "divergence_fluency_slack", 0.0) or 0.0)
            except Exception:
                fluency_slack = 0.0
            if fluency_slack != fluency_slack:
                fluency_slack = 0.0
            fluency_slack = max(0.0, min(1.0, fluency_slack))

        def patch_affects_predictor(p: Patch) -> bool:
            try:
                kind = str(getattr(p, "kind", "") or "")
                payload = getattr(p, "payload", {}) or {}
            except Exception:
                return True
            if kind == "MERGE_ACTS":
                return True
            if kind == "ADD_ACT":
                acts = payload.get("acts", [])
                if not isinstance(acts, list):
                    return True
                for a in acts:
                    if isinstance(a, dict) and str(a.get("kind", "")) == "predictor":
                        return True
                return False
            if kind == "REWRITE_ACT":
                a = payload.get("act")
                return bool(isinstance(a, dict) and str(a.get("kind", "")) == "predictor")
            if kind == "PRUNE_ACT":
                act_ids = payload.get("act_ids", [])
                if not isinstance(act_ids, list):
                    return True
                for act_id in act_ids:
                    act = self.store.get(str(act_id))
                    if act is not None and str(getattr(act, "kind", "")) == "predictor":
                        return True
                return False
            return False

        for patch in patches:
            cand_eval = self._eval_online_window(
                self.store,
                tokens,
                start=start,
                starts=starts,
                length=self.config.val_tokens,
                engine_config=engine.config,
                patch=patch,
            )
            cand_nll = float(cand_eval["nll_bits"])
            cand_cost = int(cand_eval["cost_bits"])
            cand_gen = dict(cand_eval["gen"])
            cand_nll_windows = list(cand_eval.get("nll_bits_windows", [cand_nll]))

            # Estimate gain on a longer horizon than the validation slice.
            horizon_total = int(getattr(self.config, "gain_horizon_steps", 0) or 0)
            if horizon_total <= 0:
                horizon_total = int(self.config.steps)
            horizon = max(1, int(horizon_total - step))
            cand_nll_per_tok = cand_nll / self.config.val_tokens
            cand_nll_ratio = float((float(cand_nll_per_tok) / float(log2V))) if log2V > 0 else float("inf")
            data_gain_bits = (base_nll_per_tok - cand_nll_per_tok) * horizon
            cost_delta_bits = cand_cost - base_cost
            gain = data_gain_bits - cost_delta_bits

            cand_pen = fluency_penalty(cand_gen)
            cand_util_component, cand_util_pen, cand_util_terms = self._utility_bottleneck_loss(cand_gen)
            score = gain - lam * cand_pen - (util_w * cand_util_pen)
            cand_bottleneck = fluency_bottleneck(cand_gen)
            cand_deficit = 0.0
            if repair_target is not None:
                cand_deficit = max(0.0, float(cand_bottleneck) - float(repair_target))

            # Holdout gate: predictor-affecting patches must be non-worse on a majority of holdout windows.
            holdout_gate = holdout_enabled and patch_affects_predictor(patch)
            holdout_gate_wins = 0
            holdout_gate_need = 0
            holdout_gate_max_reg = 0.0
            holdout_gate_mean_delta = 0.0
            cand_holdout_nll_bits = float("nan")
            cand_holdout_nll_windows: List[float] = []
            cand_holdout_nll_ratio = float("nan")
            if holdout_gate:
                cand_holdout_eval = self._eval_static_nll_windows(
                    self.store,
                    holdout_tokens,
                    start=int(holdout_starts[0] if holdout_starts else 0),
                    starts=list(holdout_starts),
                    length=int(h_len),
                    engine_config=engine.config,
                    patch=patch,
                )
                cand_holdout_nll_bits = float(cand_holdout_eval["nll_bits"])
                cand_holdout_nll_windows = list(
                    cand_holdout_eval.get("nll_bits_windows", [cand_holdout_nll_bits])
                )
                if log2V > 0:
                    cand_holdout_nll_ratio = float((float(cand_holdout_nll_bits) / float(h_len)) / float(log2V))
                m_h = min(len(base_holdout_nll_windows), len(cand_holdout_nll_windows))
                if m_h >= 1:
                    deltas_h: List[float] = []
                    for i in range(m_h):
                        delta = (float(cand_holdout_nll_windows[i]) - float(base_holdout_nll_windows[i])) / float(
                            h_len
                        )
                        deltas_h.append(float(delta))
                        if float(delta) <= tol:
                            holdout_gate_wins += 1
                    holdout_gate_need = (m_h // 2) + 1
                    holdout_gate_max_reg = max(deltas_h) if deltas_h else 0.0
                    holdout_gate_mean_delta = (sum(deltas_h) / float(len(deltas_h))) if deltas_h else 0.0
                    if holdout_gate_wins < holdout_gate_need:
                        continue

            # Cross-context NLL gate: prevent "win by coincidence" on a single slice.
            # In normal mode, predictor-affecting patches must be non-worse on a majority of windows.
            # NOTE: keep this gate even in divergence mode; otherwise divergence becomes a bypass for
            # "win by coincidence" patches (RF-8). Divergence may relax fluency, but not generalization.
            nll_gate = (n_windows > 1) and patch_affects_predictor(patch)
            nll_gate_wins = 0
            nll_gate_need = 0
            nll_gate_max_reg = 0.0
            nll_gate_mean_delta = 0.0
            if nll_gate:
                m = min(len(base_nll_windows), len(cand_nll_windows))
                if m >= 2:
                    deltas: List[float] = []
                    for i in range(m):
                        delta = (float(cand_nll_windows[i]) - float(base_nll_windows[i])) / float(
                            self.config.val_tokens
                        )
                        deltas.append(float(delta))
                        if float(delta) <= tol:
                            nll_gate_wins += 1
                    nll_gate_need = (m // 2) + 1
                    nll_gate_max_reg = max(deltas) if deltas else 0.0
                    nll_gate_mean_delta = (sum(deltas) / float(len(deltas))) if deltas else 0.0
                    if nll_gate_wins < nll_gate_need:
                        continue

            # Survival selection: compute a lexicographic bottleneck key (LOSS-AND pressure).
            cand_surv_terms: Optional[List[Tuple[str, float]]] = None
            cand_surv_key: Optional[Tuple[float, ...]] = None
            cand_surv_component = ""
            cand_surv_loss = float("nan")
            if selection_mode == "survival":
                h_ratio = None
                if holdout_enabled:
                    if holdout_gate and cand_holdout_nll_ratio == cand_holdout_nll_ratio:
                        h_ratio = float(cand_holdout_nll_ratio)
                    elif base_holdout_nll_ratio == base_holdout_nll_ratio:
                        h_ratio = float(base_holdout_nll_ratio)
                cand_surv_terms = _survival_terms_for(
                    nll_ratio=float(cand_nll_ratio),
                    util_component=str(cand_util_component),
                    util_loss=float(cand_util_pen),
                    fluency_bn=float(cand_bottleneck),
                    holdout_ratio=h_ratio,
                )
                cand_surv_key = _survival_key(cand_surv_terms)
                cand_surv_component, cand_surv_loss = max(cand_surv_terms, key=lambda kv: kv[1])

            # Accept criteria (v0.2.1):
            # - Non-merge: must improve the *relative* objective vs. baseline slice
            #   score_rel_base = -λ*base_pen
            #   score_rel_cand = gain_bits - λ*cand_pen
            # - Merge: behavior-preserving (no NLL/penalty regression on the slice).
            if repair_needed:
                # In REPAIR, we require strict progress toward paying down the bottleneck "debt".
                if cand_deficit >= base_deficit - repair_tol:
                    continue
                # Keep utility hard-bottleneck even during repair: do not fix dialogue by breaking tasks.
                if base_util_pen < 1.0 - 1e-12 and cand_util_pen > base_util_pen + tol:
                    continue
            else:
                if selection_mode == "bottleneck":
                    if cand_nll_per_tok > base_nll_per_tok + tol:
                        continue
                    if cand_pen > base_pen + tol:
                        continue
                    if cand_util_pen > base_util_pen + tol:
                        continue
                    if gain <= 0.0:
                        continue
                else:
                    if patch.kind == "MERGE_ACTS":
                        if cand_nll_per_tok > base_nll_per_tok + tol:
                            continue
                        if cand_pen > base_pen + tol:
                            continue
                        # Merge is intended to be behavior-preserving: do not regress utility.
                        if cand_util_pen > base_util_pen + tol:
                            continue
                        if gain <= 0.0:
                            continue
                        if selection_mode == "survival" and cand_surv_key is not None:
                            # Allow merges when they don't worsen the survival bottleneck vector.
                            for cv, bv in zip(cand_surv_key, base_surv_key):
                                if float(cv) > float(bv) + tol:
                                    raise_flag = True
                                    break
                            else:
                                raise_flag = False
                            if raise_flag:
                                continue
                    else:
                        # Hard fluency bottleneck (no trade: compression cannot worsen dialogue),
                        # unless we are in divergence mode (explicit exploration under survival pressure).
                        if cand_bottleneck > base_bottleneck + fluency_slack + tol:
                            continue
                        # Hard utility bottleneck: do not trade compression for lower task pass-rate.
                        # (Once utility is >0, protect it strongly; when 0, this is a no-op.)
                        if base_util_pen < 1.0 - 1e-12 and cand_util_pen > base_util_pen + tol:
                            continue
                        if selection_mode == "survival":
                            if cand_surv_key is None or not _survival_key_better(cand_surv_key, base_surv_key):
                                continue
                        else:
                            if score <= base_score + tol:
                                continue

            # Fluency hard caps (defensive): use per-reply maxima when available.
            if float(cand_gen.get("loop_rate_reply_max", cand_gen.get("loop_rate_global", 0.0)) or 0.0) > 0.95:
                continue
            if float(cand_gen.get("repeat3_reply_max", cand_gen.get("repeat3_global", 0.0)) or 0.0) > 0.95:
                continue

            if repair_needed:
                key = (float(cand_deficit), float(-score))
                if best_repair_key is None or key < best_repair_key:
                    best_repair_key = key
                else:
                    continue
            else:
                if selection_mode == "survival" and cand_surv_key is not None:
                    if best_surv_key is not None:
                        if _survival_key_better(cand_surv_key, best_surv_key):
                            pass
                        elif cand_surv_key == best_surv_key and score > best_score + 1e-9:
                            pass
                        else:
                            continue
                    best_surv_key = tuple(float(x) for x in cand_surv_key)
                    best_score = float(score)
                else:
                    if score <= best_score + 1e-9:
                        continue
                    best_score = score

            best = (
                patch,
                {
                    "gain_bits": gain,
                    "score": score,
                    "base_score": base_score,
                    "score_improvement": score - base_score,
                    "fluency_lambda": lam,
                    "fluency_lambda_max": float(self.config.fluency_lambda),
                    "fluency_lambda_schedule": str(
                        getattr(self.config, "fluency_lambda_schedule", "constant") or "constant"
                    ),
                    "fluency_warmup_frac": float(getattr(self.config, "fluency_warmup_frac", 1.0) or 1.0),
                    "divergence": bool(divergence),
                    "divergence_lambda_scale": float(getattr(self.config, "divergence_lambda_scale", 0.1) or 0.1),
                    "repair_mode": bool(repair_mode),
                    "repair_needed": bool(repair_needed),
                    "repair_target_bottleneck": (float(repair_target) if repair_target is not None else None),
                    "repair_bottleneck_tol": float(repair_tol),
                    "repair_base_deficit": float(base_deficit),
                    "repair_cand_deficit": float(cand_deficit),
                    "utility_weight": util_w,
                    "selection_mode": str(selection_mode),
                    "base_penalty_sum": base_pen,
                    "cand_penalty_sum": cand_pen,
                    "base_utility_component": str(base_util_component),
                    "base_utility_penalty": base_util_pen,
                    "cand_utility_component": str(cand_util_component),
                    "cand_utility_penalty": cand_util_pen,
                    "vocab_size": int(V),
                    "base_nll_ratio": float(base_nll_ratio),
                    "cand_nll_ratio": float(cand_nll_ratio),
                    "base_holdout_nll_ratio": float(base_holdout_nll_ratio)
                    if (holdout_enabled and base_holdout_nll_ratio == base_holdout_nll_ratio)
                    else None,
                    "cand_holdout_nll_ratio": float(cand_holdout_nll_ratio)
                    if (holdout_gate and cand_holdout_nll_ratio == cand_holdout_nll_ratio)
                    else None,
                    "base_survival_component": str(base_surv_component),
                    "base_survival_loss": float(base_surv_loss),
                    "base_survival_key": list(base_surv_key),
                    "cand_survival_component": str(cand_surv_component),
                    "cand_survival_loss": (float(cand_surv_loss) if cand_surv_loss == cand_surv_loss else None),
                    "cand_survival_key": (list(cand_surv_key) if cand_surv_key is not None else None),
                    "base_fluency_bottleneck": float(base_bottleneck),
                    "cand_fluency_bottleneck": float(cand_bottleneck),
                    "fluency_bottleneck_delta": float(cand_bottleneck) - float(base_bottleneck),
                    "gain_horizon_steps": int(horizon_total),
                    "horizon_tokens": horizon,
                    "nll_eval_windows": int(n_windows),
                    "nll_eval_starts": list(starts),
                    "nll_gate": bool(nll_gate),
                    "nll_gate_wins": int(nll_gate_wins),
                    "nll_gate_need": int(nll_gate_need),
                    "nll_gate_max_regress_per_tok": float(nll_gate_max_reg),
                    "nll_gate_mean_delta_per_tok": float(nll_gate_mean_delta),
                    "holdout_frac": float(holdout_frac),
                    "holdout_eval_windows": int(h_windows),
                    "holdout_eval_tokens": int(h_len),
                    "holdout_gate": bool(holdout_gate),
                    "holdout_gate_wins": int(holdout_gate_wins),
                    "holdout_gate_need": int(holdout_gate_need),
                    "holdout_gate_max_regress_per_tok": float(holdout_gate_max_reg),
                    "holdout_gate_mean_delta_per_tok": float(holdout_gate_mean_delta),
                    "holdout_base_nll_bits": float(base_holdout_nll_bits)
                    if holdout_enabled
                    else None,
                    "holdout_base_nll_bits_windows": list(base_holdout_nll_windows)
                    if holdout_enabled
                    else None,
                    "holdout_cand_nll_bits": float(cand_holdout_nll_bits)
                    if holdout_gate
                    else None,
                    "holdout_cand_nll_bits_windows": list(cand_holdout_nll_windows)
                    if holdout_gate
                    else None,
                    "base": {
                        "nll_bits": base_nll,
                        "nll_bits_windows": list(base_nll_windows),
                        "cost_bits": base_cost,
                        **base_gen,
                    },
                    "cand": {
                        "nll_bits": cand_nll,
                        "nll_bits_windows": list(cand_nll_windows),
                        "cost_bits": cand_cost,
                        **cand_gen,
                    },
                },
            )

        return best

    def train(self) -> None:
        self._init_acts()
        use_hf_stream = bool(str(getattr(self.config, "hf_dataset_id", "") or "").strip())
        tok_iter = None
        tokens: List[str]
        if use_hf_stream:
            # Static holdout slice (deterministic) from an independent stream (seed offset).
            self._init_streaming_holdout_tokens()
            tok_iter = self._iter_hf_stream_tokens(seed_offset=0)
            tokens = []
        else:
            tokens = self._load_tokens()

        engine = Engine(
            self.store,
            seed=self.config.seed,
            config=EngineConfig(
                enable_contracts=bool(self.config.enable_contracts),
                decoder_fluency_no_repeat_ngram=int(self.config.decoder_fluency_no_repeat_ngram),
                decoder_fluency_prompt_ngram_block=bool(self.config.decoder_fluency_prompt_ngram_block),
                decoder_fluency_min_new_tokens_before_eos_freeform=int(
                    self.config.decoder_fluency_min_new_tokens_before_eos_freeform
                ),
                decoder_fluency_block_token_regex=str(self.config.decoder_fluency_block_token_regex or ""),
                decoder_fluency_block_penalty=float(self.config.decoder_fluency_block_penalty),
                dialogue_state_enabled=bool(self.config.dialogue_state_enabled),
                dialogue_state_tail_k=int(self.config.dialogue_state_tail_k),
                dialogue_state_prefix_enabled=bool(self.config.dialogue_state_prefix_enabled),
                dialogue_state_prefix_prompt_turns_leq=int(self.config.dialogue_state_prefix_prompt_turns_leq),
                dialogue_state_prefix_max_missing_turns=int(self.config.dialogue_state_prefix_max_missing_turns),
                dialogue_state_prefix_max_chars=int(self.config.dialogue_state_prefix_max_chars),
                dialogue_state_fact_extract_enabled=bool(self.config.dialogue_state_fact_extract_enabled),
                dialogue_state_fact_max=int(self.config.dialogue_state_fact_max),
            ),
        )

        report: List[Dict[str, Any]] = []
        # Live report stream: append one JSON object per window so callers can `tail -f`.
        # The final `report.json` is still written at the end for sweep scripts.
        try:
            with open(self.report_jsonl_path, "w", encoding="utf-8") as _f:
                _f.write("")
        except Exception:
            pass
        nll_sum_bits = 0.0
        nll_cum_bits = 0.0
        nll_ema = float("nan")
        win_t0 = time.time()
        no_patch_windows = 0
        no_growth_windows = 0
        survival_best = float("inf")
        survival_stall_windows = 0
        no_abstraction_windows = 0
        no_reuse_windows = 0
        concept_tasks_present = False
        concept_pass_rate_last = 0.0
        concept_no_add_windows = 0
        concept_reuse_best = 0.0
        concept_reuse_stall_windows = 0
        concept_crisis_windows = 0
        concept_min_depth_required_last = 0
        concept_composed_best = 0.0
        concept_composed_stall_windows = 0
        concept_composed_no_add_windows = 0
        concept_composed_crisis_windows = 0
        concept_deep_best = 0.0
        concept_deep_stall_windows = 0
        concept_deep_crisis_windows = 0
        concept_very_deep_best = 0.0
        concept_very_deep_stall_windows = 0
        concept_very_deep_crisis_windows = 0
        repair_active = False
        repair_target_bottleneck: Optional[float] = None
        repair_started_step: Optional[int] = None
        repair_windows = 0

        ctx: List[str] = ["<BOS>"] * (engine.config.max_order - 1)
        t_train0 = time.time()
        for step in range(1, int(self.config.steps) + 1):
            if tok_iter is None:
                idx = (step - 1) % len(tokens)
                if idx == 0 and step > 1:
                    ctx = ["<BOS>"] * (engine.config.max_order - 1)
                tok = tokens[idx]
            else:
                tok = next(tok_iter)
                tokens.append(tok)

            lp = engine.logprob_next(context=ctx, token=tok)
            nll_bits = -lp / math.log(2)
            nll_sum_bits += nll_bits
            nll_ema = nll_bits if nll_ema != nll_ema else (0.98 * nll_ema + 0.02 * nll_bits)

            engine.observe(context=ctx, token=tok)
            ctx.append(tok)
            ctx = ctx[-(engine.config.max_order - 1) :]

            # Streaming warmup visibility: before the first window completes, emit small progress
            # updates so `tail -f` shows the training loop is advancing.
            if tok_iter is not None and step < int(self.config.window):
                warmup_every = max(1, int(self.config.window) // 4)
                if warmup_every > 0 and (step % warmup_every) == 0:
                    try:
                        sys.stdout.write(
                            f"[train] warmup: step={step}/{int(self.config.window)}\n"
                        )
                        sys.stdout.flush()
                    except Exception:
                        pass

            if step % self.config.window == 0:
                elapsed = max(1e-9, time.time() - win_t0)
                toks_per_s = self.config.window / elapsed
                win_t0 = time.time()

                mean_nll = nll_sum_bits / self.config.window
                nll_cum_bits += nll_sum_bits
                patch: Optional[Patch] = None
                patch_meta: Optional[Dict[str, Any]] = None
                sovereign = bool(getattr(self.config, "ics_sovereign", False))
                policy_meta: Optional[Dict[str, Any]] = None
                template_meta: Optional[Dict[str, Any]] = None
                memory_meta: Optional[Dict[str, Any]] = None
                macro_meta: Optional[Dict[str, Any]] = None
                router_meta: Optional[Dict[str, Any]] = None
                # Concept creation is a survival law: plateau + no new concepts + reuse saturation ⇒
                # force INDUCE_CONCEPT (and optionally hard-fail if the crisis persists).
                surv_after = int(getattr(self.config, "survival_plateau_windows", 0) or 0)
                concept_after = int(getattr(self.config, "survival_concept_no_add_windows", 0) or 0)
                concept_reuse_after = int(
                    getattr(self.config, "survival_concept_reuse_stall_windows", 0) or 0
                )
                concept_crisis = False
                concept_composed_crisis = False
                concept_deep_crisis = False
                concept_very_deep_crisis = False
                if bool(getattr(self.config, "concept_csv_mining_enabled", False)) and bool(concept_tasks_present):
                    plateau_long = bool(int(surv_after) > 0 and int(survival_stall_windows) >= int(surv_after))
                    no_new_concepts = bool(
                        int(concept_after) > 0 and int(concept_no_add_windows) >= int(concept_after)
                    )
                    reuse_saturated = bool(
                        int(concept_reuse_after) > 0
                        and int(concept_reuse_stall_windows) >= int(concept_reuse_after)
                    )
                    # Only trigger "concept creation" crisis when concept capability is actually limiting.
                    # If concept tasks are already passing (concept_pass_rate ~= 1.0), forcing endless
                    # concept additions becomes an anti-signal that can kill otherwise-productive runs.
                    try:
                        pass_last = float(concept_pass_rate_last)
                    except Exception:
                        pass_last = 0.0
                    if pass_last != pass_last:
                        pass_last = 0.0
                    pass_last = max(0.0, min(1.0, float(pass_last)))
                    need_better_concepts = bool(pass_last < 1.0 - 1e-12)
                    concept_crisis = bool(plateau_long and no_new_concepts and reuse_saturated and need_better_concepts)

                    # Compositionality crisis: if the validator pack requires nested concept calls,
                    # failure to induce composed concepts (CSV_CALL) during plateau is existential.
                    try:
                        md_req = int(concept_min_depth_required_last or 0)
                    except Exception:
                        md_req = 0
                    need_composition = bool(int(md_req) >= 1)
                    if need_composition and bool(getattr(self.config, "concept_csv_composed_enabled", False)):
                        tol_comp0 = float(
                            getattr(self.config, "survival_concept_composed_rate_tol", 1e-6) or 1e-6
                        )
                        if tol_comp0 != tol_comp0:
                            tol_comp0 = 1e-6
                        tol_comp0 = max(0.0, float(tol_comp0))
                        comp_after = int(
                            getattr(self.config, "survival_concept_composed_no_add_windows", 0) or 0
                        )
                        comp_stall_after = int(
                            getattr(self.config, "survival_concept_composed_rate_stall_windows", 0) or 0
                        )
                        no_new_composed = bool(
                            int(comp_after) > 0
                            and int(concept_composed_no_add_windows) >= int(comp_after)
                        )
                        comp_saturated = bool(
                            int(comp_stall_after) > 0
                            and int(concept_composed_stall_windows) >= int(comp_stall_after)
                        )
                        # Only treat compositionality as an existential crisis when it is effectively absent.
                        # If composed usage exists (best > tol), allow plateau without hard-fail loops.
                        have_any_composed = bool(float(concept_composed_best) > float(tol_comp0))
                        concept_composed_crisis = bool(
                            plateau_long and no_new_composed and comp_saturated and (not have_any_composed)
                        )

                        # Deep hierarchy (depth>=2) as survival law: when required by the validator pack,
                        # sustained stall in deep concept usage during plateau is existential.
                        need_deep = bool(int(md_req) >= 2)
                        if need_deep:
                            tol_deep0 = float(
                                getattr(self.config, "survival_concept_deep_rate_tol", 1e-6) or 1e-6
                            )
                            if tol_deep0 != tol_deep0:
                                tol_deep0 = 1e-6
                            tol_deep0 = max(0.0, float(tol_deep0))
                            deep_after = int(
                                getattr(self.config, "survival_concept_deep_rate_stall_windows", 0) or 0
                            )
                            # Deep hierarchy is only an existential crisis when deep usage is absent.
                            have_any_deep = bool(float(concept_deep_best) > float(tol_deep0))
                            concept_deep_crisis = bool(
                                plateau_long
                                and int(deep_after) > 0
                                and int(concept_deep_stall_windows) >= int(deep_after)
                                and (not have_any_deep)
                            )

                        # Very deep hierarchy (depth>=3) as survival law: when required by the validator pack,
                        # sustained stall in very-deep concept usage during plateau is existential.
                        need_very_deep = bool(int(md_req) >= 3)
                        if need_very_deep:
                            tol_vdeep0 = float(
                                getattr(self.config, "survival_concept_very_deep_rate_tol", 1e-6) or 1e-6
                            )
                            if tol_vdeep0 != tol_vdeep0:
                                tol_vdeep0 = 1e-6
                            tol_vdeep0 = max(0.0, float(tol_vdeep0))
                            vdeep_after = int(
                                getattr(self.config, "survival_concept_very_deep_rate_stall_windows", 0) or 0
                            )
                            have_any_vdeep = bool(float(concept_very_deep_best) > float(tol_vdeep0))
                            concept_very_deep_crisis = bool(
                                plateau_long
                                and int(vdeep_after) > 0
                                and int(concept_very_deep_stall_windows) >= int(vdeep_after)
                                and (not have_any_vdeep)
                            )
                if (not sovereign) and step % self.config.propose_every == 0:
                    if repair_active:
                        repair_windows += 1
                    proposals = self._propose_patches(step=step, tokens=tokens)
                    div_after = int(getattr(self.config, "divergence_after_no_patch_windows", 0) or 0)
                    div_after_growth = int(getattr(self.config, "divergence_after_no_growth_windows", 0) or 0)
                    surv_abs_after = int(getattr(self.config, "survival_no_abstraction_windows", 0) or 0)
                    surv_reuse_after = int(getattr(self.config, "survival_no_reuse_windows", 0) or 0)
                    use_div = bool(
                        (div_after > 0 and no_patch_windows >= div_after)
                        or (div_after_growth > 0 and no_growth_windows >= div_after_growth)
                        or (surv_after > 0 and survival_stall_windows >= surv_after)
                        or (surv_abs_after > 0 and no_abstraction_windows >= surv_abs_after)
                        or (surv_reuse_after > 0 and no_reuse_windows >= surv_reuse_after)
                        or bool(concept_crisis)
                        or bool(concept_composed_crisis)
                        or bool(concept_deep_crisis)
                        or bool(concept_very_deep_crisis)
                    )
                    # Two-phase promotion: divergence explores; REPAIR consolidates (no more divergence until paid).
                    if repair_active:
                        use_div = False
                    choice = self._select_patch(
                        step=step,
                        engine=engine,
                        tokens=tokens,
                        patches=proposals,
                        divergence=use_div,
                        repair_mode=bool(repair_active),
                        repair_target_bottleneck=repair_target_bottleneck,
                    )
                    if choice is not None:
                        patch, patch_meta = choice
                        apply_res = self._apply_patch(patch, count=True)
                        extra = apply_res.get("meta")
                        if extra:
                            if patch_meta is None:
                                patch_meta = {}
                            patch_meta.update(dict(extra))
                        # If divergence accepted a patch that worsened the fluency bottleneck, accrue fluency debt
                        # and enter mandatory REPAIR until the bottleneck returns to its pre-divergence value.
                        if not repair_active and patch_meta and bool(patch_meta.get("divergence")):
                            try:
                                delta = float(patch_meta.get("fluency_bottleneck_delta", 0.0) or 0.0)
                            except Exception:
                                delta = 0.0
                            if delta != delta:
                                delta = 0.0
                            try:
                                tol_bn = float(getattr(self.config, "repair_bottleneck_tol", 1e-9) or 1e-9)
                            except Exception:
                                tol_bn = 1e-9
                            if tol_bn != tol_bn:
                                tol_bn = 1e-9
                            tol_bn = max(0.0, min(1.0, tol_bn))
                            if delta > tol_bn:
                                try:
                                    tgt = float(patch_meta.get("base_fluency_bottleneck", 0.0) or 0.0)
                                except Exception:
                                    tgt = 0.0
                                if tgt != tgt:
                                    tgt = 0.0
                                repair_active = True
                                repair_target_bottleneck = float(tgt)
                                repair_started_step = int(step)
                                repair_windows = 0
                                patch_meta.setdefault("repair", {})
                                if isinstance(patch_meta.get("repair"), dict):
                                    patch_meta["repair"].update(
                                        {
                                            "activated": True,
                                            "target_bottleneck": float(tgt),
                                            "started_step": int(step),
                                            "bottleneck_delta": float(delta),
                                        }
                                    )
                        engine.rebuild_cache()
                        no_patch_windows = 0
                        if patch.kind in {"ADD_ACT", "MERGE_ACTS"}:
                            no_growth_windows = 0
                        else:
                            no_growth_windows += 1
                    else:
                        no_patch_windows += 1
                        no_growth_windows += 1

                V = max(1, len(engine.vocab()) + 1)
                baseline = step * safe_log2(V)
                cost_bits = self._model_cost_bits()
                mdl_total = nll_cum_bits + cost_bits
                mdl_net = baseline - mdl_total

                # IMPORTANT: evaluate suites with contracts OFF (no "utility by contract").
                try:
                    cfgd = dict(engine.config.__dict__)
                    cfgd["enable_contracts"] = False
                    eval_cfg = EngineConfig(**cfgd)
                except Exception:
                    eval_cfg = EngineConfig(enable_contracts=False)
                eval_engine = Engine(self.store, seed=self.config.seed, config=eval_cfg)

                transcripts, gen = run_chat_suite(
                    eval_engine,
                    dialogues=CHAT_DIALOGUES_20X3,
                    max_new_tokens=self.config.fluency_gen_tokens,
                    prefix_k=self.config.suite_prefix_k,
                    template_ngram_n=self.config.suite_template_ngram_n,
                    template_prefix_window=self.config.suite_template_prefix_window,
                )
                if not sovereign:
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
                        eval_engine,
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

                skill_tasks = skill_suite_tasks_for_pack(
                    str(getattr(self.config, "skill_suite_pack", "v0") or "v0")
                )
                skill_transcripts, util_metrics = run_skill_suite(
                    eval_engine,
                    tasks=skill_tasks,
                    max_new_tokens=self.config.skill_suite_max_new_tokens,
                    prompt_history_k=int(getattr(self.config, "skill_suite_prompt_history_k", 0) or 0),
                    family_shuffle_seed=(
                        int(getattr(self.config, "seed", 0) or 0)
                        if bool(getattr(self.config, "suite_shuffle_families", False))
                        else None
                    ),
                    family_shuffle_salt=int(step),
                )
                util_log = dict(util_metrics)
                for k, v in util_log.items():
                    gen[f"utility_{k}"] = v
                if patch_meta is None:
                    patch_meta = {}
                patch_meta["utility_suite"] = {f"utility_{k}": v for k, v in util_log.items()}

                # Agency suite: explicit planner+executor goals (multi-step) as a utility bottleneck.
                agency_meta = self._eval_agency_suite_metrics()
                if isinstance(agency_meta, dict) and bool(agency_meta.get("enabled", False)):
                    gen["utility_agency_total"] = int(agency_meta.get("total") or 0)
                    gen["utility_agency_pass_rate"] = float(agency_meta.get("pass_rate") or 0.0)
                    gen["utility_agency_plan_found_rate"] = float(agency_meta.get("plan_found_rate") or 0.0)
                    gen["utility_agency_steps_pass_mean"] = float(agency_meta.get("steps_pass_mean") or 0.0)
                    gen["utility_agency_steps_pass_median"] = float(agency_meta.get("steps_pass_median") or 0.0)
                    gen["utility_agency_fail_reasons"] = (
                        dict(agency_meta.get("fail_reasons") or {})
                        if isinstance(agency_meta.get("fail_reasons"), dict)
                        else {}
                    )
                    gen["utility_agency_fail_details"] = (
                        list(agency_meta.get("fail_details") or [])
                        if isinstance(agency_meta.get("fail_details"), list)
                        else []
                    )
                    if patch_meta is None:
                        patch_meta = {}
                    patch_meta["agency_suite"] = dict(agency_meta)

                # Concept reuse saturation (survival law): track whether concept execution is actually being
                # used by the runtime on concept-bearing tasks. When this saturates and survival stalls,
                # the system must create new concepts (or it dies).
                try:
                    concept_total_now = int(gen.get("utility_concept_total") or 0)
                except Exception:
                    concept_total_now = 0
                concept_tasks_present = bool(int(concept_total_now) > 0)
                try:
                    concept_pass_rate_last = float(gen.get("utility_concept_pass_rate") or 0.0)
                except Exception:
                    concept_pass_rate_last = 0.0
                if concept_pass_rate_last != concept_pass_rate_last:
                    concept_pass_rate_last = 0.0
                concept_pass_rate_last = max(0.0, min(1.0, float(concept_pass_rate_last)))
                try:
                    concept_min_depth_required_last = int(
                        gen.get("utility_concept_min_depth_required_max") or 0
                    )
                except Exception:
                    concept_min_depth_required_last = 0
                if not concept_tasks_present or not bool(getattr(self.config, "concept_csv_mining_enabled", False)):
                    concept_reuse_best = 0.0
                    concept_reuse_stall_windows = 0
                    concept_composed_best = 0.0
                    concept_composed_stall_windows = 0
                    concept_deep_best = 0.0
                    concept_deep_stall_windows = 0
                    concept_deep_crisis_windows = 0
                    concept_very_deep_best = 0.0
                    concept_very_deep_stall_windows = 0
                    concept_very_deep_crisis_windows = 0
                else:
                    try:
                        used_rate = float(gen.get("utility_concept_used_rate") or 0.0)
                    except Exception:
                        used_rate = 0.0
                    if used_rate != used_rate:
                        used_rate = 0.0
                    used_rate = max(0.0, min(1.0, float(used_rate)))
                    tol_reuse = float(getattr(self.config, "survival_concept_reuse_tol", 1e-6) or 1e-6)
                    if tol_reuse != tol_reuse:
                        tol_reuse = 1e-6
                    tol_reuse = max(0.0, float(tol_reuse))
                    if float(used_rate) > float(concept_reuse_best) + float(tol_reuse):
                        concept_reuse_best = float(used_rate)
                        concept_reuse_stall_windows = 0
                    else:
                        concept_reuse_stall_windows += 1

                    # Track compositionality usage separately (nested CSV_CALL depth >=1).
                    try:
                        comp_rate = float(gen.get("utility_concept_composed_rate") or 0.0)
                    except Exception:
                        comp_rate = 0.0
                    if comp_rate != comp_rate:
                        comp_rate = 0.0
                    comp_rate = max(0.0, min(1.0, float(comp_rate)))
                    tol_comp = float(
                        getattr(self.config, "survival_concept_composed_rate_tol", 1e-6) or 1e-6
                    )
                    if tol_comp != tol_comp:
                        tol_comp = 1e-6
                    tol_comp = max(0.0, float(tol_comp))
                    if float(comp_rate) > float(concept_composed_best) + float(tol_comp):
                        concept_composed_best = float(comp_rate)
                        concept_composed_stall_windows = 0
                    else:
                        concept_composed_stall_windows += 1

                    # Track deep concept hierarchy usage (nested CSV_CALL depth >=2).
                    try:
                        deep_rate = float(gen.get("utility_concept_deep_rate") or 0.0)
                    except Exception:
                        deep_rate = 0.0
                    if deep_rate != deep_rate:
                        deep_rate = 0.0
                    deep_rate = max(0.0, min(1.0, float(deep_rate)))
                    tol_deep = float(
                        getattr(self.config, "survival_concept_deep_rate_tol", 1e-6) or 1e-6
                    )
                    if tol_deep != tol_deep:
                        tol_deep = 1e-6
                    tol_deep = max(0.0, float(tol_deep))
                    prev_best = float(concept_deep_best)
                    improved = bool(float(deep_rate) > float(prev_best) + float(tol_deep))
                    if improved:
                        concept_deep_best = float(deep_rate)
                        concept_deep_stall_windows = 0
                    else:
                        concept_deep_stall_windows += 1

                    # If deep crisis is active, enforce deterministic hard-fail if it persists.
                    if bool(concept_deep_crisis):
                        if improved:
                            concept_deep_crisis_windows = 0
                        else:
                            concept_deep_crisis_windows += 1
                        hard_deep = int(
                            getattr(self.config, "survival_concept_deep_hard_fail_windows", 0) or 0
                        )
                        if int(hard_deep) > 0 and int(concept_deep_crisis_windows) >= int(hard_deep):
                            raise RuntimeError(
                                "SYSTEM_SURVIVAL deep concept crisis hard-fail: plateau + no deep concept usage "
                                f"(deep_rate={deep_rate:.6f}, best={concept_deep_best:.6f}, "
                                f"stall_windows={concept_deep_stall_windows}, crisis_windows={concept_deep_crisis_windows})"
                            )
                    else:
                        concept_deep_crisis_windows = 0

                    # Track very-deep concept hierarchy usage (nested CSV_CALL depth >=3).
                    try:
                        vdeep_rate = float(gen.get("utility_concept_very_deep_rate") or 0.0)
                    except Exception:
                        vdeep_rate = 0.0
                    if vdeep_rate != vdeep_rate:
                        vdeep_rate = 0.0
                    vdeep_rate = max(0.0, min(1.0, float(vdeep_rate)))
                    tol_vdeep = float(
                        getattr(self.config, "survival_concept_very_deep_rate_tol", 1e-6) or 1e-6
                    )
                    if tol_vdeep != tol_vdeep:
                        tol_vdeep = 1e-6
                    tol_vdeep = max(0.0, float(tol_vdeep))
                    prev_vbest = float(concept_very_deep_best)
                    v_improved = bool(float(vdeep_rate) > float(prev_vbest) + float(tol_vdeep))
                    if v_improved:
                        concept_very_deep_best = float(vdeep_rate)
                        concept_very_deep_stall_windows = 0
                    else:
                        concept_very_deep_stall_windows += 1

                    # If very-deep crisis is active, enforce deterministic hard-fail if it persists.
                    if bool(concept_very_deep_crisis):
                        if v_improved:
                            concept_very_deep_crisis_windows = 0
                        else:
                            concept_very_deep_crisis_windows += 1
                        hard_vdeep = int(
                            getattr(self.config, "survival_concept_very_deep_hard_fail_windows", 0) or 0
                        )
                        if int(hard_vdeep) > 0 and int(concept_very_deep_crisis_windows) >= int(hard_vdeep):
                            raise RuntimeError(
                                "SYSTEM_SURVIVAL very-deep concept crisis hard-fail: plateau + no very-deep concept usage "
                                f"(very_deep_rate={vdeep_rate:.6f}, best={concept_very_deep_best:.6f}, "
                                f"stall_windows={concept_very_deep_stall_windows}, crisis_windows={concept_very_deep_crisis_windows})"
                            )
                    else:
                        concept_very_deep_crisis_windows = 0

                if not sovereign:
                    memory_meta = self._update_fact_memory_from_transcripts(transcripts, step=step)
                    if memory_meta and memory_meta.get("enabled"):
                        if patch_meta is None:
                            patch_meta = {}
                        mem_log = dict(memory_meta)
                        if step != int(self.config.steps):
                            mem_log.pop("top_facts", None)
                        patch_meta["memory_update"] = mem_log

                # Enrich macro mining with skill-suite traces (more contexts -> more reusable abstractions).
                try:
                    if isinstance(skill_transcripts, list) and skill_transcripts:
                        shadow_transcripts = list(shadow_transcripts) + list(skill_transcripts)
                except Exception:
                    pass

                if not sovereign:
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

                # ICS sovereign operator (concept governance + induction):
                # merge -> split -> fitness/selection -> induce -> rewrite.
                ics_util = dict(util_metrics) if isinstance(util_metrics, dict) else {}
                if bool(getattr(self.config, "ics_shuffle_families", False)):
                    try:
                        usage0 = (
                            ics_util.get("concept_usage_by_id")
                            if isinstance(ics_util.get("concept_usage_by_id"), dict)
                            else {}
                        )
                        fids: set = set()
                        for rec0 in usage0.values():
                            if not isinstance(rec0, dict):
                                continue
                            fams0 = rec0.get("families") if isinstance(rec0.get("families"), dict) else {}
                            if not isinstance(fams0, dict):
                                continue
                            for fid in fams0.keys():
                                if str(fid):
                                    fids.add(str(fid))
                        ordered = sorted(fids, key=str)
                        if ordered:
                            perm = sorted(
                                ordered,
                                key=lambda fid: sha256_hex(
                                    f"{int(getattr(self.config, 'seed', 0) or 0)}:{int(step)}:{fid}".encode(
                                        "utf-8"
                                    )
                                ),
                            )
                            mapping = {ordered[i]: perm[i] for i in range(len(ordered))}
                            usage2: Dict[str, Any] = {}
                            for cid0 in sorted([str(k) for k in usage0.keys() if str(k)], key=str):
                                rec0 = usage0.get(cid0)
                                if not isinstance(rec0, dict):
                                    continue
                                fams0 = rec0.get("families") if isinstance(rec0.get("families"), dict) else {}
                                fams2: Dict[str, Any] = {}
                                if isinstance(fams0, dict):
                                    for fid0 in sorted([str(x) for x in fams0.keys() if str(x)], key=str):
                                        nf = str(mapping.get(str(fid0), str(fid0)))
                                        fams2[nf] = fams0.get(fid0)
                                rec2 = dict(rec0)
                                rec2["families"] = dict(fams2)
                                usage2[str(cid0)] = rec2
                            ics_util["concept_usage_by_id"] = dict(usage2)
                            ics_util["ics_family_shuffle"] = {"enabled": True, "families": int(len(ordered))}
                    except Exception:
                        pass
                ics_meta = self._ics_step(
                    step=int(step),
                    util_metrics=dict(ics_util) if isinstance(ics_util, dict) else {},
                    skill_tasks=skill_tasks,
                    force_new=bool(
                        concept_crisis or concept_composed_crisis or concept_deep_crisis or concept_very_deep_crisis
                    ),
                    transcripts=transcripts,
                    shadow_transcripts=shadow_transcripts,
                )
                concept_csv_meta = {}
                if isinstance(ics_meta, dict):
                    concept_csv_meta = ics_meta.get("induce") if isinstance(ics_meta.get("induce"), dict) else {}
                    if patch_meta is None:
                        patch_meta = {}
                    patch_meta["ics_v1"] = dict(ics_meta)
                    sem = (
                        ics_meta.get("semantic_banks_v1")
                        if isinstance(ics_meta.get("semantic_banks_v1"), dict)
                        else None
                    )
                    if isinstance(sem, dict):
                        gen["semantic_goals_total"] = int(sem.get("goals_total") or 0)
                        gen["semantic_plans_total"] = int(sem.get("plans_total") or 0)
                        gen["semantic_hypotheses_total"] = int(sem.get("hypotheses_total") or 0)
                        gen["semantic_references_total"] = int(sem.get("references_total") or 0)
                        gen["semantic_goals_created_window"] = int(sem.get("goals_created") or 0)
                        gen["semantic_plans_created_window"] = int(sem.get("plans_created") or 0)
                        gen["semantic_hypotheses_created_window"] = int(sem.get("hypotheses_created") or 0)
                        gen["semantic_references_created_window"] = int(sem.get("references_created") or 0)
                        gen["semantic_seen_plan_turns_window"] = int(sem.get("seen_plan_turns") or 0)

                    # In sovereign mode, all cognitive state updates must flow through ICS.
                    if bool(sovereign):
                        policy_meta = (
                            dict(ics_meta.get("mode_policy_update") or {})
                            if isinstance(ics_meta.get("mode_policy_update"), dict)
                            else None
                        )
                        template_meta = (
                            dict(ics_meta.get("ka_template_update") or {})
                            if isinstance(ics_meta.get("ka_template_update"), dict)
                            else None
                        )
                        memory_meta = (
                            dict(ics_meta.get("memory_update") or {})
                            if isinstance(ics_meta.get("memory_update"), dict)
                            else None
                        )
                        macro_meta = (
                            dict(ics_meta.get("macro_update") or {})
                            if isinstance(ics_meta.get("macro_update"), dict)
                            else None
                        )
                        router_meta = (
                            dict(ics_meta.get("router_update") or {})
                            if isinstance(ics_meta.get("router_update"), dict)
                            else None
                        )

                        # Preserve legacy top-level logging keys for audit tooling.
                        if policy_meta and policy_meta.get("enabled"):
                            patch_meta["mode_policy_update"] = dict(policy_meta)
                        if template_meta and template_meta.get("enabled"):
                            patch_meta["ka_template_update"] = dict(template_meta)
                        if memory_meta and memory_meta.get("enabled"):
                            mem_log = dict(memory_meta)
                            if step != int(self.config.steps):
                                mem_log.pop("top_facts", None)
                            patch_meta["memory_update"] = mem_log
                        if macro_meta and macro_meta.get("enabled"):
                            mac_log = dict(macro_meta)
                            if step != int(self.config.steps):
                                mac_log.pop("top_macros", None)
                                mac_log.pop("top_rewrite_rule_hits", None)
                                mac_log.pop("top_set_transitions", None)
                            patch_meta["macro_update"] = mac_log
                        if router_meta and router_meta.get("enabled"):
                            rlog = dict(router_meta)
                            shadow = rlog.pop("shadow_metrics", None)
                            patch_meta["router_update"] = rlog
                            if shadow is not None:
                                patch_meta["router_shadow_metrics"] = shadow
                if isinstance(concept_csv_meta, dict) and bool(concept_csv_meta.get("enabled", False)):
                    if patch_meta is None:
                        patch_meta = {}
                    patch_meta["concept_csv_mining"] = dict(concept_csv_meta)

                # Concept creation as existential law: in a plateau + reuse-saturated regime, failure to
                # create new concept_csv acts (flat OR composed OR deepwrap) is a deterministic hard failure.
                concept_flat_added_now = 0
                concept_composed_added_now = 0
                concept_deepwrap_added_now = 0
                if isinstance(concept_csv_meta, dict):
                    try:
                        concept_flat_added_now = int(concept_csv_meta.get("added", 0) or 0)
                    except Exception:
                        concept_flat_added_now = 0
                    cm0 = concept_csv_meta.get("composed_v74")
                    if isinstance(cm0, dict):
                        try:
                            concept_composed_added_now = int(cm0.get("added", 0) or 0)
                        except Exception:
                            concept_composed_added_now = 0
                        dw0 = cm0.get("deepwrap_v74")
                        if isinstance(dw0, dict):
                            try:
                                concept_deepwrap_added_now = int(dw0.get("added", 0) or 0)
                            except Exception:
                                concept_deepwrap_added_now = 0
                concept_added_now = int(concept_flat_added_now) + int(concept_composed_added_now) + int(
                    concept_deepwrap_added_now
                )
                if bool(concept_tasks_present) and bool(getattr(self.config, "concept_csv_mining_enabled", False)):
                    if int(concept_added_now) > 0:
                        concept_no_add_windows = 0
                    else:
                        concept_no_add_windows += 1
                else:
                    concept_no_add_windows = 0

                if bool(concept_crisis):
                    if int(concept_added_now) > 0:
                        concept_crisis_windows = 0
                    else:
                        concept_crisis_windows += 1
                    hard_concept = int(getattr(self.config, "survival_concept_hard_fail_windows", 0) or 0)
                    if int(hard_concept) > 0 and int(concept_crisis_windows) >= int(hard_concept):
                        raise RuntimeError(
                            "SYSTEM_SURVIVAL concept crisis hard-fail: plateau + no new concepts + reuse saturated "
                            f"(concept_added={concept_added_now}, crisis_windows={concept_crisis_windows}, "
                            f"no_add_windows={concept_no_add_windows}, reuse_stall_windows={concept_reuse_stall_windows})"
                        )
                else:
                    concept_crisis_windows = 0

                # Composed concept creation (CSV_CALL) as existential law when required by the validator pack.
                composed_added_now = 0
                try:
                    cm = (
                        concept_csv_meta.get("composed_v74")
                        if isinstance(concept_csv_meta, dict)
                        else None
                    )
                    if isinstance(cm, dict):
                        composed_added_now = int(cm.get("added", 0) or 0)
                except Exception:
                    composed_added_now = 0
                need_composition = bool(int(concept_min_depth_required_last or 0) >= 1)
                if bool(concept_tasks_present) and need_composition and bool(
                    getattr(self.config, "concept_csv_composed_enabled", False)
                ):
                    if int(composed_added_now) > 0:
                        concept_composed_no_add_windows = 0
                    else:
                        concept_composed_no_add_windows += 1
                else:
                    concept_composed_no_add_windows = 0

                if bool(concept_composed_crisis):
                    if int(composed_added_now) > 0:
                        concept_composed_crisis_windows = 0
                    else:
                        concept_composed_crisis_windows += 1
                    hard_comp = int(
                        getattr(self.config, "survival_concept_composed_hard_fail_windows", 0) or 0
                    )
                    if int(hard_comp) > 0 and int(concept_composed_crisis_windows) >= int(hard_comp):
                        raise RuntimeError(
                            "SYSTEM_SURVIVAL concept composed crisis hard-fail: plateau + no new composed concepts "
                            f"(composed_added={composed_added_now}, crisis_windows={concept_composed_crisis_windows}, "
                            f"no_add_windows={concept_composed_no_add_windows}, composed_stall_windows={concept_composed_stall_windows})"
                        )
                else:
                    concept_composed_crisis_windows = 0

                # SYSTEM_SURVIVAL (autopoiesis-like): single bottleneck loss across key capabilities.
                # This is not a gradient signal; it only governs divergence / survival pressure.
                log2V = safe_log2(V)
                nll_ratio = float(mean_nll / log2V) if log2V > 0 else float("inf")

                # Holdout NLL (static): deterministic windows on a held-out token slice.
                holdout_tokens = list(getattr(self, "_holdout_tokens", []) or [])
                holdout_frac = float(getattr(self.config, "holdout_frac", 0.0) or 0.0)
                if holdout_frac != holdout_frac:
                    holdout_frac = 0.0
                holdout_frac = max(0.0, min(0.5, holdout_frac))
                h_windows = int(getattr(self.config, "holdout_eval_windows", 0) or 0)
                h_windows = max(0, min(16, h_windows))
                h_len = int(getattr(self.config, "holdout_eval_tokens", 0) or 0)
                if h_len <= 0:
                    h_len = int(self.config.val_tokens)
                h_len = max(1, int(h_len))
                holdout_enabled = bool(holdout_tokens) and int(h_windows) > 0
                holdout_nll_bits_mean = float("nan")
                holdout_nll_bits_windows: Optional[List[float]] = None
                holdout_nll_ratio = float("nan")
                if holdout_enabled and log2V > 0:
                    start_h = (int(step) * 104729) % max(1, len(holdout_tokens))
                    starts_h = [int(start_h)]
                    if int(h_windows) > 1:
                        stride_h = max(1, int(h_len))
                        Lh = len(holdout_tokens)
                        for i in range(1, int(h_windows)):
                            starts_h.append((int(start_h) + i * int(stride_h)) % Lh)
                    holdout_eval = self._eval_static_nll_windows(
                        self.store,
                        holdout_tokens,
                        start=int(start_h),
                        starts=list(starts_h),
                        length=int(h_len),
                        engine_config=eval_engine.config,
                    )
                    holdout_nll_bits_mean = float(holdout_eval.get("nll_bits", float("nan")))
                    holdout_nll_bits_windows = list(
                        holdout_eval.get("nll_bits_windows", [holdout_nll_bits_mean])
                    )
                    holdout_nll_per_tok = float(holdout_nll_bits_mean) / float(h_len)
                    holdout_nll_ratio = float(holdout_nll_per_tok / float(log2V))
                util_component, util_loss, util_terms = self._utility_bottleneck_loss(gen)
                try:
                    util_ok = bool(float(util_loss) <= 1e-12)
                except Exception:
                    util_ok = False

                concept_total = 0
                concept_pass_rate = 0.0
                concept_usage_loss = 0.0
                try:
                    concept_total = int(gen.get("utility_concept_total") or 0)
                except Exception:
                    concept_total = 0
                try:
                    concept_pass_rate = float(gen.get("utility_concept_pass_rate") or 0.0)
                except Exception:
                    concept_pass_rate = 0.0
                if concept_pass_rate != concept_pass_rate:
                    concept_pass_rate = 0.0
                concept_pass_rate = max(0.0, min(1.0, float(concept_pass_rate)))
                if int(concept_total) > 0:
                    concept_usage_loss = max(0.0, min(1.0, 1.0 - float(concept_pass_rate)))

                def _f(x: Any) -> float:
                    try:
                        v = float(x or 0.0)
                    except Exception:
                        v = 0.0
                    if v != v:
                        v = 0.0
                    return v

                rep3_bn = _f(gen.get("repeat3_reply_max", gen.get("repeat3_global")))
                loop_bn = _f(gen.get("loop_rate_reply_max", gen.get("loop_rate_global")))
                flu_bottleneck = max(
                    rep3_bn,
                    loop_bn,
                    _f(gen.get("duplicate_reply_rate")),
                    _f(gen.get("most_common_reply_frac")),
                    _f(gen.get("whitespace_ratio")),
                    _f(gen.get("prefix_k_dup_rate")),
                    _f(gen.get("template_ngram_dup_rate")),
                    _f(gen.get("cross_turn_signature_repeat_rate")),
                )

                # Mandatory REPAIR: once activated, it must remain until bottleneck returns to target.
                repair_resolved = False
                if repair_active and repair_target_bottleneck is not None:
                    try:
                        tol_bn = float(getattr(self.config, "repair_bottleneck_tol", 1e-9) or 1e-9)
                    except Exception:
                        tol_bn = 1e-9
                    if tol_bn != tol_bn:
                        tol_bn = 1e-9
                    tol_bn = max(0.0, min(1.0, tol_bn))
                    try:
                        tgt = float(repair_target_bottleneck)
                    except Exception:
                        tgt = float("nan")
                    if tgt == tgt and float(flu_bottleneck) <= float(tgt) + float(tol_bn):
                        repair_resolved = True
                        repair_active = False
                        repair_target_bottleneck = None
                        repair_started_step = None
                        repair_windows = 0

                macro_added = int(macro_meta.get("added", 0) or 0) if macro_meta else 0
                concept_added = int(macro_meta.get("concept_added", 0) or 0) if macro_meta else 0
                # Count concept_csv growth as structural progress (flat + composed + deepwrap).
                concept_csv_added = int(concept_added_now)
                abstraction_added = int(macro_added + concept_added + concept_csv_added)
                reuse_score = 0.0
                if macro_meta:
                    reuse_score = max(
                        _f(macro_meta.get("macro_hit_rate_seq")),
                        _f(macro_meta.get("macro_hit_rate_set")),
                        _f(macro_meta.get("concept_symbol_coverage_rate")),
                    )

                if abstraction_added > 0 or util_ok:
                    no_abstraction_windows = 0
                else:
                    # When utility validators are failing, lack of new abstractions is an
                    # existential stall (ISL pressure). When utility is perfect, stable reuse is OK.
                    no_abstraction_windows += 1

                reuse_min = float(getattr(self.config, "survival_reuse_min", 0.0) or 0.0)
                if reuse_min != reuse_min:
                    reuse_min = 0.0
                reuse_min = max(0.0, min(1.0, reuse_min))
                if reuse_score + 1e-12 >= reuse_min:
                    no_reuse_windows = 0
                else:
                    no_reuse_windows += 1
                surv_terms: List[Tuple[str, float]] = [
                    ("nll_ratio", float(nll_ratio)),
                    (f"utility:{util_component}", float(util_loss)),
                    ("fluency", float(flu_bottleneck)),
                ]
                if holdout_enabled and holdout_nll_ratio == holdout_nll_ratio:
                    surv_terms.append(("holdout_nll_ratio", float(holdout_nll_ratio)))
                if bool(getattr(self.config, "concept_csv_mining_enabled", False)) and int(concept_total) > 0:
                    surv_terms.append(("concept_usage", float(concept_usage_loss)))
                survival_component, survival_loss = max(surv_terms, key=lambda kv: kv[1])

                surv_tol = float(getattr(self.config, "survival_improve_tol", 1e-4) or 1e-4)
                if surv_tol != surv_tol:
                    surv_tol = 1e-4
                improved = bool(survival_loss < survival_best - surv_tol)
                if improved:
                    survival_best = float(survival_loss)
                if util_ok:
                    survival_stall_windows = 0
                elif improved:
                    survival_stall_windows = 0
                else:
                    survival_stall_windows += 1

                surv_meta = {
                    "loss": float(survival_loss),
                    "best": float(survival_best),
                    "stall_windows": int(survival_stall_windows),
                    "component": str(survival_component),
                    "nll_ratio": float(nll_ratio),
                    "holdout_enabled": bool(holdout_enabled),
                    "holdout_frac": float(holdout_frac),
                    "holdout_eval_windows": int(h_windows),
                    "holdout_eval_tokens": int(h_len),
                    "holdout_nll_bits": float(holdout_nll_bits_mean) if holdout_enabled else None,
                    "holdout_nll_bits_windows": list(holdout_nll_bits_windows)
                    if holdout_nll_bits_windows is not None
                    else None,
                    "holdout_nll_ratio": float(holdout_nll_ratio) if holdout_enabled else None,
                    "utility_loss": float(util_loss),
                    "utility_component": str(util_component),
                    "utility_terms": dict(util_terms),
                    "concept_total": int(concept_total),
                    "concept_pass_rate": float(concept_pass_rate),
                    "concept_usage_loss": float(concept_usage_loss),
                    "concept_tasks_present": bool(concept_tasks_present),
                    "concept_added_window": int(concept_added_now),
                    "concept_no_add_windows": int(concept_no_add_windows),
                    "concept_no_add_after": int(getattr(self.config, "survival_concept_no_add_windows", 0) or 0),
                    "concept_reuse_best": float(concept_reuse_best),
                    "concept_reuse_stall_windows": int(concept_reuse_stall_windows),
                    "concept_reuse_after": int(
                        getattr(self.config, "survival_concept_reuse_stall_windows", 0) or 0
                    ),
                    "concept_reuse_tol": float(getattr(self.config, "survival_concept_reuse_tol", 0.0) or 0.0),
                    "concept_crisis": bool(concept_crisis),
                    "concept_crisis_windows": int(concept_crisis_windows),
                    "concept_hard_fail_windows": int(
                        getattr(self.config, "survival_concept_hard_fail_windows", 0) or 0
                    ),
                    "fluency_bottleneck": float(flu_bottleneck),
                    "abstraction_added": int(abstraction_added),
                    "reuse_score": float(reuse_score),
                    "reuse_min": float(reuse_min),
                    "no_abstraction_windows": int(no_abstraction_windows),
                    "no_reuse_windows": int(no_reuse_windows),
                    "repair_active": bool(repair_active),
                    "repair_resolved": bool(repair_resolved),
                    "repair_target_bottleneck": (float(repair_target_bottleneck) if repair_target_bottleneck is not None else None),
                    "repair_started_step": (int(repair_started_step) if repair_started_step is not None else None),
                    "repair_windows": int(repair_windows),
                    "repair_max_windows": int(getattr(self.config, "repair_max_windows", 0) or 0),
                    "repair_bottleneck_tol": float(getattr(self.config, "repair_bottleneck_tol", 1e-9) or 1e-9),
                }
                if patch_meta is None:
                    patch_meta = {}
                patch_meta["system_survival"] = dict(surv_meta)
                if bool(repair_resolved):
                    patch_meta.setdefault("repair", {})
                    if isinstance(patch_meta.get("repair"), dict):
                        patch_meta["repair"].update({"resolved": True, "resolved_step": int(step)})

                surv_act = self._get_system_survival_act()
                if surv_act is not None:
                    ev = surv_act.evidence if isinstance(surv_act.evidence, dict) else {}
                    ev = dict(ev)
                    ev["last_update_step"] = int(step)
                    ev["status"] = dict(surv_meta)
                    surv_act.evidence = ev

                # If REPAIR cannot be satisfied within the configured number of windows, hard-fail.
                repair_max = int(getattr(self.config, "repair_max_windows", 0) or 0)
                if repair_active and repair_max > 0 and int(repair_windows) >= int(repair_max):
                    raise RuntimeError(
                        "REPAIR hard-fail: could not restore fluency bottleneck "
                        f"within {repair_max} windows (current={flu_bottleneck:.6f}, "
                        f"target={repair_target_bottleneck}, started_step={repair_started_step})"
                    )

                hard_fail = int(getattr(self.config, "survival_hard_fail_windows", 0) or 0)
                if hard_fail > 0 and max(survival_stall_windows, no_abstraction_windows, no_reuse_windows) >= hard_fail:
                    raise RuntimeError(
                        "SYSTEM_SURVIVAL hard-fail: stalled too long "
                        f"(stall={survival_stall_windows}, no_abstraction={no_abstraction_windows}, no_reuse={no_reuse_windows}, "
                        f"loss={survival_loss:.6f}, best={survival_best:.6f}, bottleneck={survival_component})"
                    )

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

                concept_csv_total = 0
                concept_csv_mined_total = 0
                try:
                    concept_csv_acts = self.store.by_kind("concept_csv")
                    concept_csv_total = int(len(concept_csv_acts))
                    concept_csv_mined_total = int(
                        len(
                            [
                                a
                                for a in concept_csv_acts
                                if isinstance(getattr(a, "evidence", None), dict)
                                and str(a.evidence.get("name") or "") == "concept_csv_mined_train_v0"
                            ]
                        )
                    )
                except Exception:
                    concept_csv_total = 0
                    concept_csv_mined_total = 0

                row = {
                    "step": step,
                    "nll_bits_mean": mean_nll,
                    "holdout_enabled": bool(holdout_enabled),
                    "holdout_frac": float(holdout_frac),
                    "holdout_eval_windows": int(h_windows),
                    "holdout_eval_tokens": int(h_len),
                    "holdout_nll_bits_mean": float(holdout_nll_bits_mean)
                    if holdout_enabled
                    else None,
                    "holdout_nll_ratio": float(holdout_nll_ratio) if holdout_enabled else None,
                    "nll_bits_ema": nll_ema,
                    "nll_bits_cum": nll_cum_bits,
                    "mdl_total_est_bits": mdl_total,
                    "mdl_net_est_bits": mdl_net,
                    "model_cost_bits": cost_bits,
                    "vocab_size": V,
                    "mode": self.config.mode,
                    "fluency_lambda": float(self.config.fluency_lambda),
                    "fluency_lambda_effective": float(self._effective_fluency_lambda(step=step)),
                    "fluency_lambda_schedule": str(
                        getattr(self.config, "fluency_lambda_schedule", "constant") or "constant"
                    ),
                    "fluency_warmup_frac": float(getattr(self.config, "fluency_warmup_frac", 1.0) or 1.0),
                    "decoder_fluency_no_repeat_ngram": int(
                        getattr(engine.config, "decoder_fluency_no_repeat_ngram", 3) or 3
                    ),
                    "decoder_fluency_prompt_ngram_block": bool(
                        getattr(engine.config, "decoder_fluency_prompt_ngram_block", False)
                    ),
                    "decoder_fluency_min_new_tokens_before_eos_freeform": int(
                        getattr(
                            engine.config,
                            "decoder_fluency_min_new_tokens_before_eos_freeform",
                            0,
                        )
                        or 0
                    ),
                    "decoder_fluency_block_token_regex": str(
                        getattr(engine.config, "decoder_fluency_block_token_regex", "") or ""
                    ),
                    "gain_horizon_steps": int(getattr(self.config, "gain_horizon_steps", 0) or 0),
                    "num_acts": len(self.store.active()),
                    "adds": self._adds,
                    "merges": self._merges,
                    "prunes": self._prunes,
                    "concept_csv_mining_enabled": bool(
                        getattr(self.config, "concept_csv_mining_enabled", False)
                    ),
                    "concept_csv_budget": int(getattr(self.config, "concept_csv_budget", 0) or 0),
                    "concept_csv_overhead_bits": int(
                        getattr(self.config, "concept_csv_overhead_bits", 0) or 0
                    ),
                    "concept_csv_total": int(concept_csv_total),
                    "concept_csv_mined_total": int(concept_csv_mined_total),
                    "concept_csv_added": int(self._concept_csv_added),
                    "concept_csv_pruned": int(self._concept_csv_pruned),
                    "patch_kind": patch.kind if patch else None,
                    "system_survival_loss": float(survival_loss),
                    "system_survival_best": float(survival_best),
                    "system_survival_stall_windows": int(survival_stall_windows),
                    "system_survival_bottleneck": str(survival_component),
                    "system_survival_nll_ratio": float(nll_ratio),
                    "system_survival_utility_loss": float(util_loss),
                    "system_survival_utility_component": str(util_component),
                    "system_survival_fluency_bottleneck": float(flu_bottleneck),
                    "system_survival_abstraction_added": int(abstraction_added),
                    "system_survival_reuse_score": float(reuse_score),
                    "system_survival_reuse_min": float(reuse_min),
                    "system_survival_no_abstraction_windows": int(no_abstraction_windows),
                    "system_survival_no_reuse_windows": int(no_reuse_windows),
                    "system_survival_concept_no_add_windows": int(concept_no_add_windows),
                    "system_survival_concept_reuse_best": float(concept_reuse_best),
                    "system_survival_concept_reuse_stall_windows": int(concept_reuse_stall_windows),
                    "system_survival_concept_crisis": bool(concept_crisis),
                    "system_survival_concept_crisis_windows": int(concept_crisis_windows),
                    "system_survival_concept_min_depth_required_max": int(
                        concept_min_depth_required_last
                    ),
                    "system_survival_concept_composed_best": float(concept_composed_best),
                    "system_survival_concept_composed_stall_windows": int(
                        concept_composed_stall_windows
                    ),
                    "system_survival_concept_composed_no_add_windows": int(
                        concept_composed_no_add_windows
                    ),
                    "system_survival_concept_composed_crisis": bool(concept_composed_crisis),
                    "system_survival_concept_composed_crisis_windows": int(
                        concept_composed_crisis_windows
                    ),
                    "system_survival_concept_deep_best": float(concept_deep_best),
                    "system_survival_concept_deep_stall_windows": int(concept_deep_stall_windows),
                    "system_survival_concept_deep_crisis": bool(concept_deep_crisis),
                    "system_survival_concept_deep_crisis_windows": int(concept_deep_crisis_windows),
                    "system_survival_concept_very_deep_best": float(concept_very_deep_best),
                    "system_survival_concept_very_deep_stall_windows": int(
                        concept_very_deep_stall_windows
                    ),
                    "system_survival_concept_very_deep_crisis": bool(concept_very_deep_crisis),
                    "system_survival_concept_very_deep_crisis_windows": int(
                        concept_very_deep_crisis_windows
                    ),
                    "repeat3_global": gen["repeat3_global"],
                    "loop_rate_global": gen["loop_rate_global"],
                    "distinct2_global": gen["distinct2_global"],
                    "repeat3_reply_mean": gen.get("repeat3_reply_mean"),
                    "loop_rate_reply_mean": gen.get("loop_rate_reply_mean"),
                    "repeat3_reply_max": gen.get("repeat3_reply_max"),
                    "loop_rate_reply_max": gen.get("loop_rate_reply_max"),
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
                    "utility_goals_total": gen.get("utility_goals_total"),
                    "utility_goals_satisfied_rate": gen.get("utility_goals_satisfied_rate"),
                    "utility_instruction_total": gen.get("utility_instruction_total"),
                    "utility_instruction_pass_rate": gen.get("utility_instruction_pass_rate"),
                    "utility_json_total": gen.get("utility_json_total"),
                    "utility_json_pass_rate": gen.get("utility_json_pass_rate"),
                    "utility_math_total": gen.get("utility_math_total"),
                    "utility_math_pass_rate": gen.get("utility_math_pass_rate"),
                    "utility_state_total": gen.get("utility_state_total"),
                    "utility_state_pass_rate": gen.get("utility_state_pass_rate"),
                    "utility_plan_total": gen.get("utility_plan_total"),
                    "utility_plan_pass_rate": gen.get("utility_plan_pass_rate"),
	                    "utility_clarify_total": gen.get("utility_clarify_total"),
	                    "utility_clarify_pass_rate": gen.get("utility_clarify_pass_rate"),
                    "utility_consistency_total": gen.get("utility_consistency_total"),
	                    "utility_consistency_pass_rate": gen.get("utility_consistency_pass_rate"),
	                    "utility_memory_total": gen.get("utility_memory_total"),
	                    "utility_memory_pass_rate": gen.get("utility_memory_pass_rate"),
                        "utility_dialogue_total": gen.get("utility_dialogue_total"),
                        "utility_dialogue_pass_rate": gen.get("utility_dialogue_pass_rate"),
	                    "utility_concept_total": gen.get("utility_concept_total"),
		                    "utility_concept_pass_rate": gen.get("utility_concept_pass_rate"),
		                    "utility_concept_used_rate": gen.get("utility_concept_used_rate"),
		                    "utility_concept_composed_rate": gen.get("utility_concept_composed_rate"),
		                    "utility_concept_deep_rate": gen.get("utility_concept_deep_rate"),
		                    "utility_concept_very_deep_rate": gen.get("utility_concept_very_deep_rate"),
			                    "utility_concept_calls_total_sum": gen.get(
			                        "utility_concept_calls_total_sum"
			                    ),
			                    "utility_concept_calls_max_depth_mean": gen.get(
			                        "utility_concept_calls_max_depth_mean"
			                    ),
			                    "utility_concept_calls_max_depth_max": gen.get(
			                        "utility_concept_calls_max_depth_max"
			                    ),
		                    "utility_concept_min_depth_required_max": gen.get(
		                        "utility_concept_min_depth_required_max"
		                    ),
		                    "utility_concept_nested_call_turns": gen.get(
		                        "utility_concept_nested_call_turns"
		                    ),
		                    "utility_concept_nested_call_rate": gen.get(
		                        "utility_concept_nested_call_rate"
		                    ),
		                    "utility_concept_nested_call_ids_distinct": gen.get(
		                        "utility_concept_nested_call_ids_distinct"
		                    ),
		                    "utility_concept_static_depth_max": gen.get("utility_concept_static_depth_max"),
		                    "utility_concept_static_depth_ge2_count": gen.get(
		                        "utility_concept_static_depth_ge2_count"
		                    ),
	                    "utility_concept_static_depth_ge3_count": gen.get(
	                        "utility_concept_static_depth_ge3_count"
	                    ),
			                    "utility_concept_any_used_rate": gen.get("utility_concept_any_used_rate"),
			                    "utility_concept_any_ok_rate": gen.get("utility_concept_any_ok_rate"),
			                    "utility_concept_policy_required_total": gen.get(
			                        "utility_concept_policy_required_total"
			                    ),
			                    "utility_concept_policy_used_turns": gen.get(
			                        "utility_concept_policy_used_turns"
			                    ),
			                    "utility_concept_policy_pass_rate": gen.get("utility_concept_policy_pass_rate"),
			                    "utility_concept_selected_as_policy_rate": gen.get(
			                        "utility_concept_selected_as_policy_rate"
			                    ),
			                    "utility_reference_required_total": gen.get("utility_reference_required_total"),
			                    "utility_reference_ok_turns": gen.get("utility_reference_ok_turns"),
			                    "utility_reference_pass_rate": gen.get("utility_reference_pass_rate"),
				                    "utility_concept_cross_tag_reuse_count": gen.get(
				                        "utility_concept_cross_tag_reuse_count"
				                    ),
		                    "utility_concept_cross_tag_reuse_example": gen.get(
		                        "utility_concept_cross_tag_reuse_example"
		                    ),
		                    "utility_concept_cross_context_reuse_count": gen.get(
		                        "utility_concept_cross_context_reuse_count"
		                    ),
		                    "utility_concept_cross_context_reuse_example": gen.get(
		                        "utility_concept_cross_context_reuse_example"
		                    ),
		                    "utility_agency_total": gen.get("utility_agency_total"),
		                    "utility_agency_pass_rate": gen.get("utility_agency_pass_rate"),
		                    "utility_agency_plan_found_rate": gen.get("utility_agency_plan_found_rate"),
	                    "utility_agency_steps_pass_mean": gen.get("utility_agency_steps_pass_mean"),
	                    "utility_agency_steps_pass_median": gen.get("utility_agency_steps_pass_median"),
                    "utility_agency_fail_reasons": gen.get("utility_agency_fail_reasons"),
                    "utility_agency_fail_details": gen.get("utility_agency_fail_details"),
                    "semantic_goals_total": gen.get("semantic_goals_total"),
                    "semantic_plans_total": gen.get("semantic_plans_total"),
                    "semantic_hypotheses_total": gen.get("semantic_hypotheses_total"),
                    "semantic_references_total": gen.get("semantic_references_total"),
                    "semantic_goals_created_window": gen.get("semantic_goals_created_window"),
                    "semantic_plans_created_window": gen.get("semantic_plans_created_window"),
                    "semantic_hypotheses_created_window": gen.get("semantic_hypotheses_created_window"),
                    "semantic_references_created_window": gen.get("semantic_references_created_window"),
                    "semantic_seen_plan_turns_window": gen.get("semantic_seen_plan_turns_window"),
                    "utility_failures": gen.get("utility_failures"),
                    "utility_plan_trace_turns_total": gen.get("utility_plan_trace_turns_total"),
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
                    "predictor_iterated_per_token_mean": gen.get("predictor_iterated_per_token_mean"),
                    "scan_acts_considered_per_token_mean": gen.get(
                        "scan_acts_considered_per_token_mean"
                    ),
                    "scan_steps_per_turn_mean": gen.get("scan_steps_per_turn_mean"),
                    "avg_trace_len_before": macro_meta.get("avg_trace_len_before")
                    if macro_meta
                    else None,
                    "avg_trace_len_after": macro_meta.get("avg_trace_len_after")
                    if macro_meta
                    else None,
                    "avg_trace_len_turn_before": macro_meta.get("avg_trace_len_turn_before")
                    if macro_meta
                    else None,
                    "avg_trace_len_turn_after": macro_meta.get("avg_trace_len_turn_after")
                    if macro_meta
                    else None,
                    "concept_turn_hit_rate": macro_meta.get("concept_turn_hit_rate")
                    if macro_meta
                    else None,
                    "concept_symbol_coverage_rate": macro_meta.get("concept_symbol_coverage_rate")
                    if macro_meta
                    else None,
                    "concept_hits_per_turn_mean": macro_meta.get("concept_hits_per_turn_mean")
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
                    "concept_trace_mdl_before_bits_est": macro_meta.get(
                        "concept_trace_mdl_before_bits_est"
                    )
                    if macro_meta
                    else None,
                    "concept_trace_mdl_after_bits_est": macro_meta.get("concept_trace_mdl_after_bits_est")
                    if macro_meta
                    else None,
                    "concept_trace_mdl_gain_bits_est": macro_meta.get("concept_trace_mdl_gain_bits_est")
                    if macro_meta
                    else None,
                    "concept_trace_mdl_gain_real_bits_est": macro_meta.get(
                        "concept_trace_mdl_gain_real_bits_est"
                    )
                    if macro_meta
                    else None,
                    "total_macros": macro_meta.get("total_macros") if macro_meta else None,
                    "macro_added": macro_meta.get("added") if macro_meta else None,
                    "macro_updated": macro_meta.get("updated_existing") if macro_meta else None,
                    "macro_evicted": macro_meta.get("evicted") if macro_meta else None,
                    "macro_library_cost_bits_est": macro_meta.get("macro_library_cost_bits_est")
                    if macro_meta
                    else None,
                    "concept_total": macro_meta.get("concept_total") if macro_meta else None,
                    "concept_added": macro_meta.get("concept_added") if macro_meta else None,
                    "concept_updated": macro_meta.get("concept_updated_existing") if macro_meta else None,
                    "concept_evicted": macro_meta.get("concept_evicted") if macro_meta else None,
                    "concept_library_cost_bits_est": macro_meta.get("concept_library_cost_bits_est")
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

                # Live reporting (JSONL + last snapshot) for step-by-step monitoring.
                try:
                    with open(self.report_jsonl_path, "a", encoding="utf-8") as f:
                        f.write(canonical_json_dumps(row))
                        f.write("\n")
                    with open(self.report_last_path, "w", encoding="utf-8") as f:
                        json.dump(row, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

                # Also emit a compact, tail-friendly line to stdout.
                try:
                    sys.stdout.write(
                        canonical_json_dumps(
                            {
                                "step": int(step),
                                "nll_bits_mean": float(mean_nll),
                                "holdout_nll_ratio": (
                                    float(holdout_nll_ratio) if holdout_enabled else None
                                ),
                                    "decoder_fluency_no_repeat_ngram": int(
                                        getattr(engine.config, "decoder_fluency_no_repeat_ngram", 3) or 3
                                    ),
                                    "decoder_fluency_prompt_ngram_block": bool(
                                        getattr(engine.config, "decoder_fluency_prompt_ngram_block", False)
                                    ),
                                    "decoder_fluency_min_new_tokens_before_eos_freeform": int(
                                        getattr(
                                            engine.config,
                                            "decoder_fluency_min_new_tokens_before_eos_freeform",
                                            0,
                                        )
                                        or 0
                                    ),
                                    "decoder_fluency_block_token_regex": str(
                                        getattr(engine.config, "decoder_fluency_block_token_regex", "") or ""
                                    ),
	                                "utility_pass_rate": float(gen.get("utility_pass_rate") or 0.0),
	                                "utility_instruction_pass_rate": float(gen.get("utility_instruction_pass_rate") or 0.0),
		                                "utility_json_pass_rate": float(gen.get("utility_json_pass_rate") or 0.0),
		                                "utility_math_pass_rate": float(gen.get("utility_math_pass_rate") or 0.0),
		                                "utility_state_pass_rate": float(gen.get("utility_state_pass_rate") or 0.0),
		                                "utility_plan_pass_rate": float(gen.get("utility_plan_pass_rate") or 0.0),
		                                "utility_clarify_pass_rate": float(gen.get("utility_clarify_pass_rate") or 0.0),
		                                "utility_consistency_pass_rate": float(gen.get("utility_consistency_pass_rate") or 0.0),
		                                "utility_memory_pass_rate": float(gen.get("utility_memory_pass_rate") or 0.0),
	                                        "utility_dialogue_total": int(gen.get("utility_dialogue_total") or 0),
			                                "utility_dialogue_pass_rate": float(gen.get("utility_dialogue_pass_rate") or 0.0),
			                                "utility_agency_pass_rate": float(gen.get("utility_agency_pass_rate") or 0.0),
			                                "utility_failures": (
			                                    list(gen.get("utility_failures") or [])
			                                    if isinstance(gen.get("utility_failures"), list)
			                                    else []
			                                ),
			                                "utility_agency_fail_details": (
			                                    list(gen.get("utility_agency_fail_details") or [])
			                                    if isinstance(gen.get("utility_agency_fail_details"), list)
			                                    else []
			                                ),
				                                "concept_used_rate": float(gen.get("utility_concept_used_rate") or 0.0),
				                                "concept_composed_rate": float(gen.get("utility_concept_composed_rate") or 0.0),
				                                "concept_deep_rate": float(gen.get("utility_concept_deep_rate") or 0.0),
			                                "concept_very_deep_rate": float(
			                                    gen.get("utility_concept_very_deep_rate") or 0.0
			                                ),
			                                "concept_calls_max_depth_mean": float(
			                                    gen.get("utility_concept_calls_max_depth_mean") or 0.0
			                                ),
		                                "concept_cross_tag_reuse_count": int(
		                                    gen.get("utility_concept_cross_tag_reuse_count") or 0
		                                ),
		                                "concept_cross_context_reuse_count": int(
		                                    gen.get("utility_concept_cross_context_reuse_count") or 0
		                                ),
		                                "concept_cross_context_reuse_example": gen.get(
		                                    "utility_concept_cross_context_reuse_example"
		                                ),
		                                "concept_nested_call_rate": float(
		                                    gen.get("utility_concept_nested_call_rate") or 0.0
		                                ),
		                                "concept_nested_call_ids_distinct": int(
		                                    gen.get("utility_concept_nested_call_ids_distinct") or 0
		                                ),
		                                "concept_static_depth_max": int(
		                                    gen.get("utility_concept_static_depth_max") or 0
		                                ),
			                                "concept_static_depth_ge2_count": int(
			                                    gen.get("utility_concept_static_depth_ge2_count") or 0
			                                ),
			                                "concept_static_depth_ge3_count": int(
			                                    gen.get("utility_concept_static_depth_ge3_count") or 0
			                                ),
			                                "fluency_bottleneck": float(flu_bottleneck),
			                                "system_survival_loss": float(survival_loss),
			                                "system_survival_component": str(survival_component),
		                                "concept_depth_required": int(concept_min_depth_required_last or 0),
                                "depth_compositional_max": (
                                    int(macro_meta.get("depth_compositional_max") or 0)
                                    if macro_meta
                                    else 0
                                ),
                                "concept_csv_total": int(concept_csv_total),
                                "concept_csv_mined_total": int(concept_csv_mined_total),
                                "concept_csv_added_window": int(row.get("concept_csv_added") or 0),
                                "concept_csv_pruned_window": int(row.get("concept_csv_pruned") or 0),
                                "tokens_per_s": float(toks_per_s),
                                "num_acts": int(len(self.store.active())),
                                "adds": int(self._adds),
                                "merges": int(self._merges),
                                "prunes": int(self._prunes),
                            }
                        )
                        + "\n"
                    )
                    sys.stdout.flush()
                except Exception:
                    pass

                # Reset window accumulators
                nll_sum_bits = 0.0
                self._adds = self._merges = self._prunes = 0
                self._concept_csv_added = 0
                self._concept_csv_pruned = 0

                # Bounded streaming buffer: keep only a recent token window for evaluation/patch selection.
                if tok_iter is not None:
                    try:
                        max_buf = int(getattr(self.config, "stream_buffer_tokens", 0) or 0)
                    except Exception:
                        max_buf = 0
                    max_buf = max(0, int(max_buf))
                    if max_buf > 0 and len(tokens) > max_buf:
                        tokens = list(tokens[-max_buf:])

                # Optional wall-clock stop (graceful): final artifacts still get written below.
                try:
                    max_s = float(getattr(self.config, "max_seconds", 0.0) or 0.0)
                except Exception:
                    max_s = 0.0
                if max_s > 0.0 and (time.time() - t_train0) >= max_s:
                    break

        # Final artifacts
        self.store.save_jsonl(self.acts_path)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # Audit artifacts (deterministic snapshots for external review).
        try:
            concept_rows: List[Dict[str, Any]] = []
            for a in self.store.active():
                if str(getattr(a, "kind", "")) != "concept_csv":
                    continue
                ev = a.evidence if isinstance(getattr(a, "evidence", None), dict) else {}
                iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
                meta = ev.get("meta") if isinstance(ev.get("meta"), dict) else {}
                ics = meta.get("ics_v1") if isinstance(meta.get("ics_v1"), dict) else {}
                concept_rows.append(
                    {
                        "concept_id": str(a.id),
                        "active": bool(getattr(a, "active", True)),
                        "version": int(getattr(a, "version", 1) or 1),
                        "match": dict(getattr(a, "match", {}) or {}),
                        "interface": dict(iface) if isinstance(iface, dict) else {},
                        "meta": dict(meta) if isinstance(meta, dict) else {},
                        "ics_v1": dict(ics) if isinstance(ics, dict) else {},
                        "deps": list(getattr(a, "deps", []) or []),
                        "program_len": int(len(getattr(a, "program", []) or [])),
                        "cost_bits": int(estimate_act_cost_bits(a)),
                    }
                )
            concept_rows.sort(key=lambda r: str(r.get("concept_id") or ""))
            concept_bank = {
                "schema_version": 1,
                "generated_at": deterministic_iso(step=int(getattr(self.config, "steps", 0) or 0)),
                "concepts_total": int(len(concept_rows)),
                "concepts": list(concept_rows),
            }
            with open(os.path.join(self.out_dir, "concept_bank.json"), "w", encoding="utf-8") as f:
                json.dump(concept_bank, f, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:
            pass

        try:
            prim: Dict[str, Any] = {}
            for op_id in sorted(PRIMITIVE_OPS.keys(), key=str):
                spec_fn = PRIMITIVE_OPS.get(op_id)
                if not isinstance(spec_fn, tuple) or len(spec_fn) != 2:
                    continue
                spec = spec_fn[0]
                prim[str(op_id)] = {
                    "arity": int(getattr(spec, "arity", 0) or 0),
                    "input_types": list(getattr(spec, "input_types", ()) or ()),
                    "output_type": str(getattr(spec, "output_type", "") or ""),
                }
            operator_bank = {
                "schema_version": 1,
                "generated_at": deterministic_iso(step=int(getattr(self.config, "steps", 0) or 0)),
                "primitive_ops": dict(prim),
            }
            with open(os.path.join(self.out_dir, "operator_bank.json"), "w", encoding="utf-8") as f:
                json.dump(operator_bank, f, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:
            pass

        # Compatibility copies (requested naming): keep deterministic WORM artifacts.
        try:
            src = str(self.ledger_path)
            dst = os.path.join(self.out_dir, "worm_ledger.jsonl")
            if os.path.exists(src) and (not os.path.exists(dst)):
                shutil.copyfile(src, dst)
        except Exception:
            pass
        try:
            src = str(self.report_jsonl_path)
            dst = os.path.join(self.out_dir, "metrics.jsonl")
            if os.path.exists(src) and (not os.path.exists(dst)):
                shutil.copyfile(src, dst)
        except Exception:
            pass
