from __future__ import annotations

import math
import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import Act, Instruction
from .atolang import AtoLangVM, Candidate
from .metrics import detokenize, is_space, tokenize_text
from .suite import last_user_text_from_prompt, user_signature_from_prompt, user_signatures_from_prompt

SEP = "\u241f"


def ctx_key(ctx: Sequence[str]) -> str:
    return SEP.join(ctx)


@dataclass
class EngineConfig:
    max_order: int = 4
    top_k: int = 64
    alpha: float = 0.5
    order_bonus: float = 0.15
    eos_bias: float = -2.0
    repetition_recent: int = 64
    cycle_ngram: int = 3
    cycle_history: int = 512
    min_new_tokens_before_eos: int = 8
    # Gate-live (default OFF): use macro_router_v0 to restrict predictor evaluation.
    router_live_enabled: bool = False
    # Disable macro_router usage even if router_live_enabled=True (default OFF).
    disable_macro_router: bool = False
    # Proof mode: compute baseline + gate per token, count mismatches, and fall back to baseline.
    router_live_debug_compare: bool = False
    # Force evaluation of only these predictor act_ids (default OFF). Applies only to n-gram
    # predictors and only when `predictors` is not passed explicitly to `_emit_candidates`.
    force_predictor_ids: Optional[List[str]] = None
    # Force predictor evaluation by ctx_sig (mode␟ctx_key). When present, restricts predictor
    # scanning per token without relying on macro_router_v0 (default OFF).
    force_predictor_ids_by_ctx_sig: Optional[Dict[str, List[str]]] = None
    # Deterministic instruction contracts (default OFF).
    enable_contracts: bool = False


class Engine:
    def __init__(self, store, *, seed: int, config: Optional[EngineConfig] = None):
        self.store = store
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.vm = AtoLangVM(rng=self.rng)
        self.config = config or EngineConfig()
        self._tables: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._predictors: List[Act] = []
        self._predictor_by_id: Dict[str, Act] = {}
        self._predictor_order: Dict[str, int] = {}
        self._mode_selectors: List[Act] = []
        self._mode_policy: Optional[Act] = None
        self._rewrite_rules: List[Act] = []
        self._selector: Optional[Act] = None
        self._unigram: Optional[Act] = None
        self._macro_router: Optional[Act] = None
        self._macro_router_table: Optional[Dict[str, Any]] = None
        self._instruction_contract: Optional[Act] = None
        self._fact_memory: Optional[Act] = None
        self.rebuild_cache()

    def rebuild_cache(self) -> None:
        self._tables.clear()
        acts = self.store.active()
        predictors: List[Act] = []
        mode_selectors: List[Act] = []
        mode_policy: Optional[Act] = None
        rewrite_rules: List[Act] = []
        selector: Optional[Act] = None
        unigram: Optional[Act] = None
        macro_router: Optional[Act] = None
        instruction_contract: Optional[Act] = None
        fact_memory: Optional[Act] = None

        for act in acts:
            if act.kind == "predictor":
                predictors.append(act)
                table_id = str(act.evidence.get("table_id", act.id))
                act.evidence.setdefault("table_id", table_id)
                table = act.evidence.setdefault("table", {})
                self._tables[table_id] = table
                n = int(act.evidence.get("n", act.match.get("n", 1)))
                if n == 1 and unigram is None:
                    unigram = act
            elif act.kind == "rewrite_rule":
                rewrite_rules.append(act)
            elif act.kind == "mode_selector":
                mode_selectors.append(act)
            elif act.kind == "mode_policy" and mode_policy is None:
                mode_policy = act
            elif act.kind == "selector" and selector is None:
                selector = act
            elif act.kind == "compressor" and macro_router is None:
                ev = act.evidence
                if isinstance(ev, dict) and str(ev.get("name") or "") == "macro_router_v0":
                    macro_router = act
            elif act.kind == "candidate_source" and instruction_contract is None:
                ev = act.evidence
                if isinstance(ev, dict) and str(ev.get("name") or "") == "instruction_contract_v0":
                    instruction_contract = act
            elif act.kind == "memory_facts" and fact_memory is None:
                ev = act.evidence
                if isinstance(ev, dict) and str(ev.get("name") or "") == "fact_memory_v0":
                    fact_memory = act

        def _act_order(a: Act) -> Tuple[int, str]:
            n = int(a.evidence.get("n", a.match.get("n", 1)))
            return (-n, a.id)

        self._predictors = sorted(predictors, key=_act_order)
        self._predictor_by_id = {str(a.id): a for a in self._predictors}
        self._predictor_order = {str(a.id): i for i, a in enumerate(self._predictors)}
        self._mode_selectors = sorted(
            mode_selectors, key=lambda a: (str(a.evidence.get("mode", "")), a.id)
        )
        self._mode_policy = mode_policy
        self._rewrite_rules = sorted(rewrite_rules, key=lambda a: a.id)
        self._selector = selector
        self._unigram = unigram
        self._macro_router = macro_router
        self._macro_router_table = None
        if macro_router is not None:
            ev = macro_router.evidence
            if isinstance(ev, dict) and isinstance(ev.get("table"), dict):
                self._macro_router_table = ev.get("table")
        self._instruction_contract = instruction_contract
        self._fact_memory = fact_memory

    def vocab(self) -> List[str]:
        if self._unigram is None:
            return []
        table_id = str(self._unigram.evidence.get("table_id", self._unigram.id))
        table = self._tables.get(table_id, {})
        nxt = table.get("", {})
        return sorted(nxt.keys())

    def match_act(self, act: Act, *, context: Sequence[str]) -> bool:
        m = act.match or {}
        typ = m.get("type", "always")
        if typ == "always":
            return True
        if typ != "ngram":
            return True
        n = int(m.get("n", act.evidence.get("n", 1)))
        if len(context) < max(0, n - 1):
            return False

        if "prev_is_space" in m:
            prev = context[-1] if context else "<BOS>"
            want = bool(m["prev_is_space"])
            if is_space(prev) != want:
                return False
        if "prev_in" in m:
            prev = context[-1] if context else "<BOS>"
            if prev not in set(m["prev_in"]):
                return False
        if "prev_not_in" in m:
            prev = context[-1] if context else "<BOS>"
            if prev in set(m["prev_not_in"]):
                return False
        return True

    def _best_predictor(self, *, context: Sequence[str]) -> Optional[Act]:
        for act in self._predictors:
            if not self.match_act(act, context=context):
                continue
            n = int(act.evidence.get("n", act.match.get("n", 1)))
            table_id = str(act.evidence.get("table_id", act.id))
            table = self._tables.get(table_id, {})
            key = ctx_key(context[-(n - 1) :] if n > 1 else [])
            nxt = table.get(key)
            if nxt:
                return act
        return None

    def logprob_next(self, *, context: Sequence[str], token: str) -> float:
        vocab = self.vocab()
        V = max(1, len(vocab) + 1)  # +1 for unknown
        alpha = float(self.config.alpha)

        best_lp: Optional[float] = None

        def consider(act: Act, *, key: str) -> None:
            nonlocal best_lp
            table_id = str(act.evidence.get("table_id", act.id))
            table = self._tables.get(table_id, {})
            nxt = table.get(key)
            if not nxt:
                return
            total = float(sum(nxt.values()))
            cnt = float(nxt.get(token, 0))
            p = (cnt + alpha) / (total + alpha * V)
            lp = math.log(p + 1e-12)
            if best_lp is None or lp > best_lp:
                best_lp = lp

        for act in self._predictors:
            if not self.match_act(act, context=context):
                continue
            n = int(act.evidence.get("n", act.match.get("n", 1)))
            key = ctx_key(context[-(n - 1) :] if n > 1 else [])
            consider(act, key=key)

        # Fallback: uniform over current vocab (+unknown).
        if best_lp is None:
            return -math.log(V)
        return best_lp

    def observe(self, *, context: Sequence[str], token: str) -> None:
        for act in self._predictors:
            if not self.match_act(act, context=context):
                continue
            n = int(act.evidence.get("n", act.match.get("n", 1)))
            table_id = str(act.evidence.get("table_id", act.id))
            table = self._tables.get(table_id)
            if table is None:
                continue
            key = ctx_key(context[-(n - 1) :] if n > 1 else [])
            allow_new_ctx = bool(act.evidence.get("allow_new_contexts", True))
            allow_new_tok = bool(act.evidence.get("allow_new_tokens", True))
            max_contexts = int(act.evidence.get("max_contexts", 0) or 0)
            max_next = int(act.evidence.get("max_next_per_ctx", 0) or 0)
            evict_policy = str(act.evidence.get("evict_policy", "")) or "fifo"

            ctx_fifo = None
            next_fifo = None
            if evict_policy == "fifo":
                ctx_fifo = act.evidence.get("ctx_fifo")
                if not isinstance(ctx_fifo, list):
                    ctx_fifo = []
                    act.evidence["ctx_fifo"] = ctx_fifo
                next_fifo = act.evidence.get("next_fifo")
                if not isinstance(next_fifo, dict):
                    next_fifo = {}
                    act.evidence["next_fifo"] = next_fifo

            nxt = table.get(key)
            if nxt is None:
                if not allow_new_ctx:
                    continue
                if max_contexts > 0:
                    if evict_policy == "fifo":
                        while len(table) >= max_contexts:
                            if ctx_fifo:
                                old_key = ctx_fifo.pop(0)
                            else:
                                # Deterministic fallback if FIFO metadata is missing.
                                old_key = min(
                                    table.items(),
                                    key=lambda kv: (
                                        sum(int(c) for c in kv[1].values()),
                                        kv[0],
                                    ),
                                )[0]
                            table.pop(old_key, None)
                            if isinstance(next_fifo, dict):
                                next_fifo.pop(old_key, None)
                            if isinstance(ctx_fifo, list):
                                while old_key in ctx_fifo:
                                    ctx_fifo.remove(old_key)
                    elif evict_policy == "count_lex":
                        while len(table) >= max_contexts and table:
                            old_key = min(
                                table.items(),
                                key=lambda kv: (
                                    sum(int(c) for c in kv[1].values()),
                                    kv[0],
                                ),
                            )[0]
                            table.pop(old_key, None)
                            # If legacy FIFO metadata exists, keep it consistent.
                            cf = act.evidence.get("ctx_fifo")
                            if isinstance(cf, list):
                                while old_key in cf:
                                    cf.remove(old_key)
                            nf = act.evidence.get("next_fifo")
                            if isinstance(nf, dict):
                                nf.pop(old_key, None)
                nxt = {}
                table[key] = nxt
                if max_contexts > 0 and evict_policy == "fifo" and isinstance(ctx_fifo, list):
                    ctx_fifo.append(key)
                if max_next > 0 and evict_policy == "fifo" and isinstance(next_fifo, dict):
                    next_fifo.setdefault(key, [])

            if token not in nxt and not allow_new_tok:
                continue
            if token not in nxt and max_next > 0:
                if evict_policy == "fifo":
                    fifo = next_fifo.get(key) if isinstance(next_fifo, dict) else None
                    if not isinstance(fifo, list):
                        fifo = []
                        if isinstance(next_fifo, dict):
                            next_fifo[key] = fifo
                    while len(nxt) >= max_next and fifo:
                        old_tok = fifo.pop(0)
                        nxt.pop(old_tok, None)
                    if len(nxt) >= max_next and nxt:
                        # Deterministic fallback if FIFO metadata is missing.
                        old_tok = min(nxt.items(), key=lambda kv: (kv[1], kv[0]))[0]
                        nxt.pop(old_tok, None)
                    if len(nxt) >= max_next:
                        continue
                    fifo.append(token)
                elif evict_policy == "count_lex":
                    if len(nxt) >= max_next and nxt:
                        worst_tok, worst_cnt = min(nxt.items(), key=lambda kv: (int(kv[1]), kv[0]))
                        # Candidate count after this observation is 1.
                        if 1 < int(worst_cnt) or (1 == int(worst_cnt) and str(token) > str(worst_tok)):
                            continue
                        nxt.pop(worst_tok, None)
            nxt[token] = int(nxt.get(token, 0)) + 1
            act.evidence["updates"] = int(act.evidence.get("updates", 0)) + 1

    def _emit_candidates(
        self,
        *,
        context: Sequence[str],
        penalties: Optional[Dict[str, Any]] = None,
        predictors: Optional[Sequence[Act]] = None,
        trace: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Candidate]:
        vocab = self.vocab()
        penalties = penalties or {}
        merged: Dict[str, Candidate] = {}
        predictor_iterated = 0
        predictor_matched = 0
        predictor_emitted = 0
        executed_predictor_ids: List[str] = []
        pred_list: Sequence[Act] = predictors if predictors is not None else self._predictors
        if predictors is None:
            forced = self.config.force_predictor_ids
            if isinstance(forced, list) and forced:
                allowed = {str(x) for x in forced if isinstance(x, str) and str(x)}
                if allowed:
                    pred_list = [a for a in self._predictors if str(a.id) in allowed]
            else:
                by_ctx = self.config.force_predictor_ids_by_ctx_sig
                if isinstance(by_ctx, dict) and by_ctx:
                    mode_name = penalties.get("mode") if isinstance(penalties, dict) else None
                    if not isinstance(mode_name, str) or not mode_name:
                        mode_name = "default"
                    sig = f"{mode_name}{SEP}{ctx_key(context)}"
                    ids = by_ctx.get(sig)
                    if isinstance(ids, list) and ids:
                        allowed = {str(x) for x in ids if isinstance(x, str) and str(x)}
                        if allowed:
                            pred_list = [a for a in self._predictors if str(a.id) in allowed]
        for act in pred_list:
            predictor_iterated += 1
            if not self.match_act(act, context=context):
                continue
            predictor_matched += 1

            n = int(act.evidence.get("n", act.match.get("n", 1)))
            table_id = str(act.evidence.get("table_id", act.id))
            table = self._tables.get(table_id, {})
            key = ctx_key(context[-(n - 1) :] if n > 1 else [])
            nxt_counts = table.get(key)
            if not nxt_counts:
                continue
            predictor_emitted += 1
            executed_predictor_ids.append(str(act.id))

            table_id = str(act.evidence.get("table_id", act.id))
            st = self.vm.run(
                act,
                context=context,
                tables=self._tables,
                vocab=vocab,
                penalties=penalties,
                mode="emit_only",
            )
            n = int(act.evidence.get("n", act.match.get("n", 1)))
            bonus = self.config.order_bonus * max(0, n - 1)
            for tok, cand in st.candidates.items():
                score = cand.score + bonus
                prev = merged.get(tok)
                if prev is None or score > prev.score:
                    merged[tok] = Candidate(
                        token=tok,
                        score=score,
                        source_act=cand.source_act,
                        meta=dict(cand.meta),
                    )

        # Ensure EOS exists as an option.
        if "<EOS>" not in merged:
            merged["<EOS>"] = Candidate(token="<EOS>", score=float(self.config.eos_bias), source_act="__engine__")

        if isinstance(trace, dict):
            trace["predictor_iterated"] = int(predictor_iterated)
            trace["predictor_matched"] = int(predictor_matched)
            trace["predictor_emitted"] = int(predictor_emitted)
            trace["candidates_pre_rewrite"] = int(len(merged))
            trace["executed_predictor_ids"] = list(executed_predictor_ids)
        return merged

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 200,
        mode: str = "greedy",
        dialogue_id: Optional[int] = None,
        turn: Optional[int] = None,
    ) -> Dict[str, Any]:
        prompt_tokens = tokenize_text(prompt)
        context: List[str] = ["<BOS>"] * (self.config.max_order - 1)
        for t in prompt_tokens[-(self.config.max_order - 1) :]:
            context.append(t)
        context = context[-(self.config.max_order - 1) :]

        out_tokens: List[str] = list(prompt_tokens)
        gen_tokens: List[str] = []

        history_ngrams: List[Tuple[str, ...]] = []
        history_ngram_set = set()

        active_set_size = len(self.store.active())
        predictor_total = int(len(self._predictors))
        rewrite_rules_total = int(len(self._rewrite_rules))
        selector_id = self._selector.id if self._selector is not None else None

        trace_selected_act_ids: List[str] = []
        trace_selected_tokens: List[str] = []
        trace_candidates_pre: List[int] = []
        trace_candidates_post: List[int] = []
        trace_predictor_iterated: List[int] = []
        trace_predictor_matched: List[int] = []
        trace_predictor_emitted: List[int] = []
        trace_executed_predictor_ids: List[List[str]] = []
        trace_context_keys: List[str] = []
        trace_router_live_used: List[int] = []
        trace_router_live_fallback: List[int] = []
        trace_router_live_fallback_reason: List[str] = []
        trace_router_live_allowed_predictor_ids: List[List[str]] = []
        trace_router_live_predictors_iterated: List[int] = []
        trace_router_live_predictors_matched: List[int] = []
        trace_router_live_predictors_emitted: List[int] = []
        trace_baseline_predictors_iterated: List[int] = []
        trace_baseline_predictors_matched: List[int] = []
        trace_baseline_predictors_emitted: List[int] = []
        trace_router_live_mismatch: List[int] = []
        trace_router_live_debug_baseline_token: List[str] = []
        trace_router_live_debug_gate_token: List[str] = []
        trace_rewrite_rule_hit_ids: List[List[str]] = []
        trace_rewrite_rules_changed_count: List[int] = []
        trace_instruction_contract_used: List[int] = []
        trace_instruction_contract_kind: List[str] = []
        trace_instruction_contract_reason: List[str] = []

        mode_state = "default"
        mode_act_id: Optional[str] = None
        mode_source = "router"
        mode_user_sig: Optional[str] = None
        mode_policy_action: Optional[str] = None
        policy_coverage: Optional[float] = None
        if self._mode_selectors:
            # KA-MODE: if a mode policy exists and has evidence for this prompt, select the
            # mode deterministically by lowest avg penalty (ties: mode name).
            did = int(dialogue_id or 0)
            trn = int(turn or 0)

            # Deterministic per-dialogue permutation: used for tie-breaks and as a fallback.
            def _rank(a: Act) -> bytes:
                key = f"{self.seed}|{did}|{a.id}".encode("utf-8")
                return hashlib.sha256(key).digest()

            ordered = sorted(self._mode_selectors, key=lambda a: (_rank(a), a.id))
            router_modes: List[str] = []
            mode_to_act: Dict[str, Act] = {}
            for a in ordered:
                m = str(a.evidence.get("mode") or "default")
                if m not in mode_to_act:
                    mode_to_act[m] = a
                    router_modes.append(m)

            def _router_fallback(allowed: Sequence[str]) -> str:
                for m in router_modes:
                    if m in set(allowed):
                        return m
                return router_modes[0] if router_modes else "default"

            def _policy_select(
                user_sig: str, *, allowed: Sequence[str], k: int, min_trials: int, table: Dict[str, Any]
            ) -> Tuple[Optional[str], str, float]:
                row = table.get(user_sig)
                if not isinstance(row, dict):
                    return None, "fallback", 0.0

                allowed_set = set(str(x) for x in allowed)
                if not allowed_set:
                    allowed_set = set(router_modes)

                # Coverage across all modes (not just allowed).
                covered = 0
                for m in router_modes:
                    st = row.get(m)
                    trials = int(st.get("trials", 0)) if isinstance(st, dict) else 0
                    if trials > 0:
                        covered += 1
                coverage = covered / max(1, len(router_modes))

                # Determine if we're still exploring for this user_sig.
                trials_by = {}
                for m in router_modes:
                    st = row.get(m)
                    trials_by[m] = int(st.get("trials", 0)) if isinstance(st, dict) else 0

                need_explore = any(trials_by.get(m, 0) < int(min_trials) for m in allowed_set)

                if need_explore:
                    # Explore: pick lowest trials; tie-break by router order.
                    best = None
                    best_trials = 10**18
                    for m in router_modes:
                        if m not in allowed_set:
                            continue
                        tr = int(trials_by.get(m, 0))
                        if tr < best_trials:
                            best_trials = tr
                            best = m
                    return best, "explore", float(coverage)

                # Exploit: pick lowest avg penalty; tie-break by router order.
                best = None
                best_avg = float("inf")
                for m in router_modes:
                    if m not in allowed_set:
                        continue
                    st = row.get(m)
                    if not isinstance(st, dict):
                        continue
                    trials = int(st.get("trials", 0))
                    if trials <= 0:
                        continue
                    pen_sum = float(st.get("pen_sum", 0.0))
                    avg = pen_sum / max(1, trials)
                    if avg < best_avg - 1e-12:
                        best_avg = avg
                        best = m
                return best, "exploit", float(coverage)

            sel: Act
            if self._mode_policy is not None:
                ev = dict(self._mode_policy.evidence or {})
                k = int(ev.get("k", 2))
                min_trials = int(ev.get("min_trials", 1))
                table = ev.get("table", {})
                if isinstance(table, dict):
                    # Current turn user signature (always extracted from prompt).
                    cur_user_sig = user_signature_from_prompt(prompt, k=k)
                    mode_user_sig = cur_user_sig

                    # If dialogue/turn are provided, enforce a hard “no-repeat mode” constraint
                    # within the dialogue (when possible) by selecting sequentially for all turns
                    # up to `turn`, and forbidding already-used modes.
                    if dialogue_id is not None and turn is not None and trn > 0 and router_modes:
                        keys = set(table.keys())
                        sigs = [s for s in user_signatures_from_prompt(prompt, k=k) if s in keys]
                        if len(sigs) >= trn + 1:
                            sigs = sigs[: trn + 1]
                            used: set = set()
                            chosen_mode = None
                            chosen_action = None
                            chosen_cov = 0.0
                            for idx, us in enumerate(sigs):
                                allowed = [m for m in router_modes if m not in used]
                                m, action, cov = _policy_select(
                                    us, allowed=allowed, k=k, min_trials=min_trials, table=table
                                )
                                if m is None:
                                    m = _router_fallback(allowed)
                                    action = "fallback"
                                    cov = 0.0
                                used.add(m)
                                if idx == len(sigs) - 1:
                                    chosen_mode = m
                                    chosen_action = action
                                    chosen_cov = cov
                                    mode_user_sig = us
                            if chosen_mode is not None:
                                mode_state = chosen_mode
                                mode_policy_action = str(chosen_action)
                                policy_coverage = float(chosen_cov)
                                if chosen_action in {"explore", "exploit"}:
                                    mode_source = "policy"
                                sel = mode_to_act.get(chosen_mode) or ordered[int(trn) % len(ordered)]
                            else:
                                sel = ordered[int(trn) % len(ordered)]
                        else:
                            # If we can't recover per-turn signatures, fall back to per-turn selection.
                            m, action, cov = _policy_select(
                                cur_user_sig, allowed=router_modes, k=k, min_trials=min_trials, table=table
                            )
                            if m is not None:
                                mode_state = m
                                mode_policy_action = str(action)
                                policy_coverage = float(cov)
                                if action in {"explore", "exploit"}:
                                    mode_source = "policy"
                                sel = mode_to_act.get(m) or ordered[int(trn) % len(ordered)]
                            else:
                                sel = ordered[int(trn) % len(ordered)]
                    else:
                        m, action, cov = _policy_select(
                            cur_user_sig, allowed=router_modes, k=k, min_trials=min_trials, table=table
                        )
                        if m is not None:
                            mode_state = m
                            mode_policy_action = str(action)
                            policy_coverage = float(cov)
                            if action in {"explore", "exploit"}:
                                mode_source = "policy"
                            sel = mode_to_act.get(m) or ordered[int(trn) % len(ordered)]
                        else:
                            sel = ordered[int(trn) % len(ordered)]
                else:
                    sel = ordered[int(trn) % len(ordered)]
            else:
                sel = ordered[int(trn) % len(ordered)]

            st = self.vm.run(
                sel,
                context=context,
                tables=self._tables,
                vocab=self.vocab(),
                penalties={},
                mode="emit_only",
            )
            if st.mode:
                mode_state = st.mode
            else:
                mode_state = str(sel.evidence.get("mode") or "default")
            mode_act_id = sel.id

        contract_meta: Dict[str, Any] = {
            "enabled": bool(self.config.enable_contracts),
            "contract_act_id": self._instruction_contract.id if self._instruction_contract is not None else None,
            "used": False,
            "kind": "",
            "parsed_args": {},
            "reason": "",
        }
        contract_tokens: List[str] = []
        contract_kind = ""
        contract_reason = ""
        contract_act_id = self._instruction_contract.id if self._instruction_contract is not None else ""
        contract_pos = 0

        did = int(dialogue_id or 0)
        trn = int(turn or 0)

        def _strip_quotes(s: str) -> str:
            return (
                str(s)
                .strip()
                .strip("“”")
                .strip("\"'")
                .strip("`")
                .strip()
            )

        def _strip_edge_punct(s: str) -> str:
            return str(s).strip().strip(".,;:!?()[]{}").strip()

        def _mem_table() -> Optional[Dict[str, Any]]:
            act = self._fact_memory
            if act is None:
                return None
            ev = act.evidence
            if not isinstance(ev, dict) or not bool(ev.get("enabled", True)):
                return None
            table = ev.setdefault("table", {})
            if not isinstance(table, dict):
                table = {}
                ev["table"] = table
            return table

        def _mem_key(kind: str, *parts: str) -> str:
            suffix = ":".join(str(p) for p in parts if p is not None and str(p) != "")
            if suffix:
                return f"contract:{kind}:{did}:{suffix}"
            return f"contract:{kind}:{did}"

        def _mem_set(key: str, value: str) -> bool:
            table = _mem_table()
            if table is None:
                return False
            entry = table.get(key)
            if entry is None or not isinstance(entry, dict):
                entry = {
                    "value": str(value),
                    "confidence": 0.5,
                    "evidence_turns": [],
                    "last_seen_step": 0,
                    "count": 0,
                }
                table[key] = entry
            prev = str(entry.get("value", ""))
            if prev and prev != str(value):
                entry["confidence"] = min(1.0, float(entry.get("confidence", 0.0)) + 0.01)
            entry["value"] = str(value)
            entry["count"] = int(entry.get("count", 0)) + 1
            cnt = int(entry["count"])
            entry["confidence"] = float(1.0 - (0.5**cnt))
            entry["last_seen_step"] = 0
            ev_turns = entry.get("evidence_turns")
            if not isinstance(ev_turns, list):
                ev_turns = []
                entry["evidence_turns"] = ev_turns
            ev_turns.append({"dialogue_id": int(did), "turn": int(trn), "mode": str(mode_state)})
            while len(ev_turns) > 8:
                ev_turns.pop(0)
            return True

        def _mem_get(key: str) -> Optional[str]:
            table = _mem_table()
            if table is None:
                return None
            entry = table.get(key)
            if not isinstance(entry, dict):
                return None
            val = str(entry.get("value", ""))
            return val if val else None

        if bool(self.config.enable_contracts):
            act = self._instruction_contract
            if act is None:
                contract_meta["reason"] = "contract_missing"
            elif not isinstance(act.evidence, dict) or not bool(act.evidence.get("enabled", True)):
                contract_meta["reason"] = "contract_disabled"
            else:
                user_txt = str(last_user_text_from_prompt(prompt) or "").strip()
                if not user_txt:
                    contract_meta["reason"] = "empty_user_text"
                else:
                    u = user_txt

                    exact_pats = [
                        re.compile(r"(?is)^\s*responda\s+exatamente\s*:\s*(.+?)\s*$"),
                        re.compile(r"(?is)^\s*retorne\s+exatamente\s+a\s+string\s*:\s*(.+?)\s*$"),
                        re.compile(r"(?is)^\s*devolva\s+exatamente\s*:\s*(.+?)\s*$"),
                        re.compile(r"(?is)^\s*devolva\s+exatamente\s+a\s+string\s*:\s*(.+?)\s*$"),
                        re.compile(
                            r"(?is)^\s*(?:responda|retorne|devolva|exiba|escreva)\s+(?:somente|apenas)\s*(?:com)?\s*:\s*(.+?)\s*$"
                        ),
                    ]
                    for pat in exact_pats:
                        m = pat.match(u)
                        if m:
                            ans = _strip_quotes(m.group(1))
                            if ans:
                                contract_tokens = tokenize_text(ans)
                                contract_kind = "exact"
                                contract_reason = "instruction_exact"
                                contract_meta["used"] = True
                                contract_meta["kind"] = contract_kind
                                contract_meta["parsed_args"] = {"text": ans}
                                contract_meta["reason"] = contract_reason
                            break

                    if (
                        not contract_tokens
                        and re.search(r"(?i)\bjson\b", u)
                        and (
                            re.search(r"(?i)\bapenas\b", u)
                            or re.search(r"(?i)\bsomente\b", u)
                            or re.search(r"(?i)sem\s+texto\s+extra", u)
                        )
                    ):
                        required_keys: List[str] = []
                        m_keys = re.search(r'(?i)chaves\s+"([^"]+)"\s+e\s+"([^"]+)"', u)
                        if m_keys:
                            required_keys = [str(m_keys.group(1)), str(m_keys.group(2))]

                        obj: Dict[str, Any] = {}
                        for m in re.finditer(
                            r'(?i)"(?P<key>[^"]+)"\s+deve\s+ser\s+a\s+string\s+"(?P<val>[^"]*)"',
                            u,
                        ):
                            obj[str(m.group("key"))] = str(m.group("val"))
                        for m in re.finditer(
                            r'(?i)"(?P<key>[^"]+)"\s+deve\s+ser\s+(?:o\s+n[úu]mero\s+)?(?P<val>-?\d+)\b',
                            u,
                        ):
                            obj[str(m.group("key"))] = int(m.group("val"))
                        for m in re.finditer(
                            r'(?i)"(?P<key>[^"]+)"\s+deve\s+ser\s+(?P<val>true|false)\b',
                            u,
                        ):
                            obj[str(m.group("key"))] = bool(str(m.group("val")).lower() == "true")

                        if required_keys and any(k not in obj for k in required_keys):
                            contract_meta["reason"] = "json_parse_incomplete"
                        elif not obj:
                            contract_meta["reason"] = "json_parse_empty"
                        else:
                            ans = json.dumps(
                                obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                            )
                            contract_tokens = tokenize_text(ans)
                            contract_kind = "json"
                            contract_reason = "instruction_json"
                            contract_meta["used"] = True
                            contract_meta["kind"] = contract_kind
                            contract_meta["parsed_args"] = {
                                "obj": obj,
                                "required_keys": required_keys,
                            }
                            contract_meta["reason"] = contract_reason

                    if (
                        not contract_tokens
                        and re.search(r"(?i)\bn[úu]mero\b", u)
                        and (
                            re.search(r"(?i)\bapenas\b", u)
                            or re.search(r"(?i)\bsomente\b", u)
                            or re.search(r"(?i)\bs[oó]\b", u)
                            or re.search(r"(?i)\bso\b", u)
                        )
                    ):
                        m = re.search(r"(-?\d+)\s*([+\-*/])\s*(-?\d+)", u)
                        if not m:
                            contract_meta["reason"] = "math_parse_missing"
                        else:
                            a = int(m.group(1))
                            op = str(m.group(2))
                            b = int(m.group(3))
                            if op == "+":
                                val = a + b
                            elif op == "-":
                                val = a - b
                            elif op == "*":
                                val = a * b
                            else:
                                if b == 0:
                                    contract_meta["reason"] = "math_div_zero"
                                    val = None
                                else:
                                    val = int(a / b)
                            if val is not None:
                                contract_tokens = tokenize_text(str(val))
                                contract_kind = "math_int"
                                contract_reason = "instruction_math"
                                contract_meta["used"] = True
                                contract_meta["kind"] = contract_kind
                                contract_meta["parsed_args"] = {"a": a, "op": op, "b": b, "value": int(val)}
                                contract_meta["reason"] = contract_reason

                    if not contract_tokens:
                        m = re.search(r"(?i)\bsenha\s+é\s+([A-Za-z0-9_-]+)", u)
                        if m:
                            pw = _strip_edge_punct(m.group(1))
                            key = _mem_key("senha")
                            ok = _mem_set(key, pw)
                            if ok:
                                contract_tokens = tokenize_text("OK")
                                contract_kind = "memory_store"
                                contract_reason = "store_password"
                                contract_meta["used"] = True
                                contract_meta["kind"] = contract_kind
                                contract_meta["parsed_args"] = {"fact_key": key, "value": pw}
                                contract_meta["reason"] = contract_reason

                    if not contract_tokens:
                        m = re.search(
                            r"(?i)\b([A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç]+)\s+nasceu\s+em\s+(\d{4})\b",
                            u,
                        )
                        if m:
                            ent = str(m.group(1))
                            year = str(m.group(2))
                            key = _mem_key("born_year", ent.lower())
                            ok = _mem_set(key, year)
                            if ok:
                                contract_tokens = tokenize_text("OK")
                                contract_kind = "memory_store"
                                contract_reason = "store_birth_year"
                                contract_meta["used"] = True
                                contract_meta["kind"] = contract_kind
                                contract_meta["parsed_args"] = {"fact_key": key, "value": year, "entity": ent}
                                contract_meta["reason"] = contract_reason

                    if not contract_tokens:
                        m = re.search(r"(?i)palavra[- ]c[óo]digo\s*:\s*([A-Za-z0-9_-]+)", u)
                        if m:
                            code = _strip_edge_punct(m.group(1))
                            key = _mem_key("code")
                            ok = _mem_set(key, code)
                            if ok:
                                contract_tokens = tokenize_text("OK")
                                contract_kind = "memory_store"
                                contract_reason = "store_code"
                                contract_meta["used"] = True
                                contract_meta["kind"] = contract_kind
                                contract_meta["parsed_args"] = {"fact_key": key, "value": code}
                                contract_meta["reason"] = contract_reason

                    if not contract_tokens:
                        m = (
                            re.search(r"(?i)qual\s+é\s+a\s+senha\b", u)
                            or re.search(r"(?i)qual\s+a\s+senha\b", u)
                            or re.search(r"(?i)repita\s+a\s+senha\b", u)
                            or re.search(r"(?i)diga\s+a\s+senha\b", u)
                        )
                        if m:
                            key = _mem_key("senha")
                            pw = _mem_get(key)
                            if pw is not None:
                                contract_tokens = tokenize_text(pw)
                                contract_kind = "memory_query"
                                contract_reason = "recall_password"
                                contract_meta["used"] = True
                                contract_meta["kind"] = contract_kind
                                contract_meta["parsed_args"] = {"fact_key": key, "value": pw}
                                contract_meta["reason"] = contract_reason
                            else:
                                contract_meta["reason"] = "missing_password"

                    if not contract_tokens:
                        m = re.search(
                            r"(?i)em\s+que\s+ano\s+([A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç]+)\s+nasceu\b",
                            u,
                        ) or re.search(
                            r"(?i)ano\s+de\s+nascimento\s+(?:de|da|do)\s+([A-Za-zÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç]+)\b",
                            u,
                        )
                        if m:
                            ent = _strip_edge_punct(str(m.group(1)))
                            key = _mem_key("born_year", ent.lower())
                            year = _mem_get(key)
                            if year is not None:
                                contract_tokens = tokenize_text(year)
                                contract_kind = "memory_query"
                                contract_reason = "recall_birth_year"
                                contract_meta["used"] = True
                                contract_meta["kind"] = contract_kind
                                contract_meta["parsed_args"] = {"fact_key": key, "value": year, "entity": ent}
                                contract_meta["reason"] = contract_reason
                            else:
                                contract_meta["reason"] = "missing_birth_year"

                    if not contract_tokens:
                        if re.search(r"(?i)repita\s+a\s+palavra[- ]c[óo]digo\b", u) or re.search(
                            r"(?i)repita\s+o\s+c[óo]digo\b", u
                        ):
                            key = _mem_key("code")
                            code = _mem_get(key)
                            if code is not None:
                                contract_tokens = tokenize_text(code)
                                contract_kind = "memory_query"
                                contract_reason = "recall_code"
                                contract_meta["used"] = True
                                contract_meta["kind"] = contract_kind
                                contract_meta["parsed_args"] = {"fact_key": key, "value": code}
                                contract_meta["reason"] = contract_reason
                            else:
                                contract_meta["reason"] = "missing_code"

                    if not contract_tokens:
                        m = re.search(r"(?i)\binclua\b", u)
                        if m:
                            after = u[m.end() :]
                            m_tok = re.search(r"\b([A-Z0-9_-]{2,})\b", after)
                            if m_tok:
                                tok = _strip_edge_punct(m_tok.group(1))
                                if tok and re.fullmatch(r"[A-Z0-9_-]{2,}", tok) is not None:
                                    contract_tokens = tokenize_text(tok)
                                    contract_kind = "contains_token"
                                    contract_reason = "include_token"
                                    contract_meta["used"] = True
                                    contract_meta["kind"] = contract_kind
                                    contract_meta["parsed_args"] = {"token": tok}
                                    contract_meta["reason"] = contract_reason
                                elif tok:
                                    contract_meta["reason"] = "include_token_rejected"
                            else:
                                contract_meta["reason"] = "include_token_missing"

        contract_active = bool(contract_tokens)

        def penalty_view(tokens: List[str], gen: List[str]) -> Dict[str, Any]:
            filtered = [x for x in tokens if x not in {"<BOS>"}]
            recent = filtered[-self.config.repetition_recent :]
            return {
                "recent_tokens": recent,
                "history_ngrams": set(history_ngram_set),
                "gen_tokens": list(gen),
                "mode": mode_state,
            }

        def _top1_token(cands: Dict[str, Candidate]) -> Optional[str]:
            if not cands:
                return None
            return min(cands.values(), key=lambda c: (-c.score, c.token, c.source_act)).token

        def _apply_rewrite_rules(
            cands: Dict[str, Candidate],
            *,
            context: Sequence[str],
            penalties: Dict[str, Any],
        ) -> Tuple[Dict[str, Candidate], List[str]]:
            rr_hits: List[str] = []
            top_tok = _top1_token(cands)
            for rr in self._rewrite_rules:
                before_tok = top_tok
                before_score = cands.get(before_tok).score if before_tok in cands else None
                st = self.vm.run(
                    rr,
                    context=context,
                    tables=self._tables,
                    vocab=self.vocab(),
                    initial_candidates=cands,
                    penalties=penalties,
                    mode="emit_only",
                )
                cands = st.candidates
                after_tok = _top1_token(cands)
                after_score = cands.get(before_tok).score if before_tok in cands else None
                if after_tok is None:
                    after_tok = before_tok
                if (
                    before_tok is not None
                    and after_tok is not None
                    and (
                        after_tok != before_tok
                        or (before_score is None)
                        or (after_score is None)
                        or abs(float(after_score) - float(before_score)) > 1e-12
                    )
                ):
                    rr_hits.append(str(rr.id))
                top_tok = after_tok
            return cands, rr_hits

        def _select_next(
            cands: Dict[str, Candidate], *, context: Sequence[str], penalties: Dict[str, Any]
        ) -> Optional[str]:
            if not cands:
                return None

            nxt0: Optional[str] = None
            if mode == "greedy" and self._selector is not None:
                st = self.vm.run(
                    self._selector,
                    context=context,
                    tables=self._tables,
                    vocab=self.vocab(),
                    initial_candidates=cands,
                    penalties=penalties,
                    mode="select",
                )
                nxt0 = st.selected

            if nxt0 is None:
                ordered = sorted(cands.values(), key=lambda c: (-c.score, c.token, c.source_act))
                if mode == "greedy":
                    nxt0 = ordered[0].token
                else:
                    mx = ordered[0].score
                    exps = [math.exp(min(60.0, c.score - mx)) for c in ordered]
                    s = sum(exps)
                    r = self.rng.random() * s
                    acc = 0.0
                    nxt0 = ordered[-1].token
                    for c, e in zip(ordered, exps):
                        acc += e
                        if acc >= r:
                            nxt0 = c.token
                            break
            return nxt0

        def _non_ws(seq: Sequence[str]) -> List[str]:
            return [t for t in seq if t not in {"<BOS>"} and (not is_space(t))]

        # Decoder-level anti-loop / fluency controls (deterministic; must not affect contracts).
        NO_REPEAT_NGRAM = 3
        REP_WINDOW = 64
        REP_ALPHA = 0.45
        BLOCK_PENALTY = 1e6
        PROMPT_JITTER_ALPHA = 0.9
        EARLY_WS_PENALTY = 4.0
        MULTI_WS_PENALTY = 10.0
        ROUTER_DECODER_FLUENCY_ID = "antiloop_v40"

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).digest()

        def _prompt_jitter(tok: str) -> float:
            try:
                h = hashlib.sha256(prompt_hash + tok.encode("utf-8")).digest()
                v = int.from_bytes(h[:8], "big")
                return (v / float(2**64)) - 0.5
            except Exception:
                return 0.0

        def _select_next_fluency(
            cands: Dict[str, Candidate], *, context: Sequence[str], penalties: Dict[str, Any]
        ) -> Optional[str]:
            if not cands:
                return None

            # Source-lock: keep selection within the same source_act that would have won
            # under the unmodified candidate set. This preserves gate invariants while we
            # tune decoder-level fluency.
            base_tok = _select_next(cands, context=context, penalties=penalties)
            base_src = cands.get(base_tok).source_act if base_tok in cands else None

            work = cands
            if base_src is not None:
                sub = {tok: cand for tok, cand in cands.items() if cand.source_act == base_src}
                if sub:
                    work = sub

            if REP_WINDOW > 0 and REP_ALPHA > 0.0:
                recent = turn_non_ws[-REP_WINDOW:]
                if recent:
                    counts: Dict[str, int] = {}
                    for t in recent:
                        counts[t] = int(counts.get(t, 0)) + 1
                    for tok, cand in work.items():
                        if tok in {"<EOS>"} or is_space(tok):
                            continue
                        cnt = int(counts.get(tok, 0))
                        if cnt > 0:
                            cand.score -= float(REP_ALPHA) * float(cnt)

            if len(turn_non_ws) < 1 and EARLY_WS_PENALTY > 0.0:
                for tok, cand in work.items():
                    if tok == "<EOS>":
                        continue
                    if is_space(tok):
                        cand.score -= float(EARLY_WS_PENALTY)

            if MULTI_WS_PENALTY > 0.0:
                for tok, cand in work.items():
                    if tok == "<EOS>":
                        continue
                    if is_space(tok) and len(tok) > 1:
                        cand.score -= float(MULTI_WS_PENALTY) * float(len(tok) - 1)

            if PROMPT_JITTER_ALPHA > 0.0:
                for tok, cand in work.items():
                    if tok == "<EOS>":
                        continue
                    if is_space(tok):
                        continue
                    cand.score += float(PROMPT_JITTER_ALPHA) * float(_prompt_jitter(tok))

            if NO_REPEAT_NGRAM >= 2 and len(turn_non_ws) >= (NO_REPEAT_NGRAM - 1) and seen_turn_ngrams:
                prefix = tuple(turn_non_ws[-(NO_REPEAT_NGRAM - 1) :])
                eligible = [
                    tok
                    for tok in work.keys()
                    if tok not in {"<EOS>"} and (not is_space(tok))
                ]
                would_block = [
                    tok for tok in eligible if tuple(list(prefix) + [tok]) in seen_turn_ngrams
                ]
                # Dead-end guard: if everything would be blocked, relax (do not block this step).
                if eligible and len(would_block) < len(eligible):
                    for tok in would_block:
                        work[tok].score -= float(BLOCK_PENALTY)

            return _select_next(work, context=context, penalties=penalties)

        seen_turn_ngrams: set = set()
        turn_non_ws: List[str] = []

        for i in range(int(max_new_tokens)):
            ck = ctx_key(context)
            penalties = penalty_view(out_tokens + gen_tokens, gen_tokens)

            if contract_active and contract_pos < len(contract_tokens):
                nxt = str(contract_tokens[contract_pos])
                contract_pos += 1

                trace_context_keys.append(ck)
                trace_executed_predictor_ids.append([])
                trace_rewrite_rule_hit_ids.append([])
                trace_rewrite_rules_changed_count.append(0)

                trace_predictor_iterated.append(0)
                trace_predictor_matched.append(0)
                trace_predictor_emitted.append(0)
                trace_candidates_pre.append(1)
                trace_candidates_post.append(1)

                trace_router_live_used.append(0)
                trace_router_live_fallback.append(0)
                trace_router_live_fallback_reason.append("")
                trace_router_live_allowed_predictor_ids.append([])
                trace_router_live_predictors_iterated.append(0)
                trace_router_live_predictors_matched.append(0)
                trace_router_live_predictors_emitted.append(0)
                trace_baseline_predictors_iterated.append(0)
                trace_baseline_predictors_matched.append(0)
                trace_baseline_predictors_emitted.append(0)
                trace_router_live_mismatch.append(0)
                trace_router_live_debug_baseline_token.append("")
                trace_router_live_debug_gate_token.append("")

                trace_instruction_contract_used.append(1)
                trace_instruction_contract_kind.append(str(contract_kind))
                trace_instruction_contract_reason.append(str(contract_reason))

                trace_selected_act_ids.append(str(contract_act_id or "__contract__"))
                trace_selected_tokens.append(str(nxt))

                gen_tokens.append(nxt)
                filtered = [x for x in (out_tokens + gen_tokens) if x not in {"<BOS>"}]
                n = self.config.cycle_ngram
                if len(filtered) >= n:
                    ng = tuple(filtered[-n:])
                    history_ngrams.append(ng)
                    history_ngram_set.add(ng)
                    if len(history_ngrams) > self.config.cycle_history:
                        old = history_ngrams.pop(0)
                        if old not in history_ngrams:
                            history_ngram_set.discard(old)

                context.append(nxt)
                context = context[-(self.config.max_order - 1) :]

                if contract_pos >= len(contract_tokens):
                    break
                continue

            router_live_enabled = bool(self.config.router_live_enabled) and not bool(
                self.config.disable_macro_router
            )
            router_debug = bool(self.config.router_live_debug_compare)

            router_used = False
            router_fallback = False
            router_fallback_reason = ""
            allowed_predictor_ids: List[str] = []
            gated_predictors: Optional[List[Act]] = None

            if router_live_enabled:
                compat = ""
                try:
                    if self._macro_router is not None and isinstance(self._macro_router.evidence, dict):
                        compat = str(self._macro_router.evidence.get("decoder_fluency_id") or "")
                except Exception:
                    compat = ""

                # If the router was built under a different decoder regime, do not rely on
                # its top-k predictors: keep invariants by allowing all predictors (no savings).
                if compat != ROUTER_DECODER_FLUENCY_ID:
                    allowed_predictor_ids = [str(a.id) for a in self._predictors]
                    gated_predictors = list(self._predictors)
                else:
                    table = self._macro_router_table
                    if table is None or not isinstance(table, dict):
                        router_fallback = True
                        router_fallback_reason = "router_missing"
                    else:
                        ctx_sig = f"{mode_state}{SEP}{ck}"
                        entry = table.get(ctx_sig)
                        if isinstance(entry, dict):
                            # Safety: if a ctx_sig exists but has tiny evidence, treat as uncovered.
                            tot = int(entry.get("total", 0) or 0)
                            if tot > 0 and tot < 5:
                                router_fallback = True
                                router_fallback_reason = "low_total"
                            if router_fallback:
                                entry = None
                        if isinstance(entry, dict):
                            preds = entry.get("predictors") or []
                            if isinstance(preds, list):
                                allowed_predictor_ids = [
                                    str(x) for x in preds if isinstance(x, str) and x
                                ]
                    if allowed_predictor_ids:
                        present = [
                            pid for pid in allowed_predictor_ids if pid in self._predictor_by_id
                        ]
                        if present:
                            present.sort(key=lambda pid: self._predictor_order.get(pid, 0))
                            gated_predictors = [self._predictor_by_id[pid] for pid in present]
                        else:
                            if not router_fallback:
                                router_fallback = True
                                router_fallback_reason = "allowed_missing"
                    else:
                        if not router_fallback:
                            router_fallback = True
                            router_fallback_reason = "missing_ctx"

            emit_gate: Optional[Dict[str, Any]] = None
            cand_gate: Optional[Dict[str, Candidate]] = None
            if gated_predictors is not None:
                emit_gate = {}
                cand_gate = self._emit_candidates(
                    context=context,
                    penalties=penalties,
                    predictors=gated_predictors,
                    trace=emit_gate,
                )
                if int(emit_gate.get("predictor_emitted", 0) or 0) <= 0:
                    router_fallback = True
                    router_fallback_reason = "gate_empty_emit"
                    emit_gate = None
                    cand_gate = None

            emit_base: Optional[Dict[str, Any]] = None
            cand_base: Optional[Dict[str, Candidate]] = None
            if router_debug or (router_live_enabled and cand_gate is None):
                emit_base = {}
                cand_base = self._emit_candidates(
                    context=context, penalties=penalties, trace=emit_base
                )

            mismatch = False
            baseline_tok = ""
            gate_tok = ""
            if router_debug and cand_gate is not None and cand_base is not None:
                base_after, _ = _apply_rewrite_rules(
                    cand_base, context=context, penalties=penalties
                )
                gate_after, _ = _apply_rewrite_rules(
                    cand_gate, context=context, penalties=penalties
                )
                if i < self.config.min_new_tokens_before_eos:
                    if "<EOS>" in base_after:
                        base_after["<EOS>"].score -= 1e6
                    if "<EOS>" in gate_after:
                        gate_after["<EOS>"].score -= 1e6

                b_nxt = _select_next_fluency(base_after, context=context, penalties=penalties)
                g_nxt = _select_next_fluency(gate_after, context=context, penalties=penalties)
                baseline_tok = str(b_nxt or "")
                gate_tok = str(g_nxt or "")
                mismatch = bool(b_nxt != g_nxt)
                if mismatch:
                    router_fallback = True
                    router_fallback_reason = "mismatch"
                    emit_gate = None
                    cand_gate = None

            emit_trace: Dict[str, Any] = {}
            candidates: Dict[str, Candidate] = {}
            if cand_gate is not None and emit_gate is not None and not router_fallback:
                router_used = True
                emit_trace = emit_gate
                candidates = cand_gate
            else:
                router_used = False
                if router_live_enabled and not router_fallback:
                    router_fallback = True
                    router_fallback_reason = "no_gate"
                if cand_base is None or emit_base is None:
                    emit_base = {}
                    cand_base = self._emit_candidates(
                        context=context, penalties=penalties, trace=emit_base
                    )
                emit_trace = emit_base
                candidates = cand_base

            exec_pred = emit_trace.get("executed_predictor_ids") or []
            exec_pred_ids: List[str] = []
            if isinstance(exec_pred, list):
                exec_pred_ids = [str(x) for x in exec_pred if isinstance(x, str)]

            pred_iter = int(emit_trace.get("predictor_iterated", 0) or 0)
            pred_mat = int(emit_trace.get("predictor_matched", 0) or 0)
            pred_emit = int(emit_trace.get("predictor_emitted", 0) or 0)
            cand_pre = int(emit_trace.get("candidates_pre_rewrite", len(candidates)) or 0)

            base_iter = 0
            base_mat = 0
            base_emit = 0
            if isinstance(emit_base, dict):
                base_iter = int(emit_base.get("predictor_iterated", 0) or 0)
                base_mat = int(emit_base.get("predictor_matched", 0) or 0)
                base_emit = int(emit_base.get("predictor_emitted", 0) or 0)

            candidates, rr_hits = _apply_rewrite_rules(
                candidates, context=context, penalties=penalties
            )

            # EOS guardrail
            if (not contract_active) and i < self.config.min_new_tokens_before_eos and "<EOS>" in candidates:
                candidates["<EOS>"].score -= 1e6

            if not candidates:
                break

            # Anti-loop (decoder-only): apply after rewrite rules + EOS guardrail, before selection.
            # Must not run when a contract is active (contracts must be exact).
            if not contract_active:
                nxt = _select_next_fluency(candidates, context=context, penalties=penalties)
            else:
                nxt = _select_next(candidates, context=context, penalties=penalties)
            if nxt is None or nxt == "<EOS>":
                break

            src_act = candidates.get(nxt).source_act if nxt in candidates else "__unknown__"

            # Align trace arrays exactly to emitted tokens (no EOS attempt row).
            trace_context_keys.append(ck)
            trace_executed_predictor_ids.append(exec_pred_ids)
            trace_rewrite_rule_hit_ids.append(rr_hits)
            trace_rewrite_rules_changed_count.append(int(len(rr_hits)))

            trace_predictor_iterated.append(pred_iter)
            trace_predictor_matched.append(pred_mat)
            trace_predictor_emitted.append(pred_emit)
            trace_candidates_pre.append(cand_pre)
            trace_candidates_post.append(int(len(candidates)))

            trace_router_live_used.append(1 if router_used else 0)
            trace_router_live_fallback.append(1 if router_fallback else 0)
            trace_router_live_fallback_reason.append(str(router_fallback_reason))
            trace_router_live_allowed_predictor_ids.append(list(allowed_predictor_ids))
            trace_router_live_predictors_iterated.append(pred_iter)
            trace_router_live_predictors_matched.append(pred_mat)
            trace_router_live_predictors_emitted.append(pred_emit)
            trace_baseline_predictors_iterated.append(base_iter if router_debug else 0)
            trace_baseline_predictors_matched.append(base_mat if router_debug else 0)
            trace_baseline_predictors_emitted.append(base_emit if router_debug else 0)
            trace_router_live_mismatch.append(1 if mismatch else 0)
            trace_router_live_debug_baseline_token.append(baseline_tok)
            trace_router_live_debug_gate_token.append(gate_tok)

            trace_instruction_contract_used.append(0)
            trace_instruction_contract_kind.append("")
            trace_instruction_contract_reason.append("")

            trace_selected_act_ids.append(str(src_act))
            trace_selected_tokens.append(str(nxt))

            gen_tokens.append(nxt)
            if nxt not in {"<EOS>", "<BOS>"} and (not is_space(nxt)):
                turn_non_ws.append(nxt)
                if NO_REPEAT_NGRAM >= 2 and len(turn_non_ws) >= NO_REPEAT_NGRAM:
                    seen_turn_ngrams.add(tuple(turn_non_ws[-NO_REPEAT_NGRAM:]))
            # Update history sets for cycle penalties (filtered, space-free).
            filtered = [x for x in (out_tokens + gen_tokens) if x not in {"<BOS>"}]
            n = self.config.cycle_ngram
            if len(filtered) >= n:
                ng = tuple(filtered[-n:])
                history_ngrams.append(ng)
                history_ngram_set.add(ng)
                if len(history_ngrams) > self.config.cycle_history:
                    old = history_ngrams.pop(0)
                    # best-effort remove; duplicates may exist
                    if old not in history_ngrams:
                        history_ngram_set.discard(old)

            context.append(nxt)
            context = context[-(self.config.max_order - 1) :]

        # Turn-level "subgraph" trace (unique, ordered): minimal causal footprint per turn.
        # This is observability-only and must not affect generation.
        exec_pred_unique: List[str] = []
        rr_hit_unique: List[str] = []
        try:
            exec_set = {str(pid) for ids in trace_executed_predictor_ids for pid in (ids or []) if pid}
            exec_pred_unique = sorted(
                exec_set, key=lambda pid: (int(self._predictor_order.get(pid, 10**9)), pid)
            )
        except Exception:
            exec_pred_unique = []
        try:
            rr_set = {str(rid) for ids in trace_rewrite_rule_hit_ids for rid in (ids or []) if rid}
            rr_order = {str(a.id): i for i, a in enumerate(self._rewrite_rules)}
            rr_hit_unique = sorted(rr_set, key=lambda rid: (int(rr_order.get(rid, 10**9)), rid))
        except Exception:
            rr_hit_unique = []

        return {
            "prompt": prompt,
            "text": detokenize(out_tokens + gen_tokens),
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "all_tokens": out_tokens + gen_tokens,
            "trace": {
                "active_set_size": int(active_set_size),
                "predictor_total": int(predictor_total),
                "rewrite_rules_total": int(rewrite_rules_total),
                "selector_id": selector_id,
                "context_keys": trace_context_keys,
                "executed_predictor_ids": trace_executed_predictor_ids,
                "rewrite_rule_hit_ids": trace_rewrite_rule_hit_ids,
                "rewrite_rules_changed_count": trace_rewrite_rules_changed_count,
                "selected_tokens": trace_selected_tokens,
                "selected_source_act_ids": trace_selected_act_ids,
                "selected_act_ids": trace_selected_act_ids,
                "candidates_pre_rewrite": trace_candidates_pre,
                "candidates_post_rewrite": trace_candidates_post,
                "predictor_iterated": trace_predictor_iterated,
                "predictor_matched": trace_predictor_matched,
                "predictor_emitted": trace_predictor_emitted,
                "router_live_used": trace_router_live_used,
                "router_live_fallback": trace_router_live_fallback,
                "router_live_fallback_reason": trace_router_live_fallback_reason,
                "router_live_allowed_predictor_ids": trace_router_live_allowed_predictor_ids,
                "router_live_predictors_iterated": trace_router_live_predictors_iterated,
                "router_live_predictors_matched": trace_router_live_predictors_matched,
                "router_live_predictors_emitted": trace_router_live_predictors_emitted,
                "baseline_predictors_iterated": trace_baseline_predictors_iterated,
                "baseline_predictors_matched": trace_baseline_predictors_matched,
                "baseline_predictors_emitted": trace_baseline_predictors_emitted,
                "router_live_mismatch": trace_router_live_mismatch,
                "router_live_debug_baseline_token": trace_router_live_debug_baseline_token,
                "router_live_debug_gate_token": trace_router_live_debug_gate_token,
                "instruction_contract": contract_meta,
                "instruction_contract_used": trace_instruction_contract_used,
                "instruction_contract_kind": trace_instruction_contract_kind,
                "instruction_contract_reason": trace_instruction_contract_reason,
                "subgraph": {
                    "executed_predictor_act_ids": exec_pred_unique,
                    "rewrite_rule_hit_ids": rr_hit_unique,
                },
            },
            "mode": mode_state,
            "mode_act_id": mode_act_id,
            "mode_source": mode_source,
            "user_sig": mode_user_sig,
            "mode_policy_action": mode_policy_action,
            "policy_coverage": policy_coverage,
        }
