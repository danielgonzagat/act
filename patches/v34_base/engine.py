from __future__ import annotations

import math
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import Act, Instruction
from .atolang import AtoLangVM, Candidate
from .metrics import detokenize, is_space, tokenize_text
from .suite import user_signature_from_prompt, user_signatures_from_prompt

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
    # Proof mode: compute baseline + gate per token, count mismatches, and fall back to baseline.
    router_live_debug_compare: bool = False


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

        for i in range(int(max_new_tokens)):
            ck = ctx_key(context)
            penalties = penalty_view(out_tokens + gen_tokens, gen_tokens)

            router_live_enabled = bool(self.config.router_live_enabled)
            router_debug = bool(self.config.router_live_debug_compare)

            router_used = False
            router_fallback = False
            router_fallback_reason = ""
            allowed_predictor_ids: List[str] = []
            gated_predictors: Optional[List[Act]] = None

            if router_live_enabled:
                table = self._macro_router_table
                if table is None or not isinstance(table, dict):
                    router_fallback = True
                    router_fallback_reason = "router_missing"
                else:
                    ctx_sig = f"{mode_state}{SEP}{ck}"
                    entry = table.get(ctx_sig)
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
                            router_fallback = True
                            router_fallback_reason = "allowed_missing"
                    else:
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
                b_nxt = _select_next(base_after, context=context, penalties=penalties)
                g_nxt = _select_next(gate_after, context=context, penalties=penalties)
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
            if i < self.config.min_new_tokens_before_eos and "<EOS>" in candidates:
                candidates["<EOS>"].score -= 1e6

            if not candidates:
                break

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

            trace_selected_act_ids.append(str(src_act))
            trace_selected_tokens.append(str(nxt))

            gen_tokens.append(nxt)
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
            },
            "mode": mode_state,
            "mode_act_id": mode_act_id,
            "mode_source": mode_source,
            "user_sig": mode_user_sig,
            "mode_policy_action": mode_policy_action,
            "policy_coverage": policy_coverage,
        }
