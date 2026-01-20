from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import Act, Instruction
from .suite import non_ws_tokens


@dataclass
class Candidate:
    token: str
    score: float
    source_act: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VMState:
    candidates: Dict[str, Candidate] = field(default_factory=dict)
    matched: bool = True
    selected: Optional[str] = None
    mode: Optional[str] = None
    proposals: List[Dict[str, Any]] = field(default_factory=list)


class AtoLangVM:
    def __init__(self, *, rng):
        self.rng = rng

    def run(
        self,
        act: Act,
        *,
        context: Sequence[str],
        tables: Dict[str, Dict[str, Dict[str, int]]],
        vocab: Sequence[str],
        initial_candidates: Optional[Dict[str, Candidate]] = None,
        penalties: Optional[Dict[str, Any]] = None,
        mode: str = "emit_only",
    ) -> VMState:
        state = VMState()
        if initial_candidates:
            # Copy to avoid aliasing across runs.
            state.candidates = {
                tok: Candidate(
                    token=c.token,
                    score=float(c.score),
                    source_act=c.source_act,
                    meta=dict(c.meta),
                )
                for tok, c in initial_candidates.items()
            }
        penalties = penalties or {}

        for ins in act.program:
            op = ins.op
            args = ins.args

            if op == "MATCH_NGRAM":
                n = int(args["n"])
                if n <= 0:
                    state.matched = False
                else:
                    need = n - 1
                    state.matched = len(context) >= need
                if not state.matched:
                    break

            elif op == "EMIT_CANDIDATES_TOPK":
                table_id = str(args["table_id"])
                k = int(args.get("k", 32))
                n = int(args.get("n", act.match.get("n", 1)))
                ctx = context[-(n - 1) :] if n > 1 else []
                ctx_key = self._ctx_key(ctx)
                table = tables.get(table_id, {})
                nxt_counts = table.get(ctx_key)
                if not nxt_counts:
                    continue
                items = sorted(nxt_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
                total = sum(nxt_counts.values())
                for tok, cnt in items:
                    if tok not in state.candidates:
                        state.candidates[tok] = Candidate(
                            token=tok,
                            score=0.0,
                            source_act=act.id,
                            meta={"count": cnt, "total": total, "table_id": table_id, "n": n},
                        )

            elif op == "EMIT_EOS":
                tok = "<EOS>"
                if tok not in state.candidates:
                    state.candidates[tok] = Candidate(
                        token=tok, score=float(args.get("bias", 0.0)), source_act=act.id
                    )

            elif op == "SCORE_CANDIDATES":
                method = str(args.get("method", "logprob"))
                alpha = float(args.get("alpha", 0.5))
                if method not in {"logprob", "mdl_proxy"}:
                    method = "logprob"
                V = max(1, len(vocab))
                for cand in state.candidates.values():
                    if cand.token == "<EOS>":
                        continue
                    cnt = float(cand.meta.get("count", 0.0))
                    total = float(cand.meta.get("total", 0.0))
                    p = (cnt + alpha) / (total + alpha * V)
                    cand.score += math.log(p + 1e-12)

            elif op == "APPLY_PENALTY":
                kind = str(args.get("kind", "repetition"))
                strength = float(args.get("strength", 1.0))
                if kind == "repetition":
                    recent = penalties.get("recent_tokens", [])
                    for cand in state.candidates.values():
                        if cand.token in recent:
                            cand.score -= strength
                elif kind == "ngram_cycle":
                    n = int(args.get("n", 3))
                    recent = penalties.get("recent_tokens", [])
                    history = penalties.get("history_ngrams", set())
                    for cand in state.candidates.values():
                        if cand.token == "<EOS>":
                            continue
                        ngram = tuple((list(recent) + [cand.token])[-n:])
                        if ngram in history:
                            cand.score -= strength
                elif kind == "anti_template_v2":
                    gen_tokens = penalties.get("gen_tokens", [])
                    if not isinstance(gen_tokens, list):
                        gen_tokens = []
                    gen_tokens_str = [t for t in gen_tokens if isinstance(t, str)]
                    gen_non_ws = [t.lower() for t in non_ws_tokens(gen_tokens_str)]
                    pos = int(len(gen_non_ws))

                    mode_name = penalties.get("mode", "default")
                    if not isinstance(mode_name, str) or not mode_name:
                        mode_name = "default"

                    cfg: Dict[str, Any] = act.evidence
                    modes = act.evidence.get("modes")
                    if isinstance(modes, dict):
                        cand = modes.get(mode_name)
                        if not isinstance(cand, dict):
                            cand = modes.get("default")
                        if isinstance(cand, dict):
                            cfg = cand

                    prefix_window = int(cfg.get("prefix_window", act.evidence.get("prefix_window", 32)) or 32)
                    if pos >= prefix_window:
                        continue

                    ban_tokens = cfg.get("ban_tokens", act.evidence.get("ban_tokens", []))
                    ban_tokens_l = {str(t).lower() for t in ban_tokens if t is not None}
                    w_token = float(cfg.get("w_ban_token", act.evidence.get("w_ban_token", 8.0)))

                    ban_prefix_seqs = cfg.get("ban_prefix_seqs", act.evidence.get("ban_prefix_seqs", []))
                    norm_seqs: List[List[str]] = []
                    if isinstance(ban_prefix_seqs, list):
                        for seq in ban_prefix_seqs:
                            if not isinstance(seq, list):
                                continue
                            toks = [str(x).lower() for x in seq if x is not None]
                            if toks:
                                norm_seqs.append(toks)
                    w_seq = float(cfg.get("w_ban_prefix_seq", act.evidence.get("w_ban_prefix_seq", 4.0)))

                    # v0.2.6: suffix n-gram blocking anywhere within an early window (non-ws, lowercased).
                    ngram_prefix_window = int(
                        cfg.get(
                            "ngram_prefix_window", act.evidence.get("ngram_prefix_window", 32)
                        )
                        or 32
                    )
                    ban_ngram_seqs = cfg.get("ban_ngram_seqs", act.evidence.get("ban_ngram_seqs", []))
                    norm_ngram_seqs: List[List[str]] = []
                    if isinstance(ban_ngram_seqs, list):
                        for seq in ban_ngram_seqs:
                            if not isinstance(seq, list):
                                continue
                            toks = [str(x).lower() for x in seq if x is not None]
                            if toks:
                                norm_ngram_seqs.append(toks)
                    w_ng = float(cfg.get("w_ban_ngram_seq", act.evidence.get("w_ban_ngram_seq", w_seq)))
                    apply_ng = bool(pos < ngram_prefix_window)

                    for cand in state.candidates.values():
                        tok_l = str(cand.token).lower()
                        if tok_l in ban_tokens_l:
                            cand.score -= w_token
                        if norm_seqs:
                            for seq in norm_seqs:
                                if pos < len(seq) and gen_non_ws == seq[:pos] and tok_l == seq[pos]:
                                    cand.score -= w_seq
                        if apply_ng and norm_ngram_seqs:
                            for seq in norm_ngram_seqs:
                                if not seq:
                                    continue
                                if len(seq) == 1:
                                    if tok_l == seq[0]:
                                        cand.score -= w_ng
                                    continue
                                need = len(seq) - 1
                                if (
                                    len(gen_non_ws) >= need
                                    and gen_non_ws[-need:] == seq[:-1]
                                    and tok_l == seq[-1]
                                ):
                                    cand.score -= w_ng

            elif op == "SELECT_NEXT":
                if mode != "select":
                    continue
                mode_sel = str(args.get("mode", "greedy"))
                if not state.candidates:
                    state.selected = None
                    continue
                ordered = sorted(
                    state.candidates.values(), key=lambda c: (-c.score, c.token, c.source_act)
                )
                if mode_sel == "greedy":
                    state.selected = ordered[0].token
                else:
                    # sample from softmax(scores)
                    mx = ordered[0].score
                    exps = [math.exp(min(60.0, c.score - mx)) for c in ordered]
                    s = sum(exps)
                    r = self.rng.random() * s
                    acc = 0.0
                    choice = ordered[-1].token
                    for c, e in zip(ordered, exps):
                        acc += e
                        if acc >= r:
                            choice = c.token
                            break
                    state.selected = choice

            elif op == "SET_MODE":
                state.mode = str(args.get("mode", ""))

            elif op == "UPDATE_COUNT":
                # training-side opcode; handled by engine for speed
                continue

            elif op in {"PROPOSE_NEW_ACT", "MERGE_ACTS", "PRUNE_ACT"}:
                state.proposals.append({"op": op, **args})

            else:
                raise ValueError(f"Unknown opcode: {op}")

        return state

    @staticmethod
    def _ctx_key(ctx: Sequence[str]) -> str:
        # Deterministic, reversible key for small contexts.
        return "\u241f".join(ctx)
