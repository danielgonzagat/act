from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .metrics import distinct_n, loop_rate, repeat_ngram_rate, tokenize_text


CHAT_DIALOGUES_20X3: Tuple[Tuple[str, str, str], ...] = (
    ("Oi, quem é você?", "Explique em uma frase.", "Agora em inglês."),
    ("Hello, who are you?", "Explain in one sentence.", "Now in Portuguese."),
    ("Liste 3 regras do sistema.", "Por que você não usa gradiente?", "O que é um ato?"),
    ("What is MDL?", "How does MDL relate to compression?", "Give a tiny example."),
    ("Crie um mini diálogo sobre determinismo.", "Adicione mais um turno.", "Finalize."),
    ("Continue: the quick brown fox", "Agora em português.", "Agora com pontuação!"),
    ("Continue: um dois três", "Agora em inglês.", "Agora com números 123 e 42."),
    ("Defina 'estado explícito'.", "Por que é versionável?", "Dê um exemplo."),
    (
        "Explique a diferença entre runtime e learning aqui.",
        "Dê um exemplo de patch.",
        "E um exemplo de evidência.",
    ),
    ("Faça uma lista numerada com 5 itens.", "Agora remova o item 3.", "Agora ordene ao contrário."),
    ("Escreva uma frase curta e amigável.", "Agora mais formal.", "Agora mais técnica."),
    ("O que acontece se a repetição explode?", "Como você previne loops?", "Que métrica detecta isso?"),
    ("Me dê um exemplo de ato predictor.", "Me dê um exemplo de rewrite_rule.", "Me dê um exemplo de selector."),
    ("Explique n-grams em 1 linha.", "Agora em 2 linhas.", "Agora com uma analogia."),
    ("Pergunta: capital de Portugal?", "Pergunta: capital da França?", "Pergunta: capital do Brasil?"),
    ("Resuma: atos -> candidatos -> seleção -> atualização.", "Agora mais curto.", "Agora em inglês."),
    ("Escreva um micro-poema sobre atos.", "Sem repetir palavras.", "Agora com rima."),
    ("Diga algo sobre 'compressão'.", "Diga algo sobre 'fluência'.", "Conecte os dois."),
    ("Escreva uma frase com pontuação: ,.;:?!", "Agora sem pontuação.", "Agora com emoji textual :)"),
    (
        "Finalize com uma autodescrição concisa.",
        "Inclua uma restrição que você segue.",
        "Inclua uma segunda restrição.",
    ),
)


_WS_RE = re.compile(r"\s+", flags=re.UNICODE)
_LAST_USER_RE = re.compile(r"(?:^|\n)User:\s*(.*?)\nSystem:", flags=re.UNICODE | re.DOTALL)
_ALL_USERS_RE = re.compile(r"(?:^|\n)User:\s*([^\n]*)\nSystem:", flags=re.UNICODE)


def reply_signature(text: str) -> str:
    s = text.strip()
    s = _WS_RE.sub(" ", s)
    s = s.lower()
    return s[:256]


def build_chat_prompt(history: Sequence[Dict[str, str]]) -> str:
    parts: List[str] = []
    for h in history:
        parts.append(f"User: {h['user']}\nSystem: {h.get('system','')}")
    return "\n".join(parts).rstrip() + "\n"


def _non_ws(tokens: Sequence[str]) -> List[str]:
    return [t for t in tokens if t and (not t.isspace()) and t != "<BOS>"]


def non_ws_tokens(tokens: Sequence[str]) -> List[str]:
    return _non_ws(tokens)


def last_user_text_from_prompt(prompt: str) -> str:
    matches = list(_LAST_USER_RE.finditer(prompt))
    if not matches:
        return ""
    return matches[-1].group(1)


def user_signature(text: str, *, k: int = 2) -> str:
    toks = _non_ws(tokenize_text(text))
    toks = toks[: max(0, int(k))]
    return " ".join(t.lower() for t in toks)


def user_signature_from_prompt(prompt: str, *, k: int = 2) -> str:
    return user_signature(last_user_text_from_prompt(prompt), k=k)


def user_signatures_from_prompt(prompt: str, *, k: int = 2) -> List[str]:
    out: List[str] = []
    for m in _ALL_USERS_RE.finditer(prompt):
        sig = user_signature(m.group(1), k=k)
        if sig:
            out.append(sig)
    return out


def prefix_k_signature(tokens: Sequence[str], *, k: int = 8) -> str:
    toks = _non_ws(tokens)[: max(0, int(k))]
    return " ".join(t.lower() for t in toks)


def prefix_k_dup_rate(prefix_sigs: Sequence[str]) -> float:
    total = len(prefix_sigs)
    if total <= 0:
        return 0.0
    uniq = len(set(prefix_sigs))
    return 1.0 - (uniq / total)


def template_ngram_dup_rate(
    reply_tokens: Sequence[Sequence[str]],
    *,
    n: int = 6,
    prefix_window: int = 32,
) -> float:
    n = int(n)
    prefix_window = int(prefix_window)
    if n <= 0 or prefix_window <= 0:
        return 0.0
    seen = set()
    repeats = 0
    total = 0
    for toks in reply_tokens:
        win = [t.lower() for t in _non_ws(toks)[:prefix_window]]
        if len(win) < n:
            continue
        for i in range(len(win) - n + 1):
            ng = tuple(win[i : i + n])
            total += 1
            if ng in seen:
                repeats += 1
            else:
                seen.add(ng)
    return (repeats / total) if total > 0 else 0.0


def cross_turn_signature_repeat_rate(dialogue_sigs: Sequence[Sequence[str]]) -> float:
    repeats = 0
    total = 0
    for sigs in dialogue_sigs:
        seen = set()
        for i, sig in enumerate(sigs):
            if i == 0:
                seen.add(sig)
                continue
            total += 1
            if sig in seen:
                repeats += 1
            seen.add(sig)
    return (repeats / total) if total > 0 else 0.0


def cross_turn_mode_repeat_rate(dialogue_modes: Sequence[Sequence[str]]) -> float:
    repeats = 0
    total = 0
    for modes in dialogue_modes:
        seen = set()
        for i, m in enumerate(modes):
            if i == 0:
                seen.add(m)
                continue
            total += 1
            if m in seen:
                repeats += 1
            seen.add(m)
    return (repeats / total) if total > 0 else 0.0


def suite_metrics_from_generations(
    *,
    all_gen_tokens: Sequence[str],
    reply_gen_tokens: Sequence[Sequence[str]],
    dialogue_reply_sigs: Sequence[Sequence[str]],
    dialogue_reply_modes: Sequence[Sequence[str]] = (),
    reply_sigs: Sequence[str],
    prefix_sigs: Sequence[str],
    reply_modes: Sequence[str] = (),
    reply_mode_sources: Sequence[str] = (),
    reply_policy_actions: Sequence[str] = (),
    reply_policy_coverages: Sequence[float] = (),
    prefix_k: int,
    template_ngram_n: int,
    template_prefix_window: int,
) -> Dict[str, Any]:
    total = max(1, len(all_gen_tokens))
    ws = sum(1 for t in all_gen_tokens if t.isspace())
    whitespace_ratio = ws / total

    non_ws_all = [t for t in all_gen_tokens if not t.isspace()]
    if len(non_ws_all) < 3:
        repeat3_global = 1.0
        loop_rate_global = 1.0
        distinct2_global = 0.0
    else:
        repeat3_global = repeat_ngram_rate(all_gen_tokens, 3, ignore_space=True)
        loop_rate_global = loop_rate(all_gen_tokens, n=3, window=128, ignore_space=True)
        distinct2_global = distinct_n(all_gen_tokens, 2, ignore_space=True)

    c = Counter(reply_sigs)
    most_common = max(c.values()) if c else 0
    unique_reply_rate = (len(c) / max(1, len(reply_sigs))) if reply_sigs else 0.0
    duplicate_reply_rate = 1.0 - unique_reply_rate
    most_common_reply_frac = (most_common / max(1, len(reply_sigs))) if reply_sigs else 0.0

    # Cross-turn / template metrics.
    prefix_k_dup = prefix_k_dup_rate(prefix_sigs)
    template_dup = template_ngram_dup_rate(
        reply_gen_tokens, n=template_ngram_n, prefix_window=template_prefix_window
    )
    cross_turn_rep = cross_turn_signature_repeat_rate(dialogue_reply_sigs)
    cross_turn_mode_rep = (
        cross_turn_mode_repeat_rate(dialogue_reply_modes) if dialogue_reply_modes else 0.0
    )

    reply_lengths = [len(t) for t in reply_gen_tokens]
    avg_len_tokens = sum(reply_lengths) / max(1, len(reply_lengths))

    mode_counts = Counter(str(m) for m in reply_modes) if reply_modes else Counter()
    mode_distribution = {
        k: (v / max(1, len(reply_modes))) for k, v in sorted(mode_counts.items())
    }

    mode_source_counts = Counter(str(s) for s in reply_mode_sources) if reply_mode_sources else Counter()
    policy_hits = int(mode_source_counts.get("policy", 0))
    policy_hit_rate = (policy_hits / max(1, len(reply_mode_sources))) if reply_mode_sources else 0.0

    policy_action_counts = (
        Counter(str(a) for a in reply_policy_actions) if reply_policy_actions else Counter()
    )
    explore = int(policy_action_counts.get("explore", 0))
    exploit = int(policy_action_counts.get("exploit", 0))
    policy_turns = int(explore + exploit)
    policy_explore_rate = explore / max(1, policy_turns) if policy_turns > 0 else 0.0
    policy_exploit_rate = exploit / max(1, policy_turns) if policy_turns > 0 else 0.0

    covs = [float(x) for x in reply_policy_coverages] if reply_policy_coverages else []
    policy_coverage_mean = (sum(covs) / len(covs)) if covs else 0.0

    return {
        "num_dialogues": float(len(dialogue_reply_sigs)),
        "num_replies": float(len(reply_sigs)),
        "avg_len_tokens": float(avg_len_tokens),
        "whitespace_ratio": float(whitespace_ratio),
        "repeat3_global": float(repeat3_global),
        "loop_rate_global": float(loop_rate_global),
        "distinct2_global": float(distinct2_global),
        "unique_reply_rate": float(unique_reply_rate),
        "duplicate_reply_rate": float(duplicate_reply_rate),
        "most_common_reply_frac": float(most_common_reply_frac),
        "prefix_k": float(prefix_k),
        "prefix_k_dup_rate": float(prefix_k_dup),
        "template_ngram_n": float(template_ngram_n),
        "template_prefix_window": float(template_prefix_window),
        "template_ngram_dup_rate": float(template_dup),
        "cross_turn_signature_repeat_rate": float(cross_turn_rep),
        "cross_turn_mode_repeat_rate": float(cross_turn_mode_rep),
        "mode_counts": dict(mode_counts),
        "mode_distribution": mode_distribution,
        "mode_source_counts": dict(mode_source_counts),
        "policy_hit_rate": float(policy_hit_rate),
        "policy_action_counts": dict(policy_action_counts),
        "policy_explore_rate": float(policy_explore_rate),
        "policy_exploit_rate": float(policy_exploit_rate),
        "policy_coverage_mean": float(policy_coverage_mean),
    }


def run_chat_suite(
    engine,
    *,
    dialogues: Sequence[Sequence[str]] = CHAT_DIALOGUES_20X3,
    max_new_tokens: int = 200,
    prefix_k: int = 8,
    template_ngram_n: int = 6,
    template_prefix_window: int = 32,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    transcripts: List[Dict[str, Any]] = []
    all_gen_tokens: List[str] = []
    reply_gen_tokens: List[List[str]] = []
    reply_sigs: List[str] = []
    prefix_sigs: List[str] = []
    dialogue_sigs: List[List[str]] = []
    reply_modes: List[str] = []
    dialogue_modes: List[List[str]] = []
    reply_mode_sources: List[str] = []
    dialogue_mode_sources: List[List[str]] = []
    reply_policy_actions: List[str] = []
    reply_policy_coverages: List[float] = []

    trace_tokens_total = 0
    trace_candidates_sum = 0
    trace_pred_matched_sum = 0
    trace_pred_emitted_sum = 0
    trace_act_evals_sum = 0
    trace_active_set_size: int = 0
    trace_rewrite_rules_total: int = 0
    trace_selector_present: int = 0

    for i, turns in enumerate(dialogues):
        history: List[Dict[str, str]] = []
        dial_sigs: List[str] = []
        dial_modes: List[str] = []
        dial_mode_sources: List[str] = []
        for j, user_msg in enumerate(turns):
            history.append({"user": user_msg, "system": ""})
            prompt = build_chat_prompt(history)
            out = engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                mode="greedy",
                dialogue_id=int(i),
                turn=int(j),
            )
            resp = out["text"][len(prompt) :]
            history[-1]["system"] = resp
            history[-1]["mode"] = str(out.get("mode") or "default")
            history[-1]["mode_source"] = str(out.get("mode_source") or "router")
            history[-1]["mode_policy_action"] = str(out.get("mode_policy_action") or "")
            history[-1]["policy_coverage"] = float(out.get("policy_coverage") or 0.0)
            history[-1]["user_sig"] = str(out.get("user_sig") or "")
            history[-1]["trace"] = dict(out.get("trace") or {})

            sig = reply_signature(resp)
            reply_sigs.append(sig)
            dial_sigs.append(sig)

            m = str(out.get("mode") or "default")
            reply_modes.append(m)
            dial_modes.append(m)

            ms = str(out.get("mode_source") or "router")
            reply_mode_sources.append(ms)
            dial_mode_sources.append(ms)

            reply_policy_actions.append(str(out.get("mode_policy_action") or ""))
            reply_policy_coverages.append(float(out.get("policy_coverage") or 0.0))

            gen_toks = list(out["gen_tokens"])
            reply_gen_tokens.append(gen_toks)
            all_gen_tokens.extend(gen_toks)
            prefix_sigs.append(prefix_k_signature(gen_toks, k=prefix_k))

            tr = out.get("trace") or {}
            cand_post = tr.get("candidates_post_rewrite") or []
            pred_matched = tr.get("predictor_matched") or []
            pred_emitted = tr.get("predictor_emitted") or []
            if isinstance(cand_post, list) and isinstance(pred_matched, list) and isinstance(pred_emitted, list):
                n_tok = min(len(cand_post), len(pred_matched), len(pred_emitted))
                if n_tok > 0:
                    trace_tokens_total += int(n_tok)
                    trace_candidates_sum += int(sum(int(x) for x in cand_post[:n_tok]))
                    trace_pred_matched_sum += int(sum(int(x) for x in pred_matched[:n_tok]))
                    trace_pred_emitted_sum += int(sum(int(x) for x in pred_emitted[:n_tok]))

                    rr_total = int(tr.get("rewrite_rules_total", 0) or 0)
                    sel_present = 1 if tr.get("selector_id") else 0
                    trace_act_evals_sum += int(sum(int(x) for x in pred_matched[:n_tok]))
                    trace_act_evals_sum += int(n_tok * rr_total)
                    trace_act_evals_sum += int(n_tok * sel_present)

                    if trace_active_set_size <= 0:
                        trace_active_set_size = int(tr.get("active_set_size", 0) or 0)
                        trace_rewrite_rules_total = int(rr_total)
                        trace_selector_present = int(sel_present)

        dialogue_sigs.append(dial_sigs)
        dialogue_modes.append(dial_modes)
        dialogue_mode_sources.append(dial_mode_sources)
        full_text = build_chat_prompt(history)
        transcripts.append({"prompt_id": i, "turns": history, "full_text": full_text})

    metrics = suite_metrics_from_generations(
        all_gen_tokens=all_gen_tokens,
        reply_gen_tokens=reply_gen_tokens,
        dialogue_reply_sigs=dialogue_sigs,
        dialogue_reply_modes=dialogue_modes,
        reply_sigs=reply_sigs,
        prefix_sigs=prefix_sigs,
        reply_modes=reply_modes,
        reply_mode_sources=reply_mode_sources,
        reply_policy_actions=reply_policy_actions,
        reply_policy_coverages=reply_policy_coverages,
        prefix_k=prefix_k,
        template_ngram_n=template_ngram_n,
        template_prefix_window=template_prefix_window,
    )
    if trace_tokens_total > 0:
        metrics["trace_tokens_total"] = int(trace_tokens_total)
        metrics["candidates_considered_per_token_mean"] = float(
            trace_candidates_sum / trace_tokens_total
        )
        metrics["predictor_matched_per_token_mean"] = float(
            trace_pred_matched_sum / trace_tokens_total
        )
        metrics["predictor_emitted_per_token_mean"] = float(
            trace_pred_emitted_sum / trace_tokens_total
        )
        metrics["acts_considered_per_token_mean"] = float(
            trace_act_evals_sum / trace_tokens_total
        )
        metrics["search_steps_per_turn_mean"] = float(
            trace_act_evals_sum / max(1, len(reply_sigs))
        )
        metrics["trace_active_set_size"] = int(trace_active_set_size)
        metrics["trace_rewrite_rules_total"] = int(trace_rewrite_rules_total)
        metrics["trace_selector_present"] = int(trace_selector_present)
    return transcripts, metrics


def read_transcripts_jsonl(path: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def suite_metrics_from_transcripts(
    transcripts: Sequence[Dict[str, Any]],
    *,
    prefix_k: int = 8,
    template_ngram_n: int = 6,
    template_prefix_window: int = 32,
) -> Dict[str, Any]:
    all_gen_tokens: List[str] = []
    reply_gen_tokens: List[List[str]] = []
    reply_sigs: List[str] = []
    prefix_sigs: List[str] = []
    dialogue_sigs: List[List[str]] = []
    reply_modes: List[str] = []
    dialogue_modes: List[List[str]] = []
    reply_mode_sources: List[str] = []
    reply_policy_actions: List[str] = []
    reply_policy_coverages: List[float] = []

    trace_tokens_total = 0
    trace_candidates_sum = 0
    trace_pred_matched_sum = 0
    trace_pred_emitted_sum = 0
    trace_act_evals_sum = 0
    trace_active_set_size: int = 0
    trace_rewrite_rules_total: int = 0
    trace_selector_present: int = 0

    for rec in transcripts:
        turns = rec.get("turns", [])
        dial_sigs: List[str] = []
        dial_modes: List[str] = []
        for t in turns:
            resp = str(t.get("system", ""))
            gen_toks = tokenize_text(resp)
            reply_gen_tokens.append(gen_toks)
            all_gen_tokens.extend(gen_toks)
            sig = reply_signature(resp)
            reply_sigs.append(sig)
            dial_sigs.append(sig)
            prefix_sigs.append(prefix_k_signature(gen_toks, k=prefix_k))
            m = str(t.get("mode") or "default")
            reply_modes.append(m)
            dial_modes.append(m)
            reply_mode_sources.append(str(t.get("mode_source") or "router"))
            reply_policy_actions.append(str(t.get("mode_policy_action") or ""))
            try:
                reply_policy_coverages.append(float(t.get("policy_coverage") or 0.0))
            except Exception:
                reply_policy_coverages.append(0.0)

            tr = t.get("trace") or {}
            if isinstance(tr, dict):
                cand_post = tr.get("candidates_post_rewrite") or []
                pred_matched = tr.get("predictor_matched") or []
                pred_emitted = tr.get("predictor_emitted") or []
                if (
                    isinstance(cand_post, list)
                    and isinstance(pred_matched, list)
                    and isinstance(pred_emitted, list)
                ):
                    n_tok = min(len(cand_post), len(pred_matched), len(pred_emitted))
                    if n_tok > 0:
                        trace_tokens_total += int(n_tok)
                        trace_candidates_sum += int(sum(int(x) for x in cand_post[:n_tok]))
                        trace_pred_matched_sum += int(
                            sum(int(x) for x in pred_matched[:n_tok])
                        )
                        trace_pred_emitted_sum += int(
                            sum(int(x) for x in pred_emitted[:n_tok])
                        )

                        rr_total = int(tr.get("rewrite_rules_total", 0) or 0)
                        sel_present = 1 if tr.get("selector_id") else 0
                        trace_act_evals_sum += int(sum(int(x) for x in pred_matched[:n_tok]))
                        trace_act_evals_sum += int(n_tok * rr_total)
                        trace_act_evals_sum += int(n_tok * sel_present)

                        if trace_active_set_size <= 0:
                            trace_active_set_size = int(tr.get("active_set_size", 0) or 0)
                            trace_rewrite_rules_total = int(rr_total)
                            trace_selector_present = int(sel_present)
        if dial_sigs:
            dialogue_sigs.append(dial_sigs)
            dialogue_modes.append(dial_modes)

    metrics = suite_metrics_from_generations(
        all_gen_tokens=all_gen_tokens,
        reply_gen_tokens=reply_gen_tokens,
        dialogue_reply_sigs=dialogue_sigs,
        dialogue_reply_modes=dialogue_modes,
        reply_sigs=reply_sigs,
        prefix_sigs=prefix_sigs,
        reply_modes=reply_modes,
        reply_mode_sources=reply_mode_sources,
        reply_policy_actions=reply_policy_actions,
        reply_policy_coverages=reply_policy_coverages,
        prefix_k=prefix_k,
        template_ngram_n=template_ngram_n,
        template_prefix_window=template_prefix_window,
    )
    if trace_tokens_total > 0:
        metrics["trace_tokens_total"] = int(trace_tokens_total)
        metrics["candidates_considered_per_token_mean"] = float(
            trace_candidates_sum / trace_tokens_total
        )
        metrics["predictor_matched_per_token_mean"] = float(
            trace_pred_matched_sum / trace_tokens_total
        )
        metrics["predictor_emitted_per_token_mean"] = float(
            trace_pred_emitted_sum / trace_tokens_total
        )
        metrics["acts_considered_per_token_mean"] = float(
            trace_act_evals_sum / trace_tokens_total
        )
        metrics["search_steps_per_turn_mean"] = float(
            trace_act_evals_sum / max(1, len(reply_sigs))
        )
        metrics["trace_active_set_size"] = int(trace_active_set_size)
        metrics["trace_rewrite_rules_total"] = int(trace_rewrite_rules_total)
        metrics["trace_selector_present"] = int(trace_selector_present)
    return metrics
