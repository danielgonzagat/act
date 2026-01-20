from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def sha256_text_v100(text: str) -> str:
    h = hashlib.sha256()
    h.update(str(text).encode("utf-8"))
    return h.hexdigest()


def discourse_event_id_v100(event_sig: str) -> str:
    return f"discourse_event_v100_{str(event_sig)}"


def _discourse_event_sig_v100(*, prev_event_sig: str, event_body: Dict[str, Any]) -> str:
    payload = str(prev_event_sig or "") + canonical_json_dumps(event_body)
    return sha256_hex(payload.encode("utf-8"))


def _round6(x: Any) -> float:
    try:
        return float(round(float(x), 6))
    except Exception:
        return 0.0


def _tokens_v100(text: str) -> List[str]:
    return [t for t in str(text or "").lower().split() if t]


def fluency_metrics_v100(*, text: str, response_kind: str) -> Dict[str, Any]:
    toks = _tokens_v100(text)
    bigrams = list(zip(toks, toks[1:])) if len(toks) >= 2 else []
    trigrams = list(zip(toks, toks[1:], toks[2:])) if len(toks) >= 3 else []
    big_total = len(bigrams)
    tri_total = len(trigrams)
    big_unique = len(set(bigrams))
    tri_unique = len(set(trigrams))
    big_rep = (1.0 - (float(big_unique) / float(big_total))) if big_total > 0 else 0.0
    tri_rep = (1.0 - (float(tri_unique) / float(tri_total))) if tri_total > 0 else 0.0
    has_q = "?" in str(text)
    starts_with_label = str(text).startswith("Resultado:") or str(text).startswith("Resposta:")
    lines = str(text).splitlines()
    m = {
        "chars": int(len(str(text))),
        "words": int(len(toks)),
        "lines": int(len(lines)),
        "bigram_repeat_ratio": _round6(big_rep),
        "trigram_repeat_ratio": _round6(tri_rep),
        "has_question_mark": bool(has_q),
        "starts_with_label": bool(starts_with_label),
        "response_kind": str(response_kind),
    }
    return dict(m)


def fluency_score_v100(*, metrics: Dict[str, Any]) -> float:
    """
    Deterministic "fluency" score (explicit formula; not learned).
    Higher is better. Values in [0, 1] (clamped).
    """
    big_rep = float(metrics.get("bigram_repeat_ratio") or 0.0)
    tri_rep = float(metrics.get("trigram_repeat_ratio") or 0.0)
    words = int(metrics.get("words") or 0)
    chars = int(metrics.get("chars") or 0)
    has_q = bool(metrics.get("has_question_mark", False))
    has_label = bool(metrics.get("starts_with_label", False))
    kind = str(metrics.get("response_kind") or "")

    # Start slightly below 1.0 so deterministic bonuses can matter under clamping.
    score = 0.90
    score -= 0.50 * big_rep
    score -= 0.70 * tri_rep
    score -= 0.002 * max(0, int(words) - 40)
    score -= 0.001 * max(0, int(chars) - 240)
    if kind == "respond":
        score += 0.05 if has_label else 0.0
    if kind == "clarify":
        score += 0.10 if has_q else -0.20
    score = max(0.0, min(1.0, score))
    return float(round(score, 6))


def candidate_id_v100(*, variant_id: str, text_sha256: str, fragment_ids: Sequence[str]) -> str:
    body = {
        "schema_version": 100,
        "variant_id": str(variant_id),
        "text_sha256": str(text_sha256),
        "fragment_ids": [str(x) for x in fragment_ids if isinstance(x, str) and x],
    }
    return f"cand_v100_{_stable_hash_obj(body)}"


def generate_text_candidates_v100(
    *,
    base_text: str,
    response_kind: str,
    allow_wrappers: bool,
) -> List[Dict[str, Any]]:
    """
    Generate >=3 deterministic candidates.
    The caller decides allow_wrappers based on intent/renderer constraints.
    """
    base = str(base_text)
    # Variant 0: plain.
    texts: List[Tuple[str, str, List[str]]] = [("v100_plain", base, [])]

    if allow_wrappers:
        # Variant 1: "Resultado: <base>"
        texts.append(("v100_result_prefix", f"Resultado: {base}", ["frag_v100_prefix_result_v0"]))
        # Variant 2: "Resposta: <base>"
        texts.append(("v100_answer_prefix", f"Resposta: {base}", ["frag_v100_prefix_answer_v0"]))
    else:
        # Still produce distinct variants, but ensure they score lower and are not selected:
        # trailing newline and a leading space (deterministic).
        texts.append(("v100_trailing_newline", base + "\n", []))
        texts.append(("v100_leading_space", " " + base, []))

    cands: List[Dict[str, Any]] = []
    for variant_id, text, frags in texts:
        t = str(text)
        tsha = sha256_text_v100(t)
        metrics = fluency_metrics_v100(text=t, response_kind=str(response_kind))
        score = fluency_score_v100(metrics=metrics)
        cid = candidate_id_v100(variant_id=str(variant_id), text_sha256=str(tsha), fragment_ids=list(frags))
        cands.append(
            {
                "candidate_id": str(cid),
                "variant_id": str(variant_id),
                "text": str(t),
                "text_sha256": str(tsha),
                "fluency_score": float(score),
                "fluency_metrics": dict(metrics),
                "fragment_ids": [str(x) for x in frags],
            }
        )

    # Deterministic rank: score desc, words asc, chars asc, sha asc, candidate_id asc.
    def _rk(d: Dict[str, Any]) -> Tuple[Any, ...]:
        m = d.get("fluency_metrics") if isinstance(d.get("fluency_metrics"), dict) else {}
        return (
            -float(d.get("fluency_score") or 0.0),
            int(m.get("words") or 0),
            int(m.get("chars") or 0),
            str(d.get("text_sha256") or ""),
            str(d.get("candidate_id") or ""),
        )

    cands.sort(key=_rk)
    return list(cands)


@dataclass(frozen=True)
class DiscourseEventV100:
    conversation_id: str
    turn_id: str
    turn_index: int
    discourse_state_before: Dict[str, Any]
    discourse_state_after: Dict[str, Any]
    candidates_topk: List[Dict[str, Any]]
    selected_candidate_id: str
    cause_ids: Dict[str, Any]
    created_step: int
    prev_event_sig: str

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 100,
            "kind": "discourse_event_v100",
            "conversation_id": str(self.conversation_id),
            "turn_id": str(self.turn_id),
            "turn_index": int(self.turn_index),
            "discourse_state_before": dict(self.discourse_state_before) if isinstance(self.discourse_state_before, dict) else {},
            "discourse_state_after": dict(self.discourse_state_after) if isinstance(self.discourse_state_after, dict) else {},
            "candidates_topk": [dict(x) for x in self.candidates_topk if isinstance(x, dict)],
            "selected_candidate_id": str(self.selected_candidate_id),
            "cause_ids": dict(self.cause_ids) if isinstance(self.cause_ids, dict) else {},
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = _discourse_event_sig_v100(prev_event_sig=str(self.prev_event_sig or ""), event_body=dict(body))
        return dict(body, prev_event_sig=str(self.prev_event_sig or ""), event_sig=str(sig), event_id=str(discourse_event_id_v100(sig)))


def verify_discourse_event_sig_v100(ev: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(ev, dict):
        return False, "discourse_event_not_dict", {}
    prev_sig = str(ev.get("prev_event_sig") or "")
    want_sig = str(ev.get("event_sig") or "")
    body = dict(ev)
    body.pop("prev_event_sig", None)
    body.pop("event_sig", None)
    body.pop("event_id", None)
    got_sig = _discourse_event_sig_v100(prev_event_sig=str(prev_sig), event_body=body)
    if got_sig != want_sig:
        return False, "discourse_event_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    if str(ev.get("event_id") or "") != discourse_event_id_v100(want_sig):
        return False, "discourse_event_id_mismatch", {"want": discourse_event_id_v100(want_sig), "got": str(ev.get("event_id") or "")}
    return True, "ok", {}


def compute_discourse_chain_hash_v100(discourse_events: Sequence[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in discourse_events:
        if not isinstance(ev, dict):
            continue
        ids.append(str(ev.get("event_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


def render_discourse_text_v100(state: Dict[str, Any]) -> str:
    body = dict(state) if isinstance(state, dict) else {}
    # Stable single-line canonical JSON for strict reproducibility.
    return "DISCOURSE: " + canonical_json_dumps(body)


def render_dossier_text_v100() -> str:
    """
    Deterministic regulatory dossier (audit-oriented, factual).
    """
    lines: List[str] = []
    lines.append("DOSSIER (V100):")
    lines.append("A) Reexecução/Prova")
    lines.append("- Determinismo: smoke try1==try2 por hashes (store/transcript/state/parses/plans/ledgers).")
    lines.append("- Record-keeping: logs WORM JSONL hash-chained (turns/parses/plans/memory/belief/evidence/goal/discourse).")
    lines.append("- Fail-closed: comandos inválidos/ambíguos -> correção/clarificação; sem inventar dados.")
    lines.append("B) Safety por capacidades")
    lines.append("- Capacidades são atos explícitos (raw commands + concept_csv); podem ser restringidas por estado/policy/match.")
    lines.append("C) Framework mapping (alto nível)")
    lines.append("- NIST AI RMF: GOVERN/MAP/MEASURE/MANAGE via logs, verificadores, replay e gates determinísticos.")
    lines.append("- EU AI Act: rastreabilidade/record-keeping/transparência por design (sem promessas fora do log).")
    lines.append("- ISO/IEC 42001: controles de governança e evidência (políticas + auditoria).")
    lines.append("D) No-hybridization")
    lines.append("- Verificador falha se detectar dependências proibidas (torch/tensorflow/jax/transformers/openai/sentencepiece).")
    lines.append("- Motivação: misturar modelos/embeddings quebra invariantes de estado explícito e auditabilidade.")
    return "\n".join(lines)
