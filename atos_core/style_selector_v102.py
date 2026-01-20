from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .discourse_templates_v102 import TemplateV102, is_template_compatible_v102, render_template_v102
from .style_critics_v102 import fluency_score_v102, run_critics_v102, text_sha256_v102
from .style_profile_v102 import (
    EXAMPLE_YES,
    STRUCT_BULLETS,
    STRUCT_PLAIN,
    STRUCT_STEPS,
    StyleProfileV102,
    VERBOSITY_LONG,
    VERBOSITY_SHORT,
)


def _round6(x: Any) -> float:
    try:
        return float(round(float(x), 6))
    except Exception:
        return 0.0


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def style_candidate_id_v102(*, template_id: str, text_sha256: str, variant_id: str) -> str:
    body = {"schema_version": 102, "template_id": str(template_id), "text_sha256": str(text_sha256), "variant_id": str(variant_id)}
    return f"style_cand_v102_{_stable_hash_obj(body)}"


def _base_score_v102(*, template_id: str, style_profile: StyleProfileV102, response_kind: str, binding_status: str) -> float:
    score = 0.80
    if style_profile.verbosity_preference == VERBOSITY_SHORT and template_id.endswith("direct_short_v0"):
        score += 0.05
    if style_profile.verbosity_preference == VERBOSITY_LONG and ("steps" in template_id or "confusion" in template_id):
        score += 0.05
    if style_profile.structure_preference == STRUCT_STEPS and "steps" in template_id:
        score += 0.05
    if style_profile.structure_preference == STRUCT_BULLETS and "bullets" in template_id:
        score += 0.05
    if style_profile.structure_preference == STRUCT_PLAIN and ("direct" in template_id or "transition" in template_id):
        score += 0.02
    if response_kind in {"clarify", "confirm"} and binding_status in {"AMBIGUOUS", "MISS"} and "clarify" in template_id:
        score += 0.08
    if bool(style_profile.user_confusion_recent) and "confusion" in template_id:
        score += 0.10
    score = max(0.0, min(1.0, score))
    return _round6(score)


def _soft_choice_index_v102(*, seed: int, salt: str, candidates: Sequence[Dict[str, Any]]) -> int:
    """
    Deterministic soft selection among the top candidates when scores are close.
    Uses rank-based integer weights and a hash-derived pseudo-RNG.
    """
    ids = [str(c.get("candidate_id") or "") for c in candidates if isinstance(c, dict)]
    body = {"seed": int(seed), "salt": str(salt), "ids": list(ids)}
    h = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    # Use first 8 hex chars as a stable integer.
    try:
        r = int(h[:8], 16)
    except Exception:
        r = 0
    weights = [3, 2, 1, 1, 1][: len(ids)]
    total = sum(weights) if weights else 1
    pick = r % total
    acc = 0
    for i, w in enumerate(weights):
        acc += int(w)
        if pick < acc:
            return int(i)
    return 0


def build_and_select_style_candidates_v102(
    *,
    templates: Sequence[TemplateV102],
    core_text: str,
    response_kind: str,
    style_profile: StyleProfileV102,
    intent_id: str,
    slots: Dict[str, Any],
    binding_status: str,
    recent_assistant_texts: Sequence[str],
    recent_template_ids: Sequence[str],
    seed: int,
    selection_salt: str,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "candidates_topk": [...sorted...],
        "selected": {...},
        "selection": {"method": "...", "margin": float, "soft_index": int},
      }
    Candidate schema is stable and audit-friendly.
    """
    rk = str(response_kind or "")
    bs = str(binding_status or "")
    prof = style_profile
    slots2 = dict(slots) if isinstance(slots, dict) else {}

    # Filter compatible templates; always keep fallback template as an option.
    compatibles: List[TemplateV102] = []
    fallback: Optional[TemplateV102] = None
    for t in templates:
        if not isinstance(t, TemplateV102):
            continue
        if bool(t.constraints.get("fallback", False)):
            fallback = t
        if is_template_compatible_v102(template=t, response_kind=rk, style_profile=prof, binding_status=bs):
            compatibles.append(t)

    # Ensure at least 5 templates compete (even if style filters are narrow).
    if len(compatibles) < 5:
        extra = [t for t in templates if isinstance(t, TemplateV102) and t not in compatibles]
        extra.sort(key=lambda t: str(t.template_id))
        for t in extra:
            compatibles.append(t)
            if len(compatibles) >= 5:
                break

    candidates: List[Dict[str, Any]] = []
    for t in compatibles:
        rend = render_template_v102(
            template_id=str(t.template_id),
            core_text=str(core_text),
            response_kind=rk,
            style_profile=prof,
            intent_id=str(intent_id),
            slots=dict(slots2),
            binding_status=bs,
        )
        text = str(rend.get("text") or "")
        tsha = text_sha256_v102(text)
        base = _base_score_v102(template_id=str(t.template_id), style_profile=prof, response_kind=rk, binding_status=bs)
        critics = run_critics_v102(
            candidate_text=str(text),
            response_kind=rk,
            style_profile=prof,
            binding_status=bs,
            recent_assistant_texts=list(recent_assistant_texts),
            recent_template_ids=list(recent_template_ids),
            template_id=str(t.template_id),
        )
        score = fluency_score_v102(base_score=float(base), critics=dict(critics))
        cid = style_candidate_id_v102(template_id=str(t.template_id), text_sha256=str(tsha), variant_id="v102")
        candidates.append(
            {
                "schema_version": 102,
                "kind": "style_candidate_v102",
                "candidate_id": str(cid),
                "template_id": str(t.template_id),
                "renderer_id": "style_selector_v102",
                "variant_id": "v102",
                "text": str(text),
                "text_sha256": str(tsha),
                "base_score": float(base),
                "critics": dict(critics),
                "fluency_score": float(score),
                "fluency_metrics": dict(critics.get("metrics") or {}),
                "fragment_ids": [str(x) for x in rend.get("fragment_ids") or [] if isinstance(x, str) and x],
                "slots_filled": dict(rend.get("slots_filled") or {}) if isinstance(rend.get("slots_filled"), dict) else {},
            }
        )

    # Stable deterministic rank: score desc, ok first, template_id asc, sha asc, candidate_id asc.
    def _rk_c(d: Dict[str, Any]) -> Tuple[Any, ...]:
        ok = bool(((d.get("critics") or {}).get("ok")) if isinstance(d.get("critics"), dict) else False)
        m = d.get("fluency_metrics") if isinstance(d.get("fluency_metrics"), dict) else {}
        return (
            0 if ok else 1,
            -float(d.get("fluency_score") or 0.0),
            int(m.get("words") or 0),
            str(d.get("template_id") or ""),
            str(d.get("text_sha256") or ""),
            str(d.get("candidate_id") or ""),
        )

    candidates_sorted = sorted(list(candidates), key=_rk_c)
    candidates_topk = candidates_sorted[: int(top_k)]

    # Choose winner: if no ok candidates, fallback; else hard vs soft.
    ok_cands = [c for c in candidates_sorted if isinstance(c, dict) and bool(((c.get("critics") or {}).get("ok")) if isinstance(c.get("critics"), dict) else False)]

    selection = {"method": "argmax", "margin": 0.02, "soft_index": 0}
    chosen: Dict[str, Any] = {}
    if not ok_cands:
        if fallback is None:
            chosen = dict(candidates_topk[0]) if candidates_topk else {"candidate_id": "", "text": str(core_text), "text_sha256": text_sha256_v102(str(core_text))}
        else:
            # Render fallback deterministically.
            rend2 = render_template_v102(
                template_id=str(fallback.template_id),
                core_text=str(core_text),
                response_kind=rk,
                style_profile=prof,
                intent_id=str(intent_id),
                slots=dict(slots2),
                binding_status=bs,
            )
            text2 = str(rend2.get("text") or str(core_text))
            tsha2 = text_sha256_v102(text2)
            chosen = {
                "schema_version": 102,
                "kind": "style_candidate_v102",
                "candidate_id": style_candidate_id_v102(template_id=str(fallback.template_id), text_sha256=str(tsha2), variant_id="v102_fallback"),
                "template_id": str(fallback.template_id),
                "renderer_id": "style_selector_v102",
                "variant_id": "v102_fallback",
                "text": str(text2),
                "text_sha256": str(tsha2),
                "base_score": 0.0,
                "critics": {"ok": True, "total_score_delta": 0.0, "results": [], "metrics": {}},
                "fluency_score": 0.0,
                "fluency_metrics": {},
                "fragment_ids": [],
                "slots_filled": {},
            }
        selection = {"method": "fallback", "margin": 0.0, "soft_index": 0}
    else:
        # Consider soft selection only among the top-3 ok candidates when close.
        top3 = ok_cands[:3]
        if len(top3) >= 2:
            s0 = float(top3[0].get("fluency_score") or 0.0)
            s1 = float(top3[1].get("fluency_score") or 0.0)
            margin = float(selection["margin"])
            if abs(s0 - s1) <= margin:
                idx = _soft_choice_index_v102(seed=int(seed), salt=str(selection_salt), candidates=list(top3))
                selection = {"method": "soft", "margin": float(margin), "soft_index": int(idx)}
                chosen = dict(top3[int(idx)])
            else:
                chosen = dict(top3[0])
        else:
            chosen = dict(ok_cands[0])

    return {
        "candidates_topk": [dict(x) for x in candidates_topk if isinstance(x, dict)],
        "selected": dict(chosen),
        "selection": dict(selection),
    }


def explain_style_text_v102(
    *,
    style_event: Dict[str, Any],
    max_k: int = 5,
) -> str:
    """
    Deterministic explain_style renderer based only on style_event fields.
    """
    ev = dict(style_event) if isinstance(style_event, dict) else {}
    sel = str(ev.get("selected_candidate_id") or "")
    tid = str(ev.get("selected_template_id") or "")
    ok = bool(ev.get("selected_ok", False))
    line1 = f"EXPLAIN_STYLE: chosen={sel} template={tid} ok={str(ok).lower()}"

    cands = ev.get("candidates_topk") if isinstance(ev.get("candidates_topk"), list) else []
    parts: List[str] = []
    for i, c in enumerate(cands[: int(max_k)], start=1):
        if not isinstance(c, dict):
            continue
        cid = str(c.get("candidate_id") or "")
        t = str(c.get("template_id") or "")
        sc = _round6(c.get("fluency_score"))
        okc = bool(((c.get("critics") or {}).get("ok")) if isinstance(c.get("critics"), dict) else False)
        parts.append(f"{i}){cid} tmpl={t} score={sc:.6f} ok={str(okc).lower()}")
    line2 = "TOPK_STYLE: " + "; ".join(parts) if parts else "TOPK_STYLE:"

    sel_reason = ev.get("selection") if isinstance(ev.get("selection"), dict) else {}
    method = str(sel_reason.get("method") or "")
    margin = _round6(sel_reason.get("margin"))
    sidx = int(sel_reason.get("soft_index") or 0)
    line3 = f"SELECTION: method={method} margin={margin:.6f} soft_index={sidx}"

    return "\n".join([line1, line2, line3])

