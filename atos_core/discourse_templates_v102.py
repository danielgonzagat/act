from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .style_profile_v102 import (
    EXAMPLE_NO,
    EXAMPLE_YES,
    STRUCT_BULLETS,
    STRUCT_PLAIN,
    STRUCT_STEPS,
    StyleProfileV102,
    VERBOSITY_LONG,
    VERBOSITY_SHORT,
)


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def template_id_v102(name: str) -> str:
    return str(name)


@dataclass(frozen=True)
class TemplateV102:
    template_id: str
    name: str
    roles: List[str]
    constraints: Dict[str, Any]
    slots: List[str]

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 102,
            "kind": "discourse_template_v102",
            "template_id": str(self.template_id),
            "name": str(self.name),
            "roles": [str(x) for x in self.roles if isinstance(x, str) and x],
            "constraints": dict(self.constraints) if isinstance(self.constraints, dict) else {},
            "slots": [str(x) for x in self.slots if isinstance(x, str) and x],
        }
        body["template_sig"] = _stable_hash_obj(body)
        body["mdl_cost_bytes"] = int(len(canonical_json_dumps(body).encode("utf-8")))
        return dict(body)


def base_templates_v102() -> List[TemplateV102]:
    """
    Deterministic seed template library (>=8).
    Templates are "objects" (not free-form text); rendering happens in code, but
    template_id/name/constraints/slots are explicit and hashable.
    """
    out: List[TemplateV102] = []
    out.append(
        TemplateV102(
            template_id=template_id_v102("tmpl_v102_direct_short_v0"),
            name="Direto (curto)",
            roles=["respond", "confirm", "summarize"],
            constraints={"structure": [STRUCT_PLAIN], "verbosity": [VERBOSITY_SHORT]},
            slots=["CORE"],
        )
    )
    out.append(
        TemplateV102(
            template_id=template_id_v102("tmpl_v102_direct_medium_v0"),
            name="Direto (médio)",
            roles=["respond", "confirm", "summarize"],
            constraints={"structure": [STRUCT_PLAIN], "verbosity": ["MEDIUM"]},
            slots=["CORE"],
        )
    )
    out.append(
        TemplateV102(
            template_id=template_id_v102("tmpl_v102_short_plus_detail_v0"),
            name="Curto + 1 detalhe",
            roles=["respond", "summarize"],
            constraints={"structure": [STRUCT_PLAIN], "verbosity": [VERBOSITY_SHORT, "MEDIUM"]},
            slots=["CORE", "FOLLOWUP_Q"],
        )
    )
    out.append(
        TemplateV102(
            template_id=template_id_v102("tmpl_v102_steps_v0"),
            name="Passos (1/2/3)",
            roles=["respond", "summarize", "confirm"],
            constraints={"structure": [STRUCT_STEPS], "verbosity": [VERBOSITY_LONG, "MEDIUM"]},
            slots=["CORE_STEPS", "EXAMPLE"],
        )
    )
    out.append(
        TemplateV102(
            template_id=template_id_v102("tmpl_v102_bullets_v0"),
            name="Bullets",
            roles=["respond", "summarize"],
            constraints={"structure": [STRUCT_BULLETS], "verbosity": [VERBOSITY_LONG, "MEDIUM"]},
            slots=["CORE_BULLETS", "EXAMPLE"],
        )
    )
    out.append(
        TemplateV102(
            template_id=template_id_v102("tmpl_v102_transition_core_v0"),
            name="Transição suave + core",
            roles=["respond", "summarize"],
            constraints={"structure": [STRUCT_PLAIN], "verbosity": ["MEDIUM", VERBOSITY_LONG]},
            slots=["TRANSITION", "CORE"],
        )
    )
    out.append(
        TemplateV102(
            template_id=template_id_v102("tmpl_v102_clarify_ambiguous_v0"),
            name="Clarificação (ambíguo/miss)",
            roles=["clarify", "confirm"],
            constraints={"requires_question": True},
            slots=["CLARIFY"],
        )
    )
    out.append(
        TemplateV102(
            template_id=template_id_v102("tmpl_v102_confusion_reexplain_v0"),
            name="Reexplicar (confusão detectada)",
            roles=["respond", "summarize", "confirm"],
            constraints={"verbosity": [VERBOSITY_LONG]},
            slots=["CORE", "EXAMPLE", "FOLLOWUP_Q"],
        )
    )
    out.append(
        TemplateV102(
            template_id=template_id_v102("tmpl_v102_fallback_safe_v0"),
            name="Fallback seguro",
            roles=["respond", "clarify", "confirm"],
            constraints={"fallback": True},
            slots=["CORE", "FOLLOWUP_Q"],
        )
    )
    out.sort(key=lambda t: str(t.template_id))
    return out


def template_by_id_v102(templates: Sequence[TemplateV102]) -> Dict[str, TemplateV102]:
    out: Dict[str, TemplateV102] = {}
    for t in templates:
        if isinstance(t, TemplateV102) and str(t.template_id):
            out[str(t.template_id)] = t
    return dict(out)


def is_template_compatible_v102(
    *,
    template: TemplateV102,
    response_kind: str,
    style_profile: StyleProfileV102,
    binding_status: str,
) -> bool:
    roles = set([str(x) for x in template.roles if isinstance(x, str)])
    rk = str(response_kind or "")
    if rk and rk not in roles:
        return False

    c = template.constraints if isinstance(template.constraints, dict) else {}
    # Clarification templates only when binding ambiguous/miss or response_kind is clarify/confirm.
    if bool(c.get("requires_question", False)):
        if rk not in {"clarify", "confirm"}:
            return False
        if str(binding_status or "") not in {"AMBIGUOUS", "MISS"} and rk != "clarify":
            return False

    # Apply style profile filters if provided.
    allowed_struct = c.get("structure")
    if isinstance(allowed_struct, list) and allowed_struct:
        if str(style_profile.structure_preference) not in set([str(x) for x in allowed_struct]):
            return False
    allowed_verb = c.get("verbosity")
    if isinstance(allowed_verb, list) and allowed_verb:
        if str(style_profile.verbosity_preference) not in set([str(x) for x in allowed_verb]):
            return False

    return True


def _split_lines(core_text: str) -> List[str]:
    lines = [str(x).strip() for x in str(core_text).splitlines()]
    return [x for x in lines if x]


def _render_steps(lines: Sequence[str]) -> str:
    out: List[str] = []
    for i, ln in enumerate([str(x) for x in lines if isinstance(x, str) and x], start=1):
        out.append(f"{i}) {ln}")
    return "\n".join(out) if out else ""


def _render_bullets(lines: Sequence[str]) -> str:
    out: List[str] = []
    for ln in [str(x) for x in lines if isinstance(x, str) and x]:
        out.append(f"- {ln}")
    return "\n".join(out) if out else ""


def _example_for_context(intent_id: str, slots: Dict[str, Any]) -> str:
    """
    Deterministic minimal examples (no new facts).
    """
    iid = str(intent_id or "")
    if iid == "INTENT_SET":
        k = str(slots.get("k") or "x")
        v = str(slots.get("v") or "4")
        return f"Exemplo: set {k} to {v}"
    if iid == "INTENT_GET":
        k = str(slots.get("k") or "x")
        return f"Exemplo: get {k}"
    if iid == "INTENT_ADD":
        a = str(slots.get("a") or "x")
        b = str(slots.get("b") or "y")
        return f"Exemplo: add {a} and {b}"
    if iid == "INTENT_SUMMARY":
        return "Exemplo: summary"
    if iid == "INTENT_END":
        return "Exemplo: end"
    return ""


def render_template_v102(
    *,
    template_id: str,
    core_text: str,
    response_kind: str,
    style_profile: StyleProfileV102,
    intent_id: str,
    slots: Dict[str, Any],
    binding_status: str,
) -> Dict[str, Any]:
    """
    Render a template to text deterministically.
    Returns: {"text": str, "fragment_ids": [...], "slots_filled": {...}}
    """
    tid = str(template_id or "")
    core = str(core_text or "")
    rk = str(response_kind or "")
    bp = style_profile
    lines = _split_lines(core)

    frag_ids: List[str] = []
    slots_filled: Dict[str, Any] = {}

    if tid == "tmpl_v102_direct_short_v0":
        return {"text": core, "fragment_ids": frag_ids, "slots_filled": slots_filled}

    if tid == "tmpl_v102_direct_medium_v0":
        return {"text": core, "fragment_ids": frag_ids, "slots_filled": slots_filled}

    if tid == "tmpl_v102_short_plus_detail_v0":
        follow = "Se quiser, posso detalhar."
        return {"text": core + "\n" + follow, "fragment_ids": frag_ids, "slots_filled": {"FOLLOWUP_Q": follow}}

    if tid == "tmpl_v102_steps_v0":
        body = _render_steps(lines if lines else [core])
        ex = _example_for_context(str(intent_id), dict(slots)) if bp.example_preference == EXAMPLE_YES else ""
        if ex:
            body = body + "\n" + ex
        return {"text": body, "fragment_ids": frag_ids, "slots_filled": {"EXAMPLE": ex}}

    if tid == "tmpl_v102_bullets_v0":
        body = _render_bullets(lines if lines else [core])
        ex = _example_for_context(str(intent_id), dict(slots)) if bp.example_preference == EXAMPLE_YES else ""
        if ex:
            body = body + "\n" + ex
        return {"text": body, "fragment_ids": frag_ids, "slots_filled": {"EXAMPLE": ex}}

    if tid == "tmpl_v102_transition_core_v0":
        # Deterministic transition variants keyed by tone preference.
        if bp.tone_preference == "FORMAL":
            trans = "Sobre este ponto:"
        elif bp.tone_preference == "INFORMAL":
            trans = "Sobre isso:"
        else:
            trans = "Indo direto ao ponto:"
        return {"text": trans + " " + core, "fragment_ids": frag_ids, "slots_filled": {"TRANSITION": trans}}

    if tid == "tmpl_v102_clarify_ambiguous_v0":
        # Ensure question mark exists (critics will also enforce).
        prefix = "Para eu agir com precisão:"
        text = core
        if not core.startswith(prefix):
            text = prefix + " " + core
        if "?" not in text:
            text = text.rstrip(".") + "?"
        return {"text": text, "fragment_ids": frag_ids, "slots_filled": {"CLARIFY": text}}

    if tid == "tmpl_v102_confusion_reexplain_v0":
        ex = _example_for_context(str(intent_id), dict(slots)) if bp.example_preference == EXAMPLE_YES else ""
        follow = "Faz sentido?"
        text = core
        if ex:
            text = text + "\n" + ex
        text = text + "\n" + follow
        return {"text": text, "fragment_ids": frag_ids, "slots_filled": {"EXAMPLE": ex, "FOLLOWUP_Q": follow}}

    if tid == "tmpl_v102_fallback_safe_v0":
        follow = "Pode esclarecer o que você quer dizer?"
        text = core
        if rk in {"clarify", "confirm"}:
            if "?" not in text:
                text = text.rstrip(".") + "?"
        text = text + "\n" + follow
        return {"text": text, "fragment_ids": frag_ids, "slots_filled": {"FOLLOWUP_Q": follow}}

    # Unknown template: fail-closed caller should fallback.
    return {"text": core, "fragment_ids": frag_ids, "slots_filled": slots_filled, "unknown_template": True}


def render_templates_list_text_v102(*, templates: Sequence[TemplateV102], template_stats: Dict[str, Any]) -> str:
    """
    Deterministic renderer for `templates` introspection.
    """
    lines: List[str] = ["TEMPLATES:"]
    stats = template_stats if isinstance(template_stats, dict) else {}
    for i, t in enumerate(list(templates), start=1):
        if not isinstance(t, TemplateV102):
            continue
        st = stats.get(str(t.template_id)) if isinstance(stats.get(str(t.template_id)), dict) else {}
        uses = int(st.get("uses", 0) or 0)
        avg = float(st.get("avg_score", 0.0) or 0.0)
        fail = float(st.get("fail_rate", 0.0) or 0.0)
        promo = str(st.get("promotion_state") or "candidate")
        lines.append(f"{i}) id={t.template_id} name={json.dumps(t.name, ensure_ascii=False)} uses={uses} avg_score={avg:.6f} fail_rate={fail:.6f} state={promo}")
    if len(lines) == 1:
        return "TEMPLATES: (empty)"
    return "\n".join(lines)

