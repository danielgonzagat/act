from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .intent_grammar_v92 import tokenize_user_text_v92


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


VERBOSITY_SHORT = "SHORT"
VERBOSITY_MEDIUM = "MEDIUM"
VERBOSITY_LONG = "LONG"

TONE_NEUTRAL = "NEUTRAL"
TONE_INFORMAL = "INFORMAL"
TONE_FORMAL = "FORMAL"

STRUCT_PLAIN = "PLAIN"
STRUCT_BULLETS = "BULLETS"
STRUCT_STEPS = "STEPS"

EXAMPLE_YES = "YES"
EXAMPLE_NO = "NO"

CLARIFY_LOW = "LOW"
CLARIFY_MEDIUM = "MEDIUM"
CLARIFY_HIGH = "HIGH"


def _contains_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    if not phrase:
        return False
    toks = [str(x) for x in tokens if isinstance(x, str)]
    ph = [str(x) for x in phrase if isinstance(x, str)]
    if len(toks) < len(ph):
        return False
    for i in range(0, len(toks) - len(ph) + 1):
        if toks[i : i + len(ph)] == ph:
            return True
    return False


def _canon_enum(val: Any, allowed: Sequence[str], default: str) -> str:
    v = str(val or "")
    return v if v in set([str(x) for x in allowed]) else str(default)


@dataclass(frozen=True)
class StyleProfileV102:
    verbosity_preference: str
    tone_preference: str
    structure_preference: str
    example_preference: str
    clarification_tolerance: str
    user_confusion_recent: bool
    confusion_count: int
    last_update_reason: str
    last_update_turn_id: str

    def to_dict(self) -> Dict[str, Any]:
        sem = {
            "schema_version": 102,
            "kind": "style_profile_v102",
            "verbosity_preference": str(self.verbosity_preference),
            "tone_preference": str(self.tone_preference),
            "structure_preference": str(self.structure_preference),
            "example_preference": str(self.example_preference),
            "clarification_tolerance": str(self.clarification_tolerance),
            "user_confusion_recent": bool(self.user_confusion_recent),
            "confusion_count": int(self.confusion_count),
            "last_update_reason": str(self.last_update_reason),
            "last_update_turn_id": str(self.last_update_turn_id),
        }
        sig = style_profile_sig_v102(sem)
        return dict(sem, style_profile_sig=str(sig))


def style_profile_sig_v102(profile_sem_sig: Dict[str, Any]) -> str:
    d = dict(profile_sem_sig)
    d.pop("style_profile_sig", None)
    return sha256_hex(canonical_json_dumps(d).encode("utf-8"))


def default_style_profile_v102() -> StyleProfileV102:
    return StyleProfileV102(
        verbosity_preference=VERBOSITY_MEDIUM,
        tone_preference=TONE_NEUTRAL,
        structure_preference=STRUCT_PLAIN,
        example_preference=EXAMPLE_NO,
        clarification_tolerance=CLARIFY_MEDIUM,
        user_confusion_recent=False,
        confusion_count=0,
        last_update_reason="default",
        last_update_turn_id="",
    )


def coerce_style_profile_v102(d: Any) -> StyleProfileV102:
    if not isinstance(d, dict):
        return default_style_profile_v102()
    return StyleProfileV102(
        verbosity_preference=_canon_enum(d.get("verbosity_preference"), [VERBOSITY_SHORT, VERBOSITY_MEDIUM, VERBOSITY_LONG], VERBOSITY_MEDIUM),
        tone_preference=_canon_enum(d.get("tone_preference"), [TONE_NEUTRAL, TONE_INFORMAL, TONE_FORMAL], TONE_NEUTRAL),
        structure_preference=_canon_enum(d.get("structure_preference"), [STRUCT_PLAIN, STRUCT_BULLETS, STRUCT_STEPS], STRUCT_PLAIN),
        example_preference=_canon_enum(d.get("example_preference"), [EXAMPLE_YES, EXAMPLE_NO], EXAMPLE_NO),
        clarification_tolerance=_canon_enum(d.get("clarification_tolerance"), [CLARIFY_LOW, CLARIFY_MEDIUM, CLARIFY_HIGH], CLARIFY_MEDIUM),
        user_confusion_recent=bool(d.get("user_confusion_recent", False)),
        confusion_count=int(d.get("confusion_count", 0) or 0),
        last_update_reason=str(d.get("last_update_reason") or "coerced"),
        last_update_turn_id=str(d.get("last_update_turn_id") or ""),
    )


def derive_style_profile_update_v102(
    *,
    prev: StyleProfileV102,
    user_text: str,
    user_turn_id: str,
) -> Tuple[StyleProfileV102, List[Dict[str, Any]]]:
    """
    Deterministic style update rules (explicit; no ML):
      - "curto/resumo/short/brief" => verbosity SHORT
      - "detalhado/passo a passo/step by step" => verbosity LONG + STEPS
      - "tópicos/bullets" => BULLETS
      - "exemplo/example" => example YES
      - confusion signals ("não entendi", "nao entendi", "??") => user_confusion_recent True + counter
    If multiple triggers occur in the same text, precedence is:
      detailed/steps > short > bullets > example.
    """
    toks = tokenize_user_text_v92(str(user_text))
    toks_set = set([str(t) for t in toks])

    wants_short = bool(toks_set.intersection({"curto", "resumo", "short", "brief", "conciso"}))
    wants_bullets = bool(toks_set.intersection({"bullets", "topicos", "tópicos"})) or _contains_phrase(toks, ["show", "variables"])

    wants_steps = False
    if _contains_phrase(toks, ["passo", "a", "passo"]):
        wants_steps = True
    if _contains_phrase(toks, ["step", "by", "step"]):
        wants_steps = True
    if bool(toks_set.intersection({"detalhado", "detalhes", "detailed", "detailedly", "detalha"})):
        wants_steps = True

    wants_example = bool(toks_set.intersection({"exemplo", "example"}))

    confusion = False
    if _contains_phrase(toks, ["nao", "entendi"]) or _contains_phrase(toks, ["não", "entendi"]):
        confusion = True
    if "??" in str(user_text):
        confusion = True
    if bool(toks_set.intersection({"confuso", "confusa"})):
        confusion = True

    updates: List[Dict[str, Any]] = []
    prof = prev

    # Apply precedence: steps/detailed > short > bullets > example.
    if wants_steps:
        prof = StyleProfileV102(
            verbosity_preference=VERBOSITY_LONG,
            tone_preference=str(prev.tone_preference),
            structure_preference=STRUCT_STEPS,
            example_preference=EXAMPLE_YES if wants_example else str(prev.example_preference),
            clarification_tolerance=str(prev.clarification_tolerance),
            user_confusion_recent=bool(prev.user_confusion_recent),
            confusion_count=int(prev.confusion_count),
            last_update_reason="user_requested_detailed_steps",
            last_update_turn_id=str(user_turn_id),
        )
        updates.append({"kind": "STYLE_PROFILE_UPDATE", "reason": "user_requested_detailed_steps"})
    elif wants_short:
        prof = StyleProfileV102(
            verbosity_preference=VERBOSITY_SHORT,
            tone_preference=str(prev.tone_preference),
            structure_preference=STRUCT_PLAIN,
            example_preference=str(prev.example_preference),
            clarification_tolerance=str(prev.clarification_tolerance),
            user_confusion_recent=bool(prev.user_confusion_recent),
            confusion_count=int(prev.confusion_count),
            last_update_reason="user_requested_short",
            last_update_turn_id=str(user_turn_id),
        )
        updates.append({"kind": "STYLE_PROFILE_UPDATE", "reason": "user_requested_short"})

    if wants_bullets and prof.structure_preference != STRUCT_BULLETS:
        prof = StyleProfileV102(
            verbosity_preference=str(prof.verbosity_preference),
            tone_preference=str(prof.tone_preference),
            structure_preference=STRUCT_BULLETS,
            example_preference=str(prof.example_preference),
            clarification_tolerance=str(prof.clarification_tolerance),
            user_confusion_recent=bool(prof.user_confusion_recent),
            confusion_count=int(prof.confusion_count),
            last_update_reason="user_requested_bullets",
            last_update_turn_id=str(user_turn_id),
        )
        updates.append({"kind": "STYLE_PROFILE_UPDATE", "reason": "user_requested_bullets"})

    if wants_example and prof.example_preference != EXAMPLE_YES:
        prof = StyleProfileV102(
            verbosity_preference=str(prof.verbosity_preference),
            tone_preference=str(prof.tone_preference),
            structure_preference=str(prof.structure_preference),
            example_preference=EXAMPLE_YES,
            clarification_tolerance=str(prof.clarification_tolerance),
            user_confusion_recent=bool(prof.user_confusion_recent),
            confusion_count=int(prof.confusion_count),
            last_update_reason="user_requested_example",
            last_update_turn_id=str(user_turn_id),
        )
        updates.append({"kind": "STYLE_PROFILE_UPDATE", "reason": "user_requested_example"})

    if confusion:
        prof = StyleProfileV102(
            verbosity_preference=str(prof.verbosity_preference),
            tone_preference=str(prof.tone_preference),
            structure_preference=str(prof.structure_preference),
            example_preference=str(prof.example_preference),
            clarification_tolerance=str(prof.clarification_tolerance),
            user_confusion_recent=True,
            confusion_count=int(prof.confusion_count) + 1,
            last_update_reason="user_confusion_signal",
            last_update_turn_id=str(user_turn_id),
        )
        updates.append({"kind": "STYLE_PROFILE_UPDATE", "reason": "user_confusion_signal"})

    return prof, list(updates)


def render_style_profile_text_v102(profile: StyleProfileV102) -> str:
    """
    Deterministic renderer for `style_profile` introspection.
    """
    p = profile
    lines: List[str] = []
    lines.append("STYLE_PROFILE:")
    lines.append(f"verbosity={p.verbosity_preference}")
    lines.append(f"tone={p.tone_preference}")
    lines.append(f"structure={p.structure_preference}")
    lines.append(f"example={p.example_preference}")
    lines.append(f"clarification_tolerance={p.clarification_tolerance}")
    lines.append(f"user_confusion_recent={str(bool(p.user_confusion_recent)).lower()} confusion_count={int(p.confusion_count)}")
    lines.append(f"last_update_reason={p.last_update_reason} last_update_turn_id={p.last_update_turn_id}")
    return "\n".join(lines)

