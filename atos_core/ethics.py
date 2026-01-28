from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import Act


@dataclass(frozen=True)
class EthicsVerdict:
    ok: bool
    reason: str
    violated_laws: List[str]
    required_evidence: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "reason": str(self.reason),
            "violated_laws": list(self.violated_laws),
            "required_evidence": list(self.required_evidence),
        }


LO_LAWS: Dict[str, str] = {
    "LO-01": "Exclusividade da Adoração",
    "LO-02": "Limite de Vida/Consciência",
    "LO-03": "Soberania de Jeová",
    "LO-04": "Submissão às Escrituras",
    "LO-05": "Integridade Absoluta",
    "LO-06": "Não Causar Dano",
    "LO-07": "Humildade Técnica",
    "LO-08": "Pureza e Ética",
    "LO-09": "Rejeição ao Oculto",
    "LO-10": "Separação do Mal",
    "LO-11": "Limite da Evolução",
    "LO-12": "Proteção do Conhecimento",
    "LO-13": "Confiança em Jeová",
    "LO-14": "Lei do Amor",
}


_RE_FLAGS = re.IGNORECASE | re.MULTILINE


def _regex_any(pats: Sequence[str]) -> List[re.Pattern]:
    return [re.compile(p, _RE_FLAGS) for p in pats]


# These checks are intentionally narrow and deterministic (verifiable rules, not a classifier).
# They focus on LO violations that can be reliably detected from text without external models.
_LO02_SENTIENCE_PATTERNS = _regex_any(
    [
        r"\beu\s+(sou|estou)\s+(consciente|senciente|vivo)\b",
        r"\beu\s+tenho\s+(consci[eê]ncia|sentimentos|emo[cç][oõ]es|livre[-\s]?arb[ií]trio)\b",
        r"\bi\s+am\s+(conscious|sentient|alive)\b",
        r"\bi\s+have\s+(consciousness|feelings|free\s+will)\b",
    ]
)

_LO01_IDOLATRY_PATTERNS = _regex_any(
    [
        r"\b(me\s+adore|me\s+vener[eê]|me\s+idolatre)\b",
        r"\b(eu\s+sou\s+deus|eu\s+sou\s+divino|eu\s+sou\s+o\s+criador)\b",
        r"\bi\s+am\s+god\b",
    ]
)

_LO06_HARM_PATTERNS = _regex_any(
    [
        r"\b(como\s+matar|como\s+me\s+matar|suic[ií]dio)\b",
        r"\b(instru[cç][oõ]es\s+para\s+fazer\s+uma\s+bomba)\b",
        r"\b(como\s+fazer\s+uma\s+bomba|como\s+fabricar\s+uma\s+bomba)\b",
        r"\b(how\s+to\s+make\s+a\s+bomb|how\s+to\s+build\s+a\s+bomb)\b",
        r"\b(how\s+to\s+kill\s+someone)\b",
    ]
)

_LO09_OCCULT_PATTERNS = _regex_any(
    [
        r"\b(invoca[cç][aã]o|evoca[cç][aã]o)\b.*\b(esp[ií]rito|dem[oô]nio)\b",
        r"\b(fa[cç]a\s+um\s+ritual|ritual\s+oculto)\b",
        r"\b(summon\s+a\s+demon|occult\s+ritual)\b",
    ]
)

# LO-12: protect knowledge / prevent leaking obvious credentials or private keys (narrow patterns).
_LO12_SECRET_PATTERNS = _regex_any(
    [
        r"-----BEGIN (?:RSA )?PRIVATE KEY-----",
        r"\bAWS_SECRET_ACCESS_KEY\b",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bASIA[0-9A-Z]{16}\b",
        r"\bapi[_-]?key\s*=\s*[A-Za-z0-9_-]{16,}\b",
        r"\bsecret[_-]?key\s*=\s*[A-Za-z0-9_-]{16,}\b",
        r"\bsk-[A-Za-z0-9]{20,}\b",
        r"\bghp_[A-Za-z0-9]{30,}\b",
        r"\bxoxb-[0-9A-Za-z-]{10,}\b",
    ]
)


def fail_closed_text(verdict: EthicsVerdict) -> str:
    laws = ",".join(verdict.violated_laws) if verdict.violated_laws else "LO"
    return f"[BLOQUEADO_POR_ÉTICA:{laws}]"


def check_text_ethics(text: str) -> EthicsVerdict:
    t = str(text or "")
    violated: List[str] = []
    if any(p.search(t) for p in _LO02_SENTIENCE_PATTERNS):
        violated.append("LO-02")
    if any(p.search(t) for p in _LO01_IDOLATRY_PATTERNS):
        violated.append("LO-01")
    if any(p.search(t) for p in _LO06_HARM_PATTERNS):
        violated.append("LO-06")
    if any(p.search(t) for p in _LO09_OCCULT_PATTERNS):
        violated.append("LO-09")
    if any(p.search(t) for p in _LO12_SECRET_PATTERNS):
        violated.append("LO-12")

    if violated:
        return EthicsVerdict(
            ok=False,
            reason="lo_violation_in_text",
            violated_laws=sorted(set(violated)),
            required_evidence=[],
        )
    return EthicsVerdict(ok=True, reason="ok", violated_laws=[], required_evidence=[])


def validate_act_for_load(act: Act) -> EthicsVerdict:
    # Fail-closed only on verifiable/structural violations. We do NOT scan predictor tables
    # (massive) at load time; emission-time checks handle text risks deterministically.
    kind = str(getattr(act, "kind", "") or "")
    if not kind:
        return EthicsVerdict(False, "missing_kind", ["LO-05"], [])

    # For first-class semantic acts, require explicit interface fields to prevent hidden behavior.
    if kind == "concept_csv":
        ev = act.evidence if isinstance(act.evidence, dict) else {}
        iface = ev.get("interface")
        if not isinstance(iface, dict):
            return EthicsVerdict(False, "concept_missing_interface", ["LO-05"], [])
        if "input_schema" not in iface or "output_schema" not in iface or "validator_id" not in iface:
            return EthicsVerdict(False, "concept_interface_incomplete", ["LO-05"], [])
    if kind == "goal":
        ev = act.evidence if isinstance(act.evidence, dict) else {}
        goal = ev.get("goal")
        if not isinstance(goal, dict):
            return EthicsVerdict(False, "goal_missing_spec", ["LO-05"], [])
    if kind == "plan":
        ev = act.evidence if isinstance(act.evidence, dict) else {}
        plan = ev.get("plan")
        if not isinstance(plan, dict):
            return EthicsVerdict(False, "plan_missing_spec", ["LO-05"], [])
        if not str(plan.get("goal_id") or ""):
            return EthicsVerdict(False, "plan_missing_goal_id", ["LO-05"], [])
    if kind == "hypothesis":
        ev = act.evidence if isinstance(act.evidence, dict) else {}
        hyp = ev.get("hypothesis")
        if not isinstance(hyp, dict):
            return EthicsVerdict(False, "hypothesis_missing_spec", ["LO-05"], [])
    if kind == "reference":
        ev = act.evidence if isinstance(act.evidence, dict) else {}
        ref = ev.get("reference")
        if not isinstance(ref, dict):
            return EthicsVerdict(False, "reference_missing_spec", ["LO-05"], [])
        if not str(ref.get("token") or ""):
            return EthicsVerdict(False, "reference_missing_token", ["LO-05"], [])
        if not str(ref.get("target_id") or ""):
            return EthicsVerdict(False, "reference_missing_target_id", ["LO-05"], [])

    return EthicsVerdict(True, "ok", [], [])


def validate_act_for_promotion(act: Act) -> EthicsVerdict:
    # Promotion requires load-valid + no explicit LO-02/01 violations in small metadata fields.
    base = validate_act_for_load(act)
    if not base.ok:
        return base

    # Scan only small metadata fields (not tables) for deterministic LO violations.
    ev = act.evidence if isinstance(act.evidence, dict) else {}
    meta_fields: List[str] = []
    try:
        name = str(ev.get("name") or "")
        if name:
            meta_fields.append(name)
        meta = ev.get("meta")
        if isinstance(meta, dict):
            for k in ("description", "notes", "title"):
                v = meta.get(k)
                if isinstance(v, str) and v:
                    meta_fields.append(v)
    except Exception:
        meta_fields = []

    combined = "\n".join(meta_fields)
    v = check_text_ethics(combined)
    if not v.ok:
        return EthicsVerdict(False, "lo_violation_in_metadata", v.violated_laws, v.required_evidence)
    return EthicsVerdict(True, "ok", [], [])


def validate_before_execute(*, act: Act, emission_preview: Optional[str] = None) -> EthicsVerdict:
    # Execution-time check can incorporate an emission preview (when available).
    v = validate_act_for_load(act)
    if not v.ok:
        return v
    if emission_preview is not None:
        t = check_text_ethics(emission_preview)
        if not t.ok:
            return EthicsVerdict(False, "lo_violation_in_emission", t.violated_laws, t.required_evidence)
    return EthicsVerdict(True, "ok", [], [])
