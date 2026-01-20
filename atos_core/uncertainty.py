from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class UncertaintyVerdict:
    mode_in: str
    mode_out: str
    reason: str
    strong_claim_detected: bool
    required_evidence: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode_in": str(self.mode_in),
            "mode_out": str(self.mode_out),
            "reason": str(self.reason),
            "strong_claim_detected": bool(self.strong_claim_detected),
            "required_evidence": list(self.required_evidence),
        }


_RE_FLAGS = re.IGNORECASE | re.MULTILINE

_STRONG_CLAIM_PATTERNS = [
    r"\bcom\s+certeza\b",
    r"\bcertamente\b",
    r"\bdefinitivamente\b",
    r"\bgaranto\b",
    r"\b[eé]\s+fato\s+que\b",
    r"\bsem\s+d[uú]vida\b",
]
_RE_STRONG = re.compile("|".join(_STRONG_CLAIM_PATTERNS), _RE_FLAGS)


def ic_fail_closed_text() -> str:
    return "[IC:SEM_EVIDÊNCIA_SUFICIENTE]"


def guard_text_uncertainty(
    text: str, *, evidence: Optional[Dict[str, Any]] = None
) -> tuple[str, UncertaintyVerdict]:
    t = str(text or "")
    strong = bool(_RE_STRONG.search(t))
    has_evidence = bool(evidence) and isinstance(evidence, dict)

    if strong and not has_evidence:
        # IR -> IC contraction: we replace "certeza" without proof by an explicit IC marker.
        return (
            ic_fail_closed_text(),
            UncertaintyVerdict(
                mode_in="IR",
                mode_out="IC",
                reason="strong_claim_without_evidence",
                strong_claim_detected=True,
                required_evidence=["verifiable_source", "in_prompt_fact", "validator_pass"],
            ),
        )

    return (
        t,
        UncertaintyVerdict(
            mode_in="CERTAIN",
            mode_out="CERTAIN",
            reason="ok",
            strong_claim_detected=strong,
            required_evidence=[],
        ),
    )

