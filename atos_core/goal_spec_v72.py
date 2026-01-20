from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .act import canonical_json_dumps, sha256_hex


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


_SPEC_RE = re.compile(r"^\s*V72_SPEC\s*=\s*(\{.*\})\s*$")


@dataclass(frozen=True)
class GoalSpecV72:
    goal_kind: str
    bindings: Dict[str, Any]
    output_key: str
    expected: Any
    validator_id: str = "text_exact"
    created_step: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_kind", str(self.goal_kind or ""))
        object.__setattr__(self, "output_key", str(self.output_key or ""))
        object.__setattr__(self, "validator_id", str(self.validator_id or ""))
        object.__setattr__(self, "created_step", int(self.created_step or 0))
        b = self.bindings if isinstance(self.bindings, dict) else {}
        try:
            b2 = copy.deepcopy(b)
        except Exception:
            b2 = dict(b)
        object.__setattr__(self, "bindings", b2 if isinstance(b2, dict) else {})

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "goal_kind": str(self.goal_kind),
            "bindings": {str(k): self.bindings.get(k) for k in sorted(self.bindings.keys(), key=str)},
            "output_key": str(self.output_key),
            "expected": self.expected,
            "validator_id": str(self.validator_id),
        }

    def goal_sig(self) -> str:
        return _stable_hash_obj(self.to_canonical_dict())

    def goal_id(self) -> str:
        return f"goal_v72_{self.goal_sig()}"


def parse_goal_spec_from_prompt(prompt: str) -> Tuple[Optional[GoalSpecV72], str]:
    """
    Spec-first-line parser:
      first line must be: V72_SPEC=<json>
    Returns (GoalSpecV72|None, reason).
    """
    lines = str(prompt or "").splitlines()
    if not lines:
        return None, "empty_prompt"
    m = _SPEC_RE.match(lines[0].strip())
    if not m:
        return None, "missing_v72_spec_first_line"
    raw = m.group(1)
    try:
        obj = json.loads(raw)
    except Exception:
        return None, "bad_json"
    if not isinstance(obj, dict):
        return None, "spec_not_dict"

    goal_kind = str(obj.get("goal_kind") or "goal_v72")
    bindings = obj.get("bindings") if isinstance(obj.get("bindings"), dict) else {}
    output_key = str(obj.get("output_key") or "")
    expected = obj.get("expected")
    validator_id = str(obj.get("validator_id") or "text_exact")
    if not output_key:
        return None, "missing_output_key"
    return GoalSpecV72(
        goal_kind=goal_kind,
        bindings=dict(bindings),
        output_key=output_key,
        expected=expected,
        validator_id=validator_id,
        created_step=int(obj.get("created_step", 0) or 0),
    ), "ok"

