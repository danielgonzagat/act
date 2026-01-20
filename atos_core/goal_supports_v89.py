from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(canonical_json_dumps(row))
        f.write("\n")


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return iter(())
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_chained_jsonl_v89(path: str, entry: Dict[str, Any], *, prev_hash: Optional[str]) -> str:
    body = dict(entry)
    body["prev_hash"] = prev_hash
    entry_hash = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    body["entry_hash"] = entry_hash
    _append_jsonl(path, body)
    return entry_hash


def verify_chained_jsonl_v89(path: str) -> bool:
    prev: Optional[str] = None
    for row in _read_jsonl(path):
        row = dict(row)
        entry_hash = row.pop("entry_hash", None)
        if row.get("prev_hash") != prev:
            return False
        expected = sha256_hex(canonical_json_dumps(row).encode("utf-8"))
        if expected != entry_hash:
            return False
        prev = str(entry_hash)
    return True


@dataclass(frozen=True)
class SupportClaimV89:
    goal_id: str
    prior_success: float = 0.5
    prior_strength: int = 2
    prior_cost: float = 1.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": str(self.goal_id),
            "prior_success": float(self.prior_success),
            "prior_strength": int(self.prior_strength),
            "prior_cost": float(self.prior_cost),
            "note": str(self.note),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Optional["SupportClaimV89"]:
        if not isinstance(d, dict):
            return None
        goal_id = str(d.get("goal_id") or "")
        if not goal_id:
            return None
        ps = _as_float(d.get("prior_success", 0.5), 0.5)
        if ps < 0.0:
            ps = 0.0
        if ps > 1.0:
            ps = 1.0
        strength = _as_int(d.get("prior_strength", 2), 2)
        if strength < 1:
            strength = 1
        pc = _as_float(d.get("prior_cost", 1.0), 1.0)
        if pc <= 0.0:
            pc = 1.0
        note = str(d.get("note") or "")
        return SupportClaimV89(
            goal_id=str(goal_id),
            prior_success=float(ps),
            prior_strength=int(strength),
            prior_cost=float(pc),
            note=str(note),
        )


@dataclass(frozen=True)
class SupportStatsV89:
    attempts: int
    successes: int
    failures: int
    cost_sum: float
    last_step: int
    expected_success: float
    expected_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempts": int(self.attempts),
            "successes": int(self.successes),
            "failures": int(self.failures),
            "cost_sum": float(self.cost_sum),
            "last_step": int(self.last_step),
            "last_ts": int(self.last_step),
            "expected_success": float(self.expected_success),
            "expected_cost": float(self.expected_cost),
        }


def posterior_mean_beta_v89(*, prior_success: float, prior_strength: int, successes: int, attempts: int) -> float:
    ps = float(prior_success)
    if ps < 0.0:
        ps = 0.0
    if ps > 1.0:
        ps = 1.0
    strength = int(prior_strength)
    if strength < 1:
        strength = 1
    s = int(successes)
    n = int(attempts)
    if n < 0:
        n = 0
    if s < 0:
        s = 0
    if s > n:
        s = n
    alpha0 = ps * float(strength)
    beta0 = (1.0 - ps) * float(strength)
    denom = alpha0 + beta0 + float(n)
    if denom <= 0.0:
        return 0.5
    return (alpha0 + float(s)) / denom


def expected_cost_smoothing_v89(*, prior_cost: float, prior_strength: int, cost_sum: float, attempts: int) -> float:
    pc = float(prior_cost)
    if pc <= 0.0:
        pc = 1.0
    strength = int(prior_strength)
    if strength < 1:
        strength = 1
    n = int(attempts)
    if n < 0:
        n = 0
    cs = float(cost_sum)
    if cs < 0.0:
        cs = 0.0
    denom = float(strength + n)
    if denom <= 0.0:
        return pc
    return (pc * float(strength) + cs) / denom


def support_claims_from_concept_act_v89(concept_act) -> List[SupportClaimV89]:
    """
    Load supports(G) declarations from concept metadata without affecting execution.
    Convention: concept_act.evidence["supports_goals_v89"] = [SupportClaimV89 dict...]
    """
    if concept_act is None:
        return []
    ev = concept_act.evidence if isinstance(getattr(concept_act, "evidence", None), dict) else {}
    raw = ev.get("supports_goals_v89")
    if not isinstance(raw, list):
        return []
    out: List[SupportClaimV89] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        c = SupportClaimV89.from_dict(r)
        if c is None:
            continue
        out.append(c)
    # Stable: sort by (goal_id, claim_hash).
    out.sort(key=lambda c: (str(c.goal_id), _stable_hash_obj(c.to_dict())))
    return out


def list_supporting_concepts_for_goal_v89(*, store, goal_id: str) -> List[Tuple[str, SupportClaimV89]]:
    gid = str(goal_id or "")
    if not gid:
        return []
    acts = []
    try:
        acts = store.concept_acts()
    except Exception:
        acts = []
    out: List[Tuple[str, SupportClaimV89]] = []
    for act in acts:
        act_id = str(getattr(act, "id", "") or "")
        if not act_id:
            continue
        for c in support_claims_from_concept_act_v89(act):
            if str(c.goal_id) == gid:
                out.append((act_id, c))
    out.sort(key=lambda t: (str(t[0]), _stable_hash_obj(t[1].to_dict())))
    return out


def make_goal_support_evidence_event_v89(
    *,
    step: int,
    goal_id: str,
    concept_key: str,
    attempt_id: str,
    ok: bool,
    cost_used: float,
    note: str = "",
) -> Dict[str, Any]:
    return {
        "kind": "goal_support_evidence_v89",
        "time": deterministic_iso(step=int(step)),
        "ts": int(step),
        "step": int(step),
        "goal_id": str(goal_id),
        "concept_key": str(concept_key),
        "attempt_id": str(attempt_id),
        "ok": bool(ok),
        "cost_used": float(cost_used),
        "note": str(note or ""),
    }


def fold_support_stats_v89(
    *,
    events: Sequence[Dict[str, Any]],
    goal_id: str,
    concept_key: str,
    claim: SupportClaimV89,
) -> SupportStatsV89:
    gid = str(goal_id or "")
    cid = str(concept_key or "")
    attempts = 0
    successes = 0
    cost_sum = 0.0
    last_step = -1
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if str(ev.get("kind") or "") != "goal_support_evidence_v89":
            continue
        if str(ev.get("goal_id") or "") != gid:
            continue
        if str(ev.get("concept_key") or "") != cid:
            continue
        attempts += 1
        if bool(ev.get("ok", False)):
            successes += 1
        cost_sum += _as_float(ev.get("cost_used", 0.0), 0.0)
        try:
            last_step = max(last_step, int(ev.get("step", -1) or -1))
        except Exception:
            pass
    failures = attempts - successes
    expected_success = posterior_mean_beta_v89(
        prior_success=float(claim.prior_success),
        prior_strength=int(claim.prior_strength),
        successes=int(successes),
        attempts=int(attempts),
    )
    expected_cost = expected_cost_smoothing_v89(
        prior_cost=float(claim.prior_cost),
        prior_strength=int(claim.prior_strength),
        cost_sum=float(cost_sum),
        attempts=int(attempts),
    )
    return SupportStatsV89(
        attempts=int(attempts),
        successes=int(successes),
        failures=int(failures),
        cost_sum=float(cost_sum),
        last_step=int(last_step),
        expected_success=float(expected_success),
        expected_cost=float(expected_cost),
    )
