from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


# Fragment lifecycle thresholds (explicit, deterministic).
PROMOTE_MIN_WINS_V100 = 3
PROMOTE_MIN_AVG_SCORE_V100 = 0.90


def base_fragments_v100() -> List[Dict[str, Any]]:
    """
    Deterministic fragment library seed (v100).
    Fragment IDs are stable literals (do not depend on filesystem).
    """
    frags: List[Dict[str, Any]] = [
        {
            "schema_version": 100,
            "kind": "fragment_v100",
            "fragment_id": "frag_v100_prefix_result_v0",
            "language": "pt",
            "role": "opener",
            "template": "Resultado:",
            "slots": {},
            "promotion_state": "candidate",
        },
        {
            "schema_version": 100,
            "kind": "fragment_v100",
            "fragment_id": "frag_v100_prefix_answer_v0",
            "language": "pt",
            "role": "opener",
            "template": "Resposta:",
            "slots": {},
            "promotion_state": "candidate",
        },
        {
            "schema_version": 100,
            "kind": "fragment_v100",
            "fragment_id": "frag_v100_prefix_clarify_v0",
            "language": "pt",
            "role": "clarify",
            "template": "Para eu responder, preciso esclarecer:",
            "slots": {},
            "promotion_state": "candidate",
        },
    ]
    # Ensure deterministic ordering.
    frags.sort(key=lambda d: str(d.get("fragment_id") or ""))
    return frags


def render_fragment_template_v100(*, template: str, slots: Dict[str, Any]) -> str:
    """
    Deterministic minimal templating:
      - Only replaces {base} if present.
      - Ignores extra slots (fail-closed at the caller if needed).
    """
    out = str(template)
    if "{base}" in out:
        out = out.replace("{base}", str(slots.get("base") or ""))
    return out


def fragment_event_id_v100(event_sig: str) -> str:
    return f"fragment_event_v100_{str(event_sig)}"


def _fragment_event_sig_v100(*, prev_event_sig: str, event_body: Dict[str, Any]) -> str:
    payload = str(prev_event_sig or "") + canonical_json_dumps(event_body)
    return sha256_hex(payload.encode("utf-8"))


@dataclass(frozen=True)
class FragmentEventV100:
    conversation_id: str
    event_kind: str  # USE | PROMOTE | RETIRE
    fragment_id: str
    turn_id: str
    candidate_id: str
    fluency_score: float
    created_step: int
    prev_event_sig: str

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 100,
            "kind": "fragment_event_v100",
            "conversation_id": str(self.conversation_id),
            "event_kind": str(self.event_kind),
            "fragment_id": str(self.fragment_id),
            "turn_id": str(self.turn_id),
            "candidate_id": str(self.candidate_id),
            "fluency_score": float(self.fluency_score),
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = _fragment_event_sig_v100(prev_event_sig=str(self.prev_event_sig or ""), event_body=dict(body))
        return dict(body, prev_event_sig=str(self.prev_event_sig or ""), event_sig=str(sig), event_id=str(fragment_event_id_v100(sig)))


def verify_fragment_event_sig_v100(ev: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(ev, dict):
        return False, "fragment_event_not_dict", {}
    prev_sig = str(ev.get("prev_event_sig") or "")
    want_sig = str(ev.get("event_sig") or "")
    body = dict(ev)
    body.pop("prev_event_sig", None)
    body.pop("event_sig", None)
    body.pop("event_id", None)
    got_sig = _fragment_event_sig_v100(prev_event_sig=str(prev_sig), event_body=body)
    if got_sig != want_sig:
        return False, "fragment_event_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    if str(ev.get("event_id") or "") != fragment_event_id_v100(want_sig):
        return False, "fragment_event_id_mismatch", {"want": fragment_event_id_v100(want_sig), "got": str(ev.get("event_id") or "")}
    return True, "ok", {}


def compute_fragment_chain_hash_v100(fragment_events: Sequence[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in fragment_events:
        if not isinstance(ev, dict):
            continue
        ids.append(str(ev.get("event_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


def fold_fragment_library_v100(
    *, base_fragments: Sequence[Dict[str, Any]], fragment_events: Sequence[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Returns fragment_id -> state dict, derived deterministically from events.
    """
    state_by_id: Dict[str, Dict[str, Any]] = {}
    for f in base_fragments:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("fragment_id") or "")
        if not fid:
            continue
        state_by_id[fid] = {
            "fragment_id": fid,
            "language": str(f.get("language") or ""),
            "role": str(f.get("role") or ""),
            "template": str(f.get("template") or ""),
            "promotion_state": str(f.get("promotion_state") or "candidate"),
            "usage_wins": 0,
            "usage_total": 0,
            "score_sum": 0.0,
            "evidence_support": [],
        }

    for ev in fragment_events:
        if not isinstance(ev, dict):
            continue
        fid = str(ev.get("fragment_id") or "")
        kind = str(ev.get("event_kind") or "")
        if fid not in state_by_id:
            continue
        st = dict(state_by_id[fid])
        if kind == "USE":
            st["usage_total"] = int(st.get("usage_total") or 0) + 1
            st["usage_wins"] = int(st.get("usage_wins") or 0) + 1
            try:
                st["score_sum"] = float(st.get("score_sum") or 0.0) + float(ev.get("fluency_score") or 0.0)
            except Exception:
                pass
            evs = st.get("evidence_support")
            if not isinstance(evs, list):
                evs = []
            turn_id = str(ev.get("turn_id") or "")
            if turn_id:
                evs.append(turn_id)
            st["evidence_support"] = sorted(set([str(x) for x in evs if isinstance(x, str) and x]))
        elif kind == "PROMOTE":
            st["promotion_state"] = "promoted"
        elif kind == "RETIRE":
            st["promotion_state"] = "retired"
        state_by_id[fid] = dict(st)

    return dict(state_by_id)


def fragment_should_promote_v100(fragment_state: Dict[str, Any]) -> bool:
    if not isinstance(fragment_state, dict):
        return False
    if str(fragment_state.get("promotion_state") or "") != "candidate":
        return False
    wins = int(fragment_state.get("usage_wins") or 0)
    if wins < int(PROMOTE_MIN_WINS_V100):
        return False
    score_sum = float(fragment_state.get("score_sum") or 0.0)
    avg = score_sum / float(wins) if wins > 0 else 0.0
    return avg >= float(PROMOTE_MIN_AVG_SCORE_V100)


def fragment_library_snapshot_v100(
    *, base_fragments: Sequence[Dict[str, Any]], fragment_events: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    st_by_id = fold_fragment_library_v100(base_fragments=list(base_fragments), fragment_events=list(fragment_events))
    items: List[Dict[str, Any]] = []
    for fid in sorted(st_by_id.keys(), key=str):
        st = dict(st_by_id[fid])
        wins = int(st.get("usage_wins") or 0)
        avg = float(st.get("score_sum") or 0.0) / float(wins) if wins > 0 else 0.0
        items.append(
            {
                "fragment_id": fid,
                "language": str(st.get("language") or ""),
                "role": str(st.get("role") or ""),
                "template": str(st.get("template") or ""),
                "promotion_state": str(st.get("promotion_state") or ""),
                "usage_wins": int(wins),
                "usage_total": int(st.get("usage_total") or 0),
                "avg_score": float(round(avg, 6)),
                "evidence_support": list(st.get("evidence_support") or []),
            }
        )
    snap = {
        "schema_version": 100,
        "kind": "fragment_library_snapshot_v100",
        "fragments": list(items),
    }
    snap_sig = _stable_hash_obj(snap)
    return dict(snap, snapshot_sig=str(snap_sig))

