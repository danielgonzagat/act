from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .discourse_templates_v102 import TemplateV102


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def style_event_id_v102(event_sig: str) -> str:
    return f"style_event_v102_{str(event_sig)}"


def _style_event_sig_v102(*, prev_event_sig: str, event_body: Dict[str, Any]) -> str:
    payload = str(prev_event_sig or "") + canonical_json_dumps(event_body)
    return sha256_hex(payload.encode("utf-8"))


@dataclass(frozen=True)
class StyleEventV102:
    conversation_id: str
    turn_id: str
    turn_index: int
    event_kind: str
    style_profile_before: Dict[str, Any]
    style_profile_after: Dict[str, Any]
    candidates_topk: List[Dict[str, Any]]
    selected_candidate_id: str
    selected_template_id: str
    selected_ok: bool
    selection: Dict[str, Any]
    cause_ids: Dict[str, Any]
    created_step: int
    prev_event_sig: str

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 102,
            "kind": "style_event_v102",
            "conversation_id": str(self.conversation_id),
            "turn_id": str(self.turn_id),
            "turn_index": int(self.turn_index),
            "event_kind": str(self.event_kind),
            "style_profile_before": dict(self.style_profile_before) if isinstance(self.style_profile_before, dict) else {},
            "style_profile_after": dict(self.style_profile_after) if isinstance(self.style_profile_after, dict) else {},
            "candidates_topk": [dict(x) for x in self.candidates_topk if isinstance(x, dict)],
            "selected_candidate_id": str(self.selected_candidate_id),
            "selected_template_id": str(self.selected_template_id),
            "selected_ok": bool(self.selected_ok),
            "selection": dict(self.selection) if isinstance(self.selection, dict) else {},
            "cause_ids": dict(self.cause_ids) if isinstance(self.cause_ids, dict) else {},
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = _style_event_sig_v102(prev_event_sig=str(self.prev_event_sig or ""), event_body=dict(body))
        return dict(body, prev_event_sig=str(self.prev_event_sig or ""), event_sig=str(sig), event_id=str(style_event_id_v102(sig)))


def verify_style_event_sig_v102(ev: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    if not isinstance(ev, dict):
        return False, "style_event_not_dict", {}
    prev_sig = str(ev.get("prev_event_sig") or "")
    want_sig = str(ev.get("event_sig") or "")
    body = dict(ev)
    body.pop("prev_event_sig", None)
    body.pop("event_sig", None)
    body.pop("event_id", None)
    got_sig = _style_event_sig_v102(prev_event_sig=str(prev_sig), event_body=body)
    if got_sig != want_sig:
        return False, "style_event_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    if str(ev.get("event_id") or "") != style_event_id_v102(want_sig):
        return False, "style_event_id_mismatch", {"want": style_event_id_v102(want_sig), "got": str(ev.get("event_id") or "")}
    return True, "ok", {}


def compute_style_chain_hash_v102(style_events: Sequence[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in style_events:
        if not isinstance(ev, dict):
            continue
        ids.append(str(ev.get("event_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


def fold_template_stats_v102(
    *,
    templates: Sequence[TemplateV102],
    style_events: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Deterministic stats derived from STYLE_CHOSEN events.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    for t in templates:
        if not isinstance(t, TemplateV102):
            continue
        stats[str(t.template_id)] = {
            "template_id": str(t.template_id),
            "uses": 0,
            "ok_uses": 0,
            "fail_uses": 0,
            "score_sum": 0.0,
            "avg_score": 0.0,
            "fail_rate": 0.0,
            "promotion_state": "candidate",
            "mdl_cost_bytes": int(len(canonical_json_dumps(t.to_dict()).encode("utf-8"))),
            "mdl_savings_estimate_bytes": 0,
        }

    for ev in style_events:
        if not isinstance(ev, dict):
            continue
        if str(ev.get("event_kind") or "") != "STYLE_CHOSEN":
            continue
        tid = str(ev.get("selected_template_id") or "")
        if tid not in stats:
            continue
        st = dict(stats[tid])
        st["uses"] = int(st.get("uses") or 0) + 1
        if bool(ev.get("selected_ok", False)):
            st["ok_uses"] = int(st.get("ok_uses") or 0) + 1
        else:
            st["fail_uses"] = int(st.get("fail_uses") or 0) + 1
        # Record the selected candidate's score (if present).
        cands = ev.get("candidates_topk") if isinstance(ev.get("candidates_topk"), list) else []
        sel_id = str(ev.get("selected_candidate_id") or "")
        score = None
        for c in cands:
            if isinstance(c, dict) and str(c.get("candidate_id") or "") == sel_id:
                try:
                    score = float(c.get("fluency_score") or 0.0)
                except Exception:
                    score = 0.0
                break
        if score is not None:
            st["score_sum"] = float(st.get("score_sum") or 0.0) + float(score)
        stats[tid] = dict(st)

    # Finalize derived stats and promotion state.
    for tid in sorted(stats.keys(), key=str):
        st = dict(stats[tid])
        uses = int(st.get("uses") or 0)
        ok = int(st.get("ok_uses") or 0)
        fail = int(st.get("fail_uses") or 0)
        avg = float(st.get("score_sum") or 0.0) / float(ok) if ok > 0 else 0.0
        fail_rate = float(fail) / float(uses) if uses > 0 else 0.0
        st["avg_score"] = float(round(avg, 6))
        st["fail_rate"] = float(round(fail_rate, 6))

        promo = "candidate"
        # Minimal deterministic pressure (MVP):
        # promote if used enough and performs well; demote/retire if consistently bad.
        if uses >= 3 and avg >= 0.85 and fail_rate <= 0.34:
            promo = "promoted"
        if uses >= 3 and (avg <= 0.30 or fail_rate >= 0.67):
            promo = "retired"
        st["promotion_state"] = str(promo)
        stats[tid] = dict(st)

    return dict(stats)


def template_library_snapshot_v102(
    *,
    templates: Sequence[TemplateV102],
    style_events: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    stats = fold_template_stats_v102(templates=list(templates), style_events=list(style_events))
    items: List[Dict[str, Any]] = []
    for t in templates:
        if not isinstance(t, TemplateV102):
            continue
        td = t.to_dict()
        st = stats.get(str(t.template_id)) if isinstance(stats.get(str(t.template_id)), dict) else {}
        items.append(
            {
                "template_id": str(t.template_id),
                "name": str(t.name),
                "roles": list(td.get("roles") or []),
                "constraints": dict(td.get("constraints") or {}),
                "slots": list(td.get("slots") or []),
                "template_sig": str(td.get("template_sig") or ""),
                "mdl_cost_bytes": int(td.get("mdl_cost_bytes") or 0),
                "stats": dict(st),
            }
        )
    snap = {"schema_version": 102, "kind": "template_library_snapshot_v102", "templates": list(items)}
    snap_sig = _stable_hash_obj(snap)
    return dict(snap, snapshot_sig=str(snap_sig))


def sha256_file_bytes_v102(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

