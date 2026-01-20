from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .act import canonical_json_dumps, sha256_hex
from .concept_model_v103 import csg_name_v103, rule_features_from_csg_v103


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _canon_list_str(xs: Any) -> List[str]:
    if not isinstance(xs, list):
        return []
    out: List[str] = []
    for x in xs:
        s = str(x)
        if s:
            out.append(s)
    return sorted(out, key=str)


def _canon_status(status: str) -> str:
    s = str(status or "")
    if s not in {"ALIVE", "DEPRECATED", "DEAD"}:
        return "ALIVE"
    return s


def fold_concept_ledger_v103(concept_events: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Replays the concept ledger into a deterministic registry:
      - concepts_by_id: concept_id -> csg
      - csv_by_id: concept_id -> csv_state
      - name_index: name -> concept_id (latest created; tie-break lexicographic)
      - feedback_by_name: name -> {pos_sigs, neg_sigs, pos_text_norm_by_sig, neg_text_norm_by_sig}

    NOTE: This is intentionally simple and fail-closed; verifier ensures integrity.
    """
    concepts_by_id: Dict[str, Dict[str, Any]] = {}
    csv_by_id: Dict[str, Dict[str, Any]] = {}
    name_index: Dict[str, str] = {}

    feedback_by_name: Dict[str, Dict[str, Any]] = {}

    def _ensure_csv(concept_id: str) -> Dict[str, Any]:
        st = csv_by_id.get(concept_id)
        if isinstance(st, dict):
            return st
        st0 = {
            "status": "ALIVE",
            "usage_count": 0,
            "evidence_pos_count": 0,
            "evidence_neg_count": 0,
            "pos_example_sigs": [],
            "neg_example_sigs": [],
            "toc_pass": 0,
            "toc_fail": 0,
            "mdl_baseline_bits": 0,
            "mdl_model_bits": 0,
            "mdl_data_bits": 0,
            "mdl_delta_bits": 0,
            "seed_tokens": [],
            "created_at_turn_id": "",
            "last_used_turn_id": "",
        }
        csv_by_id[concept_id] = dict(st0)
        return csv_by_id[concept_id]

    for ev in concept_events:
        if not isinstance(ev, dict):
            continue
        et = str(ev.get("type") or "")
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        turn_id = str(ev.get("turn_id") or "")

        if et == "CONCEPT_CREATE":
            csg = payload.get("csg") if isinstance(payload.get("csg"), dict) else {}
            cid = str(csg.get("concept_id") or payload.get("concept_id") or "")
            if not cid:
                continue
            concepts_by_id[cid] = dict(csg)
            nm = str(csg_name_v103(csg))
            if nm:
                prev = str(name_index.get(nm) or "")
                if not prev or str(cid) < prev:
                    # Prefer deterministic canonical smallest id per name.
                    name_index[nm] = str(cid)
            st = _ensure_csv(str(cid))
            st["created_at_turn_id"] = str(st.get("created_at_turn_id") or turn_id)

        elif et == "CONCEPT_FEEDBACK":
            name = str(payload.get("name") or "")
            polarity = str(payload.get("polarity") or "")
            ex_sig = str(payload.get("example_sig") or "")
            ex_norm = str(payload.get("example_text_norm") or "")
            if not name or not ex_sig:
                continue
            fb = feedback_by_name.setdefault(name, {"pos_sigs": set(), "neg_sigs": set(), "pos_text": {}, "neg_text": {}})
            if polarity == "+":
                fb["pos_sigs"].add(ex_sig)
                if ex_norm:
                    fb["pos_text"][ex_sig] = ex_norm
            elif polarity == "-":
                fb["neg_sigs"].add(ex_sig)
                if ex_norm:
                    fb["neg_text"][ex_sig] = ex_norm
            cid = str(payload.get("concept_id") or "")
            if cid:
                st = _ensure_csv(cid)
                if polarity == "+" and ex_sig not in set(st.get("pos_example_sigs") or []):
                    st["pos_example_sigs"] = _canon_list_str(list((st.get("pos_example_sigs") or []) + [ex_sig]))
                    st["evidence_pos_count"] = int(st.get("evidence_pos_count") or 0) + 1
                if polarity == "-" and ex_sig not in set(st.get("neg_example_sigs") or []):
                    st["neg_example_sigs"] = _canon_list_str(list((st.get("neg_example_sigs") or []) + [ex_sig]))
                    st["evidence_neg_count"] = int(st.get("evidence_neg_count") or 0) + 1

        elif et == "CONCEPT_INDUCE":
            cid = str(payload.get("concept_id") or "")
            if not cid:
                continue
            st = _ensure_csv(cid)
            st["mdl_baseline_bits"] = int(payload.get("mdl_baseline_bits") or 0)
            st["mdl_model_bits"] = int(payload.get("mdl_model_bits") or 0)
            st["mdl_data_bits"] = int(payload.get("mdl_data_bits") or 0)
            st["mdl_delta_bits"] = int(payload.get("mdl_delta_bits") or 0)
            st["seed_tokens"] = _canon_list_str(payload.get("seed_tokens"))

        elif et == "CONCEPT_MATCH":
            cid = str(payload.get("concept_id") or "")
            if not cid:
                continue
            st = _ensure_csv(cid)
            st["usage_count"] = int(st.get("usage_count") or 0) + 1
            st["last_used_turn_id"] = str(turn_id)

        elif et == "CONCEPT_TOC_PASS":
            cid = str(payload.get("concept_id") or "")
            if not cid:
                continue
            st = _ensure_csv(cid)
            st["toc_pass"] = int(st.get("toc_pass") or 0) + 1

        elif et == "CONCEPT_TOC_FAIL":
            cid = str(payload.get("concept_id") or "")
            if not cid:
                continue
            st = _ensure_csv(cid)
            st["toc_fail"] = int(st.get("toc_fail") or 0) + 1

        elif et == "CONCEPT_PRUNE":
            cid = str(payload.get("concept_id") or "")
            if not cid:
                continue
            st = _ensure_csv(cid)
            st["status"] = _canon_status(str(payload.get("status") or "DEAD"))

    # Canonicalize sets in feedback.
    fb_out: Dict[str, Any] = {}
    for name in sorted(feedback_by_name.keys(), key=str):
        fb = feedback_by_name.get(name) or {}
        fb_out[name] = {
            "pos_sigs": sorted([str(x) for x in list(fb.get("pos_sigs") or set()) if str(x)], key=str),
            "neg_sigs": sorted([str(x) for x in list(fb.get("neg_sigs") or set()) if str(x)], key=str),
            "pos_text": {str(k): str((fb.get("pos_text") or {}).get(k)) for k in sorted((fb.get("pos_text") or {}).keys(), key=str)},
            "neg_text": {str(k): str((fb.get("neg_text") or {}).get(k)) for k in sorted((fb.get("neg_text") or {}).keys(), key=str)},
        }

    # Canonicalize csv lists.
    for cid in sorted(csv_by_id.keys(), key=str):
        st = csv_by_id.get(cid) or {}
        st["status"] = _canon_status(str(st.get("status") or "ALIVE"))
        st["pos_example_sigs"] = _canon_list_str(st.get("pos_example_sigs"))
        st["neg_example_sigs"] = _canon_list_str(st.get("neg_example_sigs"))
        st["seed_tokens"] = _canon_list_str(st.get("seed_tokens"))

    reg = {
        "schema_version": 103,
        "concepts_by_id": {str(cid): dict(concepts_by_id[cid]) for cid in sorted(concepts_by_id.keys(), key=str)},
        "csv_by_id": {str(cid): dict(csv_by_id[cid]) for cid in sorted(csv_by_id.keys(), key=str)},
        "name_index": {str(nm): str(name_index[nm]) for nm in sorted(name_index.keys(), key=str)},
        "feedback_by_name": dict(fb_out),
    }
    details = {
        "concepts_total": int(len(concepts_by_id)),
        "concept_events_total": int(len(concept_events)),
    }
    return reg, details


def concept_library_snapshot_v103(*, concept_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    reg, _details = fold_concept_ledger_v103(list(concept_events))
    # snapshot_sig excludes itself.
    sem = dict(reg)
    sem.pop("snapshot_sig", None)
    sig = _stable_hash_obj(sem)
    return dict(reg, snapshot_sig=str(sig))


def lookup_concept_id_by_name_v103(*, registry: Dict[str, Any], name_or_id: str) -> str:
    q = str(name_or_id or "")
    if not q:
        return ""
    c_by_id = registry.get("concepts_by_id") if isinstance(registry.get("concepts_by_id"), dict) else {}
    if q in set([str(x) for x in c_by_id.keys()]):
        return q
    idx = registry.get("name_index") if isinstance(registry.get("name_index"), dict) else {}
    # names are stored case-sensitive in this v103 (smoke uses upper-case name); lookup exact first.
    if q in idx:
        return str(idx.get(q) or "")
    # fallback: case-insensitive match on name_index keys.
    ql = q.lower()
    for nm in sorted(idx.keys(), key=str):
        if str(nm).lower() == ql:
            return str(idx.get(nm) or "")
    return ""


def explain_concept_view_v103(*, registry: Dict[str, Any], concept_id: str) -> Dict[str, Any]:
    c_by_id = registry.get("concepts_by_id") if isinstance(registry.get("concepts_by_id"), dict) else {}
    csv_by_id = registry.get("csv_by_id") if isinstance(registry.get("csv_by_id"), dict) else {}
    csg = c_by_id.get(concept_id) if isinstance(c_by_id.get(concept_id), dict) else {}
    csv = csv_by_id.get(concept_id) if isinstance(csv_by_id.get(concept_id), dict) else {}
    return {"concept_id": str(concept_id), "csg": dict(csg), "csv": dict(csv)}


def concepts_list_view_v103(*, registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    c_by_id = registry.get("concepts_by_id") if isinstance(registry.get("concepts_by_id"), dict) else {}
    csv_by_id = registry.get("csv_by_id") if isinstance(registry.get("csv_by_id"), dict) else {}
    out: List[Dict[str, Any]] = []
    for cid in sorted(c_by_id.keys(), key=str):
        csg = c_by_id.get(cid) if isinstance(c_by_id.get(cid), dict) else {}
        csv = csv_by_id.get(cid) if isinstance(csv_by_id.get(cid), dict) else {}
        out.append(
            {
                "concept_id": str(cid),
                "name": str(csg.get("name") or ""),
                "status": str(csv.get("status") or ""),
                "usage_count": int(csv.get("usage_count") or 0),
                "toc_pass": int(csv.get("toc_pass") or 0),
                "toc_fail": int(csv.get("toc_fail") or 0),
                "mdl_delta_bits": int(csv.get("mdl_delta_bits") or 0),
            }
        )
    out.sort(key=lambda d: (str(d.get("name") or ""), str(d.get("concept_id") or "")))
    return out
