from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .act import canonical_json_dumps, sha256_hex


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


def _canon_interface_slots(slots: Any) -> List[Dict[str, Any]]:
    if not isinstance(slots, list):
        return []
    out: List[Dict[str, Any]] = []
    for s in slots:
        if not isinstance(s, dict):
            continue
        out.append(
            {
                "slot_name": str(s.get("slot_name") or ""),
                "type_tag": str(s.get("type_tag") or ""),
                "constraints": dict(s.get("constraints") or {}) if isinstance(s.get("constraints"), dict) else {},
            }
        )
    out.sort(key=lambda d: (str(d.get("slot_name") or ""), str(d.get("type_tag") or "")))
    return out


def canonicalize_csg_v103(csg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonicalizes a Concept SubGraph (CSG) body for stable hashing.
    NOTE: This excludes CSV state; only immutable structure belongs here.
    """
    d = dict(csg)
    body = d.get("body") if isinstance(d.get("body"), dict) else {}
    features = _canon_list_str(body.get("features"))
    body_canon = {"op": str(body.get("op") or ""), "features": list(features)}

    out = {
        "schema_version": 103,
        "kind": "csg_v103",
        "name": str(d.get("name") or ""),
        "interface_slots": _canon_interface_slots(d.get("interface_slots")),
        "invariants": _canon_list_str(d.get("invariants")),
        "body": dict(body_canon),
    }
    return out


def csg_structural_hash_v103(csg: Dict[str, Any]) -> str:
    """
    structural_hash = sha256(canonical_json(CSG_without_ids)).
    """
    canon = canonicalize_csg_v103(csg)
    return sha256_hex(canonical_json_dumps(canon).encode("utf-8"))


def concept_id_v103(structural_hash: str) -> str:
    return f"concept_v103_{str(structural_hash)}"


def make_rule_body_v103(*, features: List[str]) -> Dict[str, Any]:
    return {"op": "RULE_MATCH_ALL", "features": _canon_list_str(list(features))}


def make_csg_rule_v103(*, name: str, features: List[str]) -> Dict[str, Any]:
    csg0 = {
        "name": str(name),
        "interface_slots": [{"slot_name": "text", "type_tag": "text", "constraints": {}}],
        "invariants": ["output_schema:{matched:bool,evidence:list[str]}"],
        "body": make_rule_body_v103(features=list(features)),
    }
    structural_hash = csg_structural_hash_v103(dict(csg0))
    cid = concept_id_v103(structural_hash)
    csg = dict(canonicalize_csg_v103(dict(csg0)), concept_id=str(cid), structural_hash=str(structural_hash))
    return csg


def rule_features_from_csg_v103(csg: Dict[str, Any]) -> List[str]:
    body = csg.get("body") if isinstance(csg.get("body"), dict) else {}
    feats = body.get("features")
    return _canon_list_str(feats)


def csg_name_v103(csg: Dict[str, Any]) -> str:
    return str(csg.get("name") or "")


def csg_status_default_v103() -> str:
    return "ALIVE"


@dataclass(frozen=True)
class InducedRuleV103:
    features: List[str]
    mdl_baseline_bits: int
    mdl_model_bits: int
    mdl_data_bits: int
    mdl_delta_bits: int
    seed_tokens: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "features": list(_canon_list_str(list(self.features))),
            "mdl_baseline_bits": int(self.mdl_baseline_bits),
            "mdl_model_bits": int(self.mdl_model_bits),
            "mdl_data_bits": int(self.mdl_data_bits),
            "mdl_delta_bits": int(self.mdl_delta_bits),
            "seed_tokens": list(_canon_list_str(list(self.seed_tokens))),
        }

