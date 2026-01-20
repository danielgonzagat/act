from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .fluency_survival_v112 import fluency_contract_v112

DIALOGUE_SURVIVAL_REASON_OK_V116 = "ok"
DIALOGUE_SURVIVAL_REASON_FLUENCY_FAIL_V116 = "dialogue_survival_fluency_fail"
DIALOGUE_SURVIVAL_REASON_UNRESOLVED_REFERENCE_V116 = "dialogue_survival_unresolved_reference"
DIALOGUE_SURVIVAL_REASON_CONTRADICTION_V116 = "dialogue_survival_contradiction_detected"
DIALOGUE_SURVIVAL_REASON_MISSING_TRANSCRIPT_V116 = "dialogue_survival_missing_transcript"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ensure_absent(path: str) -> None:
    if os.path.exists(path):
        raise ValueError(f"worm_exists:{path}")


def _write_once_json(path: str, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        raise ValueError(f"tmp_exists:{tmp}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_transcript_view_v116(transcript_jsonl: str) -> List[Dict[str, Any]]:
    rows = _read_jsonl(str(transcript_jsonl))
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        role = str(payload.get("role") or "")
        text = str(payload.get("text") or "")
        out.append({"role": str(role), "text": str(text)})
    return out


def _count_unresolved_reference_events_v116(binding_events: Sequence[Dict[str, Any]]) -> int:
    bad = 0
    for ev in list(binding_events):
        if not isinstance(ev, dict):
            continue
        t = str(ev.get("type") or "")
        if t in {"BIND_MISS", "BIND_AMBIGUOUS"}:
            bad += 1
    return int(bad)


def _unresolved_reference_final_from_flow_v116(flow_events: Sequence[Dict[str, Any]]) -> int:
    if not flow_events:
        return 0
    last = flow_events[-1] if isinstance(flow_events[-1], dict) else {}
    flags = last.get("flow_flags_v108")
    if not isinstance(flags, dict):
        return 0
    return 1 if bool(flags.get("UNRESOLVED_REFERENCE")) else 0


def _count_semantic_contradiction_flags_v116(semantic_events: Sequence[Dict[str, Any]]) -> int:
    cnt = 0
    for ev in list(semantic_events):
        if not isinstance(ev, dict):
            continue
        flags = ev.get("flags_v109")
        if not isinstance(flags, dict):
            continue
        if bool(flags.get("CONTRADICTION_UNREPAIRED")):
            cnt += 1
    return int(cnt)


@dataclass(frozen=True)
class DialogueSurvivalDecisionV116:
    ok: bool
    reason_code: str
    details: Dict[str, Any]


def decide_dialogue_survival_v116(
    *,
    fluency_ok: bool,
    fluency_reason: str,
    fluency_details: Optional[Dict[str, Any]],
    unresolved_reference_final: int,
    contradiction_flags_total: int,
    extra: Optional[Dict[str, Any]] = None,
) -> DialogueSurvivalDecisionV116:
    """
    Pure, deterministic decision function (unit-testable).
    Priority of failure (deterministic): fluency -> unresolved_reference -> contradiction.
    """
    details: Dict[str, Any] = {
        "fluency_ok": bool(fluency_ok),
        "fluency_reason": str(fluency_reason or ""),
        "fluency_details": dict(fluency_details) if isinstance(fluency_details, dict) else {},
        "unresolved_reference_final": int(unresolved_reference_final),
        "semantic_contradiction_flags_total": int(contradiction_flags_total),
    }
    if isinstance(extra, dict):
        details.update(dict(extra))

    if not bool(fluency_ok):
        return DialogueSurvivalDecisionV116(ok=False, reason_code=DIALOGUE_SURVIVAL_REASON_FLUENCY_FAIL_V116, details=dict(details))
    if int(unresolved_reference_final) != 0:
        return DialogueSurvivalDecisionV116(ok=False, reason_code=DIALOGUE_SURVIVAL_REASON_UNRESOLVED_REFERENCE_V116, details=dict(details))
    if int(contradiction_flags_total) != 0:
        return DialogueSurvivalDecisionV116(ok=False, reason_code=DIALOGUE_SURVIVAL_REASON_CONTRADICTION_V116, details=dict(details))
    return DialogueSurvivalDecisionV116(ok=True, reason_code=DIALOGUE_SURVIVAL_REASON_OK_V116, details=dict(details))


def compute_dialogue_survival_summary_v116(
    *,
    run_dir: str,
    write_summary: bool = True,
) -> DialogueSurvivalDecisionV116:
    rd = str(run_dir)
    transcript_path = os.path.join(rd, "transcript.jsonl")
    if not os.path.exists(transcript_path):
        dec = DialogueSurvivalDecisionV116(
            ok=False,
            reason_code=DIALOGUE_SURVIVAL_REASON_MISSING_TRANSCRIPT_V116,
            details={"missing_paths": ["transcript.jsonl"]},
        )
        if write_summary:
            _write_once_json(os.path.join(rd, "dialogue_survival_summary_v116.json"), _summary_obj_v116(decision=dec, sha256_paths={}))
        return dec

    transcript_view = _load_transcript_view_v116(transcript_path)
    ok_fc, reason_fc, details_fc = fluency_contract_v112(transcript_view=transcript_view)

    binding_events = _read_jsonl(os.path.join(rd, "binding_events.jsonl"))
    unresolved_total = _count_unresolved_reference_events_v116(binding_events)

    flow_events = _read_jsonl(os.path.join(rd, "flow_events.jsonl"))
    unresolved_final = _unresolved_reference_final_from_flow_v116(flow_events)

    semantic_events = _read_jsonl(os.path.join(rd, "semantic_events.jsonl"))
    contradiction_flags = _count_semantic_contradiction_flags_v116(semantic_events)

    extra = {
        "unresolved_reference_total": int(unresolved_total),
        "flow_unresolved_reference_final": int(unresolved_final),
        "transcript_sha256": _sha256_file(transcript_path),
    }

    dec = decide_dialogue_survival_v116(
        fluency_ok=bool(ok_fc),
        fluency_reason=str(reason_fc),
        fluency_details=dict(details_fc),
        unresolved_reference_final=int(unresolved_final),
        contradiction_flags_total=int(contradiction_flags),
        extra=dict(extra),
    )

    if write_summary:
        sha256_paths: Dict[str, str] = {"transcript_jsonl": str(transcript_path)}
        summary_obj = _summary_obj_v116(decision=dec, sha256_paths=sha256_paths)
        _write_once_json(os.path.join(rd, "dialogue_survival_summary_v116.json"), summary_obj)

    return dec


def _summary_obj_v116(*, decision: DialogueSurvivalDecisionV116, sha256_paths: Dict[str, str]) -> Dict[str, Any]:
    sha256: Dict[str, str] = {}
    sha256_rel: Dict[str, str] = {}
    for k, p in sorted(sha256_paths.items(), key=lambda kv: str(kv[0])):
        sha256_rel[str(k)] = str(os.path.basename(str(p)))
        if os.path.exists(p):
            sha256[str(k)] = _sha256_file(str(p))
    body = {
        "schema_version": 116,
        "kind": "dialogue_survival_summary_v116",
        "ok": bool(decision.ok),
        "reason_code": str(decision.reason_code),
        "details": dict(decision.details),
        "sha256": dict(sha256),
        "sha256_paths": dict(sha256_rel),
    }
    body["summary_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return body

