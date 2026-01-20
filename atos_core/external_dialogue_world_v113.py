from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


def _canon_path(p: str) -> str:
    return str(p).replace(os.sep, "/")


@dataclass(frozen=True)
class ExternalWorldTurnV113:
    global_turn_index: int
    conversation_id: str
    message_id: str
    timestamp: str
    role: str
    text: str
    source: str


def _turn_from_obj(obj: Dict[str, Any]) -> ExternalWorldTurnV113:
    return ExternalWorldTurnV113(
        global_turn_index=int(obj.get("global_turn_index") or 0),
        conversation_id=str(obj.get("conversation_id") or ""),
        message_id=str(obj.get("message_id") or ""),
        timestamp=str(obj.get("timestamp") or ""),
        role=str(obj.get("role") or "unknown"),
        text=str(obj.get("text") or ""),
        source=str(obj.get("source") or ""),
    )


class ExternalDialogueWorldV113:
    """
    Deterministic read-only API over dialogue_history_canonical_v113.jsonl.

    Indexing is deterministic and built in-memory (byte offsets per turn_id).
    No embeddings, no clustering, no global summarization.
    """

    def __init__(self, *, canonical_jsonl_path: str, manifest: Dict[str, Any]) -> None:
        self.canonical_jsonl_path = str(canonical_jsonl_path)
        self.manifest = dict(manifest)
        if not os.path.exists(self.canonical_jsonl_path):
            raise FileNotFoundError("external_world_v113_missing_canonical_jsonl")
        self.offsets: List[int] = []
        self.turns_total = 0
        self._build_offsets()

    def _build_offsets(self) -> None:
        self.offsets = []
        idx = 0
        with open(self.canonical_jsonl_path, "rb") as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                self.offsets.append(int(off))
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    raise ValueError("external_world_v113_json_decode_error")
                if int(obj.get("global_turn_index", -1)) != int(idx):
                    raise ValueError("external_world_v113_global_turn_index_mismatch")
                idx += 1
        self.turns_total = int(idx)

    def fetch_turn(self, turn_id: int) -> ExternalWorldTurnV113:
        idx = int(turn_id)
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError("turn_id_out_of_range")
        off = int(self.offsets[idx])
        with open(self.canonical_jsonl_path, "rb") as f:
            f.seek(off)
            line = f.readline()
        obj = json.loads(line.decode("utf-8"))
        if int(obj.get("global_turn_index", -1)) != idx:
            raise ValueError("turn_index_mismatch")
        return _turn_from_obj(obj)

    def observe_range(
        self,
        *,
        start_turn: int,
        end_turn: int,
        roles: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[ExternalWorldTurnV113]:
        s = int(start_turn)
        e = int(end_turn)
        if s < 0:
            s = 0
        if e >= len(self.offsets):
            e = len(self.offsets) - 1
        if e < s:
            return []
        role_set = set([str(r) for r in (roles or []) if isinstance(r, str) and r])
        out: List[ExternalWorldTurnV113] = []
        max_n = int(limit) if limit is not None else None
        for idx in range(s, e + 1):
            t = self.fetch_turn(idx)
            if role_set and t.role not in role_set:
                continue
            out.append(t)
            if max_n is not None and len(out) >= max_n:
                break
        return list(out)

    def search(
        self,
        *,
        query: str,
        limit: int,
        roles: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        q = str(query or "")
        if not q:
            return []
        role_set = set([str(r) for r in (roles or []) if isinstance(r, str) and r])
        out: List[Dict[str, Any]] = []
        with open(self.canonical_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                role = str(obj.get("role") or "unknown")
                if role_set and role not in role_set:
                    continue
                text = str(obj.get("text") or "")
                if q in text:
                    snippet = text[:120]
                    out.append(
                        {
                            "global_turn_index": int(obj.get("global_turn_index") or 0),
                            "conversation_id": str(obj.get("conversation_id") or ""),
                            "message_id": str(obj.get("message_id") or ""),
                            "role": role,
                            "snippet": snippet,
                        }
                    )
                    if len(out) >= int(limit):
                        break
        out.sort(key=lambda d: (int(d.get("global_turn_index") or 0), str(d.get("conversation_id") or "")))
        return list(out)


def load_world_v113(*, manifest_path: str) -> ExternalDialogueWorldV113:
    mp = str(manifest_path)
    if not os.path.exists(mp):
        raise FileNotFoundError("missing_world_manifest")
    with open(mp, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    root = os.path.dirname(os.path.abspath(mp))
    paths = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}
    canon_rel = str(paths.get("canonical_jsonl") or "")
    if not canon_rel:
        # Default to sibling file name if manifest is minimal.
        canon_rel = "dialogue_history_canonical_v113.jsonl"
    canon_path = os.path.normpath(os.path.join(root, canon_rel))
    return ExternalDialogueWorldV113(canonical_jsonl_path=canon_path, manifest=dict(manifest))

