from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


def _canon_path(p: str) -> str:
    return str(p).replace(os.sep, "/")


def _read_u64_le_list(path: str) -> List[int]:
    data = open(path, "rb").read()
    if len(data) % 8 != 0:
        raise ValueError(f"offsets_size_not_multiple_of_8:{path}:{len(data)}")
    out: List[int] = []
    for i in range(0, len(data), 8):
        out.append(int(struct.unpack("<Q", data[i : i + 8])[0]))
    return out


@dataclass(frozen=True)
class ExternalWorldTurnV111:
    global_turn_index: int
    conversation_id: str
    conversation_title: Optional[str]
    turn_in_conversation: int
    timestamp: Optional[str]
    role: str
    text: str
    message_id: str
    parent_message_id: str
    children_message_ids: List[str]
    source: str


def _turn_from_obj(obj: Dict[str, Any]) -> ExternalWorldTurnV111:
    return ExternalWorldTurnV111(
        global_turn_index=int(obj.get("global_turn_index") or 0),
        conversation_id=str(obj.get("conversation_id") or ""),
        conversation_title=str(obj.get("conversation_title")) if isinstance(obj.get("conversation_title"), str) else None,
        turn_in_conversation=int(obj.get("turn_in_conversation") or 0),
        timestamp=str(obj.get("timestamp")) if isinstance(obj.get("timestamp"), str) else None,
        role=str(obj.get("role") or "unknown"),
        text=str(obj.get("text") or ""),
        message_id=str(obj.get("message_id") or ""),
        parent_message_id=str(obj.get("parent_message_id") or ""),
        children_message_ids=[str(x) for x in (obj.get("children_message_ids") or []) if isinstance(x, str)],
        source=str(obj.get("source") or ""),
    )


class ExternalDialogueWorldV111:
    def __init__(self, *, world_root: str, manifest: Dict[str, Any]) -> None:
        self.world_root = str(world_root)
        self.manifest = dict(manifest)

        paths = self.manifest.get("paths") if isinstance(self.manifest.get("paths"), dict) else {}
        self.canonical_jsonl_path = os.path.normpath(os.path.join(self.world_root, str(paths.get("canonical_jsonl") or "")))
        self.offsets_bin_path = os.path.normpath(os.path.join(self.world_root, str(paths.get("offsets_bin") or "")))
        self.conversations_index_path = os.path.normpath(os.path.join(self.world_root, str(paths.get("conversations_index_json") or "")))

        if not (os.path.exists(self.canonical_jsonl_path) and os.path.exists(self.offsets_bin_path) and os.path.exists(self.conversations_index_path)):
            raise FileNotFoundError("external_world_missing_files")

        counts = self.manifest.get("counts") if isinstance(self.manifest.get("counts"), dict) else {}
        self.turns_total = int(counts.get("turns_total") or 0)
        self.offsets = _read_u64_le_list(self.offsets_bin_path)
        if self.turns_total and len(self.offsets) != self.turns_total:
            raise ValueError("external_world_offsets_len_mismatch")

        with open(self.conversations_index_path, "r", encoding="utf-8") as f:
            self.conversations_index = json.load(f)

    def fetch_turn(self, turn_id: int) -> ExternalWorldTurnV111:
        idx = int(turn_id)
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError("turn_id_out_of_range")
        offset = int(self.offsets[idx])
        with open(self.canonical_jsonl_path, "rb") as f:
            f.seek(offset)
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
    ) -> List[ExternalWorldTurnV111]:
        s = int(start_turn)
        e = int(end_turn)
        if s < 0:
            s = 0
        if e >= len(self.offsets):
            e = len(self.offsets) - 1
        if e < s:
            return []
        role_set = set([str(r) for r in (roles or []) if isinstance(r, str) and r])
        out: List[ExternalWorldTurnV111] = []
        max_n = int(limit) if limit is not None else None
        for idx in range(s, e + 1):
            t = self.fetch_turn(idx)
            if role_set and t.role not in role_set:
                continue
            out.append(t)
            if max_n is not None and len(out) >= max_n:
                break
        return list(out)

    def search_text(
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
        # Deterministic linear scan; used only under gating.
        with open(self.canonical_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                role = str(obj.get("role") or "unknown")
                if role_set and role not in role_set:
                    continue
                text = str(obj.get("text") or "")
                if q in text:
                    out.append(
                        {
                            "global_turn_index": int(obj.get("global_turn_index") or 0),
                            "conversation_id": str(obj.get("conversation_id") or ""),
                            "role": role,
                            "text_hash": "",  # filled by caller if needed
                        }
                    )
                    if len(out) >= int(limit):
                        break
        out.sort(key=lambda d: (int(d.get("global_turn_index") or 0), str(d.get("conversation_id") or "")))
        return list(out)


def load_world_v111(*, manifest_path: str) -> ExternalDialogueWorldV111:
    mp = str(manifest_path)
    if not os.path.exists(mp):
        raise FileNotFoundError("missing_world_manifest")
    with open(mp, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    world_root = os.path.dirname(os.path.dirname(os.path.abspath(mp)))
    return ExternalDialogueWorldV111(world_root=world_root, manifest=dict(manifest))

