from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _canon_path(p: str) -> str:
    return str(p).replace(os.sep, "/")


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class ExternalWorldHitV122:
    hit_id: str
    source: str
    snippet: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class ExternalWorldFetchV122:
    hit_id: str
    source: str
    doc_id: str
    text: str
    text_sha256: str
    truncated: bool
    offsets: Dict[str, Any]


class ExternalWorldV122:
    """
    Deterministic read-only ExternalWorld V122:
      - dialogue_history: JSONL (turn_index 0..N-1)
      - engineering_doc chunks: JSONL chunks with offsets

    No embeddings, no clustering, no global summarization.
    """

    def __init__(self, *, root_dir: str, manifest: Dict[str, Any]) -> None:
        self.root_dir = str(root_dir)
        self.manifest = dict(manifest)
        self.dialogue_path = os.path.join(self.root_dir, str(self.manifest.get("paths", {}).get("dialogue_history_canonical_jsonl") or ""))
        self.doc_chunks_path = os.path.join(self.root_dir, str(self.manifest.get("paths", {}).get("engineering_doc_chunks_jsonl") or ""))
        self.doc_plain_path = os.path.join(self.root_dir, str(self.manifest.get("paths", {}).get("engineering_doc_plain_txt") or ""))

        if not self.dialogue_path or not os.path.exists(self.dialogue_path):
            raise FileNotFoundError("external_world_v122_missing_dialogue_history")
        if not self.doc_chunks_path or not os.path.exists(self.doc_chunks_path):
            raise FileNotFoundError("external_world_v122_missing_engineering_chunks")
        if not self.doc_plain_path or not os.path.exists(self.doc_plain_path):
            raise FileNotFoundError("external_world_v122_missing_engineering_plain")

        self._dialogue_offsets: Optional[List[int]] = None
        self._dialogue_turns_total: Optional[int] = None
        self._doc_chunk_index: Optional[Dict[str, Dict[str, Any]]] = None

    def build_dialogue_offsets(self) -> None:
        offsets: List[int] = []
        idx = 0
        with open(self.dialogue_path, "rb") as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                offsets.append(int(off))
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    raise ValueError("external_world_v122_dialogue_json_decode_error")
                if int(obj.get("turn_index", -1)) != int(idx):
                    raise ValueError("external_world_v122_turn_index_mismatch")
                idx += 1
        self._dialogue_offsets = list(offsets)
        self._dialogue_turns_total = int(idx)

    def _ensure_doc_index(self) -> None:
        if self._doc_chunk_index is not None:
            return
        idx: Dict[str, Dict[str, Any]] = {}
        with open(self.doc_chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = str(obj.get("chunk_id") or "")
                if not cid:
                    continue
                idx[cid] = dict(obj)
        self._doc_chunk_index = dict(idx)

    def search(
        self,
        *,
        query: str,
        limit: int,
        source_filter: str = "all",
        roles: Optional[Sequence[str]] = None,
    ) -> List[ExternalWorldHitV122]:
        q = str(query or "")
        if not q:
            return []
        ql = q.lower()
        lim = int(limit)
        if lim <= 0:
            return []
        src = str(source_filter or "all")
        role_set = set([str(r) for r in (roles or []) if isinstance(r, str) and r])

        out: List[ExternalWorldHitV122] = []

        def _emit(hit_id: str, source: str, snippet: str, meta: Dict[str, Any]) -> None:
            out.append(ExternalWorldHitV122(hit_id=str(hit_id), source=str(source), snippet=str(snippet), meta=dict(meta)))

        if src in {"all", "dialogue_history"}:
            with open(self.dialogue_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    role = str(obj.get("role") or "unknown")
                    if role_set and role not in role_set:
                        continue
                    text = str(obj.get("text") or "")
                    if ql in text.lower():
                        tid = int(obj.get("turn_index") or 0)
                        snippet = text[:120]
                        _emit(
                            hit_id="dlg:{i}".format(i=int(tid)),
                            source="dialogue_history",
                            snippet=snippet,
                            meta={
                                "turn_index": int(tid),
                                "conversation_id": str(obj.get("conversation_id") or ""),
                                "message_id": str(obj.get("message_id") or ""),
                                "timestamp": str(obj.get("timestamp") or ""),
                                "role": role,
                            },
                        )
                        if len(out) >= lim:
                            break

        if len(out) < lim and src in {"all", "engineering_doc"}:
            self._ensure_doc_index()
            assert self._doc_chunk_index is not None
            # Deterministic scan order: chunk_id lexicographic.
            for cid in sorted(self._doc_chunk_index.keys()):
                obj = self._doc_chunk_index[cid]
                text = str(obj.get("text") or "")
                if ql in text.lower():
                    snippet = text[:120]
                    _emit(
                        hit_id="doc:{cid}".format(cid=str(cid)),
                        source="engineering_doc",
                        snippet=snippet,
                        meta={
                            "doc": str(obj.get("doc") or "Projeto_ACT"),
                            "chunk_id": str(cid),
                            "offset_start": int(obj.get("offset_start") or 0),
                            "offset_end": int(obj.get("offset_end") or 0),
                            "sha256_text": str(obj.get("sha256_text") or ""),
                        },
                    )
                    if len(out) >= lim:
                        break

        # Deterministic ordering (source, hit_id).
        out.sort(key=lambda h: (str(h.source), str(h.hit_id)))
        return list(out[:lim])

    def fetch(self, *, hit_id: str, max_chars: int) -> ExternalWorldFetchV122:
        hid = str(hit_id or "")
        lim = int(max_chars)
        if lim <= 0:
            lim = 1
        if hid.startswith("dlg:"):
            if self._dialogue_offsets is None:
                self.build_dialogue_offsets()
            assert self._dialogue_offsets is not None
            idx = int(hid.split(":", 1)[1])
            if idx < 0 or idx >= len(self._dialogue_offsets):
                raise IndexError("external_world_v122_turn_id_out_of_range")
            off = int(self._dialogue_offsets[idx])
            with open(self.dialogue_path, "rb") as f:
                f.seek(off)
                line = f.readline()
            obj = json.loads(line.decode("utf-8"))
            if int(obj.get("turn_index", -1)) != int(idx):
                raise ValueError("external_world_v122_turn_index_mismatch_fetch")
            text = str(obj.get("text") or "")
            truncated = False
            if len(text) > lim:
                text = text[:lim]
                truncated = True
            return ExternalWorldFetchV122(
                hit_id=str(hid),
                source="dialogue_history",
                doc_id="dialogue_history",
                text=str(text),
                text_sha256=sha256_hex(str(text).encode("utf-8")),
                truncated=bool(truncated),
                offsets={"turn_index": int(idx)},
            )
        if hid.startswith("doc:"):
            self._ensure_doc_index()
            assert self._doc_chunk_index is not None
            cid = hid.split(":", 1)[1]
            obj = self._doc_chunk_index.get(str(cid))
            if not isinstance(obj, dict):
                raise KeyError("external_world_v122_missing_doc_chunk")
            text = str(obj.get("text") or "")
            truncated = False
            if len(text) > lim:
                text = text[:lim]
                truncated = True
            return ExternalWorldFetchV122(
                hit_id=str(hid),
                source="engineering_doc",
                doc_id=str(obj.get("doc") or "Projeto_ACT"),
                text=str(text),
                text_sha256=sha256_hex(str(text).encode("utf-8")),
                truncated=bool(truncated),
                offsets={"offset_start": int(obj.get("offset_start") or 0), "offset_end": int(obj.get("offset_end") or 0), "chunk_id": str(cid)},
            )
        raise ValueError("external_world_v122_invalid_hit_id")


def ew_load_and_verify(*, manifest_path: str) -> ExternalWorldV122:
    mp = str(manifest_path)
    if not os.path.exists(mp):
        raise FileNotFoundError("external_world_v122_missing_manifest")
    m = _load_json(mp)
    if not isinstance(m, dict) or int(m.get("schema_version") or 0) != 122:
        raise ValueError("external_world_manifest_schema_mismatch_v122")

    root = os.path.dirname(os.path.abspath(mp))
    paths = m.get("paths") if isinstance(m.get("paths"), dict) else {}
    sha = m.get("sha256") if isinstance(m.get("sha256"), dict) else {}

    def _abs(rel: str) -> str:
        return os.path.normpath(os.path.join(root, str(rel)))

    required = {
        "dialogue_history_canonical_v122_jsonl": _abs(str(paths.get("dialogue_history_canonical_jsonl") or "")),
        "engineering_doc_plain_v122_txt": _abs(str(paths.get("engineering_doc_plain_txt") or "")),
        "engineering_doc_chunks_v122_jsonl": _abs(str(paths.get("engineering_doc_chunks_jsonl") or "")),
    }
    for k, p in required.items():
        if not p or not os.path.exists(p):
            raise ValueError("external_world_manifest_mismatch_v122")
        want = str(sha.get(k) or "")
        got = _sha256_file(p)
        if want and str(want) != str(got):
            raise ValueError("external_world_manifest_mismatch_v122")

    # Verify manifest signature if present.
    msig = str(m.get("manifest_sig") or "")
    if msig:
        body = dict(m)
        body.pop("manifest_sig", None)
        got_sig = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
        if str(got_sig) != str(msig):
            raise ValueError("external_world_manifest_mismatch_v122")

    return ExternalWorldV122(root_dir=str(root), manifest=dict(m))


def external_world_regurgitation_v122(*, assistant_text: str, fetched_text: str) -> Dict[str, Any]:
    """
    Deterministic, conservative "anti-copy" signal.
    Not a semantic judge; only detects large contiguous reuse of fetched content.
    """
    a = " ".join([t for t in str(assistant_text or "").split() if t])
    f = " ".join([t for t in str(fetched_text or "").split() if t])
    if not a or not f:
        return {"ok": True, "reason": "empty"}
    # Threshold: at least 200 chars or 1/2 of fetched (whichever is smaller but >=120).
    thr = max(120, min(200, max(120, int(len(f) // 2))))
    if len(f) < thr:
        return {"ok": True, "reason": "fetched_short"}
    # Deterministic windows.
    offsets = [0, max(0, (len(f) // 3) - (thr // 2)), max(0, (2 * len(f) // 3) - (thr // 2))]
    for off in offsets:
        if off + thr > len(f):
            off = max(0, len(f) - thr)
        window = f[off : off + thr]
        if window and window in a:
            return {"ok": False, "reason": "regurgitation_detected", "window_len": int(len(window))}
    return {"ok": True, "reason": "no_large_overlap"}

