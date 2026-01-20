from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_world_canonical_jsonl_v119(*, world_manifest_path: str, default_rel: str) -> Dict[str, Any]:
    """
    Resolve a world manifest to its canonical JSONL file in a deterministic, audit-friendly way.

    Supports both:
      - external_world/dialogue_history_canonical_v113_manifest.json (paths.canonical_jsonl relative to manifest dir)
      - external_world/manifests/world_manifest_v111.json (paths.canonical_jsonl relative to external_world root)

    Strategy:
      - read `paths.canonical_jsonl` (or fall back to `default_rel`);
      - try candidates:
          1) manifest_dir / rel
          2) manifest_dir.parent / rel
      - if rel is absolute, try rel directly.

    Returns a dict with:
      - ok, reason
      - world_manifest_input, canonical_jsonl_rel, canonical_jsonl_resolved, attempts
      - world_manifest_sha256, world_canonical_sha256
    """
    mp = Path(str(world_manifest_path)).expanduser()
    if not mp.exists():
        return {"ok": False, "reason": "missing_world_manifest", "world_manifest_input": str(world_manifest_path), "attempts": []}

    m = _load_json(mp)
    paths = m.get("paths") if isinstance(m, dict) and isinstance(m.get("paths"), dict) else {}
    rel = str(paths.get("canonical_jsonl") or str(default_rel))

    attempts: List[str] = []
    candidates: List[Path] = []
    rel_path = Path(rel)
    if rel_path.is_absolute():
        candidates.append(rel_path)
    else:
        candidates.append((mp.parent / rel_path))
        candidates.append((mp.parent.parent / rel_path))

    chosen: Optional[Path] = None
    for c in candidates:
        attempts.append(str(c))
        if c.exists():
            chosen = c.resolve()
            break

    if chosen is None:
        return {
            "ok": False,
            "reason": "missing_world_canonical_jsonl",
            "world_manifest_input": str(world_manifest_path),
            "canonical_jsonl_rel": str(rel),
            "attempts": list(attempts),
        }

    msha = _sha256_file(mp)
    csha = _sha256_file(chosen)
    return {
        "ok": True,
        "reason": "ok",
        "world_manifest_input": str(world_manifest_path),
        "canonical_jsonl_rel": str(rel),
        "canonical_jsonl_resolved": str(chosen),
        "attempts": list(attempts),
        "world_manifest_sha256": str(msha),
        "world_canonical_sha256": str(csha),
    }

