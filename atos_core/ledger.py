from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional

from .act import Patch, canonical_json_dumps, deterministic_iso, sha256_hex


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


@dataclass
class Ledger:
    path: str

    def append(
        self,
        *,
        step: int,
        patch: Optional[Patch],
        acts_hash: str,
        metrics: Dict[str, Any],
        snapshot_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        prev = self.last_hash()
        entry = {
            "time": deterministic_iso(step=int(step)),
            "step": int(step),
            "patch": patch.to_dict() if patch else None,
            "acts_hash": acts_hash,
            "metrics": metrics,
            "snapshot_path": snapshot_path,
            "prev_hash": prev,
        }
        entry_hash = sha256_hex(canonical_json_dumps(entry).encode("utf-8"))
        entry["entry_hash"] = entry_hash
        _append_jsonl(self.path, entry)
        return entry

    def iter_entries(self) -> Iterator[Dict[str, Any]]:
        yield from _read_jsonl(self.path)

    def last_hash(self) -> Optional[str]:
        last = None
        for row in self.iter_entries():
            last = row.get("entry_hash")
        return last

    def verify_chain(self) -> bool:
        prev = None
        for row in self.iter_entries():
            row = dict(row)
            entry_hash = row.pop("entry_hash", None)
            if row.get("prev_hash") != prev:
                return False
            expected = sha256_hex(canonical_json_dumps(row).encode("utf-8"))
            if expected != entry_hash:
                return False
            prev = entry_hash
        return True
