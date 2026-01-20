from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .act import Act, canonical_json_dumps, sha256_hex


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical_json_dumps(row))
            f.write("\n")
    os.replace(tmp, path)


@dataclass
class ActStore:
    acts: Dict[str, Act] = field(default_factory=dict)
    next_id_int: int = 1

    def new_id(self, prefix: str = "act") -> str:
        act_id = f"{prefix}{self.next_id_int:06d}"
        self.next_id_int += 1
        return act_id

    def add(self, act: Act) -> None:
        self.acts[act.id] = act

    def get(self, act_id: str) -> Optional[Act]:
        return self.acts.get(act_id)

    def active(self) -> List[Act]:
        return [a for a in self.acts.values() if a.active]

    def by_kind(self, kind: str) -> List[Act]:
        return [a for a in self.active() if a.kind == kind]

    def prune(self, act_id: str) -> None:
        act = self.acts.get(act_id)
        if act is None:
            return
        act.active = False

    def remove(self, act_id: str) -> None:
        self.acts.pop(act_id, None)

    def to_rows(self) -> List[Dict[str, Any]]:
        # Stable order.
        rows = [self.acts[k].to_dict() for k in sorted(self.acts.keys())]
        return rows

    def content_hash(self) -> str:
        blob = "\n".join(canonical_json_dumps(r) for r in self.to_rows()).encode("utf-8")
        return sha256_hex(blob)

    def save_jsonl(self, path: str) -> None:
        rows = self.to_rows()
        _write_jsonl(path, rows)

    @staticmethod
    def load_jsonl(path: str) -> "ActStore":
        store = ActStore()
        if not os.path.exists(path):
            return store
        max_int = 0
        for row in _read_jsonl(path):
            act = Act.from_dict(row)
            store.acts[act.id] = act
            # attempt to infer id counter
            suffix = "".join(ch for ch in act.id if ch.isdigit())
            if suffix:
                try:
                    max_int = max(max_int, int(suffix))
                except ValueError:
                    pass
        store.next_id_int = max_int + 1
        return store

