from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .act import Act, canonical_json_dumps, sha256_hex
from .ethics import validate_act_for_load


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
        verdict = validate_act_for_load(act)
        if not bool(verdict.ok):
            raise ValueError(f"ethics_fail_closed:add:{act.id}:{verdict.reason}:{verdict.violated_laws}")
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

    def content_hash(self, *, exclude_kinds: Optional[Sequence[str]] = None) -> str:
        rows: List[Dict[str, Any]]
        if exclude_kinds:
            excluded = {str(k) for k in exclude_kinds if isinstance(k, str) and str(k)}
            rows = [
                self.acts[k].to_dict()
                for k in sorted(self.acts.keys())
                if str(self.acts[k].kind) not in excluded
            ]
        else:
            rows = self.to_rows()
        blob = "\n".join(canonical_json_dumps(r) for r in rows).encode("utf-8")
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
            verdict = validate_act_for_load(act)
            if not bool(verdict.ok):
                raise ValueError(
                    f"ethics_fail_closed:load:{act.id}:{verdict.reason}:{verdict.violated_laws}"
                )
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

    def get_gate_table_act(self, act_id: str) -> Optional[Act]:
        act = self.get(str(act_id))
        if act is None or not bool(act.active):
            return None
        if str(act.kind) != "gate_table_ctxsig":
            return None
        return act

    def best_gate_table_for_hash(self, store_hash: str) -> Optional[Act]:
        want = str(store_hash or "")
        candidates: List[Act] = []
        for act in self.by_kind("gate_table_ctxsig"):
            ev = act.evidence
            if not isinstance(ev, dict):
                continue
            meta = ev.get("meta")
            if not isinstance(meta, dict):
                continue
            trained = str(
                meta.get("trained_on_store_content_hash")
                or meta.get("trained_on_store_hash")
                or ""
            )
            if trained != want:
                continue
            candidates.append(act)

        def _score(a: Act) -> float:
            meta = a.evidence.get("meta") if isinstance(a.evidence, dict) else {}
            if not isinstance(meta, dict):
                return 0.0
            val = meta.get("pct_saved_real")
            if val is None:
                val = meta.get("pct_saved")
            try:
                return float(val or 0.0)
            except Exception:
                return 0.0

        candidates.sort(key=lambda a: (-_score(a), str(a.id)))
        return candidates[0] if candidates else None

    def get_concept_act(self, act_id: str) -> Optional[Act]:
        act = self.get(str(act_id))
        if act is None or not bool(act.active):
            return None
        if str(act.kind) != "concept_csv":
            return None
        return act

    def concept_acts(self) -> List[Act]:
        return self.by_kind("concept_csv")

    def goal_acts(self) -> List[Act]:
        return self.by_kind("goal")

    def plan_acts(self) -> List[Act]:
        return self.by_kind("plan")

    def hypothesis_acts(self) -> List[Act]:
        return self.by_kind("hypothesis")

    def reference_acts(self) -> List[Act]:
        return self.by_kind("reference")
