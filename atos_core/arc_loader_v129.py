from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .grid_v124 import GridV124, grid_from_list_v124, grid_shape_v124, unique_colors_v124

ARC_LOADER_SCHEMA_VERSION_V129 = 129


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _resolve_arc_tasks_root_v129(*, arc_root: str, split: Optional[str]) -> Path:
    root = Path(str(arc_root)).resolve()
    if not root.exists():
        raise FileNotFoundError(f"arc_root_missing:{root}")

    requested = str(split or "").strip()
    candidates: List[Tuple[Path, List[str]]] = []
    for base in (root, root / "data"):
        if not base.exists():
            continue
        found: List[str] = []
        for name in ("training", "evaluation"):
            if (base / name).is_dir():
                found.append(name)
        if found:
            candidates.append((base, found))

    if candidates:
        # Prefer direct <arc_root>/<split> over <arc_root>/data/<split>.
        base, found = candidates[0]
        if requested and requested in found:
            return base / requested
        raise ValueError(f"arc_split_required:requested={requested or '<missing>'} available={','.join(found)} root={root}")

    # Flat layout: arc_root/*.json
    return root


def iter_arc_task_paths_v129(*, arc_root: str, split: Optional[str] = None) -> List[Path]:
    tasks_root = _resolve_arc_tasks_root_v129(arc_root=str(arc_root), split=split)
    paths = [p for p in tasks_root.rglob("*.json") if p.is_file()]
    paths.sort(key=lambda p: p.relative_to(tasks_root).as_posix())
    return paths


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _canonical_task_id(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _derive_meta_v129(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> Dict[str, Any]:
    shapes_in = sorted({grid_shape_v124(inp) for inp, _ in train_pairs} | {grid_shape_v124(test_in)})
    shapes_out = sorted({grid_shape_v124(out) for _, out in train_pairs})
    colors = set()
    for inp, out in train_pairs:
        colors.update(unique_colors_v124(inp))
        colors.update(unique_colors_v124(out))
    colors.update(unique_colors_v124(test_in))
    return {
        "shapes_in": [{"h": int(h), "w": int(w)} for h, w in shapes_in],
        "shapes_out": [{"h": int(h), "w": int(w)} for h, w in shapes_out],
        "colors": [int(c) for c in sorted(colors)],
    }


@dataclass(frozen=True)
class CanonicalArcTaskV129:
    task_id: str
    train_pairs: List[Tuple[GridV124, GridV124]]
    test_in: GridV124
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(ARC_LOADER_SCHEMA_VERSION_V129),
            "task_id": str(self.task_id),
            "train_pairs": [
                {"in_grid": [list(r) for r in inp], "out_grid": [list(r) for r in out]}
                for inp, out in self.train_pairs
            ],
            "test_in_grid": [list(r) for r in self.test_in],
            "meta": dict(self.meta),
        }


def load_arc_task_v129(*, root: Path, path: Path) -> CanonicalArcTaskV129:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise ValueError("arc_task_not_object")
    train_raw = raw.get("train")
    test_raw = raw.get("test")
    if not isinstance(train_raw, list) or not isinstance(test_raw, list) or not test_raw:
        raise ValueError("arc_task_missing_train_or_test")

    train_pairs: List[Tuple[GridV124, GridV124]] = []
    for pair in train_raw:
        if not isinstance(pair, dict):
            raise ValueError("arc_train_pair_not_object")
        inp = grid_from_list_v124(pair.get("input"))
        out = grid_from_list_v124(pair.get("output"))
        train_pairs.append((inp, out))

    test0 = test_raw[0]
    if not isinstance(test0, dict):
        raise ValueError("arc_test_pair_not_object")
    test_in = grid_from_list_v124(test0.get("input"))

    task_id = _canonical_task_id(root, path)
    meta = _derive_meta_v129(train_pairs=train_pairs, test_in=test_in)
    return CanonicalArcTaskV129(task_id=str(task_id), train_pairs=train_pairs, test_in=test_in, meta=meta)


def write_arc_canonical_jsonl_v129(
    *, arc_root: str, out_jsonl_path: str, limit: Optional[int] = None, split: Optional[str] = None
) -> Dict[str, Any]:
    root = Path(str(arc_root)).resolve()
    tasks_root = _resolve_arc_tasks_root_v129(arc_root=str(root), split=split)
    out_path = Path(str(out_jsonl_path))
    if out_path.exists():
        raise ValueError(f"worm_exists:{out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    task_paths = [p for p in tasks_root.rglob("*.json") if p.is_file()]
    task_paths.sort(key=lambda p: p.relative_to(tasks_root).as_posix())
    if limit is not None:
        task_paths = task_paths[: int(limit)]

    inputs: List[Dict[str, Any]] = []
    count_written = 0
    with open(out_path, "x", encoding="utf-8") as f:
        for p in task_paths:
            task = load_arc_task_v129(root=tasks_root, path=p)
            f.write(canonical_json_dumps(task.to_dict()))
            f.write("\n")
            inputs.append(
                {"task_id": str(task.task_id), "relpath": p.relative_to(tasks_root).as_posix(), "sha256": _sha256_file(p)}
            )
            count_written += 1

    manifest = {
        "schema_version": int(ARC_LOADER_SCHEMA_VERSION_V129),
        "kind": "arc_canonical_manifest_v129",
        "arc_root_input": str(root),
        "tasks_root": str(tasks_root),
        "split": str(split or ""),
        "tasks_total": int(count_written),
        "inputs": list(inputs),
        "sha256": {"canonical_jsonl": _sha256_file(out_path)},
    }
    manifest["manifest_sig"] = sha256_hex(canonical_json_dumps(manifest).encode("utf-8"))
    return manifest


def iter_canonical_tasks_v129(jsonl_path: str) -> Iterator[CanonicalArcTaskV129]:
    p = Path(str(jsonl_path))
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not isinstance(d, dict):
                continue
            task_id = str(d.get("task_id") or "")
            tps = d.get("train_pairs") if isinstance(d.get("train_pairs"), list) else []
            train_pairs: List[Tuple[GridV124, GridV124]] = []
            for tp in tps:
                if not isinstance(tp, dict):
                    continue
                inp = grid_from_list_v124(tp.get("in_grid"))
                out = grid_from_list_v124(tp.get("out_grid"))
                train_pairs.append((inp, out))
            test_in = grid_from_list_v124(d.get("test_in_grid"))
            meta = dict(d.get("meta", {})) if isinstance(d.get("meta"), dict) else {}
            yield CanonicalArcTaskV129(task_id=task_id, train_pairs=train_pairs, test_in=test_in, meta=meta)
