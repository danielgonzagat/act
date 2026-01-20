from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .grid_v124 import GridV124, grid_from_list_v124, grid_shape_v124, unique_colors_v124

ARC_LOADER_SCHEMA_VERSION_V137 = 137


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _resolve_arc_tasks_root_v137(*, arc_root: str, split: Optional[str]) -> Path:
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
            candidates.append((base, sorted(found)))

    if candidates:
        avail = sorted(set(x for _, fs in candidates for x in fs))
        if requested not in avail:
            raise ValueError(
                f"arc_split_required:requested={requested or '<missing>'} available={','.join(avail)} root={root}"
            )
        for base, fs in candidates:
            if requested in fs:
                return (base / requested).resolve()
        raise ValueError("arc_split_resolution_failed")

    if requested and requested not in ("sample", "synth"):
        raise ValueError(f"arc_split_not_found:requested={requested} root={root}")
    return root


@dataclass(frozen=True)
class ArcTaskV137:
    task_id: str
    train_pairs: Tuple[Tuple[GridV124, GridV124], ...]
    test_in: GridV124

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(ARC_LOADER_SCHEMA_VERSION_V137),
            "kind": "arc_task_v137",
            "task_id": str(self.task_id),
            "train_pairs": [
                {"in_grid": [list(r) for r in inp], "out_grid": [list(r) for r in out]}
                for inp, out in self.train_pairs
            ],
            "test_in": [list(r) for r in self.test_in],
        }


def _parse_grid_v137(x: Any) -> GridV124:
    if not isinstance(x, list):
        raise ValueError("grid_not_list")
    rows: List[List[int]] = []
    for row in x:
        if not isinstance(row, list):
            raise ValueError("grid_row_not_list")
        rows.append([int(v) for v in row])
    return grid_from_list_v124(rows)


def _parse_task_json_v137(*, path: Path, task_id: str) -> ArcTaskV137:
    obj = json.loads(path.read_text(encoding="utf-8"))
    train_pairs: List[Tuple[GridV124, GridV124]] = []
    for pair in obj.get("train", []):
        inp = _parse_grid_v137(pair.get("input"))
        out = _parse_grid_v137(pair.get("output"))
        train_pairs.append((inp, out))
    tests = obj.get("test", [])
    if not tests:
        raise ValueError("missing_test")
    test_in = _parse_grid_v137(tests[0].get("input"))
    for g in [test_in] + [p[0] for p in train_pairs] + [p[1] for p in train_pairs]:
        h, w = grid_shape_v124(g)
        if h < 0 or w < 0:
            raise ValueError("invalid_grid_shape")
        for c in unique_colors_v124(g):
            cc = int(c)
            if cc < 0 or cc > 9:
                raise ValueError("grid_color_out_of_range")
    return ArcTaskV137(task_id=str(task_id), train_pairs=tuple(train_pairs), test_in=test_in)


def write_arc_canonical_jsonl_v137(
    *, arc_root: str, split: Optional[str], limit: int, out_jsonl: Path, out_manifest: Path
) -> Dict[str, Any]:
    tasks_root = _resolve_arc_tasks_root_v137(arc_root=str(arc_root), split=split)
    if out_jsonl.exists():
        raise FileExistsError(f"worm_exists:{out_jsonl}")
    if out_manifest.exists():
        raise FileExistsError(f"worm_exists:{out_manifest}")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    task_paths = sorted(tasks_root.rglob("*.json"), key=lambda p: str(p.relative_to(tasks_root)))
    if int(limit) > 0:
        task_paths = task_paths[: int(limit)]

    inputs: List[Dict[str, Any]] = []
    rows: List[str] = []
    for p in task_paths:
        task_id = str(p.relative_to(tasks_root)).replace("\\", "/")
        sha = _sha256_file(p)
        inputs.append({"task_id": str(task_id), "path": str(p), "sha256": str(sha)})
        task = _parse_task_json_v137(path=p, task_id=task_id)
        rows.append(canonical_json_dumps(task.to_dict()))

    with open(out_jsonl, "x", encoding="utf-8") as f:
        for line in rows:
            f.write(line + "\n")

    manifest_obj: Dict[str, Any] = {
        "schema_version": int(ARC_LOADER_SCHEMA_VERSION_V137),
        "kind": "arc_manifest_v137",
        "arc_root_input": str(Path(str(arc_root)).resolve()),
        "tasks_root": str(tasks_root),
        "split": str(split or ""),
        "limit": int(limit),
        "inputs": inputs,
        "canonical_jsonl_sha256": _sha256_file(out_jsonl),
    }
    manifest_obj["manifest_sig"] = sha256_hex(canonical_json_dumps(manifest_obj).encode("utf-8"))
    with open(out_manifest, "x", encoding="utf-8") as f:
        json.dump(manifest_obj, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    return manifest_obj


def iter_canonical_tasks_v137(jsonl_path: str) -> Iterator[ArcTaskV137]:
    with open(str(jsonl_path), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            train_pairs: List[Tuple[GridV124, GridV124]] = []
            for pair in obj.get("train_pairs", []):
                inp = _parse_grid_v137(pair.get("in_grid"))
                out = _parse_grid_v137(pair.get("out_grid"))
                train_pairs.append((inp, out))
            test_in = _parse_grid_v137(obj.get("test_in"))
            yield ArcTaskV137(task_id=str(obj.get("task_id")), train_pairs=tuple(train_pairs), test_in=test_in)

