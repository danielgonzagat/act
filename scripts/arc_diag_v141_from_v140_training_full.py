#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _grid_shape(grid: Sequence[Sequence[int]]) -> Tuple[int, int]:
    h = int(len(grid))
    w = int(len(grid[0])) if h else 0
    return (h, w)


def _grid_palette(grid: Sequence[Sequence[int]]) -> Set[int]:
    out: Set[int] = set()
    for row in grid:
        for x in row:
            out.add(int(x))
    return out


def _grid_diff_cells_same_shape(inp: Sequence[Sequence[int]], out: Sequence[Sequence[int]]) -> Optional[int]:
    hi, wi = _grid_shape(inp)
    ho, wo = _grid_shape(out)
    if (hi, wi) != (ho, wo):
        return None
    diff = 0
    for r in range(hi):
        for c in range(wi):
            if int(inp[r][c]) != int(out[r][c]):
                diff += 1
    return int(diff)


def _shape_rel_for_pairs(pairs: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]]) -> str:
    if not pairs:
        return "unknown"
    if all(a == b for a, b in pairs):
        return "same"
    if all(b == (a[1], a[0]) for a, b in pairs):
        return "swap_hw"
    ratios: Set[Tuple[int, int]] = set()
    ok = True
    for (hi, wi), (ho, wo) in pairs:
        if hi <= 0 or wi <= 0 or ho <= 0 or wo <= 0:
            ok = False
            break
        if ho % hi != 0 or wo % wi != 0:
            ok = False
            break
        ratios.add((int(ho // hi), int(wo // wi)))
    if ok and len(ratios) == 1:
        ry, rx = list(ratios)[0]
        return f"scale_integer:{int(ry)}x{int(rx)}"
    return "shape_change_mixed"


def _palette_rel(p_in: Set[int], p_out: Set[int]) -> str:
    if p_out.issubset(p_in):
        if p_in == p_out:
            return "equal"
        return "subset"
    if p_in.issubset(p_out):
        return "superset"
    return "other"


def _delta_density_bin(pairs: Sequence[Tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]]) -> str:
    diffs: List[float] = []
    for inp, out in pairs:
        d = _grid_diff_cells_same_shape(inp, out)
        if d is None:
            continue
        h, w = _grid_shape(inp)
        diffs.append(float(d) / float(max(1, h * w)))
    if not diffs:
        return "n/a"
    avg = sum(diffs) / float(len(diffs))
    if avg <= 0.10:
        return "sparse<=0.10"
    if avg <= 0.30:
        return "local<=0.30"
    return "dense>0.30"


def _mode_color(grid: Sequence[Sequence[int]]) -> int:
    counts: Dict[int, int] = {}
    for row in grid:
        for x in row:
            xx = int(x)
            counts[xx] = int(counts.get(xx, 0)) + 1
    if not counts:
        return 0
    items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
    return int(items[0][0])


def _cc4_nonbg_ignore_color_components(
    grid: Sequence[Sequence[int]], *, bg: int
) -> List[Tuple[Tuple[Tuple[int, int], ...], Set[int]]]:
    h, w = _grid_shape(grid)
    if h <= 0 or w <= 0:
        return []
    visited: Set[Tuple[int, int]] = set()
    comps: List[Tuple[Tuple[Tuple[int, int], ...], Set[int]]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in visited:
                continue
            if int(grid[r][c]) == int(bg):
                continue
            q: List[Tuple[int, int]] = [(int(r), int(c))]
            visited.add((int(r), int(c)))
            cells: List[Tuple[int, int]] = []
            colors: Set[int] = set()
            while q:
                rr, cc = q.pop()
                cells.append((int(rr), int(cc)))
                vv = int(grid[rr][cc])
                if vv != int(bg):
                    colors.add(int(vv))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if nr < 0 or nc < 0 or nr >= h or nc >= w:
                        continue
                    if (nr, nc) in visited:
                        continue
                    if int(grid[nr][nc]) == int(bg):
                        continue
                    visited.add((nr, nc))
                    q.append((nr, nc))
            cells_sorted = tuple(sorted(cells, key=lambda rc: (int(rc[0]), int(rc[1]))))
            comps.append((cells_sorted, set(int(x) for x in colors)))
    comps.sort(key=lambda x: (-(len(x[0])), x[0]))
    return comps


@dataclass(frozen=True)
class TaskSigV141:
    failure_kind: str
    shape_rel: str
    palette_rel: str
    delta_density: str

    def as_key(self) -> str:
        return _stable_json(
            {
                "failure_kind": self.failure_kind,
                "shape_rel": self.shape_rel,
                "palette_rel": self.palette_rel,
                "delta_density": self.delta_density,
            }
        )


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_per_task_paths(run_dir: str) -> List[str]:
    per_task_dir = os.path.join(run_dir, "per_task")
    if not os.path.isdir(per_task_dir):
        raise SystemExit(f"missing_per_task_dir:{per_task_dir}")
    names: List[str] = []
    for fn in os.listdir(per_task_dir):
        if fn.endswith(".json"):
            names.append(fn)
    names.sort()
    return [os.path.join(per_task_dir, n) for n in names]


def _summarize_run_manifest(run_dir: str) -> Dict[str, Any]:
    manifest_path = os.path.join(run_dir, "per_task_manifest.jsonl")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"missing_per_task_manifest:{manifest_path}")
    total = solved = unknown = failed = 0
    failure_counts: Dict[str, int] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            status = str(row.get("status") or "")
            total += 1
            if status == "SOLVED":
                solved += 1
            elif status == "UNKNOWN":
                unknown += 1
            else:
                failed += 1
                fk = str(row.get("failure_kind") or "")
                if fk:
                    failure_counts[fk] = int(failure_counts.get(fk, 0)) + 1
    solve_rate = float(solved) / float(total) if total else 0.0
    return {
        "tasks_total": int(total),
        "solved": int(solved),
        "unknown": int(unknown),
        "failed": int(failed),
        "solve_rate": float(solve_rate),
        "failure_counts": {k: int(failure_counts[k]) for k in sorted(failure_counts.keys())},
    }


def _cluster_failures(run_dir: str, *, want_kinds: Set[str]) -> Dict[str, Any]:
    per_task_paths = _iter_per_task_paths(run_dir)
    clusters: Dict[str, int] = {}

    multicolor_tasks = 0
    multicolor_hit_tasks = 0
    colors_per_component_hist: Dict[int, int] = {}

    for p in per_task_paths:
        d = _load_json(p)
        scoring = d.get("scoring") if isinstance(d.get("scoring"), dict) else {}
        failure_kind = str(scoring.get("failure_kind") or "")
        if not failure_kind:
            continue
        if failure_kind not in want_kinds:
            continue

        task = d.get("task") if isinstance(d.get("task"), dict) else {}
        train_pairs = task.get("train_pairs") or []
        pair_shapes: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        pairs_grids: List[Tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]] = []
        pin: Set[int] = set()
        pout: Set[int] = set()
        for pair in train_pairs:
            if not isinstance(pair, dict):
                continue
            inp = pair.get("in_grid")
            out = pair.get("out_grid")
            if not isinstance(inp, list) or not isinstance(out, list):
                continue
            pair_shapes.append((_grid_shape(inp), _grid_shape(out)))
            pairs_grids.append((inp, out))
            pin |= _grid_palette(inp)
            pout |= _grid_palette(out)

        sig = TaskSigV141(
            failure_kind=str(failure_kind),
            shape_rel=_shape_rel_for_pairs(pair_shapes),
            palette_rel=_palette_rel(pin, pout),
            delta_density=_delta_density_bin(pairs_grids),
        )
        key = sig.as_key()
        clusters[key] = int(clusters.get(key, 0)) + 1

        if failure_kind == "MISSING_OPERATOR":
            # MULTICOLOR OBJECT SIGNAL (per spec): CC4 on (cell!=bg) ignoring color.
            multicolor_tasks += 1
            task_has_multicolor = False
            for inp, _out in pairs_grids:
                bg = _mode_color(inp)
                comps = _cc4_nonbg_ignore_color_components(inp, bg=int(bg))
                for _cells, cols in comps:
                    ncols = int(len(set(int(x) for x in cols)))
                    colors_per_component_hist[ncols] = int(colors_per_component_hist.get(ncols, 0)) + 1
                    if ncols > 1:
                        task_has_multicolor = True
            if task_has_multicolor:
                multicolor_hit_tasks += 1

    rows = sorted(((int(n), str(k)) for k, n in clusters.items()), key=lambda nk: (-int(nk[0]), str(nk[1])))
    top = [{"count": int(n), "signature": json.loads(k)} for n, k in rows[:15]]

    return {
        "clusters_top15": top,
        "clusters_total": int(len(clusters)),
        "multicolor_signal": {
            "total_missing_operator_tasks": int(multicolor_tasks),
            "tasks_with_multicolor_component": int(multicolor_hit_tasks),
            "tasks_without_multicolor_component": int(multicolor_tasks - multicolor_hit_tasks),
            "colors_per_component_hist": {str(k): int(colors_per_component_hist[k]) for k in sorted(colors_per_component_hist.keys())},
        },
    }


def _write_text_x(path: Path, text: str) -> None:
    if path.exists():
        raise SystemExit(f"worm_exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    run_dir = str(args.run_dir)
    out_path = Path(str(args.out_path))

    summary = _summarize_run_manifest(run_dir)
    missing = _cluster_failures(run_dir, want_kinds={"MISSING_OPERATOR"})
    budget = _cluster_failures(run_dir, want_kinds={"SEARCH_BUDGET_EXCEEDED"})

    lines: List[str] = []
    lines.append("# ARC_DIAG_V141_FROM_V140_TRAINING_FULL")
    lines.append("")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(_stable_json(summary))
    lines.append("")
    lines.append("## Top Clusters — MISSING_OPERATOR")
    lines.append("")
    for row in missing.get("clusters_top15", []):
        lines.append(f"- count={row.get('count')} sig={_stable_json(row.get('signature'))}")
    lines.append("")
    ms = missing.get("multicolor_signal", {})
    lines.append("## Multicolor Object Signal (MISSING_OPERATOR only)")
    lines.append("")
    lines.append(_stable_json(ms))
    lines.append("")
    lines.append("## Top Clusters — SEARCH_BUDGET_EXCEEDED")
    lines.append("")
    for row in budget.get("clusters_top15", []):
        lines.append(f"- count={row.get('count')} sig={_stable_json(row.get('signature'))}")
    lines.append("")
    _write_text_x(out_path, "\n".join(lines) + "\n")

    print(_stable_json({"ok": True, "out_path": str(out_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

