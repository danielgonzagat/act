#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
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


def _is_tile_repeat(inp: Sequence[Sequence[int]], out: Sequence[Sequence[int]]) -> Optional[Tuple[int, int]]:
    hi, wi = _grid_shape(inp)
    ho, wo = _grid_shape(out)
    if hi <= 0 or wi <= 0:
        return None
    if ho % hi != 0 or wo % wi != 0:
        return None
    ry = int(ho // hi)
    rx = int(wo // wi)
    if ry <= 0 or rx <= 0:
        return None
    for r in range(ho):
        for c in range(wo):
            if int(out[r][c]) != int(inp[r % hi][c % wi]):
                return None
    return (int(ry), int(rx))


def _is_scale_cell(inp: Sequence[Sequence[int]], out: Sequence[Sequence[int]]) -> Optional[Tuple[int, int]]:
    hi, wi = _grid_shape(inp)
    ho, wo = _grid_shape(out)
    if hi <= 0 or wi <= 0:
        return None
    if ho % hi != 0 or wo % wi != 0:
        return None
    sy = int(ho // hi)
    sx = int(wo // wi)
    if sy <= 0 or sx <= 0:
        return None
    for r in range(ho):
        for c in range(wo):
            if int(out[r][c]) != int(inp[r // sy][c // sx]):
                return None
    return (int(sy), int(sx))


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


@dataclass(frozen=True)
class TaskSigV138:
    failure_kind: str
    shape_rel: str
    palette_rel: str
    delta_density: str
    evidence: Tuple[str, ...]

    def as_key(self) -> str:
        return _stable_json(
            {
                "failure_kind": self.failure_kind,
                "shape_rel": self.shape_rel,
                "palette_rel": self.palette_rel,
                "delta_density": self.delta_density,
                "evidence": list(self.evidence),
            }
        )


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_per_task_paths(run_dir: str) -> List[str]:
    per_task_dir = os.path.join(run_dir, "per_task")
    names = []
    for fn in os.listdir(per_task_dir):
        if fn.endswith(".json") or fn.endswith(".json.json"):
            names.append(fn)
    names.sort()
    return [os.path.join(per_task_dir, n) for n in names]


def _summarize_run_manifest(run_dir: str) -> Dict[str, Any]:
    # Derive counts from per_task_manifest.jsonl for determinism (summary.json may omit counts in older runs).
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

    pruned_totals: Dict[str, int] = {}
    tried_counts: List[int] = []
    depth_counts: Dict[int, int] = {}
    op1_counts: Dict[str, int] = {}

    for p in per_task_paths:
        d = _load_json(p)
        res = d.get("result") or {}
        status = str(res.get("status") or "")
        fr = res.get("failure_reason") or {}
        failure_kind = str(fr.get("kind") or ("SOLVED" if status == "SOLVED" else "UNKNOWN" if status == "UNKNOWN" else "FAIL"))
        if failure_kind not in want_kinds:
            continue

        if failure_kind == "SEARCH_BUDGET_EXCEEDED":
            tried_counts.append(int((fr.get("details") or {}).get("candidates_tested") or 0))

        trace = res.get("trace") if isinstance(res.get("trace"), dict) else {}
        for k in [
            "pruned_by_shape_reachability",
            "pruned_by_palette_reachability",
            "pruned_by_dominated_state",
            "pruned_by_no_grid_modify_in_time",
        ]:
            if k in trace:
                pruned_totals[k] = int(pruned_totals.get(k, 0)) + int(trace.get(k) or 0)

        # small early-trace aggregates
        for tp in trace.get("trace_programs", []) if isinstance(trace.get("trace_programs"), list) else []:
            if not isinstance(tp, dict):
                continue
            depth = int(tp.get("depth") or 0)
            depth_counts[depth] = int(depth_counts.get(depth, 0)) + 1
            if depth == 1:
                steps = tp.get("steps") or []
                if isinstance(steps, list) and steps:
                    op_id = str((steps[0] or {}).get("op_id") or "")
                    if op_id:
                        op1_counts[op_id] = int(op1_counts.get(op_id, 0)) + 1

        task = d.get("task") or {}
        train_pairs = task.get("train_pairs") or []
        pair_shapes: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        pairs_grids: List[Tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]] = []
        pin: Set[int] = set()
        pout: Set[int] = set()
        for pair in train_pairs:
            inp = pair.get("in_grid")
            out = pair.get("out_grid")
            if inp is None or out is None:
                continue
            pair_shapes.append((_grid_shape(inp), _grid_shape(out)))
            pairs_grids.append((inp, out))
            pin |= _grid_palette(inp)
            pout |= _grid_palette(out)

        shape_rel = _shape_rel_for_pairs(pair_shapes)
        palette_rel = _palette_rel(pin, pout)
        delta_density = _delta_density_bin(pairs_grids)

        evidence: List[str] = []
        if shape_rel.startswith("scale_integer:"):
            tile_ok = True
            tile_ratio: Optional[Tuple[int, int]] = None
            cell_ok = True
            cell_ratio: Optional[Tuple[int, int]] = None
            for inp, out in pairs_grids:
                tr = _is_tile_repeat(inp, out)
                cr = _is_scale_cell(inp, out)
                if tr is None:
                    tile_ok = False
                else:
                    tile_ratio = tr if tile_ratio is None else tile_ratio
                    if tile_ratio != tr:
                        tile_ok = False
                if cr is None:
                    cell_ok = False
                else:
                    cell_ratio = cr if cell_ratio is None else cell_ratio
                    if cell_ratio != cr:
                        cell_ok = False
            if tile_ok and tile_ratio is not None:
                evidence.append(f"tile_repeat:{tile_ratio[0]}x{tile_ratio[1]}")
            if cell_ok and cell_ratio is not None:
                evidence.append(f"scale_cell:{cell_ratio[0]}x{cell_ratio[1]}")

        sig = TaskSigV138(
            failure_kind=str(failure_kind),
            shape_rel=str(shape_rel),
            palette_rel=str(palette_rel),
            delta_density=str(delta_density),
            evidence=tuple(sorted(evidence)),
        )
        key = sig.as_key()
        clusters[key] = int(clusters.get(key, 0)) + 1

    clusters_sorted = sorted(((k, int(v)) for k, v in clusters.items()), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return {
        "clusters_sorted": clusters_sorted,
        "pruned_totals": {k: int(pruned_totals[k]) for k in sorted(pruned_totals.keys())},
        "budget_tried_stats": {
            "n": int(len(tried_counts)),
            "min": int(min(tried_counts)) if tried_counts else 0,
            "max": int(max(tried_counts)) if tried_counts else 0,
            "avg": float(sum(tried_counts) / float(len(tried_counts))) if tried_counts else 0.0,
        },
        "depth_counts": {str(k): int(depth_counts[k]) for k in sorted(depth_counts.keys())},
        "op1_counts_top": sorted(op1_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:10],
    }


def _parse_run_arg(s: str) -> Tuple[str, str]:
    if "=" not in s:
        raise SystemExit(f"bad_run_arg:{s}")
    k, v = s.split("=", 1)
    if not k or not v:
        raise SystemExit(f"bad_run_arg:{s}")
    return (str(k), str(v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="label=run_dir (repeatable)")
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    out_path = str(args.out_path)
    if os.path.exists(out_path):
        raise SystemExit(f"worm_violation_out_path_exists:{out_path}")

    runs: List[Tuple[str, str]] = []
    for x in args.run:
        runs.append(_parse_run_arg(str(x)))
    if not runs:
        raise SystemExit("no_runs_provided")

    run_summaries: Dict[str, Any] = {}
    clusters_missing: Dict[str, Any] = {}
    clusters_budget: Dict[str, Any] = {}

    for label, run_dir in runs:
        if not os.path.isdir(run_dir):
            raise SystemExit(f"run_dir_not_found:{run_dir}")
        run_summaries[label] = _summarize_run_manifest(run_dir)
        clusters_missing[label] = _cluster_failures(run_dir, want_kinds={"MISSING_OPERATOR"})
        clusters_budget[label] = _cluster_failures(run_dir, want_kinds={"SEARCH_BUDGET_EXCEEDED"})

    lines: List[str] = []
    lines.append("# ARC_DIAG_REPORT_v138_PRE")
    lines.append("")
    lines.append("## Run summaries (derived from per_task_manifest.jsonl)")
    for label in sorted(run_summaries.keys()):
        s = run_summaries[label]
        lines.append(f"### {label}")
        lines.append(f"- tasks_total={s['tasks_total']} solved={s['solved']} unknown={s['unknown']} failed={s['failed']} solve_rate={s['solve_rate']:.4f}")
        lines.append(f"- failure_counts={_stable_json(s['failure_counts'])}")
        lines.append("")

    def emit_clusters(*, title: str, data: Dict[str, Any], top_n: int = 15) -> None:
        lines.append(f"## {title}")
        for label in sorted(data.keys()):
            cl = data[label]
            lines.append(f"### {label}")
            lines.append(f"- budget_tried_stats={_stable_json(cl['budget_tried_stats'])}")
            lines.append(f"- pruned_totals={_stable_json(cl['pruned_totals'])}")
            lines.append(f"- depth_counts={_stable_json(cl['depth_counts'])}")
            lines.append(f"- op1_counts_top={_stable_json(cl['op1_counts_top'])}")
            lines.append("")
            lines.append("| count | signature |")
            lines.append("|---:|---|")
            for key, n in cl["clusters_sorted"][: int(top_n)]:
                lines.append(f"| {int(n)} | `{key}` |")
            lines.append("")

    emit_clusters(title="MISSING_OPERATOR clusters (top)", data=clusters_missing)
    emit_clusters(title="SEARCH_BUDGET_EXCEEDED clusters (top)", data=clusters_budget)

    # Decision section: intended for V138 selection; keep non-task-specific.
    lines.append("## Decision (V138 delta)")
    lines.append("- delta_kind: harness_protocol_fix")
    lines.append("- rationale: V137 canonical tasks omit test outputs, so solve_rate is not ARC-protocol scoring; V138 will add test_out for scoring-only and support 2-attempt evaluation without exposing test_out to solver.")
    lines.append("")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

