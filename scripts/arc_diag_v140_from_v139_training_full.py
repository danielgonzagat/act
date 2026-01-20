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
class TaskSigV140:
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
    names: List[str] = []
    for fn in os.listdir(per_task_dir):
        if fn.endswith(".json") or fn.endswith(".json.json"):
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

    budget_tried: List[int] = []
    pruned_totals: Dict[str, int] = {}
    op1_counts: Dict[str, int] = {}
    depth_counts: Dict[int, int] = {}

    for p in per_task_paths:
        d = _load_json(p)
        scoring = d.get("scoring") if isinstance(d.get("scoring"), dict) else {}
        failure_kind = str(scoring.get("failure_kind") or "")
        if not failure_kind:
            # Solved tasks set failure_kind="".
            continue
        if failure_kind not in want_kinds:
            continue

        solver_results = d.get("solver_results") or []
        if not isinstance(solver_results, list) or not solver_results:
            continue
        res = solver_results[0] if isinstance(solver_results[0], dict) else {}
        fr = res.get("failure_reason") if isinstance(res.get("failure_reason"), dict) else {}

        if failure_kind == "SEARCH_BUDGET_EXCEEDED":
            details = fr.get("details") if isinstance(fr.get("details"), dict) else {}
            budget_tried.append(int(details.get("candidates_tested") or 0))

        trace = res.get("trace") if isinstance(res.get("trace"), dict) else {}
        for k in [
            "pruned_by_shape_reachability",
            "pruned_by_palette_reachability",
            "pruned_by_dominated_state",
            "pruned_by_no_grid_modify_in_time",
        ]:
            if k in trace:
                pruned_totals[k] = int(pruned_totals.get(k, 0)) + int(trace.get(k) or 0)

        # early trace aggregates (limited trace_programs)
        for tp in trace.get("trace_programs", []) if isinstance(trace.get("trace_programs"), list) else []:
            if not isinstance(tp, dict):
                continue
            depth = int(tp.get("depth") or 0)
            depth_counts[depth] = int(depth_counts.get(depth, 0)) + 1
            if depth == 1:
                steps = tp.get("steps") if isinstance(tp.get("steps"), list) else []
                if steps:
                    op_id = str((steps[0] or {}).get("op_id") or "")
                    if op_id:
                        op1_counts[op_id] = int(op1_counts.get(op_id, 0)) + 1

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

        sig = TaskSigV140(
            failure_kind=str(failure_kind),
            shape_rel=str(shape_rel),
            palette_rel=str(palette_rel),
            delta_density=str(delta_density),
            evidence=tuple(sorted(evidence)),
        )
        key = sig.as_key()
        clusters[key] = int(clusters.get(key, 0)) + 1

    clusters_sorted = sorted(((k, int(v)) for k, v in clusters.items()), key=lambda kv: (-int(kv[1]), str(kv[0])))
    stats = {
        "n": int(len(budget_tried)),
        "min": int(min(budget_tried)) if budget_tried else 0,
        "max": int(max(budget_tried)) if budget_tried else 0,
        "avg": float(sum(budget_tried) / float(len(budget_tried))) if budget_tried else 0.0,
    }
    return {
        "clusters_sorted": clusters_sorted,
        "budget_tried_stats": stats,
        "pruned_totals": {k: int(pruned_totals[k]) for k in sorted(pruned_totals.keys())},
        "depth_counts": {str(k): int(depth_counts[k]) for k in sorted(depth_counts.keys())},
        "op1_counts_top": sorted(op1_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    run_dir = str(args.run_dir)
    out_path = str(args.out_path)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run_dir_not_found:{run_dir}")
    if os.path.exists(out_path):
        raise SystemExit(f"worm_violation_out_path_exists:{out_path}")

    run_summary = _summarize_run_manifest(run_dir)
    missing = _cluster_failures(run_dir, want_kinds={"MISSING_OPERATOR"})
    budget = _cluster_failures(run_dir, want_kinds={"SEARCH_BUDGET_EXCEEDED"})

    lines: List[str] = []
    lines.append("# ARC_DIAG_V140_FROM_V139_TRAINING_FULL")
    lines.append("")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- summary: tasks_total={run_summary['tasks_total']} solved={run_summary['solved']} unknown={run_summary['unknown']} failed={run_summary['failed']} solve_rate={run_summary['solve_rate']:.4f}")
    lines.append(f"- failure_counts={_stable_json(run_summary['failure_counts'])}")
    lines.append("")

    def emit_section(title: str, cl: Dict[str, Any], top_n: int = 10) -> None:
        lines.append(f"## {title}")
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

    emit_section("MISSING_OPERATOR clusters (top)", missing)
    emit_section("SEARCH_BUDGET_EXCEEDED clusters (top)", budget)

    # Recommendation: pick the dominant SEARCH_BUDGET_EXCEEDED cluster for architectural delta.
    top_budget = budget["clusters_sorted"][0] if budget["clusters_sorted"] else None
    lines.append("## Recommendation")
    if top_budget is None:
        lines.append("- No SEARCH_BUDGET_EXCEEDED clusters found; consider targeting MISSING_OPERATOR instead.")
    else:
        key, n = top_budget
        lines.append(f"- target_cluster: `{key}` (count={int(n)})")
        lines.append("- suggested_delta_kind: abstract_reachability_pruning")
        lines.append("- rationale: dominant failures are budget-bound; introduce sound shape/palette reachability pruning based on abstract interpretation (general program-synthesis optimization, not ARC-specific).")
    lines.append("")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
