from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


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
    # Only meaningful for same-shape pairs.
    diffs: List[float] = []
    for inp, out in pairs:
        d = _grid_diff_cells_same_shape(inp, out)
        if d is None:
            continue
        h, w = _grid_shape(inp)
        denom = float(max(1, h * w))
        diffs.append(float(d) / denom)
    if not diffs:
        return "n/a"
    avg = sum(diffs) / float(len(diffs))
    if avg <= 0.10:
        return "sparse<=0.10"
    if avg <= 0.30:
        return "local<=0.30"
    return "dense>0.30"


@dataclass(frozen=True)
class TaskSigV136:
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


def _summarize_trace_programs(trace_programs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    depth_counts: Dict[str, int] = {}
    op1_counts: Dict[str, int] = {}
    mismatch_kinds: Dict[str, int] = {}
    for row in trace_programs:
        depth = int(row.get("depth") or 0)
        depth_counts[str(depth)] = int(depth_counts.get(str(depth), 0)) + 1
        mm = row.get("mismatch") or {}
        mk = str(mm.get("kind") or "")
        if mk:
            mismatch_kinds[mk] = int(mismatch_kinds.get(mk, 0)) + 1
        steps = row.get("steps") or []
        if depth == 1 and isinstance(steps, list) and steps:
            op_id = str((steps[0] or {}).get("op_id") or "")
            if op_id:
                op1_counts[op_id] = int(op1_counts.get(op_id, 0)) + 1
    return {
        "depth_counts": {k: int(depth_counts[k]) for k in sorted(depth_counts.keys(), key=lambda x: int(x))},
        "op1_counts_top": sorted(op1_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:10],
        "mismatch_kinds_top": sorted(mismatch_kinds.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:10],
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

    per_task_paths = _iter_per_task_paths(run_dir)
    if not per_task_paths:
        raise SystemExit(f"no_per_task_files:{run_dir}")

    clusters: Dict[str, int] = {}
    cluster_examples: Dict[str, List[str]] = {}
    failure_counts: Dict[str, int] = {}
    tried_counts: List[int] = []
    per_depth_agg: Dict[int, int] = {}
    op1_agg: Dict[str, int] = {}

    for p in per_task_paths:
        d = _load_json(p)
        task = d.get("task") or {}
        task_id = str(task.get("task_id") or os.path.basename(p))
        res = d.get("result") or {}
        status = str(res.get("status") or "")
        fr = res.get("failure_reason") or {}
        fk = str(fr.get("kind") or ("SOLVED" if status == "SOLVED" else "UNKNOWN" if status == "UNKNOWN" else "FAIL"))
        failure_counts[fk] = int(failure_counts.get(fk, 0)) + 1
        if fk == "SEARCH_BUDGET_EXCEEDED":
            tried_counts.append(int((fr.get("details") or {}).get("candidates_tested") or 0))

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
            inp = inp
            out = out
            pair_shapes.append((_grid_shape(inp), _grid_shape(out)))
            pairs_grids.append((inp, out))
            pin |= _grid_palette(inp)
            pout |= _grid_palette(out)

        shape_rel = _shape_rel_for_pairs(pair_shapes)
        palette_rel = _palette_rel(pin, pout)
        delta_density = _delta_density_bin(pairs_grids)

        # Evidence detectors (general, no task_id dependence).
        evidence: List[str] = []
        if shape_rel.startswith("scale_integer:"):
            # check whether it's a tile-repeat or cell-scale (across train pairs)
            tile_ok = True
            tile_ratio: Optional[Tuple[int, int]] = None
            cell_ok = True
            cell_ratio: Optional[Tuple[int, int]] = None
            for inp, out in pairs_grids:
                tr = _is_tile_repeat(inp, out)
                if tr is None:
                    tile_ok = False
                else:
                    tile_ratio = tr if tile_ratio is None else tile_ratio
                cr = _is_scale_cell(inp, out)
                if cr is None:
                    cell_ok = False
                else:
                    cell_ratio = cr if cell_ratio is None else cell_ratio
            if tile_ok and tile_ratio is not None:
                evidence.append(f"tile_repeat:{tile_ratio[0]}x{tile_ratio[1]}")
            if cell_ok and cell_ratio is not None:
                evidence.append(f"scale_cell:{cell_ratio[0]}x{cell_ratio[1]}")
        evidence.sort()

        sig = TaskSigV136(
            failure_kind=fk,
            shape_rel=shape_rel,
            palette_rel=palette_rel,
            delta_density=delta_density,
            evidence=tuple(evidence),
        )
        ck = sig.as_key()
        clusters[ck] = int(clusters.get(ck, 0)) + 1
        if ck not in cluster_examples:
            cluster_examples[ck] = []
        if len(cluster_examples[ck]) < 5:
            cluster_examples[ck].append(task_id)

        # Aggregate depth/op stats from trace_programs (first N only; still indicative).
        trace = res.get("trace") or {}
        tps = trace.get("trace_programs") or []
        if isinstance(tps, list):
            s = _summarize_trace_programs(tps)
            for k, v in (s.get("depth_counts") or {}).items():
                per_depth_agg[int(k)] = int(per_depth_agg.get(int(k), 0)) + int(v)
            for op_id, cnt in (s.get("op1_counts_top") or []):
                op1_agg[str(op_id)] = int(op1_agg.get(str(op_id), 0)) + int(cnt)

    clusters_sorted = sorted(
        clusters.items(),
        key=lambda kv: (-int(kv[1]), str(kv[0])),
    )

    per_depth_sorted = sorted(per_depth_agg.items(), key=lambda kv: int(kv[0]))
    op1_sorted = sorted(op1_agg.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:20]

    md_lines: List[str] = []
    md_lines.append("# ARC_DIAG_REPORT_v136_PRE")
    md_lines.append("")
    md_lines.append(f"- run_dir: `{run_dir}`")
    md_lines.append(f"- out_path: `{out_path}`")
    md_lines.append("")
    md_lines.append("## Failure counts")
    md_lines.append("```json")
    md_lines.append(_stable_json({k: int(failure_counts[k]) for k in sorted(failure_counts.keys())}))
    md_lines.append("```")
    md_lines.append("")
    if tried_counts:
        md_lines.append("## SEARCH_BUDGET_EXCEEDED stats")
        md_lines.append(
            f"- tasks: {len(tried_counts)} min={min(tried_counts)} p50={sorted(tried_counts)[len(tried_counts)//2]} max={max(tried_counts)}"
        )
        md_lines.append("")
    md_lines.append("## Clustered failures (by kind + structural signature)")
    md_lines.append("")
    md_lines.append("| count | failure_kind | shape_rel | palette_rel | delta_density | evidence | examples |")
    md_lines.append("|---:|---|---|---|---|---|---|")
    for ck, cnt in clusters_sorted[:50]:
        obj = json.loads(ck)
        ex = ", ".join(cluster_examples.get(ck, []))
        md_lines.append(
            f"| {int(cnt)} | {obj['failure_kind']} | {obj['shape_rel']} | {obj['palette_rel']} | {obj['delta_density']} | {','.join(obj['evidence'])} | {ex} |"
        )
    md_lines.append("")
    md_lines.append("## Early trace aggregates (from per_task.trace.trace_programs)")
    md_lines.append("")
    md_lines.append("### depth_counts (aggregated)")
    md_lines.append("```json")
    md_lines.append(_stable_json([{"depth": int(d), "count": int(c)} for d, c in per_depth_sorted]))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("### top op_id at depth=1 (aggregated, from trace_programs sample)")
    md_lines.append("```json")
    md_lines.append(_stable_json([{"op_id": str(op), "count": int(c)} for op, c in op1_sorted]))
    md_lines.append("```")
    md_lines.append("")

    report = "\n".join(md_lines) + "\n"
    report_sig = _sha256_hex(report.encode("utf-8"))

    # Write report WORM.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "x", encoding="utf-8") as f:
        f.write(report)

    stdout_obj = {
        "kind": "arc_diag_v136_pre",
        "run_dir": str(run_dir),
        "out_path": str(out_path),
        "report_sig": str(report_sig),
        "failure_counts": {k: int(failure_counts[k]) for k in sorted(failure_counts.keys())},
        "clusters_total": int(len(clusters)),
    }
    print(json.dumps(stdout_obj, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()

