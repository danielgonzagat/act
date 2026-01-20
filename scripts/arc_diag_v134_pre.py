from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.grid_v124 import GridV124, grid_shape_v124


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _grid_palette(g: GridV124) -> List[int]:
    out: List[int] = []
    seen: set = set()
    for row in g:
        for x in row:
            xx = int(x)
            if xx not in seen:
                seen.add(xx)
                out.append(xx)
    out.sort()
    return out


def _grid_diff_cells(inp: GridV124, out: GridV124) -> Optional[int]:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if (hi, wi) != (ho, wo):
        return None
    diff = 0
    for r in range(hi):
        for c in range(wi):
            if int(inp[r][c]) != int(out[r][c]):
                diff += 1
    return int(diff)


def _diff_bbox(inp: GridV124, out: GridV124) -> Optional[Tuple[int, int, int, int]]:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if (hi, wi) != (ho, wo) or hi == 0 or wi == 0:
        return None
    rmin = hi
    cmin = wi
    rmax = -1
    cmax = -1
    any_diff = False
    for r in range(hi):
        for c in range(wi):
            if int(inp[r][c]) != int(out[r][c]):
                any_diff = True
                rmin = min(rmin, int(r))
                cmin = min(cmin, int(c))
                rmax = max(rmax, int(r))
                cmax = max(cmax, int(c))
    if not any_diff:
        return None
    # half-open
    return int(rmin), int(cmin), int(rmax + 1), int(cmax + 1)


def _is_tile(inp: GridV124, out: GridV124, *, reps_h: int, reps_w: int) -> bool:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if hi <= 0 or wi <= 0:
        return False
    if ho != hi * int(reps_h) or wo != wi * int(reps_w):
        return False
    for r in range(ho):
        for c in range(wo):
            if int(out[r][c]) != int(inp[r % hi][c % wi]):
                return False
    return True


def _is_upscale_nn(inp: GridV124, out: GridV124, *, sy: int, sx: int) -> bool:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if hi <= 0 or wi <= 0:
        return False
    if ho != hi * int(sy) or wo != wi * int(sx):
        return False
    for r in range(ho):
        for c in range(wo):
            if int(out[r][c]) != int(inp[r // int(sy)][c // int(sx)]):
                return False
    return True


def _shape_relation(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> str:
    # Derive a canonical shape relation class across all train pairs.
    rels: List[str] = []
    for inp, out in train_pairs:
        hi, wi = grid_shape_v124(inp)
        ho, wo = grid_shape_v124(out)
        if (hi, wi) == (ho, wo):
            rels.append("same")
            continue
        if hi > 0 and wi > 0 and ho % hi == 0 and wo % wi == 0:
            ry = int(ho // hi)
            rx = int(wo // wi)
            if _is_upscale_nn(inp, out, sy=ry, sx=rx):
                rels.append(f"upscale_nn:{ry}x{rx}")
            elif _is_tile(inp, out, reps_h=ry, reps_w=rx):
                rels.append(f"tile:{ry}x{rx}")
            else:
                rels.append(f"multiple:{ry}x{rx}")
            continue
        if ho <= hi and wo <= wi:
            rels.append("crop_like")
            continue
        if ho >= hi and wo >= wi:
            rels.append("pad_like")
            continue
        rels.append("other")
    rels_sorted = sorted(set(rels))
    return rels_sorted[0] if len(rels_sorted) == 1 else "mixed:" + ",".join(rels_sorted)


def _palette_relation(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> str:
    rels: List[str] = []
    for inp, out in train_pairs:
        pi = set(_grid_palette(inp))
        po = set(_grid_palette(out))
        added = sorted(set(po - pi))
        removed = sorted(set(pi - po))
        if not added and not removed:
            rels.append("same")
        else:
            rels.append(f"added={','.join(map(str,added))};removed={','.join(map(str,removed))}")
    rels_sorted = sorted(set(rels))
    return rels_sorted[0] if len(rels_sorted) == 1 else "mixed:" + "|".join(rels_sorted)


def _delta_density(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> str:
    dens: List[str] = []
    for inp, out in train_pairs:
        hi, wi = grid_shape_v124(inp)
        diff = _grid_diff_cells(inp, out)
        if diff is None or hi * wi == 0:
            dens.append("shape_change")
            continue
        ratio = float(diff) / float(hi * wi)
        if ratio <= 0.1:
            dens.append("sparse")
        elif ratio <= 0.3:
            dens.append("local")
        else:
            dens.append("dense")
    dens_sorted = sorted(set(dens))
    return dens_sorted[0] if len(dens_sorted) == 1 else "mixed:" + ",".join(dens_sorted)


def _rect_evidence(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> str:
    tags: List[str] = []
    for inp, out in train_pairs:
        bb = _diff_bbox(inp, out)
        if bb is None:
            tags.append("none")
            continue
        r0, c0, r1, c1 = bb
        hi, wi = grid_shape_v124(inp)
        diff_cells = 0
        for r in range(hi):
            for c in range(wi):
                if int(inp[r][c]) != int(out[r][c]):
                    diff_cells += 1
        area = int(max(0, r1 - r0) * max(0, c1 - c0))
        if area <= 0:
            tags.append("none")
            continue
        border = int(2 * (max(0, r1 - r0)) + 2 * (max(0, c1 - c0)) - 4) if (r1 - r0) >= 2 and (c1 - c0) >= 2 else area
        if diff_cells == area:
            tags.append("rect_fill")
        elif diff_cells == border:
            tags.append("rect_border")
        else:
            tags.append("bbox_other")
    tags_sorted = sorted(set(tags))
    return tags_sorted[0] if len(tags_sorted) == 1 else "mixed:" + ",".join(tags_sorted)


def _suggest_operator(shape_rel: str, rect_ev: str) -> Optional[str]:
    if shape_rel.startswith("upscale_nn:"):
        return "repeat_grid(mode=cell)"
    if shape_rel.startswith("tile:"):
        return "repeat_grid(mode=grid)"
    if rect_ev in ("rect_fill", "rect_border"):
        return "bbox_by_color + paint_rect/draw_rect_border"
    return None


def _iter_per_task_json(run_dir: Path) -> Iterable[Path]:
    per_task_dir = run_dir / "per_task"
    for p in sorted(per_task_dir.glob("*.json"), key=lambda x: x.name):
        yield p


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", action="append", required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)

    run_dirs = [Path(p).resolve() for p in args.run_dir]
    out_path = Path(str(args.out_path)).resolve()
    if out_path.exists():
        raise SystemExit("worm_violation_out_exists")

    rows: List[Dict[str, Any]] = []
    for rd in run_dirs:
        for p in _iter_per_task_json(rd):
            obj = json.loads(p.read_text(encoding="utf-8"))
            task = obj.get("task") or {}
            result = obj.get("result") or {}
            status = str(result.get("status") or "")
            fr = result.get("failure_reason")
            failure_kind = ""
            if isinstance(fr, dict):
                failure_kind = str(fr.get("kind") or "")
            if status == "SOLVED":
                continue
            train = task.get("train_pairs") or task.get("train") or []
            # v133 per_task stores canonical task.to_dict(), which has train_pairs list of {"in_grid":...,"out_grid":...}
            train_pairs: List[Tuple[GridV124, GridV124]] = []
            if isinstance(train, list) and train and isinstance(train[0], dict) and "in_grid" in train[0]:
                for pair in train:
                    inp = tuple(tuple(int(x) for x in row) for row in pair["in_grid"])
                    out = tuple(tuple(int(x) for x in row) for row in pair["out_grid"])
                    train_pairs.append((inp, out))
            else:
                # fallback for raw ARC json shape
                for pair in train:
                    inp = tuple(tuple(int(x) for x in row) for row in pair["input"])
                    out = tuple(tuple(int(x) for x in row) for row in pair["output"])
                    train_pairs.append((inp, out))

            shape_rel = _shape_relation(train_pairs)
            palette_rel = _palette_relation(train_pairs)
            dens = _delta_density(train_pairs)
            rect_ev = _rect_evidence(train_pairs)
            suggestion = _suggest_operator(shape_rel, rect_ev)

            rows.append(
                {
                    "task_id": str(task.get("task_id") or obj.get("task_id") or p.name),
                    "run_dir": str(rd),
                    "status": str(status),
                    "failure_kind": str(failure_kind),
                    "shape_rel": str(shape_rel),
                    "palette_rel": str(palette_rel),
                    "delta_density": str(dens),
                    "rect_evidence": str(rect_ev),
                    "suggestion": str(suggestion) if suggestion is not None else "",
                }
            )

    # Group.
    groups: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = (
            f"{r['failure_kind']}|shape={r['shape_rel']}|palette={r['palette_rel']}|delta={r['delta_density']}|rect={r['rect_evidence']}"
        )
        g = groups.get(key)
        if g is None:
            g = {"count": 0, "tasks": [], "suggestion": r.get("suggestion") or ""}
            groups[key] = g
        g["count"] = int(g["count"]) + 1
        g["tasks"].append(str(r["task_id"]))

    # Render markdown (deterministic).
    lines: List[str] = []
    lines.append("# ARC_DIAG_REPORT_v134_PRE")
    lines.append("")
    lines.append(f"- inputs_sha256:")
    for rd in sorted(set(str(x) for x in run_dirs)):
        lines.append(f"  - {rd}")
    lines.append("")
    lines.append("## Failure Groups (structural)")
    lines.append("")
    lines.append("| count | failure_kind | shape | palette | delta | rect | suggestion | tasks |")
    lines.append("|---:|---|---|---|---|---|---|---|")
    for k in sorted(groups.keys()):
        g = groups[k]
        parts = k.split("|")
        fk = parts[0]
        shape = parts[1].split("=", 1)[1]
        palette = parts[2].split("=", 1)[1]
        delta = parts[3].split("=", 1)[1]
        rect = parts[4].split("=", 1)[1]
        tasks = ",".join(sorted(set(g["tasks"])))
        sugg = str(g.get("suggestion") or "")
        lines.append(f"| {int(g['count'])} | {fk} | {shape} | {palette} | {delta} | {rect} | {sugg} | {tasks} |")
    lines.append("")
    body = "\n".join(lines).strip() + "\n"
    sig = sha256_hex(canonical_json_dumps({"kind": "arc_diag_v134_pre", "body": body}).encode("utf-8"))
    body += f"\n<!-- report_sig={sig} -->\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as f:
        f.write(body)
    print(
        json.dumps(
            {
                "kind": "arc_diag_v134_pre_run",
                "ok": True,
                "out_path": str(out_path),
                "out_sha256": _sha256_file(out_path),
                "report_sig": str(sig),
                "groups": len(groups),
                "rows": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
