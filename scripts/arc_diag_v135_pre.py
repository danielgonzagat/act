#!/usr/bin/env python3
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
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
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


def _shape_relation(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> str:
    rels: List[str] = []
    for inp, out in train_pairs:
        hi, wi = grid_shape_v124(inp)
        ho, wo = grid_shape_v124(out)
        if (hi, wi) == (ho, wo):
            rels.append("same")
        else:
            rels.append("shape_change")
    rels_sorted = sorted(set(rels))
    return rels_sorted[0] if len(rels_sorted) == 1 else "mixed"


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


def _overlay_self_translate(inp: GridV124, *, dx: int, dy: int, pad: int) -> GridV124:
    hi, wi = grid_shape_v124(inp)
    out: List[List[int]] = [[int(inp[r][c]) for c in range(wi)] for r in range(hi)]
    for r in range(hi):
        for c in range(wi):
            rr = int(r - int(dy))
            cc = int(c - int(dx))
            v = int(pad)
            if 0 <= rr < hi and 0 <= cc < wi:
                v = int(inp[rr][cc])
            if v != int(pad):
                out[r][c] = int(v)
    return tuple(tuple(int(x) for x in row) for row in out)


def _has_overlay_self_translate_evidence(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> bool:
    # Diagnostic-only detector: tries to find any (dx,dy,pad) that exactly matches all train_pairs.
    # Deterministic candidate enumeration (small grids).
    if not train_pairs:
        return False
    shapes = {grid_shape_v124(inp) for inp, _ in train_pairs} | {grid_shape_v124(out) for _, out in train_pairs}
    if len(shapes) != 1:
        return False
    h, w = list(shapes)[0]
    pads: List[int] = [0]
    for inp, out in train_pairs:
        for g in (inp, out):
            hh, ww = grid_shape_v124(g)
            if hh > 0 and ww > 0:
                pads.extend([int(g[0][0]), int(g[0][ww - 1]), int(g[hh - 1][0]), int(g[hh - 1][ww - 1])])
    pads = sorted(set(int(p) for p in pads))

    for pad in pads:
        for dy in range(int(-(h - 1)), int(h)):
            for dx in range(int(-(w - 1)), int(w)):
                if dx == 0 and dy == 0:
                    continue
                ok = True
                for inp, out in train_pairs:
                    got = _overlay_self_translate(inp, dx=int(dx), dy=int(dy), pad=int(pad))
                    if got != out:
                        ok = False
                        break
                if ok:
                    return True
    return False


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
            if status == "SOLVED":
                continue
            fr = result.get("failure_reason")
            failure_kind = str(fr.get("kind") or "") if isinstance(fr, dict) else ""

            train = task.get("train_pairs") or []
            train_pairs: List[Tuple[GridV124, GridV124]] = []
            for pair in train:
                inp = tuple(tuple(int(v) for v in row) for row in pair.get("in_grid"))
                out = tuple(tuple(int(v) for v in row) for row in pair.get("out_grid"))
                train_pairs.append((inp, out))

            shape_rel = _shape_relation(train_pairs)
            pal_rel = _palette_relation(train_pairs)
            dens = _delta_density(train_pairs)
            overlay_ev = bool(_has_overlay_self_translate_evidence(train_pairs)) if shape_rel == "same" else False

            rows.append(
                {
                    "failure_kind": str(failure_kind),
                    "shape_rel": str(shape_rel),
                    "palette_rel": str(pal_rel),
                    "delta_density": str(dens),
                    "overlay_self_translate_evidence": bool(overlay_ev),
                }
            )

    # Group counts deterministically.
    counts: Dict[str, int] = {}
    for r in rows:
        key = canonical_json_dumps(r)
        counts[key] = int(counts.get(key, 0)) + 1

    lines: List[str] = []
    lines.append("# ARC_DIAG_REPORT_v135_PRE")
    lines.append("")
    lines.append("Inputs:")
    for rd in run_dirs:
        lines.append(f"- {rd.as_posix()}")
    lines.append("")
    lines.append("## Failure clusters (failure_kind + structural signature)")
    if not counts:
        lines.append("- (none)")
    else:
        for k in sorted(counts.keys(), key=lambda x: (-int(counts[x]), x)):
            obj = json.loads(k)
            lines.append(f"- count={counts[k]} signature={k}")
            if bool(obj.get("overlay_self_translate_evidence")):
                lines.append("  - recommended_operator: overlay_self_translate(dx,dy,pad)")
    lines.append("")
    body = "\n".join(lines) + "\n"
    report_sig = sha256_hex(body.encode("utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x", encoding="utf-8") as f:
        f.write(body)
        f.write(f"\nreport_sig={report_sig}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

