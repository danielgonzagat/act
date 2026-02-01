#!/usr/bin/env python3
from __future__ import annotations

"""
Generate a deterministic, ARC-format synthetic grid→grid dataset for *learn-mode* CSG mining.

Why this exists:
- ARC eval is fail-closed + concept-as-policy; strict CSV gating can yield "no root actions" when
  only slot-building primitives exist (dl==0) but no loss-reducing CSG bundles exist yet.
- We must mine *self-contained, loss-reducing* CSGs (multi-step pipelines) from non-ARC domains.
- This generator creates many tasks per family so mined patterns have cross-task support.

Constraints:
- No ARC task IDs, no dataset inspection, no per-task hacks.
- Deterministic given --seed.
- Output is ARC JSON files: {"train":[...], "test":[...]}.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.arc_ops_v141 import StateV132, apply_op_v141
from atos_core.grid_v124 import GridV124, grid_from_list_v124, grid_shape_v124


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ensure_empty_dir(path: Path) -> None:
    if path.exists():
        # WORM-safe: never overwrite.
        if any(path.iterdir()):
            raise SystemExit(f"worm_exists_nonempty:{path}")
    else:
        path.mkdir(parents=True, exist_ok=True)


def _to_ll(g: GridV124) -> List[List[int]]:
    return [[int(x) for x in row] for row in g]


def _rand_int(rng: random.Random, lo: int, hi: int) -> int:
    return int(rng.randrange(int(lo), int(hi) + 1))


def _pick_color(rng: random.Random, *, avoid: Sequence[int]) -> int:
    avoid_s = {int(x) for x in avoid}
    cand = [c for c in range(10) if int(c) not in avoid_s]
    if not cand:
        return 0
    return int(cand[int(rng.randrange(0, len(cand)))])


def _blank_grid(*, h: int, w: int, bg: int) -> List[List[int]]:
    return [[int(bg) for _ in range(int(w))] for _ in range(int(h))]


def _place_rect(
    g: List[List[int]],
    *,
    color: int,
    top: int,
    left: int,
    height: int,
    width: int,
) -> None:
    h = len(g)
    w = len(g[0]) if h else 0
    for r in range(int(top), int(top + height)):
        for c in range(int(left), int(left + width)):
            if 0 <= int(r) < int(h) and 0 <= int(c) < int(w):
                g[int(r)][int(c)] = int(color)


def _make_input_blob(
    rng: random.Random,
    *,
    bg: int,
    seed_color: int,
    h: int,
    w: int,
    distractor_colors: Sequence[int],
) -> List[List[int]]:
    g = _blank_grid(h=h, w=w, bg=bg)
    # Main seed blob (rectangle-ish).
    rh = _rand_int(rng, 2, max(2, min(6, h - 1)))
    rw = _rand_int(rng, 2, max(2, min(6, w - 1)))
    top = _rand_int(rng, 0, max(0, h - rh))
    left = _rand_int(rng, 0, max(0, w - rw))
    _place_rect(g, color=seed_color, top=top, left=left, height=rh, width=rw)

    # Optional: sparse distractor rectangles that should remain unchanged.
    for dc in list(distractor_colors)[:2]:
        if int(dc) == int(bg) or int(dc) == int(seed_color):
            continue
        dh = _rand_int(rng, 1, 3)
        dw = _rand_int(rng, 1, 3)
        dt = _rand_int(rng, 0, max(0, h - dh))
        dl = _rand_int(rng, 0, max(0, w - dw))
        _place_rect(g, color=int(dc), top=dt, left=dl, height=dh, width=dw)
    return g


def _run_program(inp: List[List[int]], *, program: Sequence[Tuple[str, Dict[str, Any]]]) -> List[List[int]]:
    st = StateV132(grid=grid_from_list_v124(inp))
    for op_id, args in program:
        st = apply_op_v141(state=st, op_id=str(op_id), args=dict(args))
    return _to_ll(st.grid)


def _family_mask_paint_overwrite(*, seed_color: int, paint_color: int) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("mask_by_color", {"color": int(seed_color)}),
        ("paint_mask", {"color": int(paint_color), "mode": "overwrite", "bg": 0}),
    ]


def _family_mask_outline_paint(*, seed_color: int, paint_color: int) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("mask_by_color", {"color": int(seed_color)}),
        ("mask_outline", {}),
        ("paint_mask", {"color": int(paint_color), "mode": "overwrite", "bg": 0}),
    ]


def _family_mask_dilate1_paint(*, seed_color: int, paint_color: int) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("mask_by_color", {"color": int(seed_color)}),
        ("mask_dilate", {"steps": 1}),
        ("paint_mask", {"color": int(paint_color), "mode": "overwrite", "bg": 0}),
    ]


def _family_mask_box_dilate1_paint(*, seed_color: int, paint_color: int) -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("mask_by_color", {"color": int(seed_color)}),
        ("mask_box_dilate", {"radius": 1}),
        ("paint_mask", {"color": int(paint_color), "mode": "overwrite", "bg": 0}),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Output dir (WORM: must be empty or absent).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks_per_family", type=int, default=100)
    ap.add_argument("--train_pairs", type=int, default=3)
    ap.add_argument("--test_pairs", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(str(args.out_dir)).resolve()
    _ensure_empty_dir(out_dir)

    rng = random.Random(int(args.seed))

    families: List[Tuple[str, Callable[[int, int], List[Tuple[str, Dict[str, Any]]]]]] = []
    families.append(("mask_paint_overwrite", lambda s, p: _family_mask_paint_overwrite(seed_color=s, paint_color=p)))
    families.append(("mask_outline_paint", lambda s, p: _family_mask_outline_paint(seed_color=s, paint_color=p)))
    families.append(("mask_dilate1_paint", lambda s, p: _family_mask_dilate1_paint(seed_color=s, paint_color=p)))
    families.append(("mask_box_dilate1_paint", lambda s, p: _family_mask_box_dilate1_paint(seed_color=s, paint_color=p)))

    idx = 0
    for fam_name, fam_prog in families:
        for j in range(int(args.tasks_per_family)):
            idx += 1
            bg = 0
            seed_color = _pick_color(rng, avoid=[bg])
            paint_color = _pick_color(rng, avoid=[bg, seed_color])
            distractor = [_pick_color(rng, avoid=[bg, seed_color, paint_color]) for _ in range(2)]

            program = fam_prog(int(seed_color), int(paint_color))

            train: List[Dict[str, Any]] = []
            for _ in range(int(args.train_pairs)):
                h = _rand_int(rng, 8, 14)
                w = _rand_int(rng, 8, 14)
                inp = _make_input_blob(
                    rng,
                    bg=bg,
                    seed_color=int(seed_color),
                    h=int(h),
                    w=int(w),
                    distractor_colors=distractor,
                )
                out = _run_program(inp, program=program)
                # Sanity: shape must match for these families.
                if tuple(int(x) for x in grid_shape_v124(grid_from_list_v124(inp))) != tuple(
                    int(x) for x in grid_shape_v124(grid_from_list_v124(out))
                ):
                    raise SystemExit("unexpected_shape_change")
                train.append({"input": inp, "output": out})

            test: List[Dict[str, Any]] = []
            for _ in range(int(args.test_pairs)):
                h = _rand_int(rng, 8, 14)
                w = _rand_int(rng, 8, 14)
                inp = _make_input_blob(
                    rng,
                    bg=bg,
                    seed_color=int(seed_color),
                    h=int(h),
                    w=int(w),
                    distractor_colors=distractor,
                )
                out = _run_program(inp, program=program)
                test.append({"input": inp, "output": out})

            task_id = f"{idx:05d}_synth_{fam_name}_{j:03d}.json"
            path = out_dir / task_id
            obj = {"train": train, "test": test}
            path.write_text(_stable_json(obj) + "\n", encoding="utf-8")

    meta = {
        "ok": True,
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "tasks_total": int(idx),
        "families": [n for n, _f in families],
        "tasks_per_family": int(args.tasks_per_family),
        "train_pairs": int(args.train_pairs),
        "test_pairs": int(args.test_pairs),
    }
    (out_dir / "MANIFEST.json").write_text(_stable_json(meta) + "\n", encoding="utf-8")
    print(_stable_json(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

