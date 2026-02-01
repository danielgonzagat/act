#!/usr/bin/env python3
from __future__ import annotations

"""
Grid-synth dataset v3: extract + patch transform + paste (non-ARC).

Goal:
- Force multi-step pipelines of the form:
    cc4 -> select_obj -> obj_patch -> patch_xform -> paste
- So learn-mode can generate traces that mine stable, reusable CSGs that
  collapse branching on ARC-like tasks.

Design constraints:
- Deterministic (seeded).
- No ARC priors; just grid→grid transformations.
- Canonicalized patches (cropped to object color bbox) so diff binders
  (diff_rmin/diff_cmin) remain stable.
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


Grid = List[List[int]]


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ensure_absent(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"worm_exists:{path}")


def _grid(h: int, w: int, bg: int) -> Grid:
    return [[int(bg) for _ in range(int(w))] for _ in range(int(h))]


def _paste(*, base: Grid, patch: Grid, top: int, left: int, transparent: Optional[int]) -> Grid:
    h = len(base)
    w = len(base[0]) if h else 0
    ph = len(patch)
    pw = len(patch[0]) if ph else 0
    out = [[int(base[r][c]) for c in range(w)] for r in range(h)]
    for r in range(ph):
        for c in range(pw):
            rr = int(top + r)
            cc = int(left + c)
            if rr < 0 or cc < 0 or rr >= h or cc >= w:
                continue
            v = int(patch[r][c])
            if transparent is not None and int(v) == int(transparent):
                continue
            out[rr][cc] = int(v)
    return out


def _rand_connected_patch(*, rng: random.Random, max_h: int, max_w: int, color: int, bg: int) -> Grid:
    ph = rng.randint(2, max(2, int(max_h)))
    pw = rng.randint(2, max(2, int(max_w)))
    patch = _grid(ph, pw, bg)
    rr = rng.randrange(ph)
    cc = rng.randrange(pw)
    patch[rr][cc] = int(color)
    steps = rng.randint(4, max(4, (ph * pw) // 2))
    for _ in range(steps):
        dr, dc = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        rr = max(0, min(ph - 1, rr + dr))
        cc = max(0, min(pw - 1, cc + dc))
        patch[rr][cc] = int(color)
    return patch


def _crop_to_color_bbox(*, patch: Grid, color: int) -> Grid:
    r0 = c0 = 10**9
    r1 = c1 = -1
    for r, row in enumerate(patch):
        for c, x in enumerate(row):
            if int(x) != int(color):
                continue
            r0 = min(int(r0), int(r))
            c0 = min(int(c0), int(c))
            r1 = max(int(r1), int(r))
            c1 = max(int(c1), int(c))
    if r1 < 0:
        raise ValueError("empty_color_bbox")
    out: Grid = []
    for r in range(int(r0), int(r1) + 1):
        out.append([int(patch[r][c]) for c in range(int(c0), int(c1) + 1)])
    return out


def _rotate90(p: Grid) -> Grid:
    h = len(p)
    w = len(p[0]) if h else 0
    out: Grid = []
    for c in range(w):
        out.append([int(p[h - 1 - r][c]) for r in range(h)])
    return out


def _rotate180(p: Grid) -> Grid:
    return _rotate90(_rotate90(p))


def _rotate270(p: Grid) -> Grid:
    return _rotate90(_rotate180(p))


def _reflect_h(p: Grid) -> Grid:
    return [[int(x) for x in reversed(row)] for row in p]


def _reflect_v(p: Grid) -> Grid:
    return [[int(x) for x in row] for row in reversed(p)]


def _transpose(p: Grid) -> Grid:
    h = len(p)
    w = len(p[0]) if h else 0
    out: Grid = []
    for c in range(w):
        out.append([int(p[r][c]) for r in range(h)])
    return out


_XFORMS: Tuple[str, ...] = (
    "patch_rotate90",
    "patch_rotate180",
    "patch_rotate270",
    "patch_reflect_h",
    "patch_reflect_v",
    "patch_transpose",
)


def _apply_xform(*, patch: Grid, op_id: str) -> Grid:
    op = str(op_id)
    if op == "patch_rotate90":
        return _rotate90(patch)
    if op == "patch_rotate180":
        return _rotate180(patch)
    if op == "patch_rotate270":
        return _rotate270(patch)
    if op == "patch_reflect_h":
        return _reflect_h(patch)
    if op == "patch_reflect_v":
        return _reflect_v(patch)
    if op == "patch_transpose":
        return _transpose(patch)
    raise ValueError(f"unknown_xform:{op}")


def _rects_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if int(ar1) < int(br0) or int(br1) < int(ar0):
        return False
    if int(ac1) < int(bc0) or int(bc1) < int(ac0):
        return False
    return True


@dataclass(frozen=True)
class _TaskSpec:
    bg: int
    obj_color: int
    grid_h: int
    grid_w: int
    xform_op_id: str
    target_top: int
    target_left: int


def _make_task(*, rng: random.Random, tid: str, train_n: int, test_n: int) -> Dict[str, Any]:
    bg = 0
    obj_color = rng.choice([c for c in range(1, 10) if c != bg])
    # Keep grids reasonably large so a fixed target placement works across variable patch sizes.
    grid_h = rng.randint(12, 18)
    grid_w = rng.randint(12, 18)
    xform_op_id = rng.choice(list(_XFORMS))
    target_top = rng.randint(0, 3)
    target_left = rng.randint(0, 3)

    spec = _TaskSpec(
        bg=int(bg),
        obj_color=int(obj_color),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        xform_op_id=str(xform_op_id),
        target_top=int(target_top),
        target_left=int(target_left),
    )

    def _gen_pair() -> Tuple[Grid, Grid]:
        base = _grid(spec.grid_h, spec.grid_w, spec.bg)
        # Keep patches small; the task rule is "always paste at the fixed target" (not random).
        patch0 = _rand_connected_patch(rng=rng, max_h=5, max_w=5, color=spec.obj_color, bg=spec.bg)
        patch = _crop_to_color_bbox(patch=patch0, color=spec.obj_color)
        xpatch = _apply_xform(patch=patch, op_id=spec.xform_op_id)

        ph = len(patch)
        pw = len(patch[0]) if ph else 0
        xh = len(xpatch)
        xw = len(xpatch[0]) if xh else 0

        new_top = max(0, min(int(spec.target_top), int(spec.grid_h - xh)))
        new_left = max(0, min(int(spec.target_left), int(spec.grid_w - xw)))
        new_rect = (int(new_top), int(new_left), int(new_top + xh - 1), int(new_left + xw - 1))

        # Place original object with no overlap with the fixed target region.
        for _ in range(200):
            old_top = rng.randint(0, max(0, spec.grid_h - ph))
            old_left = rng.randint(0, max(0, spec.grid_w - pw))
            old_rect = (int(old_top), int(old_left), int(old_top + ph - 1), int(old_left + pw - 1))
            if not _rects_intersect(old_rect, new_rect):
                break
        else:
            # Fallback: force far corner (still deterministic).
            old_top = max(0, spec.grid_h - ph)
            old_left = max(0, spec.grid_w - pw)

        inp = _paste(base=base, patch=patch, top=int(old_top), left=int(old_left), transparent=None)
        out = _paste(base=inp, patch=xpatch, top=int(new_top), left=int(new_left), transparent=None)
        return inp, out

    train: List[Dict[str, Any]] = []
    for _ in range(int(train_n)):
        gi, go = _gen_pair()
        train.append({"input": gi, "output": go})

    test: List[Dict[str, Any]] = []
    for _ in range(int(test_n)):
        gi, go = _gen_pair()
        test.append({"input": gi, "output": go})

    return {
        "train": train,
        "test": test,
        "meta": {
            "kind": "grid_synth_v3_patch_xform",
            "task_id": str(tid),
            "spec": {
                "bg": int(spec.bg),
                "obj_color": int(spec.obj_color),
                "grid_h": int(spec.grid_h),
                "grid_w": int(spec.grid_w),
                "xform_op_id": str(spec.xform_op_id),
                "target_top": int(spec.target_top),
                "target_left": int(spec.target_left),
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Output root dir (WORM: must not exist)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=600)
    ap.add_argument("--train_n", type=int, default=3)
    ap.add_argument("--test_n", type=int, default=1)
    args = ap.parse_args()

    out_root = Path(str(args.out_root)).resolve()
    _ensure_absent(out_root)
    out_root.mkdir(parents=True, exist_ok=False)

    rng = random.Random(int(args.seed))
    tasks = int(args.tasks)
    train_n = int(args.train_n)
    test_n = int(args.test_n)

    written: List[str] = []
    for i in range(1, tasks + 1):
        tid = f"{i:05d}_synth_patch_xform_{i-1:03d}.json"
        obj = _make_task(rng=rng, tid=tid, train_n=train_n, test_n=test_n)
        p = out_root / tid
        p.write_text(_stable_json(obj) + "\n", encoding="utf-8")
        written.append(str(tid))

    (out_root / "MANIFEST.json").write_text(
        _stable_json(
            {
                "kind": "grid_synth_manifest_v1",
                "dataset_kind": "grid_synth_v3_patch_xform",
                "seed": int(args.seed),
                "tasks": int(tasks),
                "train_n": int(train_n),
                "test_n": int(test_n),
                "xforms": list(_XFORMS),
                "tasks_files": written,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    print(_stable_json({"ok": True, "out_root": str(out_root), "tasks": int(tasks), "seed": int(args.seed)}))


if __name__ == "__main__":
    main()
