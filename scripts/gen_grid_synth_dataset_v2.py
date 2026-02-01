#!/usr/bin/env python3
from __future__ import annotations

"""
Grid-synth dataset v2: object duplication via extract+paste (non-ARC).

Goal:
- Generate a family that *requires* slot-building (cc4/select_obj/obj_patch) and then a grid-writing
  action (paste) so that CSG mining can learn loss-reducing CSGs that bundle "extract → paste".
- This directly targets the current ARC bottleneck where many useful built-in CSGs only build slots
  (patch/bbox) and are rejected by the strict CSV gate in eval.

Properties:
- Deterministic, WORM-safe output.
- Tasks are ARC-format JSON with train/test pairs.
- No task-id heuristics; only generic grid construction.
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


def _rand_patch(*, rng: random.Random, max_h: int, max_w: int, color: int, bg: int) -> Grid:
    """
    Generate a small connected-ish blob on a bg patch.
    """
    ph = rng.randint(2, max(2, int(max_h)))
    pw = rng.randint(2, max(2, int(max_w)))
    patch = _grid(ph, pw, bg)
    # Start from a seed cell then random-walk to add more cells.
    rr = rng.randrange(ph)
    cc = rng.randrange(pw)
    patch[rr][cc] = int(color)
    steps = rng.randint(3, max(3, (ph * pw) // 2))
    for _ in range(steps):
        dr, dc = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        rr = max(0, min(ph - 1, rr + dr))
        cc = max(0, min(pw - 1, cc + dc))
        patch[rr][cc] = int(color)
    return patch


@dataclass(frozen=True)
class _TaskSpec:
    bg: int
    obj_color: int
    grid_h: int
    grid_w: int
    target_top: int
    target_left: int


def _make_task(*, rng: random.Random, tid: str, train_n: int, test_n: int) -> Dict[str, Any]:
    bg = 0
    obj_color = rng.choice([c for c in range(1, 10) if c != bg])
    grid_h = rng.randint(9, 14)
    grid_w = rng.randint(9, 14)

    # Choose a target location near the top-left but not always (0,0).
    target_top = rng.randint(0, 3)
    target_left = rng.randint(0, 3)

    spec = _TaskSpec(
        bg=int(bg),
        obj_color=int(obj_color),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        target_top=int(target_top),
        target_left=int(target_left),
    )

    def _gen_pair() -> Tuple[Grid, Grid]:
        base = _grid(spec.grid_h, spec.grid_w, spec.bg)
        # Ensure the original object placement is strictly below/right of the target so that
        # diff bbox min corresponds to the target placement (helps binderization).
        patch = _rand_patch(rng=rng, max_h=5, max_w=5, color=spec.obj_color, bg=spec.bg)
        ph = len(patch)
        pw = len(patch[0]) if ph else 0
        # Old placement must not overlap target; keep it well away.
        old_top = rng.randint(spec.target_top + 2, max(spec.target_top + 2, spec.grid_h - ph))
        old_left = rng.randint(spec.target_left + 2, max(spec.target_left + 2, spec.grid_w - pw))
        old_top = min(old_top, spec.grid_h - ph)
        old_left = min(old_left, spec.grid_w - pw)

        inp = _paste(base=base, patch=patch, top=old_top, left=old_left, transparent=spec.bg)
        out = _paste(base=inp, patch=patch, top=spec.target_top, left=spec.target_left, transparent=spec.bg)
        return inp, out

    train: List[Dict[str, Any]] = []
    for _ in range(int(train_n)):
        gi, go = _gen_pair()
        train.append({"input": gi, "output": go})

    test: List[Dict[str, Any]] = []
    for _ in range(int(test_n)):
        gi, go = _gen_pair()
        test.append({"input": gi, "output": go})

    return {"train": train, "test": test, "meta": {"kind": "grid_synth_v2_dup_obj", "task_id": str(tid)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Output root dir (WORM: must not exist)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=400)
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
        tid = f"{i:05d}_synth_dup_obj_{i-1:03d}.json"
        obj = _make_task(rng=rng, tid=tid, train_n=train_n, test_n=test_n)
        p = out_root / tid
        p.write_text(_stable_json(obj) + "\n", encoding="utf-8")
        written.append(str(tid))

    # Metadata manifest (ignored by loader, but useful for humans).
    (out_root / "MANIFEST.json").write_text(
        _stable_json(
            {
                "kind": "grid_synth_manifest_v1",
                "dataset_kind": "grid_synth_v2_dup_obj",
                "seed": int(args.seed),
                "tasks": int(tasks),
                "train_n": int(train_n),
                "test_n": int(test_n),
                "tasks_files": written,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        _stable_json(
            {
                "ok": True,
                "out_root": str(out_root),
                "tasks": int(tasks),
                "seed": int(args.seed),
            }
        )
    )


if __name__ == "__main__":
    main()

