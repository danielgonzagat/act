from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_miner_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "arc_mine_operator_templates_v147.py"
    spec = importlib.util.spec_from_file_location("arc_mine_operator_templates_v147", str(path))
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_per_task(*, run_dir: Path, task_id: str, status: str) -> None:
    per_task_dir = run_dir / "per_task"
    per_task_dir.mkdir(parents=True, exist_ok=True)

    # Minimal task object for arc_task_family_id() fallback provenance.
    task_obj = {
        "task_id": str(task_id),
        "train_pairs": [
            {
                "in_grid": [[1, 1], [1, 1]],
                "out_grid": [[1, 1], [1, 1]],
            }
        ],
        "test_pairs": [{"in_grid": [[1, 1], [1, 1]], "out_grid": None}],
    }

    # Trace program: bbox_by_color -> crop_bbox (closure adds commit_patch).
    trace_programs = [
        {
            "loss": {"shape": 0, "cells": 0},
            "steps": [
                {"op_id": "bbox_by_color", "args": {"color": 1}},
                {"op_id": "crop_bbox", "args": {}},
            ],
        }
    ]

    solver_result = {
        "kind": "arc_solver_result_v141",
        "schema_version": 141,
        "status": str(status),
        "trace": {"trace_programs": list(trace_programs)},
    }
    if str(status) != "SOLVED":
        solver_result["failure_reason"] = {"kind": "MISSING_OPERATOR", "details": {}}

    obj = {
        "kind": "arc_per_task_v141",
        "schema_version": 141,
        "solver_results": [solver_result],
        "task": dict(task_obj),
    }

    out_path = per_task_dir / f"{task_id}.json.json"
    out_path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_operator_mining_supports_origin_run_dir_for_solved_tasks() -> None:
    miner = _load_miner_module()

    with TemporaryDirectory() as td:
        base = Path(td)
        origin_run_dir = base / "origin"
        deep_run_dir = base / "deep"
        out_path = base / "out.jsonl"

        # Task is a failure in the origin run...
        _write_per_task(run_dir=origin_run_dir, task_id="t.json", status="FAIL")
        # ...and solved in the deep run (should still be mined when origin_run_dir is provided).
        _write_per_task(run_dir=deep_run_dir, task_id="t.json", status="SOLVED")

        argv = [
            "arc_mine_operator_templates_v147.py",
            "--run_dir",
            str(deep_run_dir),
            "--origin_run_dir",
            str(origin_run_dir),
            "--out",
            str(out_path),
            "--include_solved_from_failure",
            "--min_len",
            "2",
            "--max_len",
            "5",
            "--min_support",
            "1",
            "--max_operators",
            "32",
            "--trace_max_programs_per_task",
            "20",
            "--trace_max_loss_shape",
            "0",
            "--trace_max_loss_cells",
            "120",
        ]

        old_argv = sys.argv
        try:
            sys.argv = argv
            miner.main()
        finally:
            sys.argv = old_argv

        rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows
        op_ids = rows[0].get("op_ids")
        assert op_ids == ["bbox_by_color", "crop_bbox", "commit_patch"]

