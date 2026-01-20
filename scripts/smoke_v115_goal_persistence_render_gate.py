#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.conversation_loop_v115 import run_conversation_v115
from atos_core.goal_persistence_v115 import render_fail_response_v115
from atos_core.goal_plan_eval_gate_v115 import (
    FAIL_REASON_GOAL_PLACEBO_V115,
    FAIL_REASON_RENDER_BLOCKED_NO_EVAL_SATISFIES_V115,
    verify_goal_plan_eval_law_v115,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ensure_absent(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"worm_exists:{path}")


def _write_once_json(path: Path, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise SystemExit(f"tmp_exists:{tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_absent(path)
    with open(path, "x", encoding="utf-8") as f:
        for r in rows:
            f.write(canonical_json_dumps(r))
            f.write("\n")


def _case_positive(*, base_dir: Path, seed: int) -> Dict[str, Any]:
    run_dir = base_dir / "case_00_positive"
    _ensure_absent(run_dir)
    out = run_conversation_v115(user_turn_texts=["set x to 4", "get x", "end now"], out_dir=str(run_dir), seed=int(seed))
    if not bool(out.get("gate_v115_ok", False)):
        raise SystemExit("case_positive_gate_failed")
    fr = _load_json(run_dir / "final_response_v115.json")
    if not bool(fr.get("ok", False)):
        raise SystemExit("case_positive_final_response_not_ok")
    return {"ok": True}


def _case_negative_missing_eval_satisfies(*, base_dir: Path, seed: int) -> Dict[str, Any]:
    src_dir = base_dir / "case_01_missing_eval_base"
    _ensure_absent(src_dir)
    run_conversation_v115(user_turn_texts=["set x to 4"], out_dir=str(src_dir), seed=int(seed))

    tamper_dir = base_dir / "case_01_missing_eval_tamper"
    _ensure_absent(tamper_dir)
    tamper_dir.mkdir(parents=True, exist_ok=False)
    for fn in ("conversation_turns.jsonl", "action_plans.jsonl", "objective_evals.jsonl"):
        shutil.copyfile(str(src_dir / fn), str(tamper_dir / fn))

    evals = _load_jsonl(tamper_dir / "objective_evals.jsonl")
    if not evals:
        raise SystemExit("case_neg1_empty_objective_evals")
    ev0 = dict(evals[0])
    verdict = ev0.get("verdict")
    if not isinstance(verdict, dict):
        verdict = {}
    verdict2 = dict(verdict)
    verdict2["ok"] = False
    verdict2["score"] = 0
    verdict2["reason"] = "tamper_force_fail_v115"
    ev0["verdict"] = verdict2
    evals[0] = ev0
    os.replace(str(tamper_dir / "objective_evals.jsonl"), str(tamper_dir / "objective_evals.jsonl.bak"))
    _write_jsonl(tamper_dir / "objective_evals.jsonl", evals)

    gate = verify_goal_plan_eval_law_v115(run_dir=str(tamper_dir), max_replans_per_turn=3, write_mind_graph=True, write_snapshots=True)
    if gate.ok:
        raise SystemExit("case_neg1_gate_unexpected_ok")
    if gate.reason != FAIL_REASON_RENDER_BLOCKED_NO_EVAL_SATISFIES_V115:
        raise SystemExit("case_neg1_wrong_reason:{r}".format(r=gate.reason))
    fail_text = render_fail_response_v115(str(gate.reason))
    if not fail_text.startswith("FAIL:"):
        raise SystemExit("case_neg1_fail_response_bad")
    return {"ok": True, "gate_reason": str(gate.reason)}


def _case_negative_goal_placebo(*, base_dir: Path, seed: int) -> Dict[str, Any]:
    run_dir = base_dir / "case_02_goal_placebo"
    _ensure_absent(run_dir)
    out = run_conversation_v115(user_turn_texts=[""], out_dir=str(run_dir), seed=int(seed))
    if bool(out.get("gate_v115_ok", False)):
        raise SystemExit("case_neg2_gate_unexpected_ok")
    if str(out.get("gate_v115_reason") or "") != FAIL_REASON_GOAL_PLACEBO_V115:
        raise SystemExit("case_neg2_wrong_reason:{r}".format(r=out.get("gate_v115_reason")))
    fr = _load_json(run_dir / "final_response_v115.json")
    if bool(fr.get("ok", True)):
        raise SystemExit("case_neg2_final_response_should_fail")
    if str(fr.get("fail_response_text") or "") != render_fail_response_v115(FAIL_REASON_GOAL_PLACEBO_V115):
        raise SystemExit("case_neg2_final_fail_text_mismatch")
    return {"ok": True, "gate_reason": str(out.get("gate_v115_reason") or "")}


def _run_try(*, out_dir: Path, seed: int) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    cases = {
        "positive": _case_positive(base_dir=out_dir, seed=seed),
        "neg_missing_eval_satisfies": _case_negative_missing_eval_satisfies(base_dir=out_dir, seed=seed),
        "neg_goal_placebo": _case_negative_goal_placebo(base_dir=out_dir, seed=seed),
    }
    eval_obj = {"schema_version": 115, "seed": int(seed), "cases": dict(cases)}
    _write_once_json(out_dir / "eval.json", eval_obj)
    eval_sha256 = _sha256_file(out_dir / "eval.json")
    summary = {"schema_version": 115, "seed": int(seed), "eval_sha256": str(eval_sha256)}
    _write_once_json(out_dir / "summary.json", summary)
    fail_catalog = {"schema_version": 115, "failures_total": 0, "failures": []}
    _write_once_json(out_dir / "fail_catalog_v115.json", fail_catalog)
    return {"eval_sha256": str(eval_sha256), "eval_json": eval_obj}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", default="results/run_smoke_v115_goal_persistence_render_gate")
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()

    seed = int(args.seed)
    out_base = Path(str(args.out_base))
    out1 = Path(str(out_base) + "_try1")
    out2 = Path(str(out_base) + "_try2")

    r1 = _run_try(out_dir=out1, seed=seed)
    r2 = _run_try(out_dir=out2, seed=seed)

    if canonical_json_dumps(r1["eval_json"]) != canonical_json_dumps(r2["eval_json"]):
        raise SystemExit("determinism_failed:eval_json")
    if r1["eval_sha256"] != r2["eval_sha256"]:
        raise SystemExit("determinism_failed:eval_sha256")

    core = {"schema_version": 115, "seed": int(seed), "eval_sha256": str(r1["eval_sha256"])}
    summary_sha256 = sha256_hex(canonical_json_dumps(core).encode("utf-8"))
    print(
        json.dumps(
            {
                "ok": True,
                "determinism_ok": True,
                "summary_sha256": str(summary_sha256),
                "try1_dir": str(out1),
                "try2_dir": str(out2),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

