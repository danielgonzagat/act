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
from atos_core.goal_replan_persistence_law_v121 import (
    FAIL_REASON_EXHAUSTION_WITHOUT_PROOF_V121,
    FAIL_REASON_GOAL_DIED_ON_FAIL_V121,
    OK_REASON_GOAL_EXHAUSTED_V121,
    verify_goal_replan_persistence_law_v121,
)


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _ensure_absent(path: Path) -> None:
    if path.exists():
        _fail(f"worm_exists:{path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        _fail(f"tmp_exists:{tmp}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_once_json(path: Path, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_absent(path)
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _mk_case_dir(*, out_dir: Path, case_name: str) -> Path:
    cd = out_dir / case_name
    _ensure_absent(cd)
    cd.mkdir(parents=True, exist_ok=False)
    return cd


def _write_case_replan_ok(case_dir: Path) -> None:
    # One user turn, and an action plan that shows a failed attempt then a successful replanned attempt.
    _write_jsonl(case_dir / "conversation_turns.jsonl", [{"payload": {"role": "user", "turn_id": "turn_u0", "turn_index": 0, "created_step": 0}}])
    _write_jsonl(
        case_dir / "action_plans.jsonl",
        [
            {
                "user_turn_id": "turn_u0",
                "objective_kind": "COMM_RESPOND",
                "chosen_ok": True,
                "ranked_candidates": [{"act_id": "A"}, {"act_id": "B"}],
                "attempted_actions": [
                    {"act_id": "A", "eval_id": "e0", "ok": False},
                    {"act_id": "B", "eval_id": "e1", "ok": True},
                ],
            }
        ],
    )
    _write_jsonl(
        case_dir / "objective_evals.jsonl",
        [
            {"eval_id": "e0", "verdict": {"ok": False, "reason": "fail_A"}},
            {"eval_id": "e1", "verdict": {"ok": True, "reason": ""}},
        ],
    )


def _write_case_exhausted_with_proof(case_dir: Path) -> None:
    _write_jsonl(case_dir / "conversation_turns.jsonl", [{"payload": {"role": "user", "turn_id": "turn_u0", "turn_index": 0, "created_step": 0}}])
    _write_jsonl(
        case_dir / "action_plans.jsonl",
        [
            {
                "user_turn_id": "turn_u0",
                "objective_kind": "COMM_RESPOND",
                "chosen_ok": False,
                "ranked_candidates": [{"act_id": "A"}, {"act_id": "B"}],
                "attempted_actions": [
                    {"act_id": "A", "eval_id": "e0", "ok": False},
                    {"act_id": "B", "eval_id": "e1", "ok": False},
                ],
            }
        ],
    )
    _write_jsonl(
        case_dir / "objective_evals.jsonl",
        [
            {"eval_id": "e0", "verdict": {"ok": False, "reason": "fail_A"}},
            {"eval_id": "e1", "verdict": {"ok": False, "reason": "fail_B"}},
        ],
    )


def _tamper_exhaustion_proof(*, src_case_dir: Path, dst_dir: Path) -> Path:
    _ensure_absent(dst_dir)
    shutil.copytree(str(src_case_dir), str(dst_dir), dirs_exist_ok=False)
    # Remove one eval row to break the proof deterministically.
    evals_path = dst_dir / "objective_evals.jsonl"
    rows = [json.loads(x) for x in evals_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows2 = [r for r in rows if isinstance(r, dict) and str(r.get("eval_id") or "") != "e1"]
    evals_path.write_text("\n".join([json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows2]) + "\n", encoding="utf-8")
    return dst_dir


def _smoke_once(*, out_dir: Path) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    case00 = _mk_case_dir(out_dir=out_dir, case_name="case_00_replan_required")
    _write_case_replan_ok(case00)
    r00 = verify_goal_replan_persistence_law_v121(run_dir=str(case00), max_replans_per_turn=3)
    if not r00.ok:
        _fail(f"case00_unexpected_fail:{r00.reason}")

    case01 = _mk_case_dir(out_dir=out_dir, case_name="case_01_exhausted_with_proof")
    _write_case_exhausted_with_proof(case01)
    r01 = verify_goal_replan_persistence_law_v121(run_dir=str(case01), max_replans_per_turn=3)
    if not r01.ok or str(r01.reason) != OK_REASON_GOAL_EXHAUSTED_V121:
        _fail(f"case01_expected_goal_exhausted_but:{r01.ok}:{r01.reason}")

    eval_obj = {
        "schema_version": 121,
        "kind": "smoke_v121_goal_replan_until_exhausted_eval",
        "case00": {"ok": bool(r00.ok), "reason": str(r00.reason)},
        "case01": {"ok": bool(r01.ok), "reason": str(r01.reason)},
    }
    eval_obj["eval_sig"] = sha256_hex(canonical_json_dumps(eval_obj).encode("utf-8"))
    _write_once_json(out_dir / "eval.json", eval_obj)
    core = {"schema_version": 121, "eval_sha256": _sha256_file(out_dir / "eval.json"), "eval_sig": str(eval_obj["eval_sig"])}
    summary_sha256 = sha256_hex(canonical_json_dumps(core).encode("utf-8"))
    _write_once_json(out_dir / "smoke_summary.json", {"schema_version": 121, "kind": "smoke_v121_goal_replan_until_exhausted_summary", "core": core, "summary_sha256": str(summary_sha256)})
    return {"core": dict(core), "summary_sha256": str(summary_sha256), "out_dir": str(out_dir)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()

    out_base = str(args.out_base)
    # seed currently unused (smoke is deterministic without RNG), but keep for uniform CLI and future extensions.
    _seed = int(args.seed)

    r1 = _smoke_once(out_dir=Path(out_base + "_try1"))
    r2 = _smoke_once(out_dir=Path(out_base + "_try2"))

    if canonical_json_dumps(r1["core"]) != canonical_json_dumps(r2["core"]):
        _fail("determinism_core_mismatch")
    if str(r1["summary_sha256"]) != str(r2["summary_sha256"]):
        _fail("determinism_summary_sha256_mismatch")

    # Negative tamper: corrupt exhaustion proof and require stable failure reason.
    tamper_dir = Path(out_base + "_try1_tamper")
    src_case = Path(out_base + "_try1") / "case_01_exhausted_with_proof"
    tampered_case = _tamper_exhaustion_proof(src_case_dir=src_case, dst_dir=tamper_dir)
    res = verify_goal_replan_persistence_law_v121(run_dir=str(tampered_case), max_replans_per_turn=3)
    if bool(res.ok):
        _fail("tamper_expected_fail_but_ok")
    if str(res.reason) != FAIL_REASON_EXHAUSTION_WITHOUT_PROOF_V121:
        _fail(f"tamper_reason_mismatch:{res.reason}")

    print(
        json.dumps(
            {
                "ok": True,
                "determinism_ok": True,
                "summary_sha256": str(r1["summary_sha256"]),
                "out_try1": str(out_base + "_try1"),
                "out_try2": str(out_base + "_try2"),
                "tamper_reason": str(res.reason),
                "sha256_eval_json_try1": _sha256_file(Path(out_base + "_try1") / "eval.json"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

