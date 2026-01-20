#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.conversation_loop_v115 import run_conversation_v115
from atos_core.conversation_loop_v116 import apply_dialogue_survival_as_law_v116, run_conversation_v116
from atos_core.dialogue_survival_gate_v116 import (
    DIALOGUE_SURVIVAL_REASON_FLUENCY_FAIL_V116,
    DIALOGUE_SURVIVAL_REASON_UNRESOLVED_REFERENCE_V116,
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
    out = run_conversation_v116(user_turn_texts=["set x to 4", "get x", "end now"], out_dir=str(run_dir), seed=int(seed))
    fr = _load_json(run_dir / "final_response_v116.json")
    if not bool(fr.get("ok", False)):
        raise SystemExit("case_positive_final_response_not_ok")
    if not bool(out.get("dialogue_survival_v116_ok", False)):
        raise SystemExit("case_positive_dialogue_survival_not_ok")
    return {"ok": True}


def _tamper_transcript_to_force_fluency_fail(path: Path) -> None:
    rows = _load_jsonl(path)
    if not rows:
        raise SystemExit("tamper_empty_transcript")
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        payload = r.get("payload")
        if isinstance(payload, dict) and str(payload.get("role") or "") == "assistant":
            p2 = dict(payload)
            p2["text"] = "OK"
            out_rows.append({"payload": p2})
        else:
            out_rows.append(dict(r))
    os.replace(str(path), str(path) + ".bak")
    _write_jsonl(path, out_rows)


def _tamper_flow_to_force_unresolved_final(path: Path) -> None:
    rows = _load_jsonl(path)
    if not rows:
        raise SystemExit("tamper_empty_flow")
    last = dict(rows[-1])
    flags = last.get("flow_flags_v108")
    if not isinstance(flags, dict):
        flags = {}
    flags2 = dict(flags)
    flags2["UNRESOLVED_REFERENCE"] = True
    last["flow_flags_v108"] = flags2
    rows[-1] = last
    os.replace(str(path), str(path) + ".bak")
    _write_jsonl(path, [dict(r) for r in rows if isinstance(r, dict)])


def _case_negative_fluency(*, base_dir: Path, seed: int) -> Dict[str, Any]:
    run_dir = base_dir / "case_01_neg_fluency"
    _ensure_absent(run_dir)
    run_conversation_v115(user_turn_texts=["set x to 4", "get x", "end now"], out_dir=str(run_dir), seed=int(seed))
    _tamper_transcript_to_force_fluency_fail(run_dir / "transcript.jsonl")
    applied = apply_dialogue_survival_as_law_v116(run_dir=str(run_dir), write_mind_graph=True)
    fr = _load_json(run_dir / "final_response_v116.json")
    if bool(fr.get("ok", True)):
        raise SystemExit("case_neg_fluency_unexpected_ok")
    if str(fr.get("reason") or "") != DIALOGUE_SURVIVAL_REASON_FLUENCY_FAIL_V116:
        raise SystemExit("case_neg_fluency_wrong_reason")
    if not (run_dir / "mind_graph_v116" / "mind_nodes.jsonl").exists():
        raise SystemExit("case_neg_fluency_missing_mind_graph_v116")
    nodes = _load_jsonl(run_dir / "mind_graph_v116" / "mind_nodes.jsonl")
    fail_nodes = [
        n
        for n in nodes
        if isinstance(n, dict)
        and isinstance((n.get("payload") or {}).get("ato"), dict)
        and isinstance((((n.get("payload") or {}).get("ato") or {}).get("invariants")), dict)
        and str((((n.get("payload") or {}).get("ato") or {}).get("invariants") or {}).get("eval_kind") or "")
        == "FAIL_EVENT_V116"
    ]
    if len(fail_nodes) < 1:
        raise SystemExit("case_neg_fluency_missing_fail_event_node")
    return {"ok": True, "reason": str(applied.reason)}


def _case_negative_unresolved(*, base_dir: Path, seed: int) -> Dict[str, Any]:
    run_dir = base_dir / "case_02_neg_unresolved"
    _ensure_absent(run_dir)
    run_conversation_v115(user_turn_texts=["set x to 4", "get x", "end now"], out_dir=str(run_dir), seed=int(seed))
    _tamper_flow_to_force_unresolved_final(run_dir / "flow_events.jsonl")
    applied = apply_dialogue_survival_as_law_v116(run_dir=str(run_dir), write_mind_graph=True)
    fr = _load_json(run_dir / "final_response_v116.json")
    if bool(fr.get("ok", True)):
        raise SystemExit("case_neg_unresolved_unexpected_ok")
    if str(fr.get("reason") or "") != DIALOGUE_SURVIVAL_REASON_UNRESOLVED_REFERENCE_V116:
        raise SystemExit("case_neg_unresolved_wrong_reason")
    return {"ok": True, "reason": str(applied.reason)}


def _run_try(*, out_dir: Path, seed: int) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    cases = {
        "positive": _case_positive(base_dir=out_dir, seed=seed),
        "neg_fluency": _case_negative_fluency(base_dir=out_dir, seed=seed),
        "neg_unresolved": _case_negative_unresolved(base_dir=out_dir, seed=seed),
    }
    eval_obj = {"schema_version": 116, "seed": int(seed), "cases": dict(cases)}
    _write_once_json(out_dir / "eval.json", eval_obj)
    eval_sha256 = _sha256_file(out_dir / "eval.json")
    summary = {"schema_version": 116, "seed": int(seed), "eval_sha256": str(eval_sha256)}
    _write_once_json(out_dir / "summary.json", summary)
    fail_catalog = {"schema_version": 116, "failures_total": 0, "failures": []}
    _write_once_json(out_dir / "fail_catalog_v116.json", fail_catalog)
    return {"eval_sha256": str(eval_sha256), "eval_json": eval_obj}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", default="results/run_smoke_v116_dialogue_survival_render_gate")
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

    core = {"schema_version": 116, "seed": int(seed), "eval_sha256": str(r1["eval_sha256"])}
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

