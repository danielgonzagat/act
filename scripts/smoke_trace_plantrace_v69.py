#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps
from atos_core.engine import Engine, EngineConfig
from atos_core.store import ActStore
from atos_core.suite import UTILITY_DIALOGUES_V68, run_skill_suite


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()


def sha256_canon(obj: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(obj).encode("utf-8")).hexdigest()


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"ERROR: path already exists: {path}")


def write_json(path: str, obj: Any) -> str:
    ensure_absent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)
    return sha256_file(path)


def _assert_contract_snapshots_match(trace: Dict[str, Any]) -> Tuple[bool, str]:
    ic = trace.get("instruction_contract")
    pt = trace.get("plan_trace")
    if not isinstance(ic, dict):
        return False, "instruction_contract_missing"
    if not isinstance(pt, dict):
        return False, "plan_trace_missing"
    pic = pt.get("trace_instruction_contract")
    if not isinstance(pic, dict):
        return False, "plan_trace.trace_instruction_contract_missing"
    if sha256_canon(ic) != sha256_canon(pic):
        return False, "instruction_contract_snapshot_mismatch"
    return True, ""


def _aliasing_test(*, engine: Engine) -> Dict[str, Any]:
    pt = {
        "compiler_id": "alias_test_compiler",
        "validator_id": "plan_validator",
        "expected_format": "plan",
        "constraints": ["plan_validator", "json_canonical"],
        "expected_spec_sig": "alias_test_sig",
        "goal_id": "alias_goal",
    }
    prompt = "User: hello\nSystem:\n"
    out = engine.generate(prompt=prompt, max_new_tokens=8, mode="greedy", dialogue_id=999, turn=0, plan_trace=pt)
    tr = out.get("trace") if isinstance(out, dict) else {}
    tr = tr if isinstance(tr, dict) else {}

    snap0 = tr.get("plan_trace")
    snap0 = snap0 if isinstance(snap0, dict) else {}
    before = str(snap0.get("compiler_id") or "")
    if before != "alias_test_compiler":
        return {"ok": False, "reason": "plan_trace_snapshot_wrong_before_mutation", "before": before}

    pt["compiler_id"] = "MUTATED"
    after = str((tr.get("plan_trace") or {}).get("compiler_id") or "")
    if after != "alias_test_compiler":
        return {"ok": False, "reason": "plan_trace_snapshot_mutated_via_aliasing", "after": after}

    ic = tr.get("instruction_contract")
    pic = (tr.get("plan_trace") or {}).get("trace_instruction_contract")
    if isinstance(ic, dict) and isinstance(pic, dict):
        orig_kind = str(pic.get("kind") or "")
        ic["kind"] = "MUTATED_KIND"
        after_kind = str(((tr.get("plan_trace") or {}).get("trace_instruction_contract") or {}).get("kind") or "")
        if after_kind != orig_kind:
            return {"ok": False, "reason": "contract_snapshot_mutated_via_aliasing"}

    return {"ok": True}


def eval_v69_trace(*, run_dir: str, seed: int, max_new_tokens: int) -> Dict[str, Any]:
    acts_path = os.path.join(str(run_dir), "acts.jsonl")
    store = ActStore.load_jsonl(acts_path)
    engine = Engine(store, seed=int(seed), config=EngineConfig(enable_contracts=True))

    alias_res = _aliasing_test(engine=engine)
    if not bool(alias_res.get("ok", False)):
        _fail(f"ERROR: aliasing_test_failed: {alias_res}")

    transcripts, metrics = run_skill_suite(engine, tasks=UTILITY_DIALOGUES_V68, max_new_tokens=int(max_new_tokens))
    pass_rate = float(metrics.get("pass_rate", 0.0) or 0.0)
    contract_used_rate = float(metrics.get("contract_used_rate", 0.0) or 0.0)
    contract_used_by_kind = metrics.get("contract_used_by_kind") or {}
    if not isinstance(contract_used_by_kind, dict):
        contract_used_by_kind = {}

    if abs(pass_rate - 1.0) > 1e-12:
        _fail(f"ERROR: pass_rate != 1.0 (got {pass_rate})")
    if abs(contract_used_rate - 1.0) > 1e-12:
        _fail(f"ERROR: contract_used_rate != 1.0 (got {contract_used_rate})")
    if int(contract_used_by_kind.get("plan_contract_v67", 0) or 0) <= 0:
        _fail("ERROR: expected contract_used_by_kind.plan_contract_v67 > 0")

    turns_total = 0
    plan_trace_present = 0
    plan_trace_sig_ok = 0
    contract_inside_plan_trace = 0
    contracts_match = 0
    contract_kind_ok = 0
    sample_plan_trace_sig = ""

    for rec in transcripts:
        turns = rec.get("turns") if isinstance(rec, dict) else None
        if not isinstance(turns, list):
            continue
        for t in turns:
            if not isinstance(t, dict):
                continue
            turns_total += 1
            tr = t.get("trace") if isinstance(t.get("trace"), dict) else {}

            pt = tr.get("plan_trace")
            if isinstance(pt, dict):
                plan_trace_present += 1
                if not sample_plan_trace_sig:
                    sample_plan_trace_sig = str(tr.get("plan_trace_sig") or "")
                want_sig = str(tr.get("plan_trace_sig") or "")
                got_sig = sha256_canon(pt)
                if want_sig and want_sig == got_sig:
                    plan_trace_sig_ok += 1
                pic = pt.get("trace_instruction_contract")
                if isinstance(pic, dict):
                    contract_inside_plan_trace += 1
                    if bool(pic.get("used")) and str(pic.get("kind") or "") == "plan_contract_v67":
                        contract_kind_ok += 1

            ok, _ = _assert_contract_snapshots_match(tr)
            if ok:
                contracts_match += 1

    if turns_total <= 0:
        _fail("ERROR: no_turns")
    if plan_trace_present != turns_total:
        _fail(f"ERROR: plan_trace_missing_turns: have={plan_trace_present} total={turns_total}")
    if plan_trace_sig_ok != turns_total:
        _fail(f"ERROR: plan_trace_sig_mismatch: ok={plan_trace_sig_ok} total={turns_total}")
    if contract_inside_plan_trace != turns_total:
        _fail(
            f"ERROR: missing trace_instruction_contract in plan_trace: have={contract_inside_plan_trace} total={turns_total}"
        )
    if contracts_match != turns_total:
        _fail(f"ERROR: contract_snapshot_mismatch: ok={contracts_match} total={turns_total}")
    if contract_kind_ok != turns_total:
        _fail(f"ERROR: contract_kind_not_ok: ok={contract_kind_ok} total={turns_total}")

    return {
        "run": str(run_dir),
        "seed": int(seed),
        "max_new_tokens": int(max_new_tokens),
        "utility_suite_version": "v68",
        "aliasing_test": dict(alias_res),
        "turns_total": int(turns_total),
        "plan_trace_present_turns": int(plan_trace_present),
        "plan_trace_sig_ok_turns": int(plan_trace_sig_ok),
        "contract_inside_plan_trace_turns": int(contract_inside_plan_trace),
        "contract_snapshot_match_turns": int(contracts_match),
        "contract_kind_ok_turns": int(contract_kind_ok),
        "plan_trace_sig_sample": str(sample_plan_trace_sig),
        "metrics": dict(metrics),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        default="results/run_pure_20k_v39_contracts_paraphrase_telemetry_smoke",
        help="Run dir containing acts.jsonl (read-only)",
    )
    ap.add_argument("--out_base", default="results/run_smoke_trace_plantrace_v69")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    out_base = str(args.out_base)
    run_dir = str(args.run)
    seed = int(args.seed)
    max_new_tokens = int(args.max_new_tokens)

    results: Dict[str, Any] = {"run": run_dir, "seed": seed, "max_new_tokens": max_new_tokens, "tries": {}}
    summary_shas: List[str] = []

    for t in (1, 2):
        out_dir = f"{out_base}_try{t}"
        ensure_absent(out_dir)
        os.makedirs(out_dir, exist_ok=False)

        ev = eval_v69_trace(run_dir=run_dir, seed=seed, max_new_tokens=max_new_tokens)
        eval_path = os.path.join(out_dir, "eval.json")
        eval_sha = write_json(eval_path, ev)

        m = ev.get("metrics") if isinstance(ev.get("metrics"), dict) else {}
        by_kind = (m.get("contract_used_by_kind") or {}) if isinstance(m, dict) else {}
        if not isinstance(by_kind, dict):
            by_kind = {}

        core = {
            "seed": seed,
            "run": run_dir,
            "pass_rate": float((m.get("pass_rate") if isinstance(m, dict) else 0.0) or 0.0),
            "contract_used_rate": float((m.get("contract_used_rate") if isinstance(m, dict) else 0.0) or 0.0),
            "contract_used_by_kind": dict(sorted((str(k), int(v)) for k, v in by_kind.items())),
            "turns_total": int(ev.get("turns_total", 0) or 0),
            "plan_trace_sig_sample": str(ev.get("plan_trace_sig_sample") or ""),
            "sha256_eval_json": str(eval_sha),
        }
        summary_sha = sha256_text(canonical_json_dumps(core))
        smoke = {"summary": core, "determinism": {"summary_sha256": summary_sha}}
        smoke_path = os.path.join(out_dir, "smoke_summary.json")
        smoke_sha = write_json(smoke_path, smoke)
        summary_shas.append(str(summary_sha))

        results["tries"][f"try{t}"] = {
            "out_dir": out_dir,
            "eval_json": {"path": eval_path, "sha256": eval_sha},
            "smoke_summary_json": {"path": smoke_path, "sha256": smoke_sha},
            "summary_sha256": summary_sha,
        }

    determinism_ok = bool(len(summary_shas) == 2 and summary_shas[0] == summary_shas[1])
    if not determinism_ok:
        _fail(f"ERROR: determinism mismatch: {summary_shas}")
    results["determinism"] = {"ok": True, "summary_sha256": summary_shas[0]}
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
