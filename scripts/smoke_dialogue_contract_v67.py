#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps
from atos_core.engine import Engine, EngineConfig
from atos_core.store import ActStore
from atos_core.suite import UTILITY_DIALOGUES_V67, run_skill_suite, suite_metrics_from_transcripts


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


def transcripts_text(transcripts: List[Dict[str, Any]]) -> str:
    return "\n".join(str(r.get("full_text", "")) for r in transcripts)


def eval_v67_contracts(*, run_dir: str, seed: int, max_new_tokens: int) -> Dict[str, Any]:
    acts_path = os.path.join(str(run_dir), "acts.jsonl")
    store = ActStore.load_jsonl(acts_path)
    engine = Engine(
        store,
        seed=int(seed),
        config=EngineConfig(enable_contracts=True),
    )
    transcripts, metrics = run_skill_suite(
        engine, tasks=UTILITY_DIALOGUES_V67, max_new_tokens=int(max_new_tokens)
    )
    cost_metrics = suite_metrics_from_transcripts(transcripts)
    txt = transcripts_text(transcripts)
    return {
        "run": str(run_dir),
        "seed": int(seed),
        "max_new_tokens": int(max_new_tokens),
        "utility_suite_version": "v67",
        "sha256_transcript_text": sha256_text(txt),
        "cost": {k: cost_metrics.get(k) for k in sorted(cost_metrics.keys())},
        **dict(metrics),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        default="results/run_pure_20k_v39_contracts_paraphrase_telemetry_smoke",
        help="Run dir containing acts.jsonl (read-only)",
    )
    ap.add_argument(
        "--out_base",
        default="results/run_smoke_dialogue_contract_v67",
        help="Base output path; script writes <out_base>_try1 and <out_base>_try2",
    )
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

        ev = eval_v67_contracts(run_dir=run_dir, seed=seed, max_new_tokens=max_new_tokens)
        pass_rate = float(ev.get("pass_rate", 0.0) or 0.0)
        contract_used_rate = float(ev.get("contract_used_rate", 0.0) or 0.0)
        contract_used_by_kind = ev.get("contract_used_by_kind") or {}
        if not isinstance(contract_used_by_kind, dict):
            contract_used_by_kind = {}

        if abs(pass_rate - 1.0) > 1e-12:
            _fail(f"ERROR: pass_rate != 1.0 (got {pass_rate})")
        if abs(contract_used_rate - 1.0) > 1e-12:
            _fail(f"ERROR: contract_used_rate != 1.0 (got {contract_used_rate})")
        if int(contract_used_by_kind.get("plan_contract_v67", 0) or 0) <= 0:
            _fail("ERROR: expected contract_used_by_kind.plan_contract_v67 > 0")

        eval_path = os.path.join(out_dir, "eval.json")
        eval_sha = write_json(eval_path, ev)

        core = {
            "seed": seed,
            "run": run_dir,
            "utility_suite_version": "v67",
            "pass_rate": float(pass_rate),
            "contract_used_rate": float(contract_used_rate),
            "contract_used_by_kind": dict(sorted((str(k), int(v)) for k, v in contract_used_by_kind.items())),
            "sha256_transcript_text": str(ev.get("sha256_transcript_text") or ""),
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

