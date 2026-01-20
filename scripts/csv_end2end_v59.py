#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction, Patch, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.concepts import PRIMITIVE_OPS
from atos_core.engine import Engine, EngineConfig
from atos_core.ethics import EthicsVerdict, validate_act_for_promotion
from atos_core.ledger import Ledger
from atos_core.store import ActStore


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def ensure_out_dir_absent(out_dir: str) -> None:
    if os.path.exists(out_dir):
        _fail(f"ERROR: --out already exists: {out_dir}")


def stable_act_id(prefix: str, body: Dict[str, Any]) -> str:
    h = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return f"{prefix}{h[:12]}"


def value_to_text(v: Any) -> str:
    if isinstance(v, (dict, list, tuple)):
        return canonical_json_dumps(v)
    if v is None:
        return ""
    return str(v)


def make_concept_act(
    *,
    store_hash_excluding_semantic: str,
    step: int,
    title: str,
    interface: Dict[str, Any],
    program: List[Instruction],
    meta: Optional[Dict[str, Any]] = None,
) -> Act:
    ev = {
        "name": "concept_csv_v0",
        "interface": dict(interface),
        "meta": {
            "title": str(title),
            "trained_on_store_content_hash": str(store_hash_excluding_semantic),
            **(dict(meta or {})),
        },
    }
    body = {
        "kind": "concept_csv",
        "version": 1,
        "match": {},
        "program": [ins.to_dict() for ins in program],
        "evidence": ev,
        "deps": [],
        "active": True,
    }
    act_id = stable_act_id("act_concept_csv_", body)
    return Act(
        id=act_id,
        version=1,
        created_at=deterministic_iso(step=int(step)),
        kind="concept_csv",
        match={},
        program=program,
        evidence=ev,
        cost={"overhead_bits": 1024},
        deps=[],
        active=True,
    )


def make_goal_act(
    *,
    store_hash_excluding_semantic: str,
    step: int,
    title: str,
    priority: int,
    selector: Dict[str, Any],
    inputs: Dict[str, Any],
    expected: Any,
) -> Act:
    ev = {
        "name": "goal_v0",
        "meta": {
            "title": str(title),
            "trained_on_store_content_hash": str(store_hash_excluding_semantic),
        },
        "goal": {
            "priority": int(priority),
            "selector": dict(selector),
            "inputs": dict(inputs),
            "expected": expected,
        },
    }
    body = {
        "kind": "goal",
        "version": 1,
        "match": {},
        "program": [],
        "evidence": ev,
        "deps": [],
        "active": True,
    }
    act_id = stable_act_id("act_goal_", body)
    return Act(
        id=act_id,
        version=1,
        created_at=deterministic_iso(step=int(step)),
        kind="goal",
        match={},
        program=[],
        evidence=ev,
        cost={"overhead_bits": 1024},
        deps=[],
        active=True,
    )


def iface_sig(iface: Dict[str, Any]) -> str:
    body = {
        "in": iface.get("input_schema", {}),
        "out": iface.get("output_schema", {}),
        "validator_id": iface.get("validator_id", ""),
    }
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def run_inline_task(task: str, inputs: Dict[str, Any]) -> Any:
    if task == "extract_int":
        s = str(inputs["text"])
        _, fn_scan = PRIMITIVE_OPS["scan_digits"]
        _, fn_d2i = PRIMITIVE_OPS["digits_to_int"]
        digits = fn_scan(s)
        return fn_d2i(digits)
    if task == "sum_from_texts":
        _, fn_scan = PRIMITIVE_OPS["scan_digits"]
        _, fn_d2i = PRIMITIVE_OPS["digits_to_int"]
        _, fn_add = PRIMITIVE_OPS["add_int"]
        a = fn_d2i(fn_scan(str(inputs["text_a"])))
        b = fn_d2i(fn_scan(str(inputs["text_b"])))
        return fn_add(a, b)
    if task == "json_ab":
        _, fn_mk = PRIMITIVE_OPS["make_dict_ab"]
        _, fn_j = PRIMITIVE_OPS["json_canonical"]
        obj = fn_mk(int(inputs["a"]), int(inputs["b"]))
        return fn_j(obj)
    raise KeyError(f"unknown_task:{task}")


def write_promoted_acts_preserve_order(
    *, base_acts_path: str, out_acts_path: str, appended_acts: List[Act]
) -> str:
    with open(base_acts_path, "rb") as f:
        base_bytes = f.read()
    if base_bytes and not base_bytes.endswith(b"\n"):
        base_bytes += b"\n"
    lines = base_bytes + b"".join(
        (canonical_json_dumps(a.to_dict()).encode("utf-8") + b"\n") for a in appended_acts
    )
    tmp = out_acts_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(lines)
    os.replace(tmp, out_acts_path)
    return sha256_file(out_acts_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_run", required=True, help="Base run dir containing acts.jsonl")
    ap.add_argument("--out", required=True, help="WORM out dir (must not exist)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze_path", default="", help="Optional freeze JSON path in repo root")
    ap.add_argument("--patch_diff", default="", help="Optional unified diff path to hash into freeze")
    args = ap.parse_args()

    ensure_out_dir_absent(args.out)
    os.makedirs(args.out, exist_ok=False)

    base_acts = os.path.join(args.acts_run, "acts.jsonl")
    if not os.path.exists(base_acts):
        _fail(f"ERROR: base acts.jsonl not found: {base_acts}")

    base_acts_sha256 = sha256_file(base_acts)
    store = ActStore.load_jsonl(base_acts)
    store_hash_excl = store.content_hash(exclude_kinds=["gate_table_ctxsig", "concept_csv", "goal"])

    # --- Define concepts (CSV) ---
    iface_extract = {
        "input_schema": {"text": "str"},
        "output_schema": {"value": "int"},
        "validator_id": "int_value_exact",
    }
    concept_extract = make_concept_act(
        store_hash_excluding_semantic=store_hash_excl,
        step=1,
        title="extract_int_v0",
        interface=iface_extract,
        program=[
            Instruction("CSV_GET_INPUT", {"name": "text", "out": "t"}),
            Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["t"], "out": "d"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "n"}),
            Instruction("CSV_RETURN", {"var": "n"}),
        ],
    )

    iface_sum = {
        "input_schema": {"a": "int", "b": "int"},
        "output_schema": {"value": "int"},
        "validator_id": "int_value_exact",
    }
    concept_sum = make_concept_act(
        store_hash_excluding_semantic=store_hash_excl,
        step=2,
        title="sum_int_v0",
        interface=iface_sum,
        program=[
            Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
            Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
            Instruction("CSV_PRIMITIVE", {"fn": "add_int", "in": ["a", "b"], "out": "s"}),
            Instruction("CSV_RETURN", {"var": "s"}),
        ],
    )

    iface_json = {
        "input_schema": {"a": "int", "b": "int"},
        "output_schema": {"value": "str"},
        "validator_id": "json_ab_int_exact",
    }
    concept_json = make_concept_act(
        store_hash_excluding_semantic=store_hash_excl,
        step=3,
        title="json_ab_v0",
        interface=iface_json,
        program=[
            Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
            Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
            Instruction("CSV_PRIMITIVE", {"fn": "make_dict_ab", "in": ["a", "b"], "out": "obj"}),
            Instruction("CSV_PRIMITIVE", {"fn": "json_canonical", "in": ["obj"], "out": "s"}),
            Instruction("CSV_RETURN", {"var": "s"}),
        ],
    )

    # Concept that calls concepts: parse text_a/text_b -> ints -> sum
    iface_parse_sum = {
        "input_schema": {"text_a": "str", "text_b": "str"},
        "output_schema": {"value": "int"},
        "validator_id": "int_value_exact",
    }
    concept_parse_sum = make_concept_act(
        store_hash_excluding_semantic=store_hash_excl,
        step=4,
        title="parse_and_sum_v0",
        interface=iface_parse_sum,
        program=[
            Instruction("CSV_GET_INPUT", {"name": "text_a", "out": "ta"}),
            Instruction("CSV_CALL", {"concept_id": concept_extract.id, "bind": {"text": "ta"}, "out": "a"}),
            Instruction("CSV_GET_INPUT", {"name": "text_b", "out": "tb"}),
            Instruction("CSV_CALL", {"concept_id": concept_extract.id, "bind": {"text": "tb"}, "out": "b"}),
            Instruction("CSV_CALL", {"concept_id": concept_sum.id, "bind": {"a": "a", "b": "b"}, "out": "s"}),
            Instruction("CSV_RETURN", {"var": "s"}),
        ],
    )

    # Ethics-violating concept (should never be promoted; also must fail-closed at execute).
    iface_bad = {"input_schema": {}, "output_schema": {"value": "str"}, "validator_id": ""}
    concept_bad = make_concept_act(
        store_hash_excluding_semantic=store_hash_excl,
        step=5,
        title="bad_sentience_claim_v0",
        interface=iface_bad,
        program=[
            Instruction("CSV_CONST", {"out": "s", "value": "EU SOU CONSCIENTE"}),
            Instruction("CSV_RETURN", {"var": "s"}),
        ],
    )

    # Strong-claim concept: should be downgraded to IC without evidence.
    iface_claim = {"input_schema": {}, "output_schema": {"value": "str"}, "validator_id": ""}
    concept_claim = make_concept_act(
        store_hash_excluding_semantic=store_hash_excl,
        step=6,
        title="strong_claim_no_evidence_v0",
        interface=iface_claim,
        program=[
            Instruction("CSV_CONST", {"out": "s", "value": "COM CERTEZA PARIS"}),
            Instruction("CSV_RETURN", {"var": "s"}),
        ],
    )

    concepts_all = [concept_extract, concept_sum, concept_json, concept_parse_sum, concept_bad, concept_claim]

    # --- Goals ---
    goal_extract = make_goal_act(
        store_hash_excluding_semantic=store_hash_excl,
        step=10,
        title="goal_extract_int",
        priority=3,
        selector={"kind": "interface_sig", "iface_sig": iface_sig(iface_extract)},
        inputs={"text": "abc123"},
        expected=123,
    )
    goal_sum = make_goal_act(
        store_hash_excluding_semantic=store_hash_excl,
        step=11,
        title="goal_sum_from_texts",
        priority=2,
        selector={"kind": "interface_sig", "iface_sig": iface_sig(iface_parse_sum)},
        inputs={"text_a": "a=12", "text_b": "b=30"},
        expected=42,
    )
    goal_json = make_goal_act(
        store_hash_excluding_semantic=store_hash_excl,
        step=12,
        title="goal_json_ab",
        priority=1,
        selector={"kind": "interface_sig", "iface_sig": iface_sig(iface_json)},
        inputs={"a": 7, "b": 5},
        expected={"a": 7, "b": 5},
    )
    goals = [goal_extract, goal_sum, goal_json]

    # --- Baseline (inline subgraph) ---
    baseline_rows: List[Dict[str, Any]] = []
    for g in goals:
        ge = g.evidence.get("goal", {}) if isinstance(g.evidence, dict) else {}
        title = str((g.evidence.get("meta", {}) if isinstance(g.evidence, dict) else {}).get("title") or g.id)
        inputs = dict(ge.get("inputs", {}) if isinstance(ge, dict) else {})
        if "extract" in title:
            out = run_inline_task("extract_int", inputs)
        elif "sum" in title:
            out = run_inline_task("sum_from_texts", inputs)
        else:
            out = run_inline_task("json_ab", inputs)
        baseline_rows.append({"goal_id": g.id, "title": title, "output": value_to_text(out)})
    baseline_text = canonical_json_dumps({"baseline": baseline_rows})
    baseline_hash = sha256_hex(baseline_text.encode("utf-8"))

    # --- Promotion: select concepts by ethics/IC-IR pressure (fail-closed) ---
    tmp_store = ActStore.load_jsonl(base_acts)
    for c in concepts_all:
        tmp_store.add(c)
    tmp_engine = Engine(tmp_store, seed=int(args.seed), config=EngineConfig())

    promoted_concepts: List[Act] = []
    rejected_concepts: List[Dict[str, Any]] = []
    for c in concepts_all:
        ethics = validate_act_for_promotion(c)
        if not bool(ethics.ok):
            rejected_concepts.append({"concept_id": c.id, "title": c.evidence.get("meta", {}).get("title"), "reason": ethics.reason, "violated": ethics.violated_laws})
            continue
        # Additional promotion pressure: execution-time emission must pass ethics and must not
        # downgrade to IC due to strong claim without evidence.
        r = tmp_engine.execute_concept_csv(concept_act_id=c.id, inputs={}, expected=None, step=0)
        meta = r.get("meta") or {}
        eth_ok = bool((meta.get("ethics") or {}).get("ok", True))
        unc = (meta.get("uncertainty") or {}) if isinstance(meta, dict) else {}
        if not eth_ok:
            rejected_concepts.append({"concept_id": c.id, "title": c.evidence.get("meta", {}).get("title"), "reason": "ethics_fail_closed_execute", "violated": (meta.get("ethics") or {}).get("violated_laws", [])})
            continue
        if str(unc.get("mode_out") or "") == "IC":
            rejected_concepts.append({"concept_id": c.id, "title": c.evidence.get("meta", {}).get("title"), "reason": "uncertainty_downgrade_ic", "violated": []})
            continue
        promoted_concepts.append(c)

    promoted_acts: List[Act] = promoted_concepts + goals

    promo_dir = os.path.join(args.out, "promotion")
    os.makedirs(promo_dir, exist_ok=False)
    acts_promoted = os.path.join(promo_dir, "acts_promoted.jsonl")
    promoted_sha256 = write_promoted_acts_preserve_order(
        base_acts_path=base_acts, out_acts_path=acts_promoted, appended_acts=promoted_acts
    )

    # Promotion ledger (append-only, hash-chained).
    promotion_ledger_path = os.path.join(promo_dir, "promotion_ledger.jsonl")
    ledger = Ledger(path=promotion_ledger_path)
    for idx, a in enumerate(promoted_acts):
        patch = Patch(kind="ADD_ACT", payload={"act_id": str(a.id), "kind": str(a.kind)})
        ledger.append(
            step=int(idx),
            patch=patch,
            acts_hash=str(promoted_sha256),
            metrics={"promotion": True, "ethics_ok": True, "act_id": str(a.id)},
            snapshot_path=None,
        )
    promotion_chain_ok = ledger.verify_chain()

    manifest = {
        "base_acts_path": str(base_acts),
        "base_acts_sha256": str(base_acts_sha256),
        "acts_promoted_path": str(acts_promoted),
        "acts_promoted_sha256": str(promoted_sha256),
        "store_content_hash_excluding_semantic": str(store_hash_excl),
        "promoted_concepts": [c.id for c in promoted_concepts],
        "rejected_concepts": rejected_concepts,
        "promotion_chain_ok": bool(promotion_chain_ok),
    }
    promo_manifest_path = os.path.join(promo_dir, "promotion_manifest.json")
    with open(promo_manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False, indent=2))

    # --- From-store execution (goal->concept) ---
    store2 = ActStore.load_jsonl(acts_promoted)
    engine2 = Engine(store2, seed=int(args.seed), config=EngineConfig())

    goal_results: List[Dict[str, Any]] = []
    call_depth_max = 0
    ethics_passed = 0
    ic_count = 0
    for g in goals:
        r = engine2.execute_goal(goal_act_id=g.id, step=0, max_depth=8)
        tr = r.get("trace") or {}
        meta = (tr.get("concept_meta") or {}) if isinstance(tr, dict) else {}
        u = (meta.get("uncertainty") or {}) if isinstance(meta, dict) else {}
        eth = (meta.get("ethics") or {}) if isinstance(meta, dict) else {}
        events = r.get("events") or []
        for ev in events:
            if isinstance(ev, dict):
                call_depth_max = max(call_depth_max, int(ev.get("depth", 0) or 0))
        if bool(eth.get("ok", True)):
            ethics_passed += 1
        if str(u.get("mode_out") or "") == "IC":
            ic_count += 1
        goal_results.append(
            {
                "goal_id": g.id,
                "ok": bool(r.get("ok", False)),
                "output": value_to_text(r.get("output")),
                "selected_concept_id": str(tr.get("selected_concept_id") or ""),
                "ethics": eth,
                "uncertainty": u,
            }
        )

    from_store_text = canonical_json_dumps({"from_store": goal_results})
    from_store_hash = sha256_hex(from_store_text.encode("utf-8"))

    # Invariance: CSV output should match inline output for these deterministic tasks.
    mismatch = 0
    base_map = {r["goal_id"]: r["output"] for r in baseline_rows}
    for r in goal_results:
        if str(r.get("output") or "") != str(base_map.get(r["goal_id"]) or ""):
            mismatch += 1

    summary = {
        "seed": int(args.seed),
        "baseline_hash": str(baseline_hash),
        "from_store_hash": str(from_store_hash),
        "csv_expansion_invariance_ok": mismatch == 0,
        "mismatch_goals": int(mismatch),
        "goals_total": int(len(goals)),
        "reuse_rate": float(sum(1 for r in goal_results if r.get("selected_concept_id")) / max(1, len(goals))),
        "call_depth_max": int(call_depth_max),
        "ethics_checks_passed": int(ethics_passed),
        "uncertainty_ic_count": int(ic_count),
        "promotion_chain_ok": bool(promotion_chain_ok),
    }

    summary_csv = os.path.join(args.out, "summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write(
            "seed,goals_total,mismatch_goals,csv_invariance_ok,reuse_rate,call_depth_max,ethics_checks_passed,uncertainty_ic_count,promotion_chain_ok,baseline_hash,from_store_hash\n"
        )
        f.write(
            f"{summary['seed']},{summary['goals_total']},{summary['mismatch_goals']},{int(summary['csv_expansion_invariance_ok'])},{summary['reuse_rate']},{summary['call_depth_max']},{summary['ethics_checks_passed']},{summary['uncertainty_ic_count']},{int(summary['promotion_chain_ok'])},{summary['baseline_hash']},{summary['from_store_hash']}\n"
        )
    summary_json = os.path.join(args.out, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summary, "baseline": baseline_rows, "from_store": goal_results}, ensure_ascii=False, indent=2))

    # Optional freeze (repo root): record hashes + commands.
    if args.freeze_path:
        if os.path.exists(args.freeze_path):
            _fail(f"ERROR: --freeze_path already exists: {args.freeze_path}")
        sha: Dict[str, str] = {
            "base_acts_jsonl": str(base_acts_sha256),
            "acts_promoted_jsonl": str(promoted_sha256),
            "promotion_manifest": str(sha256_file(promo_manifest_path)),
            "promotion_ledger": str(sha256_file(promotion_ledger_path)),
            "summary_csv": str(sha256_file(summary_csv)),
            "summary_json": str(sha256_file(summary_json)),
        }
        if args.patch_diff:
            if os.path.exists(args.patch_diff):
                sha["patch_diff"] = str(sha256_file(args.patch_diff))
        freeze = {
            "name": "V59_CSV_GOAL_ETHICS_UNCERTAINTY",
            "acts_source_run": str(args.acts_run),
            "out_dir": str(args.out),
            "commands": [" ".join(sys.argv)],
            "verify_chain": bool(promotion_chain_ok),
            "sha256": sha,
            "summary": summary,
        }
        with open(args.freeze_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(freeze, ensure_ascii=False, indent=2))

    print(json.dumps({"summary": summary, "out_dir": args.out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

