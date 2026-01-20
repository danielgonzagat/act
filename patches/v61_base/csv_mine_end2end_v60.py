#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Patch, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.concepts import PRIMITIVE_OPS
from atos_core.csv_miner import CsvCandidate, materialize_concept_act_from_candidate, mine_csv_candidates
from atos_core.engine import Engine, EngineConfig
from atos_core.ethics import validate_act_for_promotion
from atos_core.ledger import Ledger
from atos_core.proof import act_body_sha256_placeholder, build_concept_pcc_certificate_v1, verify_concept_pcc
from atos_core.store import ActStore
from atos_core.suite import CHAT_DIALOGUES_20X3, run_chat_suite


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


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"ERROR: path already exists: {path}")


def stable_act_id(prefix: str, body: Dict[str, Any]) -> str:
    return f"{prefix}{sha256_hex(canonical_json_dumps(body).encode('utf-8'))[:12]}"


def make_goal_act(
    *,
    step: int,
    store_hash_excl_semantic: str,
    title: str,
    iface_sig: str,
    inputs: Dict[str, Any],
    expected: Any,
    priority: int,
) -> Act:
    ev = {
        "name": "goal_v0",
        "meta": {
            "title": str(title),
            "trained_on_store_content_hash": str(store_hash_excl_semantic),
        },
        "goal": {
            "priority": int(priority),
            "selector": {"kind": "interface_sig", "iface_sig": str(iface_sig)},
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


def _iface_sig_from_act(act: Act) -> str:
    ev = act.evidence if isinstance(act.evidence, dict) else {}
    iface = ev.get("interface") if isinstance(ev, dict) else {}
    iface = iface if isinstance(iface, dict) else {}
    body = {
        "in": iface.get("input_schema", {}),
        "out": iface.get("output_schema", {}),
        "validator_id": iface.get("validator_id", ""),
    }
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _trace_program_sig(events: List[Dict[str, Any]]) -> str:
    return sha256_hex(canonical_json_dumps(events).encode("utf-8"))


def _run_inline_extract_int_with_trace(*, text: str) -> Tuple[int, List[Dict[str, Any]]]:
    _, fn_scan = PRIMITIVE_OPS["scan_digits"]
    _, fn_d2i = PRIMITIVE_OPS["digits_to_int"]
    events: List[Dict[str, Any]] = []
    events.append({"t": "GET_INPUT", "name": "text", "out": "t"})
    events.append({"t": "PRIMITIVE", "fn": "scan_digits", "in": ["t"], "out": "d"})
    events.append({"t": "PRIMITIVE", "fn": "digits_to_int", "in": ["d"], "out": "n"})
    events.append({"t": "RETURN", "var": "n"})
    digits = fn_scan(text)
    n = fn_d2i(digits)
    return int(n), events


def _transcript_hash(transcripts: List[Dict[str, Any]]) -> str:
    full = "\n".join(str(t.get("full_text") or "") for t in transcripts)
    return sha256_hex(full.encode("utf-8"))


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ensure_absent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(canonical_json_dumps(r))
            f.write("\n")
    os.replace(tmp, path)


def write_promoted_acts_preserve_order(
    *, base_acts_path: str, out_acts_path: str, appended_acts: List[Act]
) -> str:
    with open(base_acts_path, "rb") as f:
        base_bytes = f.read()
    if base_bytes and not base_bytes.endswith(b"\n"):
        base_bytes += b"\n"
    tail = b"".join(canonical_json_dumps(a.to_dict()).encode("utf-8") + b"\n" for a in appended_acts)
    tmp = out_acts_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(base_bytes)
        f.write(tail)
    os.replace(tmp, out_acts_path)
    return sha256_file(out_acts_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patch_diff", default="")
    ap.add_argument("--freeze_path", default="")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--chat_dialogues", type=int, default=2, help="How many CHAT_DIALOGUES_20X3 to use for goal-shadow demo.")
    args = ap.parse_args()

    ensure_absent(args.out)
    if args.freeze_path:
        ensure_absent(args.freeze_path)
    os.makedirs(args.out, exist_ok=False)

    base_acts = os.path.join(args.acts_run, "acts.jsonl")
    if not os.path.exists(base_acts):
        _fail(f"ERROR: missing base acts.jsonl: {base_acts}")
    base_acts_sha256 = sha256_file(base_acts)

    store = ActStore.load_jsonl(base_acts)
    store_hash_excl = store.content_hash(exclude_kinds=["gate_table_ctxsig", "concept_csv", "goal"])

    traces_dir = os.path.join(args.out, "traces")
    os.makedirs(traces_dir, exist_ok=False)
    csv_exec_path = os.path.join(traces_dir, "csv_exec.jsonl")

    # (a) Baseline inline mini-suite (deterministic) + trace for mining.
    baseline_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    episodes = max(3, int(args.episodes))
    for i in range(episodes):
        text = f"abc{i}123"
        out_int, events = _run_inline_extract_int_with_trace(text=text)
        out_text = str(int(out_int))
        inputs = {"text": str(text)}
        rec = {
            "run_id": str(args.out),
            "ctx_sig": f"inline␟extract_int␟i={i}",
            "goal_id": f"inline_goal_extract_int_{i}",
            "program_sig": _trace_program_sig(events),
            "events": list(events),
            "inputs": dict(inputs),
            "inputs_sig": sha256_hex(canonical_json_dumps(inputs).encode("utf-8")),
            "output_text": out_text,
            "output_sig": sha256_hex(out_text.encode("utf-8")),
        }
        trace_rows.append(rec)
        baseline_rows.append(
            {
                "goal_id": str(rec["goal_id"]),
                "inputs": dict(inputs),
                "expected": int(out_int),
                "expected_output_text": str(out_text),
            }
        )

    write_jsonl(csv_exec_path, trace_rows)
    csv_exec_sha256 = sha256_file(csv_exec_path)

    baseline_text = canonical_json_dumps({"baseline": baseline_rows})
    baseline_hash = sha256_hex(baseline_text.encode("utf-8"))

    # (b) Mine deterministic candidates and pick top-1.
    candidates = mine_csv_candidates(csv_exec_path, min_ops=2, max_ops=6, bits_per_op=128, overhead_bits=1024)
    mined_candidates_path = os.path.join(args.out, "mined_candidates.json")
    with open(mined_candidates_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"candidates": [c.to_dict() for c in candidates]}, ensure_ascii=False, indent=2))
    if not candidates:
        _fail("ERROR: miner produced 0 candidates")
    top1 = candidates[0]

    # (c) Materialize concept + build PCC certificate + verify.
    concept = materialize_concept_act_from_candidate(
        top1,
        step=100,
        store_content_hash_excluding_semantic=store_hash_excl,
        title="mined_extract_int_v60",
        overhead_bits=1024,
        meta={
            "trace_file_sha256": str(csv_exec_sha256),
        },
    )

    # Build test vectors from examples (>=3, deterministic, unique by expected_sig).
    uniq: set = set()
    test_vectors: List[Dict[str, Any]] = []
    for ex in top1.examples:
        if not isinstance(ex, dict):
            continue
        sig = str(ex.get("expected_sig") or "")
        if sig in uniq:
            continue
        uniq.add(sig)
        tv = {
            "inputs": dict(ex.get("inputs") or {}),
            "expected": ex.get("expected"),
            "expected_output_text": str(ex.get("expected_output_text") or ""),
        }
        test_vectors.append(tv)
        if len(test_vectors) >= 3:
            break
    if len(test_vectors) < 3:
        _fail("ERROR: not enough test vectors for PCC (need >=3)")

    ethics = validate_act_for_promotion(concept)
    if not bool(ethics.ok):
        _fail(f"ERROR: mined concept fails ethics promotion: {ethics.reason}:{ethics.violated_laws}")

    cert = build_concept_pcc_certificate_v1(
        concept,
        mined_from={
            "trace_file": str(csv_exec_path),
            "trace_file_sha256": str(csv_exec_sha256),
            "store_hash_excluding_semantic": str(store_hash_excl),
            "seed": int(args.seed),
            "candidate_sig": str(top1.candidate_sig),
            "ctx_sigs": [str(x.get("ctx_sig") or "") for x in top1.examples[:5] if isinstance(x, dict)],
        },
        test_vectors=test_vectors,
        ethics_verdict=ethics.to_dict(),
        uncertainty_policy="no_ic",
    )
    concept.evidence.setdefault("certificate_v1", cert)
    # Fill act_body_sha256 deterministically with placeholder semantics.
    try:
        concept.evidence["certificate_v1"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(concept)
    except Exception:
        pass

    proof_v = verify_concept_pcc(concept, store)
    if not bool(proof_v.ok):
        _fail(f"ERROR: PCC verify failed: {proof_v.reason}:{proof_v.details}")

    # (d) Promote: base + [concept] + [goals], preserve base order, WORM.
    promo_dir = os.path.join(args.out, "promotion")
    os.makedirs(promo_dir, exist_ok=False)
    acts_promoted = os.path.join(promo_dir, "acts_promoted.jsonl")

    concept_iface_sig = _iface_sig_from_act(concept)
    goals: List[Act] = []
    for i, row in enumerate(baseline_rows):
        title = f"goal_extract_int_{i}"
        goals.append(
            make_goal_act(
                step=200 + i,
                store_hash_excl_semantic=store_hash_excl,
                title=title,
                iface_sig=concept_iface_sig,
                inputs=dict(row["inputs"]),
                expected=row["expected"],
                priority=10,
            )
        )
    appended = [concept] + goals
    promoted_sha256 = write_promoted_acts_preserve_order(
        base_acts_path=base_acts, out_acts_path=acts_promoted, appended_acts=appended
    )

    promotion_ledger_path = os.path.join(promo_dir, "promotion_ledger.jsonl")
    ledger = Ledger(path=promotion_ledger_path)
    for idx, a in enumerate(appended):
        patch = Patch(kind="ADD_ACT", payload={"act_id": str(a.id), "kind": str(a.kind)})
        ledger.append(
            step=int(idx),
            patch=patch,
            acts_hash=str(promoted_sha256),
            metrics={"promotion": True, "act_id": str(a.id), "kind": str(a.kind)},
            snapshot_path=None,
        )
    promotion_chain_ok = ledger.verify_chain()

    promotion_manifest = {
        "base_acts_path": str(base_acts),
        "base_acts_sha256": str(base_acts_sha256),
        "acts_promoted_path": str(acts_promoted),
        "acts_promoted_sha256": str(promoted_sha256),
        "store_hash_excluding_semantic": str(store_hash_excl),
        "csv_exec_path": str(csv_exec_path),
        "csv_exec_sha256": str(csv_exec_sha256),
        "mined_top1": top1.to_dict(),
        "promoted_concept_id": str(concept.id),
        "promoted_goal_ids": [str(g.id) for g in goals],
        "promotion_chain_ok": bool(promotion_chain_ok),
        "ethics": ethics.to_dict(),
        "pcc_verify": proof_v.to_dict(),
    }
    promotion_manifest_path = os.path.join(promo_dir, "promotion_manifest.json")
    with open(promotion_manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(promotion_manifest, ensure_ascii=False, indent=2))

    # (e) From-store via goal->concept and invariance check against baseline.
    store2 = ActStore.load_jsonl(acts_promoted)
    engine2 = Engine(store2, seed=int(args.seed), config=EngineConfig())

    from_store_rows: List[Dict[str, Any]] = []
    mismatch_goals = 0
    call_depth_max = 0
    ethics_passed = 0
    ic_count = 0
    for i, g in enumerate(goals):
        r = engine2.execute_goal(goal_act_id=g.id, step=i, max_depth=8)
        tr = r.get("trace") if isinstance(r, dict) else {}
        tr = tr if isinstance(tr, dict) else {}
        meta = tr.get("concept_meta") if isinstance(tr.get("concept_meta"), dict) else {}
        eth2 = meta.get("ethics") if isinstance(meta.get("ethics"), dict) else {}
        unc2 = meta.get("uncertainty") if isinstance(meta.get("uncertainty"), dict) else {}
        if bool(eth2.get("ok", True)):
            ethics_passed += 1
        if str(unc2.get("mode_out") or "") == "IC":
            ic_count += 1
        evs = r.get("events") if isinstance(r, dict) else []
        if isinstance(evs, list):
            for ev in evs:
                if isinstance(ev, dict):
                    call_depth_max = max(call_depth_max, int(ev.get("depth", 0) or 0))
        out_text = str(meta.get("output_text") or "")
        expected_text = str(baseline_rows[i]["expected_output_text"])
        if out_text != expected_text:
            mismatch_goals += 1
        from_store_rows.append(
            {
                "goal_id": str(g.id),
                "ok": bool(r.get("ok", False)),
                "output_text": out_text,
                "expected_output_text": expected_text,
                "selected_concept_id": str(tr.get("selected_concept_id") or ""),
                "ethics": eth2,
                "uncertainty": unc2,
            }
        )

    from_store_text = canonical_json_dumps({"from_store": from_store_rows})
    from_store_hash = sha256_hex(from_store_text.encode("utf-8"))

    reuse = sum(1 for r in from_store_rows if str(r.get("selected_concept_id") or ""))
    reuse_rate = float(reuse / max(1, len(from_store_rows)))

    # (f) Goals in chat loop shadow (telemetry only, must not change tokens).
    goal_shadow_path = os.path.join(traces_dir, "goal_shadow.jsonl")
    chat_dialogues = CHAT_DIALOGUES_20X3[: max(0, int(args.chat_dialogues))]
    engine_chat_a = Engine(store2, seed=int(args.seed), config=EngineConfig())
    base_transcripts, _ = run_chat_suite(
        engine_chat_a,
        dialogues=chat_dialogues,
        max_new_tokens=int(args.max_new_tokens),
        prefix_k=8,
        template_ngram_n=6,
        template_prefix_window=32,
        csv=None,
        goal_shadow_log_path=None,
    )
    chat_hash_base = _transcript_hash(base_transcripts)

    engine_chat_b = Engine(store2, seed=int(args.seed), config=EngineConfig())
    shadow_transcripts, _ = run_chat_suite(
        engine_chat_b,
        dialogues=chat_dialogues,
        max_new_tokens=int(args.max_new_tokens),
        prefix_k=8,
        template_ngram_n=6,
        template_prefix_window=32,
        csv=None,
        goal_shadow_log_path=goal_shadow_path,
    )
    chat_hash_shadow = _transcript_hash(shadow_transcripts)
    goal_shadow_invariance_ok = chat_hash_shadow == chat_hash_base

    goal_shadow_lines = 0
    try:
        with open(goal_shadow_path, "r", encoding="utf-8") as f:
            for _ in f:
                goal_shadow_lines += 1
    except Exception:
        goal_shadow_lines = 0

    summary = {
        "seed": int(args.seed),
        "goals_total": int(len(goals)),
        "mismatch_goals": int(mismatch_goals),
        "csv_invariance_ok": bool(mismatch_goals == 0),
        "mined_candidates_total": int(len(candidates)),
        "promoted_concepts_total": 1,
        "gain_bits_est_total": int(top1.gain_bits_est),
        "reuse_rate": float(reuse_rate),
        "call_depth_max": int(call_depth_max),
        "ethics_checks_passed": int(ethics_passed),
        "uncertainty_ic_count": int(ic_count),
        "promotion_chain_ok": bool(promotion_chain_ok),
        "baseline_hash": str(baseline_hash),
        "from_store_hash": str(from_store_hash),
        "goal_shadow_invariance_ok": bool(goal_shadow_invariance_ok),
        "goal_shadow_lines": int(goal_shadow_lines),
        "chat_hash_base": str(chat_hash_base),
        "chat_hash_shadow": str(chat_hash_shadow),
    }

    summary_csv = os.path.join(args.out, "summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write(
            "seed,goals_total,mismatch_goals,csv_invariance_ok,mined_candidates_total,promoted_concepts_total,gain_bits_est_total,reuse_rate,call_depth_max,ethics_checks_passed,uncertainty_ic_count,promotion_chain_ok,goal_shadow_invariance_ok,goal_shadow_lines,baseline_hash,from_store_hash\n"
        )
        f.write(
            f"{summary['seed']},{summary['goals_total']},{summary['mismatch_goals']},{int(summary['csv_invariance_ok'])},{summary['mined_candidates_total']},{summary['promoted_concepts_total']},{summary['gain_bits_est_total']},{summary['reuse_rate']},{summary['call_depth_max']},{summary['ethics_checks_passed']},{summary['uncertainty_ic_count']},{int(summary['promotion_chain_ok'])},{int(summary['goal_shadow_invariance_ok'])},{summary['goal_shadow_lines']},{summary['baseline_hash']},{summary['from_store_hash']}\n"
        )
    summary_json = os.path.join(args.out, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "summary": summary,
                    "baseline": baseline_rows,
                    "from_store": from_store_rows,
                    "promotion_manifest": promotion_manifest,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if args.freeze_path:
        sha: Dict[str, str] = {
            "base_acts_jsonl": str(base_acts_sha256),
            "csv_exec_jsonl": str(csv_exec_sha256),
            "mined_candidates_json": str(sha256_file(mined_candidates_path)),
            "acts_promoted_jsonl": str(promoted_sha256),
            "promotion_ledger": str(sha256_file(promotion_ledger_path)),
            "promotion_manifest": str(sha256_file(promotion_manifest_path)),
            "summary_csv": str(sha256_file(summary_csv)),
            "summary_json": str(sha256_file(summary_json)),
            "goal_shadow_jsonl": str(sha256_file(goal_shadow_path)) if os.path.exists(goal_shadow_path) else "",
        }
        if args.patch_diff and os.path.exists(args.patch_diff):
            sha["patch_diff"] = str(sha256_file(args.patch_diff))

        freeze = {
            "name": "V60_CSV_MINER_PCC_GOAL_SHADOW",
            "acts_source_run": str(args.acts_run),
            "out_dir": str(args.out),
            "commands": [" ".join(sys.argv)],
            "verify_chain": bool(promotion_chain_ok),
            "sha256": sha,
            "summary": summary,
        }
        with open(args.freeze_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(freeze, ensure_ascii=False, indent=2))

    print(json.dumps({"summary": summary, "out_dir": str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

