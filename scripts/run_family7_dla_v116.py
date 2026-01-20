#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.conversation_loop_v116 import run_conversation_v116
from atos_core.external_world_gating_v113 import external_world_access_v113
from atos_core.external_world_ledger_v111 import (
    EXTERNAL_WORLD_ACTION_SEARCH_V111,
    EXTERNAL_WORLD_REASON_CODES_V111,
    compute_external_world_chain_hash_v111,
    verify_external_world_event_sig_chain_v111,
)
from atos_core.fluency_survival_v112 import fluency_contract_v112, fluency_survival_plan_v112, summarize_fluency_fail_code_v112


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


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
        _fail(f"worm_exists:{path}")


ACK_TO_CHOICE_LABEL_V112 = {
    "ok",
    "okay",
    "certo",
    "beleza",
    "blz",
    "continua",
    "continue",
    "segue",
    "vai",
    "faz",
    "pode",
    "sim",
}


def _canon_ack_token_v112(s: str) -> str:
    t = str(s or "").strip().lower()
    t = " ".join([x for x in t.split() if x])
    return t


def _choiceify_minimal_ack_v112(user_turn_texts: Sequence[str]) -> List[str]:
    out: List[str] = []
    for s in user_turn_texts:
        cs = _canon_ack_token_v112(str(s))
        if cs in ACK_TO_CHOICE_LABEL_V112:
            out.append("A")
        else:
            out.append(str(s))
    return out


def _load_jsonl_payload_view(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                continue
            out.append(dict(payload))
    return out


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


def _write_once_json(path: Path, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        _fail(f"tmp_exists:{tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _compute_external_world_access_once_v113(
    *,
    world_manifest: str,
    reason_code: str,
    query: str,
    seed: int,
) -> List[Dict[str, Any]]:
    evs, _ = external_world_access_v113(
        allowed=True,
        world_manifest=str(world_manifest),
        action=EXTERNAL_WORLD_ACTION_SEARCH_V111,
        reason_code=str(reason_code),
        args={"query": str(query), "limit": 3, "roles": ["user"]},
        seed=int(seed),
        turn_index=0,
        prev_event_sig="",
    )
    return list(evs)


def _count_unresolved_reference_events(binding_events: Sequence[Dict[str, Any]]) -> int:
    bad = 0
    for ev in binding_events:
        if not isinstance(ev, dict):
            continue
        t = str(ev.get("type") or "")
        if t in {"BIND_MISS", "BIND_AMBIGUOUS"}:
            bad += 1
    return int(bad)


def _unresolved_reference_final_from_flow(flow_events: Sequence[Dict[str, Any]]) -> int:
    if not flow_events:
        return 0
    last = flow_events[-1] if isinstance(flow_events[-1], dict) else {}
    flags = last.get("flow_flags_v108")
    if not isinstance(flags, dict):
        return 0
    return 1 if bool(flags.get("UNRESOLVED_REFERENCE")) else 0


def _count_semantic_contradiction_flags(semantic_events: Sequence[Dict[str, Any]]) -> int:
    cnt = 0
    for ev in semantic_events:
        if not isinstance(ev, dict):
            continue
        flags = ev.get("flags_v109")
        if not isinstance(flags, dict):
            continue
        if bool(flags.get("CONTRADICTION_UNREPAIRED")):
            cnt += 1
    return int(cnt)


def _write_external_world_ledger(*, task_dir: Path, events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    events_path = task_dir / "external_world_events.jsonl"
    _ensure_absent(events_path)
    if events:
        with open(events_path, "x", encoding="utf-8") as f:
            for e in events:
                f.write(canonical_json_dumps(e))
                f.write("\n")
    else:
        events_path.write_text("", encoding="utf-8")

    ok_sig, reason_sig, _ = verify_external_world_event_sig_chain_v111(list(events))
    if not ok_sig:
        _fail(f"external_world_sig_chain_fail:{reason_sig}")
    chain_hash = compute_external_world_chain_hash_v111(list(events))
    snap = {
        "schema_version": 111,
        "kind": "external_world_registry_snapshot_v111",
        "events_total": int(len(events)),
        "external_world_chain_hash_v111": str(chain_hash),
    }
    snap_path = task_dir / "external_world_registry_snapshot_v111.json"
    _write_once_json(snap_path, snap)
    return {
        "events_total": int(len(events)),
        "external_world_chain_hash_v111": str(chain_hash),
        "external_world_events_jsonl": str(events_path),
        "external_world_registry_snapshot_v111_json": str(snap_path),
    }


def _compute_freeze_manifest_v116(*, task_dir: Path, sha256_paths: Dict[str, str]) -> Dict[str, Any]:
    sha256: Dict[str, str] = {}
    sha256_rel: Dict[str, str] = {}
    for k, p in sorted(sha256_paths.items(), key=lambda kv: str(kv[0])):
        pp = str(p or "")
        sha256_rel[str(k)] = str(os.path.basename(pp)) if pp else ""
        if pp and os.path.exists(pp):
            sha256[str(k)] = _sha256_file(Path(pp))
    body = {
        "schema_version": 116,
        "kind": "freeze_manifest_v116",
        "sha256": dict(sha256),
        "sha256_paths": dict(sha256_rel),
    }
    body["manifest_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--max_tasks", default="9999")
    ap.add_argument("--max_rewrites", default="4")
    ap.add_argument("--max_replans_per_turn", default="3")
    args = ap.parse_args()

    seed = int(args.seed)
    out_dir = Path(str(args.out))
    _ensure_absent(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=False)

    tasks: List[Dict[str, Any]] = []
    with open(str(args.tasks), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    if not tasks:
        _fail("empty_tasks")

    max_tasks = int(args.max_tasks)
    max_rewrites = int(args.max_rewrites)
    max_replans = int(args.max_replans_per_turn)

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks[:max_tasks]):
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or f"task_{i:03d}")
        user_turns = task.get("user_turns") if isinstance(task.get("user_turns"), list) else []
        user_turn_texts = [str(x) for x in user_turns if isinstance(x, str)]
        user_turn_texts = _choiceify_minimal_ack_v112(user_turn_texts)
        require_fluency = bool(task.get("require_fluency", True))
        allow_external = bool(task.get("allow_external_world_once", False))
        world_manifest = str(task.get("world_manifest") or "")
        probe_reason = str(task.get("external_world_probe_reason_code") or "validator_failed_fluency_contract")

        task_dir = out_dir / "task_{i:03d}".format(i=i)
        _ensure_absent(task_dir)
        task_dir.mkdir(parents=True, exist_ok=False)

        attempts: List[Dict[str, Any]] = []
        chosen_attempt = -1

        rewrite_seeds = fluency_survival_plan_v112(base_seed=int(seed), max_attempts=int(max_rewrites))
        ext_used = False
        ext_used_reason = ""
        ext_events_final: List[Dict[str, Any]] = []

        for a, seed_used in enumerate(rewrite_seeds):
            attempt_dir = task_dir / "attempt_{a:03d}".format(a=a)
            _ensure_absent(attempt_dir)
            conv = run_conversation_v116(user_turn_texts=list(user_turn_texts), out_dir=str(attempt_dir), seed=int(seed_used), max_replans_per_turn=int(max_replans))

            ok_gate = bool(conv.get("gate_v115_ok", False))
            gate_reason = str(conv.get("gate_v115_reason") or "")
            ok_dialogue = bool(conv.get("dialogue_survival_v116_ok", False))
            reason_dialogue = str(conv.get("dialogue_survival_v116_reason") or "")

            transcript_rows = _load_jsonl_payload_view(attempt_dir / "transcript.jsonl")
            user_i = 0
            transcript_view: List[Dict[str, Any]] = []
            for r in transcript_rows:
                role = str(r.get("role") or "")
                text = str(r.get("text") or "")
                if role == "user" and user_i < len(user_turn_texts):
                    text = str(user_turn_texts[user_i])
                    user_i += 1
                transcript_view.append({"role": role, "text": text})

            ok_fc, reason_fc, details_fc = fluency_contract_v112(transcript_view=transcript_view)
            # Deterministic external world gating exercise: force probe on attempt 0 for the single allow_external task.
            if allow_external and (not ext_used) and a == 0:
                ok_fc = False
                reason_fc = "forced_external_world_probe"

            binding_events = _load_jsonl(attempt_dir / "binding_events.jsonl")
            unresolved_refs_total = _count_unresolved_reference_events(binding_events)
            flow_events = _load_jsonl(attempt_dir / "flow_events.jsonl")
            unresolved_refs_final = _unresolved_reference_final_from_flow(flow_events)
            semantic_events = _load_jsonl(attempt_dir / "semantic_events.jsonl")
            contradiction_flags = _count_semantic_contradiction_flags(semantic_events)

            ok_unresolved = unresolved_refs_final == 0
            ok_semantic = contradiction_flags == 0

            ok_attempt = bool(ok_gate) and bool(ok_unresolved) and bool(ok_semantic) and bool(ok_dialogue)
            if require_fluency:
                ok_attempt = bool(ok_attempt) and bool(ok_fc)

            attempts.append(
                {
                    "attempt_index": int(a),
                    "seed_used": int(seed_used),
                    "ok_gate_v115": bool(ok_gate),
                    "reason_gate_v115": str(gate_reason),
                    "ok_dialogue_survival_v116": bool(ok_dialogue),
                    "reason_dialogue_survival_v116": str(reason_dialogue),
                    "ok_fluency": bool(ok_fc),
                    "reason_fluency": str(summarize_fluency_fail_code_v112(str(reason_fc))),
                    "unresolved_reference_events_total": int(unresolved_refs_total),
                    "unresolved_reference_final": int(unresolved_refs_final),
                    "semantic_contradiction_flags_total": int(contradiction_flags),
                    "fluency_details": dict(details_fc),
                }
            )

            if allow_external and (not ext_used) and (not ok_fc) and world_manifest and str(probe_reason) in set(EXTERNAL_WORLD_REASON_CODES_V111):
                ext_events_final = _compute_external_world_access_once_v113(
                    world_manifest=str(world_manifest),
                    reason_code=str(probe_reason),
                    query="não invente",
                    seed=int(seed),
                )
                ext_used = True
                ext_used_reason = str(probe_reason)

            if ok_attempt:
                chosen_attempt = int(a)
                break

        _write_once_json(
            task_dir / "fluency_survival_v116.json",
            {
                "schema_version": 116,
                "task_id": str(task_id),
                "chosen_attempt_index": int(chosen_attempt),
                "attempts": list(attempts),
                "external_world_used": bool(ext_used),
                "external_world_reason_code": str(ext_used_reason),
            },
        )

        final_attempt_dir = task_dir / ("attempt_{a:03d}".format(a=chosen_attempt) if chosen_attempt >= 0 else "attempt_000")
        ext_info = _write_external_world_ledger(task_dir=final_attempt_dir, events=ext_events_final if ext_used else [])

        ok_task = bool(chosen_attempt >= 0)
        attempt_rel = "task_{i:03d}/attempt_{a:03d}".format(i=i, a=(chosen_attempt if chosen_attempt >= 0 else 0))

        attempt_row = next((x for x in attempts if isinstance(x, dict) and int(x.get("attempt_index", -1) or -1) == int(chosen_attempt)), None)
        if not isinstance(attempt_row, dict):
            attempt_row = attempts[-1] if attempts else {}

        ok_gate_final = bool(attempt_row.get("ok_gate_v115", False))
        gate_reason_final = str(attempt_row.get("reason_gate_v115") or "")
        ok_dialogue_final = bool(attempt_row.get("ok_dialogue_survival_v116", False))
        reason_dialogue_final = str(attempt_row.get("reason_dialogue_survival_v116") or "")
        ok_fluency_final = bool(attempt_row.get("ok_fluency", False))
        reason_fluency_final = str(attempt_row.get("reason_fluency") or "")
        unresolved_final_final = int(attempt_row.get("unresolved_reference_final", 0) or 0)
        contradiction_final = int(attempt_row.get("semantic_contradiction_flags_total", 0) or 0)

        ok_all = bool(ok_task) and bool(ok_gate_final) and bool(ok_dialogue_final) and (unresolved_final_final == 0) and (contradiction_final == 0)
        if require_fluency:
            ok_all = bool(ok_all) and bool(ok_fluency_final)

        results.append(
            {
                "task_id": str(task_id),
                "attempt_rel": str(attempt_rel),
                "seed_used": int(seed),
                "chosen_attempt_index": int(chosen_attempt),
                "ok_gate_v115": bool(ok_gate_final),
                "reason_gate_v115": str(gate_reason_final),
                "ok_dialogue_survival_v116": bool(ok_dialogue_final),
                "reason_dialogue_survival_v116": str(reason_dialogue_final),
                "ok_fluency": bool(ok_fluency_final),
                "reason_fluency": str(reason_fluency_final),
                "external_world_events_total": int(ext_info.get("events_total") or 0),
                "external_world_chain_hash_v111": str(ext_info.get("external_world_chain_hash_v111") or ""),
                "unresolved_reference_final": int(unresolved_final_final),
                "semantic_contradiction_flags_total": int(contradiction_final),
                "ok": bool(ok_all),
            }
        )

        if not bool(ok_all):
            reason = "unknown"
            if not ok_task:
                reason = "no_passing_attempt"
            elif not bool(ok_gate_final):
                reason = "gate_v115:" + str(gate_reason_final or "fail")
            elif not bool(ok_dialogue_final):
                reason = "dialogue_survival_v116:" + str(reason_dialogue_final or "fail")
            elif require_fluency and (not bool(ok_fluency_final)):
                reason = "fluency_v112:" + str(reason_fluency_final or "fail")
            elif unresolved_final_final != 0:
                reason = "unresolved_reference"
            elif contradiction_final != 0:
                reason = "semantic_contradiction"
            failures.append({"task_id": str(task_id), "attempt_rel": str(attempt_rel), "reason": str(reason), "validator": "family7_v116"})

        sha256_paths: Dict[str, str] = {
            "summary_json": str(final_attempt_dir / "summary.json"),
            "turns_jsonl": str(final_attempt_dir / "conversation_turns.jsonl"),
            "plans_jsonl": str(final_attempt_dir / "action_plans.jsonl"),
            "evals_jsonl": str(final_attempt_dir / "objective_evals.jsonl"),
            "final_response_v115_json": str(final_attempt_dir / "final_response_v115.json"),
            "final_response_v116_json": str(final_attempt_dir / "final_response_v116.json"),
            "dialogue_survival_summary_v116_json": str(final_attempt_dir / "dialogue_survival_summary_v116.json"),
            "goal_plan_eval_summary_v115_json": str(final_attempt_dir / "goal_plan_eval_summary_v115.json"),
            "goal_plan_eval_summary_v116_json": str(final_attempt_dir / "goal_plan_eval_summary_v116.json"),
            "goal_registry_snapshot_v115_json": str(final_attempt_dir / "goal_registry_snapshot_v115.json"),
            "mind_graph_v115_nodes_jsonl": str(final_attempt_dir / "mind_graph_v115" / "mind_nodes.jsonl"),
            "mind_graph_v115_edges_jsonl": str(final_attempt_dir / "mind_graph_v115" / "mind_edges.jsonl"),
            "mind_graph_v116_nodes_jsonl": str(final_attempt_dir / "mind_graph_v116" / "mind_nodes.jsonl"),
            "mind_graph_v116_edges_jsonl": str(final_attempt_dir / "mind_graph_v116" / "mind_edges.jsonl"),
            "fluency_survival_v116_json": str(task_dir / "fluency_survival_v116.json"),
            "external_world_events_jsonl": str(ext_info.get("external_world_events_jsonl") or ""),
            "external_world_registry_snapshot_json": str(ext_info.get("external_world_registry_snapshot_v111_json") or ""),
        }
        manifest = _compute_freeze_manifest_v116(task_dir=final_attempt_dir, sha256_paths=dict(sha256_paths))
        manifest_path = final_attempt_dir / "freeze_manifest_v116.json"
        _write_once_json(manifest_path, manifest)

    eval_obj = {
        "schema_version": 116,
        "kind": "family7_dla_eval_v116",
        "tasks_total": int(len(results)),
        "tasks_ok": int(sum(1 for r in results if isinstance(r, dict) and bool(r.get("ok", False)))),
        "results": list(results),
        "failures_total": int(len(failures)),
        "failures": list(failures),
    }
    _write_once_json(out_dir / "eval.json", dict(eval_obj))
    eval_sha256 = _sha256_file(out_dir / "eval.json")
    summary = {"schema_version": 116, "kind": "family7_dla_summary_v116", "tasks_total": int(len(results)), "tasks_ok": int(eval_obj.get("tasks_ok") or 0), "eval_sha256": str(eval_sha256)}
    _write_once_json(out_dir / "summary.json", dict(summary))
    _write_once_json(out_dir / "fail_catalog_v116.json", {"schema_version": 116, "failures_total": int(len(failures)), "failures": list(failures)})

    # Print final JSON for piping/tee (deterministic).
    print(json.dumps({"ok": True, "eval_sha256": str(eval_sha256), "tasks_ok": int(eval_obj.get("tasks_ok") or 0), "tasks_total": int(len(results))}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
