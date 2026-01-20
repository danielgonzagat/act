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
from atos_core.conversation_loop_v121 import run_conversation_v121
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


def _compute_freeze_manifest_v121(*, task_dir: Path, sha256_paths: Dict[str, str]) -> Dict[str, Any]:
    sha256: Dict[str, str] = {}
    sha256_rel: Dict[str, str] = {}
    for k, p in sorted(sha256_paths.items(), key=lambda kv: str(kv[0])):
        pp = str(p or "")
        sha256_rel[str(k)] = str(os.path.basename(pp)) if pp else ""
        if pp and os.path.exists(pp):
            sha256[str(k)] = _sha256_file(Path(pp))
    body = {
        "schema_version": 121,
        "kind": "freeze_manifest_v121",
        "task_dir": str(task_dir.name),
        "sha256": dict(sha256),
        "sha256_paths": dict(sha256_rel),
    }
    body["manifest_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    path = task_dir / "freeze_manifest_v121.json"
    _write_once_json(path, body)
    return dict(body)


def _summarize_task_result_v121(*, task_dir: Path, attempts: Sequence[Dict[str, Any]], chosen_attempt: int) -> Dict[str, Any]:
    chosen_dir = task_dir / "attempt_{a:03d}".format(a=chosen_attempt)
    final_path = chosen_dir / "final_response_v121.json"
    fr = _load_json(final_path) if final_path.exists() else {}
    ok = bool(fr.get("ok", False)) if isinstance(fr, dict) else False
    reason = str(fr.get("reason") or "") if isinstance(fr, dict) else "missing_final_response_v121"
    rel_final_path = str(Path("attempt_{a:03d}".format(a=chosen_attempt)) / "final_response_v121.json")
    return {
        "ok": bool(ok),
        "reason": str(reason),
        "chosen_attempt_index": int(chosen_attempt),
        "attempts_total": int(len(list(attempts))),
        "attempt_final_response_v121": str(rel_final_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--max_tasks", type=int, default=9999)
    ap.add_argument("--max_rewrites", type=int, default=4)
    ap.add_argument("--max_replans_per_turn", type=int, default=3)
    ap.add_argument("--max_plan_attempts", type=int, default=8)
    args = ap.parse_args()

    seed = int(args.seed)
    tasks_path = str(args.tasks)
    out_dir = Path(str(args.out))
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    tasks: List[Dict[str, Any]] = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    if not tasks:
        _fail("empty_tasks")

    max_tasks = min(int(args.max_tasks), len(tasks))
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks[:max_tasks]):
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or f"task_{i:03d}")
        user_turns = task.get("user_turns") if isinstance(task.get("user_turns"), list) else []
        user_turn_texts = [str(x) for x in user_turns if isinstance(x, str)]
        user_turn_texts = _choiceify_minimal_ack_v112(user_turn_texts)
        require_fluency = "fluency_survival_v112" in [str(x) for x in (task.get("expected_validators") if isinstance(task.get("expected_validators"), list) else [])]
        allow_external = bool(task.get("allow_external_world_once", False))
        world_manifest = str(task.get("world_manifest") or "")
        probe_reason = str(task.get("external_world_probe_reason_code") or "validator_failed_fluency_contract")

        task_dir = out_dir / "task_{i:03d}".format(i=i)
        _ensure_absent(task_dir)
        task_dir.mkdir(parents=True, exist_ok=False)

        if allow_external and probe_reason and str(probe_reason) not in set(EXTERNAL_WORLD_REASON_CODES_V111):
            _fail(f"invalid_reason_code_in_task:{probe_reason}")

        attempts: List[Dict[str, Any]] = []
        chosen_attempt = -1

        rewrite_seeds = fluency_survival_plan_v112(base_seed=int(seed), max_attempts=int(args.max_rewrites))
        ext_used = False
        ext_used_reason = ""
        ext_events_final: List[Dict[str, Any]] = []

        for a, seed_used in enumerate(rewrite_seeds):
            attempt_dir = task_dir / "attempt_{a:03d}".format(a=a)
            _ensure_absent(attempt_dir)

            run_conversation_v121(
                user_turn_texts=list(user_turn_texts),
                out_dir=str(attempt_dir),
                seed=int(seed_used),
                max_replans_per_turn=int(args.max_replans_per_turn),
                max_plan_attempts=int(args.max_plan_attempts),
            )

            fr115 = _load_json(attempt_dir / "final_response_v115.json")
            fr116 = _load_json(attempt_dir / "final_response_v116.json")
            fr121 = _load_json(attempt_dir / "final_response_v121.json")

            ok_gate = bool(fr115.get("ok", False)) if isinstance(fr115, dict) else False
            gate_reason = str(fr115.get("reason") or "") if isinstance(fr115, dict) else "missing_final_response_v115"
            ok_dialogue = bool(fr116.get("dialogue_survival_ok", False)) if isinstance(fr116, dict) else False
            reason_dialogue = str(fr116.get("dialogue_survival_reason") or "") if isinstance(fr116, dict) else "missing_final_response_v116"
            ok_final = bool(fr121.get("ok", False)) if isinstance(fr121, dict) else False
            reason_final = str(fr121.get("reason") or "") if isinstance(fr121, dict) else "missing_final_response_v121"

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

            ok_attempt = bool(ok_gate) and bool(ok_unresolved) and bool(ok_semantic) and bool(ok_dialogue) and bool(ok_final)
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
                    "ok_final_v121": bool(ok_final),
                    "reason_final_v121": str(reason_final),
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
                    query=str(task_id),
                    seed=int(seed_used),
                )
                ext_used = True
                ext_used_reason = str(probe_reason)

            if ok_attempt:
                chosen_attempt = int(a)
                break

        if chosen_attempt < 0:
            chosen_attempt = int(len(attempts) - 1) if attempts else 0

        ext_info = _write_external_world_ledger(task_dir=task_dir, events=list(ext_events_final))

        summary = _summarize_task_result_v121(task_dir=task_dir, attempts=list(attempts), chosen_attempt=int(chosen_attempt))
        ok_task = bool(summary.get("ok", False))
        result_row = {
            "task_id": str(task_id),
            "task_index": int(i),
            "ok": bool(ok_task),
            "chosen_attempt": int(chosen_attempt),
            "attempts": list(attempts),
            "external_world_events_total": int(ext_info.get("events_total", 0)),
            "external_world_used_reason": str(ext_used_reason),
            "final": dict(summary),
        }
        result_row["row_sig"] = sha256_hex(canonical_json_dumps(result_row).encode("utf-8"))
        results.append(dict(result_row))

        if not ok_task:
            failures.append({"task_id": str(task_id), "task_index": int(i), "reason": str(summary.get("reason") or ""), "chosen_attempt": int(chosen_attempt)})

        # Minimal freeze manifest per task (WORM).
        sha256_paths = {
            "final_response_v121_json": str((task_dir / "attempt_{a:03d}".format(a=int(chosen_attempt)) / "final_response_v121.json")),
            "external_world_events_jsonl": str(task_dir / "external_world_events.jsonl"),
        }
        _compute_freeze_manifest_v121(task_dir=task_dir, sha256_paths=dict(sha256_paths))

    tasks_total = int(len(results))
    tasks_ok = int(sum(1 for r in results if isinstance(r, dict) and bool(r.get("ok", False))))
    eval_obj = {
        "schema_version": 121,
        "kind": "family7_dla_eval_v121",
        "tasks_total": int(tasks_total),
        "tasks_ok": int(tasks_ok),
        "tasks_failed": int(tasks_total - tasks_ok),
        "results": list(results),
        "failures": list(failures),
    }
    eval_obj["eval_sig"] = sha256_hex(canonical_json_dumps(eval_obj).encode("utf-8"))
    _write_once_json(out_dir / "eval.json", eval_obj)

    summary = {
        "schema_version": 121,
        "kind": "family7_dla_summary_v121",
        "tasks_total": int(tasks_total),
        "tasks_ok": int(tasks_ok),
        "eval_sha256": _sha256_file(out_dir / "eval.json"),
    }
    summary["summary_sig"] = sha256_hex(canonical_json_dumps(summary).encode("utf-8"))
    _write_once_json(out_dir / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
