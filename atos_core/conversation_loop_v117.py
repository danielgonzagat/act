from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .ato_v71 import ATOv71, stable_hash_obj
from .conversation_loop_v116 import run_conversation_v116
from .goal_persistence_v115 import goal_id_v115, render_fail_response_v115
from .mind_graph_v71 import append_chained_jsonl, verify_chained_jsonl

FAIL_REASON_EXHAUSTED_PLANS_V117 = "exhausted_plans"
FAIL_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V117 = "plan_search_budget_exhausted"
FAIL_REASON_DUPLICATE_PLAN_CANDIDATE_V117 = "duplicate_plan_candidate"


def _ensure_absent(path: str) -> None:
    if os.path.exists(path):
        raise ValueError(f"worm_exists:{path}")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_once_json(path: str, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        raise ValueError(f"tmp_exists:{tmp}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _copy_file_worm(src: str, dst: str) -> None:
    _ensure_absent(dst)
    with open(src, "rb") as f:
        data = f.read()
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        raise ValueError(f"tmp_exists:{tmp}")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dst)


def _copy_tree_worm(src_dir: str, dst_dir: str) -> None:
    _ensure_absent(dst_dir)
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=False)


def _promote_attempt_to_root(*, attempt_dir: str, out_dir: str) -> None:
    """
    Copy a chosen attempt run_dir into the V117 out_dir root (WORM).
    This keeps V110/V115/V116 file layouts available at the top-level for downstream tooling.
    """
    ad = str(attempt_dir)
    od = str(out_dir)
    for name in sorted(os.listdir(ad), key=str):
        src = os.path.join(ad, name)
        dst = os.path.join(od, name)
        if os.path.isdir(src):
            # Do not copy nested attempt dirs (none expected), only regular directories (e.g., mind_graph_v116).
            if name.startswith("attempt_") or name.startswith("replan_attempt_"):
                continue
            _copy_tree_worm(src, dst)
        else:
            _copy_file_worm(src, dst)


def _extract_last_user_turn_payload(run_dir: str) -> Dict[str, Any]:
    rows = _read_jsonl(os.path.join(str(run_dir), "conversation_turns.jsonl"))
    users: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        if str(payload.get("role") or "") == "user":
            users.append(dict(payload))
    if not users:
        return {}
    users.sort(key=lambda p: (int(p.get("turn_index", 0) or 0), int(p.get("created_step", 0) or 0), str(p.get("turn_id") or "")))
    return dict(users[-1])


def _plan_row_for_user_turn(run_dir: str, user_turn_id: str) -> Dict[str, Any]:
    rows = _read_jsonl(os.path.join(str(run_dir), "action_plans.jsonl"))
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("user_turn_id") or "") == str(user_turn_id):
            return dict(row)
    return {}


def _expected_cost_for_act(plan_row: Dict[str, Any], act_id: str) -> float:
    ranked = plan_row.get("ranked_candidates") if isinstance(plan_row.get("ranked_candidates"), list) else []
    for r in ranked:
        if not isinstance(r, dict):
            continue
        if str(r.get("act_id") or "") == str(act_id):
            try:
                return float(r.get("expected_cost") or 0.0)
            except Exception:
                return 0.0
    return 0.0


def _plan_hash_from_row(plan_row: Dict[str, Any]) -> str:
    pid = str(plan_row.get("plan_id") or "")
    psig = str(plan_row.get("plan_sig") or "")
    if psig:
        return str(psig)
    if pid.startswith("action_plan_v96_"):
        return pid[len("action_plan_v96_") :]
    return sha256_hex(canonical_json_dumps(plan_row).encode("utf-8"))


def _make_fail_event_ato_v117(
    *,
    conversation_id: str,
    user_turn_id: str,
    goal_ato_id: str,
    plan_ato_id: str,
    reason_code: str,
    step: int,
    evidence: Dict[str, Any],
) -> ATOv71:
    body = {
        "schema_version": 117,
        "conversation_id": str(conversation_id),
        "user_turn_id": str(user_turn_id),
        "goal_ato_id": str(goal_ato_id),
        "plan_ato_id": str(plan_ato_id),
        "reason_code": str(reason_code),
        "evidence": dict(evidence) if isinstance(evidence, dict) else {},
    }
    fail_id = "fail_event_v117_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return ATOv71(
        ato_id=str(fail_id),
        ato_type="EVAL",
        subgraph=dict(body, satisfies=False),
        slots={},
        bindings={},
        cost=0.0,
        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}] if user_turn_id else [],
        invariants={"schema_version": 117, "eval_kind": "FAIL_EVENT_V117"},
        created_step=int(step),
        last_step=int(step),
    )


def _make_plan_ato_v117(*, plan_row: Dict[str, Any]) -> ATOv71:
    plan_id = str(plan_row.get("plan_id") or plan_row.get("plan_sig") or "")
    if not plan_id:
        raise ValueError("missing_plan_id")
    created_step = int(plan_row.get("created_step", 0) or 0)
    user_turn_id = str(plan_row.get("user_turn_id") or "")
    ranked = plan_row.get("ranked_candidates") if isinstance(plan_row.get("ranked_candidates"), list) else []
    attempted = plan_row.get("attempted_actions") if isinstance(plan_row.get("attempted_actions"), list) else []
    subgraph = {
        "schema_version": 117,
        "plan_id": str(plan_id),
        "user_turn_id": str(user_turn_id),
        "user_turn_index": int(plan_row.get("user_turn_index", -1) or -1),
        "chosen_action_id": str(plan_row.get("chosen_action_id") or ""),
        "chosen_eval_id": str(plan_row.get("chosen_eval_id") or ""),
        "chosen_ok": bool(plan_row.get("chosen_ok", False)),
        "ranked_candidates": [
            {
                "act_id": str(r.get("act_id") or ""),
                "expected_success": float(r.get("expected_success", 0.0) or 0.0),
                "expected_cost": float(r.get("expected_cost", 0.0) or 0.0),
            }
            for r in ranked
            if isinstance(r, dict) and str(r.get("act_id") or "")
        ],
        "attempted_actions": [
            {"act_id": str(a.get("act_id") or ""), "eval_id": str(a.get("eval_id") or ""), "ok": bool(a.get("ok", False))}
            for a in attempted
            if isinstance(a, dict) and str(a.get("act_id") or "")
        ],
    }
    # Deterministic ordering inside subgraph.
    subgraph["ranked_candidates"].sort(key=lambda d: (str(d.get("act_id") or "")))
    subgraph["attempted_actions"].sort(key=lambda d: (str(d.get("eval_id") or ""), str(d.get("act_id") or "")))
    return ATOv71(
        ato_id=str(plan_id),
        ato_type="PLAN",
        subgraph=dict(subgraph),
        slots={},
        bindings={},
        cost=float(len(subgraph.get("ranked_candidates") or [])),
        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}] if user_turn_id else [],
        invariants={"schema_version": 117, "plan_kind": "action_plan_v100"},
        created_step=int(created_step),
        last_step=int(created_step),
    )


def _last_entry_hash(path: str) -> str:
    last = ""
    if not os.path.exists(path):
        return last
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                last = str(obj.get("entry_hash") or last)
    return str(last)


def _append_fail_event_to_mind_graph_v117(
    *,
    mind_graph_v117_dir: str,
    fail_ato: ATOv71,
    goal_ato_id: str,
    plan_ato_id: str,
    user_turn_id: str,
    step: int,
    evidence_refs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    mg_dir = str(mind_graph_v117_dir)
    nodes_path = os.path.join(mg_dir, "mind_nodes.jsonl")
    edges_path = os.path.join(mg_dir, "mind_edges.jsonl")
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        raise ValueError("missing_mind_graph_v117_files")

    prev_nodes_hash = _last_entry_hash(nodes_path) or None
    fail_ato_dict = fail_ato.to_dict(include_sig=True)
    nodes_entry = {
        "time": deterministic_iso(step=int(step)),
        "step": int(step),
        "event": "NODE",
        "payload": {"reason": "fail_event_v117", "ato": dict(fail_ato_dict)},
    }
    new_nodes_hash = append_chained_jsonl(nodes_path, dict(nodes_entry), prev_hash=prev_nodes_hash)

    # Deterministic edges: FAIL -> GOAL, FAIL -> PLAN, FAIL -> TURN.
    prev_edges_hash = _last_entry_hash(edges_path) or None
    ev_refs_sorted = sorted([dict(x) for x in evidence_refs if isinstance(x, dict)], key=lambda d: canonical_json_dumps(d))

    def _mk_edge(dst: str) -> Dict[str, Any]:
        edge_sem_sig = {"src": str(fail_ato.ato_id), "dst": str(dst), "edge_type": "DERIVED_FROM", "evidence_refs": list(ev_refs_sorted)}
        return dict(edge_sem_sig, edge_sig=str(stable_hash_obj(edge_sem_sig)))

    dsts = [str(goal_ato_id), str(plan_ato_id), str(user_turn_id)]
    appended = 0
    for dst in sorted(set([d for d in dsts if d]), key=str):
        edge = _mk_edge(dst=str(dst))
        edge_entry = {"time": deterministic_iso(step=int(step)), "step": int(step), "event": "EDGE", "payload": {"reason": "fail_edges_v117", "edge": dict(edge)}}
        prev_edges_hash = append_chained_jsonl(edges_path, dict(edge_entry), prev_hash=prev_edges_hash)
        appended += 1

    return {"nodes_entry_hash": str(new_nodes_hash), "edges_appended_total": int(appended)}


@dataclass(frozen=True)
class AttemptRecordV117:
    attempt_index: int
    seed_used: int
    attempt_dir: str
    goal_id: str
    user_turn_id: str
    plan_id: str
    plan_hash: str
    plan_cost: float
    eval_satisfies: bool
    dialogue_survival_ok: bool
    fail_reason_code: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_index": int(self.attempt_index),
            "seed_used": int(self.seed_used),
            "attempt_dir": str(self.attempt_dir),
            "goal_id": str(self.goal_id),
            "user_turn_id": str(self.user_turn_id),
            "plan_id": str(self.plan_id),
            "plan_hash": str(self.plan_hash),
            "plan_cost": float(self.plan_cost),
            "eval_satisfies": bool(self.eval_satisfies),
            "dialogue_survival_ok": bool(self.dialogue_survival_ok),
            "fail_reason_code": str(self.fail_reason_code),
        }


def _replan_trace_obj_v117(
    *,
    conversation_id: str,
    attempts: List[AttemptRecordV117],
    chosen_attempt_index: int,
    budget_total: int,
    final_ok: bool,
    final_reason: str,
) -> Dict[str, Any]:
    body = {
        "schema_version": 117,
        "kind": "replan_trace_v117",
        "conversation_id": str(conversation_id),
        "budget_total": int(budget_total),
        "chosen_attempt_index": int(chosen_attempt_index),
        "final_ok": bool(final_ok),
        "final_reason": str(final_reason),
        "attempts": [a.to_dict() for a in list(attempts)],
    }
    body["trace_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return dict(body)


def run_conversation_v117(
    *,
    user_turn_texts: Sequence[str],
    out_dir: str,
    seed: int,
    max_plan_attempts: int = 8,
    max_replans_per_turn: int = 3,
) -> Dict[str, Any]:
    """
    V117 wrapper around V116 enforcing "replan until satisfies or exhaust/budget".

    Strategy (minimal, deterministic, WORM):
      - run up to `max_plan_attempts` full V116 attempts with seeds: seed+0, seed+1, ...
      - select the first attempt that yields final_response_v116.ok==true;
      - if none succeed: FAIL with reason plan_search_budget_exhausted (no proof of impossibility).

    The chosen attempt's artifacts are promoted (copied) into the V117 out_dir root (WORM),
    and a replan trace + mind_graph_v117 are written for auditability.
    """
    od = str(out_dir)
    _ensure_absent(od)
    os.makedirs(od, exist_ok=False)

    attempts: List[AttemptRecordV117] = []
    chosen_attempt = -1
    conversation_id_seen = ""

    for a in range(int(max_plan_attempts)):
        seed_used = int(seed) + int(a)
        attempt_dir = os.path.join(od, f"attempt_{a:03d}")
        conv = run_conversation_v116(
            user_turn_texts=list(user_turn_texts),
            out_dir=str(attempt_dir),
            seed=int(seed_used),
            max_replans_per_turn=int(max_replans_per_turn),
        )
        fr_path = os.path.join(str(attempt_dir), "final_response_v116.json")
        fr = _read_json(fr_path) if os.path.exists(fr_path) else {}
        ok_final = bool(fr.get("ok", False)) if isinstance(fr, dict) else False
        reason_final = str(fr.get("reason") or "") if isinstance(fr, dict) else "missing_final_response_v116"

        last_user = _extract_last_user_turn_payload(str(attempt_dir))
        user_turn_id = str(last_user.get("turn_id") or "")
        conversation_id = str(last_user.get("conversation_id") or str(conv.get("conversation_id") or ""))
        if conversation_id and not conversation_id_seen:
            conversation_id_seen = str(conversation_id)
        user_turn_index = int(last_user.get("turn_index", 0) or 0)
        refs = last_user.get("refs") if isinstance(last_user.get("refs"), dict) else {}
        parse_sig = str(refs.get("parse_sig") or "")
        user_text = str(last_user.get("text") or "")
        goal_id = goal_id_v115(
            conversation_id=str(conversation_id),
            user_turn_id=str(user_turn_id),
            user_turn_index=int(user_turn_index),
            parse_sig=str(parse_sig),
            user_text=str(user_text),
        )
        plan_row = _plan_row_for_user_turn(str(attempt_dir), user_turn_id)
        plan_id = str(plan_row.get("plan_id") or plan_row.get("plan_sig") or "")
        plan_hash = _plan_hash_from_row(dict(plan_row))
        chosen_action = str(plan_row.get("chosen_action_id") or "")
        plan_cost = _expected_cost_for_act(dict(plan_row), str(chosen_action)) if chosen_action else 0.0
        eval_satisfies = bool(fr.get("gate_v115_ok", False)) if isinstance(fr, dict) else False
        dialogue_ok = bool(fr.get("dialogue_survival_ok", False)) if isinstance(fr, dict) else False

        attempts.append(
            AttemptRecordV117(
                attempt_index=int(a),
                seed_used=int(seed_used),
                attempt_dir=os.path.basename(str(attempt_dir)),
                goal_id=str(goal_id),
                user_turn_id=str(user_turn_id),
                plan_id=str(plan_id),
                plan_hash=str(plan_hash),
                plan_cost=float(plan_cost),
                eval_satisfies=bool(eval_satisfies),
                dialogue_survival_ok=bool(dialogue_ok),
                fail_reason_code=str(reason_final) if not ok_final else "",
            )
        )

        if ok_final:
            chosen_attempt = int(a)
            break

    if not attempts:
        raise ValueError("no_attempts_v117")

    final_ok = chosen_attempt >= 0
    final_reason = "ok" if final_ok else FAIL_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V117
    chosen_idx = int(chosen_attempt if chosen_attempt >= 0 else (len(attempts) - 1))
    chosen_attempt_dirname = f"attempt_{chosen_idx:03d}"
    chosen_attempt_dir = os.path.join(od, chosen_attempt_dirname)

    # Promote chosen attempt artifacts to root (WORM).
    _promote_attempt_to_root(attempt_dir=str(chosen_attempt_dir), out_dir=str(od))

    # Write replanning trace (WORM).
    rep_obj = _replan_trace_obj_v117(
        conversation_id=str(conversation_id_seen),
        attempts=list(attempts),
        chosen_attempt_index=int(chosen_attempt if chosen_attempt >= 0 else -1),
        budget_total=int(max_plan_attempts),
        final_ok=bool(final_ok),
        final_reason=str(final_reason),
    )
    _write_once_json(os.path.join(od, "replan_trace_v117.json"), dict(rep_obj))

    # Create mind_graph_v117 by copying the promoted mind_graph_v116 and appending per-attempt FAIL_EVENT_V117 nodes/edges.
    mg116_dir = os.path.join(od, "mind_graph_v116")
    mg117_dir = os.path.join(od, "mind_graph_v117")
    if os.path.isdir(mg116_dir):
        os.makedirs(mg117_dir, exist_ok=False)
        _copy_file_worm(os.path.join(mg116_dir, "mind_nodes.jsonl"), os.path.join(mg117_dir, "mind_nodes.jsonl"))
        _copy_file_worm(os.path.join(mg116_dir, "mind_edges.jsonl"), os.path.join(mg117_dir, "mind_edges.jsonl"))

        # Append fail events for each failed attempt (including duplicates) deterministically by attempt_index.
        nodes_path = os.path.join(mg117_dir, "mind_nodes.jsonl")
        edges_path = os.path.join(mg117_dir, "mind_edges.jsonl")
        if not verify_chained_jsonl(nodes_path) or not verify_chained_jsonl(edges_path):
            raise ValueError("mind_graph_v117_chain_invalid_after_copy")

        # Ensure referenced nodes exist (goal + turn are from chosen attempt; plan nodes may differ).
        # We append plan nodes for failed attempts if missing.
        existing_nodes: Dict[str, Dict[str, Any]] = {}
        for row in _read_jsonl(nodes_path):
            payload = row.get("payload") if isinstance(row, dict) else None
            if not isinstance(payload, dict):
                continue
            ato = payload.get("ato")
            if not isinstance(ato, dict):
                continue
            ato_id = str(ato.get("ato_id") or "")
            if ato_id:
                existing_nodes[ato_id] = dict(ato)

        # Use chosen attempt's last user turn as the canonical turn reference.
        chosen_last_user = _extract_last_user_turn_payload(str(chosen_attempt_dir))
        chosen_user_turn_id = str(chosen_last_user.get("turn_id") or "")
        chosen_conversation_id = str(chosen_last_user.get("conversation_id") or "")
        chosen_user_turn_index = int(chosen_last_user.get("turn_index", 0) or 0)
        chosen_refs = chosen_last_user.get("refs") if isinstance(chosen_last_user.get("refs"), dict) else {}
        chosen_parse_sig = str(chosen_refs.get("parse_sig") or "")
        chosen_user_text = str(chosen_last_user.get("text") or "")
        chosen_goal_id = goal_id_v115(
            conversation_id=str(chosen_conversation_id),
            user_turn_id=str(chosen_user_turn_id),
            user_turn_index=int(chosen_user_turn_index),
            parse_sig=str(chosen_parse_sig),
            user_text=str(chosen_user_text),
        )
        # chosen plan id from chosen attempt plan row.
        chosen_plan_row = _plan_row_for_user_turn(str(chosen_attempt_dir), str(chosen_user_turn_id))
        chosen_plan_id = str(chosen_plan_row.get("plan_id") or chosen_plan_row.get("plan_sig") or "")

        for ar in list(attempts):
            if ar.attempt_index == chosen_attempt:
                continue
            if not ar.fail_reason_code:
                continue
            # Add plan node (from that attempt) if not present.
            plan_id = str(ar.plan_id or "")
            if plan_id and plan_id not in existing_nodes:
                plan_row = _plan_row_for_user_turn(os.path.join(od, str(ar.attempt_dir)), str(ar.user_turn_id))
                if isinstance(plan_row, dict) and plan_row:
                    plan_ato = _make_plan_ato_v117(plan_row=dict(plan_row))
                    prev_nodes_hash = _last_entry_hash(nodes_path) or None
                    plan_dict = plan_ato.to_dict(include_sig=True)
                    nodes_entry = {
                        "time": deterministic_iso(step=int(plan_ato.created_step)),
                        "step": int(plan_ato.created_step),
                        "event": "NODE",
                        "payload": {"reason": "plan_import_v117", "ato": dict(plan_dict)},
                    }
                    append_chained_jsonl(nodes_path, dict(nodes_entry), prev_hash=prev_nodes_hash)
                    existing_nodes[plan_id] = dict(plan_dict)

            # Append FAIL_EVENT_V117 node/edges.
            evidence = {
                "attempt_index": int(ar.attempt_index),
                "seed_used": int(ar.seed_used),
                "attempt_dir": str(ar.attempt_dir),
                "plan_hash": str(ar.plan_hash),
                "replan_trace_sig": str(rep_obj.get("trace_sig") or ""),
                "replan_trace_file": "replan_trace_v117.json",
            }
            step0 = int(chosen_last_user.get("created_step", 0) or 0)
            fail_ato = _make_fail_event_ato_v117(
                conversation_id=str(chosen_conversation_id),
                user_turn_id=str(chosen_user_turn_id),
                goal_ato_id=str(chosen_goal_id),
                plan_ato_id=str(plan_id or chosen_plan_id),
                reason_code=str(ar.fail_reason_code),
                step=int(step0),
                evidence=dict(evidence),
            )
            _append_fail_event_to_mind_graph_v117(
                mind_graph_v117_dir=str(mg117_dir),
                fail_ato=fail_ato,
                goal_ato_id=str(chosen_goal_id),
                plan_ato_id=str(plan_id or chosen_plan_id),
                user_turn_id=str(chosen_user_turn_id),
                step=int(step0),
                evidence_refs=[
                    {"kind": "turn", "turn_id": str(chosen_user_turn_id)},
                    {"kind": "replan_trace", "trace_sig": str(rep_obj.get("trace_sig") or "")},
                ],
            )

        if not verify_chained_jsonl(nodes_path) or not verify_chained_jsonl(edges_path):
            raise ValueError("mind_graph_v117_chain_invalid_after_append")

    # Write final_response_v117.json (WORM).
    fail_text = ""
    if not final_ok:
        fail_text = render_fail_response_v115(str(final_reason))
    fr117 = {
        "schema_version": 117,
        "kind": "final_response_v117",
        "ok": bool(final_ok),
        "reason": str(final_reason if not final_ok else "ok"),
        "fail_response_text": str(fail_text),
        "chosen_attempt_index": int(chosen_attempt if chosen_attempt >= 0 else -1),
        "budget_total": int(max_plan_attempts),
    }
    fr117["final_sig"] = sha256_hex(canonical_json_dumps(fr117).encode("utf-8"))
    _write_once_json(os.path.join(od, "final_response_v117.json"), dict(fr117))

    out: Dict[str, Any] = {"final_response_v117": dict(fr117), "replan_trace_v117_sig": str(rep_obj.get("trace_sig") or "")}
    out["attempts_total_v117"] = int(len(attempts))
    out["chosen_attempt_v117"] = int(chosen_attempt if chosen_attempt >= 0 else -1)
    out["gate_v117_ok"] = bool(final_ok)
    out["gate_v117_reason"] = str(final_reason)
    return dict(out)
