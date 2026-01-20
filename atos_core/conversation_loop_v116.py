from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .ato_v71 import ATOv71, stable_hash_obj
from .conversation_loop_v115 import run_conversation_v115
from .dialogue_survival_gate_v116 import DialogueSurvivalDecisionV116, compute_dialogue_survival_summary_v116
from .goal_persistence_v115 import goal_id_v115, render_fail_response_v115
from .mind_graph_v71 import append_chained_jsonl, verify_chained_jsonl


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
    with open(src, "rb") as fsrc:
        data = fsrc.read()
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        raise ValueError(f"tmp_exists:{tmp}")
    with open(tmp, "wb") as fdst:
        fdst.write(data)
    os.replace(tmp, dst)


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


def _sorted_dict_list(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs: List[Tuple[str, Dict[str, Any]]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            k = canonical_json_dumps(it)
        except Exception:
            k = str(it)
        pairs.append((k, dict(it)))
    pairs.sort(key=lambda kv: str(kv[0]))
    return [v for _, v in pairs]


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


def _make_fail_event_ato_v116(
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
        "schema_version": 116,
        "conversation_id": str(conversation_id),
        "user_turn_id": str(user_turn_id),
        "goal_ato_id": str(goal_ato_id),
        "plan_ato_id": str(plan_ato_id),
        "reason_code": str(reason_code),
        "evidence": dict(evidence) if isinstance(evidence, dict) else {},
    }
    fail_id = "fail_event_v116_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return ATOv71(
        ato_id=str(fail_id),
        ato_type="EVAL",
        subgraph=dict(body, satisfies=False),
        slots={},
        bindings={},
        cost=0.0,
        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}] if user_turn_id else [],
        invariants={"schema_version": 116, "eval_kind": "FAIL_EVENT_V116"},
        created_step=int(step),
        last_step=int(step),
    )


def _append_fail_event_to_copied_mind_graph_v116(
    *,
    run_dir: str,
    mind_graph_v116_dir: str,
    decision: DialogueSurvivalDecisionV116,
    reason_code: str,
) -> Dict[str, Any]:
    """
    Append a FAIL_EVENT_V116 node + edges into the *copied* mind_graph_v116 JSONLs, keeping chains valid.
    """
    rd = str(run_dir)
    mg_dir = str(mind_graph_v116_dir)
    nodes_path = os.path.join(mg_dir, "mind_nodes.jsonl")
    edges_path = os.path.join(mg_dir, "mind_edges.jsonl")
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        raise ValueError("missing_mind_graph_files_v116")

    last_user = _extract_last_user_turn_payload(rd)
    user_turn_id = str(last_user.get("turn_id") or "")
    if not user_turn_id:
        raise ValueError("missing_last_user_turn_id")
    conversation_id = str(last_user.get("conversation_id") or "")
    user_turn_index = int(last_user.get("turn_index", 0) or 0)
    user_text = str(last_user.get("text") or "")
    refs = last_user.get("refs") if isinstance(last_user.get("refs"), dict) else {}
    parse_sig = str(refs.get("parse_sig") or "")
    goal_ato_id = goal_id_v115(
        conversation_id=str(conversation_id),
        user_turn_id=str(user_turn_id),
        user_turn_index=int(user_turn_index),
        parse_sig=str(parse_sig),
        user_text=str(user_text),
    )
    plan_row = _plan_row_for_user_turn(rd, user_turn_id)
    plan_ato_id = str(plan_row.get("plan_id") or plan_row.get("plan_sig") or "")
    step = int(plan_row.get("created_step", last_user.get("created_step", 0) or 0) or 0)

    summary_path = os.path.join(rd, "dialogue_survival_summary_v116.json")
    summary_sig = ""
    if os.path.exists(summary_path):
        summary_obj = _read_json(summary_path)
        if isinstance(summary_obj, dict):
            summary_sig = str(summary_obj.get("summary_sig") or "")
    evidence = {
        "dialogue_survival_reason_code": str(reason_code),
        "dialogue_survival_summary_sig": str(summary_sig),
        "dialogue_survival_summary_file": os.path.basename(str(summary_path)),
        "dialogue_survival_details": dict(decision.details),
    }

    fail_ato = _make_fail_event_ato_v116(
        conversation_id=str(conversation_id),
        user_turn_id=str(user_turn_id),
        goal_ato_id=str(goal_ato_id),
        plan_ato_id=str(plan_ato_id),
        reason_code=str(reason_code),
        step=int(step),
        evidence=dict(evidence),
    )
    fail_ato_dict = fail_ato.to_dict(include_sig=True)

    prev_nodes_hash = _last_entry_hash(nodes_path) or None
    nodes_entry = {
        "time": deterministic_iso(step=int(step)),
        "step": int(step),
        "event": "NODE",
        "payload": {"reason": "fail_event_v116", "ato": dict(fail_ato_dict)},
    }
    new_nodes_hash = append_chained_jsonl(nodes_path, dict(nodes_entry), prev_hash=prev_nodes_hash)

    # Edges: FAIL -> GOAL, FAIL -> PLAN (if present), FAIL -> TURN.
    prev_edges_hash = _last_entry_hash(edges_path) or None
    edge_rows: List[Tuple[Dict[str, Any], str]] = []
    ev_refs = _sorted_dict_list(
        [
            {"kind": "turn", "turn_id": str(user_turn_id)},
            {"kind": "dialogue_survival", "summary_sig": str(summary_sig), "reason_code": str(reason_code)},
        ]
    )

    def _mk_edge(*, dst: str) -> Dict[str, Any]:
        edge_sem_sig = {"src": str(fail_ato.ato_id), "dst": str(dst), "edge_type": "DERIVED_FROM", "evidence_refs": list(ev_refs)}
        edge_sig = stable_hash_obj(edge_sem_sig)
        return dict(edge_sem_sig, edge_sig=str(edge_sig))

    dsts = [str(goal_ato_id)]
    if plan_ato_id:
        dsts.append(str(plan_ato_id))
    dsts.append(str(user_turn_id))

    # Deterministic order: dst id lex.
    for dst in sorted(set([d for d in dsts if d]), key=lambda s: str(s)):
        edge = _mk_edge(dst=str(dst))
        edge_entry = {
            "time": deterministic_iso(step=int(step)),
            "step": int(step),
            "event": "EDGE",
            "payload": {"reason": "fail_edges_v116", "edge": dict(edge)},
        }
        prev_edges_hash = append_chained_jsonl(edges_path, dict(edge_entry), prev_hash=prev_edges_hash)
        edge_rows.append((edge, str(prev_edges_hash or "")))

    return {
        "fail_ato_id": str(fail_ato.ato_id),
        "fail_ato_sig": str(fail_ato_dict.get("ato_sig") or ""),
        "new_nodes_entry_hash": str(new_nodes_hash),
        "edges_appended_total": int(len(edge_rows)),
    }


def _compute_mind_graph_sig_from_files(*, mind_graph_dir: str) -> str:
    mg_dir = str(mind_graph_dir)
    nodes_path = os.path.join(mg_dir, "mind_nodes.jsonl")
    edges_path = os.path.join(mg_dir, "mind_edges.jsonl")
    nodes: Dict[str, Dict[str, Any]] = {}
    edges_by_sig: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(nodes_path):
        payload = row.get("payload") if isinstance(row, dict) else None
        if not isinstance(payload, dict):
            continue
        ato = payload.get("ato")
        if not isinstance(ato, dict):
            continue
        ato_id = str(ato.get("ato_id") or "")
        if not ato_id:
            continue
        nodes[ato_id] = dict(ato)
    for row in _read_jsonl(edges_path):
        payload = row.get("payload") if isinstance(row, dict) else None
        if not isinstance(payload, dict):
            continue
        edge = payload.get("edge")
        if not isinstance(edge, dict):
            continue
        edge_sig = str(edge.get("edge_sig") or "")
        if not edge_sig:
            continue
        edges_by_sig[edge_sig] = dict(edge)
    nodes_list = [nodes[k] for k in sorted(nodes.keys())]
    edges_list = list(edges_by_sig.values())
    edges_list.sort(
        key=lambda e: (
            str(e.get("src") or ""),
            str(e.get("dst") or ""),
            str(e.get("edge_type") or ""),
            str(e.get("edge_sig") or ""),
        )
    )
    snap = {"schema_version": 1, "nodes": list(nodes_list), "edges": list(edges_list)}
    return sha256_hex(canonical_json_dumps(snap).encode("utf-8"))


@dataclass(frozen=True)
class ApplyResultV116:
    final_ok: bool
    reason: str
    dialogue_survival_ok: bool
    dialogue_survival_reason: str
    details: Dict[str, Any]


def apply_dialogue_survival_as_law_v116(
    *,
    run_dir: str,
    write_mind_graph: bool = True,
) -> ApplyResultV116:
    """
    Apply V116 "dialogue survival as law" on top of an existing V115 run_dir.
    Writes (WORM):
      - dialogue_survival_summary_v116.json
      - mind_graph_v116/ (copied from mind_graph_v115 + optional FAIL_EVENT_V116)
      - goal_plan_eval_summary_v116.json
      - final_response_v116.json
    """
    rd = str(run_dir)
    # 1) Compute and persist dialogue survival summary (WORM).
    decision = compute_dialogue_survival_summary_v116(run_dir=rd, write_summary=True)

    # 2) Read V115 final response (must exist in a V115 run_dir).
    fr115_path = os.path.join(rd, "final_response_v115.json")
    if not os.path.exists(fr115_path):
        raise ValueError("missing_final_response_v115")
    fr115 = _read_json(fr115_path)
    gate_ok = bool(fr115.get("ok", False)) if isinstance(fr115, dict) else False
    gate_reason = str(fr115.get("reason") or "") if isinstance(fr115, dict) else "missing_final_response_v115"

    # 3) Prepare mind_graph_v116 (copy) and optionally append FAIL_EVENT_V116.
    mg115_dir = os.path.join(rd, "mind_graph_v115")
    mg116_dir = os.path.join(rd, "mind_graph_v116")
    if write_mind_graph:
        if not os.path.isdir(mg115_dir):
            raise ValueError("missing_mind_graph_v115")
        _ensure_absent(mg116_dir)
        os.makedirs(mg116_dir, exist_ok=False)
        _copy_file_worm(os.path.join(mg115_dir, "mind_nodes.jsonl"), os.path.join(mg116_dir, "mind_nodes.jsonl"))
        _copy_file_worm(os.path.join(mg115_dir, "mind_edges.jsonl"), os.path.join(mg116_dir, "mind_edges.jsonl"))

    fail_append_info: Dict[str, Any] = {}
    final_ok = bool(gate_ok) and bool(decision.ok)
    final_reason = "ok"
    if not bool(gate_ok):
        final_ok = False
        final_reason = str(gate_reason or "gate_v115_failed")
    elif not bool(decision.ok):
        final_ok = False
        final_reason = str(decision.reason_code)
        # Add a FAIL_EVENT only for dialogue-survival failures (law extension).
        if write_mind_graph:
            fail_append_info = _append_fail_event_to_copied_mind_graph_v116(
                run_dir=rd,
                mind_graph_v116_dir=str(mg116_dir),
                decision=decision,
                reason_code=str(final_reason),
            )

    # 4) Verify mind graph chains and compute sig (best-effort; fail-closed if chain broken).
    mg_ok = {}
    mg_sig = ""
    if write_mind_graph:
        nodes_ok = bool(verify_chained_jsonl(os.path.join(mg116_dir, "mind_nodes.jsonl")))
        edges_ok = bool(verify_chained_jsonl(os.path.join(mg116_dir, "mind_edges.jsonl")))
        mg_ok = {"mind_nodes_chain_ok": bool(nodes_ok), "mind_edges_chain_ok": bool(edges_ok)}
        if not nodes_ok or not edges_ok:
            raise ValueError("mind_graph_v116_chain_invalid")
        mg_sig = _compute_mind_graph_sig_from_files(mind_graph_dir=str(mg116_dir))

    # 5) Write goal_plan_eval_summary_v116.json (WORM).
    gp115_path = os.path.join(rd, "goal_plan_eval_summary_v115.json")
    gp115 = _read_json(gp115_path) if os.path.exists(gp115_path) else {}
    gp116 = {
        "schema_version": 116,
        "kind": "goal_plan_eval_summary_v116",
        "gate_v115": dict(gp115) if isinstance(gp115, dict) else {},
        "dialogue_survival_v116": {
            "ok": bool(decision.ok),
            "reason_code": str(decision.reason_code),
            "details": dict(decision.details),
        },
        "mind_graph_v116": dict(mg_ok, mind_graph_sig=str(mg_sig)) if write_mind_graph else {},
        "fail_event_v116": dict(fail_append_info),
    }
    gp116["summary_sig"] = sha256_hex(canonical_json_dumps(gp116).encode("utf-8"))
    _write_once_json(os.path.join(rd, "goal_plan_eval_summary_v116.json"), dict(gp116))

    # 6) Write final_response_v116.json (WORM).
    fail_text = ""
    if not bool(final_ok):
        fail_text = render_fail_response_v115(str(final_reason))
    fr116 = {
        "schema_version": 116,
        "kind": "final_response_v116",
        "ok": bool(final_ok),
        "reason": str(final_reason if not final_ok else "ok"),
        "fail_response_text": str(fail_text),
        "gate_v115_ok": bool(gate_ok),
        "gate_v115_reason": str(gate_reason),
        "dialogue_survival_ok": bool(decision.ok),
        "dialogue_survival_reason": str(decision.reason_code),
    }
    fr116["final_sig"] = sha256_hex(canonical_json_dumps(fr116).encode("utf-8"))
    _write_once_json(os.path.join(rd, "final_response_v116.json"), dict(fr116))

    return ApplyResultV116(
        final_ok=bool(final_ok),
        reason=str(final_reason),
        dialogue_survival_ok=bool(decision.ok),
        dialogue_survival_reason=str(decision.reason_code),
        details={"final_response_v116": dict(fr116), "goal_plan_eval_summary_v116_sig": str(gp116.get("summary_sig") or "")},
    )


def run_conversation_v116(
    *,
    user_turn_texts: Sequence[str],
    out_dir: str,
    seed: int,
    max_replans_per_turn: int = 3,
) -> Dict[str, Any]:
    """
    V116 wrapper: extends V115 runtime-gate by making dialogue survival a law.
    Authoritative output for the run is final_response_v116.json.
    """
    res = run_conversation_v115(user_turn_texts=list(user_turn_texts), out_dir=str(out_dir), seed=int(seed), max_replans_per_turn=int(max_replans_per_turn))
    applied = apply_dialogue_survival_as_law_v116(run_dir=str(out_dir), write_mind_graph=True)
    out = dict(res)
    out["final_response_v116_ok"] = bool(applied.final_ok)
    out["final_response_v116_reason"] = str(applied.reason)
    out["dialogue_survival_v116_ok"] = bool(applied.dialogue_survival_ok)
    out["dialogue_survival_v116_reason"] = str(applied.dialogue_survival_reason)
    out["final_response_v116"] = dict(applied.details.get("final_response_v116") or {})
    return out

