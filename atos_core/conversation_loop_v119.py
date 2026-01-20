from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .ato_v71 import ATOv71, stable_hash_obj
from .conversation_loop_v110 import run_conversation_v110
from .conversation_loop_v116 import apply_dialogue_survival_as_law_v116
from .conversation_v96 import verify_chained_jsonl_v96
from .fluency_contract_v118 import FluencyContractResultV118, fluency_contract_v118
from .goal_persistence_v115 import render_fail_response_v115
from .goal_plan_eval_gate_v115 import goal_id_v115, verify_goal_plan_eval_law_v115
from .mind_graph_v71 import append_chained_jsonl, verify_chained_jsonl


FAIL_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V119 = "plan_search_budget_exhausted_v119"


def _ensure_absent(path: str) -> None:
    if os.path.exists(path):
        raise ValueError(f"worm_exists:{path}")
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        raise ValueError(f"tmp_exists:{tmp}")


def _write_once_json(path: str, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    Copy a chosen attempt run_dir into the V119 out_dir root (WORM).
    This keeps V110/V115/V116 file layouts available at the top-level for downstream tooling.
    """
    ad = str(attempt_dir)
    od = str(out_dir)
    for name in sorted(os.listdir(ad), key=str):
        src = os.path.join(ad, name)
        dst = os.path.join(od, name)
        if os.path.isdir(src):
            # Do not copy nested attempt dirs (we only keep the top-level chosen attempt artifacts).
            if name.startswith("attempt_") or name.startswith("replan_attempt_"):
                continue
            _copy_tree_worm(src, dst)
        else:
            _copy_file_worm(src, dst)


def _extract_last_user_turn_payload(run_dir: str) -> Dict[str, Any]:
    rows = _read_jsonl(os.path.join(str(run_dir), "conversation_turns.jsonl"))
    users: List[Dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload") if isinstance(row, dict) else None
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


def _make_plan_ato_v119(*, action_plan: Dict[str, Any]) -> ATOv71:
    plan_id = str(action_plan.get("plan_id") or action_plan.get("plan_sig") or "")
    if not plan_id:
        raise ValueError("missing_plan_id")
    created_step = int(action_plan.get("created_step", 0) or 0)
    user_turn_id = str(action_plan.get("user_turn_id") or "")
    user_turn_index = int(action_plan.get("user_turn_index", -1) or -1)
    ranked = action_plan.get("ranked_candidates") if isinstance(action_plan.get("ranked_candidates"), list) else []
    attempted = action_plan.get("attempted_actions") if isinstance(action_plan.get("attempted_actions"), list) else []
    subgraph = {
        "schema_version": 119,
        "plan_id": str(plan_id),
        "user_turn_id": str(user_turn_id),
        "user_turn_index": int(user_turn_index),
        "chosen_action_id": str(action_plan.get("chosen_action_id") or ""),
        "chosen_eval_id": str(action_plan.get("chosen_eval_id") or ""),
        "chosen_ok": bool(action_plan.get("chosen_ok", False)),
        "ranked_candidates": [
            {
                "act_id": str(r.get("act_id") or ""),
                "expected_success": float(r.get("expected_success", 0.0) or 0.0),
                "expected_cost": float(r.get("expected_cost", 0.0) or 0.0),
            }
            for r in ranked
            if isinstance(r, dict)
        ],
        "attempted_actions": [
            {"act_id": str(a.get("act_id") or ""), "eval_id": str(a.get("eval_id") or ""), "ok": bool(a.get("ok", False))}
            for a in attempted
            if isinstance(a, dict)
        ],
    }
    # Canonicalize deterministic ordering.
    subgraph["ranked_candidates"].sort(key=lambda d: str(d.get("act_id") or ""))
    subgraph["attempted_actions"].sort(key=lambda d: (str(d.get("eval_id") or ""), str(d.get("act_id") or "")))
    return ATOv71(
        ato_id=str(plan_id),
        ato_type="PLAN",
        subgraph=dict(subgraph),
        slots={},
        bindings={},
        cost=float(len(subgraph["ranked_candidates"])),
        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}] if user_turn_id else [],
        invariants={"schema_version": 119, "plan_kind": "action_plan_v100"},
        created_step=int(created_step),
        last_step=int(created_step),
    )


def _make_fail_event_ato_v119(
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
        "schema_version": 119,
        "conversation_id": str(conversation_id),
        "user_turn_id": str(user_turn_id),
        "goal_ato_id": str(goal_ato_id),
        "plan_ato_id": str(plan_ato_id),
        "reason_code": str(reason_code),
        "evidence": dict(evidence) if isinstance(evidence, dict) else {},
    }
    fail_id = "fail_event_v119_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return ATOv71(
        ato_id=str(fail_id),
        ato_type="EVAL",
        subgraph=dict(body, satisfies=False),
        slots={},
        bindings={},
        cost=0.0,
        evidence_refs=[{"kind": "turn", "turn_id": str(user_turn_id)}] if user_turn_id else [],
        invariants={"schema_version": 119, "eval_kind": "FAIL_EVENT_V119"},
        created_step=int(step),
        last_step=int(step),
    )


def _append_fail_event_to_mind_graph_v119(
    *,
    mind_graph_v119_dir: str,
    fail_ato: ATOv71,
    goal_ato_id: str,
    plan_ato_id: str,
    user_turn_id: str,
    step: int,
    evidence_refs: Sequence[Dict[str, Any]],
) -> None:
    mg_dir = str(mind_graph_v119_dir)
    nodes_path = os.path.join(mg_dir, "mind_nodes.jsonl")
    edges_path = os.path.join(mg_dir, "mind_edges.jsonl")
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        raise ValueError("missing_mind_graph_files_v119")

    fail_ato_dict = fail_ato.to_dict(include_sig=True)
    prev_nodes_hash = _last_entry_hash(nodes_path) or None
    nodes_entry = {
        "time": deterministic_iso(step=int(step)),
        "step": int(step),
        "event": "NODE",
        "payload": {"reason": "fail_event_v119", "ato": dict(fail_ato_dict)},
    }
    append_chained_jsonl(nodes_path, dict(nodes_entry), prev_hash=prev_nodes_hash)

    ev_refs = _sorted_dict_list([x for x in evidence_refs if isinstance(x, dict)])

    def _mk_edge(dst: str) -> Dict[str, Any]:
        edge_sem_sig = {"src": str(fail_ato.ato_id), "dst": str(dst), "edge_type": "DERIVED_FROM", "evidence_refs": list(ev_refs)}
        edge_sig = stable_hash_obj(edge_sem_sig)
        return dict(edge_sem_sig, edge_sig=str(edge_sig))

    dsts = [str(goal_ato_id), str(plan_ato_id), str(user_turn_id)]
    dsts = [d for d in dsts if d]
    for dst in dsts:
        prev_edges_hash = _last_entry_hash(edges_path) or None
        edges_entry = {
            "time": deterministic_iso(step=int(step)),
            "step": int(step),
            "event": "EDGE",
            "payload": {"reason": "fail_event_v119", "edge": _mk_edge(dst)},
        }
        append_chained_jsonl(edges_path, dict(edges_entry), prev_hash=prev_edges_hash)


def _transcript_view_from_run_dir(run_dir: str) -> List[Dict[str, Any]]:
    rows = _read_jsonl(os.path.join(str(run_dir), "transcript.jsonl"))
    out: List[Dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload") if isinstance(row, dict) else None
        if not isinstance(payload, dict):
            continue
        role = str(payload.get("role") or "")
        text = str(payload.get("text") or "")
        if role in {"user", "assistant"}:
            out.append({"role": role, "text": text})
    return out


@dataclass(frozen=True)
class AttemptRecordV119:
    attempt_index: int
    seed_used: int
    attempt_dir: str
    goal_id: str
    user_turn_id: str
    ok_final_v116: bool
    reason_final_v116: str
    ok_fluency: bool
    reason_fluency: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_index": int(self.attempt_index),
            "seed_used": int(self.seed_used),
            "attempt_dir": str(self.attempt_dir),
            "goal_id": str(self.goal_id),
            "user_turn_id": str(self.user_turn_id),
            "ok_final_v116": bool(self.ok_final_v116),
            "reason_final_v116": str(self.reason_final_v116),
            "ok_fluency": bool(self.ok_fluency),
            "reason_fluency": str(self.reason_fluency),
        }


def _replan_trace_obj_v119(
    *,
    conversation_id: str,
    attempts: List[AttemptRecordV119],
    chosen_attempt_index: int,
    budget_total: int,
    final_ok: bool,
    final_reason: str,
) -> Dict[str, Any]:
    body = {
        "schema_version": 119,
        "kind": "replan_trace_v119",
        "conversation_id": str(conversation_id),
        "budget_total": int(budget_total),
        "chosen_attempt_index": int(chosen_attempt_index),
        "final_ok": bool(final_ok),
        "final_reason": str(final_reason),
        "attempts": [a.to_dict() for a in list(attempts)],
    }
    body["trace_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return dict(body)


def run_conversation_v119(
    *,
    user_turn_texts: Sequence[str],
    out_dir: str,
    seed: int,
    max_plan_attempts: int = 8,
    max_replans_per_turn: int = 3,
) -> Dict[str, Any]:
    """
    V119: make fluency_contract_v118 a hard gate *inside* the replanning loop.

    Strategy (deterministic, WORM):
      - For each attempt seed (seed+0, seed+1, ...):
          * run base V110 pipeline with discourse_variants_v119 enabled;
          * apply V115 goal-plan-eval law (writes mind_graph_v115 + final_response_v115.json);
          * apply V116 dialogue survival as law (writes mind_graph_v116 + final_response_v116.json);
          * compute fluency_contract_v118 on transcript.jsonl and persist fluency_summary_v119.json;
          * accept the first attempt where (final_response_v116.ok==true AND fluency_ok==true).
      - If none succeed within budget: FAIL with reason plan_search_budget_exhausted_v119.

    The chosen attempt's artifacts are promoted (copied) into the V119 out_dir root (WORM),
    and a replanning trace is written for auditability.
    """
    od = str(out_dir)
    _ensure_absent(od)
    os.makedirs(od, exist_ok=False)

    attempts: List[AttemptRecordV119] = []
    chosen_attempt = -1
    conversation_id_seen = ""

    for a in range(int(max_plan_attempts)):
        seed_used = int(seed) + int(a)
        attempt_dir = os.path.join(od, f"attempt_{a:03d}")

        # 1) Run base deterministic pipeline (V110) with discourse variants enabled (V119).
        run_conversation_v110(
            user_turn_texts=list(user_turn_texts),
            out_dir=str(attempt_dir),
            seed=int(seed_used),
            discourse_variants_v119_enabled=True,
        )

        # 2) Apply V115 goal-plan-eval law (runtime gate) and write final_response_v115.json (WORM).
        gate = verify_goal_plan_eval_law_v115(
            run_dir=str(attempt_dir),
            max_replans_per_turn=int(max_replans_per_turn),
            write_mind_graph=True,
            write_snapshots=True,
        )
        fr115 = {
            "schema_version": 115,
            "kind": "final_response_v115",
            "ok": bool(gate.ok),
            "reason": str(gate.reason if not bool(gate.ok) else "ok"),
            "fail_response_text": str(render_fail_response_v115(str(gate.reason))) if not bool(gate.ok) else "",
        }
        fr115["final_sig"] = sha256_hex(canonical_json_dumps(fr115).encode("utf-8"))
        _write_once_json(os.path.join(str(attempt_dir), "final_response_v115.json"), dict(fr115))

        # 3) Apply V116 dialogue survival law on top of V115.
        applied = apply_dialogue_survival_as_law_v116(run_dir=str(attempt_dir), write_mind_graph=True)
        fr116_path = os.path.join(str(attempt_dir), "final_response_v116.json")
        fr116 = _read_json(fr116_path) if os.path.exists(fr116_path) else {}
        ok_final = bool(fr116.get("ok", False)) if isinstance(fr116, dict) else False
        reason_final = str(fr116.get("reason") or "") if isinstance(fr116, dict) else "missing_final_response_v116"

        # 4) Fluency as survival (hard gate).
        transcript_view = _transcript_view_from_run_dir(str(attempt_dir))
        ok_flu, reason_flu, details_flu = fluency_contract_v118(transcript_view=list(transcript_view))
        flu_obj = FluencyContractResultV118(ok=bool(ok_flu), reason=str(reason_flu), details=dict(details_flu)).to_dict()
        _write_once_json(os.path.join(str(attempt_dir), "fluency_summary_v119.json"), dict(flu_obj))

        last_user = _extract_last_user_turn_payload(str(attempt_dir))
        user_turn_id = str(last_user.get("turn_id") or "")
        conversation_id = str(last_user.get("conversation_id") or str(applied.details.get("conversation_id") or ""))
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

        attempts.append(
            AttemptRecordV119(
                attempt_index=int(a),
                seed_used=int(seed_used),
                attempt_dir=os.path.basename(str(attempt_dir)),
                goal_id=str(goal_id),
                user_turn_id=str(user_turn_id),
                ok_final_v116=bool(ok_final),
                reason_final_v116=str(reason_final),
                ok_fluency=bool(ok_flu),
                reason_fluency=str(reason_flu) if not bool(ok_flu) else "ok",
            )
        )

        if bool(ok_final) and bool(ok_flu):
            chosen_attempt = int(a)
            break

    if not attempts:
        raise ValueError("no_attempts_v119")

    final_ok = chosen_attempt >= 0
    final_reason = "ok" if final_ok else FAIL_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V119
    chosen_idx = int(chosen_attempt if chosen_attempt >= 0 else (len(attempts) - 1))
    chosen_attempt_dir = os.path.join(od, f"attempt_{chosen_idx:03d}")

    # Promote chosen attempt artifacts to root (WORM).
    _promote_attempt_to_root(attempt_dir=str(chosen_attempt_dir), out_dir=str(od))

    # Write replanning trace (WORM).
    rep_obj = _replan_trace_obj_v119(
        conversation_id=str(conversation_id_seen),
        attempts=list(attempts),
        chosen_attempt_index=int(chosen_attempt if chosen_attempt >= 0 else -1),
        budget_total=int(max_plan_attempts),
        final_ok=bool(final_ok),
        final_reason=str(final_reason),
    )
    _write_once_json(os.path.join(od, "replan_trace_v119.json"), dict(rep_obj))

    # Write final_response_v119.json (WORM).
    fr116_root_path = os.path.join(od, "final_response_v116.json")
    fr116_root = _read_json(fr116_root_path) if os.path.exists(fr116_root_path) else {}
    final_obj = {
        "schema_version": 119,
        "kind": "final_response_v119",
        "ok": bool(final_ok),
        "reason": str(final_reason if not final_ok else "ok"),
        "fail_response_text": str(render_fail_response_v115(str(final_reason))) if not bool(final_ok) else "",
        "upstream": {"final_response_v116": dict(fr116_root) if isinstance(fr116_root, dict) else {}},
    }
    final_obj["final_sig"] = sha256_hex(canonical_json_dumps(final_obj).encode("utf-8"))
    _write_once_json(os.path.join(od, "final_response_v119.json"), dict(final_obj))

    # Create mind_graph_v119: copy mind_graph_v116 and append FAIL_EVENT_V119 nodes for rejected attempts.
    mg116_dir = os.path.join(od, "mind_graph_v116")
    mg119_dir = os.path.join(od, "mind_graph_v119")
    if os.path.isdir(mg116_dir):
        os.makedirs(mg119_dir, exist_ok=False)
        _copy_file_worm(os.path.join(mg116_dir, "mind_nodes.jsonl"), os.path.join(mg119_dir, "mind_nodes.jsonl"))
        _copy_file_worm(os.path.join(mg116_dir, "mind_edges.jsonl"), os.path.join(mg119_dir, "mind_edges.jsonl"))
        nodes_path = os.path.join(mg119_dir, "mind_nodes.jsonl")
        edges_path = os.path.join(mg119_dir, "mind_edges.jsonl")
        if not verify_chained_jsonl(nodes_path) or not verify_chained_jsonl(edges_path):
            raise ValueError("mind_graph_v119_chain_invalid_after_copy")

        # Build a map of existing nodes (by ato_id) so we can import plan nodes deterministically.
        existing_nodes: Dict[str, Dict[str, Any]] = {}
        for row in _read_jsonl(nodes_path):
            payload = row.get("payload") if isinstance(row, dict) else None
            ato = payload.get("ato") if isinstance(payload, dict) else None
            if not isinstance(ato, dict):
                continue
            ato_id = str(ato.get("ato_id") or "")
            if ato_id:
                existing_nodes[ato_id] = dict(ato)

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
        chosen_plan_row = _plan_row_for_user_turn(str(chosen_attempt_dir), str(chosen_user_turn_id))
        chosen_plan_id = str(chosen_plan_row.get("plan_id") or chosen_plan_row.get("plan_sig") or "")
        step0 = int(chosen_plan_row.get("created_step", chosen_last_user.get("created_step", 0) or 0) or 0)

        # Append FAIL_EVENT_V119s deterministically by attempt_index, then by reason code.
        for ar in sorted(list(attempts), key=lambda a: int(a.attempt_index)):
            if int(ar.attempt_index) == int(chosen_attempt):
                continue
            attempt_subdir = os.path.join(od, f"attempt_{int(ar.attempt_index):03d}")
            plan_row = _plan_row_for_user_turn(str(attempt_subdir), str(ar.user_turn_id))
            plan_id = str(plan_row.get("plan_id") or plan_row.get("plan_sig") or "") if isinstance(plan_row, dict) else ""
            if plan_id and plan_id not in existing_nodes and isinstance(plan_row, dict) and plan_row:
                plan_ato = _make_plan_ato_v119(action_plan=dict(plan_row))
                plan_ato_dict = plan_ato.to_dict(include_sig=True)
                prev_nodes_hash = _last_entry_hash(nodes_path) or None
                nodes_entry = {
                    "time": deterministic_iso(step=int(plan_ato.created_step)),
                    "step": int(plan_ato.created_step),
                    "event": "NODE",
                    "payload": {"reason": "plan_import_v119", "ato": dict(plan_ato_dict)},
                }
                append_chained_jsonl(nodes_path, dict(nodes_entry), prev_hash=prev_nodes_hash)
                existing_nodes[str(plan_id)] = dict(plan_ato_dict)

            # For each failure type, append a fail event.
            fail_specs: List[Tuple[str, Dict[str, Any]]] = []
            if not bool(ar.ok_final_v116):
                fail_specs.append(
                    (
                        "final_v116_fail",
                        {
                            "attempt_index": int(ar.attempt_index),
                            "seed_used": int(ar.seed_used),
                            "attempt_dir": str(ar.attempt_dir),
                            "reason_final_v116": str(ar.reason_final_v116),
                        },
                    )
                )
            if not bool(ar.ok_fluency):
                # Reference the per-attempt fluency summary for auditability.
                fsum_path = os.path.join(str(attempt_subdir), "fluency_summary_v119.json")
                fsum_sig = ""
                if os.path.exists(fsum_path):
                    fobj = _read_json(fsum_path)
                    if isinstance(fobj, dict):
                        fsum_sig = str(fobj.get("result_sig") or "")
                fail_specs.append(
                    (
                        "fluency_" + str(ar.reason_fluency),
                        {
                            "attempt_index": int(ar.attempt_index),
                            "seed_used": int(ar.seed_used),
                            "attempt_dir": str(ar.attempt_dir),
                            "reason_fluency": str(ar.reason_fluency),
                            "fluency_summary_result_sig": str(fsum_sig),
                            "fluency_summary_file": os.path.basename(str(fsum_path)),
                        },
                    )
                )

            for reason_code, evidence in sorted(fail_specs, key=lambda kv: str(kv[0])):
                fail_ato = _make_fail_event_ato_v119(
                    conversation_id=str(chosen_conversation_id),
                    user_turn_id=str(chosen_user_turn_id),
                    goal_ato_id=str(chosen_goal_id),
                    plan_ato_id=str(plan_id or chosen_plan_id),
                    reason_code=str(reason_code),
                    step=int(step0),
                    evidence=dict(evidence, replan_trace_sig=str(rep_obj.get("trace_sig") or ""), replan_trace_file="replan_trace_v119.json"),
                )
                _append_fail_event_to_mind_graph_v119(
                    mind_graph_v119_dir=str(mg119_dir),
                    fail_ato=fail_ato,
                    goal_ato_id=str(chosen_goal_id),
                    plan_ato_id=str(plan_id or chosen_plan_id),
                    user_turn_id=str(chosen_user_turn_id),
                    step=int(step0),
                    evidence_refs=[
                        {"kind": "turn", "turn_id": str(chosen_user_turn_id)},
                        {"kind": "replan_trace", "trace_sig": str(rep_obj.get("trace_sig") or "")},
                        {"kind": "attempt", "attempt_index": int(ar.attempt_index), "seed_used": int(ar.seed_used)},
                    ],
                )

        if not verify_chained_jsonl(nodes_path) or not verify_chained_jsonl(edges_path):
            raise ValueError("mind_graph_v119_chain_invalid_after_append")

    # Best-effort chain checks on promoted ledgers.
    for p in [
        os.path.join(od, "transcript.jsonl"),
        os.path.join(od, "conversation_turns.jsonl"),
        os.path.join(od, "intent_parses.jsonl"),
        os.path.join(od, "objective_evals.jsonl"),
        os.path.join(od, "dialogue_trials.jsonl"),
        os.path.join(od, "action_plans.jsonl"),
    ]:
        if os.path.exists(p) and not bool(verify_chained_jsonl_v96(p)):
            raise ValueError(f"chain_invalid_promoted:{os.path.basename(p)}")

    return {"final_response_v119": dict(final_obj), "replan_trace_v119_sig": str(rep_obj.get("trace_sig") or "")}
