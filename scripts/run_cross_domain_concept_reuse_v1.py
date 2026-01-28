#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, deterministic_iso
from atos_core.concepts import PRIMITIVE_OPS
from atos_core.engine import Engine, EngineConfig
from atos_core.ledger import Ledger
from atos_core.learn import KAAbsoluteTrainer, TrainConfig
from atos_core.store import ActStore
from atos_core.suite import run_skill_suite, skill_suite_tasks_for_pack


@dataclass(frozen=True)
class DomainEval:
    pack: str
    transcripts: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    family_ids: Set[str]
    used_concepts_ok: Set[str]


def _is_seed_concept(*, store: ActStore, concept_id: str) -> bool:
    cid = str(concept_id or "")
    if not cid:
        return True
    if cid.startswith("concept_seed_") or cid.startswith("goal_seed_") or cid.startswith("act_s000000_concept_"):
        return True
    try:
        act0 = store.get_concept_act(cid)
    except Exception:
        act0 = None
    if act0 is None or not isinstance(getattr(act0, "evidence", None), dict):
        return False
    name0 = str(act0.evidence.get("name") or "")
    return name0 in {"concept_seed_v0"}


def _family_ids_from_transcripts(transcripts: Sequence[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for tr in list(transcripts):
        turns = tr.get("turns") if isinstance(tr, dict) else None
        if not isinstance(turns, list):
            continue
        for t in list(turns):
            if not isinstance(t, dict):
                continue
            tr0 = t.get("trace") if isinstance(t.get("trace"), dict) else {}
            pt = tr0.get("plan_trace") if isinstance(tr0.get("plan_trace"), dict) else {}
            if not isinstance(pt, dict):
                continue
            # Only count turns where the harness provided a validator id (i.e., validate turn).
            if not str(pt.get("validator_id") or ""):
                continue
            fid = str(pt.get("family_id") or "")
            if fid:
                out.add(fid)
    return out


def _used_concepts_ok_from_metrics(metrics: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    usage = metrics.get("concept_usage_by_id") if isinstance(metrics.get("concept_usage_by_id"), dict) else {}
    for cid in sorted([str(k) for k in usage.keys() if str(k)], key=str):
        rec = usage.get(cid) if isinstance(usage.get(cid), dict) else {}
        try:
            ok_turns = int(rec.get("ok_turns", 0) or 0)
        except Exception:
            ok_turns = 0
        if ok_turns > 0:
            out.add(str(cid))
    return out


def _metric_float(metrics: Dict[str, Any], key: str) -> float:
    try:
        v = float(metrics.get(key))
    except Exception:
        v = float("nan")
    if v != v:
        return 0.0
    return float(v)


def _run_domain_eval(
    *,
    store: ActStore,
    pack: str,
    seed: int,
    max_new_tokens: int,
    prompt_history_k: int,
) -> DomainEval:
    tasks = skill_suite_tasks_for_pack(pack)
    eng = Engine(
        store,
        seed=int(seed),
        # Cross-domain proof requires the same explicit long-dialogue state used during training:
        # long-memory dialogue tasks in sota packs depend on deterministic MEMORY_FACTS rendering
        # when prompt history is truncated.
        config=EngineConfig(
            enable_contracts=False,
            dialogue_state_enabled=True,
            dialogue_state_prefix_enabled=True,
        ),
    )
    transcripts, metrics = run_skill_suite(
        eng,
        tasks=list(tasks),
        max_new_tokens=int(max_new_tokens),
        prompt_history_k=int(prompt_history_k),
    )
    fams = _family_ids_from_transcripts(transcripts)
    used_ok = _used_concepts_ok_from_metrics(metrics)
    return DomainEval(
        pack=str(pack),
        transcripts=list(transcripts),
        metrics=dict(metrics),
        family_ids=set(fams),
        used_concepts_ok=set(used_ok),
    )


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(canonical_json_dumps(row))
        f.write("\n")


def _trace_rows_for_domain(
    *,
    domain: str,
    pack: str,
    iteration: int,
    transcripts: Sequence[Dict[str, Any]],
) -> Iterable[Dict[str, Any]]:
    for tr in list(transcripts):
        if not isinstance(tr, dict):
            continue
        task_id = str(tr.get("task_id") or "")
        turns = tr.get("turns")
        if not isinstance(turns, list):
            continue
        # Choose the first turn that has a validator id (validate turn).
        chosen: Optional[Dict[str, Any]] = None
        pt_view: Dict[str, Any] = {}
        for t in list(turns):
            if not isinstance(t, dict):
                continue
            tr0 = t.get("trace") if isinstance(t.get("trace"), dict) else {}
            pt = tr0.get("plan_trace") if isinstance(tr0.get("plan_trace"), dict) else {}
            if not isinstance(pt, dict):
                continue
            if not str(pt.get("validator_id") or ""):
                continue
            chosen = t
            pt_view = dict(pt)
            break
        if chosen is None:
            continue
        tr0 = chosen.get("trace") if isinstance(chosen.get("trace"), dict) else {}
        cm = tr0.get("concept_executor") if isinstance(tr0.get("concept_executor"), dict) else {}
        yield {
            "time": deterministic_iso(step=int(iteration)),
            "iteration": int(iteration),
            "domain": str(domain),
            "pack": str(pack),
            "task_id": str(task_id),
            "family_id": str(pt_view.get("family_id") or ""),
            "validator_id": str(pt_view.get("validator_id") or ""),
            "expected_format": str(pt_view.get("expected_format") or ""),
            "constraints": list(pt_view.get("constraints") or []),
            "concept_policy_required": bool(pt_view.get("concept_policy_required", False)),
            "concept_min_depth": int(pt_view.get("concept_min_depth", 0) or 0),
            "concept_csg_min_nodes": int(pt_view.get("concept_csg_min_nodes", 0) or 0),
            "concept_csg_min_edges": int(pt_view.get("concept_csg_min_edges", 0) or 0),
            "concept_executor": {
                "used": bool(cm.get("used", False)),
                "ok": bool(cm.get("ok", False)),
                "reason": str(cm.get("reason") or ""),
                "concept_id": str(cm.get("concept_id") or ""),
                "concept_calls_total": int(cm.get("concept_calls_total", 0) or 0),
                "concept_calls_max_depth": int(cm.get("concept_calls_max_depth", 0) or 0),
                "concept_call_ids": list(cm.get("concept_call_ids") or []),
                "concept_nested_call_ids": list(cm.get("concept_nested_call_ids") or []),
            },
        }


def _write_concept_bank(*, store: ActStore, path: str, step: int) -> None:
    from atos_core.learn import estimate_act_cost_bits

    rows: List[Dict[str, Any]] = []
    for a in store.active():
        if str(getattr(a, "kind", "")) != "concept_csv":
            continue
        ev = a.evidence if isinstance(getattr(a, "evidence", None), dict) else {}
        iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
        meta = ev.get("meta") if isinstance(ev.get("meta"), dict) else {}
        ics = meta.get("ics_v1") if isinstance(meta.get("ics_v1"), dict) else {}
        rows.append(
            {
                "concept_id": str(a.id),
                "active": bool(getattr(a, "active", True)),
                "version": int(getattr(a, "version", 1) or 1),
                "match": dict(getattr(a, "match", {}) or {}),
                "interface": dict(iface) if isinstance(iface, dict) else {},
                "meta": dict(meta) if isinstance(meta, dict) else {},
                "ics_v1": dict(ics) if isinstance(ics, dict) else {},
                "deps": list(getattr(a, "deps", []) or []),
                "program_len": int(len(getattr(a, "program", []) or [])),
                "cost_bits": int(estimate_act_cost_bits(a)),
            }
        )
    rows.sort(key=lambda r: str(r.get("concept_id") or ""))
    bank = {
        "schema_version": 1,
        "generated_at": deterministic_iso(step=int(step)),
        "concepts_total": int(len(rows)),
        "concepts": list(rows),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _write_summary_md(
    *,
    path: str,
    base_acts_path: str,
    domain_a: DomainEval,
    domain_b: DomainEval,
    reuse: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# Cross-Domain Concept Reuse Report (v1)\n")
    lines.append(f"- base_acts_path: `{base_acts_path}`")
    lines.append(f"- domain_a_pack: `{domain_a.pack}`")
    lines.append(f"- domain_b_pack: `{domain_b.pack}`\n")

    def _fmt_domain(name: str, d: DomainEval) -> None:
        m = d.metrics
        lines.append(f"## {name}")
        lines.append(f"- utility_pass_rate: `{_metric_float(m, 'pass_rate'):.4f}`")
        lines.append(f"- concept_policy_pass_rate: `{_metric_float(m, 'concept_policy_pass_rate'):.4f}`")
        lines.append(
            f"- concept_selected_as_policy_rate: `{_metric_float(m, 'concept_selected_as_policy_rate'):.4f}`"
        )
        lines.append(f"- concept_calls_max_depth_mean: `{_metric_float(m, 'concept_calls_max_depth_mean'):.3f}`")
        lines.append(f"- concept_nested_call_rate: `{_metric_float(m, 'concept_nested_call_rate'):.3f}`")
        lines.append(f"- families_total: `{len(d.family_ids)}`")
        lines.append(f"- used_concepts_ok_total: `{len(d.used_concepts_ok)}`\n")

    _fmt_domain("Initial Domain", domain_a)
    _fmt_domain("New Domain", domain_b)

    lines.append("## Cross-Domain Metrics")
    for k in sorted(reuse.keys(), key=str):
        v = reuse.get(k)
        if isinstance(v, float):
            lines.append(f"- {k}: `{v:.6f}`")
        else:
            lines.append(f"- {k}: `{v}`")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def _run_train_induce_new_concepts(
    *,
    out_dir: str,
    seed: int,
    data_path: str,
    resume_acts_path: str,
    pack: str,
    steps: int,
    window: int,
    propose_every: int,
) -> str:
    cfg = TrainConfig(
        steps=int(steps),
        window=int(window),
        propose_every=int(propose_every),
        seed=int(seed),
        mode="pure",
        selection_mode="survival",
        enable_contracts=False,
        resume_acts_path=str(resume_acts_path),
        ics_enabled=True,
        ics_sovereign=True,
        ics_semantic_banks_enabled=True,
        # Keep pressure strict: if the new domain cannot be solved with existing concepts,
        # concept induction must happen or the run should fail-closed.
        survival_plateau_windows=2,
        survival_hard_fail_windows=2,
        survival_no_abstraction_windows=3,
        survival_no_reuse_windows=3,
        concept_csv_mining_enabled=True,
        concept_csv_composed_enabled=True,
        concept_csv_budget=24,
        concept_csv_deepwrap_max_new_per_window=12,
        skill_suite_pack=str(pack),
        # Keep suites cheap (symbolic): favor concept execution, not long freeform output.
        fluency_gen_tokens=64,
        skill_suite_max_new_tokens=96,
        skill_suite_prompt_history_k=4,
    )
    os.makedirs(out_dir, exist_ok=False)
    trainer = KAAbsoluteTrainer(data_path=str(data_path), out_dir=str(out_dir), config=cfg)
    trainer.train()
    return os.path.join(out_dir, "acts.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_run", required=True, help="Run dir containing acts.jsonl (WORM, read-only).")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--domain_a_pack", default="sota_v12")
    ap.add_argument("--domain_b_pack", default="xdom_v1")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--prompt_history_k", type=int, default=4)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--max_iters", type=int, default=3)
    ap.add_argument("--train_on_fail_steps", type=int, default=2000)
    ap.add_argument("--train_on_fail_window", type=int, default=500)
    ap.add_argument("--data_path", default="data/sample_text.txt")
    args = ap.parse_args()

    base_run = os.path.abspath(str(args.base_run))
    base_acts = os.path.join(base_run, "acts.jsonl")
    if not os.path.exists(base_acts):
        raise SystemExit(f"acts_not_found:{base_acts}")

    if args.out_dir:
        out_dir = os.path.abspath(str(args.out_dir))
    else:
        out_dir = os.path.join(
            "results",
            f"cross_domain_concept_reuse_v1_{time.strftime('%Y%m%d_%H%M%S')}_seed{int(args.seed)}",
        )
    if os.path.exists(out_dir):
        raise SystemExit(f"out_dir_exists:{out_dir}")
    os.makedirs(out_dir, exist_ok=False)

    trace_path = os.path.join(out_dir, "cross_domain_trace.jsonl")
    ledger_path = os.path.join(out_dir, "final_worm_ledger.jsonl")
    summary_path = os.path.join(out_dir, "final_summary.json")
    report_path = os.path.join(out_dir, "concept_reuse_report.md")
    concept_bank_path = os.path.join(out_dir, "final_concept_bank.json")

    prim_before = sorted(PRIMITIVE_OPS.keys(), key=str)
    acts_path = str(base_acts)

    ledger = Ledger(ledger_path)

    final: Dict[str, Any] = {}
    ok = False
    for it in range(1, int(args.max_iters) + 1):
        store = ActStore.load_jsonl(acts_path)
        store_a = copy.deepcopy(store)
        store_b = copy.deepcopy(store)

        domain_a = _run_domain_eval(
            store=store_a,
            pack=str(args.domain_a_pack),
            seed=int(args.seed),
            max_new_tokens=int(args.max_new_tokens),
            prompt_history_k=int(args.prompt_history_k),
        )
        domain_b = _run_domain_eval(
            store=store_b,
            pack=str(args.domain_b_pack),
            seed=int(args.seed) + 1,
            max_new_tokens=int(args.max_new_tokens),
            prompt_history_k=int(args.prompt_history_k),
        )

        for row in _trace_rows_for_domain(
            domain="initial",
            pack=str(domain_a.pack),
            iteration=int(it),
            transcripts=domain_a.transcripts,
        ):
            _append_jsonl(trace_path, row)
        for row in _trace_rows_for_domain(
            domain="new",
            pack=str(domain_b.pack),
            iteration=int(it),
            transcripts=domain_b.transcripts,
        ):
            _append_jsonl(trace_path, row)

        schema_overlap = sorted(domain_a.family_ids.intersection(domain_b.family_ids), key=str)

        shared = sorted(domain_a.used_concepts_ok.intersection(domain_b.used_concepts_ok), key=str)
        shared_non_seed = [
            cid for cid in shared if not _is_seed_concept(store=store, concept_id=str(cid))
        ]
        b_used_ok = sorted(domain_b.used_concepts_ok, key=str)
        reuse_rate = float(len(shared)) / float(max(1, len(b_used_ok)))
        reuse_rate_non_seed = float(len(shared_non_seed)) / float(max(1, len(b_used_ok)))

        a_depth = _metric_float(domain_a.metrics, "concept_calls_max_depth_mean")
        b_depth = _metric_float(domain_b.metrics, "concept_calls_max_depth_mean")
        planning_cost_reduced = bool(b_depth < a_depth - 1e-12)

        prim_after = sorted(PRIMITIVE_OPS.keys(), key=str)
        no_new_primitives = bool(prim_after == prim_before)

        reuse_meta: Dict[str, Any] = {
            "schema_overlap": int(len(schema_overlap)),
            "schema_overlap_ids": list(schema_overlap)[:8],
            "cross_domain_concept_reuse_count": int(len(shared)),
            "cross_domain_concept_reuse_non_seed_count": int(len(shared_non_seed)),
            "cross_domain_concept_reuse_rate": float(reuse_rate),
            "cross_domain_concept_reuse_non_seed_rate": float(reuse_rate_non_seed),
            "mean_plan_depth_initial_domain": float(a_depth),
            "mean_plan_depth_new_domain": float(b_depth),
            "planning_cost_reduced": bool(planning_cost_reduced),
            "concept_policy_rate_initial_domain": float(
                _metric_float(domain_a.metrics, "concept_selected_as_policy_rate")
            ),
            "concept_policy_rate_new_domain": float(
                _metric_float(domain_b.metrics, "concept_selected_as_policy_rate")
            ),
            "no_new_primitive_ops_added": bool(no_new_primitives),
        }

        # WORM ledger entry per iteration (no patches; freeze-mode evaluation).
        try:
            acts_hash = store.content_hash()
        except Exception:
            acts_hash = ""
        ledger.append(step=int(it), patch=None, acts_hash=str(acts_hash), metrics={"reuse": dict(reuse_meta)})

        final = {
            "schema_version": 1,
            "generated_at": deterministic_iso(step=int(it)),
            "base_run": str(base_run),
            "base_acts": os.path.relpath(base_acts, base_run),
            "iteration": int(it),
            "acts_path": str(acts_path),
            "domain_a_pack": str(domain_a.pack),
            "domain_b_pack": str(domain_b.pack),
            "domain_a_metrics": dict(domain_a.metrics),
            "domain_b_metrics": dict(domain_b.metrics),
            "cross_domain": dict(reuse_meta),
        }

        # Success conditions (final proof contract).
        if (
            float(reuse_rate) > 0.0
            and float(reuse_rate_non_seed) > 0.0
            and bool(planning_cost_reduced)
            and bool(no_new_primitives)
            and int(len(schema_overlap)) == 0
            and float(reuse_meta["concept_policy_rate_initial_domain"]) >= 1.0 - 1e-12
            and float(reuse_meta["concept_policy_rate_new_domain"]) >= 1.0 - 1e-12
        ):
            ok = True
            break

        # If no reuse, induce new concepts in the new domain and retry (bounded).
        if float(reuse_rate) <= 0.0 and int(it) < int(args.max_iters):
            train_dir = os.path.join(out_dir, f"induce_iter{int(it)}")
            acts_path = _run_train_induce_new_concepts(
                out_dir=str(train_dir),
                seed=int(args.seed) + int(it) * 10,
                data_path=str(args.data_path),
                resume_acts_path=str(acts_path),
                pack=str(args.domain_b_pack),
                steps=int(args.train_on_fail_steps),
                window=int(args.train_on_fail_window),
                propose_every=int(args.train_on_fail_window),
            )

    final["ok"] = bool(ok)
    final["agi_declared_by_contract"] = bool(ok)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    # Export final concept bank snapshot (WORM copy).
    try:
        store_final = ActStore.load_jsonl(acts_path)
        _write_concept_bank(store=store_final, path=concept_bank_path, step=int(final.get("iteration", 0) or 0))
    except Exception:
        pass

    # Human-auditable report.
    try:
        _write_summary_md(
            path=report_path,
            base_acts_path=str(base_acts),
            domain_a=domain_a,
            domain_b=domain_b,
            reuse=dict(final.get("cross_domain") or {}),
        )
    except Exception:
        pass

    print(out_dir)


if __name__ == "__main__":
    main()
