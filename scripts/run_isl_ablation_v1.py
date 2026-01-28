#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, deterministic_iso, estimate_act_cost_bits
from atos_core.concepts import PRIMITIVE_OPS
from atos_core.learn import KAAbsoluteTrainer, TrainConfig


def _ensure_absent(path: str) -> None:
    if os.path.exists(path):
        raise SystemExit(f"ERROR: path already exists (WORM): {path}")


def _read_last_jsonl(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except Exception:
                continue
    return last if isinstance(last, dict) else None


def _write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _write_concept_and_operator_banks(*, trainer: KAAbsoluteTrainer) -> None:
    out_dir = str(trainer.out_dir)

    # concept_bank.json
    try:
        rows: List[Dict[str, Any]] = []
        for a in trainer.store.active():
            if str(getattr(a, "kind", "")) != "concept_csv":
                continue
            ev = a.evidence if isinstance(getattr(a, "evidence", None), dict) else {}
            iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
            meta = ev.get("meta") if isinstance(ev.get("meta"), dict) else {}
            csg = ev.get("csg_v87") if isinstance(ev.get("csg_v87"), dict) else {}
            ics = meta.get("ics_v1") if isinstance(meta.get("ics_v1"), dict) else {}
            rows.append(
                {
                    "concept_id": str(a.id),
                    "active": bool(getattr(a, "active", True)),
                    "version": int(getattr(a, "version", 1) or 1),
                    "match": dict(getattr(a, "match", {}) or {}),
                    "interface": dict(iface) if isinstance(iface, dict) else {},
                    "csg_v87": dict(csg) if isinstance(csg, dict) and csg else {},
                    "ics_v1": dict(ics) if isinstance(ics, dict) else {},
                    "deps": list(getattr(a, "deps", []) or []),
                    "program_len": int(len(getattr(a, "program", []) or [])),
                    "cost_bits": int(estimate_act_cost_bits(a)),
                }
            )
        rows.sort(key=lambda r: str(r.get("concept_id") or ""))
        _write_json(
            os.path.join(out_dir, "concept_bank.json"),
            {
                "schema_version": 1,
                "generated_at": deterministic_iso(step=int(getattr(trainer.config, "steps", 0) or 0)),
                "concepts_total": int(len(rows)),
                "concepts": list(rows),
            },
        )
    except Exception:
        pass

    # operator_bank.json
    try:
        prim: Dict[str, Any] = {}
        for op_id in sorted(PRIMITIVE_OPS.keys(), key=str):
            spec_fn = PRIMITIVE_OPS.get(op_id)
            if not isinstance(spec_fn, tuple) or len(spec_fn) != 2:
                continue
            spec = spec_fn[0]
            prim[str(op_id)] = {
                "arity": int(getattr(spec, "arity", 0) or 0),
                "input_types": list(getattr(spec, "input_types", ()) or ()),
                "output_type": str(getattr(spec, "output_type", "") or ""),
            }
        _write_json(
            os.path.join(out_dir, "operator_bank.json"),
            {
                "schema_version": 1,
                "generated_at": deterministic_iso(step=int(getattr(trainer.config, "steps", 0) or 0)),
                "primitive_ops": dict(prim),
            },
        )
    except Exception:
        pass

    # goal_bank.json
    try:
        rows: List[Dict[str, Any]] = []
        for a in trainer.store.active():
            if str(getattr(a, "kind", "")) != "goal":
                continue
            ev = a.evidence if isinstance(getattr(a, "evidence", None), dict) else {}
            goal = ev.get("goal") if isinstance(ev.get("goal"), dict) else {}
            meta = ev.get("meta") if isinstance(ev.get("meta"), dict) else {}
            rows.append(
                {
                    "goal_id": str(a.id),
                    "active": bool(getattr(a, "active", True)),
                    "version": int(getattr(a, "version", 1) or 1),
                    "goal": dict(goal) if isinstance(goal, dict) else {},
                    "meta": dict(meta) if isinstance(meta, dict) else {},
                    "cost_bits": int(estimate_act_cost_bits(a)),
                }
            )
        rows.sort(key=lambda r: str(r.get("goal_id") or ""))
        _write_json(
            os.path.join(out_dir, "goal_bank.json"),
            {
                "schema_version": 1,
                "generated_at": deterministic_iso(step=int(getattr(trainer.config, "steps", 0) or 0)),
                "goals_total": int(len(rows)),
                "goals": list(rows),
            },
        )
    except Exception:
        pass

    # plan_bank.json
    try:
        rows = []
        for a in trainer.store.active():
            if str(getattr(a, "kind", "")) != "plan":
                continue
            ev = a.evidence if isinstance(getattr(a, "evidence", None), dict) else {}
            plan = ev.get("plan") if isinstance(ev.get("plan"), dict) else {}
            meta = ev.get("meta") if isinstance(ev.get("meta"), dict) else {}
            rows.append(
                {
                    "plan_id": str(a.id),
                    "active": bool(getattr(a, "active", True)),
                    "version": int(getattr(a, "version", 1) or 1),
                    "plan": dict(plan) if isinstance(plan, dict) else {},
                    "meta": dict(meta) if isinstance(meta, dict) else {},
                    "cost_bits": int(estimate_act_cost_bits(a)),
                }
            )
        rows.sort(key=lambda r: str(r.get("plan_id") or ""))
        _write_json(
            os.path.join(out_dir, "plan_bank.json"),
            {
                "schema_version": 1,
                "generated_at": deterministic_iso(step=int(getattr(trainer.config, "steps", 0) or 0)),
                "plans_total": int(len(rows)),
                "plans": list(rows),
            },
        )
    except Exception:
        pass

    # hypothesis_bank.json
    try:
        rows = []
        for a in trainer.store.active():
            if str(getattr(a, "kind", "")) != "hypothesis":
                continue
            ev = a.evidence if isinstance(getattr(a, "evidence", None), dict) else {}
            hyp = ev.get("hypothesis") if isinstance(ev.get("hypothesis"), dict) else {}
            meta = ev.get("meta") if isinstance(ev.get("meta"), dict) else {}
            rows.append(
                {
                    "hypothesis_id": str(a.id),
                    "active": bool(getattr(a, "active", True)),
                    "version": int(getattr(a, "version", 1) or 1),
                    "hypothesis": dict(hyp) if isinstance(hyp, dict) else {},
                    "meta": dict(meta) if isinstance(meta, dict) else {},
                    "cost_bits": int(estimate_act_cost_bits(a)),
                }
            )
        rows.sort(key=lambda r: str(r.get("hypothesis_id") or ""))
        _write_json(
            os.path.join(out_dir, "hypothesis_bank.json"),
            {
                "schema_version": 1,
                "generated_at": deterministic_iso(step=int(getattr(trainer.config, "steps", 0) or 0)),
                "hypotheses_total": int(len(rows)),
                "hypotheses": list(rows),
            },
        )
    except Exception:
        pass

    # reference_bank.json
    try:
        rows = []
        for a in trainer.store.active():
            if str(getattr(a, "kind", "")) != "reference":
                continue
            ev = a.evidence if isinstance(getattr(a, "evidence", None), dict) else {}
            ref = ev.get("reference") if isinstance(ev.get("reference"), dict) else {}
            meta = ev.get("meta") if isinstance(ev.get("meta"), dict) else {}
            rows.append(
                {
                    "reference_id": str(a.id),
                    "active": bool(getattr(a, "active", True)),
                    "version": int(getattr(a, "version", 1) or 1),
                    "reference": dict(ref) if isinstance(ref, dict) else {},
                    "meta": dict(meta) if isinstance(meta, dict) else {},
                    "cost_bits": int(estimate_act_cost_bits(a)),
                }
            )
        rows.sort(key=lambda r: str(r.get("reference_id") or ""))
        _write_json(
            os.path.join(out_dir, "reference_bank.json"),
            {
                "schema_version": 1,
                "generated_at": deterministic_iso(step=int(getattr(trainer.config, "steps", 0) or 0)),
                "references_total": int(len(rows)),
                "references": list(rows),
            },
        )
    except Exception:
        pass

    # Compatibility copies for naming.
    try:
        src = os.path.join(out_dir, "ledger.jsonl")
        dst = os.path.join(out_dir, "worm_ledger.jsonl")
        if os.path.exists(src) and (not os.path.exists(dst)):
            shutil.copyfile(src, dst)
    except Exception:
        pass
    try:
        src = os.path.join(out_dir, "report.jsonl")
        dst = os.path.join(out_dir, "metrics.jsonl")
        if os.path.exists(src) and (not os.path.exists(dst)):
            shutil.copyfile(src, dst)
    except Exception:
        pass

    # ics_events.jsonl (WORM): extract ICS meta per step from the ledger hash-chain.
    try:
        src = os.path.join(out_dir, "ledger.jsonl")
        dst = os.path.join(out_dir, "ics_events.jsonl")
        if os.path.exists(src) and (not os.path.exists(dst)):
            with open(dst, "x", encoding="utf-8") as f_out:
                with open(src, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(row, dict):
                            continue
                        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                        patch_meta = metrics.get("patch_meta") if isinstance(metrics.get("patch_meta"), dict) else {}
                        ics = patch_meta.get("ics_v1") if isinstance(patch_meta.get("ics_v1"), dict) else None
                        if not isinstance(ics, dict):
                            continue
                        out_row = {
                            "step": int(row.get("step", 0) or 0),
                            "entry_hash": str(row.get("entry_hash") or ""),
                            "prev_hash": str(row.get("prev_hash") or ""),
                            "acts_hash": str(row.get("acts_hash") or ""),
                            "ics_v1": dict(ics),
                        }
                        f_out.write(canonical_json_dumps(out_row))
                        f_out.write("\n")
    except Exception:
        pass

    # summary.json (WORM-ish within a new run dir): snapshot the last metrics row.
    try:
        last = _read_last_jsonl(os.path.join(out_dir, "metrics.jsonl"))
        if isinstance(last, dict):
            _write_json(
                os.path.join(out_dir, "summary.json"),
                {
                    "schema_version": 1,
                    "generated_at": deterministic_iso(step=int(getattr(trainer.config, "steps", 0) or 0)),
                    "last_metrics": dict(last),
                },
            )
    except Exception:
        pass


def _run_one(
    *,
    name: str,
    out_dir: str,
    data_path: str,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=False)
    trainer = KAAbsoluteTrainer(data_path=str(data_path), out_dir=str(out_dir), config=cfg)

    status = "ok"
    err: Optional[str] = None
    t0 = time.time()
    try:
        trainer.train()
    except Exception as e:
        status = "error"
        err = str(e)
    finally:
        # Ensure WORM artifacts exist even on failure.
        try:
            trainer.store.save_jsonl(trainer.acts_path)
        except Exception:
            pass
        _write_concept_and_operator_banks(trainer=trainer)

    elapsed_s = float(time.time() - t0)
    last_metrics = _read_last_jsonl(os.path.join(out_dir, "metrics.jsonl"))

    return {
        "name": str(name),
        "status": str(status),
        "error": str(err or ""),
        "elapsed_s": float(elapsed_s),
        "out_dir": str(out_dir),
        "last_metrics": dict(last_metrics) if isinstance(last_metrics, dict) else None,
    }


def _survival_score_from_row(row: Dict[str, Any]) -> float:
    """
    ISL proxy score (deterministic): success - search_cost - mdl_cost - invocation_cost.
    This is a lightweight proxy for the ablation report; the trainer's hard-fail laws remain
    the actual survival mechanism.
    """
    try:
        success = float(row.get("utility_pass_rate") or 0.0)
    except Exception:
        success = 0.0
    try:
        search_cost = float(row.get("search_steps_per_turn_mean") or 0.0) / 1000.0
    except Exception:
        search_cost = 0.0
    try:
        mdl_cost = float(row.get("mdl_total_est_bits") or 0.0) / 1e6
    except Exception:
        mdl_cost = 0.0
    try:
        inv_cost = float(row.get("utility_concept_calls_total_sum") or 0.0) / 200.0
    except Exception:
        inv_cost = 0.0
    return float(success - search_cost - mdl_cost - inv_cost)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/sample_text.txt")
    ap.add_argument("--out_base", default="")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--window", type=int, default=500)
    args = ap.parse_args()

    data_path = str(args.data_path)
    if not os.path.exists(data_path):
        raise SystemExit(f"ERROR: data_path not found: {data_path}")

    if args.out_base:
        out_base = str(args.out_base)
    else:
        out_base = os.path.join(
            "results",
            f"isl_ablation_v1_{time.strftime('%Y%m%d_%H%M%S')}_seed{int(args.seed)}",
        )
    _ensure_absent(out_base)
    os.makedirs(out_base, exist_ok=False)

    base = TrainConfig(
        steps=int(args.steps),
        window=int(args.window),
        propose_every=int(args.window),
        seed=int(args.seed),
        mode="pure",
        selection_mode="survival",
        enable_contracts=False,
        ics_sovereign=True,
        ics_semantic_banks_enabled=True,
        # Keep suites cheap but structurally meaningful.
        fluency_gen_tokens=64,
        skill_suite_max_new_tokens=96,
        skill_suite_prompt_history_k=4,
        # Structural pressure knobs (these are the "pressure" in ISL).
        survival_plateau_windows=2,
        survival_hard_fail_windows=2,
        survival_no_abstraction_windows=3,
        survival_no_reuse_windows=3,
        # Concepts as survival law:
        concept_csv_mining_enabled=True,
        concept_csv_composed_enabled=True,
        concept_csv_budget=24,
        concept_csv_deepwrap_max_new_per_window=12,
        # Ensure plan depth + semantic IAC + binding + concept-as-policy + non-trivial CSG pressure exists.
        skill_suite_pack="sota_v12",
    )

    runs: List[Dict[str, Any]] = []

    # FULL (ICS + pressure)
    cfg_full = TrainConfig(**base.__dict__)
    cfg_full.ics_enabled = True
    runs.append(
        _run_one(
            name="FULL",
            out_dir=os.path.join(out_base, "full"),
            data_path=data_path,
            cfg=cfg_full,
        )
    )

    # NO-ICS (pressure remains, but sovereign operator is disabled => should collapse)
    cfg_no_ics = TrainConfig(**base.__dict__)
    cfg_no_ics.ics_enabled = False
    runs.append(
        _run_one(
            name="NO-ICS",
            out_dir=os.path.join(out_base, "no_ics"),
            data_path=data_path,
            cfg=cfg_no_ics,
        )
    )

    # NO-PRESSURE (ICS on, but remove pressure so convergence should not be forced)
    cfg_no_pressure = TrainConfig(**base.__dict__)
    cfg_no_pressure.ics_enabled = True
    cfg_no_pressure.survival_hard_fail_windows = 0
    cfg_no_pressure.survival_no_abstraction_windows = 0
    cfg_no_pressure.survival_no_reuse_windows = 0
    cfg_no_pressure.skill_suite_pack = "v0"
    runs.append(
        _run_one(
            name="NO-PRESSURE",
            out_dir=os.path.join(out_base, "no_pressure"),
            data_path=data_path,
            cfg=cfg_no_pressure,
        )
    )

    # SHUFFLE-FAMILIES (ICS on + pressure, but shuffle latent families only for ICS evidence)
    cfg_shuffle = TrainConfig(**base.__dict__)
    cfg_shuffle.ics_enabled = True
    cfg_shuffle.suite_shuffle_families = True
    runs.append(
        _run_one(
            name="SHUFFLE-FAMILIES",
            out_dir=os.path.join(out_base, "shuffle_families"),
            data_path=data_path,
            cfg=cfg_shuffle,
        )
    )

    # Report
    lines: List[str] = []
    lines.append("# ISL Ablation Report (v1)\n")
    lines.append(f"- out_base: `{out_base}`\n")
    lines.append(f"- data_path: `{data_path}`\n")
    lines.append(f"- steps: {int(args.steps)}\n")
    lines.append(f"- window: {int(args.window)}\n")
    lines.append("\n## Runs\n")
    for r in runs:
        name = str(r.get("name") or "")
        status = str(r.get("status") or "")
        err = str(r.get("error") or "")
        out_dir = str(r.get("out_dir") or "")
        row = r.get("last_metrics") if isinstance(r.get("last_metrics"), dict) else None
        score = _survival_score_from_row(row) if isinstance(row, dict) else None
        lines.append(f"### {name}\n")
        lines.append(f"- status: `{status}`\n")
        lines.append(f"- out_dir: `{out_dir}`\n")
        if score is not None:
            lines.append(f"- survival_score_proxy: `{score:.6f}`\n")
        if isinstance(row, dict):
            lines.append(f"- last_step: `{int(row.get('step', 0) or 0)}`\n")
            try:
                lines.append(f"- utility_pass_rate: `{float(row.get('utility_pass_rate') or 0.0):.4f}`\n")
            except Exception:
                pass
            try:
                lines.append(
                    f"- concept_policy_pass_rate: `{float(row.get('utility_concept_policy_pass_rate') or 0.0):.4f}`\n"
                )
            except Exception:
                pass
            try:
                lines.append(
                    f"- concept_selected_as_policy_rate: `{float(row.get('utility_concept_selected_as_policy_rate') or 0.0):.4f}`\n"
                )
            except Exception:
                pass
            try:
                lines.append(f"- concept_calls_max_depth_mean: `{float(row.get('utility_concept_calls_max_depth_mean') or 0.0):.3f}`\n")
            except Exception:
                pass
            try:
                lines.append(f"- concept_nested_call_rate: `{float(row.get('utility_concept_nested_call_rate') or 0.0):.3f}`\n")
            except Exception:
                pass
            try:
                lines.append(f"- concept_min_depth_required_max: `{int(row.get('utility_concept_min_depth_required_max') or 0)}`\n")
            except Exception:
                pass
        if err:
            lines.append(f"- error: `{err}`\n")
        lines.append("\n")

    with open(os.path.join(out_base, "ablation_report.md"), "w", encoding="utf-8") as f:
        f.write("".join(lines))

    # Replay bundle
    bundle_path = os.path.join(out_base, "replay_bundle.tar.gz")
    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(out_base, arcname=os.path.basename(out_base))

    _write_json(os.path.join(out_base, "ablation_runs.json"), {"schema_version": 1, "runs": runs})

    print(f"[isl] wrote: {os.path.join(out_base, 'ablation_report.md')}")
    print(f"[isl] wrote: {bundle_path}")


if __name__ == "__main__":
    main()
