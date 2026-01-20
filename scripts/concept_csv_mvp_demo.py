#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.concept_miner import ConceptBirthTrigger
from atos_core.concepts import (
    Concept,
    ConceptInterface,
    ConceptPolicies,
    ConceptRegistry,
    execute_binary_op,
    execute_concept_subgraph,
    execute_unary_pipeline,
    infer_unary_io_types,
    stable_hash_obj,
)
from atos_core.validators import canonical_nonneg_int_text, run_validator


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def shannon_entropy(counts: Dict[str, int]) -> float:
    total = float(sum(int(v) for v in counts.values()))
    if total <= 0:
        return 0.0
    import math

    ent = 0.0
    for v in counts.values():
        p = float(v) / total
        if p > 0:
            ent -= p * math.log(p, 2)
    return float(ent)


@dataclass
class Strategy:
    strategy_id: str
    ops: List[str]
    cost: float

    def subgraph_ref(self) -> Dict[str, Any]:
        return {"kind": "unary_pipeline_v0", "ops": list(self.ops)}


def concept_score(c: Concept, *, popularity: float) -> float:
    u = float(c.u_ema)
    k = float(c.k_ema)
    base = u / (1.0 + k)
    if popularity > 0.9:
        base -= 0.1 * (popularity - 0.9)
    return float(base)


def strategy_score(pass_rate_est: float, cost: float) -> float:
    u = float(pass_rate_est)
    k = float(cost)
    return float(u / (1.0 + k))


def pick_best_concept(concepts: List[Concept]) -> Optional[Concept]:
    if not concepts:
        return None
    concepts = sorted(concepts, key=lambda c: str(c.id))
    best = concepts[0]
    return best


def filter_alive_matching(registry: ConceptRegistry, interface: ConceptInterface) -> List[Concept]:
    t = interface.type_signature()
    return [c for c in registry.alive_concepts() if c.interface.type_signature() == t]


def plan_try_unary(
    *,
    registry: ConceptRegistry,
    birth: ConceptBirthTrigger,
    step: int,
    goal_key: str,
    interface: ConceptInterface,
    strategies: List[Strategy],
    inputs: Dict[str, Any],
    expected: Any,
    context_signature: str,
    call_depth: int,
    prefer_safe_after_fail: bool,
    last_fail_flag: bool,
) -> Tuple[Any, bool, float, str]:
    baseline_cost = float(max(s.cost for s in strategies))

    alive = filter_alive_matching(registry, interface)
    popularity_total = max(1, sum(int(c.calls_total) for c in alive))
    concept_candidates: List[Tuple[float, str, Concept]] = []
    for c in alive:
        pop = float(c.calls_total) / float(popularity_total)
        concept_candidates.append((concept_score(c, popularity=pop), str(c.id), c))
    concept_candidates.sort(key=lambda t: (-t[0], t[1]))

    # Prefer "safe" strategy once right after a failure to avoid runaway collapse.
    if bool(prefer_safe_after_fail) and bool(last_fail_flag):
        safe = max(strategies, key=lambda s: (s.cost, s.strategy_id))
        out = execute_concept_subgraph(safe.subgraph_ref(), inputs)
        vr = run_validator(interface.validator_id, out, expected)
        cost_used = float(safe.cost)
        registry.log_primitives(
            step=int(step),
            subgraph_ref=safe.subgraph_ref(),
            interface=interface,
            inputs=inputs,
            expected=expected,
            output=out,
            validator_result=vr,
            cost_used=cost_used,
            baseline_cost=baseline_cost,
            context_signature=context_signature,
            call_depth=int(call_depth),
            note="forced_safe_after_fail",
        )
        birth.observe(
            registry=registry,
            key=f"{goal_key}::{safe.strategy_id}",
            step=int(step),
            subgraph_ref=safe.subgraph_ref(),
            interface=interface,
            context_signature=context_signature,
            passed=bool(vr.passed),
            cost_used=float(safe.cost),
        )
        return out, bool(vr.passed), float(cost_used), f"primitives:{safe.strategy_id}"

    if concept_candidates:
        _, _, c = concept_candidates[0]
        out, vr, cost_used = registry.call(
            step=int(step),
            concept_id=str(c.id),
            inputs=inputs,
            expected=expected,
            context_signature=context_signature,
            call_depth=int(call_depth),
            baseline_cost=baseline_cost,
            contract_active=False,
        )
        if bool(prefer_safe_after_fail) and not bool(vr.passed):
            safe = max(strategies, key=lambda s: (s.cost, s.strategy_id))
            out2 = execute_concept_subgraph(safe.subgraph_ref(), inputs)
            vr2 = run_validator(interface.validator_id, out2, expected)
            registry.log_primitives(
                step=int(step),
                subgraph_ref=safe.subgraph_ref(),
                interface=interface,
                inputs=inputs,
                expected=expected,
                output=out2,
                validator_result=vr2,
                cost_used=float(safe.cost),
                baseline_cost=baseline_cost,
                context_signature=context_signature,
                call_depth=int(call_depth),
                note=f"fallback_after_concept_fail:concept_id={c.id}",
            )
            birth.observe(
                registry=registry,
                key=f"{goal_key}::{safe.strategy_id}",
                step=int(step),
                subgraph_ref=safe.subgraph_ref(),
                interface=interface,
                context_signature=context_signature,
                passed=bool(vr2.passed),
                cost_used=float(safe.cost),
            )
            return out2, bool(vr2.passed), float(cost_used + safe.cost), f"concept:{c.id}+fallback:{safe.strategy_id}"
        return out, bool(vr.passed), float(cost_used), f"concept:{c.id}"

    # No concept yet: choose between primitive strategies by observed pass_rate_window in the birth trigger.
    snap = birth.stats_snapshot()
    scored: List[Tuple[float, str, Strategy]] = []
    for s in strategies:
        k = f"{goal_key}::{s.strategy_id}"
        pr = float(snap.get(k, {}).get("pass_rate_window", 0.5))
        scored.append((strategy_score(pr, s.cost), s.strategy_id, s))
    scored.sort(key=lambda t: (-t[0], t[1]))
    chosen = scored[0][2]

    out = execute_concept_subgraph(chosen.subgraph_ref(), inputs)
    vr = run_validator(interface.validator_id, out, expected)
    cost_used = float(chosen.cost)
    registry.log_primitives(
        step=int(step),
        subgraph_ref=chosen.subgraph_ref(),
        interface=interface,
        inputs=inputs,
        expected=expected,
        output=out,
        validator_result=vr,
        cost_used=cost_used,
        baseline_cost=baseline_cost,
        context_signature=context_signature,
        call_depth=int(call_depth),
        note="chosen_by_score",
    )
    birth.observe(
        registry=registry,
        key=f"{goal_key}::{chosen.strategy_id}",
        step=int(step),
        subgraph_ref=chosen.subgraph_ref(),
        interface=interface,
        context_signature=context_signature,
        passed=bool(vr.passed),
        cost_used=float(chosen.cost),
    )
    if bool(prefer_safe_after_fail) and not bool(vr.passed):
        safe = max(strategies, key=lambda s: (s.cost, s.strategy_id))
        out2 = execute_concept_subgraph(safe.subgraph_ref(), inputs)
        vr2 = run_validator(interface.validator_id, out2, expected)
        registry.log_primitives(
            step=int(step),
            subgraph_ref=safe.subgraph_ref(),
            interface=interface,
            inputs=inputs,
            expected=expected,
            output=out2,
            validator_result=vr2,
            cost_used=float(safe.cost),
            baseline_cost=baseline_cost,
            context_signature=context_signature,
            call_depth=int(call_depth),
            note=f"fallback_after_primitives_fail:strategy_id={chosen.strategy_id}",
        )
        birth.observe(
            registry=registry,
            key=f"{goal_key}::{safe.strategy_id}",
            step=int(step),
            subgraph_ref=safe.subgraph_ref(),
            interface=interface,
            context_signature=context_signature,
            passed=bool(vr2.passed),
            cost_used=float(safe.cost),
        )
        return out2, bool(vr2.passed), float(cost_used + safe.cost), f"primitives:{chosen.strategy_id}+fallback:{safe.strategy_id}"
    return out, bool(vr.passed), float(cost_used), f"primitives:{chosen.strategy_id}"


def id_morphism(x: Any) -> Any:
    return x


def demo_category_checks(concepts: List[Concept]) -> Dict[str, Any]:
    unary = []
    for c in concepts:
        if not bool(c.alive):
            continue
        if str(c.subgraph_ref.get("kind", "")) != "unary_pipeline_v0":
            continue
        unary.append(c)
    unary = sorted(unary, key=lambda c: str(c.id))
    if not unary:
        return {"identity_error": 0, "assoc_error": 0, "checked": 0}

    samples = ["x=00042", "id=7", "n=0000", "v=00123"]
    identity_error = 0
    assoc_error = 0
    checked = 0

    for c in unary:
        ops = list(c.subgraph_ref.get("ops", []))
        io = infer_unary_io_types(ops)
        if io is None:
            continue
        in_t, out_t = io
        if in_t != "str":
            continue
        for s in samples:
            checked += 1
            y = execute_unary_pipeline(ops, s)
            y1 = execute_unary_pipeline(ops, id_morphism(s))
            if stable_hash_obj(y) != stable_hash_obj(y1):
                identity_error += 1

            if out_t == "str":
                y2 = id_morphism(y)
            else:
                y2 = y
            if stable_hash_obj(y) != stable_hash_obj(y2):
                identity_error += 1

    if len(unary) >= 3:
        f, g, h = unary[0], unary[1], unary[2]
        f_ops = list(f.subgraph_ref.get("ops", []))
        g_ops = list(g.subgraph_ref.get("ops", []))
        h_ops = list(h.subgraph_ref.get("ops", []))
        for s in samples:
            checked += 1
            try:
                left = execute_unary_pipeline(h_ops, execute_unary_pipeline(g_ops, execute_unary_pipeline(f_ops, s)))
                right = execute_unary_pipeline(h_ops, execute_unary_pipeline(g_ops, execute_unary_pipeline(f_ops, s)))
                if stable_hash_obj(left) != stable_hash_obj(right):
                    assoc_error += 1
            except Exception:
                assoc_error += 1

    return {"identity_error": int(identity_error), "assoc_error": int(assoc_error), "checked": int(checked)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="New WORM-safe run dir (must not exist)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()

    out_dir = str(args.out)
    if os.path.exists(out_dir):
        raise SystemExit(f"--out already exists (WORM): {out_dir}")

    rnd = random.Random(int(args.seed))

    policies = ConceptPolicies(
        ema_alpha=0.2,
        prune_min_calls=5,
        prune_min_lifetime_steps=5,
        prune_fail_streak=3,
        prune_u_threshold=0.6,
        prune_s_threshold=0.35,
    )
    registry = ConceptRegistry(run_dir=out_dir)
    birth = ConceptBirthTrigger(
        window_size=50,
        birth_min_count=5,
        birth_min_pass_rate=0.8,
        birth_min_avg_cost=2.0,
        policies=policies,
    )

    # Micro-tasks and validators
    iface_parse_int = ConceptInterface(
        input_schema={"text": "str"},
        output_schema={"value": "int"},
        validator_id="int_value_exact",
        preconditions={"has_digits": True},
        postconditions={"type": "int"},
    )
    iface_norm_text = ConceptInterface(
        input_schema={"text": "str"},
        output_schema={"text": "str"},
        validator_id="int_text_canonical_exact",
        preconditions={"has_digits": True},
        postconditions={"canonical_int_text": True},
    )

    # Goal: normalize int-text (buggy vs robust)
    norm_buggy = Strategy("norm_buggy_strip1", ["scan_digits", "strip_one_leading_zero"], cost=2.0)
    norm_full = Strategy("norm_full_int", ["scan_digits", "digits_to_int", "int_to_digits"], cost=3.0)

    # Goal: parse int value
    parse_full = Strategy("parse_full", ["scan_digits", "digits_to_int"], cost=2.0)

    totals: Dict[str, Any] = {
        "episodes": int(args.episodes),
        "top_level_pass": 0,
        "top_level_fail": 0,
        "top_level_cost_used_sum": 0.0,
        "top_level_cost_baseline_sum": 0.0,
        "births": 0,
        "prunes": 0,
    }

    normalize_last_failed = False
    known_concepts: Dict[str, int] = {}

    step = 0
    for ep in range(int(args.episodes)):
        task_kind = ["T2_norm_text", "T3_sum_ints", "T1_json_ab"][ep % 3]

        # Deterministic regime change to force a brittle concept to fail and get pruned.
        zpad = 1 if ep < 15 else 3

        def make_text_int() -> Tuple[str, int]:
            n = rnd.randint(0, 999)
            digits = ("0" * int(zpad)) + str(int(n))
            txt = f"val={digits}"
            return txt, int(str(int(digits)))

        if task_kind == "T2_norm_text":
            text, exp_int = make_text_int()
            exp_text = str(int(exp_int))
            baseline_cost = float(norm_full.cost)

            out, ok, cost_used, used = plan_try_unary(
                registry=registry,
                birth=birth,
                step=int(step),
                goal_key="goal:norm_text",
                interface=iface_norm_text,
                strategies=[norm_buggy, norm_full],
                inputs={"text": text},
                expected=exp_text,
                context_signature=f"{task_kind}␟ep={ep}",
                call_depth=0,
                prefer_safe_after_fail=True,
                last_fail_flag=bool(normalize_last_failed),
            )
            totals["top_level_cost_used_sum"] += float(cost_used)
            totals["top_level_cost_baseline_sum"] += float(baseline_cost)
            if ok:
                totals["top_level_pass"] += 1
                normalize_last_failed = False
            else:
                totals["top_level_fail"] += 1
                normalize_last_failed = True
            step += 1

            for c in registry.concepts():
                if c.id not in known_concepts:
                    known_concepts[c.id] = int(step)
                    totals["births"] += 1
            continue

        if task_kind == "T3_sum_ints":
            text_a, exp_a = make_text_int()
            text_b, exp_b = make_text_int()
            exp_sum = int(exp_a) + int(exp_b)
            exp_sum_text = str(int(exp_sum))

            a, ok_a, cost_a, used_a = plan_try_unary(
                registry=registry,
                birth=birth,
                step=int(step),
                goal_key="goal:parse_int",
                interface=iface_parse_int,
                strategies=[parse_full],
                inputs={"text": text_a},
                expected=int(exp_a),
                context_signature=f"{task_kind}:a␟ep={ep}",
                call_depth=1,
                prefer_safe_after_fail=False,
                last_fail_flag=False,
            )
            step += 1
            b, ok_b, cost_b, used_b = plan_try_unary(
                registry=registry,
                birth=birth,
                step=int(step),
                goal_key="goal:parse_int",
                interface=iface_parse_int,
                strategies=[parse_full],
                inputs={"text": text_b},
                expected=int(exp_b),
                context_signature=f"{task_kind}:b␟ep={ep}",
                call_depth=1,
                prefer_safe_after_fail=False,
                last_fail_flag=False,
            )
            step += 1

            s = execute_binary_op("add_int", a, b)
            out_text = execute_unary_pipeline(["int_to_digits"], s)
            vr = run_validator("int_text_canonical_exact", out_text, exp_sum_text)

            baseline_cost = float(parse_full.cost + parse_full.cost + 1.0 + 1.0)
            cost_used = float(cost_a + cost_b + 1.0 + 1.0)
            totals["top_level_cost_used_sum"] += float(cost_used)
            totals["top_level_cost_baseline_sum"] += float(baseline_cost)

            registry.log_primitives(
                step=int(step),
                subgraph_ref={"kind": "top_level_v0", "task": task_kind},
                interface=ConceptInterface(
                    input_schema={"text_a": "str", "text_b": "str"},
                    output_schema={"text": "str"},
                    validator_id="int_text_canonical_exact",
                ),
                inputs={"text_a": text_a, "text_b": text_b},
                expected=exp_sum_text,
                output=out_text,
                validator_result=vr,
                cost_used=float(cost_used),
                baseline_cost=float(baseline_cost),
                context_signature=f"{task_kind}␟ep={ep}",
                call_depth=0,
                note=f"sum_path:a={used_a},b={used_b},ok_a={ok_a},ok_b={ok_b}",
            )
            if bool(vr.passed):
                totals["top_level_pass"] += 1
            else:
                totals["top_level_fail"] += 1
            step += 1

            for c in registry.concepts():
                if c.id not in known_concepts:
                    known_concepts[c.id] = int(step)
                    totals["births"] += 1
            continue

        if task_kind == "T1_json_ab":
            text_a, exp_a = make_text_int()
            text_b, exp_b = make_text_int()
            exp_obj = {"a": int(exp_a), "b": int(exp_b)}

            a, ok_a, cost_a, used_a = plan_try_unary(
                registry=registry,
                birth=birth,
                step=int(step),
                goal_key="goal:parse_int",
                interface=iface_parse_int,
                strategies=[parse_full],
                inputs={"text": text_a},
                expected=int(exp_a),
                context_signature=f"{task_kind}:a␟ep={ep}",
                call_depth=1,
                prefer_safe_after_fail=False,
                last_fail_flag=False,
            )
            step += 1
            b, ok_b, cost_b, used_b = plan_try_unary(
                registry=registry,
                birth=birth,
                step=int(step),
                goal_key="goal:parse_int",
                interface=iface_parse_int,
                strategies=[parse_full],
                inputs={"text": text_b},
                expected=int(exp_b),
                context_signature=f"{task_kind}:b␟ep={ep}",
                call_depth=1,
                prefer_safe_after_fail=False,
                last_fail_flag=False,
            )
            step += 1

            obj = execute_binary_op("make_dict_ab", a, b)
            out_text = execute_unary_pipeline(["json_canonical"], obj)
            vr = run_validator("json_ab_int_exact", out_text, exp_obj)

            baseline_cost = float(parse_full.cost + parse_full.cost + 1.0 + 1.0)
            cost_used = float(cost_a + cost_b + 1.0 + 1.0)
            totals["top_level_cost_used_sum"] += float(cost_used)
            totals["top_level_cost_baseline_sum"] += float(baseline_cost)

            registry.log_primitives(
                step=int(step),
                subgraph_ref={"kind": "top_level_v0", "task": task_kind},
                interface=ConceptInterface(
                    input_schema={"text_a": "str", "text_b": "str"},
                    output_schema={"json": "str"},
                    validator_id="json_ab_int_exact",
                ),
                inputs={"text_a": text_a, "text_b": text_b},
                expected=exp_obj,
                output=out_text,
                validator_result=vr,
                cost_used=float(cost_used),
                baseline_cost=float(baseline_cost),
                context_signature=f"{task_kind}␟ep={ep}",
                call_depth=0,
                note=f"json_path:a={used_a},b={used_b},ok_a={ok_a},ok_b={ok_b}",
            )
            if bool(vr.passed):
                totals["top_level_pass"] += 1
            else:
                totals["top_level_fail"] += 1
            step += 1

            for c in registry.concepts():
                if c.id not in known_concepts:
                    known_concepts[c.id] = int(step)
                    totals["births"] += 1
            continue

    totals["top_level_pass_rate"] = float(totals["top_level_pass"]) / float(max(1, totals["episodes"]))
    totals["avg_cost_used"] = float(totals["top_level_cost_used_sum"]) / float(max(1, totals["episodes"]))
    totals["avg_cost_baseline"] = float(totals["top_level_cost_baseline_sum"]) / float(max(1, totals["episodes"]))
    totals["avg_cost_saved"] = float(totals["avg_cost_baseline"] - totals["avg_cost_used"])

    # Count prune events from evidence log.
    prunes = 0
    if os.path.exists(registry.evidence_path):
        with open(registry.evidence_path, "r", encoding="utf-8") as f:
            for line in f:
                if '"event":"PRUNE"' in line:
                    prunes += 1
    totals["prunes"] = int(prunes)

    # Types + anti-collapse metrics from telemetry
    type_counts: Dict[str, int] = {}
    concept_call_counts: Dict[str, int] = {}
    if os.path.exists(registry.telemetry_path):
        with open(registry.telemetry_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ct = row.get("concept_type")
                if isinstance(ct, str) and ct:
                    type_counts[ct] = type_counts.get(ct, 0) + 1
                if row.get("event") == "CALL":
                    cid = row.get("concept_id")
                    if isinstance(cid, str) and cid:
                        concept_call_counts[cid] = concept_call_counts.get(cid, 0) + 1

    entropy_types = shannon_entropy(type_counts)
    monopoly_alarm = None
    if concept_call_counts:
        top_cid, top_n = sorted(concept_call_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        share = float(top_n) / float(sum(concept_call_counts.values()))
        if share >= 0.8:
            monopoly_alarm = {"concept_id": top_cid, "share": share}

    category_checks = demo_category_checks(list(registry.concepts()))

    concepts_out: List[Dict[str, Any]] = []
    types_out: List[Dict[str, Any]] = []
    for c in sorted(list(registry.concepts()), key=lambda c: str(c.id)):
        concepts_out.append(
            {
                "id": str(c.id),
                "alive": bool(c.alive),
                "type_signature": str(c.interface.type_signature()),
                "validator_id": str(c.interface.validator_id),
                "subgraph_ref": dict(c.subgraph_ref),
                "U_t": float(c.u_ema),
                "K_t": float(c.k_ema),
                "S_t": float(c.s_t),
                "hits": int(c.calls_total),
                "contexts_distinct": int(c.contexts_distinct()),
                "last_seen_step": int(c.last_seen_step),
            }
        )
    for t_sig, n in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        types_out.append({"type_signature": str(t_sig), "calls": int(n)})

    alarms = {
        "entropy_types": float(entropy_types),
        "entropy_collapse": bool(entropy_types < 0.5 and sum(type_counts.values()) >= 10),
        "monopoly": monopoly_alarm,
    }

    summary = {
        "seed": int(args.seed),
        "run_dir": out_dir,
        "chains": registry.verify_chains(),
        "totals": totals,
        "concepts": concepts_out,
        "types": types_out,
        "composition": category_checks,
        "alarms": alarms,
        "paths": {
            "concepts_jsonl": registry.concepts_path,
            "concept_evidence_jsonl": registry.evidence_path,
            "concept_telemetry_jsonl": registry.telemetry_path,
        },
    }

    with open(os.path.join(out_dir, "concept_summary.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n")

    # Basic reproducibility hashes for the run dir.
    summary["sha256"] = {
        "concepts_jsonl": sha256_file(registry.concepts_path) if os.path.exists(registry.concepts_path) else None,
        "concept_evidence_jsonl": sha256_file(registry.evidence_path) if os.path.exists(registry.evidence_path) else None,
        "concept_telemetry_jsonl": sha256_file(registry.telemetry_path) if os.path.exists(registry.telemetry_path) else None,
        "concept_summary_json": sha256_file(os.path.join(out_dir, "concept_summary.json")),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
