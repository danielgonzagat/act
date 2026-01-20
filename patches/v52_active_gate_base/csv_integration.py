from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .concept_miner import ConceptBirthTrigger
from .concepts import Concept, ConceptInterface, ConceptRegistry, stable_hash_obj
from .concepts import ConceptPolicies
from .validators import run_validator


def _safe_str_list(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    return [str(v) for v in x if isinstance(v, str) and str(v)]


def _unique_preserve(xs: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def winners_unique_from_trace(trace: Dict[str, Any]) -> List[str]:
    ids = trace.get("selected_source_act_ids")
    lst = _safe_str_list(ids)
    # Filter engine-only sources.
    out = [x for x in lst if x and x not in {"__engine__", "__unknown__", "__contract__"}]
    return sorted(set(out))


def baseline_scan_cost_per_token_mean(trace: Dict[str, Any]) -> float:
    pred_iter = trace.get("predictor_iterated")
    if not isinstance(pred_iter, list):
        return 0.0
    vals = [int(x) for x in pred_iter if isinstance(x, int) and int(x) > 0]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def executed_predictors_from_turn_subgraph(trace: Dict[str, Any]) -> List[str]:
    sub = trace.get("subgraph") if isinstance(trace.get("subgraph"), dict) else {}
    ids = sub.get("executed_predictor_act_ids") if isinstance(sub, dict) else []
    return _safe_str_list(ids)


def rewrite_hits_from_turn_subgraph(trace: Dict[str, Any]) -> List[str]:
    sub = trace.get("subgraph") if isinstance(trace.get("subgraph"), dict) else {}
    ids = sub.get("rewrite_rule_hit_ids") if isinstance(sub, dict) else []
    return _safe_str_list(ids)


def contract_used_from_trace(trace: Dict[str, Any]) -> bool:
    meta = trace.get("instruction_contract")
    return bool(isinstance(meta, dict) and bool(meta.get("used")))


@dataclass
class CSVLoopConfig:
    mode: str = "shadow"  # "shadow" | "active"
    birth_min_count: int = 5
    birth_window_size: int = 200
    birth_min_pass_rate: float = 0.0
    birth_min_avg_cost: float = 1.0
    candidate_prefix_k: int = 32
    enable_empty_concept: bool = True
    enable_top1_winner_concept: bool = True
    enable_exec_prefix_concept: bool = True
    # Evaluation policy (shadow): call both "stress" and "best" to prove lifecycle.
    eval_stress: bool = True
    eval_best: bool = True


@dataclass
class CSVLoopIntegration:
    registry: ConceptRegistry
    config: CSVLoopConfig = field(default_factory=CSVLoopConfig)
    birth: ConceptBirthTrigger = field(init=False)

    # Metrics for the run (shadow planner).
    turns_observed: int = 0
    turns_contract_used: int = 0
    baseline_cost_sum: float = 0.0
    shadow_cost_sum: float = 0.0
    shadow_calls: int = 0
    shadow_pass: int = 0
    shadow_fail: int = 0

    def __post_init__(self) -> None:
        self.birth = ConceptBirthTrigger(
            window_size=int(self.config.birth_window_size),
            birth_min_count=int(self.config.birth_min_count),
            birth_min_pass_rate=float(self.config.birth_min_pass_rate),
            birth_min_avg_cost=float(self.config.birth_min_avg_cost),
            policies=ConceptPolicies(),
        )

    def _interface(self) -> ConceptInterface:
        return ConceptInterface(
            input_schema={"turn_sig": "str"},
            output_schema={"executed_predictor_act_ids": "list[str]"},
            validator_id="list_contains_all_str",
            preconditions={},
            postconditions={},
        )

    def _candidate_subgraphs(
        self,
        *,
        exec_pred_ids: List[str],
        winners: List[str],
        rr_hit_ids: List[str],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        out: List[Tuple[str, Dict[str, Any]]] = []

        if bool(self.config.enable_empty_concept):
            out.append(
                (
                    "empty_gate",
                    {
                        "kind": "engine_turn_subgraph_v0",
                        "executed_predictor_act_ids": [],
                        "rewrite_rule_hit_ids": [],
                    },
                )
            )

        if bool(self.config.enable_top1_winner_concept) and winners:
            # Take a deterministic representative winner. (sorted -> stable)
            out.append(
                (
                    "top1_winner_gate",
                    {
                        "kind": "engine_turn_subgraph_v0",
                        "executed_predictor_act_ids": [str(sorted(winners)[0])],
                        "rewrite_rule_hit_ids": [],
                    },
                )
            )

        if bool(self.config.enable_exec_prefix_concept) and exec_pred_ids:
            k = max(1, int(self.config.candidate_prefix_k))
            out.append(
                (
                    f"exec_prefix_{k}",
                    {
                        "kind": "engine_turn_subgraph_v0",
                        "executed_predictor_act_ids": list(exec_pred_ids[:k]),
                        "rewrite_rule_hit_ids": list(rr_hit_ids),
                    },
                )
            )

        return out

    def _birth_observe(
        self,
        *,
        step: int,
        context_signature: str,
        subgraph_ref: Dict[str, Any],
        passed: bool,
        cost_used: float,
    ) -> Optional[Concept]:
        iface = self._interface()
        key = stable_hash_obj({"label": "csv_turn", "subgraph_ref": subgraph_ref, "iface": iface.to_dict()})
        return self.birth.observe(
            registry=self.registry,
            key=key,
            step=int(step),
            subgraph_ref=subgraph_ref,
            interface=iface,
            context_signature=str(context_signature),
            passed=bool(passed),
            cost_used=float(cost_used),
        )

    def _pick_best_concept(self, *, iface: ConceptInterface) -> Optional[Concept]:
        alive = [c for c in self.registry.alive_concepts() if c.interface.type_signature() == iface.type_signature()]
        if not alive:
            return None
        # Score: U/(1+K) with deterministic tie-break. Small anti-monopoly penalty by call share.
        total_calls = max(1, sum(int(c.calls_total) for c in alive))
        scored: List[Tuple[float, str, Concept]] = []
        for c in alive:
            share = float(int(c.calls_total)) / float(total_calls)
            score = float(c.u_ema) / (1.0 + float(c.k_ema))
            if share > 0.9:
                score -= 0.05 * (share - 0.9)
            scored.append((score, str(c.id), c))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return scored[0][2]

    def _pick_stress_concept(self, *, iface: ConceptInterface) -> Optional[Concept]:
        alive = [c for c in self.registry.alive_concepts() if c.interface.type_signature() == iface.type_signature()]
        if not alive:
            return None
        # Prefer smallest cost (brittle) to force pruning demonstration; tie-break by id.
        alive.sort(key=lambda c: (float(c.k_ema), str(c.id)))
        return alive[0]

    def observe_turn(
        self,
        *,
        step: int,
        context_signature: str,
        trace: Dict[str, Any],
        utility_passed: Optional[bool] = None,
        suite_kind: str = "chat",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = dict(meta or {})
        self.turns_observed += 1

        contract_used = contract_used_from_trace(trace)
        if bool(contract_used):
            self.turns_contract_used += 1

        winners = winners_unique_from_trace(trace) if not bool(contract_used) else []
        exec_pred_ids = executed_predictors_from_turn_subgraph(trace) if not bool(contract_used) else []
        rr_hit_ids = rewrite_hits_from_turn_subgraph(trace) if not bool(contract_used) else []

        baseline_cost = baseline_scan_cost_per_token_mean(trace) if not bool(contract_used) else 0.0
        self.baseline_cost_sum += float(baseline_cost)

        iface = self._interface()

        # Log baseline "primitives" execution as a trace-subgraph object (observability only).
        base_out = list(exec_pred_ids)
        base_vr = run_validator(iface.validator_id, base_out, winners)
        self.registry.log_primitives(
            step=int(step),
            subgraph_ref={
                "kind": "engine_turn_trace_v0",
                "executed_predictor_act_ids": list(exec_pred_ids),
                "rewrite_rule_hit_ids": list(rr_hit_ids),
            },
            interface=iface,
            inputs={"suite": str(suite_kind), "turn_sig": str(context_signature), **meta},
            expected=list(winners),
            output=base_out,
            validator_result=base_vr,
            cost_used=float(baseline_cost),
            baseline_cost=float(baseline_cost),
            context_signature=str(context_signature),
            call_depth=0,
            note="baseline_trace_only",
        )

        # Birth triggers (purely observational; do not depend on contract mode).
        # For v51 we do not require an external utility validator: default to True.
        birth_pass = bool(utility_passed) if utility_passed is not None else True
        for _label, sub in self._candidate_subgraphs(
            exec_pred_ids=exec_pred_ids, winners=winners, rr_hit_ids=rr_hit_ids
        ):
            self._birth_observe(
                step=int(step),
                context_signature=str(context_signature),
                subgraph_ref=sub,
                passed=bool(birth_pass),
                cost_used=float(baseline_cost),
            )

        # Shadow evaluation: concept calls must never affect generation. If a contract is active,
        # skip any "active" behavior; at most log bypass.
        if str(self.config.mode) not in {"shadow", "active"}:
            return
        if bool(contract_used):
            return

        def _shadow_call(concept: Concept, role: str) -> None:
            nonlocal baseline_cost, winners
            out, vr, cost_used = self.registry.call(
                step=int(step),
                concept_id=str(concept.id),
                inputs={
                    "csv_mode": str(self.config.mode),
                    "csv_role": str(role),
                    "suite": str(suite_kind),
                    "turn_sig": str(context_signature),
                    **meta,
                },
                expected=list(winners),
                context_signature=str(context_signature),
                call_depth=0,
                baseline_cost=float(baseline_cost),
                contract_active=False,
            )
            self.shadow_calls += 1
            if bool(vr.passed):
                self.shadow_pass += 1
            else:
                self.shadow_fail += 1
            self.shadow_cost_sum += float(cost_used)

        if bool(self.config.eval_stress):
            c = self._pick_stress_concept(iface=iface)
            if c is not None:
                _shadow_call(c, role="stress")

        if bool(self.config.eval_best):
            c = self._pick_best_concept(iface=iface)
            if c is not None:
                _shadow_call(c, role="best")
