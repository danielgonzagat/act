from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

from .concepts import Concept, ConceptInterface, ConceptPolicies, ConceptRegistry


@dataclass
class CandidateStats:
    key: str
    subgraph_ref: Dict[str, Any]
    interface: ConceptInterface
    count_window: int = 0
    pass_window: int = 0
    cost_sum_window: float = 0.0
    contexts_window: Dict[str, int] = field(default_factory=dict)

    def pass_rate_window(self) -> float:
        if self.count_window <= 0:
            return 0.0
        return float(self.pass_window) / float(self.count_window)

    def avg_cost_window(self) -> float:
        if self.count_window <= 0:
            return 0.0
        return float(self.cost_sum_window) / float(self.count_window)

    def contexts_distinct_window(self) -> int:
        return len(self.contexts_window)


@dataclass(frozen=True)
class CandidateEvent:
    key: str
    context_signature: str
    passed: bool
    cost_used: float


@dataclass
class ConceptBirthTrigger:
    window_size: int = 50
    birth_min_count: int = 5
    birth_min_pass_rate: float = 0.8
    birth_min_avg_cost: float = 2.0
    policies: ConceptPolicies = field(default_factory=ConceptPolicies)

    _events: Deque[CandidateEvent] = field(default_factory=deque, init=False)
    _stats: Dict[str, CandidateStats] = field(default_factory=dict, init=False)
    _already_defined: Dict[str, str] = field(default_factory=dict, init=False)  # key -> concept_id

    def observe(
        self,
        *,
        registry: Optional[ConceptRegistry] = None,
        key: str,
        step: int,
        subgraph_ref: Dict[str, Any],
        interface: ConceptInterface,
        context_signature: str,
        passed: bool,
        cost_used: float,
    ) -> Optional[Concept]:
        k = str(key)
        ctx = str(context_signature)
        self._events.append(CandidateEvent(key=k, context_signature=ctx, passed=bool(passed), cost_used=float(cost_used)))
        st = self._stats.get(k)
        if st is None:
            st = CandidateStats(key=k, subgraph_ref=subgraph_ref, interface=interface)
            self._stats[k] = st
        st.subgraph_ref = subgraph_ref
        st.interface = interface

        st.count_window += 1
        if bool(passed):
            st.pass_window += 1
        st.cost_sum_window += float(cost_used)
        st.contexts_window[ctx] = st.contexts_window.get(ctx, 0) + 1

        if len(self._events) > int(self.window_size):
            old = self._events.popleft()
            ost = self._stats.get(old.key)
            if ost is not None:
                ost.count_window -= 1
                if bool(old.passed):
                    ost.pass_window -= 1
                ost.cost_sum_window -= float(old.cost_used)
                c = ost.contexts_window.get(old.context_signature, 0) - 1
                if c <= 0:
                    ost.contexts_window.pop(old.context_signature, None)
                else:
                    ost.contexts_window[old.context_signature] = c

        if registry is None:
            return None
        return self.maybe_birth(registry=registry, key=k, step=int(step))

    def maybe_birth(self, *, registry: ConceptRegistry, key: str, step: int) -> Optional[Concept]:
        k = str(key)
        st = self._stats.get(k)
        if st is None:
            return None
        if k in self._already_defined:
            return None
        if st.count_window < int(self.birth_min_count):
            return None
        if st.pass_rate_window() < float(self.birth_min_pass_rate):
            return None
        if st.avg_cost_window() < float(self.birth_min_avg_cost):
            return None

        c = registry.define(
            step=int(step),
            subgraph_ref=st.subgraph_ref,
            interface=st.interface,
            policies=self.policies,
            birth_reason=f"birth_trigger:key={k},count={st.count_window},pass_rate={st.pass_rate_window():.3f},avg_cost={st.avg_cost_window():.3f}",
            birth_prior={
                "u_ema": float(st.pass_rate_window()),
                "pass2_ema": float(st.pass_rate_window()),
                "k_ema": 1.0,
            },
        )
        self._already_defined[k] = str(c.id)
        return c

    def stats_snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, st in self._stats.items():
            out[str(k)] = {
                "count_window": int(st.count_window),
                "pass_rate_window": float(st.pass_rate_window()),
                "avg_cost_window": float(st.avg_cost_window()),
                "contexts_distinct_window": int(st.contexts_distinct_window()),
                "subgraph_ref": st.subgraph_ref,
                "validator_id": str(st.interface.validator_id),
            }
        return out
