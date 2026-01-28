#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import itertools
import json
import multiprocessing
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"ERROR: path already exists (WORM): {path}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=False)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_sha16(obj: Any) -> str:
    return hashlib.sha256(_json_dumps(obj).encode("utf-8")).hexdigest()[:16]


def _append_jsonl(path: str, obj: Any) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(_json_dumps(obj))
        f.write("\n")


def _write_json(path: str, obj: Any) -> None:
    ensure_absent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# MiniGrid imports (lazy)
# ----------------------------


@dataclass(frozen=True)
class MiniGridAPI:
    gym: Any
    minigrid: Any
    FullyObsWrapper: Any
    Actions: Any
    OBJECT_TO_IDX: Dict[str, int]
    IDX_TO_OBJECT: Dict[int, str]
    IDX_TO_COLOR: Dict[int, str]


def _require_minigrid() -> MiniGridAPI:
    try:
        import gymnasium as gym  # type: ignore
        import minigrid  # type: ignore
        from minigrid.core.actions import Actions  # type: ignore
        from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, OBJECT_TO_IDX  # type: ignore
        from minigrid.wrappers import FullyObsWrapper  # type: ignore
    except Exception as e:
        _fail(
            "MiniGrid not available. Create a venv and install:\n"
            "  python3 -m venv .venv && source .venv/bin/activate\n"
            "  pip install -U pip && pip install gymnasium minigrid numpy\n"
            f"Import error: {e}"
        )
    return MiniGridAPI(
        gym=gym,
        minigrid=minigrid,
        FullyObsWrapper=FullyObsWrapper,
        Actions=Actions,
        OBJECT_TO_IDX=dict(OBJECT_TO_IDX),
        IDX_TO_OBJECT=dict(IDX_TO_OBJECT),
        IDX_TO_COLOR=dict(IDX_TO_COLOR),
    )


_WORKER_API: Optional[MiniGridAPI] = None


def _worker_init() -> None:
    global _WORKER_API
    _WORKER_API = _require_minigrid()


def _worker_api() -> MiniGridAPI:
    global _WORKER_API
    if _WORKER_API is None:
        _WORKER_API = _require_minigrid()
    return _WORKER_API


# ----------------------------
# Structural artifacts
# ----------------------------


@dataclass(frozen=True)
class ParsedObject:
    obj_type: str
    color: Optional[str]
    state: int
    pos: Tuple[int, int]  # (x, y)


@dataclass(frozen=True)
class WorldState:
    width: int
    height: int
    agent_pos: Tuple[int, int]
    agent_dir: int
    carrying: Optional[ParsedObject]
    objects: Tuple[ParsedObject, ...]

    def iter_objects(self, obj_type: str) -> Iterable[ParsedObject]:
        for o in self.objects:
            if o.obj_type == obj_type:
                yield o


@dataclass(frozen=True)
class GoalSpec:
    kind: str  # "reach_goal" | "pickup"
    target_type: str
    target_color: Optional[str]


@dataclass(frozen=True)
class ConceptCall:
    name: str
    args: Dict[str, Any]


@dataclass(frozen=True)
class Plan:
    schema_id: str
    schema_name: str
    concepts: Tuple[ConceptCall, ...]
    actions: Tuple[int, ...]  # primitive MiniGrid actions (int enum values)
    schema_calls: Tuple[str, ...] = ()  # nested schema calls (2nd order), for audit/metrics
    operator_calls: Tuple[str, ...] = ()  # dynamic operator invocations (3rd order), for audit/metrics

    @property
    def depth(self) -> int:
        # "Schema + schema-calls + concepts + operator-calls" depth; primitives are execution detail.
        return 1 + len(self.schema_calls) + len(self.concepts) + len(self.operator_calls)


@dataclass
class Schema:
    schema_id: str
    name: str
    signature: Dict[str, Any]
    support: int = 0
    promoted: bool = False
    state: str = "candidate"  # candidate|quarantined|refining|promoted|deprecated
    attempts: int = 0
    failures: int = 0
    origin: str = "schema_missing"  # schema_missing|failure_cluster|seeded
    origin_cluster_id: Optional[str] = None
    components: Tuple[str, ...] = ()
    created_step: int = 0
    last_used_step: int = 0
    ttl: int = 2000
    promotion_attempts: int = 0
    last_promotion_attempt_step: int = -1
    promotion_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "name": self.name,
            "signature": dict(self.signature),
            "support": int(self.support),
            "promoted": bool(self.promoted),
            "state": str(self.state),
            "attempts": int(self.attempts),
            "failures": int(self.failures),
            "origin": str(self.origin),
            "origin_cluster_id": self.origin_cluster_id,
            "components": list(self.components),
            "created_step": int(self.created_step),
            "last_used_step": int(self.last_used_step),
            "ttl": int(self.ttl),
            "promotion_attempts": int(self.promotion_attempts),
            "last_promotion_attempt_step": int(self.last_promotion_attempt_step),
            "promotion_evidence": dict(self.promotion_evidence),
        }


@dataclass
class Operator:
    operator_id: str
    name: str
    impl: str  # built-in implementation key (no codegen)
    signature: Dict[str, Any]
    support: int = 0
    promoted: bool = False
    state: str = "candidate"  # candidate|quarantined|refining|promoted|deprecated
    attempts: int = 0
    failures: int = 0
    origin: str = "failure_cluster"  # failure_cluster|seeded
    origin_cluster_id: Optional[str] = None
    created_step: int = 0
    last_used_step: int = 0
    ttl: int = 2000
    promotion_attempts: int = 0
    last_promotion_attempt_step: int = -1
    promotion_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "operator_id": str(self.operator_id),
            "name": str(self.name),
            "impl": str(self.impl),
            "signature": dict(self.signature),
            "support": int(self.support),
            "promoted": bool(self.promoted),
            "state": str(self.state),
            "attempts": int(self.attempts),
            "failures": int(self.failures),
            "origin": str(self.origin),
            "origin_cluster_id": str(self.origin_cluster_id or ""),
            "created_step": int(self.created_step),
            "last_used_step": int(self.last_used_step),
            "ttl": int(self.ttl),
            "promotion_attempts": int(self.promotion_attempts),
            "last_promotion_attempt_step": int(self.last_promotion_attempt_step),
            "promotion_evidence": dict(self.promotion_evidence),
        }


@dataclass
class OperatorBank:
    operators: Dict[str, Operator] = field(default_factory=dict)
    promote_support: int = 3
    ttl_steps: int = 2000
    min_support_to_keep: int = 1
    quarantine_min_attempts: int = 25
    quarantine_max_success_ratio: float = 0.05

    def get(self, operator_id: str) -> Optional[Operator]:
        return self.operators.get(operator_id)

    def add(self, op: Operator) -> bool:
        if op.operator_id in self.operators:
            return False
        self.operators[op.operator_id] = op
        return True

    def mark_success(self, operator_id: str, step: int) -> bool:
        op = self.operators.get(operator_id)
        if not op:
            return False
        was_promoted = bool(op.promoted)
        op.attempts += 1
        op.support += 1
        op.last_used_step = step
        if op.state in ("candidate", "quarantined") and op.support > 0:
            op.state = "refining"
        eligible = (not op.promoted) and op.support >= self.promote_support
        return bool((not was_promoted) and eligible)

    def mark_failure(self, operator_id: str, step: int) -> None:
        op = self.operators.get(operator_id)
        if not op:
            return
        op.attempts += 1
        op.failures += 1
        op.last_used_step = step
        if op.state in ("candidate", "refining") and op.attempts >= self.quarantine_min_attempts:
            succ = max(0, int(op.attempts) - int(op.failures))
            succ_ratio = float(succ / max(1, int(op.attempts)))
            if succ_ratio <= float(self.quarantine_max_success_ratio):
                op.state = "quarantined"
        # Induced operators start in quarantined state. If they never achieve success under repeated
        # use, they must die (otherwise they remain a live escape hatch / noise source forever).
        if op.state == "quarantined" and op.support <= 0 and op.attempts >= self.quarantine_min_attempts:
            op.state = "deprecated"

    def promote(self, operator_id: str, step: int, evidence: Dict[str, Any]) -> bool:
        op = self.operators.get(operator_id)
        if not op or op.promoted:
            return False
        op.promoted = True
        op.state = "promoted"
        op.last_used_step = step
        op.promotion_evidence = dict(evidence)
        return True

    def gc(self, step: int, *, protected_operator_ids: Optional[set[str]] = None) -> List[str]:
        removed: List[str] = []
        protected_operator_ids = protected_operator_ids or set()
        for oid, op in list(self.operators.items()):
            if oid in protected_operator_ids:
                continue
            if op.state in ("quarantined", "refining", "promoted"):
                continue
            age = step - int(op.last_used_step or op.created_step)
            if age >= self.ttl_steps and op.support < self.min_support_to_keep:
                removed.append(oid)
                del self.operators[oid]
        return removed

    def stats(self) -> Dict[str, Any]:
        from collections import Counter

        promoted = sum(1 for op in self.operators.values() if op.promoted)
        state_counts = Counter(str(op.state) for op in self.operators.values())
        max_support = max([op.support for op in self.operators.values()], default=0)
        attempts_total = sum(int(op.attempts) for op in self.operators.values())
        failures_total = sum(int(op.failures) for op in self.operators.values())
        return {
            "operators_total": len(self.operators),
            "operators_promoted": promoted,
            "operators_by_state": dict(state_counts),
            "operators_max_support": max_support,
            "operators_attempts_total": attempts_total,
            "operators_failures_total": failures_total,
        }


@dataclass
class SchemaBank:
    schemas: Dict[str, Schema] = field(default_factory=dict)
    promote_support: int = 3
    ttl_steps: int = 2000
    min_support_to_keep: int = 1
    quarantine_min_attempts: int = 25
    quarantine_max_success_ratio: float = 0.05

    def get(self, schema_id: str) -> Optional[Schema]:
        return self.schemas.get(schema_id)

    def find_by_name(self, name: str) -> Optional[Schema]:
        for s in self.schemas.values():
            if s.name == name:
                return s
        return None

    def add(self, schema: Schema) -> bool:
        if schema.schema_id in self.schemas:
            return False
        self.schemas[schema.schema_id] = schema
        return True

    def mark_success(self, schema_id: str, step: int) -> bool:
        s = self.schemas.get(schema_id)
        if not s:
            return False
        was_promoted = bool(s.promoted)
        s.attempts += 1
        s.support += 1
        s.last_used_step = step
        if s.state in ("candidate", "quarantined") and s.support > 0:
            s.state = "refining"
        eligible = (not s.promoted) and s.support >= self.promote_support
        # NOTE: promotion is gated by preserves_future() in the main loop; do not auto-promote here.
        return bool((not was_promoted) and eligible)

    def mark_failure(self, schema_id: str, step: int) -> None:
        s = self.schemas.get(schema_id)
        if not s:
            return
        s.attempts += 1
        s.failures += 1
        s.last_used_step = step
        # If a schema keeps failing but is structurally recurring, quarantine it
        # (do not allow naive GC to delete "hard learning").
        if s.state in ("candidate", "refining") and s.attempts >= self.quarantine_min_attempts:
            succ = max(0, int(s.attempts) - int(s.failures))
            succ_ratio = float(succ / max(1, int(s.attempts)))
            if succ_ratio <= float(self.quarantine_max_success_ratio):
                s.state = "quarantined"

    def promote(self, schema_id: str, step: int, evidence: Dict[str, Any]) -> bool:
        s = self.schemas.get(schema_id)
        if not s or s.promoted:
            return False
        s.promoted = True
        s.state = "promoted"
        s.last_used_step = step
        s.promotion_evidence = dict(evidence)
        return True

    def gc(self, step: int, *, protected_schema_ids: Optional[set[str]] = None) -> List[str]:
        removed: List[str] = []
        protected_schema_ids = protected_schema_ids or set()
        for sid, s in list(self.schemas.items()):
            if sid in protected_schema_ids:
                continue
            if s.state in ("quarantined", "refining", "promoted"):
                continue
            age = step - int(s.last_used_step or s.created_step)
            if age >= self.ttl_steps and s.support < self.min_support_to_keep:
                removed.append(sid)
                del self.schemas[sid]
        return removed

    def stats(self) -> Dict[str, Any]:
        from collections import Counter

        promoted = sum(1 for s in self.schemas.values() if s.promoted)
        state_counts = Counter(str(s.state) for s in self.schemas.values())
        max_support = max([s.support for s in self.schemas.values()], default=0)
        attempts_total = sum(int(s.attempts) for s in self.schemas.values())
        failures_total = sum(int(s.failures) for s in self.schemas.values())
        return {
            "schemas_total": len(self.schemas),
            "schemas_promoted": promoted,
            "schemas_by_state": dict(state_counts),
            "schemas_max_support": max_support,
            "schemas_attempts_total": attempts_total,
            "schemas_failures_total": failures_total,
        }


@dataclass
class FailureCluster:
    cluster_id: str
    signature: Dict[str, Any]
    count: int = 0
    first_step: int = 0
    last_step: int = 0
    last_progress_step: int = -1
    induce_attempts: int = 0
    last_induce_step: int = -1
    induced_schema_ids: List[str] = field(default_factory=list)
    induced_operator_ids: List[str] = field(default_factory=list)
    # Deterministic witness episodes (env_id, seed) where this signature was observed.
    # Used for MAXWELL_Ω immediate validation/promotion (no per-task hacks; bounded; stable order).
    witnesses: List[Dict[str, Any]] = field(default_factory=list)
    resolved: bool = False
    dead_end: bool = False


@dataclass
class FailureClusterStore:
    threshold: int = 10
    cooldown_steps: int = 250
    max_induce_attempts: int = 3
    clusters: Dict[str, FailureCluster] = field(default_factory=dict)
    by_ctx: Dict[str, set[str]] = field(default_factory=dict)

    def observe(
        self,
        signature: Dict[str, Any],
        *,
        step: int,
        witness_env_id: Optional[str] = None,
        witness_seed: Optional[int] = None,
    ) -> FailureCluster:
        cid = f"fc_{_stable_sha16(signature)}"
        c = self.clusters.get(cid)
        if c is None:
            c = FailureCluster(cluster_id=cid, signature=dict(signature), count=0, first_step=step, last_step=step)
            self.clusters[cid] = c
            ctx = str(signature.get("cluster_ctx_key") or "")
            if ctx:
                self.by_ctx.setdefault(ctx, set()).add(cid)
        c.count += 1
        c.last_step = step
        env_id = str(witness_env_id or "")
        if env_id:
            try:
                seed = int(witness_seed or 0)
            except Exception:
                seed = 0
            w = {"env_id": str(env_id), "seed": int(seed)}
            try:
                c.witnesses = _sorted_unique_ablation_witnesses(list(c.witnesses) + [w])[:25]
            except Exception:
                # Fail-closed: keep the existing witness list if anything goes wrong.
                pass
        return c

    def mark_ctx_resolved(self, cluster_ctx_key: str) -> None:
        ids = self.by_ctx.get(str(cluster_ctx_key)) or set()
        for cid in ids:
            c = self.clusters.get(cid)
            # A context is only considered "resolved" after structural progress was attempted
            # (schema/operator induced). Otherwise, recurring failures can be masked by intermittent successes.
            if c is not None and int(c.last_progress_step) >= 0:
                c.resolved = True

    def protected_schema_ids(self) -> set[str]:
        out: set[str] = set()
        for c in self.clusters.values():
            if c.dead_end or c.resolved:
                continue
            for sid in c.induced_schema_ids:
                out.add(str(sid))
        return out

    def protected_operator_ids(self) -> set[str]:
        out: set[str] = set()
        for c in self.clusters.values():
            if c.dead_end or c.resolved:
                continue
            for oid in c.induced_operator_ids:
                out.add(str(oid))
        return out

    def eligible_clusters(self, *, step: int) -> List[FailureCluster]:
        # Deterministic order for induction attempts.
        out: List[FailureCluster] = []
        for c in self.clusters.values():
            if c.dead_end or c.resolved:
                continue
            # Missing-operator failures are existentially urgent under MAXWELL_Ω strict burn:
            # wait-for-recurrence would force Ω to burn before INDUCE_OPERATOR can fire.
            try:
                reason = str(c.signature.get("reason") or "")
            except Exception:
                reason = ""
            thr = int(self.threshold)
            if reason.startswith("missing_operator_") or reason in ("unbox_key_no_matching_key",):
                thr = min(int(thr), 1)
            if c.count < int(thr):
                continue
            if c.induce_attempts >= int(self.max_induce_attempts):
                continue
            if c.last_induce_step >= 0 and (int(step) - int(c.last_induce_step)) < int(self.cooldown_steps):
                continue
            out.append(c)
        out.sort(key=lambda c: (-int(c.count), str(c.cluster_id)))
        return out


# ----------------------------
# Ω (structural) — destructive future memory (WORM)
# ----------------------------


def _load_schema_bank_into(bank: SchemaBank, path: str) -> None:
    raw = _read_json(path)
    schemas = raw.get("schemas") if isinstance(raw, dict) else None
    if not isinstance(schemas, list):
        return
    # Deterministic insertion order.
    schemas_sorted = []
    for d in schemas:
        if not isinstance(d, dict):
            continue
        sid = str(d.get("schema_id") or "")
        if not sid:
            continue
        schemas_sorted.append(d)
    schemas_sorted.sort(key=lambda d: str(d.get("schema_id") or ""))

    for d in schemas_sorted:
        sid = str(d.get("schema_id") or "")
        if not sid:
            continue
        name = str(d.get("name") or "")
        sig = d.get("signature") if isinstance(d.get("signature"), dict) else {}
        promoted = bool(d.get("promoted", False))
        state = str(d.get("state") or ("promoted" if promoted else "candidate"))
        if promoted:
            state = "promoted"
        s = Schema(
            schema_id=sid,
            name=name,
            signature=dict(sig),
            support=int(d.get("support") or 0),
            promoted=promoted,
            state=state,
            attempts=int(d.get("attempts") or 0),
            failures=int(d.get("failures") or 0),
            origin=str(d.get("origin") or "seeded"),
            origin_cluster_id=str(d.get("origin_cluster_id") or "") or None,
            components=tuple(d.get("components") or ()),
            created_step=0,
            last_used_step=0,
            ttl=int(d.get("ttl") or 2000),
            promotion_attempts=int(d.get("promotion_attempts") or 0),
            last_promotion_attempt_step=-1,
            promotion_evidence=dict(d.get("promotion_evidence") or {}),
        )
        bank.add(s)


def _load_operator_bank_into(op_bank: OperatorBank, path: str) -> None:
    raw = _read_json(path)
    ops = raw.get("operators") if isinstance(raw, dict) else None
    if not isinstance(ops, list):
        return
    ops_sorted = []
    for d in ops:
        if not isinstance(d, dict):
            continue
        oid = str(d.get("operator_id") or "")
        if not oid:
            continue
        ops_sorted.append(d)
    ops_sorted.sort(key=lambda d: str(d.get("operator_id") or ""))

    for d in ops_sorted:
        oid = str(d.get("operator_id") or "")
        if not oid:
            continue
        name = str(d.get("name") or "")
        impl = str(d.get("impl") or "")
        sig = d.get("signature") if isinstance(d.get("signature"), dict) else {}
        promoted = bool(d.get("promoted", False))
        state = str(d.get("state") or ("promoted" if promoted else "candidate"))
        if promoted:
            state = "promoted"
        op = Operator(
            operator_id=oid,
            name=name,
            impl=impl,
            signature=dict(sig),
            support=int(d.get("support") or 0),
            promoted=promoted,
            state=state,
            attempts=int(d.get("attempts") or 0),
            failures=int(d.get("failures") or 0),
            origin=str(d.get("origin") or "seeded"),
            origin_cluster_id=str(d.get("origin_cluster_id") or "") or None,
            created_step=0,
            last_used_step=0,
            ttl=int(d.get("ttl") or 2000),
            promotion_attempts=int(d.get("promotion_attempts") or 0),
            last_promotion_attempt_step=-1,
            promotion_evidence=dict(d.get("promotion_evidence") or {}),
        )
        op_bank.add(op)


@dataclass(frozen=True)
class OmegaStateStructuralV1:
    kind: str
    created_at: str
    prev_state_sha: str
    state_sha: str
    strict_burns_total: int
    banned_env_ids: Tuple[str, ...]
    banned_cluster_ctx_keys: Tuple[str, ...]

    def to_json(self) -> Dict[str, Any]:
        return {
            "kind": str(self.kind),
            "created_at": str(self.created_at),
            "prev_state_sha": str(self.prev_state_sha or ""),
            "state_sha": str(self.state_sha or ""),
            "strict_burns_total": int(self.strict_burns_total),
            "banned_env_ids": sorted({str(x) for x in self.banned_env_ids if str(x)}),
            "banned_cluster_ctx_keys": sorted({str(x) for x in self.banned_cluster_ctx_keys if str(x)}),
        }

    def content_hash(self) -> str:
        raw = self.to_json()
        raw.pop("state_sha", None)
        return hashlib.sha256(_json_dumps(raw).encode("utf-8")).hexdigest()

    @staticmethod
    def from_path(path: str) -> "OmegaStateStructuralV1":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        st = OmegaStateStructuralV1(
            kind=str(raw.get("kind") or "omega_state_structural_v1"),
            created_at=str(raw.get("created_at") or ""),
            prev_state_sha=str(raw.get("prev_state_sha") or ""),
            state_sha=str(raw.get("state_sha") or ""),
            strict_burns_total=int(raw.get("strict_burns_total") or 0),
            banned_env_ids=tuple(str(x) for x in (raw.get("banned_env_ids") or []) if str(x)),
            banned_cluster_ctx_keys=tuple(str(x) for x in (raw.get("banned_cluster_ctx_keys") or []) if str(x)),
        )
        want = st.content_hash()
        got = str(st.state_sha or "")
        if want != got:
            raise ValueError(f"omega_state_structural_hash_mismatch:want={want},got={got}")
        return st


# ----------------------------
# Parallel episode runner
# ----------------------------


def _freeze_schema_snapshot(bank: SchemaBank) -> Dict[str, Any]:
    # Snapshot must be deterministic and read-only for workers.
    schemas: List[Dict[str, Any]] = []
    for s in sorted(bank.schemas.values(), key=lambda s: str(s.schema_id)):
        schemas.append(
            {
                "schema_id": str(s.schema_id),
                "name": str(s.name),
                "signature": dict(s.signature),
                "promoted": bool(s.promoted),
                "state": str(s.state),
                "components": list(s.components),
            }
        )
    return {"schemas": schemas}


def _freeze_operator_snapshot(op_bank: OperatorBank) -> Dict[str, Any]:
    ops: List[Dict[str, Any]] = []
    for op in sorted(op_bank.operators.values(), key=lambda o: str(o.operator_id)):
        ops.append(
            {
                "operator_id": str(op.operator_id),
                "name": str(op.name),
                "impl": str(op.impl),
                "signature": dict(op.signature),
                "promoted": bool(op.promoted),
                "state": str(op.state),
            }
        )
    return {"operators": ops}


def _schema_id_for(schema_name: str, signature: Dict[str, Any]) -> str:
    return f"schema_{_stable_sha16({'name': schema_name, 'signature': signature})}"


def _run_episode_core(api: MiniGridAPI, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure per-episode execution with a frozen schema snapshot.
    No disk writes. No global state writes.
    """
    env_id = str(payload["env_id"])
    seed = int(payload["seed"])
    schema_defs_raw = payload.get("schemas") or []
    schema_defs: List[Dict[str, Any]] = []
    if isinstance(schema_defs_raw, list):
        for d in schema_defs_raw:
            if isinstance(d, dict):
                schema_defs.append(d)
    schema_ids = {str(d.get("schema_id")) for d in schema_defs if str(d.get("schema_id") or "")}

    operator_defs_raw = payload.get("operators") or []
    operator_defs: List[Dict[str, Any]] = []
    if isinstance(operator_defs_raw, list):
        for d in operator_defs_raw:
            if isinstance(d, dict):
                operator_defs.append(d)
    operator_ids = {str(d.get("operator_id")) for d in operator_defs if str(d.get("operator_id") or "")}
    min_plan_depth = int(payload["min_plan_depth"])
    forbid_shallow_paths = bool(payload["forbid_shallow_paths"])
    fail_closed_required = bool(payload["fail_closed_required"])
    omega_banned_env_ids = {str(x) for x in (payload.get("omega_banned_env_ids") or []) if str(x)}
    omega_banned_cluster_ctx_keys = {str(x) for x in (payload.get("omega_banned_cluster_ctx_keys") or []) if str(x)}
    omega_skip_checks = bool(payload.get("omega_skip_checks", False))

    if (not omega_skip_checks) and env_id in omega_banned_env_ids:
        return {
            "env_id": env_id,
            "seed": seed,
            "episode_skipped": True,
            "skip_reason": "OMEGA_BANNED_ENV",
            "outcome": "OMEGA_SKIP",
            "reason": "OMEGA_BANNED_ENV",
            "did_fail_closed": True,
            "executed_with_plan": False,
            "episode_success": False,
            "terminated": False,
            "truncated": False,
            "steps_taken": 0,
            "plan_depth": 0,
            "concept_calls": 0,
            "schema_calls": [],
            "operator_calls": [],
            "schema_selected": None,
            "schema_selected_id": None,
            "schema_missing": False,
            "schema_suggestion": None,
            "schema_used_id": None,
            "schema_used_name": None,
            "world_features": None,
            "cluster_ctx_key": None,
            "pre_search_schema": False,
            "object_centric_ok": False,
            "conditional_ok": False,
            "hierarchical_ok": False,
            "shallow_suppressed": False,
            "failure_to_schema": False,
            "plan_fail_reason": None,
            "plan_fail_concept": None,
            "plan_fail_sim": None,
            "ctx_key": None,
            "goal": None,
        }

    # Capture as much context as possible so any exception stays inside the ontological cycle.
    stage = "init"
    world_features: Optional[Dict[str, Any]] = None
    cluster_ctx_key: Optional[str] = None
    ctx_key: Optional[str] = None
    schema_name: Optional[str] = None
    schema_id: Optional[str] = None
    goal_json: Optional[Dict[str, Any]] = None
    plan_fail_facts: Optional[Dict[str, Any]] = None
    pre_search_schema = False
    object_centric_ok = False
    shallow_suppressed = False
    conditional_ok = False
    hierarchical_ok = False

    env = api.FullyObsWrapper(api.gym.make(env_id, render_mode=None))
    try:
        stage = "reset"
        obs, _ = env.reset(seed=seed)

        stage = "parse"
        img = obs["image"]
        objects = _parse_objects(img, api)
        carrying_obj = None
        if env.unwrapped.carrying is not None:
            carrying_obj = ParsedObject(
                obj_type=getattr(env.unwrapped.carrying, "type", "unknown"),
                color=getattr(env.unwrapped.carrying, "color", None),
                state=0,
                pos=(-1, -1),
            )
        world = WorldState(
            width=int(env.unwrapped.width),
            height=int(env.unwrapped.height),
            agent_pos=tuple(int(x) for x in env.unwrapped.agent_pos),
            agent_dir=int(env.unwrapped.agent_dir),
            carrying=carrying_obj,
            objects=objects,
        )

        stage = "goal"
        goal = _derive_goal(obs, world)
        if goal is None:
            return {
                "env_id": env_id,
                "seed": seed,
                "outcome": "FAIL_CLOSED",
                "reason": "NO_GOAL",
                "did_fail_closed": True,
                "executed_with_plan": False,
                "episode_success": False,
                "terminated": False,
                "truncated": False,
                "steps_taken": 0,
                "plan_depth": 0,
                "concept_calls": 0,
                "schema_calls": [],
                "operator_calls": [],
                "schema_selected": None,
                "schema_selected_id": None,
                "schema_missing": False,
                "schema_suggestion": None,
                "schema_used_id": None,
                "schema_used_name": None,
                "world_features": None,
                "cluster_ctx_key": None,
                "pre_search_schema": False,
                "object_centric_ok": False,
                "conditional_ok": False,
                "hierarchical_ok": False,
                "shallow_suppressed": False,
                "failure_to_schema": False,
                "plan_fail_reason": None,
                "plan_fail_concept": None,
                "plan_fail_sim": None,
                "ctx_key": None,
                "goal": None,
            }
        goal_json = dataclasses.asdict(goal)

        stage = "features"
        world_features = _world_features(world, goal)
        cluster_ctx_key = _cluster_ctx_key(world_features)
        ctx_key = _stable_sha16({"env_id": env_id, "cluster_ctx_key": cluster_ctx_key})

        if (not omega_skip_checks) and cluster_ctx_key in omega_banned_cluster_ctx_keys:
            return {
                "env_id": env_id,
                "seed": seed,
                "episode_skipped": True,
                "skip_reason": "OMEGA_BANNED_CLUSTER_CTX",
                "outcome": "OMEGA_SKIP",
                "reason": "OMEGA_BANNED_CLUSTER_CTX",
                "did_fail_closed": True,
                "executed_with_plan": False,
                "episode_success": False,
                "terminated": False,
                "truncated": False,
                "steps_taken": 0,
                "plan_depth": 0,
                "concept_calls": 0,
                "schema_calls": [],
                "operator_calls": [],
                "schema_selected": None,
                "schema_selected_id": None,
                "schema_missing": False,
                "schema_suggestion": None,
                "schema_used_id": None,
                "schema_used_name": None,
                "world_features": dict(world_features),
                "cluster_ctx_key": cluster_ctx_key,
                "pre_search_schema": False,
                "object_centric_ok": False,
                "conditional_ok": False,
                "hierarchical_ok": False,
                "shallow_suppressed": False,
                "failure_to_schema": False,
                "plan_fail_reason": None,
                "plan_fail_concept": None,
                "plan_fail_sim": None,
                "ctx_key": ctx_key,
                "goal": goal_json,
            }

        stage = "schema"
        desired_schema_name, desired_signature = _schema_for(world, goal)
        desired_schema_id = _schema_id_for(desired_schema_name, desired_signature)
        pre_search_schema = True
        object_centric_ok = True
        shallow_suppressed = bool(forbid_shallow_paths and goal.kind == "reach_goal")

        def match_score(schema_sig: Dict[str, Any], desired_sig: Dict[str, Any]) -> int:
            score = 0
            for k, v in desired_sig.items():
                if schema_sig.get(k) == v:
                    score += 1
            return int(score)

        candidates: List[Tuple[Tuple[int, int, str], Dict[str, Any]]] = []
        for sd in schema_defs:
            sid = str(sd.get("schema_id") or "")
            nm = str(sd.get("name") or "")
            ssig = sd.get("signature")
            if (not sid) or (not nm) or (not isinstance(ssig, dict)):
                continue
            score = match_score(ssig, desired_signature)
            promoted_flag = 1 if bool(sd.get("promoted")) else 0
            candidates.append(((-int(score), -int(promoted_flag), sid), sd))
        candidates.sort(key=lambda t: t[0])

        chosen_schema_id: Optional[str] = None
        chosen_schema_name: Optional[str] = None
        chosen_plan: Optional[Plan] = None
        chosen_facts: Dict[str, Any] = {}
        best_failed_schema_id: Optional[str] = None
        best_failed_schema_name: Optional[str] = None
        best_failed_facts: Optional[Dict[str, Any]] = None
        for _, sd in candidates:
            sid = str(sd.get("schema_id") or "")
            nm = str(sd.get("name") or "")
            if (not sid) or (not nm) or (sid not in schema_ids):
                continue
            plan_try, facts_try = _plan_with_schema(
                api=api,
                env=env,
                world=world,
                goal=goal,
                env_id=env_id,
                seed=seed,
                schema_id=sid,
                schema_name=nm,
                operators=operator_defs,
                operator_ids=operator_ids,
                min_plan_depth=min_plan_depth,
            )
            if plan_try is not None:
                chosen_schema_id = sid
                chosen_schema_name = nm
                chosen_plan = plan_try
                chosen_facts = facts_try
                break
            if isinstance(facts_try, dict) and str(facts_try.get("plan_fail_reason") or ""):
                if best_failed_facts is None:
                    best_failed_schema_id = sid
                    best_failed_schema_name = nm
                    best_failed_facts = dict(facts_try)

        if chosen_plan is None or chosen_schema_id is None or chosen_schema_name is None:
            if desired_schema_id not in schema_ids:
                return {
                    "env_id": env_id,
                    "seed": seed,
                    "outcome": "FAIL_CLOSED",
                    "reason": "SCHEMA_MISSING",
                    "did_fail_closed": bool(fail_closed_required),
                    "executed_with_plan": False,
                    "episode_success": False,
                    "terminated": False,
                    "truncated": False,
                    "steps_taken": 0,
                    "plan_depth": 0,
                    "concept_calls": 0,
                    "schema_calls": [],
                    "operator_calls": [],
                    "schema_selected": desired_schema_name,
                    "schema_selected_id": desired_schema_id,
                    "schema_missing": True,
                    "schema_suggestion": {
                        "schema_id": desired_schema_id,
                        "name": desired_schema_name,
                        "signature": desired_signature,
                    },
                    "schema_used_id": None,
                    "schema_used_name": None,
                    "world_features": dict(world_features),
                    "cluster_ctx_key": cluster_ctx_key,
                    "pre_search_schema": pre_search_schema,
                    "object_centric_ok": object_centric_ok,
                    "conditional_ok": False,
                    "hierarchical_ok": False,
                    "shallow_suppressed": shallow_suppressed,
                    "failure_to_schema": True,
                    "plan_fail_reason": None,
                    "plan_fail_concept": None,
                    "plan_fail_sim": None,
                    "ctx_key": ctx_key,
                    "goal": goal_json,
                }

            # Schema exists but planning failed (missing operator / unreachable under current operator set).
            pf = best_failed_facts or {}
            plan_fail_reason = str(pf.get("plan_fail_reason") or "NO_PLAN")
            return {
                "env_id": env_id,
                "seed": seed,
                "outcome": "FAIL_CLOSED",
                "reason": plan_fail_reason,
                "did_fail_closed": bool(fail_closed_required),
                "executed_with_plan": False,
                "episode_success": False,
                "terminated": False,
                "truncated": False,
                "steps_taken": 0,
                "plan_depth": 0,
                "concept_calls": 0,
                "schema_calls": [],
                "operator_calls": pf.get("operator_calls") or [],
                "schema_selected": best_failed_schema_name or desired_schema_name,
                "schema_selected_id": best_failed_schema_id or desired_schema_id,
                "schema_missing": False,
                "schema_suggestion": None,
                "schema_used_id": None,
                "schema_used_name": None,
                "world_features": dict(world_features),
                "cluster_ctx_key": cluster_ctx_key,
                "pre_search_schema": pre_search_schema,
                "object_centric_ok": object_centric_ok,
                "conditional_ok": False,
                "hierarchical_ok": False,
                "shallow_suppressed": shallow_suppressed,
                "failure_to_schema": False,
                "plan_fail_reason": plan_fail_reason,
                "plan_fail_concept": pf.get("plan_fail_concept"),
                "plan_fail_sim": pf.get("plan_fail_sim"),
                "ctx_key": ctx_key,
                "goal": goal_json,
            }

        # Plan using the schema (schema -> concepts -> operators), then execute.
        stage = "plan"
        schema_id = chosen_schema_id
        schema_name = chosen_schema_name
        plan = chosen_plan
        facts = chosen_facts
        plan_fail_facts = dict(facts) if isinstance(facts, dict) else None
        conditional_ok = bool(facts.get("conditional_ok", False))
        hierarchical_ok = bool(facts.get("hierarchical_ok", False))

        if plan is None:
            plan_fail_reason = str(facts.get("plan_fail_reason") or "NO_PLAN")
            return {
                "env_id": env_id,
                "seed": seed,
                "outcome": "FAIL_CLOSED",
                "reason": plan_fail_reason,
                "did_fail_closed": bool(fail_closed_required),
                "executed_with_plan": False,
                "episode_success": False,
                "terminated": False,
                "truncated": False,
                "steps_taken": 0,
                "plan_depth": 0,
                "concept_calls": 0,
                "schema_calls": [],
                "operator_calls": facts.get("operator_calls") or [],
                "schema_selected": schema_name,
                "schema_selected_id": schema_id,
                "schema_missing": False,
                "schema_suggestion": None,
                "schema_used_id": None,
                "schema_used_name": None,
                "world_features": dict(world_features),
                "cluster_ctx_key": cluster_ctx_key,
                "pre_search_schema": pre_search_schema,
                "object_centric_ok": object_centric_ok,
                "conditional_ok": conditional_ok,
                "hierarchical_ok": hierarchical_ok,
                "shallow_suppressed": shallow_suppressed,
                "failure_to_schema": False,
                "plan_fail_reason": plan_fail_reason,
                "plan_fail_concept": facts.get("plan_fail_concept"),
                "plan_fail_sim": facts.get("plan_fail_sim"),
                "ctx_key": ctx_key,
                "goal": goal_json,
            }

        terminated = False
        truncated = False
        steps_taken = 0
        stage = "execute"
        for a in plan.actions:
            _, _, terminated, truncated, _ = env.step(int(a))
            steps_taken += 1
            if terminated or truncated:
                break

        episode_success = bool(terminated) or _episode_solved(env, goal)

        return {
            "env_id": env_id,
            "seed": seed,
            "outcome": "SUCCESS" if episode_success else "FAIL",
            "reason": None if episode_success else "PLAN_FAILED",
            "did_fail_closed": False,
            "executed_with_plan": True,
            "episode_success": bool(episode_success),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "steps_taken": int(steps_taken),
            "plan_depth": int(plan.depth),
            "concept_calls": int(len(plan.concepts)),
            "schema_calls": list(plan.schema_calls),
            "operator_calls": list(plan.operator_calls),
            "schema_selected": schema_name,
            "schema_selected_id": schema_id,
            "schema_missing": False,
            "schema_suggestion": None,
            "schema_used_id": schema_id if episode_success else None,
            "schema_used_name": schema_name if episode_success else None,
            "world_features": dict(world_features),
            "cluster_ctx_key": cluster_ctx_key,
            "pre_search_schema": pre_search_schema,
            "object_centric_ok": object_centric_ok,
            "conditional_ok": conditional_ok,
            "hierarchical_ok": hierarchical_ok,
            "shallow_suppressed": shallow_suppressed,
            "failure_to_schema": False,
            "plan_fail_reason": None if episode_success else (plan_fail_facts or {}).get("plan_fail_reason"),
            "plan_fail_concept": None if episode_success else (plan_fail_facts or {}).get("plan_fail_concept"),
            "plan_fail_sim": None if episode_success else (plan_fail_facts or {}).get("plan_fail_sim"),
            "ctx_key": ctx_key,
            "goal": goal_json,
        }
    except Exception as e:
        exception_type = type(e).__name__
        exception_stage = str(stage)
        # Ensure any exception is clusterizable even if it happens before goal parsing.
        exc_cluster_ctx_key = cluster_ctx_key or _stable_sha16(
            {"exception_stage": exception_stage, "exception_type": exception_type}
        )
        exc_ctx_key = ctx_key or _stable_sha16({"env_id": env_id, "cluster_ctx_key": exc_cluster_ctx_key})
        return {
            "env_id": env_id,
            "seed": seed,
            "outcome": "FAIL_CLOSED",
            "reason": f"EXCEPTION_{exception_stage}_{exception_type}",
            "exception_type": exception_type,
            "exception_msg": str(e),
            "exception_trace": traceback.format_exc(limit=50),
            "did_fail_closed": True,
            "executed_with_plan": False,
            "episode_success": False,
            "terminated": False,
            "truncated": False,
            "steps_taken": 0,
            "plan_depth": 0,
            "concept_calls": 0,
            "schema_calls": [],
            "operator_calls": [],
            "schema_selected": schema_name,
            "schema_selected_id": schema_id,
            "schema_missing": False,
            "schema_suggestion": None,
            "schema_used_id": None,
            "schema_used_name": None,
            "world_features": (dict(world_features) if isinstance(world_features, dict) else None),
            "cluster_ctx_key": exc_cluster_ctx_key,
            "pre_search_schema": bool(pre_search_schema),
            "object_centric_ok": bool(object_centric_ok),
            "conditional_ok": bool(conditional_ok),
            "hierarchical_ok": bool(hierarchical_ok),
            "shallow_suppressed": bool(shallow_suppressed),
            "failure_to_schema": False,
            "plan_fail_reason": None,
            "plan_fail_concept": None,
            "plan_fail_sim": None,
            "ctx_key": exc_ctx_key,
            "goal": goal_json,
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def _run_episode_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    api = _worker_api()
    return _run_episode_core(api, payload)


# ----------------------------
# Metrics Z(t)
# ----------------------------


@dataclass
class Rolling:
    window: int
    vals: List[float] = field(default_factory=list)

    def push(self, v: float) -> None:
        self.vals.append(float(v))
        if len(self.vals) > self.window:
            self.vals.pop(0)

    def mean(self) -> float:
        if not self.vals:
            return 0.0
        return float(sum(self.vals) / len(self.vals))


@dataclass
class StructuralMetrics:
    window: int = 1000
    steps: int = 0

    successes: int = 0
    fail_closed: int = 0
    pre_search_schema: int = 0
    executed_with_plan: int = 0
    object_centric_ok: int = 0
    conditional_ok: int = 0
    hierarchical_ok: int = 0
    shallow_applicable: int = 0
    shallow_suppressed: int = 0
    failure_to_structure: int = 0
    operator_calls_total: int = 0
    concept_calls_total: int = 0
    missing_operator_fail_closed: int = 0
    missing_concept_fail_closed: int = 0

    plan_depth_sum: int = 0

    z0_success_rate: Rolling = field(init=False)
    z1_fail_closed_ratio: Rolling = field(init=False)
    z2_avg_plan_depth: Rolling = field(init=False)
    z3_hier_dominance: Rolling = field(init=False)
    z4_pre_search_schema: Rolling = field(init=False)
    z5_object_centric: Rolling = field(init=False)
    z6_conditional: Rolling = field(init=False)
    z9_shallow_suppression_numer: Rolling = field(init=False)
    z9_shallow_suppression_denom: Rolling = field(init=False)
    z8_failure_to_structure_numer: Rolling = field(init=False)
    z8_failure_to_structure_denom: Rolling = field(init=False)
    z10_operator_calls_per_episode: Rolling = field(init=False)
    z11_concept_calls_per_episode: Rolling = field(init=False)
    z12_no_missing_operator: Rolling = field(init=False)
    z13_no_missing_concept: Rolling = field(init=False)

    def __post_init__(self) -> None:
        self.z0_success_rate = Rolling(self.window)
        self.z1_fail_closed_ratio = Rolling(self.window)
        self.z2_avg_plan_depth = Rolling(self.window)
        self.z3_hier_dominance = Rolling(self.window)
        self.z4_pre_search_schema = Rolling(self.window)
        self.z5_object_centric = Rolling(self.window)
        self.z6_conditional = Rolling(self.window)
        self.z9_shallow_suppression_numer = Rolling(self.window)
        self.z9_shallow_suppression_denom = Rolling(self.window)
        self.z8_failure_to_structure_numer = Rolling(self.window)
        self.z8_failure_to_structure_denom = Rolling(self.window)
        self.z10_operator_calls_per_episode = Rolling(self.window)
        self.z11_concept_calls_per_episode = Rolling(self.window)
        self.z12_no_missing_operator = Rolling(self.window)
        self.z13_no_missing_concept = Rolling(self.window)

    def update_episode(
        self,
        *,
        episode_success: bool,
        did_fail_closed: bool,
        did_pre_search_schema: bool,
        executed_with_plan: bool,
        object_centric_ok: bool,
        conditional_ok: bool,
        hierarchical_ok: bool,
        shallow_applicable: bool,
        shallow_suppressed: bool,
        failure_to_structure: bool,
        operator_calls: int,
        concept_calls: int,
        fail_reason: str,
        plan_depth: int,
    ) -> None:
        self.steps += 1
        if episode_success:
            self.successes += 1
        if did_fail_closed:
            self.fail_closed += 1
        if did_pre_search_schema:
            self.pre_search_schema += 1
        if executed_with_plan:
            self.executed_with_plan += 1
        if object_centric_ok:
            self.object_centric_ok += 1
        if conditional_ok:
            self.conditional_ok += 1
        if hierarchical_ok:
            self.hierarchical_ok += 1
        if shallow_applicable:
            self.shallow_applicable += 1
            if shallow_suppressed:
                self.shallow_suppressed += 1
        if failure_to_structure:
            self.failure_to_structure += 1
        self.operator_calls_total += int(max(0, operator_calls))
        self.concept_calls_total += int(max(0, concept_calls))

        r = str(fail_reason or "")
        is_missing_op = bool(did_fail_closed and r.startswith("missing_operator_"))
        is_missing_concept = bool(did_fail_closed and r.startswith("missing_concept_"))
        if is_missing_op:
            self.missing_operator_fail_closed += 1
        if is_missing_concept:
            self.missing_concept_fail_closed += 1
        if plan_depth > 0:
            self.plan_depth_sum += int(plan_depth)

        # Rolling Z(t) (per-episode regime signals)
        # Z1 measures fail-closed *compliance* (no primitive action without an explicit plan),
        # not the raw abort rate (which is tracked in cumulative()).
        self.z0_success_rate.push(1.0 if episode_success else 0.0)
        self.z1_fail_closed_ratio.push(1.0 if (executed_with_plan or did_fail_closed) else 0.0)
        self.z4_pre_search_schema.push(1.0 if did_pre_search_schema else 0.0)
        self.z5_object_centric.push(1.0 if object_centric_ok else 0.0)
        self.z6_conditional.push(1.0 if conditional_ok else 0.0)
        self.z3_hier_dominance.push(1.0 if hierarchical_ok else 0.0)
        self.z9_shallow_suppression_denom.push(1.0 if shallow_applicable else 0.0)
        self.z9_shallow_suppression_numer.push(1.0 if (shallow_applicable and shallow_suppressed) else 0.0)
        self.z2_avg_plan_depth.push(float(plan_depth))
        self.z10_operator_calls_per_episode.push(float(max(0, operator_calls)))
        self.z11_concept_calls_per_episode.push(float(max(0, concept_calls)))
        self.z12_no_missing_operator.push(0.0 if is_missing_op else 1.0)
        self.z13_no_missing_concept.push(0.0 if is_missing_concept else 1.0)
        is_failure = not episode_success
        self.z8_failure_to_structure_denom.push(1.0 if is_failure else 0.0)
        self.z8_failure_to_structure_numer.push(1.0 if (is_failure and failure_to_structure) else 0.0)

    def current_z(self) -> Dict[str, float]:
        denom = self.z8_failure_to_structure_denom.mean()
        numer = self.z8_failure_to_structure_numer.mean()
        # If there are no failures in the window, the conversion constraint is vacuously satisfied.
        z8 = 1.0 if denom <= 0.0 else float(numer / denom)
        denom9 = self.z9_shallow_suppression_denom.mean()
        numer9 = self.z9_shallow_suppression_numer.mean()
        # If shallow suppression is not applicable in the window, it is vacuously satisfied.
        z9 = 1.0 if denom9 <= 0.0 else float(numer9 / denom9)
        return {
            "Z0_episode_success_rate": self.z0_success_rate.mean(),
            "Z1_fail_closed_ratio": self.z1_fail_closed_ratio.mean(),
            "Z2_avg_plan_depth_before_execution": self.z2_avg_plan_depth.mean(),
            "Z3_hierarchical_dominance": self.z3_hier_dominance.mean(),
            "Z4_pre_search_schema_rate": self.z4_pre_search_schema.mean(),
            "Z5_object_centric_action_ratio": self.z5_object_centric.mean(),
            "Z6_conditional_rule_ratio": self.z6_conditional.mean(),
            "Z8_failure_to_structure_conversion_rate": z8,
            "Z9_shallow_path_suppression": z9,
            "Z10_operator_calls_per_episode": self.z10_operator_calls_per_episode.mean(),
            "Z11_concept_calls_per_episode": self.z11_concept_calls_per_episode.mean(),
            "Z12_no_missing_operator": self.z12_no_missing_operator.mean(),
            "Z13_no_missing_concept": self.z13_no_missing_concept.mean(),
        }

    def cumulative(self) -> Dict[str, Any]:
        plan_depth_mean = float(self.plan_depth_sum / max(1, self.executed_with_plan))
        operator_calls_mean = float(self.operator_calls_total / max(1, self.steps))
        concept_calls_mean = float(self.concept_calls_total / max(1, self.steps))
        return {
            "episodes": int(self.steps),
            "successes": int(self.successes),
            "success_rate": float(self.successes / max(1, self.steps)),
            "fail_closed": int(self.fail_closed),
            "fail_closed_ratio": float(self.fail_closed / max(1, self.steps)),
            "executed_with_plan": int(self.executed_with_plan),
            "pre_search_schema_rate": float(self.pre_search_schema / max(1, self.steps)),
            "object_centric_ok_rate": float(self.object_centric_ok / max(1, self.steps)),
            "conditional_ok_rate": float(self.conditional_ok / max(1, self.steps)),
            "hierarchical_ok_rate": float(self.hierarchical_ok / max(1, self.steps)),
            "shallow_suppression_rate": (
                1.0 if int(self.shallow_applicable) <= 0 else float(self.shallow_suppressed / max(1, self.shallow_applicable))
            ),
            "shallow_suppression_applicable_rate": float(self.shallow_applicable / max(1, self.steps)),
            "failure_to_structure_rate": float(self.failure_to_structure / max(1, self.steps)),
            "operator_calls_per_episode": operator_calls_mean,
            "concept_calls_per_episode": concept_calls_mean,
            "missing_operator_fail_closed": int(self.missing_operator_fail_closed),
            "missing_concept_fail_closed": int(self.missing_concept_fail_closed),
            "avg_plan_depth_before_execution": plan_depth_mean,
        }


def _default_z_thresholds() -> Dict[str, float]:
    # These are *structural regime* thresholds, not ARC scores.
    return {
        # A regime is not meaningful if it never succeeds.
        "Z0_episode_success_rate": 0.80,
        "Z1_fail_closed_ratio": 0.60,
        "Z2_avg_plan_depth_before_execution": 3.5,
        "Z3_hierarchical_dominance": 0.60,
        "Z4_pre_search_schema_rate": 0.50,
        "Z5_object_centric_action_ratio": 0.80,
        "Z6_conditional_rule_ratio": 0.50,
        # Failure must be converted into structure (when failures exist).
        "Z8_failure_to_structure_conversion_rate": 0.10,
        "Z9_shallow_path_suppression": 0.50,
        # Missing operator/concept should vanish in the stable regime window.
        "Z12_no_missing_operator": 0.95,
        "Z13_no_missing_concept": 0.95,
    }


def _regime_met(z: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    for k, thr in thresholds.items():
        if float(z.get(k, 0.0)) < float(thr):
            return False
    return True


def _latent_lower_bound(z: Dict[str, float], thresholds: Dict[str, float]) -> float:
    # Conservative: min_i min(1, z_i / thr_i)
    vals = []
    for k, thr in thresholds.items():
        thr_f = float(thr)
        if thr_f <= 0:
            continue
        vals.append(min(1.0, float(z.get(k, 0.0)) / thr_f))
    if not vals:
        return 0.0
    return float(min(vals))


# ----------------------------
# Planning (deterministic, object-centric)
# ----------------------------


def _parse_objects(img: Any, api: MiniGridAPI) -> Tuple[ParsedObject, ...]:
    # FullyObsWrapper returns a (W,H,3) tensor, i.e. axis0=x (col), axis1=y (row).
    w = int(img.shape[0])
    h = int(img.shape[1])
    out: List[ParsedObject] = []
    for x in range(w):
        for y in range(h):
            cell = img[x, y]
            obj_type = api.IDX_TO_OBJECT.get(int(cell[0]), "unknown")
            if obj_type in ("empty", "floor", "unseen"):
                continue
            color = api.IDX_TO_COLOR.get(int(cell[1]))
            state = int(cell[2])
            out.append(ParsedObject(obj_type=obj_type, color=color, state=state, pos=(x, y)))
    return tuple(out)


_RE_PICKUP = re.compile(r"pick up the ([a-z]+) ([a-z]+)")


def _derive_goal(obs: Dict[str, Any], world: WorldState) -> Optional[GoalSpec]:
    goals = list(world.iter_objects("goal"))
    if goals:
        return GoalSpec(kind="reach_goal", target_type="goal", target_color=None)

    mission = str(obs.get("mission") or "")
    m = _RE_PICKUP.search(mission)
    if m:
        color = m.group(1)
        obj_type = m.group(2)
        return GoalSpec(kind="pickup", target_type=obj_type, target_color=color)

    return None


def _neighbors4(x: int, y: int) -> List[Tuple[int, int]]:
    # Deterministic order.
    return [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]


def _bfs_path(
    *,
    w: int,
    h: int,
    start: Tuple[int, int],
    goals: Sequence[Tuple[int, int]],
    passable: Any,
) -> Optional[List[Tuple[int, int]]]:
    # Returns path including start and end.
    from collections import deque

    goal_set = set(goals)
    if start in goal_set:
        return [start]

    q = deque([start])
    prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    while q:
        cur = q.popleft()
        cx, cy = cur
        for nx, ny in _neighbors4(cx, cy):
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            nxt = (nx, ny)
            if nxt in prev:
                continue
            if not passable(nx, ny):
                continue
            prev[nxt] = cur
            if nxt in goal_set:
                # reconstruct
                path = [nxt]
                while path[-1] != start:
                    p = prev[path[-1]]
                    if p is None:
                        break
                    path.append(p)
                path.reverse()
                return path
            q.append(nxt)
    return None


def _turn_actions(api: MiniGridAPI, cur_dir: int, desired_dir: int) -> List[int]:
    # Rotate to desired_dir using minimal turns; tie (2 turns) -> left twice.
    if cur_dir == desired_dir:
        return []
    diff = (desired_dir - cur_dir) % 4
    if diff == 1:
        return [int(api.Actions.right)]
    if diff == 2:
        return [int(api.Actions.left), int(api.Actions.left)]
    if diff == 3:
        return [int(api.Actions.left)]
    return []


def _dir_for_step(dx: int, dy: int) -> Optional[int]:
    # minigrid dirs: 0:+x, 1:+y, 2:-x, 3:-y
    if dx == 1 and dy == 0:
        return 0
    if dx == 0 and dy == 1:
        return 1
    if dx == -1 and dy == 0:
        return 2
    if dx == 0 and dy == -1:
        return 3
    return None


def _adjacent_cells(pos: Tuple[int, int], w: int, h: int) -> List[Tuple[int, int]]:
    x, y = pos
    out = []
    for nx, ny in _neighbors4(x, y):
        if 0 <= nx < w and 0 <= ny < h:
            out.append((nx, ny))
    return out


def _count_bucket(n: int) -> str:
    if int(n) <= 0:
        return "0"
    if int(n) == 1:
        return "1"
    return "2+"


def _world_features(world: WorldState, goal: GoalSpec) -> Dict[str, Any]:
    locked_doors = [o for o in world.objects if o.obj_type == "door" and o.state == 2]
    keys = [o for o in world.objects if o.obj_type == "key"]
    balls = [o for o in world.objects if o.obj_type == "ball"]
    boxes = [o for o in world.objects if o.obj_type == "box"]
    blockers_adj_locked = False
    for d in locked_doors:
        adj = set(_adjacent_cells(d.pos, world.width, world.height))
        for b in itertools.chain(balls, boxes):
            if b.pos in adj:
                blockers_adj_locked = True
                break
        if blockers_adj_locked:
            break
    return {
        "grid": {"w": int(world.width), "h": int(world.height)},
        "goal_kind": str(goal.kind),
        "target_type": str(goal.target_type),
        "target_color": goal.target_color,
        "doors_locked_bucket": _count_bucket(len(locked_doors)),
        "keys_bucket": _count_bucket(len(keys)),
        "balls_bucket": _count_bucket(len(balls)),
        "boxes_bucket": _count_bucket(len(boxes)),
        "carrying_type": (world.carrying.obj_type if world.carrying else None),
        "carrying_color": (world.carrying.color if world.carrying else None),
        "blockers_adj_locked_door": bool(blockers_adj_locked),
    }


def _cluster_ctx_key(features: Dict[str, Any]) -> str:
    # Key must avoid env_id/task id; structural only.
    # Keep it low-cardinality (buckets/booleans), deterministic and serializable.
    return _stable_sha16(
        {
            "goal_kind": features.get("goal_kind"),
            "target_type": features.get("target_type"),
            "target_color": features.get("target_color"),
            "doors_locked_bucket": features.get("doors_locked_bucket"),
            "keys_bucket": features.get("keys_bucket"),
            "blockers_adj_locked_door": features.get("blockers_adj_locked_door"),
        }
    )


def _schema_for(world: WorldState, goal: GoalSpec) -> Tuple[str, Dict[str, Any]]:
    # Object-centric schema selection by world features.
    locked_doors = [o for o in world.objects if o.obj_type == "door" and o.state == 2]
    locked_bucket = "0" if len(locked_doors) == 0 else ("1" if len(locked_doors) == 1 else "2+")
    has_locked_door = bool(locked_doors)
    sig = {"goal_kind": goal.kind, "has_locked_door": bool(has_locked_door), "locked_doors_bucket": locked_bucket}

    if goal.kind == "pickup":
        # Pickup tasks can still require unlocking; multi-locked layouts need a deeper schema.
        name = "SCHEMA_UNLOCK_ALL_DOORS_THEN_PICKUP" if len(locked_doors) >= 2 else "SCHEMA_PICKUP_TARGET"
    else:
        if len(locked_doors) >= 2:
            name = "SCHEMA_UNLOCK_ALL_DOORS_THEN_GOAL"
        elif has_locked_door:
            name = "SCHEMA_UNLOCK_DOOR_THEN_GOAL"
        else:
            name = "SCHEMA_NAVIGATE_TO_GOAL"
    return name, sig


def _induce_schema(
    step: int,
    schema_name: str,
    signature: Dict[str, Any],
    *,
    origin: str = "schema_missing",
    origin_cluster_id: Optional[str] = None,
    state: str = "candidate",
    components: Tuple[str, ...] = (),
) -> Schema:
    schema_id = f"schema_{_stable_sha16({'name': schema_name, 'signature': signature})}"
    ocid = origin_cluster_id
    if ocid is None and str(origin) == "schema_missing":
        ocid = f"fc_missing_{_stable_sha16({'name': schema_name, 'signature': signature})}"
    return Schema(
        schema_id=schema_id,
        name=schema_name,
        signature=dict(signature),
        support=0,
        promoted=False,
        state=str(state),
        origin=str(origin),
        origin_cluster_id=ocid,
        components=tuple(components),
        created_step=step,
        last_used_step=step,
        ttl=2000,
    )


def _schema_components_for_name(schema_name: str) -> Tuple[str, ...]:
    name = str(schema_name)
    if name == "SCHEMA_UNLOCK_ALL_DOORS_THEN_GOAL":
        return ("SCHEMA_UNLOCK_ONE_DOOR", "SCHEMA_NAVIGATE_TO_GOAL")
    if name == "SCHEMA_UNLOCK_ALL_DOORS_THEN_PICKUP":
        return ("SCHEMA_UNLOCK_ONE_DOOR", "SCHEMA_PICKUP_TARGET")
    return ()


def _induce_schema_from_failure_cluster(step: int, cluster: FailureCluster) -> Optional[Schema]:
    # Deterministic, general induction rules for MiniGrid-like planning failures.
    sig = dict(cluster.signature)
    goal_kind = str(sig.get("goal_kind") or "")
    wf = sig.get("world_features") or {}
    doors_bucket = str(wf.get("doors_locked_bucket") or "0")

    # Multi-locked-door layouts require a composed/unrolled schema; a single-door schema will fail.
    if doors_bucket == "2+":
        if goal_kind == "reach_goal":
            name = "SCHEMA_UNLOCK_ALL_DOORS_THEN_GOAL"
            components = ("SCHEMA_UNLOCK_ONE_DOOR", "SCHEMA_NAVIGATE_TO_GOAL")
        elif goal_kind == "pickup":
            name = "SCHEMA_UNLOCK_ALL_DOORS_THEN_PICKUP"
            components = ("SCHEMA_UNLOCK_ONE_DOOR", "SCHEMA_PICKUP_TARGET")
        else:
            return None

        schema_sig = {"goal_kind": goal_kind, "has_locked_door": True, "locked_doors_bucket": "2+"}
        return _induce_schema(
            step,
            name,
            schema_sig,
            origin="failure_cluster",
            origin_cluster_id=str(cluster.cluster_id),
            state="quarantined",
            components=components,
        )

    return None


def _operator_id_for(op_name: str, *, impl: str, signature: Dict[str, Any]) -> str:
    return f"op_{_stable_sha16({'name': str(op_name), 'impl': str(impl), 'signature': signature})}"


def _induce_operator(
    step: int,
    op_name: str,
    *,
    impl: str,
    signature: Dict[str, Any],
    origin: str = "failure_cluster",
    origin_cluster_id: Optional[str] = None,
    state: str = "candidate",
) -> Operator:
    op_id = _operator_id_for(op_name, impl=str(impl), signature=dict(signature))
    return Operator(
        operator_id=op_id,
        name=str(op_name),
        impl=str(impl),
        signature=dict(signature),
        support=0,
        promoted=False,
        state=str(state),
        origin=str(origin),
        origin_cluster_id=str(origin_cluster_id or ""),
        created_step=step,
        last_used_step=step,
        ttl=2000,
    )


def _induce_operator_from_failure_cluster(step: int, cluster: FailureCluster) -> Optional[Operator]:
    # Deterministic operator discovery v2:
    # Convert recurring missing-operator failures into executable operator closures (no per-task hacks).
    sig = dict(cluster.signature)
    reason = str(sig.get("reason") or "")
    wf = sig.get("world_features") or {}
    plan_fail_concept = sig.get("plan_fail_concept") or {}
    plan_fail_concept_name = ""
    if isinstance(plan_fail_concept, dict):
        plan_fail_concept_name = str(plan_fail_concept.get("name") or "")

    doors_bucket = str(wf.get("doors_locked_bucket") or "0")
    keys_bucket = str(wf.get("keys_bucket") or "0")
    boxes_bucket = str(wf.get("boxes_bucket") or "0")
    balls_bucket = str(wf.get("balls_bucket") or "0")

    missing_kind: Optional[str] = None
    if reason.startswith("missing_operator_"):
        missing_kind = str(reason[len("missing_operator_") :])
    elif reason in ("unbox_key_no_matching_key",):
        missing_kind = "unbox_key"
    elif reason == "no_path" and plan_fail_concept_name in ("NAVIGATE_TO", "NAVIGATE_ADJACENT", ""):
        # Legacy reason: older runs emitted only "no_path". Treat as nav operator gap when blockers exist.
        missing_kind = "nav_clearing"

    candidates: List[Tuple[int, str, Operator]] = []

    # Missing operator: hidden keys inside boxes (e.g., ObstructedMaze). This must become a first-class
    # operator closure (UNBOX_KEY), not "ban until die".
    if missing_kind == "unbox_key":
        if doors_bucket != "0" and boxes_bucket != "0":
            op_name = "OP_UNBOX_KEY_V1"
            op_sig = {
                "kind": "unbox_key",
                "covers_reason": "missing_operator_unbox_key",
                "covers_concept": "UNBOX_KEY_FOR_DOOR",
            }
            op = _induce_operator(
                step,
                op_name,
                impl="UNBOX_KEY_V1",
                signature=op_sig,
                origin="failure_cluster",
                origin_cluster_id=str(cluster.cluster_id),
                state="quarantined",
            )
            candidates.append((100, str(op.operator_id), op))

    nav_need: Optional[int] = None
    if missing_kind == "nav_clearing":
        nav_need = 1
    else:
        m = re.match(r"^nav_clearing_v([0-9]+)$", str(missing_kind or ""))
        if m:
            try:
                nav_need = int(m.group(1))
            except Exception:
                nav_need = None

    if nav_need is not None and plan_fail_concept_name in ("NAVIGATE_TO", "NAVIGATE_ADJACENT", ""):
        # Stronger operator needed when a single-clear closure can't open a path (multi-blocker chokepoints).
        # This must be handled by *expanding* the reusable closure family (NAV_CLEARING_Vk), not by Ω burn.
        ver = max(1, int(nav_need))

        # If locked doors exist but neither visible keys nor boxes exist, this is not an operator gap; it's an impossible region.
        # (If boxes exist, keys can be hidden and UNBOX_KEY can unlock the region.)
        if doors_bucket != "0" and keys_bucket == "0" and boxes_bucket == "0":
            pass
        elif balls_bucket != "0" or boxes_bucket != "0":
            if ver <= 1:
                op_name = "OP_NAV_CLEARING_V1"
                impl = "NAV_CLEARING_V1"
                kind = "nav_clearing"
                covers_reason = "missing_operator_nav_clearing"
                pri = 50
            else:
                op_name = f"OP_NAV_CLEARING_V{ver}"
                impl = f"NAV_CLEARING_V{ver}"
                kind = f"nav_clearing_v{ver}"
                covers_reason = f"missing_operator_nav_clearing_v{ver}"
                pri = 60 + min(10, int(ver))

            op_sig = {
                "kind": str(kind),
                "covers_reason": str(covers_reason),
                "covers_concept": "NAVIGATE*",
            }
            op = _induce_operator(
                step,
                op_name,
                impl=str(impl),
                signature=op_sig,
                origin="failure_cluster",
                origin_cluster_id=str(cluster.cluster_id),
                state="quarantined",
            )
            candidates.append((int(pri), str(op.operator_id), op))

    if not candidates:
        return None
    candidates.sort(key=lambda t: (-int(t[0]), str(t[2].name), str(t[1])))
    return candidates[0][2]


def _sorted_unique_ablation_witnesses(witnesses: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()
    for w in witnesses:
        if not isinstance(w, dict):
            continue
        env_id = str(w.get("env_id") or "")
        seed = int(w.get("seed") or 0)
        if not env_id:
            continue
        k = (env_id, seed)
        if k in seen:
            continue
        seen.add(k)
        out.append({"env_id": env_id, "seed": seed})
    out.sort(key=lambda d: (str(d.get("env_id")), int(d.get("seed") or 0)))
    return out


def _preserves_future_schema_v1(
    *,
    api: MiniGridAPI,
    bank: SchemaBank,
    op_bank: OperatorBank,
    schema: Schema,
    contexts_used: int,
    origin_cluster_resolved: bool,
    omega_enabled: bool,
    omega_first_burn_step: Optional[int],
    ablation_trials: int,
    ablation_witnesses: Sequence[Dict[str, Any]],
    min_plan_depth: int,
    forbid_shallow_paths: bool,
    fail_closed_required: bool,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Deterministic, bounded preserves_future() gate for schema promotion.

    v1 uses strict, computable proxies (no stats/gradients):
      P1 cross-context: contexts_used >= 2
      P2 born-from-failure: origin_cluster_id != None (schema_missing is treated as failure-origin via synthetic cluster id)
      P3 unlock hard region: origin cluster resolved OR schema_missing (trivial unlock)
      P4 survive Ω: if Ω has burned at least once, schema must be used after first burn
      P5 contrafactual ablation: removing schema causes failures on witness episodes
    """
    evidence: Dict[str, Any] = {}

    # P1: cross-context reuse.
    p1 = int(contexts_used) >= 2
    evidence["P1_cross_context_ok"] = bool(p1)
    evidence["contexts_used"] = int(contexts_used)

    # P2: born from failure (cluster id required; schema_missing gets synthetic fc_missing_*).
    p2 = bool(schema.origin_cluster_id)
    evidence["P2_born_from_failure_ok"] = bool(p2)
    evidence["origin"] = str(schema.origin)
    evidence["origin_cluster_id"] = str(schema.origin_cluster_id or "")

    # P3: unlock / preserve optionality (proxy).
    p3 = bool(origin_cluster_resolved or str(schema.origin) == "schema_missing")
    evidence["P3_unlock_ok"] = bool(p3)
    evidence["origin_cluster_resolved"] = bool(origin_cluster_resolved)

    # P4: survives Ω pressure (proxy).
    p4 = True
    if omega_enabled and omega_first_burn_step is not None:
        p4 = int(schema.last_used_step) >= int(omega_first_burn_step)
    evidence["P4_survives_omega_ok"] = bool(p4)
    evidence["omega_enabled"] = bool(omega_enabled)
    evidence["omega_first_burn_step"] = int(omega_first_burn_step) if omega_first_burn_step is not None else None
    evidence["schema_last_used_step"] = int(schema.last_used_step)

    # P5: contrafactual ablation (mandatory).
    trials = max(0, int(ablation_trials))
    witnesses = _sorted_unique_ablation_witnesses(ablation_witnesses)[:trials]
    snap_all = _freeze_schema_snapshot(bank)["schemas"]
    ops_all = _freeze_operator_snapshot(op_bank)["operators"]
    snap_abl = [d for d in snap_all if str(d.get("schema_id") or "") != str(schema.schema_id)]
    ablation_results: List[Dict[str, Any]] = []
    failures_without = 0
    for w in witnesses:
        env_id = str(w.get("env_id") or "")
        seed = int(w.get("seed") or 0)
        res = _run_episode_core(
            api,
            {
                "env_id": env_id,
                "seed": seed,
                "schemas": snap_abl,
                "operators": ops_all,
                "min_plan_depth": int(min_plan_depth),
                "forbid_shallow_paths": bool(forbid_shallow_paths),
                "fail_closed_required": bool(fail_closed_required),
                "omega_skip_checks": True,
            },
        )
        ok0 = bool(res.get("episode_success", False))
        if not ok0:
            failures_without += 1
        ablation_results.append(
            {
                "env_id": env_id,
                "seed": seed,
                "success_without_schema": bool(ok0),
                "outcome": str(res.get("outcome") or ""),
                "reason": str(res.get("reason") or ""),
            }
        )
    required_failures = 1 if trials > 0 else 0
    p5 = int(failures_without) >= int(required_failures)
    evidence["P5_ablation_ok"] = bool(p5)
    evidence["ablation_trials"] = int(trials)
    evidence["ablation_failures_without_schema"] = int(failures_without)
    evidence["ablation_required_failures"] = int(required_failures)
    evidence["ablation_results"] = ablation_results

    ok = bool(p1 and p2 and p3 and p4 and p5)
    evidence["preserves_future_v1_ok"] = bool(ok)
    return ok, evidence


def _preserves_future_operator_v1(
    *,
    api: MiniGridAPI,
    bank: SchemaBank,
    op_bank: OperatorBank,
    operator: Operator,
    contexts_used: int,
    origin_cluster_resolved: bool,
    omega_enabled: bool,
    omega_first_burn_step: Optional[int],
    ablation_trials: int,
    ablation_witnesses: Sequence[Dict[str, Any]],
    min_plan_depth: int,
    forbid_shallow_paths: bool,
    fail_closed_required: bool,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Deterministic, bounded preserves_future() gate for operator promotion.

    v1 uses strict, computable proxies (no stats/gradients):
      P1 cross-context: contexts_used >= 2
      P2 born-from-failure: origin_cluster_id != ""
      P3 unlock hard region: origin cluster resolved
      P4 survive Ω: if Ω has burned at least once, operator must be used after first burn
      P5 contrafactual ablation: removing operator causes failures on witness episodes
    """
    evidence: Dict[str, Any] = {}

    p1 = int(contexts_used) >= 2
    evidence["P1_cross_context_ok"] = bool(p1)
    evidence["contexts_used"] = int(contexts_used)

    p2 = bool(operator.origin_cluster_id)
    evidence["P2_born_from_failure_ok"] = bool(p2)
    evidence["origin"] = str(operator.origin)
    evidence["origin_cluster_id"] = str(operator.origin_cluster_id or "")

    p3 = bool(origin_cluster_resolved)
    evidence["P3_unlock_ok"] = bool(p3)
    evidence["origin_cluster_resolved"] = bool(origin_cluster_resolved)

    p4 = True
    if omega_enabled and omega_first_burn_step is not None:
        p4 = int(operator.last_used_step) >= int(omega_first_burn_step)
    evidence["P4_survives_omega_ok"] = bool(p4)
    evidence["omega_enabled"] = bool(omega_enabled)
    evidence["omega_first_burn_step"] = int(omega_first_burn_step) if omega_first_burn_step is not None else None
    evidence["operator_last_used_step"] = int(operator.last_used_step)

    trials = max(0, int(ablation_trials))
    witnesses = _sorted_unique_ablation_witnesses(ablation_witnesses)[:trials]
    schemas_all = _freeze_schema_snapshot(bank)["schemas"]
    ops_all = _freeze_operator_snapshot(op_bank)["operators"]
    ops_abl = [d for d in ops_all if str(d.get("operator_id") or "") != str(operator.operator_id)]
    ablation_results: List[Dict[str, Any]] = []
    failures_without = 0
    for w in witnesses:
        env_id = str(w.get("env_id") or "")
        seed = int(w.get("seed") or 0)
        res = _run_episode_core(
            api,
            {
                "env_id": env_id,
                "seed": seed,
                "schemas": schemas_all,
                "operators": ops_abl,
                "min_plan_depth": int(min_plan_depth),
                "forbid_shallow_paths": bool(forbid_shallow_paths),
                "fail_closed_required": bool(fail_closed_required),
                "omega_skip_checks": True,
            },
        )
        ok0 = bool(res.get("episode_success", False))
        if not ok0:
            failures_without += 1
        ablation_results.append(
            {
                "env_id": env_id,
                "seed": seed,
                "success_without_operator": bool(ok0),
                "outcome": str(res.get("outcome") or ""),
                "reason": str(res.get("reason") or ""),
            }
        )

    required_failures = 1 if trials > 0 else 0
    p5 = int(failures_without) >= int(required_failures)
    evidence["P5_ablation_ok"] = bool(p5)
    evidence["ablation_trials"] = int(trials)
    evidence["ablation_failures_without_operator"] = int(failures_without)
    evidence["ablation_required_failures"] = int(required_failures)
    evidence["ablation_results"] = ablation_results

    ok = bool(p1 and p2 and p3 and p4 and p5)
    evidence["preserves_future_v1_ok"] = bool(ok)
    return ok, evidence


def _maxwell_promote_operator_from_cluster_v1(
    *,
    api: MiniGridAPI,
    bank: SchemaBank,
    op_bank: OperatorBank,
    failure_clusters: FailureClusterStore,
    cluster: FailureCluster,
    operator_id: str,
    step: int,
    operator_contexts: Dict[str, set[str]],
    operator_witnesses: Dict[str, List[Dict[str, Any]]],
    omega_enabled: bool,
    omega_first_burn_step: Optional[int],
    min_plan_depth: int,
    forbid_shallow_paths: bool,
    fail_closed_required: bool,
    ablation_trials: int,
    promotion_cooldown_steps: int,
) -> Tuple[bool, Dict[str, Any]]:
    """
    MAXWELL_Ω (operator arm) — immediate, bounded validation + promotion attempt.

    Goal: if an operator is induced from a failure cluster, try to *use it* on the
    cluster's witness episodes and promote it (with preserves_future gate) before Ω burns.

    This is deterministic, bounded, and does not use per-task identifiers. It uses only
    env_id+seed witnesses already present in the failure cluster.
    """
    meta: Dict[str, Any] = {"enabled": True, "operator_id": str(operator_id)}
    op0 = op_bank.operators.get(str(operator_id))
    if op0 is None:
        return False, {**meta, "ok": False, "reason": "operator_missing"}
    if bool(op0.promoted):
        return False, {**meta, "ok": False, "reason": "already_promoted"}

    witnesses = _sorted_unique_ablation_witnesses(cluster.witnesses)
    if not witnesses:
        return False, {**meta, "ok": False, "reason": "no_cluster_witnesses"}

    # Run a small number of witness episodes to gather: (a) operator actually used, (b) support, (c) contexts.
    want_support = max(1, int(getattr(op_bank, "promote_support", 3) or 3))
    max_runs = max(8, int(want_support))
    # If the cluster has too few observed witnesses (early failures), deterministically synthesize
    # additional env seeds for the *same env_id* to gather cross-context evidence fast enough to
    # satisfy MAXWELL_Ω under strict burn. This is not per-task: MiniGrid is the generator.
    synth_used = False
    if len(witnesses) < int(max_runs):
        env0 = str(witnesses[0].get("env_id") or "")
        seed0 = int(witnesses[0].get("seed") or 0)
        extra: List[Dict[str, Any]] = []
        for i in range(int(max_runs) - int(len(witnesses))):
            # Deterministic seed schedule (bounded); avoid collisions with the base seed.
            salt16 = _stable_sha16({"cluster_id": str(cluster.cluster_id), "step": int(step), "i": int(i)})
            try:
                off = int(str(salt16), 16)
            except Exception:
                off = 0
            extra_seed = (int(seed0) + 997 * int(i + 1) + int(off)) % (2**31 - 1)
            extra.append({"env_id": str(env0), "seed": int(extra_seed)})
        witnesses = _sorted_unique_ablation_witnesses(list(witnesses) + list(extra))
        synth_used = True
    runs: List[Dict[str, Any]] = []
    used_successes: List[Dict[str, Any]] = []

    schemas_all = _freeze_schema_snapshot(bank)["schemas"]
    ops_all = _freeze_operator_snapshot(op_bank)["operators"]
    want_ck = str(cluster.signature.get("cluster_ctx_key") or "")

    for w in witnesses[: int(max_runs)]:
        env_id = str(w.get("env_id") or "")
        seed = int(w.get("seed") or 0)
        if not env_id:
            continue
        res = _run_episode_core(
            api,
            {
                "env_id": env_id,
                "seed": seed,
                "schemas": schemas_all,
                "operators": ops_all,
                "min_plan_depth": int(min_plan_depth),
                "forbid_shallow_paths": bool(forbid_shallow_paths),
                "fail_closed_required": bool(fail_closed_required),
                "omega_skip_checks": True,
            },
        )
        ok_ep = bool(res.get("episode_success", False))
        op_calls = [str(x) for x in (res.get("operator_calls") or []) if str(x)]
        used = bool(str(operator_id) in set(op_calls))
        got_ck = str(res.get("cluster_ctx_key") or "")
        ck_match = bool(want_ck and got_ck and got_ck == want_ck)
        runs.append(
            {
                "env_id": env_id,
                "seed": int(seed),
                "episode_success": bool(ok_ep),
                "operator_used": bool(used),
                "cluster_ctx_key_match": bool(ck_match),
                "outcome": str(res.get("outcome") or ""),
                "reason": str(res.get("reason") or ""),
            }
        )

        if ok_ep and used and ck_match:
            used_successes.append({"env_id": env_id, "seed": int(seed)})
            op_bank.mark_success(str(operator_id), int(step))
            ctx = _stable_sha16({"env_id": env_id, "seed": int(seed)})
            operator_contexts.setdefault(str(operator_id), set()).add(str(ctx))
            wl0 = operator_witnesses.get(str(operator_id)) or []
            wl0 = list(wl0) + [{"env_id": env_id, "seed": int(seed)}]
            operator_witnesses[str(operator_id)] = _sorted_unique_ablation_witnesses(wl0)[:25]
        else:
            # Negative evidence: the operator was available but did not help.
            # This is still deterministic and bounded.
            if not ok_ep:
                op_bank.mark_failure(str(operator_id), int(step))

        # Stop early once we have enough successful uses to satisfy both:
        # - support for promotion eligibility
        # - >=2 contexts (cross-context)
        op_now = op_bank.operators.get(str(operator_id))
        ctx_used = len(operator_contexts.get(str(operator_id), set()))
        if op_now is not None and int(op_now.support) >= int(want_support) and int(ctx_used) >= 2:
            break

    meta["witness_runs"] = list(runs)[:10]
    meta["witness_used_successes"] = _sorted_unique_ablation_witnesses(used_successes)
    meta["synthetic_witnesses_used"] = bool(synth_used)

    if not used_successes:
        return False, {**meta, "ok": False, "reason": "operator_did_not_resolve_witnesses"}

    # Mark this context resolved (MAXWELL_Ω: success after structural progress).
    ck = str(cluster.signature.get("cluster_ctx_key") or "")
    if ck:
        failure_clusters.mark_ctx_resolved(ck)

    op0 = op_bank.operators.get(str(operator_id))
    if op0 is None:
        return False, {**meta, "ok": False, "reason": "operator_missing_after_runs"}

    # Respect promotion cooldown (even for MAXWELL).
    last_try = int(getattr(op0, "last_promotion_attempt_step", -1) or -1)
    if last_try >= 0 and (int(step) - int(last_try)) < int(promotion_cooldown_steps):
        return False, {**meta, "ok": False, "reason": "promotion_cooldown"}

    # Only attempt promotion when eligible by support (avoid a free bypass).
    if int(op0.support) < int(want_support):
        return False, {**meta, "ok": False, "reason": "insufficient_support", "support": int(op0.support)}

    op0.promotion_attempts += 1
    op0.last_promotion_attempt_step = int(step)

    contexts_used = len(operator_contexts.get(str(operator_id), set()))
    origin_cluster_resolved = bool(cluster.resolved)
    ok_pf, evidence = _preserves_future_operator_v1(
        api=api,
        bank=bank,
        op_bank=op_bank,
        operator=op0,
        contexts_used=int(contexts_used),
        origin_cluster_resolved=bool(origin_cluster_resolved),
        omega_enabled=bool(omega_enabled),
        omega_first_burn_step=omega_first_burn_step,
        ablation_trials=int(ablation_trials),
        ablation_witnesses=operator_witnesses.get(str(operator_id)) or [],
        min_plan_depth=int(min_plan_depth),
        forbid_shallow_paths=bool(forbid_shallow_paths),
        fail_closed_required=bool(fail_closed_required),
    )
    evidence = {**dict(evidence), "step": int(step), "operator_name": str(op0.name)}
    meta["promotion_attempt_ok"] = bool(ok_pf)
    meta["promotion_evidence"] = dict(evidence)

    if ok_pf and op_bank.promote(str(operator_id), int(step), evidence):
        return True, {**meta, "ok": True, "promoted": True}
    return False, {**meta, "ok": False, "promoted": False}


def _plan_with_schema(
    *,
    api: MiniGridAPI,
    env: Any,
    world: WorldState,
    goal: GoalSpec,
    env_id: str,
    seed: int,
    schema_id: str,
    schema_name: str,
    operators: Sequence[Dict[str, Any]],
    operator_ids: set[str],
    min_plan_depth: int,
) -> Tuple[Optional[Plan], Dict[str, Any]]:
    facts: Dict[str, Any] = {
        "conditional_ok": False,
        "hierarchical_ok": False,
    }

    # Derive key targets from objects (object-centric).
    w = world.width
    h = world.height
    agent_pos = world.agent_pos

    goals = list(world.iter_objects("goal"))
    if goal.kind == "reach_goal":
        if not goals:
            return None, facts
        goal_pos = goals[0].pos
    else:
        # pickup
        candidates = [o for o in world.objects if o.obj_type == goal.target_type and o.color == goal.target_color]
        if not candidates:
            return None, facts
        goal_pos = candidates[0].pos

    concepts: List[ConceptCall] = []
    schema_calls: List[str] = []
    operator_calls: List[str] = []

    def pick_operator_id(*, impl: str) -> Optional[str]:
        # Deterministic selection: promoted first, then stable id.
        promoted: List[str] = []
        others: List[str] = []
        for d in operators:
            oid = str(d.get("operator_id") or "")
            if (not oid) or (oid not in operator_ids):
                continue
            if str(d.get("impl") or "") != str(impl):
                continue
            if str(d.get("state") or "") == "deprecated":
                continue
            if bool(d.get("promoted", False)):
                promoted.append(oid)
            else:
                others.append(oid)
        if promoted:
            return sorted(promoted)[0]
        if others:
            return sorted(others)[0]
        return None

    op_by_id: Dict[str, Dict[str, Any]] = {}
    for d in operators:
        if not isinstance(d, dict):
            continue
        oid = str(d.get("operator_id") or "")
        if (not oid) or (oid not in operator_ids):
            continue
        op_by_id[str(oid)] = d

    def _nav_clearing_version(d: Dict[str, Any]) -> Optional[int]:
        impl = str(d.get("impl") or "")
        if impl.startswith("NAV_CLEARING_V"):
            try:
                ver = int(impl[len("NAV_CLEARING_V") :])
                return int(ver)
            except Exception:
                return None
        sig = d.get("signature") if isinstance(d.get("signature"), dict) else {}
        kind = str(sig.get("kind") or "")
        if kind == "nav_clearing":
            return 1
        m = re.match(r"^nav_clearing_v([0-9]+)$", kind)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    nav_clearing_ops: List[Tuple[int, bool, str]] = []
    for oid, d in op_by_id.items():
        if str(d.get("state") or "") == "deprecated":
            continue
        ver = _nav_clearing_version(d)
        if ver is None:
            continue
        nav_clearing_ops.append((int(ver), bool(d.get("promoted", False)), str(oid)))
    nav_clearing_ops.sort(key=lambda t: (int(t[0]), 0 if t[1] else 1, str(t[2])))

    def pick_nav_clearing_exact(ver: int) -> Optional[str]:
        ver = int(ver)
        cands = [t for t in nav_clearing_ops if int(t[0]) == ver]
        if not cands:
            return None
        # promoted first, then id (nav_clearing_ops already sorted by promoted/id within version).
        return str(cands[0][2])

    def pick_nav_clearing_min_ge(min_ver: int) -> Optional[str]:
        min_ver = int(min_ver)
        cands = [t for t in nav_clearing_ops if int(t[0]) >= min_ver]
        if not cands:
            return None
        # Prefer smallest adequate version; within version, promoted then id.
        cands.sort(key=lambda t: (int(t[0]), 0 if t[1] else 1, str(t[2])))
        return str(cands[0][2])

    def max_nav_clearing_version() -> int:
        return max([int(t[0]) for t in nav_clearing_ops], default=0)

    op_unbox_key_v1 = pick_operator_id(impl="UNBOX_KEY_V1")

    # Conditional rule: unlock door if needed.
    locked_doors = [o for o in world.objects if o.obj_type == "door" and o.state == 2]
    has_locked = bool(locked_doors)
    # Always evaluate at least one conditional rule (vacuous when empty):
    # "for each door, if locked then ensure key before toggle".
    facts["conditional_ok"] = True
    if has_locked:
        # Multi-door schema must unlock all locked doors, otherwise navigation/pickup can become impossible.
        unlock_all = schema_name in ("SCHEMA_UNLOCK_ALL_DOORS_THEN_GOAL", "SCHEMA_UNLOCK_ALL_DOORS_THEN_PICKUP")

        def _order_locked_doors_by_reachability(doors: List[ParsedObject]) -> List[ParsedObject]:
            # Deterministic topological order: only unlock doors that are reachable in the current
            # unlocked region (treat locked doors as walls, opened doors as passable).
            from collections import deque

            remaining = [d for d in doors if d.obj_type == "door" and d.state == 2]
            remaining = sorted(remaining, key=lambda o: (int(o.pos[1]), int(o.pos[0]), str(o.color)))
            opened: set[Tuple[int, int]] = set()
            out: List[ParsedObject] = []

            def passable_ordering(nx: int, ny: int) -> bool:
                obj = env.unwrapped.grid.get(int(nx), int(ny))
                if obj is None:
                    return True
                if obj.type in ("wall", "lava"):
                    return False
                if obj.type in ("key", "box", "ball"):
                    return False
                if obj.type == "door":
                    if (int(nx), int(ny)) in opened:
                        return True
                    if bool(getattr(obj, "is_open", False)):
                        return True
                    # Closed but unlocked doors can be toggled open during navigation.
                    if not bool(getattr(obj, "is_locked", False)):
                        return True
                    return False
                return True

            while remaining:
                start = (int(agent_pos[0]), int(agent_pos[1]))
                q = deque([start])
                visited = {start}
                while q:
                    cx, cy = q.popleft()
                    for nx, ny in _neighbors4(int(cx), int(cy)):
                        if not (0 <= int(nx) < int(w) and 0 <= int(ny) < int(h)):
                            continue
                        p = (int(nx), int(ny))
                        if p in visited:
                            continue
                        if not passable_ordering(int(nx), int(ny)):
                            continue
                        visited.add(p)
                        q.append(p)

                candidates: List[ParsedObject] = []
                for d in remaining:
                    adj = _adjacent_cells(d.pos, w, h)
                    if any(tuple(a) in visited for a in adj):
                        candidates.append(d)

                if not candidates:
                    # Fail-closed: fallback to deterministic lexical order (still auditable).
                    out.extend(remaining)
                    break

                candidates.sort(key=lambda o: (int(o.pos[1]), int(o.pos[0]), str(o.color)))
                chosen = candidates[0]
                out.append(chosen)
                opened.add((int(chosen.pos[0]), int(chosen.pos[1])))
                remaining = [d for d in remaining if d.pos != chosen.pos]

            return out

        doors_to_unlock = locked_doors if unlock_all else locked_doors[:1]
        doors_to_unlock = _order_locked_doors_by_reachability(list(doors_to_unlock)) if unlock_all else sorted(
            doors_to_unlock, key=lambda o: (int(o.pos[1]), int(o.pos[0]), str(o.color))
        )

        planned_carry_type: Optional[str] = world.carrying.obj_type if world.carrying else None
        planned_carry_color: Optional[str] = world.carrying.color if world.carrying else None
        for door in doors_to_unlock:
            door_pos = door.pos
            door_color = door.color
            # Need matching key if not carrying. IMPORTANT: in some layouts (e.g. blocked corridor),
            # we must clear movable blockers before picking up the key, because MiniGrid allows
            # carrying only one object at a time.
            keys = [o for o in world.objects if o.obj_type == "key" and (door_color is None or o.color == door_color)]
            key_pos: Optional[Tuple[int, int]] = keys[0].pos if keys else None
            needs_unbox = False
            if key_pos is None:
                # Keys can be hidden inside boxes. If boxes exist, treat this as a missing-operator gap
                # (UNBOX_KEY) rather than silently skipping the door and failing later.
                has_boxes = any(o.obj_type == "box" for o in world.objects)
                if has_boxes:
                    needs_unbox = True
                else:
                    # Door cannot be unlocked in this episode (no key, no box). Treat it as irrelevant;
                    # if it blocks the goal, navigation will fail closed.
                    continue
            if unlock_all:
                schema_calls.append("SCHEMA_UNLOCK_ONE_DOOR")

            def passable_preunlock(nx: int, ny: int) -> bool:
                obj = env.unwrapped.grid.get(int(nx), int(ny))
                if obj is None:
                    return True
                if obj.type in ("wall", "lava"):
                    return False
                # Treat movable objects as obstacles for reachability.
                if obj.type in ("key", "box", "ball"):
                    return False
                if obj.type == "door":
                    # Locked doors are only passable if we are already carrying the matching key.
                    if getattr(obj, "is_locked", False):
                        if planned_carry_type != "key":
                            return False
                        if planned_carry_color != getattr(obj, "color", None):
                            return False
                    return True
                return True

            # Clear movable blockers adjacent to the door (general MiniGrid behavior: balls/boxes can block chokepoints),
            # but only when the blocker is reachable from the current region (i.e., not behind this locked door).
            blockers = [o for o in world.objects if o.obj_type in ("ball", "box") and o.pos in _adjacent_cells(door_pos, w, h)]
            blockers.sort(key=lambda o: (int(o.pos[1]), int(o.pos[0]), str(o.obj_type), str(o.color)))
            for b in blockers:
                # Never try to "clear" the actual target object as a blocker.
                if b.pos == goal_pos:
                    continue
                # If we cannot reach an adjacent cell to pick up this blocker before unlocking, skip it.
                if _bfs_path(w=w, h=h, start=agent_pos, goals=_adjacent_cells(b.pos, w, h), passable=passable_preunlock) is None:
                    continue
                # MiniGrid constraint: cannot pick up while carrying. Ensure hands free before attempting
                # to clear blockers (otherwise the plan diverges from the env because pickup becomes a no-op).
                if planned_carry_type is not None:
                    concepts.append(
                        ConceptCall(
                            name="DROP_TO_EMPTY_NEIGHBOR",
                            args={
                                "avoid_adjacent_to": [p for p in [door_pos, key_pos, goal_pos] if p is not None],
                            },
                        )
                    )
                    planned_carry_type = None
                    planned_carry_color = None
                concepts.append(ConceptCall(name="LOCATE_BLOCKER", args={"type": b.obj_type, "color": b.color}))
                concepts.append(ConceptCall(name="NAVIGATE_ADJACENT", args={"target": b.pos}))
                concepts.append(ConceptCall(name="PICKUP_FRONT", args={"target": b.pos}))
                concepts.append(
                    ConceptCall(
                        name="DROP_TO_EMPTY_NEIGHBOR",
                        args={
                            # Avoid dropping next to critical interaction targets.
                            "avoid_adjacent_to": [p for p in [door_pos, key_pos, goal_pos] if p is not None],
                        },
                    )
                )

            # Free hands if the currently-planned carried object would prevent acquiring the required key.
            if planned_carry_type is not None:
                needs_drop = planned_carry_type != "key"
                if (not needs_drop) and door_color is not None and planned_carry_color != door_color:
                    needs_drop = True
                if needs_drop:
                    concepts.append(
                        ConceptCall(
                            name="DROP_TO_EMPTY_NEIGHBOR",
                            args={"avoid_adjacent_to": [p for p in [door_pos, key_pos, goal_pos] if p is not None]},
                        )
                    )
                    planned_carry_type = None
                    planned_carry_color = None

            if planned_carry_type is None or (door_color is not None and planned_carry_color != door_color):
                if needs_unbox:
                    concepts.append(
                        ConceptCall(
                            name="UNBOX_KEY_FOR_DOOR",
                            args={"door_color": door_color, "door_pos": door_pos},
                        )
                    )
                else:
                    assert key_pos is not None
                    concepts.append(ConceptCall(name="LOCATE_KEY", args={"color": door_color}))
                    concepts.append(ConceptCall(name="NAVIGATE_ADJACENT", args={"target": key_pos}))
                    concepts.append(ConceptCall(name="PICKUP_FRONT", args={"target": key_pos}))
                planned_carry_type = "key"
                planned_carry_color = door_color

            concepts.append(ConceptCall(name="LOCATE_LOCKED_DOOR", args={"color": door_color}))
            concepts.append(ConceptCall(name="NAVIGATE_ADJACENT", args={"target": door_pos}))
            concepts.append(ConceptCall(name="TOGGLE_FRONT", args={"target": door_pos}))

    # Always include a locate step to enforce min plan depth.
    concepts.append(ConceptCall(name="LOCATE_TARGET", args={"kind": goal.kind, "type": goal.target_type, "color": goal.target_color}))
    if goal.kind == "reach_goal":
        concepts.append(ConceptCall(name="NAVIGATE_TO", args={"target": goal_pos}))
    else:
        concepts.append(ConceptCall(name="NAVIGATE_ADJACENT", args={"target": goal_pos}))
        # MiniGrid constraint: the agent can carry only one object. Ensure free hands before pickup.
        concepts.append(ConceptCall(name="DROP_TO_EMPTY_NEIGHBOR", args={"avoid_adjacent_to": [goal_pos]}))
        concepts.append(ConceptCall(name="PICKUP_FRONT", args={"target": goal_pos}))

    # Produce primitive actions by executing concepts (schema->concept->operator).
    facts["hierarchical_ok"] = True
    actions: List[int] = []

    @dataclass
    class _Sim:
        pos: Tuple[int, int]
        dir: int
        carrying_type: Optional[str]
        carrying_color: Optional[str]
        open_doors: set = field(default_factory=set)  # set[(x,y)]
        cleared: set = field(default_factory=set)  # set[(x,y)] cells cleared by pickup
        occupied: set = field(default_factory=set)  # set[(x,y)] cells occupied by dropped objects
        dropped: Dict[Tuple[int, int], Tuple[str, Optional[str]]] = field(default_factory=dict)  # dropped objects by cell

    sim = _Sim(
        pos=agent_pos,
        dir=int(world.agent_dir),
        carrying_type=(world.carrying.obj_type if world.carrying else None),
        carrying_color=(world.carrying.color if world.carrying else None),
    )

    def is_door_open(pos: Tuple[int, int]) -> bool:
        if pos in sim.open_doors:
            return True
        obj = env.unwrapped.grid.get(int(pos[0]), int(pos[1]))
        if obj is None or obj.type != "door":
            return False
        return bool(getattr(obj, "is_open", False))

    def passable(nx: int, ny: int) -> bool:
        if (nx, ny) in sim.occupied:
            return False
        obj = env.unwrapped.grid.get(nx, ny)
        if (nx, ny) in sim.cleared:
            obj = None
        if obj is None:
            return True
        if obj.type in ("wall", "lava"):
            return False
        # Non-door objects (key/box/ball, etc.) occupy cells and are not passable in MiniGrid.
        # We must plan to adjacent cells and then use pickup, not walk through objects.
        if obj.type in ("key", "box", "ball"):
            return False
        if obj.type == "door":
            if is_door_open((nx, ny)):
                return True
            # closed/locked door: must be unlockable
            if getattr(obj, "is_locked", False):
                if sim.carrying_type != "key":
                    return False
                if sim.carrying_color != getattr(obj, "color", None):
                    return False
            return True
        return True

    def plan_follow_path(path: List[Tuple[int, int]]) -> None:
        nonlocal actions
        if len(path) <= 1:
            return
        cur = path[0]
        cur_dir = int(sim.dir)
        for nxt in path[1:]:
            dx = int(nxt[0] - cur[0])
            dy = int(nxt[1] - cur[1])
            desired_dir = _dir_for_step(dx, dy)
            if desired_dir is None:
                raise RuntimeError(f"non-4-neigh step: {cur}->{nxt}")
            turns = _turn_actions(api, cur_dir, desired_dir)
            actions.extend(turns)
            cur_dir = desired_dir

            obj = env.unwrapped.grid.get(int(nxt[0]), int(nxt[1]))
            if obj is not None and obj.type == "door":
                if not is_door_open((int(nxt[0]), int(nxt[1]))):
                    actions.append(int(api.Actions.toggle))
                    sim.open_doors.add((int(nxt[0]), int(nxt[1])))
            actions.append(int(api.Actions.forward))
            cur = nxt
        sim.pos = (int(cur[0]), int(cur[1]))
        sim.dir = int(cur_dir)

    def do_nav_to(target: Tuple[int, int], *, adjacent: bool) -> None:
        start = sim.pos
        goal_cells = _adjacent_cells(target, w, h) if adjacent else [target]
        path = _bfs_path(w=w, h=h, start=start, goals=goal_cells, passable=passable)
        if path is None:
            raise RuntimeError("no_path")
        plan_follow_path(path)

    def face_front_of(target: Tuple[int, int]) -> None:
        ax, ay = sim.pos
        tx, ty = target
        dx = tx - ax
        dy = ty - ay
        desired = _dir_for_step(dx, dy)
        if desired is None:
            raise RuntimeError("target_not_adjacent")
        for a in _turn_actions(api, int(sim.dir), int(desired)):
            actions.append(int(a))
        sim.dir = int(desired)

    def drop_to_empty_neighbor(*, avoid_adjacent_to: Sequence[Tuple[int, int]]) -> None:
        if sim.carrying_type is None:
            return
        avoid_adj = set(tuple(p) for p in (avoid_adjacent_to or []))

        def is_ok_drop_cell(pos: Tuple[int, int], *, consider_avoid: bool) -> bool:
            if pos in sim.occupied:
                return False
            obj = env.unwrapped.grid.get(int(pos[0]), int(pos[1]))
            if pos in sim.cleared:
                obj = None
            if obj is not None:
                return False
            # Avoid dropping adjacent to critical interaction targets when possible.
            if consider_avoid and avoid_adj:
                for a in avoid_adj:
                    if abs(int(pos[0]) - int(a[0])) + abs(int(pos[1]) - int(a[1])) == 1:
                        return False
            return True

        def pick_adj_drop_cell(agent_pos: Tuple[int, int], *, consider_avoid: bool, allow_relax: bool) -> Optional[Tuple[int, int]]:
            ax0, ay0 = agent_pos
            cand0 = _adjacent_cells((ax0, ay0), w, h)
            # Prefer stable, obstacle-safe drop positions.
            cand0 = sorted(cand0, key=lambda p: (int(p[1]), int(p[0])))
            for p0 in cand0:
                if is_ok_drop_cell(p0, consider_avoid=consider_avoid):
                    return p0
            if allow_relax and consider_avoid:
                for p0 in cand0:
                    if is_ok_drop_cell(p0, consider_avoid=False):
                        return p0
            return None

        def relocate_for_drop(*, consider_avoid: bool) -> Optional[Tuple[int, int]]:
            goal_agent_cells: List[Tuple[int, int]] = []
            for x in range(w):
                for y in range(h):
                    if not passable(x, y):
                        continue
                    pos0 = (int(x), int(y))
                    if pick_adj_drop_cell(pos0, consider_avoid=consider_avoid, allow_relax=False) is None:
                        continue
                    goal_agent_cells.append(pos0)
            if not goal_agent_cells:
                return None
            path = _bfs_path(w=w, h=h, start=sim.pos, goals=goal_agent_cells, passable=passable)
            if path is None:
                return None
            plan_follow_path(path)
            return pick_adj_drop_cell(sim.pos, consider_avoid=consider_avoid, allow_relax=False)

        # Prefer satisfying avoid_adjacent_to by relocating while carrying, rather than dropping back
        # into a critical chokepoint (e.g., the cell we just cleared).
        chosen = pick_adj_drop_cell(sim.pos, consider_avoid=True, allow_relax=False)
        if chosen is None:
            chosen = relocate_for_drop(consider_avoid=True)
        if chosen is None:
            chosen = pick_adj_drop_cell(sim.pos, consider_avoid=False, allow_relax=False)
        if chosen is None:
            chosen = relocate_for_drop(consider_avoid=False)

        if chosen is None:
            raise RuntimeError("no_drop_cell")
        face_front_of((int(chosen[0]), int(chosen[1])))
        actions.append(int(api.Actions.drop))
        drop_pos = (int(chosen[0]), int(chosen[1]))
        sim.occupied.add(drop_pos)
        sim.dropped[drop_pos] = (str(sim.carrying_type), sim.carrying_color)
        sim.carrying_type = None
        sim.carrying_color = None

    def try_clear_one_blocker_for_nav(*, target: Tuple[int, int], adjacent: bool) -> bool:
        # Deterministic local operator: clear a reachable movable blocker (ball/box) whose removal enables a path.
        goal_cells = _adjacent_cells(target, w, h) if adjacent else [target]
        blockers = [o for o in world.objects if o.obj_type in ("ball", "box")]
        blockers = [o for o in blockers if o.pos not in sim.occupied and o.pos not in sim.cleared and o.pos != target]
        blockers.sort(key=lambda o: (int(o.pos[1]), int(o.pos[0]), str(o.obj_type), str(o.color)))

        for b in blockers:
            if _bfs_path(w=w, h=h, start=sim.pos, goals=_adjacent_cells(b.pos, w, h), passable=passable) is None:
                continue
            # Check whether clearing this blocker makes the navigation feasible.
            sim.cleared.add((int(b.pos[0]), int(b.pos[1])))
            try:
                feasible = _bfs_path(w=w, h=h, start=sim.pos, goals=goal_cells, passable=passable) is not None
            finally:
                sim.cleared.discard((int(b.pos[0]), int(b.pos[1])))
            if not feasible:
                continue

            if sim.carrying_type is not None:
                drop_to_empty_neighbor(avoid_adjacent_to=[target])

            do_nav_to(b.pos, adjacent=True)
            face_front_of(b.pos)
            actions.append(int(api.Actions.pickup))
            sim.carrying_type = b.obj_type
            sim.carrying_color = b.color
            sim.cleared.add((int(b.pos[0]), int(b.pos[1])))
            drop_to_empty_neighbor(avoid_adjacent_to=[target])
            return True
        return False

    def _snapshot_sim_state() -> Dict[str, Any]:
        return {
            "pos": (int(sim.pos[0]), int(sim.pos[1])),
            "dir": int(sim.dir),
            "carrying_type": sim.carrying_type,
            "carrying_color": sim.carrying_color,
            "open_doors": set(sim.open_doors),
            "cleared": set(sim.cleared),
            "occupied": set(sim.occupied),
            "dropped": dict(sim.dropped),
        }

    def _restore_sim_state(snap: Dict[str, Any]) -> None:
        sim.pos = (int(snap["pos"][0]), int(snap["pos"][1]))
        sim.dir = int(snap["dir"])
        sim.carrying_type = snap.get("carrying_type")
        sim.carrying_color = snap.get("carrying_color")
        sim.open_doors = set(snap.get("open_doors") or set())
        sim.cleared = set(snap.get("cleared") or set())
        sim.occupied = set(snap.get("occupied") or set())
        sim.dropped = dict(snap.get("dropped") or {})

    def try_nav_clearing_v2(*, target: Tuple[int, int], adjacent: bool, depth_limit: int = 2) -> bool:
        """
        Deterministic operator (v2): clear up to `depth_limit` movable blockers even when no single
        blocker removal immediately yields a path (handles multi-blocker chokepoints).
        """
        blockers = [o for o in world.objects if o.obj_type in ("ball", "box")]
        blockers = [o for o in blockers if o.pos not in sim.occupied and o.pos not in sim.cleared and o.pos != target]
        # Prefer moving balls before boxes (boxes can be epistemic containers with keys).
        blockers.sort(key=lambda o: (0 if o.obj_type == "ball" else 1, int(o.pos[1]), int(o.pos[0]), str(o.color)))

        def dfs(depth: int) -> bool:
            try:
                do_nav_to(target, adjacent=adjacent)
                return True
            except RuntimeError as e:
                if str(e) != "no_path":
                    raise
                if depth >= int(depth_limit):
                    return False

            for b in blockers:
                bpos = (int(b.pos[0]), int(b.pos[1]))
                if bpos in sim.occupied or bpos in sim.cleared or bpos == target:
                    continue
                if _bfs_path(w=w, h=h, start=sim.pos, goals=_adjacent_cells(bpos, w, h), passable=passable) is None:
                    continue

                actions_len = len(actions)
                snap = _snapshot_sim_state()
                try:
                    if sim.carrying_type is not None:
                        drop_to_empty_neighbor(avoid_adjacent_to=[target])
                    do_nav_to(bpos, adjacent=True)
                    face_front_of(bpos)
                    actions.append(int(api.Actions.pickup))
                    sim.carrying_type = b.obj_type
                    sim.carrying_color = b.color
                    sim.cleared.add(bpos)
                    drop_to_empty_neighbor(avoid_adjacent_to=[target])
                    if dfs(depth + 1):
                        return True
                except RuntimeError:
                    # Treat as non-fatal inside the operator search; we revert state and try the next blocker.
                    pass
                if len(actions) > actions_len:
                    del actions[actions_len:]
                _restore_sim_state(snap)
            return False

        return bool(dfs(0))

    def nav_to_with_operators(target: Tuple[int, int], *, adjacent: bool) -> None:
        try:
            do_nav_to(target, adjacent=adjacent)
            return
        except RuntimeError as e:
            if str(e) != "no_path":
                raise
            blockers = [o for o in world.objects if o.obj_type in ("ball", "box")]
            blockers = [o for o in blockers if o.pos not in sim.occupied and o.pos not in sim.cleared and o.pos != target]
            blockers_present = bool(blockers)
            if not blockers_present:
                raise

            max_ver = max_nav_clearing_version()

            if max_ver <= 0:
                # Explicit operator gap: navigation is blocked, and we have no reusable closure that can
                # causally remove a movable blocker (ball/box). This must surface as a missing-operator
                # event so FailureCluster -> INDUCE_OPERATOR can fire deterministically (vs. "ban until die").
                raise RuntimeError("missing_operator_nav_clearing")

            # Prefer the minimal closure (v1): clear a single blocker that immediately unlocks a path.
            op_v1 = pick_nav_clearing_exact(1)
            if op_v1 is not None:
                operator_calls.append(str(op_v1))
                max_clears = 3
                for _ in range(max_clears):
                    if not try_clear_one_blocker_for_nav(target=target, adjacent=adjacent):
                        break
                    try:
                        do_nav_to(target, adjacent=adjacent)
                        return
                    except RuntimeError as e2:
                        if str(e2) != "no_path":
                            raise
                        continue

            # Multi-blocker chokepoints: expand operator family deterministically (NAV_CLEARING_Vk).
            versions = sorted({int(t[0]) for t in nav_clearing_ops if int(t[0]) >= 2})
            for ver in versions:
                op_id = pick_nav_clearing_exact(ver)
                if op_id is None:
                    continue
                operator_calls.append(str(op_id))
                # v2 uses depth_limit=3; generalize as depth_limit=ver+1 (deterministic, bounded).
                if try_nav_clearing_v2(target=target, adjacent=adjacent, depth_limit=int(ver) + 1):
                    return

            # If no available closure succeeds, but there exists a small-k clearing set that would
            # make the path feasible, demand a stronger operator instead of silently returning "no_path".
            goal_cells = _adjacent_cells(target, w, h) if adjacent else [target]
            pos_list = [(int(o.pos[0]), int(o.pos[1])) for o in blockers]
            pos_list = sorted(set(pos_list), key=lambda p: (int(p[1]), int(p[0])))[:12]
            # Deterministic operator discovery: if a small-k clearing set would unlock a path,
            # demand NAV_CLEARING_Vk explicitly (so MAXWELL_Ω can INDUCE_OPERATOR rather than "no_path").
            # Bound k to keep this check cheap (grid is tiny; combinations are manageable up to 8).
            max_k_test = min(len(pos_list), max(4, int(max_ver) + 1, 8))
            needed_k: Optional[int] = None
            for k in range(1, int(max_k_test) + 1):
                for comb in itertools.combinations(pos_list, k):
                    for p in comb:
                        sim.cleared.add((int(p[0]), int(p[1])))
                    try:
                        feasible = _bfs_path(w=w, h=h, start=sim.pos, goals=goal_cells, passable=passable) is not None
                    finally:
                        for p in comb:
                            sim.cleared.discard((int(p[0]), int(p[1])))
                    if feasible:
                        needed_k = int(k)
                        break
                if needed_k is not None:
                    break

            if needed_k is not None and int(needed_k) >= 2 and int(needed_k) > int(max_ver):
                raise RuntimeError(f"missing_operator_nav_clearing_v{int(needed_k)}")

            raise

    # Plan concepts sequentially without executing env.
    # Planning failures must never escape as exceptions: they must stay inside the ontological failure cycle.
    for c in concepts:
        try:
            if c.name == "NAVIGATE_TO":
                nav_to_with_operators(tuple(c.args["target"]), adjacent=False)
            elif c.name == "NAVIGATE_ADJACENT":
                nav_to_with_operators(tuple(c.args["target"]), adjacent=True)
            elif c.name == "UNBOX_KEY_FOR_DOOR":
                # Operator-backed concept: deterministically reveal keys hidden in boxes.
                # This is an operator gap by definition: if the operator is absent, we must fail-closed
                # with an explicit MISSING_OPERATOR reason so induction can trigger.
                door_color = c.args.get("door_color")
                door_pos = tuple(c.args.get("door_pos") or ())
                if len(door_pos) != 2:
                    raise RuntimeError("invalid_unbox_door_pos")

                # If we're already carrying the correct key, nothing to do.
                if sim.carrying_type == "key":
                    if door_color is None or sim.carrying_color == door_color:
                        continue

                if op_unbox_key_v1 is None:
                    raise RuntimeError("missing_operator_unbox_key")
                operator_calls.append(str(op_unbox_key_v1))

                # Use a deterministic env replay to observe box contents only through legal interactions
                # (toggle + pickup), avoiding any direct inspection of hidden state (box.contains).
                probe_env = api.FullyObsWrapper(api.gym.make(env_id, render_mode=None))
                try:
                    probe_env.reset(seed=seed)
                    for a0 in actions:
                        _, _, terminated0, truncated0, _ = probe_env.step(int(a0))
                        if terminated0 or truncated0:
                            raise RuntimeError("probe_env_terminated")
                    probe_stepped_len = len(actions)

                    def step_probe_new_actions() -> None:
                        nonlocal probe_stepped_len
                        for a0 in actions[probe_stepped_len:]:
                            _, _, terminated1, truncated1, _ = probe_env.step(int(a0))
                            if terminated1 or truncated1:
                                raise RuntimeError("probe_env_terminated")
                        probe_stepped_len = len(actions)

                    # First, try to reuse a previously dropped key (epistemic caching).
                    dropped_keys: List[Tuple[int, int]] = []
                    for (px, py), (t0, c0) in sim.dropped.items():
                        if str(t0) != "key":
                            continue
                        if door_color is not None and c0 != door_color:
                            continue
                        dropped_keys.append((int(px), int(py)))
                    dropped_keys.sort(key=lambda p: (int(p[1]), int(p[0])))

                    if dropped_keys:
                        kpos = dropped_keys[0]
                        nav_to_with_operators(kpos, adjacent=True)
                        step_probe_new_actions()
                        face_front_of(kpos)
                        step_probe_new_actions()
                        actions.append(int(api.Actions.pickup))
                        step_probe_new_actions()

                        carrying0 = probe_env.unwrapped.carrying
                        if carrying0 is None:
                            raise RuntimeError("unbox_pickup_failed")
                        carried_type = getattr(carrying0, "type", None)
                        carried_color = getattr(carrying0, "color", None)
                        sim.carrying_type = str(carried_type) if carried_type is not None else None
                        sim.carrying_color = carried_color
                        sim.dropped.pop((int(kpos[0]), int(kpos[1])), None)
                        sim.occupied.discard((int(kpos[0]), int(kpos[1])))

                        if sim.carrying_type == "key" and (door_color is None or sim.carrying_color == door_color):
                            continue

                    # Candidate boxes are taken from the *current* replayed env state, not from the initial
                    # WorldState snapshot, because earlier plan steps may have moved boxes.
                    boxes: List[Tuple[int, int]] = []
                    for x0 in range(int(w)):
                        for y0 in range(int(h)):
                            obj0 = probe_env.unwrapped.grid.get(int(x0), int(y0))
                            if obj0 is None:
                                continue
                            if getattr(obj0, "type", None) == "box":
                                boxes.append((int(x0), int(y0)))
                    boxes.sort(key=lambda p: (int(p[1]), int(p[0])))

                    found = False
                    for bpos in boxes:
                        # Skip boxes that are the goal itself (never destroy the target).
                        if bpos == goal_pos:
                            continue
                        # Reachability check under current simulated constraints.
                        if _bfs_path(w=w, h=h, start=sim.pos, goals=_adjacent_cells(bpos, w, h), passable=passable) is None:
                            continue

                        nav_to_with_operators(bpos, adjacent=True)
                        step_probe_new_actions()

                        face_front_of(bpos)
                        step_probe_new_actions()

                        actions.append(int(api.Actions.toggle))
                        step_probe_new_actions()

                        # After toggling, the box should reveal its contents in-place.
                        obj_after = probe_env.unwrapped.grid.get(int(bpos[0]), int(bpos[1]))
                        if obj_after is None:
                            raise RuntimeError("unbox_toggle_no_object")

                        actions.append(int(api.Actions.pickup))
                        step_probe_new_actions()

                        carrying0 = probe_env.unwrapped.carrying
                        if carrying0 is None:
                            raise RuntimeError("unbox_pickup_failed")
                        carried_type = getattr(carrying0, "type", None)
                        carried_color = getattr(carrying0, "color", None)
                        sim.carrying_type = str(carried_type) if carried_type is not None else None
                        sim.carrying_color = carried_color

                        # The opened box cell is now empty (object was picked up).
                        sim.occupied.discard((int(bpos[0]), int(bpos[1])))
                        sim.cleared.add((int(bpos[0]), int(bpos[1])))

                        if sim.carrying_type != "key":
                            # Not a key; drop and continue.
                            drop_to_empty_neighbor(avoid_adjacent_to=[door_pos, goal_pos])
                            step_probe_new_actions()
                            continue

                        if door_color is not None and sim.carrying_color != door_color:
                            # Wrong key; drop and continue.
                            drop_to_empty_neighbor(avoid_adjacent_to=[door_pos, goal_pos])
                            step_probe_new_actions()
                            continue

                        found = True
                        break

                    if not found:
                        raise RuntimeError("unbox_key_no_matching_key")
                finally:
                    try:
                        probe_env.close()
                    except Exception:
                        pass
            elif c.name == "PICKUP_FRONT":
                if sim.carrying_type is not None:
                    raise RuntimeError("pickup_while_carrying")
                target = tuple(c.args["target"])
                face_front_of(target)
                actions.append(int(api.Actions.pickup))
                # Update simulated carrying if target is key/box/ball (object-centric).
                for o in world.objects:
                    if o.pos == target and o.obj_type in ("key", "box", "ball"):
                        sim.carrying_type = o.obj_type
                        sim.carrying_color = o.color
                        sim.cleared.add((int(target[0]), int(target[1])))
                        break
            elif c.name == "DROP_TO_EMPTY_NEIGHBOR":
                avoid = [tuple(p) for p in (c.args.get("avoid_adjacent_to") or [])]
                drop_to_empty_neighbor(avoid_adjacent_to=avoid)
            elif c.name == "TOGGLE_FRONT":
                target = tuple(c.args["target"])
                # Avoid double-toggle: if navigation already crossed/auto-opened the door,
                # toggling again can close it in the real env (we model only "open" in sim).
                if is_door_open((int(target[0]), int(target[1]))):
                    continue
                face_front_of(target)
                actions.append(int(api.Actions.toggle))
                sim.open_doors.add((int(target[0]), int(target[1])))
            else:
                pass
        except Exception as e:
            msg = str(e)
            if isinstance(e, RuntimeError) and (
                msg in ("no_path", "no_drop_cell", "target_not_adjacent")
                or msg.startswith(("missing_operator_", "missing_concept_", "unbox_", "probe_env_", "invalid_"))
            ):
                facts["plan_fail_reason"] = msg
            else:
                facts["plan_fail_reason"] = type(e).__name__
            facts["plan_fail_concept"] = {"name": str(c.name), "args": dict(c.args)}
            facts["plan_fail_sim"] = {
                "pos": [int(sim.pos[0]), int(sim.pos[1])],
                "dir": int(sim.dir),
                "carrying_type": sim.carrying_type,
                "carrying_color": sim.carrying_color,
            }
            facts["operator_calls"] = sorted({str(x) for x in operator_calls if str(x)})
            return None, facts

    plan = Plan(
        schema_id=schema_id,
        schema_name=schema_name,
        concepts=tuple(concepts),
        actions=tuple(actions),
        schema_calls=tuple(schema_calls),
        operator_calls=tuple(operator_calls),
    )
    facts["operator_calls"] = sorted({str(x) for x in operator_calls if str(x)})
    if plan.depth < int(min_plan_depth):
        return None, facts
    return plan, facts


def _episode_solved(env: Any, goal: GoalSpec) -> bool:
    if goal.kind == "reach_goal":
        ax, ay = (int(x) for x in env.unwrapped.agent_pos)
        obj = env.unwrapped.grid.get(ax, ay)
        return bool(obj is not None and getattr(obj, "type", None) == "goal")
    # pickup
    carrying = env.unwrapped.carrying
    if carrying is None:
        return False
    if getattr(carrying, "type", None) != goal.target_type:
        return False
    if goal.target_color is None:
        return True
    return getattr(carrying, "color", None) == goal.target_color


# ----------------------------
# Main training loop
# ----------------------------


def _parse_env_sequence(arg: str) -> List[str]:
    out = []
    for part in (arg or "").split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="ARC zero-touch structural training (MiniGrid + coercion, no RL).")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--jobs", type=int, default=1, help="Parallel workers (batch mode). Default 1.")
    ap.add_argument("--env", default="MiniGrid-DoorKey-8x8-v0")
    ap.add_argument("--env_sequence", default="", help="Comma-separated env IDs; overrides --env if set.")
    ap.add_argument("--env_switch_every", type=int, default=5000)
    ap.add_argument("--out", default="", help="WORM output dir. Default: results/structural_train_<ts>")

    ap.add_argument("--min_plan_depth", type=int, default=3)
    ap.add_argument("--fail_closed_required", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--forbid_shallow_paths", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--stop_on_regime", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--regime_window", type=int, default=1000, help="Rolling window for Z(t) stability.")
    ap.add_argument("--log_every", type=int, default=1)

    ap.add_argument("--failure_cluster_k", type=int, default=10, help="Failures needed to trigger INDUCE_SCHEMA.")
    ap.add_argument("--failure_cluster_cooldown_steps", type=int, default=250, help="Min steps between inductions per cluster.")
    ap.add_argument("--failure_cluster_max_induce_attempts", type=int, default=3, help="Max induction attempts per cluster.")
    ap.add_argument(
        "--dead_end_on_uninducible_cluster",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop training if a persistent failure cluster cannot induce a new schema.",
    )
    ap.add_argument(
        "--dead_end_progress_timeout_steps",
        type=int,
        default=5000,
        help="If a failure cluster persists this long after induction attempts without resolving, stop (ontological dead-end).",
    )
    ap.add_argument("--omega", action=argparse.BooleanOptionalAction, default=True, help="Enable Ω (destructive future burn).")
    ap.add_argument(
        "--omega_state_in",
        default="",
        help="Optional previous omega_state_structural_v1.json (WORM). Used to persist bans across restarts/runs.",
    )
    ap.add_argument(
        "--schema_bank_in",
        default="",
        help="Optional previous schema_bank.json (WORM). Used to persist learned schemas across runs.",
    )
    ap.add_argument(
        "--operator_bank_in",
        default="",
        help="Optional previous operator_bank.json (WORM). Used to persist learned operators across runs.",
    )
    ap.add_argument(
        "--omega_bootstrap_steps",
        type=int,
        default=0,
        help="Delay strict-burn until this episode index (cold-start only; default 0 = fully strict).",
    )
    ap.add_argument(
        "--omega_max_consecutive_skips",
        type=int,
        default=2000,
        help="If this many consecutive episodes are Ω-skipped, declare omega-dead (reachable future exhausted).",
    )
    ap.add_argument(
        "--promote_support",
        type=int,
        default=3,
        help="Minimum supports before a schema is eligible for preserves_future() promotion gating.",
    )
    ap.add_argument(
        "--promotion_cooldown_steps",
        type=int,
        default=250,
        help="Minimum steps between preserves_future() promotion attempts per schema (bounded ablation).",
    )
    ap.add_argument("--ablation_trials", type=int, default=3, help="Ablation trials per promotion attempt (max).")
    args = ap.parse_args()

    api = _require_minigrid()

    env_sequence = _parse_env_sequence(str(args.env_sequence))
    if not env_sequence:
        env_sequence = [str(args.env)]
    # MiniGrid 3.x does not ship "MiniGrid-Unlock-v0"; the closest built-in is BlockedUnlockPickup.
    env_aliases = {"MiniGrid-Unlock-v0": "MiniGrid-BlockedUnlockPickup-v0"}
    env_sequence = [env_aliases.get(e, e) for e in env_sequence]

    # If we want "regime stability" across an env schedule, the rolling window must span
    # at least one full cycle of the env_sequence; otherwise we'd be certifying a regime
    # on a strict subset of environments.
    if bool(args.stop_on_regime) and len(env_sequence) > 1:
        min_window = int(args.env_switch_every) * len(env_sequence)
        if int(args.regime_window) < min_window:
            _fail(
                f"--regime_window too small for env_sequence coverage: got {int(args.regime_window)}, "
                f"need >= {min_window} (= env_switch_every*len(env_sequence))."
            )

    out_dir = str(args.out).strip() or os.path.join("results", f"structural_train_v3_{_now_ts()}_seed{int(args.seed)}")
    ensure_absent(out_dir)
    ensure_dir(out_dir)

    paths = {
        "config": os.path.join(out_dir, "config.json"),
        "progress": os.path.join(out_dir, "progress.jsonl"),
        "episodes": os.path.join(out_dir, "episodes.jsonl"),
        "failures": os.path.join(out_dir, "failures.jsonl"),
        "schema_events": os.path.join(out_dir, "schema_events.jsonl"),
        "schema_bank": os.path.join(out_dir, "schema_bank.json"),
        "operator_events": os.path.join(out_dir, "operator_events.jsonl"),
        "operator_bank": os.path.join(out_dir, "operator_bank.json"),
        "failure_clusters": os.path.join(out_dir, "failure_clusters.json"),
        "omega_events": os.path.join(out_dir, "omega_events.jsonl"),
        "omega_state": os.path.join(out_dir, "omega_state_structural_v1.json"),
        "summary": os.path.join(out_dir, "summary.json"),
    }

    thresholds = _default_z_thresholds()
    cfg = {
        "schema_version": 1,
        "kind": "structural_training_run_v3",
        "created_at": _now_ts(),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "jobs": int(args.jobs),
        "parallel_mode": "batch" if int(args.jobs) > 1 else "serial",
        "parallel_merge_order": ["env_id", "seed", "outcome"],
        "env_sequence": env_sequence,
        "env_aliases": dict(env_aliases),
        "env_switch_every": int(args.env_switch_every),
        "min_plan_depth": int(args.min_plan_depth),
        "fail_closed_required": bool(args.fail_closed_required),
        "forbid_shallow_paths": bool(args.forbid_shallow_paths),
        "stop_on_regime": bool(args.stop_on_regime),
        "regime_window": int(args.regime_window),
        "z_thresholds": dict(thresholds),
        "failure_cluster_k": int(args.failure_cluster_k),
        "failure_cluster_cooldown_steps": int(args.failure_cluster_cooldown_steps),
        "failure_cluster_max_induce_attempts": int(args.failure_cluster_max_induce_attempts),
        "dead_end_on_uninducible_cluster": bool(args.dead_end_on_uninducible_cluster),
        "dead_end_progress_timeout_steps": int(args.dead_end_progress_timeout_steps),
        "omega_enabled": bool(args.omega),
        "omega_state_in": str(args.omega_state_in or "").strip(),
        "schema_bank_in": str(args.schema_bank_in or "").strip(),
        "operator_bank_in": str(args.operator_bank_in or "").strip(),
        "omega_bootstrap_steps": int(args.omega_bootstrap_steps),
        "omega_max_consecutive_skips": int(args.omega_max_consecutive_skips),
        "promote_support": int(args.promote_support),
        "promotion_cooldown_steps": int(args.promotion_cooldown_steps),
        "ablation_trials": int(args.ablation_trials),
        "python": sys.version,
    }
    _write_json(paths["config"], cfg)

    bank = SchemaBank(promote_support=int(args.promote_support), ttl_steps=2000, min_support_to_keep=1)
    op_bank = OperatorBank(promote_support=int(args.promote_support), ttl_steps=2000, min_support_to_keep=1)
    schema_bank_in = str(args.schema_bank_in or "").strip()
    if schema_bank_in:
        _load_schema_bank_into(bank, schema_bank_in)
    operator_bank_in = str(args.operator_bank_in or "").strip()
    if operator_bank_in:
        _load_operator_bank_into(op_bank, operator_bank_in)
    metrics = StructuralMetrics(window=int(args.regime_window))
    failure_clusters = FailureClusterStore(
        threshold=int(args.failure_cluster_k),
        cooldown_steps=int(args.failure_cluster_cooldown_steps),
        max_induce_attempts=int(args.failure_cluster_max_induce_attempts),
    )
    schema_contexts: Dict[str, set[str]] = {}
    schema_witnesses: Dict[str, List[Dict[str, Any]]] = {}
    operator_contexts: Dict[str, set[str]] = {}
    operator_witnesses: Dict[str, List[Dict[str, Any]]] = {}
    regime_reached_at: Optional[int] = None
    stop_after_batch = False
    dead_end: Optional[Dict[str, Any]] = None

    omega_enabled = bool(args.omega)
    omega_prev_in = str(args.omega_state_in or "").strip()
    omega_prev_state: Optional[OmegaStateStructuralV1] = None
    if omega_enabled and omega_prev_in:
        omega_prev_state = OmegaStateStructuralV1.from_path(omega_prev_in)
    omega_state: Optional[OmegaStateStructuralV1] = omega_prev_state
    omega_banned_env_ids: set[str] = set(omega_prev_state.banned_env_ids) if omega_prev_state is not None else set()
    omega_banned_cluster_ctx_keys: set[str] = (
        set(omega_prev_state.banned_cluster_ctx_keys) if omega_prev_state is not None else set()
    )
    omega_first_burn_step: Optional[int] = None
    omega_consecutive_skips = 0

    omega_model_cluster_ctx_keys: set[str] = set()
    if omega_enabled:
        colors = sorted({str(c) for c in api.IDX_TO_COLOR.values() if str(c)})
        buckets = ["0", "1", "2+"]
        for doors_bucket in buckets:
            for keys_bucket in buckets:
                for blockers_adj in (False, True):
                    omega_model_cluster_ctx_keys.add(
                        _cluster_ctx_key(
                            {
                                "goal_kind": "reach_goal",
                                "target_type": "goal",
                                "target_color": None,
                                "doors_locked_bucket": doors_bucket,
                                "keys_bucket": keys_bucket,
                                "blockers_adj_locked_door": bool(blockers_adj),
                            }
                        )
                    )
                    for ttype in ("key", "ball", "box"):
                        for color in colors:
                            omega_model_cluster_ctx_keys.add(
                                _cluster_ctx_key(
                                    {
                                        "goal_kind": "pickup",
                                        "target_type": ttype,
                                        "target_color": color,
                                        "doors_locked_bucket": doors_bucket,
                                        "keys_bucket": keys_bucket,
                                        "blockers_adj_locked_door": bool(blockers_adj),
                                    }
                                )
                            )

    def make_omega_state(
        prev: Optional[OmegaStateStructuralV1],
        *,
        banned_env_ids: set[str],
        banned_cluster_ctx_keys: set[str],
        strict_burns_added: int,
    ) -> OmegaStateStructuralV1:
        prev_sha = str(prev.state_sha) if prev is not None else ""
        strict_total = int(prev.strict_burns_total) if prev is not None else 0
        strict_total = int(strict_total + int(strict_burns_added))
        st = OmegaStateStructuralV1(
            kind="omega_state_structural_v1",
            created_at=_now_ts(),
            prev_state_sha=prev_sha,
            state_sha="",
            strict_burns_total=int(strict_total),
            banned_env_ids=tuple(sorted({str(x) for x in banned_env_ids if str(x)})),
            banned_cluster_ctx_keys=tuple(sorted({str(x) for x in banned_cluster_ctx_keys if str(x)})),
        )
        return OmegaStateStructuralV1(**{**st.__dict__, "state_sha": st.content_hash()})

    jobs = max(1, int(args.jobs))
    executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
    if jobs > 1:
        mp_ctx = multiprocessing.get_context("spawn")
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=jobs, mp_context=mp_ctx, initializer=_worker_init)

    try:
        total_steps = int(args.steps)
        env_switch_every = int(args.env_switch_every)
        for batch_start in range(0, total_steps, jobs):
            batch_ids = list(range(batch_start, min(batch_start + jobs, total_steps)))
            snapshot = _freeze_schema_snapshot(bank)
            op_snapshot = _freeze_operator_snapshot(op_bank)
            omega_env_bans = sorted(omega_banned_env_ids) if omega_enabled else []
            omega_cluster_bans = sorted(omega_banned_cluster_ctx_keys) if omega_enabled else []

            payloads: List[Dict[str, Any]] = []
            for ep_idx in batch_ids:
                env_id = env_sequence[(ep_idx // env_switch_every) % len(env_sequence)]
                payloads.append(
                    {
                        "episode_idx": int(ep_idx),
                        "env_id": env_id,
                        "seed": int(args.seed) + int(ep_idx),
                        "schemas": snapshot["schemas"],
                        "operators": op_snapshot["operators"],
                        "min_plan_depth": int(args.min_plan_depth),
                        "forbid_shallow_paths": bool(args.forbid_shallow_paths),
                        "fail_closed_required": bool(args.fail_closed_required),
                        "omega_banned_env_ids": omega_env_bans,
                        "omega_banned_cluster_ctx_keys": omega_cluster_bans,
                    }
                )

            if executor is None:
                results = [_run_episode_core(api, p) for p in payloads]
            else:
                futures = [executor.submit(_run_episode_worker, p) for p in payloads]
                results = []
                for fut, p in zip(futures, payloads):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        exc_type = type(e).__name__
                        exc_cluster_ctx_key = _stable_sha16({"exception_stage": "worker", "exception_type": exc_type})
                        exc_ctx_key = _stable_sha16(
                            {"env_id": str(p.get("env_id") or ""), "cluster_ctx_key": exc_cluster_ctx_key}
                        )
                        results.append(
                            {
                                "env_id": str(p.get("env_id")),
                                "seed": int(p.get("seed", 0)),
                                "outcome": "FAIL_CLOSED",
                                "reason": f"WORKER_EXCEPTION_{exc_type}",
                                "exception_type": exc_type,
                                "exception_msg": str(e),
                                "did_fail_closed": True,
                                "executed_with_plan": False,
                                "episode_success": False,
                                "terminated": False,
                                "truncated": False,
                                "steps_taken": 0,
                                "plan_depth": 0,
                                "schema_calls": [],
                                "operator_calls": [],
                                "schema_selected": None,
                                "schema_selected_id": None,
                                "schema_missing": False,
                                "schema_suggestion": None,
                                "schema_used_id": None,
                                "schema_used_name": None,
                                "world_features": None,
                                "cluster_ctx_key": exc_cluster_ctx_key,
                                "pre_search_schema": False,
                                "object_centric_ok": False,
                                "conditional_ok": False,
                                "hierarchical_ok": False,
                                "shallow_suppressed": False,
                                "failure_to_schema": False,
                                "plan_fail_reason": None,
                                "plan_fail_concept": None,
                                "plan_fail_sim": None,
                                "ctx_key": exc_ctx_key,
                                "goal": None,
                            }
                        )

            # Attach episode_idx deterministically from payloads.
            for res, p in zip(results, payloads):
                res["episode_idx"] = int(p["episode_idx"])

            # Deterministic merge order (required).
            results.sort(key=lambda r: (str(r.get("env_id")), int(r.get("seed", 0)), str(r.get("outcome"))))

            # Apply structural updates one-by-one, then update metrics/logs.
            batch_failures_seen = 0
            batch_promoted_delta: List[str] = []
            batch_last_processed_step: Optional[int] = None
            for res in results:
                ep_idx = int(res["episode_idx"])
                batch_last_processed_step = ep_idx
                failure_to_schema_event = False
                failure_to_operator_event = False
                failure_to_structure_event = False
                episode_skipped = bool(res.get("episode_skipped", False))
                episode_success = bool(res.get("episode_success", False))
                operator_calls_in_ep = [str(x) for x in (res.get("operator_calls") or []) if str(x)]

                if episode_skipped:
                    omega_consecutive_skips += 1
                else:
                    omega_consecutive_skips = 0

                if omega_enabled and omega_consecutive_skips >= int(args.omega_max_consecutive_skips):
                    dead_end = {
                        "step": ep_idx,
                        "reason": "OMEGA_DEAD_EXHAUSTED_FUTURE",
                        "omega": {
                            "consecutive_skips": int(omega_consecutive_skips),
                            "banned_env_ids_total": int(len(omega_banned_env_ids)),
                            "banned_cluster_ctx_keys_total": int(len(omega_banned_cluster_ctx_keys)),
                            "reachable_cluster_ctx_keys_total": int(len(omega_model_cluster_ctx_keys)),
                        },
                    }
                    stop_after_batch = True

                if episode_skipped:
                    z = metrics.current_z()
                    lb = _latent_lower_bound(z, thresholds)
                    regime_met_now = _regime_met(z, thresholds)

                    if (ep_idx % int(args.log_every)) == 0:
                        promoted_ids = [sid for sid, s in bank.schemas.items() if s.promoted]
                        promoted_ctx_counts = [len(schema_contexts.get(sid, set())) for sid in promoted_ids]
                        z7_cross_context = float(sum(promoted_ctx_counts) / max(1, len(promoted_ctx_counts)))
                        promoted_op_ids = [oid for oid, op in op_bank.operators.items() if op.promoted]
                        promoted_op_ctx_counts = [len(operator_contexts.get(oid, set())) for oid in promoted_op_ids]
                        z7_op_cross_context = float(sum(promoted_op_ctx_counts) / max(1, len(promoted_op_ctx_counts)))
                        _append_jsonl(
                            paths["progress"],
                            {
                                "step": ep_idx,
                                "env_id": str(res.get("env_id")),
                                "seed": int(res.get("seed", 0)),
                                "bank": {**bank.stats(), "schemas_cross_context_avg": z7_cross_context},
                                "operators": {**op_bank.stats(), "operators_cross_context_avg": z7_op_cross_context},
                                "z": z,
                                "latent_lower_bound": lb,
                                "regime_met": bool(regime_met_now),
                                "episode_skipped": True,
                                "skip_reason": str(res.get("skip_reason") or res.get("reason") or ""),
                                "omega": {
                                    "enabled": bool(omega_enabled),
                                    "state_sha": str(getattr(omega_state, "state_sha", "") or ""),
                                    "prev_state_sha": str(getattr(omega_state, "prev_state_sha", "") or ""),
                                    "strict_burns_total": int(getattr(omega_state, "strict_burns_total", 0) or 0)
                                    if omega_state is not None
                                    else 0,
                                    "banned_env_ids_total": int(len(omega_banned_env_ids)),
                                    "banned_cluster_ctx_keys_total": int(len(omega_banned_cluster_ctx_keys)),
                                    "reachable_cluster_ctx_keys_total": int(len(omega_model_cluster_ctx_keys)),
                                },
                            },
                        )

                    _append_jsonl(
                        paths["episodes"],
                        {
                            "step": ep_idx,
                            "env_id": str(res.get("env_id")),
                            "seed": int(res.get("seed", 0)),
                            "outcome": res.get("outcome"),
                            "reason": res.get("reason"),
                            "episode_skipped": True,
                            "skip_reason": str(res.get("skip_reason") or ""),
                            "concept_calls": 0,
                            "operator_calls": [],
                            "goal": res.get("goal"),
                            "cluster_ctx_key": res.get("cluster_ctx_key"),
                            "world_features": res.get("world_features"),
                        },
                    )

                    if dead_end is not None:
                        break
                    continue

                suggestion = res.get("schema_suggestion")
                if isinstance(suggestion, dict):
                    # A schema-missing failure is explicit evidence that must be converted into structure.
                    failure_to_schema_event = True
                    failure_to_structure_event = True
                    sid = str(suggestion.get("schema_id"))
                    if sid and sid not in bank.schemas:
                        schema = _induce_schema(
                            ep_idx,
                            str(suggestion.get("name")),
                            dict(suggestion.get("signature") or {}),
                            origin="schema_missing",
                            state="candidate",
                            components=_schema_components_for_name(str(suggestion.get("name"))),
                        )
                        bank.add(schema)
                        _append_jsonl(paths["schema_events"], {"step": ep_idx, "event": "schema_added", "schema": schema.to_json()})

                selected_schema_id = res.get("schema_selected_id")
                selected_schema_name = res.get("schema_selected")
                if isinstance(selected_schema_id, str) and selected_schema_id in bank.schemas:
                    if episode_success:
                        ctx_key = res.get("ctx_key")
                        if isinstance(ctx_key, str) and ctx_key:
                            schema_contexts.setdefault(selected_schema_id, set()).add(ctx_key)
                        # Promotion eligibility is computed in SchemaBank; promotion itself is gated by preserves_future().
                        eligible_for_promotion = bank.mark_success(selected_schema_id, ep_idx)

                        # Ablation witnesses: keep a small deterministic set of successful episodes for this schema.
                        env_id_s = str(res.get("env_id") or "")
                        seed_s = int(res.get("seed") or 0)
                        wl0 = schema_witnesses.get(selected_schema_id) or []
                        wl0 = list(wl0) + [{"env_id": env_id_s, "seed": seed_s}]
                        wl0 = _sorted_unique_ablation_witnesses(wl0)[:25]
                        schema_witnesses[selected_schema_id] = wl0

                        s0 = bank.schemas.get(selected_schema_id)
                        if s0 is not None and eligible_for_promotion:
                            last_try = int(s0.last_promotion_attempt_step)
                            if last_try < 0 or (ep_idx - last_try) >= int(args.promotion_cooldown_steps):
                                s0.promotion_attempts += 1
                                s0.last_promotion_attempt_step = ep_idx

                                contexts_used = len(schema_contexts.get(selected_schema_id, set()))
                                origin_cluster_resolved = False
                                if str(s0.origin) == "failure_cluster" and s0.origin_cluster_id:
                                    c0 = failure_clusters.clusters.get(str(s0.origin_cluster_id))
                                    origin_cluster_resolved = bool(c0.resolved) if c0 is not None else False
                                elif str(s0.origin) == "schema_missing":
                                    origin_cluster_resolved = True

                                ok_pf, evidence = _preserves_future_schema_v1(
                                    api=api,
                                    bank=bank,
                                    op_bank=op_bank,
                                    schema=s0,
                                    contexts_used=int(contexts_used),
                                    origin_cluster_resolved=bool(origin_cluster_resolved),
                                    omega_enabled=bool(omega_enabled),
                                    omega_first_burn_step=omega_first_burn_step,
                                    ablation_trials=int(args.ablation_trials),
                                    ablation_witnesses=schema_witnesses.get(selected_schema_id) or [],
                                    min_plan_depth=int(args.min_plan_depth),
                                    forbid_shallow_paths=bool(args.forbid_shallow_paths),
                                    fail_closed_required=bool(args.fail_closed_required),
                                )
                                evidence = {**dict(evidence), "step": int(ep_idx), "schema_name": str(s0.name)}

                                _append_jsonl(
                                    paths["schema_events"],
                                    {
                                        "step": ep_idx,
                                        "event": "schema_promotion_attempt",
                                        "schema_id": selected_schema_id,
                                        "schema_name": str(s0.name),
                                        "ok": bool(ok_pf),
                                        "evidence": evidence,
                                    },
                                )

                                if ok_pf and bank.promote(selected_schema_id, ep_idx, evidence):
                                    batch_promoted_delta.append(str(selected_schema_id))
                                    _append_jsonl(
                                        paths["schema_events"],
                                        {
                                            "step": ep_idx,
                                            "event": "schema_promoted",
                                            "schema_id": selected_schema_id,
                                            "schema_name": str(s0.name),
                                            "evidence": evidence,
                                        },
                                    )
                    else:
                        bank.mark_failure(selected_schema_id, ep_idx)

                # Operator updates (conceptual operator-calls inside the plan).
                for op_id in operator_calls_in_ep:
                    if op_id not in op_bank.operators:
                        continue
                    if episode_success:
                        env_id_s = str(res.get("env_id") or "")
                        seed_s = int(res.get("seed") or 0)
                        op_ctx_key = _stable_sha16({"env_id": env_id_s, "seed": seed_s})
                        operator_contexts.setdefault(op_id, set()).add(op_ctx_key)
                        eligible_for_promotion = op_bank.mark_success(op_id, ep_idx)

                        wl0 = operator_witnesses.get(op_id) or []
                        wl0 = list(wl0) + [{"env_id": env_id_s, "seed": seed_s}]
                        wl0 = _sorted_unique_ablation_witnesses(wl0)[:25]
                        operator_witnesses[op_id] = wl0

                        op0 = op_bank.operators.get(op_id)
                        if op0 is not None and eligible_for_promotion:
                            last_try = int(op0.last_promotion_attempt_step)
                            if last_try < 0 or (ep_idx - last_try) >= int(args.promotion_cooldown_steps):
                                op0.promotion_attempts += 1
                                op0.last_promotion_attempt_step = ep_idx

                                contexts_used = len(operator_contexts.get(op_id, set()))
                                origin_cluster_resolved = False
                                if str(op0.origin) == "failure_cluster" and op0.origin_cluster_id:
                                    c0 = failure_clusters.clusters.get(str(op0.origin_cluster_id))
                                    origin_cluster_resolved = bool(c0.resolved) if c0 is not None else False

                                ok_pf, evidence = _preserves_future_operator_v1(
                                    api=api,
                                    bank=bank,
                                    op_bank=op_bank,
                                    operator=op0,
                                    contexts_used=int(contexts_used),
                                    origin_cluster_resolved=bool(origin_cluster_resolved),
                                    omega_enabled=bool(omega_enabled),
                                    omega_first_burn_step=omega_first_burn_step,
                                    ablation_trials=int(args.ablation_trials),
                                    ablation_witnesses=operator_witnesses.get(op_id) or [],
                                    min_plan_depth=int(args.min_plan_depth),
                                    forbid_shallow_paths=bool(args.forbid_shallow_paths),
                                    fail_closed_required=bool(args.fail_closed_required),
                                )
                                evidence = {**dict(evidence), "step": int(ep_idx), "operator_name": str(op0.name)}

                                _append_jsonl(
                                    paths["operator_events"],
                                    {
                                        "step": ep_idx,
                                        "event": "operator_promotion_attempt",
                                        "operator_id": op_id,
                                        "operator_name": str(op0.name),
                                        "ok": bool(ok_pf),
                                        "evidence": evidence,
                                    },
                                )

                                if ok_pf and op_bank.promote(op_id, ep_idx, evidence):
                                    batch_promoted_delta.append(str(op_id))
                                    _append_jsonl(
                                        paths["operator_events"],
                                        {
                                            "step": ep_idx,
                                            "event": "operator_promoted",
                                            "operator_id": op_id,
                                            "operator_name": str(op0.name),
                                            "evidence": evidence,
                                        },
                                    )
                    else:
                        op_bank.mark_failure(op_id, ep_idx)

                if not episode_success:
                    batch_failures_seen += 1

                # Failure clustering + induction (schema must also be born from recurring failure).
                cluster_ctx_key = res.get("cluster_ctx_key")
                if episode_success and isinstance(cluster_ctx_key, str) and cluster_ctx_key:
                    failure_clusters.mark_ctx_resolved(cluster_ctx_key)

                if not episode_success and isinstance(cluster_ctx_key, str) and cluster_ctx_key:
                    fail_reason = str(res.get("reason") or res.get("outcome") or "")
                    goal = res.get("goal") or {}
                    pf_concept = res.get("plan_fail_concept") or {}
                    pf_name = ""
                    if isinstance(pf_concept, dict):
                        pf_name = str(pf_concept.get("name") or "")
                    failure_sig = {
                        "cluster_ctx_key": cluster_ctx_key,
                        "goal_kind": str(goal.get("kind") or ""),
                        "target_type": str(goal.get("target_type") or ""),
                        "target_color": goal.get("target_color"),
                        "schema_name": str(selected_schema_name or ""),
                        "reason": fail_reason,
                        "plan_fail_concept": ({"name": pf_name} if pf_name else None),
                        "world_features": dict(res.get("world_features") or {}),
                    }
                    cluster = failure_clusters.observe(
                        failure_sig,
                        step=ep_idx,
                        witness_env_id=str(res.get("env_id") or ""),
                        witness_seed=int(res.get("seed") or 0),
                    )
                    _append_jsonl(
                        paths["failures"],
                        {
                            "step": ep_idx,
                            "cluster_id": cluster.cluster_id,
                            "cluster_count": int(cluster.count),
                            "signature": failure_sig,
                        },
                    )

                    # Schema-missing is already converted into structure via schema_suggestion/bank.add;
                    # it should be clusterable/burnable, but never treated as "uninducible" dead-end.
                    is_schema_missing_cluster = bool(failure_to_schema_event) and fail_reason == "SCHEMA_MISSING"

                    if not is_schema_missing_cluster:
                        for c in failure_clusters.eligible_clusters(step=ep_idx):
                            c.induce_attempts += 1
                            c.last_induce_step = ep_idx
                            schema = _induce_schema_from_failure_cluster(ep_idx, c)
                            schema_added = False
                            if schema is not None:
                                added = bank.add(schema)
                                if added:
                                    schema_added = True
                                    c.induced_schema_ids.append(schema.schema_id)
                                    c.last_progress_step = ep_idx
                                    _append_jsonl(
                                        paths["schema_events"],
                                        {
                                            "step": ep_idx,
                                            "event": "schema_induced_from_failure",
                                            "cluster_id": c.cluster_id,
                                            "schema": schema.to_json(),
                                        },
                                    )
                                    failure_to_schema_event = True
                                    failure_to_structure_event = True
                                else:
                                    # Even if already present, record linkage for governance.
                                    c.induced_schema_ids.append(schema.schema_id)

                            op_added = False
                            # If schema induction didn't create new structure (either None or already present),
                            # attempt operator discovery as a strict fallback (avoid "ban until die").
                            if not schema_added:
                                op = _induce_operator_from_failure_cluster(ep_idx, c)
                                if op is not None:
                                    added_op = op_bank.add(op)
                                    if added_op:
                                        op_added = True
                                        c.induced_operator_ids.append(op.operator_id)
                                        c.last_progress_step = ep_idx
                                        _append_jsonl(
                                            paths["operator_events"],
                                            {
                                                "step": ep_idx,
                                                "event": "operator_induced_from_failure",
                                                "cluster_id": c.cluster_id,
                                                "operator": op.to_json(),
                                            },
                                        )
                                        # MAXWELL_Ω: immediately validate and (if warranted) promote the induced operator
                                        # using only the cluster's deterministic witness episodes. This closes the escape
                                        # route where Ω burns before INDUCE_OPERATOR can produce a promoted delta.
                                        try:
                                            promoted_now, mm = _maxwell_promote_operator_from_cluster_v1(
                                                api=api,
                                                bank=bank,
                                                op_bank=op_bank,
                                                failure_clusters=failure_clusters,
                                                cluster=c,
                                                operator_id=str(op.operator_id),
                                                step=int(ep_idx),
                                                operator_contexts=operator_contexts,
                                                operator_witnesses=operator_witnesses,
                                                omega_enabled=bool(omega_enabled),
                                                omega_first_burn_step=omega_first_burn_step,
                                                min_plan_depth=int(args.min_plan_depth),
                                                forbid_shallow_paths=bool(args.forbid_shallow_paths),
                                                fail_closed_required=bool(args.fail_closed_required),
                                                ablation_trials=int(args.ablation_trials),
                                                promotion_cooldown_steps=int(args.promotion_cooldown_steps),
                                            )
                                        except Exception as e:
                                            promoted_now = False
                                            mm = {"enabled": True, "ok": False, "reason": f"maxwell_exception:{type(e).__name__}"}

                                        _append_jsonl(
                                            paths["operator_events"],
                                            {
                                                "step": ep_idx,
                                                "event": "operator_maxwell_promotion_attempt",
                                                "cluster_id": c.cluster_id,
                                                "operator_id": str(op.operator_id),
                                                "operator_name": str(op.name),
                                                "meta": mm,
                                            },
                                        )
                                        if bool(promoted_now):
                                            batch_promoted_delta.append(str(op.operator_id))
                                            _append_jsonl(
                                                paths["operator_events"],
                                                {
                                                    "step": ep_idx,
                                                    "event": "operator_promoted_maxwell",
                                                    "cluster_id": c.cluster_id,
                                                    "operator_id": str(op.operator_id),
                                                    "operator_name": str(op.name),
                                                    "evidence": dict(mm.get("promotion_evidence") or {}),
                                                },
                                            )
                                        failure_to_operator_event = True
                                        failure_to_structure_event = True
                                    else:
                                        c.induced_operator_ids.append(op.operator_id)

                            if schema is None and (not schema_added) and (not op_added):
                                if bool(args.dead_end_on_uninducible_cluster):
                                    c.dead_end = True
                                    dead_end = {
                                        "step": ep_idx,
                                        "reason": "UNINDUCIBLE_FAILURE_CLUSTER",
                                        "cluster": {
                                            "cluster_id": c.cluster_id,
                                            "count": int(c.count),
                                            "signature": dict(c.signature),
                                            "induce_attempts": int(c.induce_attempts),
                                        },
                                    }
                                    stop_after_batch = True
                                    break
                        if dead_end is not None:
                            break

                        # Ontological dead-end: a persistent failure cluster that cannot be resolved
                        # (no new structure emerges despite repeated induction pressure).
                        if dead_end is None and bool(args.dead_end_on_uninducible_cluster):
                            if (not cluster.resolved) and cluster.count >= int(args.failure_cluster_k):
                                if cluster.induce_attempts >= int(args.failure_cluster_max_induce_attempts):
                                    last_prog = int(cluster.last_progress_step)
                                    if last_prog < 0 or (ep_idx - last_prog) >= int(args.dead_end_progress_timeout_steps):
                                        cluster.dead_end = True
                                        dead_end = {
                                            "step": ep_idx,
                                            "reason": "PERSISTENT_FAILURE_CLUSTER",
                                            "cluster": {
                                                "cluster_id": cluster.cluster_id,
                                                "count": int(cluster.count),
                                                "signature": dict(cluster.signature),
                                                "induce_attempts": int(cluster.induce_attempts),
                                                "last_progress_step": int(cluster.last_progress_step),
                                            },
                                        }
                                        stop_after_batch = True
                                        break

                metrics.update_episode(
                    episode_success=episode_success,
                    did_fail_closed=bool(res.get("did_fail_closed", False)),
                    did_pre_search_schema=bool(res.get("pre_search_schema", False)),
                    executed_with_plan=bool(res.get("executed_with_plan", False)),
                    object_centric_ok=bool(res.get("object_centric_ok", False)),
                    conditional_ok=bool(res.get("conditional_ok", False)),
                    hierarchical_ok=bool(res.get("hierarchical_ok", False)),
                    shallow_applicable=bool(((res.get("goal") or {}).get("kind") or "") == "reach_goal"),
                    shallow_suppressed=bool(res.get("shallow_suppressed", False)),
                    failure_to_structure=bool(failure_to_structure_event),
                    operator_calls=int(len(res.get("operator_calls") or [])),
                    concept_calls=int(res.get("concept_calls") or 0),
                    fail_reason=str(res.get("reason") or ""),
                    plan_depth=int(res.get("plan_depth") or 0),
                )

                z = metrics.current_z()
                lb = _latent_lower_bound(z, thresholds)
                regime_met_now = _regime_met(z, thresholds)
                if regime_met_now and regime_reached_at is None:
                    regime_reached_at = ep_idx
                # Stop only when the *current* rolling window meets regime thresholds.
                # This enforces "stable window" semantics (no early hit, later drift).
                if bool(args.stop_on_regime) and metrics.steps >= int(args.regime_window) and regime_met_now:
                    stop_after_batch = True

                if (ep_idx % int(args.log_every)) == 0:
                    promoted_ids = [sid for sid, s in bank.schemas.items() if s.promoted]
                    promoted_ctx_counts = [len(schema_contexts.get(sid, set())) for sid in promoted_ids]
                    z7_cross_context = float(sum(promoted_ctx_counts) / max(1, len(promoted_ctx_counts)))
                    promoted_op_ids = [oid for oid, op in op_bank.operators.items() if op.promoted]
                    promoted_op_ctx_counts = [len(operator_contexts.get(oid, set())) for oid in promoted_op_ids]
                    z7_op_cross_context = float(sum(promoted_op_ctx_counts) / max(1, len(promoted_op_ctx_counts)))
                    _append_jsonl(
                        paths["progress"],
                        {
                            "step": ep_idx,
                            "env_id": str(res.get("env_id")),
                            "seed": int(res.get("seed", 0)),
                            "bank": {**bank.stats(), "schemas_cross_context_avg": z7_cross_context},
                            "operators": {**op_bank.stats(), "operators_cross_context_avg": z7_op_cross_context},
                            "z": z,
                            "latent_lower_bound": lb,
                            "regime_met": bool(regime_met_now),
                            "episode_skipped": False,
                            "episode_success": bool(episode_success),
                            "outcome": str(res.get("outcome") or ""),
                            "reason": str(res.get("reason") or ""),
                            "omega": {
                                "enabled": bool(omega_enabled),
                                "state_sha": str(getattr(omega_state, "state_sha", "") or ""),
                                "prev_state_sha": str(getattr(omega_state, "prev_state_sha", "") or ""),
                                "strict_burns_total": int(getattr(omega_state, "strict_burns_total", 0) or 0)
                                if omega_state is not None
                                else 0,
                                "banned_env_ids_total": int(len(omega_banned_env_ids)),
                                "banned_cluster_ctx_keys_total": int(len(omega_banned_cluster_ctx_keys)),
                                "reachable_cluster_ctx_keys_total": int(len(omega_model_cluster_ctx_keys)),
                            },
                        },
                    )

                _append_jsonl(
                    paths["episodes"],
                    {
                        "step": ep_idx,
                        "env_id": str(res.get("env_id")),
                        "seed": int(res.get("seed", 0)),
                        "outcome": res.get("outcome"),
                        "reason": res.get("reason"),
                        "exception": {
                            "type": res.get("exception_type"),
                            "msg": res.get("exception_msg"),
                        },
                        "goal": res.get("goal"),
                        "did_fail_closed": bool(res.get("did_fail_closed", False)),
                        "executed_with_plan": bool(res.get("executed_with_plan", False)),
                        "terminated": bool(res.get("terminated", False)),
                        "truncated": bool(res.get("truncated", False)),
                        "episode_success": bool(res.get("episode_success", False)),
                        "steps_taken": int(res.get("steps_taken") or 0),
                        "plan_depth": int(res.get("plan_depth") or 0),
                        "concept_calls": int(res.get("concept_calls") or 0),
                        "schema_calls": res.get("schema_calls") or [],
                        "operator_calls": res.get("operator_calls") or [],
                        "plan_fail": {
                            "reason": res.get("plan_fail_reason"),
                            "concept": res.get("plan_fail_concept"),
                            "sim": res.get("plan_fail_sim"),
                        },
                        "cluster_ctx_key": res.get("cluster_ctx_key"),
                        "world_features": res.get("world_features"),
                        "schema": {
                            "selected": res.get("schema_selected"),
                            "selected_id": res.get("schema_selected_id"),
                            "missing": bool(res.get("schema_missing", False)),
                            "suggestion": res.get("schema_suggestion"),
                            "id": res.get("schema_used_id"),
                            "name": res.get("schema_used_name"),
                        },
                        "shallow_suppressed": bool(res.get("shallow_suppressed", False)),
                        "facts": {
                            "pre_search_schema": bool(res.get("pre_search_schema", False)),
                            "object_centric_ok": bool(res.get("object_centric_ok", False)),
                            "conditional_ok": bool(res.get("conditional_ok", False)),
                            "hierarchical_ok": bool(res.get("hierarchical_ok", False)),
                            "failure_to_schema": bool(failure_to_schema_event),
                            "failure_to_operator": bool(failure_to_operator_event),
                            "failure_to_structure": bool(failure_to_structure_event),
                        },
                    },
                )

                if dead_end is not None:
                    break

            # GC only after the merge of the full batch (required).
            if results:
                gc_step = int(max(int(r["episode_idx"]) for r in results))
                removed = bank.gc(gc_step, protected_schema_ids=failure_clusters.protected_schema_ids())
                if removed:
                    _append_jsonl(paths["schema_events"], {"step": gc_step, "event": "gc", "removed_schema_ids": removed})
                removed_ops = op_bank.gc(gc_step, protected_operator_ids=failure_clusters.protected_operator_ids())
                if removed_ops:
                    _append_jsonl(
                        paths["operator_events"], {"step": gc_step, "event": "gc", "removed_operator_ids": removed_ops}
                    )

            # Ω canônico (MAXWELL_Ω strict burn):
            # If this batch had >=1 failure and produced no new promoted schema, destroy one reachable future subspace.
            if omega_enabled and dead_end is None:
                bs = batch_last_processed_step
                if bs is not None and int(bs) >= int(args.omega_bootstrap_steps):
                    if int(batch_failures_seen) > 0 and len(batch_promoted_delta) == 0:
                        candidates: List[FailureCluster] = []
                        for c in failure_clusters.clusters.values():
                            if c.dead_end or c.resolved:
                                continue
                            reason = str(c.signature.get("reason") or "")
                            if reason.startswith("EXCEPTION") or reason.startswith("WORKER_EXCEPTION"):
                                continue
                            ck = str(c.signature.get("cluster_ctx_key") or "")
                            if (not ck) or (ck in omega_banned_cluster_ctx_keys):
                                continue
                            candidates.append(c)

                        if not candidates:
                            dead_end = {
                                "step": int(bs),
                                "reason": "OMEGA_DEAD_NO_BURN_TARGET",
                                "omega": {
                                    "batch_failures_seen": int(batch_failures_seen),
                                    "batch_promoted_delta": list(dict.fromkeys(batch_promoted_delta)),
                                    "banned_env_ids_total": int(len(omega_banned_env_ids)),
                                    "banned_cluster_ctx_keys_total": int(len(omega_banned_cluster_ctx_keys)),
                                },
                            }
                            stop_after_batch = True
                        else:
                            # Deterministic burn choice: highest count, then earliest, then least recent progress, then lexical.
                            candidates.sort(
                                key=lambda c: (
                                    -int(c.count),
                                    int(c.first_step),
                                    int(c.last_progress_step),
                                    str(c.cluster_id),
                                )
                            )
                            burn = candidates[0]
                            burn.dead_end = True
                            burn_ck = str(burn.signature.get("cluster_ctx_key") or "")
                            omega_banned_cluster_ctx_keys.add(burn_ck)
                            omega_state = make_omega_state(
                                omega_state,
                                banned_env_ids=omega_banned_env_ids,
                                banned_cluster_ctx_keys=omega_banned_cluster_ctx_keys,
                                strict_burns_added=1,
                            )
                            if omega_first_burn_step is None:
                                omega_first_burn_step = int(bs)
                            _append_jsonl(
                                paths["omega_events"],
                                {
                                    "step": int(bs),
                                    "event": "omega_strict_burn",
                                    "burn_cluster_id": str(burn.cluster_id),
                                    "burn_cluster_ctx_key": str(burn_ck),
                                    "batch_failures_seen": int(batch_failures_seen),
                                    "batch_promoted_delta": list(dict.fromkeys(batch_promoted_delta)),
                                    "banned_env_ids_total": int(len(omega_banned_env_ids)),
                                    "banned_cluster_ctx_keys_total": int(len(omega_banned_cluster_ctx_keys)),
                                    "reachable_cluster_ctx_keys_total": int(len(omega_model_cluster_ctx_keys)),
                                    "omega_state_sha": str(getattr(omega_state, "state_sha", "") or ""),
                                    "omega_prev_state_sha": str(getattr(omega_state, "prev_state_sha", "") or ""),
                                },
                            )

                # Ω death condition (finite reachable cluster_ctx_key space).
                if dead_end is None and omega_model_cluster_ctx_keys:
                    remaining = omega_model_cluster_ctx_keys - omega_banned_cluster_ctx_keys
                    if not remaining:
                        dead_end = {
                            "step": int(bs) if bs is not None else 0,
                            "reason": "OMEGA_DEAD_NO_FUTURE_LEFT",
                            "omega": {
                                "banned_cluster_ctx_keys_total": int(len(omega_banned_cluster_ctx_keys)),
                                "reachable_cluster_ctx_keys_total": int(len(omega_model_cluster_ctx_keys)),
                            },
                        }
                        stop_after_batch = True

            if stop_after_batch:
                break
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    # Persist final schema bank snapshot (WORM inside this run dir).
    _write_json(paths["schema_bank"], {"schemas": [s.to_json() for s in bank.schemas.values()], "stats": bank.stats()})
    _write_json(
        paths["operator_bank"], {"operators": [op.to_json() for op in op_bank.operators.values()], "stats": op_bank.stats()}
    )
    _write_json(
        paths["failure_clusters"],
        {
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "count": int(c.count),
                    "first_step": int(c.first_step),
                    "last_step": int(c.last_step),
                    "last_progress_step": int(c.last_progress_step),
                    "induce_attempts": int(c.induce_attempts),
                    "last_induce_step": int(c.last_induce_step),
                    "induced_schema_ids": list(dict.fromkeys(c.induced_schema_ids)),
                    "induced_operator_ids": list(dict.fromkeys(c.induced_operator_ids)),
                    "witnesses": _sorted_unique_ablation_witnesses(c.witnesses),
                    "resolved": bool(c.resolved),
                    "dead_end": bool(c.dead_end),
                    "signature": dict(c.signature),
                }
                for c in sorted(failure_clusters.clusters.values(), key=lambda c: (-int(c.count), str(c.cluster_id)))
            ],
        },
    )

    if omega_enabled:
        omega_state = make_omega_state(
            omega_state,
            banned_env_ids=omega_banned_env_ids,
            banned_cluster_ctx_keys=omega_banned_cluster_ctx_keys,
            strict_burns_added=0,
        )
        _write_json(paths["omega_state"], omega_state.to_json())

    summary = {
        "schema_version": 1,
        "kind": "structural_training_summary_v3",
        "run_dir": out_dir,
        "config_path": paths["config"],
        "episodes_path": paths["episodes"],
        "progress_path": paths["progress"],
        "schema_bank_path": paths["schema_bank"],
        "schema_events_path": paths["schema_events"],
        "operator_bank_path": paths["operator_bank"],
        "operator_events_path": paths["operator_events"],
        "omega_state_path": paths["omega_state"] if omega_enabled else None,
        "omega_events_path": paths["omega_events"] if omega_enabled else None,
        "omega_state_in": omega_prev_in if omega_enabled else None,
        "omega": {
            "enabled": bool(omega_enabled),
            "state_sha": str(getattr(omega_state, "state_sha", "") or "") if omega_state is not None else "",
            "prev_state_sha": str(getattr(omega_state, "prev_state_sha", "") or "") if omega_state is not None else "",
            "strict_burns_total": int(getattr(omega_state, "strict_burns_total", 0) or 0) if omega_state is not None else 0,
            "banned_env_ids_total": int(len(omega_banned_env_ids)),
            "banned_cluster_ctx_keys_total": int(len(omega_banned_cluster_ctx_keys)),
            "reachable_cluster_ctx_keys_total": int(len(omega_model_cluster_ctx_keys)),
            "remaining_cluster_ctx_keys_total": int(len(omega_model_cluster_ctx_keys - omega_banned_cluster_ctx_keys))
            if omega_model_cluster_ctx_keys
            else None,
        },
        "regime_thresholds": thresholds,
        "regime_reached_at_step": regime_reached_at,
        "cumulative": metrics.cumulative(),
        "z_window": metrics.current_z(),
        "dead_end": dead_end,
        "Z7_schemas_cross_context_avg": float(
            sum(len(schema_contexts.get(sid, set())) for sid, s in bank.schemas.items() if s.promoted)
            / max(1, sum(1 for s in bank.schemas.values() if s.promoted))
        ),
        "Z7_operators_cross_context_avg": float(
            sum(len(operator_contexts.get(oid, set())) for oid, op in op_bank.operators.items() if op.promoted)
            / max(1, sum(1 for op in op_bank.operators.values() if op.promoted))
        ),
        "bank": bank.stats(),
        "operators": op_bank.stats(),
    }
    _write_json(paths["summary"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
