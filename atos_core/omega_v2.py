from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from atos_core.act import canonical_json_dumps, sha256_hex


OMEGA_SCHEMA_VERSION_V2 = 2


def _stable_json(obj: Any) -> str:
    return canonical_json_dumps(obj)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_x(path: Path, obj: Any) -> None:
    if path.exists():
        raise FileExistsError(f"worm_exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _load_concept_inductions_by_task(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load per-task induced concept births produced by:
      scripts/arc_induce_concept_templates_v146.py --induction_log

    This is the explicit "failure -> concept object birth" trace that Ω needs for:
      FAIL + Δ(PromotedConceptSet)==∅ => destroy future
    while keeping ARC solver zero-touch.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in _iter_jsonl(path):
        if str(row.get("kind") or "") != "arc_concept_induction_v146":
            continue
        task_id = str(row.get("task_id") or "")
        cid = str(row.get("concept_id") or "")
        sig = row.get("signature") if isinstance(row.get("signature"), dict) else None
        op_ids = row.get("op_ids") if isinstance(row.get("op_ids"), list) else None
        if not task_id or not cid or sig is None or not op_ids:
            continue
        op_ids_s = [str(x) for x in op_ids if str(x)]
        if not op_ids_s:
            continue
        rec = {
            "task_id": str(task_id),
            "concept_id": str(cid),
            "signature": {str(k): sig.get(k) for k in sorted(sig.keys())},
            "op_ids": list(op_ids_s),
            "cost_bits": int(row.get("cost_bits") or 10),
            "rank": int(row.get("rank") or 0),
            "source": str(row.get("source") or ""),
        }
        out.setdefault(task_id, []).append(rec)
    for tid, rows in list(out.items()):
        rows.sort(key=lambda r: (int(r.get("rank") or 0), str(r.get("concept_id") or ""), _stable_json(r)))
        out[tid] = rows
    return out


def _grid_shape(grid: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(grid, list) or not grid:
        return None
    if not isinstance(grid[0], list):
        return None
    h = int(len(grid))
    w = int(len(grid[0]))
    return (h, w)


def _grid_palette(grid: Any) -> List[int]:
    if not isinstance(grid, list):
        return []
    cols: set[int] = set()
    for row in grid:
        if not isinstance(row, list):
            continue
        for x in row:
            try:
                cols.add(int(x))
            except Exception:
                continue
    return sorted(int(c) for c in cols)


def arc_task_family_id(task: Dict[str, Any]) -> str:
    """
    Deterministic, non-task-id family id for ARC tasks.

    This is intentionally coarse: Ω bans are global and should remove regions of future,
    not individual task ids.
    """
    train_pairs = task.get("train_pairs") if isinstance(task.get("train_pairs"), list) else []
    in_shapes: List[Tuple[int, int]] = []
    out_shapes: List[Tuple[int, int]] = []
    pal_in: set[int] = set()
    pal_out: set[int] = set()
    for row in train_pairs:
        if not isinstance(row, dict):
            continue
        ig = row.get("in_grid")
        og = row.get("out_grid")
        sh_i = _grid_shape(ig)
        sh_o = _grid_shape(og)
        if sh_i is not None:
            in_shapes.append(tuple(int(x) for x in sh_i))
        if sh_o is not None:
            out_shapes.append(tuple(int(x) for x in sh_o))
        for c in _grid_palette(ig):
            pal_in.add(int(c))
        for c in _grid_palette(og):
            pal_out.add(int(c))
    def _bucket_dim(x: int) -> str:
        v = int(x)
        if v <= 5:
            return "S"
        if v <= 10:
            return "M"
        if v <= 15:
            return "L"
        return "XL"

    def _bucket_palette(n: int) -> str:
        v = int(n)
        if v <= 2:
            return "P2"
        if v <= 4:
            return "P4"
        if v <= 6:
            return "P6"
        if v <= 8:
            return "P8"
        return "P9PLUS"

    def _bucket_train_pairs(n: int) -> str:
        v = int(n)
        if v <= 1:
            return "T1"
        if v == 2:
            return "T2"
        if v == 3:
            return "T3"
        return "T4PLUS"

    # Use max dims across train pairs as a coarse family proxy.
    max_in_h = max([h for h, _w in in_shapes], default=0)
    max_in_w = max([w for _h, w in in_shapes], default=0)
    max_out_h = max([h for h, _w in out_shapes], default=0)
    max_out_w = max([w for _h, w in out_shapes], default=0)

    sig = {
        "n_train_bucket": _bucket_train_pairs(len(train_pairs)),
        "in_hw_bucket": [_bucket_dim(max_in_h), _bucket_dim(max_in_w)],
        "out_hw_bucket": [_bucket_dim(max_out_h), _bucket_dim(max_out_w)],
        "shape_change": bool((max_in_h, max_in_w) != (max_out_h, max_out_w)),
        "pal_in_bucket": _bucket_palette(len(pal_in)),
        "pal_out_bucket": _bucket_palette(len(pal_out)),
    }
    return "fam_" + sha256_hex(_stable_json(sig).encode("utf-8"))


def arc_task_context_id(task: Dict[str, Any]) -> str:
    """
    Deterministic context id for transfer tests: distinct contexts must differ structurally.
    """
    train_pairs = task.get("train_pairs") if isinstance(task.get("train_pairs"), list) else []
    # Use first train pair shapes + palette sizes as a stable proxy.
    sh_in: Optional[Tuple[int, int]] = None
    sh_out: Optional[Tuple[int, int]] = None
    pal_in: set[int] = set()
    pal_out: set[int] = set()
    if train_pairs and isinstance(train_pairs[0], dict):
        sh_in = _grid_shape(train_pairs[0].get("in_grid"))
        sh_out = _grid_shape(train_pairs[0].get("out_grid"))
    for row in train_pairs:
        if not isinstance(row, dict):
            continue
        for c in _grid_palette(row.get("in_grid")):
            pal_in.add(int(c))
        for c in _grid_palette(row.get("out_grid")):
            pal_out.add(int(c))
    sig = {
        "n_train": int(len(train_pairs)),
        "shape_in": [int(sh_in[0]), int(sh_in[1])] if sh_in is not None else None,
        "shape_out": [int(sh_out[0]), int(sh_out[1])] if sh_out is not None else None,
        "pal_in_n": int(len(pal_in)),
        "pal_out_n": int(len(pal_out)),
    }
    return "ctx_" + sha256_hex(_stable_json(sig).encode("utf-8"))


def residual_signature_key(sig: Dict[str, Any]) -> str:
    """
    Stable key for residual signature bans (region-of-future / transformation-class ban).
    """
    if not isinstance(sig, dict):
        return ""
    body = {str(k): sig.get(k) for k in sorted(sig.keys())}
    return "rsig_" + sha256_hex(_stable_json(body).encode("utf-8"))


@dataclass(frozen=True)
class OmegaParamsV2:
    # Cluster-to-induction policy (ARC-side Ω).
    cluster_k: int = 3
    cluster_cooldown_runs: int = 1
    cluster_max_induce_attempts: int = 3

    # Optional compute caps (Ω may also shrink budgets, but must not be only this).
    min_max_programs: int = 200
    burn_programs_per_dead_cluster: int = 50
    min_max_depth: int = 2
    burn_depth_every_n_dead_clusters: int = 10

    # Promotion validator thresholds (deep semantics gate).
    promote_min_support: int = 2
    promote_min_used_solved: int = 2
    promote_min_contexts: int = 2

    # Shallow-path suppression (Ω existential pressure).
    #
    # After the given run index (1-based), a solved episode is treated as a FAIL for Ω
    # unless it uses at least one concept_call. This makes “survive by solving shallowly”
    # impossible once bootstrapping ends.
    #
    # Separately, once at least one concept is already promoted, you can require that
    # a solved episode uses a previously-promoted concept_call (stronger).
    require_concept_call_after_runs: int = 0
    require_promoted_concept_call_after_runs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_k": int(self.cluster_k),
            "cluster_cooldown_runs": int(self.cluster_cooldown_runs),
            "cluster_max_induce_attempts": int(self.cluster_max_induce_attempts),
            "min_max_programs": int(self.min_max_programs),
            "burn_programs_per_dead_cluster": int(self.burn_programs_per_dead_cluster),
            "min_max_depth": int(self.min_max_depth),
            "burn_depth_every_n_dead_clusters": int(self.burn_depth_every_n_dead_clusters),
            "promote_min_support": int(self.promote_min_support),
            "promote_min_used_solved": int(self.promote_min_used_solved),
            "promote_min_contexts": int(self.promote_min_contexts),
            "require_concept_call_after_runs": int(self.require_concept_call_after_runs),
            "require_promoted_concept_call_after_runs": int(self.require_promoted_concept_call_after_runs),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OmegaParamsV2":
        return OmegaParamsV2(
            cluster_k=int(d.get("cluster_k", 3)),
            cluster_cooldown_runs=int(d.get("cluster_cooldown_runs", 1)),
            cluster_max_induce_attempts=int(d.get("cluster_max_induce_attempts", 3)),
            min_max_programs=int(d.get("min_max_programs", 200)),
            burn_programs_per_dead_cluster=int(d.get("burn_programs_per_dead_cluster", 50)),
            min_max_depth=int(d.get("min_max_depth", 2)),
            burn_depth_every_n_dead_clusters=int(d.get("burn_depth_every_n_dead_clusters", 10)),
            promote_min_support=int(d.get("promote_min_support", 2)),
            promote_min_used_solved=int(d.get("promote_min_used_solved", 2)),
            promote_min_contexts=int(d.get("promote_min_contexts", 2)),
            require_concept_call_after_runs=int(d.get("require_concept_call_after_runs", 0)),
            require_promoted_concept_call_after_runs=int(d.get("require_promoted_concept_call_after_runs", 0)),
        )


def _concept_to_json(concept: Dict[str, Any]) -> Dict[str, Any]:
    # Normalize dict fields deterministically.
    out: Dict[str, Any] = {
        "concept_id": str(concept.get("concept_id") or ""),
        "signature": concept.get("signature") if isinstance(concept.get("signature"), dict) else {},
        "op_ids": [str(x) for x in (concept.get("op_ids") if isinstance(concept.get("op_ids"), list) else []) if str(x)],
        "support": int(concept.get("support") or 0),
        "cost_bits": int(concept.get("cost_bits") or 10),
        "state": str(concept.get("state") or "candidate"),
        "created_run": int(concept.get("created_run") or 0),
        "last_run": int(concept.get("last_run") or 0),
        "used_solved": int(concept.get("used_solved") or 0),
        "contexts_used": sorted({str(x) for x in (concept.get("contexts_used") or []) if str(x)}),
        "families_used": sorted({str(x) for x in (concept.get("families_used") or []) if str(x)}),
        "origin_clusters": sorted({str(x) for x in (concept.get("origin_clusters") or []) if str(x)}),
        "emitted_from_failure": int(concept.get("emitted_from_failure") or 0),
    }
    return {str(k): out[k] for k in sorted(out.keys())}


def _operator_to_json(op: Dict[str, Any]) -> Dict[str, Any]:
    """
    "Operator" = reusable closure (macro_call) tracked by Ω.

    This is the minimal live-object schema needed for:
      - lifecycle (candidate -> refining -> promoted)
      - cross-context usage
      - contrafactual ablation
      - coupling to MAXWELL_Ω (promoted_delta_total)
    """
    out: Dict[str, Any] = {
        "operator_id": str(op.get("operator_id") or ""),
        "op_ids": [str(x) for x in (op.get("op_ids") if isinstance(op.get("op_ids"), list) else []) if str(x)],
        "support": int(op.get("support") or 0),
        "state": str(op.get("state") or "candidate"),
        "created_run": int(op.get("created_run") or 0),
        "last_run": int(op.get("last_run") or 0),
        "used_solved": int(op.get("used_solved") or 0),
        "contexts_used": sorted({str(x) for x in (op.get("contexts_used") or []) if str(x)}),
        "families_used": sorted({str(x) for x in (op.get("families_used") or []) if str(x)}),
        # Optional: operators can be linked to failure clusters by external miners.
        "origin_clusters": sorted({str(x) for x in (op.get("origin_clusters") or []) if str(x)}),
        "emitted_from_failure": int(op.get("emitted_from_failure") or 0),
    }
    return {str(k): out[k] for k in sorted(out.keys())}


def _cluster_to_json(cluster: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "cluster_key": str(cluster.get("cluster_key") or ""),
        "count": int(cluster.get("count") or 0),
        "first_run": int(cluster.get("first_run") or 0),
        "last_run": int(cluster.get("last_run") or 0),
        "last_progress_run": int(cluster.get("last_progress_run") or -1),
        "induce_attempts": int(cluster.get("induce_attempts") or 0),
        "last_induce_run": int(cluster.get("last_induce_run") or -1),
        "resolved": bool(cluster.get("resolved") or False),
        "dead_end": bool(cluster.get("dead_end") or False),
        "families_seen": sorted({str(x) for x in (cluster.get("families_seen") or []) if str(x)}),
    }
    return {str(k): out[k] for k in sorted(out.keys())}


def _family_baseline_to_json(row: Dict[str, Any]) -> Dict[str, Any]:
    best_cost = row.get("best_cost_bits_without_concepts")
    best_depth = row.get("best_depth_without_concepts")
    best_cost_m = row.get("best_cost_bits_without_macros")
    best_depth_m = row.get("best_depth_without_macros")
    out: Dict[str, Any] = {
        "family_id": str(row.get("family_id") or ""),
        "first_run": int(row.get("first_run") or 0),
        "last_run": int(row.get("last_run") or 0),
        "without_concept_attempts": int(row.get("without_concept_attempts") or 0),
        "without_concept_successes": int(row.get("without_concept_successes") or 0),
        "best_cost_bits_without_concepts": int(best_cost) if isinstance(best_cost, int) else None,
        "best_depth_without_concepts": int(best_depth) if isinstance(best_depth, int) else None,
        "without_macro_attempts": int(row.get("without_macro_attempts") or 0),
        "without_macro_successes": int(row.get("without_macro_successes") or 0),
        "best_cost_bits_without_macros": int(best_cost_m) if isinstance(best_cost_m, int) else None,
        "best_depth_without_macros": int(best_depth_m) if isinstance(best_depth_m, int) else None,
    }
    return {str(k): out[k] for k in sorted(out.keys())}


@dataclass(frozen=True)
class OmegaStateV2:
    schema_version: int
    kind: str

    base_max_depth: int
    base_max_programs: int
    params: OmegaParamsV2

    runs_total: int
    dead_clusters_total: int
    strict_burns_total: int
    max_depth_cap: int
    max_programs_cap: int

    banned_task_families: Tuple[str, ...]
    banned_residual_signatures: Tuple[str, ...]

    concepts: Tuple[Dict[str, Any], ...]
    failure_clusters: Tuple[Dict[str, Any], ...]
    operators: Tuple[Dict[str, Any], ...] = ()
    family_baselines: Tuple[Dict[str, Any], ...] = ()

    prev_state_sha: str = ""
    state_sha: str = ""

    def to_dict(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "schema_version": int(self.schema_version),
            "kind": str(self.kind),
            "base_max_depth": int(self.base_max_depth),
            "base_max_programs": int(self.base_max_programs),
            "params": self.params.to_dict(),
            "runs_total": int(self.runs_total),
            "dead_clusters_total": int(self.dead_clusters_total),
            "strict_burns_total": int(self.strict_burns_total),
            "max_depth_cap": int(self.max_depth_cap),
            "max_programs_cap": int(self.max_programs_cap),
            "banned_task_families": sorted({str(x) for x in self.banned_task_families if str(x)}),
            "banned_residual_signatures": sorted({str(x) for x in self.banned_residual_signatures if str(x)}),
            "concepts": [
                _concept_to_json(c) for c in sorted(list(self.concepts), key=lambda c: (str(c.get("concept_id") or ""), _stable_json(c)))
            ],
            "operators": [
                _operator_to_json(o)
                for o in sorted(list(self.operators or ()), key=lambda o: (str(o.get("operator_id") or ""), _stable_json(o)))
                if isinstance(o, dict)
            ],
            "failure_clusters": [
                _cluster_to_json(c)
                for c in sorted(list(self.failure_clusters), key=lambda c: (str(c.get("cluster_key") or ""), _stable_json(c)))
            ],
            "family_baselines": [
                _family_baseline_to_json(r)
                for r in sorted(list(self.family_baselines or ()), key=lambda r: (str(r.get("family_id") or ""), _stable_json(r)))
                if isinstance(r, dict)
            ],
            "prev_state_sha": str(self.prev_state_sha or ""),
        }
        body["state_sha"] = str(self.state_sha or "")
        return {str(k): body[k] for k in sorted(body.keys())}

    def content_hash(self) -> str:
        body = self.to_dict()
        body["state_sha"] = ""
        return sha256_hex(_stable_json(body).encode("utf-8"))

    @staticmethod
    def from_path(path: Path) -> "OmegaStateV2":
        raw = _read_json(path)
        if not isinstance(raw, dict):
            raise ValueError(f"bad_omega_state_v2:{path}")
        params_raw = raw.get("params") if isinstance(raw.get("params"), dict) else {}
        st = OmegaStateV2(
            schema_version=int(raw.get("schema_version", OMEGA_SCHEMA_VERSION_V2)),
            kind=str(raw.get("kind") or "omega_state_v2"),
            base_max_depth=int(raw.get("base_max_depth", 4)),
            base_max_programs=int(raw.get("base_max_programs", 4000)),
            params=OmegaParamsV2.from_dict(params_raw),
            runs_total=int(raw.get("runs_total", 0)),
            dead_clusters_total=int(raw.get("dead_clusters_total", 0)),
            strict_burns_total=int(raw.get("strict_burns_total", 0)),
            max_depth_cap=int(raw.get("max_depth_cap", int(raw.get("base_max_depth", 4)))),
            max_programs_cap=int(raw.get("max_programs_cap", int(raw.get("base_max_programs", 4000)))),
            banned_task_families=tuple(str(x) for x in (raw.get("banned_task_families") or []) if str(x)),
            banned_residual_signatures=tuple(str(x) for x in (raw.get("banned_residual_signatures") or []) if str(x)),
            concepts=tuple(x for x in (raw.get("concepts") or []) if isinstance(x, dict)),
            operators=tuple(x for x in (raw.get("operators") or []) if isinstance(x, dict)),
            failure_clusters=tuple(x for x in (raw.get("failure_clusters") or []) if isinstance(x, dict)),
            family_baselines=tuple(x for x in (raw.get("family_baselines") or []) if isinstance(x, dict)),
            prev_state_sha=str(raw.get("prev_state_sha") or ""),
            state_sha=str(raw.get("state_sha") or ""),
        )
        want = str(st.state_sha or "")
        got = st.content_hash()
        if want and want != got:
            raise ValueError(f"omega_state_v2_hash_mismatch:want={want},got={got}")
        return st

    def write_worm(self, path: Path) -> None:
        d = self.to_dict()
        if not d.get("state_sha"):
            d["state_sha"] = self.content_hash()
        _write_json_x(path, d)


def apply_omega_caps(
    *,
    want_max_depth: int,
    want_max_programs: int,
    state: Optional[OmegaStateV2],
) -> Tuple[int, int]:
    if state is None:
        return int(want_max_depth), int(want_max_programs)
    return min(int(want_max_depth), int(state.max_depth_cap)), min(int(want_max_programs), int(state.max_programs_cap))


def _concept_id_from_template_row(row: Dict[str, Any]) -> str:
    cid = str(row.get("concept_id") or "")
    return cid


def update_omega_state_v2(
    *,
    prev: Optional[OmegaStateV2],
    events_path: Path,
    concept_bank_path: Optional[Path],
    macro_bank_path: Optional[Path] = None,
    induction_log_path: Optional[Path] = None,
    base_max_depth: int,
    base_max_programs: int,
    params: Optional[OmegaParamsV2] = None,
) -> Tuple[OmegaStateV2, Dict[str, Any]]:
    params_eff = params or (prev.params if prev is not None else OmegaParamsV2())
    run_idx = int(prev.runs_total + 1) if prev is not None else 1

    base_d = int(prev.base_max_depth) if prev is not None else int(base_max_depth)
    base_p = int(prev.base_max_programs) if prev is not None else int(base_max_programs)

    # Load prev registries as dicts (mutable for update).
    concepts_by_id: Dict[str, Dict[str, Any]] = {}
    if prev is not None:
        for c in prev.concepts:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("concept_id") or "")
            if cid:
                concepts_by_id[cid] = _concept_to_json(c)

    operators_by_id: Dict[str, Dict[str, Any]] = {}
    if prev is not None:
        for o in prev.operators or ():
            if not isinstance(o, dict):
                continue
            oid = str(o.get("operator_id") or "")
            if oid:
                operators_by_id[oid] = _operator_to_json(o)

    clusters_by_key: Dict[str, Dict[str, Any]] = {}
    if prev is not None:
        for c in prev.failure_clusters:
            if not isinstance(c, dict):
                continue
            ck = str(c.get("cluster_key") or "")
            if ck:
                clusters_by_key[ck] = _cluster_to_json(c)

    banned_fams: set[str] = set(prev.banned_task_families) if prev is not None else set()
    banned_rsig: set[str] = set(prev.banned_residual_signatures) if prev is not None else set()

    baselines_by_family: Dict[str, Dict[str, Any]] = {}
    if prev is not None:
        for row in prev.family_baselines or ():
            if not isinstance(row, dict):
                continue
            fid = str(row.get("family_id") or "")
            if fid:
                baselines_by_family[fid] = _family_baseline_to_json(row)

    # Update concept metadata from bank (support/cost/signature/op_ids).
    if concept_bank_path is not None and concept_bank_path.is_file():
        for row in _iter_jsonl(concept_bank_path):
            cid = _concept_id_from_template_row(row)
            if not cid:
                continue
            cur = concepts_by_id.get(cid)
            if cur is None:
                cur = {
                    "concept_id": cid,
                    "signature": row.get("signature") if isinstance(row.get("signature"), dict) else {},
                    "op_ids": [str(x) for x in (row.get("op_ids") if isinstance(row.get("op_ids"), list) else []) if str(x)],
                    "support": int(row.get("support") or 0),
                    "cost_bits": int(row.get("cost_bits") or 10),
                    "state": "candidate",
                    "created_run": int(run_idx),
                    "last_run": int(run_idx),
                    "used_solved": 0,
                    "contexts_used": [],
                    "families_used": [],
                    "origin_clusters": [],
                    "emitted_from_failure": 0,
                }
            else:
                cur = dict(cur)
                cur["signature"] = row.get("signature") if isinstance(row.get("signature"), dict) else cur.get("signature", {})
                cur["op_ids"] = [str(x) for x in (row.get("op_ids") if isinstance(row.get("op_ids"), list) else []) if str(x)]
                cur["support"] = int(row.get("support") or 0)
                cur["cost_bits"] = int(row.get("cost_bits") or cur.get("cost_bits") or 10)
                cur["last_run"] = int(run_idx)
            concepts_by_id[cid] = _concept_to_json(cur)

    macro_rows_for_solver: List[Dict[str, Any]] = []
    # Update operator metadata from macro/operator bank (support/op_ids).
    if macro_bank_path is not None and macro_bank_path.is_file():
        for row in _iter_jsonl(macro_bank_path):
            if not isinstance(row, dict):
                continue
            kind = str(row.get("kind") or "")
            if kind not in {"arc_macro_template_v143", "arc_operator_template_v147"}:
                continue
            macro_rows_for_solver.append(dict(row))
            oid = str(row.get("macro_id") or "") if kind == "arc_macro_template_v143" else str(row.get("operator_id") or "")
            if not oid:
                continue
            op_ids_raw = row.get("op_ids")
            if not isinstance(op_ids_raw, list) or not op_ids_raw:
                continue
            op_ids = [str(x) for x in op_ids_raw if str(x)]
            if not op_ids:
                continue
            support = int(row.get("support") or 0)
            cur = operators_by_id.get(oid)
            if cur is None:
                cur = {
                    "operator_id": str(oid),
                    "op_ids": list(op_ids),
                    "support": int(support),
                    "state": "candidate",
                    "created_run": int(run_idx),
                    "last_run": int(run_idx),
                    "used_solved": 0,
                    "contexts_used": [],
                    "families_used": [],
                    "origin_clusters": [str(x) for x in (row.get("origin_clusters") or []) if str(x)] if isinstance(row.get("origin_clusters"), list) else [],
                    "emitted_from_failure": int(row.get("emitted_from_failure") or 0),
                }
            else:
                cur = dict(cur)
                cur["op_ids"] = list(op_ids)
                cur["support"] = int(support)
                cur["last_run"] = int(run_idx)
                if isinstance(row.get("origin_clusters"), list):
                    oc = set(str(x) for x in (cur.get("origin_clusters") or []) if str(x))
                    oc |= {str(x) for x in (row.get("origin_clusters") or []) if str(x)}
                    cur["origin_clusters"] = sorted(oc)
                if "emitted_from_failure" in row:
                    cur["emitted_from_failure"] = int(cur.get("emitted_from_failure") or 0) + int(row.get("emitted_from_failure") or 0)
            operators_by_id[oid] = _operator_to_json(cur)
    macro_rows_for_solver.sort(key=lambda r: (str(r.get("macro_id") or r.get("operator_id") or ""), _stable_json(r)))

    induced_by_task: Dict[str, List[Dict[str, Any]]] = {}
    if induction_log_path is not None and induction_log_path.is_file():
        induced_by_task = _load_concept_inductions_by_task(induction_log_path)

    run_dir = events_path.parent
    summary_path = run_dir / "summary.json"
    summary_obj: Dict[str, Any] = {}
    if summary_path.is_file():
        try:
            raw_sum = _read_json(summary_path)
            if isinstance(raw_sum, dict):
                summary_obj = raw_sum
        except Exception:
            summary_obj = {}

    # Track promotions in this update.
    promoted_before: set[str] = set()
    if prev is not None:
        for c in prev.concepts:
            if isinstance(c, dict) and str(c.get("state") or "") == "promoted":
                cid = str(c.get("concept_id") or "")
                if cid:
                    promoted_before.add(cid)

    promoted_ops_before: set[str] = set()
    if prev is not None:
        for o in prev.operators or ():
            if isinstance(o, dict) and str(o.get("state") or "") == "promoted":
                oid = str(o.get("operator_id") or "")
                if oid:
                    promoted_ops_before.add(oid)

    # Process events: update clusters + concept evidence.
    failures_seen = 0
    successes_seen = 0
    raw_successes_seen = 0
    shallow_suppressed_successes = 0
    shallow_suppressed_by_reason: Dict[str, int] = {}
    # Best observed (cost_bits, depth) usage per concept+family in this run.
    # We intentionally DO NOT require the solution to use only one concept:
    # composition is the target regime; ablation (P5) still prevents placebo promotion.
    best_use: Dict[Tuple[str, str], Tuple[int, int, int, str]] = {}  # (cost_bits, depth, n_concepts, task_id)
    used_task_ids_by_concept: Dict[str, Set[str]] = {}
    best_use_op: Dict[Tuple[str, str], Tuple[int, int, str]] = {}  # (cost_bits, depth, task_id)
    used_task_ids_by_operator: Dict[str, Set[str]] = {}
    for ev in _iter_jsonl(events_path):
        if str(ev.get("kind") or "") != "omega_event_v2":
            continue
        task_obj = ev.get("task") if isinstance(ev.get("task"), dict) else {}
        family_id = str(ev.get("task_family_id") or "") or arc_task_family_id(task_obj)
        context_id = str(ev.get("task_context_id") or "") or arc_task_context_id(task_obj)
        episode_success_raw = bool(ev.get("episode_success") or False)
        task_id = str(ev.get("task_id") or "")
        program_cost_bits = ev.get("program_cost_bits")
        program_depth = ev.get("program_depth")
        cost_bits_i: Optional[int] = int(program_cost_bits) if isinstance(program_cost_bits, int) else None
        depth_i: Optional[int] = int(program_depth) if isinstance(program_depth, int) else None
        induced_rows = induced_by_task.get(task_id) if task_id else None

        cids0 = ev.get("concept_calls_solution") if isinstance(ev.get("concept_calls_solution"), list) else []
        cids0_s = [str(x) for x in cids0 if str(x)]
        mids0 = ev.get("macro_calls_solution") if isinstance(ev.get("macro_calls_solution"), list) else []
        mids0_s = [str(x) for x in mids0 if str(x)]

        # Shallow-path suppression: after bootstrapping, "solved without concepts" is a FAIL for Ω.
        shallow_reason = ""
        episode_success = bool(episode_success_raw)
        if episode_success_raw:
            raw_successes_seen += 1
            # Require at least one concept_call after a given run index.
            if (
                int(params_eff.require_concept_call_after_runs) > 0
                and int(run_idx) >= int(params_eff.require_concept_call_after_runs)
                and not cids0_s
            ):
                episode_success = False
                shallow_reason = "SHALLOW_NO_CONCEPT_CALL"
            # If we already have promoted concepts, optionally require that a solved episode
            # uses at least one previously-promoted concept_call.
            elif (
                int(params_eff.require_promoted_concept_call_after_runs) > 0
                and int(run_idx) >= int(params_eff.require_promoted_concept_call_after_runs)
                and promoted_before
                and not (set(cids0_s) & set(promoted_before))
            ):
                episode_success = False
                shallow_reason = "SHALLOW_NO_PROMOTED_CONCEPT_CALL"

        if shallow_reason:
            shallow_suppressed_successes += 1
            shallow_suppressed_by_reason[shallow_reason] = int(shallow_suppressed_by_reason.get(shallow_reason, 0)) + 1

        if episode_success:
            successes_seen += 1
        else:
            failures_seen += 1

        # NOTE: Concept evidence (used_solved / contexts / families) should follow raw solver success,
        # even if Ω reclassifies a shallow success as a FAIL. Otherwise, bootstrapping becomes impossible.
        episode_solved = bool(episode_success_raw)

        if episode_solved and task_id and cids0_s:
            for cid0 in cids0_s:
                if cid0:
                    used_task_ids_by_concept.setdefault(str(cid0), set()).add(str(task_id))
        if episode_solved and task_id and mids0_s:
            for mid0 in mids0_s:
                if mid0:
                    used_task_ids_by_operator.setdefault(str(mid0), set()).add(str(task_id))
        if episode_solved and family_id and isinstance(cost_bits_i, int) and isinstance(depth_i, int) and cids0_s:
            for cid0 in cids0_s:
                k = (str(cid0), str(family_id))
                cand = (int(cost_bits_i), int(depth_i), int(len(cids0_s)), str(task_id))
                prev_best = best_use.get(k)
                if prev_best is None or cand < prev_best:
                    best_use[k] = cand
        if episode_solved and family_id and isinstance(cost_bits_i, int) and isinstance(depth_i, int) and mids0_s:
            for mid0 in mids0_s:
                k = (str(mid0), str(family_id))
                cand = (int(cost_bits_i), int(depth_i), str(task_id))
                prev_best = best_use_op.get(k)
                if prev_best is None or cand < prev_best:
                    best_use_op[k] = cand

        # Family baseline: no concept-call episodes define the baseline w/out deep semantics.
        if family_id:
            fb0 = baselines_by_family.get(family_id)
            if fb0 is None:
                fb0 = {
                    "family_id": str(family_id),
                    "first_run": int(run_idx),
                    "last_run": int(run_idx),
                    "without_concept_attempts": 0,
                    "without_concept_successes": 0,
                    "best_cost_bits_without_concepts": None,
                    "best_depth_without_concepts": None,
                    "without_macro_attempts": 0,
                    "without_macro_successes": 0,
                    "best_cost_bits_without_macros": None,
                    "best_depth_without_macros": None,
                }
            else:
                fb0 = dict(fb0)
                fb0["last_run"] = int(run_idx)
            if not cids0_s:
                fb0["without_concept_attempts"] = int(fb0.get("without_concept_attempts") or 0) + 1
                if episode_success:
                    fb0["without_concept_successes"] = int(fb0.get("without_concept_successes") or 0) + 1
                    if isinstance(cost_bits_i, int) and isinstance(depth_i, int):
                        best_cost = fb0.get("best_cost_bits_without_concepts")
                        best_depth = fb0.get("best_depth_without_concepts")
                        cand2 = (int(cost_bits_i), int(depth_i))
                        if not isinstance(best_cost, int) or not isinstance(best_depth, int):
                            fb0["best_cost_bits_without_concepts"] = int(cand2[0])
                            fb0["best_depth_without_concepts"] = int(cand2[1])
                        else:
                            prev_best2 = (int(best_cost), int(best_depth))
                            if cand2 < prev_best2:
                                fb0["best_cost_bits_without_concepts"] = int(cand2[0])
                                fb0["best_depth_without_concepts"] = int(cand2[1])
            if not mids0_s:
                fb0["without_macro_attempts"] = int(fb0.get("without_macro_attempts") or 0) + 1
                if episode_success:
                    fb0["without_macro_successes"] = int(fb0.get("without_macro_successes") or 0) + 1
                    if isinstance(cost_bits_i, int) and isinstance(depth_i, int):
                        best_cost_m = fb0.get("best_cost_bits_without_macros")
                        best_depth_m = fb0.get("best_depth_without_macros")
                        cand3 = (int(cost_bits_i), int(depth_i))
                        if not isinstance(best_cost_m, int) or not isinstance(best_depth_m, int):
                            fb0["best_cost_bits_without_macros"] = int(cand3[0])
                            fb0["best_depth_without_macros"] = int(cand3[1])
                        else:
                            prev_best3 = (int(best_cost_m), int(best_depth_m))
                            if cand3 < prev_best3:
                                fb0["best_cost_bits_without_macros"] = int(cand3[0])
                                fb0["best_depth_without_macros"] = int(cand3[1])
            baselines_by_family[family_id] = _family_baseline_to_json(fb0)

        # Failure clusters: group by residual_signature when present, else family.
        rs = ev.get("residual_signature") if isinstance(ev.get("residual_signature"), dict) else None
        if rs is None and induced_rows:
            # Use the best induced signature (rank 0) as a deterministic failure cone residual signature.
            rs0 = induced_rows[0].get("signature") if isinstance(induced_rows[0], dict) else None
            if isinstance(rs0, dict):
                rs = rs0
        cluster_key = residual_signature_key(rs) if rs else str(family_id)
        if not cluster_key:
            cluster_key = str(family_id)
        cl = clusters_by_key.get(cluster_key)
        if cl is None:
            cl = {
                "cluster_key": str(cluster_key),
                "count": 0,
                "first_run": int(run_idx),
                "last_run": int(run_idx),
                "last_progress_run": -1,
                "induce_attempts": 0,
                "last_induce_run": -1,
                "resolved": False,
                "dead_end": False,
                "families_seen": [str(family_id)],
            }
        else:
            cl = dict(cl)
            cl["last_run"] = int(run_idx)
        fs = set(str(x) for x in (cl.get("families_seen") or []) if str(x))
        if family_id:
            fs.add(str(family_id))
        cl["families_seen"] = sorted(fs)
        if not episode_success:
            cl["count"] = int(cl.get("count") or 0) + 1
        else:
            # Success in the same family is evidence of resolution for family-key clusters.
            if str(cluster_key).startswith("fam_") or str(cluster_key).startswith("fam"):
                cl["resolved"] = True
                cl["last_progress_run"] = int(run_idx)
        clusters_by_key[cluster_key] = _cluster_to_json(cl)

        # Failure cone: concept template emission links a concept to a cluster.
        tmpl0 = ev.get("concept_template") if isinstance(ev.get("concept_template"), dict) else None
        tmpl_rows: List[Dict[str, Any]] = []
        if tmpl0 is not None:
            tmpl_rows.append(dict(tmpl0))
        elif induced_rows:
            # If the solver didn't emit a template, use the best induced (rank 0) concept birth.
            ir0 = induced_rows[0]
            if isinstance(ir0, dict):
                tmpl_rows.append(
                    {
                        "kind": "arc_concept_template_v146",
                        "schema_version": 146,
                        "concept_id": str(ir0.get("concept_id") or ""),
                        "signature": dict(ir0.get("signature") or {}) if isinstance(ir0.get("signature"), dict) else {},
                        "op_ids": list(ir0.get("op_ids") or []) if isinstance(ir0.get("op_ids"), list) else [],
                        "support": 1,
                        "cost_bits": int(ir0.get("cost_bits") or 10),
                    }
                )
        for tmpl in tmpl_rows:
            cid = str(tmpl.get("concept_id") or "")
            if not cid:
                continue
            cur = concepts_by_id.get(cid)
            if cur is None:
                cur = {
                    "concept_id": cid,
                    "signature": tmpl.get("signature") if isinstance(tmpl.get("signature"), dict) else {},
                    "op_ids": [str(x) for x in (tmpl.get("op_ids") if isinstance(tmpl.get("op_ids"), list) else []) if str(x)],
                    "support": int(tmpl.get("support") or 0),
                    "cost_bits": int(tmpl.get("cost_bits") or 10),
                    "state": "candidate",
                    "created_run": int(run_idx),
                    "last_run": int(run_idx),
                    "used_solved": 0,
                    "contexts_used": [],
                    "families_used": [],
                    "origin_clusters": [],
                    "emitted_from_failure": 0,
                }
            else:
                cur = dict(cur)
            cur["emitted_from_failure"] = int(cur.get("emitted_from_failure") or 0) + 1
            oc = set(str(x) for x in (cur.get("origin_clusters") or []) if str(x))
            oc.add(str(cluster_key))
            cur["origin_clusters"] = sorted(oc)
            # candidate -> quarantined once it is a failure-born candidate.
            if str(cur.get("state") or "") == "candidate":
                cur["state"] = "quarantined"
            cur["last_run"] = int(run_idx)
            concepts_by_id[cid] = _concept_to_json(cur)

        # Successful solution: concept calls are compositional evidence.
        cids = ev.get("concept_calls_solution") if isinstance(ev.get("concept_calls_solution"), list) else []
        for cid_any in cids:
            cid = str(cid_any or "")
            if not cid:
                continue
            cur = concepts_by_id.get(cid)
            if cur is None:
                continue
            cur = dict(cur)
            if episode_solved:
                cur["used_solved"] = int(cur.get("used_solved") or 0) + 1
                ctxs = set(str(x) for x in (cur.get("contexts_used") or []) if str(x))
                ctxs.add(str(context_id))
                cur["contexts_used"] = sorted(ctxs)
                fams = set(str(x) for x in (cur.get("families_used") or []) if str(x))
                fams.add(str(family_id))
                cur["families_used"] = sorted(fams)
                # quarantined/candidate -> refining once it actually helped solve something.
                if str(cur.get("state") or "") in ("candidate", "quarantined"):
                    cur["state"] = "refining"
            cur["last_run"] = int(run_idx)
            concepts_by_id[cid] = _concept_to_json(cur)

        # Successful solution: macro_calls are reusable operator evidence.
        mids = ev.get("macro_calls_solution") if isinstance(ev.get("macro_calls_solution"), list) else []
        for mid_any in mids:
            oid = str(mid_any or "")
            if not oid:
                continue
            cur = operators_by_id.get(oid)
            if cur is None:
                continue
            cur = dict(cur)
            if episode_solved:
                cur["used_solved"] = int(cur.get("used_solved") or 0) + 1
                ctxs = set(str(x) for x in (cur.get("contexts_used") or []) if str(x))
                ctxs.add(str(context_id))
                cur["contexts_used"] = sorted(ctxs)
                fams = set(str(x) for x in (cur.get("families_used") or []) if str(x))
                fams.add(str(family_id))
                cur["families_used"] = sorted(fams)
                if str(cur.get("state") or "") == "candidate":
                    cur["state"] = "refining"
            cur["last_run"] = int(run_idx)
            operators_by_id[oid] = _operator_to_json(cur)

    # Promotion validator (deterministic deep semantics gate).
    # preserves_future() v1: promote only if a concept both transfers and reduces Ω-burn proxies.
    hard_families: Set[str] = set()
    for cl in clusters_by_key.values():
        if not isinstance(cl, dict):
            continue
        if int(cl.get("count") or 0) < int(params_eff.cluster_k):
            continue
        fs = cl.get("families_seen") if isinstance(cl.get("families_seen"), list) else []
        for fam in fs:
            fam_s = str(fam or "")
            if fam_s:
                hard_families.add(fam_s)

    promotion_evidence: Dict[str, Dict[str, Any]] = {}
    promoted_now: set[str] = set()

    def _ablation_required_and_passed(*, concept_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        P5 (contrafactual) gate: removing the concept must cause failures on at least
        one task that was previously solved using this concept under the same config.

        Deterministic, bounded, and fully auditable (evidence stored in omega_state).
        """
        cid = str(concept_id or "")
        if not cid:
            return False, {"kind": "ABLATION_INVALID_CONCEPT_ID"}
        # Only attempt ablation when we have concrete tasks where the concept was used in a solved program.
        task_ids = sorted({str(t) for t in (used_task_ids_by_concept.get(cid) or set()) if str(t)})
        if not task_ids:
            return False, {"kind": "ABLATION_NO_TASKS"}

        # Load concept bank rows and filter out this concept id.
        concept_rows: List[Dict[str, Any]] = []
        if concept_bank_path is not None and concept_bank_path.is_file():
            for row in _iter_jsonl(concept_bank_path):
                if not isinstance(row, dict):
                    continue
                if str(row.get("kind") or "") != "arc_concept_template_v146":
                    continue
                concept_rows.append(dict(row))
        concept_rows.sort(key=lambda r: (str(r.get("concept_id") or ""), _stable_json(r)))
        filtered_rows = [r for r in concept_rows if str(r.get("concept_id") or "") != cid]

        # Mirror the run's solver config as closely as possible (no per-task tuning).
        try:
            from atos_core.arc_solver_v141 import SolveConfigV141, solve_arc_task_v141  # type: ignore
            from atos_core.grid_v124 import grid_equal_v124, grid_from_list_v124  # type: ignore
        except Exception as e:
            return False, {"kind": "ABLATION_IMPORT_ERROR", "error": str(e)}

        max_depth_eff = int(summary_obj.get("max_depth") or base_max_depth)
        max_programs_eff = int(summary_obj.get("max_programs") or base_max_programs)
        trace_program_limit = int(summary_obj.get("trace_program_limit") or 80)
        max_ambiguous_outputs = int(summary_obj.get("max_trials") or 3)
        solution_cost_slack_bits = int(summary_obj.get("solution_cost_slack_bits") or 0)
        enable_point_patch_repair = bool(summary_obj.get("enable_point_patch_repair", False))
        point_patch_max_points = int(summary_obj.get("point_patch_max_points") or 12)
        enable_repair_stage = bool(summary_obj.get("enable_repair_stage", True))
        enable_residual_stage = bool(summary_obj.get("enable_residual_stage", True))
        enable_refine_stage = bool(summary_obj.get("enable_refine_stage", True))
        macro_try_on_fail_only = bool(summary_obj.get("macro_try_on_fail_only", True))
        abstraction_pressure = bool(summary_obj.get("abstraction_pressure", False))

        cfg = SolveConfigV141(
            max_depth=int(max_depth_eff),
            max_programs=int(max_programs_eff),
            trace_program_limit=int(trace_program_limit),
            max_ambiguous_outputs=int(max_ambiguous_outputs),
            max_next_steps=128,
            solution_cost_slack_bits=int(solution_cost_slack_bits),
            macro_templates=tuple(macro_rows_for_solver),
            concept_templates=tuple(filtered_rows),
            abstraction_pressure=bool(abstraction_pressure),
            macro_try_on_fail_only=bool(macro_try_on_fail_only),
            enable_repair_stage=bool(enable_repair_stage),
            enable_residual_stage=bool(enable_residual_stage),
            enable_refine_stage=bool(enable_refine_stage),
            enable_point_patch_repair=bool(enable_point_patch_repair),
            point_patch_max_points=int(point_patch_max_points),
        )

        # Deterministic sample of tasks (bounded).
        tasks_tested: List[Dict[str, Any]] = []
        failures = 0
        tested = 0
        for tid in task_ids[:5]:
            # per_task files are named like "<task_id>.json" where task_id already ends with ".json"
            # (i.e., "<task_id>.json.json").
            per_task_path = run_dir / "per_task" / f"{tid}.json"
            if not per_task_path.is_file():
                continue
            try:
                obj = _read_json(per_task_path)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            task = obj.get("task") if isinstance(obj.get("task"), dict) else {}
            train_pairs0 = task.get("train_pairs") if isinstance(task.get("train_pairs"), list) else []
            test_pairs0 = task.get("test_pairs") if isinstance(task.get("test_pairs"), list) else []
            if not train_pairs0 or not test_pairs0 or not isinstance(test_pairs0[0], dict):
                continue
            try:
                train_pairs = []
                for row in train_pairs0:
                    if not isinstance(row, dict):
                        raise ValueError("bad_train_pair")
                    ig = row.get("in_grid")
                    og = row.get("out_grid")
                    if not isinstance(ig, list) or not isinstance(og, list):
                        raise ValueError("bad_train_grid")
                    train_pairs.append((grid_from_list_v124(ig), grid_from_list_v124(og)))
                test_in = test_pairs0[0].get("in_grid")
                test_out = test_pairs0[0].get("out_grid")
                if not isinstance(test_in, list) or not isinstance(test_out, list):
                    raise ValueError("bad_test_grid")
                test_in_g = grid_from_list_v124(test_in)
                want_out_g = grid_from_list_v124(test_out)
            except Exception:
                continue

            ok = False
            status = "FAIL"
            try:
                sol = solve_arc_task_v141(train_pairs=train_pairs, test_in=test_in_g, config=cfg)
                status = str(sol.get("status") or "")
                if status == "SOLVED":
                    pred = sol.get("predicted_grid")
                    if isinstance(pred, list):
                        try:
                            pred_g = grid_from_list_v124(pred)
                            ok = bool(grid_equal_v124(pred_g, want_out_g))
                        except Exception:
                            ok = False
            except Exception:
                ok = False
                status = "EXCEPTION"

            tested += 1
            if not ok:
                failures += 1
            tasks_tested.append({"task_id": str(tid), "ok_without_concept": bool(ok), "status": str(status)})

        if tested == 0:
            return False, {"kind": "ABLATION_NO_VALID_TASKS"}
        # P5: removing the concept must cause at least one failure.
        ok_gate = int(failures) >= 1
        return ok_gate, {
            "kind": "ABLATION",
            "tasks_tested": int(tested),
            "failures_without_concept": int(failures),
            "rows": list(tasks_tested),
        }

    for cid, cur0 in list(concepts_by_id.items()):
        cur = dict(cur0)
        if str(cur.get("state") or "") == "promoted":
            promoted_now.add(cid)
            continue
        if str(cur.get("state") or "") != "refining":
            continue
        if int(cur.get("support") or 0) < int(params_eff.promote_min_support):
            continue
        if int(cur.get("used_solved") or 0) < int(params_eff.promote_min_used_solved):
            continue
        ctxs = cur.get("contexts_used") if isinstance(cur.get("contexts_used"), list) else []
        if len({str(x) for x in ctxs if str(x)}) < int(params_eff.promote_min_contexts):
            continue
        # P2: concept must be born from failure (no survival via ungrounded macros).
        origin = set(str(x) for x in (cur.get("origin_clusters") or []) if str(x))
        if not origin:
            continue
        fams_used = set(str(x) for x in (cur.get("families_used") or []) if str(x))
        # P3: must touch a hard family (recurrent failure region), not only easy regions.
        if not (fams_used & hard_families):
            continue

        # P4: compression/unlock proxy (deterministic).
        witness_ok = False
        witness: Dict[str, Any] = {}
        for fam in sorted(list(fams_used & hard_families)):
            k = (str(cid), str(fam))
            best = best_use.get(k)
            if best is None:
                continue
            cost_bits, depth, task0 = int(best[0]), int(best[1]), str(best[3])
            fb = baselines_by_family.get(str(fam), {})
            wa = int(fb.get("without_concept_attempts") or 0)
            ws = int(fb.get("without_concept_successes") or 0)
            bc = fb.get("best_cost_bits_without_concepts")
            bd = fb.get("best_depth_without_concepts")
            if isinstance(bc, int) and ws > 0:
                if int(cost_bits) < int(bc):
                    witness_ok = True
                    witness = {
                        "family_id": str(fam),
                        "task_id": str(task0),
                        "concept_cost_bits": int(cost_bits),
                        "concept_depth": int(depth),
                        "baseline_best_cost_bits_without_concepts": int(bc),
                        "baseline_best_depth_without_concepts": int(bd) if isinstance(bd, int) else None,
                        "baseline_without_concept_attempts": int(wa),
                        "baseline_without_concept_successes": int(ws),
                        "kind": "COST_IMPROVEMENT",
                    }
                    break
            else:
                # No no-concept success baseline; require repeated attempts and zero successes.
                if int(wa) >= int(params_eff.cluster_k) and int(ws) == 0:
                    witness_ok = True
                    witness = {
                        "family_id": str(fam),
                        "task_id": str(task0),
                        "concept_cost_bits": int(cost_bits),
                        "concept_depth": int(depth),
                        "baseline_without_concept_attempts": int(wa),
                        "baseline_without_concept_successes": int(ws),
                        "kind": "UNLOCKED_IMPOSSIBLE_FAMILY",
                    }
                    break
        if not witness_ok:
            continue

        # P5: contrafactual ablation gate (no promotion without it).
        ab_ok, ab_ev = _ablation_required_and_passed(concept_id=str(cid))
        if not ab_ok:
            continue
        witness["ablation"] = dict(ab_ev)

        cur["state"] = "promoted"
        concepts_by_id[cid] = _concept_to_json(cur)
        promoted_now.add(cid)
        promotion_evidence[str(cid)] = dict(witness)

    promoted_delta = sorted([c for c in promoted_now if c not in promoted_before])

    # Operator promotion (macro_call closures).
    operator_promotion_evidence: Dict[str, Dict[str, Any]] = {}
    promoted_ops_now: set[str] = set()

    def _operator_row_id(row: Dict[str, Any]) -> str:
        kind = str(row.get("kind") or "")
        if kind == "arc_macro_template_v143":
            return str(row.get("macro_id") or "")
        if kind == "arc_operator_template_v147":
            return str(row.get("operator_id") or "")
        return ""

    def _ablation_operator_required_and_passed(*, operator_id: str) -> Tuple[bool, Dict[str, Any]]:
        oid = str(operator_id or "")
        if not oid:
            return False, {"kind": "ABLATION_INVALID_OPERATOR_ID"}
        task_ids = sorted({str(t) for t in (used_task_ids_by_operator.get(oid) or set()) if str(t)})
        if not task_ids:
            return False, {"kind": "ABLATION_NO_TASKS"}

        # Filter macro bank rows to remove this operator id.
        filtered_macros = [r for r in macro_rows_for_solver if _operator_row_id(r) != oid]

        # Load concept bank rows (keep concepts constant during operator ablation).
        concept_rows: List[Dict[str, Any]] = []
        if concept_bank_path is not None and concept_bank_path.is_file():
            for row in _iter_jsonl(concept_bank_path):
                if not isinstance(row, dict):
                    continue
                if str(row.get("kind") or "") != "arc_concept_template_v146":
                    continue
                concept_rows.append(dict(row))
        concept_rows.sort(key=lambda r: (str(r.get("concept_id") or ""), _stable_json(r)))

        try:
            from atos_core.arc_solver_v141 import SolveConfigV141, solve_arc_task_v141  # type: ignore
            from atos_core.grid_v124 import grid_equal_v124, grid_from_list_v124  # type: ignore
        except Exception as e:
            return False, {"kind": "ABLATION_IMPORT_ERROR", "error": str(e)}

        max_depth_eff = int(summary_obj.get("max_depth") or base_max_depth)
        max_programs_eff = int(summary_obj.get("max_programs") or base_max_programs)
        trace_program_limit = int(summary_obj.get("trace_program_limit") or 80)
        max_ambiguous_outputs = int(summary_obj.get("max_trials") or 3)
        solution_cost_slack_bits = int(summary_obj.get("solution_cost_slack_bits") or 0)
        enable_point_patch_repair = bool(summary_obj.get("enable_point_patch_repair", False))
        point_patch_max_points = int(summary_obj.get("point_patch_max_points") or 12)
        enable_repair_stage = bool(summary_obj.get("enable_repair_stage", True))
        enable_residual_stage = bool(summary_obj.get("enable_residual_stage", True))
        enable_refine_stage = bool(summary_obj.get("enable_refine_stage", True))
        macro_try_on_fail_only = bool(summary_obj.get("macro_try_on_fail_only", True))
        abstraction_pressure = bool(summary_obj.get("abstraction_pressure", False))

        cfg = SolveConfigV141(
            max_depth=int(max_depth_eff),
            max_programs=int(max_programs_eff),
            trace_program_limit=int(trace_program_limit),
            max_ambiguous_outputs=int(max_ambiguous_outputs),
            max_next_steps=128,
            solution_cost_slack_bits=int(solution_cost_slack_bits),
            macro_templates=tuple(filtered_macros),
            concept_templates=tuple(concept_rows),
            abstraction_pressure=bool(abstraction_pressure),
            macro_try_on_fail_only=bool(macro_try_on_fail_only),
            enable_repair_stage=bool(enable_repair_stage),
            enable_residual_stage=bool(enable_residual_stage),
            enable_refine_stage=bool(enable_refine_stage),
            enable_point_patch_repair=bool(enable_point_patch_repair),
            point_patch_max_points=int(point_patch_max_points),
        )

        tasks_tested: List[Dict[str, Any]] = []
        failures = 0
        tested = 0
        for tid in task_ids[:5]:
            per_task_path = run_dir / "per_task" / f"{tid}.json"
            if not per_task_path.is_file():
                continue
            try:
                obj = _read_json(per_task_path)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            task = obj.get("task") if isinstance(obj.get("task"), dict) else {}
            train_pairs0 = task.get("train_pairs") if isinstance(task.get("train_pairs"), list) else []
            test_pairs0 = task.get("test_pairs") if isinstance(task.get("test_pairs"), list) else []
            if not train_pairs0 or not test_pairs0 or not isinstance(test_pairs0[0], dict):
                continue
            try:
                train_pairs = []
                for row in train_pairs0:
                    if not isinstance(row, dict):
                        raise ValueError("bad_train_pair")
                    ig = row.get("in_grid")
                    og = row.get("out_grid")
                    train_pairs.append((grid_from_list_v124(ig), grid_from_list_v124(og)))
                test_in_g = grid_from_list_v124(test_pairs0[0].get("in_grid"))
                want_out_g = grid_from_list_v124(test_pairs0[0].get("out_grid"))
            except Exception:
                continue

            ok = False
            status = "FAIL"
            try:
                sol = solve_arc_task_v141(train_pairs=train_pairs, test_in=test_in_g, config=cfg)
                status = str(sol.get("status") or "")
                if status == "SOLVED":
                    pred = sol.get("predicted_grid")
                    if isinstance(pred, list):
                        try:
                            pred_g = grid_from_list_v124(pred)
                            ok = bool(grid_equal_v124(pred_g, want_out_g))
                        except Exception:
                            ok = False
            except Exception:
                ok = False
                status = "EXCEPTION"

            tested += 1
            if not ok:
                failures += 1
            tasks_tested.append({"task_id": str(tid), "ok_without_operator": bool(ok), "status": str(status)})

        if tested == 0:
            return False, {"kind": "ABLATION_NO_VALID_TASKS"}
        ok_gate = int(failures) >= 1
        return ok_gate, {
            "kind": "ABLATION",
            "tasks_tested": int(tested),
            "failures_without_operator": int(failures),
            "rows": list(tasks_tested),
        }

    for oid, cur0 in list(operators_by_id.items()):
        cur = dict(cur0)
        if str(cur.get("state") or "") == "promoted":
            promoted_ops_now.add(oid)
            continue
        if str(cur.get("state") or "") != "refining":
            continue
        # P2 (born-from-failure): operators must be failure-born (no survival via success-mined closures).
        # This mirrors the concept gate above: a reusable operator only exists if it is grounded in
        # an origin failure cluster (and therefore eligible for MAXWELL_Ω selection).
        origin = set(str(x) for x in (cur.get("origin_clusters") or []) if str(x))
        if not origin:
            continue
        if int(cur.get("emitted_from_failure") or 0) <= 0:
            continue
        if int(cur.get("support") or 0) < int(params_eff.promote_min_support):
            continue
        if int(cur.get("used_solved") or 0) < int(params_eff.promote_min_used_solved):
            continue
        ctxs = cur.get("contexts_used") if isinstance(cur.get("contexts_used"), list) else []
        if len({str(x) for x in ctxs if str(x)}) < int(params_eff.promote_min_contexts):
            continue
        fams_used = set(str(x) for x in (cur.get("families_used") or []) if str(x))

        witness_ok = False
        witness: Dict[str, Any] = {}
        # preserves_future() v1 for operators:
        # - If the family was solvable without macros, the operator must reduce MDL (cost_bits).
        # - If the family was NOT solvable without macros under current caps, the operator may be
        #   promoted as an "unlock" (ablation will enforce non-placebo necessity).
        for fam in sorted(list(fams_used)):
            k = (str(oid), str(fam))
            best = best_use_op.get(k)
            if best is None:
                continue
            cost_bits, depth, task0 = int(best[0]), int(best[1]), str(best[2])
            fb = baselines_by_family.get(str(fam), {})
            wa = int(fb.get("without_macro_attempts") or 0)
            ws = int(fb.get("without_macro_successes") or 0)
            bc = fb.get("best_cost_bits_without_macros")
            bd = fb.get("best_depth_without_macros")
            if isinstance(bc, int) and ws > 0:
                if int(cost_bits) < int(bc):
                    witness_ok = True
                    witness = {
                        "family_id": str(fam),
                        "task_id": str(task0),
                        "operator_cost_bits": int(cost_bits),
                        "operator_depth": int(depth),
                        "baseline_best_cost_bits_without_macros": int(bc),
                        "baseline_best_depth_without_macros": int(bd) if isinstance(bd, int) else None,
                        "baseline_without_macro_attempts": int(wa),
                        "baseline_without_macro_successes": int(ws),
                        "kind": "COST_IMPROVEMENT",
                    }
                    break
            else:
                # "Unlocked under caps": at least one attempt without macros, zero successes.
                # We intentionally do NOT require wa>=cluster_k here because strict burn can
                # destroy future before recurrence accumulates; ablation is the anti-cheat gate.
                if int(wa) >= 1 and int(ws) == 0:
                    witness_ok = True
                    witness = {
                        "family_id": str(fam),
                        "task_id": str(task0),
                        "operator_cost_bits": int(cost_bits),
                        "operator_depth": int(depth),
                        "baseline_without_macro_attempts": int(wa),
                        "baseline_without_macro_successes": int(ws),
                        "kind": "UNLOCKED_UNDER_CAPS",
                    }
                    break
        if not witness_ok:
            continue

        ab_ok, ab_ev = _ablation_operator_required_and_passed(operator_id=str(oid))
        if not ab_ok:
            continue
        witness["ablation"] = dict(ab_ev)

        cur["state"] = "promoted"
        operators_by_id[oid] = _operator_to_json(cur)
        promoted_ops_now.add(oid)
        operator_promotion_evidence[str(oid)] = dict(witness)

    promoted_ops_delta = sorted([o for o in promoted_ops_now if o not in promoted_ops_before])

    # MAXWELL_Ω strict burn (canonic): FAIL + Δ(PromotedConceptSet)==∅ => destroy future.
    strict_burns_added = 0
    strict_burn_cluster_key = ""
    if int(failures_seen) > 0 and not promoted_delta and not promoted_ops_delta:
        candidates: List[Dict[str, Any]] = []
        for ck, cl0 in clusters_by_key.items():
            if not isinstance(cl0, dict):
                continue
            if bool(cl0.get("dead_end")) or bool(cl0.get("resolved")):
                continue
            # Avoid re-burning already banned regions.
            if str(ck).startswith("rsig_"):
                if str(ck) in banned_rsig:
                    continue
            else:
                if str(ck) in banned_fams:
                    continue
            candidates.append(dict(cl0))
        if candidates:
            candidates.sort(
                key=lambda cl: (
                    -int(cl.get("count") or 0),
                    int(cl.get("first_run") or 0),
                    int(cl.get("last_progress_run") or -1),
                    str(cl.get("cluster_key") or ""),
                )
            )
            burn = dict(candidates[0])
            ck = str(burn.get("cluster_key") or "")
            if ck:
                burn["dead_end"] = True
                clusters_by_key[ck] = _cluster_to_json(burn)
                strict_burn_cluster_key = str(ck)
                strict_burns_added = 1
                # Ban: residual signature => ban signature + propagate families; else ban family.
                if str(ck).startswith("rsig_"):
                    banned_rsig.add(str(ck))
                    fs = burn.get("families_seen") if isinstance(burn.get("families_seen"), list) else []
                    for fam in fs:
                        fam_s = str(fam or "")
                        if fam_s:
                            banned_fams.add(fam_s)
                else:
                    banned_fams.add(str(ck))

    # Cluster induction + dead-end bans (Ω destructive futures).
    dead_clusters_added = 0
    for ck, cl0 in list(clusters_by_key.items()):
        cl = dict(cl0)
        if bool(cl.get("dead_end")) or bool(cl.get("resolved")):
            continue
        # If already banned, treat as dead-end.
        if str(ck) in banned_fams or str(ck) in banned_rsig:
            cl["dead_end"] = True
            clusters_by_key[ck] = _cluster_to_json(cl)
            continue
        if int(cl.get("count") or 0) < int(params_eff.cluster_k):
            continue
        last_induce = int(cl.get("last_induce_run") or -1)
        if last_induce >= 0 and (int(run_idx) - int(last_induce)) < int(params_eff.cluster_cooldown_runs):
            continue
        # Induction attempt.
        cl["induce_attempts"] = int(cl.get("induce_attempts") or 0) + 1
        cl["last_induce_run"] = int(run_idx)

        # Did this cluster produce any newly promoted concept (Δ promoted set)?
        resolved_by_promotion = False
        for cid in promoted_delta:
            c = concepts_by_id.get(cid) or {}
            oc = set(str(x) for x in (c.get("origin_clusters") or []) if str(x))
            if str(ck) in oc:
                resolved_by_promotion = True
                break
        # Operators are also live structure under MAXWELL_Ω; promotions must count as progress.
        if not resolved_by_promotion:
            for oid in promoted_ops_delta:
                o = operators_by_id.get(oid) or {}
                oc = set(str(x) for x in (o.get("origin_clusters") or []) if str(x))
                if str(ck) in oc:
                    resolved_by_promotion = True
                    break
        if resolved_by_promotion:
            cl["resolved"] = True
            cl["last_progress_run"] = int(run_idx)
            clusters_by_key[ck] = _cluster_to_json(cl)
            continue

        # No promotion: if attempts exhausted => permanently remove reachable future subspace (ban).
        if int(cl.get("induce_attempts") or 0) >= int(params_eff.cluster_max_induce_attempts):
            cl["dead_end"] = True
            dead_clusters_added += 1
            if str(ck).startswith("rsig_"):
                banned_rsig.add(str(ck))
                # Residual signature bans must destroy reachable task families as well (structural future removal).
                fs = cl.get("families_seen") if isinstance(cl.get("families_seen"), list) else []
                for fam in fs:
                    fam_s = str(fam or "")
                    if fam_s:
                        banned_fams.add(fam_s)
            else:
                banned_fams.add(str(ck))
        clusters_by_key[ck] = _cluster_to_json(cl)

    dead_total = int(prev.dead_clusters_total) if prev is not None else 0
    dead_total = int(dead_total + int(dead_clusters_added) + int(strict_burns_added))

    strict_total = int(prev.strict_burns_total) if prev is not None else 0
    strict_total = int(strict_total + int(strict_burns_added))

    # Optional compute caps: burn on dead clusters (secondary, not primary).
    max_programs_cap = int(base_p) - int(dead_total) * int(params_eff.burn_programs_per_dead_cluster)
    max_programs_cap = max(int(params_eff.min_max_programs), int(max_programs_cap))
    depth_dec = int(dead_total) // max(1, int(params_eff.burn_depth_every_n_dead_clusters))
    max_depth_cap = int(base_d) - int(depth_dec)
    max_depth_cap = max(int(params_eff.min_max_depth), int(max_depth_cap))

    prev_sha = str(prev.content_hash()) if prev is not None else ""
    st = OmegaStateV2(
        schema_version=int(OMEGA_SCHEMA_VERSION_V2),
        kind="omega_state_v2",
        base_max_depth=int(base_d),
        base_max_programs=int(base_p),
        params=params_eff,
        runs_total=int(run_idx),
        dead_clusters_total=int(dead_total),
        strict_burns_total=int(strict_total),
        max_depth_cap=int(max_depth_cap),
        max_programs_cap=int(max_programs_cap),
        banned_task_families=tuple(sorted(banned_fams)),
        banned_residual_signatures=tuple(sorted(banned_rsig)),
        concepts=tuple([_concept_to_json(c) for c in concepts_by_id.values()]),
        failure_clusters=tuple([_cluster_to_json(c) for c in clusters_by_key.values()]),
        operators=tuple([_operator_to_json(o) for o in operators_by_id.values()]),
        family_baselines=tuple([_family_baseline_to_json(r) for r in baselines_by_family.values()]),
        prev_state_sha=prev_sha,
        state_sha="",
    )
    st2 = OmegaStateV2(**{**st.__dict__, "state_sha": st.content_hash()})

    info = {
        "runs_total": int(run_idx),
        "failures_seen": int(failures_seen),
        "successes_seen": int(successes_seen),
        "raw_successes_seen": int(raw_successes_seen),
        "shallow_suppressed_successes": int(shallow_suppressed_successes),
        "shallow_suppressed_by_reason": {str(k): int(v) for k, v in sorted(shallow_suppressed_by_reason.items())},
        "promoted_delta": list(promoted_delta),
        "promotion_evidence": {str(k): dict(v) for k, v in sorted(promotion_evidence.items(), key=lambda kv: str(kv[0]))},
        "promoted_ops_delta": list(promoted_ops_delta),
        "operator_promotion_evidence": {
            str(k): dict(v) for k, v in sorted(operator_promotion_evidence.items(), key=lambda kv: str(kv[0]))
        },
        "strict_burn_cluster_key": str(strict_burn_cluster_key),
        "strict_burns_added": int(strict_burns_added),
        "strict_burns_total": int(strict_total),
        "dead_clusters_added": int(dead_clusters_added),
        "dead_clusters_total": int(dead_total),
        "banned_task_families_total": int(len(banned_fams)),
        "banned_residual_signatures_total": int(len(banned_rsig)),
        "max_depth_cap": int(max_depth_cap),
        "max_programs_cap": int(max_programs_cap),
        "state_sha": str(st2.state_sha),
    }
    return st2, info
