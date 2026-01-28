from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from .engine import Engine, EngineConfig
from .store import ActStore
from .trace_v73 import PlanStepTraceV73, TraceV73


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _safe_deepcopy(obj: Any) -> Any:
    try:
        return copy.deepcopy(obj)
    except Exception:
        if isinstance(obj, dict):
            return dict(obj)
        if isinstance(obj, list):
            return list(obj)
        return obj


def _enumerate_subpaths(path: Sequence[str], *, max_k: int) -> List[Tuple[int, Tuple[str, ...]]]:
    out: List[Tuple[int, Tuple[str, ...]]] = []
    p = [str(x) for x in path if str(x)]
    n = int(len(p))
    for k in range(2, min(int(max_k), n) + 1):
        for i in range(0, n - k + 1):
            out.append((int(i), tuple(p[i : i + k])))
    return out


def gain_bits_est_v74(*, subpath_len: int, support: int) -> int:
    """
    MVP deterministic gain proxy (bits):
      - each removed step contributes a constant
      - additional support contexts contribute a smaller constant
    This is stable and intentionally simple (audit-first).
    """
    removed_steps = max(0, int(subpath_len) - 1)
    return int(removed_steps) * 1000 + max(0, int(support) - 1) * 100


@dataclass(frozen=True)
class MinedCandidateV74:
    subpath: Tuple[str, ...]
    support_contexts: int
    contexts: Tuple[str, ...]
    rep_trace_sig: str
    start_idx: int
    sub_sig: str
    gain_bits_est: int
    score_key: Tuple[int, int, int, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subpath": [str(x) for x in self.subpath],
            "support_contexts": int(self.support_contexts),
            "contexts": [str(x) for x in self.contexts],
            "rep_trace_sig": str(self.rep_trace_sig),
            "start_idx": int(self.start_idx),
            "sub_sig": str(self.sub_sig),
            "gain_bits_est": int(self.gain_bits_est),
            "score_key": [int(self.score_key[0]), int(self.score_key[1]), int(self.score_key[2]), str(self.score_key[3])],
        }


def mine_candidates_v74(
    *,
    traces: Sequence[TraceV73],
    max_k: int = 6,
    min_support: int = 2,
    top_k: int = 8,
) -> Tuple[List[MinedCandidateV74], Dict[str, Any]]:
    traces2 = [t for t in traces if isinstance(t, TraceV73)]
    stats: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    for tr in traces2:
        path = tr.acts_path()
        for start, sub in _enumerate_subpaths(path, max_k=int(max_k)):
            rec = stats.setdefault(sub, {"contexts": set(), "occurrences": []})
            rec["contexts"].add(str(tr.context_id))
            rec["occurrences"].append({"trace_sig": str(tr.trace_sig()), "start_idx": int(start)})

    mined: List[MinedCandidateV74] = []
    for sub, rec in stats.items():
        ctxs = sorted(set(str(x) for x in rec.get("contexts", set()) if str(x)))
        support = int(len(ctxs))
        if support < int(min_support):
            continue
        occs = rec.get("occurrences") if isinstance(rec.get("occurrences"), list) else []
        rep = None
        rep_key = None
        for o in occs:
            if not isinstance(o, dict):
                continue
            k = (str(o.get("trace_sig") or ""), int(o.get("start_idx", 0) or 0))
            if rep is None or k < rep_key:
                rep = o
                rep_key = k
        if rep is None:
            continue
        sub_sig = _stable_hash_obj([str(x) for x in sub])
        gain = gain_bits_est_v74(subpath_len=len(sub), support=support)
        # Ranking: gain DESC, len DESC, support DESC, sub_sig ASC.
        score_key = (-int(gain), -int(len(sub)), -int(support), str(sub_sig))
        mined.append(
            MinedCandidateV74(
                subpath=tuple(sub),
                support_contexts=int(support),
                contexts=tuple(ctxs),
                rep_trace_sig=str(rep.get("trace_sig") or ""),
                start_idx=int(rep.get("start_idx", 0) or 0),
                sub_sig=str(sub_sig),
                gain_bits_est=int(gain),
                score_key=score_key,
            )
        )

    mined.sort(key=lambda c: (int(c.score_key[0]), int(c.score_key[1]), int(c.score_key[2]), str(c.score_key[3])))
    out = mined[: int(top_k)] if int(top_k) > 0 else mined
    debug = {
        "subpaths_total": int(len(stats)),
        "candidates_total": int(len(mined)),
        "top_k": int(top_k),
        "ranking": {"primary": "gain_bits_est_desc", "tie_breaks": ["len_desc", "support_desc", "sub_sig_asc"], "gain_bits_est_formula": "removed_steps*1000 + (support-1)*100"},
    }
    return list(out), debug


def extract_rep_steps(
    *,
    traces_by_sig: Dict[str, TraceV73],
    rep_trace_sig: str,
    start_idx: int,
    subpath_len: int,
) -> List[PlanStepTraceV73]:
    tr = traces_by_sig.get(str(rep_trace_sig))
    if tr is None:
        raise ValueError("rep_trace_not_found")
    steps = list(tr.steps)
    s = int(start_idx)
    k = int(subpath_len)
    if s < 0 or s + k > len(steps):
        raise ValueError("rep_steps_out_of_range")
    return [steps[i] for i in range(s, s + k)]


def _external_inputs_for_steps(steps: Sequence[PlanStepTraceV73]) -> List[str]:
    produced: Set[str] = set()
    required: Set[str] = set()
    for st in steps:
        bm = st.bind_map if isinstance(st.bind_map, dict) else {}
        for _, vname in bm.items():
            vn = str(vname or "")
            if not vn:
                continue
            if vn not in produced:
                required.add(vn)
        produced.add(str(st.produces or ""))
    return [str(x) for x in sorted(required, key=str)]


def materialize_composed_act_v74(
    *,
    store_base: ActStore,
    steps: Sequence[PlanStepTraceV73],
    support_contexts: int,
    contexts: Sequence[str],
    gain_bits_est: int = 0,
    overhead_bits: int = 1024,
    seed_step: int = 0,
) -> Tuple[Act, Dict[str, Any]]:
    """
    Materialize a composed concept_csv with hygienic temporaries (_t0, _t1, ...).
    """
    if not steps:
        raise ValueError("empty_steps")

    # Determine external inputs and final output key.
    external = _external_inputs_for_steps(steps)
    out_key = str(steps[-1].produces or "")
    if not out_key:
        raise ValueError("missing_output_key")

    # Infer types from store interfaces (best-effort; default 'str').
    input_schema: Dict[str, str] = {}
    for v in external:
        input_schema[str(v)] = "str"
    for st in steps:
        act = store_base.get_concept_act(str(st.concept_id))
        if act is None:
            raise ValueError(f"missing_dep:{st.concept_id}")
        ev = act.evidence if isinstance(act.evidence, dict) else {}
        iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
        iface = iface if isinstance(iface, dict) else {}
        in_schema = iface.get("input_schema") if isinstance(iface.get("input_schema"), dict) else {}
        bm = st.bind_map if isinstance(st.bind_map, dict) else {}
        for slot, vname in bm.items():
            vn = str(vname or "")
            if vn in input_schema and str(slot) in in_schema:
                input_schema[vn] = str(in_schema.get(str(slot)) or "str")

    validator_id = "text_exact"
    output_schema: Dict[str, str] = {str(out_key): "str"}
    last_act = store_base.get_concept_act(str(steps[-1].concept_id))
    if last_act is not None and isinstance(last_act.evidence, dict):
        iface = last_act.evidence.get("interface") if isinstance(last_act.evidence.get("interface"), dict) else {}
        iface = iface if isinstance(iface, dict) else {}
        validator_id = str(iface.get("validator_id") or validator_id)
        out_schema = iface.get("output_schema") if isinstance(iface.get("output_schema"), dict) else {}
        if out_key in out_schema:
            output_schema[str(out_key)] = str(out_schema.get(out_key) or "str")

    # Heuristic: composed plan-style concepts should validate with plan_validator when they
    # output a string and have the canonical plan inputs in scope. This enables composed
    # concepts to be eligible for the plan_validator concept executor in the runtime.
    try:
        if (
            str(output_schema.get(str(out_key)) or "") == "str"
            and "goal_id" in input_schema
            and "plan" in input_schema
        ):
            validator_id = "plan_validator"
    except Exception:
        pass

    # Hygienic temporaries: map produced var -> env var.
    produced_map: Dict[str, str] = {}
    for i, st in enumerate(steps):
        prod = str(st.produces or "")
        if not prod:
            raise ValueError("step_missing_produces")
        if i == len(steps) - 1:
            produced_map[prod] = prod
        else:
            produced_map[prod] = f"_t{i}"

    program: List[Instruction] = []
    for name in sorted(input_schema.keys(), key=str):
        program.append(Instruction("CSV_GET_INPUT", {"name": str(name), "out": str(name)}))

    for i, st in enumerate(steps):
        bm = st.bind_map if isinstance(st.bind_map, dict) else {}
        bind: Dict[str, str] = {}
        for slot in sorted(bm.keys(), key=str):
            vn = str(bm.get(slot) or "")
            if vn in produced_map:
                bind[str(slot)] = str(produced_map[vn])
            else:
                bind[str(slot)] = str(vn)
        out_var = str(produced_map.get(str(st.produces or ""), str(st.produces or "")))
        program.append(Instruction("CSV_CALL", {"concept_id": str(st.concept_id), "bind": dict(bind), "out": out_var}))

    program.append(Instruction("CSV_RETURN", {"var": str(out_key)}))

    interface = {"input_schema": dict(input_schema), "output_schema": dict(output_schema), "validator_id": str(validator_id)}
    act_id_sig = _stable_hash_obj(
        {
            "schema_version": 1,
            "induced_kind": "subpath_concepts_v74",
            "subpath": [str(s.concept_id) for s in steps],
            "interface": interface,
            "program": [ins.to_dict() for ins in program],
        }
    )
    act_id = f"concept_v74_induced_{act_id_sig}"

    overhead_bits0 = int(overhead_bits)
    if overhead_bits0 < 0:
        overhead_bits0 = 0

    # Match constraints (family_ids) should propagate through composition to prevent
    # selecting a composed concept in a context where its callees are disallowed.
    #
    # Use the intersection of explicit callee family_ids when available.
    match: Dict[str, Any] = {}
    try:
        fam_inter: Optional[set] = None
        for st in steps:
            callee = store_base.get_concept_act(str(st.concept_id))
            if callee is None or not isinstance(getattr(callee, "match", None), dict):
                continue
            fams = callee.match.get("family_ids")
            if not isinstance(fams, list) or not fams:
                continue
            s = {str(x) for x in fams if isinstance(x, str) and str(x)}
            if not s:
                continue
            fam_inter = set(s) if fam_inter is None else fam_inter.intersection(s)
        if fam_inter is not None and fam_inter:
            match = {"family_ids": sorted(fam_inter, key=str)}
    except Exception:
        match = {}
    act = Act(
        id=str(act_id),
        version=1,
        created_at=deterministic_iso(step=int(seed_step)),
        kind="concept_csv",
        match=dict(match),
        program=list(program),
        evidence={
            "interface": dict(interface),
            "meta": {"birth_tags": ["plan"]},
            "induced_v74": {
                "schema_version": 1,
                "method": "subpath_concepts_v74",
                "support_contexts": int(support_contexts),
                "contexts": [str(x) for x in sorted(set(str(c) for c in contexts if str(c)), key=str)],
                "gain_bits_est": int(gain_bits_est),
                "overhead_bits": int(overhead_bits0),
            },
        },
        cost={"overhead_bits": int(overhead_bits0)},
        deps=[str(s.concept_id) for s in steps],
        active=True,
    )

    # CSG (Concept SubGraph) explicit representation (audit-first):
    # - Not used by the executor directly (concept_csv program is canonical),
    #   but recorded as explicit state so that "concepts are subgraphs" is
    #   mechanically verifiable and merge/split can reason over structure.
    try:
        from .csg_v87 import canonicalize_csg_v87, csg_hash_v87  # local import

        csg_nodes: List[Dict[str, Any]] = []
        for ins in program:
            if not isinstance(ins, Instruction) or str(ins.op) != "CSV_CALL":
                continue
            args = ins.args if isinstance(ins.args, dict) else {}
            csg_nodes.append(
                {
                    "act_id": str(args.get("concept_id") or ""),
                    "bind": dict(args.get("bind") if isinstance(args.get("bind"), dict) else {}),
                    "produces": str(args.get("out") or ""),
                    "role": "call",
                }
            )
        csg = canonicalize_csg_v87(
            {
                "schema_version": 1,
                "nodes": list(csg_nodes),
                "interface": {
                    "inputs": sorted([str(k) for k in input_schema.keys() if str(k)], key=str),
                    "outputs": [str(out_key)],
                    "validator_ids": [str(validator_id)] if str(validator_id) else [],
                    "match": dict(match),
                },
            }
        )
        # Persist in evidence (explicit state).
        ev2 = act.evidence if isinstance(act.evidence, dict) else {}
        ev2 = dict(ev2)
        ev2["csg_v87"] = dict(csg)
        # Add a stable hash pointer in meta for quick inspection without loading the whole CSG.
        meta2 = ev2.get("meta") if isinstance(ev2.get("meta"), dict) else {}
        meta2 = dict(meta2)
        meta2["csg_v87_hash"] = str(csg_hash_v87(dict(csg)))
        meta2["csg_v87_nodes"] = int(len(csg.get("nodes") or [])) if isinstance(csg.get("nodes"), list) else 0
        meta2["csg_v87_edges"] = int(len(csg.get("edges") or [])) if isinstance(csg.get("edges"), list) else 0
        ev2["meta"] = dict(meta2)
        act.evidence = dict(ev2)
    except Exception:
        pass

    debug = {
        "external_inputs": list(external),
        "output_key": str(out_key),
        "validator_id": str(validator_id),
        "hygiene": {"produced_map": dict(produced_map)},
    }
    return act, debug


def materialize_deep_wrapper_act_v74(
    *,
    store_base: ActStore,
    inner_concept_id: str,
    overhead_bits: int = 1024,
    seed_step: int = 0,
) -> Tuple[Act, Dict[str, Any]]:
    """
    Deterministically materialize a depth-raising wrapper:
      wrapper -> CSV_CALL(inner) -> RETURN.

    This is used only to satisfy existential "deep concept" survival laws when
    the system has composed concepts (depth=1) but no nested hierarchy (depth>=2).
    """
    inner_id = str(inner_concept_id or "")
    if not inner_id:
        raise ValueError("missing_inner_concept_id")
    inner = store_base.get_concept_act(inner_id)
    if inner is None:
        raise ValueError(f"inner_concept_not_found:{inner_id}")
    inner_match: Dict[str, Any] = {}
    try:
        if isinstance(getattr(inner, "match", None), dict):
            inner_match = dict(inner.match or {})
    except Exception:
        inner_match = {}
    ev = inner.evidence if isinstance(inner.evidence, dict) else {}
    iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
    iface = iface if isinstance(iface, dict) else {}
    in_schema = iface.get("input_schema") if isinstance(iface.get("input_schema"), dict) else {}
    out_schema = iface.get("output_schema") if isinstance(iface.get("output_schema"), dict) else {}
    validator_id = str(iface.get("validator_id") or "")
    if not isinstance(in_schema, dict) or not in_schema:
        raise ValueError("inner_missing_input_schema")
    if not isinstance(out_schema, dict) or not out_schema:
        raise ValueError("inner_missing_output_schema")
    if len(out_schema) != 1:
        raise ValueError("inner_output_schema_not_singleton")
    out_key = str(sorted(out_schema.keys(), key=str)[0])
    if not out_key:
        raise ValueError("inner_missing_output_key")

    program: List[Instruction] = []
    input_schema: Dict[str, str] = {str(k): str(v) for k, v in in_schema.items() if str(k)}
    output_schema: Dict[str, str] = {str(out_key): str(out_schema.get(out_key) or "str")}
    for name in sorted(input_schema.keys(), key=str):
        program.append(Instruction("CSV_GET_INPUT", {"name": str(name), "out": str(name)}))
    bind = {str(k): str(k) for k in sorted(input_schema.keys(), key=str)}
    program.append(
        Instruction("CSV_CALL", {"concept_id": str(inner_id), "bind": dict(bind), "out": str(out_key)})
    )
    program.append(Instruction("CSV_RETURN", {"var": str(out_key)}))

    interface = {"input_schema": dict(input_schema), "output_schema": dict(output_schema), "validator_id": str(validator_id)}
    act_id_sig = _stable_hash_obj(
        {
            "schema_version": 1,
            "induced_kind": "deep_wrapper_v74",
            "inner_concept_id": str(inner_id),
            "interface": interface,
            "program": [ins.to_dict() for ins in program],
        }
    )
    act_id = f"concept_v74_deepwrap_{act_id_sig}"

    overhead_bits0 = int(overhead_bits)
    if overhead_bits0 < 0:
        overhead_bits0 = 0
    act = Act(
        id=str(act_id),
        version=1,
        created_at=deterministic_iso(step=int(seed_step)),
        kind="concept_csv",
        match=dict(inner_match),
        program=list(program),
        evidence={
            "interface": dict(interface),
            "meta": {"birth_tags": ["plan"]},
            "induced_v74_deepwrap": {
                "schema_version": 1,
                "method": "deep_wrapper_v74",
                "inner_concept_id": str(inner_id),
                "overhead_bits": int(overhead_bits0),
            },
        },
        cost={"overhead_bits": int(overhead_bits0)},
        deps=[str(inner_id)],
        active=True,
    )
    try:
        from .csg_v87 import canonicalize_csg_v87, csg_hash_v87  # local import

        csg_nodes: List[Dict[str, Any]] = []
        for ins in program:
            if not isinstance(ins, Instruction) or str(ins.op) != "CSV_CALL":
                continue
            args = ins.args if isinstance(ins.args, dict) else {}
            csg_nodes.append(
                {
                    "act_id": str(args.get("concept_id") or ""),
                    "bind": dict(args.get("bind") if isinstance(args.get("bind"), dict) else {}),
                    "produces": str(args.get("out") or ""),
                    "role": "call",
                }
            )
        csg = canonicalize_csg_v87(
            {
                "schema_version": 1,
                "nodes": list(csg_nodes),
                "interface": {
                    "inputs": sorted([str(k) for k in input_schema.keys() if str(k)], key=str),
                    "outputs": [str(out_key)],
                    "validator_ids": [str(validator_id)] if str(validator_id) else [],
                    "match": dict(inner_match),
                },
            }
        )
        ev2 = act.evidence if isinstance(act.evidence, dict) else {}
        ev2 = dict(ev2)
        ev2["csg_v87"] = dict(csg)
        meta2 = ev2.get("meta") if isinstance(ev2.get("meta"), dict) else {}
        meta2 = dict(meta2)
        meta2["csg_v87_hash"] = str(csg_hash_v87(dict(csg)))
        meta2["csg_v87_nodes"] = int(len(csg.get("nodes") or [])) if isinstance(csg.get("nodes"), list) else 0
        meta2["csg_v87_edges"] = int(len(csg.get("edges") or [])) if isinstance(csg.get("edges"), list) else 0
        ev2["meta"] = dict(meta2)
        act.evidence = dict(ev2)
    except Exception:
        pass
    debug = {"inner_concept_id": str(inner_id), "output_key": str(out_key), "validator_id": str(validator_id)}
    return act, debug


def execute_steps_expected_output(
    *,
    store_base: ActStore,
    steps: Sequence[PlanStepTraceV73],
    bindings: Dict[str, Any],
    seed: int = 0,
) -> str:
    """
    Deterministically execute the original multi-step plan (steps) on bindings,
    using store_base concepts, to derive expected output for certificates.
    """
    vars_state: Dict[str, Any] = _safe_deepcopy(bindings) if isinstance(bindings, dict) else {}
    engine = Engine(store_base, seed=int(seed), config=EngineConfig(enable_contracts=False))
    for i, st in enumerate(steps):
        bm = st.bind_map if isinstance(st.bind_map, dict) else {}
        concept_inputs: Dict[str, Any] = {}
        for slot in sorted(bm.keys(), key=str):
            vn = str(bm.get(slot) or "")
            concept_inputs[str(slot)] = vars_state.get(vn)
        out = engine.execute_concept_csv(
            concept_act_id=str(st.concept_id),
            inputs=dict(concept_inputs),
            expected=None,
            step=int(i),
            max_depth=8,
            max_events=512,
            validate_output=False,
        )
        meta = out.get("meta") if isinstance(out, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        out_text = str(meta.get("output_text") or out.get("output") or "")
        vars_state[str(st.produces)] = out_text
    final_key = str(steps[-1].produces or "")
    return str(vars_state.get(final_key) or "")


def mutate_bindings_plus1_numeric(
    *,
    bindings: Dict[str, Any],
    key_preference: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Deterministic extra-vector mutation: choose a key and +1 its numeric value while preserving zero-padding.
    Only mutates one key. Fallback: if no digits, append '1'.
    """
    b = _safe_deepcopy(bindings) if isinstance(bindings, dict) else {}
    if not isinstance(b, dict):
        return {}
    keys = [str(k) for k in b.keys()]
    keys.sort(key=str)
    if key_preference:
        pref = [str(k) for k in key_preference if str(k)]
        pref2 = [k for k in pref if k in keys]
        if pref2:
            keys = pref2 + [k for k in keys if k not in set(pref2)]
    if not keys:
        return dict(b)

    k0 = keys[0]
    v0 = str(b.get(k0) or "")
    digits = "".join(ch for ch in v0 if ch.isdigit())
    if digits:
        width = len(digits)
        try:
            n = int(digits)
        except Exception:
            n = 0
        n2 = n + 1
        out_digits = str(int(n2))
        if len(out_digits) < width:
            out_digits = out_digits.zfill(width)
        b[k0] = out_digits
        return dict(b)

    b[k0] = v0 + "1"
    return dict(b)
