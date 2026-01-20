from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from .concept_registry_v70 import interface_sig_from_act, program_sha256_from_act
from .pcc_v73 import build_certificate_v1
from .store import ActStore
from .trace_v73 import PlanStepTraceV73, TraceV73


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _sorted_strs(xs: Sequence[str]) -> List[str]:
    return [str(x) for x in sorted([str(x) for x in xs if str(x)], key=str)]


@dataclass(frozen=True)
class InducedCandidateV73:
    subpath: Tuple[str, ...]
    support: int
    contexts: Tuple[str, ...]
    rep_steps: Tuple[PlanStepTraceV73, ...]
    score_key: Tuple[int, int, str]


def _enumerate_subpaths(path: Sequence[str], *, max_k: int) -> List[Tuple[int, Tuple[str, ...]]]:
    out: List[Tuple[int, Tuple[str, ...]]] = []
    p = [str(x) for x in path if str(x)]
    n = int(len(p))
    for k in range(2, min(int(max_k), n) + 1):
        for i in range(0, n - k + 1):
            out.append((int(i), tuple(p[i : i + k])))
    return out


def _select_best_subpath(
    *,
    traces: Sequence[TraceV73],
    max_k: int,
    min_support: int,
) -> Tuple[Optional[InducedCandidateV73], Dict[str, Any]]:
    stats: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for tr in traces:
        if not isinstance(tr, TraceV73):
            continue
        path = tr.acts_path()
        steps = list(tr.steps)
        for start, sub in _enumerate_subpaths(path, max_k=int(max_k)):
            if start < 0 or start + len(sub) > len(steps):
                continue
            rec = stats.setdefault(
                sub,
                {
                    "contexts": set(),
                    "occurrences": [],
                },
            )
            rec["contexts"].add(str(tr.context_id))
            rec["occurrences"].append(
                {
                    "trace_sig": str(tr.trace_sig()),
                    "start": int(start),
                    "steps": steps[int(start) : int(start) + int(len(sub))],
                }
            )

    candidates: List[InducedCandidateV73] = []
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
            k = (str(o.get("trace_sig") or ""), int(o.get("start", 0) or 0))
            if rep is None or k < rep_key:
                rep = o
                rep_key = k
        if rep is None:
            continue
        rep_steps = tuple(s for s in rep.get("steps", []) if isinstance(s, PlanStepTraceV73))
        sub_sig = _stable_hash_obj([str(x) for x in sub])
        score_key = (-int(len(sub)), -int(support), str(sub_sig))
        candidates.append(
            InducedCandidateV73(
                subpath=tuple(sub),
                support=int(support),
                contexts=tuple(ctxs),
                rep_steps=rep_steps,
                score_key=score_key,
            )
        )

    debug: Dict[str, Any] = {"subpaths_total": int(len(stats)), "candidates_total": int(len(candidates))}
    if not candidates:
        return None, {**debug, "reason": "no_candidate_meets_support"}
    candidates.sort(key=lambda c: (int(c.score_key[0]), int(c.score_key[1]), str(c.score_key[2])))
    best = candidates[0]
    return best, {**debug, "reason": "ok", "best": {"subpath": list(best.subpath), "support": int(best.support), "score_key": list(best.score_key)}}


def _slotize_interface(
    *,
    store: ActStore,
    steps: Sequence[PlanStepTraceV73],
) -> Tuple[Dict[str, str], Dict[str, str], str, Dict[str, Any]]:
    produced: Set[str] = set()
    required_external: Set[str] = set()
    for st in steps:
        bm = st.bind_map if isinstance(st.bind_map, dict) else {}
        for _, vname in bm.items():
            vn = str(vname or "")
            if not vn:
                continue
            if vn not in produced:
                required_external.add(vn)
        produced.add(str(st.produces or ""))

    output_key = str(steps[-1].produces or "") if steps else ""
    input_vars = sorted(set(str(x) for x in required_external if str(x)))

    input_schema: Dict[str, str] = {}
    for v in input_vars:
        want_t = "str"
        for st in steps:
            bm = st.bind_map if isinstance(st.bind_map, dict) else {}
            slot = None
            for k, vv in bm.items():
                if str(vv or "") == v:
                    slot = str(k or "")
                    break
            if not slot:
                continue
            act = store.get_concept_act(str(st.concept_id))
            ev = act.evidence if (act is not None and isinstance(act.evidence, dict)) else {}
            iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
            iface = iface if isinstance(iface, dict) else {}
            in_schema = iface.get("input_schema") if isinstance(iface.get("input_schema"), dict) else {}
            if slot in in_schema:
                want_t = str(in_schema.get(slot) or "str")
            break
        input_schema[str(v)] = str(want_t)

    output_schema: Dict[str, str] = {}
    validator_id = "text_exact"
    out_t = "str"
    if steps:
        last = steps[-1]
        act = store.get_concept_act(str(last.concept_id))
        ev = act.evidence if (act is not None and isinstance(act.evidence, dict)) else {}
        iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
        iface = iface if isinstance(iface, dict) else {}
        validator_id = str(iface.get("validator_id") or "text_exact")
        out_schema = iface.get("output_schema") if isinstance(iface.get("output_schema"), dict) else {}
        if output_key and output_key in out_schema:
            out_t = str(out_schema.get(output_key) or "str")
    if output_key:
        output_schema[str(output_key)] = str(out_t)

    debug = {"required_external": list(input_vars), "output_key": str(output_key), "validator_id": str(validator_id)}
    return input_schema, output_schema, str(validator_id), debug


def induce_concept_v73(
    traces: Sequence[TraceV73],
    store: ActStore,
    *,
    max_k: int = 6,
    min_support: int = 2,
    seed: int = 0,
) -> Tuple[Act, Dict[str, Any], Dict[str, Any]]:
    traces2 = [t for t in traces if isinstance(t, TraceV73)]
    if int(len(traces2)) < int(min_support):
        raise ValueError("not_enough_traces")

    best, dbg_select = _select_best_subpath(traces=traces2, max_k=int(max_k), min_support=int(min_support))
    if best is None:
        raise ValueError(f"no_inducible_subpath:{dbg_select.get('reason')}")

    steps = list(best.rep_steps)
    in_schema, out_schema, validator_id, dbg_iface = _slotize_interface(store=store, steps=steps)
    if not in_schema or not out_schema:
        raise ValueError("slotization_failed")

    # Synthesize a composed concept_csv: sequence of CSV_CALLs + CSV_RETURN.
    program: List[Instruction] = []
    for name in sorted(in_schema.keys(), key=str):
        program.append(Instruction("CSV_GET_INPUT", {"name": str(name), "out": str(name)}))
    for st in steps:
        bm = {str(k): str(st.bind_map.get(k) or "") for k in sorted(st.bind_map.keys(), key=str)}
        program.append(
            Instruction(
                "CSV_CALL",
                {
                    "concept_id": str(st.concept_id),
                    "bind": dict(bm),
                    "out": str(st.produces),
                },
            )
        )
    output_key = str(steps[-1].produces or "")
    program.append(Instruction("CSV_RETURN", {"var": str(output_key)}))

    interface = {"input_schema": dict(in_schema), "output_schema": dict(out_schema), "validator_id": str(validator_id)}

    act_id_sig = _stable_hash_obj(
        {
            "schema_version": 1,
            "induced_kind": "subpath_concepts_v1",
            "subpath": [str(x) for x in best.subpath],
            "interface": interface,
            "program": [ins.to_dict() for ins in program],
        }
    )
    act_id = f"concept_v73_induced_{act_id_sig}"

    store_hash_base = store.content_hash()
    candidate = Act(
        id=str(act_id),
        version=1,
        created_at=deterministic_iso(step=0),
        kind="concept_csv",
        match={},
        program=list(program),
        evidence={
            "interface": dict(interface),
            "induced_v73": {
                "schema_version": 1,
                "method": "subpath_concepts_v1",
                "subpath": [str(x) for x in best.subpath],
                "support": int(best.support),
                "contexts": _sorted_strs(best.contexts),
                "store_hash_base": str(store_hash_base),
            },
        },
        cost={"overhead_bits": 1024},
        deps=_sorted_strs(set(best.subpath)),
        active=True,
    )

    cert = build_certificate_v1(candidate_act=candidate, traces=traces2, store_base=store, seed=int(seed))
    if isinstance(candidate.evidence, dict):
        candidate.evidence["pcc_v73"] = {
            "certificate_kind": str(cert.get("certificate_kind") or ""),
            "certificate_sig": str(cert.get("certificate_sig") or ""),
            "store_hash_base": str(cert.get("store_hash_base") or ""),
        }

    iface_sig = interface_sig_from_act(candidate)
    prog_sha = program_sha256_from_act(candidate)
    concept_sig = _stable_hash_obj(
        {"concept_id": str(candidate.id), "interface_sig": str(iface_sig), "program_sha256": str(prog_sha)}
    )

    delta_steps = int(len(best.subpath)) - 1
    debug = {
        "selection": dict(dbg_select),
        "interface": dict(dbg_iface),
        "store_hash_base": str(store_hash_base),
        "candidate": {
            "act_id": str(candidate.id),
            "interface_sig": str(iface_sig),
            "program_sha256": str(prog_sha),
            "concept_sig": str(concept_sig),
            "subpath": [str(x) for x in best.subpath],
            "steps_len": int(len(best.subpath)),
            "delta_mdl_steps": int(delta_steps),
        },
        "certificate_sig": str(cert.get("certificate_sig") or ""),
    }
    return candidate, cert, debug

