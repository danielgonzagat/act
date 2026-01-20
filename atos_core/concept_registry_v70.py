from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .act import Act, canonical_json_dumps, deterministic_iso, sha256_hex
from .ethics import validate_act_for_promotion
from .proof import verify_concept_pcc_v2
from .store import ActStore
from .toc import detect_duplicate, toc_eval


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(canonical_json_dumps(row))
        f.write("\n")


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return iter(())
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_chained_jsonl(path: str, row: Dict[str, Any], *, prev_hash: Optional[str]) -> str:
    body = dict(row)
    body["prev_hash"] = prev_hash
    entry_hash = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    body["entry_hash"] = entry_hash
    _append_jsonl(path, body)
    return entry_hash


def verify_chained_jsonl(path: str) -> bool:
    prev = None
    for row in _read_jsonl(path):
        row = dict(row)
        entry_hash = row.pop("entry_hash", None)
        if row.get("prev_hash") != prev:
            return False
        expected = sha256_hex(canonical_json_dumps(row).encode("utf-8"))
        if expected != entry_hash:
            return False
        prev = entry_hash
    return True


def concept_sig(concept_obj: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(concept_obj).encode("utf-8"))


def interface_sig_from_act(act: Act) -> str:
    ev = act.evidence if isinstance(act.evidence, dict) else {}
    iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
    iface = iface if isinstance(iface, dict) else {}
    body = {
        "in": iface.get("input_schema", {}),
        "out": iface.get("output_schema", {}),
        "validator_id": str(iface.get("validator_id") or ""),
    }
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def program_sha256_from_act(act: Act) -> str:
    prog = [ins.to_dict() for ins in (act.program or [])]
    return sha256_hex(canonical_json_dumps(prog).encode("utf-8"))


def _slots_and_invariants(act: Act) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ev = act.evidence if isinstance(act.evidence, dict) else {}
    iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
    iface = iface if isinstance(iface, dict) else {}
    inp = iface.get("input_schema") if isinstance(iface.get("input_schema"), dict) else {}
    out = iface.get("output_schema") if isinstance(iface.get("output_schema"), dict) else {}
    slots = {
        "inputs": sorted(str(k) for k in inp.keys()),
        "outputs": sorted(str(k) for k in out.keys()),
    }
    invariants = {
        "input_schema": {str(k): str(v) for k, v in sorted(inp.items(), key=lambda kv: str(kv[0]))},
        "output_schema": {str(k): str(v) for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))},
        "validator_id": str(iface.get("validator_id") or ""),
    }
    return slots, invariants


@dataclass
class ConceptObjectV70:
    concept_id: str
    concept_state: str
    interface_sig: str
    program_sha256: str
    program_len: int
    cost: float
    evidence_refs: List[Dict[str, Any]]
    slots: Dict[str, Any]
    invariants: Dict[str, Any]
    created_step: int
    last_step: int

    toc_attempts: int = 0
    toc_failures: int = 0
    duplicate: bool = False
    duplicate_of: str = ""
    dead_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "concept_id": str(self.concept_id),
            "concept_state": str(self.concept_state),
            "interface_sig": str(self.interface_sig),
            "program_sha256": str(self.program_sha256),
            "program_len": int(self.program_len),
            "cost": float(self.cost),
            "evidence_refs": list(self.evidence_refs),
            "slots": dict(self.slots),
            "invariants": dict(self.invariants),
            "created_step": int(self.created_step),
            "last_step": int(self.last_step),
            "toc_attempts": int(self.toc_attempts),
            "toc_failures": int(self.toc_failures),
            "duplicate": bool(self.duplicate),
            "duplicate_of": str(self.duplicate_of),
            "dead_reason": str(self.dead_reason),
        }
        body["concept_sig"] = concept_sig(body)
        return body


@dataclass
class ConceptRegistryV70:
    run_dir: str
    toc_fail_threshold: int = 2
    similarity_threshold: float = 0.95

    registry_path: str = field(init=False)
    evidence_path: str = field(init=False)

    _registry_prev_hash: Optional[str] = field(default=None, init=False)
    _evidence_prev_hash: Optional[str] = field(default=None, init=False)
    _concepts: Dict[str, ConceptObjectV70] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        os.makedirs(self.run_dir, exist_ok=False)
        self.registry_path = os.path.join(self.run_dir, "concept_registry.jsonl")
        self.evidence_path = os.path.join(self.run_dir, "concept_registry_evidence.jsonl")

    def define(self, *, step: int, concept_act: Act, reason: str) -> ConceptObjectV70:
        cid = str(concept_act.id)
        if cid in self._concepts:
            return self._concepts[cid]

        iface_sig = interface_sig_from_act(concept_act)
        prog_sha = program_sha256_from_act(concept_act)
        slots, invariants = _slots_and_invariants(concept_act)
        c = ConceptObjectV70(
            concept_id=str(cid),
            concept_state="CANDIDATE",
            interface_sig=str(iface_sig),
            program_sha256=str(prog_sha),
            program_len=int(len(concept_act.program or [])),
            cost=float(len(concept_act.program or [])),
            evidence_refs=[],
            slots=slots,
            invariants=invariants,
            created_step=int(step),
            last_step=int(step),
        )
        self._concepts[cid] = c

        self._registry_prev_hash = append_chained_jsonl(
            self.registry_path,
            {
                "time": deterministic_iso(step=int(step)),
                "step": int(step),
                "event": "DEFINE",
                "reason": str(reason),
                "concept": c.to_dict(),
            },
            prev_hash=self._registry_prev_hash,
        )
        return c

    def get(self, concept_id: str) -> Optional[ConceptObjectV70]:
        return self._concepts.get(str(concept_id))

    def _append_state(self, *, step: int, concept: ConceptObjectV70, event: str, reason: str) -> None:
        self._registry_prev_hash = append_chained_jsonl(
            self.registry_path,
            {
                "time": deterministic_iso(step=int(step)),
                "step": int(step),
                "event": str(event),
                "reason": str(reason),
                "concept": concept.to_dict(),
            },
            prev_hash=self._registry_prev_hash,
        )

    def _append_evidence(self, *, step: int, row: Dict[str, Any]) -> None:
        self._evidence_prev_hash = append_chained_jsonl(
            self.evidence_path,
            {"time": deterministic_iso(step=int(step)), "step": int(step), **row},
            prev_hash=self._evidence_prev_hash,
        )

    def attempt_promote_with_toc(
        self,
        *,
        step: int,
        candidate: Act,
        store: ActStore,
        vectors_A: Sequence[Dict[str, Any]],
        vectors_B: Sequence[Dict[str, Any]],
        domain_A: str,
        domain_B: str,
        existing_for_duplicates: Sequence[Act],
    ) -> Dict[str, Any]:
        """
        ToC as LAW for promotion:
          - Must pass ethics promotion check (deterministic).
          - Must not be DUPLICATE (exact or near-duplicate).
          - Must pass ToC in both domains.
          - GC: after toc_fail_threshold failures, mark DEAD (append-only; never delete).
        """
        c = self.define(step=int(step), concept_act=candidate, reason="auto_define")

        # Freeze structural identity (fail-closed on mismatch).
        iface_sig = interface_sig_from_act(candidate)
        prog_sha = program_sha256_from_act(candidate)
        if iface_sig != c.interface_sig or prog_sha != c.program_sha256:
            c.concept_state = "DEAD"
            c.dead_reason = "structural_mismatch"
            c.last_step = int(step)
            self._append_state(step=int(step), concept=c, event="STATE", reason="structural_mismatch")
            self._append_evidence(
                step=int(step),
                row={
                    "event": "STRUCTURAL_MISMATCH",
                    "concept_id": str(c.concept_id),
                    "want_interface_sig": str(c.interface_sig),
                    "got_interface_sig": str(iface_sig),
                    "want_program_sha256": str(c.program_sha256),
                    "got_program_sha256": str(prog_sha),
                },
            )
            return {"ok": False, "reason": "structural_mismatch", "concept": c.to_dict()}

        eth = validate_act_for_promotion(candidate)
        if not bool(eth.ok):
            c.concept_state = "DEAD"
            c.dead_reason = "ethics_fail_closed_promotion"
            c.last_step = int(step)
            self._append_state(step=int(step), concept=c, event="STATE", reason="ethics_fail_closed_promotion")
            self._append_evidence(
                step=int(step),
                row={
                    "event": "PROMOTION_BLOCKED",
                    "reason": "ethics_fail_closed_promotion",
                    "concept_id": str(c.concept_id),
                    "ethics": eth.to_dict(),
                },
            )
            return {"ok": False, "reason": "ethics_fail_closed_promotion", "concept": c.to_dict()}

        dup = detect_duplicate(candidate, existing=existing_for_duplicates, similarity_threshold=float(self.similarity_threshold))
        if isinstance(dup, dict):
            c.duplicate = True
            c.duplicate_of = str(dup.get("other_id") or "")
            c.concept_state = "DEAD"
            c.dead_reason = str(dup.get("reason") or "duplicate")
            c.last_step = int(step)
            self._append_state(step=int(step), concept=c, event="STATE", reason="duplicate_block")
            self._append_evidence(
                step=int(step),
                row={
                    "event": "DUPLICATE_BLOCK",
                    "concept_id": str(c.concept_id),
                    "duplicate": dict(dup),
                },
            )
            return {"ok": False, "reason": "duplicate_block", "duplicate": dict(dup), "concept": c.to_dict()}

        # PCC is optional: verify only if a certificate_v2 is present (deterministic).
        pcc_ok = True
        pcc_verdict = None
        ev = candidate.evidence if isinstance(candidate.evidence, dict) else {}
        cert = ev.get("certificate_v2") if isinstance(ev.get("certificate_v2"), dict) else None
        if isinstance(cert, dict):
            pcc_verdict = verify_concept_pcc_v2(candidate, store)
            pcc_ok = bool(pcc_verdict.ok)
        if not bool(pcc_ok):
            c.concept_state = "DEAD"
            c.dead_reason = "pcc_failed"
            c.last_step = int(step)
            self._append_state(step=int(step), concept=c, event="STATE", reason="pcc_failed")
            self._append_evidence(
                step=int(step),
                row={
                    "event": "PCC_FAILED",
                    "concept_id": str(c.concept_id),
                    "pcc": pcc_verdict.to_dict() if pcc_verdict is not None else None,
                },
            )
            return {"ok": False, "reason": "pcc_failed", "concept": c.to_dict(), "pcc": pcc_verdict.to_dict() if pcc_verdict is not None else None}

        c.toc_attempts += 1
        toc = toc_eval(
            concept_act=candidate,
            vectors_A=list(vectors_A),
            vectors_B=list(vectors_B),
            store=store,
            domain_A=str(domain_A),
            domain_B=str(domain_B),
            min_vectors_per_domain=3,
        )
        details = toc.get("details") if isinstance(toc.get("details"), dict) else {}
        results_A = details.get("results_A") if isinstance(details.get("results_A"), list) else []
        results_B = details.get("results_B") if isinstance(details.get("results_B"), list) else []
        unc_ic = 0
        for r in list(results_A) + list(results_B):
            if isinstance(r, dict) and str(r.get("uncertainty_mode_out") or "") == "IC":
                unc_ic += 1

        self._append_evidence(
            step=int(step),
            row={
                "event": "TOC_ATTEMPT",
                "concept_id": str(c.concept_id),
                "toc": dict(toc),
                "uncertainty_ic_count": int(unc_ic),
            },
        )

        pass_A = bool(toc.get("pass_A", False))
        pass_B = bool(toc.get("pass_B", False))
        toc_ok = bool(pass_A and pass_B and int(unc_ic) == 0)

        if not toc_ok:
            c.toc_failures += 1
            c.last_step = int(step)
            reason = "toc_failed"
            if c.toc_failures >= int(self.toc_fail_threshold):
                c.concept_state = "DEAD"
                c.dead_reason = "toc_fail_threshold"
                reason = "gc_dead_toc_fail_threshold"
            self._append_state(step=int(step), concept=c, event="STATE", reason=str(reason))
            return {
                "ok": False,
                "reason": str(reason),
                "concept": c.to_dict(),
                "toc": dict(toc),
                "uncertainty_ic_count": int(unc_ic),
            }

        # Promotion: ACTIVE only if ToC OK.
        c.concept_state = "ACTIVE"
        c.last_step = int(step)
        c.evidence_refs.append(
            {
                "kind": "toc_v1",
                "domains": [str(domain_A), str(domain_B)],
                "toc_sig": sha256_hex(canonical_json_dumps(toc).encode("utf-8")),
            }
        )
        self._append_state(step=int(step), concept=c, event="STATE", reason="promoted_active_toc_ok")
        return {"ok": True, "reason": "promoted_active_toc_ok", "concept": c.to_dict(), "toc": dict(toc)}

    def verify_chains(self) -> Dict[str, bool]:
        return {
            "concept_registry_chain_ok": bool(verify_chained_jsonl(self.registry_path)),
            "concept_registry_evidence_chain_ok": bool(verify_chained_jsonl(self.evidence_path)),
        }

