from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from atos_core.act import canonical_json_dumps, sha256_hex


OMEGA_SCHEMA_VERSION_V1 = 1


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


@dataclass(frozen=True)
class OmegaParamsV1:
    """
    Ω parameters are global + deterministic (no per-task tuning).

    Ω destroys *future* (search) optionality on any failure that produces no new reusable concept object.
    The only way to survive is to make concepts (concept_templates) that compress decisions and prevent
    failures under a shrinking future.
    """

    min_max_programs: int = 200
    burn_programs_per_failure: int = 10
    min_max_depth: int = 2
    burn_depth_every_n_failures: int = 200

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_max_programs": int(self.min_max_programs),
            "burn_programs_per_failure": int(self.burn_programs_per_failure),
            "min_max_depth": int(self.min_max_depth),
            "burn_depth_every_n_failures": int(self.burn_depth_every_n_failures),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OmegaParamsV1":
        return OmegaParamsV1(
            min_max_programs=int(d.get("min_max_programs", 200)),
            burn_programs_per_failure=int(d.get("burn_programs_per_failure", 10)),
            min_max_depth=int(d.get("min_max_depth", 2)),
            burn_depth_every_n_failures=int(d.get("burn_depth_every_n_failures", 200)),
        )


@dataclass(frozen=True)
class OmegaStateV1:
    schema_version: int
    kind: str
    base_max_depth: int
    base_max_programs: int
    params: OmegaParamsV1
    burns_total: int
    max_depth_cap: int
    max_programs_cap: int
    prev_state_sha: str = ""
    state_sha: str = ""

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": int(self.schema_version),
            "kind": str(self.kind),
            "base_max_depth": int(self.base_max_depth),
            "base_max_programs": int(self.base_max_programs),
            "params": self.params.to_dict(),
            "burns_total": int(self.burns_total),
            "max_depth_cap": int(self.max_depth_cap),
            "max_programs_cap": int(self.max_programs_cap),
            "prev_state_sha": str(self.prev_state_sha or ""),
        }
        # state_sha is a derived field; never include it in its own hash.
        body["state_sha"] = str(self.state_sha or "")
        return body

    def content_hash(self) -> str:
        body = self.to_dict()
        body["state_sha"] = ""
        return sha256_hex(_stable_json(body).encode("utf-8"))

    @staticmethod
    def from_path(path: Path) -> "OmegaStateV1":
        raw = _read_json(path)
        if not isinstance(raw, dict):
            raise ValueError(f"bad_omega_state:{path}")
        params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
        st = OmegaStateV1(
            schema_version=int(raw.get("schema_version", OMEGA_SCHEMA_VERSION_V1)),
            kind=str(raw.get("kind") or "omega_state_v1"),
            base_max_depth=int(raw.get("base_max_depth", 4)),
            base_max_programs=int(raw.get("base_max_programs", 4000)),
            params=OmegaParamsV1.from_dict(params),
            burns_total=int(raw.get("burns_total", 0)),
            max_depth_cap=int(raw.get("max_depth_cap", int(raw.get("base_max_depth", 4)))),
            max_programs_cap=int(raw.get("max_programs_cap", int(raw.get("base_max_programs", 4000)))),
            prev_state_sha=str(raw.get("prev_state_sha") or ""),
            state_sha=str(raw.get("state_sha") or ""),
        )
        # Verify hash if present.
        want = str(st.state_sha or "")
        got = st.content_hash()
        if want and want != got:
            raise ValueError(f"omega_state_hash_mismatch:want={want},got={got}")
        return st

    def write_worm(self, path: Path) -> None:
        st2 = self.to_dict()
        if not st2.get("state_sha"):
            # Write with computed hash.
            st2["state_sha"] = self.content_hash()
        _write_json_x(path, st2)


def count_omega_burns_from_events(path: Path) -> int:
    burns = 0
    for ev in _iter_jsonl(path):
        if bool(ev.get("burn_applied")):
            burns += 1
    return int(burns)


def update_omega_state(
    *,
    prev: Optional[OmegaStateV1],
    burns_in_run: int,
    base_max_depth: int,
    base_max_programs: int,
    params: Optional[OmegaParamsV1] = None,
) -> OmegaStateV1:
    params_eff = params or (prev.params if prev is not None else OmegaParamsV1())
    prev_burns = int(prev.burns_total) if prev is not None else 0
    total_burns = int(prev_burns + int(burns_in_run))

    base_d = int(prev.base_max_depth) if prev is not None else int(base_max_depth)
    base_p = int(prev.base_max_programs) if prev is not None else int(base_max_programs)

    # Destroy future optionality monotonically (irreversible).
    max_programs_cap = int(base_p) - int(total_burns) * int(params_eff.burn_programs_per_failure)
    max_programs_cap = max(int(params_eff.min_max_programs), int(max_programs_cap))

    depth_dec = int(total_burns) // max(1, int(params_eff.burn_depth_every_n_failures))
    max_depth_cap = int(base_d) - int(depth_dec)
    max_depth_cap = max(int(params_eff.min_max_depth), int(max_depth_cap))

    prev_sha = str(prev.content_hash()) if prev is not None else ""
    st = OmegaStateV1(
        schema_version=int(OMEGA_SCHEMA_VERSION_V1),
        kind="omega_state_v1",
        base_max_depth=int(base_d),
        base_max_programs=int(base_p),
        params=params_eff,
        burns_total=int(total_burns),
        max_depth_cap=int(max_depth_cap),
        max_programs_cap=int(max_programs_cap),
        prev_state_sha=prev_sha,
        state_sha="",
    )
    return OmegaStateV1(**{**st.__dict__, "state_sha": st.content_hash()})


def apply_omega_caps(
    *,
    want_max_depth: int,
    want_max_programs: int,
    state: Optional[OmegaStateV1],
) -> Tuple[int, int]:
    if state is None:
        return int(want_max_depth), int(want_max_programs)
    return min(int(want_max_depth), int(state.max_depth_cap)), min(int(want_max_programs), int(state.max_programs_cap))

