from __future__ import annotations

import json
import os
from typing import Any, Dict, Sequence

from .conversation_loop_v117 import run_conversation_v117


def _write_once_json(path: str, obj: Any) -> None:
    if os.path.exists(path):
        raise ValueError(f"worm_exists:{path}")
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        raise ValueError(f"tmp_exists:{tmp}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def run_conversation_v118(
    *,
    user_turn_texts: Sequence[str],
    out_dir: str,
    seed: int,
    max_plan_attempts: int = 8,
    max_replans_per_turn: int = 3,
) -> Dict[str, Any]:
    """
    V118 wrapper around V117 baseline.

    V118 scope in this repo branch:
      - keep V117 conversation semantics stable (regression safety),
      - add a versioned final-response artifact (`final_response_v118.json`) so callers can pin to V118,
      - keep determinism/WORM semantics unchanged.
    """
    res = run_conversation_v117(
        user_turn_texts=list(user_turn_texts),
        out_dir=str(out_dir),
        seed=int(seed),
        max_plan_attempts=int(max_plan_attempts),
        max_replans_per_turn=int(max_replans_per_turn),
    )

    fr117_path = os.path.join(str(out_dir), "final_response_v117.json")
    fr117 = _read_json(fr117_path) if os.path.exists(fr117_path) else {}
    ok117 = bool(fr117.get("ok", False)) if isinstance(fr117, dict) else False
    reason117 = str(fr117.get("reason") or "") if isinstance(fr117, dict) else "missing_final_response_v117"

    final_obj = {
        "schema_version": 118,
        "kind": "final_response_v118",
        "ok": bool(ok117),
        "reason": str(reason117 if not ok117 else "ok"),
        "upstream": {"final_response_v117": dict(fr117) if isinstance(fr117, dict) else {}},
    }
    _write_once_json(os.path.join(str(out_dir), "final_response_v118.json"), dict(final_obj))

    out = dict(res)
    out["final_response_v118"] = dict(final_obj)
    return out

