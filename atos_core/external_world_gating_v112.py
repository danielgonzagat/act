from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import sha256_hex
from .external_dialogue_world_v111 import load_world_v111
from .external_world_ledger_v111 import (
    EXTERNAL_WORLD_ACTION_OBSERVE_V111,
    EXTERNAL_WORLD_ACTION_SEARCH_V111,
    EXTERNAL_WORLD_REASON_CODES_V111,
    external_world_event_to_dict_v111,
    make_external_world_event_v111,
)


def external_world_access_v112(
    *,
    allowed: bool,
    world_manifest: str,
    action: str,
    reason_code: str,
    args: Dict[str, Any],
    seed: int,
    turn_index: int,
    prev_event_sig: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deterministic gating wrapper for V111 ExternalDialogueWorld:
      - If not allowed -> fail-closed with reason external_world_access_not_allowed.
      - If reason_code invalid -> fail-closed with reason invalid_reason_code.
    Returns: (events_jsonl_payloads, result_summary).
    """
    if not bool(allowed):
        raise ValueError("external_world_access_not_allowed")
    if str(reason_code) not in EXTERNAL_WORLD_REASON_CODES_V111:
        raise ValueError("invalid_reason_code")

    world = load_world_v111(manifest_path=str(world_manifest))

    result_summary: Dict[str, Any] = {"seed": int(seed)}
    if str(action) == EXTERNAL_WORLD_ACTION_SEARCH_V111:
        q = str((args or {}).get("query") or "")
        limit = int((args or {}).get("limit") or 3)
        roles = (args or {}).get("roles") if isinstance((args or {}).get("roles"), list) else ["user"]
        matches = world.search_text(query=str(q), limit=int(limit), roles=[str(r) for r in roles if isinstance(r, str)])
        result_summary.update(
            {
                "query": str(q),
                "matches_total": int(len(matches)),
                "matches": [
                    {
                        "global_turn_index": int(m.get("global_turn_index") or 0),
                        "conversation_id": str(m.get("conversation_id") or ""),
                        "role": str(m.get("role") or ""),
                    }
                    for m in matches
                    if isinstance(m, dict)
                ],
            }
        )
    elif str(action) == EXTERNAL_WORLD_ACTION_OBSERVE_V111:
        s = int((args or {}).get("start_turn") or 0)
        e = int((args or {}).get("end_turn") or 0)
        roles = (args or {}).get("roles") if isinstance((args or {}).get("roles"), list) else ["user"]
        turns = world.observe_range(start_turn=int(s), end_turn=int(e), roles=[str(r) for r in roles if isinstance(r, str)], limit=10)
        result_summary.update(
            {
                "observed_total": int(len(turns)),
                "observed_hash": sha256_hex(
                    ("\n".join([str(t.text) for t in turns]) if turns else "").encode("utf-8")
                ),
            }
        )
    else:
        raise ValueError("invalid_external_world_action")

    ev = make_external_world_event_v111(
        event_index=0,
        turn_index=int(turn_index),
        action=str(action),
        reason_code=str(reason_code),
        args=dict(args),
        result_summary=dict(result_summary),
        prev_event_sig=str(prev_event_sig or ""),
    )
    return [external_world_event_to_dict_v111(ev)], dict(result_summary)

