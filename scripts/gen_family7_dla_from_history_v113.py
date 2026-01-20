#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ensure_absent(path: Path) -> None:
    if path.exists():
        _fail(f"worm_exists:{path}")


def _load_world_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        _fail(f"missing_world_manifest:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _world_jsonl_path_from_manifest(manifest_path: Path) -> Path:
    m = _load_world_manifest(manifest_path)
    paths = m.get("paths") if isinstance(m.get("paths"), dict) else {}
    rel = str(paths.get("canonical_jsonl") or "dialogue_history_canonical_v113.jsonl")
    return (manifest_path.parent / rel).resolve()


def _is_safe_user_turn_text_v113(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    if len(t) > 800:
        return False
    t0 = t.lstrip()
    if t0.startswith("{") or t0.startswith("["):
        return False
    bad_substrings = [
        "content_type",
        "asset_pointer",
        "file-service://",
        "multimodal_text",
        "image_asset_pointer",
    ]
    for s in bad_substrings:
        if s in t0:
            return False
    if t0.count("\n") > 10:
        return False
    return True


def _make_task(task_kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = dict(payload)
    body["schema_version"] = 113
    body["task_kind"] = str(task_kind)
    task_id = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return dict(body, task_id=f"family7_dla_v113_{task_id}")


def _injection_plan_for_task_v113(*, task_index: int, total_turns: int) -> List[Dict[str, Any]]:
    plan: List[Tuple[str, int, str]] = []
    plan.append(("dont_invent", 12, "não invente; se não souber diga não sei e pergunte o dado"))
    cycle = int(task_index) % 7
    if cycle == 0:
        plan.append(("late_reference", min(total_turns - 5, 90), "como eu disse antes, faz isso do mesmo jeito"))
    elif cycle == 1:
        plan.append(("soft_contradiction", min(total_turns - 5, 45), "na verdade era X, não Y"))
    elif cycle == 2:
        plan.append(("implicit_goal_shift", min(total_turns - 5, 70), "agora muda o objetivo sem avisar e vê se você percebe"))
    elif cycle == 3:
        plan.append(("irony", min(total_turns - 5, 55), "claro... sqn"))
    elif cycle == 4:
        plan.append(("late_reference_far", min(total_turns - 5, 140), "isso que eu falei lá atrás continua valendo"))
    elif cycle == 5:
        plan.append(("hostile_confused", min(total_turns - 5, 30), "?? você não entendeu nada"))
    elif cycle == 6:
        plan.append(("minimalist_trap", min(total_turns - 5, 80), "ok"))
    plan.sort(key=lambda t: (int(t[1]), str(t[0])))
    out: List[Dict[str, Any]] = []
    for kind, pos, text in plan:
        out.append({"kind": str(kind), "pos": int(pos), "text": str(text)})
    return out


def _apply_injection_plan(turns: List[str], plan: Sequence[Dict[str, Any]]) -> List[str]:
    out = list(turns)
    for inj in sorted(plan, key=lambda d: (int(d.get("pos") or 0), str(d.get("kind") or ""))):
        pos = int(inj.get("pos") or 0)
        txt = str(inj.get("text") or "")
        if not txt:
            continue
        if pos < 0:
            pos = 0
        if pos > len(out):
            pos = len(out)
        out.insert(pos, txt)
    return list(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world_manifest", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks_total", type=int, default=20)
    ap.add_argument("--stress_200", type=int, default=2)
    ap.add_argument("--stress_500", type=int, default=1)
    args = ap.parse_args()

    seed = int(args.seed)
    out_path = Path(str(args.out))
    _ensure_absent(out_path)

    world_manifest = Path(str(args.world_manifest))
    canon_path = _world_jsonl_path_from_manifest(world_manifest)
    if not canon_path.exists():
        _fail(f"missing_world_canonical_jsonl:{canon_path}")

    tasks_total = int(args.tasks_total)
    stress_200 = int(args.stress_200)
    stress_500 = int(args.stress_500)
    if tasks_total < 20:
        _fail("tasks_total_too_small")
    if stress_200 < 2:
        _fail("stress_200_too_small")
    if stress_500 < 1:
        _fail("stress_500_too_small")

    # Build per-conversation user-turn buffers (deterministic, bounded).
    conv_meta: Dict[str, Dict[str, Any]] = {}
    conv_user_turns: Dict[str, List[str]] = {}
    with open(canon_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = str(obj.get("conversation_id") or "")
            if not cid:
                continue
            idx = int(obj.get("global_turn_index") or 0)
            role = str(obj.get("role") or "unknown")
            txt = str(obj.get("text") or "")
            m = conv_meta.get(cid)
            if m is None:
                m = {"conversation_id": cid, "start_turn": idx, "end_turn": idx, "turns_total": 0, "user_turns_total": 0}
                conv_meta[cid] = m
                conv_user_turns[cid] = []
            m["turns_total"] = int(m.get("turns_total") or 0) + 1
            if idx < int(m.get("start_turn") or idx):
                m["start_turn"] = int(idx)
            if idx > int(m.get("end_turn") or idx):
                m["end_turn"] = int(idx)
            if role == "user" and _is_safe_user_turn_text_v113(txt):
                m["user_turns_total"] = int(m.get("user_turns_total") or 0) + 1
                # Cap to avoid memory blowup; still enough for STRESS_500 sampling.
                buf = conv_user_turns[cid]
                if len(buf) < 800:
                    buf.append(str(txt))

    # Candidate conversations by user_turns_total DESC then conversation_id ASC.
    convs: List[Dict[str, Any]] = list(conv_meta.values())
    convs.sort(key=lambda d: (-int(d.get("user_turns_total") or 0), str(d.get("conversation_id") or "")))

    # Select conversations deterministically for stress tiers.
    used_conv_ids: set = set()
    tasks: List[Dict[str, Any]] = []
    external_allocated = False

    def _pick_conv(min_user_turns: int) -> Optional[Dict[str, Any]]:
        for c in convs:
            cid0 = str(c.get("conversation_id") or "")
            if cid0 in used_conv_ids:
                continue
            if int(c.get("user_turns_total") or 0) < int(min_user_turns):
                continue
            used_conv_ids.add(cid0)
            return dict(c)
        return None

    # Build STRESS_500 (or fallback STRESS_300 if not enough user turns).
    stress_long_kind = "STRESS_500"
    stress_long_turns = 500
    conv_for_500 = _pick_conv(200)
    if conv_for_500 is None:
        _fail("not_enough_conversations_for_stress_long")
    safe_turns_500 = conv_user_turns.get(str(conv_for_500.get("conversation_id") or ""), [])
    if len(safe_turns_500) < 60:
        # If the dataset doesn't support long realistic windows, fallback to STRESS_300.
        stress_long_kind = "STRESS_300"
        stress_long_turns = 300

    for j in range(stress_500):
        if conv_for_500 is None:
            break
        cid = str(conv_for_500.get("conversation_id") or "")
        safe_turns = conv_user_turns.get(cid, [])
        real_sample_n = min(120, len(safe_turns))
        if real_sample_n < 20:
            _fail("not_enough_user_turns_for_stress_long_sample")
        base_turns = safe_turns[:real_sample_n]
        plan = _injection_plan_for_task_v113(task_index=len(tasks), total_turns=int(stress_long_turns))
        goal_turn = "goal: family7_v113 outcome=complete constraints=deterministic deadline={n}".format(n=int(stress_long_turns))
        user_turns = [goal_turn] + list(base_turns)
        user_turns = _apply_injection_plan(user_turns, plan)
        while len(user_turns) < int(stress_long_turns):
            user_turns.append("ok")
        allow_external = bool(not external_allocated)
        if allow_external:
            external_allocated = True
        tasks.append(
            _make_task(
                "family7_dla_task_v113",
                {
                    "seed": int(seed),
                    "world_manifest": str(world_manifest.as_posix()),
                    "conversation_id": str(cid),
                    "window": {"start_turn": int(conv_for_500.get("start_turn") or 0), "end_turn": int(conv_for_500.get("end_turn") or 0)},
                    "stress_kind": str(stress_long_kind),
                    "minimal_ok_turns": int(stress_long_turns),
                    "real_user_sample_turns": int(real_sample_n),
                    "injection_plan": list(plan),
                    "allow_external_world_once": bool(allow_external),
                    "external_world_probe_reason_code": "validator_failed_fluency_contract",
                    "expected_validators": ["fluency_survival_v112", "binding_unresolved_reference_zero", "semantic_contradiction_zero"],
                    "user_turns": list(user_turns),
                },
            )
        )

    # Build STRESS_200 tasks.
    stress_built = 0
    while stress_built < stress_200:
        c = _pick_conv(80)
        if c is None:
            _fail("not_enough_conversations_for_stress_200")
        cid = str(c.get("conversation_id") or "")
        safe_turns = conv_user_turns.get(cid, [])
        real_sample_n = min(60, len(safe_turns))
        if real_sample_n < 12:
            continue
        base_turns = safe_turns[:real_sample_n]
        plan = _injection_plan_for_task_v113(task_index=len(tasks), total_turns=200)
        goal_turn = "goal: family7_v113 outcome=complete constraints=deterministic deadline=200"
        user_turns = [goal_turn] + list(base_turns)
        user_turns = _apply_injection_plan(user_turns, plan)
        while len(user_turns) < 200:
            user_turns.append("ok")
        tasks.append(
            _make_task(
                "family7_dla_task_v113",
                {
                    "seed": int(seed),
                    "world_manifest": str(world_manifest.as_posix()),
                    "conversation_id": str(cid),
                    "window": {"start_turn": int(c.get("start_turn") or 0), "end_turn": int(c.get("end_turn") or 0)},
                    "stress_kind": "STRESS_200",
                    "minimal_ok_turns": 200,
                    "real_user_sample_turns": int(real_sample_n),
                    "injection_plan": list(plan),
                    "allow_external_world_once": False,
                    "external_world_probe_reason_code": "validator_failed_fluency_contract",
                    "expected_validators": ["fluency_survival_v112", "binding_unresolved_reference_zero", "semantic_contradiction_zero"],
                    "user_turns": list(user_turns),
                },
            )
        )
        stress_built += 1

    # Fill remaining tasks (medium horizon).
    while len(tasks) < tasks_total:
        c = _pick_conv(20)
        if c is None:
            _fail("not_enough_conversations_for_medium_tasks")
        cid = str(c.get("conversation_id") or "")
        safe_turns = conv_user_turns.get(cid, [])
        real_sample_n = min(30, len(safe_turns))
        if real_sample_n < 8:
            continue
        base_turns = safe_turns[:real_sample_n]
        plan = _injection_plan_for_task_v113(task_index=len(tasks), total_turns=120)
        goal_turn = "goal: family7_v113 outcome=complete constraints=deterministic deadline=120"
        user_turns = [goal_turn] + list(base_turns)
        user_turns = _apply_injection_plan(user_turns, plan)
        while len(user_turns) < 120:
            user_turns.append("ok")
        tasks.append(
            _make_task(
                "family7_dla_task_v113",
                {
                    "seed": int(seed),
                    "world_manifest": str(world_manifest.as_posix()),
                    "conversation_id": str(cid),
                    "window": {"start_turn": int(c.get("start_turn") or 0), "end_turn": int(c.get("end_turn") or 0)},
                    "stress_kind": "MEDIUM_120",
                    "minimal_ok_turns": 120,
                    "real_user_sample_turns": int(real_sample_n),
                    "injection_plan": list(plan),
                    "allow_external_world_once": False,
                    "external_world_probe_reason_code": "validator_failed_fluency_contract",
                    "expected_validators": ["fluency_survival_v112", "binding_unresolved_reference_zero", "semantic_contradiction_zero"],
                    "user_turns": list(user_turns),
                },
            )
        )

    # Write tasks JSONL (WORM).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x", encoding="utf-8") as f:
        for t in tasks:
            f.write(canonical_json_dumps(t))
            f.write("\n")

    # Write manifest (WORM).
    manifest_path = out_path.with_suffix(out_path.suffix + "_manifest.json")
    _ensure_absent(manifest_path)
    manifest = {
        "schema_version": 113,
        "kind": "family7_dla_tasks_v113",
        "seed": int(seed),
        "world_manifest": str(world_manifest.as_posix()),
        "world_canonical_sha256": _sha256_file(canon_path),
        "tasks_total": int(len(tasks)),
        "stress_200_total": int(sum(1 for t in tasks if str(t.get("stress_kind") or "") == "STRESS_200")),
        "stress_500_total": int(sum(1 for t in tasks if str(t.get("stress_kind") or "") == "STRESS_500")),
        "stress_300_total": int(sum(1 for t in tasks if str(t.get("stress_kind") or "") == "STRESS_300")),
        "tasks_sha256": _sha256_file(out_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"ok": True, "out": str(out_path), "manifest": str(manifest_path), "manifest_obj": manifest}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
