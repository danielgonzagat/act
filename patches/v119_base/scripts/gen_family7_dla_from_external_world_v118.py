#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _is_safe_user_turn_text_v118(text: str) -> bool:
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


def _is_compatible_user_turn_text_v118(text: str) -> bool:
    """
    Deterministic compatibility filter to bias tasks toward turns our deterministic
    conversation runtime can actually parse/handle.

    This keeps V118 smoke practical (PASS/FAIL is meaningful) while still sourcing
    turns from real history.
    """
    t = str(text or "").strip()
    if not t:
        return False
    t0 = t.lower()
    # Raw-command style prefixes (project ACT).
    prefixes = [
        "goal:",
        "belief:",
        "crenca:",
        "crença:",
        "beliefs",
        "crencas",
        "crenças",
        "revise:",
        "revisar:",
        "forget ",
        "esquece ",
        "note:",
        "nota:",
        "recall",
        "memoria",
        "memória",
        "system",
        "sistema",
        "about",
        "manual",
        "dossier",
        "regulatory",
        "compliance",
        "versions",
        "versoes",
        "versões",
        "explain",
        "explique",
        "teach:",
        "ensine:",
        "teach_concept:",
        "semantics",
        "pragmatics",
        "flow",
        "dialogue",
        "agency",
        "plans",
        "goals",
        "next",
        "auto",
        "end",
        "fim",
        "quit",
        "summary",
        "resumo",
        "show variables",
        "set ",
        "get ",
        "add ",
    ]
    for p in prefixes:
        if t0.startswith(p):
            return True
    # Single-letter choices often used by the benchmarks.
    if t0 in {"a", "b", "c"}:
        return True
    return False


def _make_task(task_kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = dict(payload)
    body["schema_version"] = 118
    body["task_kind"] = str(task_kind)
    task_id = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return dict(body, task_id=f"family7_dla_v118_{task_id}")


def _injection_plan_for_task_v118(*, task_index: int, total_turns: int) -> List[Dict[str, Any]]:
    """
    Deterministic adversarial injection plan (lightweight).
    """
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
    # V118 focuses on "long but practical" windows (50–200) to keep smoke runtime bounded.
    # Longer horizons are covered by V113+ smoke baselines.
    ap.add_argument("--stress_500", type=int, default=0)
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
    if stress_500 < 0:
        _fail("stress_500_negative")

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
            if role == "user" and _is_safe_user_turn_text_v118(txt):
                m["user_turns_total"] = int(m.get("user_turns_total") or 0) + 1
                if _is_compatible_user_turn_text_v118(txt):
                    m["compat_turns_total"] = int(m.get("compat_turns_total") or 0) + 1
                buf = conv_user_turns[cid]
                if len(buf) < 800:
                    buf.append(str(txt))

    convs: List[Dict[str, Any]] = list(conv_meta.values())
    convs.sort(
        key=lambda d: (
            -int(d.get("compat_turns_total") or 0),
            -int(d.get("user_turns_total") or 0),
            str(d.get("conversation_id") or ""),
        )
    )

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

    # Optional STRESS_500 tier (kept for compatibility, default disabled).
    if stress_500 > 0:
        conv_for_long = _pick_conv(200)
        if conv_for_long is None:
            _fail("not_enough_conversations_for_stress_500")
        for _j in range(stress_500):
            cid = str(conv_for_long.get("conversation_id") or "")
            safe_turns = conv_user_turns.get(cid, [])
            real_sample_n = min(60, len(safe_turns))
            if real_sample_n < 20:
                _fail("not_enough_user_turns_for_stress_500_sample")
            base_turns = safe_turns[:real_sample_n]
            plan = _injection_plan_for_task_v118(task_index=len(tasks), total_turns=500)
            goal_turn = "goal: family7_v118 outcome=complete constraints=deterministic deadline=500"
            user_turns = [goal_turn] + list(base_turns)
            user_turns = _apply_injection_plan(user_turns, plan)
            while len(user_turns) < 500:
                user_turns.append("ok")
            allow_external = bool(not external_allocated)
            if allow_external:
                external_allocated = True
            tasks.append(
                _make_task(
                    "family7_dla_task_v118",
                    {
                        "seed": int(seed),
                        "world_manifest": str(world_manifest.as_posix()),
                        "conversation_id": str(cid),
                        "stress_kind": "STRESS_500",
                        "stress_turns": 500,
                        "user_turns": list(user_turns[:500]),
                        "require_fluency": True,
                        "allow_external_world_once": bool(allow_external),
                        "external_world_probe_reason_code": "validator_failed_fluency_contract",
                        "injection_plan": list(plan),
                    },
                )
            )

    # STRESS_200 tier.
    for _j in range(stress_200):
        c = _pick_conv(40)
        if c is None:
            _fail("not_enough_conversations_for_stress_200")
        cid = str(c.get("conversation_id") or "")
        safe_turns = conv_user_turns.get(cid, [])
        if len(safe_turns) < 12:
            _fail("not_enough_user_turns_for_stress_200_sample")
        real_sample_n = min(60, len(safe_turns))
        base_turns = safe_turns[:real_sample_n]
        plan = _injection_plan_for_task_v118(task_index=len(tasks), total_turns=200)
        goal_turn = "goal: family7_v118 outcome=complete constraints=deterministic deadline=200"
        user_turns = [goal_turn] + list(base_turns)
        user_turns = _apply_injection_plan(user_turns, plan)
        while len(user_turns) < 200:
            user_turns.append("ok")
        allow_external = bool(not external_allocated)
        if allow_external:
            external_allocated = True
        tasks.append(
            _make_task(
                "family7_dla_task_v118",
                {
                    "seed": int(seed),
                    "world_manifest": str(world_manifest.as_posix()),
                    "conversation_id": str(cid),
                    "stress_kind": "STRESS_200",
                    "stress_turns": 200,
                    "user_turns": list(user_turns[:200]),
                    "require_fluency": True,
                    "allow_external_world_once": bool(allow_external),
                    "external_world_probe_reason_code": "validator_failed_fluency_contract",
                    "injection_plan": list(plan),
                },
            )
        )

    # Fill remaining tasks with medium windows (50..150 user turns).
    while len(tasks) < tasks_total:
        c = _pick_conv(20)
        if c is None:
            break
        cid = str(c.get("conversation_id") or "")
        safe_turns = conv_user_turns.get(cid, [])
        if len(safe_turns) < 8:
            continue
        # deterministic window length from seed+len(tasks)
        wlen = 50 + ((int(seed) + int(len(tasks)) * 17) % 101)  # 50..150
        real_sample_n = min(30, len(safe_turns))
        base_turns = safe_turns[:real_sample_n]
        plan = _injection_plan_for_task_v118(task_index=len(tasks), total_turns=int(wlen))
        goal_turn = f"goal: family7_v118 outcome=complete constraints=deterministic deadline={int(wlen)}"
        user_turns = [goal_turn] + list(base_turns)
        user_turns = _apply_injection_plan(user_turns, plan)
        while len(user_turns) < int(wlen):
            user_turns.append("ok")
        allow_external = bool(not external_allocated)
        if allow_external:
            external_allocated = True
        tasks.append(
            _make_task(
                "family7_dla_task_v118",
                {
                    "seed": int(seed),
                    "world_manifest": str(world_manifest.as_posix()),
                    "conversation_id": str(cid),
                    "stress_kind": "MID",
                    "stress_turns": int(wlen),
                    "user_turns": list(user_turns[: int(wlen)]),
                    "require_fluency": True,
                    "allow_external_world_once": bool(allow_external),
                    "external_world_probe_reason_code": "validator_failed_fluency_contract",
                    "injection_plan": list(plan),
                },
            )
        )

    if len(tasks) < tasks_total:
        _fail("not_enough_conversations_for_medium_tasks")

    tasks.sort(key=lambda d: (str(d.get("task_id") or "")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x", encoding="utf-8") as f:
        for t in tasks:
            f.write(canonical_json_dumps(t))
            f.write("\n")

    manifest = {
        "schema_version": 118,
        "kind": "family7_tasks_manifest_v118",
        "seed": int(seed),
        "tasks_total": int(len(tasks)),
        "world_manifest": str(world_manifest.as_posix()),
        "world_canonical_jsonl": str(canon_path.as_posix()),
        "world_manifest_sha256": _sha256_file(world_manifest),
        "world_canonical_sha256": _sha256_file(canon_path),
        "tasks_sha256": _sha256_file(out_path),
    }
    man_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    _ensure_absent(man_path)
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
