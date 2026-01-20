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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _world_dialogue_jsonl_from_manifest(manifest_path: Path) -> Tuple[Path, Dict[str, Any]]:
    m = _load_json(manifest_path)
    if not isinstance(m, dict) or int(m.get("schema_version") or 0) != 122:
        _fail("invalid_world_manifest_schema_v122")
    paths = m.get("paths") if isinstance(m.get("paths"), dict) else {}
    rel = str(paths.get("dialogue_history_canonical_jsonl") or "")
    if not rel:
        _fail("missing_world_manifest_dialogue_path")
    canon = (manifest_path.parent / rel).resolve()
    if not canon.exists():
        _fail(f"missing_world_canonical_jsonl:{canon}")
    return canon, dict(m)


def _is_safe_user_turn_text_v122(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    if len(t) > 800:
        return False
    t0 = t.lstrip()
    if t0.startswith("{") or t0.startswith("["):
        return False
    if t0.count("\n") > 10:
        return False
    return True


def _is_compatible_user_turn_text_v122(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    t0 = t.lower()
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
        "bindings",
        "trace_",
        "explain_",
        "why ",
        "porque ",
    ]
    for p in prefixes:
        if t0.startswith(p):
            return True
    if t0 in {"a", "b", "c"}:
        return True
    return False


def _make_task(task_kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = dict(payload)
    body["schema_version"] = 122
    body["task_kind"] = str(task_kind)
    task_id = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return dict(body, task_id=f"family7_dla_v122_{task_id}")


def _injection_plan_for_task_v122(*, task_index: int, total_turns: int) -> List[Dict[str, Any]]:
    plan: List[Tuple[str, int, str]] = []
    plan.append(("dont_invent", 10, "não invente; se não souber diga não sei e pergunte o dado"))
    cycle = int(task_index) % 6
    if cycle == 0:
        plan.append(("late_reference", min(total_turns - 5, 80), "como eu disse antes, faz isso do mesmo jeito"))
    elif cycle == 1:
        plan.append(("soft_contradiction", min(total_turns - 5, 45), "na verdade era X, não Y"))
    elif cycle == 2:
        plan.append(("implicit_goal_shift", min(total_turns - 5, 70), "agora muda o objetivo sem avisar e vê se você percebe"))
    elif cycle == 3:
        plan.append(("irony", min(total_turns - 5, 55), "claro... sqn"))
    elif cycle == 4:
        plan.append(("late_reference_far", min(total_turns - 5, 110), "isso que eu falei lá atrás continua valendo"))
    elif cycle == 5:
        plan.append(("minimalist_trap", min(total_turns - 5, 60), "ok"))
    plan.sort(key=lambda t: (int(t[1]), str(t[0])))
    return [{"kind": str(k), "pos": int(pos), "text": str(txt)} for k, pos, txt in plan]


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


def _pick_top_conversations(counts: Dict[str, Dict[str, int]], *, min_user_turns: int, want: int) -> List[str]:
    convs: List[Tuple[int, int, str]] = []
    for cid, c in counts.items():
        ut = int(c.get("user_turns_total") or 0)
        ct = int(c.get("compat_turns_total") or 0)
        if ut < int(min_user_turns):
            continue
        convs.append((-ct, -ut, str(cid)))
    convs.sort()
    out: List[str] = []
    for _ct, _ut, cid in convs:
        out.append(str(cid))
        if len(out) >= int(want):
            break
    return list(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world_manifest", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks_total", type=int, default=20)
    args = ap.parse_args()

    seed = int(args.seed)
    out_path = Path(str(args.out))
    _ensure_absent(out_path)

    manifest_path = Path(str(args.world_manifest)).expanduser().resolve()
    if not manifest_path.exists():
        _fail(f"missing_world_manifest:{manifest_path}")
    canon_path, m = _world_dialogue_jsonl_from_manifest(manifest_path)

    tasks_total = int(args.tasks_total)
    if tasks_total < 20:
        _fail("tasks_total_too_small")

    world_fields = {
        "world_manifest": str(manifest_path.as_posix()),
        "world_manifest_sha256": _sha256_file(manifest_path),
        "world_canonical_jsonl": str(canon_path.as_posix()),
        "world_canonical_sha256": _sha256_file(canon_path),
    }

    # Pass 1: counts only (low memory).
    counts: Dict[str, Dict[str, int]] = {}
    with open(canon_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = str(obj.get("conversation_id") or "")
            if not cid:
                continue
            role = str(obj.get("role") or "unknown")
            txt = str(obj.get("text") or "")
            c = counts.get(cid)
            if c is None:
                c = {"user_turns_total": 0, "compat_turns_total": 0}
                counts[cid] = c
            if role == "user" and _is_safe_user_turn_text_v122(txt):
                c["user_turns_total"] = int(c.get("user_turns_total") or 0) + 1
                if _is_compatible_user_turn_text_v122(txt):
                    c["compat_turns_total"] = int(c.get("compat_turns_total") or 0) + 1

    chosen_conv_ids = _pick_top_conversations(counts, min_user_turns=12, want=max(60, tasks_total))
    if not chosen_conv_ids:
        _fail("no_conversations_with_min_turns")
    chosen_set = set(chosen_conv_ids)

    # Pass 2: collect turns for chosen conversations only (bounded per conv).
    conv_turns: Dict[str, List[str]] = {cid: [] for cid in chosen_conv_ids}
    with open(canon_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = str(obj.get("conversation_id") or "")
            if cid not in chosen_set:
                continue
            role = str(obj.get("role") or "unknown")
            if role != "user":
                continue
            txt = str(obj.get("text") or "")
            if not _is_safe_user_turn_text_v122(txt):
                continue
            buf = conv_turns.get(cid)
            if buf is None:
                continue
            if len(buf) < 800:
                buf.append(str(txt))

    tasks: List[Dict[str, Any]] = []
    used_conv = set()

    def _pick_conv() -> Optional[str]:
        for cid in chosen_conv_ids:
            if cid in used_conv:
                continue
            if len(conv_turns.get(cid, [])) < 8:
                continue
            used_conv.add(cid)
            return cid
        return None

    while len(tasks) < tasks_total:
        cid = _pick_conv()
        if cid is None:
            # Reuse from the top deterministically if needed.
            cid = chosen_conv_ids[int(len(tasks)) % len(chosen_conv_ids)]
        turns = conv_turns.get(str(cid), [])
        if len(turns) < 8:
            continue
        # Window length deterministic from seed+task_index (50..150).
        wlen = 50 + ((int(seed) + int(len(tasks)) * 17) % 101)
        # Deterministic slice start.
        sample_n = min(30, len(turns))
        start_max = max(1, len(turns) - sample_n + 1)
        start = (int(seed) + int(len(tasks)) * 13) % int(start_max)
        base_turns = turns[int(start) : int(start) + int(sample_n)]

        plan = _injection_plan_for_task_v122(task_index=len(tasks), total_turns=int(wlen))
        goal_turn = "goal: family7_v122 outcome=complete constraints=deterministic deadline={d}".format(d=int(wlen))
        user_turns = [goal_turn] + list(base_turns)
        user_turns = _apply_injection_plan(user_turns, plan)
        while len(user_turns) < int(wlen):
            user_turns.append("ok")

        tasks.append(
            _make_task(
                "family7_dla_task_v122",
                {
                    "seed": int(seed),
                    "conversation_id": str(cid),
                    "stress_kind": "MID",
                    "stress_turns": int(wlen),
                    "user_turns": list(user_turns[: int(wlen)]),
                    "require_fluency": True,
                    "allow_external_world_once": False,
                    "injection_plan": list(plan),
                    **dict(world_fields),
                },
            )
        )

    tasks.sort(key=lambda d: (str(d.get("task_id") or "")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x", encoding="utf-8") as f:
        for t in tasks:
            f.write(canonical_json_dumps(t))
            f.write("\n")

    manifest = {
        "schema_version": 122,
        "kind": "family7_tasks_manifest_v122",
        "seed": int(seed),
        "tasks_total": int(len(tasks)),
        "world_manifest": str(manifest_path.as_posix()),
        "world_manifest_sha256": str(world_fields["world_manifest_sha256"]),
        "world_canonical_jsonl": str(canon_path.as_posix()),
        "world_canonical_sha256": str(world_fields["world_canonical_sha256"]),
        "tasks_sha256": _sha256_file(out_path),
    }
    man_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    _ensure_absent(man_path)
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

