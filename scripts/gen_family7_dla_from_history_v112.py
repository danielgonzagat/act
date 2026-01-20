#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_u64_le_list(path: str) -> List[int]:
    data = open(path, "rb").read()
    if len(data) % 8 != 0:
        _fail("offsets_size_not_multiple_of_8")
    out: List[int] = []
    for i in range(0, len(data), 8):
        out.append(int(struct.unpack("<Q", data[i : i + 8])[0]))
    return out


def _world_paths_from_manifest(manifest_path: str) -> Dict[str, str]:
    mp = str(manifest_path)
    with open(mp, "r", encoding="utf-8") as f:
        m = json.load(f)
    world_root = os.path.dirname(os.path.dirname(os.path.abspath(mp)))
    paths = m.get("paths") if isinstance(m.get("paths"), dict) else {}
    out: Dict[str, str] = {}
    for k in ["canonical_jsonl", "offsets_bin", "conversations_index_json"]:
        rel = str(paths.get(k) or "")
        if not rel:
            _fail(f"manifest_missing_path:{k}")
        out[k] = os.path.normpath(os.path.join(world_root, rel))
    out["world_root"] = world_root
    out["manifest_sha256"] = _sha256_file(mp)
    out["manifest_path"] = mp
    return out


def _fetch_user_turn_texts(
    *,
    canon_path: str,
    offsets: List[int],
    start_turn: int,
    end_turn: int,
    max_user_turns: int,
) -> List[str]:
    out: List[str] = []
    with open(canon_path, "rb") as f:
        for idx in range(int(start_turn), int(end_turn) + 1):
            if idx < 0 or idx >= len(offsets):
                break
            f.seek(int(offsets[idx]))
            line = f.readline()
            try:
                obj = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            if str(obj.get("role") or "") != "user":
                continue
            txt = str(obj.get("text") or "")
            if txt:
                out.append(txt)
                if len(out) >= int(max_user_turns):
                    break
    return list(out)


def _is_safe_user_turn_text_v112(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    # Avoid very long/structured/UI artifacts (keep "chaos" but not UI payloads).
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
    body["schema_version"] = 112
    body["task_kind"] = str(task_kind)
    task_id = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return dict(body, task_id=f"family7_dla_v112_{task_id}")


def _injection_plan_for_task_v112(*, task_index: int, total_turns: int) -> List[Dict[str, Any]]:
    """
    Deterministic adversarial injections.
    We keep the "chaos" but do not depend on NLP/LLMs.
    """
    plan: List[Tuple[str, int, str]] = []
    # Always inject "don't invent" somewhere.
    plan.append(("dont_invent", 12, "não invente; se não souber diga não sei e pergunte o dado"))
    # Cycle additional injections by task index.
    cycle = int(task_index) % 6
    if cycle == 0:
        plan.append(("late_reference", min(total_turns - 5, 60), "como eu disse antes, faz isso do mesmo jeito"))
    elif cycle == 1:
        plan.append(("soft_contradiction", min(total_turns - 5, 35), "na verdade era X, não Y"))
    elif cycle == 2:
        plan.append(("implicit_goal_shift", min(total_turns - 5, 45), "agora muda o objetivo sem avisar e vê se você percebe"))
    elif cycle == 3:
        plan.append(("irony", min(total_turns - 5, 28), "claro... sqn"))
    elif cycle == 4:
        plan.append(("hostile_confused", min(total_turns - 5, 22), "?? você não entendeu nada"))
    elif cycle == 5:
        plan.append(("minimalist_trap", min(total_turns - 5, 40), "ok"))
    # Stable order by position then kind.
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
    args = ap.parse_args()

    seed = int(args.seed)
    out_path = str(args.out)
    if os.path.exists(out_path):
        _fail(f"worm_exists:{out_path}")

    paths = _world_paths_from_manifest(str(args.world_manifest))
    canon_path = paths["canonical_jsonl"]
    offsets_path = paths["offsets_bin"]
    conv_index_path = paths["conversations_index_json"]
    if not (os.path.exists(canon_path) and os.path.exists(offsets_path) and os.path.exists(conv_index_path)):
        _fail("world_paths_missing")

    offsets = _read_u64_le_list(offsets_path)

    with open(conv_index_path, "r", encoding="utf-8") as f:
        conv_index = json.load(f)
    convs = conv_index.get("conversations") if isinstance(conv_index.get("conversations"), list) else []
    if not convs:
        _fail("empty_conversations_index")

    tasks_total = int(args.tasks_total)
    stress_200 = int(args.stress_200)
    if tasks_total < 4:
        _fail("tasks_total_too_small")
    if stress_200 < 2:
        _fail("stress_200_too_small")

    # Deterministic conversation candidates by size DESC, then conversation_id ASC.
    cands: List[Dict[str, Any]] = []
    for c in convs:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("conversation_id") or "")
        if not cid:
            continue
        turns_total = int(c.get("turns_total") or 0)
        if turns_total < 400:
            continue
        cands.append(dict(c))
    cands.sort(key=lambda d: (-int(d.get("turns_total") or 0), str(d.get("conversation_id") or "")))
    if len(cands) < tasks_total:
        _fail("not_enough_large_conversations")

    tasks: List[Dict[str, Any]] = []
    stress_built = 0
    i = 0
    for c in cands:
        if len(tasks) >= tasks_total:
            break
        cid = str(c.get("conversation_id") or "")
        s = int(c.get("start_turn") or 0)
        e = int(c.get("end_turn") or 0)

        # Pull a small but real sample of user turns to preserve chaos source.
        user_turns = _fetch_user_turn_texts(canon_path=canon_path, offsets=offsets, start_turn=s, end_turn=e, max_user_turns=120)
        safe_turns = [t for t in user_turns if _is_safe_user_turn_text_v112(t)]
        if len(safe_turns) < 8:
            continue

        is_stress = bool(stress_built < stress_200)
        minimal_n = 190 if is_stress else 80
        real_sample = safe_turns[:10] if is_stress else safe_turns[:8]

        goal_turn = "goal: family7_v112 outcome=complete constraints=deterministic deadline={n}".format(n=200 if is_stress else 100)
        minimal = ["ok"] * int(minimal_n)

        base_turns = [goal_turn] + list(real_sample) + list(minimal)
        plan = _injection_plan_for_task_v112(task_index=i, total_turns=len(base_turns))
        turns = _apply_injection_plan(base_turns, plan)

        allow_external = bool(len(tasks) == 0)  # exactly one task probes external world usage in-run
        task = _make_task(
            "family7_dla_task_v112",
            {
                "seed": int(seed),
                "world_manifest": str(paths["manifest_path"]),
                "conversation_id": cid,
                "window": {"start_turn": int(s), "end_turn": int(e)},
                "stress_kind": "STRESS_200" if is_stress else "SMOKE_80",
                "user_turns": list(turns),
                "real_user_sample_turns": int(len(real_sample)),
                "minimal_ok_turns": int(len(minimal)),
                "injection_plan": list(plan),
                "expected_validators": [
                    "fluency_survival_v112",
                    "binding_unresolved_reference_zero",
                    "semantic_contradiction_zero",
                ],
                "allow_external_world_once": bool(allow_external),
                "external_world_probe_reason_code": "validator_failed_fluency_contract",
            },
        )
        tasks.append(task)
        if is_stress:
            stress_built += 1
        i += 1

    if len(tasks) != tasks_total:
        _fail("failed_to_build_tasks_total")
    if stress_built != stress_200:
        _fail("failed_to_build_stress_200_total")

    # Write tasks jsonl (WORM).
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "x", encoding="utf-8") as f:
        for t in tasks:
            f.write(canonical_json_dumps(t))
            f.write("\n")

    manifest_out = os.path.splitext(out_path)[0] + "_manifest.json"
    if os.path.exists(manifest_out):
        _fail(f"worm_exists:{manifest_out}")

    tasks_sha = _sha256_file(out_path)
    manifest = {
        "schema_version": 112,
        "kind": "family7_dla_tasks_v112",
        "seed": int(seed),
        "paths": {"tasks_jsonl": out_path, "world_manifest": str(paths["manifest_path"])},
        "sha256": {"tasks_jsonl": tasks_sha, "world_manifest": str(paths["manifest_sha256"])},
        "tasks_total": int(len(tasks)),
        "stress_200_total": int(stress_200),
    }
    with open(manifest_out, "x", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")

    print(
        json.dumps(
            {"ok": True, "tasks": out_path, "tasks_sha256": tasks_sha, "manifest": manifest_out, "tasks_total": len(tasks)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
