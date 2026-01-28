#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_loop_outputs(lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    macro_bank: Optional[str] = None
    concept_bank: Optional[str] = None
    omega_state: Optional[str] = None
    summary: Optional[str] = None

    rx_bank = re.compile(r"^\[arc_loop\] next MACRO_BANK_IN(?: \([^)]+\))?: (.+)\s*$")
    rx_concept = re.compile(r"^\[arc_loop\] next CONCEPT_BANK_IN(?: \([^)]+\))?: (.+)\s*$")
    rx_omega = re.compile(r"^\[arc_loop\] next OMEGA_STATE_IN(?: \([^)]+\))?: (.+)\s*$")
    rx_sum_macro = re.compile(r"^\[arc_loop\] macro summary: (.+)\s*$")
    rx_sum_base = re.compile(r"^\[arc_loop\] summary: (.+)\s*$")

    for ln in lines:
        m = rx_bank.match(ln)
        if m:
            macro_bank = m.group(1).strip()
        m = rx_concept.match(ln)
        if m:
            concept_bank = m.group(1).strip()
        m = rx_omega.match(ln)
        if m:
            omega_state = m.group(1).strip()
        m = rx_sum_macro.match(ln)
        if m:
            summary = m.group(1).strip()
        m = rx_sum_base.match(ln)
        if m and summary is None:
            summary = m.group(1).strip()

    return macro_bank, concept_bank, omega_state, summary


def _get_kmax_from_summary(summary: Dict[str, Any]) -> int:
    by_k = summary.get("tasks_solved_by_k") if isinstance(summary.get("tasks_solved_by_k"), dict) else {}
    if not by_k:
        return 0
    ks: List[int] = []
    for k in by_k.keys():
        try:
            ks.append(int(k))
        except Exception:
            continue
    return max(ks) if ks else 0


def _write_jsonl_x(path: Path, rows: List[Dict[str, Any]]) -> None:
    if path.exists():
        raise SystemExit(f"worm_exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        for row in rows:
            f.write(_stable_json(row) + "\n")


def _concept_bank_stats(path: Path) -> Dict[str, Any]:
    total = 0
    multiop = 0
    multistage = 0
    max_support = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        total += 1
        op_ids = obj.get("op_ids") if isinstance(obj.get("op_ids"), list) else []
        op_len = len([str(x) for x in op_ids if str(x)])
        if op_len > 1:
            multiop += 1
        sig = obj.get("signature") if isinstance(obj.get("signature"), dict) else {}
        if str(sig.get("diff_kind") or "") == "MULTI_STAGE":
            multistage += 1
        try:
            max_support = max(int(max_support), int(obj.get("support") or 0))
        except Exception:
            pass
    return {
        "concepts_total": int(total),
        "concepts_multiop": int(multiop),
        "concepts_multistage": int(multistage),
        "concepts_max_support": int(max_support),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--iters", type=int, default=10, help="0 = run forever (until stop_rate_kmax or Ctrl-C)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--pressure", type=int, default=0)
    ap.add_argument("--deep", type=int, default=0)
    ap.add_argument("--tries", type=int, default=1)
    ap.add_argument("--task_timeout_s", type=int, default=0)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--max_programs", type=int, default=2000)
    ap.add_argument("--macro_bank_in", default="", help="Initial macro bank jsonl (optional)")
    ap.add_argument("--concept_bank_in", default="", help="Initial concept bank jsonl (optional)")
    ap.add_argument("--omega", type=int, default=0, help="Enable Ω (destructive ontological memory) in the loop.")
    ap.add_argument("--omega_state_in", default="", help="Initial omega_state_v1.json (optional, only used if --omega=1).")
    ap.add_argument("--stop_rate_kmax", type=float, default=0.0, help="Stop when solve_rate_kmax >= this value")
    ap.add_argument("--stop_hur", type=float, default=0.0, help="Stop when hierarchical_utilization_ratio >= this value")
    ap.add_argument(
        "--stop_multistep_solved",
        type=int,
        default=0,
        help="Stop when solved_with_multistep_concept_call >= this value (from summary.json)",
    )
    ap.add_argument("--out", default="", help="WORM jsonl output path (optional)")
    args = ap.parse_args()

    limit = int(args.limit)
    iters = int(args.iters)
    if iters < 0:
        raise SystemExit("bad_iters")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out or f"artifacts/arc_pt2_iter_v146_{ts}.jsonl").resolve()

    macro_bank_in = str(args.macro_bank_in or "").strip()
    if macro_bank_in and not Path(macro_bank_in).is_file():
        raise SystemExit(f"missing_macro_bank_in:{macro_bank_in}")

    concept_bank_in = str(args.concept_bank_in or "").strip()
    if concept_bank_in and not Path(concept_bank_in).is_file():
        raise SystemExit(f"missing_concept_bank_in:{concept_bank_in}")

    omega = int(args.omega)
    omega_state_in = str(args.omega_state_in or "").strip()
    if omega_state_in and not Path(omega_state_in).is_file():
        raise SystemExit(f"missing_omega_state_in:{omega_state_in}")

    if out_path.exists():
        raise SystemExit(f"worm_exists:{out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    stop_reason = "DONE"

    with open(out_path, "x", encoding="utf-8") as out_f:
        header = {
            "schema_version": 146,
            "kind": "arc_pt2_iter_header_v146",
            "created_at": str(ts),
            "limit": int(limit),
            "iters": int(iters),
            "jobs": int(args.jobs),
            "pressure": int(args.pressure),
            "deep": int(args.deep),
            "tries": int(args.tries),
            "task_timeout_s": int(args.task_timeout_s),
            "max_depth": int(args.max_depth),
            "max_programs": int(args.max_programs),
            "stop_rate_kmax": float(args.stop_rate_kmax),
            "macro_bank_in": str(macro_bank_in),
            "concept_bank_in": str(concept_bank_in),
            "omega": int(omega),
            "omega_state_in": str(omega_state_in),
        }
        out_f.write(_stable_json(header) + "\n")
        out_f.flush()
        rows_written += 1

        i = 0
        try:
            while True:
                if iters != 0 and i >= iters:
                    stop_reason = "DONE"
                    break

                env = os.environ.copy()
                env["JOBS"] = str(int(args.jobs))
                env["PRESSURE"] = "1" if int(args.pressure) != 0 else "0"
                env["DEEP"] = "1" if int(args.deep) != 0 else "0"
                env["TRIES"] = str(int(args.tries))
                env["TASK_TIMEOUT_S"] = str(int(args.task_timeout_s))
                env["MAX_DEPTH"] = str(int(args.max_depth))
                env["MAX_PROGRAMS"] = str(int(args.max_programs))
                env["OMEGA"] = "1" if int(omega) != 0 else "0"
                if macro_bank_in:
                    env["MACRO_BANK_IN"] = str(macro_bank_in)
                if concept_bank_in:
                    env["CONCEPT_BANK_IN"] = str(concept_bank_in)
                if int(omega) != 0 and omega_state_in:
                    env["OMEGA_STATE_IN"] = str(omega_state_in)

                cmd = ["bash", "scripts/arc_loop_pt2.sh", str(limit)]
                print(
                    _stable_json(
                        {
                            "iter": int(i + 1),
                            "cmd": cmd,
                            "macro_bank_in": macro_bank_in,
                            "concept_bank_in": concept_bank_in,
                            "omega": int(omega),
                            "omega_state_in": omega_state_in,
                        }
                    ),
                    file=sys.stderr,
                )
                started = time.monotonic()

                buf: List[str] = []
                p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert p.stdout is not None
                try:
                    for line in p.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        buf.append(line.rstrip("\n"))
                except KeyboardInterrupt:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                    try:
                        p.wait(timeout=3.0)
                    except Exception:
                        try:
                            p.kill()
                        except Exception:
                            pass
                    stop_reason = "INTERRUPTED"
                    break
                rc = p.wait()

                macro_bank_out, concept_bank_out, omega_state_out, summary_path_s = _parse_loop_outputs(buf)
                summary_path: Optional[Path] = Path(summary_path_s).resolve() if summary_path_s else None
                summary_obj: Dict[str, Any] = {}
                if summary_path is not None and summary_path.is_file():
                    sraw = _read_json(summary_path)
                    if isinstance(sraw, dict):
                        summary_obj = sraw

                tasks_total = int(summary_obj.get("tasks_total") or 0)
                solved_by_k = summary_obj.get("tasks_solved_by_k") if isinstance(summary_obj.get("tasks_solved_by_k"), dict) else {}
                kmax = _get_kmax_from_summary(summary_obj)
                solved_kmax = int(solved_by_k.get(str(int(kmax))) or 0) if kmax else 0
                rate_kmax = float(solved_kmax) / float(tasks_total) if tasks_total else 0.0
                pu = summary_obj.get("program_usage") if isinstance(summary_obj.get("program_usage"), dict) else {}
                hur = float(pu.get("hierarchical_utilization_ratio") or 0.0)
                multistep_solved = int(pu.get("solved_with_multistep_concept_call") or 0)

                cb_stats: Dict[str, Any] = {}
                cbp = Path(concept_bank_out).resolve() if concept_bank_out else None
                if cbp is not None and cbp.is_file():
                    try:
                        cb_stats = _concept_bank_stats(cbp)
                    except Exception:
                        cb_stats = {}

                row = {
                    "schema_version": 146,
                    "kind": "arc_pt2_iter_row_v146",
                    "iter": int(i + 1),
                    "limit": int(limit),
                    "env": {
                        "JOBS": env.get("JOBS", ""),
                        "PRESSURE": env.get("PRESSURE", ""),
                        "DEEP": env.get("DEEP", ""),
                        "TRIES": env.get("TRIES", ""),
                        "TASK_TIMEOUT_S": env.get("TASK_TIMEOUT_S", ""),
                    "MAX_DEPTH": env.get("MAX_DEPTH", ""),
                    "MAX_PROGRAMS": env.get("MAX_PROGRAMS", ""),
                    "OMEGA": env.get("OMEGA", ""),
                    "OMEGA_STATE_IN": env.get("OMEGA_STATE_IN", ""),
                },
                "returncode": int(rc),
                "wall_s": float(time.monotonic() - started),
                "summary_path": str(summary_path) if summary_path is not None else "",
                    "tasks_total": int(tasks_total),
                    "kmax": int(kmax),
                    "tasks_solved_kmax": int(solved_kmax),
                    "solve_rate_kmax": float(rate_kmax),
                    "hur": float(hur),
                    "multistep_solved": int(multistep_solved),
                    "concept_bank_stats": dict(cb_stats),
                    "macro_bank_in": str(macro_bank_in),
                    "macro_bank_out": str(macro_bank_out or ""),
                    "concept_bank_in": str(concept_bank_in),
                    "concept_bank_out": str(concept_bank_out or ""),
                    "omega_state_in": str(omega_state_in),
                    "omega_state_out": str(omega_state_out or ""),
                }
                out_f.write(_stable_json(row) + "\n")
                out_f.flush()
                rows_written += 1

                if rc != 0:
                    stop_reason = "SUBPROCESS_ERROR"
                    break

                if macro_bank_out:
                    macro_bank_in = str(macro_bank_out)
                if concept_bank_out:
                    concept_bank_in = str(concept_bank_out)
                if int(omega) != 0 and omega_state_out:
                    omega_state_in = str(omega_state_out)

                if float(args.stop_rate_kmax) > 0.0 and rate_kmax >= float(args.stop_rate_kmax):
                    stop_reason = "STOP_RATE_REACHED"
                    break
                if float(args.stop_hur) > 0.0 and hur >= float(args.stop_hur):
                    stop_reason = "STOP_HUR_REACHED"
                    break
                if int(args.stop_multistep_solved) > 0 and int(multistep_solved) >= int(args.stop_multistep_solved):
                    stop_reason = "STOP_MULTISTEP_REACHED"
                    break

                i += 1
        except KeyboardInterrupt:
            stop_reason = "INTERRUPTED"

        footer = {"schema_version": 146, "kind": "arc_pt2_iter_footer_v146", "rows_written": int(rows_written), "stop_reason": str(stop_reason)}
        out_f.write(_stable_json(footer) + "\n")
        out_f.flush()
        rows_written += 1

    print(_stable_json({"ok": True, "out": str(out_path), "rows_written": int(rows_written), "stop_reason": str(stop_reason)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
