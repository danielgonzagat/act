#!/usr/bin/env python3
"""
analyze_iteration.py - Analisador de Iteração para Dominância de Hierarquia

Este script analisa os resultados de uma iteração de treinamento
SEM interromper o processo em execução.

Uso:
    python -m atos_core.analyze_iteration [--watch]

Com --watch, fica monitorando o log e analisa automaticamente
quando uma iteração termina.

Schema version: 159
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .hierarchy_dominance_metrics_v159 import (
    HierarchyDominanceTracker,
    HierarchyDominanceReport,
    save_dominance_report,
)


def parse_training_log(log_path: str) -> Dict[str, Any]:
    """
    Parse training log para extrair informações de iteração.
    
    Returns:
        Dict com iteration_number, tasks_solved, accuracy, etc.
    """
    if not os.path.exists(log_path):
        return {"error": f"Log file not found: {log_path}"}
    
    with open(log_path) as f:
        content = f.read()
    
    result = {
        "iterations_found": [],
        "current_iteration": 0,
        "is_running": False,
        "last_accuracy": 0.0,
        "failures_by_reason": {},
    }
    
    # Parse iterations
    iteration_pattern = r"ITERATION\s+(\d+)/(\d+)"
    for match in re.finditer(iteration_pattern, content):
        iter_num = int(match.group(1))
        total = int(match.group(2))
        result["iterations_found"].append(iter_num)
        result["current_iteration"] = iter_num
        result["total_iterations"] = total
    
    # Parse accuracy
    accuracy_pattern = r"Solved:\s+(\d+)/(\d+)\s+\((\d+\.?\d*)%\)"
    accuracies = re.findall(accuracy_pattern, content)
    if accuracies:
        last = accuracies[-1]
        result["last_solved"] = int(last[0])
        result["last_total"] = int(last[1])
        result["last_accuracy"] = float(last[2]) / 100
    
    # Parse failures
    failures_pattern = r"Failures:\s+\{([^}]+)\}"
    failures_match = re.findall(failures_pattern, content)
    if failures_match:
        last_failures = failures_match[-1]
        # Parse dict-like string
        pairs = re.findall(r"'(\w+)':\s*(\d+)", last_failures)
        result["failures_by_reason"] = {k: int(v) for k, v in pairs}
    
    # Check if still running
    result["is_running"] = "ITERATION" in content and "FINAL EVALUATION REPORT" not in content
    
    return result


def check_training_status() -> Dict[str, Any]:
    """Check current training status."""
    import subprocess
    
    # Check if process is running
    result = subprocess.run(
        "pgrep -f 'full_training_pipeline_v158' | wc -l",
        shell=True, capture_output=True, text=True
    )
    process_count = int(result.stdout.strip())
    
    # Check log file
    log_path = "/workspaces/act/training_aggressive.log"
    log_info = parse_training_log(log_path)
    
    # Get runtime
    result = subprocess.run(
        "ps -eo etime,cmd | grep full_training_pipeline_v158 | grep -v grep | head -1",
        shell=True, capture_output=True, text=True
    )
    runtime = result.stdout.strip().split()[0] if result.stdout.strip() else "N/A"
    
    return {
        "process_count": process_count,
        "runtime": runtime,
        "log_info": log_info,
        "timestamp": datetime.now().isoformat(),
    }


def wait_for_iteration_complete(log_path: str, target_iteration: int, timeout: int = 3600) -> bool:
    """
    Aguarda até que uma iteração específica termine.
    
    Returns:
        True se iteração terminou, False se timeout
    """
    start = time.time()
    last_iteration = 0
    
    while time.time() - start < timeout:
        info = parse_training_log(log_path)
        current = info.get("current_iteration", 0)
        
        if current > last_iteration:
            print(f"   Iteration {current} in progress...")
            last_iteration = current
        
        if current >= target_iteration and len(info.get("iterations_found", [])) > target_iteration:
            # A próxima iteração já começou, então a anterior terminou
            return True
        
        if "FINAL EVALUATION REPORT" in open(log_path).read() if os.path.exists(log_path) else "":
            return True
        
        time.sleep(30)  # Check every 30 seconds
    
    return False


def analyze_current_state() -> None:
    """Analisa o estado atual do treinamento."""
    
    print("\n" + "=" * 70)
    print("TRAINING STATUS ANALYSIS")
    print("=" * 70)
    
    status = check_training_status()
    
    print(f"\n⏱️  RUNTIME: {status['runtime']}")
    print(f"🔧 ACTIVE PROCESSES: {status['process_count']}")
    
    log_info = status["log_info"]
    
    if "error" in log_info:
        print(f"\n❌ {log_info['error']}")
        return
    
    print(f"\n📊 ITERATION: {log_info.get('current_iteration', 0)}/{log_info.get('total_iterations', '?')}")
    
    if log_info.get("last_accuracy"):
        print(f"📈 LAST ACCURACY: {log_info['last_accuracy']*100:.1f}%")
        print(f"   Solved: {log_info.get('last_solved', 0)}/{log_info.get('last_total', 0)}")
    
    if log_info.get("failures_by_reason"):
        print(f"\n❌ FAILURES BY REASON:")
        total_failures = sum(log_info["failures_by_reason"].values())
        for reason, count in sorted(log_info["failures_by_reason"].items(), key=lambda x: -x[1]):
            pct = count / max(1, total_failures) * 100
            print(f"   • {reason}: {count} ({pct:.0f}%)")
        
        # Destaque SEARCH_BUDGET_EXCEEDED
        budget_exceeded = log_info["failures_by_reason"].get("SEARCH_BUDGET_EXCEEDED", 0)
        if budget_exceeded > total_failures * 0.5:
            print(f"\n⚠️  ALERT: SEARCH_BUDGET_EXCEEDED is {budget_exceeded/total_failures*100:.0f}% of failures")
            print("   → Hierarchy not yet reducing search space")
    
    if log_info.get("is_running"):
        print(f"\n🟢 STATUS: TRAINING IN PROGRESS")
    else:
        print(f"\n🔴 STATUS: TRAINING COMPLETED OR STOPPED")


def generate_dominance_report_stub() -> None:
    """
    Gera um relatório stub de dominância baseado no log atual.
    
    Nota: Sem acesso aos program_steps reais, só podemos
    analisar métricas agregadas do log.
    """
    
    print("\n" + "=" * 70)
    print("HIERARCHY DOMINANCE ANALYSIS (from log)")
    print("=" * 70)
    
    log_path = "/workspaces/act/training_aggressive.log"
    
    if not os.path.exists(log_path):
        print("❌ Training log not found")
        return
    
    info = parse_training_log(log_path)
    
    # Análise baseada em falhas
    failures = info.get("failures_by_reason", {})
    total_failures = sum(failures.values())
    
    if not total_failures:
        print("⏳ No iteration data yet - training still initializing")
        return
    
    budget_exceeded = failures.get("SEARCH_BUDGET_EXCEEDED", 0)
    budget_pct = budget_exceeded / total_failures * 100 if total_failures else 0
    
    print(f"\n📊 FAILURE ANALYSIS (Iteration {info.get('current_iteration', '?')})")
    print(f"   Total failures: {total_failures}")
    print(f"   SEARCH_BUDGET_EXCEEDED: {budget_exceeded} ({budget_pct:.0f}%)")
    
    print(f"\n🎯 HIERARCHY DOMINANCE INDICATORS:")
    
    # Indicadores de dominância
    if budget_pct > 80:
        print("   ❌ HIERARCHY NOT ACTIVE: 80%+ budget exhaustion")
        print("      → Search is exploring primitively, not using concepts")
        print("      → Expected at start, needs to decrease over iterations")
    elif budget_pct > 50:
        print("   🔶 HIERARCHY EMERGING: 50-80% budget exhaustion")
        print("      → Some concept guidance, but not dominant")
    else:
        print("   ✅ HIERARCHY GAINING GROUND: <50% budget exhaustion")
        print("      → Concepts are guiding search effectively")
    
    # Comparação com iterações anteriores
    print(f"\n📈 PROGRESS INDICATORS:")
    accuracy = info.get("last_accuracy", 0)
    print(f"   Current accuracy: {accuracy*100:.1f}%")
    
    if accuracy < 0.10:
        print("   ⚠️  Below 10% - system still learning basic patterns")
    elif accuracy < 0.30:
        print("   🔶 10-30% - some patterns emerging")
    else:
        print("   ✅ >30% - significant concept emergence expected")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS (per PHASE DE VIE DU SYSTÈME):")
    print("=" * 70)
    print("1. ⏳ Let iteration complete - DO NOT interrupt")
    print("2. 📊 After completion, analyze concept reuse patterns")
    print("3. 👀 Look for concepts appearing in 2+ tasks")
    print("4. 📉 Track if budget_exceeded% decreases across iterations")
    print("5. 🚫 NO optimization until hierarchy shows activity")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze training iteration")
    parser.add_argument("--watch", action="store_true", help="Watch mode - continuous monitoring")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in seconds")
    
    args = parser.parse_args()
    
    if args.watch:
        print("🔍 WATCH MODE - Monitoring training progress")
        print("   Press Ctrl+C to stop")
        print()
        
        try:
            while True:
                analyze_current_state()
                generate_dominance_report_stub()
                
                print(f"\n⏰ Next check in {args.interval} seconds...")
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Watch mode stopped")
    else:
        analyze_current_state()
        generate_dominance_report_stub()


if __name__ == "__main__":
    main()
