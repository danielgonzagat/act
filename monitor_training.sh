#!/bin/bash
# monitor_training.sh - Monitor training progress

LOG_FILE="/workspaces/act/training_aggressive.log"

echo "=== TRAINING MONITOR ==="
echo "Time: $(date)"
echo ""

# Check if process is running
PROC_COUNT=$(pgrep -c -f "full_training" 2>/dev/null || echo "0")
echo "Active processes: $PROC_COUNT"
echo ""

# Show log tail
echo "=== LOG OUTPUT ==="
if [ -f "$LOG_FILE" ]; then
    cat "$LOG_FILE"
else
    echo "Log file not found"
fi
