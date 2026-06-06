#!/bin/bash
# Check YOLO training status and handle completion
SERVER="cse12210210@172.18.34.26"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -p 10022"
PASS="rb6/aYMRAT#16"
LOG_DIR="/Users/robinli/Project/CV_Project/results"

yolo_done=false

# Check if YOLO job 88941 is still running
status=$(sshpass -p "$PASS" ssh $SSH_OPTS "$SERVER" \
  "scontrol show job 88941 2>/dev/null | grep JobState | awk -F'=' '{print \$2}'" 2>/dev/null)

# Check latest metrics from training log
echo "=== $(date '+%H:%M:%S') YOLO Training Status ==="
if [ "$status" = "RUNNING" ]; then
    echo "Job 88941: RUNNING"
    # Get latest training metrics
    latest=$(sshpass -p "$PASS" ssh $SSH_OPTS "$SERVER" \
      "tail -50 ~/cv_project/logs/88941_yolow.out 2>/dev/null | grep -E '^\s+[0-9]+/' | tail -3" 2>/dev/null)
    if [ -n "$latest" ]; then
        echo "$latest"
    else
        echo "(training in progress, no metrics yet)"
    fi
elif [ "$status" = "COMPLETED" ]; then
    echo "Job 88941: COMPLETED"
    yolo_done=true
elif [ "$status" = "FAILED" ]; then
    echo "Job 88941: FAILED"
    yolo_done=true
else
    echo "Job 88941: $status"
fi

# If YOLO done, backup results
if $yolo_done; then
    echo ""
    echo "=== Backing up YOLO results ==="
    EXP_DIR="$LOG_DIR/yolow_baseline_$(date +%Y%m%d_%H%M)"
    mkdir -p "$EXP_DIR"
    sshpass -p "$PASS" scp -o StrictHostKeyChecking=no -rP 10022 \
      "$SERVER:~/cv_project/logs/88941_yolow.out" \
      "$SERVER:~/cv_project/logs/88941_yolow.err" \
      "$SERVER:~/cv_project/runs/detect/results/" \
      "$EXP_DIR/" 2>/dev/null
    echo "Results saved to $EXP_DIR"
    
    # Start GDINO baseline
    echo ""
    echo "=== Starting GDINO Phase 1 ==="
    sshpass -p "$PASS" ssh $SSH_OPTS "$SERVER" \
      "cd ~/cv_project && NAME=gdino_baseline sbatch scripts/sbatch_gdino.sh" 2>/dev/null
    echo "GDINO baseline submitted"
fi
