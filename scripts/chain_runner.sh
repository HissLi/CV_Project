#!/bin/bash
QUEUE="$HOME/cv_project/experiment_queue.txt"
STATE="$HOME/cv_project/.chain_state"
LOG="$HOME/cv_project/logs"
SCRIPTS="$HOME/cv_project/scripts"

if [ -f "$STATE" ]; then
    read JOB_ID NAME LR_VAL EP_VAL < "$STATE"
else
    JOB_ID=""
fi

if [ -n "$JOB_ID" ]; then
    STATUS=$(scontrol show job $JOB_ID 2>/dev/null | grep JobState | cut -d= -f2 | cut -d' ' -f1)
    if [ "$STATUS" = "RUNNING" ] || [ "$STATUS" = "PENDING" ] || [ "$STATUS" = "COMPLETING" ]; then
        RT=$(scontrol show job $JOB_ID 2>/dev/null | grep RunTime | awk -F= '{print $2}')
        echo "$(date +%H:%M) $NAME: $STATUS (runtime $RT)"
        tail -1 "$LOG/${JOB_ID}_yolow.out" 2>/dev/null | grep -oP '\d+/\d+.*' | head -1
        exit 0
    fi
    if [ "$STATUS" = "COMPLETED" ]; then
        RT=$(scontrol show job $JOB_ID 2>/dev/null | grep RunTime | awk -F= '{print $2}')
        echo "$(date +%H:%M) $NAME: COMPLETED (runtime $RT)"
    else
        echo "$(date +%H:%M) $NAME: $STATUS - advancing to next"
    fi
else
    echo "No current job - checking queue"
fi

NEXT=$(grep -v '^#' "$QUEUE" | head -1)
if [ -z "$NEXT" ]; then
    echo "Queue empty. All experiments done!"
    rm -f "$STATE"
    exit 0
fi

grep -v "$NEXT" "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"

cd ~/cv_project
JOB_ID=$(eval $NEXT sbatch --parsable "$SCRIPTS/sbatch_yolow.sh" 2>/dev/null)
echo "$(date +%H:%M) Submitted $NEXT -> JOB $JOB_ID"
eval $NEXT
echo "$JOB_ID $name $lr $epochs" > "$STATE"
