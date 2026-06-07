#!/bin/bash
QUEUE="$HOME/cv_project/experiment_queue.txt"
STATE="$HOME/cv_project/.chain_state"
LOG="$HOME/cv_project/logs"
SCRIPTS="$HOME/cv_project/scripts/phase2_lr_sweep"

if [ -f "$STATE" ]; then
    read JOB_ID NAME LR_VAL EP_VAL TASK_NUM TOTAL_COUNT < "$STATE"
else
    JOB_ID=""
fi

if [ -n "$JOB_ID" ]; then
    STATUS=$(scontrol show job $JOB_ID 2>/dev/null | grep JobState | cut -d= -f2 | cut -d' ' -f1)
    if [ "$STATUS" = "RUNNING" ] || [ "$STATUS" = "PENDING" ] || [ "$STATUS" = "COMPLETING" ]; then
        RT=$(scontrol show job $JOB_ID 2>/dev/null | grep RunTime | awk -F= '{print $2}' | cut -d' ' -f1)
        # Get latest training progress line
        PROGRESS=$(tail -1 "$LOG/${JOB_ID}_yolow.out" 2>/dev/null | grep -oP '\d+/\d+\s+\S+\s+\S+\s+\S+\s+\S+\s+\d+\s+\d+:\s+\d+%[\s\S]*?\d+/\d+\s+\S+it/s' | tail -1)
        if [ -n "$PROGRESS" ]; then
            # Parse: epoch_current/epoch_total GPU box_loss cls_loss dfl_loss instances size: pct% ... batch/total it/s time_elapsed<time_remaining
            CUR_EP=$(echo "$PROGRESS" | grep -oP '^\s*\d+' | head -1)
            TOT_EP="$EP_VAL"
            PCT=$(echo "$PROGRESS" | grep -oP '\d+%' | head -1 | tr -d '%')
            SPEED=$(echo "$PROGRESS" | grep -oP '\d+\.\d+it/s' | tail -1)
            BATCH_INFO=$(echo "$PROGRESS" | grep -oP '\d+/\d+' | tail -1)
            # Estimate: ~25min per epoch (based on baseline observations)
            MIN_PER_EPOCH=25
            REMAINING_EPOCHS=$((TOT_EP - CUR_EP))
            REMAINING_MIN=$((REMAINING_EPOCHS * MIN_PER_EPOCH + (100 - PCT) * MIN_PER_EPOCH / 100))
            if [ "$REMAINING_MIN" -gt 60 ]; then
                ETA_STR="$((REMAINING_MIN / 60))h$((REMAINING_MIN % 60))m"
            else
                ETA_STR="${REMAINING_MIN}min"
            fi
            echo "[$(date '+%H:%M:%S')] Task $TASK_NUM/$TOTAL_COUNT — $NAME | Epoch $CUR_EP/$TOT_EP ${PCT}%"
        else
            echo "[$(date '+%H:%M:%S')] Task $TASK_NUM/$TOTAL_COUNT — $NAME: $STATUS"
            echo "  Runtime: $RT | Waiting for first batch..."
        fi
        exit 0
    fi
    if [ "$STATUS" = "COMPLETED" ]; then
        echo "[$(date '+%H:%M:%S')] Task $TASK_NUM/$TOTAL_COUNT — $NAME: COMPLETED (runtime: $(scontrol show job $JOB_ID 2>/dev/null | grep RunTime | awk -F= '{print $2}' | cut -d' ' -f1))"
    else
        echo "[$(date '+%H:%M:%S')] $NAME: $STATUS - advancing to next"
    fi
fi

# Count remaining experiments
QUEUE_REMAINING=$(grep -v '^#' "$QUEUE" 2>/dev/null | grep -c .)
TOTAL_REMAINING=$((QUEUE_REMAINING + 1))  # +1 for current job

NEXT=$(grep -v '^#' "$QUEUE" | head -1)
if [ -z "$NEXT" ]; then
    echo "Queue empty. All experiments done!"
    rm -f "$STATE"
    exit 0
fi

grep -v "$NEXT" "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"

# Calculate task position
TOTAL_DONE=$((TOTAL_COUNT - TOTAL_REMAINING + 1))
if [ -z "$TOTAL_COUNT" ]; then
    TOTAL_COUNT=$TOTAL_REMAINING
    TOTAL_DONE=1
fi

cd ~/cv_project
# Export uppercase variables (sbatch script uses LR, EPOCHS, NAME, BS)
eval $NEXT
export LR=$lr EPOCHS=$epochs NAME=$name BS=${bs:-8}
JOB_ID=$(sbatch --parsable "$SCRIPTS/sbatch_yolow.sh" 2>/dev/null)
echo "[$(date '+%H:%M:%S')] Submitted Task $TOTAL_DONE/$TOTAL_COUNT — $NEXT -> JOB $JOB_ID"
echo "$JOB_ID $name $lr $epochs $TOTAL_DONE $TOTAL_COUNT" > "$STATE"
