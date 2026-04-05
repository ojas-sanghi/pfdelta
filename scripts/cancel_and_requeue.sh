#!/bin/bash
# cancel_and_requeue.sh
# Cancels all pending jobs and resubmits their original .sh scripts

# Get all pending job IDs
PENDING=$(squeue -u osanghi -t PD -h -o "%i")

if [ -z "$PENDING" ]; then
    echo "No pending jobs found."
    exit 0
fi

echo "Found pending jobs:"
echo "$PENDING"
echo ""

# Collect script paths before cancelling
declare -a SCRIPTS

for JOBID in $PENDING; do
    SCRIPT=$(scontrol show job $JOBID | grep -oP '(?<=Command=)\S+')
    if [ -n "$SCRIPT" ]; then
        echo "Job $JOBID -> $SCRIPT"
        SCRIPTS+=("$SCRIPT")
    else
        echo "WARNING: Could not find script for job $JOBID"
    fi
done

echo ""
echo "Cancelling all pending jobs..."
scancel $PENDING
echo "Cancelled."

echo ""
echo "Resubmitting..."
for SCRIPT in "${SCRIPTS[@]}"; do
    echo "  sbatch $SCRIPT"
    sbatch "$SCRIPT"
done

echo ""
echo "Done. New queue:"
squeue -u osanghi