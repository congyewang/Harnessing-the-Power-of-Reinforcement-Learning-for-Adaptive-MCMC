#!/bin/bash

MAX_JOBS=10
MIN_JOBS=1
DECREMENT=2
MEM_FREE=10G
TIMEOUT=3600

while [[ $MAX_JOBS -ge $MIN_JOBS ]]; do
  echo "Running parallel with --jobs $MAX_JOBS"

  parallel --jobs "$MAX_JOBS" --memfree "$MEM_FREE" --timeout "$TIMEOUT" \
    --joblog joblog.txt --resume-failed <jobs.txt

  if grep -q "exitval [^0]" joblog.txt; then
    echo "Some jobs failed, reducing parallelism and retrying..."
    MAX_JOBS=$((MAX_JOBS - DECREMENT))
    if [[ $MAX_JOBS -lt $MIN_JOBS ]]; then
      MAX_JOBS=$MIN_JOBS
    fi
  else
    echo "All jobs finished successfully!"
    exit 0
  fi
done

echo "Jobs completed with reduced parallelism but some tasks may have failed."
exit 1
