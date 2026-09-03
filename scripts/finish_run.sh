#!/usr/bin/env bash
# Wait for stage 2 to complete, then run stages 3-6. Stage 2 is launched
# separately because it is the long pole; this picks up behind it so the whole
# pipeline finishes unattended.
set -u
PY=".venv/Scripts/python.exe"
MAN="results/scores/primary/manifest.json"
echo "waiting for stage 2 to complete..."
while true; do
  if [ -f "$MAN" ] && grep -q '"status": "complete"' "$MAN"; then break; fi
  # If stage 2 died without completing, retry it -- a host reset is the
  # expected failure mode here and the cache makes it cheap to resume.
  if ! tasklist 2>/dev/null | grep -qi "python.exe"; then
    echo "stage 2 not running and not complete; resuming $(date +%H:%M:%S)"
    "$PY" -u scripts/run_all.py --stage 2 --pass-gates >> results/stage2.out 2>&1
  fi
  sleep 60
done
echo "=== stage 2 complete $(date +%H:%M:%S) ==="
for stage in 3 4 5 6; do
  ok=0
  for attempt in 1 2 3; do
    echo "=== stage $stage (attempt $attempt) $(date +%H:%M:%S) ==="
    "$PY" -u scripts/run_all.py --stage "$stage" --pass-gates && { ok=1; break; }
    echo "!!! stage $stage attempt $attempt failed; resuming from manifest"
    sleep 20
  done
  [ "$ok" -eq 1 ] || { echo "!!! stage $stage failed 3 times -- stopping"; exit 1; }
done
echo "=== ALL STAGES COMPLETE $(date +%H:%M:%S) ==="
