#!/usr/bin/env bash
# Wait for stage 2, then run stages 3-6 unattended.
#
# Liveness is judged from results/heartbeat.json, which stage 2 fsyncs before
# each trajectory. "Is any python running" is not a usable check -- it matched
# unrelated interpreters and caused two stage-2 instances to compete for the
# GPU. A heartbeat older than STALE_S with the manifest still incomplete means
# the run really did die (the expected cause being a host reset), and only then
# is it resumed.
set -u
PY=".venv/Scripts/python.exe"
MAN="results/scores/primary/manifest.json"
HB="results/heartbeat.json"
STALE_S=600

complete() { [ -f "$MAN" ] && grep -q '"status": "complete"' "$MAN"; }

echo "waiting for stage 2 $(date +%H:%M:%S)"
while ! complete; do
  sleep 60
  complete && break
  if [ -f "$HB" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$HB") ))
  else
    age=$STALE_S
  fi
  if [ "$age" -ge "$STALE_S" ]; then
    echo "heartbeat stale (${age}s) and stage 2 incomplete -- resuming $(date +%H:%M:%S)"
    "$PY" -u scripts/run_all.py --stage 2 --pass-gates >> results/stage2.out 2>&1
  fi
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
