#!/usr/bin/env bash
# Chain stages 2-6 unattended. Each stage is resumable from its manifest, so a
# host reset (CRASH_MITIGATION.md) costs one trajectory, not the run: this
# retries a failed stage twice before giving up, since the most likely failure
# mode here is an abrupt reboot rather than a logic error.
set -u
PY=".venv/Scripts/python.exe"
for stage in 2 3 4 5 6; do
  ok=0
  for attempt in 1 2 3; do
    echo "=== stage $stage (attempt $attempt) $(date +%H:%M:%S) ==="
    "$PY" scripts/run_all.py --stage "$stage" --pass-gates && { ok=1; break; }
    echo "!!! stage $stage attempt $attempt failed; resuming from manifest"
    sleep 20
  done
  if [ "$ok" -ne 1 ]; then
    echo "!!! stage $stage failed 3 times -- stopping"
    exit 1
  fi
done
echo "=== ALL STAGES COMPLETE $(date +%H:%M:%S) ==="
