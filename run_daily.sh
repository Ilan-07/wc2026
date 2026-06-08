#!/usr/bin/env bash
# WC2026 live-forecast tick (launchd, every ~30 min). `predict --if-changed` re-fits + regenerates
# the dashboard ONLY when a tracked input changed (new match result or a fresh odds snapshot);
# otherwise it's a cheap no-op heartbeat — a free results pull + a hash compare. The dashboard HTML
# carries an auto-reload tag, so an open browser tab stays live without a manual refresh.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
ts=$(date +%Y-%m-%d_%H%M%S)
# launchd runs with a minimal PATH, so resolve a Python that actually has the deps (numpy/pymc/...).
# Prefer the framework interpreter this project was built with; fall back to whatever python3 is on PATH.
PY="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3 || echo python3)"

tmp="logs/.tick_${ts}.log"
# caffeinate -i keeps the Mac awake for the duration so a re-fit/publish can't be cut off by sleep.
PYTHONPATH=src caffeinate -i "$PY" cli.py predict --if-changed >"$tmp" 2>&1
rc=$?

if [ $rc -eq 0 ] && grep -q "no new data" "$tmp"; then
  # Cheap no-op: record one heartbeat line and discard the per-tick log.
  echo "[$ts] heartbeat: no new data" >> logs/heartbeat.log
  rm -f "$tmp"
elif [ $rc -eq 0 ]; then
  # A real re-fit happened — keep the full log and publish the new dashboard publicly.
  mv "$tmp" "logs/daily_${ts}.log"
  echo "[$ts] OK (re-fit) -> data/processed/wc2026_dashboard.html" >> logs/heartbeat.log
  if bash publish.sh >> "logs/daily_${ts}.log" 2>&1; then
    echo "[$ts] published to wc2026-live" >> logs/heartbeat.log
  else
    echo "[$ts] publish FAILED (see logs/daily_${ts}.log)" >> logs/heartbeat.log
  fi
else
  mv "$tmp" "logs/daily_${ts}.log"
  echo "[$ts] FAILED (see logs/daily_${ts}.log)" >> logs/heartbeat.log
fi

# Keep the 30 most recent full re-fit logs; cap the heartbeat log at its last 500 lines.
ls -1t logs/daily_*.log 2>/dev/null | tail -n +31 | xargs -I{} rm -f {} 2>/dev/null || true
tail -n 500 logs/heartbeat.log 2>/dev/null > logs/heartbeat.log.tmp 2>/dev/null && mv logs/heartbeat.log.tmp logs/heartbeat.log || true
