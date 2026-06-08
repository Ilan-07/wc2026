#!/usr/bin/env bash
# One-shot installer for the daily WC2026 forecast (macOS launchd). Run once:
#     ./install_daily.sh           # install + start the daily 09:00 job
#     ./install_daily.sh --remove  # uninstall
set -uo pipefail
cd "$(dirname "$0")"
PLIST="com.wc2026.daily.plist"
DEST="$HOME/Library/LaunchAgents/$PLIST"
LABEL="com.wc2026.daily"

if [[ "${1:-}" == "--remove" ]]; then
  launchctl unload "$DEST" 2>/dev/null || true
  rm -f "$DEST"
  echo "Removed the daily job."
  exit 0
fi

chmod +x run_daily.sh
mkdir -p "$HOME/Library/LaunchAgents"
cp "$PLIST" "$DEST"
launchctl unload "$DEST" 2>/dev/null || true
launchctl load -w "$DEST"
if launchctl list | grep -q "$LABEL"; then
  echo "✓ Daily forecast installed — runs every day at 09:00."
  echo "  Logs: logs/   ·   Remove with: ./install_daily.sh --remove"
  echo "  Run once now to test:  ./run_daily.sh"
else
  echo "⚠ Loaded but not listed — on some macOS versions you may need:"
  echo "  launchctl bootstrap gui/\$(id -u) \"$DEST\""
fi
