#!/usr/bin/env bash
# Publish the rendered dashboard to the gh-pages branch of THIS repo, served by GitHub Pages at
# https://ilan-07.github.io/worldcup2026/. The forecast-update churn lives ONLY on gh-pages, so the
# code history on main stays clean. Publishes index.html + track-record.json (+ a small README).
# Idempotent: a no-op when the dashboard hasn't changed. Called by run_daily.sh on a real re-fit.
set -uo pipefail
cd "$(dirname "$0")"

SRC="data/processed/wc2026_dashboard.html"
PUB=".publish/pages"   # a clone of this repo checked out for gh-pages publishing

[ -f "$SRC" ] || { echo "publish: no dashboard at $SRC"; exit 1; }
if [ ! -d "$PUB/.git" ]; then
  echo "publish: $PUB missing. Recreate with:"
  echo "  git clone --branch gh-pages https://github.com/Ilan-07/worldcup2026.git $PUB"
  exit 1
fi

cp "$SRC" "$PUB/index.html"
[ -f data/processed/track_record.json ] && cp data/processed/track_record.json "$PUB/track-record.json"

if [ ! -f "$PUB/README.md" ]; then
  cat > "$PUB/README.md" <<'MD'
# World Cup 2026 — Live Forecast
Live dashboard → https://ilan-07.github.io/worldcup2026/
Auto-published from the gh-pages branch of https://github.com/Ilan-07/worldcup2026. Not betting advice.
MD
fi

git -C "$PUB" add index.html track-record.json README.md 2>/dev/null || git -C "$PUB" add index.html README.md
if git -C "$PUB" diff --cached --quiet; then
  echo "publish: no change to publish"
  exit 0
fi
ts=$(date -u +"%Y-%m-%d %H:%M UTC")
git -C "$PUB" commit -q -m "Update forecast — $ts"
git -C "$PUB" push -q origin HEAD:gh-pages
echo "publish: pushed forecast update ($ts)"
