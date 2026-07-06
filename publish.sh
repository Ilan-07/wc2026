#!/usr/bin/env bash
# Publish the rendered dashboard to the gh-pages branch of THIS repo, served by GitHub Pages at
# https://ilan-07.github.io/wc2026/. The forecast-update churn lives ONLY on gh-pages, so the
# code history on main stays clean. Publishes index.html + track-record.json (+ a small README).
# Idempotent: a no-op when the dashboard hasn't changed. Called by run_daily.sh on a real re-fit.
set -uo pipefail
cd "$(dirname "$0")"

SRC="data/processed/wc2026_dashboard.html"
PUB=".publish/pages"   # a clone of this repo checked out for gh-pages publishing

[ -f "$SRC" ] || { echo "publish: no dashboard at $SRC"; exit 1; }
if [ ! -d "$PUB/.git" ]; then
  echo "publish: $PUB missing. Recreate with:"
  echo "  git clone --branch gh-pages https://github.com/Ilan-07/wc2026.git $PUB"
  exit 1
fi

cp "$SRC" "$PUB/index.html"
[ -f data/processed/track_record.json ] && cp data/processed/track_record.json "$PUB/track-record.json"
# Disable Jekyll: the dashboard is pre-built static HTML, and its CSS/JS/news text can contain tokens
# Jekyll's Liquid parser rejects (e.g. "100%}" in CSS, or "{%"/"%}" in a headline), which fails the
# Pages build. .nojekyll serves the files verbatim and immunises every future publish.
[ -f "$PUB/.nojekyll" ] || : > "$PUB/.nojekyll"

if [ ! -f "$PUB/README.md" ]; then
  cat > "$PUB/README.md" <<'MD'
# World Cup 2026 — Live Forecast
Live dashboard → https://ilan-07.github.io/wc2026/
Auto-published from the gh-pages branch of https://github.com/Ilan-07/wc2026. Not betting advice.
MD
fi

git -C "$PUB" add index.html track-record.json README.md .nojekyll 2>/dev/null || git -C "$PUB" add index.html README.md .nojekyll
if git -C "$PUB" diff --cached --quiet; then
  echo "publish: no change to publish"
  exit 0
fi
ts=$(date -u +"%Y-%m-%d %H:%M UTC")
git -C "$PUB" commit -q -m "Update forecast — $ts"
git -C "$PUB" push -q origin HEAD:gh-pages
echo "publish: pushed forecast update ($ts)"

# GitHub Pages' "Deploy from a branch" build is intermittently flaky: pushing to gh-pages triggers
# the auto-managed pages-build-deployment, whose backend sometimes returns a bare "Deployment failed,
# try again later." with no retry — so a red deploy appears even though the content is fine. The push
# above already succeeded and the live site keeps serving the last good build; here we just watch the
# resulting Pages build and, if it errors, ask GitHub to rebuild the same commit (needs `gh`). Best
# effort: never fail the publish over this — a missing `gh`, no auth, or an API hiccup is a no-op.
# launchd runs with a minimal PATH (see run_daily.sh), so Homebrew's bin dirs — where `gh` lives —
# aren't on it. Add them before resolving `gh`, so the retry watcher fires unattended, not just in a
# login shell.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
GH="$(command -v gh || true)"
[ -n "$GH" ] || { echo "publish: gh not found — skipping Pages deploy watch"; exit 0; }
REPO="Ilan-07/wc2026"
pages_status() { "$GH" api "repos/$REPO/pages/builds/latest" -q .status 2>/dev/null; }
for attempt in 1 2 3; do
  # Let the push settle, then wait for the auto-triggered build to leave the "building" state.
  sleep 15
  st=""
  for _ in $(seq 1 24); do          # up to ~2 min per attempt
    st=$(pages_status)
    [ "$st" = "building" ] || [ -z "$st" ] || break
    sleep 5
  done
  if [ "$st" = "built" ]; then
    echo "publish: Pages deploy succeeded"
    exit 0
  fi
  echo "publish: Pages deploy status='${st:-unknown}' (attempt $attempt) — requesting a rebuild"
  "$GH" api -X POST "repos/$REPO/pages/builds" >/dev/null 2>&1 || true
done
echo "publish: Pages deploy still not green after retries — GitHub backend likely degraded; the live"
echo "         site keeps serving the last good build. Re-run later: gh api -X POST repos/$REPO/pages/builds"
