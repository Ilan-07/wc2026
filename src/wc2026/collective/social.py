"""Social-media pulse feed — DISPLAY ONLY (mirrors sentiment.py's fence).

IMPORTANT: like the Google-News pulse in ``sentiment.py``, this NEVER enters the probabilistic
forecast. Social chatter is the noisiest signal there is, and the project's ablations already show
extra hand-weighted layers don't beat the market. So it lives purely in the dashboard UI: a
glanceable per-team "social pulse" the user can eyeball, never a model input. predict.py appends
it to the *render payload* after all model math is done — it cannot reach a rate, rating, or
probability.

Providers (pick via the ``SOCIAL_PROVIDER`` env var; default "x"):
  * ``x``       — official X API v2 ``recent search`` (needs a paid Bearer token). The only
                  compliant way to read X; unofficial logged-in scrapers violate ToS and get
                  accounts suspended, so they are intentionally not implemented here.
  * ``bluesky`` — free, open AT Protocol search (needs a free app password). Good no-cost option.
  * ``off``     — disable entirely (returns no pulses).

Credentials are read from gitignored files under data/raw/ (or env vars), exactly like the odds
key. Any failure (missing key, offline, rate-limited) degrades gracefully to an empty, neutral
pulse — never an exception that could break the daily run.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

# Reuse the news feed's data shapes + lexicon so the dashboard payload is identical and the
# positive/negative scoring stays consistent across the News and Social pulses.
from wc2026.collective.sentiment import NewsItem, TeamPulse, _headline_score, _ssl_context

PROVIDER = os.environ.get("SOCIAL_PROVIDER", "x").lower()  # "x" | "bluesky" | "off"
_RAW = Path("data/raw")


def _secret(env: str, filename: str) -> str | None:
    """Read a credential from an env var first, then a gitignored data/raw/ file."""
    val = os.environ.get(env)
    if val:
        return val.strip()
    p = _RAW / filename
    return p.read_text().strip() if p.exists() else None


def _pulse(items: list[NewsItem]) -> tuple[float, str]:
    """Same 0-100 squash + mood thresholds as the news pulse, for visual consistency."""
    if not items:
        return 50.0, "neutral"
    avg = sum(i.score for i in items) / len(items)
    pulse = max(0.0, min(100.0, 50.0 + 18.0 * avg))
    mood = "positive" if pulse >= 60 else "negative" if pulse <= 40 else "neutral"
    return round(pulse, 1), mood


# ---------------------------------------------------------------- X (Twitter) API v2 — compliant
def _fetch_x(team: str, limit: int, timeout: float) -> TeamPulse:
    token = _secret("X_BEARER_TOKEN", ".x_bearer_token")
    if not token:
        return TeamPulse(team=team)  # no key -> inert, neutral (feature simply off)
    # recent search: English, no retweets, football-scoped. max_results min is 10 per the API.
    query = f'"{team}" (football OR "world cup") -is:retweet lang:en'
    qs = urllib.parse.urlencode({
        "query": query,
        "max_results": max(10, min(100, limit * 2)),
        "tweet.fields": "created_at",
    })
    url = f"https://api.twitter.com/2/tweets/search/recent?{qs}"
    try:
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}", "User-Agent": "wc2026/1.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
            payload = json.loads(r.read())
    except Exception:
        return TeamPulse(team=team)  # rate-limited / offline / bad key -> neutral, never raise
    items: list[NewsItem] = []
    for tw in (payload.get("data") or [])[:limit]:
        text = " ".join((tw.get("text") or "").split())
        items.append(NewsItem(
            title=text[:200], source="X", date=(tw.get("created_at") or "")[:16],
            link=f"https://x.com/i/web/status/{tw.get('id', '')}",
            score=_headline_score(text)))
    pulse, mood = _pulse(items)
    return TeamPulse(team=team, items=items, pulse=pulse, mood=mood)


# ---------------------------------------------------------------- Bluesky (AT Protocol) — free
def _fetch_bluesky(team: str, limit: int, timeout: float) -> TeamPulse:
    handle = _secret("BSKY_HANDLE", ".bsky_handle")
    app_pw = _secret("BSKY_APP_PASSWORD", ".bsky_app_password")
    if not (handle and app_pw):
        return TeamPulse(team=team)
    try:
        # 1) open a session with the free app password -> short-lived JWT.
        body = json.dumps({"identifier": handle, "password": app_pw}).encode()
        req = urllib.request.Request(
            "https://bsky.social/xrpc/com.atproto.server.createSession",
            data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
            jwt = json.loads(r.read())["accessJwt"]
        # 2) search recent posts.
        qs = urllib.parse.urlencode({"q": f"{team} world cup", "limit": max(1, min(25, limit))})
        sreq = urllib.request.Request(
            f"https://bsky.social/xrpc/app.bsky.feed.searchPosts?{qs}",
            headers={"Authorization": f"Bearer {jwt}"})
        with urllib.request.urlopen(sreq, timeout=timeout, context=_ssl_context()) as r:
            posts = json.loads(r.read()).get("posts", [])
    except Exception:
        return TeamPulse(team=team)
    items = []
    for p in posts[:limit]:
        rec = p.get("record", {})
        text = " ".join((rec.get("text") or "").split())
        items.append(NewsItem(
            title=text[:200], source="Bluesky", date=(rec.get("createdAt") or "")[:16],
            link="https://bsky.app", score=_headline_score(text)))
    pulse, mood = _pulse(items)
    return TeamPulse(team=team, items=items, pulse=pulse, mood=mood)


_PROVIDERS = {"x": _fetch_x, "bluesky": _fetch_bluesky}


def fetch_many(teams: list[str], limit: int = 6) -> dict[str, TeamPulse]:
    """Per-team social pulse for the dashboard (display only). Empty dict if disabled."""
    if PROVIDER == "off":
        return {}
    fn = _PROVIDERS.get(PROVIDER, _fetch_x)
    return {t: fn(t, limit, 8.0) for t in teams}
