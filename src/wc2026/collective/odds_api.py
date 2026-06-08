"""Live WC2026 outright odds via The Odds API (replaces the hand-made snapshot).

Fetches "to win the tournament" decimal odds, takes the consensus (median) across bookmakers,
reconciles team names to our draw, and writes the same CSV the rest of the system already reads
(``data/raw/wc2026_outright_odds.csv``) — so nothing downstream changes.

API key handling (do NOT hardcode / commit): read from the ``ODDS_API_KEY`` environment variable,
or from an untracked local file ``data/raw/.odds_api_key``.

    export ODDS_API_KEY=...            # or: echo <key> > data/raw/.odds_api_key
    python -c "from wc2026.collective.odds_api import refresh; refresh()"
"""

from __future__ import annotations

import json
import os
import ssl
import statistics
import urllib.request
from datetime import date
from pathlib import Path

_RAW = Path(__file__).resolve().parents[3] / "data" / "raw"
_KEY_FILE = _RAW / ".odds_api_key"
_SPORT = "soccer_fifa_world_cup_winner"

# The Odds API team names -> our canonical (results.csv / Wikipedia) names.
NAME_ALIASES = {
    "USA": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def _api_key(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if os.environ.get("ODDS_API_KEY"):
        return os.environ["ODDS_API_KEY"]
    if _KEY_FILE.exists():
        return _KEY_FILE.read_text().strip()
    raise RuntimeError("No odds API key: set ODDS_API_KEY or write data/raw/.odds_api_key")


def _ssl_ctx() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl._create_unverified_context()


def fetch_outright_odds(
    api_key: str | None = None, regions: str = "us,uk,eu", timeout: float = 30.0
) -> dict[str, float]:
    """Return {canonical_team: consensus (median) decimal odds} for the WC winner market."""
    key = _api_key(api_key)
    url = (
        f"https://api.the-odds-api.com/v4/sports/{_SPORT}/odds/"
        f"?apiKey={key}&regions={regions}&markets=outrights&oddsFormat=decimal"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "wc2026/1.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
        data = json.loads(resp.read())
    if isinstance(data, dict):  # error payload
        raise RuntimeError(f"Odds API error: {data}")

    quotes: dict[str, list[float]] = {}
    for event in data:
        for bk in event.get("bookmakers", []):
            for market in bk.get("markets", []):
                for o in market.get("outcomes", []):
                    name = NAME_ALIASES.get(str(o["name"]), str(o["name"]))
                    quotes.setdefault(name, []).append(float(o["price"]))
    return {team: statistics.median(prices) for team, prices in quotes.items()}


def refresh(out: str | Path | None = None, api_key: str | None = None) -> Path:
    """Fetch live odds and (over)write the outright-odds CSV the system reads."""
    odds = fetch_outright_odds(api_key=api_key)
    p = Path(out) if out else _RAW / "wc2026_outright_odds.csv"
    lines = [
        f"# WC2026 outright odds — LIVE from The Odds API ({_SPORT}), consensus median across",
        f"# bookmakers, fetched {date.today().isoformat()}. Regenerate with odds_api.refresh().",
        "team,odds",
    ]
    lines += [f"{t},{o:.2f}" for t, o in sorted(odds.items(), key=lambda x: x[1])]
    body = "\n".join(lines) + "\n"
    p.write_text(body)
    # Also bank a dated snapshot so odds *movement* accrues over time (Gap 2, free signal).
    hist = _RAW.parent / "processed" / "odds_history"
    hist.mkdir(parents=True, exist_ok=True)
    (hist / f"odds_{date.today().strftime('%Y%m%d')}.csv").write_text(body)
    return p
