"""API-Football injuries/suspensions → availability table (gap #11, the one proven-orthogonal signal).

Injuries are information that is NOT yet in past match results, which is exactly why this is the
contextual signal worth wiring (squad reputation, DNA, etc. all failed because they mirror results).

This client fetches recent injury/suspension records, reconciles the abbreviated API names
("K. Mbappé") to our full squad names ("Kylian Mbappé"), and produces {team: [unavailable players]}
for the injury engine. Key handling (never committed): env ``API_FOOTBALL_KEY`` or the gitignored
``data/raw/.api_football_key``.

Free tier = 100 requests/day; a daily run queries a few leagues, so it stays well within budget.
"""

from __future__ import annotations

import json
import os
import ssl
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

_RAW = Path(__file__).resolve().parents[3] / "data" / "raw"
_KEY_FILE = _RAW / ".api_football_key"
_BASE = "https://v3.football.api-sports.io"

# Leagues where most WC players play their club football (API-Football league IDs).
TOP_LEAGUES = {39: "Premier League", 140: "La Liga", 135: "Serie A",
               78: "Bundesliga", 61: "Ligue 1"}
WORLD_CUP_LEAGUE = 1

# "Missing Fixture" = ruled out; "Questionable" = doubtful (partial availability penalty).
SEVERITY = {"Missing Fixture": 1.0, "Questionable": 0.5}


def _key(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if os.environ.get("API_FOOTBALL_KEY"):
        return os.environ["API_FOOTBALL_KEY"]
    if _KEY_FILE.exists():
        return _KEY_FILE.read_text().strip()
    raise RuntimeError("No API-Football key: set API_FOOTBALL_KEY or write data/raw/.api_football_key")


def _ssl_ctx() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl._create_unverified_context()


def _get(path: str, params: dict, api_key: str | None = None, timeout: float = 30.0) -> list:
    url = f"{_BASE}/{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"x-apisports-key": _key(api_key)})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as r:
        data = json.loads(r.read())
    return data.get("response", [])


def get_injuries(league: int, season: int, api_key: str | None = None) -> list:
    return _get("injuries", {"league": league, "season": season}, api_key)


def normalize(name: str) -> tuple[str, str]:
    """(surname, first-initial) with accents stripped — the join key between API and squad names."""
    s = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    parts = [p for p in s.replace(".", " ").split() if p]
    if not parts:
        return ("", "")
    surname = parts[-1].lower()
    initial = parts[0][0].lower() if parts[0] else ""
    return (surname, initial)


def current_unavailable(leagues=None, season: int = 2025, api_key: str | None = None) -> dict[tuple, dict]:
    """Most-recent injury/suspension per player across ``leagues`` -> {(surname,initial): record}."""
    leagues = leagues or list(TOP_LEAGUES)
    latest: dict[tuple, dict] = {}
    for lg in leagues:
        for rec in get_injuries(lg, season, api_key):
            p = rec.get("player", {})
            key = normalize(p.get("name", ""))
            date = rec.get("fixture", {}).get("date", "")
            if key[0] and (key not in latest or date > latest[key]["date"]):
                latest[key] = {"name": p.get("name"), "type": p.get("type"),
                               "reason": p.get("reason"), "date": date}
    return latest


def availability_table(squads, leagues=None, season: int = 2025, api_key: str | None = None) -> dict[str, list]:
    """Match injured players to our squads -> {team: [(player, type, reason)]}.

    ``squads`` is the {team: Squad} mapping from data.squads.load_squads().
    """
    unavail = current_unavailable(leagues, season, api_key)
    out: dict[str, list] = {}
    for team, sq in squads.items():
        flagged = []
        for pl in sq.players:
            rec = unavail.get(normalize(pl.name))
            if rec:
                flagged.append((pl.name, rec["type"], rec["reason"]))
        if flagged:
            out[team] = flagged
    return out


def penalties_from_availability(squads, table, k: float = 1.0):
    """Convert an availability table into injury-engine penalties (severity-weighted)."""
    from ..intelligence.injuries import availability_penalty

    pens = {}
    for team, flagged in table.items():
        # severity-weight each unavailable player's importance share
        sq = squads[team]
        out_players = [name for name, _t, _ in flagged]
        weights = {name: SEVERITY.get(typ, 1.0) for name, typ, _ in flagged}
        # weighted penalty: scale availability_penalty by per-player severity
        base = availability_penalty(sq, out_players, k=k)
        # approximate severity scaling by average severity of the flagged set
        avg_sev = sum(weights.values()) / len(weights) if weights else 1.0
        pens[team] = base * avg_sev
    return pens
