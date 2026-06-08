"""StatsBomb open-data xG loader (the one legitimate, licensed xG source).

StatsBomb publish free, openly-licensed event data for selected competitions — including the
**2018 & 2022 World Cups, Euro 2024 and Copa América 2024**. That's shot-level expected goals
(xG) for the most recent major-tournament matches of most WC2026 teams. xG is far less noisy
than actual goals, so an xG-based view of form sees through finishing variance.

Honest scope: this is *tournament* data for a handful of competitions, NOT a current club-form
feed for all ~1,200 WC2026 players. Its realistic use is an xG-vs-goals "finishing/luck" signal
and xG-based recent form for the elite teams — to be ablation-gated like everything else.

Data layout (raw.githubusercontent.com/statsbomb/open-data):
  data/competitions.json
  data/matches/{competition_id}/{season_id}.json
  data/events/{match_id}.json   (shots carry shot.statsbomb_xg)
"""

from __future__ import annotations

import json
import ssl
import urllib.request
from dataclasses import dataclass

_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Competitions with xG that matter for international form (competition_id, season_id, label).
# Every men's senior international tournament StatsBomb open-data carries shot-level xG for — adding
# AFCON 2023 (the only xG source for many African WC2026 teams) and Euro 2020 widened coverage from
# 4 tournaments / ~211 matches to 6 / ~314, and lifted teams from ~3-6 games each toward more.
INTERNATIONAL = {
    "wc2022": (43, 106, "FIFA World Cup 2022"),
    "wc2018": (43, 3, "FIFA World Cup 2018"),
    "euro2024": (55, 282, "UEFA Euro 2024"),
    "euro2020": (55, 43, "UEFA Euro 2020"),
    "copa2024": (223, 282, "Copa América 2024"),
    "afcon2023": (1267, 107, "Africa Cup of Nations 2023"),
}


def _ssl_ctx() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl._create_unverified_context()


def _get_json(url: str, timeout: float = 30.0):
    req = urllib.request.Request(url, headers={"User-Agent": "wc2026/1.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as r:
        return json.loads(r.read())


def load_matches(competition_id: int, season_id: int) -> list[dict]:
    return _get_json(f"{_BASE}/matches/{competition_id}/{season_id}.json")


def match_team_xg(match_id: int) -> dict[str, dict[str, float]]:
    """Per-team xG and goals in one match from its shot events."""
    events = _get_json(f"{_BASE}/events/{match_id}.json")
    agg: dict[str, dict[str, float]] = {}
    for e in events:
        if e.get("type", {}).get("name") != "Shot":
            continue
        team = e.get("team", {}).get("name", "")
        shot = e.get("shot", {})
        a = agg.setdefault(team, {"xg": 0.0, "goals": 0.0, "shots": 0.0})
        a["xg"] += float(shot.get("statsbomb_xg", 0.0) or 0.0)
        a["shots"] += 1
        if shot.get("outcome", {}).get("name") == "Goal":
            a["goals"] += 1
    return agg


def competition_match_records(key: str) -> list[dict]:
    """Per-match xG + goals records for a competition, chronologically ordered.

    Each record: match_id, date, home, away, home_xg, away_xg, home_goals, away_goals.
    Used to test xG's predictiveness (is past xG-diff a better leading indicator than goal-diff?).
    """
    cid, sid, _ = INTERNATIONAL[key]
    recs = []
    for m in load_matches(cid, sid):
        home = m["home_team"]["home_team_name"]
        away = m["away_team"]["away_team_name"]
        xg = match_team_xg(m["match_id"])
        recs.append({
            "match_id": m["match_id"], "date": m.get("match_date", ""),
            "home": home, "away": away,
            "home_xg": xg.get(home, {}).get("xg", 0.0),
            "away_xg": xg.get(away, {}).get("xg", 0.0),
            "home_goals": m["home_score"], "away_goals": m["away_score"],
        })
    recs.sort(key=lambda r: r["date"])
    return recs


@dataclass
class TeamXG:
    team: str
    matches: int = 0
    xg_for: float = 0.0
    xg_against: float = 0.0
    goals_for: float = 0.0
    goals_against: float = 0.0

    @property
    def xg_diff_per_match(self) -> float:
        return (self.xg_for - self.xg_against) / max(self.matches, 1)

    @property
    def finishing(self) -> float:
        """Goals minus xG (per match): + = clinical/lucky, − = wasteful/unlucky."""
        return (self.goals_for - self.xg_for) / max(self.matches, 1)


def competition_team_xg(key: str, limit: int | None = None) -> dict[str, TeamXG]:
    """Aggregate per-team xG for/against across a competition's matches.

    ``key`` is one of :data:`INTERNATIONAL`. ``limit`` caps matches fetched (for quick demos).
    Each match is a separate HTTP request, so this is bandwidth-heavy; cache the result.
    """
    cid, sid, _ = INTERNATIONAL[key]
    matches = load_matches(cid, sid)
    if limit:
        matches = matches[:limit]
    table: dict[str, TeamXG] = {}
    for m in matches:
        home = m["home_team"]["home_team_name"]
        away = m["away_team"]["away_team_name"]
        xg = match_team_xg(m["match_id"])
        h = table.setdefault(home, TeamXG(home))
        a = table.setdefault(away, TeamXG(away))
        h.matches += 1
        a.matches += 1
        h.xg_for += xg.get(home, {}).get("xg", 0.0)
        h.xg_against += xg.get(away, {}).get("xg", 0.0)
        a.xg_for += xg.get(away, {}).get("xg", 0.0)
        a.xg_against += xg.get(home, {}).get("xg", 0.0)
        h.goals_for += m["home_score"]
        h.goals_against += m["away_score"]
        a.goals_for += m["away_score"]
        a.goals_against += m["home_score"]
    return table
