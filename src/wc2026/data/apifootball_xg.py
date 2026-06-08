"""Phased xG collection from API-Football (free tier: 100 req/day, 10/min, seasons 2022-2024).

Each fixture's xG costs 1 request, so a full dataset can't be pulled in one day. This collector is
**resumable and budget-aware**: each run fetches up to a daily budget, caches what it gets, records
progress, and stops cleanly — so you run it day after day (manually or via the daily job) until the
dataset is built. Targets are ordered by World-Cup relevance (internationals first, then top clubs).

Honest limits: the free tier only serves **2022-2024**, so this gives a rich xG *history* but nothing
from the current (2025-26) season or WC2026 — the rating's xG side will be ~1-2 yrs stale on the recent
end. Comprehensive coverage takes ~1-2 weeks of daily runs. Key handling reuses api_football (gitignored).

    python -c "from wc2026.data.apifootball_xg import collect; collect()"   # run daily
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..collective.api_football import _get  # key handling + GET with the apisports header

_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "apifootball_xg"
_PROGRESS = _DIR / "_progress.json"

# (league_id, season, label) — ordered by relevance to a World Cup team rating.
TARGETS: list[tuple[int, int, str]] = [
    (1, 2022, "World Cup 2022"),
    (4, 2024, "Euro 2024"),
    (9, 2024, "Copa America 2024"),
    (5, 2022, "Nations League 2022-23"),
    (5, 2024, "Nations League 2024-25"),
    (10, 2024, "Friendlies 2024"), (10, 2023, "Friendlies 2023"), (10, 2022, "Friendlies 2022"),
    # top-5 club leagues (player form; lower WC priority) — 3 recent allowed seasons each
    (39, 2024, "Premier League 24-25"), (140, 2024, "La Liga 24-25"), (135, 2024, "Serie A 24-25"),
    (78, 2024, "Bundesliga 24-25"), (61, 2024, "Ligue 1 24-25"),
    (39, 2023, "Premier League 23-24"), (140, 2023, "La Liga 23-24"), (135, 2023, "Serie A 23-24"),
    (78, 2023, "Bundesliga 23-24"), (61, 2023, "Ligue 1 23-24"),
]


def _load_progress() -> dict:
    return json.loads(_PROGRESS.read_text()) if _PROGRESS.exists() else {"done": [], "fixtures": {}}


def _fixture_xg(fid: int) -> tuple:
    """(home_xg, away_xg) for a fixture; each may be None if xG isn't reported."""
    resp = _get("fixtures/statistics", {"fixture": fid})
    if len(resp) != 2:
        return (None, None)
    vals = []
    for team in resp:
        xg = next((s["value"] for s in team["statistics"] if s["type"] == "expected_goals"), None)
        vals.append(float(xg) if xg not in (None, "") else None)
    return (vals[0], vals[1])


def _store_path(key: str) -> Path:
    return _DIR / f"{key}.json"


def collect(daily_budget: int = 85, per_request_pause: float = 6.5) -> dict:
    """Fetch one day's batch of fixture xG, resumably. Returns a progress summary.

    ``daily_budget`` leaves headroom under the 100/day cap; ``per_request_pause`` keeps under 10/min.
    Every attempted fixture is recorded (xG or null) so it is never re-fetched.
    """
    _DIR.mkdir(parents=True, exist_ok=True)
    prog = _load_progress()
    used = collected_now = 0

    for lid, season, _label in TARGETS:
        if used >= daily_budget:
            break
        key = f"{lid}_{season}"
        if key in prog["done"]:
            continue
        store = _store_path(key)
        records = json.loads(store.read_text()) if store.exists() else {}

        if key not in prog["fixtures"]:                      # list the target's fixtures (1 request)
            fixtures = _get("fixtures", {"league": lid, "season": season})
            used += 1
            prog["fixtures"][key] = [
                {"id": f["fixture"]["id"], "date": f["fixture"]["date"][:10],
                 "home": f["teams"]["home"]["name"], "away": f["teams"]["away"]["name"],
                 "hg": f["goals"]["home"], "ag": f["goals"]["away"]}
                for f in fixtures if f["fixture"]["status"]["short"] == "FT"]
            _PROGRESS.write_text(json.dumps(prog))

        for f in prog["fixtures"][key]:
            if used >= daily_budget:
                break
            if str(f["id"]) in records:                      # already attempted
                continue
            hxg, axg = _fixture_xg(f["id"])
            used += 1
            records[str(f["id"])] = {**f, "home_xg": hxg, "away_xg": axg}
            if hxg is not None:
                collected_now += 1
            time.sleep(per_request_pause)

        store.write_text(json.dumps(records))
        if all(str(f["id"]) in records for f in prog["fixtures"][key]):
            prog["done"].append(key)
        _PROGRESS.write_text(json.dumps(prog))

    remaining = 0
    for lid, season, _ in TARGETS:
        key = f"{lid}_{season}"
        listed = prog["fixtures"].get(key, [])
        store = _store_path(key)
        have = json.loads(store.read_text()) if store.exists() else {}
        remaining += sum(1 for f in listed if str(f["id"]) not in have)
    return {"requests_used": used, "xg_collected_this_run": collected_now,
            "targets_done": len(prog["done"]), "fixtures_remaining_listed": remaining}


def load_xg_records() -> list[dict]:
    """All collected fixtures with xG -> match dicts for the xG rating."""
    out = []
    for store in sorted(_DIR.glob("*.json")):
        if store.name == "_progress.json":
            continue
        for r in json.loads(store.read_text()).values():
            out.append({"home_team": r["home"], "away_team": r["away"], "date": r["date"],
                        "home_goals": r["hg"], "away_goals": r["ag"],
                        "home_xg": r["home_xg"], "away_xg": r["away_xg"]})
    return out
