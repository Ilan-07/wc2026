"""Transfermarkt player market values → per-squad value table (the long-promised caps replacement).

Market value is the single best public proxy for player *quality/importance*: it prices in form,
age, level and scarcity in a way caps (which only measure experience — why veterans wrongly topped
the "key players" list) never can. ``intelligence.injuries.player_importance`` was written to accept
exactly this and swap it in for caps without an interface change.

Source: a **published, redistributable** Transfermarkt snapshot (salimt/football-datasets on GitHub —
the same kind of community-collected, attributable dump we already use for results), NOT live scraping
(transfermarkt.com blocks bots, and we don't evade that). Two CSVs under ``data/raw/transfermarkt/``:

    latest_market_value.csv   player_id,date_unix,value          # latest € value per player
    player_profiles.csv       player_id,player_name,citizenship,main_position,...

Values are a 2025 snapshot — current enough that the top of the list is the real modern star set
(Yamal €200m, Bellingham/Haaland/Mbappé €180m, Pedri €140m), unlike API-Football's 2022-24 lock.

Matching to our Wikipedia squads is by (nationality bucket → normalized name), so common surnames
don't collide across countries. Coverage is reported, never silently assumed.
"""

from __future__ import annotations

import csv
import re
import sys
import unicodedata
from pathlib import Path

from ..collective.api_football import normalize  # (surname, first-initial) join key
from .squads import Squad

_TM = Path(__file__).resolve().parents[3] / "data" / "raw" / "transfermarkt"
_VALUES = _TM / "latest_market_value.csv"
_PROFILES = _TM / "player_profiles.csv"

# Squad team name (Wikipedia) → substring that appears in Transfermarkt's citizenship field.
# Identity for most; only the genuine spelling/format mismatches are listed.
TEAM_NATIONALITY: dict[str, str] = {
    "United States": "United States", "USA": "United States",
    "South Korea": "Korea, South", "Korea Republic": "Korea, South",
    "IR Iran": "Iran", "Iran": "Iran",
    "Ivory Coast": "Cote d'Ivoire", "Côte d'Ivoire": "Cote d'Ivoire",
    "Cape Verde": "Cape Verde", "Curaçao": "Curacao",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Turkey": "Turkiye", "Türkiye": "Turkiye",
}


def _accentless(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _clean_tm_name(name: str) -> str:
    """Drop the ``(player_id)`` disambiguation suffix this dataset appends, e.g. 'Lamine Yamal (937958)'."""
    return re.sub(r"\s*\(\d+\)\s*$", "", name).strip()


def _full_norm(name: str) -> str:
    """Whole accent-stripped lowercased name — the strongest key (beats surname collisions)."""
    return " ".join(_accentless(name).lower().replace(".", " ").split())


def load_value_index() -> dict[str, dict]:
    """Build {citizenship_token: {"full": {norm_name: value}, "si": {(surname,initial): value}}}.

    Bucketed by every citizenship a player holds, so a dual-national (Yamal: "Spain  Equatorial
    Guinea") is findable under Spain. Within a bucket we keep the *max* value on a key collision.
    """
    if not _VALUES.exists() or not _PROFILES.exists():
        raise FileNotFoundError(f"Transfermarkt CSVs not found under {_TM} (see module docstring).")

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    values: dict[str, float] = {}
    with _VALUES.open() as f:
        for row in csv.DictReader(f):
            try:
                v = float(row["value"])
            except (TypeError, ValueError):
                continue
            if v > 0:
                values[row["player_id"]] = v

    index: dict[str, dict] = {}
    with _PROFILES.open() as f:
        for row in csv.DictReader(f):
            pid = row["player_id"]
            val = values.get(pid)
            if not val:
                continue
            name = _clean_tm_name(row.get("player_name") or "")
            full = _full_norm(name)
            si = normalize(name)
            # citizenship is space-joined country names (some contain spaces) — bucket on the raw
            # string; lookups test membership by `nationality in citizenship`, so spaces are fine.
            cit = _accentless(row.get("citizenship") or "")
            for bucket in _citizenship_buckets(cit):
                b = index.setdefault(bucket, {"full": {}, "si": {}})
                if val > b["full"].get(full, 0):
                    b["full"][full] = val
                if val > b["si"].get(si, 0):
                    b["si"][si] = val
    return index


def _citizenship_buckets(cit: str) -> list[str]:
    """The whole citizenship string, lowercased — one bucket; membership tests do the matching."""
    return [cit.lower()] if cit.strip() else []


def _lookup(bucket_keys: list[str], index: dict, full: str, si) -> float | None:
    for key, bucket in index.items():
        if not any(nat in key for nat in bucket_keys):
            continue
        if full in bucket["full"]:
            return bucket["full"][full]
    # fall back to surname+initial within the same nationality buckets
    for key, bucket in index.items():
        if not any(nat in key for nat in bucket_keys):
            continue
        if si in bucket["si"]:
            return bucket["si"][si]
    return None


def squad_values(squads: dict[str, Squad]) -> dict[str, dict[str, float]]:
    """{team: {player_name: market_value_eur}} for every player we can match. Unmatched are omitted."""
    index = load_value_index()
    out: dict[str, dict[str, float]] = {}
    for team, sq in squads.items():
        nat = _accentless(TEAM_NATIONALITY.get(team, team)).lower()
        keys = [nat]
        matched: dict[str, float] = {}
        for pl in sq.players:
            v = _lookup(keys, index, _full_norm(pl.name), normalize(pl.name))
            if v:
                matched[pl.name] = v
        out[team] = matched
    return out


def coverage_report(squads: dict[str, Squad], values: dict[str, dict[str, float]]) -> dict:
    """Per-team and overall match coverage — surfaced so we never pretend to data we don't have."""
    total = matched = 0
    per_team = {}
    for team, sq in squads.items():
        n = len(sq.players)
        m = len(values.get(team, {}))
        total += n
        matched += m
        per_team[team] = (m, n)
    return {"matched": matched, "total": total,
            "pct": round(100 * matched / total, 1) if total else 0.0, "per_team": per_team}
