"""Real international-football data loaders (plan P0/P1 data layer).

Source: the public, regularly-updated `martj42/international_results` dataset (also on Kaggle
as "International football results from 1872 to present"). Download once into ``data/raw/``:

    curl -sSL -o data/raw/results.csv \\
        https://raw.githubusercontent.com/martj42/international_results/master/results.csv
    curl -sSL -o data/raw/shootouts.csv \\
        https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv

``results.csv`` columns: date, home_team, away_team, home_score, away_score, tournament,
city, country, neutral. Crucially it already contains the **scheduled WC2026 group fixtures**
(with NA scores), so :func:`load_wc2026_groups` reconstructs the real 12-group draw directly
from the data instead of hard-coding it.

This module returns the same match-dict shape the model and Elo engine expect, so it is a
drop-in replacement for ``data/synthetic.py``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

DEFAULT_RESULTS = Path(__file__).resolve().parents[3] / "data" / "raw" / "results.csv"

# Map dataset tournament names to the Elo match-importance buckets (ratings/elo.py).
_CONTINENTAL_FINALS = (
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "CONCACAF Championship",
    "Oceania Nations Cup",
)


def tournament_importance(name: str) -> str:
    """Bucket a tournament name into a match-importance weight class."""
    n = name or ""
    if n == "FIFA World Cup":
        return "world_cup"
    if "qualification" in n:
        return "qualifier"
    if "Nations League" in n:
        return "nations_league"
    if n == "FIFA Confederations Cup":
        return "confederations"
    if any(n == f or n.startswith(f) for f in _CONTINENTAL_FINALS):
        return "continental"
    return "friendly"


def _read_raw(path: str | Path | None) -> pd.DataFrame:
    p = Path(path) if path is not None else DEFAULT_RESULTS
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download the dataset first (see module docstring)."
        )
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_results(
    path: str | Path | None = None,
    since: str | date | None = "2014-01-01",
    min_team_matches: int = 20,
    played_only: bool = True,
    keep_teams: set[str] | None = None,
) -> list[dict]:
    """Load real international results as model-ready match dicts.

    * ``since`` trims to a recent window so ratings reflect modern strength (with the
      Dixon-Coles half-life doing finer recency weighting on top).
    * ``min_team_matches`` keeps only established national teams (both sides must clear the
      threshold within the window) to avoid noisy singleton parameters.
    * ``keep_teams`` are always retained regardless of match count (e.g. the 48 WC2026
      teams, so every one of them gets a rating even if it is a minnow).
    * ``played_only`` drops fixtures with missing scores (e.g. upcoming WC2026 matches).
    """
    df = _read_raw(path)
    if since is not None:
        df = df[df["date"] >= pd.to_datetime(since)]
    if played_only:
        df = df.dropna(subset=["home_score", "away_score"])

    if min_team_matches > 0:
        counts: dict[str, int] = {}
        for col in ("home_team", "away_team"):
            for t, c in df[col].value_counts().items():
                counts[t] = counts.get(t, 0) + int(c)
        keep = {t for t, c in counts.items() if c >= min_team_matches}
        keep |= keep_teams or set()
        df = df[df["home_team"].isin(keep) & df["away_team"].isin(keep)]

    matches: list[dict] = []
    for r in df.itertuples(index=False):
        matches.append(
            {
                "home_team": r.home_team,
                "away_team": r.away_team,
                "home_score": int(r.home_score),
                "away_score": int(r.away_score),
                "neutral": bool(r.neutral),
                "date": r.date.date(),
                "importance": tournament_importance(r.tournament),
                "tournament": r.tournament,
            }
        )
    matches.sort(key=lambda m: m["date"])
    return matches


def load_wc2026_played_groups(path: str | Path | None = None) -> dict[frozenset, tuple]:
    """Played WC2026 *group-stage* matches, for conditioning the live forecast.

    Returns {frozenset({home, away}): (home_team, home_score, away_score)} for every 2026 FIFA
    World Cup fixture that (a) has a real score and (b) is between two teams in the same group
    (i.e. a group match, not a knockout). As the tournament is played and ``results.csv`` is
    re-fetched, this dict fills in and the simulator locks those results instead of sampling them.
    """
    df = _read_raw(path)
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)]
    wc = wc.dropna(subset=["home_score", "away_score"])
    groups = load_wc2026_groups(path)
    group_of = {t: g for g, ts in groups.items() for t in ts}
    out: dict[frozenset, tuple] = {}
    for r in wc.itertuples(index=False):
        h, a = r.home_team, r.away_team
        if group_of.get(h) is not None and group_of.get(h) == group_of.get(a):
            out[frozenset((h, a))] = (h, int(r.home_score), int(r.away_score))
    return out


def load_wc2026_group_venue_altitudes(path: str | Path | None = None) -> dict[frozenset, int]:
    """Map each WC2026 group fixture (frozenset of the two teams) -> its venue altitude (m).

    Lets the simulator apply venue-specific altitude effects (Mexican venues) to the right matches.
    """
    from .venues import venue_for_city

    df = _read_raw(path)
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)]
    groups = load_wc2026_groups(path)
    group_of = {t: g for g, ts in groups.items() for t in ts}
    out: dict[frozenset, int] = {}
    for r in wc.itertuples(index=False):
        h, a = r.home_team, r.away_team
        if group_of.get(h) is not None and group_of.get(h) == group_of.get(a):
            v = venue_for_city(str(r.city))
            if v is not None:
                out[frozenset((h, a))] = v.altitude_m
    return out


def load_wc2026_group_fixtures(path: str | Path | None = None) -> list[tuple]:
    """Chronological list of WC2026 group fixtures as ``(date, home, away, city)`` tuples.

    Feeds the fatigue/travel covariate (rest-days + cumulative travel per team). Only same-group
    pairings are kept, matching the venue-altitude loader.
    """
    df = _read_raw(path)
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)]
    groups = load_wc2026_groups(path)
    group_of = {t: g for g, ts in groups.items() for t in ts}
    rows = []
    for r in wc.itertuples(index=False):
        h, a = r.home_team, r.away_team
        if group_of.get(h) is not None and group_of.get(h) == group_of.get(a):
            rows.append((r.date, h, a, str(r.city)))
    rows.sort(key=lambda x: x[0])
    return rows


def load_wc2026_group_fixture_fatigue(
    path: str | Path | None = None, **kwargs
) -> dict[frozenset, dict[str, float]]:
    """Per-fixture ``frozenset({home, away}) -> {team: fatigue_penalty}`` for the WC2026 group stage."""
    from ..intelligence.conditions import fixture_fatigue_penalties

    return fixture_fatigue_penalties(load_wc2026_group_fixtures(path), **kwargs)


def load_wc2026_knockout_results(path: str | Path | None = None) -> dict[frozenset, str]:
    """Played WC2026 *knockout* results as {frozenset({a, b}): winner}, for KO conditioning.

    A knockout match is a 2026 FIFA World Cup fixture between teams from different groups. The
    winner comes from the score, or from ``shootouts.csv`` when the match was a draw after extra
    time. Empty until the knockout stage is played; populates as ``results.csv`` is re-fetched.
    """
    df = _read_raw(path)
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)]
    wc = wc.dropna(subset=["home_score", "away_score"])
    groups = load_wc2026_groups(path)
    group_of = {t: g for g, ts in groups.items() for t in ts}

    # shootout winners keyed by (date, home, away)
    sp = (Path(path).parent if path else DEFAULT_RESULTS.parent) / "shootouts.csv"
    shootouts: dict[tuple, str] = {}
    if sp.exists():
        sh = pd.read_csv(sp)
        for r in sh.itertuples(index=False):
            shootouts[(str(r.date), r.home_team, r.away_team)] = r.winner

    out: dict[frozenset, str] = {}
    for r in wc.itertuples(index=False):
        h, a = r.home_team, r.away_team
        if group_of.get(h) is not None and group_of.get(a) is not None and group_of[h] != group_of[a]:
            hs, as_ = int(r.home_score), int(r.away_score)
            if hs > as_:
                w = h
            elif as_ > hs:
                w = a
            else:
                w = shootouts.get((str(r.date.date()), h, a)) or shootouts.get((str(r.date.date()), a, h))
            if w:
                out[frozenset((h, a))] = w
    return out


def load_wc2026_played_knockouts(path: str | Path | None = None) -> dict[frozenset, tuple]:
    """Played WC2026 *knockout* scorelines as {frozenset({a, b}): (home, home_score, away_score)}.

    Mirrors :func:`load_wc2026_played_groups` but for cross-group (knockout) fixtures, so the
    display layer can lock a real scoreline once a knockout tie has been played. Empty until the
    knockout stage starts; populates as ``results.csv`` is re-fetched.
    """
    df = _read_raw(path)
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)]
    wc = wc.dropna(subset=["home_score", "away_score"])
    groups = load_wc2026_groups(path)
    group_of = {t: g for g, ts in groups.items() for t in ts}
    out: dict[frozenset, tuple] = {}
    for r in wc.itertuples(index=False):
        h, a = r.home_team, r.away_team
        if group_of.get(h) is not None and group_of.get(a) is not None and group_of[h] != group_of[a]:
            out[frozenset((h, a))] = (h, int(r.home_score), int(r.away_score))
    return out


def load_bracket_file(path: str | Path | None = None) -> list[str] | None:
    """Load a supplied Round-of-32 bracket (32 team names, one per line) once it is known.

    Default location ``data/raw/wc2026_bracket.txt``. Returns None if absent (group stage not done).
    """
    p = Path(path) if path else DEFAULT_RESULTS.parent / "wc2026_bracket.txt"
    if not p.exists():
        return None
    teams = [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return teams if len(teams) == 32 else None


def load_wc2026_r32_fixtures(path: str | Path | None = None) -> list[tuple[str, str]] | None:
    """The 16 real Round-of-32 ties as ``(home, away)``, or None until all are scheduled.

    Once the knockout draw is published, ``results.csv`` lists the actual R32 fixtures. They are
    the earliest cross-group (different-group) 2026 World Cup fixtures in which each of the 32
    qualifiers appears exactly once — later knockout rounds reuse teams, so taking ties in date
    order until 32 distinct teams are seen isolates the Round of 32 cleanly, played or not. This
    is the source of truth for the bracket: the slot template can only *re-derive* it, and its
    Annexe-C third allocation does not always match a given edition's published draw.
    """
    df = _read_raw(path)
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)].sort_values("date")
    groups = load_wc2026_groups(path)
    group_of = {t: g for g, ts in groups.items() for t in ts}
    seen: set[str] = set()
    ties: list[tuple[str, str]] = []
    for r in wc.itertuples(index=False):
        h, a = r.home_team, r.away_team
        if group_of.get(h) and group_of.get(a) and group_of[h] != group_of[a]:
            if h in seen or a in seen:
                continue  # a repeated team means we've reached a later round
            seen.add(h)
            seen.add(a)
            ties.append((h, a))
    return ties if len(ties) == 16 else None


def load_shootout_psi(path: str | Path | None = None, prior: float = 4.0) -> dict[str, float]:
    """Per-team penalty-shootout skill psi in [0,1] from historical shootout results.

    psi = shrunk win rate = (wins + prior/2) / (wins + losses + prior), so teams with few
    shootouts sit near 0.5. Feeds the knockout shootout model in ``match_model.sample_knockout``
    (P(team i wins) = sigmoid(scale * (psi_i - psi_j))). Reads ``shootouts.csv`` (date, home_team,
    away_team, winner, first_shooter).
    """
    p = Path(path) if path is not None else DEFAULT_RESULTS.parent / "shootouts.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    wins: dict[str, int] = {}
    total: dict[str, int] = {}
    for r in df.itertuples(index=False):
        for t in (r.home_team, r.away_team):
            total[t] = total.get(t, 0) + 1
        wins[r.winner] = wins.get(r.winner, 0) + 1
    return {
        t: (wins.get(t, 0) + prior / 2) / (total[t] + prior) for t in total
    }


def load_shootout_records(path: str | Path | None = None) -> list[dict]:
    """Chronological shootout history as dicts (date, home, away, winner) for the win-propensity model.

    Feeds ``model.shootout.ShootoutModel`` (Lane 2 #3), which learns the shootout bias scale from this
    history instead of the hand-set 0.4. Reads the same ``shootouts.csv`` as :func:`load_shootout_psi`.
    """
    p = Path(path) if path is not None else DEFAULT_RESULTS.parent / "shootouts.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    out = [
        {"date": str(r.date), "home": r.home_team, "away": r.away_team, "winner": r.winner}
        for r in df.itertuples(index=False)
        if isinstance(r.winner, str) and r.winner in (r.home_team, r.away_team)
    ]
    return out


def load_wc2026_groups(path: str | Path | None = None) -> dict[str, list[str]]:
    """Reconstruct the real WC2026 12-group draw from the scheduled fixtures.

    The group stage is a round-robin within each group of four, so two teams share a group
    iff they play each other in the group phase. We build that graph from the "FIFA World Cup"
    2026 fixtures and read off the connected components (cliques of 4). Groups are then labelled
    A..L by the date of each group's first kickoff.

    Once the knockout phase begins, the dataset also contains cross-group fixtures (R32 onward)
    that would merge the group cliques. We exclude them by capping each team at its three
    group opponents: walking the fixtures in date order, a match is a group edge only while
    *both* teams still have fewer than three opponents recorded. Every team plays exactly three
    group games, so all later (knockout) fixtures are naturally rejected.
    """
    df = _read_raw(path)
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)]
    if wc.empty:
        raise ValueError("No 2026 FIFA World Cup fixtures found in the dataset.")
    wc = wc.sort_values("date")

    # adjacency of group opponents (group stage = round-robin, so degree 3 per team)
    adj: dict[str, set[str]] = {}
    first_seen: dict[str, pd.Timestamp] = {}
    for r in wc.itertuples(index=False):
        a, b = r.home_team, r.away_team
        if len(adj.get(a, ())) >= 3 or len(adj.get(b, ())) >= 3:
            continue  # both teams already have their three group opponents -> knockout fixture
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
        for t in (a, b):
            if t not in first_seen or r.date < first_seen[t]:
                first_seen[t] = r.date

    # connected components
    seen: set[str] = set()
    components: list[list[str]] = []
    for team in adj:
        if team in seen:
            continue
        stack, comp = [team], []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(adj[cur] - seen)
        components.append(comp)

    components = [c for c in components if len(c) == 4]
    if len(components) != 12:
        raise ValueError(
            f"Expected 12 groups of 4, reconstructed {len(components)} "
            "(dataset fixtures may be incomplete)."
        )

    components.sort(key=lambda comp: min(first_seen[t] for t in comp))
    labels = [chr(ord("A") + i) for i in range(12)]
    return {labels[i]: sorted(components[i]) for i in range(12)}
