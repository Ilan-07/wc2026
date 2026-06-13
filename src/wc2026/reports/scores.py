"""Per-match predicted (and locked) scorelines for WC2026 — DISPLAY LAYER ONLY.

This module adds *nothing* to the forecast. The Monte-Carlo simulator (``simulate/tournament.py``)
is untouched; we simply surface the per-match numbers the fitted match model already implies, so the
dashboard and console can show "the score of each match", not just stage probabilities.

For every fixture we report three views of the same distribution:
  * expected goals (lambda, mu) — the honest continuous forecast,
  * the single most-likely *exact* scoreline + its probability (argmax of the Dixon-Coles grid),
  * win / draw / loss probabilities.

Live-aware:
  * group fixtures already played are *locked* to their real score (``played``),
  * remaining group fixtures are predicted from the model,
  * knockout fixtures appear only once the Round-of-32 bracket is known, and any KO tie already
    played is locked to its real score — so the section fills in as the tournament unfolds.

Orientation note: we mirror ``TournamentSimulator._sample`` exactly (host nation treated as the home
side, everything else neutral), so a displayed score is consistent with what the simulator samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..model.match_model import MatchModel
from ..simulate.tournament import HOST_TEAMS


@dataclass
class FixtureScore:
    """One fixture's scoreline view. ``home_score``/``away_score`` are expected goals when predicted
    and real integer goals when ``played``; the modal fields are the most-likely exact scoreline."""

    home: str
    away: str
    played: bool
    home_score: float
    away_score: float
    modal_home: int
    modal_away: int
    modal_prob: float
    p_home: float
    p_draw: float
    p_away: float
    date: str = ""
    group: str = ""
    stage: str = "group"
    round_name: str = ""

    def to_dict(self) -> dict:
        """JSON-serialisable payload for the dashboard renderer."""
        return {
            "home": self.home, "away": self.away, "played": self.played,
            "h": round(self.home_score, 2), "a": round(self.away_score, 2),
            "mh": int(self.modal_home), "ma": int(self.modal_away),
            "mp": round(self.modal_prob, 4),
            "pH": round(self.p_home, 4), "pD": round(self.p_draw, 4), "pA": round(self.p_away, 4),
            "date": self.date, "group": self.group, "stage": self.stage, "round": self.round_name,
        }


def _result_probs(gh: int, ga: int) -> tuple[float, float, float]:
    """Degenerate W/D/L for a finished match (1 on the realised outcome)."""
    return (float(gh > ga), float(gh == ga), float(gh < ga))


def score_fixture(
    model: MatchModel,
    home: str,
    away: str,
    hosts: set[str] = HOST_TEAMS,
    *,
    date: str = "",
    group: str = "",
    stage: str = "group",
    round_name: str = "",
) -> FixtureScore:
    """Predicted scoreline for ``home`` vs ``away`` (host nation gets home advantage, else neutral)."""
    a, b = home, away
    # Mirror TournamentSimulator._sample: if exactly the *away* side is a host, simulate it as home.
    if b in hosts and a not in hosts:
        grid = model.scoreline_grid(b, a, neutral=False)  # rows = away (b) goals, cols = home (a) goals
        eg_b, eg_a = model.rates(b, a, neutral=False)
        n = grid.shape[1]
        ij = int(np.argmax(grid))
        ma, mh = ij % n, ij // n  # col = home(a) goals, row = away(b) goals
        p_away = float(np.tril(grid, -1).sum())  # row (b) > col (a) => away wins
        p_home = float(np.triu(grid, 1).sum())
        p_draw = float(np.trace(grid))
        return FixtureScore(home, away, False, eg_a, eg_b, mh, ma, float(grid.max()),
                            p_home, p_draw, p_away, date, group, stage, round_name)
    neutral = not (a in hosts and b not in hosts)
    grid = model.scoreline_grid(a, b, neutral=neutral)  # rows = home (a) goals, cols = away (b) goals
    eg_a, eg_b = model.rates(a, b, neutral=neutral)
    n = grid.shape[1]
    ij = int(np.argmax(grid))
    mh, ma = ij // n, ij % n
    p_home = float(np.tril(grid, -1).sum())
    p_away = float(np.triu(grid, 1).sum())
    p_draw = float(np.trace(grid))
    return FixtureScore(home, away, False, eg_a, eg_b, mh, ma, float(grid.max()),
                        p_home, p_draw, p_away, date, group, stage, round_name)


def _played_fixture(home, away, gh, ga, *, date="", group="", stage="group", round_name="") -> FixtureScore:
    ph, pd, pa = _result_probs(gh, ga)
    return FixtureScore(home, away, True, float(gh), float(ga), int(gh), int(ga), 1.0,
                        ph, pd, pa, date, group, stage, round_name)


def build_group_scores(
    model: MatchModel,
    fixtures: list[tuple],
    played: dict[frozenset, tuple],
    groups: dict[str, list[str]],
    hosts: set[str] = HOST_TEAMS,
) -> list[FixtureScore]:
    """Score every group fixture. ``fixtures`` is ``(date, home, away, city)`` (chronological);
    ``played`` locks already-played games to their real score."""
    group_of = {t: g for g, ts in groups.items() for t in ts}
    out: list[FixtureScore] = []
    for date, home, away, _ in fixtures:
        g = group_of.get(home, "")
        ds = str(date)[:10] if date is not None else ""  # "2026-06-11 00:00:00" -> "2026-06-11"
        real = played.get(frozenset((home, away)))
        if real is not None:
            hteam, hs, as_ = real
            gh, ga = (hs, as_) if hteam == home else (as_, hs)
            out.append(_played_fixture(home, away, gh, ga, date=ds, group=g))
        else:
            out.append(score_fixture(model, home, away, hosts, date=ds, group=g))
    return out


def build_knockout_scores(
    model: MatchModel,
    bracket: list[str] | None,
    played_ko: dict[frozenset, tuple] | None = None,
    hosts: set[str] = HOST_TEAMS,
) -> list[FixtureScore]:
    """Round-of-32 scorelines from a known 32-team bracket (pairs are bracket[2i], bracket[2i+1]).
    Returns ``[]`` until the bracket is set. Already-played ties are locked to their real score."""
    if not bracket or len(bracket) != 32:
        return []
    played_ko = played_ko or {}
    out: list[FixtureScore] = []
    for i in range(0, 32, 2):
        home, away = bracket[i], bracket[i + 1]
        real = played_ko.get(frozenset((home, away)))
        if real is not None:
            hteam, hs, as_ = real
            gh, ga = (hs, as_) if hteam == home else (as_, hs)
            out.append(_played_fixture(home, away, gh, ga, stage="r32", round_name="Round of 32"))
        else:
            out.append(score_fixture(model, home, away, hosts, stage="r32", round_name="Round of 32"))
    return out


@dataclass
class ScoreSections:
    """Grouped scorelines, ready for the dashboard payload or the console formatter."""

    groups: list[FixtureScore] = field(default_factory=list)
    knockouts: list[FixtureScore] = field(default_factory=list)

    def payload(self) -> dict:
        """Nested dict the dashboard renders: per-group fixture lists + a knockout round list."""
        by_group: dict[str, list[dict]] = {}
        for fs in self.groups:
            by_group.setdefault(fs.group, []).append(fs.to_dict())
        groups = [{"group": g, "fixtures": by_group[g]} for g in sorted(by_group)]
        ko = [{"round": "Round of 32", "fixtures": [fs.to_dict() for fs in self.knockouts]}] \
            if self.knockouts else []
        return {"groups": groups, "knockouts": ko}


def build_score_sections(
    model: MatchModel,
    fixtures: list[tuple],
    played: dict[frozenset, tuple],
    groups: dict[str, list[str]],
    bracket: list[str] | None = None,
    played_ko: dict[frozenset, tuple] | None = None,
    hosts: set[str] = HOST_TEAMS,
) -> ScoreSections:
    return ScoreSections(
        groups=build_group_scores(model, fixtures, played, groups, hosts),
        knockouts=build_knockout_scores(model, bracket, played_ko, hosts),
    )


def _fmt_line(fs: FixtureScore) -> str:
    if fs.played:
        head = f"{fs.home} {int(fs.home_score)}-{int(fs.away_score)} {fs.away}  [played]"
        return f"  {head}"
    score = f"{fs.modal_home}-{fs.modal_away}"
    return (f"  {fs.home} {fs.home_score:.1f}-{fs.away_score:.1f} {fs.away}"
            f"  · likely {score} ({fs.modal_prob:.0%})"
            f" · W/D/L {fs.p_home:.0%}/{fs.p_draw:.0%}/{fs.p_away:.0%}")


def format_scores_console(sections: ScoreSections) -> str:
    """Plain-text scores block for the console (forecast.py / predict.py)."""
    lines = ["=== Predicted match scores ===",
             "(expected goals · most-likely exact score (its probability) · win/draw/loss)"]
    by_group: dict[str, list[FixtureScore]] = {}
    for fs in sections.groups:
        by_group.setdefault(fs.group, []).append(fs)
    for g in sorted(by_group):
        lines.append(f"\nGroup {g}")
        lines.extend(_fmt_line(fs) for fs in by_group[g])
    if sections.knockouts:
        lines.append("\nRound of 32")
        lines.extend(_fmt_line(fs) for fs in sections.knockouts)
    return "\n".join(lines)
