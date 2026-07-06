"""Per-match predicted (and locked) scorelines for WC2026 — DISPLAY LAYER ONLY.

This module adds *nothing* to the forecast. The Monte-Carlo simulator (``simulate/tournament.py``)
is untouched; we simply surface the per-match numbers the fitted match model already implies, so the
dashboard and console can show "the score of each match", not just stage probabilities.

For every fixture we report three views of the same distribution:
  * expected goals (lambda, mu) — the honest continuous forecast,
  * the single most-likely *exact* scoreline + its probability (argmax of the Dixon-Coles grid),
  * win / draw / loss probabilities (group) or *advancement* probabilities (knockout).

Consistency with the forecast (so a displayed score matches what the simulator samples):
  * host nation is treated as the home side, exactly like ``TournamentSimulator._sample``;
  * the same per-fixture **altitude (+ optional fatigue) log-rate shift** the simulator applies is
    applied here via ``extra_home`` (so the high-altitude Mexican venues read correctly);
  * knockout ties cannot end level — instead of a misleading "draw %", we report each side's
    **advancement probability**, computed analytically the same way ``sample_knockout`` resolves a
    level game (extra time drawn from the DC grid at ``ET_FRACTION`` of the rate, then a shootout).

Live-aware: played group/knockout games lock to their real score; knockout fixtures appear only once
the Round-of-32 bracket is known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.match_model import MatchModel
from ..simulate.tournament import HOST_TEAMS

# Mirror match_model.sample_knockout's knockout-resolution constants.
ET_FRACTION = 30.0 / 90.0
SHOOTOUT_SCALE = 0.4
# Mirror TournamentSimulator's default altitude coefficient (log-rate per 1000 m of disadvantage).
ALT_PER_1000M = 0.12


@dataclass
class FixtureScore:
    """One fixture's scoreline view. ``home_score``/``away_score`` are expected goals when predicted
    and real integer goals when ``played``; the modal fields are the most-likely exact scoreline.
    ``advance_home``/``advance_away`` are populated for knockout fixtures (where a level game is
    resolved by extra time + penalties), and are ``None`` for group fixtures."""

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
    advance_home: float | None = None
    advance_away: float | None = None
    # The frozen pre-kickoff model's PRE-match call (populated for played fixtures only), so the
    # dashboard can show what the model predicted before the game next to what actually happened.
    pre_home: float | None = None
    pre_draw: float | None = None
    pre_away: float | None = None
    pre_advance_home: float | None = None
    pre_advance_away: float | None = None

    def to_dict(self) -> dict:
        """JSON-serialisable payload for the dashboard renderer."""
        d = {
            "home": self.home, "away": self.away, "played": self.played,
            "h": round(self.home_score, 2), "a": round(self.away_score, 2),
            "mh": int(self.modal_home), "ma": int(self.modal_away),
            "mp": round(self.modal_prob, 4),
            "pH": round(self.p_home, 4), "pD": round(self.p_draw, 4), "pA": round(self.p_away, 4),
            "date": self.date, "group": self.group, "stage": self.stage, "round": self.round_name,
        }
        if self.advance_home is not None:
            d["adH"] = round(self.advance_home, 4)
            d["adA"] = round(self.advance_away if self.advance_away is not None else 0.0, 4)
        if self.pre_home is not None:  # pre-match receipt for a played fixture
            d["preH"] = round(self.pre_home, 4)
            d["preD"] = round(self.pre_draw or 0.0, 4)
            d["preA"] = round(self.pre_away or 0.0, 4)
        if self.pre_advance_home is not None:
            d["preAdH"] = round(self.pre_advance_home, 4)
            d["preAdA"] = round(self.pre_advance_away if self.pre_advance_away is not None else 0.0, 4)
        return d


def _grid_stats(grid: np.ndarray) -> tuple[float, float, float]:
    """(row-team win, draw, col-team win) probabilities for a scoreline grid."""
    return float(np.tril(grid, -1).sum()), float(np.trace(grid)), float(np.triu(grid, 1).sum())


def _shootout_home_prob(a: str, b: str, psi, shootout_model) -> float:
    """P(a wins a penalty shootout vs b), mirroring match_model.sample_knockout."""
    if shootout_model is not None:
        return float(shootout_model.win_prob(a, b, psi))
    if not psi:
        return 0.5
    d = psi.get(a, 0.0) - psi.get(b, 0.0)
    return float(1.0 / (1.0 + np.exp(-SHOOTOUT_SCALE * d)))


def _result_probs(gh: int, ga: int) -> tuple[float, float, float]:
    """Degenerate W/D/L for a finished match (1 on the realised outcome)."""
    return (float(gh > ga), float(gh == ga), float(gh < ga))


def score_fixture(
    model: MatchModel,
    home: str,
    away: str,
    hosts: set[str] = HOST_TEAMS,
    *,
    extra_home: float = 0.0,
    knockout: bool = False,
    psi=None,
    shootout_model=None,
    date: str = "",
    group: str = "",
    stage: str = "group",
    round_name: str = "",
) -> FixtureScore:
    """Predicted scoreline for ``home`` vs ``away``.

    ``extra_home`` is the home-oriented per-fixture log-rate shift (altitude + optional fatigue),
    applied exactly like the simulator (log lambda += delta on the home side). ``knockout=True``
    additionally returns each side's advancement probability instead of a standalone draw.
    """
    a, b = home, away
    swapped = b in hosts and a not in hosts  # away side is the host -> score it as home
    if swapped:
        ha, hb, neutral, sign = b, a, False, -1.0
    else:
        ha, hb, neutral, sign = a, b, not (a in hosts and b not in hosts), 1.0

    lam_h, mu_h = model.rates(ha, hb, neutral=neutral, extra_delta=sign * extra_home)
    grid = model.dc.score_matrix(lam_h, mu_h)  # rows = ha goals, cols = hb goals
    n = grid.shape[1]
    ij = int(np.argmax(grid))
    row, col = ij // n, ij % n
    p_ha, p_draw, p_hb = _grid_stats(grid)
    modal_p = float(grid.max())

    adv_ha = adv_hb = None
    if knockout:
        etg = model.dc.score_matrix(lam_h * ET_FRACTION, mu_h * ET_FRACTION)
        e_ha, e_draw, _ = _grid_stats(etg)
        s = _shootout_home_prob(ha, hb, psi, shootout_model)  # P(ha wins the shootout)
        adv_ha = p_ha + p_draw * (e_ha + e_draw * s)
        adv_hb = 1.0 - adv_ha

    # Map the (ha, hb) view back to the listed (home=a, away=b) orientation.
    if swapped:
        return FixtureScore(home, away, False, mu_h, lam_h, col, row, modal_p,
                            p_hb, p_draw, p_ha, date, group, stage, round_name, adv_hb, adv_ha)
    return FixtureScore(home, away, False, lam_h, mu_h, row, col, modal_p,
                        p_ha, p_draw, p_hb, date, group, stage, round_name, adv_ha, adv_hb)


def _played_fixture(home, away, gh, ga, *, knockout=False, date="", group="",
                    stage="group", round_name="") -> FixtureScore:
    ph, pd, pa = _result_probs(gh, ga)
    adv_h = adv_a = None
    if knockout:
        adv_h, adv_a = (1.0, 0.0) if gh > ga else (0.0, 1.0) if ga > gh else (None, None)
    return FixtureScore(home, away, True, float(gh), float(ga), int(gh), int(ga), 1.0,
                        ph, pd, pa, date, group, stage, round_name, adv_h, adv_a)


def _attach_pre(fs: FixtureScore, pre_model, home, away, hosts, *, knockout: bool,
                extra_home: float = 0.0, psi=None, shootout_model=None) -> FixtureScore:
    """Attach the frozen pre-kickoff model's PRE-match probabilities to a played fixture, so the
    dashboard can show the model's call beside the real result. No-op when ``pre_model`` is None."""
    if pre_model is None:
        return fs
    pf = score_fixture(pre_model, home, away, hosts, extra_home=extra_home, knockout=knockout,
                       psi=psi, shootout_model=shootout_model)
    fs.pre_home, fs.pre_draw, fs.pre_away = pf.p_home, pf.p_draw, pf.p_away
    if knockout:
        fs.pre_advance_home, fs.pre_advance_away = pf.advance_home, pf.advance_away
    return fs


def _group_extra(home: str, away: str, venue_alt, fatigue, alt_k: float = ALT_PER_1000M) -> float:
    """Home-oriented altitude (+ optional fatigue) log-rate delta for the LISTED home team,
    matching ``TournamentSimulator._extra``. Returns 0.0 when no covariate applies to this fixture."""
    if not venue_alt and not fatigue:
        return 0.0
    key = frozenset((home, away))
    d = 0.0
    alt = (venue_alt or {}).get(key)
    if alt:
        from ..data.venues import home_altitude
        pen = lambda t: alt_k * max(0, alt - home_altitude(t)) / 1000.0
        d += pen(away) - pen(home)
    pens = (fatigue or {}).get(key)
    if pens:
        d += pens.get(away, 0.0) - pens.get(home, 0.0)
    return d


def build_group_scores(
    model: MatchModel,
    fixtures: list[tuple],
    played: dict[frozenset, tuple],
    groups: dict[str, list[str]],
    hosts: set[str] = HOST_TEAMS,
    venue_alt: dict[frozenset, int] | None = None,
    fatigue: dict[frozenset, dict] | None = None,
    alt_k: float = ALT_PER_1000M,
    pre_model: MatchModel | None = None,
) -> list[FixtureScore]:
    """Score every group fixture. ``fixtures`` is ``(date, home, away, city)`` (chronological);
    ``played`` locks already-played games; ``venue_alt``/``fatigue`` apply the same per-fixture
    log-rate shifts the simulator uses (pass exactly what the live forecast passes). ``pre_model``
    (the frozen pre-kickoff model) attaches each played fixture's pre-match call for the receipt."""
    group_of = {t: g for g, ts in groups.items() for t in ts}
    out: list[FixtureScore] = []
    for date, home, away, _ in fixtures:
        g = group_of.get(home, "")
        ds = str(date)[:10] if date is not None else ""  # "2026-06-11 00:00:00" -> "2026-06-11"
        extra = _group_extra(home, away, venue_alt, fatigue, alt_k)
        real = played.get(frozenset((home, away)))
        if real is not None:
            hteam, hs, as_ = real
            gh, ga = (hs, as_) if hteam == home else (as_, hs)
            fs = _played_fixture(home, away, gh, ga, date=ds, group=g)
            out.append(_attach_pre(fs, pre_model, home, away, hosts, knockout=False, extra_home=extra))
        else:
            out.append(score_fixture(model, home, away, hosts, extra_home=extra, date=ds, group=g))
    return out


# Each knockout round keyed by how many teams are still alive -> (stage id, display name). Display
# names mirror the bracket column headers in the dashboard renderer so the Scores and Bracket
# sections read identically.
_KO_ROUNDS: dict[int, tuple[str, str]] = {
    32: ("r32", "Round of 32"),
    16: ("r16", "Round of 16"),
    8: ("qf", "Quarterfinals"),
    4: ("sf", "Semifinals"),
    2: ("final", "Final"),
}


def _tie_advancer_is_home(
    fs: FixtureScore, home: str, away: str,
    ko_winners: dict[frozenset, str], blended: dict[str, float] | None,
) -> bool:
    """Which side of a resolved-or-projected tie folds into the next round.

    Mirrors ``predict._knockout_advances_a`` so the projected pairings match the bracket tree
    exactly: a played, decided tie locks its real winner (``ko_winners`` includes shootout winners,
    and a decisive tie already carries a 1/0 ``advance_home`` from its real score); otherwise fall
    back to the blended (market-anchored) title probability, or — when no market blend is supplied
    (e.g. the fundamentals-only console) — the model's own advancement probability."""
    won = ko_winners.get(frozenset((home, away)))
    if won in (home, away):
        return won == home
    # A played tie that resolved (on the pitch or via a recorded shootout) knows its own advancer;
    # never let the market blend override a real result even if ``ko_winners`` wasn't supplied.
    if fs.played and fs.advance_home is not None:
        return fs.advance_home >= 0.5
    if blended is not None:
        return blended.get(home, 0.0) >= blended.get(away, 0.0)
    return fs.advance_home is None or fs.advance_home >= 0.5


def build_knockout_scores(
    model: MatchModel,
    bracket: list[str] | None,
    played_ko: dict[frozenset, tuple] | None = None,
    hosts: set[str] = HOST_TEAMS,
    psi=None,
    shootout_model=None,
    pre_model: MatchModel | None = None,
    ko_winners: dict[frozenset, str] | None = None,
    blended: dict[str, float] | None = None,
) -> list[FixtureScore]:
    """Scorelines for **every** knockout round (R32 -> R16 -> QF -> SF -> Final) from a known 32-team
    bracket (pairs are ``bracket[2i], bracket[2i+1]``). Returns ``[]`` until the bracket is set.

    Walks the same single-elimination fold the bracket tree uses: each round pairs the survivors of
    the previous one, so the projected later-round fixtures match the displayed bracket. Already-played
    ties are locked to their real score; upcoming ties carry an advancement probability (regulation +
    ET + shootout). ``pre_model`` adds each played tie's pre-match advance call for the receipt;
    ``ko_winners`` names the side that went through (so a tie level after 90' still records its
    advancer); ``blended`` is the market-anchored title probability used to project the winner of an
    unplayed tie into the next round, keeping these pairings identical to the bracket section."""
    if not bracket or len(bracket) != 32:
        return []
    played_ko = played_ko or {}
    ko_winners = ko_winners or {}
    out: list[FixtureScore] = []
    alive = list(bracket)
    while len(alive) > 1:
        stage, round_name = _KO_ROUNDS.get(len(alive), (f"r{len(alive)}", f"Last {len(alive)}"))
        nxt: list[str] = []
        for i in range(0, len(alive), 2):
            home, away = alive[i], alive[i + 1]
            real = played_ko.get(frozenset((home, away)))
            if real is not None:
                hteam, hs, as_ = real
                gh, ga = (hs, as_) if hteam == home else (as_, hs)
                fs = _played_fixture(home, away, gh, ga, knockout=True,
                                     stage=stage, round_name=round_name)
                if fs.advance_home is None:  # level after 90' — take the advancer from the shootout
                    w = ko_winners.get(frozenset((home, away)))
                    if w:
                        fs.advance_home, fs.advance_away = (1.0, 0.0) if w == home else (0.0, 1.0)
                fs = _attach_pre(fs, pre_model, home, away, hosts, knockout=True,
                                 psi=psi, shootout_model=shootout_model)
            else:
                fs = score_fixture(model, home, away, hosts, knockout=True, psi=psi,
                                   shootout_model=shootout_model, stage=stage, round_name=round_name)
            out.append(fs)
            nxt.append(home if _tie_advancer_is_home(fs, home, away, ko_winners, blended) else away)
        alive = nxt
    return out


@dataclass
class ScoreSections:
    """Grouped scorelines, ready for the dashboard payload or the console formatter."""

    groups: list[FixtureScore]
    knockouts: list[FixtureScore]

    def payload(self) -> dict:
        """Nested dict the dashboard renders: per-group fixture lists + a knockout round list."""
        by_group: dict[str, list[dict]] = {}
        for fs in self.groups:
            by_group.setdefault(fs.group, []).append(fs.to_dict())
        groups = [{"group": g, "fixtures": by_group[g]} for g in sorted(by_group)]
        # Group knockout fixtures by round, preserving the build order (R32 -> R16 -> QF -> SF ->
        # Final) so the dashboard renders one score card per round in bracket order.
        ko: list[dict] = []
        for fs in self.knockouts:
            if not ko or ko[-1]["round"] != fs.round_name:
                ko.append({"round": fs.round_name, "fixtures": []})
            ko[-1]["fixtures"].append(fs.to_dict())
        return {"groups": groups, "knockouts": ko}


def build_score_sections(
    model: MatchModel,
    fixtures: list[tuple],
    played: dict[frozenset, tuple],
    groups: dict[str, list[str]],
    bracket: list[str] | None = None,
    played_ko: dict[frozenset, tuple] | None = None,
    hosts: set[str] = HOST_TEAMS,
    venue_alt: dict[frozenset, int] | None = None,
    fatigue: dict[frozenset, dict] | None = None,
    psi=None,
    shootout_model=None,
    alt_k: float = ALT_PER_1000M,
    pre_model: MatchModel | None = None,
    ko_winners: dict[frozenset, str] | None = None,
    blended: dict[str, float] | None = None,
) -> ScoreSections:
    return ScoreSections(
        groups=build_group_scores(model, fixtures, played, groups, hosts, venue_alt, fatigue, alt_k,
                                  pre_model=pre_model),
        knockouts=build_knockout_scores(model, bracket, played_ko, hosts, psi, shootout_model,
                                        pre_model=pre_model, ko_winners=ko_winners, blended=blended),
    )


def _fmt_line(fs: FixtureScore) -> str:
    if fs.played:
        tag = "advanced" if fs.advance_home is not None else "played"
        return f"  {fs.home} {int(fs.home_score)}-{int(fs.away_score)} {fs.away}  [{tag}]"
    base = (f"  {fs.home} {fs.home_score:.1f}-{fs.away_score:.1f} {fs.away}"
            f"  · likely {fs.modal_home}-{fs.modal_away} ({fs.modal_prob:.0%})")
    if fs.advance_home is not None:  # knockout: advancement, not a standalone draw
        return f"{base} · advance {fs.advance_home:.0%}/{fs.advance_away:.0%}"
    return f"{base} · W/D/L {fs.p_home:.0%}/{fs.p_draw:.0%}/{fs.p_away:.0%}"


def format_scores_console(sections: ScoreSections) -> str:
    """Plain-text scores block for the console (forecast.py / predict.py)."""
    lines = ["=== Predicted match scores ===",
             "(expected goals · most-likely exact score (its probability) · win/draw/loss or advance%)"]
    by_group: dict[str, list[FixtureScore]] = {}
    for fs in sections.groups:
        by_group.setdefault(fs.group, []).append(fs)
    for g in sorted(by_group):
        lines.append(f"\nGroup {g}")
        lines.extend(_fmt_line(fs) for fs in by_group[g])
    if sections.knockouts:
        lines.append("\nKnockouts  (scores are regulation; advance% includes extra time + penalties)")
        current = None
        for fs in sections.knockouts:
            if fs.round_name != current:
                current = fs.round_name
                lines.append(f"\n{fs.round_name}")
            lines.append(_fmt_line(fs))
    return "\n".join(lines)
